#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising V5 Training Script

V5 uses:
1. Powerful Correctors (3.2M params)
2. Quality-Aligned Loss:
   - Clean-Referenced Predicates
   - Hard Quality Constraint (no degradation)

This ensures predicates and quality improve TOGETHER.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple

sys.stdout.reconfigure(line_buffering=True)

from neuro_symbolic_v2 import (
    BackboneWrapper, AdaptiveLambdaPredictor, SymbolicPredicates,
    OCTDataset, _NUM_THREADS, clear_memory
)
from powerful_correctors import PowerfulAdaptiveCorrectorWithLambda
from quality_aligned_loss import QualityAlignedLoss, CleanReferencedPredicates
from torch.utils.data import DataLoader


# =============================================================================
# V5 MODEL
# =============================================================================

class NeuroSymbolicDenoiserV5(nn.Module):
    """
    V5: Powerful Correctors + Quality-Aligned Training

    Architecture:
    1. Backbone: NAFNet (frozen, 7.5M params)
    2. Lambda Predictor: Per-pixel correction strength (35K params)
    3. Powerful Correctors: 5 specialized correctors (3.2M params)

    Key difference from V4:
    - Training uses clean-referenced predicates
    - Hard quality constraint prevents degradation
    """

    def __init__(self, backbone_type: str = 'nafnet', width: int = 64,
                 corrector_hidden_dim: int = 128):
        super().__init__()

        self.backbone = BackboneWrapper(backbone_type=backbone_type, width=width)

        # Standard predicates for failure map computation
        self.predicates = SymbolicPredicates()

        # Lambda predictor
        self.lambda_predictor = AdaptiveLambdaPredictor()

        # Powerful correctors
        self.corrector = PowerfulAdaptiveCorrectorWithLambda(
            enc1_channels=width,
            enc2_channels=width * 2,
            hidden_dim=corrector_hidden_dim
        )

    def load_pretrained_backbone(self, path: str):
        return self.backbone.load_pretrained(path)

    def cap_lambda_maps(self, lambda_maps: Dict, max_lambda: float = 0.2) -> Dict:
        """Cap lambda values to prevent over-correction.

        Args:
            lambda_maps: Dictionary of lambda maps for each corrector
            max_lambda: Maximum allowed lambda value

        Returns:
            Dictionary of capped lambda maps
        """
        capped = {}
        for name, lmap in lambda_maps.items():
            capped[name] = lmap.clamp(0, max_lambda)
        return capped

    def forward(self, noisy: torch.Tensor, max_lambda: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Returns:
            corrected: Final output
            backbone_out: Backbone-only output (for quality comparison)
            info: Lambda maps and other info
        """
        # 1. Backbone denoising
        backbone_out, backbone_features = self.backbone(noisy, return_features=True)

        # 2. Compute failure maps (using noisy reference for WHERE to correct)
        pred_with_maps = self.predicates(noisy, backbone_out, return_failure_maps=True)
        failure_maps = {
            name: pred_with_maps[name]['failure_map']
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }

        # 3. Predict lambda maps
        lambda_maps = self.lambda_predictor(backbone_out, failure_maps)

        # 3.5. Cap lambda maps to prevent over-correction
        lambda_maps_uncapped = lambda_maps
        lambda_maps = self.cap_lambda_maps(lambda_maps, max_lambda=max_lambda)

        # 4. Apply corrections
        corrected, corrector_info = self.corrector(
            backbone_out, noisy, lambda_maps, backbone_features,
            failure_maps=failure_maps,
            return_individual=self.training
        )

        # Compute capping statistics
        total_values = 0
        capped_values = 0
        for name in lambda_maps.keys():
            uncapped = lambda_maps_uncapped[name]
            capped = lambda_maps[name]
            total_values += uncapped.numel()
            capped_values += (uncapped > max_lambda).sum().item()

        capped_pct = (capped_values / total_values * 100) if total_values > 0 else 0.0

        info = {
            'lambda_maps': lambda_maps,
            'lambda_maps_uncapped': lambda_maps_uncapped,
            'correction_magnitude': corrector_info.get('correction_magnitude', 0),
            'raw_corrections': corrector_info.get('raw_corrections', {}),
            'max_lambda': max_lambda,
            'capped_pct': capped_pct,
        }

        return corrected, backbone_out, info


# =============================================================================
# TRAINER
# =============================================================================

class TrainerV5:
    """Trainer for V5 with quality-aligned loss."""

    def __init__(self, model: NeuroSymbolicDenoiserV5,
                 train_dataset: OCTDataset,
                 val_dataset: OCTDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4,
                 finetune_backbone: bool = False,
                 num_workers: int = 2):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.finetune_backbone = finetune_backbone

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Freeze backbone
        if not finetune_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
                {'params': model.corrector.parameters(), 'lr': lr * 2},
            ]
            print(f"Frozen backbone: training corrector (lr={lr*2:.0e}) + lambda (lr={lr*5:.0e})")
        else:
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
                {'params': model.corrector.parameters(), 'lr': lr * 2},
                {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            ]
            print(f"Joint training: backbone (lr={lr*0.1:.0e}) + corrector (lr={lr*2:.0e}) + lambda (lr={lr*5:.0e})")

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        # Quality-aligned loss
        self.loss_fn = QualityAlignedLoss(
            lambda_pred=1.0,
            lambda_quality=100.0,  # Heavy penalty for degradation
            lambda_recon=1.0
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=25, eta_min=lr * 0.01
        )

        self.best_score = 0
        self.best_psnr_delta = -float('inf')

        # Adaptive lambda capping parameters
        self.max_lambda = 0.15  # Initial max lambda
        self.max_lambda_min = 0.05  # Minimum allowed max_lambda
        self.max_lambda_max = 0.30  # Maximum allowed max_lambda
        self.psnr_delta_history = []  # Rolling history of PSNR deltas
        self.psnr_delta_window = 10  # Number of batches for rolling average

        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {total:,} total, {trainable:,} trainable")
        print(f"Training: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    def train_epoch(self) -> Dict:
        """Train one epoch."""
        self.model.train()

        metrics = {
            'loss': [], 'pred_loss': [], 'quality_loss': [], 'recon_loss': [],
            'psnr_backbone': [], 'psnr_corrected': [], 'quality_improved': [],
            'lambda_edge': [], 'lambda_contrast': [], 'lambda_sharpness': [],
            'correction_mag': [], 'max_lambda': [], 'capped_pct': [],
            'P1': [], 'P2': [], 'P3': [], 'P4': [], 'P5': [], 'P6': [],
        }

        for batch in self.train_loader:
            noisy = batch['noisy']
            clean = batch['clean']

            self.optimizer.zero_grad()

            # Forward with adaptive max_lambda
            corrected, backbone_out, info = self.model(noisy, max_lambda=self.max_lambda)

            # Quality-aligned loss
            loss_dict = self.loss_fn(
                corrected, backbone_out, clean, noisy,
                lambda_maps=info['lambda_maps']
            )
            loss = loss_dict['total']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Collect metrics
            metrics['loss'].append(loss.item())
            metrics['pred_loss'].append(loss_dict['pred_loss'].item())
            metrics['quality_loss'].append(loss_dict['quality_loss'].item())
            metrics['recon_loss'].append(loss_dict['recon_loss'].item())
            metrics['psnr_backbone'].append(loss_dict['psnr_backbone'])
            metrics['psnr_corrected'].append(loss_dict['psnr_corrected'])
            metrics['quality_improved'].append(float(loss_dict['quality_improved']))
            metrics['lambda_edge'].append(loss_dict['lambda_edge_mean'])
            metrics['lambda_contrast'].append(loss_dict['lambda_contrast_mean'])
            metrics['lambda_sharpness'].append(loss_dict['lambda_sharpness_mean'])
            metrics['correction_mag'].append(info.get('correction_magnitude', 0))
            metrics['max_lambda'].append(self.max_lambda)
            metrics['capped_pct'].append(info.get('capped_pct', 0))

            for i in range(1, 7):
                metrics[f'P{i}'].append(loss_dict[f'P{i}'])

            # Adaptive lambda scaling based on PSNR delta
            psnr_delta = loss_dict['psnr_corrected'] - loss_dict['psnr_backbone']
            self.psnr_delta_history.append(psnr_delta)

            # Keep only last N batches
            if len(self.psnr_delta_history) > self.psnr_delta_window:
                self.psnr_delta_history.pop(0)

            # Adjust max_lambda based on rolling average
            if len(self.psnr_delta_history) >= self.psnr_delta_window:
                avg_psnr_delta = np.mean(self.psnr_delta_history)

                if avg_psnr_delta < -0.5:
                    # Corrections are degrading quality - reduce max_lambda by 10%
                    self.max_lambda = max(self.max_lambda_min, self.max_lambda * 0.9)
                elif avg_psnr_delta > 0:
                    # Corrections are improving quality - gradually increase max_lambda
                    self.max_lambda = min(self.max_lambda_max, self.max_lambda * 1.02)

        return {k: np.mean(v) for k, v in metrics.items()}

    def validate(self) -> Dict:
        """Validate the model."""
        self.model.eval()

        metrics = {
            'psnr_backbone': [], 'psnr_corrected': [],
            'ssim_backbone': [], 'ssim_corrected': [],
            'quality_improved': [],
            'lambda_edge': [], 'lambda_contrast': [], 'lambda_sharpness': [],
            'correction_mag': [], 'max_lambda': [], 'capped_pct': [],
            'P1': [], 'P2': [], 'P3': [], 'P4': [], 'P5': [], 'P6': [],
        }

        with torch.no_grad():
            for batch in self.val_loader:
                noisy = batch['noisy']
                clean = batch['clean']

                # Forward with adaptive max_lambda
                corrected, backbone_out, info = self.model(noisy, max_lambda=self.max_lambda)

                # Compute loss for metrics
                loss_dict = self.loss_fn(
                    corrected, backbone_out, clean, noisy,
                    lambda_maps=info['lambda_maps']
                )

                # Collect metrics
                metrics['psnr_backbone'].append(loss_dict['psnr_backbone'])
                metrics['psnr_corrected'].append(loss_dict['psnr_corrected'])
                metrics['ssim_backbone'].append(loss_dict['ssim_backbone'])
                metrics['ssim_corrected'].append(loss_dict['ssim_corrected'])
                metrics['quality_improved'].append(float(loss_dict['quality_improved']))
                metrics['lambda_edge'].append(loss_dict['lambda_edge_mean'])
                metrics['lambda_contrast'].append(loss_dict['lambda_contrast_mean'])
                metrics['lambda_sharpness'].append(loss_dict['lambda_sharpness_mean'])
                metrics['correction_mag'].append(info.get('correction_magnitude', 0))
                metrics['max_lambda'].append(self.max_lambda)
                metrics['capped_pct'].append(info.get('capped_pct', 0))

                for i in range(1, 7):
                    metrics[f'P{i}'].append(loss_dict[f'P{i}'])

        return {k: np.mean(v) for k, v in metrics.items()}

    def train(self, epochs: int = 25, val_every: int = 5):
        """Training loop."""
        print('\n' + '#'*70)
        print('# NEURO-SYMBOLIC V5 (QUALITY-ALIGNED TRAINING)')
        print('#'*70)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train = self.train_epoch()
            train_time = time.time() - t0

            # Print training
            psnr_delta = train['psnr_corrected'] - train['psnr_backbone']
            quality_rate = train['quality_improved'] * 100

            print(f'\n[EPOCH {epoch}]')
            print(f'  Loss: {train["loss"]:.4f} (pred={train["pred_loss"]:.3f}, '
                  f'quality={train["quality_loss"]:.4f}, recon={train["recon_loss"]:.4f})')
            print(f'  PSNR: backbone={train["psnr_backbone"]:.2f}, '
                  f'corrected={train["psnr_corrected"]:.2f} (Δ={psnr_delta:+.3f})')
            print(f'  Quality improved: {quality_rate:.1f}% of pixels')
            print(f'  Lambda: edge={train["lambda_edge"]:.3f}, '
                  f'contrast={train["lambda_contrast"]:.3f}, '
                  f'sharp={train["lambda_sharpness"]:.3f}')
            print(f'  Lambda capping: max_lambda={self.max_lambda:.3f}, '
                  f'capped={train["capped_pct"]:.1f}%')
            print(f'  Predicates: P1={train["P1"]:.3f}, P2={train["P2"]:.3f}, '
                  f'P3={train["P3"]:.3f}')
            print(f'              P4={train["P4"]:.3f}, P5={train["P5"]:.3f}, '
                  f'P6={train["P6"]:.3f}')
            print(f'  Time: {train_time:.1f}s')

            # Validation
            if epoch % val_every == 0 or epoch == 1:
                val = self.validate()
                psnr_delta_val = val['psnr_corrected'] - val['psnr_backbone']
                ssim_delta_val = val['ssim_corrected'] - val['ssim_backbone']
                quality_rate_val = val['quality_improved'] * 100

                print(f'\n  [VALIDATION]')
                print(f'  PSNR: backbone={val["psnr_backbone"]:.2f}, '
                      f'corrected={val["psnr_corrected"]:.2f} (Δ={psnr_delta_val:+.3f})')
                print(f'  SSIM: backbone={val["ssim_backbone"]:.4f}, '
                      f'corrected={val["ssim_corrected"]:.4f} (Δ={ssim_delta_val:+.4f})')
                print(f'  Quality improved: {quality_rate_val:.1f}%')
                print(f'  Lambda: edge={val["lambda_edge"]:.3f}, '
                      f'contrast={val["lambda_contrast"]:.3f}, '
                      f'sharp={val["lambda_sharpness"]:.3f}')
                print(f'  Lambda capping: max_lambda={self.max_lambda:.3f}, '
                      f'capped={val["capped_pct"]:.1f}%')
                print(f'  Predicates: P1={val["P1"]:.3f}, P2={val["P2"]:.3f}, '
                      f'P3={val["P3"]:.3f}')
                print(f'              P4={val["P4"]:.3f}, P5={val["P5"]:.3f}, '
                      f'P6={val["P6"]:.3f}')

                # Score: predicate average + quality bonus
                pred_score = sum(val[f'P{i}'] for i in range(1, 7)) / 6
                quality_bonus = max(0, psnr_delta_val) * 0.1  # Bonus for quality improvement
                score = pred_score + quality_bonus

                print(f'  Score: {score:.4f} (pred={pred_score:.4f}, quality_bonus={quality_bonus:.4f})')

                # Save best model (prioritize positive PSNR delta)
                if psnr_delta_val > self.best_psnr_delta or (
                    psnr_delta_val >= 0 and score > self.best_score):
                    self.best_score = score
                    self.best_psnr_delta = psnr_delta_val
                    print(f'  *** Best model! ***')

                    Path('outputs/nsnd_v5').mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'val_metrics': val,
                        'psnr_delta': psnr_delta_val,
                        'ssim_delta': ssim_delta_val,
                    }, 'outputs/nsnd_v5/best_model_v5.pth')

            self.scheduler.step()

        print('\n' + '#'*70)
        print('# TRAINING COMPLETE')
        print('#'*70)
        print(f'\nBest PSNR delta: {self.best_psnr_delta:+.3f} dB')
        print(f'Best score: {self.best_score:.4f}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('='*70)
    print('NEURO-SYMBOLIC OCT DENOISING V5')
    print('Quality-Aligned Training with Powerful Correctors')
    print(f'Using {_NUM_THREADS} CPU threads')
    print('='*70)

    # Configuration
    FINETUNE_BACKBONE = False  # Start with frozen backbone
    BATCH_SIZE = 4
    EPOCHS = 25
    TRAIN_SAMPLES = 100
    VAL_SAMPLES = 10
    NUM_WORKERS = 2
    VAL_EVERY = 5
    PATCH_SIZE = 96

    # Create model
    print('\nInitializing V5 model...')
    model = NeuroSymbolicDenoiserV5(
        backbone_type='nafnet',
        width=64,
        corrector_hidden_dim=128
    )

    # Load pretrained backbone
    backbone_path = 'outputs/nafnet_pku37/nafnet_best.pth'
    if Path(backbone_path).exists():
        print('Loading pretrained backbone...')
        model.load_pretrained_backbone(backbone_path)

    # Load data
    print('\nLoading data...')
    train_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_train.jsonl',
        max_samples=TRAIN_SAMPLES,
        patch_size=PATCH_SIZE,
        is_train=True
    )
    val_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_val.jsonl',
        max_samples=VAL_SAMPLES,
        patch_size=0,
        is_train=False
    )
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    # Create trainer
    trainer = TrainerV5(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        finetune_backbone=FINETUNE_BACKBONE,
        num_workers=NUM_WORKERS,
    )

    # Train
    trainer.train(epochs=EPOCHS, val_every=VAL_EVERY)


if __name__ == '__main__':
    main()
