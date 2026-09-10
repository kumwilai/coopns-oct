#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising V4 Training Script

V4 uses Powerful Correctors with:
- 3.2M params (vs 400K in V3) - 8x more capacity
- Self-attention for global context
- Multi-scale feature processing
- Still interpretable (each corrector has clear semantic purpose)
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
    BackboneWrapper, AdaptiveLambdaPredictor, NeuroSymbolicLossV3,
    OCTDataset, SymbolicPredicates, _NUM_THREADS, clear_memory
)
from powerful_correctors import PowerfulAdaptiveCorrectorWithLambda
from torch.utils.data import DataLoader


# =============================================================================
# V4 MODEL
# =============================================================================

class NeuroSymbolicDenoiserV4(nn.Module):
    """
    Neuro-Symbolic Denoiser V4 with Powerful Correctors.

    Architecture:
    1. Backbone: NAFNet (7.5M params, pretrained)
    2. Symbolic Predicates: Compute failure maps
    3. Lambda Predictor: Per-pixel correction strength (35K params)
    4. Powerful Correctors: 5 specialized correctors (3.2M params)

    Total: ~10.7M params
    """

    def __init__(self, backbone_type: str = 'nafnet', width: int = 64,
                 corrector_hidden_dim: int = 128):
        super().__init__()

        # Backbone (denoising network)
        self.backbone = BackboneWrapper(backbone_type=backbone_type, width=width)

        # Symbolic predicates (WHAT and WHERE failed)
        self.predicates = SymbolicPredicates()

        # Lambda predictor (HOW MUCH correction needed)
        self.lambda_predictor = AdaptiveLambdaPredictor()

        # Powerful correctors (8x more capacity than V3)
        self.corrector = PowerfulAdaptiveCorrectorWithLambda(
            enc1_channels=width,
            enc2_channels=width * 2,
            hidden_dim=corrector_hidden_dim
        )

    def load_pretrained_backbone(self, path: str):
        """Load pretrained backbone weights."""
        return self.backbone.load_pretrained(path)

    def forward(self, noisy: torch.Tensor,
                return_details: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive lambda prediction.

        Args:
            noisy: Noisy input [B, 1, H, W]
            return_details: Return detailed correction info

        Returns:
            corrected: Corrected output [B, 1, H, W]
            info: Dictionary with lambda maps, correction stats
        """
        # 1. Backbone denoising with feature extraction
        initial, backbone_features = self.backbone(noisy, return_features=True)

        # 2. Compute failure maps (WHERE predicates fail)
        pred_with_maps = self.predicates(noisy, initial, return_failure_maps=True)
        failure_maps = {
            name: pred_with_maps[name]['failure_map']
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }

        # 3. Predict per-pixel lambda maps based on failure maps
        lambda_maps = self.lambda_predictor(initial, failure_maps)

        # 4. Apply powerful corrections weighted by lambda maps
        corrected, corrector_info = self.corrector(
            initial, noisy, lambda_maps, backbone_features,
            failure_maps=failure_maps,
            return_individual=return_details
        )

        # 5. Compute final predicate scores
        final_pred = self.predicates(noisy, corrected, return_failure_maps=False)

        # Build output dict (compatible with NeuroSymbolicLossV3)
        output = {
            'denoised': corrected,
            'initial': initial,
            'lambda_maps': lambda_maps,
            'predicate_results': final_pred,
            'correction_magnitude': corrector_info.get('correction_magnitude', 0),
            'raw_corrections': corrector_info.get('raw_corrections', {}),
        }

        if return_details:
            output['individual_corrections'] = corrector_info.get('individual_corrections', {})
            output['backbone_features'] = backbone_features

        return corrected, output


# =============================================================================
# TRAINER
# =============================================================================

class TrainerV4:
    """Trainer for V4 with powerful correctors."""

    def __init__(self, model: NeuroSymbolicDenoiserV4,
                 train_dataset: OCTDataset,
                 val_dataset: OCTDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4,
                 finetune_backbone: bool = False,
                 num_workers: int = 2):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.finetune_backbone = finetune_backbone

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, drop_last=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Setup optimizer with separate learning rates
        if finetune_backbone:
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
                {'params': model.corrector.parameters(), 'lr': lr * 2},  # Lower LR for large corrector
                {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            ]
            print(f"Joint training: backbone (lr={lr*0.1:.0e}) + corrector (lr={lr*2:.0e}) + lambda (lr={lr*5:.0e})")
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
                {'params': model.corrector.parameters(), 'lr': lr * 2},
            ]
            print(f"Frozen backbone: training corrector (lr={lr*2:.0e}) + lambda (lr={lr*5:.0e})")

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        self.loss_fn = NeuroSymbolicLossV3()

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=lr * 0.01
        )

        self.best_score = 0

        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        corrector_params = sum(p.numel() for p in model.corrector.parameters())
        lambda_params = sum(p.numel() for p in model.lambda_predictor.parameters())
        backbone_params = sum(p.numel() for p in model.backbone.parameters())

        print(f"\nModel Parameters:")
        print(f"  Backbone: {backbone_params:,} {'(frozen)' if not finetune_backbone else '(trainable)'}")
        print(f"  Corrector: {corrector_params:,} (POWERFUL)")
        print(f"  Lambda: {lambda_params:,}")
        print(f"  Total: {total:,}, Trainable: {trainable:,}")
        print(f"\nTraining: {len(train_dataset)} samples, batch_size={batch_size}")
        print(f"Validation: {len(val_dataset)} samples")

    def compute_psnr(self, pred, target):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10(1.0 / mse).item()

    def train_epoch(self) -> Dict:
        """Train one epoch."""
        self.model.train()

        total_loss = 0
        total_psnr = 0
        total_lambda_edge = 0
        total_lambda_contrast = 0
        total_lambda_sharpness = 0
        total_correction_mag = 0
        predicates = {f'P{i}': 0 for i in range(1, 7)}
        n_batches = 0

        for batch in self.train_loader:
            noisy = batch['noisy']
            clean = batch['clean']

            self.optimizer.zero_grad()

            # Forward
            corrected, output_dict = self.model(noisy)

            # Loss (output_dict contains 'denoised', 'predicate_results', etc.)
            loss_dict = self.loss_fn(output_dict, clean, noisy)
            loss = loss_dict['total']

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Stats
            total_loss += loss.item()
            total_psnr += self.compute_psnr(corrected.detach(), clean)
            total_lambda_edge += loss_dict.get('lambda_edge_mean', torch.tensor(0)).item()
            total_lambda_contrast += loss_dict.get('lambda_contrast_mean', torch.tensor(0)).item()
            total_lambda_sharpness += loss_dict.get('lambda_sharpness_mean', torch.tensor(0)).item()
            total_correction_mag += output_dict.get('correction_magnitude', 0)

            for k in predicates:
                pred_val = loss_dict.get(k, 0)
                predicates[k] += pred_val.item() if isinstance(pred_val, torch.Tensor) else pred_val

            n_batches += 1

        n = max(n_batches, 1)
        return {
            'loss': total_loss / n,
            'psnr': total_psnr / n,
            'lambda_edge': total_lambda_edge / n,
            'lambda_contrast': total_lambda_contrast / n,
            'lambda_sharpness': total_lambda_sharpness / n,
            'correction_mag': total_correction_mag / n,
            **{f'P{i}': predicates[f'P{i}'] / n for i in range(1, 7)},
        }

    def validate(self) -> Dict:
        """Validate the model."""
        self.model.eval()

        psnr_backbone = []
        psnr_corrected = []
        ssim_backbone = []
        ssim_corrected = []
        lambda_edge = []
        lambda_contrast = []
        lambda_sharpness = []
        correction_mag = []
        predicates = {f'P{i}': [] for i in range(1, 7)}

        with torch.no_grad():
            for batch in self.val_loader:
                noisy = batch['noisy']
                clean = batch['clean']

                # Backbone only
                backbone_out = self.model.backbone(noisy, return_features=False)

                # Full model
                corrected, output_dict = self.model(noisy)
                loss_dict = self.loss_fn(output_dict, clean, noisy)

                # PSNR
                psnr_backbone.append(self.compute_psnr(backbone_out, clean))
                psnr_corrected.append(self.compute_psnr(corrected, clean))

                # SSIM (simplified)
                ssim_backbone.append(self._ssim(backbone_out, clean))
                ssim_corrected.append(self._ssim(corrected, clean))

                # Lambda
                lambda_edge.append(loss_dict.get('lambda_edge_mean', torch.tensor(0)).item())
                lambda_contrast.append(loss_dict.get('lambda_contrast_mean', torch.tensor(0)).item())
                lambda_sharpness.append(loss_dict.get('lambda_sharpness_mean', torch.tensor(0)).item())
                correction_mag.append(output_dict.get('correction_magnitude', 0))

                # Predicates
                for k in predicates:
                    pred_val = loss_dict.get(k, 0)
                    predicates[k].append(pred_val.item() if isinstance(pred_val, torch.Tensor) else pred_val)

        return {
            'psnr_backbone': np.mean(psnr_backbone),
            'psnr_corrected': np.mean(psnr_corrected),
            'ssim_backbone': np.mean(ssim_backbone),
            'ssim_corrected': np.mean(ssim_corrected),
            'lambda_edge': np.mean(lambda_edge),
            'lambda_contrast': np.mean(lambda_contrast),
            'lambda_sharpness': np.mean(lambda_sharpness),
            'correction_mag': np.mean(correction_mag),
            **{f'P{i}': np.mean(predicates[f'P{i}']) for i in range(1, 7)},
        }

    def _ssim(self, img1, img2):
        """Simple SSIM computation."""
        C1, C2 = 0.01**2, 0.03**2
        mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1**2, 11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2**2, 11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1*img2, 11, stride=1, padding=5) - mu1_mu2
        ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim.mean().item()

    def train(self, epochs: int = 20, val_every: int = 5):
        """Training loop."""
        print('\n' + '#'*70)
        print('# NEURO-SYMBOLIC OCT DENOISING V4 (POWERFUL CORRECTORS)')
        print('#'*70)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - t0

            # Print training stats
            print(f'\n[EPOCH {epoch}]')
            print(f'  Loss: {train_metrics["loss"]:.4f}, PSNR: {train_metrics["psnr"]:.2f} dB')
            print(f'  Lambda: edge={train_metrics["lambda_edge"]:.3f}, '
                  f'contrast={train_metrics["lambda_contrast"]:.3f}, '
                  f'sharp={train_metrics["lambda_sharpness"]:.3f}')
            print(f'  Correction magnitude: {train_metrics["correction_mag"]:.4f}')
            print(f'  Predicates: P1={train_metrics["P1"]:.3f}, P2={train_metrics["P2"]:.3f}, '
                  f'P3={train_metrics["P3"]:.3f}')
            print(f'              P4={train_metrics["P4"]:.3f}, P5={train_metrics["P5"]:.3f}, '
                  f'P6={train_metrics["P6"]:.3f}')
            print(f'  Time: {train_time:.1f}s')

            # Validation
            if epoch % val_every == 0 or epoch == 1:
                val_metrics = self.validate()
                delta_psnr = val_metrics['psnr_corrected'] - val_metrics['psnr_backbone']
                delta_ssim = val_metrics['ssim_corrected'] - val_metrics['ssim_backbone']

                print(f'\n  [VALIDATION]')
                print(f'  Backbone PSNR: {val_metrics["psnr_backbone"]:.2f} dB, '
                      f'SSIM: {val_metrics["ssim_backbone"]:.4f}')
                print(f'  Corrected PSNR: {val_metrics["psnr_corrected"]:.2f} dB '
                      f'(Δ={delta_psnr:+.3f}), SSIM: {val_metrics["ssim_corrected"]:.4f} '
                      f'(Δ={delta_ssim:+.4f})')
                print(f'  Lambda: edge={val_metrics["lambda_edge"]:.3f}, '
                      f'contrast={val_metrics["lambda_contrast"]:.3f}, '
                      f'sharp={val_metrics["lambda_sharpness"]:.3f}')
                print(f'  Correction magnitude: {val_metrics["correction_mag"]:.4f}')
                print(f'  Predicates: P1={val_metrics["P1"]:.3f}, P2={val_metrics["P2"]:.3f}, '
                      f'P3={val_metrics["P3"]:.3f}')
                print(f'              P4={val_metrics["P4"]:.3f}, P5={val_metrics["P5"]:.3f}, '
                      f'P6={val_metrics["P6"]:.3f}')

                # Score = average predicate score
                score = sum(val_metrics[f'P{i}'] for i in range(1, 7)) / 6
                print(f'  Average predicate score: {score:.4f}')

                if score > self.best_score:
                    self.best_score = score
                    print(f'  *** Best model (score={score:.4f}) ***')

                    # Save
                    Path('outputs/nsnd_v4').mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'val_metrics': val_metrics,
                        'delta_psnr': delta_psnr,
                        'delta_ssim': delta_ssim,
                    }, 'outputs/nsnd_v4/best_model_v4.pth')

            # Step scheduler
            self.scheduler.step()

        print('\n' + '#'*70)
        print('# TRAINING COMPLETE')
        print('#'*70)
        print(f'\nBest validation score: {self.best_score:.4f}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('='*70)
    print('NEURO-SYMBOLIC OCT DENOISING V4 (POWERFUL CORRECTORS)')
    print(f'Using {_NUM_THREADS} CPU threads')
    print('='*70)

    # Configuration
    FINETUNE_BACKBONE = False  # Start with frozen backbone to force corrector learning
    BATCH_SIZE = 4
    EPOCHS = 25
    TRAIN_SAMPLES = 100
    VAL_SAMPLES = 10
    NUM_WORKERS = 2
    VAL_EVERY = 5
    PATCH_SIZE = 96
    CORRECTOR_HIDDEN_DIM = 128  # Hidden dimension for powerful correctors

    # Create V4 model
    print('\nInitializing V4 model with POWERFUL correctors...')
    model = NeuroSymbolicDenoiserV4(
        backbone_type='nafnet',
        width=64,
        corrector_hidden_dim=CORRECTOR_HIDDEN_DIM
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
    trainer = TrainerV4(
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
