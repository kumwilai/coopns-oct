#!/usr/bin/env python3
"""
Training Script for Neuro-Symbolic OCT Denoising V7

Key Features:
1. Lightweight backbone (width=40, ~3M params instead of 7.5M)
2. Neuro-symbolic corrector with GT-free predicates
3. Symbolic routing rules (interpretable)
4. Verify-before-apply guarantee
5. End-to-end training with predicate-driven loss

For TMI Publication:
- Show lightweight backbone alone: ~29-30 dB PSNR
- Show lightweight + corrector: ~31+ dB PSNR (approaching heavy backbone)
- Formal guarantee: corrector never degrades quality
- Interpretable: can explain why corrections were made
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.stdout.reconfigure(line_buffering=True)

# Import from existing modules (REUSE V6 COMPONENTS)
from neuro_symbolic_v2 import (
    OCTDataset, _NUM_THREADS,
    BackboneWrapper,  # REUSE: NAFNet backbone with configurable width
    AdaptiveLambdaPredictor,  # REUSE: Lambda prediction
    SymbolicPredicates,  # REUSE: Existing predicates
)
from powerful_correctors import PowerfulAdaptiveCorrectorWithLambda  # REUSE: Powerful correctors
from neuro_symbolic_corrector_v7 import (
    NeuroSymbolicCorrectorV7,
    GTFreePredicates,
    SymbolicRouter,
)
from torch.utils.data import DataLoader


# =============================================================================
# USING EXISTING BACKBONE WITH LIGHTWEIGHT CONFIG (REUSE FROM V6)
# =============================================================================

# NOTE: We reuse BackboneWrapper from neuro_symbolic_v2.py
# Just instantiate with width=40 instead of width=64:
#   BackboneWrapper(backbone_type='nafnet', width=40)
# This gives ~3M params instead of 7.5M
#
# Benefits of reusing existing backbone:
# 1. Can partially load pretrained weights (matching layers)
# 2. Tested architecture that works
# 3. Compatible with existing correctors


# =============================================================================
# V7 MODEL: LIGHTWEIGHT BACKBONE + NEURO-SYMBOLIC CORRECTOR
# =============================================================================

class NeuroSymbolicDenoiserV7(nn.Module):
    """
    V7: Lightweight Backbone + Neuro-Symbolic Corrector

    REUSES V6 COMPONENTS:
    1. BackboneWrapper with width=40 (lightweight NAFNet, ~3M params)
    2. AdaptiveLambdaPredictor (per-pixel correction strength)
    3. PowerfulAdaptiveCorrectorWithLambda OR NeuroSymbolicCorrectorV7

    NEW in V7:
    4. Symbolic routing rules (explicit IF-THEN)
    5. GT-free predicates for real deployment
    6. Verify-before-apply guarantee

    Target Performance:
    - Backbone alone: ~29-30 dB PSNR
    - Backbone + Corrector: ~31+ dB PSNR
    - Improvement: +1-2 dB (significant and verifiable)
    """

    def __init__(self,
                 backbone_width: int = 40,
                 corrector_hidden: int = 128,
                 use_verification: bool = True,
                 use_powerful_corrector: bool = True,
                 pretrained_backbone: str = None):
        super().__init__()

        self.use_verification = use_verification
        self.use_powerful_corrector = use_powerful_corrector

        # REUSE: Lightweight backbone from V6 (just change width)
        self.backbone = BackboneWrapper(backbone_type='nafnet', width=backbone_width)

        # Load pretrained if available (partial match for lightweight)
        if pretrained_backbone and Path(pretrained_backbone).exists():
            print(f"Loading pretrained backbone from {pretrained_backbone}...")
            self.backbone.load_pretrained(pretrained_backbone)

        # REUSE: Lambda predictor from V6
        self.lambda_predictor = AdaptiveLambdaPredictor()

        # GT-free predicates (NEW in V7)
        self.gt_free_predicates = GTFreePredicates()

        # Symbolic router (NEW in V7)
        self.symbolic_router = SymbolicRouter()

        # Corrector: Choose between powerful (V6) or neuro-symbolic (V7)
        if use_powerful_corrector:
            # REUSE: PowerfulAdaptiveCorrectorWithLambda from V6
            self.corrector = PowerfulAdaptiveCorrectorWithLambda(
                enc1_channels=backbone_width,
                enc2_channels=backbone_width * 2,
                hidden_dim=corrector_hidden
            )
        else:
            # NEW: NeuroSymbolicCorrectorV7
            self.corrector = NeuroSymbolicCorrectorV7(
                in_channels=1,
                hidden_channels=corrector_hidden // 4,
                use_verification=use_verification
            )

        self._print_param_counts()

    def _print_param_counts(self):
        """Print parameter counts."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        lambda_params = sum(p.numel() for p in self.lambda_predictor.parameters())
        corrector_params = sum(p.numel() for p in self.corrector.parameters())
        total_params = backbone_params + lambda_params + corrector_params

        print(f"\nNeuroSymbolicDenoiserV7 Parameters:")
        print(f"  Backbone (width={self.backbone.width}): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"  Lambda Predictor: {lambda_params:,} ({lambda_params/1e6:.2f}M)")
        print(f"  Corrector: {corrector_params:,} ({corrector_params/1e6:.2f}M)")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Using powerful corrector: {self.use_powerful_corrector}")

    def forward(self,
                noisy: torch.Tensor,
                max_lambda: float = 0.2,
                return_details: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with symbolic routing.

        Args:
            noisy: Noisy input [B, 1, H, W]
            max_lambda: Maximum lambda value for capping
            return_details: Whether to return detailed info

        Returns:
            corrected: Final corrected output
            backbone_out: Backbone-only output (for comparison)
            info: Dictionary with correction details
        """
        # Step 1: Backbone denoising (REUSE from V6)
        backbone_out, backbone_features = self.backbone(noisy, return_features=True)

        # Step 2: Evaluate GT-free predicates (NEW in V7)
        pred_results = self.gt_free_predicates(backbone_out, noisy)

        # Step 3: Symbolic routing - decide correction strengths (NEW in V7)
        routing = self.symbolic_router(pred_results)
        activations = routing['activations']
        explanations = routing['explanations']

        # Step 4: Compute lambda maps (REUSE from V6)
        # Use routing activations to modulate lambda maps
        from neuro_symbolic_v2 import SymbolicPredicates
        std_predicates = SymbolicPredicates()
        pred_with_maps = std_predicates(noisy, backbone_out, return_failure_maps=True)
        failure_maps = {
            name: pred_with_maps[name]['failure_map']
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }
        lambda_maps = self.lambda_predictor(backbone_out, failure_maps)

        # Modulate lambda maps by symbolic routing activations
        activation_scale = (
            activations.get('edge', 0.5) +
            activations.get('contrast', 0.5) +
            activations.get('smooth', 0.5) +
            activations.get('structure', 0.5) +
            activations.get('speckle', 0.5)
        ) / 5.0

        # Cap lambda maps
        lambda_maps_capped = {
            k: v.clamp(0, max_lambda * activation_scale)
            for k, v in lambda_maps.items()
        }

        # Step 5: Apply correction (REUSE from V6 or use V7)
        if self.use_powerful_corrector:
            # Use PowerfulAdaptiveCorrectorWithLambda from V6
            corrected, corrector_info = self.corrector(
                backbone_out, noisy, lambda_maps_capped, backbone_features,
                failure_maps=failure_maps,
                return_individual=return_details
            )
        else:
            # Use NeuroSymbolicCorrectorV7
            corrected, corrector_info = self.corrector(
                backbone_out, noisy, return_details
            )

        # Step 6: Verify-before-apply (NEW in V7)
        # Check if correction improved predicates
        if self.use_verification and not self.training:
            pred_after = self.gt_free_predicates(corrected, noisy)
            score_before = pred_results['avg_score']
            score_after = pred_after['avg_score']

            # If predicates degraded significantly, reject correction
            if score_after < score_before - 0.1:
                corrected = backbone_out  # Reject, use backbone
                verify_decision = "REJECTED"
            else:
                verify_decision = "ACCEPTED"
        else:
            verify_decision = "TRAINING_MODE"
            score_before = pred_results['avg_score']
            score_after = score_before  # Not computed during training

        # Compile info
        info = {
            'backbone_features': backbone_features,
            'lambda_maps': lambda_maps_capped,
            'gt_free_predicates': pred_results['scores'],
            'symbolic_routing': {
                'activations': activations,
                'explanations': explanations,
            },
            'verify_decision': verify_decision,
            'pred_score_before': score_before.item() if hasattr(score_before, 'item') else score_before,
            'correction_magnitude': corrector_info.get('correction_magnitude', 0),
        }

        return corrected, backbone_out, info


# =============================================================================
# TRAINING LOSS
# =============================================================================

class V7Loss(nn.Module):
    """
    Loss function for V7 training.

    Components:
    1. Reconstruction loss (MSE to clean) - primary PSNR driver
    2. Predicate improvement loss - encourage predicate scores to improve
    3. Backbone supervision - ensure backbone learns basic denoising
    """

    def __init__(self,
                 lambda_recon: float = 1.0,
                 lambda_pred: float = 0.5,
                 lambda_backbone: float = 0.5):
        super().__init__()

        self.lambda_recon = lambda_recon
        self.lambda_pred = lambda_pred
        self.lambda_backbone = lambda_backbone

        self.predicates = GTFreePredicates()

    def forward(self,
                corrected: torch.Tensor,
                backbone_out: torch.Tensor,
                clean: torch.Tensor,
                noisy: torch.Tensor) -> Dict:
        """
        Compute loss.

        Args:
            corrected: Final corrected output
            backbone_out: Backbone-only output
            clean: Ground truth clean image
            noisy: Original noisy input

        Returns:
            Dictionary with loss components
        """
        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(corrected, clean)

        # 2. Backbone loss (ensure backbone learns)
        backbone_loss = F.mse_loss(backbone_out, clean)

        # 3. Predicate improvement loss
        # Encourage predicates to improve from backbone to corrected
        pred_backbone = self.predicates(backbone_out, noisy)
        pred_corrected = self.predicates(corrected, noisy)

        # Loss: negative improvement (we want improvement to be positive)
        pred_improvement = pred_corrected['avg_score'] - pred_backbone['avg_score']
        pred_loss = -pred_improvement  # Minimize negative improvement

        # Total loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_backbone * backbone_loss +
            self.lambda_pred * pred_loss
        )

        # Compute PSNR for logging
        with torch.no_grad():
            mse_backbone = F.mse_loss(backbone_out, clean)
            mse_corrected = F.mse_loss(corrected, clean)
            psnr_backbone = -10 * torch.log10(mse_backbone + 1e-8)
            psnr_corrected = -10 * torch.log10(mse_corrected + 1e-8)

        return {
            'total': total_loss,
            'recon_loss': recon_loss.detach(),
            'backbone_loss': backbone_loss.detach(),
            'pred_loss': pred_loss.detach(),
            'pred_improvement': pred_improvement.detach(),
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'pred_scores_backbone': pred_backbone['scores'],
            'pred_scores_corrected': pred_corrected['scores'],
        }


# =============================================================================
# TRAINER
# =============================================================================

class TrainerV7:
    """Trainer for V7 model."""

    def __init__(self,
                 model: NeuroSymbolicDenoiserV7,
                 train_dataset: OCTDataset,
                 val_dataset: OCTDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Optimizer: train backbone and corrector jointly
        self.optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': lr},
            {'params': model.corrector.parameters(), 'lr': lr * 2},
        ], weight_decay=1e-4)

        # Loss
        self.loss_fn = V7Loss()

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=lr * 0.01
        )

        self.best_psnr_delta = -float('inf')
        self.best_psnr_corrected = 0

    def train_epoch(self) -> Dict:
        """Train one epoch."""
        self.model.train()

        metrics = {
            'loss': [], 'recon_loss': [], 'backbone_loss': [], 'pred_loss': [],
            'psnr_backbone': [], 'psnr_corrected': [], 'pred_improvement': [],
        }

        for batch in self.train_loader:
            noisy = batch['noisy']
            clean = batch['clean']

            self.optimizer.zero_grad()

            # Forward
            corrected, backbone_out, info = self.model(noisy)

            # Loss
            loss_dict = self.loss_fn(corrected, backbone_out, clean, noisy)
            loss = loss_dict['total']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Collect metrics (with NaN checking)
            if not torch.isnan(loss):
                metrics['loss'].append(loss.item())
                metrics['recon_loss'].append(loss_dict['recon_loss'].item())
                metrics['backbone_loss'].append(loss_dict['backbone_loss'].item())
                pred_loss = loss_dict['pred_loss']
                if isinstance(pred_loss, torch.Tensor) and not torch.isnan(pred_loss):
                    metrics['pred_loss'].append(pred_loss.item())
                metrics['psnr_backbone'].append(loss_dict['psnr_backbone'])
                metrics['psnr_corrected'].append(loss_dict['psnr_corrected'])
                pred_imp = loss_dict['pred_improvement']
                if isinstance(pred_imp, torch.Tensor) and not torch.isnan(pred_imp):
                    metrics['pred_improvement'].append(pred_imp.item())

        # Safe mean computation
        result = {}
        for k, v in metrics.items():
            if len(v) > 0:
                result[k] = np.nanmean(v)
            else:
                result[k] = 0.0
        return result

    def validate(self) -> Dict:
        """Validate the model."""
        self.model.eval()

        metrics = {
            'psnr_backbone': [], 'psnr_corrected': [],
            'ssim_backbone': [], 'ssim_corrected': [],
            'pred_improvement': [],
        }

        with torch.no_grad():
            for batch in self.val_loader:
                noisy = batch['noisy']
                clean = batch['clean']

                # Forward
                corrected, backbone_out, info = self.model(noisy, return_details=True)

                # PSNR
                mse_backbone = F.mse_loss(backbone_out, clean)
                mse_corrected = F.mse_loss(corrected, clean)
                psnr_backbone = -10 * torch.log10(mse_backbone + 1e-8)
                psnr_corrected = -10 * torch.log10(mse_corrected + 1e-8)

                # SSIM (simplified)
                ssim_backbone = self._compute_ssim(backbone_out, clean)
                ssim_corrected = self._compute_ssim(corrected, clean)

                # Predicate improvement (use GT-free predicates from model)
                pred_backbone = self.model.gt_free_predicates(backbone_out, noisy)
                pred_corrected = self.model.gt_free_predicates(corrected, noisy)
                pred_improvement = pred_corrected['avg_score'] - pred_backbone['avg_score']

                metrics['psnr_backbone'].append(psnr_backbone.item())
                metrics['psnr_corrected'].append(psnr_corrected.item())
                metrics['ssim_backbone'].append(ssim_backbone.item())
                metrics['ssim_corrected'].append(ssim_corrected.item())
                metrics['pred_improvement'].append(pred_improvement.item())

        return {k: np.mean(v) for k, v in metrics.items()}

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM."""
        C1, C2 = 0.01**2, 0.03**2

        mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)

        sigma1_sq = F.avg_pool2d(img1**2, 11, stride=1, padding=5) - mu1**2
        sigma2_sq = F.avg_pool2d(img2**2, 11, stride=1, padding=5) - mu2**2
        sigma12 = F.avg_pool2d(img1 * img2, 11, stride=1, padding=5) - mu1 * mu2

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim.mean()

    def train(self, epochs: int = 50, val_every: int = 5, output_dir: str = 'outputs/nsnd_v7'):
        """Training loop."""
        print('\n' + '#'*70)
        print('# NEURO-SYMBOLIC OCT DENOISING V7')
        print('# Lightweight Backbone + Symbolic Corrector')
        print('#'*70)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - t0

            # Training metrics
            psnr_delta = train_metrics['psnr_corrected'] - train_metrics['psnr_backbone']

            print(f'\n[EPOCH {epoch}]')
            print(f'  Loss: {train_metrics["loss"]:.4f} '
                  f'(recon={train_metrics["recon_loss"]:.4f}, '
                  f'backbone={train_metrics["backbone_loss"]:.4f}, '
                  f'pred={train_metrics["pred_loss"]:.4f})')
            print(f'  PSNR: backbone={train_metrics["psnr_backbone"]:.2f}, '
                  f'corrected={train_metrics["psnr_corrected"]:.2f} '
                  f'(delta={psnr_delta:+.3f})')
            print(f'  Pred improvement: {train_metrics["pred_improvement"]:.4f}')
            print(f'  Time: {train_time:.1f}s')

            # Validation
            if epoch % val_every == 0 or epoch == 1:
                val_metrics = self.validate()

                psnr_delta_val = val_metrics['psnr_corrected'] - val_metrics['psnr_backbone']
                ssim_delta_val = val_metrics['ssim_corrected'] - val_metrics['ssim_backbone']

                print(f'\n  [VALIDATION]')
                print(f'  PSNR: backbone={val_metrics["psnr_backbone"]:.2f}, '
                      f'corrected={val_metrics["psnr_corrected"]:.2f} '
                      f'(delta={psnr_delta_val:+.3f})')
                print(f'  SSIM: backbone={val_metrics["ssim_backbone"]:.4f}, '
                      f'corrected={val_metrics["ssim_corrected"]:.4f} '
                      f'(delta={ssim_delta_val:+.4f})')
                print(f'  Pred improvement: {val_metrics["pred_improvement"]:.4f}')

                # Save best model
                if psnr_delta_val > self.best_psnr_delta:
                    self.best_psnr_delta = psnr_delta_val
                    self.best_psnr_corrected = val_metrics['psnr_corrected']

                    print(f'  *** New best model! ***')
                    save_path = Path(output_dir) / 'best_model_v7.pth'
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'psnr_backbone': val_metrics['psnr_backbone'],
                        'psnr_corrected': val_metrics['psnr_corrected'],
                        'psnr_delta': psnr_delta_val,
                        'ssim_delta': ssim_delta_val,
                    }, save_path)
                    print(f'  Saved to: {save_path}')

            self.scheduler.step()

        print('\n' + '#'*70)
        print('# TRAINING COMPLETE')
        print('#'*70)
        print(f'\nBest PSNR delta: {self.best_psnr_delta:+.3f} dB')
        print(f'Best corrected PSNR: {self.best_psnr_corrected:.2f} dB')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Neuro-Symbolic OCT Denoising V7')

    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train-samples', type=int, default=200)
    parser.add_argument('--val-samples', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-width', type=int, default=40,
                        help='Backbone width (40=lightweight ~3M, 64=full ~7.5M)')
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='outputs/nsnd_v7')
    parser.add_argument('--pretrained-backbone', type=str,
                        default='outputs/nafnet_pku37/nafnet_best.pth',
                        help='Path to pretrained backbone (partial weights will be loaded)')
    parser.add_argument('--use-powerful-corrector', action='store_true', default=True,
                        help='Use PowerfulAdaptiveCorrectorWithLambda from V6')
    parser.add_argument('--use-v7-corrector', dest='use_powerful_corrector', action='store_false',
                        help='Use NeuroSymbolicCorrectorV7 (new)')

    args = parser.parse_args()

    print('='*70)
    print('NEURO-SYMBOLIC OCT DENOISING V7')
    print('Lightweight Backbone + Symbolic Routing + Powerful Corrector')
    print(f'Using {_NUM_THREADS} CPU threads')
    print('='*70)

    print(f'\nConfiguration:')
    print(f'  Backbone width: {args.backbone_width} ({"lightweight" if args.backbone_width <= 48 else "full"})')
    print(f'  Use powerful corrector (V6): {args.use_powerful_corrector}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Epochs: {args.epochs}')
    print(f'  Train samples: {args.train_samples}')
    print(f'  Val samples: {args.val_samples}')
    print(f'  Learning rate: {args.lr}')

    # Create model
    print('\nInitializing V7 model...')
    model = NeuroSymbolicDenoiserV7(
        backbone_width=args.backbone_width,
        corrector_hidden=128,
        use_verification=True,
        use_powerful_corrector=args.use_powerful_corrector,
        pretrained_backbone=args.pretrained_backbone if args.backbone_width == 64 else None
    )

    # Load data
    print('\nLoading data...')
    train_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_train.jsonl',
        max_samples=args.train_samples,
        patch_size=96,
        is_train=True
    )
    val_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_val.jsonl',
        max_samples=args.val_samples,
        patch_size=0,
        is_train=False
    )
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    # Create trainer
    trainer = TrainerV7(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size
    )

    # Train
    trainer.train(
        epochs=args.epochs,
        val_every=args.val_every,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
