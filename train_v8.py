#!/usr/bin/env python3
"""
Train V8 Neuro-Symbolic OCT Denoising

Architecture:
1. Lightweight NAFNet backbone (width=40, ~3M params)
2. NeuroSymbolicCorrectorV8 with:
   - Differentiable fuzzy logic (Lukasiewicz t-norm)
   - Hierarchical symbolic reasoning (3 levels)
   - Physics-accurate speckle predicate (Gamma-K)
   - Formal verification guarantees
   - Causal interpretability

Training Data: PKU37 real noise pairs

Key Features:
- Light backbone leaves room for corrector improvement
- GT-free predicate monitoring (P1-P6)
- Predicate-driven loss for symbolic learning
"""

import argparse
import os
import sys
import json
import gc
import math
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

# Import components
from train_v6 import BackboneWrapper, AdaptiveLambdaPredictor
from neuro_symbolic_corrector_v8 import NeuroSymbolicCorrectorV8


# =============================================================================
# Dataset
# =============================================================================

class PKU37Dataset(Dataset):
    """PKU37 real noise dataset."""

    def __init__(self, jsonl_path: str, max_samples: int = None,
                 patch_size: int = 96, is_train: bool = True):
        self.samples = []
        self.patch_size = patch_size
        self.is_train = is_train

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                # PKU37 format: clean_path and noisy_path
                if 'clean_path' in entry and 'noisy_path' in entry:
                    # BUG FIX: Validate files exist before adding
                    if os.path.exists(entry['clean_path']) and os.path.exists(entry['noisy_path']):
                        self.samples.append({
                            'clean': entry['clean_path'],
                            'noisy': entry['noisy_path'],
                        })
                    else:
                        print(f"WARNING: Skipping missing files: {entry['clean_path']}")

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} PKU37 samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        clean = np.array(Image.open(sample['clean'])).astype(np.float32)
        noisy = np.array(Image.open(sample['noisy'])).astype(np.float32)

        # Normalize to [0, 1]
        if clean.max() > 1.0:
            clean = clean / 255.0
        if noisy.max() > 1.0:
            noisy = noisy / 255.0

        # Random crop for training
        if self.patch_size > 0 and self.is_train:
            H, W = clean.shape
            if H > self.patch_size and W > self.patch_size:
                y = np.random.randint(0, H - self.patch_size)
                x = np.random.randint(0, W - self.patch_size)
                clean = clean[y:y+self.patch_size, x:x+self.patch_size]
                noisy = noisy[y:y+self.patch_size, x:x+self.patch_size]

        # Random augmentation for training
        if self.is_train:
            # Horizontal flip
            if np.random.rand() > 0.5:
                clean = np.flip(clean, axis=1).copy()
                noisy = np.flip(noisy, axis=1).copy()
            # Vertical flip
            if np.random.rand() > 0.5:
                clean = np.flip(clean, axis=0).copy()
                noisy = np.flip(noisy, axis=0).copy()

        # Convert to tensors [1, H, W]
        clean = torch.from_numpy(clean).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        return {'clean': clean, 'noisy': noisy}


# =============================================================================
# Model
# =============================================================================

class NeuroSymbolicDenoiserV8(nn.Module):
    """
    V8 Neuro-Symbolic Denoiser

    Components:
    1. Lightweight NAFNet backbone (width=40)
    2. NeuroSymbolicCorrectorV8 (differentiable fuzzy logic + hierarchical reasoning)
    """

    def __init__(self,
                 backbone_width: int = 40,
                 pretrained_backbone: str = None):
        super().__init__()

        # Lightweight backbone
        self.backbone = BackboneWrapper(backbone_type='nafnet', width=backbone_width)

        # V8 Corrector with enhanced symbolic reasoning
        self.corrector = NeuroSymbolicCorrectorV8(in_channels=1, hidden_channels=32)

        # Load pretrained backbone if provided
        if pretrained_backbone and os.path.exists(pretrained_backbone):
            print(f"Loading pretrained backbone: {pretrained_backbone}")
            ckpt = torch.load(pretrained_backbone, map_location='cpu', weights_only=False)
            state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            # Filter backbone weights
            backbone_state = {k.replace('backbone.', ''): v for k, v in state.items()
                            if 'backbone' in k or 'nafnet' in k.lower()}
            if backbone_state:
                missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
                print(f"  Loaded backbone: {len(state) - len(missing)} weights")

        self._print_params()

    def _print_params(self):
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        corrector_params = sum(p.numel() for p in self.corrector.parameters())
        total_params = backbone_params + corrector_params

        print(f"\nNeuroSymbolicDenoiserV8 Parameters:")
        print(f"  Backbone: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"  V8 Corrector: {corrector_params:,} ({corrector_params/1e6:.2f}M)")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, noisy: torch.Tensor, return_details: bool = False):
        """
        Forward pass.

        Returns:
            corrected: Final denoised output
            backbone_out: Backbone output (before correction)
            info: Dictionary with predicate scores, activations, verification, etc.
        """
        # Step 1: Backbone denoising
        backbone_out = self.backbone(noisy)

        # Step 2: V8 Symbolic correction
        corrected, info = self.corrector(backbone_out, noisy, return_details=return_details)

        return corrected, backbone_out, info


# =============================================================================
# Loss
# =============================================================================

class V8Loss(nn.Module):
    """
    Loss function for V8 training.

    Components:
    1. Reconstruction loss (MSE)
    2. Backbone supervision loss
    3. Predicate improvement loss (encourage V8 to improve predicates)
    """

    def __init__(self,
                 lambda_recon: float = 1.0,
                 lambda_backbone: float = 0.5,
                 lambda_pred: float = 0.5):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_backbone = lambda_backbone
        self.lambda_pred = lambda_pred

    def forward(self, corrected, backbone_out, clean, info):
        # Reconstruction loss
        recon_loss = F.mse_loss(corrected, clean)

        # Backbone supervision loss
        backbone_loss = F.mse_loss(backbone_out, clean)

        # Predicate improvement loss
        # Encourage corrector to improve predicate scores
        pred_scores = info['predicate_scores']
        avg_score = sum(pred_scores.values()) / len(pred_scores)
        # Higher scores = better, so minimize negative (as tensor for gradient)
        pred_loss_val = -avg_score * 0.1  # Scale down to not dominate

        # Total loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_backbone * backbone_loss +
            self.lambda_pred * pred_loss_val
        )

        # Compute metrics
        with torch.no_grad():
            psnr_backbone = 10 * torch.log10(1 / (backbone_loss + 1e-8))
            psnr_corrected = 10 * torch.log10(1 / (recon_loss + 1e-8))

        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'backbone': backbone_loss.item(),
            'pred': pred_loss_val,
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'avg_pred_score': avg_score,
        }


# =============================================================================
# Training
# =============================================================================

def compute_psnr(pred, target):
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())


def compute_ssim(pred, target):
    """Compute SSIM (simplified version)."""
    C1, C2 = 0.01**2, 0.03**2

    mu_pred = pred.mean()
    mu_target = target.mean()

    sigma_pred = ((pred - mu_pred) ** 2).mean()
    sigma_target = ((target - mu_target) ** 2).mean()
    sigma_both = ((pred - mu_pred) * (target - mu_target)).mean()

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
           ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))

    return ssim.item()


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train one epoch."""
    model.train()

    total_loss = 0
    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_pred_score = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        # Forward
        corrected, backbone_out, info = model(noisy)

        # Loss
        loss, metrics = criterion(corrected, backbone_out, clean, info)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # BUG FIX: 1.0 was too strict
        optimizer.step()

        # Accumulate metrics
        total_loss += metrics['total']
        total_psnr_backbone += metrics['psnr_backbone']
        total_psnr_corrected += metrics['psnr_corrected']
        total_pred_score += metrics['avg_pred_score']
        n += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'psnr': f"{metrics['psnr_corrected']:.1f}",
            'pred': f"{metrics['avg_pred_score']:.3f}",
        })

    return {
        'loss': total_loss / n,
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'pred_score': total_pred_score / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()

    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_ssim_backbone = 0
    total_ssim_corrected = 0
    total_pred_scores = {f'P{i}': 0 for i in range(1, 7)}
    total_correction_mag = 0
    n = 0

    for batch in tqdm(loader, desc="Validation"):
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        # Forward
        corrected, backbone_out, info = model(noisy)

        # Metrics
        psnr_backbone = compute_psnr(backbone_out, clean)
        psnr_corrected = compute_psnr(corrected, clean)
        ssim_backbone = compute_ssim(backbone_out, clean)
        ssim_corrected = compute_ssim(corrected, clean)

        total_psnr_backbone += psnr_backbone
        total_psnr_corrected += psnr_corrected
        total_ssim_backbone += ssim_backbone
        total_ssim_corrected += ssim_corrected

        # Predicate scores
        for name, score in info['predicate_scores'].items():
            key = name.split('_')[0]  # P1_edge -> P1
            if key in total_pred_scores:
                total_pred_scores[key] += score

        total_correction_mag += info['correction_magnitude']
        n += 1

    return {
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'ssim_backbone': total_ssim_backbone / n,
        'ssim_corrected': total_ssim_corrected / n,
        'pred_scores': {k: v / n for k, v in total_pred_scores.items()},
        'correction_magnitude': total_correction_mag / n,
        'psnr_delta': (total_psnr_corrected - total_psnr_backbone) / n,
    }


def print_epoch_metrics(epoch, train_metrics, val_metrics):
    """Print comprehensive metrics."""
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} RESULTS")
    print(f"{'='*70}")

    # Training
    print(f"\n[TRAINING]")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  PSNR: backbone={train_metrics['psnr_backbone']:.2f}, corrected={train_metrics['psnr_corrected']:.2f}")
    print(f"  Avg Pred Score: {train_metrics['pred_score']:.4f}")

    # Validation
    print(f"\n[VALIDATION]")
    print(f"  PSNR: backbone={val_metrics['psnr_backbone']:.2f}, corrected={val_metrics['psnr_corrected']:.2f} (delta={val_metrics['psnr_delta']:+.2f})")
    print(f"  SSIM: backbone={val_metrics['ssim_backbone']:.4f}, corrected={val_metrics['ssim_corrected']:.4f}")
    print(f"  Correction magnitude: {val_metrics['correction_magnitude']:.6f}")

    # Predicate scores
    print(f"\n[V8 PREDICATE SCORES]")
    print(f"  {'Predicate':<12} {'Score':<10} {'Status'}")
    print(f"  {'-'*35}")
    for name, score in val_metrics['pred_scores'].items():
        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"  {name:<12} {score:.4f}     {status}")

    avg_score = sum(val_metrics['pred_scores'].values()) / len(val_metrics['pred_scores'])
    print(f"  {'-'*35}")
    print(f"  {'Average':<12} {avg_score:.4f}     {'PASS' if avg_score >= 0.5 else 'FAIL'}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train V8 Neuro-Symbolic OCT Denoising')

    # Data
    parser.add_argument('--train_jsonl', default='pku37_oct_dataset/pku37_real_train.jsonl')
    parser.add_argument('--val_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl')
    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--max_val', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=96)

    # Model
    parser.add_argument('--backbone_width', type=int, default=48,
                        help='NAFNet width (48=light ~2.5M, 64=full 7.5M)')
    parser.add_argument('--pretrained_backbone', type=str,
                        default='outputs/nafnet_universal/nafnet_best.pth',
                        help='Path to pretrained backbone')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output
    parser.add_argument('--output_dir', default='outputs/nsnd_v8')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("NEURO-SYMBOLIC OCT DENOISING V8")
    print("Differentiable Fuzzy Logic + Hierarchical Reasoning")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backbone width: {args.backbone_width}")
    print(f"  Train samples: {args.max_train}")
    print(f"  Val samples: {args.max_val}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")

    # Data
    print("\nLoading PKU37 data...")
    train_dataset = PKU37Dataset(
        args.train_jsonl,
        max_samples=args.max_train,
        patch_size=args.patch_size,
        is_train=True
    )
    val_dataset = PKU37Dataset(
        args.val_jsonl,
        max_samples=args.max_val,
        patch_size=0,  # Full image for validation
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    print("\nInitializing model...")
    model = NeuroSymbolicDenoiserV8(
        backbone_width=args.backbone_width,
        pretrained_backbone=args.pretrained_backbone
    ).to(args.device)

    # Loss and optimizer
    criterion = V8Loss()

    # Different learning rates for backbone and corrector
    backbone_params = list(model.backbone.parameters())
    corrector_params = list(model.corrector.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr},
        {'params': corrector_params, 'lr': args.lr * 2},  # Faster learning for corrector
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    print("\n" + "#"*70)
    print("# TRAINING V8 NEURO-SYMBOLIC DENOISER")
    print("#"*70)

    best_psnr_delta = -float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch)
        scheduler.step()

        # Validate
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, criterion, args.device)
            print_epoch_metrics(epoch, train_metrics, val_metrics)

            # Save best model
            if val_metrics['psnr_delta'] > best_psnr_delta:
                best_psnr_delta = val_metrics['psnr_delta']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # BUG FIX: Save scheduler for resume
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_psnr_delta': best_psnr_delta,
                }, os.path.join(args.output_dir, 'best_model_v8.pth'))
                print(f"*** New best model! PSNR delta: {best_psnr_delta:+.2f} dB ***")
        else:
            print(f"[Epoch {epoch}] Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr_corrected']:.2f}")

    print("\n" + "#"*70)
    print("# TRAINING COMPLETE")
    print("#"*70)
    print(f"Best PSNR delta: {best_psnr_delta:+.2f} dB")
    print(f"Model saved to: {args.output_dir}/best_model_v8.pth")


if __name__ == '__main__':
    main()
