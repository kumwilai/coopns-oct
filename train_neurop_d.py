#!/usr/bin/env python3
"""
Training script for NeurOp-D: Noise-Conditioned Neural Operator for OCT Denoising

IEEE TMI-Level Contributions:
1. Continuous noise embedding (not discrete classification)
2. HyperNetwork-generated spatially-varying denoising kernels
3. Physics-constrained latent decomposition
4. Uncertainty-guided adaptive refinement

Uses same data as run_end_to_end.sh for fair comparison.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "nsnd_oct"))

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from nsnd.models.neurop_d import NeurOpD, NeurOpDLoss, count_parameters
from nsnd.utils.metrics import compute_psnr, compute_ssim


class OCTDenoiseDataset(Dataset):
    """Dataset for OCT denoising with noise annotations."""

    def __init__(self, jsonl_path, patch_size=64, max_samples=None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        # Load images
        noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

        # Center crop
        h, w = noisy.shape
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        # Convert to tensors
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()

        # Ground truth noise weights (for evaluation)
        weights = torch.tensor([
            data['weights']['speckle'],
            data['weights']['banding'],
            data['weights']['gaussian'],
            data['weights']['shot']
        ], dtype=torch.float32)

        return {
            'noisy': noisy_t,
            'clean': clean_t,
            'weights': weights,
            'path': data['noisy_path']
        }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, args):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_recon = 0
    total_physics = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad()

        # Forward pass with interpretation for physics loss
        denoised, interpretation = model(noisy, return_interpretation=True)

        # Compute loss
        loss, loss_dict = criterion(
            denoised, clean, noisy, interpretation['noise_code']
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_recon += loss_dict['recon'].item()
        total_physics += loss_dict['physics'].item()
        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{loss_dict["recon"].item():.4f}',
            'Physics': f'{loss_dict["physics"].item():.4f}',
        })

    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'physics': total_physics / num_batches,
    }


def validate(model, val_loader, device, epoch):
    """
    Validate the model.
    """
    model.eval()

    total_psnr_noisy = 0
    total_psnr_denoised = 0
    total_ssim = 0
    total_samples = 0

    # Track noise type estimation accuracy
    noise_types = ['speckle', 'gaussian', 'banding', 'shot']
    per_class_correct = {nt: 0 for nt in noise_types}
    per_class_total = {nt: 0 for nt in noise_types}

    # Track uncertainty statistics
    uncertainty_sum = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            true_weights = batch['weights'].to(device)

            # Forward pass with interpretation
            denoised, interpretation = model(noisy, return_interpretation=True)

            # Track uncertainty
            uncertainty_sum += interpretation['uncertainty'].mean().item()

            # Compute metrics
            for i in range(noisy.size(0)):
                psnr_noisy = compute_psnr(noisy[i:i+1], clean[i:i+1])
                psnr_denoised = compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim = compute_ssim(denoised[i:i+1], clean[i:i+1])

                total_psnr_noisy += psnr_noisy
                total_psnr_denoised += psnr_denoised
                total_ssim += ssim

                # Per-class accuracy (compare noise type prediction)
                # Average noise_type over spatial dimensions
                noise_type = interpretation['noise_type'][i]  # [4, H, W]
                global_pred = noise_type.mean(dim=[1, 2])  # [4]
                pred_class = global_pred.argmax().item()
                true_class = true_weights[i].argmax().item()

                true_noise_type = noise_types[true_class]
                per_class_total[true_noise_type] += 1
                if pred_class == true_class:
                    per_class_correct[true_noise_type] += 1

                total_samples += 1

    # Calculate per-class accuracy
    per_class_accuracy = {}
    correct_top1 = 0
    for nt in noise_types:
        if per_class_total[nt] > 0:
            per_class_accuracy[nt] = 100.0 * per_class_correct[nt] / per_class_total[nt]
            correct_top1 += per_class_correct[nt]
        else:
            per_class_accuracy[nt] = 0.0

    avg_psnr_noisy = total_psnr_noisy / total_samples
    avg_psnr_denoised = total_psnr_denoised / total_samples

    return {
        'psnr_noisy': avg_psnr_noisy,
        'psnr_denoised': avg_psnr_denoised,
        'ssim': total_ssim / total_samples,
        'gain_over_noisy': avg_psnr_denoised - avg_psnr_noisy,
        'top1': 100.0 * correct_top1 / total_samples,
        'per_class_accuracy': per_class_accuracy,
        'per_class_total': per_class_total,
        'avg_uncertainty': uncertainty_sum / len(val_loader),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NeurOp-D")

    # Data (same as run_end_to_end.sh)
    parser.add_argument("--train_jsonl", type=str, default="weights_duke_analysis_maps_train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="weights_duke_analysis_maps_val.jsonl")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)

    # Model architecture
    parser.add_argument("--noise_dim", type=int, default=16, help="Dimension of noise code")
    parser.add_argument("--image_width", type=int, default=64, help="Base width of U-Net")
    parser.add_argument("--kernel_size", type=int, default=5, help="Size of generated kernels")
    parser.add_argument("--max_refine_iter", type=int, default=3, help="Max adaptive refinement iterations")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Physics loss weight
    parser.add_argument("--lambda_physics", type=float, default=0.1, help="Weight for physics loss")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="checkpoints/neurop_d")

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("NeurOp-D: Noise-Conditioned Neural Operator for OCT Denoising")
    print("="*70)
    print("\nIEEE TMI-Level Contributions:")
    print("  1. Continuous noise embedding (not discrete classification)")
    print("  2. HyperNetwork-generated spatially-varying denoising kernels")
    print("  3. Physics-constrained latent decomposition")
    print("  4. Uncertainty-guided adaptive refinement")
    print("\nKey Innovation:")
    print("  Don't SELECT from fixed operators -> GENERATE the operator dynamically")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Noise dimension: {args.noise_dim}")
    print(f"  Image width: {args.image_width}")
    print(f"  Kernel size: {args.kernel_size}")
    print(f"  Max refine iterations: {args.max_refine_iter}")
    print(f"  Lambda physics: {args.lambda_physics}")
    print("="*70)

    # Load data (same as run_end_to_end.sh)
    print("\nLoading datasets (Duke analysis maps - same as run_end_to_end.sh)...")
    train_dataset = OCTDenoiseDataset(args.train_jsonl, args.patch_size, args.max_train_samples)
    val_dataset = OCTDenoiseDataset(args.val_jsonl, args.patch_size, max_samples=50)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create model
    print("\nInitializing NeurOp-D...")
    model = NeurOpD(
        noise_dim=args.noise_dim,
        image_width=args.image_width,
        kernel_size=args.kernel_size,
        max_refine_iter=args.max_refine_iter,
    ).to(args.device)

    num_params = count_parameters(model)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Components:")
    print(f"    - NoiseEncoder: Continuous per-pixel noise embedding")
    print(f"    - KernelHyperNetwork: Generates {args.kernel_size}x{args.kernel_size} kernels from noise code")
    print(f"    - NoiseConditionedConv: Applies spatially-varying kernels")
    print(f"    - AdaptiveRefinement: Uncertainty-guided iteration (max {args.max_refine_iter})")
    print(f"    - UNet backbone: {args.image_width} base width")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss
    criterion = NeurOpDLoss(lambda_physics=args.lambda_physics)

    # Training loop
    best_psnr = 0
    best_gain = 0
    best_top1 = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")
        print(f"{'='*70}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, args.device, epoch, args)
        print(f"\nTraining Metrics:")
        print(f"  Total Loss: {train_metrics['loss']:.4f}")
        print(f"  Recon Loss: {train_metrics['recon']:.4f}")
        print(f"  Physics Loss: {train_metrics['physics']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, args.device, epoch)
        print(f"\nValidation Metrics:")
        print(f"  PSNR (noisy):      {val_metrics['psnr_noisy']:.2f} dB")
        print(f"  PSNR (NeurOp-D):   {val_metrics['psnr_denoised']:.2f} dB")
        print(f"  Gain over noisy:  +{val_metrics['gain_over_noisy']:.2f} dB")
        print(f"")
        print(f"  SSIM:              {val_metrics['ssim']:.4f}")
        print(f"  Avg Uncertainty:   {val_metrics['avg_uncertainty']:.4f}")
        print(f"")
        print(f"  Top-1 Accuracy:    {val_metrics['top1']:.1f}%")
        print(f"  Per-class Accuracy:")
        for noise_type in ['speckle', 'gaussian', 'banding', 'shot']:
            acc = val_metrics['per_class_accuracy'].get(noise_type, 0.0)
            count = val_metrics['per_class_total'].get(noise_type, 0)
            print(f"    {noise_type:8s}: {acc:5.1f}%  (n={count})")

        # Step scheduler
        scheduler.step()

        # Save best
        if val_metrics['psnr_denoised'] > best_psnr:
            best_psnr = val_metrics['psnr_denoised']
            best_gain = val_metrics['gain_over_noisy']
            best_top1 = val_metrics['top1']

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'psnr': best_psnr,
                'gain_over_noisy': best_gain,
                'top1': best_top1,
                'config': {
                    'noise_dim': args.noise_dim,
                    'image_width': args.image_width,
                    'kernel_size': args.kernel_size,
                    'max_refine_iter': args.max_refine_iter,
                },
            }
            torch.save(checkpoint, Path(args.output_dir) / "best_model.pth")
            print(f"\n  Best model saved! PSNR: {best_psnr:.2f} dB (+{best_gain:.2f} dB gain)")

        # Save last model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'psnr': val_metrics['psnr_denoised'],
            'config': {
                'noise_dim': args.noise_dim,
                'image_width': args.image_width,
                'kernel_size': args.kernel_size,
                'max_refine_iter': args.max_refine_iter,
            },
        }
        torch.save(checkpoint, Path(args.output_dir) / "last_model.pth")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best PSNR (NeurOp-D):   {best_psnr:.2f} dB")
    print(f"Best Gain over Noisy:   +{best_gain:.2f} dB")
    print(f"Best Top-1 Accuracy:    {best_top1:.1f}%")
    print(f"Checkpoint:             {args.output_dir}/best_model.pth")
    print("="*70)
    print("\nIEEE TMI Contributions Summary:")
    print("  1. Continuous noise embedding - captures full noise characteristics")
    print("  2. HyperNetwork-generated kernels - infinite family of operators")
    print("  3. Physics-constrained latent - interpretable + principled")
    print("  4. Adaptive computation - efficient + better quality")
    print("="*70)


if __name__ == "__main__":
    main()
