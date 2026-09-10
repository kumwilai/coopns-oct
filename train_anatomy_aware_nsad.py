#!/usr/bin/env python3
"""
Anatomy-Aware NSAD Training Script

This script trains the AnatomyAwareSANSD model which incorporates
OCT anatomical knowledge for improved denoising.

NOVEL CONTRIBUTIONS:
1. Per-pixel noise TYPE decomposition
2. Soft routing to MULTIPLE classical operators
3. Differentiable mixture of NAMED operators
4. Full per-pixel interpretability
5. Layer-aware noise decomposition (NEW)
6. Depth-adaptive processing (NEW)
7. Anatomy-preserving constraints (NEW)
8. Anatomically-informed expert routing (NEW)

Usage:
    python train_anatomy_aware_nsad.py --epochs 30 --device cpu
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.sansd import AnatomyAwareSANSD, SANSDWithBackbone
from nsnd.models.anatomy_aware import AnatomyPreservingLoss
from nsnd.models.nafnet import NAFNetFullFiLM
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


def expert_diversity_loss(expert_outputs: dict) -> torch.Tensor:
    """Encourage experts to produce different outputs."""
    outputs = list(expert_outputs.values())
    n = len(outputs)

    diversity = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            diff = (outputs[i] - outputs[j]).abs().mean()
            diversity += diff
            count += 1

    if count > 0:
        return -diversity / count  # Negative because we want to maximize diversity
    return torch.tensor(0.0)


def expert_usage_loss(noise_type: torch.Tensor, min_usage: float = 0.05) -> torch.Tensor:
    """Encourage using all experts."""
    avg_usage = noise_type.mean(dim=[0, 2, 3])
    underused = F.relu(min_usage - avg_usage)
    return underused.sum()


def train_epoch(model, train_loader, optimizer, anatomy_loss, device, epoch, use_anatomy_loss=True):
    """Train for one epoch with anatomy-aware loss."""
    model.train()

    total_loss = 0
    total_recon = 0
    total_anatomy = 0
    total_diversity = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad()

        # Forward pass
        denoised, interpretation = model(noisy, return_interpretation=True)

        # Get backbone output and error
        backbone_out = interpretation['backbone_out']
        backbone_error = (backbone_out - clean).abs()

        # Spatially-weighted loss: emphasize regions where backbone fails
        error_weight = 1.0 + 2.0 * backbone_error
        weighted_recon = ((denoised - clean).pow(2) * error_weight).mean()

        # MEMORY FIX: Delete backbone_error and error_weight after use
        del backbone_error, error_weight

        # Anatomy-preserving loss (NEW)
        if use_anatomy_loss:
            anatomy_total, anatomy_dict = anatomy_loss(denoised, clean, noisy)
            anatomy_loss_val = anatomy_total - anatomy_dict['recon']  # Exclude recon (counted in weighted_recon)
            del anatomy_dict  # MEMORY FIX
        else:
            anatomy_loss_val = torch.tensor(0.0, device=device)

        # Expert diversity loss - extract needed data then delete
        expert_outputs = interpretation['expert_outputs']
        div_loss = expert_diversity_loss(expert_outputs)
        del expert_outputs  # MEMORY FIX

        # Expert usage loss - extract needed data then delete
        noise_type = interpretation['noise_type']
        usage_loss = expert_usage_loss(noise_type)
        del noise_type  # MEMORY FIX

        # MEMORY FIX: Delete interpretation dict and its contents
        del interpretation

        # Total loss
        total = weighted_recon + 0.1 * anatomy_loss_val + 0.1 * div_loss + 0.1 * usage_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics (extract values before deleting tensors)
        total_loss += total.item()
        total_recon += weighted_recon.item()
        total_anatomy += anatomy_loss_val.item() if isinstance(anatomy_loss_val, torch.Tensor) else 0
        total_diversity += div_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{total.item():.4f}',
            'Recon': f'{weighted_recon.item():.4f}',
        })

        # MEMORY FIX: Delete remaining tensors and clear cache periodically
        del noisy, clean, denoised, backbone_out, weighted_recon, anatomy_loss_val, div_loss, usage_loss, total
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'anatomy': total_anatomy / num_batches,
        'diversity': total_diversity / num_batches,
    }


def validate(model, val_loader, device, epoch, base_model, alpha=2.0):
    """Validate the model with comparison to FROZEN base NAFNet."""
    model.eval()
    base_model.eval()

    total_psnr_noisy = 0
    total_psnr_base = 0
    total_psnr_denoised = 0
    total_ssim_base = 0
    total_ssim = 0
    total_samples = 0

    # For noise type accuracy
    correct_top1 = 0
    per_class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}

    # For expert usage tracking
    expert_usage = torch.zeros(4)

    # For layer detection tracking
    layer_usage = torch.zeros(5)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val {epoch}")

        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            gt_weights = batch['weights'].to(device)

            # Get base model output (FROZEN - constant baseline)
            base_out = base_model(
                noisy,
                spatial_map=None,
                basis=None,
                alpha=alpha,
                gate=None,
            )

            # Get our model output
            denoised, interpretation = model(noisy, return_interpretation=True)

            # Compute metrics for each sample
            for i in range(noisy.size(0)):
                # PSNR (use tensors directly, not numpy)
                psnr_noisy = compute_psnr(noisy[i:i+1], clean[i:i+1])
                psnr_base = compute_psnr(base_out[i:i+1], clean[i:i+1])
                psnr_denoised = compute_psnr(denoised[i:i+1], clean[i:i+1])

                # SSIM (use tensors directly, not numpy)
                ssim_base = compute_ssim(base_out[i:i+1], clean[i:i+1])
                ssim_denoised = compute_ssim(denoised[i:i+1], clean[i:i+1])

                total_psnr_noisy += psnr_noisy
                total_psnr_base += psnr_base
                total_psnr_denoised += psnr_denoised
                total_ssim_base += ssim_base
                total_ssim += ssim_denoised
                total_samples += 1

                # Noise type accuracy
                gt_dominant = gt_weights[i].argmax().item()
                pred_dominant = interpretation['global_weights'][i].argmax().item()

                per_class_total[gt_dominant] += 1
                if gt_dominant == pred_dominant:
                    correct_top1 += 1
                    per_class_correct[gt_dominant] += 1

            # Track expert usage
            expert_usage += interpretation['noise_type'].mean(dim=[0, 2, 3]).sum(dim=0).cpu()

            # Track layer usage (if available)
            if interpretation.get('layer_prob') is not None:
                layer_usage += interpretation['layer_prob'].mean(dim=[0, 2, 3]).sum(dim=0).cpu()

    # Compute averages
    avg_psnr_noisy = total_psnr_noisy / total_samples
    avg_psnr_base = total_psnr_base / total_samples
    avg_psnr_denoised = total_psnr_denoised / total_samples
    avg_ssim_base = total_ssim_base / total_samples
    avg_ssim = total_ssim / total_samples

    expert_usage = expert_usage / total_samples
    layer_usage = layer_usage / total_samples if layer_usage.sum() > 0 else layer_usage

    top1_acc = 100 * correct_top1 / total_samples

    return {
        'psnr_noisy': avg_psnr_noisy,
        'psnr_base': avg_psnr_base,
        'psnr': avg_psnr_denoised,
        'ssim_base': avg_ssim_base,
        'ssim': avg_ssim,
        'top1_acc': top1_acc,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'expert_usage': expert_usage,
        'layer_usage': layer_usage,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Anatomy-Aware NSAD')

    # Data
    parser.add_argument('--train_jsonl', type=str, default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_train_samples', type=int, default=None)

    # Model
    parser.add_argument('--base_ckpt', type=str, default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--fusion_mode', type=str, default='anatomy',
                        choices=['anatomy', 'gated', 'residual'])
    parser.add_argument('--num_layer_zones', type=int, default=5)
    parser.add_argument('--use_depth_adaptive', action='store_true', default=True)
    parser.add_argument('--use_anatomy_fusion', action='store_true', default=True)
    parser.add_argument('--use_anatomy_loss', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints/anatomy_aware_nsad')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ANATOMY-AWARE NSAD TRAINING")
    print("=" * 70)
    print("\nNOVEL CONTRIBUTIONS:")
    print("  1. Per-pixel noise TYPE decomposition")
    print("  2. Soft routing to MULTIPLE classical operators")
    print("  3. Differentiable mixture of NAMED operators")
    print("  4. Full per-pixel interpretability")
    print("  5. Layer-aware noise decomposition (NEW)")
    print("  6. Depth-adaptive processing (NEW)")
    print("  7. Anatomy-preserving constraints (NEW)")
    print("  8. Anatomically-informed expert routing (NEW)")
    print("=" * 70)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = OCTDenoiseDataset(
        args.train_jsonl,
        patch_size=args.patch_size,
        max_samples=args.max_train_samples
    )
    val_dataset = OCTDenoiseDataset(
        args.val_jsonl,
        patch_size=args.patch_size,
        max_samples=50
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating Anatomy-Aware NSAD model...")
    model = AnatomyAwareSANSD(
        backbone_width=64,
        backbone_ckpt=args.base_ckpt,
        fusion_mode=args.fusion_mode,
        num_layer_zones=args.num_layer_zones,
        use_depth_adaptive=args.use_depth_adaptive,
        use_anatomy_fusion=args.use_anatomy_fusion,
    ).to(args.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create frozen base model for comparison
    print("\nLoading FROZEN baseline NAFNet for fair comparison...")
    base_model = NAFNetFullFiLM(
        img_channel=1,
        width=64,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        middle_blk_num=2,
        cond_dim=32,
    ).to(args.device)

    if os.path.exists(args.base_ckpt):
        checkpoint = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
        base_model.load_state_dict(state, strict=False)
        print(f"Loaded baseline from {args.base_ckpt}")

    # Freeze baseline
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    # Create loss functions
    anatomy_loss = AnatomyPreservingLoss(
        lambda_edge=0.1,
        lambda_structure=0.05,
        lambda_intensity=0.02,
    ).to(args.device)

    # Setup optimizer with differential learning rates
    print("\nSetting up optimizer with differential learning rates...")
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.01},  # Very low for backbone
        {'params': other_params, 'lr': args.lr},  # Normal for other modules
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_psnr = 0
    best_gain = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 70}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, anatomy_loss,
            args.device, epoch, use_anatomy_loss=args.use_anatomy_loss
        )

        print(f"\nTraining Metrics:")
        print(f"  Recon Loss: {train_metrics['recon']:.4f}")
        print(f"  Anatomy Loss: {train_metrics['anatomy']:.4f}")
        print(f"  Diversity: {train_metrics['diversity']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, args.device, epoch, base_model, args.alpha)

        print(f"\nValidation Metrics:")
        print(f"  PSNR (noisy):      {val_metrics['psnr_noisy']:.2f} dB")
        print(f"  PSNR (base):       {val_metrics['psnr_base']:.2f} dB  <- FROZEN baseline")
        print(f"  PSNR (NSAD):       {val_metrics['psnr']:.2f} dB  <- your method")
        print(f"")
        print(f"  Gain over noisy: +{val_metrics['psnr'] - val_metrics['psnr_noisy']:.2f} dB")
        print(f"  Gain over base:  +{val_metrics['psnr'] - val_metrics['psnr_base']:.2f} dB  <- KEY METRIC")
        print(f"")
        print(f"  SSIM (base):       {val_metrics['ssim_base']:.4f}")
        print(f"  SSIM (NSAD):       {val_metrics['ssim']:.4f}  (delta: +{val_metrics['ssim'] - val_metrics['ssim_base']:.4f})")
        print(f"")
        print(f"  Top-1 Accuracy:    {val_metrics['top1_acc']:.1f}%")

        # Expert usage
        print(f"\n  Expert Usage (per-pixel routing):")
        expert_names = ['speckle', 'banding', 'gaussian', 'shot']
        for i, name in enumerate(expert_names):
            bar = '#' * int(val_metrics['expert_usage'][i].item() * 40)
            print(f"    {name:8s}: {val_metrics['expert_usage'][i].item():.2f} {bar}")

        # Layer usage
        if val_metrics['layer_usage'].sum() > 0:
            print(f"\n  Layer Zone Usage (anatomy detection):")
            layer_names = ['vitreous_nfl', 'inner_retina', 'outer_nuclear', 'photoreceptors', 'rpe_choroid']
            for i, name in enumerate(layer_names):
                bar = '#' * int(val_metrics['layer_usage'][i].item() * 40)
                print(f"    {name:14s}: {val_metrics['layer_usage'][i].item():.2f} {bar}")

        # Save best model
        gain = val_metrics['psnr'] - val_metrics['psnr_base']
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            best_gain = gain
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_metrics['psnr'],
                'ssim': val_metrics['ssim'],
                'gain_over_base': gain,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"\n  Best model saved! PSNR: {val_metrics['psnr']:.2f} dB (+{gain:.2f} dB over base)")

        # Update scheduler
        scheduler.step()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best PSNR (NSAD):       {best_psnr:.2f} dB")
    print(f"Best Gain over Base:    +{best_gain:.2f} dB")
    print(f"Checkpoint:             {os.path.join(args.output_dir, 'best_model.pth')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
