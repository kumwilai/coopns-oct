#!/usr/bin/env python3
"""
Memory-Safe Anatomy-Aware NSAD Training Script

This script trains the AnatomyAwareSANSD model with:
1. Comprehensive memory management to prevent crashes
2. Memory usage monitoring
3. Anatomy ROI quality metrics
4. Overall PSNR/SSIM tracking
5. Interpretability metrics (expert usage, layer detection)

MEMORY OPTIMIZATIONS:
- pin_memory=False for CPU
- num_workers=0 for CPU
- Explicit tensor deletion after use
- Garbage collection between epochs
- Batch size reduced for CPU safety
- Periodic memory cleanup during training

Usage:
    python train_anatomy_memsafe.py --epochs 10 --device cpu
"""

import argparse
import json
import os
import sys
import gc
import psutil
import time

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

from nsnd.models.sansd import AnatomyAwareSANSD
from nsnd.models.anatomy_aware import AnatomyPreservingLoss
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.utils.metrics import compute_psnr, compute_ssim
from nsnd.utils.anatomy_metrics import (
    compute_layer_specific_psnr,
    compute_layer_specific_ssim,
    compute_edge_preservation_index,
    compute_horizontal_edge_preservation,
    LAYER_ZONES,
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
        'percent': process.memory_percent(),
    }


def print_memory_status(prefix=""):
    """Print current memory status."""
    mem = get_memory_usage()
    print(f"{prefix}Memory: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, {mem['percent']:.1f}%")


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

        # Center crop to patch_size
        h, w = noisy.shape
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        # Convert to tensors
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()

        # Ground truth noise weights
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
    """
    Memory-efficient expert diversity loss.
    Encourages experts to produce different outputs.
    """
    # Stack outputs into single tensor
    outputs = torch.stack(list(expert_outputs.values()), dim=0)  # [N, B, 1, H, W]
    n = outputs.shape[0]

    # Compute pairwise differences using broadcasting
    diff_matrix = outputs.unsqueeze(1) - outputs.unsqueeze(0)  # [N, N, B, 1, H, W]

    # Only take upper triangle (i < j)
    mask = torch.triu(torch.ones(n, n, device=outputs.device), diagonal=1).bool()
    diff_values = diff_matrix[mask]

    diversity = diff_values.abs().mean()

    # Clean up
    del outputs, diff_matrix, diff_values, mask

    return -diversity  # Negative because we want to maximize diversity


def expert_usage_loss(noise_type: torch.Tensor, min_usage: float = 0.05) -> torch.Tensor:
    """Encourage using all experts."""
    avg_usage = noise_type.mean(dim=[0, 2, 3])
    underused = F.relu(min_usage - avg_usage)
    return underused.sum()


def train_epoch(model, train_loader, optimizer, anatomy_loss, device, epoch,
                use_anatomy_loss=True, memory_check_interval=10):
    """Train for one epoch with memory management."""
    model.train()

    total_loss = 0
    total_recon = 0
    total_anatomy = 0
    total_diversity = 0
    num_batches = 0
    peak_memory = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy'].to(device, non_blocking=True)
        clean = batch['clean'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Memory efficient

        # Forward pass
        denoised, interpretation = model(noisy, return_interpretation=True)

        # Get backbone output and error
        backbone_out = interpretation['backbone_out']
        backbone_error = (backbone_out - clean).abs()

        # Spatially-weighted loss
        error_weight = 1.0 + 2.0 * backbone_error
        weighted_recon = ((denoised - clean).pow(2) * error_weight).mean()

        # MEMORY FIX: Delete intermediate tensors
        del backbone_error, error_weight

        # Anatomy-preserving loss
        if use_anatomy_loss:
            anatomy_total, anatomy_dict = anatomy_loss(denoised, clean, noisy)
            anatomy_loss_val = anatomy_total - anatomy_dict['recon']
            del anatomy_dict
        else:
            anatomy_loss_val = torch.tensor(0.0, device=device)

        # Expert diversity loss
        expert_outputs = interpretation['expert_outputs']
        div_loss = expert_diversity_loss(expert_outputs)
        del expert_outputs

        # Expert usage loss
        noise_type = interpretation['noise_type']
        usage_loss = expert_usage_loss(noise_type)
        del noise_type

        # MEMORY FIX: Delete interpretation dict
        del interpretation

        # Total loss
        total = weighted_recon + 0.1 * anatomy_loss_val + 0.1 * div_loss + 0.1 * usage_loss

        # Track values BEFORE backward
        loss_val = total.item()
        recon_val = weighted_recon.item()
        anatomy_val = anatomy_loss_val.item() if isinstance(anatomy_loss_val, torch.Tensor) else 0
        div_val = div_loss.item()

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # MEMORY FIX: Delete all tensors
        del noisy, clean, denoised, backbone_out
        del weighted_recon, anatomy_loss_val, div_loss, usage_loss, total

        # Track metrics
        total_loss += loss_val
        total_recon += recon_val
        total_anatomy += anatomy_val
        total_diversity += div_val
        num_batches += 1

        # Memory monitoring
        if (batch_idx + 1) % memory_check_interval == 0:
            mem = get_memory_usage()
            peak_memory = max(peak_memory, mem['rss_mb'])
            pbar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Mem': f'{mem["rss_mb"]:.0f}MB',
            })

            # Force garbage collection periodically
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'anatomy': total_anatomy / num_batches,
        'diversity': total_diversity / num_batches,
        'peak_memory_mb': peak_memory,
    }


def validate_with_anatomy_metrics(model, val_loader, device, epoch, base_model, alpha=2.0):
    """
    Validate with comprehensive anatomy-specific metrics.

    Returns:
        - Global PSNR/SSIM
        - Layer-specific PSNR/SSIM (anatomy ROI quality)
        - Edge preservation metrics
        - Expert usage (interpretability)
        - Layer detection (interpretability)
    """
    model.eval()
    base_model.eval()

    # Global metrics
    total_psnr_noisy = 0
    total_psnr_base = 0
    total_psnr_denoised = 0
    total_ssim_base = 0
    total_ssim_denoised = 0
    total_samples = 0

    # Layer-specific metrics (anatomy ROI quality)
    layer_psnr_sum = {zone: 0.0 for zone in LAYER_ZONES}
    layer_ssim_sum = {zone: 0.0 for zone in LAYER_ZONES}
    layer_psnr_base_sum = {zone: 0.0 for zone in LAYER_ZONES}

    # Edge preservation
    total_epi = 0
    total_horizontal_edge = 0

    # Noise type accuracy (interpretability)
    correct_top1 = 0
    per_class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}

    # Expert usage tracking (interpretability)
    expert_usage_sum = torch.zeros(4)

    # Layer detection tracking (interpretability)
    layer_usage_sum = torch.zeros(5)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val {epoch}")

        for batch in pbar:
            noisy = batch['noisy'].to(device, non_blocking=True)
            clean = batch['clean'].to(device, non_blocking=True)
            gt_weights = batch['weights'].to(device, non_blocking=True)

            # Base model output (FROZEN baseline)
            base_out = base_model(
                noisy,
                spatial_map=None,
                basis=None,
                alpha=0.0,
                gate=None,
            )

            # Our model output
            denoised, interpretation = model(noisy, return_interpretation=True)

            # Process each sample
            for i in range(noisy.size(0)):
                # Global metrics
                psnr_noisy = compute_psnr(noisy[i:i+1], clean[i:i+1])
                psnr_base = compute_psnr(base_out[i:i+1], clean[i:i+1])
                psnr_denoised = compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim_base = compute_ssim(base_out[i:i+1], clean[i:i+1])
                ssim_denoised = compute_ssim(denoised[i:i+1], clean[i:i+1])

                total_psnr_noisy += psnr_noisy
                total_psnr_base += psnr_base
                total_psnr_denoised += psnr_denoised
                total_ssim_base += ssim_base
                total_ssim_denoised += ssim_denoised

                # Layer-specific PSNR (anatomy ROI quality)
                layer_psnr = compute_layer_specific_psnr(denoised[i:i+1], clean[i:i+1])
                layer_psnr_base = compute_layer_specific_psnr(base_out[i:i+1], clean[i:i+1])
                layer_ssim = compute_layer_specific_ssim(denoised[i:i+1], clean[i:i+1])

                for zone in LAYER_ZONES:
                    layer_psnr_sum[zone] += layer_psnr.get(zone, 0)
                    layer_psnr_base_sum[zone] += layer_psnr_base.get(zone, 0)
                    layer_ssim_sum[zone] += layer_ssim.get(zone, 0)

                # Edge preservation
                edge_metrics = compute_edge_preservation_index(
                    denoised[i:i+1], clean[i:i+1], noisy[i:i+1]
                )
                total_epi += edge_metrics['epi_denoised']
                total_horizontal_edge += compute_horizontal_edge_preservation(
                    denoised[i:i+1], clean[i:i+1]
                )

                # Noise type accuracy (interpretability)
                gt_dominant = gt_weights[i].argmax().item()
                pred_dominant = interpretation['global_weights'][i].argmax().item()

                per_class_total[gt_dominant] += 1
                if gt_dominant == pred_dominant:
                    correct_top1 += 1
                    per_class_correct[gt_dominant] += 1

                total_samples += 1

            # Expert usage (interpretability)
            expert_usage_sum += interpretation['noise_type'].mean(dim=[0, 2, 3]).cpu()

            # Layer detection (interpretability)
            if interpretation.get('layer_prob') is not None:
                layer_usage_sum += interpretation['layer_prob'].mean(dim=[0, 2, 3]).cpu()

            # MEMORY FIX: Clean up
            del noisy, clean, gt_weights, base_out, denoised, interpretation

    # Compute averages
    n = total_samples
    n_batches = len(val_loader)

    results = {
        # Global metrics
        'psnr_noisy': total_psnr_noisy / n,
        'psnr_base': total_psnr_base / n,
        'psnr': total_psnr_denoised / n,
        'ssim_base': total_ssim_base / n,
        'ssim': total_ssim_denoised / n,

        # Layer-specific (anatomy ROI quality)
        'layer_psnr': {zone: layer_psnr_sum[zone] / n for zone in LAYER_ZONES},
        'layer_psnr_base': {zone: layer_psnr_base_sum[zone] / n for zone in LAYER_ZONES},
        'layer_ssim': {zone: layer_ssim_sum[zone] / n for zone in LAYER_ZONES},

        # Edge preservation
        'epi': total_epi / n,
        'horizontal_edge': total_horizontal_edge / n,

        # Interpretability
        'top1_acc': 100 * correct_top1 / n,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'expert_usage': expert_usage_sum / n_batches,
        'layer_usage': layer_usage_sum / n_batches if layer_usage_sum.sum() > 0 else layer_usage_sum,
    }

    return results


def print_validation_report(metrics, epoch):
    """Print comprehensive validation report."""
    print(f"\n{'=' * 70}")
    print(f"VALIDATION REPORT - Epoch {epoch}")
    print(f"{'=' * 70}")

    # Global metrics
    print("\n--- GLOBAL METRICS ---")
    print(f"  PSNR (noisy):      {metrics['psnr_noisy']:.2f} dB")
    print(f"  PSNR (base):       {metrics['psnr_base']:.2f} dB  <- FROZEN baseline")
    print(f"  PSNR (ours):       {metrics['psnr']:.2f} dB  <- Anatomy-Aware NSAD")
    gain = metrics['psnr'] - metrics['psnr_base']
    print(f"  GAIN over base:    {'+' if gain >= 0 else ''}{gain:.2f} dB  <- KEY METRIC")
    print(f"")
    print(f"  SSIM (base):       {metrics['ssim_base']:.4f}")
    print(f"  SSIM (ours):       {metrics['ssim']:.4f}")
    ssim_gain = metrics['ssim'] - metrics['ssim_base']
    print(f"  SSIM gain:         {'+' if ssim_gain >= 0 else ''}{ssim_gain:.4f}")

    # Anatomy ROI quality
    print("\n--- ANATOMY ROI QUALITY (Layer-Specific PSNR) ---")
    print(f"  {'Zone':<18} {'Ours':>8} {'Base':>8} {'Gain':>8}")
    total_gain = 0
    for zone in LAYER_ZONES:
        psnr_ours = metrics['layer_psnr'][zone]
        psnr_base = metrics['layer_psnr_base'][zone]
        zone_gain = psnr_ours - psnr_base
        total_gain += zone_gain
        print(f"  {zone:<18} {psnr_ours:>8.2f} {psnr_base:>8.2f} {'+' if zone_gain >= 0 else ''}{zone_gain:>7.2f}")
    avg_gain = total_gain / len(LAYER_ZONES)
    print(f"  {'AVERAGE':.<18} {'.':>8} {'.':>8} {'+' if avg_gain >= 0 else ''}{avg_gain:>7.2f}")

    # Layer-specific SSIM
    print("\n--- ANATOMY ROI QUALITY (Layer-Specific SSIM) ---")
    print(f"  {'Zone':<18} {'SSIM':>8}")
    for zone in LAYER_ZONES:
        ssim_val = metrics['layer_ssim'][zone]
        print(f"  {zone:<18} {ssim_val:>8.4f}")

    # Edge preservation
    print("\n--- EDGE PRESERVATION ---")
    print(f"  Edge Preservation Index:  {metrics['epi']:.4f}")
    print(f"  Horizontal Edges:         {metrics['horizontal_edge']:.4f}")

    # Interpretability - Expert usage
    print("\n--- INTERPRETABILITY: Expert Usage (per-pixel routing) ---")
    expert_names = ['speckle', 'banding', 'gaussian', 'shot']
    for i, name in enumerate(expert_names):
        usage = metrics['expert_usage'][i].item()
        bar = '#' * int(usage * 40)
        print(f"  {name:10s}: {usage:.3f} {bar}")

    # Interpretability - Layer detection
    if metrics['layer_usage'].sum() > 0:
        print("\n--- INTERPRETABILITY: Layer Zone Detection (anatomy) ---")
        layer_names = ['vitreous_nfl', 'inner_retina', 'outer_nuclear', 'photoreceptors', 'rpe_choroid']
        for i, name in enumerate(layer_names):
            usage = metrics['layer_usage'][i].item()
            bar = '#' * int(usage * 40)
            print(f"  {name:14s}: {usage:.3f} {bar}")

    # Interpretability - Noise type accuracy
    print("\n--- INTERPRETABILITY: Noise Type Classification ---")
    print(f"  Top-1 Accuracy: {metrics['top1_acc']:.1f}%")
    noise_names = ['speckle', 'banding', 'gaussian', 'shot']
    for i, name in enumerate(noise_names):
        correct = metrics['per_class_correct'][i]
        total = metrics['per_class_total'][i]
        acc = 100 * correct / total if total > 0 else 0
        print(f"  {name:10s}: {acc:5.1f}% ({correct}/{total})")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description='Memory-Safe Anatomy-Aware NSAD Training')

    # Data
    parser.add_argument('--train_jsonl', type=str, default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size (64x64 for memory safety)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (2 for CPU memory safety)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for CPU)')
    parser.add_argument('--max_train_samples', type=int, default=50, help='Max training samples')
    parser.add_argument('--max_val_samples', type=int, default=20, help='Max validation samples')

    # Model
    parser.add_argument('--base_ckpt', type=str, default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--fusion_mode', type=str, default='anatomy', choices=['anatomy', 'gated', 'residual'])
    parser.add_argument('--use_anatomy_loss', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints/anatomy_memsafe')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("MEMORY-SAFE ANATOMY-AWARE NSAD TRAINING")
    print("=" * 70)
    print("\nMEMORY OPTIMIZATIONS:")
    print(f"  - Batch size: {args.batch_size} (reduced for CPU)")
    print(f"  - Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  - Num workers: {args.num_workers} (0 for CPU)")
    print(f"  - Pin memory: False (CPU mode)")
    print(f"  - Explicit tensor deletion: Enabled")
    print(f"  - GC between epochs: Enabled")
    print("\nMONITORING:")
    print("  - Memory usage tracking")
    print("  - Layer-specific PSNR/SSIM (anatomy ROI)")
    print("  - Edge preservation metrics")
    print("  - Expert usage (interpretability)")
    print("  - Layer detection (interpretability)")
    print("=" * 70)

    # Initial memory check
    print_memory_status("\nInitial ")

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
        max_samples=args.max_val_samples
    )

    # MEMORY FIX: pin_memory=False for CPU
    use_pin_memory = args.device != 'cpu' and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print_memory_status("After data loading ")

    # Create model
    print("\nCreating Anatomy-Aware NSAD model...")
    model = AnatomyAwareSANSD(
        backbone_width=64,
        backbone_ckpt=args.base_ckpt,
        fusion_mode=args.fusion_mode,
        num_layer_zones=5,
        use_depth_adaptive=True,
        use_anatomy_fusion=True,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_memory_status("After model creation ")

    # Create frozen base model
    print("\nLoading FROZEN baseline NAFNet...")
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

    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    # Create loss and optimizer
    anatomy_loss = AnatomyPreservingLoss().to(args.device)

    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.01},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    print_memory_status("Before training ")

    # Training loop
    best_psnr = 0
    best_gain = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        # MEMORY FIX: Force garbage collection between epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_start = time.time()
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print_memory_status("Start of epoch ")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, anatomy_loss,
            args.device, epoch, use_anatomy_loss=args.use_anatomy_loss
        )

        print(f"\nTraining Metrics:")
        print(f"  Recon Loss: {train_metrics['recon']:.4f}")
        print(f"  Anatomy Loss: {train_metrics['anatomy']:.4f}")
        print(f"  Diversity: {train_metrics['diversity']:.4f}")
        print(f"  Peak Memory: {train_metrics['peak_memory_mb']:.1f} MB")

        # Validate with comprehensive metrics
        val_metrics = validate_with_anatomy_metrics(
            model, val_loader, args.device, epoch, base_model, args.alpha
        )

        # Print comprehensive report
        print_validation_report(val_metrics, epoch)

        # Save best model
        gain = val_metrics['psnr'] - val_metrics['psnr_base']
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            best_gain = gain
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': val_metrics['psnr'],
                'ssim': val_metrics['ssim'],
                'gain_over_base': gain,
                'layer_psnr': val_metrics['layer_psnr'],
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"\n*** Best model saved! PSNR: {val_metrics['psnr']:.2f} dB (+{gain:.2f} dB) ***")

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print_memory_status("End of epoch ")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best PSNR:            {best_psnr:.2f} dB")
    print(f"Best Gain over Base:  +{best_gain:.2f} dB")
    print(f"Checkpoint:           {os.path.join(args.output_dir, 'best_model.pth')}")
    print_memory_status("Final ")
    print("=" * 70)


if __name__ == '__main__':
    main()
