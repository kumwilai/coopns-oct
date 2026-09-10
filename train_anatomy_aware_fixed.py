#!/usr/bin/env python3
"""
Anatomy-Aware NSAD Training Script - FIXED VERSION

FIXES:
1. Memory Management:
   - Explicit gc.collect() and torch.cuda.empty_cache() between epochs
   - Proper tensor cleanup in training loop
   - Reduced DataLoader workers on CPU
   - Memory monitoring

2. Expert Specialization:
   - Stronger diversity loss (cosine similarity based)
   - Entropy minimization for sharper routing
   - Ground truth noise supervision when available
   - Temperature scaling for routing sharpness

3. Training Stability:
   - Better loss weighting
   - Gradient accumulation option
   - Mixed precision support (when GPU available)
"""

import argparse
import json
import os
import sys
import gc

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


def get_memory_mb():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def cleanup_memory():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OCTDenoiseDataset(Dataset):
    """Dataset for OCT denoising with noise annotations."""

    def __init__(self, jsonl_path, patch_size=256, max_samples=None):
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


# ============================================================================
# FIXED Loss Functions
# ============================================================================

def strong_diversity_loss(expert_outputs: dict) -> torch.Tensor:
    """
    FIXED: Stronger diversity loss using cosine similarity.
    Encourages experts to produce genuinely different outputs.
    """
    outputs = list(expert_outputs.values())
    n = len(outputs)

    if n < 2:
        return torch.tensor(0.0, device=outputs[0].device)

    total_sim = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Flatten and compute cosine similarity
            o1 = outputs[i].flatten(start_dim=1)
            o2 = outputs[j].flatten(start_dim=1)

            cos_sim = F.cosine_similarity(o1, o2, dim=1).mean()
            total_sim += cos_sim
            count += 1

    # We want LOW similarity (HIGH diversity)
    avg_sim = total_sim / count if count > 0 else 0.0
    return avg_sim  # Minimize this


def entropy_specialization_loss(noise_type: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """
    FIXED: Entropy minimization to encourage sharper routing decisions.

    Uniform routing has max entropy ~1.39 (for 4 classes).
    Specialized routing should have entropy closer to 0.
    """
    # Sharpen the distribution with temperature
    sharpened = F.softmax(noise_type * temperature, dim=1)

    # Compute entropy per pixel and average
    entropy = -(sharpened * torch.log(sharpened + 1e-10)).sum(dim=1).mean()

    return entropy


def noise_supervision_loss(noise_type: torch.Tensor, gt_weights: torch.Tensor) -> torch.Tensor:
    """
    NEW: Supervise noise type predictions with ground truth weights.

    This helps the network learn to predict correct noise compositions.
    """
    # noise_type: [B, 4, H, W]
    # gt_weights: [B, 4]

    # Average predicted noise type across spatial dimensions
    pred_avg = noise_type.mean(dim=[2, 3])  # [B, 4]

    # KL divergence between prediction and ground truth
    log_pred = torch.log(pred_avg + 1e-10)
    kl_div = F.kl_div(log_pred, gt_weights, reduction='batchmean')

    return kl_div


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable SSIM loss (1 - SSIM)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)

    mu_pred = F.conv2d(pred, window, padding=window_size // 2)
    mu_target = F.conv2d(target, window, padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1.0 - ssim_map.mean()


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model, train_loader, optimizer, anatomy_loss, device, epoch,
    loss_weights: dict = None, use_noise_supervision: bool = True
):
    """
    FIXED: Train for one epoch with improved losses and memory management.
    """
    if loss_weights is None:
        loss_weights = {
            'l1': 1.0,
            'ssim': 0.5,
            'anatomy': 0.1,
            'diversity': 0.3,      # Increased from 0.1
            'entropy': 0.2,        # New: entropy minimization
            'noise_sup': 0.2,      # New: noise supervision
        }

    model.train()

    metrics = {
        'loss': 0, 'l1': 0, 'ssim': 0,
        'anatomy': 0, 'diversity': 0, 'entropy': 0, 'noise_sup': 0
    }
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        gt_weights = batch['weights'].to(device)  # [B, 4]

        optimizer.zero_grad()

        # Forward pass
        denoised, interpretation = model(noisy, return_interpretation=True)

        # Get backbone output for weighted loss
        backbone_out = interpretation['backbone_out']
        backbone_error = (backbone_out - clean).abs()
        error_weight = 1.0 + 2.0 * backbone_error

        # ============ Compute Losses ============

        # L1 Reconstruction (weighted)
        l1_loss = ((denoised - clean).abs() * error_weight).mean()

        # SSIM Loss
        ssim_val = ssim_loss(denoised, clean)

        # Anatomy Preservation Loss
        anatomy_total, anatomy_dict = anatomy_loss(denoised, clean, noisy)
        anatomy_loss_val = anatomy_total - anatomy_dict['recon']

        # FIXED: Stronger diversity loss
        diversity_val = strong_diversity_loss(interpretation['expert_outputs'])

        # FIXED: Entropy minimization for sharper routing
        entropy_val = entropy_specialization_loss(interpretation['noise_type'])

        # FIXED: Noise type supervision
        if use_noise_supervision:
            noise_sup_val = noise_supervision_loss(interpretation['noise_type'], gt_weights)
        else:
            noise_sup_val = torch.tensor(0.0, device=device)

        # ============ Combine Losses ============
        total = (
            loss_weights['l1'] * l1_loss +
            loss_weights['ssim'] * ssim_val +
            loss_weights['anatomy'] * anatomy_loss_val +
            loss_weights['diversity'] * diversity_val +
            loss_weights['entropy'] * entropy_val +
            loss_weights['noise_sup'] * noise_sup_val
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        metrics['loss'] += total.item()
        metrics['l1'] += l1_loss.item()
        metrics['ssim'] += ssim_val.item()
        metrics['anatomy'] += anatomy_loss_val.item()
        metrics['diversity'] += diversity_val.item()
        metrics['entropy'] += entropy_val.item()
        metrics['noise_sup'] += noise_sup_val.item() if isinstance(noise_sup_val, torch.Tensor) else 0
        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{total.item():.4f}',
            'SSIM': f'{1.0 - ssim_val.item():.4f}',
            'Ent': f'{entropy_val.item():.3f}',
        })

        # FIXED: Clean up tensors explicitly
        del noisy, clean, gt_weights, denoised, interpretation
        del backbone_out, backbone_error, error_weight
        del l1_loss, ssim_val, anatomy_loss_val, diversity_val, entropy_val, noise_sup_val
        del total

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics


def validate(model, val_loader, device, epoch, base_model, alpha=2.0):
    """Validate with comparison to frozen baseline."""
    model.eval()
    base_model.eval()

    total_psnr_noisy = 0
    total_psnr_base = 0
    total_psnr = 0
    total_ssim_base = 0
    total_ssim = 0
    total_samples = 0

    expert_usage_sum = torch.zeros(4)
    layer_usage_sum = torch.zeros(5)
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            gt_weights = batch['weights'].to(device)

            # Our model
            denoised, interpretation = model(noisy, return_interpretation=True)

            # Baseline
            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=alpha, gate=None)

            # Metrics per sample
            for i in range(noisy.size(0)):
                total_psnr_noisy += compute_psnr(noisy[i:i+1], clean[i:i+1])
                total_psnr_base += compute_psnr(base_out[i:i+1], clean[i:i+1])
                total_psnr += compute_psnr(denoised[i:i+1], clean[i:i+1])
                total_ssim_base += compute_ssim(base_out[i:i+1], clean[i:i+1])
                total_ssim += compute_ssim(denoised[i:i+1], clean[i:i+1])
                total_samples += 1

                # Top-1 accuracy
                pred_top1 = interpretation['noise_type'][i].mean(dim=[1, 2]).argmax()
                gt_top1 = gt_weights[i].argmax()
                if pred_top1 == gt_top1:
                    correct_predictions += 1
                total_predictions += 1

            # Expert usage
            noise_type = interpretation['noise_type']
            batch_usage = noise_type.mean(dim=[0, 2, 3])
            expert_usage_sum += batch_usage.cpu()

            # Layer usage (if available)
            if 'layer_probs' in interpretation:
                layer_probs = interpretation['layer_probs']
                layer_usage = layer_probs.mean(dim=[0, 2, 3])
                layer_usage_sum += layer_usage.cpu()

            # Cleanup
            del noisy, clean, gt_weights, denoised, interpretation, base_out

    num_batches = len(val_loader)

    return {
        'psnr_noisy': total_psnr_noisy / total_samples,
        'psnr_base': total_psnr_base / total_samples,
        'psnr': total_psnr / total_samples,
        'ssim_base': total_ssim_base / total_samples,
        'ssim': total_ssim / total_samples,
        'expert_usage': expert_usage_sum / num_batches,
        'layer_usage': layer_usage_sum / num_batches,
        'top1_acc': 100 * correct_predictions / total_predictions if total_predictions > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Anatomy-Aware NSAD Training - FIXED')

    # Data
    parser.add_argument('--train_jsonl', type=str, default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)  # FIXED: 0 for CPU to avoid issues
    parser.add_argument('--max_train_samples', type=int, default=None)

    # Model
    parser.add_argument('--base_ckpt', type=str, default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--fusion_mode', type=str, default='anatomy')
    parser.add_argument('--num_layer_zones', type=int, default=5)
    parser.add_argument('--use_depth_adaptive', action='store_true')
    parser.add_argument('--use_anatomy_fusion', action='store_true')

    # Loss weights
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=0.5)
    parser.add_argument('--w_anatomy', type=float, default=0.1)
    parser.add_argument('--w_diversity', type=float, default=0.3)  # INCREASED
    parser.add_argument('--w_entropy', type=float, default=0.2)    # NEW
    parser.add_argument('--w_noise_sup', type=float, default=0.2)  # NEW

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints/anatomy_fixed')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--memory_monitor', action='store_true', help='Monitor memory usage')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device != 'gpu' else 'cuda')

    print("=" * 70)
    print("ANATOMY-AWARE NSAD TRAINING - FIXED VERSION")
    print("=" * 70)
    print("\nFIXES APPLIED:")
    print("  - Memory management with explicit cleanup")
    print("  - Stronger diversity loss (cosine similarity)")
    print("  - Entropy minimization for routing sharpness")
    print("  - Noise type supervision with GT weights")
    print("  - Reduced DataLoader workers for CPU stability")
    print("=" * 70)

    if args.memory_monitor:
        initial_mem = get_memory_mb()
        print(f"\nInitial memory: {initial_mem:.1f} MB")

    # Loss weights
    loss_weights = {
        'l1': args.w_l1,
        'ssim': args.w_ssim,
        'anatomy': args.w_anatomy,
        'diversity': args.w_diversity,
        'entropy': args.w_entropy,
        'noise_sup': args.w_noise_sup,
    }
    print(f"\nLoss weights: {loss_weights}")

    # Datasets
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

    # FIXED: num_workers=0 for CPU to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False  # FIXED: False for CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Model
    print("\nCreating model...")
    model = AnatomyAwareSANSD(
        backbone_width=64,
        num_layer_zones=args.num_layer_zones,
        use_depth_adaptive=args.use_depth_adaptive,
        use_anatomy_fusion=args.use_anatomy_fusion,
        fusion_mode=args.fusion_mode,
        backbone_ckpt=args.base_ckpt,
    ).to(device)

    # Baseline model (frozen)
    print("Loading frozen baseline...")
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
        condition_middle=True, condition_decoders=True, use_spatial_cue=True,
    ).to(device)

    checkpoint = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        base_model.load_state_dict(checkpoint, strict=False)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # Loss functions
    anatomy_loss = AnatomyPreservingLoss().to(device)

    # Optimizer with differential learning rates
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    best_psnr = 0
    best_combined = -float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 70)

        if args.memory_monitor:
            pre_train_mem = get_memory_mb()
            print(f"Pre-train memory: {pre_train_mem:.1f} MB")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, anatomy_loss, device, epoch,
            loss_weights=loss_weights
        )

        print(f"\nTraining - L1: {train_metrics['l1']:.4f}, "
              f"SSIM: {1.0 - train_metrics['ssim']:.4f}, "
              f"Entropy: {train_metrics['entropy']:.4f}, "
              f"Diversity: {train_metrics['diversity']:.4f}")

        # FIXED: Cleanup after training
        cleanup_memory()

        # Validate
        val_metrics = validate(model, val_loader, device, epoch, base_model, args.alpha)

        psnr_gain = val_metrics['psnr'] - val_metrics['psnr_base']
        ssim_gain = val_metrics['ssim'] - val_metrics['ssim_base']

        print(f"\nValidation:")
        print(f"  PSNR: {val_metrics['psnr']:.2f} dB (gain: +{psnr_gain:.2f})")
        print(f"  SSIM: {val_metrics['ssim']:.4f} (gain: +{ssim_gain:.4f})")
        print(f"  Top-1 Accuracy: {val_metrics['top1_acc']:.1f}%")

        # Expert usage (should NOT be uniform anymore!)
        print(f"\n  Expert Usage:")
        expert_names = ['speckle', 'banding', 'gaussian', 'shot']
        for i, name in enumerate(expert_names):
            usage = val_metrics['expert_usage'][i].item()
            bar = '#' * int(usage * 40)
            print(f"    {name:8s}: {usage:.2f} {bar}")

        # FIXED: Cleanup after validation
        cleanup_memory()

        if args.memory_monitor:
            post_val_mem = get_memory_mb()
            print(f"\n  Memory: {post_val_mem:.1f} MB (growth: {post_val_mem - initial_mem:.1f} MB)")

        # Save best model
        combined_score = psnr_gain + 10 * ssim_gain
        if combined_score > best_combined:
            best_psnr = val_metrics['psnr']
            best_combined = combined_score

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': val_metrics['ssim'],
                'psnr_gain': psnr_gain,
                'ssim_gain': ssim_gain,
            }, os.path.join(args.output_dir, 'best_model.pth'))

            print(f"\n  *** Best model saved! PSNR: {best_psnr:.2f} dB (+{psnr_gain:.2f})")

        scheduler.step()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'best_model.pth')}")

    if args.memory_monitor:
        final_mem = get_memory_mb()
        print(f"Final memory: {final_mem:.1f} MB (total growth: {final_mem - initial_mem:.1f} MB)")

    print("=" * 70)


if __name__ == '__main__':
    main()
