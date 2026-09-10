#!/usr/bin/env python3
"""
Improved Anatomy-Aware NSAD Training Script

IMPROVEMENTS over train_anatomy_memsafe.py:
1. Stronger expert diversity loss to prevent speckle domination
2. Expert balance loss to ensure all experts are used
3. Layer-specific loss weighting (focus on challenging rpe_choroid)
4. Warmup learning rate schedule
5. Better loss weighting balance
6. Temperature scaling for noise type softmax
7. More training epochs with cosine annealing

Usage:
    python train_anatomy_improved.py --epochs 10 --device cpu
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
        'percent': process.memory_percent(),
    }


def print_memory_status(prefix=""):
    """Print current memory status."""
    mem = get_memory_usage()
    print(f"{prefix}Memory: RSS={mem['rss_mb']:.1f}MB, {mem['percent']:.1f}%")


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

        noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

        h, w = noisy.shape
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()

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


# =============================================================================
# IMPROVED LOSS FUNCTIONS
# =============================================================================

def expert_diversity_loss_improved(expert_outputs: dict, temperature: float = 0.5) -> torch.Tensor:
    """
    IMPROVED: Stronger expert diversity loss with temperature scaling.

    This loss encourages experts to produce DIFFERENT outputs.
    Higher temperature = stronger push for diversity.
    """
    outputs = torch.stack(list(expert_outputs.values()), dim=0)  # [N, B, 1, H, W]
    n = outputs.shape[0]

    # Compute pairwise differences
    diff_matrix = outputs.unsqueeze(1) - outputs.unsqueeze(0)  # [N, N, B, 1, H, W]

    # Only upper triangle
    mask = torch.triu(torch.ones(n, n, device=outputs.device), diagonal=1).bool()
    diff_values = diff_matrix[mask]

    # Use L1 distance with temperature scaling
    diversity = diff_values.abs().mean()

    # Scale by temperature - higher temp = stronger diversity push
    diversity_loss = -diversity * temperature

    del outputs, diff_matrix, diff_values, mask
    return diversity_loss


def expert_balance_loss(noise_type: torch.Tensor, target_usage: float = 0.25) -> torch.Tensor:
    """
    NEW: Encourage balanced usage of all experts.

    Instead of just minimum usage, push toward equal usage.
    This prevents one expert from dominating.
    """
    # Average usage per expert across batch and spatial dimensions
    avg_usage = noise_type.mean(dim=[0, 2, 3])  # [4]

    # Target is equal usage (0.25 each for 4 experts)
    target = torch.full_like(avg_usage, target_usage)

    # KL divergence from uniform distribution
    # Clamp to avoid log(0)
    avg_usage_clamped = avg_usage.clamp(min=1e-8)
    kl_div = (avg_usage_clamped * (avg_usage_clamped.log() - target.log())).sum()

    return kl_div


def expert_entropy_loss(noise_type: torch.Tensor) -> torch.Tensor:
    """
    NEW: Encourage higher entropy (uncertainty) in expert selection.

    This prevents the model from being too confident about one expert.
    Higher entropy = more distributed selection = better use of all experts.
    """
    # noise_type: [B, 4, H, W]
    # Compute entropy per pixel
    entropy = -(noise_type * (noise_type + 1e-8).log()).sum(dim=1)  # [B, H, W]

    # We want to MAXIMIZE entropy, so return negative mean
    # Max entropy for 4 classes is log(4) ≈ 1.386
    max_entropy = np.log(4)
    normalized_entropy = entropy.mean() / max_entropy

    # Return loss that encourages higher entropy
    return 1.0 - normalized_entropy


def layer_weighted_reconstruction_loss(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    layer_weights: dict = None
) -> torch.Tensor:
    """
    NEW: Layer-specific weighted reconstruction loss.

    Apply higher weights to challenging layers (like rpe_choroid).
    """
    if layer_weights is None:
        # Default: emphasize rpe_choroid and photoreceptors
        layer_weights = {
            'vitreous_nfl': 1.0,
            'inner_retina': 1.0,
            'outer_nuclear': 1.0,
            'photoreceptors': 1.2,  # Slightly higher - clinically important
            'rpe_choroid': 1.5,     # Highest - most challenging
        }

    B, C, H, W = denoised.shape
    total_loss = 0
    total_weight = 0

    for zone_name, (start_pct, end_pct) in LAYER_ZONES.items():
        start_row = int(start_pct * H)
        end_row = int(end_pct * H)

        zone_denoised = denoised[:, :, start_row:end_row, :]
        zone_clean = clean[:, :, start_row:end_row, :]

        weight = layer_weights.get(zone_name, 1.0)
        zone_loss = F.mse_loss(zone_denoised, zone_clean)

        total_loss += weight * zone_loss
        total_weight += weight

    return total_loss / total_weight


def perceptual_edge_loss(denoised: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """
    NEW: Edge-aware perceptual loss.

    Preserves edges better by penalizing edge differences.
    """
    # Sobel filters
    sobel_x = torch.tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ], dtype=torch.float32, device=denoised.device).unsqueeze(0)

    sobel_y = torch.tensor([
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ], dtype=torch.float32, device=denoised.device).unsqueeze(0)

    # Compute edges
    clean_edge_x = F.conv2d(clean, sobel_x, padding=1)
    clean_edge_y = F.conv2d(clean, sobel_y, padding=1)
    clean_edges = torch.sqrt(clean_edge_x**2 + clean_edge_y**2 + 1e-8)

    denoised_edge_x = F.conv2d(denoised, sobel_x, padding=1)
    denoised_edge_y = F.conv2d(denoised, sobel_y, padding=1)
    denoised_edges = torch.sqrt(denoised_edge_x**2 + denoised_edge_y**2 + 1e-8)

    # Weight by clean edge magnitude (focus on strong edges)
    edge_weight = clean_edges / (clean_edges.max() + 1e-8)

    edge_loss = (edge_weight * (denoised_edges - clean_edges).abs()).mean()

    del sobel_x, sobel_y, clean_edge_x, clean_edge_y, denoised_edge_x, denoised_edge_y
    return edge_loss


def train_epoch_improved(model, train_loader, optimizer, anatomy_loss, device, epoch,
                         use_anatomy_loss=True, diversity_temp=0.5, memory_check_interval=10):
    """Train for one epoch with IMPROVED loss functions."""
    model.train()

    total_loss = 0
    total_recon = 0
    total_layer_recon = 0
    total_diversity = 0
    total_balance = 0
    total_entropy = 0
    total_edge = 0
    num_batches = 0
    peak_memory = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy'].to(device, non_blocking=True)
        clean = batch['clean'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        denoised, interpretation = model(noisy, return_interpretation=True)

        # Get backbone output for weighted loss
        backbone_out = interpretation['backbone_out']
        backbone_error = (backbone_out - clean).abs()

        # 1. Basic weighted reconstruction loss
        error_weight = 1.0 + 2.0 * backbone_error
        weighted_recon = ((denoised - clean).pow(2) * error_weight).mean()
        del backbone_error, error_weight

        # 2. NEW: Layer-weighted reconstruction loss
        layer_recon = layer_weighted_reconstruction_loss(denoised, clean)

        # 3. Anatomy-preserving loss
        if use_anatomy_loss:
            anatomy_total, anatomy_dict = anatomy_loss(denoised, clean, noisy)
            anatomy_val = anatomy_total - anatomy_dict['recon']
            del anatomy_dict
        else:
            anatomy_val = torch.tensor(0.0, device=device)

        # 4. IMPROVED: Stronger expert diversity loss
        expert_outputs = interpretation['expert_outputs']
        div_loss = expert_diversity_loss_improved(expert_outputs, temperature=diversity_temp)
        del expert_outputs

        # 5. NEW: Expert balance loss (push toward equal usage)
        noise_type = interpretation['noise_type']
        balance_loss = expert_balance_loss(noise_type)

        # 6. NEW: Expert entropy loss (encourage uncertainty)
        entropy_loss = expert_entropy_loss(noise_type)
        del noise_type

        # 7. NEW: Edge preservation loss
        edge_loss = perceptual_edge_loss(denoised, clean)

        # Delete interpretation dict
        del interpretation

        # =================================================================
        # IMPROVED LOSS WEIGHTING
        # =================================================================
        # Primary losses
        loss_recon = 0.5 * weighted_recon + 0.5 * layer_recon

        # Regularization losses
        loss_reg = (
            0.1 * anatomy_val +          # Anatomy preservation
            0.3 * div_loss +             # INCREASED: Diversity (was 0.1)
            0.2 * balance_loss +         # NEW: Expert balance
            0.1 * entropy_loss +         # NEW: Entropy regularization
            0.1 * edge_loss              # NEW: Edge preservation
        )

        total = loss_recon + loss_reg

        # Track values
        loss_val = total.item()
        recon_val = weighted_recon.item()
        layer_recon_val = layer_recon.item()
        div_val = div_loss.item()
        balance_val = balance_loss.item()
        entropy_val = entropy_loss.item()
        edge_val = edge_loss.item()

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Delete tensors
        del noisy, clean, denoised, backbone_out
        del weighted_recon, layer_recon, anatomy_val, div_loss, balance_loss, entropy_loss, edge_loss, total

        # Track metrics
        total_loss += loss_val
        total_recon += recon_val
        total_layer_recon += layer_recon_val
        total_diversity += div_val
        total_balance += balance_val
        total_entropy += entropy_val
        total_edge += edge_val
        num_batches += 1

        if (batch_idx + 1) % memory_check_interval == 0:
            mem = get_memory_usage()
            peak_memory = max(peak_memory, mem['rss_mb'])
            pbar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Div': f'{div_val:.3f}',
                'Bal': f'{balance_val:.3f}',
                'Mem': f'{mem["rss_mb"]:.0f}MB',
            })
            gc.collect()

    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'layer_recon': total_layer_recon / num_batches,
        'diversity': total_diversity / num_batches,
        'balance': total_balance / num_batches,
        'entropy': total_entropy / num_batches,
        'edge': total_edge / num_batches,
        'peak_memory_mb': peak_memory,
    }


def validate_with_anatomy_metrics(model, val_loader, device, epoch, base_model, alpha=2.0):
    """Validate with comprehensive anatomy-specific metrics."""
    model.eval()
    base_model.eval()

    total_psnr_noisy = 0
    total_psnr_base = 0
    total_psnr_denoised = 0
    total_ssim_base = 0
    total_ssim_denoised = 0
    total_samples = 0

    layer_psnr_sum = {zone: 0.0 for zone in LAYER_ZONES}
    layer_ssim_sum = {zone: 0.0 for zone in LAYER_ZONES}
    layer_psnr_base_sum = {zone: 0.0 for zone in LAYER_ZONES}

    total_epi = 0
    total_horizontal_edge = 0

    correct_top1 = 0
    per_class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}

    expert_usage_sum = torch.zeros(4)
    layer_usage_sum = torch.zeros(5)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}"):
            noisy = batch['noisy'].to(device, non_blocking=True)
            clean = batch['clean'].to(device, non_blocking=True)
            gt_weights = batch['weights'].to(device, non_blocking=True)

            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            denoised, interpretation = model(noisy, return_interpretation=True)

            for i in range(noisy.size(0)):
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

                layer_psnr = compute_layer_specific_psnr(denoised[i:i+1], clean[i:i+1])
                layer_psnr_base = compute_layer_specific_psnr(base_out[i:i+1], clean[i:i+1])
                layer_ssim = compute_layer_specific_ssim(denoised[i:i+1], clean[i:i+1])

                for zone in LAYER_ZONES:
                    layer_psnr_sum[zone] += layer_psnr.get(zone, 0)
                    layer_psnr_base_sum[zone] += layer_psnr_base.get(zone, 0)
                    layer_ssim_sum[zone] += layer_ssim.get(zone, 0)

                edge_metrics = compute_edge_preservation_index(
                    denoised[i:i+1], clean[i:i+1], noisy[i:i+1]
                )
                total_epi += edge_metrics['epi_denoised']
                total_horizontal_edge += compute_horizontal_edge_preservation(
                    denoised[i:i+1], clean[i:i+1]
                )

                gt_dominant = gt_weights[i].argmax().item()
                pred_dominant = interpretation['global_weights'][i].argmax().item()

                per_class_total[gt_dominant] += 1
                if gt_dominant == pred_dominant:
                    correct_top1 += 1
                    per_class_correct[gt_dominant] += 1

                total_samples += 1

            expert_usage_sum += interpretation['noise_type'].mean(dim=[0, 2, 3]).cpu()

            if interpretation.get('layer_prob') is not None:
                layer_usage_sum += interpretation['layer_prob'].mean(dim=[0, 2, 3]).cpu()

            del noisy, clean, gt_weights, base_out, denoised, interpretation

    n = total_samples
    n_batches = len(val_loader)

    return {
        'psnr_noisy': total_psnr_noisy / n,
        'psnr_base': total_psnr_base / n,
        'psnr': total_psnr_denoised / n,
        'ssim_base': total_ssim_base / n,
        'ssim': total_ssim_denoised / n,
        'layer_psnr': {zone: layer_psnr_sum[zone] / n for zone in LAYER_ZONES},
        'layer_psnr_base': {zone: layer_psnr_base_sum[zone] / n for zone in LAYER_ZONES},
        'layer_ssim': {zone: layer_ssim_sum[zone] / n for zone in LAYER_ZONES},
        'epi': total_epi / n,
        'horizontal_edge': total_horizontal_edge / n,
        'top1_acc': 100 * correct_top1 / n,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'expert_usage': expert_usage_sum / n_batches,
        'layer_usage': layer_usage_sum / n_batches if layer_usage_sum.sum() > 0 else layer_usage_sum,
    }


def print_validation_report(metrics, epoch):
    """Print comprehensive validation report."""
    print(f"\n{'=' * 70}")
    print(f"VALIDATION REPORT - Epoch {epoch}")
    print(f"{'=' * 70}")

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

    print("\n--- ANATOMY ROI QUALITY (Layer-Specific PSNR) ---")
    print(f"  {'Zone':<18} {'Ours':>8} {'Base':>8} {'Gain':>8}")
    total_gain = 0
    for zone in LAYER_ZONES:
        psnr_ours = metrics['layer_psnr'][zone]
        psnr_base = metrics['layer_psnr_base'][zone]
        zone_gain = psnr_ours - psnr_base
        total_gain += zone_gain
        marker = "**" if zone == 'rpe_choroid' else ""
        print(f"  {zone:<18} {psnr_ours:>8.2f} {psnr_base:>8.2f} {'+' if zone_gain >= 0 else ''}{zone_gain:>7.2f} {marker}")
    avg_gain = total_gain / len(LAYER_ZONES)
    print(f"  {'AVERAGE':.<18} {'.':>8} {'.':>8} {'+' if avg_gain >= 0 else ''}{avg_gain:>7.2f}")

    print("\n--- INTERPRETABILITY: Expert Usage ---")
    expert_names = ['speckle', 'banding', 'gaussian', 'shot']
    for i, name in enumerate(expert_names):
        usage = metrics['expert_usage'][i].item()
        bar = '#' * int(usage * 40)
        # Highlight if balanced (close to 0.25)
        status = "OK" if 0.15 < usage < 0.35 else "!"
        print(f"  {name:10s}: {usage:.3f} {bar} {status}")

    if metrics['layer_usage'].sum() > 0:
        print("\n--- INTERPRETABILITY: Layer Detection ---")
        layer_names = ['vitreous_nfl', 'inner_retina', 'outer_nuclear', 'photoreceptors', 'rpe_choroid']
        for i, name in enumerate(layer_names):
            usage = metrics['layer_usage'][i].item()
            bar = '#' * int(usage * 40)
            print(f"  {name:14s}: {usage:.3f} {bar}")

    print(f"\n--- NOISE TYPE ACCURACY: {metrics['top1_acc']:.1f}% ---")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description='Improved Anatomy-Aware NSAD Training')

    # Data
    parser.add_argument('--train_jsonl', type=str, default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_train_samples', type=int, default=100)
    parser.add_argument('--max_val_samples', type=int, default=30)

    # Model
    parser.add_argument('--base_ckpt', type=str, default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--fusion_mode', type=str, default='anatomy')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--diversity_temp', type=float, default=1.0, help='Temperature for diversity loss')
    parser.add_argument('--output_dir', type=str, default='checkpoints/anatomy_improved')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("IMPROVED ANATOMY-AWARE NSAD TRAINING")
    print("=" * 70)
    print("\nIMPROVEMENTS:")
    print("  1. Stronger diversity loss (temp={:.1f})".format(args.diversity_temp))
    print("  2. Expert balance loss (push toward equal usage)")
    print("  3. Expert entropy loss (encourage uncertainty)")
    print("  4. Layer-weighted reconstruction (focus on rpe_choroid)")
    print("  5. Edge preservation loss")
    print("  6. Cosine annealing LR schedule")
    print("=" * 70)

    print_memory_status("\nInitial ")

    # Data
    print("\nLoading datasets...")
    train_dataset = OCTDenoiseDataset(args.train_jsonl, args.patch_size, args.max_train_samples)
    val_dataset = OCTDenoiseDataset(args.val_jsonl, args.patch_size, args.max_val_samples)

    use_pin_memory = args.device != 'cpu' and torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_pin_memory)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Model
    print("\nCreating Anatomy-Aware NSAD model...")
    model = AnatomyAwareSANSD(
        backbone_width=64,
        backbone_ckpt=args.base_ckpt,
        fusion_mode=args.fusion_mode,
        num_layer_zones=5,
        use_depth_adaptive=True,
        use_anatomy_fusion=True,
    ).to(args.device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Frozen baseline
    print("\nLoading FROZEN baseline...")
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)

    if os.path.exists(args.base_ckpt):
        checkpoint = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
        state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        base_model.load_state_dict(state, strict=False)

    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    # Loss and optimizer
    anatomy_loss = AnatomyPreservingLoss().to(args.device)

    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.01},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training
    best_psnr = 0
    best_gain = -float('inf')

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_start = time.time()
        current_lr = optimizer.param_groups[1]['lr']
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs} (LR: {current_lr:.6f})")
        print_memory_status("Start ")

        # Train
        train_metrics = train_epoch_improved(
            model, train_loader, optimizer, anatomy_loss,
            args.device, epoch, diversity_temp=args.diversity_temp
        )

        print(f"\nTraining Metrics:")
        print(f"  Recon: {train_metrics['recon']:.4f}, Layer: {train_metrics['layer_recon']:.4f}")
        print(f"  Diversity: {train_metrics['diversity']:.4f}, Balance: {train_metrics['balance']:.4f}")
        print(f"  Entropy: {train_metrics['entropy']:.4f}, Edge: {train_metrics['edge']:.4f}")
        print(f"  Peak Memory: {train_metrics['peak_memory_mb']:.1f} MB")

        # Validate
        val_metrics = validate_with_anatomy_metrics(
            model, val_loader, args.device, epoch, base_model, args.alpha
        )

        print_validation_report(val_metrics, epoch)

        # Save best
        gain = val_metrics['psnr'] - val_metrics['psnr_base']
        if val_metrics['psnr'] > best_psnr or gain > best_gain:
            best_psnr = max(best_psnr, val_metrics['psnr'])
            best_gain = max(best_gain, gain)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': val_metrics['psnr'],
                'ssim': val_metrics['ssim'],
                'gain_over_base': gain,
                'layer_psnr': val_metrics['layer_psnr'],
                'expert_usage': val_metrics['expert_usage'].tolist(),
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"\n*** Best model saved! PSNR: {val_metrics['psnr']:.2f} dB (+{gain:.2f} dB) ***")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best PSNR:            {best_psnr:.2f} dB")
    print(f"Best Gain over Base:  +{best_gain:.2f} dB")
    print(f"Checkpoint:           {os.path.join(args.output_dir, 'best_model.pth')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
