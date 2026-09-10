#!/usr/bin/env python3
"""
Anatomy-Aware NSAD Training Script v2.

Improvements over v1:
1. SSIM loss added (important for structural quality)
2. Layer-specific SSIM/PSNR losses
3. Stronger diversity loss for expert specialization
4. Better reporting of layer-specific metrics
5. Tunable loss weights
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.sansd import AnatomyAwareSANSD
from nsnd.models.anatomy_aware import AnatomyPreservingLoss
from nsnd.utils.metrics import compute_psnr, compute_ssim
from nsnd.utils.anatomy_metrics import (
    compute_layer_specific_psnr,
    compute_layer_specific_ssim,
    compute_edge_preservation_index,
    compute_horizontal_edge_preservation,
    LAYER_ZONES,
)


# ============================================================================
# Dataset
# ============================================================================

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


# ============================================================================
# SSIM Loss
# ============================================================================

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable SSIM loss (1 - SSIM)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
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


def layer_specific_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    zone_weights: dict = None
) -> torch.Tensor:
    """
    Compute layer-specific SSIM loss with customizable weights per zone.

    This encourages better preservation of important layers like NFL and photoreceptors.
    """
    if zone_weights is None:
        # Default: emphasize NFL/photoreceptors (clinically important)
        zone_weights = {
            'vitreous_nfl': 2.0,      # NFL thickness is critical
            'inner_retina': 1.0,
            'outer_nuclear': 1.0,
            'photoreceptors': 2.0,    # IS/OS junction important
            'rpe_choroid': 1.5,       # RPE visibility
        }

    B, C, H, W = pred.shape
    total_loss = 0.0
    total_weight = 0.0

    for zone_name, (start_pct, end_pct) in LAYER_ZONES.items():
        start_row = int(start_pct * H)
        end_row = int(end_pct * H)

        if end_row - start_row < 11:  # Too small for SSIM
            continue

        pred_zone = pred[:, :, start_row:end_row, :]
        target_zone = target[:, :, start_row:end_row, :]

        zone_ssim_loss = ssim_loss(pred_zone, target_zone)
        weight = zone_weights.get(zone_name, 1.0)

        total_loss += weight * zone_ssim_loss
        total_weight += weight

    return total_loss / (total_weight + 1e-8)


# ============================================================================
# Expert Diversity Loss (Stronger version)
# ============================================================================

def strong_expert_diversity_loss(expert_outputs: dict) -> torch.Tensor:
    """
    Stronger diversity loss to force experts to specialize.

    Uses cosine distance and variance penalty.
    """
    outputs = list(expert_outputs.values())
    n = len(outputs)

    if n < 2:
        return torch.tensor(0.0, device=outputs[0].device)

    # 1. Pairwise cosine similarity (want low similarity = high diversity)
    cosine_sim_total = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            o1 = outputs[i].flatten(start_dim=1)
            o2 = outputs[j].flatten(start_dim=1)

            # Cosine similarity
            cos_sim = F.cosine_similarity(o1, o2, dim=1).mean()
            cosine_sim_total += cos_sim
            count += 1

    avg_cosine_sim = cosine_sim_total / count if count > 0 else 0.0

    # 2. Variance penalty: encourage different output ranges
    means = torch.stack([o.mean() for o in outputs])
    stds = torch.stack([o.std() for o in outputs])

    # We want different means and different stds
    mean_var = means.var()
    std_var = stds.var()

    # High cosine similarity is bad (want negative loss to encourage diversity)
    # Low variance in means/stds is bad
    diversity_loss = avg_cosine_sim - 0.1 * (mean_var + std_var)

    return diversity_loss


def expert_specialization_loss(noise_type: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """
    Encourage sharper expert routing (more decisive, less uniform).

    Uses entropy minimization with temperature scaling.
    """
    # Sharpen the distribution
    sharpened = F.softmax(noise_type * temperature, dim=1)

    # Entropy (want low entropy = more decisive routing)
    entropy = -(sharpened * torch.log(sharpened + 1e-10)).sum(dim=1).mean()

    return entropy


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model, train_loader, optimizer, anatomy_loss, device, epoch,
    loss_weights: dict = None
):
    """
    Train for one epoch with comprehensive losses.

    Loss components:
    - L1 reconstruction
    - SSIM (global)
    - Layer-specific SSIM
    - Anatomy preservation
    - Expert diversity
    - Expert specialization (entropy)
    """
    if loss_weights is None:
        loss_weights = {
            'l1': 1.0,
            'ssim': 0.5,           # SSIM is important
            'layer_ssim': 0.3,    # Layer-specific SSIM
            'anatomy': 0.1,
            'diversity': 0.2,      # Increased from 0.1
            'specialization': 0.1, # New: entropy minimization
        }

    model.train()

    metrics = {
        'loss': 0, 'l1': 0, 'ssim': 0, 'layer_ssim': 0,
        'anatomy': 0, 'diversity': 0, 'specialization': 0
    }
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad()

        # Forward pass
        denoised, interpretation = model(noisy, return_interpretation=True)

        # Get backbone output for weighted loss
        backbone_out = interpretation['backbone_out']
        backbone_error = (backbone_out - clean).abs()
        error_weight = 1.0 + 2.0 * backbone_error

        # ============ Compute Losses ============

        # L1 Reconstruction (weighted by backbone error)
        l1_loss = ((denoised - clean).abs() * error_weight).mean()

        # MEMORY FIX: Delete intermediate tensors early
        del backbone_error, error_weight

        # Global SSIM Loss
        global_ssim_loss = ssim_loss(denoised, clean)

        # Layer-specific SSIM Loss
        layer_ssim_val = layer_specific_ssim_loss(denoised, clean)

        # Anatomy Preservation Loss
        anatomy_total, anatomy_dict = anatomy_loss(denoised, clean, noisy)
        anatomy_loss_val = anatomy_total - anatomy_dict['recon']
        del anatomy_dict  # MEMORY FIX

        # Expert Diversity Loss - extract then delete
        expert_outputs = interpretation['expert_outputs']
        diversity_loss = strong_expert_diversity_loss(expert_outputs)
        del expert_outputs  # MEMORY FIX

        # Expert Specialization Loss (entropy minimization) - extract then delete
        noise_type = interpretation['noise_type']
        specialization_loss = expert_specialization_loss(noise_type)
        del noise_type  # MEMORY FIX

        # MEMORY FIX: Delete interpretation dict
        del interpretation

        # ============ Combine Losses ============
        total = (
            loss_weights['l1'] * l1_loss +
            loss_weights['ssim'] * global_ssim_loss +
            loss_weights['layer_ssim'] * layer_ssim_val +
            loss_weights['anatomy'] * anatomy_loss_val +
            loss_weights['diversity'] * diversity_loss +
            loss_weights['specialization'] * specialization_loss
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        metrics['loss'] += total.item()
        metrics['l1'] += l1_loss.item()
        metrics['ssim'] += global_ssim_loss.item()
        metrics['layer_ssim'] += layer_ssim_val.item()
        metrics['anatomy'] += anatomy_loss_val.item()
        metrics['diversity'] += diversity_loss.item()
        metrics['specialization'] += specialization_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{total.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'SSIM': f'{1.0 - global_ssim_loss.item():.4f}',
        })

        # MEMORY FIX: Clean up all tensors and clear cache periodically
        del noisy, clean, denoised, backbone_out, l1_loss, global_ssim_loss
        del layer_ssim_val, anatomy_total, anatomy_loss_val, diversity_loss, specialization_loss, total
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics


def validate_with_anatomy_metrics(model, val_loader, device, epoch, base_model=None, alpha=2.0):
    """
    Validation with comprehensive anatomy-specific metrics.
    """
    model.eval()
    if base_model is not None:
        base_model.eval()

    metrics = {
        'global': {'psnr': 0, 'ssim': 0, 'psnr_base': 0, 'ssim_base': 0},
        'layer_psnr': {zone: 0 for zone in LAYER_ZONES.keys()},
        'layer_ssim': {zone: 0 for zone in LAYER_ZONES.keys()},
        'layer_psnr_base': {zone: 0 for zone in LAYER_ZONES.keys()},
        'layer_ssim_base': {zone: 0 for zone in LAYER_ZONES.keys()},
        'edge': {'epi': 0, 'horizontal': 0, 'epi_base': 0, 'horizontal_base': 0},
        'expert_usage': {'speckle': 0, 'banding': 0, 'gaussian': 0, 'shot': 0},
    }
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val {epoch}")):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            # Our model
            denoised, interpretation = model(noisy, return_interpretation=True)

            # Baseline
            if base_model is not None:
                base_out = base_model(noisy, spatial_map=None, basis=None, alpha=alpha, gate=None)
            else:
                base_out = noisy

            # Per-sample metrics
            for i in range(noisy.size(0)):
                pred_i = denoised[i:i+1]
                clean_i = clean[i:i+1]
                noisy_i = noisy[i:i+1]
                base_i = base_out[i:i+1]

                # Global metrics
                metrics['global']['psnr'] += compute_psnr(pred_i, clean_i)
                metrics['global']['ssim'] += compute_ssim(pred_i, clean_i)
                metrics['global']['psnr_base'] += compute_psnr(base_i, clean_i)
                metrics['global']['ssim_base'] += compute_ssim(base_i, clean_i)

                # Layer-specific metrics
                layer_psnr = compute_layer_specific_psnr(pred_i, clean_i)
                layer_ssim = compute_layer_specific_ssim(pred_i, clean_i)
                layer_psnr_base = compute_layer_specific_psnr(base_i, clean_i)
                layer_ssim_base = compute_layer_specific_ssim(base_i, clean_i)

                for zone in LAYER_ZONES.keys():
                    metrics['layer_psnr'][zone] += layer_psnr.get(zone, 0)
                    metrics['layer_ssim'][zone] += layer_ssim.get(zone, 0)
                    metrics['layer_psnr_base'][zone] += layer_psnr_base.get(zone, 0)
                    metrics['layer_ssim_base'][zone] += layer_ssim_base.get(zone, 0)

                # Edge preservation
                edge_metrics = compute_edge_preservation_index(pred_i, clean_i, noisy_i)
                metrics['edge']['epi'] += edge_metrics['epi_denoised']
                metrics['edge']['horizontal'] += compute_horizontal_edge_preservation(pred_i, clean_i)

                if base_model is not None:
                    edge_base = compute_edge_preservation_index(base_i, clean_i, noisy_i)
                    metrics['edge']['epi_base'] += edge_base['epi_denoised']
                    metrics['edge']['horizontal_base'] += compute_horizontal_edge_preservation(base_i, clean_i)

                total_samples += 1

            # Expert usage (for this batch) - extract then delete
            noise_type = interpretation['noise_type']
            batch_usage = noise_type.mean(dim=[0, 2, 3])
            for idx, name in enumerate(['speckle', 'banding', 'gaussian', 'shot']):
                metrics['expert_usage'][name] += batch_usage[idx].item()

            # MEMORY FIX: Clean up all tensors after each batch
            del noisy, clean, denoised, base_out, noise_type, batch_usage, interpretation
            if (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Average
    for key in metrics['global']:
        metrics['global'][key] /= total_samples
    for zone in LAYER_ZONES.keys():
        metrics['layer_psnr'][zone] /= total_samples
        metrics['layer_ssim'][zone] /= total_samples
        metrics['layer_psnr_base'][zone] /= total_samples
        metrics['layer_ssim_base'][zone] /= total_samples
    for key in metrics['edge']:
        metrics['edge'][key] /= total_samples

    n_batches = len(val_loader)
    for key in metrics['expert_usage']:
        metrics['expert_usage'][key] /= n_batches

    return metrics


def print_validation_report(metrics: dict, epoch: int):
    """Print comprehensive validation report."""
    print("\n" + "=" * 70)
    print(f"VALIDATION RESULTS - Epoch {epoch}")
    print("=" * 70)

    # Global metrics
    print("\n--- Global Metrics ---")
    psnr_gain = metrics['global']['psnr'] - metrics['global']['psnr_base']
    ssim_gain = metrics['global']['ssim'] - metrics['global']['ssim_base']
    print(f"  PSNR (ours):     {metrics['global']['psnr']:.2f} dB")
    print(f"  PSNR (baseline): {metrics['global']['psnr_base']:.2f} dB")
    print(f"  PSNR gain:       {'+' if psnr_gain >= 0 else ''}{psnr_gain:.2f} dB")
    print(f"  SSIM (ours):     {metrics['global']['ssim']:.4f}")
    print(f"  SSIM (baseline): {metrics['global']['ssim_base']:.4f}")
    print(f"  SSIM gain:       {'+' if ssim_gain >= 0 else ''}{ssim_gain:.4f}")

    # Layer-specific PSNR
    print("\n--- Layer-Specific PSNR (dB) ---")
    print(f"  {'Zone':<18} {'Ours':>8} {'Base':>8} {'Gain':>8}")
    total_layer_gain = 0
    for zone in LAYER_ZONES.keys():
        ours = metrics['layer_psnr'][zone]
        base = metrics['layer_psnr_base'][zone]
        gain = ours - base
        total_layer_gain += gain
        print(f"  {zone:<18} {ours:>8.2f} {base:>8.2f} {'+' if gain >= 0 else ''}{gain:>7.2f}")
    avg_layer_gain = total_layer_gain / len(LAYER_ZONES)
    print(f"  {'Average':<18} {'-':>8} {'-':>8} {'+' if avg_layer_gain >= 0 else ''}{avg_layer_gain:>7.2f}")

    # Layer-specific SSIM
    print("\n--- Layer-Specific SSIM ---")
    print(f"  {'Zone':<18} {'Ours':>8} {'Base':>8} {'Gain':>8}")
    total_ssim_gain = 0
    for zone in LAYER_ZONES.keys():
        ours = metrics['layer_ssim'][zone]
        base = metrics['layer_ssim_base'][zone]
        gain = ours - base
        total_ssim_gain += gain
        print(f"  {zone:<18} {ours:>8.4f} {base:>8.4f} {'+' if gain >= 0 else ''}{gain:>7.4f}")
    avg_ssim_gain = total_ssim_gain / len(LAYER_ZONES)
    print(f"  {'Average':<18} {'-':>8} {'-':>8} {'+' if avg_ssim_gain >= 0 else ''}{avg_ssim_gain:>7.4f}")

    # Edge preservation
    print("\n--- Edge Preservation ---")
    epi_gain = metrics['edge']['epi'] - metrics['edge']['epi_base']
    horiz_gain = metrics['edge']['horizontal'] - metrics['edge']['horizontal_base']
    print(f"  EPI (ours):      {metrics['edge']['epi']:.4f}")
    print(f"  EPI (baseline):  {metrics['edge']['epi_base']:.4f}")
    print(f"  EPI gain:        {'+' if epi_gain >= 0 else ''}{epi_gain:.4f}")
    print(f"  Horizontal (ours):    {metrics['edge']['horizontal']:.4f}")
    print(f"  Horizontal (base):    {metrics['edge']['horizontal_base']:.4f}")
    print(f"  Horizontal gain:      {'+' if horiz_gain >= 0 else ''}{horiz_gain:.4f}")

    # Expert usage
    print("\n--- Expert Usage ---")
    for name, usage in metrics['expert_usage'].items():
        bar = '#' * int(usage * 40)
        print(f"  {name:<10}: {usage:.2f} {bar}")

    print("=" * 70)

    return psnr_gain, ssim_gain


def main():
    parser = argparse.ArgumentParser(description='Anatomy-Aware NSAD Training v2')

    # Data
    parser.add_argument('--train_jsonl', type=str, default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_train_samples', type=int, default=None)

    # Model
    parser.add_argument('--base_ckpt', type=str, default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--fusion_mode', type=str, default='anatomy', choices=['anatomy', 'gated', 'residual'])
    parser.add_argument('--num_layer_zones', type=int, default=5)
    parser.add_argument('--use_depth_adaptive', action='store_true')
    parser.add_argument('--use_anatomy_fusion', action='store_true')

    # Loss weights (NEW - tunable)
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=0.5, help='Global SSIM weight')
    parser.add_argument('--w_layer_ssim', type=float, default=0.3, help='Layer-specific SSIM weight')
    parser.add_argument('--w_anatomy', type=float, default=0.1)
    parser.add_argument('--w_diversity', type=float, default=0.2, help='Expert diversity weight')
    parser.add_argument('--w_specialization', type=float, default=0.1, help='Routing entropy weight')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints/anatomy_v2')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device != 'gpu' else 'cuda')

    print("=" * 70)
    print("ANATOMY-AWARE NSAD TRAINING v2")
    print("=" * 70)
    print("\nIMPROVEMENTS:")
    print("  - SSIM loss (global + layer-specific)")
    print("  - Stronger expert diversity")
    print("  - Entropy minimization for sharper routing")
    print("  - Layer-specific metrics reporting")
    print("=" * 70)

    # Loss weights
    loss_weights = {
        'l1': args.w_l1,
        'ssim': args.w_ssim,
        'layer_ssim': args.w_layer_ssim,
        'anatomy': args.w_anatomy,
        'diversity': args.w_diversity,
        'specialization': args.w_specialization,
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

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
    from nsnd.models.nafnet import NAFNetFullFiLM
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

    # Losses
    anatomy_loss = AnatomyPreservingLoss().to(device)

    # Optimizer with differential learning rates
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    best_psnr = 0
    best_ssim = 0
    best_combined = -float('inf')  # PSNR gain + 10 * SSIM gain

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 70)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, anatomy_loss, device, epoch,
            loss_weights=loss_weights
        )

        print(f"\nTraining - L1: {train_metrics['l1']:.4f}, "
              f"SSIM: {1.0 - train_metrics['ssim']:.4f}, "
              f"Diversity: {train_metrics['diversity']:.4f}")

        # Validate
        val_metrics = validate_with_anatomy_metrics(
            model, val_loader, device, epoch, base_model, args.alpha
        )

        psnr_gain, ssim_gain = print_validation_report(val_metrics, epoch)

        # Combined score (weight SSIM heavily as user requested)
        combined_score = psnr_gain + 10 * ssim_gain

        # Save best model
        if combined_score > best_combined:
            best_psnr = val_metrics['global']['psnr']
            best_ssim = val_metrics['global']['ssim']
            best_combined = combined_score

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': best_ssim,
                'psnr_gain': psnr_gain,
                'ssim_gain': ssim_gain,
                'metrics': val_metrics,
            }, os.path.join(args.output_dir, 'best_model.pth'))

            print(f"\n*** Best model saved! PSNR: {best_psnr:.2f} dB (+{psnr_gain:.2f}), "
                  f"SSIM: {best_ssim:.4f} (+{ssim_gain:.4f})")

        scheduler.step()

    # Final report
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'best_model.pth')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
