#!/usr/bin/env python3
"""
Quick validation script for Anatomy-Aware NSAD with comprehensive metrics.

Metrics tracked:
1. Global: PSNR, SSIM (vs baseline NAFNet)
2. Anatomy ROI: Per-layer PSNR/SSIM for 5 retinal zones
3. Interpretability: Expert usage distribution, noise type accuracy
4. Edge preservation: Layer boundary quality
"""

import gc
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
sys.path.insert(0, './nsnd_oct')

from nsnd.models.sansd import AnatomyAwareSANSD
from nsnd.models.nafnet import NAFNetFullFiLM

# Layer zones for anatomy-aware metrics
LAYER_ZONES = {
    'vitreous_nfl': (0.0, 0.15),      # Top 15%
    'inner_retina': (0.15, 0.35),     # 15-35%
    'outer_nuclear': (0.35, 0.55),    # 35-55%
    'photoreceptors': (0.55, 0.75),   # 55-75%
    'rpe_choroid': (0.75, 1.0),       # Bottom 25%
}


def compute_psnr(pred, target, data_range=1.0):
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(data_range**2 / mse).item()


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM between two tensors."""
    C1, C2 = 0.01**2, 0.03**2

    # Create gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(pred, window, padding=window_size//2)
    mu2 = F.conv2d(target, window, padding=window_size//2)

    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred**2, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(target**2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_layer_metrics(pred, target, H):
    """Compute PSNR/SSIM for each anatomical layer zone."""
    metrics = {}
    for zone_name, (start_frac, end_frac) in LAYER_ZONES.items():
        start_row = int(start_frac * H)
        end_row = int(end_frac * H)

        pred_zone = pred[:, :, start_row:end_row, :]
        target_zone = target[:, :, start_row:end_row, :]

        metrics[zone_name] = {
            'psnr': compute_psnr(pred_zone, target_zone),
            'ssim': compute_ssim(pred_zone, target_zone),
        }
    return metrics


def compute_edge_preservation(pred, target, noisy):
    """Compute edge preservation index."""
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    # Edges in clean image
    clean_gx = F.conv2d(target, sobel_x, padding=1)
    clean_gy = F.conv2d(target, sobel_y, padding=1)
    clean_edges = torch.sqrt(clean_gx**2 + clean_gy**2)

    # Edges in denoised
    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    pred_edges = torch.sqrt(pred_gx**2 + pred_gy**2)

    # EPI = correlation between edge maps
    clean_flat = clean_edges.flatten()
    pred_flat = pred_edges.flatten()

    clean_centered = clean_flat - clean_flat.mean()
    pred_centered = pred_flat - pred_flat.mean()

    corr = (clean_centered * pred_centered).sum() / \
           (torch.sqrt((clean_centered**2).sum() * (pred_centered**2).sum()) + 1e-8)

    return corr.item()


def load_sample_data(jsonl_path, num_samples=20, patch_size=64):
    """Load sample data for validation."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line.strip())

            # Load images
            noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
            clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

            # Center crop
            h, w = noisy.shape
            top = (h - patch_size) // 2
            left = (w - patch_size) // 2
            noisy = noisy[top:top+patch_size, left:left+patch_size]
            clean = clean[top:top+patch_size, left:left+patch_size]

            samples.append({
                'noisy': torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float(),
                'clean': torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float(),
                'weights': data['weights'],
            })
    return samples


def validate():
    print("=" * 70)
    print("QUICK VALIDATION - Anatomy-Aware NSAD")
    print("=" * 70)

    device = 'cpu'
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")

    # Anatomy-Aware NSAD
    model = AnatomyAwareSANSD(
        backbone_width=64,
        backbone_ckpt='outputs/nafnet_analysis_maps_w64/nafnet_best.pth',
        fusion_mode='anatomy',
        num_layer_zones=5,
        use_depth_adaptive=True,
        use_anatomy_fusion=True,
    ).to(device)
    model.eval()

    # Baseline NAFNet (for comparison)
    baseline = NAFNetFullFiLM(
        img_channel=1,
        width=64,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        middle_blk_num=2,
        cond_dim=32,
    ).to(device)

    ckpt_path = 'outputs/nafnet_analysis_maps_w64/nafnet_best.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        baseline.load_state_dict(state, strict=False)
    baseline.eval()

    # Load validation data
    print("Loading validation data...")
    val_jsonl = 'weights_duke_analysis_maps_val.jsonl'
    if not os.path.exists(val_jsonl):
        print(f"ERROR: {val_jsonl} not found")
        return

    samples = load_sample_data(val_jsonl, num_samples=30, patch_size=64)
    print(f"Loaded {len(samples)} samples")

    # Initialize metrics
    metrics = {
        'global': {
            'psnr_noisy': [], 'ssim_noisy': [],
            'psnr_baseline': [], 'ssim_baseline': [],
            'psnr_nsad': [], 'ssim_nsad': [],
        },
        'anatomy': {zone: {'psnr_baseline': [], 'ssim_baseline': [],
                          'psnr_nsad': [], 'ssim_nsad': []}
                   for zone in LAYER_ZONES.keys()},
        'edge': {'epi_baseline': [], 'epi_nsad': []},
        'interpretability': {
            'expert_usage': {'speckle': [], 'banding': [], 'gaussian': [], 'shot': []},
            'noise_type_correct': 0,
            'noise_type_total': 0,
        },
    }

    print("\nRunning validation...")
    print("-" * 70)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="Validating")):
            noisy = sample['noisy'].to(device)
            clean = sample['clean'].to(device)
            gt_weights = sample['weights']

            H = noisy.shape[2]

            # Baseline prediction
            baseline_out = baseline(noisy, spatial_map=None, basis=None, alpha=2.0, gate=None)

            # NSAD prediction with interpretation
            nsad_out, interp = model(noisy, return_interpretation=True)

            # === GLOBAL METRICS ===
            metrics['global']['psnr_noisy'].append(compute_psnr(noisy, clean))
            metrics['global']['ssim_noisy'].append(compute_ssim(noisy, clean))
            metrics['global']['psnr_baseline'].append(compute_psnr(baseline_out, clean))
            metrics['global']['ssim_baseline'].append(compute_ssim(baseline_out, clean))
            metrics['global']['psnr_nsad'].append(compute_psnr(nsad_out, clean))
            metrics['global']['ssim_nsad'].append(compute_ssim(nsad_out, clean))

            # === ANATOMY ROI METRICS ===
            baseline_layer = compute_layer_metrics(baseline_out, clean, H)
            nsad_layer = compute_layer_metrics(nsad_out, clean, H)

            for zone in LAYER_ZONES.keys():
                metrics['anatomy'][zone]['psnr_baseline'].append(baseline_layer[zone]['psnr'])
                metrics['anatomy'][zone]['ssim_baseline'].append(baseline_layer[zone]['ssim'])
                metrics['anatomy'][zone]['psnr_nsad'].append(nsad_layer[zone]['psnr'])
                metrics['anatomy'][zone]['ssim_nsad'].append(nsad_layer[zone]['ssim'])

            # === EDGE PRESERVATION ===
            metrics['edge']['epi_baseline'].append(compute_edge_preservation(baseline_out, clean, noisy))
            metrics['edge']['epi_nsad'].append(compute_edge_preservation(nsad_out, clean, noisy))

            # === INTERPRETABILITY ===
            # Expert usage
            noise_type = interp['noise_type']  # [1, 4, H, W]
            usage = noise_type.mean(dim=[0, 2, 3])  # [4]
            for idx, name in enumerate(['speckle', 'banding', 'gaussian', 'shot']):
                metrics['interpretability']['expert_usage'][name].append(usage[idx].item())

            # Noise type accuracy
            gt_dominant = max(gt_weights, key=gt_weights.get)
            pred_weights = {name: usage[idx].item() for idx, name in enumerate(['speckle', 'banding', 'gaussian', 'shot'])}
            pred_dominant = max(pred_weights, key=pred_weights.get)

            metrics['interpretability']['noise_type_total'] += 1
            if gt_dominant == pred_dominant:
                metrics['interpretability']['noise_type_correct'] += 1

            # Cleanup
            del noisy, clean, baseline_out, nsad_out, interp, noise_type
            gc.collect()

    # === PRINT RESULTS ===
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    # Global metrics
    print("\n📊 GLOBAL METRICS:")
    print("-" * 50)
    avg = lambda x: sum(x) / len(x)

    psnr_noisy = avg(metrics['global']['psnr_noisy'])
    psnr_baseline = avg(metrics['global']['psnr_baseline'])
    psnr_nsad = avg(metrics['global']['psnr_nsad'])
    ssim_baseline = avg(metrics['global']['ssim_baseline'])
    ssim_nsad = avg(metrics['global']['ssim_nsad'])

    print(f"  {'Method':<15} {'PSNR (dB)':>12} {'SSIM':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*10}")
    print(f"  {'Noisy':<15} {psnr_noisy:>12.2f} {avg(metrics['global']['ssim_noisy']):>10.4f}")
    print(f"  {'Baseline':<15} {psnr_baseline:>12.2f} {ssim_baseline:>10.4f}")
    print(f"  {'NSAD (Ours)':<15} {psnr_nsad:>12.2f} {ssim_nsad:>10.4f}")
    print(f"")
    print(f"  📈 Gain over baseline: PSNR {psnr_nsad - psnr_baseline:+.2f} dB | SSIM {ssim_nsad - ssim_baseline:+.4f}")

    # Anatomy ROI metrics
    print("\n🔬 ANATOMY ROI METRICS (Per-Layer PSNR):")
    print("-" * 50)
    print(f"  {'Layer Zone':<18} {'Baseline':>10} {'NSAD':>10} {'Gain':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")

    total_gain = 0
    for zone in LAYER_ZONES.keys():
        base_psnr = avg(metrics['anatomy'][zone]['psnr_baseline'])
        nsad_psnr = avg(metrics['anatomy'][zone]['psnr_nsad'])
        gain = nsad_psnr - base_psnr
        total_gain += gain
        marker = "✓" if gain > 0 else "✗"
        print(f"  {zone:<18} {base_psnr:>10.2f} {nsad_psnr:>10.2f} {gain:>+9.2f} {marker}")

    avg_layer_gain = total_gain / len(LAYER_ZONES)
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Average':<18} {'':<10} {'':<10} {avg_layer_gain:>+9.2f}")

    # Edge preservation
    print("\n🔲 EDGE PRESERVATION:")
    print("-" * 50)
    epi_baseline = avg(metrics['edge']['epi_baseline'])
    epi_nsad = avg(metrics['edge']['epi_nsad'])
    print(f"  Baseline EPI: {epi_baseline:.4f}")
    print(f"  NSAD EPI:     {epi_nsad:.4f}")
    print(f"  Gain:         {epi_nsad - epi_baseline:+.4f}")

    # Interpretability
    print("\n🎯 INTERPRETABILITY METRICS:")
    print("-" * 50)

    # Expert usage
    print("  Expert Usage Distribution:")
    for name in ['speckle', 'banding', 'gaussian', 'shot']:
        usage = avg(metrics['interpretability']['expert_usage'][name])
        bar = "█" * int(usage * 40)
        print(f"    {name:<10}: {usage:.2f} {bar}")

    # Noise type accuracy
    correct = metrics['interpretability']['noise_type_correct']
    total = metrics['interpretability']['noise_type_total']
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n  Noise Type Classification Accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    status_psnr = "✅" if psnr_nsad > psnr_baseline else "⚠️"
    status_ssim = "✅" if ssim_nsad > ssim_baseline else "⚠️"
    status_layer = "✅" if avg_layer_gain > 0 else "⚠️"
    status_edge = "✅" if epi_nsad > epi_baseline else "⚠️"

    print(f"  {status_psnr} Global PSNR:     {psnr_nsad - psnr_baseline:+.2f} dB over baseline")
    print(f"  {status_ssim} Global SSIM:     {ssim_nsad - ssim_baseline:+.4f} over baseline")
    print(f"  {status_layer} Anatomy Layers:  {avg_layer_gain:+.2f} dB avg gain")
    print(f"  {status_edge} Edge Preserv.:   {epi_nsad - epi_baseline:+.4f} EPI gain")
    print(f"  🎯 Interpretability: {accuracy:.1f}% noise type accuracy")

    # Issues detected
    issues = []
    if psnr_nsad <= psnr_baseline:
        issues.append("PSNR not improving over baseline")
    if ssim_nsad <= ssim_baseline:
        issues.append("SSIM not improving over baseline")
    if avg_layer_gain <= 0:
        issues.append("Average layer PSNR not improving")
    if accuracy < 50:
        issues.append("Low noise type classification accuracy")

    if issues:
        print("\n⚠️  POTENTIAL ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n✅ All metrics look healthy!")

    print("=" * 70)


if __name__ == '__main__':
    validate()
