#!/usr/bin/env python3
"""Statistical significance testing for clinical metrics.

Runs paired Wilcoxon signed-rank tests on per-image clinical metrics
comparing backbone output vs. corrected output.

Usage:
    python test_significance.py --checkpoint outputs/v8_optuna_full/best_model_cooperative.pth
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats

# Add project root
sys.path.insert(0, os.path.dirname(__file__))
from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset
)
from torch.utils.data import DataLoader


def compute_per_image_metrics(backbone_out, corrected, clean, noisy):
    """Compute all 6 standard clinical metrics for a single image.

    Returns dict with backbone and corrected values for each metric.
    """
    metrics = {}

    # --- CNR ---
    signal_mask = (clean > clean.mean()).float()
    bg_mask = 1.0 - signal_mask
    signal_sum = signal_mask.sum().clamp(min=1.0)
    bg_sum = bg_mask.sum().clamp(min=1.0)

    b_sig = (backbone_out * signal_mask).sum() / signal_sum
    b_bg = (backbone_out * bg_mask).sum() / bg_sum
    b_bg_std = torch.sqrt(((backbone_out - b_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
    metrics['cnr_backbone'] = ((b_sig - b_bg) / b_bg_std).clamp(-100, 100).item()

    c_sig = (corrected * signal_mask).sum() / signal_sum
    c_bg = (corrected * bg_mask).sum() / bg_sum
    c_bg_std = torch.sqrt(((corrected - c_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
    metrics['cnr_corrected'] = ((c_sig - c_bg) / c_bg_std).clamp(-100, 100).item()

    # --- TCI ---
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=clean.device).reshape(1, 1, 3, 3)
    clean_gy = F.conv2d(clean, sobel_y, padding=1).abs()
    backbone_gy = F.conv2d(backbone_out, sobel_y, padding=1).abs()
    corrected_gy = F.conv2d(corrected, sobel_y, padding=1).abs()
    clean_gy_mean = clean_gy.mean().clamp(min=1e-4)

    metrics['tci_backbone'] = (backbone_gy.mean() / clean_gy_mean).clamp(0, 10).item()
    metrics['tci_corrected'] = (corrected_gy.mean() / clean_gy_mean).clamp(0, 10).item()

    # --- EPI ---
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=clean.device).reshape(1, 1, 3, 3)
    clean_gx = F.conv2d(clean, sobel_x, padding=1)
    backbone_gx = F.conv2d(backbone_out, sobel_x, padding=1)
    corrected_gx = F.conv2d(corrected, sobel_x, padding=1)

    clean_edge = torch.sqrt(clean_gx**2 + clean_gy**2 + 1e-8)
    backbone_edge = torch.sqrt(backbone_gx**2 + backbone_gy**2 + 1e-8)
    corrected_edge = torch.sqrt(corrected_gx**2 + corrected_gy**2 + 1e-8)

    cf = clean_edge.view(-1)
    bf = backbone_edge.view(-1)
    cof = corrected_edge.view(-1)

    cs = cf.std().clamp(min=1e-4)
    bs = bf.std().clamp(min=1e-4)
    cos = cof.std().clamp(min=1e-4)

    cn = (cf - cf.mean()) / cs
    bn = (bf - bf.mean()) / bs
    con = (cof - cof.mean()) / cos

    metrics['epi_backbone'] = (cn * bn).mean().item()
    metrics['epi_corrected'] = (cn * con).mean().item()

    # --- Boundary Sharpness ---
    clean_gy_max = clean_gy.max().clamp(min=1e-4)
    metrics['bs_backbone'] = (backbone_gy.max() / clean_gy_max).clamp(0, 10).item()
    metrics['bs_corrected'] = (corrected_gy.max() / clean_gy_max).clamp(0, 10).item()

    # --- ENL ---
    H = backbone_out.shape[2]
    bg_b = backbone_out[0, 0, H*3//4:, :].cpu()
    bg_c = corrected[0, 0, H*3//4:, :].cpu()
    metrics['enl_backbone'] = (bg_b.mean() / bg_b.std().clamp(min=1e-6)).item() ** 2
    metrics['enl_corrected'] = (bg_c.mean() / bg_c.std().clamp(min=1e-6)).item() ** 2

    # --- SNR ---
    tissue_b = backbone_out[0, 0, :H//2, :].cpu()
    tissue_c = corrected[0, 0, :H//2, :].cpu()
    metrics['snr_backbone'] = (tissue_b.mean() / bg_b.std().clamp(min=1e-6)).item()
    metrics['snr_corrected'] = (tissue_c.mean() / bg_c.std().clamp(min=1e-6)).item()

    # --- PSNR ---
    mse_b = ((backbone_out - clean) ** 2).mean()
    mse_c = ((corrected - clean) ** 2).mean()
    metrics['psnr_backbone'] = (-10 * torch.log10(mse_b + 1e-10)).item()
    metrics['psnr_corrected'] = (-10 * torch.log10(mse_c + 1e-10)).item()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='outputs/v8_optuna_full/best_model_cooperative.pth')
    parser.add_argument('--test_jsonl', default='pku37_oct_dataset/pku37_real_test.jsonl')
    parser.add_argument('--val_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl')
    parser.add_argument('--backbone', default='nafnet')
    parser.add_argument('--pretrained_backbone', default='outputs/nafnet_pku37_w40/best_model.pth')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output', default='significance_results.json')
    args = parser.parse_args()

    device = 'cpu'

    # Load model
    print("Loading model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Strip _orig_mod. prefix from torch.compile'd models
    state_dict = ckpt['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Use val set (same as training validation) for consistent comparison
    jsonl_path = args.val_jsonl
    if os.path.exists(args.test_jsonl):
        jsonl_path = args.test_jsonl
        print(f"  Using TEST set: {jsonl_path}")
    else:
        print(f"  Using VAL set: {jsonl_path}")

    dataset = PKU37Dataset(jsonl_path, max_samples=args.max_samples, patch_size=0, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"  {len(dataset)} images")

    # Collect per-image metrics
    metric_names = ['cnr', 'tci', 'epi', 'bs', 'enl', 'snr', 'psnr']
    all_backbone = {m: [] for m in metric_names}
    all_corrected = {m: [] for m in metric_names}

    print("\nComputing per-image metrics...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            # Forward pass — call backbone + corrector directly (bypass verifier)
            # This matches the training validation methodology exactly
            backbone_out, nafnet_unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=nafnet_unc, return_details=False,
            )

            m = compute_per_image_metrics(backbone_out, corrected, clean, noisy)

            for name in metric_names:
                all_backbone[name].append(m[f'{name}_backbone'])
                all_corrected[name].append(m[f'{name}_corrected'])

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(dataset)}] PSNR: {m['psnr_backbone']:.2f}→{m['psnr_corrected']:.2f}, "
                      f"CNR: {m['cnr_backbone']:.2f}→{m['cnr_corrected']:.2f}")

            del backbone_out, corrected, noisy, clean, nafnet_unc, info

    # Statistical tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS (Wilcoxon Signed-Rank, paired)")
    print("="*80)

    display_names = {
        'cnr': 'CNR', 'tci': 'TCI', 'epi': 'EPI',
        'bs': 'Boundary Sharpness', 'enl': 'ENL', 'snr': 'SNR', 'psnr': 'PSNR'
    }

    results = {}

    print(f"\n{'Metric':<22} {'Backbone':>10} {'Corrected':>10} {'Delta':>10} {'% Impr':>8} "
          f"{'p-value':>10} {'Signif':>8} {'Effect d':>10}")
    print("-" * 100)

    for name in metric_names:
        b = np.array(all_backbone[name])
        c = np.array(all_corrected[name])
        d = c - b

        mean_b = np.mean(b)
        mean_c = np.mean(c)
        mean_d = np.mean(d)
        pct = (mean_d / abs(mean_b) * 100) if abs(mean_b) > 1e-8 else 0

        # Wilcoxon signed-rank test (two-sided)
        try:
            stat, p_value = stats.wilcoxon(d, alternative='two-sided')
        except ValueError:
            # All differences are zero
            stat, p_value = 0, 1.0

        # Cohen's d (paired)
        std_d = np.std(d, ddof=1)
        cohens_d = mean_d / std_d if std_d > 1e-8 else 0

        # Significance markers
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"

        results[name] = {
            'display_name': display_names[name],
            'backbone_mean': float(mean_b),
            'corrected_mean': float(mean_c),
            'delta_mean': float(mean_d),
            'delta_std': float(std_d),
            'pct_improvement': float(pct),
            'wilcoxon_stat': float(stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_005': bool(p_value < 0.05),
            'significant_001': bool(p_value < 0.01),
            'n_images': len(b),
            'n_positive': int(np.sum(d > 0)),
            'n_negative': int(np.sum(d < 0)),
            'n_zero': int(np.sum(d == 0)),
        }

        print(f"{display_names[name]:<22} {mean_b:>10.4f} {mean_c:>10.4f} {mean_d:>+10.4f} {pct:>+7.2f}% "
              f"{p_value:>10.2e} {sig:>8} {cohens_d:>+10.4f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    n = results[metric_names[0]]['n_images']
    print(f"  N = {n} images")
    print(f"  Significance level: * p<0.05, ** p<0.01, *** p<0.001")
    print(f"  Effect size (Cohen's d): |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")

    sig_count = sum(1 for r in results.values() if r['significant_005'] and r['display_name'] != 'PSNR')
    print(f"\n  Significant improvements (p<0.05): {sig_count}/6 clinical metrics")

    for name in metric_names:
        r = results[name]
        d_abs = abs(r['cohens_d'])
        if d_abs < 0.2:
            effect = "negligible"
        elif d_abs < 0.5:
            effect = "small"
        elif d_abs < 0.8:
            effect = "medium"
        else:
            effect = "large"

        direction = "improved" if r['delta_mean'] > 0 else "decreased"
        if name == 'psnr':
            direction = "decreased" if r['delta_mean'] < 0 else "improved"

        print(f"  {r['display_name']:<22}: {direction}, {r['n_positive']}/{n} images improved, "
              f"effect={effect} (d={r['cohens_d']:+.3f})")

    # Detailed distribution info
    print("\n" + "="*80)
    print("PER-METRIC DISTRIBUTION (delta = corrected - backbone)")
    print("="*80)
    for name in metric_names:
        b = np.array(all_backbone[name])
        c = np.array(all_corrected[name])
        d = c - b
        r = results[name]
        print(f"\n  {r['display_name']}:")
        print(f"    Mean delta: {np.mean(d):+.6f} +/- {np.std(d):.6f}")
        print(f"    Median delta: {np.median(d):+.6f}")
        print(f"    Min/Max delta: [{np.min(d):+.6f}, {np.max(d):+.6f}]")
        print(f"    Improved/Worsened/Tied: {r['n_positive']}/{r['n_negative']}/{r['n_zero']}")
        print(f"    95% CI: [{np.percentile(d, 2.5):+.6f}, {np.percentile(d, 97.5):+.6f}]")

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
