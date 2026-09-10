#!/usr/bin/env python3
"""
Optimized Neuro-Symbolic Correction

Best strategies from evaluation:
1. Failure-Guided (α=0.5): +0.021 dB total, +0.0007 dB low-error (no degradation!)
2. Uncertainty-Weighted: +0.032 dB total, -0.004 dB low-error

Now: Fine-tune α parameter and combine best approaches.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys

sys.path.insert(0, 'nsnd_oct')


class StructurePredicate:
    def __init__(self):
        self.avg_kernel_3 = torch.ones(1, 1, 3, 3) / 9
        self.avg_kernel_7 = torch.ones(1, 1, 7, 7) / 49

    def local_stats(self, x, k):
        pad = k // 2
        kernel = self.avg_kernel_3 if k == 3 else self.avg_kernel_7
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        mean = F.conv2d(x_pad, kernel)
        x_sq_pad = F.pad(x ** 2, (pad, pad, pad, pad), mode='reflect')
        var = F.conv2d(x_sq_pad, kernel) - mean ** 2
        return mean, torch.sqrt(var.clamp(min=0) + 1e-8)

    def normalize(self, f, pct=0.95):
        B = f.shape[0]
        p = torch.quantile(f.view(B, -1), pct, dim=1, keepdim=True).view(B, 1, 1, 1)
        return (f / (p + 1e-8)).clamp(0, 1)

    def __call__(self, noisy, denoised):
        residual = noisy - denoised
        int_s3, _ = self.local_stats(denoised, 3)
        int_s3_norm = self.normalize(int_s3)
        _, res_std_s3 = self.local_stats(residual, 3)
        _, res_std_s7 = self.local_stats(residual, 7)
        res_var_s3 = self.normalize(res_std_s3 ** 2)
        res_var_s7 = self.normalize(res_std_s7 ** 2)

        failure = (
            0.70 * int_s3_norm +
            0.20 * (int_s3_norm * res_var_s7) +
            0.10 * res_var_s3
        ).clamp(0, 1)
        return failure


def flip_h(x): return torch.flip(x, [-1])
def flip_v(x): return torch.flip(x, [-2])

AUGS = [
    (lambda x: x, lambda x: x),
    (flip_h, flip_h),
    (flip_v, flip_v),
    (lambda x: flip_h(flip_v(x)), lambda x: flip_h(flip_v(x))),
]

def get_sc_outputs(backbone, noisy):
    outputs = []
    for aug, inv_aug in AUGS:
        out = backbone(aug(noisy)).clamp(0, 1)
        outputs.append(inv_aug(out))
    stacked = torch.stack(outputs)
    return {
        'mean': stacked.mean(dim=0),
        'std': stacked.std(dim=0),
    }


def compute_psnr(pred, target, mask=None):
    if mask is not None:
        pred_masked = pred[mask]
        target_masked = target[mask]
        if pred_masked.numel() == 0:
            return float('nan')
        mse = F.mse_loss(pred_masked, target_masked)
    else:
        mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def strategy_failure_guided(baseline, sc_data, failure_map, alpha):
    blend_weight = (failure_map * alpha).clamp(0, 1)
    return (1 - blend_weight) * baseline + blend_weight * sc_data['mean']


def strategy_hybrid(baseline, sc_data, failure_map, alpha=0.5, uncertainty_weight=0.3):
    """
    Hybrid: Failure-guided + Uncertainty-weighted

    - Use failure map to determine WHERE to correct
    - Use uncertainty to determine HOW MUCH to correct

    If failure high AND uncertainty low → confident correction → full blend
    If failure high BUT uncertainty high → uncertain → reduce blend
    """
    # Normalize uncertainty
    uncertainty = sc_data['std']
    uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)

    # Reduce blend where uncertainty is high
    confidence = 1 - uncertainty_norm * uncertainty_weight

    # Final blend weight
    blend_weight = (failure_map * alpha * confidence).clamp(0, 1)

    return (1 - blend_weight) * baseline + blend_weight * sc_data['mean']


def main():
    print("=" * 70)
    print("OPTIMIZED NEURO-SYMBOLIC CORRECTION")
    print("=" * 70)

    # Load model
    print("\nLoading NAFNet...", flush=True)
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    backbone = NAFNet(img_channel=1, width=64, middle_blk_num=2,
                      enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2])
    ckpt = torch.load('outputs/nafnet_pku37/nafnet_best.pth',
                      map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt['state_dict'], strict=False)
    backbone.eval()

    predicate = StructurePredicate()

    # Load samples
    print("Loading samples...", flush=True)
    lines = open('pku37_oct_dataset/weights_pku37_analysis_val.jsonl').readlines()
    samples = []
    for idx in [0, 50, 100, 150, 200]:
        d = json.loads(lines[idx])
        c = torch.from_numpy(np.array(Image.open(d['clean_path']).convert('L'))).float()/255
        n = torch.from_numpy(np.array(Image.open(d['noisy_path']).convert('L'))).float()/255
        samples.append({
            'noisy': n.unsqueeze(0).unsqueeze(0),
            'clean': c.unsqueeze(0).unsqueeze(0),
            'name': Path(d['clean_path']).stem,
        })
    print(f"Loaded {len(samples)} samples\n")

    # Test different alpha values
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    hybrid_configs = [
        (0.4, 0.2), (0.5, 0.3), (0.6, 0.3), (0.5, 0.5),
    ]

    results = {}

    print("-" * 70)
    print("Testing α values for Failure-Guided strategy...")
    print("-" * 70)

    with torch.no_grad():
        for alpha in alphas:
            totals, highs, lows = [], [], []

            for sample in samples:
                noisy, clean = sample['noisy'], sample['clean']
                baseline = backbone(noisy).clamp(0, 1)
                sc_data = get_sc_outputs(backbone, noisy)
                failure = predicate(noisy, baseline)

                actual_error = (baseline - clean).abs()
                error_threshold = torch.quantile(actual_error, 0.7)
                high_mask = actual_error > error_threshold
                low_mask = ~high_mask

                output = strategy_failure_guided(baseline, sc_data, failure, alpha)

                totals.append(compute_psnr(output, clean))
                highs.append(compute_psnr(output, clean, high_mask))
                lows.append(compute_psnr(output, clean, low_mask))

            results[f'FG-α={alpha}'] = {
                'total': np.mean(totals),
                'high': np.mean(highs),
                'low': np.mean(lows),
            }

        # Baseline for comparison
        baseline_results = {'total': [], 'high': [], 'low': []}
        for sample in samples:
            noisy, clean = sample['noisy'], sample['clean']
            baseline = backbone(noisy).clamp(0, 1)

            actual_error = (baseline - clean).abs()
            error_threshold = torch.quantile(actual_error, 0.7)
            high_mask = actual_error > error_threshold
            low_mask = ~high_mask

            baseline_results['total'].append(compute_psnr(baseline, clean))
            baseline_results['high'].append(compute_psnr(baseline, clean, high_mask))
            baseline_results['low'].append(compute_psnr(baseline, clean, low_mask))

        base_total = np.mean(baseline_results['total'])
        base_high = np.mean(baseline_results['high'])
        base_low = np.mean(baseline_results['low'])

        print(f"\n{'Strategy':<20} {'Total':<12} {'Δ Total':<10} {'Δ High':<10} {'Δ Low':<10}")
        print("-" * 62)
        print(f"{'Baseline':<20} {base_total:.3f}       {'+0.0000':<10} {'+0.0000':<10} {'+0.0000':<10}")

        for name, r in results.items():
            dt = r['total'] - base_total
            dh = r['high'] - base_high
            dl = r['low'] - base_low
            print(f"{name:<20} {r['total']:.3f}       {dt:+.4f}     {dh:+.4f}     {dl:+.4f}")

        # Test hybrid
        print("\n" + "-" * 70)
        print("Testing Hybrid (Failure-Guided + Uncertainty-Weighted)...")
        print("-" * 70)

        for alpha, uw in hybrid_configs:
            totals, highs, lows = [], [], []

            for sample in samples:
                noisy, clean = sample['noisy'], sample['clean']
                baseline = backbone(noisy).clamp(0, 1)
                sc_data = get_sc_outputs(backbone, noisy)
                failure = predicate(noisy, baseline)

                actual_error = (baseline - clean).abs()
                error_threshold = torch.quantile(actual_error, 0.7)
                high_mask = actual_error > error_threshold
                low_mask = ~high_mask

                output = strategy_hybrid(baseline, sc_data, failure, alpha, uw)

                totals.append(compute_psnr(output, clean))
                highs.append(compute_psnr(output, clean, high_mask))
                lows.append(compute_psnr(output, clean, low_mask))

            r = {
                'total': np.mean(totals),
                'high': np.mean(highs),
                'low': np.mean(lows),
            }
            dt = r['total'] - base_total
            dh = r['high'] - base_high
            dl = r['low'] - base_low
            name = f"Hybrid(α={alpha},u={uw})"
            print(f"{name:<25} {r['total']:.3f}    {dt:+.4f}     {dh:+.4f}     {dl:+.4f}")
            results[name] = r

    # Find best balanced strategy
    print("\n" + "=" * 70)
    print("BEST BALANCED STRATEGIES (improves total, preserves low-error)")
    print("=" * 70)

    balanced = []
    for name, r in results.items():
        dt = r['total'] - base_total
        dl = r['low'] - base_low
        if dt > 0.01 and dl >= -0.01:  # Significant improvement, minimal degradation
            balanced.append((name, dt, dl, r['high'] - base_high))

    balanced.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Strategy':<25} {'Δ Total':<10} {'Δ High':<10} {'Δ Low':<10}")
    print("-" * 66)
    for i, (name, dt, dl, dh) in enumerate(balanced, 1):
        print(f"{i:<6} {name:<25} {dt:+.4f}     {dh:+.4f}     {dl:+.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if balanced:
        best = balanced[0]
        print(f"\n  Best Strategy: {best[0]}")
        print(f"    → Total PSNR improvement: {best[1]:+.4f} dB")
        print(f"    → High-error region improvement: {best[3]:+.4f} dB")
        print(f"    → Low-error region change: {best[2]:+.4f} dB (preserved!)")
        print(f"\n  This achieves targeted correction without degrading already-good regions.")


if __name__ == "__main__":
    main()
