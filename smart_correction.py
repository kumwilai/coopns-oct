#!/usr/bin/env python3
"""
Smart Neuro-Symbolic Correction Strategies

Key insight from analysis:
- Naive SC: +0.038 dB overall, +0.068 dB high-error, -0.088 dB low-error
- Problem: Uniform averaging hurts regions that were already good

Smart strategies:
1. Failure-Guided Blending: Blend baseline with SC based on failure map
2. Threshold-Based Selection: Only correct pixels above failure threshold
3. Uncertainty-Weighted: Weight by inverse uncertainty (confident predictions win)
4. Iterative Refinement: Multiple passes with decreasing correction strength
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys

sys.path.insert(0, 'nsnd_oct')


# =============================================================================
# STRUCTURE PREDICATE (Optimized v13)
# =============================================================================

class StructurePredicate:
    """Optimized structure predicate for error detection."""

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


# =============================================================================
# SELF-CONSISTENCY ENGINE
# =============================================================================

def flip_h(x): return torch.flip(x, [-1])
def flip_v(x): return torch.flip(x, [-2])

AUGS = [
    (lambda x: x, lambda x: x),
    (flip_h, flip_h),
    (flip_v, flip_v),
    (lambda x: flip_h(flip_v(x)), lambda x: flip_h(flip_v(x))),
]

def get_sc_outputs(backbone, noisy):
    """Get all augmentation outputs and statistics."""
    outputs = []
    for aug, inv_aug in AUGS:
        out = backbone(aug(noisy)).clamp(0, 1)
        outputs.append(inv_aug(out))

    stacked = torch.stack(outputs)
    return {
        'outputs': outputs,
        'stacked': stacked,
        'mean': stacked.mean(dim=0),
        'std': stacked.std(dim=0),
        'min': stacked.min(dim=0)[0],
        'max': stacked.max(dim=0)[0],
    }


# =============================================================================
# SMART CORRECTION STRATEGIES
# =============================================================================

def strategy_naive_mean(baseline, sc_data, failure_map, **kwargs):
    """Strategy 0: Naive mean (current approach)."""
    return sc_data['mean']


def strategy_failure_guided_blend(baseline, sc_data, failure_map, alpha=1.0, **kwargs):
    """
    Strategy 1: Failure-Guided Blending

    blend = (1 - alpha * failure) * baseline + (alpha * failure) * sc_mean

    Where failure is high, use more SC. Where failure is low, keep baseline.
    """
    blend_weight = (failure_map * alpha).clamp(0, 1)
    return (1 - blend_weight) * baseline + blend_weight * sc_data['mean']


def strategy_threshold_correction(baseline, sc_data, failure_map, threshold=0.5, **kwargs):
    """
    Strategy 2: Threshold-Based Selection

    Only correct pixels where failure > threshold.
    Leave other pixels unchanged.
    """
    mask = (failure_map > threshold).float()
    return (1 - mask) * baseline + mask * sc_data['mean']


def strategy_uncertainty_weighted(baseline, sc_data, failure_map, **kwargs):
    """
    Strategy 3: Uncertainty-Weighted Correction

    Weight baseline and SC by inverse of their uncertainties.
    High-confidence predictions (low std) get more weight.
    """
    # Use SC std as uncertainty
    sc_uncertainty = sc_data['std']

    # Baseline has no augmentations, so estimate uncertainty from local variance
    # of the difference between baseline and SC mean
    diff = (baseline - sc_data['mean']).abs()

    # Weight inversely by uncertainty
    sc_weight = 1.0 / (sc_uncertainty + 0.01)
    base_weight = 1.0 / (diff + 0.01)

    # Normalize weights
    total = sc_weight + base_weight
    sc_weight = sc_weight / total

    return (1 - sc_weight) * baseline + sc_weight * sc_data['mean']


def strategy_conservative_blend(baseline, sc_data, failure_map, threshold=0.7, blend=0.5, **kwargs):
    """
    Strategy 4: Conservative Blend

    - High failure (>threshold): Use SC mean
    - Medium failure: Blend baseline and SC
    - Low failure: Keep baseline
    """
    high_mask = (failure_map > threshold).float()
    medium_mask = ((failure_map > 0.3) & (failure_map <= threshold)).float()
    low_mask = (failure_map <= 0.3).float()

    result = (
        high_mask * sc_data['mean'] +
        medium_mask * (blend * sc_data['mean'] + (1-blend) * baseline) +
        low_mask * baseline
    )
    return result


def strategy_adaptive_blend(baseline, sc_data, failure_map, **kwargs):
    """
    Strategy 5: Adaptive Blend (Best of both worlds)

    Key insight: Use failure map to determine blend, but also consider
    whether SC actually improves (by looking at SC std).

    - If failure high AND SC std low → SC is confident about correction → use SC
    - If failure high BUT SC std high → SC unsure → partial blend
    - If failure low → keep baseline
    """
    # Normalize failure and uncertainty to same scale
    failure_norm = failure_map / (failure_map.max() + 1e-8)
    uncertainty_norm = sc_data['std'] / (sc_data['std'].max() + 1e-8)

    # SC is good when: high failure AND low uncertainty
    sc_confidence = failure_norm * (1 - uncertainty_norm)

    # Smooth blend weight
    blend_weight = torch.sigmoid(5 * (sc_confidence - 0.3))

    return (1 - blend_weight) * baseline + blend_weight * sc_data['mean']


def strategy_iterative_refinement(backbone, noisy, predicate, n_iterations=2, **kwargs):
    """
    Strategy 6: Iterative Refinement

    1. Get initial denoised
    2. Compute failure map
    3. Apply SC only to high-failure regions
    4. Repeat with decreasing intensity
    """
    current = backbone(noisy).clamp(0, 1)

    for i in range(n_iterations):
        failure = predicate(noisy, current)
        sc_data = get_sc_outputs(backbone, noisy)

        # Decreasing blend strength
        strength = 1.0 / (i + 1)
        blend_weight = (failure * strength).clamp(0, 1)

        current = (1 - blend_weight) * current + blend_weight * sc_data['mean']

    return current


# =============================================================================
# METRICS
# =============================================================================

def compute_psnr(pred, target, mask=None):
    """Compute PSNR, optionally in masked region only."""
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


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_strategies():
    """Compare all smart correction strategies."""
    print("=" * 70)
    print("SMART NEURO-SYMBOLIC CORRECTION STRATEGIES")
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

    # Define strategies
    strategies = [
        ('Baseline', None),
        ('Naive SC (mean)', strategy_naive_mean),
        ('Failure-Guided (α=0.5)', lambda b, s, f: strategy_failure_guided_blend(b, s, f, alpha=0.5)),
        ('Failure-Guided (α=1.0)', lambda b, s, f: strategy_failure_guided_blend(b, s, f, alpha=1.0)),
        ('Threshold (t=0.5)', lambda b, s, f: strategy_threshold_correction(b, s, f, threshold=0.5)),
        ('Threshold (t=0.7)', lambda b, s, f: strategy_threshold_correction(b, s, f, threshold=0.7)),
        ('Uncertainty-Weighted', strategy_uncertainty_weighted),
        ('Conservative Blend', strategy_conservative_blend),
        ('Adaptive Blend', strategy_adaptive_blend),
    ]

    # Results storage
    results = {name: {'total': [], 'high': [], 'low': []} for name, _ in strategies}

    print("-" * 70)
    print("Evaluating strategies on each sample...")
    print("-" * 70)

    with torch.no_grad():
        for i, sample in enumerate(samples):
            noisy = sample['noisy']
            clean = sample['clean']
            name = sample['name']

            # Get baseline and SC data
            baseline = backbone(noisy).clamp(0, 1)
            sc_data = get_sc_outputs(backbone, noisy)
            failure_map = predicate(noisy, baseline)

            # Create masks based on actual error (for fair comparison)
            actual_error = (baseline - clean).abs()
            error_threshold = torch.quantile(actual_error, 0.7)
            high_error_mask = actual_error > error_threshold
            low_error_mask = ~high_error_mask

            print(f"\n[{i+1}] {name}")

            for name_s, strategy_fn in strategies:
                if strategy_fn is None:
                    output = baseline
                else:
                    output = strategy_fn(baseline, sc_data, failure_map)

                psnr_total = compute_psnr(output, clean)
                psnr_high = compute_psnr(output, clean, high_error_mask)
                psnr_low = compute_psnr(output, clean, low_error_mask)

                results[name_s]['total'].append(psnr_total)
                results[name_s]['high'].append(psnr_high)
                results[name_s]['low'].append(psnr_low)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average PSNR (dB) across all samples")
    print("=" * 70)

    baseline_total = np.mean(results['Baseline']['total'])
    baseline_high = np.mean(results['Baseline']['high'])
    baseline_low = np.mean(results['Baseline']['low'])

    print(f"\n{'Strategy':<25} {'Total':<12} {'High-Err':<12} {'Low-Err':<12} {'Δ Total':<10} {'Δ High':<10} {'Δ Low':<10}")
    print("-" * 95)

    for name_s, _ in strategies:
        total = np.mean(results[name_s]['total'])
        high = np.mean(results[name_s]['high'])
        low = np.mean(results[name_s]['low'])

        delta_t = total - baseline_total
        delta_h = high - baseline_high
        delta_l = low - baseline_low

        print(f"{name_s:<25} {total:.3f}       {high:.3f}       {low:.3f}       {delta_t:+.4f}     {delta_h:+.4f}     {delta_l:+.4f}")

    # Find best strategy
    print("\n" + "=" * 70)
    print("BEST STRATEGIES")
    print("=" * 70)

    best_total = max((np.mean(results[n]['total']), n) for n, _ in strategies[1:])
    best_high = max((np.mean(results[n]['high']), n) for n, _ in strategies[1:])
    best_low = max((np.mean(results[n]['low']), n) for n, _ in strategies[1:])

    print(f"\n  Best Overall:      {best_total[1]} ({best_total[0]:.3f} dB)")
    print(f"  Best High-Error:   {best_high[1]} ({best_high[0]:.3f} dB)")
    print(f"  Best Low-Error:    {best_low[1]} ({best_low[0]:.3f} dB)")

    # Find balanced strategy (good improvement without degrading low-error)
    print("\n  Balanced (improves total AND doesn't hurt low-error):")
    for name_s, _ in strategies[1:]:
        delta_t = np.mean(results[name_s]['total']) - baseline_total
        delta_l = np.mean(results[name_s]['low']) - baseline_low
        if delta_t > 0 and delta_l >= -0.02:  # Allow tiny degradation
            print(f"    ✓ {name_s}: Δ Total = {delta_t:+.4f}, Δ Low = {delta_l:+.4f}")

    return results


if __name__ == "__main__":
    results = evaluate_strategies()
