#!/usr/bin/env python3
"""
Region-Specific Evaluation with Predicate Verification

Measures:
1. Overall PSNR/SSIM improvement
2. PSNR in HIGH-FAILURE regions (where predicate flags errors)
3. PSNR in LOW-FAILURE regions (where predicate says OK)
4. Predicate satisfaction before vs after correction
5. Correlation between failure mask and actual error
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys
from typing import Dict, Tuple

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

        # Features
        int_s3, _ = self.local_stats(denoised, 3)
        int_s3_norm = self.normalize(int_s3)
        _, res_std_s3 = self.local_stats(residual, 3)
        _, res_std_s7 = self.local_stats(residual, 7)
        res_var_s3 = self.normalize(res_std_s3 ** 2)
        res_var_s7 = self.normalize(res_std_s7 ** 2)

        # Failure map
        failure = (
            0.70 * int_s3_norm +
            0.20 * (int_s3_norm * res_var_s7) +
            0.10 * res_var_s3
        ).clamp(0, 1)

        score = 1 - failure.mean()
        satisfied = score > 0.7

        return {
            'failure_map': failure,
            'score': score.item(),
            'satisfied': satisfied.item(),
        }


# =============================================================================
# SELF-CONSISTENCY DENOISER
# =============================================================================

def flip_h(x): return torch.flip(x, [-1])
def flip_v(x): return torch.flip(x, [-2])

AUGS = [
    (lambda x: x, lambda x: x),
    (flip_h, flip_h),
    (flip_v, flip_v),
    (lambda x: flip_h(flip_v(x)), lambda x: flip_h(flip_v(x))),
]

def self_consistency_denoise(backbone, noisy):
    """Denoise with self-consistency (4 augmentations)."""
    outputs = []
    for aug, inv_aug in AUGS:
        aug_out = backbone(aug(noisy)).clamp(0, 1)
        outputs.append(inv_aug(aug_out))

    stacked = torch.stack(outputs)
    consensus = stacked.mean(dim=0)
    uncertainty = stacked.std(dim=0)

    return consensus, uncertainty


# =============================================================================
# METRICS
# =============================================================================

def compute_psnr(pred, target, mask=None):
    """Compute PSNR, optionally in masked region only."""
    if mask is not None:
        # Only compute in masked region
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


def compute_correlation(x, y):
    """Pearson correlation."""
    x_flat, y_flat = x.flatten(), y.flatten()
    x_c, y_c = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    return ((x_c * y_c).sum() / (torch.sqrt((x_c**2).sum() * (y_c**2).sum()) + 1e-8)).item()


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    print("=" * 70)
    print("REGION-SPECIFIC EVALUATION WITH PREDICATE VERIFICATION")
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

    # Results storage
    results = {
        'baseline': {
            'psnr_total': [], 'psnr_high': [], 'psnr_low': [],
            'pred_score': [], 'pred_pass': [],
            'failure_corr': [],
        },
        'consistency': {
            'psnr_total': [], 'psnr_high': [], 'psnr_low': [],
            'pred_score': [], 'pred_pass': [],
            'uncertainty_corr': [],
        },
    }

    print("-" * 70)
    print("Per-Sample Results")
    print("-" * 70)

    with torch.no_grad():
        for i, sample in enumerate(samples):
            noisy = sample['noisy']
            clean = sample['clean']
            name = sample['name']

            # === BASELINE ===
            base = backbone(noisy).clamp(0, 1)
            base_pred = predicate(noisy, base)
            actual_error = (base - clean).abs()

            # Create masks: top 30% error = high-failure region
            error_threshold = torch.quantile(actual_error, 0.7)
            high_error_mask = actual_error > error_threshold
            low_error_mask = ~high_error_mask

            # Also use predicate's failure map
            failure_threshold = torch.quantile(base_pred['failure_map'], 0.7)
            high_failure_mask = base_pred['failure_map'] > failure_threshold

            # Metrics for baseline
            psnr_base_total = compute_psnr(base, clean)
            psnr_base_high = compute_psnr(base, clean, high_error_mask)
            psnr_base_low = compute_psnr(base, clean, low_error_mask)
            failure_corr = compute_correlation(base_pred['failure_map'], actual_error)

            results['baseline']['psnr_total'].append(psnr_base_total)
            results['baseline']['psnr_high'].append(psnr_base_high)
            results['baseline']['psnr_low'].append(psnr_base_low)
            results['baseline']['pred_score'].append(base_pred['score'])
            results['baseline']['pred_pass'].append(base_pred['satisfied'])
            results['baseline']['failure_corr'].append(failure_corr)

            # === SELF-CONSISTENCY ===
            sc_out, uncertainty = self_consistency_denoise(backbone, noisy)
            sc_pred = predicate(noisy, sc_out)
            sc_error = (sc_out - clean).abs()

            # Metrics for self-consistency
            psnr_sc_total = compute_psnr(sc_out, clean)
            psnr_sc_high = compute_psnr(sc_out, clean, high_error_mask)  # Same mask as baseline for fair comparison
            psnr_sc_low = compute_psnr(sc_out, clean, low_error_mask)
            uncertainty_corr = compute_correlation(uncertainty, actual_error)

            results['consistency']['psnr_total'].append(psnr_sc_total)
            results['consistency']['psnr_high'].append(psnr_sc_high)
            results['consistency']['psnr_low'].append(psnr_sc_low)
            results['consistency']['pred_score'].append(sc_pred['score'])
            results['consistency']['pred_pass'].append(sc_pred['satisfied'])
            results['consistency']['uncertainty_corr'].append(uncertainty_corr)

            # Print per-sample
            print(f"\n[{i+1}] {name}")
            print(f"  Baseline:    PSNR={psnr_base_total:.2f} (High={psnr_base_high:.2f}, Low={psnr_base_low:.2f})")
            print(f"               Pred Score={base_pred['score']:.3f}, Pass={base_pred['satisfied']}")
            print(f"               Failure-Error Corr={failure_corr:.3f}")
            print(f"  Consistency: PSNR={psnr_sc_total:.2f} (High={psnr_sc_high:.2f}, Low={psnr_sc_low:.2f})")
            print(f"               Pred Score={sc_pred['score']:.3f}, Pass={sc_pred['satisfied']}")
            print(f"               Uncert-Error Corr={uncertainty_corr:.3f}")
            delta = psnr_sc_total - psnr_base_total
            delta_high = psnr_sc_high - psnr_base_high
            print(f"  Δ Total: {delta:+.3f} dB, Δ High-Error Region: {delta_high:+.3f} dB")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Baseline':<15} {'Consistency':<15} {'Δ':<10}")
    print("-" * 70)

    metrics = [
        ('PSNR Total (dB)', 'psnr_total'),
        ('PSNR High-Error (dB)', 'psnr_high'),
        ('PSNR Low-Error (dB)', 'psnr_low'),
        ('Predicate Score', 'pred_score'),
    ]

    for label, key in metrics:
        base_val = np.mean(results['baseline'][key])
        sc_val = np.mean(results['consistency'][key])
        delta = sc_val - base_val
        print(f"{label:<30} {base_val:<15.3f} {sc_val:<15.3f} {delta:+.4f}")

    # Predicate pass rate
    base_pass = sum(results['baseline']['pred_pass']) / len(samples)
    sc_pass = sum(results['consistency']['pred_pass']) / len(samples)
    print(f"{'Predicate Pass Rate':<30} {base_pass*100:<14.1f}% {sc_pass*100:<14.1f}% {(sc_pass-base_pass)*100:+.1f}%")

    # Correlation quality
    print(f"\n{'Failure Map ↔ Error Corr':<30} {np.mean(results['baseline']['failure_corr']):.3f}")
    print(f"{'Uncertainty ↔ Error Corr':<30} {np.mean(results['consistency']['uncertainty_corr']):.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    delta_total = np.mean(results['consistency']['psnr_total']) - np.mean(results['baseline']['psnr_total'])
    delta_high = np.mean(results['consistency']['psnr_high']) - np.mean(results['baseline']['psnr_high'])

    if delta_total > 0:
        print(f"\n✓ Overall improvement: {delta_total:+.4f} dB")
    else:
        print(f"\n✗ No overall improvement: {delta_total:+.4f} dB")

    if delta_high > delta_total:
        print(f"✓ Higher improvement in high-error regions: {delta_high:+.4f} dB")
        print("  → Correction is targeting the right areas")
    else:
        print(f"⚠ Lower improvement in high-error regions: {delta_high:+.4f} dB")

    if sc_pass > base_pass:
        print(f"✓ Predicate pass rate improved: {base_pass*100:.0f}% → {sc_pass*100:.0f}%")

    return results


if __name__ == "__main__":
    results = main()
