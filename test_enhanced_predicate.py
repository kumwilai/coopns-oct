#!/usr/bin/env python3
"""
Test Enhanced Structure Predicate v13

Compare multi-scale features against original v12.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys
import time

sys.path.insert(0, 'nsnd_oct')

from symbolic_predicates import (
    StructurePreservedPredicate,
    EnhancedStructurePredicate,
)


def pearson_correlation(x, y):
    """Compute Pearson correlation."""
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_mean = x_flat.mean()
    y_mean = y_flat.mean()
    num = ((x_flat - x_mean) * (y_flat - y_mean)).sum()
    den = torch.sqrt(((x_flat - x_mean)**2).sum() * ((y_flat - y_mean)**2).sum())
    return (num / (den + 1e-8)).item()


def load_models():
    """Load NAFNet backbone."""
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet

    backbone = NAFNet(
        img_channel=1, width=64, middle_blk_num=2,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
    )
    ckpt = torch.load("/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth",
                      map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt['state_dict'], strict=False)
    backbone.eval()
    return backbone


def load_samples(n_samples=10):
    """Load validation samples."""
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    val_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    all_lines = open(val_jsonl).readlines()
    step = max(1, len(all_lines) // n_samples)
    indices = list(range(0, len(all_lines), step))[:n_samples]

    samples = []
    for idx in indices:
        data = json.loads(all_lines[idx])
        clean = torch.from_numpy(np.array(Image.open(data['clean_path']).convert('L'))).float() / 255.0
        noisy = torch.from_numpy(np.array(Image.open(data['noisy_path']).convert('L'))).float() / 255.0
        samples.append({
            'clean': clean.unsqueeze(0).unsqueeze(0),
            'noisy': noisy.unsqueeze(0).unsqueeze(0),
            'name': Path(data['clean_path']).stem,
        })
    return samples


def evaluate_predicate(predicate, samples, backbone, name="Predicate", needs_boundaries=False):
    """Evaluate a predicate on samples."""
    correlations = []

    print(f"\nEvaluating {name}...")

    with torch.no_grad():
        for i, sample in enumerate(samples):
            noisy = sample['noisy']
            clean = sample['clean']
            B, _, H, W = noisy.shape

            # Denoise
            denoised = backbone(noisy).clamp(0, 1)

            # Get structure failure map
            if needs_boundaries:
                # Create dummy boundaries for original predicate
                boundaries = torch.tensor([[0.15, 0.30, 0.50, 0.70]]).unsqueeze(-1).expand(B, 4, W)
                result = predicate(noisy, denoised, boundaries)
            else:
                result = predicate(noisy, denoised)
            structure_failure = result['structure_failure']

            # Actual error
            actual_error = (denoised - clean).abs()

            # Resize if needed
            if structure_failure.shape != actual_error.shape:
                structure_failure = F.interpolate(
                    structure_failure, size=actual_error.shape[2:],
                    mode='bilinear', align_corners=False
                )

            # Correlation
            corr = pearson_correlation(structure_failure, actual_error)
            correlations.append(corr)

            if i < 3:  # Show first 3
                print(f"  Sample {i+1}: correlation = {corr:.4f}")

    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    print(f"  Mean correlation: {mean_corr:.4f} ± {std_corr:.4f}")

    return mean_corr, std_corr, correlations


def main():
    print("=" * 70)
    print("ENHANCED STRUCTURE PREDICATE EVALUATION")
    print("=" * 70)

    # Load models and data
    print("\nLoading models...")
    backbone = load_models()

    print("Loading samples...")
    samples = load_samples(n_samples=10)
    print(f"  Loaded {len(samples)} samples")

    # Original predicate (v12)
    print("\n" + "-" * 70)
    print("Original Structure Predicate (v12)")
    print("-" * 70)
    p_orig = StructurePreservedPredicate(learnable_weights=False)
    corr_orig, _, _ = evaluate_predicate(p_orig, samples, backbone, "v12 Original", needs_boundaries=True)

    # Enhanced predicate (v13) - multi-scale
    print("\n" + "-" * 70)
    print("Enhanced Structure Predicate (v13) - Multi-Scale")
    print("-" * 70)
    p_enhanced = EnhancedStructurePredicate(
        scales=[3, 7, 15],
        use_learned_combiner=False,
        learnable_weights=False,
    )
    corr_enhanced, _, _ = evaluate_predicate(p_enhanced, samples, backbone, "v13 Multi-Scale")

    # Enhanced predicate with calibration
    print("\n" + "-" * 70)
    print("Enhanced Structure Predicate (v13) - Calibrated")
    print("-" * 70)
    p_calibrated = EnhancedStructurePredicate(
        scales=[3, 7, 15],
        use_learned_combiner=False,
        learnable_weights=True,
    )

    # Prepare calibration data
    print("  Calibrating on first 5 samples...")
    noisy_list = [s['noisy'] for s in samples[:5]]
    clean_list = [s['clean'] for s in samples[:5]]

    with torch.no_grad():
        denoised_list = [backbone(n).clamp(0, 1) for n in noisy_list]

    # Enable gradients for calibration
    calib_result = p_calibrated.calibrate(
        noisy_list, denoised_list, clean_list,
        n_steps=100, lr=0.1
    )

    # Evaluate calibrated
    corr_calibrated, _, _ = evaluate_predicate(p_calibrated, samples, backbone, "v13 Calibrated")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  v12 Original:    {corr_orig:.4f}")
    print(f"  v13 Multi-Scale: {corr_enhanced:.4f} ({(corr_enhanced-corr_orig)*100:+.1f}%)")
    print(f"  v13 Calibrated:  {corr_calibrated:.4f} ({(corr_calibrated-corr_orig)*100:+.1f}%)")

    if corr_calibrated > 0.4:
        print("\n  ✓ ACHIEVED > 0.4 correlation!")
    elif corr_calibrated > 0.35:
        print("\n  ⚠ Good progress, but < 0.4. Try learned combiner.")
    else:
        print("\n  ✗ Need more improvements.")

    # Show learned weights
    print("\n  Calibrated weights (top 5):")
    weights = p_calibrated.get_weight_summary()
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, w in sorted_weights:
        print(f"    {name}: {w:.4f}")

    return corr_calibrated


if __name__ == "__main__":
    final_corr = main()
