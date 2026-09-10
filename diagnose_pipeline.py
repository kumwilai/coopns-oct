#!/usr/bin/env python3
"""
Diagnostic script to validate pipeline assumptions before adding complexity.

Questions to answer:
1. Do failure maps correlate with actual errors?
2. Is there headroom above NAFNet?
3. Can correctors help if given oracle failure maps?
4. What do failure regions actually look like?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys

# Import pipeline
from neuro_symbolic_parallel import (
    NeuroSymbolicParallel, PipelineConfig,
    SpecklePredicate, StructurePredicate, AnatomyPredicate
)
from parallel_correctors import ParallelCorrectors, CorrectorConfig


def pearson_correlation(x, y):
    """Compute Pearson correlation between two tensors."""
    x_flat = x.flatten()
    y_flat = y.flatten()

    x_mean = x_flat.mean()
    y_mean = y_flat.mean()

    num = ((x_flat - x_mean) * (y_flat - y_mean)).sum()
    den = torch.sqrt(((x_flat - x_mean)**2).sum() * ((y_flat - y_mean)**2).sum())

    return (num / (den + 1e-8)).item()


def load_sample_batch(n_samples=5):
    """Load multiple samples for robust statistics."""
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    val_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    clean_list = []
    noisy_list = []

    with open(val_jsonl) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data = json.loads(line)

            clean = Image.open(data['clean_path']).convert('L')
            noisy = Image.open(data['noisy_path']).convert('L')

            clean = torch.from_numpy(np.array(clean)).float() / 255.0
            noisy = torch.from_numpy(np.array(noisy)).float() / 255.0

            clean_list.append(clean)
            noisy_list.append(noisy)

    clean_batch = torch.stack(clean_list).unsqueeze(1)
    noisy_batch = torch.stack(noisy_list).unsqueeze(1)

    return noisy_batch, clean_batch


def load_models():
    """Load backbone and boundary models."""
    device = torch.device('cpu')

    # NAFNet
    sys.path.insert(0, 'nsnd_oct')
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    backbone = NAFNet(
        img_channel=1,
        width=64,
        middle_blk_num=2,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
    )
    ckpt_path = "/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt['state_dict'], strict=False)
    backbone.eval()

    # Boundary model
    from physics_enhanced_v3 import PhysicsEnsembleV3
    boundary_model = PhysicsEnsembleV3(
        in_channels=1,
        hidden_channels=48,
        num_boundaries=4,
    )
    ckpt_path = "/home/kumwilai/OCT/best_boundary_model_v4.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    boundary_model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    boundary_model.eval()

    return backbone, boundary_model


def diagnose_failure_map_correlation(noisy, clean, denoised, failure_maps):
    """
    Diagnostic 1: Do failure maps correlate with actual errors?

    If correlation is low, predicates are identifying the WRONG regions.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: Failure Map vs Actual Error Correlation")
    print("="*70)

    # Actual error map
    error_map = (denoised - clean).abs()

    print(f"\nError map stats: min={error_map.min():.4f}, max={error_map.max():.4f}, mean={error_map.mean():.4f}")

    # Correlation for each failure type
    for name, fmap in failure_maps.items():
        corr = pearson_correlation(fmap, error_map)
        coverage = fmap.mean().item() * 100

        # Also check: in high-failure regions, is error actually high?
        high_fail_mask = fmap > 0.5
        if high_fail_mask.sum() > 0:
            error_in_fail = error_map[high_fail_mask].mean().item()
            error_outside = error_map[~high_fail_mask].mean().item()
            ratio = error_in_fail / (error_outside + 1e-8)
        else:
            ratio = 0

        status = "✓ GOOD" if corr > 0.3 else ("⚠ WEAK" if corr > 0.1 else "✗ BAD")

        print(f"\n{name}:")
        print(f"  Correlation with error: {corr:.4f} {status}")
        print(f"  Coverage: {coverage:.1f}% of image")
        print(f"  Error in failure regions: {error_in_fail:.4f}")
        print(f"  Error outside failure: {error_outside:.4f}")
        print(f"  Ratio (should be >1): {ratio:.2f}")

    # Combined analysis
    combined_fail = sum(failure_maps.values())
    combined_fail = combined_fail / len(failure_maps)
    combined_corr = pearson_correlation(combined_fail, error_map)
    print(f"\nCombined failure map correlation: {combined_corr:.4f}")

    return combined_corr


def diagnose_headroom(noisy, clean, denoised):
    """
    Diagnostic 2: Is there headroom above NAFNet?

    If NAFNet is already near-optimal, no corrector can help.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: Performance Headroom Analysis")
    print("="*70)

    # NAFNet PSNR
    mse_nafnet = F.mse_loss(denoised, clean)
    psnr_nafnet = 10 * torch.log10(1.0 / mse_nafnet).item()

    # Noisy PSNR (baseline)
    mse_noisy = F.mse_loss(noisy, clean)
    psnr_noisy = 10 * torch.log10(1.0 / mse_noisy).item()

    # Theoretical limit: if we could perfectly fix error regions
    error_map = (denoised - clean).abs()

    # What if we replaced high-error regions with ground truth?
    thresholds = [0.9, 0.75, 0.5, 0.25, 0.1]
    print(f"\nNoisy PSNR (baseline): {psnr_noisy:.2f} dB")
    print(f"NAFNet PSNR (current): {psnr_nafnet:.2f} dB")
    print(f"NAFNet improvement over noisy: +{psnr_nafnet - psnr_noisy:.2f} dB")

    print(f"\nOracle correction potential (if we perfectly fixed top-X% errors):")

    for pct in [50, 25, 10, 5, 1]:
        threshold = torch.quantile(error_map, 1 - pct/100)
        oracle_mask = error_map > threshold

        # Oracle correction: replace high-error pixels with ground truth
        oracle_corrected = denoised.clone()
        oracle_corrected[oracle_mask] = clean[oracle_mask]

        mse_oracle = F.mse_loss(oracle_corrected, clean)
        psnr_oracle = 10 * torch.log10(1.0 / mse_oracle).item()
        gain = psnr_oracle - psnr_nafnet

        print(f"  Fix top {pct:2d}% errors → {psnr_oracle:.2f} dB (+{gain:.2f} dB)")

    # Perfect correction
    print(f"  Fix 100% errors → ∞ dB (ground truth)")

    print(f"\nConclusion: ", end="")
    if psnr_nafnet > 35:
        print("NAFNet already very good. Limited headroom for correction.")
    elif psnr_nafnet > 30:
        print("Moderate headroom exists. Correction could help ~1-3 dB.")
    else:
        print("Significant headroom. Correction could help substantially.")

    return psnr_nafnet


def diagnose_corrector_capacity(noisy, clean, denoised, failure_maps):
    """
    Diagnostic 3: Can correctors produce meaningful output?

    Check if correctors output anything non-zero.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Corrector Output Analysis")
    print("="*70)

    config = CorrectorConfig()
    correctors = ParallelCorrectors(config)
    correctors.eval()

    with torch.no_grad():
        result = correctors(denoised, noisy, failure_maps)

    print(f"\nCorrectors initialized with scale={config.correction_scale}")

    for name in ['residual_speckle', 'residual_anatomy', 'residual_structure']:
        r = result[name]
        print(f"\n{name}:")
        print(f"  Min: {r.min().item():.6f}")
        print(f"  Max: {r.max().item():.6f}")
        print(f"  Mean: {r.mean().item():.6f}")
        print(f"  Std: {r.std().item():.6f}")
        print(f"  Non-zero pixels: {(r.abs() > 1e-6).sum().item()}")

        if r.abs().max() < 1e-5:
            print(f"  ✗ WARNING: Output is effectively zero!")

    combined = result['combined_residual']
    print(f"\nCombined residual:")
    print(f"  Range: [{combined.min().item():.6f}, {combined.max().item():.6f}]")

    # What PSNR change does this correction produce?
    corrected = result['corrected']
    mse_before = F.mse_loss(denoised, clean)
    mse_after = F.mse_loss(corrected, clean)
    psnr_before = 10 * torch.log10(1.0 / mse_before).item()
    psnr_after = 10 * torch.log10(1.0 / mse_after).item()

    print(f"\nPSNR before correction: {psnr_before:.4f} dB")
    print(f"PSNR after correction:  {psnr_after:.4f} dB")
    print(f"Change: {psnr_after - psnr_before:+.4f} dB")

    if abs(psnr_after - psnr_before) < 0.01:
        print("✗ Correctors have NO EFFECT on output!")

    return result


def diagnose_oracle_correction(noisy, clean, denoised):
    """
    Diagnostic 4: Oracle test - if corrector knew EXACTLY where errors are,
    and applied PERFECT correction there, what's the max improvement?

    This tells us if the masked correction approach is fundamentally sound.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: Oracle Correction Test")
    print("="*70)

    error_map = (denoised - clean).abs()

    # Create oracle failure map (actual high-error regions)
    oracle_threshold = error_map.mean() + error_map.std()
    oracle_mask = (error_map > oracle_threshold).float()

    print(f"\nOracle mask coverage: {oracle_mask.mean().item()*100:.1f}%")

    # Test: simple learned correction in oracle regions
    # Simulate what a "perfect" corrector would do

    # Strategy 1: Blend toward clean in oracle regions
    blend_factors = [0.1, 0.3, 0.5, 0.7, 1.0]

    print(f"\nOracle blending test (replace errors with partial ground truth):")

    mse_original = F.mse_loss(denoised, clean)
    psnr_original = 10 * torch.log10(1.0 / mse_original).item()

    for alpha in blend_factors:
        # In oracle regions, blend toward ground truth
        corrected = denoised.clone()
        correction = alpha * (clean - denoised) * oracle_mask
        corrected = denoised + correction

        mse_corrected = F.mse_loss(corrected, clean)
        psnr_corrected = 10 * torch.log10(1.0 / mse_corrected).item()

        print(f"  α={alpha:.1f}: {psnr_corrected:.2f} dB (+{psnr_corrected - psnr_original:.2f} dB)")

    print(f"\nThis shows the MAXIMUM possible improvement if corrector")
    print(f"could perfectly identify and fix error regions.")

    return oracle_mask


def diagnose_predicate_thresholds(noisy, clean, denoised):
    """
    Diagnostic 5: Are predicate thresholds appropriate for this data?
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 5: Predicate Threshold Analysis")
    print("="*70)

    # Speckle analysis
    print("\n--- Speckle Predicate ---")
    residual = noisy - denoised

    # Compute actual CV statistics
    window = 15
    box_filter = torch.ones(1, 1, window, window) / (window * window)

    local_mean = F.conv2d(denoised, box_filter, padding=window//2)
    local_mean = torch.clamp(local_mean, min=1e-6)

    res_sq = residual ** 2
    local_var = F.conv2d(res_sq, box_filter, padding=window//2)
    local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))

    cv_map = local_std / local_mean

    print(f"Actual CV statistics:")
    print(f"  Min: {cv_map.min().item():.4f}")
    print(f"  Max: {cv_map.max().item():.4f}")
    print(f"  Mean: {cv_map.mean().item():.4f}")
    print(f"  Std: {cv_map.std().item():.4f}")
    print(f"  Median: {cv_map.median().item():.4f}")
    print(f"\nHardcoded expected CV: 0.40, tolerance: 0.14")
    print(f"Suggested CV: {cv_map.mean().item():.3f} ± {cv_map.std().item():.3f}")

    # Structure analysis
    print("\n--- Structure Predicate ---")
    sobel_x = torch.tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ], dtype=torch.float32).unsqueeze(0) / 4.0
    sobel_y = torch.tensor([
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ], dtype=torch.float32).unsqueeze(0) / 4.0

    edge_noisy = torch.sqrt(F.conv2d(noisy, sobel_x, padding=1)**2 +
                            F.conv2d(noisy, sobel_y, padding=1)**2 + 1e-8)
    edge_denoised = torch.sqrt(F.conv2d(denoised, sobel_x, padding=1)**2 +
                               F.conv2d(denoised, sobel_y, padding=1)**2 + 1e-8)

    # Simple correlation
    corr = pearson_correlation(edge_noisy, edge_denoised)
    print(f"Global edge correlation: {corr:.4f}")
    print(f"Hardcoded threshold: 0.50")
    print(f"Suggested threshold: {corr - 0.1:.3f} (to catch ~10% as failures)")


def visualize_diagnostics(noisy, clean, denoised, failure_maps, error_map):
    """Save diagnostic visualizations."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Images
    axes[0, 0].imshow(noisy[0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title('Noisy Input')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised[0, 0].numpy(), cmap='gray')
    axes[0, 1].set_title('NAFNet Output')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(clean[0, 0].numpy(), cmap='gray')
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(error_map[0, 0].numpy(), cmap='hot')
    axes[0, 3].set_title('Actual Error Map')
    axes[0, 3].axis('off')

    # Row 2: Failure maps
    axes[1, 0].imshow(failure_maps['speckle'][0, 0].numpy(), cmap='hot')
    axes[1, 0].set_title(f"Speckle Failure ({failure_maps['speckle'].mean()*100:.1f}%)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(failure_maps['anatomy'][0, 0].numpy(), cmap='hot')
    axes[1, 1].set_title(f"Anatomy Failure ({failure_maps['anatomy'].mean()*100:.1f}%)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(failure_maps['structure'][0, 0].numpy(), cmap='hot')
    axes[1, 2].set_title(f"Structure Failure ({failure_maps['structure'].mean()*100:.1f}%)")
    axes[1, 2].axis('off')

    # Overlay: error vs combined failure
    combined_fail = (failure_maps['speckle'] + failure_maps['structure']) / 2
    axes[1, 3].imshow(denoised[0, 0].numpy(), cmap='gray')
    axes[1, 3].imshow(error_map[0, 0].numpy(), cmap='Reds', alpha=0.5)
    axes[1, 3].contour(combined_fail[0, 0].numpy(), levels=[0.5], colors='cyan', linewidths=1)
    axes[1, 3].set_title('Error (red) vs Failure boundary (cyan)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('diagnostic_visualization.png', dpi=150)
    print(f"\nVisualization saved to: diagnostic_visualization.png")
    plt.close()


def main():
    print("="*70)
    print("PIPELINE DIAGNOSTIC: Validating Assumptions")
    print("="*70)

    # Load data
    print("\nLoading data...")
    noisy, clean = load_sample_batch(n_samples=1)  # Start with 1 for speed
    print(f"Data shape: {noisy.shape}")

    # Load models
    print("Loading models...")
    backbone, boundary_model = load_models()

    # Run backbone
    print("Running NAFNet...")
    with torch.no_grad():
        denoised = backbone(noisy)
        denoised = torch.clamp(denoised, 0, 1)

    # Create pipeline for predicates
    config = PipelineConfig()
    pipeline = NeuroSymbolicParallel(backbone, boundary_model, config)
    pipeline.eval()

    # Get failure maps
    print("Computing failure maps...")
    with torch.no_grad():
        boundaries = pipeline.get_boundaries(denoised)
        pred_result = pipeline.evaluate_predicates(noisy, denoised, boundaries)

    failure_maps = pred_result['failure_maps']
    error_map = (denoised - clean).abs()

    # Run diagnostics
    corr = diagnose_failure_map_correlation(noisy, clean, denoised, failure_maps)
    psnr = diagnose_headroom(noisy, clean, denoised)
    corrector_result = diagnose_corrector_capacity(noisy, clean, denoised, failure_maps)
    oracle_mask = diagnose_oracle_correction(noisy, clean, denoised)
    diagnose_predicate_thresholds(noisy, clean, denoised)

    # Visualization
    visualize_diagnostics(noisy, clean, denoised, failure_maps, error_map)

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print("\n1. FAILURE MAP QUALITY:")
    if corr > 0.3:
        print(f"   ✓ Good correlation ({corr:.3f}) - predicates identify real errors")
    elif corr > 0.1:
        print(f"   ⚠ Weak correlation ({corr:.3f}) - predicates partially useful")
    else:
        print(f"   ✗ Poor correlation ({corr:.3f}) - predicates identify WRONG regions!")

    print("\n2. HEADROOM:")
    if psnr < 30:
        print(f"   ✓ Significant headroom ({psnr:.1f} dB) - correction can help")
    elif psnr < 35:
        print(f"   ⚠ Moderate headroom ({psnr:.1f} dB) - some room for improvement")
    else:
        print(f"   ✗ Limited headroom ({psnr:.1f} dB) - NAFNet already excellent")

    print("\n3. CORRECTOR OUTPUT:")
    combined = corrector_result['combined_residual']
    if combined.abs().max() < 1e-5:
        print(f"   ✗ Correctors output ZERO - initialization problem!")
    else:
        print(f"   ✓ Correctors produce non-zero output")

    print("\n4. RECOMMENDED ACTIONS:")
    actions = []

    if corr < 0.2:
        actions.append("- Fix predicates: current ones don't match actual errors")
    if combined.abs().max() < 1e-5:
        actions.append("- Fix corrector initialization: outputs are zero")
    if psnr > 35:
        actions.append("- Consider: NAFNet may already be optimal for this data")

    if not actions:
        actions.append("- Core assumptions valid, proceed with improvements")

    for action in actions:
        print(f"   {action}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
