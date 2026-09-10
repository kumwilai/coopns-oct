#!/usr/bin/env python3
"""
Evaluation Script: Test if Symbolic Predicates Correctly Identify Pass/Fail Regions

Hypothesis: Symbolic predicates (speckle, anatomy, structure) can identify
regions where denoising fails WITHOUT needing ground truth.

Test:
1. Run NAFNet on validation images
2. Compute predicate satisfaction maps
3. Compare with actual error maps (denoised vs clean)
4. Measure correlation: do predicates identify real failure regions?

Success criteria:
- High correlation between predicate failure and actual error
- Predicates should flag high-error regions as "failed"
- Predicates should pass low-error regions as "satisfied"
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys
import time
from typing import Dict, Tuple

# Import our modules
from symbolic_predicates import (
    VerifiableDenoisingPredicate,
    SpeckleFidelityPredicate,
    AnatomyValidPredicate,
    StructurePreservedPredicate,
    FuzzyLogic,
)


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation between two tensors."""
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()

    x_mean = x_flat.mean()
    y_mean = y_flat.mean()

    num = ((x_flat - x_mean) * (y_flat - y_mean)).sum()
    den = torch.sqrt(((x_flat - x_mean)**2).sum() * ((y_flat - y_mean)**2).sum())

    return (num / (den + 1e-8)).item()


def compute_metrics(pred_failure: torch.Tensor, actual_error: torch.Tensor,
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics for predicate evaluation.

    Args:
        pred_failure: Predicted failure map from predicate [0, 1]
        actual_error: Actual error map (|denoised - clean|)
        threshold: Threshold to binarize failure prediction

    Returns:
        Dict with precision, recall, F1, correlation
    """
    # Normalize actual error to [0, 1]
    error_norm = actual_error / (actual_error.max() + 1e-8)

    # Binarize predictions
    pred_binary = (pred_failure > threshold).float()

    # Define "actual failure" as top 20% error regions
    error_threshold = torch.quantile(error_norm.flatten(), 0.8)
    actual_binary = (error_norm > error_threshold).float()

    # Compute metrics
    tp = (pred_binary * actual_binary).sum()
    fp = (pred_binary * (1 - actual_binary)).sum()
    fn = ((1 - pred_binary) * actual_binary).sum()
    tn = ((1 - pred_binary) * (1 - actual_binary)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Correlation
    corr = pearson_correlation(pred_failure, error_norm)

    # IoU (Intersection over Union)
    intersection = (pred_binary * actual_binary).sum()
    union = pred_binary.sum() + actual_binary.sum() - intersection
    iou = intersection / (union + 1e-8)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'correlation': corr,
        'iou': iou.item(),
        'pred_coverage': pred_binary.mean().item(),
        'actual_coverage': actual_binary.mean().item(),
    }


def load_models():
    """Load NAFNet backbone and boundary model."""
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
    print(f"  NAFNet loaded: PSNR={ckpt.get('psnr', 'unknown')}")

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
    print("  Boundary model loaded")

    return backbone, boundary_model


def load_validation_data(n_samples: int = 10):
    """Load validation samples."""
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    val_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    # Read all lines first to get diverse samples
    all_lines = []
    with open(val_jsonl) as f:
        all_lines = f.readlines()

    # Sample evenly across the dataset
    total = len(all_lines)
    step = max(1, total // n_samples)
    indices = list(range(0, total, step))[:n_samples]

    samples = []
    for idx in indices:
        data = json.loads(all_lines[idx])

        clean = Image.open(data['clean_path']).convert('L')
        noisy = Image.open(data['noisy_path']).convert('L')

        clean = torch.from_numpy(np.array(clean)).float() / 255.0
        noisy = torch.from_numpy(np.array(noisy)).float() / 255.0

        samples.append({
            'clean': clean.unsqueeze(0).unsqueeze(0),
            'noisy': noisy.unsqueeze(0).unsqueeze(0),
            'name': Path(data['clean_path']).stem,
            'index': idx,
        })

    return samples


def get_boundaries(boundary_model, image: torch.Tensor) -> torch.Tensor:
    """Extract boundaries from image."""
    with torch.no_grad():
        out = boundary_model(image)
        if isinstance(out, dict):
            boundaries = out.get('boundaries', out.get('boundary_positions'))
        else:
            boundaries = out
    return boundaries


def evaluate_single_image(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    backbone,
    boundary_model,
    verifier: VerifiableDenoisingPredicate,
) -> Dict:
    """Evaluate predicates on a single image."""

    with torch.no_grad():
        # Denoise
        denoised = backbone(noisy)
        denoised = torch.clamp(denoised, 0, 1)

        # Get boundaries
        boundaries = get_boundaries(boundary_model, denoised)

        # Compute actual error
        actual_error = (denoised - clean).abs()

        # Run predicates with details
        p1_result = verifier.P1_speckle(noisy, denoised, return_details=True)
        p2_result = verifier.P2_anatomy(boundaries, return_details=True)
        p3_result = verifier.P3_structure(noisy, denoised, boundaries, return_details=True)

        # Get failure maps
        # For speckle: invert score to get failure
        cv_map = p1_result.get('cv_map', torch.zeros_like(denoised))
        expected_cv = verifier.P1_speckle.expected_cv
        tolerance = verifier.P1_speckle.tolerance
        speckle_failure = 1 - FuzzyLogic.soft_in_range(cv_map, expected_cv - tolerance, expected_cv + tolerance)

        # Resize to match denoised shape if needed
        if speckle_failure.shape != denoised.shape:
            speckle_failure = F.interpolate(speckle_failure, size=denoised.shape[2:], mode='bilinear', align_corners=False)

        # For structure: use structure_failure (residual-based detection)
        if 'structure_failure' in p3_result:
            structure_failure = p3_result['structure_failure']
            if structure_failure.shape != denoised.shape:
                structure_failure = F.interpolate(structure_failure, size=denoised.shape[2:], mode='bilinear', align_corners=False)
        elif 'smoothness_failure' in p3_result:
            structure_failure = p3_result['smoothness_failure']
            if structure_failure.shape != denoised.shape:
                structure_failure = F.interpolate(structure_failure, size=denoised.shape[2:], mode='bilinear', align_corners=False)
        else:
            # Fallback
            structure_failure = torch.zeros_like(denoised)

        # Compute PSNR
        mse = F.mse_loss(denoised, clean)
        psnr = 10 * torch.log10(1.0 / mse).item()

    return {
        'denoised': denoised,
        'actual_error': actual_error,
        'psnr': psnr,
        'speckle_failure': speckle_failure,
        'structure_failure': structure_failure,
        'boundaries': boundaries,
        'p1_result': p1_result,
        'p2_result': p2_result,
        'p3_result': p3_result,
    }


def run_evaluation(n_samples: int = 10):
    """Run full evaluation."""
    print("=" * 70)
    print("SYMBOLIC PREDICATE EVALUATION")
    print("Testing if predicates correctly identify failure regions")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    backbone, boundary_model = load_models()

    # Create verifier
    verifier = VerifiableDenoisingPredicate()

    # Load data
    print(f"\nLoading {n_samples} validation samples...")
    samples = load_validation_data(n_samples)
    print(f"  Loaded {len(samples)} samples")

    # Collect results
    all_results = {
        'speckle': {'correlation': [], 'f1': [], 'precision': [], 'recall': []},
        'structure': {'correlation': [], 'f1': [], 'precision': [], 'recall': []},
        'psnr': [],
    }

    print("\n" + "-" * 70)
    print("Evaluating samples...")
    print("-" * 70)

    for i, sample in enumerate(samples):
        noisy = sample['noisy']
        clean = sample['clean']
        name = sample['name']

        start_time = time.time()
        result = evaluate_single_image(noisy, clean, backbone, boundary_model, verifier)
        elapsed = time.time() - start_time

        # Compute metrics for speckle predicate
        speckle_metrics = compute_metrics(
            result['speckle_failure'].squeeze(),
            result['actual_error'].squeeze()
        )

        # Compute metrics for structure predicate
        structure_metrics = compute_metrics(
            result['structure_failure'].squeeze(),
            result['actual_error'].squeeze()
        )

        # Store results
        all_results['psnr'].append(result['psnr'])
        for key in ['correlation', 'f1', 'precision', 'recall']:
            all_results['speckle'][key].append(speckle_metrics[key])
            all_results['structure'][key].append(structure_metrics[key])

        # Print per-sample results
        idx = sample.get('index', i)
        print(f"\n[{i+1}/{len(samples)}] {name} (idx={idx}, {elapsed:.2f}s)")
        print(f"  PSNR: {result['psnr']:.2f} dB")
        print(f"  Speckle: corr={speckle_metrics['correlation']:.3f}, F1={speckle_metrics['f1']:.3f}")
        print(f"  Structure: corr={structure_metrics['correlation']:.3f}, F1={structure_metrics['f1']:.3f}")
        print(f"  P1 satisfied: {result['p1_result']['satisfied'].item()}, score: {result['p1_result']['score'].item():.3f}")
        print(f"  P2 satisfied: {result['p2_result']['satisfied'].item()}, score: {result['p2_result']['score'].item():.3f}")
        print(f"  P3 satisfied: {result['p3_result']['satisfied'].item()}, score: {result['p3_result']['score'].item():.3f}")
        # New structure predicate metrics
        if 'smoothness_score' in result['p3_result']:
            print(f"      Smoothness: {result['p3_result']['smoothness_score'].item():.3f}, "
                  f"NoiseRed: {result['p3_result']['noise_reduction_score'].item():.3f}, "
                  f"EdgeQual: {result['p3_result']['edge_quality_score'].item():.3f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print(f"\nAverage PSNR: {np.mean(all_results['psnr']):.2f} ± {np.std(all_results['psnr']):.2f} dB")

    print("\nSpeckle Predicate (P1):")
    print(f"  Correlation with error: {np.mean(all_results['speckle']['correlation']):.4f} ± {np.std(all_results['speckle']['correlation']):.4f}")
    print(f"  F1 Score: {np.mean(all_results['speckle']['f1']):.4f} ± {np.std(all_results['speckle']['f1']):.4f}")
    print(f"  Precision: {np.mean(all_results['speckle']['precision']):.4f}")
    print(f"  Recall: {np.mean(all_results['speckle']['recall']):.4f}")

    print("\nStructure Predicate (P3):")
    print(f"  Correlation with error: {np.mean(all_results['structure']['correlation']):.4f} ± {np.std(all_results['structure']['correlation']):.4f}")
    print(f"  F1 Score: {np.mean(all_results['structure']['f1']):.4f} ± {np.std(all_results['structure']['f1']):.4f}")
    print(f"  Precision: {np.mean(all_results['structure']['precision']):.4f}")
    print(f"  Recall: {np.mean(all_results['structure']['recall']):.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    speckle_corr = np.mean(all_results['speckle']['correlation'])
    structure_corr = np.mean(all_results['structure']['correlation'])

    print("\nCorrelation with actual error:")
    if speckle_corr > 0.3:
        print(f"  ✓ Speckle predicate shows GOOD correlation ({speckle_corr:.3f})")
    elif speckle_corr > 0.1:
        print(f"  ⚠ Speckle predicate shows WEAK correlation ({speckle_corr:.3f})")
    else:
        print(f"  ✗ Speckle predicate shows POOR correlation ({speckle_corr:.3f})")

    if structure_corr > 0.3:
        print(f"  ✓ Structure predicate shows GOOD correlation ({structure_corr:.3f})")
    elif structure_corr > 0.1:
        print(f"  ⚠ Structure predicate shows WEAK correlation ({structure_corr:.3f})")
    else:
        print(f"  ✗ Structure predicate shows POOR correlation ({structure_corr:.3f})")

    print("\nConclusion:")
    if speckle_corr > 0.2 and structure_corr > 0.2:
        print("  Predicates ARE identifying regions correlated with actual errors.")
        print("  The symbolic approach has validity for guiding corrections.")
    elif speckle_corr > 0.2 or structure_corr > 0.2:
        print("  PARTIAL success - one predicate correlates, one doesn't.")
        print("  Consider adjusting the weaker predicate's thresholds.")
    else:
        print("  Predicates do NOT correlate well with actual errors.")
        print("  The predicates may be measuring different aspects of quality.")
        print("  This doesn't mean they're wrong - just different from pixel error.")

    return all_results


if __name__ == "__main__":
    results = run_evaluation(n_samples=5)
