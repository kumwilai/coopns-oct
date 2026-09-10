#!/usr/bin/env python3
"""
Evaluate CUAP-OCT Model with Neuro-Symbolic Post-Processing

This script demonstrates the integration of neural network segmentation
with symbolic post-processing for anatomically guaranteed results.

For TMI Paper:
    "Our neuro-symbolic approach combines deep learning with explicit
    anatomical rules, achieving both high accuracy and guaranteed
    anatomical plausibility."

Usage:
    python evaluate_with_symbolic.py --checkpoint path/to/best_psnr.pth
"""

import argparse
import os
import sys

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_multitask import (
    MultiTaskOCTDataset,
    MultiTaskDenoiser,
    compute_dice_score,
    compute_per_layer_dice,
    LAYER_NAMES,
)
from symbolic_postprocess import (
    SymbolicPostProcessor,
    print_clinical_report,
    LAYER_KNOWLEDGE,
)
from torch.utils.data import DataLoader


def evaluate_with_symbolic(model, loader, device, use_symbolic=True):
    """
    Evaluate model with optional symbolic post-processing.

    Returns metrics for both neural-only and neural+symbolic approaches.
    """
    model.eval()

    processor = SymbolicPostProcessor(
        enforce_ordering=True,
        enforce_continuity=True,
        check_thickness=True,
        check_completeness=True,
        verbose=False
    ) if use_symbolic else None

    # Metrics accumulators
    neural_dice_sum = 0
    symbolic_dice_sum = 0
    neural_per_layer = {name: [] for name in LAYER_NAMES}
    symbolic_per_layer = {name: [] for name in LAYER_NAMES}

    # Anatomical validity tracking
    total_samples = 0
    anatomically_valid_neural = 0
    anatomically_valid_symbolic = 0
    violations_corrected = 0

    # Clinical flags
    all_clinical_flags = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            noisy = batch['noisy'].to(device)
            seg_mask_gt = batch['seg_mask'].to(device)

            # Neural network prediction
            _, seg_logits = model(noisy)
            seg_pred_neural = seg_logits.argmax(dim=1)

            # Compute neural-only metrics
            neural_dice = compute_dice_score(seg_logits, seg_mask_gt)
            neural_dice_sum += neural_dice

            neural_layer_dice = compute_per_layer_dice(seg_logits, seg_mask_gt)
            for name in LAYER_NAMES:
                if neural_layer_dice[name] is not None:
                    neural_per_layer[name].append(neural_layer_dice[name])

            if use_symbolic:
                # Apply symbolic post-processing
                seg_pred_symbolic, reports = processor(seg_logits, return_report=True)

                # Handle batch vs single report
                if not isinstance(reports, list):
                    reports = [reports]

                # Compute symbolic metrics
                symbolic_logits = torch.zeros_like(seg_logits)
                for b in range(seg_pred_symbolic.shape[0]):
                    for c in range(seg_logits.shape[1]):
                        symbolic_logits[b, c] = (seg_pred_symbolic[b] == c).float()

                symbolic_dice = compute_dice_score(symbolic_logits, seg_mask_gt)
                symbolic_dice_sum += symbolic_dice

                symbolic_layer_dice = compute_per_layer_dice(symbolic_logits, seg_mask_gt)
                for name in LAYER_NAMES:
                    if symbolic_layer_dice[name] is not None:
                        symbolic_per_layer[name].append(symbolic_layer_dice[name])

                # Track anatomical validity
                for report in reports:
                    total_samples += 1

                    # Check neural validity (using processor on neural output)
                    _, neural_report = processor(seg_pred_neural, return_report=True)
                    if isinstance(neural_report, list):
                        neural_report = neural_report[0]

                    if neural_report.original_violations == 0:
                        anatomically_valid_neural += 1

                    if report.is_anatomically_valid:
                        anatomically_valid_symbolic += 1

                    violations_corrected += (neural_report.original_violations -
                                             report.corrected_violations)

                    all_clinical_flags.extend(report.clinical_flags)

            else:
                symbolic_dice_sum = neural_dice_sum
                symbolic_per_layer = neural_per_layer

    n_batches = len(loader)

    results = {
        'neural': {
            'mean_dice': neural_dice_sum / n_batches,
            'per_layer_dice': {name: np.mean(neural_per_layer[name])
                              if neural_per_layer[name] else 0
                              for name in LAYER_NAMES},
        },
        'symbolic': {
            'mean_dice': symbolic_dice_sum / n_batches,
            'per_layer_dice': {name: np.mean(symbolic_per_layer[name])
                              if symbolic_per_layer[name] else 0
                              for name in LAYER_NAMES},
        },
        'anatomical_validity': {
            'total_samples': total_samples,
            'neural_valid': anatomically_valid_neural,
            'neural_valid_pct': anatomically_valid_neural / total_samples * 100 if total_samples > 0 else 0,
            'symbolic_valid': anatomically_valid_symbolic,
            'symbolic_valid_pct': anatomically_valid_symbolic / total_samples * 100 if total_samples > 0 else 0,
            'violations_corrected': violations_corrected,
        },
        'clinical_flags': all_clinical_flags,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate with symbolic post-processing')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--backbone_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--val_jsonl', default='seg_data/seg_val.jsonl')
    parser.add_argument('--max_val', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print("=" * 70)
    print("NEURO-SYMBOLIC EVALUATION")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")

    # Load data
    val_ds = MultiTaskOCTDataset(args.val_jsonl, patch_size=64,
                                  max_samples=args.max_val, random_crop=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Validation samples: {len(val_ds)}")

    # Load model
    model = MultiTaskDenoiser(backbone_ckpt=args.backbone_ckpt).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print("Model loaded.")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    results = evaluate_with_symbolic(model, val_loader, args.device, use_symbolic=True)

    # Print comparison
    print("\n### Segmentation Accuracy (Dice Score)")
    print("-" * 50)
    print(f"{'Metric':<20} {'Neural Only':<15} {'Neural+Symbolic':<15}")
    print("-" * 50)
    print(f"{'Mean Dice':<20} {results['neural']['mean_dice']:<15.4f} {results['symbolic']['mean_dice']:<15.4f}")
    print("-" * 50)

    print("\n### Per-Layer Dice")
    print("-" * 50)
    print(f"{'Layer':<15} {'Neural':<12} {'Symbolic':<12} {'Change':<12}")
    print("-" * 50)
    for name in LAYER_NAMES:
        neural = results['neural']['per_layer_dice'][name]
        symbolic = results['symbolic']['per_layer_dice'][name]
        change = symbolic - neural
        print(f"{name:<15} {neural:<12.4f} {symbolic:<12.4f} {change:+.4f}")
    print("-" * 50)

    # Anatomical validity
    av = results['anatomical_validity']
    print("\n### Anatomical Validity (KEY NEURO-SYMBOLIC BENEFIT)")
    print("-" * 50)
    print(f"{'Approach':<25} {'Valid Samples':<15} {'Percentage':<15}")
    print("-" * 50)
    print(f"{'Neural Only':<25} {av['neural_valid']:<15} {av['neural_valid_pct']:.1f}%")
    print(f"{'Neural + Symbolic':<25} {av['symbolic_valid']:<15} {av['symbolic_valid_pct']:.1f}%")
    print("-" * 50)
    print(f"Violations Corrected: {av['violations_corrected']}")

    # Clinical flags
    if results['clinical_flags']:
        print("\n### Clinical Flags Detected")
        print("-" * 50)
        # Count unique flags
        from collections import Counter
        flag_counts = Counter(results['clinical_flags'])
        for flag, count in flag_counts.most_common(10):
            print(f"  [{count:3d}x] {flag}")

    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR TMI PAPER")
    print("=" * 70)
    print(f"""
Neural Network Segmentation:
  - Mean Dice: {results['neural']['mean_dice']:.4f}
  - Anatomically Valid: {av['neural_valid_pct']:.1f}%

Neural + Symbolic Post-Processing:
  - Mean Dice: {results['symbolic']['mean_dice']:.4f}
  - Anatomically Valid: {av['symbolic_valid_pct']:.1f}%
  - Violations Corrected: {av['violations_corrected']}

Key Claim:
  "Our neuro-symbolic approach guarantees {av['symbolic_valid_pct']:.0f}% anatomically
  valid segmentations compared to {av['neural_valid_pct']:.0f}% for neural-only,
  while maintaining comparable Dice scores."
""")
    print("=" * 70)


if __name__ == '__main__':
    main()
