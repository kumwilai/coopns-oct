#!/usr/bin/env python3
"""
Diagnose why thin layers (INL_OPL_ONL, IS_OS) have low Dice scores.
"""

import json
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

# Load some samples and analyze layer thicknesses
def analyze_layer_thicknesses(jsonl_path, num_samples=50):
    """Analyze the actual thickness of each layer in the dataset."""

    thicknesses = {
        'RNFL_GCL': [],      # boundary[0] to boundary[1]
        'INL_OPL_ONL': [],   # boundary[1] to boundary[2]
        'IS_OS': [],         # boundary[2] to boundary[3]
        'RPE_Choroid': [],   # boundary[3] to bottom
    }

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            entry = json.loads(line)
            mask = np.array(Image.open(entry['mask_path']))
            H, W = mask.shape

            # For each column, find boundary positions
            for col in range(W):
                col_data = mask[:, col]

                # Find boundaries (transitions)
                # boundary[0] = top of class 1 (RNFL)
                # boundary[1] = top of class 2 (INL)
                # boundary[2] = top of class 3 (IS_OS)
                # boundary[3] = top of class 4 (RPE)

                boundaries = []
                for orig_class in [1, 2, 3, 4]:
                    rows = np.where(col_data == orig_class)[0]
                    if len(rows) > 0:
                        boundaries.append(rows[0])
                    else:
                        boundaries.append(None)

                if None in boundaries:
                    continue

                # Calculate thicknesses
                thicknesses['RNFL_GCL'].append(boundaries[1] - boundaries[0])
                thicknesses['INL_OPL_ONL'].append(boundaries[2] - boundaries[1])
                thicknesses['IS_OS'].append(boundaries[3] - boundaries[2])
                thicknesses['RPE_Choroid'].append(H - boundaries[3])

    print("=" * 60)
    print("Layer Thickness Analysis (in pixels)")
    print("=" * 60)

    for layer, values in thicknesses.items():
        if len(values) > 0:
            values = np.array(values)
            print(f"\n{layer}:")
            print(f"  Mean: {values.mean():.1f} px")
            print(f"  Std:  {values.std():.1f} px")
            print(f"  Min:  {values.min()} px")
            print(f"  Max:  {values.max()} px")
            print(f"  <5px: {(values < 5).sum() / len(values) * 100:.1f}%")
            print(f"  <10px: {(values < 10).sum() / len(values) * 100:.1f}%")

    return thicknesses


def analyze_dice_sensitivity(thickness, mae):
    """
    Analyze how Dice changes with boundary MAE for different layer thicknesses.

    For a layer of thickness T between boundaries b1 and b2:
    - If both boundaries have error E in same direction: thickness stays T, position shifts
    - If both boundaries have error E in opposite directions: thickness becomes T +/- 2E
    - Dice = 2 * intersection / (pred_area + gt_area)

    For a thin layer, even small errors cause large Dice drops.
    """
    print("\n" + "=" * 60)
    print("Dice Sensitivity Analysis")
    print("=" * 60)

    # Simulate Dice for different thicknesses and MAE
    for t in [5, 10, 15, 20, 30, 50]:
        print(f"\nLayer thickness = {t} px:")
        for e in [2, 5, 10, 15]:
            # Worst case: both boundaries off in opposite directions
            # Predicted thickness = t + 2*e or t - 2*e
            pred_thick_expand = t + 2 * e
            pred_thick_shrink = max(0, t - 2 * e)

            # Approximate Dice for expansion
            intersection_exp = t  # Original layer is contained
            dice_exp = 2 * intersection_exp / (pred_thick_expand + t)

            # Approximate Dice for shrinkage
            intersection_shrink = pred_thick_shrink
            dice_shrink = 2 * intersection_shrink / (pred_thick_shrink + t) if (pred_thick_shrink + t) > 0 else 0

            print(f"  MAE={e}px: Dice(expand)={dice_exp:.3f}, Dice(shrink)={dice_shrink:.3f}")


def propose_solutions():
    """Propose solutions for thin layer detection."""

    print("\n" + "=" * 60)
    print("Proposed Solutions for Thin Layer Detection")
    print("=" * 60)

    solutions = """
1. **Thickness-Constrained Loss** (CRITICAL):
   - Add explicit loss for layer thickness: L_thick = |pred_thick - gt_thick|
   - This ensures relative positions are correct even if absolute positions have error
   - Weight higher for thin layers (INL, IS_OS)

2. **Boundary Pair Supervision**:
   - Instead of supervising boundaries independently, supervise pairs:
     - (boundary[1], boundary[2]) for INL layer
     - (boundary[2], boundary[3]) for IS_OS layer
   - Loss = |pred_thickness - gt_thickness| + |pred_center - gt_center|

3. **Minimum Thickness Constraint**:
   - Enforce predicted_thickness >= min_thickness for each layer
   - Use soft constraint: ReLU(min_thick - pred_thick)^2

4. **Adaptive Temperature**:
   - Use lower temperature (sharper) for thin layer boundaries
   - boundary[2], boundary[3] need temperature ~0.01 vs 0.03 for others

5. **Per-Layer Dice Loss** (Direct Supervision):
   - Convert boundaries to soft segmentation
   - Compute differentiable Dice loss per layer
   - Weight thin layers (INL, IS_OS) higher

6. **Coupled Boundary Prediction**:
   - Instead of predicting 4 independent boundaries, predict:
     - ILM position
     - RNFL thickness
     - INL thickness
     - IS_OS thickness
   - Derive boundaries from cumulative sum
   - This guarantees ordering and valid thicknesses
"""
    print(solutions)


if __name__ == '__main__':
    # Analyze layer thicknesses in the dataset
    print("\nAnalyzing training data...")
    thicknesses = analyze_layer_thicknesses('combined_train.jsonl', num_samples=100)

    # Analyze Dice sensitivity
    analyze_dice_sensitivity(10, 5)

    # Propose solutions
    propose_solutions()
