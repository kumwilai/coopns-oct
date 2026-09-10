#!/usr/bin/env python3
"""
Debug script to evaluate DSP-only segmentation quality.
Step 1: Check if DSP boundary detection produces good layer segmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
import sys
sys.path.insert(0, '.')

from train_tmi_enhanced import TMIEnhancedModel, remap_mask_to_4class


def load_sample(jsonl_path, idx=0):
    """Load a single sample from JSONL."""
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i == idx:
                entry = json.loads(line)
                break

    # Load images - format uses image_path (clean) and we add synthetic noise
    clean = np.array(Image.open(entry['image_path']).convert('L')) / 255.0
    mask = np.array(Image.open(entry['mask_path']))

    # Add synthetic noise (Gaussian for testing)
    noise_level = 0.1
    noisy = clean + np.random.randn(*clean.shape) * noise_level
    noisy = np.clip(noisy, 0, 1)

    # Remap to 4-class
    mask_4class = remap_mask_to_4class(mask)

    return noisy, clean, mask_4class, entry


def compute_dice(pred, gt, num_classes=4):
    """Compute per-class Dice scores."""
    dice_scores = {}
    class_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    for c in range(num_classes):
        pred_c = (pred == c).astype(float)
        gt_c = (gt == c).astype(float)

        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()

        if union > 0:
            dice = 2 * intersection / union
        else:
            dice = 1.0 if intersection == 0 else 0.0

        dice_scores[class_names[c]] = dice

    return dice_scores


def compute_boundary_mae(pred_boundaries, gt_mask, H):
    """Compute MAE for each boundary in pixels."""
    # Extract GT boundaries from mask
    gt_boundaries = []
    W = gt_mask.shape[1]

    # For each boundary (0: ILM, 1: RNFL/INL, 2: INL/IS_OS, 3: IS_OS/RPE)
    for b in range(4):
        boundary_row = np.zeros(W)
        for col in range(W):
            col_mask = gt_mask[:, col]
            if b == 0:  # ILM - top of RNFL_GCL (class 0)
                rows = np.where(col_mask == 0)[0]
                boundary_row[col] = rows[0] if len(rows) > 0 else 0
            elif b == 1:  # RNFL/INL - between class 0 and 1
                rows = np.where(col_mask == 1)[0]
                boundary_row[col] = rows[0] if len(rows) > 0 else H//3
            elif b == 2:  # INL/IS_OS - between class 1 and 2
                rows = np.where(col_mask == 2)[0]
                boundary_row[col] = rows[0] if len(rows) > 0 else H//2
            elif b == 3:  # IS_OS/RPE - between class 2 and 3
                rows = np.where(col_mask == 3)[0]
                boundary_row[col] = rows[0] if len(rows) > 0 else 2*H//3
        gt_boundaries.append(boundary_row)

    gt_boundaries = np.array(gt_boundaries)  # [4, W]

    # Compute MAE for each boundary
    mae = {}
    boundary_names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']

    # pred_boundaries is [4, W] in normalized [0,1] range
    pred_pixels = pred_boundaries * H

    for b in range(4):
        mae[boundary_names[b]] = np.abs(pred_pixels[b] - gt_boundaries[b]).mean()

    return mae, gt_boundaries


def main():
    print("=" * 60)
    print("DSP-Only Segmentation Quality Test")
    print("=" * 60)

    # Load model
    device = 'cpu'
    model = TMIEnhancedModel(
        nafnet_width=64,
        num_classes=4,
        use_layer_specific_heads=True,
        use_seg_guided_attention=True,
        use_columnar_attention=True,
        use_dsp_boundaries=True,
        dsp_only=True,
        use_adaptive_strength_map=True,
        strength_init_bias=0.0,
    ).to(device)

    # Load NAFNet weights
    ckpt_path = 'outputs/nafnet_calibrated/nafnet_best.pth'
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt

        nafnet_state = {k.replace('nafnet.', ''): v for k, v in state.items() if 'nafnet' in k}
        if nafnet_state:
            model.nafnet.load_state_dict(nafnet_state, strict=False)
            print(f"Loaded NAFNet weights from {ckpt_path}")

    model.eval()

    # Test on multiple samples
    jsonl_path = 'combined_val.jsonl'
    num_samples = 5

    all_dice = {k: [] for k in ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']}
    all_mae = {k: [] for k in ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']}

    print(f"\nTesting on {num_samples} samples from {jsonl_path}...")
    print("-" * 60)

    for idx in range(num_samples):
        noisy, clean, gt_mask, entry = load_sample(jsonl_path, idx)
        H, W = noisy.shape

        # Prepare input
        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
        clean_t = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(noisy_t, clean_for_seg=clean_t)

        # Get DSP boundaries and segmentation
        dsp_boundaries = outputs.get('dsp_boundaries', None)
        seg_logits = outputs.get('seg_logits', None)
        denoised = outputs.get('denoised', None)
        denoised_base = outputs.get('denoised_base', None)

        if dsp_boundaries is None:
            print(f"  Sample {idx}: No DSP boundaries!")
            continue

        # Convert boundaries to numpy [4, W]
        boundaries_np = dsp_boundaries[0].cpu().numpy()  # [4, W]

        # Get predicted segmentation from logits
        if seg_logits is not None:
            pred_seg = seg_logits[0].argmax(dim=0).cpu().numpy()  # [H, W]
        else:
            pred_seg = np.zeros_like(gt_mask)

        # Compute metrics
        dice_scores = compute_dice(pred_seg, gt_mask)
        mae_scores, gt_boundaries = compute_boundary_mae(boundaries_np, gt_mask, H)

        # Compute PSNR
        if denoised is not None:
            denoised_np = denoised[0, 0].cpu().numpy()
            mse = ((denoised_np - clean) ** 2).mean()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        else:
            psnr = 0

        if denoised_base is not None:
            base_np = denoised_base[0, 0].cpu().numpy()
            mse_base = ((base_np - clean) ** 2).mean()
            psnr_base = 10 * np.log10(1.0 / (mse_base + 1e-10))
        else:
            psnr_base = 0

        print(f"\nSample {idx}: {Path(entry['image_path']).name}")
        print(f"  PSNR: NAFNet={psnr_base:.2f} dB, Adaptive={psnr:.2f} dB, Gain={psnr-psnr_base:+.2f} dB")
        print(f"  Dice Scores:")
        for name, score in dice_scores.items():
            status = "OK" if score > 0.7 else "POOR" if score > 0.3 else "BAD"
            print(f"    {name}: {score:.4f} [{status}]")
            all_dice[name].append(score)

        print(f"  Boundary MAE (pixels):")
        for name, mae in mae_scores.items():
            status = "OK" if mae < 5 else "POOR" if mae < 10 else "BAD"
            print(f"    {name}: {mae:.2f} px [{status}]")
            all_mae[name].append(mae)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nAverage Dice Scores:")
    for name, scores in all_dice.items():
        avg = np.mean(scores) if scores else 0
        status = "OK" if avg > 0.7 else "POOR" if avg > 0.3 else "BAD"
        print(f"  {name}: {avg:.4f} [{status}]")

    print("\nAverage Boundary MAE (pixels):")
    for name, maes in all_mae.items():
        avg = np.mean(maes) if maes else 0
        status = "OK" if avg < 5 else "POOR" if avg < 10 else "BAD"
        print(f"  {name}: {avg:.2f} px [{status}]")

    # Overall assessment
    avg_dice = np.mean([np.mean(v) for v in all_dice.values() if v])
    avg_mae = np.mean([np.mean(v) for v in all_mae.values() if v])

    print(f"\nOverall Average Dice: {avg_dice:.4f}")
    print(f"Overall Average MAE: {avg_mae:.2f} px")

    if avg_dice < 0.3:
        print("\n⚠️  DSP segmentation is POOR - boundaries may not align with GT")
        print("   This could explain why adaptive denoising isn't helping")
    elif avg_dice < 0.7:
        print("\n⚠️  DSP segmentation is MODERATE - some misalignment with GT")
    else:
        print("\n✓  DSP segmentation looks GOOD")


if __name__ == '__main__':
    main()
