#!/usr/bin/env python3
"""
Calibrate speckle statistics on real PKU37 data.

This script measures the actual coefficient of variation (CV) of speckle noise
in PKU37 dataset to set the correct threshold for the SpeckleFidelity predicate.

Expected output:
- Mean CV across dataset
- CV distribution (std, min, max)
- Recommended threshold for predicate
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm


def compute_local_cv(residual, intensity, window_size=16, min_intensity=0.05):
    """Compute local coefficient of variation."""
    # Create averaging kernel
    kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    kernel = kernel.to(residual.device)

    # Pad for valid convolution
    pad = window_size // 2
    residual_pad = F.pad(residual, (pad, pad, pad, pad), mode='reflect')
    intensity_pad = F.pad(intensity, (pad, pad, pad, pad), mode='reflect')

    # Local variance of residual
    res_sq = residual_pad ** 2
    local_var = F.conv2d(res_sq, kernel)
    local_mean_res = F.conv2d(residual_pad, kernel)
    local_var = local_var - local_mean_res ** 2
    local_std = torch.sqrt(local_var.clamp(min=1e-8))

    # Local mean of intensity
    local_intensity = F.conv2d(intensity_pad, kernel)

    # CV = Std(residual) / Mean(intensity)
    valid_mask = local_intensity > min_intensity
    cv = local_std / local_intensity.clamp(min=min_intensity)

    # Return mean CV over valid regions
    cv_valid = cv[valid_mask]
    return cv_valid.mean().item(), cv_valid.std().item()


def load_image(path):
    """Load image and normalize to [0, 1]."""
    img = Image.open(path).convert('L')
    img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def main():
    # PKU37 paths
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    train_jsonl = pku37_root / "weights_pku37_analysis_train.jsonl"

    if not train_jsonl.exists():
        print(f"Error: {train_jsonl} not found")
        return

    # Load pairs
    pairs = []
    with open(train_jsonl) as f:
        for line in f:
            data = json.loads(line)
            pairs.append({
                'clean': data['clean_path'],
                'noisy': data['noisy_path'],
            })

    print(f"Found {len(pairs)} image pairs")
    print(f"Analyzing speckle statistics...\n")

    # Collect CV statistics
    cv_means = []
    cv_stds = []

    # Sample subset for speed
    sample_size = min(100, len(pairs))
    sample_indices = np.random.choice(len(pairs), sample_size, replace=False)

    for idx in tqdm(sample_indices, desc="Processing"):
        pair = pairs[idx]

        # Load clean and noisy (paths are absolute)
        clean_path = Path(pair['clean'])
        noisy_path = Path(pair['noisy'])

        if not clean_path.exists() or not noisy_path.exists():
            continue

        clean = load_image(clean_path)
        noisy = load_image(noisy_path)

        # Compute residual (noise)
        residual = noisy - clean

        # Compute CV
        cv_mean, cv_std = compute_local_cv(residual, clean)
        cv_means.append(cv_mean)
        cv_stds.append(cv_std)

    # Statistics
    cv_means = np.array(cv_means)
    cv_stds = np.array(cv_stds)

    print("\n" + "=" * 60)
    print("SPECKLE STATISTICS ON PKU37")
    print("=" * 60)
    print(f"\nCoefficient of Variation (CV = Std/Mean):")
    print(f"  Mean CV: {cv_means.mean():.4f}")
    print(f"  Std CV:  {cv_means.std():.4f}")
    print(f"  Min CV:  {cv_means.min():.4f}")
    print(f"  Max CV:  {cv_means.max():.4f}")

    print(f"\nRecommended settings for SpeckleFidelityPredicate:")
    print(f"  expected_cv = {cv_means.mean():.2f}")
    print(f"  tolerance = {cv_means.std() * 2:.2f}  (2 std)")

    print("\n" + "=" * 60)

    # Also check intensity distribution
    print("\nIntensity Statistics:")

    intensities_clean = []
    intensities_noisy = []

    for idx in sample_indices[:20]:  # Smaller sample
        pair = pairs[idx]
        clean_path = Path(pair['clean'])
        noisy_path = Path(pair['noisy'])

        if clean_path.exists() and noisy_path.exists():
            clean = load_image(clean_path)
            noisy = load_image(noisy_path)
            intensities_clean.append(clean.mean().item())
            intensities_noisy.append(noisy.mean().item())

    print(f"  Clean mean intensity: {np.mean(intensities_clean):.3f}")
    print(f"  Noisy mean intensity: {np.mean(intensities_noisy):.3f}")


if __name__ == "__main__":
    main()
