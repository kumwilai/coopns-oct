#!/usr/bin/env python3
"""
Analyze what features of (noisy, denoised) correlate with actual error.

Goal: Find features computable WITHOUT ground truth that predict error regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys


def pearson_correlation(x, y):
    """Compute Pearson correlation."""
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_mean = x_flat.mean()
    y_mean = y_flat.mean()
    num = ((x_flat - x_mean) * (y_flat - y_mean)).sum()
    den = torch.sqrt(((x_flat - x_mean)**2).sum() * ((y_flat - y_mean)**2).sum())
    return (num / (den + 1e-8)).item()


def compute_features(noisy, denoised):
    """Compute various features that might correlate with error."""
    features = {}

    # 1. Residual magnitude
    residual = noisy - denoised
    features['residual_magnitude'] = residual.abs()

    # 2. Residual squared
    features['residual_squared'] = residual ** 2

    # 3. Local residual mean (smoothed residual)
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    res_pad = F.pad(residual, (2, 2, 2, 2), mode='reflect')
    features['local_residual_mean'] = F.conv2d(res_pad, kernel).abs()

    # 4. Local residual variance
    res_sq_pad = F.pad(residual ** 2, (2, 2, 2, 2), mode='reflect')
    local_mean_sq = F.conv2d(res_pad, kernel) ** 2
    local_sq_mean = F.conv2d(res_sq_pad, kernel)
    features['local_residual_var'] = (local_sq_mean - local_mean_sq).clamp(min=0)

    # 5. Edge magnitude in denoised
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 4
    den_pad = F.pad(denoised, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(den_pad, sobel_x)
    gy = F.conv2d(den_pad, sobel_y)
    features['denoised_edge'] = torch.sqrt(gx**2 + gy**2 + 1e-8)

    # 6. Edge magnitude in residual
    res_pad2 = F.pad(residual, (1, 1, 1, 1), mode='reflect')
    gx_r = F.conv2d(res_pad2, sobel_x)
    gy_r = F.conv2d(res_pad2, sobel_y)
    features['residual_edge'] = torch.sqrt(gx_r**2 + gy_r**2 + 1e-8)

    # 7. Laplacian of denoised (high-frequency content)
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    features['denoised_laplacian'] = F.conv2d(den_pad, laplacian).abs()

    # 8. Laplacian of residual
    features['residual_laplacian'] = F.conv2d(res_pad2, laplacian).abs()

    # 9. Local intensity of denoised
    den_pad2 = F.pad(denoised, (2, 2, 2, 2), mode='reflect')
    features['local_intensity'] = F.conv2d(den_pad2, kernel)

    # 10. Ratio: residual / local_intensity (like CV)
    features['residual_over_intensity'] = residual.abs() / (features['local_intensity'] + 0.01)

    # 11. Difference: noisy_edge - denoised_edge
    noisy_pad = F.pad(noisy, (1, 1, 1, 1), mode='reflect')
    gx_n = F.conv2d(noisy_pad, sobel_x)
    gy_n = F.conv2d(noisy_pad, sobel_y)
    noisy_edge = torch.sqrt(gx_n**2 + gy_n**2 + 1e-8)
    features['edge_reduction'] = noisy_edge - features['denoised_edge']

    # 12. Local standard deviation of denoised
    den_sq_pad = F.pad(denoised ** 2, (2, 2, 2, 2), mode='reflect')
    local_mean = F.conv2d(den_pad2, kernel)
    local_sq_mean_d = F.conv2d(den_sq_pad, kernel)
    features['local_std_denoised'] = torch.sqrt((local_sq_mean_d - local_mean**2).clamp(min=0))

    # 13. Absolute difference from local mean
    features['diff_from_local_mean'] = (denoised - features['local_intensity']).abs()

    return features


def main():
    print("=" * 70)
    print("ANALYZING FEATURES THAT CORRELATE WITH RECONSTRUCTION ERROR")
    print("=" * 70)

    # Load models
    sys.path.insert(0, 'nsnd_oct')
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet

    backbone = NAFNet(
        img_channel=1, width=64, middle_blk_num=2,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
    )
    ckpt = torch.load("/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth",
                      map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt['state_dict'], strict=False)
    backbone.eval()

    # Load data
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    val_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    all_lines = open(val_jsonl).readlines()
    indices = list(range(0, len(all_lines), len(all_lines) // 10))[:10]

    # Accumulate correlations
    all_correlations = {}

    for idx in indices:
        data = json.loads(all_lines[idx])
        clean = torch.from_numpy(np.array(Image.open(data['clean_path']).convert('L'))).float() / 255.0
        noisy = torch.from_numpy(np.array(Image.open(data['noisy_path']).convert('L'))).float() / 255.0

        clean = clean.unsqueeze(0).unsqueeze(0)
        noisy = noisy.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            denoised = backbone(noisy).clamp(0, 1)

        # Actual error
        actual_error = (denoised - clean).abs()

        # Compute features
        features = compute_features(noisy, denoised)

        # Compute correlation of each feature with error
        for name, feat in features.items():
            # Resize if needed
            if feat.shape != actual_error.shape:
                feat = F.interpolate(feat, size=actual_error.shape[2:], mode='bilinear', align_corners=False)

            corr = pearson_correlation(feat, actual_error)

            if name not in all_correlations:
                all_correlations[name] = []
            all_correlations[name].append(corr)

    # Print results
    print("\n" + "-" * 70)
    print("FEATURE CORRELATIONS WITH ACTUAL ERROR (sorted by abs correlation)")
    print("-" * 70)

    results = []
    for name, corrs in all_correlations.items():
        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs)
        results.append((name, mean_corr, std_corr))

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<30} {'Correlation':>12} {'Std':>10}")
    print("-" * 55)
    for name, mean_corr, std_corr in results:
        marker = "***" if abs(mean_corr) > 0.3 else ("**" if abs(mean_corr) > 0.2 else ("*" if abs(mean_corr) > 0.1 else ""))
        print(f"{name:<30} {mean_corr:>12.4f} {std_corr:>10.4f} {marker}")

    print("\n" + "=" * 70)
    print("TOP FEATURES FOR STRUCTURE PREDICATE:")
    print("=" * 70)
    for name, mean_corr, std_corr in results[:5]:
        print(f"  {name}: correlation = {mean_corr:.4f}")

    return results


if __name__ == "__main__":
    results = main()
