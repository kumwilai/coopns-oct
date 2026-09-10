#!/usr/bin/env python3
"""
Evaluate NLM with noise-floor compensation on PKU37.

Approach: For each feature, estimate the noise energy fraction from the error
map and apply a multiplicative correction:
  compensated = raw * sqrt(max(1 - noise_energy_fraction, 0))
where noise_energy_fraction = mean(F(error)^2) / mean(F(denoised)^2)

This preserves the metric type (mean absolute ratio) while removing the noise
energy contribution in the variance domain.

Also caps per-image values at 1.0 (feature preservation can't exceed ground truth).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_bm3d_nlm_clinical import (
    compute_psnr, compute_ssim, denoise_nlm_img,
    _SOBEL_Y, _SOBEL_X, _LAPLACIAN,
    compute_all_metrics,
)


@torch.no_grad()
def compute_noise_correction_factors(denoised, clean):
    """Compute noise energy fraction for each feature dimension."""
    device = denoised.device
    error = denoised - clean

    sobel_y = _SOBEL_Y.to(device)
    sobel_x = _SOBEL_X.to(device)
    laplacian_k = _LAPLACIAN.to(device)

    factors = {}

    # Contrast (local variance)
    for img, prefix in [(denoised, 'den'), (error, 'err')]:
        mu = F.avg_pool2d(img, 7, 1, 3)
        var_map = (F.avg_pool2d(img**2, 7, 1, 3) - mu**2).clamp(min=0)
        factors[f'{prefix}_contrast_var'] = var_map.mean().item()
    nef_contrast = factors['err_contrast_var'] / max(factors['den_contrast_var'], 1e-8)
    factors['contrast_correction'] = max(1.0 - nef_contrast, 0) ** 0.5

    # Boundary (Sobel Y)
    d_gy = F.conv2d(denoised, sobel_y, padding=1)
    e_gy = F.conv2d(error, sobel_y, padding=1)
    nef_boundary = (e_gy**2).mean().item() / max((d_gy**2).mean().item(), 1e-8)
    factors['boundary_correction'] = max(1.0 - nef_boundary, 0) ** 0.5

    # Texture (Laplacian)
    d_lap = F.conv2d(denoised, laplacian_k, padding=1)
    e_lap = F.conv2d(error, laplacian_k, padding=1)
    nef_texture = (e_lap**2).mean().item() / max((d_lap**2).mean().item(), 1e-8)
    factors['texture_correction'] = max(1.0 - nef_texture, 0) ** 0.5

    # Edge (combined Sobel)
    d_gx = F.conv2d(denoised, sobel_x, padding=1)
    e_gx = F.conv2d(error, sobel_x, padding=1)
    nef_edge = ((e_gx**2 + e_gy**2).mean().item() /
                max((d_gx**2 + d_gy**2).mean().item(), 1e-8))
    factors['edge_correction'] = max(1.0 - nef_edge, 0) ** 0.5

    return factors


def main():
    test_jsonl = 'pku37_oct_dataset/pku37_real_test.jsonl'
    pairs = []
    with open(test_jsonl) as f:
        for line in f:
            entry = json.loads(line.strip())
            pairs.append((entry['noisy_path'], entry['clean_path']))
    print(f"Loaded {len(pairs)} test pairs")

    all_raw = []
    all_factors = []
    t0 = time.time()

    for i, (noisy_path, clean_path) in enumerate(pairs):
        noisy_np = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean_np = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        nlm_np = denoise_nlm_img(noisy_np)

        noisy_t = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0)
        clean_t = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0)
        nlm_t = torch.from_numpy(nlm_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        # Raw metrics (same as original eval)
        raw = compute_all_metrics(nlm_t, noisy_t, clean_t)
        all_raw.append(raw)

        # Correction factors
        factors = compute_noise_correction_factors(nlm_t, clean_t)
        all_factors.append(factors)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            fc, fb, ft, fe = (factors['contrast_correction'], factors['boundary_correction'],
                              factors['texture_correction'], factors['edge_correction'])
            print(f"[{i+1}/{len(pairs)}] PSNR={raw['psnr_denoised']:.2f} "
                  f"corrections: c={fc:.3f} b={fb:.3f} t={ft:.3f} e={fe:.3f} "
                  f"raw_contrast={raw['contrast_denoised']:.3f} "
                  f"comp_contrast={min(raw['contrast_denoised']*fc, 1.0):.3f} "
                  f"({elapsed:.1f}s)")

    # Per-image compensated values (raw * correction, capped at 1.0)
    compensated_per_image = []
    for raw, factors in zip(all_raw, all_factors):
        comp = {
            'psnr_denoised': raw['psnr_denoised'],
            'ssim_denoised': raw['ssim_denoised'],
            'cnr_denoised': raw['cnr_denoised'],
            'contrast_denoised': min(raw['contrast_denoised'] * factors['contrast_correction'], 1.0),
            'boundary_denoised': min(raw['boundary_denoised'] * factors['boundary_correction'], 1.0),
            'texture_denoised': min(raw['texture_denoised'] * factors['texture_correction'], 1.0),
            'edge_denoised': min(raw['edge_denoised'] * factors['edge_correction'], 1.0),
        }
        compensated_per_image.append(comp)

    # Averages
    keys = ['psnr_denoised', 'ssim_denoised', 'cnr_denoised',
            'contrast_denoised', 'boundary_denoised', 'texture_denoised', 'edge_denoised']
    raw_avg = {k: float(np.mean([m[k] for m in all_raw])) for k in keys}
    comp_avg = {k: float(np.mean([m[k] for m in compensated_per_image])) for k in keys}
    factor_avg = {k: float(np.mean([f[k] for f in all_factors]))
                  for k in ['contrast_correction', 'boundary_correction',
                            'texture_correction', 'edge_correction']}

    print(f"\n{'='*70}")
    print(f"  NLM Noise-Compensated Results on PKU37 ({len(pairs)} images)")
    print(f"{'='*70}")
    print(f"  PSNR:       {comp_avg['psnr_denoised']:.2f} dB")
    print(f"  SSIM:       {comp_avg['ssim_denoised']:.4f}")
    print(f"  CNR:        {comp_avg['cnr_denoised']:.2f}")
    print(f"\n  {'Metric':<12} {'Raw':>8} {'Factor':>8} {'Compensated':>12}")
    print(f"  {'-'*44}")
    for feat in ['contrast', 'boundary', 'texture', 'edge']:
        raw_v = raw_avg[f'{feat}_denoised']
        comp_v = comp_avg[f'{feat}_denoised']
        fac = factor_avg[f'{feat}_correction']
        print(f"  {feat:<12} {raw_v:>8.3f} {fac:>8.3f} {comp_v:>12.3f}")

    # Save
    os.makedirs('outputs/baselines', exist_ok=True)
    result = {
        'method': 'nlm_noise_compensated',
        'n_images': len(pairs),
        'raw_averages': raw_avg,
        'compensated_averages': comp_avg,
        'correction_factors': factor_avg,
        'per_image': compensated_per_image,
    }
    out_path = 'outputs/baselines/nlm_pku37_noise_compensated.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
