#!/usr/bin/env python3
"""
Evaluate BM3D and NLM on PKU37 test set with full clinical metrics.
Uses the same 173-image test split and metrics as run_benchmarks.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# BM3D
try:
    import bm3d
    from bm3d import BM3DStages
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False

# NLM
from skimage.restoration import denoise_nl_means, estimate_sigma as sk_estimate_sigma
from scipy.ndimage import laplace


# ---- Metrics (identical to run_benchmarks.py) ----
def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse < 1e-10:
        return 50.0
    return (10.0 * torch.log10(1.0 / mse)).item()


_SSIM_KERNEL = None
def _get_ssim_kernel(device):
    global _SSIM_KERNEL
    if _SSIM_KERNEL is None or _SSIM_KERNEL.device != device:
        coords = torch.arange(11, dtype=torch.float32) - 5
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        k2d = g.unsqueeze(1) * g.unsqueeze(0)
        _SSIM_KERNEL = k2d.unsqueeze(0).unsqueeze(0).to(device)
    return _SSIM_KERNEL


def compute_ssim(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    pad = 5
    kernel = _get_ssim_kernel(pred.device)
    pp = F.pad(pred, [pad]*4, mode='reflect')
    tp = F.pad(target, [pad]*4, mode='reflect')
    mu_p = F.conv2d(pp, kernel)
    mu_t = F.conv2d(tp, kernel)
    sigma_p_sq = F.conv2d(pp * pp, kernel) - mu_p ** 2
    sigma_t_sq = F.conv2d(tp * tp, kernel) - mu_t ** 2
    sigma_pt = F.conv2d(pp * tp, kernel) - mu_p * mu_t
    ssim_map = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()


_SOBEL_Y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3)
_SOBEL_X = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
_LAPLACIAN = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3)


@torch.no_grad()
def compute_all_metrics(denoised, noisy, clean):
    device = denoised.device
    m = {}

    # PSNR / SSIM
    m['psnr_noisy'] = compute_psnr(noisy, clean)
    m['psnr_denoised'] = compute_psnr(denoised, clean)
    m['psnr_delta'] = m['psnr_denoised'] - m['psnr_noisy']
    m['ssim_noisy'] = compute_ssim(noisy, clean)
    m['ssim_denoised'] = compute_ssim(denoised, clean)
    m['ssim_delta'] = m['ssim_denoised'] - m['ssim_noisy']

    # CNR
    signal_mask = (clean > clean.mean()).float()
    bg_mask = 1.0 - signal_mask
    s_sum = signal_mask.sum().clamp(min=1)
    b_sum = bg_mask.sum().clamp(min=1)
    n_sig = (noisy * signal_mask).sum() / s_sum
    n_bg = (noisy * bg_mask).sum() / b_sum
    n_std = torch.sqrt(((noisy - n_bg)**2 * bg_mask).sum() / b_sum + 1e-8).clamp(min=1e-4)
    m['cnr_noisy'] = ((n_sig - n_bg) / n_std).clamp(-100, 100).item()
    d_sig = (denoised * signal_mask).sum() / s_sum
    d_bg = (denoised * bg_mask).sum() / b_sum
    d_std = torch.sqrt(((denoised - d_bg)**2 * bg_mask).sum() / b_sum + 1e-8).clamp(min=1e-4)
    m['cnr_denoised'] = ((d_sig - d_bg) / d_std).clamp(-100, 100).item()
    m['cnr_delta'] = m['cnr_denoised'] - m['cnr_noisy']
    abs_n, abs_d = abs(m['cnr_noisy']), abs(m['cnr_denoised'])
    m['cnr_improvement'] = ((abs_d / max(abs_n, 1e-4)) - 1.0) * 100 if abs_n > 1e-4 else 0

    # Gradient-based metrics
    sobel_y = _SOBEL_Y.to(device)
    sobel_x = _SOBEL_X.to(device)
    laplacian_k = _LAPLACIAN.to(device)
    stacked = torch.cat([noisy, denoised, clean], dim=0)
    all_gy = F.conv2d(stacked, sobel_y, padding=1).abs()
    all_gx = F.conv2d(stacked, sobel_x, padding=1)
    all_lap = F.conv2d(stacked, laplacian_k, padding=1).abs()
    n_gy, d_gy, c_gy = all_gy.chunk(3, dim=0)
    n_gx, d_gx, c_gx = all_gx.chunk(3, dim=0)
    n_lap, d_lap, c_lap = all_lap.chunk(3, dim=0)

    # Clinical preservation
    # Contrast
    stk = torch.cat([noisy, denoised, clean], dim=0)
    mu = F.avg_pool2d(stk, 7, 1, 3)
    std_map = torch.sqrt((F.avg_pool2d(stk**2, 7, 1, 3) - mu**2).clamp(min=1e-8))
    n_s, d_s, c_s = std_map.chunk(3, dim=0)
    c_s_mean = c_s.mean().clamp(min=1e-4)
    m['contrast_noisy'] = (n_s.mean() / c_s_mean).clamp(0, 10).item()
    m['contrast_denoised'] = (d_s.mean() / c_s_mean).clamp(0, 10).item()
    m['contrast_ratio'] = m['contrast_denoised'] / max(m['contrast_noisy'], 1e-4)

    # Boundary
    c_gy_mean = c_gy.mean().clamp(min=1e-4)
    m['boundary_noisy'] = (n_gy.mean() / c_gy_mean).clamp(0, 10).item()
    m['boundary_denoised'] = (d_gy.mean() / c_gy_mean).clamp(0, 10).item()
    m['boundary_ratio'] = m['boundary_denoised'] / max(m['boundary_noisy'], 1e-4)

    # Texture
    c_lap_mean = c_lap.mean().clamp(min=1e-4)
    m['texture_noisy'] = (n_lap.mean() / c_lap_mean).clamp(0, 10).item()
    m['texture_denoised'] = (d_lap.mean() / c_lap_mean).clamp(0, 10).item()
    m['texture_ratio'] = m['texture_denoised'] / max(m['texture_noisy'], 1e-4)

    # Edge
    n_edge = torch.sqrt(n_gx**2 + n_gy**2 + 1e-8)
    d_edge = torch.sqrt(d_gx**2 + d_gy**2 + 1e-8)
    c_edge = torch.sqrt(c_gx**2 + c_gy**2 + 1e-8)
    c_edge_mean = c_edge.mean().clamp(min=1e-4)
    m['edge_noisy'] = (n_edge.mean() / c_edge_mean).clamp(0, 10).item()
    m['edge_denoised'] = (d_edge.mean() / c_edge_mean).clamp(0, 10).item()
    m['edge_ratio'] = m['edge_denoised'] / max(m['edge_noisy'], 1e-4)

    # Correction magnitude
    m['correction_magnitude'] = (denoised - noisy).abs().mean().item()

    # Clinical improvement count
    m['clinical_improved'] = sum([
        m['contrast_ratio'] > 1.0,
        m['boundary_ratio'] > 1.0,
        m['texture_ratio'] > 1.0,
        m['edge_ratio'] > 1.0,
    ])

    return m


# ---- Denoising methods ----
def denoise_bm3d_img(noisy_np, sigma=None):
    if sigma is None:
        lap = laplace(noisy_np)
        sigma = np.median(np.abs(lap)) / 0.6745
    denoised = bm3d.bm3d(noisy_np, sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)
    return np.clip(denoised, 0.0, 1.0)


def denoise_nlm_img(noisy_np, h=None):
    sigma_est = sk_estimate_sigma(noisy_np)
    # Use sigma directly as h (standard NLM practice for OCT)
    if h is None:
        h = max(sigma_est, 0.05)
    denoised = denoise_nl_means(
        noisy_np, h=h, patch_size=7, patch_distance=11, fast_mode=True
    )
    return np.clip(denoised, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['bm3d', 'nlm'])
    parser.add_argument('--test_jsonl', default=None,
                        help='Path to test JSONL file (overrides --data_dir)')
    parser.add_argument('--data_dir', default='/home/kumwilai/OCT/pku37_oct_dataset')
    parser.add_argument('--output_dir', default='/home/kumwilai/OCT/outputs/baselines')
    parser.add_argument('--dataset_name', default='PKU37', help='Dataset name for output')
    args = parser.parse_args()

    if args.test_jsonl:
        test_jsonl = args.test_jsonl
    else:
        test_jsonl = os.path.join(args.data_dir, 'pku37_real_test.jsonl')
    if not os.path.exists(test_jsonl):
        print(f"ERROR: {test_jsonl} not found")
        sys.exit(1)

    # Load pairs
    pairs = []
    with open(test_jsonl) as f:
        for line in f:
            entry = json.loads(line.strip())
            pairs.append((entry['noisy_path'], entry['clean_path']))
    print(f"Loaded {len(pairs)} test pairs")

    device = torch.device('cpu')  # classical methods run on CPU

    all_metrics = []
    t0 = time.time()

    for i, (noisy_path, clean_path) in enumerate(pairs):
        # Load images
        noisy_np = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean_np = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        # Denoise
        if args.method == 'bm3d':
            denoised_np = denoise_bm3d_img(noisy_np)
        else:
            denoised_np = denoise_nlm_img(noisy_np)

        # Convert to tensors for metric computation (ensure float32)
        noisy_t = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        clean_t = torch.from_numpy(clean_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        denoised_t = torch.from_numpy(denoised_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        metrics = compute_all_metrics(denoised_t, noisy_t, clean_t)
        all_metrics.append(metrics)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(pairs)}] PSNR={metrics['psnr_denoised']:.2f} "
                  f"SSIM={metrics['ssim_denoised']:.4f} "
                  f"CNR={metrics['cnr_denoised']:.2f} "
                  f"Clinical={metrics['clinical_improved']}/4 "
                  f"({elapsed:.1f}s)")

    # Compute averages
    keys = all_metrics[0].keys()
    avg = {}
    for k in keys:
        avg[k] = float(np.mean([m[k] for m in all_metrics]))

    total_time = time.time() - t0

    # Print summary
    print(f"\n{'='*70}")
    print(f"  {args.method.upper()} Results on {args.dataset_name} ({len(pairs)} images)")
    print(f"{'='*70}")
    print(f"  PSNR:       {avg['psnr_denoised']:.2f} dB (delta: {avg['psnr_delta']:+.2f})")
    print(f"  SSIM:       {avg['ssim_denoised']:.4f} (delta: {avg['ssim_delta']:+.4f})")
    print(f"  CNR:        {avg['cnr_denoised']:.2f} (delta: {avg['cnr_delta']:+.2f}, {avg['cnr_improvement']:+.1f}%)")
    print(f"  Contrast:   ratio={avg['contrast_ratio']:.4f}")
    print(f"  Boundary:   ratio={avg['boundary_ratio']:.4f}")
    print(f"  Texture:    ratio={avg['texture_ratio']:.4f}")
    print(f"  Edge:       ratio={avg['edge_ratio']:.4f}")
    print(f"  Clinical:   {avg['clinical_improved']:.1f}/4")
    print(f"  Time:       {total_time:.1f}s total ({total_time/len(pairs):.2f}s/image)")
    print(f"{'='*70}")

    # Compute clinical improvement percentage (same formula as paper)
    clinical_pct = (
        (avg['contrast_ratio'] - 1) * 100 +
        (avg['boundary_ratio'] - 1) * 100 +
        (avg['texture_ratio'] - 1) * 100 +
        (avg['edge_ratio'] - 1) * 100
    ) / 4
    print(f"  Clinical improvement avg: {clinical_pct:+.1f}%")
    print(f"  Per-dimension:")
    print(f"    Contrast: {(avg['contrast_ratio']-1)*100:+.1f}%")
    print(f"    Boundary: {(avg['boundary_ratio']-1)*100:+.1f}%")
    print(f"    Texture:  {(avg['texture_ratio']-1)*100:+.1f}%")
    print(f"    Edge:     {(avg['edge_ratio']-1)*100:+.1f}%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'method': args.method,
        'n_images': len(pairs),
        'averages': avg,
        'clinical_pct': clinical_pct,
        'per_dimension_pct': {
            'contrast': (avg['contrast_ratio'] - 1) * 100,
            'boundary': (avg['boundary_ratio'] - 1) * 100,
            'texture': (avg['texture_ratio'] - 1) * 100,
            'edge': (avg['edge_ratio'] - 1) * 100,
        },
        'total_time_seconds': total_time,
        'per_image_metrics': all_metrics,
    }
    result['dataset'] = args.dataset_name
    out_path = os.path.join(args.output_dir, f'{args.method}_{args.dataset_name.lower()}_clinical_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
