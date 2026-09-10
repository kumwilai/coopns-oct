#!/usr/bin/env python3
"""
Baseline Denoising Methods for OCT Comparison
Implements BM3D and other classical methods with evaluation metrics.
"""

import os
import sys
import argparse
import glob
from typing import Optional, Tuple, List, Dict
import time

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import json

# Try to import bm3d (install with: pip install bm3d)
try:
    import bm3d
    from bm3d import BM3DStages
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False
    BM3DStages = None
    print("WARNING: bm3d not installed. Install with: pip install bm3d")

# Import metrics from scikit-image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ========================================
# Image I/O Utilities
# ========================================

def load_image_grayscale(path: str, resize_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image as grayscale numpy array in [0, 1] range, with optional resize."""
    img = Image.open(path).convert('L')
    if resize_hw is not None:
        h, w = resize_hw
        resample = getattr(Image, "Resampling", Image).BILINEAR
        img = img.resize((w, h), resample=resample)
    img_np = np.array(img, dtype=np.float32) / 255.0
    return img_np


def save_image_grayscale(img: np.ndarray, path: str):
    """Save grayscale numpy array [0, 1] as 8-bit image."""
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode='L').save(path)


# ========================================
# Evaluation Metrics
# ========================================

def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Compute PSNR between two images."""
    return float(psnr(img1, img2, data_range=data_range))


def compute_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Compute SSIM between two images."""
    return float(ssim(img1, img2, data_range=data_range))


def compute_metrics(denoised: np.ndarray, clean: np.ndarray) -> Dict[str, float]:
    """Compute all metrics between denoised and clean images."""
    metrics = {
        'psnr': compute_psnr(denoised, clean),
        'ssim': compute_ssim(denoised, clean),
        'mse': float(np.mean((denoised - clean) ** 2))
    }
    return metrics


# ========================================
# BM3D Denoising
# ========================================

def denoise_bm3d(
    noisy: np.ndarray,
    sigma_psd: float = 0.05,
    stage_arg: str = 'all_stages'
) -> np.ndarray:
    """
    Denoise image using BM3D algorithm.

    Args:
        noisy: Input noisy image [H, W] in [0, 1]
        sigma_psd: Noise standard deviation (0.01-0.1 for OCT)
        stage_arg: 'hard_thresholding', 'wiener_filtering', or 'all_stages'

    Returns:
        Denoised image [H, W] in [0, 1]
    """
    if not BM3D_AVAILABLE:
        raise ImportError("bm3d not installed. Install with: pip install bm3d")

    # Convert string to BM3DStages enum
    stage_map = {
        'hard_thresholding': BM3DStages.HARD_THRESHOLDING,
        'wiener_filtering': BM3DStages.WIENER_FILTERING,
        'all_stages': BM3DStages.ALL_STAGES
    }
    stage_enum = stage_map.get(stage_arg, BM3DStages.ALL_STAGES)

    # BM3D expects images in [0, 1]
    denoised = bm3d.bm3d(noisy, sigma_psd=sigma_psd, stage_arg=stage_enum)

    # Clip to valid range
    denoised = np.clip(denoised, 0.0, 1.0)

    return denoised


def auto_estimate_sigma(noisy: np.ndarray, method: str = 'mad') -> float:
    """
    Automatically estimate noise standard deviation from noisy image.

    Args:
        noisy: Noisy image [H, W]
        method: 'mad' (Median Absolute Deviation) or 'percentile'

    Returns:
        Estimated sigma
    """
    if method == 'mad':
        # Use high-pass filtered image to estimate noise
        from scipy.ndimage import laplace
        laplacian = laplace(noisy)
        sigma = np.median(np.abs(laplacian)) / 0.6745
    elif method == 'percentile':
        # Use difference between adjacent pixels
        diff_h = np.diff(noisy, axis=0)
        diff_v = np.diff(noisy, axis=1)
        sigma = np.percentile(np.abs(np.concatenate([diff_h.flatten(), diff_v.flatten()])), 75) / 0.6745
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(sigma)


# ========================================
# Non-Local Means Denoising
# ========================================

def denoise_nlm(noisy: np.ndarray, h: float = 0.1, patch_size: int = 7, patch_distance: int = 11) -> np.ndarray:
    """
    Denoise using Non-Local Means.

    Args:
        noisy: Input image [H, W] in [0, 1]
        h: Filtering parameter (higher = more smoothing)
        patch_size: Size of patches to compare
        patch_distance: Search window size

    Returns:
        Denoised image
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma

    # Estimate noise sigma if needed
    sigma_est = estimate_sigma(noisy)

    # Apply NLM
    denoised = denoise_nl_means(
        noisy,
        h=h * sigma_est,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True
    )

    return np.clip(denoised, 0.0, 1.0)


# ========================================
# Batch Processing
# ========================================

def process_image_pair(
    noisy_path: str,
    clean_path: str,
    output_dir: str,
    method: str = 'bm3d',
    sigma: Optional[float] = None,
    save_output: bool = True,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Process a single noisy-clean image pair.

    Args:
        noisy_path: Path to noisy image
        clean_path: Path to clean image (ground truth)
        output_dir: Directory to save denoised image
        method: 'bm3d' or 'nlm'
        sigma: Noise sigma (auto-estimated if None)
        save_output: Whether to save denoised image

    Returns:
        Dictionary with metrics
    """
    # Load images
    noisy = load_image_grayscale(noisy_path, resize_hw=resize_hw)
    clean = load_image_grayscale(clean_path, resize_hw=resize_hw)

    # Auto-estimate sigma if not provided
    if sigma is None:
        sigma = auto_estimate_sigma(noisy)
        print(f"  Auto-estimated sigma: {sigma:.4f}")

    # Denoise
    start_time = time.time()
    if method == 'bm3d':
        denoised = denoise_bm3d(noisy, sigma_psd=sigma)
    elif method == 'nlm':
        denoised = denoise_nlm(noisy, h=sigma)
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = time.time() - start_time

    # Compute metrics
    metrics = compute_metrics(denoised, clean)
    metrics['time_seconds'] = elapsed
    metrics['sigma_used'] = sigma

    # Save denoised image
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(noisy_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_{method}_denoised{ext}")
        save_image_grayscale(denoised, output_path)
        print(f"  Saved: {output_path}")

    return metrics


def batch_evaluate(
    pair_list_path: str,
    output_dir: str,
    method: str = 'bm3d',
    sigma: Optional[float] = None,
    save_outputs: bool = True,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, any]:
    """
    Batch evaluate denoising method on multiple image pairs.

    Args:
        pair_list_path: Path to text file with "noisy_path,clean_path" per line
        output_dir: Directory to save results
        method: Denoising method
        sigma: Noise sigma (auto-estimated if None)
        save_outputs: Whether to save denoised images

    Returns:
        Summary statistics
    """
    print(f"\n{'='*80}")
    print(f"BATCH EVALUATION: {method.upper()}")
    print(f"{'='*80}\n")

    # Read pair list
    pairs = []
    with open(pair_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))

    print(f"Found {len(pairs)} image pairs\n")

    # Process each pair
    all_metrics = []
    for i, (noisy_path, clean_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] Processing: {os.path.basename(noisy_path)}")

        try:
            metrics = process_image_pair(
                noisy_path, clean_path, output_dir,
                method=method, sigma=sigma, save_output=save_outputs, resize_hw=resize_hw
            )
            all_metrics.append(metrics)
            print(f"  PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f} | Time: {metrics['time_seconds']:.2f}s\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

    # Compute summary statistics
    if not all_metrics:
        print("No images processed successfully!")
        return {}

    summary = {
        'method': method,
        'num_images': len(all_metrics),
        'mean_psnr': np.mean([m['psnr'] for m in all_metrics]),
        'std_psnr': np.std([m['psnr'] for m in all_metrics]),
        'mean_ssim': np.mean([m['ssim'] for m in all_metrics]),
        'std_ssim': np.std([m['ssim'] for m in all_metrics]),
        'mean_time': np.mean([m['time_seconds'] for m in all_metrics]),
        'total_time': np.sum([m['time_seconds'] for m in all_metrics]),
        'per_image_metrics': all_metrics
    }

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"Method:      {summary['method']}")
    print(f"Images:      {summary['num_images']}")
    print(f"PSNR:        {summary['mean_psnr']:.2f} ± {summary['std_psnr']:.2f} dB")
    print(f"SSIM:        {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    print(f"Avg Time:    {summary['mean_time']:.2f} seconds/image")
    print(f"Total Time:  {summary['total_time']:.1f} seconds")
    print(f"{'='*80}\n")

    # Save results to JSON
    results_path = os.path.join(output_dir, f"{method}_results.json")
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {results_path}\n")

    return summary


# ========================================
# Main CLI
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline Denoising Methods Evaluation for OCT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate BM3D on test set
  python eval_baselines.py --method bm3d --pair_list val_pairs.txt --output_dir results/bm3d

  # Evaluate with custom sigma
  python eval_baselines.py --method bm3d --pair_list val_pairs.txt --sigma 0.05 --output_dir results/bm3d_s005

  # Evaluate Non-Local Means
  python eval_baselines.py --method nlm --pair_list val_pairs.txt --output_dir results/nlm

  # Single image denoising
  python eval_baselines.py --method bm3d --noisy_image test.png --clean_image gt.png --output_dir results/single
        """
    )

    parser.add_argument('--method', type=str, default='bm3d', choices=['bm3d', 'nlm'],
                        help='Denoising method')
    parser.add_argument('--pair_list', type=str, default=None,
                        help='Path to pair list file (noisy,clean per line)')
    parser.add_argument('--noisy_image', type=str, default=None,
                        help='Single noisy image path')
    parser.add_argument('--clean_image', type=str, default=None,
                        help='Single clean image path (ground truth)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Noise standard deviation (auto-estimated if not provided)')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save denoised images (only compute metrics)')
    parser.add_argument('--resize_h', type=int, default=None,
                        help='Optional resize height before denoising/metrics (set with --resize_w)')
    parser.add_argument('--resize_w', type=int, default=None,
                        help='Optional resize width before denoising/metrics (set with --resize_h)')

    args = parser.parse_args()

    resize_hw = None
    if args.resize_h is not None or args.resize_w is not None:
        if args.resize_h is None or args.resize_w is None:
            parser.error("--resize_h and --resize_w must be set together")
        resize_hw = (args.resize_h, args.resize_w)
        print(f"Resizing evaluation images to {resize_hw[0]}x{resize_hw[1]}")

    # Check if BM3D is available
    if args.method == 'bm3d' and not BM3D_AVAILABLE:
        print("ERROR: BM3D not available. Install with: pip install bm3d")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Batch processing
    if args.pair_list:
        batch_evaluate(
            args.pair_list,
            args.output_dir,
            method=args.method,
            sigma=args.sigma,
            save_outputs=not args.no_save,
            resize_hw=resize_hw
        )

    # Single image processing
    elif args.noisy_image and args.clean_image:
        print(f"\nProcessing single image: {args.noisy_image}\n")
        metrics = process_image_pair(
            args.noisy_image,
            args.clean_image,
            args.output_dir,
            method=args.method,
            sigma=args.sigma,
            save_output=not args.no_save,
            resize_hw=resize_hw
        )
        print(f"\nResults:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  Time: {metrics['time_seconds']:.2f} seconds\n")

    else:
        print("ERROR: Must provide either --pair_list OR (--noisy_image AND --clean_image)")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
