#!/usr/bin/env python3
"""
Compute PSNR/SSIM metrics matching SNA-SKAN paper methodology.

Key finding: The paper computes SSIM only on "meaningful" pixels where
the clean image intensity is above a threshold (~100 in 0-255 range).
This focuses metrics on actual retinal structure, not dark background.

Usage:
    python compute_sota_metrics.py --clean clean.tif --noisy noisy.tif
    python compute_sota_metrics.py --dataset duke17
"""

import argparse
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_masked_metrics(clean, noisy, intensity_threshold=100):
    """
    Compute PSNR and SSIM on meaningful region only.

    The SNA-SKAN paper computes metrics only on pixels where
    the clean image intensity is above a threshold, focusing
    on retinal structure and ignoring dark background noise.

    Args:
        clean: Clean image (numpy array, uint8 or float)
        noisy: Noisy image (numpy array, uint8 or float)
        intensity_threshold: Pixel intensity threshold (default 100 for uint8)

    Returns:
        dict with psnr, ssim, and mask statistics
    """
    # Ensure float64 for computation
    if clean.max() <= 1.0:
        clean = (clean * 255).astype(np.float64)
        noisy = (noisy * 255).astype(np.float64)
    else:
        clean = clean.astype(np.float64)
        noisy = noisy.astype(np.float64)

    # Create mask based on clean image intensity
    mask = clean > intensity_threshold
    n_pixels = np.sum(mask)

    if n_pixels < 1000:
        return {
            'psnr': None,
            'ssim': None,
            'n_pixels': n_pixels,
            'mask_ratio': n_pixels / clean.size
        }

    # Extract masked pixels
    clean_masked = clean[mask]
    noisy_masked = noisy[mask]

    # Compute PSNR on masked region
    mse = np.mean((clean_masked / 255.0 - noisy_masked / 255.0) ** 2)
    psnr_val = 10 * np.log10(1.0 / mse) if mse > 0 else 100

    # Compute SSIM: reshape masked pixels to square for spatial computation
    side = int(np.sqrt(n_pixels))
    clean_sq = clean_masked[:side*side].reshape(side, side) / 255.0
    noisy_sq = noisy_masked[:side*side].reshape(side, side) / 255.0
    ssim_val = ssim(clean_sq, noisy_sq, data_range=1.0)

    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'n_pixels': n_pixels,
        'mask_ratio': n_pixels / clean.size
    }


def compute_full_metrics(clean, noisy):
    """
    Compute standard PSNR and SSIM on full image.

    Args:
        clean: Clean image (numpy array)
        noisy: Noisy image (numpy array)

    Returns:
        dict with psnr and ssim
    """
    if clean.max() <= 1.0:
        clean_norm = clean.astype(np.float64)
        noisy_norm = noisy.astype(np.float64)
        data_range = 1.0
    else:
        clean_norm = clean.astype(np.float64) / 255.0
        noisy_norm = noisy.astype(np.float64) / 255.0
        data_range = 1.0

    psnr_val = psnr(clean_norm, noisy_norm, data_range=data_range)
    ssim_val = ssim(clean_norm, noisy_norm, data_range=data_range)

    return {
        'psnr': psnr_val,
        'ssim': ssim_val
    }


def evaluate_duke17(dataset_dir, intensity_threshold=100):
    """
    Evaluate metrics on Duke17 dataset matching SNA-SKAN paper.

    Args:
        dataset_dir: Path to Sparsity_SDOCT_DATASET_2012 directory
        intensity_threshold: Threshold for masked metrics

    Returns:
        dict with mean metrics
    """
    subjects = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])

    all_psnr_full = []
    all_ssim_full = []
    all_psnr_masked = []
    all_ssim_masked = []

    for subject in subjects:
        subject_dir = os.path.join(dataset_dir, subject)
        files = os.listdir(subject_dir)

        clean_files = [f for f in files if 'Averaged' in f]
        noisy_files = [f for f in files if 'Raw' in f]

        if not clean_files or not noisy_files:
            continue

        clean = np.array(Image.open(os.path.join(subject_dir, clean_files[0])))
        noisy = np.array(Image.open(os.path.join(subject_dir, noisy_files[0])))

        # Full image metrics
        full_metrics = compute_full_metrics(clean, noisy)
        all_psnr_full.append(full_metrics['psnr'])
        all_ssim_full.append(full_metrics['ssim'])

        # Masked metrics (matching paper)
        masked_metrics = compute_masked_metrics(clean, noisy, intensity_threshold)
        if masked_metrics['psnr'] is not None:
            all_psnr_masked.append(masked_metrics['psnr'])
            all_ssim_masked.append(masked_metrics['ssim'])

    return {
        'n_subjects': len(subjects),
        'full': {
            'psnr': np.mean(all_psnr_full),
            'ssim': np.mean(all_ssim_full)
        },
        'masked': {
            'psnr': np.mean(all_psnr_masked),
            'ssim': np.mean(all_ssim_masked),
            'threshold': intensity_threshold
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Compute metrics matching SNA-SKAN paper')
    parser.add_argument('--clean', help='Path to clean image')
    parser.add_argument('--noisy', help='Path to noisy image')
    parser.add_argument('--denoised', help='Path to denoised image (optional)')
    parser.add_argument('--dataset', choices=['duke17'], help='Evaluate on dataset')
    parser.add_argument('--dataset_dir', default='duke_sota_datasets/Sparsity_SDOCT_DATASET_2012')
    parser.add_argument('--threshold', type=int, default=100, help='Intensity threshold for masking')
    args = parser.parse_args()

    if args.dataset == 'duke17':
        print("=" * 60)
        print("DUKE17 EVALUATION (SNA-SKAN Paper Methodology)")
        print("=" * 60)
        print()

        results = evaluate_duke17(args.dataset_dir, args.threshold)

        print(f"Number of subjects: {results['n_subjects']}")
        print()
        print("Full Image Metrics:")
        print(f"  PSNR: {results['full']['psnr']:.3f} dB")
        print(f"  SSIM: {results['full']['ssim']:.4f}")
        print()
        print(f"Masked Metrics (threshold={args.threshold}):")
        print(f"  PSNR: {results['masked']['psnr']:.3f} dB")
        print(f"  SSIM: {results['masked']['ssim']:.3f}")
        print()
        print("SNA-SKAN Paper Reports:")
        print("  PSNR: 16.273 dB")
        print("  SSIM: 0.311")
        print()
        print("=" * 60)

    elif args.clean and args.noisy:
        clean = np.array(Image.open(args.clean))
        noisy = np.array(Image.open(args.noisy))

        print("Full Image Metrics:")
        full = compute_full_metrics(clean, noisy)
        print(f"  PSNR: {full['psnr']:.3f} dB")
        print(f"  SSIM: {full['ssim']:.4f}")

        print(f"\nMasked Metrics (threshold={args.threshold}):")
        masked = compute_masked_metrics(clean, noisy, args.threshold)
        print(f"  PSNR: {masked['psnr']:.3f} dB")
        print(f"  SSIM: {masked['ssim']:.3f}")
        print(f"  Mask ratio: {masked['mask_ratio']*100:.1f}%")

        if args.denoised:
            denoised = np.array(Image.open(args.denoised))
            print(f"\nDenoised Metrics:")
            den_full = compute_full_metrics(clean, denoised)
            den_masked = compute_masked_metrics(clean, denoised, args.threshold)
            print(f"  Full - PSNR: {den_full['psnr']:.3f} dB, SSIM: {den_full['ssim']:.4f}")
            print(f"  Masked - PSNR: {den_masked['psnr']:.3f} dB, SSIM: {den_masked['ssim']:.3f}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
