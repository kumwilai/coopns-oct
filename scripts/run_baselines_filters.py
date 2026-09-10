#!/usr/bin/env python3
"""
Baseline Speckle Filters Evaluation Script

Evaluates classical log-domain speckle filters (Lee, Kuan, Frost, Bilateral, Guided)
on OCT validation images and compares with clean ground truth.

Usage:
    python scripts/run_baselines_filters.py \
        --domains data/oct/cnv data/oct/dme \
        --image_size 64 \
        --limit 100 \
        --filters lee kuan frost bilateral guided \
        --win 5 \
        --use_enl_map
"""

import os
import sys
import argparse
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from filters import (
    lee_filter_log,
    kuan_filter_log,
    frost_filter_log,
    bilateral_filter_log,
    guided_filter_log,
    estimate_enl_global_linear,
    estimate_enl_map_linear
)


def load_image(path: Path, size: int = None) -> np.ndarray:
    """
    Load and optionally resize image.

    Args:
        path: Path to image file
        size: If provided and > 0, resize to (size, size). If 0 or None, use original size.

    Returns:
        Image as numpy array in [0, 1], shape (H, W)
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    img = img.astype(np.float32) / 255.0

    if size is not None and size > 0 and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return img


def load_image_2p5d(
    noisy_dir: Path,
    clean_dir: Path,
    filename: str,
    size: int,
    context_radius: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 2.5D images (center slice with context).

    For filters, we use only the center channel for processing and metrics.

    Args:
        noisy_dir: Directory with noisy images
        clean_dir: Directory with clean images
        filename: Filename pattern (e.g., "CNV-123-5.png")
        size: Image size
        context_radius: Number of adjacent slices

    Returns:
        (noisy_center, clean_center) both shape (H, W)
    """
    # For simplicity, just load the center slice
    # Full 2.5D would require loading adjacent slices
    noisy_path = noisy_dir / filename
    clean_path = clean_dir / filename

    noisy = load_image(noisy_path, size=size)
    clean = load_image(clean_path, size=size)

    return noisy, clean


def apply_filter(
    noisy: np.ndarray,
    filter_name: str,
    args: argparse.Namespace,
    enl_global: float = None,
    enl_map: np.ndarray = None
) -> np.ndarray:
    """
    Apply specified filter to noisy image.

    Args:
        noisy: Noisy image in [0, 1]
        filter_name: Name of filter ('lee', 'kuan', 'frost', 'bilateral', 'guided')
        args: Parsed arguments
        enl_global: Pre-computed global ENL
        enl_map: Pre-computed ENL map

    Returns:
        Denoised image in [0, 1]
    """
    filter_name = filter_name.lower()

    # Determine which ENL to use
    if args.use_enl_map:
        enl_to_use = None
        enl_map_to_use = enl_map
    else:
        enl_to_use = enl_global
        enl_map_to_use = None

    if filter_name == 'lee':
        return lee_filter_log(noisy, win=args.win, enl=enl_to_use, enl_map=enl_map_to_use)

    elif filter_name == 'kuan':
        return kuan_filter_log(noisy, win=args.win, enl=enl_to_use, enl_map=enl_map_to_use)

    elif filter_name == 'frost':
        return frost_filter_log(noisy, win=args.win, enl=enl_to_use, enl_map=enl_map_to_use)

    elif filter_name == 'bilateral':
        range_sigma = args.bilateral_sigma / np.sqrt(enl_global + 1e-12)
        spatial_sigma = (args.win // 2) / 2.0
        return bilateral_filter_log(
            noisy,
            spatial_sigma=spatial_sigma,
            range_sigma=range_sigma,
            win=args.win,
            enl=enl_global
        )

    elif filter_name == 'guided':
        return guided_filter_log(
            noisy,
            guide=None,
            radius=args.guided_radius,
            eps=args.guided_eps
        )

    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def process_domain(
    domain_path: Path,
    args: argparse.Namespace,
    output_csv: Path
) -> List[Dict]:
    """
    Process all validation images in a domain.

    Args:
        domain_path: Path to domain directory (e.g., data/oct/cnv)
        args: Parsed arguments
        output_csv: Path to output CSV file

    Returns:
        List of result dictionaries
    """
    domain_name = domain_path.name

    # Find validation images
    val_noisy_dir = domain_path / 'val' / 'noisy'
    val_clean_dir = domain_path / 'val' / 'clean'

    if not val_noisy_dir.exists() or not val_clean_dir.exists():
        print(f"Warning: Validation directories not found for {domain_name}")
        return []

    # Get list of images
    noisy_files = sorted(list(val_noisy_dir.glob("*.png")))

    if args.limit > 0:
        noisy_files = noisy_files[:args.limit]

    print(f"\n{'='*80}")
    print(f"Processing domain: {domain_name}")
    print(f"{'='*80}")
    print(f"Images: {len(noisy_files)}")
    print(f"Filters: {', '.join(args.filters)}")
    if args.image_size > 0:
        print(f"Image size: {args.image_size}x{args.image_size} (resized)")
    else:
        print(f"Image size: Full resolution (no resize)")
    print(f"Window size: {args.win}")
    print(f"Use ENL map: {args.use_enl_map}")
    print()

    results = []
    progress_freq = 100  # Print progress every N images

    for idx, noisy_path in enumerate(noisy_files):
        clean_path = val_clean_dir / noisy_path.name

        if not clean_path.exists():
            if idx % progress_freq == 0 or idx == len(noisy_files) - 1:
                print(f"Warning: Clean image not found for {noisy_path.name}")
            continue

        # Load images
        try:
            # Load full resolution noisy for ENL estimation
            noisy_full = load_image(noisy_path, size=None)

            # Estimate ENL on full resolution BEFORE resizing
            enl_global = estimate_enl_global_linear(noisy_full, win=7)
            enl_map_full = estimate_enl_map_linear(noisy_full, win=7) if args.use_enl_map else None

            if args.input_2p5d:
                noisy, clean = load_image_2p5d(
                    val_noisy_dir,
                    val_clean_dir,
                    noisy_path.name,
                    args.image_size,
                    args.context_radius
                )
            else:
                # Now load resized versions for filtering
                noisy = load_image(noisy_path, size=args.image_size)
                clean = load_image(clean_path, size=args.image_size)

            # Resize ENL map if using local ENL
            if enl_map_full is not None:
                import cv2
                enl_map = cv2.resize(enl_map_full, (noisy.shape[1], noisy.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                enl_map = None
        except Exception as e:
            if idx % progress_freq == 0 or idx == len(noisy_files) - 1:
                print(f"Error loading {noisy_path.name}: {e}")
            continue

        # ENL already estimated above (on full resolution)

        # Compute baseline metrics (noisy vs clean)
        psnr_noisy = psnr(clean, noisy, data_range=1.0)
        ssim_noisy = ssim(clean, noisy, data_range=1.0)

        # Print progress every N images or on last image
        should_print = (idx % progress_freq == 0) or (idx == len(noisy_files) - 1)

        if should_print:
            print(f"[{idx+1}/{len(noisy_files)}] {noisy_path.name}")
            print(f"  ENL: {enl_global:.2f}")
            print(f"  Baseline - PSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}")

        # Apply each filter
        for filter_name in args.filters:
            t_start = time.time()

            try:
                denoised = apply_filter(noisy, filter_name, args, enl_global, enl_map)

                # Compute metrics
                psnr_denoised = psnr(clean, denoised, data_range=1.0)
                ssim_denoised = ssim(clean, denoised, data_range=1.0)

                t_elapsed = (time.time() - t_start) * 1000  # ms

                # Record result
                result = {
                    'domain': domain_name,
                    'image_id': noisy_path.name,
                    'method': filter_name,
                    'psnr': psnr_denoised,
                    'ssim': ssim_denoised,
                    'psnr_gain': psnr_denoised - psnr_noisy,
                    'ssim_gain': ssim_denoised - ssim_noisy,
                    'enl_global': enl_global,
                    'time_ms': t_elapsed
                }

                results.append(result)

                if should_print:
                    print(f"  {filter_name.upper():12s} - PSNR: {psnr_denoised:.2f} dB (+{result['psnr_gain']:+.2f}), "
                          f"SSIM: {ssim_denoised:.4f} (+{result['ssim_gain']:+.4f}), Time: {t_elapsed:.1f} ms")

            except Exception as e:
                if should_print:
                    print(f"  {filter_name.upper():12s} - ERROR: {e}")

    return results


def write_results_csv(results: List[Dict], output_path: Path):
    """Write results to CSV file."""
    if not results:
        print("No results to write")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['domain', 'image_id', 'method', 'psnr', 'ssim',
                     'psnr_gain', 'ssim_gain', 'enl_global', 'time_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\nResults written to: {output_path}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    if not results:
        return

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    # Group by method
    methods = sorted(set(r['method'] for r in results))

    print(f"\n{'Method':<15} {'PSNR (dB)':<15} {'SSIM':<15} {'PSNR Gain':<15} {'SSIM Gain':<15} {'Time (ms)'}")
    print(f"{'-'*90}")

    for method in methods:
        method_results = [r for r in results if r['method'] == method]

        avg_psnr = np.mean([r['psnr'] for r in method_results])
        avg_ssim = np.mean([r['ssim'] for r in method_results])
        avg_psnr_gain = np.mean([r['psnr_gain'] for r in method_results])
        avg_ssim_gain = np.mean([r['ssim_gain'] for r in method_results])
        avg_time = np.mean([r['time_ms'] for r in method_results])

        print(f"{method.upper():<15} {avg_psnr:>8.2f}       {avg_ssim:>8.4f}     "
              f"{avg_psnr_gain:>+8.2f}        {avg_ssim_gain:>+8.4f}       {avg_time:>8.1f}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline speckle filters on OCT images"
    )

    # Input/output
    parser.add_argument('--domains', nargs='+', default=['data/oct/dme'],
                       help='Paths to domain directories')
    parser.add_argument('--output_dir', type=str, default='outputs/filters',
                       help='Output directory for results')

    # Image parameters
    parser.add_argument('--image_size', type=int, default=0,
                       help='Resize images to this size (use 0 for full resolution, no resize)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Limit number of images per domain (0 = all)')
    parser.add_argument('--input_2p5d', action='store_true',
                       help='Use 2.5D input (use center channel for filtering)')
    parser.add_argument('--context_radius', type=int, default=1,
                       help='Context radius for 2.5D input')

    # Filter selection
    parser.add_argument('--filters', nargs='+',
                       default=['lee', 'kuan', 'frost', 'bilateral', 'guided'],
                       choices=['lee', 'kuan', 'frost', 'bilateral', 'guided'],
                       help='Filters to evaluate')

    # Filter parameters
    parser.add_argument('--win', type=int, default=5,
                       help='Window size for local filters')
    parser.add_argument('--use_enl_map', action='store_true',
                       help='Use local ENL map instead of global ENL')

    # Bilateral filter parameters
    parser.add_argument('--bilateral_sigma', type=float, default=0.15,
                       help='Range sigma coefficient for bilateral (k in k/sqrt(ENL))')

    # Guided filter parameters
    parser.add_argument('--guided_radius', type=int, default=4,
                       help='Radius for guided filter')
    parser.add_argument('--guided_eps', type=float, default=1e-3,
                       help='Regularization for guided filter')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Baseline Speckle Filters Evaluation")
    print(f"{'='*80}")
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Filters: {', '.join(args.filters)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Process each domain
    all_results = []

    for domain_str in args.domains:
        domain_path = Path(domain_str)

        if not domain_path.exists():
            print(f"Warning: Domain path does not exist: {domain_path}")
            continue

        results = process_domain(domain_path, args, Path(args.output_dir))
        all_results.extend(results)

    # Write combined results
    output_csv = Path(args.output_dir) / 'results.csv'
    write_results_csv(all_results, output_csv)

    # Print summary
    print_summary(all_results)

    print("Done!")


if __name__ == '__main__':
    main()
