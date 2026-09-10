#!/usr/bin/env python3
"""
Evaluate baseline speckle filters using pairs files.
Modified version of run_baselines_filters.py to work with train_pairs_*.txt files.
"""

import os
import sys
import argparse
import time
import csv
from pathlib import Path
from typing import List, Dict

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


def load_image(path: str, size: int = None) -> np.ndarray:
    """Load and optionally resize image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    img = img.astype(np.float32) / 255.0

    if size is not None and size > 0 and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return img


def load_pairs_file(pairs_path: str) -> List[tuple]:
    """Load pairs from text file."""
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def apply_filter(
    noisy: np.ndarray,
    filter_name: str,
    args: argparse.Namespace,
    enl_global: float = None,
    enl_map: np.ndarray = None
) -> np.ndarray:
    """Apply specified filter to noisy image."""
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


def evaluate_pairs(
    pairs: List[tuple],
    args: argparse.Namespace
) -> List[Dict]:
    """Evaluate filters on all pairs."""
    results = []
    progress_freq = 100  # Print progress every N images

    print(f"\n{'='*80}")
    print(f"Processing {len(pairs)} image pairs")
    print(f"{'='*80}")
    print(f"Filters: {', '.join(args.filters)}")
    if args.image_size > 0:
        print(f"Image size: {args.image_size}x{args.image_size} (resized)")
    else:
        print(f"Image size: Full resolution (no resize)")
    print(f"Window size: {args.win}")
    print(f"Use ENL map: {args.use_enl_map}")
    print()

    for idx, (noisy_path, clean_path) in enumerate(pairs):
        # Load images
        try:
            # Load full resolution noisy for ENL estimation
            noisy_full = load_image(noisy_path, size=None)

            # Estimate ENL on full resolution BEFORE resizing
            enl_global = estimate_enl_global_linear(noisy_full, win=7)
            enl_map_full = estimate_enl_map_linear(noisy_full, win=7) if args.use_enl_map else None

            # Now load resized versions for filtering
            noisy = load_image(noisy_path, size=args.image_size)
            clean = load_image(clean_path, size=args.image_size)

            # Resize ENL map if using local ENL
            if enl_map_full is not None:
                enl_map = cv2.resize(enl_map_full, (noisy.shape[1], noisy.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                enl_map = None
        except Exception as e:
            if idx % progress_freq == 0 or idx == len(pairs) - 1:
                print(f"Error loading pair {idx}: {e}")
            continue

        # Compute baseline metrics (noisy vs clean)
        psnr_noisy = psnr(clean, noisy, data_range=1.0)
        ssim_noisy = ssim(clean, noisy, data_range=1.0)

        # Print progress
        should_print = (idx % progress_freq == 0) or (idx == len(pairs) - 1)

        if should_print:
            print(f"[{idx+1}/{len(pairs)}] {Path(noisy_path).name}")
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
                    'image_id': Path(noisy_path).name,
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
        fieldnames = ['image_id', 'method', 'psnr', 'ssim',
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
        std_psnr = np.std([r['psnr'] for r in method_results])
        avg_ssim = np.mean([r['ssim'] for r in method_results])
        std_ssim = np.std([r['ssim'] for r in method_results])
        avg_psnr_gain = np.mean([r['psnr_gain'] for r in method_results])
        avg_ssim_gain = np.mean([r['ssim_gain'] for r in method_results])
        avg_time = np.mean([r['time_ms'] for r in method_results])

        print(f"{method.upper():<15} {avg_psnr:>7.2f}±{std_psnr:<5.2f}  {avg_ssim:>6.4f}±{std_ssim:<6.4f}  "
              f"{avg_psnr_gain:>+7.2f}        {avg_ssim_gain:>+7.4f}       {avg_time:>8.1f}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline speckle filters using pairs files"
    )

    # Input/output
    parser.add_argument('--pairs_file', type=str, required=True,
                       help='Path to pairs file (e.g., val_pairs_gaussian.txt)')
    parser.add_argument('--output_dir', type=str, default='outputs/filters',
                       help='Output directory for results')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of pairs (0 = all)')

    # Image parameters
    parser.add_argument('--image_size', type=int, default=64,
                       help='Resize images to this size (use 0 for full resolution)')

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
    print("Baseline Speckle Filters Evaluation (from pairs file)")
    print(f"{'='*80}")
    print(f"Pairs file: {args.pairs_file}")
    print(f"Filters: {', '.join(args.filters)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load pairs
    pairs = load_pairs_file(args.pairs_file)
    print(f"Loaded {len(pairs)} pairs from {args.pairs_file}")

    if args.limit > 0:
        pairs = pairs[:args.limit]
        print(f"Limited to {len(pairs)} pairs")

    # Evaluate
    results = evaluate_pairs(pairs, args)

    # Write results
    pairs_name = Path(args.pairs_file).stem
    output_csv = Path(args.output_dir) / f'{pairs_name}_results.csv'
    write_results_csv(results, output_csv)

    # Print summary
    print_summary(results)

    print("Done!")


if __name__ == '__main__':
    main()
