#!/usr/bin/env python3
"""
Evaluate the S0895611125001065 paper method on OCT validation images.

This script mirrors the I/O and reporting style of scripts/run_baselines_filters.py
to make comparisons straightforward (PSNR/SSIM vs. clean targets).

Usage:
    python scripts/run_paper_method.py \
        --domains oct/cnv oct/dme \
        --image_size 0 \
        --limit 100
"""

import argparse
from pathlib import Path
import csv
import time

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods import PaperMethod, PaperConfig  # noqa: E402


def load_image(path: Path, size: int | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = img.astype(np.float32) / 255.0
    if size is not None and size > 0 and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def process_domain(domain_path: Path, args: argparse.Namespace, method: PaperMethod) -> list[dict]:
    domain_name = domain_path.name

    val_noisy_dir = domain_path / 'val' / 'noisy'
    val_clean_dir = domain_path / 'val' / 'clean'
    if not val_noisy_dir.exists() or not val_clean_dir.exists():
        print(f"Warning: Validation directories not found for {domain_name}")
        return []

    noisy_files = sorted(list(val_noisy_dir.glob("*.png")))
    if args.limit > 0:
        noisy_files = noisy_files[:args.limit]

    print(f"\n{'='*80}")
    print(f"Processing domain: {domain_name}")
    print(f"{'='*80}")
    print(f"Images: {len(noisy_files)}")
    print(f"Image size: {'Full resolution' if args.image_size == 0 else f'{args.image_size}x{args.image_size}'}")

    results: list[dict] = []
    for idx, noisy_path in enumerate(noisy_files):
        clean_path = val_clean_dir / noisy_path.name
        if not clean_path.exists():
            if idx % 100 == 0 or idx == len(noisy_files) - 1:
                print(f"Warning: Clean image not found for {noisy_path.name}")
            continue

        noisy = load_image(noisy_path, size=args.image_size if args.image_size > 0 else None)
        clean = load_image(clean_path, size=args.image_size if args.image_size > 0 else None)

        # Baseline metrics
        psnr_noisy = psnr(clean, noisy, data_range=1.0)
        ssim_noisy = ssim(clean, noisy, data_range=1.0)

        t0 = time.time()
        denoised = method.apply(noisy, meta={"domain": domain_name, "image_id": noisy_path.name})
        dt_ms = (time.time() - t0) * 1000.0

        psnr_d = psnr(clean, denoised, data_range=1.0)
        ssim_d = ssim(clean, denoised, data_range=1.0)

        results.append({
            'domain': domain_name,
            'image_id': noisy_path.name,
            'method': 'S0895611125001065',
            'psnr': psnr_d,
            'ssim': ssim_d,
            'psnr_gain': psnr_d - psnr_noisy,
            'ssim_gain': ssim_d - ssim_noisy,
            'time_ms': dt_ms,
        })

        if idx % 100 == 0 or idx == len(noisy_files) - 1:
            print(f"[{idx+1}/{len(noisy_files)}] {noisy_path.name} -> "
                  f"PSNR {psnr_d:.2f} dB (∆ {psnr_d-psnr_noisy:+.2f}), "
                  f"SSIM {ssim_d:.4f} (∆ {ssim_d-ssim_noisy:+.4f}), "
                  f"{dt_ms:.1f} ms")

    return results


def write_results_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        print("No results to write")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['domain', 'image_id', 'method', 'psnr', 'ssim',
                      'psnr_gain', 'ssim_gain', 'time_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to: {output_path}")


def print_summary(results: list[dict]) -> None:
    if not results:
        return
    import numpy as np
    print(f"\n{'='*80}\nSUMMARY STATISTICS\n{'='*80}")
    psnrs = np.array([r['psnr'] for r in results])
    ssims = np.array([r['ssim'] for r in results])
    gains_p = np.array([r['psnr_gain'] for r in results])
    gains_s = np.array([r['ssim_gain'] for r in results])
    times = np.array([r['time_ms'] for r in results])
    print(f"PSNR: mean {psnrs.mean():.2f} dB, median {np.median(psnrs):.2f} dB")
    print(f"SSIM: mean {ssims.mean():.4f}, median {np.median(ssims):.4f}")
    print(f"Gains: PSNR {gains_p.mean():+.2f} dB, SSIM {gains_s.mean():+.4f}")
    print(f"Time: mean {times.mean():.1f} ms")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run S0895611125001065 method on OCT images")
    parser.add_argument('--domains', nargs='+', default=['oct/dme'], help='Paths to domain directories')
    parser.add_argument('--output_dir', type=str, default='outputs/paper_method', help='Output directory')
    parser.add_argument('--image_size', type=int, default=0, help='0 to keep original resolution')
    parser.add_argument('--limit', type=int, default=100, help='Limit images per domain (0 = all)')

    # Add algorithm-specific CLI params here once known

    args = parser.parse_args()

    print(f"\n{'='*80}\nS0895611125001065 Method Evaluation\n{'='*80}")
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    config = PaperConfig(image_size=args.image_size)
    method = PaperMethod(config=config)

    all_results: list[dict] = []
    for domain_str in args.domains:
        domain_path = Path(domain_str)
        if not domain_path.exists():
            print(f"Warning: Domain path does not exist: {domain_path}")
            continue
        results = process_domain(domain_path, args, method)
        all_results.extend(results)

    output_csv = Path(args.output_dir) / 'results.csv'
    write_results_csv(all_results, output_csv)
    print_summary(all_results)
    print("Done!")


if __name__ == '__main__':
    main()

