#!/usr/bin/env python3
"""
Generate noisy variants for OCT datasets.

Creates three noisy folders per split (train/val) and domain (cnv/dme/drusen/normal):
- noisy_moderate_gamma: multiplicative Gamma speckle (moderate noise)
- noisy_heavy_gamma: multiplicative Gamma speckle (heavy noise)
- noisy_gaussian: additive Gaussian noise

Noise models:
- Gamma speckle: y = x * g, g ~ Gamma(k, theta=1/k) so E[g]=1, Var[g]=1/k
  - moderate: k=4.0
  - heavy: k=1.0
- Gaussian: y = x + n, n ~ N(0, sigma^2), default sigma=0.05

All images are clipped to [0, 1] and saved as 8-bit PNG.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import cv2


def add_gaussian_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    y = x + n
    return np.clip(y, 0.0, 1.0)


def add_gamma_speckle(x: np.ndarray, k: float, rng: np.random.Generator) -> np.ndarray:
    # Gamma with shape k and scale 1/k => mean 1, var 1/k
    g = rng.gamma(shape=k, scale=1.0 / k, size=x.shape).astype(np.float32)
    y = x * g
    return np.clip(y, 0.0, 1.0)


def load_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return (img.astype(np.float32) / 255.0)


def save_gray01(path: Path, img01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), img8):
        raise RuntimeError(f"Failed to write image: {path}")


def process_split(split_dir: Path, args: argparse.Namespace) -> None:
    clean_dir = split_dir / 'clean'
    if not clean_dir.exists():
        print(f"Warning: clean dir missing, skipping: {clean_dir}")
        return

    # Output dirs
    out_mod = split_dir / 'noisy_moderate_gamma'
    out_heavy = split_dir / 'noisy_heavy_gamma'
    out_gauss = split_dir / 'noisy_gaussian'

    files = sorted([p for p in clean_dir.glob('*.png')])
    if args.limit > 0:
        files = files[:args.limit]

    rng = np.random.default_rng(args.seed)

    print(f"  Clean images: {len(files)} in {clean_dir}")
    for idx, fp in enumerate(files):
        try:
            img = load_gray01(fp)

            # Independent noise draws per variant
            y_mod = add_gamma_speckle(img, k=args.k_moderate, rng=rng)
            y_hvy = add_gamma_speckle(img, k=args.k_heavy, rng=rng)
            y_gau = add_gaussian_noise(img, sigma=args.gauss_sigma, rng=rng)

            save_gray01(out_mod / fp.name, y_mod)
            save_gray01(out_heavy / fp.name, y_hvy)
            save_gray01(out_gauss / fp.name, y_gau)

            if idx % 200 == 0 or idx == len(files) - 1:
                print(f"    [{idx+1}/{len(files)}] {fp.name}")
        except Exception as e:
            print(f"    Error processing {fp.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate noisy OCT variants')
    parser.add_argument('--root', type=str, default='oct', help='Root OCT directory')
    parser.add_argument('--domains', nargs='*', default=['cnv', 'dme', 'drusen', 'normal'],
                        help='Domains to process (subdirectories of root)')
    parser.add_argument('--splits', nargs='*', default=['train', 'val'], help='Splits to process')
    parser.add_argument('--k_moderate', type=float, default=4.0, help='Gamma shape for moderate speckle (higher => less noise)')
    parser.add_argument('--k_heavy', type=float, default=1.0, help='Gamma shape for heavy speckle (lower => more noise)')
    parser.add_argument('--gauss_sigma', type=float, default=0.05, help='Gaussian sigma (0-1 scale)')
    parser.add_argument('--limit', type=int, default=0, help='Limit images per split (0 = all)')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root path does not exist: {root}")
        sys.exit(1)

    for dom in args.domains:
        dom_dir = root / dom
        if not dom_dir.exists():
            print(f"Domain missing, skipping: {dom_dir}")
            continue
        print(f"\n=== Domain: {dom} ===")
        for split in args.splits:
            split_dir = dom_dir / split
            if not split_dir.exists():
                print(f" Split missing, skipping: {split_dir}")
                continue
            print(f"- Split: {split}")
            process_split(split_dir, args)

    print("\nDone generating noisy variants.")


if __name__ == '__main__':
    main()

