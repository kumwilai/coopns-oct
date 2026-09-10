#!/usr/bin/env python3
"""
Generate Rayleigh and Poisson noise variants for OCT images.
Creates noisy images and corresponding pairs files for training/validation.
"""

import os
import sys
from pathlib import Path
import json
from tqdm import tqdm

import torch
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_oct_denoise import (
    add_rayleigh_noise,
    add_poisson_noise,
)


def create_noisy_variants(
    clean_paths: list,
    output_dir: Path,
    noise_type: str,
    noise_params: dict
):
    """
    Generate noisy versions of clean images.

    Args:
        clean_paths: List of paths to clean images
        output_dir: Output directory for noisy images
        noise_type: 'rayleigh' or 'poisson'
        noise_params: Parameters for noise generation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {noise_type} noise variants...")
    print(f"  Input: {len(clean_paths)} clean images")
    print(f"  Output: {output_dir}")
    print(f"  Params: {noise_params}")

    pairs = []

    for clean_path in tqdm(clean_paths, desc=f"Processing {noise_type}"):
        # Load clean image
        img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {clean_path}")
            continue

        # Convert to tensor [1, 1, H, W]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Apply noise
        if noise_type == 'rayleigh':
            noisy = add_rayleigh_noise(
                img_tensor,
                scale_range=noise_params.get('scale_range', (0.2, 0.9))
            )
        elif noise_type == 'poisson':
            noisy = add_poisson_noise(
                img_tensor,
                lambda_scale=noise_params.get('lambda_scale', 1.0)
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Convert back to uint8 image
        noisy_np = noisy.squeeze().numpy()
        noisy_uint8 = (noisy_np * 255).clip(0, 255).astype(np.uint8)

        # Generate output path
        clean_path_obj = Path(clean_path)
        output_path = output_dir / clean_path_obj.name

        # Save noisy image
        cv2.imwrite(str(output_path), noisy_uint8)

        # Store pair (noisy_path, clean_path)
        pairs.append((str(output_path), clean_path))

    print(f"  Generated {len(pairs)} noisy images")
    return pairs


def load_splits(splits_file: str = 'oct_splits.json'):
    """Load train/val splits from JSON file."""
    with open(splits_file, 'r') as f:
        return json.load(f)


def write_pairs_file(pairs: list, output_file: Path):
    """Write pairs to text file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for noisy_path, clean_path in pairs:
            f.write(f"{noisy_path},{clean_path}\n")

    print(f"  Saved {len(pairs)} pairs to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Rayleigh and Poisson noise pairs")
    parser.add_argument('--splits_file', type=str, default='oct_splits.json',
                       help='Path to splits JSON file')
    parser.add_argument('--noise_types', nargs='+',
                       default=['rayleigh', 'poisson'],
                       choices=['rayleigh', 'poisson'],
                       help='Noise types to generate')
    parser.add_argument('--rayleigh_scale_range', nargs=2, type=float,
                       default=[0.2, 0.9],
                       help='Scale range for Rayleigh noise')
    parser.add_argument('--poisson_lambda', type=float, default=1.0,
                       help='Lambda scale for Poisson noise')

    args = parser.parse_args()

    print("="*80)
    print("Rayleigh and Poisson Noise Pair Generation")
    print("="*80)

    # Load splits
    print(f"\nLoading splits from {args.splits_file}...")
    splits = load_splits(args.splits_file)

    # Collect all clean paths
    train_clean_paths = []
    val_clean_paths = []

    for domain in ['cnv', 'dme', 'drusen', 'normal']:
        if domain in splits['train']:
            train_clean_paths.extend(splits['train'][domain])
        if domain in splits['val']:
            val_clean_paths.extend(splits['val'][domain])

    print(f"Found {len(train_clean_paths)} training images")
    print(f"Found {len(val_clean_paths)} validation images")

    # Generate noise variants for each type
    for noise_type in args.noise_types:
        print(f"\n{'='*80}")
        print(f"Processing: {noise_type.upper()} noise")
        print(f"{'='*80}")

        # Set noise parameters
        if noise_type == 'rayleigh':
            noise_params = {'scale_range': tuple(args.rayleigh_scale_range)}
            noise_dir_name = 'noisy_rayleigh'
        elif noise_type == 'poisson':
            noise_params = {'lambda_scale': args.poisson_lambda}
            noise_dir_name = 'noisy_poisson'

        # Process each domain
        train_pairs_all = []
        val_pairs_all = []

        for domain in ['cnv', 'dme', 'drusen', 'normal']:
            print(f"\n--- Domain: {domain.upper()} ---")

            # Training set
            if domain in splits['train']:
                domain_train_paths = splits['train'][domain]
                output_dir = Path(f'oct/{domain}/train/{noise_dir_name}')

                domain_train_pairs = create_noisy_variants(
                    domain_train_paths,
                    output_dir,
                    noise_type,
                    noise_params
                )
                train_pairs_all.extend(domain_train_pairs)

            # Validation set
            if domain in splits['val']:
                domain_val_paths = splits['val'][domain]
                output_dir = Path(f'oct/{domain}/val/{noise_dir_name}')

                domain_val_pairs = create_noisy_variants(
                    domain_val_paths,
                    output_dir,
                    noise_type,
                    noise_params
                )
                val_pairs_all.extend(domain_val_pairs)

        # Write pairs files
        print(f"\n--- Writing pairs files for {noise_type} ---")
        write_pairs_file(train_pairs_all, Path(f'train_pairs_{noise_type}.txt'))
        write_pairs_file(val_pairs_all, Path(f'val_pairs_{noise_type}.txt'))

    print(f"\n{'='*80}")
    print("Generation Complete!")
    print(f"{'='*80}")

    print("\nGenerated files:")
    for noise_type in args.noise_types:
        print(f"\n{noise_type.upper()} Noise:")
        print(f"  Training pairs: train_pairs_{noise_type}.txt")
        print(f"  Validation pairs: val_pairs_{noise_type}.txt")
        print(f"  Noisy images: oct/*/train/{noise_type}/, oct/*/val/{noise_type}/")

    print("\nYou can now train baselines on these noise types!")


if __name__ == '__main__':
    main()
