#!/usr/bin/env python3
"""
Prepare Joint Training Data for NSND-MultiTask

Combines two data sources:
- Option A: Duke DME + OCT5k with synthetic noise (has real masks)
- Option B: Duke17 + PKU37 with pseudo-masks (has real noise)

Output: Unified JSONL files for joint segmentation + denoising training
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image

# Try to import tifffile for TIFF support
try:
    from tifffile import imread as tiff_imread, imwrite as tiff_imwrite
except ImportError:
    tiff_imread = None
    tiff_imwrite = None


def load_image(path: str) -> np.ndarray:
    """Load image from various formats."""
    path = str(path)
    if path.endswith(('.tif', '.tiff')):
        if tiff_imread is None:
            raise ImportError("tifffile required for TIFF images")
        img = tiff_imread(path).astype(np.float32)
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = img[0]  # Take first slice of multi-page TIFF
    else:
        img = np.array(Image.open(path).convert('L')).astype(np.float32)
    return img


def save_image(img: np.ndarray, path: str):
    """Save image to file."""
    path = str(path)
    if path.endswith(('.tif', '.tiff')):
        if tiff_imwrite is None:
            raise ImportError("tifffile required for TIFF images")
        tiff_imwrite(path, img.astype(np.float32))
    else:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        Image.fromarray(img.astype(np.uint8)).save(path)


def add_synthetic_noise(clean: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add synthetic Gaussian noise to clean image."""
    # Normalize to [0, 1]
    if clean.max() > 1.0:
        clean = clean / 255.0

    # Add Gaussian noise
    noise = np.random.randn(*clean.shape).astype(np.float32) * noise_level
    noisy = clean + noise
    noisy = np.clip(noisy, 0, 1)

    return noisy


def generate_pseudo_masks(
    segmenter_ckpt: str,
    data_jsonl: str,
    output_dir: str,
    device: str = 'cuda',
) -> list:
    """
    Generate pseudo segmentation masks for denoising data using pretrained segmenter.

    Args:
        segmenter_ckpt: Path to pretrained segmenter checkpoint
        data_jsonl: JSONL file with noisy/clean paths
        output_dir: Directory to save pseudo masks
        device: Device to run inference on

    Returns:
        List of samples with pseudo mask paths added
    """
    print(f"\n=== Generating Pseudo Masks ===")
    print(f"Checkpoint: {segmenter_ckpt}")
    print(f"Data: {data_jsonl}")
    print(f"Output: {output_dir}")

    # Import segmenter
    sys.path.insert(0, os.path.dirname(__file__))
    from nsnd_multitask_model import LightweightLayerSegmenter

    # Load segmenter
    segmenter = LightweightLayerSegmenter(num_classes=5).to(device)

    if os.path.exists(segmenter_ckpt):
        ckpt = torch.load(segmenter_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # Filter segmenter keys if loading from full model
        seg_keys = {k.replace('segmenter.', ''): v for k, v in state_dict.items()
                    if k.startswith('segmenter.') or not any(k.startswith(p) for p in
                    ['symbolic_analyzer', 'denoiser_bank', 'fusion', 'refinement'])}

        if seg_keys:
            segmenter.load_state_dict(seg_keys, strict=False)
            print(f"Loaded segmenter weights from {segmenter_ckpt}")
        else:
            print("Warning: No segmenter weights found, using random initialization")
    else:
        print(f"Warning: Checkpoint not found: {segmenter_ckpt}")
        print("Using randomly initialized segmenter (will train from scratch)")

    segmenter.eval()

    # Create output directory
    mask_dir = Path(output_dir) / "pseudo_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    samples = []
    with open(data_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Processing {len(samples)} samples...")

    results = []
    for sample in tqdm(samples, desc="Generating masks"):
        # Load clean image (use clean for better segmentation)
        clean_path = sample.get('clean_path', sample.get('image_path'))
        if not clean_path or not os.path.exists(clean_path):
            print(f"Warning: Clean image not found for {sample}")
            continue

        clean = load_image(clean_path)

        # Normalize
        if clean.max() > 1.0:
            clean = clean / 255.0

        # Prepare tensor
        H, W = clean.shape
        clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float().to(device)

        # Generate mask
        with torch.no_grad():
            logits = segmenter(clean_t)
            mask = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Save mask
        subject = sample.get('subject', Path(clean_path).stem)
        mask_filename = f"{subject}_pseudo_mask.png"
        mask_path = str(mask_dir / mask_filename)

        # Save as PNG (values 0-4 for 5 classes)
        Image.fromarray(mask.astype(np.uint8)).save(mask_path)

        # Update sample with mask path
        result = sample.copy()
        result['mask_path'] = mask_path
        result['mask_type'] = 'pseudo'
        results.append(result)

    print(f"Generated {len(results)} pseudo masks")
    return results


def prepare_synthetic_noise_data(
    seg_data_jsonl: str,
    output_dir: str,
    noise_levels: list = [0.05, 0.10, 0.15, 0.20, 0.25],
) -> list:
    """
    Add synthetic noise to segmentation-only data.

    Args:
        seg_data_jsonl: JSONL with image_path and mask_path
        output_dir: Directory to save noisy images
        noise_levels: List of noise levels to generate

    Returns:
        List of samples with noisy/clean paths
    """
    print(f"\n=== Adding Synthetic Noise ===")
    print(f"Data: {seg_data_jsonl}")
    print(f"Noise levels: {noise_levels}")
    print(f"Output: {output_dir}")

    # Create output directory
    noisy_dir = Path(output_dir) / "synthetic_noisy"
    noisy_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation data
    samples = []
    with open(seg_data_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Processing {len(samples)} samples with {len(noise_levels)} noise levels...")

    results = []
    for sample in tqdm(samples, desc="Adding noise"):
        clean_path = sample.get('image_path')
        mask_path = sample.get('mask_path')

        if not clean_path or not os.path.exists(clean_path):
            continue

        # Load clean image
        clean = load_image(clean_path)
        if clean.max() > 1.0:
            clean = clean / 255.0

        # Generate noisy versions at different levels
        for noise_level in noise_levels:
            noisy = add_synthetic_noise(clean, noise_level)

            # Save noisy image
            basename = Path(clean_path).stem
            noisy_filename = f"{basename}_noise{int(noise_level*100):02d}.png"
            noisy_path = str(noisy_dir / noisy_filename)
            save_image(noisy, noisy_path)

            # Create sample entry
            result = {
                'noisy_path': noisy_path,
                'clean_path': clean_path,
                'mask_path': mask_path,
                'mask_type': 'real',
                'noise_level': noise_level,
                'noise_type': 'synthetic',
                'subject': sample.get('subject', basename),
            }
            results.append(result)

    print(f"Generated {len(results)} noisy samples")
    return results


def combine_datasets(
    synthetic_samples: list,
    real_noise_samples: list,
    output_jsonl: str,
    split_ratio: float = 0.9,
):
    """
    Combine synthetic noise and real noise datasets.

    Args:
        synthetic_samples: Samples with synthetic noise + real masks
        real_noise_samples: Samples with real noise + pseudo masks
        output_jsonl: Base path for output JSONL files
        split_ratio: Train/val split ratio
    """
    print(f"\n=== Combining Datasets ===")
    print(f"Synthetic noise samples: {len(synthetic_samples)}")
    print(f"Real noise samples: {len(real_noise_samples)}")

    # Combine all samples
    all_samples = synthetic_samples + real_noise_samples

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(all_samples)

    # Split
    n_train = int(len(all_samples) * split_ratio)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    # Save
    base = Path(output_jsonl)
    train_path = str(base.parent / f"{base.stem}_train.jsonl")
    val_path = str(base.parent / f"{base.stem}_val.jsonl")

    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')

    with open(val_path, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"\nOutput files:")
    print(f"  Train: {train_path} ({len(train_samples)} samples)")
    print(f"  Val:   {val_path} ({len(val_samples)} samples)")

    # Statistics
    print(f"\nDataset composition:")
    synthetic_train = sum(1 for s in train_samples if s.get('noise_type') == 'synthetic')
    real_train = len(train_samples) - synthetic_train
    print(f"  Train - Synthetic: {synthetic_train}, Real: {real_train}")

    synthetic_val = sum(1 for s in val_samples if s.get('noise_type') == 'synthetic')
    real_val = len(val_samples) - synthetic_val
    print(f"  Val   - Synthetic: {synthetic_val}, Real: {real_val}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare joint training data")

    # Input paths
    parser.add_argument('--seg_data', type=str, default='combined_train.jsonl',
                        help='Segmentation data JSONL (image_path + mask_path)')
    parser.add_argument('--denoise_data', type=str, nargs='+',
                        default=['duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl'],
                        help='Denoising data JSONL files (noisy_path + clean_path)')
    parser.add_argument('--segmenter_ckpt', type=str,
                        default='checkpoints/nsnd_test/best_model.pth',
                        help='Pretrained segmenter checkpoint for pseudo-mask generation')

    # Output
    parser.add_argument('--output_dir', type=str, default='data/joint_training',
                        help='Output directory')
    parser.add_argument('--output_name', type=str, default='joint_nsnd',
                        help='Output JSONL base name')

    # Options
    parser.add_argument('--noise_levels', type=str, default='0.05,0.10,0.15,0.20,0.25',
                        help='Comma-separated noise levels for synthetic noise')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='Train/val split ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for pseudo-mask generation')

    # Skip options
    parser.add_argument('--skip_synthetic', action='store_true',
                        help='Skip synthetic noise generation')
    parser.add_argument('--skip_pseudo_masks', action='store_true',
                        help='Skip pseudo-mask generation')

    args = parser.parse_args()

    # Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NSND Joint Training Data Preparation")
    print("=" * 60)

    # Option A: Synthetic noise + real masks
    synthetic_samples = []
    if not args.skip_synthetic and os.path.exists(args.seg_data):
        synthetic_samples = prepare_synthetic_noise_data(
            args.seg_data,
            str(output_dir),
            noise_levels,
        )
    else:
        print(f"\nSkipping synthetic noise generation")
        if not os.path.exists(args.seg_data):
            print(f"  (Segmentation data not found: {args.seg_data})")

    # Option B: Real noise + pseudo masks
    real_noise_samples = []
    if not args.skip_pseudo_masks:
        for denoise_jsonl in args.denoise_data:
            if os.path.exists(denoise_jsonl):
                samples = generate_pseudo_masks(
                    args.segmenter_ckpt,
                    denoise_jsonl,
                    str(output_dir),
                    args.device,
                )
                # Mark as real noise
                for s in samples:
                    s['noise_type'] = 'real'
                real_noise_samples.extend(samples)
            else:
                print(f"Warning: Denoising data not found: {denoise_jsonl}")
    else:
        print(f"\nSkipping pseudo-mask generation")

    # Combine datasets
    if synthetic_samples or real_noise_samples:
        output_jsonl = str(output_dir / args.output_name)
        train_path, val_path = combine_datasets(
            synthetic_samples,
            real_noise_samples,
            output_jsonl,
            args.split_ratio,
        )

        print("\n" + "=" * 60)
        print("Data preparation complete!")
        print("=" * 60)
        print(f"\nTo train NSND with this data:")
        print(f"  python train_nsnd_multitask.py \\")
        print(f"    --data_jsonl {train_path} \\")
        print(f"    --val_jsonl {val_path}")
    else:
        print("\nNo data generated. Check input paths.")


if __name__ == '__main__':
    main()
