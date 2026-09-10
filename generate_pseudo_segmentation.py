#!/usr/bin/env python3
"""
Generate Pseudo-Segmentation Labels for OCT Images

Uses gradient-based and intensity-based methods to create approximate
layer boundaries. This serves as:
1. Training data for initial segmentation model
2. Fallback if real segmentation labels are unavailable

For TMI paper: Replace with Duke DME real labels when available.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def generate_layer_mask(image, num_layers=5):
    """
    Generate pseudo layer segmentation mask.

    Returns:
        mask: [H, W] with values 0-4 for 5 layers
        boundaries: List of boundary positions
    """
    H, W = image.shape

    # Convert to tensor for processing
    img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

    # Compute vertical gradient (layer boundaries have high vertical gradient)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3) / 4.0
    grad_y = F.conv2d(F.pad(img_tensor, [1,1,1,1], mode='reflect'), sobel_y)
    grad_y = grad_y[0, 0].numpy()

    # Column-wise average gradient profile
    profile = np.mean(np.abs(grad_y), axis=1)

    # Smooth profile
    profile = gaussian_filter1d(profile, sigma=2.0)

    # Find peaks (layer boundaries)
    # We want 4 boundaries to divide into 5 layers
    peaks, properties = find_peaks(
        profile,
        height=np.percentile(profile, 60),
        distance=H // 8,  # Min distance between boundaries
        prominence=0.05
    )

    # Sort by prominence and keep top 4
    if len(peaks) > 4:
        prominences = properties['prominences']
        top_indices = np.argsort(prominences)[-4:]
        peaks = np.sort(peaks[top_indices])
    elif len(peaks) < 4:
        # If not enough peaks, use fixed percentages
        peaks = np.array([int(H * p) for p in [0.15, 0.35, 0.55, 0.75]])

    # Create mask with 5 layers
    mask = np.zeros((H, W), dtype=np.uint8)

    # Layer 0: Top to first boundary (RNFL_GCL)
    mask[0:peaks[0], :] = 0

    # Layer 1: First to second boundary (INL_OPL)
    mask[peaks[0]:peaks[1], :] = 1

    # Layer 2: Second to third boundary (ONL)
    mask[peaks[1]:peaks[2], :] = 2

    # Layer 3: Third to fourth boundary (IS_OS)
    mask[peaks[2]:peaks[3], :] = 3

    # Layer 4: Fourth to bottom (RPE_Choroid)
    mask[peaks[3]:, :] = 4

    return mask, peaks.tolist()


def process_dataset(input_jsonl, output_dir, split_name):
    """Process entire dataset and generate pseudo-labels."""

    output_dir = Path(output_dir)
    seg_dir = output_dir / "segmentation" / split_name
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Read input data
    samples = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Process each sample
    seg_samples = []
    for sample in tqdm(samples, desc=f"Generating pseudo-labels ({split_name})"):
        # Load clean image (use clean for better boundary detection)
        clean_path = sample['clean_path']
        clean_img = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        # Generate pseudo-segmentation
        seg_mask, boundaries = generate_layer_mask(clean_img)

        # Save segmentation mask
        clean_name = Path(clean_path).stem
        seg_path = seg_dir / f"{clean_name}_seg.npy"
        np.save(seg_path, seg_mask)

        # Create new sample entry
        seg_sample = {
            'clean_path': sample['clean_path'],
            'noisy_path': sample['noisy_path'],
            'seg_path': str(seg_path),
            'boundaries': boundaries,
            'num_layers': 5,
            'weights': sample.get('weights', {}),
        }
        seg_samples.append(seg_sample)

    # Save new JSONL with segmentation
    output_jsonl = output_dir / f"seg_{split_name}.jsonl"
    with open(output_jsonl, 'w') as f:
        for sample in seg_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Generated {len(seg_samples)} pseudo-labels")
    print(f"Saved to: {output_jsonl}")

    return output_jsonl


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-segmentation labels')
    parser.add_argument('--train_jsonl', default='weights_duke_analysis_maps_train.jsonl',
                       help='Training data JSONL')
    parser.add_argument('--val_jsonl', default='weights_duke_analysis_maps_val.jsonl',
                       help='Validation data JSONL')
    parser.add_argument('--output_dir', default='seg_data',
                       help='Output directory for segmentation data')

    args = parser.parse_args()

    print("="*70)
    print("GENERATING PSEUDO-SEGMENTATION LABELS")
    print("="*70)
    print("\nNOTE: These are approximate labels based on gradients.")
    print("For best results, replace with Duke DME real labels.")
    print("="*70)

    # Process training set
    print("\n[1/2] Processing training set...")
    train_jsonl = process_dataset(args.train_jsonl, args.output_dir, 'train')

    # Process validation set
    print("\n[2/2] Processing validation set...")
    val_jsonl = process_dataset(args.val_jsonl, args.output_dir, 'val')

    print("\n" + "="*70)
    print("PSEUDO-SEGMENTATION GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput:")
    print(f"  Training:   {train_jsonl}")
    print(f"  Validation: {val_jsonl}")
    print("\nNext step: python train_layer_segmentation.py")
    print("="*70)


if __name__ == '__main__':
    main()
