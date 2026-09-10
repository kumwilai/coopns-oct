#!/usr/bin/env python3
"""
Prepare Duke DME dataset for CUAP-OCT training.

Converts Duke DME 8-layer boundaries to our 5-layer segmentation masks
and creates training JSONL files.

Layer Mapping:
    Duke boundaries 1-2 (ILM → NFL-GCL)     → RNFL_GCL (class 0)
    Duke boundaries 2-4 (NFL-GCL → INL-OPL) → INL_OPL (class 1)
    Duke boundaries 4-5 (INL-OPL → OPL-ONL) → ONL (class 2)
    Duke boundaries 5-7 (OPL-ONL → OS-RPE)  → IS_OS (class 3)
    Duke boundaries 7-8 (OS-RPE → BM)       → RPE_Choroid (class 4)

Usage:
    python prepare_duke_dme_training.py --output_dir duke_dme_processed
"""

import argparse
import json
import os
import numpy as np
import scipy.io as sio
from PIL import Image
from pathlib import Path
from tqdm import tqdm


LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']


def boundaries_to_segmentation_mask(boundaries, image_height):
    """
    Convert 8 layer boundaries to 5-class segmentation mask.

    Args:
        boundaries: (8, width) array of Y positions for each boundary
        image_height: Height of the image

    Returns:
        mask: (height, width) array with values 0-4 for each layer
    """
    width = boundaries.shape[1]
    mask = np.zeros((image_height, width), dtype=np.uint8)

    # Boundaries are Y coordinates (row indices) for each A-scan
    # boundary[0] = ILM, boundary[7] = BM

    for col in range(width):
        b = boundaries[:, col]

        # Skip if boundaries are invalid
        if np.any(b <= 0) or np.any(np.isnan(b)):
            continue

        # Convert to integers and clip
        b = np.clip(b.astype(int), 0, image_height - 1)

        # Ensure boundaries are in order
        b = np.sort(b)

        # Fill each layer region
        # Above ILM (boundary 0): background (keep as 0 = RNFL_GCL for simplicity)

        # RNFL_GCL: from boundary 0 (ILM) to boundary 1 (NFL-GCL)
        mask[b[0]:b[1], col] = 0  # RNFL_GCL

        # INL_OPL: from boundary 1 (NFL-GCL) to boundary 3 (INL-OPL)
        # Note: boundaries 2 and 3 are IPL-INL and INL-OPL
        mask[b[1]:b[3], col] = 1  # INL_OPL

        # ONL: from boundary 3 (INL-OPL) to boundary 4 (OPL-ONL)
        mask[b[3]:b[4], col] = 2  # ONL

        # IS_OS: from boundary 4 (OPL-ONL) to boundary 6 (OS-RPE)
        mask[b[4]:b[6], col] = 3  # IS_OS

        # RPE_Choroid: from boundary 6 (OS-RPE) to boundary 7 (BM) and below
        mask[b[6]:, col] = 4  # RPE_Choroid

    return mask


def process_duke_dme(input_dir, output_dir, use_manual=True, min_valid_ratio=0.5):
    """
    Process all Duke DME subjects.

    Args:
        input_dir: Path to 2015_BOE_Chiu folder
        output_dir: Output directory for processed data
        use_manual: Use manual annotations (True) or automatic (False)
        min_valid_ratio: Minimum ratio of valid (non-NaN) boundary annotations required
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    mat_files = sorted(Path(input_dir).glob('Subject_*.mat'))

    all_samples = []
    skipped_count = 0

    for mat_path in tqdm(mat_files, desc='Processing subjects'):
        subject_id = mat_path.stem  # e.g., 'Subject_01'

        mat = sio.loadmat(str(mat_path))
        images = mat['images']  # (496, 768, 61)

        # Choose annotation source
        if use_manual:
            # Average of two manual annotations
            layers1 = mat['manualLayers1']  # (8, 768, 61)
            layers2 = mat['manualLayers2']  # (8, 768, 61)
            layers = (layers1 + layers2) / 2
        else:
            layers = mat['automaticLayersDME']

        height, width, num_bscans = images.shape

        for bscan_idx in range(num_bscans):
            boundaries = layers[:, :, bscan_idx]

            # Check if this B-scan has enough valid annotations
            valid_ratio = np.sum(~np.isnan(boundaries)) / boundaries.size
            if valid_ratio < min_valid_ratio:
                skipped_count += 1
                continue

            image = images[:, :, bscan_idx]

            # Convert to segmentation mask
            mask = boundaries_to_segmentation_mask(boundaries, height)

            # Verify mask has multiple classes (not just zeros)
            unique_classes = np.unique(mask)
            if len(unique_classes) < 3:  # Need at least 3 classes for valid segmentation
                skipped_count += 1
                continue

            # Save image
            img_filename = f'{subject_id}_bscan{bscan_idx:02d}.png'
            img_path = os.path.join(output_dir, 'images', img_filename)
            Image.fromarray(image).save(img_path)

            # Save mask
            mask_filename = f'{subject_id}_bscan{bscan_idx:02d}_mask.png'
            mask_path = os.path.join(output_dir, 'masks', mask_filename)
            Image.fromarray(mask).save(mask_path)

            # Add to samples list
            sample = {
                'image_path': os.path.abspath(img_path),
                'mask_path': os.path.abspath(mask_path),
                'subject': subject_id,
                'bscan_idx': bscan_idx,
            }
            all_samples.append(sample)

    print(f"Skipped {skipped_count} B-scans with insufficient annotations")
    return all_samples


def create_train_val_split(samples, val_ratio=0.2, seed=42):
    """Split samples by subject (not by B-scan)."""
    np.random.seed(seed)

    # Group by subject
    subjects = {}
    for s in samples:
        subj = s['subject']
        if subj not in subjects:
            subjects[subj] = []
        subjects[subj].append(s)

    subject_ids = list(subjects.keys())
    np.random.shuffle(subject_ids)

    n_val = max(1, int(len(subject_ids) * val_ratio))
    val_subjects = set(subject_ids[:n_val])

    train_samples = []
    val_samples = []

    for subj, samples_list in subjects.items():
        if subj in val_subjects:
            val_samples.extend(samples_list)
        else:
            train_samples.extend(samples_list)

    return train_samples, val_samples


def save_jsonl(samples, output_path):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Prepare Duke DME for training')
    parser.add_argument('--input_dir', default='duke_dme_dataset/2015_BOE_Chiu',
                       help='Path to Duke DME mat files')
    parser.add_argument('--output_dir', default='duke_dme_processed',
                       help='Output directory')
    parser.add_argument('--use_manual', action='store_true', default=True,
                       help='Use manual annotations (default: True)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation split ratio')
    args = parser.parse_args()

    print("=" * 60)
    print("DUKE DME DATASET PREPARATION")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Using manual annotations: {args.use_manual}")
    print()

    # Process all subjects
    samples = process_duke_dme(args.input_dir, args.output_dir, args.use_manual)
    print(f"\nTotal samples: {len(samples)}")

    # Split into train/val
    train_samples, val_samples = create_train_val_split(samples, args.val_ratio)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Save JSONL files
    train_path = os.path.join(args.output_dir, 'duke_dme_train.jsonl')
    val_path = os.path.join(args.output_dir, 'duke_dme_val.jsonl')

    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")

    # Summary
    print()
    print("=" * 60)
    print("LAYER MAPPING")
    print("=" * 60)
    print("Duke 8 boundaries → Our 5 layers:")
    print("  0: RNFL_GCL (ILM to NFL-GCL)")
    print("  1: INL_OPL (NFL-GCL to INL-OPL)")
    print("  2: ONL (INL-OPL to OPL-ONL)")
    print("  3: IS_OS (OPL-ONL to OS-RPE)")
    print("  4: RPE_Choroid (OS-RPE to BM)")
    print("=" * 60)


if __name__ == '__main__':
    main()
