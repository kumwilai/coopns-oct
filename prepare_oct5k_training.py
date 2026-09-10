#!/usr/bin/env python3
"""
Prepare OCT5k dataset for CUAP-OCT training.

OCT5k provides layer segmentation masks for AMD, DME, and Normal OCT scans.
This script matches masks to original images and converts to our 5-layer format.

OCT5k Layer Mapping (6 classes → 5 classes):
    OCT5k class 0: Background (above ILM) → merge with class 0
    OCT5k class 1: ILM to OPL-Henles → RNFL_GCL (our class 0)
    OCT5k class 2: OPL-Henles to IS/OS → INL_OPL (our class 1)
    OCT5k class 3: IS/OS to IBRPE → ONL (our class 2)
    OCT5k class 4: IBRPE to OBRPE → IS_OS (our class 3)
    OCT5k class 5: Below OBRPE → RPE_Choroid (our class 4)

Usage:
    python prepare_oct5k_training.py --output_dir oct5k_processed
"""

import argparse
import json
import os
import re
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Our 5-layer format
LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']

# OCT5k to our format mapping
# OCT5k: 0=background, 1=RNFL+GCL+IPL, 2=INL+OPL, 3=ONL, 4=IS/OS, 5=RPE+Choroid
OCT5K_TO_OURS = {
    0: 0,  # Background → RNFL_GCL (will be above retina)
    1: 0,  # ILM to OPL-Henles → RNFL_GCL
    2: 1,  # OPL-Henles to IS/OS → INL_OPL
    3: 2,  # IS/OS to IBRPE → ONL
    4: 3,  # IBRPE to OBRPE → IS_OS
    5: 4,  # Below OBRPE → RPE_Choroid
}


def convert_mask(oct5k_mask):
    """Convert OCT5k 6-class mask to our 5-class format."""
    our_mask = np.zeros_like(oct5k_mask, dtype=np.uint8)
    for oct5k_class, our_class in OCT5K_TO_OURS.items():
        our_mask[oct5k_mask == oct5k_class] = our_class
    return our_mask


def normalize_path_component(s):
    """Normalize path component for matching."""
    # Remove extra spaces, convert to lowercase
    s = re.sub(r'\s+', ' ', s.strip().lower())
    # Remove parentheses differences
    s = s.replace('(', '').replace(')', '')
    return s


def find_matching_image(mask_path, images_base_dir):
    """
    Find the corresponding image for a mask.

    Mask path example:
    .../Masks_Manual/Grading_1/AMD Part1/AMD (1).E2E/2- 25- 2017 9- 10- 42 PM/Image 9.png

    Image path example:
    .../Dataset_3x50_Final/AMD/AMD (1)/Image (9).TIFF
    """
    mask_path = Path(mask_path)

    # Extract components from mask path
    parts = mask_path.parts

    # Find disease type and subject
    disease_part = None
    subject_part = None
    image_name = mask_path.stem  # e.g., "Image 9"

    for i, part in enumerate(parts):
        if 'AMD' in part and 'Part' in part:
            disease_part = 'AMD'
        elif 'DME' in part:
            disease_part = 'DME'
        elif 'Normal' in part and 'Part' in part:
            disease_part = 'Normal'

        # Find subject folder (e.g., "AMD (1).E2E")
        if '.E2E' in part:
            # Extract subject number
            match = re.search(r'(\w+)\s*\((\d+)\)', part)
            if match:
                subject_part = f"{match.group(1).upper()} ({match.group(2)})"

    if not disease_part or not subject_part:
        return None

    # Extract image number
    img_match = re.search(r'Image\s*(\d+)', image_name)
    if not img_match:
        return None
    img_num = img_match.group(1)

    # Construct expected image path
    # Try different naming conventions
    possible_paths = [
        images_base_dir / disease_part / subject_part / f"Image ({img_num}).TIFF",
        images_base_dir / disease_part / subject_part / f"Image {img_num}.TIFF",
        images_base_dir / disease_part / subject_part / f"Image({img_num}).TIFF",
    ]

    for img_path in possible_paths:
        if img_path.exists():
            return img_path

    # Try case-insensitive search
    disease_dir = images_base_dir / disease_part
    if disease_dir.exists():
        for subj_dir in disease_dir.iterdir():
            if normalize_path_component(subj_dir.name) == normalize_path_component(subject_part):
                for img_file in subj_dir.glob("*.TIFF"):
                    if f"({img_num})" in img_file.name or f" {img_num}." in img_file.name:
                        return img_file

    return None


def process_oct5k(masks_dir, images_dir, output_dir, grading='Grading_1'):
    """
    Process OCT5k dataset.

    Args:
        masks_dir: Path to OCT5k/Masks/Masks_Manual
        images_dir: Path to extracted images (Dataset_3x50_Final)
        output_dir: Output directory for processed data
        grading: Which grading to use (Grading_1, Grading_2, or Grading_3)
    """
    masks_dir = Path(masks_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)

    grading_dir = masks_dir / grading
    if not grading_dir.exists():
        print(f"ERROR: Grading directory not found: {grading_dir}")
        return []

    all_samples = []
    matched = 0
    unmatched = 0

    # Find all mask files
    mask_files = list(grading_dir.rglob("*.png"))
    print(f"Found {len(mask_files)} masks in {grading}")

    for mask_path in tqdm(mask_files, desc=f'Processing {grading}'):
        # Find matching image
        img_path = find_matching_image(mask_path, images_dir)

        if img_path is None:
            unmatched += 1
            continue

        matched += 1

        # Load and convert mask
        oct5k_mask = np.array(Image.open(mask_path))
        our_mask = convert_mask(oct5k_mask)

        # Load image
        image = np.array(Image.open(img_path))

        # Resize image to match mask if needed
        if image.shape != our_mask.shape:
            image = np.array(Image.fromarray(image).resize(
                (our_mask.shape[1], our_mask.shape[0]),
                Image.Resampling.BILINEAR
            ))

        # Determine disease type
        disease = 'unknown'
        for part in mask_path.parts:
            if 'AMD' in part:
                disease = 'AMD'
                break
            elif 'DME' in part:
                disease = 'DME'
                break
            elif 'Normal' in part:
                disease = 'Normal'
                break

        # Create unique filename
        safe_name = f"{disease}_{mask_path.parent.parent.name}_{mask_path.stem}".replace(' ', '_')
        safe_name = re.sub(r'[^\w\-_]', '', safe_name)

        # Save image
        img_filename = f'{safe_name}.png'
        img_out_path = output_dir / 'images' / img_filename
        Image.fromarray(image).save(img_out_path)

        # Save mask
        mask_filename = f'{safe_name}_mask.png'
        mask_out_path = output_dir / 'masks' / mask_filename
        Image.fromarray(our_mask).save(mask_out_path)

        # Add to samples list
        sample = {
            'image_path': str(img_out_path.absolute()),
            'mask_path': str(mask_out_path.absolute()),
            'disease': disease,
            'source': 'oct5k',
            'grading': grading,
        }
        all_samples.append(sample)

    print(f"Matched: {matched}, Unmatched: {unmatched}")
    return all_samples


def create_train_val_split(samples, val_ratio=0.2, seed=42):
    """Split samples by disease type to ensure balanced splits."""
    np.random.seed(seed)

    # Group by disease
    by_disease = {}
    for s in samples:
        disease = s.get('disease', 'unknown')
        if disease not in by_disease:
            by_disease[disease] = []
        by_disease[disease].append(s)

    train_samples = []
    val_samples = []

    for disease, disease_samples in by_disease.items():
        np.random.shuffle(disease_samples)
        n_val = max(1, int(len(disease_samples) * val_ratio))
        val_samples.extend(disease_samples[:n_val])
        train_samples.extend(disease_samples[n_val:])

    return train_samples, val_samples


def save_jsonl(samples, output_path):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Prepare OCT5k for training')
    parser.add_argument('--masks_dir',
                        default='oct5k_dataset/OCT5k/Masks/Masks_Manual',
                        help='Path to OCT5k masks')
    parser.add_argument('--images_dir',
                        default='oct5k_dataset/OCT5k/Scripts/Macular_Dataset_Heidelberg/Dataset_3x50_Final',
                        help='Path to extracted images')
    parser.add_argument('--output_dir', default='oct5k_processed',
                        help='Output directory')
    parser.add_argument('--grading', default='Grading_1',
                        choices=['Grading_1', 'Grading_2', 'Grading_3'],
                        help='Which grading to use')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio')
    args = parser.parse_args()

    print("=" * 60)
    print("OCT5k DATASET PREPARATION")
    print("=" * 60)
    print(f"Masks: {args.masks_dir}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Grading: {args.grading}")
    print()

    # Process dataset
    samples = process_oct5k(
        args.masks_dir,
        args.images_dir,
        args.output_dir,
        args.grading
    )

    print(f"\nTotal matched samples: {len(samples)}")

    if len(samples) == 0:
        print("ERROR: No samples matched! Check paths.")
        return

    # Split into train/val
    train_samples, val_samples = create_train_val_split(samples, args.val_ratio)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Save JSONL files
    train_path = os.path.join(args.output_dir, 'oct5k_train.jsonl')
    val_path = os.path.join(args.output_dir, 'oct5k_val.jsonl')

    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")

    # Print disease distribution
    print("\n" + "=" * 60)
    print("DISEASE DISTRIBUTION")
    print("=" * 60)
    for split_name, split_samples in [('Train', train_samples), ('Val', val_samples)]:
        print(f"\n{split_name}:")
        disease_counts = {}
        for s in split_samples:
            d = s.get('disease', 'unknown')
            disease_counts[d] = disease_counts.get(d, 0) + 1
        for d, c in sorted(disease_counts.items()):
            print(f"  {d}: {c}")

    # Summary
    print("\n" + "=" * 60)
    print("LAYER MAPPING")
    print("=" * 60)
    print("OCT5k 6 classes → Our 5 layers:")
    print("  OCT5k 0 (background) + 1 (ILM-OPL) → 0: RNFL_GCL")
    print("  OCT5k 2 (OPL-IS/OS)                → 1: INL_OPL")
    print("  OCT5k 3 (IS/OS-IBRPE)              → 2: ONL")
    print("  OCT5k 4 (IBRPE-OBRPE)              → 3: IS_OS")
    print("  OCT5k 5 (below OBRPE)              → 4: RPE_Choroid")
    print("=" * 60)


if __name__ == '__main__':
    main()
