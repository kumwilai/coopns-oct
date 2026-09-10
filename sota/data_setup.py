
import os
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional
import random
import csv

CLASSES = ['cnv', 'dme', 'drusen', 'normal']

def _load_vendor_map(vendor_map_file: Optional[str]) -> Dict[str, str]:
    """
    Load a path->vendor map from JSON or CSV (columns: path,vendor).
    Paths can be absolute or relative; matching is substring based.
    """
    if not vendor_map_file:
        return {}
    path = Path(vendor_map_file)
    if not path.is_file():
        raise FileNotFoundError(f"Vendor map file not found: {vendor_map_file}")
    mapping: Dict[str, str] = {}
    if path.suffix.lower() == ".json":
        data = json.load(open(path, "r"))
        for k, v in data.items():
            mapping[str(k)] = str(v)
    else:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "path" not in reader.fieldnames or "vendor" not in reader.fieldnames:
                raise ValueError("CSV vendor map must have columns: path,vendor")
            for row in reader:
                mapping[str(row["path"])] = str(row["vendor"])
    return mapping


def _infer_vendor(img_path: str, vendor_map: Dict[str, str]) -> str:
    """
    Infer vendor from mapping; fallback to 'unknown'.
    Matching is performed by checking if a mapping key is a substring of the path.
    """
    for key, vendor in vendor_map.items():
        if key in img_path:
            return vendor
    return "unknown"


def create_stratified_splits(
    oct_root: str,
    output_file: str,
    test_split_ratio: float = 0.15,
    val_split_ratio: float = 0.15,
    vendor_map_file: Optional[str] = None,
):
    """
    Scans a directory of OCT images organized by class, creates stratified train/val/test splits,
    and saves them to a JSON file.

    Args:
        oct_root (str): Path to the root directory containing class subfolders (e.g., 'oct/').
        output_file (str): Path to save the JSON file with split information.
        test_split_ratio (float): The proportion of the dataset to include in the test split.
        val_split_ratio (float): The proportion of the dataset to include in the val split.
        vendor_map_file (str, optional): JSON or CSV mapping of image path substrings to vendor names.
    """
    root = Path(oct_root)
    if not root.is_dir():
        raise FileNotFoundError(f"OCT root directory not found at: {root}")

    vendor_map = _load_vendor_map(vendor_map_file)

    splits: Dict[str, Dict[str, List[str]]] = {'train': {}, 'val': {}, 'test': {}}
    all_files_by_class_vendor: Dict[str, Dict[str, List[str]]] = {cls: {} for cls in CLASSES}

    print("Gathering files...")
    for cls in CLASSES:
        class_dir = root / cls
        # Assuming the raw data is in a 'train/clean' structure as per the original layout
        # This might need adjustment if the raw data is in a different structure
        clean_dir = class_dir / 'train' / 'clean'
        if not clean_dir.is_dir():
            print(f"Warning: Directory not found for class '{cls}': {clean_dir}")
            continue

        files = sorted(glob.glob(str(clean_dir / '*.png')))
        # Bucket by vendor
        vendor_buckets: Dict[str, List[str]] = {}
        for f in files:
            vendor = _infer_vendor(f, vendor_map)
            vendor_buckets.setdefault(vendor, []).append(f)
        all_files_by_class_vendor[cls] = vendor_buckets
        total = sum(len(v) for v in vendor_buckets.values())
        print(f"Found {total} images for class '{cls}' across vendors: { {k: len(v) for k,v in vendor_buckets.items()} }")

    print("\nCreating splits...")
    for cls, vendor_dict in all_files_by_class_vendor.items():
        if not vendor_dict:
            continue

        splits['train'][cls], splits['val'][cls], splits['test'][cls] = [], [], []
        for vendor, files in vendor_dict.items():
            files = files.copy()
            random.shuffle(files)

            n_total = len(files)
            n_test = int(n_total * test_split_ratio)
            n_val = int(n_total * val_split_ratio)
            n_train = n_total - n_test - n_val

            splits['test'][cls].extend(files[:n_test])
            splits['val'][cls].extend(files[n_test : n_test + n_val])
            splits['train'][cls].extend(files[n_test + n_val :])

            print(f"Class '{cls}' vendor '{vendor}': Train={n_train}, Val={n_val}, Test={n_test}")

    print(f"\nSaving splits to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=4)
    print("Done.")

def load_splits(split_file: str) -> Dict[str, Dict[str, List[str]]]:
    """Loads pre-computed splits from a JSON file."""
    with open(split_file, 'r') as f:
        splits = json.load(f)
    return splits

if __name__ == '__main__':
    # Example of how to run this script
    # This assumes the script is run from the root of the project
    create_stratified_splits(
        oct_root='oct',
        output_file='oct_splits.json'
    )
