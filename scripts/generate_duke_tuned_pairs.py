#!/usr/bin/env python3
"""
Generate synthetic noisy pairs using DUKE-LEARNED realistic noise composition.

Uses fixed weights learned from Duke OCT dataset analysis:
  - Speckle: 83.8%
  - Banding: 4.3%
  - Gaussian: 4.5%
  - Shot: 7.4%

This creates training data that matches real OCT noise distribution,
improving cross-dataset generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "nsnd_oct"))

from nsnd.training.synthetic_noise import add_oct_noise_mixture


# Duke-learned realistic noise composition (from 6,057 patches)
DUKE_LEARNED_WEIGHTS = {
    "speckle": 0.8381,
    "banding": 0.0428,
    "gaussian": 0.0449,
    "shot": 0.0742,
}


def make_noisy_path(clean_path: Path, noisy_name: str) -> Path:
    parts = list(clean_path.parts)
    try:
        idx = parts.index("clean")
    except ValueError:
        raise ValueError(f"Clean path missing 'clean' folder: {clean_path}")
    parts[idx] = noisy_name
    return Path(*parts)


def sample_params(param_scale: float) -> Dict[str, float]:
    """Sample noise parameters (same as original)."""
    scale = float(param_scale)
    return {
        "speckle_k": float(np.random.uniform(1.5, 4.0) / scale),
        "speckle_depth_gain": float(np.random.uniform(0.4, 1.0)),
        "banding_freq": int(np.random.choice([15, 20, 30, 40])),
        "banding_amp": float(np.random.uniform(0.06, 0.12) * scale),
        "gaussian_sigma": float(np.random.uniform(0.02, 0.06) * scale),
        "gaussian_depth_gain": float(np.random.uniform(0.2, 0.8)),
        "shot_peak": float(np.random.uniform(30.0, 120.0) / scale),
        "shot_depth_gain": float(np.random.uniform(0.4, 1.0)),
        "use_depth_profile": True,
    }


def select_paths(
    split_dict: Dict[str, List[str]],
    max_per_class: int | None,
    max_total: int | None,
) -> List[Path]:
    """Select paths with optional limits."""
    per_class: Dict[str, List[Path]] = {}
    for cls, items in split_dict.items():
        paths = [Path(p) for p in items]
        if max_per_class is not None:
            paths = paths[:max_per_class]
        per_class[cls] = paths

    if max_total is None:
        selected = []
        for cls in sorted(per_class.keys()):
            selected.extend(per_class[cls])
        return selected

    # Round-robin to keep balance
    classes = sorted(per_class.keys())
    idx = {c: 0 for c in classes}
    selected = []
    while len(selected) < max_total:
        progressed = False
        for c in classes:
            i = idx[c]
            if i < len(per_class[c]):
                selected.append(per_class[c][i])
                idx[c] += 1
                progressed = True
                if len(selected) >= max_total:
                    break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training pairs with Duke-learned realistic noise composition"
    )
    parser.add_argument("--splits_json", type=str, required=True,
                       help="Path to splits JSON file")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                       help="Which split to generate pairs for")
    parser.add_argument("--noisy_name", type=str, default="noisy_duke_tuned",
                       help="Name for noisy image folder")
    parser.add_argument("--joint_stats", type=str, default="",
                       help="Path to joint stats JSON (weights+params)")

    # Noise composition options
    parser.add_argument("--use_duke_weights", action="store_true", default=True,
                       help="Use Duke-learned weights (default: True)")
    parser.add_argument("--custom_weights", type=float, nargs=4, default=None,
                       help="Custom weights [speckle banding gaussian shot] (overrides --use_duke_weights)")

    # Noise parameters
    parser.add_argument("--param_scale", type=float, default=1.0,
                       help="Scale for noise parameters")
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed for reproducibility")

    # Data selection
    parser.add_argument("--max_images_per_class", type=int, default=None,
                       help="Max images per class")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Max total images")

    # Output
    parser.add_argument("--pairs_out", type=str, default="/home/kumwilai/OCT/train_pairs_duke_tuned.txt",
                       help="Output path for pairs file")
    parser.add_argument("--weights_out", type=str, default="/home/kumwilai/OCT/weights_duke_tuned.jsonl",
                       help="Output path for weights JSONL")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing noisy images")

    args = parser.parse_args()

    # Load splits
    splits = json.loads(Path(args.splits_json).read_text())
    split_dict = splits.get(args.split)
    if split_dict is None:
        raise SystemExit(f"Split '{args.split}' not found in {args.splits_json}")

    # Select clean paths
    clean_paths = select_paths(split_dict, args.max_images_per_class, args.max_images)
    if not clean_paths:
        raise SystemExit("No clean paths selected.")

    print("="*80)
    print("GENERATING DUKE-TUNED SYNTHETIC PAIRS")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Images: {len(clean_paths)}")
    print(f"Output: {args.pairs_out}")
    print()

    joint_stats = None
    if args.joint_stats:
        joint_path = Path(args.joint_stats)
        if not joint_path.exists():
            raise SystemExit(f"Joint stats not found: {joint_path}")
        joint_stats = json.loads(joint_path.read_text())
        print(f"Using JOINT stats from: {joint_path}")
        weights_dict = None
    else:
        # Determine weights to use
        if args.custom_weights:
            weights_array = np.array(args.custom_weights, dtype=np.float32)
            weights_array /= weights_array.sum()  # Normalize
            weights_dict = {
                "speckle": float(weights_array[0]),
                "banding": float(weights_array[1]),
                "gaussian": float(weights_array[2]),
                "shot": float(weights_array[3]),
            }
            print("Using CUSTOM weights:")
        else:
            weights_dict = DUKE_LEARNED_WEIGHTS.copy()
            print("Using DUKE-LEARNED weights:")

    if weights_dict is not None:
        print(f"  Speckle:  {weights_dict['speckle']:.4f} ({weights_dict['speckle']*100:.1f}%)")
        print(f"  Banding:  {weights_dict['banding']:.4f} ({weights_dict['banding']*100:.1f}%)")
        print(f"  Gaussian: {weights_dict['gaussian']:.4f} ({weights_dict['gaussian']*100:.1f}%)")
        print(f"  Shot:     {weights_dict['shot']:.4f} ({weights_dict['shot']*100:.1f}%)")
        print()
    print("="*80)
    print()

    # Setup output
    pairs_path = Path(args.pairs_out)
    weights_path = Path(args.weights_out) if args.weights_out else None
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path:
        weights_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_lines = []
    weights_f = weights_path.open("w", encoding="utf-8") if weights_path else None

    # Generate noisy pairs
    for idx, clean_path in enumerate(clean_paths):
        # Set seed for reproducibility
        np.random.seed(args.seed + idx)
        torch.manual_seed(args.seed + idx)

        # Load clean image
        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
        clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()

        # Sample noise parameters (fixed weights or joint stats)
        if joint_stats is not None:
            mean = np.array(joint_stats["mean"], dtype=np.float64)
            cov = np.array(joint_stats["cov"], dtype=np.float64)
            z = np.random.multivariate_normal(mean, cov)
            logit_w = z[:3]
            logits = np.array([logit_w[0], logit_w[1], logit_w[2], 0.0], dtype=np.float64)
            logits = logits - logits.max()
            exp_logits = np.exp(logits)
            w = exp_logits / exp_logits.sum()
            weights_dict = {
                "speckle": float(w[0]),
                "banding": float(w[1]),
                "gaussian": float(w[2]),
                "shot": float(w[3]),
            }
            log_params = z[3:7]
            raw_params = np.exp(log_params)
            clip = joint_stats.get("param_clip", {})
            def _clip_param(val, key):
                if key in clip:
                    low, high = clip[key]
                    return float(np.clip(val, low, high))
                return float(val)
            params = {
                "speckle_k": _clip_param(raw_params[0], "speckle_k"),
                "banding_amp": _clip_param(raw_params[1], "banding_amp"),
                "gaussian_sigma": _clip_param(raw_params[2], "gaussian_sigma"),
                "shot_peak": _clip_param(raw_params[3], "shot_peak"),
                "speckle_depth_gain": float(np.random.uniform(0.4, 1.0)),
                "banding_freq": int(np.random.choice([15, 20, 30, 40])),
                "gaussian_depth_gain": float(np.random.uniform(0.2, 0.8)),
                "shot_depth_gain": float(np.random.uniform(0.4, 1.0)),
                "use_depth_profile": True,
            }
            if args.param_scale != 1.0:
                scale = float(args.param_scale)
                params["speckle_k"] = float(params["speckle_k"] / scale)
                params["banding_amp"] = float(params["banding_amp"] * scale)
                params["gaussian_sigma"] = float(params["gaussian_sigma"] * scale)
                params["shot_peak"] = float(params["shot_peak"] / scale)
        else:
            params = sample_params(args.param_scale)

        # Add noise with Duke-learned composition
        noisy_tensor = add_oct_noise_mixture(clean_tensor, weights_dict, params)
        noisy = noisy_tensor[0, 0].cpu().numpy()

        # Save noisy image
        noisy_path = make_noisy_path(clean_path, args.noisy_name)
        noisy_path.parent.mkdir(parents=True, exist_ok=True)

        if noisy_path.exists() and not args.overwrite:
            pass
        else:
            Image.fromarray((noisy * 255.0).clip(0, 255).astype(np.uint8)).save(noisy_path)

        # Record pair
        pairs_lines.append(f"{clean_path}\t{noisy_path}\n")

        # Save weights/params metadata
        if weights_f:
            metadata = {
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path),
                "weights": weights_dict,
                "params": params,
            }
            weights_f.write(json.dumps(metadata) + "\n")

        # Progress
        if (idx + 1) % 100 == 0 or idx == len(clean_paths) - 1:
            print(f"  Processed: {idx + 1}/{len(clean_paths)}")

    # Write pairs file
    with pairs_path.open("w", encoding="utf-8") as pf:
        pf.write(f"# Duke-tuned synthetic pairs ({args.split} split)\n")
        if joint_stats is not None:
            wmean = joint_stats.get("weights_mean", [0.0, 0.0, 0.0, 0.0])
            pf.write(
                f"# Noise composition (joint mean): Speckle {wmean[0]:.4f}, "
                f"Banding {wmean[1]:.4f}, Gaussian {wmean[2]:.4f}, Shot {wmean[3]:.4f}\n"
            )
        else:
            pf.write(f"# Noise composition: Speckle {weights_dict['speckle']:.4f}, ")
            pf.write(f"Banding {weights_dict['banding']:.4f}, ")
            pf.write(f"Gaussian {weights_dict['gaussian']:.4f}, ")
            pf.write(f"Shot {weights_dict['shot']:.4f}\n")
        pf.write("# Format: clean_path<TAB>noisy_path\n")
        pf.write("#\n")
        for line in pairs_lines:
            pf.write(line)

    if weights_f:
        weights_f.close()

    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)
    print(f"✓ Generated {len(clean_paths)} noisy/clean pairs")
    print(f"✓ Pairs file: {pairs_path}")
    if weights_path:
        print(f"✓ Weights JSONL: {weights_path}")
    print()
    if joint_stats is not None:
        wmean = joint_stats.get("weights_mean", [0.0, 0.0, 0.0, 0.0])
        print("Noise composition (joint mean):")
        print(f"  - Speckle:  {wmean[0]:.4f}")
        print(f"  - Banding:  {wmean[1]:.4f}")
        print(f"  - Gaussian: {wmean[2]:.4f}")
        print(f"  - Shot:     {wmean[3]:.4f}")
        print("  - Jointly sampled with correlated parameters")
    else:
        print("Noise composition (Duke-learned):")
        print(f"  - Speckle-dominant (83.8%): Matches real OCT")
        print(f"  - Balanced minor components (4-7% each)")
        print(f"  - Fixed weights for all samples (consistent)")
    print()
    print("Expected improvements:")
    print(f"  - Duke synthetic: +3-4 dB (25→29 dB)")
    print(f"  - Duke human OCT: +2-4 dB (23→26 dB)")
    print("="*80)


if __name__ == "__main__":
    main()
