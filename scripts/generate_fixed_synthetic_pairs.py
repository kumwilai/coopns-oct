#!/usr/bin/env python3
"""
Generate fixed synthetic noisy pairs for a split using the OCT noise mixture.

Creates noisy images under a new folder name and writes:
  - pair list file (noisy,clean)
  - optional JSONL with weights/params for Top-1 evaluation
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


def make_noisy_path(clean_path: Path, noisy_name: str) -> Path:
    parts = list(clean_path.parts)
    try:
        idx = parts.index("clean")
    except ValueError:
        raise ValueError(f"Clean path missing 'clean' folder: {clean_path}")
    parts[idx] = noisy_name
    return Path(*parts)


def sample_params(param_scale: float) -> Dict[str, float]:
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

    # Round-robin to keep balance.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_json", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--noisy_name", type=str, default="noisy_realistic_fixed")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--param_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--balanced_top1", action="store_true",
                        help="Force balanced dominant noise classes for Top-1 evaluation.")
    parser.add_argument("--dominant_min", type=float, default=0.7,
                        help="Minimum dominant weight when --balanced_top1 is used.")
    parser.add_argument("--dominant_max", type=float, default=0.9,
                        help="Maximum dominant weight when --balanced_top1 is used.")
    parser.add_argument("--remainder_alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-dominant weights when balanced.")
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--pairs_out", type=str, default="pairs_realistic_fixed.txt")
    parser.add_argument("--weights_out", type=str, default="weights_realistic_fixed.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = json.loads(Path(args.splits_json).read_text())
    split_dict = splits.get(args.split)
    if split_dict is None:
        raise SystemExit(f"Split '{args.split}' not found in {args.splits_json}")

    clean_paths = select_paths(split_dict, args.max_images_per_class, args.max_images)
    if not clean_paths:
        raise SystemExit("No clean paths selected.")

    pairs_path = Path(args.pairs_out)
    weights_path = Path(args.weights_out) if args.weights_out else None
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path:
        weights_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = float(args.alpha)
    alpha_vec = np.full(4, alpha, dtype=np.float32)

    pairs_lines = []
    weights_f = weights_path.open("w", encoding="utf-8") if weights_path else None

    for idx, clean_path in enumerate(clean_paths):
        np.random.seed(args.seed + idx)
        torch.manual_seed(args.seed + idx)

        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
        clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()

        if args.balanced_top1:
            dominant_idx = idx % 4
            dom_min = float(args.dominant_min)
            dom_max = max(dom_min, float(args.dominant_max))
            dom_weight = np.random.uniform(dom_min, dom_max)
            remainder = max(1.0 - dom_weight, 1e-6)
            other_alpha = np.full(3, float(args.remainder_alpha), dtype=np.float32)
            other_weights = np.random.dirichlet(other_alpha) * remainder
            weights = [0.0, 0.0, 0.0, 0.0]
            weights[dominant_idx] = float(dom_weight)
            o = iter(other_weights.tolist())
            for j in range(4):
                if j == dominant_idx:
                    continue
                weights[j] = float(next(o))
        else:
            weights = np.random.dirichlet(alpha_vec).tolist()

        weights_dict = {
            "speckle": float(weights[0]),
            "banding": float(weights[1]),
            "gaussian": float(weights[2]),
            "shot": float(weights[3]),
        }
        params = sample_params(args.param_scale)

        noisy_tensor = add_oct_noise_mixture(clean_tensor, weights_dict, params)
        noisy = noisy_tensor[0, 0].cpu().numpy()
        noisy_path = make_noisy_path(clean_path, args.noisy_name)
        noisy_path.parent.mkdir(parents=True, exist_ok=True)
        if noisy_path.exists() and not args.overwrite:
            pass
        else:
            Image.fromarray((noisy * 255.0).clip(0, 255).astype(np.uint8)).save(noisy_path)

        pairs_lines.append(f"{noisy_path.as_posix()},{clean_path.as_posix()}")
        if weights_f:
            rec = {
                "noisy": noisy_path.as_posix(),
                "clean": clean_path.as_posix(),
                "weights": weights_dict,
                "params": params,
            }
            weights_f.write(json.dumps(rec) + "\n")

    pairs_path.write_text("\n".join(pairs_lines) + "\n")
    if weights_f:
        weights_f.close()
    print(f"✓ Wrote pairs: {pairs_path} ({len(pairs_lines)})")
    if weights_path:
        print(f"✓ Wrote weights: {weights_path}")


if __name__ == "__main__":
    main()
