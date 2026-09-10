#!/usr/bin/env python3
"""
Fit a joint distribution of (weights, params) from Duke noisy/clean pairs.

We estimate per-image:
  - weights: from the hybrid analyzer on noisy images
  - params: from residuals (noisy - clean)

Then fit a joint Gaussian in:
  [log(w1/w4), log(w2/w4), log(w3/w4),
   log(speckle_k), log(banding_amp), log(gaussian_sigma), log(shot_peak)]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "nsnd_oct"))

from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer


def load_pairs(pairs_file: Path) -> list[tuple[Path, Path]]:
    pairs = []
    with pairs_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            noisy_path, clean_path = parts
            pairs.append((Path(noisy_path), Path(clean_path)))
    return pairs


def analyze_image_weights(
    model: HybridCNNSymbolicAnalyzer,
    image_path: Path,
    patch_size: int,
    stride: int,
    device: str,
) -> dict[str, float]:
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    h, w = img.shape
    weights = {k: [] for k in ("speckle", "banding", "gaussian", "shot")}

    model.eval()
    with torch.no_grad():
        for top in range(0, max(1, h - patch_size + 1), stride):
            for left in range(0, max(1, w - patch_size + 1), stride):
                patch_top = min(top, h - patch_size)
                patch_left = min(left, w - patch_size)
                patch = img[patch_top:patch_top + patch_size, patch_left:patch_left + patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                weights_dict, _ = model(patch_tensor)
                for key in weights.keys():
                    weights[key].append(float(weights_dict[key].item()))

    return {k: float(np.mean(weights[k])) for k in weights.keys()}


def estimate_params_from_residual(noisy: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    eps = 1e-6
    residual = noisy - clean

    gaussian_sigma = float(residual.std())
    row_means = residual.mean(axis=1)
    banding_amp = float(0.5 * (row_means.max() - row_means.min()))

    speckle_ratio = residual / (clean + eps)
    speckle_var = float(speckle_ratio.var())
    speckle_k = float(1.0 / (speckle_var + eps))

    mean_intensity = float(clean.mean())
    var_resid = float(residual.var())
    shot_peak = float(mean_intensity / (var_resid + eps))

    return {
        "speckle_k": max(speckle_k, eps),
        "banding_amp": max(banding_amp, eps),
        "gaussian_sigma": max(gaussian_sigma, eps),
        "shot_peak": max(shot_peak, eps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic_pairs",
        type=str,
        default="/home/kumwilai/OCT/duke_datasets/organized_test_pairs/test_pairs_synthetic.txt",
    )
    parser.add_argument(
        "--human_pairs",
        type=str,
        default="/home/kumwilai/OCT/duke_datasets/organized_test_pairs/test_pairs_human.txt",
    )
    parser.add_argument("--analyzer_ckpt", type=str, default="checkpoints/hybrid_cnn_symbolic.pth")
    parser.add_argument("--output_json", type=str, default="results/duke_joint_noise_stats.json")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--target_weights", type=float, nargs=4, default=None,
                        help="Optional target weights [speckle banding gaussian shot]")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    synthetic_pairs = Path(args.synthetic_pairs)
    human_pairs = Path(args.human_pairs)

    pairs = []
    if synthetic_pairs.exists():
        pairs.extend(load_pairs(synthetic_pairs))
    if human_pairs.exists():
        pairs.extend(load_pairs(human_pairs))

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    if not pairs:
        raise SystemExit("No Duke pairs found to fit joint stats.")

    print("=" * 80)
    print("FITTING DUKE JOINT NOISE STATS")
    print("=" * 80)
    print(f"Pairs: {len(pairs)}")
    print(f"Analyzer ckpt: {args.analyzer_ckpt}")
    print(f"Output: {args.output_json}")
    print()

    device = args.device
    model = HybridCNNSymbolicAnalyzer()
    ckpt = torch.load(args.analyzer_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    weights_list = []
    params_list = []

    for noisy_path, clean_path in tqdm(pairs, desc="Estimating stats"):
        if not noisy_path.exists() or not clean_path.exists():
            continue

        weights = analyze_image_weights(
            model, noisy_path, patch_size=args.patch_size, stride=args.stride, device=device
        )
        noisy = np.array(Image.open(noisy_path).convert("L"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
        params = estimate_params_from_residual(noisy, clean)

        weights_list.append([weights["speckle"], weights["banding"], weights["gaussian"], weights["shot"]])
        params_list.append([params["speckle_k"], params["banding_amp"], params["gaussian_sigma"], params["shot_peak"]])

    if not weights_list:
        raise SystemExit("No valid pairs processed.")

    weights_arr = np.array(weights_list, dtype=np.float64)
    params_arr = np.array(params_list, dtype=np.float64)

    # Normalize and logit-transform weights
    eps = 1e-6
    weights_arr = np.clip(weights_arr, eps, 1.0)
    weights_arr = weights_arr / weights_arr.sum(axis=1, keepdims=True)
    logit_weights = np.log(weights_arr[:, :3] / weights_arr[:, 3:4])

    # Log-transform params
    log_params = np.log(np.clip(params_arr, eps, None))

    # Joint statistics
    X = np.concatenate([logit_weights, log_params], axis=1)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)

    weights_mean_raw = weights_arr.mean(axis=0)
    if args.target_weights:
        tw = np.array(args.target_weights, dtype=np.float64)
        tw = np.clip(tw, eps, None)
        tw = tw / tw.sum()
        target_logit = np.log(tw[:3] / tw[3])
        mean[:3] = target_logit
        weights_mean = tw
    else:
        weights_mean = weights_mean_raw

    # Clip ranges for params
    param_low = np.percentile(params_arr, 1, axis=0)
    param_high = np.percentile(params_arr, 99, axis=0)

    out = {
        "version": "duke_joint_v1",
        "mean": mean.tolist(),
        "cov": cov.tolist(),
        "weights_mean": weights_mean.tolist(),
        "weights_mean_raw": weights_mean_raw.tolist(),
        "weights_std": weights_arr.std(axis=0).tolist(),
        "params_mean": params_arr.mean(axis=0).tolist(),
        "params_std": params_arr.std(axis=0).tolist(),
        "param_clip": {
            "speckle_k": [float(param_low[0]), float(param_high[0])],
            "banding_amp": [float(param_low[1]), float(param_high[1])],
            "gaussian_sigma": [float(param_low[2]), float(param_high[2])],
            "shot_peak": [float(param_low[3]), float(param_high[3])],
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("\n✓ Wrote joint stats to:", out_path)


if __name__ == "__main__":
    main()
