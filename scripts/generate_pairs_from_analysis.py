#!/usr/bin/env python3
"""
Generate synthetic noisy pairs using analysis-derived noise statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, shift, map_coordinates

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "nsnd_oct"))

from nsnd.training.synthetic_noise import add_oct_noise_mixture


def _looks_clean(path: str) -> bool:
    p = path.lower()
    return any(token in p for token in ("clean", "gt", "target", "label"))


def _looks_noisy(path: str) -> bool:
    p = path.lower()
    return any(token in p for token in ("noisy", "noise", "corrupt", "input"))


def _resolve_path(raw: str, base_dir: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def make_noisy_path(clean_path: Path, noisy_name: str) -> Path:
    parts = list(clean_path.parts)
    try:
        idx = parts.index("clean")
    except ValueError:
        raise ValueError(f"Clean path missing 'clean' folder: {clean_path}")
    parts[idx] = noisy_name
    return Path(*parts)


def make_clean_path(clean_path: Path, clean_name: str) -> Path:
    parts = list(clean_path.parts)
    try:
        idx = parts.index("clean")
    except ValueError:
        raise ValueError(f"Clean path missing 'clean' folder: {clean_path}")
    parts[idx] = clean_name
    return Path(*parts)


def elastic_deform(image: np.ndarray, alpha: float, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    if alpha <= 0:
        return image
    shape = image.shape
    dx = gaussian_filter((rng.rand(*shape) * 2.0 - 1.0), sigma) * alpha
    dy = gaussian_filter((rng.rand(*shape) * 2.0 - 1.0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy, x + dx)
    warped = map_coordinates(image, indices, order=1, mode="reflect")
    return warped.reshape(shape)


def apply_transforms(
    image: np.ndarray,
    blur_sigma: float,
    shift_max: float,
    elastic_alpha: float,
    elastic_sigma: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    out = image
    if blur_sigma > 0:
        out = gaussian_filter(out, sigma=float(blur_sigma))
    if shift_max > 0:
        dy = rng.uniform(-shift_max, shift_max)
        dx = rng.uniform(-shift_max, shift_max)
        out = shift(out, shift=(dy, dx), order=1, mode="reflect")
    if elastic_alpha > 0:
        out = elastic_deform(out, float(elastic_alpha), float(elastic_sigma), rng)
    return out


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


def _parse_pairs_file(pairs_path: Path, pairs_order: str) -> List[Path]:
    clean_paths: List[Path] = []
    base_dir = pairs_path.parent
    for line in pairs_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "\t" in s:
            parts = s.split("\t")
        elif "," in s:
            parts = s.split(",")
        else:
            parts = s.split()
        if len(parts) < 1:
            continue
        if len(parts) == 1:
            clean_raw = parts[0].strip()
        else:
            first, second = parts[0].strip(), parts[1].strip()
            if pairs_order == "clean_first":
                clean_raw = first
            elif pairs_order == "noisy_first":
                clean_raw = second
            else:
                if _looks_clean(first) or _looks_noisy(second):
                    clean_raw = first
                elif _looks_clean(second) or _looks_noisy(first):
                    clean_raw = second
                else:
                    clean_raw = first
        clean_paths.append(_resolve_path(clean_raw, base_dir))
    return clean_paths


def _load_analysis(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    if "composition_weights" not in data:
        raise ValueError("analysis_json missing composition_weights")
    if "parameters" not in data:
        raise ValueError("analysis_json missing parameters")
    return data


def _jitter_value(base: float, jitter: float, min_val: float | None = None) -> float:
    if jitter <= 0:
        val = base
    else:
        val = base * (1.0 + np.random.uniform(-jitter, jitter))
    if min_val is not None:
        val = max(min_val, val)
    return float(val)


def sample_params_from_analysis(
    analysis: Dict[str, object],
    jitter: float,
    speckle_snr_factor: float,
    param_scale: float,
) -> Dict[str, float]:
    params = analysis["parameters"]

    speckle_k = float(params.get("speckle_k", 3.0))
    speckle_corr = params.get("speckle_correlation_sigma", params.get("speckle_correlation", 1.2))
    banding_freq = float(params.get("banding_freq", 1.0 / 30.0))
    banding_amp = float(params.get("banding_amp", 0.08))
    gaussian_sigma = float(params.get("gaussian_sigma", 0.04))
    shot_gain = float(params.get("shot_gain", 80.0))

    scale = max(float(param_scale), 1e-3)

    # banding_freq in analysis is cycles/pixel → convert to period (pixels per cycle)
    period = 1.0 / max(banding_freq, 1e-6)
    period = _jitter_value(period, jitter, min_val=2.0)
    period = int(min(512, max(2, round(period))))

    return {
        "speckle_k": _jitter_value(speckle_k / scale, jitter, min_val=0.1),
        "speckle_base_k": _jitter_value(speckle_k / scale, jitter, min_val=0.1),
        "speckle_snr_factor": float(speckle_snr_factor),
        "speckle_correlation": _jitter_value(float(speckle_corr), jitter, min_val=0.0),
        "use_signal_dependent_speckle": True,
        "speckle_depth_gain": float(np.random.uniform(0.4, 1.0)),
        "banding_freq": period,
        "banding_amp": _jitter_value(banding_amp * scale, jitter, min_val=0.0),
        "gaussian_sigma": _jitter_value(gaussian_sigma * scale, jitter, min_val=0.0),
        "gaussian_depth_gain": float(np.random.uniform(0.2, 0.8)),
        "shot_peak": _jitter_value(shot_gain / scale, jitter, min_val=1.0),
        "shot_depth_gain": float(np.random.uniform(0.4, 1.0)),
        "use_depth_profile": True,
    }


def sample_weights(analysis: Dict[str, object], mode: str, alpha_scale: float) -> Dict[str, float]:
    mean = np.array(analysis["composition_weights"], dtype=np.float64)
    mean = mean / mean.sum()
    if mode == "fixed":
        weights = mean
    else:
        alpha = np.array(analysis.get("dirichlet_alpha", mean * 5.0), dtype=np.float64)
        alpha = np.maximum(alpha * float(alpha_scale), 1e-3)
        weights = np.random.dirichlet(alpha)
    return {
        "speckle": float(weights[0]),
        "banding": float(weights[1]),
        "gaussian": float(weights[2]),
        "shot": float(weights[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy pairs using analysis-derived noise stats."
    )
    parser.add_argument("--analysis_json", type=str, required=True)
    parser.add_argument("--splits_json", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--pairs_in", type=str, default="")
    parser.add_argument("--pairs_order", type=str, default="auto",
                        choices=["auto", "clean_first", "noisy_first"])
    parser.add_argument("--noisy_name", type=str, default="noisy_analysis")
    parser.add_argument("--save_noise_maps", action="store_true",
                        help="Save per-pixel noise maps alongside noisy images.")
    parser.add_argument("--noise_maps_name", type=str, default="",
                        help="Folder name for noise maps (defaults to noisy_name + '_maps').")
    parser.add_argument("--pairs_out", type=str, required=True)
    parser.add_argument("--weights_out", type=str, required=True)
    parser.add_argument("--param_jitter", type=float, default=0.1)
    parser.add_argument("--param_scale", type=float, default=1.0)
    parser.add_argument("--post_blur_sigma", type=float, default=0.0)
    parser.add_argument("--shift_max", type=float, default=0.0)
    parser.add_argument("--elastic_alpha", type=float, default=0.0)
    parser.add_argument("--elastic_sigma", type=float, default=8.0)
    parser.add_argument("--clean_blur_sigma", type=float, default=0.0)
    parser.add_argument("--clean_shift_max", type=float, default=0.0)
    parser.add_argument("--clean_elastic_alpha", type=float, default=0.0)
    parser.add_argument("--clean_elastic_sigma", type=float, default=8.0)
    parser.add_argument("--avg_clean_frames", type=int, default=1)
    parser.add_argument("--clean_name", type=str, default="")
    parser.add_argument("--weights_mode", type=str, default="dirichlet",
                        choices=["dirichlet", "fixed"])
    parser.add_argument("--alpha_scale", type=float, default=1.0)
    parser.add_argument("--speckle_snr_factor", type=float, default=0.5)
    parser.add_argument("--max_images_per_class", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    save_maps = bool(args.save_noise_maps or args.noise_maps_name)
    noise_maps_name = args.noise_maps_name.strip() or f"{args.noisy_name}_maps"
    if save_maps and args.avg_clean_frames > 1:
        print("⚠ save_noise_maps requested; forcing avg_clean_frames=1 for alignment.")
        args.avg_clean_frames = 1

    if bool(args.splits_json) == bool(args.pairs_in):
        raise SystemExit("Provide exactly one of --splits_json or --pairs_in.")

    analysis_path = Path(args.analysis_json)
    if not analysis_path.exists():
        raise SystemExit(f"analysis_json not found: {analysis_path}")
    analysis = _load_analysis(analysis_path)

    if args.splits_json:
        splits = json.loads(Path(args.splits_json).read_text())
        split_dict = splits.get(args.split)
        if split_dict is None:
            raise SystemExit(f"Split '{args.split}' not found in {args.splits_json}")
        clean_paths = select_paths(split_dict, args.max_images_per_class, args.max_images)
    else:
        clean_paths = _parse_pairs_file(Path(args.pairs_in), args.pairs_order)
        if args.max_images is not None:
            clean_paths = clean_paths[: args.max_images]

    if not clean_paths:
        raise SystemExit("No clean paths selected.")

    if not args.overwrite:
        conflicts: List[str] = []
        for clean_path in clean_paths:
            noisy_path = make_noisy_path(clean_path, args.noisy_name)
            if noisy_path.exists():
                conflicts.append(str(noisy_path))
            if save_maps:
                maps_path = make_noisy_path(clean_path, noise_maps_name).with_suffix(".npz")
                if maps_path.exists():
                    conflicts.append(str(maps_path))
            needs_clean_save = (
                args.avg_clean_frames > 1
                or args.clean_blur_sigma > 0
                or args.clean_shift_max > 0
                or args.clean_elastic_alpha > 0
                or bool(args.clean_name)
            )
            if needs_clean_save:
                clean_name = args.clean_name or ("clean_avg" if args.avg_clean_frames > 1 else "clean_analysis")
                clean_out_path = make_clean_path(clean_path, clean_name)
                if clean_out_path.exists():
                    conflicts.append(str(clean_out_path))
        if conflicts:
            sample = "\n".join(conflicts[:5])
            raise SystemExit(
                "Found existing outputs (use --overwrite to replace). Sample:\n" + sample
            )

    pairs_path = Path(args.pairs_out)
    weights_path = Path(args.weights_out)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_lines: List[str] = []
    with weights_path.open("w", encoding="utf-8") as wf:
        for idx, clean_path in enumerate(clean_paths):
            np.random.seed(args.seed + idx)
            torch.manual_seed(args.seed + idx)

            clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
            clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()

            weights = sample_weights(analysis, args.weights_mode, args.alpha_scale)
            params = sample_params_from_analysis(
                analysis,
                args.param_jitter,
                args.speckle_snr_factor,
                args.param_scale,
            )

            avg_frames = max(1, int(args.avg_clean_frames))
            noisy_frames = []
            noise_maps = None
            for _ in range(avg_frames):
                if save_maps:
                    noisy_tensor, noise_maps = add_oct_noise_mixture(
                        clean_tensor, weights, params, return_maps=True
                    )
                else:
                    noisy_tensor = add_oct_noise_mixture(clean_tensor, weights, params)
                noisy_frames.append(noisy_tensor[0, 0].cpu().numpy())
            noisy = noisy_frames[0]
            if avg_frames > 1:
                clean_avg = np.mean(noisy_frames, axis=0)
            else:
                clean_avg = None
            transform_seed = args.seed + idx
            rng = np.random.RandomState(transform_seed)
            noisy = apply_transforms(
                noisy,
                args.post_blur_sigma,
                args.shift_max,
                args.elastic_alpha,
                args.elastic_sigma,
                rng,
            )
            if save_maps and noise_maps is not None:
                maps_np = {}
                for key, tensor in noise_maps.items():
                    arr = tensor[0, 0].cpu().numpy()
                    if args.post_blur_sigma > 0 or args.shift_max > 0 or args.elastic_alpha > 0:
                        rng_map = np.random.RandomState(transform_seed)
                        arr = apply_transforms(
                            arr,
                            args.post_blur_sigma,
                            args.shift_max,
                            args.elastic_alpha,
                            args.elastic_sigma,
                            rng_map,
                        )
                    maps_np[key] = arr.astype(np.float32)

            noisy_path = make_noisy_path(clean_path, args.noisy_name)
            noisy_path.parent.mkdir(parents=True, exist_ok=True)
            if noisy_path.exists() and not args.overwrite:
                raise SystemExit(f"Output exists (use --overwrite): {noisy_path}")
            Image.fromarray((noisy * 255.0).clip(0, 255).astype(np.uint8)).save(noisy_path)

            maps_path = None
            if save_maps and noise_maps is not None:
                maps_path = make_noisy_path(clean_path, noise_maps_name).with_suffix(".npz")
                maps_path.parent.mkdir(parents=True, exist_ok=True)
                if maps_path.exists() and not args.overwrite:
                    raise SystemExit(f"Output exists (use --overwrite): {maps_path}")
                np.savez_compressed(maps_path, **maps_np)

            clean_out_path = clean_path
            clean_img = None
            if clean_avg is not None:
                clean_img = clean_avg

            needs_clean_save = (
                clean_img is not None
                or args.clean_blur_sigma > 0
                or args.clean_shift_max > 0
                or args.clean_elastic_alpha > 0
            )
            if needs_clean_save:
                if clean_img is None:
                    clean_img = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
                if args.clean_name:
                    clean_name = args.clean_name
                else:
                    clean_name = "clean_avg" if clean_avg is not None else "clean_analysis"
                clean_out_path = make_clean_path(clean_path, clean_name)
                clean_out_path.parent.mkdir(parents=True, exist_ok=True)
                if clean_out_path.exists() and not args.overwrite:
                    raise SystemExit(f"Output exists (use --overwrite): {clean_out_path}")
                rng_clean = np.random.RandomState(args.seed + idx + 97)
                clean_img = apply_transforms(
                    clean_img,
                    args.clean_blur_sigma,
                    args.clean_shift_max,
                    args.clean_elastic_alpha,
                    args.clean_elastic_sigma,
                    rng_clean,
                )
                Image.fromarray((clean_img * 255.0).clip(0, 255).astype(np.uint8)).save(clean_out_path)

            pairs_lines.append(f"{clean_out_path}\t{noisy_path}\n")
            record = {
                "clean_path": str(clean_out_path),
                "noisy_path": str(noisy_path),
                "weights": weights,
                "params": params,
            }
            if maps_path is not None:
                record["noise_maps_path"] = str(maps_path)
            wf.write(json.dumps(record) + "\n")

            if (idx + 1) % 200 == 0 or idx == len(clean_paths) - 1:
                print(f"  Processed: {idx + 1}/{len(clean_paths)}", flush=True)

    header = [
        f"# Analysis JSON: {analysis_path}",
        f"# Weights mode: {args.weights_mode}",
        f"# Param jitter: {args.param_jitter}",
        "# Format: clean_path<TAB>noisy_path",
        "#",
    ]
    pairs_path.write_text("\n".join(header) + "\n" + "".join(pairs_lines))
    print(f"✓ Wrote pairs: {pairs_path} ({len(clean_paths)})")
    print(f"✓ Wrote weights: {weights_path}")


if __name__ == "__main__":
    main()
