#!/usr/bin/env python3
"""
Calibrate analysis-based synthetic noise strength to match target PSNR/SSIM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, shift, map_coordinates

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "nsnd_oct"))

from nsnd.training.synthetic_noise import add_oct_noise_mixture
from nsnd.utils.metrics import compute_psnr, compute_ssim


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


def select_paths(split_dict: Dict[str, List[str]]) -> List[Path]:
    selected = []
    for cls in sorted(split_dict.keys()):
        selected.extend(Path(p) for p in split_dict[cls])
    return selected


def parse_pairs_file(pairs_path: Path, pairs_order: str) -> List[Path]:
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


def load_analysis(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    if "composition_weights" not in data:
        raise ValueError("analysis_json missing composition_weights")
    if "parameters" not in data:
        raise ValueError("analysis_json missing parameters")
    return data


def jitter_value(base: float, jitter: float, min_val: float | None = None) -> float:
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

    period = 1.0 / max(banding_freq, 1e-6)
    period = jitter_value(period, jitter, min_val=2.0)
    period = int(min(512, max(2, round(period))))

    return {
        "speckle_k": jitter_value(speckle_k / scale, jitter, min_val=0.1),
        "speckle_base_k": jitter_value(speckle_k / scale, jitter, min_val=0.1),
        "speckle_snr_factor": float(speckle_snr_factor),
        "speckle_correlation": jitter_value(float(speckle_corr), jitter, min_val=0.0),
        "use_signal_dependent_speckle": True,
        "speckle_depth_gain": float(np.random.uniform(0.4, 1.0)),
        "banding_freq": period,
        "banding_amp": jitter_value(banding_amp * scale, jitter, min_val=0.0),
        "gaussian_sigma": jitter_value(gaussian_sigma * scale, jitter, min_val=0.0),
        "gaussian_depth_gain": float(np.random.uniform(0.2, 0.8)),
        "shot_peak": jitter_value(shot_gain / scale, jitter, min_val=1.0),
        "shot_depth_gain": float(np.random.uniform(0.4, 1.0)),
        "use_depth_profile": True,
    }


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


def compute_metrics(
    clean_paths: List[Path],
    analysis: Dict[str, object],
    param_scale: float,
    weights_mode: str,
    alpha_scale: float,
    param_jitter: float,
    speckle_snr_factor: float,
    blur_sigma: float,
    shift_max: float,
    elastic_alpha: float,
    elastic_sigma: float,
    avg_clean_frames: int,
    clean_blur_sigma: float,
    clean_shift_max: float,
    clean_elastic_alpha: float,
    clean_elastic_sigma: float,
    seed: int,
) -> tuple[float, float]:
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    for idx, clean_path in enumerate(clean_paths):
        np.random.seed(seed + idx)
        torch.manual_seed(seed + idx)
        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
        clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()

        weights = sample_weights(analysis, weights_mode, alpha_scale)
        params = sample_params_from_analysis(analysis, param_jitter, speckle_snr_factor, param_scale)
        avg_frames = max(1, int(avg_clean_frames))
        noisy_frames = []
        for _ in range(avg_frames):
            noisy_tensor = add_oct_noise_mixture(clean_tensor, weights, params)
            noisy_frames.append(noisy_tensor[0, 0].cpu().numpy())

        noisy_np = noisy_frames[0]
        if avg_frames > 1:
            clean_np = np.mean(noisy_frames, axis=0)
        else:
            clean_np = clean_tensor[0, 0].cpu().numpy()

        rng = np.random.RandomState(seed + idx)
        noisy_np = apply_transforms(noisy_np, blur_sigma, shift_max, elastic_alpha, elastic_sigma, rng)
        rng_clean = np.random.RandomState(seed + idx + 97)
        clean_np = apply_transforms(
            clean_np,
            clean_blur_sigma,
            clean_shift_max,
            clean_elastic_alpha,
            clean_elastic_sigma,
            rng_clean,
        )
        noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0)

        psnr_vals.append(compute_psnr(noisy_tensor, clean_tensor))
        ssim_vals.append(compute_ssim(noisy_tensor, clean_tensor))

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_json", type=str, required=True)
    parser.add_argument("--splits_json", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--pairs_in", type=str, default="")
    parser.add_argument("--pairs_order", type=str, default="auto",
                        choices=["auto", "clean_first", "noisy_first"])
    parser.add_argument("--sample_count", type=int, default=200)
    parser.add_argument("--param_scales", type=str, default="")
    parser.add_argument("--param_scale_min", type=float, default=0.5)
    parser.add_argument("--param_scale_max", type=float, default=5.0)
    parser.add_argument("--param_scale_steps", type=int, default=10)
    parser.add_argument("--weights_mode", type=str, default="dirichlet",
                        choices=["dirichlet", "fixed"])
    parser.add_argument("--alpha_scale", type=float, default=1.0)
    parser.add_argument("--param_jitter", type=float, default=0.1)
    parser.add_argument("--speckle_snr_factor", type=float, default=0.5)
    parser.add_argument("--blur_sigmas", type=str, default="0")
    parser.add_argument("--shift_maxes", type=str, default="0")
    parser.add_argument("--elastic_alphas", type=str, default="0")
    parser.add_argument("--elastic_sigma", type=float, default=8.0)
    parser.add_argument("--avg_clean_frames", type=int, default=1)
    parser.add_argument("--clean_blur_sigmas", type=str, default="0")
    parser.add_argument("--clean_shift_maxes", type=str, default="0")
    parser.add_argument("--clean_elastic_alphas", type=str, default="0")
    parser.add_argument("--clean_elastic_sigma", type=float, default=8.0)
    parser.add_argument("--target_psnr", type=float, required=True)
    parser.add_argument("--target_ssim", type=float, required=True)
    parser.add_argument("--ssim_weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if bool(args.splits_json) == bool(args.pairs_in):
        raise SystemExit("Provide exactly one of --splits_json or --pairs_in.")

    analysis = load_analysis(Path(args.analysis_json))

    if args.splits_json:
        splits = json.loads(Path(args.splits_json).read_text())
        split_dict = splits.get(args.split)
        if split_dict is None:
            raise SystemExit(f"Split '{args.split}' not found in {args.splits_json}")
        clean_paths = select_paths(split_dict)
    else:
        clean_paths = parse_pairs_file(Path(args.pairs_in), args.pairs_order)

    if args.sample_count > 0:
        clean_paths = clean_paths[: args.sample_count]
    if not clean_paths:
        raise SystemExit("No clean paths selected.")

    if args.param_scales:
        scales = [float(s) for s in args.param_scales.split(",") if s.strip()]
    else:
        steps = max(2, int(args.param_scale_steps))
        scales = np.linspace(args.param_scale_min, args.param_scale_max, steps).tolist()

    blur_sigmas = [float(s) for s in args.blur_sigmas.split(",") if s.strip()]
    shift_maxes = [float(s) for s in args.shift_maxes.split(",") if s.strip()]
    elastic_alphas = [float(s) for s in args.elastic_alphas.split(",") if s.strip()]
    clean_blur_sigmas = [float(s) for s in args.clean_blur_sigmas.split(",") if s.strip()]
    clean_shift_maxes = [float(s) for s in args.clean_shift_maxes.split(",") if s.strip()]
    clean_elastic_alphas = [float(s) for s in args.clean_elastic_alphas.split(",") if s.strip()]

    best = None
    for scale in scales:
        for sigma in blur_sigmas:
            for shift_max in shift_maxes:
                for elastic_alpha in elastic_alphas:
                    for clean_blur in clean_blur_sigmas:
                        for clean_shift in clean_shift_maxes:
                            for clean_elastic in clean_elastic_alphas:
                                psnr, ssim = compute_metrics(
                                    clean_paths,
                                    analysis,
                                    scale,
                                    args.weights_mode,
                                    args.alpha_scale,
                                    args.param_jitter,
                                    args.speckle_snr_factor,
                                    sigma,
                                    shift_max,
                                    elastic_alpha,
                                    args.elastic_sigma,
                                    args.avg_clean_frames,
                                    clean_blur,
                                    clean_shift,
                                    clean_elastic,
                                    args.clean_elastic_sigma,
                                    args.seed,
                                )
                                err = (psnr - args.target_psnr) ** 2 + args.ssim_weight * (ssim - args.target_ssim) ** 2
                                print(
                                    f"scale={scale:.3f} | blur={sigma:.3f} | shift={shift_max:.3f} | "
                                    f"elastic={elastic_alpha:.3f} | clean_blur={clean_blur:.3f} | "
                                    f"clean_shift={clean_shift:.3f} | clean_elastic={clean_elastic:.3f} | "
                                    f"PSNR={psnr:.2f} | SSIM={ssim:.4f} | err={err:.4f}"
                                )
                                if best is None or err < best["err"]:
                                    best = {
                                        "scale": scale,
                                        "sigma": sigma,
                                        "shift": shift_max,
                                        "elastic": elastic_alpha,
                                        "clean_blur": clean_blur,
                                        "clean_shift": clean_shift,
                                        "clean_elastic": clean_elastic,
                                        "psnr": psnr,
                                        "ssim": ssim,
                                        "err": err,
                                    }

    print("\nBest scale:")
    print(
        f"  scale={best['scale']:.3f} | blur={best['sigma']:.3f} | shift={best['shift']:.3f} | "
        f"elastic={best['elastic']:.3f} | clean_blur={best['clean_blur']:.3f} | "
        f"clean_shift={best['clean_shift']:.3f} | clean_elastic={best['clean_elastic']:.3f} | "
        f"PSNR={best['psnr']:.2f} | SSIM={best['ssim']:.4f} | err={best['err']:.4f}"
    )


if __name__ == "__main__":
    main()
