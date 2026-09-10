#!/usr/bin/env python3
"""
Analyze empirical OCT noise composition and parameters from real datasets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def _load_grayscale(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        img = np.load(path)
        if img.ndim >= 3:
            mid = img.shape[0] // 2
            img = img[mid]
    else:
        from PIL import Image

        img = Image.open(path).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / (img.max() + 1e-6)
    return img


def _uniform_filter(image: np.ndarray, window_size: int) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter

        return uniform_filter(image, window_size)
    except Exception:
        kernel = np.ones((window_size, window_size), dtype=np.float32)
        kernel = kernel / kernel.sum()
        pad = window_size // 2
        padded = np.pad(image, pad, mode="reflect")
        out = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                patch = padded[i : i + window_size, j : j + window_size]
                out[i, j] = float(np.sum(patch * kernel))
        return out


def analyze_speckle(image: np.ndarray, window_size: int = 15) -> Dict[str, float]:
    """
    Speckle is multiplicative: observed = clean * speckle
    In homogeneous regions: CV = std/mean = 1/sqrt(k) for Gamma(k,1/k)
    """
    local_mean = _uniform_filter(image, window_size)
    local_sq_mean = _uniform_filter(image ** 2, window_size)
    local_var = local_sq_mean - local_mean ** 2
    local_std = np.sqrt(np.maximum(local_var, 0))

    cv = local_std / (local_mean + 1e-6)
    mask = local_mean > np.percentile(image, 30)
    median_cv = np.median(cv[mask]) if np.any(mask) else float(np.median(cv))
    estimated_k = 1.0 / (median_cv ** 2 + 1e-6)

    speckle_score = np.clip(median_cv / 0.5, 0.0, 1.0)

    # Estimate spatial correlation from speckle ratio
    ratio = image / (local_mean + 1e-6)
    ratio = ratio - np.mean(ratio)
    if ratio.size > 1:
        shifted = np.roll(ratio, shift=1, axis=1)
        corr = np.corrcoef(ratio.ravel(), shifted.ravel())[0, 1]
    else:
        corr = 0.0
    corr = float(np.clip(corr, 0.01, 0.999))
    correlation_sigma = math.sqrt(max(1e-6, -1.0 / (2.0 * math.log(corr))))

    return {
        "k": float(np.clip(estimated_k, 1.5, 10.0)),
        "cv": float(median_cv),
        "score": float(speckle_score),
        "correlation": float(corr),
        "correlation_sigma": float(correlation_sigma),
    }


def analyze_banding(image: np.ndarray) -> Dict[str, float]:
    """
    Banding = horizontal stripes = energy at low horizontal frequencies in FFT
    """
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    h, w = image.shape
    center_h, center_w = h // 2, w // 2
    band_half = min(20, max(2, h // 8))

    horizontal_band = magnitude[center_h - band_half : center_h + band_half + 1, center_w].copy()
    dc_start = max(0, band_half - 2)
    dc_end = min(horizontal_band.shape[0], band_half + 3)
    horizontal_band[dc_start:dc_end] = 0.0

    total_energy = float(magnitude.sum())
    band_energy = float(horizontal_band.sum())

    banding_score = np.clip(band_energy / (total_energy * 0.01 + 1e-6), 0.0, 1.0)

    peak_idx = int(np.argmax(horizontal_band))
    dominant_freq = abs(peak_idx - band_half) / float(h)

    row_means = image.mean(axis=1)
    amplitude = float((row_means.max() - row_means.min()) / 2.0)

    return {
        "frequency": float(dominant_freq),
        "amplitude": float(amplitude),
        "score": float(banding_score),
    }


def analyze_gaussian(image: np.ndarray) -> Dict[str, float]:
    """
    Gaussian noise is signal-independent.
    Estimate from dark/uniform regions where other noise is minimal.
    """
    dark_threshold = np.percentile(image, 15)
    dark_mask = image < dark_threshold

    if dark_mask.sum() < 100:
        return {"sigma": 0.03, "score": 0.1}

    dark_values = image[dark_mask]
    sigma = float(dark_values.std())
    gaussian_score = float(np.clip(sigma / 0.1, 0.0, 1.0))

    return {"sigma": sigma, "score": gaussian_score}


def analyze_shot(image: np.ndarray, window_size: int = 15) -> Dict[str, float]:
    """
    Shot noise (Poisson): variance proportional to mean
    """
    local_mean = _uniform_filter(image, window_size)
    local_sq_mean = _uniform_filter(image ** 2, window_size)
    local_var = local_sq_mean - local_mean ** 2

    mean_flat = local_mean.flatten()[::100]
    var_flat = local_var.flatten()[::100]

    if mean_flat.size < 2:
        corr = 0.0
    else:
        try:
            from scipy.stats import pearsonr

            corr, _ = pearsonr(mean_flat, var_flat)
        except Exception:
            corr = np.corrcoef(mean_flat, var_flat)[0, 1]

    shot_score = float(np.clip(corr, 0.0, 1.0))

    valid = mean_flat > 0.01
    if valid.sum() > 10:
        gain = float(np.median(mean_flat[valid] / (var_flat[valid] + 1e-6)))
    else:
        gain = 100.0

    return {
        "gain": float(np.clip(gain, 10.0, 500.0)),
        "correlation": float(corr),
        "score": shot_score,
    }


def analyze_dataset(image_dir: str, num_samples: int = 100, dataset_name: str = "unknown") -> Dict[str, object]:
    """Analyze noise composition across dataset."""
    image_dir = Path(image_dir)
    image_paths: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.npy"):
        image_paths.extend(image_dir.glob(ext))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    if len(image_paths) > num_samples:
        image_paths = list(np.random.choice(image_paths, num_samples, replace=False))

    all_results = []
    for path in tqdm(image_paths, desc=f"Analyzing {dataset_name}"):
        img = _load_grayscale(path)
        result = {
            "speckle": analyze_speckle(img),
            "banding": analyze_banding(img),
            "gaussian": analyze_gaussian(img),
            "shot": analyze_shot(img),
        }
        all_results.append(result)

    agg = {
        "speckle_k": float(np.median([r["speckle"]["k"] for r in all_results])),
        "speckle_cv": float(np.median([r["speckle"]["cv"] for r in all_results])),
        "speckle_correlation": float(np.median([r["speckle"]["correlation"] for r in all_results])),
        "speckle_correlation_sigma": float(np.median([r["speckle"]["correlation_sigma"] for r in all_results])),
        "banding_freq": float(np.median([r["banding"]["frequency"] for r in all_results])),
        "banding_amp": float(np.median([r["banding"]["amplitude"] for r in all_results])),
        "gaussian_sigma": float(np.median([r["gaussian"]["sigma"] for r in all_results])),
        "shot_gain": float(np.median([r["shot"]["gain"] for r in all_results])),
    }

    scores = np.array(
        [
            np.median([r["speckle"]["score"] for r in all_results]),
            np.median([r["banding"]["score"] for r in all_results]),
            np.median([r["gaussian"]["score"] for r in all_results]),
            np.median([r["shot"]["score"] for r in all_results]),
        ],
        dtype=np.float32,
    )
    scores = np.clip(scores, 1e-6, None)
    weights = scores / scores.sum()

    concentration = 5.0
    alpha_vec = (weights * concentration).tolist()

    output = {
        "dataset": dataset_name,
        "num_images": len(image_paths),
        "composition_weights": weights.tolist(),
        "dirichlet_alpha": alpha_vec,
        "parameters": agg,
    }

    out_path = Path(f"{dataset_name}_noise_params.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== {dataset_name} Noise Analysis ===")
    print(
        "Composition: speckle={:.2f}, banding={:.2f}, gaussian={:.2f}, shot={:.2f}".format(
            weights[0], weights[1], weights[2], weights[3]
        )
    )
    print(f"Dirichlet alpha: {alpha_vec}")
    print(f"Speckle k: {agg['speckle_k']:.2f}, corr sigma: {agg['speckle_correlation_sigma']:.2f}")
    print(f"Banding freq: {agg['banding_freq']:.4f}, amp: {agg['banding_amp']:.4f}")
    print(f"Gaussian sigma: {agg['gaussian_sigma']:.4f}")
    print(f"Shot gain: {agg['shot_gain']:.1f}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OCT noise composition.")
    parser.add_argument("--duke_dir", type=str, help="Path to Duke images")
    parser.add_argument("--pku37_dir", type=str, help="Path to PKU37 images")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    if args.duke_dir:
        analyze_dataset(args.duke_dir, args.num_samples, "duke")
    if args.pku37_dir:
        analyze_dataset(args.pku37_dir, args.num_samples, "pku37")


if __name__ == "__main__":
    main()
