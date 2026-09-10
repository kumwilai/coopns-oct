#!/usr/bin/env python3
"""
Create a realistic noisy OCT dataset variant from existing clean images.

Outputs:
  - A new noisy folder for each clean image: replace `/clean/` with `/noisy_realistic/`
  - Pair-list files: train_pairs_realistic.txt, val_pairs_realistic.txt, test_pairs_realistic.txt

Noise model (realistic OCT-inspired):
  - Multiplicative speckle (Gamma/Rayleigh-like) with spatial correlation
  - Depth- and intensity-dependent variance (heteroscedastic)
  - Poisson-like shot noise (optional per image)
  - Mild structured banding (optional per image)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterable

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
except Exception as e:  # pragma: no cover
    gaussian_filter = None
    _GAUSSIAN_FILTER_IMPORT_ERROR = e


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class NoiseConfig:
    speckle_k_range: Tuple[float, float] = (5.0, 20.0)  # Gamma shape; lower -> heavier speckle
    speckle_corr_sigma_range: Tuple[float, float] = (0.5, 1.5)  # Gaussian blur for spatial correlation
    speckle_depth_alpha_range: Tuple[float, float] = (0.3, 1.0)  # depth-dependent variance boost

    poisson_prob: float = 0.6
    poisson_peak_range: Tuple[float, float] = (20.0, 80.0)

    gauss_sigma_range: Tuple[float, float] = (0.003, 0.015)
    gauss_intensity_beta_range: Tuple[float, float] = (0.5, 1.5)  # more noise in dark regions

    banding_prob: float = 0.3
    banding_amp_range: Tuple[float, float] = (0.002, 0.01)
    banding_smooth_sigma_range: Tuple[float, float] = (2.0, 6.0)


# Fixed parameter order for simulator-inversion training.
# These are normalized to [0,1] relative to the *effective* sampling ranges used by the generator.
SIMINV_PARAM_NAMES: Tuple[str, ...] = (
    "speckle_k",
    "speckle_corr_sigma",
    "speckle_depth_alpha",
    "poisson_peak",
    "gauss_sigma0",
    "gauss_beta",
    "banding_amp",
    "banding_smooth_sigma",
)


def _normalize_scalar(x: float, lo: float, hi: float) -> float:
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        return 0.0
    return float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))

def _parse_range_f64(text: str) -> Tuple[float, float]:
    """Parse a range argument.

    Accepts:
      - "x" -> (x, x)
      - "a,b" or "a:b" -> (a, b)
    """
    s = str(text).strip()
    if not s:
        raise ValueError("Empty range string.")
    if "," in s:
        a, b = s.split(",", 1)
    elif ":" in s:
        a, b = s.split(":", 1)
    else:
        x = float(s)
        return (x, x)
    lo = float(a.strip())
    hi = float(b.strip())
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def _apply_overrides(cfg: NoiseConfig, args: argparse.Namespace) -> NoiseConfig:
    """Apply CLI overrides (if provided) after severity preset."""
    out = cfg
    if args.speckle_k_range is not None:
        out = dc_replace(out, speckle_k_range=_parse_range_f64(args.speckle_k_range))
    if args.speckle_corr_sigma_range is not None:
        out = dc_replace(out, speckle_corr_sigma_range=_parse_range_f64(args.speckle_corr_sigma_range))
    if args.speckle_depth_alpha_range is not None:
        out = dc_replace(out, speckle_depth_alpha_range=_parse_range_f64(args.speckle_depth_alpha_range))

    if args.poisson_prob is not None:
        out = dc_replace(out, poisson_prob=float(args.poisson_prob))
    if args.poisson_peak_range is not None:
        out = dc_replace(out, poisson_peak_range=_parse_range_f64(args.poisson_peak_range))

    if args.gauss_sigma_range is not None:
        out = dc_replace(out, gauss_sigma_range=_parse_range_f64(args.gauss_sigma_range))
    if args.gauss_intensity_beta_range is not None:
        out = dc_replace(out, gauss_intensity_beta_range=_parse_range_f64(args.gauss_intensity_beta_range))

    if args.banding_prob is not None:
        out = dc_replace(out, banding_prob=float(args.banding_prob))
    if args.banding_amp_range is not None:
        out = dc_replace(out, banding_amp_range=_parse_range_f64(args.banding_amp_range))
    if args.banding_smooth_sigma_range is not None:
        out = dc_replace(out, banding_smooth_sigma_range=_parse_range_f64(args.banding_smooth_sigma_range))

    return out


def apply_severity_preset(cfg: NoiseConfig, severity: str) -> NoiseConfig:
    """Return a new NoiseConfig adjusted by severity.

    - mild: current defaults (light speckle + light additive)
    - medium: stronger speckle/heteroscedastic + more banding
    - hard: heavy speckle/shot/electronic + frequent banding
    """
    if severity == "mild":
        return cfg
    if severity == "medium":
        return NoiseConfig(
            speckle_k_range=(2.0, 8.0),
            speckle_corr_sigma_range=(1.0, 3.0),
            speckle_depth_alpha_range=(0.6, 1.5),
            poisson_prob=0.8,
            poisson_peak_range=(10.0, 50.0),
            gauss_sigma_range=(0.008, 0.03),
            gauss_intensity_beta_range=(1.0, 2.5),
            banding_prob=0.5,
            banding_amp_range=(0.004, 0.02),
            banding_smooth_sigma_range=cfg.banding_smooth_sigma_range,
        )
    if severity == "hard":
        return NoiseConfig(
            speckle_k_range=(1.2, 4.0),
            speckle_corr_sigma_range=(1.5, 4.0),
            speckle_depth_alpha_range=(1.0, 2.0),
            poisson_prob=0.9,
            poisson_peak_range=(5.0, 30.0),
            gauss_sigma_range=(0.015, 0.05),
            gauss_intensity_beta_range=(1.5, 3.0),
            banding_prob=0.7,
            banding_amp_range=(0.008, 0.035),
            banding_smooth_sigma_range=(1.5, 5.0),
        )
    raise ValueError(f"Unknown severity preset: {severity}")

def load_gray01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def save_gray01(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr_uint8, mode="L").save(path)


def _require_gaussian_filter() -> None:
    if gaussian_filter is None:
        raise ImportError(
            f"scipy.ndimage.gaussian_filter is required for realistic noise. "
            f"Import error: {_GAUSSIAN_FILTER_IMPORT_ERROR}"
        )


def apply_realistic_oct_noise(
    clean: np.ndarray,
    rng: np.random.Generator,
    cfg: NoiseConfig,
    scale: float = 1.0,
) -> np.ndarray:
    _require_gaussian_filter()
    H, W = clean.shape

    # Depth map (axial): stronger speckle deeper in tissue
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    depth_map = y ** rng.uniform(1.0, 2.0)

    # Multiplicative speckle: Gamma-distributed with mean 1
    k = rng.uniform(*cfg.speckle_k_range)
    speckle = rng.gamma(shape=k, scale=1.0 / k, size=(H, W)).astype(np.float32)
    speckle = gaussian_filter(speckle, sigma=rng.uniform(*cfg.speckle_corr_sigma_range))

    alpha = rng.uniform(*cfg.speckle_depth_alpha_range)
    speckle = 1.0 + (speckle - 1.0) * (1.0 + alpha * depth_map) * scale
    noisy = clean * speckle

    # Poisson / shot noise (heteroscedastic additive component)
    if rng.random() < cfg.poisson_prob:
        peak = rng.uniform(*cfg.poisson_peak_range) / max(1e-6, scale)
        peak = max(2.0, peak)
        noisy = rng.poisson(np.clip(noisy, 0.0, 1.0) * peak).astype(np.float32) / peak

    # Additive Gaussian electronics noise, stronger in dark regions
    sigma0 = rng.uniform(*cfg.gauss_sigma_range) * scale
    beta = rng.uniform(*cfg.gauss_intensity_beta_range)
    gauss_sigma_map = sigma0 * (1.0 + beta * (1.0 - clean))
    noisy = noisy + rng.normal(loc=0.0, scale=gauss_sigma_map, size=(H, W)).astype(np.float32)

    # Mild banding / fixed-pattern (per-row bias)
    if rng.random() < cfg.banding_prob:
        amp = rng.uniform(*cfg.banding_amp_range) * scale
        row_bias = rng.normal(0.0, amp, size=(H, 1)).astype(np.float32)
        row_bias = gaussian_filter(row_bias, sigma=rng.uniform(*cfg.banding_smooth_sigma_range))
        noisy = noisy + row_bias

    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def apply_realistic_oct_noise_with_params(
    clean: np.ndarray,
    rng: np.random.Generator,
    cfg: NoiseConfig,
    scale: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Same noise model as apply_realistic_oct_noise, but also returns the sampled parameters."""
    _require_gaussian_filter()
    H, W = clean.shape

    # Depth map (axial): stronger speckle deeper in tissue
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    depth_map = y ** rng.uniform(1.0, 2.0)

    # Multiplicative speckle: Gamma-distributed with mean 1
    k = float(rng.uniform(*cfg.speckle_k_range))
    speckle = rng.gamma(shape=k, scale=1.0 / max(1e-6, k), size=(H, W)).astype(np.float32)
    corr_sigma = float(rng.uniform(*cfg.speckle_corr_sigma_range))
    speckle = gaussian_filter(speckle, sigma=corr_sigma)

    alpha = float(rng.uniform(*cfg.speckle_depth_alpha_range))
    speckle = 1.0 + (speckle - 1.0) * (1.0 + alpha * depth_map) * float(scale)
    noisy = clean * speckle

    # Poisson / shot noise
    poisson_on = bool(rng.random() < cfg.poisson_prob)
    poisson_peak = 0.0
    if poisson_on:
        raw_peak = float(rng.uniform(*cfg.poisson_peak_range))
        poisson_peak = float(max(2.0, raw_peak / max(1e-6, float(scale))))
        noisy = rng.poisson(np.clip(noisy, 0.0, 1.0) * poisson_peak).astype(np.float32) / poisson_peak

    # Additive Gaussian electronics noise
    gauss_sigma0 = float(rng.uniform(*cfg.gauss_sigma_range) * float(scale))
    gauss_beta = float(rng.uniform(*cfg.gauss_intensity_beta_range))
    gauss_sigma_map = gauss_sigma0 * (1.0 + gauss_beta * (1.0 - clean))
    noisy = noisy + rng.normal(loc=0.0, scale=gauss_sigma_map, size=(H, W)).astype(np.float32)

    # Banding / fixed-pattern (per-row bias)
    banding_on = bool(rng.random() < cfg.banding_prob)
    banding_amp = 0.0
    banding_smooth_sigma = 0.0
    if banding_on:
        banding_amp = float(rng.uniform(*cfg.banding_amp_range) * float(scale))
        row_bias = rng.normal(0.0, banding_amp, size=(H, 1)).astype(np.float32)
        banding_smooth_sigma = float(rng.uniform(*cfg.banding_smooth_sigma_range))
        row_bias = gaussian_filter(row_bias, sigma=banding_smooth_sigma)
        noisy = noisy + row_bias

    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)

    # Effective sampling ranges (after scale transforms used by the generator).
    peak_lo, peak_hi = cfg.poisson_peak_range
    peak_lo_eff, peak_hi_eff = float(peak_lo / max(1e-6, float(scale))), float(peak_hi / max(1e-6, float(scale)))
    gsig_lo, gsig_hi = cfg.gauss_sigma_range
    gsig_lo_eff, gsig_hi_eff = float(gsig_lo * float(scale)), float(gsig_hi * float(scale))
    bamp_lo, bamp_hi = cfg.banding_amp_range
    bamp_lo_eff, bamp_hi_eff = float(bamp_lo * float(scale)), float(bamp_hi * float(scale))

    ranges_eff = {
        "speckle_k": list(map(float, cfg.speckle_k_range)),
        "speckle_corr_sigma": list(map(float, cfg.speckle_corr_sigma_range)),
        "speckle_depth_alpha": list(map(float, cfg.speckle_depth_alpha_range)),
        "poisson_peak": [peak_lo_eff, peak_hi_eff],
        "gauss_sigma0": [gsig_lo_eff, gsig_hi_eff],
        "gauss_beta": list(map(float, cfg.gauss_intensity_beta_range)),
        "banding_amp": [bamp_lo_eff, bamp_hi_eff],
        "banding_smooth_sigma": list(map(float, cfg.banding_smooth_sigma_range)),
    }

    vec = [
        _normalize_scalar(k, *ranges_eff["speckle_k"]),
        _normalize_scalar(corr_sigma, *ranges_eff["speckle_corr_sigma"]),
        _normalize_scalar(alpha, *ranges_eff["speckle_depth_alpha"]),
        _normalize_scalar(poisson_peak, *ranges_eff["poisson_peak"]) if poisson_on else 0.0,
        _normalize_scalar(gauss_sigma0, *ranges_eff["gauss_sigma0"]),
        _normalize_scalar(gauss_beta, *ranges_eff["gauss_beta"]),
        _normalize_scalar(banding_amp, *ranges_eff["banding_amp"]) if banding_on else 0.0,
        _normalize_scalar(banding_smooth_sigma, *ranges_eff["banding_smooth_sigma"]) if banding_on else 0.0,
    ]

    params = {
        "param_names": list(SIMINV_PARAM_NAMES),
        "params_raw": {
            "speckle_k": k,
            "speckle_corr_sigma": corr_sigma,
            "speckle_depth_alpha": alpha,
            "poisson_on": poisson_on,
            "poisson_peak": float(poisson_peak),
            "gauss_sigma0": gauss_sigma0,
            "gauss_beta": gauss_beta,
            "banding_on": banding_on,
            "banding_amp": float(banding_amp),
            "banding_smooth_sigma": float(banding_smooth_sigma),
            "scale": float(scale),
        },
        "ranges_effective": ranges_eff,
        "params_norm_vec": vec,
        "params_norm": {k: float(v) for k, v in zip(SIMINV_PARAM_NAMES, vec)},
    }
    return noisy, params


def iter_clean_paths_from_splits(
    splits: Dict[str, Dict[str, List[str]]],
    split_names: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Flatten clean paths from selected splits (train/val/test)."""
    paths: List[Path] = []
    split_iter = split_names if split_names is not None else splits.keys()
    for split_name in split_iter:
        split_dict = splits.get(split_name, {})
        for items in split_dict.values():
            for p in items:
                path = Path(p)
                if path.suffix.lower() in IMG_EXTS:
                    paths.append(path)
    # De-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def iter_clean_paths_stratified(
    splits: Dict[str, Dict[str, List[str]]],
    split_names: Sequence[str],
    rng: np.random.Generator,
    max_total: Optional[int] = None,
    max_per_class: Optional[int] = None,
) -> List[Path]:
    """Select clean paths stratified by class using round-robin sampling.

    If max_per_class is set, take up to that many from each class.
    If max_total is set, take up to that many total across classes.
    """
    per_class: Dict[str, List[Path]] = {}
    for split_name in split_names:
        split_dict = splits.get(split_name, {})
        for cls, items in split_dict.items():
            per_class.setdefault(cls, [])
            for p in items:
                path = Path(p)
                if path.suffix.lower() in IMG_EXTS:
                    per_class[cls].append(path)

    # Deduplicate within each class and shuffle for unbiased sampling
    for cls, items in per_class.items():
        uniq = list(dict.fromkeys(str(p) for p in items))
        paths = [Path(s) for s in uniq]
        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        per_class[cls] = paths

    classes = list(per_class.keys())
    selected: List[Path] = []
    idx: Dict[str, int] = {c: 0 for c in classes}
    limit = max_total if max_total is not None else float("inf")

    while len(selected) < limit:
        progressed = False
        for c in classes:
            i = idx[c]
            if i < len(per_class[c]):
                selected.append(per_class[c][i])
                idx[c] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def make_noisy_path(clean_path: Path, noisy_name: str) -> Path:
    parts = list(clean_path.parts)
    try:
        idx = parts.index("clean")
    except ValueError:
        raise ValueError(f"Clean path does not contain a 'clean' folder: {clean_path}")
    parts[idx] = noisy_name
    return Path(*parts)


def write_pair_lists(
    splits: Dict[str, Dict[str, List[str]]],
    noisy_name: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_dict in splits.items():
        out_path = out_dir / f"{split}_pairs_realistic.txt"
        lines: List[str] = []
        for cls, items in split_dict.items():
            for clean_str in items:
                clean_path = Path(clean_str)
                noisy_path = make_noisy_path(clean_path, noisy_name)
                lines.append(f"{noisy_path.as_posix()},{clean_path.as_posix()}")
        out_path.write_text("\n".join(lines) + "\n")
        print(f"✓ Wrote {out_path} ({len(lines)} pairs)")


def write_subset_pairs(
    clean_paths: List[Path],
    noisy_name: str,
    out_path: Path,
) -> None:
    """Write a single pair list for only the provided clean_paths (useful for --max_images runs)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for clean_path in clean_paths:
        noisy_path = make_noisy_path(clean_path, noisy_name)
        if noisy_path.exists() and clean_path.exists():
            lines.append(f"{noisy_path.as_posix()},{clean_path.as_posix()}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"✓ Wrote subset pairs: {out_path} ({len(lines)} pairs)")

def _clean_paths_from_pairs_file(pairs_path: Path) -> List[Path]:
    """Parse clean paths from a noisy,clean pair list file."""
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pairs list file not found: {pairs_path}")
    clean_paths: List[Path] = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s:
                toks = [t.strip() for t in s.split(",", 1)]
            else:
                toks = s.split()
            if len(toks) != 2:
                raise ValueError(f"Expected 'noisy,clean' per line in {pairs_path}, got: {s}")
            _noisy, clean = toks
            clean_paths.append(Path(clean))
    # De-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in clean_paths:
        sp = p.as_posix()
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)
    return uniq


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create realistic noisy OCT dataset variant.")
    parser.add_argument("--splits_json", type=str, default="oct_splits.json",
                        help="Path to oct_splits.json with train/val/test clean paths.")
    parser.add_argument("--subset_pairs", type=str, default=None,
                        help="Optional: generate noise only for clean paths referenced by this pair list "
                             "(expects 'noisy,clean' per line). If set, split selection is ignored and "
                             "a subset pairlist is written.")
    parser.add_argument("--noisy_name", type=str, default="noisy_realistic",
                        help="Folder name to store new noisy images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing noisy images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity", type=str, default="mild", choices=["mild", "medium", "hard"],
                        help="Noise severity preset (medium/hard are more speckle/heteroscedastic).")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Global multiplier for noise magnitude (e.g., 2.0 for stronger noise).")
    parser.add_argument("--speckle_k_range", type=str, default=None,
                        help="Override Gamma speckle shape range 'lo,hi' (lower -> heavier tails).")
    parser.add_argument("--speckle_corr_sigma_range", type=str, default=None,
                        help="Override speckle correlation sigma range 'lo,hi' (higher -> more correlated).")
    parser.add_argument("--speckle_depth_alpha_range", type=str, default=None,
                        help="Override depth variance boost range 'lo,hi' (higher -> noisier deeper tissue).")
    parser.add_argument("--poisson_prob", type=float, default=None,
                        help="Override Poisson probability (0 disables).")
    parser.add_argument("--poisson_peak_range", type=str, default=None,
                        help="Override Poisson peak range 'lo,hi' (lower -> stronger shot noise).")
    parser.add_argument("--gauss_sigma_range", type=str, default=None,
                        help="Override additive Gaussian sigma range 'lo,hi'.")
    parser.add_argument("--gauss_intensity_beta_range", type=str, default=None,
                        help="Override intensity-dependent Gaussian beta range 'lo,hi'.")
    parser.add_argument("--banding_prob", type=float, default=None,
                        help="Override banding probability (0 disables).")
    parser.add_argument("--banding_amp_range", type=str, default=None,
                        help="Override banding amplitude range 'lo,hi'.")
    parser.add_argument("--banding_smooth_sigma_range", type=str, default=None,
                        help="Override banding smoothing sigma range 'lo,hi'.")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Process only first N images (debug).")
    parser.add_argument("--max_images_per_class", type=int, default=None,
                        help="If set, sample up to N images per class (stratified) for the selected split.")
    parser.add_argument("--stratified_split", type=str, default=None, choices=["train", "val", "test", "all"],
                        help="When limiting images, sample stratified by class from this split.")
    parser.add_argument("--progress_every", type=int, default=500,
                        help="Print progress every N images processed (0 disables).")
    parser.add_argument("--pairlists_out_dir", type=str, default=".",
                        help="Directory to write *_pairs_realistic.txt.")
    parser.add_argument("--write_params_jsonl", action="store_true",
                        help="Also write per-image noise parameters as JSONL under pairlists_out_dir.")
    args = parser.parse_args(argv)

    splits = None
    if args.subset_pairs is None:
        splits_path = Path(args.splits_json)
        splits = json.loads(splits_path.read_text())

    rng = np.random.default_rng(args.seed)
    cfg = apply_severity_preset(NoiseConfig(), args.severity)
    try:
        cfg = _apply_overrides(cfg, args)
    except ValueError as e:
        parser.error(str(e))
    noise_scale = float(args.noise_scale)
    if noise_scale <= 0:
        parser.error("--noise_scale must be > 0")
    print(f"Noise preset: severity={args.severity} | scale={noise_scale}")
    print(f"Noise config: {cfg}")

    # Choose clean paths: either from a subset pairs file or from splits.json
    if args.subset_pairs is not None:
        clean_paths = _clean_paths_from_pairs_file(Path(args.subset_pairs))
        if args.max_images is not None:
            clean_paths = clean_paths[:args.max_images]
        print(f"Found {len(clean_paths)} unique clean images from subset pairs: {args.subset_pairs}")
    else:
        assert splits is not None
        # Determine which splits to consider for generation
        if args.stratified_split is None:
            split_names = list(splits.keys())
        elif args.stratified_split == "all":
            split_names = list(splits.keys())
        else:
            split_names = [args.stratified_split]

        # Choose clean paths (optionally stratified)
        if (args.max_images is not None or args.max_images_per_class is not None) and args.stratified_split is not None:
            clean_paths = iter_clean_paths_stratified(
                splits,
                split_names=split_names,
                rng=rng,
                max_total=args.max_images,
                max_per_class=args.max_images_per_class,
            )
        else:
            clean_paths = iter_clean_paths_from_splits(splits, split_names=split_names)
            if args.max_images is not None:
                clean_paths = clean_paths[:args.max_images]

        print(f"Found {len(clean_paths)} unique clean images in splits.")

    total = len(clean_paths)
    n_written = 0
    n_skipped = 0
    processed = 0
    start_t = time.time()
    progress_every = max(0, int(args.progress_every))

    out_dir = Path(args.pairlists_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_jsonl_path = out_dir / f"noise_params_{args.noisy_name}.jsonl" if args.write_params_jsonl else None
    params_f = params_jsonl_path.open("w", encoding="utf-8") if params_jsonl_path is not None else None

    for clean_path in clean_paths:
        processed += 1
        noisy_path = make_noisy_path(clean_path, args.noisy_name)
        if noisy_path.exists() and not args.overwrite:
            n_skipped += 1
        else:
            clean = load_gray01(clean_path)
            if params_f is not None:
                noisy, params = apply_realistic_oct_noise_with_params(clean, rng, cfg, scale=noise_scale)
                rec = {"noisy": noisy_path.as_posix(), "clean": clean_path.as_posix(), **params}
                params_f.write(json.dumps(rec) + "\n")
            else:
                noisy = apply_realistic_oct_noise(clean, rng, cfg, scale=noise_scale)
            save_gray01(noisy, noisy_path)
            n_written += 1
        if progress_every > 0 and (processed % progress_every == 0 or processed == total):
            elapsed = max(1e-6, time.time() - start_t)
            rate = processed / elapsed
            pct = 100.0 * processed / max(1, total)
            print(
                f"[NoiseGen] {processed}/{total} ({pct:.1f}%) "
                f"written={n_written} skipped={n_skipped} rate={rate:.2f} img/s",
                flush=True,
            )

    print(f"✓ Done. Wrote/updated {n_written} noisy images into '{args.noisy_name}' folders.")
    if params_f is not None:
        params_f.close()
        print(f"✓ Wrote noise parameter metadata: {params_jsonl_path}")
    if splits is not None and args.subset_pairs is None:
        write_pair_lists(splits, args.noisy_name, out_dir)

    # Always write a subset pair list when running in subset mode or when limiting images.
    if args.subset_pairs is not None or args.max_images is not None or args.max_images_per_class is not None:
        subset_out = out_dir / "pairs_realistic_generated.txt"
        write_subset_pairs(clean_paths, args.noisy_name, subset_out)


if __name__ == "__main__":
    main()
