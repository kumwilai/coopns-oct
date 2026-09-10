#!/usr/bin/env python3
"""
Evaluate adaptivity of CASA vs classical baselines under different synthetic noise types and strengths at 64x64.

Outputs:
 - CSV: per-image, per-(noise_type, level, method) PSNR/SSIM/runtime
 - Summary JSON: averages per (noise_type, level, method)
 - Optional sample visuals per configuration
"""

import os
import time
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Local imports
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_oct_denoise import (
    build_model,
    test_time_adaptation,
    resize_to,
    add_rayleigh_noise,
    add_poisson_noise,
    add_mixed_gaussian_noise,
    add_gaussian_additive_noise,
)
from filters import (
    lee_filter_log,
    kuan_filter_log,
    frost_filter_log,
    bilateral_filter_log,
    guided_filter_log,
    estimate_enl_global_linear,
)


def to_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img.astype(np.float32) / 255.0)


def load_clean_images(dir_clean: Path, limit: int, size: int) -> List[Tuple[str, np.ndarray]]:
    files = sorted([p for p in dir_clean.glob('*.png')])[:limit]
    imgs = []
    for p in files:
        img = to_gray01(p)
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        imgs.append((p.name, img))
    return imgs


def simple_box(x: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.blur(x, (k, k))


def simple_gauss(x: np.ndarray, k: int = 5, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(x, (k, k), sigmaX=sigma)


def simple_median(x: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.medianBlur((x * 255).astype(np.uint8), k).astype(np.float32) / 255.0


def synth_noise(torch_img: torch.Tensor, noise_type: str, level: float) -> torch.Tensor:
    # torch_img: [1,1,H,W]
    if noise_type == 'rayleigh':
        # interpret level as sigma
        b = torch_img.shape[0]
        noisy = add_rayleigh_noise(torch_img, (level, level))
    elif noise_type == 'poisson':
        # level ~peak
        b = torch_img.shape[0]
        noisy = add_poisson_noise(torch_img, lambda_scale=level/30.0)  # base 30 scaled
    elif noise_type == 'gaussian_add':
        noisy = add_gaussian_additive_noise(torch_img, (level, level))
    elif noise_type == 'gaussian_mult':
        noisy = add_mixed_gaussian_noise(torch_img, (level, level))
    else:
        raise ValueError(noise_type)
    return noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean_dir', type=str, default='oct/normal/val/clean')
    ap.add_argument('--limit', type=int, default=24)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--out_dir', type=str, default='outputs/adaptivity_synth_64')
    ap.add_argument('--ckpt', type=str, default='checkpoints/casa_finetune64_phys/finetuned.pth')
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--tta_steps', type=int, default=20)
    ap.add_argument('--tta_lr', type=float, default=8e-4)
    ap.add_argument('--use_blind_gating', action='store_true', help='Wrap CASA with blind estimator + residual gating')
    ap.add_argument('--noise_types', type=str, default='', help='Comma-separated subset of noise types to test')
    ap.add_argument('--levels', type=str, default='', help='Comma-separated subset of numeric levels to test')
    args = ap.parse_args()

    assert args.size <= 64, 'Do not use resolution more than 64'
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load clean images
    clean_dir = Path(args.clean_dir)
    items = load_clean_images(clean_dir, args.limit, args.size)

    # Build CASA model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(base_channels=32, residual_mode=True, adapter_type='casa')
    model.to(device)
    if args.ckpt and os.path.isfile(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"Loaded checkpoint: {args.ckpt}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")

    # Optional blind estimator gating
    if args.use_blind_gating:
        from adaptive_oct_denoise import SpectralNoiseCharacterizer, AdaptiveDenoiserWithBlindEstimation
        est = SpectralNoiseCharacterizer().to(device)
        model = AdaptiveDenoiserWithBlindEstimation(est, model).to(device)

    # Prepare CSV
    csv_path = out / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'noise_type', 'level', 'method', 'psnr', 'ssim', 'time_ms'])

        # Configs
        configs = [
            ('rayleigh', [0.3, 0.6, 0.9]),
            ('poisson', [20.0, 30.0, 40.0]),
            ('gaussian_add', [0.03, 0.06, 0.10]),
            ('gaussian_mult', [0.01, 0.05, 0.10]),
        ]

        # Apply filters if provided
        allowed_noise = set([s.strip() for s in args.noise_types.split(',') if s.strip()])
        level_filter = None
        if args.levels:
            try:
                level_filter = set([float(x) for x in args.levels.split(',') if x.strip()])
            except Exception:
                level_filter = None

        for noise_type, levels in configs:
            if allowed_noise and (noise_type not in allowed_noise):
                continue
            for level in levels:
                if level_filter is not None and (float(level) not in level_filter):
                    continue
                print(f"\n=== Noise: {noise_type}, level={level} ===")
                for name, clean_np in items:
                    # Create torch clean
                    clean_t = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0)
                    noisy_t = synth_noise(clean_t, noise_type, float(level)).clamp(0,1)
                    noisy_np = noisy_t.squeeze().numpy()

                    # Baselines (classical)
                    enl = estimate_enl_global_linear(noisy_np, win=7)
                    # Noisy baseline
                    t0 = time.time(); p = psnr(clean_np, noisy_np, data_range=1.0); s = ssim(clean_np, noisy_np, data_range=1.0); dt = (time.time()-t0)*1000
                    writer.writerow([name, noise_type, level, 'noisy', p, s, dt])
                    # Simple filters
                    for mname, fn in [('box3x3', simple_box), ('gauss5x5', lambda x: simple_gauss(x,5,1.0)), ('median3x3', simple_median)]:
                        t0 = time.time(); den = fn(noisy_np); p = psnr(clean_np, den, data_range=1.0); s = ssim(clean_np, den, data_range=1.0); dt = (time.time()-t0)*1000
                        writer.writerow([name, noise_type, level, mname, p, s, dt])
                    # Advanced filters
                    for mname, fn in [('lee', lee_filter_log), ('kuan', kuan_filter_log), ('frost', lambda x: frost_filter_log(x, win=3)), ('bilateral', lambda x: bilateral_filter_log(x, win=5, enl=enl)), ('guided', lambda x: guided_filter_log(x, radius=4))]:
                        t0 = time.time(); den = fn(noisy_np); p = psnr(clean_np, den, data_range=1.0); s = ssim(clean_np, den, data_range=1.0); dt = (time.time()-t0)*1000
                        writer.writerow([name, noise_type, level, mname, p, s, dt])

                    # CASA
                    noisy_t = noisy_t.to(device)
                    t0 = time.time()
                    if args.tta:
                        pred = test_time_adaptation(model, noisy_t.clone(), num_steps=args.tta_steps, lr=args.tta_lr)
                    else:
                        with torch.no_grad():
                            pred = model(noisy_t).detach().cpu()
                    dt = (time.time()-t0)*1000
                    p = psnr(clean_np, pred.squeeze().numpy(), data_range=1.0)
                    s = ssim(clean_np, pred.squeeze().numpy(), data_range=1.0)
                    writer.writerow([name, noise_type, level, 'casa_tta' if args.tta else 'casa', p, s, dt])

    # Aggregate summary
    import pandas as pd
    df = pd.read_csv(csv_path)
    summary = df.groupby(['noise_type','level','method']).agg({'psnr':['mean','std'],'ssim':['mean','std'],'time_ms':['mean','std']}).reset_index()
    summary_path = out / 'summary.json'
    # Convert MultiIndex columns
    summary.columns = ['_'.join([c for c in col if c]) for col in summary.columns.values]
    summary.to_json(summary_path, orient='records', indent=2)
    print(f"\nSaved per-image CSV to: {csv_path}")
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == '__main__':
    main()
