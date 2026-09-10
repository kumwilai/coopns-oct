#!/usr/bin/env python3
"""
Evaluate CASA (Adaptive Denoiser) at 64x64 with optional TTA and compare to simple filters.

Uses the pairs list file to ensure a fair, paired evaluation at exactly 64x64.
"""

import os
import argparse
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_oct_denoise import (
    build_model,
    resize_to,
    test_time_adaptation,
)


@torch.no_grad()
def to_gray01(path: str) -> torch.Tensor:
    t = read_image(path)  # [C,H,W] uint8
    if t.dtype != torch.float32:
        t = t.float() / 255.0
    if t.shape[0] == 1:
        gray = t
    elif t.shape[0] == 3:
        r, g, b = t[0:1], t[1:2], t[2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = t[0:1]
    return gray.clamp(0.0, 1.0)


def simple_box(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    kernel = torch.ones((1, 1, k, k), dtype=x.dtype, device=x.device) / (k * k)
    return F.conv2d(x, kernel, padding=k // 2)


def simple_gauss(x: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    ax = torch.arange(-(k // 2), k // 2 + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, k2d, padding=k // 2)


def simple_median3(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=3, padding=1)  # [B, 9*C, L]
    patches = patches.view(B, C, 9, H * W)
    med, _ = patches.median(dim=2)
    return med.view(B, C, H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', type=str, default='oct_val_pairs_24.txt', help='noisy,clean pairs list')
    p.add_argument('--ckpt', type=str, default='checkpoints/casa_finetune64/finetuned.pth', help='model checkpoint to load')
    p.add_argument('--size', type=int, default=64, help='resize to NxN (must be <=64 per instruction)')
    p.add_argument('--tta', action='store_true', help='enable test-time adaptation')
    p.add_argument('--tta_steps', type=int, default=25)
    p.add_argument('--tta_lr', type=float, default=8e-4)
    p.add_argument('--anchor_weight', type=float, default=0.05, help='TTA anchor weight (fidelity)')
    p.add_argument('--out_json', type=str, default='', help='optional path to write metrics JSON')
    args = p.parse_args()

    assert args.size <= 64, 'Do not use resolution more than 64'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build CASA model
    model = build_model(base_channels=32, residual_mode=True, adapter_type='casa')
    model.to(device)

    if os.path.isfile(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"Loaded checkpoint: {args.ckpt}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")

    resize = resize_to((args.size, args.size))

    # Load pairs
    pairs = []
    with open(args.pairs, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                a, b = line.split(',', 1)
            else:
                a, b = line.split()
            pairs.append((a.strip(), b.strip()))

    # Metrics
    casa_ps, casa_ss = [], []
    box_ps, box_ss = [], []
    gau_ps, gau_ss = [], []
    med_ps, med_ss = [], []

    for noisy_path, clean_path in pairs:
        noisy = resize(to_gray01(noisy_path)).unsqueeze(0)  # [1,1,H,W]
        clean = resize(to_gray01(clean_path)).unsqueeze(0)

        # Simple filters
        box = simple_box(noisy)
        gau = simple_gauss(noisy)
        med = simple_median3(noisy)

        # CASA prediction (with optional TTA)
        if args.tta:
            pred = test_time_adaptation(model, noisy.clone(), num_steps=args.tta_steps, lr=args.tta_lr, anchor_weight=args.anchor_weight)
        else:
            with torch.no_grad():
                pred = model(noisy.to(device)).cpu()

        # Convert to numpy for metrics
        n_c = clean.squeeze().numpy()
        n_pred = pred.squeeze().numpy()
        n_box = box.squeeze().numpy()
        n_gau = gau.squeeze().numpy()
        n_med = med.squeeze().numpy()

        casa_ps.append(psnr(n_c, n_pred, data_range=1.0))
        casa_ss.append(ssim(n_c, n_pred, data_range=1.0))
        box_ps.append(psnr(n_c, n_box, data_range=1.0))
        box_ss.append(ssim(n_c, n_box, data_range=1.0))
        gau_ps.append(psnr(n_c, n_gau, data_range=1.0))
        gau_ss.append(ssim(n_c, n_gau, data_range=1.0))
        med_ps.append(psnr(n_c, n_med, data_range=1.0))
        med_ss.append(ssim(n_c, n_med, data_range=1.0))

    def s(arr):
        return float(np.mean(arr))

    avg_casa_psnr, avg_casa_ssim = s(casa_ps), s(casa_ss)
    print("\nAverages over {} pairs at {}x{}:".format(len(pairs), args.size, args.size))
    print("- CASA{}:     PSNR {:.2f} dB, SSIM {:.4f}".format("+TTA" if args.tta else "", avg_casa_psnr, avg_casa_ssim))
    print("- Box3x3:     PSNR {:.2f} dB, SSIM {:.4f}".format(s(box_ps), s(box_ss)))
    print("- Gaussian5x5: PSNR {:.2f} dB, SSIM {:.4f}".format(s(gau_ps), s(gau_ss)))
    print("- Median3x3:  PSNR {:.2f} dB, SSIM {:.4f}".format(s(med_ps), s(med_ss)))

    # Optional JSON output for compare_results.py compatibility
    if args.out_json:
        import json
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump({
                'test_psnr': avg_casa_psnr,
                'test_ssim': avg_casa_ssim,
                'image_size': args.size,
                'tta': bool(args.tta),
                'pairs': len(pairs),
            }, f, indent=2)
        print(f"Metrics JSON written to: {args.out_json}")


if __name__ == '__main__':
    main()
