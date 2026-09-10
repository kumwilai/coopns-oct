#!/usr/bin/env python3
"""
Evaluate NAFNet backbone and CoopNS-OCT on clinical feature metrics.

Computes absolute feature values (ratio to clean GT, 1.0 = perfect) for:
  - Contrast (local std in 7x7 windows)
  - Boundary (Sobel vertical gradient magnitude)
  - Texture (Laplacian variance)
  - Edge (combined Sobel magnitude)
  - PSNR, SSIM, CNR

These are directly comparable with BM3D/NLM results from eval_bm3d_nlm_clinical.py.

Usage:
    python eval_model_clinical.py \
        --checkpoint benchmarks/pretrained/best_model_cooperative.pth \
        --backbone benchmarks/pretrained/nafnet_backbone.pth \
        --test_jsonl pku37_oct_dataset/pku37_real_test.jsonl \
        --dataset_name PKU37
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    compute_psnr,
    compute_ssim,
)


@torch.no_grad()
def compute_clinical_metrics(output, clean, device):
    """Compute clinical feature values as ratio to clean GT (1.0 = perfect)."""
    m = {}

    # PSNR / SSIM
    m['psnr'] = compute_psnr(output, clean)
    m['ssim'] = compute_ssim(output, clean)

    # CNR
    signal_mask = (clean > clean.mean()).float()
    bg_mask = 1.0 - signal_mask
    s_sum = signal_mask.sum().clamp(min=1)
    b_sum = bg_mask.sum().clamp(min=1)
    sig = (output * signal_mask).sum() / s_sum
    bg = (output * bg_mask).sum() / b_sum
    bg_std = torch.sqrt(((output - bg)**2 * bg_mask).sum() / b_sum + 1e-8).clamp(min=1e-4)
    m['cnr'] = ((sig - bg) / bg_std).clamp(-100, 100).item()

    # Sobel / Laplacian kernels
    sobel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=device).view(1,1,3,3)
    sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device).view(1,1,3,3)
    laplacian = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=device).view(1,1,3,3)

    # Contrast (local std in 7x7 windows)
    for img, prefix in [(output, 'output'), (clean, 'clean')]:
        mu = F.avg_pool2d(img, 7, 1, 3)
        std_map = torch.sqrt((F.avg_pool2d(img**2, 7, 1, 3) - mu**2).clamp(min=1e-8))
        m[f'{prefix}_contrast'] = std_map.mean().item()

    clean_contrast = max(m['clean_contrast'], 1e-6)
    m['contrast'] = m['output_contrast'] / clean_contrast

    # Boundary (vertical gradient)
    out_gy = F.conv2d(output, sobel_y, padding=1).abs()
    clean_gy = F.conv2d(clean, sobel_y, padding=1).abs()
    clean_gy_mean = clean_gy.mean().clamp(min=1e-4).item()
    m['boundary'] = out_gy.mean().item() / clean_gy_mean

    # Texture (Laplacian)
    out_lap = F.conv2d(output, laplacian, padding=1).abs()
    clean_lap = F.conv2d(clean, laplacian, padding=1).abs()
    clean_lap_mean = clean_lap.mean().clamp(min=1e-4).item()
    m['texture'] = out_lap.mean().item() / clean_lap_mean

    # Edge (combined Sobel magnitude)
    out_gx = F.conv2d(output, sobel_x, padding=1)
    clean_gx = F.conv2d(clean, sobel_x, padding=1)
    out_edge = torch.sqrt(out_gx**2 + out_gy**2 + 1e-8)
    clean_edge = torch.sqrt(clean_gx**2 + clean_gy**2 + 1e-8)
    clean_edge_mean = clean_edge.mean().clamp(min=1e-4).item()
    m['edge'] = out_edge.mean().item() / clean_edge_mean

    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--backbone_name', type=str, default='nafnet',
                        choices=['nafnet', 'dncnn', 'swinir', 'kbnet', 'mambair'])
    parser.add_argument('--test_jsonl', required=True)
    parser.add_argument('--dataset_name', default='PKU37')
    parser.add_argument('--output_dir', default='outputs/baselines')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model (backbone={args.backbone_name})...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.backbone
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    del ckpt

    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Load test pairs
    pairs = []
    with open(args.test_jsonl) as f:
        for line in f:
            entry = json.loads(line.strip())
            pairs.append((entry['noisy_path'], entry['clean_path']))
    print(f"Loaded {len(pairs)} test pairs")

    # Evaluate
    backbone_metrics = []
    corrected_metrics = []
    t0 = time.time()

    for i, (noisy_path, clean_path) in enumerate(pairs):
        noisy_np = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean_np = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        noisy_t = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).to(device)
        clean_t = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            corrected, backbone_out, info = model(noisy_t)

        bm = compute_clinical_metrics(backbone_out, clean_t, device)
        cm = compute_clinical_metrics(corrected, clean_t, device)
        backbone_metrics.append(bm)
        corrected_metrics.append(cm)

        if (i + 1) % max(1, len(pairs) // 5) == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(pairs)}] BB PSNR={bm['psnr']:.2f} Corr PSNR={cm['psnr']:.2f} "
                  f"BB contrast={bm['contrast']:.4f} Corr contrast={cm['contrast']:.4f} "
                  f"({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0

    # Compute averages
    keys = ['psnr', 'ssim', 'cnr', 'contrast', 'boundary', 'texture', 'edge']
    bb_avg = {k: np.mean([m[k] for m in backbone_metrics]) for k in keys}
    co_avg = {k: np.mean([m[k] for m in corrected_metrics]) for k in keys}

    # Print results
    ds = args.dataset_name
    print(f"\n{'='*70}")
    print(f"  NAFNet Backbone Results on {ds} ({len(pairs)} images)")
    print(f"{'='*70}")
    print(f"  PSNR:       {bb_avg['psnr']:.2f} dB")
    print(f"  SSIM:       {bb_avg['ssim']:.4f}")
    print(f"  CNR:        {bb_avg['cnr']:.2f}")
    print(f"  Contrast:   {bb_avg['contrast']:.4f}")
    print(f"  Boundary:   {bb_avg['boundary']:.4f}")
    print(f"  Texture:    {bb_avg['texture']:.4f}")
    print(f"  Edge:       {bb_avg['edge']:.4f}")

    print(f"\n{'='*70}")
    print(f"  CoopNS-OCT Results on {ds} ({len(pairs)} images)")
    print(f"{'='*70}")
    print(f"  PSNR:       {co_avg['psnr']:.2f} dB")
    print(f"  SSIM:       {co_avg['ssim']:.4f}")
    print(f"  CNR:        {co_avg['cnr']:.2f}")
    print(f"  Contrast:   {co_avg['contrast']:.4f}")
    print(f"  Boundary:   {co_avg['boundary']:.4f}")
    print(f"  Texture:    {co_avg['texture']:.4f}")
    print(f"  Edge:       {co_avg['edge']:.4f}")

    print(f"\n  Time: {elapsed:.1f}s total ({elapsed/len(pairs):.2f}s/image)")

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'dataset': ds,
        'n_images': len(pairs),
        'backbone_averages': bb_avg,
        'corrected_averages': co_avg,
        'backbone_per_image': backbone_metrics,
        'corrected_per_image': corrected_metrics,
    }
    out_path = os.path.join(args.output_dir, f'model_clinical_{ds.lower()}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
