#!/usr/bin/env python3
"""
Debug full batch 1 to see what's happening to each image.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adaptive_oct_denoise import (
    PairedOCTDataset,
    build_model,
    compute_psnr,
    device,
    resize_to,
)

# Import the actual function from the file
import sys
sys.path.insert(0, '.')
from eval_progressive_refinement import (
    multi_strategy_ensemble,
    estimate_noise_level,
    multi_scale_denoise,
)

# Load model
model = build_model(
    base_channels=48,
    residual_mode=True,
    adapter_type="casa",
    backbone_type="noise2void",
)

checkpoint = torch.load("checkpoints/casa_n2v_residual/best_model_ema.pth", map_location=device)
if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device).eval()

# Load first batch
transform = resize_to((64, 64))
dataset = PairedOCTDataset("val_pairs_universal.txt", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

noisy, clean = next(iter(loader))
noisy = noisy.to(device)
clean = clean.to(device)

print("Batch 1 Analysis (all 16 images):")
print("=" * 80)
print(f"{'Img':<5} {'Noise':<10} {'MS-PSNR':<12} {'Multi-Strat':<12} {'Diff':<10} {'Applied'}")
print("=" * 80)

batch_psnrs_ms = []
batch_psnrs_strat = []

for i in range(noisy.shape[0]):
    single_noisy = noisy[i:i+1]
    single_clean = clean[i:i+1]

    # Estimate noise
    noise_level = estimate_noise_level(single_noisy)

    # Multi-scale only
    ms_result = multi_scale_denoise(model, single_noisy)
    ms_psnr = compute_psnr(ms_result, single_clean)

    # Multi-strategy
    strat_result = multi_strategy_ensemble(model, single_noisy, use_tta=False)
    strat_psnr = compute_psnr(strat_result, single_clean)

    batch_psnrs_ms.append(ms_psnr)
    batch_psnrs_strat.append(strat_psnr)

    diff = strat_psnr - ms_psnr

    # Determine what was applied
    if noise_level < 0.05:
        applied = "MS only"
    elif noise_level > 0.12:
        applied = "70% MS + 30% Prog"
    elif noise_level > 0.08:
        applied = "85% MS + 15% Prog"
    else:
        applied = "90% MS + 10% Prog"

    marker = "✓" if diff >= -0.1 else "✗"
    print(f"{i+1:<5} {noise_level:<10.4f} {ms_psnr:<12.2f} {strat_psnr:<12.2f} {diff:<10.2f} {applied:<20} {marker}")

import numpy as np
print("=" * 80)
print(f"Average MS PSNR:       {np.mean(batch_psnrs_ms):.2f} dB")
print(f"Average Strat PSNR:    {np.mean(batch_psnrs_strat):.2f} dB")
print(f"Expected batch avg:    28.86 dB (from running evaluation)")
print("=" * 80)
