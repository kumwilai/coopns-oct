#!/usr/bin/env python3
"""
Test if processing batch vs individual images makes a difference.
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

def multi_scale_denoise(model, noisy, scales=[1.0, 0.75, 0.5]):
    H, W = noisy.shape[2:]
    outputs = []

    for scale in scales:
        if scale != 1.0:
            scaled_h, scaled_w = int(H * scale), int(W * scale)
            scaled_input = F.interpolate(noisy, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
        else:
            scaled_input = noisy

        with torch.no_grad():
            scaled_output = model(scaled_input)

        if scale != 1.0:
            output = F.interpolate(scaled_output, size=(H, W), mode='bilinear', align_corners=False)
        else:
            output = scaled_output

        outputs.append(output)

    weights = [0.5, 0.3, 0.2]
    combined = sum(w * out for w, out in zip(weights, outputs))

    return combined.clamp(0, 1)

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

print("Comparing batch vs individual processing:")
print("=" * 80)

# Process whole batch at once
print("\n1. Processing whole batch at once:")
batch_result = multi_scale_denoise(model, noisy)
batch_psnrs = []
for i in range(batch_result.shape[0]):
    psnr = compute_psnr(batch_result[i:i+1], clean[i:i+1])
    batch_psnrs.append(psnr)
    print(f"   Image {i+1}: {psnr:.2f} dB")

import numpy as np
print(f"   Average: {np.mean(batch_psnrs):.2f} dB")

# Process individual images
print("\n2. Processing images individually:")
individual_psnrs = []
for i in range(noisy.shape[0]):
    single_result = multi_scale_denoise(model, noisy[i:i+1])
    psnr = compute_psnr(single_result, clean[i:i+1])
    individual_psnrs.append(psnr)
    print(f"   Image {i+1}: {psnr:.2f} dB")

print(f"   Average: {np.mean(individual_psnrs):.2f} dB")

print("\n" + "=" * 80)
print(f"Difference: {np.mean(batch_psnrs) - np.mean(individual_psnrs):.4f} dB")
if abs(np.mean(batch_psnrs) - np.mean(individual_psnrs)) < 0.01:
    print("✓ Results are identical")
else:
    print("✗ Results differ!")
