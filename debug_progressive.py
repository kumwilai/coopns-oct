#!/usr/bin/env python3
"""
Debug version to see what's happening in first batch.
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

def estimate_noise_level(image):
    diff_h = torch.diff(image, dim=2)
    diff_v = torch.diff(image, dim=3)
    mad = torch.median(torch.abs(torch.cat([diff_h.flatten(), diff_v.flatten()])))
    noise_est = mad / 0.6745
    return noise_est.item()

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

print("Testing Batch 1 (first image):")
print("=" * 60)

# Test first image
test_noisy = noisy[0:1]
test_clean = clean[0:1]

# Estimate noise level
noise_level = estimate_noise_level(test_noisy)
print(f"Noise level: {noise_level:.4f}")

# Test multi-scale only
ms_result = multi_scale_denoise(model, test_noisy)
ms_psnr = compute_psnr(ms_result, test_clean)
print(f"\nMulti-scale only PSNR: {ms_psnr:.2f} dB")

# Test direct denoising
with torch.no_grad():
    direct_result = model(test_noisy)
direct_psnr = compute_psnr(direct_result, test_clean)
print(f"Direct denoising PSNR: {direct_psnr:.2f} dB")

# Test if blending would be applied
if noise_level < 0.05:
    print(f"\nDecision: noise_level ({noise_level:.4f}) < 0.05")
    print("Strategy: 100% Multi-scale only")
    print(f"Expected PSNR: {ms_psnr:.2f} dB")
else:
    print(f"\nDecision: noise_level ({noise_level:.4f}) >= 0.05")
    print("Strategy: Will blend with progressive refinement")
    print("This might degrade performance!")

print("=" * 60)
print("\nExpected: Batch 1 should get ~32.5 dB from multi-scale")
print(f"Actual multi-scale PSNR: {ms_psnr:.2f} dB")
