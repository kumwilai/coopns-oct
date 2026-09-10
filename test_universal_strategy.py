#!/usr/bin/env python3
"""
Test the universal multi-strategy approach on different batches.
"""
import torch
from torch.utils.data import DataLoader
from adaptive_oct_denoise import PairedOCTDataset, resize_to

def estimate_noise_level(image):
    diff_h = torch.diff(image, dim=2)
    diff_v = torch.diff(image, dim=3)
    mad = torch.median(torch.abs(torch.cat([diff_h.flatten(), diff_v.flatten()])))
    noise_est = mad / 0.6745
    return noise_est.item()

# Test on batches
transform = resize_to((64, 64))
dataset = PairedOCTDataset("val_pairs_universal.txt", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

print("Universal Multi-Strategy Approach")
print("=" * 80)
print(f"{'Batch':<8} {'Type':<20} {'Noise':<12} {'Strategy'}")
print("=" * 80)

# Test representative batches
test_batches = list(range(1, 11)) + list(range(35, 45)) + list(range(68, 78))

for batch_idx, (noisy, clean) in enumerate(loader):
    batch_num = batch_idx + 1
    if batch_num not in test_batches:
        continue

    # Test first image in batch
    noise_level = estimate_noise_level(noisy[0:1])

    batch_type = ""
    if batch_num <= 34:
        batch_type = "Gaussian"
    elif batch_num < 68:
        batch_type = "Moderate non-Gauss"
    else:
        batch_type = "Heavy non-Gauss"

    # Determine strategy based on noise level
    if noise_level < 0.05:
        strategy = "100% Multi-scale only"
        expected_psnr = "~32.5 dB"
    elif noise_level > 0.12:
        strategy = "70% Multi-scale + 30% Progressive"
        expected_psnr = "Improved heavy noise"
    elif noise_level > 0.08:
        strategy = "85% Multi-scale + 15% Progressive"
        expected_psnr = "Slight improvement"
    else:
        strategy = "90% Multi-scale + 10% Progressive"
        expected_psnr = "~32 dB (Gaussian preserved)"

    print(f"{batch_num:<8} {batch_type:<20} {noise_level:<12.4f} {strategy:<35} {expected_psnr}")

print("=" * 80)
print("\nKey Design Principles:")
print("  1. Multi-scale is the base (excellent for Gaussian)")
print("  2. Progressive refinement blended adaptively (helps non-Gaussian)")
print("  3. Light noise (Gaussian): 90-100% multi-scale → preserves 32.5 dB")
print("  4. Heavy noise: 70% multi-scale + 30% progressive → improves non-Gaussian")
print("\nExpected Results:")
print("  - Gaussian batches: ~32 dB (maintained)")
print("  - Non-Gaussian batches: Improved (progressive refinement helps)")
