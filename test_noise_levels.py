#!/usr/bin/env python3
"""
Test actual noise levels for different batches.
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

print("Testing noise levels:")
print("=" * 70)
print(f"{'Batch':<10} {'Noise Level':<15} {'Decision':<30}")
print("=" * 70)

# Test batches 1-5 (should be Gaussian), 35-40 (non-Gaussian), 68-73 (heavy)
test_batches = list(range(1, 6)) + list(range(35, 41)) + list(range(68, 74))

for batch_idx, (noisy, clean) in enumerate(loader):
    batch_num = batch_idx + 1
    if batch_num not in test_batches:
        continue

    # Test first image in batch
    noise_level = estimate_noise_level(noisy[0:1])

    if noise_level < 0.10:
        decision = "Multi-scale only"
    elif noise_level > 0.15:
        decision = "Heavy blend (30% MS, 40% Prog)"
    else:
        decision = "Moderate blend (40% MS, 30% Prog)"

    batch_type = ""
    if batch_num <= 34:
        batch_type = "(Gaussian)"
    elif batch_num < 68:
        batch_type = "(Moderate non-Gaussian)"
    else:
        batch_type = "(Heavy non-Gaussian)"

    print(f"Batch {batch_num:<4} {batch_type:<20} {noise_level:<15.4f} {decision}")

print("=" * 70)
print("\nThreshold: noise_level < 0.10 → Multi-scale only (~32.5 dB)")
print("           noise_level ≥ 0.10 → Blend with progressive refinement")
