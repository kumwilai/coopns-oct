#!/usr/bin/env python3
"""
Test if Gaussian detection is working correctly.
"""
import torch
from torch.utils.data import DataLoader
from adaptive_oct_denoise import PairedOCTDataset, resize_to

def is_gaussian_noise(image):
    """Test version with debug output."""
    diff_h = torch.diff(image, dim=2)
    diff_v = torch.diff(image, dim=3)
    noise_samples = torch.cat([diff_h.flatten(), diff_v.flatten()])

    mean = noise_samples.mean()
    std = noise_samples.std()
    skewness = ((noise_samples - mean) ** 3).mean() / (std ** 3 + 1e-6)

    is_gauss = abs(skewness.item()) < 0.3

    print(f"  Skewness: {skewness.item():.4f}, Is Gaussian: {is_gauss}")
    return is_gauss

# Test on first few batches
transform = resize_to((64, 64))
dataset = PairedOCTDataset("val_pairs_universal.txt", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

print("Testing Gaussian detection on first 5 batches:")
print("=" * 60)

for batch_idx, (noisy, clean) in enumerate(loader):
    if batch_idx >= 5:
        break

    print(f"\nBatch {batch_idx + 1}:")
    # Test first image in batch
    is_gauss = is_gaussian_noise(noisy[0:1])

print("\n" + "=" * 60)
print("Note: Batches 1-34 should be Gaussian (skewness ~0)")
print("      Batches 35+ are non-Gaussian (skewness > 0.3)")
