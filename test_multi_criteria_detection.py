#!/usr/bin/env python3
"""
Test multi-criteria Gaussian detection on different batches.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adaptive_oct_denoise import PairedOCTDataset, resize_to

def is_gaussian_noise(image):
    """Multi-criteria detector with Laplacian filtering and debug output."""
    # Step 1: Isolate noise using Laplacian
    kernel = torch.tensor([[[
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ]]], dtype=image.dtype, device=image.device)

    B, C, H, W = image.shape
    laplacian = F.conv2d(image, kernel, padding=1)
    noise_samples = laplacian.flatten()

    # Step 2: Compute statistical moments
    mean = noise_samples.mean()
    std = noise_samples.std() + 1e-6
    centered = noise_samples - mean

    # Skewness
    skewness = (centered ** 3).mean() / (std ** 3)

    # Kurtosis
    kurtosis = (centered ** 4).mean() / (std ** 4) - 3

    # Step 3: Spatial uniformity
    patch_size = 16
    patch_vars = []
    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patch = laplacian[:, :, i:i+patch_size, j:j+patch_size]
            patch_vars.append(patch.var())

    if len(patch_vars) > 1:
        patch_vars_tensor = torch.stack(patch_vars)
        var_cv = patch_vars_tensor.std() / (patch_vars_tensor.mean() + 1e-6)
    else:
        var_cv = torch.tensor(0.0)

    # Individual tests
    skew_gauss = abs(skewness.item()) < 0.8
    kurt_gauss = abs(kurtosis.item()) < 1.5
    var_uniform = var_cv.item() < 0.6

    # Voting
    votes = int(skew_gauss) + int(kurt_gauss) + int(var_uniform)
    is_gaussian = votes >= 2

    return {
        'is_gaussian': is_gaussian,
        'skewness': skewness.item(),
        'kurtosis': kurtosis.item(),
        'var_cv': var_cv.item(),
        'skew_pass': skew_gauss,
        'kurt_pass': kurt_gauss,
        'var_pass': var_uniform,
        'votes': votes
    }

# Test on batches
transform = resize_to((64, 64))
dataset = PairedOCTDataset("val_pairs_universal.txt", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

print("Multi-Criteria Gaussian Detection Test")
print("=" * 100)
print(f"{'Batch':<8} {'Type':<20} {'Skew':<10} {'Kurt':<10} {'VarCV':<10} {'S':<3} {'K':<3} {'V':<3} {'Votes':<6} {'Decision'}")
print("=" * 100)

# Test representative batches
test_batches = list(range(1, 11)) + list(range(35, 45)) + list(range(68, 78))

for batch_idx, (noisy, clean) in enumerate(loader):
    batch_num = batch_idx + 1
    if batch_num not in test_batches:
        continue

    # Test first image in batch
    result = is_gaussian_noise(noisy[0:1])

    batch_type = ""
    if batch_num <= 34:
        batch_type = "Gaussian"
    elif batch_num < 68:
        batch_type = "Moderate non-Gauss"
    else:
        batch_type = "Heavy non-Gauss"

    decision = "GAUSSIAN" if result['is_gaussian'] else "NON-GAUSSIAN"

    # Mark if decision matches expected
    expected_gauss = (batch_num <= 34)
    correct = (result['is_gaussian'] == expected_gauss)
    marker = "✓" if correct else "✗"

    print(f"{batch_num:<8} {batch_type:<20} "
          f"{result['skewness']:>9.3f} {result['kurtosis']:>9.3f} {result['var_cv']:>9.3f} "
          f"{'✓' if result['skew_pass'] else '✗':<3} "
          f"{'✓' if result['kurt_pass'] else '✗':<3} "
          f"{'✓' if result['var_pass'] else '✗':<3} "
          f"{result['votes']}/3    {decision:<15} {marker}")

print("=" * 100)
print("\nCriteria (Laplacian-based noise isolation):")
print("  S (Skewness): |skew| < 0.8 for Gaussian")
print("  K (Kurtosis): |kurt| < 1.5 for Gaussian")
print("  V (Var CV):   var_cv < 0.6 for Gaussian")
print("  Decision: ≥2 votes → Gaussian")
print("\nExpected:")
print("  Batches 1-34:   GAUSSIAN")
print("  Batches 35-101: NON-GAUSSIAN")
