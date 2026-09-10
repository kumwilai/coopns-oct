#!/usr/bin/env python3
"""
Verify the advanced noise estimation method.
"""
import torch
from torch.utils.data import DataLoader
from adaptive_oct_denoise import PairedOCTDataset, resize_to, device
import sys
sys.path.insert(0, '.')
from eval_progressive_refinement import estimate_noise_advanced, estimate_noise_level

# Load validation data
transform = resize_to((64, 64))
dataset = PairedOCTDataset('val_pairs_universal.txt', transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

print('ADVANCED NOISE ESTIMATION VERIFICATION')
print('=' * 90)
print(f'{"Batch":<8} {"Expected":<12} {"Simple MAD":<12} {"Adv Noise":<12} {"Est Type":<15} {"Conf":<8} {"Match":<6}')
print('=' * 90)

# Test representative batches from each noise category
test_batches = list(range(1, 6)) + list(range(35, 40)) + list(range(68, 76))

correct_type = 0
correct_level = 0
total = 0

for batch_idx, (noisy, clean) in enumerate(loader):
    if (batch_idx + 1) not in test_batches:
        continue

    noisy = noisy.to(device)

    # Test first image in batch
    # Simple MAD estimation
    simple_noise = estimate_noise_level(noisy[0:1])

    # Advanced estimation
    adv_noise, noise_type, confidence = estimate_noise_advanced(noisy[0:1])

    # Determine expected type
    if batch_idx < 34:
        expected_type = 'Gaussian'
        expected_noise = 'Low (<0.08)'
    elif batch_idx < 68:
        expected_type = 'Moderate'
        expected_noise = 'Med (0.08-0.12)'
    else:
        expected_type = 'Heavy non-Gauss'
        expected_noise = 'High (>0.12)'

    # Check type match
    type_match = '✓' if (expected_type == 'Gaussian' and noise_type == 'gaussian') or \
                        (expected_type != 'Gaussian' and noise_type != 'gaussian') else '✗'

    # Check level match
    if expected_type == 'Gaussian' and adv_noise < 0.08:
        level_match = True
    elif expected_type == 'Moderate' and 0.08 <= adv_noise <= 0.15:
        level_match = True
    elif expected_type == 'Heavy non-Gauss' and adv_noise > 0.10:
        level_match = True
    else:
        level_match = False

    level_symbol = '✓' if level_match else '✗'

    total += 1
    if type_match == '✓':
        correct_type += 1
    if level_match:
        correct_level += 1

    print(f'{batch_idx+1:<8} {expected_type:<12} {simple_noise:<12.4f} {adv_noise:<12.4f} {noise_type:<15} {confidence:<8.2f} {type_match} {level_symbol}')

print('=' * 90)
print(f'\nSUMMARY:')
print(f'Type Detection Accuracy: {correct_type}/{total} ({100*correct_type/total:.1f}%)')
print(f'Level Detection Accuracy: {correct_level}/{total} ({100*correct_level/total:.1f}%)')
print('=' * 90)

# Additional test: Statistical properties
print('\n\nDETAILED STATISTICS FOR SAMPLE BATCHES:')
print('=' * 90)

sample_batches = [1, 36, 69]  # One from each category
loader2 = DataLoader(dataset, batch_size=16, shuffle=False)

for batch_idx, (noisy, clean) in enumerate(loader2):
    if (batch_idx + 1) not in sample_batches:
        continue

    noisy = noisy.to(device)

    # Get detailed stats
    noise_level, noise_type, confidence = estimate_noise_advanced(noisy[0:1])

    # Compute raw statistics for verification
    img = noisy[0:1]

    # Find homogeneous patches
    patch_size = 8
    stride = 4
    B, C, H, W = img.shape

    patches = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = img[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    patch_vars = [p.var().item() for p in patches]
    sorted_indices = sorted(range(len(patch_vars)), key=lambda i: patch_vars[i])
    homogeneous_indices = sorted_indices[:len(sorted_indices)//3]

    # Collect noise samples
    noise_samples = []
    for idx in homogeneous_indices[:20]:
        patch = patches[idx]
        noise = patch - patch.mean()
        noise_samples.append(noise.flatten())

    noise_samples = torch.cat(noise_samples)

    # Compute moments
    mean = noise_samples.mean().item()
    std = noise_samples.std().item()
    z = (noise_samples - mean) / (std + 1e-8)
    skewness = (z ** 3).mean().item()
    kurtosis = (z ** 4).mean().item() - 3

    expected = 'Gaussian' if batch_idx < 34 else 'Moderate' if batch_idx < 68 else 'Heavy'

    print(f'\nBatch {batch_idx+1} ({expected}):')
    print(f'  Estimated: {noise_type} (confidence={confidence:.2f})')
    print(f'  Noise level: {noise_level:.4f}')
    print(f'  Raw statistics:')
    print(f'    - Skewness: {skewness:.3f} (Gaussian≈0, Rayleigh≈0.63, Gamma>1)')
    print(f'    - Kurtosis: {kurtosis:.3f} (Gaussian≈0, Rayleigh≈0.24, Gamma>2)')
    print(f'    - Std dev: {std:.4f}')
    print(f'  Decision logic:')
    if abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
        print(f'    → Classified as Gaussian (|skew|<0.5 and |kurt|<1.0)')
    elif 0.4 < skewness < 0.85 and -0.5 < kurtosis < 1.0:
        print(f'    → Classified as Rayleigh (skew≈0.63, kurt≈0.24)')
    elif skewness > 1.2 and kurtosis > 2.0:
        print(f'    → Classified as Gamma (high skew and kurt)')
    else:
        print(f'    → Classified as non-Gaussian (does not fit patterns)')

print('\n' + '=' * 90)
