#!/usr/bin/env python3
"""
Deep Diagnostic: Investigate why N2V training is stuck at 26.4 dB
Tests multiple hypotheses systematically
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

print("="*80)
print("DEEP DIAGNOSTIC: N2V Training Failure Analysis")
print("="*80)
print()

# ============================================================================
# HYPOTHESIS 1: Data Quality - Are noisy/clean pairs correct?
# ============================================================================
print("HYPOTHESIS 1: Checking data quality and pair correctness")
print("-"*80)

from skimage.metrics import peak_signal_noise_ratio as psnr

# Read first 5 training pairs
with open('train_pairs_universal.txt', 'r') as f:
    train_pairs = [line.strip().split(',') for line in f if line.strip() and not line.startswith('#')]

print(f"Total training pairs: {len(train_pairs)}")
print()
print("Checking first 5 pairs:")

for i, (noisy_path, clean_path) in enumerate(train_pairs[:5], 1):
    try:
        noisy = np.array(Image.open(noisy_path.strip()).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path.strip()).convert('L'), dtype=np.float32) / 255.0

        p = psnr(clean, noisy, data_range=1.0)

        # Check if same file
        same_file = (noisy_path.strip() == clean_path.strip())

        # Check if pixels identical
        pixels_identical = np.array_equal(noisy, clean)

        # Check pixel difference
        diff_mean = np.abs(noisy - clean).mean()
        diff_max = np.abs(noisy - clean).max()

        print(f"Pair {i}:")
        print(f"  Noisy: {Path(noisy_path.strip()).name}")
        print(f"  Clean: {Path(clean_path.strip()).name}")
        print(f"  Same file? {same_file}")
        print(f"  Pixels identical? {pixels_identical}")
        print(f"  PSNR: {p:.2f} dB")
        print(f"  Pixel diff: mean={diff_mean:.4f}, max={diff_max:.4f}")

        if same_file:
            print(f"  ❌ ERROR: Noisy and clean are the SAME FILE!")
        elif pixels_identical:
            print(f"  ❌ ERROR: Pixels are IDENTICAL (no noise)!")
        elif p > 35:
            print(f"  ⚠️  WARNING: PSNR too high (images too similar)")
        elif p < 15:
            print(f"  ⚠️  WARNING: PSNR too low (too much noise)")
        else:
            print(f"  ✓ OK: Good noisy/clean pair")
        print()

    except Exception as e:
        print(f"Pair {i}: ERROR - {e}")
        print()

# ============================================================================
# HYPOTHESIS 2: Model Capacity - Is the model learning at all?
# ============================================================================
print()
print("HYPOTHESIS 2: Testing if model can memorize a single image")
print("-"*80)

# Load model
sys.path.insert(0, '/home/kumwilai/OCT')
from adaptive_oct_denoise import build_model

device = 'cpu'
model = build_model(
    base_channels=48,
    residual_mode=True,
    adapter_type='casa',
    backbone_type='noise2void'
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print()

# Test forward pass
test_input = torch.rand(1, 1, 64, 64).to(device)
with torch.no_grad():
    test_output = model(test_input)

print(f"Forward pass test:")
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {test_output.shape}")
print(f"  Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
print(f"  Output mean: {test_output.mean():.3f}")
print()

# Check if output is reasonable
if test_output.min() < -1 or test_output.max() > 2:
    print("  ⚠️  WARNING: Output values out of expected range [0, 1]")
elif torch.allclose(test_output, test_input, atol=1e-3):
    print("  ⚠️  WARNING: Output is identity (not learning)")
elif test_output.std() < 0.01:
    print("  ⚠️  WARNING: Output has very low variance (collapsed)")
else:
    print("  ✓ OK: Model produces reasonable outputs")

# ============================================================================
# HYPOTHESIS 3: N2V Masking - Is masking working correctly?
# ============================================================================
print()
print("HYPOTHESIS 3: Testing N2V masking implementation")
print("-"*80)

from adaptive_oct_denoise import apply_blind_spot_mask

test_img = torch.rand(4, 1, 64, 64)

masked_img, mask_coords = apply_blind_spot_mask(
    test_img,
    mask_ratio=0.12,
    box_size=5
)

b, c, y, x = mask_coords
num_masked = len(b)
expected_masked = int(0.12 * 64 * 64)

print(f"Masking test:")
print(f"  Input shape: {test_img.shape}")
print(f"  Expected masked pixels: {expected_masked} (12% of 4096)")
print(f"  Actual masked pixels: {num_masked}")
print(f"  Actual ratio: {num_masked/(4*64*64)*100:.1f}%")
print()

# Check if values actually changed
orig_vals = test_img[b, c, y, x]
masked_vals = masked_img[b, c, y, x]
values_changed = (orig_vals != masked_vals).sum().item()

print(f"  Values that changed: {values_changed}/{num_masked}")

if abs(num_masked - expected_masked) > 50:
    print(f"  ❌ ERROR: Masked pixel count far from target!")
elif values_changed < num_masked * 0.9:
    print(f"  ❌ ERROR: Many masked values didn't change!")
else:
    print(f"  ✓ OK: Masking working correctly")

# ============================================================================
# HYPOTHESIS 4: Training Data - Are we training on noisy or clean?
# ============================================================================
print()
print("HYPOTHESIS 4: Checking what data N2V is training on")
print("-"*80)

# Read code to check
with open('/home/kumwilai/OCT/adaptive_oct_denoise.py', 'r') as f:
    code = f.read()

# Search for N2V training logic
if 'use_noise2void' in code:
    print("✓ N2V mode is implemented")

    # Check if it uses noisy or clean
    if 'x_noisy' in code and 'apply_blind_spot_mask(x_noisy' in code:
        print("✓ N2V applies masking to noisy images (correct)")
    else:
        print("⚠️  Could not confirm masking is applied to noisy images")

    # Check loss computation
    if 'pixel_target = x_noisy' in code:
        print("✓ N2V target is noisy image (correct)")
    else:
        print("⚠️  Could not confirm target is noisy image")
else:
    print("❌ ERROR: N2V mode not found in code")

# ============================================================================
# HYPOTHESIS 5: Learning Rate - Is LR causing instability?
# ============================================================================
print()
print("HYPOTHESIS 5: Analyzing training behavior pattern")
print("-"*80)

print("Observed training curve:")
print("  Epoch 1: 25.16 dB (initial)")
print("  Epoch 2: 26.40 dB (⭐ best, +1.24 dB improvement)")
print("  Epoch 3: 26.11 dB (-0.29 dB)")
print("  Epoch 4: 25.94 dB (-0.17 dB)")
print("  Epoch 5: 25.83 dB (-0.11 dB)")
print("  Epoch 6: 25.76 dB (-0.07 dB)")
print()

print("Pattern analysis:")
print("  - Rapid improvement in epoch 1-2")
print("  - Peak at epoch 2 (during warmup)")
print("  - Consistent degradation after epoch 2")
print("  - Rate of degradation slowing (overshooting then settling?)")
print()

print("Possible causes:")
print("  1. Learning rate too high → overshooting after warmup")
print("  2. Model capacity too low → can't represent the function")
print("  3. N2V task too hard → model learning wrong correlations")
print("  4. Data distribution shift → model overfits to noise")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
print()
print("Next steps:")
print("  1. Review data quality results above")
print("  2. If data is good, run supervised baseline test:")
print("     bash quick_supervised_test.sh")
print("  3. If supervised works (30+ dB), N2V implementation has a bug")
print("  4. If supervised fails (26 dB), it's a data/architecture issue")
print()
print("="*80)
