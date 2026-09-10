#!/usr/bin/env python3
"""
Comprehensive Training Diagnostics for CASA+N2V
Checks data quality, model initialization, and N2V masking
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

print("="*80)
print("CASA+N2V TRAINING DIAGNOSTICS")
print("="*80)

# ============================================================================
# 1. CHECK DATA QUALITY
# ============================================================================
print("\n[1/5] Checking Training Data Quality...")

train_pairs_file = "train_pairs_universal.txt"
val_pairs_file = "val_pairs_universal.txt"

if not os.path.exists(train_pairs_file):
    print(f"❌ ERROR: {train_pairs_file} not found!")
    sys.exit(1)

# Read first 5 pairs
train_pairs = []
with open(train_pairs_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            if len(parts) == 2:
                train_pairs.append((parts[0].strip(), parts[1].strip()))

if len(train_pairs) == 0:
    print(f"❌ ERROR: No valid pairs found in {train_pairs_file}")
    sys.exit(1)

print(f"✓ Found {len(train_pairs)} pairs to check")

# Check each pair
data_issues = []
for i, (noisy_path, clean_path) in enumerate(train_pairs, 1):
    print(f"\n  Pair {i}:")
    print(f"    Noisy: {noisy_path}")
    print(f"    Clean: {clean_path}")

    # Check files exist
    if not os.path.exists(noisy_path):
        print(f"    ❌ Noisy file not found!")
        data_issues.append(f"Pair {i}: noisy file missing")
        continue
    if not os.path.exists(clean_path):
        print(f"    ❌ Clean file not found!")
        data_issues.append(f"Pair {i}: clean file missing")
        continue

    # Load images
    try:
        noisy = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0
    except Exception as e:
        print(f"    ❌ Failed to load images: {e}")
        data_issues.append(f"Pair {i}: loading failed")
        continue

    # Check shapes
    if noisy.shape != clean.shape:
        print(f"    ❌ Shape mismatch: noisy={noisy.shape}, clean={clean.shape}")
        data_issues.append(f"Pair {i}: shape mismatch")
        continue

    # Check if identical
    if np.array_equal(noisy, clean):
        print(f"    ❌ CRITICAL: Noisy and clean are IDENTICAL!")
        data_issues.append(f"Pair {i}: identical images")
        continue

    # Compute metrics
    input_psnr = psnr(clean, noisy, data_range=1.0)
    input_ssim = ssim(clean, noisy, data_range=1.0)
    mean_diff = np.abs(noisy - clean).mean()

    print(f"    Shape: {noisy.shape}")
    print(f"    Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"    Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"    Mean difference: {mean_diff:.4f}")
    print(f"    Input PSNR: {input_psnr:.2f} dB")
    print(f"    Input SSIM: {input_ssim:.4f}")

    # Validate
    if input_psnr > 35:
        print(f"    ⚠️  WARNING: Images too similar (PSNR={input_psnr:.1f} dB > 35 dB)")
        data_issues.append(f"Pair {i}: too similar (PSNR={input_psnr:.1f})")
    elif input_psnr < 18:
        print(f"    ⚠️  WARNING: Images too different (PSNR={input_psnr:.1f} dB < 18 dB)")
        data_issues.append(f"Pair {i}: too different (PSNR={input_psnr:.1f})")
    elif mean_diff < 0.01:
        print(f"    ⚠️  WARNING: Very small difference (mean={mean_diff:.4f})")
        data_issues.append(f"Pair {i}: minimal difference")
    else:
        print(f"    ✓ Data looks good (PSNR={input_psnr:.1f} dB)")

# Summary
print("\n" + "="*80)
if data_issues:
    print("❌ DATA ISSUES DETECTED:")
    for issue in data_issues:
        print(f"  - {issue}")
    print("\n⚠️  These issues will prevent proper training!")
else:
    print("✓ All checked pairs have good data quality")
print("="*80)

# ============================================================================
# 2. CHECK META-LEARNING CHECKPOINT
# ============================================================================
print("\n[2/5] Checking Meta-Learning Checkpoint...")

meta_checkpoint = "checkpoints/casa_noise2void/meta_adapter.pth"
if os.path.exists(meta_checkpoint):
    size_mb = os.path.getsize(meta_checkpoint) / (1024 * 1024)
    print(f"✓ Meta checkpoint found: {meta_checkpoint}")
    print(f"  Size: {size_mb:.2f} MB")

    # Try loading
    try:
        state = torch.load(meta_checkpoint, map_location='cpu')
        if isinstance(state, dict):
            print(f"  Contains {len(state)} parameter tensors")
            total_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
            print(f"  Total parameters: {total_params:,}")
        else:
            print(f"  Type: {type(state)}")

        if size_mb < 0.1:
            print("  ⚠️  WARNING: Checkpoint seems too small!")
        else:
            print("  ✓ Meta-learning checkpoint looks valid")
    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
else:
    print(f"⚠️  Meta checkpoint not found: {meta_checkpoint}")
    print("   This is expected if you haven't run meta-learning yet")

# ============================================================================
# 3. TEST N2V MASKING
# ============================================================================
print("\n[3/5] Testing Noise2Void Masking...")

try:
    from adaptive_oct_denoise import apply_blind_spot_mask

    # Create test image
    test_img = torch.rand(1, 1, 64, 64)
    mask_ratio = 0.12
    box_size = 5

    print(f"  Test image: {test_img.shape}")
    print(f"  Mask ratio: {mask_ratio} ({mask_ratio*100:.0f}%)")
    print(f"  Box size: {box_size}x{box_size}")

    # Apply masking
    masked_img, mask_coords = apply_blind_spot_mask(test_img, mask_ratio=mask_ratio, box_size=box_size)

    # Count masked pixels
    batch_idx, channel_idx, y_coords, x_coords = mask_coords
    num_masked = len(batch_idx)
    total_pixels = test_img.shape[2] * test_img.shape[3]
    actual_ratio = num_masked / total_pixels

    print(f"  Total pixels: {total_pixels}")
    print(f"  Masked pixels: {num_masked}")
    print(f"  Actual ratio: {actual_ratio:.3f} ({actual_ratio*100:.1f}%)")

    # Check if masked values are different
    original_vals = test_img[batch_idx, channel_idx, y_coords, x_coords]
    masked_vals = masked_img[batch_idx, channel_idx, y_coords, x_coords]
    num_changed = (original_vals != masked_vals).sum().item()

    print(f"  Changed values: {num_changed}/{num_masked}")

    if num_changed == 0:
        print("  ❌ ERROR: No pixels were actually masked!")
    elif abs(actual_ratio - mask_ratio) > 0.05:
        print(f"  ⚠️  WARNING: Actual ratio ({actual_ratio:.1%}) differs from target ({mask_ratio:.1%})")
    else:
        print("  ✓ N2V masking works correctly")

except Exception as e:
    print(f"  ❌ Failed to test masking: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. CHECK MODEL INITIALIZATION
# ============================================================================
print("\n[4/5] Checking Model Initialization...")

try:
    from adaptive_oct_denoise import build_model

    model = build_model(
        base_channels=48,
        residual_mode=False,
        adapter_type='casa',
        backbone_type='noise2void'
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    test_input = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        output = model(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")

    if output.shape != test_input.shape:
        print(f"  ❌ ERROR: Output shape mismatch!")
    elif torch.allclose(output, test_input, atol=1e-3):
        print(f"  ⚠️  WARNING: Output is too similar to input (identity function)")
    else:
        print("  ✓ Model initialization looks good")

except Exception as e:
    print(f"  ❌ Failed to initialize model: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. VALIDATE TRAINING CONFIGURATION
# ============================================================================
print("\n[5/5] Validating Training Configuration...")

config_issues = []

# Check recommended settings
print("  Checking hyperparameters...")

recommended = {
    'base_channels': 48,
    'n2v_mask_ratio': 0.12,
    'n2v_box_size': 5,
    'batch_size': 4,
    'finetune_lr_adapter': 5e-4,
    'finetune_lr_backbone': 1e-4,
    'warmup_epochs': 8,
    'num_meta_epochs': 15,
}

print("\n  Recommended configuration:")
for key, val in recommended.items():
    print(f"    --{key} {val}")

# ============================================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

critical_issues = []
warnings = []

if data_issues:
    critical_issues.append("Data quality issues detected")
if not os.path.exists(meta_checkpoint):
    warnings.append("No meta-learning checkpoint found")

if critical_issues:
    print("\n❌ CRITICAL ISSUES (must fix before training):")
    for issue in critical_issues:
        print(f"  - {issue}")
    print("\n🔧 RECOMMENDED ACTIONS:")
    if "Data quality" in str(critical_issues):
        print("  1. Verify your train_pairs_universal.txt file")
        print("  2. Check that noisy images are actually noisy (PSNR 20-25 dB)")
        print("  3. Ensure clean and noisy images are different files")
        print("  4. Re-generate pairs if needed")
elif warnings:
    print("\n⚠️  WARNINGS (non-critical):")
    for warning in warnings:
        print(f"  - {warning}")
    print("\n💡 You can proceed with training")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\n🚀 Ready to train with:")
    print("\n    python -u adaptive_oct_denoise.py \\")
    print("      --clean_root meta_clean/ \\")
    print("      --paired_list train_pairs_universal.txt \\")
    print("      --val_paired_list val_pairs_universal.txt \\")
    print("      --output_dir checkpoints/casa_noise2void \\")
    print("      --backbone noise2void \\")
    print("      --adapter casa \\")
    print("      --base_channels 48 \\")
    print("      --num_meta_epochs 15 \\")
    print("      --finetune_epochs 100 \\")
    print("      --batch_size 4 \\")
    print("      --ema \\")
    print("      --amp \\")
    print("      --seed 42")

print("="*80 + "\n")
