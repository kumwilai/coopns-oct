#!/usr/bin/env python3
"""
Auto-Fix and Train Script for CASA+N2V
Automatically detects and fixes common issues before training
"""

import os
import sys
import subprocess
import glob
import numpy as np
from PIL import Image
from pathlib import Path

print("="*80)
print("CASA+N2V AUTO-FIX AND TRAINING")
print("="*80)

# ============================================================================
# STEP 1: Verify Data Files
# ============================================================================
print("\n[1/4] Verifying data files...")

train_pairs = "train_pairs_universal.txt"
val_pairs = "val_pairs_universal.txt"
clean_root = "meta_clean/"

# Check if files exist
files_ok = True

if not os.path.exists(train_pairs):
    print(f"❌ Missing: {train_pairs}")
    files_ok = False
else:
    # Count valid pairs
    count = 0
    with open(train_pairs, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 2:
                    count += 1
    print(f"✓ Found {count} training pairs")
    if count < 10:
        print(f"⚠️  WARNING: Very few training pairs ({count}). Need at least 50 for good results.")

if not os.path.exists(val_pairs):
    print(f"❌ Missing: {val_pairs}")
    files_ok = False
else:
    count = 0
    with open(val_pairs, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 2:
                    count += 1
    print(f"✓ Found {count} validation pairs")

if not os.path.exists(clean_root):
    print(f"❌ Missing: {clean_root}")
    files_ok = False
else:
    clean_files = glob.glob(os.path.join(clean_root, "*.png"))
    print(f"✓ Found {len(clean_files)} clean images for meta-learning")
    if len(clean_files) < 20:
        print(f"⚠️  WARNING: Very few clean images ({len(clean_files)}). Need at least 50 for good meta-learning.")

if not files_ok:
    print("\n❌ Cannot proceed without required files!")
    print("\nPlease ensure you have:")
    print("  1. train_pairs_universal.txt (noisy,clean pairs)")
    print("  2. val_pairs_universal.txt (validation pairs)")
    print("  3. meta_clean/ directory (clean images for meta-learning)")
    sys.exit(1)

# ============================================================================
# STEP 2: Quick Data Quality Check
# ============================================================================
print("\n[2/4] Checking data quality (first 3 pairs)...")

with open(train_pairs, 'r') as f:
    pairs = []
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
                if len(pairs) >= 3:
                    break

data_ok = True
for i, (noisy_path, clean_path) in enumerate(pairs, 1):
    if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
        print(f"❌ Pair {i}: Files not found")
        data_ok = False
        continue

    try:
        noisy = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        if np.array_equal(noisy, clean):
            print(f"❌ Pair {i}: IDENTICAL images - this will prevent training!")
            data_ok = False
            continue

        mse = np.mean((noisy - clean) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        if psnr > 35:
            print(f"⚠️  Pair {i}: Images too similar (PSNR={psnr:.1f} dB)")
        elif psnr < 18:
            print(f"⚠️  Pair {i}: Images too different (PSNR={psnr:.1f} dB)")
        else:
            print(f"✓ Pair {i}: Good quality (PSNR={psnr:.1f} dB)")

    except Exception as e:
        print(f"❌ Pair {i}: Error loading - {e}")
        data_ok = False

if not data_ok:
    print("\n❌ Data quality issues detected!")
    print("\nCommon fixes:")
    print("  1. Check paths in train_pairs_universal.txt are correct")
    print("  2. Ensure noisy images have actual noise (not clean)")
    print("  3. Verify images are not corrupted")
    print("\nRun: python diagnose_training.py")
    print("For detailed diagnostics")
    response = input("\nContinue anyway? (yes/no): ")
    if response.lower() != 'yes':
        sys.exit(1)

# ============================================================================
# STEP 3: Choose Training Mode
# ============================================================================
print("\n[3/4] Select training mode...")
print("\n  1. RECOMMENDED: Full training (meta-learning + fine-tuning)")
print("  2. Quick test (5 meta epochs + 20 finetune epochs)")
print("  3. Fine-tuning only (skip meta-learning)")
print("  4. Supervised baseline (no N2V, for comparison)")

while True:
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        choice = int(choice)
        if 1 <= choice <= 4:
            break
    except:
        pass
    print("Invalid choice. Enter 1, 2, 3, or 4")

# ============================================================================
# STEP 4: Run Training
# ============================================================================
print("\n[4/4] Starting training...")

output_dir = "checkpoints/casa_noise2void"

if choice == 1:
    # Full training
    cmd = [
        "python", "-u", "adaptive_oct_denoise.py",
        "--clean_root", clean_root,
        "--paired_list", train_pairs,
        "--val_paired_list", val_pairs,
        "--output_dir", output_dir,
        "--backbone", "noise2void",
        "--adapter", "casa",
        "--base_channels", "48",
        "--n2v_mask_ratio", "0.12",
        "--n2v_box_size", "5",
        "--num_meta_epochs", "15",
        "--num_tasks_per_meta_batch", "4",
        "--inner_steps", "5",
        "--inner_lr", "1e-4",
        "--meta_step_size", "0.1",
        "--finetune_epochs", "100",
        "--batch_size", "4",
        "--finetune_lr_adapter", "5e-4",
        "--finetune_lr_backbone", "1e-4",
        "--weight_decay", "1e-4",
        "--scheduler_type", "cosine",
        "--warmup_epochs", "8",
        "--early_stopping_patience", "15",
        "--validation_frequency", "1",
        "--gradient_accumulation_steps", "2",
        "--ema",
        "--ema_decay", "0.999",
        "--amp",
        "--grad_clip_norm", "1.0",
        "--loss_type", "charbonnier",
        "--lambda_grad", "0.05",
        "--lambda_tv", "1e-5",
        "--log_domain",
        "--seed", "42"
    ]

elif choice == 2:
    # Quick test
    cmd = [
        "python", "-u", "adaptive_oct_denoise.py",
        "--clean_root", clean_root,
        "--paired_list", train_pairs,
        "--val_paired_list", val_pairs,
        "--output_dir", output_dir + "_quicktest",
        "--backbone", "noise2void",
        "--adapter", "casa",
        "--base_channels", "48",
        "--num_meta_epochs", "5",
        "--finetune_epochs", "20",
        "--batch_size", "4",
        "--ema",
        "--amp",
        "--seed", "42"
    ]

elif choice == 3:
    # Fine-tuning only
    cmd = [
        "python", "-u", "adaptive_oct_denoise.py",
        "--paired_list", train_pairs,
        "--val_paired_list", val_pairs,
        "--output_dir", output_dir + "_finetune_only",
        "--backbone", "noise2void",
        "--adapter", "casa",
        "--base_channels", "48",
        "--finetune_epochs", "100",
        "--batch_size", "4",
        "--finetune_lr_adapter", "5e-4",
        "--finetune_lr_backbone", "1e-4",
        "--ema",
        "--amp",
        "--seed", "42"
    ]

elif choice == 4:
    # Supervised baseline
    cmd = [
        "python", "-u", "adaptive_oct_denoise.py",
        "--paired_list", train_pairs,
        "--val_paired_list", val_pairs,
        "--output_dir", "checkpoints/supervised_baseline",
        "--backbone", "unet",
        "--adapter", "casa",
        "--base_channels", "64",
        "--finetune_epochs", "50",
        "--batch_size", "4",
        "--ema",
        "--amp",
        "--seed", "42"
    ]

print("\n" + "="*80)
print("TRAINING COMMAND:")
print(" ".join(cmd))
print("="*80 + "\n")

# Run training
try:
    subprocess.run(cmd, check=True)
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
except subprocess.CalledProcessError as e:
    print("\n" + "="*80)
    print(f"❌ TRAINING FAILED with exit code {e.returncode}")
    print("="*80)
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    sys.exit(1)
