#!/usr/bin/env python3
"""
Validate OCT training data for corruption before training.
Scans the dataset and reports any problematic samples.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))

import torch
import argparse
from torch.utils.data import DataLoader

print("=" * 80)
print("OCT TRAINING DATA VALIDATION")
print("=" * 80)
print()

def validate_batch(batch_idx, noisy, clean, batch_size):
    """
    Validate a single batch and return list of issues found.

    Returns:
        list of (check_name, severity, message) tuples
    """
    issues = []

    # Check 1: NaN/Inf in noisy
    if torch.isnan(noisy).any():
        issues.append(("NaN", "CRITICAL", f"Batch {batch_idx}: NaN detected in noisy input"))
    if torch.isinf(noisy).any():
        issues.append(("Inf", "CRITICAL", f"Batch {batch_idx}: Inf detected in noisy input"))

    # Check 2: NaN/Inf in clean
    if torch.isnan(clean).any():
        issues.append(("NaN", "CRITICAL", f"Batch {batch_idx}: NaN detected in clean input"))
    if torch.isinf(clean).any():
        issues.append(("Inf", "CRITICAL", f"Batch {batch_idx}: Inf detected in clean input"))

    # Check 3: Value range [0, 1] with small margin
    noisy_min, noisy_max = noisy.min().item(), noisy.max().item()
    if noisy_min < -0.1 or noisy_max > 1.1:
        issues.append(("Range", "ERROR", f"Batch {batch_idx}: Noisy out of range [{noisy_min:.3f}, {noisy_max:.3f}]"))

    clean_min, clean_max = clean.min().item(), clean.max().item()
    if clean_min < -0.1 or clean_max > 1.1:
        issues.append(("Range", "ERROR", f"Batch {batch_idx}: Clean out of range [{clean_min:.3f}, {clean_max:.3f}]"))

    # Check 4: Variance (detect constant images)
    noisy_std = noisy.std().item()
    if noisy_std < 1e-6:
        issues.append(("Variance", "ERROR", f"Batch {batch_idx}: Noisy has no variance (std={noisy_std:.2e})"))

    clean_std = clean.std().item()
    if clean_std < 1e-6:
        issues.append(("Variance", "ERROR", f"Batch {batch_idx}: Clean has no variance (std={clean_std:.2e})"))

    # Check 5: Noise level sanity
    residual_mag = (noisy - clean).abs().mean().item()
    if residual_mag > 0.5:
        issues.append(("Noise", "WARNING", f"Batch {batch_idx}: Unrealistic noise level {residual_mag:.3f}"))

    # Check 6: Extreme values within batch (per-sample check)
    for i in range(noisy.size(0)):
        sample_noisy = noisy[i]
        sample_clean = clean[i]

        # Check individual sample statistics
        sample_noisy_std = sample_noisy.std().item()
        if sample_noisy_std < 1e-6:
            issues.append(("Sample", "WARNING", f"Batch {batch_idx}, Sample {i}: Constant noisy image"))

        sample_clean_std = sample_clean.std().item()
        if sample_clean_std < 1e-6:
            issues.append(("Sample", "WARNING", f"Batch {batch_idx}, Sample {i}: Constant clean image"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate OCT training data")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--max_batches", type=int, default=100, help="Max batches to check (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Print all checks, not just issues")
    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Training directory: {args.train_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max batches: {args.max_batches if args.max_batches > 0 else 'All'}")
    print()

    # Import dataset
    try:
        from scripts.train_hybrid_nsnd_multitask import OCTPairedDataset
        print("✅ Successfully imported OCTPairedDataset")
    except ImportError as e:
        print(f"❌ Failed to import dataset: {e}")
        print("   Make sure you're running from the OCT project root")
        sys.exit(1)

    # Load dataset
    try:
        dataset = OCTPairedDataset(args.train_dir)
        print(f"✅ Loaded dataset: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create dataloader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"✅ Created DataLoader: {len(loader)} batches")
    print()

    # Validation loop
    print("=" * 80)
    print("VALIDATING DATA...")
    print("=" * 80)
    print()

    total_batches = 0
    total_issues = 0
    critical_count = 0
    error_count = 0
    warning_count = 0

    issue_summary = {
        "NaN": 0,
        "Inf": 0,
        "Range": 0,
        "Variance": 0,
        "Noise": 0,
        "Sample": 0,
    }

    max_batches = args.max_batches if args.max_batches > 0 else len(loader)

    try:
        for batch_idx, batch in enumerate(loader, start=1):
            if batch_idx > max_batches:
                break

            # Unpack batch
            if isinstance(batch, (list, tuple)):
                # Handle different batch formats
                if len(batch) >= 2:
                    noisy = batch[0]
                    clean = batch[1]
                else:
                    print(f"⚠ Warning: Batch {batch_idx} has unexpected format")
                    continue
            else:
                print(f"⚠ Warning: Batch {batch_idx} is not a list/tuple")
                continue

            # Validate
            issues = validate_batch(batch_idx, noisy, clean, args.batch_size)

            if issues:
                total_issues += len(issues)
                for check_name, severity, message in issues:
                    issue_summary[check_name] += 1

                    if severity == "CRITICAL":
                        critical_count += 1
                        print(f"❌ CRITICAL: {message}")
                    elif severity == "ERROR":
                        error_count += 1
                        print(f"⚠️  ERROR: {message}")
                    else:
                        warning_count += 1
                        if args.verbose:
                            print(f"⚠  WARNING: {message}")
            elif args.verbose:
                print(f"✅ Batch {batch_idx}: OK")

            total_batches += 1

            # Progress indicator
            if batch_idx % 20 == 0:
                print(f"   ... checked {batch_idx}/{max_batches} batches ...")

    except KeyboardInterrupt:
        print()
        print("⚠ Validation interrupted by user")
        print()
    except Exception as e:
        print()
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Batches checked: {total_batches}")
    print(f"Total issues found: {total_issues}")
    print()

    if total_issues > 0:
        print("Issue breakdown:")
        print(f"  ❌ CRITICAL (NaN/Inf): {critical_count}")
        print(f"  ⚠️  ERRORS (Range/Variance): {error_count}")
        print(f"  ⚠  WARNINGS: {warning_count}")
        print()

        print("Issue types:")
        for issue_type, count in issue_summary.items():
            if count > 0:
                print(f"  {issue_type}: {count}")
        print()

        if critical_count > 0:
            print("=" * 80)
            print("❌ CRITICAL ISSUES FOUND!")
            print("=" * 80)
            print("Your dataset contains NaN/Inf values that will cause training to fail.")
            print()
            print("Recommended actions:")
            print("  1. Regenerate the corrupted samples")
            print("  2. Check data preprocessing pipeline")
            print("  3. Remove corrupted files from training directory")
            print()
            sys.exit(1)

        elif error_count > 0:
            print("=" * 80)
            print("⚠️  ERRORS FOUND")
            print("=" * 80)
            print("Your dataset has samples with invalid ranges or no variance.")
            print()
            print("Recommended actions:")
            print("  1. Check data normalization (should be in [0, 1])")
            print("  2. Remove constant/empty images")
            print("  3. Verify preprocessing pipeline")
            print()
            sys.exit(1)

        else:
            print("=" * 80)
            print("⚠  WARNINGS ONLY")
            print("=" * 80)
            print("Dataset has some unusual samples but should be trainable.")
            print("Training will automatically skip problematic batches.")
            print()
    else:
        print("=" * 80)
        print("✅ ALL CHECKS PASSED!")
        print("=" * 80)
        print("Dataset looks healthy - no issues detected.")
        print()


if __name__ == "__main__":
    main()
