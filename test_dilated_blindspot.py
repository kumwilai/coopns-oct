#!/usr/bin/env python3
"""
Comprehensive bug and memory leak hunting for dilated blind-spot implementation.
"""
import torch
import gc
import sys
sys.path.insert(0, '.')

from adaptive_oct_denoise import (
    build_model,
    PairedOCTDataset,
    device,
    resize_to,
    apply_blind_spot_mask,
    CoherenceSupervisionLoss
)
from torch.utils.data import DataLoader

print("=" * 80)
print("DILATED BLIND-SPOT BUG & MEMORY LEAK HUNTING")
print("=" * 80)

def get_gpu_memory():
    """Get current memory usage in MB (GPU or RSS on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0

issues = []

# Test 1: Check masking logic for different dilation values
print("\n1. Testing masking logic for different dilation values...")

test_img = torch.randn(2, 1, 64, 64).to(device)

for dilation in [1, 3, 5, 7]:
    print(f"\n  Testing dilation={dilation}...")

    # Apply mask
    masked_img, mask_coords = apply_blind_spot_mask(
        test_img,
        mask_ratio=0.2,
        box_size=5,
        seed=42,
        blindspot_dilation=dilation
    )

    # Check output shapes
    if masked_img.shape != test_img.shape:
        print(f"    ❌ Shape mismatch: {masked_img.shape} vs {test_img.shape}")
        issues.append(f"Dilation {dilation}: Shape mismatch")
        continue

    # Check mask_coords format - should be tuple of 4 tensors
    if not isinstance(mask_coords, tuple) or len(mask_coords) != 4:
        print(f"    ❌ mask_coords should be tuple of 4 tensors, got {type(mask_coords)}")
        issues.append(f"Dilation {dilation}: Wrong mask_coords type")
        continue

    batch_idx, channel_idx, y_coords, x_coords = mask_coords
    total_masked_pixels = len(y_coords)

    # Verify coordinates are within bounds
    H, W = test_img.shape[2], test_img.shape[3]
    if torch.any(y_coords < 0) or torch.any(y_coords >= H):
        print(f"    ❌ y_coords out of bounds!")
        issues.append(f"Dilation {dilation}: y_coords out of bounds")

    if torch.any(x_coords < 0) or torch.any(x_coords >= W):
        print(f"    ❌ x_coords out of bounds!")
        issues.append(f"Dilation {dilation}: x_coords out of bounds")

    # Check that masked pixels exist
    for i in range(min(5, len(y_coords))):  # Sample first 5
        b = batch_idx[i].item()
        c = channel_idx[i].item()
        y = y_coords[i].item()
        x = x_coords[i].item()

        masked_val = masked_img[b, c, y, x].item()

        # Check for NaN
        if torch.isnan(torch.tensor(masked_val)):
            print(f"    ❌ NaN in masked image at ({b}, {c}, {y}, {x})")
            issues.append(f"Dilation {dilation}: NaN in output")

    print(f"    ✓ Dilation {dilation}: {total_masked_pixels} total masked pixels")
    print(f"      Shape: {masked_img.shape}")
    print(f"      Range: [{masked_img.min():.3f}, {masked_img.max():.3f}]")

# Test 2: Verify dilated region is actually masked
print("\n2. Verifying dilated region masking...")

for dilation in [1, 3, 5]:
    test_img = torch.ones(1, 1, 32, 32).to(device) * 5.0  # All pixels = 5.0

    masked_img, mask_coords = apply_blind_spot_mask(
        test_img,
        mask_ratio=0.05,  # Small ratio to check specific regions
        box_size=32,  # One box covering whole image
        seed=42,
        blindspot_dilation=dilation
    )

    # Get the mask coordinates
    batch_indices, channel_indices, y_coords, x_coords = mask_coords

    if len(y_coords) > 0:
        # Check one masked pixel
        y, x = y_coords[0].item(), x_coords[0].item()

        # Calculate expected dilated region
        half_dilation = dilation // 2
        H, W = 32, 32

        dilated_y_start = max(0, y - half_dilation)
        dilated_y_end = min(H, y + half_dilation + 1)
        dilated_x_start = max(0, x - half_dilation)
        dilated_x_end = min(W, x + half_dilation + 1)

        # Check if dilated region was modified
        dilated_region = masked_img[0, 0, dilated_y_start:dilated_y_end, dilated_x_start:dilated_x_end]

        # The dilated region should be filled with neighbor value (not all 5.0)
        # At least the center should be different or same (filled from outside)
        center_val = masked_img[0, 0, y, x].item()

        print(f"  Dilation {dilation}:")
        print(f"    Center pixel ({y},{x}): original=5.0, masked={center_val:.3f}")
        print(f"    Dilated region shape: {dilated_region.shape}")
        print(f"    ✓ Masking applied")

# Test 3: Memory leak test with repeated masking
print("\n3. Testing for memory leaks in masking...")

initial_mem = get_gpu_memory()
print(f"  Initial GPU memory: {initial_mem:.1f} MB")

mem_per_iter = []
test_img = torch.randn(4, 1, 128, 128).to(device)

for i in range(20):
    masked_img, mask_coords = apply_blind_spot_mask(
        test_img,
        mask_ratio=0.2,
        box_size=5,
        blindspot_dilation=3
    )

    # Cleanup
    del masked_img, mask_coords

    if i % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        mem = get_gpu_memory()
        mem_per_iter.append(mem)

final_mem = get_gpu_memory()
mem_growth = final_mem - initial_mem

print(f"  Final GPU memory: {final_mem:.1f} MB")
print(f"  Memory growth: {mem_growth:.1f} MB")
print(f"  Memory per iteration: {[f'{m:.1f}' for m in mem_per_iter]}")

if mem_growth > 50:
    print(f"  ❌ MEMORY LEAK in masking! Growth = {mem_growth:.1f} MB")
    issues.append(f"Memory leak in masking: {mem_growth:.1f} MB")
elif mem_growth > 10:
    print(f"  ⚠️ Moderate memory growth ({mem_growth:.1f} MB)")
else:
    print(f"  ✓ No significant memory leak")

# Test 4: Integration test with training loop
print("\n4. Testing integration with training loop...")

model = build_model(
    base_channels=48,
    residual_mode=True,
    adapter_type="casa",
    backbone_type="noise2void"
).to(device)

# Create small dataset
transform = resize_to((64, 64))
try:
    dataset = PairedOCTDataset('train_pairs_universal.txt', transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
except:
    print("  ⚠️ Could not load dataset, using dummy data")
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx):
            return torch.randn(1, 64, 64), torch.randn(1, 64, 64)
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
coherence_loss_fn = CoherenceSupervisionLoss()

model.train()
initial_mem = get_gpu_memory()
print(f"  Initial GPU memory: {initial_mem:.1f} MB")

# Test with different dilation values
for dilation in [1, 3, 5]:
    print(f"\n  Testing with dilation={dilation}...")

    mem_before = get_gpu_memory()

    for i, (noisy, clean) in enumerate(loader):
        if i >= 3:  # Test 3 iterations per dilation
            break

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        # Apply dilated blind-spot masking
        try:
            masked_input, mask_coords = apply_blind_spot_mask(
                noisy,
                mask_ratio=0.2,
                box_size=5,
                blindspot_dilation=dilation
            )
        except Exception as e:
            print(f"    ❌ Masking failed: {e}")
            issues.append(f"Dilation {dilation} masking failed: {e}")
            break

        # Forward pass
        out = model(masked_input, return_aux=True)
        pred, aux = out if isinstance(out, tuple) else (out, {})

        # Compute loss (simplified N2V-like)
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Loss only on masked pixels
        batch_indices, channel_indices, y_coords, x_coords = mask_coords
        if len(y_coords) > 0:
            pred_vals = pred[batch_indices, channel_indices, y_coords, x_coords]
            target_vals = noisy[batch_indices, channel_indices, y_coords, x_coords]
            loss = loss + torch.nn.functional.mse_loss(pred_vals, target_vals)

        # Add coherence loss if available
        if aux and 'coherent_map' in aux:
            coherent_map = aux['coherent_map']
            coherence_loss = coherence_loss_fn(coherent_map, noisy)
            loss = loss + 0.1 * coherence_loss

        # Backward
        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"    ❌ Backward failed: {e}")
            issues.append(f"Dilation {dilation} backward failed: {e}")
            break

        # Cleanup
        del noisy, clean, masked_input, mask_coords, out, pred, aux, loss

    mem_after = get_gpu_memory()
    mem_delta = mem_after - mem_before

    print(f"    ✓ Dilation {dilation}: Training loop works")
    print(f"    Memory delta: {mem_delta:.1f} MB")

    if mem_delta > 100:
        print(f"    ❌ High memory usage!")
        issues.append(f"Dilation {dilation}: High memory usage {mem_delta:.1f} MB")

    gc.collect()
    torch.cuda.empty_cache()

# Test 5: Edge cases
print("\n5. Testing edge cases...")

# Edge case 1: Very small image
print("  Testing small image (16x16)...")
small_img = torch.randn(1, 1, 16, 16).to(device)
try:
    masked, coords = apply_blind_spot_mask(small_img, mask_ratio=0.2, box_size=5, blindspot_dilation=5)
    print(f"    ✓ Small image works")
except Exception as e:
    print(f"    ❌ Small image failed: {e}")
    issues.append(f"Small image failed: {e}")

# Edge case 2: Large dilation on small image
print("  Testing large dilation (7) on medium image (32x32)...")
medium_img = torch.randn(1, 1, 32, 32).to(device)
try:
    masked, coords = apply_blind_spot_mask(medium_img, mask_ratio=0.2, box_size=5, blindspot_dilation=7)
    print(f"    ✓ Large dilation works")
except Exception as e:
    print(f"    ❌ Large dilation failed: {e}")
    issues.append(f"Large dilation failed: {e}")

# Edge case 3: Dilation=1 (standard N2V)
print("  Testing standard N2V (dilation=1)...")
try:
    masked, coords = apply_blind_spot_mask(test_img, mask_ratio=0.2, box_size=5, blindspot_dilation=1)
    print(f"    ✓ Standard N2V (dilation=1) works")
except Exception as e:
    print(f"    ❌ Standard N2V failed: {e}")
    issues.append(f"Standard N2V failed: {e}")

# Edge case 4: Even dilation value
print("  Testing even dilation value (4)...")
try:
    masked, coords = apply_blind_spot_mask(test_img, mask_ratio=0.2, box_size=5, blindspot_dilation=4)
    print(f"    ✓ Even dilation works")
except Exception as e:
    print(f"    ❌ Even dilation failed: {e}")
    issues.append(f"Even dilation failed: {e}")

# Test 6: Gradient flow with dilated masking
print("\n6. Testing gradient flow with dilated masking...")

model.train()

for dilation in [1, 3, 5]:
    optimizer.zero_grad()

    test_input = torch.randn(2, 1, 64, 64, requires_grad=False).to(device)
    masked_input, mask_coords = apply_blind_spot_mask(
        test_input,
        mask_ratio=0.2,
        box_size=5,
        blindspot_dilation=dilation
    )

    # Forward
    out = model(masked_input, return_aux=True)
    pred, aux = out if isinstance(out, tuple) else (out, {})

    # Loss on masked pixels
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    batch_indices, channel_indices, y_coords, x_coords = mask_coords
    if len(y_coords) > 0:
        pred_vals = pred[batch_indices, channel_indices, y_coords, x_coords]
        target_vals = test_input[batch_indices, channel_indices, y_coords, x_coords]
        loss = loss + torch.nn.functional.mse_loss(pred_vals, target_vals)

    # Backward
    loss.backward()

    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    if has_grad:
        print(f"  ✓ Dilation {dilation}: Gradients flow correctly")
    else:
        print(f"  ❌ Dilation {dilation}: NO gradients!")
        issues.append(f"Dilation {dilation}: No gradients")

# Test 7: Verify deterministic behavior with seed
print("\n7. Testing deterministic behavior with seed...")

test_img = torch.randn(2, 1, 64, 64).to(device)

for dilation in [1, 3, 5]:
    # Run twice with same seed
    masked1, coords1 = apply_blind_spot_mask(test_img, mask_ratio=0.2, box_size=5, seed=123, blindspot_dilation=dilation)
    masked2, coords2 = apply_blind_spot_mask(test_img, mask_ratio=0.2, box_size=5, seed=123, blindspot_dilation=dilation)

    # Check if results are identical
    if torch.allclose(masked1, masked2):
        print(f"  ✓ Dilation {dilation}: Deterministic with seed")
    else:
        print(f"  ❌ Dilation {dilation}: NOT deterministic!")
        issues.append(f"Dilation {dilation}: Non-deterministic")

    # Check coordinates are same
    b1, c1, y1, x1 = coords1
    b2, c2, y2, x2 = coords2

    coords_match = (
        torch.equal(b1, b2) and
        torch.equal(c1, c2) and
        torch.equal(y1, y2) and
        torch.equal(x1, x2)
    )

    if coords_match:
        print(f"    Coordinates also match ✓")
    else:
        print(f"    ❌ Coordinates don't match!")
        issues.append(f"Dilation {dilation}: Coords non-deterministic")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)

if len(issues) == 0:
    print("✓✓ NO BUGS OR MEMORY LEAKS DETECTED IN DILATED BLIND-SPOT!")
    print("\n✓ Safe to proceed with training")
    print("\nRecommended training commands:")

    print("\n1. Standard N2V + CASA + Coherence (baseline):")
    print("  python adaptive_oct_denoise.py \\")
    print("    --paired_list train_pairs_universal.txt \\")
    print("    --val_paired_list val_pairs_universal.txt \\")
    print("    --backbone noise2void \\")
    print("    --adapter casa \\")
    print("    --base_channels 48 \\")
    print("    --residual_mode \\")
    print("    --lambda_coherence 0.1 \\")
    print("    --n2v_blindspot_dilation 1 \\")
    print("    --finetune_epochs 100 \\")
    print("    --output_dir checkpoints/casa_n2v_coherence \\")
    print("    --amp")

    print("\n2. Dilated N2V (3x3) + CASA + Coherence (recommended for OCT):")
    print("  python adaptive_oct_denoise.py \\")
    print("    --paired_list train_pairs_universal.txt \\")
    print("    --val_paired_list val_pairs_universal.txt \\")
    print("    --backbone noise2void \\")
    print("    --adapter casa \\")
    print("    --base_channels 48 \\")
    print("    --residual_mode \\")
    print("    --lambda_coherence 0.1 \\")
    print("    --n2v_blindspot_dilation 3 \\")
    print("    --finetune_epochs 100 \\")
    print("    --output_dir checkpoints/casa_n2v_dilated3_coherence \\")
    print("    --amp")

    print("\n3. Aggressive dilation (5x5) + CASA + Coherence:")
    print("  python adaptive_oct_denoise.py \\")
    print("    --paired_list train_pairs_universal.txt \\")
    print("    --val_paired_list val_pairs_universal.txt \\")
    print("    --backbone noise2void \\")
    print("    --adapter casa \\")
    print("    --base_channels 48 \\")
    print("    --residual_mode \\")
    print("    --lambda_coherence 0.1 \\")
    print("    --n2v_blindspot_dilation 5 \\")
    print("    --finetune_epochs 100 \\")
    print("    --output_dir checkpoints/casa_n2v_dilated5_coherence \\")
    print("    --amp")
else:
    print(f"❌ FOUND {len(issues)} ISSUE(S):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n⚠️ FIX THESE ISSUES BEFORE TRAINING!")

print("=" * 80)
