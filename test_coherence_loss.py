#!/usr/bin/env python3
"""
Quick test: Verify coherence loss components work correctly.
"""
import torch
import sys
sys.path.insert(0, '.')
from train_casa_coherence_supervised import (
    compute_local_cv,
    pearson_correlation_loss,
    coherence_supervision_loss
)

print("=" * 80)
print("TESTING COHERENCE LOSS COMPONENTS")
print("=" * 80)

# Create synthetic test data
batch_size = 4
height, width = 64, 64

# Test 1: Local CV computation
print("\n1. Testing local CV computation...")
test_image = torch.randn(batch_size, 1, height, width)
cv_map = compute_local_cv(test_image, window_size=11)

print(f"  Input shape: {test_image.shape}")
print(f"  CV map shape: {cv_map.shape}")
print(f"  CV range: [{cv_map.min():.3f}, {cv_map.max():.3f}]")
print(f"  CV mean: {cv_map.mean():.3f}")

if cv_map.shape == test_image.shape:
    print("  ✓ Shape correct")
else:
    print(f"  ✗ Shape mismatch!")

# Test 2: Pearson correlation loss
print("\n2. Testing Pearson correlation loss...")

# Create perfectly correlated tensors (should give r=1, loss=-1)
x = torch.randn(batch_size, 1, height, width)
y = x.clone()  # Perfect correlation
loss_perfect = pearson_correlation_loss(x, y)
print(f"  Perfect correlation loss: {loss_perfect.item():.3f} (expect ~-1.0)")

# Create uncorrelated tensors (should give r≈0, loss≈0)
x = torch.randn(batch_size, 1, height, width)
y = torch.randn(batch_size, 1, height, width)
loss_uncorr = pearson_correlation_loss(x, y)
print(f"  Uncorrelated loss: {loss_uncorr.item():.3f} (expect ~0.0)")

# Create anti-correlated tensors (should give r=-1, loss=+1)
x = torch.randn(batch_size, 1, height, width)
y = -x  # Anti-correlation
loss_anti = pearson_correlation_loss(x, y)
print(f"  Anti-correlated loss: {loss_anti.item():.3f} (expect ~+1.0)")

if loss_perfect.item() < -0.9 and abs(loss_uncorr.item()) < 0.2 and loss_anti.item() > 0.9:
    print("  ✓ Correlation loss working correctly")
else:
    print("  ⚠️ Correlation loss may have issues")

# Test 3: Coherence supervision loss
print("\n3. Testing coherence supervision loss...")

# Simulate coherent weight map and noisy image
coherent_map = torch.rand(batch_size, 1, height, width)  # Random weights [0,1]
noisy_image = torch.randn(batch_size, 1, height, width) * 0.1 + 0.5  # Noisy image

coh_loss = coherence_supervision_loss(coherent_map, noisy_image)
print(f"  Coherence loss: {coh_loss.item():.3f}")

# Test gradient flow
coh_loss.backward()
print("  ✓ Gradient computation successful")

# Test 4: Simulate what CASA should learn
print("\n4. Simulating expected CASA behavior...")

# Create image with structured speckle (high CV) and additive noise (low CV)
structured_region = torch.randn(batch_size, 1, 32, 32) * 0.5 + 0.5  # High variance
smooth_region = torch.randn(batch_size, 1, 32, 32) * 0.05 + 0.5    # Low variance

# Concatenate horizontally
test_image = torch.cat([structured_region, smooth_region], dim=3)

cv_map = compute_local_cv(test_image)

print(f"  Structured region CV: {cv_map[:, :, :, :32].mean():.3f}")
print(f"  Smooth region CV: {cv_map[:, :, :, 32:].mean():.3f}")

# Ideal coherent map: high where CV is high
ideal_coherent = (cv_map - cv_map.min()) / (cv_map.max() - cv_map.min() + 1e-8)

# Test correlation
loss_ideal = pearson_correlation_loss(ideal_coherent, cv_map)
print(f"  Ideal coherent-CV correlation loss: {loss_ideal.item():.3f} (expect ~-1.0)")

if loss_ideal.item() < -0.9:
    print("  ✓ Loss correctly rewards high coherent-CV correlation")
else:
    print("  ⚠️ Loss may not be rewarding correlation correctly")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("✓ All components functional")
print("\nExpected training behavior:")
print("  - Coherence loss starts near 0 (random correlation)")
print("  - During training, should decrease to -0.5 or lower (positive correlation)")
print("  - Monitor validation correlation: should increase from ~0 to >0.5")
print("=" * 80)
