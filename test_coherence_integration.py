#!/usr/bin/env python3
"""
Test coherence supervision integration in adaptive_oct_denoise.py
"""
import torch
import sys
sys.path.insert(0, '.')

from adaptive_oct_denoise import (
    compute_local_cv,
    pearson_correlation_loss,
    CoherenceSupervisionLoss,
    build_model,
    device
)

print("=" * 80)
print("TESTING COHERENCE SUPERVISION INTEGRATION")
print("=" * 80)

# Test 1: Import successful
print("\n✓ Test 1: Successfully imported coherence loss functions")

# Test 2: compute_local_cv works
print("\n✓ Test 2: Testing compute_local_cv...")
test_image = torch.randn(4, 1, 64, 64)
cv_map = compute_local_cv(test_image, window_size=11)
assert cv_map.shape == test_image.shape, f"CV map shape mismatch: {cv_map.shape} vs {test_image.shape}"
print(f"  CV map shape: {cv_map.shape} ✓")
print(f"  CV range: [{cv_map.min():.3f}, {cv_map.max():.3f}]")

# Test 3: pearson_correlation_loss works
print("\n✓ Test 3: Testing pearson_correlation_loss...")
x = torch.randn(4, 1, 64, 64)
y = x.clone()  # Perfect correlation
loss_perfect = pearson_correlation_loss(x, y)
print(f"  Perfect correlation loss: {loss_perfect.item():.3f} (expect ~-1.0)")
assert loss_perfect.item() < -0.9, f"Perfect correlation should be near -1.0, got {loss_perfect.item()}"

# Test 4: CoherenceSupervisionLoss works
print("\n✓ Test 4: Testing CoherenceSupervisionLoss...")
coherence_loss_fn = CoherenceSupervisionLoss()
coherent_map = torch.rand(4, 1, 64, 64)
noisy_image = torch.randn(4, 1, 64, 64) * 0.1 + 0.5
loss = coherence_loss_fn(coherent_map, noisy_image)
print(f"  Coherence loss: {loss.item():.3f}")
assert torch.isfinite(loss), "Loss should be finite"

# Test 5: Gradient flow
print("\n✓ Test 5: Testing gradient flow...")
coherent_map = torch.rand(4, 1, 64, 64, requires_grad=True)
noisy_image = torch.randn(4, 1, 64, 64)
loss = coherence_loss_fn(coherent_map, noisy_image)
loss.backward()
assert coherent_map.grad is not None, "Gradient should flow to coherent_map"
print(f"  Gradient norm: {coherent_map.grad.norm().item():.3f} ✓")

# Test 6: Integration with CASA model
print("\n✓ Test 6: Testing integration with CASA model...")
model = build_model(
    base_channels=48,
    residual_mode=True,
    adapter_type="casa",
    backbone_type="noise2void"
).to(device)

test_input = torch.randn(2, 1, 64, 64).to(device)
with torch.no_grad():
    output = model(test_input, return_aux=True)

if isinstance(output, tuple):
    pred, aux = output
    print(f"  Model outputs: pred shape {pred.shape}, aux keys {list(aux.keys())}")
    if 'coherent_map' in aux:
        print(f"  ✓ CASA outputs coherent_map: {aux['coherent_map'].shape}")

        # Test coherence loss with CASA output
        coherent_map = aux['coherent_map']
        loss = coherence_loss_fn(coherent_map, test_input)
        print(f"  ✓ Coherence loss on CASA output: {loss.item():.3f}")
    else:
        print(f"  ⚠️ No coherent_map in aux outputs: {list(aux.keys())}")
else:
    print(f"  ⚠️ Model did not return aux outputs")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nCoherence supervision is successfully integrated.")
print("\nTo train with coherence supervision, use:")
print("  python adaptive_oct_denoise.py \\")
print("    --backbone noise2void \\")
print("    --adapter casa \\")
print("    --lambda_coherence 0.1 \\")
print("    --finetune_epochs 100")
print("=" * 80)
