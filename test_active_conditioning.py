#!/usr/bin/env python3
"""
Test script for Active Feature Modulation (Conditioning) implementation.

Tests:
1. NoiseConditioner initialization and identity output
2. Swin head with conditioning
3. NAFNet head with conditioning
4. Forward pass with different noise vectors
5. Gradient flow through conditioner
6. Backward compatibility (heads work without conditioning)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("ACTIVE FEATURE MODULATION TEST")
print("=" * 80)
print()

# Test 1: NoiseConditioner Identity Initialization
print("Test 1: NoiseConditioner Identity Initialization")
print("-" * 80)
try:
    from nsnd.models.noise_conditioner import NoiseConditioner

    conditioner = NoiseConditioner(noise_dim=4, feature_channels=32, hidden_dim=16)

    # Check initialization
    fc2_weight_mean = conditioner.fc2.weight.abs().mean().item()
    fc2_bias_mean = conditioner.fc2.bias.mean().item()

    print(f"   fc2.weight mean: {fc2_weight_mean:.6f} (should be ~0.0001)")
    print(f"   fc2.bias mean: {fc2_bias_mean:.2f} (should be ~4.0)")

    assert fc2_weight_mean < 0.001, "❌ fc2.weight not near zero!"
    assert abs(fc2_bias_mean - 4.0) < 0.1, f"❌ fc2.bias not near 4.0! Got {fc2_bias_mean:.2f}"

    # Test identity output
    noise_vec = torch.tensor([[0.7, 0.1, 0.1, 0.1]])  # Example noise vector
    gamma = conditioner(noise_vec)

    print(f"   Output gamma shape: {gamma.shape}")
    print(f"   Output gamma range: [{gamma.min():.3f}, {gamma.max():.3f}]")
    print(f"   Output gamma mean: {gamma.mean():.3f} (should be ~0.98)")

    assert gamma.shape == (1, 32), f"❌ Wrong shape! Got {gamma.shape}"
    assert gamma.min() > 0.0 and gamma.max() <= 1.0, "❌ Gamma not in range (0, 1]!"
    assert gamma.mean() > 0.90, f"❌ Gamma mean {gamma.mean():.3f} not near 1.0 (identity)!"

    print(f"✅ NoiseConditioner initialized with identity output")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 2: Swin Head with Conditioning
print("Test 2: Swin Head with Conditioning")
print("-" * 80)
try:
    from nsnd.models.swin_head import SwinResidualHead

    # Create head WITH conditioning
    head = SwinResidualHead(
        img_size=64,
        in_chans=2,
        embed_dim=32,
        use_conditioning=True,
        conditioner_hidden=16,
    )

    assert hasattr(head, 'conditioner'), "❌ Swin head missing conditioner!"
    assert isinstance(head.conditioner, NoiseConditioner), "❌ Conditioner wrong type!"

    # Test forward pass
    batch_size = 2
    head_input = torch.randn(batch_size, 2, 64, 64)
    noise_vec = torch.tensor([[0.9, 0.05, 0.03, 0.02], [0.1, 0.1, 0.7, 0.1]])

    with torch.no_grad():
        output_with_cond = head(head_input, condition_vector=noise_vec)
        output_without_cond = head(head_input, condition_vector=None)

    print(f"   Output with conditioning: shape={output_with_cond.shape}")
    print(f"   Output without conditioning: shape={output_without_cond.shape}")
    print(f"   Difference norm: {(output_with_cond - output_without_cond).norm():.6f}")

    # Initially should be nearly identical (identity gamma)
    diff = (output_with_cond - output_without_cond).abs().mean().item()
    print(f"   Mean absolute difference: {diff:.6f} (should be small initially)")

    assert output_with_cond.shape == (batch_size, 1, 64, 64), "❌ Wrong output shape!"
    assert not torch.isnan(output_with_cond).any(), "❌ NaN in output!"

    print(f"✅ Swin head with conditioning works correctly")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 3: NAFNet Head with Conditioning
print("Test 3: NAFNet Head with Conditioning")
print("-" * 80)
try:
    from nsnd.models.adaptive_multihead_refinement import NAFNetResidualHead

    # Create head WITH conditioning
    head = NAFNetResidualHead(
        width=16,
        use_conditioning=True,
        conditioner_hidden=16,
    )

    assert hasattr(head, 'conditioner'), "❌ NAFNet head missing conditioner!"
    assert head.conditioner is not None, "❌ Conditioner is None!"

    # Test forward pass
    head_input = torch.randn(2, 2, 64, 64)
    noise_vec = torch.tensor([[0.8, 0.1, 0.05, 0.05], [0.2, 0.6, 0.1, 0.1]])

    with torch.no_grad():
        output = head(head_input, condition_vector=noise_vec)

    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

    assert output.shape == (2, 1, 64, 64), "❌ Wrong output shape!"
    assert not torch.isnan(output).any(), "❌ NaN in output!"
    assert output.min() >= -1.0 and output.max() <= 1.0, "❌ Output not bounded by tanh!"

    print(f"✅ NAFNet head with conditioning works correctly")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 4: Adaptive Behavior Test
print("Test 4: Adaptive Behavior (Different Noise Vectors)")
print("-" * 80)
try:
    # Create conditioner
    conditioner = NoiseConditioner(noise_dim=4, feature_channels=32)

    # Train it a bit so it learns to differentiate
    optimizer = torch.optim.Adam(conditioner.parameters(), lr=1e-2)

    # Dummy training: encourage different outputs for different noise types
    for _ in range(100):
        high_speckle = torch.tensor([[0.9, 0.05, 0.03, 0.02]])
        low_speckle = torch.tensor([[0.1, 0.1, 0.7, 0.1]])

        gamma_high = conditioner(high_speckle)
        gamma_low = conditioner(low_speckle)

        # Loss: encourage difference
        loss = -F.mse_loss(gamma_high, gamma_low)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test differentiation
    with torch.no_grad():
        gamma_high = conditioner(high_speckle)
        gamma_low = conditioner(low_speckle)

    diff = (gamma_high - gamma_low).abs().mean().item()
    print(f"   High speckle gamma mean: {gamma_high.mean():.3f}")
    print(f"   Low speckle gamma mean: {gamma_low.mean():.3f}")
    print(f"   Difference: {diff:.3f}")

    assert diff > 0.01, f"❌ Conditioner not learning to differentiate! Diff={diff:.3f}"

    print(f"✅ Conditioner learns to produce different modulation for different noise")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 5: Gradient Flow
print("Test 5: Gradient Flow Through Conditioner")
print("-" * 80)
try:
    # Use just the conditioner to test gradient flow more directly
    conditioner = NoiseConditioner(noise_dim=4, feature_channels=32)

    # Forward pass
    noise_vec = torch.tensor([[0.7, 0.1, 0.1, 0.1]], requires_grad=False)
    features = torch.randn(1, 32, 64, 64, requires_grad=True)

    # Get modulation
    gamma = conditioner(noise_vec)  # [1, 32]
    gamma_map = gamma.view(1, 32, 1, 1)

    # Apply modulation
    modulated = features * gamma_map

    # Compute loss
    target = torch.randn(1, 32, 64, 64)
    loss = F.mse_loss(modulated, target)

    # Backward
    loss.backward()

    # Check conditioner gradients
    has_grads = False
    max_grad = 0.0
    for name, param in conditioner.named_parameters():
        if param.grad is not None:
            has_grads = True
            max_grad = max(max_grad, param.grad.abs().max().item())
            print(f"   {name}: grad_max={param.grad.abs().max():.6f}")

    assert has_grads, "❌ No gradients in conditioner!"
    assert max_grad > 0, "❌ Gradients are zero!"

    print(f"✅ Gradients flow correctly through conditioner")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 6: Backward Compatibility
print("Test 6: Backward Compatibility (Heads Without Conditioning)")
print("-" * 80)
try:
    # Create head WITHOUT conditioning
    head_no_cond = SwinResidualHead(
        img_size=64,
        in_chans=2,
        embed_dim=32,
        use_conditioning=False,
    )

    assert head_no_cond.conditioner is None, "❌ Conditioner should be None!"

    # Test forward pass (should work without condition_vector)
    head_input = torch.randn(1, 2, 64, 64)

    with torch.no_grad():
        output_old_style = head_no_cond(head_input)  # No condition_vector arg
        output_new_style = head_no_cond(head_input, condition_vector=None)  # Explicit None

    assert torch.allclose(output_old_style, output_new_style), "❌ Outputs differ!"

    print(f"   Legacy head (use_conditioning=False) works correctly")
    print(f"✅ Backward compatibility maintained")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 7: Parameter Count
print("Test 7: Parameter Count Analysis")
print("-" * 80)
try:
    # Count params in conditioner
    conditioner = NoiseConditioner(noise_dim=4, feature_channels=32, hidden_dim=16)
    total_params = sum(p.numel() for p in conditioner.parameters())
    trainable_params = sum(p.numel() for p in conditioner.parameters() if p.requires_grad)

    print(f"   Conditioner parameters: {total_params}")
    print(f"   Trainable parameters: {trainable_params}")

    expected = 4*16 + 16 + 16*32 + 32  # fc1.weight + fc1.bias + fc2.weight + fc2.bias
    print(f"   Expected: {expected}")

    assert total_params == expected, f"❌ Param count mismatch! Got {total_params}, expected {expected}"
    assert trainable_params == total_params, "❌ Some params not trainable!"

    # Compare to head size
    head = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32, use_conditioning=True)
    head_params = sum(p.numel() for p in head.parameters())
    cond_params = sum(p.numel() for p in head.conditioner.parameters())

    overhead = (cond_params / head_params) * 100

    print(f"   Total head parameters: {head_params:,}")
    print(f"   Conditioner parameters: {cond_params:,}")
    print(f"   Overhead: {overhead:.3f}% (negligible!)")

    assert overhead < 2.0, f"❌ Conditioner overhead too high! {overhead:.2f}%"

    print(f"✅ Parameter overhead is negligible ({overhead:.3f}% < 2%)")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Final Summary
print("=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)
print()
print("Summary:")
print("✅ NoiseConditioner initializes with identity output (~0.98)")
print("✅ Swin head supports conditioning (late fusion after Swin blocks)")
print("✅ NAFNet head supports conditioning (before final projection)")
print("✅ Conditioner learns to differentiate noise types")
print("✅ Gradients flow correctly through conditioner")
print("✅ Backward compatible (legacy heads still work)")
print("✅ Parameter overhead negligible (<2%)")
print()
print("🎉 Active Feature Modulation is ready to use!")
print()
print("To enable in training:")
print("  bash train_final_surgical.sh --use_head_conditioning")
print()
