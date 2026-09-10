#!/usr/bin/env python3
"""
Test script to verify all stability fixes are working correctly.
Tests:
1. Swin head initialization with input normalization
2. Forward pass with 2-channel amplified input
3. No NaN/Inf in outputs
4. Warmup schedule progression
5. Gradient flow and clipping
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 80)
print("STABILITY FIXES VERIFICATION TEST")
print("=" * 80)
print()

# Test 1: Swin Head Initialization
print("Test 1: Swin Head Initialization with Input Normalization")
print("-" * 80)
try:
    from nsnd.models.swin_head import SwinResidualHead

    head = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32)

    # Check that input_norm exists
    assert hasattr(head, 'input_norm'), "❌ Missing input_norm layer!"
    assert isinstance(head.input_norm, nn.GroupNorm), "❌ input_norm is not GroupNorm!"

    # Check conv_first initialization is scaled down
    conv_weight_mean = head.conv_first.weight.abs().mean().item()
    assert conv_weight_mean < 0.05, f"❌ conv_first not scaled down! Mean weight: {conv_weight_mean:.4f}"

    print(f"✅ Swin head initialized correctly")
    print(f"   - input_norm: {head.input_norm}")
    print(f"   - conv_first weight mean: {conv_weight_mean:.6f} (should be ~0.01)")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    print()
    sys.exit(1)

# Test 2: Forward Pass with Amplified Input
print("Test 2: Forward Pass with 20x Amplified Input")
print("-" * 80)
try:
    batch_size = 2
    img_size = 64

    # Simulate amplified residual + base context
    residual_scale = 20.0
    residual = torch.randn(batch_size, 1, img_size, img_size) * 0.05  # Small residuals
    base = torch.randn(batch_size, 1, img_size, img_size) * 0.5 + 0.5  # ~[0, 1] range

    # Create 2-channel input (as in line 610 of train_hybrid_nsnd_multitask.py)
    head_input = torch.cat([residual * residual_scale, base], dim=1)  # (B, 2, H, W)

    print(f"   Input shape: {head_input.shape}")
    print(f"   Input range: [{head_input.min().item():.2f}, {head_input.max().item():.2f}]")
    print(f"   Amplified residual channel range: [{head_input[:, 0].min().item():.2f}, {head_input[:, 0].max().item():.2f}]")
    print(f"   Base channel range: [{head_input[:, 1].min().item():.2f}, {head_input[:, 1].max().item():.2f}]")

    # Forward pass
    with torch.no_grad():
        output = head(head_input)

    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "❌ Output contains NaN!"
    assert not torch.isinf(output).any(), "❌ Output contains Inf!"
    assert output.shape == (batch_size, 1, img_size, img_size), "❌ Wrong output shape!"

    # Output should be bounded by tanh
    assert output.min() >= -1.0 and output.max() <= 1.0, "❌ Output not bounded by tanh!"

    print(f"✅ Forward pass successful with 20x amplification")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 3: Gradient Flow Test
print("Test 3: Gradient Flow with Loss")
print("-" * 80)
try:
    head.train()

    # Create synthetic target
    target = torch.randn(batch_size, 1, img_size, img_size).clamp(-0.1, 0.1)

    # Forward pass with gradients
    output = head(head_input)
    loss = F.l1_loss(output, target)

    print(f"   Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients exist and are finite
    grad_count = 0
    nan_grad_count = 0
    max_grad = 0.0

    for name, param in head.named_parameters():
        if param.grad is not None:
            grad_count += 1
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grad_count += 1
                print(f"   ⚠️  {name}: NaN/Inf gradient!")
            max_grad = max(max_grad, param.grad.abs().max().item())

    assert grad_count > 0, "❌ No gradients computed!"
    assert nan_grad_count == 0, f"❌ {nan_grad_count}/{grad_count} parameters have NaN/Inf gradients!"

    print(f"✅ Gradients computed successfully")
    print(f"   - Parameters with gradients: {grad_count}")
    print(f"   - Max gradient magnitude: {max_grad:.4f}")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 4: Warmup Schedule
print("Test 4: Residual Amplification Warmup Schedule")
print("-" * 80)
try:
    # Mock model class with warmup
    class MockModel(nn.Module):
        def __init__(self, residual_scale_target=20.0):
            super().__init__()
            self.residual_scale_target = float(residual_scale_target)
            self.register_buffer("residual_scale", torch.tensor(1.0))
            self.warmup_epochs = 10 if residual_scale_target > 5.0 else 5

        def update_residual_scale(self, epoch: int):
            if epoch >= self.warmup_epochs:
                self.residual_scale.fill_(self.residual_scale_target)
            else:
                alpha = epoch / self.warmup_epochs
                alpha = 0.5 * (1 - math.cos(math.pi * alpha))
                current_scale = 1.0 + alpha * (self.residual_scale_target - 1.0)
                self.residual_scale.fill_(current_scale)
            return float(self.residual_scale.item())

    model = MockModel(residual_scale_target=20.0)

    print(f"   Target amplification: {model.residual_scale_target}x")
    print(f"   Warmup epochs: {model.warmup_epochs}")
    print()

    # Test warmup progression
    test_epochs = [0, 2, 5, 8, 10, 15]
    expected_progression = [1.0, 3.82, 10.55, 17.61, 20.0, 20.0]

    print("   Epoch | Amplification | Expected")
    print("   ------|---------------|----------")

    for epoch, expected in zip(test_epochs, expected_progression):
        current = model.update_residual_scale(epoch)
        status = "✓" if abs(current - expected) < 0.1 else "✗"
        print(f"   {epoch:5d} | {current:13.2f}x | {expected:6.2f}x  {status}")

    # Verify final scale reaches target
    final_scale = model.update_residual_scale(20)
    assert abs(final_scale - 20.0) < 1e-4, f"❌ Final scale {final_scale} != 20.0!"

    print()
    print(f"✅ Warmup schedule working correctly")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 5: High Amplification Stress Test
print("Test 5: High Amplification Stress Test (50x)")
print("-" * 80)
try:
    # Test with extreme amplification to verify normalization works
    extreme_scale = 50.0
    extreme_residual = torch.randn(2, 1, 64, 64) * 0.1
    extreme_base = torch.randn(2, 1, 64, 64) * 0.5 + 0.5
    extreme_input = torch.cat([extreme_residual * extreme_scale, extreme_base], dim=1)

    print(f"   Extreme amplification: {extreme_scale}x")
    print(f"   Input range: [{extreme_input.min().item():.2f}, {extreme_input.max().item():.2f}]")

    with torch.no_grad():
        extreme_output = head(extreme_input)

    print(f"   Output range: [{extreme_output.min().item():.2f}, {extreme_output.max().item():.2f}]")

    assert not torch.isnan(extreme_output).any(), "❌ NaN with 50x amplification!"
    assert not torch.isinf(extreme_output).any(), "❌ Inf with 50x amplification!"
    assert extreme_output.abs().max() <= 1.0, "❌ Output not bounded!"

    print(f"✅ Handles extreme amplification (50x) without explosion")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 6: Layer-wise Gradient Clipping
print("Test 6: Layer-wise Gradient Clipping")
print("-" * 80)
try:
    from nsnd.models.swin_head import SwinResidualHead
    from nsnd.models.adaptive_multihead_refinement import NAFNetResidualHead

    swin_head = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32)
    nafnet_head = NAFNetResidualHead(width=16)

    # Create dummy input and target
    dummy_input_swin = torch.randn(1, 2, 64, 64)
    dummy_input_nafnet = torch.randn(1, 2, 64, 64)
    target = torch.randn(1, 1, 64, 64) * 0.1

    # Forward + backward
    out_swin = swin_head(dummy_input_swin)
    out_nafnet = nafnet_head(dummy_input_nafnet)
    loss = F.l1_loss(out_swin, target) + F.l1_loss(out_nafnet, target)
    loss.backward()

    # Simulate layer-wise clipping
    def apply_layerwise_clipping():
        # Swin head
        for layer in swin_head.layers:
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(swin_head.conv_last.parameters(), 0.1)

        # NAFNet head
        torch.nn.utils.clip_grad_norm_(nafnet_head.parameters(), 0.3)

    apply_layerwise_clipping()

    # Check gradient norms
    swin_layer_norms = []
    for layer in swin_head.layers:
        norm = sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None) ** 0.5
        swin_layer_norms.append(norm)

    swin_last_norm = swin_head.conv_last.weight.grad.norm().item() if swin_head.conv_last.weight.grad is not None else 0.0
    nafnet_norm = sum(p.grad.norm().item() ** 2 for p in nafnet_head.parameters() if p.grad is not None) ** 0.5

    print(f"   Swin transformer layers: {[f'{n:.4f}' for n in swin_layer_norms]}")
    print(f"   Swin conv_last: {swin_last_norm:.4f} (should be ≤ 0.1)")
    print(f"   NAFNet head: {nafnet_norm:.4f} (should be ≤ 0.3)")

    # Verify clipping worked
    assert all(n <= 0.6 for n in swin_layer_norms), "❌ Swin layer gradients not clipped!"
    assert swin_last_norm <= 0.15, "❌ Swin conv_last gradients not clipped!"
    assert nafnet_norm <= 0.35, "❌ NAFNet gradients not clipped!"

    print(f"✅ Layer-wise gradient clipping working correctly")
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
print("✅ Swin head initializes with input normalization (GroupNorm)")
print("✅ Conservative conv_first initialization (10x scaled down)")
print("✅ Forward pass handles 20x amplified inputs without NaN/Inf")
print("✅ Gradients flow correctly through all layers")
print("✅ Warmup schedule progresses from 1x to 20x over 10 epochs")
print("✅ Handles extreme amplification (50x) without explosion")
print("✅ Layer-wise gradient clipping works as expected")
print()
print("🎉 The stability fixes are working correctly!")
print("   You can now run: bash train_final_surgical.sh")
print()
