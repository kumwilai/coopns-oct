#!/usr/bin/env python3
"""
Debug script to find the exact source of Swin head loss explosion.
Tests each component in isolation to identify the failure point.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 80)
print("DEBUGGING SWIN HEAD LOSS EXPLOSION")
print("=" * 80)
print()

# Test 1: Basic Swin Head Forward Pass
print("Test 1: Basic Swin Head (No Conditioning, No Amplification)")
print("-" * 80)
try:
    from nsnd.models.swin_head import SwinResidualHead

    head = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32, use_conditioning=False)

    # Realistic input (not amplified)
    residual = torch.randn(2, 1, 64, 64) * 0.05  # Small residuals
    base = torch.rand(2, 1, 64, 64) * 0.5 + 0.5  # Base in [0.5, 1.0]
    head_input = torch.cat([residual, base], dim=1)  # No amplification yet

    with torch.no_grad():
        output = head(head_input)

    print(f"   Input range: [{head_input.min():.3f}, {head_input.max():.3f}]")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Has NaN: {torch.isnan(output).any()}")
    print(f"   Has Inf: {torch.isinf(output).any()}")

    assert not torch.isnan(output).any(), "❌ NaN in basic forward!"
    assert not torch.isinf(output).any(), "❌ Inf in basic forward!"
    print("✅ Basic forward pass works")
    print()
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: With 5x Amplification
print("Test 2: With 5x Residual Amplification")
print("-" * 80)
try:
    residual_scale = 5.0
    residual = torch.randn(2, 1, 64, 64) * 0.05
    base = torch.rand(2, 1, 64, 64) * 0.5 + 0.5
    head_input = torch.cat([residual * residual_scale, base], dim=1)  # 5x amplification

    with torch.no_grad():
        output = head(head_input)

    print(f"   Amplified input range: [{head_input.min():.3f}, {head_input.max():.3f}]")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Has NaN: {torch.isnan(output).any()}")
    print(f"   Has Inf: {torch.isinf(output).any()}")

    assert not torch.isnan(output).any(), "❌ NaN with 5x amplification!"
    assert not torch.isinf(output).any(), "❌ Inf with 5x amplification!"
    print("✅ 5x amplification works")
    print()
except Exception as e:
    print(f"❌ FAILED at 5x amplification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: With Extreme Amplification (20x)
print("Test 3: With Extreme 20x Amplification")
print("-" * 80)
try:
    residual_scale = 20.0
    residual = torch.randn(2, 1, 64, 64) * 0.05
    base = torch.rand(2, 1, 64, 64) * 0.5 + 0.5
    head_input = torch.cat([residual * residual_scale, base], dim=1)

    with torch.no_grad():
        output = head(head_input)

    print(f"   Extreme input range: [{head_input.min():.3f}, {head_input.max():.3f}]")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Has NaN: {torch.isnan(output).any()}")
    print(f"   Has Inf: {torch.isinf(output).any()}")

    if torch.isnan(output).any() or torch.isinf(output).any():
        print("⚠️  WARNING: NaN/Inf at 20x (expected, need normalization)")
    else:
        print("✅ Even 20x amplification works!")
    print()
except Exception as e:
    print(f"⚠️  Expected failure at 20x: {e}")
    print()

# Test 4: With Conditioning
print("Test 4: With Active Feature Modulation (Conditioning)")
print("-" * 80)
try:
    head_cond = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32, use_conditioning=True)

    residual_scale = 5.0
    residual = torch.randn(2, 1, 64, 64) * 0.05
    base = torch.rand(2, 1, 64, 64) * 0.5 + 0.5
    head_input = torch.cat([residual * residual_scale, base], dim=1)

    # Noise vectors
    condition_vector = torch.tensor([
        [0.9, 0.05, 0.03, 0.02],
        [0.1, 0.1, 0.7, 0.1]
    ])

    with torch.no_grad():
        output = head_cond(head_input, condition_vector=condition_vector)

    print(f"   Condition vector: {condition_vector}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Has NaN: {torch.isnan(output).any()}")
    print(f"   Has Inf: {torch.isinf(output).any()}")

    assert not torch.isnan(output).any(), "❌ NaN with conditioning!"
    assert not torch.isinf(output).any(), "❌ Inf with conditioning!"
    print("✅ Conditioning works")
    print()
except Exception as e:
    print(f"❌ FAILED with conditioning: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Gradient Flow with Loss
print("Test 5: Backward Pass with Loss (Training Simulation)")
print("-" * 80)
try:
    head_train = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32, use_conditioning=True)
    head_train.train()

    residual_scale = 5.0
    residual = torch.randn(4, 1, 64, 64) * 0.05  # Larger batch
    base = torch.rand(4, 1, 64, 64) * 0.5 + 0.5
    head_input = torch.cat([residual * residual_scale, base], dim=1)

    condition_vector = torch.rand(4, 4)
    condition_vector = condition_vector / condition_vector.sum(dim=1, keepdim=True)  # Normalize

    # Forward
    output = head_train(head_input, condition_vector=condition_vector)

    # Create realistic target (small residual correction)
    target = torch.randn(4, 1, 64, 64) * 0.02

    # Loss
    loss = F.l1_loss(output, target)

    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Backward
    loss.backward()

    # Check gradients
    max_grad = 0.0
    nan_count = 0
    for name, param in head_train.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"   ❌ NaN/Inf gradient in {name}")
                nan_count += 1
            max_grad = max(max_grad, param.grad.abs().max().item())

    print(f"   Max gradient: {max_grad:.6f}")
    print(f"   NaN gradients: {nan_count}")

    assert nan_count == 0, "❌ NaN gradients found!"
    assert loss.item() < 100, f"❌ Loss too high: {loss.item()}"
    print("✅ Backward pass works")
    print()
except Exception as e:
    print(f"❌ FAILED in backward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Stress Test with Real Training Conditions
print("Test 6: Stress Test (10 iterations, realistic noise)")
print("-" * 80)
try:
    head_stress = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32, use_conditioning=True)
    optimizer = torch.optim.Adam(head_stress.parameters(), lr=5e-4)

    losses = []
    for i in range(10):
        # Simulate real training data
        residual_scale = 5.0
        residual = torch.randn(2, 1, 64, 64) * 0.1  # Larger residuals
        base = torch.rand(2, 1, 64, 64) * 0.8 + 0.1  # Realistic base
        head_input = torch.cat([residual * residual_scale, base], dim=1)

        condition_vector = torch.rand(2, 4)
        condition_vector = condition_vector / condition_vector.sum(dim=1, keepdim=True)

        # Forward
        output = head_stress(head_input, condition_vector=condition_vector)

        # Loss
        target = torch.randn(2, 1, 64, 64) * 0.05
        loss = F.l1_loss(output, target)

        # Check for explosion
        if loss.item() > 100 or torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"   ❌ EXPLOSION at iteration {i}: loss={loss.item():.2f}")
            print(f"      Input range: [{head_input.min():.3f}, {head_input.max():.3f}]")
            print(f"      Output range: [{output.min():.3f}, {output.max():.3f}]")
            raise ValueError("Loss explosion detected!")

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (as in training script)
        torch.nn.utils.clip_grad_norm_(head_stress.parameters(), 0.5)

        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 3 == 0:
            print(f"   Iteration {i+1}: loss={loss.item():.6f}")

    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Loss trend: {losses[0]:.6f} → {losses[-1]:.6f}")
    print("✅ Stress test passed - stable training!")
    print()
except Exception as e:
    print(f"❌ FAILED stress test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check Activation Statistics in Swin Blocks
print("Test 7: Internal Activation Statistics")
print("-" * 80)
try:
    head_debug = SwinResidualHead(img_size=64, in_chans=2, embed_dim=32, use_conditioning=True)

    residual_scale = 5.0
    residual = torch.randn(2, 1, 64, 64) * 0.05
    base = torch.rand(2, 1, 64, 64) * 0.5 + 0.5
    head_input = torch.cat([residual * residual_scale, base], dim=1)

    # Hook to capture intermediate activations
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                }
        return hook

    # Register hooks
    head_debug.input_norm.register_forward_hook(hook_fn('input_norm'))
    head_debug.conv_first.register_forward_hook(hook_fn('conv_first'))
    head_debug.norm.register_forward_hook(hook_fn('norm'))
    if head_debug.conditioner:
        head_debug.conditioner.register_forward_hook(hook_fn('conditioner'))
    head_debug.conv_last.register_forward_hook(hook_fn('conv_last'))

    # Forward pass
    condition_vector = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.2, 0.3, 0.3, 0.2]])
    with torch.no_grad():
        output = head_debug(head_input, condition_vector=condition_vector)

    # Print statistics
    print("   Layer activations:")
    for name, stats in activations.items():
        status = "✅" if not stats['has_nan'] and not stats['has_inf'] else "❌"
        print(f"   {status} {name:15s}: mean={stats['mean']:7.3f}, std={stats['std']:6.3f}, "
              f"range=[{stats['min']:7.3f}, {stats['max']:7.3f}]")

    # Check for problems
    has_issues = any(s['has_nan'] or s['has_inf'] for s in activations.values())
    if has_issues:
        print("   ❌ Found NaN/Inf in activations!")
        sys.exit(1)
    else:
        print("   ✅ All activations healthy!")
    print()
except Exception as e:
    print(f"❌ FAILED activation check: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("✅ ALL DEBUGGING TESTS PASSED!")
print("=" * 80)
print()
print("Swin head is working correctly in isolation.")
print("The loss explosion must be coming from:")
print("  1. Interaction with other components (base NAFNet, other heads)")
print("  2. Data issues (corrupted samples, extreme noise)")
print("  3. Training dynamics (learning rate, loss computation)")
print()
print("Next: Run integration test with full model...")
print()
