#!/usr/bin/env python3
"""
Debug the FULL training pipeline to find where loss explosion occurs.
Tests the exact configuration from train_final_surgical.sh.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

print("=" * 80)
print("DEBUGGING FULL TRAINING PIPELINE")
print("=" * 80)
print()

# Test 1: Load Model with Exact Training Configuration
print("Test 1: Initialize Full Model")
print("-" * 80)
try:
    from nsnd_oct.scripts.train_hybrid_nsnd_multitask import MultiTaskHybridNSND

    model = MultiTaskHybridNSND(
        hybrid_analyzer_ckpt="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth",
        device="cpu",
        base_nafnet_width=64,
        base_nafnet_type="full",
        base_enc_blk_nums=[2, 2, 2],
        base_dec_blk_nums=[2, 2, 2],
        base_middle_blk_num=2,
        residual_head_width=32,
        use_swin_speckle=True,
        residual_scale=5.0,
        use_head_conditioning=True,
        conditioner_hidden=16,
    )

    print(f"   ✅ Model initialized")
    print(f"   - Base NAFNet width: 64")
    print(f"   - Residual head width: 32")
    print(f"   - Swin speckle: True")
    print(f"   - Conditioning: True")
    print(f"   - Residual scale: 5.0")
    print()
except Exception as e:
    print(f"   ❌ FAILED to initialize model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load Base NAFNet Checkpoint
print("Test 2: Load Base NAFNet Checkpoint")
print("-" * 80)
try:
    base_ckpt_path = "outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
    if Path(base_ckpt_path).exists():
        ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt

        # Load base NAFNet
        model.base_denoiser.load_state_dict(state, strict=False)
        print(f"   ✅ Loaded base NAFNet from {base_ckpt_path}")
    else:
        print(f"   ⚠️  Base checkpoint not found, using random init")
    print()
except Exception as e:
    print(f"   ⚠️  Could not load base checkpoint: {e}")
    print("   Continuing with random init...")
    print()

# Test 3: Simulate Real Training Data
print("Test 3: Forward Pass with Realistic Data")
print("-" * 80)
try:
    # Create realistic noisy OCT image
    batch_size = 2
    noisy = torch.rand(batch_size, 1, 64, 64) * 0.8 + 0.1  # OCT images typically in [0.1, 0.9]

    # Add realistic noise
    speckle_noise = torch.randn_like(noisy) * 0.1 * noisy  # Multiplicative
    gaussian_noise = torch.randn_like(noisy) * 0.05  # Additive
    noisy = (noisy + speckle_noise + gaussian_noise).clamp(0.0, 1.0)

    print(f"   Noisy input range: [{noisy.min():.3f}, {noisy.max():.3f}]")

    # Forward pass
    with torch.no_grad():
        output, weights_dict, extras = model(noisy)

    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Has NaN in output: {torch.isnan(output).any()}")
    print(f"   Has Inf in output: {torch.isinf(output).any()}")

    # Check intermediate outputs
    if 'base_output' in extras:
        base_out = extras['base_output']
        print(f"   Base output range: [{base_out.min():.3f}, {base_out.max():.3f}]")
        print(f"   Base has NaN: {torch.isnan(base_out).any()}")

    if 'expert_outputs' in extras:
        for name, expert_out in extras['expert_outputs'].items():
            has_nan = torch.isnan(expert_out).any()
            has_inf = torch.isinf(expert_out).any()
            status = "❌" if has_nan or has_inf else "✅"
            print(f"   {status} {name:10s} expert: [{expert_out.min():.3f}, {expert_out.max():.3f}] "
                  f"NaN={has_nan}, Inf={has_inf}")

    assert not torch.isnan(output).any(), "❌ NaN in output!"
    assert not torch.isinf(output).any(), "❌ Inf in output!"
    print("   ✅ Forward pass successful")
    print()
except Exception as e:
    print(f"   ❌ FAILED forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Training Step with Loss
print("Test 4: Full Training Step (Loss + Backward)")
print("-" * 80)
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Create training batch
    noisy = torch.rand(4, 1, 64, 64) * 0.8 + 0.1
    clean = torch.rand(4, 1, 64, 64) * 0.8 + 0.1

    # Forward
    output, weights_dict, extras = model(noisy)

    # Compute loss (as in training script)
    denoising_loss = F.l1_loss(output, clean)

    print(f"   Denoising loss: {denoising_loss.item():.6f}")

    # Check if loss is reasonable
    if denoising_loss.item() > 100:
        print(f"   ❌ EXPLOSION: Loss = {denoising_loss.item():.2f}")
        print(f"   Output stats: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")
        print(f"   Clean stats: min={clean.min():.3f}, max={clean.max():.3f}, mean={clean.mean():.3f}")
        raise ValueError("Loss explosion!")

    # Backward
    optimizer.zero_grad()
    denoising_loss.backward()

    # Check gradients
    nan_grads = []
    large_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grads.append(name)
            grad_norm = param.grad.norm().item()
            if grad_norm > 100:
                large_grads.append((name, grad_norm))

    if nan_grads:
        print(f"   ❌ NaN gradients in: {nan_grads[:5]}")
        raise ValueError("NaN gradients!")

    if large_grads:
        print(f"   ⚠️  Large gradients (>100):")
        for name, norm in large_grads[:5]:
            print(f"      {name}: {norm:.2f}")

    # Gradient clipping (as in training script)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    print("   ✅ Training step successful")
    print()
except Exception as e:
    print(f"   ❌ FAILED training step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Multiple Training Steps
print("Test 5: Run 20 Training Steps")
print("-" * 80)
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    losses = []
    for step in range(20):
        # Generate batch
        noisy = torch.rand(4, 1, 64, 64) * 0.8 + 0.1
        # Add realistic noise
        speckle = torch.randn_like(noisy) * 0.1 * noisy
        gaussian = torch.randn_like(noisy) * 0.05
        noisy = (noisy + speckle + gaussian).clamp(0.0, 1.0)

        clean = torch.rand(4, 1, 64, 64) * 0.8 + 0.1

        # Forward
        output, weights_dict, extras = model(noisy)

        # Loss
        loss = F.l1_loss(output, clean)

        # Check for explosion
        if loss.item() > 100 or torch.isnan(loss).any():
            print(f"   ❌ EXPLOSION at step {step}: loss={loss.item():.2f}")
            print(f"      Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
            print(f"      Output range: [{output.min():.3f}, {output.max():.3f}]")
            if 'expert_outputs' in extras:
                for name, exp_out in extras['expert_outputs'].items():
                    print(f"      {name}: [{exp_out.min():.3f}, {exp_out.max():.3f}]")
            raise ValueError(f"Loss explosion at step {step}!")

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if hasattr(model, 'residual_heads') and model.residual_heads is not None:
            for head in model.residual_heads.values():
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.3)

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 5 == 0:
            print(f"   Step {step+1:2d}: loss={loss.item():.6f}")

    print(f"   Loss trend: {losses[0]:.6f} → {losses[-1]:.6f}")
    print("   ✅ 20 training steps stable!")
    print()
except Exception as e:
    print(f"   ❌ FAILED at step {step}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("✅ FULL PIPELINE TEST PASSED!")
print("=" * 80)
print()
print("Model is stable in controlled test.")
print()
print("If training still fails, the issue is likely:")
print("  1. Data corruption in actual dataset")
print("  2. Extreme/pathological samples")
print("  3. Learning rate too high for some component")
print("  4. Mismatch between checkpoint and current architecture")
print()
print("Next steps:")
print("  1. Check first few batches of actual training data")
print("  2. Add data validation in DataLoader")
print("  3. Try lower learning rate (1e-4 instead of 5e-4)")
print()
