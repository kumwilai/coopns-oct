#!/usr/bin/env python3
"""
Verify CASA model outputs coherent_map in both train and eval modes.
Critical check before starting training!
"""
import torch
import sys
sys.path.insert(0, '.')

from adaptive_oct_denoise import build_model, device

print("=" * 80)
print("VERIFYING CASA MODEL OUTPUTS")
print("=" * 80)

# Build CASA model
model = build_model(
    base_channels=48,
    residual_mode=True,
    adapter_type="casa",
    backbone_type="noise2void"
).to(device)

test_input = torch.randn(2, 1, 64, 64).to(device)

# Test 1: Eval mode
print("\n1. Testing EVAL mode...")
model.eval()
with torch.no_grad():
    output = model(test_input, return_aux=True)

if isinstance(output, tuple):
    pred, aux = output
    print(f"  ✓ Returns tuple: (pred, aux)")
    print(f"  pred shape: {pred.shape}")
    print(f"  aux keys: {list(aux.keys())}")

    if 'coherent_map' in aux and 'incoherent_map' in aux:
        print(f"  ✓ coherent_map shape: {aux['coherent_map'].shape}")
        print(f"  ✓ incoherent_map shape: {aux['incoherent_map'].shape}")

        # Check values
        coherent = aux['coherent_map']
        incoherent = aux['incoherent_map']
        weight_sum = coherent + incoherent

        print(f"\n  Coherent range: [{coherent.min():.3f}, {coherent.max():.3f}]")
        print(f"  Incoherent range: [{incoherent.min():.3f}, {incoherent.max():.3f}]")
        print(f"  Sum (coherent + incoherent): mean={weight_sum.mean():.3f}, std={weight_sum.std():.3f}")

        if abs(weight_sum.mean() - 1.0) < 0.1:
            print(f"  ✓ Weights approximately sum to 1.0")
        else:
            print(f"  ⚠️ Weights don't sum to 1.0!")
    else:
        print(f"  ❌ MISSING coherent_map or incoherent_map in aux!")
        print(f"     Available keys: {list(aux.keys())}")
else:
    print(f"  ❌ Model does not return tuple! Got type: {type(output)}")

# Test 2: Train mode (CRITICAL - this is what we'll use during training!)
print("\n2. Testing TRAIN mode (with gradients)...")
model.train()
test_input_train = torch.randn(2, 1, 64, 64, requires_grad=True).to(device)

output = model(test_input_train, return_aux=True)

if isinstance(output, tuple):
    pred, aux = output
    print(f"  ✓ Returns tuple: (pred, aux)")
    print(f"  pred shape: {pred.shape}")
    print(f"  aux keys: {list(aux.keys())}")

    if 'coherent_map' in aux:
        coherent = aux['coherent_map']
        print(f"  ✓ coherent_map shape: {coherent.shape}")
        print(f"  coherent_map requires_grad: {coherent.requires_grad}")

        # Test gradient flow
        print(f"\n3. Testing gradient flow...")
        from adaptive_oct_denoise import CoherenceSupervisionLoss
        coherence_loss_fn = CoherenceSupervisionLoss()

        # Compute coherence loss
        coherence_loss = coherence_loss_fn(coherent, test_input_train)
        print(f"  Coherence loss: {coherence_loss.item():.3f}")

        # Backward
        coherence_loss.backward()

        # Check gradients
        has_grad = False
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append((name, param.grad.norm().item()))
                has_grad = True

        if has_grad:
            print(f"  ✓ Gradients computed successfully")
            print(f"  Gradient norms (top 5):")
            for name, norm in sorted(grad_norms, key=lambda x: -x[1])[:5]:
                print(f"    {name}: {norm:.6f}")
        else:
            print(f"  ❌ NO GRADIENTS computed!")
    else:
        print(f"  ❌ MISSING coherent_map in train mode!")
        print(f"     This is CRITICAL - training will fail!")
else:
    print(f"  ❌ Model does not return tuple in train mode!")

# Test 3: Check CASA adapter structure
print("\n4. Checking CASA adapter structure...")
for name, module in model.named_modules():
    if 'adapter' in name:
        print(f"  {name}: {module.__class__.__name__}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)

# Final verdict
eval_ok = False
train_ok = False

model.eval()
with torch.no_grad():
    out = model(test_input, return_aux=True)
    if isinstance(out, tuple) and 'coherent_map' in out[1]:
        eval_ok = True

model.train()
out = model(test_input, return_aux=True)
if isinstance(out, tuple) and 'coherent_map' in out[1]:
    train_ok = True

if eval_ok and train_ok:
    print("✓ CASA model outputs are CORRECT in both train and eval modes")
    print("✓ Safe to proceed with training!")
    print("\nRecommended training command:")
    print("  python adaptive_oct_denoise.py \\")
    print("    --paired_list train_pairs_universal.txt \\")
    print("    --val_paired_list val_pairs_universal.txt \\")
    print("    --backbone noise2void \\")
    print("    --adapter casa \\")
    print("    --base_channels 48 \\")
    print("    --residual_mode \\")
    print("    --lambda_coherence 0.1 \\")
    print("    --finetune_epochs 100 \\")
    print("    --output_dir checkpoints/casa_coherence_supervised")
else:
    print("❌ CASA model outputs are INCORRECT!")
    if not eval_ok:
        print("   - EVAL mode missing coherent_map")
    if not train_ok:
        print("   - TRAIN mode missing coherent_map")
    print("   DO NOT TRAIN - FIX THE MODEL FIRST!")

print("=" * 80)
