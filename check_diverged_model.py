#!/usr/bin/env python3
"""Check the diverged model state."""
import sys
sys.path.append('sota/models')

import torch
from nafnet_fair import NAFNet

print("="*80)
print("Checking Diverged Model (Epoch 2)")
print("="*80)

model = NAFNet(width=48, middle_blk_num=2, enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2])

# Try to load the checkpoint from epoch 2 if it exists
try:
    checkpoint = torch.load('outputs/nafnet_universal/nafnet_checkpoint_epoch10.pth', map_location='cpu')
    print("✓ Found epoch 10 checkpoint")
except:
    try:
        # Try the final model
        state = torch.load('outputs/nafnet_universal/nafnet_final.pth', map_location='cpu')
        model.load_state_dict(state)
        print("✓ Loaded final model")
    except:
        print("⚠️  No saved model found")
        sys.exit(1)

# Check for extreme parameter values
print("\n" + "="*80)
print("Parameter Statistics")
print("="*80)

extreme_params = []
for name, param in model.named_parameters():
    param_max = param.abs().max().item()
    param_mean = param.abs().mean().item()

    if param_max > 100:
        extreme_params.append((name, param_max))

    if 'beta' in name or 'gamma' in name:
        print(f"{name:60s} max={param_max:8.4f} mean={param_mean:8.4f}")

if extreme_params:
    print(f"\n⚠️  Found {len(extreme_params)} parameters with extreme values (>100):")
    for name, value in extreme_params[:10]:
        print(f"  {name}: {value:.2f}")
else:
    print("\n✓ No extremely large parameters found")

# Check beta/gamma specifically
print("\n" + "="*80)
print("Beta/Gamma Analysis")
print("="*80)

beta_values = []
gamma_values = []

for i, encoder in enumerate(model.encoders):
    for j, block in enumerate(encoder):
        beta_val = block.beta.abs().max().item()
        gamma_val = block.gamma.abs().max().item()
        beta_values.append(beta_val)
        gamma_values.append(gamma_val)

for i, decoder in enumerate(model.decoders):
    for j, block in enumerate(decoder):
        beta_val = block.beta.abs().max().item()
        gamma_val = block.gamma.abs().max().item()
        beta_values.append(beta_val)
        gamma_values.append(gamma_val)

print(f"Beta:  min={min(beta_values):.4f}, max={max(beta_values):.4f}, mean={sum(beta_values)/len(beta_values):.4f}")
print(f"Gamma: min={min(gamma_values):.4f}, max={max(gamma_values):.4f}, mean={sum(gamma_values)/len(gamma_values):.4f}")

if max(beta_values) > 1000 or max(gamma_values) > 1000:
    print("⚠️  CRITICAL: Beta/Gamma have exploded!")
elif max(beta_values) > 10 or max(gamma_values) > 10:
    print("⚠️  WARNING: Beta/Gamma are very large")
else:
    print("✓ Beta/Gamma values are reasonable")

# Test forward pass
print("\n" + "="*80)
print("Forward Pass Test")
print("="*80)

model.eval()
x = torch.randn(1, 1, 64, 64).clamp(0, 1)
with torch.no_grad():
    try:
        y = model(x)
        print(f"Input range:  [{x.min():.4f}, {x.max():.4f}]")
        print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")

        if torch.isnan(y).any():
            print("⚠️  NaN in output!")
        elif torch.isinf(y).any():
            print("⚠️  Inf in output!")
        elif (y < 0).any() or (y > 1.1).any():
            print("⚠️  Output out of expected range [0, 1]")
        else:
            print("✓ Forward pass successful")
    except Exception as e:
        print(f"⚠️  Forward pass failed: {e}")

print("\n" + "="*80)
