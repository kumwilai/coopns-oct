#!/usr/bin/env python3
"""Investigate NAFNet training divergence."""
import sys
sys.path.append('sota/models')

import torch
from nafnet_fair import NAFNet
from adaptive_oct_denoise import PairedOCTDataset, resize_to
from torch.utils.data import DataLoader

print("="*80)
print("Investigating NAFNet Training Divergence")
print("="*80)

# Load dataset
tfm = resize_to((64, 64))
ds = PairedOCTDataset('train_pairs_universal.txt', transform=tfm)
loader = DataLoader(ds, batch_size=4, shuffle=False)

# Load the best model from epoch 1
model = NAFNet(width=48, middle_blk_num=2, enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2])
try:
    state = torch.load('outputs/nafnet_universal/nafnet_best.pth', map_location='cpu')
    model.load_state_dict(state)
    print("✓ Loaded best model from epoch 1")
except:
    print("⚠️  Could not load saved model, using fresh initialization")

model.eval()

# Check for NaN/Inf in model parameters
print("\n" + "="*80)
print("Checking Model Parameters")
print("="*80)
nan_params = []
inf_params = []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_params.append(name)
    if torch.isinf(param).any():
        inf_params.append(name)

if nan_params:
    print(f"⚠️  NaN parameters found: {len(nan_params)}")
    for name in nan_params[:5]:
        print(f"  - {name}")
else:
    print("✓ No NaN parameters")

if inf_params:
    print(f"⚠️  Inf parameters found: {len(inf_params)}")
    for name in inf_params[:5]:
        print(f"  - {name}")
else:
    print("✓ No Inf parameters")

# Check beta/gamma values
print("\n" + "="*80)
print("Beta/Gamma Values Across Blocks")
print("="*80)
beta_values = []
gamma_values = []

for i, encoder in enumerate(model.encoders):
    for j, block in enumerate(encoder):
        beta_values.append(block.beta.abs().max().item())
        gamma_values.append(block.gamma.abs().max().item())
        if i == 0 and j == 0:
            print(f"Encoder[{i}][{j}] beta: {block.beta.mean():.4f}, gamma: {block.gamma.mean():.4f}")

for i, decoder in enumerate(model.decoders):
    for j, block in enumerate(decoder):
        beta_values.append(block.beta.abs().max().item())
        gamma_values.append(block.gamma.abs().max().item())

print(f"Beta max across all blocks: {max(beta_values):.4f}")
print(f"Gamma max across all blocks: {max(gamma_values):.4f}")

if max(beta_values) > 10 or max(gamma_values) > 10:
    print("⚠️  WARNING: Beta/Gamma values are very large!")

# Test on problematic batch (around batch 1520)
print("\n" + "="*80)
print("Testing Batch 1520 (where divergence occurred)")
print("="*80)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch_idx, (x_noisy, y_clean) in enumerate(loader):
    if batch_idx != 1519:  # Batch 1520 (0-indexed)
        continue

    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Input range: [{x_noisy.min():.4f}, {x_noisy.max():.4f}]")
    print(f"  Target range: [{y_clean.min():.4f}, {y_clean.max():.4f}]")

    # Forward pass
    pred = model(x_noisy)
    print(f"  Output range: [{pred.min():.4f}, {pred.max():.4f}]")

    if torch.isnan(pred).any():
        print("  ⚠️  NaN in output!")
    if torch.isinf(pred).any():
        print("  ⚠️  Inf in output!")

    # Compute loss
    def charbonnier(x, y, eps=1e-3):
        return torch.mean(torch.sqrt((x - y) ** 2 + eps * eps))

    loss = charbonnier(pred, y_clean)
    print(f"  Loss: {loss.item():.4f}")

    if torch.isnan(loss).any():
        print("  ⚠️  NaN in loss!")
    if torch.isinf(loss).any():
        print("  ⚠️  Inf in loss!")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norms = []
    nan_grads = []
    inf_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if torch.isinf(param.grad).any():
                inf_grads.append(name)

    print(f"  Gradient norm: max={max(grad_norms):.4f}, mean={sum(grad_norms)/len(grad_norms):.4f}")

    if nan_grads:
        print(f"  ⚠️  NaN gradients in {len(nan_grads)} parameters")
        for name in nan_grads[:3]:
            print(f"    - {name}")

    if inf_grads:
        print(f"  ⚠️  Inf gradients in {len(inf_grads)} parameters")
        for name in inf_grads[:3]:
            print(f"    - {name}")

    if max(grad_norms) > 100:
        print(f"  ⚠️  WARNING: Gradient explosion! Max gradient norm: {max(grad_norms):.2f}")

    break

print("\n" + "="*80)
print("Investigation Complete")
print("="*80)
