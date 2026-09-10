#!/usr/bin/env python3
"""Quick diagnostic for NAFNet training issues."""
import sys
sys.path.append('sota/models')

import torch
from nafnet_fair import NAFNet
from adaptive_oct_denoise import PairedOCTDataset, resize_to

# Load a small sample
tfm = resize_to((64, 64))
ds = PairedOCTDataset('train_pairs_universal.txt', transform=tfm)

# Get one sample
x_noisy, y_clean = ds[0]
x_noisy = x_noisy.unsqueeze(0)  # Add batch dim
y_clean = y_clean.unsqueeze(0)

print("Data diagnostics:")
print(f"  Noisy range: [{x_noisy.min():.4f}, {x_noisy.max():.4f}]")
print(f"  Clean range: [{y_clean.min():.4f}, {y_clean.max():.4f}]")
print(f"  Noisy mean: {x_noisy.mean():.4f}, std: {x_noisy.std():.4f}")
print(f"  Clean mean: {y_clean.mean():.4f}, std: {y_clean.std():.4f}")

# Create model
model = NAFNet(width=48, middle_blk_num=2, enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2])
model.eval()

# Forward pass
with torch.no_grad():
    pred = model(x_noisy)

print(f"\nModel output:")
print(f"  Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
print(f"  Pred mean: {pred.mean():.4f}, std: {pred.std():.4f}")

# Check losses
def charbonnier(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps * eps))

# Linear domain loss
loss_linear = charbonnier(pred, y_clean)
print(f"\nLinear domain Charbonnier loss: {loss_linear.item():.4f}")

# Log domain loss
pred_log = torch.log(pred.clamp(min=1e-6))
clean_log = torch.log(y_clean.clamp(min=1e-6))
loss_log = charbonnier(pred_log, clean_log)
print(f"Log domain Charbonnier loss: {loss_log.item():.4f}")

print(f"\nLog values:")
print(f"  Pred log range: [{pred_log.min():.4f}, {pred_log.max():.4f}]")
print(f"  Clean log range: [{clean_log.min():.4f}, {clean_log.max():.4f}]")

# Check initialization
print(f"\nModel initialization check:")
first_block = model.encoders[0][0]
print(f"  Beta: mean={first_block.beta.mean().item():.4f}, range=[{first_block.beta.min().item():.4f}, {first_block.beta.max().item():.4f}]")
print(f"  Gamma: mean={first_block.gamma.mean().item():.4f}, range=[{first_block.gamma.min().item():.4f}, {first_block.gamma.max().item():.4f}]")
