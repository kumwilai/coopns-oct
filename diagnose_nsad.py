#!/usr/bin/env python3
"""Diagnose why NSAD is performing worse than baseline."""

import sys
import torch
sys.path.insert(0, '.')
sys.path.insert(0, './nsnd_oct')

from nsnd.models.sansd import AnatomyAwareSANSD

device = 'cpu'

# Create model
model = AnatomyAwareSANSD(
    backbone_width=64,
    backbone_ckpt='outputs/nafnet_analysis_maps_w64/nafnet_best.pth',
    fusion_mode='anatomy',
).to(device)
model.eval()

# Test input
x = torch.randn(1, 1, 64, 64)

with torch.no_grad():
    out, interp = model(x, return_interpretation=True)

print("=" * 60)
print("NSAD DIAGNOSTIC")
print("=" * 60)

print("\n📊 Output Comparison:")
print(f"  Backbone output range: [{interp['backbone_out'].min():.3f}, {interp['backbone_out'].max():.3f}]")
print(f"  Symbolic output range: [{interp['symbolic_out'].min():.3f}, {interp['symbolic_out'].max():.3f}]")
print(f"  Final output range:    [{out.min():.3f}, {out.max():.3f}]")

print("\n📊 Symbolic Correction:")
correction = interp['symbolic_correction']
print(f"  Correction range: [{correction.min():.3f}, {correction.max():.3f}]")
print(f"  Correction mean:  {correction.mean():.4f}")
print(f"  Correction std:   {correction.std():.4f}")

# Check fusion gate
if interp.get('fusion_gate') is not None:
    gate = interp['fusion_gate']
    print(f"\n📊 Fusion Gate:")
    print(f"  Gate range: [{gate.min():.3f}, {gate.max():.3f}]")
    print(f"  Gate mean:  {gate.mean():.4f}")
else:
    print("\n📊 Fusion Gate: Not available")

print("\n📊 Noise Type Distribution:")
noise_type = interp['noise_type']
names = ['speckle', 'banding', 'gaussian', 'shot']
usage = noise_type.mean(dim=[0, 2, 3])
for i, name in enumerate(names):
    print(f"  {name}: {usage[i]:.3f}")

print("\n📊 Expert Outputs:")
for name, out_tensor in interp['expert_outputs'].items():
    print(f"  {name}: range=[{out_tensor.min():.3f}, {out_tensor.max():.3f}], mean={out_tensor.mean():.3f}")

# Key insight
print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

backbone_quality = interp['backbone_out'].std().item()
symbolic_quality = interp['symbolic_out'].std().item()
correction_magnitude = correction.abs().mean().item()

print(f"\n  Backbone std:        {backbone_quality:.4f}")
print(f"  Symbolic std:        {symbolic_quality:.4f}")
print(f"  Correction magnitude: {correction_magnitude:.4f}")

if correction_magnitude > 0.1:
    print("\n  ⚠️  ISSUE: Symbolic correction is too large!")
    print("     The untrained symbolic operators are overriding the good backbone.")
    print("     SOLUTION: Reduce initial correction weight or train the model first.")
