"""Count parameters for NAFNet vs NSND to ensure fair comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))

import torch
import torch.nn as nn
from nsnd.models.nafnet import NAFNetSmall
from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer

# Helper function
def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

print("=" * 80)
print("PARAMETER COUNT COMPARISON")
print("=" * 80)
print()

# 1. Standalone NAFNet (baseline from user's command)
print("1. STANDALONE NAFNET (Baseline)")
print("-" * 80)
nafnet_baseline = NAFNetSmall(img_channel=1, width=64)
nafnet_params = count_parameters(nafnet_baseline)
print(f"NAFNet-64:     {nafnet_params:,} parameters ({nafnet_params/1e6:.2f}M)")
print()

# 2. User's current NSND configuration
print("2. USER'S CURRENT NSND CONFIGURATION")
print("-" * 80)

# Analyzer
analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=True)
analyzer_params = count_parameters(analyzer)
print(f"Analyzer:      {analyzer_params:,} parameters ({analyzer_params/1e6:.2f}M)")

# Base NAFNet (user's config: width=32)
base_nafnet = NAFNetSmall(img_channel=1, width=32)
base_params = count_parameters(base_nafnet)
print(f"Base NAFNet-32: {base_params:,} parameters ({base_params/1e6:.2f}M)")

# Shared trunk (user's config: width=24)
shared_trunk = NAFNetSmall(img_channel=8, width=24)  # adapter_channels=8
shared_params = count_parameters(shared_trunk)
print(f"Shared Trunk-24: {shared_params:,} parameters ({shared_params/1e6:.2f}M)")

# 4x adapters (8 channels -> 8 hidden -> 1 output)
adapter_params = 4 * (8*8*3*3 + 8 + 8*8*3*3 + 8 + 8*1*3*3 + 1)
print(f"4x Adapters:    {adapter_params:,} parameters ({adapter_params/1e6:.3f}M)")

# Joint expert (simplified estimate)
joint_expert_params = 16 * 16 * 3 * 3 * 2  # Rough estimate
print(f"Joint Expert:   {joint_expert_params:,} parameters ({joint_expert_params/1e6:.3f}M)")

nsnd_total = base_params + shared_params + adapter_params + joint_expert_params
print(f"\nNSND Denoiser Total: {nsnd_total:,} parameters ({nsnd_total/1e6:.2f}M)")
print(f"NSND Full (with Analyzer): {nsnd_total + analyzer_params:,} parameters ({(nsnd_total + analyzer_params)/1e6:.2f}M)")
print()

# 3. Fair comparison - match parameter count
print("3. RECOMMENDED NSND CONFIGURATION (Parameter-Matched)")
print("-" * 80)

# Calculate required widths to match NAFNet-64
# NAFNet-64 ≈ 0.75M params
# We need: base + shared + adapters ≈ 0.75M
# With base=48, shared_trunk=40, adapters=32

base_fair = NAFNetSmall(img_channel=1, width=48)
base_fair_params = count_parameters(base_fair)
print(f"Base NAFNet-48: {base_fair_params:,} parameters ({base_fair_params/1e6:.2f}M)")

shared_fair = NAFNetSmall(img_channel=32, width=40)  # adapter_channels=32
shared_fair_params = count_parameters(shared_fair)
print(f"Shared Trunk-40: {shared_fair_params:,} parameters ({shared_fair_params/1e6:.2f}M)")

# 4x larger adapters (32 channels -> 24 hidden -> 1 output)
adapter_fair_params = 4 * (32*24*3*3 + 24 + 24*24*3*3 + 24 + 24*1*3*3 + 1)
print(f"4x Adapters (32ch): {adapter_fair_params:,} parameters ({adapter_fair_params/1e6:.3f}M)")

joint_fair_params = 32 * 32 * 3 * 3 * 2
print(f"Joint Expert (32ch): {joint_fair_params:,} parameters ({joint_fair_params/1e6:.3f}M)")

nsnd_fair_total = base_fair_params + shared_fair_params + adapter_fair_params + joint_fair_params
print(f"\nNSND Denoiser Total: {nsnd_fair_total:,} parameters ({nsnd_fair_total/1e6:.2f}M)")
print(f"NSND Full (with Analyzer): {nsnd_fair_total + analyzer_params:,} parameters ({(nsnd_fair_total + analyzer_params)/1e6:.2f}M)")
print()

# Comparison
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"NAFNet-64 baseline:        {nafnet_params:,} params ({nafnet_params/1e6:.2f}M)")
print(f"User's NSND denoiser:      {nsnd_total:,} params ({nsnd_total/1e6:.2f}M) - {100*nsnd_total/nafnet_params:.1f}% of baseline")
print(f"Recommended NSND denoiser: {nsnd_fair_total:,} params ({nsnd_fair_total/1e6:.2f}M) - {100*nsnd_fair_total/nafnet_params:.1f}% of baseline")
print()

# Additional analysis
print("PARAMETER BUDGET BREAKDOWN (Recommended NSND):")
print(f"  Base denoiser:    {100*base_fair_params/nsnd_fair_total:.1f}%")
print(f"  Shared trunk:     {100*shared_fair_params/nsnd_fair_total:.1f}%")
print(f"  Adapters:         {100*adapter_fair_params/nsnd_fair_total:.1f}%")
print(f"  Joint expert:     {100*joint_fair_params/nsnd_fair_total:.1f}%")
print()

print("NOTES FOR PUBLICATION:")
print("- NAFNet baseline uses ALL parameters for denoising")
print("- NSND splits capacity: base denoising + noise-specific refinement")
print("- Fair comparison requires matching denoiser parameters (excluding analyzer)")
print("- Analyzer parameters should be reported separately as it's shared across tasks")
