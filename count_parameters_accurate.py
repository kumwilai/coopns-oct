"""Accurate parameter count for NAFNet vs NSND."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))
sys.path.insert(0, str(Path(__file__).parent / "sota" / "models"))

import torch
from nafnet_fair import NAFNet
from nsnd.models.nafnet import NAFNetSmall
from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 80)
print("ACCURATE PARAMETER COUNT COMPARISON")
print("=" * 80)
print()

# 1. User's actual NAFNet baseline configuration
print("1. USER'S NAFNET BASELINE (from command)")
print("-" * 80)
nafnet_user = NAFNet(
    img_channel=1,
    width=64,
    middle_blk_num=2,
    enc_blk_nums=[2, 2, 2],
    dec_blk_nums=[2, 2, 2]
)
nafnet_user_params = count_parameters(nafnet_user)
print(f"Command: --width 64 --middle_blk_num 2")
print(f"Architecture: enc=[2,2,2], dec=[2,2,2], middle=2")
print(f"Parameters: {nafnet_user_params:,} ({nafnet_user_params/1e6:.2f}M)")
print()

# 2. User's current NSND configuration
print("2. USER'S CURRENT NSND CONFIGURATION")
print("-" * 80)

analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=True)
analyzer_params = count_parameters(analyzer)
print(f"Analyzer:               {analyzer_params:,} ({analyzer_params/1e6:.2f}M)")

# Base NAFNet-32 (uses NAFNetSmall architecture)
base_nafnet = NAFNetSmall(img_channel=1, width=32)
base_params = count_parameters(base_nafnet)
print(f"Base NAFNetSmall-32:    {base_params:,} ({base_params/1e6:.2f}M)")

# Shared trunk (NAFNetSmall with adapter_channels input)
shared_trunk = NAFNetSmall(img_channel=8, width=24)
shared_params = count_parameters(shared_trunk)
print(f"Shared Trunk-24:        {shared_params:,} ({shared_params/1e6:.2f}M)")

# Estimate adapters more accurately
# in_proj: 1->8 (3x3) = 1*8*9 + 8 = 80
# 4x adapter: (8->8 conv + 8->1 conv) with tanh
adapter_single = (8*8*9 + 8) + (8*8*9 + 8) + (8*1*9 + 1)
adapter_params = 4 * adapter_single
print(f"4x Adapters:            {adapter_params:,} ({adapter_params/1e6:.3f}M)")

# Joint expert estimate
joint_expert_params = 4608  # From previous estimate
print(f"Joint Expert:           {joint_expert_params:,} ({joint_expert_params/1e6:.3f}M)")

nsnd_denoiser_total = base_params + shared_params + adapter_params + joint_expert_params
print(f"\n  NSND Denoiser Total:  {nsnd_denoiser_total:,} ({nsnd_denoiser_total/1e6:.2f}M)")
print(f"  NSND Full:            {nsnd_denoiser_total + analyzer_params:,} ({(nsnd_denoiser_total + analyzer_params)/1e6:.2f}M)")
print()

# 3. Fair NAFNet architecture (matches default in nafnet_fair.py)
print("3. FAIR NAFNET CONFIGURATION (width=48, mentioned in code)")
print("-" * 80)
nafnet_fair = NAFNet(
    img_channel=1,
    width=48,
    middle_blk_num=2,
    enc_blk_nums=[2, 2, 2],
    dec_blk_nums=[2, 2, 2]
)
nafnet_fair_params = count_parameters(nafnet_fair)
print(f"Parameters: {nafnet_fair_params:,} ({nafnet_fair_params/1e6:.2f}M)")
print()

# 4. Recommended NSND to match user's NAFNet-64
print("4. RECOMMENDED NSND (to match NAFNet width=64)")
print("-" * 80)
print("Goal: Match ~", nafnet_user_params, "parameters")
print()

# Option A: Use full NAFNet for base (not NAFNetSmall)
print("Option A: Use full NAFNet architecture for base denoiser")
base_full = NAFNet(img_channel=1, width=40, middle_blk_num=2, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1])
base_full_params = count_parameters(base_full)

shared_full = NAFNet(img_channel=32, width=32, middle_blk_num=1, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1])
shared_full_params = count_parameters(shared_full)

adapter_large_params = 4 * ((32*24*9 + 24) + (24*24*9 + 24) + (24*1*9 + 1))
joint_large_params = 18432

total_option_a = base_full_params + shared_full_params + adapter_large_params + joint_large_params
print(f"  Base NAFNet-40:       {base_full_params:,} ({base_full_params/1e6:.2f}M)")
print(f"  Shared NAFNet-32:     {shared_full_params:,} ({shared_full_params/1e6:.2f}M)")
print(f"  Adapters (32->24ch):  {adapter_large_params:,} ({adapter_large_params/1e6:.3f}M)")
print(f"  Joint Expert:         {joint_large_params:,} ({joint_large_params/1e6:.3f}M)")
print(f"  Total:                {total_option_a:,} ({total_option_a/1e6:.2f}M)")
print(f"  Match ratio:          {100*total_option_a/nafnet_user_params:.1f}%")
print()

# Option B: Keep NAFNetSmall but increase widths dramatically
print("Option B: Use NAFNetSmall with larger widths")
base_small_large = NAFNetSmall(img_channel=1, width=56)
base_small_large_params = count_parameters(base_small_large)

shared_small_large = NAFNetSmall(img_channel=32, width=48)
shared_small_large_params = count_parameters(shared_small_large)

total_option_b = base_small_large_params + shared_small_large_params + adapter_large_params + joint_large_params
print(f"  Base NAFNetSmall-56:  {base_small_large_params:,} ({base_small_large_params/1e6:.2f}M)")
print(f"  Shared NAFNetSmall-48: {shared_small_large_params:,} ({shared_small_large_params/1e6:.2f}M)")
print(f"  Adapters (32->24ch):  {adapter_large_params:,} ({adapter_large_params/1e6:.3f}M)")
print(f"  Joint Expert:         {joint_large_params:,} ({joint_large_params/1e6:.3f}M)")
print(f"  Total:                {total_option_b:,} ({total_option_b/1e6:.2f}M)")
print(f"  Match ratio:          {100*total_option_b/nafnet_user_params:.1f}%")
print()

# Summary
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"NAFNet-64 (user's baseline):     {nafnet_user_params:,} ({nafnet_user_params/1e6:.2f}M)")
print(f"NAFNet-48 (fair baseline):       {nafnet_fair_params:,} ({nafnet_fair_params/1e6:.2f}M)")
print(f"User's NSND denoiser:            {nsnd_denoiser_total:,} ({nsnd_denoiser_total/1e6:.2f}M)")
print(f"  → {100*nsnd_denoiser_total/nafnet_user_params:.1f}% of NAFNet-64")
print(f"  → {100*nsnd_denoiser_total/nafnet_fair_params:.1f}% of NAFNet-48")
print()
print("RECOMMENDED:")
print(f"Option A (use full NAFNet arch):  {total_option_a:,} ({total_option_a/1e6:.2f}M) - {100*total_option_a/nafnet_user_params:.1f}% of baseline")
print(f"Option B (use NAFNetSmall-56/48): {total_option_b:,} ({total_option_b/1e6:.2f}M) - {100*total_option_b/nafnet_user_params:.1f}% of baseline")
print()

print("NOTES:")
print("- Your NAFNet baseline uses full UNet (enc=[2,2,2], dec=[2,2,2], middle=2)")
print("- Your NSND uses NAFNetSmall (enc=[1,1,1,1], dec=[1,1,1,1], middle=1)")
print("- This architectural mismatch explains the capacity gap")
print("- For fair comparison, NSND should use similar architecture depth")
