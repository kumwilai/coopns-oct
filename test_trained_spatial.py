#!/usr/bin/env python3
"""
Test trained spatial refiner to verify it learned spatial variation.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from nsnd_oct.scripts.train_hybrid_nsnd_multitask import MultiTaskHybridNSND

print("=" * 80)
print("TESTING TRAINED SPATIAL REFINER")
print("=" * 80)

# Load trained checkpoint
ckpt_path = 'checkpoints/multitask_hybrid_nsnd_lambda0p08to0p05_cosine_best.pth'
if not Path(ckpt_path).exists():
    print(f"✗ Checkpoint not found: {ckpt_path}")
    print("Using untrained model for comparison...")
    ckpt_path = None

# Create model
print("\n[1/3] Loading model...")
model = MultiTaskHybridNSND(
    hybrid_analyzer_ckpt='checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth',
    device='cpu',
    use_symbolic_branch=True,
    use_log_domain_analyzer=True,
    symbolic_type='neuro',
    ns_use_neural_predicates=True,
    ns_use_neural_weights=True,
    use_base_nafnet=True,
    base_nafnet_type='full',
    base_nafnet_width=64,
    base_enc_blk_nums=[2, 2, 2],
    base_dec_blk_nums=[2, 2, 2],
    base_middle_blk_num=2,
    shared_residual=True,
    shared_trunk_width=32,
    shared_adapter_channels=96,
    shared_adapter_hidden=64,
    residual_blend_init=0.35,
    use_joint_signal_expert=True,
    joint_expert_channels=96,
    joint_mix_init=0.05,
    use_spatial_weights=True,
    spatial_feature_channels=64,
    spatial_hidden_channels=32,
)

if ckpt_path:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"✓ Loaded trained checkpoint from epoch {ckpt.get('epoch', '?')}")
else:
    # Load base NAFNet only
    base_ckpt = torch.load('outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth',
                           map_location='cpu', weights_only=False)
    model.base_denoiser.load_state_dict(base_ckpt, strict=False)
    print("✓ Loaded base NAFNet only (spatial refiner untrained)")

model.eval()

# Load a real test image
print("\n[2/3] Loading test image...")
from PIL import Image
import json

pairs_file = Path('val_pairs_duke_analysis.txt')
with open(pairs_file) as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            noisy_path = parts[0]
            break

noisy_img = np.array(Image.open(noisy_path).convert('L')) / 255.0
H, W = noisy_img.shape
crop = 256
y = (H - crop) // 2
x = (W - crop) // 2
noisy_crop = noisy_img[y:y+crop, x:x+crop]
noisy_tensor = torch.from_numpy(noisy_crop).float().unsqueeze(0).unsqueeze(0)

print(f"✓ Loaded image: {Path(noisy_path).name}")

# Run inference
print("\n[3/3] Testing spatial variation...")
with torch.no_grad():
    denoised, pred_weights, extras = model(noisy_tensor)

if 'spatial_weight_maps' in extras:
    spatial_maps = extras['spatial_weight_maps'][0]  # (4, H, W)

    print("\n✓ Spatial weight maps generated!")
    print(f"  Shape: {spatial_maps.shape}")

    # Compute spatial statistics
    noise_names = ['speckle', 'banding', 'gaussian', 'shot']
    print("\n  Spatial Variation (std per noise type):")
    has_variation = False
    for i, name in enumerate(noise_names):
        mean = spatial_maps[i].mean().item()
        std = spatial_maps[i].std().item()
        min_val = spatial_maps[i].min().item()
        max_val = spatial_maps[i].max().item()
        range_val = max_val - min_val

        # Check if there's meaningful spatial variation
        if std > 0.001 or range_val > 0.01:
            has_variation = True
            marker = "✓ SPATIAL VARIATION!"
        else:
            marker = "  (uniform)"

        print(f"    {name:8s}: mean={mean:.4f}, std={std:.4f}, range=[{min_val:.4f}, {max_val:.4f}] {marker}")

    print("\n" + "=" * 80)
    if has_variation:
        print("✓ SUCCESS: Spatial refiner learned spatial variation!")
        print("The model can now adapt denoising per-pixel based on local noise type.")
    else:
        print("⚠ Spatial refiner not yet trained (uniform weights)")
        print("After full training, spatial weights will vary across the image.")
    print("=" * 80)

    # Compare with global weights
    print("\n  Global vs Spatial Comparison:")
    for i, name in enumerate(noise_names):
        global_w = pred_weights[name].item()
        spatial_mean = spatial_maps[i].mean().item()
        diff = abs(global_w - spatial_mean)
        print(f"    {name:8s}: global={global_w:.4f}, spatial_mean={spatial_mean:.4f}, diff={diff:.4f}")

else:
    print("\n✗ ERROR: No spatial weight maps found!")

print("")
