"""Analyze routing weights from original 27.47 dB model"""

import torch
import numpy as np
from pathlib import Path

checkpoint_path = Path('/home/kumwilai/OCT/nsnd_oct/checkpoints/adaptive_multihead_best.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*90)
print("ORIGINAL 27.47 dB MODEL - ROUTING WEIGHT ANALYSIS")
print("="*90)

routing_profiles = checkpoint['routing_profiles']

# Extract weights for each noise type
weights_dict = {}
for noise_type in ['speckle', 'banding', 'gaussian', 'shot']:
    weights_dict[noise_type] = np.array(routing_profiles[noise_type])

num_images = len(weights_dict['speckle'])
print(f"\nAnalyzing {num_images} test images\n")

# Per noise-type statistics
print("="*90)
print("ROUTING STATISTICS PER NOISE TYPE")
print("="*90)

for noise_type in ['speckle', 'banding', 'gaussian', 'shot']:
    weights = weights_dict[noise_type]
    print(f"\n{noise_type.upper()}:")
    print(f"  Mean:   {weights.mean():.6f}")
    print(f"  Std:    {weights.std():.6f}")
    print(f"  Min:    {weights.min():.6f}")
    print(f"  Max:    {weights.max():.6f}")
    print(f"  Median: {np.median(weights):.6f}")

# Check normalization (should sum to ~1.0)
print("\n" + "="*90)
print("NORMALIZATION CHECK")
print("="*90)

for i in range(min(5, num_images)):
    total = sum(weights_dict[k][i] for k in ['speckle', 'banding', 'gaussian', 'shot'])
    weights_str = ", ".join([
        f"{k[0].upper()}={weights_dict[k][i]:.3f}"
        for k in ['speckle', 'banding', 'gaussian', 'shot']
    ])
    print(f"Image {i:2d}: [{weights_str}] → Sum = {total:.6f}")

# Overall statistics
print("\n" + "="*90)
print("VERDICT: WAS ROUTING UNIFORM OR DIVERSE?")
print("="*90)

all_means = [weights_dict[k].mean() for k in ['speckle', 'banding', 'gaussian', 'shot']]
all_stds = [weights_dict[k].std() for k in ['speckle', 'banding', 'gaussian', 'shot']]

print(f"\nAverage weights:")
for k in ['speckle', 'banding', 'gaussian', 'shot']:
    print(f"  {k.capitalize():12s}: {weights_dict[k].mean():.6f}")

# Check if close to uniform (0.25, 0.25, 0.25, 0.25)
distances_from_uniform = [abs(m - 0.25) for m in all_means]
max_distance = max(distances_from_uniform)

print(f"\nMax distance from uniform (0.25): {max_distance:.6f}")

if max_distance < 0.02:
    print("\n❌ ROUTING WAS ESSENTIALLY UNIFORM")
    print("   All heads weighted ~0.25")
else:
    print("\n✅ ROUTING WAS NOT UNIFORM")
    print(f"   Heads had different average weights!")
    print(f"\n   Dominant head: {['speckle', 'banding', 'gaussian', 'shot'][np.argmax(all_means)]}")
    print(f"   Least used head: {['speckle', 'banding', 'gaussian', 'shot'][np.argmin(all_means)]}")

# Variance across images (routing diversity)
print("\n" + "="*90)
print("ROUTING DIVERSITY ACROSS IMAGES")
print("="*90)

avg_cross_image_std = np.mean([weights_dict[k].std() for k in ['speckle', 'banding', 'gaussian', 'shot']])

print(f"\nAverage std deviation across images: {avg_cross_image_std:.6f}")

if avg_cross_image_std > 0.05:
    print("✅ ROUTING WAS ADAPTIVE (different routing for different images)")
else:
    print("❌ ROUTING WAS STATIC (same routing for all images)")

print("="*90)
