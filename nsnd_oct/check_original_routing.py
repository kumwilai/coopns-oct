"""Check if original 27.47 dB model had diverse routing"""

import torch
import numpy as np
from pathlib import Path

checkpoint_path = Path('/home/kumwilai/OCT/nsnd_oct/checkpoints/adaptive_multihead_best.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("=" * 90)
print("ANALYZING ORIGINAL 27.47 dB MODEL ROUTING PROFILES")
print("=" * 90)

if 'routing_profiles' in checkpoint:
    routing_profiles = checkpoint['routing_profiles']

    print(f"\nFound routing profiles for {len(routing_profiles)} images")

    # Collect all routing weights
    all_weights = {
        'speckle': [],
        'banding': [],
        'gaussian': [],
        'shot': []
    }

    for img_idx, profile in routing_profiles.items():
        for noise_type in ['speckle', 'banding', 'gaussian', 'shot']:
            if noise_type in profile:
                weight = profile[noise_type]
                if isinstance(weight, torch.Tensor):
                    weight = weight.item()
                all_weights[noise_type].append(weight)

    # Analyze statistics
    print("\n" + "=" * 90)
    print("ROUTING STATISTICS FROM ORIGINAL TRAINING")
    print("=" * 90)

    for noise_type in ['speckle', 'banding', 'gaussian', 'shot']:
        weights = np.array(all_weights[noise_type])
        if len(weights) > 0:
            print(f"\n{noise_type.capitalize()}:")
            print(f"  Mean:   {weights.mean():.4f}")
            print(f"  Std:    {weights.std():.4f}")
            print(f"  Min:    {weights.min():.4f}")
            print(f"  Max:    {weights.max():.4f}")
            print(f"  Median: {np.median(weights):.4f}")

    # Check if routing is uniform (all ~0.25)
    all_means = [np.mean(all_weights[k]) for k in ['speckle', 'banding', 'gaussian', 'shot']]
    all_stds = [np.std(all_weights[k]) for k in ['speckle', 'banding', 'gaussian', 'shot']]

    avg_mean = np.mean(all_means)
    avg_std = np.mean(all_stds)

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    print(f"\nAverage weight across all heads: {avg_mean:.4f}")
    print(f"Average std deviation: {avg_std:.4f}")

    if avg_std < 0.05:
        print("\n❌ ROUTING WAS UNIFORM (all weights ~0.25)")
        print("   The original model did NOT use adaptive routing!")
    else:
        print("\n✅ ROUTING WAS DIVERSE")
        print("   The original model DID use adaptive routing!")
        print(f"   Different images had different noise type distributions")

    # Show a few examples
    print("\n" + "=" * 90)
    print("EXAMPLE ROUTING PROFILES (First 10 images)")
    print("=" * 90)

    for i, (img_idx, profile) in enumerate(list(routing_profiles.items())[:10]):
        weights_str = ", ".join([
            f"{k}={profile[k]:.3f}" if k in profile else f"{k}=N/A"
            for k in ['speckle', 'banding', 'gaussian', 'shot']
        ])
        print(f"Image {img_idx:3s}: {weights_str}")

else:
    print("\n⚠ No routing_profiles found in checkpoint")
    print("Cannot determine if original routing was diverse or uniform")

print("=" * 90)
