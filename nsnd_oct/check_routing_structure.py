"""Check routing_profiles structure"""

import torch
from pathlib import Path

checkpoint_path = Path('/home/kumwilai/OCT/nsnd_oct/checkpoints/adaptive_multihead_best.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*90)
print("CHECKPOINT STRUCTURE ANALYSIS")
print("="*90)

print("\nTop-level keys:")
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, dict):
        print(f"  {key}: dict with {len(value)} entries")
    elif isinstance(value, (int, float, torch.Tensor)):
        print(f"  {key}: {type(value).__name__} = {value}")
    else:
        print(f"  {key}: {type(value).__name__}")

if 'routing_profiles' in checkpoint:
    print("\n" + "="*90)
    print("ROUTING_PROFILES STRUCTURE")
    print("="*90)

    routing_profiles = checkpoint['routing_profiles']
    print(f"\nType: {type(routing_profiles)}")
    print(f"Keys: {list(routing_profiles.keys())}")

    # Examine first entry
    for key in list(routing_profiles.keys())[:2]:
        print(f"\nrouting_profiles['{key}']:")
        value = routing_profiles[key]
        print(f"  Type: {type(value)}")

        if isinstance(value, dict):
            print(f"  Keys: {list(value.keys())[:5]}")
            for k2 in list(value.keys())[:3]:
                v2 = value[k2]
                print(f"    {k2}: {type(v2).__name__} = {v2}")
        elif isinstance(value, torch.Tensor):
            print(f"  Shape: {value.shape}")
            print(f"  Values: {value}")
        else:
            print(f"  Value: {value}")
