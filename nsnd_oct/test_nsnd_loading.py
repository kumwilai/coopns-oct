"""Test NSND checkpoint loading with fixed depth"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from nsnd import NSNDModel
import torch

print("Testing NSND checkpoint loading...")

model = NSNDModel(device='cpu')
checkpoint_path = Path('/home/kumwilai/OCT/nsnd_oct/checkpoints/nsnd_64x64_lightweight.pth')

print(f"\nModel gaussian_denoiser has {len(model.denoiser_bank.gaussian_denoiser.model)} layers")
print(f"Expected: 12 layers (depth=5)")

# Show layer structure
print("\nLayer structure:")
for i, layer in enumerate(model.denoiser_bank.gaussian_denoiser.model):
    print(f"  Layer {i}: {layer}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\nCheckpoint has {len(checkpoint)} parameters")
print(f"Model expects {len(model.state_dict())} parameters")

# Try loading
try:
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print(f"\n✅ SUCCESS: Loaded checkpoint!")

    if len(missing_keys) > 0:
        print(f"\nMissing keys ({len(missing_keys)}):")
        for key in missing_keys[:10]:
            print(f"  - {key}")
    else:
        print("\n✅ No missing keys!")

    if len(unexpected_keys) > 0:
        print(f"\nUnexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")
    else:
        print("\n✅ No unexpected keys!")

    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        print("\n" + "="*70)
        print("🎉 PERFECT MATCH: Checkpoint loaded without any issues!")
        print("="*70)

except Exception as e:
    print(f"\n❌ FAILED: {e}")
