"""Debug checkpoint keys to understand architecture mismatch"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from nsnd import NSNDModel
from nsnd.models.adaptive_multihead_refinement import AdaptiveMultiHeadRefinement

checkpoint_dir = Path('/home/kumwilai/OCT/nsnd_oct/checkpoints')
device = 'cpu'

# Load checkpoint
checkpoint_path = checkpoint_dir / 'adaptive_multihead_best.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

print("=" * 90)
print("CHECKPOINT STRUCTURE ANALYSIS")
print("=" * 90)

if isinstance(checkpoint, dict):
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print(f"\nCheckpoint has {len(state_dict)} parameter keys")
    print("\nAll checkpoint keys:")
    for i, key in enumerate(sorted(state_dict.keys())):
        shape = state_dict[key].shape
        print(f"  {i+1:2d}. {key:60s} {str(shape):30s}")

    # Metadata
    if 'psnr' in checkpoint:
        print(f"\n{'='*90}")
        print("METADATA")
        print(f"{'='*90}")
        print(f"PSNR: {checkpoint['psnr']:.6f} dB")
        print(f"SSIM: {checkpoint['ssim']:.6f}")
        if 'improvement' in checkpoint:
            print(f"Improvement: {checkpoint['improvement']:.6f} dB")

print(f"\n{'='*90}")
print("CURRENT MODEL STRUCTURE")
print(f"{'='*90}")

# Create current model
nsnd_model = NSNDModel(device=device, gaussian_depth=5)
model = AdaptiveMultiHeadRefinement(
    nsnd_symbolic_analyzer=nsnd_model.symbolic_analyzer,
    channels=16,
    device=device,
    dropout=0.0
)

current_state = model.state_dict()
print(f"\nCurrent model has {len(current_state)} parameter keys")
print("\nAll current model keys:")
for i, key in enumerate(sorted(current_state.keys())):
    shape = current_state[key].shape
    print(f"  {i+1:2d}. {key:60s} {str(shape):30s}")

# Find mismatches
print(f"\n{'='*90}")
print("KEY COMPARISON")
print(f"{'='*90}")

checkpoint_keys = set(state_dict.keys())
current_keys = set(current_state.keys())

missing_in_checkpoint = current_keys - checkpoint_keys
unexpected_in_checkpoint = checkpoint_keys - current_keys

print(f"\nMissing in checkpoint ({len(missing_in_checkpoint)} keys):")
for key in sorted(missing_in_checkpoint):
    shape = current_state[key].shape
    print(f"  - {key:60s} {str(shape):30s}")

print(f"\nUnexpected in checkpoint ({len(unexpected_in_checkpoint)} keys):")
for key in sorted(unexpected_in_checkpoint):
    shape = state_dict[key].shape
    print(f"  - {key:60s} {str(shape):30s}")

# Try loading with strict=False
print(f"\n{'='*90}")
print("LOADING TEST")
print(f"{'='*90}")

missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"\nLoading with strict=False:")
print(f"  Missing keys: {len(missing_keys)}")
print(f"  Unexpected keys: {len(unexpected_keys)}")

if len(missing_keys) > 0:
    print(f"\nMissing keys details:")
    for key in sorted(missing_keys):
        print(f"  - {key}")

if len(unexpected_keys) > 0:
    print(f"\nUnexpected keys details:")
    for key in sorted(unexpected_keys):
        print(f"  - {key}")
