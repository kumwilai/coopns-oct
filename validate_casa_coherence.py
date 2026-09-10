#!/usr/bin/env python3
"""
CRITICAL VALIDATION: Check if CASA is learning coherent/incoherent decomposition.

If CASA is working:
- Coherence maps should be spatially varying (not uniform)
- Should correlate with local speckle statistics
- Should differ between Gaussian vs non-Gaussian noise

If CASA is broken:
- Maps will be uniform or random
- No correlation with image statistics
- Performance mystery is solved (it's just spatial attention)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from adaptive_oct_denoise import (
    PairedOCTDataset, build_model, device, resize_to
)

print("=" * 80)
print("CASA COHERENCE DECOMPOSITION VALIDATION")
print("=" * 80)

# Load model
print("\n1. Loading model...")
model = build_model(
    base_channels=48,
    residual_mode=True,
    adapter_type="casa",
    backbone_type="noise2void"
)

checkpoint = torch.load("checkpoints/casa_n2v_residual/best_model_ema.pth", map_location=device)
if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device).eval()
print("✓ Model loaded")

# Check if model returns auxiliary outputs
print("\n2. Checking if model returns coherence maps...")
test_input = torch.randn(1, 1, 64, 64).to(device)

try:
    with torch.no_grad():
        # Try to get auxiliary outputs
        output = model(test_input)

    print("⚠️ Model only returns output tensor")
    print("Checking model architecture for CASA adapter...")

    # Check if CASA adapter exists in model
    has_casa = False
    for name, module in model.named_modules():
        if 'casa' in name.lower() or 'adapter' in name.lower():
            print(f"  Found: {name} ({type(module).__name__})")
            has_casa = True

    if not has_casa:
        print("❌ CRITICAL: No CASA adapter found in model!")
        print("   Model might be vanilla N2V without CASA")
        exit(1)
    else:
        print("⚠️ CASA adapter exists but doesn't return auxiliary outputs")
        print("   Need to modify forward pass to return coherence maps")

        # Try to manually extract from adapter
        print("\n3. Attempting to manually extract coherence maps...")

        # Forward pass and hook into adapter
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    activations[name] = output
                else:
                    activations[name] = output
            return hook

        # Register hooks on adapter modules
        hooks = []
        for name, module in model.named_modules():
            if 'casa' in name.lower() or 'adapter' in name.lower():
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)

        with torch.no_grad():
            _ = model(test_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if activations:
            print(f"✓ Captured {len(activations)} adapter activations")
            for name, act in activations.items():
                if isinstance(act, tuple):
                    print(f"  {name}: tuple of {len(act)} tensors")
                    for i, a in enumerate(act):
                        if isinstance(a, torch.Tensor):
                            print(f"    [{i}]: shape {a.shape}")
                else:
                    print(f"  {name}: shape {act.shape}")
        else:
            print("❌ No activations captured")

except Exception as e:
    print(f"❌ Error during forward pass: {e}")
    exit(1)

# Load test images from different noise types
print("\n4. Loading test images (Gaussian, Moderate, Heavy)...")
transform = resize_to((64, 64))
dataset = PairedOCTDataset('val_pairs_universal.txt', transform=transform)

# Representative samples: Batch 1 (Gaussian), Batch 36 (Moderate), Batch 69 (Heavy)
test_indices = [0, 560, 1088]
test_labels = ['Gaussian (Batch 1)', 'Moderate (Batch 36)', 'Heavy Gamma (Batch 69)']

print("\n5. Analyzing adapter behavior on different noise types...")
print("=" * 80)

for idx, label in zip(test_indices, test_labels):
    noisy, clean = dataset[idx]
    noisy_input = noisy.unsqueeze(0).to(device)

    print(f"\n{label}:")
    print("-" * 80)

    # Forward pass
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    hooks = []
    for name, module in model.named_modules():
        if 'casa' in name.lower() or 'adapter' in name.lower():
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)

    with torch.no_grad():
        output = model(noisy_input)

    for hook in hooks:
        hook.remove()

    # Analyze activations
    if activations:
        for name, act in activations.items():
            if isinstance(act, torch.Tensor) and len(act.shape) == 4:
                # Spatial activation map
                act_mean = act.mean().item()
                act_std = act.std().item()
                act_min = act.min().item()
                act_max = act.max().item()

                print(f"  {name}:")
                print(f"    Mean: {act_mean:.4f}, Std: {act_std:.4f}")
                print(f"    Range: [{act_min:.4f}, {act_max:.4f}]")

                # Check if it's just uniform (bad sign)
                if act_std < 0.01:
                    print(f"    ⚠️ WARNING: Nearly uniform - not adapting spatially!")
                else:
                    print(f"    ✓ Spatially varying")
    else:
        print("  ⚠️ No activations captured")

# Compute statistics on image to see if adapter should adapt
print("\n6. Computing expected coherence indicators...")
print("=" * 80)

for idx, label in zip(test_indices, test_labels):
    noisy, clean = dataset[idx]
    noisy_np = noisy.squeeze().numpy()

    print(f"\n{label}:")
    print("-" * 80)

    # Compute local coefficient of variation (CV)
    # High CV = structured speckle = should use coherent processing
    # Low CV = additive noise = should use incoherent processing

    from scipy.ndimage import uniform_filter

    window_size = 11
    mean_local = uniform_filter(noisy_np, size=window_size)
    mean_sq_local = uniform_filter(noisy_np**2, size=window_size)
    var_local = mean_sq_local - mean_local**2
    std_local = np.sqrt(np.maximum(var_local, 0))
    cv_local = std_local / (mean_local + 1e-8)

    cv_mean = cv_local.mean()
    cv_std = cv_local.std()

    print(f"  Local Coefficient of Variation (CV):")
    print(f"    Mean: {cv_mean:.4f}, Std: {cv_std:.4f}")
    print(f"    Range: [{cv_local.min():.4f}, {cv_local.max():.4f}]")

    # Expected coherence interpretation
    if cv_mean > 0.5:
        print(f"    → High CV: Expect CASA to favor coherent processing")
    elif cv_mean < 0.3:
        print(f"    → Low CV: Expect CASA to favor incoherent processing")
    else:
        print(f"    → Moderate CV: Expect CASA to mix both")

    # Compute SNR estimate
    # High SNR = cleaner = less adaptation needed
    # Low SNR = noisier = more adaptation needed
    signal_estimate = uniform_filter(noisy_np, size=31)
    noise_estimate = noisy_np - signal_estimate
    snr_db = 10 * np.log10((signal_estimate**2).mean() / (noise_estimate**2).mean() + 1e-8)

    print(f"  Estimated SNR: {snr_db:.2f} dB")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY:")
print("=" * 80)

print("""
Expected behavior if CASA is working correctly:

1. ✓ Adapter activations should be spatially varying (std > 0.01)
2. ✓ Activations should differ between Gaussian vs non-Gaussian images
3. ✓ High CV regions → higher coherent processing
4. ✓ Low CV regions → higher incoherent processing

If ANY of these fail → CASA is not learning physics-based decomposition.

Next steps:
- If CASA is working: Focus on interpretability experiments (your unique contribution)
- If CASA is broken: Debug architecture or pivot to simpler spatial attention
""")

print("\n" + "=" * 80)
print("To visualize adapter behavior, run:")
print("  python visualize_casa_maps.py")
print("=" * 80)
