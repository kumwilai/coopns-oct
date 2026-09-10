#!/usr/bin/env python3
"""
Visualize CASA coherent/incoherent decomposition maps.

Critical validation: Do decomposition weights correlate with image statistics?
- High CV (structured speckle) → High coherent weight
- Low CV (additive noise) → High incoherent weight
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr
from adaptive_oct_denoise import (
    PairedOCTDataset, build_model, device, resize_to, compute_psnr
)

print("=" * 80)
print("CASA COHERENCE DECOMPOSITION VISUALIZATION")
print("=" * 80)

# Load model
print("\nLoading model...")
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

# Load test images
print("\nLoading test images...")
transform = resize_to((64, 64))
dataset = PairedOCTDataset('val_pairs_universal.txt', transform=transform)

# Test on representative samples
test_indices = [0, 560, 1088]  # Gaussian, Moderate, Heavy
test_labels = ['Gaussian (Batch 1)', 'Moderate (Batch 36)', 'Heavy Gamma (Batch 69)']

# Setup figure
fig, axes = plt.subplots(len(test_indices), 6, figsize=(20, 10))
fig.suptitle('CASA Coherent/Incoherent Decomposition Analysis', fontsize=16, fontweight='bold')

correlations = []

for row_idx, (img_idx, label) in enumerate(zip(test_indices, test_labels)):
    print(f"\nProcessing {label}...")

    noisy, clean = dataset[img_idx]
    noisy_input = noisy.unsqueeze(0).to(device)

    # Forward pass and capture decomposition maps
    captured_maps = {'decomposition': None}

    def get_decomposition(name):
        def hook(model, input, output):
            if 'decomposition_net' in name and not any(x in name for x in ['0', '1', '2', '3']):
                captured_maps['decomposition'] = output.detach()
        return hook

    # Register hook
    hook_handle = None
    for name, module in model.named_modules():
        if name == 'adapter.decomposition_net':
            hook_handle = module.register_forward_hook(get_decomposition(name))
            break

    # Forward pass
    with torch.no_grad():
        denoised = model(noisy_input)

    # Remove hook
    if hook_handle:
        hook_handle.remove()

    # Extract maps
    decomposition_maps = captured_maps['decomposition']
    if decomposition_maps is not None and decomposition_maps.shape[1] == 2:
        coherent_map = decomposition_maps[0, 0].cpu().numpy()  # Channel 0
        incoherent_map = decomposition_maps[0, 1].cpu().numpy()  # Channel 1
        print(f"  ✓ Extracted decomposition maps: {decomposition_maps.shape}")
    else:
        print(f"  ⚠️ Could not extract decomposition maps!")
        continue

    # Compute local CV (ground truth indicator)
    noisy_np = noisy.squeeze().cpu().numpy()

    window_size = 11
    mean_local = uniform_filter(noisy_np, size=window_size)
    mean_sq_local = uniform_filter(noisy_np**2, size=window_size)
    var_local = mean_sq_local - mean_local**2
    std_local = np.sqrt(np.maximum(var_local, 0))
    cv_local = std_local / (mean_local + 1e-8)

    # Compute PSNR
    psnr = compute_psnr(denoised, clean.unsqueeze(0).to(device))

    # Compute correlation between coherent weight and CV
    corr_coherent_cv, p_value = pearsonr(coherent_map.flatten(), cv_local.flatten())
    correlations.append((label, corr_coherent_cv, p_value))

    print(f"  Coherent-CV correlation: r={corr_coherent_cv:.3f}, p={p_value:.2e}")
    print(f"  Coherent weight range: [{coherent_map.min():.3f}, {coherent_map.max():.3f}]")
    print(f"  Incoherent weight range: [{incoherent_map.min():.3f}, {incoherent_map.max():.3f}]")
    print(f"  Denoising PSNR: {psnr:.2f} dB")

    # Plot
    # Column 0: Noisy input
    im0 = axes[row_idx, 0].imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
    axes[row_idx, 0].set_title(f'Noisy\n{label}', fontsize=10)
    axes[row_idx, 0].axis('off')

    # Column 1: Denoised output
    denoised_np = denoised.squeeze().cpu().numpy()
    im1 = axes[row_idx, 1].imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
    axes[row_idx, 1].set_title(f'Denoised\nPSNR={psnr:.1f} dB', fontsize=10)
    axes[row_idx, 1].axis('off')

    # Column 2: Coherent weight map
    im2 = axes[row_idx, 2].imshow(coherent_map, cmap='hot', vmin=0, vmax=1)
    axes[row_idx, 2].set_title(f'Coherent Weight\n(mean={coherent_map.mean():.2f})', fontsize=10)
    axes[row_idx, 2].axis('off')
    plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046)

    # Column 3: Incoherent weight map
    im3 = axes[row_idx, 3].imshow(incoherent_map, cmap='hot', vmin=0, vmax=1)
    axes[row_idx, 3].set_title(f'Incoherent Weight\n(mean={incoherent_map.mean():.2f})', fontsize=10)
    axes[row_idx, 3].axis('off')
    plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046)

    # Column 4: Local CV (expected coherence indicator)
    im4 = axes[row_idx, 4].imshow(cv_local, cmap='viridis', vmin=0, vmax=1.5)
    axes[row_idx, 4].set_title(f'Local CV\n(mean={cv_local.mean():.2f})', fontsize=10)
    axes[row_idx, 4].axis('off')
    plt.colorbar(im4, ax=axes[row_idx, 4], fraction=0.046)

    # Column 5: Scatter plot (Coherent vs CV)
    # Downsample for plotting
    stride = 2
    coherent_flat = coherent_map[::stride, ::stride].flatten()
    cv_flat = cv_local[::stride, ::stride].flatten()

    axes[row_idx, 5].scatter(cv_flat, coherent_flat, alpha=0.3, s=1)
    axes[row_idx, 5].set_xlabel('Local CV', fontsize=9)
    axes[row_idx, 5].set_ylabel('Coherent Weight', fontsize=9)
    axes[row_idx, 5].set_title(f'Correlation\nr={corr_coherent_cv:.3f}', fontsize=10)
    axes[row_idx, 5].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(cv_flat, coherent_flat, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(cv_flat.min(), cv_flat.max(), 100)
    axes[row_idx, 5].plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('results/casa_decomposition_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: results/casa_decomposition_analysis.png")

# Summary
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS SUMMARY:")
print("=" * 80)

for label, corr, p_val in correlations:
    significance = "✓ Significant" if p_val < 0.001 else "⚠️ Not significant"
    interpretation = ""

    if corr > 0.5:
        interpretation = "✓ Strong positive (CASA learns physics!)"
    elif corr > 0.3:
        interpretation = "⚠️ Moderate positive (partial learning)"
    elif corr > 0:
        interpretation = "⚠️ Weak positive (questionable)"
    else:
        interpretation = "❌ Negative/no correlation (CASA not learning!)"

    print(f"{label}:")
    print(f"  Coherent-CV correlation: r={corr:.3f}, p={p_val:.2e}")
    print(f"  {significance}")
    print(f"  Interpretation: {interpretation}")
    print()

print("=" * 80)
print("\nEXPECTATIONS:")
print("If CASA is learning physics-based decomposition:")
print("  ✓ r > 0.5: Strong correlation between coherent weight and local CV")
print("  ✓ High CV regions → High coherent weight (structured speckle)")
print("  ✓ Low CV regions → High incoherent weight (additive noise)")
print("\nIf r < 0.3: CASA is NOT learning meaningful decomposition")
print("  → May just be learning spatial attention without physics")
print("=" * 80)

# Additional analysis: Check if weights sum to 1
print("\n" + "=" * 80)
print("DECOMPOSITION VALIDITY CHECK:")
print("=" * 80)

for img_idx, label in zip(test_indices, test_labels):
    noisy, _ = dataset[img_idx]
    noisy_input = noisy.unsqueeze(0).to(device)

    # Get decomposition maps
    captured_maps2 = {'decomposition': None}

    def get_decomposition(name):
        def hook(model, input, output):
            if 'decomposition_net' in name and not any(x in name for x in ['0', '1', '2', '3']):
                captured_maps2['decomposition'] = output.detach()
        return hook

    hook_handle = None
    for name, module in model.named_modules():
        if name == 'adapter.decomposition_net':
            hook_handle = module.register_forward_hook(get_decomposition(name))
            break

    with torch.no_grad():
        _ = model(noisy_input)

    if hook_handle:
        hook_handle.remove()

    decomposition_maps = captured_maps2['decomposition']
    if decomposition_maps is not None:
        coherent = decomposition_maps[0, 0].cpu().numpy()
        incoherent = decomposition_maps[0, 1].cpu().numpy()
        weight_sum = coherent + incoherent

        sum_mean = weight_sum.mean()
        sum_std = weight_sum.std()

        print(f"{label}:")
        print(f"  Coherent + Incoherent sum: {sum_mean:.4f} ± {sum_std:.4f}")

        if abs(sum_mean - 1.0) < 0.01 and sum_std < 0.01:
            print(f"  ✓ Weights properly normalized (sum ≈ 1.0)")
        else:
            print(f"  ⚠️ Weights not properly normalized!")

print("=" * 80)
