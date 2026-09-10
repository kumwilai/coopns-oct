#!/usr/bin/env python3
"""
Quick Comparison: Spatial Weights vs Global Weights

Trains both models for 5 epochs and compares:
1. PSNR on uniform noise
2. PSNR on heterogeneous noise (4 quadrants)
3. Spatial adaptation capability
4. Training time
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from nsnd_oct.scripts.train_hybrid_nsnd_multitask import (
    MultiTaskHybridNSND,
    multitask_loss,
)

# Noise generation
def add_speckle_noise(clean, k=4.0):
    noise = np.random.gamma(k, 1.0/k, clean.shape)
    return np.clip(clean * noise, 0, 1)

def add_gaussian_noise(clean, sigma=0.1):
    noise = np.random.normal(0, sigma, clean.shape)
    return np.clip(clean + noise, 0, 1)

def add_banding_noise(clean, num_bands=20, intensity=0.15):
    H, W = clean.shape
    bands = np.zeros_like(clean)
    band_height = H // num_bands
    for i in range(num_bands):
        y_start = i * band_height
        y_end = min((i + 1) * band_height, H)
        band_value = np.random.uniform(-intensity, intensity)
        bands[y_start:y_end, :] = band_value
    return np.clip(clean + bands, 0, 1)

def add_shot_noise(clean, scale=0.05):
    scaled = clean * 255.0
    noisy = np.random.poisson(scaled + 1e-6) / 255.0
    noisy = noisy * (1.0 + np.random.normal(0, scale, clean.shape))
    return np.clip(noisy, 0, 1)

def create_clean_image(size=128):
    """Create synthetic OCT-like clean image."""
    clean = np.zeros((size, size), dtype=np.float32)
    for i in range(5):
        y_center = size // 6 * (i + 1)
        thickness = 15
        for y in range(max(0, y_center - thickness//2), min(size, y_center + thickness//2)):
            clean[y, :] = 0.6 + 0.3 * np.random.rand()
    clean += 0.1 * np.random.rand(size, size)
    return np.clip(clean, 0, 1)


class UniformNoiseDataset(Dataset):
    """Dataset with UNIFORM noise (same type everywhere)."""

    def __init__(self, num_samples=50, size=128):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean = create_clean_image(self.size)

        # Random uniform noise type
        noise_type = np.random.randint(0, 4)
        if noise_type == 0:
            noisy = add_speckle_noise(clean, k=3.0)
            weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        elif noise_type == 1:
            noisy = add_banding_noise(clean, num_bands=15, intensity=0.2)
            weights = torch.tensor([0.0, 1.0, 0.0, 0.0])
        elif noise_type == 2:
            noisy = add_gaussian_noise(clean, sigma=0.15)
            weights = torch.tensor([0.0, 0.0, 1.0, 0.0])
        else:
            noisy = add_shot_noise(clean, scale=0.08)
            weights = torch.tensor([0.0, 0.0, 0.0, 1.0])

        return {
            'noisy': torch.from_numpy(noisy).float().unsqueeze(0),
            'clean': torch.from_numpy(clean).float().unsqueeze(0),
            'weights': weights,
        }


class HeterogeneousNoiseDataset(Dataset):
    """Dataset with HETEROGENEOUS noise (4 quadrants)."""

    def __init__(self, num_samples=50, size=128):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean = create_clean_image(self.size)

        # Add different noise to each quadrant
        noisy = np.zeros_like(clean)
        half = self.size // 2

        noisy[:half, :half] = add_speckle_noise(clean[:half, :half], k=3.0)
        noisy[:half, half:] = add_banding_noise(clean[:half, half:], num_bands=10, intensity=0.2)
        noisy[half:, :half] = add_gaussian_noise(clean[half:, :half], sigma=0.15)
        noisy[half:, half:] = add_shot_noise(clean[half:, half:], scale=0.08)

        # Use dominant noise (speckle) as label for simplicity
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])

        return {
            'noisy': torch.from_numpy(noisy).float().unsqueeze(0),
            'clean': torch.from_numpy(clean).float().unsqueeze(0),
            'weights': weights,
        }


def train_model(use_spatial_weights, num_epochs=5):
    """Train model and return results."""

    print(f"\n{'='*80}")
    print(f"Training {'WITH' if use_spatial_weights else 'WITHOUT'} Spatial Weights")
    print(f"{'='*80}")

    # Create model
    model = MultiTaskHybridNSND(
        hybrid_analyzer_ckpt='checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth',
        device='cpu',
        use_symbolic_branch=True,
        use_log_domain_analyzer=True,
        symbolic_type='neuro',
        ns_use_neural_predicates=True,
        ns_use_neural_weights=True,
        use_base_nafnet=False,  # Disable for speed
        shared_residual=True,
        shared_trunk_width=16,
        shared_adapter_channels=32,
        shared_adapter_hidden=16,
        residual_blend_init=0.35,
        use_spatial_weights=use_spatial_weights,  # KEY DIFFERENCE!
        spatial_feature_channels=32,
        spatial_hidden_channels=16,
    )

    # Create datasets
    train_dataset = UniformNoiseDataset(num_samples=50, size=128)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    start_time = time.time()
    losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        batches = 0

        for batch in train_loader:
            noisy = batch['noisy']
            clean = batch['clean']
            true_weights = batch['weights']

            denoised, pred_weights, extras = model(noisy)

            loss, denoise_loss, interp_loss, _ = multitask_loss(
                denoised, clean, pred_weights, true_weights, lambda_interp=0.05
            )

            if loss.item() > 100:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        losses.append(avg_loss)
        print(f"  Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time

    return model, losses, training_time


def evaluate_model(model, use_spatial_weights):
    """Evaluate on uniform and heterogeneous noise."""

    print(f"\n[Evaluating {'SPATIAL' if use_spatial_weights else 'GLOBAL'} model...]")

    model.eval()

    # Test on uniform noise
    uniform_dataset = UniformNoiseDataset(num_samples=20, size=128)
    uniform_psnrs = []

    for i in range(len(uniform_dataset)):
        sample = uniform_dataset[i]
        noisy = sample['noisy'].unsqueeze(0)
        clean = sample['clean'].unsqueeze(0)

        with torch.no_grad():
            denoised, _, _ = model(noisy)

        psnr = -10 * np.log10(((denoised - clean) ** 2).mean().item())
        uniform_psnrs.append(psnr)

    # Test on heterogeneous noise
    hetero_dataset = HeterogeneousNoiseDataset(num_samples=20, size=128)
    hetero_psnrs = []

    for i in range(len(hetero_dataset)):
        sample = hetero_dataset[i]
        noisy = sample['noisy'].unsqueeze(0)
        clean = sample['clean'].unsqueeze(0)

        with torch.no_grad():
            denoised, _, extras = model(noisy)

        psnr = -10 * np.log10(((denoised - clean) ** 2).mean().item())
        hetero_psnrs.append(psnr)

    # Check spatial adaptation
    spatial_adaptation = False
    if 'spatial_weight_maps' in extras:
        spatial_maps = extras['spatial_weight_maps'][0]
        # Check if weights vary spatially (std > 0.01)
        std_per_type = spatial_maps.std(dim=(1, 2))
        spatial_adaptation = (std_per_type > 0.01).any().item()

    return {
        'uniform_psnr': np.mean(uniform_psnrs),
        'uniform_psnr_std': np.std(uniform_psnrs),
        'hetero_psnr': np.mean(hetero_psnrs),
        'hetero_psnr_std': np.std(hetero_psnrs),
        'spatial_adaptation': spatial_adaptation,
    }


def visualize_comparison(model_global, model_spatial):
    """Visualize both models on same heterogeneous noise image."""

    print("\n[Generating comparison visualization...]")

    # Create test image
    dataset = HeterogeneousNoiseDataset(num_samples=1, size=128)
    sample = dataset[0]
    noisy = sample['noisy'].unsqueeze(0)
    clean = sample['clean'].unsqueeze(0)

    # Test both models
    model_global.eval()
    model_spatial.eval()

    with torch.no_grad():
        denoised_global, weights_global, extras_global = model_global(noisy)
        denoised_spatial, weights_spatial, extras_spatial = model_spatial(noisy)

    psnr_global = -10 * np.log10(((denoised_global - clean) ** 2).mean().item())
    psnr_spatial = -10 * np.log10(((denoised_spatial - clean) ** 2).mean().item())

    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)

    # Row 1: Inputs and outputs
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(noisy[0, 0].numpy(), cmap='gray')
    ax.set_title('Noisy Input\n(4 Quadrants)', fontsize=12, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(clean[0, 0].numpy(), cmap='gray')
    ax.set_title('Clean Reference', fontsize=12, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(denoised_global[0, 0].numpy(), cmap='gray')
    ax.set_title(f'GLOBAL Weights\nPSNR: {psnr_global:.2f} dB', fontsize=12, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(denoised_spatial[0, 0].numpy(), cmap='gray')
    delta = psnr_spatial - psnr_global
    ax.set_title(f'SPATIAL Weights\nPSNR: {psnr_spatial:.2f} dB ({delta:+.2f})',
                 fontsize=12, fontweight='bold',
                 color='green' if delta > 0 else 'red')
    ax.axis('off')

    # Difference maps
    ax = fig.add_subplot(gs[0, 4])
    diff = np.abs(denoised_spatial[0, 0].numpy() - denoised_global[0, 0].numpy())
    im = ax.imshow(diff, cmap='hot')
    ax.set_title('Absolute Difference\n(Spatial - Global)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Global weights (scalar)
    ax = fig.add_subplot(gs[1, 0])
    noise_names = ['Speckle', 'Banding', 'Gaussian', 'Shot']
    global_vals = [weights_global[name.lower()].item() for name in noise_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax.bar(range(4), global_vals, color=colors, alpha=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(noise_names, rotation=45)
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 1)
    ax.set_title('GLOBAL Weights\n(Same Everywhere)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Row 2: Spatial weight maps
    if 'spatial_weight_maps' in extras_spatial:
        spatial_maps = extras_spatial['spatial_weight_maps'][0]

        for i, name in enumerate(noise_names):
            ax = fig.add_subplot(gs[1, i+1])
            im = ax.imshow(spatial_maps[i].numpy(), cmap='RdYlGn_r', vmin=0, vmax=1)
            mean_val = spatial_maps[i].mean().item()
            std_val = spatial_maps[i].std().item()
            ax.set_title(f'{name}\nμ={mean_val:.3f}, σ={std_val:.3f}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 3: Dominant noise maps
        ax = fig.add_subplot(gs[2, 0:2])
        ax.text(0.5, 0.5, 'GLOBAL:\nUniform blending\neverywhere\n(no spatial map)',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.axis('off')

        ax = fig.add_subplot(gs[2, 2:4])
        dominant = torch.argmax(spatial_maps, dim=0).numpy()
        cmap_custom = ListedColormap(colors)
        im = ax.imshow(dominant, cmap=cmap_custom)
        ax.set_title('SPATIAL: Dominant Noise (Per-Pixel)', fontsize=12, fontweight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
        cbar.set_ticklabels(noise_names)

        # Check adaptation
        H, W = spatial_maps.shape[1], spatial_maps.shape[2]
        half = H // 2
        quad_means = []
        expected = ['Speckle', 'Banding', 'Gaussian', 'Shot']
        for idx, (y_slice, x_slice) in enumerate([
            (slice(0, half), slice(0, half)),
            (slice(0, half), slice(half, W)),
            (slice(half, H), slice(0, half)),
            (slice(half, H), slice(half, W)),
        ]):
            quad_weights = spatial_maps[:, y_slice, x_slice].mean(dim=(1, 2))
            dominant_idx = torch.argmax(quad_weights).item()
            quad_means.append((expected[idx], noise_names[dominant_idx],
                             dominant_idx == idx))

        adaptation_text = "Quadrant Analysis:\\n"
        for exp, pred, correct in quad_means:
            status = "✓" if correct else "✗"
            adaptation_text += f"{status} {exp}: {pred}\\n"

        ax = fig.add_subplot(gs[2, 4])
        ax.text(0.1, 0.5, adaptation_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.axis('off')
    else:
        ax = fig.add_subplot(gs[1:3, 1:5])
        ax.text(0.5, 0.5, 'NO SPATIAL WEIGHT MAPS\n(Global weights only)',
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.axis('off')

    plt.suptitle('Comparison: Global vs Spatial Adaptive Weights', fontsize=16, fontweight='bold')

    save_path = 'comparison_spatial_vs_global.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")


def main():
    """Run complete comparison."""

    print("="*80)
    print("COMPARISON: Global Weights vs Spatial Weights")
    print("="*80)
    print("\nThis test will:")
    print("  1. Train WITHOUT spatial weights (5 epochs)")
    print("  2. Train WITH spatial weights (5 epochs)")
    print("  3. Compare PSNR on uniform noise")
    print("  4. Compare PSNR on heterogeneous noise")
    print("  5. Check spatial adaptation capability")
    print("="*80)

    # Train both models
    model_global, losses_global, time_global = train_model(use_spatial_weights=False, num_epochs=5)
    model_spatial, losses_spatial, time_spatial = train_model(use_spatial_weights=True, num_epochs=5)

    # Evaluate both
    results_global = evaluate_model(model_global, use_spatial_weights=False)
    results_spatial = evaluate_model(model_spatial, use_spatial_weights=True)

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\n{'Metric':<30} {'GLOBAL':<20} {'SPATIAL':<20} {'Difference':<15}")
    print("-"*85)

    print(f"{'Training Time':<30} {time_global:>18.1f}s {time_spatial:>18.1f}s {time_spatial-time_global:>14.1f}s")
    print(f"{'Final Loss':<30} {losses_global[-1]:>18.4f} {losses_spatial[-1]:>18.4f} {losses_spatial[-1]-losses_global[-1]:>14.4f}")

    print(f"\n{'Uniform Noise PSNR':<30} {results_global['uniform_psnr']:>17.2f} dB {results_spatial['uniform_psnr']:>17.2f} dB {results_spatial['uniform_psnr']-results_global['uniform_psnr']:>13.2f} dB")
    print(f"{'  (std dev)':<30} {results_global['uniform_psnr_std']:>18.2f} {results_spatial['uniform_psnr_std']:>18.2f}")

    print(f"\n{'Heterogeneous PSNR':<30} {results_global['hetero_psnr']:>17.2f} dB {results_spatial['hetero_psnr']:>17.2f} dB {results_spatial['hetero_psnr']-results_global['hetero_psnr']:>13.2f} dB")
    print(f"{'  (std dev)':<30} {results_global['hetero_psnr_std']:>18.2f} {results_spatial['hetero_psnr_std']:>18.2f}")

    print(f"\n{'Spatial Adaptation':<30} {'No':>20} {'Yes' if results_spatial['spatial_adaptation'] else 'No':>20}")

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    hetero_gain = results_spatial['hetero_psnr'] - results_global['hetero_psnr']
    uniform_gain = results_spatial['uniform_psnr'] - results_global['uniform_psnr']

    if hetero_gain > 0.5:
        print(f"✓ Spatial weights IMPROVE heterogeneous noise by {hetero_gain:.2f} dB!")
    elif hetero_gain > 0:
        print(f"~ Spatial weights slightly improve heterogeneous noise by {hetero_gain:.2f} dB")
    else:
        print(f"⚠ Spatial weights show no improvement yet ({hetero_gain:.2f} dB)")
        print("  → Needs longer training (20-40 epochs) to learn spatial patterns")

    if results_spatial['spatial_adaptation']:
        print("✓ Spatial weights ARE adapting per-pixel!")
    else:
        print("⚠ Spatial weights NOT yet adapting (still learning)")

    print("\nNote: With only 5 epochs, spatial refiner may not have learned yet.")
    print("      For full benefit, train for 30-40 epochs.")
    print("="*80)

    # Visualize
    visualize_comparison(model_global, model_spatial)

    print("\n✓ Comparison complete! Check 'comparison_spatial_vs_global.png'")


if __name__ == '__main__':
    main()
