#!/usr/bin/env python3
"""
Residual Analysis for Per-Layer OCT Denoising Investigation

This script investigates WHY per-layer denoising is failing compared to baseline NAFNet.

KEY QUESTIONS:
1. What do the residuals (clean - nafnet_output) actually look like?
2. Are residuals spatially uniform or structured?
3. Are residuals larger at boundaries?
4. Do different layers have different residual patterns?
5. Is the blending causing artifacts?
6. What is the fundamental assumption being violated?

HYPOTHESIS TO TEST:
"Per-layer processing fails because NAFNet residuals are NOT layer-specific."
If true: residuals should be spatially uniform, not correlated with layers.
If false: residuals should show clear layer-specific patterns.

Author: Diagnostic Analysis
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

# Add paths for local imports
sys.path.insert(0, '/home/kumwilai/OCT')
sys.path.insert(0, '/home/kumwilai/OCT/nsnd_oct')

from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
PKU37_ROOT = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"
NAFNET_CHECKPOINT = "/home/kumwilai/OCT/checkpoints/nafnet_w40_realistic_best.pth"
OUTPUT_DIR = "/home/kumwilai/OCT/residual_analysis"

# Layer definitions (4 layers based on existing codebase)
NUM_LAYERS = 4
LAYER_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
BOUNDARY_NAMES = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']


# =============================================================================
# Data Loading
# =============================================================================
def load_pku37_images(root: str, max_samples: int = 10, split: str = 'val') -> List[Dict]:
    """Load image pairs from PKU37 dataset."""
    clean_dir = os.path.join(root, "clean")
    noisy_dir = os.path.join(root, "noisy")

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])

    # Split: 80% train, 20% val
    n_total = len(clean_files)
    n_train = int(n_total * 0.8)

    if split == 'train':
        clean_files = clean_files[:n_train]
    else:
        clean_files = clean_files[n_train:]

    samples = []
    for fname in clean_files[:max_samples]:
        clean_path = os.path.join(clean_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # Find corresponding noisy images
        noisy_files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.startswith(base_name) and f.endswith('.tif')
        ])

        if noisy_files:
            noisy_path = os.path.join(noisy_dir, noisy_files[0])

            clean = np.array(Image.open(clean_path), dtype=np.float32) / 255.0
            noisy = np.array(Image.open(noisy_path), dtype=np.float32) / 255.0

            samples.append({
                'clean': clean,
                'noisy': noisy,
                'name': fname,
            })

    logger.info(f"Loaded {len(samples)} samples from PKU37 {split} set")
    return samples


def load_nafnet(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load pre-trained NAFNet model."""
    # Try different checkpoint paths if needed
    if not os.path.exists(checkpoint_path):
        alt_paths = [
            "/home/kumwilai/OCT/nsnd_oct/checkpoints/nafnet_w32_realistic_best.pth",
            "/home/kumwilai/OCT/checkpoints/nafnet_w41_realistic_best.pth",
            "/home/kumwilai/OCT/nsnd_oct/checkpoints/nafnet_synthetic_best.pth",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                checkpoint_path = alt
                break

    logger.info(f"Loading NAFNet from {checkpoint_path}")

    # Determine width from checkpoint name
    if 'w40' in checkpoint_path or 'w41' in checkpoint_path:
        width = 40
    elif 'w32' in checkpoint_path:
        width = 32
    elif 'w16' in checkpoint_path:
        width = 16
    else:
        width = 32  # default

    model = NAFNet(
        img_channel=1,
        width=width,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
        cond_dim=4,
        condition_middle=True,
        condition_decoders=True,
        use_spatial_cue=True,
        spatial_cue_channels=4,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Try loading with strict=False to handle missing keys
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"NAFNet loaded successfully (width={width})")
    return model


# =============================================================================
# Boundary Detection (Heuristic based on intensity gradients)
# =============================================================================
def detect_boundaries_heuristic(image: np.ndarray, num_boundaries: int = 4) -> np.ndarray:
    """
    Detect layer boundaries using simple gradient-based heuristics.

    This is a simple approach that:
    1. Computes vertical gradients
    2. Finds peaks in gradient magnitude
    3. Returns boundary positions

    Args:
        image: [H, W] grayscale image
        num_boundaries: Number of boundaries to detect

    Returns:
        boundaries: [num_boundaries, W] boundary positions (pixel rows)
    """
    H, W = image.shape

    # Compute vertical gradient (Sobel)
    sobel_y = ndimage.sobel(image, axis=0)
    gradient_magnitude = np.abs(sobel_y)

    # Smooth horizontally to reduce noise
    gradient_smooth = ndimage.uniform_filter1d(gradient_magnitude, size=5, axis=1)

    # For each column, find the top N gradient peaks
    boundaries = np.zeros((num_boundaries, W))

    for w in range(W):
        column = gradient_smooth[:, w]

        # Find local maxima
        peaks = []
        for h in range(5, H - 5):
            if column[h] > column[h-1] and column[h] > column[h+1]:
                if column[h] > column[h-2] and column[h] > column[h+2]:
                    peaks.append((h, column[h]))

        # Sort by gradient magnitude and take top N
        peaks.sort(key=lambda x: -x[1])
        peaks = peaks[:num_boundaries * 2]  # Take extra for filtering

        # Sort by position (top to bottom)
        peaks.sort(key=lambda x: x[0])

        # Assign to boundaries (distribute evenly if not enough peaks)
        if len(peaks) >= num_boundaries:
            for i in range(num_boundaries):
                idx = i * len(peaks) // num_boundaries
                boundaries[i, w] = peaks[idx][0]
        else:
            # Fallback: divide image evenly
            for i in range(num_boundaries):
                boundaries[i, w] = H * (i + 1) / (num_boundaries + 1)

    # Smooth boundaries
    for i in range(num_boundaries):
        boundaries[i] = ndimage.uniform_filter1d(boundaries[i], size=11)

    return boundaries


def compute_layer_masks(boundaries: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Compute soft layer masks from boundary positions.

    Args:
        boundaries: [num_boundaries, W] boundary positions
        H: Image height
        W: Image width

    Returns:
        masks: [num_layers, H, W] soft layer masks (sum to 1)
    """
    num_boundaries = boundaries.shape[0]
    num_layers = num_boundaries + 1

    # Create row indices [H, W]
    rows = np.arange(H).reshape(-1, 1).repeat(W, axis=1)

    masks = np.zeros((num_layers, H, W))

    # Layer 0: from top to first boundary
    masks[0] = (rows < boundaries[0]).astype(float)

    # Middle layers
    for i in range(1, num_boundaries):
        masks[i] = ((rows >= boundaries[i-1]) & (rows < boundaries[i])).astype(float)

    # Last layer: from last boundary to bottom
    masks[-1] = (rows >= boundaries[-1]).astype(float)

    # Add soft transitions at boundaries (3-pixel window)
    sigma = 2.0
    for i in range(num_layers):
        masks[i] = ndimage.gaussian_filter(masks[i], sigma=(sigma, 0))

    # Normalize to sum to 1
    total = masks.sum(axis=0, keepdims=True)
    masks = masks / np.maximum(total, 1e-8)

    return masks


# =============================================================================
# Residual Analysis
# =============================================================================
def analyze_residuals(
    samples: List[Dict],
    nafnet: nn.Module,
    device: torch.device,
) -> Dict:
    """
    Comprehensive residual analysis.

    For each image:
    1. Run NAFNet to get denoised output
    2. Compute residual: (clean - nafnet_output)
    3. Detect layer boundaries (heuristic)
    4. Analyze residual statistics per layer
    """
    results = {
        'per_image': [],
        'global': defaultdict(list),
    }

    for sample in samples:
        clean = sample['clean']
        noisy = sample['noisy']
        name = sample['name']
        H, W = clean.shape

        # Run NAFNet
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            denoised_tensor = nafnet(noisy_tensor)

        denoised = denoised_tensor.squeeze().cpu().numpy()

        # Compute residual: what NAFNet got wrong
        residual = clean - denoised

        # Also compute noise (what was removed)
        noise_removed = noisy - denoised

        # Detect boundaries using clean image (best case) and denoised image
        boundaries_clean = detect_boundaries_heuristic(clean, NUM_LAYERS - 1)
        boundaries_denoised = detect_boundaries_heuristic(denoised, NUM_LAYERS - 1)

        # Compute layer masks
        masks_clean = compute_layer_masks(boundaries_clean, H, W)
        masks_denoised = compute_layer_masks(boundaries_denoised, H, W)

        # Analyze residuals per layer
        layer_stats = {}
        for i, layer_name in enumerate(LAYER_NAMES):
            mask = masks_clean[i]
            layer_pixels = mask > 0.5  # Binary for statistics

            if layer_pixels.sum() > 0:
                layer_residual = residual[layer_pixels]
                layer_stats[layer_name] = {
                    'mean': float(np.mean(layer_residual)),
                    'std': float(np.std(layer_residual)),
                    'abs_mean': float(np.mean(np.abs(layer_residual))),
                    'max': float(np.max(np.abs(layer_residual))),
                    'n_pixels': int(layer_pixels.sum()),
                }

        # Analyze residuals at boundaries vs. interior
        boundary_residuals = []
        interior_residuals = []

        for i in range(boundaries_clean.shape[0]):
            for w in range(W):
                row = int(boundaries_clean[i, w])
                # Boundary region: +/- 3 pixels
                for dr in range(-3, 4):
                    r = row + dr
                    if 0 <= r < H:
                        boundary_residuals.append(residual[r, w])

        # Interior: exclude boundary regions
        boundary_mask = np.zeros((H, W), dtype=bool)
        for i in range(boundaries_clean.shape[0]):
            for w in range(W):
                row = int(boundaries_clean[i, w])
                for dr in range(-3, 4):
                    r = row + dr
                    if 0 <= r < H:
                        boundary_mask[r, w] = True

        interior_residuals = residual[~boundary_mask].flatten()

        boundary_stats = {
            'boundary_abs_mean': float(np.mean(np.abs(boundary_residuals))),
            'interior_abs_mean': float(np.mean(np.abs(interior_residuals))),
            'boundary_std': float(np.std(boundary_residuals)),
            'interior_std': float(np.std(interior_residuals)),
        }

        # Compute PSNR
        mse = np.mean((clean - denoised) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 50.0

        input_mse = np.mean((clean - noisy) ** 2)
        input_psnr = 10 * np.log10(1.0 / input_mse) if input_mse > 1e-10 else 50.0

        image_result = {
            'name': name,
            'psnr': psnr,
            'input_psnr': input_psnr,
            'gain': psnr - input_psnr,
            'layer_stats': layer_stats,
            'boundary_stats': boundary_stats,
            'residual_global_std': float(np.std(residual)),
            'residual_global_mean': float(np.mean(residual)),
        }

        results['per_image'].append(image_result)

        # Aggregate
        results['global']['psnr'].append(psnr)
        results['global']['gain'].append(psnr - input_psnr)
        for layer_name in LAYER_NAMES:
            if layer_name in layer_stats:
                results['global'][f'{layer_name}_abs_mean'].append(layer_stats[layer_name]['abs_mean'])
                results['global'][f'{layer_name}_std'].append(layer_stats[layer_name]['std'])

        results['global']['boundary_abs_mean'].append(boundary_stats['boundary_abs_mean'])
        results['global']['interior_abs_mean'].append(boundary_stats['interior_abs_mean'])

    # Compute global averages
    results['summary'] = {k: float(np.mean(v)) for k, v in results['global'].items()}

    return results


def visualize_residuals(
    samples: List[Dict],
    nafnet: nn.Module,
    device: torch.device,
    output_dir: str,
    max_images: int = 5,
):
    """Create visualizations of residual patterns."""
    os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(samples[:max_images]):
        clean = sample['clean']
        noisy = sample['noisy']
        name = sample['name']
        H, W = clean.shape

        # Run NAFNet
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            denoised_tensor = nafnet(noisy_tensor)
        denoised = denoised_tensor.squeeze().cpu().numpy()

        # Compute residual and noise
        residual = clean - denoised
        noise_removed = noisy - denoised

        # Detect boundaries
        boundaries = detect_boundaries_heuristic(clean, NUM_LAYERS - 1)

        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Row 1: Images
        axes[0, 0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'Noisy Input')

        axes[0, 1].imshow(denoised, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('NAFNet Denoised')

        axes[0, 2].imshow(clean, cmap='gray', vmin=0, vmax=1)
        # Overlay boundaries
        for i in range(boundaries.shape[0]):
            axes[0, 2].plot(range(W), boundaries[i], 'r-', linewidth=1, alpha=0.7)
        axes[0, 2].set_title('Clean (GT) + Detected Boundaries')

        # Residual with colorbar
        vmax = max(0.1, np.percentile(np.abs(residual), 99))
        im = axes[0, 3].imshow(residual, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[0, 3].set_title(f'Residual (clean - denoised)')
        plt.colorbar(im, ax=axes[0, 3])

        # Row 2: Analysis
        # Residual magnitude per row (averaged across columns)
        residual_per_row = np.mean(np.abs(residual), axis=1)
        axes[1, 0].plot(residual_per_row, range(H), 'b-', linewidth=1)
        axes[1, 0].set_ylim(H, 0)
        axes[1, 0].set_xlabel('Mean |Residual|')
        axes[1, 0].set_ylabel('Row (depth)')
        axes[1, 0].set_title('Residual Magnitude vs Depth')

        # Mark boundaries
        for i, b in enumerate(boundaries):
            y = np.mean(b)
            axes[1, 0].axhline(y, color='r', linestyle='--', alpha=0.7, label=f'B{i}' if i == 0 else '')
        axes[1, 0].legend()

        # Histogram of residuals per layer
        masks = compute_layer_masks(boundaries, H, W)
        colors = ['blue', 'green', 'orange', 'red']

        for i, (layer_name, color) in enumerate(zip(LAYER_NAMES, colors)):
            mask = masks[i] > 0.5
            if mask.sum() > 0:
                layer_residual = residual[mask]
                axes[1, 1].hist(layer_residual, bins=50, alpha=0.5, color=color,
                               label=f'{layer_name}: std={np.std(layer_residual):.4f}', density=True)

        axes[1, 1].set_xlabel('Residual Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Residual Distribution by Layer')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(-0.15, 0.15)

        # Spatial correlation of residuals
        # Does residual correlate with image intensity?
        axes[1, 2].scatter(clean.flatten()[::100], residual.flatten()[::100], alpha=0.3, s=1)
        axes[1, 2].set_xlabel('Clean Intensity')
        axes[1, 2].set_ylabel('Residual')
        axes[1, 2].set_title('Residual vs Intensity')

        # Compute correlation
        corr, pval = pearsonr(clean.flatten(), residual.flatten())
        axes[1, 2].text(0.05, 0.95, f'Corr: {corr:.3f}\np={pval:.2e}',
                       transform=axes[1, 2].transAxes, verticalalignment='top')

        # Residual magnitude map (smoothed)
        residual_magnitude = np.abs(residual)
        residual_smooth = ndimage.uniform_filter(residual_magnitude, size=5)
        im2 = axes[1, 3].imshow(residual_smooth, cmap='hot', vmin=0)
        # Overlay boundaries
        for i in range(boundaries.shape[0]):
            axes[1, 3].plot(range(W), boundaries[i], 'c-', linewidth=1, alpha=0.8)
        axes[1, 3].set_title('Residual Magnitude (smoothed)')
        plt.colorbar(im2, ax=axes[1, 3])

        # Compute PSNR
        mse = np.mean((clean - denoised) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 50.0

        plt.suptitle(f'{name} | PSNR: {psnr:.2f} dB', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'residual_analysis_{idx:02d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization: {save_path}")


def analyze_blending_artifacts(
    samples: List[Dict],
    nafnet: nn.Module,
    device: torch.device,
) -> Dict:
    """
    Analyze potential blending artifacts from per-layer processing.

    Key question: When we blend layer outputs with soft masks, do we introduce errors?

    Simulation:
    1. Apply NAFNet to get baseline output
    2. Simulate per-layer processing (same output, but masked and blended)
    3. Measure error introduced by blending alone
    """
    results = {
        'per_image': [],
        'summary': {},
    }

    for sample in samples:
        clean = sample['clean']
        noisy = sample['noisy']
        name = sample['name']
        H, W = clean.shape

        # Run NAFNet
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            denoised_tensor = nafnet(noisy_tensor)
        denoised = denoised_tensor.squeeze().cpu().numpy()

        # Detect boundaries
        boundaries = detect_boundaries_heuristic(clean, NUM_LAYERS - 1)
        masks = compute_layer_masks(boundaries, H, W)

        # Simulate blending: use same denoised output but mask and recombine
        # This isolates the blending error from any per-layer head errors
        blended = np.zeros_like(denoised)
        for i in range(NUM_LAYERS):
            blended += denoised * masks[i]

        # Blending error (should be near zero if masks sum to 1)
        blending_error = np.mean(np.abs(denoised - blended))

        # Now simulate per-layer with small perturbations (what if heads have small errors?)
        # Add random noise to each layer proportional to layer intensity
        per_layer_errors = [0.001, 0.002, 0.005, 0.01, 0.02]  # Test different error levels

        layer_error_impact = {}
        for err_level in per_layer_errors:
            perturbed_layers = []
            for i in range(NUM_LAYERS):
                # Add random perturbation
                perturbation = np.random.randn(H, W) * err_level
                perturbed_layers.append(denoised + perturbation)

            # Blend perturbed layers
            blended_perturbed = np.zeros_like(denoised)
            for i in range(NUM_LAYERS):
                blended_perturbed += perturbed_layers[i] * masks[i]

            # Error from perturbation + blending
            total_error = np.mean((clean - blended_perturbed) ** 2)
            baseline_error = np.mean((clean - denoised) ** 2)

            layer_error_impact[err_level] = {
                'total_mse': float(total_error),
                'baseline_mse': float(baseline_error),
                'psnr_drop': float(10 * np.log10(total_error / baseline_error)) if baseline_error > 0 else 0,
            }

        # Analyze boundary discontinuities
        # Check if there are sudden changes at layer boundaries
        boundary_discontinuities = []
        for i in range(boundaries.shape[0]):
            for w in range(W):
                row = int(boundaries[i, w])
                if 3 <= row < H - 3:
                    # Gradient across boundary
                    above = np.mean(denoised[row-3:row, w])
                    below = np.mean(denoised[row:row+3, w])
                    boundary_discontinuities.append(abs(above - below))

        results['per_image'].append({
            'name': name,
            'blending_error': float(blending_error),
            'mean_boundary_discontinuity': float(np.mean(boundary_discontinuities)),
            'layer_error_impact': layer_error_impact,
        })

    # Compute summary
    results['summary'] = {
        'mean_blending_error': float(np.mean([r['blending_error'] for r in results['per_image']])),
        'mean_boundary_discontinuity': float(np.mean([r['mean_boundary_discontinuity'] for r in results['per_image']])),
    }

    return results


def test_hypothesis_layer_specific(results: Dict) -> str:
    """
    Test the hypothesis: "Do residuals have layer-specific patterns?"

    If per-layer denoising should help, we expect:
    1. Different layers have significantly different residual statistics
    2. Residuals are NOT spatially uniform
    3. Boundary regions have different residual patterns than interior

    If per-layer denoising is fundamentally flawed, we expect:
    1. Residual statistics are similar across layers
    2. Residuals are largely uniform (just noise)
    3. No correlation with layer structure
    """
    summary = results['summary']

    report = []
    report.append("=" * 70)
    report.append("HYPOTHESIS TEST: Do residuals have layer-specific patterns?")
    report.append("=" * 70)
    report.append("")

    # Test 1: Layer variance
    layer_stds = [summary.get(f'{name}_std', 0) for name in LAYER_NAMES]
    layer_means = [summary.get(f'{name}_abs_mean', 0) for name in LAYER_NAMES]

    report.append("TEST 1: Layer-specific residual variance")
    report.append("-" * 40)
    for name, std, mean in zip(LAYER_NAMES, layer_stds, layer_means):
        report.append(f"  {name:15s}: mean|res|={mean:.5f}, std={std:.5f}")

    std_variance = np.std(layer_stds) / (np.mean(layer_stds) + 1e-8)
    if std_variance > 0.1:
        report.append(f"  -> SIGNIFICANT variance across layers (CV={std_variance:.3f})")
        report.append(f"  -> Per-layer processing MAY help")
    else:
        report.append(f"  -> LOW variance across layers (CV={std_variance:.3f})")
        report.append(f"  -> Per-layer processing likely WON'T help")
    report.append("")

    # Test 2: Boundary vs Interior
    boundary_mean = summary.get('boundary_abs_mean', 0)
    interior_mean = summary.get('interior_abs_mean', 0)

    report.append("TEST 2: Boundary vs Interior residuals")
    report.append("-" * 40)
    report.append(f"  Boundary mean |residual|: {boundary_mean:.5f}")
    report.append(f"  Interior mean |residual|: {interior_mean:.5f}")

    if boundary_mean > interior_mean * 1.1:
        report.append(f"  -> Boundaries have HIGHER residuals ({100*(boundary_mean/interior_mean - 1):.1f}% more)")
        report.append(f"  -> Boundary-aware processing MAY help")
    else:
        report.append(f"  -> Boundaries NOT significantly worse")
        report.append(f"  -> Boundary-aware processing likely WON'T help")
    report.append("")

    # Test 3: Overall quality
    psnr = summary.get('psnr', 0)
    gain = summary.get('gain', 0)

    report.append("TEST 3: NAFNet baseline quality")
    report.append("-" * 40)
    report.append(f"  NAFNet PSNR: {psnr:.2f} dB")
    report.append(f"  PSNR Gain:   {gain:+.2f} dB")

    if psnr > 35:
        report.append(f"  -> High baseline quality - hard to improve!")
    elif psnr > 33:
        report.append(f"  -> Good baseline quality - modest improvements possible")
    else:
        report.append(f"  -> Lower baseline - room for improvement")
    report.append("")

    # Final verdict
    report.append("=" * 70)
    report.append("VERDICT")
    report.append("=" * 70)

    # Determine verdict
    layer_varies = std_variance > 0.1
    boundary_harder = boundary_mean > interior_mean * 1.1
    high_baseline = psnr > 34

    if layer_varies and boundary_harder:
        verdict = "MIXED: Residuals show SOME layer-specific patterns."
        verdict += "\n  Per-layer processing has theoretical justification,"
        verdict += "\n  but implementation must be careful to not introduce MORE error."
    elif layer_varies:
        verdict = "WEAK: Only layer variance observed."
        verdict += "\n  Per-layer processing MAY help but evidence is weak."
    elif boundary_harder:
        verdict = "WEAK: Only boundary issues observed."
        verdict += "\n  Focus on boundary handling, not per-layer heads."
    else:
        verdict = "NEGATIVE: Residuals appear UNIFORM across layers."
        verdict += "\n  Per-layer processing is unlikely to help."
        verdict += "\n  The fundamental assumption is VIOLATED."

    if high_baseline:
        verdict += "\n\n  ADDITIONAL: NAFNet baseline is already strong."
        verdict += "\n  Adding complexity risks introducing more error than it fixes."

    report.append(verdict)
    report.append("=" * 70)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze NAFNet residuals for per-layer denoising')
    parser.add_argument('--max_samples', type=int, default=10, help='Max images to analyze')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=NAFNET_CHECKPOINT, help='NAFNet checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    samples = load_pku37_images(PKU37_ROOT, max_samples=args.max_samples)

    # Load NAFNet
    nafnet = load_nafnet(args.checkpoint, device)

    # Run residual analysis
    logger.info("Running residual analysis...")
    results = analyze_residuals(samples, nafnet, device)

    # Save results
    results_path = os.path.join(args.output_dir, 'residual_analysis.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': results['summary'],
            'per_image': results['per_image'],
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  NAFNet PSNR:        {results['summary']['psnr']:.2f} dB")
    print(f"  PSNR Gain:          {results['summary']['gain']:+.2f} dB")

    print(f"\nPer-Layer Residual Statistics (mean |residual|):")
    for name in LAYER_NAMES:
        key = f'{name}_abs_mean'
        if key in results['summary']:
            print(f"  {name:15s}: {results['summary'][key]:.5f}")

    print(f"\nBoundary vs Interior:")
    print(f"  Boundary mean |residual|: {results['summary']['boundary_abs_mean']:.5f}")
    print(f"  Interior mean |residual|: {results['summary']['interior_abs_mean']:.5f}")

    # Hypothesis test
    hypothesis_report = test_hypothesis_layer_specific(results)
    print("\n" + hypothesis_report)

    # Save hypothesis report
    report_path = os.path.join(args.output_dir, 'hypothesis_test.txt')
    with open(report_path, 'w') as f:
        f.write(hypothesis_report)
    logger.info(f"Saved hypothesis report to {report_path}")

    # Analyze blending artifacts
    logger.info("Analyzing blending artifacts...")
    blending_results = analyze_blending_artifacts(samples, nafnet, device)

    print("\n" + "=" * 70)
    print("BLENDING ARTIFACT ANALYSIS")
    print("=" * 70)
    print(f"  Mean blending error: {blending_results['summary']['mean_blending_error']:.6f}")
    print(f"  Mean boundary discontinuity: {blending_results['summary']['mean_boundary_discontinuity']:.5f}")

    # Show impact of per-layer errors
    if blending_results['per_image']:
        sample = blending_results['per_image'][0]
        print("\n  Impact of per-layer errors on PSNR:")
        for err_level, impact in sample['layer_error_impact'].items():
            print(f"    Layer error std={err_level:.3f}: PSNR drop = {impact['psnr_drop']:.3f} dB")

    # Save blending analysis
    blending_path = os.path.join(args.output_dir, 'blending_analysis.json')
    with open(blending_path, 'w') as f:
        json.dump(blending_results, f, indent=2)
    logger.info(f"Saved blending analysis to {blending_path}")

    # Create visualizations
    if args.visualize:
        logger.info("Creating visualizations...")
        visualize_residuals(samples, nafnet, device, args.output_dir)

    print("\n" + "=" * 70)
    print(f"Analysis complete. Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
