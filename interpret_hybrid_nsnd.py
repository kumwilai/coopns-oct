#!/usr/bin/env python3
"""
Spatial Interpretability Analysis for Hybrid NSND
Generates actionable insights for TMI paper:
- Spatial noise attribution maps
- Uncertainty estimation
- Anomaly detection
- Quality control warnings
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class SpatialNoiseAnalyzer:
    """Extract spatial interpretability from Hybrid NSND model."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def generate_patch_based_maps(self, image: torch.Tensor,
                                   patch_size: int = 32,
                                   stride: int = 16) -> Dict:
        """
        Generate spatial noise attribution using sliding window.

        Args:
            image: (1, 1, H, W) input image
            patch_size: Size of sliding window
            stride: Stride for sliding window

        Returns:
            Dict with:
            - 'noise_weights': (4, H', W') spatial weight maps
            - 'dominant_noise': (H', W') argmax of noise types
            - 'uncertainty': (H', W') entropy-based uncertainty
            - 'expert_residuals': Dict of what each head removed
        """
        _, _, H, W = image.shape

        h_out = (H - patch_size) // stride + 1
        w_out = (W - patch_size) // stride + 1

        noise_weights = torch.zeros(4, h_out, w_out)
        uncertainty_map = torch.zeros(h_out, w_out)
        expert_residuals = {
            'speckle': torch.zeros(h_out, w_out),
            'banding': torch.zeros(h_out, w_out),
            'gaussian': torch.zeros(h_out, w_out),
            'shot': torch.zeros(h_out, w_out),
        }

        with torch.no_grad():
            for i, y in enumerate(range(0, H - patch_size + 1, stride)):
                for j, x in enumerate(range(0, W - patch_size + 1, stride)):
                    patch = image[:, :, y:y+patch_size, x:x+patch_size]

                    output, weights, extras = self.model(patch.to(self.device))

                    # Store weights
                    noise_weights[0, i, j] = weights['speckle'].item()
                    noise_weights[1, i, j] = weights['banding'].item()
                    noise_weights[2, i, j] = weights['gaussian'].item()
                    noise_weights[3, i, j] = weights['shot'].item()

                    # Uncertainty (entropy-based)
                    w = torch.stack([
                        weights['speckle'],
                        weights['banding'],
                        weights['gaussian'],
                        weights['shot']
                    ])
                    entropy = -torch.sum(w * torch.log(w + 1e-8))
                    uncertainty_map[i, j] = entropy.item()

                    # What each expert removed
                    if 'expert_outputs' in extras:
                        for name in expert_residuals.keys():
                            residual = torch.abs(patch - extras['expert_outputs'][name])
                            expert_residuals[name][i, j] = residual.mean().item()

        return {
            'noise_weights': noise_weights,
            'dominant_noise': torch.argmax(noise_weights, dim=0),
            'uncertainty': uncertainty_map,
            'expert_residuals': expert_residuals
        }

    def detect_anomalies(self, spatial_maps: Dict,
                         uncertainty_threshold: float = 0.8) -> Dict:
        """Detect regions requiring clinical attention."""
        uncertainty = spatial_maps['uncertainty']
        dominant = spatial_maps['dominant_noise']

        high_uncertainty_mask = uncertainty > uncertainty_threshold
        banding_dominant = (dominant == 1)
        banding_regions = banding_dominant.sum().item() / dominant.numel()
        noise_changes = torch.abs(dominant[:-1, :] - dominant[1:, :]).sum()

        warnings_list = []
        if banding_regions > 0.3:
            warnings_list.append("⚠ HIGH BANDING (>30%): Check scanner alignment")
        if uncertainty.mean().item() > 0.7:
            warnings_list.append("⚠ HIGH UNCERTAINTY: Manual review recommended")
        if banding_regions > 0.5 and uncertainty.mean().item() > 0.6:
            warnings_list.append("⚠ CRITICAL: Possible scanner malfunction")

        return {
            'high_uncertainty_regions': high_uncertainty_mask,
            'banding_percentage': banding_regions * 100,
            'spatial_heterogeneity': noise_changes.item(),
            'quality_score': 1.0 - uncertainty.mean().item(),
            'warnings': warnings_list
        }


def visualize_model_output(image: torch.Tensor,
                           denoised: torch.Tensor,
                           extras: Dict,
                           save_path: str = None):
    """
    Visualize direct model output including spatial weight maps if available.

    Args:
        image: (B, 1, H, W) noisy input
        denoised: (B, 1, H, W) denoised output
        extras: Dict from model forward pass
        save_path: Where to save figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Input/output
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image[0, 0].cpu().numpy(), cmap='gray')
    ax1.set_title('Noisy Input', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(denoised[0, 0].cpu().detach().numpy(), cmap='gray')
    ax2.set_title('Denoised Output', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Check if spatial weight maps are available
    if 'spatial_weight_maps' in extras:
        spatial_maps = extras['spatial_weight_maps'][0]  # (4, H, W)

        # Plot each noise type's spatial weight
        noise_names = ['Speckle', 'Banding', 'Gaussian', 'Shot']
        for i, name in enumerate(noise_names):
            row, col = (0, 2+i) if i < 2 else (1, i-2)
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(spatial_maps[i].cpu().numpy(), cmap='RdYlGn_r', vmin=0, vmax=1)
            ax.set_title(f'{name} Weight Map', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Dominant noise type (argmax)
        ax_dom = fig.add_subplot(gs[1, 2])
        dominant = torch.argmax(spatial_maps, dim=0).cpu().numpy()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        cmap = ListedColormap(colors)
        im = ax_dom.imshow(dominant, cmap=cmap)
        ax_dom.set_title('Dominant Noise (Spatial)', fontsize=14, fontweight='bold')
        ax_dom.axis('off')
        cbar = plt.colorbar(im, ax=ax_dom, ticks=[0,1,2,3], fraction=0.046)
        cbar.set_ticklabels(noise_names)

        # Uncertainty (entropy)
        ax_unc = fig.add_subplot(gs[1, 3])
        entropy = -torch.sum(spatial_maps * torch.log(spatial_maps + 1e-8), dim=0)
        im = ax_unc.imshow(entropy.cpu().numpy(), cmap='hot')
        ax_unc.set_title('Uncertainty Map', fontsize=14, fontweight='bold')
        ax_unc.axis('off')
        plt.colorbar(im, ax=ax_unc, fraction=0.046)

        fig.text(0.02, 0.5, '✓ SPATIAL WEIGHTS ENABLED\nPer-pixel adaptive denoising',
                fontsize=12, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    else:
        fig.text(0.02, 0.5, '⚠ Global weights only\n(uniform blending)',
                fontsize=12, color='orange', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Expert outputs if available
    if 'expert_outputs' in extras:
        expert_out = extras['expert_outputs']
        noise_names = ['speckle', 'banding', 'gaussian', 'shot']
        for i, name in enumerate(noise_names):
            ax = fig.add_subplot(gs[2, i])
            if name in expert_out:
                ax.imshow(expert_out[name][0, 0].cpu().detach().numpy(), cmap='gray')
                ax.set_title(f'{name.capitalize()} Expert', fontsize=10)
            ax.axis('off')

    plt.suptitle('Spatially-Adaptive NSND Analysis', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def visualize_interpretability(image: torch.Tensor,
                               spatial_maps: Dict,
                               anomalies: Dict,
                               denoised: torch.Tensor = None,
                               save_path: str = None):
    """Create TMI-quality visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Input
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image[0, 0].cpu(), cmap='gray')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Individual weight maps
    noise_names = ['Speckle', 'Banding', 'Gaussian', 'Shot']
    for i, name in enumerate(noise_names):
        row, col = (0, 1 + i) if i < 3 else (1, 0)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(spatial_maps['noise_weights'][i].cpu(),
                      cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f'{name} Weight', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Dominant noise type
    ax2 = fig.add_subplot(gs[1, 1])
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    cmap = ListedColormap(colors)
    im2 = ax2.imshow(spatial_maps['dominant_noise'].cpu(), cmap=cmap)
    ax2.set_title('Dominant Noise (Spatial)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    cbar = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3], fraction=0.046)
    cbar.set_ticklabels(noise_names)

    # Uncertainty
    ax3 = fig.add_subplot(gs[1, 2])
    im3 = ax3.imshow(spatial_maps['uncertainty'].cpu(), cmap='hot')
    ax3.set_title('Uncertainty Map', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, label='Entropy')

    # High uncertainty overlay
    ax4 = fig.add_subplot(gs[1, 3])
    overlay = image[0, 0].cpu().numpy()
    mask = anomalies['high_uncertainty_regions'].cpu().numpy()
    ax4.imshow(overlay, cmap='gray', alpha=0.7)
    ax4.imshow(mask, cmap='Reds', alpha=0.5)
    ax4.set_title('Alert Regions', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # What each expert removed
    for i, (name, residual) in enumerate(spatial_maps['expert_residuals'].items()):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(residual.cpu(), cmap='hot')
        ax.set_title(f'{name.capitalize()} Removed', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Quality metrics
    metrics_text = f"""Quality Score: {anomalies['quality_score']:.2%}
Banding: {anomalies['banding_percentage']:.1f}%
Heterogeneity: {anomalies['spatial_heterogeneity']:.0f}

{'Warnings:' if anomalies['warnings'] else '✓ No issues'}
"""
    if anomalies['warnings']:
        metrics_text += "\n".join(anomalies['warnings'])

    fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


print("Interpretability analysis module loaded.")
print("Usage: from interpret_hybrid_nsnd import SpatialNoiseAnalyzer, visualize_interpretability")
