"""
Visualization utilities for NSND

Includes:
- Noise decomposition visualization
- Uncertainty map overlay
- Component denoiser comparison
- Noise analysis reports
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import seaborn as sns

# Suppress matplotlib warnings in headless environments
import matplotlib
matplotlib.use('Agg')


def visualize_noise_decomposition(
    noisy_input: torch.Tensor,
    symbolic_weights: Dict[str, torch.Tensor],
    features: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
):
    """
    Visualize symbolic noise decomposition

    Shows:
    - Input image
    - Detected noise weights (bar chart)
    - Key feature maps (CV, kurtosis, FFT power)

    Args:
        noisy_input: Noisy image [1, 1, H, W]
        symbolic_weights: Noise composition weights
        features: Extracted features
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert to numpy
    img = noisy_input[0, 0].cpu().numpy()

    # 1. Input image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Input Noisy Image')
    axes[0, 0].axis('off')

    # 2. Noise composition (bar chart)
    components = ['speckle', 'banding', 'gaussian', 'shot']
    weights = [symbolic_weights[c][0].item() * 100 for c in components]

    axes[0, 1].bar(components, weights, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[0, 1].set_ylabel('Weight (%)')
    axes[0, 1].set_title('Detected Noise Composition')
    axes[0, 1].set_ylim(0, 100)
    for i, w in enumerate(weights):
        axes[0, 1].text(i, w + 2, f'{w:.1f}%', ha='center')

    # 3. CV map
    cv_map = features['cv_map'][0, 0].cpu().numpy()
    im = axes[0, 2].imshow(cv_map, cmap='hot')
    axes[0, 2].set_title(f'CV Map (mean={cv_map.mean():.2f})')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # 4. Kurtosis map
    kurtosis = features['kurtosis'][0, 0].cpu().numpy()
    im = axes[1, 0].imshow(kurtosis, cmap='viridis')
    axes[1, 0].set_title(f'Kurtosis Map (mean={kurtosis.mean():.2f})')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])

    # 5. Gradient magnitude
    grad_mag = features['gradient_magnitude'][0, 0].cpu().numpy()
    im = axes[1, 1].imshow(grad_mag, cmap='gray')
    axes[1, 1].set_title('Gradient Magnitude (Edges)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])

    # 6. Depth profile
    depth_profile = features['depth_profile'][0, 0].cpu().numpy()
    axes[1, 2].plot(depth_profile.mean(axis=1))
    axes[1, 2].set_title('Axial Intensity Profile')
    axes[1, 2].set_xlabel('Depth (pixels)')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_uncertainty(
    denoised: torch.Tensor,
    uncertainty: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Overlay uncertainty map on denoised output

    Green = high confidence
    Red = low confidence (uncertain)

    Args:
        denoised: Denoised image [1, 1, H, W]
        uncertainty: Uncertainty map [1, 1, H, W] in [0, 1]
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert to numpy
    img = denoised[0, 0].cpu().numpy()
    unc = uncertainty[0, 0].cpu().numpy()

    # 1. Denoised image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Denoised Output')
    axes[0].axis('off')

    # 2. Uncertainty map
    im = axes[1].imshow(unc, cmap='RdYlGn_r')  # Red = uncertain, Green = confident
    axes[1].set_title('Uncertainty Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Uncertainty')

    # 3. Overlay
    # Create RGB image
    rgb = np.stack([img, img, img], axis=-1)

    # Create colored overlay
    # Uncertain regions = red tint
    # Confident regions = green tint
    overlay = rgb.copy()
    overlay[..., 0] = overlay[..., 0] * (1 + unc * 0.5)  # Add red for uncertain
    overlay[..., 1] = overlay[..., 1] * (1 + (1-unc) * 0.5)  # Add green for confident

    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay)
    axes[2].set_title('Denoised + Uncertainty Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_component_denoisers(
    noisy_input: torch.Tensor,
    denoised_components: Dict[str, torch.Tensor],
    final_output: torch.Tensor,
    symbolic_weights: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
):
    """
    Compare outputs from all component denoisers

    Args:
        noisy_input: Noisy input [1, 1, H, W]
        denoised_components: Dict of component-denoised images
        final_output: Final fused output [1, 1, H, W]
        symbolic_weights: Noise composition weights
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert to numpy
    noisy = noisy_input[0, 0].cpu().numpy()
    final = final_output[0, 0].cpu().numpy()

    # 1. Noisy input
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Noisy Input')
    axes[0, 0].axis('off')

    # 2-5. Component denoisers
    components = ['speckle', 'banding', 'gaussian', 'shot']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]

    for comp, pos in zip(components, positions):
        img = denoised_components[comp][0, 0].cpu().numpy()
        weight = symbolic_weights[comp][0].item() * 100

        axes[pos].imshow(img, cmap='gray')
        axes[pos].set_title(f'{comp.capitalize()} Denoiser\n(weight: {weight:.1f}%)')
        axes[pos].axis('off')

    # 6. Final fused output
    axes[1, 2].imshow(final, cmap='gray')
    axes[1, 2].set_title('Final Fused Output')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def generate_noise_report(
    symbolic_weights: Dict[str, torch.Tensor],
    confidence: torch.Tensor,
    features: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate clinical-friendly noise analysis report

    Args:
        symbolic_weights: Noise composition weights
        confidence: Overall confidence score
        features: Extracted features
        save_path: Optional path to save report

    Returns:
        report: Text report
    """
    report = []
    report.append("=" * 60)
    report.append("NSND-OCT Noise Analysis Report")
    report.append("=" * 60)
    report.append("")

    # Noise composition
    report.append("Detected Noise Composition:")
    components = ['speckle', 'banding', 'gaussian', 'shot']

    for comp in components:
        weight_val = symbolic_weights[comp]
        if hasattr(weight_val, 'item'):
            weight = weight_val.item() if weight_val.numel() == 1 else weight_val[0].item()
        else:
            weight = float(weight_val)
        weight_pct = weight * 100
        bar = "█" * int(weight_pct / 5) + "░" * (20 - int(weight_pct / 5))
        report.append(f"  {comp.capitalize():12s}: {bar} {weight_pct:5.1f}%")

    report.append("")

    # Confidence
    conf = confidence[0].item() * 100
    report.append(f"Analysis Confidence: {conf:.1f}%")

    if conf > 80:
        report.append("  → HIGH confidence - Reliable noise decomposition")
    elif conf > 50:
        report.append("  → MEDIUM confidence - Generally reliable")
    else:
        report.append("  → LOW confidence - Results may be uncertain")

    report.append("")

    # Feature statistics
    report.append("Feature Statistics:")
    cv_mean = features['cv_map'][0].mean().item()
    kurt_mean = features['kurtosis'][0].mean().item()
    vpower = features['vertical_power'][0, 0, 0, 0].item()

    report.append(f"  Mean CV:              {cv_mean:.3f}")
    report.append(f"  Mean Kurtosis:        {kurt_mean:.2f}")
    report.append(f"  Vertical FFT Power:   {vpower:.4f}")

    report.append("")

    # Clinical interpretation
    report.append("Clinical Interpretation:")

    max_component = max(symbolic_weights.items(), key=lambda x: x[1][0].item() if x[0] != '_confidence' else 0)

    if max_component[0] == 'speckle':
        report.append("  → Dominant SPECKLE noise detected")
        report.append("    Likely due to coherent interference in OCT imaging")
        report.append("    Recommendation: Anisotropic diffusion-based denoising")

    elif max_component[0] == 'banding':
        report.append("  → Significant BANDING artifacts detected")
        report.append("    May indicate scanner calibration issues")
        report.append("    Recommendation: Frequency-domain filtering + scanner check")

    elif max_component[0] == 'gaussian':
        report.append("  → Prominent GAUSSIAN noise detected")
        report.append("    Likely from thermal/electronic noise in detector")
        report.append("    Recommendation: Standard Gaussian denoising")

    elif max_component[0] == 'shot':
        report.append("  → SHOT NOISE (Poisson) detected")
        report.append("    Depth-dependent photon counting noise")
        report.append("    Recommendation: Variance-stabilizing transform")

    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)

    return report_text
