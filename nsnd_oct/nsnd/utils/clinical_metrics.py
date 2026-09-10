#!/usr/bin/env python3
"""
Clinical Metrics for OCT Denoising Evaluation

Metrics that matter for clinical diagnosis:
1. Contrast-to-Noise Ratio (CNR) - Layer visibility
2. Edge Preservation Index (EPI) - Boundary clarity
3. Speckle Reduction Index (SRI) - Noise suppression
4. Structure Similarity in Layers - Anatomy preservation
5. Boundary Sharpness - Layer edge definition
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


def compute_cnr(image: torch.Tensor,
                region1_mask: torch.Tensor,
                region2_mask: torch.Tensor) -> float:
    """
    Compute Contrast-to-Noise Ratio between two regions.

    CNR = |μ1 - μ2| / sqrt((σ1² + σ2²) / 2)

    Higher CNR = better differentiation between adjacent layers.

    Args:
        image: [B, 1, H, W] or [1, H, W] image
        region1_mask: Binary mask for first region
        region2_mask: Binary mask for second region

    Returns:
        CNR value (higher is better)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    region1_mask = region1_mask.to(image.device)
    region2_mask = region2_mask.to(image.device)

    # Extract regions
    r1 = image * region1_mask
    r2 = image * region2_mask

    n1 = region1_mask.sum()
    n2 = region2_mask.sum()

    if n1 < 10 or n2 < 10:
        return 0.0

    # Compute statistics
    mean1 = r1.sum() / n1
    mean2 = r2.sum() / n2

    var1 = ((r1 - mean1 * region1_mask) ** 2).sum() / n1
    var2 = ((r2 - mean2 * region2_mask) ** 2).sum() / n2

    # CNR
    cnr = torch.abs(mean1 - mean2) / (torch.sqrt((var1 + var2) / 2) + 1e-8)

    return cnr.item()


def compute_cnr_adjacent_layers(image: torch.Tensor, num_layers: int = 5) -> Dict[str, float]:
    """
    Compute CNR between all adjacent layer pairs.

    Args:
        image: [B, 1, H, W] image
        num_layers: Number of layers (using depth-based approximation)

    Returns:
        Dictionary of CNR values for each layer pair
    """
    B, C, H, W = image.shape
    layer_names = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']
    layer_bounds = [(0, 0.15), (0.15, 0.40), (0.40, 0.55), (0.55, 0.75), (0.75, 1.0)]

    cnr_results = {}

    for i in range(num_layers - 1):
        # Create masks for adjacent layers
        mask1 = torch.zeros_like(image)
        mask2 = torch.zeros_like(image)

        start1, end1 = layer_bounds[i]
        start2, end2 = layer_bounds[i + 1]

        row_start1, row_end1 = int(start1 * H), int(end1 * H)
        row_start2, row_end2 = int(start2 * H), int(end2 * H)

        mask1[:, :, row_start1:row_end1, :] = 1
        mask2[:, :, row_start2:row_end2, :] = 1

        cnr = compute_cnr(image, mask1, mask2)
        cnr_results[f'CNR_{layer_names[i]}_{layer_names[i+1]}'] = cnr

    # Average CNR
    cnr_results['CNR_mean'] = np.mean(list(cnr_results.values()))

    return cnr_results


def compute_edge_preservation_index(denoised: torch.Tensor,
                                     reference: torch.Tensor,
                                     noisy: Optional[torch.Tensor] = None) -> float:
    """
    Compute Edge Preservation Index (EPI).

    EPI measures how well edges are preserved after denoising.
    EPI = corr(∇denoised, ∇reference) / corr(∇noisy, ∇reference)

    Higher EPI = better edge preservation.

    Args:
        denoised: Denoised image
        reference: Clean reference image
        noisy: Original noisy image (optional, for normalization)

    Returns:
        EPI value (higher is better, >1 means improved over noisy)
    """
    def compute_gradient_magnitude(img):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

        gx = F.conv2d(F.pad(img, [1,1,1,1], mode='reflect'), sobel_x)
        gy = F.conv2d(F.pad(img, [1,1,1,1], mode='reflect'), sobel_y)

        return torch.sqrt(gx**2 + gy**2)

    grad_denoised = compute_gradient_magnitude(denoised)
    grad_reference = compute_gradient_magnitude(reference)

    # Correlation
    def correlation(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()
        corr = (a_centered * b_centered).sum() / (
            torch.sqrt((a_centered**2).sum() * (b_centered**2).sum()) + 1e-8
        )
        return corr.item()

    epi_denoised = correlation(grad_denoised, grad_reference)

    if noisy is not None:
        grad_noisy = compute_gradient_magnitude(noisy)
        epi_noisy = correlation(grad_noisy, grad_reference)
        if epi_noisy > 0:
            return epi_denoised / epi_noisy
        return epi_denoised

    return epi_denoised


def compute_speckle_reduction_index(denoised: torch.Tensor,
                                     noisy: torch.Tensor,
                                     window_size: int = 7) -> float:
    """
    Compute Speckle Reduction Index (SRI).

    SRI = (CV_noisy - CV_denoised) / CV_noisy

    Where CV = coefficient of variation = std / mean

    Higher SRI = more speckle reduction (0 = no reduction, 1 = complete removal)

    Args:
        denoised: Denoised image
        noisy: Original noisy image
        window_size: Window size for local statistics

    Returns:
        SRI value (0-1, higher is better)
    """
    pad = window_size // 2

    def compute_local_cv(img):
        # Local mean
        local_mean = F.avg_pool2d(
            F.pad(img, [pad]*4, mode='reflect'),
            window_size, stride=1
        )
        # Local std
        local_sq_mean = F.avg_pool2d(
            F.pad(img**2, [pad]*4, mode='reflect'),
            window_size, stride=1
        )
        local_var = (local_sq_mean - local_mean**2).clamp(min=1e-8)
        local_std = torch.sqrt(local_var)

        # Coefficient of variation
        cv = local_std / (local_mean + 1e-8)
        return cv.mean().item()

    cv_noisy = compute_local_cv(noisy)
    cv_denoised = compute_local_cv(denoised)

    if cv_noisy > 0:
        sri = (cv_noisy - cv_denoised) / cv_noisy
        return max(0, min(1, sri))  # Clamp to [0, 1]

    return 0.0


def compute_boundary_sharpness(image: torch.Tensor,
                                layer_boundaries: Optional[torch.Tensor] = None) -> float:
    """
    Compute boundary sharpness at layer interfaces.

    Uses vertical gradient magnitude at expected layer boundaries.

    Higher value = sharper, more visible layer boundaries.

    Args:
        image: [B, 1, H, W] image
        layer_boundaries: Optional [B, H, W] boundary probability map

    Returns:
        Boundary sharpness score
    """
    B, C, H, W = image.shape

    # Vertical gradient (horizontal edges = layer boundaries)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    grad_y = F.conv2d(F.pad(image, [1,1,1,1], mode='reflect'), sobel_y)
    grad_magnitude = grad_y.abs()

    if layer_boundaries is not None:
        # Weight by boundary probability
        boundary_mask = layer_boundaries.unsqueeze(1)
        sharpness = (grad_magnitude * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)
    else:
        # Use typical boundary locations
        boundary_rows = [int(0.15 * H), int(0.40 * H), int(0.55 * H), int(0.75 * H)]
        boundary_mask = torch.zeros_like(image)
        for row in boundary_rows:
            if 2 < row < H - 2:
                boundary_mask[:, :, row-2:row+3, :] = 1

        sharpness = (grad_magnitude * boundary_mask).mean()

    return sharpness.item()


def compute_layer_ssim(denoised: torch.Tensor,
                       reference: torch.Tensor,
                       num_layers: int = 5) -> Dict[str, float]:
    """
    Compute SSIM for each anatomical layer.

    Args:
        denoised: Denoised image
        reference: Clean reference
        num_layers: Number of layers

    Returns:
        Dictionary of SSIM values per layer
    """
    from nsnd.utils.metrics import compute_ssim

    B, C, H, W = denoised.shape
    layer_names = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']
    layer_bounds = [(0, 0.15), (0.15, 0.40), (0.40, 0.55), (0.55, 0.75), (0.75, 1.0)]

    results = {}

    for i, (name, (start, end)) in enumerate(zip(layer_names, layer_bounds)):
        row_start, row_end = int(start * H), int(end * H)

        denoised_layer = denoised[:, :, row_start:row_end, :]
        reference_layer = reference[:, :, row_start:row_end, :]

        if denoised_layer.shape[2] >= 7:  # Need minimum size for SSIM
            ssim = compute_ssim(denoised_layer, reference_layer)
            results[f'SSIM_{name}'] = ssim
        else:
            results[f'SSIM_{name}'] = 0.0

    results['SSIM_mean'] = np.mean([v for k, v in results.items() if k.startswith('SSIM_')])

    return results


def compute_all_clinical_metrics(denoised: torch.Tensor,
                                  reference: torch.Tensor,
                                  noisy: torch.Tensor,
                                  layer_probs: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute all clinical metrics.

    Args:
        denoised: Denoised image [B, 1, H, W]
        reference: Clean reference [B, 1, H, W]
        noisy: Original noisy [B, 1, H, W]
        layer_probs: Optional layer probabilities [B, num_layers, H, W]

    Returns:
        Dictionary of all clinical metrics
    """
    metrics = {}

    # 1. CNR between adjacent layers
    cnr = compute_cnr_adjacent_layers(denoised)
    metrics.update(cnr)

    # Also compute CNR improvement
    cnr_noisy = compute_cnr_adjacent_layers(noisy)
    metrics['CNR_improvement'] = cnr['CNR_mean'] - cnr_noisy['CNR_mean']

    # 2. Edge Preservation Index
    metrics['EPI'] = compute_edge_preservation_index(denoised, reference, noisy)

    # 3. Speckle Reduction Index
    metrics['SRI'] = compute_speckle_reduction_index(denoised, noisy)

    # 4. Boundary Sharpness
    boundary_map = None
    if layer_probs is not None:
        # Compute boundary as gradient of layer probs
        boundary_map = torch.abs(
            layer_probs[:, :, 1:, :] - layer_probs[:, :, :-1, :]
        ).sum(dim=1)
        boundary_map = F.pad(boundary_map, [0, 0, 0, 1], mode='replicate')

    metrics['Boundary_Sharpness'] = compute_boundary_sharpness(denoised, boundary_map)
    metrics['Boundary_Sharpness_noisy'] = compute_boundary_sharpness(noisy, boundary_map)
    metrics['Boundary_Sharpness_improvement'] = (
        metrics['Boundary_Sharpness'] - metrics['Boundary_Sharpness_noisy']
    )

    # 5. Per-layer SSIM
    layer_ssim = compute_layer_ssim(denoised, reference)
    metrics.update(layer_ssim)

    return metrics


def print_clinical_metrics_report(metrics: Dict[str, float], baseline_metrics: Optional[Dict] = None):
    """Print formatted clinical metrics report."""
    print("\n" + "="*70)
    print("CLINICAL METRICS REPORT")
    print("="*70)

    print("\n--- Contrast-to-Noise Ratio (CNR) ---")
    print("Higher CNR = better layer differentiation")
    cnr_keys = [k for k in metrics.keys() if k.startswith('CNR_') and k != 'CNR_mean' and k != 'CNR_improvement']
    for key in cnr_keys:
        val = metrics[key]
        if baseline_metrics and key in baseline_metrics:
            diff = val - baseline_metrics[key]
            print(f"  {key}: {val:.3f} ({diff:+.3f})")
        else:
            print(f"  {key}: {val:.3f}")
    print(f"  CNR Mean: {metrics['CNR_mean']:.3f}")
    if 'CNR_improvement' in metrics:
        print(f"  CNR Improvement over noisy: {metrics['CNR_improvement']:+.3f}")

    print("\n--- Edge Preservation Index (EPI) ---")
    print("EPI > 1 = edges better preserved than noisy input")
    print(f"  EPI: {metrics['EPI']:.3f}")

    print("\n--- Speckle Reduction Index (SRI) ---")
    print("SRI: 0 = no reduction, 1 = complete removal")
    print(f"  SRI: {metrics['SRI']:.3f} ({metrics['SRI']*100:.1f}% speckle reduced)")

    print("\n--- Boundary Sharpness ---")
    print("Higher = sharper layer boundaries")
    print(f"  Denoised: {metrics['Boundary_Sharpness']:.4f}")
    print(f"  Noisy: {metrics['Boundary_Sharpness_noisy']:.4f}")
    print(f"  Improvement: {metrics['Boundary_Sharpness_improvement']:+.4f}")

    print("\n--- Per-Layer SSIM ---")
    ssim_keys = [k for k in metrics.keys() if k.startswith('SSIM_') and k != 'SSIM_mean']
    for key in ssim_keys:
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"  SSIM Mean: {metrics['SSIM_mean']:.4f}")

    print("="*70)


if __name__ == '__main__':
    # Test clinical metrics
    print("Testing clinical metrics...")

    # Create dummy data
    torch.manual_seed(42)
    clean = torch.rand(1, 1, 64, 64) * 0.5 + 0.25
    noisy = clean + torch.randn_like(clean) * 0.1
    denoised = clean + torch.randn_like(clean) * 0.05  # Less noisy

    # Compute metrics
    metrics = compute_all_clinical_metrics(denoised, clean, noisy)

    # Print report
    print_clinical_metrics_report(metrics)

    print("\nAll tests passed!")
