"""
Anatomy-Specific Metrics for OCT Denoising Evaluation.

This module provides metrics that specifically evaluate:
1. Layer-specific PSNR/SSIM (per retinal zone)
2. Boundary/edge preservation
3. Contrast-to-noise ratio (CNR)
4. Clinical relevance metrics

These metrics better demonstrate the value of anatomy-aware denoising
compared to global PSNR/SSIM alone.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from .metrics import compute_psnr, compute_ssim


# Retinal layer zones (approximate depth percentages in B-scans)
LAYER_ZONES = {
    'vitreous_nfl': (0.0, 0.20),      # NFL, GCL - top bright layers
    'inner_retina': (0.20, 0.40),      # IPL, INL - inner plexiform
    'outer_nuclear': (0.40, 0.60),     # OPL, ONL - nuclear layers (darker)
    'photoreceptors': (0.60, 0.80),    # IS/OS junction - bright band
    'rpe_choroid': (0.80, 1.0),        # RPE, choroid - bottom layers
}


def compute_layer_mask(height: int, zone: Tuple[float, float], device: torch.device) -> torch.Tensor:
    """Create a binary mask for a specific layer zone."""
    start_pct, end_pct = zone
    start_row = int(start_pct * height)
    end_row = int(end_pct * height)

    mask = torch.zeros(height, device=device)
    mask[start_row:end_row] = 1.0
    return mask


def compute_layer_specific_psnr(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    zones: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Compute PSNR separately for each retinal layer zone.

    Args:
        denoised: Denoised output [B, 1, H, W] or [1, H, W]
        clean: Clean target [B, 1, H, W] or [1, H, W]
        zones: Dictionary of zone_name -> (start_pct, end_pct)

    Returns:
        Dictionary of zone_name -> PSNR value
    """
    if zones is None:
        zones = LAYER_ZONES

    if denoised.ndim == 3:
        denoised = denoised.unsqueeze(0)
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)

    B, C, H, W = denoised.shape
    device = denoised.device

    layer_psnr = {}

    for zone_name, (start_pct, end_pct) in zones.items():
        start_row = int(start_pct * H)
        end_row = int(end_pct * H)

        # Extract zone
        denoised_zone = denoised[:, :, start_row:end_row, :]
        clean_zone = clean[:, :, start_row:end_row, :]

        # Compute PSNR for this zone
        mse = torch.mean((denoised_zone - clean_zone) ** 2)
        if mse.item() == 0:
            layer_psnr[zone_name] = float('inf')
        else:
            psnr = 20 * torch.log10(torch.tensor(1.0, device=device) / torch.sqrt(mse))
            layer_psnr[zone_name] = psnr.item()

    return layer_psnr


def compute_layer_specific_ssim(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    zones: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Compute SSIM separately for each retinal layer zone.
    """
    if zones is None:
        zones = LAYER_ZONES

    if denoised.ndim == 3:
        denoised = denoised.unsqueeze(0)
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)

    B, C, H, W = denoised.shape

    layer_ssim = {}

    for zone_name, (start_pct, end_pct) in zones.items():
        start_row = int(start_pct * H)
        end_row = int(end_pct * H)

        # Extract zone
        denoised_zone = denoised[:, :, start_row:end_row, :]
        clean_zone = clean[:, :, start_row:end_row, :]

        # Compute SSIM for this zone (only if zone is large enough)
        zone_height = end_row - start_row
        if zone_height >= 11:  # Minimum for SSIM window
            ssim_val = compute_ssim(denoised_zone, clean_zone)
            layer_ssim[zone_name] = ssim_val
        else:
            layer_ssim[zone_name] = 0.0

    return layer_ssim


def compute_edge_preservation_index(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Edge Preservation Index (EPI) - how well edges are preserved.

    Higher EPI = better edge preservation
    """
    # Sobel filters for edge detection
    sobel_x = torch.tensor([
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    ], dtype=torch.float32, device=denoised.device).unsqueeze(0)

    sobel_y = torch.tensor([
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    ], dtype=torch.float32, device=denoised.device).unsqueeze(0)

    # Compute edges for all images
    clean_edge_x = F.conv2d(clean, sobel_x, padding=1)
    clean_edge_y = F.conv2d(clean, sobel_y, padding=1)
    clean_edges = torch.sqrt(clean_edge_x**2 + clean_edge_y**2)

    denoised_edge_x = F.conv2d(denoised, sobel_x, padding=1)
    denoised_edge_y = F.conv2d(denoised, sobel_y, padding=1)
    denoised_edges = torch.sqrt(denoised_edge_x**2 + denoised_edge_y**2)

    noisy_edge_x = F.conv2d(noisy, sobel_x, padding=1)
    noisy_edge_y = F.conv2d(noisy, sobel_y, padding=1)
    noisy_edges = torch.sqrt(noisy_edge_x**2 + noisy_edge_y**2)

    # Edge Preservation Index
    # EPI = correlation between denoised edges and clean edges
    clean_flat = clean_edges.flatten()
    denoised_flat = denoised_edges.flatten()
    noisy_flat = noisy_edges.flatten()

    # Normalize
    clean_norm = (clean_flat - clean_flat.mean()) / (clean_flat.std() + 1e-8)
    denoised_norm = (denoised_flat - denoised_flat.mean()) / (denoised_flat.std() + 1e-8)
    noisy_norm = (noisy_flat - noisy_flat.mean()) / (noisy_flat.std() + 1e-8)

    # Correlation
    epi_denoised = (clean_norm * denoised_norm).mean().item()
    epi_noisy = (clean_norm * noisy_norm).mean().item()

    # Edge MSE (lower is better)
    edge_mse_denoised = F.mse_loss(denoised_edges, clean_edges).item()
    edge_mse_noisy = F.mse_loss(noisy_edges, clean_edges).item()

    return {
        'epi_denoised': epi_denoised,
        'epi_noisy': epi_noisy,
        'epi_improvement': epi_denoised - epi_noisy,
        'edge_mse_denoised': edge_mse_denoised,
        'edge_mse_noisy': edge_mse_noisy,
        'edge_mse_improvement': edge_mse_noisy - edge_mse_denoised,
    }


def compute_horizontal_edge_preservation(
    denoised: torch.Tensor,
    clean: torch.Tensor,
) -> float:
    """
    Specifically measures preservation of horizontal edges (layer boundaries).

    OCT B-scans have prominent horizontal layer boundaries.
    This metric measures how well these are preserved.
    """
    # Vertical Sobel (detects horizontal edges)
    sobel_v = torch.tensor([
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    ], dtype=torch.float32, device=denoised.device).unsqueeze(0)

    clean_h_edges = F.conv2d(clean, sobel_v, padding=1)
    denoised_h_edges = F.conv2d(denoised, sobel_v, padding=1)

    # Measure preservation of strong edges
    # Strong edges in clean image
    edge_threshold = clean_h_edges.abs().quantile(0.9)
    strong_edge_mask = clean_h_edges.abs() > edge_threshold

    if strong_edge_mask.sum() > 0:
        # How well are these strong edges preserved?
        clean_strong = clean_h_edges[strong_edge_mask]
        denoised_strong = denoised_h_edges[strong_edge_mask]

        # Correlation
        clean_norm = (clean_strong - clean_strong.mean()) / (clean_strong.std() + 1e-8)
        denoised_norm = (denoised_strong - denoised_strong.mean()) / (denoised_strong.std() + 1e-8)

        preservation = (clean_norm * denoised_norm).mean().item()
    else:
        preservation = 0.0

    return preservation


def compute_contrast_to_noise_ratio(
    image: torch.Tensor,
    zone1: Tuple[float, float],
    zone2: Tuple[float, float],
) -> float:
    """
    Compute Contrast-to-Noise Ratio between two adjacent zones.

    CNR = |mean(zone1) - mean(zone2)| / sqrt(var(zone1) + var(zone2))

    Higher CNR = better contrast between layers (clinically important)
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape

    start1, end1 = int(zone1[0] * H), int(zone1[1] * H)
    start2, end2 = int(zone2[0] * H), int(zone2[1] * H)

    region1 = image[:, :, start1:end1, :]
    region2 = image[:, :, start2:end2, :]

    mean1 = region1.mean()
    mean2 = region2.mean()
    var1 = region1.var()
    var2 = region2.var()

    cnr = abs(mean1 - mean2) / (torch.sqrt(var1 + var2) + 1e-8)

    return cnr.item()


def compute_layer_cnr(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """
    Compute CNR between adjacent layer zones.

    Returns CNR for clean, noisy, and denoised images.
    """
    adjacent_pairs = [
        ('vitreous_nfl', 'inner_retina'),
        ('inner_retina', 'outer_nuclear'),
        ('outer_nuclear', 'photoreceptors'),
        ('photoreceptors', 'rpe_choroid'),
    ]

    results = {}

    for zone1_name, zone2_name in adjacent_pairs:
        zone1 = LAYER_ZONES[zone1_name]
        zone2 = LAYER_ZONES[zone2_name]

        pair_name = f'{zone1_name}_{zone2_name}'

        cnr_clean = compute_contrast_to_noise_ratio(clean, zone1, zone2)
        cnr_noisy = compute_contrast_to_noise_ratio(noisy, zone1, zone2)
        cnr_denoised = compute_contrast_to_noise_ratio(denoised, zone1, zone2)

        results[pair_name] = {
            'clean': cnr_clean,
            'noisy': cnr_noisy,
            'denoised': cnr_denoised,
            'improvement': cnr_denoised - cnr_noisy,
        }

    return results


def compute_gradient_magnitude_similarity(
    denoised: torch.Tensor,
    clean: torch.Tensor,
) -> float:
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD).

    GMSD is often better than SSIM for measuring structural distortion.
    Lower GMSD = better quality (unlike SSIM where higher is better).
    """
    # Prewitt filters for gradient
    h_x = torch.tensor([
        [[1/3, 0, -1/3],
         [1/3, 0, -1/3],
         [1/3, 0, -1/3]]
    ], dtype=torch.float32, device=denoised.device).unsqueeze(0)

    h_y = torch.tensor([
        [[1/3, 1/3, 1/3],
         [0, 0, 0],
         [-1/3, -1/3, -1/3]]
    ], dtype=torch.float32, device=denoised.device).unsqueeze(0)

    # Compute gradients
    clean_gx = F.conv2d(clean, h_x, padding=1)
    clean_gy = F.conv2d(clean, h_y, padding=1)
    clean_gm = torch.sqrt(clean_gx**2 + clean_gy**2)

    denoised_gx = F.conv2d(denoised, h_x, padding=1)
    denoised_gy = F.conv2d(denoised, h_y, padding=1)
    denoised_gm = torch.sqrt(denoised_gx**2 + denoised_gy**2)

    # Gradient Magnitude Similarity
    c = 0.0026  # constant for numerical stability
    gms = (2 * clean_gm * denoised_gm + c) / (clean_gm**2 + denoised_gm**2 + c)

    # GMSD is the standard deviation of GMS
    gmsd = torch.std(gms).item()

    # Also return mean GMS (higher is better, like SSIM)
    mean_gms = torch.mean(gms).item()

    return {'gmsd': gmsd, 'mean_gms': mean_gms}


def compute_all_anatomy_metrics(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compute all anatomy-specific metrics.

    Returns comprehensive metrics for paper reporting.
    """
    results = {
        'global': {},
        'layer_specific': {},
        'edge_preservation': {},
        'contrast': {},
        'gradient': {},
    }

    # Global metrics
    results['global']['psnr_noisy'] = compute_psnr(noisy, clean)
    results['global']['psnr_denoised'] = compute_psnr(denoised, clean)
    results['global']['ssim_denoised'] = compute_ssim(denoised, clean)

    if baseline is not None:
        results['global']['psnr_baseline'] = compute_psnr(baseline, clean)
        results['global']['ssim_baseline'] = compute_ssim(baseline, clean)

    # Layer-specific metrics
    results['layer_specific']['psnr'] = compute_layer_specific_psnr(denoised, clean)
    results['layer_specific']['ssim'] = compute_layer_specific_ssim(denoised, clean)

    if baseline is not None:
        results['layer_specific']['psnr_baseline'] = compute_layer_specific_psnr(baseline, clean)
        results['layer_specific']['ssim_baseline'] = compute_layer_specific_ssim(baseline, clean)

    # Edge preservation
    results['edge_preservation'] = compute_edge_preservation_index(denoised, clean, noisy)
    results['edge_preservation']['horizontal'] = compute_horizontal_edge_preservation(denoised, clean)

    if baseline is not None:
        baseline_edge = compute_edge_preservation_index(baseline, clean, noisy)
        results['edge_preservation']['baseline_epi'] = baseline_edge['epi_denoised']
        results['edge_preservation']['baseline_horizontal'] = compute_horizontal_edge_preservation(baseline, clean)

    # CNR
    results['contrast'] = compute_layer_cnr(denoised, clean, noisy)

    # GMSD
    results['gradient'] = compute_gradient_magnitude_similarity(denoised, clean)
    if baseline is not None:
        baseline_gmsd = compute_gradient_magnitude_similarity(baseline, clean)
        results['gradient']['baseline_gmsd'] = baseline_gmsd['gmsd']
        results['gradient']['baseline_mean_gms'] = baseline_gmsd['mean_gms']

    return results


def print_anatomy_metrics_report(metrics: Dict, show_baseline: bool = True):
    """Print a formatted report of anatomy metrics."""
    print("\n" + "=" * 70)
    print("ANATOMY-SPECIFIC METRICS REPORT")
    print("=" * 70)

    # Global
    print("\n--- Global Metrics ---")
    print(f"  PSNR (noisy):    {metrics['global']['psnr_noisy']:.2f} dB")
    print(f"  PSNR (denoised): {metrics['global']['psnr_denoised']:.2f} dB")
    if show_baseline and 'psnr_baseline' in metrics['global']:
        print(f"  PSNR (baseline): {metrics['global']['psnr_baseline']:.2f} dB")
        gain = metrics['global']['psnr_denoised'] - metrics['global']['psnr_baseline']
        print(f"  PSNR gain:       {'+' if gain >= 0 else ''}{gain:.2f} dB")
    print(f"  SSIM (denoised): {metrics['global']['ssim_denoised']:.4f}")
    if show_baseline and 'ssim_baseline' in metrics['global']:
        print(f"  SSIM (baseline): {metrics['global']['ssim_baseline']:.4f}")
        ssim_gain = metrics['global']['ssim_denoised'] - metrics['global']['ssim_baseline']
        print(f"  SSIM gain:       {'+' if ssim_gain >= 0 else ''}{ssim_gain:.4f}")

    # Layer-specific
    print("\n--- Layer-Specific PSNR (dB) ---")
    print(f"  {'Zone':<18} {'Denoised':>10}", end='')
    if show_baseline and 'psnr_baseline' in metrics['layer_specific']:
        print(f" {'Baseline':>10} {'Gain':>10}", end='')
    print()

    for zone in LAYER_ZONES.keys():
        psnr = metrics['layer_specific']['psnr'].get(zone, 0)
        print(f"  {zone:<18} {psnr:>10.2f}", end='')
        if show_baseline and 'psnr_baseline' in metrics['layer_specific']:
            base_psnr = metrics['layer_specific']['psnr_baseline'].get(zone, 0)
            gain = psnr - base_psnr
            print(f" {base_psnr:>10.2f} {'+' if gain >= 0 else ''}{gain:>9.2f}", end='')
        print()

    # Layer-specific SSIM
    print("\n--- Layer-Specific SSIM ---")
    print(f"  {'Zone':<18} {'Denoised':>10}", end='')
    if show_baseline and 'ssim_baseline' in metrics['layer_specific']:
        print(f" {'Baseline':>10} {'Gain':>10}", end='')
    print()

    for zone in LAYER_ZONES.keys():
        ssim = metrics['layer_specific']['ssim'].get(zone, 0)
        print(f"  {zone:<18} {ssim:>10.4f}", end='')
        if show_baseline and 'ssim_baseline' in metrics['layer_specific']:
            base_ssim = metrics['layer_specific']['ssim_baseline'].get(zone, 0)
            gain = ssim - base_ssim
            print(f" {base_ssim:>10.4f} {'+' if gain >= 0 else ''}{gain:>9.4f}", end='')
        print()

    # Edge preservation
    print("\n--- Edge Preservation ---")
    print(f"  EPI (denoised):     {metrics['edge_preservation']['epi_denoised']:.4f}")
    print(f"  EPI (noisy):        {metrics['edge_preservation']['epi_noisy']:.4f}")
    print(f"  EPI improvement:    {metrics['edge_preservation']['epi_improvement']:.4f}")
    print(f"  Horizontal edges:   {metrics['edge_preservation']['horizontal']:.4f}")
    if show_baseline and 'baseline_epi' in metrics['edge_preservation']:
        print(f"  EPI (baseline):     {metrics['edge_preservation']['baseline_epi']:.4f}")
        print(f"  Horiz (baseline):   {metrics['edge_preservation']['baseline_horizontal']:.4f}")

    # Gradient similarity
    print("\n--- Gradient Magnitude Similarity ---")
    print(f"  GMSD (denoised):    {metrics['gradient']['gmsd']:.4f} (lower is better)")
    print(f"  Mean GMS:           {metrics['gradient']['mean_gms']:.4f} (higher is better)")
    if show_baseline and 'baseline_gmsd' in metrics['gradient']:
        print(f"  GMSD (baseline):    {metrics['gradient']['baseline_gmsd']:.4f}")
        print(f"  Mean GMS (base):    {metrics['gradient']['baseline_mean_gms']:.4f}")

    # CNR summary
    print("\n--- Contrast-to-Noise Ratio (CNR) ---")
    total_cnr_improvement = 0
    count = 0
    for pair, values in metrics['contrast'].items():
        improvement = values['improvement']
        total_cnr_improvement += improvement
        count += 1
        print(f"  {pair}: {values['denoised']:.3f} (improve: {'+' if improvement >= 0 else ''}{improvement:.3f})")

    avg_cnr_improvement = total_cnr_improvement / count if count > 0 else 0
    print(f"  Average CNR improvement: {'+' if avg_cnr_improvement >= 0 else ''}{avg_cnr_improvement:.3f}")

    print("\n" + "=" * 70)
