#!/usr/bin/env python3
"""
Analyze Backbone Weaknesses for Symbolic Correction

Goal: Find WHERE the backbone fails so correctors can target those areas.
Focus: Clinical utility over PSNR.

Analysis:
1. Error distribution by region type (edges, boundaries, smooth areas)
2. Clinical metric failures per region
3. Systematic patterns in backbone errors
4. Actionable insights for corrector design
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))


def load_image(path: str) -> torch.Tensor:
    """Load image as tensor [1, 1, H, W]."""
    img = np.array(Image.open(path)).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def compute_edge_map(img: torch.Tensor) -> torch.Tensor:
    """Compute edge magnitude using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

    gx = F.conv2d(F.pad(img, [1, 1, 1, 1], mode='reflect'), sobel_x)
    gy = F.conv2d(F.pad(img, [1, 1, 1, 1], mode='reflect'), sobel_y)
    return torch.sqrt(gx**2 + gy**2)


def compute_local_variance(img: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Compute local variance map."""
    padding = kernel_size // 2

    # Local mean
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=img.dtype, device=img.device)
    kernel = kernel / (kernel_size * kernel_size)

    local_mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
    local_mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)

    local_var = local_mean_sq - local_mean**2
    return local_var.clamp(min=0)


def identify_regions(clean: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Segment image into clinically relevant regions.

    Returns masks for:
    - edges: High gradient regions (layer boundaries)
    - smooth: Low variance homogeneous regions
    - texture: Medium variance regions
    - bright: High intensity (NFL, RPE)
    - dark: Low intensity (vitreous, choroid)
    """
    B, C, H, W = clean.shape
    edge_map = compute_edge_map(clean)
    var_map = compute_local_variance(clean)

    # Ensure same size as clean
    edge_map = edge_map[:, :, :H, :W]
    var_map = var_map[:, :, :H, :W]

    # Normalize for thresholding
    edge_norm = edge_map / (edge_map.max() + 1e-8)
    var_norm = var_map / (var_map.max() + 1e-8)

    # Define regions
    regions = {
        'edge': edge_norm > 0.15,           # Strong edges (layer boundaries)
        'smooth': var_norm < 0.02,           # Homogeneous regions
        'texture': (var_norm >= 0.02) & (var_norm < 0.1) & (edge_norm <= 0.15),
        'bright': clean > 0.7,               # High reflectivity layers
        'dark': clean < 0.2,                 # Low reflectivity regions
        'boundary_zone': torch.zeros_like(clean, dtype=torch.bool),  # Layer boundaries
    }

    # Identify horizontal layer boundaries (vertical gradient)
    vert_grad = F.conv2d(F.pad(clean, [0, 0, 1, 1], mode='reflect'),
                         torch.tensor([[-1], [0], [1]], dtype=clean.dtype).view(1, 1, 3, 1))
    vert_grad = vert_grad[:, :, :H, :W]
    regions['boundary_zone'] = vert_grad.abs() > 0.1

    return regions


def analyze_errors_by_region(error_map: torch.Tensor, regions: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
    """Analyze error statistics per region."""
    results = {}

    total_pixels = error_map.numel()

    for name, mask in regions.items():
        mask_float = mask.float()
        region_pixels = mask_float.sum().item()

        if region_pixels > 0:
            # Masked error statistics
            masked_error = error_map * mask_float
            region_mae = masked_error.abs().sum().item() / region_pixels
            region_mse = (masked_error**2).sum().item() / region_pixels
            region_max = (error_map.abs() * mask_float).max().item()

            # What fraction of total error comes from this region?
            total_abs_error = error_map.abs().sum().item()
            region_error_contribution = masked_error.abs().sum().item() / (total_abs_error + 1e-8)

            results[name] = {
                'pixels': int(region_pixels),
                'pixel_fraction': region_pixels / total_pixels,
                'mae': region_mae,
                'rmse': np.sqrt(region_mse),
                'max_error': region_max,
                'error_contribution': region_error_contribution,
                'error_density': region_mae / (total_abs_error / total_pixels + 1e-8),  # Relative to average
            }
        else:
            results[name] = {'pixels': 0, 'pixel_fraction': 0, 'mae': 0, 'rmse': 0,
                           'max_error': 0, 'error_contribution': 0, 'error_density': 0}

    return results


def analyze_clinical_failures(backbone_out: torch.Tensor, clean: torch.Tensor,
                             noisy: torch.Tensor) -> Dict[str, any]:
    """Analyze where backbone fails on clinical metrics."""

    results = {}

    # 1. Edge Preservation Analysis
    edge_clean = compute_edge_map(clean)
    edge_backbone = compute_edge_map(backbone_out)
    edge_diff = (edge_clean - edge_backbone).abs()

    # Where are edges lost?
    edge_loss_mask = (edge_clean > 0.1) & (edge_backbone < edge_clean * 0.7)
    edge_loss_fraction = edge_loss_mask.float().mean().item()

    results['edge_preservation'] = {
        'edges_lost_fraction': edge_loss_fraction,
        'mean_edge_difference': edge_diff.mean().item(),
        'worst_edge_loss': edge_diff.max().item(),
    }

    # 2. Contrast Analysis
    # Compare local contrast (std in windows)
    def local_contrast(img, size=15):
        padding = size // 2
        kernel = torch.ones(1, 1, size, size, dtype=img.dtype) / (size * size)
        mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
        mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)
        return torch.sqrt((mean_sq - mean**2).clamp(min=0))

    contrast_clean = local_contrast(clean)
    contrast_backbone = local_contrast(backbone_out)
    contrast_ratio = contrast_backbone / (contrast_clean + 1e-6)

    # Where is contrast reduced?
    contrast_loss_mask = contrast_ratio < 0.8

    results['contrast'] = {
        'contrast_reduced_fraction': contrast_loss_mask.float().mean().item(),
        'mean_contrast_ratio': contrast_ratio.mean().item(),
        'worst_contrast_loss': contrast_ratio.min().item(),
    }

    # 3. Layer Boundary Sharpness
    # Vertical gradient at typical boundary locations
    vert_grad_clean = F.conv2d(F.pad(clean, [0, 0, 1, 1], mode='reflect'),
                                torch.tensor([[[-1], [0], [1]]], dtype=clean.dtype).view(1, 1, 3, 1))
    vert_grad_backbone = F.conv2d(F.pad(backbone_out, [0, 0, 1, 1], mode='reflect'),
                                   torch.tensor([[[-1], [0], [1]]], dtype=backbone_out.dtype).view(1, 1, 3, 1))

    # Focus on strong boundaries
    boundary_mask = vert_grad_clean.abs() > 0.1
    if boundary_mask.sum() > 0:
        clean_boundary_strength = (vert_grad_clean.abs() * boundary_mask.float()).sum() / boundary_mask.sum()
        backbone_boundary_strength = (vert_grad_backbone.abs() * boundary_mask.float()).sum() / boundary_mask.sum()
        boundary_preservation = (backbone_boundary_strength / (clean_boundary_strength + 1e-6)).item()
    else:
        boundary_preservation = 1.0

    results['boundary_sharpness'] = {
        'boundary_preservation_ratio': boundary_preservation,
        'boundaries_blurred': boundary_preservation < 0.9,
    }

    # 4. Texture Preservation (in textured regions)
    var_clean = compute_local_variance(clean, kernel_size=5)
    var_backbone = compute_local_variance(backbone_out, kernel_size=5)

    # Textured regions in clean image
    texture_mask = (var_clean > 0.001) & (var_clean < 0.05)
    if texture_mask.sum() > 0:
        texture_ratio = (var_backbone * texture_mask.float()).sum() / (var_clean * texture_mask.float()).sum()
    else:
        texture_ratio = torch.tensor(1.0)

    results['texture'] = {
        'texture_preservation_ratio': texture_ratio.item(),
        'over_smoothed': texture_ratio.item() < 0.7,
    }

    # 5. Noise Residual Analysis
    residual = backbone_out - clean

    # Is residual correlated with edges? (Bad - means edge artifacts)
    edge_residual_corr = (residual.abs() * edge_clean).mean() / (residual.abs().mean() * edge_clean.mean() + 1e-8)

    results['residual_analysis'] = {
        'mean_residual': residual.mean().item(),
        'residual_std': residual.std().item(),
        'edge_correlated_error': edge_residual_corr.item(),  # >1 means errors concentrate at edges
    }

    return results


def run_analysis(val_jsonl: str, backbone_path: str, max_samples: int = 10):
    """Run full backbone weakness analysis."""

    print("=" * 70)
    print("BACKBONE WEAKNESS ANALYSIS FOR CLINICAL UTILITY")
    print("=" * 70)

    # Load backbone
    from nsnd.models.nafnet import NAFNetSmall
    backbone = NAFNetSmall(img_channel=1, width=40)

    ckpt = torch.load(backbone_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    backbone.load_state_dict(state_dict, strict=False)
    backbone.eval()
    print(f"Loaded backbone from {backbone_path}")

    # Load validation data
    samples = []
    with open(val_jsonl, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if os.path.exists(entry.get('clean_path', '')) and os.path.exists(entry.get('noisy_path', '')):
                samples.append(entry)
    samples = samples[:max_samples]
    print(f"Analyzing {len(samples)} validation samples\n")

    # Aggregate results
    all_region_errors = []
    all_clinical_failures = []

    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}...", end='\r')

        clean = load_image(sample['clean_path'])
        noisy = load_image(sample['noisy_path'])

        with torch.no_grad():
            backbone_out = backbone(noisy).clamp(0, 1)

        error_map = backbone_out - clean

        # Region analysis
        regions = identify_regions(clean)
        region_errors = analyze_errors_by_region(error_map, regions)
        all_region_errors.append(region_errors)

        # Clinical failure analysis
        clinical = analyze_clinical_failures(backbone_out, clean, noisy)
        all_clinical_failures.append(clinical)

    print("\n")

    # Aggregate region errors
    print("=" * 70)
    print("ERROR DISTRIBUTION BY REGION")
    print("=" * 70)
    print(f"{'Region':<15} {'Pixels%':>10} {'MAE':>10} {'RMSE':>10} {'Error%':>10} {'Density':>10}")
    print("-" * 70)

    region_names = ['edge', 'smooth', 'texture', 'bright', 'dark', 'boundary_zone']
    aggregated_regions = {}

    for name in region_names:
        values = [r[name] for r in all_region_errors if r[name]['pixels'] > 0]
        if values:
            aggregated_regions[name] = {
                'pixel_fraction': np.mean([v['pixel_fraction'] for v in values]),
                'mae': np.mean([v['mae'] for v in values]),
                'rmse': np.mean([v['rmse'] for v in values]),
                'error_contribution': np.mean([v['error_contribution'] for v in values]),
                'error_density': np.mean([v['error_density'] for v in values]),
            }
            v = aggregated_regions[name]
            print(f"{name:<15} {v['pixel_fraction']*100:>9.1f}% {v['mae']:>10.4f} {v['rmse']:>10.4f} "
                  f"{v['error_contribution']*100:>9.1f}% {v['error_density']:>10.2f}x")

    # Find highest error density regions
    print("\n" + "=" * 70)
    print("KEY FINDINGS: WHERE BACKBONE STRUGGLES")
    print("=" * 70)

    sorted_by_density = sorted(aggregated_regions.items(), key=lambda x: x[1]['error_density'], reverse=True)
    print("\nRegions with highest error density (errors per pixel relative to average):")
    for name, stats in sorted_by_density[:3]:
        print(f"  - {name}: {stats['error_density']:.2f}x average error density")

    # Aggregate clinical failures
    print("\n" + "=" * 70)
    print("CLINICAL METRIC FAILURES")
    print("=" * 70)

    # Edge preservation
    edge_loss_fracs = [c['edge_preservation']['edges_lost_fraction'] for c in all_clinical_failures]
    print(f"\nEdge Preservation:")
    print(f"  - Edges lost: {np.mean(edge_loss_fracs)*100:.1f}% of strong edges are weakened")
    print(f"  - Mean edge difference: {np.mean([c['edge_preservation']['mean_edge_difference'] for c in all_clinical_failures]):.4f}")

    # Contrast
    contrast_loss_fracs = [c['contrast']['contrast_reduced_fraction'] for c in all_clinical_failures]
    contrast_ratios = [c['contrast']['mean_contrast_ratio'] for c in all_clinical_failures]
    print(f"\nContrast:")
    print(f"  - Regions with reduced contrast: {np.mean(contrast_loss_fracs)*100:.1f}%")
    print(f"  - Mean contrast ratio: {np.mean(contrast_ratios):.2f} (1.0 = perfect, <1 = contrast lost)")

    # Boundary sharpness
    boundary_ratios = [c['boundary_sharpness']['boundary_preservation_ratio'] for c in all_clinical_failures]
    print(f"\nLayer Boundaries:")
    print(f"  - Boundary sharpness preserved: {np.mean(boundary_ratios)*100:.1f}%")
    blurred_count = sum(1 for c in all_clinical_failures if c['boundary_sharpness']['boundaries_blurred'])
    print(f"  - Samples with blurred boundaries: {blurred_count}/{len(all_clinical_failures)}")

    # Texture
    texture_ratios = [c['texture']['texture_preservation_ratio'] for c in all_clinical_failures]
    over_smoothed = sum(1 for c in all_clinical_failures if c['texture']['over_smoothed'])
    print(f"\nTexture Preservation:")
    print(f"  - Texture variance preserved: {np.mean(texture_ratios)*100:.1f}%")
    print(f"  - Over-smoothed samples: {over_smoothed}/{len(all_clinical_failures)}")

    # Residual analysis
    edge_corrs = [c['residual_analysis']['edge_correlated_error'] for c in all_clinical_failures]
    print(f"\nError Pattern:")
    print(f"  - Edge-correlated error: {np.mean(edge_corrs):.2f}x (>1 = errors at edges)")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR SYMBOLIC CORRECTORS")
    print("=" * 70)

    recommendations = []

    if np.mean(edge_loss_fracs) > 0.1:
        recommendations.append(("EDGE CORRECTOR", "HIGH",
                               f"{np.mean(edge_loss_fracs)*100:.0f}% edges weakened - add edge enhancement"))

    if np.mean(contrast_ratios) < 0.9:
        recommendations.append(("CONTRAST CORRECTOR", "HIGH",
                               f"Contrast at {np.mean(contrast_ratios)*100:.0f}% - boost local contrast"))

    if np.mean(boundary_ratios) < 0.95:
        recommendations.append(("BOUNDARY CORRECTOR", "MEDIUM",
                               f"Boundaries at {np.mean(boundary_ratios)*100:.0f}% - sharpen layer transitions"))

    if np.mean(texture_ratios) < 0.8:
        recommendations.append(("TEXTURE CORRECTOR", "MEDIUM",
                               f"Texture at {np.mean(texture_ratios)*100:.0f}% - reduce over-smoothing"))

    if aggregated_regions.get('boundary_zone', {}).get('error_density', 0) > 1.5:
        recommendations.append(("BOUNDARY-ZONE FOCUS", "HIGH",
                               f"Boundary zones have {aggregated_regions['boundary_zone']['error_density']:.1f}x error density"))

    if not recommendations:
        print("\nBackbone performs well across all clinical metrics.")
        print("Consider: Focus on interpretability and controllability rather than correction.")
    else:
        print(f"\nFound {len(recommendations)} areas for improvement:\n")
        for name, priority, reason in sorted(recommendations, key=lambda x: 0 if x[1]=="HIGH" else 1):
            print(f"  [{priority}] {name}")
            print(f"        {reason}\n")

    return aggregated_regions, all_clinical_failures, recommendations


if __name__ == "__main__":
    run_analysis(
        val_jsonl="pku37_val.jsonl",
        backbone_path="outputs/nafnet_pku37_w40/best_model.pth",
        max_samples=10
    )
