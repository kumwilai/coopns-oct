#!/usr/bin/env python3
"""
Clinical Enhancement Module for OCT Denoising

Innovation: Multi-scale adaptive enhancement that pushes clinical metrics
(contrast, edge, boundary, texture) toward 10-15% improvement target.

Key techniques:
1. Edge-aware local contrast enhancement (learnable CLAHE-like)
2. Multi-scale boundary sharpening
3. Texture-preserving enhancement
4. Adaptive strength based on local statistics

This module provides ADDITIONAL corrections on top of existing correctors,
specifically targeting clinical metrics improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class LearnableLocalContrastEnhancer(nn.Module):
    """
    Learnable local contrast enhancement inspired by CLAHE.

    Unlike fixed CLAHE, this learns optimal local contrast enhancement
    parameters from data, adapting to OCT image characteristics.

    Key innovation: Learnable clip limit and tile size equivalent via
    adaptive local histogram stretching.

    OPTIMIZED: Uses unfold-based local statistics for better efficiency.
    """

    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        # Learnable enhancement strength (varies by local statistics)
        self.strength_net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),  # Input: local mean, local std
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        # Learnable contrast curve parameters - MODERATE for balanced improvement
        self.gamma = nn.Parameter(torch.tensor(1.5))  # Reduced from 2.0 for moderate contrast enhancement
        self.clip_factor = nn.Parameter(torch.tensor(0.7))  # Reduced from 0.85 to prevent overcorrection

        # Pre-compute uniform kernel for box filter (more efficient than avg_pool2d)
        self.register_buffer('_box_kernel', torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable local contrast enhancement.

        Returns additive correction for contrast improvement.
        """
        B, C, H, W = x.shape

        # OPTIMIZATION: Compute x_squared once and reuse
        x_sq = x * x  # Faster than x**2

        # OPTIMIZATION: Pad once and reuse for both mean and sq_mean
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x_sq_padded = F.pad(x_sq, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        # OPTIMIZATION: Cache dtype/device-converted kernel to avoid repeated .to() calls
        if not hasattr(self, '_box_kernel_cached') or self._box_kernel_cached.dtype != x.dtype or self._box_kernel_cached.device != x.device:
            self._box_kernel_cached = self._box_kernel.to(dtype=x.dtype, device=x.device)
        box_kernel = self._box_kernel_cached
        local_mean = F.conv2d(x_padded, box_kernel)
        local_sq_mean = F.conv2d(x_sq_padded, box_kernel)

        # FIX: Clamp local_var to prevent NaN from sqrt of negative values
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=0.0)
        # FIX: Add epsilon inside sqrt for additional numerical stability
        local_std = torch.sqrt(local_var + 1e-8)

        # FIX: Ensure sizes match after pooling (defensive check)
        if local_mean.shape[-2:] != (H, W):
            local_mean = F.interpolate(local_mean, size=(H, W), mode='bilinear', align_corners=False)
            local_std = F.interpolate(local_std, size=(H, W), mode='bilinear', align_corners=False)

        # Compute adaptive enhancement strength
        stats = torch.cat([local_mean, local_std], dim=1)
        strength = self.strength_net(stats)

        # Local contrast enhancement: stretch toward local mean
        # OPTIMIZATION: Cache gamma computation
        gamma = F.softplus(self.gamma)
        deviation = x - local_mean

        # Enhance deviation (increase local contrast)
        # OPTIMIZATION: Fuse operations
        enhanced_deviation = deviation * (1.0 + gamma * strength)

        # Clip to prevent artifacts
        clip = torch.sigmoid(self.clip_factor)
        max_enhance = (local_std + 1e-6) * (3.0 * clip)  # Max deviation based on local std
        enhanced_deviation = torch.clamp(enhanced_deviation, -max_enhance, max_enhance)

        # Return as additive correction
        correction = enhanced_deviation - deviation

        # FIX: Reduced clamp from ±0.5 to ±0.15 to prevent over-correction
        correction = torch.clamp(correction, -0.15, 0.15)

        return correction


class MultiScaleBoundarySharpener(nn.Module):
    """
    Multi-scale boundary sharpening for OCT images.

    OCT images have boundaries at multiple scales:
    - Fine: Individual layer boundaries
    - Medium: Major retinal layers
    - Coarse: Retina-vitreous interface

    This module detects and sharpens boundaries at each scale.

    OPTIMIZED: Uses single multi-scale convolution with grouped processing.
    """

    def __init__(self):
        super().__init__()

        # Edge detection at multiple scales
        self.scales = [1, 2, 4]  # Fine, medium, coarse
        self.num_scales = len(self.scales)

        # OPTIMIZATION: Single shared feature extractor for all scales
        # Instead of 3 separate Conv->BN->ReLU->Conv pipelines
        self.shared_conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.shared_bn = nn.BatchNorm2d(8)
        self.shared_relu = nn.LeakyReLU(0.2, inplace=True)

        # Per-scale output convolutions (lightweight 1x1 projections)
        self.scale_projections = nn.ModuleList([
            nn.Conv2d(8, 1, 1) for _ in self.scales
        ])

        # Scale combination weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

        # Overall strength - MODERATE for balanced edge improvement
        self.strength = nn.Parameter(torch.tensor(0.5))  # Reduced from 0.8 to prevent overcorrection

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.shared_conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.ones_(self.shared_bn.weight)
        nn.init.zeros_(self.shared_bn.bias)
        for proj in self.scale_projections:
            nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale boundary sharpening.

        Returns additive correction for boundary enhancement.
        """
        B, C, H, W = x.shape

        # OPTIMIZATION: Pre-compute weights once
        weights = F.softmax(self.scale_weights, dim=0)
        strength = torch.sigmoid(self.strength)

        # OPTIMIZATION: Accumulate directly instead of list append + sum
        total_correction = torch.zeros_like(x)

        for i, scale in enumerate(self.scales):
            # Downsample
            if scale > 1:
                x_scaled = F.avg_pool2d(x, scale)
            else:
                x_scaled = x

            # OPTIMIZATION: Use shared layers
            features = self.shared_conv1(x_scaled)
            features = self.shared_bn(features)
            features = self.shared_relu(features)
            edge_correction = self.scale_projections[i](features)

            # Upsample back to original size
            if scale > 1:
                edge_correction = F.interpolate(edge_correction, size=(H, W),
                                                 mode='bilinear', align_corners=False)

            # OPTIMIZATION: Accumulate in-place with weight multiplication
            total_correction = total_correction + edge_correction * weights[i]

        # Apply overall strength and clamp - increased limit for clinical improvement
        result = torch.clamp(total_correction * strength, -0.15, 0.15)

        return result


class TexturePreservingEnhancer(nn.Module):
    """
    Enhances image while preserving texture statistics.

    Problem: Standard enhancement can destroy fine texture (e.g., speckle patterns
    that carry diagnostic information, subtle layer textures).

    Solution: Separate texture from structure, enhance structure only,
    then recombine with STRONGLY preserved texture.

    OPTIMIZED: Uses efficient box filter and reduced intermediate allocations.

    FIX: Previous version was DEGRADING texture by 5%. Now:
    - Increased texture_boost_factor from 0.1 to 0.3 for better texture preservation
    - Added texture_preservation_strength to explicitly preserve original texture
    - Structure correction is now masked to avoid texture regions
    """

    def __init__(self):
        super().__init__()

        # Texture extraction (high-frequency)
        self.texture_kernel_size = 5
        self.pad = self.texture_kernel_size // 2

        # Structure enhancement - now more conservative
        self.structure_enhancer = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )

        # Texture preservation weight - INCREASED from 0.5 to 0.7 to prevent degradation
        self.texture_weight = nn.Parameter(torch.tensor(0.7))

        # NEW: Explicit texture preservation strength - ensures texture is preserved
        self.texture_preservation_strength = nn.Parameter(torch.tensor(1.5))  # INCREASED for strong texture preservation

        # OPTIMIZATION: Pre-compute box kernel for structure extraction
        ks = self.texture_kernel_size
        self.register_buffer('_box_kernel', torch.ones(1, 1, ks, ks) / (ks * ks))

        self._init_weights()

    def _init_weights(self):
        for m in self.structure_enhancer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply texture-preserving enhancement.

        Returns additive correction that enhances structure while PRESERVING texture.

        FIX: Previous version caused 5% texture DEGRADATION. Now:
        - Structure correction is attenuated in high-texture regions
        - Texture is explicitly preserved with higher weight
        - Overall result is texture-neutral or texture-positive
        """
        B, C, H, W = x.shape

        # OPTIMIZATION: Use conv2d with box kernel instead of pad + avg_pool2d
        # FIX: Cache dtype/device-converted kernel
        if not hasattr(self, '_box_kernel_cached') or self._box_kernel_cached.dtype != x.dtype or self._box_kernel_cached.device != x.device:
            self._box_kernel_cached = self._box_kernel.to(dtype=x.dtype, device=x.device)
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        structure = F.conv2d(x_padded, self._box_kernel_cached)

        # FIX: Ensure structure matches input size (defensive check)
        if structure.shape[-2:] != (H, W):
            structure = F.interpolate(structure, size=(H, W), mode='bilinear', align_corners=False)

        # Compute texture (high-frequency component)
        texture = x - structure
        texture_weight = torch.sigmoid(self.texture_weight)
        texture_pres = torch.sigmoid(self.texture_preservation_strength)

        # Enhance structure only
        structure_correction = self.structure_enhancer(structure)

        # FIX: Compute texture magnitude to identify high-texture regions
        # Structure correction should be ATTENUATED in high-texture regions
        texture_magnitude = texture.abs()
        texture_mask = torch.sigmoid(texture_magnitude * 10.0 - 0.5)  # High where texture is strong

        # FIX: Attenuate structure correction in texture regions to prevent degradation
        # This is the key fix - structure changes shouldn't destroy texture
        structure_correction_masked = structure_correction * (1.0 - texture_mask * 0.5)

        # FIX: Increased texture_boost_factor to 0.6 for strong texture preservation
        # This compensates for texture loss from contrast/boundary enhancement
        texture_boost_factor = 0.6 * texture_weight * texture_pres
        texture_boost = texture * texture_boost_factor

        # Combine: masked structure correction + texture boost
        # The texture_boost now ADDS back texture that might be lost
        result = structure_correction_masked + texture_boost

        # FIX: Reduced clamp from ±0.5 to ±0.15 to prevent over-correction
        result = torch.clamp(result, -0.15, 0.15)

        return result


class ClinicalEnhancementModule(nn.Module):
    """
    Main clinical enhancement module combining all enhancement techniques.

    This module provides ADDITIONAL clinical improvement on top of
    existing neuro-symbolic corrections.

    Target: Push clinical metrics from ~5% to 10-15% improvement.

    Components:
    1. Local contrast enhancement (improves contrast ratio)
    2. Multi-scale boundary sharpening (improves edge/boundary metrics)
    3. Texture-preserving enhancement (improves texture ratio)
    """

    def __init__(self):
        super().__init__()

        self.contrast_enhancer = LearnableLocalContrastEnhancer()
        self.boundary_sharpener = MultiScaleBoundarySharpener()
        self.texture_enhancer = TexturePreservingEnhancer()

        # Restore settings from clinical_final3 (achieved +33.8% clinical, +1.3% CNR)
        self.contrast_weight = nn.Parameter(torch.tensor(0.6))
        self.boundary_weight = nn.Parameter(torch.tensor(0.55))
        self.texture_weight = nn.Parameter(torch.tensor(0.7))

        # Overall clinical enhancement strength
        # BALANCED: 0.42 is middle ground between aggressive (0.5) and conservative (0.35)
        # Target: ~15-20% clinical improvement with ~3-4 dB PSNR drop
        self.overall_strength = nn.Parameter(torch.tensor(0.42))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply clinical enhancement.

        Args:
            x: [B, 1, H, W] input image (backbone output)

        Returns:
            correction: [B, 1, H, W] clinical enhancement correction
            info: Dict with component statistics
        """
        # OPTIMIZATION: Pre-compute all sigmoid values once (they're scalars)
        cw = torch.sigmoid(self.contrast_weight)
        bw = torch.sigmoid(self.boundary_weight)
        tw = torch.sigmoid(self.texture_weight)
        strength = torch.sigmoid(self.overall_strength)

        # Get individual corrections
        contrast_corr = self.contrast_enhancer(x)
        boundary_corr = self.boundary_sharpener(x)
        texture_corr = self.texture_enhancer(x)

        # OPTIMIZATION: Fuse combine and strength multiplication
        # combined = cw * contrast_corr + bw * boundary_corr + tw * texture_corr
        # correction = combined * strength
        # Equivalent to: correction = strength * (cw * contrast_corr + bw * boundary_corr + tw * texture_corr)
        correction = (cw * contrast_corr + bw * boundary_corr + tw * texture_corr) * strength

        # FIX: Reduced clamp from ±0.42 to ±0.15 to prevent over-correction
        correction = torch.clamp(correction, -0.15, 0.15)

        # OPTIMIZATION: Use torch.no_grad() for info computation (not needed for backward pass)
        with torch.no_grad():
            info = {
                'contrast_correction_mag': contrast_corr.abs().mean().item(),
                'boundary_correction_mag': boundary_corr.abs().mean().item(),
                'texture_correction_mag': texture_corr.abs().mean().item(),
                'total_clinical_correction_mag': correction.abs().mean().item(),
                'contrast_weight': cw.item(),
                'boundary_weight': bw.item(),
                'texture_weight': tw.item(),
                'overall_strength': strength.item(),
            }

        return correction, info


class ClinicalMetricsLoss(nn.Module):
    """
    Loss function specifically designed to improve clinical metrics.

    Directly optimizes for:
    1. Local contrast ratio (backbone vs corrected)
    2. Edge strength preservation/improvement
    3. Boundary sharpness
    4. Texture variance preservation

    OPTIMIZED: Uses combined Sobel kernel and efficient local contrast computation.
    """

    def __init__(self):
        super().__init__()

        # OPTIMIZATION: Combined Sobel filters for single conv2d call
        # Stack sobel_x and sobel_y into a single kernel with 2 output channels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Shape: [2, 1, 3, 3] - 2 output channels (gx, gy), 1 input channel
        sobel_combined = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)
        self.register_buffer('sobel_combined', sobel_combined)

        # Laplacian for texture
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

        # OPTIMIZATION: Pre-compute box kernel for local contrast
        kernel_size = 7
        self._contrast_kernel_size = kernel_size
        self._contrast_pad = kernel_size // 2
        box_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        self.register_buffer('_box_kernel', box_kernel)

    def compute_edge_strength(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge strength using Sobel filters."""
        # FIX: Cache dtype/device-converted kernel
        if not hasattr(self, '_sobel_combined_cached') or self._sobel_combined_cached.dtype != x.dtype or self._sobel_combined_cached.device != x.device:
            self._sobel_combined_cached = self.sobel_combined.to(dtype=x.dtype, device=x.device)
        # OPTIMIZATION: Single conv2d call for both gradients
        grads = F.conv2d(x, self._sobel_combined_cached, padding=1)
        gx = grads[:, 0:1, :, :]
        gy = grads[:, 1:2, :, :]
        # OPTIMIZATION: Fuse operations - gx*gx is faster than gx**2
        edge_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return edge_mag

    def compute_local_contrast(self, x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """Compute local contrast (local std)."""
        pad = self._contrast_pad
        # OPTIMIZATION: Compute x_sq once
        x_sq = x * x

        # OPTIMIZATION: Pad once, use conv2d with box kernel
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x_sq_padded = F.pad(x_sq, (pad, pad, pad, pad), mode='reflect')

        # FIX: Cache dtype/device-converted kernel
        if not hasattr(self, '_contrast_box_kernel_cached') or self._contrast_box_kernel_cached.dtype != x.dtype or self._contrast_box_kernel_cached.device != x.device:
            self._contrast_box_kernel_cached = self._box_kernel.to(dtype=x.dtype, device=x.device)
        local_mean = F.conv2d(x_padded, self._contrast_box_kernel_cached)
        local_sq_mean = F.conv2d(x_sq_padded, self._contrast_box_kernel_cached)

        # Variance = E[X^2] - E[X]^2, clamp to >= 0
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=0.0)
        result = torch.sqrt(local_var + 1e-8)
        return result

    def compute_texture(self, x: torch.Tensor) -> torch.Tensor:
        """Compute texture using Laplacian."""
        # FIX: Cache dtype/device-converted kernel
        if not hasattr(self, '_laplacian_cached') or self._laplacian_cached.dtype != x.dtype or self._laplacian_cached.device != x.device:
            self._laplacian_cached = self.laplacian.to(dtype=x.dtype, device=x.device)
        result = F.conv2d(x, self._laplacian_cached, padding=1).abs()
        return result

    def forward(self,
                corrected: torch.Tensor,
                backbone: torch.Tensor,
                clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute clinical metrics improvement loss.

        This loss REWARDS improvement and PENALIZES degradation of clinical metrics.
        """
        eps = 1e-8

        # OPTIMIZATION: Compute all metrics in batch-friendly manner
        # Stack inputs for batch processing where possible
        edge_b = self.compute_edge_strength(backbone)
        edge_c = self.compute_edge_strength(corrected)
        edge_clean = self.compute_edge_strength(clean)

        contrast_b = self.compute_local_contrast(backbone)
        contrast_c = self.compute_local_contrast(corrected)
        contrast_clean = self.compute_local_contrast(clean)

        texture_b = self.compute_texture(backbone)
        texture_c = self.compute_texture(corrected)
        texture_clean = self.compute_texture(clean)

        # OPTIMIZATION: Compute all means at once
        edge_clean_mean = edge_clean.mean().clamp(min=eps)
        contrast_clean_mean = contrast_clean.mean().clamp(min=eps)
        texture_clean_mean = texture_clean.mean().clamp(min=eps)

        edge_b_mean = edge_b.mean()
        edge_c_mean = edge_c.mean()
        contrast_b_mean = contrast_b.mean()
        contrast_c_mean = contrast_c.mean()
        texture_b_mean = texture_b.mean()
        texture_c_mean = texture_c.mean()

        # FIX BOTTLENECK #1: Use RELATIVE IMPROVEMENT instead of preservation ratio
        # Old: contrast_improvement = (corrected/clean) - (backbone/clean)  ← always tiny!
        # New: contrast_improvement = (corrected - backbone) / backbone  ← actual improvement %
        # This directly measures how much better corrected is compared to backbone

        eps = 1e-6

        # Relative improvement: how much corrected improves over backbone (as percentage)
        # Positive = corrected is better, Negative = corrected is worse
        contrast_improvement = (contrast_c_mean - contrast_b_mean) / (contrast_b_mean + eps)
        edge_improvement = (edge_c_mean - edge_b_mean) / (edge_b_mean + eps)
        texture_change = (texture_c_mean - texture_b_mean) / (texture_b_mean + eps)

        # Also compute preservation ratios for logging
        edge_ratio_b = edge_b_mean / edge_clean_mean
        edge_ratio_c = edge_c_mean / edge_clean_mean
        contrast_ratio_b = contrast_b_mean / contrast_clean_mean
        contrast_ratio_c = contrast_c_mean / contrast_clean_mean
        texture_ratio_b = texture_b_mean / texture_clean_mean
        texture_ratio_c = texture_c_mean / texture_clean_mean

        # === Loss Components ===
        # TARGET: 10-15% clinical improvement
        # Penalize if contrast improvement < 12% (target 15%)
        # Penalize if edge improvement < 10% (target 12%)
        # Use stronger multipliers to push toward targets
        contrast_loss = torch.clamp(F.relu(0.12 - contrast_improvement) * 15.0, max=3.0)
        edge_loss = torch.clamp(F.relu(0.10 - edge_improvement) * 15.0, max=3.0)
        # Texture: only penalize if it gets WORSE
        texture_loss = torch.clamp(F.relu(-texture_change) * 5.0, max=1.5)

        # Combined loss - higher cap to allow stronger gradients
        total_loss = torch.clamp(contrast_loss + edge_loss + texture_loss, max=7.0)

        # OPTIMIZATION: Use torch.no_grad() for metrics dict (not needed for backward pass)
        with torch.no_grad():
            metrics = {
                'edge_ratio_backbone': edge_ratio_b.item(),
                'edge_ratio_corrected': edge_ratio_c.item(),
                'edge_improvement': edge_improvement.item(),
                'contrast_ratio_backbone': contrast_ratio_b.item(),
                'contrast_ratio_corrected': contrast_ratio_c.item(),
                'contrast_improvement': contrast_improvement.item(),
                'texture_ratio_backbone': texture_ratio_b.item(),
                'texture_ratio_corrected': texture_ratio_c.item(),
                'texture_change': texture_change.item(),
            }

        return total_loss, metrics
