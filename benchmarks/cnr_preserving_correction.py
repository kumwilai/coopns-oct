#!/usr/bin/env python3
"""
CNR-Preserving Adaptive Correction Module

Innovation: Region-aware correction that treats tissue and background differently
to preserve Contrast-to-Noise Ratio (CNR = (signal - background) / std_background)

Key insight: CNR degrades when:
1. Background noise increases (corrections add artifacts to dark regions)
2. Signal-background contrast decreases

Solution:
- Detect tissue vs background regions adaptively
- Apply smoothing-only corrections to background (reduce noise)
- Apply contrast-enhancing corrections to tissue (boost signal)
- Gate corrections based on expected CNR impact

Performance optimizations applied:
- Cached smoothing kernel normalization
- Fused mask computations to reduce intermediate tensors
- Vectorized regional stats with single pass computation
- Reduced redundant clamp operations
- Minimized finite checks using torch.nan_to_num
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class AdaptiveRegionDetector(nn.Module):
    """
    Learns to detect tissue vs background regions adaptively.

    Unlike simple thresholding, this uses learned features to identify:
    - Tissue: Bright regions with structure (retinal layers, vessels)
    - Background: Dark regions that should remain smooth (vitreous, below RPE)
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Lightweight feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Tissue probability predictor
        self.tissue_head = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        # Learnable threshold for intensity-based prior
        self.intensity_weight = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tissue_mask: [B, 1, H, W] probability of being tissue (0=background, 1=tissue)
            background_mask: [B, 1, H, W] probability of being background

        Optimizations:
        - Fused min/max computation using aminmax
        - Single clamp operation at the end instead of multiple
        - Eliminated redundant intermediate tensor (intensity_prior alias)
        """
        # Learned features
        feat = self.features(x)
        learned_tissue = self.tissue_head(feat)

        # Intensity-based prior (bright = tissue, dark = background)
        # GRADIENT FIX: aminmax doesn't support backprop, use separate min/max ops
        # Note: This is slightly slower but allows gradients to flow through
        x_min = x.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        # OPTIMIZATION: Fused normalization - avoid intermediate x_norm variable
        x_range = x_max - x_min
        x_range.clamp_(min=1e-8)  # In-place clamp to avoid new tensor
        intensity_prior = (x - x_min) / x_range

        # Combine learned and intensity-based
        weight = torch.sigmoid(self.intensity_weight)
        # OPTIMIZATION: Compute tissue_mask directly with clamped result
        # lerp is faster than manual interpolation: lerp(a, b, w) = a + w * (b - a)
        tissue_mask = torch.lerp(intensity_prior, learned_tissue, weight).clamp_(0.0, 1.0)

        # OPTIMIZATION: Compute background_mask in-place style
        # Since tissue_mask is already clamped to [0,1], background_mask will be too
        background_mask = 1.0 - tissue_mask

        return tissue_mask, background_mask


class CNRGatedCorrection(nn.Module):
    """
    Gates corrections based on expected CNR impact.

    Innovation: Computes expected CNR change from correction and scales
    the correction to prevent CNR degradation.

    CNR = (mean_tissue - mean_background) / std_background

    A correction degrades CNR if:
    1. It increases std_background (adds noise to dark regions)
    2. It decreases (mean_tissue - mean_background)

    This module scales corrections to prevent these effects.

    CNR FIX (v2): Added direct background suppression mode.
    Key insight: Background regions should receive MINIMAL correction regardless
    of other factors. The previous approach scaled corrections by bg_penalty,
    but this still allowed some correction to leak through.

    New approach:
    - Tissue regions: Apply full correction scaled by tissue_bonus and cnr_improvement_factor
    - Background regions: Apply NEAR-ZERO correction (only smoothing allowed from BackgroundSmoothingCorrector)
    """

    def __init__(self):
        super().__init__()

        # Learnable sensitivity parameters
        # CNR FIX: Increased background_noise_sensitivity for more aggressive suppression
        self.background_noise_sensitivity = nn.Parameter(torch.tensor(5.0))
        # ENHANCED: Increased from 1.0 to 2.0 for better contrast sensitivity
        self.contrast_sensitivity = nn.Parameter(torch.tensor(2.0))

        # Minimum correction scale (don't completely zero out)
        # CNR FIX: Reduced to 0.02 for stronger suppression in background
        self.min_scale = 0.02

        # CNR FIX: Background suppression strength (learnable)
        # High value = more aggressive suppression in background
        self.background_suppression_strength = nn.Parameter(torch.tensor(0.9))

    def compute_regional_stats(self,
                                x: torch.Tensor,
                                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std for a soft-masked region.

        Optimizations:
        - Single pass computation reusing weighted values
        - In-place clamp operations where possible
        - Use torch.nan_to_num instead of expensive isfinite checks
        - Avoid creating intermediate tensors
        """
        # OPTIMIZATION: Precompute weighted x once, reuse for mean and variance
        weighted_x = x * mask
        mask_sum = mask.sum(dim=[2, 3], keepdim=True).clamp_(min=1.0)

        # Weighted mean
        mean = weighted_x.sum(dim=[2, 3], keepdim=True) / mask_sum

        # OPTIMIZATION: Compute variance in single expression
        # Var = E[X^2] - E[X]^2 (weighted version), but this can be negative due to precision
        # Use traditional (x - mean)^2 approach but optimize tensor operations
        diff = x - mean
        var = (diff * diff * mask).sum(dim=[2, 3], keepdim=True) / mask_sum

        # OPTIMIZATION: Fused clamp and sqrt - clamp variance first, then sqrt
        std = torch.sqrt(var.clamp_(min=1e-6))

        # OPTIMIZATION: Use nan_to_num instead of expensive isfinite().all() check
        # This handles NaN/Inf in a single vectorized operation
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6)

        return mean, std

    def forward(self,
                backbone_out: torch.Tensor,
                correction: torch.Tensor,
                tissue_mask: torch.Tensor,
                background_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Gate correction based on expected CNR impact.

        Optimizations:
        - Reuse regional stats computation (compute corrected stats in single pass)
        - Use in-place operations where safe
        - Replace expensive isfinite checks with nan_to_num
        - Fused clamp operations
        - Avoid redundant tensor creation
        """
        # Compute current CNR components
        tissue_mean, _ = self.compute_regional_stats(backbone_out, tissue_mask)
        bg_mean, bg_std = self.compute_regional_stats(backbone_out, background_mask)

        # OPTIMIZATION: bg_std already clamped in compute_regional_stats
        current_cnr = ((tissue_mean - bg_mean) / bg_std).clamp_(-100.0, 100.0)

        # Compute expected CNR components after correction
        # OPTIMIZATION: Add correction in-place to avoid creating new tensor
        corrected = backbone_out + correction
        tissue_mean_after, _ = self.compute_regional_stats(corrected, tissue_mask)
        bg_mean_after, bg_std_after = self.compute_regional_stats(corrected, background_mask)

        expected_cnr = ((tissue_mean_after - bg_mean_after) / bg_std_after).clamp_(-100.0, 100.0)
        cnr_change = expected_cnr - current_cnr

        # === Gating Logic ===
        # OPTIMIZATION: Compute bg_penalty with fused operations
        bg_noise_increase = (bg_std_after - bg_std).clamp_(min=0.0)
        # exp(-sensitivity * noise_increase), clamped to prevent overflow
        bg_penalty = torch.exp(
            (-self.background_noise_sensitivity * bg_noise_increase).clamp_(-20.0, 20.0)
        )

        # OPTIMIZATION: Compute tissue_bonus with fused operations
        contrast_change = (tissue_mean_after - bg_mean_after) - (tissue_mean - bg_mean)
        tissue_bonus = torch.sigmoid(
            (self.contrast_sensitivity * contrast_change).clamp_(-20.0, 20.0)
        )

        # OPTIMIZATION: Use nan_to_num instead of expensive isfinite checks
        bg_penalty = torch.nan_to_num(bg_penalty, nan=self.min_scale, posinf=1.0, neginf=self.min_scale)
        tissue_bonus = torch.nan_to_num(tissue_bonus, nan=0.5, posinf=1.0, neginf=0.0)

        # === ENHANCED: CNR Improvement Reward (NEW) ===
        # If CNR improves, boost the correction scale; if it degrades, suppress it
        # This directly targets the -1.3% CNR degradation issue
        cnr_improvement_factor = torch.where(
            cnr_change > 0,
            # CNR improved: boost scale by up to 1.5x
            1.0 + 0.5 * torch.tanh(cnr_change),
            # CNR degraded: suppress scale more aggressively
            torch.exp(2.0 * cnr_change).clamp(min=0.1)  # Stronger suppression for degradation
        )
        cnr_improvement_factor = torch.nan_to_num(cnr_improvement_factor, nan=1.0, posinf=1.5, neginf=0.1)

        # === CNR FIX (v2): Direct background suppression ===
        # Key insight: Background regions should get NEAR-ZERO correction to preserve CNR
        # Previous approach: background_mask * bg_penalty (still allows some correction)
        # New approach: Directly suppress background with strong exponential decay
        bg_suppression = torch.sigmoid(self.background_suppression_strength)  # 0.9 -> ~0.71

        # Tissue scale: Apply full correction with tissue_bonus and cnr_improvement_factor
        tissue_scale = tissue_mask * tissue_bonus * cnr_improvement_factor

        # Background scale: NEAR-ZERO correction (only allow tiny adjustments)
        # bg_penalty is applied but heavily attenuated by (1 - bg_suppression)
        # With bg_suppression=0.9, background gets at most ~0.3 of bg_penalty
        background_scale = background_mask * bg_penalty * (1.0 - bg_suppression)

        # Combined scale - tissue gets full correction, background gets heavily suppressed
        scale = (tissue_scale + background_scale).clamp_(self.min_scale, 2.0)
        scale = torch.nan_to_num(scale, nan=self.min_scale, posinf=self.min_scale, neginf=self.min_scale)

        # Apply gating with fused clamp
        gated_correction = (correction * scale).clamp_(-0.15, 0.15)
        gated_correction = torch.nan_to_num(gated_correction, nan=0.0, posinf=0.15, neginf=-0.15)

        # OPTIMIZATION: Compute info dict values efficiently
        # Use detach to avoid tracking gradients for logging
        info = {
            'cnr_before': current_cnr.mean().item(),
            'cnr_after_ungated': expected_cnr.mean().item(),
            'cnr_change': cnr_change.mean().item(),
            'avg_scale': scale.mean().item(),
            'bg_penalty': bg_penalty.mean().item(),
            'tissue_bonus': tissue_bonus.mean().item(),
            'cnr_improvement_factor': cnr_improvement_factor.mean().item(),
            'bg_suppression': bg_suppression.item(),  # CNR FIX: track background suppression
            'tissue_scale_mean': tissue_scale.mean().item(),  # CNR FIX: track tissue scale
            'background_scale_mean': background_scale.mean().item(),  # CNR FIX: track background scale
        }
        # OPTIMIZATION: Single pass nan check for info dict
        for k, v in info.items():
            if not math.isfinite(v):
                info[k] = 0.0

        return gated_correction, info


class BackgroundSmoothingCorrector(nn.Module):
    """
    Specialized corrector for background regions.

    Instead of applying the same correction everywhere, this applies
    a smoothing/denoising correction to background regions only.

    This REDUCES std_background, which IMPROVES CNR.

    CNR FIX (v2): Enhanced multi-scale smoothing for better noise reduction.
    Key insight: Single-scale Gaussian may not be enough to reduce background noise.
    New approach: Multi-scale smoothing with adaptive strength based on local noise estimate.

    Optimizations:
    - Cached normalized kernel to avoid recomputation every forward pass
    - Use register_buffer for fixed Gaussian kernel (faster than learnable for smoothing)
    - Reduced clamp operations by combining bounds
    """

    def __init__(self):
        super().__init__()

        # OPTIMIZATION: Use fixed Gaussian kernel as buffer (not learnable)
        # Small kernel for fine-scale smoothing
        gaussian_kernel_3x3 = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 16.0
        self.register_buffer('smooth_kernel_3x3', gaussian_kernel_3x3)

        # CNR FIX: Larger kernel for coarse-scale smoothing (better noise reduction)
        gaussian_kernel_5x5 = torch.tensor([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=torch.float32).view(1, 1, 5, 5) / 256.0
        self.register_buffer('smooth_kernel_5x5', gaussian_kernel_5x5)

        # CNR FIX: Increased smoothing strength for better background noise reduction
        # This directly improves CNR by reducing std_background (denominator)
        self.strength = nn.Parameter(torch.tensor(0.45))  # Increased from 0.25 to 0.45

        # CNR FIX: Multi-scale mixing ratio (balance between fine and coarse smoothing)
        self.coarse_weight = nn.Parameter(torch.tensor(0.6))  # Favor coarse smoothing for better noise reduction

        # OPTIMIZATION: Cache for normalized kernel on different devices
        self._cached_kernel: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None

    def forward(self, x: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale smoothing to background regions.

        CNR FIX (v2): Enhanced multi-scale smoothing approach:
        1. Fine-scale smoothing (3x3) for subtle noise
        2. Coarse-scale smoothing (5x5) for stronger noise reduction
        3. Adaptive mixing based on learned coarse_weight

        This directly improves CNR by reducing std_background (denominator).

        Optimizations:
        - Use pre-normalized kernel buffer (no runtime normalization)
        - Fused correction computation
        - Single clamp at the end
        - Use nan_to_num instead of isfinite check
        """
        # CNR FIX: Multi-scale smoothing for better background noise reduction
        # Fine-scale smoothing (3x3 kernel)
        smoothed_fine = F.conv2d(x, self.smooth_kernel_3x3, padding=1)

        # Coarse-scale smoothing (5x5 kernel) - better for noise reduction
        smoothed_coarse = F.conv2d(x, self.smooth_kernel_5x5, padding=2)

        # Adaptive mixing based on learned coarse_weight
        coarse_w = torch.sigmoid(self.coarse_weight)
        smoothed = coarse_w * smoothed_coarse + (1.0 - coarse_w) * smoothed_fine

        # Compute correction with strength
        strength = torch.sigmoid(self.strength)
        correction = (smoothed - x) * (background_mask * strength)

        # OPTIMIZATION: Single clamp at the end instead of multiple clamps
        correction = correction.clamp_(-0.15, 0.15)

        # OPTIMIZATION: Use nan_to_num instead of expensive isfinite().all() check
        correction = torch.nan_to_num(correction, nan=0.0, posinf=0.15, neginf=-0.15)

        return correction


class CNRPreservingCorrectionModule(nn.Module):
    """
    Main module that integrates all CNR-preserving innovations.

    This module wraps around existing corrections and ensures they don't degrade CNR:

    1. Detects tissue vs background regions
    2. Applies background smoothing (reduces noise, improves CNR denominator)
    3. Gates clinical corrections based on expected CNR impact
    4. Combines everything with region-aware weighting

    Usage:
        cnr_module = CNRPreservingCorrectionModule()
        final_correction, info = cnr_module(backbone_out, clinical_correction)
        output = backbone_out + final_correction
    """

    def __init__(self):
        super().__init__()

        self.region_detector = AdaptiveRegionDetector()
        self.cnr_gate = CNRGatedCorrection()
        self.bg_smoother = BackgroundSmoothingCorrector()

        # Balance between clinical correction and CNR preservation
        # CNR FIX (v2): Rebalanced weights to fix -7.5% CNR degradation
        # Key insight:
        # - Clinical corrections in tissue regions are GOOD for CNR (increase numerator)
        # - Clinical corrections in background are BAD for CNR (increase denominator noise)
        # - Background smoothing is GOOD for CNR (decrease denominator noise)
        #
        # Strategy: Allow strong clinical correction (gated to tissue only by cnr_gate)
        # and apply strong background smoothing to reduce noise
        self.clinical_weight = nn.Parameter(torch.tensor(0.5))  # Increased back to 0.5 for clinical
        # CNR FIX: Increased smoothing_weight for stronger background noise reduction
        self.smoothing_weight = nn.Parameter(torch.tensor(0.6))  # Increased from 0.4 to 0.6

    def forward(self,
                backbone_out: torch.Tensor,
                clinical_correction: torch.Tensor,
                cached_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Apply CNR-preserving correction.

        Optimizations:
        - Removed redundant device transfers (submodules handle this)
        - Removed try-except blocks (use nan_to_num for robustness instead)
        - Fused weight computation and clamping
        - Single final clamp instead of multiple
        - Efficient info dict computation

        Args:
            backbone_out: Denoised image from backbone [B, 1, H, W]
            clinical_correction: Clinical correction to gate [B, 1, H, W]
            cached_masks: Optional pre-computed (tissue_mask, background_mask) to avoid
                          redundant region_detector calls. If provided, these are used
                          directly instead of calling region_detector.
        """
        gate_info: Dict = {}

        # Step 1: Detect regions - use cached masks if available
        if cached_masks is not None:
            tissue_mask, background_mask = cached_masks
            target_shape = backbone_out.shape[2:]
            # Interpolate if shapes don't match
            if tissue_mask.shape[2:] != target_shape:
                tissue_mask = F.interpolate(tissue_mask, size=target_shape, mode='bilinear', align_corners=False)
                background_mask = F.interpolate(background_mask, size=target_shape, mode='bilinear', align_corners=False)
        else:
            # Fallback: detect regions - masks are already clamped by region_detector
            tissue_mask, background_mask = self.region_detector(backbone_out)

        # Step 2: Apply background smoothing
        bg_smoothing = self.bg_smoother(backbone_out, background_mask)

        # Step 3: Gate clinical correction
        gated_clinical, gate_info = self.cnr_gate(
            backbone_out, clinical_correction, tissue_mask, background_mask
        )

        # Step 4: Combine corrections with learned weights
        # OPTIMIZATION: Sigmoid output is already in [0, 1], no need to clamp to [0, 2]
        clinical_w = torch.sigmoid(self.clinical_weight)
        smoothing_w = torch.sigmoid(self.smoothing_weight)

        # OPTIMIZATION: Fused final correction computation with single clamp
        final_correction = (clinical_w * gated_clinical + smoothing_w * bg_smoothing).clamp_(-0.15, 0.15)

        # OPTIMIZATION: Use nan_to_num instead of expensive isfinite check
        final_correction = torch.nan_to_num(final_correction, nan=0.0, posinf=0.15, neginf=-0.15)

        # OPTIMIZATION: Compute info dict efficiently without safe_item wrapper
        info = {
            'tissue_mask_mean': tissue_mask.mean().item(),
            'background_mask_mean': background_mask.mean().item(),
            'bg_smoothing_magnitude': bg_smoothing.abs().mean().item(),
            'gated_clinical_magnitude': gated_clinical.abs().mean().item(),
            'final_correction_magnitude': final_correction.abs().mean().item(),
            'clinical_weight': clinical_w.item(),
            'smoothing_weight': smoothing_w.item(),
            **gate_info
        }
        # OPTIMIZATION: Single pass nan check for info dict
        for k, v in info.items():
            if isinstance(v, float) and not math.isfinite(v):
                info[k] = 0.0

        return final_correction, info

    def compute_cnr(self,
                    x: torch.Tensor,
                    tissue_mask: Optional[torch.Tensor] = None,
                    background_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute CNR for an image using detected or provided masks.

        Optimizations:
        - bg_std is already clamped in compute_regional_stats
        - Single fused division and clamp operation
        """
        if tissue_mask is None or background_mask is None:
            tissue_mask, background_mask = self.region_detector(x)

        tissue_mean, _ = self.cnr_gate.compute_regional_stats(x, tissue_mask)
        bg_mean, bg_std = self.cnr_gate.compute_regional_stats(x, background_mask)

        # OPTIMIZATION: bg_std already clamped in compute_regional_stats, fused clamp
        return ((tissue_mean - bg_mean) / bg_std).clamp_(-100.0, 100.0)
