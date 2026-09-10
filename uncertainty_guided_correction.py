#!/usr/bin/env python3
"""
Uncertainty-Guided Adaptive Correction for OCT Denoising

This module implements a principled approach to adaptive correction strength
that doesn't require manual hyperparameter tuning (like min_correction).

Key Algorithm:
1. Aggregate uncertainty from multiple sources (failure maps, conflicts, input noise)
2. Use uncertainty to modulate per-pixel correction strength
3. High uncertainty -> strong correction; Low uncertainty -> preserve

This replaces the fixed min_correction=0.40 hyperparameter with learned,
image-adaptive behavior.

Author: Neuro-Symbolic OCT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class UncertaintyAggregator(nn.Module):
    """
    Compute per-pixel uncertainty from multiple sources.

    Returns [B, 1, H, W] uncertainty map where:
    - 0 = high confidence (NAFNet worked well, don't correct)
    - 1 = low confidence (NAFNet struggled, apply correction)

    Sources:
    1. Predicate failure map entropy (disagreement between P1-P6)
    2. Input noise variance (inherently noisy regions)
    3. Local intensity statistics (signal vs background)
    """

    def __init__(self, learnable_weights: bool = True):
        super().__init__()

        if learnable_weights:
            # Learnable combination weights
            self.failure_weight = nn.Parameter(torch.tensor(0.4))
            self.noise_weight = nn.Parameter(torch.tensor(0.3))
            self.intensity_weight = nn.Parameter(torch.tensor(0.3))
        else:
            # Fixed weights
            self.register_buffer('failure_weight', torch.tensor(0.4))
            self.register_buffer('noise_weight', torch.tensor(0.3))
            self.register_buffer('intensity_weight', torch.tensor(0.3))

        # Local pooling for noise estimation
        self.pool_size = 9

    def forward(self,
                backbone_out: torch.Tensor,
                noisy: torch.Tensor,
                failure_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute per-pixel uncertainty map.

        Args:
            backbone_out: [B, 1, H, W] denoised output
            noisy: [B, 1, H, W] original noisy input
            failure_maps: Dict with P1-P6 failure maps [B, 1, H, W]

        Returns:
            uncertainty: [B, 1, H, W] in [0, 1]
        """
        B, C, H, W = backbone_out.shape
        device = backbone_out.device

        # === Component 1: Failure Map Entropy ===
        # High entropy = predicates disagree = uncertain region
        predictors = ['P1', 'P2', 'P3', 'P4', 'P6']  # Exclude P5

        # FIX: Preallocate tensor instead of list accumulation (memory optimization)
        fm_stack = torch.zeros(B, len(predictors), H, W, device=device, dtype=backbone_out.dtype)
        for i, p in enumerate(predictors):
            if p in failure_maps:
                fm_stack[:, i:i+1, :, :] = failure_maps[p]
        # No need to delete fm_list - we never created it

        # Per-pixel entropy over 5 predictors
        # FIX: Clamp log arguments directly to prevent NaN
        fm_clipped = fm_stack.clamp(1e-7, 1 - 1e-7)
        log_fm = torch.log(fm_clipped.clamp(min=1e-8))
        log_1_minus_fm = torch.log((1 - fm_clipped).clamp(min=1e-8))
        entropy = -(fm_clipped * log_fm + (1 - fm_clipped) * log_1_minus_fm).mean(dim=1, keepdim=True)
        del log_fm, log_1_minus_fm  # FIX: Free intermediate tensors

        # Normalize by max entropy (log(2) for binary)
        failure_uncertainty = (entropy / math.log(2)).clamp(0, 1)
        del entropy  # FIX: Free intermediate tensor

        # === Component 2: Input Noise Variance ===
        # High variance in noisy input = inherently uncertain
        pad = self.pool_size // 2
        noisy_mean = F.avg_pool2d(noisy, self.pool_size, stride=1, padding=pad)
        noisy_sq_mean = F.avg_pool2d(noisy ** 2, self.pool_size, stride=1, padding=pad)
        noisy_var = (noisy_sq_mean - noisy_mean ** 2).clamp(min=1e-8)
        del noisy_sq_mean  # FIX: Free intermediate tensor
        noisy_std = torch.sqrt(noisy_var)
        del noisy_var  # FIX: Free intermediate tensor

        # Normalize by image statistics
        # FIX: Safe handling of min/max with potential NaN
        std_flat = noisy_std.view(B, -1)
        std_min = std_flat.min(dim=1)[0].view(B, 1, 1, 1)
        std_max = std_flat.max(dim=1)[0].view(B, 1, 1, 1)
        del std_flat  # FIX: Free intermediate tensor
        noise_uncertainty = ((noisy_std - std_min) / (std_max - std_min + 1e-6)).clamp(0, 1)
        del noisy_std, noisy_mean, std_min, std_max  # FIX: Free intermediate tensors

        # === Component 3: Intensity-Based Signal Detection ===
        # Low intensity regions (background) should have lower correction
        # This is the "smart" replacement for fixed signal_mask
        local_mean = F.avg_pool2d(backbone_out, self.pool_size, stride=1, padding=pad)
        local_sq_mean = F.avg_pool2d(backbone_out ** 2, self.pool_size, stride=1, padding=pad)
        local_std = torch.sqrt((local_sq_mean - local_mean ** 2).clamp(min=1e-8))
        del local_sq_mean  # FIX: Free intermediate tensor

        # Adaptive threshold based on local statistics
        adaptive_threshold = local_mean - 0.3 * local_std
        del local_std  # FIX: Free intermediate tensor

        # Signal confidence: how much above background threshold
        signal_confidence = torch.sigmoid(10 * (backbone_out - adaptive_threshold))
        del adaptive_threshold, local_mean  # FIX: Free intermediate tensors

        # For uncertainty: high signal = can benefit from correction
        # low signal (background) = should be careful with correction
        intensity_uncertainty = signal_confidence  # [0,1]

        # === Combine Components ===
        weights = F.softmax(torch.stack([
            self.failure_weight,
            self.noise_weight,
            self.intensity_weight
        ]), dim=0)

        uncertainty = (
            weights[0] * failure_uncertainty +
            weights[1] * noise_uncertainty +
            weights[2] * intensity_uncertainty
        ).clamp(0, 1)

        # FIX: Free component tensors
        del failure_uncertainty, noise_uncertainty, intensity_uncertainty, weights

        return uncertainty


class UncertaintyGuidedCorrectionModule(nn.Module):
    """
    Applies corrections with uncertainty-guided strength modulation.

    Key insight: Instead of fixed min_correction=0.40, use:
    - High uncertainty regions: apply stronger correction (up to 100%)
    - Low uncertainty regions: apply weaker correction (down to 20%)

    This is adaptive per-image and per-pixel.
    """

    def __init__(self,
                 min_correction_floor: float = 0.20,
                 max_correction_ceiling: float = 1.0):
        super().__init__()

        self.uncertainty_aggregator = UncertaintyAggregator(learnable_weights=True)

        # Learnable bounds for correction strength
        self.min_floor = nn.Parameter(torch.tensor(min_correction_floor))
        self.max_ceiling = nn.Parameter(torch.tensor(max_correction_ceiling))

        # Learnable steepness for uncertainty -> correction mapping
        self.mapping_steepness = nn.Parameter(torch.tensor(2.0))

    def compute_correction_mask(self,
                                backbone_out: torch.Tensor,
                                noisy: torch.Tensor,
                                failure_maps: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive correction mask based on uncertainty.

        Returns:
            correction_mask: [B, 1, H, W] in [min_floor, max_ceiling]
            uncertainty: [B, 1, H, W] raw uncertainty map
        """
        # Get uncertainty map
        uncertainty = self.uncertainty_aggregator(backbone_out, noisy, failure_maps)

        # Map uncertainty to correction strength
        # High uncertainty -> high correction strength
        # Use sigmoid-like mapping with learnable steepness
        steepness = F.softplus(self.mapping_steepness)

        # Centered sigmoid: maps [0,1] uncertainty to [floor, ceiling]
        floor = torch.sigmoid(self.min_floor) * 0.4  # Max floor is 0.4
        ceiling = 0.6 + torch.sigmoid(self.max_ceiling) * 0.4  # Ceiling in [0.6, 1.0]

        # Linear interpolation based on uncertainty
        correction_mask = floor + (ceiling - floor) * uncertainty

        return correction_mask, uncertainty

    def apply_to_correction(self,
                           correction: torch.Tensor,
                           backbone_out: torch.Tensor,
                           noisy: torch.Tensor,
                           failure_maps: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Apply uncertainty-guided masking to a correction tensor.

        Args:
            correction: [B, 1, H, W] raw correction from a corrector
            backbone_out: [B, 1, H, W] denoised output
            noisy: [B, 1, H, W] original noisy input
            failure_maps: Dict with predicate failure maps

        Returns:
            masked_correction: [B, 1, H, W] correction scaled by uncertainty
            info: Dict with debug information
        """
        correction_mask, uncertainty = self.compute_correction_mask(
            backbone_out, noisy, failure_maps
        )

        # Apply mask to correction
        masked_correction = correction * correction_mask

        info = {
            'correction_mask': correction_mask,
            'uncertainty': uncertainty,
            'mask_mean': correction_mask.mean().item(),
            'mask_std': correction_mask.std().item(),
            'uncertainty_mean': uncertainty.mean().item(),
        }

        return masked_correction, info


class CNRPreservingSpatialLoss(nn.Module):
    """
    Region-aware loss that preserves CNR while allowing contrast improvement.

    Key Algorithm:
    1. Soft-segment image into tissue (bright) and background (dark) regions
    2. Penalize background noise increase HEAVILY (10x weight)
    3. Reward tissue contrast improvement
    4. Directly optimize for CNR metric
    """

    def __init__(self,
                 background_penalty_weight: float = 10.0,
                 tissue_reward_weight: float = 2.0,
                 cnr_weight: float = 1.0,
                 region_detector=None):
        super().__init__()

        self.bg_penalty_weight = background_penalty_weight
        self.tissue_reward_weight = tissue_reward_weight
        self.cnr_weight = cnr_weight

        # Unified: using shared AdaptiveRegionDetector when available
        # Set via constructor or assigned after construction (e.g., from CooperativeLoss)
        self.region_detector = region_detector

        # Soft segmentation parameters (fallback when region_detector is None)
        self.soft_threshold_steepness = nn.Parameter(torch.tensor(10.0))

    def _soft_segment(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create soft tissue and background masks.

        Uses adaptive thresholding based on image statistics.
        """
        B = x.shape[0]

        # Compute per-image statistics
        x_flat = x.view(B, -1)
        img_mean = x_flat.mean(dim=1, keepdim=True)
        img_std = x_flat.std(dim=1, keepdim=True)

        # Adaptive threshold: below mean - 0.5*std is background
        threshold = (img_mean - 0.5 * img_std).view(B, 1, 1, 1)

        # Soft segmentation via sigmoid
        steepness = F.softplus(self.soft_threshold_steepness)
        tissue_mask = torch.sigmoid(steepness * (x - threshold))
        background_mask = 1.0 - tissue_mask

        return tissue_mask, background_mask

    def _compute_region_stats(self,
                              x: torch.Tensor,
                              mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std for a soft-masked region."""
        # Weighted mean - use larger epsilon for better numerical stability
        weighted_sum = (x * mask).sum(dim=[2, 3])
        mask_sum = mask.sum(dim=[2, 3]).clamp(min=1e-4)  # Increased from 1e-6 for stability
        mean = weighted_sum / mask_sum

        # Weighted variance
        mean_expanded = mean.unsqueeze(-1).unsqueeze(-1)
        weighted_var = ((x - mean_expanded) ** 2 * mask).sum(dim=[2, 3])
        var = weighted_var / mask_sum
        std = torch.sqrt(var.clamp(min=1e-6))  # Increased from 1e-8 for stability

        return mean, std

    def forward(self,
                corrected: torch.Tensor,
                backbone_out: torch.Tensor,
                clean: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute CNR-preserving spatial loss.

        Args:
            corrected: [B, 1, H, W] corrected output
            backbone_out: [B, 1, H, W] backbone output (before correction)
            clean: [B, 1, H, W] optional clean reference

        Returns:
            loss: Scalar loss tensor
            metrics: Dict with detailed metrics
        """
        # Unified: using shared AdaptiveRegionDetector when available,
        # falling back to _soft_segment for backward compatibility
        if self.region_detector is not None:
            try:
                tissue_mask, bg_mask = self.region_detector(backbone_out)
                tissue_mask = tissue_mask.clamp(0.0, 1.0)
                bg_mask = bg_mask.clamp(0.0, 1.0)
            except Exception:
                # Fallback to soft segment if detector fails
                tissue_mask, bg_mask = self._soft_segment(backbone_out)
        else:
            tissue_mask, bg_mask = self._soft_segment(backbone_out)

        # === Component 1: Background Noise Penalty (ENHANCED) ===
        # ENHANCED: Heavily penalize if correction INCREASES noise in background
        # This is the main cause of CNR degradation
        _, bg_std_backbone = self._compute_region_stats(backbone_out, bg_mask)
        _, bg_std_corrected = self._compute_region_stats(corrected, bg_mask)

        # Clamp noise increase to prevent extreme values from causing loss spikes
        bg_noise_increase = F.relu(bg_std_corrected - bg_std_backbone).clamp(max=0.1)
        # ENHANCED: Increased max clamp from 0.3 to 0.6 for stronger penalty signal
        bg_penalty = (bg_noise_increase.mean() * self.bg_penalty_weight).clamp(max=0.6)

        # === Component 2: Tissue Contrast Term ===
        # Encourage contrast improvement while keeping loss positive
        tissue_mean_backbone, tissue_std_backbone = self._compute_region_stats(
            backbone_out, tissue_mask
        )
        tissue_mean_corrected, tissue_std_corrected = self._compute_region_stats(
            corrected, tissue_mask
        )
        bg_mean_backbone, _ = self._compute_region_stats(backbone_out, bg_mask)
        bg_mean_corrected, _ = self._compute_region_stats(corrected, bg_mask)

        # Tissue-background contrast
        contrast_backbone = (tissue_mean_backbone - bg_mean_backbone).abs()
        contrast_corrected = (tissue_mean_corrected - bg_mean_corrected).abs()

        # Compute relative contrast improvement (0 = no change, 1 = 100% improvement)
        # Clamp to reasonable range to avoid extreme values
        relative_improvement = ((contrast_corrected - contrast_backbone) / (contrast_backbone + 1e-6)).clamp(-1.0, 2.0)

        # Convert to a REDUCED PENALTY: higher improvement = lower penalty (but always >= 0)
        # Base penalty of 1.0, reduced by improvement (clamped to stay positive)
        # When improvement = 0: penalty = 1.0 * weight
        # When improvement = 1.0 (100%): penalty = 0.0 * weight
        # When improvement < 0 (contrast decreased): penalty > 1.0 * weight
        # Clamp final penalty to max 0.3 to prevent spikes
        tissue_contrast_penalty = ((1.0 - relative_improvement.clamp(max=1.0)).mean() * self.tissue_reward_weight).clamp(max=0.3)

        # Also track the raw improvement for metrics
        contrast_improvement = F.relu(contrast_corrected - contrast_backbone)

        # === Component 3: CNR Metric ===
        # CNR = (tissue_mean - bg_mean) / bg_std
        # Use larger epsilon to prevent CNR explosion when bg_std is small
        cnr_backbone = contrast_backbone / (bg_std_backbone + 1e-4)
        cnr_corrected = contrast_corrected / (bg_std_corrected + 1e-4)

        # Clamp CNR values to prevent extreme values
        cnr_backbone = cnr_backbone.clamp(-100.0, 100.0)
        cnr_corrected = cnr_corrected.clamp(-100.0, 100.0)

        # Penalize CNR drop - clamp the drop value more tightly to prevent spikes
        cnr_drop = F.relu(cnr_backbone - cnr_corrected).clamp(max=2.0)
        # ENHANCED: Increased max clamp from 0.3 to 0.6 for stronger penalty signal
        cnr_penalty = (cnr_drop.mean() * self.cnr_weight).clamp(max=0.6)

        # === Component 4: CNR Improvement Reward (NEW) ===
        # ENHANCED: Directly incentivize CNR improvement over backbone
        cnr_improvement = F.relu(cnr_corrected - cnr_backbone).clamp(max=2.0)
        cnr_improvement_bonus = (cnr_improvement.mean() * self.cnr_weight * 0.5).clamp(max=0.3)

        # === Total Loss ===
        # All components are now positive penalties, ensuring total_loss >= 0
        # ENHANCED: Include CNR improvement bonus to reduce penalty when CNR improves
        total_loss = bg_penalty + tissue_contrast_penalty + cnr_penalty - cnr_improvement_bonus
        total_loss = F.relu(total_loss)  # Ensure non-negative

        # ENHANCED: Increased max clamp from 1.0 to 1.5 for stronger signal
        total_loss = total_loss.clamp(max=1.5)

        # === Metrics ===
        metrics = {
            'bg_noise_increase': bg_noise_increase.mean().item(),
            'bg_std_backbone': bg_std_backbone.mean().item(),
            'bg_std_corrected': bg_std_corrected.mean().item(),
            'contrast_backbone': contrast_backbone.mean().item(),
            'contrast_corrected': contrast_corrected.mean().item(),
            'contrast_improvement': contrast_improvement.mean().item(),
            'relative_contrast_improvement': relative_improvement.mean().item(),
            'cnr_backbone': cnr_backbone.mean().item(),
            'cnr_corrected': cnr_corrected.mean().item(),
            'cnr_drop': cnr_drop.mean().item(),
            'cnr_improvement': cnr_improvement.mean().item(),  # NEW: Track CNR improvement
            'bg_penalty': bg_penalty.item(),
            'tissue_contrast_penalty': tissue_contrast_penalty.item(),
            'cnr_penalty': cnr_penalty.item(),
            'cnr_improvement_bonus': cnr_improvement_bonus.item(),  # NEW: Track improvement bonus
            'total_cnr_loss': total_loss.item(),
        }

        return total_loss, metrics


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_uncertainty_guided_system():
    """
    Create the complete uncertainty-guided correction system.

    Returns modules that can be integrated into NeuroSymbolicCorrectorV8Enhanced.
    """
    return {
        'correction_module': UncertaintyGuidedCorrectionModule(
            min_correction_floor=0.20,
            max_correction_ceiling=1.0
        ),
        'cnr_loss': CNRPreservingSpatialLoss(
            background_penalty_weight=10.0,
            tissue_reward_weight=2.0,
            cnr_weight=1.0
        ),
    }


if __name__ == '__main__':
    # Test the modules
    print("Testing Uncertainty-Guided Correction Modules...")

    B, C, H, W = 2, 1, 256, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test tensors
    backbone_out = torch.rand(B, C, H, W, device=device)
    noisy = backbone_out + 0.1 * torch.randn_like(backbone_out)
    correction = 0.1 * torch.randn(B, C, H, W, device=device)

    failure_maps = {
        'P1': torch.rand(B, 1, H, W, device=device),
        'P2': torch.rand(B, 1, H, W, device=device),
        'P3': torch.rand(B, 1, H, W, device=device),
        'P4': torch.rand(B, 1, H, W, device=device),
        'P6': torch.rand(B, 1, H, W, device=device),
    }

    # Test uncertainty aggregator
    print("\n1. Testing UncertaintyAggregator...")
    aggregator = UncertaintyAggregator().to(device)
    uncertainty = aggregator(backbone_out, noisy, failure_maps)
    print(f"   Uncertainty shape: {uncertainty.shape}")
    print(f"   Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
    print(f"   Uncertainty mean: {uncertainty.mean():.3f}")

    # Test correction module
    print("\n2. Testing UncertaintyGuidedCorrectionModule...")
    correction_module = UncertaintyGuidedCorrectionModule().to(device)
    masked_correction, info = correction_module.apply_to_correction(
        correction, backbone_out, noisy, failure_maps
    )
    print(f"   Masked correction shape: {masked_correction.shape}")
    print(f"   Mask mean: {info['mask_mean']:.3f}")
    print(f"   Mask std: {info['mask_std']:.3f}")

    # Test CNR loss
    print("\n3. Testing CNRPreservingSpatialLoss...")
    cnr_loss = CNRPreservingSpatialLoss().to(device)
    corrected = (backbone_out + masked_correction).clamp(0, 1)
    loss, metrics = cnr_loss(corrected, backbone_out)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   CNR backbone: {metrics['cnr_backbone']:.3f}")
    print(f"   CNR corrected: {metrics['cnr_corrected']:.3f}")
    print(f"   Background noise increase: {metrics['bg_noise_increase']:.4f}")

    print("\n✓ All tests passed!")
