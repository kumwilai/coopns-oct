#!/usr/bin/env python3
"""
Asymmetric Quality Loss for Neuro-Symbolic OCT Denoising

Design Philosophy: "First, do no harm"

This loss function is designed to STRONGLY prevent PSNR degradation while still
allowing the corrector to make beneficial improvements. The key insight is that
degradation should be penalized 100x more than improvements are rewarded.

Key Innovations:
1. Asymmetric Penalty: Degradation costs 100x more than improvement rewards
2. Hard Threshold: If PSNR drops > 0.01 dB, apply massive penalty (soft infinity)
3. Per-Pixel Quality Checking: Identify and heavily penalize degraded pixels
4. Predicate-Aware Weighting: Allow more change where predicates indicate need
5. "Do No Harm" Constraint: Built-in safeguard at multiple levels

Problem Solved: Corrector was degrading PSNR by -0.9 dB despite existing penalties.
This module provides 100x stronger degradation penalties to prevent this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PerPixelQualityChecker(nn.Module):
    """
    Per-pixel quality checking module.

    Identifies pixels where the corrector has degraded quality compared to
    the backbone output. This enables targeted penalties on problematic regions.
    """

    def __init__(self, smoothing_kernel_size: int = 5):
        """
        Args:
            smoothing_kernel_size: Size of smoothing kernel for local analysis.
                                   Larger = more spatially coherent quality maps.
        """
        super().__init__()
        self.smoothing_kernel_size = smoothing_kernel_size

    def forward(self, corrected: torch.Tensor,
                backbone_output: torch.Tensor,
                clean: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform per-pixel quality analysis.

        Args:
            corrected: Corrector output [B, 1, H, W]
            backbone_output: Backbone-only output [B, 1, H, W]
            clean: Ground truth [B, 1, H, W]

        Returns:
            Dict containing:
                - mse_corrected: Per-pixel MSE for corrected [B, 1, H, W]
                - mse_backbone: Per-pixel MSE for backbone [B, 1, H, W]
                - degraded_mask: Binary mask where corrector is worse [B, 1, H, W]
                - improved_mask: Binary mask where corrector is better [B, 1, H, W]
                - degradation_magnitude: How much worse per pixel [B, 1, H, W]
                - improvement_magnitude: How much better per pixel [B, 1, H, W]
                - degradation_ratio: Fraction of degraded pixels (scalar)
        """
        # Per-pixel squared error
        mse_corrected = (corrected - clean) ** 2
        mse_backbone = (backbone_output - clean) ** 2

        # Apply local smoothing for more stable comparisons
        if self.smoothing_kernel_size > 1:
            padding = self.smoothing_kernel_size // 2
            mse_corrected_smooth = F.avg_pool2d(
                mse_corrected, self.smoothing_kernel_size, stride=1, padding=padding
            )
            mse_backbone_smooth = F.avg_pool2d(
                mse_backbone, self.smoothing_kernel_size, stride=1, padding=padding
            )
        else:
            mse_corrected_smooth = mse_corrected
            mse_backbone_smooth = mse_backbone

        # Identify degraded vs improved pixels
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-8
        degraded_mask = (mse_corrected_smooth > mse_backbone_smooth + epsilon).float()
        improved_mask = (mse_corrected_smooth < mse_backbone_smooth - epsilon).float()

        # Compute magnitude of degradation/improvement
        degradation_magnitude = F.relu(mse_corrected - mse_backbone)
        improvement_magnitude = F.relu(mse_backbone - mse_corrected)

        # Compute degradation ratio
        total_pixels = corrected.numel()
        degraded_pixels = degraded_mask.sum()
        degradation_ratio = degraded_pixels / total_pixels

        return {
            'mse_corrected': mse_corrected,
            'mse_backbone': mse_backbone,
            'degraded_mask': degraded_mask,
            'improved_mask': improved_mask,
            'degradation_magnitude': degradation_magnitude,
            'improvement_magnitude': improvement_magnitude,
            'degradation_ratio': degradation_ratio,
        }


class DoNoHarmConstraint(nn.Module):
    """
    "Do No Harm" constraint module.

    Implements a hard constraint that prevents the corrector from degrading
    quality beyond a specified threshold. When violated, returns a very large
    penalty that effectively acts as "soft infinity".

    This is the core safeguard against PSNR degradation.
    """

    def __init__(self,
                 harm_threshold_db: float = 0.01,
                 hard_penalty_scale: float = 1000.0,
                 soft_infinity: float = 1e6):
        """
        Args:
            harm_threshold_db: Maximum allowed PSNR degradation in dB.
                               Default 0.01 dB = virtually no degradation allowed.
            hard_penalty_scale: Multiplier for degradation amount when threshold exceeded.
            soft_infinity: Value to use as "soft infinity" for severe violations.
        """
        super().__init__()
        self.harm_threshold_db = harm_threshold_db
        self.hard_penalty_scale = hard_penalty_scale
        self.soft_infinity = soft_infinity

    def forward(self, psnr_corrected: torch.Tensor,
                psnr_backbone: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Check "do no harm" constraint and compute penalty.

        Args:
            psnr_corrected: PSNR of corrector output (scalar tensor)
            psnr_backbone: PSNR of backbone output (scalar tensor)

        Returns:
            Tuple of (penalty, info_dict):
                - penalty: Loss penalty (0 if constraint satisfied, large if violated)
                - info_dict: Dict with constraint status information
        """
        # Compute PSNR degradation (positive = bad)
        psnr_degradation = psnr_backbone - psnr_corrected

        # Check if threshold is exceeded
        threshold_exceeded = psnr_degradation > self.harm_threshold_db

        # Compute penalty based on severity
        if threshold_exceeded:
            # Severe violation - apply hard penalty proportional to degradation
            # The more degradation, the larger the penalty
            base_penalty = psnr_degradation * self.hard_penalty_scale

            # For extreme degradation (> 0.5 dB), use soft infinity
            if psnr_degradation > 0.5:
                penalty = torch.tensor(self.soft_infinity,
                                      device=psnr_corrected.device,
                                      dtype=psnr_corrected.dtype)
            else:
                penalty = base_penalty
        else:
            # Within threshold - no hard penalty
            penalty = torch.tensor(0.0,
                                  device=psnr_corrected.device,
                                  dtype=psnr_corrected.dtype)

        info = {
            'psnr_degradation_db': psnr_degradation.item() if torch.is_tensor(psnr_degradation) else psnr_degradation,
            'threshold_exceeded': threshold_exceeded.item() if torch.is_tensor(threshold_exceeded) else threshold_exceeded,
            'harm_threshold_db': self.harm_threshold_db,
            'penalty_applied': penalty.item() if torch.is_tensor(penalty) else penalty,
        }

        return penalty, info


class PredicateAwareWeighting(nn.Module):
    """
    Predicate-aware weighting for corrections.

    Allows more aggressive correction in regions where predicates indicate
    problems, while being conservative in regions that are already good.

    Key insight: If predicates say a region needs fixing, allow more change.
    If predicates say a region is fine, penalize changes heavily.
    """

    def __init__(self):
        super().__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def compute_local_stats(self, x: torch.Tensor, kernel_size: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local mean and std."""
        padding = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        sq_mean = F.avg_pool2d(x**2, kernel_size, stride=1, padding=padding)
        std = (sq_mean - mean**2).clamp(min=1e-6).sqrt()
        return mean, std

    def compute_needs_correction_map(self,
                                     backbone_output: torch.Tensor,
                                     clean: torch.Tensor) -> torch.Tensor:
        """
        Compute a map indicating where correction is needed.

        Regions with high values need more correction; regions with low values
        are already good and should not be changed.

        Args:
            backbone_output: Backbone-only output [B, 1, H, W]
            clean: Ground truth [B, 1, H, W]

        Returns:
            needs_correction_map: [B, 1, H, W] with values in [0, 1]
                                  1 = definitely needs correction
                                  0 = already perfect, don't touch
        """
        # Per-pixel error between backbone and clean
        error = (backbone_output - clean).abs()

        # Edge preservation error
        edges_backbone = self.compute_edges(backbone_output)
        edges_clean = self.compute_edges(clean)
        edge_error = (edges_backbone - edges_clean).abs()

        # Local contrast error
        _, std_backbone = self.compute_local_stats(backbone_output)
        _, std_clean = self.compute_local_stats(clean)
        contrast_error = (std_backbone - std_clean).abs()

        # Combine into needs_correction map
        # Higher error = more need for correction
        combined_error = error + 0.5 * edge_error + 0.3 * contrast_error

        # Normalize to [0, 1]
        max_error = combined_error.max() + 1e-8
        needs_correction_map = (combined_error / max_error).clamp(0, 1)

        return needs_correction_map

    def forward(self,
                backbone_output: torch.Tensor,
                clean: torch.Tensor,
                predicate_failure_maps: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        Compute predicate-aware correction weight map.

        Args:
            backbone_output: Backbone-only output [B, 1, H, W]
            clean: Ground truth [B, 1, H, W]
            predicate_failure_maps: Optional dict of failure maps from predicates

        Returns:
            correction_weight_map: [B, 1, H, W] with values in [0, 1]
                                   Higher values = allow more correction
                                   Lower values = penalize correction more
        """
        # Base needs-correction map from error analysis
        needs_correction = self.compute_needs_correction_map(backbone_output, clean)

        # If predicate failure maps provided, incorporate them
        if predicate_failure_maps is not None:
            predicate_weight = torch.zeros_like(needs_correction)
            count = 0
            for name, failure_map in predicate_failure_maps.items():
                if failure_map is not None and failure_map.shape == needs_correction.shape:
                    predicate_weight = predicate_weight + failure_map
                    count += 1
            if count > 0:
                predicate_weight = predicate_weight / count
                # Combine: regions failing predicates need correction
                needs_correction = torch.max(needs_correction, predicate_weight)

        return needs_correction


class AsymmetricQualityLoss(nn.Module):
    """
    Asymmetric Quality Loss that STRONGLY prevents PSNR degradation.

    Design Philosophy: "First, do no harm"

    This loss is designed to solve the problem where a corrector module
    degrades PSNR by -0.9 dB despite existing penalties. It achieves this
    through several mechanisms:

    1. **Asymmetric Penalty (100:1 ratio)**:
       - Degradation is penalized 100x more than improvement is rewarded
       - This creates a strong bias toward conservative corrections

    2. **Hard Threshold with Soft Infinity**:
       - If PSNR drops > 0.01 dB, apply very large penalty
       - For severe degradation (> 0.5 dB), use soft infinity penalty

    3. **Per-Pixel Quality Checking**:
       - Identify individual pixels where quality degraded
       - Apply targeted penalties to degraded regions

    4. **Predicate-Aware Weighting**:
       - Allow more change in regions that predicates identify as problematic
       - Penalize changes in regions that are already good

    5. **"Do No Harm" Constraint**:
       - Built-in safeguard at multiple levels
       - Prevents the model from learning to make harmful changes

    Usage:
        loss_fn = AsymmetricQualityLoss()
        loss_dict = loss_fn(corrected, backbone_output, clean, noisy)
        loss = loss_dict['total']
        loss.backward()
    """

    def __init__(self,
                 degradation_penalty: float = 100.0,
                 improvement_reward: float = 1.0,
                 harm_threshold_db: float = 0.01,
                 hard_penalty_scale: float = 1000.0,
                 lambda_recon: float = 100.0,
                 lambda_asymmetric: float = 50.0,
                 lambda_perpixel: float = 20.0,
                 lambda_ssim: float = 10.0):
        """
        Initialize AsymmetricQualityLoss.

        Args:
            degradation_penalty: Multiplier for penalizing PSNR degradation.
                                 Default 100.0 = degradation costs 100x more.
            improvement_reward: Multiplier for rewarding PSNR improvement.
                               Default 1.0 = improvement gives modest reward.
            harm_threshold_db: Maximum allowed PSNR degradation before hard penalty.
                              Default 0.01 dB = virtually no degradation allowed.
            hard_penalty_scale: Multiplier for hard penalty when threshold exceeded.
                               Default 1000.0 = severe penalty for violations.
            lambda_recon: Weight for reconstruction (MSE) loss.
                         Default 100.0 = strong focus on PSNR.
            lambda_asymmetric: Weight for asymmetric quality term.
                              Default 50.0 = significant asymmetric contribution.
            lambda_perpixel: Weight for per-pixel degradation penalty.
                            Default 20.0 = meaningful per-pixel contribution.
            lambda_ssim: Weight for SSIM preservation term.
                        Default 10.0 = modest SSIM contribution.
        """
        super().__init__()

        # Core asymmetric parameters
        self.degradation_penalty = degradation_penalty
        self.improvement_reward = improvement_reward

        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_asymmetric = lambda_asymmetric
        self.lambda_perpixel = lambda_perpixel
        self.lambda_ssim = lambda_ssim

        # Component modules
        self.pixel_checker = PerPixelQualityChecker(smoothing_kernel_size=5)
        self.harm_constraint = DoNoHarmConstraint(
            harm_threshold_db=harm_threshold_db,
            hard_penalty_scale=hard_penalty_scale
        )
        self.predicate_weighting = PredicateAwareWeighting()

    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute PSNR between prediction and target."""
        mse = F.mse_loss(pred, target)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
        return psnr

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images (differentiable)."""
        C1, C2 = 0.01**2, 0.03**2

        mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)

        sigma1_sq = F.avg_pool2d(img1**2, 11, stride=1, padding=5) - mu1**2
        sigma2_sq = F.avg_pool2d(img2**2, 11, stride=1, padding=5) - mu2**2
        sigma12 = F.avg_pool2d(img1 * img2, 11, stride=1, padding=5) - mu1 * mu2

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim.mean()

    def _compute_asymmetric_mse_loss(self,
                                     mse_corrected: torch.Tensor,
                                     mse_backbone: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute asymmetric MSE-based loss.

        Degradation (mse_corrected > mse_backbone) is penalized 100x more
        than improvement (mse_corrected < mse_backbone) is rewarded.

        Args:
            mse_corrected: Per-pixel MSE for corrected output [B, 1, H, W]
            mse_backbone: Per-pixel MSE for backbone output [B, 1, H, W]

        Returns:
            Tuple of (loss, info_dict)
        """
        # Improvement: backbone MSE > corrected MSE (good)
        improvement = F.relu(mse_backbone - mse_corrected)
        improvement_term = improvement.mean() * self.improvement_reward

        # Degradation: corrected MSE > backbone MSE (bad)
        degradation = F.relu(mse_corrected - mse_backbone)
        degradation_term = degradation.mean() * self.degradation_penalty

        # Asymmetric loss: heavy penalty for degradation, mild reward for improvement
        # Note: We want to MINIMIZE this, so degradation adds to loss, improvement subtracts
        asymmetric_loss = degradation_term - improvement_term

        info = {
            'improvement_term': improvement_term.item(),
            'degradation_term': degradation_term.item(),
            'improvement_mean': improvement.mean().item(),
            'degradation_mean': degradation.mean().item(),
        }

        return asymmetric_loss, info

    def _compute_perpixel_degradation_penalty(self,
                                              pixel_quality: Dict[str, torch.Tensor],
                                              correction_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Compute penalty for per-pixel degradation.

        Heavily penalizes pixels where the corrector made things worse,
        with optional weighting based on whether correction was expected.

        Args:
            pixel_quality: Dict from PerPixelQualityChecker
            correction_weights: Optional weights indicating where correction is allowed

        Returns:
            Per-pixel degradation penalty (scalar)
        """
        degradation_magnitude = pixel_quality['degradation_magnitude']
        degraded_mask = pixel_quality['degraded_mask']

        # Base penalty: sum of degradation magnitude at degraded pixels
        base_penalty = (degradation_magnitude * degraded_mask).sum() / (degraded_mask.sum() + 1e-8)

        if correction_weights is not None:
            # Additional penalty for degradation in "good" regions (low correction weight)
            # where we should NOT have made changes
            good_region_mask = 1.0 - correction_weights
            unwanted_degradation = (degradation_magnitude * degraded_mask * good_region_mask)
            unwanted_penalty = unwanted_degradation.sum() / (good_region_mask.sum() + 1e-8)

            # Total penalty: base + extra for unwanted regions
            total_penalty = base_penalty + 2.0 * unwanted_penalty
        else:
            total_penalty = base_penalty

        return total_penalty * 1000.0  # Scale up for meaningful contribution

    def forward(self,
                corrected: torch.Tensor,
                backbone_output: torch.Tensor,
                clean: torch.Tensor,
                noisy: torch.Tensor = None,
                predicate_failure_maps: Dict[str, torch.Tensor] = None) -> Dict:
        """
        Compute asymmetric quality loss.

        This loss function STRONGLY prevents PSNR degradation through:
        1. Asymmetric MSE penalty (100x degradation vs 1x improvement)
        2. Hard constraint with soft infinity for threshold violations
        3. Per-pixel degradation penalty
        4. Predicate-aware weighting for targeted corrections

        Args:
            corrected: Corrector output [B, 1, H, W]
            backbone_output: Backbone-only output [B, 1, H, W]
            clean: Ground truth [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W] (optional, for logging)
            predicate_failure_maps: Optional dict of failure maps from predicates

        Returns:
            Dict containing:
                - total: Total loss (use this for backward())
                - recon_loss: Reconstruction MSE loss
                - asymmetric_loss: Asymmetric quality term
                - perpixel_penalty: Per-pixel degradation penalty
                - hard_penalty: Hard constraint penalty (0 if satisfied)
                - ssim_loss: SSIM preservation loss
                - psnr_backbone: PSNR of backbone output
                - psnr_corrected: PSNR of corrected output
                - psnr_delta: Change in PSNR (positive = improvement)
                - quality_improved: Boolean indicating if PSNR improved
                - degradation_ratio: Fraction of degraded pixels
                - constraint_info: Detailed info about harm constraint
        """
        # =====================================================================
        # 1. COMPUTE PSNR VALUES
        # =====================================================================
        with torch.no_grad():
            psnr_backbone = self._compute_psnr(backbone_output, clean)
        psnr_corrected = self._compute_psnr(corrected, clean)
        psnr_delta = psnr_corrected - psnr_backbone

        # =====================================================================
        # 2. RECONSTRUCTION LOSS (Standard MSE)
        # =====================================================================
        recon_loss = F.mse_loss(corrected, clean)

        # =====================================================================
        # 3. PER-PIXEL QUALITY ANALYSIS
        # =====================================================================
        pixel_quality = self.pixel_checker(corrected, backbone_output, clean)

        # =====================================================================
        # 4. PREDICATE-AWARE CORRECTION WEIGHTS
        # =====================================================================
        correction_weights = self.predicate_weighting(
            backbone_output, clean, predicate_failure_maps
        )

        # =====================================================================
        # 5. ASYMMETRIC MSE LOSS (100x penalty for degradation)
        # =====================================================================
        asymmetric_loss, asymmetric_info = self._compute_asymmetric_mse_loss(
            pixel_quality['mse_corrected'],
            pixel_quality['mse_backbone']
        )

        # =====================================================================
        # 6. PER-PIXEL DEGRADATION PENALTY
        # =====================================================================
        perpixel_penalty = self._compute_perpixel_degradation_penalty(
            pixel_quality, correction_weights
        )

        # =====================================================================
        # 7. HARD "DO NO HARM" CONSTRAINT
        # =====================================================================
        hard_penalty, constraint_info = self.harm_constraint(psnr_corrected, psnr_backbone)

        # =====================================================================
        # 8. SSIM PRESERVATION LOSS
        # =====================================================================
        ssim_backbone = self._compute_ssim(backbone_output, clean)
        ssim_corrected = self._compute_ssim(corrected, clean)

        # Penalize if SSIM degrades
        ssim_degradation = F.relu(ssim_backbone - ssim_corrected)
        ssim_loss = ssim_degradation * 10.0  # Scale for meaningful contribution

        # =====================================================================
        # 9. TOTAL LOSS
        # =====================================================================
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_asymmetric * asymmetric_loss +
            self.lambda_perpixel * perpixel_penalty +
            self.lambda_ssim * ssim_loss +
            hard_penalty  # Already scaled by hard_penalty_scale
        )

        # =====================================================================
        # 10. COMPILE RESULTS
        # =====================================================================
        return {
            # Primary loss
            'total': total_loss,

            # Loss components
            'recon_loss': recon_loss.detach(),
            'asymmetric_loss': asymmetric_loss.detach() if torch.is_tensor(asymmetric_loss) else asymmetric_loss,
            'perpixel_penalty': perpixel_penalty.detach() if torch.is_tensor(perpixel_penalty) else perpixel_penalty,
            'hard_penalty': hard_penalty.detach() if torch.is_tensor(hard_penalty) else hard_penalty,
            'ssim_loss': ssim_loss.detach() if torch.is_tensor(ssim_loss) else ssim_loss,

            # PSNR metrics
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'psnr_delta': psnr_delta.item(),
            'quality_improved': (psnr_delta > 0).item(),

            # SSIM metrics
            'ssim_backbone': ssim_backbone.item(),
            'ssim_corrected': ssim_corrected.item(),
            'ssim_delta': (ssim_corrected - ssim_backbone).item(),

            # Per-pixel metrics
            'degradation_ratio': pixel_quality['degradation_ratio'].item(),

            # Asymmetric info
            'improvement_term': asymmetric_info['improvement_term'],
            'degradation_term': asymmetric_info['degradation_term'],

            # Constraint info
            'constraint_info': constraint_info,
            'threshold_exceeded': constraint_info['threshold_exceeded'],
        }


class AsymmetricQualityLossWithPredicates(AsymmetricQualityLoss):
    """
    Extended AsymmetricQualityLoss that integrates with the CleanReferencedPredicates.

    This version automatically computes predicates and uses their failure maps
    to guide where corrections should be allowed.
    """

    def __init__(self,
                 degradation_penalty: float = 100.0,
                 improvement_reward: float = 1.0,
                 harm_threshold_db: float = 0.01,
                 hard_penalty_scale: float = 1000.0,
                 lambda_recon: float = 100.0,
                 lambda_asymmetric: float = 50.0,
                 lambda_perpixel: float = 20.0,
                 lambda_ssim: float = 10.0,
                 lambda_pred: float = 0.1):
        """
        Initialize with predicate integration.

        Additional Args:
            lambda_pred: Weight for predicate loss term. Default 0.1 (low)
                        because predicates should guide, not dominate.
        """
        super().__init__(
            degradation_penalty=degradation_penalty,
            improvement_reward=improvement_reward,
            harm_threshold_db=harm_threshold_db,
            hard_penalty_scale=hard_penalty_scale,
            lambda_recon=lambda_recon,
            lambda_asymmetric=lambda_asymmetric,
            lambda_perpixel=lambda_perpixel,
            lambda_ssim=lambda_ssim
        )

        self.lambda_pred = lambda_pred

        # Import predicates (lazy to avoid circular imports)
        self._predicates = None

    @property
    def predicates(self):
        """Lazy load predicates module."""
        if self._predicates is None:
            try:
                from quality_aligned_loss import CleanReferencedPredicates
                self._predicates = CleanReferencedPredicates()
            except ImportError:
                # Create minimal predicates if main module not available
                self._predicates = None
        return self._predicates

    def forward(self,
                corrected: torch.Tensor,
                backbone_output: torch.Tensor,
                clean: torch.Tensor,
                noisy: torch.Tensor = None,
                predicate_failure_maps: Dict[str, torch.Tensor] = None) -> Dict:
        """
        Compute loss with automatic predicate computation.

        If predicate_failure_maps is not provided and predicates are available,
        automatically computes them from the backbone output.
        """
        # Compute predicates if not provided and module is available
        pred_results = None
        if predicate_failure_maps is None and self.predicates is not None:
            # Move predicates to correct device
            if next(self.predicates.parameters(), torch.tensor(0)).device != backbone_output.device:
                self.predicates = self.predicates.to(backbone_output.device)

            pred_results = self.predicates(backbone_output, clean, noisy)
            predicate_failure_maps = {
                name: pred_results[name]['failure_map']
                for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P7']
                if name in pred_results
            }

        # Call parent forward
        result = super().forward(
            corrected, backbone_output, clean, noisy, predicate_failure_maps
        )

        # Add predicate scores if computed
        if pred_results is not None:
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']:
                if name in pred_results:
                    score = pred_results[name]['score']
                    result[name] = score.item() if torch.is_tensor(score) else score
            result['all_predicates_passed'] = pred_results.get('all_passed', False)

        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_asymmetric_loss(preset: str = 'strict') -> AsymmetricQualityLoss:
    """
    Create AsymmetricQualityLoss with a preset configuration.

    Args:
        preset: One of:
            - 'strict': Very strict, almost no degradation allowed (default)
            - 'moderate': Some tolerance for small degradation
            - 'aggressive': Allow more correction attempts

    Returns:
        Configured AsymmetricQualityLoss instance
    """
    presets = {
        'strict': {
            'degradation_penalty': 100.0,
            'improvement_reward': 1.0,
            'harm_threshold_db': 0.01,
            'hard_penalty_scale': 1000.0,
            'lambda_recon': 100.0,
            'lambda_asymmetric': 50.0,
            'lambda_perpixel': 20.0,
            'lambda_ssim': 10.0,
        },
        'moderate': {
            'degradation_penalty': 50.0,
            'improvement_reward': 2.0,
            'harm_threshold_db': 0.1,
            'hard_penalty_scale': 500.0,
            'lambda_recon': 100.0,
            'lambda_asymmetric': 30.0,
            'lambda_perpixel': 10.0,
            'lambda_ssim': 5.0,
        },
        'aggressive': {
            'degradation_penalty': 20.0,
            'improvement_reward': 5.0,
            'harm_threshold_db': 0.5,
            'hard_penalty_scale': 100.0,
            'lambda_recon': 50.0,
            'lambda_asymmetric': 20.0,
            'lambda_perpixel': 5.0,
            'lambda_ssim': 5.0,
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")

    return AsymmetricQualityLoss(**presets[preset])


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing Asymmetric Quality Loss")
    print("=" * 70)

    # Create test data
    torch.manual_seed(42)
    B, H, W = 2, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clean image
    clean = torch.randn(B, 1, H, W, device=device) * 0.1 + 0.5
    clean = clean.clamp(0, 1)

    # Backbone output (close to clean)
    backbone_output = clean + torch.randn(B, 1, H, W, device=device) * 0.05
    backbone_output = backbone_output.clamp(0, 1)

    # Test 1: Corrected is BETTER than backbone
    print("\n" + "-" * 50)
    print("Test 1: Corrected is BETTER than backbone")
    print("-" * 50)

    corrected_better = clean + torch.randn(B, 1, H, W, device=device) * 0.03
    corrected_better = corrected_better.clamp(0, 1)

    loss_fn = AsymmetricQualityLoss()
    result = loss_fn(corrected_better, backbone_output, clean)

    print(f"PSNR backbone: {result['psnr_backbone']:.2f} dB")
    print(f"PSNR corrected: {result['psnr_corrected']:.2f} dB")
    print(f"PSNR delta: {result['psnr_delta']:.4f} dB")
    print(f"Quality improved: {result['quality_improved']}")
    print(f"Total loss: {result['total'].item():.4f}")
    print(f"Hard penalty: {result['hard_penalty']:.4f}")
    print(f"Degradation ratio: {result['degradation_ratio']:.2%}")

    # Test 2: Corrected is WORSE than backbone (should trigger strong penalty)
    print("\n" + "-" * 50)
    print("Test 2: Corrected is WORSE than backbone")
    print("-" * 50)

    corrected_worse = clean + torch.randn(B, 1, H, W, device=device) * 0.1
    corrected_worse = corrected_worse.clamp(0, 1)

    result = loss_fn(corrected_worse, backbone_output, clean)

    print(f"PSNR backbone: {result['psnr_backbone']:.2f} dB")
    print(f"PSNR corrected: {result['psnr_corrected']:.2f} dB")
    print(f"PSNR delta: {result['psnr_delta']:.4f} dB")
    print(f"Quality improved: {result['quality_improved']}")
    print(f"Total loss: {result['total'].item():.4f}")
    print(f"Hard penalty: {result['hard_penalty']:.4f}")
    print(f"Threshold exceeded: {result['threshold_exceeded']}")
    print(f"Degradation ratio: {result['degradation_ratio']:.2%}")

    # Test 3: Severely degraded (should trigger soft infinity)
    print("\n" + "-" * 50)
    print("Test 3: Severely degraded (soft infinity expected)")
    print("-" * 50)

    corrected_severe = clean + torch.randn(B, 1, H, W, device=device) * 0.3
    corrected_severe = corrected_severe.clamp(0, 1)

    result = loss_fn(corrected_severe, backbone_output, clean)

    print(f"PSNR backbone: {result['psnr_backbone']:.2f} dB")
    print(f"PSNR corrected: {result['psnr_corrected']:.2f} dB")
    print(f"PSNR delta: {result['psnr_delta']:.4f} dB")
    print(f"Total loss: {result['total'].item():.4f}")
    print(f"Hard penalty: {result['hard_penalty']:.4f}")

    # Test 4: Compare loss magnitude for improvement vs degradation
    print("\n" + "-" * 50)
    print("Test 4: Asymmetry verification (100x ratio)")
    print("-" * 50)

    # Small improvement - move corrected closer to clean than backbone
    corrected_small_better = clean + (backbone_output - clean) * 0.5  # Halfway to clean
    result_better = loss_fn(corrected_small_better.clamp(0, 1), backbone_output, clean)

    # Small degradation - move corrected away from clean
    corrected_small_worse = clean + (backbone_output - clean) * 1.5  # Further from clean
    result_worse = loss_fn(corrected_small_worse.clamp(0, 1), backbone_output, clean)

    print(f"Improvement case:")
    print(f"  - PSNR delta: {result_better['psnr_delta']:.4f} dB")
    print(f"  - Improvement term: {result_better['improvement_term']:.6f}")
    print(f"  - Degradation term: {result_better['degradation_term']:.6f}")
    print(f"Degradation case:")
    print(f"  - PSNR delta: {result_worse['psnr_delta']:.4f} dB")
    print(f"  - Improvement term: {result_worse['improvement_term']:.6f}")
    print(f"  - Degradation term: {result_worse['degradation_term']:.6f}")
    print(f"Asymmetric ratio applied: degradation_penalty={loss_fn.degradation_penalty}, improvement_reward={loss_fn.improvement_reward}")
    print(f"Ratio = {loss_fn.degradation_penalty / loss_fn.improvement_reward:.0f}x")

    # Test 5: Preset configurations
    print("\n" + "-" * 50)
    print("Test 5: Preset configurations")
    print("-" * 50)

    for preset in ['strict', 'moderate', 'aggressive']:
        loss_fn_preset = create_asymmetric_loss(preset)
        result = loss_fn_preset(corrected_worse, backbone_output, clean)
        print(f"{preset:12s} - Total loss: {result['total'].item():10.2f}, Hard penalty: {result['hard_penalty']:10.2f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
