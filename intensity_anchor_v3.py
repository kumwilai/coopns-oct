#!/usr/bin/env python3
"""
IntensityAnchoredBoundaryLoss V3 - Best of V1 + V2

Key insights from V1 vs V2 comparison:
- V2 has better detection (robust ensemble method)
- V1 improves more during training (simpler loss)
- Too many constraints can conflict

V3 Strategy:
1. Use V2's robust detection (ensemble of methods)
2. Simpler loss function with adaptive weighting
3. Curriculum learning: focus on anchors first, then add refinements
4. Learnable loss weights (uncertainty-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class IntensityAnchoredBoundaryLossV3(nn.Module):
    """
    V3: Robust detection + Adaptive loss weighting + Curriculum learning
    """

    def __init__(
        self,
        # Base weights (will be adapted)
        lambda_ilm_anchor: float = 1.0,
        lambda_rpe_anchor: float = 1.0,
        lambda_edge_align: float = 0.5,
        lambda_smoothness: float = 0.2,
        # Curriculum settings
        use_curriculum: bool = True,
        curriculum_epochs: int = 10,  # Epochs before full loss
        # Adaptive weighting
        use_adaptive_weights: bool = True,
    ):
        super().__init__()

        self.base_lambda_ilm = lambda_ilm_anchor
        self.base_lambda_rpe = lambda_rpe_anchor
        self.base_lambda_edge = lambda_edge_align
        self.base_lambda_smooth = lambda_smoothness

        self.use_curriculum = use_curriculum
        self.curriculum_epochs = curriculum_epochs
        self.use_adaptive_weights = use_adaptive_weights

        # Current epoch for curriculum (updated externally)
        self.current_epoch = 0

        # Learnable log-variance for uncertainty weighting (Kendall et al.)
        if use_adaptive_weights:
            self.log_var_ilm = nn.Parameter(torch.zeros(1))
            self.log_var_rpe = nn.Parameter(torch.zeros(1))
            self.log_var_edge = nn.Parameter(torch.zeros(1))
            self.log_var_smooth = nn.Parameter(torch.zeros(1))

        # Edge detection kernels
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

        # Gaussian smoothing for robust gradient
        self.register_buffer('gaussian', self._make_gaussian_kernel(5, 1.0))

    def _make_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel for smoothing."""
        x = torch.arange(size) - size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        kernel = gauss.outer(gauss)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum learning."""
        self.current_epoch = epoch

    def get_curriculum_weight(self, loss_type: str) -> float:
        """
        Get curriculum weight for a loss type.

        Early epochs: focus on anchor losses (fundamental alignment)
        Later epochs: add edge alignment and smoothness (refinement)
        """
        if not self.use_curriculum:
            return 1.0

        progress = min(1.0, self.current_epoch / max(1, self.curriculum_epochs))

        if loss_type in ['ilm_anchor', 'rpe_anchor']:
            # Anchor losses: full weight from start
            return 1.0
        elif loss_type == 'edge_align':
            # Edge alignment: ramp up from 0.2 to 1.0
            return 0.2 + 0.8 * progress
        elif loss_type == 'smoothness':
            # Smoothness: ramp up from 0.1 to 1.0
            return 0.1 + 0.9 * progress
        else:
            return 1.0

    def detect_retina_band(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Robust retina detection (from V2) with improvements.
        """
        B, C, H, W = image.shape
        device = image.device

        # Smooth the image first
        padded = F.pad(image, (2, 2, 2, 2), mode='replicate')
        smoothed = F.conv2d(padded, self.gaussian.to(device), padding=0)

        intensity = smoothed[:, 0, :, :]  # [B, H, W]

        # Method 1: Adaptive intensity thresholding (per-column)
        # Use multiple thresholds and combine
        col_mean = intensity.mean(dim=1, keepdim=True)  # [B, 1, W]
        col_std = intensity.std(dim=1, keepdim=True)  # [B, 1, W]

        # Multiple threshold levels
        thresh_low = col_mean + 0.3 * col_std
        thresh_mid = col_mean + 0.5 * col_std
        thresh_high = col_mean + 0.7 * col_std

        # Combine masks with weighted voting
        mask_low = (intensity > thresh_low).float()
        mask_mid = (intensity > thresh_mid).float()
        mask_high = (intensity > thresh_high).float()

        combined_mask = (mask_low + mask_mid + mask_high) / 3  # Soft mask

        # Method 2: Gradient-based detection
        padded_grad = F.pad(image, (1, 1, 1, 1), mode='replicate')
        grad_y = F.conv2d(padded_grad, self.sobel_y.to(device), padding=0)[:, 0, :, :]

        # Positive gradient (dark-to-bright): potential ILM
        # Negative gradient (bright-to-dark): potential RPE
        grad_pos = F.relu(grad_y)
        grad_neg = F.relu(-grad_y)

        # Weight gradients by intensity (stronger signal in bright regions)
        grad_pos_weighted = grad_pos * intensity
        grad_neg_weighted = grad_neg * intensity

        # Find row indices
        row_idx = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

        # Soft argmax for ILM (top boundary)
        # Focus on upper half of image for ILM
        upper_mask = (row_idx < H * 0.6).float()
        top_score = (grad_pos_weighted + combined_mask * 0.5) * upper_mask
        top_weights = F.softmax(top_score * 20, dim=1)  # Sharp softmax
        ilm_soft = (top_weights * row_idx).sum(dim=1) / (H - 1)

        # Soft argmax for RPE (bottom boundary)
        # Focus on lower half
        lower_mask = (row_idx > H * 0.3).float()
        bottom_score = (grad_neg_weighted + combined_mask * 0.5) * lower_mask
        bottom_weights = F.softmax(bottom_score * 20, dim=1)
        rpe_soft = (bottom_weights * row_idx).sum(dim=1) / (H - 1)

        # Method 1 result: percentile from combined mask
        cumsum = combined_mask.cumsum(dim=1)
        total = cumsum[:, -1, :].clamp(min=0.1)

        target_10pct = total * 0.10
        target_90pct = total * 0.90

        above_10 = cumsum >= target_10pct.unsqueeze(1)
        above_90 = cumsum >= target_90pct.unsqueeze(1)

        ilm_pct = above_10.float().argmax(dim=1).float() / (H - 1)
        rpe_pct = above_90.float().argmax(dim=1).float() / (H - 1)

        # Ensemble: combine soft argmax and percentile methods
        # Weight by gradient strength (trust gradient more when strong)
        grad_strength = (grad_pos.max(dim=1).values + grad_neg.max(dim=1).values) / 2
        grad_strength_norm = grad_strength / (grad_strength.max() + 1e-8)
        alpha = torch.sigmoid(grad_strength_norm * 5 - 2)  # 0.2-0.8 range

        retina_top = alpha * ilm_soft + (1 - alpha) * ilm_pct
        retina_bottom = alpha * rpe_soft + (1 - alpha) * rpe_pct

        # Ensure valid ordering
        retina_bottom = torch.maximum(retina_bottom, retina_top + 0.15)
        retina_top = torch.clamp(retina_top, 0.05, 0.7)
        retina_bottom = torch.clamp(retina_bottom, 0.3, 0.95)

        # Confidence: based on band width, gradient strength, and consistency
        band_width = retina_bottom - retina_top
        width_conf = torch.exp(-((band_width - 0.35) ** 2) / 0.1)  # Peak at 35% width

        # Cross-column consistency
        top_std = retina_top.std(dim=1, keepdim=True)
        bottom_std = retina_bottom.std(dim=1, keepdim=True)
        consistency_conf = torch.exp(-(top_std + bottom_std) * 10)

        confidence = (width_conf + grad_strength_norm + consistency_conf.expand_as(width_conf)) / 3

        return retina_top, retina_bottom, confidence

    def ilm_anchor_loss(
        self,
        boundaries: torch.Tensor,
        retina_top: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """ILM anchor with confidence weighting and Huber loss for robustness."""
        ilm = boundaries[:, 0, :]
        error = ilm - retina_top

        # Huber loss (less sensitive to outliers)
        loss = F.huber_loss(ilm, retina_top, reduction='none', delta=0.05)

        # Weight by confidence
        weighted_loss = loss * confidence

        return weighted_loss.mean()

    def rpe_anchor_loss(
        self,
        boundaries: torch.Tensor,
        retina_bottom: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """RPE anchor with confidence weighting."""
        rpe = boundaries[:, -1, :]
        rpe_target = retina_bottom * 0.95

        loss = F.huber_loss(rpe, rpe_target, reduction='none', delta=0.05)
        weighted_loss = loss * confidence

        return weighted_loss.mean()

    def edge_alignment_loss(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Boundaries should align with intensity edges.
        Uses soft sampling for differentiability.
        """
        B, C, H, W = image.shape
        device = image.device

        # Compute edges
        padded = F.pad(image, (1, 1, 1, 1), mode='replicate')
        edges = torch.abs(F.conv2d(padded, self.sobel_y.to(device), padding=0))

        # Normalize edges
        edges = edges / (edges.max() + 1e-8)

        # For each boundary, sample edge strength using soft indexing
        boundaries_px = boundaries * (H - 1)  # [B, 4, W] in pixel coords

        total_edge_strength = torch.zeros(1, device=device)

        for i in range(boundaries.shape[1]):
            b_pos = boundaries_px[:, i, :].unsqueeze(2)  # [B, W, 1]

            # Create soft sampling weights (Gaussian around boundary position)
            row_idx = torch.arange(H, device=device, dtype=torch.float32)
            dist = (row_idx.view(1, 1, H) - b_pos) ** 2  # [B, W, H]
            weights = torch.exp(-dist / 2)  # sigma=1
            weights = weights / weights.sum(dim=2, keepdim=True)

            # Sample edge strength
            edges_per_col = edges[:, 0, :, :].permute(0, 2, 1)  # [B, W, H]
            edge_at_boundary = (edges_per_col * weights).sum(dim=2)  # [B, W]

            total_edge_strength = total_edge_strength + edge_at_boundary.mean()

        # Loss: want high edge strength (so return negative)
        return 1.0 - total_edge_strength / boundaries.shape[1]

    def smoothness_loss(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Encourage smooth boundaries with adaptive penalty.
        Penalize sudden jumps more than gradual changes.
        """
        # First derivative (velocity)
        diff1 = boundaries[:, :, 1:] - boundaries[:, :, :-1]

        # Second derivative (acceleration) - penalize sudden changes
        diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]

        # Combine: L1 on velocity + L2 on acceleration
        velocity_loss = torch.abs(diff1).mean()
        accel_loss = (diff2 ** 2).mean()

        return velocity_loss + 0.5 * accel_loss

    def forward(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all losses with adaptive weighting."""
        device = boundaries.device

        # Detect retina
        retina_top, retina_bottom, confidence = self.detect_retina_band(image)

        # Compute individual losses
        ilm_loss = self.ilm_anchor_loss(boundaries, retina_top, confidence)
        rpe_loss = self.rpe_anchor_loss(boundaries, retina_bottom, confidence)
        edge_loss = self.edge_alignment_loss(boundaries, image)
        smooth_loss = self.smoothness_loss(boundaries)

        # Get curriculum weights
        w_ilm_curr = self.get_curriculum_weight('ilm_anchor')
        w_rpe_curr = self.get_curriculum_weight('rpe_anchor')
        w_edge_curr = self.get_curriculum_weight('edge_align')
        w_smooth_curr = self.get_curriculum_weight('smoothness')

        # Adaptive weighting using learned uncertainty
        if self.use_adaptive_weights:
            # Kendall et al. multi-task learning with uncertainty
            precision_ilm = torch.exp(-self.log_var_ilm)
            precision_rpe = torch.exp(-self.log_var_rpe)
            precision_edge = torch.exp(-self.log_var_edge)
            precision_smooth = torch.exp(-self.log_var_smooth)

            total = (
                precision_ilm * self.base_lambda_ilm * w_ilm_curr * ilm_loss + self.log_var_ilm +
                precision_rpe * self.base_lambda_rpe * w_rpe_curr * rpe_loss + self.log_var_rpe +
                precision_edge * self.base_lambda_edge * w_edge_curr * edge_loss + self.log_var_edge +
                precision_smooth * self.base_lambda_smooth * w_smooth_curr * smooth_loss + self.log_var_smooth
            )
        else:
            total = (
                self.base_lambda_ilm * w_ilm_curr * ilm_loss +
                self.base_lambda_rpe * w_rpe_curr * rpe_loss +
                self.base_lambda_edge * w_edge_curr * edge_loss +
                self.base_lambda_smooth * w_smooth_curr * smooth_loss
            )

        losses = {
            'ilm_anchor': ilm_loss.item(),
            'rpe_anchor': rpe_loss.item(),
            'edge_align': edge_loss.item(),
            'smoothness': smooth_loss.item(),
            'total': total.item() if torch.is_tensor(total) else total,
            'detected_retina_top': retina_top.mean().item(),
            'detected_retina_bottom': retina_bottom.mean().item(),
            'confidence': confidence.mean().item(),
            'curriculum_progress': min(1.0, self.current_epoch / max(1, self.curriculum_epochs)),
        }

        if self.use_adaptive_weights:
            losses['w_ilm'] = precision_ilm.item()
            losses['w_rpe'] = precision_rpe.item()
            losses['w_edge'] = precision_edge.item()
            losses['w_smooth'] = precision_smooth.item()

        return total, losses


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing IntensityAnchoredBoundaryLossV3")
    print("=" * 60)

    # Test data
    B, C, H, W = 2, 1, 128, 128
    image = torch.zeros(B, C, H, W)
    for b in range(B):
        image[b, 0, :, :] = torch.rand(H, W) * 0.1
        image[b, 0, 40:90, :] = 0.5 + torch.rand(50, W) * 0.3

    boundaries = torch.zeros(B, 4, W)
    boundaries[:, 0, :] = 0.33
    boundaries[:, 1, :] = 0.40
    boundaries[:, 2, :] = 0.50
    boundaries[:, 3, :] = 0.60

    loss_fn = IntensityAnchoredBoundaryLossV3()

    # Test at different curriculum stages
    for epoch in [0, 5, 10, 15]:
        loss_fn.set_epoch(epoch)
        total_loss, loss_dict = loss_fn(boundaries, image)
        print(f"\nEpoch {epoch} (progress={loss_dict['curriculum_progress']:.1%}):")
        print(f"  Total loss: {loss_dict['total']:.4f}")
        print(f"  ILM anchor: {loss_dict['ilm_anchor']:.4f}")
        print(f"  RPE anchor: {loss_dict['rpe_anchor']:.4f}")
        print(f"  Edge align: {loss_dict['edge_align']:.4f}")
        print(f"  Smoothness: {loss_dict['smoothness']:.4f}")

    print("\nV3 features:")
    print("  - Robust detection from V2 (ensemble)")
    print("  - Simpler loss (4 terms vs 8)")
    print("  - Curriculum learning (anchors first)")
    print("  - Adaptive weights (uncertainty-based)")
    print("  - Huber loss for robustness")
