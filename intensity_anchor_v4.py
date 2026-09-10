#!/usr/bin/env python3
"""
IntensityAnchoredBoundaryLoss V4 - Best of All Worlds

Lessons learned:
- V1: Simple loss = good learning, but poor detection
- V2: Rich detection = best initial, but over-constrains
- V3: Curriculum helps but adaptive weights were unstable

V4 Strategy:
1. V2's robust ensemble detection (best initial accuracy)
2. V1's simple loss structure (best learning)
3. Confidence weighting (trust good detections more)
4. Only 2 additional constraints: smoothness + gradient strength
5. Fixed (not learned) weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class IntensityAnchoredBoundaryLossV4(nn.Module):
    """
    V4: Robust detection + Simple learning + Targeted constraints
    """

    def __init__(
        self,
        lambda_ilm_anchor: float = 1.0,
        lambda_rpe_anchor: float = 1.0,
        lambda_edge_align: float = 0.3,  # Reduced from V1
        lambda_smoothness: float = 0.1,  # Light smoothness
        use_confidence: bool = True,
    ):
        super().__init__()

        self.lambda_ilm = lambda_ilm_anchor
        self.lambda_rpe = lambda_rpe_anchor
        self.lambda_edge = lambda_edge_align
        self.lambda_smooth = lambda_smoothness
        self.use_confidence = use_confidence

        # Sobel filter
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

        # Gaussian for smoothing
        self.register_buffer('gaussian', self._gaussian_kernel(5, 1.0))

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(size) - size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        kernel = gauss.outer(gauss)
        return (kernel / kernel.sum()).view(1, 1, size, size)

    def detect_retina_band(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Robust detection from V2 (ensemble of intensity + gradient methods).
        MEMORY-OPTIMIZED: Aggressive cleanup of intermediate tensors.
        """
        B, C, H, W = image.shape
        device = image.device

        # Use registered buffers directly (no .to(device) needed - buffers auto-move)
        gaussian = self.gaussian
        sobel_y = self.sobel_y

        # Smooth image
        padded = F.pad(image, (2, 2, 2, 2), mode='replicate')
        smoothed = F.conv2d(padded, gaussian, padding=0)
        del padded  # Free memory
        intensity = smoothed[:, 0, :, :]  # [B, H, W]
        del smoothed  # Free memory

        # === Method 1: Multi-threshold intensity ===
        col_mean = intensity.mean(dim=1, keepdim=True)
        col_std = intensity.std(dim=1, keepdim=True).clamp(min=1e-6)

        # Combine multiple thresholds (compute in-place to save memory)
        combined_mask = (intensity > col_mean + 0.3 * col_std).float()
        combined_mask = combined_mask + (intensity > col_mean + 0.5 * col_std).float()
        combined_mask = combined_mask + (intensity > col_mean + 0.7 * col_std).float()
        combined_mask = combined_mask / 3
        del col_mean, col_std  # Free memory

        # Percentile detection
        cumsum = combined_mask.cumsum(dim=1)
        total = cumsum[:, -1, :].clamp(min=0.1)

        target_10 = (total * 0.10).unsqueeze(1)
        target_90 = (total * 0.90).unsqueeze(1)

        ilm_intensity = (cumsum >= target_10).float().argmax(dim=1).float() / (H - 1)
        rpe_intensity = (cumsum >= target_90).float().argmax(dim=1).float() / (H - 1)
        del cumsum, target_10, target_90, total  # Free memory

        # === Method 2: Gradient-based ===
        padded_grad = F.pad(image, (1, 1, 1, 1), mode='replicate')
        grad_y = F.conv2d(padded_grad, sobel_y, padding=0)[:, 0, :, :]
        del padded_grad  # Free memory

        grad_pos = F.relu(grad_y) * intensity  # Dark-to-bright (ILM)
        grad_neg = F.relu(-grad_y) * intensity  # Bright-to-dark (RPE)
        del grad_y  # Free memory

        # Pre-compute row indices once
        row_idx = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

        # Soft argmax with focused regions
        upper_mask = (row_idx < H * 0.6).float()
        lower_mask = (row_idx > H * 0.3).float()

        top_score = (grad_pos + combined_mask * 0.3) * upper_mask
        top_weights = F.softmax(top_score * 15, dim=1)
        ilm_gradient = (top_weights * row_idx).sum(dim=1) / (H - 1)
        del top_score, top_weights, upper_mask  # Free memory

        bottom_score = (grad_neg + combined_mask * 0.3) * lower_mask
        bottom_weights = F.softmax(bottom_score * 15, dim=1)
        rpe_gradient = (bottom_weights * row_idx).sum(dim=1) / (H - 1)
        del bottom_score, bottom_weights, lower_mask, row_idx, combined_mask  # Free memory

        # === Ensemble ===
        # BUG FIX: Use proper dimension for max to handle batches correctly
        grad_strength = (grad_pos.amax(dim=(1, 2), keepdim=True) +
                         grad_neg.amax(dim=(1, 2), keepdim=True)).squeeze(-1)  # [B, W]
        del grad_pos, grad_neg, intensity  # Free memory

        # Normalize per-batch (not global)
        grad_max = grad_strength.amax(dim=1, keepdim=True).clamp(min=1e-8)
        grad_strength_norm = grad_strength / grad_max
        del grad_strength, grad_max  # Free memory
        alpha = torch.sigmoid(grad_strength_norm * 3 - 1)  # 0.3-0.7 range

        retina_top = alpha * ilm_gradient + (1 - alpha) * ilm_intensity
        retina_bottom = alpha * rpe_gradient + (1 - alpha) * rpe_intensity
        del ilm_gradient, ilm_intensity, rpe_gradient, rpe_intensity  # Free memory

        # Enforce constraints (in-place where safe)
        retina_bottom = torch.maximum(retina_bottom, retina_top + 0.15)
        retina_top = retina_top.clamp(0.05, 0.60)
        retina_bottom = retina_bottom.clamp(0.30, 0.95)

        # Confidence: band width + gradient strength + consistency
        band_width = retina_bottom - retina_top
        width_conf = torch.exp(-((band_width - 0.35) ** 2) / 0.05)
        del band_width  # Free memory

        # Compute consistency (avoid extra allocations)
        top_std = retina_top.std(dim=1, keepdim=True)
        bottom_std = retina_bottom.std(dim=1, keepdim=True)
        consistency = 1.0 / (1.0 + (top_std + bottom_std) * 5)
        del top_std, bottom_std  # Free memory

        confidence = (width_conf + grad_strength_norm + consistency.expand_as(width_conf)) / 3
        del width_conf, grad_strength_norm, consistency, alpha  # Free memory

        return retina_top, retina_bottom, confidence

    def anchor_loss(
        self,
        boundary: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        tolerance: float = 0.02,
    ) -> torch.Tensor:
        """
        Anchor loss with tolerance band and confidence weighting.
        Uses smooth L1 (Huber) for robustness.
        """
        error = torch.abs(boundary - target)
        loss = F.relu(error - tolerance)  # No penalty within tolerance

        if self.use_confidence:
            loss = loss * confidence

        return loss.mean()

    def edge_alignment_loss(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified edge alignment using soft sampling.
        MEMORY-OPTIMIZED: Process boundaries one at a time to avoid large [B, 4, W, H] tensors.
        """
        B, C, H, W = image.shape
        device = image.device
        num_boundaries = boundaries.shape[1]

        # Compute edges (use registered buffer directly)
        padded = F.pad(image, (1, 1, 1, 1), mode='replicate')
        edges = torch.abs(F.conv2d(padded, self.sobel_y, padding=0))
        # Normalize per-image for stability
        edges_max = edges.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        edges = edges / edges_max  # [B, 1, H, W]

        # edges: [B, 1, H, W] -> [B, W, H] for sampling
        edges_col = edges[:, 0, :, :].permute(0, 2, 1)  # [B, W, H]

        # row_idx: [1, 1, H]
        row_idx = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H)

        # Process boundaries ONE AT A TIME to save memory
        # Instead of creating [B, 4, W, H], we create [B, W, H] per boundary
        total_edge_strength = 0.0
        for i in range(num_boundaries):
            # boundaries[:, i, :] is [B, W] -> [B, W, 1]
            boundary_px = (boundaries[:, i, :] * (H - 1)).unsqueeze(-1)  # [B, W, 1]

            # Compute distances: [B, W, H]
            dist_sq = (row_idx - boundary_px) ** 2

            # Soft weights with sigma=1
            weights = torch.exp(-dist_sq)  # [B, W, H]
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Sample edges at boundary position: [B, W, H] * [B, W, H] -> sum over H -> [B, W]
            edge_at_boundary = (weights * edges_col).sum(dim=-1)  # [B, W]
            total_edge_strength = total_edge_strength + edge_at_boundary.mean()

            # Free memory
            del dist_sq, weights, edge_at_boundary

        # Average over all boundaries
        return 1.0 - total_edge_strength / num_boundaries

    def smoothness_loss(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Light total variation on boundaries."""
        diff = boundaries[:, :, 1:] - boundaries[:, :, :-1]
        return torch.abs(diff).mean()

    def forward(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute losses.
        """
        # Detect retina
        retina_top, retina_bottom, confidence = self.detect_retina_band(image)

        # Compute losses
        ilm = boundaries[:, 0, :]
        rpe = boundaries[:, -1, :]
        rpe_target = retina_bottom * 0.95

        ilm_loss = self.anchor_loss(ilm, retina_top, confidence)
        rpe_loss = self.anchor_loss(rpe, rpe_target, confidence)
        edge_loss = self.edge_alignment_loss(boundaries, image)
        smooth_loss = self.smoothness_loss(boundaries)

        # Total loss (simple weighted sum)
        total = (
            self.lambda_ilm * ilm_loss +
            self.lambda_rpe * rpe_loss +
            self.lambda_edge * edge_loss +
            self.lambda_smooth * smooth_loss
        )

        losses = {
            'ilm_anchor': ilm_loss.item(),
            'rpe_anchor': rpe_loss.item(),
            'edge_align': edge_loss.item(),
            'smoothness': smooth_loss.item(),
            'total': total.item(),
            'detected_retina_top': retina_top.mean().item(),
            'detected_retina_bottom': retina_bottom.mean().item(),
            'confidence': confidence.mean().item(),
        }

        return total, losses


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing IntensityAnchoredBoundaryLossV4")
    print("=" * 50)

    B, H, W = 2, 128, 128
    image = torch.zeros(B, 1, H, W)
    for b in range(B):
        image[b, 0] = torch.rand(H, W) * 0.1
        image[b, 0, 35:85] = 0.5 + torch.rand(50, W) * 0.3

    boundaries = torch.zeros(B, 4, W)
    boundaries[:, 0] = 0.33
    boundaries[:, 1] = 0.40
    boundaries[:, 2] = 0.50
    boundaries[:, 3] = 0.60

    loss_fn = IntensityAnchoredBoundaryLossV4()

    import time
    start = time.time()
    total, losses = loss_fn(boundaries, image)
    print(f"\nTime: {time.time() - start:.3f}s")
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")

    print("\nV4 features:")
    print("  - V2's robust ensemble detection")
    print("  - V1's simple loss structure")
    print("  - Confidence weighting")
    print("  - Minimal constraints (smooth + edge)")
    print("  - Fixed weights (no instability)")
