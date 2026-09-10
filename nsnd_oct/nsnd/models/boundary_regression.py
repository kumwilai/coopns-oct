#!/usr/bin/env python3
"""
Direct Boundary Regression for OCT Layer Analysis

KEY TMI CONTRIBUTION: Direct regression of boundary positions instead of
segmentation-based detection.

Why this is better than segmentation:
1. IS/OS is only 5-10 pixels thick - hard to segment accurately
2. Segmentation → boundary extraction adds error
3. Direct regression is a simpler problem (1D per column)
4. Can achieve sub-pixel accuracy with proper loss

Boundaries predicted (from top to bottom):
- ILM: Inner Limiting Membrane (top of retina)
- RNFL_GCL: Bottom of RNFL/GCL complex
- INL_TOP: Top of INL (bottom of IPL)
- OPL_ONL: OPL/ONL boundary
- ELM: External Limiting Membrane
- IS_OS: Inner/Outer Segment junction (CRITICAL for clinical use)
- OS_RPE: Top of RPE
- BM: Bruch's Membrane (bottom of RPE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


# Boundary names for 4-class compatible version
BOUNDARY_NAMES_4CLASS = [
    'RNFL_GCL_top',      # Top of RNFL_GCL (ILM)
    'RNFL_GCL_bottom',   # Bottom of RNFL_GCL
    'INL_OPL_ONL_bottom', # Bottom of INL_OPL_ONL (top of IS_OS)
    'IS_OS_bottom',      # Bottom of IS_OS (top of RPE) - CRITICAL
    'RPE_Choroid_bottom', # Bottom of RPE_Choroid
]

# Clinical importance weights
BOUNDARY_WEIGHTS = {
    'RNFL_GCL_top': 1.0,       # ILM
    'RNFL_GCL_bottom': 2.0,    # Important for RNFL thickness (glaucoma)
    'INL_OPL_ONL_bottom': 1.5, # ELM region
    'IS_OS_bottom': 3.0,       # CRITICAL - IS/OS junction
    'RPE_Choroid_bottom': 1.5, # RPE bottom
}


class BoundaryRegressionHead(nn.Module):
    """
    Direct boundary position regression from columnar features.

    Takes columnar features [B, W, H, C] and predicts boundary positions
    [B, W, num_boundaries] as normalized coordinates (0-1).
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        num_boundaries: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries

        # Per-column MLP for boundary prediction
        self.column_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention pooling over height dimension
        self.height_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Boundary position predictor
        self.boundary_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_boundaries),
            nn.Sigmoid(),  # Output normalized positions (0-1)
        )

        # Uncertainty estimation (optional, for confidence)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, num_boundaries),
            nn.Softplus(),  # Positive uncertainty
        )

    def forward(
        self, col_features: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            col_features: [B, W, H, C] columnar features
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            boundaries: [B, W, num_boundaries] normalized positions (0-1)
            uncertainty: [B, W, num_boundaries] uncertainty estimates (optional)
        """
        B, W, H, C = col_features.shape

        # Encode each column: [B, W, H, hidden]
        col_encoded = self.column_encoder(col_features)

        # Attention pooling over height: [B, W, H, 1]
        attn_weights = self.height_attention(col_encoded)
        attn_weights = F.softmax(attn_weights, dim=2)

        # Weighted sum: [B, W, hidden]
        col_pooled = (col_encoded * attn_weights).sum(dim=2)

        # Predict boundary positions: [B, W, num_boundaries]
        boundaries = self.boundary_predictor(col_pooled)

        # Enforce ordering constraint in forward pass
        # Boundaries should be in ascending order (top to bottom)
        boundaries = self._enforce_ordering(boundaries)

        uncertainty = None
        if return_uncertainty:
            uncertainty = self.uncertainty_head(col_pooled)

        return boundaries, uncertainty

    def _enforce_ordering(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Enforce that boundaries are in ascending order (top to bottom).

        Uses cumulative softmax to ensure ordering while remaining differentiable.
        """
        B, W, N = boundaries.shape

        # Convert to differences (should all be positive)
        # boundaries[i] = sum(deltas[0:i+1])
        deltas = torch.zeros_like(boundaries)
        deltas[:, :, 0] = boundaries[:, :, 0]
        deltas[:, :, 1:] = boundaries[:, :, 1:] - boundaries[:, :, :-1]

        # Ensure positive deltas (minimum gap)
        min_gap = 0.01  # Minimum 1% of image height between boundaries
        deltas = F.softplus(deltas) + min_gap

        # Reconstruct ordered boundaries
        ordered = torch.cumsum(deltas, dim=2)

        # Normalize to [0, 1]
        ordered = ordered / (ordered[:, :, -1:] + 1e-8)

        # Clamp to valid range
        ordered = torch.clamp(ordered, 0.01, 0.99)

        return ordered


class BoundaryRefiner(nn.Module):
    """
    Refines boundary predictions using spatial context and anatomical constraints.

    Applies:
    1. Horizontal smoothing (boundaries should be continuous)
    2. Thickness constraints (layers have known thickness ranges)
    3. Cross-column consistency
    """

    def __init__(
        self,
        num_boundaries: int = 5,
        hidden_dim: int = 64,
        kernel_size: int = 11,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries

        # Learnable 1D convolution for horizontal smoothing per boundary
        self.smoothers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim, 1, kernel_size, padding=kernel_size // 2),
            )
            for _ in range(num_boundaries)
        ])

        # Cross-boundary refinement
        self.cross_refiner = nn.Sequential(
            nn.Conv1d(num_boundaries, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, num_boundaries, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundaries: [B, W, num_boundaries]
        Returns:
            refined: [B, W, num_boundaries]
        """
        B, W, N = boundaries.shape

        # Apply per-boundary smoothing
        smoothed = []
        for i, smoother in enumerate(self.smoothers):
            b_i = boundaries[:, :, i:i+1].transpose(1, 2)  # [B, 1, W]
            s_i = smoother(b_i) + b_i  # Residual
            smoothed.append(s_i.transpose(1, 2))  # [B, W, 1]

        smoothed = torch.cat(smoothed, dim=2)  # [B, W, N]

        # Cross-boundary refinement
        refined = smoothed.transpose(1, 2)  # [B, N, W]
        refined = self.cross_refiner(refined) + refined
        refined = refined.transpose(1, 2)  # [B, W, N]

        # Re-enforce ordering
        refined = self._enforce_ordering(refined)

        return refined

    def _enforce_ordering(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Same ordering enforcement as BoundaryRegressionHead."""
        B, W, N = boundaries.shape

        deltas = torch.zeros_like(boundaries)
        deltas[:, :, 0] = boundaries[:, :, 0]
        deltas[:, :, 1:] = boundaries[:, :, 1:] - boundaries[:, :, :-1]

        min_gap = 0.01
        deltas = F.relu(deltas) + min_gap

        ordered = torch.cumsum(deltas, dim=2)
        ordered = ordered / (ordered[:, :, -1:] + 1e-8)
        ordered = torch.clamp(ordered, 0.01, 0.99)

        return ordered


class BoundaryLoss(nn.Module):
    """
    Loss functions for boundary regression.

    Combines:
    1. L1/L2 position loss
    2. Smoothness loss (penalize discontinuities)
    3. Ordering loss (ensure correct order)
    4. Thickness loss (layer thickness constraints)
    """

    def __init__(
        self,
        num_boundaries: int = 5,
        lambda_smooth: float = 0.5,
        lambda_order: float = 0.1,
        lambda_thickness: float = 0.1,
        boundary_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.lambda_smooth = lambda_smooth
        self.lambda_order = lambda_order
        self.lambda_thickness = lambda_thickness

        # Per-boundary weights
        if boundary_weights is None:
            boundary_weights = BOUNDARY_WEIGHTS
        self.boundary_weights = boundary_weights

        # Create weight tensor
        weights = torch.ones(num_boundaries)
        for i, name in enumerate(BOUNDARY_NAMES_4CLASS[:num_boundaries]):
            weights[i] = boundary_weights.get(name, 1.0)
        self.register_buffer('weights', weights)

        # Thickness constraints (as fraction of image height)
        # [min_thickness, max_thickness] for each layer (between boundaries)
        self.register_buffer('thickness_min', torch.tensor([
            0.05,  # RNFL_GCL: 5% min
            0.10,  # INL_OPL_ONL: 10% min
            0.02,  # IS_OS: 2% min
            0.05,  # RPE_Choroid: 5% min
        ]))
        self.register_buffer('thickness_max', torch.tensor([
            0.25,  # RNFL_GCL: 25% max
            0.40,  # INL_OPL_ONL: 40% max
            0.10,  # IS_OS: 10% max
            0.30,  # RPE_Choroid: 30% max
        ]))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: [B, W, num_boundaries] predicted boundaries (normalized 0-1)
            target: [B, W, num_boundaries] target boundaries (normalized 0-1)
            mask: [B, W] valid column mask (optional)

        Returns:
            total_loss: Combined loss
            stats: Dictionary of individual loss components
        """
        B, W, N = pred.shape
        device = pred.device

        if mask is None:
            mask = torch.ones(B, W, device=device)

        # Expand mask for boundaries
        mask_expanded = mask.unsqueeze(-1)  # [B, W, 1]

        # 1. Position loss (weighted L1)
        pos_error = torch.abs(pred - target)  # [B, W, N]
        pos_error = pos_error * self.weights.view(1, 1, -1)  # Weight by importance
        pos_loss = (pos_error * mask_expanded).sum() / (mask_expanded.sum() * N + 1e-8)

        # 2. Smoothness loss (penalize horizontal discontinuities)
        pred_grad = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])  # [B, W-1, N]
        target_grad = torch.abs(target[:, 1:, :] - target[:, :-1, :])
        smooth_loss = F.l1_loss(pred_grad, target_grad, reduction='mean')

        # 3. Ordering loss (boundaries should be in ascending order)
        deltas = pred[:, :, 1:] - pred[:, :, :-1]  # [B, W, N-1]
        order_violations = F.relu(-deltas)  # Penalize negative deltas
        order_loss = order_violations.mean()

        # 4. Thickness loss (layer thicknesses within expected range)
        pred_thickness = pred[:, :, 1:] - pred[:, :, :-1]  # [B, W, N-1]
        n_layers = min(len(self.thickness_min), N - 1)

        thickness_loss = torch.tensor(0.0, device=device)
        for i in range(n_layers):
            t = pred_thickness[:, :, i]
            too_thin = F.relu(self.thickness_min[i] - t)
            too_thick = F.relu(t - self.thickness_max[i])
            thickness_loss = thickness_loss + (too_thin.mean() + too_thick.mean())
        thickness_loss = thickness_loss / n_layers

        # Combined loss
        total_loss = (
            pos_loss +
            self.lambda_smooth * smooth_loss +
            self.lambda_order * order_loss +
            self.lambda_thickness * thickness_loss
        )

        stats = {
            'boundary_pos_loss': pos_loss.item(),
            'boundary_smooth_loss': smooth_loss.item(),
            'boundary_order_loss': order_loss.item(),
            'boundary_thickness_loss': thickness_loss.item(),
            'boundary_total_loss': total_loss.item(),
        }

        # Per-boundary MAE
        for i, name in enumerate(BOUNDARY_NAMES_4CLASS[:N]):
            mae = (torch.abs(pred[:, :, i] - target[:, :, i]) * mask).sum() / (mask.sum() + 1e-8)
            stats[f'{name}_mae'] = mae.item()

        return total_loss, stats


def extract_boundaries_from_segmentation(
    seg_mask: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Extract boundary positions from segmentation mask.

    For training, we derive target boundaries from segmentation ground truth.

    Args:
        seg_mask: [B, H, W] segmentation mask with class indices
        num_classes: Number of classes

    Returns:
        boundaries: [B, W, num_classes+1] normalized boundary positions
    """
    B, H, W = seg_mask.shape
    device = seg_mask.device

    # For 4-class segmentation, we extract 5 boundaries:
    # Top of class 0, boundary 0-1, boundary 1-2, boundary 2-3, bottom of class 3
    num_boundaries = num_classes + 1
    boundaries = torch.zeros(B, W, num_boundaries, device=device)

    for b in range(B):
        for w in range(W):
            col = seg_mask[b, :, w]  # [H]

            # Find transitions
            current_class = -1
            for h in range(H):
                c = col[h].item()
                if c != current_class:
                    if c >= 0 and c < num_classes:
                        # This is the top of class c
                        if c == 0:
                            boundaries[b, w, 0] = h / H  # Top of first class
                        # Also record as bottom of previous class
                        if current_class >= 0:
                            boundaries[b, w, current_class + 1] = h / H
                    current_class = c

            # Bottom of last class
            if current_class >= 0 and current_class < num_classes:
                boundaries[b, w, current_class + 1] = 1.0

    return boundaries


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    print("Testing Boundary Regression...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test boundary regression head
    B, W, H, C = 2, 512, 256, 128
    col_features = torch.randn(B, W, H, C).to(device)

    head = BoundaryRegressionHead(in_dim=C, num_boundaries=5).to(device)
    boundaries, uncertainty = head(col_features, return_uncertainty=True)

    print(f"Columnar features: {col_features.shape}")
    print(f"Boundaries: {boundaries.shape}")
    print(f"Uncertainty: {uncertainty.shape}")
    print(f"Boundary range: [{boundaries.min():.3f}, {boundaries.max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in head.parameters()):,}")

    # Test refiner
    refiner = BoundaryRefiner(num_boundaries=5).to(device)
    refined = refiner(boundaries)
    print(f"Refined boundaries: {refined.shape}")

    # Test loss
    target = torch.rand(B, W, 5).to(device).sort(dim=-1)[0]  # Sorted random
    loss_fn = BoundaryLoss(num_boundaries=5)
    loss, stats = loss_fn(boundaries, target)
    print(f"\nLoss: {loss.item():.4f}")
    print("Stats:", {k: f"{v:.4f}" for k, v in stats.items()})

    # Test boundary extraction from segmentation
    seg_mask = torch.randint(0, 4, (B, 256, 512)).to(device)
    extracted = extract_boundaries_from_segmentation(seg_mask, num_classes=4)
    print(f"\nExtracted boundaries from seg: {extracted.shape}")

    print("\nAll tests passed!")
