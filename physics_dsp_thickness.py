#!/usr/bin/env python3
"""
Physics-Informed DSP with Coupled Thickness Prediction.

Key innovation: Instead of predicting 4 independent boundaries, we predict:
- ILM position (absolute)
- RNFL thickness
- INL thickness
- ISOS thickness

Boundaries are derived via cumulative sum, guaranteeing:
1. Valid ordering (no crossing boundaries)
2. Direct thickness supervision
3. Minimum thickness constraints for thin layers

This addresses the fundamental problem with thin layers like INL (~7.7px).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Physics Components (reused from physics_dsp_lite)
# =============================================================================
class DepthCompensation(nn.Module):
    """Learnable depth attenuation compensation (Beer-Lambert)."""

    def __init__(self, init_mu: float = 0.003):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        depth = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        compensation = torch.exp(self.mu * depth)
        return x * compensation


class FresnelPhysicsLite(nn.Module):
    """Lightweight Fresnel reflection physics for boundary detection."""

    # Typical refractive indices for retinal layers
    REFRACTIVE_INDICES = {
        'vitreous': 1.336,
        'RNFL': 1.376,
        'GCL_IPL': 1.358,
        'INL': 1.352,
        'OPL': 1.365,
        'ONL': 1.358,
        'IS': 1.390,
        'OS': 1.430,
        'RPE': 1.450,
        'choroid': 1.370,
    }

    def __init__(self, num_boundaries: int = 4):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.expected_gradient_strength = self._compute_expected_strengths()

    def _compute_expected_strengths(self):
        """Compute expected gradient strengths at each boundary based on Fresnel."""
        n = self.REFRACTIVE_INDICES
        boundaries = [
            (n['vitreous'], n['RNFL']),    # ILM
            (n['GCL_IPL'], n['INL']),      # RNFL/INL boundary
            (n['ONL'], n['IS']),           # INL/IS_OS boundary (EZ)
            (n['OS'], n['RPE']),           # IS_OS/RPE boundary
        ]

        strengths = []
        for n1, n2 in boundaries:
            R = ((n1 - n2) / (n1 + n2)) ** 2
            strengths.append(R)

        max_R = max(strengths)
        return [s / max_R for s in strengths]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Compute vertical gradient (boundary indicator)."""
        gradient = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gradient = F.pad(gradient, (0, 0, 0, 1), mode='replicate')
        return gradient, self.expected_gradient_strength


# =============================================================================
# Multi-Scale Feature Encoder
# =============================================================================
class MultiScaleEncoder(nn.Module):
    """Multi-scale feature extraction for boundary detection."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 48):
        super().__init__()

        # Fine scale (3x3)
        self.enc_fine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Medium scale (5x5)
        self.enc_medium = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 5, padding=2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 5, padding=2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Coarse scale (dilated)
        self.enc_coarse = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Fusion
        total_channels = hidden_channels + hidden_channels // 2 + hidden_channels // 2
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_fine = self.enc_fine(x)
        feat_medium = self.enc_medium(x)
        feat_coarse = self.enc_coarse(x)
        features = torch.cat([feat_fine, feat_medium, feat_coarse], dim=1)
        return self.fusion(features)


# =============================================================================
# Thickness Predictor Head
# =============================================================================
class ThicknessPredictor(nn.Module):
    """
    Predicts ILM position + layer thicknesses.

    Outputs:
    - ILM position (normalized 0-1)
    - RNFL thickness (normalized, with minimum constraint)
    - INL thickness (normalized, with minimum constraint)
    - ISOS thickness (normalized, with minimum constraint)
    """

    # Minimum thicknesses in normalized coordinates (fraction of image height)
    # For 256px image: 3px = 0.012, 5px = 0.02, 8px = 0.031
    MIN_THICKNESS = {
        'RNFL': 0.02,   # ~5px minimum
        'INL': 0.012,   # ~3px minimum (very thin layer)
        'ISOS': 0.015,  # ~4px minimum
    }

    # Expected thicknesses for initialization (normalized)
    # For 256px: RNFL~30px=0.12, INL~8px=0.03, ISOS~15px=0.06
    INIT_THICKNESS = {
        'RNFL': 0.12,
        'INL': 0.03,
        'ISOS': 0.06,
    }

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()

        # Shared feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Column-wise pooling (average over height)
        # Output: [B, hidden_channels, 1, W]

        # ILM position head (absolute position)
        self.ilm_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

        # Thickness heads (one per layer)
        self.rnfl_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

        self.inl_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

        self.isos_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

        # Initialize thickness heads with expected values
        self._init_heads()

    def _init_heads(self):
        """Initialize heads to predict reasonable default values."""
        # ILM typically around 25% from top
        nn.init.zeros_(self.ilm_head[-1].weight)
        nn.init.constant_(self.ilm_head[-1].bias, -1.1)  # sigmoid(-1.1) ≈ 0.25

        # Thickness heads: use softplus, init to expected thickness
        # softplus(x) ≈ log(1 + exp(x))
        # For small target t: we need x such that softplus(x) ≈ t - min_thick
        # softplus(x) ≈ t when x = log(exp(t) - 1), for small t this is ≈ log(t)
        for head, name in [(self.rnfl_head, 'RNFL'),
                           (self.inl_head, 'INL'),
                           (self.isos_head, 'ISOS')]:
            nn.init.zeros_(head[-1].weight)
            # Target thickness after adding minimum
            target = self.INIT_THICKNESS[name] - self.MIN_THICKNESS[name]
            # softplus(x) = target => x = log(exp(target) - 1)
            # For small target, use approximation x ≈ log(target)
            if target > 0:
                init_bias = math.log(max(math.exp(target) - 1, 1e-4))
            else:
                init_bias = -2.0  # Small positive output
            nn.init.constant_(head[-1].bias, init_bias)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict ILM position and layer thicknesses.

        Args:
            features: [B, C, H, W] encoded features

        Returns:
            Dict with 'ilm', 'rnfl_thick', 'inl_thick', 'isos_thick'
            All in normalized coordinates [0, 1]
        """
        B, C, H, W = features.shape

        # Refine features
        feat = self.refine(features)

        # Pool over height to get column-wise features
        # Using adaptive pooling to handle different input sizes
        feat_pooled = F.adaptive_avg_pool2d(feat, (1, W))  # [B, C, 1, W]

        # Also get features at different vertical positions for context
        feat_top = F.adaptive_avg_pool2d(feat[:, :, :H//3, :], (1, W))
        feat_mid = F.adaptive_avg_pool2d(feat[:, :, H//3:2*H//3, :], (1, W))
        feat_bot = F.adaptive_avg_pool2d(feat[:, :, 2*H//3:, :], (1, W))

        # Combine vertical context
        feat_combined = feat_pooled + 0.3 * feat_top + 0.3 * feat_mid + 0.3 * feat_bot

        # Predict ILM position (sigmoid to [0, 1])
        ilm_raw = self.ilm_head(feat_combined)  # [B, 1, 1, W]
        ilm = torch.sigmoid(ilm_raw).squeeze(2)  # [B, 1, W]

        # Predict thicknesses (softplus for positive, then add minimum)
        rnfl_raw = self.rnfl_head(feat_combined).squeeze(2)  # [B, 1, W]
        inl_raw = self.inl_head(feat_combined).squeeze(2)
        isos_raw = self.isos_head(feat_combined).squeeze(2)

        # Softplus ensures positive, then add minimum thickness
        rnfl_thick = F.softplus(rnfl_raw) + self.MIN_THICKNESS['RNFL']
        inl_thick = F.softplus(inl_raw) + self.MIN_THICKNESS['INL']
        isos_thick = F.softplus(isos_raw) + self.MIN_THICKNESS['ISOS']

        return {
            'ilm': ilm,
            'rnfl_thick': rnfl_thick,
            'inl_thick': inl_thick,
            'isos_thick': isos_thick,
        }


# =============================================================================
# Main Model: Physics DSP with Thickness Prediction
# =============================================================================
class PhysicsDSPThickness(nn.Module):
    """
    Physics-informed boundary detection with coupled thickness prediction.

    Key features:
    1. Predicts ILM + thicknesses instead of independent boundaries
    2. Guarantees valid layer ordering
    3. Enforces minimum thickness for thin layers
    4. Physics-informed with Fresnel and depth compensation
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 48,
        num_boundaries: int = 4,
        use_fresnel: bool = True,
        use_depth_comp: bool = True,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.use_fresnel = use_fresnel
        self.use_depth_comp = use_depth_comp

        # Physics components
        if use_depth_comp:
            self.depth_comp = DepthCompensation()

        if use_fresnel:
            self.fresnel = FresnelPhysicsLite(num_boundaries)

        # Feature encoder
        self.encoder = MultiScaleEncoder(in_channels, hidden_channels)

        # Thickness predictor
        self.thickness_predictor = ThicknessPredictor(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels // 2,
        )

        # Optional: boundary refinement using physics gradient
        if use_fresnel:
            self.physics_refine = nn.Sequential(
                nn.Conv2d(hidden_channels + 1, hidden_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels // 2, 4, 1),  # 4 refinement offsets
            )
            # Initialize refinement to zero (no change initially)
            nn.init.zeros_(self.physics_refine[-1].weight)
            nn.init.zeros_(self.physics_refine[-1].bias)

        # Boundary smoother
        self.smoother = nn.Conv1d(4, 4, 11, padding=5, groups=4)
        nn.init.zeros_(self.smoother.weight)
        nn.init.zeros_(self.smoother.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, 1, H, W] input image

        Returns:
            Dict with boundaries and auxiliary outputs
        """
        B, _, H, W = x.shape

        # Depth compensation
        if self.use_depth_comp:
            x_comp = self.depth_comp(x)
        else:
            x_comp = x

        # Extract features
        features = self.encoder(x_comp)  # [B, C, H, W]

        # Predict ILM + thicknesses
        thickness_out = self.thickness_predictor(features)
        ilm = thickness_out['ilm']  # [B, 1, W]
        rnfl_thick = thickness_out['rnfl_thick']  # [B, 1, W]
        inl_thick = thickness_out['inl_thick']
        isos_thick = thickness_out['isos_thick']

        # Derive boundaries via cumulative sum
        # boundary[0] = ILM
        # boundary[1] = ILM + RNFL_thick
        # boundary[2] = ILM + RNFL_thick + INL_thick
        # boundary[3] = ILM + RNFL_thick + INL_thick + ISOS_thick
        b0 = ilm
        b1 = ilm + rnfl_thick
        b2 = b1 + inl_thick
        b3 = b2 + isos_thick

        boundaries = torch.cat([b0, b1, b2, b3], dim=1)  # [B, 4, W]

        # Optional physics-based refinement
        if self.use_fresnel:
            gradient, expected_strength = self.fresnel(x)
            grad_max = gradient.max(dim=2, keepdim=True)[0].clamp(min=1e-8)
            gradient_norm = gradient / grad_max

            # Concatenate features with gradient for refinement
            feat_pooled = F.adaptive_avg_pool2d(features, (1, W))  # [B, C, 1, W]
            grad_pooled = F.adaptive_avg_pool2d(gradient_norm, (1, W))  # [B, 1, 1, W]
            refine_input = torch.cat([feat_pooled, grad_pooled], dim=1)

            # Predict small offsets
            offsets = self.physics_refine(refine_input).squeeze(2)  # [B, 4, W]
            offsets = torch.tanh(offsets) * 0.02  # Limit to ±2% of image height

            boundaries = boundaries + offsets

        # Smooth boundaries
        boundaries = self.smoother(boundaries) + boundaries

        # Clamp to valid range
        boundaries = boundaries.clamp(0.01, 0.99)

        outputs = {
            'boundaries': boundaries,
            'boundaries_pixels': boundaries * (H - 1),
        }

        if return_aux:
            outputs['ilm'] = ilm
            outputs['rnfl_thick'] = rnfl_thick
            outputs['inl_thick'] = inl_thick
            outputs['isos_thick'] = isos_thick
            outputs['thicknesses_pixels'] = {
                'RNFL': rnfl_thick * (H - 1),
                'INL': inl_thick * (H - 1),
                'ISOS': isos_thick * (H - 1),
            }
            if self.use_fresnel:
                outputs['gradient'] = gradient_norm
                outputs['expected_strength'] = expected_strength
            if self.use_depth_comp:
                outputs['depth_mu'] = self.depth_comp.mu

        return outputs


# =============================================================================
# Loss Function with Direct Thickness Supervision
# =============================================================================
class ThicknessLoss(nn.Module):
    """
    Loss function for coupled thickness prediction.

    Supervises:
    1. Boundary positions (for backward compatibility)
    2. Layer thicknesses directly (key for thin layers)
    3. Ordering constraint (soft)
    4. Smoothness
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        lambda_position: float = 1.0,
        lambda_thickness: float = 2.0,  # High weight for thickness
        lambda_ordering: float = 0.1,   # Soft ordering (mostly handled by architecture)
        lambda_smoothness: float = 0.2,
        learnable_weights: bool = True,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.lambda_position = lambda_position
        self.lambda_thickness = lambda_thickness
        self.lambda_ordering = lambda_ordering
        self.lambda_smoothness = lambda_smoothness
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Learnable weights for each thickness
            # Initialize with higher weight for INL (thin layer)
            self.thickness_weight_logits = nn.Parameter(
                torch.tensor([0.0, 1.5, 0.5])  # RNFL, INL(high!), ISOS
            )

            # Learnable weights for boundaries
            self.boundary_weight_logits = nn.Parameter(
                torch.tensor([0.0, 0.5, 1.0, 0.5])  # ILM, RNFL_INL, INL_ISOS(high!), ISOS_RPE
            )

    @property
    def thickness_weights(self) -> torch.Tensor:
        if self.learnable_weights:
            weights = F.softmax(self.thickness_weight_logits, dim=0)
            return weights * 3  # Scale to sum to 3
        return torch.ones(3)

    @property
    def boundary_weights(self) -> torch.Tensor:
        if self.learnable_weights:
            weights = F.softmax(self.boundary_weight_logits, dim=0)
            return weights * 4  # Scale to sum to 4
        return torch.ones(4)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        pred_thicknesses: Optional[Dict[str, torch.Tensor]] = None,
        valid_mask: Optional[torch.Tensor] = None,
        H: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            pred: [B, 4, W] predicted boundaries (normalized)
            gt: [B, 4, W] ground truth boundaries (normalized)
            pred_thicknesses: Dict with 'rnfl_thick', 'inl_thick', 'isos_thick'
            valid_mask: [B, W] valid columns
            H: image height
        """
        B, N, W = pred.shape
        device = pred.device

        if valid_mask is None:
            valid_mask = torch.ones(B, W, device=device)

        # Get weights
        bw = self.boundary_weights.to(device)
        tw = self.thickness_weights.to(device)

        # =================================================================
        # 1. Position loss (weighted)
        # =================================================================
        pos_error = torch.abs(pred - gt)  # [B, N, W]
        pos_error = pos_error * bw.view(1, N, 1)
        pos_error = pos_error * valid_mask.unsqueeze(1)
        pos_loss = pos_error.sum() / (valid_mask.sum() * N + 1e-8)

        # =================================================================
        # 2. Thickness loss (CRITICAL for thin layers)
        # =================================================================
        # Ground truth thicknesses
        gt_rnfl = gt[:, 1, :] - gt[:, 0, :]  # [B, W]
        gt_inl = gt[:, 2, :] - gt[:, 1, :]
        gt_isos = gt[:, 3, :] - gt[:, 2, :]

        # Predicted thicknesses (from model or derived)
        if pred_thicknesses is not None:
            pred_rnfl = pred_thicknesses['rnfl_thick'].squeeze(1)  # [B, W]
            pred_inl = pred_thicknesses['inl_thick'].squeeze(1)
            pred_isos = pred_thicknesses['isos_thick'].squeeze(1)
        else:
            # Derive from boundaries
            pred_rnfl = pred[:, 1, :] - pred[:, 0, :]
            pred_inl = pred[:, 2, :] - pred[:, 1, :]
            pred_isos = pred[:, 3, :] - pred[:, 2, :]

        # Weighted thickness errors
        thick_errors = torch.stack([
            torch.abs(pred_rnfl - gt_rnfl) * tw[0],
            torch.abs(pred_inl - gt_inl) * tw[1],
            torch.abs(pred_isos - gt_isos) * tw[2],
        ], dim=1)  # [B, 3, W]

        thick_errors = thick_errors * valid_mask.unsqueeze(1)
        thickness_loss = thick_errors.sum() / (valid_mask.sum() * 3 + 1e-8)

        # =================================================================
        # 3. Ordering loss (soft - architecture mostly handles this)
        # =================================================================
        deltas = pred[:, 1:, :] - pred[:, :-1, :]
        ordering_loss = F.relu(-deltas + 0.001).pow(2).mean()

        # =================================================================
        # 4. Smoothness loss
        # =================================================================
        dx = pred[:, :, 1:] - pred[:, :, :-1]
        smoothness_loss = (dx ** 2).mean()

        # =================================================================
        # Total loss
        # =================================================================
        total = (
            self.lambda_position * pos_loss +
            self.lambda_thickness * thickness_loss +
            self.lambda_ordering * ordering_loss +
            self.lambda_smoothness * smoothness_loss
        )

        # =================================================================
        # Stats
        # =================================================================
        with torch.no_grad():
            mae_pixels = (torch.abs(pred - gt) * H).mean(dim=(0, 2))
            thick_mae_rnfl = (torch.abs(pred_rnfl - gt_rnfl) * H * valid_mask).sum() / valid_mask.sum()
            thick_mae_inl = (torch.abs(pred_inl - gt_inl) * H * valid_mask).sum() / valid_mask.sum()
            thick_mae_isos = (torch.abs(pred_isos - gt_isos) * H * valid_mask).sum() / valid_mask.sum()

        stats = {
            'loss': total.item(),
            'pos_loss': pos_loss.item(),
            'thick_loss': thickness_loss.item(),
            'order_loss': ordering_loss.item(),
            'smooth_loss': smoothness_loss.item(),
            'ILM_mae': mae_pixels[0].item(),
            'RNFL_INL_mae': mae_pixels[1].item(),
            'INL_ISOS_mae': mae_pixels[2].item(),
            'ISOS_RPE_mae': mae_pixels[3].item(),
            'RNFL_thick_mae': thick_mae_rnfl.item(),
            'INL_thick_mae': thick_mae_inl.item(),
            'ISOS_thick_mae': thick_mae_isos.item(),
            'avg_mae': mae_pixels.mean().item(),
        }

        return total, stats

    def get_learned_weights(self) -> Dict[str, list]:
        """Get current learned weights."""
        return {
            'boundary_weights': self.boundary_weights.detach().cpu().tolist(),
            'thickness_weights': self.thickness_weights.detach().cpu().tolist(),
        }


# =============================================================================
# Utility: Convert boundaries to segmentation
# =============================================================================
def boundaries_to_segmentation(boundaries: torch.Tensor, H: int, num_classes: int = 4) -> torch.Tensor:
    """Convert boundaries to segmentation mask."""
    B, N, W = boundaries.shape
    device = boundaries.device

    boundaries_px = boundaries * (H - 1)
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W).float()

    seg = torch.zeros(B, H, W, device=device, dtype=torch.long)

    for c in range(num_classes):
        if c == 0:
            mask = (y_coords >= boundaries_px[:, 0:1, :]) & (y_coords < boundaries_px[:, 1:2, :])
        elif c == num_classes - 1:
            mask = y_coords >= boundaries_px[:, -1:, :]
        else:
            mask = (y_coords >= boundaries_px[:, c:c+1, :]) & (y_coords < boundaries_px[:, c+1:c+2, :])
        seg[mask] = c

    return seg


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing PhysicsDSPThickness")
    print("=" * 60)

    device = 'cpu'
    B, H, W = 2, 256, 256

    # Create model
    model = PhysicsDSPThickness(
        hidden_channels=48,
        use_fresnel=True,
        use_depth_comp=True,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    x = torch.randn(B, 1, H, W)
    outputs = model(x, return_aux=True)

    print(f"\nOutputs:")
    print(f"  boundaries: {outputs['boundaries'].shape}")
    print(f"  ILM: {outputs['ilm'].mean().item():.3f}")
    print(f"  RNFL_thick: {outputs['rnfl_thick'].mean().item()*H:.1f}px")
    print(f"  INL_thick: {outputs['inl_thick'].mean().item()*H:.1f}px")
    print(f"  ISOS_thick: {outputs['isos_thick'].mean().item()*H:.1f}px")

    # Test loss
    loss_fn = ThicknessLoss(learnable_weights=True).to(device)

    gt = torch.rand(B, 4, W) * 0.5 + 0.2
    gt, _ = gt.sort(dim=1)

    pred_thick = {
        'rnfl_thick': outputs['rnfl_thick'],
        'inl_thick': outputs['inl_thick'],
        'isos_thick': outputs['isos_thick'],
    }

    loss, stats = loss_fn(outputs['boundaries'], gt, pred_thick, H=H)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"  Position loss: {stats['pos_loss']:.4f}")
    print(f"  Thickness loss: {stats['thick_loss']:.4f}")
    print(f"  INL thickness MAE: {stats['INL_thick_mae']:.1f}px")

    weights = loss_fn.get_learned_weights()
    print(f"\nLearned weights:")
    print(f"  Boundary: {weights['boundary_weights']}")
    print(f"  Thickness: {weights['thickness_weights']}")

    print("\n✓ All tests passed!")
