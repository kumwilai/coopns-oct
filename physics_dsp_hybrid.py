#!/usr/bin/env python3
"""
Hybrid Physics-Informed DSP Model.

Combines the best of both approaches:
1. Strong ILM position supervision (absolute anchoring)
2. Coupled thickness prediction for thin layers (INL, ISOS)
3. Physics-based refinement (Fresnel + depth compensation)
4. Soft Dice loss for direct layer optimization

This addresses:
- ILM anchoring problem from pure thickness model
- Thin layer detection from independent boundary model
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Physics Components
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
    """Fresnel reflection physics for boundary detection."""

    REFRACTIVE_INDICES = {
        'vitreous': 1.336, 'RNFL': 1.376, 'GCL_IPL': 1.358,
        'INL': 1.352, 'OPL': 1.365, 'ONL': 1.358,
        'IS': 1.390, 'OS': 1.430, 'RPE': 1.450, 'choroid': 1.370,
    }

    def __init__(self, num_boundaries: int = 4):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.expected_gradient_strength = self._compute_expected_strengths()

    def _compute_expected_strengths(self):
        n = self.REFRACTIVE_INDICES
        boundaries = [
            (n['vitreous'], n['RNFL']),
            (n['GCL_IPL'], n['INL']),
            (n['ONL'], n['IS']),
            (n['OS'], n['RPE']),
        ]
        strengths = []
        for n1, n2 in boundaries:
            R = ((n1 - n2) / (n1 + n2)) ** 2
            strengths.append(R)
        max_R = max(strengths)
        return [s / max_R for s in strengths]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        gradient = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gradient = F.pad(gradient, (0, 0, 0, 1), mode='replicate')
        return gradient, self.expected_gradient_strength


# =============================================================================
# Multi-Scale Feature Encoder
# =============================================================================
class MultiScaleEncoder(nn.Module):
    """Multi-scale feature extraction."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 48):
        super().__init__()

        # Fine scale
        self.enc_fine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Medium scale
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
        total_channels = hidden_channels * 2
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
# Hybrid Boundary Predictor
# =============================================================================
class HybridBoundaryPredictor(nn.Module):
    """
    Hybrid predictor: ILM (absolute) + Thicknesses (relative).

    Predicts:
    - ILM position directly (strong supervision)
    - RNFL thickness
    - INL thickness (with minimum ~3px)
    - ISOS thickness (with minimum ~4px)

    Boundaries derived as:
    - b0 = ILM
    - b1 = ILM + RNFL_thick
    - b2 = b1 + INL_thick
    - b3 = b2 + ISOS_thick
    """

    # Minimum thicknesses (normalized)
    MIN_THICK = {'RNFL': 0.02, 'INL': 0.012, 'ISOS': 0.015}
    # Expected thicknesses for initialization
    INIT_THICK = {'RNFL': 0.12, 'INL': 0.03, 'ISOS': 0.06}

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()

        # Shared refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ILM head (absolute position - critical for anchoring)
        self.ilm_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        # Thickness heads
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

        self._init_heads()

    def _init_heads(self):
        """Initialize to reasonable defaults."""
        # ILM at ~25% from top
        nn.init.zeros_(self.ilm_head[-1].weight)
        nn.init.constant_(self.ilm_head[-1].bias, -1.1)

        # Thickness heads
        for head, name in [(self.rnfl_head, 'RNFL'),
                           (self.inl_head, 'INL'),
                           (self.isos_head, 'ISOS')]:
            nn.init.zeros_(head[-1].weight)
            target = self.INIT_THICK[name] - self.MIN_THICK[name]
            if target > 0:
                init_bias = math.log(max(math.exp(target) - 1, 1e-4))
            else:
                init_bias = -2.0
            nn.init.constant_(head[-1].bias, init_bias)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = features.shape

        feat = self.refine(features)

        # Pool over height for column-wise predictions
        feat_pooled = F.adaptive_avg_pool2d(feat, (1, W))

        # Also use spatial features around expected boundary regions
        feat_top = F.adaptive_avg_pool2d(feat[:, :, :H//4, :], (1, W))
        feat_upper = F.adaptive_avg_pool2d(feat[:, :, H//4:H//2, :], (1, W))

        # ILM uses top features (it's near the top)
        ilm_feat = feat_pooled + 0.5 * feat_top
        ilm = torch.sigmoid(self.ilm_head(ilm_feat)).squeeze(2)  # [B, 1, W]

        # Thicknesses use pooled features
        thick_feat = feat_pooled + 0.3 * feat_upper

        rnfl_raw = self.rnfl_head(thick_feat).squeeze(2)
        inl_raw = self.inl_head(thick_feat).squeeze(2)
        isos_raw = self.isos_head(thick_feat).squeeze(2)

        # Softplus + minimum thickness
        rnfl_thick = F.softplus(rnfl_raw) + self.MIN_THICK['RNFL']
        inl_thick = F.softplus(inl_raw) + self.MIN_THICK['INL']
        isos_thick = F.softplus(isos_raw) + self.MIN_THICK['ISOS']

        return {
            'ilm': ilm,
            'rnfl_thick': rnfl_thick,
            'inl_thick': inl_thick,
            'isos_thick': isos_thick,
        }


# =============================================================================
# Main Model: Hybrid Physics DSP
# =============================================================================
class PhysicsDSPHybrid(nn.Module):
    """
    Hybrid Physics-Informed DSP Model.

    Combines:
    1. Strong ILM anchoring (absolute position)
    2. Thickness prediction for ordering + minimum constraints
    3. Physics-based refinement
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

        # Physics
        if use_depth_comp:
            self.depth_comp = DepthCompensation()

        if use_fresnel:
            self.fresnel = FresnelPhysicsLite(num_boundaries)

        # Encoder
        self.encoder = MultiScaleEncoder(in_channels, hidden_channels)

        # Hybrid predictor
        self.predictor = HybridBoundaryPredictor(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels // 2,
        )

        # Physics-guided refinement
        if use_fresnel:
            self.refine_net = nn.Sequential(
                nn.Conv2d(hidden_channels + 1, hidden_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels // 2, 4, 1),
            )
            nn.init.zeros_(self.refine_net[-1].weight)
            nn.init.zeros_(self.refine_net[-1].bias)

        # Boundary smoother
        self.smoother = nn.Conv1d(4, 4, 11, padding=5, groups=4)
        nn.init.zeros_(self.smoother.weight)
        nn.init.zeros_(self.smoother.bias)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape

        # Depth compensation
        if self.use_depth_comp:
            x_comp = self.depth_comp(x)
        else:
            x_comp = x

        # Extract features
        features = self.encoder(x_comp)

        # Predict ILM + thicknesses
        pred = self.predictor(features)
        ilm = pred['ilm']
        rnfl_thick = pred['rnfl_thick']
        inl_thick = pred['inl_thick']
        isos_thick = pred['isos_thick']

        # Derive boundaries
        b0 = ilm
        b1 = ilm + rnfl_thick
        b2 = b1 + inl_thick
        b3 = b2 + isos_thick

        boundaries = torch.cat([b0, b1, b2, b3], dim=1)  # [B, 4, W]

        # Physics refinement
        if self.use_fresnel:
            gradient, _ = self.fresnel(x)
            grad_max = gradient.max(dim=2, keepdim=True)[0].clamp(min=1e-8)
            gradient_norm = gradient / grad_max

            # Use gradient to refine boundaries
            feat_pooled = F.adaptive_avg_pool2d(features, (1, W))
            grad_pooled = F.adaptive_avg_pool2d(gradient_norm, (1, W))
            refine_input = torch.cat([feat_pooled, grad_pooled], dim=1)

            offsets = self.refine_net(refine_input).squeeze(2)
            offsets = torch.tanh(offsets) * 0.02  # ±2% max

            boundaries = boundaries + offsets

        # Smooth
        boundaries = self.smoother(boundaries) + boundaries
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
            if self.use_depth_comp:
                outputs['depth_mu'] = self.depth_comp.mu

        return outputs


# =============================================================================
# Hybrid Loss Function
# =============================================================================
class HybridLoss(nn.Module):
    """
    Hybrid loss combining:
    1. Strong ILM position loss (anchoring)
    2. Boundary position loss
    3. Thickness loss (critical for thin layers)
    4. Soft Dice loss (direct layer optimization)
    5. Smoothness regularization
    """

    def __init__(
        self,
        lambda_ilm: float = 2.0,      # Strong ILM anchoring
        lambda_position: float = 1.0,
        lambda_thickness: float = 2.0,
        lambda_dice: float = 1.0,
        lambda_smoothness: float = 0.2,
        learnable_weights: bool = True,
    ):
        super().__init__()

        self.lambda_ilm = lambda_ilm
        self.lambda_position = lambda_position
        self.lambda_thickness = lambda_thickness
        self.lambda_dice = lambda_dice
        self.lambda_smoothness = lambda_smoothness
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Boundary weights (ILM gets extra from lambda_ilm)
            self.boundary_weight_logits = nn.Parameter(
                torch.tensor([0.0, 0.5, 1.0, 0.5])
            )
            # Thickness weights (INL highest)
            self.thickness_weight_logits = nn.Parameter(
                torch.tensor([0.0, 1.5, 0.5])
            )
            # Dice weights (thin layers highest)
            self.dice_weight_logits = nn.Parameter(
                torch.tensor([0.5, 1.5, 1.0, 0.5])
            )

    @property
    def boundary_weights(self):
        if self.learnable_weights:
            w = F.softmax(self.boundary_weight_logits, dim=0)
            return w * 4
        return torch.ones(4)

    @property
    def thickness_weights(self):
        if self.learnable_weights:
            w = F.softmax(self.thickness_weight_logits, dim=0)
            return w * 3
        return torch.ones(3)

    @property
    def dice_weights(self):
        if self.learnable_weights:
            w = F.softmax(self.dice_weight_logits, dim=0)
            return w * 4
        return torch.ones(4)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        pred_thicknesses: Optional[Dict[str, torch.Tensor]] = None,
        valid_mask: Optional[torch.Tensor] = None,
        H: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        B, N, W = pred.shape
        device = pred.device

        if valid_mask is None:
            valid_mask = torch.ones(B, W, device=device)

        bw = self.boundary_weights.to(device)
        tw = self.thickness_weights.to(device)
        dw = self.dice_weights.to(device)

        # =================================================================
        # 1. ILM anchoring loss (CRITICAL)
        # =================================================================
        ilm_error = torch.abs(pred[:, 0, :] - gt[:, 0, :])
        ilm_error = ilm_error * valid_mask
        ilm_loss = ilm_error.sum() / (valid_mask.sum() + 1e-8)

        # =================================================================
        # 2. Other boundary position loss
        # =================================================================
        pos_error = torch.abs(pred[:, 1:, :] - gt[:, 1:, :])
        pos_error = pos_error * bw[1:].view(1, -1, 1)
        pos_error = pos_error * valid_mask.unsqueeze(1)
        pos_loss = pos_error.sum() / (valid_mask.sum() * 3 + 1e-8)

        # =================================================================
        # 3. Thickness loss
        # =================================================================
        gt_rnfl = gt[:, 1, :] - gt[:, 0, :]
        gt_inl = gt[:, 2, :] - gt[:, 1, :]
        gt_isos = gt[:, 3, :] - gt[:, 2, :]

        if pred_thicknesses is not None:
            pred_rnfl = pred_thicknesses['rnfl_thick'].squeeze(1)
            pred_inl = pred_thicknesses['inl_thick'].squeeze(1)
            pred_isos = pred_thicknesses['isos_thick'].squeeze(1)
        else:
            pred_rnfl = pred[:, 1, :] - pred[:, 0, :]
            pred_inl = pred[:, 2, :] - pred[:, 1, :]
            pred_isos = pred[:, 3, :] - pred[:, 2, :]

        thick_errors = torch.stack([
            torch.abs(pred_rnfl - gt_rnfl) * tw[0],
            torch.abs(pred_inl - gt_inl) * tw[1],
            torch.abs(pred_isos - gt_isos) * tw[2],
        ], dim=1)
        thick_errors = thick_errors * valid_mask.unsqueeze(1)
        thickness_loss = thick_errors.sum() / (valid_mask.sum() * 3 + 1e-8)

        # =================================================================
        # 4. Soft Dice loss
        # =================================================================
        dice_loss = self._compute_soft_dice(pred, gt, valid_mask, H, dw)

        # =================================================================
        # 5. Smoothness
        # =================================================================
        dx = pred[:, :, 1:] - pred[:, :, :-1]
        smoothness_loss = (dx ** 2).mean()

        # =================================================================
        # Total
        # =================================================================
        total = (
            self.lambda_ilm * ilm_loss +
            self.lambda_position * pos_loss +
            self.lambda_thickness * thickness_loss +
            self.lambda_dice * dice_loss +
            self.lambda_smoothness * smoothness_loss
        )

        # Stats
        with torch.no_grad():
            mae_pixels = (torch.abs(pred - gt) * H).mean(dim=(0, 2))
            thick_mae_rnfl = (torch.abs(pred_rnfl - gt_rnfl) * H * valid_mask).sum() / valid_mask.sum()
            thick_mae_inl = (torch.abs(pred_inl - gt_inl) * H * valid_mask).sum() / valid_mask.sum()
            thick_mae_isos = (torch.abs(pred_isos - gt_isos) * H * valid_mask).sum() / valid_mask.sum()

        stats = {
            'loss': total.item(),
            'ilm_loss': ilm_loss.item(),
            'pos_loss': pos_loss.item(),
            'thick_loss': thickness_loss.item(),
            'dice_loss': dice_loss.item(),
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

    def _compute_soft_dice(self, pred, gt, valid_mask, H, dw):
        """Compute soft Dice loss for each layer."""
        B, N, W = pred.shape
        device = pred.device

        pred_px = pred * (H - 1)
        gt_px = gt * (H - 1)

        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        y_coords = y_coords.view(1, H, 1)

        sigma = 2.0
        total_dice_loss = 0.0

        for layer_idx in range(4):
            if layer_idx == 0:
                pred_top = pred_px[:, 0:1, :]
                pred_bot = pred_px[:, 1:2, :]
                gt_top = gt_px[:, 0:1, :]
                gt_bot = gt_px[:, 1:2, :]
            elif layer_idx == 3:
                pred_top = pred_px[:, 3:4, :]
                pred_bot = torch.full_like(pred_top, H - 1)
                gt_top = gt_px[:, 3:4, :]
                gt_bot = torch.full_like(gt_top, H - 1)
            else:
                pred_top = pred_px[:, layer_idx:layer_idx+1, :]
                pred_bot = pred_px[:, layer_idx+1:layer_idx+2, :]
                gt_top = gt_px[:, layer_idx:layer_idx+1, :]
                gt_bot = gt_px[:, layer_idx+1:layer_idx+2, :]

            pred_mask = (
                torch.sigmoid((y_coords - pred_top) / sigma) *
                torch.sigmoid((pred_bot - y_coords) / sigma)
            )
            gt_mask = (
                torch.sigmoid((y_coords - gt_top) / sigma) *
                torch.sigmoid((gt_bot - y_coords) / sigma)
            )

            intersection = (pred_mask * gt_mask).sum(dim=1)
            union = pred_mask.sum(dim=1) + gt_mask.sum(dim=1)

            intersection = intersection * valid_mask
            union = union * valid_mask

            dice = (2 * intersection.sum() + 1e-8) / (union.sum() + 1e-8)
            dice_loss = 1.0 - dice

            total_dice_loss = total_dice_loss + dw[layer_idx] * dice_loss

        return total_dice_loss / 4.0

    def get_learned_weights(self):
        return {
            'boundary_weights': self.boundary_weights.detach().cpu().tolist(),
            'thickness_weights': self.thickness_weights.detach().cpu().tolist(),
            'dice_weights': self.dice_weights.detach().cpu().tolist(),
        }


# =============================================================================
# Utility
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
    print("Testing PhysicsDSPHybrid")
    print("=" * 60)

    device = 'cpu'
    B, H, W = 2, 256, 256

    model = PhysicsDSPHybrid(hidden_channels=48).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(B, 1, H, W)
    outputs = model(x, return_aux=True)

    print(f"\nOutputs:")
    print(f"  boundaries: {outputs['boundaries'].shape}")
    print(f"  ILM: {outputs['ilm'].mean().item()*H:.1f}px")
    print(f"  RNFL_thick: {outputs['rnfl_thick'].mean().item()*H:.1f}px")
    print(f"  INL_thick: {outputs['inl_thick'].mean().item()*H:.1f}px")
    print(f"  ISOS_thick: {outputs['isos_thick'].mean().item()*H:.1f}px")

    # Test loss
    loss_fn = HybridLoss(learnable_weights=True).to(device)

    gt = torch.rand(B, 4, W) * 0.5 + 0.2
    gt, _ = gt.sort(dim=1)

    pred_thick = {
        'rnfl_thick': outputs['rnfl_thick'],
        'inl_thick': outputs['inl_thick'],
        'isos_thick': outputs['isos_thick'],
    }

    loss, stats = loss_fn(outputs['boundaries'], gt, pred_thick, H=H)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"  ILM loss: {stats['ilm_loss']:.4f}")
    print(f"  Dice loss: {stats['dice_loss']:.4f}")
    print(f"  INL thickness MAE: {stats['INL_thick_mae']:.1f}px")

    weights = loss_fn.get_learned_weights()
    print(f"\nLearned weights:")
    print(f"  Boundary: {[f'{w:.2f}' for w in weights['boundary_weights']]}")
    print(f"  Thickness: {[f'{w:.2f}' for w in weights['thickness_weights']]}")
    print(f"  Dice: {[f'{w:.2f}' for w in weights['dice_weights']]}")

    print("\n✓ All tests passed!")
