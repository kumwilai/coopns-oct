#!/usr/bin/env python3
"""
Two-Stage Physics DSP Model for OCT Boundary Detection.

Stage 1: Independent ILM boundary prediction (best absolute positioning)
Stage 2: Thickness prediction for remaining layers (best relative accuracy)

This combines the strengths of both approaches:
- ILM from independent prediction: ~6.7px MAE
- Thickness for thin layers: ~1px MAE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================
class ConvBlock(nn.Module):
    """Conv + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Downsample block: MaxPool + 2x ConvBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    """Upsample block: Upsample + Concat + 2x ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# =============================================================================
# Two-Stage Model
# =============================================================================
class PhysicsDSPTwoStage(nn.Module):
    """
    Two-stage model for OCT boundary detection.

    Stage 1: Predict ILM boundary independently (best absolute position)
    Stage 2: Predict thicknesses for RNFL, INL, ISOS (best relative accuracy)

    Final boundaries:
        b0 = ILM (from Stage 1)
        b1 = b0 + RNFL_thickness
        b2 = b1 + INL_thickness
        b3 = b2 + ISOS_thickness
    """

    # Minimum thickness constraints (in normalized units for H=256)
    MIN_THICKNESS = {
        'rnfl': 5.0 / 256,   # RNFL ≥ 5 pixels
        'inl': 3.0 / 256,    # INL ≥ 3 pixels (thin layer!)
        'isos': 4.0 / 256,   # ISOS ≥ 4 pixels
    }

    # Expected initial thicknesses (in normalized units)
    INIT_THICKNESS = {
        'rnfl': 25.0 / 256,  # ~25 pixels
        'inl': 8.0 / 256,    # ~8 pixels (thin!)
        'isos': 15.0 / 256,  # ~15 pixels
    }

    def __init__(self, in_channels=1, hidden_channels=48):
        super().__init__()

        H = hidden_channels

        # Shared encoder
        self.enc1 = nn.Sequential(ConvBlock(in_channels, H), ConvBlock(H, H))
        self.enc2 = DownBlock(H, H * 2)
        self.enc3 = DownBlock(H * 2, H * 4)
        self.enc4 = DownBlock(H * 4, H * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(H * 8, H * 16),
            ConvBlock(H * 16, H * 16),
        )

        # Decoder
        self.dec4 = UpBlock(H * 16, H * 8, H * 8)
        self.dec3 = UpBlock(H * 8, H * 4, H * 4)
        self.dec2 = UpBlock(H * 4, H * 2, H * 2)
        self.dec1 = UpBlock(H * 2, H, H)

        # Stage 1: ILM prediction head (independent, direct supervision)
        self.ilm_head = nn.Sequential(
            nn.Conv2d(H, H // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(H // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Stage 2: Thickness prediction heads
        # Each predicts a thickness value per column
        self.thickness_heads = nn.ModuleDict()
        for name in ['rnfl', 'inl', 'isos']:
            self.thickness_heads[name] = nn.Sequential(
                nn.Conv2d(H, H // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(H // 2, 1, 1),
            )

        # Initialize thickness heads for proper initial values
        self._init_thickness_heads()

        # Stage 3: Residual correction heads (to compensate for accumulated errors)
        # These predict small corrections for b1, b2, b3
        self.correction_heads = nn.ModuleDict()
        for name in ['b1', 'b2', 'b3']:
            self.correction_heads[name] = nn.Sequential(
                nn.Conv2d(H, H // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(H // 4, 1, 1),
                nn.Tanh(),  # Output in [-1, 1]
            )

        # Initialize correction heads to output near-zero
        self._init_correction_heads()

        # Learnable scale for corrections (start small, let model learn to increase)
        self.correction_scale = nn.Parameter(torch.tensor([0.02, 0.02, 0.02]))  # ~5px max initially

        # Physics enhancement module (optional refinement)
        self.physics_refine = PhysicsRefinement(H)

    def _init_thickness_heads(self):
        """Initialize thickness heads to produce reasonable initial values."""
        for name, head in self.thickness_heads.items():
            # Get the final conv layer
            final_conv = head[-1]

            # Calculate bias for softplus to output expected thickness
            # We need: softplus(bias) + MIN = INIT
            # So: softplus(bias) = INIT - MIN = target
            # Inverse softplus: if softplus(x) = y, then x = log(exp(y) - 1)
            target = self.INIT_THICKNESS[name] - self.MIN_THICKNESS[name]

            if target > 0.01:
                # For larger targets, use exact inverse
                init_bias = math.log(math.exp(target) - 1)
            else:
                # For small targets, log(exp(y)-1) ≈ log(y) which is very negative
                # softplus(log(y)) ≈ y for small y
                init_bias = math.log(max(target, 1e-6))

            # Zero the weights so output is purely the bias initially
            nn.init.zeros_(final_conv.weight)
            nn.init.constant_(final_conv.bias, init_bias)

            # Also zero the first conv layer weights for stable init
            first_conv = head[0]
            nn.init.zeros_(first_conv.weight)
            nn.init.zeros_(first_conv.bias)

    def _init_correction_heads(self):
        """Initialize correction heads to output near-zero initially."""
        for name, head in self.correction_heads.items():
            # Zero all weights and biases so output starts near zero
            for layer in head:
                if hasattr(layer, 'weight'):
                    nn.init.zeros_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, return_aux=False):
        """
        Forward pass.

        Args:
            x: Input image [B, 1, H, W]
            return_aux: If True, return auxiliary outputs for loss computation

        Returns:
            Dictionary with 'boundaries' [B, 4, W] and optional auxiliary outputs
        """
        B, _, H_img, W = x.shape

        # Encoder
        e1 = self.enc1(x)       # [B, H, H_img, W]
        e2 = self.enc2(e1)      # [B, 2H, H_img/2, W/2]
        e3 = self.enc3(e2)      # [B, 4H, H_img/4, W/4]
        e4 = self.enc4(e3)      # [B, 8H, H_img/8, W/8]

        # Bottleneck
        b = self.bottleneck(e4) # [B, 16H, H_img/16, W/16]

        # Decoder
        d4 = self.dec4(b, e4)   # [B, 8H, H_img/8, W/8]
        d3 = self.dec3(d4, e3)  # [B, 4H, H_img/4, W/4]
        d2 = self.dec2(d3, e2)  # [B, 2H, H_img/2, W/2]
        d1 = self.dec1(d2, e1)  # [B, H, H_img, W]

        # Stage 1: ILM prediction (independent)
        ilm_map = self.ilm_head(d1)  # [B, 1, H_img, W]
        ilm = ilm_map.mean(dim=2).squeeze(1)  # [B, W] - column-wise average

        # Stage 2: Thickness predictions
        thicknesses = {}
        for name in ['rnfl', 'inl', 'isos']:
            raw = self.thickness_heads[name](d1)  # [B, 1, H_img, W]
            raw = raw.mean(dim=2).squeeze(1)  # [B, W]

            # Apply softplus + minimum constraint
            min_thick = self.MIN_THICKNESS[name]
            thicknesses[name] = F.softplus(raw) + min_thick

        # Compute boundaries from ILM + thicknesses (before correction)
        b0 = ilm                                    # ILM
        b1_raw = b0 + thicknesses['rnfl']          # RNFL/GCL boundary
        b2_raw = b1_raw + thicknesses['inl']       # INL/ISOS boundary
        b3_raw = b2_raw + thicknesses['isos']      # ISOS/RPE boundary

        # Stage 3: Apply residual corrections to compensate for accumulated errors
        corrections = {}
        for i, name in enumerate(['b1', 'b2', 'b3']):
            corr_map = self.correction_heads[name](d1)  # [B, 1, H_img, W]
            corr = corr_map.mean(dim=2).squeeze(1)  # [B, W], in [-1, 1]
            corrections[name] = corr * self.correction_scale[i]  # Scale to ~[-0.02, 0.02]

        # Apply corrections
        b1 = b1_raw + corrections['b1']
        b2 = b2_raw + corrections['b2']
        b3 = b3_raw + corrections['b3']

        # Ensure ordering: b0 < b1 < b2 < b3
        b1 = torch.maximum(b1, b0 + 0.01)
        b2 = torch.maximum(b2, b1 + 0.01)
        b3 = torch.maximum(b3, b2 + 0.01)

        # Stack boundaries
        boundaries = torch.stack([b0, b1, b2, b3], dim=1)  # [B, 4, W]

        # Clamp to valid range
        boundaries = boundaries.clamp(0, 1)

        result = {'boundaries': boundaries}

        if return_aux:
            result.update({
                'ilm_raw': ilm,
                'rnfl_thick': thicknesses['rnfl'],
                'inl_thick': thicknesses['inl'],
                'isos_thick': thicknesses['isos'],
                'corrections': corrections,
                'correction_scale': self.correction_scale.detach(),
                'boundaries_pre_correction': torch.stack([b0, b1_raw, b2_raw, b3_raw], dim=1),
                'features': d1,
            })

        return result


class PhysicsRefinement(nn.Module):
    """
    Physics-based boundary refinement.

    Applies small learned corrections based on:
    - Local image gradients
    - Boundary smoothness
    - Cross-boundary consistency
    """

    def __init__(self, feature_channels):
        super().__init__()

        # Learnable refinement weights (small corrections)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(feature_channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),  # 4 boundaries
            nn.Tanh(),  # Small corrections in [-1, 1]
        )

        # Scale factor for refinement (start small)
        self.refine_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, boundaries, features):
        """
        Apply physics-based refinement.

        Args:
            boundaries: [B, 4, W] predicted boundaries
            features: [B, C, H, W] encoder features

        Returns:
            Refined boundaries [B, 4, W]
        """
        # Get refinement from features
        refine = self.refine_conv(features)  # [B, 4, H, W]
        refine = refine.mean(dim=2)  # [B, 4, W]

        # Apply small correction
        refined = boundaries + self.refine_scale * refine

        # Ensure ordering constraint: b0 < b1 < b2 < b3
        refined = self._enforce_ordering(refined)

        # Clamp to valid range
        refined = refined.clamp(0, 1)

        return refined

    def _enforce_ordering(self, boundaries):
        """Ensure boundaries are properly ordered."""
        B, _, W = boundaries.shape

        b0 = boundaries[:, 0, :]
        b1 = torch.maximum(boundaries[:, 1, :], b0 + 0.01)
        b2 = torch.maximum(boundaries[:, 2, :], b1 + 0.01)
        b3 = torch.maximum(boundaries[:, 3, :], b2 + 0.01)

        return torch.stack([b0, b1, b2, b3], dim=1)


# =============================================================================
# Loss Function
# =============================================================================
class TwoStageLoss(nn.Module):
    """
    Loss function for two-stage model.

    Components:
    1. ILM loss: Strong supervision for Stage 1 (λ=2.0)
    2. Position loss: MAE for all boundaries
    3. Thickness loss: Direct supervision on predicted thicknesses
    4. Soft Dice loss: Direct layer overlap optimization
    """

    def __init__(
        self,
        lambda_ilm: float = 2.0,
        lambda_position: float = 1.0,
        lambda_thickness: float = 2.0,
        lambda_dice: float = 1.0,
        learnable_weights: bool = True,
    ):
        super().__init__()

        self.lambda_ilm = lambda_ilm
        self.lambda_position = lambda_position
        self.lambda_thickness = lambda_thickness
        self.lambda_dice = lambda_dice

        # Learnable per-boundary weights
        if learnable_weights:
            # Boundary weights: emphasize thin layer boundaries
            self.log_boundary_weights = nn.Parameter(torch.tensor([0.0, 0.0, 0.5, 0.0]))
            # Thickness weights: emphasize INL (thinnest)
            self.log_thickness_weights = nn.Parameter(torch.tensor([0.0, 0.7, 0.0]))
            # Dice weights: emphasize thin layers
            self.log_dice_weights = nn.Parameter(torch.tensor([0.0, 0.5, 0.3, 0.0]))
        else:
            self.register_buffer('log_boundary_weights', torch.zeros(4))
            self.register_buffer('log_thickness_weights', torch.zeros(3))
            self.register_buffer('log_dice_weights', torch.zeros(4))

    def forward(self, pred_bounds, gt_bounds, pred_aux, valid_mask, H):
        """
        Compute total loss.

        Args:
            pred_bounds: [B, 4, W] predicted boundaries
            gt_bounds: [B, 4, W] ground truth boundaries
            pred_aux: Dict with 'ilm_raw', 'rnfl_thick', 'inl_thick', 'isos_thick'
            valid_mask: [B, W] valid columns mask
            H: Image height for pixel conversion

        Returns:
            total_loss, stats_dict
        """
        B, _, W = pred_bounds.shape

        # Get weights
        boundary_weights = F.softmax(self.log_boundary_weights, dim=0) * 4
        thickness_weights = F.softmax(self.log_thickness_weights, dim=0) * 3
        dice_weights = F.softmax(self.log_dice_weights, dim=0) * 4

        # 1. ILM loss (Stage 1 - critical!)
        ilm_loss = F.l1_loss(
            pred_bounds[:, 0, :] * valid_mask,
            gt_bounds[:, 0, :] * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-8)

        # 2. Position loss for all boundaries
        position_loss = 0
        for i in range(4):
            diff = (pred_bounds[:, i, :] - gt_bounds[:, i, :]).abs()
            weighted_diff = diff * valid_mask * boundary_weights[i]
            position_loss = position_loss + weighted_diff.sum() / (valid_mask.sum() + 1e-8)
        position_loss = position_loss / 4

        # 3. Thickness loss
        gt_thick_rnfl = gt_bounds[:, 1, :] - gt_bounds[:, 0, :]
        gt_thick_inl = gt_bounds[:, 2, :] - gt_bounds[:, 1, :]
        gt_thick_isos = gt_bounds[:, 3, :] - gt_bounds[:, 2, :]

        thickness_loss = 0
        thickness_names = ['rnfl', 'inl', 'isos']
        gt_thicks = [gt_thick_rnfl, gt_thick_inl, gt_thick_isos]

        for i, (name, gt_t) in enumerate(zip(thickness_names, gt_thicks)):
            pred_t = pred_aux[f'{name}_thick']
            diff = (pred_t - gt_t).abs() * valid_mask * thickness_weights[i]
            thickness_loss = thickness_loss + diff.sum() / (valid_mask.sum() + 1e-8)
        thickness_loss = thickness_loss / 3

        # 4. Soft Dice loss
        dice_loss = self._compute_soft_dice_loss(pred_bounds, gt_bounds, H, dice_weights)

        # Total loss
        total_loss = (
            self.lambda_ilm * ilm_loss +
            self.lambda_position * position_loss +
            self.lambda_thickness * thickness_loss +
            self.lambda_dice * dice_loss
        )

        # Compute stats for logging
        with torch.no_grad():
            stats = self._compute_stats(pred_bounds, gt_bounds, pred_aux, valid_mask, H)

        return total_loss, stats

    def _compute_soft_dice_loss(self, pred_bounds, gt_bounds, H, weights):
        """Compute differentiable soft Dice loss."""
        B, _, W = pred_bounds.shape

        # Create coordinate grid [H, W]
        y_coords = torch.linspace(0, 1, H, device=pred_bounds.device)
        y_coords = y_coords.view(1, H, 1)  # [1, H, 1]

        sigma = 2.0 / H  # Soft boundary width
        total_dice_loss = 0

        # Layer 0: above b0 to b1 (RNFL/GCL)
        # Layer 1: b1 to b2 (INL/OPL/ONL)
        # Layer 2: b2 to b3 (IS/OS)
        # Layer 3: below b3 (RPE/Choroid)

        layer_configs = [
            (0, 1),  # Layer 0: b0 to b1
            (1, 2),  # Layer 1: b1 to b2
            (2, 3),  # Layer 2: b2 to b3
        ]

        for layer_idx, (top_idx, bot_idx) in enumerate(layer_configs):
            # Predicted soft mask
            pred_top = pred_bounds[:, top_idx, :].unsqueeze(1)  # [B, 1, W]
            pred_bot = pred_bounds[:, bot_idx, :].unsqueeze(1)  # [B, 1, W]

            pred_mask = (
                torch.sigmoid((y_coords - pred_top) / sigma) *
                torch.sigmoid((pred_bot - y_coords) / sigma)
            )  # [B, H, W]

            # Ground truth soft mask
            gt_top = gt_bounds[:, top_idx, :].unsqueeze(1)
            gt_bot = gt_bounds[:, bot_idx, :].unsqueeze(1)

            gt_mask = (
                torch.sigmoid((y_coords - gt_top) / sigma) *
                torch.sigmoid((gt_bot - y_coords) / sigma)
            )  # [B, H, W]

            # Dice coefficient
            intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)

            # Weighted Dice loss
            total_dice_loss = total_dice_loss + weights[layer_idx] * (1 - dice.mean())

        return total_dice_loss / 3

    def _compute_stats(self, pred_bounds, gt_bounds, pred_aux, valid_mask, H):
        """Compute statistics for logging."""
        # Boundary MAEs in pixels
        diff = (pred_bounds - gt_bounds).abs() * H

        stats = {
            'ILM_mae': (diff[:, 0, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
            'RNFL_INL_mae': (diff[:, 1, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
            'INL_ISOS_mae': (diff[:, 2, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
            'ISOS_RPE_mae': (diff[:, 3, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
        }
        stats['avg_mae'] = sum(stats.values()) / 4

        # Thickness MAEs in pixels
        gt_thick_rnfl = (gt_bounds[:, 1, :] - gt_bounds[:, 0, :]) * H
        gt_thick_inl = (gt_bounds[:, 2, :] - gt_bounds[:, 1, :]) * H
        gt_thick_isos = (gt_bounds[:, 3, :] - gt_bounds[:, 2, :]) * H

        pred_thick_rnfl = pred_aux['rnfl_thick'] * H
        pred_thick_inl = pred_aux['inl_thick'] * H
        pred_thick_isos = pred_aux['isos_thick'] * H

        stats['RNFL_thick_mae'] = ((pred_thick_rnfl - gt_thick_rnfl).abs() * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8)
        stats['INL_thick_mae'] = ((pred_thick_inl - gt_thick_inl).abs() * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8)
        stats['ISOS_thick_mae'] = ((pred_thick_isos - gt_thick_isos).abs() * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8)

        return stats

    def get_learned_weights(self):
        """Get current learned weights for logging."""
        return {
            'boundary_weights': (F.softmax(self.log_boundary_weights, dim=0) * 4).detach().cpu().numpy(),
            'thickness_weights': (F.softmax(self.log_thickness_weights, dim=0) * 3).detach().cpu().numpy(),
            'dice_weights': (F.softmax(self.log_dice_weights, dim=0) * 4).detach().cpu().numpy(),
        }


# =============================================================================
# Utility Functions
# =============================================================================
def boundaries_to_segmentation(boundaries, H, num_classes=4):
    """
    Convert boundary predictions to segmentation mask.

    Args:
        boundaries: [B, 4, W] boundary positions (normalized 0-1)
        H: Height of output segmentation
        num_classes: Number of classes (4 for retinal layers)

    Returns:
        segmentation: [B, H, W] with class labels 0-3
    """
    B, _, W = boundaries.shape
    device = boundaries.device

    # Convert to pixel coordinates
    bounds_px = (boundaries * (H - 1)).long()  # [B, 4, W]

    # Create output segmentation
    seg = torch.zeros(B, H, W, dtype=torch.long, device=device)

    # Create row indices
    rows = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

    # Class 0: above b0 (background/vitreous) - actually rows < b0
    # Class 0 (RNFL/GCL): b0 <= rows < b1
    # Class 1 (INL/OPL/ONL): b1 <= rows < b2
    # Class 2 (IS/OS): b2 <= rows < b3
    # Class 3 (RPE/Choroid): rows >= b3

    b0 = bounds_px[:, 0, :].unsqueeze(1)  # [B, 1, W]
    b1 = bounds_px[:, 1, :].unsqueeze(1)
    b2 = bounds_px[:, 2, :].unsqueeze(1)
    b3 = bounds_px[:, 3, :].unsqueeze(1)

    # Assign classes based on row position relative to boundaries
    seg = torch.where(rows >= b0, torch.ones_like(seg) * 0, seg)  # Class 0 (RNFL)
    seg = torch.where(rows >= b1, torch.ones_like(seg) * 1, seg)  # Class 1 (INL)
    seg = torch.where(rows >= b2, torch.ones_like(seg) * 2, seg)  # Class 2 (IS/OS)
    seg = torch.where(rows >= b3, torch.ones_like(seg) * 3, seg)  # Class 3 (RPE)

    return seg


if __name__ == '__main__':
    # Quick test
    model = PhysicsDSPTwoStage(hidden_channels=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    out = model(x, return_aux=True)

    print(f"Boundaries shape: {out['boundaries'].shape}")
    print(f"ILM raw shape: {out['ilm_raw'].shape}")
    print(f"RNFL thickness: {out['rnfl_thick'].mean().item() * 256:.1f} px")
    print(f"INL thickness: {out['inl_thick'].mean().item() * 256:.1f} px")
    print(f"ISOS thickness: {out['isos_thick'].mean().item() * 256:.1f} px")

    # Test loss
    loss_fn = TwoStageLoss()
    gt_bounds = torch.rand(2, 4, 256).sort(dim=1)[0]  # Sorted boundaries
    valid_mask = torch.ones(2, 256)

    pred_aux = {
        'rnfl_thick': out['rnfl_thick'],
        'inl_thick': out['inl_thick'],
        'isos_thick': out['isos_thick'],
    }

    loss, stats = loss_fn(out['boundaries'], gt_bounds, pred_aux, valid_mask, 256)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Stats: {stats}")
