#!/usr/bin/env python3
"""
Ensemble Physics DSP Model for OCT Boundary Detection.

Combines two complementary approaches:
1. Independent boundaries: Best absolute positioning (Dice 0.52)
2. Thickness-based: Best relative accuracy (INL MAE 3.97px)

The ensemble learns to blend both predictions optimally.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
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
# Ensemble Model
# =============================================================================
class PhysicsDSPEnsemble(nn.Module):
    """
    Ensemble model combining independent and thickness-based boundaries.

    Branch 1: Independent boundary prediction (4 separate boundaries)
    Branch 2: ILM + thickness prediction (cumulative structure)
    Fusion: Learnable per-boundary blending weights

    Final boundary = α * independent + (1-α) * thickness-based
    """

    # Thickness constraints
    MIN_THICKNESS = {
        'rnfl': 5.0 / 256,
        'inl': 3.0 / 256,
        'isos': 4.0 / 256,
    }

    INIT_THICKNESS = {
        'rnfl': 25.0 / 256,
        'inl': 8.0 / 256,
        'isos': 15.0 / 256,
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

        # Branch 1: Independent boundary heads (4 boundaries)
        self.independent_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(H, H // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(H // 2, 1, 1),
                nn.Sigmoid(),
            ) for _ in range(4)
        ])

        # Branch 2: Thickness-based prediction
        # ILM head (same as first independent head structure)
        self.ilm_head = nn.Sequential(
            nn.Conv2d(H, H // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(H // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Thickness heads
        self.thickness_heads = nn.ModuleDict()
        for name in ['rnfl', 'inl', 'isos']:
            self.thickness_heads[name] = nn.Sequential(
                nn.Conv2d(H, H // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(H // 2, 1, 1),
            )

        self._init_thickness_heads()

        # Fusion: Learnable blending weights per boundary
        # α closer to 1 = more independent, closer to 0 = more thickness-based
        # Initialize: b0 uses independent (good ILM), b1-b3 blend more thickness
        self.blend_logits = nn.Parameter(torch.tensor([2.0, 0.5, -0.5, -0.5]))

        # Adaptive fusion: predict per-pixel blend weights from features
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(H, H // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(H // 4, 4, 1),  # 4 boundaries
        )

    def _init_thickness_heads(self):
        """Initialize thickness heads for proper initial values."""
        for name, head in self.thickness_heads.items():
            final_conv = head[-1]
            target = self.INIT_THICKNESS[name] - self.MIN_THICKNESS[name]

            if target > 0.01:
                init_bias = math.log(math.exp(target) - 1)
            else:
                init_bias = math.log(max(target, 1e-6))

            nn.init.zeros_(final_conv.weight)
            nn.init.constant_(final_conv.bias, init_bias)

            first_conv = head[0]
            nn.init.zeros_(first_conv.weight)
            nn.init.zeros_(first_conv.bias)

    def forward(self, x, return_aux=False):
        B, _, H_img, W = x.shape

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Branch 1: Independent boundaries
        independent_bounds = []
        for head in self.independent_heads:
            boundary_map = head(d1)  # [B, 1, H, W]
            boundary = boundary_map.mean(dim=2).squeeze(1)  # [B, W]
            independent_bounds.append(boundary)
        independent_bounds = torch.stack(independent_bounds, dim=1)  # [B, 4, W]

        # Branch 2: Thickness-based boundaries
        ilm = self.ilm_head(d1).mean(dim=2).squeeze(1)  # [B, W]

        thicknesses = {}
        for name in ['rnfl', 'inl', 'isos']:
            raw = self.thickness_heads[name](d1).mean(dim=2).squeeze(1)
            thicknesses[name] = F.softplus(raw) + self.MIN_THICKNESS[name]

        # Compute thickness-based boundaries
        b0_thick = ilm
        b1_thick = b0_thick + thicknesses['rnfl']
        b2_thick = b1_thick + thicknesses['inl']
        b3_thick = b2_thick + thicknesses['isos']
        thickness_bounds = torch.stack([b0_thick, b1_thick, b2_thick, b3_thick], dim=1)

        # Fusion: Combine both branches
        # Global blend weights (learnable)
        global_alpha = torch.sigmoid(self.blend_logits)  # [4]

        # Adaptive blend weights (per-pixel, per-boundary)
        adaptive_logits = self.adaptive_fusion(d1)  # [B, 4, H, W]
        adaptive_alpha = torch.sigmoid(adaptive_logits.mean(dim=2))  # [B, 4, W]

        # Combine global and adaptive (50-50 mix)
        alpha = 0.5 * global_alpha.view(1, 4, 1) + 0.5 * adaptive_alpha

        # Blend: final = alpha * independent + (1-alpha) * thickness
        fused_bounds = alpha * independent_bounds + (1 - alpha) * thickness_bounds

        # Ensure ordering
        b0 = fused_bounds[:, 0, :]
        b1 = torch.maximum(fused_bounds[:, 1, :], b0 + 0.01)
        b2 = torch.maximum(fused_bounds[:, 2, :], b1 + 0.01)
        b3 = torch.maximum(fused_bounds[:, 3, :], b2 + 0.01)

        boundaries = torch.stack([b0, b1, b2, b3], dim=1).clamp(0, 1)

        result = {'boundaries': boundaries}

        if return_aux:
            result.update({
                'independent_bounds': independent_bounds,
                'thickness_bounds': thickness_bounds,
                'blend_alpha': alpha,
                'global_alpha': global_alpha,
                'rnfl_thick': thicknesses['rnfl'],
                'inl_thick': thicknesses['inl'],
                'isos_thick': thicknesses['isos'],
                'features': d1,
            })

        return result


# =============================================================================
# Loss Function
# =============================================================================
class EnsembleLoss(nn.Module):
    """
    Loss function for ensemble model.

    Supervises both branches + fused output:
    1. Independent branch loss
    2. Thickness branch loss (ILM + thickness MAE)
    3. Fused output loss
    4. Soft Dice loss
    """

    def __init__(
        self,
        lambda_independent: float = 1.0,
        lambda_thickness: float = 1.5,
        lambda_fused: float = 2.0,
        lambda_dice: float = 1.0,
    ):
        super().__init__()

        self.lambda_independent = lambda_independent
        self.lambda_thickness = lambda_thickness
        self.lambda_fused = lambda_fused
        self.lambda_dice = lambda_dice

        # Learnable per-boundary weights (emphasize thin layers)
        self.log_boundary_weights = nn.Parameter(torch.tensor([0.0, 0.0, 0.5, 0.0]))

    def forward(self, outputs, gt_bounds, valid_mask, H):
        B, _, W = gt_bounds.shape

        pred_fused = outputs['boundaries']
        pred_indep = outputs['independent_bounds']
        pred_thick = outputs['thickness_bounds']

        # Boundary weights
        boundary_weights = F.softmax(self.log_boundary_weights, dim=0) * 4

        # 1. Independent branch loss
        indep_loss = self._boundary_loss(pred_indep, gt_bounds, valid_mask, boundary_weights)

        # 2. Thickness branch loss
        thick_loss = self._boundary_loss(pred_thick, gt_bounds, valid_mask, boundary_weights)

        # Add thickness-specific supervision
        gt_rnfl = gt_bounds[:, 1, :] - gt_bounds[:, 0, :]
        gt_inl = gt_bounds[:, 2, :] - gt_bounds[:, 1, :]
        gt_isos = gt_bounds[:, 3, :] - gt_bounds[:, 2, :]

        thick_mae = (
            ((outputs['rnfl_thick'] - gt_rnfl).abs() * valid_mask).sum() +
            ((outputs['inl_thick'] - gt_inl).abs() * valid_mask).sum() * 2.0 +  # Emphasize INL
            ((outputs['isos_thick'] - gt_isos).abs() * valid_mask).sum()
        ) / (valid_mask.sum() * 3 + 1e-8)

        thick_loss = thick_loss + thick_mae

        # 3. Fused output loss (most important)
        fused_loss = self._boundary_loss(pred_fused, gt_bounds, valid_mask, boundary_weights)

        # 4. Soft Dice loss on fused output
        dice_loss = self._soft_dice_loss(pred_fused, gt_bounds, H)

        # Total loss
        total_loss = (
            self.lambda_independent * indep_loss +
            self.lambda_thickness * thick_loss +
            self.lambda_fused * fused_loss +
            self.lambda_dice * dice_loss
        )

        # Compute stats
        with torch.no_grad():
            stats = self._compute_stats(outputs, gt_bounds, valid_mask, H)

        return total_loss, stats

    def _boundary_loss(self, pred, gt, valid_mask, weights):
        """Weighted boundary MAE loss."""
        loss = 0
        for i in range(4):
            diff = (pred[:, i, :] - gt[:, i, :]).abs()
            loss = loss + (diff * valid_mask * weights[i]).sum() / (valid_mask.sum() + 1e-8)
        return loss / 4

    def _soft_dice_loss(self, pred_bounds, gt_bounds, H):
        """Soft Dice loss for layer overlap."""
        B, _, W = pred_bounds.shape

        y_coords = torch.linspace(0, 1, H, device=pred_bounds.device).view(1, H, 1)
        sigma = 2.0 / H

        total_dice = 0
        for top_idx, bot_idx in [(0, 1), (1, 2), (2, 3)]:
            pred_top = pred_bounds[:, top_idx, :].unsqueeze(1)
            pred_bot = pred_bounds[:, bot_idx, :].unsqueeze(1)
            pred_mask = (
                torch.sigmoid((y_coords - pred_top) / sigma) *
                torch.sigmoid((pred_bot - y_coords) / sigma)
            )

            gt_top = gt_bounds[:, top_idx, :].unsqueeze(1)
            gt_bot = gt_bounds[:, bot_idx, :].unsqueeze(1)
            gt_mask = (
                torch.sigmoid((y_coords - gt_top) / sigma) *
                torch.sigmoid((gt_bot - y_coords) / sigma)
            )

            intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            total_dice = total_dice + (1 - dice.mean())

        return total_dice / 3

    def _compute_stats(self, outputs, gt_bounds, valid_mask, H):
        pred = outputs['boundaries']
        diff = (pred - gt_bounds).abs() * H

        stats = {
            'ILM_mae': (diff[:, 0, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
            'RNFL_INL_mae': (diff[:, 1, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
            'INL_ISOS_mae': (diff[:, 2, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
            'ISOS_RPE_mae': (diff[:, 3, :] * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8),
        }
        stats['avg_mae'] = sum(stats.values()) / 4

        # Thickness stats
        gt_inl = (gt_bounds[:, 2, :] - gt_bounds[:, 1, :]) * H
        pred_inl = outputs['inl_thick'] * H
        stats['INL_thick_mae'] = ((pred_inl - gt_inl).abs() * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8)

        # Blend alpha stats
        alpha = outputs['global_alpha']
        stats['alpha_b0'] = alpha[0].item()
        stats['alpha_b1'] = alpha[1].item()
        stats['alpha_b2'] = alpha[2].item()
        stats['alpha_b3'] = alpha[3].item()

        return stats

    def get_learned_weights(self):
        return {
            'boundary_weights': (F.softmax(self.log_boundary_weights, dim=0) * 4).detach().cpu().numpy(),
        }


# =============================================================================
# Utility Functions
# =============================================================================
def boundaries_to_segmentation(boundaries, H, num_classes=4):
    B, _, W = boundaries.shape
    device = boundaries.device

    bounds_px = (boundaries * (H - 1)).long()
    seg = torch.zeros(B, H, W, dtype=torch.long, device=device)
    rows = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

    b0 = bounds_px[:, 0, :].unsqueeze(1)
    b1 = bounds_px[:, 1, :].unsqueeze(1)
    b2 = bounds_px[:, 2, :].unsqueeze(1)
    b3 = bounds_px[:, 3, :].unsqueeze(1)

    seg = torch.where(rows >= b0, torch.ones_like(seg) * 0, seg)
    seg = torch.where(rows >= b1, torch.ones_like(seg) * 1, seg)
    seg = torch.where(rows >= b2, torch.ones_like(seg) * 2, seg)
    seg = torch.where(rows >= b3, torch.ones_like(seg) * 3, seg)

    return seg


if __name__ == '__main__':
    model = PhysicsDSPEnsemble(hidden_channels=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 1, 256, 256)
    out = model(x, return_aux=True)

    print(f"\nBoundaries shape: {out['boundaries'].shape}")
    print(f"Independent bounds shape: {out['independent_bounds'].shape}")
    print(f"Thickness bounds shape: {out['thickness_bounds'].shape}")

    alpha = out['global_alpha']
    print(f"\nGlobal blend weights (α):")
    print(f"  b0 (ILM): {alpha[0].item():.2f} (higher = more independent)")
    print(f"  b1 (RNFL/INL): {alpha[1].item():.2f}")
    print(f"  b2 (INL/ISOS): {alpha[2].item():.2f}")
    print(f"  b3 (ISOS/RPE): {alpha[3].item():.2f}")

    print(f"\nThickness predictions:")
    print(f"  RNFL: {out['rnfl_thick'].mean().item() * 256:.1f}px")
    print(f"  INL: {out['inl_thick'].mean().item() * 256:.1f}px")
    print(f"  ISOS: {out['isos_thick'].mean().item() * 256:.1f}px")
