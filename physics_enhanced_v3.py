#!/usr/bin/env python3
"""
Physics-Enhanced Ensemble Model v3 for OCT Boundary Detection.

Enhancements over v2:
1. Depth-aware boundary loss weighting (deeper = higher weight)
2. Enhanced Fresnel loss with absolute targets
3. Multi-scale boundary prediction fusion
4. Boundary smoothness regularization
5. Layer-specific physics refinement

Maintains all original physics contributions:
- Beer-Lambert depth compensation
- Boundary gradient alignment
- Layer intensity consistency
- Fresnel reflection consistency
- Physics-guided refinement
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks (same as v2)
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
# Physics Modules
# =============================================================================
class DepthCompensation(nn.Module):
    """Beer-Lambert depth compensation."""
    def __init__(self, mu_min=0.3, mu_max=0.95):
        super().__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_range = mu_max - mu_min
        self.log_mu = nn.Parameter(torch.tensor(0.0))

    def get_mu(self):
        return self.mu_min + self.mu_range * torch.sigmoid(self.log_mu)

    def forward(self, image):
        B, C, H, W = image.shape
        depth = torch.linspace(0, 1, H, device=image.device).view(1, 1, H, 1)
        mu = self.get_mu()
        compensation = torch.exp(mu * depth)
        compensated = image * compensation.clamp(max=5.0)
        return compensated


class BoundaryGradientModule(nn.Module):
    """Extract gradient information at boundaries."""
    def __init__(self):
        super().__init__()
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [0,  0,  0],
                                 [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, image):
        return F.conv2d(image, self.sobel_y, padding=1)


class MultiScaleBoundaryHead(nn.Module):
    """Multi-scale boundary prediction with fusion."""
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales

        # Per-scale prediction heads
        self.scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, 4, 1),
                nn.Sigmoid(),
            ) for _ in scales
        ])

        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] feature map
        Returns:
            boundaries: [B, 4, W] fused multi-scale predictions
        """
        B, C, H, W = features.shape
        predictions = []

        for scale, head in zip(self.scales, self.scale_heads):
            if scale > 1:
                # Downsample, predict, upsample
                scaled = F.avg_pool2d(features, scale)
                pred = head(scaled)
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
            else:
                pred = head(features)

            # Average over height to get boundary positions
            boundaries = pred.mean(dim=2)  # [B, 4, W]
            predictions.append(boundaries)

        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * p for w, p in zip(weights, predictions))

        return fused


class LayerSpecificRefinement(nn.Module):
    """
    Layer-specific physics refinement.

    Different layers have different optical properties:
    - RNFL: High reflectivity, clear boundary
    - INL: Low contrast, thin layer
    - IS/OS: Highest reflectivity (Fresnel)
    - RPE: Strong reflection but variable
    """
    def __init__(self, feature_channels):
        super().__init__()

        self.gradient_module = BoundaryGradientModule()

        # Per-boundary refinement networks
        self.boundary_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels + 2, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Tanh(),
            ) for _ in range(4)
        ])

        # Per-boundary refinement scales (learnable)
        self.refine_scales = nn.Parameter(torch.tensor([0.02, 0.03, 0.04, 0.04]))

    def forward(self, boundaries, features, image, compensated_image):
        """
        Args:
            boundaries: [B, 4, W] predicted boundaries
            features: [B, C, H, W] encoder features
            image: [B, 1, H, W] original image
            compensated_image: [B, 1, H, W] depth-compensated image
        """
        B, _, W = boundaries.shape
        gradients = self.gradient_module(image)

        # Concatenate physics features
        combined = torch.cat([features, gradients, compensated_image], dim=1)

        refined_boundaries = []
        for i, refiner in enumerate(self.boundary_refiners):
            correction = refiner(combined)  # [B, 1, H, W]
            correction = correction.mean(dim=2).squeeze(1)  # [B, W]

            scale = torch.sigmoid(self.refine_scales[i]) * 0.1  # Max 10% correction
            refined = boundaries[:, i, :] + scale * correction
            refined_boundaries.append(refined)

        return torch.stack(refined_boundaries, dim=1).clamp(0, 1)


# =============================================================================
# Enhanced Physics Ensemble Model
# =============================================================================
class PhysicsEnsembleV3(nn.Module):
    """
    Physics-enhanced ensemble model v3 with all enhancements.
    """

    # Thickness values as fractions of image height (normalized 0-1)
    # These are independent of image resolution
    MIN_THICKNESS = {
        'rnfl': 5.0 / 256,   # ~2% of image height
        'inl': 3.0 / 256,    # ~1.2% of image height
        'isos': 4.0 / 256,   # ~1.5% of image height
    }

    INIT_THICKNESS = {
        'rnfl': 25.0 / 256,  # ~10% of image height
        'inl': 8.0 / 256,    # ~3% of image height
        'isos': 15.0 / 256,  # ~6% of image height
    }

    def __init__(self, in_channels=1, hidden_channels=48, num_boundaries=4):
        super().__init__()
        H = hidden_channels

        # Physics preprocessing
        self.depth_compensation = DepthCompensation()

        # Shared encoder
        self.enc1 = nn.Sequential(ConvBlock(in_channels + 1, H), ConvBlock(H, H))
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

        # Multi-scale boundary prediction (instead of independent heads)
        self.multiscale_head = MultiScaleBoundaryHead(H, scales=[1, 2, 4])

        # Thickness-based branch
        self.ilm_head = nn.Sequential(
            nn.Conv2d(H, H // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(H // 2, 1, 1),
            nn.Sigmoid(),
        )

        self.thickness_heads = nn.ModuleDict()
        for name in ['rnfl', 'inl', 'isos']:
            self.thickness_heads[name] = nn.Sequential(
                nn.Conv2d(H, H // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(H // 2, 1, 1),
            )

        self._init_thickness_heads()

        # Learnable blend weights
        self.blend_logits = nn.Parameter(torch.tensor([1.0, 0.0, -0.5, -0.5]))

        # Adaptive fusion
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(H, H // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(H // 4, 4, 1),
        )

        # Layer-specific physics refinement
        self.layer_refine = LayerSpecificRefinement(H)

    def _init_thickness_heads(self):
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
            nn.init.normal_(first_conv.weight, mean=0, std=0.01)
            nn.init.zeros_(first_conv.bias)

    def forward(self, x, return_aux=False):
        B, _, H_img, W = x.shape

        # Physics preprocessing
        x_compensated = self.depth_compensation(x)
        x_combined = torch.cat([x, x_compensated], dim=1)

        # Encoder
        e1 = self.enc1(x_combined)
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

        # Multi-scale independent boundaries
        independent_bounds = self.multiscale_head(d1)  # [B, 4, W]

        # Thickness-based boundaries
        ilm = self.ilm_head(d1).mean(dim=2).squeeze(1)
        thicknesses = {}
        for name in ['rnfl', 'inl', 'isos']:
            raw = self.thickness_heads[name](d1).mean(dim=2).squeeze(1)
            thicknesses[name] = F.softplus(raw) + self.MIN_THICKNESS[name]

        b0_thick = ilm
        b1_thick = b0_thick + thicknesses['rnfl']
        b2_thick = b1_thick + thicknesses['inl']
        b3_thick = b2_thick + thicknesses['isos']
        thickness_bounds = torch.stack([b0_thick, b1_thick, b2_thick, b3_thick], dim=1)

        # Fusion
        global_alpha = torch.sigmoid(self.blend_logits)
        adaptive_logits = self.adaptive_fusion(d1)
        adaptive_alpha = torch.sigmoid(adaptive_logits.mean(dim=2))
        alpha = 0.7 * global_alpha.view(1, 4, 1) + 0.3 * adaptive_alpha

        fused_bounds = alpha * independent_bounds + (1 - alpha) * thickness_bounds

        # Soft ordering constraints
        steepness = 100.0
        gap = 0.01

        b0 = fused_bounds[:, 0, :]
        threshold_1 = b0 + gap
        b1 = threshold_1 + F.softplus((fused_bounds[:, 1, :] - threshold_1) * steepness) / steepness

        threshold_2 = b1 + gap
        b2 = threshold_2 + F.softplus((fused_bounds[:, 2, :] - threshold_2) * steepness) / steepness

        threshold_3 = b2 + gap
        b3 = threshold_3 + F.softplus((fused_bounds[:, 3, :] - threshold_3) * steepness) / steepness

        fused_bounds = torch.stack([b0, b1, b2, b3], dim=1).clamp(0, 1)

        # Layer-specific physics refinement
        boundaries = self.layer_refine(fused_bounds, d1, x, x_compensated)

        # Final ordering
        gap_final = 0.005
        b0 = boundaries[:, 0, :]
        thresh_1 = b0 + gap_final
        b1 = thresh_1 + F.softplus((boundaries[:, 1, :] - thresh_1) * steepness) / steepness

        thresh_2 = b1 + gap_final
        b2 = thresh_2 + F.softplus((boundaries[:, 2, :] - thresh_2) * steepness) / steepness

        thresh_3 = b2 + gap_final
        b3 = thresh_3 + F.softplus((boundaries[:, 3, :] - thresh_3) * steepness) / steepness

        boundaries = torch.stack([b0, b1, b2, b3], dim=1).clamp(0, 1)

        result = {'boundaries': boundaries}

        if return_aux:
            result.update({
                'independent_bounds': independent_bounds,
                'thickness_bounds': thickness_bounds,
                'fused_bounds': fused_bounds,
                'blend_alpha': alpha,
                'global_alpha': global_alpha,
                'rnfl_thick': thicknesses['rnfl'],
                'inl_thick': thicknesses['inl'],
                'isos_thick': thicknesses['isos'],
                'depth_mu': self.depth_compensation.get_mu(),
                'refine_scales': self.layer_refine.refine_scales,
            })

        return result


# =============================================================================
# Enhanced Loss Function
# =============================================================================
class PhysicsLossV3(nn.Module):
    """
    Enhanced loss with depth-aware weighting and improved Fresnel.
    """

    REFRACTIVE_INDICES = [1.00, 1.36, 1.35, 1.40, 1.38]

    def __init__(
        self,
        lambda_position: float = 2.0,
        lambda_thickness: float = 1.5,
        lambda_dice: float = 2.5,  # Increased for better layer segmentation
        lambda_gradient: float = 0.4,
        lambda_intensity: float = 0.3,
        lambda_fresnel: float = 0.3,
        lambda_smooth: float = 0.2,
        lambda_order: float = 5.0,  # Strong penalty for boundary ordering violations
    ):
        super().__init__()

        self.lambda_position = lambda_position
        self.lambda_thickness = lambda_thickness
        self.lambda_order = lambda_order
        self.lambda_dice = lambda_dice
        self.lambda_gradient = lambda_gradient
        self.lambda_intensity = lambda_intensity
        self.lambda_fresnel = lambda_fresnel
        self.lambda_smooth = lambda_smooth

        self.gradient_module = BoundaryGradientModule()

        # Fresnel reflection coefficients
        fresnel_R = []
        for i in range(4):
            n1, n2 = self.REFRACTIVE_INDICES[i], self.REFRACTIVE_INDICES[i + 1]
            R = ((n1 - n2) / (n1 + n2)) ** 2
            fresnel_R.append(R)
        max_R = max(fresnel_R)
        fresnel_R = [r / max_R for r in fresnel_R]
        self.register_buffer('fresnel_weights', torch.tensor(fresnel_R))

        # Depth-aware boundary weights (deeper boundaries get higher weight)
        # b0 (ILM): 1.0, b1: 1.3, b2: 1.6, b3: 2.0
        depth_weights = torch.tensor([1.0, 1.3, 1.6, 2.0])
        self.register_buffer('depth_weights', depth_weights)

    def forward(self, outputs, gt_bounds, valid_mask, H, image=None, physics_warmup=1.0):
        B, _, W = gt_bounds.shape
        pred = outputs['boundaries']

        # 1. Depth-aware position loss
        position_loss = 0
        for i in range(4):
            diff = (pred[:, i, :] - gt_bounds[:, i, :]).abs()
            weighted_diff = (diff * valid_mask * self.depth_weights[i]).sum()
            position_loss = position_loss + weighted_diff / (valid_mask.sum() + 1e-8)
        position_loss = position_loss / 4

        # 2. Thickness loss with extra weight on INL
        gt_rnfl = gt_bounds[:, 1, :] - gt_bounds[:, 0, :]
        gt_inl = gt_bounds[:, 2, :] - gt_bounds[:, 1, :]
        gt_isos = gt_bounds[:, 3, :] - gt_bounds[:, 2, :]

        thickness_loss = (
            ((outputs['rnfl_thick'] - gt_rnfl).abs() * valid_mask).sum() +
            ((outputs['inl_thick'] - gt_inl).abs() * valid_mask).sum() * 2.0 +
            ((outputs['isos_thick'] - gt_isos).abs() * valid_mask).sum() * 1.5
        ) / (valid_mask.sum() * 3 + 1e-8)

        # 3. Soft Dice loss
        dice_loss = self._soft_dice_loss(pred, gt_bounds, H)

        # 4. Gradient alignment loss
        gradient_loss = torch.tensor(0.0, device=pred.device)
        if image is not None and self.lambda_gradient > 0 and physics_warmup > 0:
            gradient_loss = self._gradient_alignment_loss(pred, image, H)

        # 5. Layer intensity consistency
        intensity_loss = torch.tensor(0.0, device=pred.device)
        if image is not None and self.lambda_intensity > 0 and physics_warmup > 0:
            intensity_loss = self._intensity_consistency_loss(pred, image, H)

        # 6. Enhanced Fresnel loss
        fresnel_loss = torch.tensor(0.0, device=pred.device)
        if image is not None and self.lambda_fresnel > 0 and physics_warmup > 0:
            fresnel_loss = self._enhanced_fresnel_loss(pred, image, H)

        # 7. Boundary smoothness loss
        smooth_loss = self._smoothness_loss(pred)

        # 8. Monotonic ordering loss - ensure b0 < b1 < b2 < b3
        order_loss = self._ordering_loss(pred)

        # Total loss
        total_loss = (
            self.lambda_position * position_loss +
            self.lambda_thickness * thickness_loss +
            self.lambda_dice * dice_loss +
            physics_warmup * self.lambda_gradient * gradient_loss +
            physics_warmup * self.lambda_intensity * intensity_loss +
            physics_warmup * self.lambda_fresnel * fresnel_loss +
            self.lambda_smooth * smooth_loss +
            self.lambda_order * order_loss
        )

        # Stats
        with torch.no_grad():
            stats = self._compute_stats(outputs, gt_bounds, valid_mask, H)
            stats['dice_loss'] = dice_loss.item()
            stats['order_loss'] = order_loss.item()
            stats['gradient_loss'] = gradient_loss.item()
            stats['intensity_loss'] = intensity_loss.item()
            stats['fresnel_loss'] = fresnel_loss.item()
            stats['smooth_loss'] = smooth_loss.item()

        return total_loss, stats

    def _soft_dice_loss(self, pred_bounds, gt_bounds, H):
        """
        Improved soft Dice loss covering all 4 layer regions:
        - Layer 0 (RNFL_GCL): from b0 to b1
        - Layer 1 (INL_OPL_ONL): from b1 to b2
        - Layer 2 (IS_OS): from b2 to b3
        - Layer 3 (RPE_Choroid): from b3 to bottom (1.0)

        With layer-specific weighting to emphasize harder inner layers.
        """
        B, _, W = pred_bounds.shape
        y_coords = torch.linspace(0, 1, H, device=pred_bounds.device).view(1, H, 1)
        sigma = 2.0 / H

        # Layer weights: inner layers (RNFL, INL, IS_OS) are harder, give them more weight
        # RPE_Choroid is easier, give it less weight
        layer_weights = [2.0, 2.5, 2.0, 1.0]  # RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid

        total_dice_loss = 0
        total_weight = 0

        # Define layer boundaries: (top_boundary_idx, bottom_boundary_idx or None for 1.0)
        layer_defs = [
            (0, 1),      # RNFL_GCL: b0 to b1
            (1, 2),      # INL_OPL_ONL: b1 to b2
            (2, 3),      # IS_OS: b2 to b3
            (3, None),   # RPE_Choroid: b3 to bottom (1.0)
        ]

        for layer_idx, (top_idx, bot_idx) in enumerate(layer_defs):
            weight = layer_weights[layer_idx]

            # Predicted layer mask
            pred_top = pred_bounds[:, top_idx, :].unsqueeze(1)  # [B, 1, W]
            if bot_idx is not None:
                pred_bot = pred_bounds[:, bot_idx, :].unsqueeze(1)
            else:
                pred_bot = torch.ones_like(pred_top)  # Bottom of image

            pred_mask = (
                torch.sigmoid((y_coords - pred_top) / sigma) *
                torch.sigmoid((pred_bot - y_coords) / sigma)
            )  # [B, H, W]

            # Ground truth layer mask
            gt_top = gt_bounds[:, top_idx, :].unsqueeze(1)
            if bot_idx is not None:
                gt_bot = gt_bounds[:, bot_idx, :].unsqueeze(1)
            else:
                gt_bot = torch.ones_like(gt_top)

            gt_mask = (
                torch.sigmoid((y_coords - gt_top) / sigma) *
                torch.sigmoid((gt_bot - y_coords) / sigma)
            )

            # Dice coefficient
            intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)

            # Weighted dice loss
            total_dice_loss = total_dice_loss + weight * (1 - dice.mean())
            total_weight += weight

        return total_dice_loss / total_weight

    def _gradient_alignment_loss(self, pred_bounds, image, H):
        B, _, img_H, W = image.shape
        gradients = self.gradient_module(image)
        grad_magnitude = gradients.abs()
        grad_max = grad_magnitude.max() + 1e-8
        grad_normalized = grad_magnitude / grad_max

        # Use actual image height for proper dimension matching
        y_coords = torch.linspace(0, 1, img_H, device=image.device).view(1, 1, img_H, 1)
        sigma = 2.0 / img_H

        total_alignment = 0
        for i in range(4):
            bound_pos = pred_bounds[:, i, :].unsqueeze(1).unsqueeze(2)
            weights = torch.exp(-((y_coords - bound_pos) ** 2) / (2 * sigma ** 2))
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
            sampled_grad = (grad_normalized * weights).sum(dim=2)
            total_alignment = total_alignment + sampled_grad.mean()

        return 1.0 - total_alignment / 4

    def _intensity_consistency_loss(self, pred_bounds, image, H):
        B, _, img_H, W = image.shape
        # Use actual image height for y_coords, not the passed H parameter
        y_coords = torch.linspace(0, 1, img_H, device=image.device).view(1, img_H, 1)
        sigma = 2.0 / img_H

        # No need to resize - use image as-is
        img_2d = image.squeeze(1)  # [B, H, W]

        total_var = 0
        for top_idx, bot_idx in [(0, 1), (1, 2), (2, 3)]:
            pred_top = pred_bounds[:, top_idx, :].unsqueeze(1)
            pred_bot = pred_bounds[:, bot_idx, :].unsqueeze(1)

            layer_mask = (
                torch.sigmoid((y_coords - pred_top) / sigma) *
                torch.sigmoid((pred_bot - y_coords) / sigma)
            )

            masked_sum = (img_2d * layer_mask).sum(dim=(1, 2))
            mask_sum = layer_mask.sum(dim=(1, 2)) + 1e-8
            layer_mean = masked_sum / mask_sum

            diff_sq = (img_2d - layer_mean.view(B, 1, 1)) ** 2
            layer_var = (diff_sq * layer_mask).sum(dim=(1, 2)) / mask_sum

            total_var = total_var + layer_var.mean()

        return total_var / 3

    def _enhanced_fresnel_loss(self, pred_bounds, image, H):
        """
        Enhanced Fresnel loss that encourages:
        1. Correct relative gradient strengths at boundaries
        2. Boundaries to align with actual gradient peaks
        """
        B, _, img_H, W = image.shape
        gradients = self.gradient_module(image)
        grad_magnitude = gradients.abs()

        # Use actual image height for proper dimension matching
        y_coords = torch.linspace(0, 1, img_H, device=image.device).view(1, 1, img_H, 1)
        sigma = 2.0 / img_H

        boundary_strengths = []
        for i in range(4):
            bound_pos = pred_bounds[:, i, :].unsqueeze(1).unsqueeze(2)
            weights = torch.exp(-((y_coords - bound_pos) ** 2) / (2 * sigma ** 2))
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
            sampled_grad = (grad_magnitude * weights).sum(dim=2)
            boundary_strengths.append(sampled_grad.mean(dim=(1, 2)))

        observed = torch.stack(boundary_strengths, dim=1)
        observed_max = observed.max(dim=1, keepdim=True)[0] + 1e-8
        observed_norm = observed / observed_max

        expected = self.fresnel_weights.view(1, 4).expand(B, 4)
        loss = F.smooth_l1_loss(observed_norm, expected)

        return loss

    def _smoothness_loss(self, pred_bounds):
        """Penalize high-frequency oscillations in boundary predictions."""
        # First-order smoothness (adjacent column differences)
        diff1 = (pred_bounds[:, :, 1:] - pred_bounds[:, :, :-1]).abs()

        # Second-order smoothness (curvature)
        diff2 = (pred_bounds[:, :, 2:] - 2 * pred_bounds[:, :, 1:-1] + pred_bounds[:, :, :-2]).abs()

        return diff1.mean() + 0.5 * diff2.mean()

    def _ordering_loss(self, pred_bounds):
        """
        Penalize boundary ordering violations: enforce b0 < b1 < b2 < b3.

        Uses a soft margin loss that penalizes when boundaries get too close
        or cross each other. Minimum gap enforced between adjacent boundaries.
        """
        # Minimum gap between boundaries (in normalized [0,1] coordinates)
        # This ensures layers have minimum thickness
        min_gap = 0.02  # ~5px at 256px resolution

        total_violation = 0
        for i in range(3):
            # Gap between boundary i and i+1
            gap = pred_bounds[:, i+1, :] - pred_bounds[:, i, :]

            # Penalize when gap < min_gap (boundaries too close or crossed)
            # Using ReLU: penalty = max(0, min_gap - gap)
            violation = torch.relu(min_gap - gap)
            total_violation = total_violation + violation.mean()

        return total_violation / 3

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

        gt_inl = (gt_bounds[:, 2, :] - gt_bounds[:, 1, :]) * H
        pred_inl = outputs['inl_thick'] * H
        stats['INL_thick_mae'] = ((pred_inl - gt_inl).abs() * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8)

        alpha = outputs['global_alpha']
        stats['alpha_b0'] = alpha[0].item()
        stats['alpha_b2'] = alpha[2].item()

        stats['depth_mu'] = outputs['depth_mu'].item()

        return stats


# =============================================================================
# Utility Functions
# =============================================================================
def boundaries_to_segmentation(boundaries, H, num_classes=4):
    B, _, W = boundaries.shape
    device = boundaries.device

    # Avoid division issues when H=1
    H_scale = max(H - 1, 1)
    bounds_px = (boundaries * H_scale).long().clamp(0, H - 1)
    rows = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

    b0 = bounds_px[:, 0, :].unsqueeze(1)
    b1 = bounds_px[:, 1, :].unsqueeze(1)
    b2 = bounds_px[:, 2, :].unsqueeze(1)
    b3 = bounds_px[:, 3, :].unsqueeze(1)

    seg = torch.where(rows >= b0, 0, -1)
    seg = torch.where(rows >= b1, 1, seg)
    seg = torch.where(rows >= b2, 2, seg)
    seg = torch.where(rows >= b3, 3, seg)
    seg = seg.clamp(min=0)

    return seg


if __name__ == '__main__':
    model = PhysicsEnsembleV3(hidden_channels=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 1, 256, 256)
    out = model(x, return_aux=True)

    print(f"\nPhysics parameters:")
    print(f"  Depth attenuation (μ): {out['depth_mu'].item():.3f}")
    print(f"  Refinement scales: {out['refine_scales'].detach().numpy()}")

    alpha = out['global_alpha']
    print(f"\nBlend weights (α):")
    print(f"  b0: {alpha[0].item():.2f}, b1: {alpha[1].item():.2f}")
    print(f"  b2: {alpha[2].item():.2f}, b3: {alpha[3].item():.2f}")
