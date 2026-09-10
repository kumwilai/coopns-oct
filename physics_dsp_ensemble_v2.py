#!/usr/bin/env python3
"""
Physics-Enhanced Ensemble Model for OCT Boundary Detection.

Combines:
1. Independent boundaries (best absolute positioning)
2. Thickness-based boundaries (best thin layer accuracy)
3. Physics-based components:
   - Depth attenuation (Beer-Lambert law)
   - Boundary gradient alignment
   - Layer intensity priors
   - Physics-guided refinement
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
# Physics Modules
# =============================================================================
class DepthCompensation(nn.Module):
    """
    Compensate for signal attenuation with depth (Beer-Lambert law).

    In OCT, signal intensity decreases exponentially with depth:
    I(z) = I_0 * exp(-μ * z)

    We learn the attenuation coefficient and apply inverse compensation.
    """

    def __init__(self, mu_min=0.3, mu_max=0.95):
        super().__init__()
        # Learnable attenuation coefficient (μ)
        # Use sigmoid mapping to constrain to [mu_min, mu_max]
        # This prevents μ from collapsing to 0 or going to infinity
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_range = mu_max - mu_min

        # Initialize log_mu to produce μ ≈ 0.7 (middle of range)
        # sigmoid(0) = 0.5 → μ = 0.3 + 0.5 * 0.65 = 0.625
        self.log_mu = nn.Parameter(torch.tensor(0.0))

    def get_mu(self):
        """Get the clamped μ value."""
        return self.mu_min + self.mu_range * torch.sigmoid(self.log_mu)

    def forward(self, image):
        """
        Apply depth compensation to image.

        Args:
            image: [B, 1, H, W] OCT image

        Returns:
            compensated: [B, 1, H, W] depth-compensated image
        """
        B, C, H, W = image.shape

        # Create depth coordinate (0 at top, 1 at bottom)
        depth = torch.linspace(0, 1, H, device=image.device).view(1, 1, H, 1)

        # Attenuation coefficient (clamped to valid range)
        mu = self.get_mu()

        # Compensation factor: exp(μ * z) to counteract exp(-μ * z)
        compensation = torch.exp(mu * depth)

        # Apply compensation (with clipping for stability)
        compensated = image * compensation.clamp(max=5.0)

        return compensated


class BoundaryGradientModule(nn.Module):
    """
    Extract gradient information at boundaries.

    OCT boundaries occur at interfaces between tissues with different
    optical properties, resulting in intensity gradients.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels for gradient computation
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [0,  0,  0],
                                 [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, image):
        """
        Compute vertical gradients (boundaries are horizontal in OCT).

        Args:
            image: [B, 1, H, W] OCT image

        Returns:
            gradients: [B, 1, H, W] vertical gradient magnitude
        """
        # Compute vertical gradient
        grad_y = F.conv2d(image, self.sobel_y, padding=1)

        return grad_y


class PhysicsGuidedRefinement(nn.Module):
    """
    Refine boundary predictions using physics-based image features.

    Uses:
    1. Local intensity gradients
    2. Depth-compensated features (passed from parent)
    3. Expected layer properties
    """

    def __init__(self, feature_channels):
        super().__init__()

        self.gradient_module = BoundaryGradientModule()
        # NOTE: No DepthCompensation here - use the one from parent model to avoid
        # duplicate learnable parameters and redundant computation

        # Combine image features with physics features
        # Input: feature_channels + 2 (gradient + compensated intensity)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(feature_channels + 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 1),  # 4 boundary corrections
            nn.Tanh(),
        )

        # Learnable refinement scale (start small)
        self.refine_scale = nn.Parameter(torch.tensor(0.02))

    def forward(self, boundaries, features, image, compensated_image):
        """
        Refine boundaries using physics-based features.

        Args:
            boundaries: [B, 4, W] predicted boundaries
            features: [B, C, H, W] encoder features
            image: [B, 1, H, W] original image
            compensated_image: [B, 1, H, W] depth-compensated image (from parent)

        Returns:
            refined: [B, 4, W] refined boundaries
        """
        # Compute physics features
        gradients = self.gradient_module(image)  # [B, 1, H, W]
        # Use pre-computed compensated image from parent (avoids duplicate DepthCompensation)

        # Concatenate with learned features
        combined = torch.cat([features, gradients, compensated_image], dim=1)

        # Predict refinement
        refinement = self.refine_conv(combined)  # [B, 4, H, W]
        refinement = refinement.mean(dim=2)  # [B, 4, W]

        # Apply small correction
        refined = boundaries + self.refine_scale * refinement

        return refined.clamp(0, 1)


# =============================================================================
# Physics-Enhanced Ensemble Model
# =============================================================================
class PhysicsEnsemble(nn.Module):
    """
    Physics-enhanced ensemble model for OCT boundary detection.

    Architecture:
    1. Depth compensation preprocessing
    2. Shared encoder-decoder
    3. Branch 1: Independent boundaries
    4. Branch 2: Thickness-based boundaries
    5. Physics-guided fusion and refinement
    """

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

        # Physics preprocessing
        self.depth_compensation = DepthCompensation()

        # Shared encoder (takes original + compensated)
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

        # Branch 1: Independent boundary heads
        self.independent_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(H, H // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(H // 2, 1, 1),
                nn.Sigmoid(),
            ) for _ in range(4)
        ])

        # Branch 2: Thickness-based prediction
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
        # Initialize closer to 0 to avoid sigmoid saturation and improve gradient flow
        # sigmoid(1.0) = 0.73, sigmoid(0.0) = 0.5, sigmoid(-0.5) = 0.38
        self.blend_logits = nn.Parameter(torch.tensor([1.0, 0.0, -0.5, -0.5]))

        # Adaptive fusion
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(H, H // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(H // 4, 4, 1),
        )

        # Physics-guided refinement
        self.physics_refine = PhysicsGuidedRefinement(H)

    def _init_thickness_heads(self):
        for name, head in self.thickness_heads.items():
            final_conv = head[-1]
            target = self.INIT_THICKNESS[name] - self.MIN_THICKNESS[name]

            if target > 0.01:
                init_bias = math.log(math.exp(target) - 1)
            else:
                init_bias = math.log(max(target, 1e-6))

            # Initialize final layer to produce desired initial thickness
            nn.init.zeros_(final_conv.weight)
            nn.init.constant_(final_conv.bias, init_bias)

            # NOTE: Don't zero first_conv weights - this blocks gradients!
            # Use small random init instead for gradient flow
            first_conv = head[0]
            nn.init.normal_(first_conv.weight, mean=0, std=0.01)
            nn.init.zeros_(first_conv.bias)

    def forward(self, x, return_aux=False):
        B, _, H_img, W = x.shape

        # Physics preprocessing: depth compensation
        x_compensated = self.depth_compensation(x)

        # Concatenate original and compensated
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

        # Branch 1: Independent boundaries
        independent_bounds = []
        for head in self.independent_heads:
            boundary_map = head(d1)
            boundary = boundary_map.mean(dim=2).squeeze(1)
            independent_bounds.append(boundary)
        independent_bounds = torch.stack(independent_bounds, dim=1)

        # Branch 2: Thickness-based boundaries
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

        # Fusion: blend independent and thickness-based predictions
        # Global alpha (learnable, per-boundary): determines base blend ratio
        # Adaptive alpha (from features, per-pixel): allows local adjustment
        # 70% global + 30% adaptive to ensure blend_logits gets strong gradients
        global_alpha = torch.sigmoid(self.blend_logits)
        adaptive_logits = self.adaptive_fusion(d1)
        adaptive_alpha = torch.sigmoid(adaptive_logits.mean(dim=2))
        alpha = 0.7 * global_alpha.view(1, 4, 1) + 0.3 * adaptive_alpha

        fused_bounds = alpha * independent_bounds + (1 - alpha) * thickness_bounds

        # Ensure ordering using SOFT constraints (preserves gradients)
        # Using softplus approximation: max(a, b) ≈ b + softplus(a - b) / steepness
        steepness = 100.0  # Higher = sharper transition
        gap = 0.01  # Minimum gap between boundaries

        b0 = fused_bounds[:, 0, :]
        threshold_1 = b0 + gap
        b1 = threshold_1 + F.softplus((fused_bounds[:, 1, :] - threshold_1) * steepness) / steepness

        threshold_2 = b1 + gap
        b2 = threshold_2 + F.softplus((fused_bounds[:, 2, :] - threshold_2) * steepness) / steepness

        threshold_3 = b2 + gap
        b3 = threshold_3 + F.softplus((fused_bounds[:, 3, :] - threshold_3) * steepness) / steepness

        fused_bounds = torch.stack([b0, b1, b2, b3], dim=1).clamp(0, 1)

        # Physics-guided refinement (pass pre-computed compensated image)
        boundaries = self.physics_refine(fused_bounds, d1, x, x_compensated)

        # Final ordering enforcement using SOFT constraints
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
                'refine_scale': self.physics_refine.refine_scale,
                # NOTE: Removed 'features': d1 to prevent memory leak
                # Storing full feature map keeps computation graph alive
            })

        return result


# =============================================================================
# Physics-Enhanced Loss Function
# =============================================================================
class PhysicsEnsembleLoss(nn.Module):
    """
    Loss function with physics-based components.

    Components:
    1. Boundary position loss
    2. Thickness loss
    3. Soft Dice loss
    4. Boundary gradient alignment loss (physics)
    5. Layer intensity consistency loss (physics)
    6. Fresnel reflection consistency loss (physics)
    """

    # Refractive indices for retinal layers (from literature)
    # Order: [above_ILM, RNFL, INL_OPL_ONL, IS_OS, RPE_Choroid]
    REFRACTIVE_INDICES = [1.00, 1.36, 1.35, 1.40, 1.38]

    def __init__(
        self,
        lambda_position: float = 2.0,
        lambda_thickness: float = 1.5,
        lambda_dice: float = 1.0,
        lambda_gradient: float = 0.5,
        lambda_intensity: float = 0.3,
        lambda_fresnel: float = 0.2,
    ):
        super().__init__()

        self.lambda_position = lambda_position
        self.lambda_thickness = lambda_thickness
        self.lambda_dice = lambda_dice
        self.lambda_gradient = lambda_gradient
        self.lambda_intensity = lambda_intensity
        self.lambda_fresnel = lambda_fresnel

        self.gradient_module = BoundaryGradientModule()

        # Precompute expected Fresnel reflection strengths at each boundary
        # R = ((n1 - n2) / (n1 + n2))^2
        fresnel_R = []
        for i in range(4):
            n1, n2 = self.REFRACTIVE_INDICES[i], self.REFRACTIVE_INDICES[i + 1]
            R = ((n1 - n2) / (n1 + n2)) ** 2
            fresnel_R.append(R)
        # Normalize to relative strengths (max = 1.0)
        max_R = max(fresnel_R)
        fresnel_R = [r / max_R for r in fresnel_R]
        self.register_buffer('fresnel_weights', torch.tensor(fresnel_R))

        # Learnable boundary weights
        self.log_boundary_weights = nn.Parameter(torch.tensor([0.0, 0.0, 0.5, 0.0]))

    def forward(self, outputs, gt_bounds, valid_mask, H, image=None, physics_warmup=1.0):
        """
        Compute loss with optional physics warmup.

        Args:
            physics_warmup: Scale factor for physics losses (0.0 to 1.0).
                           Use 0.0 at start, gradually increase to 1.0.
        """
        B, _, W = gt_bounds.shape

        pred = outputs['boundaries']
        boundary_weights = F.softmax(self.log_boundary_weights, dim=0) * 4

        # 1. Position loss
        position_loss = self._boundary_loss(pred, gt_bounds, valid_mask, boundary_weights)

        # 2. Thickness loss
        gt_rnfl = gt_bounds[:, 1, :] - gt_bounds[:, 0, :]
        gt_inl = gt_bounds[:, 2, :] - gt_bounds[:, 1, :]
        gt_isos = gt_bounds[:, 3, :] - gt_bounds[:, 2, :]

        thickness_loss = (
            ((outputs['rnfl_thick'] - gt_rnfl).abs() * valid_mask).sum() +
            ((outputs['inl_thick'] - gt_inl).abs() * valid_mask).sum() * 2.0 +
            ((outputs['isos_thick'] - gt_isos).abs() * valid_mask).sum()
        ) / (valid_mask.sum() * 3 + 1e-8)

        # 3. Soft Dice loss
        dice_loss = self._soft_dice_loss(pred, gt_bounds, H)

        # 4. Gradient alignment loss (physics) - scaled by warmup
        gradient_loss = torch.tensor(0.0, device=pred.device)
        if image is not None and self.lambda_gradient > 0 and physics_warmup > 0:
            gradient_loss = self._gradient_alignment_loss(pred, image, H)

        # 5. Layer intensity consistency (physics) - scaled by warmup
        intensity_loss = torch.tensor(0.0, device=pred.device)
        if image is not None and self.lambda_intensity > 0 and physics_warmup > 0:
            intensity_loss = self._intensity_consistency_loss(pred, image, H)

        # 6. Fresnel reflection consistency (physics) - scaled by warmup
        fresnel_loss = torch.tensor(0.0, device=pred.device)
        if image is not None and self.lambda_fresnel > 0 and physics_warmup > 0:
            fresnel_loss = self._fresnel_consistency_loss(pred, image, H)

        # Total loss (physics losses scaled by warmup factor)
        total_loss = (
            self.lambda_position * position_loss +
            self.lambda_thickness * thickness_loss +
            self.lambda_dice * dice_loss +
            physics_warmup * self.lambda_gradient * gradient_loss +
            physics_warmup * self.lambda_intensity * intensity_loss +
            physics_warmup * self.lambda_fresnel * fresnel_loss
        )

        # Stats
        with torch.no_grad():
            stats = self._compute_stats(outputs, gt_bounds, valid_mask, H)
            stats['gradient_loss'] = gradient_loss.item()
            stats['intensity_loss'] = intensity_loss.item()
            stats['fresnel_loss'] = fresnel_loss.item()

        return total_loss, stats

    def _boundary_loss(self, pred, gt, valid_mask, weights):
        loss = 0
        for i in range(4):
            diff = (pred[:, i, :] - gt[:, i, :]).abs()
            loss = loss + (diff * valid_mask * weights[i]).sum() / (valid_mask.sum() + 1e-8)
        return loss / 4

    def _soft_dice_loss(self, pred_bounds, gt_bounds, H):
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

    def _gradient_alignment_loss(self, pred_bounds, image, H):
        """
        Encourage boundaries to align with strong intensity gradients.

        At true boundaries, there should be strong vertical gradients
        due to tissue interfaces.

        Uses differentiable soft sampling instead of .long() to maintain gradient flow.

        Returns a POSITIVE loss that decreases when boundaries align with gradients.
        """
        B, _, W = pred_bounds.shape

        # Compute image gradients
        gradients = self.gradient_module(image)  # [B, 1, H, W]
        grad_magnitude = gradients.abs()

        # Normalize gradient magnitude to [0, 1] range for stable loss
        grad_max = grad_magnitude.max() + 1e-8
        grad_normalized = grad_magnitude / grad_max

        # Create y-coordinate grid for soft sampling
        y_coords = torch.linspace(0, 1, H, device=image.device).view(1, 1, H, 1)  # [1, 1, H, 1]

        # Use soft Gaussian sampling instead of hard indexing to maintain gradient flow
        sigma = 2.0 / H  # Sampling width

        total_alignment = 0
        for i in range(4):
            # Get boundary positions (normalized 0-1)
            bound_pos = pred_bounds[:, i, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, W]

            # Soft attention weights centered at boundary position
            # Higher weight where y_coord is close to boundary
            weights = torch.exp(-((y_coords - bound_pos) ** 2) / (2 * sigma ** 2))  # [B, 1, H, W]
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)  # Normalize

            # Differentiable sampling: weighted sum of gradients
            sampled_grad = (grad_normalized * weights).sum(dim=2)  # [B, 1, W]

            # Accumulate alignment score (higher = better)
            total_alignment = total_alignment + sampled_grad.mean()

        # Convert to loss: 1 - alignment (so minimizing loss maximizes alignment)
        # This keeps loss in [0, 1] range instead of large negative values
        return 1.0 - total_alignment / 4

    def _intensity_consistency_loss(self, pred_bounds, image, H):
        """
        Encourage consistent intensity within layers.

        Each layer should have relatively uniform intensity.
        Penalize high variance within predicted layers.
        """
        B, _, img_H, W = image.shape

        # Create soft layer masks
        y_coords = torch.linspace(0, 1, H, device=image.device).view(1, H, 1)
        sigma = 2.0 / H

        # Resize image ONCE outside the loop (more efficient)
        if img_H != H:
            img_resized = F.interpolate(image, size=(H, W), mode='bilinear', align_corners=True)
        else:
            img_resized = image
        img_resized = img_resized.squeeze(1)  # [B, H, W]

        total_var = 0
        for top_idx, bot_idx in [(0, 1), (1, 2), (2, 3)]:
            pred_top = pred_bounds[:, top_idx, :].unsqueeze(1)
            pred_bot = pred_bounds[:, bot_idx, :].unsqueeze(1)

            layer_mask = (
                torch.sigmoid((y_coords - pred_top) / sigma) *
                torch.sigmoid((pred_bot - y_coords) / sigma)
            )  # [B, H, W]

            # Compute weighted mean and variance within layer
            masked_sum = (img_resized * layer_mask).sum(dim=(1, 2))
            mask_sum = layer_mask.sum(dim=(1, 2)) + 1e-8
            layer_mean = masked_sum / mask_sum

            # Variance
            diff_sq = (img_resized - layer_mean.view(B, 1, 1)) ** 2
            layer_var = (diff_sq * layer_mask).sum(dim=(1, 2)) / mask_sum

            total_var = total_var + layer_var.mean()

        return total_var / 3

    def _fresnel_consistency_loss(self, pred_bounds, image, H):
        """
        Fresnel reflection consistency loss.

        At tissue interfaces, the reflected signal strength depends on
        refractive index mismatch (Fresnel equations). Boundaries with
        larger refractive index differences should have stronger gradients.

        This loss encourages:
        - IS/OS boundary (largest Δn) to have strongest gradient
        - ILM (air/tissue) to have strong gradient
        - INL boundaries (small Δn) to have weaker gradients

        Loss = Σ_i |observed_strength_i - expected_strength_i|
        where expected_strength is based on Fresnel coefficients.
        """
        B, _, W = pred_bounds.shape

        # Compute image gradients
        gradients = self.gradient_module(image)  # [B, 1, H, W]
        grad_magnitude = gradients.abs()

        # Create y-coordinate grid for soft sampling
        y_coords = torch.linspace(0, 1, H, device=image.device).view(1, 1, H, 1)
        sigma = 2.0 / H

        # Sample gradient strength at each boundary
        boundary_strengths = []
        for i in range(4):
            bound_pos = pred_bounds[:, i, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, W]

            # Soft attention weights centered at boundary position
            weights = torch.exp(-((y_coords - bound_pos) ** 2) / (2 * sigma ** 2))
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

            # Sample gradient at boundary
            sampled_grad = (grad_magnitude * weights).sum(dim=2)  # [B, 1, W]
            boundary_strengths.append(sampled_grad.mean(dim=(1, 2)))  # [B]

        # Stack and normalize observed strengths
        observed = torch.stack(boundary_strengths, dim=1)  # [B, 4]
        observed_max = observed.max(dim=1, keepdim=True)[0] + 1e-8
        observed_norm = observed / observed_max

        # Expected relative strengths from Fresnel coefficients
        expected = self.fresnel_weights.view(1, 4).expand(B, 4)

        # Loss: difference between observed and expected relative strengths
        # Use smooth L1 for robustness
        loss = F.smooth_l1_loss(observed_norm, expected)

        return loss

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
        stats['refine_scale'] = outputs['refine_scale'].item()

        return stats

    def get_learned_weights(self):
        return {
            'boundary_weights': (F.softmax(self.log_boundary_weights, dim=0) * 4).detach().cpu().numpy(),
        }


# =============================================================================
# Utility Functions
# =============================================================================
def boundaries_to_segmentation(boundaries, H, num_classes=4):
    """
    Convert boundary positions to segmentation mask.

    Classes:
    - 0: RNFL_GCL (between b0 and b1)
    - 1: INL_OPL_ONL (between b1 and b2)
    - 2: IS_OS (between b2 and b3)
    - 3: RPE_Choroid (below b3)
    - Background (above b0) also gets 0, but ground truth doesn't have class 0
    """
    B, _, W = boundaries.shape
    device = boundaries.device

    bounds_px = (boundaries * (H - 1)).long()
    rows = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

    b0 = bounds_px[:, 0, :].unsqueeze(1)
    b1 = bounds_px[:, 1, :].unsqueeze(1)
    b2 = bounds_px[:, 2, :].unsqueeze(1)
    b3 = bounds_px[:, 3, :].unsqueeze(1)

    # Build segmentation using scalars (more efficient than tensor multiplication)
    # Start with class 0 for everything >= b0 (RNFL layer)
    # Then override with higher class indices for deeper layers
    seg = torch.where(rows >= b0, 0, -1)  # -1 for background above ILM
    seg = torch.where(rows >= b1, 1, seg)
    seg = torch.where(rows >= b2, 2, seg)
    seg = torch.where(rows >= b3, 3, seg)

    # Convert -1 to 0 (background treated same as RNFL for Dice since GT has no class 0)
    seg = seg.clamp(min=0)

    return seg


if __name__ == '__main__':
    model = PhysicsEnsemble(hidden_channels=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 1, 256, 256)
    out = model(x, return_aux=True)

    print(f"\nPhysics parameters:")
    print(f"  Depth attenuation (μ): {out['depth_mu'].item():.3f}")
    print(f"  Refinement scale: {out['refine_scale'].item():.4f}")

    alpha = out['global_alpha']
    print(f"\nBlend weights (α):")
    print(f"  b0: {alpha[0].item():.2f}, b1: {alpha[1].item():.2f}")
    print(f"  b2: {alpha[2].item():.2f}, b3: {alpha[3].item():.2f}")

    print(f"\nThickness predictions:")
    print(f"  RNFL: {out['rnfl_thick'].mean().item() * 256:.1f}px")
    print(f"  INL: {out['inl_thick'].mean().item() * 256:.1f}px")
    print(f"  ISOS: {out['isos_thick'].mean().item() * 256:.1f}px")
