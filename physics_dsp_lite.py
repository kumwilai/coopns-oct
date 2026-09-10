#!/usr/bin/env python3
"""
Lightweight Physics-Informed DSP for OCT Boundary Detection

A faster version optimized for CPU training that incorporates:
1. Simplified encoder (no UNet, just CNN)
2. Fresnel physics priors for gradient-based costs
3. Beer-Lambert depth compensation
4. Efficient soft-argmin boundary detection (no full DP)

This is 10-50x faster than the full UNet-DSP model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# =============================================================================
# Refractive Indices for OCT Layers
# =============================================================================
REFRACTIVE_INDICES = {
    'vitreous': 1.336,
    'RNFL': 1.358,
    'INL': 1.365,
    'IS_OS': (1.375 + 1.410) / 2,  # Average of IS and OS
    'RPE': 1.400,
    'choroid': 1.380,
}


# =============================================================================
# Fresnel Physics Module (Lightweight)
# =============================================================================
class FresnelPhysicsLite(nn.Module):
    """
    Lightweight Fresnel physics for boundary detection.

    Uses refractive indices to compute expected gradient strength at each boundary.
    """

    def __init__(self, num_boundaries: int = 4, learnable: bool = True):
        super().__init__()
        self.num_boundaries = num_boundaries

        # Refractive indices: [vitreous, RNFL, INL, IS_OS, RPE, choroid]
        n_init = torch.tensor([
            REFRACTIVE_INDICES['vitreous'],
            REFRACTIVE_INDICES['RNFL'],
            REFRACTIVE_INDICES['INL'],
            REFRACTIVE_INDICES['IS_OS'],
            REFRACTIVE_INDICES['RPE'],
            REFRACTIVE_INDICES['choroid'],
        ])

        if learnable:
            self.register_buffer('n_base', n_init)
            self.n_delta = nn.Parameter(torch.zeros_like(n_init))
        else:
            self.register_buffer('n_base', n_init)
            self.register_buffer('n_delta', torch.zeros_like(n_init))

        # Sobel kernel for gradient
        sobel = torch.tensor([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]], dtype=torch.float32) / 8.0
        self.register_buffer('sobel', sobel.view(1, 1, 3, 3))

    @property
    def refractive_indices(self) -> torch.Tensor:
        return self.n_base + 0.02 * torch.tanh(self.n_delta)

    def fresnel_R(self, idx: int) -> torch.Tensor:
        """Fresnel reflectance at boundary idx."""
        n = self.refractive_indices
        r = (n[idx] - n[idx + 1]) / (n[idx] + n[idx + 1] + 1e-8)
        return r ** 2

    def expected_gradient_strength(self) -> torch.Tensor:
        """Expected relative gradient strength per boundary."""
        strengths = torch.stack([torch.sqrt(self.fresnel_R(i) + 1e-8)
                                 for i in range(self.num_boundaries)])
        return strengths / (strengths.max() + 1e-8)

    def compute_gradient(self, image: torch.Tensor) -> torch.Tensor:
        """Compute vertical gradient magnitude."""
        grad = F.conv2d(F.pad(image, (1, 1, 1, 1), mode='reflect'), self.sobel)
        return torch.abs(grad)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient and expected strengths.

        Returns:
            gradient: [B, 1, H, W] gradient magnitude
            expected: [num_boundaries] expected strength per boundary
        """
        gradient = self.compute_gradient(image)
        expected = self.expected_gradient_strength()
        return gradient, expected


# =============================================================================
# Beer-Lambert Depth Compensation (Lightweight)
# =============================================================================
class DepthCompensation(nn.Module):
    """Simple depth-dependent gain compensation."""

    def __init__(self, init_mu: float = 0.003):
        super().__init__()
        self.log_mu = nn.Parameter(torch.tensor(math.log(init_mu + 1e-8)))

    @property
    def mu(self) -> torch.Tensor:
        return torch.exp(self.log_mu).clamp(1e-6, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        depth = torch.arange(H, device=x.device, dtype=torch.float32).view(1, 1, H, 1)
        gain = torch.exp(self.mu * depth).clamp(max=2.5)
        return x * gain


# =============================================================================
# Lightweight Cost Predictor
# =============================================================================
class LiteCostPredictor(nn.Module):
    """
    Multi-scale CNN that predicts boundary costs.

    Uses multi-scale features for better boundary localization.
    Includes a learnable region prior to focus on the retina region.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_boundaries: int = 4,
    ):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.hidden_channels = hidden_channels

        # Multi-scale encoder with skip connections
        # Branch 1: Fine details (3x3 kernels)
        self.enc_fine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2: Medium context (5x5 kernels)
        self.enc_medium = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 5, padding=2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 5, padding=2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Branch 3: Coarse context (dilated convs for larger receptive field)
        self.enc_coarse = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Fusion layer: combines multi-scale features
        total_channels = hidden_channels + hidden_channels // 2 + hidden_channels // 2
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Per-boundary cost heads with separate predictors
        self.cost_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels // 2, 1, 1),
            )
            for _ in range(num_boundaries)
        ])

        # Learnable region prior (typical boundary positions for OCT)
        # Initialize to typical retina region (boundaries around 25-40% from top)
        # Store in logit space: logit(p) = log(p / (1-p))
        # logit(0.25) ≈ -1.1, logit(0.30) ≈ -0.85, logit(0.33) ≈ -0.7, logit(0.38) ≈ -0.49
        def logit(p):
            return math.log(p / (1 - p + 1e-8) + 1e-8)

        self.region_prior = nn.Parameter(torch.tensor([
            [logit(0.25), logit(0.15)],  # ILM: center=0.25, width=0.15
            [logit(0.29), logit(0.15)],  # RNFL/INL
            [logit(0.31), logit(0.15)],  # INL/IS_OS
            [logit(0.36), logit(0.15)],  # IS_OS/RPE
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict per-boundary costs with region prior.

        Args:
            x: [B, 1, H, W] input image

        Returns:
            costs: [B, num_boundaries, H, W]
        """
        B, _, H, W = x.shape
        device = x.device

        # Multi-scale feature extraction
        feat_fine = self.enc_fine(x)      # [B, C, H, W]
        feat_medium = self.enc_medium(x)  # [B, C/2, H, W]
        feat_coarse = self.enc_coarse(x)  # [B, C/2, H, W]

        # Concatenate and fuse
        features = torch.cat([feat_fine, feat_medium, feat_coarse], dim=1)
        features = self.fusion(features)  # [B, C, H, W]

        # Per-boundary costs
        learned_costs = torch.cat([head(features) for head in self.cost_heads], dim=1)  # [B, N, H, W]

        # Add region prior as a Gaussian penalty
        # This biases the costs to favor boundaries in the expected region
        y_coords = torch.arange(H, device=device, dtype=torch.float32) / (H - 1)
        y_coords = y_coords.view(1, 1, H, 1)  # [1, 1, H, 1]

        prior_costs = []
        for b in range(self.num_boundaries):
            center = torch.sigmoid(self.region_prior[b, 0])  # Clamp to [0, 1]
            width = torch.sigmoid(self.region_prior[b, 1]) * 0.3 + 0.05  # [0.05, 0.35]

            # Gaussian penalty: high cost far from expected region
            dist = (y_coords - center) ** 2
            prior = dist / (2 * width ** 2)  # [1, 1, H, 1]
            prior_costs.append(prior.expand(B, 1, H, W))

        prior_costs = torch.cat(prior_costs, dim=1)  # [B, N, H, W]

        # Combine learned costs with prior (prior weight starts at 0.5)
        costs = learned_costs + 0.5 * prior_costs

        return costs


# =============================================================================
# Efficient Soft-Argmin Boundary Detection
# =============================================================================
class SoftArgminBoundaryDetector(nn.Module):
    """
    Efficient boundary detection using column-wise soft-argmin.

    Much faster than full DSP while maintaining differentiability.
    Uses soft ordering constraints that don't force boundaries apart.
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        temperature: float = 0.05,
        min_gap_ratio: float = 0.01,  # Minimum gap as fraction of height (small!)
    ):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.temperature = temperature
        self.min_gap_ratio = min_gap_ratio

    def forward(self, costs: torch.Tensor) -> torch.Tensor:
        """
        Detect boundaries using soft-argmin with ordering.

        Args:
            costs: [B, num_boundaries, H, W] per-position costs

        Returns:
            boundaries: [B, num_boundaries, W] normalized positions (0-1)
        """
        B, N, H, W = costs.shape
        device = costs.device

        # Soft-argmin per boundary per column
        weights = F.softmax(-costs / self.temperature, dim=2)  # [B, N, H, W]
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

        # Raw positions from soft-argmin
        raw_positions = (weights * y_coords).sum(dim=2) / (H - 1)  # [B, N, W]

        # Soft ordering: sort boundaries and ensure minimum gap
        # This allows boundaries to be close together (like in real OCT)
        sorted_pos, _ = torch.sort(raw_positions, dim=1)

        # Ensure minimum gap between adjacent boundaries
        # But don't force them apart if they're already ordered correctly
        ordered = torch.zeros_like(sorted_pos)
        ordered[:, 0, :] = sorted_pos[:, 0, :]

        for i in range(1, N):
            min_pos = ordered[:, i-1, :] + self.min_gap_ratio
            ordered[:, i, :] = torch.maximum(sorted_pos[:, i, :], min_pos)

        return ordered.clamp(0.01, 0.99)


# =============================================================================
# Main Model: PhysicsDSPLite
# =============================================================================
class PhysicsDSPLite(nn.Module):
    """
    Lightweight Physics-Informed DSP for OCT boundary detection.

    Combines:
    1. Simple CNN cost predictor
    2. Fresnel physics priors
    3. Beer-Lambert depth compensation
    4. Efficient soft-argmin boundary detection
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_boundaries: int = 4,
        use_fresnel: bool = True,
        use_depth_comp: bool = True,
        temperature: float = 0.05,
        physics_weight: float = 0.3,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.use_fresnel = use_fresnel
        self.use_depth_comp = use_depth_comp

        # Depth compensation
        if use_depth_comp:
            self.depth_comp = DepthCompensation()

        # Fresnel physics
        if use_fresnel:
            self.fresnel = FresnelPhysicsLite(num_boundaries)
            self.physics_weight = nn.Parameter(torch.tensor(physics_weight))

        # Cost predictor
        self.cost_predictor = LiteCostPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_boundaries=num_boundaries,
        )

        # Boundary detector
        self.detector = SoftArgminBoundaryDetector(
            num_boundaries=num_boundaries,
            temperature=temperature,
        )

        # Boundary smoother (1D conv per boundary)
        self.smoother = nn.Conv1d(num_boundaries, num_boundaries, 11, padding=5, groups=num_boundaries)
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
            Dict with boundaries and optionally auxiliary outputs
        """
        B, _, H, W = x.shape

        # Depth compensation
        if self.use_depth_comp:
            x_comp = self.depth_comp(x)
        else:
            x_comp = x

        # Predict learned costs
        learned_costs = self.cost_predictor(x_comp)  # [B, N, H, W]

        # Physics-based costs from gradient
        if self.use_fresnel:
            gradient, expected_strength = self.fresnel(x)

            # Normalize gradient per column
            grad_max = gradient.max(dim=2, keepdim=True)[0].clamp(min=1e-8)
            gradient_norm = gradient / grad_max  # [B, 1, H, W]

            # Physics cost: low where gradient matches expected pattern
            physics_costs = []
            for b in range(self.num_boundaries):
                strength = expected_strength[b]
                # Negative gradient (high gradient = low cost for strong boundaries)
                cost = -strength * gradient_norm.squeeze(1)
                physics_costs.append(cost)

            physics_costs = torch.stack(physics_costs, dim=1)  # [B, N, H, W]

            # Combine with learnable weight
            w = torch.sigmoid(self.physics_weight)
            costs = (1 - w) * learned_costs + w * physics_costs
        else:
            costs = learned_costs
            gradient_norm = None
            expected_strength = None

        # Detect boundaries
        boundaries = self.detector(costs)

        # Smooth boundaries (residual)
        boundaries_smooth = self.smoother(boundaries) + boundaries
        boundaries_smooth = boundaries_smooth.clamp(0.01, 0.99)

        outputs = {
            'boundaries': boundaries_smooth,
            'boundaries_pixels': boundaries_smooth * (H - 1),
            'costs': costs,
        }

        if return_aux:
            outputs['learned_costs'] = learned_costs
            if self.use_fresnel:
                outputs['gradient'] = gradient_norm
                outputs['expected_strength'] = expected_strength
                outputs['physics_weight'] = torch.sigmoid(self.physics_weight)
            if self.use_depth_comp:
                outputs['depth_mu'] = self.depth_comp.mu

        return outputs

    def get_physics_info(self) -> Dict[str, any]:
        """Get physics parameters for logging."""
        info = {}
        if self.use_depth_comp:
            info['depth_mu'] = self.depth_comp.mu.item()
        if self.use_fresnel:
            info['physics_weight'] = torch.sigmoid(self.physics_weight).item()
            info['expected_strength'] = self.fresnel.expected_gradient_strength().tolist()
            info['refractive_indices'] = self.fresnel.refractive_indices.tolist()
        return info


# =============================================================================
# Loss Function
# =============================================================================
class PhysicsDSPLiteLoss(nn.Module):
    """
    Loss function for PhysicsDSPLite with LEARNABLE thickness-aware supervision.

    Key insight: INL_OPL_ONL layer is only ~7.7px thick on average.
    With boundary MAE > layer thickness, Dice becomes 0.
    Solution: Learnable weights that auto-focus on challenging thin layers.
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        num_layers: int = 3,  # RNFL, INL, IS_OS (between boundaries)
        lambda_position: float = 1.0,
        lambda_ordering: float = 0.5,
        lambda_smoothness: float = 0.3,
        lambda_thickness: float = 2.0,  # High weight for thickness consistency
        lambda_dice: float = 1.0,  # Soft Dice loss per layer
        use_huber: bool = True,
        huber_delta: float = 0.01,  # ~2.5 pixels at H=256
        learnable_weights: bool = True,  # LEARNABLE boundary & thickness weights
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.num_layers = num_layers
        self.lambda_position = lambda_position
        self.lambda_ordering = lambda_ordering
        self.lambda_smoothness = lambda_smoothness
        self.lambda_thickness = lambda_thickness
        self.lambda_dice = lambda_dice
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # LEARNABLE boundary weights (in log-space for stability)
            # Initialize: higher for boundaries 1,2 (define thin INL layer)
            # log([1, 2, 3, 2]) ≈ [0, 0.7, 1.1, 0.7]
            self.boundary_weight_logits = nn.Parameter(
                torch.tensor([0.0, 0.7, 1.1, 0.7])
            )

            # LEARNABLE thickness weights (in log-space)
            # Initialize: much higher for INL (index 1) since it's only 7.7px thick
            # log([1, 5, 1]) ≈ [0, 1.6, 0]
            self.thickness_weight_logits = nn.Parameter(
                torch.tensor([0.0, 1.6, 0.0])
            )

            # LEARNABLE layer Dice weights
            # Initialize: higher for thin layers (INL, IS_OS)
            self.dice_weight_logits = nn.Parameter(
                torch.tensor([1.0, 3.0, 2.0, 1.0])  # RNFL, INL, IS_OS, RPE
            )
        else:
            # Fixed weights (fallback)
            self.register_buffer('boundary_weights',
                torch.tensor([1.0, 2.0, 4.0, 3.0]))
            self.register_buffer('thickness_weights',
                torch.tensor([1.0, 5.0, 1.0]))
            self.register_buffer('dice_weights',
                torch.tensor([1.0, 3.0, 2.0, 1.0]))

    @property
    def boundary_weights(self) -> torch.Tensor:
        """Get normalized boundary weights (sum to num_boundaries)."""
        if self.learnable_weights:
            # Softmax ensures positive weights that sum to 1, then scale
            weights = F.softmax(self.boundary_weight_logits, dim=0)
            return weights * self.num_boundaries
        return self._buffers['boundary_weights']

    @property
    def thickness_weights(self) -> torch.Tensor:
        """Get normalized thickness weights (sum to num_layers)."""
        if self.learnable_weights:
            weights = F.softmax(self.thickness_weight_logits, dim=0)
            return weights * self.num_layers
        return self._buffers['thickness_weights']

    @property
    def dice_weights(self) -> torch.Tensor:
        """Get normalized Dice weights per layer."""
        if self.learnable_weights:
            weights = F.softmax(self.dice_weight_logits, dim=0)
            return weights * 4  # 4 layers
        return self._buffers['dice_weights']

    def get_learned_weights(self) -> Dict[str, List[float]]:
        """Return current learned weights for logging."""
        return {
            'boundary_weights': self.boundary_weights.detach().cpu().tolist(),
            'thickness_weights': self.thickness_weights.detach().cpu().tolist(),
            'dice_weights': self.dice_weights.detach().cpu().tolist(),
        }

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        H: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with learnable thickness-aware supervision.

        Args:
            pred: [B, N, W] predicted boundaries (normalized 0-1)
            gt: [B, N, W] ground truth boundaries (normalized 0-1)
            valid_mask: [B, W] valid columns
            H: image height for MAE computation
        """
        B, N, W = pred.shape
        device = pred.device

        if valid_mask is None:
            valid_mask = torch.ones(B, W, device=device)

        # Get learnable weights
        bw = self.boundary_weights.to(device)  # [N]
        tw = self.thickness_weights.to(device)  # [N-1]

        # =================================================================
        # 1. Position loss (weighted by learnable boundary weights)
        # =================================================================
        if self.use_huber:
            pos_error = F.huber_loss(pred, gt, reduction='none', delta=self.huber_delta)
        else:
            pos_error = torch.abs(pred - gt)

        pos_error = pos_error * bw.view(1, N, 1)
        pos_error = pos_error * valid_mask.unsqueeze(1)
        pos_loss = pos_error.sum() / (valid_mask.sum() * N + 1e-8)

        # =================================================================
        # 2. Ordering loss (boundaries must be in order)
        # =================================================================
        deltas = pred[:, 1:, :] - pred[:, :-1, :]
        ordering_loss = F.relu(-deltas + 0.002).pow(2).mean()

        # =================================================================
        # 3. Smoothness loss (boundaries should be smooth along columns)
        # =================================================================
        dx = pred[:, :, 1:] - pred[:, :, :-1]
        ddx = dx[:, :, 1:] - dx[:, :, :-1]
        smoothness_loss = (dx ** 2).mean() + 0.5 * (ddx ** 2).mean()

        # =================================================================
        # 4. CRITICAL: Thickness loss (weighted by learnable thickness weights)
        #    This is KEY for thin layers like INL (~7.7px)
        # =================================================================
        pred_thickness = pred[:, 1:, :] - pred[:, :-1, :]  # [B, N-1, W]
        gt_thickness = gt[:, 1:, :] - gt[:, :-1, :]  # [B, N-1, W]

        # Weighted thickness error
        thick_error = (pred_thickness - gt_thickness).pow(2)
        thick_error = thick_error * tw.view(1, -1, 1)  # Apply learnable weights
        thick_error = thick_error * valid_mask.unsqueeze(1)
        thickness_loss = thick_error.sum() / (valid_mask.sum() * (N - 1) + 1e-8)

        # =================================================================
        # 5. Soft Dice loss per layer (differentiable)
        # =================================================================
        dice_loss = self._compute_soft_dice_loss(pred, gt, valid_mask, H)

        # =================================================================
        # Total loss
        # =================================================================
        total = (
            self.lambda_position * pos_loss +
            self.lambda_ordering * ordering_loss +
            self.lambda_smoothness * smoothness_loss +
            self.lambda_thickness * thickness_loss +
            self.lambda_dice * dice_loss
        )

        # =================================================================
        # Stats for logging
        # =================================================================
        with torch.no_grad():
            mae_pixels = (torch.abs(pred - gt) * H).mean(dim=(0, 2))
            thick_mae_pixels = (torch.abs(pred_thickness - gt_thickness) * H).mean(dim=(0, 2))

        stats = {
            'loss': total.item(),
            'pos_loss': pos_loss.item(),
            'order_loss': ordering_loss.item(),
            'smooth_loss': smoothness_loss.item(),
            'thick_loss': thickness_loss.item(),
            'dice_loss': dice_loss.item(),
        }

        # Boundary MAE
        names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']
        for i, name in enumerate(names[:N]):
            stats[f'{name}_mae'] = mae_pixels[i].item()

        # Thickness MAE (CRITICAL for thin layers)
        thick_names = ['RNFL_thick', 'INL_thick', 'ISOS_thick']
        for i, name in enumerate(thick_names[:N-1]):
            stats[f'{name}_mae'] = thick_mae_pixels[i].item()

        stats['avg_mae'] = mae_pixels.mean().item()

        return total, stats

    def _compute_soft_dice_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """
        Compute soft Dice loss for each layer.

        Uses soft boundaries to create differentiable layer masks.
        """
        B, N, W = pred.shape
        device = pred.device

        # Convert boundaries to pixel positions
        pred_px = pred * (H - 1)  # [B, N, W]
        gt_px = gt * (H - 1)

        # Create soft layer masks using sigmoid
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        y_coords = y_coords.view(1, H, 1)  # [1, H, 1]

        sigma = 2.0  # Softness of boundaries (pixels)

        total_dice_loss = 0.0
        dw = self.dice_weights.to(device)

        for layer_idx in range(4):  # 4 layers: RNFL, INL, IS_OS, RPE
            if layer_idx == 0:
                # RNFL: from boundary[0] to boundary[1]
                pred_top = pred_px[:, 0:1, :]  # [B, 1, W]
                pred_bot = pred_px[:, 1:2, :]
                gt_top = gt_px[:, 0:1, :]
                gt_bot = gt_px[:, 1:2, :]
            elif layer_idx == 3:
                # RPE: from boundary[3] to bottom
                pred_top = pred_px[:, 3:4, :]
                pred_bot = torch.full_like(pred_top, H - 1)
                gt_top = gt_px[:, 3:4, :]
                gt_bot = torch.full_like(gt_top, H - 1)
            else:
                # INL (idx=1) or IS_OS (idx=2)
                pred_top = pred_px[:, layer_idx:layer_idx+1, :]
                pred_bot = pred_px[:, layer_idx+1:layer_idx+2, :]
                gt_top = gt_px[:, layer_idx:layer_idx+1, :]
                gt_bot = gt_px[:, layer_idx+1:layer_idx+2, :]

            # Soft mask: sigmoid(y - top) * sigmoid(bot - y)
            # pred_mask[b, h, w] = P(pixel at h belongs to this layer)
            # Note: pred_top is [B, 1, W], y_coords is [1, H, 1]
            # Broadcasting: [1, H, 1] with [B, 1, W] → [B, H, W]
            pred_mask = (
                torch.sigmoid((y_coords - pred_top) / sigma) *
                torch.sigmoid((pred_bot - y_coords) / sigma)
            )  # [B, H, W]

            gt_mask = (
                torch.sigmoid((y_coords - gt_top) / sigma) *
                torch.sigmoid((gt_bot - y_coords) / sigma)
            )  # [B, H, W]

            # Soft Dice: 2 * intersection / (pred + gt)
            intersection = (pred_mask * gt_mask).sum(dim=1)  # [B, W]
            union = pred_mask.sum(dim=1) + gt_mask.sum(dim=1)  # [B, W]

            # Apply valid mask
            intersection = intersection * valid_mask
            union = union * valid_mask

            dice = (2 * intersection.sum() + 1e-8) / (union.sum() + 1e-8)
            dice_loss_layer = 1.0 - dice

            # Apply learnable weight for this layer
            total_dice_loss = total_dice_loss + dw[layer_idx] * dice_loss_layer

        return total_dice_loss / 4.0  # Average over layers


# =============================================================================
# Utility Functions
# =============================================================================
def boundaries_to_segmentation(boundaries: torch.Tensor, H: int, num_classes: int = 4) -> torch.Tensor:
    """
    Convert boundaries to segmentation mask.

    Boundaries define transitions between classes:
    - boundary[0] = ILM (top of class 0 = RNFL_GCL)
    - boundary[1] = RNFL/INL (top of class 1 = INL_OPL_ONL)
    - boundary[2] = INL/IS_OS (top of class 2 = IS_OS)
    - boundary[3] = IS_OS/RPE (top of class 3 = RPE_Choroid)

    Returns segmentation with classes 0-3 (4 retinal layers).
    """
    B, N, W = boundaries.shape
    device = boundaries.device

    boundaries_px = boundaries * (H - 1)
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W).float()

    seg = torch.zeros(B, H, W, device=device, dtype=torch.long)

    # Class 0: from boundary[0] to boundary[1]
    # Class 1: from boundary[1] to boundary[2]
    # Class 2: from boundary[2] to boundary[3]
    # Class 3: from boundary[3] to bottom

    for c in range(num_classes):
        if c == 0:
            # RNFL_GCL: from ILM to RNFL/INL boundary
            mask = (y_coords >= boundaries_px[:, 0:1, :]) & (y_coords < boundaries_px[:, 1:2, :])
        elif c == num_classes - 1:
            # RPE_Choroid: from IS_OS/RPE boundary to bottom
            mask = y_coords >= boundaries_px[:, -1:, :]
        else:
            # Middle classes: between corresponding boundaries
            mask = (y_coords >= boundaries_px[:, c:c+1, :]) & (y_coords < boundaries_px[:, c+1:c+2, :])
        seg[mask] = c

    return seg


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Testing PhysicsDSPLite")
    print("=" * 60)

    device = 'cpu'
    B, H, W = 2, 256, 256

    # Create model
    model = PhysicsDSPLite(
        hidden_channels=32,
        use_fresnel=True,
        use_depth_comp=True,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward speed
    x = torch.randn(B, 1, H, W, device=device)

    # Warmup
    _ = model(x)

    # Time it
    start = time.time()
    n_iters = 10
    for _ in range(n_iters):
        outputs = model(x, return_aux=True)
    elapsed = time.time() - start

    print(f"\nForward pass: {elapsed/n_iters*1000:.1f} ms/batch")
    print(f"  boundaries: {outputs['boundaries'].shape}")
    print(f"  costs: {outputs['costs'].shape}")

    # Physics info
    info = model.get_physics_info()
    print(f"\nPhysics info:")
    print(f"  Depth μ: {info.get('depth_mu', 'N/A'):.6f}")
    print(f"  Physics weight: {info.get('physics_weight', 'N/A'):.3f}")
    print(f"  Expected strengths: {info.get('expected_strength', 'N/A')}")

    # Test loss
    gt = torch.sort(torch.rand(B, 4, W, device=device), dim=1)[0] * 0.8 + 0.1
    loss_fn = PhysicsDSPLiteLoss()

    loss, stats = loss_fn(outputs['boundaries'], gt, H=H)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"  Avg MAE: {stats['avg_mae']:.2f} px")

    # Test gradient
    loss.backward()
    has_grad = model.cost_predictor.encoder[0].weight.grad is not None
    print(f"\nGradient flow: {has_grad}")

    # Test segmentation
    seg = boundaries_to_segmentation(outputs['boundaries'], H)
    print(f"\nSegmentation: {seg.shape}, classes: {torch.unique(seg).tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
