#!/usr/bin/env python3
"""
Physics-Informed UNet-DSP for OCT Boundary Detection

Novel architecture combining:
1. UNet encoder for multi-scale feature extraction
2. Rayleigh-aware noise normalization (OCT speckle is Rayleigh, not Gaussian)
3. Beer-Lambert depth compensation (deeper layers have weaker signal)
4. Fresnel gradient physics (refractive index transitions create boundaries)
5. Hybrid cost decoder combining learned + physics costs
6. Differentiable shortest path with ordering constraints

Key Physics:
- Fresnel: R = ((n1 - n2) / (n1 + n2))² determines boundary gradient strength
- Rayleigh: OCT amplitude ~ Rayleigh(σ), not Gaussian
- Beer-Lambert: I(z) = I₀ × exp(-μz) causes depth-dependent attenuation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# =============================================================================
# Refractive Indices for OCT Layers (from literature)
# =============================================================================
REFRACTIVE_INDICES = {
    'vitreous': 1.336,
    'RNFL': 1.358,
    'GCL': 1.358,
    'INL': 1.365,
    'OPL': 1.365,
    'ONL': 1.360,
    'IS': 1.375,
    'OS': 1.410,  # Highest - strong IS/OS boundary
    'RPE': 1.400,
    'choroid': 1.380,
}


# =============================================================================
# UNet Encoder with Rayleigh-Aware Normalization
# =============================================================================
class RayleighAwareEncoder(nn.Module):
    """
    UNet-style encoder with Rayleigh noise normalization.

    Key insight: OCT speckle follows Rayleigh distribution with spatially
    varying σ. By estimating σ locally and normalizing features, we make
    the encoder robust to noise variations across different layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        use_noise_estimation: bool = True,
    ):
        super().__init__()

        self.use_noise_estimation = use_noise_estimation
        self.num_levels = num_levels

        # Local noise (σ) estimator
        if use_noise_estimation:
            self.noise_estimator = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Softplus(),  # Ensure positive σ
            )
            # Initialize to output ~0.1 (reasonable noise level)
            nn.init.constant_(self.noise_estimator[-2].bias, -2.0)

        # Encoder levels
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch_in = in_channels
        for level in range(num_levels):
            ch_out = base_channels * (2 ** level)

            self.encoders.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ))

            if level < num_levels - 1:
                self.downsamplers.append(nn.MaxPool2d(2))

            ch_in = ch_out

        self.out_channels = ch_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with noise normalization.

        Args:
            x: [B, 1, H, W] input image

        Returns:
            features: [B, C, H', W'] bottleneck features
            skip_features: List of skip connection features
            sigma_map: [B, 1, H, W] estimated noise map (if enabled)
        """
        # Estimate local noise level
        sigma_map = None
        if self.use_noise_estimation:
            sigma_map = self.noise_estimator(x) + 0.01  # Minimum σ
            # Normalize input by local σ (Rayleigh-aware normalization)
            x_normalized = x / (sigma_map + 1e-8)
        else:
            x_normalized = x

        # Encoder with skip connections
        skip_features = []
        h = x_normalized

        for level in range(self.num_levels):
            h = self.encoders[level](h)
            skip_features.append(h)

            if level < self.num_levels - 1:
                h = self.downsamplers[level](h)

        return h, skip_features, sigma_map


# =============================================================================
# Beer-Lambert Depth Compensation
# =============================================================================
class BeerLambertCompensation(nn.Module):
    """
    Compensates for depth-dependent signal attenuation in OCT.

    Physics: I(z) = I₀ × exp(-μz) where μ is the attenuation coefficient.

    To compensate, we multiply by exp(+μz), which amplifies deeper signals.
    The attenuation coefficient μ is learnable to adapt to different tissues.
    """

    def __init__(
        self,
        init_mu: float = 0.005,
        learnable: bool = True,
        max_gain: float = 3.0,
    ):
        """
        Args:
            init_mu: Initial attenuation coefficient (per pixel)
            learnable: If True, μ is learnable
            max_gain: Maximum gain factor to prevent explosion
        """
        super().__init__()

        self.max_gain = max_gain

        if learnable:
            # Use log-space for stability
            self.log_mu = nn.Parameter(torch.tensor(math.log(init_mu + 1e-8)))
        else:
            self.register_buffer('log_mu', torch.tensor(math.log(init_mu + 1e-8)))

    @property
    def mu(self) -> torch.Tensor:
        """Get attenuation coefficient (always positive)."""
        return torch.exp(self.log_mu).clamp(1e-6, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply depth compensation.

        Args:
            x: [B, C, H, W] input features

        Returns:
            x_compensated: [B, C, H, W] depth-compensated features
        """
        B, C, H, W = x.shape
        device = x.device

        # Create depth coordinate (0 at top, H-1 at bottom)
        depth = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

        # Compensation gain: exp(μ × depth)
        gain = torch.exp(self.mu * depth)

        # Clamp gain to prevent explosion at bottom of image
        gain = gain.clamp(max=self.max_gain)

        return x * gain


# =============================================================================
# Fresnel Physics Cost Module
# =============================================================================
class FresnelPhysicsCost(nn.Module):
    """
    Physics-based boundary cost using Fresnel equations.

    Key insight: At layer boundaries, refractive index changes cause
    Fresnel reflections. The reflectance R = ((n1-n2)/(n1+n2))² determines
    the expected gradient strength at each boundary.

    This module computes:
    1. Expected gradient strength per boundary (from Fresnel)
    2. Actual image gradients (multi-scale)
    3. Physics cost: penalize positions where gradient doesn't match expected
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        learnable_indices: bool = True,
        gradient_scales: List[int] = [1, 2, 4],
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.gradient_scales = gradient_scales

        # Initialize refractive indices from literature
        # Layers: [vitreous, RNFL, INL, IS_OS, RPE, choroid]
        n_init = torch.tensor([
            REFRACTIVE_INDICES['vitreous'],
            REFRACTIVE_INDICES['RNFL'],
            REFRACTIVE_INDICES['INL'],
            (REFRACTIVE_INDICES['IS'] + REFRACTIVE_INDICES['OS']) / 2,
            REFRACTIVE_INDICES['RPE'],
            REFRACTIVE_INDICES['choroid'],
        ])

        if learnable_indices:
            self.register_buffer('n_base', n_init)
            self.n_delta = nn.Parameter(torch.zeros_like(n_init))
        else:
            self.register_buffer('n_base', n_init)
            self.register_buffer('n_delta', torch.zeros_like(n_init))

        # Learnable weight for physics contribution
        self.physics_weight = nn.Parameter(torch.tensor(0.3))

        # Per-boundary learnable scaling
        self.boundary_scale = nn.Parameter(torch.ones(num_boundaries))

        # Gradient kernels for each scale
        for scale in gradient_scales:
            kernel = self._create_gradient_kernel(scale)
            self.register_buffer(f'grad_kernel_{scale}', kernel)

    def _create_gradient_kernel(self, scale: int) -> torch.Tensor:
        """Create vertical gradient kernel at given scale."""
        size = 2 * scale + 1

        # Gaussian-weighted Sobel-like kernel
        sigma = scale / 2.0 + 0.5
        x = torch.arange(size, dtype=torch.float32) - size // 2
        gaussian = torch.exp(-x**2 / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()

        # Vertical derivative
        deriv = torch.arange(size, dtype=torch.float32) - size // 2
        deriv = deriv / (deriv.abs().max() + 1e-8)

        # 2D kernel: gaussian in x, derivative in y
        kernel = deriv.view(-1, 1) * gaussian.view(1, -1)
        kernel = kernel / (kernel.abs().sum() + 1e-8)

        return kernel.view(1, 1, size, size)

    @property
    def refractive_indices(self) -> torch.Tensor:
        """Get current refractive indices with learned perturbation."""
        return self.n_base + 0.05 * torch.tanh(self.n_delta)

    def fresnel_reflectance(self, boundary_idx: int) -> torch.Tensor:
        """Compute Fresnel reflectance at boundary: R = ((n1-n2)/(n1+n2))²."""
        n = self.refractive_indices
        n1 = n[boundary_idx]
        n2 = n[boundary_idx + 1]
        r = (n1 - n2) / (n1 + n2 + 1e-8)
        return r ** 2

    def expected_gradient_strength(self) -> torch.Tensor:
        """
        Compute expected relative gradient strength for each boundary.

        Gradient strength ∝ √R (amplitude from intensity reflectance).
        """
        strengths = []
        for b in range(self.num_boundaries):
            R = self.fresnel_reflectance(b)
            strengths.append(torch.sqrt(R + 1e-8))

        strengths = torch.stack(strengths)

        # Normalize to [0, 1]
        strengths = strengths / (strengths.max() + 1e-8)

        # Apply learnable per-boundary scaling
        strengths = strengths * torch.abs(self.boundary_scale)

        return strengths

    def compute_multiscale_gradient(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude at multiple scales.

        Args:
            image: [B, 1, H, W] input image

        Returns:
            gradient: [B, 1, H, W] combined gradient magnitude
        """
        gradients = []

        for scale in self.gradient_scales:
            kernel = getattr(self, f'grad_kernel_{scale}')
            pad = kernel.shape[-1] // 2

            grad = F.conv2d(F.pad(image, (pad, pad, pad, pad), mode='reflect'), kernel)
            gradients.append(torch.abs(grad))

        # Combine scales (learnable would be better, but start simple)
        combined = sum(gradients) / len(gradients)

        return combined

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics-based cost volumes.

        Args:
            image: [B, 1, H, W] input OCT image

        Returns:
            physics_costs: [B, num_boundaries, H, W] physics-based costs
            aux: Dictionary with auxiliary outputs
        """
        B, _, H, W = image.shape
        device = image.device

        # Compute image gradient
        gradient = self.compute_multiscale_gradient(image)  # [B, 1, H, W]

        # Normalize gradient per column (A-scan normalization)
        grad_max = gradient.max(dim=2, keepdim=True)[0].clamp(min=1e-8)
        gradient_norm = gradient / grad_max

        # Expected gradient strength per boundary
        expected_strength = self.expected_gradient_strength()  # [num_boundaries]

        # Physics cost: how well does each position match expected gradient?
        # Low cost = gradient matches expected pattern
        # High cost = gradient doesn't match

        physics_costs = []
        for b in range(self.num_boundaries):
            expected = expected_strength[b]

            # Cost: negative correlation with expected gradient
            # Strong gradient positions have low cost for boundaries expecting strong gradients
            cost = -expected * gradient_norm.squeeze(1)  # [B, H, W]

            # Add slight preference for being near strong gradients
            # (boundaries should be AT gradient peaks, not nearby)
            cost = cost + 0.1 * (1 - gradient_norm.squeeze(1))

            physics_costs.append(cost)

        physics_costs = torch.stack(physics_costs, dim=1)  # [B, num_boundaries, H, W]

        # Scale by learnable physics weight
        physics_weight = torch.sigmoid(self.physics_weight)
        physics_costs = physics_costs * physics_weight

        aux = {
            'gradient': gradient,
            'gradient_norm': gradient_norm,
            'expected_strength': expected_strength,
            'physics_weight': physics_weight,
            'refractive_indices': self.refractive_indices,
        }

        return physics_costs, aux


# =============================================================================
# UNet Decoder with Skip Connections
# =============================================================================
class UNetDecoder(nn.Module):
    """
    UNet decoder that reconstructs spatial resolution with skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_levels: int = 4,
        out_channels: int = 64,
    ):
        super().__init__()

        self.num_levels = num_levels

        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in range(num_levels - 1, 0, -1):
            ch_in = base_channels * (2 ** level)
            ch_skip = base_channels * (2 ** (level - 1))
            ch_out = ch_skip

            self.upsamplers.append(nn.ConvTranspose2d(ch_in, ch_in, 2, stride=2))

            self.decoders.append(nn.Sequential(
                nn.Conv2d(ch_in + ch_skip, ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ))

        # Final projection
        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode with skip connections.

        Args:
            x: [B, C, H', W'] bottleneck features
            skip_features: List of encoder skip features (finest to coarsest)

        Returns:
            out: [B, out_channels, H, W] decoded features at original resolution
        """
        # Reverse skip features (coarsest to finest, excluding bottleneck)
        skips = skip_features[:-1][::-1]

        h = x
        for i, (upsampler, decoder, skip) in enumerate(zip(
            self.upsamplers, self.decoders, skips
        )):
            h = upsampler(h)

            # Handle size mismatch
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        return self.final(h)


# =============================================================================
# Hybrid Cost Decoder (Learned + Physics)
# =============================================================================
class HybridCostDecoder(nn.Module):
    """
    Combines learned costs from UNet with physics-based Fresnel costs.

    Final cost = learned_cost + λ × physics_cost

    where λ is learnable per boundary.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        num_boundaries: int = 4,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries

        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Per-boundary cost heads
        self.cost_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels // 2, 1, 1),
            )
            for _ in range(num_boundaries)
        ])

        # Learnable combination weights (one per boundary)
        # Initialized to give equal weight to learned and physics
        self.combination_weights = nn.Parameter(torch.zeros(num_boundaries))

    def forward(
        self,
        features: torch.Tensor,
        physics_costs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined learned + physics costs.

        Args:
            features: [B, C, H, W] UNet features
            physics_costs: [B, num_boundaries, H, W] physics costs

        Returns:
            costs: [B, num_boundaries, H, W] combined costs
        """
        B, _, H, W = features.shape

        # Shared processing
        shared = self.shared(features)

        # Per-boundary learned costs
        learned_costs = []
        for head in self.cost_heads:
            cost = head(shared)  # [B, 1, H, W]
            learned_costs.append(cost.squeeze(1))

        learned_costs = torch.stack(learned_costs, dim=1)  # [B, num_boundaries, H, W]

        # Combination weights (sigmoid to keep in [0, 1])
        weights = torch.sigmoid(self.combination_weights).view(1, -1, 1, 1)

        # Combined: (1 - w) × learned + w × physics
        # When w = 0.5, equal contribution
        combined = (1 - weights) * learned_costs + weights * physics_costs

        return combined


# =============================================================================
# Differentiable Shortest Path
# =============================================================================
class DifferentiableShortestPath(nn.Module):
    """
    Finds optimal boundaries through cost volumes using differentiable DP.

    Key features:
    1. Soft-min for differentiable path selection
    2. Ordering constraints (boundaries must be top-to-bottom)
    3. Minimum gap between adjacent boundaries
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        temperature: float = 0.1,
        min_gap: int = 5,
        smoothness_weight: float = 1.0,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.temperature = temperature
        self.min_gap = min_gap
        self.smoothness_weight = smoothness_weight

        # Smoothness penalty for jumps
        max_jump = 20
        self.register_buffer(
            'smoothness_penalty',
            torch.arange(max_jump, dtype=torch.float32) ** 2 * smoothness_weight
        )

    def soft_min(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Differentiable soft-minimum."""
        return -self.temperature * torch.logsumexp(-x / self.temperature, dim=dim)

    def forward(self, costs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find optimal boundaries.

        Args:
            costs: [B, num_boundaries, H, W] per-pixel costs

        Returns:
            boundaries: [B, num_boundaries, W] normalized positions (0-1)
            path_costs: [B, num_boundaries, W] accumulated costs
        """
        B, N, H, W = costs.shape
        device = costs.device

        boundaries = []
        path_costs = []
        prev_boundary = None

        for b in range(N):
            boundary_cost = costs[:, b, :, :]  # [B, H, W]

            # Apply ordering constraint
            if prev_boundary is not None:
                y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
                min_valid = (prev_boundary.unsqueeze(1) + self.min_gap).clamp(max=H - 1)
                invalid_mask = y_coords < min_valid
                boundary_cost = boundary_cost.masked_fill(invalid_mask, 1e6)

            # Find optimal path using soft DP
            boundary_pos, path_cost = self._soft_dp(boundary_cost)

            boundaries.append(boundary_pos)
            path_costs.append(path_cost)

            prev_boundary = boundary_pos.detach() * H

        boundaries = torch.stack(boundaries, dim=1)
        path_costs = torch.stack(path_costs, dim=1)

        return boundaries, path_costs

    def _soft_dp(self, cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft dynamic programming for one boundary."""
        B, H, W = cost.shape
        device = cost.device

        # Pre-compute jump penalty matrix
        h_indices = torch.arange(H, device=device)
        jump_distances = torch.abs(h_indices.unsqueeze(0) - h_indices.unsqueeze(1))
        jump_distances = jump_distances.clamp(max=len(self.smoothness_penalty) - 1)
        jump_penalty_matrix = self.smoothness_penalty[jump_distances]

        # DP
        dp = torch.full((B, H, W), float('inf'), device=device)
        dp[:, :, 0] = cost[:, :, 0]

        for w in range(1, W):
            prev_costs = dp[:, :, w-1]
            transition_costs = prev_costs.unsqueeze(1) + jump_penalty_matrix.T.unsqueeze(0)
            min_prev_costs = self.soft_min(transition_costs, dim=2)
            dp[:, :, w] = min_prev_costs + cost[:, :, w]

        # Handle inf
        finite_costs = torch.where(torch.isinf(dp), torch.full_like(dp, 1e6), dp)

        # Soft-argmin
        weights = F.softmax(-finite_costs / self.temperature, dim=1)
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
        boundary_positions = (weights * y_coords).sum(dim=1) / (H - 1)
        path_cost = self.soft_min(finite_costs, dim=1)

        return boundary_positions, path_cost


# =============================================================================
# Main Model: PhysicsInformedDSPModel
# =============================================================================
class PhysicsInformedDSPModel(nn.Module):
    """
    Physics-Informed UNet-DSP for OCT Boundary Detection.

    Combines:
    1. UNet encoder with Rayleigh-aware noise normalization
    2. Beer-Lambert depth compensation
    3. Fresnel physics-based cost priors
    4. Hybrid cost decoder (learned + physics)
    5. Differentiable shortest path with ordering constraints
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
        num_boundaries: int = 4,
        use_noise_estimation: bool = True,
        use_depth_compensation: bool = True,
        use_fresnel_physics: bool = True,
        temperature: float = 0.1,
        min_gap: int = 5,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.use_depth_compensation = use_depth_compensation
        self.use_fresnel_physics = use_fresnel_physics

        # Encoder
        self.encoder = RayleighAwareEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            use_noise_estimation=use_noise_estimation,
        )

        # Beer-Lambert compensation
        if use_depth_compensation:
            self.depth_compensation = BeerLambertCompensation(
                init_mu=0.005,
                learnable=True,
            )

        # Decoder
        self.decoder = UNetDecoder(
            in_channels=self.encoder.out_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            out_channels=base_channels,
        )

        # Fresnel physics
        if use_fresnel_physics:
            self.fresnel_physics = FresnelPhysicsCost(
                num_boundaries=num_boundaries,
                learnable_indices=True,
            )

        # Cost decoder
        self.cost_decoder = HybridCostDecoder(
            in_channels=base_channels,
            hidden_channels=base_channels,
            num_boundaries=num_boundaries,
        )

        # DSP
        self.dsp = DifferentiableShortestPath(
            num_boundaries=num_boundaries,
            temperature=temperature,
            min_gap=min_gap,
        )

        # Boundary smoother
        self.smoother = nn.Sequential(
            nn.Conv1d(num_boundaries, num_boundaries * 2, 11, padding=5, groups=num_boundaries),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_boundaries * 2, num_boundaries, 11, padding=5, groups=num_boundaries),
        )
        nn.init.zeros_(self.smoother[-1].weight)
        nn.init.zeros_(self.smoother[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, 1, H, W] input OCT image
            return_aux: If True, return auxiliary outputs

        Returns:
            Dict with:
                boundaries: [B, num_boundaries, W] normalized positions (0-1)
                boundaries_pixels: [B, num_boundaries, W] positions in pixels
                costs: [B, num_boundaries, H, W] cost volumes
                sigma_map: [B, 1, H, W] noise estimation (if enabled)
        """
        B, C, H, W = x.shape

        # Encode with noise estimation
        bottleneck, skip_features, sigma_map = self.encoder(x)

        # Depth compensation
        if self.use_depth_compensation:
            bottleneck = self.depth_compensation(bottleneck)
            skip_features = [self.depth_compensation(f) for f in skip_features]

        # Decode
        features = self.decoder(bottleneck, skip_features)

        # Physics costs
        if self.use_fresnel_physics:
            physics_costs, physics_aux = self.fresnel_physics(x)
        else:
            physics_costs = torch.zeros(B, self.num_boundaries, H, W, device=x.device)
            physics_aux = {}

        # Combined costs
        costs = self.cost_decoder(features, physics_costs)

        # DSP
        boundaries, path_costs = self.dsp(costs)

        # Smooth boundaries
        boundaries_smooth = self.smoother(boundaries) + boundaries

        # Enforce ordering
        boundaries_ordered = self._enforce_ordering(boundaries_smooth)
        boundaries_ordered = boundaries_ordered.clamp(0.01, 0.99)

        outputs = {
            'boundaries': boundaries_ordered,
            'boundaries_pixels': boundaries_ordered * (H - 1),
            'costs': costs,
            'path_costs': path_costs,
        }

        if sigma_map is not None:
            outputs['sigma_map'] = sigma_map

        if return_aux:
            outputs['features'] = features
            outputs['physics_aux'] = physics_aux
            if self.use_depth_compensation:
                outputs['attenuation_mu'] = self.depth_compensation.mu

        return outputs

    def _enforce_ordering(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Enforce boundaries are in ascending order."""
        B, N, W = boundaries.shape

        min_gap = 0.015
        min_first = 0.01

        delta_first = boundaries[:, 0:1, :].clamp(min=min_first)
        delta_rest = boundaries[:, 1:, :] - boundaries[:, :-1, :]
        delta_rest = torch.maximum(delta_rest, torch.full_like(delta_rest, min_gap))

        deltas = torch.cat([delta_first, delta_rest], dim=1)
        ordered = torch.cumsum(deltas, dim=1)

        max_val = ordered[:, -1:, :]
        needs_renorm = (max_val > 0.99) | (max_val < 0.5)
        scale = torch.where(needs_renorm, 0.95 / (max_val + 1e-8), torch.ones_like(max_val))

        return ordered * scale

    def get_physics_summary(self) -> Dict[str, any]:
        """Get summary of physics parameters for logging."""
        summary = {
            'use_depth_compensation': self.use_depth_compensation,
            'use_fresnel_physics': self.use_fresnel_physics,
        }

        if self.use_depth_compensation:
            summary['attenuation_mu'] = self.depth_compensation.mu.item()

        if self.use_fresnel_physics:
            summary['refractive_indices'] = self.fresnel_physics.refractive_indices.tolist()
            summary['expected_gradient_strength'] = self.fresnel_physics.expected_gradient_strength().tolist()
            summary['physics_weight'] = torch.sigmoid(self.fresnel_physics.physics_weight).item()

        # Cost combination weights
        weights = torch.sigmoid(self.cost_decoder.combination_weights)
        summary['cost_combination_weights'] = weights.tolist()

        return summary


# =============================================================================
# Loss Function
# =============================================================================
class PhysicsInformedDSPLoss(nn.Module):
    """
    Loss function for PhysicsInformedDSPModel.

    Combines:
    1. Position loss (L1 between predicted and GT boundaries)
    2. Cost supervision (encourage low cost at GT positions)
    3. Ordering loss (penalize violations)
    4. Smoothness loss (penalize discontinuities)
    5. Physics consistency loss (match Fresnel gradient patterns)
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        lambda_position: float = 1.0,
        lambda_cost: float = 0.5,
        lambda_ordering: float = 0.1,
        lambda_smoothness: float = 0.5,
        lambda_physics: float = 0.2,
        boundary_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.num_boundaries = num_boundaries
        self.lambda_position = lambda_position
        self.lambda_cost = lambda_cost
        self.lambda_ordering = lambda_ordering
        self.lambda_smoothness = lambda_smoothness
        self.lambda_physics = lambda_physics

        if boundary_weights is None:
            # Clinical importance: ILM, RNFL/INL, INL/IS_OS (critical), IS_OS/RPE (critical)
            boundary_weights = [1.0, 2.0, 5.0, 4.0]

        self.register_buffer('boundary_weights', torch.tensor(boundary_weights))

    def forward(
        self,
        pred_boundaries: torch.Tensor,
        gt_boundaries: torch.Tensor,
        costs: Optional[torch.Tensor] = None,
        physics_aux: Optional[Dict] = None,
        valid_mask: Optional[torch.Tensor] = None,
        H: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            pred_boundaries: [B, num_boundaries, W] predicted (normalized 0-1)
            gt_boundaries: [B, num_boundaries, W] ground truth (normalized 0-1)
            costs: [B, num_boundaries, H, W] cost volumes
            physics_aux: Auxiliary physics outputs
            valid_mask: [B, W] valid columns mask
            H: Image height

        Returns:
            loss: Total loss
            stats: Dictionary of components
        """
        B, N, W = pred_boundaries.shape
        device = pred_boundaries.device

        if valid_mask is None:
            valid_mask = torch.ones(B, W, device=device)

        # 1. Position loss (weighted L1)
        pos_error = torch.abs(pred_boundaries - gt_boundaries)
        pos_error = pos_error * self.boundary_weights.view(1, N, 1)
        pos_error = pos_error * valid_mask.unsqueeze(1)
        pos_loss = pos_error.sum() / (valid_mask.sum() * N + 1e-8)

        # 2. Cost supervision
        cost_loss = torch.tensor(0.0, device=device)
        if costs is not None:
            gt_pixels = (gt_boundaries * (H - 1)).long().clamp(0, H - 1)

            # Vectorized gather
            costs_flat = costs.view(B * N, H, W)
            gt_flat = gt_pixels.view(B * N, W)
            batch_idx = torch.arange(B * N, device=device).view(-1, 1).expand(-1, W)
            col_idx = torch.arange(W, device=device).view(1, -1).expand(B * N, -1)

            gt_costs = costs_flat[batch_idx, gt_flat, col_idx].view(B, N, W)
            cost_loss = (gt_costs * valid_mask.unsqueeze(1)).mean()

        # 3. Ordering loss
        deltas = pred_boundaries[:, 1:, :] - pred_boundaries[:, :-1, :]
        ordering_loss = F.relu(-deltas).mean()

        # 4. Smoothness loss
        dx = pred_boundaries[:, :, 1:] - pred_boundaries[:, :, :-1]
        smoothness_loss = (dx ** 2).mean()

        # 5. Physics consistency (if available)
        physics_loss = torch.tensor(0.0, device=device)
        if physics_aux is not None and 'expected_strength' in physics_aux:
            # Encourage gradient at boundaries to match expected Fresnel pattern
            expected = physics_aux['expected_strength']
            gradient_norm = physics_aux.get('gradient_norm', None)

            if gradient_norm is not None:
                # Sample gradient at predicted boundary positions
                pred_pixels = (pred_boundaries * (H - 1)).long().clamp(0, H - 1)

                # Simple approximation: mean gradient along predicted boundary
                B_dim, N_dim, W_dim = pred_boundaries.shape
                grad_at_boundary = torch.zeros(B_dim, N_dim, device=device)

                for b in range(N_dim):
                    for batch in range(B_dim):
                        y_pos = pred_pixels[batch, b, :]
                        grad_vals = gradient_norm[batch, 0, y_pos, torch.arange(W_dim, device=device)]
                        grad_at_boundary[batch, b] = grad_vals.mean()

                # Loss: gradient should match expected pattern
                grad_at_boundary_norm = grad_at_boundary / (grad_at_boundary.sum(dim=1, keepdim=True) + 1e-8)
                expected_norm = expected / (expected.sum() + 1e-8)

                physics_loss = F.mse_loss(grad_at_boundary_norm, expected_norm.unsqueeze(0).expand(B_dim, -1))

        # Total
        total = (
            self.lambda_position * pos_loss +
            self.lambda_cost * cost_loss +
            self.lambda_ordering * ordering_loss +
            self.lambda_smoothness * smoothness_loss +
            self.lambda_physics * physics_loss
        )

        # Stats
        with torch.no_grad():
            mae_pixels = (torch.abs(pred_boundaries - gt_boundaries) * H)
            mae_per_boundary = mae_pixels.mean(dim=(0, 2))

        stats = {
            'total_loss': total.item(),
            'position_loss': pos_loss.item(),
            'cost_loss': cost_loss.item(),
            'ordering_loss': ordering_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'physics_loss': physics_loss.item(),
        }

        boundary_names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']
        for i, name in enumerate(boundary_names[:N]):
            stats[f'{name}_mae_px'] = mae_per_boundary[i].item()

        stats['avg_mae_px'] = mae_per_boundary.mean().item()

        return total, stats


# =============================================================================
# Utility: Boundaries to Segmentation
# =============================================================================
def boundaries_to_segmentation(
    boundaries: torch.Tensor,
    H: int,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Convert boundary positions to segmentation mask.

    Args:
        boundaries: [B, num_boundaries, W] normalized positions (0-1)
        H: Image height
        num_classes: Number of classes (boundaries + 1)

    Returns:
        segmentation: [B, H, W] class labels
    """
    B, N, W = boundaries.shape
    device = boundaries.device

    # Convert to pixels
    boundaries_px = boundaries * (H - 1)

    # Create segmentation
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

    segmentation = torch.zeros(B, H, W, device=device, dtype=torch.long)

    for c in range(num_classes):
        if c == 0:
            # First class: above first boundary
            mask = y_coords < boundaries_px[:, 0:1, :]
        elif c == num_classes - 1:
            # Last class: below last boundary
            mask = y_coords >= boundaries_px[:, -1:, :]
        else:
            # Middle classes: between boundaries
            mask = (y_coords >= boundaries_px[:, c-1:c, :]) & (y_coords < boundaries_px[:, c:c+1, :])

        segmentation[mask] = c

    return segmentation


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing PhysicsInformedDSPModel")
    print("=" * 60)

    device = 'cpu'
    B, H, W = 2, 256, 256

    # Create model
    model = PhysicsInformedDSPModel(
        in_channels=1,
        base_channels=32,
        num_levels=4,
        num_boundaries=4,
        use_noise_estimation=True,
        use_depth_compensation=True,
        use_fresnel_physics=True,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(B, 1, H, W, device=device)

    print("\nForward pass...")
    outputs = model(x, return_aux=True)

    print(f"  boundaries shape: {outputs['boundaries'].shape}")
    print(f"  costs shape: {outputs['costs'].shape}")
    if 'sigma_map' in outputs:
        print(f"  sigma_map shape: {outputs['sigma_map'].shape}")

    # Test physics summary
    summary = model.get_physics_summary()
    print(f"\nPhysics summary:")
    print(f"  Attenuation μ: {summary.get('attenuation_mu', 'N/A'):.6f}")
    print(f"  Physics weight: {summary.get('physics_weight', 'N/A'):.3f}")
    print(f"  Expected gradient strengths: {summary.get('expected_gradient_strength', 'N/A')}")
    print(f"  Cost combination weights: {summary.get('cost_combination_weights', 'N/A')}")

    # Test loss
    gt_boundaries = torch.sort(torch.rand(B, 4, W, device=device), dim=1)[0]
    gt_boundaries = gt_boundaries * 0.8 + 0.1  # Keep in [0.1, 0.9]

    loss_fn = PhysicsInformedDSPLoss(num_boundaries=4).to(device)

    loss, stats = loss_fn(
        outputs['boundaries'],
        gt_boundaries,
        costs=outputs['costs'],
        physics_aux=outputs.get('physics_aux'),
        H=H,
    )

    print(f"\nLoss: {loss.item():.4f}")
    print(f"  Position: {stats['position_loss']:.4f}")
    print(f"  Cost: {stats['cost_loss']:.4f}")
    print(f"  Physics: {stats['physics_loss']:.4f}")
    print(f"  Avg MAE: {stats['avg_mae_px']:.2f} px")

    # Test gradient flow
    loss.backward()

    has_encoder_grad = model.encoder.encoders[0][0].weight.grad is not None
    has_fresnel_grad = model.fresnel_physics.n_delta.grad is not None
    has_depth_grad = model.depth_compensation.log_mu.grad is not None

    print(f"\nGradient flow:")
    print(f"  Encoder: {has_encoder_grad}")
    print(f"  Fresnel n_delta: {has_fresnel_grad}")
    print(f"  Depth μ: {has_depth_grad}")

    # Test segmentation conversion
    seg = boundaries_to_segmentation(outputs['boundaries'], H)
    print(f"\nSegmentation shape: {seg.shape}")
    print(f"  Classes present: {torch.unique(seg).tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
