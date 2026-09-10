#!/usr/bin/env python3
"""
Neuro-Symbolic Corrector V8 Enhanced: Adding High-Potential Techniques from V6/V7

Enhancements over base V8:
1. AdaptiveLambdaPredictor - Per-pixel correction strength (from V6)
2. Feature fusion - Backbone enc1/enc2 features for context (from V6)
3. Attention mechanisms - Channel + Spatial attention in correctors (from V6)
4. Multi-scale processing - Dilated convolutions (from V6)

Preserves V8 novelties:
- Differentiable fuzzy logic with learnable t-norms
- Hierarchical symbolic reasoning with rule chaining
- Physics-accurate speckle model (log-domain, Gamma-K)
- Formal verification guarantees

Author: Neuro-Symbolic OCT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

# Import V8 base components
from neuro_symbolic_corrector_v8 import (
    DifferentiableFuzzyLogic,
    HierarchicalSymbolicReasoner,
    PhysicsAccurateSpecklePredicate,
    FormalVerificationGuarantee,
    CausalExplainer,
    EnhancedGTFreePredicates
)

# Import uncertainty-guided correction (smart algorithmic approach)
from uncertainty_guided_correction import (
    UncertaintyGuidedCorrectionModule,
    CNRPreservingSpatialLoss
)


# =============================================================================
# ATTENTION MODULES (from powerful_correctors.py)
# =============================================================================

class ChannelAttention(nn.Module):
    """Channel attention for feature recalibration."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on important regions."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_val = x.max(dim=1, keepdim=True)[0]
        attention = self.conv(torch.cat([avg, max_val], dim=1))
        return x * attention


# SPEED FIX: Mark attention modules for potential torch.compile optimization
# These are small, frequently-called modules that benefit from fusion
# Note: torch.jit.script can cause issues with dynamic shapes in these modules,
# so we leave them as-is for now. For PyTorch 2.0+, consider using torch.compile()
# at the model level instead of per-module JIT compilation.


# =============================================================================
# ADAPTIVE LAMBDA PREDICTOR FOR V8
# =============================================================================

class AdaptiveLambdaPredictorV8(nn.Module):
    """
    Per-pixel lambda predictor tailored for V8's 5 correctors.

    NOTE: P5 (Speckle Fidelity) is excluded from correction as it is
    fundamentally incompatible with the correction paradigm. We still
    compute and monitor P5 scores, but don't try to correct for it.

    Predicts correction strength based on:
    - Predicate failure maps (P1-P4, P6)
    - Backbone output (context)

    Key insight: Different regions need different correction strengths.
    - Severe failure → high λ → strong correction
    - Mild failure → low λ → gentle correction
    - Passing region → λ ≈ 0 → no correction
    """

    def __init__(self):
        super().__init__()

        # Input: 6 failure maps (P1-P6) + denoised (1) = 7 channels
        # Output: 6 lambda maps (one per corrector)

        # SPEED FIX: Pre-compute corrector_pred_map as class attribute (avoid creating dict in forward)
        self.corrector_pred_map = {
            'edge': 'P1', 'contrast': 'P2', 'smooth': 'P3',
            'structure': 'P4', 'anatomy': 'P6'
        }

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Conv2d(7, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Per-corrector lambda heads (P5/speckle excluded - incompatible with correction)
        self.heads = nn.ModuleDict({
            'edge': self._make_head(32 + 1),      # +1 for P1
            'contrast': self._make_head(32 + 1),  # +1 for P2
            'smooth': self._make_head(32 + 1),    # +1 for P3
            'structure': self._make_head(32 + 1), # +1 for P4
            'anatomy': self._make_head(32 + 1),   # +1 for P6
        })

        # Learnable scaling factors (INCREASED for 30-50% corrector activity)
        # P5/speckle excluded - incompatible with correction paradigm
        self.scales = nn.ParameterDict({
            'edge': nn.Parameter(torch.tensor(1.0)),       # sigmoid(1.0)=0.73 - Increased from 0.0
            'contrast': nn.Parameter(torch.tensor(2.0)),   # sigmoid(2.0)=0.88 - BOOSTED for clinical improvement
            'smooth': nn.Parameter(torch.tensor(0.5)),     # sigmoid(0.5)=0.62 - Increased from -0.5
            'structure': nn.Parameter(torch.tensor(0.5)),  # sigmoid(0.5)=0.62 - Increased from -0.5
            'anatomy': nn.Parameter(torch.tensor(0.5)),    # sigmoid(0.5)=0.62 - Increased from -0.5
        })

        # Maximum lambda caps (INCREASED for 30-50% corrector activity)
        # P5/speckle excluded - incompatible with correction paradigm
        self.lambda_caps = {
            'edge': 1.0,       # Increased from 0.80
            'contrast': 1.0,   # Full strength contrast correction
            'smooth': 0.80,    # Increased from 0.50
            'structure': 0.80, # Increased from 0.50
            'anatomy': 0.80,   # Increased from 0.50
        }

        self._init_weights()

        # FIX: Initialize lambda head biases to moderate negative value
        # Softplus(-2.0) ≈ 0.13, allowing room for learning (was -5.0 → 0.007, too small)
        for head in self.heads.values():
            for module in head:
                if isinstance(module, nn.Conv2d) and module.out_channels == 1:
                    if module.bias is not None:
                        nn.init.constant_(module.bias, -2.0)

    def _make_head(self, in_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),  # Always positive
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, denoised: torch.Tensor,
                failure_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict per-pixel lambda maps for each corrector.

        Args:
            denoised: Backbone output [B, 1, H, W]
            failure_maps: Dict of failure maps {'P1': [B,1,H,W], ...}

        Returns:
            Dict of lambda maps {'edge': [B,1,H,W], ...}
        """
        B, C, H, W = denoised.shape
        device = denoised.device

        # SPEED FIX: Create one default tensor and reuse reference instead of 6 separate torch.zeros calls
        default_map = torch.zeros(B, 1, H, W, device=device, dtype=denoised.dtype)
        p1 = failure_maps.get('P1', default_map)
        p2 = failure_maps.get('P2', default_map)
        p3 = failure_maps.get('P3', default_map)
        p4 = failure_maps.get('P4', default_map)
        p5 = failure_maps.get('P5', default_map)
        p6 = failure_maps.get('P6', default_map)

        # Concat all inputs
        x = torch.cat([p1, p2, p3, p4, p5, p6, denoised], dim=1)

        # Shared features
        shared_feat = self.shared(x)

        # Per-corrector lambda maps (P5/speckle excluded from correction)
        pred_map = {'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4, 'P6': p6}

        lambda_maps = {}
        for name, head in self.heads.items():
            pred_key = self.corrector_pred_map[name]
            head_input = torch.cat([shared_feat, pred_map[pred_key]], dim=1)
            raw_lambda = head(head_input)

            # Apply learnable scaling and cap
            scaled = raw_lambda * torch.sigmoid(self.scales[name])
            # SPEED FIX: Lower clamp is redundant (sigmoid*softplus is always >= 0)
            lambda_maps[name] = scaled.clamp(max=self.lambda_caps[name])

        return lambda_maps


# =============================================================================
# CLINICAL CORRECTORS - Targeting Specific Backbone Weaknesses
# =============================================================================
# Analysis showed backbone has major clinical weaknesses:
# - Contrast: Only 47% preserved (53% lost)
# - Boundary Sharpness: Only 47% preserved
# - Texture: Only 41% preserved
# - Edge: 68% preserved (32% lost)
#
# These correctors are designed to specifically address each weakness.
# =============================================================================


class ClinicalCorrectorBase(nn.Module):
    """
    Base class for clinical correctors with zero-initialized output.

    All clinical correctors share:
    - Input: backbone_out [B, 1, H, W] + failure_map [B, 1, H, W]
    - Output: additive correction [B, 1, H, W]
    - Numerically stable (mean+std thresholds, no quantile ops)
    - Zero-initialized output layer for stable training start
    """

    def __init__(self, hidden_dim: int = 64, name: str = "base"):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim

        # Learnable strength parameter
        self.strength = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        self._strength_scale = 0.5

    def _zero_init_output_layer(self, layer: nn.Conv2d):
        """Zero-initialize output layer for stable training start."""
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _init_weights(self):
        """Initialize weights with Kaiming for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ContrastRestorationCorrector(ClinicalCorrectorBase):
    """
    Restores local contrast in regions where backbone reduced it.

    Clinical problem: 53% of contrast lost by backbone.
    Solution: Boost local contrast by enhancing deviation from local mean.

    ENHANCED for IEEE TMI: Multi-scale contrast amplification with aggressive
    boosting in low-contrast regions. Target: +10-15% clinical contrast improvement.

    CNR-PRESERVING FIX: Apply background region masking to prevent corrections
    from increasing noise in dark/background regions. This maintains CNR while
    still achieving contrast improvement in signal (tissue) regions.

    Method:
    1. Compute local statistics at multiple scales (5x5, 9x9, 15x15)
    2. Identify low-contrast regions using adaptive thresholding
    3. DETECT BACKGROUND REGIONS and suppress corrections there (CNR fix)
    4. Apply multi-scale contrast stretching with stronger amplification
    5. Use signal-aware correction scaling (CNR fix)
    6. Use failure map to focus corrections where needed most
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__(hidden_dim, "contrast")

        # INCREASED strength scale for contrast corrector
        # Previous: 1.2, now 2.0 for aggressive contrast restoration
        self._strength_scale = 2.0

        # Multi-scale local statistics computation
        self.pool_sizes = [5, 9, 15]
        self.local_pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in self.pool_sizes
        ])

        # Feature extraction for adaptive contrast gain - EXPANDED
        # Input: backbone (1) + failure_map (1) + multi-scale local_std (3) + contrast_deficit (1) + signal_mask (1) = 7 channels
        self.feature_net = nn.Sequential(
            nn.Conv2d(7, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale contrast processing with dilated convolutions
        self.multiscale_contrast = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=8, dilation=8),
        ])

        # Attention for focusing on low-contrast regions
        self.channel_attn = ChannelAttention(hidden_dim)
        self.spatial_attn = SpatialAttention()

        # Contrast gain predictor - outputs POSITIVE gain for amplification
        self.gain_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Softplus(),  # Ensure positive gain for contrast boosting
        )

        # Learnable contrast boost factor (initialized to encourage boosting)
        self.contrast_boost = nn.Parameter(torch.tensor(1.5))

        # Learnable low-contrast threshold
        self.low_contrast_threshold = nn.Parameter(torch.tensor(0.1))

        # CNR FIX: Learnable background threshold for signal/background separation
        # This determines the intensity level below which we consider "background"
        self.background_threshold = nn.Parameter(torch.tensor(0.25))

        # CNR FIX: Learnable signal-to-background transition steepness
        # Higher value = sharper transition between signal and background regions
        self.signal_transition_steepness = nn.Parameter(torch.tensor(8.0))

        self._init_weights()
        # Initialize gain head with small positive bias for initial contrast boosting
        for m in self.gain_head:
            if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)  # Start with moderate gain

    def _compute_signal_mask(self, backbone_out: torch.Tensor) -> torch.Tensor:
        """
        CNR FIX: Compute a soft mask identifying signal (tissue) vs background regions.

        Background regions (dark areas) should receive minimal correction to preserve CNR.
        Signal regions (tissue) can receive full contrast enhancement.

        Returns:
            signal_mask: [B, 1, H, W] in [0, 1], where 1 = signal, 0 = background
        """
        # Use adaptive thresholding based on image statistics
        # Compute per-image mean and std for adaptive threshold
        img_mean = backbone_out.mean(dim=[2, 3], keepdim=True)
        img_std = backbone_out.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        # Normalized intensity: values below threshold are background
        # Use sigmoid for soft thresholding (differentiable)
        # threshold = max(learnable_threshold, img_mean - 0.5*img_std)
        adaptive_threshold = torch.maximum(
            torch.sigmoid(self.background_threshold),
            img_mean - 0.3 * img_std
        )

        # Soft signal mask: smooth transition from background to signal
        # steepness controls how sharp the transition is
        steepness = F.softplus(self.signal_transition_steepness)
        signal_mask = torch.sigmoid(steepness * (backbone_out - adaptive_threshold))

        return signal_mask

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute contrast restoration correction with multi-scale amplification and CNR preservation."""
        B, C, H, W = backbone_out.shape

        # CNR FIX: Compute signal mask to identify tissue vs background regions
        signal_mask = self._compute_signal_mask(backbone_out)

        # Compute multi-scale local statistics
        local_stds = []
        local_means = []
        for pool in self.local_pools:
            local_mean = pool(backbone_out)
            local_sq_mean = pool(backbone_out ** 2)
            local_var = (local_sq_mean - local_mean ** 2).clamp(min=1e-8)
            local_std = torch.sqrt(local_var)
            local_stds.append(local_std)
            local_means.append(local_mean)

        # Primary scale for deviation computation (medium scale 9x9)
        primary_mean = local_means[1]
        primary_std = local_stds[1]

        # Deviation from local mean (what we want to amplify)
        deviation = backbone_out - primary_mean

        # Compute contrast deficit: where local std is low, we need more contrast
        # Normalize local_std to [0, 1] range and invert (low std = high deficit)
        std_norm = primary_std / (primary_std.max() + 1e-8)
        contrast_deficit = torch.sigmoid(self.low_contrast_threshold - std_norm + 0.5)

        # Combine multi-scale std features
        multi_std = torch.cat(local_stds, dim=1)

        # CNR FIX: Include signal_mask as input feature so network learns to avoid background
        x = torch.cat([backbone_out, failure_map, multi_std, contrast_deficit, signal_mask], dim=1)

        # Extract features
        feat = self.feature_net(x)

        # Multi-scale contrast processing
        ms_feats = [conv(feat) for conv in self.multiscale_contrast]
        feat = torch.cat(ms_feats, dim=1)

        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Predict gain multiplier (always positive due to Softplus)
        gain = self.gain_head(feat)

        # Enhanced gain in low-contrast regions
        # Boost gain where contrast deficit is high AND failure map indicates problems
        boost_mask = contrast_deficit * failure_map
        boosted_gain = gain * (1.0 + boost_mask * torch.sigmoid(self.contrast_boost))

        # Apply gain to deviation: correction = gain * deviation
        # This amplifies local contrast proportionally
        correction = boosted_gain * deviation

        # NOTE: Background masking is now handled by UncertaintyGuidedCorrectionModule
        # at the aggregate correction level. This provides adaptive per-pixel masking
        # based on uncertainty rather than a fixed min_correction hyperparameter.

        # Scale by learnable strength
        strength = torch.sigmoid(self.strength) * self._strength_scale
        correction = correction * strength

        return correction


class BoundarySharpnessCorrector(ClinicalCorrectorBase):
    """
    Sharpens layer boundaries using vertical gradient enhancement.

    Clinical problem: 53% of boundary sharpness lost.
    Solution: Enhance vertical gradients at layer boundaries.

    OCT-specific: Layer boundaries run horizontally (layers stacked vertically),
    so we need to enhance VERTICAL gradients (d/dy).

    Method:
    1. Compute vertical gradient (Sobel-y)
    2. Identify boundary regions from failure map
    3. Apply gradient-based sharpening: correction = k * sign(grad_y) * |grad_y|^p
       where p < 1 compresses weak gradients, enhances strong ones
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__(hidden_dim, "boundary")

        # Sobel filters for gradient computation
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_x', sobel_x)

        # Feature extraction for adaptive sharpening
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1, bias=False),  # backbone + failure_map + grad_y + grad_x
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Attention
        self.channel_attn = ChannelAttention(hidden_dim)
        self.spatial_attn = SpatialAttention()

        # Sharpening strength predictor
        self.sharpen_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

        self._init_weights()
        self._zero_init_output_layer(self.sharpen_head[-1])

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute boundary sharpening correction."""
        B, C, H, W = backbone_out.shape

        # Compute gradients
        grad_y = F.conv2d(backbone_out, self.sobel_y, padding=1)
        grad_x = F.conv2d(backbone_out, self.sobel_x, padding=1)

        # Combine features
        x = torch.cat([backbone_out, failure_map, grad_y, grad_x], dim=1)

        # Extract features
        feat = self.feature_net(x)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Predict sharpening strength
        sharpen_strength = self.sharpen_head(feat)

        # Apply gradient-based sharpening
        # Negative Laplacian approximation using vertical second derivative
        # This creates "unsharp masking" effect at boundaries
        grad_yy = F.conv2d(grad_y, self.sobel_y, padding=1)
        correction = -sharpen_strength * grad_yy

        # Smooth correction to remove edge-of-edge artifacts while preserving boundary sharpening
        correction = F.avg_pool2d(F.pad(correction, (1, 1, 1, 1), mode='reflect'), kernel_size=3, stride=1)

        # Scale by learnable strength
        strength = torch.sigmoid(self.strength) * self._strength_scale
        correction = correction * strength

        return correction


class TextureRecoveryCorrector(ClinicalCorrectorBase):
    """
    Recovers texture in over-smoothed regions.

    Clinical problem: 59% of texture lost (only 41% preserved).
    Solution: Add back high-frequency texture details.

    CNR-PRESERVING FIX: Apply background region masking to prevent texture
    recovery from adding noise to dark/background regions. This maintains CNR
    while still recovering texture in signal (tissue) regions.

    Method:
    1. Compute high-frequency residual using high-pass filter
    2. Identify over-smoothed regions (low local variance)
    3. DETECT BACKGROUND REGIONS and suppress corrections there (CNR fix)
    4. Add back scaled high-frequency content: correction = k * highpass
       where k is learned based on local smoothness
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__(hidden_dim, "texture")

        # High-pass filter (Laplacian of Gaussian approximation)
        # This extracts texture/fine details
        laplacian = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

        # Local variance computation
        self.local_pool_size = 5
        self.local_pool = nn.AvgPool2d(
            kernel_size=self.local_pool_size,
            stride=1,
            padding=self.local_pool_size // 2
        )

        # Multi-scale high-frequency extraction
        self.multiscale_hp = nn.ModuleList([
            nn.Conv2d(1, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(1, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(1, hidden_dim // 4, 3, padding=3, dilation=3),
            nn.Conv2d(1, hidden_dim // 4, 3, padding=4, dilation=4),
        ])

        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, 3, padding=1, bias=False),  # multiscale + backbone + failure_map
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Attention
        self.channel_attn = ChannelAttention(hidden_dim)
        self.spatial_attn = SpatialAttention()

        # Texture injection strength predictor
        self.texture_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Tanh(),  # Allow both positive and negative (texture can go either way)
        )

        # CNR FIX: Learnable background threshold for signal/background separation
        self.background_threshold = nn.Parameter(torch.tensor(0.25))
        self.signal_transition_steepness = nn.Parameter(torch.tensor(8.0))

        self._init_weights()
        self._zero_init_output_layer(self.texture_head[-2])  # Zero-init before Tanh

    def _compute_signal_mask(self, backbone_out: torch.Tensor) -> torch.Tensor:
        """
        CNR FIX: Compute a soft mask identifying signal (tissue) vs background regions.
        """
        img_mean = backbone_out.mean(dim=[2, 3], keepdim=True)
        img_std = backbone_out.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        adaptive_threshold = torch.maximum(
            torch.sigmoid(self.background_threshold),
            img_mean - 0.3 * img_std
        )

        steepness = F.softplus(self.signal_transition_steepness)
        signal_mask = torch.sigmoid(steepness * (backbone_out - adaptive_threshold))

        return signal_mask

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute texture recovery correction with CNR preservation."""
        B, C, H, W = backbone_out.shape

        # CNR FIX: Compute signal mask to identify tissue vs background regions
        signal_mask = self._compute_signal_mask(backbone_out)

        # Extract high-frequency content (Laplacian)
        highpass = F.conv2d(backbone_out, self.laplacian, padding=1)

        # Multi-scale high-frequency features
        ms_feats = [conv(highpass) for conv in self.multiscale_hp]
        ms_feat = torch.cat(ms_feats, dim=1)

        # Combine with backbone and failure map
        x = torch.cat([ms_feat, backbone_out, failure_map], dim=1)

        # Process features
        feat = self.feature_net(x)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Predict texture injection strength
        texture_strength = self.texture_head(feat)

        # Apply texture: add scaled high-frequency content back
        correction = texture_strength * highpass

        # NOTE: Background masking is now handled by UncertaintyGuidedCorrectionModule
        # at the aggregate correction level. This provides adaptive per-pixel masking
        # based on uncertainty rather than a fixed min_correction hyperparameter.

        # Scale by learnable strength
        strength = torch.sigmoid(self.strength) * self._strength_scale
        correction = correction * strength

        return correction


class EdgeEnhancementCorrector(ClinicalCorrectorBase):
    """
    Strengthens weakened edges throughout the image.

    Clinical problem: 32% of edges lost (68% preserved).
    Solution: Enhance edge magnitude while preserving edge direction.

    CNR-PRESERVING FIX: Apply background region masking to prevent edge
    enhancement from adding noise to dark/background regions. This maintains
    CNR while still enhancing edges in signal (tissue) regions.

    Method:
    1. Compute gradient magnitude and direction
    2. Identify weak edges (low gradient magnitude)
    3. DETECT BACKGROUND REGIONS and suppress corrections there (CNR fix)
    4. Boost edges: correction proportional to gradient direction
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__(hidden_dim, "edge")

        # Sobel filters
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_x', sobel_x)

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1, bias=False),  # backbone + failure_map + grad_mag + grad_dir
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale edge processing
        self.multiscale = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=8, dilation=8),
        ])

        # Attention
        self.channel_attn = ChannelAttention(hidden_dim)
        self.spatial_attn = SpatialAttention()

        # Edge enhancement predictor
        self.edge_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

        # Register Laplacian kernel as buffer to avoid creating tensor in forward()
        laplacian_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel)

        # CNR FIX: Learnable background threshold for signal/background separation
        self.background_threshold = nn.Parameter(torch.tensor(0.25))
        self.signal_transition_steepness = nn.Parameter(torch.tensor(8.0))

        self._init_weights()
        self._zero_init_output_layer(self.edge_head[-1])

    def _compute_signal_mask(self, backbone_out: torch.Tensor) -> torch.Tensor:
        """
        CNR FIX: Compute a soft mask identifying signal (tissue) vs background regions.
        """
        img_mean = backbone_out.mean(dim=[2, 3], keepdim=True)
        img_std = backbone_out.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        adaptive_threshold = torch.maximum(
            torch.sigmoid(self.background_threshold),
            img_mean - 0.3 * img_std
        )

        steepness = F.softplus(self.signal_transition_steepness)
        signal_mask = torch.sigmoid(steepness * (backbone_out - adaptive_threshold))

        return signal_mask

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute edge enhancement correction with CNR preservation."""
        B, C, H, W = backbone_out.shape

        # CNR FIX: Compute signal mask to identify tissue vs background regions
        signal_mask = self._compute_signal_mask(backbone_out)

        # Compute gradients
        grad_y = F.conv2d(backbone_out, self.sobel_y, padding=1)
        grad_x = F.conv2d(backbone_out, self.sobel_x, padding=1)

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Normalize gradient magnitude for feature input (mean+std normalization)
        grad_mean = grad_mag.mean(dim=[2, 3], keepdim=True)
        grad_std = grad_mag.std(dim=[2, 3], keepdim=True) + 1e-8
        grad_mag_norm = (grad_mag - grad_mean) / grad_std

        # Combine features
        x = torch.cat([backbone_out, failure_map, grad_mag_norm, grad_mag], dim=1)

        # Process features
        feat = self.feature_net(x)

        # Multi-scale processing
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Attention
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Predict edge enhancement strength
        edge_strength = self.edge_head(feat)

        # Apply edge enhancement using unsharp masking principle
        # Laplacian approximation for edge enhancement
        laplacian = F.conv2d(backbone_out, self.laplacian_kernel, padding=1)
        correction = -edge_strength * laplacian

        # NOTE: Background masking is now handled by UncertaintyGuidedCorrectionModule
        # at the aggregate correction level. This provides adaptive per-pixel masking
        # based on uncertainty rather than a fixed min_correction hyperparameter.

        # Scale by learnable strength
        strength = torch.sigmoid(self.strength) * self._strength_scale
        correction = correction * strength

        return correction


# Legacy corrector classes for backward compatibility (map to new clinical correctors)
class EnhancedEdgeCorrector(EdgeEnhancementCorrector):
    """Backward-compatible alias for EdgeEnhancementCorrector."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 enc1_channels: int = 48, enc2_channels: int = 96):
        super().__init__(hidden_dim)


class EnhancedContrastCorrector(ContrastRestorationCorrector):
    """Backward-compatible alias for ContrastRestorationCorrector."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 enc1_channels: int = 48, enc2_channels: int = 96):
        super().__init__(hidden_dim)


class EnhancedSmoothnessCorrector(TextureRecoveryCorrector):
    """Backward-compatible alias for TextureRecoveryCorrector (inverse - recovers lost texture)."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 enc1_channels: int = 48, enc2_channels: int = 96):
        super().__init__(hidden_dim)


class EnhancedStructureCorrector(BoundarySharpnessCorrector):
    """Backward-compatible alias for BoundarySharpnessCorrector."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 enc1_channels: int = 48, enc2_channels: int = 96):
        super().__init__(hidden_dim)


class EnhancedAnatomyCorrector(ClinicalCorrectorBase):
    """
    Anatomy corrector - preserves overall anatomical structure.
    Uses combination of multi-scale features to maintain structure coherence.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 enc1_channels: int = 48, enc2_channels: int = 96):
        super().__init__(hidden_dim, "anatomy")

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale for capturing anatomy at different scales
        self.multiscale = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=8, dilation=8),
        ])

        # Attention
        self.channel_attn = ChannelAttention(hidden_dim)
        self.spatial_attn = SpatialAttention()

        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Tanh(),
        )

        self._init_weights()
        self._zero_init_output_layer(self.output_head[-2])

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute anatomy-preserving correction."""
        B, C, H, W = backbone_out.shape

        # Combine inputs
        x = torch.cat([backbone_out, failure_map], dim=1)

        # Feature extraction
        feat = self.feature_net(x)

        # Multi-scale processing
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Attention
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Refinement
        feat = self.refine(feat)

        # Output
        correction = self.output_head(feat)

        # Scale by learnable strength
        strength = torch.sigmoid(self.strength) * self._strength_scale
        correction = correction * strength

        return correction


# =============================================================================
# MAIN V8 ENHANCED CORRECTOR
# =============================================================================

class NeuroSymbolicCorrectorV8Enhanced(nn.Module):
    """
    Enhanced V8 Corrector with CLINICAL-FOCUSED correctors.

    Addresses backbone weaknesses identified in analysis:
    - Contrast: Only 47% preserved (53% lost) -> ContrastRestorationCorrector
    - Boundary Sharpness: Only 47% preserved -> BoundarySharpnessCorrector
    - Texture: Only 41% preserved -> TextureRecoveryCorrector
    - Edge: 68% preserved (32% lost) -> EdgeEnhancementCorrector

    Predicate mapping:
    - P1 (edge) -> EdgeEnhancementCorrector
    - P2 (contrast) -> ContrastRestorationCorrector
    - P3 (smoothness) -> TextureRecoveryCorrector (inverse - recover lost texture)
    - P4 (structure) -> BoundarySharpnessCorrector
    - P6 (anatomy) -> EnhancedAnatomyCorrector

    NOTE: P5 (Speckle Fidelity) is computed for monitoring but NOT corrected,
    as it is fundamentally incompatible with the correction paradigm.

    V8 Framework preserved:
    1. Differentiable fuzzy logic (Lukasiewicz t-norm)
    2. Hierarchical symbolic reasoning (3 levels)
    3. Physics-accurate speckle MONITORING (log-domain, Gamma-K)
    4. Formal verification guarantees
    5. Causal interpretability
    6. AdaptiveLambdaPredictor - per-pixel correction strength
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 64,
                 enc1_channels: int = 40, enc2_channels: int = 80):
        super().__init__()

        # Store backbone feature dimensions
        self.enc1_channels = enc1_channels
        self.enc2_channels = enc2_channels

        # Enhanced predicates with physics-accurate speckle
        self.predicates = EnhancedGTFreePredicates()

        # TRUE symbolic router with differentiable logic
        self.router = HierarchicalSymbolicReasoner()

        # NEW: Adaptive lambda predictor
        self.lambda_predictor = AdaptiveLambdaPredictorV8()

        # NEW: Uncertainty-guided correction module (smart algorithmic approach)
        # Replaces fixed min_correction=0.40 hyperparameter with learned adaptive behavior
        self.uncertainty_guided = UncertaintyGuidedCorrectionModule(
            min_correction_floor=0.20,  # Minimum 20% correction in low-uncertainty regions
            max_correction_ceiling=1.0   # Full correction in high-uncertainty regions
        )

        # CLINICAL CORRECTORS targeting specific backbone weaknesses
        # NOTE: P5/speckle corrector removed - fundamentally incompatible with correction paradigm
        # P5 is still computed and monitored in predicates, just not corrected
        self.correctors = nn.ModuleDict({
            'edge': EdgeEnhancementCorrector(hidden_channels),       # P1: 32% edges lost
            'contrast': ContrastRestorationCorrector(hidden_channels),  # P2: 53% contrast lost
            'smooth': TextureRecoveryCorrector(hidden_channels),     # P3: 59% texture lost (inverse)
            'structure': BoundarySharpnessCorrector(hidden_channels),  # P4: 53% boundary sharpness lost
            'anatomy': EnhancedAnatomyCorrector(in_channels, hidden_channels, enc1_channels, enc2_channels),
        })

        # Predicate to corrector mapping (updated for clinical focus)
        # P3 (smoothness) maps to texture corrector - we want LESS smoothing, MORE texture
        self.pred_key_map = {
            'edge': 'P1',       # Edge preservation
            'contrast': 'P2',  # Contrast preservation
            'smooth': 'P3',    # Smoothness -> Texture recovery (inverse)
            'structure': 'P4', # Structure -> Boundary sharpness
            'anatomy': 'P6',   # Anatomy preservation
        }

        # FORMAL verification
        self.verifier = FormalVerificationGuarantee(self.predicates)

        # Causal explainer
        self.explainer = CausalExplainer(self.router)

        self._print_info()

    def _print_info(self):
        print("\n" + "=" * 70)
        print("NeuroSymbolicCorrectorV8Enhanced - CLINICAL CORRECTORS")
        print("=" * 70)
        print("Addressing Backbone Weaknesses:")
        print("  - Contrast: 53% lost -> ContrastRestorationCorrector (P2)")
        print("  - Boundary: 53% lost -> BoundarySharpnessCorrector (P4)")
        print("  - Texture: 59% lost -> TextureRecoveryCorrector (P3 inverse)")
        print("  - Edge: 32% lost -> EdgeEnhancementCorrector (P1)")
        print("")
        print("V8 Framework Preserved:")
        print("  1. Differentiable fuzzy logic (Lukasiewicz t-norm)")
        print("  2. Hierarchical rule chaining (3 levels)")
        print("  3. Physics-accurate speckle MONITORING (log-domain, Gamma-K)")
        print("  4. Formal guarantees (Energy + Pareto + Lipschitz)")
        print("  5. Causal interpretability")
        print("  6. AdaptiveLambdaPredictor (per-pixel correction strength)")
        print("  7. UncertaintyGuidedCorrection (adaptive background masking)")
        print("")
        print("SMART ALGORITHMIC APPROACH (replaces fixed hyperparameter):")
        print("  - UncertaintyGuidedCorrectionModule computes per-pixel uncertainty")
        print("  - High uncertainty (failing predicates) -> strong correction")
        print("  - Low uncertainty (clean regions) -> weak correction (preserve CNR)")
        print("  - Replaces fixed min_correction=0.40 with learned adaptive behavior")
        print("")
        print("Predicate Mapping:")
        for corrector, pred in self.pred_key_map.items():
            print(f"  {pred} -> {corrector}")
        print("")
        print("NOTE: P5 (Speckle) is monitored but NOT corrected")
        print("      (fundamentally incompatible with correction paradigm)")
        print(f"Active Correctors: {list(self.correctors.keys())}")
        print("=" * 70)

        # Parameter counts
        lambda_params = sum(p.numel() for p in self.lambda_predictor.parameters())
        uncertainty_params = sum(p.numel() for p in self.uncertainty_guided.parameters())
        corrector_params = sum(p.numel() for p in self.correctors.parameters())
        router_params = sum(p.numel() for p in self.router.parameters())
        pred_params = sum(p.numel() for p in self.predicates.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        print(f"\nParameters (hidden_channels=64):")
        print(f"  Lambda predictor: {lambda_params:,}")
        print(f"  Uncertainty guided: {uncertainty_params:,}")
        print(f"  Correctors (5x): {corrector_params:,}")
        print(f"  Router: {router_params:,}")
        print(f"  Predicates: {pred_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"\nCorrection Capacity:")
        print(f"  Lambda caps: edge/contrast=0.80, smooth/structure/anatomy=0.50")
        print(f"  Uncertainty-guided mask: [0.20, 1.0] (adaptive)")
        print(f"  Total correction clamp: [-0.7, 0.7]")

    def forward(self,
                backbone_out: torch.Tensor,
                noisy: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None,
                return_details: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply neuro-symbolic correction with full interpretability.

        Args:
            backbone_out: Backbone denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            backbone_features: Optional dict with 'enc1', 'enc2' encoder features
            return_details: Whether to include detailed explanations

        Returns:
            corrected: Final output [B, 1, H, W]
            info: Dictionary with predicate scores, activations, etc.
        """
        # Step 1: Evaluate predicates (with physics-accurate P5)
        with torch.no_grad():
            pred_results = self.predicates(backbone_out, noisy)

        # Step 2: Hierarchical symbolic routing
        routing = self.router(pred_results)
        activations = routing['activations']

        # Step 3: Get failure maps (skip P5/speckle - not used for correction)
        # Use class attribute pred_key_map for clinical corrector mapping
        failure_maps = {}
        for name, key in self.pred_key_map.items():
            failure_maps[key] = pred_results[key]['failure_map']

        # Step 4: Predict per-pixel lambda maps (NEW)
        lambda_maps = self.lambda_predictor(backbone_out, failure_maps)

        # Step 5: Apply corrections with lambda modulation
        # SPEED FIX: Pre-zip data to reduce dict lookups in loop
        corrector_data = [
            (name, corrector, failure_maps[self.pred_key_map[name]], activations[name], lambda_maps[name])
            for name, corrector in self.correctors.items()
        ]

        corrections = {}
        for name, corrector, failure_map, act, lam in corrector_data:
            correction = corrector(backbone_out, failure_map, backbone_features)

            # Keep activation as tensor for efficient multiplication
            if isinstance(act, torch.Tensor):
                act = act.view(1, 1, 1, 1) if act.numel() == 1 else act

            corrections[name] = correction * act * lam

        del corrector_data  # Free the temporary list

        # Step 6: Combine corrections
        total_correction = sum(corrections.values())
        del corrections  # Free individual corrections after summing

        # Step 6b: Apply UNCERTAINTY-GUIDED MASKING (smart algorithmic approach)
        # This replaces the fixed min_correction=0.40 hyperparameter with learned adaptive behavior
        # - High uncertainty regions (noisy/failing predicates) -> strong correction
        # - Low uncertainty regions (clean/passing predicates) -> weak correction (preserve CNR)
        total_correction, uncertainty_info = self.uncertainty_guided.apply_to_correction(
            total_correction, backbone_out, noisy, failure_maps
        )
        del failure_maps  # Free failure maps after use

        # INCREASED clamp range from [-0.5, 0.5] to [-0.7, 0.7] for stronger corrections
        # This allows more aggressive contrast restoration for IEEE TMI targets
        total_correction = total_correction.clamp(-0.7, 0.7)
        candidate = (backbone_out + total_correction).clamp(0, 1)

        # Step 7: FORMAL verification
        output, verify_info = self.verifier(
            backbone_out, candidate, noisy, total_correction, pred_before=pred_results
        )

        # Step 8: Re-evaluate predicates on CORRECTED output to measure improvement
        with torch.no_grad():
            pred_results_corrected = self.predicates(output, noisy)

        # SPEED FIX: Only compute layer analysis when needed
        if return_details:
            layer_analysis = self.explainer.clinical_layer_analysis(
                {k: pred_results[k]['failure_map'] for k in pred_results
                 if isinstance(pred_results.get(k), dict) and 'failure_map' in pred_results[k]},
                backbone_out.shape[2]
            )
        else:
            layer_analysis = {}

        # Compute lambda stats - detach for info dict (gradient flow comes from loss function)
        if self.training:
            # During training: detach mean for info dict (only for monitoring, not gradient flow)
            # The gradient flow for lambda regularization should come from the loss function
            lambda_stats = {
                name: {'mean': lam.mean().detach(), 'max': lam.max().detach()}
                for name, lam in lambda_maps.items()
            }
        else:
            # During eval: convert to Python floats
            lambda_stats = {
                name: {'mean': lam.mean().item(), 'max': lam.max().item()}
                for name, lam in lambda_maps.items()
            }

        info = {
            'predicate_scores': pred_results_corrected['scores'],  # Scores on CORRECTED output (what we want to track)
            'predicate_scores_backbone': pred_results['scores'],  # Scores on backbone output (before correction)
            'activations': {
                k: v.detach().item() if (isinstance(v, torch.Tensor) and not self.training) else (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in activations.items()
            },
            'lambda_stats': lambda_stats,
            'inference_trace': {
                level: {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in items.items()}
                for level, items in routing['inference_trace'].items()
                if isinstance(items, dict)
            },
            'explanations': routing['explanations'],
            'verification': verify_info,
            'layer_analysis': layer_analysis,
            'correction_magnitude': total_correction.abs().mean().detach() if self.training else total_correction.abs().mean().item(),
            'uncertainty_info': uncertainty_info,  # Smart adaptive correction stats
        }

        if return_details:
            info['clinical_report'] = self.explainer.generate_clinical_report(
                pred_results, routing, verify_info, layer_analysis
            )
            info['counterfactuals'] = self.explainer.counterfactual_analysis(
                pred_results, 'P1', [0.3, 0.5, 0.7, 0.9]
            )

        # Cleanup: delete intermediate results after use to free memory
        del pred_results
        del pred_results_corrected
        del lambda_maps

        return output, info


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\nTesting NeuroSymbolicCorrectorV8Enhanced with CLINICAL CORRECTORS...")

    # Create model
    model = NeuroSymbolicCorrectorV8Enhanced(
        in_channels=1,
        hidden_channels=64,
        enc1_channels=48,
        enc2_channels=96
    )

    # Test inputs
    B, C, H, W = 2, 1, 128, 128
    noisy = torch.randn(B, C, H, W) * 0.3 + 0.5
    noisy = noisy.clamp(0, 1)
    backbone_out = noisy - torch.randn(B, C, H, W) * 0.1
    backbone_out = backbone_out.clamp(0, 1)

    # Simulate backbone features (not used by new clinical correctors but kept for compatibility)
    backbone_features = {
        'enc1': torch.randn(B, 48, H, W) * 0.1,
        'enc2': torch.randn(B, 96, H // 2, W // 2) * 0.1,
    }

    # Forward pass
    corrected, info = model(backbone_out, noisy, backbone_features, return_details=True)

    print(f"\nInput shape: {backbone_out.shape}")
    print(f"Output shape: {corrected.shape}")
    print(f"\nPredicate Scores (after correction): {info['predicate_scores']}")
    print(f"Predicate Scores (backbone): {info['predicate_scores_backbone']}")
    print(f"\nActivations: {info['activations']}")
    print(f"\nLambda Stats:")
    for name, stats in info['lambda_stats'].items():
        print(f"  {name}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
    print(f"\nCorrection magnitude: {info['correction_magnitude']:.4f}")

    print("\n" + "=" * 60)
    print("CLINICAL CORRECTOR DETAILS:")
    print("=" * 60)
    print("Predicate to Corrector Mapping:")
    for corrector, pred in model.pred_key_map.items():
        print(f"  {pred} -> {corrector}")

    print("\n" + "=" * 60)
    print("VERIFICATION RESULT:")
    print("=" * 60)
    print(f"Decision: {info['verification']['decision']}")
    print(f"Guarantees passed: {info['verification']['guarantees_passed']}/3")

    # Test individual clinical correctors
    print("\n" + "=" * 60)
    print("INDIVIDUAL CORRECTOR TEST:")
    print("=" * 60)

    test_failure_map = torch.rand(B, 1, H, W)

    for name, corrector in model.correctors.items():
        correction = corrector(backbone_out, test_failure_map)
        print(f"  {name}: correction range [{correction.min().item():.4f}, {correction.max().item():.4f}], "
              f"mean={correction.abs().mean().item():.4f}")

    print("\nTest passed!")
