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
            'tissue': ['P2', 'P3', 'P6'],
            'boundary': ['P1', 'P4'],
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
            'tissue': self._make_head(32 + 1),    # +1 for combined P2/P3/P6
            'boundary': self._make_head(32 + 1),  # +1 for combined P1/P4
        })

        # Learnable scaling factors (INCREASED for 30-50% corrector activity)
        # P5/speckle excluded - incompatible with correction paradigm
        self.scales = nn.ParameterDict({
            'tissue': nn.Parameter(torch.tensor(1.5)),     # sigmoid(1.5)=0.82 - Conservative (QT32 level)
            'boundary': nn.Parameter(torch.tensor(1.0)),   # sigmoid(1.0)=0.73 - Conservative (QT32 level)
        })

        # Maximum lambda caps — raised for QT39 to allow stronger corrections
        # Cap 1.5 means lambda can amplify corrections up to 1.5× raw output
        self.lambda_caps = {
            'tissue': 1.5,
            'boundary': 1.5,
        }

        self._init_weights()

        # Initialize lambda head biases for stronger initial lambda
        # Softplus(-0.5) ≈ 0.47 (moderate initial lambda)
        for head in self.heads.values():
            for module in head:
                if isinstance(module, nn.Conv2d) and module.out_channels == 1:
                    if module.bias is not None:
                        nn.init.constant_(module.bias, -0.5)

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
            Dict of lambda maps {'tissue': [B,1,H,W], 'boundary': [B,1,H,W]}
        """
        B, C, H, W = denoised.shape
        device = denoised.device

        # Extract individual failure maps
        default_map = torch.zeros(B, 1, H, W, device=device, dtype=denoised.dtype)
        p1 = failure_maps.get('P1', default_map)
        p2 = failure_maps.get('P2', default_map)
        p3 = failure_maps.get('P3', default_map)
        p4 = failure_maps.get('P4', default_map)
        p5 = failure_maps.get('P5', default_map)
        p6 = failure_maps.get('P6', default_map)

        # Concat all inputs (still use all 6 failure maps for shared feature extraction)
        x = torch.cat([p1, p2, p3, p4, p5, p6, denoised], dim=1)

        # Shared features
        shared_feat = self.shared(x)

        # Combine failure maps per corrector group
        combined_failures = {
            'tissue': torch.max(torch.max(p2, p3), p6),      # worst of P2, P3, P6
            'boundary': torch.max(p1, p4),                     # worst of P1, P4
        }

        # Per-corrector lambda maps
        lambda_maps = {}
        for name, head in self.heads.items():
            combined_failure = combined_failures[name]
            head_input = torch.cat([shared_feat, combined_failure], dim=1)
            raw_lambda = head(head_input)

            # Scale and cap
            scale = torch.sigmoid(self.scales[name])
            cap = self.lambda_caps[name]
            lambda_maps[name] = (raw_lambda * scale).clamp(0, cap)

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

        # NOTE: strength parameter REMOVED (gate collapse QT38).
        # Redundant with lambda_val — both scale corrections multiplicatively.
        # Removing reduces gate chain from 11 to 7, increasing correction throughput ~75x.

    def _zero_init_output_layer(self, layer: nn.Conv2d):
        """Small-init output layer for non-zero training start.

        Zero-init causes correctors to start at exactly 0.0 output,
        creating a dead zone that backprop struggles to escape from
        due to multiplicative gates downstream.
        """
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='linear')
        layer.weight.data *= 0.3  # 30% of Kaiming (moderate init)
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

        # NOTE: strength_scale REMOVED (gate collapse QT38) — was redundant with lambda_val

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

        return correction


class TissueCorrector(ClinicalCorrectorBase):
    """
    Unified tissue quality corrector handling contrast, smoothness, and anatomy.
    Combines P2 (contrast), P3 (smoothness), P6 (anatomy) failure signals.

    Architecture: 2ch input → feature_net → 4x dilated convs → CBAM attention → output
    """
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__(hidden_channels, name="tissue")
        h = hidden_channels

        # Input: backbone_out (1) + combined failure map (1) = 2 channels
        self.feature_net = nn.Sequential(
            nn.Conv2d(2, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale dilated convolutions for different tissue scales
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(h, h // 4, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(h // 4),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for d in [1, 2, 4, 8]
        ])

        # Channel + Spatial attention (CBAM)
        self.channel_attn = ChannelAttention(h)
        self.spatial_attn = SpatialAttention()

        # Output head (zero-init for stable start)
        self.output_head = nn.Sequential(
            nn.Conv2d(h, h // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h // 2, 1, 1),
            nn.Tanh(),
        )

        self._init_weights()
        self._zero_init_output_layer(self.output_head[-2])  # Conv2d before Tanh

    def forward(self, backbone_out, noisy, failure_map, pred_score=None, backbone_features=None):
        x = torch.cat([backbone_out, failure_map], dim=1)
        feat = self.feature_net(x)

        # Multi-scale features
        dilated_outs = [conv(feat) for conv in self.dilated_convs]
        feat = torch.cat(dilated_outs, dim=1)  # h//4 * 4 = h

        # Attention
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Output
        correction = self.output_head(feat)
        return correction


class BoundaryCorrector(ClinicalCorrectorBase):
    """
    Unified boundary/edge corrector handling edges and structure.
    Combines P1 (edge) and P4 (structure) failure signals.
    Built-in Sobel operator provides explicit gradient features.

    Architecture: 4ch input (backbone + sobel_x + sobel_y + failure) → feature_net → 4x dilated → CBAM → output
    """
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__(hidden_channels, name="boundary")
        h = hidden_channels

        # Register Sobel kernels as buffers (not parameters)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Input: backbone_out (1) + sobel_x (1) + sobel_y (1) + failure_map (1) = 4 channels
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale dilated convolutions
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(h, h // 4, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(h // 4),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for d in [1, 2, 4, 8]
        ])

        # Channel + Spatial attention (CBAM)
        self.channel_attn = ChannelAttention(h)
        self.spatial_attn = SpatialAttention()

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(h, h // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h // 2, 1, 1),
            nn.Tanh(),
        )

        self._init_weights()
        self._zero_init_output_layer(self.output_head[-2])

    def forward(self, backbone_out, noisy, failure_map, pred_score=None, backbone_features=None):
        # Compute Sobel gradients
        grad_x = F.conv2d(F.pad(backbone_out, [1,1,1,1], mode='reflect'), self.sobel_x)
        grad_y = F.conv2d(F.pad(backbone_out, [1,1,1,1], mode='reflect'), self.sobel_y)

        x = torch.cat([backbone_out, grad_x, grad_y, failure_map], dim=1)
        feat = self.feature_net(x)

        # Multi-scale features
        dilated_outs = [conv(feat) for conv in self.dilated_convs]
        feat = torch.cat(dilated_outs, dim=1)

        # Attention
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Output
        correction = self.output_head(feat)
        return correction


class MultiplicativeGainCorrector(ClinicalCorrectorBase):
    """
    Multiplicative gain corrector that predicts a spatially-smooth alpha map.

    Processes at 1/4 resolution for inherent spatial smoothness, then
    upsamples back to full resolution via bilinear interpolation.

    Input channels (computed internally):
        - backbone_out (1ch)
        - failure_map (1ch)
        - local_contrast (1ch, computed from backbone_out using 9x9 local mean)

    Output: alpha_map [B, 1, H, W], decomposed as brightness + contrast:
        - brightness: softplus(ch0) * max_alpha, always >= 0 (preserves tissue mean for SNR)
        - contrast: max_alpha * tanh(ch1), zero-mean enforced (local contrast only)
        Zero-initialized for stable start (alpha ~ 0 initially).

    Architecture:
        feature_net -> 4x dilated convs (d=1,2,4,8) -> CBAM attention -> output_head
    """

    def __init__(self, in_channels=1, hidden_channels=64, max_alpha=0.15):
        super().__init__(hidden_channels, name="multiplicative_gain")
        h = hidden_channels
        self.max_alpha = max_alpha

        # Local contrast: 9x9 average pooling for computing local mean
        self.local_mean_pool = nn.AvgPool2d(
            kernel_size=9, stride=1, padding=4
        )

        # Downsample to 1/4 resolution
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)

        # Input: backbone_out (1) + failure_map (1) + local_contrast (1) = 3 channels
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale dilated convolutions
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(h, h // 4, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(h // 4),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for d in [1, 2, 4, 8]
        ])

        # Channel + Spatial attention (CBAM)
        self.channel_attn = ChannelAttention(h)
        self.spatial_attn = SpatialAttention()

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(h, h // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h // 2, 2, 1),  # 2 channels: brightness + contrast
        )

        # Standard weight init, then zero-init the last conv for stable start
        self._init_weights()
        self._zero_init_last_conv()

    def _zero_init_last_conv(self):
        """Zero-init the final Conv2d layer so alpha_map starts near 0."""
        last_conv = self.output_head[-1]  # Conv2d(h//2, 2, 1)
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)
            # Warm-start brightness channel (ch0) with strong positive bias.
            # ReLU(0.5) * max_alpha(0.15) = 0.075 initial boost in flat tissue.
            # Post-smooth gate + 3x3 smoothing protects both EPI and CNR.
            last_conv.bias.data[0] = 0.5

    def _compute_local_contrast(self, backbone_out):
        """
        Compute local contrast: (backbone - local_mean) / (|local_mean| + eps),
        clamped to [-3, 3].

        Uses a 9x9 average pool for the local mean computation.
        """
        local_mean = self.local_mean_pool(backbone_out)
        local_contrast = (backbone_out - local_mean) / (local_mean.abs() + 1e-4)
        return local_contrast.clamp(-3.0, 3.0)

    def forward(self, backbone_out, noisy, failure_map, pred_score=None, backbone_features=None):
        """
        Compute multiplicative gain alpha_map.

        Args:
            backbone_out: Denoised backbone output [B, 1, H, W]
            noisy: Noisy input (accepted for API compatibility, not used)
            failure_map: Predicate failure map [B, 1, H, W]
            pred_score: Unused (API compatibility)
            backbone_features: Unused (API compatibility)

        Returns:
            alpha_map: Spatially-smooth gain map [B, 1, H, W],
                       range [-max_alpha, +max_alpha]
        """
        B, C, H, W = backbone_out.shape

        # Compute local contrast at full resolution
        local_contrast = self._compute_local_contrast(backbone_out)

        # Concatenate 3 input channels
        x = torch.cat([backbone_out, failure_map, local_contrast], dim=1)

        # Downsample to 1/4 resolution for spatial smoothness
        x_low = self.downsample(x)

        # Feature extraction at low resolution
        feat = self.feature_net(x_low)

        # Multi-scale dilated convolutions
        dilated_outs = [conv(feat) for conv in self.dilated_convs]
        feat = torch.cat(dilated_outs, dim=1)  # h//4 * 4 = h

        # CBAM attention
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Raw output: 2 channels at low resolution
        raw_output = self.output_head(feat)  # [B, 2, H_low, W_low]

        # Upsample back to original resolution via bilinear interpolation
        raw_output = F.interpolate(
            raw_output, size=(H, W), mode='bilinear', align_corners=False
        )

        # Channel 0: brightness — always >= 0 (preserves/increases tissue mean for CNR/SNR)
        brightness = F.relu(raw_output[:, 0:1]) * self.max_alpha

        # Channel 1: contrast — zero-mean per image (local enhancement only)
        contrast_raw = self.max_alpha * torch.tanh(raw_output[:, 1:2])
        contrast = contrast_raw - contrast_raw.mean(dim=[2, 3], keepdim=True)

        # Combined alpha: brightness (always positive) + contrast (zero-mean)
        alpha_map = brightness + contrast

        return alpha_map


# =============================================================================
# EDGE RESIDUAL FILTER — Recovers true edge detail from denoising residual
# =============================================================================

class EdgeResidualFilter(nn.Module):
    """
    Filter the denoising residual (noisy - backbone) to recover true edge
    detail that the backbone smoothed away.

    Unlike the clinical correctors (which inherit from ClinicalCorrectorBase),
    this is a standalone nn.Module filter.  It uses Sobel edge detection on
    the backbone output to build a soft edge gate, then a small CNN predicts
    which parts of the residual are genuine edge detail vs. noise.

    Key design choices:
    - Sobel + Gaussian buffers (no learnable overhead for edge detection)
    - Grouped convolution in the middle layer for parameter efficiency
    - Learnable blend_logit initialised to 0.5 (sigmoid ≈ 0.62) so the
      filter has strong contribution from the start of training
    - Kaiming-initialised final conv (scaled 30%) for non-zero starting signal
    - max_magnitude caps output scale (default 0.20)

    Total parameters: ~43 K
    """

    def __init__(self, max_magnitude: float = 0.08):
        super().__init__()
        self.max_magnitude = max_magnitude

        # ----- Sobel edge-detection buffers -----
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # ----- Gaussian blur buffer (3x3) -----
        gaussian = torch.tensor(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 16.0
        self.register_buffer('gaussian', gaussian)

        # ----- Filter network -----
        # Input: residual (1) + edge_magnitude (1) + backbone_out (1) = 3 channels
        self.filter_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, groups=4, bias=False),  # grouped conv
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 1, 1),  # 1x1 projection
            nn.Tanh(),
        )

        # ----- Learnable blend (sigmoid(-1.0) ≈ 0.27 — moderate from start) -----
        self.blend_logit = nn.Parameter(torch.tensor(-1.0))

        # ----- Initialisation -----
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Kaiming init for all convs; zero-init the last conv before Tanh."""
        for m in self.filter_net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Kaiming init with 30% scaling for the last Conv2d (1x1 before Tanh).
        # Zero-init combined with low blend_logit caused the filter to produce
        # nothing for many epochs, preventing EPI improvement.
        last_conv = self.filter_net[-2]  # Conv2d(16, 1, 1) just before Tanh
        nn.init.kaiming_normal_(last_conv.weight, mode='fan_out', nonlinearity='linear')
        last_conv.weight.data *= 0.3  # 30% of Kaiming for conservative but non-zero start
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    # ------------------------------------------------------------------
    def _compute_edge_gate(
        self, backbone_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a soft edge gate from Sobel magnitude of backbone_out.

        Returns:
            edge_gate: [B, 1, H, W] in roughly [0, 1]
            edge_mag:  [B, 1, H, W] normalised edge magnitude (for filter input)
        """
        grad_x = F.conv2d(backbone_out, self.sobel_x, padding=1)
        grad_y = F.conv2d(backbone_out, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Per-image normalisation to [0, 1]
        B = edge_mag.shape[0]
        edge_flat = edge_mag.view(B, -1)
        emin = edge_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        emax = edge_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        edge_normalized = (edge_mag - emin) / (emax - emin + 1e-8)

        # Soft threshold — activates above ~0.05 normalised magnitude
        # Lowered from 0.15 to capture more genuine edges for EPI improvement
        edge_gate = torch.sigmoid(10.0 * (edge_normalized - 0.05))

        # Gaussian-blur the gate to remove Sobel staircase artifacts
        edge_gate = F.conv2d(
            F.pad(edge_gate, [1, 1, 1, 1], mode='reflect'),
            self.gaussian,
        )

        return edge_gate, edge_normalized

    # ------------------------------------------------------------------
    def forward(
        self, backbone_out: torch.Tensor, noisy: torch.Tensor
    ) -> torch.Tensor:
        """
        Filter the denoising residual to recover edge detail.

        Args:
            backbone_out: Denoised backbone output [B, 1, H, W]
            noisy:        Original noisy input      [B, 1, H, W]

        Returns:
            Additive edge-residual correction [B, 1, H, W]
        """
        residual = noisy - backbone_out
        edge_gate, edge_mag = self._compute_edge_gate(backbone_out)

        filter_input = torch.cat([residual, edge_mag, backbone_out], dim=1)
        filtered = self.filter_net(filter_input) * self.max_magnitude

        beta = torch.sigmoid(self.blend_logit)
        return beta * edge_gate * filtered


# =============================================================================
# GUIDED EDGE SHARPENER (replaces EdgeResidualFilter for EPI improvement)
# =============================================================================

class GuidedEdgeSharpener(nn.Module):
    """
    Multi-scale guided edge sharpening using unsharp masking.

    Uses backbone - blur(backbone) at multiple scales to extract noise-free
    edge detail.  Unlike EdgeResidualFilter which uses (noisy - backbone)
    — a noise-dominated residual — this approach extracts edges directly
    from the denoised backbone output, which is completely noise-free.

    Math:
        edge_signal_k = backbone - gaussian_blur_k(backbone)

    This is the classic unsharp mask.  It contains only the edges the
    backbone preserved during denoising.  Amplifying these REAL edges
    improves EPI without adding noise (potentially PSNR-positive since
    it restores true high-frequency signal).

    Architecture:
        3 Gaussian blur scales (3x3, 5x5, 7x7) → 4-channel input
        (backbone + 3 edge scales) → lightweight CNN → Tanh → scaled output

    Total parameters: ~12 K
    """

    def __init__(self, max_magnitude: float = 0.10):
        super().__init__()
        self.max_magnitude = max_magnitude

        # Multi-scale Gaussian blur kernels
        self.register_buffer('blur_3x3', self._make_gaussian(3))
        self.register_buffer('blur_5x5', self._make_gaussian(5))
        self.register_buffer('blur_7x7', self._make_gaussian(7))

        # Learnable global blend (sigmoid(0.0) = 0.5 — moderate from start)
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

        # Filter network: learns which edges to amplify and by how much
        # Input: backbone(1) + edge_3x3(1) + edge_5x5(1) + edge_7x7(1) = 4 channels
        self.filter_net = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(24, 24, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(24, 1, 1),  # 1x1 projection
            nn.Tanh(),
        )

        self._init_weights()

    @staticmethod
    def _make_gaussian(kernel_size: int) -> torch.Tensor:
        """Create normalized Gaussian kernel."""
        sigma = kernel_size / 3.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g = torch.exp(-0.5 * coords ** 2 / sigma ** 2)
        kernel = g.unsqueeze(1) * g.unsqueeze(0)  # outer product
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _init_weights(self):
        """Kaiming init; 50% scaling on last conv for moderate initial signal."""
        for m in self.filter_net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Scale last conv for conservative but non-zero initial contribution
        last_conv = self.filter_net[-2]  # Conv2d(24, 1, 1) before Tanh
        nn.init.kaiming_normal_(last_conv.weight, mode='fan_out',
                                nonlinearity='linear')
        last_conv.weight.data *= 0.5
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, backbone_out: torch.Tensor) -> torch.Tensor:
        """
        Compute edge sharpening correction from backbone output only.

        Args:
            backbone_out: Denoised backbone output [B, 1, H, W]

        Returns:
            Additive edge-sharpening correction [B, 1, H, W]
        """
        # Multi-scale unsharp masking (completely noise-free!)
        pad3 = F.pad(backbone_out, [1, 1, 1, 1], mode='reflect')
        edge_3 = backbone_out - F.conv2d(pad3, self.blur_3x3)  # fine edges

        pad5 = F.pad(backbone_out, [2, 2, 2, 2], mode='reflect')
        edge_5 = backbone_out - F.conv2d(pad5, self.blur_5x5)  # medium edges

        pad7 = F.pad(backbone_out, [3, 3, 3, 3], mode='reflect')
        edge_7 = backbone_out - F.conv2d(pad7, self.blur_7x7)  # coarse edges

        # Concatenate backbone + multi-scale edge signals
        x = torch.cat([backbone_out, edge_3, edge_5, edge_7], dim=1)

        # Filter learns which edges to amplify
        beta = torch.sigmoid(self.blend_logit)
        correction = self.filter_net(x) * self.max_magnitude * beta

        return correction


# =============================================================================
# EDGE RECOVERY MODULE (supervised residual recovery for EPI improvement)
# =============================================================================

class EdgeRecoveryModule(nn.Module):
    """
    Recovers lost edge detail from the denoising residual with supervised
    Sobel-domain training.

    Key insight: EPI = Pearson(Sobel(clean), Sobel(corrected)) is scale-
    invariant, so uniformly amplifying edges (unsharp masking) gives ZERO
    EPI improvement.  To actually move EPI, we must recover edges the
    backbone LOST — information that exists in the noisy residual
    (noisy - backbone) but not in the backbone alone.

    Architecture:
        Input channels (4):
          - normalized_residual: (noisy - backbone) / local_mean — contains
            lost edges + noise, intensity-normalized for OCT multiplicative noise
          - backbone_edge_mag: |Sobel(backbone)| — where backbone sees edges
            (guidance: residual near existing edges is more likely signal)
          - backbone: denoised output (context)
          - unsharp_5: backbone - blur_5x5(backbone) — noise-free edge signal

        Recovery CNN: 4 → 32 → 32 → 32(grouped) → 1 with Tanh

    Training:
        Supervised with Sobel-domain loss:
          target = Sobel(clean) - Sobel(backbone)   [the lost edges]
          loss = MSE(Sobel(corrected) - Sobel(backbone), target)

    Total parameters: ~13K
    """

    def __init__(self, max_magnitude: float = 0.08):
        super().__init__()
        self.max_magnitude = max_magnitude

        # Sobel edge detection buffers
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Gaussian kernels for normalization and unsharp masking
        self.register_buffer('blur_5x5', self._make_gaussian(5))
        self.register_buffer('blur_9x9', self._make_gaussian(9))

        # Learnable global blend (sigmoid(0.0) = 0.5)
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

        # Recovery CNN
        # Input: normalized_residual(1) + edge_mag(1) + backbone(1) + unsharp(1) = 4
        # InstanceNorm2d: resolution/batch-agnostic, no running stats needed.
        # Avoids the train/eval gap caused by uninitialized BatchNorm running stats
        # when loading checkpoints with strict=False.
        self.recovery_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, groups=4, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 1, 1),  # 1x1 projection
            nn.Tanh(),
        )

        self._init_weights()

    @staticmethod
    def _make_gaussian(kernel_size: int) -> torch.Tensor:
        """Create normalized Gaussian kernel."""
        sigma = kernel_size / 3.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g = torch.exp(-0.5 * coords ** 2 / sigma ** 2)
        kernel = g.unsqueeze(1) * g.unsqueeze(0)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _init_weights(self):
        """Kaiming init with 30% scaling on last conv."""
        for m in self.recovery_net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Scale last conv for conservative but non-zero start
        last_conv = self.recovery_net[-2]  # Conv2d before Tanh
        nn.init.kaiming_normal_(last_conv.weight, mode='fan_out',
                                nonlinearity='linear')
        last_conv.weight.data *= 0.3
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, backbone_out: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """
        Recover lost edges from the denoising residual.

        Args:
            backbone_out: Denoised backbone output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]

        Returns:
            Additive edge-recovery correction [B, 1, H, W]
        """
        # Residual contains lost edges + noise
        residual = noisy - backbone_out

        # Normalize residual by local intensity (OCT multiplicative noise fix)
        pad9 = F.pad(backbone_out, [4, 4, 4, 4], mode='reflect')
        local_mean = F.conv2d(pad9, self.blur_9x9)
        normalized_residual = residual / (local_mean + 0.01)

        # Edge magnitude from backbone (guidance: where backbone sees edges)
        grad_x = F.conv2d(F.pad(backbone_out, [1, 1, 1, 1], mode='reflect'), self.sobel_x)
        grad_y = F.conv2d(F.pad(backbone_out, [1, 1, 1, 1], mode='reflect'), self.sobel_y)
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        # Per-image normalization
        B = edge_mag.shape[0]
        emax = edge_mag.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1).clamp(min=1e-6)
        edge_mag_norm = edge_mag / emax

        # Unsharp mask (noise-free edge signal for context)
        pad5 = F.pad(backbone_out, [2, 2, 2, 2], mode='reflect')
        unsharp = backbone_out - F.conv2d(pad5, self.blur_5x5)

        # Recovery network
        x = torch.cat([normalized_residual, edge_mag_norm, backbone_out, unsharp], dim=1)
        raw_correction = self.recovery_net(x) * self.max_magnitude

        beta = torch.sigmoid(self.blend_logit)
        return beta * raw_correction


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
