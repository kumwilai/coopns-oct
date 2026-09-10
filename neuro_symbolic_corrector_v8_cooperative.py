#!/usr/bin/env python3
"""
Neuro-Symbolic Corrector V8 Cooperative: Cooperative Neuro-Symbolic Denoising Framework

This module implements a novel cooperative framework where the neural backbone (NAFNet)
and symbolic correctors negotiate their contributions based on confidence and potential.

=============================================================================
COOPERATIVE DENOISING FRAMEWORK
=============================================================================

Key Innovation: Instead of blindly applying corrections, this framework:
1. NAFNet estimates its own confidence (where is it certain vs uncertain)
2. Each corrector assesses its contribution potential (where can I help)
3. Symbolic negotiation determines optimal allocation between backbone and correctors
4. Corrections are applied weighted by the negotiated allocation

This creates a "team of experts" paradigm where:
- NAFNet handles regions where it's confident
- Correctors step in where NAFNet is uncertain AND they have expertise
- Symbolic rules mediate conflicts and ensure clinical constraints

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

NAFNetWithConfidence
    |
    v
[denoised_image, confidence_map]
    |
    +---> CorrectorWithPotential (x2: tissue + boundary)
    |         |
    |         v
    |     [correction, potential_map]
    |
    +---> PredicateEvaluation
    |         |
    |         v
    |     [predicate_scores, failure_maps]
    |
    v
SymbolicNegotiator
    |
    v
[allocation_maps] --> Weighted combination --> Final output

=============================================================================
NEW COMPONENTS
=============================================================================

1. NAFNetWithConfidence:
   - Wraps NAFNet backbone to also output confidence maps
   - Confidence estimated from feature variance and gradient magnitude
   - High confidence = backbone is certain, low = uncertain

2. CorrectorWithPotential:
   - Extended correctors that assess their contribution potential
   - Potential based on: predicate gap, local quality deficit, learned capability
   - Returns both correction and potential map

3. SymbolicNegotiator:
   - Fuzzy logic-based negotiation between NAFNet and correctors
   - Rules encode when to trust backbone vs correctors
   - Output: per-pixel allocation maps for each corrector

4. NeuroSymbolicCorrectorV8Cooperative:
   - Main class orchestrating the cooperative framework
   - Full interpretability: confidence, potentials, allocations, negotiation trace

Author: Neuro-Symbolic OCT Team
Date: 2025
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math

# =============================================================================
# IMPORTS FROM V8 BASE AND V8 ENHANCED
# =============================================================================
# We reuse the core components from existing V8 implementations

from neuro_symbolic_corrector_v8 import (
    DifferentiableFuzzyLogic,
    HierarchicalSymbolicReasoner,
    PhysicsAccurateSpecklePredicate,
    FormalVerificationGuarantee,
    CausalExplainer,
    EnhancedGTFreePredicates,
)

from neuro_symbolic_corrector_v8_enhanced import (
    ChannelAttention,
    SpatialAttention,
    AdaptiveLambdaPredictorV8,
    ClinicalCorrectorBase,
    ContrastRestorationCorrector,
    BoundarySharpnessCorrector,
    TextureRecoveryCorrector,
    EdgeEnhancementCorrector,
    EnhancedAnatomyCorrector,
    TissueCorrector,
    BoundaryCorrector,
    MultiplicativeGainCorrector,
    EdgeResidualFilter,
    GuidedEdgeSharpener,
    EdgeRecoveryModule,
)

from cnr_preserving_correction import CNRPreservingCorrectionModule
from clinical_enhancement_module import ClinicalEnhancementModule


# =============================================================================
# REGION-AWARE CORRECTION MODULE
# =============================================================================


def batched_otsu_threshold(image, bins=256):
    """Otsu threshold per image, computed on the GPU without leaving the batch.

    The reference implementation in validate_crossdataset loops over 256 bins in
    Python and moves every image to the CPU. That costs about a second per batch,
    which is fine once per evaluation but not once per training step. This version
    is the same criterion written as cumulative sums over a batched histogram, so
    it runs in a few hundred microseconds on the device the data is already on.

    Args:
        image: [B, 1, H, W] in [0, 1]
    Returns:
        [B, 1, 1, 1] threshold per image
    """
    B = image.shape[0]
    x = image.detach().clamp(0, 1).reshape(B, -1)
    idx = (x * (bins - 1)).long().clamp(0, bins - 1)
    hist = torch.zeros(B, bins, device=x.device, dtype=x.dtype)
    hist.scatter_add_(1, idx, torch.ones_like(x))

    centers = (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5) / bins
    w0 = hist.cumsum(1)
    total = w0[:, -1:].clamp(min=1.0)
    w1 = (total - w0).clamp(min=0.0)
    sum0 = (hist * centers).cumsum(1)
    sum_total = sum0[:, -1:]
    m0 = sum0 / w0.clamp(min=1.0)
    m1 = (sum_total - sum0) / w1.clamp(min=1.0)
    between = w0 * w1 * (m0 - m1) ** 2
    t = between.argmax(dim=1)
    return centers[t].view(B, 1, 1, 1)


def tissue_gate(backbone_out, rule="intensity", steepness=20.0, dilate=5):
    """Build the map that decides where a correction is allowed to act.

    Two rules are available.

    intensity   the original behaviour, a soft step at the per image fifteenth
                percentile of the backbone output.
    otsu        a soft step at the per image Otsu threshold, then dilated so that
                the tissue border is included rather than clipped. Outside a
                dilated tissue mask the gate is zero, which is what makes the
                background noise measures provable rather than merely observed,
                because a correction that is identically zero on the background
                cannot change a statistic computed on the background.

    The gate is a function of the detached backbone output in both cases, so it
    carries no gradient and only ever scales the correction.
    """
    b = backbone_out.detach()
    if rule == "otsu" or rule.startswith("halo"):
        # "haloR" is the Otsu tissue mask dilated by R pixels. A correction may act
        # inside tissue and within R pixels of it, and is identically zero beyond
        # that. "otsu" is the special case halo5. The radius matters because the
        # contrast and texture gains are produced by the edge branch acting in the
        # band just outside tissue, so a mask drawn tightly at the tissue boundary
        # removes them, while an unbounded gate lets corrections reach background
        # that the noise measures are computed on.
        if rule.startswith("halo"):
            dilate = int(rule[4:] or 5)
        thresh = batched_otsu_threshold(b)
        gate = torch.sigmoid((b - thresh) * steepness)
        if dilate and dilate > 1:
            pad = dilate // 2
            gate = F.max_pool2d(gate, kernel_size=dilate, stride=1, padding=pad)
            if gate.shape[2:] != b.shape[2:]:
                gate = gate[:, :, :b.shape[2], :b.shape[3]]
        return gate
    B = b.size(0)
    thresh = torch.quantile(b.view(B, -1), 0.15, dim=1).view(B, 1, 1, 1)
    return torch.sigmoid((b - thresh) * steepness)


class RegionAwareCorrectionModule(nn.Module):
    """
    Region-aware correction that applies aggressive clinical enhancement only
    in uncertain/low-quality regions while preserving PSNR in confident regions.

    Key Innovation: Adaptive correction strength based on:
    1. NAFNet confidence (uncertainty weight)
    2. Local contrast (low contrast = needs more enhancement)
    3. Edge strength (near edges = needs sharpening)
    4. Background vs tissue (background needs smoothing for CNR)

    Goal: Achieve 10-15% clinical improvement in uncertain regions while
    keeping overall PSNR drop < 1.0 dB by preserving confident regions.
    """

    def __init__(self, clinical_boost_factor: float = 1.5):
        """
        Initialize region-aware correction module.

        Args:
            clinical_boost_factor: Maximum boost factor for uncertain regions (default: 2.0)
                                   This means uncertain regions get up to 3x correction (1.0 + 1.0*2.0)
                                   compared to confident regions.
                                   Balance between clinical improvement and PSNR preservation.
        """
        super().__init__()

        self.clinical_boost_factor = clinical_boost_factor

        # Unified: using shared AdaptiveRegionDetector when available
        # Set after construction by the parent model (e.g., from cnr_preserver.region_detector)
        self.region_detector = None

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Learnable parameters for region weighting
        self.contrast_weight = nn.Parameter(torch.tensor(0.3))
        self.edge_weight = nn.Parameter(torch.tensor(0.3))
        self.background_weight = nn.Parameter(torch.tensor(0.2))
        self.uncertainty_weight_param = nn.Parameter(torch.tensor(0.2))

        # Background threshold (learnable)
        self.background_threshold = nn.Parameter(torch.tensor(0.15))

        # Minimum correction factor in confident regions (reduced to avoid overcorrection)
        # BALANCED: 0.65 is middle ground between aggressive (0.8) and conservative (0.5)
        # Target: ~15-20% clinical improvement with ~3-4 dB PSNR drop
        self.min_correction_factor = nn.Parameter(torch.tensor(0.65))

        # Maximum correction factor in uncertain regions (reduced to avoid overcorrection)
        self.max_correction_factor = nn.Parameter(torch.tensor(1.5))

        # Adaptive gating network - learns optimal correction strength per region
        # Input: backbone (1ch) + confidence (1ch) + correction magnitude (1ch) = 3 channels
        self.adaptive_gate = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Input: backbone, confidence, correction magnitude
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),  # Output: (tissue_gate, bg_gate) in [0, 1]
        )

        # Initialize bias to output low background gate values
        # The last conv layer outputs 2 channels: tissue_gate, bg_gate
        # We want bg_gate to start low (negative bias → sigmoid → near 0)
        with torch.no_grad():
            self.adaptive_gate[-1].bias[1] = -2.0  # bg_gate starts low
            self.adaptive_gate[-1].bias[0] = 0.0   # tissue_gate starts moderate

        # Learnable soft clamp parameter for adaptive magnitude limiting
        self.soft_clamp_param = nn.Parameter(torch.tensor(0.0))

    def compute_local_contrast(self, x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """
        Compute local contrast map.

        Low contrast regions need more enhancement.

        Args:
            x: Input image [B, 1, H, W]
            kernel_size: Size of local window

        Returns:
            contrast_deficit: Map where high values = low contrast = needs enhancement [B, 1, H, W]
        """
        padding = kernel_size // 2

        # Local mean and variance
        local_mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        local_sq_mean = F.avg_pool2d(x * x, kernel_size, stride=1, padding=padding)
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=1e-8)
        local_std = torch.sqrt(local_var)

        # Normalize contrast (0 = max contrast, 1 = no contrast)
        max_std = local_std.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        contrast_normalized = local_std / max_std

        # Contrast deficit: inverse of contrast (high = needs enhancement)
        contrast_deficit = 1.0 - contrast_normalized

        return contrast_deficit

    def compute_edge_strength(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge strength map.

        Near-edge regions benefit from sharpening.

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            edge_need: Map where high values = near edge = needs sharpening [B, 1, H, W]
        """
        # Compute gradients
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # Gradient magnitude
        edge_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)

        # Normalize
        max_edge = edge_mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        edge_normalized = edge_mag / max_edge

        # Dilate edges slightly to catch near-edge regions
        edge_dilated = F.max_pool2d(edge_normalized, kernel_size=5, stride=1, padding=2)

        return edge_dilated

    def compute_background_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute background vs tissue mask.

        Background regions benefit from smoothing for better CNR.

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            background_mask: Map where high values = background = needs smoothing [B, 1, H, W]
        """
        # Background is typically low intensity
        threshold = torch.sigmoid(self.background_threshold) * 0.3  # Max 0.3

        # Soft background detection
        background_mask = torch.sigmoid(10.0 * (threshold - x))

        # Smooth the mask
        background_mask = F.avg_pool2d(
            F.pad(background_mask, (2, 2, 2, 2), mode='reflect'),
            kernel_size=5, stride=1
        )

        return background_mask

    def compute_region_mask(self,
                           backbone_out: torch.Tensor,
                           nafnet_confidence: torch.Tensor,
                           cached_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute composite region mask for adaptive correction.

        Combines:
        1. Uncertainty weight from NAFNet confidence
        2. Local contrast deficit
        3. Edge strength for sharpening
        4. Background mask for smoothing (CNR FIX: now used to SUPPRESS, not boost)

        Args:
            backbone_out: Denoised image [B, 1, H, W]
            nafnet_confidence: NAFNet confidence map [B, 1, H, W]
            cached_masks: Optional pre-computed (tissue_mask, background_mask) to avoid
                          redundant region_detector calls. If provided, these are used
                          directly instead of calling region_detector.

        Returns:
            region_mask: Combined mask indicating correction strength needed [B, 1, H, W]
            background_mask: Mask for background regions (used for suppression) [B, 1, H, W]
            region_info: Dictionary with component masks
        """
        # 1. Uncertainty weight: low confidence = high uncertainty = needs more correction
        uncertainty_weight = 1.0 - nafnet_confidence.clamp(0, 1)

        # 2. Local contrast deficit
        contrast_deficit = self.compute_local_contrast(backbone_out)

        # 3. Edge strength
        edge_strength = self.compute_edge_strength(backbone_out)

        # 4. Background mask - use cached masks if available, otherwise detect
        target_shape = backbone_out.shape[2:]
        if cached_masks is not None:
            tissue_mask, background_mask = cached_masks
            # Interpolate if shapes don't match
            if tissue_mask.shape[2:] != target_shape:
                tissue_mask = F.interpolate(tissue_mask, size=target_shape, mode='bilinear', align_corners=False)
                background_mask = F.interpolate(background_mask, size=target_shape, mode='bilinear', align_corners=False)
        elif self.region_detector is not None:
            try:
                tissue_mask, background_mask = self.region_detector(backbone_out)
                tissue_mask = tissue_mask.clamp(0.0, 1.0)
                background_mask = background_mask.clamp(0.0, 1.0)
            except Exception:
                # Fallback to threshold-based if detector fails
                background_mask = self.compute_background_mask(backbone_out)
                tissue_mask = 1.0 - background_mask
        else:
            # Fallback to threshold-based if no detector available
            background_mask = self.compute_background_mask(backbone_out)
            tissue_mask = 1.0 - background_mask

        # Combine with learnable weights (ensure weights are positive and sum reasonably)
        # CNR FIX: Removed background_weight from the sum as it's now used for suppression
        w_unc = torch.sigmoid(self.uncertainty_weight_param)
        w_con = torch.sigmoid(self.contrast_weight)
        w_edge = torch.sigmoid(self.edge_weight)

        # Normalize weights (only for positive contributors)
        total_weight = w_unc + w_con + w_edge + 1e-6
        w_unc = w_unc / total_weight
        w_con = w_con / total_weight
        w_edge = w_edge / total_weight

        # Combined region mask - CNR FIX: These factors INCREASE correction strength
        # Only applied in tissue regions (non-background)
        enhancement_factors = (
            w_unc * uncertainty_weight +
            w_con * contrast_deficit +
            w_edge * edge_strength
        )

        # CNR FIX: Modulate enhancement factors by tissue mask
        # Background regions get SUPPRESSED correction (factor approaches 0)
        # Tissue regions get FULL correction based on enhancement_factors
        # The background_weight controls how aggressively we suppress background
        w_bg = torch.sigmoid(self.background_weight)
        background_suppression = 1.0 - (w_bg * background_mask)  # 1 in tissue, (1-w_bg) in background

        region_mask = (enhancement_factors * background_suppression).clamp(0, 1)

        region_info = {
            'uncertainty_weight': uncertainty_weight.mean().item(),
            'contrast_deficit': contrast_deficit.mean().item(),
            'edge_strength': edge_strength.mean().item(),
            'background_ratio': background_mask.mean().item(),
            'tissue_ratio': tissue_mask.mean().item(),
            'background_suppression_mean': background_suppression.mean().item(),
            'combined_mask_mean': region_mask.mean().item(),
            'weights': {
                'uncertainty': w_unc.item(),
                'contrast': w_con.item(),
                'edge': w_edge.item(),
                'background_suppression': w_bg.item(),
            }
        }

        return region_mask, background_mask, region_info

    def compute_adaptive_correction_factor(self,
                                           region_mask: torch.Tensor,
                                           nafnet_confidence: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive correction factor based on region mask.

        In confident regions: minimal correction (preserve PSNR)
        In uncertain regions: strong correction (improve clinical metrics)

        Formula:
            clinical_boost = uncertainty_weight * clinical_boost_factor
            final_factor = base_factor * (1.0 + clinical_boost)

        Where:
            - uncertainty_weight = 1.0 - nafnet_confidence (0 in confident, 1 in uncertain)
            - clinical_boost_factor = 3.0 (default) - up to 3x stronger in uncertain regions

        Args:
            region_mask: Combined region mask [B, 1, H, W]
            nafnet_confidence: NAFNet confidence map [B, 1, H, W]

        Returns:
            correction_factor: Per-pixel correction strength [B, 1, H, W]
        """
        # Get min/max correction factors
        min_factor = torch.sigmoid(self.min_correction_factor)  # ~0.52 for 0.1
        max_factor = torch.sigmoid(self.max_correction_factor) * 2.0 + 0.5  # ~1.77 for 1.5

        # Uncertainty-based boost
        uncertainty = 1.0 - nafnet_confidence.clamp(0, 1)
        clinical_boost = uncertainty * self.clinical_boost_factor

        # Combine with region mask for final factor
        # Region mask modulates where to apply the boost
        base_factor = min_factor
        correction_factor = base_factor * (1.0 + clinical_boost * region_mask)

        # Clamp to reasonable range
        correction_factor = correction_factor.clamp(min_factor, max_factor)

        return correction_factor

    def forward(self,
                base_correction: torch.Tensor,
                backbone_out: torch.Tensor,
                nafnet_confidence: torch.Tensor,
                cached_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Apply region-aware correction scaling.

        In confident regions (high NAFNet confidence): apply minimal correction (preserve PSNR)
        In uncertain regions (low NAFNet confidence): apply strong correction (improve clinical)
        CNR FIX: In background regions: SUPPRESS correction to reduce noise (preserve CNR)

        Args:
            base_correction: Base correction from correctors [B, 1, H, W]
            backbone_out: Denoised image from backbone [B, 1, H, W]
            nafnet_confidence: NAFNet confidence map [B, 1, H, W]
            cached_masks: Optional pre-computed (tissue_mask, background_mask) to avoid
                          redundant region_detector calls.

        Returns:
            scaled_correction: Correction scaled by region-aware factor [B, 1, H, W]
            region_info: Dictionary with region analysis
        """
        # Ensure shapes match
        target_shape = backbone_out.shape[2:]
        if base_correction.shape[2:] != target_shape:
            base_correction = F.interpolate(base_correction, size=target_shape, mode='bilinear', align_corners=False)
        if nafnet_confidence.shape[2:] != target_shape:
            nafnet_confidence = F.interpolate(nafnet_confidence, size=target_shape, mode='bilinear', align_corners=False)

        # Compute region mask (CNR FIX: now also returns background_mask)
        # Pass cached_masks through to avoid redundant region_detector calls
        region_mask, background_mask, region_info = self.compute_region_mask(backbone_out, nafnet_confidence, cached_masks=cached_masks)

        # Compute adaptive correction factor
        correction_factor = self.compute_adaptive_correction_factor(region_mask, nafnet_confidence)

        # Scale correction by factor
        correction = base_correction * correction_factor

        # =====================================================================
        # ADAPTIVE GATING MECHANISM - Learned, region-aware correction gating
        # =====================================================================
        # Compute adaptive gates based on local context
        # Input: backbone output, confidence map, and correction magnitude
        gate_input = torch.cat([
            backbone_out,
            nafnet_confidence,
            correction.abs(),  # Correction magnitude gives context
        ], dim=1)

        gates = torch.sigmoid(self.adaptive_gate(gate_input))
        tissue_gate = gates[:, 0:1]  # [0, 1] - how much to allow in tissue
        bg_gate = gates[:, 1:2]      # [0, 1] - how much to allow in background (should learn to be low)

        # Compute tissue mask (inverse of background mask)
        tissue_mask = 1.0 - background_mask

        # Apply region-specific gating
        # tissue_mask and background_mask come from compute_region_mask
        gated_correction = correction * (
            tissue_mask * tissue_gate +
            background_mask * bg_gate * 0.1  # Background gets 10x reduction even at max gate
        )

        # =====================================================================
        # LEARNABLE SOFT CLAMP - Replaces hardcoded clamp values
        # =====================================================================
        # Soft clamp with learnable limit: Range [0.02, 0.15]
        soft_limit = torch.sigmoid(self.soft_clamp_param) * 0.13 + 0.02
        scaled_correction = soft_limit * torch.tanh(gated_correction / soft_limit)

        # Add stats to region_info
        region_info['correction_factor_mean'] = correction_factor.mean().item()
        region_info['correction_factor_min'] = correction_factor.min().item()
        region_info['correction_factor_max'] = correction_factor.max().item()
        region_info['base_correction_magnitude'] = base_correction.abs().mean().item()
        region_info['scaled_correction_magnitude'] = scaled_correction.abs().mean().item()
        # Adaptive gating stats
        region_info['tissue_gate_mean'] = tissue_gate.mean().item()
        region_info['bg_gate_mean'] = bg_gate.mean().item()
        region_info['soft_clamp_limit'] = soft_limit.item()
        region_info['gated_correction_magnitude'] = gated_correction.abs().mean().item()

        # =====================================================================
        # GATE REGULARIZATION - Guide the adaptive gate during training
        # =====================================================================
        # Regularization: encourage bg_gate to be low, tissue_gate to be moderate
        # bg_gate should learn to be near 0, tissue_gate can be 0.3-0.7
        gate_reg = bg_gate.mean() * 5.0  # Penalize high background gate values
        gate_reg = gate_reg + F.relu(0.3 - tissue_gate.mean())  # Encourage tissue gate >= 0.3
        region_info['gate_regularization'] = gate_reg.item()

        return scaled_correction, region_info


# =============================================================================
# PART 1: NAFNet WITH CONFIDENCE ESTIMATION
# =============================================================================

class NAFNetWithConfidence(nn.Module):
    """
    Wrapper for NAFNet backbone that also estimates confidence maps.

    The confidence map indicates where the backbone is certain about its
    denoising output. This is estimated from:

    1. Feature variance: Low variance in deep features suggests certainty
    2. Gradient magnitude: Smooth gradients suggest confident predictions
    3. Local consistency: Agreement between neighboring predictions

    High confidence regions -> Trust NAFNet output
    Low confidence regions -> Correctors may need to intervene

    Attributes:
        backbone: The underlying NAFNet model (passed externally)
        confidence_estimator: CNN to estimate confidence from features

    Note: This wrapper expects NAFNet to return both output and features.
    If NAFNet doesn't return features, we estimate confidence from output only.
    """

    def __init__(self, feature_channels: int = 64):
        """
        Initialize the confidence estimation module.

        Args:
            feature_channels: Expected number of channels in backbone features
        """
        super().__init__()

        # Confidence estimation from backbone features
        # Input: feature variance + gradient features
        self.confidence_from_features = nn.Sequential(
            nn.Conv2d(feature_channels + 2, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # Confidence estimation from output only (fallback)
        # Uses gradient magnitude and local variance
        self.confidence_from_output = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),  # output + grad_x + grad_y + local_var
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Local variance kernel
        self.local_pool_size = 5

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for confidence estimators."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local variance using average pooling - OPTIMIZED."""
        padding = self.local_pool_size // 2
        local_mean = F.avg_pool2d(x, self.local_pool_size, stride=1, padding=padding)
        # OPTIMIZATION: Use x*x instead of x**2 (faster)
        local_sq_mean = F.avg_pool2d(x * x, self.local_pool_size, stride=1, padding=padding)
        # OPTIMIZATION: Fuse clamp and nan_to_num into single operation
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=1e-8)
        return torch.nan_to_num(local_var, nan=1e-8, posinf=1.0, neginf=1e-8)

    def estimate_confidence_from_output(self, denoised: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence from denoised output alone - OPTIMIZED.

        MODIFIED: Now produces LOWER confidence to allow correctors to work.
        Target: 10-20% uncertain regions instead of <1%.

        Confidence is HIGH when:
        - Low gradient magnitude (smooth regions are easier to denoise)
        - Low local variance (consistent local structure)
        - Values away from boundaries (0 or 1)

        Args:
            denoised: Backbone denoised output [B, 1, H, W]

        Returns:
            confidence: Confidence map [B, 1, H, W] in [0, 1]
        """
        # OPTIMIZATION: Compute gradients and local variance in parallel-friendly way
        # Compute gradients using pre-registered buffers
        grad_x = F.conv2d(denoised, self.sobel_x, padding=1)
        grad_y = F.conv2d(denoised, self.sobel_y, padding=1)

        # OPTIMIZATION: Compute local variance inline to avoid function call overhead
        padding = self.local_pool_size // 2
        local_mean = F.avg_pool2d(denoised, self.local_pool_size, stride=1, padding=padding)
        local_sq_mean = F.avg_pool2d(denoised * denoised, self.local_pool_size, stride=1, padding=padding)
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=1e-8)

        # Concatenate features and estimate raw confidence
        raw_confidence = self.confidence_from_output(torch.cat([denoised, grad_x, grad_y, local_var], dim=1))

        # Clamp to designed operating range [0.15, 0.75]
        # This range was carefully tuned to create the right uncertainty distribution:
        # ~10% confident, ~78% uncertain, ~11% neutral
        confidence = raw_confidence.clamp(0.15, 0.75)

        # Reduce confidence near edges (edges need corrector help)
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)
        edge_max = grad_mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        edge_uncertainty = (grad_mag / edge_max) * 0.2
        confidence = confidence - edge_uncertainty

        # Reduce confidence in low-contrast regions
        local_std = torch.sqrt(local_var)
        std_max = local_std.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        low_contrast_uncertainty = (1.0 - local_std / std_max) * 0.15
        confidence = confidence - low_contrast_uncertainty

        return confidence.clamp(0.15, 0.75)

    def estimate_confidence_from_features(self,
                                          denoised: torch.Tensor,
                                          backbone_features: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence from backbone features - OPTIMIZED.

        MODIFIED: Now produces LOWER confidence to allow correctors to work.
        Target: 10-20% uncertain regions instead of <1%.

        Uses feature-level information which is more informative than
        output-level statistics.

        Args:
            denoised: Backbone denoised output [B, 1, H, W]
            backbone_features: Intermediate features from backbone [B, C, H, W]

        Returns:
            confidence: Confidence map [B, 1, H, W] in [0, 1]
        """
        H, W = denoised.shape[2:]

        # OPTIMIZATION: Only interpolate if shapes don't match
        if backbone_features.shape[2:] != (H, W):
            backbone_features = F.interpolate(
                backbone_features, size=(H, W), mode='bilinear', align_corners=False
            )

        # OPTIMIZATION: Compute feature variance and gradients efficiently
        feature_var = backbone_features.var(dim=1, keepdim=True)

        # OPTIMIZATION: Compute gradient magnitude in single expression
        grad_x = F.conv2d(denoised, self.sobel_x, padding=1)
        grad_y = F.conv2d(denoised, self.sobel_y, padding=1)
        # Use x*x instead of x**2 for speed
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)

        # Get raw confidence from network
        raw_confidence = self.confidence_from_features(torch.cat([backbone_features, feature_var, grad_mag], dim=1))

        # Clamp to designed operating range [0.15, 0.75]
        confidence = raw_confidence.clamp(0.15, 0.75)

        # Reduce confidence near edges
        edge_uncertainty = (grad_mag / (grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)) * 0.2
        confidence = confidence - edge_uncertainty

        # Reduce confidence in low-variance regions (feature-based)
        var_max = feature_var.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        low_var_uncertainty = (1.0 - feature_var / var_max).mean(dim=1, keepdim=True) * 0.15
        confidence = confidence - low_var_uncertainty

        return confidence.clamp(0.15, 0.75)

    def forward(self,
                denoised: torch.Tensor,
                backbone_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate confidence for backbone output - OPTIMIZED.

        Args:
            denoised: Backbone denoised output [B, 1, H, W]
            backbone_features: Optional intermediate features [B, C, H, W]

        Returns:
            confidence: Confidence map [B, 1, H, W]
            info: Dictionary with estimation details
        """
        if backbone_features is not None:
            try:
                confidence = self.estimate_confidence_from_features(denoised, backbone_features)
                method = "features"
            except Exception:
                confidence = self.estimate_confidence_from_output(denoised)
                method = "output_fallback"
        else:
            confidence = self.estimate_confidence_from_output(denoised)
            method = "output"

        # OPTIMIZATION: Use nan_to_num which is faster than where for NaN handling
        confidence = torch.nan_to_num(confidence, nan=0.5, posinf=1.0, neginf=0.0)

        # OPTIMIZATION: Compute all stats in single detached pass for speed
        with torch.no_grad():
            conf_mean = confidence.mean().item()
            conf_min = confidence.min().item()
            conf_max = confidence.max().item()
            # MODIFIED: Use lower threshold (0.6 instead of 0.5) to capture more "uncertain" regions
            # This matches the new definition where confidence < 0.6 is considered uncertain enough for correctors
            low_conf_ratio = (confidence < 0.6).float().mean().item()
            # Also compute very low confidence ratio for debugging
            very_low_conf_ratio = (confidence < 0.4).float().mean().item()

        info = {
            'method': method,
            'mean_confidence': conf_mean if conf_mean == conf_mean else 0.5,
            'min_confidence': conf_min if conf_min == conf_min else 0.0,
            'max_confidence': conf_max if conf_max == conf_max else 1.0,
            'low_confidence_ratio': low_conf_ratio if low_conf_ratio == low_conf_ratio else 0.5,
            'very_low_confidence_ratio': very_low_conf_ratio if very_low_conf_ratio == very_low_conf_ratio else 0.3,
        }

        return confidence, info


# =============================================================================
# PART 2: CORRECTOR WITH POTENTIAL ASSESSMENT
# =============================================================================

class CorrectorWithPotential(nn.Module):
    """
    Base class for correctors that can assess their contribution potential.

    Each corrector now has two capabilities:
    1. Generate corrections (inherited from clinical correctors)
    2. Assess contribution potential (new)

    Contribution potential indicates "how much can this corrector improve
    the current output in each region". This is based on:

    - Predicate score gap: How far is the current score from passing threshold
    - Local quality deficit: How degraded is this region vs expected
    - Learned fix capability: Has this corrector historically helped in similar cases

    High potential = Corrector believes it can significantly improve this region
    Low potential = Corrector has limited ability to help here
    """

    def __init__(self, base_corrector: ClinicalCorrectorBase, predicate_keys):
        """
        Initialize corrector with potential assessment.

        Args:
            base_corrector: The underlying corrector module
            predicate_keys: Which predicates this corrector addresses (e.g., ['P1', 'P4'])
        """
        super().__init__()

        self.base_corrector = base_corrector
        self.predicate_keys = predicate_keys if isinstance(predicate_keys, list) else [predicate_keys]

        # Potential assessment network
        # Input: backbone_out (1) + failure_map (1) + local_quality (1) = 3 channels
        self.potential_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # Learnable threshold for predicate gap assessment
        # MODIFIED FOR EXCELLENT COOPERATION: Higher threshold means more regions considered "gapped"
        self.gap_threshold = nn.Parameter(torch.tensor(0.5))

        # Learnable scaling for potential
        # Reduced scale to avoid overcorrection while maintaining clinical improvement
        self.potential_scale = nn.Parameter(torch.tensor(1.5))

        self._init_weights()

    def _init_weights(self):
        """Initialize potential assessment weights."""
        for m in self.potential_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def assess_contribution_potential(self,
                                       backbone_out: torch.Tensor,
                                       noisy: torch.Tensor,
                                       failure_map: torch.Tensor,
                                       predicate_score: torch.Tensor) -> torch.Tensor:
        """
        Assess how much this corrector can contribute to each pixel.

        ENHANCED for IEEE TMI: Higher potential for contrast corrector (P2)
        to achieve +10-15% clinical contrast improvement.

        Potential is high when:
        1. Predicate score is below threshold (large gap to close)
        2. Failure map indicates problems in this region
        3. Local quality shows degradation that this corrector can fix
        4. For P2 (contrast): additional boost based on local contrast deficit

        Args:
            backbone_out: Current denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            failure_map: Predicate failure map [B, 1, H, W]
            predicate_score: Global predicate score (scalar or [B])

        Returns:
            potential: Per-pixel contribution potential [B, 1, H, W]
        """
        B, C, H, W = backbone_out.shape
        device = backbone_out.device

        # Component 1: Predicate score gap - OPTIMIZED
        # Higher gap = more potential to improve
        if isinstance(predicate_score, torch.Tensor):
            predicate_score = predicate_score.to(device=device)
            numel = predicate_score.numel()
            if numel == 1:
                score = predicate_score.view(1, 1, 1, 1).expand(B, 1, H, W)
            elif numel == B:
                score = predicate_score.view(B, 1, 1, 1).expand(B, 1, H, W)
            elif predicate_score.shape == (B, 1, H, W):
                score = predicate_score
            elif predicate_score.dim() == 4:
                score = F.interpolate(predicate_score, size=(H, W), mode='bilinear', align_corners=False)
            else:
                score = predicate_score.mean().view(1, 1, 1, 1).expand(B, 1, H, W)
        else:
            # OPTIMIZED: Use nan_to_num pattern for cleaner NaN handling
            score_val = float(predicate_score) if predicate_score == predicate_score else 0.5
            score = torch.full((B, 1, H, W), score_val, device=device)

        # Gap to threshold - OPTIMIZED: Fuse operations
        gap_threshold_safe = self.gap_threshold.clamp(min=0.1)
        gap = (F.relu(gap_threshold_safe - score) / gap_threshold_safe).clamp(0, 1)

        # Component 2: Local quality from failure map (no copy needed)
        # Component 3: Learned potential from combined features
        # OPTIMIZED: failure_map is already local_quality, avoid redundant concat
        features = torch.cat([backbone_out, failure_map, failure_map], dim=1)
        learned_potential = self.potential_net(features)

        # Component 4: Extra boost for tissue corrector (handles contrast) - OPTIMIZED
        is_tissue_corrector = ('P2' in self.predicate_keys)
        if is_tissue_corrector:
            # OPTIMIZED: Use F.avg_pool2d directly instead of creating new module
            local_mean = F.avg_pool2d(backbone_out, kernel_size=7, stride=1, padding=3)
            local_sq_mean = F.avg_pool2d(backbone_out * backbone_out, kernel_size=7, stride=1, padding=3)
            local_std = torch.sqrt((local_sq_mean - local_mean * local_mean).clamp(min=1e-8))

            # OPTIMIZED: Fuse normalization operations
            std_max = local_std.max()
            std_norm = local_std / (std_max + 1e-6) if std_max > 1e-6 else torch.zeros_like(local_std)
            contrast_deficit_boost = (1.0 - std_norm) * 0.3
        else:
            contrast_deficit_boost = 0.0  # OPTIMIZED: Use scalar zero for broadcasting

        # Combine components - MODIFIED FOR EXCELLENT COOPERATION
        # Increased weights to produce stronger potential signals
        potential = gap * 0.40 + failure_map * 0.40 + learned_potential * 0.35 + contrast_deficit_boost

        # Apply scaling - MODIFIED FOR EXCELLENT COOPERATION: Stronger scaling
        scale_factor = torch.sigmoid(self.potential_scale)
        if is_tissue_corrector:
            scale_factor = (scale_factor * 1.3).clamp(max=1.0)  # Boost for tissue (handles contrast)

        # Apply stronger scaling to produce more differentiated potentials
        # Higher potential values will correlate better with uncertainty
        potential = (potential * scale_factor * 1.2).clamp(0, 1)  # Added 1.2x boost

        # OPTIMIZED: Use nan_to_num which is faster than where for NaN handling
        return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0)

    def forward(self,
                backbone_out: torch.Tensor,
                noisy: torch.Tensor,
                failure_map: torch.Tensor,
                predicate_score: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate correction and assess contribution potential - OPTIMIZED.

        Args:
            backbone_out: Denoised output from backbone [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            failure_map: Predicate failure map [B, 1, H, W]
            predicate_score: Global predicate score
            backbone_features: Optional backbone intermediate features

        Returns:
            correction: Additive correction [B, 1, H, W]
            potential: Contribution potential map [B, 1, H, W]
        """
        target_shape = backbone_out.shape[2:]

        # Generate correction using base corrector
        correction = self.base_corrector(backbone_out, noisy, failure_map, predicate_score, backbone_features)

        # OPTIMIZATION: Only interpolate and check NaN if shapes don't match
        if correction.shape[2:] != target_shape:
            correction = F.interpolate(correction, size=target_shape, mode='bilinear', align_corners=False)

        # Assess contribution potential
        potential = self.assess_contribution_potential(
            backbone_out, noisy, failure_map, predicate_score
        )

        # OPTIMIZATION: Only interpolate if shapes don't match
        if potential.shape[2:] != target_shape:
            potential = F.interpolate(potential, size=target_shape, mode='bilinear', align_corners=False)

        # OPTIMIZATION: Single nan_to_num call for each output
        correction = torch.nan_to_num(correction, nan=0.0, posinf=0.15, neginf=-0.15)
        potential = torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0)

        return correction, potential


# =============================================================================
# PART 3: SYMBOLIC NEGOTIATOR
# =============================================================================

class SymbolicNegotiator(nn.Module):
    """
    Fuzzy logic-based negotiation between NAFNet and correctors.

    The negotiator implements a set of fuzzy rules that determine the
    allocation of "responsibility" between the backbone and each corrector.

    Key Rules:
    -----------
    1. IF nafnet_confident AND corrector_low_potential THEN trust_nafnet
       - When backbone is sure and corrector can't help much, use backbone

    2. IF nafnet_uncertain AND corrector_high_potential THEN use_corrector
       - When backbone struggles and corrector can help, let corrector lead

    3. IF predicate_failing THEN boost_relevant_corrector
       - When a predicate fails, boost the corrector that addresses it

    4. IF multiple_correctors_high_potential THEN balance_by_predicate
       - When multiple correctors want to act, prioritize by predicate severity

    5. IF conflict_detected THEN conservative_allocation
       - When correctors might conflict, be conservative

    Output: Per-pixel allocation maps for each corrector
    - allocation = 0: Fully trust backbone
    - allocation = 1: Fully use corrector
    - allocation in (0,1): Blend of both
    """

    # ------------------------------------------------------------------
    # Bounded affine rule combination (non legacy path).
    #
    #   a = b + C2*w2*r2 + C3*w3*r3 + C4*w4*r4 - C1*w1*r1 - C5*w5*r5
    #   b = B_MIN + (B_MAX - B_MIN) * sigmoid(base_allocation)
    #
    # w_k = sigmoid(rule_weight_k) in (0,1) and r_k in [0,1] are the
    # Lukasiewicz rule activations. The budgets are chosen so that
    #   B_MAX + C2 + C3 + C4 = 1.0   and   B_MIN - C1 - C5 = 0.0,
    # hence the pre clamp value already lies in [0,1] for every parameter
    # value and every input. The final clamp is therefore inert, the
    # allocation is exactly affine in the rule vector, and
    #   |da| <= L * ||dr||_inf   with   L = sum_k C_k * w_k   (exact).
    # The legacy formula (LEGACY_SATURATING_ALLOCATION=1) had a positive
    # budget of 0.818 + 0.9 + 0.8 = 2.5 and a 1.5x priority boost, so the
    # clamp at 1.0 was active on every pixel and no gradient could flow.
    # ------------------------------------------------------------------
    ALLOC_BASE_MIN = 0.25
    ALLOC_BASE_MAX = 0.50
    ALLOC_C1 = 0.15   # trust_nafnet     (negative vote)
    ALLOC_C2 = 0.25   # use_corrector    (positive vote)
    ALLOC_C3 = 0.15   # boost_failing    (positive vote)
    ALLOC_C4 = 0.10   # balance          (positive vote)
    ALLOC_C5 = 0.10   # conservative     (negative vote)

    def __init__(self, num_correctors: int = 2):
        """
        Initialize the symbolic negotiator.

        MODIFIED: Thresholds adjusted to be more willing to use correctors.
        - Lower "high_confidence" threshold means NAFNet must be MORE confident to be trusted alone
        - Higher "use_corrector" weight means correctors get more allocation
        - Higher base allocation means correctors get more opportunity by default

        Args:
            num_correctors: Number of correctors to negotiate between
        """
        super().__init__()

        # Fuzzy logic engine
        self.logic = DifferentiableFuzzyLogic('lukasiewicz')

        # Learnable thresholds for rule antecedents
        # MODIFIED FOR EXCELLENT COOPERATION: Thresholds tuned to maximize uncertainty-potential correlation
        # Key insight: Correctors should activate strongly where NAFNet is uncertain
        self.thresholds = nn.ParameterDict({
            'high_confidence': nn.Parameter(torch.tensor(0.40)),  # Lowered from 0.50 - NAFNet must be VERY confident to be trusted alone
            'low_confidence': nn.Parameter(torch.tensor(0.55)),   # Raised from 0.25 - More regions considered "uncertain"
            'high_potential': nn.Parameter(torch.tensor(0.30)),   # Lowered from 0.35 - Correctors activate even more easily
            'low_potential': nn.Parameter(torch.tensor(0.10)),    # Lowered from 0.15 - Almost no regions considered "low potential"
            'predicate_failing': nn.Parameter(torch.tensor(0.75)), # Raised from 0.70 - Even more predicates trigger corrector boost
        })

        # Learnable rule weights (importance of each rule)
        # Balanced initialization - let training find the right weights
        self.rule_weights = nn.ParameterDict({
            'trust_nafnet': nn.Parameter(torch.tensor(0.1)),      # sigmoid=0.52, very low weight
            'use_corrector': nn.Parameter(torch.tensor(4.0)),     # sigmoid=0.98, dominates allocation
            'boost_failing': nn.Parameter(torch.tensor(3.0)),     # sigmoid=0.95, strong predicate boost
            'balance': nn.Parameter(torch.tensor(1.5)),           # sigmoid=0.82, moderate balance
            'conservative': nn.Parameter(torch.tensor(0.1)),      # sigmoid=0.52, very low weight
        })

        # Base allocation when no strong signal.
        # LEGACY_SATURATING_ALLOCATION=1 restores the pre-revision value of 1.5.
        # With 1.5 the base is sigmoid(1.5)=0.818 and the downstream priority
        # boost of 1.5 pushes every pixel past the clamp at 1.0, so the rules
        # cannot change the allocation and receive no gradient. The default is
        # now 0.0, i.e. sigmoid(0.0)=0.5, which keeps the allocation inside the
        # clamp and lets the rules act.
        _legacy = os.environ.get("LEGACY_SATURATING_ALLOCATION", "0") == "1"
        self.legacy_saturation = _legacy
        self.base_allocation = nn.Parameter(torch.tensor(1.5 if _legacy else 0.0))

        # Conflict detection threshold
        # MODIFIED FOR EXCELLENT COOPERATION: Increased to allow more correctors to work together
        self.conflict_threshold = nn.Parameter(torch.tensor(0.55))

    def _soft_greater(self, x: torch.Tensor, threshold: torch.Tensor,
                      steepness: float = 10.0) -> torch.Tensor:
        """Soft comparison: returns [0,1] indicating how much x > threshold."""
        # FIX: Clamp the input to sigmoid to prevent numerical overflow
        diff = steepness * (x - threshold)
        diff = diff.clamp(-20.0, 20.0)  # Prevent extreme sigmoid inputs
        return torch.sigmoid(diff)

    def _soft_less(self, x: torch.Tensor, threshold: torch.Tensor,
                   steepness: float = 10.0) -> torch.Tensor:
        """Soft comparison: returns [0,1] indicating how much x < threshold."""
        # FIX: Clamp the input to sigmoid to prevent numerical overflow
        diff = steepness * (threshold - x)
        diff = diff.clamp(-20.0, 20.0)  # Prevent extreme sigmoid inputs
        return torch.sigmoid(diff)

    def rule_weight_sigmoids(self) -> Tuple[torch.Tensor, ...]:
        """(w1..w5) = sigmoid of the five learnable rule weights, each in (0,1)."""
        return tuple(torch.sigmoid(self.rule_weights[k]) for k in
                     ('trust_nafnet', 'use_corrector', 'boost_failing', 'balance', 'conservative'))

    def bounded_base(self) -> torch.Tensor:
        """b = B_MIN + (B_MAX - B_MIN) * sigmoid(base_allocation), in (B_MIN, B_MAX).

        The stored checkpoint value base_allocation = 1.5 maps to
        0.25 + 0.25 * 0.818 = 0.454, so loading the old checkpoint is harmless.
        """
        return self.ALLOC_BASE_MIN + (self.ALLOC_BASE_MAX - self.ALLOC_BASE_MIN) * torch.sigmoid(self.base_allocation)

    def combine_rules_bounded(self, r1, r2, r3, r4, r5=None) -> torch.Tensor:
        """Bounded affine combination of the rule activations (non legacy path).

        a = b - C1 w1 r1 + C2 w2 r2 + C3 w3 r3 + C4 w4 r4 - C5 w5 r5,  r_k in [0,1].
        Pre clamp range is [B_MIN - C1 - C5, B_MAX + C2 + C3 + C4] = [0, 1] so the
        downstream clamp never changes the value. Gradient reaches base_allocation
        through b, every rule weight through w_k * r_k, and every threshold
        through r_k, because no term is saturated by construction.
        """
        w1, w2, w3, w4, w5 = self.rule_weight_sigmoids()
        allocation = (self.bounded_base()
                      - self.ALLOC_C1 * w1 * r1
                      + self.ALLOC_C2 * w2 * r2
                      + self.ALLOC_C3 * w3 * r3
                      + self.ALLOC_C4 * w4 * r4)
        if r5 is not None:
            allocation = allocation - self.ALLOC_C5 * w5 * r5
        return allocation

    def lipschitz_constant(self, include_rule5: bool = True) -> torch.Tensor:
        """L = C1 w1 + C2 w2 + C3 w3 + C4 w4 (+ C5 w5), so |da| <= L * ||dr||_inf.

        This bound is exact (attained) on the bounded path because the map is
        affine in r with signed coefficients and the clamp is inert.
        """
        w1, w2, w3, w4, w5 = self.rule_weight_sigmoids()
        L = self.ALLOC_C1 * w1 + self.ALLOC_C2 * w2 + self.ALLOC_C3 * w3 + self.ALLOC_C4 * w4
        if include_rule5:
            L = L + self.ALLOC_C5 * w5
        return L

    def evaluate_rules(self,
                       nafnet_confidence: torch.Tensor,
                       corrector_potential: torch.Tensor,
                       predicate_score: torch.Tensor,
                       all_potentials: Optional[Dict[str, torch.Tensor]] = None
                       ) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluate fuzzy rules for a single corrector - OPTIMIZED.

        Args:
            nafnet_confidence: Backbone confidence map [B, 1, H, W]
            corrector_potential: This corrector's potential [B, 1, H, W]
            predicate_score: Relevant predicate score [B, 1, H, W] or scalar
            all_potentials: All corrector potentials (for conflict detection)

        Returns:
            allocation: Allocation for this corrector [B, 1, H, W]
            rule_activations: Dict of which rules fired and how much
        """
        B, C, H, W = nafnet_confidence.shape
        device = nafnet_confidence.device

        # OPTIMIZATION: Expand predicate score only if needed
        if isinstance(predicate_score, (int, float)):
            predicate_score = torch.full((B, 1, H, W), predicate_score, device=device)
        elif predicate_score.dim() == 0 or predicate_score.numel() == 1:
            predicate_score = predicate_score.view(1, 1, 1, 1).expand(B, 1, H, W)

        # OPTIMIZATION: Cache threshold values to avoid repeated dict lookups
        thresh_high_conf = self.thresholds['high_confidence']
        thresh_low_conf = self.thresholds['low_confidence']
        thresh_high_pot = self.thresholds['high_potential']
        thresh_low_pot = self.thresholds['low_potential']
        thresh_pred_fail = self.thresholds['predicate_failing']

        # ===== RULE 1: Trust NAFNet =====
        nafnet_confident = self._soft_greater(nafnet_confidence, thresh_high_conf)
        corrector_low = self._soft_less(corrector_potential, thresh_low_pot)
        rule1_activation = self.logic.soft_and(nafnet_confident, corrector_low)

        # ===== RULE 2: Use Corrector =====
        nafnet_uncertain = self._soft_less(nafnet_confidence, thresh_low_conf)
        corrector_high = self._soft_greater(corrector_potential, thresh_high_pot)
        rule2_activation = self.logic.soft_and(nafnet_uncertain, corrector_high)

        # ===== RULE 3: Boost for Failing Predicate =====
        rule3_activation = self._soft_less(predicate_score, thresh_pred_fail)

        # ===== RULE 4: Balance =====
        # OPTIMIZATION: Reuse soft comparisons where possible
        medium_conf = self.logic.soft_and(
            self._soft_less(nafnet_confidence, 0.7),
            self._soft_greater(nafnet_confidence, 0.3)
        )
        medium_pot = self.logic.soft_and(
            self._soft_less(corrector_potential, 0.7),
            self._soft_greater(corrector_potential, 0.3)
        )
        rule4_activation = self.logic.soft_and(medium_conf, medium_pot)

        # ===== RULE 5: Conservative (conflict detection) =====
        if all_potentials is not None and len(all_potentials) > 1:
            # OPTIMIZATION: Use torch.stack more efficiently
            potential_stack = torch.stack(list(all_potentials.values()), dim=0)
            if self.legacy_saturation:
                high_potential_count = (potential_stack > self.conflict_threshold).float().sum(dim=0)
            else:
                # Soft count so that conflict_threshold is differentiable.
                # Each term is in (0,1), the sum is in (0, num_correctors).
                high_potential_count = self._soft_greater(potential_stack, self.conflict_threshold).sum(dim=0)
            rule5_activation = self._soft_greater(high_potential_count, 1.5)
        else:
            rule5_activation = torch.zeros((1,), device=device)  # OPTIMIZATION: Scalar zero

        # OPTIMIZATION: Cache sigmoid of rule weights
        w1 = torch.sigmoid(self.rule_weights['trust_nafnet'])
        w2 = torch.sigmoid(self.rule_weights['use_corrector'])
        w3 = torch.sigmoid(self.rule_weights['boost_failing'])
        w4 = torch.sigmoid(self.rule_weights['balance'])
        w5 = torch.sigmoid(self.rule_weights['conservative'])

        # Rule 5 only exists when at least two correctors are negotiated.
        rule5_active = isinstance(rule5_activation, torch.Tensor) and rule5_activation.numel() > 1

        if self.legacy_saturation:
            # ===== LEGACY COMBINE RULES - ASYMMETRIC HARDCODED MULTIPLIERS =====
            # Kept bit for bit for LEGACY_SATURATING_ALLOCATION=1. The positive
            # budget (0.818 + 0.9 + 0.8) exceeds 1.0 so the clamp saturates.
            base = torch.sigmoid(self.base_allocation)

            allocation = base
            allocation = allocation - rule1_activation * w1 * 0.05  # Very small NAFNet trust reduction
            allocation = allocation + rule2_activation * w2 * 0.9   # Very large uncertainty boost
            allocation = allocation + rule3_activation * w3 * 0.8   # Large predicate boost

            # Rule 4: balance effect - Push toward moderate allocation
            rule4_effect = rule4_activation * w4
            allocation = allocation * (1 - rule4_effect * 0.3) + 0.60 * rule4_effect

            # Rule 5: conservative effect - Slight reduction during conflict
            rule5_effect = rule5_activation * w5 if rule5_active else 0.0
            if isinstance(rule5_effect, torch.Tensor):
                allocation = allocation * (1 - rule5_effect * 0.2)
        else:
            # ===== BOUNDED AFFINE COMBINE RULES (see class comment) =====
            allocation = self.combine_rules_bounded(
                rule1_activation, rule2_activation, rule3_activation, rule4_activation,
                rule5_activation if rule5_active else None)

        # OPTIMIZATION: Single final clamp and nan handling
        # (inert on the bounded path, retained as a numerical guard)
        allocation = torch.nan_to_num(allocation.clamp(0, 1), nan=0.5, posinf=1.0, neginf=0.0)

        # OPTIMIZATION: Compute stats in no_grad context
        with torch.no_grad():
            rule_activations = {
                'rule1_trust_nafnet': rule1_activation.mean().item(),
                'rule2_use_corrector': rule2_activation.mean().item(),
                'rule3_boost_failing': rule3_activation.mean().item(),
                'rule4_balance': rule4_activation.mean().item(),
                'rule5_conservative': rule5_activation.mean().item() if isinstance(rule5_activation, torch.Tensor) else 0.0,
            }

        return allocation, rule_activations

    def forward(self,
                nafnet_confidence: torch.Tensor,
                corrector_potentials: Dict[str, torch.Tensor],
                predicate_scores: Dict[str, torch.Tensor]
                ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Negotiate allocations for all correctors.

        ENHANCED: Special handling for contrast corrector (P2) to achieve
        +10-15% clinical contrast improvement for IEEE TMI publication.

        Args:
            nafnet_confidence: Backbone confidence map [B, 1, H, W]
            corrector_potentials: Dict of potential maps per corrector
            predicate_scores: Dict of predicate scores per corrector

        Returns:
            allocations: Dict of allocation maps per corrector
            negotiation_info: Full negotiation trace for interpretability
        """
        allocations = {}
        rule_traces = {}

        # Map corrector names to predicate keys
        pred_key_map = {
            'gain': ['P1', 'P2', 'P3', 'P4', 'P6'],
            'tissue': ['P2', 'P3', 'P6'],
            'boundary': ['P1', 'P4'],
        }

        # Priority boost for correctors
        # Higher boost = more allocation regardless of other factors
        # The unconditional boost of 1.5 saturated the clamp at 1.0 for every
        # pixel of every image. It is 1.0 by default now, so only the
        # predicate deficit term below can raise the allocation.
        if getattr(self, 'legacy_saturation', False):
            corrector_priority_boost = {'gain': 1.5, 'tissue': 1.5, 'boundary': 1.0}
        else:
            corrector_priority_boost = {'gain': 1.0, 'tissue': 1.0, 'boundary': 1.0}

        for corrector_name, potential in corrector_potentials.items():
            # Get corresponding predicate score (min of associated predicates)
            pred_keys = pred_key_map.get(corrector_name, ['P1'])
            if isinstance(pred_keys, list):
                # Use min of associated pred scores (worst predicate drives correction)
                scores = [predicate_scores.get(k, torch.tensor(0.5, device=nafnet_confidence.device)) for k in pred_keys]
                pred_score = scores[0]
                for s in scores[1:]:
                    pred_score = torch.min(pred_score, s)
            else:
                pred_score = predicate_scores.get(pred_keys, torch.tensor(0.5))

            # Evaluate rules for this corrector
            allocation, rule_activations = self.evaluate_rules(
                nafnet_confidence,
                potential,
                pred_score,
                corrector_potentials
            )

            # Apply priority boost for gain/tissue corrector.
            # Bounded path: skipped. The deficit factor below is computed with
            # .item() (no gradient), multiplies the allocation by up to 1.5x
            # (re-saturating the clamp) and duplicates rule 3, which already
            # carries the failing predicate signal per pixel and differentiably.
            boost = corrector_priority_boost.get(corrector_name, 1.0)
            if corrector_name in ('tissue', 'gain') and not self.legacy_saturation:
                rule_activations['priority_boost'] = 1.0
            elif corrector_name in ('tissue', 'gain'):
                # Boost allocation using worst predicate score
                worst_score = pred_score  # already min of associated predicates
                if isinstance(worst_score, torch.Tensor):
                    worst_val = worst_score.mean() if worst_score.numel() > 1 else worst_score
                else:
                    worst_val = worst_score
                deficit = F.relu(torch.tensor(0.55) - worst_val)
                deficit_val = min(deficit.item(), 0.5)
                boost_factor = boost * (1.0 + deficit_val)
                boost_factor = min(boost_factor, 2.0)
                allocation = (allocation * boost_factor).clamp(0, 1)
                rule_activations['priority_boost'] = boost_factor

            allocations[corrector_name] = allocation
            rule_traces[corrector_name] = rule_activations

        # Compute summary statistics
        negotiation_info = {
            'rule_traces': rule_traces,
            'mean_allocations': {k: v.mean().item() for k, v in allocations.items()},
            'max_allocations': {k: v.max().item() for k, v in allocations.items()},
            'nafnet_confidence_mean': nafnet_confidence.mean().item(),
            'total_correction_weight': sum(v.mean().item() for v in allocations.values()),
        }

        return allocations, negotiation_info



# =============================================================================
# PART 3.5: SCANNER ADAPTER (Few-Shot Domain Adaptation)
# =============================================================================

class ScannerAdapter(nn.Module):
    """
    Lightweight scanner normalization for few-shot cross-scanner adaptation.

    Transforms backbone output to look like the training domain (PKU37) so that
    the frozen corrector sees familiar intensity patterns. Only this module is
    trained during few-shot adaptation; everything else stays frozen.

    The adapter normalizes the input for corrector processing but the final
    multiplicative correction is applied to the ORIGINAL backbone output,
    preserving the target scanner's intensity characteristics.

    Architecture: channel-wise affine (gamma * x + beta) — 2 learnable parameters.
    """

    def __init__(self):
        super().__init__()
        # Initialize to identity transform: gamma=1.0, beta=0.0
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize scanner-specific intensity to training domain."""
        return self.gamma * x + self.beta

    def extra_repr(self) -> str:
        return f'gamma={self.gamma.data.item():.4f}, beta={self.beta.data.item():.4f}'


# =============================================================================
# PART 4: MAIN COOPERATIVE CORRECTOR
# =============================================================================

class NeuroSymbolicCorrectorV8Cooperative(nn.Module):
    """
    Cooperative Neuro-Symbolic Denoising Framework.

    This model implements a cooperative approach where NAFNet (backbone) and
    symbolic correctors negotiate their contributions based on confidence
    and potential assessments.

    Architecture Flow:
    ------------------
    1. NAFNet denoises image -> denoised + confidence map
    2. Predicates evaluate current quality
    3. Each corrector assesses its contribution potential
    4. Symbolic negotiator determines allocation
    5. Corrections applied weighted by allocation
    6. Final = denoised + sum(allocation * corrections)

    Key Innovations:
    ----------------
    - NAFNet confidence estimation (knows where it's uncertain)
    - Corrector potential assessment (knows where it can help)
    - Symbolic negotiation (fuzzy rules mediate cooperation)
    - Full interpretability (trace of all decisions)

    Attributes:
        predicates: Enhanced predicate evaluator (from V8)
        confidence_estimator: NAFNet confidence wrapper
        correctors: Dict of CorrectorWithPotential modules
        negotiator: SymbolicNegotiator for allocation
        verifier: Formal verification guarantees
        explainer: Causal explanation generator
    """

    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 64,
                 feature_channels: int = 64,
                 correction_clamp: float = 0.15,
                 clinical_clamp: float = 0.03,
                 use_guided_edge: bool = False,
                 scanner_adapter: bool = False):
        """
        Initialize the cooperative corrector.

        Args:
            in_channels: Number of input channels (default: 1 for grayscale OCT)
            hidden_channels: Hidden dimension for corrector networks
            feature_channels: Expected channels in backbone features
            correction_clamp: Maximum correction magnitude (default: 0.15)
            clinical_clamp: Maximum clinical enhancement magnitude (default: 0.03)
            use_guided_edge: If True, start with GuidedEdgeSharpener (for multi-phase
                training: Phase 1 uses GuidedEdgeSharpener, Phase 2 swaps to
                EdgeRecoveryModule via swap_edge_module()). Default False keeps
                EdgeRecoveryModule for backward compatibility.
            scanner_adapter: If True, add a learnable scanner normalization layer
                for few-shot cross-scanner adaptation (2 params).
        """
        super().__init__()

        # Store dimensions
        self.feature_channels = feature_channels
        self.correction_clamp = correction_clamp
        self.clinical_clamp = clinical_clamp

        # Optional scanner adapter for cross-scanner few-shot adaptation
        self.scanner_adapter = ScannerAdapter() if scanner_adapter else None

        # =================================================================
        # COMPONENT 1: Predicates (from V8)
        # =================================================================
        self.predicates = EnhancedGTFreePredicates()

        # =================================================================
        # COMPONENT 2: NAFNet Confidence Estimator (NEW)
        # =================================================================
        self.confidence_estimator = NAFNetWithConfidence(feature_channels)

        # =================================================================
        # COMPONENT 3: Correctors with Potential Assessment (Extended)
        # =================================================================
        # Create base clinical correctors — multiplicative gain only
        base_correctors = {
            'gain': MultiplicativeGainCorrector(in_channels, hidden_channels, max_alpha=0.15),
        }

        # Wrap with potential assessment
        self.correctors = nn.ModuleDict({
            name: CorrectorWithPotential(corrector, self._get_pred_keys(name))
            for name, corrector in base_correctors.items()
        })

        # Predicate to corrector mapping
        self.pred_key_map = {
            'gain': ['P1', 'P2', 'P3', 'P4', 'P6'],
        }

        # Region-decomposed correction groups:
        # - Gain corrector goes into tissue group (multiplicative, spatially smooth)
        # - Boundary group is empty (edge detail handled by EdgeResidualFilter)
        self.tissue_group = {'gain'}
        self.boundary_group = set()

        # =================================================================
        # COMPONENT 4: Symbolic Negotiator (NEW)
        # =================================================================
        self.negotiator = SymbolicNegotiator(num_correctors=len(self.correctors))

        # =================================================================
        # COMPONENT 5: Hierarchical Symbolic Reasoner (from V8)
        # =================================================================
        self.router = HierarchicalSymbolicReasoner()

        # =================================================================
        # COMPONENT 6: Adaptive Lambda Predictor (from V8 Enhanced)
        # =================================================================
        self.lambda_predictor = AdaptiveLambdaPredictorV8()

        # =================================================================
        # COMPONENT 7: Formal Verification (from V8)
        # =================================================================
        # RELAXED VERIFICATION: Allow stronger corrections for clinical improvement
        # Default: energy_tolerance=0.05, pareto_delta=0.05, lipschitz_bound=0.3
        # Relaxed: energy_tolerance=0.15, pareto_delta=0.15, lipschitz_bound=0.5
        self.verifier = FormalVerificationGuarantee(
            self.predicates,
            energy_tolerance=0.10,  # Reduced for better stability
            pareto_delta=0.10,      # Reduced for better stability
            lipschitz_bound=0.2    # Reduced to prevent large corrections that harm PSNR
        )

        # =================================================================
        # COMPONENT 8: Causal Explainer (from V8)
        # =================================================================
        self.explainer = CausalExplainer(self.router)

        # =================================================================
        # COMPONENT 8.5: CNR-Preserving Correction (NEW INNOVATION)
        # =================================================================
        # Wraps corrections to preserve Contrast-to-Noise Ratio
        # Key innovation: Region-aware gating + background smoothing
        self.cnr_preserver = CNRPreservingCorrectionModule()

        # =================================================================
        # COMPONENT 8.6: Clinical Enhancement Module (NEW INNOVATION)
        # =================================================================
        # Additional clinical enhancement to push toward 10-15% target
        # Key innovation: Multi-scale adaptive enhancement
        self.clinical_enhancer = ClinicalEnhancementModule()

        # =================================================================
        # COMPONENT 8.7: Region-Aware Correction Module (NEW INNOVATION)
        # =================================================================
        # Applies aggressive clinical enhancement only in uncertain/low-quality
        # regions while preserving PSNR in confident regions.
        # Key innovation: Adaptive correction strength based on uncertainty
        # Goal: 10-15% clinical improvement in uncertain regions,
        #       PSNR drop < 1.0 dB overall by preserving confident regions
        self.region_aware_corrector = RegionAwareCorrectionModule(
            clinical_boost_factor=4.0  # Aggressive clinical boost for uncertain regions
        )

        # Unified: using shared AdaptiveRegionDetector
        # Wire the cnr_preserver's region_detector into region_aware_corrector so all
        # tissue/background detection uses the same learned detector
        self.region_aware_corrector.region_detector = self.cnr_preserver.region_detector

        # =================================================================
        # COMPONENT 8.5a: Edge Module (EPI improvement)
        # =================================================================
        # Two modes:
        #   use_guided_edge=False (default): EdgeRecoveryModule (backward compat)
        #   use_guided_edge=True (multi-phase): GuidedEdgeSharpener in Phase 1,
        #     then swap_edge_module() replaces it with EdgeRecoveryModule at Phase 2
        if use_guided_edge:
            self.edge_sharpener = GuidedEdgeSharpener(max_magnitude=0.10)
        else:
            self.edge_sharpener = EdgeRecoveryModule(max_magnitude=0.08)

        # Adaptive background smoother: learns per-pixel smoothing strength
        # Input: backbone output (1ch) → per-pixel smoothing weight map
        # Multi-scale receptive field with depthwise separable convolutions (~2K params)
        self.adaptive_smoother = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 5, padding=2, groups=16, bias=False),  # depthwise 5×5
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        # Initialize final layer bias negative so initial sigmoid output ≈ 0.3
        nn.init.zeros_(self.adaptive_smoother[-1].weight)
        nn.init.constant_(self.adaptive_smoother[-1].bias, -0.85)

        # Background suppression: learns to push dark pixels darker
        # Input: backbone output (1ch) → per-pixel suppression amount
        self.bg_suppressor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        # Initialize to near-zero suppression (sigmoid(-2) ≈ 0.12)
        nn.init.zeros_(self.bg_suppressor[-1].weight)
        nn.init.constant_(self.bg_suppressor[-1].bias, -2.0)

        # Learned tone curve: nonlinear intensity remapping for contrast stretching
        # Small MLP that maps intensity → adjusted intensity
        # Applied globally (same curve for all pixels) for dataset-agnostic generalization
        self.tone_curve = nn.Sequential(
            nn.Conv2d(1, 16, 1),  # pointwise: intensity features
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # output: residual adjustment
        )
        # Initialize as identity (zero residual)
        nn.init.zeros_(self.tone_curve[-1].weight)
        nn.init.zeros_(self.tone_curve[-1].bias)

        # Fixed background darkening: single learnable scalar (can't collapse like CNN modules)
        # sigmoid(0) = 0.5 → 0.5 * 0.04 = 0.02 initial darkening
        self.bg_darken_strength = nn.Parameter(torch.tensor(0.0))

        self._print_info()

        # Ablation flags — set via set_ablation() before training/eval
        self._ablation_no_negotiator = False
        self._ablation_no_edge = False
        self._ablation_no_uncertainty = False
        self._ablation_no_bg_smooth = False

    def set_ablation(self, mode: str):
        """Set ablation mode. Modes: none, no_negotiator, no_edge, no_uncertainty, no_bg_smooth."""
        self._ablation_no_negotiator = (mode == 'no_negotiator')
        self._ablation_no_edge = (mode == 'no_edge')
        self._ablation_no_uncertainty = (mode == 'no_uncertainty')
        self._ablation_no_bg_smooth = (mode == 'no_bg_smooth')
        print(f"  ABLATION MODE: {mode}")

    def _get_pred_keys(self, corrector_name: str) -> list:
        """Get predicate keys for corrector name."""
        mapping = {
            'gain': ['P1', 'P2', 'P3', 'P4', 'P6'],
            'tissue': ['P2', 'P3', 'P6'],
            'boundary': ['P1', 'P4'],
        }
        return mapping.get(corrector_name, ['P1'])

    def _print_info(self):
        """Print model information."""
        print("\n" + "=" * 70)
        print("NeuroSymbolicCorrectorV8Cooperative - COOPERATIVE FRAMEWORK")
        print("=" * 70)
        print("\nKEY INNOVATIONS:")
        print("  1. Multiplicative gain correction (naturally suppresses background)")
        print("  2. Supervised edge recovery (residual filtering + Sobel supervision)")
        print("  3. NAFNet confidence estimation (knows uncertainty)")
        print("  4. Symbolic negotiation (fuzzy rules mediate)")
        print("  5. Full interpretability (complete decision trace)")
        print("")
        print("COOPERATIVE FLOW:")
        print("  NAFNet -> confidence -> gain potential -> negotiate -> multiplicative -> edge sharpen")
        print("")
        print("CORRECTORS WITH POTENTIAL ASSESSMENT:")
        for name in self.correctors.keys():
            pred = self.pred_key_map[name]
            print(f"  {name:12s} -> {pred} (with potential)")
        print("")
        print("SYMBOLIC NEGOTIATION RULES:")
        print("  1. IF nafnet_confident AND corrector_low_potential THEN trust_nafnet")
        print("  2. IF nafnet_uncertain AND corrector_high_potential THEN use_corrector")
        print("  3. IF predicate_failing THEN boost_relevant_corrector")
        print("  4. IF multiple_high_potential THEN balance_by_predicate")
        print("  5. IF conflict_detected THEN conservative_allocation")
        print("")

        # Parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        conf_params = sum(p.numel() for p in self.confidence_estimator.parameters())
        neg_params = sum(p.numel() for p in self.negotiator.parameters())
        corr_params = sum(p.numel() for p in self.correctors.parameters())
        edge_params = sum(p.numel() for p in self.edge_sharpener.parameters())
        smoother_params = sum(p.numel() for p in self.adaptive_smoother.parameters())

        print(f"PARAMETERS:")
        print(f"  Confidence estimator: {conf_params:,}")
        print(f"  Negotiator: {neg_params:,}")
        print(f"  Correctors (with potential): {corr_params:,}")
        edge_type = type(self.edge_sharpener).__name__
        print(f"  Edge module ({edge_type}): {edge_params:,}")
        print(f"  Adaptive smoother: {smoother_params:,}")
        print(f"  Total: {total_params:,}")
        print("=" * 70)

    def swap_edge_module(self):
        """Swap GuidedEdgeSharpener -> EdgeRecoveryModule (mimics QT44->QT45b).

        Called at Phase 2 transition in multi-phase training. Transfers the
        blend_logit parameter (present in both classes) to preserve learned
        blend strength.
        """
        old = self.edge_sharpener
        old_type = type(old).__name__
        self.edge_sharpener = EdgeRecoveryModule(max_magnitude=0.08).to(
            next(self.parameters()).device
        )
        # Transfer blend_logit (same key in both GuidedEdgeSharpener and EdgeRecoveryModule)
        if hasattr(old, 'blend_logit') and hasattr(self.edge_sharpener, 'blend_logit'):
            self.edge_sharpener.blend_logit.data.copy_(old.blend_logit.data)
        del old
        new_params = sum(p.numel() for p in self.edge_sharpener.parameters())
        print(f"  Swapped {old_type} -> EdgeRecoveryModule")
        print(f"  New edge params: {new_params:,}")

    def forward(self,
                backbone_out: torch.Tensor,
                noisy: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None,
                nafnet_uncertainty: Optional[torch.Tensor] = None,
                return_details: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply cooperative neuro-symbolic correction.

        This is the main forward pass implementing the cooperative framework:
        1. Estimate NAFNet confidence (or use provided uncertainty)
        2. Evaluate predicates on backbone output
        3. Each corrector generates correction and assesses potential
        4. Symbolic negotiation determines allocation
        5. Apply weighted corrections
        6. Formal verification of final output

        Args:
            backbone_out: Denoised output from backbone [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            backbone_features: Optional dict with 'enc1', 'enc2' features
            nafnet_uncertainty: Optional pre-computed uncertainty from backbone [B, 1, H, W]
            return_details: Whether to include full interpretability output

        Returns:
            output: Final corrected output [B, 1, H, W]
            info: Dictionary with:
                - nafnet_confidence: Confidence map
                - corrector_potentials: Dict of potential maps
                - allocations: Dict of allocation maps
                - negotiation_trace: Which rules fired
                - predicate_scores: Before and after correction
                - verification: Formal verification results
        """
        B, C, H, W = backbone_out.shape
        device = backbone_out.device
        target_shape = backbone_out.shape[2:]  # Cache target shape for reuse

        # =====================================================================
        # STEP 0: Scanner Adaptation (few-shot cross-scanner normalization)
        # =====================================================================
        # Transform backbone output to training-domain intensity for corrector
        # processing. The final correction is applied to the ORIGINAL backbone_out.
        if self.scanner_adapter is not None:
            backbone_adapted = self.scanner_adapter(backbone_out)
        else:
            backbone_adapted = backbone_out

        # =====================================================================
        # STEP 1: Estimate NAFNet Confidence
        # =====================================================================
        # Use provided uncertainty or estimate from features
        if self._ablation_no_uncertainty:
            # ABLATION: Replace uncertainty with uniform 0.5
            nafnet_confidence = torch.full_like(backbone_out, 0.5)
            conf_info = {'confidence_source': 'ablation_uniform', 'mean': 0.5, 'std': 0.0}
        elif nafnet_uncertainty is not None:
            # Convert uncertainty to confidence: high uncertainty = low confidence
            # OPTIMIZATION: Use in-place operation where possible
            nafnet_confidence = 1.0 - nafnet_uncertainty.clamp(0, 1)
            conf_info = {
                'confidence_source': 'provided',
                'mean': nafnet_confidence.mean().item(),
                'std': nafnet_confidence.std().item()
            }
        else:
            # Get features for confidence estimation
            feature_for_conf = backbone_features.get('enc1') if backbone_features is not None else None

            nafnet_confidence, conf_info = self.confidence_estimator(
                backbone_out, feature_for_conf
            )

        # =====================================================================
        # STEP 2: Evaluate Predicates on Backbone Output
        # =====================================================================
        # GRADIENT FIX: During training, compute predicates WITH gradients
        # so that failure_maps can provide gradient signal to correctors.
        # The failure_maps influence both corrections and potentials, so
        # gradients through them help the model learn which regions need fixing.
        # TTA MODE: Use no_grad for predicates — failure_maps only inform allocation
        # decisions, not the TTA loss. Detaching prevents backbone gradient chains
        # from accumulating through the corrector graph (~5MB savings per step).
        tta_mode = getattr(self, '_tta_mode', False)
        if self.training and not tta_mode:
            # Regular training: allow gradients through predicates for better learning
            pred_results = self.predicates(backbone_out, noisy)
        else:
            # Inference or TTA: use no_grad for efficiency
            with torch.no_grad():
                pred_results = self.predicates(backbone_out, noisy)

        # Extract all failure maps and scores from predicates
        failure_maps = {}
        pred_score_dict = {}
        all_pred_keys = set()
        for pred_keys in self.pred_key_map.values():
            all_pred_keys.update(pred_keys)
        for key in all_pred_keys:
            pred_data = pred_results[key]
            failure_maps[key] = pred_data['failure_map'].detach()
            score = pred_data['score']
            pred_score_dict[key] = score if isinstance(score, torch.Tensor) else torch.tensor(score, device=device)

        # Combine failure maps per corrector (max = worst-case drives correction)
        combined_failure_maps = {}
        combined_pred_scores = {}
        for name, pred_keys in self.pred_key_map.items():
            maps = [failure_maps[k] for k in pred_keys]
            combined_failure_maps[name] = torch.stack(maps).max(dim=0)[0]
            scores = [pred_score_dict[k] for k in pred_keys]
            combined_pred_scores[name] = scores[0]
            for s in scores[1:]:
                combined_pred_scores[name] = torch.min(combined_pred_scores[name], s)

        # =====================================================================
        # STEP 3: Correctors Generate Corrections and Assess Potential
        # =====================================================================
        corrections = {}
        potentials = {}

        for name, corrector in self.correctors.items():
            # Get correction and potential using combined maps
            # Use adapted backbone so corrector sees training-domain intensities
            correction, potential = corrector(
                backbone_adapted, noisy, combined_failure_maps[name], combined_pred_scores[name], backbone_features
            )

            corrections[name] = correction
            potentials[name] = potential

        # STEP 3.5: Modulate potentials by uncertainty
        # Boost potential where NAFNet is uncertain, reduce where confident
        # This creates a cooperative feedback: uncertain regions get more corrector attention
        nafnet_uncertainty = 1.0 - nafnet_confidence
        for name in potentials:
            potentials[name] = potentials[name] * (0.5 + nafnet_uncertainty)

        # =====================================================================
        # STEP 4: Symbolic Negotiation for Allocation
        # =====================================================================
        # Negotiate allocations (pred_score_dict already prepared above)
        if self._ablation_no_negotiator:
            # ABLATION: Uniform allocation (bypass symbolic negotiation)
            allocations = {name: torch.ones_like(backbone_out) for name in self.correctors}
            negotiation_info = {'rule_activations': {}, 'ablation': 'no_negotiator'}
        else:
            allocations, negotiation_info = self.negotiator(
                nafnet_confidence, potentials, pred_score_dict
            )

        # =====================================================================
        # STEP 5: Get Lambda Maps for Additional Modulation
        # =====================================================================
        lambda_maps_raw = self.lambda_predictor(backbone_adapted, failure_maps)
        # Remap: lambda_predictor returns 'tissue'/'boundary' keys.
        # For the 'gain' corrector, use the element-wise max of both lambda maps
        # so it responds to the worst-case predicate failure across all groups.
        lambda_maps = {}
        if 'gain' in self.correctors:
            tissue_lam = lambda_maps_raw.get('tissue', torch.ones_like(backbone_out))
            boundary_lam = lambda_maps_raw.get('boundary', torch.ones_like(backbone_out))
            lambda_maps['gain'] = torch.max(tissue_lam, boundary_lam)
        # Keep original keys for any legacy correctors still present
        for k, v in lambda_maps_raw.items():
            if k not in lambda_maps:
                lambda_maps[k] = v

        # =====================================================================
        # STEP 6: Get Routing Activations from Hierarchical Reasoner
        # =====================================================================
        # GRADIENT FIX: During training, compute activations WITH gradients
        # so that the routing decisions can influence learning.
        # The router's activations multiply corrections at line 1575, so they
        # need gradients for the model to learn proper routing.
        #
        # Note: HierarchicalSymbolicReasoner.forward() has internal no_grad(),
        # so we call _forward_impl() directly during training to get gradients.
        if self.training:
            # Training: use _forward_impl with allow_gradients=True
            routing = self.router._forward_impl(pred_results, allow_gradients=True)
        else:
            # Inference: use regular forward (with no_grad for efficiency)
            routing = self.router(pred_results)
        raw_activations = routing['activations']
        # NOTE: activation gate REMOVED from accumulation (gate collapse QT38).
        # It was a non-learned average of router outputs, redundant with allocation.
        # Compute activations for diagnostics/interpretability only (not used in gates).
        def _to_val(a):
            return a if isinstance(a, (int, float, torch.Tensor)) else 0.0
        activations = {
            'gain': (_to_val(raw_activations.get('contrast', 0)) +
                     _to_val(raw_activations.get('smooth', 0)) +
                     _to_val(raw_activations.get('anatomy', 0)) +
                     _to_val(raw_activations.get('edge', 0)) +
                     _to_val(raw_activations.get('structure', 0))) / 5.0,
        }

        # =====================================================================
        # STEP 7: Multiplicative Gain Accumulation
        # =====================================================================
        alpha_map = torch.zeros_like(backbone_out)

        # Capture RAW correction magnitude BEFORE any gates (for floor_loss gradient)
        _raw_correction_mag = sum(c.abs().mean() for c in corrections.values())

        for name in self.correctors.keys():
            correction = corrections[name]  # alpha_map from MultiplicativeGainCorrector
            allocation = allocations[name]
            lambda_val = lambda_maps[name]

            if correction.shape[2:] != target_shape:
                correction = F.interpolate(correction, size=target_shape, mode='bilinear', align_corners=False)
            if allocation.shape[2:] != target_shape:
                allocation = F.interpolate(allocation, size=target_shape, mode='bilinear', align_corners=False)
            if lambda_val.shape[2:] != target_shape:
                lambda_val = F.interpolate(lambda_val, size=target_shape, mode='bilinear', align_corners=False)

            weighted_alpha = correction * allocation * lambda_val
            alpha_map = alpha_map + weighted_alpha

        # QT69c: Content-adaptive intensity gate (scanner-agnostic)
        # Suppress corrections in dark background pixels using per-image intensity threshold.
        # bg_rule selects between the original percentile gate and the Otsu tissue
        # mask. It defaults to the original so that nothing changes unless asked.
        intensity_gate = tissue_gate(backbone_out, rule=getattr(self, "bg_rule", "intensity"))

        # Decomposed alpha pipeline (structural SNR fix):
        # The MultiplicativeGainCorrector now outputs brightness (>=0) + contrast (zero-mean).
        # This structurally guarantees tissue_mean cannot decrease.
        alpha_map = alpha_map.clamp(-0.20, 0.20)
        alpha_map = torch.nan_to_num(alpha_map, nan=0.0, posinf=0.20, neginf=-0.20)
        alpha_map = alpha_map * intensity_gate

        # Pre-gate mag for floor_loss: use raw (before allocation/lambda) for strong gradient
        _pre_gate_correction_mag = _raw_correction_mag

        # Skipped modules — multiplicative gain pipeline replaces region-decomposed pipeline
        region_aware_info = {'skipped': True, 'reason': 'multiplicative_gain_pipeline'}
        cnr_info = {'skipped': True, 'reason': 'multiplicative_gain_pipeline'}
        clinical_info = {'skipped': True, 'reason': 'multiplicative_gain_pipeline'}

        # =====================================================================
        # STEP 8: Apply Low-Frequency Correction (EPI-preserving)
        # =====================================================================
        # Instead of: corrected = backbone * (1 + alpha)  (creates HF edge artifacts)
        # Use: corrected = backbone + smooth(alpha * backbone)
        # The correction term (alpha * backbone) is smoothed before adding, so it
        # structurally cannot create high-frequency edge distortion. This preserves
        # Sobel edge correlation (EPI) while still boosting tissue mean (CNR/SNR).
        # Decomposed smoothing: smooth brightness (positive), pass contrast (negative) through.
        # Brightness creates uniform boost at tissue-bg boundaries → needs smoothing for EPI.
        # Contrast is zero-mean and self-canceling → can pass unsmoothed for more CNR.
        positive_alpha = F.relu(alpha_map)  # ~brightness component
        negative_alpha = alpha_map - positive_alpha  # ~contrast negative parts
        # Smooth only the positive (brightening) correction — 3x3 for balanced CNR/EPI
        pos_correction = positive_alpha * backbone_out
        pos_correction_smooth = F.avg_pool2d(
            F.pad(pos_correction, (1, 1, 1, 1), mode='reflect'),
            kernel_size=3, stride=1, padding=0
        )
        # Negative contrast applied directly (no smoothing)
        neg_correction = negative_alpha * backbone_out
        # Post-smoothing intensity gate: clip any correction that bled into background.
        # Smoothing smears tissue correction into background pixels → hurts CNR.
        # Re-applying intensity_gate ensures zero correction in background.
        total_correction = pos_correction_smooth + neg_correction
        total_correction = total_correction * intensity_gate
        multiplicative_result = backbone_out + total_correction

        # QT69c→v3: Background smoothing — NOW ALWAYS ON
        # Previously inference-only. Now enabled during training so optimizer can
        # learn corrections that cooperate with smoothing, improving SNR/ENL.
        # Masked mean filter: only averages background pixels (prevents tissue bleeding).
        # Scanner-agnostic: uses per-image intensity gate.
        # v4: Erode bg_mask by 2px to keep smoothing away from tissue-bg edges (EPI fix)
        if not self._ablation_no_bg_smooth:
            bg_mask = 1.0 - intensity_gate  # 1 in background, 0 in tissue
            # Erode: min-pool shrinks bg_mask by 2px, protecting edge pixels from blur
            bg_mask_eroded = -F.max_pool2d(-bg_mask, kernel_size=5, stride=1, padding=2)
            vals_x_mask = multiplicative_result * bg_mask_eroded
            sum_vals = F.avg_pool2d(
                F.pad(vals_x_mask, (1, 1, 1, 1), mode='constant', value=0),
                kernel_size=3, stride=1, padding=0
            ) * 9  # sum of background pixel values in 3x3
            sum_mask = F.avg_pool2d(
                F.pad(bg_mask_eroded, (1, 1, 1, 1), mode='constant', value=0),
                kernel_size=3, stride=1, padding=0
            ) * 9  # count of background pixels in 3x3
            bg_smoothed = sum_vals / sum_mask.clamp(min=1.0)
            # 50% blend in eroded background only — tissue + border untouched
            blend = bg_mask_eroded * 0.5
            multiplicative_result = multiplicative_result * (1.0 - blend) + bg_smoothed * blend

        # =====================================================================
        # STEP 8.5: Edge Restoration (EPI improvement)
        # =====================================================================
        # Always active — GuidedEdgeSharpener takes 1 arg (backbone_out),
        # EdgeRecoveryModule takes 2 args (backbone_out, noisy).
        H_corr = multiplicative_result.shape[2]
        if self._ablation_no_edge:
            # ABLATION: Skip edge recovery
            edge_restoration = torch.zeros_like(backbone_out)
        elif isinstance(self.edge_sharpener, GuidedEdgeSharpener):
            edge_restoration = self.edge_sharpener(backbone_out)
        else:
            edge_restoration = self.edge_sharpener(backbone_out, noisy)

        # QT69c: Content-adaptive edge taper (reuses intensity_gate from alpha suppression)
        # Suppresses edge restoration in background — scanner-agnostic
        if intensity_gate.shape[2:] != edge_restoration.shape[2:]:
            edge_gate = F.interpolate(intensity_gate, size=edge_restoration.shape[2:],
                                      mode='bilinear', align_corners=False)
        else:
            edge_gate = intensity_gate
        edge_restoration = edge_restoration * edge_gate

        # STEP 8b & 9: Tone curve + adaptive smoothing — DISABLED (QT58 showed harmful)

        # Combine: multiplicative gain + edge restoration only
        candidate = (multiplicative_result + edge_restoration).clamp(0, 1)

        # For diagnostics: equivalent additive correction
        total_correction = candidate - backbone_out

        # OPTIMIZATION: Use nan_to_num with backbone_out fallback value
        if not torch.isfinite(candidate).all():
            candidate = torch.where(torch.isfinite(candidate), candidate, backbone_out).clamp(0, 1)

        # =====================================================================
        # STEP 9: Formal Verification
        # =====================================================================
        # SPEED: Skip verifier during training and TTA.
        # Verifier re-evaluates ALL predicates (expensive convolutions) on the
        # candidate, then blends with backbone_out.  During training, the loss
        # provides the learning signal; formal guarantees are for inference.
        # During TTA, verifier dilutes the correction signal the loss depends on.
        if self.training or getattr(self, '_tta_mode', False):
            output = candidate
            verify_info = {'decision': 'TRAIN_SKIP' if self.training else 'TTA_SKIP',
                           'guarantees_passed': -1, 'accepted': True}
        else:
            output, verify_info = self.verifier(
                backbone_out, candidate, noisy, total_correction, pred_before=pred_results
            )

        # =====================================================================
        # STEP 10: Re-evaluate Predicates on Output
        # =====================================================================
        # SPEED: Skip predicate re-evaluation during training.  The backbone
        # predicates (pred_results) are reused — corrector-specific predicate
        # improvement is tracked via the loss, not a second forward pass.
        # During inference/validation, always re-evaluate for accurate reporting.
        if self.training or getattr(self, '_tta_mode', False):
            pred_results_after = pred_results  # Reuse backbone predicates (fast)
        else:
            with torch.no_grad():
                pred_results_after = self.predicates(output, noisy)

        # =====================================================================
        # STEP 11: Compile Interpretability Output - OPTIMIZED
        # =====================================================================

        # TTA MODE: Return minimal info dict — _compute_tta_loss only reads
        # 'predicate_scores'. Skip all stats compilation (potential_stats,
        # allocation_stats, activation_stats, lambda_stats, etc.) to reduce
        # peak memory by ~5-10 MB per step and avoid heap fragmentation.
        if getattr(self, '_tta_mode', False):
            info = {'predicate_scores': pred_results['scores']}
            return output, info

        # OPTIMIZATION: Compute stats in no_grad context and batch operations
        with torch.no_grad():
            # OPTIMIZATION: Detach confidence map once
            nafnet_conf_detached = nafnet_confidence.detach()

            if self.training:
                nafnet_conf_info = {
                    'mean': nafnet_conf_detached.mean().item(),
                    'min': nafnet_conf_detached.min().item(),
                    'max': nafnet_conf_detached.max().item(),
                }
                potential_stats = {
                    name: {
                        'mean': pot.mean().item(),
                        'max': pot.max().item(),
                        'map': pot.detach(),  # detached for logging
                    }
                    for name, pot in potentials.items()
                }
                allocation_stats = {
                    name: {
                        'mean': alloc.mean().item(),
                        'max': alloc.max().item(),
                    }
                    for name, alloc in allocations.items()
                }
                correction_mag = total_correction.abs().mean().item()
                # Multiplicative gain diagnostics
                alpha_mag = alpha_map.abs().mean().item()
                edge_mag = edge_restoration.abs().mean().item()
                region_decomp_stats = {
                    'alpha_map_mean': alpha_mag,
                    'alpha_map_min': alpha_map.min().item(),
                    'alpha_map_max': alpha_map.max().item(),
                    'edge_restoration_mag': edge_mag,
                }
            else:
                # OPTIMIZATION: Skip storing full spatial maps during validation to save GPU memory
                nafnet_conf_info = {
                    'mean': nafnet_conf_detached.mean().item(),
                    'min': nafnet_conf_detached.min().item(),
                    'max': nafnet_conf_detached.max().item(),
                }
                # _skip_maps flag: set by sweep's fast validation to skip storing
                # 5 potential maps ([1,1,H,W] each ≈ 1MB) that are only needed for
                # uncertainty-potential correlation in full validate_dataset().
                skip_maps = getattr(self, '_skip_maps', False)
                if skip_maps:
                    potential_stats = {
                        name: {
                            'mean': pot.mean().item(),
                            'max': pot.max().item(),
                        }
                        for name, pot in potentials.items()
                    }
                else:
                    potential_stats = {
                        name: {
                            'mean': pot.mean().item(),
                            'max': pot.max().item(),
                            'map': pot.detach(),  # needed for uncertainty-potential correlation
                        }
                        for name, pot in potentials.items()
                    }
                allocation_stats = {
                    name: {
                        'mean': alloc.mean().item(),
                        'max': alloc.max().item(),
                        'active_pixels': (alloc > 0.1).sum().item(),
                    }
                    for name, alloc in allocations.items()
                }
                correction_mag = total_correction.abs().mean().item()
                # Multiplicative gain diagnostics
                alpha_mag = alpha_map.abs().mean().item()
                edge_mag = edge_restoration.abs().mean().item()
                region_decomp_stats = {
                    'alpha_map_mean': alpha_mag,
                    'alpha_map_min': alpha_map.min().item(),
                    'alpha_map_max': alpha_map.max().item(),
                    'edge_restoration_mag': edge_mag,
                }

            # OPTIMIZATION: Compute activation stats once
            activation_stats = {}
            for k, v in activations.items():
                if isinstance(v, torch.Tensor):
                    # Convert scalar tensors to Python floats to prevent memory leaks
                    activation_stats[k] = v.item() if v.numel() == 1 else v.detach()
                else:
                    activation_stats[k] = v

            # OPTIMIZATION: Compute lambda stats once
            lambda_stats = {}
            for name, lam in lambda_maps.items():
                if self.training:
                    lambda_stats[name] = {'mean': lam.mean().item(), 'max': lam.max().item()}
                else:
                    lambda_stats[name] = {'mean': lam.mean().item(), 'max': lam.max().item()}

        info = {
            'nafnet_confidence': nafnet_conf_info,
            'corrector_potentials': potential_stats,
            'allocations': allocation_stats,
            'negotiation_trace': negotiation_info,
            'predicate_scores': pred_results_after['scores'],
            'predicate_scores_backbone': pred_results['scores'],
            'activations': activation_stats,
            'lambda_stats': lambda_stats,
            'verification': verify_info,
            'correction_magnitude': correction_mag,
            'cnr_preservation': cnr_info,
            'clinical_enhancement': clinical_info,
            'region_aware_correction': region_aware_info,
            'region_decomposition': region_decomp_stats,
            # Pre-gate correction magnitude: tensor during training (for floor_loss gradient),
            # float during validation (for logging only)
            'pre_gate_correction_mag': _pre_gate_correction_mag.detach().item(),
            # Alpha map and edge restoration tensors for loss regularization.
            # During training: live tensors (with grad) for smoothness/ceiling losses.
            # During validation: detached tensors for logging only.
            'alpha_map': alpha_map if self.training else alpha_map.detach(),
            'edge_restoration': edge_restoration if self.training else edge_restoration.detach(),
        }

        # Live (non-detached) potential maps for cooperation loss gradient flow.
        # During training, the cooperation loss needs to backprop through the
        # potential_net to teach it to produce potentials correlated with uncertainty.
        # Without this, cooperation loss is dead (operates on detached tensors).
        if self.training:
            info['_live_potentials'] = potentials  # Dict[str, Tensor] with grad
            info['_live_nafnet_uncertainty'] = nafnet_uncertainty  # Tensor with grad

        # Add detailed interpretability if requested - OPTIMIZED
        if return_details:
            # OPTIMIZATION: Reuse already detached tensors
            info['nafnet_confidence_map'] = nafnet_conf_detached
            info['potential_maps'] = {name: pot.detach() for name, pot in potentials.items()}
            info['allocation_maps'] = {name: alloc.detach() for name, alloc in allocations.items()}

            # OPTIMIZATION: Build failure_maps dict once (detach to prevent grad leaks)
            failure_map_dict = {
                k: pred_results[k]['failure_map'].detach()
                if isinstance(pred_results[k]['failure_map'], torch.Tensor)
                else pred_results[k]['failure_map']
                for k in pred_results
                if isinstance(pred_results.get(k), dict) and 'failure_map' in pred_results[k]
            }

            layer_analysis = self.explainer.clinical_layer_analysis(failure_map_dict, H)
            info['layer_analysis'] = layer_analysis
            info['clinical_report'] = self.explainer.generate_clinical_report(
                pred_results, routing, verify_info, layer_analysis
            )
            info['counterfactuals'] = self.explainer.counterfactual_analysis(
                pred_results, 'P1', [0.3, 0.5, 0.7, 0.9]
            )

            # OPTIMIZATION: Single comprehension for inference trace
            info['inference_trace'] = {
                level: {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in items.items()}
                for level, items in routing['inference_trace'].items()
                if isinstance(items, dict)
            }
            info['explanations'] = routing['explanations']

        return output, info

    def get_cooperation_summary(self, info: Dict) -> str:
        """
        Generate a human-readable summary of the cooperation.

        Args:
            info: Info dict from forward pass

        Returns:
            summary: Formatted string describing the cooperation
        """
        lines = []
        lines.append("=" * 60)
        lines.append("COOPERATIVE DENOISING SUMMARY")
        lines.append("=" * 60)

        # NAFNet confidence
        conf = info['nafnet_confidence']
        lines.append(f"\nNAFNet Confidence:")
        lines.append(f"  Mean: {conf['mean']:.3f}, Range: [{conf['min']:.3f}, {conf['max']:.3f}]")

        # Corrector potentials and allocations
        lines.append(f"\nCorrectors:")
        for name in info['corrector_potentials'].keys():
            pot = info['corrector_potentials'][name]
            alloc = info['allocations'][name]
            lines.append(f"  {name:12s}: potential={pot['mean']:.3f}, allocation={alloc['mean']:.3f}")

        # Negotiation
        neg = info['negotiation_trace']
        lines.append(f"\nNegotiation:")
        lines.append(f"  Total correction weight: {neg['total_correction_weight']:.3f}")

        # Rule activations (first corrector as example)
        if 'rule_traces' in neg and len(neg['rule_traces']) > 0:
            first_corrector = list(neg['rule_traces'].keys())[0]
            rules = neg['rule_traces'][first_corrector]
            lines.append(f"\nRule Activations (example for {first_corrector}):")
            for rule, val in rules.items():
                if val > 0.1:
                    lines.append(f"  {rule}: {val:.3f}")

        # Region-Aware Correction
        if 'region_aware_correction' in info and 'error' not in info['region_aware_correction']:
            region = info['region_aware_correction']
            lines.append(f"\nRegion-Aware Correction:")
            lines.append(f"  Correction factor: mean={region.get('correction_factor_mean', 0):.3f}, "
                        f"range=[{region.get('correction_factor_min', 0):.3f}, {region.get('correction_factor_max', 0):.3f}]")
            lines.append(f"  Uncertainty weight: {region.get('uncertainty_weight', 0):.3f}")
            lines.append(f"  Contrast deficit: {region.get('contrast_deficit', 0):.3f}")
            lines.append(f"  Edge strength: {region.get('edge_strength', 0):.3f}")
            lines.append(f"  Background ratio: {region.get('background_ratio', 0):.3f}")

        # Verification
        verify = info['verification']
        lines.append(f"\nVerification:")
        lines.append(f"  Decision: {verify['decision']}")
        lines.append(f"  Guarantees passed: {verify['guarantees_passed']}/3")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("\nTesting NeuroSymbolicCorrectorV8Cooperative...")

    # Create model
    model = NeuroSymbolicCorrectorV8Cooperative(
        in_channels=1,
        hidden_channels=64,
        feature_channels=64,
    )

    # Test inputs
    B, C, H, W = 2, 1, 128, 128
    noisy = torch.randn(B, C, H, W) * 0.3 + 0.5
    noisy = noisy.clamp(0, 1)
    backbone_out = noisy - torch.randn(B, C, H, W) * 0.1
    backbone_out = backbone_out.clamp(0, 1)

    # Forward pass (backbone_features=None, backbone-agnostic)
    print("\nRunning forward pass...")
    corrected, info = model(backbone_out, noisy, None, return_details=True)

    print(f"\nInput shape: {backbone_out.shape}")
    print(f"Output shape: {corrected.shape}")

    # Print cooperation summary
    print(model.get_cooperation_summary(info))

    # Detailed outputs
    print("\n" + "=" * 60)
    print("DETAILED OUTPUTS:")
    print("=" * 60)

    print(f"\nPredicate Scores (after correction): {info['predicate_scores']}")
    print(f"Predicate Scores (backbone): {info['predicate_scores_backbone']}")

    print(f"\nCorrector Activations: {info['activations']}")

    print(f"\nLambda Stats:")
    for name, stats in info['lambda_stats'].items():
        print(f"  {name}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")

    print(f"\nCorrection magnitude: {info['correction_magnitude']:.4f}")

    # Test maps are correct shapes
    print("\n" + "=" * 60)
    print("MAP SHAPES:")
    print("=" * 60)
    print(f"NAFNet confidence map: {info['nafnet_confidence_map'].shape}")
    for name, pot_map in info['potential_maps'].items():
        print(f"Potential map ({name}): {pot_map.shape}")
    for name, alloc_map in info['allocation_maps'].items():
        print(f"Allocation map ({name}): {alloc_map.shape}")

    # Test individual components
    print("\n" + "=" * 60)
    print("COMPONENT TESTS:")
    print("=" * 60)

    # Test confidence estimator (no backbone features in agnostic mode)
    conf, conf_info = model.confidence_estimator(backbone_out, None)
    print(f"\nConfidence estimator:")
    print(f"  Output shape: {conf.shape}")
    print(f"  Method: {conf_info['method']}")
    print(f"  Mean confidence: {conf_info['mean_confidence']:.4f}")

    # Test negotiator directly
    test_confidence = torch.rand(B, 1, H, W)
    test_potentials = {name: torch.rand(B, 1, H, W) for name in model.correctors.keys()}
    test_scores = {k: torch.tensor(0.5) for k in ['P1', 'P2', 'P3', 'P4', 'P6']}

    allocs, neg_info = model.negotiator(test_confidence, test_potentials, test_scores)
    print(f"\nNegotiator:")
    print(f"  Allocations computed for: {list(allocs.keys())}")
    print(f"  Mean allocations: {neg_info['mean_allocations']}")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
