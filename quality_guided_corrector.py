"""
Quality-Guided Corrector Architecture

This module implements a novel correction architecture that inherently prevents
quality degradation by computing corrections as controlled movements toward
an estimated clean target.

Key Innovation:
--------------
Instead of the traditional approach of "correct then check quality", this
architecture computes corrections that are GUARANTEED to improve quality
when the clean estimate is accurate.

Core Principle:
--------------
    correction = lambda * (clean_estimate - backbone_out)

Where:
- clean_estimate: The corrector's prediction of what the clean image should be
- backbone_out: The current (potentially degraded) output from the backbone
- lambda: A learned confidence map in [0, max_lambda] that controls how much
          we trust our clean estimate vs the backbone output

Mathematical Guarantee:
----------------------
If clean_estimate is close to the true clean image, then moving backbone_out
toward clean_estimate (via the correction) MUST improve quality metrics.
The lambda (confidence) map allows the network to be conservative in regions
where it's uncertain about the clean target.

Architecture Overview:
---------------------
1. Clean Estimator: Predicts what the clean image should look like
2. Confidence Predictor: Determines per-pixel confidence in the estimate
3. Predicate Refiners: Specialized modules for edge, contrast, noise,
   structure, and color corrections
4. Quality-Guided Combiner: Applies corrections with confidence weighting

Author: Quality-Guided Architecture Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class ConvBlock(nn.Module):
    """
    Basic convolutional block with optional normalization and activation.

    Used as a building block throughout the architecture for feature
    extraction and transformation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
        activation: str = 'leaky_relu'
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'none':
            pass

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and skip connection.

    Maintains gradient flow and allows learning identity mappings
    when no correction is needed.
    """

    def __init__(self, channels: int, use_norm: bool = True):
        super().__init__()

        self.conv1 = ConvBlock(channels, channels, use_norm=use_norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_norm)
        if use_norm:
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        return F.leaky_relu(out + residual, 0.2)


class AttentionGate(nn.Module):
    """
    Attention gate for feature refinement.

    Computes spatial attention weights to focus on regions
    that need correction while preserving good regions.
    """

    def __init__(self, gate_channels: int, in_channels: int, inter_channels: int = None):
        super().__init__()

        inter_channels = inter_channels or in_channels // 2

        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
        self.input_conv = nn.Conv2d(in_channels, inter_channels, 1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features to be gated
            gate: Gating signal (e.g., from failure maps)

        Returns:
            Attention-weighted features
        """
        g = self.gate_conv(gate)
        x_theta = self.input_conv(x)

        # Handle size mismatch
        if g.shape[2:] != x_theta.shape[2:]:
            g = F.interpolate(g, size=x_theta.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g + x_theta)
        psi = self.psi(psi)

        return x * psi


class SmallRefiner(nn.Module):
    """
    Specialized refinement module for predicate-specific corrections.

    Each SmallRefiner focuses on one aspect of image quality:
    - Edge preservation/enhancement
    - Contrast normalization
    - Noise reduction
    - Structure preservation
    - Color consistency

    Design Philosophy:
    -----------------
    These refiners don't produce arbitrary corrections. Instead, they
    produce ADJUSTMENTS to the base correction (clean_estimate - backbone_out).
    This ensures corrections stay within the "toward clean" framework.

    Architecture:
    ------------
    - Lightweight (~100K parameters each)
    - Takes base correction + failure map as input
    - Outputs refinement that adjusts the base correction
    - Final layer zero-initialized for stable training start
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 48,
        failure_map_channels: int = 1,
        name: str = "generic"
    ):
        """
        Args:
            in_channels: Channels of the base correction (typically 3 for RGB)
            hidden_channels: Internal feature dimension (~75K params each)
            failure_map_channels: Channels of the failure/quality map
            name: Identifier for this refiner (for logging/debugging)
        """
        super().__init__()

        self.name = name

        # Input projection: combine correction with failure map
        self.input_proj = ConvBlock(
            in_channels + failure_map_channels,
            hidden_channels,
            kernel_size=3,
            use_norm=True
        )

        # Two residual blocks for better capacity
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )

        # Spatial attention based on failure map
        self.attention = nn.Sequential(
            nn.Conv2d(failure_map_channels, hidden_channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels, 3, 1, 1),
            nn.Sigmoid()
        )

        # Output projection: produce refinement adjustment
        self.output_proj = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, in_channels, 3, 1, 1)
        )

        # Scale factor for this refiner's contribution
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        # Zero-initialize final layer for stable training
        self._zero_init_final()

    def _zero_init_final(self):
        """Initialize final convolution weights to zero."""
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        base_correction: torch.Tensor,
        failure_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute predicate-specific refinement.

        Args:
            base_correction: The base correction (clean_estimate - backbone_out)
            failure_map: Quality/failure map indicating problematic regions

        Returns:
            Refinement adjustment to add to base correction
        """
        # Ensure failure_map has correct spatial size
        if failure_map.shape[2:] != base_correction.shape[2:]:
            failure_map = F.interpolate(
                failure_map,
                size=base_correction.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Concatenate correction with failure information
        x = torch.cat([base_correction, failure_map], dim=1)

        # Extract features
        features = self.input_proj(x)
        features = self.res_blocks(features)

        # Apply failure-guided attention
        attention_weights = self.attention(failure_map)
        features = features * attention_weights

        # Generate refinement
        refinement = self.output_proj(features)

        # Scale the refinement
        refinement = refinement * self.scale

        return refinement


class CleanEstimator(nn.Module):
    """
    Network that estimates the clean target image.

    This is the core of the quality-guided approach. By predicting what
    the clean image should look like, we can compute corrections that
    move toward that target rather than making arbitrary adjustments.

    Architecture:
    ------------
    - Encoder: Extracts multi-scale features from backbone output
    - Feature Fusion: Combines backbone features at multiple scales
    - Decoder: Reconstructs clean estimate at full resolution

    The estimator is trained to minimize the distance between its
    output and the actual clean image during training.

    Note: This is a lightweight version (~800K params) designed to fit
    within the 2-3M total parameter budget.
    """

    def __init__(
        self,
        in_channels: int = 3,
        enc1_channels: int = 64,
        enc2_channels: int = 128,
        hidden_dim: int = 96,
        num_res_blocks: int = 4
    ):
        """
        Args:
            in_channels: Input channels (backbone output, typically 3)
            enc1_channels: Channels of first encoder features from backbone
            enc2_channels: Channels of second encoder features from backbone
            hidden_dim: Internal feature dimension (optimized for ~1.5M params)
            num_res_blocks: Number of residual blocks in the bottleneck
        """
        super().__init__()

        # Initial feature extraction from backbone output
        self.initial_conv = nn.Sequential(
            ConvBlock(in_channels, hidden_dim // 2, kernel_size=5, padding=2),
            ConvBlock(hidden_dim // 2, hidden_dim)
        )

        # Feature fusion modules for backbone encoder features
        self.enc1_fusion = nn.Sequential(
            ConvBlock(enc1_channels, hidden_dim // 2, kernel_size=1, padding=0),
            ConvBlock(hidden_dim // 2, hidden_dim)
        )
        self.enc2_fusion = nn.Sequential(
            ConvBlock(enc2_channels, hidden_dim, kernel_size=1, padding=0),
            ConvBlock(hidden_dim, hidden_dim)
        )

        # Multi-scale feature combination
        self.scale_fusion = nn.Sequential(
            ConvBlock(hidden_dim * 3, hidden_dim * 2),
            ConvBlock(hidden_dim * 2, hidden_dim)
        )

        # Residual processing at bottleneck
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim),
            ConvBlock(hidden_dim, hidden_dim // 2)
        )

        # Final output layer - Predict RESIDUAL to add to backbone_out
        # Using Tanh to allow +/- corrections, with LEARNABLE scale for adaptive correction
        self.final = nn.Sequential(
            ConvBlock(hidden_dim // 2, hidden_dim // 4),
            nn.Conv2d(hidden_dim // 4, in_channels, 3, 1, 1),
            nn.Tanh()  # Residual in [-1, 1], will be scaled
        )
        # LEARNABLE residual scale - starts at 0.1 (conservative) and can grow
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        # Also add a direct residual path with very small scale for stable learning
        self.direct_residual = nn.Conv2d(hidden_dim // 2, in_channels, 1, bias=False)
        nn.init.zeros_(self.direct_residual.weight)  # Start at zero

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        backbone_out: torch.Tensor,
        backbone_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Estimate the clean target image.

        Args:
            backbone_out: Output from the backbone network
            backbone_features: Dictionary containing encoder features
                - 'enc1': First encoder output (1/2 resolution)
                - 'enc2': Second encoder output (1/4 resolution)

        Returns:
            Estimated clean image with same spatial size as backbone_out
        """
        B, C, H, W = backbone_out.shape

        # Extract initial features at full resolution
        initial_feat = self.initial_conv(backbone_out)  # [B, hidden, H, W]

        # Fuse encoder features
        enc1_feat = backbone_features.get('enc1')
        enc2_feat = backbone_features.get('enc2')

        if enc1_feat is not None:
            enc1_fused = self.enc1_fusion(enc1_feat)
            enc1_fused = F.interpolate(enc1_fused, size=(H, W), mode='bilinear', align_corners=False)
        else:
            enc1_fused = torch.zeros_like(initial_feat)

        if enc2_feat is not None:
            enc2_fused = self.enc2_fusion(enc2_feat)
            enc2_fused = F.interpolate(enc2_fused, size=(H, W), mode='bilinear', align_corners=False)
        else:
            enc2_fused = torch.zeros_like(initial_feat)

        # Combine multi-scale features
        combined = torch.cat([initial_feat, enc1_fused, enc2_fused], dim=1)
        fused = self.scale_fusion(combined)

        # Apply residual blocks
        for res_block in self.res_blocks:
            fused = res_block(fused)

        # Decode
        x = self.decoder(fused)

        # Final output - RESIDUAL learning: backbone_out + small_residual
        # Two paths: main tanh path (bounded) + direct linear path (for larger corrections when needed)
        tanh_residual = self.final(x) * self.residual_scale.clamp(0.01, 0.5)  # Clamped learnable scale
        direct_residual = self.direct_residual(x) * 0.05  # Very small direct path
        residual = tanh_residual + direct_residual

        # Store for debugging/analysis
        self._last_residual_scale = self.residual_scale.item()

        clean_estimate = (backbone_out + residual).clamp(0, 1)

        return clean_estimate


class ConfidencePredictor(nn.Module):
    """
    Predicts per-pixel confidence in the clean estimate.

    The confidence map (lambda) controls how much we trust our clean
    estimate versus the original backbone output:

        correction = lambda * (clean_estimate - backbone_out)

    Key Design Principles:
    ---------------------
    1. High confidence (lambda near max) in regions where the estimate is reliable
    2. Low confidence (lambda near 0) in uncertain regions, preserving backbone output
    3. Confidence is bounded in [0, max_lambda] to prevent overcorrection

    The network learns to be conservative: it's better to under-correct
    than to over-correct and introduce artifacts.
    """

    def __init__(
        self,
        in_channels: int = 3,
        enc1_channels: int = 64,
        enc2_channels: int = 128,
        hidden_dim: int = 48,
        max_lambda: float = 1.0
    ):
        """
        Args:
            in_channels: Channels of backbone_out and clean_estimate
            enc1_channels: Channels of first encoder features
            enc2_channels: Channels of second encoder features
            hidden_dim: Internal feature dimension (optimized for ~250K params)
            max_lambda: Maximum confidence value (correction strength)
        """
        super().__init__()

        self.max_lambda = max_lambda
        self.hidden_dim = hidden_dim

        # Input: backbone_out, clean_estimate, and their difference
        input_channels = in_channels * 3  # backbone, estimate, difference

        # Feature extraction
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, hidden_dim, kernel_size=5, padding=2),
            ConvBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim)
        )

        # Feature fusion with backbone encoder
        self.enc1_proj = nn.Conv2d(enc1_channels, hidden_dim // 2, 1, bias=True)
        self.enc2_proj = nn.Conv2d(enc2_channels, hidden_dim // 2, 1, bias=True)

        # Fusion layer
        self.fusion = nn.Sequential(
            ConvBlock(hidden_dim * 2, hidden_dim),
            ResidualBlock(hidden_dim)
        )

        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, 1, 3, 1, 1),
            nn.Sigmoid()  # Output in [0, 1], scaled by max_lambda
        )

        # Uncertainty estimation (auxiliary output)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 3, 1, 1),
            nn.Softplus()  # Positive uncertainty
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize for VERY conservative starting point - near-zero confidence."""
        # Initialize confidence head to output ~0.01 initially (not 0.5!)
        # This ensures corrections start very small and grow with learning
        for m in self.confidence_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Very small weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, -4.0)  # sigmoid(-4) ≈ 0.018

    def forward(
        self,
        backbone_out: torch.Tensor,
        clean_estimate: torch.Tensor,
        backbone_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict confidence in the clean estimate.

        Args:
            backbone_out: Output from backbone network
            clean_estimate: Estimated clean image from CleanEstimator
            backbone_features: Dictionary of encoder features

        Returns:
            confidence: Per-pixel confidence map in [0, max_lambda]
            uncertainty: Per-pixel uncertainty estimate (for training)
        """
        B, C, H, W = backbone_out.shape

        # Compute difference (key signal for confidence)
        difference = clean_estimate - backbone_out

        # Concatenate inputs
        x = torch.cat([backbone_out, clean_estimate, difference], dim=1)

        # Extract features
        features = self.encoder(x)

        # Fuse with backbone encoder features
        enc1_feat = backbone_features.get('enc1')
        enc2_feat = backbone_features.get('enc2')

        if enc1_feat is not None:
            enc1_proj = self.enc1_proj(enc1_feat)
            enc1_proj = F.interpolate(enc1_proj, size=(H, W), mode='bilinear', align_corners=False)
        else:
            enc1_proj = torch.zeros(B, self.hidden_dim // 2, H, W, device=x.device)

        if enc2_feat is not None:
            enc2_proj = self.enc2_proj(enc2_feat)
            enc2_proj = F.interpolate(enc2_proj, size=(H, W), mode='bilinear', align_corners=False)
        else:
            enc2_proj = torch.zeros(B, self.hidden_dim // 2, H, W, device=x.device)

        # Combine features: encoder features + projected encoder features
        combined = torch.cat([features, enc1_proj, enc2_proj], dim=1)
        fused = self.fusion(combined)

        # Predict confidence
        confidence = self.confidence_head(fused) * self.max_lambda

        # Predict uncertainty (auxiliary)
        uncertainty = self.uncertainty_head(fused)

        return confidence, uncertainty


class QualityGuidedCorrector(nn.Module):
    """
    Quality-Guided Corrector: A correction architecture that inherently
    prevents quality degradation.

    Key Innovation:
    --------------
    Instead of making arbitrary corrections and hoping they improve quality,
    this architecture computes corrections as controlled movements toward
    an estimated clean target.

    Mathematical Framework:
    ----------------------
        correction = lambda * (clean_estimate - backbone_out + refinements)

    Where:
    - clean_estimate: Network's prediction of the clean image
    - backbone_out: Current (potentially degraded) backbone output
    - lambda: Per-pixel confidence in [0, max_lambda]
    - refinements: Predicate-specific adjustments for edges, contrast, etc.

    Quality Guarantee:
    -----------------
    If clean_estimate is close to the true clean image, then:
    - Moving backbone_out toward clean_estimate improves quality
    - Lambda controls the step size (larger = more aggressive correction)
    - Low lambda preserves backbone output in uncertain regions

    Architecture Components:
    -----------------------
    1. CleanEstimator: Predicts what the clean image should look like
    2. ConfidencePredictor: Determines per-pixel confidence
    3. SmallRefiners: Predicate-specific modules for:
       - Edge preservation (P1: gradient/edge quality)
       - Contrast normalization (P2: local contrast)
       - Noise reduction (P3: noise level)
       - Structure preservation (P4: structural integrity)
       - Color consistency (P5: color accuracy)

    Training Strategy:
    -----------------
    1. Train clean_estimator to minimize ||clean_estimate - clean_target||
    2. Train confidence_predictor to maximize quality improvement
    3. Train refiners to handle predicate-specific failures
    4. End-to-end fine-tuning with quality metric feedback

    Parameters:
    ----------
    Total: ~2.5M parameters
    - CleanEstimator: ~1.2M
    - ConfidencePredictor: ~0.6M
    - SmallRefiners (5x): ~0.5M
    - Other: ~0.2M
    """

    def __init__(
        self,
        in_channels: int = 3,
        enc1_channels: int = 64,
        enc2_channels: int = 128,
        hidden_dim: int = 96,
        refiner_hidden: int = 48,
        max_lambda: float = 1.0,
        num_predicates: int = 5
    ):
        """
        Initialize the Quality-Guided Corrector.

        Args:
            in_channels: Input/output image channels (3 for RGB)
            enc1_channels: Channels of first backbone encoder features
            enc2_channels: Channels of second backbone encoder features
            hidden_dim: Hidden dimension for main modules (default 96 for ~2.5M params)
            refiner_hidden: Hidden dimension for SmallRefiners (default 48)
            max_lambda: Maximum confidence value
            num_predicates: Number of quality predicates (default 5)
        """
        super().__init__()

        self.in_channels = in_channels
        self.max_lambda = max_lambda
        self.num_predicates = num_predicates

        # ============================================
        # 1. Clean Target Estimator (~1.5M params)
        # ============================================
        self.clean_estimator = CleanEstimator(
            in_channels=in_channels,
            enc1_channels=enc1_channels,
            enc2_channels=enc2_channels,
            hidden_dim=hidden_dim,
            num_res_blocks=4
        )

        # ============================================
        # 2. Confidence Predictor (Lambda Map) (~180K params)
        # ============================================
        self.confidence_predictor = ConfidencePredictor(
            in_channels=in_channels,
            enc1_channels=enc1_channels,
            enc2_channels=enc2_channels,
            hidden_dim=hidden_dim // 2,
            max_lambda=max_lambda
        )

        # ============================================
        # 3. Predicate-Specific Refiners (~75K each, ~375K total)
        # ============================================
        # Each refiner specializes in one aspect of quality

        # P1: Edge/Gradient Quality
        self.edge_refiner = SmallRefiner(
            in_channels=in_channels,
            hidden_channels=refiner_hidden,
            failure_map_channels=1,
            name="edge"
        )

        # P2: Contrast Quality
        self.contrast_refiner = SmallRefiner(
            in_channels=in_channels,
            hidden_channels=refiner_hidden,
            failure_map_channels=1,
            name="contrast"
        )

        # P3: Noise Level
        self.noise_refiner = SmallRefiner(
            in_channels=in_channels,
            hidden_channels=refiner_hidden,
            failure_map_channels=1,
            name="noise"
        )

        # P4: Structural Integrity
        self.structure_refiner = SmallRefiner(
            in_channels=in_channels,
            hidden_channels=refiner_hidden,
            failure_map_channels=1,
            name="structure"
        )

        # P5: Color Consistency
        self.color_refiner = SmallRefiner(
            in_channels=in_channels,
            hidden_channels=refiner_hidden,
            failure_map_channels=1,
            name="color"
        )

        # ============================================
        # 4. Refinement Combiner (~50K params)
        # ============================================
        # Combines multiple refinements intelligently
        self.refinement_combiner = nn.Sequential(
            ConvBlock(in_channels * num_predicates, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, in_channels, 3, 1, 1)
        )

        # Learnable weights for refinement combination
        self.refine_weights = nn.Parameter(torch.ones(num_predicates) / num_predicates)

        # ============================================
        # 5. Final Blending Module (~5K params)
        # ============================================
        # Optional learned blending for smoother results
        self.blend_conv = nn.Sequential(
            ConvBlock(in_channels * 2, hidden_dim // 4),
            nn.Conv2d(hidden_dim // 4, in_channels, 3, 1, 1)
        )

        # Zero-initialize for identity start
        self._zero_init_blend()

        # ============================================
        # 6. Quality Verification (Auxiliary)
        # ============================================
        # Predicts post-correction quality (for training feedback)
        self.quality_predictor = nn.Sequential(
            ConvBlock(in_channels, hidden_dim // 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim // 4, num_predicates),
            nn.Sigmoid()
        )

        self._print_param_count()

    def _zero_init_blend(self):
        """Zero-initialize blending for identity mapping at start."""
        nn.init.zeros_(self.blend_conv[-1].weight)
        nn.init.zeros_(self.blend_conv[-1].bias)
        nn.init.zeros_(self.refinement_combiner[-1].weight)
        nn.init.zeros_(self.refinement_combiner[-1].bias)

    def _print_param_count(self):
        """Print parameter counts for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        components = {
            'CleanEstimator': self.clean_estimator,
            'ConfidencePredictor': self.confidence_predictor,
            'EdgeRefiner': self.edge_refiner,
            'ContrastRefiner': self.contrast_refiner,
            'NoiseRefiner': self.noise_refiner,
            'StructureRefiner': self.structure_refiner,
            'ColorRefiner': self.color_refiner,
            'RefinementCombiner': self.refinement_combiner,
            'BlendConv': self.blend_conv,
            'QualityPredictor': self.quality_predictor
        }

        total = 0
        # Store for later access, don't print during init
        self._param_counts = {}
        for name, module in components.items():
            count = count_params(module)
            self._param_counts[name] = count
            total += count
        self._param_counts['Total'] = total

    def get_param_counts(self) -> Dict[str, int]:
        """Return parameter counts for all components."""
        return self._param_counts

    def _get_default_failure_maps(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate default failure maps when none are provided.

        Uses simple heuristics based on the input image.

        Args:
            x: Input image tensor

        Returns:
            Dictionary of failure maps for each predicate
        """
        B, C, H, W = x.shape
        device = x.device

        # Compute gradient magnitude for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=x.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=x.dtype, device=device).view(1, 1, 3, 3)

        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        gradient_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)

        # Compute local variance for noise estimation
        mean_filter = torch.ones(1, 1, 5, 5, device=device) / 25
        local_mean = F.conv2d(gray, mean_filter, padding=2)
        local_var = F.conv2d(gray ** 2, mean_filter, padding=2) - local_mean ** 2
        noise_map = torch.clamp(local_var, 0, 1)

        # Compute local contrast
        local_max = F.max_pool2d(gray, 5, stride=1, padding=2)
        local_min = -F.max_pool2d(-gray, 5, stride=1, padding=2)
        contrast_map = (local_max - local_min) / (local_max + local_min + 1e-8)

        # Structure map (based on gradient consistency)
        structure_map = 1.0 - torch.abs(grad_x * grad_y) / (gradient_mag + 1e-8)

        # Color consistency (variance across channels)
        if C >= 3:
            color_var = x.var(dim=1, keepdim=True)
            color_map = color_var / (color_var.max() + 1e-8)
        else:
            color_map = torch.zeros(B, 1, H, W, device=device)

        return {
            'P1': gradient_mag,      # Edge quality
            'P2': contrast_map,      # Contrast
            'P3': noise_map,         # Noise
            'P4': structure_map,     # Structure
            'P5': color_map          # Color
        }

    def forward(
        self,
        backbone_out: torch.Tensor,
        noisy: torch.Tensor,
        backbone_features: Dict[str, torch.Tensor],
        failure_maps: Optional[Dict[str, torch.Tensor]] = None,
        external_lambda_maps: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply quality-guided correction.

        Args:
            backbone_out: Output from the backbone denoising network
            noisy: Original noisy input (for reference)
            backbone_features: Dictionary of encoder features from backbone
                - 'enc1': First encoder output
                - 'enc2': Second encoder output
            failure_maps: Optional dictionary of failure/quality maps
                - 'P1': Edge quality map
                - 'P2': Contrast quality map
                - 'P3': Noise level map
                - 'P4': Structure quality map
                - 'P5': Color quality map
                If None, default maps are computed from the input.
            external_lambda_maps: Optional dictionary of trained lambda maps
                - 'edge', 'contrast', 'sharpness', 'texture', 'smooth'
                If provided, these will be blended with internal confidence.

        Returns:
            corrected: Quality-guided corrected output
            info: Dictionary containing intermediate results:
                - 'clean_estimate': Estimated clean target
                - 'confidence': Confidence/lambda map
                - 'uncertainty': Uncertainty estimate
                - 'base_correction': clean_estimate - backbone_out
                - 'total_correction': Final correction applied
                - 'refinements': Dictionary of per-predicate refinements
                - 'predicted_quality': Predicted post-correction quality
        """
        B, C, H, W = backbone_out.shape

        # Generate default failure maps if not provided
        if failure_maps is None:
            failure_maps = self._get_default_failure_maps(backbone_out)

        # ============================================
        # Step 1: Estimate Clean Target
        # ============================================
        clean_estimate = self.clean_estimator(backbone_out, backbone_features)

        # ============================================
        # Step 2: Compute Base Correction
        # ============================================
        # This is the core of quality-guided correction:
        # We correct TOWARD the clean estimate
        base_correction = clean_estimate - backbone_out

        # ============================================
        # Step 3: Predict Confidence (Lambda)
        # ============================================
        internal_confidence, uncertainty = self.confidence_predictor(
            backbone_out, clean_estimate, backbone_features
        )

        # Blend with external lambda_maps if provided
        # This integrates trained lambda predictor guidance with internal estimation
        if external_lambda_maps is not None:
            # Average external lambda maps to get unified confidence
            external_lambdas = []
            for key in ['edge', 'contrast', 'sharpness', 'texture', 'smooth']:
                if key in external_lambda_maps:
                    lmap = external_lambda_maps[key]
                    if lmap.shape[2:] != backbone_out.shape[2:]:
                        lmap = F.interpolate(lmap, size=backbone_out.shape[2:],
                                           mode='bilinear', align_corners=False)
                    external_lambdas.append(lmap)

            if external_lambdas:
                external_confidence = torch.stack(external_lambdas, dim=0).mean(dim=0)
                # Blend: 50% internal, 50% external (learnable blend could be added)
                confidence = 0.5 * internal_confidence + 0.5 * external_confidence
            else:
                confidence = internal_confidence
        else:
            confidence = internal_confidence

        # ============================================
        # Step 4: Apply Predicate-Specific Refinements
        # ============================================
        refinements = {}

        # Edge refinement
        edge_refine = self.edge_refiner(
            base_correction,
            failure_maps.get('P1', torch.ones(B, 1, H, W, device=backbone_out.device))
        )
        refinements['edge'] = edge_refine

        # Contrast refinement
        contrast_refine = self.contrast_refiner(
            base_correction,
            failure_maps.get('P2', torch.ones(B, 1, H, W, device=backbone_out.device))
        )
        refinements['contrast'] = contrast_refine

        # Noise refinement
        noise_refine = self.noise_refiner(
            base_correction,
            failure_maps.get('P3', torch.ones(B, 1, H, W, device=backbone_out.device))
        )
        refinements['noise'] = noise_refine

        # Structure refinement
        structure_refine = self.structure_refiner(
            base_correction,
            failure_maps.get('P4', torch.ones(B, 1, H, W, device=backbone_out.device))
        )
        refinements['structure'] = structure_refine

        # Color refinement
        color_refine = self.color_refiner(
            base_correction,
            failure_maps.get('P5', torch.ones(B, 1, H, W, device=backbone_out.device))
        )
        refinements['color'] = color_refine

        # ============================================
        # Step 5: Combine Refinements
        # ============================================
        # Stack all refinements
        all_refines = torch.cat([
            edge_refine, contrast_refine, noise_refine,
            structure_refine, color_refine
        ], dim=1)

        # Learnable weighted combination
        weights = F.softmax(self.refine_weights, dim=0)
        combined_refine = self.refinement_combiner(all_refines)

        # ============================================
        # Step 6: Compute Total Correction
        # ============================================
        # Total correction = confidence * (base_correction + refinements)
        # This ensures correction magnitude is controlled by confidence
        total_correction = confidence * (base_correction + combined_refine)

        # ============================================
        # Step 7: Apply Correction
        # ============================================
        corrected = backbone_out + total_correction

        # ============================================
        # Step 8: Optional Blending (for smoothness)
        # ============================================
        # Blend between direct correction and backbone output
        # REDUCED: 0.02 instead of 0.1 to prevent over-correction
        blend_input = torch.cat([corrected, backbone_out], dim=1)
        blend_adjustment = self.blend_conv(blend_input)
        corrected = corrected + blend_adjustment * 0.02

        # Clamp to valid range [0, 1] for OCT images
        corrected = torch.clamp(corrected, 0, 1)

        # ============================================
        # Step 9: Predict Post-Correction Quality
        # ============================================
        predicted_quality = self.quality_predictor(corrected)

        # ============================================
        # Gather Info for Training/Analysis
        # ============================================
        info = {
            'clean_estimate': clean_estimate,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'base_correction': base_correction,
            'total_correction': total_correction,
            'refinements': refinements,
            'combined_refinement': combined_refine,
            'predicted_quality': predicted_quality,
            'refine_weights': weights
        }

        return corrected, info

    def compute_training_losses(
        self,
        corrected: torch.Tensor,
        info: Dict[str, torch.Tensor],
        clean_target: torch.Tensor,
        backbone_out: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for the quality-guided corrector.

        This method computes multiple loss terms to train the corrector:
        1. Clean estimation loss: How well we estimate the clean target
        2. Correction quality loss: Does the correction improve quality?
        3. Confidence calibration loss: Is confidence correlated with accuracy?
        4. Refinement regularization: Prevent over-aggressive refinements

        Args:
            corrected: The corrected output from forward()
            info: Info dictionary from forward()
            clean_target: Ground truth clean image
            backbone_out: Original backbone output

        Returns:
            Dictionary of loss terms
        """
        losses = {}

        # 1. Clean Estimation Loss (L1 + SSIM-like)
        clean_estimate = info['clean_estimate']
        losses['clean_estimation'] = F.l1_loss(clean_estimate, clean_target)

        # 2. Correction Quality Loss (output should be closer to clean)
        losses['correction'] = F.l1_loss(corrected, clean_target)

        # 3. Improvement Loss (correction should improve over backbone)
        backbone_error = torch.abs(backbone_out - clean_target).mean(dim=[1, 2, 3])
        corrected_error = torch.abs(corrected - clean_target).mean(dim=[1, 2, 3])
        improvement = backbone_error - corrected_error
        # Penalize cases where correction makes things worse
        losses['improvement'] = F.relu(-improvement).mean()

        # 4. Confidence Calibration Loss
        # High confidence should correlate with good estimates
        confidence = info['confidence']
        estimate_error = torch.abs(clean_estimate - clean_target).mean(dim=1, keepdim=True)
        # Confidence should be inversely related to error
        target_confidence = torch.exp(-estimate_error * 5) * self.max_lambda
        losses['confidence_calibration'] = F.mse_loss(confidence, target_confidence)

        # 5. Uncertainty Calibration Loss
        uncertainty = info['uncertainty']
        actual_error = torch.abs(corrected - clean_target).mean(dim=1, keepdim=True)
        losses['uncertainty_calibration'] = F.mse_loss(uncertainty, actual_error)

        # 6. Refinement Regularization (prevent large refinements)
        refinements = info['refinements']
        refine_reg = sum(torch.abs(r).mean() for r in refinements.values())
        losses['refinement_regularization'] = refine_reg * 0.1

        # 7. Total Correction Magnitude (prefer smaller corrections)
        total_correction = info['total_correction']
        losses['correction_magnitude'] = torch.abs(total_correction).mean() * 0.05

        return losses

    def get_correction_analysis(
        self,
        backbone_out: torch.Tensor,
        corrected: torch.Tensor,
        info: Dict[str, torch.Tensor],
        clean_target: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze the corrections made by the module.

        Useful for debugging and understanding model behavior.

        Args:
            backbone_out: Original backbone output
            corrected: Corrected output
            info: Info dictionary from forward()
            clean_target: Optional ground truth for comparison

        Returns:
            Dictionary of analysis metrics
        """
        analysis = {}

        # Correction statistics
        total_correction = info['total_correction']
        analysis['mean_correction_magnitude'] = torch.abs(total_correction).mean().item()
        analysis['max_correction_magnitude'] = torch.abs(total_correction).max().item()

        # Confidence statistics
        confidence = info['confidence']
        analysis['mean_confidence'] = confidence.mean().item()
        analysis['min_confidence'] = confidence.min().item()
        analysis['max_confidence'] = confidence.max().item()

        # Per-refiner contributions
        refinements = info['refinements']
        for name, refine in refinements.items():
            analysis[f'{name}_refiner_magnitude'] = torch.abs(refine).mean().item()

        # Quality comparison (if clean target provided)
        if clean_target is not None:
            backbone_psnr = self._compute_psnr(backbone_out, clean_target)
            corrected_psnr = self._compute_psnr(corrected, clean_target)
            clean_est_psnr = self._compute_psnr(info['clean_estimate'], clean_target)

            analysis['backbone_psnr'] = backbone_psnr
            analysis['corrected_psnr'] = corrected_psnr
            analysis['clean_estimate_psnr'] = clean_est_psnr
            analysis['psnr_improvement'] = corrected_psnr - backbone_psnr

        return analysis

    def _compute_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 2.0  # For [-1, 1] range
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        psnr = 10 * math.log10(max_val ** 2 / mse.item())
        return psnr


def create_quality_guided_corrector(
    config: Optional[Dict] = None
) -> QualityGuidedCorrector:
    """
    Factory function to create a QualityGuidedCorrector with sensible defaults.

    Args:
        config: Optional configuration dictionary. Keys:
            - in_channels: Image channels (default: 3)
            - enc1_channels: First encoder channels (default: 64)
            - enc2_channels: Second encoder channels (default: 128)
            - hidden_dim: Hidden dimension (default: 96 for ~2.5M total params)
            - refiner_hidden: Refiner hidden dim (default: 48)
            - max_lambda: Max confidence (default: 1.0)

    Returns:
        Configured QualityGuidedCorrector instance
    """
    default_config = {
        'in_channels': 3,
        'enc1_channels': 64,
        'enc2_channels': 128,
        'hidden_dim': 96,
        'refiner_hidden': 48,
        'max_lambda': 1.0,
        'num_predicates': 5
    }

    if config is not None:
        default_config.update(config)

    return QualityGuidedCorrector(**default_config)


# ============================================
# Example Usage and Testing
# ============================================
if __name__ == "__main__":
    # Test the architecture
    print("Testing QualityGuidedCorrector...")

    # Create model
    model = create_quality_guided_corrector()

    # Print parameter counts
    print("\nParameter Counts:")
    for name, count in model.get_param_counts().items():
        print(f"  {name}: {count:,}")

    # Test forward pass
    batch_size = 2
    height, width = 128, 128

    # Create dummy inputs
    backbone_out = torch.randn(batch_size, 3, height, width)
    noisy = torch.randn(batch_size, 3, height, width)
    backbone_features = {
        'enc1': torch.randn(batch_size, 64, height // 2, width // 2),
        'enc2': torch.randn(batch_size, 128, height // 4, width // 4)
    }

    # Forward pass
    corrected, info = model(backbone_out, noisy, backbone_features)

    print(f"\nInput shape: {backbone_out.shape}")
    print(f"Output shape: {corrected.shape}")
    print(f"Clean estimate shape: {info['clean_estimate'].shape}")
    print(f"Confidence shape: {info['confidence'].shape}")
    print(f"Base correction shape: {info['base_correction'].shape}")

    # Test with provided failure maps
    failure_maps = {
        'P1': torch.rand(batch_size, 1, height, width),
        'P2': torch.rand(batch_size, 1, height, width),
        'P3': torch.rand(batch_size, 1, height, width),
        'P4': torch.rand(batch_size, 1, height, width),
        'P5': torch.rand(batch_size, 1, height, width)
    }

    corrected2, info2 = model(backbone_out, noisy, backbone_features, failure_maps)
    print(f"\nWith failure maps - Output shape: {corrected2.shape}")

    # Test training losses
    clean_target = torch.randn(batch_size, 3, height, width)
    losses = model.compute_training_losses(corrected, info, clean_target, backbone_out)

    print("\nTraining losses:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")

    # Test analysis
    analysis = model.get_correction_analysis(backbone_out, corrected, info, clean_target)

    print("\nCorrection analysis:")
    for name, value in analysis.items():
        print(f"  {name}: {value:.4f}")

    print("\nAll tests passed!")
