#!/usr/bin/env python3
"""
SafeCorrector: A Correction Architecture That Starts as Identity

Key Principle: Corrections should be ZERO at initialization and only grow
when the network is confident they help.

This module implements a novel "safety-first" correction architecture where:
1. Zero-initialized final layers ensure identity mapping at start
2. Confidence gating: correction = confidence * residual, where confidence starts near 0
3. Direct supervision: if correction hurts quality, penalize heavily

Design Philosophy:
-----------------
Traditional correctors often start with random corrections that can hurt quality.
SafeCorrector inverts this: it starts as a perfect identity function and only
learns to make corrections when it has gathered enough evidence that they help.

Mathematical Guarantee:
----------------------
At initialization:
    output = input + 0 * anything = input  (perfect identity)

During training:
    correction = sigmoid(confidence_logit) * gated_residual

Where:
- confidence_logit is initialized to -6 (sigmoid(-6) ~ 0.0025)
- gated_residual is zero-initialized (starts at 0)
- Both must become non-zero for any correction to occur

This double-gating ensures corrections only emerge when:
1. The network has learned a meaningful residual (gated_residual != 0)
2. The network is confident in that residual (confidence > threshold)

Author: Safe Correction Architecture Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ZeroInitConv(nn.Conv2d):
    """
    Convolution with zero-initialized weights and bias.

    Ensures the layer outputs zero at initialization, making it easy to
    build networks that start as identity functions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.zeros_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)


class SafeConvBlock(nn.Module):
    """
    Convolutional block with optional zero initialization of final layer.

    When zero_init=True, this block outputs zero at initialization,
    contributing nothing until the network learns useful features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_norm: bool = True,
        activation: str = 'leaky_relu',
        zero_init: bool = False
    ):
        super().__init__()

        if zero_init:
            self.conv = ZeroInitConv(in_channels, out_channels, kernel_size, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_norm)
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()

        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'none':
            self.act = nn.Identity()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlockWithGate(nn.Module):
    """
    Residual block with a learnable gate that starts at zero.

    At initialization:
        output = input + 0 * residual_path = input

    The gate gradually opens as the network learns useful features.
    """

    def __init__(self, channels: int, use_norm: bool = True):
        super().__init__()

        self.conv1 = SafeConvBlock(channels, channels, use_norm=use_norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_norm)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='linear')

        self.norm = nn.InstanceNorm2d(channels, affine=True) if use_norm else nn.Identity()

        # Learnable gate initialized to near-zero (sigmoid(-4) ~ 0.018)
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.norm(residual)

        # Gate controls how much of the residual passes through
        gate_value = torch.sigmoid(self.gate)

        return x + gate_value * residual


# =============================================================================
# CONFIDENCE GATING MODULE
# =============================================================================

class ConfidenceGate(nn.Module):
    """
    Learns a per-pixel confidence map that gates corrections.

    Key Design:
    - Confidence starts near zero (initialized with negative bias)
    - Confidence is bounded in [0, max_confidence]
    - High confidence requires evidence from features

    The confidence is computed from:
    1. The proposed correction magnitude (large corrections need more confidence)
    2. Local image statistics (uncertain regions get lower confidence)
    3. Backbone features (leverage encoder's semantic understanding)
    """

    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 64,
        hidden_dim: int = 32,
        max_confidence: float = 1.0,
        initial_confidence: float = 0.01
    ):
        """
        Args:
            in_channels: Channels of the image/correction
            feature_channels: Channels of backbone features
            hidden_dim: Internal feature dimension
            max_confidence: Maximum confidence value
            initial_confidence: Confidence at initialization (~sigmoid(-4.6) = 0.01)
        """
        super().__init__()

        self.max_confidence = max_confidence

        # Compute initial bias for desired starting confidence
        # sigmoid(bias) = initial_confidence => bias = log(p/(1-p))
        self.initial_bias = math.log(initial_confidence / (1 - initial_confidence + 1e-8))

        # Feature extraction from image and proposed correction
        # Input: image (in_channels) + proposed_correction (in_channels) + magnitude (1)
        self.feat_extract = nn.Sequential(
            SafeConvBlock(in_channels * 2 + 1, hidden_dim, kernel_size=5, padding=2),
            SafeConvBlock(hidden_dim, hidden_dim),
            ResidualBlockWithGate(hidden_dim)
        )

        # Backbone feature integration
        self.backbone_proj = nn.Conv2d(feature_channels, hidden_dim // 2, 1, bias=True)
        nn.init.xavier_uniform_(self.backbone_proj.weight, gain=0.1)
        nn.init.zeros_(self.backbone_proj.bias)

        # Confidence head - outputs logits, converted to [0, max_confidence] via sigmoid
        self.confidence_head = nn.Sequential(
            SafeConvBlock(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1)
        )

        # Initialize confidence head to output near-zero confidence
        self._init_for_low_confidence()

    def _init_for_low_confidence(self):
        """Initialize to output low confidence at start."""
        # Find the final conv layer and set bias to initial_bias
        final_conv = self.confidence_head[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, self.initial_bias)

    def forward(
        self,
        image: torch.Tensor,
        proposed_correction: torch.Tensor,
        backbone_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute confidence map for the proposed correction.

        Args:
            image: Current image [B, C, H, W]
            proposed_correction: Proposed correction to apply [B, C, H, W]
            backbone_features: Features from backbone encoder [B, F, H', W']

        Returns:
            confidence: Per-pixel confidence in [0, max_confidence]
            confidence_logits: Raw logits (for regularization)
        """
        B, C, H, W = image.shape

        # Compute correction magnitude (how "risky" is this correction)
        magnitude = torch.norm(proposed_correction, dim=1, keepdim=True)

        # Concatenate inputs
        x = torch.cat([image, proposed_correction, magnitude], dim=1)

        # Extract features
        features = self.feat_extract(x)

        # Integrate backbone features
        backbone_proj = self.backbone_proj(backbone_features)
        if backbone_proj.shape[2:] != features.shape[2:]:
            backbone_proj = F.interpolate(
                backbone_proj, size=features.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Combine features
        combined = torch.cat([features, backbone_proj], dim=1)

        # Predict confidence logits
        confidence_logits = self.confidence_head(combined)

        # Convert to bounded confidence
        confidence = torch.sigmoid(confidence_logits) * self.max_confidence

        return confidence, confidence_logits


# =============================================================================
# SAFE RESIDUAL PREDICTOR
# =============================================================================

class SafeResidualPredictor(nn.Module):
    """
    Predicts a correction residual that is zero-initialized.

    Key Design:
    - All paths through the network are zero-initialized
    - Output is bounded by tanh and a learnable (but initially small) scale
    - Multi-scale feature processing for robust predictions

    At initialization:
        residual = 0 (due to zero-initialized final layers)
    """

    def __init__(
        self,
        in_channels: int,
        enc_channels: int = 64,
        hidden_dim: int = 64,
        initial_scale: float = 0.01
    ):
        """
        Args:
            in_channels: Image channels
            enc_channels: Backbone encoder channels
            hidden_dim: Internal feature dimension
            initial_scale: Initial maximum correction magnitude
        """
        super().__init__()

        self.initial_scale = initial_scale

        # Input projection
        self.input_proj = SafeConvBlock(in_channels * 2, hidden_dim)  # image + noisy

        # Backbone feature integration
        self.enc_proj = nn.Conv2d(enc_channels, hidden_dim // 2, 1, bias=True)
        nn.init.xavier_uniform_(self.enc_proj.weight, gain=0.1)
        nn.init.zeros_(self.enc_proj.bias)

        # Multi-scale processing (dilated convolutions)
        self.multiscale = nn.ModuleList([
            SafeConvBlock(hidden_dim + hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            SafeConvBlock(hidden_dim + hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=2),
            SafeConvBlock(hidden_dim + hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=4),
            SafeConvBlock(hidden_dim + hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=8),
        ])
        # Update dilation for each conv
        self.multiscale[1].conv.dilation = (2, 2)
        self.multiscale[2].conv.dilation = (4, 4)
        self.multiscale[3].conv.dilation = (8, 8)

        # Feature refinement
        self.refine = nn.Sequential(
            ResidualBlockWithGate(hidden_dim),
            ResidualBlockWithGate(hidden_dim)
        )

        # Output head - ZERO INITIALIZED for identity start
        self.output_head = nn.Sequential(
            SafeConvBlock(hidden_dim, hidden_dim // 2),
            ZeroInitConv(hidden_dim // 2, in_channels, 3, padding=1)  # Zero-init!
        )

        # Learnable scale, initialized very small
        # This grows during training as the network learns useful corrections
        self.scale = nn.Parameter(torch.tensor(math.log(initial_scale)))  # log-scale for stability

    def forward(
        self,
        image: torch.Tensor,
        noisy: torch.Tensor,
        backbone_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict correction residual.

        Args:
            image: Current denoised image [B, C, H, W]
            noisy: Original noisy input [B, C, H, W]
            backbone_features: Features from backbone encoder

        Returns:
            residual: Predicted correction (zero at initialization)
        """
        B, C, H, W = image.shape

        # Input features
        x = torch.cat([image, noisy], dim=1)
        features = self.input_proj(x)

        # Integrate backbone features
        enc_proj = self.enc_proj(backbone_features)
        if enc_proj.shape[2:] != features.shape[2:]:
            enc_proj = F.interpolate(enc_proj, size=features.shape[2:], mode='bilinear', align_corners=False)

        combined = torch.cat([features, enc_proj], dim=1)

        # Multi-scale processing
        ms_feats = [conv(combined) for conv in self.multiscale]
        features = torch.cat(ms_feats, dim=1)  # Back to hidden_dim

        # Refine
        features = self.refine(features)

        # Output (zero at initialization due to ZeroInitConv)
        raw_residual = self.output_head(features)

        # Bound residual with tanh and scale
        # At init: scale = exp(log(0.01)) = 0.01, and raw_residual ~ 0
        # So output ~ 0 (identity)
        scale = torch.exp(self.scale).clamp(max=0.5)  # Max scale of 0.5
        residual = torch.tanh(raw_residual) * scale

        return residual


# =============================================================================
# VERIFICATION MODULE
# =============================================================================

class CorrectionVerifier(nn.Module):
    """
    Verifies whether a proposed correction improves quality.

    This module learns to predict whether applying a correction will
    improve or hurt image quality. During training, it receives direct
    supervision from actual quality measurements.

    Key Design:
    - Predicts a "quality delta" score for each pixel
    - Positive score means correction helps, negative means it hurts
    - Can be used to gate corrections during inference
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32
    ):
        super().__init__()

        # Input: original image, corrected image, correction
        self.encoder = nn.Sequential(
            SafeConvBlock(in_channels * 3, hidden_dim),
            ResidualBlockWithGate(hidden_dim),
            SafeConvBlock(hidden_dim, hidden_dim // 2)
        )

        # Quality delta prediction
        self.quality_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1], represents quality change
        )

        # Initialize to predict zero (neutral) quality change
        nn.init.zeros_(self.quality_head[0].weight)
        nn.init.zeros_(self.quality_head[0].bias)

    def forward(
        self,
        original: torch.Tensor,
        corrected: torch.Tensor,
        correction: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict quality change from applying the correction.

        Args:
            original: Original image before correction
            corrected: Image after applying correction
            correction: The correction that was applied

        Returns:
            quality_delta: Per-pixel quality change prediction in [-1, 1]
                          Positive = improvement, Negative = degradation
        """
        x = torch.cat([original, corrected, correction], dim=1)
        features = self.encoder(x)
        quality_delta = self.quality_head(features)
        return quality_delta


# =============================================================================
# SAFE CORRECTOR - MAIN MODULE
# =============================================================================

class SafeCorrector(nn.Module):
    """
    Safe Corrector: A correction architecture that starts as identity
    and only learns to correct when confident it helps.

    Key Principles:
    1. Zero output at initialization (perfect identity)
    2. Confidence gating: correction = confidence * residual
    3. Verification: only allow corrections that pass quality check

    Architecture:
    ------------
    1. SafeResidualPredictor: Predicts correction residual (zero-init)
    2. ConfidenceGate: Predicts per-pixel confidence (starts near 0)
    3. CorrectionVerifier: Verifies correction improves quality

    Forward Pass:
    ------------
        residual = SafeResidualPredictor(image, noisy, features)  # ~0 at init
        confidence = ConfidenceGate(image, residual, features)    # ~0 at init
        correction = confidence * residual                        # ~0 at init
        output = image + correction                               # = image at init

    The correction only becomes non-zero when BOTH:
    - The residual predictor learns a useful correction
    - The confidence gate learns to trust that correction

    This double-gating ensures safety: even if one component makes a mistake,
    the other can prevent harmful corrections.

    Training:
    --------
    The module includes a verification loss that heavily penalizes corrections
    that hurt quality (measured by PSNR, SSIM, etc.). This encourages the
    network to be conservative and only correct when confident.
    """

    def __init__(
        self,
        in_channels: int = 1,
        enc1_channels: int = 64,
        enc2_channels: int = 128,
        hidden_dim: int = 64,
        max_confidence: float = 1.0,
        initial_confidence: float = 0.01,
        initial_scale: float = 0.01,
        use_verification: bool = True
    ):
        """
        Initialize the Safe Corrector.

        Args:
            in_channels: Image channels (1 for grayscale OCT)
            enc1_channels: First backbone encoder channels
            enc2_channels: Second backbone encoder channels
            hidden_dim: Internal feature dimension
            max_confidence: Maximum confidence value
            initial_confidence: Confidence at initialization (should be very small)
            initial_scale: Initial maximum correction magnitude
            use_verification: Whether to use the verification module
        """
        super().__init__()

        self.in_channels = in_channels
        self.max_confidence = max_confidence
        self.use_verification = use_verification

        # Backbone feature adapters - both output hidden_dim for easy combination
        self.adapt_enc1 = nn.Conv2d(enc1_channels, hidden_dim, 1, bias=True)
        self.adapt_enc2 = nn.Conv2d(enc2_channels, hidden_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.adapt_enc1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.adapt_enc2.weight, gain=0.1)
        nn.init.zeros_(self.adapt_enc1.bias)
        nn.init.zeros_(self.adapt_enc2.bias)

        # 1. Safe Residual Predictor (zero-initialized output)
        self.residual_predictor = SafeResidualPredictor(
            in_channels=in_channels,
            enc_channels=hidden_dim,
            hidden_dim=hidden_dim,
            initial_scale=initial_scale
        )

        # 2. Confidence Gate (starts near zero)
        self.confidence_gate = ConfidenceGate(
            in_channels=in_channels,
            feature_channels=hidden_dim,
            hidden_dim=hidden_dim // 2,
            max_confidence=max_confidence,
            initial_confidence=initial_confidence
        )

        # 3. Correction Verifier (optional)
        if use_verification:
            self.verifier = CorrectionVerifier(
                in_channels=in_channels,
                hidden_dim=hidden_dim // 2
            )
        else:
            self.verifier = None

        # Global safety gate - an additional learnable gate that starts closed
        # This provides a global "off switch" that opens as training progresses
        self.global_gate = nn.Parameter(torch.tensor(-6.0))  # sigmoid(-6) ~ 0.0025

        # Statistics tracking
        self.register_buffer('correction_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))

        self._print_param_count()

    def _print_param_count(self):
        """Print parameter counts for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total = count_params(self)
        residual = count_params(self.residual_predictor)
        confidence = count_params(self.confidence_gate)
        verifier = count_params(self.verifier) if self.verifier else 0

        self._param_counts = {
            'Total': total,
            'ResidualPredictor': residual,
            'ConfidenceGate': confidence,
            'Verifier': verifier
        }

    def get_param_counts(self) -> Dict[str, int]:
        """Return parameter counts."""
        return self._param_counts

    def forward(
        self,
        denoised: torch.Tensor,
        noisy: torch.Tensor,
        backbone_features: Dict[str, torch.Tensor],
        lambda_maps: Optional[Dict[str, torch.Tensor]] = None,
        verification_threshold: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply safe correction.

        At initialization, this is approximately an identity function:
            output ~ denoised

        As training progresses, corrections emerge only where the network
        is confident they help.

        Args:
            denoised: Denoised image from backbone [B, C, H, W]
            noisy: Original noisy input [B, C, H, W]
            backbone_features: Dictionary with 'enc1' and 'enc2' features
            lambda_maps: Optional external lambda maps (for compatibility)
            verification_threshold: Minimum verification score to apply correction

        Returns:
            corrected: Safe-corrected output
            info: Dictionary with intermediate results
        """
        B, C, H, W = denoised.shape

        # Get and adapt backbone features
        enc1 = backbone_features.get('enc1')
        enc2 = backbone_features.get('enc2')

        # Combine encoder features
        f1 = self.adapt_enc1(enc1)
        if f1.shape[2:] != denoised.shape[2:]:
            f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)

        f2 = self.adapt_enc2(enc2)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

        features = f1 + F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        # ============================================
        # Step 1: Predict Residual (zero at init)
        # ============================================
        residual = self.residual_predictor(denoised, noisy, features)

        # ============================================
        # Step 2: Compute Confidence (near-zero at init)
        # ============================================
        confidence, confidence_logits = self.confidence_gate(
            denoised, residual, features
        )

        # ============================================
        # Step 3: Apply Global Safety Gate
        # ============================================
        global_gate = torch.sigmoid(self.global_gate)
        gated_confidence = confidence * global_gate

        # ============================================
        # Step 4: Compute Gated Correction
        # ============================================
        # correction = confidence * residual
        # At initialization: ~0 * ~0 = ~0
        correction = gated_confidence * residual

        # ============================================
        # Step 5: Verification (if enabled)
        # ============================================
        if self.verifier is not None:
            corrected_candidate = denoised + correction
            verification_score = self.verifier(denoised, corrected_candidate, correction)

            # Optional: mask out corrections with negative verification
            if verification_threshold > 0:
                # Only keep corrections where verification > threshold
                keep_mask = (verification_score > verification_threshold).float()
                correction = correction * keep_mask
        else:
            verification_score = None

        # ============================================
        # Step 6: Apply Correction
        # ============================================
        corrected = denoised + correction
        corrected = torch.clamp(corrected, 0, 1)

        # ============================================
        # Track Statistics
        # ============================================
        with torch.no_grad():
            correction_mag = correction.abs().mean().item()
            idx = self.history_idx.item() % 100
            self.correction_history[idx] = correction_mag
            self.history_idx += 1

        # ============================================
        # Gather Info
        # ============================================
        info = {
            'residual': residual,
            'confidence': confidence,
            'confidence_logits': confidence_logits,
            'global_gate': global_gate,
            'gated_confidence': gated_confidence,
            'correction': correction,
            'verification_score': verification_score,
            'correction_magnitude': correction.abs().mean().item(),
            'residual_magnitude': residual.abs().mean().item(),
            'mean_confidence': confidence.mean().item(),
            'global_gate_value': global_gate.item(),
            'residual_scale': torch.exp(self.residual_predictor.scale).item()
        }

        return corrected, info

    def compute_safe_loss(
        self,
        corrected: torch.Tensor,
        info: Dict[str, torch.Tensor],
        clean_target: torch.Tensor,
        backbone_out: torch.Tensor,
        psnr_weight: float = 10.0,
        confidence_reg_weight: float = 0.1,
        magnitude_reg_weight: float = 0.05
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses with heavy penalty for quality degradation.

        Key loss terms:
        1. Reconstruction loss (L1): Standard pixel loss
        2. Quality degradation penalty: Heavy penalty if correction hurts PSNR
        3. Confidence regularization: Encourage low confidence when uncertain
        4. Magnitude regularization: Prefer smaller corrections

        Args:
            corrected: Corrected output from forward()
            info: Info dictionary from forward()
            clean_target: Ground truth clean image
            backbone_out: Original backbone output
            psnr_weight: Weight for PSNR degradation penalty
            confidence_reg_weight: Weight for confidence regularization
            magnitude_reg_weight: Weight for magnitude regularization

        Returns:
            Dictionary of loss terms
        """
        losses = {}

        # 1. Reconstruction Loss
        losses['reconstruction'] = F.l1_loss(corrected, clean_target)

        # 2. Quality Degradation Penalty
        # Compute per-sample MSE
        backbone_mse = ((backbone_out - clean_target) ** 2).mean(dim=[1, 2, 3])
        corrected_mse = ((corrected - clean_target) ** 2).mean(dim=[1, 2, 3])

        # Penalty when correction makes MSE worse (higher)
        degradation = F.relu(corrected_mse - backbone_mse)
        losses['degradation_penalty'] = degradation.mean() * psnr_weight

        # 3. Verification Loss (if verifier is used)
        if info['verification_score'] is not None:
            # Ground truth: positive where correction helps, negative where it hurts
            per_pixel_improvement = (
                (backbone_out - clean_target).abs() -
                (corrected.detach() - clean_target).abs()
            ).mean(dim=1, keepdim=True)

            # Scale to [-1, 1]
            target_verification = torch.tanh(per_pixel_improvement * 10)

            losses['verification'] = F.mse_loss(
                info['verification_score'],
                target_verification
            )

        # 4. Confidence Regularization
        # Penalize high confidence when correction doesn't help
        confidence = info['confidence']
        correction = info['correction']

        # Where correction is large but doesn't help clean, penalize confidence
        correction_magnitude = correction.abs().mean(dim=1, keepdim=True)
        actual_error_reduction = (
            (backbone_out - clean_target).abs() -
            (corrected - clean_target).abs()
        ).mean(dim=1, keepdim=True)

        # High confidence with negative error reduction is bad
        confidence_error = confidence * F.relu(-actual_error_reduction)
        losses['confidence_reg'] = confidence_error.mean() * confidence_reg_weight

        # 5. Magnitude Regularization (prefer smaller corrections)
        losses['magnitude_reg'] = correction.abs().mean() * magnitude_reg_weight

        # 6. Confidence Prior (encourage low confidence early in training)
        # This gradually relaxes as the global_gate opens
        global_gate = info['global_gate']
        confidence_prior = confidence.mean() * (1 - global_gate)  # Decreases as gate opens
        losses['confidence_prior'] = confidence_prior * 0.01

        return losses

    def get_safety_analysis(
        self,
        backbone_out: torch.Tensor,
        corrected: torch.Tensor,
        info: Dict[str, torch.Tensor],
        clean_target: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze the safety and effectiveness of corrections.

        Returns metrics about how "safe" the corrections are.
        """
        analysis = {}

        # Basic statistics
        analysis['correction_magnitude'] = info['correction_magnitude']
        analysis['residual_magnitude'] = info['residual_magnitude']
        analysis['mean_confidence'] = info['mean_confidence']
        analysis['global_gate'] = info['global_gate_value']
        analysis['residual_scale'] = info['residual_scale']

        # Safety metrics
        analysis['max_correction'] = info['correction'].abs().max().item()
        analysis['high_confidence_ratio'] = (info['confidence'] > 0.5).float().mean().item()

        # Historical statistics
        history = self.correction_history[:self.history_idx.item()]
        if len(history) > 0:
            analysis['avg_correction_history'] = history.mean().item()

        # Quality comparison (if clean target available)
        if clean_target is not None:
            backbone_psnr = self._compute_psnr(backbone_out, clean_target)
            corrected_psnr = self._compute_psnr(corrected, clean_target)

            analysis['backbone_psnr'] = backbone_psnr
            analysis['corrected_psnr'] = corrected_psnr
            analysis['psnr_improvement'] = corrected_psnr - backbone_psnr
            analysis['quality_improved'] = corrected_psnr > backbone_psnr

        return analysis

    def _compute_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 1.0
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, target)
        if mse < 1e-10:
            return 100.0
        psnr = 10 * math.log10(max_val ** 2 / mse.item())
        return psnr

    def reset_to_identity(self):
        """
        Reset the corrector to identity function.

        Useful for debugging or when you want to restart training.
        """
        # Reset global gate
        self.global_gate.data.fill_(-6.0)

        # Reset residual predictor scale
        self.residual_predictor.scale.data.fill_(math.log(0.01))

        # Reset output layer of residual predictor
        nn.init.zeros_(self.residual_predictor.output_head[-1].weight)
        nn.init.zeros_(self.residual_predictor.output_head[-1].bias)

        # Reset confidence gate
        self.confidence_gate._init_for_low_confidence()

        # Reset gated residual blocks
        for module in self.modules():
            if isinstance(module, ResidualBlockWithGate):
                module.gate.data.fill_(-4.0)

        print("SafeCorrector reset to identity function.")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_safe_corrector(
    in_channels: int = 1,
    enc1_channels: int = 64,
    enc2_channels: int = 128,
    hidden_dim: int = 64,
    max_confidence: float = 1.0,
    use_verification: bool = True
) -> SafeCorrector:
    """
    Factory function to create a SafeCorrector with sensible defaults.

    Args:
        in_channels: Image channels (1 for grayscale OCT)
        enc1_channels: First backbone encoder channels
        enc2_channels: Second backbone encoder channels
        hidden_dim: Internal feature dimension
        max_confidence: Maximum confidence value
        use_verification: Whether to use verification module

    Returns:
        Configured SafeCorrector instance
    """
    return SafeCorrector(
        in_channels=in_channels,
        enc1_channels=enc1_channels,
        enc2_channels=enc2_channels,
        hidden_dim=hidden_dim,
        max_confidence=max_confidence,
        initial_confidence=0.01,
        initial_scale=0.01,
        use_verification=use_verification
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing SafeCorrector - Identity at Initialization")
    print("=" * 70)

    # Create model
    model = create_safe_corrector(
        in_channels=1,
        enc1_channels=64,
        enc2_channels=128,
        hidden_dim=64
    )

    # Print parameter counts
    print("\nParameter Counts:")
    for name, count in model.get_param_counts().items():
        print(f"  {name}: {count:,}")

    # Create test inputs
    B, H, W = 2, 64, 64
    denoised = torch.rand(B, 1, H, W)
    noisy = denoised + torch.randn_like(denoised) * 0.1
    backbone_features = {
        'enc1': torch.randn(B, 64, H, W),
        'enc2': torch.randn(B, 128, H // 2, W // 2)
    }

    # Test forward pass at initialization
    print("\n" + "=" * 70)
    print("INITIALIZATION TEST - Should be near-identity")
    print("=" * 70)

    with torch.no_grad():
        corrected, info = model(denoised, noisy, backbone_features)

    # Compute difference from identity
    identity_diff = (corrected - denoised).abs()

    print(f"\nForward pass successful!")
    print(f"Input shape: {denoised.shape}")
    print(f"Output shape: {corrected.shape}")

    print(f"\n--- Correction Statistics at Initialization ---")
    print(f"Correction magnitude: {info['correction_magnitude']:.6f}")
    print(f"Residual magnitude:   {info['residual_magnitude']:.6f}")
    print(f"Mean confidence:      {info['mean_confidence']:.6f}")
    print(f"Global gate:          {info['global_gate_value']:.6f}")
    print(f"Residual scale:       {info['residual_scale']:.6f}")

    print(f"\n--- Identity Check ---")
    print(f"Mean absolute diff from input:  {identity_diff.mean().item():.8f}")
    print(f"Max absolute diff from input:   {identity_diff.max().item():.8f}")

    # Verify near-identity
    if identity_diff.mean().item() < 0.001:
        print("\n[PASS] SafeCorrector is near-identity at initialization!")
    else:
        print("\n[WARNING] Correction is larger than expected at init")

    # Test with clean target (for loss computation)
    print("\n" + "=" * 70)
    print("LOSS COMPUTATION TEST")
    print("=" * 70)

    clean_target = denoised  # Use denoised as clean for testing
    losses = model.compute_safe_loss(corrected, info, clean_target, denoised)

    print("\nTraining losses:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.6f}")

    # Test safety analysis
    print("\n" + "=" * 70)
    print("SAFETY ANALYSIS")
    print("=" * 70)

    analysis = model.get_safety_analysis(denoised, corrected, info, clean_target)
    print("\nSafety analysis:")
    for name, value in analysis.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")

    # Test reset to identity
    print("\n" + "=" * 70)
    print("RESET TO IDENTITY TEST")
    print("=" * 70)

    # Manually modify some parameters
    model.global_gate.data.fill_(0.0)  # Open the gate
    model.residual_predictor.scale.data.fill_(math.log(0.5))  # Increase scale

    with torch.no_grad():
        corrected_modified, info_modified = model(denoised, noisy, backbone_features)

    print(f"After modification - Correction magnitude: {info_modified['correction_magnitude']:.6f}")

    # Reset
    model.reset_to_identity()

    with torch.no_grad():
        corrected_reset, info_reset = model(denoised, noisy, backbone_features)

    print(f"After reset - Correction magnitude: {info_reset['correction_magnitude']:.6f}")

    if info_reset['correction_magnitude'] < 0.001:
        print("\n[PASS] Reset to identity successful!")
    else:
        print("\n[WARNING] Reset may not have fully restored identity")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
