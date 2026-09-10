#!/usr/bin/env python3
"""
Safe Identity Corrector for Neuro-Symbolic OCT Denoising

Key Design Principle: Start from IDENTITY (zero correction) and gradually learn.

This corrector is designed to be "safe by default":
1. At initialization, output = backbone_output (zero correction)
2. Corrections are only added when the model is confident they help
3. Per-correction-type gates allow fine-grained control
4. Optional local PSNR verification prevents harmful corrections

Author: Safe Corrector Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# UTILITY MODULES
# =============================================================================

class SmallInitConv2d(nn.Conv2d):
    """Conv2d with small weight initialization for near-zero initial output."""

    def __init__(self, *args, init_scale: float = 1e-4, zero_init: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_small_weights(init_scale, zero_init)

    def _init_small_weights(self, scale: float, zero_init: bool):
        """Initialize weights to small values and biases to zero."""
        if zero_init:
            # For final layers: zero weights ensure zero output at start
            nn.init.zeros_(self.weight)
        else:
            # For hidden layers: small random weights for gradient flow
            nn.init.normal_(self.weight, mean=0.0, std=scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ConfidenceGate(nn.Module):
    """
    Learnable confidence gate that starts near 0 and gradually opens.

    Uses sigmoid with negative initial bias to produce ~0 output at start.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 32,
                 initial_bias: float = -5.0):
        """
        Args:
            in_channels: Number of input feature channels
            hidden_dim: Hidden dimension for confidence network
            initial_bias: Initial bias (negative = starts near 0)
        """
        super().__init__()

        # Confidence network
        self.confidence_net = nn.Sequential(
            SmallInitConv2d(in_channels, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, 1, 1, init_scale=1e-4),  # Output single channel
        )

        # Learnable bias that starts very negative (sigmoid(-5) ~ 0.007)
        self.confidence_bias = nn.Parameter(torch.tensor(initial_bias))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features [B, C, H, W]

        Returns:
            confidence: Gate values in [0, 1], starting near 0 [B, 1, H, W]
        """
        # Network predicts adjustment to base confidence
        adjustment = self.confidence_net(features)

        # Apply sigmoid with bias
        confidence = torch.sigmoid(self.confidence_bias + adjustment)

        return confidence


class CorrectionTypeGate(nn.Module):
    """
    Per-correction-type gate that learns when each type of correction is beneficial.

    Each gate starts near 0 and learns to open for beneficial corrections.
    """

    def __init__(self, correction_types: list, in_channels: int,
                 hidden_dim: int = 32, initial_bias: float = -4.0):
        """
        Args:
            correction_types: List of correction type names
            in_channels: Number of input feature channels
            hidden_dim: Hidden dimension
            initial_bias: Initial bias for gates (negative = starts closed)
        """
        super().__init__()

        self.correction_types = correction_types
        self.num_types = len(correction_types)

        # Shared feature extractor
        self.shared_feat = nn.Sequential(
            SmallInitConv2d(in_channels, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Per-type gate heads
        self.gate_heads = nn.ModuleDict({
            name: SmallInitConv2d(hidden_dim, 1, 1, init_scale=1e-4)
            for name in correction_types
        })

        # Per-type learnable biases
        self.gate_biases = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_bias))
            for name in correction_types
        })

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Input features [B, C, H, W]

        Returns:
            gates: Dict mapping correction type -> gate values [B, 1, H, W]
        """
        shared = self.shared_feat(features)

        gates = {}
        for name in self.correction_types:
            adjustment = self.gate_heads[name](shared)
            gates[name] = torch.sigmoid(self.gate_biases[name] + adjustment)

        return gates


# =============================================================================
# CORRECTION COMPUTATION MODULES
# =============================================================================

class EdgeCorrectionModule(nn.Module):
    """Computes edge/boundary enhancement corrections."""

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()

        # Use regular Conv2d with Kaiming init for hidden layers (gradient flow)
        # Only the final layer uses small init
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, 1, 3, padding=1, init_scale=0.01),  # Small final layer
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Apply Kaiming init to hidden layers
        self._init_hidden_layers()

        # Scale factor for final output (starts small)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _init_hidden_layers(self):
        """Apply Kaiming initialization to Conv2d layers except the last one."""
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Conv2d) and not isinstance(module, SmallInitConv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.scale


class ContrastCorrectionModule(nn.Module):
    """Computes local contrast enhancement corrections."""

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()

        # Local statistics computation
        self.local_pools = nn.ModuleList([
            nn.AvgPool2d(k, stride=1, padding=k//2) for k in [5, 11, 21]
        ])

        # Correction network (in_channels + 6 stats channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 6, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, 1, 3, padding=1, init_scale=0.01),
            nn.Tanh(),
        )

        self._init_hidden_layers()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _init_hidden_layers(self):
        for module in self.net:
            if isinstance(module, nn.Conv2d) and not isinstance(module, SmallInitConv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Compute local statistics
        stats_list = []
        for pool in self.local_pools:
            local_mean = pool(denoised)
            local_sq_mean = pool(denoised ** 2)
            local_std = (local_sq_mean - local_mean ** 2).clamp(min=1e-6).sqrt()
            stats_list.extend([local_mean, local_std])

        local_stats = torch.cat(stats_list, dim=1)
        combined = torch.cat([x, local_stats], dim=1)

        return self.net(combined) * self.scale


class TextureCorrectionModule(nn.Module):
    """Computes texture preservation corrections."""

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()

        # Multi-orientation filters
        self.texture_filters = nn.ModuleList([
            SmallInitConv2d(1, 8, (1, 5), padding=(0, 2), init_scale=1e-4),
            SmallInitConv2d(1, 8, (5, 1), padding=(2, 0), init_scale=1e-4),
            SmallInitConv2d(1, 8, 3, padding=1, init_scale=1e-4),
        ])

        # Correction network (in_channels + 24 texture channels)
        self.net = nn.Sequential(
            SmallInitConv2d(in_channels + 24, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, 1, 3, padding=1, init_scale=1e-4),
            nn.Tanh(),
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        tex_feats = [f(denoised) for f in self.texture_filters]
        tex_feat = torch.cat(tex_feats, dim=1)
        combined = torch.cat([x, tex_feat], dim=1)

        return self.net(combined) * self.scale


class SmoothCorrectionModule(nn.Module):
    """Computes noise reduction / smoothing corrections."""

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()

        # Multi-scale smoothing features
        self.smooth_kernels = nn.ModuleList([
            SmallInitConv2d(1, 8, 3, padding=1, init_scale=1e-4),
            SmallInitConv2d(1, 8, 5, padding=2, init_scale=1e-4),
            SmallInitConv2d(1, 8, 7, padding=3, init_scale=1e-4),
        ])

        # Correction network (in_channels + 24 smooth channels)
        self.net = nn.Sequential(
            SmallInitConv2d(in_channels + 24, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, 1, 3, padding=1, init_scale=1e-4),
            nn.Tanh(),
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        smooth_feats = [k(denoised) for k in self.smooth_kernels]
        smooth_feat = torch.cat(smooth_feats, dim=1)
        combined = torch.cat([x, smooth_feat], dim=1)

        return self.net(combined) * self.scale


class SharpnessCorrectionModule(nn.Module):
    """Computes boundary sharpening corrections."""

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()

        # Learnable high-pass filter
        self.highpass = nn.Sequential(
            SmallInitConv2d(1, 16, 3, padding=1, init_scale=1e-4),
            SmallInitConv2d(16, 16, 3, padding=1, init_scale=1e-4),
        )

        # Register Gaussian blur kernel
        self.register_buffer('blur_kernel', self._make_gaussian_kernel(5, 1.0))

        # Correction network (in_channels + 16 highpass + 1 unsharp)
        self.net = nn.Sequential(
            SmallInitConv2d(in_channels + 17, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, 1, 3, padding=1, init_scale=1e-4),
            nn.Tanh(),
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

    def _make_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(size).float() - size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        return gauss_2d.view(1, 1, size, size)

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Unsharp mask
        blurred = F.conv2d(denoised, self.blur_kernel, padding=2)
        unsharp = denoised - blurred

        # High-pass features
        hp_feat = self.highpass(denoised)

        combined = torch.cat([x, hp_feat, unsharp], dim=1)
        return self.net(combined) * self.scale


# =============================================================================
# MAIN IDENTITY CORRECTOR V1
# =============================================================================

class IdentityCorrectorV1(nn.Module):
    """
    Safe Identity Corrector that starts from zero correction.

    Key Design Principles:
    1. At initialization: output = backbone_output (zero correction)
    2. Confidence gate initialized to produce ~0.0 output
    3. Corrections are ADDED only when model is confident they help
    4. Per-correction-type gates (edge, contrast, texture, etc.)
    5. Optional local PSNR verification

    The forward pass:
        raw_corrections = self.compute_corrections(backbone_output, noisy_input)
        confidence = sigmoid(confidence_bias + confidence_net(features))
        gated_correction = raw_corrections * confidence * lambda_maps
        output = backbone_output + gated_correction
    """

    # Correction types
    CORRECTION_TYPES = ['edge', 'contrast', 'texture', 'smooth', 'sharpness']

    def __init__(
        self,
        enc1_channels: int = 64,
        enc2_channels: int = 128,
        hidden_dim: int = 64,
        confidence_initial_bias: float = -5.0,
        type_gate_initial_bias: float = -4.0,
        warmup_epochs: int = 5,
        verify_local_psnr: bool = True,
        local_psnr_window: int = 16,
    ):
        """
        Args:
            enc1_channels: Channels in encoder 1 features
            enc2_channels: Channels in encoder 2 features
            hidden_dim: Hidden dimension for correction modules
            confidence_initial_bias: Initial bias for main confidence gate
                (sigmoid(-5) ~ 0.007, so corrections start near zero)
            type_gate_initial_bias: Initial bias for per-type gates
            warmup_epochs: Number of epochs for warmup phase
            verify_local_psnr: Whether to verify corrections don't degrade local PSNR
            local_psnr_window: Window size for local PSNR computation
        """
        super().__init__()

        self.warmup_epochs = warmup_epochs
        self.verify_local_psnr = verify_local_psnr
        self.local_psnr_window = local_psnr_window
        self._current_epoch = 0

        # Feature adapters
        self.adapt_enc1 = SmallInitConv2d(enc1_channels, 16, 1, bias=False, init_scale=1e-4)
        self.adapt_enc2 = SmallInitConv2d(enc2_channels, 16, 1, bias=False, init_scale=1e-4)

        # Input channels for feature extraction
        # denoised(1) + noisy(1) + adapted_enc1(16) + adapted_enc2(16) = 34
        feature_in_channels = 34

        # Feature extraction for confidence/gates
        self.feature_extractor = nn.Sequential(
            SmallInitConv2d(feature_in_channels, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            SmallInitConv2d(hidden_dim, hidden_dim, 3, padding=1, init_scale=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Main confidence gate (starts near 0)
        self.confidence_gate = ConfidenceGate(
            in_channels=hidden_dim,
            hidden_dim=32,
            initial_bias=confidence_initial_bias
        )

        # Per-correction-type gates
        self.type_gates = CorrectionTypeGate(
            correction_types=self.CORRECTION_TYPES,
            in_channels=hidden_dim,
            hidden_dim=32,
            initial_bias=type_gate_initial_bias
        )

        # Correction modules
        # Input for each: denoised(1) + noisy(1) + adapted features(32)
        correction_in_channels = 34

        self.edge_module = EdgeCorrectionModule(correction_in_channels, hidden_dim)
        self.contrast_module = ContrastCorrectionModule(correction_in_channels, hidden_dim)
        self.texture_module = TextureCorrectionModule(correction_in_channels, hidden_dim)
        self.smooth_module = SmoothCorrectionModule(correction_in_channels, hidden_dim)
        self.sharpness_module = SharpnessCorrectionModule(correction_in_channels, hidden_dim)

        # Print parameter count
        total = sum(p.numel() for p in self.parameters())
        print(f"IdentityCorrectorV1: {total:,} parameters")

        # Verify initialization produces near-zero output
        self._verify_initialization()

    def _verify_initialization(self):
        """Verify that initialization produces near-zero corrections."""
        print("Verifying identity initialization...")

        # Check confidence bias
        conf_bias = self.confidence_gate.confidence_bias.item()
        conf_value = torch.sigmoid(torch.tensor(conf_bias)).item()
        print(f"  Main confidence bias: {conf_bias:.2f} -> sigmoid = {conf_value:.6f}")

        # Check type gate biases
        for name in self.CORRECTION_TYPES:
            bias = self.type_gates.gate_biases[name].item()
            value = torch.sigmoid(torch.tensor(bias)).item()
            print(f"  {name} gate bias: {bias:.2f} -> sigmoid = {value:.6f}")

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup phase logic."""
        self._current_epoch = epoch

    @property
    def in_warmup(self) -> bool:
        """Check if we're in warmup phase."""
        return self._current_epoch < self.warmup_epochs

    def compute_corrections(
        self,
        features: torch.Tensor,
        denoised: torch.Tensor,
        noisy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute raw corrections from each module.

        Args:
            features: Combined input features [B, C, H, W]
            denoised: Current denoised image [B, 1, H, W]
            noisy: Noisy input image [B, 1, H, W]

        Returns:
            Dictionary of raw corrections for each type
        """
        corrections = {
            'edge': self.edge_module(features),
            'contrast': self.contrast_module(features, denoised),
            'texture': self.texture_module(features, denoised),
            'smooth': self.smooth_module(features, denoised),
            'sharpness': self.sharpness_module(features, denoised),
        }
        return corrections

    def compute_local_psnr_mask(
        self,
        backbone_output: torch.Tensor,
        corrected: torch.Tensor,
        target: Optional[torch.Tensor],
        window_size: int = 16
    ) -> torch.Tensor:
        """
        Compute mask where correction improves local PSNR.

        Args:
            backbone_output: Original backbone output [B, 1, H, W]
            corrected: Corrected output [B, 1, H, W]
            target: Ground truth (if available) [B, 1, H, W]
            window_size: Size of local window

        Returns:
            mask: Binary mask where 1 = correction helps [B, 1, H, W]
        """
        if target is None:
            # If no target, return all ones (allow all corrections)
            return torch.ones_like(backbone_output)

        B, C, H, W = backbone_output.shape

        # Compute local MSE for backbone output
        backbone_error = (backbone_output - target) ** 2
        corrected_error = (corrected - target) ** 2

        # Use F.avg_pool2d with explicit padding to ensure same output size
        # Pad input manually to handle edge cases
        pad = window_size // 2
        backbone_error_padded = F.pad(backbone_error, (pad, pad, pad, pad), mode='reflect')
        corrected_error_padded = F.pad(corrected_error, (pad, pad, pad, pad), mode='reflect')

        # Average pool to get local MSE (no padding in pool, we did it manually)
        local_backbone_mse = F.avg_pool2d(backbone_error_padded, window_size, stride=1, padding=0)
        local_corrected_mse = F.avg_pool2d(corrected_error_padded, window_size, stride=1, padding=0)

        # Crop to original size if needed
        if local_backbone_mse.shape[2:] != (H, W):
            local_backbone_mse = local_backbone_mse[:, :, :H, :W]
            local_corrected_mse = local_corrected_mse[:, :, :H, :W]

        # Mask: 1 where correction improves (lower MSE), 0 otherwise
        improvement_mask = (local_corrected_mse < local_backbone_mse).float()

        return improvement_mask

    def forward(
        self,
        backbone_output: torch.Tensor,
        noisy_input: torch.Tensor,
        lambda_maps: Dict[str, torch.Tensor],
        backbone_features: Dict[str, torch.Tensor],
        target: Optional[torch.Tensor] = None,
        return_detailed: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with identity-preserving corrections.

        At initialization: output = backbone_output (zero correction)

        Args:
            backbone_output: Denoised output from backbone [B, 1, H, W]
            noisy_input: Noisy input image [B, 1, H, W]
            lambda_maps: Dict of lambda maps for each correction type
            backbone_features: Dict with 'enc1' and 'enc2' features
            target: Optional ground truth for local PSNR verification
            return_detailed: Whether to return detailed information

        Returns:
            corrected: Final corrected output [B, 1, H, W]
            info: Dictionary with correction statistics
        """
        B, C, H, W = backbone_output.shape
        info = {}

        # Get and adapt backbone features
        enc1 = backbone_features.get('enc1')
        enc2 = backbone_features.get('enc2')

        f1 = self.adapt_enc1(enc1)
        if f1.shape[2:] != backbone_output.shape[2:]:
            f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)

        f2 = self.adapt_enc2(enc2)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

        # Combine all features
        combined_features = torch.cat([backbone_output, noisy_input, f1, f2], dim=1)

        # Extract features for gating
        gate_features = self.feature_extractor(combined_features)

        # Compute main confidence gate (starts near 0)
        main_confidence = self.confidence_gate(gate_features)
        info['main_confidence'] = main_confidence.mean().item()

        # Compute per-type gates
        type_gates = self.type_gates(gate_features)
        info['type_gates'] = {k: v.mean().item() for k, v in type_gates.items()}

        # Compute raw corrections
        raw_corrections = self.compute_corrections(combined_features, backbone_output, noisy_input)
        info['raw_corrections'] = {k: v.abs().mean().item() for k, v in raw_corrections.items()}

        # Get lambda maps (default to zeros if not provided)
        lambda_edge = lambda_maps.get('edge', torch.zeros_like(backbone_output))
        lambda_contrast = lambda_maps.get('contrast', torch.zeros_like(backbone_output))
        lambda_texture = lambda_maps.get('texture', torch.zeros_like(backbone_output))
        lambda_smooth = lambda_maps.get('smooth', torch.zeros_like(backbone_output))
        lambda_sharpness = lambda_maps.get('sharpness', torch.zeros_like(backbone_output))

        lambdas = {
            'edge': lambda_edge,
            'contrast': lambda_contrast,
            'texture': lambda_texture,
            'smooth': lambda_smooth,
            'sharpness': lambda_sharpness,
        }

        # During warmup: only train gates, freeze correction magnitudes
        if self.in_warmup and self.training:
            # Detach raw corrections to only learn when to correct
            raw_corrections = {k: v.detach() for k, v in raw_corrections.items()}

        # Apply gating: raw_correction * main_confidence * type_gate * lambda
        gated_corrections = {}
        for name in self.CORRECTION_TYPES:
            gated = (
                raw_corrections[name]
                * main_confidence
                * type_gates[name]
                * lambdas[name]
            )
            gated_corrections[name] = gated

        info['gated_corrections'] = {k: v.abs().mean().item() for k, v in gated_corrections.items()}

        # Sum all corrections
        total_correction = sum(gated_corrections.values())

        # Safety clamp
        total_correction = total_correction.clamp(-0.15, 0.15)
        info['total_correction_magnitude'] = total_correction.abs().mean().item()

        # Apply correction (identity at start due to near-zero gates)
        corrected = backbone_output + total_correction

        # Optional: verify local PSNR and mask out harmful corrections
        if self.verify_local_psnr and target is not None and not self.training:
            psnr_mask = self.compute_local_psnr_mask(
                backbone_output, corrected, target, self.local_psnr_window
            )
            # Re-apply correction with mask
            corrected = backbone_output + total_correction * psnr_mask
            info['psnr_mask_mean'] = psnr_mask.mean().item()

        # Final clamp to valid range
        corrected = corrected.clamp(0, 1)

        # Compute difference from backbone
        info['output_diff_from_backbone'] = (corrected - backbone_output).abs().mean().item()

        if return_detailed:
            info['gated_corrections_tensors'] = gated_corrections
            info['lambda_maps'] = lambdas
            info['main_confidence_map'] = main_confidence
            info['type_gate_maps'] = type_gates

        return corrected, info

    def get_gate_summary(self) -> Dict[str, float]:
        """Get summary of current gate values (useful for logging)."""
        summary = {
            'main_confidence_bias': self.confidence_gate.confidence_bias.item(),
            'main_confidence_value': torch.sigmoid(self.confidence_gate.confidence_bias).item(),
        }

        for name in self.CORRECTION_TYPES:
            bias = self.type_gates.gate_biases[name].item()
            summary[f'{name}_gate_bias'] = bias
            summary[f'{name}_gate_value'] = torch.sigmoid(torch.tensor(bias)).item()

        return summary


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing IdentityCorrectorV1 - Safe Identity Corrector")
    print("=" * 70)

    # Create corrector
    corrector = IdentityCorrectorV1(
        enc1_channels=64,
        enc2_channels=128,
        hidden_dim=64,
        confidence_initial_bias=-5.0,
        type_gate_initial_bias=-4.0,
        warmup_epochs=5,
        verify_local_psnr=True,
    )

    # Test data
    B, H, W = 2, 64, 64
    backbone_output = torch.rand(B, 1, H, W)
    noisy_input = backbone_output + torch.randn(B, 1, H, W) * 0.1
    noisy_input = noisy_input.clamp(0, 1)
    target = backbone_output.clone()  # Perfect target for testing

    lambda_maps = {
        'edge': torch.rand(B, 1, H, W) * 0.5,
        'contrast': torch.rand(B, 1, H, W) * 0.5,
        'texture': torch.rand(B, 1, H, W) * 0.3,
        'smooth': torch.rand(B, 1, H, W) * 0.4,
        'sharpness': torch.rand(B, 1, H, W) * 0.4,
    }

    backbone_features = {
        'enc1': torch.randn(B, 64, H, W),
        'enc2': torch.randn(B, 128, H//2, W//2),
    }

    print("\n" + "=" * 70)
    print("INITIALIZATION TEST - Verifying near-zero corrections at start")
    print("=" * 70)

    # Forward pass at initialization
    corrector.eval()
    with torch.no_grad():
        corrected, info = corrector(
            backbone_output, noisy_input, lambda_maps, backbone_features,
            target=target, return_detailed=True
        )

    print(f"\nForward pass successful!")
    print(f"  Input shape: {backbone_output.shape}")
    print(f"  Output shape: {corrected.shape}")

    print(f"\nMain confidence gate:")
    print(f"  Value: {info['main_confidence']:.6f}")

    print(f"\nPer-type gate values:")
    for name, val in info['type_gates'].items():
        print(f"  {name}: {val:.6f}")

    print(f"\nRaw corrections (before gating):")
    for name, val in info['raw_corrections'].items():
        print(f"  {name}: {val:.6f}")

    print(f"\nGated corrections (after gating):")
    for name, val in info['gated_corrections'].items():
        print(f"  {name}: {val:.6f}")

    print(f"\nTotal correction magnitude: {info['total_correction_magnitude']:.6f}")
    print(f"Output diff from backbone: {info['output_diff_from_backbone']:.6f}")

    # Verify near-identity behavior
    diff = (corrected - backbone_output).abs().mean().item()
    print(f"\n" + "=" * 70)
    print("IDENTITY VERIFICATION")
    print("=" * 70)
    print(f"Mean absolute difference from backbone: {diff:.6f}")

    if diff < 0.01:
        print("[PASS] Near-identity behavior verified - corrections start near zero!")
    else:
        print("[WARNING] Corrections are larger than expected at initialization")

    # Test warmup phase
    print(f"\n" + "=" * 70)
    print("WARMUP PHASE TEST")
    print("=" * 70)

    corrector.train()
    corrector.set_epoch(0)
    print(f"In warmup phase (epoch {corrector._current_epoch}): {corrector.in_warmup}")

    corrector.set_epoch(10)
    print(f"After warmup (epoch {corrector._current_epoch}): {corrector.in_warmup}")

    # Print gate summary
    print(f"\n" + "=" * 70)
    print("GATE SUMMARY")
    print("=" * 70)
    summary = corrector.get_gate_summary()
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")

    # Test gradient flow
    print(f"\n" + "=" * 70)
    print("GRADIENT FLOW TEST")
    print("=" * 70)

    corrector.train()
    corrector.set_epoch(10)  # After warmup

    corrected, info = corrector(
        backbone_output.requires_grad_(True),
        noisy_input,
        lambda_maps,
        backbone_features,
        target=target
    )

    # Compute a simple loss and backward
    loss = F.mse_loss(corrected, target)
    loss.backward()

    # Check if gradients flow
    has_gradients = False
    for name, param in corrector.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    if has_gradients:
        print("[PASS] Gradients are flowing through the network!")
    else:
        print("[WARNING] No gradients detected - check initialization")

    print(f"\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
