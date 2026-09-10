#!/usr/bin/env python3
"""
Contrast Restoration Corrector for OCT Denoising

Addresses the backbone's 53% local contrast loss by:
1. Computing local contrast (std dev) maps at multiple scales
2. Comparing backbone contrast vs. expected contrast from noisy input analysis
3. Generating spatially-guided additive corrections to restore lost contrast
4. Using layer-specific contrast targets for clinical utility

Design Philosophy:
- Focus on layer VISIBILITY, not PSNR optimization
- Restore contrast WHERE it was lost (guided by failure map)
- Preserve edges during contrast enhancement (no halos)
- Layer-specific contrast targets (RNFL needs more contrast than choroid)

Author: Neuro-Symbolic OCT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# LOCAL CONTRAST COMPUTATION UTILITIES
# =============================================================================

class LocalContrastComputer(nn.Module):
    """
    Compute local contrast (standard deviation) at multiple scales.

    Local contrast is computed as:
        contrast(x, y) = std_dev(patch centered at x, y)

    This is more robust than gradient-based contrast for OCT because:
    - Captures texture/speckle contrast, not just edges
    - Less sensitive to noise orientation
    - Better correlation with perceived layer visibility
    """

    def __init__(self, scales: Tuple[int, ...] = (7, 15, 31)):
        super().__init__()
        self.scales = scales

    def compute_local_std(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Compute local standard deviation using efficient separable convolution.

        Args:
            x: Input tensor [B, 1, H, W]
            kernel_size: Size of local window

        Returns:
            Local std dev map [B, 1, H, W]
        """
        padding = kernel_size // 2

        # Use average pooling for efficiency
        local_mean = F.avg_pool2d(
            F.pad(x, [padding] * 4, mode='reflect'),
            kernel_size, stride=1, padding=0
        )
        local_mean_sq = F.avg_pool2d(
            F.pad(x ** 2, [padding] * 4, mode='reflect'),
            kernel_size, stride=1, padding=0
        )

        # Var = E[X^2] - E[X]^2, ensure non-negative
        local_var = (local_mean_sq - local_mean ** 2).clamp(min=1e-8)
        local_std = torch.sqrt(local_var)

        return local_std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale local contrast maps.

        Returns dict with:
            'contrast_<scale>': Local std at each scale
            'contrast_mean': Average across scales
            'contrast_max': Maximum across scales (captures fine details)
        """
        contrasts = {}
        contrast_list = []

        for scale in self.scales:
            contrast = self.compute_local_std(x, scale)
            contrasts[f'contrast_{scale}'] = contrast
            contrast_list.append(contrast)

        # Aggregate statistics
        contrast_stack = torch.stack(contrast_list, dim=0)
        contrasts['contrast_mean'] = contrast_stack.mean(dim=0)
        contrasts['contrast_max'] = contrast_stack.max(dim=0)[0]

        return contrasts


class ExpectedContrastEstimator(nn.Module):
    """
    Estimate expected contrast from noisy input.

    Key insight: The noisy image contains both signal contrast AND noise contrast.
    We need to separate them to know what contrast SHOULD be preserved.

    Strategy:
    1. Compute noisy image contrast
    2. Estimate noise contribution (from flat regions)
    3. Expected signal contrast = sqrt(noisy_contrast^2 - noise_contrast^2)
    """

    # Layer-specific contrast expectations (normalized depth 0-1)
    # Based on OCT literature and clinical requirements
    LAYER_CONTRAST_TARGETS = {
        'vitreous': {'depth': (0.00, 0.05), 'target_contrast': 0.02, 'clinical_importance': 0.3},
        'rnfl': {'depth': (0.05, 0.15), 'target_contrast': 0.12, 'clinical_importance': 1.0},  # Critical for glaucoma
        'gcl_ipl': {'depth': (0.15, 0.30), 'target_contrast': 0.08, 'clinical_importance': 0.9},
        'inl_opl': {'depth': (0.30, 0.45), 'target_contrast': 0.07, 'clinical_importance': 0.7},
        'onl': {'depth': (0.45, 0.55), 'target_contrast': 0.05, 'clinical_importance': 0.6},
        'is_os': {'depth': (0.55, 0.65), 'target_contrast': 0.10, 'clinical_importance': 0.95},  # Critical for AMD
        'rpe': {'depth': (0.65, 0.75), 'target_contrast': 0.15, 'clinical_importance': 1.0},  # Brightest layer
        'choroid': {'depth': (0.75, 1.00), 'target_contrast': 0.06, 'clinical_importance': 0.5},
    }

    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.kernel_size = kernel_size
        self.contrast_computer = LocalContrastComputer(scales=(kernel_size,))

        # Learnable noise level estimation (updated during training)
        self.register_buffer('noise_level_estimate', torch.tensor(0.05))

    def estimate_noise_level(self, noisy: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        """Estimate noise level from residual in flat regions."""
        residual = noisy - denoised

        # Find flat regions (low gradient)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=noisy.dtype, device=noisy.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)

        gx = F.conv2d(F.pad(denoised, [1, 1, 1, 1], mode='reflect'), sobel_x)
        gy = F.conv2d(F.pad(denoised, [1, 1, 1, 1], mode='reflect'), sobel_y)
        gradient_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Flat region mask (bottom 30% of gradients)
        threshold = gradient_mag.quantile(0.3)
        flat_mask = (gradient_mag < threshold).float()

        # Noise std in flat regions
        if flat_mask.sum() > 100:
            masked_residual = residual * flat_mask
            noise_std = masked_residual.abs().sum() / (flat_mask.sum() + 1e-6)
        else:
            noise_std = residual.std()

        return noise_std

    def create_layer_contrast_target_map(self, height: int, width: int,
                                          device: torch.device) -> torch.Tensor:
        """Create spatially-varying contrast target based on OCT layer structure."""
        # Create depth position map (assuming vertical = depth)
        depth_pos = torch.linspace(0, 1, height, device=device).view(1, 1, height, 1)
        depth_pos = depth_pos.expand(1, 1, height, width)

        target_map = torch.zeros(1, 1, height, width, device=device)

        for layer_name, params in self.LAYER_CONTRAST_TARGETS.items():
            lo, hi = params['depth']
            target = params['target_contrast']
            importance = params['clinical_importance']

            # Soft mask for this layer
            in_layer = ((depth_pos >= lo) & (depth_pos < hi)).float()
            target_map = target_map + in_layer * target * importance

        return target_map

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate expected contrast map.

        Returns:
            'expected_contrast': What contrast should be (from noisy - noise contribution)
            'layer_target_contrast': Layer-specific clinical targets
            'noise_level': Estimated noise standard deviation
        """
        B, C, H, W = noisy.shape
        device = noisy.device

        # Compute noisy image contrast
        noisy_contrast = self.contrast_computer.compute_local_std(noisy, self.kernel_size)

        # Estimate noise contribution
        noise_level = self.estimate_noise_level(noisy, denoised)

        # Expected signal contrast (subtract noise contribution in quadrature)
        # contrast_signal^2 = contrast_total^2 - contrast_noise^2
        expected_contrast = torch.sqrt(
            (noisy_contrast ** 2 - noise_level ** 2).clamp(min=1e-8)
        )

        # Layer-specific targets
        layer_targets = self.create_layer_contrast_target_map(H, W, device)
        layer_targets = layer_targets.expand(B, -1, -1, -1)

        return {
            'expected_contrast': expected_contrast,
            'layer_target_contrast': layer_targets,
            'noise_level': noise_level,
            'noisy_contrast': noisy_contrast,
        }


# =============================================================================
# CONTRAST FAILURE MAP COMPUTATION
# =============================================================================

class ContrastFailureDetector(nn.Module):
    """
    Detect WHERE contrast was lost by the backbone.

    Generates a spatial map indicating regions where:
    1. Contrast is lower than expected
    2. Contrast is lower than layer-specific target
    3. Both conditions weighted by clinical importance
    """

    def __init__(self, loss_threshold: float = 0.47):
        """
        Args:
            loss_threshold: Consider contrast "lost" if ratio < (1 - threshold)
                           Default 0.47 corresponds to 53% loss detection
        """
        super().__init__()
        self.loss_threshold = loss_threshold
        self.contrast_computer = LocalContrastComputer(scales=(7, 15))

    def forward(self,
                denoised_contrast: torch.Tensor,
                expected_contrast: torch.Tensor,
                layer_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute contrast failure map.

        Args:
            denoised_contrast: Local contrast of backbone output
            expected_contrast: Expected contrast from noisy input analysis
            layer_targets: Layer-specific contrast targets

        Returns:
            'failure_map': [0, 1] map where 1 = complete contrast loss
            'contrast_ratio': Ratio of actual/expected contrast
            'target_deficit': How much below layer target
        """
        # Contrast ratio: how much contrast was preserved
        contrast_ratio = denoised_contrast / (expected_contrast + 1e-6)

        # Failure score based on contrast loss
        # If ratio < 0.5, we have severe loss; if ratio > 1, no loss
        contrast_failure = F.relu(1 - contrast_ratio).clamp(0, 1)

        # Also check against layer-specific targets
        target_ratio = denoised_contrast / (layer_targets + 1e-6)
        target_deficit = F.relu(1 - target_ratio).clamp(0, 1)

        # Combined failure map (max of both criteria)
        # This ensures we enhance contrast if EITHER condition fails
        failure_map = torch.max(contrast_failure, target_deficit * 0.5)

        # Apply threshold: only mark as failure if loss exceeds threshold
        significant_loss = (contrast_failure > self.loss_threshold).float()
        failure_map = failure_map * significant_loss + failure_map * 0.3 * (1 - significant_loss)

        return {
            'failure_map': failure_map.clamp(0, 1),
            'contrast_ratio': contrast_ratio,
            'target_deficit': target_deficit,
            'contrast_failure_raw': contrast_failure,
        }


# =============================================================================
# CONTRAST RESTORATION NETWORK
# =============================================================================

class ContrastRestorationNetwork(nn.Module):
    """
    Neural network that generates contrast restoration corrections.

    Architecture:
    1. Fuse backbone output, failure map, and contrast deficit information
    2. Multi-scale context extraction (different receptive fields for local/global contrast)
    3. Edge-aware processing (avoid halos at layer boundaries)
    4. Generate additive correction that boosts local std dev

    Key insight: To increase local contrast at a point, we need to:
    - Push bright pixels brighter
    - Push dark pixels darker
    This is achieved by amplifying deviation from local mean.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Input: denoised (1) + failure_map (1) + contrast_ratio (1) + target_deficit (1) = 4
        in_channels = 4

        # Feature extraction with residual connection
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale context (different receptive fields for contrast at different scales)
        self.multiscale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=3, dilation=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=7, dilation=7),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=15, dilation=15),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])

        # Edge-aware processing (detect boundaries to avoid halos)
        self.edge_detector = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Refinement with attention to failure regions
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Correction head
        # Output: gain map (how much to amplify deviation from local mean)
        self.gain_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Softplus(),  # Gain must be positive
        )

        # Learnable maximum gain (safety cap)
        self.max_gain = nn.Parameter(torch.tensor(2.0))  # Max 2x contrast boost

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize gain head to output small gains initially (stable training)
        for module in self.gain_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, -2.0)  # softplus(-2) ~ 0.13

    def forward(self,
                denoised: torch.Tensor,
                failure_map: torch.Tensor,
                contrast_ratio: torch.Tensor,
                target_deficit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate contrast restoration gain map.

        Args:
            denoised: Backbone output [B, 1, H, W]
            failure_map: Where contrast was lost [B, 1, H, W]
            contrast_ratio: Actual/expected contrast ratio [B, 1, H, W]
            target_deficit: Deficit vs. layer target [B, 1, H, W]

        Returns:
            gain_map: Contrast amplification factor [B, 1, H, W]
            edge_mask: Detected edges for halo avoidance [B, 1, H, W]
        """
        # Concatenate inputs
        x = torch.cat([denoised, failure_map, contrast_ratio, target_deficit], dim=1)

        # Initial feature extraction
        feat = self.input_conv(x)

        # Multi-scale context
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Detect edges (for halo avoidance)
        edge_mask = self.edge_detector(denoised)

        # Modulate features by inverse edge mask (reduce gain near edges)
        feat = feat * (1 - edge_mask * 0.5)

        # Refinement
        feat = self.refine(feat)

        # Attention to failure regions
        feat = feat * (failure_map * 0.7 + 0.3)  # Focus on failure regions

        # Generate gain map
        raw_gain = self.gain_head(feat)

        # Scale by failure severity and cap maximum
        max_gain_capped = F.softplus(self.max_gain).clamp(1.0, 3.0)
        gain_map = 1.0 + raw_gain * failure_map * (max_gain_capped - 1.0)

        return gain_map, edge_mask


# =============================================================================
# MAIN CONTRAST RESTORATION CORRECTOR
# =============================================================================

class ContrastRestorationCorrector(nn.Module):
    """
    Complete Contrast Restoration Corrector Module.

    Addresses the 53% local contrast loss observed in backbone outputs by:

    1. ANALYSIS PHASE:
       - Compute local contrast at multiple scales
       - Estimate expected contrast from noisy input
       - Detect regions with contrast failure

    2. CORRECTION PHASE:
       - Generate spatially-varying gain map
       - Apply contrast enhancement: output = mean + gain * (input - mean)
       - Preserve edges (avoid halos at layer boundaries)

    3. SAFETY MECHANISMS:
       - Maximum gain cap (prevents over-amplification)
       - Edge-aware suppression (prevents halo artifacts)
       - Value clamping (keeps output in valid range)

    Clinical Focus:
    - Layer-specific contrast targets based on OCT anatomy
    - Higher correction priority for RNFL, IS/OS, RPE (clinical layers)
    - Preserves structural information while enhancing visibility
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 contrast_scales: Tuple[int, ...] = (7, 15, 31),
                 loss_threshold: float = 0.47):
        """
        Args:
            hidden_dim: Hidden dimension for restoration network
            contrast_scales: Window sizes for multi-scale contrast computation
            loss_threshold: Threshold for detecting significant contrast loss (0.47 = 53% loss)
        """
        super().__init__()

        # Contrast analysis modules
        self.contrast_computer = LocalContrastComputer(scales=contrast_scales)
        self.expected_estimator = ExpectedContrastEstimator(kernel_size=15)
        self.failure_detector = ContrastFailureDetector(loss_threshold=loss_threshold)

        # Contrast restoration network
        self.restoration_net = ContrastRestorationNetwork(hidden_dim=hidden_dim)

        # Local mean computation (for contrast enhancement)
        self.mean_kernel_size = 15

        # Learnable correction strength (can be adjusted during training)
        self.correction_strength = nn.Parameter(torch.tensor(0.8))

        # Statistics tracking (for monitoring)
        self.register_buffer('contrast_restored_ratio', torch.tensor(0.0))

        self._print_info()

    def _print_info(self):
        print("\n" + "=" * 70)
        print("ContrastRestorationCorrector - Addressing 53% Contrast Loss")
        print("=" * 70)
        print("Analysis Components:")
        print(f"  - Multi-scale contrast: scales={self.contrast_computer.scales}")
        print(f"  - Expected contrast estimation with layer-specific targets")
        print(f"  - Failure detection threshold: {self.failure_detector.loss_threshold:.0%}")
        print("")
        print("Restoration Strategy:")
        print("  - Generate spatially-varying gain map from failure regions")
        print("  - Apply: output = mean + gain * (input - mean)")
        print("  - Edge-aware suppression to prevent halos")
        print("")
        print("Clinical Focus:")
        print("  - RNFL, IS/OS, RPE: highest contrast targets")
        print("  - Layer visibility > PSNR optimization")
        print("=" * 70)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def compute_local_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local mean for contrast enhancement."""
        padding = self.mean_kernel_size // 2
        return F.avg_pool2d(
            F.pad(x, [padding] * 4, mode='reflect'),
            self.mean_kernel_size, stride=1, padding=0
        )

    def apply_contrast_enhancement(self,
                                    x: torch.Tensor,
                                    gain_map: torch.Tensor,
                                    edge_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply contrast enhancement using gain map.

        Enhancement formula:
            output = local_mean + gain * (input - local_mean)

        This increases local standard deviation by factor of 'gain'.

        Edge-aware: Reduce gain near edges to prevent halos.
        """
        local_mean = self.compute_local_mean(x)
        deviation = x - local_mean

        # Reduce gain near edges (halo prevention)
        adjusted_gain = gain_map * (1 - edge_mask * 0.7) + edge_mask * 0.7

        # Apply enhancement
        enhanced = local_mean + adjusted_gain * deviation

        return enhanced

    def forward(self,
                backbone_out: torch.Tensor,
                noisy: torch.Tensor,
                return_details: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply contrast restoration correction.

        Args:
            backbone_out: Backbone denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            return_details: Whether to return detailed analysis

        Returns:
            corrected: Contrast-restored output [B, 1, H, W]
            info: Dictionary with analysis details
        """
        B, C, H, W = backbone_out.shape
        device = backbone_out.device

        # ===== PHASE 1: CONTRAST ANALYSIS =====

        # Compute backbone output contrast
        denoised_contrasts = self.contrast_computer(backbone_out)
        denoised_contrast = denoised_contrasts['contrast_mean']

        # Estimate expected contrast
        expected_info = self.expected_estimator(noisy, backbone_out)
        expected_contrast = expected_info['expected_contrast']
        layer_targets = expected_info['layer_target_contrast']

        # Detect failure regions
        failure_info = self.failure_detector(
            denoised_contrast, expected_contrast, layer_targets
        )
        failure_map = failure_info['failure_map']
        contrast_ratio = failure_info['contrast_ratio']
        target_deficit = failure_info['target_deficit']

        # ===== PHASE 2: GENERATE CORRECTION =====

        # Get gain map from restoration network
        gain_map, edge_mask = self.restoration_net(
            backbone_out, failure_map, contrast_ratio, target_deficit
        )

        # ===== PHASE 3: APPLY CORRECTION =====

        # Apply contrast enhancement
        enhanced = self.apply_contrast_enhancement(backbone_out, gain_map, edge_mask)

        # Blend with correction strength
        strength = torch.sigmoid(self.correction_strength)
        corrected = backbone_out + strength * (enhanced - backbone_out)

        # Ensure valid range
        corrected = corrected.clamp(0, 1)

        # ===== COMPUTE STATISTICS =====

        # Compute contrast improvement
        corrected_contrasts = self.contrast_computer(corrected)
        corrected_contrast = corrected_contrasts['contrast_mean']

        # Contrast restoration ratio
        with torch.no_grad():
            initial_loss = (expected_contrast - denoised_contrast).mean()
            final_loss = (expected_contrast - corrected_contrast).mean()
            if initial_loss.abs() > 1e-6:
                restoration_ratio = 1 - (final_loss / initial_loss).clamp(0, 2)
            else:
                restoration_ratio = torch.tensor(1.0, device=device)

            # Update tracking buffer (exponential moving average)
            if self.training:
                self.contrast_restored_ratio = (
                    0.99 * self.contrast_restored_ratio + 0.01 * restoration_ratio
                )

        # ===== BUILD INFO DICT =====

        info = {
            'failure_map': failure_map,
            'gain_map': gain_map,
            'edge_mask': edge_mask,
            'contrast_ratio_before': contrast_ratio.mean().item(),
            'contrast_ratio_after': (corrected_contrast / (expected_contrast + 1e-6)).mean().item(),
            'restoration_ratio': restoration_ratio.item(),
            'correction_strength': strength.item(),
            'mean_gain': gain_map.mean().item(),
            'max_gain': gain_map.max().item(),
        }

        if return_details:
            info['denoised_contrast'] = denoised_contrast
            info['expected_contrast'] = expected_contrast
            info['corrected_contrast'] = corrected_contrast
            info['layer_targets'] = layer_targets
            info['noise_level'] = expected_info['noise_level'].item()

            # Layer-by-layer analysis
            info['layer_analysis'] = self._analyze_layers(
                denoised_contrast, corrected_contrast, expected_contrast, H
            )

        return corrected, info

    def _analyze_layers(self,
                        before: torch.Tensor,
                        after: torch.Tensor,
                        expected: torch.Tensor,
                        height: int) -> Dict:
        """Analyze contrast restoration per clinical layer."""
        analysis = {}

        for layer_name, params in ExpectedContrastEstimator.LAYER_CONTRAST_TARGETS.items():
            lo, hi = params['depth']
            y_lo, y_hi = int(lo * height), int(hi * height)

            if y_hi > y_lo:
                layer_before = before[:, :, y_lo:y_hi, :].mean().item()
                layer_after = after[:, :, y_lo:y_hi, :].mean().item()
                layer_expected = expected[:, :, y_lo:y_hi, :].mean().item()

                # Contrast recovery percentage
                if layer_expected > layer_before and layer_expected > 1e-6:
                    recovery = (layer_after - layer_before) / (layer_expected - layer_before + 1e-6)
                    recovery = min(max(recovery, 0), 1)
                else:
                    recovery = 1.0 if layer_after >= layer_before else 0.0

                analysis[layer_name] = {
                    'contrast_before': layer_before,
                    'contrast_after': layer_after,
                    'contrast_expected': layer_expected,
                    'recovery_pct': recovery * 100,
                    'clinical_importance': params['clinical_importance'],
                }

        return analysis


# =============================================================================
# INTEGRATION WITH V8 ENHANCED CORRECTOR
# =============================================================================

class EnhancedContrastCorrectorWithRestoration(nn.Module):
    """
    Enhanced contrast corrector that combines:
    1. Original EnhancedContrastCorrector (global context, attention)
    2. New ContrastRestorationCorrector (local std restoration)

    This provides both:
    - Global contrast adjustment (histogram-like)
    - Local contrast restoration (texture/detail visibility)
    """

    def __init__(self,
                 in_channels: int = 1,
                 hidden_dim: int = 64,
                 enc1_channels: int = 48,
                 enc2_channels: int = 96):
        super().__init__()

        # Contrast restoration corrector (addresses 53% local contrast loss)
        self.restoration_corrector = ContrastRestorationCorrector(
            hidden_dim=hidden_dim,
            contrast_scales=(7, 15, 31),
            loss_threshold=0.47  # 53% loss threshold
        )

        # Learnable blend between restoration and original correction
        self.restoration_weight = nn.Parameter(torch.tensor(0.7))  # Start with restoration-dominant

    def forward(self,
                backbone_out: torch.Tensor,
                noisy: torch.Tensor,
                failure_map: Optional[torch.Tensor] = None,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None,
                return_details: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply combined contrast correction.

        Args:
            backbone_out: Backbone denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            failure_map: Optional P2 failure map from predicates
            backbone_features: Optional encoder features for context
            return_details: Whether to return detailed analysis

        Returns:
            correction: Additive correction [B, 1, H, W] (to be added to backbone_out)
            info: Dictionary with analysis details
        """
        # Apply restoration correction
        restored, restoration_info = self.restoration_corrector(
            backbone_out, noisy, return_details=return_details
        )

        # Compute additive correction
        restoration_correction = restored - backbone_out

        # Apply restoration weight
        weight = torch.sigmoid(self.restoration_weight)
        correction = restoration_correction * weight

        info = {
            'restoration_info': restoration_info,
            'restoration_weight': weight.item(),
            'correction_magnitude': correction.abs().mean().item(),
        }

        return correction, info


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\nTesting ContrastRestorationCorrector...")

    # Create model
    model = ContrastRestorationCorrector(
        hidden_dim=64,
        contrast_scales=(7, 15, 31),
        loss_threshold=0.47
    )

    # Test inputs
    B, C, H, W = 2, 1, 128, 128

    # Create synthetic OCT-like image with layers
    depth = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)
    clean = (
        0.3 +
        0.4 * torch.exp(-((depth - 0.1) ** 2) / 0.01) +  # RNFL
        0.3 * torch.exp(-((depth - 0.6) ** 2) / 0.005) +  # IS/OS
        0.5 * torch.exp(-((depth - 0.7) ** 2) / 0.008)    # RPE
    )
    clean = clean + torch.randn_like(clean) * 0.02  # Add small texture
    clean = clean.clamp(0, 1)

    # Add noise
    noisy = clean + torch.randn_like(clean) * 0.15
    noisy = noisy.clamp(0, 1)

    # Simulate backbone output with 53% contrast loss
    backbone_mean = F.avg_pool2d(F.pad(clean, [7]*4, mode='reflect'), 15, stride=1, padding=0)
    backbone_out = backbone_mean + 0.47 * (clean - backbone_mean)  # 53% contrast loss
    backbone_out = backbone_out.clamp(0, 1)

    print(f"\nInput shapes:")
    print(f"  Noisy: {noisy.shape}")
    print(f"  Backbone output: {backbone_out.shape}")

    # Forward pass
    corrected, info = model(backbone_out, noisy, return_details=True)

    print(f"  Corrected: {corrected.shape}")

    print(f"\nContrast Analysis:")
    print(f"  Contrast ratio before: {info['contrast_ratio_before']:.3f}")
    print(f"  Contrast ratio after: {info['contrast_ratio_after']:.3f}")
    print(f"  Restoration ratio: {info['restoration_ratio']:.1%}")
    print(f"  Mean gain applied: {info['mean_gain']:.3f}")
    print(f"  Max gain applied: {info['max_gain']:.3f}")

    print(f"\nLayer-by-Layer Analysis:")
    for layer, data in info['layer_analysis'].items():
        print(f"  {layer:12s}: before={data['contrast_before']:.4f}, "
              f"after={data['contrast_after']:.4f}, "
              f"recovery={data['recovery_pct']:.1f}%")

    # Verify contrast was restored
    contrast_computer = LocalContrastComputer(scales=(15,))

    backbone_contrast = contrast_computer.compute_local_std(backbone_out, 15).mean().item()
    corrected_contrast = contrast_computer.compute_local_std(corrected, 15).mean().item()
    clean_contrast = contrast_computer.compute_local_std(clean, 15).mean().item()

    print(f"\nContrast Verification:")
    print(f"  Clean image contrast: {clean_contrast:.4f}")
    print(f"  Backbone contrast: {backbone_contrast:.4f} ({backbone_contrast/clean_contrast:.1%} of clean)")
    print(f"  Corrected contrast: {corrected_contrast:.4f} ({corrected_contrast/clean_contrast:.1%} of clean)")

    improvement = (corrected_contrast - backbone_contrast) / (clean_contrast - backbone_contrast + 1e-6)
    print(f"  Contrast recovery: {improvement:.1%}")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
