#!/usr/bin/env python3
"""
Pathology Preservation Module for OCT Denoising

This module detects and preserves pathological features (drusen, fluid, atrophy)
that typical denoising over-smooths. Pathology detection is UNSUPERVISED -
it uses statistical features without requiring pathology labels.

Key Components:
1. PathologyDetector - Detects potential pathological regions using unsupervised features
2. PathologyPreservationModule - Creates pathology map and applies correction gating
3. PathologyPreservationLoss - Loss function to preserve pathology features

Pathology Detection Features (all unsupervised):
- Abnormal intensity patterns (very bright/dark spots)
- Irregular texture (high local variance)
- Disrupted layer structure (breaks in horizontal continuity)
- Fluid-like patterns (dark pockets with smooth boundaries)

Author: Neuro-Symbolic OCT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# PATHOLOGY DETECTOR - Unsupervised Feature Extraction
# =============================================================================

class PathologyDetector(nn.Module):
    """
    Unsupervised pathology detector using statistical features.

    Detects potential pathological regions without ground truth labels by
    identifying statistical anomalies in OCT images:

    1. Abnormal Intensity: Very bright (drusen) or dark (fluid/atrophy) spots
    2. Irregular Texture: High local variance indicates pathological changes
    3. Layer Disruption: Breaks in horizontal layer continuity
    4. Fluid Patterns: Dark regions with smooth internal texture but sharp boundaries

    All features are computed in a fully differentiable manner.
    """

    def __init__(self,
                 intensity_threshold_low: float = 0.15,
                 intensity_threshold_high: float = 0.85,
                 variance_multiplier: float = 2.0,
                 layer_kernel_size: int = 15,
                 fluid_dark_threshold: float = 0.3):
        """
        Args:
            intensity_threshold_low: Threshold for dark anomalies (below this is abnormal)
            intensity_threshold_high: Threshold for bright anomalies (above this is abnormal)
            variance_multiplier: Multiplier for variance threshold (mean + multiplier * std)
            layer_kernel_size: Horizontal kernel size for layer continuity analysis
            fluid_dark_threshold: Threshold for fluid detection (dark regions)
        """
        super().__init__()

        self.intensity_threshold_low = intensity_threshold_low
        self.intensity_threshold_high = intensity_threshold_high
        self.variance_multiplier = variance_multiplier
        self.layer_kernel_size = layer_kernel_size
        self.fluid_dark_threshold = fluid_dark_threshold

        # Local statistics computation (7x7 window)
        self.local_pool_size = 7
        self.local_pool = nn.AvgPool2d(
            kernel_size=self.local_pool_size,
            stride=1,
            padding=self.local_pool_size // 2
        )

        # Horizontal kernel for layer continuity (1 x kernel_size)
        # Detects breaks in horizontal layer structure
        self.register_buffer(
            'horizontal_kernel',
            torch.ones(1, 1, 1, layer_kernel_size) / layer_kernel_size
        )

        # Sobel filters for gradient computation
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Laplacian for boundary sharpness (fluid detection)
        laplacian = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

        # Learnable combination weights (initialized to equal weights)
        self.feature_weights = nn.Parameter(torch.ones(4) / 4)

    def _compute_local_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute local mean and standard deviation.

        Returns:
            local_mean: [B, 1, H, W]
            local_std: [B, 1, H, W]
        """
        local_mean = self.local_pool(x)
        local_sq_mean = self.local_pool(x ** 2)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=1e-8)
        local_std = torch.sqrt(local_var)
        return local_mean, local_std

    def detect_intensity_anomalies(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect abnormal intensity patterns (very bright/dark spots).

        Drusen appear as bright deposits, fluid/atrophy appear as dark regions.
        Uses soft thresholding for differentiability.

        Returns:
            anomaly_map: [B, 1, H, W] probability map of intensity anomalies
        """
        # Soft thresholding using sigmoid
        # Dark anomalies: sigmoid(-(x - low_thresh) / temperature)
        # Bright anomalies: sigmoid((x - high_thresh) / temperature)
        temperature = 0.05  # Controls sharpness of transition

        dark_anomaly = torch.sigmoid(
            -(x - self.intensity_threshold_low) / temperature
        )
        bright_anomaly = torch.sigmoid(
            (x - self.intensity_threshold_high) / temperature
        )

        # Combine: any intensity anomaly
        intensity_anomaly = dark_anomaly + bright_anomaly
        return intensity_anomaly.clamp(0, 1)

    def detect_texture_irregularity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect irregular texture using local variance analysis.

        Pathological regions often have different texture than normal retinal layers.
        High local variance indicates potential pathology.

        Returns:
            irregularity_map: [B, 1, H, W] probability map of texture irregularity
        """
        local_mean, local_std = self._compute_local_stats(x)

        # Global statistics for adaptive thresholding
        # Use per-image statistics for batch processing
        global_mean = local_std.mean(dim=[2, 3], keepdim=True)
        global_std = local_std.std(dim=[2, 3], keepdim=True) + 1e-8

        # Adaptive threshold: mean + multiplier * std
        threshold = global_mean + self.variance_multiplier * global_std

        # Soft thresholding
        temperature = global_std * 0.5  # Adaptive temperature
        irregularity = torch.sigmoid((local_std - threshold) / temperature)

        return irregularity

    def detect_layer_disruption(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect disrupted layer structure (breaks in horizontal continuity).

        Normal OCT images have smooth horizontal layers. Pathology disrupts
        this continuity, creating vertical gradients and discontinuities.

        Returns:
            disruption_map: [B, 1, H, W] probability map of layer disruption
        """
        # Compute horizontal smoothness (along each row)
        # Low horizontal variance = continuous layer
        # High horizontal variance = disrupted layer

        # Apply horizontal smoothing
        h_kernel = self.horizontal_kernel.to(dtype=x.dtype)
        x_h_smooth = F.conv2d(x, h_kernel, padding=(0, self.layer_kernel_size // 2))

        # Difference from horizontally smoothed version indicates disruption
        h_difference = (x - x_h_smooth).abs()

        # Also compute vertical gradient (layer boundaries should be horizontal)
        # Strong vertical gradients in unusual locations indicate disruption
        grad_y = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1)

        # Combine horizontal disruption with vertical gradient anomalies
        # Normalize and combine
        h_diff_norm = h_difference / (h_difference.mean(dim=[2, 3], keepdim=True) + 1e-8)
        grad_y_norm = grad_y.abs() / (grad_y.abs().mean(dim=[2, 3], keepdim=True) + 1e-8)

        # Areas with both high horizontal disruption AND unusual vertical gradients
        # are likely pathological
        disruption = (h_diff_norm + grad_y_norm) / 2

        # Normalize to [0, 1] using sigmoid
        disruption = torch.sigmoid(disruption - 1.0)  # Centered around 1.0

        return disruption

    def detect_fluid_patterns(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect fluid-like patterns (dark pockets with smooth internal boundaries).

        Fluid in OCT appears as:
        1. Dark regions (low intensity)
        2. Relatively smooth internal texture
        3. Sharp boundaries (high edge magnitude at borders)

        Returns:
            fluid_map: [B, 1, H, W] probability map of fluid-like patterns
        """
        # Feature 1: Dark regions
        temperature = 0.05
        dark_mask = torch.sigmoid(
            -(x - self.fluid_dark_threshold) / temperature
        )

        # Feature 2: Low internal texture (smooth interiors)
        local_mean, local_std = self._compute_local_stats(x)

        # In dark regions, check for smoothness
        global_std_mean = local_std.mean(dim=[2, 3], keepdim=True) + 1e-8
        smooth_interior = torch.sigmoid(
            -(local_std - global_std_mean * 0.5) / (global_std_mean * 0.25)
        )

        # Feature 3: Sharp boundaries (high gradient magnitude)
        grad_x = F.conv2d(x, self.sobel_x.to(dtype=x.dtype), padding=1)
        grad_y = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Normalize gradient magnitude
        grad_mean = grad_mag.mean(dim=[2, 3], keepdim=True) + 1e-8
        sharp_boundary = torch.sigmoid((grad_mag - grad_mean) / grad_mean)

        # Dilate dark mask to include boundary regions
        dark_dilated = F.max_pool2d(dark_mask, kernel_size=5, stride=1, padding=2)

        # Fluid pattern: dark region OR (at boundary AND smooth nearby AND sharp edge)
        # This captures both the dark interior and the sharp boundary
        fluid_interior = dark_mask * smooth_interior
        fluid_boundary = (dark_dilated - dark_mask).clamp(0, 1) * sharp_boundary

        fluid_pattern = (fluid_interior + fluid_boundary).clamp(0, 1)

        return fluid_pattern

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect all pathological features.

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            Dict containing:
                - intensity_anomaly: Abnormal intensity map
                - texture_irregularity: Irregular texture map
                - layer_disruption: Layer disruption map
                - fluid_pattern: Fluid-like pattern map
                - combined: Weighted combination of all features
        """
        # Detect individual features
        intensity = self.detect_intensity_anomalies(x)
        texture = self.detect_texture_irregularity(x)
        layer = self.detect_layer_disruption(x)
        fluid = self.detect_fluid_patterns(x)

        # Combine with learnable weights
        weights = F.softmax(self.feature_weights, dim=0)
        combined = (
            weights[0] * intensity +
            weights[1] * texture +
            weights[2] * layer +
            weights[3] * fluid
        )

        return {
            'intensity_anomaly': intensity,
            'texture_irregularity': texture,
            'layer_disruption': layer,
            'fluid_pattern': fluid,
            'combined': combined,
            'weights': weights,
        }


# =============================================================================
# PATHOLOGY PRESERVATION MODULE
# =============================================================================

class PathologyPreservationModule(nn.Module):
    """
    Main module for pathology-aware denoising correction.

    This module:
    1. Detects potential pathological regions using unsupervised features
    2. Creates a pathology probability map [0, 1]
    3. Provides pathology-aware correction gating (reduce correction in pathology regions)
    4. Extracts pathology features for preservation loss computation

    Key insight: In pathological regions, we want LESS aggressive denoising
    to preserve potentially diagnostic features that might look like "noise"
    to a standard denoiser.
    """

    def __init__(self,
                 hidden_dim: int = 32,
                 use_learned_combination: bool = True,
                 min_gate_value: float = 0.3):
        """
        Args:
            hidden_dim: Hidden dimension for feature refinement network
            use_learned_combination: Whether to learn combination weights
            min_gate_value: Minimum gate value in pathology regions (prevents complete blocking)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.min_gate_value = min_gate_value

        # Pathology detector
        self.detector = PathologyDetector()

        # Feature refinement network (refines raw detection to probability map)
        # Input: 5 channels (4 feature maps + original image)
        self.refine_net = nn.Sequential(
            nn.Conv2d(5, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # Pathology feature extractor (for preservation loss)
        # Extracts features that characterize pathological regions
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Local contrast extractor (pathology often has distinct local contrast)
        self.local_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_pathology_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute pathology probability map.

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            pathology_map: [B, 1, H, W] probability map (1.0 = likely pathology)
            detection_info: Dict with individual detection maps
        """
        # Get raw detections
        detection = self.detector(x)

        # Concatenate for refinement
        features = torch.cat([
            x,
            detection['intensity_anomaly'],
            detection['texture_irregularity'],
            detection['layer_disruption'],
            detection['fluid_pattern'],
        ], dim=1)

        # Refine to final probability map
        pathology_map = self.refine_net(features)

        return pathology_map, detection

    def get_correction_gate(self, pathology_map: torch.Tensor) -> torch.Tensor:
        """
        Compute correction gate from pathology map.

        In pathology regions: low gate value -> reduced correction
        In normal regions: high gate value -> full correction

        Args:
            pathology_map: [B, 1, H, W] pathology probability

        Returns:
            gate: [B, 1, H, W] correction gate (1 = full correction, min_gate = reduced)
        """
        # Gate = 1 - pathology_prob * (1 - min_gate)
        # When pathology_prob = 0: gate = 1 (full correction)
        # When pathology_prob = 1: gate = min_gate (reduced correction)
        gate = 1.0 - pathology_map * (1.0 - self.min_gate_value)
        return gate

    def extract_pathology_features(self, x: torch.Tensor,
                                    pathology_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from pathological regions for preservation loss.

        Args:
            x: Input image [B, 1, H, W]
            pathology_map: [B, 1, H, W] pathology probability

        Returns:
            Dict containing:
                - deep_features: Learned features [B, hidden_dim, H, W]
                - local_contrast: Local standard deviation [B, 1, H, W]
                - local_mean: Local mean intensity [B, 1, H, W]
                - masked_features: Features weighted by pathology map
        """
        # Deep features
        deep_features = self.feature_extractor(x)

        # Local statistics
        local_mean = self.local_pool(x)
        local_sq_mean = self.local_pool(x ** 2)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=1e-8)
        local_std = torch.sqrt(local_var)

        # Mask features by pathology probability
        # This focuses the preservation loss on actual pathology regions
        masked_deep = deep_features * pathology_map
        masked_contrast = local_std * pathology_map
        masked_mean = local_mean * pathology_map

        return {
            'deep_features': deep_features,
            'local_contrast': local_std,
            'local_mean': local_mean,
            'masked_deep_features': masked_deep,
            'masked_contrast': masked_contrast,
            'masked_mean': masked_mean,
            'pathology_map': pathology_map,
        }

    def forward(self, noisy: torch.Tensor,
                backbone_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for pathology preservation.

        Args:
            noisy: Noisy input image [B, 1, H, W]
            backbone_out: Denoised backbone output [B, 1, H, W]

        Returns:
            Dict containing:
                - pathology_map: Pathology probability map
                - correction_gate: Gate for modulating correction strength
                - noisy_pathology_features: Features from noisy image in pathology regions
                - backbone_pathology_features: Features from backbone output in pathology regions
                - detection_info: Individual detection maps
        """
        # Detect pathology on noisy image (we want to find what might be pathology)
        pathology_map, detection_info = self.get_pathology_map(noisy)

        # Compute correction gate
        correction_gate = self.get_correction_gate(pathology_map)

        # Extract features for preservation loss
        noisy_features = self.extract_pathology_features(noisy, pathology_map)
        backbone_features = self.extract_pathology_features(backbone_out, pathology_map)

        return {
            'pathology_map': pathology_map,
            'correction_gate': correction_gate,
            'noisy_pathology_features': noisy_features,
            'backbone_pathology_features': backbone_features,
            'detection_info': detection_info,
        }


# =============================================================================
# PATHOLOGY PRESERVATION LOSS
# =============================================================================

class PathologyPreservationLoss(nn.Module):
    """
    Loss function for preserving pathological features during denoising.

    Key principle: In detected pathology regions, penalize when the corrected
    output loses features that were present in the noisy input. This prevents
    over-smoothing of potentially diagnostic features.

    Loss components:
    1. Feature preservation: MSE between corrected and noisy features in pathology regions
    2. Contrast preservation: Penalize reduced local contrast in pathology regions
    3. Texture preservation: Penalize reduced texture variance in pathology regions
    4. Boundary preservation: Preserve sharp boundaries within pathology regions

    Can be added to V8EnhancedLoss with learnable uncertainty weighting.
    """

    def __init__(self,
                 hidden_dim: int = 32,
                 lambda_feature: float = 1.0,
                 lambda_contrast: float = 0.5,
                 lambda_texture: float = 0.5,
                 lambda_boundary: float = 0.3,
                 use_uncertainty_weighting: bool = True):
        """
        Args:
            hidden_dim: Hidden dimension for PathologyPreservationModule
            lambda_feature: Weight for deep feature preservation loss
            lambda_contrast: Weight for contrast preservation loss
            lambda_texture: Weight for texture preservation loss
            lambda_boundary: Weight for boundary preservation loss
            use_uncertainty_weighting: Whether to use learnable uncertainty weights
        """
        super().__init__()

        self.lambda_feature = lambda_feature
        self.lambda_contrast = lambda_contrast
        self.lambda_texture = lambda_texture
        self.lambda_boundary = lambda_boundary
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Pathology preservation module
        self.pathology_module = PathologyPreservationModule(hidden_dim=hidden_dim)

        # Learnable uncertainty parameters (for integration with V8EnhancedLoss)
        if use_uncertainty_weighting:
            self.log_sigma = nn.ParameterDict({
                'pathology_feature': nn.Parameter(torch.tensor(1.0)),
                'pathology_contrast': nn.Parameter(torch.tensor(1.0)),
                'pathology_texture': nn.Parameter(torch.tensor(1.0)),
                'pathology_boundary': nn.Parameter(torch.tensor(1.0)),
            })

        # Sobel filters for boundary preservation
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Local variance computation
        self.local_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

    def _weighted_loss(self, loss: torch.Tensor, name: str) -> torch.Tensor:
        """Apply uncertainty weighting to a loss term."""
        if self.use_uncertainty_weighting:
            log_sig = self.log_sigma[name]
            precision_weight = 0.5 * torch.exp(-log_sig)
            regularization = 0.5 * log_sig
            return precision_weight * loss + regularization
        else:
            weight_map = {
                'pathology_feature': self.lambda_feature,
                'pathology_contrast': self.lambda_contrast,
                'pathology_texture': self.lambda_texture,
                'pathology_boundary': self.lambda_boundary,
            }
            return weight_map[name] * loss

    def feature_preservation_loss(self,
                                   corrected_features: torch.Tensor,
                                   noisy_features: torch.Tensor,
                                   pathology_map: torch.Tensor) -> torch.Tensor:
        """
        Preserve deep features in pathology regions.

        Penalize when corrected features differ too much from noisy features
        in detected pathology regions.
        """
        # MSE between features, weighted by pathology probability
        diff = (corrected_features - noisy_features) ** 2

        # Expand pathology map to match feature channels
        if pathology_map.shape[1] != diff.shape[1]:
            pathology_weight = pathology_map.expand_as(diff)
        else:
            pathology_weight = pathology_map

        # Weighted MSE
        weighted_diff = diff * pathology_weight
        loss = weighted_diff.sum() / (pathology_weight.sum() + 1e-8)

        return loss

    def contrast_preservation_loss(self,
                                    corrected: torch.Tensor,
                                    noisy: torch.Tensor,
                                    pathology_map: torch.Tensor) -> torch.Tensor:
        """
        Preserve local contrast in pathology regions.

        Penalize when local contrast (std) in corrected is LESS than in noisy
        for pathology regions. This prevents over-smoothing pathological features.
        """
        # Compute local std
        def local_std(x):
            local_mean = self.local_pool(x)
            local_sq_mean = self.local_pool(x ** 2)
            local_var = (local_sq_mean - local_mean ** 2).clamp(min=1e-8)
            return torch.sqrt(local_var)

        corrected_std = local_std(corrected)
        noisy_std = local_std(noisy)

        # Penalize when corrected contrast < noisy contrast (ReLU for asymmetric loss)
        contrast_deficit = F.relu(noisy_std - corrected_std)

        # Weight by pathology probability
        weighted_deficit = contrast_deficit * pathology_map
        loss = weighted_deficit.sum() / (pathology_map.sum() + 1e-8)

        return loss

    def texture_preservation_loss(self,
                                   corrected: torch.Tensor,
                                   noisy: torch.Tensor,
                                   pathology_map: torch.Tensor) -> torch.Tensor:
        """
        Preserve texture variance in pathology regions.

        Penalize when local variance in corrected is LESS than in noisy
        for pathology regions. This prevents over-smoothing fine pathological textures.
        """
        # Compute local variance
        def local_var(x):
            local_mean = self.local_pool(x)
            local_sq_mean = self.local_pool(x ** 2)
            return (local_sq_mean - local_mean ** 2).clamp(min=1e-8)

        corrected_var = local_var(corrected)
        noisy_var = local_var(noisy)

        # Penalize when corrected variance < noisy variance
        variance_deficit = F.relu(noisy_var - corrected_var)

        # Weight by pathology probability
        weighted_deficit = variance_deficit * pathology_map
        loss = weighted_deficit.sum() / (pathology_map.sum() + 1e-8)

        return loss

    def boundary_preservation_loss(self,
                                    corrected: torch.Tensor,
                                    noisy: torch.Tensor,
                                    pathology_map: torch.Tensor) -> torch.Tensor:
        """
        Preserve boundaries/edges within pathology regions.

        Pathological features often have characteristic boundaries (e.g., fluid pockets,
        drusen edges). Penalize when edge magnitude in corrected is LESS than in noisy.
        """
        # Compute gradient magnitude
        def grad_mag(x):
            grad_x = F.conv2d(x, self.sobel_x.to(dtype=x.dtype), padding=1)
            grad_y = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1)
            return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        corrected_grad = grad_mag(corrected)
        noisy_grad = grad_mag(noisy)

        # Penalize when corrected edges < noisy edges
        edge_deficit = F.relu(noisy_grad - corrected_grad)

        # Weight by pathology probability
        weighted_deficit = edge_deficit * pathology_map
        loss = weighted_deficit.sum() / (pathology_map.sum() + 1e-8)

        return loss

    def forward(self,
                corrected: torch.Tensor,
                backbone_out: torch.Tensor,
                noisy: torch.Tensor,
                pathology_info: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute pathology preservation loss.

        Args:
            corrected: Final corrected output [B, 1, H, W]
            backbone_out: Backbone denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            pathology_info: Optional pre-computed pathology info from PathologyPreservationModule

        Returns:
            total_loss: Weighted sum of all pathology preservation losses
            loss_info: Dict with individual loss values and pathology statistics
        """
        # Get pathology information
        if pathology_info is None:
            pathology_info = self.pathology_module(noisy, backbone_out)

        pathology_map = pathology_info['pathology_map']

        # Extract features for corrected output
        corrected_features = self.pathology_module.extract_pathology_features(
            corrected, pathology_map
        )
        noisy_features = pathology_info['noisy_pathology_features']

        # Compute individual losses
        feature_loss = self.feature_preservation_loss(
            corrected_features['masked_deep_features'],
            noisy_features['masked_deep_features'],
            pathology_map
        )

        contrast_loss = self.contrast_preservation_loss(
            corrected, noisy, pathology_map
        )

        texture_loss = self.texture_preservation_loss(
            corrected, noisy, pathology_map
        )

        boundary_loss = self.boundary_preservation_loss(
            corrected, noisy, pathology_map
        )

        # Combine with uncertainty weighting
        total_loss = (
            self._weighted_loss(feature_loss, 'pathology_feature') +
            self._weighted_loss(contrast_loss, 'pathology_contrast') +
            self._weighted_loss(texture_loss, 'pathology_texture') +
            self._weighted_loss(boundary_loss, 'pathology_boundary')
        )

        # Compute pathology statistics for monitoring
        with torch.no_grad():
            pathology_coverage = pathology_map.mean()
            pathology_max = pathology_map.max()

            # Detection breakdown
            detection_info = pathology_info['detection_info']
            detection_stats = {
                'intensity': detection_info['intensity_anomaly'].mean().item(),
                'texture': detection_info['texture_irregularity'].mean().item(),
                'layer': detection_info['layer_disruption'].mean().item(),
                'fluid': detection_info['fluid_pattern'].mean().item(),
            }

            # Learned weights
            detector_weights = detection_info['weights'].detach().cpu().numpy()

        loss_info = {
            'pathology_total': total_loss.item(),
            'pathology_feature': feature_loss.item(),
            'pathology_contrast': contrast_loss.item(),
            'pathology_texture': texture_loss.item(),
            'pathology_boundary': boundary_loss.item(),
            'pathology_coverage': pathology_coverage.item(),
            'pathology_max': pathology_max.item(),
            'pathology_detection': detection_stats,
            'pathology_detector_weights': {
                'intensity': detector_weights[0],
                'texture': detector_weights[1],
                'layer': detector_weights[2],
                'fluid': detector_weights[3],
            },
        }

        # Add learned uncertainties if using uncertainty weighting
        if self.use_uncertainty_weighting:
            uncertainties = {
                name: torch.exp(0.5 * log_sig).item()
                for name, log_sig in self.log_sigma.items()
            }
            loss_info['pathology_uncertainties'] = uncertainties

        return total_loss, loss_info

    def get_correction_gate(self, noisy: torch.Tensor,
                            backbone_out: torch.Tensor) -> torch.Tensor:
        """
        Get correction gate for pathology-aware correction.

        This gate can be multiplied with correction magnitude to reduce
        correction strength in pathology regions.

        Args:
            noisy: Noisy input [B, 1, H, W]
            backbone_out: Backbone output [B, 1, H, W]

        Returns:
            gate: [B, 1, H, W] correction gate (1 = full, min = reduced in pathology)
        """
        pathology_map, _ = self.pathology_module.get_pathology_map(noisy)
        return self.pathology_module.get_correction_gate(pathology_map)


# =============================================================================
# COMBINED LOSS WITH PATHOLOGY PRESERVATION
# =============================================================================

class V8EnhancedLossWithPathology(nn.Module):
    """
    Extension of V8EnhancedLoss that includes pathology preservation.

    This wrapper combines the existing V8EnhancedLoss with PathologyPreservationLoss.
    Can be used as a drop-in replacement for V8EnhancedLoss.

    Usage:
        criterion = V8EnhancedLossWithPathology(base_loss=existing_loss)
        # or
        criterion = V8EnhancedLossWithPathology()  # Creates new V8EnhancedLoss
    """

    def __init__(self,
                 base_loss: Optional[nn.Module] = None,
                 pathology_weight: float = 0.5,
                 pathology_hidden_dim: int = 32,
                 use_uncertainty_weighting: bool = True):
        """
        Args:
            base_loss: Existing V8EnhancedLoss instance (or None to create new)
            pathology_weight: Overall weight for pathology preservation loss
            pathology_hidden_dim: Hidden dimension for pathology module
            use_uncertainty_weighting: Use uncertainty weighting for pathology losses
        """
        super().__init__()

        # Base loss (V8EnhancedLoss)
        if base_loss is not None:
            self.base_loss = base_loss
        else:
            # Import and create V8EnhancedLoss
            from train_v8_enhanced import V8EnhancedLoss
            self.base_loss = V8EnhancedLoss(use_uncertainty_weighting=use_uncertainty_weighting)

        # Pathology preservation loss
        self.pathology_loss = PathologyPreservationLoss(
            hidden_dim=pathology_hidden_dim,
            use_uncertainty_weighting=use_uncertainty_weighting
        )

        # Learnable weight for pathology loss (using uncertainty weighting)
        if use_uncertainty_weighting:
            self.log_sigma_pathology = nn.Parameter(torch.tensor(1.0))
        self.pathology_weight = pathology_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting

    def forward(self, corrected, backbone_out, clean, info, noisy=None):
        """
        Compute combined loss with pathology preservation.

        Args:
            corrected: Corrected output
            backbone_out: Backbone output
            clean: Ground truth clean image
            info: Info dict from corrector
            noisy: Original noisy input (required for pathology detection)

        Returns:
            total_loss: Combined loss
            metrics: Dict with all loss metrics
        """
        # Compute base loss
        base_total, base_metrics = self.base_loss(corrected, backbone_out, clean, info)

        # Compute pathology loss if noisy is provided
        if noisy is not None:
            pathology_total, pathology_metrics = self.pathology_loss(
                corrected, backbone_out, noisy
            )

            # Apply uncertainty weighting to pathology loss
            if self.use_uncertainty_weighting:
                precision = 0.5 * torch.exp(-self.log_sigma_pathology)
                regularization = 0.5 * self.log_sigma_pathology
                pathology_weighted = precision * pathology_total + regularization
            else:
                pathology_weighted = self.pathology_weight * pathology_total

            # Combine losses
            total_loss = base_total + pathology_weighted

            # Merge metrics
            metrics = {**base_metrics, **pathology_metrics}
            metrics['total'] = total_loss.item()
            metrics['base_loss'] = base_total.item()
            metrics['pathology_loss_weighted'] = pathology_weighted.item()

            if self.use_uncertainty_weighting:
                metrics['pathology_overall_uncertainty'] = torch.exp(
                    0.5 * self.log_sigma_pathology
                ).item()
        else:
            # No pathology loss if noisy not provided
            total_loss = base_total
            metrics = base_metrics

        return total_loss, metrics


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing Pathology Preservation Module")
    print("=" * 70)

    # Test configuration
    B, C, H, W = 2, 1, 128, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Test tensor shape: [{B}, {C}, {H}, {W}]")

    # Create test data
    noisy = torch.rand(B, C, H, W, device=device) * 0.5 + 0.25  # Random in [0.25, 0.75]

    # Add some simulated pathological features
    # Dark spot (fluid)
    noisy[0, 0, 40:60, 50:70] = 0.1
    # Bright spot (drusen)
    noisy[0, 0, 80:95, 30:50] = 0.9
    # High variance region
    noisy[1, 0, 20:50, 60:90] = torch.rand(30, 30, device=device) * 0.6 + 0.2

    # Simulated backbone output (smoothed version)
    backbone_out = F.avg_pool2d(
        F.pad(noisy, [2, 2, 2, 2], mode='reflect'),
        kernel_size=5, stride=1
    )

    # Simulated clean image
    clean = torch.rand(B, C, H, W, device=device) * 0.4 + 0.3

    print("\n" + "-" * 50)
    print("1. Testing PathologyDetector")
    print("-" * 50)

    detector = PathologyDetector().to(device)
    detection = detector(noisy)

    print(f"Intensity anomaly range: [{detection['intensity_anomaly'].min():.3f}, {detection['intensity_anomaly'].max():.3f}]")
    print(f"Texture irregularity range: [{detection['texture_irregularity'].min():.3f}, {detection['texture_irregularity'].max():.3f}]")
    print(f"Layer disruption range: [{detection['layer_disruption'].min():.3f}, {detection['layer_disruption'].max():.3f}]")
    print(f"Fluid pattern range: [{detection['fluid_pattern'].min():.3f}, {detection['fluid_pattern'].max():.3f}]")
    print(f"Combined map range: [{detection['combined'].min():.3f}, {detection['combined'].max():.3f}]")
    print(f"Learned weights: {detection['weights'].detach().cpu().numpy()}")

    print("\n" + "-" * 50)
    print("2. Testing PathologyPreservationModule")
    print("-" * 50)

    module = PathologyPreservationModule(hidden_dim=32).to(device)
    output = module(noisy, backbone_out)

    print(f"Pathology map shape: {output['pathology_map'].shape}")
    print(f"Pathology map range: [{output['pathology_map'].min():.3f}, {output['pathology_map'].max():.3f}]")
    print(f"Pathology coverage: {output['pathology_map'].mean():.3f}")
    print(f"Correction gate range: [{output['correction_gate'].min():.3f}, {output['correction_gate'].max():.3f}]")

    print("\n" + "-" * 50)
    print("3. Testing PathologyPreservationLoss")
    print("-" * 50)

    loss_module = PathologyPreservationLoss(hidden_dim=32).to(device)

    # Simulated corrected output
    corrected = backbone_out + torch.randn_like(backbone_out) * 0.02
    corrected = corrected.clamp(0, 1)

    loss, loss_info = loss_module(corrected, backbone_out, noisy)

    print(f"Total pathology loss: {loss.item():.4f}")
    print(f"Feature loss: {loss_info['pathology_feature']:.4f}")
    print(f"Contrast loss: {loss_info['pathology_contrast']:.4f}")
    print(f"Texture loss: {loss_info['pathology_texture']:.4f}")
    print(f"Boundary loss: {loss_info['pathology_boundary']:.4f}")
    print(f"Pathology coverage: {loss_info['pathology_coverage']:.3f}")
    print(f"Detection breakdown: {loss_info['pathology_detection']}")
    print(f"Detector weights: {loss_info['pathology_detector_weights']}")

    print("\n" + "-" * 50)
    print("4. Testing Gradient Flow")
    print("-" * 50)

    # Test backward pass
    loss.backward()

    # Check gradients
    detector_grads = sum(p.grad.abs().sum().item() for p in loss_module.pathology_module.detector.parameters() if p.grad is not None)
    refine_grads = sum(p.grad.abs().sum().item() for p in loss_module.pathology_module.refine_net.parameters() if p.grad is not None)

    print(f"Detector gradient sum: {detector_grads:.4f}")
    print(f"Refine network gradient sum: {refine_grads:.4f}")

    print("\n" + "-" * 50)
    print("5. Testing Parameter Count")
    print("-" * 50)

    detector_params = sum(p.numel() for p in loss_module.pathology_module.detector.parameters())
    module_params = sum(p.numel() for p in loss_module.pathology_module.parameters())
    loss_params = sum(p.numel() for p in loss_module.parameters())

    print(f"PathologyDetector: {detector_params:,} parameters")
    print(f"PathologyPreservationModule: {module_params:,} parameters")
    print(f"PathologyPreservationLoss (total): {loss_params:,} parameters")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
