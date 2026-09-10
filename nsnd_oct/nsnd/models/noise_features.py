"""
Principled Noise Feature Extraction for OCT

This module extracts PHYSICAL features that distinguish noise types:
1. Speckle: Multiplicative, high coefficient of variation, signal-correlated
2. Banding: Periodic, frequency peaks, horizontal patterns
3. Gaussian: Additive uniform, constant variance, signal-independent
4. Shot: Signal-dependent variance, Poisson-like

These features are based on noise physics, not learned - they WILL work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoiseFeatureExtractor(nn.Module):
    """
    Extract hand-crafted noise features that distinguish noise types.

    Features computed per-pixel (with local windows):
    - Local mean, std, coefficient of variation
    - Local frequency content (DCT-based)
    - Signal-variance correlation
    - Horizontal vs vertical gradient ratio
    """

    def __init__(self, window_size: int = 7):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2

        # Sobel filters for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 4.0)

        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 4.0)

        # Laplacian for high-frequency content
        self.register_buffer('laplacian', torch.tensor([
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        ], dtype=torch.float32).unsqueeze(0))

        # Horizontal line detector (for banding)
        self.register_buffer('horizontal_detector', torch.tensor([
            [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]
        ], dtype=torch.float32).unsqueeze(0) / 6.0)

    def local_stats(self, x: torch.Tensor) -> tuple:
        """Compute local mean and std using average pooling."""
        # Local mean
        local_mean = F.avg_pool2d(
            F.pad(x, [self.pad]*4, mode='reflect'),
            self.window_size, stride=1
        )

        # Local variance = E[X^2] - E[X]^2
        local_sq_mean = F.avg_pool2d(
            F.pad(x**2, [self.pad]*4, mode='reflect'),
            self.window_size, stride=1
        )
        local_var = (local_sq_mean - local_mean**2).clamp(min=1e-8)
        local_std = torch.sqrt(local_var)

        return local_mean, local_std, local_var

    def forward(self, x: torch.Tensor) -> dict:
        """
        Extract noise-distinguishing features.

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            Dictionary of feature maps, each [B, 1, H, W]
        """
        B, C, H, W = x.shape
        features = {}

        # =================================================================
        # 1. LOCAL STATISTICS (distinguish speckle vs gaussian)
        # =================================================================
        local_mean, local_std, local_var = self.local_stats(x)

        # Coefficient of variation: std/mean
        # HIGH for speckle (multiplicative), LOW for gaussian (additive)
        features['coef_variation'] = local_std / (local_mean + 1e-8)

        # Normalized local std (signal-independent noise measure)
        features['local_std'] = local_std

        # =================================================================
        # 2. SIGNAL-VARIANCE CORRELATION (distinguish shot noise)
        # =================================================================
        # For shot noise: variance ∝ signal (Poisson)
        # Compute correlation between local_mean and local_var

        mean_centered = local_mean - local_mean.mean(dim=[2, 3], keepdim=True)
        var_centered = local_var - local_var.mean(dim=[2, 3], keepdim=True)

        # Local correlation (using larger window)
        large_pad = 3
        mean_local = F.avg_pool2d(F.pad(mean_centered, [large_pad]*4, mode='reflect'), 7, stride=1)
        var_local = F.avg_pool2d(F.pad(var_centered, [large_pad]*4, mode='reflect'), 7, stride=1)

        correlation = (mean_local * var_local) / (
            torch.sqrt(mean_local**2 + 1e-8) * torch.sqrt(var_local**2 + 1e-8) + 1e-8
        )
        # HIGH correlation = shot noise, LOW = gaussian
        features['signal_var_corr'] = correlation

        # =================================================================
        # 3. FREQUENCY CONTENT (distinguish banding)
        # =================================================================
        # Banding appears as horizontal stripes = strong horizontal frequency

        # Horizontal gradient (detects vertical edges)
        grad_x = F.conv2d(F.pad(x, [1]*4, mode='reflect'), self.sobel_x)
        # Vertical gradient (detects horizontal edges/banding)
        grad_y = F.conv2d(F.pad(x, [1]*4, mode='reflect'), self.sobel_y)

        grad_x_mag = grad_x.abs()
        grad_y_mag = grad_y.abs()

        # Ratio of horizontal to vertical gradient
        # HIGH = horizontal patterns (banding), LOW = isotropic
        features['horizontal_ratio'] = grad_y_mag / (grad_x_mag + grad_y_mag + 1e-8)

        # Direct horizontal line detection
        horizontal_response = F.conv2d(F.pad(x, [1]*4, mode='reflect'), self.horizontal_detector)
        features['horizontal_lines'] = horizontal_response.abs()

        # =================================================================
        # 4. HIGH FREQUENCY CONTENT (overall noise level)
        # =================================================================
        laplacian_response = F.conv2d(F.pad(x, [1]*4, mode='reflect'), self.laplacian)
        features['high_freq'] = laplacian_response.abs()

        # =================================================================
        # 5. LOCAL ENTROPY (texture complexity)
        # =================================================================
        # Approximate entropy using local histogram variance
        # Speckle has characteristic texture
        local_range = F.max_pool2d(F.pad(x, [self.pad]*4, mode='reflect'),
                                    self.window_size, stride=1) - \
                      F.max_pool2d(F.pad(-x, [self.pad]*4, mode='reflect'),
                                    self.window_size, stride=1).neg()
        features['local_range'] = local_range

        return features


class NoiseTypeClassifier(nn.Module):
    """
    Classify noise type from hand-crafted features.

    Uses a SIMPLE LINEAR classifier with PHYSICS-BASED initialization.
    This ensures features are actually used for classification.
    """

    def __init__(self, num_noise_types: int = 4):
        super().__init__()
        self.feature_extractor = NoiseFeatureExtractor()

        # Number of features from extractor
        num_features = 7  # coef_var, local_std, signal_var_corr, horiz_ratio, horiz_lines, high_freq, local_range

        # SIMPLE linear classifier (1x1 conv = per-pixel linear)
        # This forces direct use of physics features
        self.classifier = nn.Conv2d(num_features, num_noise_types, 1, bias=True)

        # Initialize with PHYSICS-BASED priors
        self._init_physics_prior()

    def _init_physics_prior(self):
        """Initialize weights based on KNOWN noise physics relationships."""
        with torch.no_grad():
            # Feature order: coef_var, local_std, signal_var_corr, horiz_ratio, horiz_lines, high_freq, local_range
            # Output order: speckle, banding, gaussian, shot

            # Initialize to zero first
            self.classifier.weight.zero_()
            self.classifier.bias.zero_()

            # Speckle (idx=0): HIGH coef_variation (idx=0), HIGH local_range (idx=6)
            self.classifier.weight[0, 0, 0, 0] = 2.0   # coef_var -> speckle
            self.classifier.weight[0, 6, 0, 0] = 1.0   # local_range -> speckle

            # Banding (idx=1): HIGH horizontal_ratio (idx=3), HIGH horizontal_lines (idx=4)
            self.classifier.weight[1, 3, 0, 0] = 2.0   # horiz_ratio -> banding
            self.classifier.weight[1, 4, 0, 0] = 2.0   # horiz_lines -> banding

            # Gaussian (idx=2): LOW coef_variation (idx=0), LOW signal_var_corr (idx=2)
            # High weight on LOCAL_STD indicates additive noise
            self.classifier.weight[2, 0, 0, 0] = -1.5  # LOW coef_var -> gaussian
            self.classifier.weight[2, 1, 0, 0] = 1.5   # high local_std -> gaussian
            self.classifier.weight[2, 2, 0, 0] = -1.0  # LOW sig_var_corr -> gaussian

            # Shot (idx=3): HIGH signal_var_corr (idx=2) - variance proportional to signal
            self.classifier.weight[3, 2, 0, 0] = 2.5   # sig_var_corr -> shot
            self.classifier.weight[3, 0, 0, 0] = 0.5   # moderate coef_var -> shot

            # Small positive bias for gaussian (it's common background)
            self.classifier.bias[2] = 0.5

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Classify noise type per pixel.

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            noise_type: Per-pixel noise type probabilities [B, 4, H, W]
            features: Dictionary of extracted features
        """
        # Extract hand-crafted features
        features = self.feature_extractor(x)

        # Normalize features for stable training
        coef_var = (features['coef_variation'] - 0.3) / 0.2  # Center around typical value
        local_std = (features['local_std'] - 0.1) / 0.1
        sig_var_corr = features['signal_var_corr']  # Already centered
        horiz_ratio = (features['horizontal_ratio'] - 0.5) / 0.1  # Center around 0.5
        horiz_lines = features['horizontal_lines'] / 0.1
        high_freq = features['high_freq'] / 0.1
        local_range = (features['local_range'] - 0.3) / 0.2

        # Stack normalized features
        feature_tensor = torch.cat([
            coef_var,
            local_std,
            sig_var_corr,
            horiz_ratio,
            horiz_lines,
            high_freq,
            local_range,
        ], dim=1)  # [B, 7, H, W]

        # Classify using simple linear mapping
        logits = self.classifier(feature_tensor)  # [B, 4, H, W]
        noise_type = F.softmax(logits, dim=1)

        return noise_type, features


class DistinctSymbolicExperts(nn.Module):
    """
    Architecturally DISTINCT experts for each noise type.

    Each expert uses a DIFFERENT processing strategy:
    - Speckle: Log-domain processing (multiplicative → additive)
    - Banding: Frequency-domain notch filtering
    - Gaussian: Local adaptive Wiener-like filter
    - Shot: Variance-stabilizing transform (Anscombe)
    """

    def __init__(self):
        super().__init__()

        # Speckle expert: Log-domain denoising
        self.speckle_expert = SpeckleExpert()

        # Banding expert: Frequency-domain processing
        self.banding_expert = BandingExpert()

        # Gaussian expert: Local adaptive filter
        self.gaussian_expert = GaussianExpert()

        # Shot expert: VST-based processing
        self.shot_expert = ShotExpert()

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> dict:
        """Apply all experts and return their outputs."""
        return {
            'speckle': self.speckle_expert(x, noise_level),
            'banding': self.banding_expert(x, noise_level),
            'gaussian': self.gaussian_expert(x, noise_level),
            'shot': self.shot_expert(x, noise_level),
        }


class SpeckleExpert(nn.Module):
    """
    Speckle denoising via log-domain processing.

    Speckle is multiplicative: observed = clean * speckle_noise
    In log domain: log(observed) = log(clean) + log(speckle_noise)
    This becomes ADDITIVE, easier to filter.
    """

    def __init__(self):
        super().__init__()
        # Simple denoising in log domain
        self.log_filter = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        # Strength modulation
        self.strength = nn.Conv2d(1, 1, 1)
        nn.init.ones_(self.strength.weight)
        nn.init.zeros_(self.strength.bias)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        # Transform to log domain (avoid log(0))
        x_log = torch.log(x.clamp(min=1e-6))

        # Filter in log domain
        filtered_log = x_log + self.log_filter(x_log)

        # Transform back
        filtered = torch.exp(filtered_log)

        # Modulate by noise level
        strength = torch.sigmoid(self.strength(noise_level))
        output = strength * filtered + (1 - strength) * x

        return output.clamp(0, 1)


class BandingExpert(nn.Module):
    """
    Banding removal via frequency-aware filtering.

    Banding appears as horizontal stripes = vertical frequency components.
    We detect and suppress these specifically.
    """

    def __init__(self):
        super().__init__()
        # Vertical-aware filter (tall kernel to detect horizontal bands)
        self.band_detector = nn.Conv2d(1, 8, (7, 3), padding=(3, 1))

        # Adaptive notch filter
        self.notch_filter = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1),  # 1 image + 8 band features
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        self.strength = nn.Conv2d(1, 1, 1)
        nn.init.ones_(self.strength.weight)
        nn.init.zeros_(self.strength.bias)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        # Detect banding patterns
        band_features = self.band_detector(x)

        # Combine with image for adaptive filtering
        combined = torch.cat([x, band_features], dim=1)
        correction = self.notch_filter(combined)

        # Apply correction
        strength = torch.sigmoid(self.strength(noise_level))
        output = x + strength * correction

        return output.clamp(0, 1)


class GaussianExpert(nn.Module):
    """
    Gaussian noise removal via local adaptive filtering.

    Gaussian noise is additive and uniform.
    Local Wiener-like filtering works well.
    """

    def __init__(self):
        super().__init__()
        # Estimate local signal and noise
        self.estimator = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.ReLU(inplace=True),
        )

        # Wiener-like gain
        self.gain = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

        # Local mean (for Wiener filtering)
        self.local_mean = nn.AvgPool2d(5, stride=1, padding=2)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        # Local mean
        mean = self.local_mean(x)

        # Estimate adaptive gain
        features = self.estimator(x)
        gain = torch.sigmoid(self.gain(features))

        # Wiener-like: output = mean + gain * (x - mean)
        # gain = signal_var / (signal_var + noise_var)
        # Low gain in flat regions (trust mean), high gain in edges (trust signal)
        output = mean + gain * (x - mean)

        return output.clamp(0, 1)


class ShotExpert(nn.Module):
    """
    Shot noise removal via variance-stabilizing transform.

    Shot noise has variance proportional to signal (Poisson).
    Anscombe transform: 2*sqrt(x + 3/8) makes variance ~constant.
    """

    def __init__(self):
        super().__init__()
        # Filter in stabilized domain
        self.stabilized_filter = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        self.strength = nn.Conv2d(1, 1, 1)
        nn.init.ones_(self.strength.weight)
        nn.init.zeros_(self.strength.bias)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        # Anscombe transform (generalized for [0,1] range)
        # Scale to reasonable range first
        x_scaled = x * 255.0  # Approximate photon counts
        x_anscombe = 2.0 * torch.sqrt(x_scaled + 3.0/8.0)

        # Filter in stabilized domain
        filtered_anscombe = x_anscombe + self.stabilized_filter(x_anscombe / 32.0) * 32.0  # Normalize

        # Inverse Anscombe
        filtered_scaled = (filtered_anscombe / 2.0) ** 2 - 3.0/8.0
        filtered = filtered_scaled / 255.0

        # Modulate by noise level
        strength = torch.sigmoid(self.strength(noise_level))
        output = strength * filtered + (1 - strength) * x

        return output.clamp(0, 1)
