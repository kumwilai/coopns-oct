"""
Symbolic Noise Analyzer: Neural perception + Symbolic reasoning

Combines learned feature extraction with interpretable symbolic rules
for noise decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from ..symbolic.rules import SymbolicReasoningEngine
from ..symbolic.neuro_symbolic import NeuroSymbolicReasoner


FEATURE_KEYS = [
    "local_mean",
    "local_std",
    "cv_map",
    "kurtosis",
    "vertical_power",
    "vertical_peak_ratio",
    "high_freq_ratio",
    "gradient_magnitude",
    "depth_profile",
]


class NoiseFeatureExtractor(nn.Module):
    """
    Neural network that extracts statistical features for symbolic reasoning

    Computes local statistics in sliding windows:
    - Local mean, std, CV
    - Kurtosis (speckle indicator)
    - Vertical FFT power + peakiness (banding indicator)
    - High-frequency energy ratio (speckle indicator)
    - Gradient magnitude (edge preservation)
    - Depth profile (shot noise indicator)
    """

    def __init__(
        self,
        window_size: int = 7,
        kurtosis_window: int = 15,
        learnable_features: bool = False,
        high_freq_cutoff: float = 0.35,
    ):
        super().__init__()
        self.window_size = window_size
        self.kurtosis_window = kurtosis_window
        self.learnable = learnable_features
        self.high_freq_cutoff = float(high_freq_cutoff)

        # Optional learnable feature refinement
        if learnable_features:
            self.feature_refiner = nn.Sequential(
                nn.Conv2d(len(FEATURE_KEYS), 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, len(FEATURE_KEYS), 1),
            )
        else:
            self.feature_refiner = nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract noise-relevant features

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            features: Dictionary of computed features
        """
        B, C, H, W = x.shape
        assert C == 1, "Expecting single-channel images"

        features = {}

        # 1. Local mean
        features['local_mean'] = self._local_mean(x, self.window_size)

        # 2. Local std
        features['local_std'] = self._local_std(x, self.window_size)

        # 3. Coefficient of variation (CV)
        features['cv_map'] = features['local_std'] / (features['local_mean'] + 1e-6)

        # 4. Kurtosis (fourth moment)
        features['kurtosis'] = self._local_kurtosis(x, self.kurtosis_window)

        # 5. Frequency-domain features (banding + speckle indicators)
        features.update(self._frequency_features(x))

        # 6. Gradient magnitude (for edge-aware processing)
        features['gradient_magnitude'] = self._sobel_gradient(x)

        # 7. Depth profile (axial intensity decay)
        features['depth_profile'] = self._axial_intensity_profile(x)

        # Optional: Refine features with learned convolutions
        if self.learnable:
            stacked = torch.cat([features[k] for k in FEATURE_KEYS], dim=1)
            refined = self.feature_refiner(stacked)
            for i, k in enumerate(FEATURE_KEYS):
                features[k] = refined[:, i:i+1]

        return features

    def _local_mean(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Compute local mean using average pooling"""
        pad = window // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.avg_pool2d(x_padded, kernel_size=window, stride=1)

    def _local_std(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Compute local standard deviation"""
        pad = window // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')

        # Compute local mean
        mean = F.avg_pool2d(x_padded, kernel_size=window, stride=1)

        # Compute local variance
        x_sq_padded = F.pad(x ** 2, (pad, pad, pad, pad), mode='reflect')
        mean_sq = F.avg_pool2d(x_sq_padded, kernel_size=window, stride=1)
        var = mean_sq - mean ** 2
        var = var.clamp_min(0.0)

        return torch.sqrt(var + 1e-8)

    def _local_kurtosis(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """
        Compute local kurtosis (fourth standardized moment)
        Kurtosis > 3: heavy-tailed (speckle)
        Kurtosis ≈ 3: Gaussian
        """
        pad = window // 2

        # Compute local mean and std using consistent padding
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        mean = F.avg_pool2d(x_padded, kernel_size=window, stride=1)

        # Compute variance
        x_sq_padded = F.pad(x ** 2, (pad, pad, pad, pad), mode='reflect')
        mean_sq = F.avg_pool2d(x_sq_padded, kernel_size=window, stride=1)
        var = mean_sq - mean ** 2
        var = var.clamp_min(0.0)
        std = torch.sqrt(var + 1e-8)

        # Compute fourth moment
        x_4th_padded = F.pad(x ** 4, (pad, pad, pad, pad), mode='reflect')
        m4 = F.avg_pool2d(x_4th_padded, kernel_size=window, stride=1)

        kurtosis = m4 / ((std ** 4) + 1e-8)
        return kurtosis

    def _frequency_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute frequency-domain features for banding/speckle detection.
        """
        B, C, H, W = x.shape

        # Apply 2D FFT
        fft = torch.fft.fft2(x.squeeze(1))
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        total_power = magnitude.sum(dim=(-2, -1), keepdim=True)

        # Extract vertical frequency band (center horizontal line)
        center_h = H // 2
        vertical_band = magnitude[:, center_h-2:center_h+3, :]  # 5-pixel band
        vertical_power = vertical_band.sum(dim=(-2, -1)) / (total_power.squeeze(-1).squeeze(-1) + 1e-8)

        # Peakiness of vertical band (periodic banding indicator)
        band_mean = vertical_band.mean(dim=(-2, -1))
        band_peak = vertical_band.max(dim=-1).values.max(dim=-1).values
        peak_ratio = band_peak / (band_mean + 1e-8)
        vertical_peak_ratio = peak_ratio / (peak_ratio + 1.0)

        # High-frequency energy ratio (speckle indicator)
        mask = self._high_freq_mask(H, W, device=x.device, dtype=magnitude.dtype)
        high_power = (magnitude * mask).sum(dim=(-2, -1))
        high_freq_ratio = high_power / (total_power.squeeze(-1).squeeze(-1) + 1e-8)

        return {
            "vertical_power": vertical_power.view(B, 1, 1, 1).expand(B, 1, H, W),
            "vertical_peak_ratio": vertical_peak_ratio.view(B, 1, 1, 1).expand(B, 1, H, W),
            "high_freq_ratio": high_freq_ratio.view(B, 1, 1, 1).expand(B, 1, H, W),
        }

    def _high_freq_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a radial high-frequency mask for FFT magnitude."""
        yy = torch.linspace(-0.5, 0.5, h, device=device, dtype=dtype)
        xx = torch.linspace(-0.5, 0.5, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_y ** 2 + grid_x ** 2)
        return (radius >= self.high_freq_cutoff).to(dtype)

    def _sobel_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operator"""
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), sobel_x)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), sobel_y)

        return torch.sqrt(gx ** 2 + gy ** 2)

    def _axial_intensity_profile(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute average axial (depth) intensity profile

        Returns depth map normalized to [0, 1]
        """
        B, C, H, W = x.shape

        # Average across lateral dimension
        axial_profile = x.mean(dim=-1, keepdim=True)  # [B, 1, H, 1]

        # Broadcast to full spatial dimensions
        depth_map = axial_profile.expand(B, 1, H, W)

        return depth_map


class SymbolicNoiseAnalyzer(nn.Module):
    """
    Complete symbolic noise analyzer

    Combines neural feature extraction with symbolic reasoning
    to produce interpretable noise composition
    """

    def __init__(
        self,
        window_size: int = 7,
        kurtosis_window: int = 15,
        learnable_features: bool = False,
    ):
        super().__init__()

        self.feature_extractor = NoiseFeatureExtractor(
            window_size=window_size,
            kurtosis_window=kurtosis_window,
            learnable_features=learnable_features,
        )

        self.reasoning_engine = SymbolicReasoningEngine()

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Analyze noise composition

        Args:
            x: Input noisy image [B, 1, H, W]

        Returns:
            weights: Noise component weights {speckle, banding, gaussian, shot}
            features: Extracted features (for visualization/debugging)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Apply symbolic reasoning
        weights = self.reasoning_engine.infer(features)

        return weights, features

    def analyze_and_explain(self, x: torch.Tensor) -> str:
        """
        Generate human-readable noise analysis report

        Args:
            x: Input noisy image [B, 1, H, W]

        Returns:
            report: Text explanation of detected noise
        """
        weights, _ = self.forward(x)

        report = "="*50 + "\n"
        report += "NSND-OCT Noise Analysis Report\n"
        report += "="*50 + "\n\n"
        report += "Detected Noise Composition:\n"

        for component in ['speckle', 'banding', 'gaussian', 'shot']:
            weight = weights[component].mean().item() * 100
            report += f"├── {component.capitalize()}: {weight:.1f}%\n"

        confidence = weights['_confidence'].mean().item() * 100
        report += f"\nAnalysis Confidence: {confidence:.1f}%\n\n"

        report += "Interpretation:\n"
        report += self.reasoning_engine.explain(
            {k: v.mean() for k, v in weights.items()},
            threshold=0.1
        )

        report += "\n" + "="*50

        return report

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Return overall confidence score"""
        weights, _ = self.forward(x)
        return weights['_confidence']


class NeuroSymbolicNoiseAnalyzer(nn.Module):
    """
    Neuro-symbolic analyzer with learnable predicates and rule weights.
    """

    def __init__(
        self,
        window_size: int = 7,
        kurtosis_window: int = 15,
        learnable_features: bool = False,
        use_neural_predicates: bool = False,
        predicate_hidden: int = 64,
        use_neural_weights: bool = True,
        weight_hidden: int = 64,
    ):
        super().__init__()
        self.feature_extractor = NoiseFeatureExtractor(
            window_size=window_size,
            kurtosis_window=kurtosis_window,
            learnable_features=learnable_features,
        )
        self.use_neural_predicates = bool(use_neural_predicates)
        self.use_neural_weights = bool(use_neural_weights)
        self.reasoner = NeuroSymbolicReasoner()

        if self.use_neural_predicates:
            self.predicate_head = nn.Sequential(
                nn.Conv2d(len(FEATURE_KEYS), predicate_hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(predicate_hidden, predicate_hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(predicate_hidden, 8),
                nn.Sigmoid(),
            )
        else:
            self.predicate_head = None

        if self.use_neural_weights:
            self.weight_head = nn.Sequential(
                nn.Conv2d(len(FEATURE_KEYS), weight_hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(weight_hidden, weight_hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(weight_hidden, 4),
                nn.Softmax(dim=-1),
            )
            self.weight_mix_logit = nn.Parameter(torch.tensor(5.0))  # Start fully neural
        else:
            self.weight_head = None
            self.weight_mix_logit = None

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        features = self.feature_extractor(x)
        stacked = torch.cat([features[k] for k in FEATURE_KEYS], dim=1)

        override_predicates = None
        if self.use_neural_predicates and self.predicate_head is not None:
            pred_vec = self.predicate_head(stacked)
            override_predicates = {
                'high_cv': pred_vec[:, 0],
                'high_kurtosis': pred_vec[:, 1],
                'low_kurtosis': pred_vec[:, 2],
                'uniform_variance': pred_vec[:, 3],
                'banding_present': pred_vec[:, 4],
                'banding_peaky': pred_vec[:, 5],
                'high_freq': pred_vec[:, 6],
                'depth_dependent': pred_vec[:, 7],
            }

        weights_sym, predicates, extra = self.reasoner(features, override_predicates=override_predicates)
        weights = weights_sym
        weights_neural = None
        if self.use_neural_weights and self.weight_head is not None:
            weights_neural = self.weight_head(stacked)
            mix = torch.sigmoid(self.weight_mix_logit)
            weights_stack = torch.stack(
                [
                    weights_sym['speckle'],
                    weights_sym['banding'],
                    weights_sym['gaussian'],
                    weights_sym['shot'],
                ],
                dim=1,
            )
            combined = (1.0 - mix) * weights_stack + mix * weights_neural
            combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
            combined = combined.clamp(min=1e-6)
            weights = {
                'speckle': combined[:, 0],
                'banding': combined[:, 1],
                'gaussian': combined[:, 2],
                'shot': combined[:, 3],
                '_confidence': weights_sym['_confidence'],
            }
        features_out = dict(features)
        features_out['predicates'] = predicates
        if override_predicates is not None:
            features_out['neural_predicates'] = override_predicates
        features_out['symbolic_weights'] = weights_sym
        if weights_neural is not None:
            features_out['neural_weights'] = weights_neural
            features_out['weight_mix'] = torch.sigmoid(self.weight_mix_logit)
        features_out.update(extra)
        return weights, features_out

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        weights, _ = self.forward(x)
        return weights['_confidence']
