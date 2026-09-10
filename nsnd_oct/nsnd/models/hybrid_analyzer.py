"""
Hybrid CNN-Symbolic Analyzer

Uses CNN for feature extraction + Symbolic rules for interpretable reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FFTBandingDetector(nn.Module):
    """
    Detect horizontal banding artifacts using FFT.
    Banding = horizontal stripes = energy along vertical axis in FFT.
    """

    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.15))
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        fft = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)

        center_h, center_w = H // 2, W // 2
        band_width = 1
        vertical_band = magnitude[:, :, :, center_w - band_width:center_w + band_width + 1].clone()
        vertical_band[:, :, center_h - 2:center_h + 3, :] = 0.0

        band_energy = vertical_band.sum(dim=[1, 2, 3])
        total_energy = magnitude.sum(dim=[1, 2, 3]) + 1e-6
        banding_ratio = band_energy / total_energy
        banding_score = torch.sigmoid(self.temperature * (banding_ratio - self.threshold))

        return banding_score


class ShotNoiseDetector(nn.Module):
    """
    Detect shot noise by measuring local mean-variance relationship.
    For Poisson: variance ∝ mean. Shot noise shows consistent var/mean ratio.
    """

    def __init__(self, window_size: int = 15):
        super().__init__()
        self.pool = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)
        self.threshold = nn.Parameter(torch.tensor(0.3))
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = self.pool(x)
        local_sq_mean = self.pool(x ** 2)
        local_var = torch.clamp(local_sq_mean - local_mean ** 2, min=1e-8)

        ratio = local_var / (local_mean + 1e-6)
        valid_mask = (local_mean > 0.05) & (local_mean < 0.95)
        valid_mask = valid_mask.float()

        ratio_masked = ratio * valid_mask
        valid_count = valid_mask.sum(dim=[2, 3])
        ratio_mean = ratio_masked.sum(dim=[2, 3]) / (valid_count + 1e-6)
        ratio_var = ((ratio_masked - ratio_mean.unsqueeze(-1).unsqueeze(-1)) ** 2 * valid_mask).sum(dim=[2, 3])
        ratio_var = ratio_var / (valid_count + 1e-6)
        ratio_std = torch.sqrt(ratio_var + 1e-8)
        ratio_cv = ratio_std / (ratio_mean + 1e-6)

        consistency = 1.0 / (1.0 + ratio_cv)
        min_valid = max(1.0, 0.05 * float(x.size(2) * x.size(3)))
        low_valid = valid_count < min_valid
        consistency = torch.where(low_valid, torch.zeros_like(consistency), consistency)
        consistency_score = torch.sigmoid(self.temperature * (consistency.squeeze(1) - self.threshold))

        # Correlation between local mean and variance (Poisson: high correlation).
        mean_flat = local_mean.view(local_mean.size(0), -1)
        var_flat = local_var.view(local_var.size(0), -1)
        mask_flat = valid_mask.view(valid_mask.size(0), -1)
        denom = mask_flat.sum(dim=1, keepdim=True) + 1e-6
        mean_mu = (mean_flat * mask_flat).sum(dim=1, keepdim=True) / denom
        var_mu = (var_flat * mask_flat).sum(dim=1, keepdim=True) / denom
        mean_centered = (mean_flat - mean_mu) * mask_flat
        var_centered = (var_flat - var_mu) * mask_flat
        cov = (mean_centered * var_centered).sum(dim=1) / (denom.squeeze(1) + 1e-6)
        mean_std = torch.sqrt((mean_centered ** 2).sum(dim=1) / (denom.squeeze(1) + 1e-6) + 1e-8)
        var_std = torch.sqrt((var_centered ** 2).sum(dim=1) / (denom.squeeze(1) + 1e-6) + 1e-8)
        corr = cov / (mean_std * var_std + 1e-6)
        corr = torch.clamp(corr, -1.0, 1.0)
        corr_score = torch.sigmoid(self.temperature * (corr - self.threshold))

        shot_score = 0.5 * consistency_score + 0.5 * corr_score

        return shot_score


class SpeckleStatsDetector(nn.Module):
    """
    Detect speckle noise using coefficient-of-variation statistics.
    """

    def __init__(self, window_size: int = 15):
        super().__init__()
        self.pool = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)
        self.threshold = nn.Parameter(torch.tensor(0.35))
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = self.pool(x)
        local_sq_mean = self.pool(x ** 2)
        local_var = torch.clamp(local_sq_mean - local_mean ** 2, min=1e-8)
        local_std = torch.sqrt(local_var + 1e-8)
        cv = local_std / (local_mean + 1e-6)

        valid_mask = local_mean > 0.05
        valid_mask = valid_mask.float()
        valid_count = valid_mask.sum(dim=[2, 3])

        cv_masked = cv * valid_mask
        cv_mean = cv_masked.sum(dim=[2, 3]) / (valid_count + 1e-6)

        min_valid = max(1.0, 0.05 * float(x.size(2) * x.size(3)))
        low_valid = valid_count < min_valid
        cv_mean = torch.where(low_valid, torch.zeros_like(cv_mean), cv_mean)

        speckle_score = torch.sigmoid(self.temperature * (cv_mean.squeeze(1) - self.threshold))
        return speckle_score


class SignalLevelModulator(nn.Module):
    """
    Modulate noise composition based on overall signal level.
    """

    def __init__(self, num_noise_types: int = 4):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_noise_types),
            nn.Tanh(),
        )
        self.strength = nn.Parameter(torch.tensor(0.15))

    def forward(self, weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        signal_level = x.mean(dim=[1, 2, 3], keepdim=True).view(x.size(0), 1)
        modulation = self.modulator(signal_level)
        adjusted = weights * (1.0 + self.strength * modulation)
        adjusted = torch.clamp(adjusted, min=1e-6)
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True)
        return adjusted


class HybridCNNSymbolicAnalyzer(nn.Module):
    """
    CNN features → Interpretable symbolic noise weights

    Best of both worlds:
    - CNN: Learn discriminative features from data
    - Symbolic: Interpretable, domain-aligned reasoning
    """

    def __init__(self, use_log_domain: bool = False):
        super().__init__()
        self.use_log_domain = bool(use_log_domain)

        # CNN feature encoder (learned)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        if self.use_log_domain:
            self.log_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
            )
            self.fusion = nn.Conv2d(256, 128, kernel_size=1)
        else:
            self.log_encoder = None
            self.fusion = None

        # Symbolic predicates (learned thresholds)
        self.speckle_threshold = nn.Parameter(torch.tensor(0.5))
        self.banding_threshold = nn.Parameter(torch.tensor(0.5))
        self.gaussian_threshold = nn.Parameter(torch.tensor(0.5))
        self.shot_threshold = nn.Parameter(torch.tensor(0.5))

        # Feature extractors for symbolic predicates
        self.speckle_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.banding_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.gaussian_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.shot_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fft_banding_detector = FFTBandingDetector()
        self.speckle_stats_detector = SpeckleStatsDetector()
        self.shot_stats_detector = ShotNoiseDetector()
        self.signal_modulator = SignalLevelModulator(num_noise_types=4)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.speckle_blend = nn.Parameter(torch.tensor(0.5))
        self.banding_blend = nn.Parameter(torch.tensor(0.5))
        self.shot_blend = nn.Parameter(torch.tensor(0.5))
        self.score_scale = nn.Parameter(torch.ones(4))
        self.score_bias = nn.Parameter(torch.zeros(4))
        self.score_affine = nn.Linear(4, 4)
        with torch.no_grad():
            self.score_affine.weight.copy_(torch.eye(4))
            self.score_affine.bias.zero_()

        # Continuous noise-parameter regressor (normalized to [0, 1])
        self.param_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )
        self.param_logvar_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        nn.init.zeros_(self.param_logvar_head[-1].weight)
        nn.init.zeros_(self.param_logvar_head[-1].bias)

    @staticmethod
    def _apply_threshold(score: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        score = score.clamp(1e-4, 1.0 - 1e-4)
        logit = torch.log(score / (1.0 - score))
        return torch.sigmoid(logit - threshold)

    def compute_predicates(
        self,
        x: torch.Tensor,
        noise_weights: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            signal_level = x.mean(dim=[1, 2, 3])
            predicates = {
                "high_speckle": noise_weights[:, 0] > 0.35,
                "banding_present": noise_weights[:, 1] > 0.15,
                "gaussian_dominant": noise_weights[:, 2] > 0.25,
                "high_shot": noise_weights[:, 3] > 0.25,
                "low_light": signal_level < 0.3,
                "high_confidence": confidence > 0.7,
                "mixed_noise": noise_weights.max(dim=1).values < 0.5,
            }
        return predicates

    def forward(
        self,
        x: torch.Tensor,
        return_predicates: bool = False,
        temperature: float | torch.Tensor | None = None,
        return_feature_map: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Args:
            x: [B, 1, H, W] noisy image

        Returns:
            weights: Dict with keys {speckle, banding, gaussian, shot, _confidence}
            features: dict with interpretable intermediate values
        """
        # Extract CNN features
        features_map = self.encoder(x)
        if self.use_log_domain and self.log_encoder is not None and self.fusion is not None:
            log_x = torch.log(torch.clamp(x, min=1e-6))
            features_log = self.log_encoder(log_x)
            features_map = self.fusion(torch.cat([features_map, features_log], dim=1))

        # Compute symbolic predicates (interpretable)
        speckle_score_cnn = self._apply_threshold(
            self.speckle_detector(features_map).squeeze(-1),
            self.speckle_threshold,
        )
        speckle_score_stats = self.speckle_stats_detector(x)
        speckle_alpha = torch.sigmoid(self.speckle_blend)
        speckle_score = speckle_alpha * speckle_score_cnn + (1.0 - speckle_alpha) * speckle_score_stats
        banding_score_cnn = self._apply_threshold(
            self.banding_detector(features_map).squeeze(-1),
            self.banding_threshold,
        )
        banding_score_fft = self.fft_banding_detector(x)
        banding_alpha = torch.sigmoid(self.banding_blend)
        banding_score = banding_alpha * banding_score_cnn + (1.0 - banding_alpha) * banding_score_fft
        gaussian_score = self._apply_threshold(
            self.gaussian_detector(features_map).squeeze(-1),
            self.gaussian_threshold,
        )
        shot_score_cnn = self._apply_threshold(
            self.shot_detector(features_map).squeeze(-1),
            self.shot_threshold,
        )
        shot_score_stats = self.shot_stats_detector(x)
        shot_alpha = torch.sigmoid(self.shot_blend)
        shot_score = shot_alpha * shot_score_cnn + (1.0 - shot_alpha) * shot_score_stats
        param_pred = self.param_head(features_map)
        param_logvar = self.param_logvar_head(features_map)

        # Symbolic reasoning: scores → weights via normalization
        scores = torch.stack([speckle_score, banding_score, gaussian_score, shot_score], dim=-1)
        score_scale = torch.clamp(self.score_scale, min=0.1)
        scores = scores * score_scale + self.score_bias
        scores = self.score_affine(scores)
        if temperature is None:
            temp = self.temperature
        else:
            temp = self.temperature * temperature
        temp = torch.clamp(temp, min=1e-3)
        weights_tensor = F.softmax(scores / temp, dim=-1)
        weights_tensor = self.signal_modulator(weights_tensor, x)

        # Compute confidence (inverse of entropy)
        entropy = -(weights_tensor * torch.log(weights_tensor + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(4.0, device=x.device))
        confidence = 1.0 - (entropy / max_entropy)

        # Format for NSND compatibility
        weights_dict = {
            'speckle': weights_tensor[:, 0],
            'banding': weights_tensor[:, 1],
            'gaussian': weights_tensor[:, 2],
            'shot': weights_tensor[:, 3],
            '_confidence': confidence,
        }

        # Return interpretable features
        features = {
            'speckle_score': speckle_score,
            'speckle_score_stats': speckle_score_stats,
            'banding_score': banding_score,
            'banding_score_fft': banding_score_fft,
            'gaussian_score': gaussian_score,
            'shot_score': shot_score,
            'shot_score_stats': shot_score_stats,
            'raw_scores': scores,
            'score_scale': score_scale.detach(),
            'score_bias': self.score_bias.detach(),
            'symbolic_weights': weights_tensor,
            'param_pred': param_pred,
            'param_logvar': param_logvar,
        }
        if return_feature_map:
            features["feature_map"] = features_map
        if return_predicates:
            features["predicates"] = self.compute_predicates(x, weights_tensor, confidence)

        return weights_dict, features

    def explain(self, x: torch.Tensor) -> str:
        """Generate human-readable explanation"""
        weights_dict, features = self.forward(x)

        report = "="*50 + "\n"
        report += "Hybrid CNN-Symbolic Noise Analysis\n"
        report += "="*50 + "\n\n"

        # Extract weights
        w = torch.stack([
            weights_dict['speckle'][0],
            weights_dict['banding'][0],
            weights_dict['gaussian'][0],
            weights_dict['shot'][0],
        ]).detach().cpu().numpy()

        names = ['Speckle', 'Banding', 'Gaussian', 'Shot']

        report += "Detected Composition:\n"
        for i, name in enumerate(names):
            report += f"├── {name}: {w[i]*100:.1f}%\n"

        report += "\nSymbolic Scores (Interpretable):\n"
        for key in ['speckle_score', 'banding_score', 'gaussian_score', 'shot_score']:
            score = features[key][0].item()
            report += f"├── {key}: {score:.3f}\n"

        dominant = names[w.argmax()]
        confidence = weights_dict['_confidence'][0].item()
        report += f"\nDominant Noise: {dominant}\n"
        report += f"Confidence: {confidence:.3f}\n"
        report += "="*50

        return report
