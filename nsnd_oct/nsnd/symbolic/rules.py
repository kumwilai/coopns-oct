"""
Differentiable symbolic rules for noise classification

Each rule implements soft logic for interpretable noise detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftRule:
    """Base class for differentiable symbolic rules"""

    def __init__(self, name: str, temperature: float = 1.0):
        self.name = name
        self.temperature = temperature

    def evaluate(self, features: dict) -> torch.Tensor:
        """
        Evaluate rule on features
        Returns: soft probability in [0, 1]
        """
        raise NotImplementedError

    def soft_equal(self, x: torch.Tensor, target: float, sigma: float = 0.2) -> torch.Tensor:
        """Soft equality: 1 if x ≈ target, 0 otherwise"""
        return torch.exp(-((x - target) ** 2) / (2 * sigma ** 2))

    def soft_greater(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft greater-than using sigmoid"""
        return torch.sigmoid((x - threshold) / self.temperature)

    def soft_less(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft less-than"""
        return torch.sigmoid((threshold - x) / self.temperature)

    def soft_and(self, *conditions: torch.Tensor) -> torch.Tensor:
        """Soft AND: product of probabilities"""
        result = conditions[0]
        for cond in conditions[1:]:
            result = result * cond
        return result

    def soft_or(self, *conditions: torch.Tensor) -> torch.Tensor:
        """Soft OR: probabilistic sum"""
        result = 1.0 - (1.0 - conditions[0])
        for cond in conditions[1:]:
            result = result * (1.0 - cond)
        return 1.0 - result


class SpeckleRule(SoftRule):
    """
    Rule: IF CV ≈ 1.0 AND high_kurtosis AND high_freq THEN multiplicative_speckle

    Rationale: Fully developed speckle has CV = sqrt(ENL)/ENL ≈ 1
    """

    def __init__(self):
        super().__init__("speckle", temperature=0.5)
        self.target_cv = 1.0
        self.min_kurtosis = 3.0  # Rayleigh/Gamma has kurtosis > Gaussian
        self.min_high_freq = 0.35

    def evaluate(self, features: dict) -> torch.Tensor:
        cv_map = features['cv_map']
        kurtosis = features['kurtosis']
        high_freq = features['high_freq_ratio']

        # Global CV close to 1.0
        cv_mean = cv_map.mean(dim=(-2, -1))  # [B]
        cv_condition = self.soft_equal(cv_mean, self.target_cv, sigma=0.3)

        # High kurtosis (heavy-tailed distribution)
        kurt_mean = kurtosis.mean(dim=(-2, -1))  # [B]
        kurt_condition = self.soft_greater(kurt_mean, self.min_kurtosis)

        # High-frequency energy ratio
        high_freq_mean = high_freq.mean(dim=(-2, -1))
        freq_condition = self.soft_greater(high_freq_mean, self.min_high_freq)

        # Combined evidence
        return self.soft_and(cv_condition, kurt_condition, freq_condition)  # [B]


class BandingRule(SoftRule):
    """
    Rule: IF high_vertical_frequency_power AND vertical_peakiness THEN banding_artifact

    Rationale: Horizontal banding shows up as vertical lines in Fourier spectrum
    """

    def __init__(self):
        super().__init__("banding", temperature=0.3)
        self.threshold_power = 0.15  # Normalized FFT power
        self.threshold_peak = 0.6  # Peakiness of vertical band

    def evaluate(self, features: dict) -> torch.Tensor:
        vertical_power = features['vertical_power']
        vertical_peak = features['vertical_peak_ratio']

        # High vertical frequency concentration - take mean over spatial dims
        power_mean = vertical_power.mean(dim=(-2, -1))  # [B]
        power_condition = self.soft_greater(power_mean, self.threshold_power)

        peak_mean = vertical_peak.mean(dim=(-2, -1))
        peak_condition = self.soft_greater(peak_mean, self.threshold_peak)

        return self.soft_and(power_condition, peak_condition)  # [B]


class GaussianRule(SoftRule):
    """
    Rule: IF uniform_variance AND low_kurtosis THEN gaussian_noise

    Rationale: Gaussian noise has kurtosis ≈ 3 and spatially uniform variance

    Note: Real OCT noise often has Gaussian component even when kurtosis > 3,
    so we add a baseline weight to ensure Gaussian denoiser is always available
    """

    def __init__(self, baseline_weight: float = 0.3):
        super().__init__("gaussian", temperature=0.5)
        self.target_kurtosis = 3.0
        self.max_var_std = 0.2  # Relaxed from 0.1 to 0.2
        self.baseline_weight = baseline_weight  # Minimum weight for Gaussian

    def evaluate(self, features: dict) -> torch.Tensor:
        kurtosis = features['kurtosis']
        local_std = features['local_std']

        # Kurtosis close to 3.0 (relaxed sigma from 0.5 to 1.0)
        kurt_mean = kurtosis.mean(dim=(-2, -1))  # [B]
        kurt_condition = self.soft_equal(kurt_mean, self.target_kurtosis, sigma=1.0)

        # Uniform variance (std of std is low)
        var_of_var = local_std.std(dim=(-2, -1))  # [B]
        uniform_condition = self.soft_less(var_of_var, self.max_var_std)

        # Combine conditions
        rule_score = self.soft_and(kurt_condition, uniform_condition)  # [B]

        # Add baseline: never go below baseline_weight
        # This ensures Gaussian denoiser is always available for real OCT noise
        baseline_tensor = torch.full_like(rule_score, self.baseline_weight)
        return torch.maximum(rule_score, baseline_tensor)  # [B]


class ShotNoiseRule(SoftRule):
    """
    Rule: IF depth_dependent_variance THEN shot_noise

    Rationale: Shot noise (Poisson) has variance = mean, decreases with depth in OCT
    """

    def __init__(self):
        super().__init__("shot", temperature=0.5)
        self.min_correlation = 0.4  # Correlation between variance and depth

    def evaluate(self, features: dict) -> torch.Tensor:
        local_std = features['local_std']
        depth_profile = features['depth_profile']

        # Compute correlation between variance and depth
        # Flatten spatial dimensions
        B = local_std.shape[0]
        std_flat = local_std.reshape(B, -1)
        depth_flat = depth_profile.reshape(B, -1)

        # Pearson correlation
        correlation = self._pearson_correlation(std_flat, depth_flat)  # [B]

        # High correlation indicates depth-dependent noise
        corr_condition = self.soft_greater(correlation, self.min_correlation)  # [B]

        # Ensure consistent shape
        if corr_condition.dim() > 1:
            corr_condition = corr_condition.squeeze()

        return corr_condition  # [B]

    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient"""
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)

        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = torch.sqrt((x_centered ** 2).sum(dim=-1) * (y_centered ** 2).sum(dim=-1))

        return numerator / (denominator + 1e-8)


class SymbolicReasoningEngine:
    """
    Forward reasoning engine that applies symbolic rules
    Returns interpretable noise composition
    """

    def __init__(self):
        self.rules = {
            'speckle': SpeckleRule(),
            'banding': BandingRule(),
            'gaussian': GaussianRule(),
            'shot': ShotNoiseRule(),
        }

    def infer(self, features: dict) -> dict:
        """
        Apply all rules and return normalized weights

        Args:
            features: Dictionary of computed features

        Returns:
            weights: Dictionary of noise component weights (sum to 1)
            confidences: Per-component confidence scores
        """
        raw_weights = {}
        for name, rule in self.rules.items():
            w = rule.evaluate(features)
            # Ensure all weights are [B] shaped (squeeze any extra dims)
            while w.dim() > 1:
                w = w.squeeze(-1)
            raw_weights[name] = w

        # Stack weights [B, N]
        weight_tensor = torch.stack([raw_weights[k] for k in self.rules.keys()], dim=-1)

        # Normalize to sum to 1
        total = weight_tensor.sum(dim=-1, keepdim=True) + 1e-8
        normalized = weight_tensor / total

        # Compute overall confidence (entropy-based)
        # High confidence = low entropy (peaked distribution)
        # Low confidence = high entropy (uniform distribution)
        entropy = -(normalized * torch.log(normalized + 1e-8)).sum(dim=-1)
        max_entropy = np.log(len(self.rules))
        confidence = 1.0 - (entropy / max_entropy)

        # Convert back to dict
        weights = {
            name: normalized[..., i]
            for i, name in enumerate(self.rules.keys())
        }
        weights['_confidence'] = confidence

        return weights

    def explain(self, weights: dict, threshold: float = 0.1) -> str:
        """
        Generate human-readable explanation

        Args:
            weights: Noise composition weights
            threshold: Minimum weight to report

        Returns:
            explanation: String describing dominant noise types
        """
        explanations = []

        if weights['speckle'].item() > threshold:
            explanations.append(
                f"Multiplicative speckle ({weights['speckle'].item()*100:.1f}%): "
                "Coherent interference pattern typical of OCT imaging"
            )

        if weights['banding'].item() > threshold:
            explanations.append(
                f"Horizontal banding ({weights['banding'].item()*100:.1f}%): "
                "Periodic artifacts likely from scanner electronics"
            )

        if weights['gaussian'].item() > threshold:
            explanations.append(
                f"Additive Gaussian noise ({weights['gaussian'].item()*100:.1f}%): "
                "Thermal/electronic noise from detector"
            )

        if weights['shot'].item() > threshold:
            explanations.append(
                f"Shot noise ({weights['shot'].item()*100:.1f}%): "
                "Photon counting noise, depth-dependent"
            )

        if not explanations:
            explanations.append("Mixed or unclassified noise composition")

        return "\n".join(explanations)
