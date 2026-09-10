"""
Neuro-symbolic reasoning with learnable predicates and rule weights.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePredicate(nn.Module):
    """Learnable thresholded predicate with soft truth values."""

    def __init__(self, init_threshold: float, greater: bool = True, init_scale: float = 0.5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(init_threshold)))
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        self.greater = bool(greater)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-3
        if self.greater:
            return torch.sigmoid((x - self.threshold) / scale)
        return torch.sigmoid((self.threshold - x) / scale)


class NeuroSymbolicReasoner(nn.Module):
    """
    Differentiable rule engine with learnable predicates and rule strengths.
    """

    def __init__(self):
        super().__init__()
        self.p_high_cv = LearnablePredicate(1.0, greater=True)
        self.p_high_kurtosis = LearnablePredicate(3.0, greater=True)
        self.p_low_kurtosis = LearnablePredicate(3.0, greater=False)
        self.p_uniform_var = LearnablePredicate(0.2, greater=False)
        self.p_vertical_power = LearnablePredicate(0.15, greater=True)
        self.p_vertical_peak = LearnablePredicate(0.6, greater=True)
        self.p_high_freq = LearnablePredicate(0.35, greater=True)
        self.p_depth_corr = LearnablePredicate(0.4, greater=True)

        self.rule_logits = nn.Parameter(torch.zeros(4))
        self.gaussian_bias = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        override_predicates: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict, Dict]:
        # Reduce feature maps to per-image scalars
        cv_mean = features['cv_map'].mean(dim=(-2, -1)).squeeze(-1)
        kurt_mean = features['kurtosis'].mean(dim=(-2, -1)).squeeze(-1)
        var_of_var = features['local_std'].std(dim=(-2, -1)).squeeze(-1)
        vert_power = features['vertical_power'].mean(dim=(-2, -1)).squeeze(-1)
        vert_peak = features['vertical_peak_ratio'].mean(dim=(-2, -1)).squeeze(-1)
        high_freq = features['high_freq_ratio'].mean(dim=(-2, -1)).squeeze(-1)
        depth_corr = self._depth_variance_correlation(features['local_std'], features['depth_profile'])

        if override_predicates is None:
            predicates = {
                'high_cv': self.p_high_cv(cv_mean),
                'high_kurtosis': self.p_high_kurtosis(kurt_mean),
                'low_kurtosis': self.p_low_kurtosis(kurt_mean),
                'uniform_variance': self.p_uniform_var(var_of_var),
                'banding_present': self.p_vertical_power(vert_power),
                'banding_peaky': self.p_vertical_peak(vert_peak),
                'high_freq': self.p_high_freq(high_freq),
                'depth_dependent': self.p_depth_corr(depth_corr),
            }
        else:
            predicates = {}
            for key in [
                'high_cv',
                'high_kurtosis',
                'low_kurtosis',
                'uniform_variance',
                'banding_present',
                'banding_peaky',
                'high_freq',
                'depth_dependent',
            ]:
                p = override_predicates[key]
                while p.dim() > 1:
                    p = p.squeeze(-1)
                predicates[key] = p.clamp(0.0, 1.0)

        # Rule scores
        rule_strength = F.softplus(self.rule_logits)
        speckle_score = rule_strength[0] * predicates['high_cv'] * predicates['high_kurtosis'] * predicates['high_freq']
        banding_score = rule_strength[1] * predicates['banding_present'] * predicates['banding_peaky']
        gaussian_score = rule_strength[2] * predicates['uniform_variance'] * predicates['low_kurtosis']
        gaussian_score = gaussian_score + F.softplus(self.gaussian_bias)
        shot_score = rule_strength[3] * predicates['depth_dependent']

        score_stack = torch.stack([speckle_score, banding_score, gaussian_score, shot_score], dim=-1)
        total = score_stack.sum(dim=-1, keepdim=True) + 1e-8
        normalized = score_stack / total

        # Confidence via entropy
        entropy = -(normalized * torch.log(normalized + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(normalized.shape[-1]), device=normalized.device))
        confidence = 1.0 - (entropy / max_entropy)

        weights = {
            'speckle': normalized[..., 0],
            'banding': normalized[..., 1],
            'gaussian': normalized[..., 2],
            'shot': normalized[..., 3],
            '_confidence': confidence,
        }

        rule_scores = {
            'speckle': speckle_score,
            'banding': banding_score,
            'gaussian': gaussian_score,
            'shot': shot_score,
        }

        scalars = {
            'cv_mean': cv_mean,
            'kurt_mean': kurt_mean,
            'var_of_var': var_of_var,
            'vert_power': vert_power,
            'vert_peak': vert_peak,
            'high_freq': high_freq,
            'depth_corr': depth_corr,
        }

        return weights, predicates, {'rule_scores': rule_scores, 'scalars': scalars}

    @staticmethod
    def logic_regularization(predicates: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encourage near-binary predicate truth values."""
        terms = [(p * (1.0 - p)).mean() for p in predicates.values()]
        return torch.stack(terms).mean()

    @staticmethod
    def _depth_variance_correlation(local_std: torch.Tensor, depth_profile: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation between variance and depth."""
        b = local_std.shape[0]
        std_flat = local_std.reshape(b, -1)
        depth_flat = depth_profile.reshape(b, -1)

        std_centered = std_flat - std_flat.mean(dim=-1, keepdim=True)
        depth_centered = depth_flat - depth_flat.mean(dim=-1, keepdim=True)

        numerator = (std_centered * depth_centered).sum(dim=-1)
        denominator = torch.sqrt((std_centered ** 2).sum(dim=-1) * (depth_centered ** 2).sum(dim=-1))
        return numerator / (denominator + 1e-8)
