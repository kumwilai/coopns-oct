#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising v2 - Enhanced Architecture (CPU Optimized)

Key improvements over v1:
1. Skip connections from noisy input to corrector
2. Multi-scale predicate evaluation and correction
3. Joint training of backbone + corrector
4. Memory-efficient CPU implementation

Stays truly neuro-symbolic:
- Symbolic predicates define clinical quality
- Failure maps guide WHERE corrections apply
- Conflict resolution with priority rules
- Interpretable corrections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from PIL import Image
import time
import sys
import gc
import os
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, 'nsnd_oct')

# =============================================================================
# CPU OPTIMIZATION SETTINGS
# =============================================================================
# Set optimal number of threads for CPU
_NUM_THREADS = min(8, os.cpu_count() or 4)
torch.set_num_threads(_NUM_THREADS)
torch.set_num_interop_threads(2)  # For parallel ops between operations

# Disable gradient computation for validation by default
torch.set_grad_enabled(True)

# Use optimized CPU backend
torch.backends.mkl.is_available() and None  # MKL check
torch.backends.mkldnn.is_available() and None  # oneDNN check


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def clear_memory():
    """Force garbage collection for CPU."""
    gc.collect()


# =============================================================================
# SYMBOLIC PREDICATES (CPU Optimized)
# =============================================================================

class SobelFilter(nn.Module):
    """Fixed Sobel filter for edge detection - CPU optimized."""

    def __init__(self):
        super().__init__()
        # Stacked kernels for single conv2d call (2 output channels)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_combined = torch.cat([sobel_x, sobel_y], dim=0)  # [2, 1, 3, 3]
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_combined', sobel_combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single conv2d for both directions (faster on CPU)
        grad = F.conv2d(x, self.sobel_combined, padding=1)  # [B, 2, H, W]
        gx, gy = grad[:, 0:1], grad[:, 1:2]
        return torch.sqrt(gx.square() + gy.square() + 1e-8)

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both gx and gy for reuse."""
        grad = F.conv2d(x, self.sobel_combined, padding=1)
        return grad[:, 0:1], grad[:, 1:2]

    def vertical(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.sobel_y, padding=1).abs()


class GaussianSmooth(nn.Module):
    """Fixed Gaussian smoothing - CPU optimized with separable convolution."""

    def __init__(self, sigma: float = 1.0, kernel_size: int = 5):
        super().__init__()
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # Separable convolution is faster: 2*k ops instead of k^2
        self.register_buffer('kernel_h', kernel_1d.view(1, 1, 1, kernel_size))
        self.register_buffer('kernel_v', kernel_1d.view(1, 1, kernel_size, 1))
        self.padding_h = (0, kernel_size // 2)
        self.padding_v = (kernel_size // 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Separable convolution: horizontal then vertical (faster than 2D)
        x = F.conv2d(F.pad(x, (self.padding_h[1], self.padding_h[1], 0, 0), mode='replicate'),
                     self.kernel_h)
        x = F.conv2d(F.pad(x, (0, 0, self.padding_v[0], self.padding_v[0]), mode='replicate'),
                     self.kernel_v)
        return x


class SymbolicPredicates(nn.Module):
    """
    Clinical quality predicates for OCT images.

    P1: Boundary Detectability - layer boundaries must be detectable
    P2: Layer Contrast - sufficient contrast between layers
    P3: Noise Reduction - noise should be reduced in flat regions
    P4: Structure Preservation - anatomical structures preserved

    Speed optimized: shared computations are cached and reused.
    """

    def __init__(self, learnable_thresholds: bool = True):
        super().__init__()
        self.sobel = SobelFilter()
        self.smooth = GaussianSmooth(sigma=1.0)

        # Priority weights for conflict resolution (P1 > P5 > P4 > P2 > P6 > P3)
        self.priorities = {'P1': 1.0, 'P5': 0.95, 'P4': 0.9, 'P2': 0.7, 'P6': 0.6, 'P3': 0.5}

        # Pre-compute priority tensor for resolve_conflicts
        # Order: P1, P2, P3, P4, P5, P6 (indices 0, 1, 2, 3, 4, 5)
        priority_values = torch.tensor([1.0, 0.7, 0.5, 0.9, 0.95, 0.6]).view(-1, 1, 1, 1, 1)
        self.register_buffer('priority_tensor', priority_values)

        # Learnable thresholds with sigmoid to keep in (0, 1)
        # Initialize with logit values - VERY STRICT to force corrector usage
        # logit(x) = log(x / (1-x))
        if learnable_thresholds:
            # VERY STRICT thresholds - backbone alone should NOT pass all
            # This forces the corrector to improve the output
            self.threshold_logits = nn.ParameterDict({
                'P1': nn.Parameter(torch.tensor(2.94)),   # sigmoid(2.94) ≈ 0.95 (very strict)
                'P2': nn.Parameter(torch.tensor(2.94)),   # sigmoid(2.94) ≈ 0.95 (very strict)
                'P3': nn.Parameter(torch.tensor(2.20)),   # sigmoid(2.20) ≈ 0.90
                'P4': nn.Parameter(torch.tensor(2.94)),   # sigmoid(2.94) ≈ 0.95 (very strict)
                'P5': nn.Parameter(torch.tensor(2.20)),   # sigmoid(2.20) ≈ 0.90
                'P6': nn.Parameter(torch.tensor(2.20)),   # sigmoid(2.20) ≈ 0.90
            })
            self.learnable_thresholds = True
        else:
            # Fixed thresholds (very strict for demonstration)
            self.register_buffer('threshold_P1', torch.tensor(0.95))
            self.register_buffer('threshold_P2', torch.tensor(0.95))
            self.register_buffer('threshold_P3', torch.tensor(0.90))
            self.register_buffer('threshold_P4', torch.tensor(0.95))
            self.register_buffer('threshold_P5', torch.tensor(0.90))
            self.register_buffer('threshold_P6', torch.tensor(0.90))
            self.learnable_thresholds = False

        # Temperature for soft thresholding during training (higher = sharper)
        self.soft_threshold_temp = 10.0

    def get_threshold(self, name: str) -> torch.Tensor:
        """Get threshold value for a predicate."""
        if self.learnable_thresholds:
            return torch.sigmoid(self.threshold_logits[name])
        else:
            return getattr(self, f'threshold_{name}')

    def get_thresholds_dict(self) -> Dict[str, float]:
        """Get all thresholds as a dictionary (for logging)."""
        return {name: self.get_threshold(name).item() for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']}

    def check_passed(self, score: torch.Tensor, name: str, soft: bool = False) -> bool:
        """Check if score passes threshold. Use soft=True during training for gradient flow."""
        threshold = self.get_threshold(name)
        if soft:
            # Soft pass probability (differentiable)
            return torch.sigmoid((score - threshold) * self.soft_threshold_temp)
        else:
            # Hard threshold for inference
            return score.item() > threshold.item()

    def _compute_shared(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict:
        """Pre-compute shared operations - CPU optimized."""
        shared = {}

        # noisy_smooth used in P1 and P4
        with torch.no_grad():
            shared['noisy_smooth'] = self.smooth(noisy)
            shared['edges_ref'] = self.sobel.vertical(shared['noisy_smooth'])

        # P1/P3: edges of denoised - single combined sobel pass
        gx, gy = self.sobel.forward_both(denoised)
        shared['edges_den_vert'] = gy.abs()  # P1: vertical edges
        with torch.no_grad():
            shared['edges_den_mag'] = torch.sqrt(gx.square() + gy.square() + 1e-8)
            shared['flat_mask'] = (shared['edges_den_mag'] < 0.1).float()
            shared['flat_mask_sum'] = shared['flat_mask'].sum().item()

        # P2/P3: avg_pool - compute denoised**2 once, reuse for both pool sizes
        denoised_sq = denoised.square()
        shared['mean_7'] = F.avg_pool2d(denoised, 7, stride=1, padding=3)
        shared['mean_5'] = F.avg_pool2d(denoised, 5, stride=1, padding=2)
        shared['sq_mean_7'] = F.avg_pool2d(denoised_sq, 7, stride=1, padding=3)
        shared['sq_mean_5'] = F.avg_pool2d(denoised_sq, 5, stride=1, padding=2)
        del denoised_sq

        # P2: noisy local std (no grad)
        with torch.no_grad():
            noisy_sq = noisy.square()
            mean_noisy = F.avg_pool2d(noisy, 7, stride=1, padding=3)
            sq_mean_noisy = F.avg_pool2d(noisy_sq, 7, stride=1, padding=3)
            del noisy_sq
            shared['local_std_noisy'] = (sq_mean_noisy - mean_noisy.square()).clamp_(min=1e-6).sqrt_()

        # P4: noisy_smooth pooling (no grad)
        with torch.no_grad():
            shared['mu2'] = F.avg_pool2d(shared['noisy_smooth'], 5, stride=1, padding=2)
            shared['sigma2_sq'] = F.avg_pool2d(shared['noisy_smooth'] ** 2, 5, stride=1, padding=2) - shared['mu2'] ** 2

        return shared

    def compute_P1(self, noisy: torch.Tensor, denoised: torch.Tensor,
                   shared: Optional[Dict] = None) -> Dict:
        """P1: Boundary Detectability"""
        if shared is None:
            # Fallback for standalone calls
            with torch.no_grad():
                noisy_smooth = self.smooth(noisy)
                edges_ref = self.sobel.vertical(noisy_smooth)
            edges_den = self.sobel.vertical(denoised)
        else:
            edges_ref = shared['edges_ref']
            edges_den = shared['edges_den_vert']

        # Normalize - use detached max for stability
        with torch.no_grad():
            ref_max = edges_ref.max() + 1e-6
            den_max = max(edges_den.max().item(), 0.01)
        edges_ref_norm = edges_ref / ref_max
        edges_den_norm = edges_den / den_max

        # Boundary mask (top 30% strongest edges)
        boundary_mask = (edges_ref_norm > 0.3).float()

        # Preservation ratio
        preservation = (edges_den_norm / (edges_ref_norm + 1e-4)).clamp(0, 2)
        mask_sum = boundary_mask.sum() + 1e-6
        mean_preservation = (preservation * boundary_mask).sum() / mask_sum

        # Score
        score = (1 - torch.abs(mean_preservation - 1) * 0.5).clamp(0, 1)

        # Failure map
        weakness = (1 - edges_den_norm / (edges_ref_norm + 1e-4)).clamp(0, 1)
        failure_map = boundary_mask * weakness

        return {'score': score, 'passed': self.check_passed(score, 'P1'),
                'failure_map': failure_map}

    def compute_P2(self, noisy: torch.Tensor, denoised: torch.Tensor,
                   shared: Optional[Dict] = None) -> Dict:
        """P2: Layer Contrast - compare to smoothed reference (not raw noisy)."""
        # Compute local std of denoised
        if shared is None:
            mean = F.avg_pool2d(denoised, 7, stride=1, padding=3)
            sq_mean = F.avg_pool2d(denoised ** 2, 7, stride=1, padding=3)
        else:
            mean = shared['mean_7']
            sq_mean = shared['sq_mean_7']

        local_std = torch.sqrt((sq_mean - mean ** 2).clamp(min=1e-6))

        # Compare to SMOOTHED noisy reference (removes noise, keeps true contrast)
        with torch.no_grad():
            if shared is not None and 'noisy_smooth' in shared:
                noisy_smooth = shared['noisy_smooth']
            else:
                noisy_smooth = self.smooth(noisy)
            mean_ref = F.avg_pool2d(noisy_smooth, 7, stride=1, padding=3)
            sq_mean_ref = F.avg_pool2d(noisy_smooth ** 2, 7, stride=1, padding=3)
            local_std_ref = torch.sqrt((sq_mean_ref - mean_ref ** 2).clamp(min=1e-6))
            # Normalize reference std
            ref_max = local_std_ref.max() + 1e-6

        # Contrast preservation ratio (now comparing apples to apples)
        contrast_ratio = (local_std / ref_max) / (local_std_ref / ref_max + 1e-4)
        # Score: how well contrast is preserved (1.0 = same as reference)
        score = (1 - torch.abs(contrast_ratio - 1.0) * 0.5).mean().clamp(0, 1)
        failure_map = (1 - contrast_ratio).clamp(0, 1)

        return {'score': score, 'passed': self.check_passed(score, 'P2'),
                'failure_map': failure_map}

    def compute_P3(self, noisy: torch.Tensor, denoised: torch.Tensor,
                   shared: Optional[Dict] = None) -> Dict:
        """P3: Noise Reduction"""
        if shared is None:
            with torch.no_grad():
                edges = self.sobel(denoised)
                flat_mask = (edges < 0.1).float()
                flat_mask_sum = flat_mask.sum().item()
        else:
            flat_mask = shared['flat_mask']
            flat_mask_sum = shared['flat_mask_sum']

        if flat_mask_sum < 100:
            score = (denoised.mean() * 0.0 + 1.0).clamp(0, 1)
            return {'score': score,
                    'passed': True,
                    'failure_map': torch.zeros_like(denoised).detach()}

        # Use cached pooling if available
        if shared is None:
            mean = F.avg_pool2d(denoised, 5, stride=1, padding=2)
            sq_mean = F.avg_pool2d(denoised ** 2, 5, stride=1, padding=2)
        else:
            mean = shared['mean_5']
            sq_mean = shared['sq_mean_5']

        local_var = (sq_mean - mean ** 2).clamp(min=0)
        flat_var = (local_var * flat_mask).sum() / (flat_mask_sum + 1e-6)
        score = torch.exp(-flat_var * 50).clamp(0, 1)
        failure_map = flat_mask * local_var * 10

        return {'score': score, 'passed': self.check_passed(score, 'P3'),
                'failure_map': failure_map.clamp(0, 1)}

    def compute_P4(self, noisy: torch.Tensor, denoised: torch.Tensor,
                   shared: Optional[Dict] = None) -> Dict:
        """P4: Structure Preservation"""
        if shared is None:
            with torch.no_grad():
                noisy_smooth = self.smooth(noisy)
                mu2 = F.avg_pool2d(noisy_smooth, 5, stride=1, padding=2)
                sigma2_sq = F.avg_pool2d(noisy_smooth ** 2, 5, stride=1, padding=2) - mu2 ** 2
            mu1 = F.avg_pool2d(denoised, 5, stride=1, padding=2)
            sq_mean_1 = F.avg_pool2d(denoised ** 2, 5, stride=1, padding=2)
        else:
            noisy_smooth = shared['noisy_smooth']
            mu2 = shared['mu2']
            sigma2_sq = shared['sigma2_sq']
            mu1 = shared['mean_5']
            sq_mean_1 = shared['sq_mean_5']

        sigma1_sq = sq_mean_1 - mu1 ** 2
        sigma12 = F.avg_pool2d(denoised * noisy_smooth, 5, stride=1, padding=2) - mu1 * mu2

        C3 = 0.03 ** 2 / 2
        structure = (sigma12 + C3) / (torch.sqrt((sigma1_sq.clamp(min=0) * sigma2_sq.clamp(min=0)).clamp(min=1e-8)) + C3)

        score = structure.mean().clamp(0, 1)
        failure_map = (1 - structure).clamp(0, 1)

        return {'score': score, 'passed': self.check_passed(score, 'P4'),
                'failure_map': failure_map}

    def compute_P5(self, noisy: torch.Tensor, denoised: torch.Tensor,
                   shared: Optional[Dict] = None) -> Dict:
        """P5: Boundary Sharpness - sharp layer boundaries for segmentation.

        Measures gradient magnitude at detected boundary locations.
        Higher sharpness = better for downstream layer segmentation.
        """
        # Get edge locations from smoothed noisy (reference boundaries)
        with torch.no_grad():
            if shared is not None and 'noisy_smooth' in shared:
                noisy_smooth = shared['noisy_smooth']
            else:
                noisy_smooth = self.smooth(noisy)
            edges_ref = self.sobel(noisy_smooth)
            # Boundary mask: strong edges in reference
            boundary_mask = (edges_ref > edges_ref.mean() + edges_ref.std()).float()
            mask_sum = boundary_mask.sum() + 1e-6

        # Compute gradient magnitude of denoised at boundary locations
        edges_den = self.sobel(denoised)

        # Sharpness: gradient strength at boundaries (normalized)
        with torch.no_grad():
            ref_strength = (edges_ref * boundary_mask).sum() / mask_sum
        den_strength = (edges_den * boundary_mask).sum() / mask_sum

        # Score: ratio of denoised sharpness to reference (clamped)
        sharpness_ratio = (den_strength / (ref_strength + 1e-6)).clamp(0, 2)
        score = (1 - torch.abs(sharpness_ratio - 1) * 0.5).clamp(0, 1)

        # Failure map: where boundaries are not sharp enough
        with torch.no_grad():
            edge_weakness = (edges_ref - edges_den).clamp(min=0)
        failure_map = boundary_mask * edge_weakness / (edges_ref.max() + 1e-6)

        return {'score': score, 'passed': self.check_passed(score, 'P5'),
                'failure_map': failure_map.clamp(0, 1)}

    def compute_P6(self, noisy: torch.Tensor, denoised: torch.Tensor,
                   shared: Optional[Dict] = None) -> Dict:
        """P6: Speckle Suppression - reduce OCT speckle noise.

        Measures speckle contrast REDUCTION from noisy to denoised.
        Good denoising should reduce speckle contrast in homogeneous regions.
        """
        # Find homogeneous regions using SMOOTHED noisy reference (more reliable)
        with torch.no_grad():
            if shared is not None and 'noisy_smooth' in shared:
                noisy_smooth = shared['noisy_smooth']
            else:
                noisy_smooth = self.smooth(noisy)
            edges_ref = self.sobel(noisy_smooth)
            # Homogeneous mask: regions with low edges in reference (use median-based threshold)
            edge_threshold = edges_ref.mean() * 0.5  # Adaptive threshold
            homogeneous_mask = (edges_ref < edge_threshold).float()
            mask_sum = homogeneous_mask.sum()

        if mask_sum < 100:
            # Not enough homogeneous regions - return perfect score
            score = (denoised.mean() * 0.0 + 1.0).clamp(0, 1)
            return {'score': score, 'passed': True,
                    'failure_map': torch.zeros_like(denoised).detach()}

        # Compute local stats for denoised
        if shared is not None:
            mean_den = shared['mean_5']
            sq_mean_den = shared['sq_mean_5']
        else:
            mean_den = F.avg_pool2d(denoised, 5, stride=1, padding=2)
            sq_mean_den = F.avg_pool2d(denoised ** 2, 5, stride=1, padding=2)

        local_std_den = torch.sqrt((sq_mean_den - mean_den ** 2).clamp(min=1e-6))
        # Speckle contrast of denoised (lower is better)
        speckle_den = local_std_den / (mean_den.abs() + 0.01)  # Larger epsilon for stability

        # Compute reference speckle contrast for comparison
        with torch.no_grad():
            mean_noisy = F.avg_pool2d(noisy, 5, stride=1, padding=2)
            sq_mean_noisy = F.avg_pool2d(noisy ** 2, 5, stride=1, padding=2)
            local_std_noisy = torch.sqrt((sq_mean_noisy - mean_noisy ** 2).clamp(min=1e-6))
            speckle_noisy = local_std_noisy / (mean_noisy.abs() + 0.01)
            # Reference speckle in homogeneous regions
            ref_speckle = (speckle_noisy * homogeneous_mask).sum() / (mask_sum + 1e-6)

        # Denoised speckle in homogeneous regions
        den_speckle = (speckle_den * homogeneous_mask).sum() / (mask_sum + 1e-6)

        # Score: ratio of speckle reduction (1.0 = no change, >1.0 = reduced speckle)
        # We want den_speckle < ref_speckle
        reduction_ratio = ref_speckle / (den_speckle + 1e-6)
        # Score: higher reduction = better, capped at 1.0
        score = (1 - torch.exp(-reduction_ratio + 1)).clamp(0, 1)

        # Failure map: regions where speckle is still high
        with torch.no_grad():
            failure_map = homogeneous_mask * (speckle_den / (speckle_noisy + 1e-6)).clamp(0, 1)

        return {'score': score, 'passed': self.check_passed(score, 'P6'),
                'failure_map': failure_map.clamp(0, 1)}

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor,
                return_failure_maps: bool = True) -> Dict:
        """Evaluate all 6 predicates with shared computation."""
        # Pre-compute shared operations once
        shared = self._compute_shared(noisy, denoised)

        # Evaluate all 6 predicates using cached values
        p1 = self.compute_P1(noisy, denoised, shared)
        p2 = self.compute_P2(noisy, denoised, shared)
        p3 = self.compute_P3(noisy, denoised, shared)
        p4 = self.compute_P4(noisy, denoised, shared)
        p5 = self.compute_P5(noisy, denoised, shared)
        p6 = self.compute_P6(noisy, denoised, shared)

        # Free shared cache
        del shared

        all_passed = (p1['passed'] and p2['passed'] and p3['passed'] and
                      p4['passed'] and p5['passed'] and p6['passed'])

        result = {
            'P1': {'score': p1['score'], 'passed': p1['passed']},
            'P2': {'score': p2['score'], 'passed': p2['passed']},
            'P3': {'score': p3['score'], 'passed': p3['passed']},
            'P4': {'score': p4['score'], 'passed': p4['passed']},
            'P5': {'score': p5['score'], 'passed': p5['passed']},
            'P6': {'score': p6['score'], 'passed': p6['passed']},
            'all_passed': all_passed,
        }

        if return_failure_maps:
            result['P1']['failure_map'] = p1['failure_map'].detach()
            result['P2']['failure_map'] = p2['failure_map'].detach()
            result['P3']['failure_map'] = p3['failure_map'].detach()
            result['P4']['failure_map'] = p4['failure_map'].detach()
            result['P5']['failure_map'] = p5['failure_map'].detach()
            result['P6']['failure_map'] = p6['failure_map'].detach()

        return result

    def resolve_conflicts(self, failure_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Resolve conflicts using priority rules: P1 > P5 > P4 > P2 > P6 > P3
        Returns exclusive masks for each predicate.

        Speed-optimized: vectorized operations, pre-registered priority tensor.
        """
        # Stack failure maps once for vectorized computation
        names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        stacked = torch.stack([failure_maps[n].detach() for n in names], dim=0)  # [6, B, 1, H, W]
        weighted = stacked * self.priority_tensor  # Use pre-registered buffer

        # Find max priority at each pixel
        max_priority, max_idx = weighted.max(dim=0)  # [B, 1, H, W]
        active = max_priority > 0.01

        # Create exclusive masks using argmax (vectorized)
        exclusive = {}
        order_to_idx = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3, 'P5': 4, 'P6': 5}

        for name in ['P1', 'P5', 'P4', 'P2', 'P6', 'P3']:  # Priority order
            idx = order_to_idx[name]
            is_winner = (max_idx == idx) & active
            exclusive[name] = (is_winner.float() * stacked[idx]).detach()

        del stacked, weighted, max_priority, max_idx, active
        return exclusive


# =============================================================================
# HETEROGENEOUS CORRECTOR ARCHITECTURE
# Each group uses specialized architecture suited to its task:
# - Edge (P1, P5): Offset-Modulated Conv + Edge-Guided Attention
# - Texture (P2, P4, P6): Multi-Scale + Local Window Attention
# - Smooth (P3): Frequency-Aware + Edge-Preserving
# =============================================================================

class FailureGatedFusion(nn.Module):
    """
    Gate backbone features using failure map intensity.
    High failure → gate opens → more feature influence.
    """

    def __init__(self, feature_channels: int, out_channels: int = 16):
        super().__init__()
        self.adapt = nn.Conv2d(feature_channels, out_channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(1, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.adapt.weight, gain=0.5)
        nn.init.xavier_uniform_(self.gate[0].weight, gain=0.5)

    def forward(self, features: torch.Tensor, failure_map: torch.Tensor) -> torch.Tensor:
        f = self.adapt(features)
        if failure_map.shape[2:] != f.shape[2:]:
            failure_map = F.interpolate(failure_map, size=f.shape[2:], mode='bilinear', align_corners=False)
        g = self.gate(failure_map)
        return f * g


# =============================================================================
# EDGE CORRECTOR V2 (P1, P5): Multi-Scale Dilated + Directional Edge Enhancement
# STRONGER ARCHITECTURE for boundary detectability and sharpness
# =============================================================================

class MultiScaleDilatedBlock(nn.Module):
    """
    Multi-scale dilated convolution block for large receptive field.
    OPTIMIZED: 3 scales instead of 4, depthwise separable for speed.

    Receptive fields: 3, 5, 9 pixels (dilations 1, 2, 4)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_ch = out_channels // 3

        # Depthwise separable dilated convolutions (faster than regular)
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
        )
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, mid_ch + out_channels % 3, 1, bias=False),  # Handle remainder
        )

        # Fusion
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.conv_d1(x)
        d2 = self.conv_d2(x)
        d4 = self.conv_d4(x)
        out = torch.cat([d1, d2, d4], dim=1)
        return F.leaky_relu(self.bn(out), 0.2, inplace=True)


class DirectionalEdgeModule(nn.Module):
    """
    Directional edge enhancement module.
    OPTIMIZED: Smaller kernels, fused operations.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Horizontal edge branch (1x3 + 3x1 = asymmetric for horizontal edges)
        self.h_conv = nn.Conv2d(channels, channels // 2, (1, 3), padding=(0, 1), bias=False)

        # Vertical edge branch (3x1)
        self.v_conv = nn.Conv2d(channels, channels // 2, (3, 1), padding=(1, 0), bias=False)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Lightweight channel attention
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = F.leaky_relu(self.h_conv(x), 0.2, inplace=True)
        v = F.leaky_relu(self.v_conv(x), 0.2, inplace=True)

        combined = torch.cat([h, v], dim=1)
        fused = self.fusion(combined)

        att = self.gate(fused).view(B, C, 1, 1)
        return fused * att + x


class EdgeSpatialAttention(nn.Module):
    """
    Spatial attention guided by edge gradients.
    Focuses processing on boundary regions.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Attention from edge magnitude + features
        self.att_conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # Compute edge magnitude
        gx = F.conv2d(image, self.sobel_x, padding=1)
        gy = F.conv2d(image, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        # Spatial attention
        combined = torch.cat([features, edge_mag], dim=1)
        attention = self.att_conv(combined)

        return features * (1 + attention)  # Boost edges, don't suppress non-edges


class EdgeCorrector(nn.Module):
    """
    STRONGER Edge Corrector for P1 (Boundary Detectability) and P5 (Sharpness).

    Architecture improvements:
    1. Multi-scale dilated convolutions (RF: 3-17 pixels)
    2. Directional edge processing (H/V separately for OCT layers)
    3. Edge-guided spatial attention
    4. Residual dense connections
    5. Larger capacity (~120K params vs ~50K)

    Input: denoised(1) + noisy(1) + lambda(1) + enc1(16) = 19 channels
    """

    def __init__(self, in_channels: int = 19):
        super().__init__()

        # Initial feature extraction with multi-scale dilated conv
        self.ms_block1 = MultiScaleDilatedBlock(in_channels, 64)
        self.ms_block2 = MultiScaleDilatedBlock(64, 64)

        # Directional edge enhancement
        self.dir_edge = DirectionalEdgeModule(64)

        # Edge-guided spatial attention
        self.edge_att = EdgeSpatialAttention(64)

        # Dense connection fusion
        self.dense_fusion = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 1, bias=False),  # Fuse ms_block2 + dir_edge
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Refinement with residual
        self.refine = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output with sharpening bias
        self.output = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )

        # Learnable strength (initialized conservatively)
        self.strength = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self._strength_scale = 0.15  # Max 15% correction

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature extraction
        f1 = self.ms_block1(x)
        f2 = self.ms_block2(f1)

        # Directional edge enhancement
        f_dir = self.dir_edge(f2)

        # Dense fusion
        f_dense = self.dense_fusion(torch.cat([f2, f_dir], dim=1))

        # Edge-guided attention (use denoised for edge detection)
        f_att = self.edge_att(f_dense, denoised)

        # Refinement
        f_ref = self.refine(f_att)

        # Output
        out = self.output(f_ref)

        bounded_strength = self.strength.sigmoid() * self._strength_scale
        return out * bounded_strength


# =============================================================================
# TEXTURE CORRECTOR (P2, P4, P6): Multi-Scale + Local Window Attention
# Designed for: Layer contrast, structure preservation, speckle suppression
# Key insight: Texture needs multi-scale context and cross-region relationships
# =============================================================================

class LocalWindowAttention(nn.Module):
    """
    Efficient local window attention (Swin-style).
    Computes attention within local windows for CPU efficiency.

    CPU Optimizations:
    - Pre-computed relative position bias (avoids repeated indexing)
    - Fused operations where possible
    - Reduced memory allocations
    """

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection - combined for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute relative position index and register as buffer
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # Pre-computed window size squared (used multiple times)
        self.ws_sq = window_size * window_size

    def _get_rel_pos_bias(self) -> torch.Tensor:
        """Get relative position bias, pre-shaped for attention addition."""
        # Shape: [num_heads, ws*ws, ws*ws]
        return self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.ws_sq, self.ws_sq, self.num_heads).permute(2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape
        nH, nW = Hp // ws, Wp // ws
        num_windows = nH * nW

        # Reshape to windows: [B, C, H, W] -> [B*num_windows, ws*ws, C]
        # Optimized: fewer intermediate shapes
        x = x.view(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * num_windows, self.ws_sq, C)

        # QKV projection
        qkv = self.qkv(x)  # [B*nW, ws*ws, 3*C]
        qkv = qkv.view(B * num_windows, self.ws_sq, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*nW, heads, ws*ws, head_dim]
        q, k, v = qkv.unbind(0)  # More efficient than indexing

        # Scaled dot-product attention
        # Use torch.baddbmm for fused multiply-add where beneficial
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative position bias (pre-computed shape)
        attn = attn + self._get_rel_pos_bias().unsqueeze(0)

        # Softmax and attention application
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape: [B*nW, heads, ws*ws, head_dim] -> [B*nW, ws*ws, C]
        out = out.transpose(1, 2).reshape(B * num_windows, self.ws_sq, C)
        out = self.proj(out)

        # Reshape back: [B*num_windows, ws*ws, C] -> [B, C, H, W]
        out = out.view(B, nH, nW, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module for global context.
    OPTIMIZED: Fewer scales (2 instead of 4), efficient pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_sizes: list = [1, 4]):
        super().__init__()
        branch_ch = out_channels // 2

        # Global context (1x1)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_ch, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Local context (4x4)
        self.local_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fusion
        self.fusion = nn.Conv2d(in_channels + out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        g = F.interpolate(self.global_pool(x), size=(H, W), mode='nearest')
        l = F.interpolate(self.local_pool(x), size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x, g, l], dim=1)
        return F.leaky_relu(self.bn(self.fusion(out)), 0.2, inplace=True)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Learns to emphasize important feature channels.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.att(x)


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention for relating features at different resolutions.
    Helps understand layer boundaries that span multiple scales.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Query from fine scale, Key/Value from coarse scale
        self.q_conv = nn.Conv2d(channels, channels // 2, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels // 2, 1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, 1, bias=False)

        self.scale = (channels // 2) ** -0.5
        self.out_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fine: High-resolution features [B, C, H, W]
            coarse: Low-resolution features [B, C, H/2, W/2]
        """
        B, C, H, W = fine.shape

        # Upsample coarse to match fine
        coarse_up = F.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=False)

        # Q from fine, K/V from coarse
        q = self.q_conv(fine)  # [B, C/2, H, W]
        k = self.k_conv(coarse_up)  # [B, C/2, H, W]
        v = self.v_conv(coarse_up)  # [B, C, H, W]

        # Compute attention (spatial-wise, averaged over channels)
        q_flat = q.view(B, -1, H * W)  # [B, C/2, H*W]
        k_flat = k.view(B, -1, H * W)  # [B, C/2, H*W]

        # Channel-wise attention map
        attn = torch.bmm(q_flat.transpose(1, 2), k_flat) * self.scale  # [B, H*W, H*W] - too big!

        # Instead, use simpler spatial attention
        attn_spatial = (q * k).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        attn_spatial = torch.sigmoid(attn_spatial)

        # Apply attention to values
        out = v * attn_spatial
        return self.out_conv(out) + fine  # Residual


class EnhancedMultiScaleFusion(nn.Module):
    """
    Enhanced multi-scale fusion.
    OPTIMIZED: 3 scales, single conv per scale, efficient pooling.
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        ch = out_channels // 3

        # Scale 1: Full resolution (single conv)
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Scale 2: 1/2 resolution
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Scale 4: 1/4 resolution
        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels, ch + out_channels % 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch + out_channels % 3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Lightweight channel attention
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        f1 = self.scale1(x)

        x2 = F.avg_pool2d(x, 2)
        f2 = F.interpolate(self.scale2(x2), size=(H, W), mode='bilinear', align_corners=False)

        x4 = F.avg_pool2d(x2, 2)
        f4 = F.interpolate(self.scale4(x4), size=(H, W), mode='bilinear', align_corners=False)

        fused = torch.cat([f1, f2, f4], dim=1)
        att = self.gate(fused).view(B, -1, 1, 1)
        return fused * att


class ContrastEnhancementModule(nn.Module):
    """
    Explicit contrast enhancement module.
    OPTIMIZED: Use avg_pool instead of large grouped conv.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Contrast modulation (simplified)
        self.modulation = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use avg_pool for local mean (much faster than grouped conv)
        local_mean = F.avg_pool2d(x, 5, stride=1, padding=2)

        # Deviation from mean (contrast signal)
        deviation = x - local_mean

        # Learn contrast modulation
        combined = torch.cat([x, deviation], dim=1)
        modulation = self.modulation(combined)

        return x + deviation * modulation


class TextureCorrector(nn.Module):
    """
    STRONGER Texture Corrector for P2 (Layer Contrast), P4 (Structure), P6 (Speckle).

    Architecture improvements:
    1. Pyramid Pooling for global context (1x1, 2x2, 4x4, 8x8)
    2. 4-scale multi-scale fusion (1x, 2x, 4x, 8x)
    3. Channel attention (SE blocks)
    4. Explicit contrast enhancement module
    5. Local window attention with larger window
    6. Larger capacity (~150K params vs ~80K)

    Input: denoised(1) + noisy(1) + lambda(1) + enc1(16) + enc2(16) = 35 channels
    """

    def __init__(self, in_channels: int = 35):
        super().__init__()

        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Pyramid pooling for global context
        self.ppm = PyramidPoolingModule(64, 64, pool_sizes=[1, 2, 4, 8])

        # Enhanced multi-scale fusion
        self.multi_scale = EnhancedMultiScaleFusion(64, 64)

        # Contrast enhancement
        self.contrast = ContrastEnhancementModule(64)

        # Local window attention (window=8 for speed, still effective)
        self.local_attention = LocalWindowAttention(dim=64, window_size=8, num_heads=4)

        # Channel attention
        self.ca = ChannelAttention(64)

        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )

        # Learnable strength
        self.strength = nn.Parameter(torch.tensor(0.0))
        self._strength_scale = 0.15

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial projection
        h = self.proj(x)

        # Global context via PPM
        h = self.ppm(h)

        # Multi-scale fusion
        h = self.multi_scale(h) + h  # Residual

        # Contrast enhancement
        h = self.contrast(h)

        # Local attention
        h = h + self.local_attention(h)  # Residual

        # Channel attention
        h = self.ca(h)

        # Refinement
        h = self.refine(h)

        # Output
        out = self.output(h)

        bounded_strength = self.strength.sigmoid() * self._strength_scale
        return out * bounded_strength


# =============================================================================
# SMOOTH CORRECTOR (P3): Frequency-Aware + Edge-Preserving
# Designed for: Noise reduction in flat regions
# Key insight: Separate high-freq (noise/edges) from low-freq (structure)
# =============================================================================

class FrequencyDecomposition(nn.Module):
    """
    Learnable frequency decomposition.
    Separates image into low-frequency (structure) and high-frequency (detail/noise).
    """

    def __init__(self, channels: int = 1):
        super().__init__()

        # Low-pass filter (learnable Gaussian-like)
        self.low_pass = nn.Sequential(
            nn.Conv2d(channels, 8, 5, padding=2, bias=False),
            nn.Conv2d(8, channels, 5, padding=2, bias=False),
        )

        # Initialize as Gaussian blur
        with torch.no_grad():
            # Create Gaussian kernel
            sigma = 1.5
            k = 5
            x = torch.arange(k).float() - k // 2
            gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            gaussian_2d = gaussian_1d.view(-1, 1) @ gaussian_1d.view(1, -1)

            # Initialize first conv
            for i in range(8):
                self.low_pass[0].weight[i, 0] = gaussian_2d + torch.randn_like(gaussian_2d) * 0.01
            # Initialize second conv to average
            self.low_pass[1].weight.fill_(1.0 / 8.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low_freq = self.low_pass(x)
        high_freq = x - low_freq
        return low_freq, high_freq


class EdgePreservingFusion(nn.Module):
    """
    Fuses low and high frequency components while preserving edges.
    Uses edge map to decide where to preserve high frequencies.
    """

    def __init__(self, channels: int = 32):
        super().__init__()

        # Edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Edge-aware gating
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, channels, 1),
            nn.Sigmoid()
        )

    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, low_features: torch.Tensor, high_features: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:
        # Compute edge map
        edges = self.compute_edges(image)

        # Edge-aware gating (preserve high freq at edges)
        edge_gate = self.edge_gate(edges)

        # Fuse with edge awareness
        combined = torch.cat([low_features, high_features * edge_gate, edges], dim=1)
        return self.fusion(combined)


# =============================================================================
# P2 CONTRAST CORRECTOR: CLAHE-inspired adaptive contrast enhancement
# =============================================================================

class ContrastCorrector(nn.Module):
    """
    Specialized corrector for P2 (Layer Contrast).

    Key insight: P2 measures local standard deviation preservation.
    If P2 is low, denoising reduced contrast too much.

    Solution: CLAHE-inspired adaptive contrast enhancement
    - Compute local statistics at multiple scales
    - Learn per-pixel contrast gain
    - Apply histogram-stretching-like enhancement
    """

    def __init__(self, in_channels: int = 19):
        super().__init__()

        # Multi-scale local statistics
        self.scales = [5, 9, 15]  # Different neighborhood sizes

        # Feature extraction from input + statistics
        # in_channels + 3 scales * 2 (mean, std) = in_channels + 6
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels + 6, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Contrast gain predictor (per-pixel gain factor)
        self.gain_predictor = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),  # Gain >= 0
        )

        # Learnable reference statistics matching
        self.stats_matcher = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),
        )

        # Learnable strength
        self.strength = nn.Parameter(torch.tensor(0.0))
        self._strength_scale = 0.20  # Allow stronger contrast correction

    def compute_local_stats(self, x: torch.Tensor, kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local mean and std."""
        padding = kernel_size // 2
        local_mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        local_sq_mean = F.avg_pool2d(x ** 2, kernel_size, stride=1, padding=padding)
        local_std = (local_sq_mean - local_mean ** 2).clamp(min=1e-6).sqrt()
        return local_mean, local_std

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Compute multi-scale statistics of denoised
        stats_list = []
        for ks in self.scales:
            mean, std = self.compute_local_stats(denoised, ks)
            stats_list.extend([mean, std])

        # Only use the middle scale for the main stats
        local_mean = stats_list[2]  # scale=9 mean
        local_std = stats_list[3]   # scale=9 std

        # Concatenate input with statistics
        stats_tensor = torch.cat(stats_list, dim=1)
        combined = torch.cat([x, stats_tensor], dim=1)

        # Extract features
        features = self.feature_net(combined)

        # Predict per-pixel contrast gain
        gain = self.gain_predictor(features)  # [B, 1, H, W], >= 0
        gain = gain.clamp(0.5, 2.0)  # Limit gain range

        # Predict adjustment
        adjustment = self.stats_matcher(features)

        # Apply contrast enhancement: boost deviation from local mean
        deviation = denoised - local_mean
        enhanced_deviation = deviation * gain

        # Correction = enhanced - original = (gain - 1) * deviation
        correction = enhanced_deviation - deviation + adjustment * local_std

        bounded_strength = self.strength.sigmoid() * self._strength_scale
        return correction * bounded_strength


# =============================================================================
# P5 SHARPNESS CORRECTOR: Unsharp masking + high-frequency boost
# =============================================================================

class SharpnessCorrector(nn.Module):
    """
    Specialized corrector for P5 (Boundary Sharpness).

    Key insight: P5 measures edge gradient strength at boundaries.
    If P5 is low, boundaries are too smooth/blurry.

    Solution: Learnable unsharp masking + edge enhancement
    - Extract high-frequency (edge) components
    - Learn edge-aware sharpening kernel
    - Boost edges without amplifying noise
    """

    def __init__(self, in_channels: int = 19):
        super().__init__()

        # Learnable Laplacian-like kernels for edge detection
        self.edge_detector = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.Conv2d(8, 4, 3, padding=1, bias=False),
        )

        # Initialize with edge-detection-like weights
        with torch.no_grad():
            # First layer: multiple edge orientations
            self.edge_detector[0].weight.data.normal_(0, 0.1)
            # Second layer: combine edges
            self.edge_detector[1].weight.data.normal_(0, 0.1)

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels + 4, 48, 3, padding=1, bias=False),  # +4 for edge channels
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Sharpening strength predictor (per-pixel)
        self.sharpen_predictor = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),  # 0-1 range
        )

        # Edge-aware high-frequency extractor
        self.hf_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        # Learnable strength
        self.strength = nn.Parameter(torch.tensor(0.0))
        self._strength_scale = 0.25  # Allow stronger sharpening

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Compute learnable edge map
        edges = self.edge_detector(denoised)

        # Compute high-frequency component (unsharp mask style)
        # Low-pass filter
        blurred = F.avg_pool2d(
            F.pad(denoised, (2, 2, 2, 2), mode='replicate'),
            5, stride=1
        )
        high_freq = denoised - blurred  # High-frequency = original - blurred

        # Concatenate input with edge info
        combined = torch.cat([x, edges], dim=1)

        # Extract features
        features = self.feature_net(combined)

        # Predict per-pixel sharpening strength
        sharpen_map = self.sharpen_predictor(features)  # [B, 1, H, W]

        # Learnable high-frequency enhancement
        hf_enhanced = self.hf_extractor(high_freq)

        # Sharpening = boost high frequency at edge locations
        correction = sharpen_map * (high_freq + hf_enhanced * 0.5)

        bounded_strength = self.strength.sigmoid() * self._strength_scale
        return correction * bounded_strength


class SmoothCorrector(nn.Module):
    """
    Specialized corrector for smoothing-related predicates (P3).

    Architecture:
    - Frequency decomposition (separate noise from structure)
    - Process low/high frequency separately
    - Edge-preserving fusion (don't over-smooth edges)

    Input: denoised(1) + noisy(1) + failure_map(1) + enc2(16) = 19 channels
    """

    def __init__(self, in_channels: int = 19):
        super().__init__()

        # Frequency decomposition of input
        self.freq_decompose = FrequencyDecomposition(channels=1)

        # Process low-frequency (structure preservation)
        self.low_freq_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Process high-frequency (noise suppression while preserving edges)
        self.high_freq_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Edge-preserving fusion
        self.edge_fusion = EdgePreservingFusion(channels=32)

        # Final output
        self.output = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )

        # Learnable strength
        self.strength = nn.Parameter(torch.tensor(0.3))
        self._strength_scale = 0.06  # More conservative for smoothing

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Decompose denoised into frequency components
        low_freq, high_freq = self.freq_decompose(denoised)

        # Create frequency-aware inputs
        # For low-freq processing: emphasize structure
        x_low = x.clone()
        x_low[:, 0:1] = low_freq  # Replace denoised channel with low-freq

        # For high-freq processing: emphasize detail/noise
        x_high = x.clone()
        x_high[:, 0:1] = high_freq  # Replace denoised channel with high-freq

        # Process separately
        low_features = self.low_freq_net(x_low)
        high_features = self.high_freq_net(x_high)

        # Edge-preserving fusion
        fused = self.edge_fusion(low_features, high_features, denoised)

        # Output
        out = self.output(fused)

        bounded_strength = self.strength.sigmoid() * self._strength_scale
        return out * bounded_strength


# =============================================================================
# HETEROGENEOUS FEATURE-AWARE CORRECTOR
# Combines all specialized correctors
# =============================================================================

class FeatureAwareCorrector(nn.Module):
    """
    Heterogeneous feature-aware corrector with specialized architectures per group.

    Groups and their specialized architectures:
    - Edge (P1, P5): Offset-modulated conv + edge-guided attention
    - Texture (P2, P4, P6): Multi-scale + local window attention
    - Smooth (P3): Frequency-aware + edge-preserving

    Each architecture is designed for its specific task while using
    backbone features (enc1, enc2) for informed correction.
    """

    def __init__(self, enc1_channels: int = 64, enc2_channels: int = 128):
        super().__init__()

        # Feature adapters with gating
        self.gate_enc1 = FailureGatedFusion(enc1_channels, out_channels=16)
        self.gate_enc2 = FailureGatedFusion(enc2_channels, out_channels=16)

        # Specialized correctors
        # Edge: denoised(1) + noisy(1) + failure(1) + enc1(16) = 19 channels
        self.edge_corrector = EdgeCorrector(in_channels=19)

        # Texture: denoised(1) + noisy(1) + failure(1) + enc1(16) + enc2(16) = 35 channels
        self.texture_corrector = TextureCorrector(in_channels=35)

        # Smooth: denoised(1) + noisy(1) + failure(1) + enc2(16) = 19 channels
        self.smooth_corrector = SmoothCorrector(in_channels=19)

        # Predicate to group mapping
        self.pred_to_group = {
            'P1': 'edge', 'P5': 'edge',
            'P2': 'texture', 'P4': 'texture', 'P6': 'texture',
            'P3': 'smooth'
        }

        # Priority order
        self.order = ['P1', 'P5', 'P2', 'P4', 'P6', 'P3']

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                exclusive_masks: Dict[str, torch.Tensor],
                backbone_features: Dict[str, torch.Tensor],
                return_individual: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply feature-aware corrections.

        Args:
            denoised: Backbone output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            exclusive_masks: Conflict-resolved failure maps
            backbone_features: {'enc1': [B, 64, H, W], 'enc2': [B, 128, H/2, W/2]}
            return_individual: Return individual corrections for loss

        Returns:
            corrected: Final corrected output
            info: Correction statistics
        """
        B, C, H, W = denoised.shape
        info = {}

        enc1 = backbone_features.get('enc1')  # [B, 64, H, W]
        enc2 = backbone_features.get('enc2')  # [B, 128, H/2, W/2]

        # Pre-compute mask sums
        mask_sums = {name: exclusive_masks[name].sum().item()
                     for name in self.order if name in exclusive_masks}

        has_corrections = any(s >= 1 for s in mask_sums.values())
        if not has_corrections:
            if return_individual:
                info['individual_corrections'] = {}
                info['individual_masks'] = {}
                info['group_corrections'] = {}
            return denoised, info

        # Initialize accumulators
        total_correction = torch.zeros_like(denoised)

        if return_individual:
            individual_corrections = {}
            individual_masks = {}
            group_corrections = {'edge': [], 'texture': [], 'smooth': []}

        # Process each predicate
        for name in self.order:
            if name not in exclusive_masks:
                continue

            mask = exclusive_masks[name]
            mask_sum = mask_sums.get(name, 0)

            if mask_sum < 1:
                continue

            group = self.pred_to_group[name]

            # Prepare features based on group and call specialized corrector
            if group == 'edge':
                # Edge corrector: Offset-modulated conv + edge-guided attention
                # Uses enc1 (fine edges) + needs denoised for edge detection
                f1 = self.gate_enc1(enc1, mask)
                if f1.shape[2:] != denoised.shape[2:]:
                    f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)
                x = torch.cat([denoised, noisy, mask, f1], dim=1)
                correction = self.edge_corrector(x, denoised)  # Pass denoised for edge detection
                del f1, x

            elif group == 'texture':
                # Texture corrector: Multi-scale + local window attention
                # Uses enc1 + enc2 (textures + local structure)
                f1 = self.gate_enc1(enc1, mask)
                if f1.shape[2:] != denoised.shape[2:]:
                    f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)
                f2 = self.gate_enc2(enc2, mask)
                f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)
                x = torch.cat([denoised, noisy, mask, f1, f2], dim=1)
                correction = self.texture_corrector(x)
                del f1, f2, x

            else:  # smooth
                # Smooth corrector: Frequency-aware + edge-preserving
                # Uses enc2 (distinguish texture from noise) + needs denoised for freq decomposition
                f2 = self.gate_enc2(enc2, mask)
                f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)
                x = torch.cat([denoised, noisy, mask, f2], dim=1)
                correction = self.smooth_corrector(x, denoised)  # Pass denoised for freq decomposition
                del f2, x

            # Apply mask to correction
            masked_correction = correction * mask
            total_correction = total_correction + masked_correction

            if return_individual:
                individual_corrections[name] = masked_correction
                individual_masks[name] = mask
                group_corrections[group].append((name, masked_correction.detach()))
            else:
                del masked_correction

            del correction
            info[name] = {'applied': True, 'mask_sum': mask_sum, 'group': group}

        # Store for loss computation
        if return_individual:
            info['individual_corrections'] = individual_corrections
            info['individual_masks'] = individual_masks
            info['group_corrections'] = group_corrections

        corrected = (denoised + total_correction).clamp(0, 1)
        del total_correction

        return corrected, info


# Legacy corrector for backwards compatibility
class MultiScaleCorrector(FeatureAwareCorrector):
    """Alias for FeatureAwareCorrector for backwards compatibility."""
    pass


# =============================================================================
# FAST/LITE CORRECTORS (CPU Optimized)
# Simplified architectures for ~3x speedup with minimal quality loss
# =============================================================================

class EdgeCorrectorLite(nn.Module):
    """
    Fast edge corrector - replaces expensive offset-modulated conv with
    standard convolutions + simple edge-aware weighting.
    """

    def __init__(self, in_channels: int = 19):
        super().__init__()

        # Simple conv stack (no offset prediction)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # Output
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),
        )

        self.strength = nn.Parameter(torch.tensor(0.3))
        self._strength_scale = 0.08

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        out = self.out(h)
        return out * (self.strength.sigmoid() * self._strength_scale)


class TextureCorrectorLite(nn.Module):
    """
    Fast texture corrector - replaces multi-scale + attention with
    dilated convolutions for multi-scale receptive field.
    """

    def __init__(self, in_channels: int = 35):
        super().__init__()

        # Dilated convs for multi-scale without resize
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=4, dilation=4, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        # Fuse + output
        self.fuse = nn.Conv2d(32 * 3, 32, 1, bias=False)
        self.out = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Tanh(),
        )

        self.strength = nn.Parameter(torch.tensor(0.3))
        self._strength_scale = 0.08

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        h = self.fuse(torch.cat([h1, h2, h3], dim=1))
        out = self.out(h)
        return out * (self.strength.sigmoid() * self._strength_scale)


class SmoothCorrectorLite(nn.Module):
    """
    Fast smooth corrector - replaces learnable frequency decomposition with
    fixed Gaussian + simple edge-preserving blend.
    """

    def __init__(self, in_channels: int = 19):
        super().__init__()

        # Fixed Gaussian for low-pass (separable for speed)
        kernel_size = 5
        sigma = 1.5
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        self.register_buffer('gauss_h', kernel_1d.view(1, 1, 1, kernel_size))
        self.register_buffer('gauss_v', kernel_1d.view(1, 1, kernel_size, 1))

        # Simple correction network
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, 3, padding=1, bias=False),  # +2 for low/high freq
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Tanh(),
        )

        self.strength = nn.Parameter(torch.tensor(0.3))
        self._strength_scale = 0.06

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Fast separable Gaussian
        pad = 2
        low = F.conv2d(F.pad(denoised, (pad, pad, 0, 0), mode='replicate'), self.gauss_h)
        low = F.conv2d(F.pad(low, (0, 0, pad, pad), mode='replicate'), self.gauss_v)
        high = denoised - low

        # Concat with input
        h = torch.cat([x, low, high], dim=1)
        out = self.net(h)
        return out * (self.strength.sigmoid() * self._strength_scale)


class AdaptiveLambdaPredictorLite(nn.Module):
    """
    Fast lambda predictor - smaller network, fewer layers.
    """

    def __init__(self):
        super().__init__()

        # Lighter shared features (7 input channels)
        self.shared = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Single conv per head (no separate feature extraction)
        self.edge_head = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus(),
        )
        self.texture_head = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus(),
        )
        self.smooth_head = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus(),
        )

        self.scale = nn.Parameter(torch.tensor(0.0))
        self.lambda_cap = 2.0

    def forward(self, denoised: torch.Tensor,
                failure_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, C, H, W = denoised.shape

        # Stack failure maps
        p1 = failure_maps.get('P1', torch.zeros(B, 1, H, W, device=denoised.device))
        p2 = failure_maps.get('P2', torch.zeros(B, 1, H, W, device=denoised.device))
        p3 = failure_maps.get('P3', torch.zeros(B, 1, H, W, device=denoised.device))
        p4 = failure_maps.get('P4', torch.zeros(B, 1, H, W, device=denoised.device))
        p5 = failure_maps.get('P5', torch.zeros(B, 1, H, W, device=denoised.device))
        p6 = failure_maps.get('P6', torch.zeros(B, 1, H, W, device=denoised.device))

        x = torch.cat([p1, p2, p3, p4, p5, p6, denoised], dim=1)
        feat = self.shared(x)

        scale = torch.sigmoid(self.scale)
        return {
            'edge': torch.clamp(self.edge_head(feat) * scale, 0, self.lambda_cap),
            'texture': torch.clamp(self.texture_head(feat) * scale, 0, self.lambda_cap),
            'smooth': torch.clamp(self.smooth_head(feat) * scale, 0, self.lambda_cap),
        }


class AdaptiveCorrectorWithLambdaLite(nn.Module):
    """
    Fast corrector using lite sub-correctors.
    """

    def __init__(self, enc1_channels: int = 64, enc2_channels: int = 128):
        super().__init__()

        # Simpler feature adapters (fewer output channels)
        self.adapt_enc1 = nn.Conv2d(enc1_channels, 8, 1, bias=False)
        self.adapt_enc2 = nn.Conv2d(enc2_channels, 8, 1, bias=False)

        # Lite correctors with reduced input channels
        # Edge: denoised(1) + noisy(1) + lambda(1) + enc1(8) = 11 channels
        self.edge_corrector = EdgeCorrectorLite(in_channels=11)

        # Texture: denoised(1) + noisy(1) + lambda(1) + enc1(8) + enc2(8) = 19 channels
        self.texture_corrector = TextureCorrectorLite(in_channels=19)

        # Smooth: denoised(1) + noisy(1) + lambda(1) + enc2(8) = 11 channels
        self.smooth_corrector = SmoothCorrectorLite(in_channels=11)

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                lambda_maps: Dict[str, torch.Tensor],
                backbone_features: Dict[str, torch.Tensor],
                return_individual: bool = False) -> Tuple[torch.Tensor, Dict]:
        B, C, H, W = denoised.shape
        info = {}

        enc1 = backbone_features.get('enc1')
        enc2 = backbone_features.get('enc2')

        # Adapt features
        f1 = self.adapt_enc1(enc1)
        if f1.shape[2:] != (H, W):
            f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)

        f2 = self.adapt_enc2(enc2)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

        # Get lambdas
        lambda_edge = lambda_maps.get('edge', torch.zeros_like(denoised))
        lambda_texture = lambda_maps.get('texture', torch.zeros_like(denoised))
        lambda_smooth = lambda_maps.get('smooth', torch.zeros_like(denoised))

        # Edge correction
        edge_in = torch.cat([denoised, noisy, lambda_edge, f1], dim=1)
        raw_edge = self.edge_corrector(edge_in, denoised)
        edge_correction = lambda_edge * raw_edge

        # Texture correction
        texture_in = torch.cat([denoised, noisy, lambda_texture, f1, f2], dim=1)
        raw_texture = self.texture_corrector(texture_in)
        texture_correction = lambda_texture * raw_texture

        # Smooth correction
        smooth_in = torch.cat([denoised, noisy, lambda_smooth, f2], dim=1)
        raw_smooth = self.smooth_corrector(smooth_in, denoised)
        smooth_correction = lambda_smooth * raw_smooth

        # Combine
        total = edge_correction + texture_correction + smooth_correction
        corrected = (denoised + total).clamp(0, 1)

        info['correction_magnitude'] = total.abs().mean().item()

        if return_individual:
            info['individual_corrections'] = {
                'edge': edge_correction,
                'texture': texture_correction,
                'smooth': smooth_correction,
            }
            info['lambda_maps'] = lambda_maps

        return corrected, info


# =============================================================================
# ADAPTIVE LAMBDA PREDICTOR
# Learns per-pixel correction strength based on failure severity
# =============================================================================

class AdaptiveLambdaPredictor(nn.Module):
    """
    Predicts per-pixel lambda maps for each corrector group.

    Instead of binary masks (correct or not), predicts continuous correction
    strength based on:
    - Failure map intensity (how badly did predicate fail?)
    - Local image features (context-aware)

    Key insight: Different regions need different correction strengths.
    - Severe failure → high λ → strong correction
    - Mild failure → low λ → gentle correction
    - Passing region → λ ≈ 0 → no correction

    The lambda predictor is trained with a DIFFERENT objective than correctors:
    - Correctors: Minimize specialized losses (edge/texture/smooth)
    - Lambda predictor: Maximize final quality + predicate satisfaction
    """

    def __init__(self, num_groups: int = 3):
        super().__init__()

        # Input: 6 failure maps (P1-P6) + denoised (1) = 7 channels
        # Output: 3 lambda maps (edge, texture, smooth)

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Conv2d(7, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Per-group lambda heads
        # Each head predicts a per-pixel lambda map for its group

        # Edge group (P1 only): Boundary detection
        self.edge_head = nn.Sequential(
            nn.Conv2d(32 + 1, 16, 3, padding=1, bias=False),  # +1 for P1 map
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),
        )

        # Contrast group (P2): Layer contrast enhancement - NEW
        self.contrast_head = nn.Sequential(
            nn.Conv2d(32 + 1, 16, 3, padding=1, bias=False),  # +1 for P2 map
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),
        )

        # Sharpness group (P5): Boundary sharpening - NEW
        self.sharpness_head = nn.Sequential(
            nn.Conv2d(32 + 1, 16, 3, padding=1, bias=False),  # +1 for P5 map
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),
        )

        # Texture group (P4, P6): Structure preservation, speckle
        self.texture_head = nn.Sequential(
            nn.Conv2d(32 + 2, 16, 3, padding=1, bias=False),  # +2 for P4, P6 maps
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),
        )

        # Smooth group (P3): Noise reduction
        self.smooth_head = nn.Sequential(
            nn.Conv2d(32 + 1, 16, 3, padding=1, bias=False),  # +1 for P3 map
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),
        )

        # Learnable scaling factors
        self.edge_scale = nn.Parameter(torch.tensor(-2.0))
        self.contrast_scale = nn.Parameter(torch.tensor(-1.5))  # Less conservative for P2
        self.sharpness_scale = nn.Parameter(torch.tensor(-1.5))  # Less conservative for P5
        self.texture_scale = nn.Parameter(torch.tensor(-3.0))
        self.smooth_scale = nn.Parameter(torch.tensor(-3.0))

        # Maximum lambda caps (reduced to prevent PSNR degradation)
        self.lambda_cap_edge = 0.10
        self.lambda_cap_contrast = 0.10
        self.lambda_cap_sharpness = 0.10
        self.lambda_cap_texture = 0.05
        self.lambda_cap_smooth = 0.08

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, denoised: torch.Tensor,
                failure_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict per-pixel lambda maps for each corrector group.

        Args:
            denoised: Backbone output [B, 1, H, W]
            failure_maps: Dict of failure maps {'P1': [B,1,H,W], ...}

        Returns:
            Dict of lambda maps {'edge': [B,1,H,W], 'texture': [B,1,H,W], 'smooth': [B,1,H,W]}
        """
        B, C, H, W = denoised.shape

        # Stack all failure maps
        p1 = failure_maps.get('P1', torch.zeros(B, 1, H, W, device=denoised.device))
        p2 = failure_maps.get('P2', torch.zeros(B, 1, H, W, device=denoised.device))
        p3 = failure_maps.get('P3', torch.zeros(B, 1, H, W, device=denoised.device))
        p4 = failure_maps.get('P4', torch.zeros(B, 1, H, W, device=denoised.device))
        p5 = failure_maps.get('P5', torch.zeros(B, 1, H, W, device=denoised.device))
        p6 = failure_maps.get('P6', torch.zeros(B, 1, H, W, device=denoised.device))

        # Concat: [failure_maps (6), denoised (1)] = 7 channels
        x = torch.cat([p1, p2, p3, p4, p5, p6, denoised], dim=1)

        # Shared features
        shared_feat = self.shared(x)

        # Edge lambda: informed by P1 (boundary detection)
        edge_input = torch.cat([shared_feat, p1], dim=1)
        raw_edge = self.edge_head(edge_input)
        lambda_edge = torch.clamp(raw_edge * torch.sigmoid(self.edge_scale), 0, self.lambda_cap_edge)

        # Contrast lambda: informed by P2 (layer contrast) - NEW
        contrast_input = torch.cat([shared_feat, p2], dim=1)
        raw_contrast = self.contrast_head(contrast_input)
        lambda_contrast = torch.clamp(raw_contrast * torch.sigmoid(self.contrast_scale), 0, self.lambda_cap_contrast)

        # Sharpness lambda: informed by P5 (boundary sharpness) - NEW
        sharpness_input = torch.cat([shared_feat, p5], dim=1)
        raw_sharpness = self.sharpness_head(sharpness_input)
        lambda_sharpness = torch.clamp(raw_sharpness * torch.sigmoid(self.sharpness_scale), 0, self.lambda_cap_sharpness)

        # Texture lambda: informed by P4, P6 (structure, speckle)
        texture_input = torch.cat([shared_feat, p4, p6], dim=1)
        raw_texture = self.texture_head(texture_input)
        lambda_texture = torch.clamp(raw_texture * torch.sigmoid(self.texture_scale), 0, self.lambda_cap_texture)

        # Smooth lambda: informed by P3
        smooth_input = torch.cat([shared_feat, p3], dim=1)
        raw_smooth = self.smooth_head(smooth_input)
        lambda_smooth = torch.clamp(raw_smooth * torch.sigmoid(self.smooth_scale), 0, self.lambda_cap_smooth)

        return {
            'edge': lambda_edge,
            'contrast': lambda_contrast,
            'sharpness': lambda_sharpness,
            'texture': lambda_texture,
            'smooth': lambda_smooth,
        }


class AdaptiveCorrectorWithLambda(nn.Module):
    """
    Feature-aware corrector with adaptive lambda maps.

    Key difference from FeatureAwareCorrector:
    - Instead of binary masks, uses continuous lambda maps
    - Lambda maps control per-pixel correction strength
    - Lambda maps are ALSO passed to correctors as input (WHERE info)
    - correction = λ_map * raw_correction

    This allows:
    1. Corrector knows WHERE to focus corrections (lambda as input)
    2. Gradients flow through λ to the lambda predictor (lambda as weight)
    """

    def __init__(self, enc1_channels: int = 64, enc2_channels: int = 128):
        super().__init__()

        # Feature adapters
        self.adapt_enc1 = nn.Conv2d(enc1_channels, 16, 1, bias=False)
        self.adapt_enc2 = nn.Conv2d(enc2_channels, 16, 1, bias=False)
        nn.init.xavier_uniform_(self.adapt_enc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.adapt_enc2.weight, gain=0.5)

        # Specialized correctors - 5 correctors for 5 groups
        # Edge (P1): denoised(1) + noisy(1) + lambda(1) + enc1(16) = 19 channels
        self.edge_corrector = EdgeCorrector(in_channels=19)

        # Contrast (P2): denoised(1) + noisy(1) + lambda(1) + enc1(16) = 19 channels - NEW
        self.contrast_corrector = ContrastCorrector(in_channels=19)

        # Sharpness (P5): denoised(1) + noisy(1) + lambda(1) + enc1(16) = 19 channels - NEW
        self.sharpness_corrector = SharpnessCorrector(in_channels=19)

        # Texture (P4, P6): denoised(1) + noisy(1) + lambda(1) + enc1(16) + enc2(16) = 35 channels
        self.texture_corrector = TextureCorrector(in_channels=35)

        # Smooth (P3): denoised(1) + noisy(1) + lambda(1) + enc2(16) = 19 channels
        self.smooth_corrector = SmoothCorrector(in_channels=19)

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                lambda_maps: Dict[str, torch.Tensor],
                backbone_features: Dict[str, torch.Tensor],
                failure_maps: Dict[str, torch.Tensor] = None,
                return_individual: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply corrections weighted by lambda maps AND gated by failure maps.

        Args:
            denoised: Backbone output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            lambda_maps: Per-pixel lambda maps {'edge': [B,1,H,W], ...}
            backbone_features: {'enc1': [...], 'enc2': [...]}
            failure_maps: Per-pixel failure maps {'P1': [B,1,H,W], ...} - GATES corrections
            return_individual: Return individual corrections for loss

        Returns:
            corrected: Final corrected output
            info: Correction statistics
        """
        B, C, H, W = denoised.shape
        info = {}

        enc1 = backbone_features.get('enc1')
        enc2 = backbone_features.get('enc2')

        # Adapt backbone features
        f1 = self.adapt_enc1(enc1)
        if f1.shape[2:] != denoised.shape[2:]:
            f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)

        f2 = self.adapt_enc2(enc2)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

        # Get lambda maps (5 types now)
        lambda_edge = lambda_maps.get('edge', torch.zeros_like(denoised))
        lambda_contrast = lambda_maps.get('contrast', torch.zeros_like(denoised))
        lambda_sharpness = lambda_maps.get('sharpness', torch.zeros_like(denoised))
        lambda_texture = lambda_maps.get('texture', torch.zeros_like(denoised))
        lambda_smooth = lambda_maps.get('smooth', torch.zeros_like(denoised))

        # Get failure map gates (each corrector has its own gate)
        if failure_maps is not None:
            gate_edge = failure_maps.get('P1', torch.zeros_like(denoised))
            gate_contrast = failure_maps.get('P2', torch.zeros_like(denoised))
            gate_sharpness = failure_maps.get('P5', torch.zeros_like(denoised))
            gate_texture = torch.max(
                failure_maps.get('P4', torch.zeros_like(denoised)),
                failure_maps.get('P6', torch.zeros_like(denoised))
            )
            gate_smooth = failure_maps.get('P3', torch.zeros_like(denoised))
        else:
            gate_edge = torch.ones_like(denoised)
            gate_contrast = torch.ones_like(denoised)
            gate_sharpness = torch.ones_like(denoised)
            gate_texture = torch.ones_like(denoised)
            gate_smooth = torch.ones_like(denoised)

        # =====================================================================
        # EDGE CORRECTION (P1 - Boundary Detection)
        # =====================================================================
        edge_input = torch.cat([denoised, noisy, lambda_edge, f1], dim=1)
        raw_edge_correction = self.edge_corrector(edge_input, denoised)
        edge_correction = gate_edge * lambda_edge * raw_edge_correction

        # =====================================================================
        # CONTRAST CORRECTION (P2 - Layer Contrast) - NEW
        # =====================================================================
        contrast_input = torch.cat([denoised, noisy, lambda_contrast, f1], dim=1)
        raw_contrast_correction = self.contrast_corrector(contrast_input, denoised)
        contrast_correction = gate_contrast * lambda_contrast * raw_contrast_correction

        # =====================================================================
        # SHARPNESS CORRECTION (P5 - Boundary Sharpness) - NEW
        # =====================================================================
        sharpness_input = torch.cat([denoised, noisy, lambda_sharpness, f1], dim=1)
        raw_sharpness_correction = self.sharpness_corrector(sharpness_input, denoised)
        sharpness_correction = gate_sharpness * lambda_sharpness * raw_sharpness_correction

        # =====================================================================
        # TEXTURE CORRECTION (P4, P6 - Structure, Speckle)
        # =====================================================================
        texture_input = torch.cat([denoised, noisy, lambda_texture, f1, f2], dim=1)
        raw_texture_correction = self.texture_corrector(texture_input)
        texture_correction = gate_texture * lambda_texture * raw_texture_correction

        # =====================================================================
        # SMOOTH CORRECTION (P3 - Noise Reduction)
        # =====================================================================
        smooth_input = torch.cat([denoised, noisy, lambda_smooth, f2], dim=1)
        raw_smooth_correction = self.smooth_corrector(smooth_input, denoised)
        smooth_correction = gate_smooth * lambda_smooth * raw_smooth_correction

        # =====================================================================
        # COMBINE CORRECTIONS (5 corrections now)
        # =====================================================================
        total_correction = (edge_correction + contrast_correction + sharpness_correction +
                           texture_correction + smooth_correction)
        corrected = (denoised + total_correction).clamp(0, 1)

        # Statistics
        info['edge_lambda_mean'] = lambda_edge.mean().item()
        info['contrast_lambda_mean'] = lambda_contrast.mean().item()
        info['sharpness_lambda_mean'] = lambda_sharpness.mean().item()
        info['texture_lambda_mean'] = lambda_texture.mean().item()
        info['smooth_lambda_mean'] = lambda_smooth.mean().item()
        info['correction_magnitude'] = total_correction.abs().mean().item()

        if return_individual:
            info['individual_corrections'] = {
                'edge': edge_correction,
                'contrast': contrast_correction,
                'sharpness': sharpness_correction,
                'texture': texture_correction,
                'smooth': smooth_correction,
            }
            info['raw_corrections'] = {
                'edge': raw_edge_correction.detach(),
                'contrast': raw_contrast_correction.detach(),
                'sharpness': raw_sharpness_correction.detach(),
                'texture': raw_texture_correction.detach(),
                'smooth': raw_smooth_correction.detach(),
            }
            info['lambda_maps'] = {
                'edge': lambda_edge,
                'contrast': lambda_contrast,
                'sharpness': lambda_sharpness,
                'texture': lambda_texture,
                'smooth': lambda_smooth,
            }

        return corrected, info


# =============================================================================
# BACKBONE WRAPPER (Works with any denoising network)
# =============================================================================

class BackboneWrapper(nn.Module):
    """
    Wrapper for any denoising backbone.
    Exposes intermediate features for multi-scale correction.

    Feature extraction:
    - enc1: [B, width, H, W] - fine edges, details
    - enc2: [B, width*2, H/2, W/2] - textures, local patterns
    """

    def __init__(self, backbone_type: str = 'nafnet', width: int = 64):
        super().__init__()

        if backbone_type == 'nafnet':
            from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
            self.backbone = NAFNet(
                img_channel=1, width=width,
                middle_blk_num=2,
                enc_blk_nums=[2, 2, 2],
                dec_blk_nums=[2, 2, 2]
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        self.backbone_type = backbone_type
        self.width = width

    def load_pretrained(self, path: str) -> bool:
        """Load pretrained weights."""
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)

            model_state = self.backbone.state_dict()
            compatible = {k: v for k, v in state_dict.items()
                         if k in model_state and v.shape == model_state[k].shape}

            if len(compatible) == 0:
                print("No compatible weights found")
                del ckpt, state_dict, model_state  # Free memory
                return False

            self.backbone.load_state_dict(compatible, strict=False)
            print(f"Loaded {len(compatible)}/{len(model_state)} weights")
            print(f"Checkpoint PSNR: {ckpt.get('psnr', 'N/A')}")

            # Free checkpoint memory
            del ckpt, state_dict, compatible, model_state
            gc.collect()
            return True
        except Exception as e:
            print(f"Failed to load: {e}")
            return False

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through backbone.

        Args:
            x: Input image [B, 1, H, W]
            return_features: If True, also return intermediate encoder features

        Returns:
            If return_features=False: denoised output [B, 1, H, W]
            If return_features=True: (denoised, {'enc1': ..., 'enc2': ...})
        """
        if not return_features:
            return self.backbone(x).clamp(0, 1)

        # Manual forward to extract intermediate features
        # This mirrors NAFNetFullFiLM.forward() but captures encoder outputs
        bb = self.backbone
        B, C, H, W = x.shape
        inp = bb.check_image_size(x)

        # Feature extraction
        feat = bb.intro(inp)

        # Encoder pass - capture features
        encs = []
        for encoder, down in zip(bb.encoders, bb.downs):
            feat = encoder(feat)
            encs.append(feat)
            feat = down(feat)

        # Middle blocks
        if bb.condition_middle:
            for block in bb.middle_blks:
                feat = block(feat, None, None, None, 0.1)
        else:
            feat = bb.middle_blks(feat)

        # Decoder pass
        for stage_name, decoder, up, enc_skip in zip(
            bb.dbm_stage_names, bb.decoders, bb.ups, encs[::-1]
        ):
            feat = up(feat)
            feat = feat + enc_skip
            if bb.condition_decoders:
                for block in decoder:
                    feat = block(feat, None, None, None, 0.1)
            else:
                feat = decoder(feat)

        # Output
        out = bb.ending(feat)
        out = out + inp
        denoised = out[:, :, :H, :W].clamp(0, 1)

        # Return features (only enc1 and enc2 to save memory)
        # enc1: [B, 64, H, W], enc2: [B, 128, H/2, W/2]
        features = {
            'enc1': encs[0][:, :, :H, :W].detach(),  # Detach to save memory
            'enc2': encs[1][:, :, :H//2, :W//2].detach(),
        }

        # Clean up
        del encs, feat, inp, out

        return denoised, features


# =============================================================================
# COMPLETE NEURO-SYMBOLIC DENOISER V2
# =============================================================================

class NeuroSymbolicDenoiserV2(nn.Module):
    """
    Enhanced Neuro-Symbolic OCT Denoiser with SINGLE-PASS FEATURE-AWARE CORRECTION.

    Architecture:
    - Backbone (NAFNet): Initial denoising + feature extraction
    - Feature-Aware Corrector: Single-pass correction using backbone features
    - Symbolic predicates: Guide WHERE corrections apply

    Corrector has FULL INFORMATION to correct what backbone cannot:
    - denoised: What backbone produced
    - noisy: Original signal (what backbone worked with)
    - failure_map: WHERE predicates failed
    - enc1 (64ch): Fine edges/details from backbone encoder
    - enc2 (128ch): Textures/patterns from backbone encoder

    Key insight: With backbone features + failure guidance, corrector can
    make informed corrections in a SINGLE PASS.
    """

    def __init__(self, backbone_type: str = 'nafnet', width: int = 64):
        super().__init__()

        # Neural components
        self.backbone = BackboneWrapper(backbone_type, width)

        # Feature-aware corrector with backbone feature dimensions
        # enc1: width channels, enc2: width*2 channels
        self.corrector = FeatureAwareCorrector(
            enc1_channels=width,      # 64 for width=64
            enc2_channels=width * 2   # 128 for width=64
        )

        # Symbolic components
        self.predicates = SymbolicPredicates()

    def load_pretrained_backbone(self, path: str) -> bool:
        return self.backbone.load_pretrained(path)

    def forward(self, noisy: torch.Tensor, clean: torch.Tensor = None,
                return_details: bool = False) -> Dict:
        """
        SINGLE-PASS forward with feature-aware neuro-symbolic correction.

        Args:
            noisy: Noisy input [B, 1, H, W]
            clean: Clean reference [B, 1, H, W] - used during training
            return_details: If True, return additional info

        Returns:
            Dict with 'denoised', 'initial', 'predicate_results', etc.
        """
        # =====================================================================
        # STEP 1: Backbone denoising + feature extraction
        # =====================================================================
        initial, backbone_features = self.backbone(noisy, return_features=True)

        # Store initial predicate results (on backbone output)
        init_pred = self.predicates(noisy, initial, return_failure_maps=False)
        initial_predicate_results = {
            name: {
                'score': init_pred[name]['score'].detach() if isinstance(init_pred[name]['score'], torch.Tensor) else init_pred[name]['score'],
                'passed': init_pred[name]['passed']
            }
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }

        # Check if already passing all predicates
        if init_pred['all_passed']:
            del backbone_features
            return {
                'denoised': initial,
                'initial': initial.detach(),
                'predicate_results': init_pred,
                'initial_predicate_results': initial_predicate_results,
                'correction_applied': False,
                'correction_info': {},
            }

        # =====================================================================
        # STEP 2: Evaluate predicates to get failure maps
        # =====================================================================
        pred_with_maps = self.predicates(noisy, initial, return_failure_maps=True)

        failure_maps = {
            name: pred_with_maps[name]['failure_map']
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }

        # Resolve conflicts using priority rules
        exclusive_masks = self.predicates.resolve_conflicts(failure_maps)
        del failure_maps

        # =====================================================================
        # STEP 3: SINGLE-PASS Feature-aware correction
        # Corrector has ALL information needed:
        # - initial (backbone output)
        # - noisy (original input)
        # - exclusive_masks (WHERE to correct)
        # - backbone_features (HOW to correct - enc1, enc2)
        # =====================================================================
        corrected, corr_info = self.corrector(
            initial, noisy, exclusive_masks,
            backbone_features=backbone_features,
            return_individual=self.training
        )

        # Clean up backbone features (no longer needed)
        del backbone_features, exclusive_masks

        # =====================================================================
        # STEP 4: Final predicate evaluation
        # =====================================================================
        final_pred = self.predicates(noisy, corrected, return_failure_maps=False)

        # Compute correction magnitude
        correction = corrected - initial.detach()
        correction_magnitude = correction.abs().mean().item()

        output = {
            'denoised': corrected,
            'initial': initial.detach(),
            'predicate_results': final_pred,
            'initial_predicate_results': initial_predicate_results,
            'correction_applied': True,
            'correction_magnitude': correction_magnitude,
            'correction_info': corr_info,
        }

        return output


# =============================================================================
# NEURO-SYMBOLIC DENOISER V3: ADAPTIVE LAMBDA
# =============================================================================

class NeuroSymbolicDenoiserV3(nn.Module):
    """
    Neuro-Symbolic OCT Denoiser with ADAPTIVE LAMBDA PREDICTION.

    Key innovation: Instead of binary masks, learns per-pixel correction strength.

    Architecture:
    - Backbone (NAFNet): Initial denoising + feature extraction
    - Symbolic Predicates: Compute failure maps (WHAT failed, WHERE)
    - Lambda Predictor: Predicts per-pixel correction strength (HOW MUCH)
    - Adaptive Corrector: Applies weighted corrections (HOW)

    Training objectives:
    - Corrector: Specialized losses (edge/texture/smooth match clean)
    - Lambda Predictor: Final quality + predicate satisfaction

    This separation allows:
    - Corrector to learn WHAT corrections to make
    - Lambda predictor to learn HOW MUCH correction each pixel needs
    """

    def __init__(self, backbone_type: str = 'nafnet', width: int = 64):
        super().__init__()

        # Neural components
        self.backbone = BackboneWrapper(backbone_type, width)

        # Symbolic predicates (unchanged - still defines WHAT and WHERE)
        self.predicates = SymbolicPredicates()

        # Lambda predictor (NEW - learns HOW MUCH)
        self.lambda_predictor = AdaptiveLambdaPredictor()

        # Adaptive corrector (learns HOW, weighted by lambda)
        self.corrector = AdaptiveCorrectorWithLambda(
            enc1_channels=width,
            enc2_channels=width * 2
        )

    def load_pretrained_backbone(self, path: str) -> bool:
        return self.backbone.load_pretrained(path)

    def forward(self, noisy: torch.Tensor, clean: torch.Tensor = None,
                return_details: bool = False) -> Dict:
        """
        Forward pass with adaptive lambda prediction.
        OPTIMIZED: Reduced predicate calls from 3 to 2, better memory management.
        """
        # =====================================================================
        # STEP 1: Backbone denoising + feature extraction
        # =====================================================================
        initial, backbone_features = self.backbone(noisy, return_features=True)

        # =====================================================================
        # STEP 2: Get failure maps AND initial scores in ONE call (OPTIMIZED)
        # =====================================================================
        pred_with_maps = self.predicates(noisy, initial, return_failure_maps=True)

        # Extract initial results and failure maps from same call
        initial_predicate_results = {
            name: {
                'score': pred_with_maps[name]['score'].detach() if isinstance(pred_with_maps[name]['score'], torch.Tensor) else pred_with_maps[name]['score'],
                'passed': pred_with_maps[name]['passed']
            }
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }

        failure_maps = {
            name: pred_with_maps[name]['failure_map']
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }
        del pred_with_maps  # Free memory

        # =====================================================================
        # STEP 3: Lambda Predictor - predict per-pixel correction strength
        # =====================================================================
        lambda_maps = self.lambda_predictor(initial, failure_maps)

        # =====================================================================
        # STEP 4: Adaptive correction weighted by lambda maps AND gated by failure maps
        # =====================================================================
        corrected, corr_info = self.corrector(
            initial, noisy, lambda_maps,
            backbone_features=backbone_features,
            failure_maps=failure_maps,  # NEW: gate corrections by failure maps
            return_individual=self.training
        )

        # Clean up large tensors
        del backbone_features

        # =====================================================================
        # STEP 5: Final predicate evaluation
        # =====================================================================
        final_pred = self.predicates(noisy, corrected, return_failure_maps=False)

        # Compute correction magnitude
        correction = corrected - initial.detach()
        correction_magnitude = correction.abs().mean().item()

        output = {
            'denoised': corrected,
            'initial': initial.detach(),
            'predicate_results': final_pred,
            'initial_predicate_results': initial_predicate_results,
            'correction_applied': True,
            'correction_magnitude': correction_magnitude,
            'correction_info': corr_info,
            'lambda_maps': {k: v.detach() for k, v in lambda_maps.items()},
            'failure_maps': {k: v.detach() for k, v in failure_maps.items()},
        }

        # Keep lambda_maps with gradients for loss computation
        if self.training:
            output['lambda_maps_grad'] = lambda_maps

        return output


# =============================================================================
# TRAINING LOSS V3: DUAL OBJECTIVES
# =============================================================================

class NeuroSymbolicLossV3(nn.Module):
    """
    Dual-objective loss for adaptive lambda architecture.

    Two separate objectives:
    1. CORRECTOR OBJECTIVE: Learn WHAT corrections to make
       - Edge corrector: Match clean edges
       - Texture corrector: Match clean texture/structure
       - Smooth corrector: Reduce noise in flat regions

    2. LAMBDA PREDICTOR OBJECTIVE: Learn HOW MUCH to correct
       - Quality: Final output close to clean
       - Predicates: All predicates should pass (hinge loss)

    Key insight: Corrector learns the correction "vocabulary",
    Lambda predictor learns when/how much to use each correction.
    """

    def __init__(self):
        super().__init__()

        # Sobel filters - COMBINED for single conv2d (OPTIMIZED)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_combined = torch.cat([sobel_x, sobel_y], dim=0)  # [2, 1, 3, 3]
        self.register_buffer('sobel_combined', sobel_combined)

        # Learnable predicate thresholds
        def logit(p):
            return torch.log(torch.tensor(p) / (1 - torch.tensor(p)))

        self.threshold_P1 = nn.Parameter(logit(0.90))
        self.threshold_P2 = nn.Parameter(logit(0.70))  # Lowered: current ~0.66
        self.threshold_P3 = nn.Parameter(logit(0.90))
        self.threshold_P4 = nn.Parameter(logit(0.90))
        self.threshold_P5 = nn.Parameter(logit(0.75))  # Lowered: current ~0.74
        self.threshold_P6 = nn.Parameter(logit(0.95))

        # Cache for clean edges (avoid recomputing)
        self._cached_edges_clean = None
        self._cached_clean_id = None

    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using combined Sobel (OPTIMIZED)."""
        grad = F.conv2d(x, self.sobel_combined, padding=1)  # [B, 2, H, W]
        return (grad[:, 0:1].square() + grad[:, 1:2].square() + 1e-8).sqrt()

    def compute_local_std(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Compute local standard deviation (OPTIMIZED with in-place)."""
        padding = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        sq_mean = F.avg_pool2d(x.square(), kernel_size, stride=1, padding=padding)
        return (sq_mean - mean.square()).clamp_(min=1e-6).sqrt_()

    # =========================================================================
    # CORRECTOR LOSSES: Learn WHAT corrections to make
    # =========================================================================

    def corrector_loss_edge(self, corrected: torch.Tensor, clean: torch.Tensor,
                            lambda_edge: torch.Tensor) -> torch.Tensor:
        """Edge corrector should make edges match clean."""
        edges_corr = self.compute_edges(corrected)
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)

        # Weight by lambda with clamped normalization to prevent amplification
        # Clamp weight to max 10x to avoid sparse lambda explosion
        lambda_sum = lambda_edge.sum() + 1e-6
        weight = torch.clamp(lambda_edge / (lambda_sum / lambda_edge.numel() + 1e-6), 0, 10.0)
        loss = ((edges_corr - edges_clean) ** 2 * weight).sum() / (weight.sum() + 1e-6)
        return loss

    def corrector_loss_texture(self, corrected: torch.Tensor, clean: torch.Tensor,
                               lambda_texture: torch.Tensor) -> torch.Tensor:
        """Texture corrector should match clean local statistics."""
        std_corr = self.compute_local_std(corrected)
        with torch.no_grad():
            std_clean = self.compute_local_std(clean)

        # Weight by lambda with clamped normalization
        lambda_sum = lambda_texture.sum() + 1e-6
        weight = torch.clamp(lambda_texture / (lambda_sum / lambda_texture.numel() + 1e-6), 0, 10.0)
        loss = ((std_corr - std_clean) ** 2 * weight).sum() / (weight.sum() + 1e-6)
        return loss

    def corrector_loss_smooth(self, corrected: torch.Tensor, clean: torch.Tensor,
                              lambda_smooth: torch.Tensor) -> torch.Tensor:
        """Smooth corrector should reduce variance in flat regions."""
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)
            flat_regions = (edges_clean < 0.1).float()

        local_var = self.compute_local_std(corrected) ** 2
        # Weight by lambda * flat_regions with clamped normalization
        raw_weight = lambda_smooth * flat_regions
        lambda_sum = raw_weight.sum() + 1e-6
        weight = torch.clamp(raw_weight / (lambda_sum / raw_weight.numel() + 1e-6), 0, 10.0)
        loss = (local_var * weight).sum() / (weight.sum() + 1e-6)
        return loss

    def corrector_loss_contrast(self, corrected: torch.Tensor, clean: torch.Tensor,
                                lambda_contrast: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """
        P2 Contrast corrector: MATCH clean local std (not just preserve noisy).

        Key insight: P2 fails because local_std of corrected < local_std of clean.
        Solution: Minimize |local_std(corrected) - local_std(clean)|
        """
        std_corr = self.compute_local_std(corrected)
        with torch.no_grad():
            std_clean = self.compute_local_std(clean)

        # Direct matching to clean contrast (not preservation of noisy)
        lambda_sum = lambda_contrast.sum() + 1e-6
        weight = torch.clamp(lambda_contrast / (lambda_sum / lambda_contrast.numel() + 1e-6), 0, 10.0)
        loss = ((std_corr - std_clean) ** 2 * weight).sum() / (weight.sum() + 1e-6)
        return loss

    def corrector_loss_sharpness(self, corrected: torch.Tensor, clean: torch.Tensor,
                                 lambda_sharpness: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """
        P5 Sharpness corrector: MATCH clean edge strength at boundaries.

        Key insight: P5 fails because edge gradient at boundaries is weaker than reference.
        Solution: Maximize edge strength at boundary locations to match clean.
        """
        edges_corr = self.compute_edges(corrected)
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)
            # Identify boundary regions (strong edges in clean)
            boundary_mask = (edges_clean > edges_clean.mean()).float()

        # Minimize edge difference at boundaries (encourage sharpening)
        lambda_sum = lambda_sharpness.sum() + 1e-6
        weight = torch.clamp(lambda_sharpness * boundary_mask / (lambda_sum / lambda_sharpness.numel() + 1e-6), 0, 10.0)
        loss = ((edges_corr - edges_clean) ** 2 * weight).sum() / (weight.sum() + 1e-6)
        return loss

    # =========================================================================
    # LAMBDA PREDICTOR LOSS: Learn HOW MUCH to correct
    # =========================================================================

    def lambda_loss(self, corrected: torch.Tensor, clean: torch.Tensor,
                    predicate_results: Dict) -> torch.Tensor:
        """
        Lambda predictor objective: Final quality + predicate satisfaction.

        This is the key objective that trains the lambda predictor to
        produce correction strengths that result in:
        1. High-quality output (close to clean)
        2. Passing predicates
        """
        # Quality loss: reconstruction should be good
        quality_loss = F.mse_loss(corrected, clean)

        # Predicate satisfaction loss (hinge loss)
        # Penalize only when below threshold
        pred_loss = torch.tensor(0.0, device=corrected.device)

        scores = {
            'P1': predicate_results['P1']['score'],
            'P2': predicate_results['P2']['score'],
            'P3': predicate_results['P3']['score'],
            'P4': predicate_results['P4']['score'],
            'P5': predicate_results['P5']['score'],
            'P6': predicate_results['P6']['score'],
        }

        thresholds = {
            'P1': torch.sigmoid(self.threshold_P1),  # Sigmoid to keep in [0,1]
            'P2': torch.sigmoid(self.threshold_P2),
            'P3': torch.sigmoid(self.threshold_P3),
            'P4': torch.sigmoid(self.threshold_P4),
            'P5': torch.sigmoid(self.threshold_P5),
            'P6': torch.sigmoid(self.threshold_P6),
        }

        # Weights for each predicate
        weights = {'P1': 0.10, 'P2': 0.30, 'P3': 0.05, 'P4': 0.10, 'P5': 0.30, 'P6': 0.15}

        for name in scores:
            score = scores[name]
            if isinstance(score, torch.Tensor):
                threshold = thresholds[name]
                # Hinge loss: penalize when score < threshold
                pred_loss = pred_loss + weights[name] * F.relu(threshold - score)

        # DIRECT P2/P5 MAXIMIZATION: Add negative score loss to push scores UP
        # This is the key innovation: don't just penalize below threshold,
        # actively maximize the scores for P2 and P5
        p2_score = scores['P2']
        p5_score = scores['P5']
        if isinstance(p2_score, torch.Tensor) and isinstance(p5_score, torch.Tensor):
            # Loss = -score means gradient pushes score UP
            direct_p2p5_loss = -0.5 * (p2_score + p5_score)
        else:
            direct_p2p5_loss = torch.tensor(0.0, device=corrected.device)

        # Combined lambda objective
        # Quality + hinge predicates + direct P2/P5 maximization
        return quality_loss + 2.0 * pred_loss + 1.0 * direct_p2p5_loss

    # =========================================================================
    # FORWARD: Compute all losses
    # =========================================================================

    def forward(self, output: Dict, clean: torch.Tensor, noisy: torch.Tensor = None) -> Dict:
        """
        Compute dual-objective losses. OPTIMIZED: Cache clean edges, compute once.
        """
        corrected = output['denoised']
        pred_results = output['predicate_results']

        # Get lambda maps (5 types now)
        lambda_maps = output.get('lambda_maps_grad', output.get('lambda_maps', {}))
        lambda_edge = lambda_maps.get('edge', torch.zeros_like(corrected))
        lambda_contrast = lambda_maps.get('contrast', torch.zeros_like(corrected))
        lambda_sharpness = lambda_maps.get('sharpness', torch.zeros_like(corrected))
        lambda_texture = lambda_maps.get('texture', torch.zeros_like(corrected))
        lambda_smooth = lambda_maps.get('smooth', torch.zeros_like(corrected))

        # =====================================================================
        # PRE-COMPUTE SHARED VALUES (OPTIMIZED)
        # =====================================================================
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)
            std_clean = self.compute_local_std(clean)
            flat_regions = (edges_clean < 0.1).float()
            boundary_mask = (edges_clean > edges_clean.mean()).float()

        edges_corr = self.compute_edges(corrected)
        std_corr = self.compute_local_std(corrected)

        # =====================================================================
        # CORRECTOR LOSSES (using cached values)
        # =====================================================================
        # Edge loss (P1)
        lambda_sum = lambda_edge.sum() + 1e-6
        weight_e = torch.clamp(lambda_edge / (lambda_sum / lambda_edge.numel() + 1e-6), 0, 10.0)
        loss_edge = ((edges_corr - edges_clean) ** 2 * weight_e).sum() / (weight_e.sum() + 1e-6)

        # Contrast loss (P2) - Match clean local std
        lambda_sum = lambda_contrast.sum() + 1e-6
        weight_c = torch.clamp(lambda_contrast / (lambda_sum / lambda_contrast.numel() + 1e-6), 0, 10.0)
        loss_contrast = ((std_corr - std_clean) ** 2 * weight_c).sum() / (weight_c.sum() + 1e-6)

        # Sharpness loss (P5) - Match clean edge strength at boundaries
        lambda_sum = lambda_sharpness.sum() + 1e-6
        weight_sh = torch.clamp(lambda_sharpness * boundary_mask / (lambda_sum / lambda_sharpness.numel() + 1e-6), 0, 10.0)
        loss_sharpness = ((edges_corr - edges_clean) ** 2 * weight_sh).sum() / (weight_sh.sum() + 1e-6)

        # Texture loss (P4, P6)
        lambda_sum = lambda_texture.sum() + 1e-6
        weight_t = torch.clamp(lambda_texture / (lambda_sum / lambda_texture.numel() + 1e-6), 0, 10.0)
        loss_texture = ((std_corr - std_clean) ** 2 * weight_t).sum() / (weight_t.sum() + 1e-6)

        # Smooth loss (P3)
        local_var = std_corr.square()
        raw_weight = lambda_smooth * flat_regions
        lambda_sum = raw_weight.sum() + 1e-6
        weight_s = torch.clamp(raw_weight / (lambda_sum / raw_weight.numel() + 1e-6), 0, 10.0)
        loss_smooth = (local_var * weight_s).sum() / (weight_s.sum() + 1e-6)

        # Include all 5 corrector losses
        loss_corrector = loss_edge + loss_contrast + loss_sharpness + loss_texture + loss_smooth

        # =====================================================================
        # LAMBDA PREDICTOR LOSS
        # =====================================================================
        loss_lambda = self.lambda_loss(corrected, clean, pred_results)

        # =====================================================================
        # QUALITY PRESERVATION LOSS (NEW)
        # Penalize when correction HURTS quality (MSE or SSIM degradation)
        # =====================================================================
        initial = output.get('initial')
        loss_preserve = torch.tensor(0.0, device=corrected.device)
        if initial is not None:
            # MSE degradation penalty
            with torch.no_grad():
                mse_initial = F.mse_loss(initial, clean, reduction='none').mean(dim=(1,2,3))
            mse_corrected = F.mse_loss(corrected, clean, reduction='none').mean(dim=(1,2,3))
            mse_degradation = F.relu(mse_corrected - mse_initial)

            # SSIM degradation penalty (simplified differentiable SSIM approximation)
            # Using local means and variances which are already differentiable
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            with torch.no_grad():
                mu_init = F.avg_pool2d(initial, 7, stride=1, padding=3)
                mu_clean = F.avg_pool2d(clean, 7, stride=1, padding=3)
                sigma_init_sq = F.avg_pool2d(initial**2, 7, stride=1, padding=3) - mu_init**2
                sigma_clean_sq = F.avg_pool2d(clean**2, 7, stride=1, padding=3) - mu_clean**2
                sigma_init_clean = F.avg_pool2d(initial*clean, 7, stride=1, padding=3) - mu_init*mu_clean
                ssim_init = ((2*mu_init*mu_clean + C1)*(2*sigma_init_clean + C2)) / \
                           ((mu_init**2 + mu_clean**2 + C1)*(sigma_init_sq + sigma_clean_sq + C2))
                ssim_init_mean = ssim_init.mean(dim=(1,2,3))

            mu_corr = F.avg_pool2d(corrected, 7, stride=1, padding=3)
            sigma_corr_sq = F.avg_pool2d(corrected**2, 7, stride=1, padding=3) - mu_corr**2
            sigma_corr_clean = F.avg_pool2d(corrected*clean, 7, stride=1, padding=3) - mu_corr*mu_clean
            ssim_corr = ((2*mu_corr*mu_clean + C1)*(2*sigma_corr_clean + C2)) / \
                       ((mu_corr**2 + mu_clean**2 + C1)*(sigma_corr_sq + sigma_clean_sq + C2))
            ssim_corr_mean = ssim_corr.mean(dim=(1,2,3))

            # Penalize when SSIM decreases (ssim_corr < ssim_init)
            ssim_degradation = F.relu(ssim_init_mean - ssim_corr_mean)

            # Combined preservation loss
            loss_preserve = mse_degradation.mean() * 10.0 + ssim_degradation.mean() * 5.0

        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        total = loss_corrector + loss_lambda + loss_preserve

        return {
            'total': total,
            'corrector': loss_corrector.detach(),
            'lambda': loss_lambda.detach(),
            'preserve': loss_preserve.detach(),
            'edge': loss_edge.detach(),
            'contrast': loss_contrast.detach(),
            'sharpness': loss_sharpness.detach(),
            'texture': loss_texture.detach(),
            'smooth': loss_smooth.detach(),
            'lambda_edge_mean': lambda_edge.mean().detach(),
            'lambda_contrast_mean': lambda_contrast.mean().detach(),
            'lambda_sharpness_mean': lambda_sharpness.mean().detach(),
            'lambda_texture_mean': lambda_texture.mean().detach(),
            'lambda_smooth_mean': lambda_smooth.mean().detach(),
            'P1': pred_results['P1']['score'].detach() if isinstance(pred_results['P1']['score'], torch.Tensor) else pred_results['P1']['score'],
            'P2': pred_results['P2']['score'].detach() if isinstance(pred_results['P2']['score'], torch.Tensor) else pred_results['P2']['score'],
            'P3': pred_results['P3']['score'].detach() if isinstance(pred_results['P3']['score'], torch.Tensor) else pred_results['P3']['score'],
            'P4': pred_results['P4']['score'].detach() if isinstance(pred_results['P4']['score'], torch.Tensor) else pred_results['P4']['score'],
            'P5': pred_results['P5']['score'].detach() if isinstance(pred_results['P5']['score'], torch.Tensor) else pred_results['P5']['score'],
            'P6': pred_results['P6']['score'].detach() if isinstance(pred_results['P6']['score'], torch.Tensor) else pred_results['P6']['score'],
        }


# =============================================================================
# TRAINING LOSS V2 (LEGACY)
# =============================================================================

class NeuroSymbolicLossV2(nn.Module):
    """
    Combined loss with SPECIALIZED PER-CORRECTOR LOSSES.

    Each corrector has a loss tailored to its specific goal:
    - P1, P5 (Edge): Edge preservation/sharpening loss
    - P2 (Contrast): Local contrast enhancement loss
    - P3, P6 (Smoothing): Variance reduction in flat regions
    - P4 (Structure): SSIM-based structure preservation

    Key principle: Each corrector optimizes for WHAT IT'S DESIGNED TO DO.
    """

    def __init__(self, lambda_pred: float = 0.0,  # Disabled - conflicts with clean-based losses
                 lambda_preserve: float = 15.0,   # STRONGER quality preservation (was 8.0)
                 lambda_detail: float = 0.2,      # More detail preservation (was 0.1)
                 lambda_corrector: float = 1.5):  # Reduced corrector loss (was 3.0)
        super().__init__()
        self.lambda_pred = lambda_pred  # Set to 0 to remove noisy-reference conflict
        self.lambda_preserve = lambda_preserve  # Penalize quality degradation
        self.lambda_detail = lambda_detail
        self.lambda_corrector = lambda_corrector  # Per-predicate specialized losses

        # Sobel filters for edge detection - stacked for single conv2d call
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_combined = torch.cat([sobel_x, sobel_y], dim=0)  # [2, 1, 3, 3]
        self.register_buffer('sobel_combined', sobel_combined)

        # Gaussian smoothing for reference edge extraction (separable for speed)
        kernel_size = 5
        sigma = 1.0
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        self.register_buffer('gaussian_h', kernel_1d.view(1, 1, 1, kernel_size))
        self.register_buffer('gaussian_v', kernel_1d.view(1, 1, kernel_size, 1))

    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using stacked Sobel filters (CPU optimized)."""
        grad = F.conv2d(x, self.sobel_combined, padding=1)  # [B, 2, H, W]
        return (grad[:, 0:1].square() + grad[:, 1:2].square() + 1e-8).sqrt_()

    def compute_local_std(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Compute local standard deviation - CPU optimized."""
        padding = kernel_size // 2
        x_sq = x.square()
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        sq_mean = F.avg_pool2d(x_sq, kernel_size, stride=1, padding=padding)
        del x_sq
        return (sq_mean - mean.square()).clamp_(min=1e-6).sqrt_()

    # =========================================================================
    # SPECIALIZED PER-CORRECTOR LOSSES
    # =========================================================================

    def loss_P1_edge_preservation(self, corrected: torch.Tensor, clean: torch.Tensor,
                                   mask: torch.Tensor) -> torch.Tensor:
        """P1: Boundary Detectability - edges should match clean reference."""
        edges_corrected = self.compute_edges(corrected)
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)
        # MSE between edges, focused on masked regions
        mask_sum = mask.sum() + 1e-6
        loss = ((edges_corrected - edges_clean) ** 2 * mask).sum() / mask_sum
        return loss

    def loss_P2_contrast_enhancement(self, corrected: torch.Tensor, clean: torch.Tensor,
                                      mask: torch.Tensor) -> torch.Tensor:
        """P2: Layer Contrast - local std should match clean reference."""
        std_corrected = self.compute_local_std(corrected)
        with torch.no_grad():
            std_clean = self.compute_local_std(clean)
        # Want corrected std to match clean std (not just maximize)
        mask_sum = mask.sum() + 1e-6
        loss = ((std_corrected - std_clean) ** 2 * mask).sum() / mask_sum
        return loss

    def loss_P3_noise_reduction(self, corrected: torch.Tensor, clean: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
        """P3: Noise Reduction - minimize variance in flat regions."""
        # Identify flat regions (low edges in clean)
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)
            flat_mask = (edges_clean < 0.1).float() * mask
            flat_sum = flat_mask.sum() + 1e-6
        # Minimize local variance in flat regions
        local_var = self.compute_local_std(corrected) ** 2
        loss = (local_var * flat_mask).sum() / flat_sum
        return loss

    def loss_P4_structure_preservation(self, corrected: torch.Tensor, clean: torch.Tensor,
                                         mask: torch.Tensor) -> torch.Tensor:
        """P4: Structure Preservation - SSIM-based loss."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu1 = F.avg_pool2d(corrected, 5, stride=1, padding=2)
        mu2 = F.avg_pool2d(clean, 5, stride=1, padding=2)
        # FIX: clamp variance to min=0 to prevent numerical issues from float precision
        sigma1_sq = (F.avg_pool2d(corrected ** 2, 5, stride=1, padding=2) - mu1 ** 2).clamp(min=0)
        sigma2_sq = (F.avg_pool2d(clean ** 2, 5, stride=1, padding=2) - mu2 ** 2).clamp(min=0)
        sigma12 = F.avg_pool2d(corrected * clean, 5, stride=1, padding=2) - mu1 * mu2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        # Loss = 1 - SSIM (want to maximize SSIM)
        mask_sum = mask.sum() + 1e-6
        loss = ((1 - ssim) * mask).sum() / mask_sum
        return loss

    def loss_P5_sharpness(self, corrected: torch.Tensor, clean: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """P5: Boundary Sharpness - gradient magnitude should match clean."""
        edges_corrected = self.compute_edges(corrected)
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)
            # Focus on strong edges in clean - FIX: add .float() for boolean tensor
            boundary_mask = (edges_clean > edges_clean.mean()).float() * mask
            boundary_sum = boundary_mask.sum() + 1e-6
        # Want edges to match clean at boundaries
        loss = ((edges_corrected - edges_clean) ** 2 * boundary_mask).sum() / boundary_sum
        return loss

    def loss_P6_speckle_suppression(self, corrected: torch.Tensor, clean: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
        """P6: Speckle Suppression - reduce std/mean in homogeneous regions."""
        with torch.no_grad():
            edges_clean = self.compute_edges(clean)
            homogeneous_mask = (edges_clean < 0.05).float() * mask
            homogeneous_sum = homogeneous_mask.sum() + 1e-6
        # Speckle contrast = std / mean
        local_std = self.compute_local_std(corrected)
        local_mean = F.avg_pool2d(corrected, 5, stride=1, padding=2)
        speckle = local_std / (local_mean.abs() + 0.01)
        # Minimize speckle in homogeneous regions
        loss = (speckle * homogeneous_mask).sum() / homogeneous_sum
        return loss

    def compute_group_loss(self, denoised: torch.Tensor, clean: torch.Tensor,
                            group_name: str, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for a corrector GROUP (edge/texture/smooth).

        Args:
            denoised: Current denoised output
            clean: Clean reference
            group_name: 'edge', 'texture', or 'smooth'
            mask: Combined mask for this group's predicates

        Returns:
            Group-specific loss
        """
        mask_sum = mask.sum() + 1e-6

        if group_name == 'edge':
            # Edge group (P1, P5): Edge preservation loss
            edges_den = self.compute_edges(denoised)
            with torch.no_grad():
                edges_clean = self.compute_edges(clean)
            loss = ((edges_den - edges_clean) ** 2 * mask).sum() / mask_sum
            del edges_den, edges_clean

        elif group_name == 'texture':
            # Texture group (P2, P4, P6): Combined contrast + structure loss
            # Contrast component
            std_den = self.compute_local_std(denoised)
            with torch.no_grad():
                std_clean = self.compute_local_std(clean)
            loss_contrast = ((std_den - std_clean) ** 2 * mask).sum() / mask_sum
            del std_den, std_clean

            # Structure component (simplified SSIM)
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            mu1 = F.avg_pool2d(denoised, 5, stride=1, padding=2)
            mu2 = F.avg_pool2d(clean, 5, stride=1, padding=2)
            sigma12 = F.avg_pool2d(denoised * clean, 5, stride=1, padding=2) - mu1 * mu2
            structure = (sigma12 + C2) / (mu1.abs() * mu2.abs() + C2)
            loss_struct = ((1 - structure) * mask).sum() / mask_sum
            del mu1, mu2, sigma12, structure

            loss = 0.5 * loss_contrast + 0.5 * loss_struct

        else:  # smooth
            # Smooth group (P3): Variance reduction in flat regions
            with torch.no_grad():
                edges_clean = self.compute_edges(clean)
                flat_mask = (edges_clean < 0.1).float() * mask
                flat_sum = flat_mask.sum() + 1e-6
                del edges_clean
            local_var = self.compute_local_std(denoised) ** 2
            loss = (local_var * flat_mask).sum() / flat_sum
            del local_var, flat_mask

        return loss

    def compute_per_corrector_loss(self, initial: torch.Tensor, clean: torch.Tensor,
                                    correction_info: Dict) -> Dict[str, torch.Tensor]:
        """
        SPECIALIZED PER-GROUP LOSSES: Each corrector group optimizes for its specific goal.

        Groups:
        - Edge (P1, P5): Edge preservation/sharpening loss
        - Texture (P2, P4, P6): Contrast + structure loss
        - Smooth (P3): Variance reduction in flat regions
        """
        losses = {}

        individual_corrections = correction_info.get('individual_corrections', {})
        individual_masks = correction_info.get('individual_masks', {})

        if not individual_corrections:
            return losses

        # Group predicates by their corrector group
        pred_to_group = {
            'P1': 'edge', 'P5': 'edge',
            'P2': 'texture', 'P4': 'texture', 'P6': 'texture',
            'P3': 'smooth'
        }

        # Aggregate corrections and masks by group
        group_corrections = {'edge': [], 'texture': [], 'smooth': []}
        group_masks = {'edge': [], 'texture': [], 'smooth': []}

        for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            if name not in individual_corrections:
                continue

            group = pred_to_group[name]
            group_corrections[group].append(individual_corrections[name])
            group_masks[group].append(individual_masks[name])

        # Compute loss per group (memory efficient: one forward per group)
        for group_name in ['edge', 'texture', 'smooth']:
            if not group_corrections[group_name]:
                continue

            # Combine corrections and masks for this group
            combined_correction = sum(group_corrections[group_name])
            combined_mask = sum(group_masks[group_name]).clamp(0, 1)

            # Temporary output with group correction
            temp_output = (initial + combined_correction).clamp(0, 1)

            # Compute group-specific loss
            losses[group_name] = self.compute_group_loss(temp_output, clean, group_name, combined_mask)

            # Memory cleanup
            del combined_correction, combined_mask, temp_output

        return losses

    def compute_detail_loss(self, denoised: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """
        Detail preservation loss: penalize when denoised edges are weaker than reference.
        CPU optimized with separable gaussian convolution.
        """
        # Smooth noisy with separable gaussian (faster than 2D)
        noisy_smooth = F.conv2d(F.pad(noisy, (2, 2, 0, 0), mode='replicate'), self.gaussian_h)
        noisy_smooth = F.conv2d(F.pad(noisy_smooth, (0, 0, 2, 2), mode='replicate'), self.gaussian_v)

        # Compute edges
        edges_denoised = self.compute_edges(denoised)
        with torch.no_grad():
            edges_ref = self.compute_edges(noisy_smooth)
            ref_max = edges_ref.max() + 1e-6
            edges_ref_norm = edges_ref / ref_max

        edges_den_norm = edges_denoised / ref_max

        # Only penalize where denoised edges are WEAKER than reference
        edge_weakness = (edges_ref_norm - edges_den_norm).clamp_(min=0)

        # Focus on significant edges (ignore weak texture)
        significant_mask = (edges_ref_norm > 0.1).float()
        loss_detail = (edge_weakness * significant_mask).sum() / (significant_mask.sum() + 1e-6)

        return loss_detail

    def forward(self, output: Dict, clean: torch.Tensor, noisy: torch.Tensor = None,
                predicates_module: nn.Module = None) -> Dict:
        """
        Compute SINGLE-PASS loss with group-specialized corrector losses.

        Loss components:
        1. Reconstruction: MSE(corrected, clean)
        2. Preservation: Penalize if correction degrades PSNR
        3. Detail: Penalize edge loss vs reference
        4. Group losses: Specialized per corrector group (edge/texture/smooth)
        """
        denoised = output['denoised']
        initial = output['initial']
        pred = output['predicate_results']
        correction_info = output.get('correction_info', {})

        # =====================================================================
        # RECONSTRUCTION LOSS
        # =====================================================================
        loss_recon = F.mse_loss(denoised, clean)
        loss_initial = F.mse_loss(initial, clean)

        # =====================================================================
        # PSNR PRESERVATION: penalize if correction makes output worse
        # =====================================================================
        loss_preserve = F.relu(loss_recon - loss_initial)

        # =====================================================================
        # DETAIL PRESERVATION: penalize oversmoothing
        # =====================================================================
        if noisy is not None:
            loss_detail = self.compute_detail_loss(denoised, noisy)
        else:
            loss_detail = torch.tensor(0.0, device=denoised.device)

        # =====================================================================
        # GROUP-SPECIALIZED CORRECTOR LOSSES
        # Each group optimizes for its specific task:
        # - Edge (P1, P5): Edge preservation
        # - Texture (P2, P4, P6): Contrast + structure
        # - Smooth (P3): Variance reduction
        # =====================================================================
        per_group_losses = self.compute_per_corrector_loss(initial, clean, correction_info)

        loss_corrector = torch.tensor(0.0, device=denoised.device)
        for name, loss in per_group_losses.items():
            loss_corrector = loss_corrector + loss

        # =====================================================================
        # PREDICATE SCORE LOSSES (for monitoring)
        # =====================================================================
        loss_P1 = 1 - pred['P1']['score']
        loss_P2 = 1 - pred['P2']['score']
        loss_P3 = 1 - pred['P3']['score']
        loss_P4 = 1 - pred['P4']['score']
        loss_P5 = 1 - pred['P5']['score']
        loss_P6 = 1 - pred['P6']['score']
        loss_pred = (0.25 * loss_P1 + 0.15 * loss_P2 + 0.10 * loss_P3 +
                     0.20 * loss_P4 + 0.15 * loss_P5 + 0.15 * loss_P6)

        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        total = (loss_recon +
                 self.lambda_corrector * loss_corrector +
                 self.lambda_pred * loss_pred +
                 self.lambda_preserve * loss_preserve +
                 self.lambda_detail * loss_detail)

        # Per-group loss breakdown for monitoring
        per_group_details = {f'loss_{k}': v.detach() for k, v in per_group_losses.items()}

        return {
            'total': total,
            'reconstruction': loss_recon.detach(),
            'corrector': loss_corrector.detach() if isinstance(loss_corrector, torch.Tensor) else loss_corrector,
            'predicate': loss_pred.detach() if isinstance(loss_pred, torch.Tensor) else loss_pred,
            'preserve': loss_preserve.detach(),
            'detail': loss_detail.detach(),
            'P1': loss_P1.detach() if isinstance(loss_P1, torch.Tensor) else loss_P1,
            'P2': loss_P2.detach() if isinstance(loss_P2, torch.Tensor) else loss_P2,
            'P3': loss_P3.detach() if isinstance(loss_P3, torch.Tensor) else loss_P3,
            'P4': loss_P4.detach() if isinstance(loss_P4, torch.Tensor) else loss_P4,
            'P5': loss_P5.detach() if isinstance(loss_P5, torch.Tensor) else loss_P5,
            'P6': loss_P6.detach() if isinstance(loss_P6, torch.Tensor) else loss_P6,
            **per_group_details,
        }


# =============================================================================
# DATA LOADING (Memory Efficient)
# =============================================================================

class OCTDataset(Dataset):
    """OCT dataset with lazy loading - inherits from torch Dataset for DataLoader."""

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None,
                 patch_size: int = 128, is_train: bool = True):
        super().__init__()
        self.samples = []
        self.patch_size = patch_size
        self.is_train = is_train

        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        if max_samples and len(lines) > max_samples:
            # Use linspace for truly uniform sampling across the dataset
            indices = np.linspace(0, len(lines) - 1, max_samples, dtype=int)
            lines = [lines[i] for i in indices]

        for line_idx, line in enumerate(lines):
            try:
                data = json.loads(line)
                self.samples.append({
                    'clean_path': data['clean_path'],
                    'noisy_path': data['noisy_path'],
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping malformed line {line_idx}: {e}", flush=True)
                continue

        # Free parsed lines to save memory
        del lines

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # CPU-optimized image loading
        with Image.open(sample['clean_path']) as clean_img:
            clean_np = np.array(clean_img.convert('L'), dtype=np.float32)
        with Image.open(sample['noisy_path']) as noisy_img:
            noisy_np = np.array(noisy_img.convert('L'), dtype=np.float32)

        # Normalize in-place with numpy (faster than torch)
        clean_np *= (1.0 / 255.0)
        noisy_np *= (1.0 / 255.0)

        if self.is_train and self.patch_size > 0:
            H, W = clean_np.shape
            ps = self.patch_size

            if H >= ps and W >= ps:
                # Random crop in numpy (faster than torch slicing)
                top = np.random.randint(0, H - ps + 1)
                left = np.random.randint(0, W - ps + 1)
                clean_np = clean_np[top:top+ps, left:left+ps].copy()
                noisy_np = noisy_np[top:top+ps, left:left+ps].copy()
            else:
                # Pad with numpy
                pad_h = max(0, ps - H)
                pad_w = max(0, ps - W)
                if pad_h > 0 or pad_w > 0:
                    clean_np = np.pad(clean_np, ((0, pad_h), (0, pad_w)), mode='edge')
                    noisy_np = np.pad(noisy_np, ((0, pad_h), (0, pad_w)), mode='edge')
                clean_np = clean_np[:ps, :ps]
                noisy_np = noisy_np[:ps, :ps]

            # Random flips in numpy (faster)
            if np.random.rand() > 0.5:
                clean_np = np.ascontiguousarray(clean_np[:, ::-1])
                noisy_np = np.ascontiguousarray(noisy_np[:, ::-1])
            if np.random.rand() > 0.5:
                clean_np = np.ascontiguousarray(clean_np[::-1, :])
                noisy_np = np.ascontiguousarray(noisy_np[::-1, :])

        # Convert to tensor at the end (faster than early conversion)
        clean = torch.from_numpy(clean_np).unsqueeze(0)
        noisy = torch.from_numpy(noisy_np).unsqueeze(0)

        return {'clean': clean, 'noisy': noisy}


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Speed-optimized trainer with DataLoader support."""

    def __init__(self, model: NeuroSymbolicDenoiserV2,
                 train_dataset: OCTDataset,
                 val_dataset: OCTDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4,
                 finetune_backbone: bool = True,
                 num_workers: int = 2):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

        # CPU-optimized DataLoaders (no pin_memory, fewer workers)
        # Too many workers on CPU can cause overhead - use 0-2
        cpu_workers = min(num_workers, 2)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_workers,
            pin_memory=False,  # Not needed for CPU
            drop_last=False,
            persistent_workers=cpu_workers > 0,
            prefetch_factor=2 if cpu_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Validation uses variable-size images
            shuffle=False,
            num_workers=0,  # No workers for validation (overhead not worth it)
            pin_memory=False,
        )

        # Setup optimizer - include learnable thresholds if available
        # Get threshold parameters (may be empty if not learnable)
        threshold_params = []
        if hasattr(model, 'predicates') and hasattr(model.predicates, 'threshold_logits'):
            threshold_params = list(model.predicates.threshold_logits.parameters())

        if finetune_backbone:
            param_groups = [
                {'params': model.corrector.parameters(), 'lr': lr * 3},
                {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            ]
            if threshold_params:
                param_groups.append({'params': threshold_params, 'lr': lr * 0.5})  # Slow threshold learning
            self.optimizer = torch.optim.Adam(param_groups)
            print("Joint training: backbone (lr=1e-5) + corrector (lr=3e-4) + thresholds (lr=5e-5)")
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
            param_groups = [{'params': model.corrector.parameters(), 'lr': lr * 3}]
            if threshold_params:
                param_groups.append({'params': threshold_params, 'lr': lr * 0.5})
            self.optimizer = torch.optim.Adam(param_groups)
            print("Frozen backbone: training corrector + thresholds")

        self.loss_fn = NeuroSymbolicLossV2()
        self.best_score = 0

        print(f"Training: {len(train_dataset)} samples, batch_size={batch_size}, workers={num_workers}")
        print(f"Validation: {len(val_dataset)} samples")

    def train_epoch(self) -> Dict:
        """Train one epoch with speed-optimized batch processing."""
        self.model.train()

        # Accumulators for metrics (avoid list appends for speed)
        total_loss = 0.0
        total_psnr = 0.0
        total_detail = 0.0
        total_corrector = 0.0  # Per-group specialized loss
        total_correction_mag = 0.0
        total_p1, total_p2, total_p3, total_p4, total_p5, total_p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        n_corrections = 0
        samples_processed = 0
        start_time = time.time()

        # Cache predicates module reference
        predicates_module = getattr(self.model, 'predicates', None)

        for batch in self.train_loader:
            clean = batch['clean']
            noisy = batch['noisy']
            batch_size = clean.shape[0]

            # Forward - pass clean for iterative loss computation
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(noisy, clean=clean, return_details=False)

            # Loss
            losses = self.loss_fn(output, clean, noisy=noisy, predicates_module=predicates_module)
            loss = losses['total']

            # Backward (before extracting metrics to overlap compute)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate metrics (extracted after backward for potential overlap)
            with torch.no_grad():
                mse = F.mse_loss(output['denoised'], clean).item()
                total_psnr += 10 * np.log10(1.0 / max(mse, 1e-10))

            total_loss += losses['total'].item()
            total_detail += losses['detail'].item() if isinstance(losses['detail'], torch.Tensor) else losses['detail']
            total_corrector += losses['corrector'].item() if isinstance(losses['corrector'], torch.Tensor) else losses['corrector']
            total_p1 += 1 - (losses['P1'].item() if isinstance(losses['P1'], torch.Tensor) else losses['P1'])
            total_p2 += 1 - (losses['P2'].item() if isinstance(losses['P2'], torch.Tensor) else losses['P2'])
            total_p3 += 1 - (losses['P3'].item() if isinstance(losses['P3'], torch.Tensor) else losses['P3'])
            total_p4 += 1 - (losses['P4'].item() if isinstance(losses['P4'], torch.Tensor) else losses['P4'])
            total_p5 += 1 - (losses['P5'].item() if isinstance(losses['P5'], torch.Tensor) else losses['P5'])
            total_p6 += 1 - (losses['P6'].item() if isinstance(losses['P6'], torch.Tensor) else losses['P6'])

            # Track correction magnitude
            if output.get('correction_applied', False):
                total_correction_mag += output.get('correction_magnitude', 0.0)
                n_corrections += 1

            n_batches += 1
            samples_processed += batch_size

            # No del needed - variables are reassigned each iteration

        elapsed = time.time() - start_time
        clear_memory()

        # Return averages
        n = max(n_batches, 1)
        nc = max(n_corrections, 1)
        return {
            'loss': total_loss / n,
            'psnr': total_psnr / n,
            'detail': total_detail / n,
            'corrector': total_corrector / n,  # Per-group specialized loss
            'correction_mag': total_correction_mag / nc if n_corrections > 0 else 0.0,
            'correction_rate': n_corrections / n,
            'P1': total_p1 / n,
            'P2': total_p2 / n,
            'P3': total_p3 / n,
            'P4': total_p4 / n,
            'P5': total_p5 / n,
            'P6': total_p6 / n,
            'time': elapsed,
            'samples_per_sec': samples_processed / max(elapsed, 1e-6)
        }

    @staticmethod
    def compute_ssim_fast(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute SSIM with smaller kernel for speed (7x7 instead of 11x11)."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2

        # Use 7x7 kernel with stride 2 for ~4x speedup
        mu1 = F.avg_pool2d(img1, 7, stride=2, padding=3)
        mu2 = F.avg_pool2d(img2, 7, stride=2, padding=3)

        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # FIX: clamp variance to min=0 to prevent numerical issues from float precision
        # CPU optimized: use .square() and in-place ops
        sigma1_sq = (F.avg_pool2d(img1.square(), 7, stride=2, padding=3) - mu1_sq).clamp_(min=0)
        sigma2_sq = (F.avg_pool2d(img2.square(), 7, stride=2, padding=3) - mu2_sq).clamp_(min=0)
        sigma12 = F.avg_pool2d(img1 * img2, 7, stride=2, padding=3) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    @torch.inference_mode()  # Faster than torch.no_grad()
    def validate(self, max_size: int = 256, quick: bool = True) -> Dict:
        """
        Validate model with speed optimizations.

        Args:
            max_size: Max image dimension (smaller = faster). Default 256 for speed.
            quick: If True, use fast SSIM and skip some metrics for speed.
        """
        self.model.eval()

        metrics = {
            # Objective metrics: backbone vs corrected (both compared to clean)
            'psnr_backbone': [], 'psnr_corrected': [],
            'ssim_backbone': [], 'ssim_corrected': [],
            # Predicate scores for all 6 predicates
            'P1_init': [], 'P2_init': [], 'P3_init': [], 'P4_init': [], 'P5_init': [], 'P6_init': [],
            'P1_final': [], 'P2_final': [], 'P3_final': [], 'P4_final': [], 'P5_final': [], 'P6_final': [],
            'P1_pass': [], 'P2_pass': [], 'P3_pass': [], 'P4_pass': [], 'P5_pass': [], 'P6_pass': [],
            'all_pass': [],
        }

        start_time = time.time()

        for batch in self.val_loader:
            clean = batch['clean']
            noisy = batch['noisy']

            # Center-crop large images (smaller max_size = faster)
            _, _, H, W = clean.shape
            if H > max_size or W > max_size:
                new_h = min(H, max_size)
                new_w = min(W, max_size)
                h_start = (H - new_h) // 2
                w_start = (W - new_w) // 2
                clean = clean[:, :, h_start:h_start+new_h, w_start:w_start+new_w].contiguous()
                noisy = noisy[:, :, h_start:h_start+new_h, w_start:w_start+new_w].contiguous()

            output = self.model(noisy, return_details=False)

            backbone_out = output['initial']
            corrected_out = output['denoised']

            # PSNR: backbone vs corrected (both compared to clean)
            mse_backbone = F.mse_loss(backbone_out, clean).item()
            mse_corrected = F.mse_loss(corrected_out, clean).item()
            metrics['psnr_backbone'].append(10 * np.log10(1.0 / max(mse_backbone, 1e-10)))
            metrics['psnr_corrected'].append(10 * np.log10(1.0 / max(mse_corrected, 1e-10)))

            # SSIM: use fast version for quicker validation
            metrics['ssim_backbone'].append(self.compute_ssim_fast(backbone_out, clean))
            metrics['ssim_corrected'].append(self.compute_ssim_fast(corrected_out, clean))

            # Initial predicates (on backbone output) - 6 predicates
            init_pred = output.get('initial_predicate_results', {})
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
                if init_pred and name in init_pred:
                    score = init_pred[name]['score']
                    metrics[f'{name}_init'].append(score.item() if isinstance(score, torch.Tensor) else score)
                else:
                    score = output['predicate_results'][name]['score']
                    metrics[f'{name}_init'].append(score.item() if isinstance(score, torch.Tensor) else score)

            # Final predicates (on corrected output) - 6 predicates
            final_pred = output['predicate_results']
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
                score = final_pred[name]['score']
                metrics[f'{name}_final'].append(score.item() if isinstance(score, torch.Tensor) else score)
                metrics[f'{name}_pass'].append(float(final_pred[name]['passed']))

            metrics['all_pass'].append(float(final_pred['all_passed']))

            del output, clean, noisy, init_pred, final_pred, backbone_out, corrected_out

        elapsed = time.time() - start_time
        clear_memory()

        return {k: (np.mean(v) if len(v) > 0 else 0.0) for k, v in metrics.items()} | {'time': elapsed}

    def print_results(self, epoch: int, train: Dict, val: Dict):
        """Print training results."""
        print(f"\n{'='*70}", flush=True)
        print(f"EPOCH {epoch}", flush=True)
        print(f"{'='*70}", flush=True)

        print(f"\n[TRAINING] Loss: {train['loss']:.4f}, PSNR: {train['psnr']:.2f} dB", flush=True)
        print(f"  Losses: detail={train['detail']:.4f}, corrector={train.get('corrector', 0):.4f}", flush=True)
        print(f"  Predicates: P1={train['P1']:.3f}, P2={train['P2']:.3f}, P3={train['P3']:.3f}, "
              f"P4={train['P4']:.3f}, P5={train['P5']:.3f}, P6={train['P6']:.3f}", flush=True)
        corr_mag = train.get('correction_mag', 0)
        corr_rate = train.get('correction_rate', 0) * 100
        print(f"  Corrector: magnitude={corr_mag:.4f}, applied={corr_rate:.1f}%", flush=True)
        print(f"  Time: {train['time']:.1f}s ({train['samples_per_sec']:.2f} samples/sec)", flush=True)

        print(f"\n[VALIDATION] Improvement from Backbone → Corrected:", flush=True)

        # Objective metrics: PSNR and SSIM improvement
        delta_psnr = val['psnr_corrected'] - val['psnr_backbone']
        delta_ssim = val['ssim_corrected'] - val['ssim_backbone']
        print(f"  PSNR: {val['psnr_backbone']:.2f} → {val['psnr_corrected']:.2f} dB (Δ={delta_psnr:+.3f})", flush=True)
        print(f"  SSIM: {val['ssim_backbone']:.4f} → {val['ssim_corrected']:.4f} (Δ={delta_ssim:+.4f})", flush=True)

        print(f"\n  Predicate Scores (Backbone → Corrected):", flush=True)
        for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            init = val.get(f'{name}_init')
            final = val[f'{name}_final']
            delta = final - init if init is not None else 0
            init_display = init if init is not None else 0
            pass_rate = val[f'{name}_pass'] * 100
            print(f"    {name}: {init_display:.3f} → {final:.3f} (Δ={delta:+.3f}) | Pass: {pass_rate:.1f}%", flush=True)

        all_pass = val['all_pass'] * 100
        print(f"\n  ALL PASS Rate: {all_pass:.1f}%", flush=True)

        # Print learned thresholds if available
        if hasattr(self.model, 'predicates') and hasattr(self.model.predicates, 'get_thresholds_dict'):
            thresholds = self.model.predicates.get_thresholds_dict()
            print(f"  Learned Thresholds: P1={thresholds['P1']:.3f}, P2={thresholds['P2']:.3f}, "
                  f"P3={thresholds['P3']:.3f}, P4={thresholds['P4']:.3f}", flush=True)

        print(f"  Validation time: {val['time']:.1f}s", flush=True)
        print(f"{'='*70}", flush=True)
        sys.stdout.flush()

    def train(self, epochs: int = 20, save_path: str = 'outputs/neuro_symbolic_v2',
              val_every: int = 1, quick_val: bool = True):
        """
        Full training loop.

        Args:
            epochs: Number of training epochs
            save_path: Directory to save checkpoints
            val_every: Validate every N epochs (default 1, set higher for speed)
            quick_val: Use fast validation settings (smaller crops, fast SSIM)
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}", flush=True)
        print("# NEURO-SYMBOLIC OCT DENOISING V2", flush=True)
        print(f"{'#'*70}", flush=True)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()

            # Validate every val_every epochs (or always on last epoch)
            if epoch % val_every == 0 or epoch == epochs:
                val_metrics = self.validate(max_size=256 if quick_val else 512, quick=quick_val)
                self.print_results(epoch, train_metrics, val_metrics)
            else:
                # Quick print without validation
                print(f"\n[EPOCH {epoch}] Loss: {train_metrics['loss']:.4f}, "
                      f"PSNR: {train_metrics['psnr']:.2f} dB, "
                      f"Time: {train_metrics['time']:.1f}s", flush=True)
                val_metrics = None

            # Only compute score and save if we validated
            if val_metrics is None:
                clear_memory()
                continue

            # Score: predicate improvement + PSNR/SSIM preservation (6 predicates)
            avg_pred = (val_metrics['P1_final'] + val_metrics['P2_final'] +
                       val_metrics['P3_final'] + val_metrics['P4_final'] +
                       val_metrics['P5_final'] + val_metrics['P6_final']) / 6
            psnr_delta = val_metrics['psnr_corrected'] - val_metrics['psnr_backbone']
            ssim_delta = val_metrics['ssim_corrected'] - val_metrics['ssim_backbone']
            # Bonus if correction doesn't degrade PSNR/SSIM
            psnr_bonus = max(0, min(0.1, (psnr_delta + 2) / 20))
            ssim_bonus = max(0, min(0.05, ssim_delta * 5))
            score = avg_pred + psnr_bonus + ssim_bonus + val_metrics['all_pass'] * 0.1

            if score > self.best_score:
                self.best_score = score
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'val_metrics': val_metrics,
                }, f"{save_path}/best_model.pth")
                print(f"\n  *** Saved best model (score: {score:.4f}) ***", flush=True)

            clear_memory()

        print(f"\n{'#'*70}", flush=True)
        print("# TRAINING COMPLETE", flush=True)
        print(f"{'#'*70}", flush=True)
        sys.stdout.flush()


# =============================================================================
# TRAINER V3: ADAPTIVE LAMBDA
# =============================================================================

class TrainerV3:
    """
    Trainer for NeuroSymbolicDenoiserV3 with adaptive lambda.

    Key differences from TrainerV2:
    - Uses NeuroSymbolicLossV3 (dual objective)
    - Separate learning rates for lambda predictor vs corrector
    - Tracks lambda statistics
    """

    def __init__(self, model: NeuroSymbolicDenoiserV3,
                 train_dataset: OCTDataset,
                 val_dataset: OCTDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4,
                 finetune_backbone: bool = False,
                 num_workers: int = 2):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

        # DataLoaders
        cpu_workers = min(num_workers, 2)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=cpu_workers, pin_memory=False, drop_last=False,
            persistent_workers=cpu_workers > 0,
            prefetch_factor=2 if cpu_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False,
        )

        # Setup optimizer with separate learning rates:
        # - Lambda predictor: Higher LR (learns correction strength)
        # - Corrector: Medium LR (learns what corrections to make)
        # - Backbone: Low LR (fine-tuning)
        # - Loss thresholds: Low LR (stability)
        if finetune_backbone:
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},  # Lambda learns fast
                {'params': model.corrector.parameters(), 'lr': lr * 3},
                {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            ]
            print("Joint training: backbone (lr=1e-5) + corrector (lr=3e-4) + lambda (lr=5e-4)")
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
                {'params': model.corrector.parameters(), 'lr': lr * 3},
            ]
            print("Frozen backbone: training corrector (lr=3e-4) + lambda (lr=5e-4)")

        self.optimizer = torch.optim.Adam(param_groups)
        self.loss_fn = NeuroSymbolicLossV3()

        # Add loss thresholds to optimizer
        threshold_params = [p for n, p in self.loss_fn.named_parameters() if 'threshold' in n]
        if threshold_params:
            self.optimizer.add_param_group({'params': threshold_params, 'lr': lr * 0.1})

        self.best_score = 0

        print(f"Training: {len(train_dataset)} samples, batch_size={batch_size}")
        print(f"Validation: {len(val_dataset)} samples")

    def train_epoch(self) -> Dict:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        total_corrector = 0.0
        total_lambda = 0.0
        total_psnr = 0.0
        total_lambda_edge = 0.0
        total_lambda_texture = 0.0
        total_lambda_smooth = 0.0
        total_p1, total_p2, total_p3, total_p4, total_p5, total_p6 = 0., 0., 0., 0., 0., 0.
        n_batches = 0
        start_time = time.time()

        for batch in self.train_loader:
            clean = batch['clean']
            noisy = batch['noisy']

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(noisy, clean=clean)

            losses = self.loss_fn(output, clean, noisy)
            loss = losses['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                mse = F.mse_loss(output['denoised'], clean).item()
                total_psnr += 10 * np.log10(1.0 / max(mse, 1e-10))

            total_loss += losses['total'].item()
            total_corrector += losses['corrector'].item()
            total_lambda += losses['lambda'].item()
            total_lambda_edge += losses['lambda_edge_mean'].item()
            total_lambda_texture += losses['lambda_texture_mean'].item()
            total_lambda_smooth += losses['lambda_smooth_mean'].item()
            total_p1 += losses['P1'] if isinstance(losses['P1'], float) else losses['P1'].item()
            total_p2 += losses['P2'] if isinstance(losses['P2'], float) else losses['P2'].item()
            total_p3 += losses['P3'] if isinstance(losses['P3'], float) else losses['P3'].item()
            total_p4 += losses['P4'] if isinstance(losses['P4'], float) else losses['P4'].item()
            total_p5 += losses['P5'] if isinstance(losses['P5'], float) else losses['P5'].item()
            total_p6 += losses['P6'] if isinstance(losses['P6'], float) else losses['P6'].item()

            n_batches += 1

        elapsed = time.time() - start_time
        n = max(n_batches, 1)

        return {
            'loss': total_loss / n,
            'corrector': total_corrector / n,
            'lambda_loss': total_lambda / n,
            'psnr': total_psnr / n,
            'lambda_edge': total_lambda_edge / n,
            'lambda_texture': total_lambda_texture / n,
            'lambda_smooth': total_lambda_smooth / n,
            'P1': total_p1 / n, 'P2': total_p2 / n, 'P3': total_p3 / n,
            'P4': total_p4 / n, 'P5': total_p5 / n, 'P6': total_p6 / n,
            'time': elapsed,
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate model."""
        self.model.eval()

        psnr_backbone_list = []
        psnr_corrected_list = []
        ssim_backbone_list = []
        ssim_corrected_list = []
        pred_scores = {f'P{i}': [] for i in range(1, 7)}
        lambda_stats = {'edge': [], 'texture': [], 'smooth': []}
        all_pass_count = 0

        for batch in self.val_loader:
            clean = batch['clean']
            noisy = batch['noisy']

            output = self.model(noisy, clean)
            initial = output['initial']
            corrected = output['denoised']

            # PSNR
            mse_backbone = F.mse_loss(initial, clean).item()
            mse_corrected = F.mse_loss(corrected, clean).item()
            psnr_backbone_list.append(10 * np.log10(1.0 / max(mse_backbone, 1e-10)))
            psnr_corrected_list.append(10 * np.log10(1.0 / max(mse_corrected, 1e-10)))

            # SSIM (simplified)
            ssim_backbone_list.append(Trainer.compute_ssim_fast(initial, clean))
            ssim_corrected_list.append(Trainer.compute_ssim_fast(corrected, clean))

            # Predicates
            pred = output['predicate_results']
            for i in range(1, 7):
                name = f'P{i}'
                score = pred[name]['score']
                pred_scores[name].append(score.item() if isinstance(score, torch.Tensor) else score)

            if pred['all_passed']:
                all_pass_count += 1

            # Lambda stats
            lambda_maps = output.get('lambda_maps', {})
            for key in ['edge', 'texture', 'smooth']:
                if key in lambda_maps:
                    lambda_stats[key].append(lambda_maps[key].mean().item())

        n = len(psnr_backbone_list)
        return {
            'psnr_backbone': np.mean(psnr_backbone_list),
            'psnr_corrected': np.mean(psnr_corrected_list),
            'ssim_backbone': np.mean(ssim_backbone_list),
            'ssim_corrected': np.mean(ssim_corrected_list),
            'P1': np.mean(pred_scores['P1']), 'P2': np.mean(pred_scores['P2']),
            'P3': np.mean(pred_scores['P3']), 'P4': np.mean(pred_scores['P4']),
            'P5': np.mean(pred_scores['P5']), 'P6': np.mean(pred_scores['P6']),
            'all_pass': all_pass_count / n,
            'lambda_edge': np.mean(lambda_stats['edge']) if lambda_stats['edge'] else 0,
            'lambda_texture': np.mean(lambda_stats['texture']) if lambda_stats['texture'] else 0,
            'lambda_smooth': np.mean(lambda_stats['smooth']) if lambda_stats['smooth'] else 0,
        }

    def train(self, epochs: int = 15, val_every: int = 3, save_path: str = 'outputs/nsnd_v3'):
        """Full training loop."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}", flush=True)
        print("# NEURO-SYMBOLIC OCT DENOISING V3 (ADAPTIVE LAMBDA)", flush=True)
        print(f"{'#'*70}", flush=True)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()

            if epoch % val_every == 0 or epoch == epochs:
                val_metrics = self.validate()

                # Print results with flush for real-time output
                print(f"\n{'='*70}", flush=True)
                print(f"EPOCH {epoch}", flush=True)
                print(f"{'='*70}", flush=True)
                print(f"\n[TRAINING]", flush=True)
                print(f"  Loss: {train_metrics['loss']:.4f} (corr={train_metrics['corrector']:.4f}, λ={train_metrics['lambda_loss']:.4f})", flush=True)
                print(f"  PSNR: {train_metrics['psnr']:.2f} dB", flush=True)
                print(f"  Lambda: edge={train_metrics['lambda_edge']:.3f}, texture={train_metrics['lambda_texture']:.3f}, smooth={train_metrics['lambda_smooth']:.3f}", flush=True)
                print(f"  Predicates: P1={train_metrics['P1']:.3f}, P2={train_metrics['P2']:.3f}, P3={train_metrics['P3']:.3f}", flush=True)
                print(f"              P4={train_metrics['P4']:.3f}, P5={train_metrics['P5']:.3f}, P6={train_metrics['P6']:.3f}", flush=True)
                print(f"  Time: {train_metrics['time']:.1f}s", flush=True)

                print(f"\n[VALIDATION]", flush=True)
                psnr_delta = val_metrics['psnr_corrected'] - val_metrics['psnr_backbone']
                ssim_delta = val_metrics['ssim_corrected'] - val_metrics['ssim_backbone']
                print(f"  PSNR: {val_metrics['psnr_backbone']:.2f} → {val_metrics['psnr_corrected']:.2f} dB (Δ={psnr_delta:+.3f})", flush=True)
                print(f"  SSIM: {val_metrics['ssim_backbone']:.4f} → {val_metrics['ssim_corrected']:.4f} (Δ={ssim_delta:+.4f})", flush=True)
                print(f"  Lambda: edge={val_metrics['lambda_edge']:.3f}, texture={val_metrics['lambda_texture']:.3f}, smooth={val_metrics['lambda_smooth']:.3f}", flush=True)
                print(f"  Predicates: P1={val_metrics['P1']:.3f}, P2={val_metrics['P2']:.3f}, P3={val_metrics['P3']:.3f}", flush=True)
                print(f"              P4={val_metrics['P4']:.3f}, P5={val_metrics['P5']:.3f}, P6={val_metrics['P6']:.3f}", flush=True)
                print(f"  ALL PASS Rate: {val_metrics['all_pass']*100:.1f}%", flush=True)
                print(f"{'='*70}", flush=True)

                # Compute score
                avg_pred = (val_metrics['P1'] + val_metrics['P2'] + val_metrics['P3'] +
                           val_metrics['P4'] + val_metrics['P5'] + val_metrics['P6']) / 6
                psnr_bonus = max(0, min(0.1, (psnr_delta + 2) / 20))
                score = avg_pred + psnr_bonus + val_metrics['all_pass'] * 0.1

                if score > self.best_score:
                    self.best_score = score
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'loss_state_dict': self.loss_fn.state_dict(),
                        'val_metrics': val_metrics,
                    }, f"{save_path}/best_model_v3.pth")
                    print(f"\n  *** Saved best model (score: {score:.4f}) ***", flush=True)

            else:
                print(f"\n[EPOCH {epoch}] Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB, "
                      f"λ_edge={train_metrics['lambda_edge']:.3f}, Time: {train_metrics['time']:.1f}s", flush=True)

            clear_memory()
            sys.stdout.flush()

        print(f"\n{'#'*70}", flush=True)
        print("# TRAINING COMPLETE", flush=True)
        print(f"{'#'*70}\n", flush=True)


def main_v3():
    """Main function for V3 (adaptive lambda) training."""
    print("="*70)
    print("NEURO-SYMBOLIC OCT DENOISING V3 (ADAPTIVE LAMBDA)")
    print(f"Using {_NUM_THREADS} CPU threads")
    print("="*70)

    # Configuration
    FINETUNE_BACKBONE = False
    BATCH_SIZE = 4
    EPOCHS = 15
    TRAIN_SAMPLES = 100
    VAL_SAMPLES = 20
    NUM_WORKERS = 2
    VAL_EVERY = 3
    PATCH_SIZE = 96

    # Create V3 model
    print("\nInitializing V3 model...")
    model = NeuroSymbolicDenoiserV3(backbone_type='nafnet', width=64)

    # Load pretrained backbone
    backbone_path = 'outputs/nafnet_pku37/nafnet_best.pth'
    if Path(backbone_path).exists():
        print("Loading pretrained backbone...")
        model.load_pretrained_backbone(backbone_path)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lambda_params = sum(p.numel() for p in model.lambda_predictor.parameters())
    corrector_params = sum(p.numel() for p in model.corrector.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    print(f"  Lambda predictor: {lambda_params:,}")
    print(f"  Corrector: {corrector_params:,}")

    # Load data
    print("\nLoading data...")
    train_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_train.jsonl',
        max_samples=TRAIN_SAMPLES,
        patch_size=PATCH_SIZE,
        is_train=True
    )
    val_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_val.jsonl',
        max_samples=VAL_SAMPLES,
        patch_size=0,
        is_train=False
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create V3 trainer
    trainer = TrainerV3(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        finetune_backbone=FINETUNE_BACKBONE,
        num_workers=NUM_WORKERS,
    )

    # Train
    trainer.train(epochs=EPOCHS, val_every=VAL_EVERY)


# =============================================================================
# MAIN
# =============================================================================

def try_compile_model(model: nn.Module) -> nn.Module:
    """Try to compile model with torch.compile for PyTorch 2.0+."""
    if hasattr(torch, 'compile'):
        try:
            # Use 'reduce-overhead' mode for best training performance
            compiled = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile (reduce-overhead mode)")
            return compiled
        except Exception as e:
            print(f"torch.compile failed, using eager mode: {e}")
    return model


def main():
    print("="*70, flush=True)
    print("NEURO-SYMBOLIC OCT DENOISING V2 (CPU Optimized)", flush=True)
    print(f"Using {_NUM_THREADS} CPU threads", flush=True)
    print("="*70, flush=True)

    # CPU-OPTIMIZED Configuration
    FINETUNE_BACKBONE = False  # Freeze backbone for faster training on CPU
    BATCH_SIZE = 4             # Smaller batch for CPU (less memory pressure)
    EPOCHS = 15
    TRAIN_SAMPLES = 100        # Training images
    VAL_SAMPLES = 20           # Testing images
    NUM_WORKERS = 2            # 0-2 workers optimal for CPU
    USE_COMPILE = False        # torch.compile often slower on CPU
    VAL_EVERY = 3              # Validate every 3 epochs
    QUICK_VAL = True           # Use fast validation settings
    PATCH_SIZE = 96            # Smaller patches for CPU speed

    # Create model with smaller width for CPU
    print("\nInitializing model...", flush=True)
    model = NeuroSymbolicDenoiserV2(backbone_type='nafnet', width=64)

    # Load pretrained backbone
    backbone_path = 'outputs/nafnet_pku37/nafnet_best.pth'
    if Path(backbone_path).exists():
        print("Loading pretrained backbone...", flush=True)
        model.load_pretrained_backbone(backbone_path)

    # Optionally compile model for faster execution
    if USE_COMPILE:
        model = try_compile_model(model)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable", flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    train_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_train.jsonl',
        max_samples=TRAIN_SAMPLES,
        patch_size=PATCH_SIZE,  # Smaller patches for CPU speed
        is_train=True
    )
    val_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_val.jsonl',
        max_samples=VAL_SAMPLES,
        patch_size=0,
        is_train=False
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)

    # Create trainer with parallel data loading
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        finetune_backbone=FINETUNE_BACKBONE,
        num_workers=NUM_WORKERS,
    )

    # Train with speed optimizations
    trainer.train(epochs=EPOCHS, val_every=VAL_EVERY, quick_val=QUICK_VAL)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'v3':
        main_v3()
    else:
        main()
