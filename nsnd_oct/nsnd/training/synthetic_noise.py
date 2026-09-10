"""
Synthetic OCT noise generation for training and evaluation

Implements realistic OCT noise models:
- Multiplicative speckle (Rayleigh/Gamma)
- Horizontal banding artifacts
- Additive Gaussian noise
- Poisson shot noise
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple


def _gaussian_kernel2d(sigma: float, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    if sigma <= 0:
        return torch.tensor([], device=device, dtype=dtype), 0
    radius = int(max(1, math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d, radius


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    kernel_2d, radius = _gaussian_kernel2d(sigma, x.device, x.dtype)
    if radius <= 0:
        return x
    kernel = kernel_2d.view(1, 1, kernel_2d.size(0), kernel_2d.size(1))
    return F.conv2d(x, kernel, padding=radius)


def signal_dependent_speckle(
    clean: torch.Tensor,
    base_k: float = 4.0,
    snr_factor: float = 0.5,
    correlation_sigma: float = 1.2,
) -> torch.Tensor:
    """
    Generate signal-dependent speckle (mean=1) with spatial correlation.
    """
    eps = 1e-6
    clean_max = clean.amax(dim=(2, 3), keepdim=True)
    clean_norm = clean / (clean_max + eps)
    local_k = base_k * (1.0 + snr_factor * clean_norm)
    local_k = local_k.clamp_min(1e-3)

    gamma = torch.distributions.Gamma(concentration=local_k, rate=local_k)
    speckle = gamma.sample()

    if correlation_sigma > 0:
        speckle = _gaussian_blur(speckle, correlation_sigma)
        speckle = speckle / (speckle.mean(dim=(2, 3), keepdim=True) + eps)

    return speckle


def add_oct_noise_mixture(
    clean_img: torch.Tensor,
    weights: Dict[str, float],
    noise_params: Optional[Dict] = None,
    return_maps: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Add controlled mixture of OCT noise types

    Args:
        clean_img: Clean input [B, 1, H, W] in range [0, 1]
        weights: Dict of noise weights {'speckle': 0.6, 'banding': 0.2, ...}
        noise_params: Optional dict of noise parameters

    Returns:
        noisy: Noisy image [B, 1, H, W]
    """
    if noise_params is None:
        noise_params = {}

    B, C, H, W = clean_img.shape
    device = clean_img.device

    noisy = clean_img.clone()
    noise_maps = None
    if return_maps:
        noise_maps = {
            "speckle": torch.zeros_like(noisy),
            "banding": torch.zeros_like(noisy),
            "gaussian": torch.zeros_like(noisy),
            "shot": torch.zeros_like(noisy),
        }
    use_depth_profile = noise_params.get('use_depth_profile', True)
    depth_map = None
    if use_depth_profile:
        depth = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
        depth_map = depth.expand(B, 1, H, W)

    # 1. Multiplicative speckle (Gamma distribution)
    if weights.get('speckle', 0) > 0:
        prev = noisy
        use_signal_dependent = noise_params.get('use_signal_dependent_speckle', True)
        if use_signal_dependent:
            base_k = noise_params.get('speckle_base_k', noise_params.get('speckle_k', 3.0))
            snr_factor = noise_params.get('speckle_snr_factor', 0.5)
            correlation_sigma = noise_params.get('speckle_correlation', 1.2)
            speckle = signal_dependent_speckle(
                clean_img,
                base_k=float(base_k),
                snr_factor=float(snr_factor),
                correlation_sigma=float(correlation_sigma),
            )
        else:
            # Gamma speckle: shape parameter controls noise level
            k = noise_params.get('speckle_k', 3.0)  # Higher k = less noise
            scale = 1.0 / k

            # Generate Gamma-distributed speckle
            speckle = torch.from_numpy(
                np.random.gamma(k, scale, (B, 1, H, W))
            ).float().to(device)

        # Apply multiplicatively
        depth_gain = noise_params.get('speckle_depth_gain', 0.6)
        speckle_scale = 1.0 + depth_gain * depth_map if depth_map is not None else 1.0
        noisy = noisy * (1.0 + weights['speckle'] * speckle_scale * (speckle - 1.0))
        if noise_maps is not None:
            noise_maps["speckle"] = noisy - prev

    # 2. Horizontal banding artifacts
    if weights.get('banding', 0) > 0:
        prev = noisy
        freq = noise_params.get('banding_freq', 30)  # Pixels per band
        amplitude = noise_params.get('banding_amp', 0.05)

        # Create sinusoidal banding pattern
        y_coords = torch.arange(H, dtype=torch.float32, device=device)
        banding = amplitude * torch.sin(2 * np.pi * y_coords / freq)
        banding = banding.view(1, 1, H, 1).expand(B, 1, H, W)

        # Add to image
        noisy = noisy + weights['banding'] * banding
        if noise_maps is not None:
            noise_maps["banding"] = noisy - prev

    # 3. Additive Gaussian noise
    if weights.get('gaussian', 0) > 0:
        prev = noisy
        sigma = noise_params.get('gaussian_sigma', 0.04)

        gaussian_noise = torch.randn(B, 1, H, W, device=device) * sigma
        depth_gain = noise_params.get('gaussian_depth_gain', 0.4)
        gaussian_scale = 1.0 + depth_gain * depth_map if depth_map is not None else 1.0
        noisy = noisy + weights['gaussian'] * gaussian_scale * gaussian_noise
        if noise_maps is not None:
            noise_maps["gaussian"] = noisy - prev

    # 4. Poisson shot noise
    if weights.get('shot', 0) > 0:
        prev = noisy
        peak_photons = noise_params.get('shot_peak', 80.0)

        # Ensure non-negative before Poisson (required for valid rate)
        noisy_clamped = torch.clamp(noisy, min=0.0)

        # Scale to photon counts, apply Poisson, scale back
        scaled = noisy_clamped * peak_photons
        noisy_poisson = torch.poisson(scaled) / peak_photons

        depth_gain = noise_params.get('shot_depth_gain', 0.8)
        shot_scale = 1.0 + depth_gain * depth_map if depth_map is not None else 1.0

        # Blend with original
        noisy = (1 - weights['shot']) * noisy + weights['shot'] * (noisy + shot_scale * (noisy_poisson - noisy))
        if noise_maps is not None:
            noise_maps["shot"] = noisy - prev

    # Clip to valid range
    noisy = torch.clamp(noisy, 0.0, 1.0)

    if noise_maps is not None:
        return noisy, noise_maps
    return noisy


class OCTNoiseGenerator:
    """
    Flexible OCT noise generator with randomized parameters

    Samples noise compositions from realistic distributions
    """

    def __init__(
        self,
        speckle_range: Tuple[float, float] = (0.3, 0.8),
        banding_range: Tuple[float, float] = (0.0, 0.3),
        gaussian_range: Tuple[float, float] = (0.05, 0.2),
        shot_range: Tuple[float, float] = (0.0, 0.15),
        ensure_sum_one: bool = True,
        speckle_base_k: float | None = None,
        speckle_snr_factor: float = 0.5,
        speckle_correlation: float = 1.2,
        use_signal_dependent_speckle: bool = True,
    ):
        """
        Args:
            speckle_range: Min/max weight for speckle
            banding_range: Min/max weight for banding
            gaussian_range: Min/max weight for Gaussian
            shot_range: Min/max weight for shot noise
            ensure_sum_one: Whether to normalize weights to sum to 1
        """
        self.ranges = {
            'speckle': speckle_range,
            'banding': banding_range,
            'gaussian': gaussian_range,
            'shot': shot_range,
        }
        self.ensure_sum_one = ensure_sum_one
        self.speckle_base_k = speckle_base_k
        self.speckle_snr_factor = float(speckle_snr_factor)
        self.speckle_correlation = float(speckle_correlation)
        self.use_signal_dependent_speckle = bool(use_signal_dependent_speckle)

    def sample_weights(self) -> Dict[str, float]:
        """Sample random noise composition weights"""
        weights = {}

        for component, (min_w, max_w) in self.ranges.items():
            weights[component] = np.random.uniform(min_w, max_w)

        if self.ensure_sum_one:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def sample_params(self) -> Dict[str, float]:
        """Sample random noise parameters"""
        speckle_k = np.random.uniform(1.5, 4.0)
        base_k = speckle_k if self.speckle_base_k is None else float(self.speckle_base_k)
        return {
            # Speckle
            'speckle_k': base_k if self.use_signal_dependent_speckle else speckle_k,
            'speckle_base_k': base_k,
            'speckle_snr_factor': float(self.speckle_snr_factor),
            'speckle_correlation': float(self.speckle_correlation),
            'use_signal_dependent_speckle': bool(self.use_signal_dependent_speckle),
            'speckle_depth_gain': np.random.uniform(0.4, 1.0),

            # Banding
            'banding_freq': np.random.choice([15, 20, 30, 40]),  # Discrete frequencies
            'banding_amp': np.random.uniform(0.06, 0.12),

            # Gaussian
            'gaussian_sigma': np.random.uniform(0.02, 0.06),
            'gaussian_depth_gain': np.random.uniform(0.2, 0.8),

            # Shot
            'shot_peak': np.random.uniform(30.0, 120.0),
            'shot_depth_gain': np.random.uniform(0.4, 1.0),
            'use_depth_profile': True,
        }

    def generate(
        self,
        clean_img: torch.Tensor,
        weights: Optional[Dict[str, float]] = None,
        params: Optional[Dict] = None,
        return_maps: bool = False,
    ) -> Tuple[torch.Tensor, Dict, Dict] | Tuple[torch.Tensor, Dict, Dict, Dict[str, torch.Tensor]]:
        """
        Generate noisy image with random or specified noise

        Args:
            clean_img: Clean input [B, 1, H, W]
            weights: Optional fixed weights (otherwise sampled)
            params: Optional fixed params (otherwise sampled)

        Returns:
            noisy: Noisy image
            weights: Used weights
            params: Used parameters
        """
        if weights is None:
            weights = self.sample_weights()

        if params is None:
            params = self.sample_params()

        if return_maps:
            noisy, noise_maps = add_oct_noise_mixture(clean_img, weights, params, return_maps=True)
            return noisy, weights, params, noise_maps
        noisy = add_oct_noise_mixture(clean_img, weights, params, return_maps=False)
        return noisy, weights, params


def create_training_pair(
    clean_img: torch.Tensor,
    noise_generator: OCTNoiseGenerator,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Create a training pair with known noise composition

    Args:
        clean_img: Clean image [B, 1, H, W]
        noise_generator: Noise generator instance

    Returns:
        noisy: Noisy image
        clean: Original clean image
        weights: Ground truth noise composition
    """
    noisy, weights, _ = noise_generator.generate(clean_img)

    return noisy, clean_img, weights


def create_blind2unblind_pair(
    clean_img: torch.Tensor,
    noise_generator: OCTNoiseGenerator,
    mask_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create Blind2Unblind training pair

    Args:
        clean_img: Clean image [B, 1, H, W]
        noise_generator: Noise generator
        mask_ratio: Ratio of pixels to mask

    Returns:
        noisy_masked: Noisy image with masked pixels
        noisy_full: Full noisy image (target)
        mask: Binary mask [B, 1, H, W]
    """
    # Generate two independent noise realizations
    noisy_full, _, _ = noise_generator.generate(clean_img)

    # Create random mask
    B, C, H, W = clean_img.shape
    mask = (torch.rand(B, 1, H, W, device=clean_img.device) < mask_ratio).float()

    # Masked noisy image (replace masked pixels with another realization)
    noisy_alt, _, _ = noise_generator.generate(clean_img)
    noisy_masked = (1 - mask) * noisy_full + mask * noisy_alt

    return noisy_masked, noisy_full, mask
