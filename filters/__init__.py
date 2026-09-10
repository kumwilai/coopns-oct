"""
Speckle Filters for OCT Image Denoising

This module provides classical speckle-aware denoising filters that operate
in log-domain for multiplicative noise. All filters support ENL-based auto-tuning.

Available Filters:
- Lee Filter: Adaptive filter using local statistics
- Kuan Filter: MMSE-based variant of Lee filter
- Frost Filter: Exponentially weighted adaptive filter
- Bilateral Filter: Edge-preserving spatial-range filter
- Guided Filter: Fast edge-preserving filter

Usage:
    from models.filters import lee_filter_log, bilateral_filter_log

    # Apply Lee filter
    denoised = lee_filter_log(noisy_image, win=5)

    # Apply bilateral filter with auto-tuning
    denoised = bilateral_filter_log(noisy_image, win=5)
"""

from .enl_utils import (
    estimate_enl_map_linear,
    estimate_enl_global_linear,
    estimate_enl_from_homogeneous_region,
    to_log,
    from_log,
    normalize_log,
    denormalize_log,
    torch_to_numpy,
    numpy_to_torch,
    estimate_enl_torch
)

from .lee_kuan_frost import (
    lee_filter_log,
    kuan_filter_log,
    frost_filter_log,
    adaptive_wiener_log
)

from .bilateral_guided import (
    bilateral_filter_log,
    guided_filter_log,
    weighted_guided_filter_log,
    fast_bilateral_log
)

__all__ = [
    # ENL utilities
    'estimate_enl_map_linear',
    'estimate_enl_global_linear',
    'estimate_enl_from_homogeneous_region',
    'to_log',
    'from_log',
    'normalize_log',
    'denormalize_log',
    'torch_to_numpy',
    'numpy_to_torch',
    'estimate_enl_torch',
    # Lee/Kuan/Frost filters
    'lee_filter_log',
    'kuan_filter_log',
    'frost_filter_log',
    'adaptive_wiener_log',
    # Bilateral/Guided filters
    'bilateral_filter_log',
    'guided_filter_log',
    'weighted_guided_filter_log',
    'fast_bilateral_log',
]

__version__ = '1.0.0'
