"""
Lee, Kuan, and Frost Speckle Filters (Log-Domain)

These filters are designed for multiplicative speckle noise in OCT/SAR images.
All processing is done in log-domain to handle multiplicative noise as additive.

References:
- Lee (1980): "Digital image enhancement and noise filtering by use of local statistics"
- Kuan et al. (1985): "Adaptive noise smoothing filter for images with signal-dependent noise"
- Frost et al. (1982): "A model for radar images and its application to adaptive digital filtering"
"""

import numpy as np
from scipy import ndimage
from .enl_utils import estimate_enl_global_linear, estimate_enl_map_linear, to_log, from_log


def lee_filter_log(
    x: np.ndarray,
    win: int = 5,
    enl: float = None,
    enl_map: np.ndarray = None
) -> np.ndarray:
    """
    Lee filter in log-domain for speckle reduction.

    The Lee filter uses local statistics to adaptively smooth the image
    while preserving edges. In log-domain, it applies:

    y = μ + G * (x - μ)

    where G = max(0, (σ^2 - σ_n^2) / σ^2) is the gain factor.

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        win: Window size for local statistics (default: 5)
        enl: Global ENL value (estimated if None)
        enl_map: Local ENL map (estimated if None)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    x = x.astype(np.float32)

    # Estimate ENL if not provided
    if enl is None and enl_map is None:
        enl = estimate_enl_global_linear(x, win=win)

    # Compute in log domain
    x_log = to_log(x)

    # Create uniform kernel
    kernel = np.ones((win, win), dtype=np.float32) / (win * win)

    # Compute local mean in log domain
    local_mean = ndimage.convolve(x_log, kernel, mode='reflect')

    # Compute local variance in log domain
    local_mean_sq = ndimage.convolve(x_log**2, kernel, mode='reflect')
    local_var = np.maximum(local_mean_sq - local_mean**2, 1e-12)

    # Estimate noise variance from ENL
    if enl_map is not None:
        # Use local ENL map
        noise_var = 1.0 / (enl_map + 1e-12)
    else:
        # Use global ENL
        noise_var = 1.0 / (enl + 1e-12)

    # Compute Lee gain: G = max(0, (σ^2 - σ_n^2) / σ^2)
    gain = np.maximum(0.0, (local_var - noise_var) / (local_var + 1e-12))

    # Apply filter: y = μ + G * (x - μ)
    y_log = local_mean + gain * (x_log - local_mean)

    # Convert back to linear domain
    y = from_log(y_log)

    return y


def kuan_filter_log(
    x: np.ndarray,
    win: int = 5,
    enl: float = None,
    enl_map: np.ndarray = None
) -> np.ndarray:
    """
    Kuan filter in log-domain for speckle reduction.

    The Kuan filter is similar to Lee but uses a different gain formulation
    based on MMSE (Minimum Mean Square Error):

    G = 1 - σ_n^2 / σ^2

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        win: Window size for local statistics (default: 5)
        enl: Global ENL value (estimated if None)
        enl_map: Local ENL map (estimated if None)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    x = x.astype(np.float32)

    # Estimate ENL if not provided
    if enl is None and enl_map is None:
        enl = estimate_enl_global_linear(x, win=win)

    # Compute in log domain
    x_log = to_log(x)

    # Create uniform kernel
    kernel = np.ones((win, win), dtype=np.float32) / (win * win)

    # Compute local mean in log domain
    local_mean = ndimage.convolve(x_log, kernel, mode='reflect')

    # Compute local variance in log domain
    local_mean_sq = ndimage.convolve(x_log**2, kernel, mode='reflect')
    local_var = np.maximum(local_mean_sq - local_mean**2, 1e-12)

    # Estimate noise variance from ENL
    if enl_map is not None:
        noise_var = 1.0 / (enl_map + 1e-12)
    else:
        noise_var = 1.0 / (enl + 1e-12)

    # Compute Kuan gain: G = 1 - σ_n^2 / σ^2
    gain = np.maximum(0.0, 1.0 - noise_var / (local_var + 1e-12))

    # Apply filter: y = μ + G * (x - μ)
    y_log = local_mean + gain * (x_log - local_mean)

    # Convert back to linear domain
    y = from_log(y_log)

    return y


def frost_filter_log(
    x: np.ndarray,
    win: int = 5,
    enl: float = None,
    enl_map: np.ndarray = None,
    damping: float = None
) -> np.ndarray:
    """
    Frost filter in log-domain for speckle reduction.

    The Frost filter uses exponentially weighted averaging based on
    local coefficient of variation. The weights decrease exponentially
    with distance from the center pixel, modulated by local statistics.

    w(r) ∝ exp(-α * |r - r_center|)

    where α is the damping factor derived from local statistics.

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        win: Window size (default: 5)
        enl: Global ENL value (estimated if None)
        enl_map: Local ENL map (estimated if None)
        damping: Damping factor (auto-computed if None)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    x = x.astype(np.float32)

    # Estimate ENL if not provided
    if enl is None and enl_map is None:
        enl = estimate_enl_global_linear(x, win=win)

    # Compute in log domain
    x_log = to_log(x)

    h, w = x_log.shape
    half_win = win // 2

    # Pad image for boundary handling
    x_log_pad = np.pad(x_log, half_win, mode='reflect')

    # Create distance matrix from center
    y_coords, x_coords = np.ogrid[-half_win:half_win+1, -half_win:half_win+1]
    distances = np.sqrt(x_coords**2 + y_coords**2).astype(np.float32)

    # Output array
    y_log = np.zeros_like(x_log)

    # Create uniform kernel for local stats
    kernel = np.ones((win, win), dtype=np.float32) / (win * win)

    # Compute local coefficient of variation in log domain
    local_mean = ndimage.convolve(x_log, kernel, mode='reflect')
    local_mean_sq = ndimage.convolve(x_log**2, kernel, mode='reflect')
    local_var = np.maximum(local_mean_sq - local_mean**2, 1e-12)
    local_cv = np.sqrt(local_var) / (np.abs(local_mean) + 1e-6)

    # Compute damping factor from ENL if not provided
    if damping is None:
        if enl_map is not None:
            alpha = 1.0 / np.sqrt(enl_map + 1e-12)
        else:
            alpha = 1.0 / np.sqrt(enl + 1e-12)
    else:
        alpha = damping

    # Apply Frost filter using sliding window
    for i in range(h):
        for j in range(w):
            # Extract window
            window = x_log_pad[i:i+win, j:j+win]

            # Compute adaptive weights
            if isinstance(alpha, np.ndarray):
                alpha_local = alpha[i, j]
            else:
                alpha_local = alpha

            # Exponential weights based on distance and local CV
            weights = np.exp(-alpha_local * local_cv[i, j] * distances)
            weights = weights / (np.sum(weights) + 1e-12)

            # Weighted average
            y_log[i, j] = np.sum(weights * window)

    # Convert back to linear domain
    y = from_log(y_log)

    return y


def adaptive_wiener_log(
    x: np.ndarray,
    win: int = 5,
    enl: float = None
) -> np.ndarray:
    """
    Adaptive Wiener filter in log-domain.

    Similar to Lee filter but uses Wiener filtering framework.

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        win: Window size (default: 5)
        enl: ENL value (estimated if None)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    # Adaptive Wiener is essentially the same as Lee in log domain
    return lee_filter_log(x, win=win, enl=enl)
