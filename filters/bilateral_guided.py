"""
Bilateral and Guided Filters (Log-Domain)

Bilateral filtering: Edge-preserving smoothing based on spatial and range similarity.
Guided filtering: Fast edge-preserving filter using a guidance image.

Both filters are adapted for log-domain processing to handle multiplicative speckle noise.
"""

import numpy as np
from scipy import ndimage
try:
    from skimage.restoration import denoise_bilateral
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from .enl_utils import estimate_enl_global_linear, to_log, from_log, normalize_log, denormalize_log


def bilateral_filter_log(
    x: np.ndarray,
    spatial_sigma: float = None,
    range_sigma: float = None,
    win: int = 5,
    enl: float = None
) -> np.ndarray:
    """
    Bilateral filter in log-domain for speckle reduction.

    The bilateral filter performs edge-preserving smoothing by weighting
    pixels based on both spatial distance and intensity similarity.

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        spatial_sigma: Spatial gaussian sigma (default: win//2 / 2)
        range_sigma: Range gaussian sigma (default: auto from ENL)
        win: Window size (default: 5)
        enl: ENL value for auto-tuning range_sigma (estimated if None)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for bilateral filter. Install with: pip install scikit-image")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    x = x.astype(np.float32)

    # Auto-tune parameters from ENL if not provided
    if enl is None:
        enl = estimate_enl_global_linear(x, win=7)

    if spatial_sigma is None:
        spatial_sigma = (win // 2) / 2.0

    if range_sigma is None:
        # Derive from ENL: higher ENL = less noise = smaller range sigma
        # Empirical formula: range_sigma ≈ k / sqrt(ENL)
        k = 0.15
        range_sigma = k / np.sqrt(enl + 1e-12)

    # Convert to log domain
    x_log = to_log(x)

    # Normalize to [0, 1] for bilateral filter
    x_log_norm, min_val, max_val = normalize_log(x_log)

    # Apply bilateral filter in normalized log domain
    y_log_norm = denoise_bilateral(
        x_log_norm,
        sigma_color=range_sigma,
        sigma_spatial=spatial_sigma,
        channel_axis=None,
        mode='reflect'
    )

    # Denormalize
    y_log = denormalize_log(y_log_norm, min_val, max_val)

    # Convert back to linear domain
    y = from_log(y_log)

    return y


def guided_filter_log(
    x: np.ndarray,
    guide: np.ndarray = None,
    radius: int = 4,
    eps: float = 1e-3
) -> np.ndarray:
    """
    Guided filter in log-domain for speckle reduction.

    The guided filter is a fast edge-preserving filter that uses a guidance image
    to control the filtering. It's faster than bilateral filtering and produces
    less gradient reversal artifacts.

    Reference:
    He, K., Sun, J., & Tang, X. (2013). "Guided image filtering."
    IEEE TPAMI, 35(6), 1397-1409.

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        guide: Guidance image (default: use input image), shape (H, W)
        radius: Box filter radius (default: 4)
        eps: Regularization parameter (default: 1e-3)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    x = x.astype(np.float32)

    # Convert to log domain
    x_log = to_log(x)

    # Use input as guide if not provided
    if guide is None:
        guide_log = x_log
    else:
        guide_log = to_log(guide.astype(np.float32))

    # Apply guided filter in log domain
    y_log = _guided_filter_impl(guide_log, x_log, radius, eps)

    # Convert back to linear domain
    y = from_log(y_log)

    return y


def _guided_filter_impl(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    Core implementation of guided filter using box filters (fast via integral images).

    Args:
        I: Guidance image (H, W)
        p: Input image to be filtered (H, W)
        r: Box filter radius
        eps: Regularization

    Returns:
        Filtered image (H, W)
    """
    # Box filter function using scipy's uniform filter
    def box_filter(img, radius):
        # Window size for uniform filter
        size = 2 * radius + 1
        return ndimage.uniform_filter(img, size=size, mode='reflect')

    # Compute mean of I, p, and products
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)
    mean_II = box_filter(I * I, r)

    # Covariance and variance
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Smooth coefficients
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    # Output
    q = mean_a * I + mean_b

    return q.astype(np.float32)


def weighted_guided_filter_log(
    x: np.ndarray,
    guide: np.ndarray = None,
    radius: int = 4,
    eps: float = 1e-3,
    lambda_: float = 1e-4
) -> np.ndarray:
    """
    Weighted guided filter for better edge preservation.

    Adds spatial weighting to the guided filter for improved performance
    on images with strong edges.

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        guide: Guidance image (default: use input), shape (H, W)
        radius: Box filter radius (default: 4)
        eps: Regularization (default: 1e-3)
        lambda_: Edge awareness parameter (default: 1e-4)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    x = x.astype(np.float32)

    # Convert to log domain
    x_log = to_log(x)

    if guide is None:
        guide_log = x_log
    else:
        guide_log = to_log(guide.astype(np.float32))

    # Compute gradient magnitude for edge weighting
    grad_y, grad_x = np.gradient(guide_log)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Edge-aware weights (higher at edges)
    weights = 1.0 / (1.0 + grad_mag / (lambda_ + 1e-12))

    # Weighted guided filter
    y_log = _weighted_guided_filter_impl(guide_log, x_log, weights, radius, eps)

    # Convert back to linear domain
    y = from_log(y_log)

    return y


def _weighted_guided_filter_impl(
    I: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    r: int,
    eps: float
) -> np.ndarray:
    """
    Weighted guided filter implementation.

    Args:
        I: Guidance image (H, W)
        p: Input image (H, W)
        w: Weight map (H, W)
        r: Radius
        eps: Regularization

    Returns:
        Filtered image (H, W)
    """
    def box_filter(img, radius):
        size = 2 * radius + 1
        return ndimage.uniform_filter(img, size=size, mode='reflect')

    # Weighted means
    mean_I = box_filter(w * I, r) / (box_filter(w, r) + 1e-12)
    mean_p = box_filter(w * p, r) / (box_filter(w, r) + 1e-12)
    mean_Ip = box_filter(w * I * p, r) / (box_filter(w, r) + 1e-12)
    mean_II = box_filter(w * I * I, r) / (box_filter(w, r) + 1e-12)

    # Weighted covariance and variance
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Smooth coefficients
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    # Output
    q = mean_a * I + mean_b

    return q.astype(np.float32)


def fast_bilateral_log(
    x: np.ndarray,
    sigma_spatial: float = 2.0,
    sigma_range: float = 0.1,
    grid_size: int = 8
) -> np.ndarray:
    """
    Fast approximate bilateral filter using downsampling.

    For large images, this provides a faster alternative to full bilateral filtering.

    Args:
        x: Input image in linear domain [0, 1], shape (H, W)
        sigma_spatial: Spatial sigma
        sigma_range: Range sigma
        grid_size: Downsampling factor for speed (default: 8)

    Returns:
        Filtered image in linear domain [0, 1], shape (H, W)
    """
    # For now, fallback to regular bilateral
    # A full fast bilateral implementation would use bilateral grid
    return bilateral_filter_log(x, spatial_sigma=sigma_spatial, range_sigma=sigma_range)
