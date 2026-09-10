"""
ENL (Equivalent Number of Looks) Estimation Utilities

This module provides functions to estimate ENL for speckle noise in OCT images.
ENL is a measure of the effective number of independent samples averaged together,
inversely related to speckle noise variance.

ENL ≈ μ^2 / σ^2 (local mean squared divided by local variance)
"""

import numpy as np
from scipy import ndimage
import torch


def estimate_enl_map_linear(x: np.ndarray, win: int = 7) -> np.ndarray:
    """
    Estimate local ENL map from linear-domain image.

    Args:
        x: Input image in linear domain, shape (H, W), values in [0, 1]
        win: Window size for local statistics (default: 7)

    Returns:
        ENL map of shape (H, W), values clamped to [1, 50]

    Formula:
        ENL(i,j) = μ_local(i,j)^2 / σ^2_local(i,j)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    # Ensure float32
    x = x.astype(np.float32)

    # Create uniform kernel for averaging
    kernel = np.ones((win, win), dtype=np.float32) / (win * win)

    # Compute local mean using convolution (mode='reflect' for boundary)
    local_mean = ndimage.convolve(x, kernel, mode='reflect')

    # Compute local variance: E[X^2] - E[X]^2
    local_mean_sq = ndimage.convolve(x**2, kernel, mode='reflect')
    local_var = np.maximum(local_mean_sq - local_mean**2, 1e-12)

    # ENL = μ^2 / σ^2
    enl_map = (local_mean**2) / (local_var + 1e-12)

    # Clamp to reasonable range [1, 50]
    enl_map = np.clip(enl_map, 1.0, 50.0)

    return enl_map.astype(np.float32)


def estimate_enl_global_linear(x: np.ndarray, win: int = 7) -> float:
    """
    Estimate global ENL as the mean of local ENL map.

    Args:
        x: Input image in linear domain, shape (H, W), values in [0, 1]
        win: Window size for local statistics (default: 7)

    Returns:
        Scalar ENL value
    """
    enl_map = estimate_enl_map_linear(x, win=win)
    return float(np.mean(enl_map))


def estimate_enl_from_homogeneous_region(x: np.ndarray, roi: tuple = None) -> float:
    """
    Estimate ENL from a homogeneous region (if known).

    Args:
        x: Input image in linear domain, shape (H, W)
        roi: Region of interest as (y_start, y_end, x_start, x_end)
             If None, uses central 25% of image

    Returns:
        ENL estimated from the ROI
    """
    if roi is None:
        # Use central quarter
        h, w = x.shape
        y_start, y_end = h // 4, 3 * h // 4
        x_start, x_end = w // 4, 3 * w // 4
    else:
        y_start, y_end, x_start, x_end = roi

    region = x[y_start:y_end, x_start:x_end]

    mean = np.mean(region)
    var = np.var(region)

    enl = (mean**2) / (var + 1e-12)
    return float(np.clip(enl, 1.0, 50.0))


def to_log(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Convert linear-domain image to log-domain.

    Args:
        x: Input image in [0, 1]
        epsilon: Small value to avoid log(0)

    Returns:
        Log-domain image
    """
    x_clipped = np.clip(x, epsilon, 1.0)
    return np.log(x_clipped).astype(np.float32)


def from_log(x_log: np.ndarray) -> np.ndarray:
    """
    Convert log-domain image back to linear domain.

    Args:
        x_log: Log-domain image

    Returns:
        Linear-domain image clipped to [0, 1]
    """
    x_linear = np.exp(x_log)
    return np.clip(x_linear, 0.0, 1.0).astype(np.float32)


def normalize_log(x_log: np.ndarray) -> tuple:
    """
    Normalize log-domain image to [0, 1] and return normalization params.

    Args:
        x_log: Log-domain image

    Returns:
        Tuple of (normalized_image, min_val, max_val) for denormalization
    """
    min_val = np.min(x_log)
    max_val = np.max(x_log)

    if max_val - min_val < 1e-12:
        # Constant image
        return x_log.copy(), min_val, max_val

    x_norm = (x_log - min_val) / (max_val - min_val)
    return x_norm.astype(np.float32), float(min_val), float(max_val)


def denormalize_log(x_norm: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Denormalize log-domain image from [0, 1] back to original scale.

    Args:
        x_norm: Normalized image in [0, 1]
        min_val: Minimum value from normalization
        max_val: Maximum value from normalization

    Returns:
        Denormalized log-domain image
    """
    if max_val - min_val < 1e-12:
        return x_norm.copy()

    x_log = x_norm * (max_val - min_val) + min_val
    return x_log.astype(np.float32)


# Torch wrappers for compatibility
def torch_to_numpy(x: torch.Tensor, select_center: bool = True) -> np.ndarray:
    """
    Convert torch tensor to numpy array, optionally selecting center channel.

    Args:
        x: Torch tensor of shape (C, H, W) or (H, W)
        select_center: If True and C > 1, select middle channel

    Returns:
        Numpy array of shape (H, W)
    """
    x_np = x.detach().cpu().numpy()

    if x_np.ndim == 3:
        if select_center and x_np.shape[0] > 1:
            # Select center channel
            center_idx = x_np.shape[0] // 2
            x_np = x_np[center_idx]
        elif x_np.shape[0] == 1:
            x_np = x_np[0]

    return x_np.astype(np.float32)


def numpy_to_torch(x: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy array to torch tensor.

    Args:
        x: Numpy array of shape (H, W)
        device: Torch device

    Returns:
        Torch tensor of shape (1, H, W)
    """
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    return x_tensor


def estimate_enl_torch(x: torch.Tensor, win: int = 7, global_enl: bool = True) -> float | torch.Tensor:
    """
    Torch wrapper for ENL estimation.

    Args:
        x: Torch tensor (C, H, W) or (H, W)
        win: Window size
        global_enl: If True, return scalar; else return map

    Returns:
        ENL value (float) or ENL map (torch.Tensor)
    """
    x_np = torch_to_numpy(x, select_center=True)

    if global_enl:
        return estimate_enl_global_linear(x_np, win=win)
    else:
        enl_map = estimate_enl_map_linear(x_np, win=win)
        return numpy_to_torch(enl_map, device=x.device)
