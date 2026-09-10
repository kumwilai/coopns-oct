"""
Evaluation metrics for image denoising

Includes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- ENL (Equivalent Number of Looks) for speckle
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio

    Args:
        pred: Predicted image [B, 1, H, W]
        target: Ground truth image [B, 1, H, W]
        max_val: Maximum pixel value (default: 1.0)

    Returns:
        psnr: PSNR in dB
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return float('inf')

    max_val_t = mse.new_tensor(max_val)
    psnr = 20 * torch.log10(max_val_t / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index (simplified version)

    Args:
        pred: Predicted image [B, 1, H, W]
        target: Ground truth image [B, 1, H, W]
        window_size: Size of Gaussian window
        max_val: Maximum pixel value

    Returns:
        ssim: SSIM score in [0, 1]
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)

    # Convert to numpy for simplicity (detach to handle grad tensors)
    pred_np = pred[0, 0].detach().cpu().numpy()
    target_np = target[0, 0].detach().cpu().numpy()

    # Compute local statistics using sliding window
    from scipy.ndimage import uniform_filter

    mu1 = uniform_filter(pred_np, size=window_size)
    mu2 = uniform_filter(target_np, size=window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(pred_np ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(target_np ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(pred_np * target_np, size=window_size) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    ssim_score = ssim_map.mean()

    return float(ssim_score)


def compute_enl(image: torch.Tensor, roi_size: int = 50) -> float:
    """
    Compute Equivalent Number of Looks (ENL) for speckle quantification

    Higher ENL = less speckle noise

    ENL = (mean / std)^2 in homogeneous region

    Args:
        image: Image [B, 1, H, W]
        roi_size: Size of region of interest

    Returns:
        enl: ENL score
    """
    img = image[0, 0].cpu().numpy()
    H, W = img.shape

    # Select central homogeneous region
    h_start = (H - roi_size) // 2
    w_start = (W - roi_size) // 2

    roi = img[h_start:h_start+roi_size, w_start:w_start+roi_size]

    mean = roi.mean()
    std = roi.std()

    enl = (mean / std) ** 2 if std > 0 else 0.0

    return float(enl)


def evaluate_denoising(
    pred: torch.Tensor,
    target: torch.Tensor,
    noisy: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of denoising performance

    Args:
        pred: Predicted clean image [B, 1, H, W]
        target: Ground truth clean image [B, 1, H, W]
        noisy: Optional noisy input [B, 1, H, W]

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {}

    # PSNR
    metrics['psnr'] = compute_psnr(pred, target)

    # SSIM
    try:
        metrics['ssim'] = compute_ssim(pred, target)
    except:
        metrics['ssim'] = 0.0  # Fallback if scipy not available

    # ENL
    metrics['enl'] = compute_enl(pred)

    # If noisy input provided, compute gains
    if noisy is not None:
        metrics['noisy_psnr'] = compute_psnr(noisy, target)
        metrics['psnr_gain'] = metrics['psnr'] - metrics['noisy_psnr']

        try:
            metrics['noisy_ssim'] = compute_ssim(noisy, target)
            metrics['ssim_gain'] = metrics['ssim'] - metrics['noisy_ssim']
        except:
            pass

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print evaluation metrics"""
    print("=" * 50)
    print("Denoising Evaluation Metrics")
    print("=" * 50)

    for key, value in metrics.items():
        if 'psnr' in key.lower():
            print(f"{key:20s}: {value:6.2f} dB")
        elif 'ssim' in key.lower():
            print(f"{key:20s}: {value:6.4f}")
        elif 'enl' in key.lower():
            print(f"{key:20s}: {value:6.2f}")
        else:
            print(f"{key:20s}: {value:6.4f}")

    print("=" * 50)
