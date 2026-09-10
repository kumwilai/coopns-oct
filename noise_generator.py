"""
Realistic OCT noise generator for NSND training.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

DUKE_PRESET = {
    "alpha_vec": [2.75, 1.0, 0.75, 0.5],
    "speckle_k": 3.0,
    "speckle_base_k": 3.0,
    "speckle_snr_factor": 0.4,
    "speckle_correlation": 1.2,
    "banding_frequency": (0.02, 0.08),
    "banding_amplitude": (0.05, 0.15),
    "gaussian_sigma": (0.02, 0.08),
    "shot_gain": (50.0, 200.0),
}

PKU37_PRESET = {
    "alpha_vec": [3.0, 0.5, 1.0, 0.5],
    "speckle_k": 3.5,
    "speckle_base_k": 3.5,
    "speckle_snr_factor": 0.45,
    "speckle_correlation": 1.4,
    "banding_frequency": (0.02, 0.06),
    "banding_amplitude": (0.04, 0.12),
    "gaussian_sigma": (0.02, 0.07),
    "shot_gain": (60.0, 180.0),
}

HIGH_SPECKLE = {
    "alpha_vec": [5.0, 0.5, 0.5, 0.2],
    "speckle_k": 2.5,
    "speckle_base_k": 2.5,
    "speckle_snr_factor": 0.55,
    "speckle_correlation": 1.6,
    "banding_frequency": (0.02, 0.06),
    "banding_amplitude": (0.04, 0.10),
    "gaussian_sigma": (0.02, 0.06),
    "shot_gain": (60.0, 150.0),
}

DEFAULT_ALPHA = np.array(DUKE_PRESET["alpha_vec"], dtype=np.float32)


def _normalize_clean_np(clean: np.ndarray) -> np.ndarray:
    clean = clean.astype(np.float32)
    clean = np.clip(clean, 0.0, 1.0)
    return clean


def _gaussian_kernel1d_np(sigma: float) -> np.ndarray:
    radius = int(max(1, math.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel


def _gaussian_blur_np(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=sigma, mode="reflect")
    except Exception:
        kernel = _gaussian_kernel1d_np(sigma)
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, img)
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, tmp)


def _resolve_param(value: Any, low: float, high: float, rng: np.random.RandomState | np.random.Generator) -> float:
    if value is None:
        return float(rng.uniform(low, high))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(rng.uniform(value[0], value[1]))
    return float(value)


def _sample_weights_np(params: Optional[Dict[str, Any]], rng: np.random.RandomState | np.random.Generator) -> np.ndarray:
    if params and "weights" in params and params["weights"] is not None:
        weights = np.array(params["weights"], dtype=np.float32)
    else:
        alpha = DEFAULT_ALPHA
        if params and "alpha_vec" in params and params["alpha_vec"] is not None:
            alpha = np.array(params["alpha_vec"], dtype=np.float32)
        weights = rng.dirichlet(alpha)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / np.sum(weights)
    return weights.astype(np.float32)


def _generate_speckle_np(
    clean: np.ndarray,
    base_k: float,
    snr_factor: float,
    correlation_sigma: float,
    rng: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    eps = 1e-6
    clean_max = max(float(clean.max()), eps)
    local_k = base_k * (1.0 + snr_factor * (clean / clean_max))
    local_k = np.clip(local_k, 1e-3, None)
    speckle = rng.gamma(shape=local_k, scale=1.0 / local_k)
    speckle = _gaussian_blur_np(speckle, correlation_sigma)
    speckle = speckle / (float(speckle.mean()) + eps)
    return speckle.astype(np.float32)


def generate_oct_noise(
    clean: np.ndarray | torch.Tensor,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, Dict[str, Any]]:
    """
    Generate realistic OCT noise (numpy version).
    """
    if torch.is_tensor(clean):
        return generate_oct_noise_torch(clean, params)

    rng = np.random.default_rng()
    clean_np = np.asarray(clean)
    if clean_np.ndim == 3 and clean_np.shape[0] == 1:
        clean_np = clean_np[0]
    if clean_np.ndim == 3 and clean_np.shape[-1] == 1:
        clean_np = clean_np[..., 0]
    clean_np = _normalize_clean_np(clean_np)

    weights = _sample_weights_np(params, rng)

    base_k = _resolve_param(
        params.get("speckle_base_k") if params else None,
        2.0,
        5.0,
        rng,
    )
    if params and params.get("speckle_k") is not None:
        base_k = _resolve_param(params.get("speckle_k"), 2.0, 5.0, rng)
    snr_factor = _resolve_param(
        params.get("speckle_snr_factor") if params else None,
        0.2,
        0.6,
        rng,
    )
    correlation_sigma = _resolve_param(
        params.get("speckle_correlation") if params else None,
        0.8,
        2.0,
        rng,
    )

    banding_freq_val = None
    banding_amp_val = None
    if params:
        banding_freq_val = params.get("banding_frequency")
        if banding_freq_val is None:
            banding_freq_val = params.get("banding_freq")
        banding_amp_val = params.get("banding_amplitude")
        if banding_amp_val is None:
            banding_amp_val = params.get("banding_amp")
    banding_freq = _resolve_param(banding_freq_val, 0.02, 0.08, rng)
    banding_amp = _resolve_param(banding_amp_val, 0.05, 0.15, rng)
    banding_phase = float(rng.uniform(0.0, 2.0 * math.pi))

    gaussian_sigma = _resolve_param(
        params.get("gaussian_sigma") if params else None,
        0.02,
        0.08,
        rng,
    )

    shot_gain_val = None
    if params:
        shot_gain_val = params.get("shot_gain")
        if shot_gain_val is None:
            shot_gain_val = params.get("shot_peak")
    shot_gain = _resolve_param(shot_gain_val, 50.0, 200.0, rng)

    noisy = clean_np.copy()

    speckle = _generate_speckle_np(clean_np, base_k, snr_factor, correlation_sigma, rng)
    noisy = noisy * (1.0 + weights[0] * (speckle - 1.0))

    h, w = clean_np.shape
    y = np.arange(h, dtype=np.float32).reshape(h, 1)
    banding = banding_amp * np.sin(2.0 * math.pi * banding_freq * y + banding_phase)
    noisy = noisy + weights[1] * banding

    gaussian = rng.standard_normal(size=clean_np.shape).astype(np.float32) * gaussian_sigma
    noisy = noisy + weights[2] * gaussian

    shot = rng.poisson(clean_np * shot_gain) / shot_gain
    noisy = noisy + weights[3] * (shot - clean_np)

    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)

    noise_params = {
        "alpha_vec": (params or {}).get("alpha_vec", DEFAULT_ALPHA.tolist()),
        "speckle_base_k": base_k,
        "speckle_snr_factor": snr_factor,
        "speckle_correlation": correlation_sigma,
        "banding_frequency": banding_freq,
        "banding_amplitude": banding_amp,
        "banding_phase": banding_phase,
        "gaussian_sigma": gaussian_sigma,
        "shot_gain": shot_gain,
    }

    return noisy, weights, noise_params


def _gaussian_kernel2d_torch(sigma: float, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    if sigma <= 0:
        return torch.tensor([], device=device, dtype=dtype), 0
    radius = int(max(1, math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d, radius


def _gaussian_blur_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    kernel_2d, radius = _gaussian_kernel2d_torch(sigma, x.device, x.dtype)
    if radius <= 0:
        return x
    kernel = kernel_2d.view(1, 1, kernel_2d.size(0), kernel_2d.size(1))
    return F.conv2d(x, kernel, padding=radius)


def _sample_param_torch(
    value: Any,
    low: float,
    high: float,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return low + (high - low) * torch.rand(batch, device=device, dtype=dtype)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]) + (float(value[1]) - float(value[0])) * torch.rand(
            batch, device=device, dtype=dtype
        )
    return torch.full((batch,), float(value), device=device, dtype=dtype)


def _sample_weights_torch(
    params: Optional[Dict[str, Any]],
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if params and params.get("weights") is not None:
        weights = torch.tensor(params["weights"], device=device, dtype=dtype)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).repeat(batch, 1)
    else:
        alpha = DEFAULT_ALPHA
        if params and params.get("alpha_vec") is not None:
            alpha = np.array(params["alpha_vec"], dtype=np.float32)
        alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
        weights = torch.distributions.Dirichlet(alpha_t).sample((batch,))
    weights = weights.clamp_min(1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights


def generate_oct_noise_torch(
    clean: torch.Tensor,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any] | List[Dict[str, Any]]]:
    """
    Generate realistic OCT noise (torch version, GPU-friendly).
    """
    if clean.ndim == 2:
        clean_t = clean.unsqueeze(0).unsqueeze(0)
    elif clean.ndim == 3:
        if clean.shape[0] == 1:
            clean_t = clean.unsqueeze(0)
        else:
            clean_t = clean.unsqueeze(1)
    else:
        clean_t = clean

    clean_t = clean_t.float().clamp(0.0, 1.0)
    device = clean_t.device
    dtype = clean_t.dtype
    batch, _, h, w = clean_t.shape

    weights = _sample_weights_torch(params, batch, device, dtype)

    base_k = _sample_param_torch(
        (params or {}).get("speckle_base_k"), 2.0, 5.0, batch, device, dtype
    )
    if params and params.get("speckle_k") is not None:
        base_k = _sample_param_torch(params.get("speckle_k"), 2.0, 5.0, batch, device, dtype)
    snr_factor = _sample_param_torch(
        (params or {}).get("speckle_snr_factor"), 0.2, 0.6, batch, device, dtype
    )
    correlation_sigma = _sample_param_torch(
        (params or {}).get("speckle_correlation"), 0.8, 2.0, batch, device, dtype
    )
    banding_freq_val = (params or {}).get("banding_frequency")
    if banding_freq_val is None:
        banding_freq_val = (params or {}).get("banding_freq")
    banding_freq = _sample_param_torch(
        banding_freq_val,
        0.02,
        0.08,
        batch,
        device,
        dtype,
    )
    banding_amp_val = (params or {}).get("banding_amplitude")
    if banding_amp_val is None:
        banding_amp_val = (params or {}).get("banding_amp")
    banding_amp = _sample_param_torch(
        banding_amp_val,
        0.05,
        0.15,
        batch,
        device,
        dtype,
    )
    gaussian_sigma = _sample_param_torch(
        (params or {}).get("gaussian_sigma"), 0.02, 0.08, batch, device, dtype
    )
    shot_gain_val = (params or {}).get("shot_gain")
    if shot_gain_val is None:
        shot_gain_val = (params or {}).get("shot_peak")
    shot_gain = _sample_param_torch(
        shot_gain_val,
        50.0,
        200.0,
        batch,
        device,
        dtype,
    )

    eps = 1e-6
    clean_max = clean_t.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    local_k = base_k.view(batch, 1, 1, 1) * (1.0 + snr_factor.view(batch, 1, 1, 1) * clean_t / clean_max)
    local_k = local_k.clamp_min(1e-3)
    speckle = torch.distributions.Gamma(concentration=local_k, rate=local_k).sample()

    if batch > 1 and not torch.allclose(correlation_sigma, correlation_sigma[0]):
        speckle_list = []
        for idx in range(batch):
            speckle_blur = _gaussian_blur_torch(
                speckle[idx:idx + 1], float(correlation_sigma[idx].item())
            )
            speckle_blur = speckle_blur / (speckle_blur.mean(dim=(2, 3), keepdim=True) + eps)
            speckle_list.append(speckle_blur)
        speckle = torch.cat(speckle_list, dim=0)
    else:
        sigma_val = float(correlation_sigma[0].item())
        speckle = _gaussian_blur_torch(speckle, sigma_val)
        speckle = speckle / (speckle.mean(dim=(2, 3), keepdim=True) + eps)

    noisy = clean_t * (1.0 + weights[:, 0].view(batch, 1, 1, 1) * (speckle - 1.0))

    y = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    phase = 2.0 * math.pi * torch.rand(batch, device=device, dtype=dtype).view(batch, 1, 1)
    banding = banding_amp.view(batch, 1, 1) * torch.sin(
        2.0 * math.pi * banding_freq.view(batch, 1, 1) * y + phase
    )
    banding = banding.unsqueeze(1).expand(batch, 1, h, w)
    noisy = noisy + weights[:, 1].view(batch, 1, 1, 1) * banding

    gaussian = torch.randn_like(clean_t) * gaussian_sigma.view(batch, 1, 1, 1)
    noisy = noisy + weights[:, 2].view(batch, 1, 1, 1) * gaussian

    shot = torch.poisson(clean_t * shot_gain.view(batch, 1, 1, 1)) / shot_gain.view(batch, 1, 1, 1)
    noisy = noisy + weights[:, 3].view(batch, 1, 1, 1) * (shot - clean_t)

    noisy = noisy.clamp(0.0, 1.0)

    params_list = []
    for idx in range(batch):
        params_list.append(
            {
                "alpha_vec": (params or {}).get("alpha_vec", DEFAULT_ALPHA.tolist()),
                "speckle_base_k": float(base_k[idx].item()),
                "speckle_snr_factor": float(snr_factor[idx].item()),
                "speckle_correlation": float(correlation_sigma[idx].item()),
                "banding_frequency": float(banding_freq[idx].item()),
                "banding_amplitude": float(banding_amp[idx].item()),
                "banding_phase": float(phase[idx].item()),
                "gaussian_sigma": float(gaussian_sigma[idx].item()),
                "shot_gain": float(shot_gain[idx].item()),
            }
        )

    if clean.ndim == 2:
        return noisy[0, 0], weights[0], params_list[0]
    if clean.ndim == 3 and clean.shape[0] == 1:
        return noisy[0], weights[0], params_list[0]
    return noisy, weights, params_list


class NoisyOCTDataset(Dataset):
    """
    Dataset that generates noisy OCT pairs on-the-fly.
    """

    def __init__(
        self,
        clean_dir: str | Path,
        preset: Optional[Dict[str, Any]] = None,
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        return_torch: bool = True,
    ):
        self.clean_dir = Path(clean_dir)
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean dir not found: {self.clean_dir}")
        self.transform = transform
        self.return_torch = bool(return_torch)

        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        paths: List[Path] = []
        for ext in exts:
            paths.extend(self.clean_dir.glob(ext))
        self.paths = sorted(paths)
        if max_samples is not None:
            self.paths = self.paths[: int(max_samples)]
        if not self.paths:
            raise ValueError(f"No images found in {self.clean_dir}")

        self.params = dict(preset) if preset else {}

    def __len__(self) -> int:
        return len(self.paths)

    def _load_clean(self, path: Path) -> np.ndarray | torch.Tensor:
        img = Image.open(path).convert("L")
        if self.transform is not None:
            transformed = self.transform(img)
            if torch.is_tensor(transformed):
                return transformed.float()
            if isinstance(transformed, np.ndarray):
                return transformed.astype(np.float32)
            img = transformed
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def __getitem__(self, idx: int):
        clean = self._load_clean(self.paths[idx])
        noisy, weights, noise_params = generate_oct_noise(clean, self.params)

        if torch.is_tensor(clean):
            clean_t = clean
            if clean_t.ndim == 2:
                clean_t = clean_t.unsqueeze(0)
            if clean_t.ndim == 3 and clean_t.shape[0] != 1:
                clean_t = clean_t.unsqueeze(0)
            noisy_t = noisy
            if noisy_t.ndim == 2:
                noisy_t = noisy_t.unsqueeze(0)
            weights_t = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=clean_t.dtype)
            return noisy_t, clean_t, weights_t, noise_params

        clean_np = clean
        if self.return_torch:
            clean_t = torch.from_numpy(clean_np).unsqueeze(0).float()
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
            weights_t = torch.from_numpy(weights).float()
            return noisy_t, clean_t, weights_t, noise_params
        return noisy, clean_np, weights, noise_params


def extract_noise_stats(dataset: Dataset, max_samples: int = 1000) -> Dict[str, Any]:
    """
    Summarize weights and parameter statistics for a dataset.
    """
    weights_list: List[np.ndarray] = []
    params_accum: Dict[str, List[float]] = {}

    for idx in range(min(len(dataset), max_samples)):
        sample = dataset[idx]
        if len(sample) >= 3:
            weights = sample[2]
            params = sample[3] if len(sample) > 3 else {}
        else:
            continue

        if torch.is_tensor(weights):
            weights = weights.detach().cpu().numpy()
        weights_list.append(np.array(weights, dtype=np.float32))

        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (float, int)):
                    params_accum.setdefault(key, []).append(float(value))

    weights_arr = np.stack(weights_list, axis=0) if weights_list else np.zeros((0, 4), dtype=np.float32)
    stats = {
        "num_samples": len(weights_list),
        "weights_mean": weights_arr.mean(axis=0).tolist() if weights_list else None,
        "weights_std": weights_arr.std(axis=0).tolist() if weights_list else None,
    }
    param_stats = {}
    for key, values in params_accum.items():
        arr = np.array(values, dtype=np.float32)
        param_stats[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
    stats["params"] = param_stats
    return stats


__all__ = [
    "generate_oct_noise",
    "generate_oct_noise_torch",
    "NoisyOCTDataset",
    "DUKE_PRESET",
    "PKU37_PRESET",
    "HIGH_SPECKLE",
    "extract_noise_stats",
]
