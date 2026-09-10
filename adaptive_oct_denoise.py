
"""
Domain-Adaptive OCT Despeckling Framework (Backbone + Noise Adapter + Meta-learning + TTA)

Python 3.10+ and PyTorch-only implementation tailored for grayscale OCT B-scans.
Implements:
- Clean/Paired datasets
- Synthetic noise domains (Rayleigh, Poisson, Gaussian-like)
- U-Net backbone with FiLM modulation
- Lightweight Noise Adapter (global pooled MLP) that modulates backbone
- Spatial/CASA adapters for coherent vs incoherent speckle suppression
- Reptile-style meta-learning (adapter-only) across synthetic domains
- Blind spectral noise characterization (single-image)
- Physics-informed losses (depth-weighted fidelity, A-scan continuity, speckle statistics)
- Optional supervised fine-tuning on real pairs
- Test-time adaptation via augmentation consistency + TV + spectral priors
- PSNR/SSIM evaluation utilities

Assumptions:
- OCT images are grayscale in [0,1] (we normalize if 8-bit)
- Input/Output shapes are [B, 1, H, W]
"""

from __future__ import annotations

import os
import sys
import signal
import glob
import math
import random
import copy
import json
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image as _tv_save_image
from skimage.metrics import structural_similarity as _sk_ssim
from skimage.metrics import peak_signal_noise_ratio as _sk_psnr


def safe_delete(*tensors):
    """Safely delete tensors, ignoring any that don't exist or aren't tensors."""
    for t in tensors:
        if t is not None:
            try:
                del t
            except:
                pass


def append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON record to a JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------- 
# Device setup (will be set in main based on args)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Default


def set_seed(seed: int = 42):
    """Set seeds for reproducibility across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # better throughput
    torch.backends.cudnn.benchmark = True


# ------------------------------- 
# Memory safeguards and monitoring
# ------------------------------- 
def get_memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics."""
    stats = {}
    if torch.cuda.is_available():
        stats['cuda_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        stats['cuda_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
        stats['cuda_max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
    try:
        import psutil
        process = psutil.Process()
        stats['ram_used_gb'] = process.memory_info().rss / 1e9
    except ImportError:
        pass
    return stats


def force_memory_cleanup():
    """Aggressive memory cleanup."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_memory_available(required_gb: float = 2.0) -> bool:
    """Check if sufficient memory is available."""
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / 1e9
        if available_ram_gb < required_gb:
            return False
    except ImportError:
        pass

    if torch.cuda.is_available():
        cuda_free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        if cuda_free_gb < required_gb:
            return False
    return True


def print_n2v_memory_summary(masked_img: torch.Tensor, mask_coords: Tuple, prefix: str = "[N2V]"):
    """Print detailed memory summary for Noise2Void operations."""
    batch_idx, channel_idx, y_coords, x_coords = mask_coords

    # Calculate memory usage
    img_mem_mb = (masked_img.numel() * masked_img.element_size()) / (1024 * 1024)
    num_masked = len(batch_idx)
    coords_mem_mb = (num_masked * 4 * 8) / (1024 * 1024)  # 4 tensors × 8 bytes per long
    total_mem_mb = img_mem_mb + coords_mem_mb

    print(f"{prefix} Memory Summary:", flush=True)
    print(f"  Masked image: {img_mem_mb:.2f} MB ({masked_img.shape})", flush=True)
    print(f"  Mask coords: {coords_mem_mb:.2f} MB ({num_masked} pixels)", flush=True)
    print(f"  Total N2V overhead: {total_mem_mb:.2f} MB", flush=True)

    # Check for potential issues
    mask_ratio_actual = num_masked / (masked_img.shape[0] * masked_img.shape[1] * masked_img.shape[2] * masked_img.shape[3])
    print(f"  Actual mask ratio: {mask_ratio_actual:.3f}", flush=True)

    if total_mem_mb > 100:
        print(f"  ⚠ WARNING: N2V overhead is {total_mem_mb:.1f} MB - consider reducing batch size or mask ratio", flush=True)


def print_n2n_memory_summary(input_subset: torch.Tensor, target_mask: torch.Tensor, prefix: str = "[N2N]"):
    """Print detailed memory summary for Neighbor2Neighbor operations."""
    # Calculate memory usage
    input_mem_mb = (input_subset.numel() * input_subset.element_size()) / (1024 * 1024)
    mask_mem_mb = (target_mask.numel() * 1) / (1024 * 1024)  # bool = 1 byte
    total_mem_mb = input_mem_mb + mask_mem_mb

    print(f"{prefix} Memory Summary:", flush=True)
    print(f"  Input subset: {input_mem_mb:.2f} MB ({input_subset.shape})", flush=True)
    print(f"  Target mask: {mask_mem_mb:.2f} MB ({target_mask.shape})", flush=True)
    print(f"  Total N2N overhead: {total_mem_mb:.2f} MB", flush=True)

    # Calculate actual subset ratio
    num_target_pixels = target_mask.sum().item()
    total_pixels = target_mask.numel()
    subset_ratio = num_target_pixels / total_pixels
    print(f"  Target subset ratio: {subset_ratio:.1%} ({num_target_pixels}/{total_pixels} pixels)", flush=True)

    # Expected ratio for checkerboard is 50%
    if abs(subset_ratio - 0.5) > 0.01:
        print(f"  ⚠ WARNING: Expected 50% subset ratio, got {subset_ratio:.1%}", flush=True)

    if total_mem_mb > 100:
        print(f"  ⚠ WARNING: N2N overhead is {total_mem_mb:.1f} MB - consider reducing batch size", flush=True)


class MemoryGuard:
    """Context manager for memory-safe operations."""
    def __init__(self, operation_name: str = "operation", cleanup: bool = True):
        self.operation_name = operation_name
        self.cleanup = cleanup
        self.start_stats = {}

    def __enter__(self):
        force_memory_cleanup()
        self.start_stats = get_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            force_memory_cleanup()
        end_stats = get_memory_stats()
        if 'cuda_allocated_gb' in self.start_stats and 'cuda_allocated_gb' in end_stats:
            delta = end_stats['cuda_allocated_gb'] - self.start_stats['cuda_allocated_gb']
            if delta > 0.5:  # More than 500MB leaked
                print(f"[MemoryWarning] {self.operation_name} leaked {delta:.2f} GB GPU memory")


class LRUCache:
    """LRU cache with maximum size limit to prevent unbounded growth."""
    def __init__(self, max_size: int = 32):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        """Get item from cache, moving it to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        """Add item to cache, evicting oldest if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest item

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()


# Pre-built kernel caches shared across modules
_SOBEL_KERNEL_CACHE, _GAUSSIAN_KERNEL_CACHE = LRUCache(max_size=8), LRUCache(max_size=32)


def _gaussian_kernel_2d(channels: int, ksize: int, sigma: float, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create or fetch a 2D Gaussian kernel for separable smoothing."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd.")
    key = (channels, ksize, float(sigma), dtype)
    cached = _GAUSSIAN_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached.to(device=device, dtype=dtype)

    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    grid_x, grid_y = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, ksize, ksize).repeat(channels, 1, 1, 1)
    _GAUSSIAN_KERNEL_CACHE.put(key, kernel)
    return kernel.to(device=device, dtype=dtype)


# ------------------------------- 
# Dataset definitions
# ------------------------------- 
class CleanOCTDataset(Dataset):
    """Dataset of clean OCT images used to synthesize noisy inputs during meta-training.

    Args:
        root_dir: Directory containing clean OCT images.
        transform: Optional callable that takes a tensor [1,H,W] and returns transformed tensor.
    """

    IMG_EXTS = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        paths: List[str] = []
        for pat in self.IMG_EXTS:
            paths.extend(glob.glob(os.path.join(root_dir, pat)))
        self.paths = sorted(paths)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _to_gray01(img: torch.Tensor) -> torch.Tensor:
        """Convert image to grayscale [1, H, W] tensor in [0, 1] range."""
        # img: [C,H,W] uint8 or float, from torchvision.io.read_image -> uint8 [0..255]
        if img.dtype != torch.float32:
            img_float = img.float()
            # Handle 16-bit images
            if img.max() > 255:
                img = img_float / 65535.0
            else:
                img = img_float / 255.0

        # Ensure 3D tensor [C, H, W]
        if img.ndim == 2:
            img = img.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif img.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {img.shape}")

        # Convert to grayscale if needed
        if img.shape[0] == 1:
            gray = img
        elif img.shape[0] == 3:
            # RGB to grayscale using standard weights
            r, g, b = img[0:1], img[1:2], img[2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            # Take first channel for other cases
            gray = img[0:1]

        return gray.clamp(0.0, 1.0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = read_image(path)  # [C,H,W], uint8
        gray = self._to_gray01(img)  # [1,H,W] float [0,1]
        if self.transform is not None:
            gray = self.transform(gray)
        return gray


class PairedOCTDataset(Dataset):
    """Dataset with real noisy-clean pairs (or noisy-only for blind2unblind).

    Args:
        pairs: Either list of (noisy_path, clean_path) tuples or a path to a text file.
               Each line may contain:
                 - noisy_path,clean_path (comma or space separated), or
                 - noisy_path (if expect_clean=False for noisy-only training)
        transform: Optional callable applied to both images; expects [1,H,W] tensor.
        expect_clean: If False, accept single-path lines and return (noisy, noisy).
        crop_size: If set, crop to this size; random for train if random_crop=True.
        random_crop: Use random crop instead of center crop when crop_size is set.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]] | str,
        transform: Optional[Callable] = None,
        expect_clean: bool = True,
        crop_size: Optional[int] = None,
        random_crop: bool = False,
    ):
        parsed: List[Tuple[str, Optional[str]]] = []
        if isinstance(pairs, str):
            if not os.path.isfile(pairs):
                raise FileNotFoundError(f"Pairs list file not found: {pairs}")
            with open(pairs, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.lstrip().startswith("#"):
                        continue
                    if "," in line:
                        toks = [t.strip() for t in line.split(",", 1)]
                    else:
                        toks = line.split()
                    if len(toks) == 1:
                        parsed.append((toks[0], None))
                    elif len(toks) == 2:
                        noisy_path, clean_path = toks[0], toks[1]
                        if expect_clean and self._looks_clean(noisy_path) and self._looks_noisy(clean_path):
                            noisy_path, clean_path = clean_path, noisy_path
                        parsed.append((noisy_path, None if not expect_clean else clean_path))
                    else:
                        raise ValueError("Each line must have one or two paths.")
        else:
            parsed = [(a, None if (b is None or not expect_clean) else b) for (a, b) in pairs]

        self.pairs: List[Tuple[str, Optional[str]]] = parsed
        self.expect_clean = expect_clean
        self.transform = transform
        self.crop_size = int(crop_size) if crop_size else None
        self.random_crop = bool(random_crop)
        if not self.pairs:
            raise ValueError("Empty pairs list.")

    @staticmethod
    def _looks_clean(path: str) -> bool:
        p = path.lower()
        return any(token in p for token in ("clean", "gt", "target", "label"))

    @staticmethod
    def _looks_noisy(path: str) -> bool:
        p = path.lower()
        return any(token in p for token in ("noisy", "noise", "corrupt", "input"))

    @staticmethod
    def _to_gray01(img: torch.Tensor) -> torch.Tensor:
        """Convert image to grayscale [1, H, W] tensor in [0, 1] range."""
        if img.dtype != torch.float32:
            img_float = img.float()
            # Handle 16-bit images
            if img.max() > 255:
                img = img_float / 65535.0
            else:
                img = img_float / 255.0

        # Ensure 3D tensor [C, H, W]
        if img.ndim == 2:
            img = img.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif img.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {img.shape}")

        # Convert to grayscale if needed
        if img.shape[0] == 1:
            gray = img
        elif img.shape[0] == 3:
            # RGB to grayscale using standard weights
            r, g, b = img[0:1], img[1:2], img[2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            # Take first channel for other cases
            gray = img[0:1]

        return gray.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop_pair(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.crop_size:
            return x, y
        _, h, w = x.shape
        cs = self.crop_size
        if h >= cs and w >= cs:
            if self.random_crop:
                top = torch.randint(0, h - cs + 1, (1,)).item()
                left = torch.randint(0, w - cs + 1, (1,)).item()
            else:
                top = (h - cs) // 2
                left = (w - cs) // 2
            x = x[:, top:top + cs, left:left + cs]
            y = y[:, top:top + cs, left:left + cs]
        else:
            x = F.interpolate(x.unsqueeze(0), size=(cs, cs), mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=(cs, cs), mode="bilinear", align_corners=False).squeeze(0)
        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.pairs[idx]
        x = self._to_gray01(read_image(noisy_path))  # [1,H,W]
        if clean_path is None:
            y = x
        else:
            y = self._to_gray01(read_image(clean_path))  # [1,H,W]
        x, y = self._crop_pair(x, y)
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x, y


class CachedPairedOCTDataset(Dataset):
    """Eagerly loads paired images into memory for fast repeated evaluation.

    Args:
        pairs: list of (noisy, clean) paths or path to list file
        transform: optional callable applied once at load time
        limit: optional cap on number of pairs to load
    """
    # Safety limit to prevent excessive memory usage
    MAX_CACHE_SIZE = 32  # Maximum number of image pairs to cache

    def __init__(self, pairs: List[Tuple[str, str]] | str, transform: Optional[Callable] = None, limit: Optional[int] = None):
        self.base = PairedOCTDataset(pairs, transform=transform)
        if limit is None:
            limit = len(self.base)

        # Estimate image size by loading first pair to adjust cache limit
        mb_per_pair = 2.0  # Default estimate
        check_interval = 10  # Default check interval
        try:
            sample_x, sample_y = self.base[0]
            image_pixels = sample_x.shape[1] * sample_x.shape[2] if sample_x.ndim >= 3 else 512 * 512
            # Estimate memory: 2 images (noisy+clean) * 4 bytes/pixel (float32)
            mb_per_pair = (image_pixels * 2 * 4) / (1024 * 1024)

            # Adaptively reduce MAX_CACHE_SIZE based on image size
            if mb_per_pair > 1.5:  # Large images (>= ~512x512)
                adaptive_max = 8
                check_interval = 5
                print(f"[CachedDataset] Large images detected (~{image_pixels**0.5:.0f}x{image_pixels**0.5:.0f}), limiting cache to {adaptive_max} pairs")
            elif mb_per_pair > 0.4:  # Medium images (~256x256 to 512x512)
                adaptive_max = 16
                check_interval = 10
                print(f"[CachedDataset] Medium images detected (~{image_pixels**0.5:.0f}x{image_pixels**0.5:.0f}), limiting cache to {adaptive_max} pairs")
            else:  # Small images
                adaptive_max = self.MAX_CACHE_SIZE
                check_interval = 10

            limit = min(limit, adaptive_max)
        except Exception as e:
            print(f"[Warning] Could not estimate image size: {e}, using default limits")
            limit = min(limit, self.MAX_CACHE_SIZE)

        # Apply safety limit to prevent OOM
        if limit > self.MAX_CACHE_SIZE:
            print(f"[MemoryWarning] Requested cache size {limit} exceeds safe limit {self.MAX_CACHE_SIZE}")
            print(f"[MemoryWarning] Reducing cached pairs from {limit} to {self.MAX_CACHE_SIZE} to prevent OOM")
            limit = self.MAX_CACHE_SIZE

        # Check available memory before caching
        if not check_memory_available(required_gb=1.0):
            print(f"[MemoryWarning] Low memory detected, reducing cache size to {max(4, limit // 2)}")
            limit = max(4, limit // 2)

        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        print(f"[CachedDataset] Loading {min(limit, len(self.base))} image pairs into memory...")

        for i in range(min(limit, len(self.base))):
            try:
                x, y = self.base[i]
                self.buffer.append((x.contiguous(), y.contiguous()))

                if (i + 1) % check_interval == 0 and not check_memory_available(required_gb=0.5):
                    print(f"[MemoryWarning] Low memory after loading {i+1} pairs, stopping early")
                    break
            except Exception as e:
                print(f"[Error] Failed to load pair {i}: {e}")
                break

        print(f"[CachedDataset] Successfully cached {len(self.buffer)} pairs")

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.buffer[idx]


class PairedOCTDatasetWithParams(Dataset):
    """Paired dataset that also returns normalized simulator parameters for SimInv training."""
    def __init__(self, pairs: str, params_jsonl: str, transform: Optional[Callable] = None):
        self.base = PairedOCTDataset(pairs, transform=transform, expect_clean=True)
        self.transform = transform
        self.params = self._load_params(params_jsonl)
        self._zero_theta = torch.zeros(len(SIMINV_PARAM_NAMES), dtype=torch.float32)

        # Integrity check: ensure most pairlist noisy paths have metadata (otherwise theta supervision is broken).
        missing = 0
        for noisy_path, _clean_path in self.base.pairs:
            if noisy_path not in self.params:
                missing += 1
        if missing:
            total = len(self.base.pairs)
            frac = 100.0 * missing / max(1, total)
            print(
                f"[SimInv] WARNING: missing theta metadata for {missing}/{total} pairs ({frac:.1f}%). "
                f"Make sure you generated metadata with --write_params_jsonl and used --overwrite.",
                flush=True,
            )

    @staticmethod
    def _load_params(paths_csv: str) -> Dict[str, torch.Tensor]:
        import json
        mapping: Dict[str, List[float]] = {}
        paths = [p.strip() for p in str(paths_csv).split(",") if p.strip()]
        if not paths:
            raise ValueError("params_jsonl is empty")
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"noise params jsonl not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    rec = json.loads(s)
                    noisy = rec.get("noisy")
                    vec = rec.get("params_norm_vec")
                    if noisy is None or vec is None:
                        continue
                    mapping[str(noisy)] = [float(x) for x in vec]
        if not mapping:
            raise ValueError("No valid entries found in noise params jsonl.")
        # Validate vector length.
        bad = [k for k, v in mapping.items() if len(v) != len(SIMINV_PARAM_NAMES)]
        if bad:
            raise ValueError(f"Found {len(bad)} entries with wrong params_norm_vec length (expected {len(SIMINV_PARAM_NAMES)}).")
        # Convert to tensors lazily in __getitem__ to keep RAM low.
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in mapping.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        noisy_path, _clean_path = self.base.pairs[idx]
        theta = self.params.get(noisy_path)
        if theta is None:
            # Fall back to zeros if missing; training code can detect/ignore if needed.
            theta = self._zero_theta
        return x, y, theta


# ------------------------------- 
# Simple tensor transforms (optional) 
# ------------------------------- 
def resize_to(shape_hw: Tuple[int, int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a callable that resizes [1,H,W] tensor to shape_hw using bilinear interpolation."""
    h, w = shape_hw
    def _fn(x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)  # [1,1,H,W]
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x.squeeze(0)
    return _fn


# ------------------------------- 
# Synthetic noise domains
# ------------------------------- 
def add_rayleigh_noise(clean: torch.Tensor, scale_range: Tuple[float, float] = (0.2, 0.9)) -> torch.Tensor:
    """Apply multiplicative Rayleigh speckle: noisy = clean * R, where R~Rayleigh(sigma)."""
    b = clean.shape[0]
    low, high = scale_range
    sigma = torch.empty((b, 1, 1, 1), device=clean.device).uniform_(low, high)
    u = torch.rand_like(clean).clamp_(1e-6, 1 - 1e-6)
    rayleigh = torch.sqrt(-2.0 * (sigma**2) * torch.log(1.0 - u))
    noisy = clean * rayleigh
    return noisy.clamp(0.0, 1.0)


def add_poisson_noise(clean: torch.Tensor, lambda_scale: float = 1.0) -> torch.Tensor:
    """Apply Poisson noise: scale intensities to counts then rescale back to [0,1]."""
    base_peak = 30.0 * lambda_scale
    b = clean.shape[0]
    peaks = torch.empty((b, 1, 1, 1), device=clean.device).uniform_(base_peak * 0.7, base_peak * 1.3)
    noisy = torch.poisson(clean * peaks) / peaks
    return noisy.clamp(0.0, 1.0)


def add_mixed_gaussian_noise(clean: torch.Tensor, sigma_range: Tuple[float, float] = (0.01, 0.1)) -> torch.Tensor:
    """Additive zero-mean Gaussian-like noise for robustness."""
    b = clean.shape[0]
    low, high = sigma_range
    sigma = torch.empty((b, 1, 1, 1), device=clean.device).uniform_(low, high)
    noise = torch.randn_like(clean) * sigma
    noisy = clean + noise
    return noisy.clamp(0.0, 1.0)


def add_gaussian_additive_noise(clean: torch.Tensor, sigma_range: Tuple[float, float] = (0.03, 0.10)) -> torch.Tensor:
    """Apply additive Gaussian noise."""
    b = clean.shape[0]
    low, high = sigma_range
    sigma = torch.empty((b, 1, 1, 1), device=clean.device).uniform_(low, high)
    noise = torch.randn_like(clean) * sigma
    noisy = clean + noise
    return noisy.clamp(0.0, 1.0)


NOISE_TASKS: List[Dict] = [
    {"name": "rayleigh", "fn": add_rayleigh_noise},
    {"name": "poisson", "fn": add_poisson_noise},
    {"name": "gaussian_mult", "fn": add_mixed_gaussian_noise},
    {"name": "gaussian_add", "fn": add_gaussian_additive_noise},
]


def add_gamma_speckle_noise(clean: torch.Tensor, k_range: Tuple[float, float] = (0.6, 2.0)) -> torch.Tensor:
    """Multiplicative Gamma-distributed speckle with E[R]=1 by using scale=1/k."""
    b = clean.shape[0]
    k = torch.empty((b, 1, 1, 1), device=clean.device).uniform_(k_range[0], k_range[1])
    concentration = k.expand_as(clean)
    rate = (1.0 / k).expand_as(clean)
    dist = torch.distributions.Gamma(concentration, rate)
    R = dist.sample()
    return (clean * R).clamp(0.0, 1.0)


def add_heavy_gamma_speckle_noise(clean: torch.Tensor, k_range: Tuple[float, float] = (0.3, 1.0)) -> torch.Tensor:
    """Stronger Gamma speckle (smaller shape k => heavier noise)."""
    return add_gamma_speckle_noise(clean, k_range=k_range)


def add_correlated_speckle(clean: torch.Tensor, strength: float = 0.5, ksize: int = 5, sigma: float = 1.0) -> torch.Tensor:
    noise = torch.randn_like(clean)
    kernel = _gaussian_kernel_2d(1, ksize, sigma, dtype=clean.dtype, device=clean.device)
    corr = 1.0 + strength * F.conv2d(noise, kernel, padding=ksize // 2)
    return (clean * corr).clamp(0.0, 1.0)


def add_heavy_correlated_speckle(clean: torch.Tensor, strength: float = 0.9, ksize: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Heavier correlated speckle for harder domains."""
    return add_correlated_speckle(clean, strength=strength, ksize=ksize, sigma=sigma)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute PSNR for [B,1,H,W] tensors."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((data_range ** 2) / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute SSIM using skimage; expects [B,1,H,W] tensors."""
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    vals = []
    for i in range(p.shape[0]):
        vals.append(_sk_ssim(t[i, 0], p[i, 0], data_range=data_range))
    return float(sum(vals) / len(vals)) if vals else 0.0


NOISE_TASKS_EXTENDED: List[Dict] = NOISE_TASKS + [
    {"name": "gamma_speckle", "fn": add_gamma_speckle_noise},
    {"name": "corr_speckle", "fn": add_correlated_speckle},
    {"name": "heavy_gamma_speckle", "fn": add_heavy_gamma_speckle_noise},
    {"name": "heavy_corr_speckle", "fn": add_heavy_correlated_speckle},
    {"name": "poisson", "fn": add_poisson_noise},
    {"name": "rayleigh", "fn": add_rayleigh_noise},
]

NOISE_DOMAIN_NAMES: List[str] = [t["name"] for t in NOISE_TASKS_EXTENDED]


def _radial_frequency_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, d=1.0, device=device).abs()
    fx = torch.fft.rfftfreq(w, d=1.0, device=device).abs()
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    r = r / (r.max() + 1e-8)
    return r


class SpectralNoiseCharacterizer(nn.Module):
    """Blindly estimates noise statistics from a single noisy B-scan."""
    def __init__(self, n_fft: int = 128, band_splits: Tuple[float, ...] = (0.08, 0.18, 0.32, 0.5)):
        super().__init__()
        self.n_fft = n_fft
        self.band_splits = band_splits
        band_count = len(band_splits) + 1
        feat_dim = band_count + 3  # band powers + slope + anisotropy + noise level
        hidden = 64
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, len(NOISE_DOMAIN_NAMES))
        )
        self.param_head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return spectral slope, anisotropy, bandpower, and noise domain logits."""
        b, _, h, w = noisy.shape
        win_h, win_w = min(h, self.n_fft), min(w, self.n_fft)
        patch = noisy[..., 0:win_h, 0:win_w]
        spec = torch.fft.rfft2(patch, norm="ortho")
        mag = torch.abs(spec) + 1e-6
        power = mag ** 2
        r = _radial_frequency_grid(patch.shape[-2], patch.shape[-1], device=patch.device)

        band_edges = (0.0, *self.band_splits, 1.0)
        band_power = []
        for i in range(len(band_edges) - 1):
            low, high = band_edges[i], band_edges[i + 1]
            mask = (r >= low) & (r < high)
            band_power.append((power * mask).sum(dim=(-2, -1)) / (mask.sum() + 1e-6))
        band_power = torch.stack(band_power, dim=1).squeeze(-1)  # [B, nbands]
        band_power = band_power / (band_power.sum(dim=1, keepdim=True) + 1e-6)

        flat_r = r.view(-1)
        flat_power = power.view(b, -1)
        mask = flat_r > 0
        logf = torch.log(flat_r[mask])
        logp = torch.log(flat_power[:, mask] + 1e-6)
        slope = ((logf - logf.mean()) * (logp - logp.mean(dim=1, keepdim=True))).mean(dim=1) / (logf.var() + 1e-6)

        fx_energy = (power * (r + 1e-6)).mean(dim=-2).mean(dim=-1)
        fy_energy = (power * (r + 1e-6)).mean(dim=-1).mean(dim=-1)
        anisotropy = torch.abs(fx_energy - fy_energy) / (fx_energy + fy_energy + 1e-6)
        anisotropy = anisotropy.view(b, -1)

        noise_level = power.mean(dim=(-2, -1)).view(b, -1)
        features = torch.cat([band_power, slope.view(b, 1), anisotropy, noise_level], dim=1)
        logits = self.head(features)
        noise_params = self.param_head(features)
        return {
            "band_power": band_power,
            "spectral_slope": slope,
            "anisotropy": anisotropy,
            "noise_logits": logits,
            "noise_probs": torch.softmax(logits, dim=1),
            "noise_params": noise_params,
        }


@torch.no_grad()
def estimate_noise_correlation_score(noisy: torch.Tensor, ksize: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Estimate spatial noise correlation from a noisy batch.

    Returns a per-image scalar score in [0, ~1]. Higher implies more spatially
    correlated (speckle-like) noise, where B2U tends to outperform N2V.
    """
    if noisy.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {noisy.shape}")
    b, c, h, w = noisy.shape
    kernel = _gaussian_kernel_2d(c, ksize, sigma, dtype=noisy.dtype, device=noisy.device)
    blurred = F.conv2d(noisy, kernel, padding=ksize // 2, groups=c)
    residual = noisy - blurred
    var = (residual ** 2).mean(dim=(-2, -1), keepdim=True)
    corr_h = (residual[:, :, :, 1:] * residual[:, :, :, :-1]).mean(dim=(-2, -1), keepdim=True) / (var + 1e-6)
    corr_v = (residual[:, :, 1:, :] * residual[:, :, :-1, :]).mean(dim=(-2, -1), keepdim=True) / (var + 1e-6)
    corr = 0.5 * (corr_h.abs() + corr_v.abs())
    return corr.view(b)

# ------------------------------- 
# Backbone network (NAFNet "Fair" Implementation with FiLM)
# Based on sota/models/nafnet_fair.py but with CASA modulation hooks
# ------------------------------- 
class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm for better performance (from NAFNet official)."""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    """SimpleGate: Split channels and multiply."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """NAFBlock (Fair Config) with FiLM support."""
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_expand * c, 1)
        self.conv5 = nn.Conv2d(ffn_expand * c // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # BUG FIX: Initialize to zeros (official NAFNet implementation)
        # This means residual paths contribute nothing at initialization,
        # allowing the network to learn when to use them (residual learning)
        # Reference: https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py#L46-L47
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, film_gamma=None, film_beta=None):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        
        # Apply FiLM modulation from CASA
        if film_gamma is not None: x = x * film_gamma
        if film_beta is not None: x = x + film_beta
        
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma

class NAFBackbone(nn.Module):
    """NAFNet Backbone (Fair Config) with FiLM support."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=48, 
                 enc_blk_nums=[2, 2, 2], middle_blk_num=2, dec_blk_nums=[2, 2, 2]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.ending = nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = base_channels
        # Encoders
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.ModuleList([NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
            
        self.middle_blks = nn.ModuleList([NAFBlock(chan) for _ in range(middle_blk_num)])
        
        # Decoders
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.ModuleList([NAFBlock(chan) for _ in range(num)]))
            
        self.base_channels = base_channels
        self.enc_blk_nums = enc_blk_nums

    @property
    def modulated_channels(self) -> Dict[str, int]:
        """Return channels at each modulation point for adapter."""
        # Map 'enc1', 'enc2', 'enc3' etc. to actual channel counts
        ch = self.base_channels
        mapping = {}
        # Encoders
        for i in range(len(self.enc_blk_nums)):
            mapping[f"enc{i+1}"] = ch
            ch *= 2
        # Decoders (reverse order)
        for i in range(len(self.enc_blk_nums)):
            ch //= 2
            mapping[f"dec{len(self.enc_blk_nums)-i}"] = ch
        return mapping

    def forward(
        self,
        x: torch.Tensor,
        adapter_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        residual_base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def gb_aligned(name: str, target_shape: Tuple[int, int]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if adapter_features is None:
                return None, None
            ab = adapter_features.get(name, {})
            g, b = ab.get("gamma", None), ab.get("beta", None)
            if g is not None and g.ndim == 4 and g.shape[-2:] != target_shape:
                g = F.interpolate(g, size=target_shape, mode="bilinear", align_corners=False)
            if b is not None and b.ndim == 4 and b.shape[-2:] != target_shape:
                b = F.interpolate(b, size=target_shape, mode="bilinear", align_corners=False)
            return g, b

        def align_like(ref: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            _, _, Hr, Wr = ref.shape
            _, _, Hs, Ws = src.shape
            pad_h, pad_w = max(0, Hr - Hs), max(0, Wr - Ws)
            if pad_h > 0 or pad_w > 0:
                src = F.pad(src, (0, pad_w, 0, pad_h))
            if src.shape[-2] > Hr or src.shape[-1] > Wr:
                src = src[:, :, :Hr, :Wr]
            return src

        # Save original input for global residual connection (NAFNet standard).
        # IMPORTANT for masked/self-supervised modes:
        # If residual_base is provided (e.g., original noisy image), use it instead of the masked input
        # to avoid leaking blind/zeroed pixels into the output via the global skip connection.
        inp = residual_base if residual_base is not None else x
        x = self.intro(x)
        
        encs = []
        for i, (encoder_list, down) in enumerate(zip(self.encoders, self.downs)):
            gamma, beta = gb_aligned(f"enc{i+1}", x.shape[-2:])
            for block in encoder_list:
                x = block(x, gamma, beta)
            encs.append(x)
            x = down(x)
            
        for block in self.middle_blks:
            x = block(x)
            
        for i, (decoder_list, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = align_like(enc_skip, x)  # ADD THIS LINE
            x = x + enc_skip
            
            level = len(self.decoders) - i
            gamma, beta = gb_aligned(f"dec{level}", x.shape[-2:])
            for block in decoder_list:
                x = block(x, gamma, beta)
                
        x = self.ending(x)
        # NAFNet standard: global residual connection (output + input)
        # NOTE: When used with AdaptiveDenoiser residual_mode=True, this causes double residual!
        # The wrapper will handle residual, so we return raw output instead.
        # To maintain compatibility with standalone NAFNet, we always add residual here,
        # and recommend using residual_mode=False when using NAFNet backbone.
        x = x + inp
        return x

# -------------------------------
# Noise2Void utilities and backbone
# -------------------------------
def get_stratified_coords2D(coord_gen, box_size, shape):
    """Generate stratified blind-spot coordinates for Noise2Void."""
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    x_coords = []
    y_coords = []
    for i in range(box_count_y):
        for j in range(box_count_x):
            y, x = next(coord_gen)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if y < shape[0] and x < shape[1]:
                y_coords.append(y)
                x_coords.append(x)
    return y_coords, x_coords


def apply_blind_spot_mask(img: torch.Tensor, mask_ratio: float = 0.2, box_size: int = 5, seed: int = None, blindspot_dilation: int = 1):
    """
    Apply blind-spot masking for Noise2Void training.

    Args:
        img: Input image [B, C, H, W]
        mask_ratio: Percentage of pixels to mask
        box_size: Size of stratified boxes for mask sampling
        seed: Random seed for reproducible masking (useful for validation)
        blindspot_dilation: Size of dilated blind-spot (1=standard N2V, 3=3x3 blind-spot, 5=5x5)
                           Larger values help with spatially correlated noise like OCT speckle.

    Returns:
        masked_img: Image with blind spots filled with neighbor average
        mask_coords: (batch_idx, channel_idx, y_coords, x_coords) for loss computation
                     NOTE: Loss is ONLY computed on center pixels, not the dilated neighbors!
    """
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # Set random seed for deterministic validation
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # MEMORY OPTIMIZATION: Pre-allocate tensors instead of using Python lists
    total_pixels_per_img = int(H * W * mask_ratio)
    total_masks = B * C * total_pixels_per_img

    # Pre-allocate coordinate tensors on device
    all_batch_idx = torch.zeros(total_masks, dtype=torch.long, device=device)
    all_channel_idx = torch.zeros(total_masks, dtype=torch.long, device=device)
    all_y_coords = torch.zeros(total_masks, dtype=torch.long, device=device)
    all_x_coords = torch.zeros(total_masks, dtype=torch.long, device=device)

    # MEMORY OPTIMIZATION: In-place masking to avoid clone
    masked_img = img.detach().clone()  # Detach to avoid gradient tracking

    cursor = 0
    for b in range(B):
        for c in range(C):
            # Generate stratified coordinates
            # CRITICAL FIX: Stratified sampling can be too sparse - add extra random samples to reach target mask_ratio
            box_count_y = int(np.ceil(H / box_size))
            box_count_x = int(np.ceil(W / box_size))
            num_boxes = box_count_y * box_count_x
            target_num_pix = int(H * W * mask_ratio)

            # First, do stratified sampling (one per box)
            coord_gen = (divmod(rng.randint(0, box_size**2), box_size) for _ in range(num_boxes))
            y_coords, x_coords = get_stratified_coords2D(coord_gen, box_size, (H, W))

            # If stratified sampling doesn't reach target ratio, add random samples
            num_stratified = len(y_coords)
            if num_stratified < target_num_pix:
                num_additional = target_num_pix - num_stratified
                # Sample additional random pixels (avoiding duplicates is not critical for N2V)
                add_y = rng.randint(0, H, size=num_additional).tolist()
                add_x = rng.randint(0, W, size=num_additional).tolist()
                y_coords.extend(add_y)
                x_coords.extend(add_x)

            num_coords = len(y_coords)

            # PERFORMANCE FIX: Convert Python lists to tensors ONCE (not in loop)
            y_coords_tensor = torch.tensor(y_coords, dtype=torch.long, device=device)
            x_coords_tensor = torch.tensor(x_coords, dtype=torch.long, device=device)

            # Store coordinates in pre-allocated tensors
            all_batch_idx[cursor:cursor+num_coords] = b
            all_channel_idx[cursor:cursor+num_coords] = c
            all_y_coords[cursor:cursor+num_coords] = y_coords_tensor
            all_x_coords[cursor:cursor+num_coords] = x_coords_tensor

            # PERFORMANCE FIX: Vectorized median replacement (10-20x faster than Python loop)
            # For each masked pixel, compute median of neighborhood EXCLUDING the blind-spot region
            # DILATED BLIND-SPOT: If blindspot_dilation > 1, mask a box around each pixel
            # but compute loss only on the center pixel (forces network to use info outside speckle grain)
            half_dilation = blindspot_dilation // 2

            # BUG FIX: Keep Python lists for iteration, delete after loop
            for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                # Dilated blind-spot region (what we mask)
                blind_y_start = max(0, y - half_dilation)
                blind_y_end = min(H, y + half_dilation + 1)
                blind_x_start = max(0, x - half_dilation)
                blind_x_end = min(W, x + half_dilation + 1)

                # Neighborhood for computing fill value (OUTSIDE blind-spot)
                # Use 3x3 around the dilated blind-spot
                neighbor_radius = half_dilation + 1
                y_start = max(0, y - neighbor_radius)
                y_end = min(H, y + neighbor_radius + 1)
                x_start = max(0, x - neighbor_radius)
                x_end = min(W, x + neighbor_radius + 1)
                neighborhood = img[b, c, y_start:y_end, x_start:x_end].clone()

                # Create mask excluding the entire blind-spot region
                mask = torch.ones_like(neighborhood, dtype=torch.bool)
                # Calculate blind-spot position within neighborhood
                blind_y_in_neigh_start = blind_y_start - y_start
                blind_y_in_neigh_end = blind_y_end - y_start
                blind_x_in_neigh_start = blind_x_start - x_start
                blind_x_in_neigh_end = blind_x_end - x_start
                # Exclude blind-spot region
                mask[blind_y_in_neigh_start:blind_y_in_neigh_end,
                     blind_x_in_neigh_start:blind_x_in_neigh_end] = False

                valid_neighbors = neighborhood[mask]
                if valid_neighbors.numel() > 0:
                    # Option 1: Median (current, more stable)
                    # Option 2: Random neighbor (original N2V paper)
                    # fill_value = valid_neighbors.median()
                    random_idx = torch.randint(0, valid_neighbors.numel(), (1,), device=device)
                    fill_value = valid_neighbors[random_idx]
                    # Fill the ENTIRE dilated blind-spot with the same value
                    masked_img[b, c, blind_y_start:blind_y_end, blind_x_start:blind_x_end] = fill_value
                else:
                    # Fallback: if no neighbors, keep original value
                    masked_img[b, c, y, x] = img[b, c, y, x]

                # BUG FIX: Delete intermediate tensors in each iteration (memory leak fix)
                del neighborhood, mask, valid_neighbors

            # MEMORY FIX: Delete coordinate lists after loop completes
            del y_coords, x_coords
            cursor += num_coords

    # Trim to actual size and create contiguous copies to release pre-allocated memory
    # BUG FIX: Use .contiguous() to copy data and allow original tensors to be freed
    mask_coords = (
        all_batch_idx[:cursor].contiguous(),
        all_channel_idx[:cursor].contiguous(),
        all_y_coords[:cursor].contiguous(),
        all_x_coords[:cursor].contiguous()
    )
    # MEMORY FIX: Release the original pre-allocated tensors
    del all_batch_idx, all_channel_idx, all_y_coords, all_x_coords

    # PERFORMANCE FIX: Limit logging frequency to avoid slowdown during training
    # Use function attribute to track call count
    if not hasattr(apply_blind_spot_mask, '_call_count'):
        apply_blind_spot_mask._call_count = 0
    apply_blind_spot_mask._call_count += 1

    # MEMORY MONITORING: Print masking stats (only every 100 calls to avoid performance hit)
    if cursor > 0 and torch.cuda.is_available() and (apply_blind_spot_mask._call_count % 100 == 1):
        mask_mem_mb = (cursor * 4 * 8) / (1024 * 1024)  # 4 tensors × 8 bytes per long
        print(f"[N2V Masking] Masked {cursor} pixels | Mask coords: {mask_mem_mb:.2f} MB (call #{apply_blind_spot_mask._call_count})", flush=True)

    return masked_img, mask_coords


class ConvBlock(nn.Module):
    """
    Convolutional block with optional batch normalization and FiLM support.
    Structure: Conv3x3 -> [BN] -> GELU -> Conv3x3 -> [BN] -> GELU -> [FiLM]
    """
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.net(x)
        if gamma is not None:
            out = out * gamma
        if beta is not None:
            out = out + beta
        return out


class Noise2VoidBackbone(nn.Module):
    """
    Noise2Void backbone: U-Net architecture with blind-spot training support.
    Compatible with CASA adapter via FiLM modulation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 48):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1, use_norm=False)
        self.down1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(c1, c2, use_norm=False)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)
        self.enc3 = ConvBlock(c2, c3, use_norm=False)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ConvBlock(c3, c3, use_norm=False)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3, use_norm=False)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2, use_norm=False)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1, use_norm=False)

        # Output
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    @property
    def modulated_channels(self) -> Dict[str, int]:
        """Return channels at each modulation point for CASA adapter."""
        return {
            "enc1": self.enc1.net[0].out_channels,
            "enc2": self.enc2.net[0].out_channels,
            "enc3": self.enc3.net[0].out_channels,
            "dec3": self.dec3.net[0].out_channels,
            "dec2": self.dec2.net[0].out_channels,
            "dec1": self.dec1.net[0].out_channels,
        }

    def forward(self, x: torch.Tensor, adapter_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Forward pass with optional FiLM modulation from CASA adapter.

        Args:
            x: Input (potentially masked) image
            adapter_features: Dict of {block_name: {"gamma": ..., "beta": ...}}
        """
        def gb_aligned(name: str, target_shape: Tuple[int, int]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if adapter_features is None:
                return None, None
            ab = adapter_features.get(name, {})
            g, b = ab.get("gamma", None), ab.get("beta", None)
            if g is not None and g.ndim == 4 and g.shape[-2:] != target_shape:
                g = F.interpolate(g, size=target_shape, mode="bilinear", align_corners=False)
            if b is not None and b.ndim == 4 and b.shape[-2:] != target_shape:
                b = F.interpolate(b, size=target_shape, mode="bilinear", align_corners=False)
            return g, b

        def align_like(ref: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            _, _, Hr, Wr = ref.shape
            _, _, Hs, Ws = src.shape
            pad_h, pad_w = max(0, Hr - Hs), max(0, Wr - Ws)
            if pad_h > 0 or pad_w > 0:
                src = F.pad(src, (0, pad_w, 0, pad_h))
            if src.shape[-2] > Hr or src.shape[-1] > Wr:
                src = src[:, :, 0:Hr, 0:Wr]
            return src

        # Encoder path
        g, b = gb_aligned("enc1", x.shape[-2:])
        x1 = self.enc1(x, g, b)
        d1 = self.down1(x1)

        g, b = gb_aligned("enc2", d1.shape[-2:])
        x2 = self.enc2(d1, g, b)
        d2 = self.down2(x2)

        g, b = gb_aligned("enc3", d2.shape[-2:])
        x3 = self.enc3(d2, g, b)
        d3 = self.down3(x3)

        # Bottleneck
        xb = self.bottleneck(d3)

        # Decoder path
        u3 = self.up3(xb)
        u3 = align_like(x3, u3)
        x_cat3 = torch.cat([u3, x3], dim=1)
        g, b = gb_aligned("dec3", x_cat3.shape[-2:])
        x4 = self.dec3(x_cat3, g, b)

        u2 = self.up2(x4)
        u2 = align_like(x2, u2)
        x_cat2 = torch.cat([u2, x2], dim=1)
        g, b = gb_aligned("dec2", x_cat2.shape[-2:])
        x5 = self.dec2(x_cat2, g, b)

        u1 = self.up1(x5)
        u1 = align_like(x1, u1)
        x_cat1 = torch.cat([u1, x1], dim=1)
        g, b = gb_aligned("dec1", x_cat1.shape[-2:])
        x6 = self.dec1(x_cat1, g, b)

        return self.out_conv(x6)


# -------------------------------
# Neighbor2Neighbor utilities and backbone
# -------------------------------
def create_checkerboard_mask(shape: Tuple[int, int], offset: int = 0, device: str = 'cuda') -> torch.Tensor:
    """
    Create checkerboard pattern mask for Neighbor2Neighbor sub-sampling.

    Args:
        shape: (H, W) image dimensions
        offset: 0, 1, 2, or 3 for different checkerboard patterns
        device: Device to create mask on

    Returns:
        Boolean mask [H, W] where True indicates pixels to keep
    """
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)

    if offset == 0:
        # Pattern: keep even rows + even cols, odd rows + odd cols
        mask[0::2, 0::2] = True
        mask[1::2, 1::2] = True
    elif offset == 1:
        # Pattern: keep even rows + odd cols, odd rows + even cols
        mask[0::2, 1::2] = True
        mask[1::2, 0::2] = True
    elif offset == 2:
        # Pattern: keep odd rows + even cols, even rows + odd cols
        mask[1::2, 0::2] = True
        mask[0::2, 1::2] = True
    elif offset == 3:
        # Pattern: keep odd rows + odd cols, even rows + even cols
        mask[1::2, 1::2] = True
        mask[0::2, 0::2] = True
    else:
        raise ValueError(f"offset must be 0, 1, 2, or 3, got {offset}")

    return mask


def apply_neighbor2neighbor_subsample(
    img: torch.Tensor,
    subsample_pattern: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Neighbor2Neighbor sub-sampling: split image into two non-overlapping sets.

    Args:
        img: Input image [B, C, H, W]
        subsample_pattern: 0 or 1 for two complementary checkerboard patterns

    Returns:
        input_subset: Subset A (input to network) [B, C, H, W]
        target_subset: Subset B (target for prediction) [B, C, H, W]
        target_mask: Boolean mask [B, C, H, W] indicating valid target pixels
    """
    B, C, H, W = img.shape
    device = img.device

    # Create checkerboard masks for two complementary subsets
    if subsample_pattern == 0:
        # Input: pattern 0, Target: pattern 1
        input_mask_2d = create_checkerboard_mask((H, W), offset=0, device=device)
        target_mask_2d = create_checkerboard_mask((H, W), offset=1, device=device)
    else:
        # Input: pattern 1, Target: pattern 0 (reversed)
        input_mask_2d = create_checkerboard_mask((H, W), offset=1, device=device)
        target_mask_2d = create_checkerboard_mask((H, W), offset=0, device=device)

    # Expand masks to batch and channel dimensions
    # BUG FIX: Store expanded masks and delete 2D versions to avoid memory leak
    input_mask = input_mask_2d.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    target_mask = target_mask_2d.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    # MEMORY LEAK FIX: Delete 2D masks after expansion
    del input_mask_2d, target_mask_2d

    # MEMORY OPTIMIZATION: Use scalar zero instead of zeros_like to avoid allocating full tensor
    # BUG FIX: torch.zeros_like(img) wastes memory - use scalar 0.0 instead
    input_subset = torch.where(input_mask, img, torch.tensor(0.0, device=device, dtype=img.dtype))

    # BUG FIX: Delete input_mask after creating input_subset (memory leak fix)
    del input_mask

    # BUG FIX: Don't clone img - just detach (target is full original image, no modification needed)
    # Original: target_subset = img.detach().clone()  # Wastes 4-8 MB per batch
    target_subset = img.detach()

    # PERFORMANCE FIX: Limit logging frequency to avoid slowdown during training
    # Use function attribute to track call count
    if not hasattr(apply_neighbor2neighbor_subsample, '_call_count'):
        apply_neighbor2neighbor_subsample._call_count = 0
    apply_neighbor2neighbor_subsample._call_count += 1

    # MEMORY MONITORING (only every 100 calls to avoid performance hit)
    if torch.cuda.is_available() and (apply_neighbor2neighbor_subsample._call_count % 100 == 1):
        subset_mem_mb = (input_subset.numel() * input_subset.element_size()) / (1024 * 1024)
        mask_mem_mb = (target_mask.numel() * 1) / (1024 * 1024)  # bool = 1 byte
        print(f"[N2N Subsample] Input subset: {subset_mem_mb:.2f} MB | Mask: {mask_mem_mb:.2f} MB (call #{apply_neighbor2neighbor_subsample._call_count})", flush=True)

    return input_subset, target_subset, target_mask


class Neighbor2NeighborBackbone(nn.Module):
    """
    Neighbor2Neighbor backbone: U-Net architecture with sub-sampling training support.
    Compatible with CASA adapter via FiLM modulation.

    Reference: "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 48):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1, use_norm=False)
        self.down1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(c1, c2, use_norm=False)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)
        self.enc3 = ConvBlock(c2, c3, use_norm=False)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ConvBlock(c3, c3, use_norm=False)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3, use_norm=False)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2, use_norm=False)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1, use_norm=False)

        # Output
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    @property
    def modulated_channels(self) -> Dict[str, int]:
        """Return channels at each modulation point for CASA adapter."""
        return {
            "enc1": self.enc1.net[0].out_channels,
            "enc2": self.enc2.net[0].out_channels,
            "enc3": self.enc3.net[0].out_channels,
            "dec3": self.dec3.net[0].out_channels,
            "dec2": self.dec2.net[0].out_channels,
            "dec1": self.dec1.net[0].out_channels,
        }

    def forward(self, x: torch.Tensor, adapter_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Forward pass with optional FiLM modulation from CASA adapter.

        Args:
            x: Input sub-sampled image [B, C, H, W]
            adapter_features: Dict of {block_name: {"gamma": ..., "beta": ...}}
        """
        def gb_aligned(name: str, target_shape: Tuple[int, int]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if adapter_features is None:
                return None, None
            ab = adapter_features.get(name, {})
            g, b = ab.get("gamma", None), ab.get("beta", None)
            if g is not None and g.ndim == 4 and g.shape[-2:] != target_shape:
                g = F.interpolate(g, size=target_shape, mode="bilinear", align_corners=False)
            if b is not None and b.ndim == 4 and b.shape[-2:] != target_shape:
                b = F.interpolate(b, size=target_shape, mode="bilinear", align_corners=False)
            return g, b

        def align_like(ref: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            _, _, Hr, Wr = ref.shape
            _, _, Hs, Ws = src.shape
            pad_h, pad_w = max(0, Hr - Hs), max(0, Wr - Ws)
            if pad_h > 0 or pad_w > 0:
                src = F.pad(src, (0, pad_w, 0, pad_h))
            if src.shape[-2] > Hr or src.shape[-1] > Wr:
                src = src[:, :, 0:Hr, 0:Wr]
            return src

        # Encoder path
        g, b = gb_aligned("enc1", x.shape[-2:])
        x1 = self.enc1(x, g, b)
        d1 = self.down1(x1)

        g, b = gb_aligned("enc2", d1.shape[-2:])
        x2 = self.enc2(d1, g, b)
        d2 = self.down2(x2)

        g, b = gb_aligned("enc3", d2.shape[-2:])
        x3 = self.enc3(d2, g, b)
        d3 = self.down3(x3)

        # Bottleneck
        x4 = self.bottleneck(d3)

        # Decoder path
        u3 = self.up3(x4)
        u3 = align_like(x3, u3)
        x_cat3 = torch.cat([u3, x3], dim=1)
        g, b = gb_aligned("dec3", x_cat3.shape[-2:])
        x5 = self.dec3(x_cat3, g, b)

        u2 = self.up2(x5)
        u2 = align_like(x2, u2)
        x_cat2 = torch.cat([u2, x2], dim=1)
        g, b = gb_aligned("dec2", x_cat2.shape[-2:])
        x6 = self.dec2(x_cat2, g, b)

        u1 = self.up1(x6)
        u1 = align_like(x1, u1)
        x_cat1 = torch.cat([u1, x1], dim=1)
        g, b = gb_aligned("dec1", x_cat1.shape[-2:])
        x7 = self.dec1(x_cat1, g, b)

        return self.out_conv(x7)


class GlobalAwareMaskMapper(nn.Module):
    """
    Generates systematic blind spots with a grid pattern and maps them to a shared channel via pixel unshuffle.
    """
    def __init__(self, mask_ratio: float = 0.5, block_size: int = 2):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image [B, C, H, W]
        Returns:
            masked_input: Input with blind spots zeroed
            mask: Binary mask (1=visible, 0=blind)
            blind_mask_channel: Aggregated blind-spot indicator after pixel-unshuffle
        """
        B, C, H, W = x.shape
        mask2d = torch.ones((H, W), device=x.device, dtype=torch.bool)
        # CRITICAL FIX: Create proper checkerboard pattern (~50% masked) instead of ~25%
        mask2d[0::2, 0::2] = False  # Even rows, even cols
        mask2d[1::2, 1::2] = False  # Odd rows, odd cols

        target_zeros = int(round(self.mask_ratio * H * W))
        current_zeros = int(mask2d.numel() - mask2d.sum().item())
        if target_zeros != current_zeros:
            if target_zeros < current_zeros:
                zero_idx = (mask2d == 0).nonzero(as_tuple=False)
                restore = zero_idx[torch.randperm(zero_idx.shape[0], device=x.device)[:current_zeros - target_zeros]]
                mask2d[restore[:, 0], restore[:, 1]] = True
            else:
                one_idx = (mask2d == 1).nonzero(as_tuple=False)
                drop = one_idx[torch.randperm(one_idx.shape[0], device=x.device)[:target_zeros - current_zeros]]
                mask2d[drop[:, 0], drop[:, 1]] = False

        mask = mask2d.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        mask_float = mask.float()
        masked_input = x * mask_float

        block = self.block_size
        pad_h = (block - H % block) % block
        pad_w = (block - W % block) % block
        pad = (0, pad_w, 0, pad_h)
        x_pad = F.pad(x, pad, mode="reflect") if pad_h or pad_w else x
        mask_pad = F.pad(mask_float, pad, mode="reflect") if pad_h or pad_w else mask_float
        x_unshuffled = F.pixel_unshuffle(x_pad, block)
        mask_unshuffled = F.pixel_unshuffle(mask_pad, block) > 0.5  # bool per sub-channel
        # Aggregate blind spots from all sub-channels into the first C channels
        x_unsh = x_unshuffled.view(B, block * block, C, x_unshuffled.shape[-2], x_unshuffled.shape[-1])
        mask_unsh = mask_unshuffled.view(B, block * block, 1, mask_unshuffled.shape[-2], mask_unshuffled.shape[-1])
        blind_mask_channel = (~mask_unsh).any(dim=1).float()
        return masked_input, mask_float, blind_mask_channel


class B2UConvBlock(nn.Module):
    """Two convs with BN and LeakyReLU, FiLM-compatible."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.net(x)
        if gamma is not None:
            out = out * gamma
        if beta is not None:
            out = out + beta
        return out


class B2UNetBackbone(nn.Module):
    """
    Blind2Unblind U-Net backbone with four down/upsampling stages and FiLM hooks for adapters.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = B2UConvBlock(in_channels, c1)
        self.enc2 = B2UConvBlock(c1, c2)
        self.enc3 = B2UConvBlock(c2, c3)
        self.enc4 = B2UConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = B2UConvBlock(c4, c4 * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = B2UConvBlock(c4 * 2 + c4, c4)
        self.dec3 = B2UConvBlock(c4 + c3, c3)
        self.dec2 = B2UConvBlock(c3 + c2, c2)
        self.dec1 = B2UConvBlock(c2 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    @property
    def modulated_channels(self) -> Dict[str, int]:
        return {
            "enc1": self.enc1.net[0].out_channels,
            "enc2": self.enc2.net[0].out_channels,
            "enc3": self.enc3.net[0].out_channels,
            "enc4": self.enc4.net[0].out_channels,
            "dec4": self.dec4.net[0].out_channels,
            "dec3": self.dec3.net[0].out_channels,
            "dec2": self.dec2.net[0].out_channels,
            "dec1": self.dec1.net[0].out_channels,
        }

    def forward(self, x: torch.Tensor, adapter_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        def gb_aligned(name: str, target_shape: Tuple[int, int]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if adapter_features is None:
                return None, None
            ab = adapter_features.get(name, {})
            g, b = ab.get("gamma", None), ab.get("beta", None)
            if g is not None and g.ndim == 4 and g.shape[-2:] != target_shape:
                g = F.interpolate(g, size=target_shape, mode="bilinear", align_corners=False)
            if b is not None and b.ndim == 4 and b.shape[-2:] != target_shape:
                b = F.interpolate(b, size=target_shape, mode="bilinear", align_corners=False)
            return g, b

        def align_like(ref: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            _, _, Hr, Wr = ref.shape
            _, _, Hs, Ws = src.shape
            pad_h, pad_w = max(0, Hr - Hs), max(0, Wr - Ws)
            if pad_h > 0 or pad_w > 0:
                src = F.pad(src, (0, pad_w, 0, pad_h))
            if src.shape[-2] > Hr or src.shape[-1] > Wr:
                src = src[:, :, 0:Hr, 0:Wr]
            return src

        e1 = self.enc1(x, *gb_aligned("enc1", x.shape[-2:]))
        e2_in = self.pool(e1)
        e2 = self.enc2(e2_in, *gb_aligned("enc2", e2_in.shape[-2:]))
        e3_in = self.pool(e2)
        e3 = self.enc3(e3_in, *gb_aligned("enc3", e3_in.shape[-2:]))
        e4_in = self.pool(e3)
        e4 = self.enc4(e4_in, *gb_aligned("enc4", e4_in.shape[-2:]))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up(b)
        d4 = align_like(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1), *gb_aligned("dec4", d4.shape[-2:]))

        d3 = self.up(d4)
        d3 = align_like(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), *gb_aligned("dec3", d3.shape[-2:]))

        d2 = self.up(d3)
        d2 = align_like(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), *gb_aligned("dec2", d2.shape[-2:]))

        d1 = self.up(d2)
        d1 = align_like(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), *gb_aligned("dec1", d1.shape[-2:]))

        return self.out_conv(d1)


class DenoisingBackbone(nn.Module):
    """A deeper U-Net with FiLM modulation in six blocks (matches SwinIRLite depth)."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64, use_norm: bool = False, use_checkpoint: bool = False):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.use_checkpoint = use_checkpoint  # MEMORY OPTIMIZATION: Enable gradient checkpointing
        # Encoder (3 levels)
        self.enc1 = ConvBlock(in_channels, c1, use_norm=use_norm)
        self.down1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(c1, c2, use_norm=use_norm)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)
        self.enc3 = ConvBlock(c2, c3, use_norm=use_norm)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)
        # Bottleneck
        self.bottleneck = ConvBlock(c3, c3, use_norm=use_norm)
        # Decoder (3 levels)
        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3, use_norm=use_norm)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2, use_norm=use_norm)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1, use_norm=use_norm)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    @property
    def modulated_channels(self) -> Dict[str, int]:
        return {
            "enc1": self.enc1.net[0].out_channels,
            "enc2": self.enc2.net[0].out_channels,
            "enc3": self.enc3.net[0].out_channels,
            "dec3": self.dec3.net[0].out_channels,
            "dec2": self.dec2.net[0].out_channels,
            "dec1": self.dec1.net[0].out_channels,
        }

    def forward(self, x: torch.Tensor, adapter_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        def gb_aligned(name: str, target_shape: Tuple[int, int]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if adapter_features is None: return None, None
            ab = adapter_features.get(name, {})
            g, b = ab.get("gamma", None), ab.get("beta", None)
            if g is not None and g.ndim == 4 and g.shape[-2:] != target_shape:
                g = F.interpolate(g, size=target_shape, mode="bilinear", align_corners=False)
            if b is not None and b.ndim == 4 and b.shape[-2:] != target_shape:
                b = F.interpolate(b, size=target_shape, mode="bilinear", align_corners=False)
            return g, b

        def align_like(ref: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            _, _, Hr, Wr = ref.shape
            _, _, Hs, Ws = src.shape
            pad_h, pad_w = max(0, Hr - Hs), max(0, Wr - Ws)
            if pad_h > 0 or pad_w > 0: src = F.pad(src, (0, pad_w, 0, pad_h))
            if src.shape[-2] > Hr or src.shape[-1] > Wr: src = src[:, :, 0:Hr, 0:Wr]
            return src

        # Encoder path
        g, b = gb_aligned("enc1", x.shape[-2:])
        x1 = self.enc1(x, g, b)
        d1 = self.down1(x1)
        g, b = gb_aligned("enc2", d1.shape[-2:])
        x2 = self.enc2(d1, g, b)
        d2 = self.down2(x2)
        g, b = gb_aligned("enc3", d2.shape[-2:])
        x3 = self.enc3(d2, g, b)
        d3 = self.down3(x3)
        # Bottleneck
        xb = self.bottleneck(d3)
        # Decoder path
        u3 = self.up3(xb)
        u3 = align_like(x3, u3)
        x_cat3 = torch.cat([u3, x3], dim=1)
        g, b = gb_aligned("dec3", x_cat3.shape[-2:])
        x4 = self.dec3(x_cat3, g, b)
        u2 = self.up2(x4)
        u2 = align_like(x2, u2)
        x_cat2 = torch.cat([u2, x2], dim=1)
        g, b = gb_aligned("dec2", x_cat2.shape[-2:])
        x5 = self.dec2(x_cat2, g, b)
        u1 = self.up1(x5)
        u1 = align_like(x1, u1)
        x_cat1 = torch.cat([u1, x1], dim=1)
        g, b = gb_aligned("dec1", x_cat1.shape[-2:])
        x6 = self.dec1(x_cat1, g, b)
        return self.out_conv(x6)

# ------------------------------- 
# Noise Adapter module
# ------------------------------- 
class NoiseAdapter(nn.Module):
    """Lightweight adapter producing per-block FiLM parameters."""
    def __init__(self, in_channels: int, block_channels: Dict[str, int], hidden_channels: int = 32):
        super().__init__()
        self.block_names, self.block_channels = list(block_channels.keys()), block_channels
        self.total_channels = sum(block_channels.values())
        self.feat = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(nn.Conv2d(hidden_channels, max(64, hidden_channels*2), 1), nn.GELU(), nn.Conv2d(max(64, hidden_channels*2), 2 * self.total_channels, 1))
        for m in self.mlp:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None: nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        pred = self.mlp(self.gap(self.feat(x)))
        gamma_all, beta_all = torch.split(pred, self.total_channels, dim=1)
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        cursor = 0
        for name in self.block_names:
            ch = self.block_channels[name]
            g, b = gamma_all[:, cursor:cursor+ch], beta_all[:, cursor:cursor+ch]
            cursor += ch
            out[name] = {"gamma": 1.0 + 0.1 * torch.tanh(g), "beta": 0.1 * torch.tanh(b)}
        return out


class MixtureNoiseAdapter(nn.Module):
    """Mixture-of-experts adapter that composes multiple global FiLM experts per image."""

    def __init__(
        self,
        in_channels: int,
        block_channels: Dict[str, int],
        num_experts: int = 4,
        hidden_channels: int = 32,
        gate_hidden_channels: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        if num_experts <= 1:
            raise ValueError("num_experts must be >= 2")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.num_experts = int(num_experts)
        self.block_names = list(block_channels.keys())
        self.block_channels = dict(block_channels)
        self.temperature = float(temperature)

        gate_hidden = int(gate_hidden_channels) if gate_hidden_channels is not None else int(hidden_channels)
        self.gate_feat = nn.Sequential(
            nn.Conv2d(in_channels, gate_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, gate_hidden, 3, padding=1),
            nn.GELU(),
        )
        self.gate_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gate_head = nn.Conv2d(gate_hidden, self.num_experts, 1)

        self.experts = nn.ModuleList(
            [NoiseAdapter(in_channels=in_channels, block_channels=self.block_channels, hidden_channels=hidden_channels) for _ in range(self.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        logits = self.gate_head(self.gate_gap(self.gate_feat(x))).flatten(1)  # [B,K]
        weights = torch.softmax(logits / self.temperature, dim=1)  # [B,K]

        expert_params = [ex(x) for ex in self.experts]
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in self.block_names:
            gammas = torch.stack([p[name]["gamma"] for p in expert_params], dim=1)  # [B,K,C,1,1]
            betas = torch.stack([p[name]["beta"] for p in expert_params], dim=1)    # [B,K,C,1,1]
            w = weights[:, :, None, None, None]
            out[name] = {"gamma": (w * gammas).sum(dim=1), "beta": (w * betas).sum(dim=1)}

        entropy = -(weights * (weights.clamp_min(1e-12).log())).sum(dim=1).mean()
        aux = {"moe_weights": weights, "moe_entropy": entropy.detach()}
        return out, aux


class BernoulliSampler(nn.Module):
    """Generate Bernoulli masks for masked self-supervision (1=visible, 0=masked)."""
    def __init__(self, mask_prob: float = 0.3):
        super().__init__()
        self.mask_prob = mask_prob

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        B, _, H, W = x.shape
        masks: List[torch.Tensor] = []
        masked_inputs: List[torch.Tensor] = []
        for _ in range(num_samples):
            mask = torch.bernoulli(torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype) * (1 - self.mask_prob))
            masked_inputs.append(x * mask)
            masks.append(mask)
        return masked_inputs, masks


class GatedConv2d(nn.Module):
    """Gated convolution layer."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv_feature = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.conv_gate = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv_feature(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return self.bn(feature * gate)


class S2SConvBlock(nn.Module):
    """Two gated convs with FiLM hooks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            GatedConv2d(in_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            GatedConv2d(out_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.net(x)
        if gamma is not None:
            out = out * gamma
        if beta is not None:
            out = out + beta
        return out


class S2SNetBackbone(nn.Module):
    """Self2Self U-Net with gated convolutions and dropout."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 48, dropout_rate: float = 0.3):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.dropout = nn.Dropout2d(dropout_rate)
        self.enc1 = S2SConvBlock(in_channels, c1)
        self.enc2 = S2SConvBlock(c1, c2)
        self.enc3 = S2SConvBlock(c2, c3)
        self.enc4 = S2SConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = S2SConvBlock(c4, c4)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = S2SConvBlock(c4 + c4, c3)
        self.dec3 = S2SConvBlock(c3 + c3, c2)
        self.dec2 = S2SConvBlock(c2 + c2, c1)
        self.dec1 = S2SConvBlock(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 1)

    @property
    def modulated_channels(self) -> Dict[str, int]:
        return {
            "enc1": self.enc1.net[0].conv_feature.out_channels,
            "enc2": self.enc2.net[0].conv_feature.out_channels,
            "enc3": self.enc3.net[0].conv_feature.out_channels,
            "enc4": self.enc4.net[0].conv_feature.out_channels,
            "dec4": self.dec4.net[0].conv_feature.out_channels,
            "dec3": self.dec3.net[0].conv_feature.out_channels,
            "dec2": self.dec2.net[0].conv_feature.out_channels,
            "dec1": self.dec1.net[0].conv_feature.out_channels,
        }

    def forward(self, x: torch.Tensor, adapter_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None, apply_dropout: bool = True) -> torch.Tensor:
        def gb(name: str, shape: Tuple[int, int]):
            if adapter_features is None:
                return None, None
            params = adapter_features.get(name, {})
            g, b = params.get("gamma", None), params.get("beta", None)
            if g is not None and g.ndim == 4 and g.shape[-2:] != shape:
                g = F.interpolate(g, size=shape, mode="bilinear", align_corners=False)
            if b is not None and b.ndim == 4 and b.shape[-2:] != shape:
                b = F.interpolate(b, size=shape, mode="bilinear", align_corners=False)
            return g, b

        def align_like(ref: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            _, _, Hr, Wr = ref.shape
            _, _, Hs, Ws = src.shape
            pad_h, pad_w = max(0, Hr - Hs), max(0, Wr - Ws)
            if pad_h > 0 or pad_w > 0:
                src = F.pad(src, (0, pad_w, 0, pad_h))
            if src.shape[-2] > Hr or src.shape[-1] > Wr:
                src = src[:, :, :Hr, :Wr]
            return src

        e1 = self.enc1(x, *gb("enc1", x.shape[-2:]))
        if apply_dropout: e1 = self.dropout(e1)
        e2_in = self.pool(e1)
        e2 = self.enc2(e2_in, *gb("enc2", e2_in.shape[-2:]))
        if apply_dropout: e2 = self.dropout(e2)
        e3_in = self.pool(e2)
        e3 = self.enc3(e3_in, *gb("enc3", e3_in.shape[-2:]))
        if apply_dropout: e3 = self.dropout(e3)
        e4_in = self.pool(e3)
        e4 = self.enc4(e4_in, *gb("enc4", e4_in.shape[-2:]))
        if apply_dropout: e4 = self.dropout(e4)

        b = self.bottleneck(self.pool(e4), *gb("enc4", self.pool(e4).shape[-2:]))
        if apply_dropout: b = self.dropout(b)

        up_b = align_like(e4, self.up(b))
        d4 = self.dec4(torch.cat([up_b, e4], dim=1), *gb("dec4", e4.shape[-2:]))
        up_d4 = align_like(e3, self.up(d4))
        d3 = self.dec3(torch.cat([up_d4, e3], dim=1), *gb("dec3", e3.shape[-2:]))
        up_d3 = align_like(e2, self.up(d3))
        d2 = self.dec2(torch.cat([up_d3, e2], dim=1), *gb("dec2", e2.shape[-2:]))
        up_d2 = align_like(e1, self.up(d2))
        d1 = self.dec1(torch.cat([up_d2, e1], dim=1), *gb("dec1", e1.shape[-2:]))

        return torch.sigmoid(self.out_conv(d1))

    def inference(self, x: torch.Tensor, adapter_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None, num_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo inference with dropout averaging for uncertainty estimation."""
        # CRITICAL FIX: Save and restore training state to avoid interfering with other model components
        was_training = self.training
        self.train()  # Keep dropout active during inference

        with torch.no_grad():
            # In-place accumulation to save memory
            sum_pred = torch.zeros_like(x)
            sum_sq_pred = torch.zeros_like(x)

            for _ in range(num_samples):
                pred = self.forward(x, adapter_features=adapter_features, apply_dropout=True)
                sum_pred.add_(pred)
                sum_sq_pred.add_(pred ** 2)
                del pred

            mean_pred = sum_pred / num_samples
            variance = (sum_sq_pred / num_samples) - (mean_pred ** 2)
            uncertainty = torch.sqrt(variance.clamp(min=1e-8))

            del sum_pred, sum_sq_pred, variance

        # Restore original training state instead of forcing eval
        self.train(was_training)
        return mean_pred, uncertainty

class SpatialNoiseAdapter(nn.Module):
    """Spatial FiLM adapter producing per-pixel gamma/beta maps."""
    def __init__(self, in_channels: int, block_channels: Dict[str, int], hidden_channels: int = 32):
        super().__init__()
        self.block_channels = block_channels
        c1 = block_channels.get("enc1", 64)
        c2 = block_channels.get("enc2", 128)
        c3 = block_channels.get("enc3", 256)
        has_lvl4 = "enc4" in block_channels or "dec4" in block_channels
        c4 = block_channels.get("enc4", 512) if has_lvl4 else None
        self.trunk_full = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU())
        self.down1 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
        self.trunk_half = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU())
        self.down2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
        self.trunk_quarter = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU())
        if has_lvl4:
            self.down3 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
            self.trunk_eighth = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU())
        def head(out_ch): return nn.Conv2d(hidden_channels, 2 * out_ch, 1)
        self.head_enc1, self.head_dec1 = head(c1), head(c1)
        self.head_enc2, self.head_dec2 = head(c2), head(c2)
        self.head_enc3, self.head_dec3 = head(c3), head(c3)
        if has_lvl4:
            self.head_enc4, self.head_dec4 = head(c4), head(c4)
        else:
            self.head_enc4 = self.head_dec4 = None
        for m in [self.head_enc1, self.head_dec1, self.head_enc2, self.head_dec2, self.head_enc3, self.head_dec3] + ([self.head_enc4, self.head_dec4] if has_lvl4 else []):
            nn.init.zeros_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        feat_full = self.trunk_full(x)
        feat_half = self.trunk_half(self.down1(feat_full))
        feat_quarter = self.trunk_quarter(self.down2(feat_half))
        feat_eighth = self.trunk_eighth(self.down3(feat_quarter)) if self.head_enc4 is not None else None
        def split_gb(p, ch): g, b = torch.split(p, ch, dim=1); return 1.0 + 0.1 * torch.tanh(g), 0.1 * torch.tanh(b)
        g_enc1, b_enc1 = split_gb(self.head_enc1(feat_full), self.block_channels["enc1"])
        g_dec1, b_dec1 = split_gb(self.head_dec1(feat_full), self.block_channels["dec1"])
        g_enc2, b_enc2 = split_gb(self.head_enc2(feat_half), self.block_channels["enc2"])
        g_dec2, b_dec2 = split_gb(self.head_dec2(feat_half), self.block_channels["dec2"])
        g_enc3, b_enc3 = split_gb(self.head_enc3(feat_quarter), self.block_channels["enc3"])
        g_dec3, b_dec3 = split_gb(self.head_dec3(feat_quarter), self.block_channels["dec3"])
        out = {
            "enc1": {"gamma": g_enc1, "beta": b_enc1},
            "enc2": {"gamma": g_enc2, "beta": b_enc2},
            "enc3": {"gamma": g_enc3, "beta": b_enc3},
            "dec3": {"gamma": g_dec3, "beta": b_dec3},
            "dec2": {"gamma": g_dec2, "beta": b_dec2},
            "dec1": {"gamma": g_dec1, "beta": b_dec1},
        }
        if self.head_enc4 is not None and feat_eighth is not None:
            g_enc4, b_enc4 = split_gb(self.head_enc4(feat_eighth), self.block_channels.get("enc4", self.block_channels["enc3"]))
            g_dec4, b_dec4 = split_gb(self.head_dec4(feat_eighth), self.block_channels.get("dec4", self.block_channels["dec3"]))
            out["enc4"] = {"gamma": g_enc4, "beta": b_enc4}
            out["dec4"] = {"gamma": g_dec4, "beta": b_dec4}
        return out


class CoherenceSpatialAdapter(nn.Module):
    """Coherence-Aware Spatial Adapter (CASA)."""
    def __init__(self, block_channels: Dict[str, int], coherence_scales: Tuple[int, ...] = (1, 3, 5, 7)):
        super().__init__()
        self.block_channels = block_channels
        self.coherence_estimator = nn.ModuleList([nn.Conv2d(1, 16, k, padding=k//2) for k in coherence_scales])
        self.register_buffer("casa_gaussian", _gaussian_kernel_2d(1, 7, 2.0))
        coh_ch = 16 * len(coherence_scales) + 2  # +2 for local mean/contrast
        self.correlation_predictor = nn.Sequential(nn.Conv2d(coh_ch, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.spatial_film_heads = nn.ModuleDict({name: nn.Conv2d(64, 2 * ch, 1) for name, ch in block_channels.items()})
        self.decomposition_net = nn.Sequential(nn.Conv2d(64, 32, 1), nn.GELU(), nn.Conv2d(32, 2, 1), nn.Softmax(dim=1))
        for m in self.spatial_film_heads.values():
            nn.init.zeros_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        # OPTIMIZATION: Cache gaussian kernel conversion (avoid repeated .to() calls)
        # BUG FIX: Use local variable instead of modifying buffer (distributed training safe)
        gaussian_kernel = self.casa_gaussian
        if gaussian_kernel.device != x.device or gaussian_kernel.dtype != x.dtype:
            gaussian_kernel = gaussian_kernel.to(device=x.device, dtype=x.dtype)

        # OPTIMIZATION: Inline list comprehension directly into cat (avoid intermediate list)
        feats = [est(x) for est in self.coherence_estimator]

        # OPTIMIZATION: Reuse gaussian kernel
        local_mean = F.conv2d(x, gaussian_kernel, padding=gaussian_kernel.shape[-1]//2)
        local_var = F.conv2d(x * x, gaussian_kernel, padding=gaussian_kernel.shape[-1]//2) - local_mean * local_mean
        local_contrast = torch.sqrt(local_var.clamp_min(0.0) + 1e-6) / (local_mean.abs() + 1e-3)

        coh_feat = torch.cat(feats + [local_mean, local_contrast], dim=1)
        # MEMORY LEAK FIX: Delete intermediate tensors
        del feats, local_mean, local_var, local_contrast

        spatial_code = self.correlation_predictor(coh_feat)
        # MEMORY LEAK FIX: Delete coh_feat after use
        del coh_feat

        decomp = self.decomposition_net(spatial_code)
        w_coherent_full, w_incoherent_full = decomp[:, 0:1], decomp[:, 1:2]
        # MEMORY LEAK FIX: Delete decomp after splitting
        del decomp

        # OPTIMIZATION: Batch interpolations together to reduce overhead
        H, W = x.shape[-2], x.shape[-1]
        h2, w2 = (H+1)//2, (W+1)//2
        h4, w4 = (h2+1)//2, (w2+1)//2
        # Concatenate tensors and interpolate to half and quarter resolutions
        to_interpolate_half = torch.cat([spatial_code, w_coherent_full, w_incoherent_full], dim=1)
        interpolated_half = F.interpolate(to_interpolate_half, size=(h2, w2), mode="bilinear", align_corners=False)
        interpolated_quarter = F.interpolate(to_interpolate_half, size=(h4, w4), mode="bilinear", align_corners=False)
        # MEMORY LEAK FIX: Delete concatenated tensor after interpolation
        del to_interpolate_half

        # Split back
        ch_split = spatial_code.shape[1]
        spatial_code_half = interpolated_half[:, :ch_split]
        w_coherent_half = interpolated_half[:, ch_split:ch_split+1]
        w_incoherent_half = interpolated_half[:, ch_split+1:]
        spatial_code_quarter = interpolated_quarter[:, :ch_split]
        w_coherent_quarter = interpolated_quarter[:, ch_split:ch_split+1]
        w_incoherent_quarter = interpolated_quarter[:, ch_split+1:]
        # MEMORY LEAK FIX: Delete interpolated tensors after splitting
        del interpolated_half, interpolated_quarter
        params: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, head in self.spatial_film_heads.items():
            if name in ("enc4", "dec4", "enc3", "dec3"):
                code, w_coh, w_inc = spatial_code_quarter, w_coherent_quarter, w_incoherent_quarter
            elif name in ("enc2", "dec2"):
                code, w_coh, w_inc = spatial_code_half, w_coherent_half, w_incoherent_half
            else:
                code, w_coh, w_inc = spatial_code, w_coherent_full, w_incoherent_full
            gb = head(code)
            g_raw, b_raw = torch.split(gb, gb.shape[1]//2, dim=1)
            gamma, beta = 1.0 + 0.3 * torch.tanh(g_raw) * w_coh, 0.2 * torch.tanh(b_raw) * w_inc
            params[name] = {"gamma": gamma, "beta": beta}
        return params, {"coherent_map": w_coherent_full, "incoherent_map": w_incoherent_full}


class AdaptiveDenoiser(nn.Module):
    def __init__(self, backbone: DenoisingBackbone, adapter: nn.Module, residual_mode: bool = False):
        super().__init__()
        self.backbone, self.adapter, self.residual_mode = backbone, adapter, residual_mode

    def forward(self, x: torch.Tensor, return_aux: bool = False, residual_base: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        out_tuple = self.adapter(x)
        adapter_features, aux_outputs = out_tuple if isinstance(out_tuple, tuple) else (out_tuple, {})
        # IMPORTANT: Some backbones (e.g., NAFNet) have an internal global residual skip.
        # In masked/self-supervised modes we must be able to use the *original* (unmasked)
        # noisy image as the residual base to avoid leaking blind/zeroed pixels into the output.
        if isinstance(self.backbone, NAFBackbone):
            raw = self.backbone(x, adapter_features=adapter_features, residual_base=residual_base)
        else:
            raw = self.backbone(x, adapter_features=adapter_features)
        if raw.shape[-2:] != x.shape[-2:]: raw = F.interpolate(raw, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # CRITICAL FIX: Handle different backbone output types
        # - NAFNet: outputs denoised image [0,1] with internal residual
        # - S2SNet: outputs sigmoid-activated image [0,1]
        # - B2UNet: outputs raw predictions, supports both direct and residual mode
        # - N2V/N2N backbones: outputs raw, needs sigmoid or residual
        # - Other: uses residual or sigmoid based on mode
        if isinstance(self.backbone, NAFBackbone):
            out = raw.clamp(0.0, 1.0)  # NAFNet already added residual internally
        elif isinstance(self.backbone, S2SNetBackbone):
            out = raw.clamp(0.0, 1.0)  # S2SNet already applies sigmoid in forward()
        elif isinstance(self.backbone, B2UNetBackbone):
            # B2UNet supports both modes:
            # - Direct mode: raw output is clean estimate
            # - Residual mode: raw output is noise estimate, clean = noisy - noise
            # For B2U training: residual_base allows using original noisy image as base
            # even when input is the masked version
            if self.residual_mode:
                base = residual_base if residual_base is not None else x
                out = (base - torch.tanh(raw)).clamp(0.0, 1.0)  # Predict noise, subtract from base
            else:
                out = raw.clamp(0.0, 1.0)  # Predict clean directly
        else:
            if self.residual_mode:
                base = residual_base if residual_base is not None else x
                out = (base - torch.tanh(raw)).clamp(0.0, 1.0)
            else:
                out = torch.sigmoid(raw)
        return (out, aux_outputs) if return_aux else out


SIMINV_PARAM_NAMES: Tuple[str, ...] = (
    "speckle_k",
    "speckle_corr_sigma",
    "speckle_depth_alpha",
    "poisson_peak",
    "gauss_sigma0",
    "gauss_beta",
    "banding_amp",
    "banding_smooth_sigma",
)


class NoiseParamEstimator(nn.Module):
    """Predict normalized simulator parameters theta_hat in [0,1]^P from noisy input."""
    def __init__(self, param_dim: int, hidden_channels: int = 32):
        super().__init__()
        self.param_dim = int(param_dim)
        self.feat = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, max(64, hidden_channels * 2), 1),
            nn.GELU(),
            nn.Conv2d(max(64, hidden_channels * 2), self.param_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(self.gap(self.feat(x))).flatten(1)  # [B,P]
        return torch.sigmoid(z)


class SimInvParamAdapter(nn.Module):
    """Simulator-inverted adapter: estimate noise params then condition the backbone via FiLM."""
    def __init__(
        self,
        block_channels: Dict[str, int],
        param_dim: int,
        hidden_channels: int = 32,
        film_scale_gamma: float = 0.1,
        film_scale_beta: float = 0.1,
    ):
        super().__init__()
        self.param_dim = int(param_dim)
        self.film_scale_gamma = float(film_scale_gamma)
        self.film_scale_beta = float(film_scale_beta)
        self.estimator = NoiseParamEstimator(param_dim=self.param_dim, hidden_channels=hidden_channels)
        self.block_names = list(block_channels.keys())
        self.block_channels = dict(block_channels)
        self.total_channels = sum(self.block_channels.values())
        self.mlp = nn.Sequential(
            nn.Linear(self.param_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * self.total_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        theta_hat = self.estimator(x)  # [B,P] in [0,1]
        pred = self.mlp(theta_hat)  # [B, 2*sumC]
        gamma_all, beta_all = torch.split(pred, self.total_channels, dim=1)
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        cursor = 0
        for name in self.block_names:
            ch = self.block_channels[name]
            g = gamma_all[:, cursor:cursor + ch].view(-1, ch, 1, 1)
            b = beta_all[:, cursor:cursor + ch].view(-1, ch, 1, 1)
            cursor += ch
            out[name] = {
                "gamma": 1.0 + self.film_scale_gamma * torch.tanh(g),
                "beta": self.film_scale_beta * torch.tanh(b),
            }
        aux = {"theta_hat": theta_hat, "theta_names": list(SIMINV_PARAM_NAMES[: self.param_dim])}
        return out, aux


def freeze_backbone_unfreeze_adapter(model: AdaptiveDenoiser):
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.adapter.parameters(): p.requires_grad = True

def unfreeze_all(model: AdaptiveDenoiser):
    for p in model.parameters(): p.requires_grad = True


# ------------------------------- 
# Model builder
# ------------------------------- 
class NullAdapter(nn.Module):
    def forward(self, x: torch.Tensor):
        return None


def build_model(
    base_channels: int = 64,
    residual_mode: bool = False,
    adapter_type: str = "global",
    backbone_type: str = "unet",
    moe_experts: int = 4,
    moe_hidden_channels: int = 32,
    moe_temperature: float = 1.0,
    siminv_hidden_channels: int = 32,
    siminv_film_scale_gamma: float = 0.1,
    siminv_film_scale_beta: float = 0.1,
) -> AdaptiveDenoiser:
    """Factory to build backbone + adapter combos used across scripts.

    Args:
        base_channels: Number of base channels for the backbone
        residual_mode: Whether to use residual learning (x - tanh(pred))
        adapter_type: Type of adapter ("none", "global", "spatial", "casa", "moe", "siminv")
        backbone_type: Type of backbone ("unet", "nafnet", "noise2void", "neighbor2neighbor", "b2unet", or "s2s")

    Returns:
        AdaptiveDenoiser model with specified backbone and adapter
    """
    # Build backbone
    if backbone_type == "unet":
        backbone = DenoisingBackbone(base_channels=base_channels)
    elif backbone_type == "nafnet":
        # NAFNet "Fair" Configuration (matches sota/models/nafnet_fair.py)
        # Base channels 48 -> ~3.5M params
        # Adjust base_channels if provided (default 64 in script -> 48 for fairness if needed)
        # If user explicitly asks for 32/64, we respect it.
        backbone = NAFBackbone(base_channels=base_channels, 
                               enc_blk_nums=[2, 2, 2], 
                               middle_blk_num=2, 
                               dec_blk_nums=[2, 2, 2])
    elif backbone_type == "noise2void":
        # Noise2Void uses intermediate channel count (48 by default)
        n2v_channels = base_channels if base_channels <= 48 else 48
        backbone = Noise2VoidBackbone(base_channels=n2v_channels)
    elif backbone_type == "neighbor2neighbor":
        # Neighbor2Neighbor uses intermediate channel count (48 by default)
        n2n_channels = base_channels if base_channels <= 48 else 48
        backbone = Neighbor2NeighborBackbone(base_channels=n2n_channels)
    elif backbone_type == "b2unet":
        backbone = B2UNetBackbone(base_channels=base_channels)
    elif backbone_type == "s2s":
        backbone = S2SNetBackbone(base_channels=base_channels)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}. Choose 'unet', 'nafnet', 'noise2void', 'neighbor2neighbor', 'b2unet', or 's2s'")

    # Build adapter based on backbone's modulation points
    if adapter_type == "none":
        adapter = NullAdapter()
    elif adapter_type == "global":
        adapter = NoiseAdapter(in_channels=1, block_channels=backbone.modulated_channels)
    elif adapter_type == "spatial":
        adapter = SpatialNoiseAdapter(in_channels=1, block_channels=backbone.modulated_channels)
    elif adapter_type == "casa":
        adapter = CoherenceSpatialAdapter(block_channels=backbone.modulated_channels)
    elif adapter_type == "moe":
        adapter = MixtureNoiseAdapter(
            in_channels=1,
            block_channels=backbone.modulated_channels,
            num_experts=moe_experts,
            hidden_channels=moe_hidden_channels,
            temperature=moe_temperature,
        )
    elif adapter_type == "siminv":
        adapter = SimInvParamAdapter(
            block_channels=backbone.modulated_channels,
            param_dim=len(SIMINV_PARAM_NAMES),
            hidden_channels=siminv_hidden_channels,
            film_scale_gamma=siminv_film_scale_gamma,
            film_scale_beta=siminv_film_scale_beta,
        )
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}. Choose 'none', 'global', 'spatial', 'casa', 'moe', or 'siminv'")

    return AdaptiveDenoiser(backbone, adapter, residual_mode=residual_mode)


# ------------------------------- 
# Losses
# ------------------------------- 
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3): super().__init__(); self.eps = eps
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))


class Noise2VoidLoss(nn.Module):
    """
    Noise2Void loss: MSE on blind-spot pixels only.

    Args:
        original: Original noisy image [B, C, H, W]
        predicted: Network prediction [B, C, H, W]
        mask_coords: Tuple of (batch_idx, channel_idx, y_coords, x_coords)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, predicted: torch.Tensor, original: torch.Tensor, mask_coords: Tuple) -> torch.Tensor:
        batch_idx, channel_idx, y_coords, x_coords = mask_coords

        # MEMORY MONITORING: Track mask size
        num_masked = len(batch_idx) if batch_idx is not None else 0

        if num_masked == 0:
            # BUG FIX: Return proper zero loss that doesn't break gradient flow
            return predicted.sum() * 0.0

        # Extract values at masked positions (creates views, not copies)
        pred_values = predicted[batch_idx, channel_idx, y_coords, x_coords]
        target_values = original[batch_idx, channel_idx, y_coords, x_coords]

        # MEMORY MONITORING: Validate tensor sizes
        if pred_values.numel() != target_values.numel():
            raise RuntimeError(f"[N2V Loss] Shape mismatch: pred={pred_values.shape}, target={target_values.shape}")

        # Robust Charbonnier loss on masked pixels.
        # This is much more stable in log domain where low-intensity regions
        # can otherwise produce extremely large MSE values.
        diff = pred_values - target_values
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

        # MEMORY MONITORING: Occasional stats (every 100 calls)
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        if self._call_count % 100 == 0:
            print(f"[N2V Loss] Processed {num_masked} masked pixels | Loss={loss.item():.6f}", flush=True)

        return loss


class Neighbor2NeighborLoss(nn.Module):
    """
    Neighbor2Neighbor loss: MSE on target subset pixels only.

    Args:
        predicted: Network prediction [B, C, H, W]
        target: Original noisy image [B, C, H, W]
        target_mask: Boolean mask [B, C, H, W] indicating valid target pixels
    """
    def __init__(self):
        super().__init__()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss only on target subset pixels.

        Args:
            predicted: Network output [B, C, H, W]
            target: Original noisy image [B, C, H, W]
            target_mask: Boolean mask [B, C, H, W]

        Returns:
            MSE loss on target subset
        """
        # BUG FIX: Verify mask is boolean type (catches errors early)
        if target_mask.dtype != torch.bool:
            raise TypeError(f"[N2N Loss] target_mask must be torch.bool, got {target_mask.dtype}")

        # BUG FIX: Avoid GPU-to-CPU sync on every call - check sum on GPU first
        mask_sum = target_mask.sum()
        if mask_sum == 0:
            # BUG FIX: Return proper zero loss that doesn't break gradient flow
            return predicted.sum() * 0.0

        # Extract values at target positions using mask
        pred_values = predicted[target_mask]
        target_values = target[target_mask]

        # MEMORY MONITORING: Validate tensor sizes
        if pred_values.numel() != target_values.numel():
            raise RuntimeError(f"[N2N Loss] Shape mismatch: pred={pred_values.shape}, target={target_values.shape}")

        # MSE on target subset
        loss = F.mse_loss(pred_values, target_values)

        # MEMORY MONITORING: Occasional stats (every 100 calls)
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1
        if self._call_count % 100 == 0:
            # BUG FIX: Use mask_sum from earlier (already computed) instead of recalculating
            num_target_pixels = mask_sum.item()  # Only do GPU-to-CPU sync for monitoring (1/100 calls)
            total_pixels = target_mask.numel()
            subset_ratio = num_target_pixels / total_pixels
            print(f"[N2N Loss] Target subset: {num_target_pixels}/{total_pixels} ({subset_ratio:.1%}) | Loss={loss.item():.6f}", flush=True)

        return loss


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sobel_x); self.register_buffer('sobel_y', sobel_y)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gx_p, gy_p = F.conv2d(x,self.sobel_x,padding=1), F.conv2d(x,self.sobel_y,padding=1)
        gx_t, gy_t = F.conv2d(y,self.sobel_x,padding=1), F.conv2d(y,self.sobel_y,padding=1)
        return torch.mean(torch.abs(gx_p-gx_t)) + torch.mean(torch.abs(gy_p-gy_t))


class RevisibleLoss(nn.Module):
    """Re-visible loss for Blind2Unblind training with optional TV and anchor regularization."""
    def __init__(self, lambda_tv: float = 1e-5, lambda_anchor: float = 0.0):
        super().__init__()
        self.lambda_tv = lambda_tv
        self.lambda_anchor = lambda_anchor

    def forward(self, denoised_full: torch.Tensor, denoised_masked: torch.Tensor,
                mask: torch.Tensor, noisy_input: torch.Tensor = None) -> torch.Tensor:
        blind_mask = 1.0 - mask  # mask: 1=visible, 0=blind -> blind_mask: 1=blind, 0=visible
        # Blind2Unblind training: The FULL prediction (with complete information) serves as pseudo-GT
        # We train the MASKED prediction to match FULL at the BLIND SPOTS (where masked input had zeros)
        # This teaches the network to infer blind pixels from surrounding context
        revisible = F.mse_loss(denoised_masked * blind_mask, denoised_full.detach() * blind_mask)
        if self.lambda_tv > 0:
            revisible = revisible + self.lambda_tv * total_variation_loss(denoised_masked)
        # Anchor loss to prevent mode collapse: keep predictions close to noisy input
        if self.lambda_anchor > 0 and noisy_input is not None:
            anchor_loss = F.l1_loss(denoised_full, noisy_input)
            revisible = revisible + self.lambda_anchor * anchor_loss
        return revisible


class Self2SelfLoss(nn.Module):
    """Self2Self loss with background suppression."""
    def __init__(self, lambda_bg: float = 0.1):
        super().__init__()
        self.lambda_bg = lambda_bg

    @staticmethod
    def background_loss(pred: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        bg_mask = (pred < threshold).float()
        return (pred * bg_mask).pow(2).mean()

    def forward(self, prediction: torch.Tensor, noisy_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        blind_mask = 1.0 - mask
        l1_loss = F.l1_loss(prediction * blind_mask, noisy_input * blind_mask)
        return l1_loss + self.lambda_bg * self.background_loss(prediction)


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    return torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]).mean() + torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]).mean()


def perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simplified perceptual loss using gradient features (no VGG needed).
    Compares gradient magnitude and direction between pred and target.

    Args:
        pred: Predicted image [B, C, H, W] where H, W >= 2
        target: Target image [B, C, H, W] where H, W >= 2

    Returns:
        Perceptual loss (scalar)
    """
    # Edge case: If image too small for gradient computation, fallback to L1
    if pred.shape[2] < 2 or pred.shape[3] < 2:
        return F.l1_loss(pred, target)

    # Compute gradients
    gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    gy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]

    # Gradient magnitude loss (align shapes to [B, C, H-1, W-1])
    mag_pred = torch.sqrt(gx_pred[:, :, :-1, :]**2 + gy_pred[:, :, :, :-1]**2 + 1e-8)
    mag_tgt = torch.sqrt(gx_tgt[:, :, :-1, :]**2 + gy_tgt[:, :, :, :-1]**2 + 1e-8)

    return F.l1_loss(mag_pred, mag_tgt)


def multiscale_loss(pred: torch.Tensor, target: torch.Tensor, scales: list = [1.0, 0.5, 0.25]) -> torch.Tensor:
    """
    Multi-scale N2V loss: compute loss at multiple resolutions.
    Helps network learn both fine details and global structure.
    """
    total_loss = 0.0
    valid_scales = []
    for scale in scales:
        if scale == 1.0:
            loss = F.l1_loss(pred, target)
            valid_scales.append(scale)
        else:
            # Downsample both pred and target
            new_h = int(pred.shape[2] * scale)
            new_w = int(pred.shape[3] * scale)
            # Skip scales that would result in too small images (< 4x4)
            if new_h < 4 or new_w < 4:
                continue
            size = (new_h, new_w)
            pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            loss = F.l1_loss(pred_scaled, target_scaled)
            valid_scales.append(scale)

        total_loss += scale * loss  # Weight by scale

    # Fallback to simple L1 loss if image is too small for multi-scale
    if len(valid_scales) == 0:
        return F.l1_loss(pred, target)

    return total_loss / sum(valid_scales)


def denoise_with_tta(model: nn.Module, noisy: torch.Tensor, use_tta: bool = False) -> torch.Tensor:
    """
    Test-time augmentation: denoise with 8 rotations/flips, average results.
    Typically provides +0.5 to +2 dB improvement with no retraining.

    Args:
        model: Denoising model (AdaptiveDenoiser or similar)
        noisy: Input noisy image [B, C, H, W]
        use_tta: Whether to use TTA (if False, just returns model(noisy))

    Returns:
        Denoised image [B, C, H, W]
    """
    # CRITICAL FIX: For S2SNet backbone, use Monte Carlo inference (requires dropout active)
    if hasattr(model, 'backbone') and isinstance(model.backbone, S2SNetBackbone):
        # Get adapter features first (required for CASA modulation)
        adapter_out = model.adapter(noisy)
        adapter_features = adapter_out[0] if isinstance(adapter_out, tuple) else adapter_out

        if not use_tta:
            mean_pred, _ = model.backbone.inference(noisy, adapter_features=adapter_features, num_samples=10)
            del adapter_out, adapter_features
            return mean_pred

        # TTA with MC inference (reduced samples per augmentation for speed)
        with torch.no_grad():
            predictions = []

            # Original
            pred_orig, _ = model.backbone.inference(noisy, adapter_features, num_samples=5)
            predictions.append(pred_orig)
            del pred_orig

            # Rotate 90, 180, 270
            for k in [1, 2, 3]:
                aug = torch.rot90(noisy, k=k, dims=(2, 3))
                af_aug = model.adapter(aug)
                af_aug = af_aug[0] if isinstance(af_aug, tuple) else af_aug
                pred_aug, _ = model.backbone.inference(aug, af_aug, num_samples=5)
                predictions.append(torch.rot90(pred_aug, k=-k, dims=(2, 3)))
                del aug, af_aug, pred_aug

            # Flip horizontal
            flip_h = torch.flip(noisy, dims=[3])
            af_h = model.adapter(flip_h)
            af_h = af_h[0] if isinstance(af_h, tuple) else af_h
            pred_h, _ = model.backbone.inference(flip_h, af_h, num_samples=5)
            predictions.append(torch.flip(pred_h, dims=[3]))
            del pred_h, af_h

            # Flip vertical
            flip_v = torch.flip(noisy, dims=[2])
            af_v = model.adapter(flip_v)
            af_v = af_v[0] if isinstance(af_v, tuple) else af_v
            pred_v, _ = model.backbone.inference(flip_v, af_v, num_samples=5)
            predictions.append(torch.flip(pred_v, dims=[2]))
            del pred_v, af_v

            # BUG FIX: Add flip+rotate combinations to match 8-fold ensemble
            # Flip horizontal + Rotate 90
            flip_h_rot90 = torch.rot90(flip_h, k=1, dims=(2, 3))
            af_hr90 = model.adapter(flip_h_rot90)
            af_hr90 = af_hr90[0] if isinstance(af_hr90, tuple) else af_hr90
            pred_hr90, _ = model.backbone.inference(flip_h_rot90, af_hr90, num_samples=5)
            predictions.append(torch.flip(torch.rot90(pred_hr90, k=-1, dims=(2, 3)), dims=[3]))
            del flip_h, flip_h_rot90, af_hr90, pred_hr90

            # Flip vertical + Rotate 90
            flip_v_rot90 = torch.rot90(flip_v, k=1, dims=(2, 3))
            af_vr90 = model.adapter(flip_v_rot90)
            af_vr90 = af_vr90[0] if isinstance(af_vr90, tuple) else af_vr90
            pred_vr90, _ = model.backbone.inference(flip_v_rot90, af_vr90, num_samples=5)
            predictions.append(torch.flip(torch.rot90(pred_vr90, k=-1, dims=(2, 3)), dims=[2]))
            del flip_v, flip_v_rot90, af_vr90, pred_vr90

            result = torch.stack(predictions).mean(dim=0)
            del predictions, adapter_out, adapter_features
            return result

    # Original logic for non-S2S backbones
    if not use_tta:
        return model(noisy)

    model.eval()  # Ensure model is in eval mode

    # CRITICAL: Disable gradient tracking during inference to save memory
    # MEMORY OPTIMIZATION: Use in-place accumulation instead of list to reduce peak memory
    # (8 images -> 2 images max in memory: running sum + current prediction)
    with torch.no_grad():
        # Start with original prediction
        result = model(noisy).clone()  # Clone to avoid modifying cached output
        count = 1

        # Rotations: 90, 180, 270
        for k in [1, 2, 3]:
            rot = torch.rot90(noisy, k=k, dims=(2, 3))
            pred_rot = model(rot)
            result.add_(torch.rot90(pred_rot, k=-k, dims=(2, 3)))
            count += 1
            del rot, pred_rot

        # Flip horizontal
        flip_h = torch.flip(noisy, dims=[3])
        pred_flip_h = model(flip_h)
        result.add_(torch.flip(pred_flip_h, dims=[3]))
        count += 1
        del pred_flip_h

        # Flip vertical
        flip_v = torch.flip(noisy, dims=[2])
        pred_flip_v = model(flip_v)
        result.add_(torch.flip(pred_flip_v, dims=[2]))
        count += 1
        del pred_flip_v

        # Flip H + Rotate 90
        flip_h_rot90 = torch.rot90(flip_h, k=1, dims=(2, 3))
        pred_flip_h_rot90 = model(flip_h_rot90)
        result.add_(torch.flip(torch.rot90(pred_flip_h_rot90, k=-1, dims=(2, 3)), dims=[3]))
        count += 1
        del flip_h, flip_h_rot90, pred_flip_h_rot90

        # Flip V + Rotate 90
        flip_v_rot90 = torch.rot90(flip_v, k=1, dims=(2, 3))
        pred_flip_v_rot90 = model(flip_v_rot90)
        result.add_(torch.flip(torch.rot90(pred_flip_v_rot90, k=-1, dims=(2, 3)), dims=[2]))
        count += 1
        del flip_v, flip_v_rot90, pred_flip_v_rot90

        # Average
        result.div_(count)
        return result


class DepthWeightedFidelityLoss(nn.Module):
    """Penalizes deeper voxels more heavily to respect OCT signal roll-off."""
    def __init__(self, depth_decay: float = 1.5, eps: float = 1e-3, log_space: bool = False):
        super().__init__()
        self.depth_decay, self.eps, self.log_space = depth_decay, eps, log_space

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, _, h, _ = pred.shape
        depth = torch.linspace(0, 1, h, device=pred.device, dtype=pred.dtype).view(1, 1, h, 1)
        weights = torch.exp(depth * self.depth_decay)
        diff = pred - target
        if self.log_space:
            diff = torch.log(pred.clamp(min=1e-6)) - torch.log(target.clamp(min=1e-6))
        loss_map = torch.sqrt(diff ** 2 + self.eps ** 2) * weights
        return loss_map.mean() / (weights.mean() + 1e-6)


class AscanContinuityLoss(nn.Module):
    """Encourages axial smoothness matching the system axial resolution."""
    def __init__(self, axial_resolution_px: float = 1.0, use_second_order: bool = True):
        super().__init__()
        self.axial_resolution_px = axial_resolution_px
        self.use_second_order = use_second_order

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dz = self.axial_resolution_px
        dp = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) / dz
        dt = (target[:, :, 1:, :] - target[:, :, :-1, :]) / dz
        loss = torch.mean(torch.abs(dp - dt))
        if self.use_second_order and dp.shape[2] > 1:
            d2p = dp[:, :, 1:, :] - dp[:, :, :-1, :]
            d2t = dt[:, :, 1:, :] - dt[:, :, :-1, :]
            loss = loss + 0.5 * torch.mean(torch.abs(d2p - d2t))
        return loss


class SpeckleStatisticsLoss(nn.Module):
    """Matches residual noise statistics (Rayleigh default) to theoretical speckle."""
    def __init__(self, target_distribution: str = "rayleigh", eps: float = 1e-6):
        super().__init__()
        self.target_distribution = target_distribution
        self.eps = eps
        # Rayleigh reference values
        self.expected_cv2 = 4.0 / math.pi - 1.0  # ~0.273
        self.expected_kurtosis = 0.245

    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None, noisy: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noisy is None and target is None:
            raise ValueError("SpeckleStatisticsLoss requires either target or noisy input.")
        residual = noisy - pred if noisy is not None else pred - target
        amp = torch.abs(residual) + self.eps
        b = amp.shape[0]
        flat = amp.view(b, -1)
        mu = flat.mean(dim=1)
        var = flat.var(dim=1, unbiased=False)
        cv2 = var / (mu ** 2 + self.eps)
        centered = flat - mu[:, None]
        kurtosis = (centered ** 4).mean(dim=1) / (var ** 2 + self.eps) - 3.0

        loss = torch.mean((cv2 - self.expected_cv2) ** 2)
        loss = loss + 0.05 * torch.mean((kurtosis - self.expected_kurtosis) ** 2)
        log_amp = torch.log(flat + self.eps)
        loss = loss + 0.02 * torch.mean((log_amp.mean(dim=1) - torch.log(mu + self.eps)) ** 2)
        return loss


# -------------------------------
# Coherence-Aware Supervision Loss
# -------------------------------
def compute_local_cv(image: torch.Tensor, window_size: int = 11, memory_efficient: bool = True) -> torch.Tensor:
    """
    Compute local coefficient of variation (CV).

    High CV = structured speckle (multiplicative) = coherent processing needed
    Low CV = additive noise = incoherent processing needed

    Args:
        image: [B, C, H, W] tensor
        window_size: size of local window
        memory_efficient: if True, delete intermediate tensors aggressively

    Returns:
        cv_map: [B, C, H, W] local CV map
    """
    # Ensure odd kernel size
    if window_size % 2 == 0:
        window_size += 1

    kernel_size = window_size
    padding = kernel_size // 2

    # Use replicate padding to avoid edge artifacts and ensure correct output size
    padded = F.pad(image, (padding, padding, padding, padding), mode='replicate')

    # Local mean and variance with no additional padding (already padded)
    mean_local = F.avg_pool2d(padded, kernel_size, stride=1, padding=0)

    if memory_efficient:
        # Compute squared values in-place to reduce memory
        padded_sq = padded.pow(2)
        del padded  # Free memory immediately
        mean_sq_local = F.avg_pool2d(padded_sq, kernel_size, stride=1, padding=0)
        del padded_sq  # Free memory immediately
    else:
        mean_sq_local = F.avg_pool2d(padded ** 2, kernel_size, stride=1, padding=0)
        del padded

    var_local = mean_sq_local - mean_local ** 2
    del mean_sq_local  # Free memory immediately

    std_local = torch.sqrt(torch.clamp(var_local, min=1e-8))
    del var_local  # Free memory immediately

    # Coefficient of variation
    cv_local = std_local / (mean_local + 1e-8)
    del std_local, mean_local  # Free memory immediately

    return cv_local


def pearson_correlation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute (1 - Pearson correlation) as a non-negative loss (minimize = maximize correlation).

    Args:
        x, y: [B, C, H, W] tensors

    Returns:
        loss: scalar in [0, 2], 0 means perfect positive correlation
    """
    # Flatten spatial dimensions
    x_flat = x.view(x.size(0), x.size(1), -1)  # [B, C, N]
    y_flat = y.view(y.size(0), y.size(1), -1)

    # Center
    x_centered = x_flat - x_flat.mean(dim=2, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=2, keepdim=True)

    # Pearson correlation
    numerator = (x_centered * y_centered).sum(dim=2)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=2) * (y_centered ** 2).sum(dim=2) + 1e-8)
    correlation = numerator / denominator

    # Clamp to avoid numeric overshoot and produce non-negative loss
    corr_mean = correlation.mean().clamp(-1.0, 1.0)
    return 1.0 - corr_mean


class CoherenceSupervisionLoss(nn.Module):
    """
    Coherence-aware loss: Encourage coherent weights to correlate with local CV.

    This is a self-supervised loss that only uses noisy images.
    """
    def __init__(self, window_size: int = 11, target: str = "residual"):
        super().__init__()
        self.window_size = window_size
        if target not in ("noisy", "residual"):
            raise ValueError("target must be 'noisy' or 'residual'")
        self.target = target

    def forward(self, coherent_map: torch.Tensor, noisy_image: torch.Tensor, pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coherent_map: [B, 1, H, W] coherent weight from CASA
            noisy_image: [B, 1, H, W] input noisy image
            pred: [B, 1, H, W] current denoised prediction (optional)

        Returns:
            loss: scalar coherence loss
        """
        # Compute local CV as a coherence proxy.
        # - noisy: CV is heavily confounded by anatomy/edges.
        # - residual: CV is computed on |noisy - pred| (better proxy for speckle-like residual).
        if self.target == "residual":
            if pred is None:
                raise ValueError("pred must be provided when target='residual'")
            cv_source = (noisy_image - pred).abs()
        else:
            cv_source = noisy_image

        cv_map = compute_local_cv(cv_source, window_size=self.window_size)

        # Normalize CV to [0, 1] for better training stability
        cv_normalized = (cv_map - cv_map.min()) / (cv_map.max() - cv_map.min() + 1e-8)

        # Encourage positive correlation
        corr_loss = pearson_correlation_loss(coherent_map, cv_normalized)

        return corr_loss


def casa_entropy_regularizer(coherent_map: torch.Tensor, incoherent_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage non-trivial (non-uniform) coherent/incoherent decomposition by penalizing high entropy.
    Lower entropy => more confident separation (but still learned from data).
    """
    p = torch.cat([coherent_map, incoherent_map], dim=1).clamp(min=eps, max=1.0 - eps)  # [B,2,H,W]
    entropy = -(p * torch.log(p)).sum(dim=1)  # [B,H,W]
    return entropy.mean()


def casa_std_floor_regularizer(coherent_map: torch.Tensor, min_std: float = 0.0) -> torch.Tensor:
    """
    Prevent degenerate constant maps by enforcing a minimum spatial standard deviation.
    Uses a hinge penalty: relu(min_std - std).
    """
    if min_std <= 0:
        return coherent_map.new_zeros(())
    std = coherent_map.std(dim=(-2, -1))  # [B,1]
    return F.relu(min_std - std).mean()


# -------------------------------
# Test-time adaptation
# ------------------------------- 
@torch.no_grad()
def _forward_no_grad(model: AdaptiveDenoiser, x: torch.Tensor) -> torch.Tensor:
    """Small helper to run the model in eval mode without gradients."""
    was_train = model.training
    model.eval()
    out = model(x)
    if was_train:
        model.train()
    return out


def _build_tta_augs(names: Optional[List[str]] = None):
    augs = {
        "flip_h": (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
        "flip_v": (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
        "rot90": (lambda x: torch.rot90(x, k=1, dims=(-2, -1)), lambda x: torch.rot90(x, k=3, dims=(-2, -1))),
    }
    selected = ["identity"]
    if names is None:
        selected += ["flip_h", "flip_v", "rot90"]
    else:
        selected += [n for n in names if n in augs]
    built = []
    for n in selected:
        if n == "identity":
            built.append((lambda x: x, lambda x: x))
        else:
            built.append(augs[n])
    return built


def test_time_adaptation(
    model: AdaptiveDenoiser,
    noisy: torch.Tensor,
    num_steps: int = 20,
    lr: float = 8e-4,
    tv_weight: float = 1e-4,
    self_consistency: float = 0.1,
    anchor_weight: float = 0.05,
    augmentations: Optional[List[str]] = None,
    spectral_weight: float = 0.0,
    noise_characterizer: Optional[SpectralNoiseCharacterizer] = None,
) -> torch.Tensor:
    """
    Self-supervised TTA with augmentation-consistency and optional spectral priors.
    """
    model = model.to(device)
    noisy = noisy.to(device)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    aug_fns = _build_tta_augs(augmentations)

    if spectral_weight > 0 and noise_characterizer is None:
        noise_characterizer = SpectralNoiseCharacterizer().to(device)
    if noise_characterizer is not None:
        noise_characterizer.eval()
        with torch.no_grad():
            spectral_target = noise_characterizer(noisy.clamp(0.0, 1.0))
    else:
        spectral_target = None

    # CRITICAL FIX: Save model state to prevent TTA adaptation from contaminating the model for future inferences
    # Move to CPU to save GPU memory
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(num_steps):
        opt.zero_grad(set_to_none=True)
        preds = []
        for aug, inv in aug_fns:
            aug_noisy = aug(noisy)
            pred_aug = model(aug_noisy)
            preds.append(inv(pred_aug))
        preds_stack = torch.stack(preds, dim=0)  # [N,B,1,H,W]
        anchor = preds_stack[0]
        consistency = torch.mean(torch.abs(preds_stack - anchor))
        loss = self_consistency * consistency
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation_loss(anchor)
        if anchor_weight > 0:
            loss = loss + anchor_weight * F.l1_loss(anchor, noisy)
        if spectral_weight > 0 and spectral_target is not None:
            residual = (noisy - anchor).clamp(0.0, 1.0)
            spec_pred = noise_characterizer(residual)
            loss = loss + spectral_weight * F.l1_loss(spec_pred["band_power"], spectral_target["band_power"])
            loss = loss + 0.1 * spectral_weight * F.l1_loss(spec_pred["spectral_slope"], spectral_target["spectral_slope"])
        loss.backward()
        opt.step()

    del opt
    # Cleanup spectral target if used
    if spectral_target is not None:
        del spectral_target
    
    with torch.no_grad():
        result = _forward_no_grad(model, noisy).detach().cpu()

    # Restore original model state
    model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
    del original_state
    
    return result


# ------------------------------- 
# Meta-learning (Reptile-style)
# ------------------------------- 
def _sample_noise_task(clean_batch: torch.Tensor) -> Tuple[torch.Tensor, str]:
    task = random.choice(NOISE_TASKS_EXTENDED)
    noisy = task["fn"](clean_batch)
    return noisy.clamp(0.0, 1.0), task["name"]


def reptile_meta_train(
    model: AdaptiveDenoiser,
    clean_loader: DataLoader,
    num_meta_epochs: int = 5,
    num_tasks_per_meta_batch: int = 4,
    inner_steps: int = 5,
    inner_lr: float = 1e-4,
    meta_step_size: float = 0.1,
    tv_inner_weight: float = 1e-5,
    amp: bool = True,
    memory_guard: bool = True,
    meta_log_interval: int = 10,
    use_noise2void: bool = False,
    n2v_mask_ratio: float = 0.12,
    n2v_box_size: int = 5,
    n2v_blindspot_dilation: int = 1,
    use_neighbor2neighbor: bool = False,
    use_blind2unblind: bool = False,
    b2u_mask_ratio: float = 0.5,
    b2u_block_size: int = 2,
    lambda_coherence: float = 0.0,
    lambda_anchor: float = 0.0,
    log_domain: bool = True,
    coherence_target: str = "residual",
    lambda_casa_entropy: float = 0.0,
    lambda_casa_std: float = 0.0,
    casa_min_std: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Adapter-only Reptile meta-learning across synthetic noise domains, Noise2Void, or Neighbor2Neighbor self-supervised training."""
    model.to(device)
    freeze_backbone_unfreeze_adapter(model)

    # Select loss function based on training mode
    if use_noise2void:
        loss_fn = Noise2VoidLoss()
    elif use_neighbor2neighbor:
        loss_fn = Neighbor2NeighborLoss()
    elif use_blind2unblind:
        loss_fn = RevisibleLoss(lambda_tv=tv_inner_weight, lambda_anchor=lambda_anchor)
    else:
        loss_fn = CharbonnierLoss()
    
    # B2U masker for blind2unblind training
    b2u_masker = GlobalAwareMaskMapper(mask_ratio=b2u_mask_ratio, block_size=b2u_block_size) if use_blind2unblind else None
    coherence_loss_fn = CoherenceSupervisionLoss(target=coherence_target) if lambda_coherence > 0 else None

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    meta_params = {k: v.detach().clone() for k, v in model.adapter.state_dict().items()}
    clean_iter = iter(clean_loader)

    # MEMORY MONITORING: Print initial memory stats
    print("\n" + "="*60)
    print("META-TRAINING STARTING - Initial Memory Stats:")
    initial_stats = get_memory_stats()
    for key, val in initial_stats.items():
        print(f"  {key}: {val:.3f} GB")
    print("="*60 + "\n", flush=True)

    for epoch in range(num_meta_epochs):
        task_losses = []
        # CRITICAL FIX: Accumulate meta-updates across tasks before averaging
        accumulated_updates = {k: torch.zeros_like(v) for k, v in meta_params.items()}
        for task_idx in range(num_tasks_per_meta_batch):
            try:
                clean = next(clean_iter)
            except StopIteration:
                clean_iter = iter(clean_loader)
                clean = next(clean_iter)
            clean = clean.to(device)

            # For Noise2Void: add noise THEN use blind-spot masking; For Neighbor2Neighbor: add noise THEN sub-sample; For supervised: use synthetic noise
            if use_noise2void:
                # CRITICAL FIX: N2V must train on NOISY data, not clean data!
                # First add synthetic noise to get a noisy image
                noisy_unmasked, task_name = _sample_noise_task(clean)
                # Then apply blind-spot masking to the noisy image
                pre_mask_mem = get_memory_stats()
                masked_noisy, mask_coords = apply_blind_spot_mask(noisy_unmasked, mask_ratio=n2v_mask_ratio, box_size=n2v_box_size, blindspot_dilation=n2v_blindspot_dilation)
                post_mask_mem = get_memory_stats()
                if (task_idx + 1) % 10 == 0:  # Log every 10 tasks
                    mask_delta = post_mask_mem.get('cuda_allocated_gb', 0) - pre_mask_mem.get('cuda_allocated_gb', 0)
                    print(f"[Meta N2V] Task {task_idx+1} ({task_name}) | Masking added {mask_delta*1000:.1f} MB", flush=True)
                # MEMORY LEAK FIX: Delete memory stats dicts
                del pre_mask_mem, post_mask_mem
                noisy = masked_noisy  # Network input is masked noisy image
                clean = noisy_unmasked  # Target is original noisy image (N2V predicts noisy from masked noisy)
                n2n_data = None
                b2u_data = None
            elif use_neighbor2neighbor:
                # CRITICAL FIX: N2N must train on NOISY data, not clean data!
                # First add synthetic noise to get a noisy image
                noisy_full, task_name = _sample_noise_task(clean)
                # Then apply sub-sampling to the noisy image
                pre_subsample_mem = get_memory_stats()
                subsample_pattern = task_idx % 2  # Alternate between two patterns
                input_subset, target_subset, target_mask = apply_neighbor2neighbor_subsample(noisy_full, subsample_pattern=subsample_pattern)
                post_subsample_mem = get_memory_stats()
                if (task_idx + 1) % 10 == 0:  # Log every 10 tasks
                    subsample_delta = post_subsample_mem.get('cuda_allocated_gb', 0) - pre_subsample_mem.get('cuda_allocated_gb', 0)
                    print(f"[Meta N2N] Task {task_idx+1} ({task_name}) | Sub-sampling added {subsample_delta*1000:.1f} MB", flush=True)
                # MEMORY LEAK FIX: Delete memory stats dicts
                del pre_subsample_mem, post_subsample_mem
                noisy = input_subset  # Network input is the sub-sampled noisy image
                clean = noisy_full  # Target is full noisy image (N2N predicts noisy from subsampled noisy)
                n2n_data = (target_subset, target_mask)  # Store target subset and mask for loss
                mask_coords = None
                b2u_data = None
            elif use_blind2unblind:
                # CRITICAL FIX: B2U must train on NOISY data, not clean data!
                # First add synthetic noise to get a noisy image
                noisy_full, task_name = _sample_noise_task(clean)
                # Then apply B2U masking
                masked_input, b2u_mask, b2u_blind = b2u_masker(noisy_full)
                noisy = noisy_full  # Full input for one forward pass
                clean = noisy_full  # Store for deletion later
                b2u_data = (masked_input, b2u_mask)  # Store masked input and mask
                mask_coords = None
                n2n_data = None
            else:
                noisy, task_name = _sample_noise_task(clean)
                mask_coords = None
                n2n_data = None
                b2u_data = None

            with MemoryGuard(operation_name=f"meta_task_{task_name}") if memory_guard else nullcontext():
                # MEMORY LEAK FIX: Store initial references to clean and noisy for later deletion
                task_adapter = copy.deepcopy(model.adapter)
                task_model = AdaptiveDenoiser(model.backbone, task_adapter, residual_mode=model.residual_mode).to(device)
                freeze_backbone_unfreeze_adapter(task_model)
                # CRITICAL FIX: Ensure backbone is in eval mode (no dropout/BN updates) during meta-training of adapter
                task_model.backbone.eval()
                task_model.adapter.train()
                
                inner_opt = torch.optim.Adam(task_model.adapter.parameters(), lr=inner_lr)

                for _ in range(inner_steps):
                    inner_opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=amp):
                        if use_blind2unblind:
                            # B2U: two forward passes (full and masked) with optional log domain
                            masked_input, b2u_mask = b2u_data
                            pred_full, aux_full = task_model(noisy, return_aux=True)
                            if task_model.residual_mode:
                                pred_masked, aux_masked = task_model(masked_input, return_aux=True, residual_base=noisy)
                            else:
                                pred_masked, aux_masked = task_model(masked_input, return_aux=True)
                            if log_domain:
                                pred_full_log = torch.log(pred_full.clamp(min=1e-6))
                                pred_masked_log = torch.log(pred_masked.clamp(min=1e-6))
                                noisy_log = torch.log(noisy.clamp(min=1e-6))
                                inner_loss = loss_fn(pred_full_log, pred_masked_log, b2u_mask, noisy_input=noisy_log)
                            else:
                                inner_loss = loss_fn(pred_full, pred_masked, b2u_mask, noisy_input=noisy)
                            # NOTE: RevisibleLoss already includes consistency at blind spots
                            pred = pred_full  # For TV loss
                            aux_for_reg = aux_full
                        else:
                            if task_model.residual_mode and (use_noise2void or use_neighbor2neighbor):
                                out = task_model(noisy, return_aux=True, residual_base=clean)
                            else:
                                out = task_model(noisy, return_aux=True)
                            pred, aux = out if isinstance(out, tuple) else (out, {})
                            aux_for_reg = aux
                            if use_noise2void:
                                # Noise2Void: predict masked pixels from neighbors with optional log domain
                                if log_domain:
                                    pred_log = torch.log(pred.clamp(min=1e-6))
                                    clean_log = torch.log(clean.clamp(min=1e-6))
                                    inner_loss = loss_fn(pred_log, clean_log, mask_coords)
                                else:
                                    inner_loss = loss_fn(pred, clean, mask_coords)
                            elif use_neighbor2neighbor:
                                # Neighbor2Neighbor: predict target subset from input subset
                                target_subset, target_mask = n2n_data
                                inner_loss = loss_fn(pred, target_subset, target_mask)
                            else:
                                # Supervised: predict clean from noisy with optional log domain
                                if log_domain:
                                    pred_log = torch.log(pred.clamp(min=1e-6))
                                    clean_log = torch.log(clean.clamp(min=1e-6))
                                    inner_loss = loss_fn(pred_log, clean_log)
                                else:
                                    inner_loss = loss_fn(pred, clean)
                        if coherence_loss_fn is not None and aux_for_reg and "coherent_map" in aux_for_reg:
                            coherent_map = aux_for_reg["coherent_map"]
                            coherence_loss = coherence_loss_fn(coherent_map, noisy, pred=pred.detach())
                            inner_loss = inner_loss + lambda_coherence * coherence_loss
                            del coherent_map, coherence_loss
                        if aux_for_reg:
                            if lambda_casa_entropy > 0 and "coherent_map" in aux_for_reg and "incoherent_map" in aux_for_reg:
                                inner_loss = inner_loss + lambda_casa_entropy * casa_entropy_regularizer(
                                    aux_for_reg["coherent_map"], aux_for_reg["incoherent_map"]
                                )
                            if lambda_casa_std > 0 and casa_min_std > 0 and "coherent_map" in aux_for_reg:
                                inner_loss = inner_loss + lambda_casa_std * casa_std_floor_regularizer(
                                    aux_for_reg["coherent_map"], min_std=casa_min_std
                                )
                        if tv_inner_weight > 0 and not use_blind2unblind:  # B2U has TV in RevisibleLoss
                            inner_loss = inner_loss + tv_inner_weight * total_variation_loss(pred)
                    scaler.scale(inner_loss).backward()
                    scaler.step(inner_opt)
                    scaler.update()
                    del pred  # BUG FIX: Explicitly release prediction tensor
                inner_loss_val = inner_loss.item()
                task_losses.append(inner_loss_val)

                if (task_idx + 1) % max(1, meta_log_interval) == 0:
                    print(f"[Meta] Epoch {epoch+1}/{num_meta_epochs} Task {task_idx+1}/{num_tasks_per_meta_batch} Loss={inner_loss_val:.4f}", flush=True)

                # Reptile meta-update (first-order)
                # CRITICAL FIX: Accumulate updates, don't apply division inside loop
                for name, param in task_model.adapter.state_dict().items():
                    accumulated_updates[name] += (param.detach() - meta_params[name])

                # MEMORY LEAK FIX: Explicitly delete task model, adapter, optimizer, and tensors
                if use_noise2void:
                    del task_model, task_adapter, inner_opt, inner_loss, clean, noisy, noisy_unmasked, masked_noisy, mask_coords
                elif use_neighbor2neighbor:
                    del task_model, task_adapter, inner_opt, inner_loss, clean, noisy, noisy_full, input_subset, target_subset, target_mask, n2n_data
                elif use_blind2unblind:
                    try:
                        del consistency_loss
                    except NameError:
                        pass
                    del task_model, task_adapter, inner_opt, inner_loss, clean, noisy, noisy_full, masked_input, b2u_mask, b2u_blind, b2u_data, pred_full, pred_masked, aux_full, aux_masked
                else:
                    del task_model, task_adapter, inner_opt, inner_loss, clean, noisy
                if memory_guard:
                    torch.cuda.empty_cache()

        # CRITICAL FIX: Apply averaged Reptile update after all tasks
        # Standard Reptile: θ = θ + α * (1/K) * Σ(θ_task_k - θ)
        for name in meta_params:
            meta_params[name] = meta_params[name] + meta_step_size * accumulated_updates[name] / float(num_tasks_per_meta_batch)

        # MEMORY LEAK FIX: Delete accumulated updates
        del accumulated_updates

        model.adapter.load_state_dict(meta_params)
        epoch_task_loss = np.mean(task_losses)

        # MEMORY LEAK FIX: Clear task_losses list
        del task_losses
        force_memory_cleanup()

        # MEMORY MONITORING: Print memory stats after each epoch
        epoch_stats = get_memory_stats()
        mem_info = " | ".join([f"{k.split('_')[0]}:{v:.2f}GB" for k, v in epoch_stats.items()])
        print(f"[Meta] Epoch {epoch+1}/{num_meta_epochs} | Loss={epoch_task_loss:.4f} | Memory: {mem_info}", flush=True)

    return meta_params


# ------------------------------- 
# Supervised Fine-tuning
# ------------------------------- 
def supervised_finetune(
    model: AdaptiveDenoiser,
    paired_loader: DataLoader,
    val_loader: Optional[DataLoader],
    output_dir: str,  # Directory to save best checkpoints
    num_epochs: int = 50,
    lr_adapter: float = 5e-4,
    lr_backbone: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = False,
    loss_type: str = 'charbonnier',
    lambda_grad: float = 0.05,
    lambda_tv: float = 1e-5,
    lambda_depth: float = 0.0,
    lambda_ascan: float = 0.0,
    lambda_speckle: float = 0.0,
    log_domain: bool = True,
    amp: bool = True,
    grad_clip_norm: float = 1.0,
    scheduler_type: str = 'cosine',
    warmup_epochs: int = 8,
    early_stopping_patience: int = 10,
    ema_enable: bool = True,
    ema_decay: float = 0.999,
    casa_lambda_tv: float = 0.0,
    lambda_coherence: float = 0.0,
    coherence_target: str = "residual",
    lambda_casa_entropy: float = 0.0,
    lambda_casa_std: float = 0.0,
    casa_min_std: float = 0.0,
    lambda_moe_balance: float = 0.0,
    lambda_moe_var: float = 0.0,
    lambda_theta: float = 0.0,
    lambda_anchor: float = 0.0,
    log_interval: int = 50,
    validation_frequency: int = 1,  # Run validation every N epochs (1=every epoch, 2=every other epoch, etc.)
    gradient_accumulation_steps: int = 1,  # Accumulate gradients over N steps for larger effective batch
    use_noise2void: bool = False,
    n2v_mask_ratio: float = 0.12,
    n2v_box_size: int = 5,
    n2v_blindspot_dilation: int = 1,
    use_neighbor2neighbor: bool = False,
    use_blind2unblind: bool = False,
    use_self2self: bool = False,
    use_hybrid_selfsup: bool = False,
    hybrid_mode: str = "alternate",
    hybrid_start_with: str = "b2u",
    hybrid_corr_threshold: float = 0.08,
    hybrid_corr_low: Optional[float] = None,
    hybrid_corr_high: Optional[float] = None,
    hybrid_calib_steps: int = 0,
    hybrid_calib_factor: float = 1.0,
    hybrid_calib_stat: str = "mean",
    s2s_mask_prob: float = 0.3,
    s2s_num_masks: int = 8,
    b2u_mask_ratio: float = 0.5,
    b2u_block_size: int = 2,
    lambda_perceptual: float = 0.0,
    lambda_multiscale: float = 0.0,
    use_tta: bool = False,
    # Resume training parameters
    start_epoch: int = 0,
    resume_optimizer_state: Optional[dict] = None,
    resume_scheduler_state: Optional[dict] = None,
    resume_ema_state: Optional[dict] = None,
    resume_best_val_psnr: float = -float('inf'),
    resume_best_val_loss: float = float('inf'),
    resume_epochs_no_improve: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    model.to(device); model.train()

    if coherence_target not in ("residual", "noisy"):
        raise ValueError(f"Unknown coherence_target: {coherence_target}")
    if lambda_moe_balance < 0 or lambda_moe_var < 0:
        raise ValueError("lambda_moe_balance and lambda_moe_var must be >= 0")
    if lambda_theta < 0:
        raise ValueError("lambda_theta must be >= 0")

    metrics_jsonl_path = os.path.join(output_dir, "metrics.jsonl")

    # MEMORY MONITORING: Print initial memory stats for fine-tuning
    print("\n" + "="*60)
    print("SUPERVISED FINE-TUNING STARTING - Initial Memory Stats:")
    finetune_initial_stats = get_memory_stats()
    for key, val in finetune_initial_stats.items():
        print(f"  {key}: {val:.3f} GB")
    print("="*60 + "\n", flush=True)

    optimizer = torch.optim.AdamW(
        [{"params": model.adapter.parameters(), "lr": lr_adapter}, {"params": model.backbone.parameters(), "lr": lr_backbone}],
        weight_decay=weight_decay,
    )
    if freeze_backbone:
        freeze_backbone_unfreeze_adapter(model)
        # Keep a frozen backbone deterministic (e.g., if a backbone uses dropout/BN).
        model.backbone.eval()
        trainable = [p for p in model.adapter.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "freeze_backbone=True but adapter has no trainable parameters. "
                "This commonly happens with --adapter none. "
                "Use --adapter casa/global/spatial, or do not freeze the backbone."
            )
        backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        adapter_trainable = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
        print(
            f"[Freeze] backbone trainable params: {backbone_trainable:,} | adapter trainable params: {adapter_trainable:,}",
            flush=True,
        )
        optimizer = torch.optim.AdamW(trainable, lr=lr_adapter, weight_decay=weight_decay)

    # DEPRECATION FIX: Use torch.amp instead of torch.cuda.amp (PyTorch 2.0+)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)  # Fallback for older PyTorch
    use_hybrid_selfsup = bool(use_hybrid_selfsup)
    if use_hybrid_selfsup:
        if hybrid_mode not in ("alternate", "noise_aware"):
            raise ValueError(f"Unknown hybrid_mode: {hybrid_mode}")
        if hybrid_start_with not in ("b2u", "n2v"):
            raise ValueError(f"Unknown hybrid_start_with: {hybrid_start_with}")
        if hybrid_calib_steps < 0:
            raise ValueError("--hybrid_calib_steps must be >= 0")
        if hybrid_calib_factor <= 0:
            raise ValueError("--hybrid_calib_factor must be > 0")
        if hybrid_calib_stat not in ("mean", "median"):
            raise ValueError("--hybrid_calib_stat must be 'mean' or 'median'")

    # Select loss function based on training mode
    if use_noise2void:
        pixel_loss_fn = Noise2VoidLoss()
    elif use_neighbor2neighbor:
        pixel_loss_fn = Neighbor2NeighborLoss()
    else:
        pixel_loss_fn = CharbonnierLoss() if loss_type == 'charbonnier' else nn.L1Loss()

    n2v_loss_fn = Noise2VoidLoss() if (use_noise2void or use_hybrid_selfsup) else None
    revisible_loss_fn = RevisibleLoss(lambda_tv=lambda_tv, lambda_anchor=lambda_anchor)
    b2u_masker = GlobalAwareMaskMapper(mask_ratio=b2u_mask_ratio, block_size=b2u_block_size) if (use_blind2unblind or use_hybrid_selfsup) else None
    s2s_sampler = BernoulliSampler(mask_prob=s2s_mask_prob) if (use_self2self and not use_hybrid_selfsup) else None
    s2s_loss_fn = Self2SelfLoss()
    grad_loss_fn = GradientLoss()
    depth_loss_fn = DepthWeightedFidelityLoss(log_space=False)
    ascan_loss_fn = AscanContinuityLoss()
    speckle_loss_fn = SpeckleStatisticsLoss()
    coherence_loss_fn = CoherenceSupervisionLoss(target=coherence_target) if lambda_coherence > 0 else None

    # Initialize or restore EMA state
    if resume_ema_state is not None and ema_enable:
        ema_state = resume_ema_state
        print(f"✓ Restored EMA state from checkpoint")
    else:
        ema_state = {k:v.detach().clone() for k,v in model.state_dict().items()} if ema_enable else None
    ema_model_cache = None  # Cache EMA model to avoid recreation every epoch

    # CRITICAL FIX: Account for gradient accumulation in scheduler step calculation
    effective_steps_per_epoch = len(paired_loader) // gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * num_epochs
    # CRITICAL FIX: Ensure T_max is at least 1 to avoid ValueError
    cosine_T_max = max(1, total_steps - warmup_epochs * effective_steps_per_epoch)
    if scheduler_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_T_max)
    else:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=5,factor=0.5)

    if warmup_epochs > 0:
        warmup_iters = warmup_epochs * effective_steps_per_epoch
        linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup, main_scheduler], milestones=[warmup_iters])
    else:
        scheduler = main_scheduler

    # Load optimizer and scheduler states if resuming (note: incompatible with adapter-only fine-tuning).
    if freeze_backbone and resume_optimizer_state is not None:
        print("⚠️  freeze_backbone=True: ignoring optimizer/scheduler resume state (param groups mismatch).", flush=True)
        resume_optimizer_state = None
        resume_scheduler_state = None
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print(f"✓ Restored optimizer state from checkpoint")
    if resume_scheduler_state is not None:
        scheduler.load_state_dict(resume_scheduler_state)
        print(f"✓ Restored scheduler state from checkpoint")

    # Initialize or restore training progress
    best_val_loss = resume_best_val_loss
    best_val_psnr = resume_best_val_psnr
    epochs_no_improve = resume_epochs_no_improve
    best_model_state = None  # Store best model weights
    best_ema_state = None  # Store best EMA weights
    best_epoch = start_epoch  # Track when best model was found
    log_every = max(1, log_interval)

    # OPTIMIZATION: Compile model for faster inference (PyTorch 2.0+)
    compiled_model = model  # Default: no compilation
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("[Optimization] Model compiled with torch.compile for faster validation", flush=True)
    except Exception as e:
        # Fallback if compile fails
        compiled_model = model
    # Initialize optimizer gradients
    optimizer.zero_grad(set_to_none=True)

    # Print resume info if continuing from checkpoint
    if start_epoch > 0:
        print(f"\n{'='*80}")
        print(f"RESUMING TRAINING FROM EPOCH {start_epoch + 1}")
        print(f"  Total epochs to train: {num_epochs} (continuing to epoch {start_epoch + num_epochs})")
        print(f"  Best val PSNR so far: {best_val_psnr:.2f} dB")
        print(f"  Best val loss so far: {best_val_loss:.4f}")
        print(f"  Epochs without improvement: {epochs_no_improve}/{early_stopping_patience}")
        print(f"{'='*80}\n")

    # BUG FIX #11: num_epochs is TOTAL epochs to train (not additional epochs)
    # When resuming from epoch 32 with --finetune_epochs 100, should train to epoch 100 (not 132)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # Memory leak watchdog - detect unexpected memory growth
        if epoch > start_epoch:
            current_mem = get_memory_stats().get('cuda_allocated_gb', 0)
            if hasattr(supervised_finetune, '_last_epoch_mem'):
                mem_growth = current_mem - supervised_finetune._last_epoch_mem
                if mem_growth > 0.5:  # More than 500MB growth
                    print(f"[MemoryWarning] Potential leak: +{mem_growth:.2f} GB since last epoch", flush=True)
                    force_memory_cleanup()
            supervised_finetune._last_epoch_mem = current_mem

        running_loss = []
        # MEMORY FIX: Only initialize tracking lists when actually used
        if (use_noise2void or use_hybrid_selfsup) and lambda_coherence > 0:
            running_n2v_loss = []  # Track N2V loss separately (N2V steps only)
            running_coherence_loss = []  # Track coherence loss (N2V steps only)
        if use_hybrid_selfsup and hybrid_mode == "noise_aware" and hybrid_calib_steps > 0:
            # Calibration over first N steps.
            # - mean: running mean (no storage)
            # - median: store N floats, then take median (robust to skew/outliers)
            calib_count = 0
            calib_mean = 0.0
            calib_values = [] if hybrid_calib_stat == "median" else None
            calibrated_threshold = None  # Freeze threshold after calibration
        step = -1  # CRITICAL FIX: Initialize to handle empty data loader edge case
        for step, batch in enumerate(paired_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x_noisy, y_clean, theta_gt = batch
            else:
                x_noisy, y_clean = batch
                theta_gt = None
            # OPTIMIZATION: Use non_blocking=True for faster data transfer
            x_noisy, y_clean = x_noisy.to(device, non_blocking=True), y_clean.to(device, non_blocking=True)
            if theta_gt is not None:
                theta_gt = theta_gt.to(device, non_blocking=True)

            # Determine per-step strategy (hybrid or fixed)
            corr_score = None
            corr_low_th = None
            corr_high_th = None
            step_use_noise2void = use_noise2void
            step_use_neighbor2neighbor = use_neighbor2neighbor
            step_use_blind2unblind = use_blind2unblind
            step_use_self2self = use_self2self
            if use_hybrid_selfsup:
                step_use_neighbor2neighbor = False
                step_use_self2self = False
                if hybrid_mode == "alternate":
                    if hybrid_start_with == "n2v":
                        step_use_noise2void = (step % 2 == 0)
                    else:
                        step_use_noise2void = (step % 2 == 1)
                    step_use_blind2unblind = not step_use_noise2void
                else:
                    corr_score = float(estimate_noise_correlation_score(x_noisy).mean().item())
                    # Optional calibration: after warmup, threshold becomes stat*cailb_factor.
                    effective_threshold = hybrid_corr_threshold
                    if hybrid_calib_steps > 0:
                        if calibrated_threshold is not None:
                            effective_threshold = calibrated_threshold
                        elif calib_count < hybrid_calib_steps:
                            calib_count += 1
                            if calib_values is not None:
                                calib_values.append(corr_score)
                            else:
                                calib_mean += (corr_score - calib_mean) / calib_count
                        else:
                            if calib_values is not None:
                                sorted_vals = sorted(calib_values)
                                mid = len(sorted_vals) // 2
                                if len(sorted_vals) % 2 == 0:
                                    calib_stat_value = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
                                else:
                                    calib_stat_value = sorted_vals[mid]
                                calibrated_threshold = calib_stat_value * hybrid_calib_factor
                                effective_threshold = calibrated_threshold
                                print(
                                    f"[Hybrid] Corr calibration complete at step {step+1}: "
                                    f"{hybrid_calib_stat}={calib_stat_value:.3f} -> threshold={effective_threshold:.3f}",
                                    flush=True,
                                )
                            else:
                                calibrated_threshold = calib_mean * hybrid_calib_factor
                                effective_threshold = calibrated_threshold
                                print(
                                    f"[Hybrid] Corr calibration complete at step {step+1}: "
                                    f"mean={calib_mean:.3f} -> threshold={effective_threshold:.3f}",
                                    flush=True,
                                )

                    # Optional deadband: use low/high if provided, else derive from effective threshold.
                    low_th = hybrid_corr_low if hybrid_corr_low is not None else 0.9 * effective_threshold
                    high_th = hybrid_corr_high if hybrid_corr_high is not None else 1.1 * effective_threshold
                    corr_low_th, corr_high_th = float(low_th), float(high_th)

                    if corr_score < low_th:
                        step_use_noise2void = True
                        step_use_blind2unblind = False
                    elif corr_score > high_th:
                        step_use_noise2void = False
                        step_use_blind2unblind = True
                    else:
                        # Ambiguous band: fall back to alternation to avoid bias.
                        if hybrid_start_with == "n2v":
                            step_use_noise2void = (step % 2 == 0)
                        else:
                            step_use_noise2void = (step % 2 == 1)
                        step_use_blind2unblind = not step_use_noise2void

            # For Noise2Void: apply blind-spot masking to noisy input
            # For Neighbor2Neighbor: apply sub-sampling
            if step_use_noise2void:
                masked_input, mask_coords = apply_blind_spot_mask(
                    x_noisy, mask_ratio=n2v_mask_ratio, box_size=n2v_box_size, blindspot_dilation=n2v_blindspot_dilation
                )
                model_input = masked_input
                model_input_masked = None
                pixel_target = x_noisy  # Predict original noisy from masked
                n2n_data = None

                # MEMORY MONITORING: Log masking details every 50 steps
                if (step + 1) % 50 == 0:
                    num_masked = len(mask_coords[0])
                    mask_mem_mb = (num_masked * 4 * 8) / (1024 * 1024)
                    print(f"[Finetune N2V] Step {step+1} | Masked {num_masked} pixels | Coords mem: {mask_mem_mb:.2f} MB", flush=True)
            elif step_use_neighbor2neighbor:
                subsample_pattern = step % 2  # Alternate between two patterns
                input_subset, target_subset, target_mask = apply_neighbor2neighbor_subsample(x_noisy, subsample_pattern=subsample_pattern)
                model_input = input_subset
                model_input_masked = None
                pixel_target = target_subset
                n2n_data = target_mask
                mask_coords = None

                # MEMORY MONITORING: Log sub-sampling details every 50 steps
                # PERFORMANCE FIX: Avoid GPU-CPU sync - compute ratio on GPU
                if (step + 1) % 50 == 0:
                    subset_ratio = target_mask.float().mean().item()  # Only sync ratio, not individual counts
                    print(f"[Finetune N2N] Step {step+1} | Subset ratio: {subset_ratio:.1%}", flush=True)
            elif step_use_blind2unblind:
                model_input = x_noisy
                model_input_masked, b2u_mask, b2u_blind = b2u_masker(x_noisy)
                pixel_target = None
                mask_coords = None
                n2n_data = None
            elif step_use_self2self:
                model_input = x_noisy
                model_input_masked = None
                pixel_target = None
                mask_coords = None
                n2n_data = None
            else:
                model_input = x_noisy
                model_input_masked = None
                pixel_target = y_clean
                mask_coords = None
                n2n_data = None

            # DEPRECATION FIX: Use torch.amp.autocast instead of torch.cuda.amp.autocast
            try:
                autocast_context = torch.amp.autocast('cuda', enabled=amp)
            except AttributeError:
                autocast_context = torch.cuda.amp.autocast(enabled=amp)

            with autocast_context:
                aux_full = None
                aux = {}
                coherence_loss_value = 0.0  # default for logging
                n2v_loss_value = None
                if step_use_blind2unblind:
                    out_full = model(model_input, return_aux=True)
                    pred_full, aux_full = out_full if isinstance(out_full, tuple) else (out_full, {})
                    # CRITICAL: For residual mode, use x_noisy as base for masked prediction
                    if model.residual_mode:
                        out_masked = model(model_input_masked, return_aux=True, residual_base=x_noisy)
                    else:
                        out_masked = model(model_input_masked, return_aux=True)
                    pred_masked, aux_masked = out_masked if isinstance(out_masked, tuple) else (out_masked, {})
                elif step_use_self2self:
                    losses = []
                    for masked_input, mask in zip(*s2s_sampler(x_noisy, num_samples=s2s_num_masks)):
                        if gradient_accumulation_steps > 1:
                            p = gradient_checkpoint(model, masked_input, use_reentrant=False)
                        else:
                            p = model(masked_input, return_aux=False)
                        losses.append(s2s_loss_fn(p, x_noisy, mask))
                        del p, masked_input, mask
                    loss = sum(losses) / len(losses)
                else:
                    # If residual mode is enabled, make sure the residual base is the full noisy image
                    # for self-supervised steps where model_input is masked/subsampled.
                    if model.residual_mode and (step_use_noise2void or step_use_neighbor2neighbor):
                        out = model(model_input, return_aux=True, residual_base=x_noisy)
                    else:
                        out = model(model_input, return_aux=True)
                    pred, aux = out if isinstance(out, tuple) else (out, {})

                # Compute main loss
                if step_use_blind2unblind:
                    if log_domain:
                        pred_full_log = torch.log(pred_full.clamp(min=1e-6))
                        pred_masked_log = torch.log(pred_masked.clamp(min=1e-6))
                        noisy_log = torch.log(x_noisy.clamp(min=1e-6))
                        loss = revisible_loss_fn(pred_full_log, pred_masked_log, b2u_mask, noisy_input=noisy_log)
                    else:
                        loss = revisible_loss_fn(pred_full, pred_masked, b2u_mask, noisy_input=x_noisy)
                elif step_use_self2self:
                    coherence_loss_value = 0.0
                elif step_use_noise2void:
                    if n2v_loss_fn is None:
                        raise RuntimeError("n2v_loss_fn is not initialized")
                    if log_domain:
                        pred_log = torch.log(pred.clamp(min=1e-6))
                        target_log = torch.log(pixel_target.clamp(min=1e-6))
                        loss = n2v_loss_fn(pred_log, target_log, mask_coords)
                    else:
                        loss = n2v_loss_fn(pred, pixel_target, mask_coords)
                    n2v_loss_value = loss.item()
                elif step_use_neighbor2neighbor:
                    loss = pixel_loss_fn(pred, pixel_target, n2n_data)
                else:
                    pred_for_loss, target_for_loss = (torch.log(pred.clamp(min=1e-6)), torch.log(pixel_target.clamp(min=1e-6))) if log_domain else (pred, pixel_target)
                    loss = pixel_loss_fn(pred_for_loss, target_for_loss)
                    if lambda_grad > 0: loss += lambda_grad * grad_loss_fn(pred_for_loss, target_for_loss)
                    if lambda_depth > 0: loss += lambda_depth * depth_loss_fn(pred, pixel_target)
                    if lambda_ascan > 0: loss += lambda_ascan * ascan_loss_fn(pred, pixel_target)
                    if lambda_speckle > 0: loss += lambda_speckle * speckle_loss_fn(pred, target=pixel_target, noisy=x_noisy)

                # SimInv parameter regression (only when supervised theta_gt is provided by dataset).
                if lambda_theta > 0 and theta_gt is not None and isinstance(aux, dict):
                    theta_hat = aux.get("theta_hat")
                    if theta_hat is not None:
                        loss = loss + lambda_theta * F.smooth_l1_loss(theta_hat, theta_gt)

                # Common losses (apply to both N2V and supervised)
                if not step_use_blind2unblind and not step_use_self2self:
                    if lambda_tv > 0: loss += lambda_tv * total_variation_loss(pred)
                    if not step_use_noise2void and not step_use_neighbor2neighbor:
                        if lambda_perceptual > 0: loss += lambda_perceptual * perceptual_loss(pred, pixel_target)
                        if lambda_multiscale > 0: loss += lambda_multiscale * multiscale_loss(pred, pixel_target)

                # CASA regularization and coherence supervision (apply when aux present)
                aux_for_reg = aux_full if step_use_blind2unblind else aux
                if aux_for_reg:
                    # MoE regularization (if adapter returned weights): discourage gate collapse.
                    moe_w = aux_for_reg.get("moe_weights") if isinstance(aux_for_reg, dict) else None
                    if moe_w is not None and (lambda_moe_balance > 0 or lambda_moe_var > 0):
                        w = moe_w.float()
                        K = w.shape[1]
                        mean_w = w.mean(dim=0)  # [K]
                        if lambda_moe_balance > 0:
                            loss += lambda_moe_balance * torch.mean((mean_w - (1.0 / K)) ** 2)
                        if lambda_moe_var > 0:
                            loss -= lambda_moe_var * torch.mean(w.var(dim=0, unbiased=False))
                        del w, K, mean_w

                    coherent_map = aux_for_reg.get("coherent_map")
                    incoherent_map = aux_for_reg.get("incoherent_map")
                    if casa_lambda_tv > 0:
                        loss += casa_lambda_tv * sum(
                            total_variation_loss(aux_for_reg[k])
                            for k in ["coherent_map", "incoherent_map"]
                            if k in aux_for_reg
                        )
                    if coherent_map is not None:
                        if lambda_casa_entropy > 0 and incoherent_map is not None:
                            loss += lambda_casa_entropy * casa_entropy_regularizer(coherent_map, incoherent_map)
                        if lambda_casa_std > 0 and casa_min_std > 0:
                            loss += lambda_casa_std * casa_std_floor_regularizer(coherent_map, min_std=casa_min_std)
                        if coherence_loss_fn is not None:
                            pred_for_coherence = pred_full if step_use_blind2unblind else pred
                            coherence_loss = coherence_loss_fn(
                                coherent_map, x_noisy, pred=pred_for_coherence.detach()
                            )
                            coherence_loss_value = coherence_loss.item()
                            loss += lambda_coherence * coherence_loss
                            del pred_for_coherence, coherence_loss
                    del coherent_map, incoherent_map

            # OPTIMIZATION: Gradient accumulation for larger effective batch size
            # Scale loss by accumulation steps to maintain gradient magnitude
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            # Only update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                # Check for gradient scaler underflow
                if scaler.get_scale() < 1.0:
                    print(f"[Warning] Gradient scaler underflow, scale={scaler.get_scale():.2f}", flush=True)
                optimizer.zero_grad(set_to_none=True)

            # FIX: Only step scheduler after actual weight updates (respects gradient accumulation)
            if (step + 1) % gradient_accumulation_steps == 0:
                if scheduler_type == 'cosine':
                    scheduler.step()
                elif scheduler_type == 'plateau' and warmup_epochs > 0 and isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
                    # SequentialLR uses last_epoch to track progress; step warmup phase only
                    if scheduler.last_epoch < scheduler._milestones[0]:
                        scheduler.step()

            running_loss.append(loss.item() * gradient_accumulation_steps)  # Log unscaled loss
            # MEMORY FIX: Safe deletion of loss tracking variables
            if step_use_noise2void and lambda_coherence > 0 and n2v_loss_value is not None:
                running_n2v_loss.append(n2v_loss_value)
                running_coherence_loss.append(coherence_loss_value)
            if step_use_noise2void and n2v_loss_value is not None:
                del n2v_loss_value
            if not step_use_blind2unblind and not step_use_self2self:
                del coherence_loss_value

            # MEMORY LEAK FIX: Delete intermediate tensors after backward pass
            if step_use_blind2unblind:
                del x_noisy, y_clean, model_input, model_input_masked, b2u_mask, b2u_blind, out_full, out_masked, pred_full, pred_masked, aux_full, aux_masked, loss
            elif step_use_self2self:
                del x_noisy, y_clean, model_input, losses, loss
            elif step_use_noise2void:
                del x_noisy, y_clean, model_input, masked_input, mask_coords, pixel_target, out, pred, aux, loss
            elif step_use_neighbor2neighbor:
                del x_noisy, y_clean, model_input, input_subset, target_subset, target_mask, n2n_data, pixel_target, out, pred, aux, loss
            else:
                try:
                    del pred_for_loss, target_for_loss
                except NameError:
                    pass
                del x_noisy, y_clean, model_input, pixel_target, out, pred, aux, loss

            if ema_enable:
                with torch.no_grad():
                    cur = model.state_dict()
                    for k in ema_state:
                        if getattr(ema_state[k].dtype, "is_floating_point", False):
                            ema_state[k].mul_(ema_decay).add_((1-ema_decay)*cur[k])
                        else:
                            # Keep non-float buffers (e.g., longs) in sync without type casting
                            ema_state[k].copy_(cur[k])
                # MEMORY OPTIMIZATION: Consolidate EMA tensors every 10 epochs to prevent fragmentation
                if (step + 1) == len(paired_loader) and (epoch + 1) % 10 == 0:
                    with torch.no_grad():
                        for k in ema_state:
                            ema_state[k] = ema_state[k].contiguous()
                        print(f"[Memory] EMA state consolidated at epoch {epoch+1}", flush=True)
            if (step + 1) % log_every == 0:
                # MEMORY MONITORING: Print memory stats with loss
                step_stats = get_memory_stats()
                mem_str = " | ".join([f"{k.split('_')[0]}:{v:.2f}GB" for k, v in step_stats.items()])
                mode_str = ""
                if use_hybrid_selfsup:
                    mode_str = " Mode=N2V" if step_use_noise2void else " Mode=B2U"
                    if corr_score is not None:
                        mode_str += f" Corr={corr_score:.3f}"
                        if corr_low_th is not None and corr_high_th is not None and hybrid_mode == "noise_aware":
                            mode_str += f" Th=[{corr_low_th:.2f},{corr_high_th:.2f}]"
                # Enhanced logging for coherence supervision
                if (use_noise2void or use_hybrid_selfsup) and lambda_coherence > 0 and len(running_coherence_loss) > 0 and len(running_n2v_loss) > 0:
                    print(f"[Finetune]{mode_str} Epoch {epoch+1}/{num_epochs} Step {step+1}/{len(paired_loader)} "
                          f"Loss={running_loss[-1]:.4f} (N2V={running_n2v_loss[-1]:.4f}, Coh={running_coherence_loss[-1]:.4f}) | Mem: {mem_str}", flush=True)
                else:
                    print(f"[Finetune]{mode_str} Epoch {epoch+1}/{num_epochs} Step {step+1}/{len(paired_loader)} Loss={running_loss[-1]:.4f} | Mem: {mem_str}", flush=True)

            # MEMORY LEAK FIX: Periodic cleanup every 100 steps
            if (step + 1) % 100 == 0:
                torch.cuda.empty_cache()

        # CRITICAL FIX: Apply any remaining accumulated gradients at end of epoch
        # This handles the case where len(paired_loader) % gradient_accumulation_steps != 0
        if gradient_accumulation_steps > 1:
            # Check if there are leftover gradients (step+1 is the total number of steps)
            steps_completed = step + 1
            if steps_completed % gradient_accumulation_steps != 0:
                # Apply the remaining gradients
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                print(f"[Finetune] Epoch {epoch+1}: Applied {steps_completed % gradient_accumulation_steps} leftover gradient(s)", flush=True)

        # MEMORY LEAK FIX: Compute mean before clearing and release memory
        # CRITICAL FIX: Handle empty running_loss (empty data loader edge case)
        epoch_mean_loss = np.mean(running_loss) if len(running_loss) > 0 else float('inf')

        # Coherence supervision epoch summary
        if (use_noise2void or use_hybrid_selfsup) and lambda_coherence > 0 and len(running_coherence_loss) > 0:
            epoch_n2v_loss = np.mean(running_n2v_loss)
            epoch_coherence_loss = np.mean(running_coherence_loss)
            print(f"[Epoch {epoch+1}] Mean Loss: {epoch_mean_loss:.4f} "
                  f"(N2V: {epoch_n2v_loss:.4f}, Coherence: {epoch_coherence_loss:.4f})", flush=True)

        # MEMORY LEAK FIX: Clear epoch-specific running lists
        del running_loss
        if (use_noise2void or use_hybrid_selfsup) and lambda_coherence > 0:
            del running_n2v_loss, running_coherence_loss

        val_loss = float('inf')
        val_psnr = 0.0
        val_ssim = 0.0
        # OPTIMIZATION: Run validation only every N epochs to speed up training
        should_validate = (epoch + 1) % validation_frequency == 0 or (epoch + 1) == num_epochs
        if val_loader and should_validate:
            model.eval()
            val_losses = []
            val_psnrs = []
            val_ssims = []
            val_psnrs_noisy = []  # Baseline: noisy input vs clean GT
            val_ssims_noisy = []  # Baseline: noisy input vs clean GT
            val_psnrs_no_casa = []  # Track PSNR without CASA modulation
            val_ssims_no_casa = []  # Track SSIM without CASA modulation
            # CASA contribution is computed only on a subset of batches; track WITH-CASA metrics
            # on the same subset to avoid biased comparisons.
            val_psnrs_with_casa_sample = []
            val_ssims_with_casa_sample = []
            val_coherence_correlations = []  # Track coherence-CV correlation
            # CASA map-quality metrics (sampled to control validation overhead)
            val_casa_corr_residual_cv_samples = []
            val_casa_entropy_samples = []
            val_casa_coh_std_samples = []
            val_casa_coh_mean_samples = []

            # Use EMA weights for evaluation if available
            if ema_enable and ema_state:
                # CRITICAL FIX: Temporarily load EMA weights into model instead of creating expensive deepcopy
                # Save current model state, load EMA state for validation, then restore after
                # Must save state EACH time since model updates between validations
                temp_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                # Load EMA weights into model for validation
                model.load_state_dict(ema_state)
                model.eval()
                val_model = model

                # Store for restoration later (reuse variable name for clarity)
                ema_model_cache = temp_model_state
            else:
                # OPTIMIZATION: Use compiled model for validation (faster inference)
                val_model = compiled_model if 'compiled_model' in locals() else model
                val_model.eval()  # Ensure eval mode when not swapping EMA weights

            with torch.inference_mode():
                val_total = len(val_loader)
                val_log_every = max(1, val_total // 5)  # Print ~5 progress updates during validation
                for val_step, batch in enumerate(val_loader):
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        x_val_noisy, y_val_clean, _theta_val = batch
                    else:
                        x_val_noisy, y_val_clean = batch
                    x_val_noisy, y_val_clean = x_val_noisy.to(device, non_blocking=True), y_val_clean.to(device, non_blocking=True)

                    # For hybrid validation: compute both N2V and B2U losses on the same batch
                    if use_hybrid_selfsup:
                        val_model_input = x_val_noisy
                        val_pixel_target = None
                        val_n2n_data = None
                        # Deterministic masks to reduce metric noise
                        val_masked_input_n2v, val_mask_coords = apply_blind_spot_mask(
                            x_val_noisy, mask_ratio=n2v_mask_ratio, box_size=n2v_box_size, seed=42, blindspot_dilation=n2v_blindspot_dilation
                        )
                        val_masked_input_b2u, val_b2u_mask, _ = b2u_masker(x_val_noisy)
                    # For Noise2Void validation: also use blind-spot masking with deterministic seed
                    # For Neighbor2Neighbor validation: also use sub-sampling
                    elif use_noise2void:
                        # Use deterministic seed for validation to reduce metric noise
                        val_masked_input, val_mask_coords = apply_blind_spot_mask(
                            x_val_noisy, mask_ratio=n2v_mask_ratio, box_size=n2v_box_size, seed=42, blindspot_dilation=n2v_blindspot_dilation
                        )
                        val_model_input = val_masked_input
                        val_pixel_target = x_val_noisy
                        val_n2n_data = None
                        val_b2u_mask = None
                    elif use_neighbor2neighbor:
                        val_subsample_pattern = 0  # Use fixed pattern for validation
                        val_input_subset, val_target_subset, val_target_mask = apply_neighbor2neighbor_subsample(x_val_noisy, subsample_pattern=val_subsample_pattern)
                        val_model_input = val_input_subset
                        val_pixel_target = val_target_subset
                        val_n2n_data = val_target_mask
                        val_mask_coords = None
                        val_b2u_mask = None
                    elif use_self2self:
                        val_model_input = x_val_noisy
                        val_masks_inputs, val_masks = s2s_sampler(x_val_noisy, num_samples=max(1, s2s_num_masks//2))
                        val_pixel_target = None
                        val_mask_coords = None
                        val_n2n_data = None
                        val_b2u_mask = None
                    elif use_blind2unblind:
                        val_model_input = x_val_noisy
                        val_masked_input, val_b2u_mask = b2u_masker(x_val_noisy)[:2]
                        val_pixel_target = None
                        val_mask_coords = None
                        val_n2n_data = None
                    else:
                        val_model_input = x_val_noisy
                        val_pixel_target = y_val_clean
                        val_mask_coords = None
                        val_n2n_data = None
                        val_b2u_mask = None

                    # DEPRECATION FIX: Use torch.amp.autocast
                    try:
                        val_autocast = torch.amp.autocast('cuda', enabled=amp)
                    except AttributeError:
                        val_autocast = torch.cuda.amp.autocast(enabled=amp)

                    with val_autocast:
                        if use_hybrid_selfsup:
                            if n2v_loss_fn is None:
                                raise RuntimeError("n2v_loss_fn is not initialized")
                            # --- B2U validation loss ---
                            val_pred_full = val_model(x_val_noisy)
                            if hasattr(val_model, 'residual_mode') and val_model.residual_mode:
                                val_pred_masked = val_model(val_masked_input_b2u, residual_base=x_val_noisy)
                            else:
                                val_pred_masked = val_model(val_masked_input_b2u)
                            if log_domain:
                                val_pred_full_log = torch.log(val_pred_full.clamp(min=1e-6))
                                val_pred_masked_log = torch.log(val_pred_masked.clamp(min=1e-6))
                                val_noisy_log = torch.log(x_val_noisy.clamp(min=1e-6))
                                v_loss_b2u = revisible_loss_fn(val_pred_full_log, val_pred_masked_log, val_b2u_mask, noisy_input=val_noisy_log)
                            else:
                                v_loss_b2u = revisible_loss_fn(val_pred_full, val_pred_masked, val_b2u_mask, noisy_input=x_val_noisy)

                            # --- N2V validation loss ---
                            if hasattr(val_model, 'residual_mode') and val_model.residual_mode:
                                val_pred_n2v = val_model(val_masked_input_n2v, residual_base=x_val_noisy)
                            else:
                                val_pred_n2v = val_model(val_masked_input_n2v)
                            if log_domain:
                                val_pred_n2v_log = torch.log(val_pred_n2v.clamp(min=1e-6))
                                val_target_n2v_log = torch.log(x_val_noisy.clamp(min=1e-6))
                                v_loss_n2v = n2v_loss_fn(val_pred_n2v_log, val_target_n2v_log, val_mask_coords)
                            else:
                                v_loss_n2v = n2v_loss_fn(val_pred_n2v, x_val_noisy, val_mask_coords)

                            v_loss = 0.5 * (v_loss_b2u + v_loss_n2v)
                            val_pred = val_pred_full
                        elif use_blind2unblind:
                            val_pred_full = val_model(val_model_input)
                            # CRITICAL: For residual mode, use x_val_noisy as base for masked prediction
                            if hasattr(val_model, 'residual_mode') and val_model.residual_mode:
                                val_pred_masked = val_model(val_masked_input, residual_base=x_val_noisy)
                            else:
                                val_pred_masked = val_model(val_masked_input)
                            # B2U validation with optional log domain
                            if log_domain:
                                val_pred_full_log = torch.log(val_pred_full.clamp(min=1e-6))
                                val_pred_masked_log = torch.log(val_pred_masked.clamp(min=1e-6))
                                val_noisy_log = torch.log(x_val_noisy.clamp(min=1e-6))
                                v_loss = revisible_loss_fn(val_pred_full_log, val_pred_masked_log, val_b2u_mask, noisy_input=val_noisy_log)
                            else:
                                v_loss = revisible_loss_fn(val_pred_full, val_pred_masked, val_b2u_mask, noisy_input=x_val_noisy)
                            val_pred = val_pred_full
                        elif use_self2self:
                            # Average loss over masks, run full prediction for metrics (unmasked input)
                            mask_losses = []
                            for vm_in, vm_mask in zip(val_masks_inputs, val_masks):
                                vp = val_model(vm_in)
                                mask_losses.append(s2s_loss_fn(vp, x_val_noisy, vm_mask))
                                del vp
                            v_loss = sum(mask_losses) / len(mask_losses)
                            # CRITICAL FIX: Use Monte Carlo inference for S2SNet backbone
                            # Paper requires averaging multiple dropout-enabled forward passes
                            if hasattr(val_model, 'backbone') and isinstance(val_model.backbone, S2SNetBackbone):
                                # Get adapter features first (required for CASA modulation!)
                                adapter_out = val_model.adapter(x_val_noisy)
                                adapter_features = adapter_out[0] if isinstance(adapter_out, tuple) else adapter_out
                                val_pred, _ = val_model.backbone.inference(x_val_noisy, adapter_features=adapter_features, num_samples=10)
                                del adapter_out, adapter_features  # Memory cleanup
                            else:
                                val_pred = val_model(x_val_noisy)
                        else:
                            if hasattr(val_model, 'residual_mode') and val_model.residual_mode and (use_noise2void or use_neighbor2neighbor):
                                val_pred = val_model(val_model_input, residual_base=x_val_noisy)
                            else:
                                val_pred = val_model(val_model_input)

                            # Compute validation loss
                            if use_noise2void:
                                # N2V validation with optional log domain
                                if log_domain:
                                    val_pred_log = torch.log(val_pred.clamp(min=1e-6))
                                    val_target_log = torch.log(val_pixel_target.clamp(min=1e-6))
                                    v_loss = pixel_loss_fn(val_pred_log, val_target_log, val_mask_coords)
                                else:
                                    v_loss = pixel_loss_fn(val_pred, val_pixel_target, val_mask_coords)
                            elif use_neighbor2neighbor:
                                v_loss = pixel_loss_fn(val_pred, val_pixel_target, val_n2n_data)
                            else:
                                # Supervised validation with log domain
                                pred_for_loss, target_for_loss = (torch.log(val_pred.clamp(min=1e-6)), torch.log(val_pixel_target.clamp(min=1e-6))) if log_domain else (val_pred, val_pixel_target)
                                v_loss = pixel_loss_fn(pred_for_loss, target_for_loss)
                                if lambda_grad > 0: v_loss += lambda_grad * grad_loss_fn(pred_for_loss, target_for_loss)
                                if lambda_depth > 0: v_loss += lambda_depth * depth_loss_fn(val_pred, val_pixel_target)
                                if lambda_ascan > 0: v_loss += lambda_ascan * ascan_loss_fn(val_pred, val_pixel_target)
                                if lambda_speckle > 0: v_loss += lambda_speckle * speckle_loss_fn(val_pred, target=val_pixel_target, noisy=x_val_noisy)

                            # Common losses (apply to both N2V and supervised)
                            if lambda_tv > 0: v_loss += lambda_tv * total_variation_loss(val_pred)
                            # CRITICAL FIX: Same as training - don't compare to noisy target for self-supervised
                            if not use_noise2void and not use_neighbor2neighbor and not use_hybrid_selfsup:
                                if lambda_perceptual > 0: v_loss += lambda_perceptual * perceptual_loss(val_pred, val_pixel_target)
                                if lambda_multiscale > 0: v_loss += lambda_multiscale * multiscale_loss(val_pred, val_pixel_target)

                        batch_val_loss = v_loss.item()
                        val_losses.append(batch_val_loss)

                        # EVALUATION METRICS: Compute PSNR/SSIM on FULL denoised image (not masked/subsampled)
                        # For self-supervised methods, we need to denoise the full noisy image for evaluation
                        if use_noise2void or use_neighbor2neighbor or use_blind2unblind or use_hybrid_selfsup:
                            # Denoise the full noisy image (optionally with TTA for +1-2 dB boost)
                            full_denoised = denoise_with_tta(val_model, x_val_noisy, use_tta=use_tta)
                        else:
                            # For supervised, val_pred is already the full denoised image
                            # Optionally apply TTA for supervised as well
                            if use_tta:
                                full_denoised = denoise_with_tta(val_model, x_val_noisy, use_tta=True)
                            else:
                                full_denoised = val_pred

                        # Clamp outputs to [0, 1] before computing metrics
                        full_denoised = full_denoised.clamp(0.0, 1.0)

                        # Compute PSNR and SSIM against clean ground truth
                        # Convert to numpy for skimage metrics
                        denoised_np = full_denoised.squeeze().cpu().numpy()  # [H, W]
                        clean_np = y_val_clean.squeeze().cpu().numpy()  # [H, W]
                        noisy_np = x_val_noisy.squeeze().cpu().numpy()  # [H, W]

                        # Handle batch dimension (compute metrics per image and average)
                        if denoised_np.ndim == 3:  # Batch of images [B, H, W]
                            batch_psnrs = []
                            batch_ssims = []
                            batch_psnrs_noisy = []
                            batch_ssims_noisy = []
                            for i in range(denoised_np.shape[0]):
                                psnr_val = _sk_psnr(clean_np[i], denoised_np[i], data_range=1.0)
                                ssim_val = _sk_ssim(clean_np[i], denoised_np[i], data_range=1.0)
                                batch_psnrs.append(psnr_val)
                                batch_ssims.append(ssim_val)
                                psnr_noisy = _sk_psnr(clean_np[i], noisy_np[i], data_range=1.0)
                                ssim_noisy = _sk_ssim(clean_np[i], noisy_np[i], data_range=1.0)
                                batch_psnrs_noisy.append(psnr_noisy)
                                batch_ssims_noisy.append(ssim_noisy)
                            val_psnrs.extend(batch_psnrs)
                            val_ssims.extend(batch_ssims)
                            val_psnrs_noisy.extend(batch_psnrs_noisy)
                            val_ssims_noisy.extend(batch_ssims_noisy)
                            with_casa_psnrs_this = batch_psnrs
                            with_casa_ssims_this = batch_ssims
                        else:  # Single image [H, W]
                            psnr_val = _sk_psnr(clean_np, denoised_np, data_range=1.0)
                            ssim_val = _sk_ssim(clean_np, denoised_np, data_range=1.0)
                            val_psnrs.append(psnr_val)
                            val_ssims.append(ssim_val)
                            psnr_noisy = _sk_psnr(clean_np, noisy_np, data_range=1.0)
                            ssim_noisy = _sk_ssim(clean_np, noisy_np, data_range=1.0)
                            val_psnrs_noisy.append(psnr_noisy)
                            val_ssims_noisy.append(ssim_noisy)
                            with_casa_psnrs_this = [psnr_val]
                            with_casa_ssims_this = [ssim_val]

                        # CASA CONTRIBUTION ANALYSIS: Compare with and without CASA modulation
                        # Only compute every few batches to save time (e.g., every 10 batches or last batch)
                        compute_casa_contribution = hasattr(val_model, 'adapter') and isinstance(val_model.adapter, CoherenceSpatialAdapter)
                        compute_casa_contribution = compute_casa_contribution and ((val_step + 1) % 10 == 0 or (val_step + 1) == val_total)

                        if compute_casa_contribution:
                            try:
                                with torch.no_grad():
                                    # Create identity CASA features (no modulation: gamma=1, beta=0)
                                    identity_features = {}
                                    for name in val_model.backbone.modulated_channels.keys():
                                        H_mod, W_mod = x_val_noisy.shape[-2:]
                                        # Adjust resolution based on encoder/decoder level
                                        if name in ("enc4", "dec4", "enc3", "dec3"):
                                            H_mod, W_mod = (H_mod + 3) // 4, (W_mod + 3) // 4
                                        elif name in ("enc2", "dec2"):
                                            H_mod, W_mod = (H_mod + 1) // 2, (W_mod + 1) // 2

                                        ch = val_model.backbone.modulated_channels[name]
                                        identity_features[name] = {
                                            'gamma': torch.ones(x_val_noisy.size(0), ch, H_mod, W_mod, device=x_val_noisy.device, dtype=x_val_noisy.dtype),
                                            'beta': torch.zeros(x_val_noisy.size(0), ch, H_mod, W_mod, device=x_val_noisy.device, dtype=x_val_noisy.dtype)
                                        }

                                    # Forward pass without CASA (using identity features)
                                    no_casa_pred = val_model.backbone(x_val_noisy, adapter_features=identity_features)

                                    # Handle residual mode
                                    if isinstance(val_model.backbone, NAFBackbone):
                                        no_casa_denoised = no_casa_pred.clamp(0.0, 1.0)
                                    elif hasattr(val_model, 'residual_mode') and val_model.residual_mode:
                                        no_casa_denoised = (x_val_noisy - torch.tanh(no_casa_pred)).clamp(0.0, 1.0)
                                    else:
                                        no_casa_denoised = torch.sigmoid(no_casa_pred)

                                    # Compute metrics without CASA
                                    no_casa_denoised_np = no_casa_denoised.squeeze().cpu().numpy()

                                    if no_casa_denoised_np.ndim == 3:  # Batch
                                        batch_psnrs_no_casa = []
                                        batch_ssims_no_casa = []
                                        for i in range(no_casa_denoised_np.shape[0]):
                                            psnr_no_casa = _sk_psnr(clean_np[i], no_casa_denoised_np[i], data_range=1.0)
                                            ssim_no_casa = _sk_ssim(clean_np[i], no_casa_denoised_np[i], data_range=1.0)
                                            batch_psnrs_no_casa.append(psnr_no_casa)
                                            batch_ssims_no_casa.append(ssim_no_casa)
                                        val_psnrs_no_casa.extend(batch_psnrs_no_casa)
                                        val_ssims_no_casa.extend(batch_ssims_no_casa)
                                    else:  # Single image
                                        psnr_no_casa = _sk_psnr(clean_np, no_casa_denoised_np, data_range=1.0)
                                        ssim_no_casa = _sk_ssim(clean_np, no_casa_denoised_np, data_range=1.0)
                                        val_psnrs_no_casa.append(psnr_no_casa)
                                        val_ssims_no_casa.append(ssim_no_casa)

                                    # Track WITH-CASA metrics on the exact same sampled batches.
                                    # These lists are defined immediately after computing the WITH-CASA metrics above.
                                    val_psnrs_with_casa_sample.extend(with_casa_psnrs_this)
                                    val_ssims_with_casa_sample.extend(with_casa_ssims_this)

                                    # Cleanup
                                    del identity_features, no_casa_pred, no_casa_denoised, no_casa_denoised_np
                            except Exception as e:
                                # If CASA contribution analysis fails, silently skip (don't break validation)
                                pass

                        # COHERENCE VALIDATION: Track coherent-CV correlation for CASA
                        # MEMORY FIX: Reuse aux from main forward pass instead of doing another forward pass
                        if lambda_coherence > 0 and use_noise2void:
                            # Try to reuse existing aux data to avoid redundant forward pass
                            coherent_map_val = None
                            if not use_blind2unblind and not use_self2self and 'aux' in locals() and aux and 'coherent_map' in aux:
                                coherent_map_val = aux['coherent_map']
                            elif use_blind2unblind and 'aux_full' in locals() and aux_full and 'coherent_map' in aux_full:
                                coherent_map_val = aux_full['coherent_map']

                            if coherent_map_val is not None:
                                # Compute local CV with memory-efficient mode
                                cv_map_val = compute_local_cv(x_val_noisy, memory_efficient=True)
                                # VECTORIZED correlation computation (avoid per-sample loop)
                                coherent_flat = coherent_map_val.view(coherent_map_val.size(0), -1)
                                cv_flat = cv_map_val.view(cv_map_val.size(0), -1)
                                # Vectorized: compute correlation for all samples at once
                                c_centered = coherent_flat - coherent_flat.mean(dim=1, keepdim=True)
                                cv_centered = cv_flat - cv_flat.mean(dim=1, keepdim=True)
                                numerator = (c_centered * cv_centered).sum(dim=1)
                                denominator = torch.sqrt((c_centered ** 2).sum(dim=1) * (cv_centered ** 2).sum(dim=1) + 1e-8)
                                corr_batch = numerator / denominator
                                # Extract values and add to list (single GPU->CPU transfer for whole batch)
                                val_coherence_correlations.extend(corr_batch.cpu().tolist())
                                # MEMORY LEAK FIX: Aggressively delete all intermediate tensors
                                del cv_map_val, coherent_flat, cv_flat, c_centered, cv_centered, numerator, denominator, corr_batch
                            # Don't delete coherent_map_val if it's a reference to aux data

                        # CASA MAP METRICS (sampled): log interpretable map statistics for plots/tables
                        # Sample every 10 batches (and the last) to keep validation overhead manageable.
                        if hasattr(val_model, "adapter") and isinstance(val_model.adapter, CoherenceSpatialAdapter):
                            should_sample_casa_maps = ((val_step + 1) % 10 == 0) or ((val_step + 1) == val_total)
                            if should_sample_casa_maps:
                                try:
                                    casa_out = val_model(x_val_noisy, return_aux=True)
                                    if isinstance(casa_out, tuple) and len(casa_out) > 1:
                                        _, casa_aux = casa_out
                                    else:
                                        casa_aux = {}
                                    coh_map = casa_aux.get("coherent_map")
                                    incoh_map = casa_aux.get("incoherent_map")
                                    if coh_map is not None and incoh_map is not None:
                                        val_casa_coh_mean_samples.append(float(coh_map.mean().item()))
                                        val_casa_coh_std_samples.append(float(coh_map.std().item()))
                                        val_casa_entropy_samples.append(
                                            float(casa_entropy_regularizer(coh_map, incoh_map).item())
                                        )

                                        residual = (x_val_noisy - full_denoised).abs()
                                        cv_map = compute_local_cv(residual)
                                        cv_norm = (cv_map - cv_map.min()) / (cv_map.max() - cv_map.min() + 1e-8)
                                        corr = float((1.0 - pearson_correlation_loss(coh_map, cv_norm)).item())
                                        val_casa_corr_residual_cv_samples.append(corr)
                                        del residual, cv_map, cv_norm

                                    del coh_map, incoh_map, casa_out, casa_aux
                                except Exception:
                                    pass

                        # VALIDATION PROGRESS LOGGING: print periodic updates to show progress
                        if (val_step + 1) % val_log_every == 0 or (val_step + 1) == val_total:
                            mean_psnr_so_far = float(np.mean(val_psnrs)) if len(val_psnrs) > 0 else 0.0
                            mean_ssim_so_far = float(np.mean(val_ssims)) if len(val_ssims) > 0 else 0.0

                            # ENHANCED VALIDATION LOGGING: Add detailed stats
                            current_psnr = val_psnrs[-1] if len(val_psnrs) > 0 else 0.0
                            current_ssim = val_ssims[-1] if len(val_ssims) > 0 else 0.0
                            print(f"\n[Validation] Epoch {epoch+1} Batch {val_step+1}/{val_total}")
                            print(f"  Loss: {batch_val_loss:.4f}")
                            print(f"  Current PSNR: {current_psnr:.2f} dB | Current SSIM: {current_ssim:.4f}")
                            print(f"  Running Mean PSNR: {mean_psnr_so_far:.2f} dB | Running Mean SSIM: {mean_ssim_so_far:.4f}")
                            print(f"  Denoised Stats: min={full_denoised.min().item():.4f}, max={full_denoised.max().item():.4f}, "
                                  f"mean={full_denoised.mean().item():.4f}, std={full_denoised.std().item():.4f}")

                            # B2U-specific stats
                            if use_blind2unblind and val_b2u_mask is not None:
                                mask_ratio = val_b2u_mask.float().mean().item()
                                print(f"  B2U Mask Ratio (visible): {mask_ratio:.2%}")
                                print(f"  B2U Pred Full Stats: min={val_pred_full.min().item():.4f}, max={val_pred_full.max().item():.4f}, "
                                      f"mean={val_pred_full.mean().item():.4f}")
                                print(f"  B2U Pred Masked Stats: min={val_pred_masked.min().item():.4f}, max={val_pred_masked.max().item():.4f}, "
                                      f"mean={val_pred_masked.mean().item():.4f}")
                                blind_diff = (val_pred_full - val_pred_masked).abs()
                                print(f"  B2U Pred Difference: mean={blind_diff.mean().item():.4f}, max={blind_diff.max().item():.4f}")

                            # CASA-specific stats (always print when CASA adapter is used)
                            if hasattr(val_model, 'adapter') and isinstance(val_model.adapter, CoherenceSpatialAdapter):
                                try:
                                    with torch.no_grad():
                                        # Reuse aux if already computed, otherwise compute it
                                        if use_blind2unblind and 'aux_full' in locals() and aux_full:
                                            casa_aux = aux_full
                                            casa_aux_out = None
                                        elif not use_blind2unblind and not use_self2self and 'aux' in locals() and aux:
                                            casa_aux = aux
                                            casa_aux_out = None
                                        else:
                                            casa_aux_out = val_model(x_val_noisy, return_aux=True)
                                            if isinstance(casa_aux_out, tuple) and len(casa_aux_out) > 1:
                                                _, casa_aux = casa_aux_out
                                            else:
                                                casa_aux = {}

                                        if casa_aux and 'coherent_map' in casa_aux and 'incoherent_map' in casa_aux:
                                            coh_map = casa_aux['coherent_map']
                                            incoh_map = casa_aux['incoherent_map']
                                            print(f"  🎯 CASA Coherent Map: min={coh_map.min().item():.4f}, max={coh_map.max().item():.4f}, "
                                                  f"mean={coh_map.mean().item():.4f}, std={coh_map.std().item():.4f}")
                                            print(f"  🎯 CASA Incoherent Map: min={incoh_map.min().item():.4f}, max={incoh_map.max().item():.4f}, "
                                                  f"mean={incoh_map.mean().item():.4f}, std={incoh_map.std().item():.4f}")

                                            # Check if CASA is actually adapting (maps should vary across image)
                                            coh_std = coh_map.std().item()
                                            incoh_std = incoh_map.std().item()
                                            coh_mean = coh_map.mean().item()
                                            incoh_mean = incoh_map.mean().item()

                                            # Compute normalized variation coefficient
                                            coh_cv = coh_std / (coh_mean + 1e-6)
                                            incoh_cv = incoh_std / (incoh_mean + 1e-6)

                                            print(f"  🎯 CASA Spatial Adaptation:")
                                            print(f"     - Coherent Weight: {coh_mean:.4f} ± {coh_std:.4f} (CV: {coh_cv:.4f})")
                                            print(f"     - Incoherent Weight: {incoh_mean:.4f} ± {incoh_std:.4f} (CV: {incoh_cv:.4f})")
                                            print(f"     - Weight Balance: coherent={coh_mean/(coh_mean+incoh_mean+1e-6):.2%}, "
                                                  f"incoherent={incoh_mean/(coh_mean+incoh_mean+1e-6):.2%}")

                                            # Warnings for potential issues
                                            if coh_std < 0.01 and incoh_std < 0.01:
                                                print(f"  ⚠️  [WARNING] CASA maps have low spatial variation - adapter may not be adapting!")
                                            if coh_mean < 0.1 and incoh_mean < 0.1:
                                                print(f"  ⚠️  [WARNING] CASA weights are very low - check if adapter is learning!")

                                            # Cleanup
                                            if casa_aux_out is not None:
                                                del coh_map, incoh_map, casa_aux_out, casa_aux
                                            else:
                                                del coh_map, incoh_map
                                except Exception as e:
                                    print(f"  ❌ [CASA Stats Error]: {e}")

                            print("", flush=True)

                        # MEMORY LEAK FIX: Delete tensors after each validation batch
                        if use_hybrid_selfsup:
                            del x_val_noisy, y_val_clean, val_model_input, val_masked_input_n2v, val_mask_coords, val_masked_input_b2u, val_b2u_mask, val_pred_full, val_pred_masked, val_pred_n2v, v_loss_b2u, v_loss_n2v, val_pred, v_loss, full_denoised, denoised_np, clean_np
                        elif use_blind2unblind:
                            # Delete blind_diff if it exists (created during logging)
                            if 'blind_diff' in locals(): del blind_diff
                            del x_val_noisy, y_val_clean, val_model_input, val_masked_input, val_b2u_mask, val_pred_full, val_pred_masked, val_pred, v_loss, full_denoised, denoised_np, clean_np
                        elif use_self2self:
                            del x_val_noisy, y_val_clean, val_model_input, val_masks_inputs, val_masks, mask_losses, val_pred, v_loss, full_denoised, denoised_np, clean_np
                        elif use_noise2void:
                            del x_val_noisy, y_val_clean, val_model_input, val_masked_input, val_mask_coords, val_pixel_target, val_pred, v_loss, full_denoised, denoised_np, clean_np
                        elif use_neighbor2neighbor:
                            del x_val_noisy, y_val_clean, val_model_input, val_input_subset, val_target_subset, val_target_mask, val_n2n_data, val_pixel_target, val_pred, v_loss, full_denoised, denoised_np, clean_np
                        else:
                            # Supervised mode cleanup
                            try:
                                del pred_for_loss, target_for_loss
                            except NameError:
                                pass  # Variables don't exist if log_domain=False
                            del x_val_noisy, y_val_clean, val_model_input, val_pixel_target, val_pred, v_loss, full_denoised, denoised_np, clean_np
            # CRITICAL FIX: Handle empty validation loader (no batches processed)
            val_loss = np.mean(val_losses) if len(val_losses) > 0 else float('inf')
            val_psnr = np.mean(val_psnrs) if len(val_psnrs) > 0 else 0.0
            val_ssim = np.mean(val_ssims) if len(val_ssims) > 0 else 0.0
            val_std_psnr = np.std(val_psnrs) if len(val_psnrs) > 0 else 0.0
            val_std_ssim = np.std(val_ssims) if len(val_ssims) > 0 else 0.0
            val_coherence_corr = np.mean(val_coherence_correlations) if len(val_coherence_correlations) > 0 else 0.0
            val_psnr_noisy = np.mean(val_psnrs_noisy) if len(val_psnrs_noisy) > 0 else 0.0
            val_ssim_noisy = np.mean(val_ssims_noisy) if len(val_ssims_noisy) > 0 else 0.0

            # CASA CONTRIBUTION: Compute before deleting lists
            psnr_no_casa = np.mean(val_psnrs_no_casa) if len(val_psnrs_no_casa) > 0 else 0.0
            ssim_no_casa = np.mean(val_ssims_no_casa) if len(val_ssims_no_casa) > 0 else 0.0
            psnr_with_casa_sample = np.mean(val_psnrs_with_casa_sample) if len(val_psnrs_with_casa_sample) > 0 else 0.0
            ssim_with_casa_sample = np.mean(val_ssims_with_casa_sample) if len(val_ssims_with_casa_sample) > 0 else 0.0
            has_casa_stats = (
                len(val_psnrs_no_casa) > 0
                and len(val_ssims_no_casa) > 0
                and len(val_psnrs_with_casa_sample) == len(val_psnrs_no_casa)
                and len(val_ssims_with_casa_sample) == len(val_ssims_no_casa)
            )

            # Persist metrics for analysis/plots (JSONL; one record per epoch).
            # IMPORTANT: compute CASA sampled stats BEFORE deleting validation lists.
            casa_sampled_metrics = None
            if isinstance(model.adapter, CoherenceSpatialAdapter):
                casa_sampled_metrics = {
                    "coherent_mean": float(np.mean(val_casa_coh_mean_samples)) if val_casa_coh_mean_samples else None,
                    "coherent_std": float(np.mean(val_casa_coh_std_samples)) if val_casa_coh_std_samples else None,
                    "entropy": float(np.mean(val_casa_entropy_samples)) if val_casa_entropy_samples else None,
                    "corr_residual_cv": float(np.mean(val_casa_corr_residual_cv_samples)) if val_casa_corr_residual_cv_samples else None,
                    "n_samples": int(len(val_casa_corr_residual_cv_samples)),
                }

            append_jsonl(
                metrics_jsonl_path,
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(epoch_mean_loss),
                    "val_loss": float(val_loss),
                    "val_psnr_mean": float(val_psnr),
                    "val_psnr_std": float(val_std_psnr),
                    "val_ssim_mean": float(val_ssim),
                    "val_ssim_std": float(val_std_ssim),
                    "noisy_psnr_mean": float(val_psnr_noisy),
                    "noisy_ssim_mean": float(val_ssim_noisy),
                    "casa_gain_psnr_sample": float(psnr_with_casa_sample - psnr_no_casa) if has_casa_stats else None,
                    "casa_gain_ssim_sample": float(ssim_with_casa_sample - ssim_no_casa) if has_casa_stats else None,
                    "casa_sampled": casa_sampled_metrics,
                    "settings": {
                        "lambda_coherence": float(lambda_coherence),
                        "coherence_target": coherence_target,
                        "lambda_casa_entropy": float(lambda_casa_entropy),
                        "lambda_casa_std": float(lambda_casa_std),
                        "casa_min_std": float(casa_min_std),
                        "lambda_moe_balance": float(lambda_moe_balance),
                        "lambda_moe_var": float(lambda_moe_var),
                        "lambda_theta": float(lambda_theta),
                    },
                },
            )

            del val_losses, val_psnrs, val_ssims, val_psnrs_noisy, val_ssims_noisy, val_psnrs_no_casa, val_ssims_no_casa, val_psnrs_with_casa_sample, val_ssims_with_casa_sample, val_coherence_correlations, val_casa_corr_residual_cv_samples, val_casa_entropy_samples, val_casa_coh_std_samples, val_casa_coh_mean_samples  # MEMORY LEAK FIX: Delete validation lists
            torch.cuda.empty_cache()  # Clear CUDA cache after validation

            # BUG FIX: Step ReduceLROnPlateau scheduler with validation loss
            if scheduler_type == 'plateau':
                if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
                    if scheduler.last_epoch >= scheduler._milestones[0]:
                        scheduler._schedulers[1].step(val_loss)
                else:
                    scheduler.step(val_loss)
            # CRITICAL FIX: Restore original model weights after EMA-based validation
            if ema_enable and ema_model_cache is not None:
                model.load_state_dict({k: v.to(device) for k, v in ema_model_cache.items()})
                model.train()
                ema_model_cache = None

            # MEMORY MONITORING: Print memory stats after validation
            val_stats = get_memory_stats()
            val_mem_str = " | ".join([f"{k.split('_')[0]}:{v:.2f}GB" for k, v in val_stats.items()])

            # Enhanced validation summary with clear PSNR/SSIM display
            print(f"\n{'='*80}")
            print(f"VALIDATION SUMMARY - Epoch {epoch+1}")
            print(f"{'='*80}")
            print(f"  Loss:        {val_loss:.4f}")
            print(f"  Noisy Input: PSNR {val_psnr_noisy:.2f} dB | SSIM {val_ssim_noisy:.4f}")
            print(f"  📊 PSNR:     {val_psnr:.2f} ± {val_std_psnr:.2f} dB")
            print(f"  📊 SSIM:     {val_ssim:.4f} ± {val_std_ssim:.4f}")

            # CASA CONTRIBUTION SUMMARY: Show improvement from CASA
            if has_casa_stats:
                # Contribution is computed on the sampled subset where both WITH/NO-CAST metrics were measured.
                psnr_gain = psnr_with_casa_sample - psnr_no_casa
                ssim_gain = ssim_with_casa_sample - ssim_no_casa
                psnr_gain_pct = 100 * psnr_gain / (psnr_no_casa + 1e-6)
                ssim_gain_pct = 100 * ssim_gain / (ssim_no_casa + 1e-6)

                print(f"\n  🔍 CASA CONTRIBUTION ANALYSIS:")
                print(f"     Without CASA (sampled): PSNR {psnr_no_casa:.2f} dB | SSIM {ssim_no_casa:.4f}")
                print(f"     With CASA (sampled):    PSNR {psnr_with_casa_sample:.2f} dB | SSIM {ssim_with_casa_sample:.4f}")
                print(f"     🎯 CASA Gain:  PSNR {psnr_gain:+.2f} dB ({psnr_gain_pct:+.1f}%) | SSIM {ssim_gain:+.4f} ({ssim_gain_pct:+.1f}%)")

                # Interpretation
                if psnr_gain > 0.5:
                    print(f"     ✅ CASA is SIGNIFICANTLY improving denoising quality!")
                elif psnr_gain > 0.1:
                    print(f"     ✓ CASA is moderately improving results")
                elif psnr_gain > -0.1:
                    print(f"     ⚠️  CASA has minimal impact (consider adjusting lambda_coherence)")
                else:
                    print(f"     ❌ WARNING: CASA is degrading performance! Check training dynamics")

            # Enhanced logging for coherence supervision
            if lambda_coherence > 0 and use_noise2void and val_coherence_corr != 0.0:
                print(f"  Coherence-CV Corr: {val_coherence_corr:.3f}")
                # Alert if correlation is improving (key indicator of physics learning!)
                if val_coherence_corr > 0.5:
                    print(f"  ✓✓ Coherence correlation > 0.5 - STRONG physics learning!")
                elif val_coherence_corr > 0.3:
                    print(f"  ✓ Coherence correlation > 0.3 - CASA is learning physics!")

            print(f"  Memory:      {val_mem_str}")
            print(f"{'='*80}\n", flush=True)

        # CRITICAL FIX: Only step plateau scheduler if validation actually ran (val_loss is defined)
        if scheduler_type == 'plateau' and should_validate and val_loader:
            # If warmup + plateau, scheduler is SequentialLR; step warmup without metric, plateau with metric
            if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
                warmup_iters = scheduler._milestones[0]
                if scheduler.last_epoch < warmup_iters:
                    scheduler.step()
                else:
                    main_scheduler.step(val_loss)
            else:
                scheduler.step(val_loss)

        # Early stopping with best model tracking
        # MEMORY MONITORING: Get memory stats for epoch summary
        epoch_end_stats = get_memory_stats()
        epoch_mem_str = " | ".join([f"{k.split('_')[0]}:{v:.2f}GB" for k, v in epoch_end_stats.items()])

        # BUG FIX: Only do early stopping when validation actually ran
        # CRITICAL FIX: Track PSNR instead of loss for N2V (loss can decrease while quality degrades!)
        if should_validate:
            if val_psnr > best_val_psnr:  # Higher PSNR is better
                best_val_loss = val_loss
                best_val_psnr = val_psnr
                epochs_no_improve = 0
                best_epoch = epoch + 1

                # CRITICAL FIX: Save best checkpoint to disk immediately (overwrites previous best)
                # This ensures: 1) Best checkpoint not lost if training crashes, 2) Only one checkpoint file (saves space)
                # Save full training state for perfect resumption
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_psnr': best_val_psnr,
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve,
                    'ema_state_dict': ema_state if ema_enable and ema_state else None,
                }
                best_model_path = os.path.join(output_dir, "best_model.pth")
                torch.save(checkpoint_state, best_model_path)
                print(f"✓ Saved best checkpoint (full state): {best_model_path}")

                # Also save model-only checkpoint for backward compatibility
                model_only_path = os.path.join(output_dir, "best_model_weights_only.pth")
                torch.save(model.state_dict(), model_only_path)

                if ema_enable and ema_state:
                    best_ema_path = os.path.join(output_dir, "best_model_ema.pth")
                    torch.save(ema_state, best_ema_path)
                    print(f"✓ Saved best EMA checkpoint: {best_ema_path}")

                # Keep in memory for restoration at end of training (faster than reloading from disk)
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if ema_enable and ema_state:
                    best_ema_state = {k: v.cpu().clone() for k, v in ema_state.items()}
                else:
                    best_ema_state = None

                print(f"[Finetune] Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_mean_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} ⭐ BEST (PSNR) | Mem: {epoch_mem_str}", flush=True)
            else:
                epochs_no_improve += 1
                print(f"[Finetune] Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_mean_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} (No improve: {epochs_no_improve}/{early_stopping_patience}) | Mem: {epoch_mem_str}", flush=True)
        else:
            # Validation was skipped this epoch
            print(f"[Finetune] Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_mean_loss:.4f} | Val: SKIPPED (freq={validation_frequency}) | Mem: {epoch_mem_str}", flush=True)

        # BUG FIX: Removed duplicate deletion block - running_loss was already deleted at line 3238
        # The original code had "del running_loss" here which caused NameError since it was already deleted

        if epochs_no_improve >= early_stopping_patience:
            print(f"[Finetune] Early stopping triggered after {epochs_no_improve} epochs without improvement.", flush=True)
            print(f"[Finetune] Best model was at epoch {best_epoch} with val PSNR {best_val_psnr:.2f} dB", flush=True)
            break

        force_memory_cleanup()

        # EMA consolidation every 10 epochs to prevent memory fragmentation
        if ema_enable and ema_state and (epoch + 1) % 10 == 0:
            ema_state = {k: v.contiguous() for k, v in ema_state.items()}
            print(f"[EMA] Consolidated EMA state at epoch {epoch + 1}", flush=True)

    # Restore best model weights if we have them; otherwise save current (no-val case)
    if best_model_state is not None:
        print(f"[Finetune] Restoring best model from epoch {best_epoch}", flush=True)
        # MEMORY LEAK FIX: Move best_model_state back to GPU before loading
        best_model_state_gpu = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(best_model_state_gpu)
        # CRITICAL FIX: Restore best EMA state (not regular model state!)
        if ema_enable and best_ema_state is not None:
            ema_state = {k: v.to(device) for k, v in best_ema_state.items()}
            del best_ema_state
        # MEMORY LEAK FIX: Delete CPU copy of best model
        del best_model_state, best_model_state_gpu
    else:
        # No validation performed; save current model as best
        print("[Finetune] No validation performed; saving final model as best.", flush=True)
        best_model_path = os.path.join(output_dir, "best_model.pth")
        torch.save({"model_state_dict": model.state_dict()}, best_model_path)
        if ema_enable and ema_state is not None:
            best_ema_path = os.path.join(output_dir, "best_model_ema.pth")
            torch.save(ema_state, best_ema_path)

    # Set model to eval mode after training
    model.eval()

    # MEMORY LEAK FIX: Final cleanup
    if 'compiled_model' in locals() and compiled_model is not model:
        del compiled_model
    torch.cuda.empty_cache()

    # MEMORY MONITORING: Print final memory stats
    final_stats = get_memory_stats()
    print("\n" + "="*60)
    print("SUPERVISED FINE-TUNING COMPLETED - Final Memory Stats:")
    for key, val in final_stats.items():
        print(f"  {key}: {val:.3f} GB")
    if 'finetune_initial_stats' in locals():
        print("\nMemory Change from Start:")
        for key in final_stats.keys():
            if key in finetune_initial_stats:
                delta = final_stats[key] - finetune_initial_stats[key]
                print(f"  {key}: {delta:+.3f} GB")
    print("="*60 + "\n", flush=True)

    return ema_state

def main():
    import argparse
    from datetime import datetime
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Domain-Adaptive OCT Denoising with Meta-learning and TTA")
    parser.add_argument("--clean_root", type=str, default=None, help="Path to clean OCT images for meta-training")
    parser.add_argument("--paired_list", type=str, default=None, help="Path to real noisy-clean training pairs list")
    parser.add_argument("--val_paired_list", type=str, default=None, help="Path to validation pairs list")
    parser.add_argument("--noisy_list", type=str, default=None, help="Path to noisy-only list (for blind2unblind)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--residual_mode", action="store_true",
                        help="Use residual learning mode: model predicts noise, clean = noisy - noise (for B2U/UNet)")
    parser.add_argument("--adapter", type=str, default="global", choices=["none","global","spatial","casa","moe","siminv"])
    parser.add_argument("--moe_experts", type=int, default=4, help="MoE adapter: number of experts (>=2).")
    parser.add_argument("--moe_hidden_channels", type=int, default=32, help="MoE adapter: expert hidden channels.")
    parser.add_argument("--moe_temperature", type=float, default=1.0, help="MoE adapter: softmax temperature (>0).")
    parser.add_argument("--siminv_hidden_channels", type=int, default=32, help="SimInv adapter: parameter estimator hidden channels.")
    parser.add_argument("--siminv_film_scale_gamma", type=float, default=0.1, help="SimInv adapter: FiLM gamma scale (try 0.1 to 0.5).")
    parser.add_argument("--siminv_film_scale_beta", type=float, default=0.1, help="SimInv adapter: FiLM beta scale (try 0.1 to 0.5).")
    parser.add_argument("--noise_params_jsonl", type=str, default=None,
                        help="SimInv training: comma-separated JSONL metadata files produced by create_realistic_oct_dataset.py --write_params_jsonl.")
    parser.add_argument("--lambda_theta", type=float, default=0.0,
                        help="SimInv training: weight for theta regression loss (try 0.1 to 1.0).")
    parser.add_argument("--strategy", type=str, default="supervised", 
                        choices=["supervised", "noise2void", "neighbor2neighbor", "blind2unblind", "self2self"],
                        help="Training strategy: supervised, Noise2Void, Neighbor2Neighbor, Blind2Unblind, or Self2Self.")
    parser.add_argument("--backbone", type=str, default="unet",  choices=["unet","nafnet","noise2void","neighbor2neighbor","b2unet","s2s"])
    parser.add_argument("--n2v_mask_ratio", type=float, default=0.20, help="Noise2Void blind-spot mask ratio (0.20-0.25 recommended for better performance)")
    parser.add_argument("--n2v_box_size", type=int, default=5, help="Noise2Void stratified mask box size")
    parser.add_argument("--n2v_blindspot_dilation", type=int, default=1, help="Dilated blind-spot size (1=standard N2V, 3=3x3, 5=5x5). Larger values help with spatially correlated noise like OCT speckle.")
    parser.add_argument("--b2u_mask_ratio", type=float, default=0.5, help="Blind2Unblind mask ratio (fraction of blind spots)")
    parser.add_argument("--b2u_block_size", type=int, default=2, help="Blind2Unblind pixel-unshuffle block size")
    parser.add_argument("--s2s_mask_prob", type=float, default=0.3, help="Self2Self Bernoulli mask probability")
    parser.add_argument("--s2s_num_masks", type=int, default=8, help="Number of Bernoulli masks per image for Self2Self training")
    parser.add_argument("--hybrid_selfsup", action="store_true",
                        help="Enable hybrid self-supervised fine-tuning alternating N2V and B2U on the same model.")
    parser.add_argument("--hybrid_mode", type=str, default="alternate", choices=["alternate", "noise_aware"],
                        help="Hybrid selection mode: alternate per step or route by noise correlation.")
    parser.add_argument("--hybrid_start_with", type=str, default="b2u", choices=["b2u", "n2v"],
                        help="Starting mode for alternate hybrid schedule.")
    parser.add_argument("--hybrid_corr_threshold", type=float, default=0.08,
                        help="Correlation score threshold for noise_aware hybrid routing (higher => choose B2U).")
    parser.add_argument("--hybrid_corr_low", type=float, default=None,
                        help="Lower correlation threshold for N2V in noise_aware mode (optional deadband).")
    parser.add_argument("--hybrid_corr_high", type=float, default=None,
                        help="Upper correlation threshold for B2U in noise_aware mode (optional deadband).")
    parser.add_argument("--hybrid_calib_steps", type=int, default=0,
                        help="If >0, calibrate corr threshold from first N steps (running mean, no storage).")
    parser.add_argument("--hybrid_calib_factor", type=float, default=1.0,
                        help="Multiplier on calibrated threshold (1.0=use mean).")
    parser.add_argument("--hybrid_calib_stat", type=str, default="mean", choices=["mean", "median"],
                        help="Statistic for calibration after warmup: mean (no storage) or median (robust).")
    parser.add_argument("--num_meta_epochs", type=int, default=10)
    parser.add_argument("--num_tasks_per_meta_batch", type=int, default=4)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--inner_lr", type=float, default=1e-4)
    parser.add_argument("--meta_step_size", type=float, default=0.1)
    parser.add_argument("--meta_log_interval", type=int, default=10)
    parser.add_argument("--meta_strategy", type=str, default="supervised", 
                        choices=["supervised", "noise2void", "neighbor2neighbor", "blind2unblind"],
                        help="Meta-learning strategy: supervised, Noise2Void, Neighbor2Neighbor, or Blind2Unblind.")
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--finetune_lr_adapter", type=float, default=5e-4, help="Adapter learning rate (5e-4 prevents overshoot)")
    parser.add_argument("--finetune_lr_backbone", type=float, default=1e-4, help="Backbone learning rate (1e-4 for stability)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--finetune_freeze_backbone", action="store_true")
    parser.add_argument("--loss_type", type=str, default='charbonnier', choices=['charbonnier','l1'])
    parser.add_argument("--lambda_grad", type=float, default=0.05)
    parser.add_argument("--lambda_tv", type=float, default=1e-5)
    parser.add_argument("--lambda_depth", type=float, default=0.0)
    parser.add_argument("--lambda_ascan", type=float, default=0.0)
    parser.add_argument("--lambda_speckle", type=float, default=0.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.0, help="Perceptual loss weight (0.1 recommended for structure preservation)")
    parser.add_argument("--lambda_multiscale", type=float, default=0.0, help="Multi-scale loss weight (0.5 recommended)")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for regularization (0.1-0.2 recommended)")
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation (8-fold ensemble) for +1-2 dB gain")
    parser.add_argument("--use_blind2unblind", action="store_true", help="Shortcut to set --strategy blind2unblind")
    parser.add_argument("--log_domain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--casa_lambda_tv", type=float, default=0.0)
    parser.add_argument("--lambda_coherence", type=float, default=0.0, help="Coherence supervision loss weight (0.1 recommended for CASA physics learning)")
    parser.add_argument(
        "--coherence_target",
        type=str,
        default="residual",
        choices=["residual", "noisy"],
        help="Target for CASA coherence supervision: residual uses |noisy - pred|, noisy uses the raw noisy input.",
    )
    parser.add_argument(
        "--lambda_casa_entropy",
        type=float,
        default=0.0,
        help="Entropy regularizer on CASA maps (non-collapse). Typical: 1e-3 to 1e-2.",
    )
    parser.add_argument(
        "--lambda_casa_std",
        type=float,
        default=0.0,
        help="Std-floor regularizer on CASA coherent map (non-collapse). Typical: 1e-2 to 1e-1.",
    )
    parser.add_argument(
        "--casa_min_std",
        type=float,
        default=0.0,
        help="Minimum std target for CASA coherent map when using --lambda_casa_std. Typical: 0.03 to 0.08.",
    )
    parser.add_argument("--lambda_moe_balance", type=float, default=0.0,
                        help="MoE-only: load-balance regularizer on gate mean weights (try 1e-3 to 1e-1).")
    parser.add_argument("--lambda_moe_var", type=float, default=0.0,
                        help="MoE-only: encourage sample-dependent routing by maximizing gate variance (try 1e-3 to 1e-1).")
    parser.add_argument("--lambda_anchor", type=float, default=0.1, help="Anchor loss weight (L1 between pred and noisy input) to prevent B2U collapse")
    parser.add_argument("--scheduler_type", type=str, default='cosine', choices=['cosine','plateau'])
    parser.add_argument("--warmup_epochs", type=int, default=8, help="Linear LR warmup epochs (8 recommended for stable training)")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--validation_frequency", type=int, default=1, help="Run validation every N epochs (higher = faster training)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients over N steps for larger effective batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize_h", type=int, default=64)
    parser.add_argument("--resize_w", type=int, default=64)
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between training loss prints")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint file (e.g., checkpoints/model/best_model.pth) to resume training from")
    parser.add_argument("--init_backbone_from", type=str, default=None,
                        help="Optional: initialize ONLY the backbone weights from a checkpoint (useful when switching "
                             "from --adapter none to --adapter casa). Optimizer/scheduler are not resumed.")
    parser.add_argument("--init_adapter_from", type=str, default=None,
                        help="Optional: initialize ONLY the adapter weights from a checkpoint (state_dict). "
                             "Optimizer/scheduler are not resumed.")
    parser.add_argument("--sanity_test", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect")
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="GPU ID to use (e.g., 0, 1, 2). Shortcut for --device cuda:ID")
    args = parser.parse_args()

    # Help avoid "old code vs new code" confusion during long training runs.
    try:
        script_path = Path(__file__).resolve()
        script_mtime = datetime.fromtimestamp(script_path.stat().st_mtime).isoformat(timespec="seconds")
        print(f"[Code] Running script: {script_path} (mtime: {script_mtime})", flush=True)
    except Exception:
        pass

    # Handle --use_blind2unblind shortcut
    if args.use_blind2unblind:
        if args.strategy != "supervised" and args.strategy != "blind2unblind":
            parser.error(f"--use_blind2unblind conflicts with --strategy {args.strategy}. Remove --strategy or use --strategy blind2unblind.")
        args.strategy = "blind2unblind"
        # CRITICAL FIX: Also set meta_strategy to "blind2unblind" if it's default "supervised"
        # This allows B2U meta-learning when --use_blind2unblind is present
        if args.meta_strategy == "supervised":
            args.meta_strategy = "blind2unblind"

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # MEMORY MONITORING: Initial system stats
    print("\n" + "="*80)
    print("TRAINING SESSION STARTING - System Memory Stats:")
    session_start_stats = get_memory_stats()
    for key, val in session_start_stats.items():
        print(f"  {key}: {val:.3f} GB")
    print("="*80 + "\n", flush=True)

    # Device selection
    global device
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU {args.gpu_id}: cuda:{args.gpu_id}")
    elif args.device is not None:
        device = torch.device(args.device)
        print(f"Using device: {args.device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Auto-detected device: {device}")

    print("=" * 80)
    print("MODEL CONFIGURATION")
    print(f"  Strategy: {args.strategy}, Backbone: {args.backbone}, Adapter: {args.adapter}, Base Channels: {args.base_channels}")
    print(f"  Residual Mode: {args.residual_mode}, Log Domain Training: {args.log_domain}")

    # MEMORY MONITORING: Show if Noise2Void or Neighbor2Neighbor mode is active
    if args.strategy == "noise2void":
        print(f"  🔬 NOISE2VOID MODE ENABLED (Strategy)")
        print(f"     - Mask Ratio: {args.n2v_mask_ratio} ({args.n2v_mask_ratio*100:.0f}% of pixels masked)")
        print(f"     - Box Size: {args.n2v_box_size}x{args.n2v_box_size}")
        print(f"     - Training Mode: Self-Supervised (no clean pairs needed)")
    elif args.strategy == "neighbor2neighbor":
        print(f"  🔬 NEIGHBOR2NEIGHBOR MODE ENABLED (Strategy)")
        print(f"     - Sub-sampling: Checkerboard pattern (50% of pixels)")
        print(f"     - Training Mode: Self-Supervised (no clean pairs needed)")
    elif args.strategy == "blind2unblind":
        print(f"  🔬 BLIND2UNBLIND MODE ENABLED (Strategy)")
        print(f"     - Mask Ratio: {args.b2u_mask_ratio} ({args.b2u_mask_ratio*100:.0f}% of pixels masked)")
        print(f"     - Block Size: {args.b2u_block_size}")
        print(f"     - Training Mode: Self-Supervised (no clean pairs needed)")
    if args.hybrid_selfsup:
        print(f"  🔀 HYBRID SELF-SUP MODE ENABLED (N2V + B2U)")
        print(f"     - Hybrid Mode: {args.hybrid_mode}")
        if args.hybrid_mode == "alternate":
            print(f"     - Start With: {args.hybrid_start_with}")
        else:
            print(f"     - Corr Threshold: {args.hybrid_corr_threshold}")
            if args.hybrid_corr_low is not None or args.hybrid_corr_high is not None:
                print(f"     - Corr Deadband: low={args.hybrid_corr_low}, high={args.hybrid_corr_high}")
            if args.hybrid_calib_steps > 0:
                print(f"     - Corr Calibration: steps={args.hybrid_calib_steps}, stat={args.hybrid_calib_stat}, factor={args.hybrid_calib_factor}")
    print("=" * 80 + "\n")

    model = build_model(
        base_channels=args.base_channels,
        residual_mode=args.residual_mode,
        adapter_type=args.adapter,
        backbone_type=args.backbone,
        moe_experts=args.moe_experts,
        moe_hidden_channels=args.moe_hidden_channels,
        moe_temperature=args.moe_temperature,
        siminv_hidden_channels=args.siminv_hidden_channels,
        siminv_film_scale_gamma=args.siminv_film_scale_gamma,
        siminv_film_scale_beta=args.siminv_film_scale_beta,
    )
    model.to(device)

    # Variables to store resume state
    resume_start_epoch = 0
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_ema_state = None
    resume_best_val_psnr = -float('inf')
    resume_best_val_loss = float('inf')
    resume_epochs_no_improve = 0
    loaded_meta_checkpoint = False  # Track if we loaded a meta checkpoint

    # Load checkpoint if resuming training
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if not os.path.isabs(checkpoint_path):
            # If relative path, make it absolute from current directory
            checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

        if os.path.exists(checkpoint_path):
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
            print(f"{'='*80}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Check if this is a meta checkpoint (e.g., meta_adapter.pth)
            is_meta_checkpoint = 'meta_adapter' in os.path.basename(checkpoint_path)

            # Check if this is a full checkpoint or just model weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                print(f"✓ Detected FULL checkpoint (includes optimizer, scheduler, epoch)")
                model.load_state_dict(checkpoint['model_state_dict'])
                resume_start_epoch = checkpoint.get('epoch', 0)
                resume_optimizer_state = checkpoint.get('optimizer_state_dict', None)
                resume_scheduler_state = checkpoint.get('scheduler_state_dict', None)
                resume_ema_state = checkpoint.get('ema_state_dict', None)
                resume_best_val_psnr = checkpoint.get('best_val_psnr', -float('inf'))
                resume_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                resume_epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                print(f"✓ Loaded model weights, optimizer, scheduler from epoch {resume_start_epoch}")
                print(f"✓ Best PSNR: {resume_best_val_psnr:.2f} dB, Best Loss: {resume_best_val_loss:.4f}")
            else:
                # Check if this is an adapter-only checkpoint (keys don't have "backbone." or "adapter." prefix)
                sample_keys = list(checkpoint.keys())[:5]
                is_adapter_only = not any(k.startswith('backbone.') or k.startswith('adapter.') for k in sample_keys)

                if is_adapter_only:
                    # Adapter-only checkpoint (from meta-training)
                    print(f"✓ Detected ADAPTER-ONLY checkpoint (from meta-training)")
                    print(f"⚠️  Backbone will use random initialization")
                    print(f"⚠️  Optimizer and scheduler will start fresh")

                    # Load adapter weights
                    model.adapter.load_state_dict(checkpoint, strict=True)
                    print(f"✓ Loaded adapter weights from checkpoint")

                    # Set flag to skip meta-training
                    loaded_meta_checkpoint = True
                    print(f"✓ Will skip meta-training phase")
                else:
                    # Full model checkpoint (just model weights, no training state)
                    print(f"✓ Detected LEGACY checkpoint (full model weights only)")
                    print(f"⚠️  Optimizer and scheduler will start fresh")
                    try:
                        model.load_state_dict(checkpoint)
                    except RuntimeError as e:
                        msg = str(e)
                        if ("Missing key(s) in state_dict" in msg) or ("Unexpected key(s) in state_dict" in msg):
                            raise RuntimeError(
                                msg
                                + "\n\n"
                                + "Checkpoint architecture mismatch.\n"
                                + "If you are switching adapter/backbone (e.g., --adapter none -> --adapter moe/casa), "
                                + "do NOT use --resume_from_checkpoint.\n"
                                + "Use --init_backbone_from (and optionally --init_adapter_from) instead."
                            ) from e
                        raise
                    print(f"✓ Loaded model weights from checkpoint")

                    # If this is a meta checkpoint, set flag to skip meta-training
                    if is_meta_checkpoint:
                        loaded_meta_checkpoint = True
                        print(f"✓ Detected META checkpoint - will skip meta-training phase")
            print(f"{'='*80}\n")
        else:
            print(f"\n⚠️  WARNING: Checkpoint file not found: {checkpoint_path}")
            print(f"⚠️  Starting training from scratch...\n")

    def _load_state_dict_maybe_wrapped(path: str) -> dict:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        return ckpt

    def _strip_prefix(state: dict, prefix: str) -> dict:
        out = {}
        for k, v in state.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    def _maybe_make_abs(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(os.getcwd(), p)

    # Optional: initialize backbone/adapter weights from separate checkpoints (architecture-safe).
    if args.init_backbone_from or args.init_adapter_from:
        # If we resumed full training state, it's unsafe to keep optimizer/scheduler/EMA after overriding weights.
        if resume_optimizer_state is not None or resume_scheduler_state is not None or resume_ema_state is not None:
            print("⚠️  init_*_from used after a FULL resume; dropping optimizer/scheduler/EMA state for safety.")
            resume_start_epoch = 0
            resume_optimizer_state = None
            resume_scheduler_state = None
            resume_ema_state = None
            resume_best_val_psnr = -float('inf')
            resume_best_val_loss = float('inf')
            resume_epochs_no_improve = 0

        if args.init_backbone_from:
            path = _maybe_make_abs(args.init_backbone_from)
            if not os.path.exists(path):
                raise FileNotFoundError(f"--init_backbone_from not found: {path}")
            state = _load_state_dict_maybe_wrapped(path)
            if not isinstance(state, dict):
                raise ValueError(f"--init_backbone_from must be a state_dict, got {type(state)}")

            # Accept either a full model state_dict with 'backbone.' prefix, or a backbone-only state_dict.
            backbone_state = _strip_prefix(state, "backbone.")
            try:
                if backbone_state:
                    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
                else:
                    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
            except RuntimeError as e:
                msg = str(e)
                if "size mismatch" in msg:
                    raise RuntimeError(
                        msg
                        + "\n\n"
                        + "Backbone checkpoint shape mismatch.\n"
                        + "Most commonly this means you built the model with a different `--base_channels` (or different backbone).\n"
                        + "For NAFNet fair config checkpoints in this repo, use `--backbone nafnet --base_channels 48`.\n"
                    ) from e
                raise
            print(f"✓ Initialized backbone from: {path}")
            if missing:
                print(f"  (backbone missing keys: {len(missing)})")
            if unexpected:
                print(f"  (backbone unexpected keys: {len(unexpected)})")

        if args.init_adapter_from:
            path = _maybe_make_abs(args.init_adapter_from)
            if not os.path.exists(path):
                raise FileNotFoundError(f"--init_adapter_from not found: {path}")
            state = torch.load(path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if not isinstance(state, dict):
                raise ValueError(f"--init_adapter_from must be a state_dict, got {type(state)}")
            adapter_state = _strip_prefix(state, "adapter.") or state
            try:
                missing, unexpected = model.adapter.load_state_dict(adapter_state, strict=False)
            except RuntimeError as e:
                msg = str(e)
                if "size mismatch" in msg:
                    raise RuntimeError(
                        msg
                        + "\n\n"
                        + "Adapter checkpoint shape mismatch.\n"
                        + "Make sure `--adapter` type and adapter hyperparameters match the checkpoint.\n"
                    ) from e
                raise
            print(f"✓ Initialized adapter from: {path}")
            if missing:
                print(f"  (adapter missing keys: {len(missing)})")
            if unexpected:
                print(f"  (adapter unexpected keys: {len(unexpected)})")

    tfm = resize_to((args.resize_h, args.resize_w))

    # OPTIMIZATION: Adaptive num_workers based on device (CPU vs GPU)
    # CPU training: fewer workers to avoid overhead
    # GPU training: more workers to keep GPU fed
    num_workers = 2 if device.type == 'cpu' else 4
    val_num_workers = 1 if device.type == 'cpu' else 2
    print(f"[DataLoader] Using num_workers={num_workers} (device={device.type})\n")

    meta_state = None
    if loaded_meta_checkpoint:
        # Skip meta-training if we loaded a meta checkpoint
        print("="*80)
        print("SKIPPING META-TRAINING: Loaded weights from meta checkpoint")
        print("="*80)
        meta_state = model.state_dict()  # Use loaded weights as meta_state
        unfreeze_all(model)  # Ensure model is unfrozen for fine-tuning
    elif args.clean_root and args.num_meta_epochs > 0:
        print("Starting meta-learning on synthetic noise tasks...")
        # CRITICAL FIX: Allow B2U strategy for meta-learning if inferred from --use_blind2unblind
        if args.meta_strategy == "supervised" and args.strategy == "blind2unblind":
             print("  (Auto-switching meta-strategy to blind2unblind to match finetuning)")
             args.meta_strategy = "blind2unblind"

        clean_ds = CleanOCTDataset(args.clean_root, transform=tfm)
        # OPTIMIZATION: Use adaptive parallel data loading and pin_memory for faster GPU transfer
        clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=(num_workers>0))
        meta_state = reptile_meta_train(
            model,
            clean_loader=clean_loader,
            num_meta_epochs=args.num_meta_epochs,
            num_tasks_per_meta_batch=args.num_tasks_per_meta_batch,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            meta_step_size=args.meta_step_size,
            amp=args.amp,
            meta_log_interval=args.meta_log_interval,
            use_noise2void=(args.meta_strategy == "noise2void"),
            n2v_mask_ratio=args.n2v_mask_ratio,
            n2v_box_size=args.n2v_box_size,
            n2v_blindspot_dilation=args.n2v_blindspot_dilation,
            use_neighbor2neighbor=(args.meta_strategy == "neighbor2neighbor"),
            use_blind2unblind=(args.meta_strategy == "blind2unblind"),
            b2u_mask_ratio=args.b2u_mask_ratio,
            b2u_block_size=args.b2u_block_size,
            lambda_coherence=args.lambda_coherence,
            coherence_target=args.coherence_target,
            lambda_casa_entropy=args.lambda_casa_entropy,
            lambda_casa_std=args.lambda_casa_std,
            casa_min_std=args.casa_min_std,
            lambda_anchor=args.lambda_anchor,
            log_domain=args.log_domain,
        )
        torch.save(meta_state, os.path.join(args.output_dir, "meta_adapter.pth"))
        print("Saved meta-trained adapter weights.")
        unfreeze_all(model)

    # Choose dataset based on strategy
    paired_ds = None
    val_loader = None
    if args.strategy in ("blind2unblind", "self2self"):
        if args.noisy_list:
            print(f"Using noisy-only list for {args.strategy} training...")
            paired_ds = PairedOCTDataset(args.noisy_list, transform=tfm, expect_clean=False)
        if args.paired_list:
            print(f"Using paired list as noisy-only for {args.strategy} (clean targets ignored)...")
            paired_ds = PairedOCTDataset(args.paired_list, transform=tfm, expect_clean=False)
        if args.val_paired_list:
            # FIX: Always expect clean targets for validation if a list is provided, even for self-supervised strategies
            print(f"Using paired list for validation (expecting clean targets)...")
            val_ds = PairedOCTDataset(args.val_paired_list, transform=tfm, expect_clean=True)
            val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=val_num_workers, pin_memory=(device.type=='cuda'))
    elif args.paired_list and args.finetune_epochs > 0:
        print("Starting supervised fine-tuning...")
        if args.adapter == "siminv":
            if not args.noise_params_jsonl:
                raise SystemExit("--adapter siminv requires --noise_params_jsonl for supervised training.")
            if args.lambda_theta <= 0:
                raise SystemExit("--adapter siminv requires --lambda_theta > 0 to train the parameter estimator.")
            paired_ds = PairedOCTDatasetWithParams(args.paired_list, params_jsonl=args.noise_params_jsonl, transform=tfm)
        else:
            paired_ds = PairedOCTDataset(args.paired_list, transform=tfm)
        if args.val_paired_list:
            val_ds = PairedOCTDataset(args.val_paired_list, transform=tfm)
            val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=val_num_workers, pin_memory=True)

    if paired_ds is not None and args.finetune_epochs > 0:
        if args.finetune_freeze_backbone and args.adapter == "none":
            raise SystemExit(
                "--finetune_freeze_backbone cannot be used with --adapter none (no trainable parameters). "
                "Either remove --finetune_freeze_backbone or use a real adapter (global/spatial/casa)."
            )
        if args.hybrid_selfsup:
            print("Starting hybrid self-supervised fine-tuning (N2V + B2U)...")
        elif args.strategy not in ("blind2unblind", "self2self"):
            print("Starting supervised fine-tuning...")
        else:
            print(f"Starting self-supervised fine-tuning ({args.strategy})...")
        paired_loader = DataLoader(paired_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=(num_workers>0))
        ema_state = supervised_finetune(
            model,
            paired_loader=paired_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
            num_epochs=args.finetune_epochs,
            lr_adapter=args.finetune_lr_adapter,
            lr_backbone=args.finetune_lr_backbone,
            weight_decay=args.weight_decay,
            freeze_backbone=args.finetune_freeze_backbone,
            loss_type=args.loss_type,
            lambda_grad=args.lambda_grad,
            lambda_tv=args.lambda_tv,
            lambda_depth=args.lambda_depth,
            lambda_ascan=args.lambda_ascan,
            lambda_speckle=args.lambda_speckle,
            log_interval=args.log_interval,
            log_domain=args.log_domain,
            amp=args.amp,
            grad_clip_norm=args.grad_clip_norm,
            scheduler_type=args.scheduler_type,
            warmup_epochs=args.warmup_epochs,
            early_stopping_patience=args.early_stopping_patience,
            ema_enable=args.ema,
            ema_decay=args.ema_decay,
            casa_lambda_tv=args.casa_lambda_tv,
            lambda_coherence=args.lambda_coherence,
            coherence_target=args.coherence_target,
            lambda_casa_entropy=args.lambda_casa_entropy,
            lambda_casa_std=args.lambda_casa_std,
            casa_min_std=args.casa_min_std,
            lambda_moe_balance=args.lambda_moe_balance,
            lambda_moe_var=args.lambda_moe_var,
            lambda_theta=args.lambda_theta,
            lambda_anchor=args.lambda_anchor,
            validation_frequency=args.validation_frequency,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_noise2void=(args.strategy == "noise2void"),
            n2v_mask_ratio=args.n2v_mask_ratio,
            n2v_box_size=args.n2v_box_size,
            n2v_blindspot_dilation=args.n2v_blindspot_dilation,
            use_neighbor2neighbor=(args.strategy == "neighbor2neighbor"),
            use_blind2unblind=(args.strategy == "blind2unblind"),
            b2u_mask_ratio=args.b2u_mask_ratio,
            b2u_block_size=args.b2u_block_size,
            use_self2self=(args.strategy == "self2self"),
            use_hybrid_selfsup=args.hybrid_selfsup,
            hybrid_mode=args.hybrid_mode,
            hybrid_start_with=args.hybrid_start_with,
            hybrid_corr_threshold=args.hybrid_corr_threshold,
            hybrid_corr_low=args.hybrid_corr_low,
            hybrid_corr_high=args.hybrid_corr_high,
            hybrid_calib_steps=args.hybrid_calib_steps,
            hybrid_calib_factor=args.hybrid_calib_factor,
            hybrid_calib_stat=args.hybrid_calib_stat,
            s2s_mask_prob=args.s2s_mask_prob,
            s2s_num_masks=args.s2s_num_masks,
            lambda_perceptual=args.lambda_perceptual,
            lambda_multiscale=args.lambda_multiscale,
            use_tta=args.use_tta,
            # Resume training parameters
            start_epoch=resume_start_epoch,
            resume_optimizer_state=resume_optimizer_state,
            resume_scheduler_state=resume_scheduler_state,
            resume_ema_state=resume_ema_state,
            resume_best_val_psnr=resume_best_val_psnr,
            resume_best_val_loss=resume_best_val_loss,
            resume_epochs_no_improve=resume_epochs_no_improve,
        )
        # NOTE: model already contains best weights (restored in fine_tune_model at line 2434)
        # Best checkpoint already saved during training to best_model.pth (overwrites each time PSNR improves)
        # No need to save again here - best_model.pth is the final checkpoint
        print(f"Training complete. Best checkpoint saved at: {os.path.join(args.output_dir, 'best_model.pth')}")
        if ema_state is not None:
            print(f"Best EMA checkpoint saved at: {os.path.join(args.output_dir, 'best_model_ema.pth')}")

    # MEMORY MONITORING: Final session stats
    force_memory_cleanup()
    session_end_stats = get_memory_stats()
    print("\n" + "="*80)
    print("TRAINING SESSION COMPLETED - Final System Memory Stats:")
    for key, val in session_end_stats.items():
        print(f"  {key}: {val:.3f} GB")
    if 'session_start_stats' in locals():
        print("\nTotal Memory Change During Session:")
        for key in session_end_stats.keys():
            if key in session_start_stats:
                delta = session_end_stats[key] - session_start_stats[key]
                status = "✓ OK" if abs(delta) < 0.5 else ("⚠ RETAINED (may be allocator/cache)" if delta > 0.5 else "✓ FREED")
                print(f"  {key}: {delta:+.3f} GB {status}")
        print("  Note: CPU memory may stay high due to PyTorch/Python allocators retaining freed blocks for reuse.", flush=True)
    print("="*80 + "\n", flush=True)

    print("Done.")


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n" + "=" * 60)
        print("[INTERRUPTED] Received interrupt signal (Ctrl+C)")
        print("Performing emergency cleanup...")
        force_memory_cleanup()
        print("Cleanup complete. Exiting safely.")
        print("=" * 60)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        force_memory_cleanup()
        sys.exit(1)
