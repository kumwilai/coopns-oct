import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.io import read_image
import torchvision.transforms as T

from .data_setup import load_splits

CLASSES = ['cnv', 'dme', 'drusen', 'normal']

def _to_gray01(img: torch.Tensor) -> torch.Tensor:
    if img.dtype != torch.float32:
        img = img.float() / 255.0
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if img.shape[0] == 1:
        return img
    r, g, b = img[0:1], img[1:2], img[2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.clamp(0.0, 1.0)

def gather_clean_paths(
    split_file: str,
    split: str = 'train',
    classes: Optional[List[str]] = None
) -> List[str]:
    """Gathers clean image paths from a pre-computed splits file."""
    classes = classes or CLASSES
    splits = load_splits(split_file)
    
    paths = []
    if split not in splits:
        raise ValueError(f"Split '{split}' not found in splits file.")
    
    for cls in classes:
        if cls in splits[split]:
            paths.extend(splits[split][cls])
    
    return paths

def gather_pairs(
    split_file: str,
    split: str = 'val',
    classes: Optional[List[str]] = None,
    noisy_folder: str = 'noisy_gaussian'
) -> List[Tuple[str, str]]:
    """
    Return (noisy, clean) pairs by using a splits file and deriving noisy paths.
    """
    classes = classes or CLASSES
    clean_paths = gather_clean_paths(split_file, split, classes)
    
    pairs: List[Tuple[str, str]] = []
    for clean_path_str in clean_paths:
        clean_path = Path(clean_path_str)
        filename = clean_path.name
        
        # This part of the path is tricky. Let's find the 'train' or 'val' part
        try:
            split_name_part = 'train' if 'train' in clean_path.parts else 'val'
            # Find parent directory that is the class name
            parent_dir = clean_path.parent
            while parent_dir.name not in CLASSES:
                parent_dir = parent_dir.parent
            
            noisy_path = parent_dir / split_name_part / noisy_folder / filename
        
            if noisy_path.is_file():
                pairs.append((str(noisy_path), clean_path_str))

        except Exception:
             # Fallback for paths that might not conform to the expected structure
            noisy_path_alt = clean_path.parent.parent / noisy_folder / filename
            if noisy_path_alt.is_file():
                pairs.append((str(noisy_path_alt), clean_path_str))

    return pairs


def _compute_sampling_weights(
    img: torch.Tensor,
    depth_boost: float,
    edge_boost: float,
) -> torch.Tensor:
    """
    Build a spatial weight map that prefers deeper rows and edge-rich regions.
    img: [1,H,W] in [0,1]
    """
    _, H, W = img.shape
    weight = torch.ones((H, W), device=img.device, dtype=img.dtype)

    if depth_boost > 1.0:
        depth_weights = torch.linspace(1.0, depth_boost, H, device=img.device, dtype=img.dtype).view(H, 1)
        weight = weight * depth_weights

    if edge_boost > 0.0:
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        gx = F.conv2d(img.unsqueeze(0), sobel_x, padding=1)
        gy = F.conv2d(img.unsqueeze(0), sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0).squeeze(0)
        grad_norm = grad_mag / (grad_mag.max() + 1e-6)
        weight = weight * (1.0 + edge_boost * grad_norm)

    weight = weight.clamp(min=1e-6)
    return weight


def _sample_patch_coords(weight: torch.Tensor, patch_size: int) -> Tuple[int, int]:
    """Sample a patch top-left coordinate given a weight map."""
    H, W = weight.shape
    flat_idx = torch.multinomial(weight.view(-1), num_samples=1).item()
    top = flat_idx // W
    left = flat_idx % W
    top = int(max(0, min(top, H - patch_size)))
    left = int(max(0, min(left, W - patch_size)))
    return top, left


def _maybe_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    """Upscale minimally if the image is smaller than the requested patch size."""
    _, H, W = img.shape
    if H < size or W < size:
        img = F.interpolate(img.unsqueeze(0), size=(max(size, H), max(size, W)), mode='bilinear', align_corners=False).squeeze(0)
    return img


class SyntheticNoiseDataset(Dataset):
    """
    Dataset for synthetic pre-training.
    Loads clean patches, applies aggressive augmentation, and adds a random mix of synthetic noise.
    Returns (noisy_patch, clean_patch) pairs.
    """
    def __init__(
        self,
        image_paths: List[str],
        size: int = 64,
        augment: bool = True,
        depth_boost: float = 1.0,
        edge_boost: float = 0.0,
        patches_per_image: int = 1,
        noise_config: Optional[Dict] = None,
    ):
        self.paths = image_paths
        self.size = size
        self.augment = augment
        self.depth_boost = depth_boost
        self.edge_boost = edge_boost
        self.patches_per_image = max(1, patches_per_image)
        self.noise_config = noise_config or {
            "prob_log": 0.5,
            "speckle_prob": 0.5,
            "speckle_sigma": (0.1, 0.3),
            "gaussian_prob": 0.3,
            "gaussian_sigma": (0.01, 0.10),
            "poisson_prob": 0.3,
            "poisson_peak": (20, 60),
        }
        self._base_repeats = 50

        if augment:
            # Per user spec, light OCT-aware augmentations
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=5, scale=(0.95, 1.05)),
                T.ElasticTransform(alpha=25.0, sigma=3.0),
                T.ColorJitter(brightness=0.05, contrast=0.0, saturation=0.0, hue=0.0), # brightness
                T.Lambda(lambda x: x * (0.9 + 0.2 * torch.rand(1))), # Gamma
            ])

    def __len__(self):
        # Allow for many epochs over the same data by returning a large number
        return len(self.paths) * self._base_repeats * self.patches_per_image

    def __getitem__(self, idx):
        # Modulo index to loop over the dataset
        img_path = self.paths[idx % len(self.paths)]
        
        clean_img = _to_gray01(read_image(img_path))
        clean_img = _maybe_resize(clean_img, self.size)

        # --- Patch Sampling (depth + edge aware) ---
        weights = _compute_sampling_weights(clean_img, self.depth_boost, self.edge_boost)
        top, left = _sample_patch_coords(weights, self.size)
        clean_patch = clean_img[:, top:top+self.size, left:left+self.size]

        if self.augment:
            clean_patch = self.aug_transform(clean_patch)
        
        # Clamp after augmentation to ensure valid range before adding noise
        clean_patch.clamp_(0.0, 1.0)

        noisy_patch = self._add_synthetic_noise(clean_patch)

        return noisy_patch, clean_patch

    def _add_synthetic_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Applies a random combination of noise types."""
        img = img.clone()
        cfg = self.noise_config

        # Homomorphic transform for multiplicative noise
        use_log = torch.rand(1).item() < cfg.get("prob_log", 0.5)
        if use_log:
            img = torch.log(img + 1e-6)

        # 1. Speckle (Gamma/Rayleigh) - simplified as multiplicative noise in log domain
        if torch.rand(1).item() < cfg.get("speckle_prob", 0.5):
            sigma_low, sigma_hi = cfg.get("speckle_sigma", (0.1, 0.3))
            noise = torch.randn_like(img) * torch.empty(1, device=img.device).uniform_(sigma_low, sigma_hi)
            img = img + noise

        # 2. Gaussian
        if torch.rand(1).item() < cfg.get("gaussian_prob", 0.3):
            sigma_low, sigma_hi = cfg.get("gaussian_sigma", (0.01, 0.10))
            noise = torch.randn_like(img) * torch.empty(1, device=img.device).uniform_(sigma_low, sigma_hi)
            img = img + noise
            
        # 3. Poisson
        if torch.rand(1).item() < cfg.get("poisson_prob", 0.3):
            # Poisson is signal-dependent. Higher signal -> more noise
            # Scale to a peak photon count, apply noise, then scale back.
            peak_low, peak_hi = cfg.get("poisson_peak", (20, 60))
            peak = torch.empty(1, device=img.device).uniform_(peak_low, peak_hi)
            if use_log:
                # This is tricky in log domain, approximate with signal-dependent Gaussian
                signal_dep_noise = torch.randn_like(img) * (img - img.min()).max() / peak
                img = img + signal_dep_noise
            else:
                 img = torch.poisson(img * peak) / peak

        if use_log:
            img = torch.exp(img).clamp(0.0, 1.0)
            
        return img.clamp(0.0, 1.0)


class PairsPatchDataset(Dataset):
    """Samples 64x64 patches from (noisy, clean) pairs."""
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        size: int = 64,
        augment: bool = True,
        depth_boost: float = 1.0,
        edge_boost: float = 0.0,
    ):
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.depth_boost = depth_boost
        self.edge_boost = edge_boost
        
        if augment:
            # Per user spec, light OCT-aware augmentations
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=5, scale=(0.95, 1.05)),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        np, cp = self.pairs[idx]
        noisy = _to_gray01(read_image(np))
        clean = _to_gray01(read_image(cp))

        noisy = _maybe_resize(noisy, self.size)
        clean = _maybe_resize(clean, self.size)

        # Use weights from clean target for stability
        weights = _compute_sampling_weights(clean, self.depth_boost, self.edge_boost)
        top, left = _sample_patch_coords(weights, self.size)
        noisy_patch = noisy[:, top:top+self.size, left:left+self.size]
        clean_patch = clean[:, top:top+self.size, left:left+self.size]

        if self.augment:
            # Apply same geometric transform to both
            stacked = torch.cat([noisy_patch, clean_patch], dim=0)
            stacked_aug = self.aug_transform(stacked)
            noisy_patch, clean_patch = torch.chunk(stacked_aug, 2, dim=0)
            
            # Separate intensity augs
            noisy_patch = (noisy_patch * (0.95 + 0.1 * torch.rand(1))).clamp(0, 1)
            # Don't augment brightness of clean target
            
        return noisy_patch, clean_patch


class MultiFrameRepeatDataset(Dataset):
    """
    For Speckle2Speckle: load N>=2 repeats from a stack directory, optionally register, then emit paired patches.
    Expects each item to be a directory containing multiple frames (sorted by name).
    """
    def __init__(
        self,
        stack_dirs: List[str],
        size: int = 64,
        augment: bool = True,
        max_frames: int = 2,
        register: bool = True,
        depth_boost: float = 1.0,
        edge_boost: float = 0.0,
    ):
        self.stack_dirs = stack_dirs
        self.size = size
        self.augment = augment
        self.max_frames = max_frames
        self.register = register
        self.depth_boost = depth_boost
        self.edge_boost = edge_boost

        if augment:
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=5, scale=(0.95, 1.05)),
            ])

    def __len__(self):
        return len(self.stack_dirs)

    def _load_frames(self, dir_path: str) -> List[torch.Tensor]:
        files = sorted(glob.glob(os.path.join(dir_path, "*.png")))[: self.max_frames]
        frames = [_to_gray01(read_image(f)) for f in files]
        return frames

    def _register(self, ref: torch.Tensor, mov: torch.Tensor) -> torch.Tensor:
        try:
            import cv2
        except ImportError:
            return mov  # fallback if OpenCV not installed

        r = (ref.squeeze().numpy() * 255).astype("uint8")
        m = (mov.squeeze().numpy() * 255).astype("uint8")
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(r, m, warp, cv2.MOTION_EUCLIDEAN, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
            aligned = cv2.warpAffine(m, warp, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            out = torch.from_numpy(aligned).float().unsqueeze(0) / 255.0
            return out.clamp(0, 1)
        except Exception:
            return mov

    def __getitem__(self, idx):
        frames = self._load_frames(self.stack_dirs[idx])
        if len(frames) < 2:
            raise ValueError(f"Not enough frames in {self.stack_dirs[idx]}")
        ref, mov = frames[0], frames[1]
        if self.register:
            mov = self._register(ref, mov)

        ref = _maybe_resize(ref, self.size)
        mov = _maybe_resize(mov, self.size)

        weights = _compute_sampling_weights(ref, self.depth_boost, self.edge_boost)
        top, left = _sample_patch_coords(weights, self.size)
        ref_patch = ref[:, top:top+self.size, left:left+self.size]
        mov_patch = mov[:, top:top+self.size, left:left+self.size]

        if self.augment:
            stacked = torch.cat([ref_patch, mov_patch], dim=0)
            stacked = self.aug_transform(stacked)
            ref_patch, mov_patch = torch.chunk(stacked, 2, dim=0)

        return ref_patch, mov_patch
