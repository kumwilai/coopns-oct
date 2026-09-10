"""Shared OCT dataset loader with realistic synthetic noise parameters"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from nsnd.training.synthetic_noise import OCTNoiseGenerator


class OCTSyntheticNoiseDataset(Dataset):
    """
    Dataset with realistic synthetic noise using exact parameters from synthetic_noise.py

    Noise parameter ranges (realistic setting):
    - Speckle: k=1.5-4.0, depth_gain=0.4-1.0
    - Banding: freq={15,20,30,40}, amp=0.06-0.12
    - Gaussian: sigma=0.02-0.06, depth_gain=0.2-0.8
    - Shot: peak=30-120, depth_gain=0.4-1.0
    - Depth profile: enabled
    """

    def __init__(
        self,
        data_root: Path,
        split: str = 'train',
        max_samples: int = 200,
        crop_size: int = 64,
        alpha: float = 0.2,
        alpha_vec: list[float] | None = None,
        param_scale: float = 1.0,
        random_crop: bool = False,
        use_depth_profile: bool = True,
        return_weights: bool = False,
        return_params: bool = False,
        pure_noise_prob: float = 0.0,
        pure_noise_epsilon: float = 0.02,
        speckle_base_k: float = 4.0,
        speckle_snr_factor: float = 0.5,
        speckle_correlation: float = 1.2,
        use_signal_dependent_speckle: bool = True,
        return_ids: bool = False,
    ):
        self.crop_size = int(crop_size)
        self.alpha = float(alpha)
        self.alpha_vec = None
        if alpha_vec:
            if len(alpha_vec) != 4:
                raise ValueError("alpha_vec must have 4 values.")
            self.alpha_vec = np.array(alpha_vec, dtype=np.float32)
        self.param_scale = float(param_scale)
        self.random_crop = bool(random_crop)
        self.use_depth_profile = use_depth_profile
        self.return_weights = bool(return_weights)
        self.return_params = bool(return_params)
        self.pure_noise_prob = float(pure_noise_prob)
        self.pure_noise_epsilon = float(pure_noise_epsilon)
        self.return_ids = bool(return_ids)
        self.speckle_base_k = float(speckle_base_k)
        self.speckle_snr_factor = float(speckle_snr_factor)
        self.speckle_correlation = float(speckle_correlation)
        self.use_signal_dependent_speckle = bool(use_signal_dependent_speckle)

        # Initialize noise generator with realistic parameters
        self.noise_gen = OCTNoiseGenerator(
            speckle_base_k=self.speckle_base_k,
            speckle_snr_factor=self.speckle_snr_factor,
            speckle_correlation=self.speckle_correlation,
            use_signal_dependent_speckle=self.use_signal_dependent_speckle,
        )
        self.clean_paths = []

        for pathology in ['cnv', 'dme', 'drusen', 'normal']:
            clean_dir = data_root / pathology / split / 'clean'
            if not clean_dir.exists():
                continue
            files = sorted(clean_dir.glob('*.png'))[:max_samples // 4]
            self.clean_paths.extend(files)

        alpha_desc = f"alpha_vec={self.alpha_vec.tolist()}" if self.alpha_vec is not None else f"alpha={alpha}"
        print(f"{split}: {len(self.clean_paths)} images "
              f"(Dirichlet {alpha_desc}, param_scale={param_scale}, depth_profile={use_depth_profile})")

    def __len__(self):
        return len(self.clean_paths)

    def sample_noise_params(self):
        """Sample realistic noise parameters with optional scaling"""
        if self.use_signal_dependent_speckle:
            base_k = self.speckle_base_k / max(self.param_scale, 1e-6)
            speckle_k = base_k
        else:
            speckle_k = np.random.uniform(1.5, 4.0) / self.param_scale
            base_k = speckle_k
        params = {
            # Speckle parameters
            'speckle_k': speckle_k,
            'speckle_base_k': base_k,
            'speckle_snr_factor': self.speckle_snr_factor,
            'speckle_correlation': self.speckle_correlation,
            'use_signal_dependent_speckle': self.use_signal_dependent_speckle,
            'speckle_depth_gain': np.random.uniform(0.4, 1.0),

            # Banding parameters
            'banding_freq': np.random.choice([15, 20, 30, 40]),
            'banding_amp': np.random.uniform(0.06, 0.12) * self.param_scale,

            # Gaussian parameters
            'gaussian_sigma': np.random.uniform(0.02, 0.06) * self.param_scale,
            'gaussian_depth_gain': np.random.uniform(0.2, 0.8),

            # Shot noise parameters
            'shot_peak': np.random.uniform(30, 120) / self.param_scale,
            'shot_depth_gain': np.random.uniform(0.4, 1.0),

            # Depth profile
            'use_depth_profile': self.use_depth_profile,
        }
        return params

    def _param_ranges(self):
        scale = self.param_scale
        return {
            'speckle_k': (1.5 / scale, 4.0 / scale),
            'banding_amp': (0.06 * scale, 0.12 * scale),
            'gaussian_sigma': (0.02 * scale, 0.06 * scale),
            'shot_peak': (30.0 / scale, 120.0 / scale),
        }

    @staticmethod
    def _normalize(value: float, vmin: float, vmax: float) -> float:
        return float((value - vmin) / (max(vmax - vmin, 1e-8)))

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        # Crop strategy: random for train, center for val
        h, w = clean.shape
        if h >= self.crop_size and w >= self.crop_size:
            if self.random_crop:
                top = np.random.randint(0, h - self.crop_size + 1)
                left = np.random.randint(0, w - self.crop_size + 1)
            else:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
            clean = clean[top:top + self.crop_size, left:left + self.crop_size]
        else:
            img_pil = Image.fromarray((clean * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            clean = np.array(img_pil, dtype=np.float32) / 255.0

        clean = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()

        # Sample noise composition from Dirichlet
        if self.alpha_vec is None:
            alpha = np.full(4, self.alpha, dtype=np.float32)
        else:
            alpha = self.alpha_vec
        sample = np.random.dirichlet(alpha)
        if self.pure_noise_prob > 0 and np.random.rand() < self.pure_noise_prob:
            idx = np.random.randint(0, 4)
            eps = min(max(self.pure_noise_epsilon, 0.0), 0.25)
            sample = np.full(4, eps, dtype=np.float32)
            sample[idx] = 1.0 - eps * 3.0
        weights = {
            'speckle': float(sample[0]),
            'banding': float(sample[1]),
            'gaussian': float(sample[2]),
            'shot': float(sample[3]),
        }

        # Sample noise parameters
        noise_params = self.sample_noise_params()

        # Generate noisy image with realistic parameters
        noisy, _, _ = self.noise_gen.generate(clean, weights=weights, params=noise_params)

        true_weights = torch.tensor(sample, dtype=torch.float32)
        if self.return_params:
            ranges = self._param_ranges()
            true_params = torch.tensor(
                [
                    self._normalize(noise_params['speckle_k'], *ranges['speckle_k']),
                    self._normalize(noise_params['banding_amp'], *ranges['banding_amp']),
                    self._normalize(noise_params['gaussian_sigma'], *ranges['gaussian_sigma']),
                    self._normalize(noise_params['shot_peak'], *ranges['shot_peak']),
                ],
                dtype=torch.float32,
            )
            if self.return_weights:
                out = (noisy[0], clean[0], true_weights, true_params)
            else:
                out = (noisy[0], clean[0], true_params)
        elif self.return_weights:
            out = (noisy[0], clean[0], true_weights)
        else:
            out = (noisy[0], clean[0])
        if self.return_ids:
            return (*out, str(clean_path))
        return out


def _param_ranges(scale: float) -> dict:
    return {
        'speckle_k': (1.5 / scale, 4.0 / scale),
        'banding_amp': (0.06 * scale, 0.12 * scale),
        'gaussian_sigma': (0.02 * scale, 0.06 * scale),
        'shot_peak': (30.0 / scale, 120.0 / scale),
    }


class PairedOCTCropDataset(Dataset):
    """Paired noisy-clean dataset with shared cropping and optional maps."""

    def __init__(
        self,
        pairs: str,
        crop_size: int = 64,
        random_crop: bool = False,
        return_weights: bool = False,
        return_params: bool = False,
        return_noise_maps: bool = False,
        weights_jsonl: str | None = None,
        param_scale: float = 1.0,
        expect_clean: bool = True,
        return_ids: bool = False,
        max_samples: int | None = None,
    ):
        self.crop_size = int(crop_size)
        self.random_crop = bool(random_crop)
        self.return_weights = bool(return_weights)
        self.return_params = bool(return_params)
        self.return_noise_maps = bool(return_noise_maps)
        self.param_scale = float(param_scale)
        self.return_ids = bool(return_ids)

        self.pairs = self._load_pairs(pairs, expect_clean=expect_clean)
        if max_samples is not None and max_samples > 0:
            self.pairs = self.pairs[:max_samples]
        if not self.pairs:
            raise ValueError("Empty pairs list.")

        self.weights_map = {}
        self.params_map = {}
        self.maps_map = {}
        if weights_jsonl:
            self.weights_map, self.params_map, self.maps_map = self._load_weights_jsonl(weights_jsonl)

        if self.return_weights and not self.weights_map:
            raise ValueError("weights_jsonl is required to return weights.")
        if self.return_params and not self.params_map:
            raise ValueError("weights_jsonl with params is required to return params.")
        if self.return_noise_maps and not self.maps_map:
            raise ValueError("weights_jsonl with noise_maps_path is required to return noise maps.")

        if self.return_weights:
            missing = [p for p, _ in self.pairs if p not in self.weights_map]
            if missing:
                raise ValueError(f"weights_jsonl missing {len(missing)} noisy paths.")
        if self.return_params:
            missing = [p for p, _ in self.pairs if p not in self.params_map]
            if missing:
                raise ValueError(f"weights_jsonl missing params for {len(missing)} noisy paths.")
        if self.return_noise_maps:
            missing = [p for p, _ in self.pairs if p not in self.maps_map]
            if missing:
                raise ValueError(f"weights_jsonl missing noise maps for {len(missing)} noisy paths.")

    @staticmethod
    def _looks_clean(path: str) -> bool:
        p = path.lower()
        return any(token in p for token in ("clean", "gt", "target", "label"))

    @staticmethod
    def _looks_noisy(path: str) -> bool:
        p = path.lower()
        return any(token in p for token in ("noisy", "noise", "corrupt", "input"))

    @staticmethod
    def _resolve_path(raw: str, base_dir: Path) -> str:
        p = Path(raw)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p)

    def _load_pairs(self, pairs_path: str, expect_clean: bool) -> list[tuple[str, str]]:
        path = Path(pairs_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pairs list file not found: {pairs_path}")
        base_dir = path.parent
        parsed = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.lstrip().startswith("#"):
                    continue
                if "," in line:
                    toks = [t.strip() for t in line.split(",", 1)]
                else:
                    toks = line.split()
                if len(toks) != 2 and expect_clean:
                    raise ValueError("Each line must have two paths.")
                if len(toks) < 2:
                    continue
                noisy_path, clean_path = toks[0], toks[1]
                if expect_clean and self._looks_clean(noisy_path) and self._looks_noisy(clean_path):
                    noisy_path, clean_path = clean_path, noisy_path
                noisy_path = self._resolve_path(noisy_path, base_dir)
                clean_path = self._resolve_path(clean_path, base_dir)
                parsed.append((noisy_path, clean_path))
        return parsed

    def _load_weights_jsonl(
        self, weights_jsonl: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
        weights_map: dict[str, torch.Tensor] = {}
        params_map: dict[str, torch.Tensor] = {}
        maps_map: dict[str, str] = {}
        ranges = _param_ranges(self.param_scale)
        base_dir = Path(weights_jsonl).parent

        def _norm_scalar(value: float, vmin: float, vmax: float) -> float:
            denom = max(vmax - vmin, 1e-6)
            return float(min(1.0, max(0.0, (value - vmin) / denom)))

        with open(weights_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                noisy = rec.get("noisy") or rec.get("noisy_path") or rec.get("input")
                weights = rec.get("weights")
                params = rec.get("params")
                maps_path = rec.get("noise_maps_path") or rec.get("noise_map_path") or rec.get("noise_maps")
                if not noisy:
                    continue
                noisy_path = Path(noisy)
                if not noisy_path.is_absolute():
                    noisy_path = (base_dir / noisy_path).resolve()
                else:
                    noisy_path = noisy_path.resolve()
                noisy_path = str(noisy_path)
                if weights:
                    weights_map[noisy_path] = torch.tensor(
                        [
                            weights["speckle"],
                            weights["banding"],
                            weights["gaussian"],
                            weights["shot"],
                        ],
                        dtype=torch.float32,
                    )
                if params:
                    try:
                        params_map[noisy_path] = torch.tensor(
                            [
                                _norm_scalar(float(params["speckle_k"]), *ranges["speckle_k"]),
                                _norm_scalar(float(params["banding_amp"]), *ranges["banding_amp"]),
                                _norm_scalar(float(params["gaussian_sigma"]), *ranges["gaussian_sigma"]),
                                _norm_scalar(float(params["shot_peak"]), *ranges["shot_peak"]),
                            ],
                            dtype=torch.float32,
                        )
                    except KeyError:
                        continue
                if maps_path:
                    maps_file = Path(maps_path)
                    if not maps_file.is_absolute():
                        maps_file = (base_dir / maps_file).resolve()
                    else:
                        maps_file = maps_file.resolve()
                    maps_map[noisy_path] = str(maps_file)
        return weights_map, params_map, maps_map

    def _load_noise_maps(self, path: str) -> np.ndarray:
        maps_path = Path(path)
        if not maps_path.exists():
            raise FileNotFoundError(f"Noise map file not found: {maps_path}")
        if maps_path.suffix.lower() == ".npz":
            data = np.load(maps_path)
            if "maps" in data:
                maps = data["maps"]
            else:
                maps = np.stack(
                    [
                        data["speckle"],
                        data["banding"],
                        data["gaussian"],
                        data["shot"],
                    ],
                    axis=0,
                )
        else:
            maps = np.load(maps_path)
        if maps.ndim != 3 or maps.shape[0] != 4:
            raise ValueError(f"Noise maps must have shape (4, H, W). Got {maps.shape}")
        return maps.astype(np.float32)

    def _crop_pair(self, noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = noisy.shape
        if h >= self.crop_size and w >= self.crop_size:
            if self.random_crop:
                top = np.random.randint(0, h - self.crop_size + 1)
                left = np.random.randint(0, w - self.crop_size + 1)
            else:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
            noisy = noisy[top:top + self.crop_size, left:left + self.crop_size]
            clean = clean[top:top + self.crop_size, left:left + self.crop_size]
        else:
            noisy_img = Image.fromarray((noisy * 255.0).astype(np.uint8))
            clean_img = Image.fromarray((clean * 255.0).astype(np.uint8))
            noisy_img = noisy_img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            clean_img = clean_img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            noisy = np.array(noisy_img, dtype=np.float32) / 255.0
            clean = np.array(clean_img, dtype=np.float32) / 255.0
        return noisy, clean

    def _crop_triplet(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        maps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = noisy.shape
        if h >= self.crop_size and w >= self.crop_size:
            if self.random_crop:
                top = np.random.randint(0, h - self.crop_size + 1)
                left = np.random.randint(0, w - self.crop_size + 1)
            else:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
            noisy = noisy[top:top + self.crop_size, left:left + self.crop_size]
            clean = clean[top:top + self.crop_size, left:left + self.crop_size]
            maps = maps[:, top:top + self.crop_size, left:left + self.crop_size]
            return noisy, clean, maps

        noisy_img = Image.fromarray((noisy * 255.0).astype(np.uint8))
        clean_img = Image.fromarray((clean * 255.0).astype(np.uint8))
        noisy_img = noisy_img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        clean_img = clean_img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        noisy = np.array(noisy_img, dtype=np.float32) / 255.0
        clean = np.array(clean_img, dtype=np.float32) / 255.0

        maps_t = torch.from_numpy(maps).unsqueeze(0)
        maps_t = F.interpolate(
            maps_t,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        maps = maps_t.squeeze(0).numpy().astype(np.float32)
        return noisy, clean, maps

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        noisy = np.array(Image.open(noisy_path).convert("L"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
        maps = None
        if self.return_noise_maps:
            maps = self._load_noise_maps(self.maps_map[noisy_path])
            noisy, clean, maps = self._crop_triplet(noisy, clean, maps)
        else:
            noisy, clean = self._crop_pair(noisy, clean)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()

        if self.return_params:
            params = self.params_map[noisy_path]
            if self.return_weights:
                out = (noisy_t, clean_t, self.weights_map[noisy_path], params)
            else:
                out = (noisy_t, clean_t, params)
        elif self.return_weights:
            out = (noisy_t, clean_t, self.weights_map[noisy_path])
        else:
            out = (noisy_t, clean_t)
        if self.return_noise_maps and maps is not None:
            maps_t = torch.from_numpy(maps).float()
            out = (*out, maps_t)
        if self.return_ids:
            return (*out, str(noisy_path))
        return out
