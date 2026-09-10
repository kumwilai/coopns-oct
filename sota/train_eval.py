import os
import time
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from adaptive_oct_denoise import (
    PairedOCTDataset,
    resize_to,
    compute_psnr,
    compute_ssim,
)

from .data import MultiFrameRepeatDataset
from .models.drunet import DRUNet
from .models.swinir_lite import SwinIRLite
from .models.blindspot_unet import BlindSpotUNet
from .models.speckle2speckle import Speckle2Speckle


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((x - y) ** 2 + eps * eps))


def gradient_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx_p, gy_p = F.conv2d(x, sobel_x, padding=1), F.conv2d(x, sobel_y, padding=1)
    gx_t, gy_t = F.conv2d(y, sobel_x, padding=1), F.conv2d(y, sobel_y, padding=1)
    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(img[:, :, 1:, :], img[:, :, :-1, :]) + F.l1_loss(img[:, :, :, 1:], img[:, :, :, :-1])


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == 'drunet':
        return DRUNet()
    if name == 'nafnet':
        # Lazy import to avoid hard dependency if NAFBackbone is missing
        from .models.nafnet_wrapper import NAFNetGray
        return NAFNetGray(base_channels=32)
    if name == 'swinir':
        return SwinIRLite()
    if name == 'noise2void':
        return BlindSpotUNet(base=32)
    if name == 'speckle2speckle':
        return Speckle2Speckle(base=32)
    raise ValueError(f'Unknown model: {name}')


def evaluate_pairs(model: nn.Module, pairs_file: str, size: int = 64) -> Tuple[float, float, float]:
    tfm = resize_to((size, size))
    ds = PairedOCTDataset(pairs_file, transform=tfm)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_images = len(ds)
    print(f"[Evaluation] Processing {total_images} images...")

    ps_list, ss_list = [], []
    t0 = time.time()
    with torch.no_grad():
        for idx, (x_noisy, y_clean) in enumerate(dl):
            x_noisy = x_noisy.to(device)
            pred = model(x_noisy).detach().cpu()
            ps_list.append(compute_psnr(pred, y_clean))
            ss_list.append(compute_ssim(pred, y_clean))

            # Print progress every 500 images or at the end
            if (idx + 1) % 500 == 0 or (idx + 1) == total_images:
                progress_pct = 100.0 * (idx + 1) / total_images
                avg_psnr = sum(ps_list) / len(ps_list)
                avg_ssim = sum(ss_list) / len(ss_list)
                print(f"  [{idx+1:>4}/{total_images}] ({progress_pct:>5.1f}%) - Running avg: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}", flush=True)

    dt = (time.time() - t0) * 1000.0
    n = max(1, len(ds))
    final_psnr = float(sum(ps_list) / len(ps_list))
    final_ssim = float(sum(ss_list) / len(ss_list))

    print(f"[Evaluation] Complete - Final PSNR: {final_psnr:.2f} dB, SSIM: {final_ssim:.4f}\n", flush=True)

    return final_psnr, final_ssim, float(dt / n)


def _train_supervised(
    model: nn.Module,
    pairs_file: str,
    size: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    grad_w: float = 0.05,
    tv_w: float = 1e-6,
    log_domain: bool = True,
):
    tfm = resize_to((size, size))
    ds = PairedOCTDataset(pairs_file, transform=tfm)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    total_batches = len(dl)
    print(f"[Supervised] Training on {len(ds)} pairs, {total_batches} batches per epoch")
    print(f"[Supervised] Progress will be shown every 50 batches")

    for ep in range(1, epochs + 1):
        running = []
        for batch_idx, (x_noisy, y_clean) in enumerate(dl):
            x_noisy = x_noisy.to(device)
            y_clean = y_clean.to(device)
            pred = model(x_noisy)
            pred_t, tgt_t = (torch.log(pred.clamp(min=1e-6)), torch.log(y_clean.clamp(min=1e-6))) if log_domain else (pred, y_clean)
            loss = charbonnier(pred_t, tgt_t)
            if grad_w > 0:
                loss = loss + grad_w * gradient_loss(pred_t, tgt_t)
            if tv_w > 0:
                loss = loss + tv_w * tv_loss(pred)
            opt.zero_grad(); loss.backward(); opt.step()
            running.append(loss.item())

            # Print progress every 50 batches (much more frequent)
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = sum(running[-50:]) / len(running[-50:])
                progress_pct = 100.0 * (batch_idx + 1) / total_batches
                print(f"  Epoch {ep}/{epochs} [{batch_idx+1:>4}/{total_batches}] ({progress_pct:>5.1f}%) loss={avg_loss:.4f}", flush=True)

        epoch_loss = sum(running) / max(1, len(running))
        print(f"[Supervised] Epoch {ep}/{epochs} completed - avg loss={epoch_loss:.4f}\n", flush=True)


class SingleNoisyDataset(Dataset):
    def __init__(self, pairs_file: str, size: int, noise_aug: bool = False):
        self.base = PairedOCTDataset(pairs_file, transform=resize_to((size, size)))
        self.noise_aug = noise_aug
        self.size = size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_noisy, y_clean = self.base[idx]
        if self.noise_aug:
            # re-synthesize noisy from clean to simulate independent speckle
            clean = y_clean.unsqueeze(0)
            from adaptive_oct_denoise import add_rayleigh_noise
            noisy1 = add_rayleigh_noise(clean)
            noisy2 = add_rayleigh_noise(clean)
            return noisy1.squeeze(0), noisy2.squeeze(0)
        return x_noisy, y_clean


def _train_speckle2speckle(
    model: nn.Module,
    pairs_file: str,
    size: int = 64,
    epochs: int = 6,
    lr: float = 1e-3,
    grad_w: float = 0.05,
    tv_w: float = 1e-6,
    log_domain: bool = True,
    repeats_file: Optional[str] = None,
):
    # Prefer real repeats if provided; else synthesize independent speckle from clean
    if repeats_file and os.path.isfile(repeats_file):
        with open(repeats_file, "r") as f:
            stack_dirs: List[str] = [l.strip() for l in f if l.strip()]
        ds = MultiFrameRepeatDataset(stack_dirs, size=size, register=True)
    else:
        ds = SingleNoisyDataset(pairs_file, size=size, noise_aug=True)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    total_batches = len(dl)
    print(f"[Speckle2Speckle] Training on {len(ds)} pairs, {total_batches} batches per epoch")
    print(f"[Speckle2Speckle] Progress will be shown every 50 batches")

    for ep in range(1, epochs + 1):
        running = []
        for batch_idx, (noisy1, noisy2) in enumerate(dl):
            noisy1 = noisy1.to(device); noisy2 = noisy2.to(device)
            pred = model(noisy1)
            pred_t, tgt_t = (torch.log(pred.clamp(min=1e-6)), torch.log(noisy2.clamp(min=1e-6))) if log_domain else (pred, noisy2)
            loss = charbonnier(pred_t, tgt_t)
            if grad_w > 0:
                loss = loss + grad_w * gradient_loss(pred_t, tgt_t)
            if tv_w > 0:
                loss = loss + tv_w * tv_loss(pred)
            opt.zero_grad(); loss.backward(); opt.step()
            running.append(loss.item())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = sum(running[-50:]) / len(running[-50:])
                progress_pct = 100.0 * (batch_idx + 1) / total_batches
                print(f"  Epoch {ep}/{epochs} [{batch_idx+1:>4}/{total_batches}] ({progress_pct:>5.1f}%) loss={avg_loss:.4f}", flush=True)

        epoch_loss = sum(running) / max(1, len(running))
        print(f"[Speckle2Speckle] Epoch {ep}/{epochs} completed - avg loss={epoch_loss:.4f}\n", flush=True)


def _mask_random(x: torch.Tensor, mask_ratio: float = 0.05):
    """Create random mask for Noise2Void blind-spot training.

    Note: True Noise2Void uses architectural blind-spot (receptive field masking).
    This is a simplified version using random pixel masking.
    """
    B, C, H, W = x.shape
    mask = (torch.rand(B, 1, H, W, device=x.device) < mask_ratio).float()
    return x, mask  # Don't mask input, just return mask for loss computation


def _train_noise2void(
    model: nn.Module,
    pairs_file: str,
    size: int = 64,
    epochs: int = 6,
    lr: float = 1e-3,
    mask_ratio: float = 0.07,
    tv_w: float = 1e-6,
    log_domain: bool = True,
):
    """Train using Noise2Void strategy: predict masked pixels from neighbors.

    Note: BlindSpotUNet model should have blind-spot architecture built-in.
    This training uses random pixel masking for the loss.
    """
    ds = SingleNoisyDataset(pairs_file, size=size, noise_aug=False)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    total_batches = len(dl)
    print(f"[Noise2Void] Training on {len(ds)} pairs, {total_batches} batches per epoch")
    print(f"[Noise2Void] Progress will be shown every 50 batches")

    for ep in range(1, epochs + 1):
        running = []
        for batch_idx, (noisy, _) in enumerate(dl):
            noisy = noisy.to(device)
            noisy_in, mask = _mask_random(noisy, mask_ratio=mask_ratio)
            pred = model(noisy_in)
            pred_t, tgt_t = (torch.log(pred.clamp(min=1e-6)), torch.log(noisy.clamp(min=1e-6))) if log_domain else (pred, noisy)
            # Only compute loss on masked pixels (blind-spot principle)
            loss = (torch.abs(pred_t - tgt_t) * mask).sum() / (mask.sum() + 1e-6)
            if tv_w > 0:
                loss = loss + tv_w * tv_loss(pred)
            opt.zero_grad(); loss.backward(); opt.step()
            running.append(loss.item())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = sum(running[-50:]) / len(running[-50:])
                progress_pct = 100.0 * (batch_idx + 1) / total_batches
                print(f"  Epoch {ep}/{epochs} [{batch_idx+1:>4}/{total_batches}] ({progress_pct:>5.1f}%) loss={avg_loss:.4f}", flush=True)

        epoch_loss = sum(running) / max(1, len(running))
        print(f"[Noise2Void] Epoch {ep}/{epochs} completed - avg loss={epoch_loss:.4f}\n", flush=True)


def train_and_eval(
    model_name: str,
    pairs_file: str,
    size: int = 64,
    out_dir: str = 'outputs/sota_64',
    epochs: int = 8,
    repeats_file: Optional[str] = None,
) -> Dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model = build_model(model_name)
    name = model_name.lower()
    if name in ['drunet', 'nafnet', 'swinir']:
        _train_supervised(model, pairs_file, size=size, epochs=epochs)
    elif name == 'noise2void':
        _train_noise2void(model, pairs_file, size=size, epochs=max(6, epochs))
    elif name == 'speckle2speckle':
        _train_speckle2speckle(model, pairs_file, size=size, epochs=max(6, epochs), repeats_file=repeats_file)
    ps, ss, ms = evaluate_pairs(model, pairs_file, size=size)
    torch.save(model.state_dict(), os.path.join(out_dir, f'{name}.pth'))
    res = {'model': name, 'psnr': ps, 'ssim': ss, 'time_ms': ms}
    with open(os.path.join(out_dir, f'{name}_metrics.json'), 'w') as f:
        import json; json.dump(res, f, indent=2)
    return res
