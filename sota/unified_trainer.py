import os
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .train_eval import build_model
from .data import gather_clean_paths, gather_pairs, CleanImagePatchDataset, PairsPatchDataset
from adaptive_oct_denoise import (
    add_rayleigh_noise,
    add_poisson_noise,
    add_mixed_gaussian_noise,
    add_gaussian_additive_noise,
    compute_psnr,
    compute_ssim,
)


def _synthetic_mixed(noisy_from: torch.Tensor) -> torch.Tensor:
    """Apply a random noise process to a clean tensor [B,1,H,W]."""
    r = torch.rand(1).item()
    if r < 0.25:
        return add_rayleigh_noise(noisy_from)
    elif r < 0.5:
        return add_poisson_noise(noisy_from, lambda_scale=1.0)
    elif r < 0.75:
        return add_mixed_gaussian_noise(noisy_from, (0.01, 0.10))
    else:
        return add_gaussian_additive_noise(noisy_from, (0.03, 0.10))


def _to_log(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def pretrain_synthetic(model: nn.Module, clean_paths: List[str], size: int = 64, epochs: int = 60, batch_size: int = 4, log_domain: bool = False) -> None:
    ds = CleanImagePatchDataset(clean_paths, size=size, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for clean in dl:
            clean = clean.to(dev)
            noisy = _synthetic_mixed(clean)
            pred = model(noisy)
            if log_domain:
                loss = F.smooth_l1_loss(_to_log(pred), _to_log(clean))
            else:
                loss = F.smooth_l1_loss(pred, clean)
            loss = loss + 0.05 * (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            del pred, loss, noisy; torch.cuda.empty_cache() if dev.type == 'cuda' else None


def finetune_real(model: nn.Module, pairs: List[tuple], size: int = 64, epochs: int = 30, batch_size: int = 4, log_domain: bool = False) -> None:
    ds = PairsPatchDataset(pairs, size=size, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    for _ in range(epochs):
        for noisy, clean in dl:
            noisy = noisy.to(dev); clean = clean.to(dev)
            pred = model(noisy)
            if log_domain:
                loss = F.smooth_l1_loss(_to_log(pred), _to_log(clean))
            else:
                loss = F.smooth_l1_loss(pred, clean)
            loss = loss + 0.05 * (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            del pred, loss; torch.cuda.empty_cache() if dev.type == 'cuda' else None


def eval_on_pairs(model: nn.Module, pairs: List[tuple], size: int = 64) -> Dict:
    ds = PairsPatchDataset(pairs, size=size, augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    model.eval()
    ps, ss = [], []
    with torch.no_grad():
        for noisy, clean in dl:
            noisy = noisy.to(dev)
            pred = model(noisy).cpu()
            ps.append(compute_psnr(pred, clean))
            ss.append(compute_ssim(pred, clean))
    import numpy as np
    return {'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss))}


def run_unified_training(methods: List[str], oct_root: str, classes: List[str], noisy_folder: str,
                         size: int = 64, out_dir: str = 'outputs/unified_64', pretrain_epochs: int = 80, finetune_epochs: int = 30,
                         log_domain: bool = False) -> List[Dict]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    clean_paths = gather_clean_paths(oct_root, split='train', classes=classes)
    pairs_val = gather_pairs(oct_root, split='val', classes=classes, noisy_folder=noisy_folder)
    results = []
    import gc
    for m in methods:
        print(f"\n=== {m} unified training at {size}x{size} ===")
        model = build_model(m)
        # Pretrain on synthetic from clean
        pretrain_synthetic(model, clean_paths, size=size, epochs=pretrain_epochs, batch_size=4, log_domain=log_domain)
        # Finetune on real noisy-clean pairs (if any)
        if len(pairs_val) > 0:
            finetune_real(model, pairs_val, size=size, epochs=finetune_epochs, batch_size=4, log_domain=log_domain)
        # Evaluate on the same val pairs
        metrics = eval_on_pairs(model, pairs_val, size=size)
        ckpt = Path(out_dir) / f'{m}_unified.pth'
        torch.save(model.state_dict(), ckpt)
        metrics.update({'model': m, 'ckpt': str(ckpt)})
        with open(Path(out_dir) / f'{m}_metrics.json', 'w') as f:
            import json; json.dump(metrics, f, indent=2)
        print(metrics)
        results.append(metrics)
        # Free model + optimizer memory before next method
        del model; gc.collect()
    # Save summary
    with open(Path(out_dir) / 'summary.json', 'w') as f:
        import json; json.dump(results, f, indent=2)
    return results
