"""
Evaluate adaptive denoiser vs baselines under unseen noise.

This script:
- Meta-trains on a subset of synthetic noise tasks (excludes a chosen unseen noise)
- Evaluates on synthetic OCT-like images corrupted by the unseen noise
- Compares PSNR/SSIM against simple baselines (noisy, box, gaussian, median)
- Optionally applies TTA per test image

Usage examples:
- CPU-quick sanity: python scripts/eval_unseen_noise.py --unseen gamma_speckle --epochs 1 --tasks 2 --inner_steps 2 --num_train 32 --num_test 16
- More thorough:   python scripts/eval_unseen_noise.py --unseen gaussian_add --epochs 2 --tasks 4 --inner_steps 4 --tta
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import copy
import torch
import torch.nn.functional as F

# Import components from the main module
import os, sys
# Ensure repo root (containing adaptive_oct_denoise.py) is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from adaptive_oct_denoise import (
    device,
    build_model,
    meta_train,
    evaluate_model_fast,
    compute_psnr,
    compute_ssim,
    _box_filter,
    _gaussian_filter,
    _median_filter_3x3,
    _synthetic_oct_like_batch,
    add_rayleigh_noise,
    add_poisson_noise,
    add_mixed_gaussian_noise,
    add_gaussian_additive_noise,
    add_gamma_speckle_noise,
    add_correlated_speckle,
    test_time_adaptation,
    get_memory_stats,
    force_memory_cleanup,
    set_seed,
)


class SyntheticCleanDataset(torch.utils.data.Dataset):
    """Yields synthetic OCT-like clean images [1,H,W] in [0,1]."""
    def __init__(self, n: int, h: int = 192, w: int = 256):
        self.n, self.h, self.w = n, h, w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _synthetic_oct_like_batch(1, self.h, self.w).squeeze(0)


def make_noise_registry() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "rayleigh": add_rayleigh_noise,
        "poisson": add_poisson_noise,
        "gaussian_mult": add_mixed_gaussian_noise,
        "gaussian_add": add_gaussian_additive_noise,
        "gamma_speckle": add_gamma_speckle_noise,
        "corr_speckle": add_correlated_speckle,
    }


def evaluate_on_unseen(
    model: torch.nn.Module,
    clean_batch: torch.Tensor,
    noise_fn: Callable[[torch.Tensor], torch.Tensor],
    tta: bool = False,
    tta_steps: int = 12,
    tta_lr: float = 6e-4,
) -> Dict[str, float]:
    """Evaluate model and baselines on a clean batch with a given unseen noise.

    Returns dict of average PSNR/SSIM per method.
    """
    clean = clean_batch.to(device)
    noisy = noise_fn(clean).clamp(0, 1)

    # Baselines
    def avg_psnr_ssim(pred):
        return compute_psnr(pred, clean), compute_ssim(pred, clean)

    results = {}
    results["Noisy"] = avg_psnr_ssim(noisy)
    results["Box3x3"] = avg_psnr_ssim(_box_filter(noisy, 3))
    results["Gaussian5x5"] = avg_psnr_ssim(_gaussian_filter(noisy, 5, 1.0))
    results["Median3x3"] = avg_psnr_ssim(_median_filter_3x3(noisy))

    # Model
    if tta:
        # Adapt per-image (per-sample) independently to avoid leakage
        psnrs, ssims = [], []
        for i in range(clean.shape[0]):
            x = noisy[i:i+1].detach().cpu()
            # Clone on CPU to keep memory contained
            m = type(model) if not isinstance(model, torch.nn.DataParallel) else model.module
            m = copy.deepcopy(model).cpu()
            # Enable grads inside TTA
            torch.set_grad_enabled(True)
            pred_i = test_time_adaptation(m, x, num_steps=tta_steps, lr=tta_lr)
            torch.set_grad_enabled(False)
            ps, ss = compute_psnr(pred_i, clean[i:i+1].cpu()), compute_ssim(pred_i, clean[i:i+1].cpu())
            psnrs.append(ps); ssims.append(ss)
            del m, pred_i
            force_memory_cleanup()
        results["Model (TTA)"] = (float(np.mean(psnrs)), float(np.mean(ssims)))
    else:
        with torch.no_grad():
            pred = model(noisy)
        results["Model"] = avg_psnr_ssim(pred)

    # Flatten to {method_psnr, method_ssim}
    out = {}
    for k, (p, s) in results.items():
        out[f"{k}_psnr"] = float(p)
        out[f"{k}_ssim"] = float(s)
    return out


def main():
    parser = argparse.ArgumentParser("Unseen-noise evaluation for adaptive OCT denoiser")
    parser.add_argument("--unseen", type=str, default="gamma_speckle",
                        choices=["rayleigh","poisson","gaussian_mult","gaussian_add","gamma_speckle","corr_speckle"],
                        help="Noise held out from meta-training and used only at test")
    parser.add_argument("--epochs", type=int, default=1, help="Meta epochs")
    parser.add_argument("--tasks", type=int, default=3, help="Tasks per meta epoch")
    parser.add_argument("--inner_steps", type=int, default=3, help="Inner steps per task")
    parser.add_argument("--inner_lr", type=float, default=2e-4, help="Inner-loop LR")
    parser.add_argument("--meta_step", type=float, default=0.2, help="Reptile meta step size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size of clean images")
    parser.add_argument("--num_train", type=int, default=64, help="# synthetic clean images for meta-training")
    parser.add_argument("--num_test", type=int, default=24, help="# synthetic clean images for evaluation")
    parser.add_argument("--h", type=int, default=192)
    parser.add_argument("--w", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--residual", action="store_true", help="Use residual prediction mode")
    parser.add_argument("--adapter", type=str, default="global", choices=["global","spatial","casa"])
    parser.add_argument("--tta", action="store_true", help="Apply TTA during evaluation")
    parser.add_argument("--tta_steps", type=int, default=12)
    parser.add_argument("--tta_lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Report memory
    print("=" * 80)
    print("Initial memory status:")
    for k, v in get_memory_stats().items():
        print(f"  {k}: {v:.2f} GB")
    print("=" * 80)

    # Prepare datasets/loaders
    train_ds = SyntheticCleanDataset(args.num_train, args.h, args.w)
    test_clean = _synthetic_oct_like_batch(args.num_test, args.h, args.w)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    # Build model
    model = build_model(base_channels=args.base_channels, residual_mode=args.residual, adapter_type=args.adapter)
    model.to(device)

    # Prepare noise tasks for meta (exclude unseen)
    reg = make_noise_registry()
    assert args.unseen in reg, f"Unknown unseen noise: {args.unseen}"
    noise_tasks = [{"name": k, "fn": v} for k, v in reg.items() if k != args.unseen]

    print(f"Meta-training on tasks: {[t['name'] for t in noise_tasks]} | holding out: {args.unseen}")
    meta_train(
        model=model,
        clean_loader=train_loader,
        noise_tasks=noise_tasks,
        num_meta_epochs=args.epochs,
        num_tasks_per_meta_batch=args.tasks,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        meta_step_size=args.meta_step,
        train_backbone_in_inner=False,
        use_charbonnier=False,
        lambda_grad=0.0,
        amp=False,
    )

    # Evaluate on unseen noise
    model.eval()
    with torch.no_grad():
        results = evaluate_on_unseen(
            model=model,
            clean_batch=test_clean,
            noise_fn=reg[args.unseen],
            tta=args.tta,
            tta_steps=args.tta_steps,
            tta_lr=args.tta_lr,
        )

    # Pretty print
    print("\n" + "=" * 80)
    print(f"UNSEEN-NOISE EVAL: {args.unseen}")
    print("=" * 80)
    header = f"{'Method':<18} {'PSNR (dB)':>12} {'SSIM':>12}"
    print(header)
    print("-" * len(header))

    methods = [
        ("Noisy", results.get("Noisy_psnr", 0.0), results.get("Noisy_ssim", 0.0)),
        ("Box3x3", results.get("Box3x3_psnr", 0.0), results.get("Box3x3_ssim", 0.0)),
        ("Gaussian5x5", results.get("Gaussian5x5_psnr", 0.0), results.get("Gaussian5x5_ssim", 0.0)),
        ("Median3x3", results.get("Median3x3_psnr", 0.0), results.get("Median3x3_ssim", 0.0)),
    ]
    model_key = "Model (TTA)" if args.tta else "Model"
    methods.append((model_key, results.get(f"{model_key}_psnr", 0.0), results.get(f"{model_key}_ssim", 0.0)))

    for name, psnr, ssim in methods:
        print(f"{name:<18} {psnr:>12.3f} {ssim:>12.4f}")

    # Advantage over best baseline
    base_psnrs = [psnr for name, psnr, _ in methods if name not in (model_key,)]
    base_names = [name for name, _, _ in methods if name not in (model_key,)]
    best_idx = int(np.argmax(base_psnrs))
    best_base_name = base_names[best_idx]
    best_base_psnr = base_psnrs[best_idx]
    model_psnr = results.get(f"{model_key}_psnr", 0.0)
    print("-" * len(header))
    print(f"Best baseline: {best_base_name} ({best_base_psnr:.3f} dB)")
    print(f"{model_key}: {model_psnr:.3f} dB | Advantage: {model_psnr - best_base_psnr:+.3f} dB")

    # Memory footer
    print("\nFinal memory status:")
    for k, v in get_memory_stats().items():
        print(f"  {k}: {v:.2f} GB")


if __name__ == "__main__":
    main()
