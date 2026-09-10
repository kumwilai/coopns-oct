#!/usr/bin/env python3
"""
Smart ensemble that adapts to noise characteristics.
Uses CASA for high-SNR, BM3D for low-SNR, weighted combination in between.
"""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False

from adaptive_oct_denoise import (
    PairedOCTDataset,
    build_model,
    compute_psnr,
    compute_ssim,
    device,
    resize_to,
)


def estimate_snr(image):
    """Estimate signal-to-noise ratio of the image."""
    # Simple SNR estimate: mean / std
    mean = image.mean()
    std = image.std()
    snr = mean / (std + 1e-6)
    return snr.item()


def smart_ensemble_denoise(model, noisy_batch, bm3d_sigma=0.02):
    """
    Smart ensemble:
    - High SNR (clean regions): Trust CASA more (0.8 CASA, 0.2 BM3D)
    - Medium SNR: Balanced (0.6 CASA, 0.4 BM3D)
    - Low SNR (very noisy): Trust BM3D more (0.4 CASA, 0.6 BM3D)
    """
    if not BM3D_AVAILABLE:
        with torch.no_grad():
            return model(noisy_batch)

    # CASA denoising
    with torch.no_grad():
        casa_output = model(noisy_batch)

    batch_size = noisy_batch.shape[0]
    results = []

    for i in range(batch_size):
        noisy_np = noisy_batch[i, 0].cpu().numpy()
        casa_np = casa_output[i, 0].cpu().numpy()

        # Estimate SNR
        snr = estimate_snr(noisy_batch[i])

        # Apply BM3D
        bm3d_output = bm3d.bm3d(noisy_np, sigma_psd=bm3d_sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)

        # Adaptive weighting based on SNR
        if snr > 10:  # High SNR - trust CASA
            weight_casa = 0.8
        elif snr > 5:  # Medium SNR - balanced
            weight_casa = 0.6
        else:  # Low SNR - trust BM3D
            weight_casa = 0.4

        # Weighted combination
        final = weight_casa * casa_np + (1 - weight_casa) * bm3d_output
        final = np.clip(final, 0.0, 1.0)

        results.append(torch.from_numpy(final))

    output = torch.stack(results).unsqueeze(1).to(device)
    return output


def evaluate_smart_ensemble(
    checkpoint_path: str,
    val_pairs: str,
    adapter: str = "casa",
    backbone: str = "noise2void",
    base_channels: int = 48,
    residual_mode: bool = True,
    bm3d_sigma: float = 0.02,
    output_json: str = None,
):
    if not BM3D_AVAILABLE:
        print("ERROR: BM3D not available. Install with: pip install bm3d")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Architecture: {adapter.upper()} + {backbone.upper()} | residual={residual_mode}")
    print(f"Smart Ensemble: SNR-adaptive CASA/BM3D weighting")

    # Build and load model
    model = build_model(
        base_channels=base_channels,
        residual_mode=residual_mode,
        adapter_type=adapter,
        backbone_type=backbone,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).eval()

    # Load validation data
    transform = resize_to((64, 64))
    val_dataset = PairedOCTDataset(val_pairs, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Evaluating on {len(val_dataset)} validation pairs...")
    print("-" * 60)

    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(val_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Smart ensemble
            pred = smart_ensemble_denoise(model, noisy, bm3d_sigma=bm3d_sigma)

            # Compute metrics
            batch_psnrs, batch_ssims = [], []
            for i in range(pred.shape[0]):
                batch_psnrs.append(compute_psnr(pred[i:i+1], clean[i:i+1]))
                batch_ssims.append(compute_ssim(pred[i:i+1], clean[i:i+1]))

            psnr_list.extend(batch_psnrs)
            ssim_list.extend(batch_ssims)

            samples_processed = min((batch_idx + 1) * 16, len(val_dataset))
            print(
                f"[Batch {batch_idx+1:>4}/{len(val_loader)} | {samples_processed}/{len(val_dataset)}] "
                f"Batch PSNR {np.mean(batch_psnrs):.2f} dB, SSIM {np.mean(batch_ssims):.4f} | "
                f"Running PSNR {np.mean(psnr_list):.2f} dB, SSIM {np.mean(ssim_list):.4f}",
                flush=True,
            )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Method: Smart Ensemble (SNR-Adaptive)")
    print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print("=" * 60)

    results = {
        "checkpoint": checkpoint_path,
        "val_pairs": val_pairs,
        "method": "Smart Ensemble (SNR-Adaptive CASA/BM3D)",
        "architecture": {
            "adapter": adapter,
            "backbone": backbone,
            "base_channels": base_channels,
            "residual_mode": residual_mode,
        },
        "bm3d_sigma": bm3d_sigma,
        "num_samples": len(val_dataset),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_pairs", type=str, required=True)
    parser.add_argument("--adapter", type=str, default="casa")
    parser.add_argument("--backbone", type=str, default="noise2void")
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--residual_mode", action="store_true")
    parser.add_argument("--bm3d_sigma", type=float, default=0.02)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    evaluate_smart_ensemble(
        args.checkpoint,
        args.val_pairs,
        args.adapter,
        args.backbone,
        args.base_channels,
        args.residual_mode,
        args.bm3d_sigma,
        args.output_json,
    )
