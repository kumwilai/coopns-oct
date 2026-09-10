#!/usr/bin/env python3
"""
Hybrid approach: Combine N2V+CASA with BM3D for improved performance.
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
    print("WARNING: bm3d not available. Install with: pip install bm3d")

from adaptive_oct_denoise import (
    PairedOCTDataset,
    build_model,
    compute_psnr,
    compute_ssim,
    device,
    resize_to,
)


def hybrid_denoise(model, noisy_batch, hybrid_mode="casa_then_bm3d", bm3d_sigma=0.02):
    """
    Hybrid denoising combining N2V+CASA with BM3D.

    Modes:
    - "casa_then_bm3d": CASA first, then BM3D refine
    - "bm3d_then_casa": BM3D first, then CASA refine
    - "average": Average of both outputs
    - "weighted": Weighted combination (0.7*CASA + 0.3*BM3D)
    """
    if not BM3D_AVAILABLE:
        return model(noisy_batch)

    # N2V+CASA denoising
    with torch.no_grad():
        casa_output = model(noisy_batch)

    # Process each image in batch with BM3D
    batch_size = noisy_batch.shape[0]
    results = []

    for i in range(batch_size):
        # Convert to numpy for BM3D
        noisy_np = noisy_batch[i, 0].cpu().numpy()
        casa_np = casa_output[i, 0].cpu().numpy()

        if hybrid_mode == "casa_then_bm3d":
            # Use CASA output as input to BM3D (refine CASA result)
            bm3d_output = bm3d.bm3d(casa_np, sigma_psd=bm3d_sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            final = bm3d_output

        elif hybrid_mode == "bm3d_then_casa":
            # BM3D first, then CASA refines
            bm3d_output = bm3d.bm3d(noisy_np, sigma_psd=bm3d_sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            bm3d_torch = torch.from_numpy(bm3d_output).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                casa_refined = model(bm3d_torch)
            final = casa_refined[0, 0].cpu().numpy()

        elif hybrid_mode == "average":
            # Simple average
            bm3d_output = bm3d.bm3d(noisy_np, sigma_psd=bm3d_sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            final = 0.5 * casa_np + 0.5 * bm3d_output

        elif hybrid_mode == "weighted":
            # Weighted: trust CASA more for structure, BM3D for smoothness
            bm3d_output = bm3d.bm3d(noisy_np, sigma_psd=bm3d_sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            final = 0.7 * casa_np + 0.3 * bm3d_output

        else:
            raise ValueError(f"Unknown hybrid mode: {hybrid_mode}")

        final = np.clip(final, 0.0, 1.0)
        results.append(torch.from_numpy(final))

    # Stack back to batch
    output = torch.stack(results).unsqueeze(1).to(device)
    return output


def evaluate_hybrid(
    checkpoint_path: str,
    val_pairs: str,
    adapter: str = "casa",
    backbone: str = "noise2void",
    base_channels: int = 48,
    residual_mode: bool = True,
    hybrid_mode: str = "casa_then_bm3d",
    bm3d_sigma: float = 0.02,
    output_json: str = None,
):
    if not BM3D_AVAILABLE:
        print("ERROR: BM3D not available. Install with: pip install bm3d")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Architecture: {adapter.upper()} + {backbone.upper()} | residual={residual_mode}")
    print(f"Hybrid mode: {hybrid_mode} | BM3D sigma: {bm3d_sigma}")

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

            # Hybrid denoising
            pred = hybrid_denoise(model, noisy, hybrid_mode=hybrid_mode, bm3d_sigma=bm3d_sigma)

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
    print(f"Hybrid mode: {hybrid_mode}")
    print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print("=" * 60)

    results = {
        "checkpoint": checkpoint_path,
        "val_pairs": val_pairs,
        "method": f"Hybrid: {hybrid_mode}",
        "architecture": {
            "adapter": adapter,
            "backbone": backbone,
            "base_channels": base_channels,
            "residual_mode": residual_mode,
        },
        "hybrid": {
            "mode": hybrid_mode,
            "bm3d_sigma": bm3d_sigma,
        },
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
    parser.add_argument("--hybrid_mode", type=str, default="casa_then_bm3d",
                       choices=["casa_then_bm3d", "bm3d_then_casa", "average", "weighted"])
    parser.add_argument("--bm3d_sigma", type=float, default=0.02)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    evaluate_hybrid(
        args.checkpoint,
        args.val_pairs,
        args.adapter,
        args.backbone,
        args.base_channels,
        args.residual_mode,
        args.hybrid_mode,
        args.bm3d_sigma,
        args.output_json,
    )
