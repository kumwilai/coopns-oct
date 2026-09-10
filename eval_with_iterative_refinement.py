#!/usr/bin/env python3
"""
Evaluate N2V+CASA with iterative refinement (BM3D-inspired two-stage approach).
"""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from adaptive_oct_denoise import (
    PairedOCTDataset,
    build_model,
    compute_psnr,
    compute_ssim,
    denoise_with_tta,
    device,
    resize_to,
)


def iterative_refinement_denoise(model, noisy, num_iterations=2, alpha=0.5):
    """
    Iterative refinement: denoise → extract residual → denoise residual → combine

    Args:
        model: Denoising model
        noisy: Input noisy image
        num_iterations: Number of refinement iterations
        alpha: Weight for residual correction (0.5 = balanced)
    """
    current = noisy

    for i in range(num_iterations):
        # Denoise current estimate
        denoised = model(current)

        if i < num_iterations - 1:
            # Extract residual noise
            residual = current - denoised

            # Denoise the residual
            denoised_residual = model(residual.clamp(0, 1))

            # Update estimate: remove denoised residual
            current = denoised - alpha * denoised_residual
            current = current.clamp(0, 1)
        else:
            current = denoised

    return current


def two_stage_denoise(model, noisy, use_tta=False):
    """
    Two-stage denoising inspired by BM3D:
    - Stage 1: Initial denoising (like hard-thresholding)
    - Stage 2: Refine using residual (like Wiener filtering)
    """
    # Stage 1: Initial denoising
    if use_tta:
        stage1 = denoise_with_tta(model, noisy, use_tta=True)
    else:
        stage1 = model(noisy)

    # Stage 2: Denoise the residual
    residual = (noisy - stage1).clamp(0, 1)
    residual_denoised = model(residual)

    # Combine: original denoised - some fraction of denoised residual
    # This removes remaining noise while preserving details
    final = stage1 - 0.3 * residual_denoised

    return final.clamp(0, 1)


def evaluate_with_refinement(
    checkpoint_path: str,
    val_pairs: str,
    adapter: str = "casa",
    backbone: str = "noise2void",
    base_channels: int = 48,
    residual_mode: bool = True,
    use_tta: bool = False,
    refinement_mode: str = "two_stage",  # "none", "two_stage", "iterative"
    num_iterations: int = 2,
    output_json: str = None,
):
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Architecture: {adapter.upper()} + {backbone.upper()} | residual={residual_mode}")
    print(f"Refinement mode: {refinement_mode} | TTA: {use_tta}")

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

            # Apply refinement strategy
            if refinement_mode == "two_stage":
                pred = two_stage_denoise(model, noisy, use_tta=use_tta)
            elif refinement_mode == "iterative":
                pred = iterative_refinement_denoise(model, noisy, num_iterations=num_iterations)
            else:  # "none"
                if use_tta:
                    pred = denoise_with_tta(model, noisy, use_tta=True)
                else:
                    pred = model(noisy)

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
    print(f"Refinement: {refinement_mode}")
    print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print("=" * 60)

    results = {
        "checkpoint": checkpoint_path,
        "val_pairs": val_pairs,
        "architecture": {
            "adapter": adapter,
            "backbone": backbone,
            "base_channels": base_channels,
            "residual_mode": residual_mode,
        },
        "refinement": {
            "mode": refinement_mode,
            "use_tta": use_tta,
            "num_iterations": num_iterations if refinement_mode == "iterative" else None,
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
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--refinement_mode", type=str, default="two_stage",
                       choices=["none", "two_stage", "iterative"])
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    evaluate_with_refinement(
        args.checkpoint,
        args.val_pairs,
        args.adapter,
        args.backbone,
        args.base_channels,
        args.residual_mode,
        args.use_tta,
        args.refinement_mode,
        args.num_iterations,
        args.output_json,
    )
