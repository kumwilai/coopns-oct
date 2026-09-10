#!/usr/bin/env python3
"""
Adaptive denoising strategies inspired by BM3D's key principles:
1. Multi-scale processing
2. Adaptive sigma estimation per patch
3. Non-local patch-based refinement
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptive_oct_denoise import (
    PairedOCTDataset,
    build_model,
    compute_psnr,
    compute_ssim,
    device,
    resize_to,
)


def estimate_noise_per_patch(image, patch_size=8):
    """
    Estimate noise level for each patch (like BM3D's adaptive processing).
    Uses local variance as a proxy for noise level.
    """
    B, C, H, W = image.shape

    # Compute local variance using sliding window
    kernel = torch.ones(1, 1, patch_size, patch_size, device=image.device) / (patch_size ** 2)

    # Use same padding to preserve dimensions exactly
    padding = patch_size // 2
    mean = F.conv2d(image, kernel, padding=padding)
    mean_sq = F.conv2d(image ** 2, kernel, padding=padding)

    # Ensure dimensions match by cropping/padding if needed
    if mean.shape != image.shape:
        mean = F.interpolate(mean, size=(H, W), mode='bilinear', align_corners=False)
        mean_sq = F.interpolate(mean_sq, size=(H, W), mode='bilinear', align_corners=False)

    variance = mean_sq - mean ** 2

    # Noise estimate (higher variance = more noise)
    noise_map = torch.sqrt(variance.clamp(min=1e-6))

    return noise_map


def multi_scale_denoise(model, noisy, scales=[1.0, 0.75, 0.5]):
    """
    Multi-scale denoising (process at different resolutions and combine).
    Like BM3D's multi-scale approach.
    """
    H, W = noisy.shape[2:]
    outputs = []

    for scale in scales:
        if scale != 1.0:
            # Resize to scale
            scaled_h, scaled_w = int(H * scale), int(W * scale)
            scaled_input = F.interpolate(noisy, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
        else:
            scaled_input = noisy

        # Denoise at this scale
        with torch.no_grad():
            scaled_output = model(scaled_input)

        # Resize back to original
        if scale != 1.0:
            output = F.interpolate(scaled_output, size=(H, W), mode='bilinear', align_corners=False)
        else:
            output = scaled_output

        outputs.append(output)

    # Weighted combination (prefer higher scales for structure)
    weights = [0.5, 0.3, 0.2]  # Higher weight for original scale
    combined = sum(w * out for w, out in zip(weights, outputs))

    return combined.clamp(0, 1)


def patch_based_refinement(model, noisy, denoised, patch_size=16, overlap=8):
    """
    Patch-based refinement inspired by BM3D's non-local approach.
    Process patches independently and blend them.
    """
    B, C, H, W = noisy.shape

    # Extract overlapping patches
    stride = patch_size - overlap
    patches = []
    positions = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = noisy[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            positions.append((i, j))

    # Process each patch
    refined_patches = []
    for patch in patches:
        with torch.no_grad():
            refined = model(patch)
        refined_patches.append(refined)

    # Reconstruct with blending
    output = torch.zeros_like(noisy)
    counts = torch.zeros_like(noisy)

    for (i, j), patch in zip(positions, refined_patches):
        output[:, :, i:i+patch_size, j:j+patch_size] += patch
        counts[:, :, i:i+patch_size, j:j+patch_size] += 1

    # Average overlapping regions
    output = output / counts.clamp(min=1)

    # Blend with original denoised (50/50)
    return (0.5 * denoised + 0.5 * output).clamp(0, 1)


def adaptive_strength_denoise(model, noisy):
    """
    Adaptive denoising strength based on local noise estimate.
    High-noise regions: full denoising
    Low-noise regions: gentle denoising (preserve details)
    """
    # Get model output
    with torch.no_grad():
        denoised = model(noisy)

    # Estimate local noise level
    noise_map = estimate_noise_per_patch(noisy, patch_size=8)

    # Ensure noise_map has same shape as denoised/noisy
    if noise_map.shape != noisy.shape:
        H, W = noisy.shape[2:]
        noise_map = F.interpolate(noise_map, size=(H, W), mode='bilinear', align_corners=False)

    # Normalize noise map to [0, 1]
    noise_map_norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min() + 1e-6)

    # Adaptive blending: high noise → use more denoised, low noise → keep more original
    # But inverse: low noise areas should trust the denoised more
    alpha = 1.0 - 0.5 * noise_map_norm  # Range: 0.5 to 1.0

    output = alpha * denoised + (1 - alpha) * noisy

    return output.clamp(0, 1)


def bm3d_inspired_denoise(model, noisy, mode="multi_scale"):
    """
    Apply BM3D-inspired techniques.

    Modes:
    - "multi_scale": Denoise at multiple scales and combine
    - "multi_scale_conservative": Multi-scale with trust towards higher scale
    - "adaptive_strength": Adaptive denoising based on local noise
    - "patch_refine": Patch-based refinement
    - "combined": Multi-scale only (adaptive strength removed for robustness)
    """
    if mode == "multi_scale":
        return multi_scale_denoise(model, noisy, scales=[1.0, 0.75, 0.5])

    elif mode == "multi_scale_conservative":
        # More conservative: trust the full-resolution output more
        return multi_scale_denoise(model, noisy, scales=[1.0, 0.75])

    elif mode == "adaptive_strength":
        return adaptive_strength_denoise(model, noisy)

    elif mode == "patch_refine":
        # First get initial denoising
        with torch.no_grad():
            denoised = model(noisy)
        # Then refine with patches
        return patch_based_refinement(model, noisy, denoised)

    elif mode == "combined":
        # Use multi-scale only - more robust across noise types
        # Adaptive strength hurts non-Gaussian noise
        return multi_scale_denoise(model, noisy, scales=[1.0, 0.75, 0.5])

    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_adaptive(
    checkpoint_path: str,
    val_pairs: str,
    adapter: str = "casa",
    backbone: str = "noise2void",
    base_channels: int = 48,
    residual_mode: bool = True,
    adaptive_mode: str = "multi_scale",
    output_json: str = None,
):
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Architecture: {adapter.upper()} + {backbone.upper()} | residual={residual_mode}")
    print(f"Adaptive mode: {adaptive_mode}")

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

            # Apply BM3D-inspired technique
            pred = bm3d_inspired_denoise(model, noisy, mode=adaptive_mode)

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
    print(f"Adaptive mode: {adaptive_mode}")
    print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print("=" * 60)

    results = {
        "checkpoint": checkpoint_path,
        "val_pairs": val_pairs,
        "method": f"Adaptive: {adaptive_mode}",
        "architecture": {
            "adapter": adapter,
            "backbone": backbone,
            "base_channels": base_channels,
            "residual_mode": residual_mode,
        },
        "adaptive_mode": adaptive_mode,
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
    parser.add_argument("--adaptive_mode", type=str, default="multi_scale",
                       choices=["multi_scale", "multi_scale_conservative", "adaptive_strength", "patch_refine", "combined"])
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    evaluate_adaptive(
        args.checkpoint,
        args.val_pairs,
        args.adapter,
        args.backbone,
        args.base_channels,
        args.residual_mode,
        args.adaptive_mode,
        args.output_json,
    )
