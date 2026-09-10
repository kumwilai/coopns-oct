"""
Evaluate a trained checkpoint on a paired validation set, with optional 8x TTA.
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


def evaluate(
    checkpoint_path: str,
    val_pairs: str,
    adapter: str = "casa",
    backbone: str = "unet",
    image_size: int = 64,
    batch_size: int = 16,
    base_channels: int | None = None,
    residual_mode: bool = False,
    use_tta: bool = False,
    output_json: str = None,
):
    print(f"Loading checkpoint: {checkpoint_path}")
    print(
        f"Architecture: {adapter.upper()} + {backbone.upper()} | residual={residual_mode} | TTA={'ON' if use_tta else 'OFF'}"
    )

    # Pick sensible default base channels if not provided
    if base_channels is None:
        if backbone in ["noise2void", "neighbor2neighbor"]:
            base_channels = 48
        elif backbone == "nafnet":
            base_channels = 32
        else:
            base_channels = 64

    # Build model
    model = build_model(
        base_channels=base_channels,
        residual_mode=residual_mode,
        adapter_type=adapter,
        backbone_type=backbone,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).eval()

    # Load validation data
    transform = resize_to((image_size, image_size))
    val_dataset = PairedOCTDataset(val_pairs, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Evaluating on {len(val_dataset)} validation pairs...")
    print(f"Batch size: {batch_size}, Total batches: {len(val_loader)}")
    print("-" * 60)

    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(val_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            if use_tta:
                pred = denoise_with_tta(model, noisy, use_tta=True)
            else:
                pred = model(noisy)

            # Compute metrics per image to avoid batch-size bias
            batch_psnrs, batch_ssims = [], []
            for i in range(pred.shape[0]):
                batch_psnrs.append(compute_psnr(pred[i : i + 1], clean[i : i + 1]))
                batch_ssims.append(compute_ssim(pred[i : i + 1], clean[i : i + 1]))

            psnr_list.extend(batch_psnrs)
            ssim_list.extend(batch_ssims)

            samples_processed = min((batch_idx + 1) * batch_size, len(val_dataset))
            print(
                f"[Batch {batch_idx+1:>4}/{len(val_loader)} | {samples_processed}/{len(val_dataset)}] "
                f"Batch PSNR {np.mean(batch_psnrs):.2f} dB, SSIM {np.mean(batch_ssims):.4f} | "
                f"Running PSNR {np.mean(psnr_list):.2f} dB, SSIM {np.mean(ssim_list):.4f}",
                flush=True,
            )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
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
            "use_tta": use_tta,
            "image_size": image_size,
        },
        "num_samples": len(val_dataset),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "psnr_min": float(np.min(psnr_list)),
        "psnr_max": float(np.max(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
        "ssim_min": float(np.min(ssim_list)),
        "ssim_max": float(np.max(ssim_list)),
    }

    # Save results to JSON if output path specified
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
    parser.add_argument("--adapter", type=str, default="casa", choices=["global", "spatial", "casa"])
    parser.add_argument(
        "--backbone", type=str, default="unet", choices=["unet", "nafnet", "noise2void", "neighbor2neighbor"]
    )
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
        help="Override base channels (default: 64 unet, 48 n2v/n2n, 32 nafnet)",
    )
    parser.add_argument("--residual_mode", action="store_true", help="Use residual inference (x - tanh(pred))")
    parser.add_argument("--use_tta", action="store_true", help="Enable 8x rotation/flip TTA during evaluation")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results as JSON")
    args = parser.parse_args()

    evaluate(
        args.checkpoint,
        args.val_pairs,
        args.adapter,
        args.backbone,
        args.image_size,
        args.batch_size,
        args.base_channels,
        args.residual_mode,
        args.use_tta,
        args.output_json,
    )
