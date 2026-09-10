#!/usr/bin/env python3
"""
Evaluate OCT denoising model with Test-Time Augmentation (TTA).
Uses geometric transforms (flip/rotate) for augmentation-based TTA.
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# Import from the main training script
from adaptive_oct_denoise import (
    AdaptiveDenoiser,
    B2UNetBackbone,
    DenoisingBackbone,
    NAFBackbone,
    S2SNetBackbone,
    NoiseAdapter,
    SpatialNoiseAdapter,
    CoherenceSpatialAdapter,
    PairedOCTDataset,
    resize_to,
)


def model_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Safely call model, handling tuple outputs.

    AdaptiveDenoiser.forward() may return:
    - Just output tensor (when return_aux=False, default)
    - (output, aux_dict) tuple (when return_aux=True and adapter is CASA)

    Args:
        model: Denoising model
        x: Input tensor

    Returns:
        Output tensor only (discards aux info if present)
    """
    result = model(x)
    if isinstance(result, tuple):
        return result[0]
    return result


def tta_augmentation(model: nn.Module, x: torch.Tensor, num_augments: int = 8) -> torch.Tensor:
    """
    Test-Time Augmentation using geometric transforms.
    Uses running mean to avoid storing all predictions simultaneously (OOM-safe).

    Args:
        model: Trained denoising model
        x: Input tensor [B, C, H, W]
        num_augments: Number of augmentations (4 or 8)
            - 4: flips only (horizontal, vertical, both)
            - 8: flips + rotations (full 8-fold ensemble)

    Returns:
        Averaged prediction [B, C, H, W]
    """
    model.eval()

    with torch.no_grad():
        # Running mean: accumulate sum, divide at end
        result = model_forward(model, x)  # 1. Original
        count = 1

        # 2. Horizontal flip
        pred = model_forward(model, torch.flip(x, dims=[-1]))
        result.add_(torch.flip(pred, dims=[-1]))
        count += 1
        del pred

        # 3. Vertical flip
        pred = model_forward(model, torch.flip(x, dims=[-2]))
        result.add_(torch.flip(pred, dims=[-2]))
        count += 1
        del pred

        # 4. Both flips (180° rotation equivalent)
        pred = model_forward(model, torch.flip(x, dims=[-1, -2]))
        result.add_(torch.flip(pred, dims=[-1, -2]))
        count += 1
        del pred

        if num_augments == 8:
            x_rot90 = torch.rot90(x, k=1, dims=[-2, -1])

            # 5. 90° rotation
            pred = model_forward(model, x_rot90)
            result.add_(torch.rot90(pred, k=-1, dims=[-2, -1]))
            count += 1
            del pred

            # 6. 90° + horizontal flip
            pred = model_forward(model, torch.flip(x_rot90, dims=[-1]))
            result.add_(torch.rot90(torch.flip(pred, dims=[-1]), k=-1, dims=[-2, -1]))
            count += 1
            del pred

            # 7. 90° + vertical flip
            pred = model_forward(model, torch.flip(x_rot90, dims=[-2]))
            result.add_(torch.rot90(torch.flip(pred, dims=[-2]), k=-1, dims=[-2, -1]))
            count += 1
            del pred

            # 8. 90° + both flips
            pred = model_forward(model, torch.flip(x_rot90, dims=[-1, -2]))
            result.add_(torch.rot90(torch.flip(pred, dims=[-1, -2]), k=-1, dims=[-2, -1]))
            count += 1
            del pred

            del x_rot90

    result.div_(count)
    return result


def build_model(backbone_name: str, adapter_type: str, base_channels: int = 64,
                dropout_rate: float = 0.0, residual_mode: str = 'residual') -> AdaptiveDenoiser:
    """Build the denoising model."""
    # Build backbone
    if backbone_name == 'b2unet':
        backbone = B2UNetBackbone(in_channels=1, out_channels=1, base_channels=base_channels)
    elif backbone_name == 'nafnet':
        backbone = NAFBackbone(in_channels=1, out_channels=1, base_channels=base_channels)
    elif backbone_name == 's2s':
        backbone = S2SNetBackbone(in_channels=1, out_channels=1, base_channels=base_channels, dropout_rate=dropout_rate)
    else:  # unet
        backbone = DenoisingBackbone(in_channels=1, out_channels=1, base_channels=base_channels)

    # Build adapter
    if adapter_type == 'casa':
        adapter = CoherenceSpatialAdapter(block_channels=backbone.modulated_channels)
    elif adapter_type == 'spatial':
        adapter = SpatialNoiseAdapter(block_channels=backbone.modulated_channels)
    else:  # global
        adapter = NoiseAdapter()

    return AdaptiveDenoiser(backbone, adapter, residual_mode=residual_mode)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    """
    Compute PSNR and SSIM metrics.

    Uses appropriate SSIM window size based on image dimensions:
    - For small images (< 64): win_size=3
    - For medium images (64-128): win_size=7 (default)
    - For large images (> 128): win_size=11
    """
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    # Determine appropriate window size for SSIM
    # Based on image dimensions (handle both [H,W] and [B,H,W])
    if pred_np.ndim == 3:
        height = pred_np.shape[1]
    else:
        height = pred_np.shape[0]

    if height < 64:
        win_size = 3
    elif height < 128:
        win_size = 7
    else:
        win_size = 11

    # Handle batch dimension
    if pred_np.ndim == 3:  # [B, H, W]
        psnrs = []
        ssims = []
        for i in range(pred_np.shape[0]):
            psnr = sk_psnr(target_np[i], pred_np[i], data_range=1.0)
            ssim = sk_ssim(target_np[i], pred_np[i], data_range=1.0, win_size=win_size)
            psnrs.append(psnr)
            ssims.append(ssim)
        return np.mean(psnrs), np.mean(ssims)
    else:  # [H, W]
        psnr = sk_psnr(target_np, pred_np, data_range=1.0)
        ssim = sk_ssim(target_np, pred_np, data_range=1.0, win_size=win_size)
        return psnr, ssim


def main():
    parser = argparse.ArgumentParser(description='Evaluate OCT Denoising with TTA Augmentation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val_pairs', type=str, required=True, help='Validation pairs list file')
    parser.add_argument('--backbone', type=str, default='b2unet',
                        choices=['unet', 'b2unet', 'nafnet', 's2s'],
                        help='Backbone architecture')
    parser.add_argument('--adapter', type=str, default='casa',
                        choices=['global', 'spatial', 'casa'],
                        help='Adapter type')
    parser.add_argument('--base_channels', type=int, default=48,
                        help='Base number of channels in backbone')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Input image size (will be resized to this)')
    parser.add_argument('--num_augments', type=int, default=8, choices=[4, 8],
                        help='Number of TTA augments (4=flips only, 8=flips+rotations)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--residual_mode', type=str, default='residual',
                        choices=['residual', 'direct'],
                        help='Output mode: residual (pred=noisy-noise) or direct (pred=clean)')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate (for S2S backbone)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Optional path to save detailed results as CSV')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Build model
    model = build_model(
        backbone_name=args.backbone,
        adapter_type=args.adapter,
        base_channels=args.base_channels,
        dropout_rate=args.dropout_rate,
        residual_mode=args.residual_mode
    ).to(device)

    # Load weights
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded from epoch {ckpt.get('epoch', 'unknown')}")
        print(f"Best val PSNR: {ckpt.get('best_val_psnr', 'unknown'):.2f} dB")
    elif 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw state dict")

    model.eval()

    # Dataset
    if not os.path.exists(args.val_pairs):
        print(f"ERROR: Validation pairs file not found: {args.val_pairs}")
        sys.exit(1)

    transform = resize_to((args.image_size, args.image_size))
    ds = PairedOCTDataset(args.val_pairs, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n{'='*80}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Validation pairs: {args.val_pairs}")
    print(f"  Total images: {len(ds)}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Base channels: {args.base_channels}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  TTA augments: {args.num_augments}")
    print(f"  Residual mode: {args.residual_mode}")
    print(f"{'='*80}\n")

    # Evaluation
    results_base = {'psnr': [], 'ssim': []}
    results_tta = {'psnr': [], 'ssim': []}
    detailed_results = []

    print("Evaluating...")
    print(f"{'Image':<8} {'Base PSNR':>10} {'Base SSIM':>10} {'TTA PSNR':>10} {'TTA SSIM':>10} {'PSNR Gain':>10} {'SSIM Gain':>10}")
    print("-" * 80)

    # Start timing
    eval_start_time = time.time()

    for i, (x_noisy, x_clean) in enumerate(dl, 1):
        x_noisy, x_clean = x_noisy.to(device), x_clean.to(device)

        with torch.no_grad():
            # Base prediction (no TTA) - handle potential tuple output
            pred_base = model_forward(model, x_noisy).clamp(0.0, 1.0)

            # TTA prediction - tta_augmentation already uses model_forward internally
            pred_tta = tta_augmentation(model, x_noisy, num_augments=args.num_augments).clamp(0.0, 1.0)

        # Compute metrics
        psnr_base, ssim_base = compute_metrics(pred_base, x_clean)
        psnr_tta, ssim_tta = compute_metrics(pred_tta, x_clean)

        results_base['psnr'].append(psnr_base)
        results_base['ssim'].append(ssim_base)
        results_tta['psnr'].append(psnr_tta)
        results_tta['ssim'].append(ssim_tta)

        gain_psnr = psnr_tta - psnr_base
        gain_ssim = ssim_tta - ssim_base

        # Print progress
        print(f"{i:4d}/{len(dl):<3} {psnr_base:10.2f} {ssim_base:10.4f} {psnr_tta:10.2f} {ssim_tta:10.4f} {gain_psnr:+10.2f} {gain_ssim:+10.4f}")

        # Store detailed results
        detailed_results.append({
            'image_idx': i,
            'psnr_base': psnr_base,
            'ssim_base': ssim_base,
            'psnr_tta': psnr_tta,
            'ssim_tta': ssim_tta,
            'psnr_gain': gain_psnr,
            'ssim_gain': gain_ssim
        })

        del pred_base, pred_tta, x_noisy, x_clean

    # End timing
    eval_total_time = time.time() - eval_start_time
    time_per_image = eval_total_time / len(ds)

    # Summary statistics
    mean_psnr_base = np.mean(results_base['psnr'])
    mean_psnr_tta = np.mean(results_tta['psnr'])
    mean_ssim_base = np.mean(results_base['ssim'])
    mean_ssim_tta = np.mean(results_tta['ssim'])

    std_psnr_base = np.std(results_base['psnr'])
    std_psnr_tta = np.std(results_tta['psnr'])
    std_ssim_base = np.std(results_base['ssim'])
    std_ssim_tta = np.std(results_tta['ssim'])

    print("\n" + "="*80)
    print(f"FINAL RESULTS ({len(ds)} images)")
    print("="*80)
    print(f"  Base Model:")
    print(f"    PSNR = {mean_psnr_base:.2f} ± {std_psnr_base:.2f} dB")
    print(f"    SSIM = {mean_ssim_base:.4f} ± {std_ssim_base:.4f}")
    print(f"\n  TTA ({args.num_augments}-fold):")
    print(f"    PSNR = {mean_psnr_tta:.2f} ± {std_psnr_tta:.2f} dB")
    print(f"    SSIM = {mean_ssim_tta:.4f} ± {std_ssim_tta:.4f}")
    print(f"\n  TTA Improvement:")
    print(f"    PSNR = {mean_psnr_tta - mean_psnr_base:+.2f} dB ({100*(mean_psnr_tta - mean_psnr_base)/mean_psnr_base:+.1f}%)")
    print(f"    SSIM = {mean_ssim_tta - mean_ssim_base:+.4f} ({100*(mean_ssim_tta - mean_ssim_base)/mean_ssim_base:+.1f}%)")
    print(f"\n  Performance:")
    print(f"    Total time: {eval_total_time:.1f}s")
    print(f"    Time per image: {time_per_image:.3f}s")
    print(f"    Throughput: {len(ds)/eval_total_time:.2f} images/sec")
    print("="*80)

    # Save detailed results if requested
    if args.save_results:
        import csv
        with open(args.save_results, 'w', newline='') as f:
            fieldnames = ['image_idx', 'psnr_base', 'ssim_base', 'psnr_tta', 'ssim_tta', 'psnr_gain', 'ssim_gain']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
        print(f"\nDetailed results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
