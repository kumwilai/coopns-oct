#!/usr/bin/env python3
"""
Compare Multiple Denoising Methods on OCT Data
Evaluates BM3D, NLM, and trained models (CASA+N2V, NAFNet, etc.)
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Import from eval_baselines
from eval_baselines import (
    load_image_grayscale, save_image_grayscale,
    compute_metrics, denoise_bm3d, denoise_nlm,
    auto_estimate_sigma, BM3D_AVAILABLE
)


# ========================================
# Model-based Denoising
# ========================================

def denoise_with_model(
    noisy: np.ndarray,
    model_path: str,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Denoise image using a trained PyTorch model (CASA+N2V, NAFNet, etc.)

    Args:
        noisy: Input image [H, W] in [0, 1]
        model_path: Path to checkpoint (.pth file)
        device: 'cuda' or 'cpu'

    Returns:
        Denoised image [H, W] in [0, 1]
    """
    # Import adaptive_oct_denoise
    sys.path.insert(0, os.path.dirname(__file__))
    from adaptive_oct_denoise import build_model, AdaptiveDenoiser

    # Determine model configuration from filename
    if 'noise2void' in model_path.lower() or 'n2v' in model_path.lower():
        backbone = 'noise2void'
        base_channels = 48
    elif 'nafnet' in model_path.lower():
        backbone = 'nafnet'
        base_channels = 32
    else:
        backbone = 'unet'
        base_channels = 64

    # Determine adapter type
    if 'casa' in model_path.lower():
        adapter = 'casa'
    elif 'spatial' in model_path.lower():
        adapter = 'spatial'
    else:
        adapter = 'global'

    print(f"  Loading model: backbone={backbone}, adapter={adapter}, channels={base_channels}")

    # Build model
    model = build_model(
        base_channels=base_channels,
        residual_mode=False,
        adapter_type=adapter,
        backbone_type=backbone
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Prepare input
    noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    # Denoise
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
        if isinstance(denoised_tensor, tuple):
            denoised_tensor = denoised_tensor[0]  # Remove aux outputs

    # Convert back to numpy
    denoised = denoised_tensor.squeeze().cpu().numpy()
    denoised = np.clip(denoised, 0.0, 1.0)

    return denoised


# ========================================
# Multi-Method Comparison
# ========================================

def compare_all_methods(
    noisy_path: str,
    clean_path: str,
    output_dir: str,
    model_paths: Dict[str, str],
    sigma: float = None,
    device: str = 'cuda',
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, Dict]:
    """
    Compare all methods on a single image.

    Args:
        noisy_path: Path to noisy image
        clean_path: Path to clean image
        output_dir: Output directory
        model_paths: Dict of {method_name: model_checkpoint_path}
        sigma: Noise sigma for classical methods
        device: Device for models

    Returns:
        Dictionary of results for each method
    """
    # Load images
    noisy = load_image_grayscale(noisy_path, resize_hw=resize_hw)
    clean = load_image_grayscale(clean_path, resize_hw=resize_hw)

    if sigma is None:
        sigma = auto_estimate_sigma(noisy)
        print(f"Auto-estimated sigma: {sigma:.4f}")

    results = {}

    # Input (noisy) metrics
    noisy_metrics = compute_metrics(noisy, clean)
    results['Noisy Input'] = {
        'image': noisy,
        'metrics': noisy_metrics
    }

    # BM3D
    if BM3D_AVAILABLE:
        print("Running BM3D...")
        bm3d_denoised = denoise_bm3d(noisy, sigma_psd=sigma)
        bm3d_metrics = compute_metrics(bm3d_denoised, clean)
        results['BM3D'] = {
            'image': bm3d_denoised,
            'metrics': bm3d_metrics
        }
        save_image_grayscale(bm3d_denoised, os.path.join(output_dir, 'bm3d_denoised.png'))

    # NLM
    print("Running NLM...")
    nlm_denoised = denoise_nlm(noisy, h=sigma)
    nlm_metrics = compute_metrics(nlm_denoised, clean)
    results['NLM'] = {
        'image': nlm_denoised,
        'metrics': nlm_metrics
    }
    save_image_grayscale(nlm_denoised, os.path.join(output_dir, 'nlm_denoised.png'))

    # Trained models
    for method_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found: {model_path}")
            continue

        print(f"Running {method_name}...")
        try:
            model_denoised = denoise_with_model(noisy, model_path, device=device)
            model_metrics = compute_metrics(model_denoised, clean)
            results[method_name] = {
                'image': model_denoised,
                'metrics': model_metrics
            }
            safe_name = method_name.lower().replace(' ', '_').replace('+', '_')
            save_image_grayscale(model_denoised, os.path.join(output_dir, f'{safe_name}_denoised.png'))
        except Exception as e:
            print(f"ERROR running {method_name}: {e}")

    # Save clean reference
    save_image_grayscale(clean, os.path.join(output_dir, 'clean_reference.png'))
    save_image_grayscale(noisy, os.path.join(output_dir, 'noisy_input.png'))

    return results


def create_comparison_figure(
    results: Dict[str, Dict],
    output_path: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    crop_box: tuple = None
):
    """
    Create visual comparison figure with zoom insets.

    Args:
        results: Results from compare_all_methods
        output_path: Path to save figure
        clean: Clean reference image
        noisy: Noisy input
        crop_box: (y0, x0, h, w) for zoom inset
    """
    methods = list(results.keys())
    n_methods = len(methods)

    # Create figure
    fig, axes = plt.subplots(2, n_methods, figsize=(4*n_methods, 8))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    for idx, method in enumerate(methods):
        img = results[method]['image']
        metrics = results[method]['metrics']

        # Full image
        axes[0, idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f"{method}\nPSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.3f}")
        axes[0, idx].axis('off')

        # Zoom inset
        if crop_box:
            y0, x0, h, w = crop_box
            cropped = img[y0:y0+h, x0:x0+w]
            axes[1, idx].imshow(cropped, cmap='gray', vmin=0, vmax=1)
            axes[1, idx].set_title(f"{method} (Zoom)")
        else:
            axes[1, idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[1, idx].set_title(f"{method}")
        axes[1, idx].axis('off')

        # Draw rectangle on full image
        if crop_box:
            from matplotlib.patches import Rectangle
            rect = Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')
            axes[0, idx].add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {output_path}")


def batch_compare(
    pair_list: str,
    output_dir: str,
    model_paths: Dict[str, str],
    sigma: float = None,
    device: str = 'cuda',
    save_individual: bool = False,
    resize_hw: Optional[Tuple[int, int]] = None,
):
    """
    Batch comparison across multiple image pairs.

    Args:
        pair_list: Path to pair list file
        output_dir: Output directory
        model_paths: Dict of model paths
        sigma: Noise sigma
        device: Device for models
        save_individual: Save individual comparison figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read pairs
    pairs = []
    with open(pair_list, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))

    print(f"\n{'='*80}")
    print(f"BATCH COMPARISON: {len(pairs)} images")
    print(f"Methods: Noisy, BM3D, NLM, {', '.join(model_paths.keys())}")
    print(f"{'='*80}\n")

    # Aggregate results
    all_results = {method: [] for method in ['Noisy Input', 'BM3D', 'NLM'] + list(model_paths.keys())}

    for i, (noisy_path, clean_path) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing: {os.path.basename(noisy_path)}")

        img_output_dir = os.path.join(output_dir, f"image_{i:03d}")
        os.makedirs(img_output_dir, exist_ok=True)

        try:
            results = compare_all_methods(
                noisy_path, clean_path, img_output_dir,
                model_paths=model_paths, sigma=sigma, device=device, resize_hw=resize_hw
            )

            # Aggregate metrics
            for method, data in results.items():
                all_results[method].append(data['metrics'])

            # Save individual figure
            if save_individual:
                clean = load_image_grayscale(clean_path, resize_hw=resize_hw)
                noisy = load_image_grayscale(noisy_path, resize_hw=resize_hw)
                create_comparison_figure(
                    results,
                    os.path.join(img_output_dir, 'comparison.png'),
                    clean, noisy
                )

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Compute summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    summary = {}
    for method, metrics_list in all_results.items():
        if not metrics_list:
            continue

        psnr_values = [m['psnr'] for m in metrics_list]
        ssim_values = [m['ssim'] for m in metrics_list]

        summary[method] = {
            'count': len(metrics_list),
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
        }

        print(f"{method:20s} | PSNR: {summary[method]['psnr_mean']:.2f} ± {summary[method]['psnr_std']:.2f} dB | SSIM: {summary[method]['ssim_mean']:.4f} ± {summary[method]['ssim_std']:.4f}")

    # Save summary
    summary_path = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print(f"{'='*80}\n")

    return summary


# ========================================
# Main
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare Multiple Denoising Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare BM3D, NLM, and CASA+N2V on test set
  python compare_methods.py \\
    --pair_list val_pairs.txt \\
    --output_dir results/comparison \\
    --casa_n2v checkpoints/casa_noise2void/finetuned.pth

  # Compare multiple models
  python compare_methods.py \\
    --pair_list val_pairs.txt \\
    --output_dir results/comparison \\
    --casa_n2v checkpoints/casa_noise2void/finetuned.pth \\
    --nafnet checkpoints/casa_nafnet/finetuned.pth \\
    --unet checkpoints/casa_unet/finetuned.pth

  # Single image comparison
  python compare_methods.py \\
    --noisy_image test_noisy.png \\
    --clean_image test_clean.png \\
    --output_dir results/single \\
    --casa_n2v checkpoints/casa_noise2void/finetuned.pth
        """
    )

    parser.add_argument('--pair_list', type=str, help='Path to pair list file')
    parser.add_argument('--noisy_image', type=str, help='Single noisy image')
    parser.add_argument('--clean_image', type=str, help='Single clean image')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    # Model paths
    parser.add_argument('--casa_n2v', type=str, help='CASA+N2V checkpoint')
    parser.add_argument('--nafnet', type=str, help='NAFNet checkpoint')
    parser.add_argument('--unet', type=str, help='U-Net checkpoint')
    parser.add_argument('--custom_model', type=str, action='append', nargs=2, metavar=('NAME', 'PATH'),
                        help='Custom model: --custom_model "MyModel" path/to/checkpoint.pth')

    parser.add_argument('--sigma', type=float, default=None, help='Noise sigma (auto-estimated if not provided)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for models')
    parser.add_argument('--save_figures', action='store_true', help='Save individual comparison figures')
    parser.add_argument('--resize_h', type=int, default=None,
                        help='Optional resize height before denoising/metrics (set with --resize_w)')
    parser.add_argument('--resize_w', type=int, default=None,
                        help='Optional resize width before denoising/metrics (set with --resize_h)')

    args = parser.parse_args()

    resize_hw = None
    if args.resize_h is not None or args.resize_w is not None:
        if args.resize_h is None or args.resize_w is None:
            parser.error("--resize_h and --resize_w must be set together")
        resize_hw = (args.resize_h, args.resize_w)
        print(f"Resizing evaluation images to {resize_hw[0]}x{resize_hw[1]}")

    # Build model paths dict
    model_paths = {}
    if args.casa_n2v:
        model_paths['CASA+N2V'] = args.casa_n2v
    if args.nafnet:
        model_paths['NAFNet'] = args.nafnet
    if args.unet:
        model_paths['U-Net'] = args.unet
    if args.custom_model:
        for name, path in args.custom_model:
            model_paths[name] = path

    os.makedirs(args.output_dir, exist_ok=True)

    # Batch processing
    if args.pair_list:
        batch_compare(
            args.pair_list,
            args.output_dir,
            model_paths=model_paths,
            sigma=args.sigma,
            device=args.device,
            save_individual=args.save_figures,
            resize_hw=resize_hw
        )

    # Single image
    elif args.noisy_image and args.clean_image:
        results = compare_all_methods(
            args.noisy_image,
            args.clean_image,
            args.output_dir,
            model_paths=model_paths,
            sigma=args.sigma,
            device=args.device,
            resize_hw=resize_hw
        )

        # Print results
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        for method, data in results.items():
            m = data['metrics']
            print(f"{method:20s} | PSNR: {m['psnr']:.2f} dB | SSIM: {m['ssim']:.4f}")
        print(f"{'='*60}\n")

        # Create figure
        clean = load_image_grayscale(args.clean_image, resize_hw=resize_hw)
        noisy = load_image_grayscale(args.noisy_image, resize_hw=resize_hw)
        create_comparison_figure(
            results,
            os.path.join(args.output_dir, 'comparison.png'),
            clean, noisy
        )

    else:
        print("ERROR: Must provide either --pair_list OR (--noisy_image AND --clean_image)")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
