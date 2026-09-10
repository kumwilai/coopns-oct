#!/usr/bin/env python3
"""
Comprehensive Evaluation for Interpretable Anatomy-Aware OCT Denoising

Reports:
1. Overall PSNR/SSIM metrics
2. Per-anatomy-region metrics (NFL, GCL, INL, OPL, RPE/Choroid)
3. Interpretability metrics (physics features, layer detection)
4. Comparison with baseline
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_soft_conditioning import SoftConditionedDenoiser, OCTDenoiseDataset
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.utils.metrics import compute_psnr, compute_ssim


# Anatomy layer definitions (approximate depth percentages)
ANATOMY_LAYERS = {
    'NFL_GCL': (0.0, 0.15),      # Nerve Fiber Layer + Ganglion Cell Layer (top 15%)
    'IPL_INL': (0.15, 0.35),     # Inner Plexiform + Inner Nuclear Layer
    'OPL_ONL': (0.35, 0.55),     # Outer Plexiform + Outer Nuclear Layer
    'IS_OS': (0.55, 0.75),       # Inner/Outer Segments (photoreceptors)
    'RPE_Choroid': (0.75, 1.0),  # RPE + Choroid (bottom 25%)
}


def create_layer_masks(height, width):
    """Create masks for each anatomy layer based on depth."""
    masks = {}
    for name, (start, end) in ANATOMY_LAYERS.items():
        mask = torch.zeros(1, 1, height, width)
        start_row = int(start * height)
        end_row = int(end * height)
        mask[:, :, start_row:end_row, :] = 1.0
        masks[name] = mask
    return masks


def compute_metrics_per_region(pred, target, masks):
    """Compute PSNR and SSIM for each anatomy region."""
    metrics = {}

    for name, mask in masks.items():
        mask = mask.to(pred.device)

        # Masked PSNR
        pred_masked = pred * mask
        target_masked = target * mask

        # Only compute where mask is active
        mask_sum = mask.sum()
        if mask_sum > 0:
            mse = ((pred_masked - target_masked) ** 2).sum() / mask_sum
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            metrics[f'{name}_psnr'] = psnr.item()

            # Simplified SSIM for region
            pred_region = pred_masked.sum() / mask_sum
            target_region = target_masked.sum() / mask_sum
            metrics[f'{name}_mean_diff'] = abs(pred_region.item() - target_region.item())
        else:
            metrics[f'{name}_psnr'] = 0.0
            metrics[f'{name}_mean_diff'] = 0.0

    return metrics


def compute_edge_preservation(pred, target, clean):
    """Compute edge preservation metric."""
    # Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)

    def get_edges(img):
        gx = F.conv2d(F.pad(img, [1,1,1,1], mode='reflect'), sobel_x)
        gy = F.conv2d(F.pad(img, [1,1,1,1], mode='reflect'), sobel_y)
        return torch.sqrt(gx**2 + gy**2)

    pred_edges = get_edges(pred)
    clean_edges = get_edges(clean)

    # Correlation between predicted and clean edges
    pred_flat = pred_edges.flatten()
    clean_flat = clean_edges.flatten()

    correlation = torch.corrcoef(torch.stack([pred_flat, clean_flat]))[0, 1]
    return correlation.item() if not torch.isnan(correlation) else 0.0


def compute_contrast_to_noise(pred, layer_masks):
    """Compute Contrast-to-Noise Ratio between adjacent layers."""
    cnr_metrics = {}
    layer_names = list(layer_masks.keys())

    for i in range(len(layer_names) - 1):
        name1, name2 = layer_names[i], layer_names[i + 1]
        mask1 = layer_masks[name1].to(pred.device)
        mask2 = layer_masks[name2].to(pred.device)

        # Mean and std in each region
        region1 = pred * mask1
        region2 = pred * mask2

        mean1 = region1.sum() / mask1.sum()
        mean2 = region2.sum() / mask2.sum()

        std1 = torch.sqrt(((region1 - mean1 * mask1) ** 2).sum() / mask1.sum())
        std2 = torch.sqrt(((region2 - mean2 * mask2) ** 2).sum() / mask2.sum())

        # CNR = |mean1 - mean2| / sqrt((std1^2 + std2^2) / 2)
        cnr = abs(mean1 - mean2) / (torch.sqrt((std1**2 + std2**2) / 2) + 1e-8)
        cnr_metrics[f'CNR_{name1}_{name2}'] = cnr.item()

    return cnr_metrics


def evaluate_model(model, base_model, val_loader, device, output_dir):
    """Comprehensive model evaluation."""
    model.eval()
    base_model.eval()

    # Initialize accumulators
    overall_metrics = {
        'psnr_base': [], 'psnr_ours': [],
        'ssim_base': [], 'ssim_ours': [],
        'edge_preservation': [],
    }

    layer_metrics_base = {name: [] for name in ANATOMY_LAYERS.keys()}
    layer_metrics_ours = {name: [] for name in ANATOMY_LAYERS.keys()}
    cnr_metrics_base = []
    cnr_metrics_ours = []

    interpretability = {
        'coef_var_mean': [], 'coef_var_std': [],
        'sig_var_corr': [], 'horiz_ratio': [],
        'layer_entropy': [], 'refinement_mag': [],
    }

    # Sample for visualization
    vis_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            B, C, H, W = noisy.shape
            layer_masks = create_layer_masks(H, W)

            # Get outputs
            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            denoised, features = model(noisy, return_features=True)

            for i in range(B):
                n, c, d, b = noisy[i:i+1], clean[i:i+1], denoised[i:i+1], base_out[i:i+1]

                # Overall metrics
                overall_metrics['psnr_base'].append(compute_psnr(b, c))
                overall_metrics['psnr_ours'].append(compute_psnr(d, c))
                overall_metrics['ssim_base'].append(compute_ssim(b, c))
                overall_metrics['ssim_ours'].append(compute_ssim(d, c))
                overall_metrics['edge_preservation'].append(compute_edge_preservation(d, b, c))

                # Per-layer metrics
                region_base = compute_metrics_per_region(b, c, layer_masks)
                region_ours = compute_metrics_per_region(d, c, layer_masks)

                for name in ANATOMY_LAYERS.keys():
                    layer_metrics_base[name].append(region_base[f'{name}_psnr'])
                    layer_metrics_ours[name].append(region_ours[f'{name}_psnr'])

                # CNR metrics
                cnr_base = compute_contrast_to_noise(b, layer_masks)
                cnr_ours = compute_contrast_to_noise(d, layer_masks)
                cnr_metrics_base.append(cnr_base)
                cnr_metrics_ours.append(cnr_ours)

            # Interpretability metrics
            raw = features['raw_features']
            interpretability['coef_var_mean'].append(raw['coef_variation'].mean().item())
            interpretability['coef_var_std'].append(raw['coef_variation'].std().item())
            interpretability['sig_var_corr'].append(raw['signal_var_corr'].mean().item())
            interpretability['horiz_ratio'].append(raw['horizontal_ratio'].mean().item())

            layer_prob = features['layer_prob']
            layer_entropy = -(layer_prob * torch.log(layer_prob + 1e-8)).sum(dim=1).mean()
            interpretability['layer_entropy'].append(layer_entropy.item())
            interpretability['refinement_mag'].append(features['refinement'].abs().mean().item())

            # Save samples for visualization
            if batch_idx < 3:
                vis_samples.append({
                    'noisy': noisy[0].cpu(),
                    'clean': clean[0].cpu(),
                    'base': base_out[0].cpu(),
                    'ours': denoised[0].cpu(),
                    'layer_prob': layer_prob[0].cpu(),
                    'coef_var': raw['coef_variation'][0].cpu(),
                    'refinement': features['refinement'][0].cpu(),
                })

    # Compute final metrics
    results = {
        'overall': {
            'psnr_base': np.mean(overall_metrics['psnr_base']),
            'psnr_ours': np.mean(overall_metrics['psnr_ours']),
            'psnr_gain': np.mean(overall_metrics['psnr_ours']) - np.mean(overall_metrics['psnr_base']),
            'ssim_base': np.mean(overall_metrics['ssim_base']),
            'ssim_ours': np.mean(overall_metrics['ssim_ours']),
            'ssim_gain': np.mean(overall_metrics['ssim_ours']) - np.mean(overall_metrics['ssim_base']),
            'edge_preservation': np.mean(overall_metrics['edge_preservation']),
        },
        'per_layer': {},
        'cnr': {},
        'interpretability': {k: np.mean(v) for k, v in interpretability.items()},
    }

    # Per-layer results
    for name in ANATOMY_LAYERS.keys():
        results['per_layer'][name] = {
            'psnr_base': np.mean(layer_metrics_base[name]),
            'psnr_ours': np.mean(layer_metrics_ours[name]),
            'psnr_gain': np.mean(layer_metrics_ours[name]) - np.mean(layer_metrics_base[name]),
        }

    # CNR results
    cnr_keys = list(cnr_metrics_base[0].keys())
    for key in cnr_keys:
        base_vals = [m[key] for m in cnr_metrics_base]
        ours_vals = [m[key] for m in cnr_metrics_ours]
        results['cnr'][key] = {
            'base': np.mean(base_vals),
            'ours': np.mean(ours_vals),
            'gain': np.mean(ours_vals) - np.mean(base_vals),
        }

    # Create visualizations
    create_visualizations(vis_samples, results, output_dir)

    return results


def create_visualizations(samples, results, output_dir):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sample comparison
    if samples:
        fig, axes = plt.subplots(len(samples), 5, figsize=(15, 3*len(samples)))
        if len(samples) == 1:
            axes = [axes]

        for idx, sample in enumerate(samples):
            axes[idx][0].imshow(sample['noisy'][0].numpy(), cmap='gray')
            axes[idx][0].set_title('Noisy')
            axes[idx][0].axis('off')

            axes[idx][1].imshow(sample['base'][0].numpy(), cmap='gray')
            axes[idx][1].set_title('Baseline')
            axes[idx][1].axis('off')

            axes[idx][2].imshow(sample['ours'][0].numpy(), cmap='gray')
            axes[idx][2].set_title('Ours')
            axes[idx][2].axis('off')

            axes[idx][3].imshow(sample['clean'][0].numpy(), cmap='gray')
            axes[idx][3].set_title('Clean')
            axes[idx][3].axis('off')

            # Show layer probability as color overlay
            layer_prob = sample['layer_prob'].numpy()
            layer_vis = np.argmax(layer_prob, axis=0)
            axes[idx][4].imshow(layer_vis, cmap='viridis')
            axes[idx][4].set_title('Layer Detection')
            axes[idx][4].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_comparison.png'), dpi=150)
        plt.close()

    # 2. Per-layer PSNR comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = list(results['per_layer'].keys())
    x = np.arange(len(layers))
    width = 0.35

    base_psnr = [results['per_layer'][l]['psnr_base'] for l in layers]
    ours_psnr = [results['per_layer'][l]['psnr_ours'] for l in layers]

    bars1 = ax.bar(x - width/2, base_psnr, width, label='Baseline', color='#3498db')
    bars2 = ax.bar(x + width/2, ours_psnr, width, label='Ours', color='#2ecc71')

    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR by Anatomy Region')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in layers], fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add gain annotations
    for i, (b, o) in enumerate(zip(base_psnr, ours_psnr)):
        gain = o - b
        color = 'green' if gain > 0 else 'red'
        ax.annotate(f'{gain:+.2f}', xy=(i + width/2, o), ha='center', va='bottom',
                   fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_by_layer.png'), dpi=150)
    plt.close()

    # 3. Interpretability visualization
    if samples:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        sample = samples[0]

        axes[0].imshow(sample['coef_var'][0].numpy(), cmap='hot')
        axes[0].set_title('Coefficient of Variation\n(Speckle Indicator)')
        axes[0].axis('off')

        axes[1].imshow(sample['refinement'][0].numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[1].set_title('Refinement Map\n(Per-pixel Adaptation)')
        axes[1].axis('off')

        layer_prob = sample['layer_prob'].numpy()
        axes[2].imshow(np.argmax(layer_prob, axis=0), cmap='viridis')
        axes[2].set_title('Anatomy Layer Detection')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interpretability.png'), dpi=150)
        plt.close()


def print_report(results):
    """Print comprehensive evaluation report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("Interpretable Anatomy-Aware OCT Denoising")
    print("="*80)

    # Overall metrics
    print("\n" + "-"*80)
    print("1. OVERALL METRICS")
    print("-"*80)
    o = results['overall']
    print(f"{'Metric':<20} {'Baseline':>12} {'Ours':>12} {'Gain':>12}")
    print(f"{'-'*56}")
    print(f"{'PSNR (dB)':<20} {o['psnr_base']:>12.2f} {o['psnr_ours']:>12.2f} {o['psnr_gain']:>+12.2f}")
    print(f"{'SSIM':<20} {o['ssim_base']:>12.4f} {o['ssim_ours']:>12.4f} {o['ssim_gain']:>+12.4f}")
    print(f"{'Edge Preservation':<20} {'-':>12} {o['edge_preservation']:>12.4f} {'-':>12}")

    # Per-layer metrics
    print("\n" + "-"*80)
    print("2. PER-ANATOMY-REGION PSNR (dB)")
    print("-"*80)
    print(f"{'Region':<20} {'Baseline':>12} {'Ours':>12} {'Gain':>12}")
    print(f"{'-'*56}")

    for name, metrics in results['per_layer'].items():
        gain_str = f"{metrics['psnr_gain']:+.2f}"
        gain_color = "+" if metrics['psnr_gain'] > 0 else ""
        print(f"{name:<20} {metrics['psnr_base']:>12.2f} {metrics['psnr_ours']:>12.2f} {gain_str:>12}")

    # CNR metrics
    print("\n" + "-"*80)
    print("3. CONTRAST-TO-NOISE RATIO (CNR) BETWEEN LAYERS")
    print("-"*80)
    print(f"{'Layer Pair':<30} {'Baseline':>10} {'Ours':>10} {'Gain':>10}")
    print(f"{'-'*60}")

    for name, metrics in results['cnr'].items():
        short_name = name.replace('CNR_', '').replace('_', ' vs ')
        print(f"{short_name:<30} {metrics['base']:>10.3f} {metrics['ours']:>10.3f} {metrics['gain']:>+10.3f}")

    # Interpretability metrics
    print("\n" + "-"*80)
    print("4. INTERPRETABILITY METRICS")
    print("-"*80)
    interp = results['interpretability']
    print(f"  Coefficient of Variation (mean):    {interp['coef_var_mean']:.4f}")
    print(f"  Coefficient of Variation (std):     {interp['coef_var_std']:.4f}  [Higher = more per-pixel variation]")
    print(f"  Signal-Variance Correlation:        {interp['sig_var_corr']:.4f}  [Indicates shot noise presence]")
    print(f"  Horizontal Ratio:                   {interp['horiz_ratio']:.4f}  [Indicates banding presence]")
    print(f"  Layer Detection Entropy:            {interp['layer_entropy']:.4f}  [Lower = more confident layers]")
    print(f"  Refinement Magnitude:               {interp['refinement_mag']:.4f}  [Amount of correction applied]")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Overall PSNR Improvement:  {o['psnr_gain']:+.2f} dB")
    print(f"  Overall SSIM Improvement:  {o['ssim_gain']:+.4f}")

    # Find best and worst layer improvements
    gains = [(name, m['psnr_gain']) for name, m in results['per_layer'].items()]
    best_layer = max(gains, key=lambda x: x[1])
    worst_layer = min(gains, key=lambda x: x[1])

    print(f"\n  Best Layer Improvement:    {best_layer[0]} ({best_layer[1]:+.2f} dB)")
    print(f"  Worst Layer Improvement:   {worst_layer[0]} ({worst_layer[1]:+.2f} dB)")

    print("\n  Model demonstrates:")
    print("    - Per-pixel adaptive denoising (via physics features)")
    print("    - Anatomy-aware processing (via layer detection)")
    print("    - Interpretable decisions (visualizable feature maps)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/soft_conditioned/best.pth')
    parser.add_argument('--base_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--val_jsonl', default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_val', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='evaluation_results')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")

    # Load our model
    model = SoftConditionedDenoiser(backbone_ckpt=args.base_ckpt).to(args.device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}, using untrained model")

    # Load baseline model
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)
    ckpt = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
    base_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    base_model.eval()

    # Load data
    from torch.utils.data import DataLoader
    val_ds = OCTDenoiseDataset(args.val_jsonl, args.patch_size, args.max_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Evaluating on {len(val_ds)} samples...")

    # Run evaluation
    results = evaluate_model(model, base_model, val_loader, args.device, args.output_dir)

    # Print report
    print_report(results)

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
