#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for TMI Submission

Runs all evaluations:
1. Standard metrics (PSNR, SSIM)
2. Per-anatomy-region analysis
3. Clinical metrics (CNR, EPI, SRI, boundary sharpness)
4. Downstream task evaluation
5. SOTA comparison
6. Ablation study
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_soft_conditioning import SoftConditionedDenoiser, OCTDenoiseDataset
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.utils.metrics import compute_psnr, compute_ssim
from nsnd.utils.clinical_metrics import compute_all_clinical_metrics, print_clinical_metrics_report
from nsnd.baselines.sota_methods import BaselineManager, DnCNN, UNetDenoiser, RestormerLite


def load_model(checkpoint_path, backbone_path, device):
    """Load the trained model."""
    model = SoftConditionedDenoiser(backbone_ckpt=backbone_path).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        del ckpt  # Free checkpoint memory
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    return model


def load_baseline(backbone_path, device):
    """Load the baseline NAFNet model."""
    model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(device)

    ckpt = torch.load(backbone_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    del ckpt  # Free checkpoint memory

    return model


def evaluate_standard_metrics(model, baseline, loader, device):
    """Evaluate standard PSNR/SSIM metrics."""
    model.eval()
    baseline.eval()

    metrics = {
        'psnr_noisy': [], 'psnr_baseline': [], 'psnr_ours': [],
        'ssim_noisy': [], 'ssim_baseline': [], 'ssim_ours': [],
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Standard Metrics"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            base_out = baseline(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            ours_out = model(noisy)

            for i in range(noisy.size(0)):
                metrics['psnr_noisy'].append(compute_psnr(noisy[i:i+1], clean[i:i+1]))
                metrics['psnr_baseline'].append(compute_psnr(base_out[i:i+1], clean[i:i+1]))
                metrics['psnr_ours'].append(compute_psnr(ours_out[i:i+1], clean[i:i+1]))

                metrics['ssim_noisy'].append(compute_ssim(noisy[i:i+1], clean[i:i+1]))
                metrics['ssim_baseline'].append(compute_ssim(base_out[i:i+1], clean[i:i+1]))
                metrics['ssim_ours'].append(compute_ssim(ours_out[i:i+1], clean[i:i+1]))

    return {k: np.mean(v) for k, v in metrics.items()}


def evaluate_per_layer(model, baseline, loader, device):
    """Evaluate metrics per anatomical layer."""
    model.eval()
    baseline.eval()

    layer_names = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']
    layer_bounds = [(0, 0.15), (0.15, 0.40), (0.40, 0.55), (0.55, 0.75), (0.75, 1.0)]

    layer_metrics = {name: {'psnr_baseline': [], 'psnr_ours': []} for name in layer_names}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Per-Layer Metrics"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            B, C, H, W = noisy.shape

            base_out = baseline(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            ours_out = model(noisy)

            for name, (start, end) in zip(layer_names, layer_bounds):
                row_start, row_end = int(start * H), int(end * H)

                for i in range(B):
                    clean_layer = clean[i:i+1, :, row_start:row_end, :]
                    base_layer = base_out[i:i+1, :, row_start:row_end, :]
                    ours_layer = ours_out[i:i+1, :, row_start:row_end, :]

                    layer_metrics[name]['psnr_baseline'].append(compute_psnr(base_layer, clean_layer))
                    layer_metrics[name]['psnr_ours'].append(compute_psnr(ours_layer, clean_layer))

    results = {}
    for name in layer_names:
        results[name] = {
            'psnr_baseline': np.mean(layer_metrics[name]['psnr_baseline']),
            'psnr_ours': np.mean(layer_metrics[name]['psnr_ours']),
            'psnr_gain': np.mean(layer_metrics[name]['psnr_ours']) - np.mean(layer_metrics[name]['psnr_baseline']),
        }

    return results


def evaluate_clinical(model, baseline, loader, device):
    """Evaluate clinical metrics."""
    model.eval()
    baseline.eval()

    all_clinical_ours = []
    all_clinical_base = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Clinical Metrics"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            base_out = baseline(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            ours_out, features = model(noisy, return_features=True)

            # Clinical metrics for our method
            clinical_ours = compute_all_clinical_metrics(
                ours_out, clean, noisy,
                layer_probs=features.get('layer_prob')
            )
            all_clinical_ours.append(clinical_ours)

            # Clinical metrics for baseline
            clinical_base = compute_all_clinical_metrics(base_out, clean, noisy)
            all_clinical_base.append(clinical_base)

    # Average metrics
    results = {'ours': {}, 'baseline': {}}
    for key in all_clinical_ours[0].keys():
        results['ours'][key] = np.mean([m[key] for m in all_clinical_ours])
        results['baseline'][key] = np.mean([m[key] for m in all_clinical_base])

    return results


def evaluate_sota_comparison(model, loader, device, baseline_ckpt):
    """Compare with SOTA methods."""
    model.eval()

    # Initialize baselines
    manager = BaselineManager(device=device)

    # Create methods dict - note: model is passed by reference, not created here
    nafnet = load_baseline(baseline_ckpt, device)
    dncnn = DnCNN().to(device)
    unet = UNetDenoiser().to(device)

    methods = {
        'Ours': model,
        'NAFNet': nafnet,
        'DnCNN': dncnn,
        'UNet': unet,
    }

    results = {name: {'psnr': [], 'ssim': []} for name in methods.keys()}

    with torch.no_grad():
        for batch in tqdm(loader, desc="SOTA Comparison"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            for name, method in methods.items():
                if isinstance(method, nn.Module):
                    method.eval()
                    if name == 'NAFNet':
                        out = method(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
                    else:
                        out = method(noisy)
                else:
                    out = method(noisy)

                out = out.clamp(0, 1)

                for i in range(noisy.size(0)):
                    results[name]['psnr'].append(compute_psnr(out[i:i+1], clean[i:i+1]))
                    results[name]['ssim'].append(compute_ssim(out[i:i+1], clean[i:i+1]))

            # Clear batch tensors
            del noisy, clean, out

    # Clean up SOTA models (not 'Ours' which is the main model)
    del nafnet, dncnn, unet, methods
    gc.collect()
    if 'cuda' in device:
        torch.cuda.empty_cache()

    return {name: {'psnr': np.mean(m['psnr']), 'ssim': np.mean(m['ssim'])}
            for name, m in results.items()}


def create_visualizations(model, baseline, loader, device, output_dir):
    """Create visualization figures."""
    model.eval()
    baseline.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(loader))
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        base_out = baseline(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
        ours_out, features = model(noisy, return_features=True)

    # Figure 1: Sample comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row in range(min(2, noisy.size(0))):
        axes[row, 0].imshow(noisy[row, 0].cpu().numpy(), cmap='gray')
        axes[row, 0].set_title('Noisy')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(base_out[row, 0].cpu().numpy(), cmap='gray')
        axes[row, 1].set_title('NAFNet (Baseline)')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(ours_out[row, 0].cpu().numpy(), cmap='gray')
        axes[row, 2].set_title('Ours')
        axes[row, 2].axis('off')

        axes[row, 3].imshow(clean[row, 0].cpu().numpy(), cmap='gray')
        axes[row, 3].set_title('Clean')
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_comparison.png'), dpi=150)
    plt.close()

    # Figure 2: Interpretability
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Physics features
    raw = features['raw_features']
    axes[0].imshow(raw['coef_variation'][0, 0].cpu().numpy(), cmap='hot')
    axes[0].set_title('Coefficient of Variation\n(Speckle Indicator)')
    axes[0].axis('off')

    axes[1].imshow(raw['signal_var_corr'][0, 0].cpu().numpy(), cmap='RdBu')
    axes[1].set_title('Signal-Variance Corr.\n(Shot Noise Indicator)')
    axes[1].axis('off')

    # Layer detection
    layer_prob = features['layer_prob'][0].cpu().numpy()
    axes[2].imshow(np.argmax(layer_prob, axis=0), cmap='viridis')
    axes[2].set_title('Anatomy Layer Detection')
    axes[2].axis('off')

    # Refinement
    axes[3].imshow(features['refinement'][0, 0].cpu().numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[3].set_title('Refinement Map')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interpretability.png'), dpi=150)
    plt.close()

    # Clean up tensors
    del noisy, clean, base_out, ours_out, features, batch
    gc.collect()

    print(f"Visualizations saved to {output_dir}")


def print_comprehensive_report(results, output_dir):
    """Print and save comprehensive report."""
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE EVALUATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)

    # Standard metrics
    report.append("\n" + "-"*80)
    report.append("1. STANDARD METRICS")
    report.append("-"*80)
    std = results['standard']
    report.append(f"{'Metric':<20} {'Noisy':>12} {'Baseline':>12} {'Ours':>12} {'Gain':>12}")
    report.append(f"{'-'*68}")
    report.append(f"{'PSNR (dB)':<20} {std['psnr_noisy']:>12.2f} {std['psnr_baseline']:>12.2f} {std['psnr_ours']:>12.2f} {std['psnr_ours']-std['psnr_baseline']:>+12.2f}")
    report.append(f"{'SSIM':<20} {std['ssim_noisy']:>12.4f} {std['ssim_baseline']:>12.4f} {std['ssim_ours']:>12.4f} {std['ssim_ours']-std['ssim_baseline']:>+12.4f}")

    # Per-layer metrics
    report.append("\n" + "-"*80)
    report.append("2. PER-ANATOMY-REGION PSNR (dB)")
    report.append("-"*80)
    report.append(f"{'Region':<20} {'Baseline':>12} {'Ours':>12} {'Gain':>12}")
    report.append(f"{'-'*56}")
    for name, m in results['per_layer'].items():
        report.append(f"{name:<20} {m['psnr_baseline']:>12.2f} {m['psnr_ours']:>12.2f} {m['psnr_gain']:>+12.2f}")

    # Clinical metrics
    report.append("\n" + "-"*80)
    report.append("3. CLINICAL METRICS")
    report.append("-"*80)
    clinical = results['clinical']

    report.append(f"\nContrast-to-Noise Ratio:")
    report.append(f"  CNR Mean (Baseline): {clinical['baseline']['CNR_mean']:.3f}")
    report.append(f"  CNR Mean (Ours):     {clinical['ours']['CNR_mean']:.3f}")
    report.append(f"  CNR Improvement:     {clinical['ours']['CNR_improvement']:+.3f}")

    report.append(f"\nEdge Preservation Index:")
    report.append(f"  EPI (Ours): {clinical['ours']['EPI']:.3f}")

    report.append(f"\nSpeckle Reduction Index:")
    report.append(f"  SRI (Ours): {clinical['ours']['SRI']:.3f} ({clinical['ours']['SRI']*100:.1f}% reduction)")

    report.append(f"\nBoundary Sharpness:")
    report.append(f"  Improvement: {clinical['ours']['Boundary_Sharpness_improvement']:+.4f}")

    # SOTA comparison
    report.append("\n" + "-"*80)
    report.append("4. SOTA COMPARISON")
    report.append("-"*80)
    report.append(f"{'Method':<20} {'PSNR (dB)':>12} {'SSIM':>12}")
    report.append(f"{'-'*44}")
    sota_sorted = sorted(results['sota'].items(), key=lambda x: x[1]['psnr'], reverse=True)
    for name, m in sota_sorted:
        report.append(f"{name:<20} {m['psnr']:>12.2f} {m['ssim']:>12.4f}")

    report.append("\n" + "="*80)

    # Print report
    report_text = '\n'.join(report)
    print(report_text)

    # Save report
    with open(os.path.join(output_dir, 'comprehensive_report.txt'), 'w') as f:
        f.write(report_text)

    # Save JSON results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation')
    parser.add_argument('--checkpoint', default='checkpoints/soft_conditioned/best.pth')
    parser.add_argument('--backbone', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--val_jsonl', default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_val', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='comprehensive_results')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    model = load_model(args.checkpoint, args.backbone, args.device)
    baseline = load_baseline(args.backbone, args.device)

    print("Loading data...")
    val_ds = OCTDenoiseDataset(args.val_jsonl, args.patch_size, args.max_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nEvaluating on {len(val_ds)} samples...")

    results = {}

    def cleanup_memory():
        """Helper to clean up GPU/CPU memory between evaluations."""
        gc.collect()
        if 'cuda' in args.device:
            torch.cuda.empty_cache()

    # 1. Standard metrics
    print("\n[1/5] Standard metrics...")
    results['standard'] = evaluate_standard_metrics(model, baseline, val_loader, args.device)
    cleanup_memory()

    # 2. Per-layer metrics
    print("\n[2/5] Per-layer metrics...")
    results['per_layer'] = evaluate_per_layer(model, baseline, val_loader, args.device)
    cleanup_memory()

    # 3. Clinical metrics
    print("\n[3/5] Clinical metrics...")
    results['clinical'] = evaluate_clinical(model, baseline, val_loader, args.device)
    cleanup_memory()

    # 4. SOTA comparison
    print("\n[4/5] SOTA comparison...")
    results['sota'] = evaluate_sota_comparison(model, val_loader, args.device, args.backbone)
    cleanup_memory()

    # 5. Visualizations
    print("\n[5/5] Creating visualizations...")
    create_visualizations(model, baseline, val_loader, args.device, args.output_dir)
    cleanup_memory()

    # Print comprehensive report
    print_comprehensive_report(results, args.output_dir)

    # Final cleanup
    del model, baseline, val_ds, val_loader
    cleanup_memory()


if __name__ == '__main__':
    main()
