#!/usr/bin/env python3
"""
Compare NAFNet denoising with and without layer-adaptive refinement.

This script evaluates:
1. Base NAFNet (no layer adaptation)
2. NAFNet + Layer-adaptive refinement (neuro-symbolic)
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet

# Import calibrated noise
try:
    from calibrated_oct_noise import add_calibrated_oct_noise
    HAS_CALIBRATED_NOISE = True
except ImportError:
    HAS_CALIBRATED_NOISE = False


def add_noise(image, profile='duke17', scale=1.0):
    """Add calibrated OCT noise."""
    if HAS_CALIBRATED_NOISE:
        return add_calibrated_oct_noise(image, dataset=profile, noise_scale=scale, random_mix=True)

    # Fallback
    params = {
        'duke17': {'speckle': 0.50, 'gaussian': 0.10},
        'duke28': {'speckle': 0.48, 'gaussian': 0.10},
        'pku37':  {'speckle': 0.35, 'gaussian': 0.07},
    }
    p = params.get(profile, params['duke17'])

    H, W = image.shape
    speckle = 1.0 + p['speckle'] * scale * (np.random.exponential(1.0, (H, W)) - 1.0)
    noisy = image * np.maximum(speckle, 0.01)
    gaussian = np.random.randn(H, W).astype(np.float32) * p['gaussian'] * scale
    noisy = noisy + gaussian
    return np.clip(noisy, 0, 1).astype(np.float32)


def compute_psnr(pred, target):
    """Compute PSNR."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 50.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM."""
    from scipy.ndimage import uniform_filter
    C1, C2 = 0.01**2, 0.03**2

    mu_pred = uniform_filter(pred, window_size)
    mu_target = uniform_filter(target, window_size)

    sigma_pred_sq = uniform_filter(pred**2, window_size) - mu_pred**2
    sigma_target_sq = uniform_filter(target**2, window_size) - mu_target**2
    sigma_pred_target = uniform_filter(pred*target, window_size) - mu_pred*mu_target

    ssim = ((2*mu_pred*mu_target + C1)*(2*sigma_pred_target + C2)) / \
           ((mu_pred**2 + mu_target**2 + C1)*(sigma_pred_sq + sigma_target_sq + C2))
    return np.mean(ssim)


def get_layer_regions(H, num_layers=4):
    """Get approximate layer regions for OCT image."""
    # Approximate retinal layer positions (normalized)
    # ILM: ~0.15-0.25, GCL: ~0.25-0.35, INL/OPL/ONL: ~0.35-0.55, IS/OS/RPE: ~0.55-0.75
    layer_boundaries = [
        (0.15, 0.30),  # RNFL_GCL
        (0.30, 0.50),  # INL_OPL_ONL
        (0.50, 0.65),  # IS_OS
        (0.65, 0.85),  # RPE_Choroid
    ]

    regions = []
    for start, end in layer_boundaries:
        row_start = int(H * start)
        row_end = int(H * end)
        regions.append((row_start, row_end))

    return regions


def evaluate_model(model, samples, device, noise_profile='duke17'):
    """Evaluate model on samples."""
    model.eval()

    metrics = {
        'psnr_noisy': [],
        'psnr_denoised': [],
        'ssim_noisy': [],
        'ssim_denoised': [],
        'layer_psnr': [[] for _ in range(4)],  # Per-layer PSNR
    }

    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    for sample in samples:
        clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        noisy = add_noise(clean, noise_profile, 1.0)

        H, W = clean.shape

        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        noisy_padded = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode='reflect')

        x = torch.from_numpy(noisy_padded).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            y = model(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        denoised = y.squeeze().cpu().numpy()[:H, :W]

        # Overall metrics
        metrics['psnr_noisy'].append(compute_psnr(noisy, clean))
        metrics['psnr_denoised'].append(compute_psnr(denoised, clean))
        metrics['ssim_noisy'].append(compute_ssim(noisy, clean))
        metrics['ssim_denoised'].append(compute_ssim(denoised, clean))

        # Per-layer metrics
        layer_regions = get_layer_regions(H)
        for i, (row_start, row_end) in enumerate(layer_regions):
            clean_layer = clean[row_start:row_end, :]
            denoised_layer = denoised[row_start:row_end, :]
            metrics['layer_psnr'][i].append(compute_psnr(denoised_layer, clean_layer))

    # Aggregate
    results = {
        'psnr_noisy': np.mean(metrics['psnr_noisy']),
        'psnr_denoised': np.mean(metrics['psnr_denoised']),
        'ssim_noisy': np.mean(metrics['ssim_noisy']),
        'ssim_denoised': np.mean(metrics['ssim_denoised']),
    }

    for i, name in enumerate(layer_names):
        results[f'layer_{name}'] = np.mean(metrics['layer_psnr'][i])

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_val', type=int, default=10)
    parser.add_argument('--nafnet_ckpt', default='outputs/nafnet_calibrated/nafnet_best.pth')
    parser.add_argument('--nsnd_ckpt', default='outputs/neurosymbolic_denoising/best_model.pth')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load samples
    samples = []
    with open(args.val_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:args.max_val]

    print("=" * 70)
    print("COMPARISON: WITH vs WITHOUT LAYER-ADAPTIVE DENOISING")
    print("=" * 70)
    print(f"Samples: {len(samples)}")
    print()

    # 1. Base NAFNet (no layer adaptation)
    print("Loading Base NAFNet...")
    nafnet = NAFNet(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32
    ).to(device)

    ckpt = torch.load(args.nafnet_ckpt, map_location=device, weights_only=False)
    nafnet.load_state_dict(ckpt['state_dict'])

    print("Evaluating Base NAFNet...")
    nafnet_results = evaluate_model(nafnet, samples, device)

    # 2. NSND with layer adaptation (if checkpoint exists)
    nsnd_results = None
    if os.path.exists(args.nsnd_ckpt):
        print("\nLoading Neuro-Symbolic model with layer adaptation...")
        # For fair comparison, we evaluate the NAFNet part of NSND
        # The full NSND includes layer-specific refinement
        try:
            from train_neurosymbolic_denoising import NeuroSymbolicDenoiser

            nsnd = NeuroSymbolicDenoiser(
                hidden_channels=48,
                nafnet_width=64,
            ).to(device)

            ckpt = torch.load(args.nsnd_ckpt, map_location=device, weights_only=False)
            # Try multiple possible keys for state dict
            state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            nsnd.load_state_dict(state, strict=False)
            print(f"  Loaded NSND (epoch {ckpt.get('epoch', '?')}, PSNR={ckpt.get('psnr', ckpt.get('best_psnr', 'N/A'))})")

            # Evaluate NSND
            print("Evaluating Neuro-Symbolic model...")
            nsnd.eval()

            nsnd_metrics = {
                'psnr_noisy': [],
                'psnr_denoised': [],
                'ssim_noisy': [],
                'ssim_denoised': [],
                'layer_psnr': [[] for _ in range(4)],
            }

            layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

            for sample in samples:
                clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
                noisy = add_noise(clean, 'duke17', 1.0)

                H, W = clean.shape
                pad_h = (8 - H % 8) % 8
                pad_w = (8 - W % 8) % 8
                noisy_padded = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode='reflect')

                x = torch.from_numpy(noisy_padded).float().unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = nsnd(x, return_symbolic=False)

                denoised = outputs['denoised'].squeeze().cpu().numpy()[:H, :W]

                nsnd_metrics['psnr_noisy'].append(compute_psnr(noisy, clean))
                nsnd_metrics['psnr_denoised'].append(compute_psnr(denoised, clean))
                nsnd_metrics['ssim_noisy'].append(compute_ssim(noisy, clean))
                nsnd_metrics['ssim_denoised'].append(compute_ssim(denoised, clean))

                layer_regions = get_layer_regions(H)
                for i, (row_start, row_end) in enumerate(layer_regions):
                    clean_layer = clean[row_start:row_end, :]
                    denoised_layer = denoised[row_start:row_end, :]
                    nsnd_metrics['layer_psnr'][i].append(compute_psnr(denoised_layer, clean_layer))

            nsnd_results = {
                'psnr_noisy': np.mean(nsnd_metrics['psnr_noisy']),
                'psnr_denoised': np.mean(nsnd_metrics['psnr_denoised']),
                'ssim_noisy': np.mean(nsnd_metrics['ssim_noisy']),
                'ssim_denoised': np.mean(nsnd_metrics['ssim_denoised']),
            }
            for i, name in enumerate(layer_names):
                nsnd_results[f'layer_{name}'] = np.mean(nsnd_metrics['layer_psnr'][i])

        except Exception as e:
            print(f"Error loading NSND: {e}")
            nsnd_results = None

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n[OVERALL METRICS]")
    print(f"{'Method':<30} {'PSNR (dB)':<15} {'SSIM':<15} {'PSNR Gain':<15}")
    print("-" * 70)
    print(f"{'Noisy Input':<30} {nafnet_results['psnr_noisy']:.2f}          {nafnet_results['ssim_noisy']:.4f}          -")
    print(f"{'Base NAFNet (no adaptation)':<30} {nafnet_results['psnr_denoised']:.2f}          {nafnet_results['ssim_denoised']:.4f}          +{nafnet_results['psnr_denoised']-nafnet_results['psnr_noisy']:.2f}")

    if nsnd_results:
        print(f"{'NSND (layer-adaptive)':<30} {nsnd_results['psnr_denoised']:.2f}          {nsnd_results['ssim_denoised']:.4f}          +{nsnd_results['psnr_denoised']-nsnd_results['psnr_noisy']:.2f}")
        improvement = nsnd_results['psnr_denoised'] - nafnet_results['psnr_denoised']
        print(f"\n{'Layer-adaptive improvement:':<30} {'+' if improvement > 0 else ''}{improvement:.2f} dB")

    print("\n[PER-LAYER PSNR (dB)]")
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
    print(f"{'Layer':<20} {'Base NAFNet':<15} {'NSND':<15} {'Diff':<15}")
    print("-" * 60)

    for name in layer_names:
        base = nafnet_results.get(f'layer_{name}', 0)
        nsnd = nsnd_results.get(f'layer_{name}', 0) if nsnd_results else 0
        diff = nsnd - base if nsnd_results else 0
        print(f"{name:<20} {base:.2f}           {nsnd:.2f}           {'+' if diff > 0 else ''}{diff:.2f}")

    print("\n" + "=" * 70)
    print("SOTA COMPARISON (SNA-SKAN, unsupervised)")
    print("=" * 70)
    print("  Duke17: PSNR=26.65dB, SSIM=0.814")
    print("  Duke28: PSNR=27.84dB, SSIM=0.827")

    if nsnd_results:
        gap = nsnd_results['psnr_denoised'] - 26.65
        print(f"\n  OUR MODEL: PSNR={nsnd_results['psnr_denoised']:.2f}dB, SSIM={nsnd_results['ssim_denoised']:.4f}")
        print(f"  GAP TO SOTA: {'+' if gap > 0 else ''}{gap:.2f} dB")
    else:
        gap = nafnet_results['psnr_denoised'] - 26.65
        print(f"\n  Base NAFNet: PSNR={nafnet_results['psnr_denoised']:.2f}dB")
        print(f"  GAP TO SOTA: {'+' if gap > 0 else ''}{gap:.2f} dB")

    print("=" * 70)


if __name__ == '__main__':
    main()
