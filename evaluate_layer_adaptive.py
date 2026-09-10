#!/usr/bin/env python3
"""
Evaluate layer-adaptive denoising advantages.

Compares:
1. Base NAFNet (no layer adaptation)
2. NAFNet + Layer-Specific Heads (layer adaptation)

Shows per-layer PSNR/SSIM improvements.
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_val', type=int, default=10)
    parser.add_argument('--nafnet_ckpt', default='outputs/nafnet_calibrated/nafnet_best.pth')
    parser.add_argument('--layer_adaptive_ckpt', default='outputs/tmi_v2_final/best_psnr.pth')
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
    print("LAYER-ADAPTIVE DENOISING EVALUATION")
    print("=" * 70)
    print(f"Samples: {len(samples)}")
    print()

    # 1. Base NAFNet (no layer adaptation)
    print("Loading Base NAFNet (no layer adaptation)...")
    nafnet = NAFNet(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32
    ).to(device)

    ckpt = torch.load(args.nafnet_ckpt, map_location=device, weights_only=False)
    nafnet.load_state_dict(ckpt['state_dict'])
    nafnet.eval()

    # 2. Layer-adaptive model - load NAFNet + LayerHeads only
    print("Loading Layer-Adaptive model...")
    try:
        from nsnd_oct.nsnd.models.layer_specific_heads import EnhancedLayerSpecificDenoiser

        # Load checkpoint
        ckpt_la = torch.load(args.layer_adaptive_ckpt, map_location=device, weights_only=False)
        ckpt_args = ckpt_la.get('args', {})
        state = ckpt_la.get('model_state_dict', ckpt_la.get('state_dict', ckpt_la))

        # Create NAFNet with layer heads
        layer_nafnet = NAFNet(
            img_channel=1, width=64,
            enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
            middle_blk_num=2, cond_dim=32
        ).to(device)

        layer_heads = EnhancedLayerSpecificDenoiser(
            encoder_channels=64,
            n_layer_classes=4,
            hidden_channels=ckpt_args.get('head_hidden_channels', 64),
            num_head_blocks=ckpt_args.get('num_head_blocks', 2),
            dropout=ckpt_args.get('head_dropout', 0.1),
        ).to(device)

        # Load NAFNet weights
        nafnet_state = {k.replace('nafnet.', ''): v for k, v in state.items() if k.startswith('nafnet.')}
        layer_nafnet.load_state_dict(nafnet_state, strict=False)

        # Load layer heads weights
        heads_state = {k.replace('layer_heads.', ''): v for k, v in state.items() if k.startswith('layer_heads.')}
        layer_heads.load_state_dict(heads_state, strict=False)

        layer_nafnet.eval()
        layer_heads.eval()

        print(f"  Best PSNR: {ckpt_la.get('best_psnr', 'N/A')}")
        print(f"  NAFNet keys loaded: {len(nafnet_state)}")
        print(f"  Layer heads keys loaded: {len(heads_state)}")
        has_layer_model = True
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        has_layer_model = False

    # Evaluate
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    nafnet_metrics = {'psnr': [], 'ssim': [], 'layer_psnr': [[] for _ in range(4)]}
    layer_metrics = {'psnr': [], 'ssim': [], 'layer_psnr': [[] for _ in range(4)]}
    noisy_metrics = {'psnr': [], 'ssim': []}

    print("\nEvaluating...")
    for i, sample in enumerate(samples):
        clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        noisy = add_noise(clean, 'duke17', 1.0)

        H, W = clean.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        noisy_padded = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode='reflect')

        x = torch.from_numpy(noisy_padded).float().unsqueeze(0).unsqueeze(0).to(device)

        # Base NAFNet
        with torch.no_grad():
            y_nafnet = nafnet(x, spatial_map=None, basis=None, alpha=0.0, gate=None)
        denoised_nafnet = y_nafnet.squeeze().cpu().numpy()[:H, :W]

        # Layer-adaptive: NAFNet features -> Layer heads
        if has_layer_model:
            with torch.no_grad():
                # Get NAFNet features (we need encoder features, not final output)
                # Since NAFNet doesn't expose features directly, use its output and apply layer heads
                y_layer_nafnet = layer_nafnet(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

                # Get encoder features for layer heads (use intro features)
                features = layer_nafnet.intro(x)  # [B, width, H, W]

                # Apply layer-specific heads (no seg mask - use uniform mixing)
                head_results = layer_heads(features, seg_mask=None, seg_probs=None, boundary_mask=None)

                # Combine NAFNet output with layer-specific refinement
                layer_output = head_results['output']
                denoised_layer = (y_layer_nafnet + layer_output).squeeze().cpu().numpy()[:H, :W]

        # Metrics
        noisy_metrics['psnr'].append(compute_psnr(noisy, clean))
        noisy_metrics['ssim'].append(compute_ssim(noisy, clean))

        nafnet_metrics['psnr'].append(compute_psnr(denoised_nafnet, clean))
        nafnet_metrics['ssim'].append(compute_ssim(denoised_nafnet, clean))

        if has_layer_model:
            layer_metrics['psnr'].append(compute_psnr(denoised_layer, clean))
            layer_metrics['ssim'].append(compute_ssim(denoised_layer, clean))

        # Per-layer metrics
        layer_regions = get_layer_regions(H)
        for j, (row_start, row_end) in enumerate(layer_regions):
            clean_layer = clean[row_start:row_end, :]
            nafnet_layer = denoised_nafnet[row_start:row_end, :]
            nafnet_metrics['layer_psnr'][j].append(compute_psnr(nafnet_layer, clean_layer))

            if has_layer_model:
                layer_region = denoised_layer[row_start:row_end, :]
                layer_metrics['layer_psnr'][j].append(compute_psnr(layer_region, clean_layer))

        print(f"  Sample {i+1}/{len(samples)}: NAFNet={nafnet_metrics['psnr'][-1]:.2f}dB", end="")
        if has_layer_model:
            print(f", Layer-Adaptive={layer_metrics['psnr'][-1]:.2f}dB")
        else:
            print()

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n[OVERALL METRICS]")
    print(f"{'Method':<30} {'PSNR (dB)':<15} {'SSIM':<15} {'PSNR Gain':<15}")
    print("-" * 70)

    noisy_psnr = np.mean(noisy_metrics['psnr'])
    noisy_ssim = np.mean(noisy_metrics['ssim'])
    nafnet_psnr = np.mean(nafnet_metrics['psnr'])
    nafnet_ssim = np.mean(nafnet_metrics['ssim'])

    print(f"{'Noisy Input':<30} {noisy_psnr:.2f}          {noisy_ssim:.4f}          -")
    print(f"{'Base NAFNet (no adaptation)':<30} {nafnet_psnr:.2f}          {nafnet_ssim:.4f}          +{nafnet_psnr-noisy_psnr:.2f}")

    if has_layer_model:
        layer_psnr = np.mean(layer_metrics['psnr'])
        layer_ssim = np.mean(layer_metrics['ssim'])
        print(f"{'Layer-Adaptive NAFNet':<30} {layer_psnr:.2f}          {layer_ssim:.4f}          +{layer_psnr-noisy_psnr:.2f}")
        improvement = layer_psnr - nafnet_psnr
        print(f"\n{'Layer-adaptive improvement:':<30} {'+' if improvement > 0 else ''}{improvement:.2f} dB")

    print("\n[PER-LAYER PSNR (dB)]")
    print(f"{'Layer':<20} {'Base NAFNet':<15} {'Layer-Adapt':<15} {'Diff':<15}")
    print("-" * 60)

    for j, name in enumerate(layer_names):
        base = np.mean(nafnet_metrics['layer_psnr'][j])
        if has_layer_model:
            adapt = np.mean(layer_metrics['layer_psnr'][j])
            diff = adapt - base
            print(f"{name:<20} {base:.2f}           {adapt:.2f}           {'+' if diff > 0 else ''}{diff:.2f}")
        else:
            print(f"{name:<20} {base:.2f}           N/A             N/A")

    print("\n" + "=" * 70)
    print("SOTA COMPARISON (SNA-SKAN, unsupervised)")
    print("=" * 70)
    print("  Duke17: PSNR=26.65dB, SSIM=0.814")
    print("  Duke28: PSNR=27.84dB, SSIM=0.827")

    if has_layer_model:
        gap = layer_psnr - 26.65
        print(f"\n  Layer-Adaptive: PSNR={layer_psnr:.2f}dB, SSIM={layer_ssim:.4f}")
        print(f"  GAP TO SOTA: {'+' if gap > 0 else ''}{gap:.2f} dB")
    else:
        gap = nafnet_psnr - 26.65
        print(f"\n  Base NAFNet: PSNR={nafnet_psnr:.2f}dB, SSIM={nafnet_ssim:.4f}")
        print(f"  GAP TO SOTA: {'+' if gap > 0 else ''}{gap:.2f} dB")

    print("=" * 70)


if __name__ == '__main__':
    main()
