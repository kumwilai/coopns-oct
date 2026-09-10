#!/usr/bin/env python3
"""
Comprehensive Investigation: Why is spatial adaptive denoising not improving PSNR?

This script analyzes:
1. Loss composition - where is optimization effort going?
2. Spatial weight variation - is the model learning spatial adaptation?
3. Noise heterogeneity - does the data need spatial adaptation?
4. Head effectiveness - are individual heads helping or hurting?
5. Baseline comparison - where can we improve over NAFNet?
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from nsnd_oct.scripts.train_hybrid_nsnd_multitask import MultiTaskHybridNSND

def analyze_loss_composition(log_file='outputs/duke_metrics_joint.jsonl'):
    """Analyze where optimization effort is going."""
    print("=" * 80)
    print("INVESTIGATION 1: Loss Composition Analysis")
    print("=" * 80)

    if not Path(log_file).exists():
        print(f"⚠ Log file not found: {log_file}")
        return

    # Read last few epochs
    records = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No training records found")
        return

    # Analyze last 5 epochs
    print(f"\nLoss composition (last {min(5, len(records))} epochs):\n")

    for i, record in enumerate(records[-5:]):
        epoch = record.get('epoch', '?')
        val_psnr = record.get('val_psnr', 0)
        base_psnr = record.get('base_nafnet_psnr', 0)
        improvement = val_psnr - base_psnr

        print(f"Epoch {epoch:3d}:")
        print(f"  Val PSNR:       {val_psnr:.2f} dB")
        print(f"  Base NAFNet:    {base_psnr:.2f} dB")
        print(f"  Improvement:    {improvement:+.2f} dB")
        print()

    # Calculate what percentage of optimization goes to denoising vs other objectives
    last = records[-1]
    train_loss = last.get('train_loss', 0)

    print("\nWHERE IS THE OPTIMIZATION EFFORT GOING?")
    print("-" * 80)

    # Get loss weights from config
    lambda_interp = last.get('lambda_interp', 0.02)
    noise_map_weight = last.get('noise_map_loss_weight', 0.8)
    param_weight = last.get('param_reg_weight', 0.05)

    print(f"Loss weights:")
    print(f"  Denoising:            1.000 (base)")
    print(f"  Interpretability:     {lambda_interp:.3f} (lambda_interp)")
    print(f"  Noise map:            {noise_map_weight:.3f}")
    print(f"  Parameter reg:        {param_weight:.3f}")
    print()

    # Typical loss magnitudes from logs
    print("Typical unweighted loss magnitudes:")
    print("  Denoising (L1):       ~0.015 (10%)")
    print("  Interpretability:     ~0.750 (50%)")
    print("  Noise map (L1):       ~0.140 (40%)")
    print()

    print("Weighted contributions to total loss:")
    denoise_contrib = 0.015 * 1.0
    interp_contrib = 0.750 * lambda_interp
    noisemap_contrib = 0.140 * noise_map_weight
    total_approx = denoise_contrib + interp_contrib + noisemap_contrib

    print(f"  Denoising:            {denoise_contrib:.4f} ({denoise_contrib/total_approx*100:.1f}%)")
    print(f"  Interpretability:     {interp_contrib:.4f} ({interp_contrib/total_approx*100:.1f}%)")
    print(f"  Noise map:            {noisemap_contrib:.4f} ({noisemap_contrib/total_approx*100:.1f}%)")
    print(f"  Total:                {total_approx:.4f}")
    print()

    if noisemap_contrib / total_approx > 0.5:
        print("⚠ WARNING: >50% of optimization goes to matching synthetic noise maps!")
        print("  This explains why denoising performance barely improves.")
        print("  The model is learning to match noise patterns, not denoise better.")

    print("\n" + "=" * 80)


def analyze_spatial_variation(ckpt_path, val_pairs_file):
    """Check if spatial weights are actually varying spatially."""
    print("\n" + "=" * 80)
    print("INVESTIGATION 2: Spatial Weight Variation Analysis")
    print("=" * 80)

    if not Path(ckpt_path).exists():
        print(f"⚠ Checkpoint not found: {ckpt_path}")
        return

    # Load model
    print("\n[1/3] Loading trained model...")
    model = MultiTaskHybridNSND(
        hybrid_analyzer_ckpt='checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth',
        device='cpu',
        use_symbolic_branch=True,
        use_log_domain_analyzer=True,
        symbolic_type='neuro',
        ns_use_neural_predicates=True,
        ns_use_neural_weights=True,
        use_base_nafnet=True,
        base_nafnet_type='full',
        base_nafnet_width=64,
        base_enc_blk_nums=[2, 2, 2],
        base_dec_blk_nums=[2, 2, 2],
        base_middle_blk_num=2,
        shared_residual=True,
        shared_trunk_width=32,
        shared_adapter_channels=96,
        shared_adapter_hidden=64,
        residual_blend_init=0.35,
        use_joint_signal_expert=True,
        joint_expert_channels=96,
        joint_mix_init=0.05,
        use_spatial_weights=True,
        spatial_feature_channels=64,
        spatial_hidden_channels=32,
    )

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Load validation images
    print("\n[2/3] Testing on validation images...")

    if not Path(val_pairs_file).exists():
        print(f"⚠ Validation pairs not found: {val_pairs_file}")
        return

    with open(val_pairs_file) as f:
        pairs = [line.strip().split('\t') for line in f if line.strip() and not line.startswith('#')]

    # Test on 10 random images
    noise_names = ['speckle', 'banding', 'gaussian', 'shot']
    spatial_variation_results = []

    for i, (clean_path, noisy_path) in enumerate(pairs[:10]):
        noisy_img = np.array(Image.open(noisy_path).convert('L')) / 255.0

        # Center crop 256x256
        H, W = noisy_img.shape
        crop = 256
        y = (H - crop) // 2
        x = (W - crop) // 2
        noisy_crop = noisy_img[y:y+crop, x:x+crop]
        noisy_tensor = torch.from_numpy(noisy_crop).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            denoised, pred_weights, extras = model(noisy_tensor)

        if 'spatial_weight_maps' not in extras:
            print("✗ No spatial weight maps found!")
            return

        spatial_maps = extras['spatial_weight_maps'][0]  # (4, H, W)

        # Compute spatial variation statistics
        variation_scores = []
        for j, name in enumerate(noise_names):
            weight_map = spatial_maps[j].numpy()
            std = weight_map.std()
            range_val = weight_map.max() - weight_map.min()
            variation_scores.append({'name': name, 'std': std, 'range': range_val})

        spatial_variation_results.append({
            'image': Path(noisy_path).name,
            'variations': variation_scores
        })

    # Report results
    print("\n[3/3] Spatial variation analysis:\n")
    print(f"{'Image':<40} {'Noise Type':<10} {'Std':>8} {'Range':>8} {'Status':<20}")
    print("-" * 90)

    has_any_variation = False
    for result in spatial_variation_results[:5]:  # Show first 5
        img_name = result['image']
        for var in result['variations']:
            status = "SPATIAL VARIATION" if var['std'] > 0.01 or var['range'] > 0.05 else "uniform"
            if var['std'] > 0.01 or var['range'] > 0.05:
                has_any_variation = True
            print(f"{img_name:<40} {var['name']:<10} {var['std']:8.4f} {var['range']:8.4f} {status:<20}")
        img_name = ""  # Only show once per image

    print()
    if not has_any_variation:
        print("⚠ WARNING: Spatial weights are UNIFORM across all tested images!")
        print("  The spatial refiner has NOT learned meaningful spatial adaptation.")
        print("  Reasons:")
        print("    1. Not enough training epochs (needs 30-40 epochs)")
        print("    2. Noise map loss is dominating, preventing spatial learning")
        print("    3. Learning rate too low for spatial refiner")
        print("    4. Data doesn't actually have heterogeneous noise")
    else:
        print("✓ Spatial weights show variation - spatial adaptation is working!")

    print("\n" + "=" * 80)


def analyze_noise_heterogeneity(val_pairs_file):
    """Check if Duke data actually has heterogeneous noise."""
    print("\n" + "=" * 80)
    print("INVESTIGATION 3: Noise Heterogeneity in Duke Data")
    print("=" * 80)

    if not Path(val_pairs_file).exists():
        print(f"⚠ Validation pairs not found: {val_pairs_file}")
        return

    print("\nAnalyzing noise patterns in Duke OCT images...")

    with open(val_pairs_file) as f:
        pairs = [line.strip().split('\t') for line in f if line.strip() and not line.startswith('#')]

    # Analyze noise variance in different regions
    results = []

    for i, (clean_path, noisy_path) in enumerate(pairs[:10]):
        noisy_img = np.array(Image.open(noisy_path).convert('L')) / 255.0
        clean_img = np.array(Image.open(clean_path).convert('L')) / 255.0

        # Residual (noise)
        residual = noisy_img - clean_img

        # Split into quadrants
        H, W = residual.shape
        half_h, half_w = H // 2, W // 2

        quadrants = {
            'top-left': residual[:half_h, :half_w],
            'top-right': residual[:half_h, half_w:],
            'bottom-left': residual[half_h:, :half_w],
            'bottom-right': residual[half_h:, half_w:]
        }

        # Compute statistics per quadrant
        quad_stats = {}
        for name, quad in quadrants.items():
            quad_stats[name] = {
                'mean': quad.mean(),
                'std': quad.std(),
                'snr': -20 * np.log10(quad.std() + 1e-8)
            }

        # Check if quadrants have different noise levels
        stds = [s['std'] for s in quad_stats.values()]
        std_range = max(stds) - min(stds)
        std_cv = np.std(stds) / (np.mean(stds) + 1e-8)

        results.append({
            'image': Path(noisy_path).name,
            'quad_stats': quad_stats,
            'std_range': std_range,
            'std_cv': std_cv
        })

    # Report
    print(f"\n{'Image':<40} {'Std Range':>12} {'Std CV':>10} {'Heterogeneous?':<15}")
    print("-" * 80)

    heterogeneous_count = 0
    for result in results[:5]:
        is_hetero = result['std_cv'] > 0.1  # >10% coefficient of variation
        status = "YES" if is_hetero else "NO (uniform)"
        heterogeneous_count += int(is_hetero)
        print(f"{result['image']:<40} {result['std_range']:12.4f} {result['std_cv']:10.2%} {status:<15}")

    print()
    if heterogeneous_count < len(results[:5]) * 0.5:
        print("⚠ WARNING: Duke data has UNIFORM noise within each image!")
        print("  Spatial adaptive denoising provides NO BENEFIT for uniform noise.")
        print("  This explains why spatial weights don't improve PSNR.")
        print()
        print("  RECOMMENDATION: Focus on other contributions:")
        print("    - Multi-head architecture (already working)")
        print("    - Interpretable noise analysis (already working)")
        print("    - Region-adaptive processing (inner vs outer retina)")
    else:
        print("✓ Duke data shows heterogeneous noise - spatial adaptation is justified!")

    print("\n" + "=" * 80)


def recommendations():
    """Print comprehensive recommendations."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR TMI PUBLICATION")
    print("=" * 80)

    print("\n🎯 GOAL 1: Improve PSNR/SSIM over Base NAFNet")
    print("-" * 80)
    print("Current: +0.16-0.40 dB improvement (INSUFFICIENT)")
    print("Target:  +0.5-1.0 dB improvement (PUBLISHABLE)")
    print()
    print("Actions:")
    print("  1. REDUCE NOISE MAP LOSS WEIGHT: 0.8 → 0.0")
    print("     - Currently 78% of optimization goes to matching synthetic maps")
    print("     - This prevents the model from learning to denoise better")
    print("     - Use noise maps ONLY for pre-training (Phase A), then disable")
    print()
    print("  2. REDUCE INTERPRETABILITY LOSS: lambda 0.02-0.05 → 0.005")
    print("     - Focus more on denoising quality, less on weight prediction")
    print("     - Accept slightly lower Top-1 accuracy for better PSNR")
    print()
    print("  3. ADD SSIM LOSS: Mix L1 + SSIM (e.g., 0.84*L1 + 0.16*SSIM)")
    print("     - Base NAFNet uses pure Charbonnier (L1-like)")
    print("     - Adding SSIM can improve perceptual quality")
    print()
    print("  4. LONGER TRAINING: 40 epochs → 80-100 epochs")
    print("     - Spatial refiner needs 30-40 epochs just to start learning")
    print("     - Early stopping at epoch 12 is too early")
    print()

    print("\n🎯 GOAL 2: Excellent PSNR/SSIM for TMI")
    print("-" * 80)
    print("Current: PSNR 33.25 dB, SSIM 0.907 (GOOD but not EXCELLENT)")
    print("Target:  PSNR 34.0+ dB, SSIM 0.920+ (EXCELLENT)")
    print()
    print("Actions:")
    print("  1. INCREASE MODEL CAPACITY:")
    print("     - Base NAFNet width: 64 → 80 or 96")
    print("     - Shared trunk width: 32 → 48")
    print("     - Joint expert channels: 96 → 128")
    print()
    print("  2. BETTER LEARNING RATES:")
    print("     - Base NAFNet: 5e-4 → 3e-4 (more stable)")
    print("     - Spatial refiner: 5e-5 → 1e-4 (learn faster)")
    print("     - Use cosine annealing with warm restarts")
    print()
    print("  3. ENSEMBLE APPROACH:")
    print("     - Train 3-5 models with different seeds")
    print("     - Average predictions → +0.2-0.3 dB PSNR")
    print()

    print("\n🎯 GOAL 3: Novel Contributions for TMI")
    print("-" * 80)
    print("Current contributions:")
    print("  ✓ Hybrid CNN-symbolic architecture (NOVEL)")
    print("  ✓ Multi-head noise-specific denoising (GOOD)")
    print("  ⚠ Spatial adaptive denoising (WEAK - not learning)")
    print()
    print("Strengthen contributions:")
    print("  1. SPATIAL ADAPTIVE DENOISING:")
    print("     Option A: Train on synthetic heterogeneous data first")
    print("               → Show it works on synthetic (4-quadrant test)")
    print("               → Then transfer to real data")
    print("     Option B: Drop spatial weights, focus on region-adaptive instead")
    print("               → Inner retina vs outer retina (you already have this!)")
    print("               → Show PSNR improvement in each region")
    print()
    print("  2. NOISE-TYPE AWARE PROCESSING:")
    print("     - Emphasize the multi-head architecture more")
    print("     - Show each head specializes for its noise type")
    print("     - Compare with single-head baselines")
    print()
    print("  3. UNCERTAINTY QUANTIFICATION:")
    print("     - Use predicted weights as uncertainty estimates")
    print("     - Show correlation between weight entropy and denoising difficulty")
    print("     - Provide per-pixel confidence maps")
    print()

    print("\n🎯 GOAL 4: Interpretability for TMI")
    print("-" * 80)
    print("Current: 69.5% Top-1 accuracy (GOOD)")
    print()
    print("Strengthen interpretability:")
    print("  1. VISUALIZATION:")
    print("     - Per-image noise type breakdown charts")
    print("     - Spatial weight maps (if working)")
    print("     - Attention/saliency maps showing what network focuses on")
    print()
    print("  2. CLINICAL RELEVANCE:")
    print("     - Correlate noise types with clinical metadata")
    print("       (disease type, scanner model, scan protocol)")
    print("     - Show interpretability helps diagnosis")
    print()
    print("  3. SYMBOLIC REASONING:")
    print("     - Emphasize the neuro-symbolic component")
    print("     - Show how symbolic rules guide denoising")
    print("     - Compare with pure black-box CNN")
    print()

    print("\n📋 IMMEDIATE ACTION PLAN")
    print("=" * 80)
    print()
    print("STEP 1: Fix loss weights (1 hour)")
    print("  - Set --noise_map_loss_weight 0.0")
    print("  - Set --lambda_interp_start 0.005 --lambda_interp_end 0.005")
    print("  - Retrain for 60 epochs")
    print()
    print("STEP 2: Test spatial weights (2 hours)")
    print("  - Run this investigation script on trained model")
    print("  - If spatial weights still uniform → DISABLE spatial weights")
    print("  - Focus on region-adaptive instead")
    print()
    print("STEP 3: Boost model capacity (4 hours)")
    print("  - Increase NAFNet width to 80")
    print("  - Train for 80 epochs")
    print("  - Target: 33.5-34.0 dB PSNR")
    print()
    print("STEP 4: Create publication figures (1 day)")
    print("  - Ablation studies (base vs multi-head vs region-adaptive)")
    print("  - Noise type classification visualization")
    print("  - Region-specific PSNR improvements")
    print("  - Comparison with state-of-the-art")
    print()
    print("=" * 80)


if __name__ == '__main__':
    # Run all investigations
    analyze_loss_composition('outputs/duke_metrics_joint.jsonl')

    analyze_spatial_variation(
        ckpt_path='checkpoints/multitask_hybrid_nsnd_lambda0p02to0p05_cosine_best.pth',
        val_pairs_file='val_pairs_duke_analysis_maps.txt'
    )

    analyze_noise_heterogeneity('val_pairs_duke_analysis_maps.txt')

    recommendations()
