#!/usr/bin/env python3
"""
Evaluate trained model on real-world OCT datasets (Duke, PKU37).

This script tests generalization to REAL noise (not synthetic) which is
critical for TMI publication - shows the model works on unseen real data.

Datasets:
- Duke SBSDI 2013: Real human OCT with frame-averaged clean references
- PKU37: Real OCT denoising benchmark with 37 subjects

Usage:
    python evaluate_real_world.py --checkpoint path/to/best.pth --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.utils.metrics import compute_psnr, compute_ssim
from nsnd.models.nafnet import NAFNetFullFiLM
from skimage.metrics import structural_similarity as ssim_sk


def compute_masked_metrics(clean, noisy, intensity_threshold=100):
    """
    Compute PSNR and SSIM on meaningful region only.

    The SNA-SKAN paper computes metrics only on pixels where the clean image
    intensity is above a threshold, focusing on retinal structure and ignoring
    dark background noise.

    Args:
        clean: Clean image (numpy array, 0-1 or 0-255)
        noisy: Noisy/denoised image (numpy array, 0-1 or 0-255)
        intensity_threshold: Pixel intensity threshold (default 100 for uint8)

    Returns:
        dict with psnr, ssim, and mask statistics
    """
    # Ensure float64 for computation
    if clean.max() <= 1.0:
        clean = (clean * 255).astype(np.float64)
        noisy = (noisy * 255).astype(np.float64)
    else:
        clean = clean.astype(np.float64)
        noisy = noisy.astype(np.float64)

    # Create mask based on clean image intensity
    mask = clean > intensity_threshold
    n_pixels = np.sum(mask)

    if n_pixels < 1000:
        return {'psnr': None, 'ssim': None, 'n_pixels': n_pixels, 'mask_ratio': n_pixels / clean.size}

    # Extract masked pixels
    clean_masked = clean[mask]
    noisy_masked = noisy[mask]

    # Compute PSNR on masked region
    mse = np.mean((clean_masked / 255.0 - noisy_masked / 255.0) ** 2)
    psnr_val = 10 * np.log10(1.0 / mse) if mse > 0 else 100

    # Compute SSIM: reshape masked pixels to square for spatial computation
    side = int(np.sqrt(n_pixels))
    clean_sq = clean_masked[:side*side].reshape(side, side) / 255.0
    noisy_sq = noisy_masked[:side*side].reshape(side, side) / 255.0
    ssim_val = ssim_sk(clean_sq, noisy_sq, data_range=1.0)

    return {'psnr': psnr_val, 'ssim': ssim_val, 'n_pixels': n_pixels, 'mask_ratio': n_pixels / clean.size}


def load_baseline_model(backbone_path, device):
    """Load baseline NAFNet model for comparison."""
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    )

    if backbone_path and os.path.exists(backbone_path):
        ckpt = torch.load(backbone_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        print(f"Loaded baseline NAFNet from {backbone_path}")
    else:
        print("WARNING: No baseline checkpoint, using random weights")

    base_model = base_model.to(device)
    base_model.eval()
    return base_model


def denoise_with_baseline(base_model, noisy_img, patch_size=64, stride=48, device='cuda'):
    """Denoise image using baseline NAFNet model."""
    h, w = noisy_img.shape
    denoised = np.zeros_like(noisy_img)
    weights = np.zeros_like(noisy_img)

    def _forward_baseline(x):
        with torch.no_grad():
            # NAFNet forward without conditioning
            return base_model(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

    with torch.no_grad():
        for top in range(0, max(1, h - patch_size + 1), stride):
            for left in range(0, max(1, w - patch_size + 1), stride):
                top = min(top, h - patch_size)
                left = min(left, w - patch_size)

                patch = noisy_img[top:top+patch_size, left:left+patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = _forward_baseline(patch_tensor).squeeze().cpu().numpy()

                denoised[top:top+patch_size, left:left+patch_size] += denoised_patch
                weights[top:top+patch_size, left:left+patch_size] += 1.0

    weights = np.maximum(weights, 1e-8)
    denoised = denoised / weights
    denoised = np.clip(denoised, 0, 1)
    return denoised


def load_model(checkpoint_path, device):
    """Load trained MultiTaskDenoiser model."""
    from train_multitask import MultiTaskDenoiser

    model = MultiTaskDenoiser(backbone_ckpt=None, segmenter_ckpt=None)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def denoise_image_patches(model, noisy_img, patch_size=64, stride=48, device='cuda'):
    """
    Denoise large image using overlapping patches.

    Args:
        model: Denoising model
        noisy_img: Noisy image (H, W) in range [0, 1]
        patch_size: Size of patches (default 64x64)
        stride: Stride for patch extraction (overlap = patch_size - stride)
        device: Device to run inference on

    Returns:
        Denoised image (H, W) in range [0, 1]
    """
    h, w = noisy_img.shape

    def _forward_denoise(x):
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        return output

    # Small images: direct processing
    if h <= patch_size and w <= patch_size:
        # Pad to patch_size if needed
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        if pad_h > 0 or pad_w > 0:
            noisy_img = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode='reflect')

        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            denoised_tensor = _forward_denoise(noisy_tensor)
        result = denoised_tensor.squeeze().cpu().numpy()

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            result = result[:h, :w]
        return result

    # Large images: patch-based processing
    denoised = np.zeros_like(noisy_img)
    weights = np.zeros_like(noisy_img)

    with torch.no_grad():
        # Process patches with overlap
        for top in range(0, max(1, h - patch_size + 1), stride):
            for left in range(0, max(1, w - patch_size + 1), stride):
                # Clamp to valid range
                top = min(top, h - patch_size)
                left = min(left, w - patch_size)

                patch = noisy_img[top:top+patch_size, left:left+patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = _forward_denoise(patch_tensor).squeeze().cpu().numpy()

                denoised[top:top+patch_size, left:left+patch_size] += denoised_patch
                weights[top:top+patch_size, left:left+patch_size] += 1.0

        # Handle edges if needed
        if (w - patch_size) % stride != 0 and w > patch_size:
            left = w - patch_size
            for top in range(0, max(1, h - patch_size + 1), stride):
                top = min(top, h - patch_size)
                patch = noisy_img[top:top+patch_size, left:left+patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = _forward_denoise(patch_tensor).squeeze().cpu().numpy()
                denoised[top:top+patch_size, left:left+patch_size] += denoised_patch
                weights[top:top+patch_size, left:left+patch_size] += 1.0

        if (h - patch_size) % stride != 0 and h > patch_size:
            top = h - patch_size
            for left in range(0, max(1, w - patch_size + 1), stride):
                left = min(left, w - patch_size)
                patch = noisy_img[top:top+patch_size, left:left+patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = _forward_denoise(patch_tensor).squeeze().cpu().numpy()
                denoised[top:top+patch_size, left:left+patch_size] += denoised_patch
                weights[top:top+patch_size, left:left+patch_size] += 1.0

    # Average overlapping regions
    weights = np.maximum(weights, 1e-8)
    denoised = denoised / weights
    denoised = np.clip(denoised, 0, 1)

    return denoised


def denoise_image_with_tta(model, noisy_img, patch_size=64, stride=48, device='cuda'):
    """
    Denoise image using Test-Time Augmentation (TTA).

    Applies multiple augmentations (flips) and averages results for more robust denoising.
    Expected improvement: +0.2-0.5 dB PSNR over non-TTA.

    Augmentations:
    - Original
    - Horizontal flip
    - Vertical flip
    - Both flips (180° rotation)

    Args:
        model: Denoising model
        noisy_img: Noisy image (H, W) in range [0, 1]
        patch_size: Size of patches
        stride: Stride for patch extraction
        device: Device to run inference on

    Returns:
        Denoised image (H, W) in range [0, 1]
    """
    outputs = []

    # Original
    out_orig = denoise_image_patches(model, noisy_img, patch_size, stride, device)
    outputs.append(out_orig)

    # Horizontal flip
    noisy_hflip = np.flip(noisy_img, axis=1).copy()
    out_hflip = denoise_image_patches(model, noisy_hflip, patch_size, stride, device)
    out_hflip = np.flip(out_hflip, axis=1)
    outputs.append(out_hflip)

    # Vertical flip
    noisy_vflip = np.flip(noisy_img, axis=0).copy()
    out_vflip = denoise_image_patches(model, noisy_vflip, patch_size, stride, device)
    out_vflip = np.flip(out_vflip, axis=0)
    outputs.append(out_vflip)

    # Both flips (equivalent to 180° rotation)
    noisy_both = np.flip(np.flip(noisy_img, axis=0), axis=1).copy()
    out_both = denoise_image_patches(model, noisy_both, patch_size, stride, device)
    out_both = np.flip(np.flip(out_both, axis=0), axis=1)
    outputs.append(out_both)

    # Average all outputs
    denoised = np.mean(outputs, axis=0)
    denoised = np.clip(denoised, 0, 1)

    return denoised


def evaluate_duke_human(model, device, duke_path, base_model=None, use_tta=False):
    """
    Evaluate on Duke human OCT dataset.

    Args:
        model: Trained model
        device: Device
        duke_path: Path to duke_datasets/organized_test_pairs/human/
        base_model: Optional baseline NAFNet model for comparison
        use_tta: Use Test-Time Augmentation for improved results

    Returns:
        Dict with PSNR/SSIM metrics
    """
    print("\n" + "="*70)
    tta_str = " + TTA" if use_tta else ""
    print(f"EVALUATING ON DUKE HUMAN OCT (Real Noise){tta_str}")
    print("="*70)

    noisy_dir = Path(duke_path) / 'noisy'
    clean_dir = Path(duke_path) / 'clean'

    if not noisy_dir.exists():
        print(f"Duke human dataset not found at {duke_path}")
        return None

    noisy_files = sorted(list(noisy_dir.glob('*.tif')) + list(noisy_dir.glob('*.png')))

    psnr_noisy_list = []
    psnr_denoised_list = []
    psnr_baseline_list = []
    ssim_noisy_list = []
    ssim_denoised_list = []
    ssim_baseline_list = []

    for noisy_path in tqdm(noisy_files, desc="Duke Human"):
        # Find corresponding clean
        clean_path = clean_dir / noisy_path.name
        if not clean_path.exists():
            # Try different extension
            clean_path = clean_dir / (noisy_path.stem + '.tif')
        if not clean_path.exists():
            clean_path = clean_dir / (noisy_path.stem + '.png')
        if not clean_path.exists():
            continue

        # Load images
        noisy_img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean_img = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        # Denoise with our model (with or without TTA)
        if use_tta:
            denoised_img = denoise_image_with_tta(model, noisy_img, device=device)
        else:
            denoised_img = denoise_image_patches(model, noisy_img, device=device)

        # Denoise with baseline (if provided)
        if base_model is not None:
            baseline_img = denoise_with_baseline(base_model, noisy_img, device=device)

        # Compute metrics
        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_img).unsqueeze(0).unsqueeze(0)
        denoised_tensor = torch.from_numpy(denoised_img).unsqueeze(0).unsqueeze(0)

        psnr_noisy = compute_psnr(noisy_tensor, clean_tensor)
        psnr_denoised = compute_psnr(denoised_tensor, clean_tensor)
        ssim_noisy = compute_ssim(noisy_tensor, clean_tensor)
        ssim_denoised = compute_ssim(denoised_tensor, clean_tensor)

        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)
        ssim_noisy_list.append(ssim_noisy)
        ssim_denoised_list.append(ssim_denoised)

        if base_model is not None:
            baseline_tensor = torch.from_numpy(baseline_img).unsqueeze(0).unsqueeze(0)
            psnr_baseline = compute_psnr(baseline_tensor, clean_tensor)
            ssim_baseline = compute_ssim(baseline_tensor, clean_tensor)
            psnr_baseline_list.append(psnr_baseline)
            ssim_baseline_list.append(ssim_baseline)

    if len(psnr_noisy_list) == 0:
        print("No valid pairs found")
        return None

    results = {
        'dataset': 'Duke Human',
        'n_images': len(psnr_noisy_list),
        'psnr_noisy': np.mean(psnr_noisy_list),
        'psnr_denoised': np.mean(psnr_denoised_list),
        'psnr_gain': np.mean(psnr_denoised_list) - np.mean(psnr_noisy_list),
        'ssim_noisy': np.mean(ssim_noisy_list),
        'ssim_denoised': np.mean(ssim_denoised_list),
        'ssim_gain': np.mean(ssim_denoised_list) - np.mean(ssim_noisy_list),
    }

    # Add baseline metrics if available
    if psnr_baseline_list:
        results['psnr_baseline'] = np.mean(psnr_baseline_list)
        results['ssim_baseline'] = np.mean(ssim_baseline_list)
        results['psnr_gain_vs_baseline'] = np.mean(psnr_denoised_list) - np.mean(psnr_baseline_list)
        results['ssim_gain_vs_baseline'] = np.mean(ssim_denoised_list) - np.mean(ssim_baseline_list)

    print(f"\nDuke Human Results ({results['n_images']} images):")
    print(f"  PSNR (noisy):    {results['psnr_noisy']:.2f} dB")
    if 'psnr_baseline' in results:
        print(f"  PSNR (baseline): {results['psnr_baseline']:.2f} dB")
    print(f"  PSNR (ours):     {results['psnr_denoised']:.2f} dB")
    print(f"  PSNR GAIN (vs noisy):    {results['psnr_gain']:+.2f} dB")
    if 'psnr_gain_vs_baseline' in results:
        print(f"  PSNR GAIN (vs baseline): {results['psnr_gain_vs_baseline']:+.2f} dB  <-- KEY METRIC")
    print(f"  SSIM (noisy):    {results['ssim_noisy']:.4f}")
    if 'ssim_baseline' in results:
        print(f"  SSIM (baseline): {results['ssim_baseline']:.4f}")
    print(f"  SSIM (ours):     {results['ssim_denoised']:.4f}")
    print(f"  SSIM GAIN (vs noisy):    {results['ssim_gain']:+.4f}")
    if 'ssim_gain_vs_baseline' in results:
        print(f"  SSIM GAIN (vs baseline): {results['ssim_gain_vs_baseline']:+.4f}")

    return results


def evaluate_pku37(model, device, pku_path, base_model=None, use_tta=False):
    """
    Evaluate on PKU37 OCT denoising benchmark.

    PKU37 has 37 clean images and multiple noisy frames per clean.
    We evaluate by matching noisy images to their corresponding clean reference.

    Args:
        model: Trained model
        device: Device
        pku_path: Path to PKU37_OCT_Denoising/PKU37_OCT_Denoising/
        base_model: Optional baseline NAFNet model for comparison
        use_tta: Use Test-Time Augmentation for improved results

    Returns:
        Dict with PSNR/SSIM metrics
    """
    print("\n" + "="*70)
    tta_str = " + TTA" if use_tta else ""
    print(f"EVALUATING ON PKU37 OCT (Real Noise){tta_str}")
    print("="*70)

    noisy_dir = Path(pku_path) / 'noisy'
    clean_dir = Path(pku_path) / 'clean'

    if not noisy_dir.exists():
        print(f"PKU37 dataset not found at {pku_path}")
        return None

    # PKU37 naming: clean images are 000001.tif to 000037.tif (roughly)
    # Noisy images are 000101.tif, 000102.tif, etc. (multiple per clean)
    clean_files = sorted(list(clean_dir.glob('*.tif')))

    if len(clean_files) == 0:
        print("No clean reference images found")
        return None

    psnr_noisy_list = []
    psnr_denoised_list = []
    psnr_baseline_list = []
    ssim_noisy_list = []
    ssim_denoised_list = []
    ssim_baseline_list = []

    # Map noisy images to clean references
    # PKU37 convention: noisy 000101-000150 -> clean 000001, noisy 000201-000250 -> clean 000002, etc.
    for clean_path in tqdm(clean_files, desc="PKU37"):
        clean_img = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        # Extract clean number (e.g., 000001 -> 1)
        try:
            clean_num = int(clean_path.stem)
        except ValueError:
            continue

        # Find corresponding noisy images (e.g., clean 1 -> noisy 101-150)
        noisy_start = clean_num * 100 + 1
        noisy_end = clean_num * 100 + 50

        noisy_count = 0
        for noisy_num in range(noisy_start, noisy_end + 1):
            noisy_path = noisy_dir / f'{noisy_num:06d}.tif'
            if not noisy_path.exists():
                continue

            noisy_img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0

            # Skip if sizes don't match
            if noisy_img.shape != clean_img.shape:
                continue

            # Denoise with our model (with or without TTA)
            if use_tta:
                denoised_img = denoise_image_with_tta(model, noisy_img, device=device)
            else:
                denoised_img = denoise_image_patches(model, noisy_img, device=device)

            # Denoise with baseline (if provided)
            if base_model is not None:
                baseline_img = denoise_with_baseline(base_model, noisy_img, device=device)

            # Compute metrics
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0)
            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0).unsqueeze(0)
            denoised_tensor = torch.from_numpy(denoised_img).unsqueeze(0).unsqueeze(0)

            psnr_noisy = compute_psnr(noisy_tensor, clean_tensor)
            psnr_denoised = compute_psnr(denoised_tensor, clean_tensor)
            ssim_noisy = compute_ssim(noisy_tensor, clean_tensor)
            ssim_denoised = compute_ssim(denoised_tensor, clean_tensor)

            psnr_noisy_list.append(psnr_noisy)
            psnr_denoised_list.append(psnr_denoised)
            ssim_noisy_list.append(ssim_noisy)
            ssim_denoised_list.append(ssim_denoised)

            if base_model is not None:
                baseline_tensor = torch.from_numpy(baseline_img).unsqueeze(0).unsqueeze(0)
                psnr_baseline = compute_psnr(baseline_tensor, clean_tensor)
                ssim_baseline = compute_ssim(baseline_tensor, clean_tensor)
                psnr_baseline_list.append(psnr_baseline)
                ssim_baseline_list.append(ssim_baseline)

            noisy_count += 1

            # Limit to first 5 noisy frames per clean to save time
            if noisy_count >= 5:
                break

    if len(psnr_noisy_list) == 0:
        print("No valid pairs found")
        return None

    results = {
        'dataset': 'PKU37',
        'n_images': len(psnr_noisy_list),
        'psnr_noisy': np.mean(psnr_noisy_list),
        'psnr_denoised': np.mean(psnr_denoised_list),
        'psnr_gain': np.mean(psnr_denoised_list) - np.mean(psnr_noisy_list),
        'ssim_noisy': np.mean(ssim_noisy_list),
        'ssim_denoised': np.mean(ssim_denoised_list),
        'ssim_gain': np.mean(ssim_denoised_list) - np.mean(ssim_noisy_list),
    }

    # Add baseline metrics if available
    if psnr_baseline_list:
        results['psnr_baseline'] = np.mean(psnr_baseline_list)
        results['ssim_baseline'] = np.mean(ssim_baseline_list)
        results['psnr_gain_vs_baseline'] = np.mean(psnr_denoised_list) - np.mean(psnr_baseline_list)
        results['ssim_gain_vs_baseline'] = np.mean(ssim_denoised_list) - np.mean(ssim_baseline_list)

    print(f"\nPKU37 Results ({results['n_images']} noisy frames):")
    print(f"  PSNR (noisy):    {results['psnr_noisy']:.2f} dB")
    if 'psnr_baseline' in results:
        print(f"  PSNR (baseline): {results['psnr_baseline']:.2f} dB")
    print(f"  PSNR (ours):     {results['psnr_denoised']:.2f} dB")
    print(f"  PSNR GAIN (vs noisy):    {results['psnr_gain']:+.2f} dB")
    if 'psnr_gain_vs_baseline' in results:
        print(f"  PSNR GAIN (vs baseline): {results['psnr_gain_vs_baseline']:+.2f} dB  <-- KEY METRIC")
    print(f"  SSIM (noisy):    {results['ssim_noisy']:.4f}")
    if 'ssim_baseline' in results:
        print(f"  SSIM (baseline): {results['ssim_baseline']:.4f}")
    print(f"  SSIM (ours):     {results['ssim_denoised']:.4f}")
    print(f"  SSIM GAIN (vs noisy):    {results['ssim_gain']:+.4f}")
    if 'ssim_gain_vs_baseline' in results:
        print(f"  SSIM GAIN (vs baseline): {results['ssim_gain_vs_baseline']:+.4f}")

    return results


def evaluate_duke_sota(model, device, duke_path, dataset_name='Duke17', base_model=None,
                       use_tta=False, use_masked_ssim=True, intensity_threshold=100):
    """
    Evaluate on Duke17 or Duke28 SOTA datasets with SNA-SKAN methodology.

    These datasets are used in SNA-SKAN paper for benchmarking OCT denoising.
    Uses masked SSIM for fair comparison with paper results.

    Args:
        model: Trained denoising model
        device: Device (cuda or cpu)
        duke_path: Path to Sparsity_SDOCT_DATASET_2012 (Duke17) or Duke28_2015 (Duke28)
        dataset_name: 'Duke17' or 'Duke28' for logging
        base_model: Baseline NAFNet for comparison
        use_tta: Use test-time augmentation
        use_masked_ssim: Use masked SSIM (SNA-SKAN methodology)
        intensity_threshold: Threshold for masked metrics (default 100)

    Returns:
        dict with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {dataset_name} (SNA-SKAN Benchmark)")
    print(f"{'='*70}")

    if not os.path.exists(duke_path):
        print(f"Dataset not found: {duke_path}")
        return None

    # Find subject folders
    subjects = sorted([d for d in os.listdir(duke_path)
                      if os.path.isdir(os.path.join(duke_path, d))])

    if len(subjects) == 0:
        print(f"No subjects found in {duke_path}")
        return None

    print(f"Found {len(subjects)} subjects")

    # Metrics lists
    psnr_noisy_list, psnr_denoised_list, psnr_baseline_list = [], [], []
    ssim_noisy_list, ssim_denoised_list, ssim_baseline_list = [], [], []
    psnr_noisy_masked_list, psnr_denoised_masked_list, psnr_baseline_masked_list = [], [], []
    ssim_noisy_masked_list, ssim_denoised_masked_list, ssim_baseline_masked_list = [], [], []

    for subject in tqdm(subjects, desc=dataset_name):
        subject_dir = os.path.join(duke_path, subject)
        files = os.listdir(subject_dir)

        # Find clean (Averaged) and noisy (Raw) images
        clean_files = [f for f in files if 'Averaged' in f]
        noisy_files = [f for f in files if 'Raw' in f]

        if not clean_files or not noisy_files:
            continue

        clean_path = os.path.join(subject_dir, clean_files[0])
        noisy_path = os.path.join(subject_dir, noisy_files[0])

        clean_img = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0
        noisy_img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0

        if clean_img.shape != noisy_img.shape:
            continue

        # Denoise with our model
        if use_tta:
            denoised_img = denoise_image_with_tta(model, noisy_img, device=device)
        else:
            denoised_img = denoise_image_patches(model, noisy_img, device=device)

        # Denoise with baseline
        baseline_img = None
        if base_model is not None:
            baseline_img = denoise_with_baseline(base_model, noisy_img, device=device)

        # Standard metrics
        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_img).unsqueeze(0).unsqueeze(0)
        denoised_tensor = torch.from_numpy(denoised_img).unsqueeze(0).unsqueeze(0)

        psnr_noisy_list.append(compute_psnr(noisy_tensor, clean_tensor))
        psnr_denoised_list.append(compute_psnr(denoised_tensor, clean_tensor))
        ssim_noisy_list.append(compute_ssim(noisy_tensor, clean_tensor))
        ssim_denoised_list.append(compute_ssim(denoised_tensor, clean_tensor))

        if baseline_img is not None:
            baseline_tensor = torch.from_numpy(baseline_img).unsqueeze(0).unsqueeze(0)
            psnr_baseline_list.append(compute_psnr(baseline_tensor, clean_tensor))
            ssim_baseline_list.append(compute_ssim(baseline_tensor, clean_tensor))

        # Masked metrics (SNA-SKAN methodology)
        if use_masked_ssim:
            noisy_masked = compute_masked_metrics(clean_img, noisy_img, intensity_threshold)
            denoised_masked = compute_masked_metrics(clean_img, denoised_img, intensity_threshold)

            if noisy_masked['psnr'] is not None:
                psnr_noisy_masked_list.append(noisy_masked['psnr'])
                ssim_noisy_masked_list.append(noisy_masked['ssim'])
                psnr_denoised_masked_list.append(denoised_masked['psnr'])
                ssim_denoised_masked_list.append(denoised_masked['ssim'])

            if baseline_img is not None:
                baseline_masked = compute_masked_metrics(clean_img, baseline_img, intensity_threshold)
                if baseline_masked['psnr'] is not None:
                    psnr_baseline_masked_list.append(baseline_masked['psnr'])
                    ssim_baseline_masked_list.append(baseline_masked['ssim'])

    if len(psnr_noisy_list) == 0:
        print("No valid image pairs found")
        return None

    # Build results
    results = {
        'dataset': dataset_name,
        'n_images': len(psnr_noisy_list),
        'psnr_noisy': np.mean(psnr_noisy_list),
        'psnr_denoised': np.mean(psnr_denoised_list),
        'ssim_noisy': np.mean(ssim_noisy_list),
        'ssim_denoised': np.mean(ssim_denoised_list),
    }

    if psnr_baseline_list:
        results['psnr_baseline'] = np.mean(psnr_baseline_list)
        results['ssim_baseline'] = np.mean(ssim_baseline_list)

    # Add masked metrics
    if use_masked_ssim and psnr_noisy_masked_list:
        results['psnr_noisy_masked'] = np.mean(psnr_noisy_masked_list)
        results['psnr_denoised_masked'] = np.mean(psnr_denoised_masked_list)
        results['ssim_noisy_masked'] = np.mean(ssim_noisy_masked_list)
        results['ssim_denoised_masked'] = np.mean(ssim_denoised_masked_list)

        if psnr_baseline_masked_list:
            results['psnr_baseline_masked'] = np.mean(psnr_baseline_masked_list)
            results['ssim_baseline_masked'] = np.mean(ssim_baseline_masked_list)

    # Print results
    print(f"\n{dataset_name} Results ({results['n_images']} images):")
    print(f"\nStandard Metrics:")
    print(f"  PSNR (noisy):    {results['psnr_noisy']:.2f} dB")
    if 'psnr_baseline' in results:
        print(f"  PSNR (baseline): {results['psnr_baseline']:.2f} dB")
    print(f"  PSNR (ours):     {results['psnr_denoised']:.2f} dB")
    print(f"  SSIM (noisy):    {results['ssim_noisy']:.4f}")
    if 'ssim_baseline' in results:
        print(f"  SSIM (baseline): {results['ssim_baseline']:.4f}")
    print(f"  SSIM (ours):     {results['ssim_denoised']:.4f}")

    if use_masked_ssim and 'psnr_noisy_masked' in results:
        print(f"\nMasked Metrics (SNA-SKAN methodology, threshold={intensity_threshold}):")
        print(f"  PSNR (noisy):    {results['psnr_noisy_masked']:.2f} dB")
        if 'psnr_baseline_masked' in results:
            print(f"  PSNR (baseline): {results['psnr_baseline_masked']:.2f} dB")
        print(f"  PSNR (ours):     {results['psnr_denoised_masked']:.2f} dB")
        print(f"  SSIM (noisy):    {results['ssim_noisy_masked']:.3f}  <-- Compare with SNA-SKAN 0.311")
        if 'ssim_baseline_masked' in results:
            print(f"  SSIM (baseline): {results['ssim_baseline_masked']:.3f}")
        print(f"  SSIM (ours):     {results['ssim_denoised_masked']:.3f}  <-- KEY METRIC FOR TMI")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate on real-world OCT datasets')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--baseline_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth',
                       help='Path to baseline NAFNet checkpoint for comparison')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--duke_path', default='duke_datasets/organized_test_pairs/human',
                       help='Path to Duke human dataset')
    parser.add_argument('--pku_path', default='pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising',
                       help='Path to PKU37 dataset')
    parser.add_argument('--duke17_path', default='duke_sota_datasets/Sparsity_SDOCT_DATASET_2012',
                       help='Path to Duke17 SOTA benchmark')
    parser.add_argument('--duke28_path', default='duke_sota_datasets/Duke28_2015',
                       help='Path to Duke28 SOTA benchmark')
    parser.add_argument('--output_file', default=None, help='Output JSON file for results')
    parser.add_argument('--no_baseline', action='store_true', help='Skip baseline comparison')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use Test-Time Augmentation (4x flip ensemble) for improved results')
    parser.add_argument('--use_masked_ssim', action='store_true',
                       help='Use masked SSIM (SNA-SKAN methodology) for Duke17/Duke28')
    parser.add_argument('--intensity_threshold', type=int, default=100,
                       help='Intensity threshold for masked metrics (default 100)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("="*70)
    tta_str = " + TTA" if args.use_tta else ""
    print(f"REAL-WORLD OCT EVALUATION (with baseline comparison){tta_str}")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Baseline: {args.baseline_ckpt}")
    print(f"Device: {args.device}")
    print(f"TTA: {'Enabled (4x flip ensemble)' if args.use_tta else 'Disabled'}")
    print("="*70)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load baseline model for comparison
    base_model = None
    if not args.no_baseline and os.path.exists(args.baseline_ckpt):
        base_model = load_baseline_model(args.baseline_ckpt, args.device)
    elif not args.no_baseline:
        print(f"WARNING: Baseline checkpoint not found at {args.baseline_ckpt}")
        print("Running without baseline comparison. Use --no_baseline to suppress this warning.")

    all_results = {}

    # Evaluate on Duke Human (original)
    duke_results = evaluate_duke_human(model, args.device, args.duke_path, base_model, use_tta=args.use_tta)
    if duke_results:
        all_results['duke_human'] = duke_results

    # Evaluate on PKU37
    pku_results = evaluate_pku37(model, args.device, args.pku_path, base_model, use_tta=args.use_tta)
    if pku_results:
        all_results['pku37'] = pku_results

    # Evaluate on Duke17 SOTA benchmark (SNA-SKAN comparison)
    if os.path.exists(args.duke17_path):
        duke17_results = evaluate_duke_sota(
            model, args.device, args.duke17_path, dataset_name='Duke17',
            base_model=base_model, use_tta=args.use_tta,
            use_masked_ssim=args.use_masked_ssim,
            intensity_threshold=args.intensity_threshold
        )
        if duke17_results:
            all_results['duke17'] = duke17_results

    # Evaluate on Duke28 SOTA benchmark (SNA-SKAN comparison)
    if os.path.exists(args.duke28_path):
        duke28_results = evaluate_duke_sota(
            model, args.device, args.duke28_path, dataset_name='Duke28',
            base_model=base_model, use_tta=args.use_tta,
            use_masked_ssim=args.use_masked_ssim,
            intensity_threshold=args.intensity_threshold
        )
        if duke28_results:
            all_results['duke28'] = duke28_results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: REAL-WORLD GENERALIZATION")
    print("="*70)

    if 'duke_human' in all_results:
        r = all_results['duke_human']
        print(f"\nDuke Human (n={r['n_images']}):")
        print(f"  PSNR: {r['psnr_noisy']:.2f} -> {r['psnr_denoised']:.2f} dB (vs noisy: {r['psnr_gain']:+.2f} dB)")
        if 'psnr_baseline' in r:
            print(f"  PSNR: baseline {r['psnr_baseline']:.2f} -> ours {r['psnr_denoised']:.2f} dB (vs baseline: {r['psnr_gain_vs_baseline']:+.2f} dB)")
        print(f"  SSIM: {r['ssim_noisy']:.4f} -> {r['ssim_denoised']:.4f} (vs noisy: {r['ssim_gain']:+.4f})")
        if 'ssim_baseline' in r:
            print(f"  SSIM: baseline {r['ssim_baseline']:.4f} -> ours {r['ssim_denoised']:.4f} (vs baseline: {r['ssim_gain_vs_baseline']:+.4f})")

    if 'pku37' in all_results:
        r = all_results['pku37']
        print(f"\nPKU37 (n={r['n_images']}):")
        print(f"  PSNR: {r['psnr_noisy']:.2f} -> {r['psnr_denoised']:.2f} dB (vs noisy: {r['psnr_gain']:+.2f} dB)")
        if 'psnr_baseline' in r:
            print(f"  PSNR: baseline {r['psnr_baseline']:.2f} -> ours {r['psnr_denoised']:.2f} dB (vs baseline: {r['psnr_gain_vs_baseline']:+.2f} dB)")
        print(f"  SSIM: {r['ssim_noisy']:.4f} -> {r['ssim_denoised']:.4f} (vs noisy: {r['ssim_gain']:+.4f})")
        if 'ssim_baseline' in r:
            print(f"  SSIM: baseline {r['ssim_baseline']:.4f} -> ours {r['ssim_denoised']:.4f} (vs baseline: {r['ssim_gain_vs_baseline']:+.4f})")

    # Duke17/Duke28 SOTA benchmarks (SNA-SKAN comparison)
    for key in ['duke17', 'duke28']:
        if key in all_results:
            r = all_results[key]
            print(f"\n{r['dataset']} (n={r['n_images']}, SNA-SKAN benchmark):")
            print(f"  PSNR: {r['psnr_noisy']:.2f} -> {r['psnr_denoised']:.2f} dB")
            if 'ssim_noisy_masked' in r:
                print(f"  Masked SSIM (SNA-SKAN method):")
                print(f"    Noisy:    {r['ssim_noisy_masked']:.3f} (paper: 0.311)")
                print(f"    Ours:     {r['ssim_denoised_masked']:.3f}")
                if 'ssim_baseline_masked' in r:
                    print(f"    Baseline: {r['ssim_baseline_masked']:.3f}")

    print("\n" + "="*70)
    if base_model is not None:
        print("KEY METRIC: 'vs baseline' shows improvement over NAFNet backbone")
        print("This is comparable to training metrics (+1.5 dB expected)")
    print("These results on REAL noise demonstrate generalization beyond")
    print("synthetic training data - critical for clinical applicability.")
    if any(k in all_results for k in ['duke17', 'duke28']):
        print("\nSNA-SKAN paper reports: PSNR=16.273dB, SSIM=0.311 (masked) for noisy Duke17")
    print("="*70)

    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    return all_results


if __name__ == '__main__':
    main()
