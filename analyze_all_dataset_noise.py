#!/usr/bin/env python3
"""
Comprehensive Noise Analysis for All OCT Datasets

Analyzes noise characteristics from Duke17, Duke28, PKU37 and other datasets
to create calibrated synthetic noise that matches real OCT noise as closely as possible.

Computes:
1. Noise composition weights (speckle, banding, gaussian, shot)
2. Speckle parameters (shape k, coefficient of variation, spatial correlation)
3. Banding parameters (frequency, amplitude)
4. Gaussian parameters (sigma, vertical/horizontal correlation)
5. Shot noise parameters (gain)
6. Global statistics (PSNR, SSIM baseline, noise std)
7. Spatial correlation structure (vertical and horizontal)

Usage:
    python analyze_all_dataset_noise.py
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from scipy import ndimage, signal, stats
from scipy.optimize import minimize_scalar
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_psnr_ssim(clean, noisy, threshold=100/255):
    """Compute PSNR and SSIM (both full and masked)."""
    clean = clean.astype(np.float64)
    noisy = noisy.astype(np.float64)

    # Full image metrics
    mse_full = np.mean((clean - noisy) ** 2)
    psnr_full = 10 * np.log10(1.0 / mse_full) if mse_full > 0 else 50
    ssim_full = ssim(clean, noisy, data_range=1.0)

    # Masked metrics (retinal tissue only, matching SOTA papers)
    mask = clean > threshold
    n_pixels = np.sum(mask)

    if n_pixels > 1000:
        clean_masked = clean[mask]
        noisy_masked = noisy[mask]
        mse_masked = np.mean((clean_masked - noisy_masked) ** 2)
        psnr_masked = 10 * np.log10(1.0 / mse_masked) if mse_masked > 0 else 50

        # SSIM on reshaped square
        side = int(np.sqrt(n_pixels))
        if side >= 7:
            clean_sq = clean_masked[:side*side].reshape(side, side)
            noisy_sq = noisy_masked[:side*side].reshape(side, side)
            ssim_masked = ssim(clean_sq, noisy_sq, data_range=1.0)
        else:
            ssim_masked = ssim_full
    else:
        psnr_masked = psnr_full
        ssim_masked = ssim_full

    return {
        'psnr_full': psnr_full,
        'ssim_full': ssim_full,
        'psnr_masked': psnr_masked,
        'ssim_masked': ssim_masked,
        'mask_ratio': n_pixels / clean.size
    }


def compute_noise_residual(clean, noisy):
    """Compute noise residual and basic statistics."""
    residual = noisy - clean

    return {
        'mean': float(np.mean(residual)),
        'std': float(np.std(residual)),
        'min': float(np.min(residual)),
        'max': float(np.max(residual)),
        'skewness': float(stats.skew(residual.flatten())),
        'kurtosis': float(stats.kurtosis(residual.flatten())),
    }


def compute_spatial_correlation(residual, max_lag=20):
    """Compute spatial autocorrelation in vertical and horizontal directions."""
    H, W = residual.shape

    # Normalize residual
    residual_norm = (residual - np.mean(residual)) / (np.std(residual) + 1e-8)

    # Vertical correlation (along A-scans)
    v_corr = []
    for lag in range(max_lag):
        if lag == 0:
            v_corr.append(1.0)
        else:
            corr = np.mean(residual_norm[:-lag, :] * residual_norm[lag:, :])
            v_corr.append(float(corr))

    # Horizontal correlation (across A-scans)
    h_corr = []
    for lag in range(max_lag):
        if lag == 0:
            h_corr.append(1.0)
        else:
            corr = np.mean(residual_norm[:, :-lag] * residual_norm[:, lag:])
            h_corr.append(float(corr))

    # Compute correlation length (lag where correlation drops to 1/e)
    def find_correlation_length(corr_vals):
        target = 1.0 / np.e
        for i, c in enumerate(corr_vals):
            if c < target:
                return i
        return len(corr_vals)

    return {
        'vertical': v_corr,
        'horizontal': h_corr,
        'v_corr_length': find_correlation_length(v_corr),
        'h_corr_length': find_correlation_length(h_corr),
        'v_corr_at_1': v_corr[1] if len(v_corr) > 1 else 0,
        'h_corr_at_1': h_corr[1] if len(h_corr) > 1 else 0,
    }


def analyze_speckle_characteristics(clean, noisy):
    """Analyze speckle noise characteristics (multiplicative noise)."""
    # Avoid division by zero
    clean_safe = np.maximum(clean, 0.01)

    # Speckle is multiplicative: noisy = clean * speckle
    # So speckle = noisy / clean
    speckle_ratio = noisy / clean_safe

    # Only analyze in regions with sufficient signal
    mask = clean > 0.1
    if np.sum(mask) < 1000:
        mask = clean > 0.05

    speckle_valid = speckle_ratio[mask]

    # Speckle typically follows Gamma distribution
    # Gamma(k, theta) where mean = k*theta, var = k*theta^2
    mean_s = np.mean(speckle_valid)
    var_s = np.var(speckle_valid)

    # Estimate shape parameter k
    if var_s > 0:
        k = (mean_s ** 2) / var_s
        theta = var_s / mean_s
    else:
        k = 1.0
        theta = 1.0

    # Coefficient of variation
    cv = np.std(speckle_valid) / (np.mean(speckle_valid) + 1e-8)

    # Spatial correlation of speckle
    speckle_2d = np.ones_like(clean) * mean_s
    speckle_2d[mask] = speckle_valid.reshape(-1)[:np.sum(mask)]
    speckle_residual = speckle_2d - mean_s

    # Simple correlation at lag 1
    v_corr = np.mean(speckle_residual[:-1, :] * speckle_residual[1:, :]) / (var_s + 1e-8)
    h_corr = np.mean(speckle_residual[:, :-1] * speckle_residual[:, 1:]) / (var_s + 1e-8)

    return {
        'mean': float(mean_s),
        'std': float(np.std(speckle_valid)),
        'cv': float(cv),
        'gamma_k': float(k),
        'gamma_theta': float(theta),
        'v_correlation': float(v_corr),
        'h_correlation': float(h_corr),
    }


def analyze_banding_artifacts(residual):
    """Analyze horizontal banding artifacts (electronic interference)."""
    H, W = residual.shape

    # Compute row means (banding appears as horizontal stripes)
    row_means = np.mean(residual, axis=1)

    # Compute power spectrum of row means
    fft_rows = np.fft.fft(row_means)
    power = np.abs(fft_rows) ** 2
    freqs = np.fft.fftfreq(H)

    # Find dominant low-frequency components (banding)
    # Focus on low frequencies (0.01 to 0.1 cycles/pixel)
    low_freq_mask = (np.abs(freqs) > 0.005) & (np.abs(freqs) < 0.15)
    low_freq_power = power[low_freq_mask]
    low_freq_freqs = np.abs(freqs[low_freq_mask])

    if len(low_freq_power) > 0:
        # Dominant banding frequency
        max_idx = np.argmax(low_freq_power)
        dominant_freq = low_freq_freqs[max_idx]
        dominant_power = low_freq_power[max_idx]

        # Banding amplitude
        banding_amplitude = np.std(row_means)

        # Ratio of banding power to total
        banding_ratio = np.sum(low_freq_power) / (np.sum(power) + 1e-8)
    else:
        dominant_freq = 0.0
        dominant_power = 0.0
        banding_amplitude = 0.0
        banding_ratio = 0.0

    return {
        'amplitude': float(banding_amplitude),
        'dominant_freq': float(dominant_freq),
        'power_ratio': float(banding_ratio),
        'row_std': float(np.std(row_means)),
    }


def analyze_gaussian_component(residual, speckle_contribution):
    """Analyze additive Gaussian noise component."""
    # Approximate Gaussian component by subtracting estimated speckle
    # This is an approximation since noise components are mixed

    # Use high-pass filtered residual to isolate Gaussian
    from scipy.ndimage import gaussian_filter
    low_freq = gaussian_filter(residual, sigma=3)
    high_freq = residual - low_freq

    gaussian_sigma = np.std(high_freq)

    # Vertical correlation in high-frequency component
    hf_norm = (high_freq - np.mean(high_freq)) / (gaussian_sigma + 1e-8)
    v_corr = np.mean(hf_norm[:-1, :] * hf_norm[1:, :])
    h_corr = np.mean(hf_norm[:, :-1] * hf_norm[:, 1:])

    return {
        'sigma': float(gaussian_sigma),
        'v_correlation': float(v_corr),
        'h_correlation': float(h_corr),
    }


def analyze_shot_noise(clean, noisy):
    """Analyze signal-dependent (Poisson/shot) noise."""
    # Shot noise variance is proportional to signal intensity
    # Var(noise) = gain * signal

    # Bin by intensity and compute variance
    n_bins = 10
    intensities = []
    variances = []

    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = (clean >= low) & (clean < high)

        if np.sum(mask) > 100:
            noise = noisy[mask] - clean[mask]
            intensities.append((low + high) / 2)
            variances.append(np.var(noise))

    # Fit linear model: variance = gain * intensity + offset
    if len(intensities) >= 3:
        intensities = np.array(intensities)
        variances = np.array(variances)

        # Simple linear regression
        A = np.vstack([intensities, np.ones_like(intensities)]).T
        result = np.linalg.lstsq(A, variances, rcond=None)
        gain, offset = result[0]

        # R-squared
        predicted = gain * intensities + offset
        ss_res = np.sum((variances - predicted) ** 2)
        ss_tot = np.sum((variances - np.mean(variances)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
    else:
        gain = 0.0
        offset = 0.0
        r_squared = 0.0

    return {
        'gain': float(max(0, gain)),
        'offset': float(offset),
        'r_squared': float(r_squared),
        'is_signal_dependent': r_squared > 0.5 and gain > 0,
    }


def estimate_noise_composition(clean, noisy, speckle_stats, banding_stats, gaussian_stats, shot_stats):
    """Estimate relative weights of different noise components."""
    residual = noisy - clean
    total_var = np.var(residual)

    if total_var < 1e-10:
        return {'speckle': 0.25, 'banding': 0.25, 'gaussian': 0.25, 'shot': 0.25}

    # Estimate variance contributions
    # Speckle: multiplicative noise variance ~ (cv^2) * E[signal^2]
    speckle_var = (speckle_stats['cv'] ** 2) * np.mean(clean ** 2)

    # Banding: variance from row means
    banding_var = banding_stats['amplitude'] ** 2

    # Gaussian: from high-frequency component
    gaussian_var = gaussian_stats['sigma'] ** 2

    # Shot: signal-dependent component
    shot_var = shot_stats['gain'] * np.mean(clean) if shot_stats['gain'] > 0 else 0

    # Normalize to get weights
    total_estimated = speckle_var + banding_var + gaussian_var + shot_var + 1e-10

    weights = {
        'speckle': float(speckle_var / total_estimated),
        'banding': float(banding_var / total_estimated),
        'gaussian': float(gaussian_var / total_estimated),
        'shot': float(shot_var / total_estimated),
    }

    # Normalize to sum to 1
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    return weights


def analyze_image_pair(clean, noisy):
    """Comprehensive analysis of a clean-noisy image pair."""
    # Normalize to 0-1
    if clean.max() > 1:
        clean = clean.astype(np.float64) / 255.0
        noisy = noisy.astype(np.float64) / 255.0
    else:
        clean = clean.astype(np.float64)
        noisy = noisy.astype(np.float64)

    residual = noisy - clean

    # Basic metrics
    metrics = compute_psnr_ssim(clean, noisy)
    noise_stats = compute_noise_residual(clean, noisy)
    spatial_corr = compute_spatial_correlation(residual)

    # Component analysis
    speckle_stats = analyze_speckle_characteristics(clean, noisy)
    banding_stats = analyze_banding_artifacts(residual)
    gaussian_stats = analyze_gaussian_component(residual, speckle_stats)
    shot_stats = analyze_shot_noise(clean, noisy)

    # Composition weights
    composition = estimate_noise_composition(
        clean, noisy, speckle_stats, banding_stats, gaussian_stats, shot_stats
    )

    return {
        'metrics': metrics,
        'noise_stats': noise_stats,
        'spatial_correlation': spatial_corr,
        'speckle': speckle_stats,
        'banding': banding_stats,
        'gaussian': gaussian_stats,
        'shot': shot_stats,
        'composition': composition,
    }


def analyze_duke17(dataset_dir):
    """Analyze Duke17 (Sparsity_SDOCT_DATASET_2012) dataset."""
    print("\n" + "="*70)
    print("ANALYZING DUKE17 DATASET")
    print("="*70)

    subjects = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])

    all_results = []

    for subject in tqdm(subjects, desc="Duke17"):
        subject_dir = os.path.join(dataset_dir, subject)
        files = os.listdir(subject_dir)

        clean_files = [f for f in files if 'Averaged' in f]
        noisy_files = [f for f in files if 'Raw' in f]

        if not clean_files or not noisy_files:
            continue

        clean = np.array(Image.open(os.path.join(subject_dir, clean_files[0])))
        noisy = np.array(Image.open(os.path.join(subject_dir, noisy_files[0])))

        result = analyze_image_pair(clean, noisy)
        result['subject'] = subject
        all_results.append(result)

    return aggregate_results(all_results, 'Duke17')


def analyze_duke28(dataset_dir):
    """Analyze Duke28 (Duke2013_SBSDI) dataset."""
    print("\n" + "="*70)
    print("ANALYZING DUKE28 DATASET")
    print("="*70)

    # Find subdirectories with image pairs
    all_results = []

    for root, dirs, files in os.walk(dataset_dir):
        # Look for Raw and Averaged pairs
        raw_files = [f for f in files if 'Raw' in f and f.endswith(('.tif', '.png', '.bmp'))]
        avg_files = [f for f in files if 'Averaged' in f and f.endswith(('.tif', '.png', '.bmp'))]

        for raw_f in raw_files[:5]:  # Limit for speed
            # Try to find matching averaged file
            base = raw_f.replace('Raw', '').replace('raw', '')
            avg_matches = [f for f in avg_files if base in f.replace('Averaged', '').replace('averaged', '')]

            if avg_matches:
                try:
                    noisy = np.array(Image.open(os.path.join(root, raw_f)))
                    clean = np.array(Image.open(os.path.join(root, avg_matches[0])))

                    if noisy.shape == clean.shape:
                        result = analyze_image_pair(clean, noisy)
                        result['file'] = raw_f
                        all_results.append(result)
                except Exception as e:
                    continue

    if not all_results:
        print("  No valid pairs found in Duke28")
        return None

    return aggregate_results(all_results, 'Duke28')


def analyze_pku37(dataset_dir):
    """Analyze PKU37 dataset."""
    print("\n" + "="*70)
    print("ANALYZING PKU37 DATASET")
    print("="*70)

    # Find the image pairs
    pku_dir = os.path.join(dataset_dir, 'PKU37_OCT_Denoising')
    if not os.path.exists(pku_dir):
        pku_dir = dataset_dir

    all_results = []

    # Look for paired directories (noisy/clean or similar)
    for root, dirs, files in os.walk(pku_dir):
        # Check for noisy/clean naming patterns
        noisy_files = [f for f in files if 'noisy' in f.lower() or 'raw' in f.lower()]
        clean_files = [f for f in files if 'clean' in f.lower() or 'gt' in f.lower() or 'averaged' in f.lower()]

        if noisy_files and clean_files:
            for nf in noisy_files[:10]:  # Limit for speed
                # Find matching clean
                base = nf.replace('noisy', '').replace('Noisy', '').replace('raw', '').replace('Raw', '')
                cf_matches = [f for f in clean_files if base in f.replace('clean', '').replace('Clean', '').replace('gt', '').replace('GT', '')]

                if cf_matches:
                    try:
                        noisy = np.array(Image.open(os.path.join(root, nf)))
                        clean = np.array(Image.open(os.path.join(root, cf_matches[0])))

                        if noisy.shape == clean.shape and len(noisy.shape) == 2:
                            result = analyze_image_pair(clean, noisy)
                            result['file'] = nf
                            all_results.append(result)
                    except:
                        continue

    # Alternative: look for numbered pairs
    if not all_results:
        for i in range(1, 50):
            for ext in ['.png', '.tif', '.bmp']:
                noisy_path = os.path.join(pku_dir, f'noisy_{i:03d}{ext}')
                clean_path = os.path.join(pku_dir, f'clean_{i:03d}{ext}')

                if os.path.exists(noisy_path) and os.path.exists(clean_path):
                    try:
                        noisy = np.array(Image.open(noisy_path))
                        clean = np.array(Image.open(clean_path))

                        if noisy.shape == clean.shape:
                            result = analyze_image_pair(clean, noisy)
                            result['file'] = f'pair_{i}'
                            all_results.append(result)
                    except:
                        continue

    if not all_results:
        print("  No valid pairs found in PKU37, checking pairs file...")

        # Try reading from pairs file
        pairs_file = os.path.join(dataset_dir, 'pku37_test_pairs.txt')
        if os.path.exists(pairs_file):
            with open(pairs_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        noisy_path, clean_path = parts[0], parts[1]
                        if os.path.exists(noisy_path) and os.path.exists(clean_path):
                            try:
                                noisy = np.array(Image.open(noisy_path))
                                clean = np.array(Image.open(clean_path))
                                if noisy.shape == clean.shape:
                                    result = analyze_image_pair(clean, noisy)
                                    result['file'] = os.path.basename(noisy_path)
                                    all_results.append(result)
                                    if len(all_results) >= 20:  # Limit
                                        break
                            except:
                                continue

    if not all_results:
        print("  No valid pairs found in PKU37")
        return None

    return aggregate_results(all_results, 'PKU37')


def aggregate_results(results, dataset_name):
    """Aggregate analysis results from multiple images."""
    if not results:
        return None

    n = len(results)
    print(f"  Analyzed {n} image pairs")

    # Aggregate metrics
    agg = {
        'dataset': dataset_name,
        'num_images': n,
        'metrics': {
            'psnr_full': np.mean([r['metrics']['psnr_full'] for r in results]),
            'ssim_full': np.mean([r['metrics']['ssim_full'] for r in results]),
            'psnr_masked': np.mean([r['metrics']['psnr_masked'] for r in results]),
            'ssim_masked': np.mean([r['metrics']['ssim_masked'] for r in results]),
        },
        'noise_stats': {
            'mean': np.mean([r['noise_stats']['mean'] for r in results]),
            'std': np.mean([r['noise_stats']['std'] for r in results]),
            'std_range': [
                np.min([r['noise_stats']['std'] for r in results]),
                np.max([r['noise_stats']['std'] for r in results]),
            ],
        },
        'spatial_correlation': {
            'v_corr_at_1': np.mean([r['spatial_correlation']['v_corr_at_1'] for r in results]),
            'h_corr_at_1': np.mean([r['spatial_correlation']['h_corr_at_1'] for r in results]),
            'v_corr_length': np.mean([r['spatial_correlation']['v_corr_length'] for r in results]),
            'h_corr_length': np.mean([r['spatial_correlation']['h_corr_length'] for r in results]),
            'v_corr_profile': np.mean([r['spatial_correlation']['vertical'] for r in results], axis=0).tolist(),
            'h_corr_profile': np.mean([r['spatial_correlation']['horizontal'] for r in results], axis=0).tolist(),
        },
        'speckle': {
            'cv': np.mean([r['speckle']['cv'] for r in results]),
            'gamma_k': np.mean([r['speckle']['gamma_k'] for r in results]),
            'v_correlation': np.mean([r['speckle']['v_correlation'] for r in results]),
            'h_correlation': np.mean([r['speckle']['h_correlation'] for r in results]),
        },
        'banding': {
            'amplitude': np.mean([r['banding']['amplitude'] for r in results]),
            'dominant_freq': np.mean([r['banding']['dominant_freq'] for r in results]),
            'power_ratio': np.mean([r['banding']['power_ratio'] for r in results]),
        },
        'gaussian': {
            'sigma': np.mean([r['gaussian']['sigma'] for r in results]),
            'v_correlation': np.mean([r['gaussian']['v_correlation'] for r in results]),
            'h_correlation': np.mean([r['gaussian']['h_correlation'] for r in results]),
        },
        'shot': {
            'gain': np.mean([r['shot']['gain'] for r in results]),
            'is_signal_dependent': np.mean([r['shot']['is_signal_dependent'] for r in results]) > 0.5,
        },
        'composition': {
            'speckle': np.mean([r['composition']['speckle'] for r in results]),
            'banding': np.mean([r['composition']['banding'] for r in results]),
            'gaussian': np.mean([r['composition']['gaussian'] for r in results]),
            'shot': np.mean([r['composition']['shot'] for r in results]),
        },
    }

    return agg


def print_dataset_summary(result):
    """Print summary of dataset analysis."""
    if result is None:
        return

    print(f"\n{result['dataset']} Summary ({result['num_images']} images):")
    print("-" * 50)

    print(f"  PSNR: {result['metrics']['psnr_masked']:.2f} dB (masked), {result['metrics']['psnr_full']:.2f} dB (full)")
    print(f"  SSIM: {result['metrics']['ssim_masked']:.4f} (masked), {result['metrics']['ssim_full']:.4f} (full)")
    print(f"  Noise std: {result['noise_stats']['std']:.4f} [{result['noise_stats']['std_range'][0]:.4f}, {result['noise_stats']['std_range'][1]:.4f}]")

    print(f"\n  Spatial Correlation:")
    print(f"    Vertical (lag=1):   {result['spatial_correlation']['v_corr_at_1']:.4f}")
    print(f"    Horizontal (lag=1): {result['spatial_correlation']['h_corr_at_1']:.4f}")
    print(f"    V corr length: {result['spatial_correlation']['v_corr_length']:.1f}, H corr length: {result['spatial_correlation']['h_corr_length']:.1f}")

    print(f"\n  Noise Composition:")
    print(f"    Speckle:  {result['composition']['speckle']*100:.1f}%")
    print(f"    Banding:  {result['composition']['banding']*100:.1f}%")
    print(f"    Gaussian: {result['composition']['gaussian']*100:.1f}%")
    print(f"    Shot:     {result['composition']['shot']*100:.1f}%")

    print(f"\n  Speckle Parameters:")
    print(f"    CV: {result['speckle']['cv']:.4f}, Gamma k: {result['speckle']['gamma_k']:.2f}")
    print(f"    V corr: {result['speckle']['v_correlation']:.4f}, H corr: {result['speckle']['h_correlation']:.4f}")

    print(f"\n  Banding Parameters:")
    print(f"    Amplitude: {result['banding']['amplitude']:.4f}, Dominant freq: {result['banding']['dominant_freq']:.4f}")

    print(f"\n  Gaussian Parameters:")
    print(f"    Sigma: {result['gaussian']['sigma']:.4f}")
    print(f"    V corr: {result['gaussian']['v_correlation']:.4f}, H corr: {result['gaussian']['h_correlation']:.4f}")

    print(f"\n  Shot Noise:")
    print(f"    Gain: {result['shot']['gain']:.4f}, Signal-dependent: {result['shot']['is_signal_dependent']}")


def compute_combined_parameters(results):
    """Compute combined parameters across all datasets."""
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        return None

    # Weight by number of images
    total_images = sum(r['num_images'] for r in valid_results)

    def weighted_avg(key_path):
        """Compute weighted average for a nested key."""
        vals = []
        weights = []
        for r in valid_results:
            val = r
            for k in key_path.split('.'):
                val = val[k]
            vals.append(val)
            weights.append(r['num_images'])
        return np.average(vals, weights=weights)

    combined = {
        'num_datasets': len(valid_results),
        'total_images': total_images,
        'datasets_analyzed': [r['dataset'] for r in valid_results],

        'metrics': {
            'psnr_masked': weighted_avg('metrics.psnr_masked'),
            'ssim_masked': weighted_avg('metrics.ssim_masked'),
        },

        'noise_stats': {
            'std': weighted_avg('noise_stats.std'),
        },

        'spatial_correlation': {
            'v_corr_at_1': weighted_avg('spatial_correlation.v_corr_at_1'),
            'h_corr_at_1': weighted_avg('spatial_correlation.h_corr_at_1'),
            'v_corr_length': weighted_avg('spatial_correlation.v_corr_length'),
            'h_corr_length': weighted_avg('spatial_correlation.h_corr_length'),
        },

        'composition': {
            'speckle': weighted_avg('composition.speckle'),
            'banding': weighted_avg('composition.banding'),
            'gaussian': weighted_avg('composition.gaussian'),
            'shot': weighted_avg('composition.shot'),
        },

        'speckle_params': {
            'cv': weighted_avg('speckle.cv'),
            'gamma_k': weighted_avg('speckle.gamma_k'),
            'v_correlation': weighted_avg('speckle.v_correlation'),
            'h_correlation': weighted_avg('speckle.h_correlation'),
        },

        'banding_params': {
            'amplitude': weighted_avg('banding.amplitude'),
            'dominant_freq': weighted_avg('banding.dominant_freq'),
        },

        'gaussian_params': {
            'sigma': weighted_avg('gaussian.sigma'),
            'v_correlation': weighted_avg('gaussian.v_correlation'),
            'h_correlation': weighted_avg('gaussian.h_correlation'),
        },

        'shot_params': {
            'gain': weighted_avg('shot.gain'),
        },
    }

    # Compute Dirichlet alpha parameters for sampling
    comp = combined['composition']
    scale = 10.0  # Concentration parameter
    combined['dirichlet_alpha'] = {
        'speckle': comp['speckle'] * scale,
        'banding': comp['banding'] * scale,
        'gaussian': comp['gaussian'] * scale,
        'shot': comp['shot'] * scale,
        'scale': scale,
    }

    return combined


def main():
    print("="*70)
    print("COMPREHENSIVE OCT NOISE ANALYSIS")
    print("="*70)
    print("Analyzing noise characteristics from all available datasets")
    print("to calibrate synthetic noise generation.\n")

    results = []

    # Duke17
    duke17_dir = 'duke_sota_datasets/Sparsity_SDOCT_DATASET_2012'
    if os.path.exists(duke17_dir):
        results.append(analyze_duke17(duke17_dir))

    # Duke28
    duke28_dir = 'duke_sota_datasets/Duke2013_SBSDI'
    if os.path.exists(duke28_dir):
        results.append(analyze_duke28(duke28_dir))

    # PKU37
    pku37_dir = 'pku37_oct_dataset'
    if os.path.exists(pku37_dir):
        results.append(analyze_pku37(pku37_dir))

    # Print summaries
    print("\n" + "="*70)
    print("DATASET SUMMARIES")
    print("="*70)

    for r in results:
        print_dataset_summary(r)

    # Compute combined parameters
    combined = compute_combined_parameters(results)

    if combined:
        print("\n" + "="*70)
        print("COMBINED PARAMETERS (Weighted Average)")
        print("="*70)
        print(f"Datasets: {combined['datasets_analyzed']}")
        print(f"Total images: {combined['total_images']}")

        print(f"\nTarget Metrics:")
        print(f"  PSNR (masked): {combined['metrics']['psnr_masked']:.2f} dB")
        print(f"  SSIM (masked): {combined['metrics']['ssim_masked']:.4f}")

        print(f"\nNoise Composition Weights:")
        for k, v in combined['composition'].items():
            print(f"  {k:10s}: {v*100:.1f}%")

        print(f"\nDirichlet Alpha (for sampling):")
        for k, v in combined['dirichlet_alpha'].items():
            if k != 'scale':
                print(f"  {k:10s}: {v:.3f}")

        print(f"\nSpeckle Parameters:")
        for k, v in combined['speckle_params'].items():
            print(f"  {k:15s}: {v:.4f}")

        print(f"\nSpatial Correlation:")
        print(f"  Vertical (lag=1):   {combined['spatial_correlation']['v_corr_at_1']:.4f}")
        print(f"  Horizontal (lag=1): {combined['spatial_correlation']['h_corr_at_1']:.4f}")

        print(f"\nBanding Parameters:")
        for k, v in combined['banding_params'].items():
            print(f"  {k:15s}: {v:.4f}")

        print(f"\nGaussian Parameters:")
        for k, v in combined['gaussian_params'].items():
            print(f"  {k:15s}: {v:.4f}")

        print(f"\nShot Parameters:")
        for k, v in combined['shot_params'].items():
            print(f"  {k:15s}: {v:.4f}")

    # Save results
    output = {
        'per_dataset': {r['dataset']: r for r in results if r is not None},
        'combined': combined,
    }

    output_path = 'noise_analysis_all_datasets.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
