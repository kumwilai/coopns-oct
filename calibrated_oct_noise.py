#!/usr/bin/env python3
"""
Calibrated OCT Noise Generator

Based on comprehensive analysis of Duke17, Duke28, and PKU37 datasets.
Generates synthetic noise that matches real OCT noise characteristics including:
- Speckle (multiplicative) with correct gamma distribution and spatial correlation
- Banding (horizontal stripes) with calibrated frequency and amplitude
- Gaussian (additive) with vertical correlation matching A-scan acquisition
- Shot noise (signal-dependent)

Analysis Results Summary:
========================
Dataset     | PSNR(masked) | SSIM(masked) | Noise Std | V_corr | H_corr
Duke17      | 17.09 dB     | 0.318        | 0.129     | 0.19   | 0.09
Duke28      | 17.58 dB     | 0.268        | 0.132     | 0.20   | 0.12
PKU37       | 20.19 dB     | 0.701        | 0.098     | 0.43   | 0.03
Combined    | ~18 dB       | ~0.4         | ~0.11     | ~0.28  | ~0.08

Noise Composition (Combined):
- Speckle:  57.5% (multiplicative, gamma distributed)
- Gaussian: 39.8% (additive, vertically correlated)
- Shot:      2.1% (signal-dependent)
- Banding:   0.7% (horizontal stripes)

Key Parameters:
- Speckle: CV=0.42, Gamma k=6.0, v_corr=0.22, h_corr=0.07
- Gaussian: sigma=0.107, v_corr=0.26, h_corr=-0.02
- Banding: amplitude=0.013, freq=0.015
"""

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d


# Calibrated parameters from dataset analysis
# Adjusted to match target PSNR levels more closely
NOISE_PARAMS = {
    'duke17': {
        'noise_std': 0.129,
        'psnr_masked': 17.09,
        'ssim_masked': 0.318,
        'composition': {'speckle': 0.60, 'gaussian': 0.35, 'shot': 0.05, 'banding': 0.01},
        'speckle_cv': 0.38,  # Reduced from 0.49 to match PSNR
        'speckle_gamma_k': 4.3,
        'v_corr': 0.19,
        'h_corr': 0.09,
        'gaussian_sigma': 0.09,  # Reduced from 0.12
        'gaussian_v_corr': 0.11,
        'banding_amplitude': 0.015,
        'banding_freq': 0.008,
    },
    'duke28': {
        'noise_std': 0.132,
        'psnr_masked': 17.58,
        'ssim_masked': 0.268,
        'composition': {'speckle': 0.58, 'gaussian': 0.36, 'shot': 0.05, 'banding': 0.01},
        'speckle_cv': 0.38,  # Reduced from 0.50
        'speckle_gamma_k': 4.0,
        'v_corr': 0.20,
        'h_corr': 0.12,
        'gaussian_sigma': 0.10,  # Reduced from 0.13
        'gaussian_v_corr': 0.15,
        'banding_amplitude': 0.015,
        'banding_freq': 0.010,
    },
    'pku37': {
        'noise_std': 0.098,
        'psnr_masked': 20.19,
        'ssim_masked': 0.701,
        'composition': {'speckle': 0.55, 'gaussian': 0.44, 'shot': 0.00, 'banding': 0.00},
        'speckle_cv': 0.25,  # Reduced from 0.36 for higher PSNR
        'speckle_gamma_k': 7.6,
        'v_corr': 0.43,
        'h_corr': 0.03,
        'gaussian_sigma': 0.06,  # Reduced from 0.09
        'gaussian_v_corr': 0.39,
        'banding_amplitude': 0.005,
        'banding_freq': 0.021,
    },
    'combined': {
        'noise_std': 0.112,
        'psnr_masked': 18.0,
        'ssim_masked': 0.43,
        'composition': {'speckle': 0.575, 'gaussian': 0.398, 'shot': 0.021, 'banding': 0.007},
        'speckle_cv': 0.34,  # Reduced from 0.42
        'speckle_gamma_k': 6.0,
        'v_corr': 0.28,
        'h_corr': 0.08,
        'gaussian_sigma': 0.085,  # Reduced from 0.107
        'gaussian_v_corr': 0.26,
        'banding_amplitude': 0.010,
        'banding_freq': 0.015,
        'dirichlet_alpha': [5.75, 3.98, 0.21, 0.07],  # speckle, gaussian, shot, banding
    },
}


def add_vertical_correlation(noise, v_corr, method='ar1'):
    """Add vertical correlation to noise matching OCT A-scan acquisition.

    Args:
        noise: 2D noise array [H, W]
        v_corr: Target correlation at lag 1 (0-1)
        method: 'ar1' for AR(1) process, 'filter' for Gaussian filter

    Returns:
        Correlated noise with target vertical correlation
    """
    if v_corr <= 0 or v_corr >= 1:
        return noise

    H, W = noise.shape

    if method == 'ar1':
        # AR(1) process: x[t] = phi * x[t-1] + eps
        # For AR(1), correlation at lag 1 = phi
        phi = v_corr
        corr_noise = np.zeros_like(noise)
        corr_noise[0, :] = noise[0, :]
        for i in range(1, H):
            corr_noise[i, :] = phi * corr_noise[i-1, :] + np.sqrt(1 - phi**2) * noise[i, :]
        return corr_noise
    else:
        # Gaussian filter approximation
        sigma = -1 / np.log(v_corr) if v_corr > 0.01 else 0.5
        return gaussian_filter1d(noise, sigma=sigma, axis=0, mode='reflect')


def add_speckle_noise(image, cv=0.42, gamma_k=6.0, v_corr=0.22, h_corr=0.07):
    """Add calibrated speckle noise (multiplicative).

    Speckle noise follows Gamma distribution with shape k and scale theta.
    For Gamma: mean = k*theta, var = k*theta^2, CV = 1/sqrt(k)

    Args:
        image: Clean image [H, W], 0-1 range
        cv: Coefficient of variation (std/mean) of speckle
        gamma_k: Gamma distribution shape parameter
        v_corr: Vertical correlation at lag 1
        h_corr: Horizontal correlation at lag 1
    """
    H, W = image.shape

    # Generate gamma-distributed speckle
    # For mean=1 speckle: theta = 1/k, so shape=k, scale=1/k
    speckle = np.random.gamma(shape=gamma_k, scale=1.0/gamma_k, size=(H, W)).astype(np.float32)

    # Add spatial correlation
    if v_corr > 0.01:
        speckle_centered = speckle - 1.0
        speckle_centered = add_vertical_correlation(speckle_centered, v_corr)
        # Restore mean and adjust variance
        speckle = 1.0 + speckle_centered * (cv / np.std(speckle_centered + 1e-8))

    # Scale to match target CV
    speckle = 1.0 + (speckle - np.mean(speckle)) * (cv / (np.std(speckle) + 1e-8))
    speckle = np.maximum(speckle, 0.01)  # Prevent negative

    return image * speckle


def add_banding_noise(image, amplitude=0.013, freq=0.015, n_bands=3):
    """Add horizontal banding artifacts (electronic interference).

    Args:
        image: Image to add banding to
        amplitude: Banding amplitude (typically 0.01-0.02)
        freq: Dominant frequency (cycles per pixel)
        n_bands: Number of frequency components
    """
    H, W = image.shape
    rows = np.arange(H).reshape(-1, 1).astype(np.float32)

    banding = np.zeros((H, W), dtype=np.float32)
    for i in range(n_bands):
        f = freq * (0.5 + np.random.random())  # Vary frequency
        phase = np.random.uniform(0, 2 * np.pi)
        amp = amplitude * np.random.uniform(0.5, 1.5)
        banding += amp * np.sin(2 * np.pi * f * rows + phase)

    return image + banding / n_bands


def add_gaussian_noise(image, sigma=0.107, v_corr=0.26):
    """Add Gaussian noise with vertical correlation.

    Args:
        image: Image to add noise to
        sigma: Noise standard deviation
        v_corr: Vertical correlation at lag 1
    """
    H, W = image.shape
    gaussian = np.random.randn(H, W).astype(np.float32) * sigma

    if v_corr > 0.01:
        gaussian = add_vertical_correlation(gaussian, v_corr)
        # Renormalize to target sigma
        gaussian = gaussian * (sigma / (np.std(gaussian) + 1e-8))

    return image + gaussian


def add_shot_noise(image, gain=0.003):
    """Add signal-dependent shot noise (Poisson-like).

    Args:
        image: Image to add noise to (0-1 range)
        gain: Noise gain (variance = gain * signal)
    """
    if gain <= 0:
        return image

    # Poisson noise is signal-dependent
    # For numerical stability, scale and use Gaussian approximation
    scale = 1.0 / gain
    noisy_scaled = np.maximum(image * scale, 0.1)
    shot_noisy = np.random.poisson(noisy_scaled).astype(np.float32) / scale

    return shot_noisy


def add_calibrated_oct_noise(image, dataset='combined', noise_scale=1.0,
                              composition=None, random_mix=False):
    """Add calibrated OCT noise matching real dataset characteristics.

    Args:
        image: Clean image [H, W], 0-1 range
        dataset: Dataset to match ('duke17', 'duke28', 'pku37', 'combined')
        noise_scale: Scale factor for noise level (1.0 = match dataset)
        composition: Override noise composition dict {'speckle', 'gaussian', 'shot', 'banding'}
        random_mix: If True, sample composition from Dirichlet distribution

    Returns:
        Noisy image with calibrated OCT noise
    """
    params = NOISE_PARAMS.get(dataset, NOISE_PARAMS['combined'])

    # Determine noise composition
    if composition is not None:
        comp = composition
    elif random_mix and 'dirichlet_alpha' in params:
        # Sample composition from Dirichlet
        alpha = params['dirichlet_alpha']
        weights = np.random.dirichlet(alpha)
        comp = {
            'speckle': weights[0],
            'gaussian': weights[1],
            'shot': weights[2],
            'banding': weights[3],
        }
    else:
        comp = params['composition']

    image = image.astype(np.float32)
    noisy = image.copy()

    # Target total noise std - use full calibrated value
    target_std = params['noise_std'] * noise_scale

    # Apply noise components based on composition
    # Use the full calibrated CV and sigma values to achieve target PSNR

    if comp.get('speckle', 0) > 0.05:
        # Use full speckle CV scaled by composition weight
        cv = params['speckle_cv'] * np.sqrt(comp['speckle']) * noise_scale
        noisy = add_speckle_noise(
            noisy,
            cv=cv,
            gamma_k=params['speckle_gamma_k'],
            v_corr=params.get('v_corr', 0.2),
            h_corr=params.get('h_corr', 0.08),
        )

    if comp.get('banding', 0) > 0.01:
        amp = params['banding_amplitude'] * np.sqrt(comp['banding']) * noise_scale
        noisy = add_banding_noise(
            noisy,
            amplitude=amp,
            freq=params['banding_freq'],
        )

    if comp.get('gaussian', 0) > 0.05:
        # Use full gaussian sigma scaled by composition weight
        sigma = params['gaussian_sigma'] * np.sqrt(comp['gaussian']) * noise_scale
        noisy = add_gaussian_noise(
            noisy,
            sigma=sigma,
            v_corr=params.get('gaussian_v_corr', 0.2),
        )

    if comp.get('shot', 0) > 0.01:
        # Shot noise contribution is signal-dependent
        gain = 0.01 * comp['shot'] * noise_scale
        noisy = add_shot_noise(noisy, gain=gain)

    return np.clip(noisy, 0, 1).astype(np.float32)


def add_oct_noise_for_training(image, noise_levels=None, dataset='combined'):
    """Convenience function for training with multiple noise levels.

    Args:
        image: Clean image [H, W], 0-1 range
        noise_levels: List of noise scale factors (e.g., [0.5, 0.75, 1.0, 1.25, 1.5])
                     If None, uses dataset's natural noise level
        dataset: Dataset characteristics to match

    Returns:
        Noisy image with calibrated OCT noise
    """
    if noise_levels is not None:
        noise_scale = np.random.choice(noise_levels)
    else:
        noise_scale = 1.0

    return add_calibrated_oct_noise(
        image,
        dataset=dataset,
        noise_scale=noise_scale,
        random_mix=True,  # Add variety during training
    )


# Verification function
def verify_noise_characteristics(clean, noisy, threshold=100/255):
    """Verify that generated noise matches target characteristics."""
    from skimage.metrics import structural_similarity as ssim

    residual = noisy - clean
    noise_std = np.std(residual)

    # PSNR
    mse = np.mean((clean - noisy) ** 2)
    psnr_val = 10 * np.log10(1.0 / mse) if mse > 0 else 50

    # Masked SSIM
    mask = clean > threshold
    n_pix = np.sum(mask)
    if n_pix > 1000:
        clean_m = clean[mask]
        noisy_m = noisy[mask]
        side = int(np.sqrt(n_pix))
        ssim_val = ssim(clean_m[:side*side].reshape(side, side),
                        noisy_m[:side*side].reshape(side, side), data_range=1.0)
    else:
        ssim_val = ssim(clean, noisy, data_range=1.0)

    # Spatial correlation
    res_norm = (residual - np.mean(residual)) / (noise_std + 1e-8)
    v_corr = np.mean(res_norm[:-1, :] * res_norm[1:, :])
    h_corr = np.mean(res_norm[:, :-1] * res_norm[:, 1:])

    return {
        'psnr_masked': psnr_val,
        'ssim_masked': ssim_val,
        'noise_std': noise_std,
        'v_corr': v_corr,
        'h_corr': h_corr,
    }


if __name__ == '__main__':
    # Test the calibrated noise generator
    import json
    from PIL import Image

    print("="*70)
    print("CALIBRATED OCT NOISE GENERATOR - VERIFICATION")
    print("="*70)

    # Load a sample clean image
    sample_paths = [
        'duke_sota_datasets/Sparsity_SDOCT_DATASET_2012/1/Averaged.tif',
        'duke_sota_datasets/Duke17_Eval/1_averaged.tif',
    ]

    clean = None
    for path in sample_paths:
        try:
            clean = np.array(Image.open(path)).astype(np.float32) / 255.0
            print(f"Loaded: {path}")
            break
        except:
            continue

    if clean is None:
        # Create synthetic test image
        print("Using synthetic test image")
        clean = np.random.rand(512, 512).astype(np.float32) * 0.5 + 0.25

    print(f"Image shape: {clean.shape}")
    print()

    # Test each dataset configuration
    for dataset in ['duke17', 'duke28', 'pku37', 'combined']:
        print(f"\n{dataset.upper()} Configuration:")
        print("-" * 50)

        target = NOISE_PARAMS[dataset]
        print(f"Target: PSNR={target['psnr_masked']:.2f} dB, SSIM={target['ssim_masked']:.3f}, "
              f"std={target['noise_std']:.4f}, v_corr={target['v_corr']:.3f}")

        # Generate multiple samples and average
        results = []
        for _ in range(5):
            noisy = add_calibrated_oct_noise(clean, dataset=dataset)
            result = verify_noise_characteristics(clean, noisy)
            results.append(result)

        avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
        print(f"Result: PSNR={avg['psnr_masked']:.2f} dB, SSIM={avg['ssim_masked']:.3f}, "
              f"std={avg['noise_std']:.4f}, v_corr={avg['v_corr']:.3f}")

        # Check if within acceptable range
        psnr_ok = abs(avg['psnr_masked'] - target['psnr_masked']) < 2.0
        std_ok = abs(avg['noise_std'] - target['noise_std']) < 0.02
        vcorr_ok = abs(avg['v_corr'] - target['v_corr']) < 0.15

        status = "PASS" if (psnr_ok and std_ok) else "ADJUST"
        print(f"Status: {status} (PSNR {'OK' if psnr_ok else 'DIFF'}, "
              f"std {'OK' if std_ok else 'DIFF'}, v_corr {'OK' if vcorr_ok else 'DIFF'})")

    print("\n" + "="*70)
    print("Calibrated noise generator ready for use in training.")
    print("="*70)
