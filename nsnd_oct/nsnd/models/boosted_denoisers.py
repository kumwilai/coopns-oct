"""
Boosted Denoisers to Exceed Gaussian Baseline

Strategies to achieve PSNR > 25.96 dB (beating simple Gaussian σ=1.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple


class ResidualRefinementDenoiser(nn.Module):
    """
    Two-stage denoising:
    1. First pass: Gaussian σ=1.5 (25.96 dB baseline)
    2. Second pass: Denoise the residual using lightweight network

    Target: 26-27 dB (beat Gaussian by 0-1 dB)
    """

    def __init__(self, nsnd_model=None, device='cpu'):
        super().__init__()
        self.device = device

        # Lightweight residual denoiser (very small to avoid overfitting)
        self.residual_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh()  # Residual in [-1, 1]
        ).to(device)

        self.nsnd_model = nsnd_model
        self.alpha = 0.3  # Residual blending weight (learnable)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Two-stage denoising

        Args:
            noisy: [B, 1, H, W]

        Returns:
            refined: [B, 1, H, W]
        """
        # Stage 1: Gaussian baseline
        gaussian_out = self._apply_gaussian(noisy, sigma=1.5)

        # Stage 2: Refine residual
        residual_input = noisy - gaussian_out
        residual_refined = self.residual_net(residual_input)

        # Final output
        output = gaussian_out + self.alpha * residual_refined

        return output

    def _apply_gaussian(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian filter"""
        B = img.shape[0]
        filtered = []
        for i in range(B):
            img_np = img[i, 0].cpu().numpy()
            img_filt = gaussian_filter(img_np, sigma=sigma)
            filtered.append(torch.from_numpy(img_filt))
        return torch.stack(filtered, dim=0).unsqueeze(1).to(img.device)


class AdaptiveGaussianDenoiser(nn.Module):
    """
    Spatially-adaptive Gaussian filtering

    Instead of fixed σ=1.5, adapt sigma locally based on:
    - Local noise variance (smooth more in noisy regions)
    - Local gradient (smooth less near edges)

    Target: 26-27 dB (beat fixed Gaussian)
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.sigma_min = 0.8
        self.sigma_max = 2.5

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Gaussian denoising

        Args:
            noisy: [B, 1, H, W]

        Returns:
            denoised: [B, 1, H, W]
        """
        B, C, H, W = noisy.shape

        # Estimate local noise variance
        local_var = self._estimate_local_variance(noisy, window=7)

        # Estimate local gradient (edge strength)
        gradient = self._compute_gradient(noisy)

        # Compute adaptive sigma
        # High variance → high sigma (smooth more)
        # High gradient → low sigma (preserve edges)
        sigma_map = self._compute_adaptive_sigma(local_var, gradient)

        # Apply spatially-varying Gaussian
        denoised = self._apply_adaptive_gaussian(noisy, sigma_map)

        return denoised

    def _estimate_local_variance(self, img: torch.Tensor, window: int = 7) -> torch.Tensor:
        """Estimate local variance using sliding window"""
        # Use avg pooling for local mean
        pad = window // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')

        local_mean = F.avg_pool2d(img_padded, kernel_size=window, stride=1)
        local_mean_sq = F.avg_pool2d(img_padded ** 2, kernel_size=window, stride=1)

        local_var = local_mean_sq - local_mean ** 2
        return torch.clamp(local_var, min=0)

    def _compute_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), sobel_x)
        grad_y = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), sobel_y)

        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return gradient

    def _compute_adaptive_sigma(self, variance: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute spatially-varying sigma

        High variance → high sigma (noisy region, smooth more)
        High gradient → low sigma (edge region, preserve)
        """
        # Normalize variance and gradient to [0, 1]
        var_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
        grad_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)

        # Sigma increases with variance, decreases with gradient
        sigma_map = self.sigma_min + (self.sigma_max - self.sigma_min) * var_norm * (1 - 0.5 * grad_norm)

        return sigma_map

    def _apply_adaptive_gaussian(self, img: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        """
        Apply spatially-varying Gaussian (approximate with fixed scales)

        Since true spatially-varying Gaussian is expensive, we:
        1. Denoise at multiple fixed sigmas
        2. Blend based on local sigma_map
        """
        B = img.shape[0]

        # Pre-compute denoised versions at different scales
        sigmas = [0.8, 1.2, 1.5, 2.0, 2.5]
        denoised_scales = []

        for sigma in sigmas:
            denoised = self._apply_gaussian_fixed(img, sigma)
            denoised_scales.append(denoised)

        # Stack: [B, num_scales, 1, H, W]
        denoised_scales = torch.stack(denoised_scales, dim=1)

        # For each pixel, select based on local sigma
        # Softmax over scales based on distance to sigma_map
        weights = []
        for sigma in sigmas:
            dist = torch.abs(sigma_map - sigma)
            weight = torch.exp(-dist / 0.3)  # Soft selection
            weights.append(weight)

        weights = torch.stack(weights, dim=1)  # [B, num_scales, 1, H, W]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted combination
        output = (denoised_scales * weights).sum(dim=1)

        return output

    def _apply_gaussian_fixed(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply fixed Gaussian"""
        B = img.shape[0]
        filtered = []
        for i in range(B):
            img_np = img[i, 0].cpu().numpy()
            img_filt = gaussian_filter(img_np, sigma=sigma)
            filtered.append(torch.from_numpy(img_filt))
        return torch.stack(filtered, dim=0).unsqueeze(1).to(img.device)


class MultiScaleGaussianFusion(nn.Module):
    """
    Optimal fusion of Gaussian at multiple scales

    Instead of single σ=1.5, compute weighted combination of:
    σ ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5}

    Learn optimal weights (potentially spatially-varying)

    Target: 26-28 dB
    """

    def __init__(self, learnable=True, device='cpu'):
        super().__init__()
        self.device = device
        self.sigmas = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
        self.n_scales = len(self.sigmas)

        if learnable:
            # Learn global weights for each scale
            self.scale_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)
        else:
            # Fixed uniform weights
            self.register_buffer('scale_weights', torch.ones(self.n_scales) / self.n_scales)

        self.learnable = learnable

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale Gaussian fusion

        Args:
            noisy: [B, 1, H, W]

        Returns:
            fused: [B, 1, H, W]
        """
        # Denoise at each scale
        denoised_scales = []
        for sigma in self.sigmas:
            denoised = self._apply_gaussian(noisy, sigma)
            denoised_scales.append(denoised)

        # Stack: [B, n_scales, 1, H, W]
        denoised_stack = torch.stack(denoised_scales, dim=1)

        # Normalize weights
        weights = F.softmax(self.scale_weights, dim=0)

        # Weighted fusion: [B, 1, H, W]
        output = (denoised_stack * weights.view(1, self.n_scales, 1, 1, 1)).sum(dim=1)

        return output

    def _apply_gaussian(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian filter"""
        B = img.shape[0]
        filtered = []
        for i in range(B):
            img_np = img[i, 0].cpu().numpy()
            img_filt = gaussian_filter(img_np, sigma=sigma)
            filtered.append(torch.from_numpy(img_filt))
        return torch.stack(filtered, dim=0).unsqueeze(1).to(img.device)


class WienerFilterDenoiser(nn.Module):
    """
    Wiener filtering in frequency domain

    Optimal linear filter when signal and noise spectra are known

    H(f) = |S(f)|² / (|S(f)|² + |N(f)|²)

    where S(f) = signal spectrum, N(f) = noise spectrum

    Target: 26-27 dB
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Wiener filtering

        Args:
            noisy: [B, 1, H, W]

        Returns:
            denoised: [B, 1, H, W]
        """
        B, C, H, W = noisy.shape

        denoised = []
        for i in range(B):
            img = noisy[i, 0].cpu().numpy()

            # Estimate noise power spectrum
            noise_power = self._estimate_noise_power(img)

            # Compute FFT
            img_fft = np.fft.fft2(img)
            img_power = np.abs(img_fft) ** 2

            # Wiener filter
            signal_power = np.maximum(img_power - noise_power, 0)
            wiener_filter = signal_power / (signal_power + noise_power + 1e-8)

            # Apply filter
            img_filtered_fft = img_fft * wiener_filter
            img_filtered = np.fft.ifft2(img_filtered_fft).real

            denoised.append(torch.from_numpy(img_filtered.astype(np.float32)))

        output = torch.stack(denoised, dim=0).unsqueeze(1).to(noisy.device)
        return output

    def _estimate_noise_power(self, img: np.ndarray) -> float:
        """Estimate noise power spectrum (assume white noise)"""
        # Simple estimate: variance of high-frequency components
        img_fft = np.fft.fft2(img)
        power_spectrum = np.abs(img_fft) ** 2

        # Assume noise is white → uniform power across frequencies
        # Estimate from high-frequency region
        H, W = img.shape
        hf_region = power_spectrum[H//2:, W//2:]  # High-frequency quadrant
        noise_power = np.median(hf_region)

        return noise_power


def train_boosted_denoiser(model, train_loader, device='cpu', epochs=30, lr=1e-3):
    """
    Train boosted denoiser to beat Gaussian baseline

    Target: PSNR > 25.96 dB
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining Boosted Denoiser")
    print("="*60)
    print(f"Target: Beat Gaussian baseline (25.96 dB)")
    print("="*60)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Forward
            output = model(noisy)

            # Loss: MSE
            loss = F.mse_loss(output, clean)

            # PSNR
            psnr = 10 * torch.log10(1.0 / (loss + 1e-8))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_psnr += psnr.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB")

    print("="*60)
    print("Training complete!")

    return model
