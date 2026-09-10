"""
Physics-based component denoisers

Each denoiser is optimized for a specific noise type:
- Speckle: Anisotropic diffusion (edge-preserving)
- Banding: Fourier notch filtering
- Gaussian: BM3D-inspired or lightweight DnCNN
- Shot: Variance-stabilizing transform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .nafnet import NAFNetSmall


class SpeckleDenoiser(nn.Module):
    """
    Anisotropic Diffusion for multiplicative speckle noise

    Based on Perona-Malik diffusion with edge-stopping function
    optimized for OCT retinal layers
    """

    def __init__(
        self,
        iterations: int = 10,
        kappa: float = 50.0,
        gamma: float = 0.1,
        edge_function: str = 'tukey',
    ):
        """
        Args:
            iterations: Number of diffusion iterations
            kappa: Edge threshold (higher = more smoothing)
            gamma: Time step (0.1-0.25 for stability)
            edge_function: 'perona-malik' or 'tukey' (better for OCT)
        """
        super().__init__()
        self.iterations = iterations
        self.kappa = kappa
        self.gamma = gamma
        self.edge_function = edge_function

        # Make parameters learnable (optional)
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(kappa)), requires_grad=False)
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anisotropic diffusion

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            denoised: Smoothed image [B, 1, H, W]
        """
        kappa = torch.exp(self.log_kappa)
        gamma = torch.exp(self.log_gamma)

        img = x.clone()

        for _ in range(self.iterations):
            # Compute gradients in 4 directions
            nabla_N = img[:, :, :-1, :] - img[:, :, 1:, :]   # North
            nabla_S = img[:, :, 1:, :] - img[:, :, :-1, :]   # South
            nabla_E = img[:, :, :, 1:] - img[:, :, :, :-1]   # East
            nabla_W = img[:, :, :, :-1] - img[:, :, :, 1:]   # West

            # Edge-stopping conductance function
            if self.edge_function == 'perona-malik':
                c_N = torch.exp(-(nabla_N / kappa) ** 2)
                c_S = torch.exp(-(nabla_S / kappa) ** 2)
                c_E = torch.exp(-(nabla_E / kappa) ** 2)
                c_W = torch.exp(-(nabla_W / kappa) ** 2)
            else:  # Tukey's biweight (better for OCT)
                c_N = self._tukey_biweight(nabla_N, kappa)
                c_S = self._tukey_biweight(nabla_S, kappa)
                c_E = self._tukey_biweight(nabla_E, kappa)
                c_W = self._tukey_biweight(nabla_W, kappa)

            # Update equation: I_t+1 = I_t + gamma * div(c * grad I)
            update = torch.zeros_like(img)

            # North/South contributions
            update[:, :, 1:, :] += gamma * c_N * nabla_N
            update[:, :, :-1, :] -= gamma * c_S * nabla_S

            # East/West contributions
            update[:, :, :, 1:] += gamma * c_E * nabla_E
            update[:, :, :, :-1] -= gamma * c_W * nabla_W

            img = img + update

        return img

    def _tukey_biweight(self, grad: torch.Tensor, kappa: float) -> torch.Tensor:
        """
        Tukey's biweight edge-stopping function
        Better than exponential for OCT (robust to outliers)
        """
        ratio = grad / kappa
        mask = (torch.abs(ratio) <= 1.0).float()
        return mask * (1.0 - ratio ** 2) ** 2


class BandingRemover(nn.Module):
    """
    Fourier-based notch filter for horizontal banding artifacts

    Removes periodic horizontal lines via frequency-domain filtering
    """

    def __init__(
        self,
        notch_freqs: list = [0.02, 0.04, 0.08],
        notch_width: int = 2,
        adaptive: bool = True,
    ):
        """
        Args:
            notch_freqs: Normalized frequencies to suppress (0-0.5)
            notch_width: Width of notch in pixels
            adaptive: If True, detect frequencies automatically
        """
        super().__init__()
        self.notch_freqs = notch_freqs
        self.notch_width = notch_width
        self.adaptive = adaptive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove banding via FFT notch filter

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            filtered: Banding-removed image [B, 1, H, W]
        """
        B, C, H, W = x.shape

        # Apply 2D FFT
        x_complex = x.squeeze(1)  # [B, H, W]
        fft = torch.fft.fft2(x_complex)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        # Create notch filter mask
        if self.adaptive:
            # Detect dominant horizontal frequencies
            magnitude = torch.abs(fft_shifted)
            center_h = H // 2
            vertical_spectrum = magnitude[:, center_h-5:center_h+5, :].mean(dim=1)  # [B, W]

            # Find peaks (excluding DC) - returns list of frequencies
            freqs = self._find_peaks(vertical_spectrum, num_peaks=3)
        else:
            freqs = self.notch_freqs

        # Create mask
        mask = torch.ones_like(fft_shifted)
        for freq in freqs:
            notch_idx = int(abs(freq) * H)
            if notch_idx > 0:
                # Suppress both positive and negative frequencies
                w = self.notch_width
                mask[:, H//2 - notch_idx - w : H//2 - notch_idx + w, :] = 0
                mask[:, H//2 + notch_idx - w : H//2 + notch_idx + w, :] = 0

        # Apply filter
        fft_filtered = fft_shifted * mask

        # Inverse FFT
        fft_unshifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        filtered = torch.fft.ifft2(fft_unshifted).real

        return filtered.unsqueeze(1)

    def _find_peaks(self, spectrum: torch.Tensor, num_peaks: int = 3) -> list:
        """Find dominant frequency peaks"""
        B, W = spectrum.shape
        freqs_list = []

        for b in range(B):
            spec_1d = spectrum[b].detach().cpu().numpy()
            # Simple peak detection (exclude DC component)
            spec_1d[W//2-5:W//2+5] = 0  # Zero out DC
            peak_indices = np.argsort(spec_1d)[-num_peaks:]

            # Convert to frequencies immediately
            freqs = [(int(p) - W//2) / W for p in peak_indices]
            freqs_list.extend(freqs)

        return freqs_list


class GaussianDenoiser(nn.Module):
    """
    Lightweight DnCNN-style denoiser for additive Gaussian noise

    Uses residual learning: predicts noise, then subtracts from input
    """

    def __init__(self, depth: int = 7, channels: int = 32):
        """
        Args:
            depth: Number of convolutional layers
            channels: Number of feature channels
        """
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(1, channels, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        # Last layer (predict noise)
        layers.append(nn.Conv2d(channels, 1, 3, padding=1, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise via residual learning

        Args:
            x: Noisy input [B, 1, H, W]

        Returns:
            clean: Denoised output [B, 1, H, W]
        """
        noise = self.model(x)
        clean = x - noise
        return clean


class NAFNetDenoiser(nn.Module):
    """NAFNet-based denoiser head for stronger neural denoising."""

    def __init__(self, width: int = 16):
        super().__init__()
        self.net = NAFNetSmall(img_channel=1, width=width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShotNoiseCorrector(nn.Module):
    """
    Variance-stabilizing transform (VST) for Poisson shot noise

    Applies Anscombe transform, then Gaussian denoising, then inverse transform
    """

    def __init__(self, gaussian_denoiser: Optional[nn.Module] = None):
        """
        Args:
            gaussian_denoiser: Denoiser to apply in stabilized domain
                             If None, uses simple Gaussian blur
        """
        super().__init__()

        if gaussian_denoiser is None:
            # Simple Gaussian blur
            self.denoiser = lambda x: F.avg_pool2d(
                F.pad(x, (2, 2, 2, 2), mode='reflect'),
                kernel_size=5,
                stride=1
            )
        else:
            self.denoiser = gaussian_denoiser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denoise Poisson noise via VST

        Args:
            x: Noisy input (Poisson-distributed) [B, 1, H, W]

        Returns:
            clean: Denoised output [B, 1, H, W]
        """
        # Anscombe transform: stabilizes Poisson variance to ~1
        transformed = 2.0 * torch.sqrt(x + 3.0/8.0)

        # Denoise in stabilized domain
        denoised_transformed = self.denoiser(transformed)

        # Inverse Anscombe transform
        clean = (denoised_transformed / 2.0) ** 2 - 3.0/8.0
        clean = torch.clamp(clean, min=0.0)

        return clean


class ComponentDenoiserBank(nn.Module):
    """
    Convenience wrapper for all component denoisers
    """

    def __init__(
        self,
        speckle_iterations: int = 10,
        banding_adaptive: bool = True,
        gaussian_depth: int = 5,  # Changed from 7 to 5 to match checkpoint
        gaussian_type: str = "dncnn",
        shot_type: str = "vst",
        nafnet_width: int = 16,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.device = device

        # Register denoisers as submodules so they're included in state_dict
        self.speckle_denoiser = SpeckleDenoiser(
            iterations=speckle_iterations
        ).to(device)

        self.banding_denoiser = BandingRemover(
            adaptive=banding_adaptive
        ).to(device)

        if gaussian_type == "nafnet":
            self.gaussian_denoiser = NAFNetDenoiser(width=nafnet_width).to(device)
        else:
            self.gaussian_denoiser = GaussianDenoiser(depth=gaussian_depth).to(device)

        if shot_type == "nafnet":
            self.shot_denoiser = NAFNetDenoiser(width=nafnet_width).to(device)
        elif shot_type == "vst_nafnet":
            self.shot_denoiser = ShotNoiseCorrector(
                gaussian_denoiser=NAFNetDenoiser(width=nafnet_width).to(device)
            ).to(device)
        else:
            self.shot_denoiser = ShotNoiseCorrector().to(device)

        # Keep dict interface for compatibility
        self.denoisers = {
            'speckle': self.speckle_denoiser,
            'banding': self.banding_denoiser,
            'gaussian': self.gaussian_denoiser,
            'shot': self.shot_denoiser,
        }

    def denoise(self, x: torch.Tensor, component: str) -> torch.Tensor:
        """
        Apply specific component denoiser

        Args:
            x: Input image [B, 1, H, W]
            component: One of {'speckle', 'banding', 'gaussian', 'shot'}

        Returns:
            denoised: Component-denoised image
        """
        if component not in self.denoisers:
            raise ValueError(f"Unknown component: {component}")

        return self.denoisers[component](x)

    def denoise_all(self, x: torch.Tensor) -> dict:
        """
        Apply all component denoisers

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            denoised_dict: Dictionary of component-denoised images (clamped to [0,1])
        """
        return {
            component: torch.clamp(denoiser(x), 0, 1)
            for component, denoiser in self.denoisers.items()
        }

    def get_denoiser(self, component: str) -> nn.Module:
        """Get specific denoiser module"""
        return self.denoisers[component]
