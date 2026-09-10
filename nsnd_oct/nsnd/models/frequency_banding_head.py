"""
Learnable Frequency-Domain Banding Removal Head

NOVEL CONTRIBUTION:
- Combines FFT with learnable neural networks for OCT banding removal
- Learns to identify and suppress banding frequencies adaptively
- Frequency-domain processing preserves spatial structure better than convolutions

Key Innovation:
Instead of fixed notch filters or spatial convolutions, we learn:
1. Which frequencies correspond to banding artifacts
2. How much to suppress each frequency
3. Adaptive per-image frequency filtering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableFrequencyBandingHead(nn.Module):
    """
    Learnable frequency-domain banding removal

    Architecture:
    1. FFT → Frequency domain
    2. Learn frequency importance weights (which frequencies are banding?)
    3. Learn suppression strength (how much to remove?)
    4. IFFT → Spatial domain
    5. Residual refinement

    This is NOVEL because:
    - Existing methods use fixed notch filters (manual frequency selection)
    - CNNs learn spatial patterns but not frequency patterns
    - We learn WHICH frequencies to suppress and BY HOW MUCH
    """

    def __init__(self, channels=16, num_freq_bins=32):
        super().__init__()

        self.num_freq_bins = num_freq_bins

        # Frequency analyzer: Learn which frequencies are important
        # Input: Frequency spectrum features → Output: Frequency weights
        self.freq_analyzer = nn.Sequential(
            nn.Linear(num_freq_bins * 2, 64),  # *2 for horizontal + vertical
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_freq_bins),
            nn.Sigmoid()  # Frequency suppression weights [0, 1]
        )

        # Spatial refinement after frequency filtering
        # Learns to combine frequency-filtered result with input
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),  # 2 channels: filtered + original
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: Residual from Gaussian baseline [B, 1, H, W]
        Returns:
            refined_residual: Denoised residual [B, 1, H, W]
        """
        B, C, H, W = residual.shape

        # Step 1: Analyze frequency spectrum to identify banding
        freq_weights = self._analyze_frequencies(residual)  # [B, num_freq_bins]

        # Step 2: Apply frequency-domain filtering
        filtered = self._frequency_filter(residual, freq_weights)  # [B, 1, H, W]

        # Step 3: Spatial refinement (combine filtered + original)
        combined = torch.cat([filtered, residual], dim=1)  # [B, 2, H, W]
        refined = self.spatial_refine(combined)  # [B, 1, H, W]

        return refined

    def _analyze_frequencies(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analyze frequency spectrum to learn which frequencies to suppress

        Args:
            x: Input [B, 1, H, W]
        Returns:
            freq_weights: Suppression weights [B, num_freq_bins]
        """
        B, C, H, W = x.shape

        # FFT (Real FFT since input is real-valued)
        fft = torch.fft.rfft2(x, dim=(-2, -1))  # [B, 1, H, W//2+1] complex
        fft_mag = torch.abs(fft)  # [B, 1, H, W//2+1]

        # Extract frequency profiles (horizontal and vertical)
        # Horizontal profile: Average over vertical frequencies
        h_profile = fft_mag.mean(dim=2).squeeze(1)  # [B, W//2+1]

        # Vertical profile: Average over horizontal frequencies
        v_profile = fft_mag.mean(dim=3).squeeze(1)  # [B, H]

        # Bin into num_freq_bins for learnable analysis
        h_binned = self._bin_frequencies(h_profile, self.num_freq_bins)  # [B, num_freq_bins]
        v_binned = self._bin_frequencies(v_profile, self.num_freq_bins)  # [B, num_freq_bins]

        # Concatenate horizontal + vertical
        freq_features = torch.cat([h_binned, v_binned], dim=1)  # [B, num_freq_bins*2]

        # Learn which frequencies to suppress
        freq_weights = self.freq_analyzer(freq_features)  # [B, num_freq_bins]

        return freq_weights

    def _bin_frequencies(self, profile: torch.Tensor, num_bins: int) -> torch.Tensor:
        """
        Bin frequency profile into num_bins bins

        Args:
            profile: [B, F] frequency profile
            num_bins: Number of bins
        Returns:
            binned: [B, num_bins]
        """
        B, F = profile.shape
        bin_size = F // num_bins

        binned = []
        for i in range(num_bins):
            start = i * bin_size
            end = start + bin_size if i < num_bins - 1 else F
            bin_avg = profile[:, start:end].mean(dim=1)
            binned.append(bin_avg)

        return torch.stack(binned, dim=1)  # [B, num_bins]

    def _frequency_filter(self, x: torch.Tensor, freq_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply learned frequency-domain filtering

        Args:
            x: Input [B, 1, H, W]
            freq_weights: Frequency suppression weights [B, num_freq_bins]
        Returns:
            filtered: Frequency-filtered output [B, 1, H, W]
        """
        B, C, H, W = x.shape

        # FFT
        fft = torch.fft.rfft2(x, dim=(-2, -1))  # [B, 1, H, W//2+1] complex

        # Create frequency mask from learned weights
        # Map num_freq_bins weights to full frequency space
        freq_mask = self._create_frequency_mask(freq_weights, H, W)  # [B, 1, H, W//2+1]

        # Apply mask (suppress banding frequencies)
        # freq_weights close to 0 → suppress, close to 1 → keep
        fft_filtered = fft * freq_mask

        # IFFT back to spatial domain
        filtered = torch.fft.irfft2(fft_filtered, s=(H, W), dim=(-2, -1))  # [B, 1, H, W]

        return filtered

    def _create_frequency_mask(self, freq_weights: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Create 2D frequency mask from 1D frequency weights

        Args:
            freq_weights: [B, num_freq_bins]
            H, W: Spatial dimensions
        Returns:
            mask: [B, 1, H, W//2+1]
        """
        B = freq_weights.shape[0]
        num_bins = self.num_freq_bins

        # Create frequency grid
        freq_y = torch.fft.fftfreq(H, device=freq_weights.device)  # [H]
        freq_x = torch.fft.rfftfreq(W, device=freq_weights.device)  # [W//2+1]

        # Create 2D grid
        freq_y = freq_y.view(H, 1).expand(H, W//2+1)  # [H, W//2+1]
        freq_x = freq_x.view(1, W//2+1).expand(H, W//2+1)  # [H, W//2+1]

        # Radial frequency
        freq_radial = torch.sqrt(freq_y**2 + freq_x**2)  # [H, W//2+1]

        # Normalize to [0, num_bins-1]
        freq_radial_normalized = freq_radial / freq_radial.max() * (num_bins - 1)

        # Create mask for each batch item (avoid in-place operations)
        batch_masks = []

        for b in range(B):
            bin_contributions = []
            for i in range(num_bins):
                # Find frequencies in this bin
                in_bin = ((freq_radial_normalized >= i) & (freq_radial_normalized < i + 1)).float()
                # Weight by learned freq_weight
                contribution = in_bin * freq_weights[b, i]
                bin_contributions.append(contribution)

            # Sum contributions
            batch_mask = torch.stack(bin_contributions, dim=0).sum(dim=0)  # [H, W//2+1]
            batch_masks.append(batch_mask.unsqueeze(0))  # [1, H, W//2+1]

        # Stack all batches
        masks = torch.stack(batch_masks, dim=0)  # [B, 1, H, W//2+1]

        # Normalize
        masks = masks.clamp(0, 1)

        return masks

    def visualize_learned_frequencies(self, residual: torch.Tensor) -> dict:
        """
        Visualize what the network learned about banding frequencies

        Args:
            residual: Input residual [B, 1, H, W]
        Returns:
            viz: Dictionary with visualization data
        """
        with torch.no_grad():
            freq_weights = self._analyze_frequencies(residual)

            # Get frequency spectrum
            fft = torch.fft.rfft2(residual, dim=(-2, -1))
            fft_mag = torch.abs(fft)

            return {
                'freq_weights': freq_weights.cpu().numpy(),  # [B, num_freq_bins]
                'fft_magnitude': fft_mag.cpu().numpy(),      # [B, 1, H, W//2+1]
                'suppressed_freqs': (freq_weights < 0.5).cpu().numpy()  # Which freqs are suppressed
            }


class HybridBandingHead(nn.Module):
    """
    Hybrid approach: Frequency-domain + Spatial refinement

    Combines:
    1. Learnable frequency-domain filtering (removes periodic banding)
    2. Spatial convolutions (handles non-periodic artifacts)

    This gives best of both worlds:
    - Frequency domain: Perfect for periodic patterns (banding)
    - Spatial domain: Good for local patterns (texture, edges)
    """

    def __init__(self, channels=16, num_freq_bins=32):
        super().__init__()

        # Frequency-domain path
        self.freq_path = LearnableFrequencyBandingHead(channels, num_freq_bins)

        # Spatial-domain path (original BandingResidualHead style)
        self.spatial_path = nn.Sequential(
            # Directional filters
            nn.Conv2d(1, channels//2, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, channels//2, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),  # freq + spatial
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: Residual from Gaussian baseline [B, 1, H, W]
        Returns:
            refined_residual: Denoised residual [B, 1, H, W]
        """
        # Frequency path
        freq_refined = self.freq_path(residual)  # [B, 1, H, W]

        # Spatial path
        h_feat = self.spatial_path[0](residual)  # Horizontal
        v_feat = self.spatial_path[2](residual)  # Vertical
        spatial_feat = torch.cat([h_feat, v_feat], dim=1)  # [B, channels, H, W]

        # Fuse frequency + spatial
        combined = torch.cat([freq_refined, spatial_feat], dim=1)
        refined = self.fusion(combined)

        return refined
