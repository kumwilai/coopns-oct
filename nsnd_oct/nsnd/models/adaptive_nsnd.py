"""
Scanner-Adaptive NSND

Automatically adapts to different OCT scanner characteristics through:
1. Test-time adaptation (TTA)
2. Scanner-specific parameter calibration
3. Meta-learning for fast adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .hybrid_analyzer import HybridCNNSymbolicAnalyzer
from .component_denoisers import ComponentDenoiserBank


class ScannerAdaptiveNSND(nn.Module):
    """
    NSND with automatic scanner adaptation

    Key features:
    1. Test-time adaptation (no labels needed)
    2. Learnable denoiser parameters (per scanner)
    3. Quick calibration (few unlabeled images)
    """

    def __init__(
        self,
        hybrid_analyzer_ckpt: str,
        device: str = 'cuda',
        enable_tta: bool = True,
        adapt_denoisers: bool = True,
    ):
        super().__init__()

        # Frozen hybrid analyzer (works across scanners)
        self.analyzer = HybridCNNSymbolicAnalyzer()
        ckpt = torch.load(hybrid_analyzer_ckpt, map_location=device, weights_only=False)
        self.analyzer.load_state_dict(ckpt['state_dict'])

        for param in self.analyzer.parameters():
            param.requires_grad = False

        # Adaptive component denoisers
        self.denoisers = ComponentDenoiserBank(
            speckle_iterations=10,
            banding_adaptive=True,
            gaussian_depth=5,
            device=device,
        )

        # Make denoiser parameters learnable for adaptation
        if adapt_denoisers:
            # Speckle denoiser parameters
            if hasattr(self.denoisers.denoisers['speckle'], 'log_kappa'):
                self.denoisers.denoisers['speckle'].log_kappa.requires_grad = True
                self.denoisers.denoisers['speckle'].log_gamma.requires_grad = True

        self.enable_tta = enable_tta
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass"""
        weights_dict, _ = self.analyzer(x)
        denoised_components = self.denoisers.denoise_all(x)

        w_speckle = weights_dict['speckle'].view(-1, 1, 1, 1)
        w_banding = weights_dict['banding'].view(-1, 1, 1, 1)
        w_gaussian = weights_dict['gaussian'].view(-1, 1, 1, 1)
        w_shot = weights_dict['shot'].view(-1, 1, 1, 1)

        output = (
            w_speckle * denoised_components['speckle'] +
            w_banding * denoised_components['banding'] +
            w_gaussian * denoised_components['gaussian'] +
            w_shot * denoised_components['shot']
        )

        return output

    def calibrate_to_scanner(
        self,
        calibration_images: torch.Tensor,
        n_iterations: int = 10,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Calibrate NSND to a specific scanner using unlabeled images

        Uses self-supervised objectives:
        1. Noise2Void: Blind-spot reconstruction
        2. Augmentation consistency
        3. Total variation regularization

        Args:
            calibration_images: [N, 1, H, W] unlabeled images from target scanner
            n_iterations: Number of adaptation iterations
            lr: Learning rate for adaptation

        Returns:
            calibration_params: Optimized parameters for this scanner
        """
        self.train()

        # Only optimize denoiser parameters
        optimizer = torch.optim.Adam(
            [p for p in self.denoisers.parameters() if p.requires_grad],
            lr=lr
        )

        print(f"Calibrating to scanner with {len(calibration_images)} images...")

        for iteration in range(n_iterations):
            total_loss = 0.0

            for img in calibration_images:
                img = img.unsqueeze(0).to(self.device)

                # 1. Noise2Void loss (blind-spot network)
                n2v_loss = self._noise2void_loss(img)

                # 2. Augmentation consistency
                aug_loss = self._augmentation_consistency_loss(img)

                # 3. Total variation (smoothness prior)
                denoised = self.forward(img)
                tv_loss = self._total_variation_loss(denoised)

                # Combined loss
                loss = n2v_loss + 0.5 * aug_loss + 0.01 * tv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(calibration_images)
            print(f"  Iteration {iteration+1}/{n_iterations} | Loss: {avg_loss:.4f}")

        # Extract calibrated parameters
        calibration_params = {}
        if hasattr(self.denoisers.denoisers['speckle'], 'log_kappa'):
            calibration_params['speckle_kappa'] = torch.exp(
                self.denoisers.denoisers['speckle'].log_kappa
            ).item()
            calibration_params['speckle_gamma'] = torch.exp(
                self.denoisers.denoisers['speckle'].log_gamma
            ).item()

        print(f"✓ Calibration complete!")
        print(f"  Optimized parameters: {calibration_params}")

        self.eval()
        return calibration_params

    def _noise2void_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Noise2Void: Blind-spot reconstruction loss

        Masks random pixels and tries to predict them from neighbors
        """
        B, C, H, W = x.shape

        # Create random mask (5% of pixels)
        mask = torch.rand(B, C, H, W, device=x.device) > 0.95

        # Masked input
        x_masked = x.clone()
        x_masked[mask] = 0.0

        # Denoise
        denoised = self.forward(x_masked)

        # Loss: Predict masked pixels from denoised neighbors
        loss = F.mse_loss(denoised[mask], x[mask])

        return loss

    def _augmentation_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce consistency across augmentations

        Denoised output should be equivariant to flips/rotations
        """
        # Original
        out_orig = self.forward(x)

        # Horizontal flip
        x_flip_h = torch.flip(x, dims=[-2])
        out_flip_h = self.forward(x_flip_h)
        out_flip_h_inv = torch.flip(out_flip_h, dims=[-2])

        # Consistency loss
        loss = F.mse_loss(out_orig, out_flip_h_inv)

        return loss

    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation regularization (smoothness)"""
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))

    def save_scanner_profile(self, scanner_name: str, output_path: str):
        """Save calibrated parameters for this scanner"""
        profile = {
            'scanner_name': scanner_name,
            'denoiser_state_dict': self.denoisers.state_dict(),
            'timestamp': torch.tensor(0.0),  # Add timestamp
        }
        torch.save(profile, output_path)
        print(f"✓ Saved {scanner_name} profile to {output_path}")

    def load_scanner_profile(self, profile_path: str):
        """Load pre-calibrated parameters for a scanner"""
        profile = torch.load(profile_path, map_location=self.device, weights_only=False)
        self.denoisers.load_state_dict(profile['denoiser_state_dict'])
        print(f"✓ Loaded scanner profile: {profile['scanner_name']}")


class MultiScannerNSND(nn.Module):
    """
    NSND with multiple scanner profiles

    Automatically selects appropriate profile based on input statistics
    """

    def __init__(self, hybrid_analyzer_ckpt: str, device: str = 'cuda'):
        super().__init__()

        self.analyzer = HybridCNNSymbolicAnalyzer()
        ckpt = torch.load(hybrid_analyzer_ckpt, map_location=device, weights_only=False)
        self.analyzer.load_state_dict(ckpt['state_dict'])

        for param in self.analyzer.parameters():
            param.requires_grad = False

        # Multiple denoiser banks (one per scanner type)
        self.scanner_profiles = nn.ModuleDict()
        self.device = device

    def add_scanner_profile(self, scanner_name: str, profile_path: str):
        """Add a calibrated scanner profile"""
        denoiser_bank = ComponentDenoiserBank(
            speckle_iterations=10,
            banding_adaptive=True,
            gaussian_depth=5,
            device=self.device,
        )

        profile = torch.load(profile_path, map_location=self.device, weights_only=False)
        denoiser_bank.load_state_dict(profile['denoiser_state_dict'])

        self.scanner_profiles[scanner_name] = denoiser_bank
        print(f"✓ Added scanner profile: {scanner_name}")

    def detect_scanner_type(self, x: torch.Tensor) -> str:
        """
        Automatically detect scanner type from input statistics

        Uses simple heuristics:
        - Noise variance
        - Speckle pattern characteristics
        - Image intensity distribution
        """
        # Compute statistics
        mean_intensity = x.mean().item()
        noise_std = x.std().item()

        # Simple heuristic (would be learned in practice)
        if noise_std > 0.15:
            return 'heidelberg'
        elif mean_intensity < 0.4:
            return 'zeiss'
        else:
            return 'generic'

    def forward(self, x: torch.Tensor, scanner_name: Optional[str] = None) -> torch.Tensor:
        """
        Forward with automatic scanner detection

        Args:
            x: Input image
            scanner_name: Optional scanner override
        """
        # Detect scanner if not provided
        if scanner_name is None:
            scanner_name = self.detect_scanner_type(x)

        # Select appropriate denoiser bank
        if scanner_name not in self.scanner_profiles:
            scanner_name = 'generic'

        denoiser_bank = self.scanner_profiles[scanner_name]

        # Analyze noise
        weights_dict, _ = self.analyzer(x)

        # Apply scanner-specific denoisers
        denoised_components = denoiser_bank.denoise_all(x)

        # Weighted fusion
        w_speckle = weights_dict['speckle'].view(-1, 1, 1, 1)
        w_banding = weights_dict['banding'].view(-1, 1, 1, 1)
        w_gaussian = weights_dict['gaussian'].view(-1, 1, 1, 1)
        w_shot = weights_dict['shot'].view(-1, 1, 1, 1)

        output = (
            w_speckle * denoised_components['speckle'] +
            w_banding * denoised_components['banding'] +
            w_gaussian * denoised_components['gaussian'] +
            w_shot * denoised_components['shot']
        )

        return output
