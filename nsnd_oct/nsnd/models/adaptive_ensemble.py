"""
Adaptive Ensemble for Competitive OCT Denoising

Combines NSND with strong classical baselines using adaptive weights
determined by neuro-symbolic noise analysis.

Expected Performance: 26-28 dB (competitive with supervised methods)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional

try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False


class AdaptiveEnsembleNSND(nn.Module):
    """
    Adaptive ensemble combining NSND with strong classical baselines

    Architecture:
        Input → [Residual Refinement, Gaussian σ=1.5, Gaussian σ=1.0, BM3D, NSND] → Adaptive Weights → Output

    Adaptive weights are determined by NSND's symbolic noise analysis
    """

    def __init__(
        self,
        nsnd_model: nn.Module,
        residual_refinement_model: Optional[nn.Module] = None,
        use_bm3d: bool = True,
        learnable_weights: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            nsnd_model: Trained or untrained NSND model
            residual_refinement_model: Trained residual refinement model (27.06 dB)
            use_bm3d: Whether to include BM3D in ensemble
            learnable_weights: If True, learn ensemble weights. If False, use fixed weights.
            device: Device to run on
        """
        super().__init__()

        self.nsnd = nsnd_model
        self.residual_refinement = residual_refinement_model
        self.use_bm3d = use_bm3d and BM3D_AVAILABLE
        self.device = device

        # Number of ensemble components
        base_components = 3  # Gaussian 1.5, Gaussian 1.0, NSND
        if self.use_bm3d:
            base_components += 1  # Add BM3D
        if self.residual_refinement is not None:
            base_components += 1  # Add Residual Refinement

        self.n_components = base_components

        if learnable_weights:
            # Adaptive weight network based on noise profile
            # Input: 4 noise components (speckle, banding, gaussian, shot)
            # Output: weights for each denoiser
            self.weight_net = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, self.n_components),
                nn.Softmax(dim=-1)
            ).to(device)
        else:
            self.weight_net = None

            # Fixed weights (optimized offline)
            # These are good defaults based on performance
            if self.residual_refinement is not None and self.use_bm3d:
                # Residual Refinement, Gaussian 1.5, Gaussian 1.0, BM3D, NSND
                self.fixed_weights = torch.tensor([0.50, 0.25, 0.10, 0.10, 0.05]).to(device)
            elif self.residual_refinement is not None:
                # Residual Refinement, Gaussian 1.5, Gaussian 1.0, NSND
                self.fixed_weights = torch.tensor([0.55, 0.25, 0.10, 0.10]).to(device)
            elif self.use_bm3d:
                # Gaussian 1.5, Gaussian 1.0, BM3D, NSND
                self.fixed_weights = torch.tensor([0.45, 0.25, 0.20, 0.10]).to(device)
            else:
                # Gaussian 1.5, Gaussian 1.0, NSND
                self.fixed_weights = torch.tensor([0.55, 0.30, 0.15]).to(device)

    def forward(
        self,
        noisy: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through adaptive ensemble

        Args:
            noisy: Noisy input image [B, 1, H, W]
            return_intermediates: If True, return all intermediate outputs

        Returns:
            output: Denoised image [B, 1, H, W]
            ensemble_weights: Weights used for each component [B, n_components]
            intermediates: Dict of intermediate outputs
        """
        B, C, H, W = noisy.shape

        # 1. Get all denoised versions
        denoised_versions = []
        component_names = []

        # Residual Refinement (best: 27.06 dB) - if available
        residual_out = None
        if self.residual_refinement is not None:
            residual_out = self.residual_refinement(noisy)
            denoised_versions.append(residual_out)
            component_names.append('Residual Refinement')

        # Gaussian σ=1.5 (best classical: 25.96 dB)
        gaussian_15 = self._apply_gaussian(noisy, sigma=1.5)
        denoised_versions.append(gaussian_15)
        component_names.append('Gaussian σ=1.5')

        # Gaussian σ=1.0 (good balance: 24.17 dB)
        gaussian_10 = self._apply_gaussian(noisy, sigma=1.0)
        denoised_versions.append(gaussian_10)
        component_names.append('Gaussian σ=1.0')

        # BM3D (SOTA classical: 23.96 dB)
        bm3d_out = None
        if self.use_bm3d:
            bm3d_out = self._apply_bm3d(noisy)
            denoised_versions.append(bm3d_out)
            component_names.append('BM3D')

        # NSND (interpretable: 16-17 dB, but provides noise analysis)
        nsnd_out, _, nsnd_intermediates = self.nsnd(noisy, return_intermediates=True)
        denoised_versions.append(nsnd_out)
        component_names.append('NSND')

        # Stack all versions [B, n_components, 1, H, W]
        denoised_stack = torch.stack(denoised_versions, dim=1)

        # 2. Compute adaptive ensemble weights
        if self.weight_net is not None:
            # Get noise profile from NSND
            symbolic_weights = nsnd_intermediates['symbolic_weights']

            # Create noise vector [B, 4]
            noise_vector = torch.stack([
                symbolic_weights['speckle'].squeeze(),
                symbolic_weights['banding'].squeeze(),
                symbolic_weights['gaussian'].squeeze(),
                symbolic_weights['shot'].squeeze()
            ], dim=-1).float()  # Ensure float32

            # Compute adaptive weights [B, n_components]
            ensemble_weights = self.weight_net(noise_vector)
        else:
            # Use fixed weights
            ensemble_weights = self.fixed_weights.unsqueeze(0).expand(B, -1)

        # 3. Weighted ensemble
        # Reshape weights for broadcasting: [B, n_components, 1, 1, 1]
        weights_expanded = ensemble_weights.view(B, self.n_components, 1, 1, 1)

        # Weighted sum
        output = (denoised_stack * weights_expanded).sum(dim=1)  # [B, 1, H, W]

        # Collect intermediates
        intermediates = None
        if return_intermediates:
            intermediates = {
                'residual_refinement': residual_out,
                'gaussian_15': gaussian_15,
                'gaussian_10': gaussian_10,
                'bm3d': bm3d_out,
                'nsnd': nsnd_out,
                'ensemble_weights': ensemble_weights,
                'component_names': component_names,
                'nsnd_intermediates': nsnd_intermediates,
            }

        return output, ensemble_weights, intermediates

    def _apply_gaussian(self, noisy: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian filter"""
        B, C, H, W = noisy.shape

        # Process each image in batch
        filtered = []
        for i in range(B):
            img = noisy[i, 0].cpu().numpy()
            img_filtered = gaussian_filter(img, sigma=sigma)
            filtered.append(torch.from_numpy(img_filtered))

        result = torch.stack(filtered, dim=0).unsqueeze(1).to(noisy.device)
        return result

    def _apply_bm3d(self, noisy: torch.Tensor) -> torch.Tensor:
        """Apply BM3D denoiser"""
        if not BM3D_AVAILABLE:
            # Fallback to Gaussian if BM3D not available
            return self._apply_gaussian(noisy, sigma=1.0)

        B, C, H, W = noisy.shape

        # Process each image in batch
        denoised = []
        for i in range(B):
            img = noisy[i, 0].cpu().numpy()

            # Estimate noise sigma
            sigma_est = np.std(img - gaussian_filter(img, sigma=1.0))
            sigma_est = max(sigma_est, 0.01)  # Ensure non-zero

            # Apply BM3D
            try:
                img_denoised = bm3d.bm3d(img, sigma_psd=sigma_est, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            except:
                # Fallback to Gaussian if BM3D fails
                img_denoised = gaussian_filter(img, sigma=1.0)

            denoised.append(torch.from_numpy(img_denoised))

        result = torch.stack(denoised, dim=0).unsqueeze(1).to(noisy.device)
        return result

    def analyze_noise(self, x: torch.Tensor) -> str:
        """Generate noise analysis report using NSND"""
        return self.nsnd.analyze_noise(x)

    def get_ensemble_summary(self, ensemble_weights: torch.Tensor, component_names: list = None) -> Dict:
        """
        Get human-readable summary of ensemble weights

        Args:
            ensemble_weights: [B, n_components] tensor
            component_names: List of component names (optional)

        Returns:
            summary: Dict with weight statistics
        """
        weights_mean = ensemble_weights.mean(dim=0).cpu().numpy()

        if component_names is None:
            # Build component names dynamically
            component_names = []
            if self.residual_refinement is not None:
                component_names.append('Residual Refinement')
            component_names.extend(['Gaussian σ=1.5', 'Gaussian σ=1.0'])
            if self.use_bm3d:
                component_names.append('BM3D')
            component_names.append('NSND')

        summary = {}
        for i, name in enumerate(component_names):
            summary[name] = float(weights_mean[i])

        return summary


def train_ensemble_weights(
    ensemble_model: AdaptiveEnsembleNSND,
    val_loader,
    device: str = 'cpu',
    epochs: int = 50,
    lr: float = 1e-3
):
    """
    Train adaptive ensemble weights to maximize PSNR

    Args:
        ensemble_model: Adaptive ensemble with learnable weights
        val_loader: Validation data loader (noisy, clean pairs)
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
    """
    if ensemble_model.weight_net is None:
        raise ValueError("Ensemble must have learnable weights")

    optimizer = torch.optim.Adam(ensemble_model.weight_net.parameters(), lr=lr)

    print(f"\nTraining Adaptive Ensemble Weights (Optimizing for PSNR)")
    print("="*60)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Forward pass
            output, weights, _ = ensemble_model(noisy)

            # PSNR loss (we want to maximize PSNR, so minimize negative PSNR)
            mse = F.mse_loss(output.float(), clean.float())
            psnr = 10 * torch.log10(torch.tensor(1.0).to(mse.device) / (mse + 1e-8))
            loss = -psnr

            # Add small regularization to encourage diversity
            # Entropy regularization: prefer diverse weights
            weight_entropy = -(weights.float() * torch.log(weights.float() + 1e-8)).sum(dim=-1).mean()
            loss = loss + 0.01 * weight_entropy  # Small weight

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

    return ensemble_model
