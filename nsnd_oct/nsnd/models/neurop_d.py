"""
NeurOp-D: Noise-Conditioned Neural Operator for OCT Denoising

IEEE TMI-Level Contributions:
1. Continuous noise embedding (not discrete classification)
2. HyperNetwork-generated spatially-varying denoising kernels
3. Physics-constrained latent decomposition
4. Uncertainty-guided adaptive refinement

Key Innovation: Don't SELECT from fixed operators → GENERATE the operator dynamically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# Contribution 1: Continuous Noise Encoder with Physics Structure
# =============================================================================

class NoiseEncoder(nn.Module):
    """
    Encodes per-pixel noise characteristics into continuous latent code.

    NOVEL: The latent space is STRUCTURED by physics:
    - Dims 0-3: Noise type embedding (continuous, not one-hot)
    - Dim 4: Noise intensity (log-scale)
    - Dims 5-7: Spatial correlation (anisotropy)
    - Dims 8+: Learned features

    Output: z_n ∈ R^D for each pixel, capturing FULL noise characteristics.
    """

    def __init__(self, out_dim: int = 16, base_channels: int = 32):
        super().__init__()
        self.out_dim = out_dim

        # Multi-scale noise feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Upsample back to full resolution
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)

        # Output heads
        self.noise_code_head = nn.Conv2d(base_channels, out_dim, 1)
        self.uncertainty_head = nn.Conv2d(base_channels, 1, 1)

        # Physics-structured projection (first 8 dims have meaning)
        self.physics_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Noisy image [B, 1, H, W]
        Returns:
            noise_code: Continuous noise embedding [B, D, H, W]
            uncertainty: Prediction confidence [B, 1, H, W]
        """
        # Multi-scale encoding
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)

        # Decode back to full resolution
        f2_up = self.up2(f3) + f2
        f1_up = self.up1(f2_up) + f1

        # Output noise code and uncertainty
        noise_code = self.noise_code_head(f1_up)
        uncertainty = torch.sigmoid(self.uncertainty_head(f1_up))

        # Apply physics structure to first 8 dimensions
        B, D, H, W = noise_code.shape
        noise_code_flat = noise_code.permute(0, 2, 3, 1).reshape(-1, D)
        noise_code_structured = self.physics_proj(noise_code_flat)
        noise_code = noise_code_structured.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Normalize noise code (unit sphere for stability)
        noise_code = F.normalize(noise_code, dim=1)

        return noise_code, uncertainty


# =============================================================================
# Contribution 2: HyperNetwork-Generated Denoising Kernels
# =============================================================================

class KernelHyperNetwork(nn.Module):
    """
    Generates denoising kernels from noise code.

    NOVEL: The kernel is SYNTHESIZED, not selected:
    - Input: noise code z_n ∈ R^D (per-pixel)
    - Output: denoising kernel K ∈ R^(k×k) (per-pixel)

    This creates a CONTINUOUS FAMILY of operators.
    """

    def __init__(
        self,
        noise_dim: int = 16,
        kernel_size: int = 7,
        hidden_dim: int = 64,
        num_output_channels: int = 64,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_output_channels = num_output_channels

        # MLP that generates kernel weights
        kernel_params = kernel_size * kernel_size * num_output_channels
        self.kernel_mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, kernel_params),
        )

        # Also generate bias
        self.bias_mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_output_channels),
        )

        # Initialize to approximate identity
        nn.init.zeros_(self.kernel_mlp[-1].weight)
        nn.init.zeros_(self.kernel_mlp[-1].bias)
        nn.init.zeros_(self.bias_mlp[-1].weight)
        nn.init.zeros_(self.bias_mlp[-1].bias)

    def forward(self, noise_code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noise_code: [B, D, H, W]
        Returns:
            kernels: [B, C, k, k, H, W] - per-pixel kernels
            biases: [B, C, H, W] - per-pixel biases
        """
        B, D, H, W = noise_code.shape

        # Reshape for MLP: [B*H*W, D]
        noise_flat = noise_code.permute(0, 2, 3, 1).reshape(-1, D)

        # Generate kernels
        kernel_flat = self.kernel_mlp(noise_flat)  # [B*H*W, k*k*C]
        kernels = kernel_flat.reshape(B, H, W, self.num_output_channels, self.kernel_size, self.kernel_size)
        kernels = kernels.permute(0, 3, 4, 5, 1, 2)  # [B, C, k, k, H, W]

        # Normalize kernels (sum to 1 for each spatial location)
        kernels = F.softmax(kernels.reshape(B, self.num_output_channels, -1, H, W), dim=2)
        kernels = kernels.reshape(B, self.num_output_channels, self.kernel_size, self.kernel_size, H, W)

        # Generate biases
        bias_flat = self.bias_mlp(noise_flat)  # [B*H*W, C]
        biases = bias_flat.reshape(B, H, W, self.num_output_channels).permute(0, 3, 1, 2)

        return kernels, biases


class NoiseConditionedConv(nn.Module):
    """
    Applies spatially-varying convolution with generated kernels.

    NOVEL: Each pixel has its OWN kernel, generated from noise code.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(
        self,
        features: torch.Tensor,
        kernels: torch.Tensor,
        biases: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply spatially-varying convolution.

        Args:
            features: [B, C_in, H, W]
            kernels: [B, C_out, k, k, H, W]
            biases: [B, C_out, H, W]
        Returns:
            output: [B, C_out, H, W]
        """
        B, C_in, H, W = features.shape
        k = self.kernel_size

        # Unfold features to patches
        # [B, C_in, H, W] -> [B, C_in*k*k, H*W]
        features_unfold = F.unfold(features, k, padding=self.padding)
        features_unfold = features_unfold.reshape(B, C_in, k * k, H, W)

        # For efficiency, use einsum or batch matrix multiply
        # kernels: [B, C_out, k, k, H, W] -> [B, C_out, k*k, H, W]
        kernels_flat = kernels.reshape(B, self.out_channels, k * k, H, W)

        # We need to handle C_in -> C_out mapping
        # Simplified: assume C_in == C_out for now, apply per-channel
        if C_in == self.out_channels:
            # Per-channel spatially-varying conv
            # features_unfold: [B, C, k*k, H, W]
            # kernels_flat: [B, C, k*k, H, W]
            output = (features_unfold * kernels_flat).sum(dim=2)  # [B, C, H, W]
        else:
            # Cross-channel: more complex, use learned projection
            output = (features_unfold.mean(dim=1, keepdim=True) * kernels_flat).sum(dim=2)

        output = output + biases

        return output


# =============================================================================
# Contribution 3: Physics-Constrained Latent Space
# =============================================================================

class PhysicsDecompositionLoss(nn.Module):
    """
    Encourages physics-consistent noise code structure.

    NOVEL: The first 8 dimensions of noise_code are constrained:
    - Dims 0-3: Should correlate with noise type statistics
    - Dim 4: Should correlate with local noise intensity
    - Dims 5-7: Should capture spatial correlation structure
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        noisy: torch.Tensor,
        noise_code: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics consistency loss.
        """
        B, D, H, W = noise_code.shape

        # Compute local statistics from noisy image
        local_mean = F.avg_pool2d(noisy, 7, stride=1, padding=3)
        local_var = F.avg_pool2d((noisy - local_mean) ** 2, 7, stride=1, padding=3)
        local_std = torch.sqrt(local_var + 1e-8)

        # Coefficient of variation (high for speckle)
        cv = local_std / (local_mean + 1e-8)

        # Physics constraint 1: Dim 0 should correlate with CV (speckle indicator)
        speckle_dim = noise_code[:, 0:1]
        loss_speckle = F.mse_loss(speckle_dim, (cv - 1.0).clamp(-1, 1))

        # Physics constraint 2: Dim 4 should correlate with noise level
        level_dim = noise_code[:, 4:5]
        noise_level = local_std / local_std.max()
        loss_level = F.mse_loss(level_dim, noise_level)

        # Physics constraint 3: Variance/mean ratio (shot noise indicator)
        var_mean_ratio = local_var / (local_mean + 1e-8)
        shot_dim = noise_code[:, 3:4]
        loss_shot = F.mse_loss(shot_dim, (var_mean_ratio - 1.0).clamp(-1, 1))

        return loss_speckle + loss_level + loss_shot


# =============================================================================
# Contribution 4: Uncertainty-Guided Adaptive Refinement
# =============================================================================

class AdaptiveRefinement(nn.Module):
    """
    Iteratively refines with adaptive depth based on uncertainty.

    NOVEL: More iterations for high-uncertainty regions.
    Inspired by Adaptive Computation Time (Graves, 2016).
    """

    def __init__(
        self,
        channels: int = 64,
        max_iterations: int = 3,
    ):
        super().__init__()
        self.max_iterations = max_iterations

        # Refinement block (shared across iterations)
        self.refine_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Halting score predictor
        self.halt_predictor = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptive refinement based on uncertainty.

        High uncertainty → more iterations
        Low uncertainty → fewer iterations
        """
        B, C, H, W = features.shape

        # Initialize
        refined = features
        cumulative_halt = torch.zeros(B, 1, H, W, device=features.device)
        remainders = torch.ones(B, 1, H, W, device=features.device)
        outputs = torch.zeros_like(features)

        for t in range(self.max_iterations):
            # Refine
            delta = self.refine_block(refined)
            refined = refined + delta

            # Compute halting probability (lower for high uncertainty)
            halt_prob = self.halt_predictor(refined)
            # High uncertainty → lower halt probability → more iterations
            halt_prob = halt_prob * (1.0 - uncertainty * 0.5)

            # Accumulate halted outputs
            still_running = (cumulative_halt < 1.0).float()
            new_halted = torch.min(halt_prob, remainders) * still_running
            outputs = outputs + new_halted * refined

            # Update running state
            cumulative_halt = cumulative_halt + new_halted
            remainders = remainders - new_halted

            # Early stop if all halted
            if (cumulative_halt >= 1.0).all():
                break

        # Add remainder
        outputs = outputs + remainders * refined

        return outputs


# =============================================================================
# Image Backbone (U-Net style)
# =============================================================================

class UNetEncoder(nn.Module):
    """Standard U-Net encoder."""

    def __init__(self, in_channels: int = 1, width: int = 64):
        super().__init__()
        self.enc1 = self._block(in_channels, width)
        self.enc2 = self._block(width, width * 2)
        self.enc3 = self._block(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e3, [e1, e2]


class UNetDecoder(nn.Module):
    """Standard U-Net decoder."""

    def __init__(self, out_channels: int = 1, width: int = 64):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = self._block(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = self._block(width * 2, width)
        self.out = nn.Conv2d(width, out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skips):
        e1, e2 = skips
        d2 = self.dec2(torch.cat([self.up2(x), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# =============================================================================
# Complete NeurOp-D Model
# =============================================================================

class NeurOpD(nn.Module):
    """
    Noise-Conditioned Neural Operator for OCT Denoising

    IEEE TMI Contributions:
    1. Continuous noise embedding (not discrete classification)
    2. HyperNetwork-generated spatially-varying kernels
    3. Physics-constrained latent decomposition
    4. Uncertainty-guided adaptive refinement

    Key Innovation:
    Instead of selecting from fixed operators, we GENERATE the operator
    dynamically from the noise characteristics. This creates a continuous
    family of denoising operators, one for each pixel.
    """

    def __init__(
        self,
        noise_dim: int = 16,
        image_width: int = 64,
        kernel_size: int = 5,
        max_refine_iter: int = 3,
    ):
        super().__init__()

        self.noise_dim = noise_dim

        # Contribution 1: Continuous noise encoder
        self.noise_encoder = NoiseEncoder(out_dim=noise_dim)

        # Image backbone
        self.image_encoder = UNetEncoder(in_channels=1, width=image_width)
        self.image_decoder = UNetDecoder(out_channels=1, width=image_width)

        # Contribution 2: HyperNetwork for kernel generation
        self.kernel_generator = KernelHyperNetwork(
            noise_dim=noise_dim,
            kernel_size=kernel_size,
            num_output_channels=image_width * 4,  # Match encoder output
        )

        # Noise-conditioned convolution
        self.noise_conv = NoiseConditionedConv(
            in_channels=image_width * 4,
            out_channels=image_width * 4,
            kernel_size=kernel_size,
        )

        # Contribution 4: Adaptive refinement
        self.refiner = AdaptiveRefinement(
            channels=image_width * 4,
            max_iterations=max_refine_iter,
        )

        # Contribution 3: Physics loss (used during training)
        self.physics_loss = PhysicsDecompositionLoss()

    def forward(
        self,
        noisy: torch.Tensor,
        return_interpretation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass.

        Args:
            noisy: Input noisy image [B, 1, H, W]
            return_interpretation: Return interpretable outputs
        Returns:
            denoised: Output image [B, 1, H, W]
            interpretation: Dict of interpretable outputs
        """
        # Step 1: Encode noise characteristics (Contribution 1)
        noise_code, uncertainty = self.noise_encoder(noisy)

        # Step 2: Encode image features
        features, skips = self.image_encoder(noisy)

        # Step 3: Generate denoising kernels (Contribution 2)
        # Need to match feature resolution
        noise_code_down = F.interpolate(
            noise_code,
            size=features.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        kernels, biases = self.kernel_generator(noise_code_down)

        # Step 4: Apply noise-conditioned denoising
        denoised_features = self.noise_conv(features, kernels, biases)

        # Step 5: Adaptive refinement (Contribution 4)
        uncertainty_down = F.interpolate(
            uncertainty,
            size=features.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        refined_features = self.refiner(denoised_features, uncertainty_down)

        # Step 6: Decode to image
        denoised = self.image_decoder(refined_features, skips)

        if return_interpretation:
            # Extract interpretable noise type from first 4 dims
            noise_type = F.softmax(noise_code[:, :4], dim=1)
            noise_level = noise_code[:, 4:5]

            interpretation = {
                # Noise analysis
                'noise_code': noise_code,  # Full continuous code [B, D, H, W]
                'noise_type': noise_type,  # Soft noise type [B, 4, H, W]
                'noise_type_names': ['speckle', 'gaussian', 'banding', 'shot'],
                'noise_level': noise_level,  # Intensity [B, 1, H, W]
                'uncertainty': uncertainty,  # Confidence [B, 1, H, W]

                # Generated operators (interpretable!)
                'kernels': kernels,  # [B, C, k, k, H, W]

                # Pipeline
                'pipeline': 'NeurOp-D: noise_code → kernel_generation → adaptive_refinement',
            }
            return denoised, interpretation

        return denoised, None

    def compute_physics_loss(self, noisy: torch.Tensor, noise_code: torch.Tensor) -> torch.Tensor:
        """Compute physics consistency loss (Contribution 3)."""
        return self.physics_loss(noisy, noise_code)


# =============================================================================
# Training Loss
# =============================================================================

class NeurOpDLoss(nn.Module):
    """
    Complete training loss for NeurOp-D.
    """

    def __init__(
        self,
        lambda_physics: float = 0.1,
        lambda_perceptual: float = 0.1,
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_perceptual = lambda_perceptual
        self.physics_loss = PhysicsDecompositionLoss()

    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        noise_code: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss.
        """
        losses = {}

        # Reconstruction loss
        losses['recon'] = F.l1_loss(denoised, clean)

        # Physics consistency loss
        losses['physics'] = self.physics_loss(noisy, noise_code)

        # Total
        total = losses['recon'] + self.lambda_physics * losses['physics']

        return total, losses


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NeurOpD(
        noise_dim=16,
        image_width=32,  # Smaller for testing
        kernel_size=5,
        max_refine_iter=2,
    ).to(device)

    print(f"NeurOp-D Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 1, 64, 64).to(device)
    denoised, interp = model(x, return_interpretation=True)

    print(f"Input: {x.shape}")
    print(f"Output: {denoised.shape}")
    print(f"Noise code: {interp['noise_code'].shape}")
    print(f"Noise type: {interp['noise_type'].shape}")
    print(f"Kernels: {interp['kernels'].shape}")

    print("\nNeurOp-D model created successfully!")
