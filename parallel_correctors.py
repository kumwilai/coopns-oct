#!/usr/bin/env python3
"""
Parallel Predicate-Specific Correctors with Masked Loss.

Each predicate failure type has a specialized correction architecture:
- SpeckleCorrector: Local intensity/noise statistics adjustment
- AnatomyCorrector: Column-wise boundary refinement
- StructureCorrector: Gradient-domain edge enhancement

Training and evaluation use masked loss - only measure improvement
in regions where correction is applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CorrectorConfig:
    """Configuration for parallel correctors."""
    # Speckle corrector
    speckle_channels: int = 32
    speckle_kernel: int = 5  # Local statistics window

    # Anatomy corrector
    anatomy_channels: int = 32
    anatomy_depth: int = 3  # Column-wise layers

    # Structure corrector
    structure_channels: int = 32
    structure_scales: int = 3  # Multi-scale gradients

    # Combination
    correction_scale: float = 0.1
    min_failure_threshold: float = 0.01  # Skip correction if failure too small


class SpeckleCorrector(nn.Module):
    """
    Corrector for speckle fidelity failures.

    Specialization: Local intensity statistics adjustment.
    - Uses local mean/std computation
    - Adjusts residual noise level to match expected CV
    """

    def __init__(self, config: CorrectorConfig):
        super().__init__()
        ch = config.speckle_channels
        k = config.speckle_kernel

        # Local statistics estimation
        self.local_mean = nn.Conv2d(1, 1, k, padding=k//2, bias=False)
        self.local_mean.weight.data.fill_(1.0 / (k * k))
        self.local_mean.weight.requires_grad = False

        # Intensity-aware correction
        self.intensity_encoder = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),  # [denoised, local_mean, local_std]
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Failure-guided refinement
        self.failure_guided = nn.Sequential(
            nn.Conv2d(ch + 1, ch, 3, padding=1),  # +1 for failure map
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output residual
        self.to_residual = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    def forward(self, denoised: torch.Tensor, failure_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            denoised: [B, 1, H, W] current denoised image
            failure_map: [B, 1, H, W] speckle failure regions

        Returns:
            residual: [B, 1, H, W] correction to add
        """
        # Skip if failure region is negligible (memory optimization)
        if failure_map.sum() < 1.0:
            return torch.zeros_like(denoised)

        # Compute local statistics
        local_mean = self.local_mean(denoised)
        local_sq_mean = self.local_mean(denoised ** 2)
        local_std = torch.sqrt(torch.clamp(local_sq_mean - local_mean ** 2, min=1e-8))
        del local_sq_mean  # Free intermediate

        # Encode intensity context
        intensity_features = torch.cat([denoised, local_mean, local_std], dim=1)
        del local_mean, local_std
        feat = self.intensity_encoder(intensity_features)
        del intensity_features

        # Guide by failure map
        feat_guided = torch.cat([feat, failure_map], dim=1)
        del feat
        feat_refined = self.failure_guided(feat_guided)
        del feat_guided

        # Generate residual (masked by failure)
        residual = self.to_residual(feat_refined)
        residual = residual * failure_map  # Only correct where failure exists

        return residual


class AnatomyCorrector(nn.Module):
    """
    Corrector for anatomy validity failures.

    Specialization: Column-wise boundary refinement.
    - Processes each A-scan (column) with shared weights
    - Respects vertical layer structure of OCT
    - Uses 1D convolutions along depth axis
    """

    def __init__(self, config: CorrectorConfig):
        super().__init__()
        ch = config.anatomy_channels
        depth = config.anatomy_depth

        # Column-wise encoder (1D along height)
        self.column_encoder = nn.ModuleList()
        self.column_encoder.append(nn.Conv1d(1, ch, 3, padding=1))
        for _ in range(depth - 1):
            self.column_encoder.append(nn.Conv1d(ch, ch, 3, padding=1))

        # Failure integration (2D to capture spatial context of failure)
        self.failure_integration = nn.Sequential(
            nn.Conv2d(ch + 1, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Column-wise decoder
        self.column_decoder = nn.ModuleList()
        for _ in range(depth - 1):
            self.column_decoder.append(nn.Conv1d(ch, ch, 3, padding=1))
        self.column_decoder.append(nn.Conv1d(ch, 1, 3, padding=1))

        # Initialize last layer to zero
        nn.init.zeros_(self.column_decoder[-1].weight)
        nn.init.zeros_(self.column_decoder[-1].bias)

    def forward(self, denoised: torch.Tensor, failure_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            denoised: [B, 1, H, W] current denoised image
            failure_map: [B, 1, H, W] anatomy failure regions

        Returns:
            residual: [B, 1, H, W] correction to add
        """
        B, C, H, W = denoised.shape

        # Memory-efficient: use 2D conv with kernel (H, 1) instead of reshaping to B*W tensors
        # This avoids creating B*W separate gradient paths

        # Skip if failure region is negligible (memory optimization)
        if failure_map.sum() < 1.0:
            return torch.zeros_like(denoised)

        # Process columns: [B, 1, H, W] -> [B*W, 1, H]
        # Use contiguous() to avoid memory fragmentation
        x = denoised.permute(0, 3, 1, 2).contiguous().view(B * W, 1, H)

        # Column-wise encoding
        for layer in self.column_encoder:
            x = F.relu(layer(x), inplace=True)

        # Reshape back to 2D: [B*W, ch, H] -> [B, ch, H, W]
        ch = x.shape[1]
        x = x.view(B, W, ch, H).permute(0, 2, 3, 1).contiguous()

        # Integrate failure map (2D context)
        x = torch.cat([x, failure_map], dim=1)
        x = self.failure_integration(x)

        # Reshape for column-wise decoding: [B, ch, H, W] -> [B*W, ch, H]
        ch = x.shape[1]
        x = x.permute(0, 3, 1, 2).contiguous().view(B * W, ch, H)

        # Column-wise decoding
        for i, layer in enumerate(self.column_decoder):
            if i < len(self.column_decoder) - 1:
                x = F.relu(layer(x), inplace=True)
            else:
                x = layer(x)

        # Reshape: [B*W, 1, H] -> [B, 1, H, W]
        residual = x.view(B, W, 1, H).permute(0, 2, 3, 1).contiguous()

        # Mask by failure
        residual = residual * failure_map

        return residual


class StructureCorrector(nn.Module):
    """
    Corrector for structure preservation failures.

    Specialization: Gradient-domain edge enhancement.
    - Operates in gradient domain (Sobel edges)
    - Multi-scale processing for different edge frequencies
    - Preserves edges that exist in noisy input
    """

    def __init__(self, config: CorrectorConfig):
        super().__init__()
        ch = config.structure_channels
        scales = config.structure_scales

        # Sobel filters (fixed)
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 4.0)

        # Multi-scale gradient encoder
        self.scale_encoders = nn.ModuleList()
        for s in range(scales):
            self.scale_encoders.append(nn.Sequential(
                nn.Conv2d(4, ch, 3, padding=1),  # [grad_x_noisy, grad_y_noisy, grad_x_den, grad_y_den]
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.ReLU(inplace=True),
            ))

        # Failure-guided fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * scales + 1, ch, 3, padding=1),  # +1 for failure map
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Edge residual generator
        self.to_residual = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )
        nn.init.zeros_(self.to_residual[-1].weight)
        nn.init.zeros_(self.to_residual[-1].bias)

    def compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Sobel gradients."""
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                failure_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            denoised: [B, 1, H, W] current denoised image
            noisy: [B, 1, H, W] original noisy input
            failure_map: [B, 1, H, W] structure failure regions

        Returns:
            residual: [B, 1, H, W] correction to add
        """
        # Skip if failure region is negligible (memory optimization)
        if failure_map.sum() < 1.0:
            return torch.zeros_like(denoised)

        # Multi-scale processing with memory optimization
        scale_features = []
        current_noisy = noisy
        current_den = denoised

        for s, encoder in enumerate(self.scale_encoders):
            # Get gradients at current scale
            gx_n, gy_n = self.compute_gradients(current_noisy)
            gx_d, gy_d = self.compute_gradients(current_den)

            # Encode gradient differences
            grad_input = torch.cat([gx_n, gy_n, gx_d, gy_d], dim=1)
            feat = encoder(grad_input)

            # Free intermediate gradient tensors
            del gx_n, gy_n, gx_d, gy_d, grad_input

            # Upsample if needed
            if s > 0:
                feat = F.interpolate(feat, size=denoised.shape[2:], mode='bilinear', align_corners=False)

            scale_features.append(feat)

            # Downsample for next scale
            if s < len(self.scale_encoders) - 1:
                current_noisy = F.avg_pool2d(current_noisy, 2)
                current_den = F.avg_pool2d(current_den, 2)

        # Fuse scales with failure map
        multi_scale = torch.cat(scale_features + [failure_map], dim=1)
        del scale_features  # Free list
        fused = self.fusion(multi_scale)
        del multi_scale

        # Generate residual
        residual = self.to_residual(fused)
        residual = residual * failure_map  # Only correct failure regions

        return residual


class ParallelCorrectors(nn.Module):
    """
    Parallel predicate-specific correctors with masked loss.

    All three correctors run in parallel and their outputs are combined.
    Training uses masked loss - only measuring improvement in failure regions.
    """

    def __init__(self, config: Optional[CorrectorConfig] = None):
        super().__init__()
        self.config = config or CorrectorConfig()

        # Specialized correctors
        self.speckle_corrector = SpeckleCorrector(self.config)
        self.anatomy_corrector = AnatomyCorrector(self.config)
        self.structure_corrector = StructureCorrector(self.config)

        # Learnable combination weights
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Speckle weight
        self.beta = nn.Parameter(torch.tensor(1.0))   # Anatomy weight
        self.gamma = nn.Parameter(torch.tensor(1.0)) # Structure weight

        self.scale = self.config.correction_scale

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                failure_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply parallel corrections.

        Args:
            denoised: [B, 1, H, W] current denoised image
            noisy: [B, 1, H, W] original noisy input
            failure_maps: Dict with 'speckle', 'anatomy', 'structure' failure maps

        Returns:
            Dict with:
                - 'corrected': final corrected image
                - 'residual_speckle': speckle correction
                - 'residual_anatomy': anatomy correction
                - 'residual_structure': structure correction
                - 'combined_residual': weighted sum of all residuals
        """
        # Get failure maps
        speckle_fail = failure_maps.get('speckle', torch.zeros_like(denoised))
        anatomy_fail = failure_maps.get('anatomy', torch.zeros_like(denoised))
        structure_fail = failure_maps.get('structure', torch.zeros_like(denoised))

        # Parallel corrections
        r_speckle = self.speckle_corrector(denoised, speckle_fail)
        r_anatomy = self.anatomy_corrector(denoised, anatomy_fail)
        r_structure = self.structure_corrector(denoised, noisy, structure_fail)

        # Weighted combination
        combined = (
            self.alpha * r_speckle +
            self.beta * r_anatomy +
            self.gamma * r_structure
        ) * self.scale

        # Apply correction
        corrected = denoised + combined
        corrected = torch.clamp(corrected, 0, 1)

        return {
            'corrected': corrected,
            'residual_speckle': r_speckle,
            'residual_anatomy': r_anatomy,
            'residual_structure': r_structure,
            'combined_residual': combined,
        }


class MaskedLoss(nn.Module):
    """
    Masked loss for predicate-specific correction.

    Only measures improvement in regions where correction was applied.
    Provides separate metrics for each predicate type.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                failure_maps: Dict[str, torch.Tensor],
                residuals: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute masked losses.

        Args:
            pred: [B, 1, H, W] corrected prediction
            target: [B, 1, H, W] ground truth
            failure_maps: Dict with failure maps for each predicate
            residuals: Dict with residuals from each corrector

        Returns:
            Dict with:
                - 'total': total masked loss
                - 'speckle': loss in speckle failure regions
                - 'anatomy': loss in anatomy failure regions
                - 'structure': loss in structure failure regions
                - 'psnr_speckle': PSNR in speckle regions
                - 'psnr_anatomy': PSNR in anatomy regions
                - 'psnr_structure': PSNR in structure regions
        """
        losses = {}

        # Compute loss for each predicate region
        for name in ['speckle', 'anatomy', 'structure']:
            mask = failure_maps.get(name, torch.zeros_like(pred))
            mask_sum = mask.sum() + self.eps

            # MSE in masked region
            squared_error = (pred - target) ** 2
            masked_mse = (squared_error * mask).sum() / mask_sum
            losses[name] = masked_mse

            # PSNR in masked region
            if mask.sum() > 0:
                psnr = 10 * torch.log10(1.0 / (masked_mse + self.eps))
                losses[f'psnr_{name}'] = psnr
            else:
                losses[f'psnr_{name}'] = torch.tensor(float('inf'))

        # Total loss (weighted by failure area)
        total_mask = (
            failure_maps.get('speckle', torch.zeros_like(pred)) +
            failure_maps.get('anatomy', torch.zeros_like(pred)) +
            failure_maps.get('structure', torch.zeros_like(pred))
        ).clamp(0, 1)

        total_mask_sum = total_mask.sum() + self.eps
        total_mse = ((pred - target) ** 2 * total_mask).sum() / total_mask_sum
        losses['total'] = total_mse
        losses['total_psnr'] = 10 * torch.log10(1.0 / (total_mse + self.eps))

        # Also compute global metrics for comparison
        global_mse = F.mse_loss(pred, target)
        losses['global'] = global_mse
        losses['global_psnr'] = 10 * torch.log10(1.0 / (global_mse + self.eps))

        return losses


class MaskedMetrics:
    """
    Evaluation metrics computed only on failure regions.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'speckle': {'mse_sum': 0, 'count': 0},
            'anatomy': {'mse_sum': 0, 'count': 0},
            'structure': {'mse_sum': 0, 'count': 0},
            'total': {'mse_sum': 0, 'count': 0},
            'global': {'mse_sum': 0, 'count': 0},
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               failure_maps: Dict[str, torch.Tensor]):
        """Update metrics with a batch."""
        with torch.no_grad():
            for name in ['speckle', 'anatomy', 'structure']:
                mask = failure_maps.get(name, torch.zeros_like(pred))
                if mask.sum() > 0:
                    masked_mse = ((pred - target) ** 2 * mask).sum()
                    self.metrics[name]['mse_sum'] += masked_mse.item()
                    self.metrics[name]['count'] += mask.sum().item()

            # Total masked region
            total_mask = sum(
                failure_maps.get(n, torch.zeros_like(pred))
                for n in ['speckle', 'anatomy', 'structure']
            ).clamp(0, 1)

            if total_mask.sum() > 0:
                masked_mse = ((pred - target) ** 2 * total_mask).sum()
                self.metrics['total']['mse_sum'] += masked_mse.item()
                self.metrics['total']['count'] += total_mask.sum().item()

            # Global
            global_mse = ((pred - target) ** 2).sum()
            self.metrics['global']['mse_sum'] += global_mse.item()
            self.metrics['global']['count'] += pred.numel()

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        results = {}
        for name, data in self.metrics.items():
            if data['count'] > 0:
                mse = data['mse_sum'] / data['count']
                psnr = 10 * np.log10(1.0 / (mse + 1e-8))
                results[f'{name}_psnr'] = psnr
                results[f'{name}_mse'] = mse
            else:
                results[f'{name}_psnr'] = float('inf')
                results[f'{name}_mse'] = 0.0
        return results


# For numpy operations in MaskedMetrics
import numpy as np


if __name__ == "__main__":
    # Test the correctors
    print("Testing Parallel Correctors with Masked Loss")
    print("=" * 60)

    device = torch.device('cpu')
    B, H, W = 2, 256, 256

    # Create test inputs
    noisy = torch.rand(B, 1, H, W)
    denoised = torch.rand(B, 1, H, W)
    target = torch.rand(B, 1, H, W)

    # Create sample failure maps
    failure_maps = {
        'speckle': (torch.rand(B, 1, H, W) > 0.7).float(),
        'anatomy': (torch.rand(B, 1, H, W) > 0.8).float(),
        'structure': (torch.rand(B, 1, H, W) > 0.75).float(),
    }

    print(f"\nInput shapes:")
    print(f"  Noisy: {noisy.shape}")
    print(f"  Denoised: {denoised.shape}")
    print(f"  Failure maps: speckle={failure_maps['speckle'].sum():.0f}px, "
          f"anatomy={failure_maps['anatomy'].sum():.0f}px, "
          f"structure={failure_maps['structure'].sum():.0f}px")

    # Create correctors
    config = CorrectorConfig()
    correctors = ParallelCorrectors(config)
    masked_loss = MaskedLoss()

    print(f"\nModel parameters: {sum(p.numel() for p in correctors.parameters()):,}")

    # Forward pass
    print("\nRunning parallel correction...")
    outputs = correctors(denoised, noisy, failure_maps)

    print(f"\nOutput shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")

    # Compute masked loss
    losses = masked_loss(outputs['corrected'], target, failure_maps, outputs)

    print(f"\nMasked Losses:")
    for k, v in losses.items():
        if 'psnr' in k:
            print(f"  {k}: {v.item():.2f} dB")
        else:
            print(f"  {k}: {v.item():.6f}")

    # Test metrics
    print("\n" + "=" * 60)
    print("Testing MaskedMetrics for evaluation")

    metrics = MaskedMetrics()
    metrics.update(outputs['corrected'], target, failure_maps)
    results = metrics.compute()

    print("\nEvaluation Metrics (only in failure regions):")
    for k, v in results.items():
        if 'psnr' in k:
            print(f"  {k}: {v:.2f} dB")

    print("\n" + "=" * 60)
    print("SUCCESS: Parallel correctors with masked loss working!")
