#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising with Parallel Correctors.

Complete pipeline:
1. NAFNet backbone for initial denoising
2. Symbolic predicates evaluate quality and generate failure maps
3. Parallel correctors fix each failure type with specialized architecture
4. Masked loss trains each corrector only on its failure regions
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import os
import sys

# Import our modules
from parallel_correctors import ParallelCorrectors, CorrectorConfig, MaskedLoss, MaskedMetrics


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Predicates
    speckle_cv: float = 0.40
    speckle_tolerance: float = 0.14
    structure_threshold: float = 0.5
    min_layer_thickness: float = 0.03

    # Correctors
    corrector_config: CorrectorConfig = None

    # Pipeline
    max_iterations: int = 3
    convergence_threshold: float = 0.001

    def __post_init__(self):
        if self.corrector_config is None:
            self.corrector_config = CorrectorConfig()


class SpecklePredicate(nn.Module):
    """Speckle fidelity predicate with spatial failure map."""

    def __init__(self, expected_cv: float = 0.40, tolerance: float = 0.14, window: int = 15):
        super().__init__()
        self.expected_cv = expected_cv
        self.tolerance = tolerance
        self.window = window

        # Local statistics filters
        self.register_buffer('box_filter',
            torch.ones(1, 1, window, window) / (window * window))

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate speckle fidelity.

        Returns:
            Dict with 'satisfied', 'loss', 'failure_map', 'cv_map'
        """
        residual = noisy - denoised

        # Local CV computation
        local_mean = F.conv2d(denoised, self.box_filter, padding=self.window//2)
        local_mean = torch.clamp(local_mean, min=1e-6)

        res_sq = residual ** 2
        local_var = F.conv2d(res_sq, self.box_filter, padding=self.window//2)
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))

        cv_map = local_std / local_mean

        # Failure map: where CV deviates from expected
        cv_error = torch.abs(cv_map - self.expected_cv)
        failure_map = (cv_error > self.tolerance).float()

        # Soft failure map for gradient
        soft_failure = torch.sigmoid((cv_error - self.tolerance) * 10)

        # Global metrics
        cv_mean = cv_map.mean()
        satisfied = (torch.abs(cv_mean - self.expected_cv) < self.tolerance)
        loss = F.mse_loss(cv_map, torch.full_like(cv_map, self.expected_cv))

        return {
            'satisfied': satisfied,
            'loss': loss,
            'failure_map': soft_failure,
            'cv_map': cv_map,
            'cv_mean': cv_mean,
        }


class AnatomyPredicate(nn.Module):
    """Anatomy validity predicate with spatial failure map."""

    def __init__(self, num_boundaries: int = 4, min_thickness: float = 0.03):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.min_thickness = min_thickness

    def forward(self, boundaries: torch.Tensor, image_size: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate anatomy validity.

        Args:
            boundaries: [B, num_boundaries, W] normalized boundary positions
            image_size: (H, W) of the image for failure map

        Returns:
            Dict with 'satisfied', 'loss', 'failure_map'
        """
        B, N, W_bound = boundaries.shape

        # Check ordering violations
        ordering_violations = torch.zeros(B, 1, W_bound, device=boundaries.device)
        thickness_violations = torch.zeros(B, 1, W_bound, device=boundaries.device)

        for i in range(N - 1):
            # Ordering: boundary[i] should be < boundary[i+1]
            order_fail = (boundaries[:, i, :] >= boundaries[:, i+1, :]).float()
            ordering_violations[:, 0, :] += order_fail

            # Thickness: gap should be >= min_thickness
            gap = boundaries[:, i+1, :] - boundaries[:, i, :]
            thick_fail = (gap < self.min_thickness).float()
            thickness_violations[:, 0, :] += thick_fail

        # Combine violations
        total_violations = ordering_violations + thickness_violations
        failure_map_1d = (total_violations > 0).float()

        # Expand to full image size
        if image_size is not None:
            H, W = image_size
            # Resize width if needed
            if W_bound != W:
                failure_map_1d = F.interpolate(
                    failure_map_1d.unsqueeze(1), size=(1, W), mode='nearest'
                ).squeeze(1)
            # Expand height
            failure_map_2d = failure_map_1d.unsqueeze(2).expand(B, 1, H, W)
        else:
            # Just expand height to match width
            failure_map_2d = failure_map_1d.unsqueeze(2).expand(B, 1, W_bound, W_bound)

        # Global satisfaction
        satisfied = (failure_map_1d.sum() == 0)
        loss = total_violations.mean()

        return {
            'satisfied': satisfied,
            'loss': loss,
            'failure_map': failure_map_2d,
            'ordering_violations': ordering_violations,
            'thickness_violations': thickness_violations,
        }


class StructurePredicate(nn.Module):
    """Structure preservation predicate with spatial failure map."""

    def __init__(self, threshold: float = 0.5, window: int = 15):
        super().__init__()
        self.threshold = threshold
        self.window = window

        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 4.0)

        # Local correlation filter
        self.register_buffer('box_filter',
            torch.ones(1, 1, window, window) / (window * window))

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate structure preservation via edge correlation.

        Returns:
            Dict with 'satisfied', 'loss', 'failure_map', 'correlation_map'
        """
        # Compute edges
        edge_noisy = self._compute_edges(noisy)
        edge_denoised = self._compute_edges(denoised)

        # Local correlation
        corr_map = self._local_correlation(edge_noisy, edge_denoised)

        # Failure map: where correlation is low
        failure_map = (corr_map < self.threshold).float()
        soft_failure = torch.sigmoid((self.threshold - corr_map) * 10)

        # Global metrics
        corr_mean = corr_map.mean()
        satisfied = (corr_mean > self.threshold)
        loss = F.relu(self.threshold - corr_map).mean()

        return {
            'satisfied': satisfied,
            'loss': loss,
            'failure_map': soft_failure,
            'correlation_map': corr_map,
            'correlation_mean': corr_mean,
        }

    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def _local_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute local correlation map."""
        # Local means
        x_mean = F.conv2d(x, self.box_filter, padding=self.window//2)
        y_mean = F.conv2d(y, self.box_filter, padding=self.window//2)

        # Local covariance
        xy_mean = F.conv2d(x * y, self.box_filter, padding=self.window//2)
        cov = xy_mean - x_mean * y_mean

        # Local variances
        x2_mean = F.conv2d(x**2, self.box_filter, padding=self.window//2)
        y2_mean = F.conv2d(y**2, self.box_filter, padding=self.window//2)
        var_x = x2_mean - x_mean**2
        var_y = y2_mean - y_mean**2

        # Correlation
        std_xy = torch.sqrt(torch.clamp(var_x * var_y, min=1e-8))
        corr = cov / std_xy
        return torch.clamp(corr, -1, 1)


class NeuroSymbolicParallel(nn.Module):
    """
    Complete neuro-symbolic pipeline with parallel correctors.

    Architecture:
        noisy -> backbone -> denoised -> predicates -> failure_maps
                                   |                        |
                                   v                        v
                              boundaries          parallel_correctors
                                                        |
                                                        v
                                                   corrected
                                                        |
                                              (iterate if needed)
    """

    def __init__(self, backbone: nn.Module, boundary_model: nn.Module,
                 config: Optional[PipelineConfig] = None):
        super().__init__()
        self.config = config or PipelineConfig()

        # Neural components
        self.backbone = backbone
        self.boundary_model = boundary_model

        # Symbolic predicates
        self.speckle_pred = SpecklePredicate(
            expected_cv=self.config.speckle_cv,
            tolerance=self.config.speckle_tolerance,
        )
        self.anatomy_pred = AnatomyPredicate(
            min_thickness=self.config.min_layer_thickness,
        )
        self.structure_pred = StructurePredicate(
            threshold=self.config.structure_threshold,
        )

        # Parallel correctors
        self.correctors = ParallelCorrectors(self.config.corrector_config)

        # Loss
        self.masked_loss = MaskedLoss()

    def get_boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """Extract boundaries from boundary model output."""
        with torch.no_grad():
            out = self.boundary_model(x)
            if isinstance(out, dict):
                boundaries = out.get('boundaries', out.get('boundary_positions'))
            else:
                boundaries = out
        return boundaries

    def evaluate_predicates(self, noisy: torch.Tensor, denoised: torch.Tensor,
                           boundaries: torch.Tensor) -> Dict:
        """Evaluate all predicates and get failure maps."""
        # Get image size
        B, C, H, W = denoised.shape

        # Predicates are for evaluation only - no gradients needed
        # This prevents OOM from building huge computation graphs
        with torch.no_grad():
            # Speckle
            speckle_result = self.speckle_pred(noisy, denoised)

            # Anatomy (pass image size for proper failure map)
            anatomy_result = self.anatomy_pred(boundaries, image_size=(H, W))

            # Structure
            structure_result = self.structure_pred(noisy, denoised)

            # Combine
            all_satisfied = (
                speckle_result['satisfied'] and
                anatomy_result['satisfied'] and
                structure_result['satisfied']
            )

            # Detach failure maps to prevent gradient flow through masks
            failure_maps = {
                'speckle': speckle_result['failure_map'].detach(),
                'anatomy': anatomy_result['failure_map'].detach(),
                'structure': structure_result['failure_map'].detach(),
            }

        return {
            'all_satisfied': all_satisfied,
            'failure_maps': failure_maps,
            'speckle': speckle_result,
            'anatomy': anatomy_result,
            'structure': structure_result,
        }

    def forward(self, noisy: torch.Tensor, return_details: bool = False) -> Dict:
        """
        Forward pass with iterative refinement.

        Args:
            noisy: [B, 1, H, W] noisy input
            return_details: if True, return intermediate results

        Returns:
            Dict with 'denoised', 'iterations', 'all_satisfied', etc.
        """
        # Initial denoising
        denoised = self.backbone(noisy)
        denoised = torch.clamp(denoised, 0, 1)

        # Get boundaries
        boundaries = self.get_boundaries(denoised)

        # Track iterations
        iteration_history = []

        for iteration in range(self.config.max_iterations):
            # Evaluate predicates
            pred_result = self.evaluate_predicates(noisy, denoised, boundaries)

            # Record state
            iteration_history.append({
                'iteration': iteration,
                'all_satisfied': pred_result['all_satisfied'].item() if isinstance(pred_result['all_satisfied'], torch.Tensor) else pred_result['all_satisfied'],
                'speckle_cv': pred_result['speckle']['cv_mean'].item(),
                'structure_corr': pred_result['structure']['correlation_mean'].item(),
            })

            # Check convergence
            if pred_result['all_satisfied']:
                break

            # Apply parallel corrections
            correction_result = self.correctors(
                denoised, noisy, pred_result['failure_maps']
            )
            denoised = correction_result['corrected']

            # MEMORY FIX: Delete intermediate results to prevent accumulation across iterations
            del pred_result, correction_result

            # Update boundaries for next iteration
            boundaries = self.get_boundaries(denoised)

        # MEMORY FIX: Collect garbage after iteration loop
        gc.collect()

        # Final evaluation
        final_result = self.evaluate_predicates(noisy, denoised, boundaries)

        output = {
            'denoised': denoised,
            'iterations': iteration + 1,
            'all_satisfied': final_result['all_satisfied'],
            'failure_maps': final_result['failure_maps'],
        }

        if return_details:
            output['history'] = iteration_history
            output['speckle'] = final_result['speckle']
            output['anatomy'] = final_result['anatomy']
            output['structure'] = final_result['structure']

        return output

    def compute_loss(self, noisy: torch.Tensor, target: torch.Tensor) -> Dict:
        """
        Compute masked loss for training.

        Returns separate losses for each predicate type,
        measured only in failure regions.
        """
        # Initial denoising
        denoised = self.backbone(noisy)
        denoised = torch.clamp(denoised, 0, 1)

        # Get boundaries
        boundaries = self.get_boundaries(denoised)

        # Evaluate predicates to get failure maps
        pred_result = self.evaluate_predicates(noisy, denoised, boundaries)

        # Apply corrections
        correction_result = self.correctors(
            denoised, noisy, pred_result['failure_maps']
        )
        corrected = correction_result['corrected']

        # Compute masked losses
        losses = self.masked_loss(
            corrected, target,
            pred_result['failure_maps'],
            correction_result,
        )

        # Add predicate losses for regularization
        losses['pred_speckle'] = pred_result['speckle']['loss']
        losses['pred_anatomy'] = pred_result['anatomy']['loss']
        losses['pred_structure'] = pred_result['structure']['loss']

        # Total training loss
        losses['train_total'] = (
            losses['total'] +
            0.1 * losses['pred_speckle'] +
            0.1 * losses['pred_anatomy'] +
            0.1 * losses['pred_structure']
        )

        return losses, corrected, pred_result['failure_maps']


def test_pipeline():
    """Test the full pipeline."""
    print("=" * 70)
    print("NEURO-SYMBOLIC OCT DENOISING WITH PARALLEL CORRECTORS")
    print("=" * 70)

    device = torch.device('cpu')

    # Load models
    print("\nLoading models...")

    # Try to load NAFNet
    try:
        sys.path.insert(0, 'nsnd_oct')
        from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
        backbone = NAFNet(
            img_channel=1,
            width=64,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2],
            dec_blk_nums=[2, 2, 2],
        )
        ckpt_path = "/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            backbone.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"  NAFNet loaded: PSNR={ckpt.get('psnr', 'unknown')}")
    except Exception as e:
        print(f"  NAFNet not available: {e}")
        # Simple fallback
        backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    # Load boundary model
    try:
        from physics_enhanced_v3 import PhysicsEnsembleV3
        boundary_model = PhysicsEnsembleV3(
            in_channels=1,
            hidden_channels=48,
            num_boundaries=4,
        )
        ckpt_path = "/home/kumwilai/OCT/best_boundary_model_v4.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            boundary_model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            print("  Boundary model loaded")
    except Exception as e:
        print(f"  Boundary model not available: {e}")
        boundary_model = None

    if boundary_model is None:
        # Create dummy boundary model
        class DummyBoundary(nn.Module):
            def forward(self, x):
                B, C, H, W = x.shape
                # Return dummy boundaries
                boundaries = torch.linspace(0.2, 0.8, 4).unsqueeze(0).unsqueeze(-1)
                boundaries = boundaries.expand(B, 4, W)
                return {'boundaries': boundaries}
        boundary_model = DummyBoundary()

    # Create pipeline
    config = PipelineConfig()
    pipeline = NeuroSymbolicParallel(backbone, boundary_model, config)
    pipeline.eval()

    print(f"\nPipeline parameters:")
    print(f"  Backbone: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"  Correctors: {sum(p.numel() for p in pipeline.correctors.parameters()):,}")

    # Test with sample
    print("\n" + "-" * 70)
    print("Testing on sample image...")

    # Load a sample
    from pathlib import Path
    import json
    from PIL import Image

    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    val_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    if val_jsonl.exists():
        with open(val_jsonl) as f:
            data = json.loads(f.readline())

        clean = Image.open(data['clean_path']).convert('L')
        noisy = Image.open(data['noisy_path']).convert('L')

        clean = torch.from_numpy(np.array(clean)).float() / 255.0
        noisy = torch.from_numpy(np.array(noisy)).float() / 255.0

        clean = clean.unsqueeze(0).unsqueeze(0)
        noisy = noisy.unsqueeze(0).unsqueeze(0)
    else:
        # Random test
        noisy = torch.rand(1, 1, 256, 256)
        clean = torch.rand(1, 1, 256, 256)

    print(f"  Input shape: {noisy.shape}")

    # Forward pass
    with torch.no_grad():
        output = pipeline(noisy, return_details=True)

    print(f"\nResults:")
    print(f"  Iterations: {output['iterations']}")
    print(f"  All satisfied: {output['all_satisfied']}")

    # PSNR
    mse = F.mse_loss(output['denoised'], clean)
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"  PSNR: {psnr.item():.2f} dB")

    # Failure region stats
    print(f"\nFailure regions:")
    for name, fmap in output['failure_maps'].items():
        pct = 100 * fmap.mean().item()
        print(f"  {name}: {pct:.1f}% of image")

    # Iteration history
    print(f"\nIteration history:")
    for h in output['history']:
        print(f"  Iter {h['iteration']}: satisfied={h['all_satisfied']}, "
              f"CV={h['speckle_cv']:.3f}, corr={h['structure_corr']:.3f}")

    # Free memory before training test
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test training mode
    print("\n" + "-" * 70)
    print("Testing training mode with masked loss...")

    pipeline.train()
    # Use smaller image for training test to avoid OOM
    noisy_small = F.interpolate(noisy, size=(128, 128), mode='bilinear', align_corners=False)
    clean_small = F.interpolate(clean, size=(128, 128), mode='bilinear', align_corners=False)
    losses, corrected, failure_maps = pipeline.compute_loss(noisy_small, clean_small)

    print(f"\nMasked Training Losses:")
    for name, val in losses.items():
        if isinstance(val, torch.Tensor):
            print(f"  {name}: {val.item():.6f}")

    print(f"\nFailure coverage:")
    for name, fmap in failure_maps.items():
        print(f"  {name}: {100*fmap.mean().item():.1f}%")

    print("\n" + "=" * 70)
    print("SUCCESS: Pipeline with parallel correctors working!")
    print("=" * 70)


if __name__ == "__main__":
    test_pipeline()
