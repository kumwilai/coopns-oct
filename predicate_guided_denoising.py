#!/usr/bin/env python3
"""
Predicate-Guided OCT Denoising

A closed-loop neuro-symbolic denoising system where symbolic predicates
actively guide the denoising process:

1. Initial denoising (backbone)
2. Evaluate predicates → get spatial failure maps
3. Apply targeted refinement where predicates fail
4. Iterate until all predicates pass

Key Innovation:
- Predicates are not just for verification, but for GUIDANCE
- Each predicate failure triggers a specific corrective loss
- Spatial maps show WHERE to refine, not just IF to refine

Author: Neuro-Symbolic OCT Denoising Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math


# =============================================================================
# SPATIAL FAILURE MAPS
# =============================================================================

class SpeckleFailureMap(nn.Module):
    """
    Compute spatial map showing WHERE speckle statistics are violated.

    Output: [B, 1, H, W] map where high values = CV deviation from expected
    """

    def __init__(
        self,
        expected_cv: float = 0.40,
        tolerance: float = 0.14,
        window_size: int = 16,
        min_intensity: float = 0.05,
    ):
        super().__init__()
        self.expected_cv = expected_cv
        self.tolerance = tolerance
        self.window_size = window_size
        self.min_intensity = min_intensity

        # Averaging kernel
        kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        self.register_buffer('avg_kernel', kernel)

    def forward(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute speckle failure map.

        Returns:
            failure_map: [B, 1, H, W] - high where CV deviates
            loss: scalar - differentiable loss to correct failures
            info: dict with statistics
        """
        B, C, H, W = noisy.shape
        residual = noisy - denoised

        # Compute local statistics
        pad = self.window_size // 2
        residual_pad = F.pad(residual, (pad, pad, pad, pad), mode='reflect')
        denoised_pad = F.pad(denoised, (pad, pad, pad, pad), mode='reflect')

        # Local variance of residual
        res_sq = residual_pad ** 2
        local_var = F.conv2d(res_sq, self.avg_kernel)
        local_mean_res = F.conv2d(residual_pad, self.avg_kernel)
        local_var = local_var - local_mean_res ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-8))

        # Local mean intensity
        local_intensity = F.conv2d(denoised_pad, self.avg_kernel)

        # Local CV
        local_cv = local_std / local_intensity.clamp(min=self.min_intensity)

        # Resize to match input
        local_cv = F.interpolate(local_cv, size=(H, W), mode='bilinear', align_corners=False)

        # Failure map: normalized deviation from expected CV
        cv_deviation = (local_cv - self.expected_cv).abs()
        failure_map = (cv_deviation / self.tolerance).clamp(0, 2)  # 0-2 range

        # Mask low-intensity regions
        intensity_mask = (denoised > self.min_intensity).float()
        failure_map = failure_map * intensity_mask

        # Compute loss: push CV toward expected value
        # Weight by failure severity
        loss = (cv_deviation * failure_map).mean()

        # Determine failure type
        cv_mean = local_cv.mean().item()
        if cv_mean < self.expected_cv - self.tolerance:
            failure_type = "under_denoised"  # CV too low, residual too small
        elif cv_mean > self.expected_cv + self.tolerance:
            failure_type = "over_denoised"   # CV too high, removed structure
        else:
            failure_type = "ok"

        info = {
            'cv_mean': cv_mean,
            'cv_expected': self.expected_cv,
            'failure_type': failure_type,
            'failure_ratio': (failure_map > 1.0).float().mean().item(),
        }

        return failure_map, loss, info


class AnatomyFailureMap(nn.Module):
    """
    Compute spatial map showing WHERE anatomical constraints are violated.

    Output: [B, 1, H, W] map where high values = boundary violations
    """

    def __init__(
        self,
        min_layer_thickness: float = 0.03,
        ilm_range: Tuple[float, float] = (0.05, 0.45),
        rpe_range: Tuple[float, float] = (0.40, 0.90),
        boundary_sigma: float = 10.0,
    ):
        super().__init__()
        self.min_layer_thickness = min_layer_thickness
        self.ilm_range = ilm_range
        self.rpe_range = rpe_range
        self.boundary_sigma = boundary_sigma

    def forward(
        self,
        boundaries: torch.Tensor,
        height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute anatomy failure map.

        Args:
            boundaries: [B, N, W] boundary positions (normalized 0-1)
            height: image height

        Returns:
            failure_map: [B, 1, H, W] - high near invalid boundaries
            loss: scalar - differentiable loss to correct
            info: dict with statistics
        """
        B, N, W = boundaries.shape
        device = boundaries.device

        # Check ordering violations: b[i+1] - b[i] should be > 0
        diffs = boundaries[:, 1:, :] - boundaries[:, :-1, :]  # [B, N-1, W]
        ordering_violations = F.relu(-diffs + 0.001)  # Positive where violated

        # Check thickness violations
        thickness_violations = F.relu(self.min_layer_thickness - diffs)

        # Check position violations
        ilm = boundaries[:, 0, :]  # [B, W]
        rpe = boundaries[:, -1, :]  # [B, W]

        ilm_violations = F.relu(self.ilm_range[0] - ilm) + F.relu(ilm - self.ilm_range[1])
        rpe_violations = F.relu(self.rpe_range[0] - rpe) + F.relu(rpe - self.rpe_range[1])

        # Combine violations per column
        total_violations = (
            ordering_violations.sum(dim=1) +  # [B, W]
            thickness_violations.sum(dim=1) +
            ilm_violations +
            rpe_violations
        )  # [B, W]

        # Create spatial failure map
        # High values near boundaries that have violations
        rows = torch.arange(height, device=device, dtype=torch.float32)
        rows = rows.view(1, 1, -1, 1)  # [1, 1, H, 1]

        boundaries_px = boundaries * (height - 1)  # [B, N, W]
        boundaries_px = boundaries_px.unsqueeze(2)  # [B, N, 1, W]

        # Distance to each boundary
        distances = (rows - boundaries_px).abs()  # [B, N, H, W]

        # Gaussian weight around boundaries
        weights = torch.exp(-distances ** 2 / (2 * self.boundary_sigma ** 2))

        # Weight by violations (expand violations to match)
        # For each boundary, use the violation of the layer below it
        layer_violations = torch.cat([
            ordering_violations,
            thickness_violations[:, -1:, :]  # Last layer
        ], dim=1)  # [B, N, W]

        layer_violations = layer_violations.unsqueeze(2)  # [B, N, 1, W]

        # Weighted failure map
        failure_map = (weights * layer_violations).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Normalize
        failure_map = failure_map / (failure_map.max() + 1e-8)

        # Loss: minimize all violations
        loss = total_violations.mean()

        info = {
            'ordering_violations': ordering_violations.sum().item(),
            'thickness_violations': thickness_violations.sum().item(),
            'position_violations': (ilm_violations + rpe_violations).sum().item(),
            'total_violations': total_violations.sum().item(),
        }

        return failure_map, loss, info


class StructureFailureMap(nn.Module):
    """
    Compute spatial map showing WHERE structure (edges) were lost.

    Output: [B, 1, H, W] map where high values = edges lost during denoising
    """

    def __init__(
        self,
        edge_threshold: float = 0.5,
    ):
        super().__init__()
        self.edge_threshold = edge_threshold

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_edges(self, img: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude."""
        img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(img_pad, self.sobel_x)
        grad_y = F.conv2d(img_pad, self.sobel_y)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    def forward(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute structure failure map.

        Returns:
            failure_map: [B, 1, H, W] - high where edges were lost
            loss: scalar - differentiable loss to restore edges
            info: dict with statistics
        """
        B, C, H, W = noisy.shape

        # Compute edges
        edge_noisy = self.compute_edges(noisy)
        edge_denoised = self.compute_edges(denoised)

        # Normalize edges
        edge_noisy_norm = edge_noisy / (edge_noisy.mean() + 1e-8)
        edge_denoised_norm = edge_denoised / (edge_denoised.mean() + 1e-8)

        # Structure loss: edges in noisy should be preserved in denoised
        # Failure map: where noisy has edges but denoised doesn't
        edge_loss_map = F.relu(edge_noisy_norm - edge_denoised_norm)

        # Focus on significant edges (not noise)
        significant_edges = (edge_noisy_norm > edge_noisy_norm.mean()).float()
        failure_map = edge_loss_map * significant_edges

        # Optionally weight by boundary proximity
        if boundaries is not None:
            boundary_weight = self._create_boundary_weight(boundaries, H)
            failure_map = failure_map * (1 + boundary_weight)  # Extra weight near boundaries

        # Normalize
        failure_map = failure_map / (failure_map.max() + 1e-8)

        # Loss: preserve edges
        loss = failure_map.mean()

        # Edge correlation
        edge_correlation = (edge_noisy_norm * edge_denoised_norm).sum() / (
            torch.sqrt((edge_noisy_norm ** 2).sum() * (edge_denoised_norm ** 2).sum()) + 1e-8
        )

        info = {
            'edge_correlation': edge_correlation.item(),
            'edge_loss_mean': edge_loss_map.mean().item(),
            'failure_ratio': (failure_map > 0.5).float().mean().item(),
        }

        return failure_map, loss, info

    def _create_boundary_weight(self, boundaries: torch.Tensor, height: int) -> torch.Tensor:
        """Create weight map that's high near boundaries."""
        B, N, W = boundaries.shape
        device = boundaries.device
        sigma = 10.0

        rows = torch.arange(height, device=device, dtype=torch.float32)
        rows = rows.view(1, 1, -1, 1)

        boundaries_px = (boundaries * (height - 1)).unsqueeze(2)
        distances = (rows - boundaries_px).abs()
        min_distance = distances.min(dim=1)[0]  # [B, H, W]

        weight = torch.exp(-min_distance ** 2 / (2 * sigma ** 2))
        return weight.unsqueeze(1)  # [B, 1, H, W]


# =============================================================================
# GUIDED REFINEMENT NETWORK
# =============================================================================

class GuidedRefinementBlock(nn.Module):
    """
    Refinement block that takes failure maps as guidance.
    """

    def __init__(self, channels: int = 32):
        super().__init__()

        # Feature extraction from denoised image
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Failure map processing (3 failure maps → guidance)
        self.guidance_net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid(),  # Attention weights
        )

        # Refinement prediction
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Tanh(),  # Residual in [-1, 1]
        )

        # Scale factor (learnable, starts small)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        denoised: torch.Tensor,
        failure_maps: torch.Tensor,  # [B, 3, H, W] - stacked failure maps
    ) -> torch.Tensor:
        """
        Compute guided residual.

        Args:
            denoised: [B, 1, H, W] current denoised image
            failure_maps: [B, 3, H, W] stacked (speckle, anatomy, structure) maps

        Returns:
            residual: [B, 1, H, W] refinement to add
        """
        # Extract features
        features = self.feat_extract(denoised)

        # Compute guidance attention from failure maps
        guidance = self.guidance_net(failure_maps)

        # Apply guidance
        guided_features = features * guidance

        # Predict residual
        residual = self.refine(guided_features)

        # Scale and weight by total failure
        total_failure = failure_maps.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        residual = residual * self.scale * total_failure

        return residual


# =============================================================================
# PREDICATE-GUIDED DENOISER
# =============================================================================

@dataclass
class RefinementResult:
    """Result of one refinement iteration."""
    denoised: torch.Tensor
    boundaries: torch.Tensor
    failure_maps: Dict[str, torch.Tensor]
    losses: Dict[str, float]
    satisfied: Dict[str, bool]
    iteration: int


class PredicateGuidedDenoiser(nn.Module):
    """
    Complete predicate-guided denoising system.

    Architecture:
    1. Backbone denoiser (e.g., NAFNet) - initial denoising
    2. Boundary detector - for anatomy predicate
    3. Failure map generators - spatial predicate evaluation
    4. Guided refinement network - iterative correction

    Training:
    - Phase 1: Train backbone with L1 loss (supervised)
    - Phase 2: Train refinement with predicate losses (self-supervised)
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        boundary_model: nn.Module = None,
        refinement_channels: int = 32,
        max_iterations: int = 3,
        # Predicate thresholds
        speckle_cv: float = 0.40,
        speckle_tolerance: float = 0.14,
        min_layer_thickness: float = 0.03,
        edge_threshold: float = 0.5,
    ):
        super().__init__()

        self.max_iterations = max_iterations

        # Backbone denoiser (can be pretrained NAFNet)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = self._simple_backbone()

        # Boundary detector (can be pretrained)
        if boundary_model is not None:
            self.boundary_model = boundary_model
        else:
            self.boundary_model = None

        # Failure map generators
        self.speckle_map = SpeckleFailureMap(
            expected_cv=speckle_cv,
            tolerance=speckle_tolerance,
        )
        self.anatomy_map = AnatomyFailureMap(
            min_layer_thickness=min_layer_thickness,
        )
        self.structure_map = StructureFailureMap(
            edge_threshold=edge_threshold,
        )

        # Guided refinement network
        self.refinement = GuidedRefinementBlock(channels=refinement_channels)

        # Hard constraint projector for boundaries
        from oct_symbolic_knowledge import SymbolicConstraints
        self.constraint_projector = SymbolicConstraints()

    def _simple_backbone(self):
        """Simple backbone if none provided."""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def get_boundaries(self, img: torch.Tensor) -> torch.Tensor:
        """Get boundaries from image."""
        if self.boundary_model is not None:
            with torch.no_grad():
                out = self.boundary_model(img, return_aux=True)
                boundaries = out['boundaries']
        else:
            # Dummy boundaries if no model
            B, _, H, W = img.shape
            boundaries = torch.linspace(0.2, 0.7, 4).view(1, 4, 1).expand(B, 4, W)
            boundaries = boundaries.to(img.device)

        # Apply hard constraint projection
        boundaries = self.constraint_projector.project_to_valid_space(boundaries)
        return boundaries

    def evaluate_predicates(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float], Dict[str, bool]]:
        """
        Evaluate all predicates and get failure maps.

        Returns:
            combined_failure: [B, 3, H, W] stacked failure maps
            individual_maps: dict of individual maps
            losses: dict of predicate losses
            satisfied: dict of predicate satisfaction
        """
        B, _, H, W = noisy.shape

        # Evaluate each predicate
        speckle_fail, speckle_loss, speckle_info = self.speckle_map(noisy, denoised)
        anatomy_fail, anatomy_loss, anatomy_info = self.anatomy_map(boundaries, H)
        structure_fail, structure_loss, structure_info = self.structure_map(
            noisy, denoised, boundaries
        )

        # Stack failure maps
        combined_failure = torch.cat([
            speckle_fail,
            anatomy_fail,
            structure_fail,
        ], dim=1)  # [B, 3, H, W]

        # Individual maps
        individual_maps = {
            'speckle': speckle_fail,
            'anatomy': anatomy_fail,
            'structure': structure_fail,
        }

        # Losses
        losses = {
            'speckle': speckle_loss.item() if torch.is_tensor(speckle_loss) else speckle_loss,
            'anatomy': anatomy_loss.item() if torch.is_tensor(anatomy_loss) else anatomy_loss,
            'structure': structure_loss.item() if torch.is_tensor(structure_loss) else structure_loss,
        }

        # Satisfaction (based on mean failure)
        satisfied = {
            'speckle': speckle_info['failure_type'] == 'ok',
            'anatomy': anatomy_info['total_violations'] < 0.01,
            'structure': structure_info['edge_correlation'] > 0.5,
        }

        return combined_failure, individual_maps, losses, satisfied

    def forward(
        self,
        noisy: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with iterative predicate-guided refinement.

        Args:
            noisy: [B, 1, H, W] noisy input
            return_intermediates: whether to return all iterations

        Returns:
            dict with:
                'denoised': final denoised image
                'boundaries': detected boundaries
                'iterations': number of iterations used
                'satisfied': dict of final predicate satisfaction
                'failure_maps': final failure maps
        """
        B, C, H, W = noisy.shape
        intermediates = []

        # Step 1: Initial denoising with backbone
        denoised = self.backbone(noisy)
        denoised = torch.clamp(denoised, 0, 1)

        # Step 2: Get boundaries
        boundaries = self.get_boundaries(denoised)

        # Step 3: Iterative refinement
        for iteration in range(self.max_iterations):
            # Evaluate predicates
            combined_failure, maps, losses, satisfied = self.evaluate_predicates(
                noisy, denoised, boundaries
            )

            if return_intermediates:
                intermediates.append(RefinementResult(
                    denoised=denoised.clone(),
                    boundaries=boundaries.clone(),
                    failure_maps=maps,
                    losses=losses,
                    satisfied=satisfied,
                    iteration=iteration,
                ))

            # Check if all predicates satisfied
            if all(satisfied.values()):
                break

            # Apply guided refinement
            residual = self.refinement(denoised, combined_failure)
            denoised = denoised + residual
            denoised = torch.clamp(denoised, 0, 1)

            # Update boundaries for refined image
            boundaries = self.get_boundaries(denoised)

        # Final evaluation
        combined_failure, maps, losses, satisfied = self.evaluate_predicates(
            noisy, denoised, boundaries
        )

        result = {
            'denoised': denoised,
            'boundaries': boundaries,
            'iterations': iteration + 1,
            'satisfied': satisfied,
            'all_satisfied': all(satisfied.values()),
            'failure_maps': maps,
            'losses': losses,
        }

        if return_intermediates:
            result['intermediates'] = intermediates

        return result

    def compute_loss(
        self,
        noisy: torch.Tensor,
        clean: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        If clean is provided: supervised backbone + self-supervised refinement
        If clean is None: fully self-supervised via predicates
        """
        outputs = self.forward(noisy, return_intermediates=True)

        losses = {}
        total_loss = torch.tensor(0.0, device=noisy.device)

        # Supervised loss (if clean available)
        if clean is not None:
            l1_loss = F.l1_loss(outputs['denoised'], clean)
            losses['l1'] = l1_loss.item()
            total_loss = total_loss + l1_loss

        # Self-supervised predicate losses
        combined_failure, maps, pred_losses, _ = self.evaluate_predicates(
            noisy, outputs['denoised'], outputs['boundaries']
        )

        # Speckle loss: push CV toward expected
        speckle_loss = self.speckle_map(noisy, outputs['denoised'])[1]
        losses['speckle'] = speckle_loss.item() if torch.is_tensor(speckle_loss) else speckle_loss
        total_loss = total_loss + 0.5 * speckle_loss

        # Structure loss: preserve edges
        structure_loss = self.structure_map(noisy, outputs['denoised'], outputs['boundaries'])[1]
        losses['structure'] = structure_loss.item() if torch.is_tensor(structure_loss) else structure_loss
        total_loss = total_loss + 0.5 * structure_loss

        # Anatomy loss: valid boundaries
        anatomy_loss = self.anatomy_map(outputs['boundaries'], noisy.shape[2])[1]
        losses['anatomy'] = anatomy_loss.item() if torch.is_tensor(anatomy_loss) else anatomy_loss
        total_loss = total_loss + 0.1 * anatomy_loss

        losses['total'] = total_loss.item()

        return total_loss, losses


# =============================================================================
# TESTING
# =============================================================================

def test_predicate_guided_denoising():
    """Test the predicate-guided denoising system."""
    print("=" * 70)
    print("TESTING PREDICATE-GUIDED DENOISING")
    print("=" * 70)

    torch.manual_seed(42)

    # Create synthetic OCT-like image
    B, H, W = 1, 256, 64
    clean = torch.zeros(B, 1, H, W)
    clean[:, :, 50:180, :] = 0.7
    clean[:, :, 80:120, :] = 0.3
    clean[:, :, 140:160, :] = 0.9

    # Add noise
    noise = torch.randn_like(clean) * 0.1
    noisy = (clean + noise).clamp(0, 1)

    print(f"\nInput shape: {noisy.shape}")

    # Create model (without pretrained backbone/boundary)
    model = PredicateGuidedDenoiser(
        backbone=None,
        boundary_model=None,
        max_iterations=3,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n" + "-" * 70)
    print("FORWARD PASS")
    print("-" * 70)

    with torch.no_grad():
        outputs = model(noisy, return_intermediates=True)

    print(f"Iterations used: {outputs['iterations']}")
    print(f"All predicates satisfied: {outputs['all_satisfied']}")
    print(f"\nPredicate satisfaction:")
    for name, sat in outputs['satisfied'].items():
        print(f"  {name}: {'✓' if sat else '✗'}")

    print(f"\nPredicate losses:")
    for name, loss in outputs['losses'].items():
        print(f"  {name}: {loss:.4f}")

    # Test intermediates
    if 'intermediates' in outputs:
        print(f"\nIteration progress:")
        for inter in outputs['intermediates']:
            sat_str = ", ".join([f"{k}={'✓' if v else '✗'}" for k, v in inter.satisfied.items()])
            print(f"  Iter {inter.iteration}: {sat_str}")

    # Test training loss
    print("\n" + "-" * 70)
    print("TRAINING LOSS")
    print("-" * 70)

    model.train()
    loss, loss_dict = model.compute_loss(noisy, clean)

    print(f"Total loss: {loss.item():.4f}")
    for name, val in loss_dict.items():
        print(f"  {name}: {val:.4f}")

    # Test gradient flow
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"\nGradient norm: {grad_norm:.4f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_predicate_guided_denoising()
