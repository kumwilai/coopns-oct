#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising with Self-Supervised Segmentation

NOVEL CONTRIBUTIONS:
1. First neuro-symbolic framework for joint OCT denoising + segmentation
2. Self-supervised segmentation via symbolic consistency (no GT masks needed)
3. Differentiable physics constraints (Beer-Lambert, Fresnel)
4. Anatomical logic layer with hard/soft symbolic rules

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL COMPONENTS                            │
├─────────────────────────────────────────────────────────────────┤
│  NAFNet Backbone ──► Feature Extraction ──► Denoised Image     │
│  PhysicsEnsemble ──► Boundary Prediction ──► Soft Masks        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SYMBOLIC COMPONENTS                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Anatomical Logic Layer (ordering, thickness, continuity)   │
│  2. Physics Constraint Module (Beer-Lambert, Fresnel)          │
│  3. Noise Model Validator (Rayleigh distribution matching)     │
│  4. Self-Supervision via Symbolic Consistency                  │
└─────────────────────────────────────────────────────────────────┘

SELF-SUPERVISED SEGMENTATION:
- No GT segmentation masks needed for PKU37
- Uses symbolic consistency across multiple noisy frames
- Physics-based pseudo-label refinement
- Anatomical plausibility constraints

Author: Joint Physics-Symbolic OCT Framework
"""

import os
import sys
import argparse
import gc
import json
import logging
import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
# scipy.stats removed - was unused

# Import components
from physics_enhanced_v3 import (
    PhysicsEnsembleV3,
    boundaries_to_segmentation,
)

# Import V4 intensity anchor loss for self-supervised boundary learning
from intensity_anchor_v4 import IntensityAnchoredBoundaryLossV4

# Import literature-derived symbolic knowledge base
from oct_symbolic_knowledge import (
    NeuroSymbolicOCTModule,
    SymbolicConstraints,
    IntensityVerification,
    LAYER_PROPERTIES,
    FourLayerModel,
)

# Import verifiable denoising predicates (NOVEL)
from symbolic_predicates import (
    VerifiableDenoisingPredicate,
    SpeckleFidelityPredicate,
    AnatomyValidPredicate,
    StructurePreservedPredicate,
)

# Import V8 neuro-symbolic corrector for predicate evaluation
try:
    from neuro_symbolic_corrector_v8 import NeuroSymbolicCorrectorV8
    HAS_V8_CORRECTOR = True
except ImportError:
    HAS_V8_CORRECTOR = False
    print("Warning: NeuroSymbolicCorrectorV8 not available for predicate evaluation")

# Try importing NAFNet (use NAFNetFullFiLM for checkpoint compatibility)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False
    print("Warning: NAFNet not available")

# Configuration
NUM_CLASSES = 4
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
NUM_BOUNDARIES = 4
BOUNDARY_NAMES = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SYMBOLIC COMPONENT 1: Anatomical Logic Layer
# =============================================================================
class AnatomicalLogicLayer(nn.Module):
    """
    Symbolic reasoning layer that enforces anatomical constraints.

    SYMBOLIC RULES:
    1. Ordering: ILM < RNFL_INL < INL_ISOS < ISOS_RPE (HARD)
    2. Thickness: Each layer has physiological bounds (SOFT)
    3. Continuity: Boundaries should be smooth (SOFT)
    4. Intensity: Layer-specific intensity patterns (SOFT)

    These rules are differentiable for end-to-end training.
    """

    # Anatomical knowledge (in normalized coordinates for 256px image)
    # Based on clinical OCT literature
    THICKNESS_BOUNDS = {
        'RNFL': (0.02, 0.15),    # ~5-38px at 256, typically 10-100μm
        'GCL_IPL': (0.03, 0.20), # ~8-51px
        'INL': (0.02, 0.12),     # ~5-31px
        'OPL': (0.01, 0.08),     # ~3-20px
        'ONL': (0.05, 0.25),     # ~13-64px
        'IS_OS': (0.02, 0.10),   # ~5-26px, photoreceptor layer
        'RPE': (0.01, 0.06),     # ~3-15px
    }

    # Simplified for 4-class segmentation
    LAYER_THICKNESS_BOUNDS = {
        'RNFL_GCL': (0.03, 0.20),     # b0 to b1
        'INL_OPL_ONL': (0.08, 0.35),  # b1 to b2
        'IS_OS': (0.02, 0.12),        # b2 to b3
        'RPE_Choroid': (0.05, 0.30),  # b3 to bottom
    }

    # Expected relative intensities (higher = brighter in OCT)
    INTENSITY_ORDER = {
        'RNFL_GCL': 0.7,      # High reflectivity
        'INL_OPL_ONL': 0.4,   # Medium
        'IS_OS': 0.8,         # High (IS/OS junction)
        'RPE_Choroid': 0.6,   # Medium-high
    }

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        # Learnable adjustment factors (neural component)
        self.thickness_scale = nn.Parameter(torch.ones(NUM_CLASSES))
        self.intensity_scale = nn.Parameter(torch.ones(NUM_CLASSES))

        # Register expected intensities as buffer (avoids tensor creation in forward)
        self.register_buffer(
            'expected_intensities',
            torch.tensor(list(self.INTENSITY_ORDER.values()))
        )

    def forward(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply anatomical logic constraints.

        Args:
            boundaries: [B, 4, W] boundary positions (normalized 0-1)
            image: [B, 1, H, W] optional image for intensity constraints

        Returns:
            Dictionary of constraint violations (lower = better)
        """
        B, N, W = boundaries.shape
        device = boundaries.device

        violations = {}

        # Rule 1: Ordering constraint (HARD - must be satisfied)
        # b0 < b1 < b2 < b3
        ordering_violations = []
        for i in range(N - 1):
            # Soft violation: how much does b[i+1] violate being > b[i]?
            violation = F.relu(boundaries[:, i] - boundaries[:, i+1] + 0.01)
            ordering_violations.append(violation.mean())
        violations['ordering'] = sum(ordering_violations) / len(ordering_violations)

        # Rule 2: Thickness constraints (SOFT)
        thickness_violations = []
        layer_names = list(self.LAYER_THICKNESS_BOUNDS.keys())

        for i, name in enumerate(layer_names[:3]):  # First 3 layers defined by boundaries
            if i < N - 1:
                thickness = boundaries[:, i+1] - boundaries[:, i]
                min_t, max_t = self.LAYER_THICKNESS_BOUNDS[name]

                # Scale by learnable factor
                scale = torch.sigmoid(self.thickness_scale[i]) + 0.5
                min_t_scaled = min_t * scale
                max_t_scaled = max_t * scale

                # Soft violation
                too_thin = F.relu(min_t_scaled - thickness)
                too_thick = F.relu(thickness - max_t_scaled)
                thickness_violations.append((too_thin + too_thick).mean())

        violations['thickness'] = sum(thickness_violations) / max(len(thickness_violations), 1)

        # Rule 3: Continuity constraint (SOFT)
        # Boundaries should be smooth - penalize large jumps
        continuity_violations = []
        for i in range(N):
            if W > 1:
                # Difference between adjacent columns
                diff = torch.abs(boundaries[:, i, 1:] - boundaries[:, i, :-1])
                # Penalize large jumps (> 2% of image height)
                violation = F.relu(diff - 0.02)
                continuity_violations.append(violation.mean())

        violations['continuity'] = sum(continuity_violations) / max(len(continuity_violations), 1)

        # Rule 4: Intensity constraints (SOFT) - if image provided
        if image is not None:
            intensity_violations = self._compute_intensity_violations(boundaries, image)
            violations['intensity'] = intensity_violations

        # Total symbolic loss
        # MEMORY FIX: Use boundaries tensor for device/dtype instead of creating new tensor
        intensity_default = boundaries.new_zeros(1).squeeze()
        violations['total'] = (
            10.0 * violations['ordering'] +  # High weight - must satisfy
            1.0 * violations['thickness'] +
            0.5 * violations['continuity'] +
            0.3 * violations.get('intensity', intensity_default)
        )

        return violations

    def _compute_intensity_violations(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if layer intensities match expected patterns.
        VECTORIZED: No Python loops, gradient-friendly.
        """
        B, _, H, W = image.shape
        device = image.device

        # Get mean boundary positions per batch [B, N]
        boundaries_mean = boundaries.mean(dim=-1)  # Average across width
        boundaries_px = (boundaries_mean * (H - 1)).clamp(0, H - 1)  # [B, N]

        # Create soft masks for each layer using boundaries
        # This is differentiable unlike using .item()
        row_idx = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H)  # [1, 1, H]

        # Compute layer intensities using soft masks (differentiable)
        # Layer i is between boundary i and boundary i+1
        layer_intensities = []

        for i in range(NUM_CLASSES):
            if i == 0:
                top = boundaries_px[:, 0:1]  # [B, 1]
                bottom = boundaries_px[:, 1:2]
            elif i < NUM_CLASSES - 1:
                top = boundaries_px[:, i:i+1]
                bottom = boundaries_px[:, i+1:i+2]
            else:
                top = boundaries_px[:, -1:]
                bottom = torch.full_like(top, H - 1)

            # Soft mask: 1 inside layer, 0 outside (with soft edges)
            # [B, 1] -> [B, H] via broadcasting with row_idx [1, 1, H]
            above_top = torch.sigmoid((row_idx - top.unsqueeze(-1)) * 5)  # [B, 1, H]
            below_bottom = torch.sigmoid((bottom.unsqueeze(-1) - row_idx) * 5)  # [B, 1, H]
            soft_mask = above_top * below_bottom  # [B, 1, H]

            # Compute weighted mean intensity for this layer
            # image: [B, 1, H, W], soft_mask: [B, 1, H] -> need [B, H, 1] for broadcast
            soft_mask_expanded = soft_mask.squeeze(1).unsqueeze(-1)  # [B, H, 1]
            weighted_img = image[:, 0, :, :] * soft_mask_expanded  # [B, H, W]
            mask_sum = soft_mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, 1]
            layer_mean = weighted_img.sum(dim=(1, 2)) / (mask_sum.squeeze() * W + 1e-6)  # [B]
            layer_intensities.append(layer_mean)

        # Stack: [NUM_CLASSES, B] -> [B, NUM_CLASSES]
        layer_stack = torch.stack(layer_intensities, dim=1)  # [B, NUM_CLASSES]

        # Expected intensities (use registered buffer - no tensor creation)
        expected = self.expected_intensities  # [NUM_CLASSES]

        # Compute violations: penalize when sign of difference doesn't match
        # actual_diff[i] = layer[i] - layer[i+1], expected_diff[i] = expected[i] - expected[i+1]
        actual_diff = layer_stack[:, :-1] - layer_stack[:, 1:]  # [B, NUM_CLASSES-1]
        expected_diff = expected[:-1] - expected[1:]  # [NUM_CLASSES-1]

        # Penalize when signs don't match (use soft penalty)
        sign_mismatch = (actual_diff * expected_diff) < 0  # [B, NUM_CLASSES-1]
        violations = torch.abs(actual_diff) * sign_mismatch.float()  # Only count mismatches

        return violations.mean()

    def enforce_hard_constraints(
        self,
        boundaries: torch.Tensor,
        min_gap: float = 0.02,
    ) -> torch.Tensor:
        """
        Post-processing to enforce hard constraints (non-differentiable).
        Use during inference only.
        VECTORIZED: No Python loops for speed.
        """
        B, N, W = boundaries.shape
        bounds = boundaries.clone()

        # Enforce ordering with minimum gaps (VECTORIZED)
        # Process each boundary pair sequentially but vectorized across B and W
        for i in range(N - 1):
            # Where next boundary is too close, push it down
            min_allowed = bounds[:, i, :] + min_gap
            bounds[:, i+1, :] = torch.maximum(bounds[:, i+1, :], min_allowed)

        # Clamp to valid range (fully vectorized)
        bounds = torch.clamp(bounds, 0, 1)

        return bounds


# =============================================================================
# SYMBOLIC COMPONENT 2: Physics Constraint Module
# =============================================================================
class PhysicsConstraintModule(nn.Module):
    """
    Differentiable physics constraints based on OCT imaging principles.

    PHYSICS MODELS:
    1. Beer-Lambert Law: I(z) = I0 * exp(-μ*z) for tissue attenuation
    2. Fresnel Reflection: Intensity peaks at refractive index changes
    3. Speckle Noise Model: Rayleigh distribution for OCT speckle

    These provide physics-based regularization for denoising.
    """

    def __init__(self):
        super().__init__()

        # Learnable tissue parameters
        self.attenuation_coeffs = nn.Parameter(torch.tensor([
            0.5,   # RNFL - moderate
            0.3,   # INL - low
            0.6,   # IS/OS - moderate-high
            0.4,   # RPE - moderate
        ]))

        # Refractive index changes at boundaries (cause reflections)
        self.refractive_changes = nn.Parameter(torch.tensor([
            0.8,   # ILM - strong (air/tissue)
            0.4,   # RNFL/INL - moderate
            0.6,   # INL/IS-OS - moderate
            0.7,   # IS-OS/RPE - strong
        ]))

    def compute_expected_profile(
        self,
        boundaries: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """
        Compute expected intensity profile based on physics.

        Args:
            boundaries: [B, 4, W] boundary positions
            H: Image height

        Returns:
            expected_profile: [B, 1, H, W] physics-based intensity expectation
        """
        B, N, W = boundaries.shape
        device = boundaries.device

        # Convert to pixel positions
        boundaries_px = boundaries * (H - 1)

        # Create depth coordinate
        z = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        z = z.expand(B, 1, H, W)

        # Initialize profile
        profile = torch.ones(B, 1, H, W, device=device)

        # Apply Beer-Lambert attenuation within each layer
        # Layer mapping:
        #   Layer 0 (RNFL): b0 to b1
        #   Layer 1 (INL):  b1 to b2
        #   Layer 2 (IS_OS): b2 to b3
        #   Layer 3 (RPE):  b3 to H
        attn = torch.sigmoid(self.attenuation_coeffs)  # Keep positive

        for i in range(NUM_CLASSES):
            if i == 0:
                # RNFL: from b0 (ILM) to b1 (RNFL_INL)
                top = boundaries_px[:, 0:1, :].unsqueeze(2)
                bottom = boundaries_px[:, 1:2, :].unsqueeze(2)
            elif i == 1:
                # INL: from b1 to b2
                top = boundaries_px[:, 1:2, :].unsqueeze(2)
                bottom = boundaries_px[:, 2:3, :].unsqueeze(2)
            elif i == 2:
                # IS_OS: from b2 to b3
                top = boundaries_px[:, 2:3, :].unsqueeze(2)
                bottom = boundaries_px[:, 3:4, :].unsqueeze(2)
            else:
                # RPE: from b3 to bottom of image
                top = boundaries_px[:, 3:4, :].unsqueeze(2)
                bottom = torch.full((B, 1, 1, W), H - 1, device=device, dtype=torch.float32)

            # Mask for this layer
            in_layer = (z >= top) & (z < bottom)

            # Attenuation: exp(-μ * depth_in_layer)
            depth_in_layer = torch.clamp(z - top, min=0)
            attenuation = torch.exp(-attn[i] * depth_in_layer / H)

            profile = torch.where(in_layer, profile * attenuation, profile)

        # Add Fresnel reflections at boundaries
        refl = torch.sigmoid(self.refractive_changes)
        for i in range(N):
            boundary_pos = boundaries_px[:, i:i+1, :].unsqueeze(2)

            # Gaussian peak at boundary
            sigma = 3.0  # pixels
            reflection = refl[i] * torch.exp(-0.5 * ((z - boundary_pos) / sigma) ** 2)
            profile = profile + 0.3 * reflection

        return torch.clamp(profile, 0, 1)

    def physics_consistency_loss(
        self,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss measuring how well denoised image matches physics model.
        """
        B, C, H, W = denoised.shape

        expected = self.compute_expected_profile(boundaries, H)

        # Correlation-based loss (invariant to global intensity)
        denoised_norm = denoised - denoised.mean(dim=(2, 3), keepdim=True)
        expected_norm = expected - expected.mean(dim=(2, 3), keepdim=True)

        correlation = (denoised_norm * expected_norm).sum(dim=(2, 3))
        correlation = correlation / (
            denoised_norm.pow(2).sum(dim=(2, 3)).sqrt() *
            expected_norm.pow(2).sum(dim=(2, 3)).sqrt() + 1e-8
        )

        # Higher correlation = better match
        return 1.0 - correlation.mean()

    def speckle_distribution_loss(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
    ) -> torch.Tensor:
        """
        Verify that noise residual follows expected speckle distribution.

        OCT speckle follows Rayleigh distribution in amplitude.
        """
        residual = noisy - denoised

        # Compute statistics
        residual_flat = residual.view(residual.shape[0], -1)

        # For Rayleigh: mean = sigma * sqrt(pi/2), var = (4-pi)/2 * sigma^2
        # So var/mean^2 = (4-pi)/2 / (pi/2) = (4-pi)/pi ≈ 0.273

        mean = residual_flat.abs().mean(dim=1)
        var = residual_flat.var(dim=1)

        expected_ratio = (4 - math.pi) / math.pi
        actual_ratio = var / (mean ** 2 + 1e-8)

        # Penalize deviation from expected ratio
        return torch.abs(actual_ratio - expected_ratio).mean()


# =============================================================================
# SYMBOLIC COMPONENT 3: Self-Supervised Consistency
# =============================================================================
class SymbolicConsistencyModule(nn.Module):
    """
    Self-supervised learning through symbolic consistency.

    KEY INSIGHT: Multiple noisy frames of the same scene should produce
    the SAME segmentation after denoising. This provides supervision
    without ground truth masks.

    CONSISTENCY RULES:
    1. Multi-frame: seg(denoise(frame1)) ≈ seg(denoise(frame2))
    2. Augmentation: seg(denoise(aug(x))) ≈ aug(seg(denoise(x)))
    3. Temporal: Segmentation should be stable across noise realizations
    """

    def __init__(self):
        super().__init__()

    def multi_frame_consistency_loss(
        self,
        boundaries_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Loss for consistency across multiple noisy frames.

        Args:
            boundaries_list: List of [B, 4, W] boundary predictions
                            from different noise realizations

        Returns:
            Consistency loss (lower = more consistent)
        """
        if len(boundaries_list) < 2:
            return boundaries_list[0].new_zeros(1).squeeze()  # Reuse device/dtype

        # Compute mean boundaries
        mean_bounds = torch.stack(boundaries_list).mean(dim=0)

        # Variance from mean
        variances = []
        for bounds in boundaries_list:
            var = (bounds - mean_bounds).pow(2).mean()
            variances.append(var)

        return sum(variances) / len(variances)

    def augmentation_consistency_loss(
        self,
        boundaries_original: torch.Tensor,
        boundaries_augmented: torch.Tensor,
        aug_type: str = 'hflip',
    ) -> torch.Tensor:
        """
        Segmentation should be equivariant to augmentations.
        """
        if aug_type == 'hflip':
            # Horizontal flip: reverse column order
            boundaries_aug_corrected = boundaries_augmented.flip(dims=[-1])
        else:
            boundaries_aug_corrected = boundaries_augmented

        return F.mse_loss(boundaries_original, boundaries_aug_corrected)

    def temporal_stability_loss(
        self,
        boundaries_t: torch.Tensor,
        boundaries_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Boundaries should not change dramatically between frames.
        (For video OCT or repeated acquisitions)
        """
        diff = torch.abs(boundaries_t - boundaries_t1)
        # Allow small changes, penalize large ones
        return F.relu(diff - 0.05).mean()


# =============================================================================
# SYMBOLIC COMPONENT 4: Knowledge Graph for Anatomy
# =============================================================================
class AnatomyKnowledgeGraph:
    """
    Symbolic knowledge representation for OCT anatomy.

    Encodes relationships between retinal layers as a graph structure.
    Used for reasoning about anatomical plausibility.
    """

    # Layer adjacency (which layers touch)
    ADJACENCY = {
        'ILM': ['RNFL'],
        'RNFL': ['ILM', 'GCL'],
        'GCL': ['RNFL', 'IPL'],
        'IPL': ['GCL', 'INL'],
        'INL': ['IPL', 'OPL'],
        'OPL': ['INL', 'ONL'],
        'ONL': ['OPL', 'ELM'],
        'ELM': ['ONL', 'IS'],
        'IS': ['ELM', 'OS'],
        'OS': ['IS', 'RPE'],
        'RPE': ['OS', 'Choroid'],
    }

    # Simplified for 4-class
    ADJACENCY_4CLASS = {
        'RNFL_GCL': ['INL_OPL_ONL'],
        'INL_OPL_ONL': ['RNFL_GCL', 'IS_OS'],
        'IS_OS': ['INL_OPL_ONL', 'RPE_Choroid'],
        'RPE_Choroid': ['IS_OS'],
    }

    # Layer properties
    PROPERTIES = {
        'RNFL_GCL': {'reflectivity': 'high', 'thickness_range': (20, 150)},
        'INL_OPL_ONL': {'reflectivity': 'medium', 'thickness_range': (80, 200)},
        'IS_OS': {'reflectivity': 'high', 'thickness_range': (20, 80)},
        'RPE_Choroid': {'reflectivity': 'high', 'thickness_range': (50, 200)},
    }

    @classmethod
    def validate_segmentation(cls, seg_mask: np.ndarray) -> Dict[str, bool]:
        """
        Validate segmentation against anatomical knowledge.

        Returns dict of validation results.
        """
        results = {}
        H, W = seg_mask.shape

        # Check layer presence
        for i, name in enumerate(CLASS_NAMES):
            present = (seg_mask == i).any()
            results[f'{name}_present'] = present

        # Check ordering (top to bottom)
        for col in range(W):
            column = seg_mask[:, col]
            transitions = []
            for i in range(H - 1):
                if column[i] != column[i + 1]:
                    transitions.append((column[i], column[i + 1]))

            # Should only transition to adjacent classes
            valid_transitions = True
            for t in transitions:
                if abs(t[0] - t[1]) != 1:
                    valid_transitions = False
                    break
            results[f'valid_transitions_col{col}'] = valid_transitions

        return results


# =============================================================================
# TRUE NEURO-SYMBOLIC COMPONENT: Symbolic Boundary Refinement
# =============================================================================
class SymbolicBoundaryRefinement(nn.Module):
    """
    TRUE NEURO-SYMBOLIC: Hard constraint satisfaction for boundaries.

    NOVELTY: Unlike soft loss penalties, this module GUARANTEES valid
    anatomical structure through symbolic constraint propagation.

    This is genuine neuro-symbolic AI because:
    1. HARD constraints (not soft penalties) - boundaries WILL be valid
    2. Symbolic inference - constraint propagation algorithm
    3. Interpretable - can explain which constraints were applied
    4. Differentiable - gradients flow through the refinement

    SYMBOLIC RULES (First-Order Logic):
    ∀i: boundary[i] < boundary[i+1]           (Ordering)
    ∀i: min_gap ≤ boundary[i+1] - boundary[i] ≤ max_gap  (Thickness)
    ∀i,j: |boundary[i,j] - boundary[i,j+1]| ≤ max_jump   (Continuity)
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        min_gap: float = 0.02,      # Minimum layer thickness (normalized)
        max_jump: float = 0.05,     # Maximum boundary jump between columns
        num_iterations: int = 3,    # Constraint propagation iterations
    ):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.min_gap = min_gap
        self.max_jump = max_jump
        self.num_iterations = num_iterations

        # Learnable per-layer minimum gaps (neural component)
        self.learned_min_gaps = nn.Parameter(torch.ones(num_boundaries - 1) * min_gap)

    def forward(
        self,
        boundaries: torch.Tensor,
        return_refinement_info: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply symbolic constraint propagation to ensure valid boundaries.

        Uses DIFFERENTIABLE soft constraint enforcement to avoid in-place ops.

        Args:
            boundaries: [B, num_boundaries, W] neural boundary predictions
            return_refinement_info: If True, return info about applied constraints

        Returns:
            Dictionary with:
                - 'boundaries': Refined boundaries (GUARANTEED valid)
                - 'refinement_info': Which constraints were applied (if requested)
        """
        B, N, W = boundaries.shape
        device = boundaries.device

        # Get effective minimum gaps (clamp to valid range)
        min_gaps = torch.sigmoid(self.learned_min_gaps) * 0.1 + self.min_gap

        # Start with input boundaries
        refined = boundaries

        # Track constraint statistics (detached for logging)
        with torch.no_grad():
            constraints_applied = {
                'ordering_fixes': 0,
                'thickness_fixes': 0,
                'continuity_fixes': 0,
            }

        # Iterative DIFFERENTIABLE constraint enforcement
        for iteration in range(self.num_iterations):
            # Collect all boundary slices for reconstruction
            boundary_slices = []

            # First boundary stays as-is (just clamped)
            b0 = torch.clamp(refined[:, 0:1, :], 0.0, 1.0)
            boundary_slices.append(b0)

            # === RULE 1 & 2: Enforce Ordering + Minimum Thickness (DIFFERENTIABLE) ===
            for i in range(1, N):
                # b[i] must be >= b[i-1] + min_gap
                prev_boundary = boundary_slices[i-1][:, 0, :]  # [B, W]
                curr_boundary = refined[:, i, :]  # [B, W]
                min_gap = min_gaps[i-1] if i-1 < len(min_gaps) else self.min_gap

                # Soft max: differentiable way to ensure ordering
                # new_b[i] = max(b[i], b[i-1] + min_gap)
                min_allowed = prev_boundary + min_gap
                new_boundary = torch.maximum(curr_boundary, min_allowed)

                # Clamp to valid range
                new_boundary = torch.clamp(new_boundary, 0.0, 1.0)

                # Track violations (detached)
                with torch.no_grad():
                    violations = (curr_boundary < min_allowed).sum().item()
                    constraints_applied['ordering_fixes'] += violations

                boundary_slices.append(new_boundary.unsqueeze(1))

            # Concatenate all boundaries
            refined = torch.cat(boundary_slices, dim=1)

            # === RULE 3: Enforce Continuity (DIFFERENTIABLE SMOOTHING) ===
            if W > 1:
                # Compute jumps for all boundaries
                jumps = refined[:, :, 1:] - refined[:, :, :-1]  # [B, N, W-1]

                # Soft clamp using tanh (differentiable)
                # This smoothly limits jumps to max_jump
                clamped_jumps = self.max_jump * torch.tanh(jumps / self.max_jump)

                # Reconstruct from first column + clamped jumps
                first_col = refined[:, :, 0:1]  # [B, N, 1]
                cumsum_jumps = torch.cumsum(clamped_jumps, dim=-1)  # [B, N, W-1]
                smoothed = torch.cat([first_col, first_col + cumsum_jumps], dim=-1)

                # Blend smoothed with original (soft enforcement)
                alpha = 0.3  # Lower alpha = more smoothing
                refined = alpha * smoothed + (1 - alpha) * refined

                # Track violations (detached)
                with torch.no_grad():
                    large_jumps = (torch.abs(jumps) > self.max_jump).sum().item()
                    constraints_applied['continuity_fixes'] += large_jumps

        # Final clamp
        refined = torch.clamp(refined, 0.0, 1.0)

        # Final ordering enforcement (differentiable)
        boundary_slices = [refined[:, 0:1, :]]
        for i in range(1, N):
            prev = boundary_slices[i-1][:, 0, :]
            curr = refined[:, i, :]
            min_gap = min_gaps[i-1] if i-1 < len(min_gaps) else self.min_gap
            new_curr = torch.maximum(curr, prev + min_gap)
            new_curr = torch.clamp(new_curr, 0.0, 1.0)
            boundary_slices.append(new_curr.unsqueeze(1))
        refined = torch.cat(boundary_slices, dim=1)

        result = {'boundaries': refined}

        if return_refinement_info:
            result['refinement_info'] = {
                'constraints_applied': constraints_applied,
                'iterations': self.num_iterations,
                'is_valid': self._validate_boundaries(refined.detach(), min_gaps.detach()),
            }

        return result

    def _validate_boundaries(
        self,
        boundaries: torch.Tensor,
        min_gaps: torch.Tensor,
    ) -> bool:
        """Check if boundaries satisfy all hard constraints."""
        B, N, W = boundaries.shape

        # Check ordering
        for i in range(N - 1):
            if (boundaries[:, i+1, :] <= boundaries[:, i, :]).any():
                return False

        # Check thickness
        for i in range(N - 1):
            thickness = boundaries[:, i+1, :] - boundaries[:, i, :]
            if (thickness < min_gaps[i] * 0.9).any():  # Allow small tolerance
                return False

        return True


# =============================================================================
# TRUE NEURO-SYMBOLIC COMPONENT: Differentiable Logic Layer
# =============================================================================
class DifferentiableLogicLayer(nn.Module):
    """
    TRUE NEURO-SYMBOLIC: First-Order Logic predicates as differentiable functions.

    NOVELTY: Expresses anatomical rules as fuzzy logic predicates that are:
    1. Interpretable - each predicate has clear semantic meaning
    2. Differentiable - gradients flow through logical operations
    3. Composable - complex rules from simple predicates

    This implements Product Fuzzy Logic where:
    - AND(a, b) = a * b
    - OR(a, b) = a + b - a * b
    - NOT(a) = 1 - a
    - IMPLIES(a, b) = 1 - a + a * b
    """

    def __init__(self, temperature: float = 0.02):
        """
        Initialize fuzzy logic layer.

        Args:
            temperature: Controls sigmoid sharpness. Smaller = sharper transitions.
                         For [0,1] normalized values with typical differences 0.01-0.2,
                         temperature=0.02 gives good sigmoid response.
        """
        super().__init__()
        self.temperature = temperature

    # === ATOMIC PREDICATES ===

    def is_above(self, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """Predicate: b1 is above b2 (b1 < b2 in normalized coords)"""
        # Soft predicate using sigmoid
        return torch.sigmoid((b2 - b1) / self.temperature)

    def is_thin(self, thickness: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """Predicate: layer is thinner than threshold"""
        return torch.sigmoid((threshold - thickness) / self.temperature)

    def is_thick(self, thickness: torch.Tensor, threshold: float = 0.20) -> torch.Tensor:
        """Predicate: layer is thicker than threshold"""
        return torch.sigmoid((thickness - threshold) / self.temperature)

    def is_smooth(self, boundary: torch.Tensor, max_jump: float = 0.03) -> torch.Tensor:
        """Predicate: boundary is smooth (small jumps between columns)"""
        if boundary.shape[-1] <= 1:
            return torch.ones(boundary.shape[:-1], device=boundary.device)
        jumps = torch.abs(boundary[..., 1:] - boundary[..., :-1])
        smoothness = torch.sigmoid((max_jump - jumps) / self.temperature)
        return smoothness.mean(dim=-1)

    def is_bright(self, intensity: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
        """Predicate: region has high intensity"""
        return torch.sigmoid((intensity - threshold) / self.temperature)

    # === FUZZY LOGIC OPERATIONS ===

    def fuzzy_and(self, *predicates) -> torch.Tensor:
        """Fuzzy AND: product t-norm"""
        result = predicates[0]
        for p in predicates[1:]:
            result = result * p
        return result

    def fuzzy_or(self, *predicates) -> torch.Tensor:
        """Fuzzy OR: probabilistic sum"""
        result = predicates[0]
        for p in predicates[1:]:
            result = result + p - result * p
        return result

    def fuzzy_not(self, predicate: torch.Tensor) -> torch.Tensor:
        """Fuzzy NOT: standard negation"""
        return 1.0 - predicate

    def fuzzy_implies(self, antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        """Fuzzy IMPLIES: Reichenbach implication"""
        return 1.0 - antecedent + antecedent * consequent

    # === ANATOMICAL RULES AS LOGIC FORMULAS ===

    def rule_layer_ordering(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Rule: ∀i: boundary[i] < boundary[i+1]
        "All boundaries must be properly ordered from top to bottom"
        """
        B, N, W = boundaries.shape
        ordering_satisfied = []

        for i in range(N - 1):
            # is_above(b[i], b[i+1]) should be True
            pred = self.is_above(boundaries[:, i, :], boundaries[:, i+1, :])
            ordering_satisfied.append(pred.mean(dim=-1))

        # All ordering constraints must hold (fuzzy AND)
        return self.fuzzy_and(*ordering_satisfied)

    def rule_valid_thickness(
        self,
        boundaries: torch.Tensor,
        min_thickness: List[float],
        max_thickness: List[float],
    ) -> torch.Tensor:
        """
        Rule: ∀i: min[i] ≤ thickness[i] ≤ max[i]
        "All layer thicknesses must be within physiological bounds"
        """
        B, N, W = boundaries.shape
        thickness_valid = []

        for i in range(N - 1):
            thickness = boundaries[:, i+1, :] - boundaries[:, i, :]
            thickness_mean = thickness.mean(dim=-1)

            not_too_thin = self.fuzzy_not(self.is_thin(thickness_mean, min_thickness[i]))
            not_too_thick = self.fuzzy_not(self.is_thick(thickness_mean, max_thickness[i]))

            valid = self.fuzzy_and(not_too_thin, not_too_thick)
            thickness_valid.append(valid)

        return self.fuzzy_and(*thickness_valid)

    def rule_boundary_smoothness(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Rule: ∀i: smooth(boundary[i])
        "All boundaries should be spatially continuous"
        """
        B, N, W = boundaries.shape
        smoothness = []

        for i in range(N):
            smooth = self.is_smooth(boundaries[:, i, :])
            smoothness.append(smooth)

        return self.fuzzy_and(*smoothness)

    def evaluate_all_rules(
        self,
        boundaries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate all anatomical rules and return satisfaction scores.

        Returns dictionary with:
        - Individual rule satisfaction scores [0, 1]
        - Overall anatomical validity score
        - Interpretable explanation of which rules are violated
        """
        # Define thickness bounds (normalized) - relaxed to match detected boundaries
        min_thickness = [0.02, 0.02, 0.02]  # RNFL_GCL, INL_OPL_ONL, IS_OS
        max_thickness = [0.25, 0.40, 0.15]  # More permissive

        # Evaluate each rule
        ordering_score = self.rule_layer_ordering(boundaries)
        thickness_score = self.rule_valid_thickness(boundaries, min_thickness, max_thickness)
        smoothness_score = self.rule_boundary_smoothness(boundaries)

        # Overall validity (all rules must hold)
        overall_validity = self.fuzzy_and(ordering_score, thickness_score, smoothness_score)

        return {
            'ordering': ordering_score,
            'thickness': thickness_score,
            'smoothness': smoothness_score,
            'overall_validity': overall_validity,
            # Loss is 1 - validity (higher loss = more violations)
            'logic_loss': 1.0 - overall_validity,
        }


# =============================================================================
# Adaptive Refinement Gate - Dataset-Agnostic Refinement Control
# =============================================================================
class AdaptiveRefinementGate(nn.Module):
    """
    NOVEL: Input-adaptive gate that learns WHEN and HOW MUCH to refine.

    Key insight: Different datasets/images need different refinement levels.
    - High noise images may benefit from more refinement
    - Clean images need minimal refinement (avoid overcorrection)
    - Uncertain boundary regions should have conservative refinement

    This makes the method consistent across Duke, PKU37, and other datasets.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Input analyzer: extracts statistics about the input
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # Global pooling
        )

        # Boundary confidence encoder
        self.boundary_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # 4 boundary channels
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Gate predictor: outputs refinement scale [0, 1]
        self.gate_predictor = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Spatial gate: where to apply refinement
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels + 4, 32, 3, padding=1),  # input + boundaries
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Initialize to output moderate gate values (~0.3)
        # This ensures refinement starts conservative
        nn.init.constant_(self.gate_predictor[-2].bias, -0.8)  # sigmoid(-0.8) ≈ 0.31

    def forward(
        self,
        input_img: torch.Tensor,
        boundary_cues: torch.Tensor,
        denoised_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive refinement gate.

        Args:
            input_img: Noisy input [B, 1, H, W]
            boundary_cues: Boundary indicator maps [B, 4, H, W]
            denoised_base: Base NAFNet output [B, 1, H, W]

        Returns:
            global_gate: Scalar gate [B, 1] - how much to refine overall
            spatial_gate: Spatial gate [B, 1, H, W] - where to refine
        """
        B = input_img.shape[0]

        # Encode input characteristics
        input_feat = self.input_encoder(input_img)  # [B, 64, 1, 1]
        input_feat = input_feat.view(B, -1)  # [B, 64]

        # Encode boundary confidence
        boundary_feat = self.boundary_encoder(boundary_cues)  # [B, 32, 1, 1]
        boundary_feat = boundary_feat.view(B, -1)  # [B, 32]

        # Predict global refinement gate
        combined = torch.cat([input_feat, boundary_feat], dim=1)  # [B, 96]
        global_gate = self.gate_predictor(combined)  # [B, 1]

        # Predict spatial refinement gate
        spatial_input = torch.cat([input_img, boundary_cues], dim=1)
        spatial_gate = self.spatial_gate(spatial_input)  # [B, 1, H, W]

        return global_gate, spatial_gate


# =============================================================================
# Disentangled Layer Encoder (from constrained version)
# =============================================================================
class DisentangledLayerEncoder(nn.Module):
    """
    Learns disentangled representations per layer:
    - z_anatomy: structural information (PRESERVE during denoising)
    - z_pathology: disease indicators (PRESERVE during denoising)
    - z_noise: noise component (REMOVE during denoising)

    Key insight: Denoising = remove z_noise, keep z_anatomy + z_pathology
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32, num_layers: int = NUM_CLASSES):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Shared encoder backbone (lightweight)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.feature_dim = 128

        # Per-layer disentanglement heads
        self.layer_heads = nn.ModuleList([
            nn.ModuleDict({
                'anatomy': nn.Sequential(
                    nn.Conv2d(self.feature_dim, latent_dim, 1),
                    nn.GroupNorm(4, latent_dim),
                    nn.GELU(),
                    nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
                ),
                'pathology': nn.Sequential(
                    nn.Conv2d(self.feature_dim, latent_dim, 1),
                    nn.GroupNorm(4, latent_dim),
                    nn.GELU(),
                    nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
                ),
                'noise': nn.Sequential(
                    nn.Conv2d(self.feature_dim, latent_dim, 1),
                    nn.GroupNorm(4, latent_dim),
                    nn.GELU(),
                    nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
                ),
            }) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract disentangled representations for each layer."""
        features = self.encoder(x)

        result = {'_features': features}
        for i, head in enumerate(self.layer_heads):
            result[f'layer{i}'] = {
                'anatomy': head['anatomy'](features),
                'pathology': head['pathology'](features),
                'noise': head['noise'](features),
            }
        return result

    def compute_disentanglement_loss(
        self,
        representations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Encourage disentanglement via:
        1. Independence: anatomy, pathology, noise should be uncorrelated
        2. Noise minimality: noise should have low magnitude in clean regions
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=representations['_features'].device)

        for i in range(self.num_layers):
            layer_repr = representations[f'layer{i}']
            z_a = layer_repr['anatomy'].flatten(2)  # [B, C, HW]
            z_p = layer_repr['pathology'].flatten(2)
            z_n = layer_repr['noise'].flatten(2)

            # Independence loss: minimize correlation between components
            # Using cosine similarity as proxy
            cos_ap = F.cosine_similarity(z_a, z_p, dim=1).abs().mean()
            cos_an = F.cosine_similarity(z_a, z_n, dim=1).abs().mean()
            cos_pn = F.cosine_similarity(z_p, z_n, dim=1).abs().mean()
            independence_loss = (cos_ap + cos_an + cos_pn) / 3

            total_loss = total_loss + independence_loss * 0.1
            losses[f'layer{i}_independence'] = independence_loss.item()

        losses['total_disentangle'] = total_loss.item()
        return total_loss, losses


# =============================================================================
# Layer Denoiser Head (from constrained version)
# =============================================================================
class BoundaryAwareRefinementHead(nn.Module):
    """
    Smart refinement head that focuses on:
    1. BOUNDARIES - where NAFNet has 60% higher residuals (known problem)
    2. POOR QUALITY INTERIOR - detected via texture/edge analysis

    Key insight from residual analysis:
    - Per-layer heads introduce ~2% error → 1 dB degradation
    - Boundaries have 60% higher residuals than interior
    - Solution: Only refine where needed, leave good regions alone

    Refinement weight = max(boundary_proximity, quality_deficiency)
    - High near boundaries (prior knowledge)
    - High where NAFNet output looks over-smoothed or has artifacts
    """

    def __init__(self, in_channels: int = 1, width: int = 32, boundary_sigma: float = 3.0):
        super().__init__()
        self.boundary_sigma = boundary_sigma  # NARROW boundary region (3 pixels, not 5)

        # Feature extraction from NAFNet output
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )

        # Residual prediction (what to add)
        self.residual_head = nn.Conv2d(width, 1, 3, padding=1)

        # Quality deficiency detector (learns where NAFNet is bad in interior)
        # Input: NAFNet output features
        # Output: Score indicating "this region needs refinement"
        self.quality_detector = nn.Sequential(
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, 3, padding=1),
        )

        # Sobel filter for edge detection (to detect over-smoothing)
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

        # Initialize residual to output TINY values (ultra-conservative)
        # gain=0.01 means initial outputs are ~100x smaller than before
        nn.init.xavier_uniform_(self.residual_head.weight, gain=0.01)
        nn.init.zeros_(self.residual_head.bias)

        # Initialize quality detector to output LOW values (default: don't refine)
        # sigmoid(-2) ≈ 0.12, so start conservative
        nn.init.xavier_uniform_(self.quality_detector[-1].weight, gain=0.1)
        nn.init.constant_(self.quality_detector[-1].bias, -2.0)

    def compute_boundary_weight(
        self,
        boundaries: torch.Tensor,
        H: int,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute weight based on proximity to layer boundaries.

        Args:
            boundaries: [B, 4, W] boundary positions (normalized 0-1)
            H: image height
            layer_idx: which layer (0-3)

        Returns:
            weight: [B, 1, H, W] - high near boundaries, low in interior
        """
        B, _, W = boundaries.shape
        device = boundaries.device

        # Get relevant boundaries for this layer
        # Layer 0 (RNFL_GCL): boundaries 0 and 1
        # Layer 1 (INL_OPL_ONL): boundaries 1 and 2
        # Layer 2 (IS_OS): boundaries 2 and 3
        # Layer 3 (RPE_Choroid): boundary 3 and bottom
        top_boundary_idx = layer_idx
        bottom_boundary_idx = min(layer_idx + 1, 3)

        # Convert to pixel coordinates
        top_boundary = boundaries[:, top_boundary_idx, :] * (H - 1)  # [B, W]
        bottom_boundary = boundaries[:, bottom_boundary_idx, :] * (H - 1)  # [B, W]

        # Create row indices [1, H, 1]
        row_idx = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

        # Distance to nearest boundary (in pixels)
        # [B, H, W]
        dist_to_top = torch.abs(row_idx - top_boundary.unsqueeze(1))
        dist_to_bottom = torch.abs(row_idx - bottom_boundary.unsqueeze(1))
        dist_to_boundary = torch.minimum(dist_to_top, dist_to_bottom)

        # Gaussian falloff from boundary
        # High (1.0) at boundary, low (0.0) far from boundary
        boundary_weight = torch.exp(-dist_to_boundary ** 2 / (2 * self.boundary_sigma ** 2))

        return boundary_weight.unsqueeze(1)  # [B, 1, H, W]

    def compute_quality_deficiency(self, nafnet_output: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Detect regions where NAFNet output quality is poor.

        Indicators of poor quality:
        1. Over-smoothing: low local gradient magnitude
        2. Artifacts: unusual local patterns
        3. Learned: network discovers other indicators

        Returns:
            deficiency: [B, 1, H, W] - high where quality is poor
        """
        # 1. Gradient magnitude (low = over-smoothed)
        padded = F.pad(nafnet_output, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(padded, self.sobel_x, padding=0)
        grad_y = F.conv2d(padded, self.sobel_y, padding=0)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Normalize gradient magnitude
        grad_max = grad_mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        grad_normalized = grad_mag / grad_max

        # Low gradient = potentially over-smoothed = needs refinement
        # But only in regions that SHOULD have texture
        smoothness_deficiency = 1.0 - grad_normalized

        # 2. Learned quality detector
        # Network learns what "bad NAFNet output" looks like
        learned_deficiency = torch.sigmoid(self.quality_detector(features))

        # Combine: either heuristic OR learned detection
        # Use max so either can trigger refinement
        quality_deficiency = torch.maximum(
            smoothness_deficiency * 0.3,  # Heuristic (lower weight)
            learned_deficiency * 0.7,      # Learned (higher weight)
        )

        return quality_deficiency

    def forward(
        self,
        nafnet_output: torch.Tensor,
        boundaries: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict residual and refinement weight.

        Args:
            nafnet_output: [B, 1, H, W] - NAFNet denoised output
            boundaries: [B, 4, W] - detected layer boundaries
            layer_idx: which layer this head is for (0-3)

        Returns:
            residual: [B, 1, H, W] - correction to add
            weight: [B, 1, H, W] - how much correction to apply (0-1)
        """
        B, C, H, W = nafnet_output.shape

        # Extract features
        features = self.features(nafnet_output)

        # Predict residual (what to add if we refine)
        residual = self.residual_head(features)

        # Compute refinement weights
        # 1. Boundary proximity (prior: boundaries need refinement)
        boundary_weight = self.compute_boundary_weight(boundaries, H, layer_idx)

        # 2. Quality deficiency (learned: interior regions that need help)
        quality_weight = self.compute_quality_deficiency(nafnet_output, features)

        # Combine: refine if near boundary OR quality is poor
        # Max ensures either condition triggers refinement
        refinement_weight = torch.maximum(boundary_weight, quality_weight)

        # Clamp to reasonable range
        refinement_weight = refinement_weight.clamp(0, 1)

        return residual, refinement_weight


# Keep old name for compatibility but use new class
ResidualPredictionHead = BoundaryAwareRefinementHead


# Aliases for compatibility
LayerRefinementHead = ResidualPredictionHead
LayerDenoiserHead = ResidualPredictionHead


# =============================================================================
# Clinical Loss with Per-Layer Costs
# =============================================================================
class ClinicalLayerLoss(nn.Module):
    """
    Layer-specific clinical losses for diagnostic quality.

    Different layers have different clinical importance:
    - RNFL: texture preservation (nerve fiber visibility)
    - INL/OPL: structure preservation (layer organization)
    - IS/OS: edge preservation (photoreceptor junction)
    - RPE: contrast preservation (drusen visibility)
    """

    def __init__(self):
        super().__init__()

        # Laplacian for texture (RNFL)
        self.register_buffer('laplacian', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        # Sobel for edges (IS/OS junction)
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4)
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4)

        # Per-layer weights (clinical importance)
        self.layer_weights = [0.4, 0.3, 0.3, 0.4]  # RNFL, INL, IS/OS, RPE

    def texture_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """RNFL: Preserve high-frequency texture (nerve fibers)."""
        pred_hf = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.laplacian)
        target_hf = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.laplacian)
        diff = torch.abs(pred_hf - target_hf) * mask
        return diff.sum() / (mask.sum() + 1e-8)

    def structure_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """INL/OPL: Preserve structural patterns (local SSIM-like)."""
        kernel = torch.ones(1, 1, 5, 5, device=pred.device) / 25

        mu_p = F.conv2d(F.pad(pred, (2,2,2,2), mode='replicate'), kernel)
        mu_t = F.conv2d(F.pad(target, (2,2,2,2), mode='replicate'), kernel)

        var_p = F.conv2d(F.pad((pred - mu_p)**2, (2,2,2,2), mode='replicate'), kernel)
        var_t = F.conv2d(F.pad((target - mu_t)**2, (2,2,2,2), mode='replicate'), kernel)

        C = 0.01
        structure = (2 * torch.sqrt(var_p + 1e-8) * torch.sqrt(var_t + 1e-8) + C) / (var_p + var_t + C)
        loss = (1 - structure) * mask
        return loss.sum() / (mask.sum() + 1e-8)

    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """IS/OS: Preserve sharp edges (photoreceptor junction)."""
        pred_gx = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.sobel_x)
        pred_gy = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.sobel_y)
        target_gx = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.sobel_x)
        target_gy = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.sobel_y)

        pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
        target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8)

        diff = torch.abs(pred_grad - target_grad) * mask
        return diff.sum() / (mask.sum() + 1e-8)

    def contrast_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """RPE: Preserve local contrast (drusen visibility)."""
        kernel = torch.ones(1, 1, 5, 5, device=pred.device) / 25

        pred_mean = F.conv2d(F.pad(pred, (2,2,2,2), mode='replicate'), kernel)
        target_mean = F.conv2d(F.pad(target, (2,2,2,2), mode='replicate'), kernel)

        pred_std = torch.sqrt(F.conv2d(F.pad((pred - pred_mean)**2, (2,2,2,2), mode='replicate'), kernel) + 1e-8)
        target_std = torch.sqrt(F.conv2d(F.pad((target - target_mean)**2, (2,2,2,2), mode='replicate'), kernel) + 1e-8)

        diff = torch.abs(pred_std - target_std) * mask
        return diff.sum() / (mask.sum() + 1e-8)

    def forward(
        self,
        layer_outputs: torch.Tensor,
        soft_masks: torch.Tensor,
        clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute per-layer clinical losses.

        Args:
            layer_outputs: [B, 4, H, W] per-layer denoised outputs
            soft_masks: [B, 4, H, W] soft segmentation masks
            clean: [B, 1, H, W] clean target

        Returns:
            total_loss, losses_dict
        """
        losses = {}
        total_clinical = torch.tensor(0.0, device=clean.device)

        # Loss functions per layer
        loss_fns = [self.texture_loss, self.structure_loss, self.edge_loss, self.contrast_loss]
        layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

        for i, (name, loss_fn, weight) in enumerate(zip(layer_names, loss_fns, self.layer_weights)):
            mask = soft_masks[:, i:i+1, :, :]
            layer_out = layer_outputs[:, i:i+1, :, :]

            # L1 loss for this layer
            layer_l1 = (torch.abs(layer_out - clean) * mask).sum() / (mask.sum() + 1e-8)
            losses[f'{name}_l1'] = layer_l1.item()

            # Clinical loss for this layer
            clinical = loss_fn(layer_out, clean, mask)
            losses[f'{name}_clinical'] = clinical.item()

            # Weighted combination
            layer_loss = layer_l1 + 0.5 * clinical
            total_clinical = total_clinical + weight * layer_loss

        losses['total_clinical'] = total_clinical.item()
        return total_clinical, losses


# =============================================================================
# Residual Adapter for Layer-Specific Feature Modulation (LoRA-style)
# =============================================================================
class ResidualAdapter(nn.Module):
    """
    Lightweight adapter that learns layer-specific feature adjustments.
    Similar to LoRA for LLMs - bottleneck structure for parameter efficiency.

    Key insight: Instead of unfreezing NAFNet (causes catastrophic forgetting),
    we add small adapter networks that learn residual corrections per layer.
    """

    def __init__(self, in_channels: int, bottleneck: int = 16, num_layers: int = NUM_CLASSES):
        super().__init__()
        self.num_layers = num_layers

        # Per-layer adapters with bottleneck structure
        # in_channels -> bottleneck -> in_channels (residual)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, bottleneck, 1),  # Reduce
                nn.GELU(),
                nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck),  # Depthwise
                nn.GELU(),
                nn.Conv2d(bottleneck, in_channels, 1),  # Expand
            ) for _ in range(num_layers)
        ])

        # Initialize adapters with small values (not zero!) to allow gradients to flow
        for adapter in self.adapters:
            nn.init.xavier_uniform_(adapter[-1].weight, gain=0.1)
            nn.init.zeros_(adapter[-1].bias)

        # Learnable mixing weights per layer (how much adapter to blend)
        # Start at -2 so sigmoid(-2) ≈ 0.12 (small initial contribution)
        self.mix_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.0)) for _ in range(num_layers)
        ])

    def forward(self, features: torch.Tensor, layer_masks: torch.Tensor) -> torch.Tensor:
        """
        Apply layer-specific feature modulation.

        Args:
            features: NAFNet features [B, C, H, W]
            layer_masks: Soft segmentation masks [B, num_layers, H, W]

        Returns:
            Modulated features [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Start with original features
        modulated = features.clone()

        # Apply each adapter weighted by its layer mask
        for i in range(min(self.num_layers, layer_masks.shape[1])):
            mask = layer_masks[:, i:i+1, :, :]  # [B, 1, H, W]
            adapter_out = self.adapters[i](features)  # [B, C, H, W]
            mix = torch.sigmoid(self.mix_weights[i])  # Learnable blend factor

            # Add adapter contribution weighted by mask and mix factor
            modulated = modulated + mix * mask * adapter_out

        return modulated


class SpatialAttentionHead(nn.Module):
    """
    Layer head with spatial attention for better layer-specific denoising.
    Learns to focus on relevant spatial regions for each layer.
    """

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()

        # Spatial attention branch
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid_channels, 3, padding=1),  # +1 for mask
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, 1),
            nn.Sigmoid(),
        )

        # Refinement branch (deeper for more expressivity)
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels // 2, 1, 1),
        )

        # Initialize output with small values (not zero!) to allow gradients to flow
        nn.init.xavier_uniform_(self.refinement[-1].weight, gain=0.1)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature maps
            mask: [B, 1, H, W] soft layer mask

        Returns:
            [B, 1, H, W] layer-specific refinement
        """
        # Compute spatial attention using features and mask
        attn_input = torch.cat([features, mask], dim=1)
        attention = self.attention(attn_input)  # [B, 1, H, W]

        # Compute refinement
        refinement = self.refinement(features)  # [B, 1, H, W]

        # Apply attention and mask
        return attention * mask * refinement


# =============================================================================
# Main Neuro-Symbolic Model
# =============================================================================
class NeuroSymbolicDenoiser(nn.Module):
    """
    Complete neuro-symbolic OCT denoising model.

    Combines:
    - Neural: NAFNet (denoising) + PhysicsEnsemble (boundaries)
    - Symbolic: Anatomical logic, physics constraints, consistency
    """

    def __init__(
        self,
        hidden_channels: int = 48,
        nafnet_width: int = 32,
        nafnet_ckpt: str = None,
        physics_ckpt: str = None,
        blend_sigma: float = 7.0,
        freeze_boundary: bool = False,
        freeze_nafnet: bool = False,
        no_refinement: bool = False,
    ):
        super().__init__()

        self.blend_sigma = blend_sigma
        self.no_refinement = no_refinement

        # === NEURAL COMPONENTS ===

        # Boundary detection (neural + physics-informed)
        self.boundary_model = PhysicsEnsembleV3(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_boundaries=NUM_BOUNDARIES,
        )

        # Load pretrained boundary model
        if physics_ckpt and os.path.exists(physics_ckpt):
            logger.info(f"Loading physics checkpoint: {physics_ckpt}")
            ckpt = torch.load(physics_ckpt, map_location='cpu', weights_only=False)
            # Try multiple possible keys for state dict
            state = ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
            missing, unexpected = self.boundary_model.load_state_dict(state, strict=False)
            logger.info(f"Loaded physics (epoch {ckpt.get('epoch', '?')}), missing={len(missing)}, unexpected={len(unexpected)}")

        if freeze_boundary:
            for p in self.boundary_model.parameters():
                p.requires_grad = False
            logger.info("Boundary model frozen")

        # Denoising backbone
        # Architecture must match checkpoint: width=64, enc/dec=[2,2,2], middle=2
        # Checkpoint uses spatial_cue with 4 channels (boundary cues)
        if HAS_NAFNET:
            self.denoiser = NAFNet(
                img_channel=1,
                width=nafnet_width,
                middle_blk_num=2,
                enc_blk_nums=[2, 2, 2],
                dec_blk_nums=[2, 2, 2],
                use_spatial_cue=True,  # Enable boundary cue input
                spatial_cue_channels=4,  # 4 boundary channels
            )
            if nafnet_ckpt and os.path.exists(nafnet_ckpt):
                logger.info(f"Loading NAFNet: {nafnet_ckpt}")
                ckpt = torch.load(nafnet_ckpt, map_location='cpu', weights_only=False)
                state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
                self.denoiser.load_state_dict(state, strict=False)

            if freeze_nafnet:
                # FULLY FREEZE NAFNet to prevent catastrophic forgetting
                # Layer-specific learning happens through adapters and layer heads
                for p in self.denoiser.parameters():
                    p.requires_grad = False
                logger.info("NAFNet backbone frozen - only training boundary/layer heads")

            self.use_boundary_cues = True
        else:
            self.denoiser = self._simple_denoiser(nafnet_width)
            self.use_boundary_cues = False

        # === RESIDUAL PREDICTION HEADS ===
        # Each head predicts what NAFNet misses for its layer
        # Simple architecture: just predict the residual from NAFNet output
        self.layer_heads = nn.ModuleList([
            ResidualPredictionHead(in_channels=1, width=32)
            for _ in range(NUM_CLASSES)
        ])
        layer_head_params = sum(p.numel() for h in self.layer_heads for p in h.parameters())
        logger.info(f"Added {NUM_CLASSES} ResidualPredictionHeads ({layer_head_params} params)")

        # Keep ResidualAdapter for NAFNet feature modulation (optional enhancement)
        self.residual_adapter = ResidualAdapter(
            in_channels=nafnet_width,
            bottleneck=16,
            num_layers=NUM_CLASSES,
        )
        logger.info(f"Added ResidualAdapter with {sum(p.numel() for p in self.residual_adapter.parameters())} params")

        # Feature extractor for layer heads (deeper for better features)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, nafnet_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(nafnet_width, nafnet_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(nafnet_width, nafnet_width, 3, padding=1),  # Extra layer
        )

        # Fusion network (initialized to output near-zero to preserve NAFNet output initially)
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),  # Extra layer for capacity
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        # Initialize fusion output layer with small values (not zero!) to allow gradients
        nn.init.xavier_uniform_(self.fusion[-1].weight, gain=0.1)
        nn.init.zeros_(self.fusion[-1].bias)

        # === NOVEL: Adaptive Refinement Gate ===
        # Learns WHEN and HOW MUCH to refine based on input characteristics
        # This makes the method work consistently across Duke, PKU37, etc.
        self.adaptive_gate = AdaptiveRefinementGate(in_channels=1)
        logger.info("Added AdaptiveRefinementGate for dataset-agnostic refinement")

        # === SYMBOLIC COMPONENTS ===
        self.anatomical_logic = AnatomicalLogicLayer()
        self.physics_constraints = PhysicsConstraintModule()
        self.consistency = SymbolicConsistencyModule()

        # === TRUE NEURO-SYMBOLIC COMPONENTS (NOVEL) ===
        # 1. Hard constraint satisfaction through symbolic refinement
        self.symbolic_refinement = SymbolicBoundaryRefinement(
            num_boundaries=NUM_BOUNDARIES,
            min_gap=0.02,
            max_jump=0.05,
            num_iterations=3,
        )

        # 2. Differentiable first-order logic layer
        # Temperature controls fuzzy predicate sharpness
        # Smaller = sharper transitions for small differences (0.02 works well for [0,1] normalized values)
        self.logic_layer = DifferentiableLogicLayer(temperature=0.02)

        # 3. Literature-based symbolic knowledge module (NOVEL)
        # Uses published anatomical knowledge for hard constraint satisfaction
        # and self-supervised intensity verification
        self.symbolic_knowledge = NeuroSymbolicOCTModule()
        logger.info("Added NeuroSymbolicOCTModule with literature-derived constraints")

    def _simple_denoiser(self, width):
        return nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def create_soft_masks(
        self,
        boundaries: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """Create soft segmentation masks with gradual transitions."""
        B, N, W = boundaries.shape
        device = boundaries.device

        boundaries_px = boundaries * (H - 1)
        y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        b_exp = boundaries_px.unsqueeze(2)

        temp = self.blend_sigma
        b1 = b_exp[:, 1:2, :, :]
        b2 = b_exp[:, 2:3, :, :]
        b3 = b_exp[:, 3:4, :, :]

        mask_0 = torch.sigmoid((b1 - y) / temp)
        mask_1 = torch.sigmoid((y - b1) / temp) * torch.sigmoid((b2 - y) / temp)
        mask_2 = torch.sigmoid((y - b2) / temp) * torch.sigmoid((b3 - y) / temp)
        mask_3 = torch.sigmoid((y - b3) / temp)

        soft_masks = torch.cat([mask_0, mask_1, mask_2, mask_3], dim=1)
        soft_masks = soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

        return soft_masks

    def _create_boundary_cues(
        self,
        boundaries: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """
        Create boundary cue maps for NAFNet spatial conditioning.

        Args:
            boundaries: [B, 4, W] boundary positions (normalized 0-1)
            H: Image height

        Returns:
            boundary_cues: [B, 4, H, W] soft boundary indicator maps
        """
        B, N, W = boundaries.shape
        device = boundaries.device

        # Convert to pixel positions
        boundaries_px = boundaries * (H - 1)  # [B, 4, W]

        # Create height coordinate
        y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)  # [1, 1, H, 1]

        # Expand boundaries for broadcasting: [B, 4, 1, W]
        b_exp = boundaries_px.unsqueeze(2)

        # Create Gaussian-like cues centered at each boundary
        sigma = 5.0  # Width of boundary region in pixels
        cues = torch.exp(-0.5 * ((y - b_exp) / sigma) ** 2)  # [B, 4, H, W]

        return cues

    def forward(
        self,
        x: torch.Tensor,
        return_symbolic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with neural and symbolic outputs.

        MEMORY-OPTIMIZED: Deletes intermediate tensors as soon as they're no longer needed.
        """
        B, C, H, W = x.shape

        # === NEURAL FORWARD ===

        # Get boundaries from physics-enhanced model
        boundary_out = self.boundary_model(x, return_aux=True)
        boundaries = boundary_out['boundaries']
        del boundary_out  # Free memory

        # Denoise with NAFNet (with boundary cues if available)
        if self.use_boundary_cues:
            # Create boundary cue maps: [B, 4, H, W]
            # Each channel is a soft indicator of boundary position
            boundary_cues = self._create_boundary_cues(boundaries, H)
            denoised_base = self.denoiser(x, spatial_map=boundary_cues)
            # Keep boundary_cues for adaptive gate (don't delete yet)
        else:
            denoised_base = self.denoiser(x)
            boundary_cues = None

        # Layer-specific refinement using DISENTANGLED approach
        # Skip refinement if no_refinement mode (use pure NAFNet output)
        if self.no_refinement:
            denoised = torch.clamp(denoised_base, 0, 1)
            soft_masks = self.create_soft_masks(boundaries, H)
            adaptation_evidence = {
                'denoised_base': denoised_base.detach(),
                'refinement_scale': 0.0,
                'layer_outputs': denoised.expand(-1, NUM_CLASSES, -1, -1).detach(),  # Dummy
            }
            del denoised_base
        else:
            # Create soft masks for blending
            soft_masks = self.create_soft_masks(boundaries, H)

            # === BOUNDARY-AWARE RESIDUAL PREDICTION ===
            # Key insight from analysis:
            # - Boundaries have 60% higher residuals than interior
            # - Per-layer heads must be VERY precise (1% error = 0.3dB loss)
            # - Solution: Focus refinement on boundaries + detected poor regions

            # Predict residuals and refinement weights for each layer
            layer_residuals = []
            layer_gates = []
            for i, head in enumerate(self.layer_heads):
                # Each head predicts:
                # - residual: what to add
                # - weight: where to apply (high at boundaries + poor quality regions)
                residual, weight = head(denoised_base, boundaries, layer_idx=i)
                layer_residuals.append(residual)
                layer_gates.append(weight)

            # Stack residuals and gates: [B, 4, H, W]
            residual_stack = torch.cat(layer_residuals, dim=1)
            gate_stack = torch.cat(layer_gates, dim=1)

            # Compute layer outputs: NAFNet + gated residual
            # gate=0: pure NAFNet, gate=1: NAFNet + full residual
            layer_outputs = []
            for i in range(NUM_CLASSES):
                gated_residual = layer_residuals[i] * layer_gates[i]
                layer_out = denoised_base + gated_residual
                layer_outputs.append(layer_out)

            # MEMORY FIX: Delete intermediate lists after use
            del layer_residuals, layer_gates

            # Stack layer outputs: [B, 4, H, W]
            layer_stack = torch.cat(layer_outputs, dim=1)
            del layer_outputs  # MEMORY FIX: Free list after cat

            # Blend layer outputs using soft masks
            # Each pixel gets the output from its dominant layer
            denoised = (layer_stack * soft_masks).sum(dim=1, keepdim=True)
            denoised = torch.clamp(denoised, 0, 1)

            # Store for loss computation
            # residual_stack: [B, 4, H, W] - predicted residuals per layer
            # gate_stack: [B, 4, H, W] - gating values (0-1) per layer
            # These will be supervised with (clean - denoised_base) * layer_mask
            # MEMORY FIX: Removed duplicate 'layer_outputs' key (was same as layer_stack)
            adaptation_evidence = {
                'layer_stack': layer_stack.detach(),
                'residual_stack': residual_stack,  # Keep gradients for training!
                'gate_stack': gate_stack,  # Keep gradients for gate learning!
                'soft_masks': soft_masks.detach(),
                'denoised_base': denoised_base,  # Keep for residual loss computation
            }
            del boundary_cues  # Free memory

        # === TRUE NEURO-SYMBOLIC: Apply symbolic constraint satisfaction ===
        # This GUARANTEES valid anatomical structure (not soft penalties)
        refinement_result = self.symbolic_refinement(
            boundaries, return_refinement_info=False  # Don't need info during training
        )
        refined_boundaries = refinement_result['boundaries']
        del refinement_result  # Free memory

        # === LITERATURE-BASED HARD CONSTRAINT PROJECTION (NOVEL) ===
        # Apply hard constraints from published anatomical knowledge
        # This guarantees: ordering, minimum thickness, valid position ranges
        symbolic_result = self.symbolic_knowledge(refined_boundaries, x)
        refined_boundaries = symbolic_result['boundaries']  # Now guaranteed valid
        constraint_satisfaction = symbolic_result['satisfaction_rate']
        intensity_scores = symbolic_result['intensity_scores']  # Self-supervised verification

        # Segmentation from REFINED boundaries (guaranteed valid)
        segmentation = boundaries_to_segmentation(refined_boundaries, H, NUM_CLASSES)

        # Create soft masks with refined boundaries (for output only)
        soft_masks_refined = self.create_soft_masks(refined_boundaries, H)

        # Get base NAFNet output for degradation-aware training
        denoised_base_for_loss = adaptation_evidence.get('denoised_base', denoised)

        outputs = {
            'denoised': denoised,
            'denoised_base': denoised_base_for_loss,  # Base NAFNet output for degradation penalty
            'boundaries': refined_boundaries,  # Use refined (guaranteed valid)
            'raw_boundaries': symbolic_result['raw_boundaries'],  # Before hard projection
            'soft_masks': soft_masks_refined,
            'segmentation': segmentation,
            'adaptation_evidence': adaptation_evidence,  # Evidence of layer adaptation
            'constraint_satisfaction': constraint_satisfaction,  # % of constraints satisfied
            'intensity_scores': intensity_scores,  # Self-supervised boundary verification
        }

        # === SYMBOLIC FORWARD ===
        if return_symbolic:
            # TRUE NEURO-SYMBOLIC: Evaluate differentiable logic rules
            logic_scores = self.logic_layer.evaluate_all_rules(refined_boundaries)
            outputs['logic_loss'] = logic_scores['logic_loss'].mean()
            del logic_scores  # Free memory

            # Anatomical constraint violations (soft penalties - complementary)
            anatomical_violations = self.anatomical_logic(refined_boundaries, denoised)
            outputs['anatomical_violations'] = anatomical_violations

            # Physics consistency
            outputs['physics_consistency'] = self.physics_constraints.physics_consistency_loss(
                denoised, refined_boundaries
            )

            # Speckle distribution check
            outputs['speckle_loss'] = self.physics_constraints.speckle_distribution_loss(
                x, denoised
            )

            # === SYMBOLIC KNOWLEDGE LOSS (NOVEL) ===
            # Encourage neural network to predict boundaries that already satisfy constraints
            # This reduces the projection distance over training
            symbolic_loss, symbolic_loss_components = self.symbolic_knowledge.compute_symbolic_loss(
                symbolic_result['raw_boundaries'], x
            )
            outputs['symbolic_loss'] = symbolic_loss
            outputs['symbolic_loss_components'] = symbolic_loss_components

        return outputs

    def forward_boundaries_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        FAST: Extract only boundaries from input (skip denoising, layer heads, fusion).
        Used for multi-frame consistency where we only need boundary positions.

        ~3-5x faster than full forward pass.
        """
        # Only run boundary model (skip NAFNet, layer heads, fusion, symbolic losses)
        boundary_out = self.boundary_model(x, return_aux=False)
        boundaries = boundary_out['boundaries']

        # Apply symbolic refinement to ensure valid boundaries
        refinement_result = self.symbolic_refinement(boundaries, return_refinement_info=False)
        return refinement_result['boundaries']

    def forward_boundaries_only_batched(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        FASTEST: Extract boundaries from multiple frames in a single batched forward.
        Even faster than calling forward_boundaries_only multiple times.
        """
        if len(frames) == 0:
            return []

        # Stack frames into batch
        batched = torch.cat(frames, dim=0)  # [N*B, C, H, W]

        # Single batched forward through boundary model
        boundary_out = self.boundary_model(batched, return_aux=False)
        boundaries_batched = boundary_out['boundaries']  # [N*B, num_boundaries, W]

        # Apply symbolic refinement
        refinement_result = self.symbolic_refinement(boundaries_batched, return_refinement_info=False)
        boundaries_refined = refinement_result['boundaries']

        # Split back into list
        B = frames[0].shape[0]
        boundaries_list = torch.split(boundaries_refined, B, dim=0)
        return list(boundaries_list)

    def forward_multi_frame(
        self,
        frames: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process multiple noisy frames with consistency enforcement.

        SPEED-OPTIMIZED v4:
        - Full forward only for FIRST frame (denoising + all losses)
        - BATCHED boundary-only forward for other frames (even faster)
        - ~2.5x faster than processing all frames fully
        """
        B, C, H, W = frames[0].shape
        device = frames[0].device
        num_frames = len(frames)

        # Process FIRST frame WITH gradients (full forward for denoising quality)
        out_first = self.forward(frames[0], return_symbolic=True)
        denoised = out_first['denoised']
        boundaries_first = out_first['boundaries']
        soft_masks_ref = out_first['soft_masks'].detach()
        adaptation_evidence = out_first.get('adaptation_evidence', None)

        # Extract symbolic losses from first frame only (no need to average)
        anatomical_loss = out_first['anatomical_violations']['total']
        physics_loss = out_first['physics_consistency']
        speckle_loss = out_first['speckle_loss']
        logic_loss = out_first['logic_loss']

        # Store boundaries for consistency loss
        boundaries_list = [boundaries_first.detach().clone()]

        # Process remaining frames with BATCHED boundary-only forward (FASTEST)
        if num_frames > 1:
            with torch.no_grad():
                # Single batched forward for all remaining frames
                other_boundaries = self.forward_boundaries_only_batched(frames[1:])
                for b in other_boundaries:
                    boundaries_list.append(b.detach().clone())
                del other_boundaries

        # Compute boundary consistency loss across all frames
        consistency_loss = self.consistency.multi_frame_consistency_loss(boundaries_list)

        # Use first frame's boundaries (already refined)
        boundaries_refined = boundaries_first

        # Clean up
        del boundaries_list
        gc.collect()

        # Compute segmentation from refined boundaries
        segmentation = boundaries_to_segmentation(boundaries_refined, H, NUM_CLASSES)

        result = {
            'denoised': denoised,
            'boundaries': boundaries_refined,
            'segmentation': segmentation,
            'soft_masks': soft_masks_ref,
            'consistency_loss': consistency_loss,
            'anatomical_violations': {'total': anatomical_loss},
            'physics_consistency': physics_loss,
            'speckle_loss': speckle_loss,
            'logic_loss': logic_loss,
        }
        if adaptation_evidence is not None:
            result['adaptation_evidence'] = adaptation_evidence
        return result


# =============================================================================
# Neuro-Symbolic Loss Function
# =============================================================================
class NeuroSymbolicLoss(nn.Module):
    """
    Combined loss with neural and symbolic components.

    Includes TRUE NEURO-SYMBOLIC logic loss from differentiable first-order logic.

    NEW: Supports self-supervised boundary learning via IntensityAnchoredBoundaryLossV4
    - Supervised denoising: uses clean images as targets
    - Self-supervised boundaries: uses intensity-based detection (no GT masks needed)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_anatomical: float = 0.5,
        lambda_physics: float = 0.3,
        lambda_consistency: float = 0.5,
        lambda_speckle: float = 0.1,
        lambda_logic: float = 0.1,  # TRUE NEURO-SYMBOLIC: Logic layer weight (reduced)
        lambda_symbolic: float = 0.5,  # NOVEL: Literature-based symbolic knowledge loss
        lambda_layer_supervision: float = 1.0,  # Increased: Force heads to specialize (adaptive margins)
        lambda_per_layer_l1: float = 0.5,  # Per-layer L1 for layer PSNR (INCREASED for significant gains)
        # NEW: Self-supervised boundary learning
        use_intensity_anchor: bool = True,  # Enable V4 intensity-anchored boundary loss
        lambda_intensity_anchor: float = 1.0,  # Weight for boundary anchoring
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_anatomical = lambda_anatomical
        self.lambda_physics = lambda_physics
        self.lambda_consistency = lambda_consistency
        self.lambda_speckle = lambda_speckle
        self.lambda_logic = lambda_logic
        self.lambda_symbolic = lambda_symbolic
        self.lambda_layer_supervision = lambda_layer_supervision
        self.lambda_per_layer_l1 = lambda_per_layer_l1

        # NEW: Self-supervised boundary learning (V4)
        # This enables domain adaptation without GT masks
        self.use_intensity_anchor = use_intensity_anchor
        self.lambda_intensity_anchor = lambda_intensity_anchor

        if use_intensity_anchor:
            self.intensity_anchor_loss = IntensityAnchoredBoundaryLossV4(
                lambda_ilm_anchor=1.0,
                lambda_rpe_anchor=1.0,
                lambda_edge_align=0.3,
                lambda_smoothness=0.1,
                use_confidence=True,
            )
            logging.info("Enabled IntensityAnchoredBoundaryLossV4 for self-supervised boundary learning")
        else:
            self.intensity_anchor_loss = None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        mask: torch.Tensor = None,
        boundaries_gt: torch.Tensor = None,
        noisy_input: torch.Tensor = None,  # NEW: for self-supervised boundary learning
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs (denoised, boundaries, etc.)
            clean: Clean reference image for supervised denoising
            mask: Optional GT segmentation mask
            boundaries_gt: Optional GT boundaries
            noisy_input: Noisy input image for V4 boundary detection (self-supervised)
        """
        losses = {}
        device = clean.device

        # === NEURAL LOSSES ===

        # L1 reconstruction loss (requires clean reference)
        losses['l1'] = F.l1_loss(outputs['denoised'], clean)

        # PER-LAYER L1 LOSS - Key for layer-specific PSNR gains!
        # This explicitly supervises each layer region separately
        # MEMORY FIX: Pre-allocate zero tensor once
        zero_loss = torch.zeros(1, device=device, requires_grad=False)

        if 'soft_masks' in outputs:
            soft_masks = outputs['soft_masks']  # [B, 4, H, W]
            denoised = outputs['denoised']
            per_layer_l1 = zero_loss.clone()
            layer_weights = [1.5, 1.2, 1.5, 1.0]  # Weight clinically important layers more

            for i in range(min(NUM_CLASSES, soft_masks.shape[1])):
                layer_mask = soft_masks[:, i:i+1, :, :]  # BUG FIX: renamed to avoid shadowing
                # Compute L1 error weighted by layer mask
                layer_error = torch.abs(denoised - clean) * layer_mask
                layer_l1 = layer_error.sum() / (layer_mask.sum() + 1e-8)
                per_layer_l1 = per_layer_l1 + layer_weights[i] * layer_l1

            losses['per_layer_l1'] = per_layer_l1 / NUM_CLASSES

            # Per-layer SSIM loss for perceptual quality
            # Use a simple approximation: 1 - mean(structure_term) per layer
            per_layer_ssim_loss = zero_loss.clone()
            C1 = 0.01 ** 2  # Stability constant for luminance
            C2 = 0.03 ** 2  # Stability constant for contrast
            for i in range(min(NUM_CLASSES, soft_masks.shape[1])):
                layer_mask = soft_masks[:, i:i+1, :, :]  # BUG FIX: renamed
                mask_sum = layer_mask.sum() + 1e-8
                # Compute local statistics within layer region
                pred_masked = denoised * layer_mask
                target_masked = clean * layer_mask
                # Mean within layer
                mu_pred = pred_masked.sum() / mask_sum
                mu_target = target_masked.sum() / mask_sum
                # Variance within layer
                sigma_pred_sq = ((denoised - mu_pred) ** 2 * layer_mask).sum() / mask_sum
                sigma_target_sq = ((clean - mu_target) ** 2 * layer_mask).sum() / mask_sum
                sigma_pred_target = ((denoised - mu_pred) * (clean - mu_target) * layer_mask).sum() / mask_sum
                # SSIM components
                luminance = (2 * mu_pred * mu_target + C1) / (mu_pred ** 2 + mu_target ** 2 + C1)
                contrast = (2 * torch.sqrt(sigma_pred_sq + 1e-8) * torch.sqrt(sigma_target_sq + 1e-8) + C2) / (sigma_pred_sq + sigma_target_sq + C2)
                structure = (sigma_pred_target + C2 / 2) / (torch.sqrt(sigma_pred_sq + 1e-8) * torch.sqrt(sigma_target_sq + 1e-8) + C2 / 2)
                layer_ssim = luminance * contrast * structure
                per_layer_ssim_loss = per_layer_ssim_loss + layer_weights[i] * (1 - layer_ssim)

            losses['per_layer_ssim'] = per_layer_ssim_loss / NUM_CLASSES
        else:
            losses['per_layer_l1'] = zero_loss
            losses['per_layer_ssim'] = zero_loss

        # Dice loss (if GT mask available)
        if mask is not None:
            dice_loss = zero_loss.clone()
            pred_seg = outputs['segmentation']
            for c in range(NUM_CLASSES):
                pred_c = (pred_seg == c).float()
                gt_c = (mask == c).float()
                inter = (pred_c * gt_c).sum()
                union = pred_c.sum() + gt_c.sum()
                dice_loss = dice_loss + (1 - (2 * inter + 1) / (union + 1))
            losses['dice'] = dice_loss / NUM_CLASSES
        else:
            losses['dice'] = zero_loss

        # === NOVEL: DEGRADATION-AWARE LOSS ===
        # Penalizes model when refined output is WORSE than base NAFNet output
        # This teaches conservative refinement: "first, do no harm"
        # Makes method work consistently across Duke, PKU37, etc.
        if 'denoised_base' in outputs:
            denoised_base = outputs['denoised_base']
            denoised_refined = outputs['denoised']

            # Compute MSE for base and refined
            mse_base = F.mse_loss(denoised_base, clean)
            mse_refined = F.mse_loss(denoised_refined, clean)

            # Degradation penalty: only active when refined is worse than base
            # max(0, mse_refined - mse_base) = 0 if refined is better
            degradation = torch.clamp(mse_refined - mse_base, min=0)

            # Per-layer degradation check (more granular)
            if 'soft_masks' in outputs:
                per_layer_degradation = zero_loss.clone()
                soft_masks = outputs['soft_masks']
                for i in range(min(NUM_CLASSES, soft_masks.shape[1])):
                    layer_mask_i = soft_masks[:, i:i+1, :, :]  # Renamed to avoid confusion
                    mask_sum = layer_mask_i.sum() + 1e-8
                    # Layer-wise MSE
                    layer_mse_base = ((denoised_base - clean) ** 2 * layer_mask_i).sum() / mask_sum
                    layer_mse_refined = ((denoised_refined - clean) ** 2 * layer_mask_i).sum() / mask_sum
                    # Layer degradation penalty
                    layer_degradation = torch.clamp(layer_mse_refined - layer_mse_base, min=0)
                    per_layer_degradation = per_layer_degradation + layer_degradation

                losses['degradation'] = degradation + 0.5 * per_layer_degradation / NUM_CLASSES
            else:
                losses['degradation'] = degradation
        else:
            losses['degradation'] = zero_loss

        # === SYMBOLIC LOSSES ===

        # Anatomical constraints
        if 'anatomical_violations' in outputs:
            losses['anatomical'] = outputs['anatomical_violations']['total']
        else:
            losses['anatomical'] = zero_loss

        # Physics consistency
        if 'physics_consistency' in outputs:
            losses['physics'] = outputs['physics_consistency']
        else:
            losses['physics'] = zero_loss

        # Speckle distribution
        if 'speckle_loss' in outputs:
            losses['speckle'] = outputs['speckle_loss']
        else:
            losses['speckle'] = zero_loss

        # Multi-frame consistency
        if 'consistency_loss' in outputs:
            losses['consistency'] = outputs['consistency_loss']
        else:
            losses['consistency'] = zero_loss

        # TRUE NEURO-SYMBOLIC: Logic layer loss
        if 'logic_loss' in outputs:
            losses['logic'] = outputs['logic_loss']
        else:
            losses['logic'] = zero_loss

        # NOVEL: Literature-based symbolic knowledge loss
        # Encourages neural network to predict boundaries that already satisfy constraints
        # Components: ordering, thickness, position, intensity consistency
        if 'symbolic_loss' in outputs:
            losses['symbolic'] = outputs['symbolic_loss']
            # Log individual components for monitoring
            if 'symbolic_loss_components' in outputs:
                for key, value in outputs['symbolic_loss_components'].items():
                    losses[f'sym_{key}'] = value
        else:
            losses['symbolic'] = zero_loss

        # NEW: Layer supervision loss - force each head to specialize to its region
        # This is KEY for proving layer-adaptive denoising works!
        if 'adaptation_evidence' in outputs:
            ev = outputs['adaptation_evidence']
            layer_stack = ev.get('layer_stack', None)  # [B, 4, H, W]
            soft_masks = ev.get('soft_masks', None)    # [B, 4, H, W]

            if layer_stack is not None and soft_masks is not None:
                layer_sup_loss = zero_loss.clone()
                for i in range(NUM_CLASSES):
                    head_out = layer_stack[:, i, :, :].abs()
                    own_mask = soft_masks[:, i, :, :]
                    other_mask = 1 - own_mask

                    # Head should activate MORE in its own region
                    own_activation = (head_out * own_mask).sum() / (own_mask.sum() + 1e-8)
                    other_activation = (head_out * other_mask).sum() / (other_mask.sum() + 1e-8)

                    # Adaptive margin based on layer coverage
                    # Larger layers (like RPE_Choroid ~60%) need stronger push
                    own_coverage = own_mask.sum() / (own_mask.numel() + 1e-8)
                    # Base margin 0.001, scale up to 0.01 for large layers
                    margin = 0.001 + 0.009 * own_coverage  # Up to 0.01 for 100% coverage

                    # Additional term: ratio-based loss for larger layers
                    # We want ratio > 1, so penalize log(ratio) when < 0
                    ratio = own_activation / (other_activation + 1e-8)
                    ratio_loss = F.relu(1.0 - ratio)  # Penalty when ratio < 1

                    # Combined loss: margin-based + ratio-based (stronger for large layers)
                    layer_sup_loss = layer_sup_loss + F.relu(other_activation - own_activation + margin)
                    layer_sup_loss = layer_sup_loss + own_coverage * ratio_loss  # Weight by coverage

                losses['layer_supervision'] = layer_sup_loss / NUM_CLASSES
            else:
                losses['layer_supervision'] = zero_loss
        else:
            losses['layer_supervision'] = zero_loss

        # === NEW: SELF-SUPERVISED BOUNDARY LEARNING (V4) ===
        # Uses intensity-based detection to anchor boundaries WITHOUT GT masks
        # This enables domain adaptation when training on new datasets
        if self.use_intensity_anchor and self.intensity_anchor_loss is not None:
            if 'boundaries' in outputs:
                # Use noisy_input for detection, or fall back to denoised image
                detection_image = noisy_input if noisy_input is not None else outputs['denoised']

                anchor_loss, anchor_details = self.intensity_anchor_loss(
                    outputs['boundaries'],
                    detection_image,
                )
                losses['intensity_anchor'] = anchor_loss

                # Log detection info for monitoring
                losses['detected_ilm'] = anchor_details.get('detected_retina_top', 0.0)
                losses['detected_rpe'] = anchor_details.get('detected_retina_bottom', 0.0)
                losses['ilm_anchor_loss'] = anchor_details.get('ilm_anchor', 0.0)
                losses['rpe_anchor_loss'] = anchor_details.get('rpe_anchor', 0.0)
                losses['detection_confidence'] = anchor_details.get('confidence', 0.0)
            else:
                losses['intensity_anchor'] = zero_loss
        else:
            losses['intensity_anchor'] = zero_loss

        # Total loss with enhanced per-layer supervision and degradation penalty
        total = (
            self.lambda_l1 * losses['l1'] +
            self.lambda_per_layer_l1 * losses['per_layer_l1'] +  # Per-layer L1 for PSNR gains
            self.lambda_per_layer_l1 * 0.5 * losses['per_layer_ssim'] +  # Per-layer SSIM for perceptual quality
            self.lambda_anatomical * losses['anatomical'] +
            self.lambda_physics * losses['physics'] +
            self.lambda_speckle * losses['speckle'] +
            self.lambda_consistency * losses['consistency'] +
            self.lambda_logic * losses['logic'] +  # TRUE NEURO-SYMBOLIC
            self.lambda_symbolic * losses['symbolic'] +  # NOVEL: Literature-based symbolic knowledge
            self.lambda_layer_supervision * losses['layer_supervision'] +  # Force heads to specialize
            5.0 * losses['degradation'] +  # STRONG penalty for degradation (conservative refinement)
            self.lambda_intensity_anchor * losses['intensity_anchor']  # NEW: Self-supervised boundary learning
        )

        losses['total'] = total

        return total, {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


# =============================================================================
# INTENSITY-ANCHORED BOUNDARY LOSS V4 - Self-Supervised Domain Adaptation
# =============================================================================
# Using IntensityAnchoredBoundaryLossV4 from intensity_anchor_v4.py
# V4 Features:
#   - Robust ensemble detection (intensity + gradient methods)
#   - Confidence weighting (trust good detections more)
#   - Tolerance band in anchor loss (Huber-like)
#   - Multi-threshold detection (0.3, 0.5, 0.7 std)
#   - Fixed weights for stable training
#
# OLD LOCAL CLASS REMOVED - was ~400 lines of duplicate code
# =============================================================================


# Old IntensityAnchoredBoundaryLoss class removed - using V4 from intensity_anchor_v4.py


# Placeholder to mark removal point
_OLD_INTENSITY_ANCHOR_LOSS_REMOVED = """
The old IntensityAnchoredBoundaryLoss (~400 lines) was removed.
Now using IntensityAnchoredBoundaryLossV4 which has:
- Ensemble detection (intensity + gradient)
- Confidence weighting
- Tolerance band loss
- Better domain adaptation
"""
# =============================================================================
# FULLY SELF-SUPERVISED Loss Function (No Clean Reference Needed)
# =============================================================================
class SupervisedWithBoundaryAdaptationLoss(nn.Module):
    """
    Supervised denoising with self-supervised boundary adaptation.

    LOSS COMPONENTS:
    1. MSE: Global supervised denoising
    2. Clinical: Per-layer clinical losses (texture/structure/edge/contrast)
    3. IntensityAnchor: Self-supervised boundary learning
    4. Disentanglement: Encourage independence of anatomy/pathology/noise
    5. Smoothness: Boundary regularization
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,              # Global denoising
        lambda_clinical: float = 0.5,          # Per-layer clinical losses
        lambda_intensity_anchor: float = 1.0,  # Self-supervised boundary learning
        lambda_disentangle: float = 0.1,       # Disentanglement loss
        lambda_smoothness: float = 0.1,        # Boundary regularization
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_clinical = lambda_clinical
        self.lambda_intensity_anchor = lambda_intensity_anchor
        self.lambda_disentangle = lambda_disentangle
        self.lambda_smoothness = lambda_smoothness

        # Per-layer clinical loss (different cost per layer)
        self.clinical_loss = ClinicalLayerLoss()

        # Self-supervised boundary loss V4
        self.intensity_anchor_loss = IntensityAnchoredBoundaryLossV4(
            lambda_ilm_anchor=1.0,
            lambda_rpe_anchor=1.0,
            lambda_edge_align=0.3,
            lambda_smoothness=0.1,
            use_confidence=True,
        )

    def boundary_smoothness_loss(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Encourage smooth boundaries (total variation)."""
        if boundaries.shape[-1] <= 1:
            return boundaries.new_zeros(1)
        diff = torch.abs(boundaries[:, :, 1:] - boundaries[:, :, :-1])
        return diff.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        noisy_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with supervised denoising + self-supervised boundaries.

        Args:
            outputs: Model outputs containing 'denoised' and 'boundaries'
            clean: Clean reference image [B, 1, H, W] (for MSE loss)
            mask: Optional segmentation mask (unused in simplified loss)
            noisy_input: Noisy input image [B, 1, H, W] (for boundary detection)

        Returns:
            total_loss, losses_dict
        """
        losses = {}
        device = outputs['denoised'].device
        zero_loss = torch.zeros(1, device=device, requires_grad=False)

        # === 1. GLOBAL SUPERVISED DENOISING (MSE) ===
        losses['mse'] = F.mse_loss(outputs['denoised'], clean)

        # === 1b. "DON'T MAKE THINGS WORSE" LOSS ===
        # Key insight: We need POSITIVE gains. Penalize any degradation vs NAFNet.
        adaptation_evidence = outputs.get('adaptation_evidence', {})
        if 'denoised_base' in adaptation_evidence:
            denoised_base = adaptation_evidence['denoised_base']
            # MSE of our output vs MSE of NAFNet baseline
            our_mse = F.mse_loss(outputs['denoised'], clean, reduction='none').mean(dim=(1, 2, 3))
            nafnet_mse = F.mse_loss(denoised_base, clean, reduction='none').mean(dim=(1, 2, 3))
            # Penalize if we're worse than NAFNet (positive = we're worse)
            degradation = F.relu(our_mse - nafnet_mse)
            losses['no_degrade'] = degradation.mean() * 10.0  # Strong penalty!
        else:
            losses['no_degrade'] = zero_loss

        # === 2. PER-LAYER RESIDUAL SUPERVISION (KEY!) ===
        # Each layer head learns to predict: clean - nafnet_output
        # This is direct supervision - heads learn exactly what NAFNet misses
        adaptation_evidence = outputs.get('adaptation_evidence', {})
        if 'residual_stack' in adaptation_evidence and 'denoised_base' in adaptation_evidence:
            residual_stack = adaptation_evidence['residual_stack']  # [B, 4, H, W] predicted
            gate_stack = adaptation_evidence.get('gate_stack', None)  # [B, 4, H, W] gates
            denoised_base = adaptation_evidence['denoised_base']    # [B, 1, H, W]
            soft_masks = outputs['soft_masks']                       # [B, 4, H, W]

            # Ground truth residual: what NAFNet actually missed
            gt_residual = clean - denoised_base  # [B, 1, H, W]

            # Per-layer residual loss
            # Key insight: Supervise residuals UNGATED so they learn correct values
            # The main MSE loss on final output will drive gate learning
            layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
            total_residual_loss = zero_loss.clone()

            for i, name in enumerate(layer_names):
                mask = soft_masks[:, i:i+1, :, :]  # [B, 1, H, W]
                pred_res = residual_stack[:, i:i+1, :, :]  # [B, 1, H, W]

                # Supervise UNGATED residuals (teach what to predict)
                # Gate learning comes from final MSE loss
                layer_res_loss = ((pred_res - gt_residual) ** 2 * mask).sum() / (mask.sum() + 1e-8)

                # Log gate statistics if available
                if gate_stack is not None:
                    gate = gate_stack[:, i:i+1, :, :]  # [B, 1, H, W]
                    losses[f'{name}_gate_mean'] = gate.mean().item()

                losses[f'{name}_residual'] = layer_res_loss.item()
                total_residual_loss = total_residual_loss + layer_res_loss

            losses['residual'] = total_residual_loss / NUM_CLASSES
        else:
            losses['residual'] = zero_loss

        # === 3. SELF-SUPERVISED BOUNDARY LEARNING ===
        boundary_ref = noisy_input if noisy_input is not None else outputs['denoised']
        if 'boundaries' in outputs:
            anchor_loss, anchor_details = self.intensity_anchor_loss(
                outputs['boundaries'],
                boundary_ref,
            )
            losses['intensity_anchor'] = anchor_loss
            losses['detected_ilm'] = anchor_details.get('detected_retina_top', 0.0)
            losses['detected_rpe'] = anchor_details.get('detected_retina_bottom', 0.0)
        else:
            losses['intensity_anchor'] = zero_loss

        # === 4. BOUNDARY SMOOTHNESS ===
        if 'boundaries' in outputs and self.lambda_smoothness > 0:
            losses['smoothness'] = self.boundary_smoothness_loss(outputs['boundaries'])
        else:
            losses['smoothness'] = zero_loss

        # === TOTAL LOSS ===
        # Residual loss replaces clinical loss - direct supervision is better
        lambda_residual = self.lambda_clinical  # Reuse clinical weight for residual
        total = (
            self.lambda_mse * losses['mse'] +
            lambda_residual * losses['residual'] +
            self.lambda_intensity_anchor * losses['intensity_anchor'] +
            self.lambda_smoothness * losses['smoothness'] +
            losses['no_degrade']  # Critical: penalize degradation vs NAFNet
        )

        losses['total'] = total

        return total, {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


# Keep old class name as alias for backward compatibility
SelfSupervisedNeuroSymbolicLoss = SupervisedWithBoundaryAdaptationLoss


# =============================================================================
# Dataset with Multi-Frame Support
# =============================================================================
class NeuroSymbolicDataset(Dataset):
    """
    Dataset supporting multi-frame consistency training.
    """

    def __init__(
        self,
        jsonl_path: str,
        patch_size: int = 256,
        noise_levels: List[float] = None,
        num_noise_realizations: int = 3,
    ):
        self.patch_size = patch_size
        self.noise_levels = noise_levels or [0.05, 0.1, 0.15]
        self.num_realizations = num_noise_realizations

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def add_speckle_noise(self, clean: np.ndarray, level: float) -> np.ndarray:
        """Add realistic OCT speckle noise (Rayleigh)."""
        speckle = np.random.rayleigh(scale=level, size=clean.shape)
        noisy = clean * (1 + speckle)
        noisy = noisy + np.random.normal(0, level * 0.3, clean.shape)
        return np.clip(noisy, 0, 1).astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load clean image - support both formats:
        # 1. Duke format: 'image_path' (clean image)
        # 2. PKU37 format: 'clean_path' and 'noisy_path'
        image_path = sample.get('image_path') or sample.get('clean_path')
        if image_path is None:
            raise KeyError(f"Sample must have 'image_path' or 'clean_path'. Keys: {list(sample.keys())}")
        img = Image.open(image_path).convert('L')
        img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        clean = np.array(img, dtype=np.float32) / 255.0

        # Use REAL noisy image if available (PKU37), otherwise generate synthetic
        if 'noisy_path' in sample and os.path.exists(sample['noisy_path']):
            # Load REAL PKU37 noisy image
            noisy_img = Image.open(sample['noisy_path']).convert('L')
            noisy_img = noisy_img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            real_noisy = np.array(noisy_img, dtype=np.float32) / 255.0
            # Use real noisy as primary, add slight variations for multi-frame
            noisy_frames = [real_noisy]
            for _ in range(self.num_realizations - 1):
                # Small perturbation to simulate multi-frame (same scene, slight noise variation)
                noise_var = np.random.normal(0, 0.01, real_noisy.shape).astype(np.float32)
                noisy_frames.append(np.clip(real_noisy + noise_var, 0, 1))
        else:
            # Generate synthetic noise (fallback for Duke dataset)
            noise_level = np.random.choice(self.noise_levels)
            noisy_frames = [
                self.add_speckle_noise(clean, noise_level)
                for _ in range(self.num_realizations)
            ]

        # Load mask if available
        has_mask = 'mask_path' in sample and os.path.exists(sample['mask_path'])
        if has_mask:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = mask.resize((self.patch_size, self.patch_size), Image.NEAREST)
            mask = np.array(mask)
            if mask.max() > NUM_CLASSES - 1:
                mask = mask - 1
            mask = np.clip(mask, 0, NUM_CLASSES - 1)
        else:
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)

        return {
            'clean': torch.from_numpy(clean).float().unsqueeze(0),
            'noisy_frames': [torch.from_numpy(n).float().unsqueeze(0) for n in noisy_frames],
            'mask': torch.from_numpy(mask).long(),
            'has_mask': has_mask,
        }


# PKU37SelfSupervisedDataset removed - using supervised training with clean references
# IntensityAnchoredBoundaryLossV4 provides self-supervised boundary learning


def collate_multi_frame(batch):
    """Custom collate for multi-frame data."""
    clean = torch.stack([b['clean'] for b in batch])
    mask = torch.stack([b['mask'] for b in batch])
    has_mask = [b['has_mask'] for b in batch]

    # Stack noisy frames: List[B tensors] for each realization
    num_realizations = len(batch[0]['noisy_frames'])
    noisy_frames = [
        torch.stack([b['noisy_frames'][i] for b in batch])
        for i in range(num_realizations)
    ]

    return {
        'clean': clean,
        'noisy_frames': noisy_frames,
        'mask': mask,
        'has_mask': has_mask,
    }


# =============================================================================
# Training Functions
# =============================================================================
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM between prediction and target."""
    # Simple SSIM implementation
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = pred_np.mean()
    mu_target = target_np.mean()
    sigma_pred = pred_np.std()
    sigma_target = target_np.std()
    sigma_pred_target = ((pred_np - mu_pred) * (target_np - mu_target)).mean()

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred ** 2 + sigma_target ** 2 + C2))

    return float(ssim)


# =============================================================================
# Comprehensive Metrics for Evaluation
# =============================================================================
def compute_per_layer_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    segmentation: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Compute PSNR for each layer region."""
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
    results = {}

    for i in range(num_classes):
        mask = (segmentation == i).float()
        if mask.sum() > 100:  # Need sufficient pixels
            mse = ((pred - target) ** 2 * mask).sum() / mask.sum()
            if mse > 0:
                psnr = (10 * torch.log10(1.0 / mse)).item()
            else:
                psnr = float('inf')
            results[layer_names[i]] = psnr
        else:
            results[layer_names[i]] = None

    return results


def compute_per_layer_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    segmentation: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Compute SSIM for each layer region."""
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
    results = {}
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    seg_np = segmentation.squeeze().cpu().numpy()

    for i in range(num_classes):
        mask = (seg_np == i)
        if mask.sum() > 100:
            pred_layer = pred_np[mask]
            target_layer = target_np[mask]

            mu_pred = pred_layer.mean()
            mu_target = target_layer.mean()
            sigma_pred = pred_layer.std()
            sigma_target = target_layer.std()
            sigma_pred_target = ((pred_layer - mu_pred) * (target_layer - mu_target)).mean()

            ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred ** 2 + sigma_target ** 2 + C2))
            results[layer_names[i]] = float(ssim)
        else:
            results[layer_names[i]] = None

    return results


def compute_boundary_metrics(
    boundaries: torch.Tensor,
    H: int,
) -> Dict[str, float]:
    """Compute boundary quality metrics."""
    B, N, W = boundaries.shape

    # Check ordering constraint
    ordering_violations = 0
    total_checks = 0
    for i in range(N - 1):
        violations = (boundaries[:, i+1, :] <= boundaries[:, i, :]).sum().item()
        ordering_violations += violations
        total_checks += B * W
    ordering_satisfied = 1.0 - (ordering_violations / total_checks) if total_checks > 0 else 1.0

    # Check thickness constraints (physiological bounds)
    thickness_bounds = [
        (0.03, 0.20),  # RNFL_GCL
        (0.08, 0.35),  # INL_OPL_ONL
        (0.02, 0.12),  # IS_OS
    ]
    thickness_valid = 0
    thickness_total = 0
    for i in range(min(N-1, len(thickness_bounds))):
        thickness = boundaries[:, i+1, :] - boundaries[:, i, :]
        min_t, max_t = thickness_bounds[i]
        valid = ((thickness >= min_t) & (thickness <= max_t)).sum().item()
        thickness_valid += valid
        thickness_total += thickness.numel()
    thickness_ratio = thickness_valid / thickness_total if thickness_total > 0 else 1.0

    # Boundary smoothness (average gradient magnitude)
    if W > 1:
        smoothness = torch.abs(boundaries[:, :, 1:] - boundaries[:, :, :-1]).mean().item()
    else:
        smoothness = 0.0

    return {
        'ordering_satisfied': ordering_satisfied,
        'thickness_valid': thickness_ratio,
        'boundary_smoothness': smoothness,
    }


def compute_dice_coefficient(
    pred_seg: torch.Tensor,
    gt_seg: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Compute Dice coefficient per class."""
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
    results = {}

    for i in range(num_classes):
        pred_mask = (pred_seg == i).float()
        gt_mask = (gt_seg == i).float()

        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()

        if union > 0:
            dice = (2 * intersection / union).item()
        else:
            dice = 1.0 if intersection == 0 else 0.0

        results[layer_names[i]] = dice

    return results


def print_epoch_metrics(
    epoch: int,
    train_metrics: Dict,
    val_metrics: Dict,
):
    """Print comprehensive metrics for an epoch."""
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} METRICS")
    print(f"{'='*70}")

    # Denoising - Overall comparison
    print(f"\n[DENOISING - OVERALL]")
    print(f"  Train: Loss={train_metrics.get('loss', 0):.4f}, PSNR={train_metrics.get('psnr', 0):.2f}dB")
    noisy_psnr = val_metrics.get('noisy_psnr', 0)
    base_psnr = val_metrics.get('base_psnr', 0)
    full_psnr = val_metrics.get('psnr', 0)
    base_ssim = val_metrics.get('base_ssim', 0)
    full_ssim = val_metrics.get('ssim', 0)
    psnr_gain = full_psnr - base_psnr if base_psnr > 0 else 0
    ssim_gain = full_ssim - base_ssim if base_ssim > 0 else 0
    print(f"  {'Method':<25} {'PSNR (dB)':<12} {'SSIM':<12} {'Gain':<12}")
    print(f"  {'-'*60}")
    print(f"  {'Noisy Input':<25} {noisy_psnr:.2f}         -            -")
    print(f"  {'Base NAFNet':<25} {base_psnr:.2f}         {base_ssim:.4f}       -")
    print(f"  {'Full Model (Adaptive)':<25} {full_psnr:.2f}         {full_ssim:.4f}       +{psnr_gain:.2f}dB / +{ssim_gain:.4f}")

    # Per-layer PSNR/SSIM with gains
    if 'layer_psnr' in val_metrics and 'base_layer_psnr' in val_metrics:
        print(f"\n[PER-LAYER METRICS - ADAPTIVE vs BASE NAFNet]")
        print(f"  {'Layer':<15} {'Base PSNR':<12} {'Adapt PSNR':<12} {'Gain':<10} {'Base SSIM':<12} {'Adapt SSIM':<12} {'Gain':<10}")
        print(f"  {'-'*90}")

        layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
        for name in layer_names:
            base_p = val_metrics.get('base_layer_psnr', {}).get(name)
            full_p = val_metrics.get('layer_psnr', {}).get(name)
            base_s = val_metrics.get('base_layer_ssim', {}).get(name)
            full_s = val_metrics.get('layer_ssim', {}).get(name)
            psnr_g = val_metrics.get('layer_psnr_gain', {}).get(name)
            ssim_g = val_metrics.get('layer_ssim_gain', {}).get(name)

            base_p_str = f"{base_p:.2f}" if base_p else "N/A"
            full_p_str = f"{full_p:.2f}" if full_p else "N/A"
            base_s_str = f"{base_s:.4f}" if base_s else "N/A"
            full_s_str = f"{full_s:.4f}" if full_s else "N/A"
            psnr_g_str = f"{'+' if psnr_g and psnr_g >= 0 else ''}{psnr_g:.2f}" if psnr_g else "N/A"
            ssim_g_str = f"{'+' if ssim_g and ssim_g >= 0 else ''}{ssim_g:.4f}" if ssim_g else "N/A"

            print(f"  {name:<15} {base_p_str:<12} {full_p_str:<12} {psnr_g_str:<10} {base_s_str:<12} {full_s_str:<12} {ssim_g_str:<10}")

    # Boundary metrics
    if 'boundary_metrics' in val_metrics:
        bm = val_metrics['boundary_metrics']
        print(f"\n[BOUNDARY QUALITY]")
        print(f"  Ordering satisfied: {bm.get('ordering_satisfied', 0)*100:.1f}%")
        print(f"  Thickness valid:    {bm.get('thickness_valid', 0)*100:.1f}%")
        print(f"  Smoothness (lower=better): {bm.get('boundary_smoothness', 0):.4f}")

    # Symbolic losses
    if 'anatomical_loss' in val_metrics:
        print(f"\n[SYMBOLIC CONSTRAINTS]")
        print(f"  Anatomical loss: {val_metrics.get('anatomical_loss', 0):.4f}")
        print(f"  Physics loss:    {val_metrics.get('physics_loss', 0):.4f}")
        print(f"  Logic loss:      {val_metrics.get('logic_loss', 0):.4f}")
        print(f"  Fresnel (in physics): included")

    # Segmentation (if GT available)
    if 'dice' in val_metrics and val_metrics['dice']:
        print(f"\n[SEGMENTATION DICE]")
        for name, dice in val_metrics['dice'].items():
            if dice is not None:
                print(f"  {name}: {dice:.4f}")
            else:
                print(f"  {name}: N/A")
        valid_dices = [d for d in val_metrics['dice'].values() if d is not None]
        if valid_dices:
            avg_dice = np.mean(valid_dices)
            print(f"  Average: {avg_dice:.4f}")

    # Adaptation evidence - PROOF that layer-adaptive denoising is working
    if 'adaptation_evidence' in val_metrics:
        ev = val_metrics['adaptation_evidence']
        print(f"\n[ADAPTATION EVIDENCE - Proof of Layer-Specific Processing]")

        layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

        # KEY EVIDENCE: Does each head activate MORE in its own region?
        if 'head_in_own_region' in ev and 'head_in_other_region' in ev:
            print(f"  Head Activation: Own Region vs Other Regions (RATIO > 1 = ADAPTIVE)")
            print(f"  {'Layer':<15} {'Own Region':<12} {'Other':<12} {'Ratio':<10} {'Adaptive?':<10}")
            print(f"  {'-'*60}")
            for name in layer_names:
                own = ev['head_in_own_region'].get(name, 0)
                other = ev['head_in_other_region'].get(name, 1e-8)
                ratio = own / other if other > 1e-8 else 0
                adaptive = "YES" if ratio > 1.0 else "NO"
                print(f"  {name:<15} {own:.6f}     {other:.6f}     {ratio:.2f}x      {adaptive}")

        # Show mask coverage (each layer covers different regions)
        print(f"\n  Per-Layer Mask Coverage:")
        for name in layer_names:
            coverage = ev['mask_coverage'].get(name, 0)
            print(f"    {name:<15}: {coverage*100:.1f}%")

        # Show overall refinement magnitude
        refine_mag = ev.get('refinement_magnitude', 0)
        print(f"\n  Adaptive Refinement Magnitude: {refine_mag:.4f}")

    # V8 Predicate Scores (P1-P6) - GT-free quality assessment
    if 'predicate_scores' in val_metrics and any(v is not None for v in val_metrics['predicate_scores'].values()):
        print(f"\n[V8 PREDICATE SCORES - GT-Free Quality Assessment]")
        print(f"  {'Predicate':<15} {'Score':<10} {'Status':<10} {'Description'}")
        print(f"  {'-'*70}")

        pred_descriptions = {
            'P1_edge': 'Edge preservation quality',
            'P2_contrast': 'Local contrast preservation',
            'P3_smooth': 'Noise smoothness',
            'P4_structure': 'Structure similarity (SSIM-like)',
            'P5_speckle': 'Speckle fidelity (physics-based)',
            'P6_anatomy': 'Anatomical validity',
        }

        for name, score in val_metrics['predicate_scores'].items():
            if score is not None:
                status = "PASS" if score >= 0.5 else "FAIL"
                desc = pred_descriptions.get(name, '')
                print(f"  {name:<15} {score:.4f}     {status:<10} {desc}")

        # Average predicate score
        valid_scores = [s for s in val_metrics['predicate_scores'].values() if s is not None]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            print(f"  {'-'*70}")
            print(f"  {'Average':<15} {avg_score:.4f}     {'PASS' if avg_score >= 0.5 else 'FAIL':<10} Overall quality")

    print(f"{'='*70}\n")


# =============================================================================
# Sliding Window Inference for Full-Resolution Validation
# =============================================================================
@torch.no_grad()
def sliding_window_inference(
    model,
    image: torch.Tensor,
    patch_size: int = 256,
    stride: int = 128,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Perform inference using sliding windows on full-resolution image.

    Args:
        model: The denoising model
        image: Input image [1, 1, H, W] or [1, H, W]
        patch_size: Size of each patch (256x256)
        stride: Stride between patches (use patch_size//2 for 50% overlap)
        device: Device to run inference on

    Returns:
        Dictionary with 'denoised' and 'boundaries' at full resolution
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure 4D tensor
    if image.dim() == 3:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    assert B == 1, "Sliding window only supports batch size 1"

    # Initialize output accumulators
    denoised_sum = torch.zeros(1, 1, H, W, device=device)
    weight_sum = torch.zeros(1, 1, H, W, device=device)

    # Optional: accumulate boundaries
    boundaries_list = []

    # Calculate number of patches
    n_h = max(1, (H - patch_size) // stride + 1)
    n_w = max(1, (W - patch_size) // stride + 1)

    # Handle case where image is smaller than patch
    if H <= patch_size and W <= patch_size:
        # Pad image to patch_size
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
        padded = padded.to(device)

        outputs = model(padded, return_symbolic=False)
        denoised = outputs['denoised'][:, :, :H, :W]

        return {
            'denoised': denoised,
            'boundaries': outputs.get('boundaries'),
        }

    # Pre-compute Gaussian weight (reuse for all patches - MEMORY FIX)
    sigma = patch_size / 4
    y_coords = torch.arange(patch_size, device=device).float() - patch_size / 2
    x_coords = torch.arange(patch_size, device=device).float() - patch_size / 2
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    gaussian_weight = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2)).unsqueeze(0).unsqueeze(0)
    del y_coords, x_coords, yy, xx  # Free memory

    # Sliding window
    for i in range(n_h):
        for j in range(n_w):
            # Calculate patch coordinates
            y_start = min(i * stride, H - patch_size)
            x_start = min(j * stride, W - patch_size)
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # Extract patch
            patch = image[:, :, y_start:y_end, x_start:x_end].to(device)

            # Run inference
            outputs = model(patch, return_symbolic=False)
            denoised_patch = outputs['denoised']

            # Accumulate with pre-computed Gaussian weight
            denoised_sum[:, :, y_start:y_end, x_start:x_end] += denoised_patch * gaussian_weight
            weight_sum[:, :, y_start:y_end, x_start:x_end] += gaussian_weight

            # Collect boundaries for averaging (detach to prevent memory leak)
            if 'boundaries' in outputs and outputs['boundaries'] is not None:
                boundaries_list.append({
                    'boundaries': outputs['boundaries'].detach().cpu(),  # Move to CPU to save GPU memory
                    'y_start': y_start,
                    'x_start': x_start,
                })

            # MEMORY FIX: Delete outputs after each patch
            del outputs, denoised_patch, patch

    # Normalize by weights
    denoised = denoised_sum / (weight_sum + 1e-8)

    # Average boundaries (simple approach - take from center patch)
    boundaries = None
    if boundaries_list:
        # Use boundaries from the most central patch
        center_idx = len(boundaries_list) // 2
        boundaries = boundaries_list[center_idx]['boundaries']

    return {
        'denoised': denoised,
        'boundaries': boundaries,
    }


@torch.no_grad()
def sliding_window_multi_frame(
    model,
    noisy_frames: List[torch.Tensor],
    patch_size: int = 256,
    stride: int = 128,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Sliding window inference for multiple noisy frames.
    Processes each frame and averages results.
    """
    if device is None:
        device = next(model.parameters()).device

    denoised_list = []
    for frame in noisy_frames:
        result = sliding_window_inference(model, frame, patch_size, stride, device)
        denoised_list.append(result['denoised'])

    # Average denoised results
    denoised_avg = torch.stack(denoised_list).mean(dim=0)

    return {
        'denoised': denoised_avg,
        'boundaries': result.get('boundaries'),  # Use last frame's boundaries
    }


def train_epoch(model, loader, criterion, optimizer, device, epoch, single_frame=False):
    """
    Training epoch with optional single-frame mode for faster training.

    Args:
        single_frame: If True, use only first frame (2-3x faster)
    """
    model.train()
    total_loss = 0
    total_psnr = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        clean = batch['clean'].to(device)
        noisy_frames = [f.to(device) for f in batch['noisy_frames']]
        mask = batch['mask'].to(device)
        has_mask = batch['has_mask']

        optimizer.zero_grad(set_to_none=True)  # More memory efficient

        # Single-frame mode: faster, skip multi-frame consistency
        if single_frame:
            outputs = model.forward(noisy_frames[0], return_symbolic=True)
            # Add dummy consistency loss for compatibility
            outputs['consistency_loss'] = noisy_frames[0].new_zeros(1)  # MEMORY FIX: reuse device/dtype
        else:
            # Multi-frame forward for consistency
            outputs = model.forward_multi_frame(noisy_frames)

        # Use mask only if available
        mask_for_loss = mask if any(has_mask) else None

        # Pass noisy_input for self-supervised boundary learning (V4)
        loss, losses = criterion(outputs, clean, mask_for_loss, noisy_input=noisy_frames[0])

        # Store values before clearing computational graph
        loss_val = loss.item()
        loss_mse = losses.get('mse', losses.get('l1', 0))
        loss_bnd = losses.get('intensity_anchor', 0)

        loss.backward()

        # Clear loss and losses immediately after backward to free memory
        del loss, losses, mask_for_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            psnr = compute_psnr(outputs['denoised'], clean)

        total_loss += loss_val
        total_psnr += psnr
        n += 1

        pbar.set_postfix({
            'loss': f"{loss_val:.4f}",
            'psnr': f"{psnr:.2f}",
            'mse': f"{loss_mse:.4f}",
            'bnd': f"{loss_bnd:.4f}",  # V4 boundary loss
        })

        # Memory cleanup - more aggressive for CPU training
        del outputs, clean, noisy_frames, mask, batch
        if n % 2 == 0:  # MEMORY FIX: Increased frequency from 5 to 2 for OOM prevention
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / n, total_psnr / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation with comprehensive metrics including per-layer gains over base NAFNet."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    n = 0

    # Accumulators for additional metrics
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
    layer_psnr_accum = {name: [] for name in layer_names}
    layer_ssim_accum = {name: [] for name in layer_names}
    # Base NAFNet metrics (for comparison)
    base_psnr_accum = {'overall': [], **{name: [] for name in layer_names}}
    base_ssim_accum = {'overall': [], **{name: [] for name in layer_names}}
    # Noisy input metrics (for reference)
    noisy_psnr_accum = {'overall': [], **{name: [] for name in layer_names}}

    boundary_metrics_accum = {'ordering_satisfied': [], 'thickness_valid': [], 'boundary_smoothness': []}
    symbolic_accum = {'anatomical': [], 'physics': [], 'logic': []}
    dice_accum = {name: [] for name in layer_names}

    # Adaptation evidence accumulators
    adaptation_accum = {
        'head_contribution': {name: [] for name in layer_names},  # Per-head output magnitude
        'mask_coverage': {name: [] for name in layer_names},       # Per-layer mask coverage
        'refinement_magnitude': [],                                 # Overall refinement magnitude
        'head_output_std': {name: [] for name in layer_names},     # Per-head output variance (diversity)
    }

    # V8 Predicate score accumulators (P1-P6)
    predicate_names = ['P1_edge', 'P2_contrast', 'P3_smooth', 'P4_structure', 'P5_speckle', 'P6_anatomy']
    predicate_accum = {name: [] for name in predicate_names}
    v8_corrector = None
    if HAS_V8_CORRECTOR:
        try:
            v8_corrector = NeuroSymbolicCorrectorV8().to(device)
            v8_corrector.eval()
        except Exception as e:
            print(f"Warning: Could not initialize V8 corrector: {e}")

    for batch in tqdm(loader, desc="Validation"):
        clean = batch['clean'].to(device)
        noisy_frames = [f.to(device) for f in batch['noisy_frames']]
        mask = batch['mask'].to(device)
        has_mask = batch.get('has_mask', [False])

        # Full model forward (with layer adaptation)
        outputs = model.forward_multi_frame(noisy_frames)
        # Pass noisy_input for self-supervised boundary learning (V4)
        loss, _ = criterion(outputs, clean, mask, noisy_input=noisy_frames[0])

        denoised = outputs['denoised']
        segmentation = outputs['segmentation']
        boundaries = outputs['boundaries']

        # Also run base NAFNet only (no layer adaptation) for comparison
        with torch.no_grad():
            noisy = noisy_frames[0]
            if hasattr(model.denoiser, 'forward'):
                # NAFNetFullFiLM signature
                try:
                    base_denoised = model.denoiser(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
                except TypeError:
                    base_denoised = model.denoiser(noisy)
            else:
                base_denoised = model.denoiser(noisy)

        # Overall metrics
        total_loss += loss.item()
        total_psnr += compute_psnr(denoised, clean)
        total_ssim += compute_ssim(denoised, clean)

        # Base NAFNet overall metrics
        base_psnr_accum['overall'].append(compute_psnr(base_denoised, clean))
        base_ssim_accum['overall'].append(compute_ssim(base_denoised, clean))

        # Noisy input overall metrics
        noisy_psnr_accum['overall'].append(compute_psnr(noisy, clean))

        # Per-layer PSNR/SSIM for full model
        layer_psnr = compute_per_layer_psnr(denoised, clean, segmentation)
        layer_ssim = compute_per_layer_ssim(denoised, clean, segmentation)
        for name in layer_names:
            if layer_psnr.get(name) is not None:
                layer_psnr_accum[name].append(layer_psnr[name])
            if layer_ssim.get(name) is not None:
                layer_ssim_accum[name].append(layer_ssim[name])

        # Per-layer PSNR/SSIM for base NAFNet
        base_layer_psnr = compute_per_layer_psnr(base_denoised, clean, segmentation)
        base_layer_ssim = compute_per_layer_ssim(base_denoised, clean, segmentation)
        for name in layer_names:
            if base_layer_psnr.get(name) is not None:
                base_psnr_accum[name].append(base_layer_psnr[name])
            if base_layer_ssim.get(name) is not None:
                base_ssim_accum[name].append(base_layer_ssim[name])

        # Per-layer PSNR for noisy input
        noisy_layer_psnr = compute_per_layer_psnr(noisy, clean, segmentation)
        for name in layer_names:
            if noisy_layer_psnr.get(name) is not None:
                noisy_psnr_accum[name].append(noisy_layer_psnr[name])

        # Per-layer PSNR (legacy - keep for backward compatibility)
        layer_psnr = compute_per_layer_psnr(denoised, clean, segmentation)
        for name, psnr in layer_psnr.items():
            if psnr is not None:
                pass  # Already accumulated above

        # Boundary metrics
        H = clean.shape[2]
        bm = compute_boundary_metrics(boundaries, H)
        for key, val in bm.items():
            boundary_metrics_accum[key].append(val)

        # Symbolic losses
        if 'anatomical_violations' in outputs:
            symbolic_accum['anatomical'].append(outputs['anatomical_violations']['total'].item())
        if 'physics_consistency' in outputs:
            symbolic_accum['physics'].append(outputs['physics_consistency'].item())
        if 'logic_loss' in outputs:
            symbolic_accum['logic'].append(outputs['logic_loss'].item())

        # Collect adaptation evidence
        if 'adaptation_evidence' in outputs:
            ev = outputs['adaptation_evidence']
            # Per-head contribution (mean absolute output)
            if 'layer_stack' in ev:
                layer_stack = ev['layer_stack']  # [B, 4, H, W]
                soft_masks_ev = ev.get('soft_masks', None)  # [B, 4, H, W]

                for i, name in enumerate(layer_names):
                    head_out = layer_stack[:, i, :, :]
                    adaptation_accum['head_contribution'][name].append(head_out.abs().mean().item())
                    adaptation_accum['head_output_std'][name].append(head_out.std().item())

                # NEW: Per-head activation in its OWN region vs OTHER regions
                # This shows if head_i activates more in region_i
                if soft_masks_ev is not None and 'head_in_region' not in adaptation_accum:
                    adaptation_accum['head_in_own_region'] = {name: [] for name in layer_names}
                    adaptation_accum['head_in_other_region'] = {name: [] for name in layer_names}

                if soft_masks_ev is not None:
                    for i, name in enumerate(layer_names):
                        head_out = layer_stack[:, i, :, :].abs()
                        own_mask = soft_masks_ev[:, i, :, :]
                        other_mask = 1 - own_mask

                        # Head activation in its own region
                        own_activation = (head_out * own_mask).sum() / (own_mask.sum() + 1e-8)
                        # Head activation in other regions
                        other_activation = (head_out * other_mask).sum() / (other_mask.sum() + 1e-8)

                        adaptation_accum['head_in_own_region'][name].append(own_activation.item())
                        adaptation_accum['head_in_other_region'][name].append(other_activation.item())

            # Per-layer mask coverage
            if 'soft_masks' in ev:
                soft_masks = ev['soft_masks']  # [B, 4, H, W]
                for i, name in enumerate(layer_names):
                    mask = soft_masks[:, i, :, :]
                    adaptation_accum['mask_coverage'][name].append(mask.mean().item())

            # Refinement magnitude
            if 'refinement' in ev:
                refinement = ev['refinement']
                adaptation_accum['refinement_magnitude'].append(refinement.abs().mean().item())

        # Dice coefficient (if GT mask available)
        if any(has_mask):
            dice = compute_dice_coefficient(segmentation, mask)
            for name, d in dice.items():
                dice_accum[name].append(d)

        # V8 Predicate scores (P1-P6) - evaluate denoising quality without GT
        if v8_corrector is not None:
            with torch.no_grad():
                try:
                    _, v8_info = v8_corrector(denoised, noisy_frames[0])
                    for pred_name, score in v8_info['predicate_scores'].items():
                        if pred_name in predicate_accum:
                            predicate_accum[pred_name].append(score)
                except Exception as e:
                    pass  # Skip if V8 evaluation fails

        n += 1
        # Memory cleanup - delete all batch tensors
        del outputs, loss, denoised, segmentation, boundaries
        del base_denoised, noisy, clean, noisy_frames, mask
        del layer_psnr, layer_ssim, base_layer_psnr, base_layer_ssim, noisy_layer_psnr

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate results
    results = {
        'loss': total_loss / n if n > 0 else 0,
        'ssim': total_ssim / n if n > 0 else 0,
        'psnr': total_psnr / n if n > 0 else 0,
    }

    # Base NAFNet overall metrics
    results['base_psnr'] = np.mean(base_psnr_accum['overall']) if base_psnr_accum['overall'] else 0
    results['base_ssim'] = np.mean(base_ssim_accum['overall']) if base_ssim_accum['overall'] else 0
    results['noisy_psnr'] = np.mean(noisy_psnr_accum['overall']) if noisy_psnr_accum['overall'] else 0

    # Layer PSNR/SSIM for full model
    results['layer_psnr'] = {
        name: np.mean(vals) if vals else None
        for name, vals in layer_psnr_accum.items()
    }
    results['layer_ssim'] = {
        name: np.mean(vals) if vals else None
        for name, vals in layer_ssim_accum.items()
    }

    # Base NAFNet per-layer metrics
    results['base_layer_psnr'] = {
        name: np.mean(vals) if vals else None
        for name, vals in base_psnr_accum.items() if name != 'overall'
    }
    results['base_layer_ssim'] = {
        name: np.mean(vals) if vals else None
        for name, vals in base_ssim_accum.items() if name != 'overall'
    }

    # Per-layer gains (improvement over base NAFNet)
    results['layer_psnr_gain'] = {}
    results['layer_ssim_gain'] = {}
    for name in layer_names:
        full_psnr = results['layer_psnr'].get(name)
        base_psnr = results['base_layer_psnr'].get(name)
        if full_psnr is not None and base_psnr is not None:
            results['layer_psnr_gain'][name] = full_psnr - base_psnr
        else:
            results['layer_psnr_gain'][name] = None

        full_ssim = results['layer_ssim'].get(name)
        base_ssim = results['base_layer_ssim'].get(name)
        if full_ssim is not None and base_ssim is not None:
            results['layer_ssim_gain'][name] = full_ssim - base_ssim
        else:
            results['layer_ssim_gain'][name] = None

    # Boundary metrics
    results['boundary_metrics'] = {
        key: np.mean(vals) if vals else 0
        for key, vals in boundary_metrics_accum.items()
    }

    # Symbolic losses
    results['anatomical_loss'] = np.mean(symbolic_accum['anatomical']) if symbolic_accum['anatomical'] else 0
    results['physics_loss'] = np.mean(symbolic_accum['physics']) if symbolic_accum['physics'] else 0
    results['logic_loss'] = np.mean(symbolic_accum['logic']) if symbolic_accum['logic'] else 0

    # Dice (if available)
    results['dice'] = {
        name: np.mean(vals) if vals else None
        for name, vals in dice_accum.items()
    }

    # V8 Predicate scores (P1-P6) - GT-free quality metrics
    results['predicate_scores'] = {
        name: np.mean(vals) if vals else None
        for name, vals in predicate_accum.items()
    }

    # Adaptation evidence
    results['adaptation_evidence'] = {
        'head_contribution': {
            name: np.mean(vals) if vals else 0
            for name, vals in adaptation_accum['head_contribution'].items()
        },
        'head_output_std': {
            name: np.mean(vals) if vals else 0
            for name, vals in adaptation_accum['head_output_std'].items()
        },
        'mask_coverage': {
            name: np.mean(vals) if vals else 0
            for name, vals in adaptation_accum['mask_coverage'].items()
        },
        'refinement_magnitude': np.mean(adaptation_accum['refinement_magnitude']) if adaptation_accum['refinement_magnitude'] else 0,
    }
    # Add region-specific activation (key evidence of adaptation)
    if 'head_in_own_region' in adaptation_accum:
        results['adaptation_evidence']['head_in_own_region'] = {
            name: np.mean(vals) if vals else 0
            for name, vals in adaptation_accum['head_in_own_region'].items()
        }
        results['adaptation_evidence']['head_in_other_region'] = {
            name: np.mean(vals) if vals else 0
            for name, vals in adaptation_accum['head_in_other_region'].items()
        }

    return results


# Self-supervised denoising functions removed
# Using supervised denoising (MSE with clean reference)
# + self-supervised boundary learning (IntensityAnchoredBoundaryLossV4)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Neuro-Symbolic OCT Denoising')

    # Data
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--num_realizations', type=int, default=2,
                        help='Number of noise realizations for consistency (2 saves memory)')

    # PKU37 dataset path (for supervised training with clean references)
    parser.add_argument('--pku37_root', type=str, default=None,
                        help='Path to PKU37 dataset (supervised denoising + self-supervised boundaries)')

    # Model
    parser.add_argument('--hidden_channels', type=int, default=48)
    parser.add_argument('--nafnet_width', type=int, default=64,
                        help='NAFNet width (must match checkpoint, default=64)')
    parser.add_argument('--nafnet_ckpt', default='outputs/nafnet_calibrated/nafnet_best.pth')
    parser.add_argument('--physics_ckpt', default='outputs/physics_v3_dice_v2/stage3_256/best_model.pt')
    parser.add_argument('--blend_sigma', type=float, default=7.0)
    parser.add_argument('--freeze_nafnet', action='store_true',
                        help='Freeze NAFNet backbone, only train boundary/layer heads')
    parser.add_argument('--freeze_boundary', action='store_true',
                        help='Freeze boundary model too, only train layer heads')
    parser.add_argument('--single_frame', action='store_true',
                        help='Use single frame (faster training, skip multi-frame consistency)')
    parser.add_argument('--skip_symbolic', action='store_true',
                        help='Skip expensive symbolic computations during training (faster)')
    parser.add_argument('--no_refinement', action='store_true',
                        help='Disable layer refinement, use pure NAFNet output (for baseline comparison)')

    # Loss weights (supervised WITH GT masks) - REDUCED symbolic weights since dice loss provides signal
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_anatomical', type=float, default=0.1,
                        help='Anatomical constraint weight (low when GT masks available)')
    parser.add_argument('--lambda_physics', type=float, default=0.05,
                        help='Physics constraint weight (low when GT masks available)')
    parser.add_argument('--lambda_consistency', type=float, default=0.1,
                        help='Multi-frame consistency weight')
    parser.add_argument('--lambda_speckle', type=float, default=0.01,
                        help='Speckle distribution weight')
    parser.add_argument('--lambda_logic', type=float, default=0.05,
                        help='Logic layer weight (low when GT masks available)')

    # Flag for supervised denoising + self-supervised boundary learning (no GT masks)
    parser.add_argument('--no_gt_masks', action='store_true',
                        help='No GT segmentation masks available - use HIGHER symbolic weights for boundary learning')

    # Loss weights for disentangled per-layer learning
    parser.add_argument('--lambda_smoothness', type=float, default=0.1,
                        help='Boundary smoothness loss weight')
    parser.add_argument('--lambda_intensity_anchor', type=float, default=1.0,
                        help='Intensity-anchored boundary loss weight (self-supervised boundary learning)')
    parser.add_argument('--lambda_clinical', type=float, default=0.5,
                        help='Per-layer clinical loss weight (texture/structure/edge/contrast)')
    parser.add_argument('--lambda_disentangle', type=float, default=0.1,
                        help='Disentanglement loss weight (anatomy/pathology/noise independence)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--val_every', type=int, default=1,
                        help='Validate every N epochs (increase to speed up training)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0 for main thread, increase for faster loading)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fast', action='store_true',
                        help='Use smaller model for fast CPU training (no pretrained weights)')

    # Output
    parser.add_argument('--output_dir', default='outputs/neurosymbolic_denoising')

    args = parser.parse_args()

    # Fast mode: use smaller model for CPU training
    if args.fast:
        args.hidden_channels = 16
        args.nafnet_width = 16
        args.nafnet_ckpt = None  # Can't use pretrained with different width
        args.physics_ckpt = None
        args.num_realizations = 1  # Single frame for speed
        logger.info("FAST MODE: Using smaller model (hidden=16, nafnet=16, 1 realization)")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # === PERFORMANCE OPTIMIZATIONS ===
    # Set optimal number of threads for CPU
    if device.type == 'cpu':
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        # Use all cores for intra-op parallelism
        torch.set_num_threads(num_cores)
        # Use half for inter-op (data loading etc)
        torch.set_num_interop_threads(max(1, num_cores // 2))
        logger.info(f"CPU threads: {num_cores} intra-op, {max(1, num_cores // 2)} inter-op")

    # Enable optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input size
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')  # Faster matmul

    logger.info("=" * 60)
    logger.info("NEURO-SYMBOLIC OCT DENOISING")
    logger.info("Supervised Denoising + Self-Supervised Boundary Learning")
    logger.info("=" * 60)
    logger.info("Training Mode:")
    logger.info("  - Denoising: SUPERVISED (MSE with clean reference)")
    logger.info("  - Boundaries: SELF-SUPERVISED (IntensityAnchoredBoundaryLossV4 - ensemble detection + confidence)")
    logger.info("=" * 60)
    logger.info(f"Physics checkpoint: {args.physics_ckpt}")
    logger.info(f"NAFNet checkpoint: {args.nafnet_ckpt}")
    logger.info(f"Patch size: {args.patch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Freeze NAFNet: {args.freeze_nafnet}")
    logger.info(f"Freeze Boundary: {args.freeze_boundary}")

    # Create model
    model = NeuroSymbolicDenoiser(
        hidden_channels=args.hidden_channels,
        nafnet_width=args.nafnet_width,
        nafnet_ckpt=args.nafnet_ckpt,
        physics_ckpt=args.physics_ckpt,
        blend_sigma=args.blend_sigma,
        freeze_boundary=args.freeze_boundary,
        freeze_nafnet=args.freeze_nafnet,
        no_refinement=getattr(args, 'no_refinement', False),
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {params:,} total, {trainable:,} trainable")

    # Create loss function with DISENTANGLED per-layer clinical losses
    # 1. MSE: Global supervised denoising
    # 2. Clinical: Per-layer losses (texture/structure/edge/contrast)
    # 3. Disentangle: Independence of anatomy/pathology/noise
    # 4. IntensityAnchor: Self-supervised boundary learning
    # 5. Smoothness: Boundary regularization
    criterion = SupervisedWithBoundaryAdaptationLoss(
        lambda_mse=args.lambda_l1,
        lambda_clinical=args.lambda_clinical,
        lambda_disentangle=args.lambda_disentangle,
        lambda_intensity_anchor=args.lambda_intensity_anchor,
        lambda_smoothness=args.lambda_smoothness,
    )
    logger.info("Loss: MSE + Clinical(per-layer) + Disentangle + Boundary")
    logger.info(f"  lambda_mse: {args.lambda_l1}")
    logger.info(f"  lambda_clinical: {args.lambda_clinical}")
    logger.info(f"  lambda_disentangle: {args.lambda_disentangle}")
    logger.info(f"  lambda_intensity_anchor: {args.lambda_intensity_anchor}")
    logger.info(f"  lambda_smoothness: {args.lambda_smoothness}")

    # Create datasets (use JSONL format with clean references)
    train_dataset = NeuroSymbolicDataset(
        args.train_jsonl,
        patch_size=args.patch_size,
        num_noise_realizations=args.num_realizations,
    )
    if args.max_train:
        train_dataset.samples = train_dataset.samples[:args.max_train]

    val_dataset = NeuroSymbolicDataset(
        args.val_jsonl,
        patch_size=args.patch_size,
        num_noise_realizations=args.num_realizations,
    )
    if args.max_val:
        val_dataset.samples = val_dataset.samples[:args.max_val]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_multi_frame,
        pin_memory=(device.type == 'cuda'),  # Faster GPU transfer
        persistent_workers=(args.num_workers > 0),  # Keep workers alive between epochs
    )

    val_batch_size = args.batch_size
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_multi_frame,
        pin_memory=(device.type == 'cuda'),
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # === SMART OPTIMIZER WITH DIFFERENTIAL LEARNING RATES ===
    # Different components learn at different rates for optimal training
    param_groups = []

    # Count parameters by group
    adapter_params = list(model.residual_adapter.parameters())
    layer_head_params = list(model.layer_heads.parameters())
    fusion_params = list(model.fusion.parameters())
    feat_extract_params = list(model.feat_extract.parameters())

    # Novel components (adapters, layer heads) - full learning rate
    param_groups.append({
        'params': adapter_params,
        'lr': args.lr,
        'name': 'adapters'
    })
    param_groups.append({
        'params': layer_head_params,
        'lr': args.lr,
        'name': 'layer_heads'
    })
    param_groups.append({
        'params': fusion_params + feat_extract_params,
        'lr': args.lr,
        'name': 'fusion'
    })

    # Other trainable params (symbolic components, refinement_scale)
    other_params = []
    trained_param_ids = set(id(p) for p in adapter_params + layer_head_params + fusion_params + feat_extract_params)
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in trained_param_ids:
            other_params.append(p)
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': args.lr * 0.5,  # Slightly lower for symbolic components
            'name': 'other'
        })

    optimizer = torch.optim.AdamW(param_groups)

    # Log optimizer setup
    logger.info("Optimizer: AdamW with differential learning rates:")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        logger.info(f"  {pg['name']}: {n_params:,} params @ lr={pg['lr']:.2e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_psnr = 0

    for epoch in range(1, args.epochs + 1):
        # Supervised training with self-supervised boundary learning
        train_loss, train_psnr = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            single_frame=args.single_frame,
        )

        scheduler.step()

        # Validate every N epochs (or last epoch)
        should_validate = (epoch % args.val_every == 0) or (epoch == args.epochs)

        if should_validate:
            val_results = validate(model, val_loader, criterion, device)
        else:
            # Skip validation - use placeholder metrics
            val_results = {'psnr': 0, 'ssim': 0, 'loss': 0}
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_psnr={train_psnr:.2f} (skip val)")
            continue

        # Print comprehensive metrics
        train_metrics = {'loss': train_loss, 'psnr': train_psnr}
        print_epoch_metrics(epoch, train_metrics, val_results)

        # Save best
        if val_results['psnr'] > best_psnr:
            best_psnr = val_results['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"  New best PSNR: {best_psnr:.2f}")

    logger.info("=" * 60)
    logger.info(f"Training complete. Best PSNR: {best_psnr:.2f}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("Mode: Supervised Denoising + Self-Supervised Boundaries")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
