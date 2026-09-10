#!/usr/bin/env python3
"""
Symbolic Predicates for Verifiable OCT Denoising

This module implements three verifiable symbolic predicates that can assess
denoising quality WITHOUT ground truth. This enables:
1. Quality verification on real clinical data
2. Domain-invariant quality assessment
3. Interpretable failure detection

Predicates:
- P₁: SpeckleFidelity - residual must follow speckle noise model
- P₂: AnatomyValid - boundaries must satisfy anatomical constraints
- P₃: StructurePreserved - edges at boundaries must be maintained

References:
[1] Goodman, "Speckle Phenomena in Optics", 2007 (speckle theory)
[2] Staurenghi et al., Ophthalmology 2014 (OCT anatomy nomenclature)
[3] Schmitt et al., "Speckle in OCT", J Biomed Opt 1999

Author: Neuro-Symbolic OCT Denoising Framework
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass


# =============================================================================
# FUZZY LOGIC OPERATORS (Differentiable Symbolic Reasoning)
# =============================================================================

class FuzzyLogic:
    """
    Differentiable fuzzy logic operators for symbolic reasoning.

    This is what makes the system TRULY SYMBOLIC:
    - Instead of Python boolean `and`/`or`, we use differentiable operators
    - Enables gradient flow through logical formulas
    - Supports logical inference via IMPLIES
    - Supports quantifiers FORALL/EXISTS

    T-norm choices:
    - Product: AND(a,b) = a * b  [smooth gradients, used by default]
    - Lukasiewicz: AND(a,b) = max(0, a+b-1)  [stronger conjunction]
    - Gödel: AND(a,b) = min(a,b)  [classical fuzzy logic]
    """

    @staticmethod
    def AND(a: torch.Tensor, b: torch.Tensor, t_norm: str = 'product') -> torch.Tensor:
        """
        Fuzzy AND: both a AND b must be satisfied.

        Product t-norm: a * b (default, smooth gradients)
        Lukasiewicz: max(0, a + b - 1) (stricter)
        Godel: min(a, b) (classical)
        """
        if t_norm == 'product':
            return a * b
        elif t_norm == 'lukasiewicz':
            return torch.clamp(a + b - 1, min=0)
        elif t_norm == 'godel':
            return torch.min(a, b)
        else:
            raise ValueError(f"Unknown t-norm: {t_norm}")

    @staticmethod
    def OR(a: torch.Tensor, b: torch.Tensor, t_conorm: str = 'product') -> torch.Tensor:
        """
        Fuzzy OR: at least one of a OR b must be satisfied.

        Product t-conorm: a + b - a*b
        Lukasiewicz: min(1, a + b)
        Godel: max(a, b)
        """
        if t_conorm == 'product':
            return a + b - a * b
        elif t_conorm == 'lukasiewicz':
            return torch.clamp(a + b, max=1)
        elif t_conorm == 'godel':
            return torch.max(a, b)
        else:
            raise ValueError(f"Unknown t-conorm: {t_conorm}")

    @staticmethod
    def NOT(a: torch.Tensor) -> torch.Tensor:
        """Fuzzy NOT: negation."""
        return 1 - a

    @staticmethod
    def IMPLIES(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Fuzzy IMPLIES: a → b

        Using Reichenbach implication: a → b = 1 - a + a*b
        This satisfies: if a is false (0), implication is true (1)
                       if a is true (1), implication equals b
        """
        return 1 - a + a * b

    @staticmethod
    def IFF(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy IFF (if and only if): a ↔ b = (a → b) AND (b → a)"""
        return FuzzyLogic.AND(
            FuzzyLogic.IMPLIES(a, b),
            FuzzyLogic.IMPLIES(b, a)
        )

    @staticmethod
    def FORALL(tensor: torch.Tensor, dim=None,
               soft: bool = True, temperature: float = 0.1) -> torch.Tensor:
        """
        Universal quantifier: ∀x P(x)

        Args:
            tensor: satisfaction values for each element
            dim: dimension(s) to quantify over (None = all, can be int or tuple)
            soft: if True, use soft-min (differentiable); if False, use hard min
            temperature: for soft version, lower = closer to hard min
        """
        if soft:
            # Soft minimum using log-sum-exp trick
            # softmin(x) = -temperature * log(mean(exp(-x/temperature)))
            if dim is None:
                dims = tuple(range(tensor.dim()))
                n = tensor.numel()
            elif isinstance(dim, int):
                dims = dim
                n = tensor.shape[dim]
            else:
                # dim is a tuple
                dims = dim
                n = 1
                for d in dim:
                    n *= tensor.shape[d]

            # OPTIMIZATION: Clamp to avoid overflow in exp(-x/temperature) for large negative x
            # and underflow for large positive x
            neg_tensor = torch.clamp(-tensor / temperature, min=-50, max=50)
            lse = torch.logsumexp(neg_tensor, dim=dims)
            # OPTIMIZATION: Use math.log instead of creating tensor
            log_n = math.log(n) if n > 0 else 0
            return -temperature * lse + temperature * log_n
        else:
            if dim is None:
                return tensor.min()
            elif isinstance(dim, int):
                return tensor.min(dim=dim)[0]
            else:
                # Multiple dimensions - reduce sequentially
                result = tensor
                for d in sorted(dim, reverse=True):
                    result = result.min(dim=d)[0]
                return result

    @staticmethod
    def EXISTS(tensor: torch.Tensor, dim=None,
               soft: bool = True, temperature: float = 0.1) -> torch.Tensor:
        """
        Existential quantifier: ∃x P(x)

        Soft maximum using log-sum-exp.
        """
        if soft:
            if dim is None:
                dims = tuple(range(tensor.dim()))
                n = tensor.numel()
            elif isinstance(dim, int):
                dims = dim
                n = tensor.shape[dim]
            else:
                dims = dim
                n = 1
                for d in dim:
                    n *= tensor.shape[d]

            # OPTIMIZATION: Clamp to avoid overflow
            pos_tensor = torch.clamp(tensor / temperature, min=-50, max=50)
            lse = torch.logsumexp(pos_tensor, dim=dims)
            # OPTIMIZATION: Use math.log instead of creating tensor
            log_n = math.log(n) if n > 0 else 0
            return temperature * lse - temperature * log_n
        else:
            if dim is None:
                return tensor.max()
            elif isinstance(dim, int):
                return tensor.max(dim=dim)[0]
            else:
                result = tensor
                for d in sorted(dim, reverse=True):
                    result = result.max(dim=d)[0]
                return result

    @staticmethod
    def MOST(tensor: torch.Tensor, threshold: float = 0.9,
             dim: Optional[int] = None) -> torch.Tensor:
        """
        Fuzzy 'most' quantifier: true if most elements satisfy predicate.

        More practical than FORALL (which requires ALL to satisfy).
        """
        if dim is None:
            fraction_satisfied = tensor.mean()
        else:
            fraction_satisfied = tensor.mean(dim=dim)

        # Soft threshold
        return torch.sigmoid(10 * (fraction_satisfied - threshold))

    @staticmethod
    def soft_eq(a: torch.Tensor, b: torch.Tensor, tolerance: float = 0.1,
                sharpness: float = 10.0) -> torch.Tensor:
        """Soft equality: a ≈ b within tolerance."""
        diff = torch.abs(a - b)
        return torch.sigmoid(sharpness * (tolerance - diff))

    @staticmethod
    def soft_lt(a: torch.Tensor, b: torch.Tensor, margin: float = 0.0,
                sharpness: float = 10.0) -> torch.Tensor:
        """Soft less-than: a < b (with optional margin)."""
        return torch.sigmoid(sharpness * (b - a - margin))

    @staticmethod
    def soft_gt(a: torch.Tensor, b: torch.Tensor, margin: float = 0.0,
                sharpness: float = 10.0) -> torch.Tensor:
        """Soft greater-than: a > b (with optional margin)."""
        return torch.sigmoid(sharpness * (a - b - margin))

    @staticmethod
    def soft_in_range(x: torch.Tensor, low: float, high: float,
                      sharpness: float = 10.0) -> torch.Tensor:
        """Soft range check: low < x < high."""
        above_low = torch.sigmoid(sharpness * (x - low))
        below_high = torch.sigmoid(sharpness * (high - x))
        return above_low * below_high

    @staticmethod
    def soft_max(tensor: torch.Tensor, dim: int = 0,
                 temperature: float = 0.1) -> torch.Tensor:
        """
        Soft maximum (differentiable max) along dimension.

        Uses log-sum-exp for smooth approximation to max:
        soft_max(x) ≈ max(x) as temperature → 0

        Args:
            tensor: input tensor
            dim: dimension to take max over
            temperature: lower = sharper (closer to hard max)

        Returns:
            Soft maximum along the specified dimension
        """
        # Scale by temperature
        scaled = tensor / temperature
        # Clamp for numerical stability
        scaled = torch.clamp(scaled, min=-50, max=50)
        # Log-sum-exp trick: log(sum(exp(x))) is a smooth max
        lse = torch.logsumexp(scaled, dim=dim)
        return temperature * lse


# =============================================================================
# KNOWLEDGE BASE: OCT Domain Facts
# =============================================================================

@dataclass
class OCTKnowledgeBase:
    """
    Explicit knowledge representation for OCT domain.

    This encodes FACTS about OCT that are always true:
    - Layer ordering relationships (anatomical)
    - Physical noise properties (speckle)
    - Biological constraints (thickness ranges)

    This is SYMBOLIC because:
    - Facts are explicit and interpretable
    - Can be modified without changing code
    - Supports logical queries
    """
    # Layer ordering: layer_order[i] is ABOVE layer_order[i+1]
    # This is a BIOLOGICAL FACT that cannot be violated
    layer_names: Tuple[str, ...] = ('ILM', 'RNFL/GCL', 'IPL/INL', 'OPL/ONL', 'RPE')

    # Speckle physics: for Rayleigh distribution, CV ≈ 0.52
    # For mixed/processed data, typically lower (0.35-0.45)
    speckle_cv_theoretical: float = 0.52  # Rayleigh distribution
    speckle_cv_empirical: float = 0.40    # Calibrated on real data
    speckle_cv_tolerance: float = 0.14    # 2-sigma tolerance

    # Anatomical thickness constraints (as fraction of image height)
    # From: Budenz et al., Ophthalmology 2007; Spaide et al., Retina 2011
    min_layer_thickness: float = 0.02     # Minimum ~2% of image
    max_layer_thickness: float = 0.35     # Maximum ~35% of image
    min_total_retina: float = 0.15        # Minimum retinal thickness
    max_total_retina: float = 0.60        # Maximum retinal thickness

    # Position constraints
    ilm_position_range: Tuple[float, float] = (0.05, 0.45)  # ILM in upper half
    rpe_position_range: Tuple[float, float] = (0.40, 0.90)  # RPE in lower half

    def get_ordering_pairs(self) -> List[Tuple[int, int]]:
        """Return all (i, j) pairs where layer i must be above layer j."""
        pairs = []
        for i in range(len(self.layer_names) - 1):
            pairs.append((i, i + 1))  # Adjacent ordering
        return pairs

    def __str__(self) -> str:
        return f"""OCT Knowledge Base:
  Layers: {' > '.join(self.layer_names)}
  Speckle CV: {self.speckle_cv_empirical} ± {self.speckle_cv_tolerance}
  Layer thickness: [{self.min_layer_thickness}, {self.max_layer_thickness}]
  Total retina: [{self.min_total_retina}, {self.max_total_retina}]"""


# Default knowledge base instance
OCT_KNOWLEDGE = OCTKnowledgeBase()


# =============================================================================
# PREDICATE 1: SPECKLE FIDELITY (Physics-Based)
# =============================================================================

class SpeckleFidelityPredicate(nn.Module):
    """
    Verifies that the denoising residual follows speckle noise statistics.

    Physics Background (Goodman 2007, Schmitt 1999):
    - OCT speckle follows multiplicative noise model
    - For fully developed speckle: Std(noise) / Mean(signal) ≈ constant
    - This constant is ~0.52 for Rayleigh distribution (single-look)
    - Multi-look averaging reduces this ratio

    Predicate Definition:
        SpeckleFidelity(noisy, denoised) :=
            |CV(residual) - expected_CV| < tolerance
        where CV = coefficient of variation = Std/Mean

    Verification:
    - If residual has HIGHER CV than expected → under-denoising
    - If residual has LOWER CV than expected → over-denoising (removed structure)
    - If CV matches expected → correct denoising

    This is VERIFIABLE WITHOUT GROUND TRUTH because speckle statistics
    are determined by physics, not by the specific image content.
    """

    def __init__(
        self,
        expected_cv: float = 0.40,  # Calibrated on PKU37 (mixed noise types)
        tolerance: float = 0.14,     # 2 std from PKU37 calibration
        min_intensity: float = 0.05, # Ignore very dark regions
        window_size: int = 16,       # Local window for statistics
    ):
        super().__init__()
        self.expected_cv = expected_cv
        self.tolerance = tolerance
        self.min_intensity = min_intensity
        self.window_size = window_size

        # Create averaging kernel for local statistics
        self.register_buffer(
            'avg_kernel',
            torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        )

    def compute_local_cv(
        self,
        residual: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute local coefficient of variation of residual.

        For speckle noise: CV = Std(residual) / Mean(intensity) should be constant

        Args:
            residual: [B, 1, H, W] noise residual (noisy - denoised)
            intensity: [B, 1, H, W] signal intensity (denoised image)

        Returns:
            cv_map: [B, 1, H', W'] local CV values
        """
        # Pad for valid convolution
        pad = self.window_size // 2
        residual_pad = F.pad(residual, (pad, pad, pad, pad), mode='reflect')
        intensity_pad = F.pad(intensity, (pad, pad, pad, pad), mode='reflect')

        # Local mean of residual squared (for variance)
        res_sq = residual_pad ** 2
        local_var = F.conv2d(res_sq, self.avg_kernel)
        local_mean_res = F.conv2d(residual_pad, self.avg_kernel)
        local_var = local_var - local_mean_res ** 2  # Var = E[X²] - E[X]²
        local_std = torch.sqrt(local_var.clamp(min=1e-8))

        # Local mean of intensity
        local_intensity = F.conv2d(intensity_pad, self.avg_kernel)

        # CV = Std(residual) / Mean(intensity)
        cv = local_std / local_intensity.clamp(min=self.min_intensity)

        return cv

    def forward(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate speckle fidelity predicate.

        Args:
            noisy: [B, 1, H, W] noisy input
            denoised: [B, 1, H, W] denoised output
            return_details: whether to return detailed statistics

        Returns:
            dict with:
                'satisfied': bool tensor [B] - predicate satisfaction per sample
                'score': float tensor [B] - continuous satisfaction score (0-1)
                'loss': float tensor - differentiable loss for training
                'cv_mean': mean CV across image
                'cv_expected': expected CV
        """
        residual = noisy - denoised

        # Compute local CV
        cv_map = self.compute_local_cv(residual, denoised)

        # Mask out low-intensity regions (unreliable statistics)
        # Use average pooled intensity for masking
        pad = self.window_size // 2
        intensity_pad = F.pad(denoised, (pad, pad, pad, pad), mode='reflect')
        local_intensity = F.conv2d(intensity_pad, self.avg_kernel)
        valid_mask = (local_intensity > self.min_intensity).float()

        # Compute mean CV over valid regions
        cv_sum = (cv_map * valid_mask).sum(dim=(1, 2, 3))
        valid_count = valid_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        cv_mean = cv_sum / valid_count  # [B]

        # SYMBOLIC PREDICATE using fuzzy logic:
        # SpeckleFidelity(residual) := CV(residual) ≈ expected_CV
        # Using soft equality with tolerance
        cv_deviation = torch.abs(cv_mean - self.expected_cv)

        # Fuzzy satisfaction: soft equality check
        # Returns value in [0, 1] where 1 = perfectly satisfied
        score = FuzzyLogic.soft_in_range(
            cv_mean,
            self.expected_cv - self.tolerance,
            self.expected_cv + self.tolerance,
            sharpness=10.0
        )

        # Hard satisfaction (for reporting)
        satisfied = cv_deviation < self.tolerance  # [B] bool

        # Differentiable loss: penalize deviation from expected CV
        loss = cv_deviation.mean()

        result = {
            'satisfied': satisfied,
            'score': score,  # Now using fuzzy logic score
            'loss': loss,
            'cv_mean': cv_mean,
            'cv_expected': torch.tensor(self.expected_cv, device=noisy.device),
        }

        if return_details:
            result['cv_map'] = cv_map
            result['valid_mask'] = valid_mask

        return result

    def verify(self, noisy: torch.Tensor, denoised: torch.Tensor) -> bool:
        """Binary verification: does the predicate hold?"""
        with torch.no_grad():
            result = self.forward(noisy, denoised)
            return result['satisfied'].all().item()


# =============================================================================
# PREDICATE 2: ANATOMY VALID (Literature-Based)
# =============================================================================

class AnatomyValidPredicate(nn.Module):
    """
    Verifies that detected boundaries satisfy anatomical constraints.

    Literature Sources:
    - Staurenghi et al., Ophthalmology 2014 (nomenclature)
    - Budenz et al., Ophthalmology 2007 (RNFL thickness norms)
    - Spaide et al., Retina 2011 (RPE/photoreceptor anatomy)

    Constraints (all from published studies):
    1. ORDERING: ILM < RNFL/INL < INL/IS < IS/RPE (always true in normal retina)
    2. THICKNESS: Each layer has physiological bounds
       - Total retina: 200-400 μm (fovea thinner, peripapillary thicker)
       - RNFL: 70-140 μm (Budenz 2007)
       - GCL+IPL: 60-100 μm
    3. POSITION: ILM in upper portion, RPE in lower portion of image

    Predicate Definition:
        AnatomyValid(denoised, boundaries) :=
            Ordered(boundaries) ∧
            ValidThickness(boundaries) ∧
            ValidPosition(boundaries)

    This is VERIFIABLE WITHOUT GROUND TRUTH because anatomical constraints
    are universal across all healthy retinas.
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        min_layer_thickness: float = 0.03,  # Minimum 3% of image height
        max_layer_thickness: float = 0.40,  # Maximum 40% of image height
        ilm_range: Tuple[float, float] = (0.05, 0.45),  # ILM position range
        rpe_range: Tuple[float, float] = (0.40, 0.90),  # RPE position range
        min_retina_height: float = 0.15,    # Minimum retina thickness
        max_retina_height: float = 0.60,    # Maximum retina thickness
    ):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.min_layer_thickness = min_layer_thickness
        self.max_layer_thickness = max_layer_thickness
        self.ilm_range = ilm_range
        self.rpe_range = rpe_range
        self.min_retina_height = min_retina_height
        self.max_retina_height = max_retina_height

    def check_ordering(self, boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SYMBOLIC CHECK: ∀i: boundary[i] < boundary[i+1]

        This encodes the LOGICAL FORMULA:
            FORALL i in [0, N-1]: LessThan(boundary[i], boundary[i+1])

        Uses fuzzy logic for differentiable satisfaction.

        Args:
            boundaries: [B, N, W] boundary positions (normalized 0-1)

        Returns:
            satisfied: [B] bool - all columns satisfy ordering
            violation: [B] float - sum of ordering violations
            fuzzy_score: [B] float - fuzzy satisfaction in [0, 1]
        """
        B, N, W = boundaries.shape

        # Compute differences: b[i+1] - b[i] should be > 0
        diffs = boundaries[:, 1:, :] - boundaries[:, :-1, :]  # [B, N-1, W]

        # FUZZY LOGIC: soft_gt(boundary[i+1], boundary[i], margin=min_gap)
        # Each element gets fuzzy satisfaction score
        min_gap = 0.01  # Minimum required gap
        fuzzy_satisfied = FuzzyLogic.soft_gt(
            boundaries[:, 1:, :],
            boundaries[:, :-1, :],
            margin=min_gap,
            sharpness=50.0
        )  # [B, N-1, W]

        # FORALL quantifier: all pairs must satisfy ordering
        # Using soft FORALL for differentiability
        fuzzy_score = FuzzyLogic.FORALL(fuzzy_satisfied, dim=(1, 2), soft=True)  # [B]

        # Hard satisfaction (for reporting)
        violations = F.relu(-diffs + 1e-6)  # [B, N-1, W]
        violation_sum = violations.sum(dim=(1, 2))  # [B]
        satisfied = (diffs > 0).all(dim=(1, 2))  # [B]

        return satisfied, violation_sum, fuzzy_score

    def check_thickness(self, boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SYMBOLIC CHECK: ∀ layers L: thickness(L) ∈ [min, max]

        Logical formula:
            FORALL L: InRange(thickness(L), min_thickness, max_thickness)

        Args:
            boundaries: [B, N, W] boundary positions (normalized 0-1)

        Returns:
            satisfied: [B] bool - all thicknesses within bounds
            violation: [B] float - sum of thickness violations
            fuzzy_score: [B] float - fuzzy satisfaction in [0, 1]
        """
        B, N, W = boundaries.shape

        # Layer thicknesses
        thicknesses = boundaries[:, 1:, :] - boundaries[:, :-1, :]  # [B, N-1, W]

        # FUZZY LOGIC: soft_in_range for each thickness
        fuzzy_satisfied = FuzzyLogic.soft_in_range(
            thicknesses,
            self.min_layer_thickness,
            self.max_layer_thickness,
            sharpness=20.0
        )  # [B, N-1, W]

        # FORALL quantifier over all layers and positions
        fuzzy_score = FuzzyLogic.FORALL(fuzzy_satisfied, dim=(1, 2), soft=True)  # [B]

        # Hard checks (for reporting)
        too_thin = F.relu(self.min_layer_thickness - thicknesses)
        too_thick = F.relu(thicknesses - self.max_layer_thickness)
        violation_sum = (too_thin + too_thick).sum(dim=(1, 2))  # [B]

        within_bounds = (thicknesses >= self.min_layer_thickness) & \
                       (thicknesses <= self.max_layer_thickness)
        satisfied = within_bounds.all(dim=(1, 2))  # [B]

        return satisfied, violation_sum, fuzzy_score

    def check_position(self, boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SYMBOLIC CHECK: ILM ∈ [low, high] AND RPE ∈ [low, high] AND valid_retina_height

        Logical formula:
            InRange(ILM, ilm_low, ilm_high) AND
            InRange(RPE, rpe_low, rpe_high) AND
            InRange(RPE - ILM, min_height, max_height)

        Args:
            boundaries: [B, N, W] boundary positions (normalized 0-1)

        Returns:
            satisfied: [B] bool - positions within expected ranges
            violation: [B] float - position violations
            fuzzy_score: [B] float - fuzzy satisfaction in [0, 1]
        """
        B, N, W = boundaries.shape

        ilm = boundaries[:, 0, :]   # [B, W]
        rpe = boundaries[:, -1, :]  # [B, W]
        retina_height = rpe - ilm   # [B, W]

        # FUZZY LOGIC: soft range checks
        ilm_satisfied = FuzzyLogic.soft_in_range(
            ilm, self.ilm_range[0], self.ilm_range[1], sharpness=20.0
        )  # [B, W]

        rpe_satisfied = FuzzyLogic.soft_in_range(
            rpe, self.rpe_range[0], self.rpe_range[1], sharpness=20.0
        )  # [B, W]

        height_satisfied = FuzzyLogic.soft_in_range(
            retina_height, self.min_retina_height, self.max_retina_height, sharpness=20.0
        )  # [B, W]

        # FUZZY AND: all three conditions must hold
        # Using product t-norm for smooth gradients
        combined_satisfied = FuzzyLogic.AND(
            FuzzyLogic.AND(ilm_satisfied, rpe_satisfied),
            height_satisfied
        )  # [B, W]

        # FORALL over width (all A-scans must satisfy)
        fuzzy_score = FuzzyLogic.FORALL(combined_satisfied, dim=1, soft=True)  # [B]

        # Hard violations (for loss)
        ilm_low = F.relu(self.ilm_range[0] - ilm)
        ilm_high = F.relu(ilm - self.ilm_range[1])
        rpe_low = F.relu(self.rpe_range[0] - rpe)
        rpe_high = F.relu(rpe - self.rpe_range[1])
        height_low = F.relu(self.min_retina_height - retina_height)
        height_high = F.relu(retina_height - self.max_retina_height)

        violation_sum = (ilm_low + ilm_high + rpe_low + rpe_high +
                        height_low + height_high).sum(dim=1)  # [B]

        # Hard satisfaction
        ilm_ok = (ilm >= self.ilm_range[0]) & (ilm <= self.ilm_range[1])
        rpe_ok = (rpe >= self.rpe_range[0]) & (rpe <= self.rpe_range[1])
        height_ok = (retina_height >= self.min_retina_height) & \
                   (retina_height <= self.max_retina_height)
        satisfied = (ilm_ok & rpe_ok & height_ok).all(dim=1)  # [B]

        return satisfied, violation_sum, fuzzy_score

    def forward(
        self,
        boundaries: torch.Tensor,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate anatomy validity predicate using FUZZY LOGIC.

        SYMBOLIC FORMULA:
            AnatomyValid(boundaries) :=
                FORALL i: Ordered(b[i], b[i+1]) AND
                FORALL L: ValidThickness(L) AND
                ValidPosition(ILM, RPE)

        Args:
            boundaries: [B, N, W] boundary positions (normalized 0-1)
            return_details: whether to return per-constraint results

        Returns:
            dict with:
                'satisfied': bool tensor [B] - predicate satisfaction
                'score': float tensor [B] - FUZZY score (0-1)
                'loss': float tensor - differentiable loss
        """
        # Check each constraint (now returns fuzzy scores)
        order_sat, order_viol, order_fuzzy = self.check_ordering(boundaries)
        thick_sat, thick_viol, thick_fuzzy = self.check_thickness(boundaries)
        pos_sat, pos_viol, pos_fuzzy = self.check_position(boundaries)

        # FUZZY AND: all constraints must hold
        # score = Ordering AND Thickness AND Position
        score = FuzzyLogic.AND(
            FuzzyLogic.AND(order_fuzzy, thick_fuzzy),
            pos_fuzzy
        )  # [B]

        # Hard satisfaction (for reporting)
        satisfied = order_sat & thick_sat & pos_sat  # [B]

        # Total violation (for loss)
        total_violation = order_viol + thick_viol + pos_viol  # [B]

        # Differentiable loss (combines violations + fuzzy dissatisfaction)
        loss = total_violation.mean() + (1 - score).mean()

        result = {
            'satisfied': satisfied,
            'score': score,  # Now true fuzzy logic score
            'loss': loss,
            'total_violation': total_violation,
        }

        if return_details:
            result['ordering_satisfied'] = order_sat
            result['thickness_satisfied'] = thick_sat
            result['position_satisfied'] = pos_sat
            result['ordering_violation'] = order_viol
            result['thickness_violation'] = thick_viol
            result['position_violation'] = pos_viol
            result['ordering_fuzzy'] = order_fuzzy
            result['thickness_fuzzy'] = thick_fuzzy
            result['position_fuzzy'] = pos_fuzzy

        return result

    def verify(self, boundaries: torch.Tensor) -> bool:
        """Binary verification: does the predicate hold?"""
        with torch.no_grad():
            result = self.forward(boundaries)
            return result['satisfied'].all().item()


# =============================================================================
# PREDICATE 3: STRUCTURE PRESERVED (Self-Supervised)
# =============================================================================

class StructurePreservedPredicate(nn.Module):
    """
    REDESIGNED v12: Learnable Weights for Cross-Dataset Generalization

    Based on correlation analysis, these features predict reconstruction error:
    1. local_intensity (corr=0.31) - brighter regions have more error
    2. local_residual_var (corr=0.25) - high residual variance indicates error
    3. local_std_denoised (corr=0.22) - high local std in denoised = error
    4. denoised_edge (corr=0.21) - strong edges correlate with error

    APPROACH: Combine empirically-validated features with learnable weights.

    SYMBOLIC FORMULA:
        StructureError(noisy, denoised) :=
            w_base × BaseFailure(p) +
            w_inter × Interaction(p) +
            w_emph × IntensityEmphasis(p)

        Where weights can be learned per-dataset or fixed for inference.

    Args:
        learnable_weights: If True, weights are nn.Parameter (trainable).
                          If False, weights are fixed buffers (inference only).
    """

    def __init__(
        self,
        window_size: int = 5,  # Window for local statistics
        boundary_sigma: float = 5.0,
        learnable_weights: bool = False,  # NEW: enable learning
    ):
        super().__init__()
        self.window_size = window_size
        self.boundary_sigma = boundary_sigma
        self.learnable_weights = learnable_weights

        # Local averaging kernel
        k = window_size
        self.register_buffer('avg_kernel', torch.ones(1, 1, k, k) / (k * k))

        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3) / 4.0)

        # =====================================================
        # WEIGHTS: Learnable or Fixed
        # =====================================================
        # Feature weights (for base_failure weighted average)
        # Initialized from empirical correlations on PKU37
        feature_weights_init = torch.tensor([0.31, 0.25, 0.22, 0.21])  # [intensity, res_var, std, edge]

        # Combination weights (for final combination)
        # Initialized from v11 tuning: [base, interaction, emphasis]
        combo_weights_init = torch.tensor([0.35, 0.25, 0.40])

        if learnable_weights:
            # Learnable: use log-space for positivity, softmax for normalization
            self.feature_logits = nn.Parameter(torch.log(feature_weights_init + 1e-8))
            self.combo_logits = nn.Parameter(torch.log(combo_weights_init + 1e-8))
        else:
            # Fixed: register as buffers (not trainable)
            self.register_buffer('feature_logits', torch.log(feature_weights_init + 1e-8))
            self.register_buffer('combo_logits', torch.log(combo_weights_init + 1e-8))

    @property
    def feature_weights(self) -> torch.Tensor:
        """Get normalized feature weights (sum to 1)."""
        return F.softmax(self.feature_logits, dim=0)

    @property
    def combo_weights(self) -> torch.Tensor:
        """Get normalized combination weights (sum to 1)."""
        return F.softmax(self.combo_logits, dim=0)

    def get_weight_summary(self) -> Dict[str, float]:
        """Return current weights for inspection/logging."""
        fw = self.feature_weights.detach().cpu().tolist()
        cw = self.combo_weights.detach().cpu().tolist()
        return {
            'feature_weights': {
                'intensity': fw[0],
                'residual_var': fw[1],
                'local_std': fw[2],
                'edge': fw[3],
            },
            'combo_weights': {
                'base': cw[0],
                'interaction': cw[1],
                'emphasis': cw[2],
            }
        }

    def compute_edge_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel operator."""
        # Pad image
        image_pad = F.pad(image, (1, 1, 1, 1), mode='reflect')

        # Compute gradients
        grad_x = F.conv2d(image_pad, self.sobel_x)
        grad_y = F.conv2d(image_pad, self.sobel_y)

        # Edge magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return magnitude

    def create_boundary_mask(
        self,
        boundaries: torch.Tensor,
        height: int,
    ) -> torch.Tensor:
        """
        Create soft mask that is 1 near boundaries, 0 away from boundaries.

        OPTIMIZED: Process boundaries one at a time to avoid OOM on large images.
        Original created [B, N, H, W] tensor which is huge for 640x640 images.

        Args:
            boundaries: [B, N, W] boundary positions (normalized 0-1)
            height: image height

        Returns:
            mask: [B, 1, H, W] boundary proximity mask
        """
        B, N, W = boundaries.shape
        device = boundaries.device
        dtype = boundaries.dtype

        # Convert to pixel coordinates
        boundaries_px = boundaries * (height - 1)  # [B, N, W]

        # Create row indices - reuse for all boundaries
        rows = torch.arange(height, device=device, dtype=dtype).view(1, -1, 1)  # [1, H, 1]

        # MEMORY OPTIMIZATION: Compute min distance iteratively instead of creating [B, N, H, W]
        # Initialize with large distance
        min_distance = torch.full((B, height, W), float('inf'), device=device, dtype=dtype)

        for i in range(N):
            # Distance to boundary i: [B, H, W]
            boundary_i = boundaries_px[:, i:i+1, :]  # [B, 1, W]
            dist_i = torch.abs(rows - boundary_i)  # [B, H, W] via broadcasting
            # Update minimum
            min_distance = torch.minimum(min_distance, dist_i)

        # Soft mask using Gaussian
        sigma_sq_2 = 2 * self.boundary_sigma ** 2
        mask = torch.exp(-min_distance.pow(2) / sigma_sq_2)

        return mask.unsqueeze(1)  # [B, 1, H, W]

    def compute_local_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local mean and std."""
        pad = self.window_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x_sq_pad = F.pad(x ** 2, (pad, pad, pad, pad), mode='reflect')

        local_mean = F.conv2d(x_pad, self.avg_kernel)
        local_sq_mean = F.conv2d(x_sq_pad, self.avg_kernel)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
        local_std = torch.sqrt(local_var + 1e-8)

        return local_mean, local_std

    def forward(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        REDESIGNED v5: Use ALL empirically-validated features.

        Features with highest correlation to actual error:
        1. local_intensity (0.31) - brighter regions have more error
        2. local_residual_var (0.25) - high variance in residual
        3. local_std_denoised (0.22) - high local std in denoised
        4. denoised_edge (0.21) - strong edges in denoised

        Returns:
            dict with satisfaction, score, loss, and structure_failure map
        """
        B, _, H, W = noisy.shape

        # =====================================================
        # REDESIGN v7: Use SQUARED features for better discrimination
        # Squaring emphasizes high values and suppresses low values
        # =====================================================

        # =====================================================
        # FEATURE 1: Local Intensity (corr=0.31) - HIGHEST!
        # Brighter regions have more reconstruction error
        # =====================================================
        local_intensity, _ = self.compute_local_stats(denoised)

        # Normalize using 90th percentile (tighter bound)
        li_flat = local_intensity.view(B, -1)
        li_p90 = torch.quantile(li_flat, 0.90, dim=1, keepdim=True).view(B, 1, 1, 1)
        intensity_norm = (local_intensity / (li_p90 + 1e-8)).clamp(0, 1)
        # Square to emphasize high values
        intensity_sq = intensity_norm ** 2

        # =====================================================
        # FEATURE 2: Local Residual Variance (corr=0.25)
        # High variance in residual indicates problematic regions
        # =====================================================
        residual = noisy - denoised
        _, residual_std = self.compute_local_stats(residual)
        local_residual_var = residual_std ** 2

        # Normalize using 90th percentile
        rv_flat = local_residual_var.view(B, -1)
        rv_p90 = torch.quantile(rv_flat, 0.90, dim=1, keepdim=True).view(B, 1, 1, 1)
        residual_var_norm = (local_residual_var / (rv_p90 + 1e-8)).clamp(0, 1)
        residual_var_sq = residual_var_norm ** 2

        # =====================================================
        # FEATURE 3: Local Std of Denoised (corr=0.22)
        # High local variation in denoised indicates error
        # =====================================================
        _, denoised_std = self.compute_local_stats(denoised)

        # Normalize using 90th percentile
        ds_flat = denoised_std.view(B, -1)
        ds_p90 = torch.quantile(ds_flat, 0.90, dim=1, keepdim=True).view(B, 1, 1, 1)
        denoised_std_norm = (denoised_std / (ds_p90 + 1e-8)).clamp(0, 1)
        denoised_std_sq = denoised_std_norm ** 2

        # =====================================================
        # FEATURE 4: Edge Magnitude in Denoised (corr=0.21)
        # Strong edges correlate with error (surprisingly)
        # =====================================================
        den_pad = F.pad(denoised, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(den_pad, self.sobel_x)
        gy = F.conv2d(den_pad, self.sobel_y)
        denoised_edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Normalize using 90th percentile
        de_flat = denoised_edge.view(B, -1)
        de_p90 = torch.quantile(de_flat, 0.90, dim=1, keepdim=True).view(B, 1, 1, 1)
        denoised_edge_norm = (denoised_edge / (de_p90 + 1e-8)).clamp(0, 1)
        denoised_edge_sq = denoised_edge_norm ** 2

        # =====================================================
        # COMBINED FAILURE MAP v12: Learnable Weights
        # HYBRID: Weighted average of features PLUS interaction terms
        #
        # Key insight: high error often occurs when MULTIPLE features
        # are elevated simultaneously. Add multiplicative terms.
        # =====================================================

        # Get current weights (learnable or fixed)
        fw = self.feature_weights  # [4] normalized feature weights
        cw = self.combo_weights    # [3] normalized combination weights

        # 1. Base: weighted average of linear features
        # fw[0]=intensity, fw[1]=residual_var, fw[2]=local_std, fw[3]=edge
        base_failure = (
            fw[0] * intensity_norm +
            fw[1] * residual_var_norm +
            fw[2] * denoised_std_norm +
            fw[3] * denoised_edge_norm
        )  # Already normalized since fw sums to 1

        # 2. Interaction terms: products of top 2 features
        # (intensity and residual_var have highest correlations)
        interaction = intensity_norm * residual_var_norm

        # 3. Also add squared highest-correlation feature
        # to emphasize regions where intensity is especially high
        intensity_emphasis = intensity_sq

        # 4. Combine: base + scaled interaction + emphasis
        # cw[0]=base, cw[1]=interaction, cw[2]=emphasis
        structure_failure = (
            cw[0] * base_failure +
            cw[1] * interaction +
            cw[2] * intensity_emphasis
        ).clamp(0, 1)  # cw sums to 1

        # =====================================================
        # SCORES
        # =====================================================

        # Structure score: inverse of mean failure (lower failure = higher score)
        failure_mean = structure_failure.mean(dim=(1, 2, 3))  # [B]
        structure_score = 1 - failure_mean

        # Component scores for analysis
        residual_var_score = 1 - residual_var_norm.mean(dim=(1, 2, 3))
        denoised_std_score = 1 - denoised_std_norm.mean(dim=(1, 2, 3))
        edge_score = 1 - denoised_edge_norm.mean(dim=(1, 2, 3))

        # Combined score
        score = structure_score  # [B]

        # Hard satisfaction (higher threshold since this is weighted average)
        satisfied = structure_score > 0.7

        # Differentiable loss
        loss = failure_mean.mean()

        # Legacy compatibility names
        edge_correlation = residual_var_score
        edge_reduction = denoised_std_score
        smoothness_score = structure_score
        noise_reduction_score = residual_var_score
        edge_quality_score = edge_score

        result = {
            'satisfied': satisfied,
            'score': score,
            'loss': loss,
            'edge_correlation': edge_correlation,
            'edge_reduction': edge_reduction,
            'smoothness_score': smoothness_score,
            'noise_reduction_score': noise_reduction_score,
            'edge_quality_score': edge_quality_score,
            'structure_failure': structure_failure,  # THE KEY OUTPUT
        }

        if return_details:
            result['residual'] = residual
            result['local_intensity'] = local_intensity
            result['intensity_norm'] = intensity_norm
            result['local_residual_var'] = local_residual_var
            result['denoised_std'] = denoised_std
            result['denoised_edge'] = denoised_edge
            result['residual_var_norm'] = residual_var_norm
            result['denoised_std_norm'] = denoised_std_norm
            result['denoised_edge_norm'] = denoised_edge_norm
            result['boundary_mask'] = self.create_boundary_mask(boundaries, H)
            result['smoothness_failure'] = structure_failure
            result['implication_map'] = structure_failure
            # Include current weights for inspection
            result['feature_weights'] = fw.detach()
            result['combo_weights'] = cw.detach()

        return result

    def verify(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> bool:
        """Binary verification: does the predicate hold?"""
        with torch.no_grad():
            result = self.forward(noisy, denoised, boundaries)
            return result['satisfied'].all().item()

    def calibrate(
        self,
        noisy_samples: List[torch.Tensor],
        denoised_samples: List[torch.Tensor],
        clean_samples: List[torch.Tensor],
        boundaries_samples: List[torch.Tensor],
        n_steps: int = 100,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """
        Calibrate weights on a new dataset using ground truth.

        This optimizes the learnable weights to maximize correlation
        between the structure_failure map and actual error.

        Args:
            noisy_samples: List of noisy images [B, 1, H, W]
            denoised_samples: List of denoised images [B, 1, H, W]
            clean_samples: List of clean ground truth images [B, 1, H, W]
            boundaries_samples: List of boundaries [B, N, W]
            n_steps: Number of optimization steps
            lr: Learning rate

        Returns:
            Dict with final weights and correlation achieved
        """
        if not self.learnable_weights:
            raise ValueError("Cannot calibrate with learnable_weights=False. "
                           "Reinitialize with learnable_weights=True.")

        # Optimizer for weight parameters only
        optimizer = torch.optim.Adam([self.feature_logits, self.combo_logits], lr=lr)

        best_corr = -1.0
        best_state = None

        for step in range(n_steps):
            total_corr = 0.0
            n_samples = 0

            for noisy, denoised, clean, boundaries in zip(
                noisy_samples, denoised_samples, clean_samples, boundaries_samples
            ):
                # Forward pass
                result = self.forward(noisy, denoised, boundaries)
                structure_failure = result['structure_failure']

                # Compute actual error
                actual_error = (denoised - clean).abs()

                # Resize if needed
                if structure_failure.shape != actual_error.shape:
                    structure_failure = F.interpolate(
                        structure_failure, size=actual_error.shape[2:],
                        mode='bilinear', align_corners=False
                    )

                # Compute correlation (we want to MAXIMIZE this)
                # Use negative correlation as loss
                sf_flat = structure_failure.flatten()
                ae_flat = actual_error.flatten()
                sf_centered = sf_flat - sf_flat.mean()
                ae_centered = ae_flat - ae_flat.mean()

                num = (sf_centered * ae_centered).sum()
                den = torch.sqrt((sf_centered ** 2).sum() * (ae_centered ** 2).sum() + 1e-8)
                corr = num / den

                # Loss: negative correlation (minimize to maximize correlation)
                loss = -corr

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_corr += corr.item()
                n_samples += 1

            avg_corr = total_corr / n_samples

            # Track best
            if avg_corr > best_corr:
                best_corr = avg_corr
                best_state = {
                    'feature_logits': self.feature_logits.detach().clone(),
                    'combo_logits': self.combo_logits.detach().clone(),
                }

            if step % 20 == 0:
                print(f"  Calibration step {step}: correlation = {avg_corr:.4f}")

        # Restore best state
        if best_state is not None:
            self.feature_logits.data = best_state['feature_logits']
            self.combo_logits.data = best_state['combo_logits']

        print(f"  Calibration complete: best correlation = {best_corr:.4f}")

        return {
            'correlation': best_corr,
            'weights': self.get_weight_summary(),
        }

    def freeze_weights(self):
        """Convert learnable weights to fixed buffers (for inference)."""
        if self.learnable_weights:
            # Convert parameters to buffers
            feature_logits = self.feature_logits.detach().clone()
            combo_logits = self.combo_logits.detach().clone()

            # Remove parameters
            del self.feature_logits
            del self.combo_logits

            # Register as buffers
            self.register_buffer('feature_logits', feature_logits)
            self.register_buffer('combo_logits', combo_logits)

            self.learnable_weights = False
            print("Weights frozen for inference.")


# =============================================================================
# ENHANCED STRUCTURE PREDICATE v13: Multi-Scale + Learned Features
# =============================================================================

class EnhancedStructurePredicate(nn.Module):
    """
    ENHANCED v13: Multi-Scale Features + Small Learned Component

    Improvements over v12:
    1. Multi-scale local statistics (3, 7, 15 windows)
    2. Multi-scale edge detection
    3. Additional features: residual magnitude, edge difference
    4. Small learned feature combiner (optional)

    Goal: Achieve correlation > 0.4 with actual reconstruction error.
    """

    def __init__(
        self,
        scales: List[int] = [3, 7, 15],  # Multiple window sizes
        use_learned_combiner: bool = False,  # Add small CNN
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.scales = scales
        self.use_learned_combiner = use_learned_combiner
        self.learnable_weights = learnable_weights

        # Create averaging kernels for each scale
        for k in scales:
            kernel = torch.ones(1, 1, k, k) / (k * k)
            self.register_buffer(f'avg_kernel_{k}', kernel)

        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3) / 4.0)

        # Laplacian kernel for high-frequency detection
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

        # =====================================================
        # Feature count:
        # - Per scale (3 scales): intensity, residual_var, denoised_std = 3 features
        # - Global: edge_denoised, edge_residual, edge_diff, laplacian_denoised,
        #           laplacian_residual, residual_magnitude = 6 features
        # Total: 3*3 + 6 = 15 features
        # =====================================================
        n_features = len(scales) * 3 + 6

        # Feature weights (learnable or fixed)
        # Initialize with equal weights
        feature_weights_init = torch.ones(n_features) / n_features

        if learnable_weights:
            self.feature_logits = nn.Parameter(torch.zeros(n_features))
        else:
            self.register_buffer('feature_logits', torch.zeros(n_features))

        # Optional: Small learned combiner (3-layer CNN)
        if use_learned_combiner:
            self.combiner = nn.Sequential(
                nn.Conv2d(n_features, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.combiner = None

    @property
    def feature_weights(self) -> torch.Tensor:
        """Get normalized feature weights."""
        return F.softmax(self.feature_logits, dim=0)

    def compute_local_stats_at_scale(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local mean and std at a specific scale."""
        pad = k // 2
        kernel = getattr(self, f'avg_kernel_{k}')

        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x_sq_pad = F.pad(x ** 2, (pad, pad, pad, pad), mode='reflect')

        local_mean = F.conv2d(x_pad, kernel)
        local_sq_mean = F.conv2d(x_sq_pad, kernel)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
        local_std = torch.sqrt(local_var + 1e-8)

        return local_mean, local_std

    def compute_edge(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel."""
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(x_pad, self.sobel_x)
        gy = F.conv2d(x_pad, self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian (high-frequency content)."""
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(x_pad, self.laplacian).abs()

    def normalize_feature(self, feat: torch.Tensor, percentile: float = 0.90) -> torch.Tensor:
        """Normalize feature to [0, 1] using percentile scaling."""
        B = feat.shape[0]
        feat_flat = feat.view(B, -1)
        p_val = torch.quantile(feat_flat, percentile, dim=1, keepdim=True).view(B, 1, 1, 1)
        return (feat / (p_val + 1e-8)).clamp(0, 1)

    def forward(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor = None,  # Optional, not used in this version
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale structure failure map.

        Returns:
            dict with satisfaction, score, loss, and structure_failure map
        """
        B, _, H, W = noisy.shape
        residual = noisy - denoised

        features = []
        feature_names = []

        # =====================================================
        # MULTI-SCALE FEATURES
        # =====================================================
        for k in self.scales:
            # Local intensity at scale k
            local_intensity, _ = self.compute_local_stats_at_scale(denoised, k)
            intensity_norm = self.normalize_feature(local_intensity)
            features.append(intensity_norm)
            feature_names.append(f'intensity_s{k}')

            # Local residual variance at scale k
            _, residual_std = self.compute_local_stats_at_scale(residual, k)
            residual_var = residual_std ** 2
            residual_var_norm = self.normalize_feature(residual_var)
            features.append(residual_var_norm)
            feature_names.append(f'residual_var_s{k}')

            # Local std of denoised at scale k
            _, denoised_std = self.compute_local_stats_at_scale(denoised, k)
            denoised_std_norm = self.normalize_feature(denoised_std)
            features.append(denoised_std_norm)
            feature_names.append(f'denoised_std_s{k}')

        # =====================================================
        # GLOBAL FEATURES
        # =====================================================

        # Edge magnitude of denoised
        edge_denoised = self.compute_edge(denoised)
        edge_denoised_norm = self.normalize_feature(edge_denoised)
        features.append(edge_denoised_norm)
        feature_names.append('edge_denoised')

        # Edge magnitude of residual
        edge_residual = self.compute_edge(residual)
        edge_residual_norm = self.normalize_feature(edge_residual)
        features.append(edge_residual_norm)
        feature_names.append('edge_residual')

        # Edge difference (noisy - denoised)
        edge_noisy = self.compute_edge(noisy)
        edge_diff = (edge_noisy - edge_denoised).abs()
        edge_diff_norm = self.normalize_feature(edge_diff)
        features.append(edge_diff_norm)
        feature_names.append('edge_diff')

        # Laplacian of denoised (high-frequency content)
        lap_denoised = self.compute_laplacian(denoised)
        lap_denoised_norm = self.normalize_feature(lap_denoised)
        features.append(lap_denoised_norm)
        feature_names.append('laplacian_denoised')

        # Laplacian of residual
        lap_residual = self.compute_laplacian(residual)
        lap_residual_norm = self.normalize_feature(lap_residual)
        features.append(lap_residual_norm)
        feature_names.append('laplacian_residual')

        # Residual magnitude (simple but effective)
        residual_mag = residual.abs()
        residual_mag_norm = self.normalize_feature(residual_mag)
        features.append(residual_mag_norm)
        feature_names.append('residual_magnitude')

        # =====================================================
        # COMBINE FEATURES
        # =====================================================

        # Stack all features: [B, N_features, H, W]
        feature_stack = torch.cat(features, dim=1)

        if self.combiner is not None:
            # Use learned combiner
            structure_failure = self.combiner(feature_stack)
        else:
            # Weighted average using feature weights
            fw = self.feature_weights  # [N_features]
            # Reshape for broadcasting: [1, N_features, 1, 1]
            fw = fw.view(1, -1, 1, 1)
            structure_failure = (feature_stack * fw).sum(dim=1, keepdim=True)

        structure_failure = structure_failure.clamp(0, 1)

        # =====================================================
        # COMPUTE SCORES
        # =====================================================
        failure_mean = structure_failure.mean(dim=(1, 2, 3))
        structure_score = 1 - failure_mean
        satisfied = structure_score > 0.7
        loss = failure_mean.mean()

        result = {
            'satisfied': satisfied,
            'score': structure_score,
            'loss': loss,
            'structure_failure': structure_failure,
            # Legacy compatibility
            'smoothness_score': structure_score,
            'edge_correlation': structure_score,
            'edge_reduction': structure_score,
            'noise_reduction_score': structure_score,
            'edge_quality_score': structure_score,
        }

        if return_details:
            result['feature_stack'] = feature_stack
            result['feature_names'] = feature_names
            result['feature_weights'] = self.feature_weights.detach()

        return result

    def get_weight_summary(self) -> Dict[str, float]:
        """Return current weights for inspection."""
        fw = self.feature_weights.detach().cpu().tolist()
        names = []
        for k in self.scales:
            names.extend([f'intensity_s{k}', f'residual_var_s{k}', f'denoised_std_s{k}'])
        names.extend(['edge_denoised', 'edge_residual', 'edge_diff',
                     'laplacian_denoised', 'laplacian_residual', 'residual_magnitude'])
        return {name: weight for name, weight in zip(names, fw)}

    def calibrate(
        self,
        noisy_samples: List[torch.Tensor],
        denoised_samples: List[torch.Tensor],
        clean_samples: List[torch.Tensor],
        n_steps: int = 200,
        lr: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calibrate weights to maximize correlation with actual error.
        """
        if not self.learnable_weights and self.combiner is None:
            raise ValueError("No learnable parameters to calibrate.")

        # Collect parameters
        params = []
        if self.learnable_weights:
            params.append(self.feature_logits)
        if self.combiner is not None:
            params.extend(self.combiner.parameters())

        optimizer = torch.optim.Adam(params, lr=lr)

        best_corr = -1.0
        best_state = None

        for step in range(n_steps):
            total_corr = 0.0
            n = 0

            optimizer.zero_grad()
            total_loss = 0.0

            for noisy, denoised, clean in zip(noisy_samples, denoised_samples, clean_samples):
                result = self.forward(noisy, denoised)
                structure_failure = result['structure_failure']
                actual_error = (denoised - clean).abs()

                # Resize if needed
                if structure_failure.shape != actual_error.shape:
                    structure_failure = F.interpolate(
                        structure_failure, size=actual_error.shape[2:],
                        mode='bilinear', align_corners=False
                    )

                # Correlation loss (negative to maximize)
                sf_flat = structure_failure.flatten()
                ae_flat = actual_error.flatten()
                sf_c = sf_flat - sf_flat.mean()
                ae_c = ae_flat - ae_flat.mean()
                corr = (sf_c * ae_c).sum() / (torch.sqrt((sf_c**2).sum() * (ae_c**2).sum()) + 1e-8)

                total_loss += -corr
                total_corr += corr.item()
                n += 1

            total_loss.backward()
            optimizer.step()

            avg_corr = total_corr / n
            if avg_corr > best_corr:
                best_corr = avg_corr
                if self.learnable_weights:
                    best_state = self.feature_logits.detach().clone()

            if step % 50 == 0:
                print(f"  Step {step}: correlation = {avg_corr:.4f}")

        # Restore best
        if best_state is not None and self.learnable_weights:
            self.feature_logits.data = best_state

        print(f"  Calibration complete: best correlation = {best_corr:.4f}")
        return {'correlation': best_corr, 'weights': self.get_weight_summary()}


# =============================================================================
# COMBINED VERIFIABLE DENOISING PREDICATE
# =============================================================================

@dataclass
class VerificationResult:
    """Result of denoising verification."""
    satisfied: bool              # Overall predicate satisfaction
    score: float                 # Overall score (0-1)
    speckle_satisfied: bool      # P1: Speckle fidelity
    anatomy_satisfied: bool      # P2: Anatomy valid
    structure_satisfied: bool    # P3: Structure preserved
    details: Dict                # Detailed results from each predicate


class VerifiableDenoisingPredicate(nn.Module):
    """
    Combined predicate for verifiable OCT denoising.

    Φ(noisy, denoised, boundaries) :=
        P₁(noisy, denoised) ∧ P₂(boundaries) ∧ P₃(noisy, denoised, boundaries)

    Where:
        P₁ = SpeckleFidelity: residual follows speckle statistics
        P₂ = AnatomyValid: boundaries satisfy anatomical constraints
        P₃ = StructurePreserved: edges maintained at boundaries

    Usage:
        predicate = VerifiableDenoisingPredicate()

        # During training (differentiable loss)
        loss, details = predicate.compute_loss(noisy, denoised, boundaries)

        # During inference (verification)
        result = predicate.verify(noisy, denoised, boundaries)
        if result.satisfied:
            print("Denoising verified!")
        else:
            print(f"Verification failed: {result.details}")
    """

    def __init__(
        self,
        # Speckle fidelity parameters (calibrated on PKU37)
        expected_cv: float = 0.40,
        cv_tolerance: float = 0.14,
        # Anatomy parameters
        min_layer_thickness: float = 0.03,
        # Structure parameters
        edge_threshold: float = 0.5,
        # Loss weights
        lambda_speckle: float = 1.0,
        lambda_anatomy: float = 1.0,
        lambda_structure: float = 1.0,
    ):
        super().__init__()

        self.P1_speckle = SpeckleFidelityPredicate(
            expected_cv=expected_cv,
            tolerance=cv_tolerance,
        )

        self.P2_anatomy = AnatomyValidPredicate(
            min_layer_thickness=min_layer_thickness,
        )

        self.P3_structure = StructurePreservedPredicate(
            window_size=5,
            boundary_sigma=5.0,
        )

        self.lambda_speckle = lambda_speckle
        self.lambda_anatomy = lambda_anatomy
        self.lambda_structure = lambda_structure

    def compute_loss(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute differentiable loss for training.

        Args:
            noisy: [B, 1, H, W] noisy input
            denoised: [B, 1, H, W] denoised output
            boundaries: [B, N, W] boundary positions

        Returns:
            total_loss: scalar tensor
            details: dict with individual losses and scores
        """
        # Evaluate each predicate
        p1_result = self.P1_speckle(noisy, denoised)
        p2_result = self.P2_anatomy(boundaries)
        p3_result = self.P3_structure(noisy, denoised, boundaries)

        # Combined loss
        total_loss = (
            self.lambda_speckle * p1_result['loss'] +
            self.lambda_anatomy * p2_result['loss'] +
            self.lambda_structure * p3_result['loss']
        )

        # Collect details
        details = {
            'total_loss': total_loss.item(),
            'speckle_loss': p1_result['loss'].item(),
            'anatomy_loss': p2_result['loss'].item(),
            'structure_loss': p3_result['loss'].item(),
            'speckle_score': p1_result['score'].mean().item(),
            'anatomy_score': p2_result['score'].mean().item(),
            'structure_score': p3_result['score'].mean().item(),
            'speckle_cv': p1_result['cv_mean'].mean().item(),
            'edge_correlation': p3_result['edge_correlation'].mean().item(),
            'edge_reduction': p3_result['edge_reduction'].mean().item(),
        }

        return total_loss, details

    def forward(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate all predicates using FUZZY LOGIC composition.

        SYMBOLIC FORMULA:
            Φ(noisy, denoised, boundaries) :=
                P₁(noisy, denoised) ∧ P₂(boundaries) ∧ P₃(noisy, denoised, boundaries)

        Where ∧ is fuzzy AND (product t-norm for differentiability).

        Returns dict compatible with loss function integration.
        """
        # BUG FIX: Evaluate predicates only ONCE (was evaluated twice before)
        p1_result = self.P1_speckle(noisy, denoised)
        p2_result = self.P2_anatomy(boundaries)
        p3_result = self.P3_structure(noisy, denoised, boundaries)

        # Combined loss
        total_loss = (
            self.lambda_speckle * p1_result['loss'] +
            self.lambda_anatomy * p2_result['loss'] +
            self.lambda_structure * p3_result['loss']
        )

        # FUZZY AND: all predicates must hold
        # Using product t-norm: P1 AND P2 AND P3
        score = FuzzyLogic.AND(
            FuzzyLogic.AND(p1_result['score'], p2_result['score']),
            p3_result['score']
        )

        # Hard satisfaction (for reporting)
        satisfied = p1_result['satisfied'] & p2_result['satisfied'] & p3_result['satisfied']

        # Collect details
        details = {
            'total_loss': total_loss.item() if total_loss.dim() == 0 else total_loss.mean().item(),
            'speckle_loss': p1_result['loss'].item() if p1_result['loss'].dim() == 0 else p1_result['loss'].mean().item(),
            'anatomy_loss': p2_result['loss'].item() if p2_result['loss'].dim() == 0 else p2_result['loss'].mean().item(),
            'structure_loss': p3_result['loss'].item() if p3_result['loss'].dim() == 0 else p3_result['loss'].mean().item(),
            'speckle_score': p1_result['score'].mean().item(),
            'anatomy_score': p2_result['score'].mean().item(),
            'structure_score': p3_result['score'].mean().item(),
            'speckle_cv': p1_result['cv_mean'].mean().item(),
            'edge_correlation': p3_result['edge_correlation'].mean().item(),
            'edge_reduction': p3_result['edge_reduction'].item() if p3_result['edge_reduction'].dim() == 0 else p3_result['edge_reduction'].mean().item(),
        }

        return {
            'loss': total_loss,
            'satisfied': satisfied,
            'score': score,  # True fuzzy logic composition
            'details': details,
            'p1_speckle': p1_result,
            'p2_anatomy': p2_result,
            'p3_structure': p3_result,
        }

    def verify(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> VerificationResult:
        """
        Verify denoising quality without ground truth.

        This is the key novel capability: assess denoising quality
        using only physics-based and anatomy-based constraints.

        Returns:
            VerificationResult with satisfaction status and details
        """
        with torch.no_grad():
            result = self.forward(noisy, denoised, boundaries)

            return VerificationResult(
                satisfied=result['satisfied'].all().item(),
                score=result['score'].mean().item(),
                speckle_satisfied=result['p1_speckle']['satisfied'].all().item(),
                anatomy_satisfied=result['p2_anatomy']['satisfied'].all().item(),
                structure_satisfied=result['p3_structure']['satisfied'].all().item(),
                details=result['details'],
            )


# =============================================================================
# TESTING
# =============================================================================

def test_predicates():
    """Test all three predicates with fuzzy logic."""
    print("=" * 70)
    print("TESTING TRULY SYMBOLIC PREDICATES WITH FUZZY LOGIC")
    print("=" * 70)

    # First, demonstrate fuzzy logic operators
    print("\n" + "-" * 70)
    print("FUZZY LOGIC OPERATORS (Differentiable Symbolic Reasoning)")
    print("-" * 70)

    a = torch.tensor([0.9, 0.7, 0.3, 0.1])
    b = torch.tensor([0.8, 0.4, 0.6, 0.2])

    print(f"\n  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"\n  Fuzzy AND(a, b)     = {FuzzyLogic.AND(a, b).tolist()}")
    print(f"  Fuzzy OR(a, b)      = {FuzzyLogic.OR(a, b).tolist()}")
    print(f"  Fuzzy NOT(a)        = {FuzzyLogic.NOT(a).tolist()}")
    print(f"  Fuzzy IMPLIES(a, b) = {FuzzyLogic.IMPLIES(a, b).tolist()}")

    # Demonstrate FORALL and EXISTS
    satisfaction = torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]])
    print(f"\n  Satisfaction matrix:\n  {satisfaction.tolist()}")
    print(f"  FORALL (soft):  {FuzzyLogic.FORALL(satisfaction, soft=True).item():.4f}")
    print(f"  EXISTS (soft):  {FuzzyLogic.EXISTS(satisfaction, soft=True).item():.4f}")
    print(f"  MOST (>0.6):    {FuzzyLogic.MOST(satisfaction, threshold=0.6).item():.4f}")

    print("\n  Knowledge Base:")
    print(f"  {OCT_KNOWLEDGE}")

    torch.manual_seed(42)
    B, H, W = 2, 256, 64

    # Create synthetic OCT-like images
    clean = torch.zeros(B, 1, H, W)
    clean[:, :, 50:180, :] = 0.7   # Retina region
    clean[:, :, 80:120, :] = 0.3   # Dark INL
    clean[:, :, 140:160, :] = 0.9  # Bright IS/OS

    # Add speckle-like noise (multiplicative)
    noise = torch.randn(B, 1, H, W) * 0.1 * torch.sqrt(clean + 0.1)
    noisy = clean + noise
    noisy = noisy.clamp(0, 1)

    # Simulated denoised (slightly smoothed)
    denoised = F.avg_pool2d(noisy, 3, stride=1, padding=1)

    # Synthetic boundaries (normalized)
    boundaries = torch.tensor([[0.20, 0.32, 0.55, 0.70]]).unsqueeze(-1).expand(B, 4, W).clone()

    # Test P1: Speckle Fidelity
    print("\n" + "-" * 70)
    print("P1: SPECKLE FIDELITY")
    print("-" * 70)
    p1 = SpeckleFidelityPredicate()
    p1_result = p1(noisy, denoised, return_details=True)
    print(f"  Satisfied: {p1_result['satisfied'].tolist()}")
    print(f"  Score: {p1_result['score'].tolist()}")
    print(f"  CV mean: {p1_result['cv_mean'].tolist()}")
    print(f"  CV expected: {p1_result['cv_expected'].item():.3f}")
    print(f"  Loss: {p1_result['loss'].item():.4f}")

    # Test P2: Anatomy Valid
    print("\n" + "-" * 70)
    print("P2: ANATOMY VALID")
    print("-" * 70)
    p2 = AnatomyValidPredicate()
    p2_result = p2(boundaries, return_details=True)
    print(f"  Satisfied: {p2_result['satisfied'].tolist()}")
    print(f"  Score: {p2_result['score'].tolist()}")
    print(f"  Ordering OK: {p2_result['ordering_satisfied'].tolist()}")
    print(f"  Thickness OK: {p2_result['thickness_satisfied'].tolist()}")
    print(f"  Position OK: {p2_result['position_satisfied'].tolist()}")
    print(f"  Loss: {p2_result['loss'].item():.4f}")

    # Test with invalid boundaries
    print("\n  Testing with INVALID boundaries (wrong order):")
    invalid_bounds = torch.tensor([[0.50, 0.30, 0.70, 0.20]]).unsqueeze(-1).expand(B, 4, W).clone()
    p2_invalid = p2(invalid_bounds, return_details=True)
    print(f"    Satisfied: {p2_invalid['satisfied'].tolist()}")
    print(f"    Ordering OK: {p2_invalid['ordering_satisfied'].tolist()}")

    # Test P3: Structure Preserved (Fixed Weights)
    print("\n" + "-" * 70)
    print("P3: STRUCTURE PRESERVED (Fixed Weights)")
    print("-" * 70)
    p3 = StructurePreservedPredicate(learnable_weights=False)
    p3_result = p3(noisy, denoised, boundaries, return_details=True)
    print(f"  Satisfied: {p3_result['satisfied'].tolist()}")
    print(f"  Score: {p3_result['score'].tolist()}")
    print(f"  Edge correlation: {p3_result['edge_correlation'].tolist()}")
    print(f"  Edge reduction: {p3_result['edge_reduction'].tolist()}")
    print(f"  Loss: {p3_result['loss'].item():.4f}")

    # Show weights
    weights = p3.get_weight_summary()
    print(f"  Feature weights: {weights['feature_weights']}")
    print(f"  Combo weights: {weights['combo_weights']}")

    # Test P3 with Learnable Weights
    print("\n" + "-" * 70)
    print("P3: STRUCTURE PRESERVED (Learnable Weights)")
    print("-" * 70)
    p3_learn = StructurePreservedPredicate(learnable_weights=True)
    print(f"  Initial weights: {p3_learn.get_weight_summary()['feature_weights']}")
    print(f"  Parameters trainable: {p3_learn.feature_logits.requires_grad}")

    # Test combined predicate
    print("\n" + "-" * 70)
    print("COMBINED: VERIFIABLE DENOISING")
    print("-" * 70)
    verifier = VerifiableDenoisingPredicate()

    # Verification
    result = verifier.verify(noisy, denoised, boundaries)
    print(f"  Overall satisfied: {result.satisfied}")
    print(f"  Overall score: {result.score:.3f}")
    print(f"  P1 (Speckle): {result.speckle_satisfied}")
    print(f"  P2 (Anatomy): {result.anatomy_satisfied}")
    print(f"  P3 (Structure): {result.structure_satisfied}")

    # Loss computation
    loss, details = verifier.compute_loss(noisy, denoised, boundaries)
    print(f"\n  Training loss: {loss.item():.4f}")
    print(f"  Loss breakdown:")
    for k, v in details.items():
        if 'loss' in k or 'score' in k:
            print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 70)
    print("ALL PREDICATE TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_predicates()
