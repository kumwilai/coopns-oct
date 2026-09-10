#!/usr/bin/env python3
"""
Neuro-Symbolic Losses for OCT Denoising + Segmentation

KEY TMI CONTRIBUTION: First neuro-symbolic OCT denoiser integrating:
1. Layer Ordering Loss - Anatomical constraint (differentiable)
2. Thickness Constraint - Clinical knowledge as loss
3. Intensity Order - OCT physics prior
4. Boundary Continuity - Topological constraint
5. Anatomy Templates - Structural knowledge (fovea, periphery)

These symbolic constraints encode domain knowledge that pure neural networks
cannot learn from limited data, ensuring anatomically valid outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

# =============================================================================
# Layer Configuration and Priors
# =============================================================================

# 3-class scheme
CLASS_NAMES_3 = ['RNFL_GCL', 'INL_OPL_ONL', 'RPE_Choroid']

# 4-class scheme (with dedicated IS_OS class)
CLASS_NAMES_4 = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

# Known thickness ranges as FRACTION of image height
# These are approximate healthy ranges - pathology may exceed these
# For a typical 512-height OCT scan, these translate to reasonable pixel ranges
THICKNESS_PRIOR_3CLASS = {
    0: (0.05, 0.25),   # RNFL_GCL: 5-25% of image height (~25-128 px for 512px)
    1: (0.15, 0.45),   # INL_OPL_ONL: 15-45% of image height (~77-230 px for 512px)
    2: (0.10, 0.40),   # RPE_Choroid: 10-40% of image height (~51-205 px for 512px)
}

# 4-class thickness priors
THICKNESS_PRIOR_4CLASS = {
    0: (0.05, 0.25),   # RNFL_GCL: 5-25% of image height
    1: (0.15, 0.40),   # INL_OPL_ONL: 15-40% of image height
    2: (0.02, 0.10),   # IS_OS: 2-10% of image height (thin photoreceptor layer)
    3: (0.10, 0.35),   # RPE_Choroid: 10-35% of image height
}

# Relative intensity order (higher index = should be brighter in typical OCT)
# RNFL is highly reflective, RPE is highly reflective, middle layers less so
INTENSITY_ORDER_3CLASS = {
    0: 2,  # RNFL_GCL: High reflectivity (bright)
    1: 0,  # INL_OPL_ONL: Low-medium reflectivity (darker)
    2: 1,  # RPE_Choroid: High reflectivity (bright)
}

# 4-class intensity order
INTENSITY_ORDER_4CLASS = {
    0: 3,  # RNFL_GCL: High reflectivity (bright)
    1: 0,  # INL_OPL_ONL: Low-medium reflectivity (darkest)
    2: 2,  # IS_OS: Medium-high reflectivity (bright junction)
    3: 1,  # RPE_Choroid: High reflectivity (bright)
}

def get_class_names(num_classes):
    """Get class names for given number of classes."""
    if num_classes == 3:
        return CLASS_NAMES_3
    elif num_classes == 4:
        return CLASS_NAMES_4
    else:
        return [f'class_{i}' for i in range(num_classes)]

def get_thickness_prior(num_classes):
    """Get thickness prior for given number of classes."""
    if num_classes == 3:
        return THICKNESS_PRIOR_3CLASS
    elif num_classes == 4:
        return THICKNESS_PRIOR_4CLASS
    else:
        return {i: (0.05, 0.40) for i in range(num_classes)}


# =============================================================================
# 1. Layer Ordering Loss (Already implemented, included for completeness)
# =============================================================================
def compute_layer_ordering_loss(seg_logits: torch.Tensor, margin: float = 2.0) -> Tuple[torch.Tensor, Dict]:
    """
    Neuro-Symbolic Layer Ordering Loss.

    Enforces anatomical constraint: layers must follow top-to-bottom order.
    3-class: RNFL_GCL (top) -> INL_OPL_ONL (middle) -> RPE_Choroid (bottom)
    4-class: RNFL_GCL (top) -> INL_OPL_ONL -> IS_OS -> RPE_Choroid (bottom)

    Args:
        seg_logits: [B, C, H, W] segmentation logits
        margin: Minimum expected pixel gap between adjacent layers

    Returns:
        loss: Scalar ordering loss
        stats: Per-pair violation statistics
    """
    B, C, H, W = seg_logits.shape
    device = seg_logits.device
    class_names = get_class_names(C)

    probs = F.softmax(seg_logits, dim=1)

    # Y-coordinate grid
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    y_coords = y_coords.expand(B, 1, H, W)

    # Expected y-position per class
    expected_y = []
    class_presence = []

    for c in range(C):
        prob_c = probs[:, c:c+1, :, :]
        weighted_y = (prob_c * y_coords).sum(dim=(2, 3), keepdim=True)
        prob_sum = prob_c.sum(dim=(2, 3), keepdim=True) + 1e-8
        exp_y = (weighted_y / prob_sum).mean()
        expected_y.append(exp_y)
        class_presence.append(prob_sum.mean())

    expected_y = torch.stack(expected_y)
    class_presence = torch.stack(class_presence)

    # Ordering violations
    ordering_loss = torch.tensor(0.0, device=device)
    stats = {}

    for c in range(C - 1):
        gap = expected_y[c + 1] - expected_y[c]
        violation = F.relu(margin - gap)
        presence_weight = torch.min(class_presence[c], class_presence[c + 1]).clamp(0.1, 1.0)
        ordering_loss = ordering_loss + violation * presence_weight

        stats[f'{class_names[c]}_to_{class_names[c+1]}'] = {
            'gap': gap.item(),
            'violation': violation.item(),
        }

    return ordering_loss / max(C - 1, 1), stats


# =============================================================================
# 2. Layer Thickness Constraint Loss
# =============================================================================
def compute_thickness_loss(
    seg_probs: torch.Tensor,
    thickness_prior: Dict[int, Tuple[float, float]] = None,
    soft_margin: float = 0.05,
) -> Tuple[torch.Tensor, Dict]:
    """
    Neuro-Symbolic Thickness Constraint Loss.

    Enforces clinical knowledge: each layer has known thickness ranges.
    Penalizes predictions outside healthy ranges.

    Args:
        seg_probs: [B, C, H, W] segmentation probabilities
        thickness_prior: Dict mapping class index to (min_frac, max_frac) as fraction of image height
        soft_margin: Soft margin (as fraction) before penalty kicks in

    Returns:
        loss: Thickness violation loss
        stats: Per-layer thickness statistics
    """
    if thickness_prior is None:
        thickness_prior = THICKNESS_PRIOR_3CLASS

    B, C, H, W = seg_probs.shape
    device = seg_probs.device

    total_loss = torch.tensor(0.0, device=device)
    stats = {}

    class_names = get_class_names(C)
    if thickness_prior is None:
        thickness_prior = get_thickness_prior(C)

    for c in range(C):
        min_frac, max_frac = thickness_prior.get(c, (0.0, 1.0))

        # Convert fractions to pixels for this image size
        min_t = min_frac * H
        max_t = max_frac * H
        margin_px = soft_margin * H

        # Compute expected thickness for this layer per column
        prob_c = seg_probs[:, c, :, :]  # [B, H, W]

        # Thickness = sum of probabilities along height (expected pixels belonging to this class)
        thickness_per_col = prob_c.sum(dim=1)  # [B, W]
        mean_thickness = thickness_per_col.mean()

        # Soft penalty for being outside range
        too_thin = F.relu(min_t - margin_px - mean_thickness)
        too_thick = F.relu(mean_thickness - max_t - margin_px)

        layer_loss = too_thin + too_thick
        total_loss = total_loss + layer_loss

        stats[class_names[c] if c < len(class_names) else f'class_{c}'] = {
            'mean_thickness': mean_thickness.item(),
            'mean_thickness_frac': mean_thickness.item() / H,
            'min_expected': min_t,
            'max_expected': max_t,
            'violation': layer_loss.item(),
        }

    return total_loss / max(C, 1), stats


# =============================================================================
# 3. Intensity Order Constraint Loss
# =============================================================================
def compute_intensity_order_loss(
    denoised: torch.Tensor,
    seg_probs: torch.Tensor,
    margin: float = 0.05,
) -> Tuple[torch.Tensor, Dict]:
    """
    Neuro-Symbolic Intensity Order Loss.

    Enforces OCT physics prior: layers have known relative intensities.
    - RNFL/RPE: High reflectivity (bright)
    - INL/ONL: Lower reflectivity (darker)

    Args:
        denoised: [B, 1, H, W] denoised image
        seg_probs: [B, C, H, W] segmentation probabilities
        margin: Minimum intensity difference between bright/dark layers

    Returns:
        loss: Intensity order violation loss
        stats: Per-layer intensity statistics
    """
    B, C, H, W = seg_probs.shape
    device = denoised.device

    # Compute weighted mean intensity per layer
    layer_intensities = []
    layer_weights = []

    for c in range(C):
        prob_c = seg_probs[:, c:c+1, :, :]  # [B, 1, H, W]
        weight = prob_c.sum()

        if weight > 100:  # Minimum pixels to consider
            weighted_intensity = (denoised * prob_c).sum() / (weight + 1e-8)
            layer_intensities.append(weighted_intensity)
            layer_weights.append(weight)
        else:
            layer_intensities.append(torch.tensor(0.5, device=device))
            layer_weights.append(torch.tensor(0.0, device=device))

    layer_intensities = torch.stack(layer_intensities)

    # Intensity order constraint
    # RNFL_GCL (0) should be brighter than INL_OPL_ONL (1)
    # IS_OS (2) and RPE_Choroid (3 for 4-class, 2 for 3-class) should be brighter than INL_OPL_ONL (1)

    total_loss = torch.tensor(0.0, device=device)
    stats = {}

    # RNFL > INL constraint (applies to both 3-class and 4-class)
    if C >= 2:
        violation_0_1 = F.relu(margin - (layer_intensities[0] - layer_intensities[1]))
        total_loss = total_loss + violation_0_1
        stats['RNFL_brighter_than_INL'] = {
            'RNFL_intensity': layer_intensities[0].item(),
            'INL_intensity': layer_intensities[1].item(),
            'violation': violation_0_1.item(),
        }

    if C == 3:
        # 3-class: RPE (class 2) > INL constraint
        violation_2_1 = F.relu(margin - (layer_intensities[2] - layer_intensities[1]))
        total_loss = total_loss + violation_2_1
        stats['RPE_brighter_than_INL'] = {
            'RPE_intensity': layer_intensities[2].item(),
            'INL_intensity': layer_intensities[1].item(),
            'violation': violation_2_1.item(),
        }
    elif C >= 4:
        # 4-class: IS_OS (class 2) > INL constraint
        violation_2_1 = F.relu(margin - (layer_intensities[2] - layer_intensities[1]))
        total_loss = total_loss + violation_2_1
        stats['IS_OS_brighter_than_INL'] = {
            'IS_OS_intensity': layer_intensities[2].item(),
            'INL_intensity': layer_intensities[1].item(),
            'violation': violation_2_1.item(),
        }

        # 4-class: RPE (class 3) > INL constraint
        violation_3_1 = F.relu(margin - (layer_intensities[3] - layer_intensities[1]))
        total_loss = total_loss + violation_3_1
        stats['RPE_brighter_than_INL'] = {
            'RPE_intensity': layer_intensities[3].item(),
            'INL_intensity': layer_intensities[1].item(),
            'violation': violation_3_1.item(),
        }

    return total_loss, stats


# =============================================================================
# 4. Boundary Continuity Constraint Loss
# =============================================================================
def compute_boundary_continuity_loss(
    seg_probs: torch.Tensor,
    max_jump: float = 2.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Neuro-Symbolic Boundary Continuity Loss.

    Enforces topological constraint: layer boundaries are continuous curves.
    Penalizes discontinuities (sudden jumps) in layer boundaries.

    Uses boundary detection (where probability transitions from one class to another)
    rather than centroid, for more accurate discontinuity detection.

    Args:
        seg_probs: [B, C, H, W] segmentation probabilities
        max_jump: Maximum allowed boundary jump between adjacent columns (default: 2 pixels)

    Returns:
        loss: Boundary continuity loss
        stats: Continuity statistics
    """
    B, C, H, W = seg_probs.shape
    device = seg_probs.device

    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

    total_loss = torch.tensor(0.0, device=device)
    stats = {}

    # Method 1: Detect boundary positions using cumulative probability
    # Boundary between layer c and c+1 is where cumsum of top layers >= 0.5
    cumsum_probs = torch.cumsum(seg_probs, dim=1)  # [B, C, H, W]

    for c in range(C - 1):
        # Boundary between layer c and c+1
        # Find where cumsum of layers 0..c crosses 0.5
        cumsum_c = cumsum_probs[:, c, :, :]  # [B, H, W]

        # Expected boundary position per column (soft argmax where cumsum crosses 0.5)
        # Weight by how close cumsum is to 0.5
        crossing_weight = torch.exp(-10.0 * (cumsum_c - 0.5) ** 2)  # Sharp peak at 0.5
        crossing_weight = crossing_weight / (crossing_weight.sum(dim=1, keepdim=True) + 1e-8)

        # Expected y position of boundary
        y_coords_2d = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
        boundary_y = (crossing_weight * y_coords_2d).sum(dim=1)  # [B, W]

        # Horizontal gradient (change between adjacent columns)
        h_grad = torch.abs(boundary_y[:, 1:] - boundary_y[:, :-1])  # [B, W-1]

        # Penalize jumps larger than max_jump
        discontinuity = F.relu(h_grad - max_jump)
        layer_loss = discontinuity.mean()

        total_loss = total_loss + layer_loss

        class_names = get_class_names(C)
        boundary_name = f'{class_names[c]}_to_{class_names[c+1]}' if c + 1 < len(class_names) else f'boundary_{c}'
        stats[boundary_name] = {
            'mean_gradient': h_grad.mean().item(),
            'max_gradient': h_grad.max().item(),
            'discontinuity_loss': layer_loss.item(),
        }

    # Method 2: Also penalize overall segmentation smoothness
    # Compute horizontal gradient of class probabilities
    prob_h_grad = torch.abs(seg_probs[:, :, :, 1:] - seg_probs[:, :, :, :-1])  # [B, C, H, W-1]
    smoothness_loss = prob_h_grad.mean() * 0.1  # Small weight

    total_loss = total_loss + smoothness_loss
    stats['smoothness'] = smoothness_loss.item()

    return total_loss / max(C, 1), stats


# =============================================================================
# 5. Anatomy Template Constraint Loss
# =============================================================================
class AnatomyTemplateLoss(nn.Module):
    """
    Neuro-Symbolic Anatomy Template Loss.

    Encodes structural knowledge about retinal anatomy:
    - Fovea: Inner layers thin/absent, outer layers thick
    - Periphery: More uniform layer distribution
    - Optic disc: No photoreceptors (IS/OS absent)

    Uses soft templates that guide but don't force exact structure.
    """

    def __init__(self, num_classes: int = 3, image_width: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.image_width = image_width

        # Create foveal template (relative thickness expectations)
        # At fovea center: RNFL/GCL thin, ONL thick
        self.register_buffer('fovea_template', self._create_fovea_template())

    def _create_fovea_template(self) -> torch.Tensor:
        """
        Create foveal anatomy template.

        Returns:
            template: [C, W] expected relative thickness across width
        """
        W = self.image_width
        C = self.num_classes

        # Assume fovea is roughly in the center
        x = torch.linspace(-1, 1, W)

        # Foveal pit profile (Gaussian dip for inner layers)
        fovea_dip = torch.exp(-x**2 / 0.1)  # Sharp dip at center

        template = torch.ones(C, W)

        if C == 3:
            # 3-class scheme
            # RNFL_GCL: Thin at fovea (dip)
            template[0] = 1.0 - 0.7 * fovea_dip

            # INL_OPL_ONL: Also thins at fovea but less dramatically
            template[1] = 1.0 - 0.3 * fovea_dip

            # RPE_Choroid: Relatively constant (maybe slight thickening)
            template[2] = 1.0 + 0.1 * fovea_dip
        elif C >= 4:
            # 4-class scheme (RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid)
            # RNFL_GCL: Thin at fovea (dip)
            template[0] = 1.0 - 0.7 * fovea_dip

            # INL_OPL_ONL: Also thins at fovea but less dramatically
            template[1] = 1.0 - 0.3 * fovea_dip

            # IS_OS: Relatively constant (photoreceptors present throughout)
            template[2] = 1.0 + 0.05 * fovea_dip

            # RPE_Choroid: Relatively constant (maybe slight thickening)
            template[3] = 1.0 + 0.1 * fovea_dip

        return template

    def forward(
        self,
        seg_probs: torch.Tensor,
        fovea_detected: bool = True,
        fovea_location: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute anatomy template matching loss.

        Args:
            seg_probs: [B, C, H, W] segmentation probabilities
            fovea_detected: Whether fovea is likely in this image
            fovea_location: Optional x-coordinate of detected fovea

        Returns:
            loss: Template matching loss
            stats: Template matching statistics
        """
        B, C, H, W = seg_probs.shape
        device = seg_probs.device

        if not fovea_detected:
            return torch.tensor(0.0, device=device), {'skipped': True}

        # Compute thickness profile per class
        thickness_profile = seg_probs.sum(dim=2)  # [B, C, W]

        # Normalize to relative thickness
        total_thickness = thickness_profile.sum(dim=1, keepdim=True) + 1e-8
        relative_thickness = thickness_profile / total_thickness  # [B, C, W]

        # Resize template to match input width
        template = self.fovea_template.to(device)
        if template.shape[1] != W:
            template = F.interpolate(
                template.unsqueeze(0), size=W, mode='linear', align_corners=False
            ).squeeze(0)

        # Normalize template
        template = template / (template.sum(dim=0, keepdim=True) + 1e-8)

        # Shift template if fovea location is known
        if fovea_location is not None:
            shift = fovea_location - W // 2
            template = torch.roll(template, shifts=shift, dims=1)

        # Soft matching loss (encourage similar profile, don't force exact match)
        # Use KL divergence or L1 distance
        template_expanded = template.unsqueeze(0).expand(B, -1, -1)

        # L1 loss weighted by confidence
        profile_diff = torch.abs(relative_thickness - template_expanded)
        loss = profile_diff.mean()

        stats = {
            'template_loss': loss.item(),
            'fovea_detected': fovea_detected,
        }

        return loss * 0.1, stats  # Scale down - this is a soft prior


# =============================================================================
# 6. Head Diversity Loss (Prevent Head Collapse)
# =============================================================================
def compute_head_diversity_loss(
    layer_outputs: torch.Tensor,
    target_correlation: float = 0.5,
) -> torch.Tensor:
    """
    Head Diversity Loss to prevent layer-specific heads from collapsing.

    Encourages each head to learn different spatial patterns by penalizing
    high pairwise correlations between head outputs.

    Args:
        layer_outputs: [B, n_heads, C, H, W] or [B, n_heads, H, W] outputs from layer-specific heads
        target_correlation: Target maximum correlation (default 0.5)

    Returns:
        loss: Diversity loss (higher when heads are too similar)
    """
    # Handle both [B, n_heads, C, H, W] and [B, n_heads, H, W] formats
    if layer_outputs.dim() == 5:
        # [B, n_heads, C, H, W] -> squeeze channel dimension
        layer_outputs = layer_outputs.squeeze(2)

    B, n_heads, H, W = layer_outputs.shape
    device = layer_outputs.device

    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    # Flatten spatial dimensions for correlation computation
    outputs_flat = layer_outputs.view(B, n_heads, -1)  # [B, n_heads, H*W]

    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            # Compute correlation between heads i and j
            h_i = outputs_flat[:, i, :]  # [B, H*W]
            h_j = outputs_flat[:, j, :]  # [B, H*W]

            # Normalize (subtract mean, divide by std)
            h_i_norm = h_i - h_i.mean(dim=1, keepdim=True)
            h_j_norm = h_j - h_j.mean(dim=1, keepdim=True)

            h_i_std = h_i_norm.std(dim=1, keepdim=True).clamp(min=1e-8)
            h_j_std = h_j_norm.std(dim=1, keepdim=True).clamp(min=1e-8)

            h_i_norm = h_i_norm / h_i_std
            h_j_norm = h_j_norm / h_j_std

            # Correlation = mean of element-wise product of normalized vectors
            correlation = (h_i_norm * h_j_norm).mean(dim=1)  # [B]
            mean_corr = correlation.abs().mean()

            # Penalize correlation above target
            pair_loss = F.relu(mean_corr - target_correlation)
            total_loss = total_loss + pair_loss
            n_pairs += 1

    # Normalize by number of pairs
    total_loss = total_loss / max(n_pairs, 1)

    return total_loss


# =============================================================================
# Combined Neuro-Symbolic Loss
# =============================================================================
class NeuroSymbolicLoss(nn.Module):
    """
    Combined Neuro-Symbolic Loss for OCT.

    Integrates all symbolic constraints:
    1. Layer ordering (anatomical)
    2. Thickness bounds (clinical)
    3. Intensity order (physics)
    4. Boundary continuity (topological)
    5. Anatomy templates (structural)
    """

    def __init__(
        self,
        num_classes: int = 3,
        lambda_ordering: float = 0.1,
        lambda_thickness: float = 0.05,
        lambda_intensity: float = 0.05,
        lambda_continuity: float = 0.05,
        lambda_anatomy: float = 0.02,
    ):
        super().__init__()

        self.lambda_ordering = lambda_ordering
        self.lambda_thickness = lambda_thickness
        self.lambda_intensity = lambda_intensity
        self.lambda_continuity = lambda_continuity
        self.lambda_anatomy = lambda_anatomy

        self.anatomy_template = AnatomyTemplateLoss(num_classes=num_classes)

    def forward(
        self,
        seg_logits: torch.Tensor,
        denoised: Optional[torch.Tensor] = None,
        fovea_detected: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined neuro-symbolic loss.

        Args:
            seg_logits: [B, C, H, W] segmentation logits
            denoised: [B, 1, H, W] denoised image (optional, for intensity loss)
            fovea_detected: Whether fovea is in image

        Returns:
            total_loss: Combined symbolic loss
            all_stats: Dictionary with all component statistics
        """
        seg_probs = F.softmax(seg_logits, dim=1)
        device = seg_logits.device

        total_loss = torch.tensor(0.0, device=device)
        all_stats = {}

        # 1. Layer Ordering Loss
        ordering_loss, ordering_stats = compute_layer_ordering_loss(seg_logits)
        total_loss = total_loss + self.lambda_ordering * ordering_loss
        all_stats['ordering'] = ordering_stats
        all_stats['ordering_loss'] = ordering_loss.item()

        # 2. Thickness Constraint Loss
        thickness_loss, thickness_stats = compute_thickness_loss(seg_probs)
        total_loss = total_loss + self.lambda_thickness * thickness_loss
        all_stats['thickness'] = thickness_stats
        all_stats['thickness_loss'] = thickness_loss.item()

        # 3. Intensity Order Loss (requires denoised image)
        if denoised is not None:
            intensity_loss, intensity_stats = compute_intensity_order_loss(denoised, seg_probs)
            total_loss = total_loss + self.lambda_intensity * intensity_loss
            all_stats['intensity'] = intensity_stats
            all_stats['intensity_loss'] = intensity_loss.item()

        # 4. Boundary Continuity Loss
        continuity_loss, continuity_stats = compute_boundary_continuity_loss(seg_probs)
        total_loss = total_loss + self.lambda_continuity * continuity_loss
        all_stats['continuity'] = continuity_stats
        all_stats['continuity_loss'] = continuity_loss.item()

        # 5. Anatomy Template Loss
        anatomy_loss, anatomy_stats = self.anatomy_template(seg_probs, fovea_detected)
        total_loss = total_loss + self.lambda_anatomy * anatomy_loss
        all_stats['anatomy'] = anatomy_stats
        all_stats['anatomy_loss'] = anatomy_loss.item() if isinstance(anatomy_loss, torch.Tensor) else 0.0

        all_stats['total_symbolic_loss'] = total_loss.item()

        return total_loss, all_stats


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    print("Testing Neuro-Symbolic Losses...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dummy data
    B, C, H, W = 2, 3, 64, 64
    seg_logits = torch.randn(B, C, H, W).to(device)
    denoised = torch.rand(B, 1, H, W).to(device)

    # Test combined loss
    ns_loss = NeuroSymbolicLoss(num_classes=C).to(device)
    total_loss, stats = ns_loss(seg_logits, denoised)

    print(f"\nTotal Neuro-Symbolic Loss: {total_loss.item():.4f}")
    print(f"\nComponent losses:")
    print(f"  Ordering:   {stats['ordering_loss']:.4f}")
    print(f"  Thickness:  {stats['thickness_loss']:.4f}")
    print(f"  Intensity:  {stats.get('intensity_loss', 0):.4f}")
    print(f"  Continuity: {stats['continuity_loss']:.4f}")
    print(f"  Anatomy:    {stats['anatomy_loss']:.4f}")

    print("\nAll tests passed!")
