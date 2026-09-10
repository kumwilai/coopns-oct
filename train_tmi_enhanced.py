#!/usr/bin/env python3
"""
TMI Enhanced Training: Layer-Specific Denoising with Clinical Losses

This script implements Phase 1A-3 of the TMI training pipeline, adding:
1. Layer-Specific Denoising Heads - Dedicated processing per anatomical layer
2. Segmentation-Guided Attention - Focus on boundaries and clinical regions
3. Clinical Importance Weighted Losses - Prioritize RNFL/IS-OS
4. Boundary Sharpness Loss - Preserve sharp layer transitions

Based on train_phase1a2_joint_boundary.py with TMI enhancements.
"""

import os
import sys
import argparse
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# Import from existing codebase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.nafnet import NAFNetFullFiLM, NAFNet
from nsnd.models.layer_specific_heads import EnhancedLayerSpecificDenoiser
from nsnd.models.seg_guided_attention import SegmentationGuidedAttention
from nsnd.losses.clinical_weighted import (
    TMIJointLoss,
    ClinicalWeightedL1Loss,
    LayerSSIMLoss,
    BoundarySharpnessLoss,
    compute_psnr,
    compute_ssim,
    # TMI v3.3: Physics-Informed Losses
    RayleighLikelihoodLoss,
    LayerIntensityConsistencyLoss,
    PhysicsInformedDenoisingLoss,
    InterferometricConsistencyLoss,
)
from nsnd.losses.neuro_symbolic import NeuroSymbolicLoss, compute_head_diversity_loss

# TMI v2: Columnar attention and boundary regression
from nsnd.models.columnar_attention import ColumnarEncoder
from nsnd.models.boundary_regression import (
    BoundaryRegressionHead,
    BoundaryRefiner,
    BoundaryLoss,
    extract_boundaries_from_segmentation,
    BOUNDARY_NAMES_4CLASS,
)

# TMI v3: Differentiable Shortest Path (DSP) for boundary detection
# Novel approach: treats boundaries as shortest paths through cost volumes
# Memory efficient: O(H×W) instead of O(H×W×C²), CPU-friendly
from nsnd.models.differentiable_shortest_path import (
    DifferentiableShortestPath,
    DSPBoundaryDetector,
    DSPBoundaryLoss,
    BoundaryCostDecoder,
    boundaries_to_segmentation as dsp_boundaries_to_segmentation,
    extract_boundaries_from_mask as dsp_extract_boundaries,
    # TMI v3.3: Physics-Informed Fresnel Gradient Matching
    FresnelGradientCost,
    FresnelAwareBoundaryCostDecoder,
    PhysicsAwareDSPBoundaryDetector,
    REFRACTIVE_INDICES,
)

# TMI v3.1: Adapter-based Continual Learning for DSP
# Enables domain-specific adaptation without forgetting previous domains
# Uses bottleneck adapters (~5-10% params) while keeping backbone frozen
from nsnd.models.dsp_adapters import (
    DSPAdapter,
    BoundaryAdapter,
    AdapterBank,
    AdaptiveDSPBoundaryDetector,
    ContinualDSPTrainer,
)

# Try to import calibrated noise
try:
    from calibrated_oct_noise import add_calibrated_oct_noise
    HAS_CALIBRATED_NOISE = True
except ImportError:
    HAS_CALIBRATED_NOISE = False

# Configuration
NUM_CLASSES = 4  # 4-class: RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid (clinical regions)
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
NUM_BOUNDARIES = 4  # 4 boundaries: ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE
BOUNDARY_NAMES = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']

# Import post-processing functions from enhanced segmenter
from train_seg_4class_improved import (
    smooth_boundary_curve,
    detect_boundary_outliers,
    interpolate_outliers,
    postprocess_boundaries,
    EnhancedOCTSegDataset,
)


# =============================================================================
# Memory Monitoring Utilities
# =============================================================================
import psutil

def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def check_memory_safe(limit_mb=3500):
    """Check if memory usage is below the limit."""
    current_mb = get_memory_usage_mb()
    return current_mb < limit_mb, current_mb

def aggressive_memory_cleanup():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Learning without Forgetting (LwF) - Knowledge Distillation Loss
# =============================================================================
class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for Learning without Forgetting (LwF).

    Prevents catastrophic forgetting during joint training by using a
    pre-trained teacher model to provide soft targets. The student model
    learns to match the teacher's output distribution, preserving previously
    learned segmentation knowledge.

    Key insight: When denoising improves, segmentation degrades (catastrophic forgetting).
    LwF addresses this by adding a KL divergence loss between student and teacher outputs.

    Reference: Li & Hoiem, "Learning without Forgetting", ECCV 2016
    """

    def __init__(self, temperature=4.0, alpha=0.5):
        """
        Args:
            temperature: Softmax temperature for soft targets (higher = softer)
            alpha: Weighting between hard labels (1-alpha) and soft labels (alpha)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, hard_labels=None):
        """
        Compute KD loss between student and teacher outputs.

        Args:
            student_logits: [B, C, H, W] student segmentation logits
            teacher_logits: [B, C, H, W] teacher segmentation logits (frozen)
            hard_labels: [B, H, W] optional ground truth labels for hybrid loss

        Returns:
            kd_loss: Scalar knowledge distillation loss
            stats: Dictionary with loss components
        """
        T = self.temperature
        B, C, H, W = student_logits.shape

        # Soft targets from teacher (temperature-scaled softmax)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # Student log-softmax (temperature-scaled)
        student_log_soft = F.log_softmax(student_logits / T, dim=1)

        # KL divergence: KL(teacher || student)
        # NOTE: Use 'none' reduction and manually average over all dimensions
        # to get properly normalized loss (batchmean only divides by batch size)
        kd_loss_raw = F.kl_div(
            student_log_soft,
            teacher_soft.detach(),  # Teacher is frozen
            reduction='none'
        )  # [B, C, H, W]

        # Average over all dimensions (B, C, H, W) and scale by T^2
        kd_loss = kd_loss_raw.mean() * (T * T)

        stats = {
            'kd_loss': kd_loss.item(),
            'temperature': T,
        }

        # Optional: Hybrid loss with hard labels
        if hard_labels is not None:
            ce_loss = F.cross_entropy(student_logits, hard_labels)
            total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
            stats['ce_loss'] = ce_loss.item()
            stats['total_loss'] = total_loss.item()
            return total_loss, stats

        return kd_loss, stats


def load_teacher_segmenter(teacher_ckpt_path, device, num_classes=4):
    """
    Load pre-trained segmenter as frozen teacher for LwF.

    Args:
        teacher_ckpt_path: Path to teacher checkpoint
        device: Device to load model to
        num_classes: Number of segmentation classes

    Returns:
        teacher_segmenter: Frozen teacher model in eval mode
    """
    # Load checkpoint to inspect input channels
    state = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)

    if 'model_state_dict' in state:
        state = state['model_state_dict']

    # Extract only segmenter weights (they have 'segmenter.' prefix in full model checkpoint)
    segmenter_state = {}
    for k, v in state.items():
        if k.startswith('segmenter.'):
            new_key = k[len('segmenter.'):]  # Remove 'segmenter.' prefix
            segmenter_state[new_key] = v
        elif not any(k.startswith(p) for p in ['nafnet.', 'layer_heads.', 'seg_attention.', 'feature_proj.']):
            # Also accept direct segmenter weights (without prefix)
            segmenter_state[k] = v

    # If no segmenter-prefixed weights, try loading directly
    if len(segmenter_state) == 0:
        segmenter_state = state

    # Detect input channels from checkpoint (enc1.0.weight has shape [out_ch, in_ch, kH, kW])
    enc1_weight_key = 'enc1.0.weight'
    if enc1_weight_key in segmenter_state:
        teacher_in_channels = segmenter_state[enc1_weight_key].shape[1]
        print(f"  Teacher checkpoint uses {teacher_in_channels} input channel(s)")
    else:
        teacher_in_channels = 1  # Default to 1 for older checkpoints
        print(f"  Could not detect input channels, assuming {teacher_in_channels}")

    # Create teacher segmenter with correct input channels
    teacher = HybridBoundarySegmenterV2(
        in_channels=teacher_in_channels,
        num_classes=num_classes,
        base_filters=32,
    ).to(device)

    teacher.load_state_dict(segmenter_state, strict=False)

    # Freeze teacher and set to eval mode
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Store input channels for inference
    teacher.expected_in_channels = teacher_in_channels

    # Verify teacher is frozen
    n_params = sum(p.numel() for p in teacher.parameters())
    n_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"  Teacher segmenter loaded: {n_params:,} params, {n_trainable} trainable (should be 0)")

    return teacher


# =============================================================================
# Lightweight Columnar Encoder (for memory efficiency and faster convergence)
# =============================================================================
class LightweightColumnarEncoder(nn.Module):
    """
    Memory-efficient columnar encoder using depthwise separable convolutions.

    Replaces full attention with efficient convolutions while still capturing:
    - Intra-column (depth) relationships via vertical depthwise convs
    - Inter-column (spatial) relationships via horizontal depthwise convs
    """

    def __init__(self, in_channels=64, dim=64):
        super().__init__()
        self.dim = dim

        # Project to columnar dimension
        self.input_proj = nn.Conv2d(in_channels, dim, 1)

        # Depth-wise (vertical) processing - captures layer relationships
        self.depth_conv = nn.Sequential(
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Cross-column (horizontal) processing - ensures spatial continuity
        self.cross_conv = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Mix channels
        self.mix = nn.Conv2d(dim, dim, 1)
        self.output_proj = nn.Conv2d(dim, in_channels, 1)

        # Initialize output projection to small values for stable residual
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] features from backbone
        Returns:
            out: [B, C, H, W] enhanced features
            col_features: [B, W, H, dim] columnar features for boundary regression
        """
        B, C, H, W = x.shape

        x_proj = self.input_proj(x)
        x_depth = self.depth_conv(x_proj)
        x_cross = self.cross_conv(x_depth)
        x_mix = self.mix(x_cross)
        out = self.output_proj(x_mix) + x  # Residual connection

        # Columnar features for boundary regression [B, W, H, dim]
        col_features = x_proj.permute(0, 3, 2, 1)  # [B, W, H, C]

        return out, col_features


# =============================================================================
# Neuro-Symbolic Ordering Loss (Key TMI Contribution)
# =============================================================================
def compute_symbolic_ordering_loss(seg_logits, margin=2.0):
    """
    Neuro-Symbolic Ordering Loss: Penalizes anatomical layer ordering violations.

    KEY TMI CONTRIBUTION: Enforces hard anatomical constraints in a differentiable way.

    In OCT images, retinal layers MUST follow a strict top-to-bottom ordering:
        Layer 0 (RNFL_GCL) - topmost
        Layer 1 (INL_OPL_ONL) - middle-upper
        Layer 2 (IS_OS) - middle-lower (photoreceptor junction)
        Layer 3 (RPE_Choroid) - bottommost

    The loss computes the expected y-position for each layer class based on
    segmentation probabilities, then penalizes cases where a lower layer has a
    smaller y-position than an upper layer (ordering violation).

    Args:
        seg_logits: [B, C, H, W] - segmentation logits (3 classes)
        margin: Minimum expected pixel distance between adjacent layers

    Returns:
        ordering_loss: Scalar loss (0 if all layers correctly ordered, >0 if violations)
        violation_stats: Dict with per-pair violation statistics
    """
    B, C, H, W = seg_logits.shape
    device = seg_logits.device

    # Convert logits to probabilities
    probs = F.softmax(seg_logits, dim=1)  # [B, C, H, W]

    # Create y-coordinate grid (row indices)
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    y_coords = y_coords.expand(B, 1, H, W)  # [B, 1, H, W]

    # Compute expected y-position for each class
    expected_y = []
    class_presence = []

    for c in range(C):
        prob_c = probs[:, c:c+1, :, :]  # [B, 1, H, W]

        # Weighted y-position
        weighted_y = (prob_c * y_coords).sum(dim=2, keepdim=True)  # [B, 1, 1, W]
        prob_sum = prob_c.sum(dim=2, keepdim=True) + 1e-8  # [B, 1, 1, W]

        # Expected y for this class
        exp_y_c = weighted_y / prob_sum  # [B, 1, 1, W]
        exp_y_c_mean = exp_y_c.mean()
        expected_y.append(exp_y_c_mean)
        class_presence.append(prob_sum.mean())

    expected_y = torch.stack(expected_y)  # [C]
    class_presence = torch.stack(class_presence)  # [C]

    # Compute ordering violations
    ordering_loss = torch.tensor(0.0, device=device)
    violation_stats = {}

    for c in range(C - 1):
        # Expected gap: E[y|c+1] - E[y|c] should be >= margin
        gap = expected_y[c + 1] - expected_y[c]

        # Violation if gap < margin
        violation = F.relu(margin - gap)

        # Weight by presence of both classes
        presence_weight = torch.min(class_presence[c], class_presence[c + 1])
        presence_weight = torch.clamp(presence_weight, 0.1, 1.0)

        weighted_violation = violation * presence_weight
        ordering_loss = ordering_loss + weighted_violation

        # Stats for monitoring
        with torch.no_grad():
            violation_stats[f'{CLASS_NAMES[c]}_{CLASS_NAMES[c+1]}'] = {
                'gap': gap.item(),
                'violation': violation.item(),
            }

    # Normalize by number of pairs
    ordering_loss = ordering_loss / (C - 1)

    # Smoothness term: penalize large jumps
    smoothness_loss = torch.tensor(0.0, device=device)
    for c in range(C - 1):
        gap = expected_y[c + 1] - expected_y[c]
        max_reasonable_gap = H / C * 2
        excess_gap = F.relu(gap - max_reasonable_gap)
        smoothness_loss = smoothness_loss + excess_gap * 0.01

    total_loss = ordering_loss + smoothness_loss

    return total_loss, violation_stats


def compute_layer_intensity_consistency(denoised, seg_probs):
    """
    TMI v3.3: Layer Intensity Consistency Loss.

    Physics-based constraint ensuring denoised layers have physically plausible
    relative intensities based on OCT imaging physics:
    - RPE: Highest reflectivity (melanin granules, ~1.4 relative)
    - RNFL: High reflectivity (nerve fibers, ~1.2 relative)
    - IS_OS: Medium-high (photoreceptor junction, ~1.1 relative)
    - INL: Lowest reflectivity (cell bodies, ~0.7 relative)

    Ordering constraint: RPE > RNFL > IS_OS > INL

    Args:
        denoised: [B, 1, H, W] - Denoised OCT image
        seg_probs: [B, 4, H, W] - Segmentation probabilities (softmax output)

    Returns:
        loss: Scalar loss penalizing intensity ordering violations
    """
    B, C, H, W = seg_probs.shape
    device = seg_probs.device

    # Expected relative intensities from OCT physics
    # Index: 0=RNFL_GCL, 1=INL_OPL_ONL, 2=IS_OS, 3=RPE_Choroid
    expected_intensities = torch.tensor([1.2, 0.7, 1.1, 1.4], device=device)

    # Compute mean intensity per layer (weighted by segmentation probability)
    layer_intensities = []
    layer_presence = []

    for c in range(C):
        prob_c = seg_probs[:, c:c+1, :, :]  # [B, 1, H, W]
        prob_sum = prob_c.sum() + 1e-8

        # Weighted mean intensity for this layer
        weighted_intensity = (denoised * prob_c).sum() / prob_sum
        layer_intensities.append(weighted_intensity)
        layer_presence.append(prob_sum / (B * H * W))

    layer_intensities = torch.stack(layer_intensities)  # [4]
    layer_presence = torch.stack(layer_presence)  # [4]

    # Ordering violations: RPE(3) > RNFL(0) > IS_OS(2) > INL(1)
    # Pairs to check: (3,0), (0,2), (2,1)
    ordering_pairs = [(3, 0), (0, 2), (2, 1)]

    ordering_loss = torch.tensor(0.0, device=device)
    for high_idx, low_idx in ordering_pairs:
        # High intensity layer should have greater intensity than low intensity layer
        gap = layer_intensities[high_idx] - layer_intensities[low_idx]
        violation = F.relu(-gap + 0.05)  # 0.05 margin

        # Weight by presence of both layers
        presence_weight = torch.min(layer_presence[high_idx], layer_presence[low_idx])
        presence_weight = torch.clamp(presence_weight, 0.1, 1.0)

        ordering_loss = ordering_loss + violation * presence_weight

    # Normalize by number of pairs
    ordering_loss = ordering_loss / len(ordering_pairs)

    return ordering_loss


# =====================================================================
# TMI v4.1: Adaptive Strength Map Loss Functions
# CRITICAL: Without these, the strength predictor has NO gradient signal!
# =====================================================================

def compute_oracle_strength_loss(
    strength_map: torch.Tensor,
    denoised_base: torch.Tensor,
    head_output: torch.Tensor,
    clean: torch.Tensor,
    min_strength: float = 0.3,
    max_strength: float = 0.8,
    scale: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Oracle-guided loss for adaptive strength predictor.

    Key Insight: The strength_map should be HIGH where heads improve over NAFNet base,
    and LOW where heads make things worse.

    Oracle target: Compare |head - clean| vs |base - clean| per pixel
    - If head is better: target_strength should be high
    - If base is better: target_strength should be low

    Args:
        strength_map: [B, 1, H, W] predicted strength from AdaptiveStrengthPredictor
        denoised_base: [B, 1, H, W] NAFNet base output
        head_output: [B, 1, H, W] combined layer-specific head output
        clean: [B, 1, H, W] ground truth clean image
        min_strength: Minimum target strength to prevent collapse to 0
        max_strength: Maximum target strength to prevent collapse to 1
        scale: Sensitivity to improvement (higher = sharper transition)

    Returns:
        loss: Scalar loss
        stats: Dictionary with debugging info
    """
    # Compute per-pixel errors
    base_error = torch.abs(denoised_base - clean)  # [B, 1, H, W]
    head_error = torch.abs(head_output - clean)    # [B, 1, H, W]

    # Oracle: how much better is head compared to base?
    # improvement > 0 means head is better, improvement < 0 means base is better
    improvement = base_error - head_error  # [B, 1, H, W]

    # Convert improvement to target strength using sigmoid
    # Scale so that improvement of 0.1 (10% of dynamic range) maps to ~0.73
    raw_target = torch.sigmoid(scale * improvement)  # [B, 1, H, W]

    # Clamp to [min_strength, max_strength] to prevent trivial solutions
    target_strength = min_strength + (max_strength - min_strength) * raw_target

    # L2 loss between predicted and oracle target
    # This is sufficient - the oracle target already encodes where heads help
    total_loss = F.mse_loss(strength_map, target_strength.detach())

    # Stats for debugging
    with torch.no_grad():
        improvement_positive = (improvement > 0.01).float()  # Where head helps
        stats = {
            'strength_loss': total_loss.item(),
            'oracle_target_mean': target_strength.mean().item(),
            'pred_strength_mean': strength_map.mean().item(),
            'improvement_positive_ratio': improvement_positive.mean().item(),
            'head_better_pixels': (improvement > 0).float().mean().item(),
        }

    return total_loss, stats


def compute_strength_regularization(
    strength_map: torch.Tensor,
    lambda_smooth: float = 0.1,
    lambda_entropy: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Regularization for strength map to prevent degenerate solutions.

    Components:
    1. Spatial smoothness: TV loss to encourage spatially coherent strength
    2. Entropy regularization: Prevent collapse to all 0 or all 1

    Args:
        strength_map: [B, 1, H, W] predicted strength
        lambda_smooth: Weight for smoothness loss
        lambda_entropy: Weight for entropy regularization

    Returns:
        loss: Scalar regularization loss
        stats: Dictionary with debugging info
    """
    # 1. Total Variation (TV) smoothness loss
    dx = strength_map[:, :, :, 1:] - strength_map[:, :, :, :-1]
    dy = strength_map[:, :, 1:, :] - strength_map[:, :, :-1, :]
    tv_loss = (dx.abs().mean() + dy.abs().mean()) * lambda_smooth

    # 2. Entropy regularization to prevent trivial solutions
    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    # Maximum at p=0.5, minimum at p=0 or p=1
    eps = 1e-6
    s = strength_map.clamp(eps, 1-eps)
    entropy = -s * torch.log(s) - (1-s) * torch.log(1-s)
    # We want to MAXIMIZE entropy (encourage diversity), so minimize negative entropy
    entropy_loss = -entropy.mean() * lambda_entropy

    total_loss = tv_loss + entropy_loss

    with torch.no_grad():
        stats = {
            'strength_tv_loss': tv_loss.item(),
            'strength_entropy_loss': entropy_loss.item(),
            'strength_variance': strength_map.var().item(),
        }

    return total_loss, stats


# =====================================================================
# TMI v4.1: Curriculum Learning for Strength Map
# =====================================================================

class StrengthCurriculumScheduler:
    """
    Curriculum learning for adaptive strength map.

    Phases:
    1. Warmup (epochs 0-2): Start with high fixed strength (0.5) to train heads
    2. Learn (epochs 3+): Enable strength learning with oracle loss
    """

    def __init__(
        self,
        warmup_epochs: int = 2,
        initial_strength: float = 0.5,
    ):
        self.warmup_epochs = warmup_epochs
        self.initial_strength = initial_strength

    def get_config(self, epoch: int) -> dict:
        """Get strength configuration for current epoch."""
        if epoch < self.warmup_epochs:
            # Warmup: use fixed high strength, don't train predictor
            return {
                'use_fixed_strength': True,
                'fixed_strength': self.initial_strength,
                'lambda_strength_map': 0.0,  # No oracle loss
                'freeze_strength_predictor': True,
            }
        else:
            # Learning phase: enable oracle loss, train predictor
            return {
                'use_fixed_strength': False,
                'fixed_strength': None,
                'lambda_strength_map': 1.0,
                'freeze_strength_predictor': False,
            }

    def apply_to_model(self, model, epoch: int):
        """Apply curriculum settings to model."""
        config = self.get_config(epoch)

        if hasattr(model, 'strength_predictor') and model.strength_predictor is not None:
            for param in model.strength_predictor.parameters():
                param.requires_grad = not config['freeze_strength_predictor']

        return config


NOISE_PROFILES = ['duke17', 'duke28', 'pku37']


def remap_mask_to_4class(mask, is_os_thickness=30):
    """Remap raw mask to 4-class (clinically relevant regions).

    Data format: 0=background, 1=RNFL, 2=GCL, 3=INL+OPL+ONL, 4=IS_OS+RPE+Choroid
    4-class (clinical): RNFL_GCL(0), INL_OPL_ONL(1), IS_OS(2), RPE_Choroid(3)

    IMPORTANT: Duke dataset labels Class 4 as everything from IS_OS to bottom.
    We split Class 4 into:
    - IS_OS (class 2): Top ~30 pixels of Class 4 region (photoreceptor junction)
    - RPE_Choroid (class 3): Everything below IS_OS

    This mapping ensures proper 4-class segmentation:
    - RNFL_GCL: Glaucoma diagnosis (nerve fiber + ganglion cell layers)
    - INL_OPL_ONL: Inner/outer nuclear layers
    - IS_OS: Photoreceptor junction (critical for retinal diseases, ~30px thick)
    - RPE_Choroid: Retinal pigment epithelium + choroid (AMD diagnosis)
    """
    H, W = mask.shape
    new_mask = np.zeros_like(mask)

    # Direct mappings for classes 0-3
    new_mask[mask == 0] = 0  # Background -> RNFL_GCL region (top)
    new_mask[mask == 1] = 0  # RNFL -> RNFL_GCL
    new_mask[mask == 2] = 0  # GCL -> RNFL_GCL
    new_mask[mask == 3] = 1  # INL/OPL/ONL -> INL_OPL_ONL

    # Handle Class 4: Split into IS_OS (top) and RPE_Choroid (bottom)
    # For each column, find where Class 4 starts and split it
    for col in range(W):
        col_mask = mask[:, col]
        class4_positions = np.where(col_mask == 4)[0]

        if len(class4_positions) == 0:
            continue

        # Top of Class 4 region is the IS_OS junction
        is_os_start = class4_positions[0]
        is_os_end = min(is_os_start + is_os_thickness, H)

        # IS_OS: first 'is_os_thickness' pixels of Class 4
        new_mask[is_os_start:is_os_end, col] = 2  # IS_OS

        # RPE_Choroid: everything below IS_OS
        if is_os_end < H:
            # Only mark as RPE_Choroid if still within Class 4 region
            for row in range(is_os_end, H):
                if col_mask[row] == 4 or col_mask[row] == 0:  # Class 4 or background below retina
                    new_mask[row, col] = 3  # RPE_Choroid

    # Handle explicit RPE/Choroid labels if present (mask >= 5)
    new_mask[mask >= 5] = 3  # RPE/Choroid -> RPE_Choroid

    return new_mask


# Keep for backward compatibility
def remap_mask_to_3class(mask):
    """Remap 5-class mask to 3-class (deprecated - use remap_mask_to_4class)."""
    new_mask = np.zeros_like(mask)
    new_mask[mask == 0] = 0  # RNFL_GCL
    new_mask[mask == 1] = 1  # INL
    new_mask[mask == 2] = 1  # ONL -> INL_OPL_ONL
    new_mask[mask == 3] = 1  # IS_OS -> assign to INL_OPL_ONL
    new_mask[mask == 4] = 2  # RPE_Choroid
    return new_mask


def extract_is_os_boundary(mask_5class, is_os_thickness=30):
    """Extract IS_OS region (top portion of raw mask value 4) for training BCE loss.

    Returns only the IS_OS junction region (top ~30 pixels of Class 4), NOT the
    entire Class 4 region which includes RPE and Choroid.

    Note: mask_5class uses raw mask values (0-4), NOT 4-class indices.
    - mask value 4 = IS_OS + RPE + Choroid combined in Duke dataset
    - We extract only the top 'is_os_thickness' pixels of Class 4 as IS_OS
    """
    H, W = mask_5class.shape
    is_os_mask = np.zeros((H, W), dtype=np.float32)

    for col in range(W):
        col_mask = mask_5class[:, col]
        class4_positions = np.where(col_mask == 4)[0]

        if len(class4_positions) == 0:
            continue

        # Top of Class 4 region is the IS_OS junction
        is_os_start = class4_positions[0]
        is_os_end = min(is_os_start + is_os_thickness, H)

        # Only mark the IS_OS band (not the entire Class 4)
        is_os_mask[is_os_start:is_os_end, col] = 1.0

    return is_os_mask


def add_calibrated_noise_by_profile(image, profile='duke17', noise_scale=1.0):
    """Add calibrated OCT noise matching specific dataset profile."""
    if HAS_CALIBRATED_NOISE:
        return add_calibrated_oct_noise(image, dataset=profile, noise_scale=noise_scale, random_mix=True)

    # Fallback
    params = {
        'duke17': {'speckle_scale': 0.50, 'gaussian_std': 0.10},
        'duke28': {'speckle_scale': 0.48, 'gaussian_std': 0.10},
        'pku37': {'speckle_scale': 0.35, 'gaussian_std': 0.07},
    }
    p = params.get(profile, params['duke17'])

    speckle = 1.0 + p['speckle_scale'] * noise_scale * (np.random.exponential(1.0, image.shape) - 1.0)
    noisy = image * np.maximum(speckle, 0.01)
    gaussian = np.random.randn(*image.shape).astype(np.float32) * p['gaussian_std'] * noise_scale
    noisy = noisy + gaussian

    return np.clip(noisy, 0, 1).astype(np.float32)


class TMITrainDataset(Dataset):
    """Training dataset with patches for TMI enhanced training.

    Supports stratified sampling to ensure all retinal layers are well-represented:
    - 50% IS/OS focused (center region)
    - 25% RNFL/GCL focused (top region)
    - 25% RPE/Choroid focused (bottom region)
    """

    def __init__(self, jsonl_path, max_samples=None, noise_levels=None, patch_size=64,
                 stratified=False, center_focused=False, is_os_depth_range=(0.50, 0.80)):
        """
        Args:
            jsonl_path: Path to JSONL file with sample paths
            max_samples: Maximum samples to use
            noise_levels: List of noise scales
            patch_size: Patch size for training
            stratified: If True, use stratified sampling (50% IS/OS, 25% top, 25% bottom)
            center_focused: If True, always crop patches centered around IS/OS layer (legacy)
            is_os_depth_range: Expected depth range of IS/OS (fraction of image height)
        """
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.noise_levels = noise_levels or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.patch_size = patch_size
        self.stratified = stratified
        self.center_focused = center_focused
        self.is_os_depth_range = is_os_depth_range

        # Assign noise profiles
        np.random.seed(42)
        n_samples = len(self.samples)
        group_assignments = np.random.choice(3, size=n_samples)
        self.sample_noise_profiles = [NOISE_PROFILES[g] for g in group_assignments]

    def __len__(self):
        return len(self.samples) * len(self.noise_levels)

    def _get_stratified_crop(self, H, W, mask_5class, idx):
        """Get a crop position using stratified sampling.

        Strategy:
        - 50% of samples: IS/OS focused (center, 50-75% depth)
        - 25% of samples: RNFL/GCL focused (top, 0-25% depth)
        - 25% of samples: RPE/Choroid focused (bottom, 75-100% depth)
        """
        # Determine which region to sample based on index
        region_selector = idx % 4  # 0,1 = IS/OS, 2 = top, 3 = bottom

        if region_selector <= 1:  # 50% - IS/OS focused
            # Find IS/OS layer in the mask (class 3 in 5-class)
            is_os_rows = np.where(np.any(mask_5class == 3, axis=1))[0]
            if len(is_os_rows) > 0:
                center = (is_os_rows.min() + is_os_rows.max()) // 2
            else:
                center = int(H * 0.65)  # Default IS/OS position
            jitter = np.random.randint(-self.patch_size // 4, self.patch_size // 4)
            top = center - self.patch_size // 2 + jitter

        elif region_selector == 2:  # 25% - RNFL/GCL focused (top)
            # Find RNFL/GCL region (class 1 in 5-class = RNFL, class 2 = GCL)
            rnfl_rows = np.where(np.any((mask_5class == 1) | (mask_5class == 2), axis=1))[0]
            if len(rnfl_rows) > 0:
                center = (rnfl_rows.min() + rnfl_rows.max()) // 2
            else:
                center = int(H * 0.10)  # Default top position
            jitter = np.random.randint(-self.patch_size // 6, self.patch_size // 6)
            top = center - self.patch_size // 2 + jitter

        else:  # 25% - RPE/Choroid focused (bottom)
            # Find RPE/Choroid region (class 4 in 5-class = RPE, class 5 = Choroid if exists)
            rpe_rows = np.where(np.any(mask_5class >= 4, axis=1))[0]
            if len(rpe_rows) > 0:
                center = (rpe_rows.min() + rpe_rows.max()) // 2
            else:
                center = int(H * 0.85)  # Default bottom position
            jitter = np.random.randint(-self.patch_size // 6, self.patch_size // 6)
            top = center - self.patch_size // 2 + jitter

        # Clamp to valid range
        top = max(0, min(top, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        return top, left

    def _get_is_os_focused_crop(self, H, W, mask_5class):
        """Get a crop position that ensures IS/OS layer is included (legacy)."""
        is_os_rows = np.where(np.any(mask_5class == 3, axis=1))[0]

        if len(is_os_rows) > 0:
            is_os_center = (is_os_rows.min() + is_os_rows.max()) // 2
            jitter = np.random.randint(-self.patch_size // 4, self.patch_size // 4)
            top = is_os_center - self.patch_size // 2 + jitter
        else:
            depth_min, depth_max = self.is_os_depth_range
            expected_center = int(H * (depth_min + depth_max) / 2)
            jitter = np.random.randint(-self.patch_size // 4, self.patch_size // 4)
            top = expected_center - self.patch_size // 2 + jitter

        top = max(0, min(top, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        return top, left

    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        noise_idx = idx // len(self.samples)
        noise_scale = self.noise_levels[noise_idx]

        sample = self.samples[sample_idx]
        clean_image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        H, W = clean_image.shape

        if self.stratified:
            # Stratified sampling: 50% IS/OS, 25% top, 25% bottom
            top, left = self._get_stratified_crop(H, W, mask_5class, idx)
            mask_patch = mask_5class[top:top+self.patch_size, left:left+self.patch_size]
        elif self.center_focused:
            # Center-focused cropping (legacy): ensure IS/OS is in the patch
            top, left = self._get_is_os_focused_crop(H, W, mask_5class)
            mask_patch = mask_5class[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Original random cropping with layer diversity check
            for _ in range(10):
                top = np.random.randint(0, max(1, H - self.patch_size))
                left = np.random.randint(0, max(1, W - self.patch_size))
                mask_patch = mask_5class[top:top+self.patch_size, left:left+self.patch_size]
                if len(np.unique(remap_mask_to_4class(mask_patch))) >= 2:
                    break

        clean_patch = clean_image[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Add noise
        noise_profile = self.sample_noise_profiles[sample_idx]
        noisy_patch = add_calibrated_noise_by_profile(clean_patch, profile=noise_profile, noise_scale=noise_scale)

        mask_4class = remap_mask_to_4class(mask_patch)
        is_os_boundary = extract_is_os_boundary(mask_patch)

        # Create depth encoding: normalized Y position in original image
        # This tells the model WHERE in the image this patch came from
        depth_start = top / H  # Normalized start position (0=top, 1=bottom)
        depth_end = (top + self.patch_size) / H
        depth_channel = np.linspace(depth_start, depth_end, self.patch_size).reshape(-1, 1)
        depth_channel = np.broadcast_to(depth_channel, (self.patch_size, self.patch_size))

        return {
            'noisy': torch.from_numpy(noisy_patch).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_patch.astype(np.float32)).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_4class).long(),  # 4-class segmentation target
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
            'depth': torch.from_numpy(depth_channel.astype(np.float32)).float().unsqueeze(0),
        }


class TMIValDataset(Dataset):
    """Validation dataset with patches (consistent with training).

    For each sample, extracts multiple stratified patches to cover all layers:
    - Patch at IS/OS region
    - Patch at RNFL/GCL region
    - Patch at RPE/Choroid region
    """

    def __init__(self, jsonl_path, max_samples=None, noise_profile='combined',
                 patch_size=128, patches_per_image=3):
        """
        Args:
            jsonl_path: Path to JSONL file
            max_samples: Maximum samples to use
            noise_profile: Noise profile for synthetic noise
            patch_size: Patch size (should match training)
            patches_per_image: Number of patches per image (3 for stratified)
        """
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.noise_profile = noise_profile
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, idx):
        sample_idx = idx // self.patches_per_image
        patch_type = idx % self.patches_per_image  # 0=IS/OS, 1=top, 2=bottom

        sample = self.samples[sample_idx]
        clean_image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        H, W = clean_image.shape

        # Deterministic patch positions for reproducible validation
        np.random.seed(idx)  # Reproducible

        if patch_type == 0:  # IS/OS focused
            is_os_rows = np.where(np.any(mask_5class == 3, axis=1))[0]
            if len(is_os_rows) > 0:
                center = (is_os_rows.min() + is_os_rows.max()) // 2
            else:
                center = int(H * 0.65)
            top = max(0, min(center - self.patch_size // 2, H - self.patch_size))

        elif patch_type == 1:  # RNFL/GCL focused (top)
            rnfl_rows = np.where(np.any((mask_5class == 1) | (mask_5class == 2), axis=1))[0]
            if len(rnfl_rows) > 0:
                center = (rnfl_rows.min() + rnfl_rows.max()) // 2
            else:
                center = int(H * 0.10)
            top = max(0, min(center - self.patch_size // 2, H - self.patch_size))

        else:  # RPE/Choroid focused (bottom)
            rpe_rows = np.where(np.any(mask_5class >= 4, axis=1))[0]
            if len(rpe_rows) > 0:
                center = (rpe_rows.min() + rpe_rows.max()) // 2
            else:
                center = int(H * 0.85)
            top = max(0, min(center - self.patch_size // 2, H - self.patch_size))

        # Use center of image horizontally for consistency
        left = max(0, (W - self.patch_size) // 2)

        # Extract patches
        clean_patch = clean_image[top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask_5class[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        noisy_patch = add_calibrated_noise_by_profile(clean_patch, profile=self.noise_profile, noise_scale=1.0)
        mask_4class = remap_mask_to_4class(mask_patch)
        is_os_boundary = extract_is_os_boundary(mask_patch)

        # Create depth encoding: normalized Y position in original image
        depth_start = top / H
        depth_end = (top + self.patch_size) / H
        depth_channel = np.linspace(depth_start, depth_end, self.patch_size).reshape(-1, 1)
        depth_channel = np.broadcast_to(depth_channel, (self.patch_size, self.patch_size))

        return {
            'noisy': torch.from_numpy(noisy_patch).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_patch.astype(np.float32)).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_4class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
            'depth': torch.from_numpy(depth_channel.astype(np.float32)).float().unsqueeze(0),
            'patch_type': patch_type,  # For debugging: 0=IS/OS, 1=top, 2=bottom
        }


class SimpleBoundarySegmenter(nn.Module):
    """Simple 4-class segmenter for clinical OCT regions.

    4 classes: RNFL_GCL(0), INL_OPL_ONL(1), IS_OS(2), RPE_Choroid(3)
    No separate boundary_head - IS_OS is now a segmentation class.
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=32):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Output: single 4-class segmentation head (no separate boundary_head)
        self.seg_head = nn.Conv2d(base_filters, num_classes, 1)


# =============================================================================
# Enhanced Hybrid Segmenter with Sub-Pixel Boundary Regression (KEY FOR <3px MAE)
# =============================================================================

def soft_argmax_1d(heatmap, beta=100.0):
    """
    Soft argmax for sub-pixel precision boundary detection.

    Instead of hard argmax (integer position), uses softmax-weighted
    average of positions for differentiable sub-pixel localization.

    Args:
        heatmap: [B, H, W] or [B, 1, H, W] - per-column probability distribution
        beta: Temperature for softmax (higher = sharper, closer to hard argmax)

    Returns:
        positions: [B, W] - sub-pixel y-coordinates
    """
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)  # [B, H, W]

    B, H, W = heatmap.shape
    device = heatmap.device

    # Create y-coordinate grid
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    y_coords = y_coords.expand(B, H, W)  # [B, H, W]

    # Soft argmax: weighted average of positions
    weights = F.softmax(beta * heatmap, dim=1)  # [B, H, W]
    positions = (weights * y_coords).sum(dim=1)  # [B, W]

    return positions


def create_gaussian_heatmap(y_positions, height, sigma=2.0):
    """
    Create Gaussian heatmap from boundary y-positions.

    Used for training: converts discrete boundary position to
    soft Gaussian distribution for regression target.

    Args:
        y_positions: [B, W] - boundary y-coordinates (can be sub-pixel)
        height: int - image height
        sigma: float - Gaussian standard deviation (controls sharpness)

    Returns:
        heatmap: [B, H, W] - Gaussian heatmaps centered at boundary positions
    """
    B, W = y_positions.shape
    device = y_positions.device

    # Create y-coordinate grid
    y_coords = torch.arange(height, device=device, dtype=torch.float32).view(1, -1, 1)
    y_coords = y_coords.expand(B, height, W)  # [B, H, W]

    # Expand positions for broadcasting
    y_pos = y_positions.unsqueeze(1)  # [B, 1, W]

    # Gaussian: exp(-0.5 * ((y - mu) / sigma)^2)
    heatmap = torch.exp(-0.5 * ((y_coords - y_pos) / sigma) ** 2)

    # Normalize each column to sum to 1
    heatmap = heatmap / (heatmap.sum(dim=1, keepdim=True) + 1e-8)

    return heatmap


def extract_boundary_positions_subpixel(mask, num_classes=4):
    """
    Extract ground truth positions for ALL 4 boundaries with sub-pixel precision.

    OPTIMIZED: Fully vectorized - no Python loops. ~100x faster than loop version.

    Boundaries (4 total for 4 classes):
      - Boundary 0: ILM (top of class 0 - top of retina)
      - Boundary 1: Class 0 / Class 1 interface (RNFL_GCL / INL_OPL_ONL)
      - Boundary 2: Class 1 / Class 2 interface (INL_OPL_ONL / IS_OS) <- critical
      - Boundary 3: Class 2 / Class 3 interface (IS_OS / RPE_Choroid) <- critical

    Args:
        mask: [B, H, W] - class labels (0 to num_classes-1)

    Returns:
        boundaries: [B, 4, W] - y-positions (sub-pixel)
        valid_mask: [B, 4, W] - bool mask for valid boundaries
    """
    B, H, W = mask.shape
    num_boundaries = 4  # Fixed at 4 boundaries
    device = mask.device

    boundaries = torch.zeros(B, num_boundaries, W, device=device)
    valid_mask = torch.zeros(B, num_boundaries, W, dtype=torch.bool, device=device)

    # Create row indices [1, H, 1] for broadcasting
    row_indices = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

    # Helper function: find first occurrence of class c in each column
    def find_first(class_idx):
        class_mask = (mask == class_idx).float()  # [B, H, W]
        # Set non-class positions to H (will be filtered by min)
        masked_rows = torch.where(class_mask > 0, row_indices, torch.tensor(float(H), device=device))
        first_pos = masked_rows.min(dim=1)[0]  # [B, W]
        has_class = (first_pos < H)  # [B, W] - True if class exists
        first_pos = torch.where(has_class, first_pos, torch.zeros_like(first_pos))
        return first_pos, has_class

    # Helper function: find last occurrence of class c in each column
    def find_last(class_idx):
        class_mask = (mask == class_idx).float()  # [B, H, W]
        # Set non-class positions to -1 (will be filtered by max)
        masked_rows = torch.where(class_mask > 0, row_indices, torch.tensor(-1.0, device=device))
        last_pos = masked_rows.max(dim=1)[0]  # [B, W]
        has_class = (last_pos >= 0)  # [B, W]
        last_pos = torch.where(has_class, last_pos, torch.zeros_like(last_pos))
        return last_pos, has_class

    # Find first/last positions for each class
    first_0, has_0 = find_first(0)
    last_0, _ = find_last(0)
    first_1, has_1 = find_first(1)
    last_1, _ = find_last(1)
    first_2, has_2 = find_first(2)
    last_2, _ = find_last(2)
    first_3, has_3 = find_first(3)

    # Boundary 0: ILM (top of class 0)
    boundaries[:, 0, :] = first_0
    valid_mask[:, 0, :] = has_0

    # Boundary 1: Class 0 / Class 1 interface (midpoint)
    boundaries[:, 1, :] = (last_0 + first_1) / 2.0
    valid_mask[:, 1, :] = has_0 & has_1

    # Boundary 2: Class 1 / Class 2 interface (midpoint)
    boundaries[:, 2, :] = (last_1 + first_2) / 2.0
    valid_mask[:, 2, :] = has_1 & has_2

    # Boundary 3: Class 2 / Class 3 interface (midpoint)
    boundaries[:, 3, :] = (last_2 + first_3) / 2.0
    valid_mask[:, 3, :] = has_2 & has_3

    return boundaries, valid_mask


class GaussianBoundaryLoss(nn.Module):
    """
    Loss for Gaussian heatmap boundary regression with IS/OS emphasis.

    Key features:
    - Uses KL divergence between predicted and target Gaussian heatmaps
    - Higher weight on IS/OS boundaries (indices 2 and 3) for clinical importance
    - Smooth L1 on extracted positions for direct supervision
    - Supports 4 boundaries: ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE

    Boundary indices:
      - 0: ILM (top of retina)
      - 1: RNFL_GCL / INL_OPL_ONL
      - 2: INL_OPL_ONL / IS_OS (critical)
      - 3: IS_OS / RPE_Choroid (critical)
    """

    def __init__(self, num_boundaries=4, is_os_weight=3.0, sigma=2.0):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.is_os_weight = is_os_weight
        self.sigma = sigma
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_heatmaps, pred_positions, gt_positions, valid_mask, height):
        """
        Args:
            pred_heatmaps: [B, 4, H, W] - predicted Gaussian heatmaps
            pred_positions: [B, 4, W] - predicted y-positions (from soft argmax)
            gt_positions: [B, 4, W] - ground truth y-positions
            valid_mask: [B, 4, W] - valid boundary mask
            height: int - image height for normalization

        Returns:
            loss: scalar
            stats: dict with per-boundary MAE
        """
        B, num_b, H, W = pred_heatmaps.shape
        device = pred_heatmaps.device

        # BUG FIX: Initialize as tensor, not float, to ensure .item() works
        total_loss = torch.tensor(0.0, device=device)
        stats = {}

        # Weights for each boundary (IS/OS boundaries get higher weight)
        # FIX: boundary 1 increased from 2.0 to 5.0 to prevent class 1 collapse
        boundary_weights = torch.ones(num_b, device=device)
        if num_b > 0:
            boundary_weights[0] = 1.0  # ILM - normal weight
        if num_b > 1:
            boundary_weights[1] = 5.0  # RNFL/INL - INCREASED (was 2.0) - critical for thin class 1
        if num_b > 2:
            boundary_weights[2] = self.is_os_weight  # INL/IS_OS - critical
        if num_b > 3:
            boundary_weights[3] = self.is_os_weight  # IS_OS/RPE - critical

        boundary_names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']

        for i in range(min(num_b, self.num_boundaries)):
            mask_i = valid_mask[:, i, :]  # [B, W]

            # Require at least 50 valid columns for reliable boundary loss
            if mask_i.sum() < 50:
                continue

            # Create target Gaussian heatmap
            gt_heatmap = create_gaussian_heatmap(gt_positions[:, i, :], H, self.sigma)  # [B, H, W]
            pred_heatmap = pred_heatmaps[:, i, :, :]  # [B, H, W]

            # KL divergence loss (with numerical stability)
            pred_log = torch.log(pred_heatmap + 1e-8)
            gt_log = torch.log(gt_heatmap + 1e-8)
            kl_div = (gt_heatmap * (gt_log - pred_log)).sum(dim=1)  # [B, W]
            kl_loss = (kl_div * mask_i.float()).sum() / (mask_i.sum() + 1e-8)

            # Position loss (direct supervision on extracted positions)
            pred_pos = pred_positions[:, i, :]  # [B, W]
            gt_pos = gt_positions[:, i, :]  # [B, W]
            pos_error = self.smooth_l1(pred_pos, gt_pos)  # [B, W]
            pos_loss = (pos_error * mask_i.float()).sum() / (mask_i.sum() + 1e-8)

            # Combined loss with boundary weight
            boundary_loss = (kl_loss + pos_loss) * boundary_weights[i]
            total_loss = total_loss + boundary_loss

            # Stats: MAE in pixels for this boundary
            with torch.no_grad():
                mae = (torch.abs(pred_pos - gt_pos) * mask_i.float()).sum() / (mask_i.sum() + 1e-8)
                if i < len(boundary_names):
                    stats[f'{boundary_names[i]}_mae'] = mae.item()

        # Normalize by number of boundaries
        total_loss = total_loss / max(num_b, 1)

        return total_loss, stats


# =============================================================================
# Columnar Attention for IS/OS boundary detection (from improved segmenter)
# =============================================================================
class ColumnarAttentionLight(nn.Module):
    """
    Lightweight column-wise attention for OCT layer segmentation.
    OCT layers are roughly horizontal, so we apply attention along vertical axis.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.horizontal_smooth = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3),
                                           groups=channels, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Sample columns for efficiency on large images
        if W > 64:
            x_sampled = x[:, :, :, ::2]
            out_sampled = self._attention(x_sampled)
            out = F.interpolate(out_sampled, size=(H, W), mode='bilinear', align_corners=False)
        else:
            out = self._attention(x)
        return self.norm(x + self.gamma * out)

    def _attention(self, x):
        B, C, H, W = x.shape
        q = self.query(x).permute(0, 3, 2, 1).reshape(B * W, H, -1)
        k = self.key(x).permute(0, 3, 2, 1).reshape(B * W, H, -1)
        v = self.value(x).permute(0, 3, 2, 1).reshape(B * W, H, -1)
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = (attn @ v).reshape(B, W, H, C).permute(0, 3, 2, 1)
        return self.horizontal_smooth(out)


class HybridBoundarySegmenterV2(nn.Module):
    """
    Enhanced Hybrid Segmenter with Gaussian Heatmap Boundary Regression.

    Key innovations for achieving <3px IS/OS MAE:
    1. Segmentation head for coarse structure (4-class)
    2. Gaussian heatmap prediction for each boundary (sub-pixel precision)
    3. Soft argmax for differentiable position extraction
    4. Deep supervision from intermediate features
    5. [NEW] Dilated convolutions for larger receptive field (covers IS/OS ~8.5px)
    6. [NEW] Columnar attention for horizontal layer continuity
    7. [UPDATED] 4 boundaries for complete layer structure

    Boundaries (4 total):
      - Boundary 0: ILM (top of retina - top of RNFL_GCL)
      - Boundary 1: RNFL_GCL / INL_OPL_ONL interface
      - Boundary 2: INL_OPL_ONL / IS_OS interface (CRITICAL)
      - Boundary 3: IS_OS / RPE_Choroid interface (CRITICAL)
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_boundaries = 4  # 4 boundaries: ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE

        # Shared Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # [IMPROVED] enc3 with dilated convolutions for larger receptive field
        # Dilation=2 captures ~27px context, critical for IS/OS layer (~8.5px thick)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )

        # [IMPROVED] Columnar attention for horizontal layer continuity
        self.col_attn = ColumnarAttentionLight(base_filters*4, reduction=4)

        # Segmentation Decoder
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Segmentation head (4-class)
        self.seg_head = nn.Conv2d(base_filters, num_classes, 1)

        # =================================================================
        # Gaussian Heatmap Boundary Heads (KEY FOR SUB-PIXEL PRECISION)
        # =================================================================
        # Each boundary gets its own heatmap prediction
        # Output is [B, num_boundaries, H, W] - one heatmap per boundary

        self.boundary_heatmap_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, self.num_boundaries, 1),
        )

        # Deep supervision: predict boundaries from intermediate decoder
        self.boundary_deep_head = nn.Conv2d(base_filters*2, self.num_boundaries, 1)

        # Learnable temperature for soft argmax (starts at 100, can be learned)
        self.soft_argmax_beta = nn.Parameter(torch.tensor(100.0))

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder
        e1 = self.enc1(x)  # [B, 32, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # [B, 128, H/4, W/4]

        # [IMPROVED] Apply columnar attention for layer continuity
        e3 = self.col_attn(e3)

        # Decoder
        d2 = self.up2(e3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 64, H/2, W/2]

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # [B, 32, H, W]

        # Segmentation output
        seg_logits = self.seg_head(d1)  # [B, 4, H, W]
        seg_probs = F.softmax(seg_logits, dim=1)

        # =================================================================
        # Gaussian Heatmap Boundary Prediction (4 boundaries)
        # =================================================================
        # Raw heatmap logits
        boundary_logits = self.boundary_heatmap_head(d1)  # [B, 4, H, W]

        # Convert to probabilities (softmax over height dimension for each boundary)
        # This gives a probability distribution over y-positions for each column
        boundary_heatmaps = F.softmax(boundary_logits, dim=2)  # [B, 4, H, W]

        # Extract sub-pixel positions using soft argmax
        boundary_positions = []
        for i in range(self.num_boundaries):
            heatmap_i = boundary_heatmaps[:, i, :, :]  # [B, H, W]
            pos_i = soft_argmax_1d(heatmap_i, beta=self.soft_argmax_beta)  # [B, W]
            boundary_positions.append(pos_i)
        boundary_positions = torch.stack(boundary_positions, dim=1)  # [B, 4, W]

        # Deep supervision from intermediate decoder
        d2_up = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=False)
        boundary_deep_logits = self.boundary_deep_head(d2_up)  # [B, 4, H, W]
        boundary_deep_heatmaps = F.softmax(boundary_deep_logits, dim=2)

        # IS_OS boundary probability (for backward compatibility)
        is_os_prob = seg_probs[:, 2:3, :, :]  # Class 2 = IS_OS

        return {
            'seg_logits': seg_logits,
            'seg_probs': seg_probs,
            'boundary_logits': is_os_prob,  # Backward compatibility
            'boundary_heatmaps': boundary_heatmaps,  # [B, 4, H, W]
            'boundary_positions': boundary_positions,  # [B, 3, W] - sub-pixel!
            'boundary_deep_heatmaps': boundary_deep_heatmaps,  # Deep supervision
            'features': d1,
        }


# =============================================================================
# TMI v4: ADAPTIVE STRENGTH PREDICTOR - Per-Pixel Denoising Strength
# =============================================================================
# Novel contribution: Instead of a scalar adaptive_scale, predict per-pixel
# denoising strength based on:
# 1. Local noise level (from high-frequency content)
# 2. Layer type (from DSP segmentation probabilities)
# 3. Boundary proximity (softer denoising near boundaries to preserve edges)
# 4. Local contrast (preserve high-contrast regions)
# =============================================================================

class AdaptiveStrengthPredictor(nn.Module):
    """
    Predicts per-pixel denoising strength map for adaptive layer-specific denoising.

    Key Innovation: Instead of a single scalar (adaptive_scale), this module
    predicts a spatially-varying strength map that determines how much of the
    layer-specific refinement to apply at each pixel.

    Output strength ∈ [0, 1] where:
        - 0 = use only NAFNet base output (no layer-specific refinement)
        - 1 = use full layer-specific head output

    The strength adapts based on:
        - Layer probabilities: different layers need different denoising strengths
        - Boundary proximity: reduce refinement near boundaries to preserve edges
        - Local noise level: apply more refinement in noisier regions
        - Feature uncertainty: reduce refinement where model is uncertain

    Final denoising formula:
        denoised = denoised_base + strength_map × (head_output - denoised_base)
               = (1 - strength_map) × denoised_base + strength_map × head_output
    """

    def __init__(
        self,
        feature_channels: int = 64,
        n_layers: int = 4,
        hidden_channels: int = 32,
        use_noise_estimation: bool = True,
        use_boundary_awareness: bool = True,
        boundary_sigma: float = 10.0,  # Softness of boundary influence (pixels)
        min_strength: float = 0.0,     # Minimum strength (prevents total ignore of heads)
        max_strength: float = 1.0,     # Maximum strength
        init_bias: float = -1.0,       # Initialize to low strength (sigmoid(-1) ≈ 0.27)
    ):
        """
        Args:
            feature_channels: Number of channels from NAFNet features
            n_layers: Number of layer classes (4 for OCT)
            hidden_channels: Hidden layer channels
            use_noise_estimation: If True, estimate local noise level
            use_boundary_awareness: If True, reduce strength near boundaries
            boundary_sigma: Softness of boundary proximity influence
            min_strength: Minimum output strength
            max_strength: Maximum output strength
            init_bias: Initial bias for output (controls starting strength)
        """
        super().__init__()

        self.n_layers = n_layers
        self.use_noise_estimation = use_noise_estimation
        self.use_boundary_awareness = use_boundary_awareness
        self.boundary_sigma = boundary_sigma
        self.min_strength = min_strength
        self.max_strength = max_strength

        # Input channels: features + layer_probs + (optional) noise_map + (optional) boundary_dist
        in_channels = feature_channels + n_layers
        if use_noise_estimation:
            in_channels += 1  # noise level map
        if use_boundary_awareness:
            in_channels += 1  # boundary distance map

        # Lightweight prediction network (~20K params)
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        # Initialize final layer to produce low initial strength
        # This prevents untrained layer heads from corrupting output
        # BUG FIX: Use small non-zero weights instead of zeros to allow gradient flow!
        # Zero weights would block gradients: d(output)/d(input) = weights = 0
        # TMI v4.1: Increased std from 0.01 to 0.1 for stronger gradient flow
        nn.init.normal_(self.predictor[-1].weight, mean=0.0, std=0.1)
        nn.init.constant_(self.predictor[-1].bias, init_bias)

        # Learnable per-layer base strengths
        # Different layers may need different denoising intensities
        self.layer_strength_bias = nn.Parameter(torch.zeros(n_layers))

        # High-pass filter for noise estimation (Laplacian)
        if use_noise_estimation:
            laplacian = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('laplacian_kernel', laplacian)

    def estimate_noise_level(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate local noise level using high-frequency content.

        Uses Laplacian filter to detect high-frequency components,
        which correlate with noise level in homogeneous regions.

        Args:
            image: Input image [B, 1, H, W]

        Returns:
            noise_map: Estimated noise level [B, 1, H, W]
        """
        # Apply Laplacian filter
        laplacian = F.conv2d(image, self.laplacian_kernel, padding=1)

        # Local variance of Laplacian (noise indicator)
        # Use average pooling to get local statistics
        laplacian_sq = laplacian ** 2
        local_var = F.avg_pool2d(laplacian_sq, kernel_size=5, stride=1, padding=2)

        # BUG FIX: Normalize per-sample, not across batch
        # Compute max for each sample independently
        B = local_var.shape[0]
        local_var_flat = local_var.view(B, -1)  # [B, H*W]
        max_vals = local_var_flat.max(dim=1, keepdim=True)[0]  # [B, 1]
        max_vals = max_vals.view(B, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting

        # Normalize to [0, 1] range per sample
        noise_map = local_var / (max_vals + 1e-8)

        return noise_map

    def compute_boundary_distance(
        self,
        boundaries: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Compute distance to nearest boundary for each pixel.

        Memory-efficient implementation that processes boundaries sequentially
        instead of creating large [B, N, H, W] tensors.

        Args:
            boundaries: DSP boundary positions [B, num_boundaries, W] in normalized [0,1]
            H: Image height
            W: Image width

        Returns:
            distance_map: Distance to nearest boundary [B, 1, H, W], normalized
        """
        B, N, W_bound = boundaries.shape
        device = boundaries.device

        # Convert boundaries to pixel positions
        boundaries_px = boundaries * (H - 1)  # [B, N, W]

        # Create y-coordinate grid [H, 1] - memory efficient, not expanded
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)  # [H, 1]

        # Initialize min_distance with large value
        min_distance = torch.full((B, 1, H, W), float('inf'), device=device)

        # Process each boundary sequentially to avoid large memory allocation
        for n in range(N):
            # Get boundary positions for this boundary [B, W]
            boundary_n = boundaries_px[:, n, :]  # [B, W]

            # Expand for broadcasting: [B, 1, 1, W] vs [1, 1, H, 1]
            boundary_expanded = boundary_n.view(B, 1, 1, W)  # [B, 1, 1, W]

            # Compute distance: |y - boundary_y| for all pixels
            # y_coords is [H, 1], boundary_expanded is [B, 1, 1, W]
            # Result: [B, 1, H, W]
            dist_to_boundary = torch.abs(y_coords.view(1, 1, H, 1) - boundary_expanded)

            # Update minimum distance
            min_distance = torch.minimum(min_distance, dist_to_boundary)

        # Normalize by image height
        distance_map = min_distance / H

        return distance_map

    def forward(
        self,
        features: torch.Tensor,
        layer_probs: torch.Tensor,
        image: torch.Tensor = None,
        boundaries: torch.Tensor = None,
    ) -> dict:
        """
        Predict per-pixel denoising strength.

        Args:
            features: NAFNet features [B, C, H, W]
            layer_probs: Layer probabilities from DSP [B, n_layers, H, W]
            image: Input/denoised image for noise estimation [B, 1, H, W]
            boundaries: DSP boundary positions [B, num_boundaries, W]

        Returns:
            Dict containing:
                - 'strength_map': Per-pixel strength [B, 1, H, W]
                - 'noise_map': Estimated noise level (if enabled)
                - 'boundary_distance': Distance to boundaries (if enabled)
        """
        B, C, H, W = features.shape
        device = features.device

        # Start with features and layer probabilities
        inputs = [features, layer_probs]

        # Optional: estimate noise level
        # BUG FIX: Must always provide noise_map if use_noise_estimation is True (channel count)
        noise_map = None
        if self.use_noise_estimation:
            if image is not None:
                noise_map = self.estimate_noise_level(image)
            else:
                # Fallback: use zeros if image not provided
                noise_map = torch.zeros(B, 1, H, W, device=device)
            inputs.append(noise_map)

        # Optional: compute boundary distance
        # BUG FIX: Must always provide boundary_distance if use_boundary_awareness is True (channel count)
        boundary_distance = None
        if self.use_boundary_awareness:
            if boundaries is not None:
                boundary_distance = self.compute_boundary_distance(boundaries, H, W)
            else:
                # Fallback: use large distance (far from boundaries) if no boundaries provided
                # This means no boundary-aware modulation, strength will be uniform
                boundary_distance = torch.ones(B, 1, H, W, device=device) * 0.5  # Normalized distance
            inputs.append(boundary_distance)

        # Concatenate all inputs
        x = torch.cat(inputs, dim=1)

        # Predict base strength
        strength_logits = self.predictor(x)  # [B, 1, H, W]

        # Add layer-specific bias
        # Each pixel gets bias based on which layer it belongs to
        layer_bias = (layer_probs * self.layer_strength_bias.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        strength_logits = strength_logits + layer_bias

        # Apply boundary-aware modulation (reduce strength near boundaries)
        if self.use_boundary_awareness and boundary_distance is not None:
            # Closer to boundary → lower strength (preserve edges)
            # boundary_weight ∈ [0, 1], 0 at boundary, 1 far from boundary
            boundary_weight = 1 - torch.exp(-boundary_distance * H / self.boundary_sigma)
            strength_logits = strength_logits + torch.log(boundary_weight + 0.1)  # Smooth modulation

        # Convert to probability and clamp to [min, max]
        strength_map = torch.sigmoid(strength_logits)
        strength_map = self.min_strength + (self.max_strength - self.min_strength) * strength_map

        return {
            'strength_map': strength_map,
            'noise_map': noise_map,
            'boundary_distance': boundary_distance,
            'strength_logits': strength_logits,  # For debugging
        }


class TMIEnhancedModel(nn.Module):
    """
    TMI Enhanced Joint Denoising + Segmentation Model (v3).

    Combines:
    - NAFNet backbone for denoising
    - HybridBoundarySegmenterV2 for 4-class segmentation + sub-pixel boundary detection
    - Layer-specific denoising heads (one per clinical region)
    - Segmentation-guided attention
    - Columnar attention for OCT structure exploitation
    - [NEW v3] Differentiable Shortest Path (DSP) for boundary detection
      * Memory efficient: O(H×W) instead of O(H×W×C²)
      * CPU-friendly: no attention matrices
      * Guaranteed layer ordering via dynamic programming
      * Joint training: boundaries guide denoising, denoising helps boundaries

    4 classes: RNFL_GCL(0), INL_OPL_ONL(1), IS_OS(2), RPE_Choroid(3)
    4 boundaries: ILM, RNFL/INL, INL/IS_OS (critical!), IS_OS/RPE (critical!)
    """

    def __init__(
        self,
        num_classes=4,
        nafnet_width=64,  # Match checkpoint: width=64
        use_layer_specific_heads=True,
        use_seg_guided_attention=True,
        use_columnar_attention=False,  # Columnar attention
        use_boundary_regression=False,  # Legacy boundary regression (memory intensive)
        use_hybrid_segmenter=True,  # HybridBoundarySegmenterV2 with Gaussian heatmaps
        use_dsp_boundaries=True,  # NEW v3: Differentiable Shortest Path (recommended!)
        dsp_only=False,  # NEW v3.2: DSP-only mode (no segmenter, derive everything from boundaries)
        num_head_blocks=2,
        head_hidden_channels=64,
        head_dropout=0.1,
        columnar_dim=128,  # Columnar feature dimension
        num_columnar_blocks=2,  # Number of columnar transformer blocks
        dsp_smoothness_weight=1.0,  # NEW v3: DSP smoothness weight
        dsp_temperature=0.1,  # NEW v3: DSP soft-min temperature
        dsp_min_gap=5,  # NEW v3: Minimum gap between boundaries (pixels)
        # NEW v3.1: Continual learning with adapters
        use_continual_learning=False,  # Enable adapter-based continual learning
        adapter_bottleneck_ratio=4,  # Bottleneck ratio for adapters
        adapter_dropout=0.1,  # Dropout for adapters
        initial_domain='spectralis',  # Initial domain name for adapters
        # NEW v3.3: Physics-Informed Fresnel Gradient Matching
        use_fresnel_physics=False,  # Enable Fresnel physics for boundary detection
        fresnel_physics_weight=0.3,  # Initial weight for physics costs (0-1)
        fresnel_learnable=True,  # Allow learning refractive indices
        # NEW v4: Adaptive Strength Map (per-pixel denoising strength)
        use_adaptive_strength_map=True,  # Enable per-pixel strength prediction
        strength_hidden_channels=32,  # Hidden channels for strength predictor
        strength_boundary_sigma=10.0,  # Softness of boundary influence
        strength_init_bias=-1.0,  # Initial bias (sigmoid(-1) ≈ 0.27)
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_layer_specific_heads = use_layer_specific_heads
        self.use_seg_guided_attention = use_seg_guided_attention
        self.use_columnar_attention = use_columnar_attention
        self.use_boundary_regression = use_boundary_regression
        self.use_hybrid_segmenter = use_hybrid_segmenter
        self.dsp_only = dsp_only  # NEW v3.2: DSP-only mode

        # BUG FIX #10: DSP-only mode requires DSP boundaries to be enabled
        if dsp_only and not use_dsp_boundaries:
            print("[WARNING] dsp_only=True requires use_dsp_boundaries=True. Enabling DSP.")
            use_dsp_boundaries = True
        self.use_dsp_boundaries = use_dsp_boundaries  # NEW v3: DSP

        # NEW v3.3: Fresnel physics for boundary detection
        self.use_fresnel_physics = use_fresnel_physics
        self.fresnel_physics_weight = fresnel_physics_weight
        self.fresnel_learnable = fresnel_learnable

        # NAFNet backbone for denoising (use NAFNetFullFiLM to match checkpoint)
        self.nafnet = NAFNetFullFiLM(
            img_channel=1,
            width=nafnet_width,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2],  # Match checkpoint: 3 stages with 2 blocks each
            dec_blk_nums=[2, 2, 2],  # Match checkpoint
        )

        # =====================================================================
        # Segmenter: Skip in DSP-only mode (boundaries provide everything)
        # =====================================================================
        if dsp_only:
            # DSP-only mode: no segmenter, derive seg_logits from DSP boundaries
            self.segmenter = None
            print("[DSP-ONLY] Segmenter DISABLED - all segmentation derived from DSP boundaries")
        elif use_hybrid_segmenter:
            # NEW: HybridBoundarySegmenterV2 with Gaussian heatmap boundary detection
            # Key for achieving <3px IS/OS MAE
            self.segmenter = HybridBoundarySegmenterV2(
                in_channels=2,  # image + depth
                num_classes=num_classes,
                base_filters=32,
            )
        else:
            # Legacy: SimpleBoundarySegmenter (no sub-pixel boundary)
            self.segmenter = SimpleBoundarySegmenter(
                in_channels=2,  # image + depth
                num_classes=num_classes,
                base_filters=32,
            )

        # TMI Enhancement: Segmentation-guided attention
        if use_seg_guided_attention:
            self.seg_attention = SegmentationGuidedAttention(
                n_classes=num_classes,
                feature_dim=nafnet_width,
            )

        # TMI Enhancement: Layer-specific denoising heads (one per clinical region)
        self.use_adaptive_strength_map = use_adaptive_strength_map
        if use_layer_specific_heads:
            self.layer_heads = EnhancedLayerSpecificDenoiser(
                encoder_channels=nafnet_width,
                n_layer_classes=num_classes,  # 4 heads for 4 classes
                hidden_channels=head_hidden_channels,
                num_head_blocks=num_head_blocks,
                dropout=head_dropout,
            )

            # NEW v4: Adaptive Strength Map - per-pixel denoising strength
            if use_adaptive_strength_map:
                self.strength_predictor = AdaptiveStrengthPredictor(
                    feature_channels=nafnet_width,
                    n_layers=num_classes,
                    hidden_channels=strength_hidden_channels,
                    use_noise_estimation=True,
                    use_boundary_awareness=True,
                    boundary_sigma=strength_boundary_sigma,
                    min_strength=0.0,
                    max_strength=1.0,
                    init_bias=strength_init_bias,
                )
                print(f"[TMI v4] Adaptive Strength Predictor enabled (init_bias={strength_init_bias})")
            else:
                # Fallback: scalar adaptive_scale (legacy mode)
                self.adaptive_scale = nn.Parameter(torch.tensor(0.1))
                print("[TMI v4] Using legacy scalar adaptive_scale")

        # Feature extraction from NAFNet (get intermediate features)
        self.feature_proj = nn.Conv2d(1, nafnet_width, 3, padding=1)

        # =====================================================================
        # NEW TMI v2: Columnar Attention (exploits OCT's columnar structure)
        # Using LIGHTWEIGHT depthwise separable convolutions for faster convergence
        # =====================================================================
        if use_columnar_attention:
            # Lightweight columnar encoder - depthwise separable convs instead of attention
            self.columnar_encoder = LightweightColumnarEncoder(
                in_channels=nafnet_width,
                dim=columnar_dim,
            )

        # =====================================================================
        # NEW TMI v2: Direct Boundary Regression (for precise boundary detection)
        # =====================================================================
        if use_boundary_regression:
            num_boundaries = num_classes + 1  # 5 boundaries for 4 classes

            # Use columnar features if available, else project from spatial
            boundary_input_dim = columnar_dim if use_columnar_attention else nafnet_width

            self.boundary_head = BoundaryRegressionHead(
                in_dim=boundary_input_dim,
                hidden_dim=256,
                num_boundaries=num_boundaries,
                dropout=head_dropout,
            )

            self.boundary_refiner = BoundaryRefiner(
                num_boundaries=num_boundaries,
                hidden_dim=64,
                kernel_size=11,
            )

            # If not using columnar encoder, need to project spatial to columnar
            if not use_columnar_attention:
                self.spatial_to_columnar = nn.Conv2d(nafnet_width, boundary_input_dim, 1)

        # =====================================================================
        # NEW TMI v3: Differentiable Shortest Path (DSP) for boundary detection
        # Novel approach: treats boundaries as shortest paths through cost volumes
        # Memory efficient: O(H×W), CPU-friendly, guaranteed layer ordering
        # =====================================================================
        self.use_continual_learning = use_continual_learning
        self.initial_domain = initial_domain

        if use_dsp_boundaries:
            if use_continual_learning:
                # TMI v3.1: Adaptive DSP with domain-specific adapters
                # Enables continual learning without forgetting
                self.dsp_detector = AdaptiveDSPBoundaryDetector(
                    in_channels=nafnet_width,
                    hidden_channels=head_hidden_channels,
                    num_boundaries=num_classes,  # 4 boundaries for 4-class segmentation
                    smoothness_weight=dsp_smoothness_weight,
                    temperature=dsp_temperature,
                    min_gap=dsp_min_gap,
                    adapter_bottleneck_ratio=adapter_bottleneck_ratio,
                    adapter_dropout=adapter_dropout,
                    initial_domain=initial_domain,
                )
            elif use_fresnel_physics:
                # NEW v3.3: Physics-Aware DSP with Fresnel gradient matching
                # Uses refractive index physics to improve boundary detection
                self.dsp_detector = PhysicsAwareDSPBoundaryDetector(
                    in_channels=nafnet_width,
                    hidden_channels=head_hidden_channels,
                    num_boundaries=num_classes,  # 4 boundaries for 4-class segmentation
                    smoothness_weight=dsp_smoothness_weight,
                    temperature=dsp_temperature,
                    min_gap=dsp_min_gap,
                    use_gradient_hint=True,  # Use image gradients as hints
                    use_soft_dtw=False,  # Faster straight-through estimator
                    # Physics parameters
                    use_fresnel_physics=True,
                    initial_physics_weight=fresnel_physics_weight,
                    learnable_physics=fresnel_learnable,
                )
                print(f"[DSP] Physics-Aware DSP enabled (Fresnel physics weight: {fresnel_physics_weight:.2f})")
            else:
                # Standard DSP without adapters or physics
                self.dsp_detector = DSPBoundaryDetector(
                    in_channels=nafnet_width,
                    hidden_channels=head_hidden_channels,
                    num_boundaries=num_classes,  # 4 boundaries for 4-class segmentation
                    smoothness_weight=dsp_smoothness_weight,
                    temperature=dsp_temperature,
                    min_gap=dsp_min_gap,
                    use_gradient_hint=True,  # Use image gradients as hints
                    use_soft_dtw=False,  # Faster straight-through estimator
                )

            # Boundary-to-attention: use DSP boundaries to guide denoising
            # Creates attention maps focused on boundary regions
            self.boundary_attention_proj = nn.Sequential(
                nn.Conv1d(num_classes, nafnet_width, 1),
                nn.ReLU(inplace=True),
            )

        # Gradient checkpointing flag
        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage."""
        self.use_gradient_checkpointing = True
        print("Gradient checkpointing ENABLED - trading compute for memory")

    def _forward_nafnet(self, x):
        """NAFNet forward for checkpointing."""
        return self.nafnet(x)

    def _forward_segmenter(self, x):
        """Segmenter forward for checkpointing."""
        # BUG FIX #11: Handle None segmenter (DSP-only mode)
        if self.segmenter is None:
            raise RuntimeError(
                "_forward_segmenter called but segmenter is None. "
                "This should not happen in DSP-only mode."
            )
        return self.segmenter(x)

    def get_physics_summary(self) -> dict:
        """
        Get summary of Fresnel physics parameters for logging.

        Returns dict with:
            - fresnel_enabled: bool
            - refractive_indices: list of learned n values
            - expected_gradient_strength: list per boundary
            - fusion_weights: list of physics/learned fusion weights
        """
        if not self.use_fresnel_physics or not hasattr(self, 'dsp_detector'):
            return {'fresnel_enabled': False}

        # Get physics summary from the DSP detector
        if hasattr(self.dsp_detector, 'get_physics_summary'):
            return self.dsp_detector.get_physics_summary()
        else:
            return {'fresnel_enabled': False}

    def forward(self, noisy, return_all=True, clean_for_seg=None, depth=None):
        """
        Forward pass.

        Args:
            noisy: Noisy input [B, 1, H, W]
            return_all: Whether to return all intermediate outputs
            clean_for_seg: Optional clean image for segmentation (prevents domain shift)
                          During training, pass clean to keep segmentation stable.
                          During inference, leave None to use denoised output.
            depth: Optional depth encoding [B, 1, H, W] indicating vertical position
                   Values in [0, 1] where 0=top, 1=bottom of original image.
                   Critical for patch-based training to maintain spatial awareness.

        Returns:
            Dict with denoised, seg_logits, boundary_positions, etc.
        """
        B, C, H, W = noisy.shape

        # Base denoising with NAFNet (with optional gradient checkpointing)
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            denoised_base = checkpoint(self._forward_nafnet, noisy, use_reentrant=False)
        else:
            denoised_base = self.nafnet(noisy)

        # Segmentation: use clean image if provided (training), otherwise denoised (inference)
        # This prevents domain shift: segmenter was trained on clean images
        seg_input = clean_for_seg if clean_for_seg is not None else denoised_base

        # Add depth encoding if provided (critical for patch-based training)
        if depth is not None:
            seg_input = torch.cat([seg_input, depth], dim=1)  # [B, 2, H, W]
        else:
            # Default: create depth encoding assuming full image (0 to 1)
            depth_default = torch.linspace(0, 1, H, device=noisy.device).view(1, 1, H, 1).expand(B, 1, H, W)
            seg_input = torch.cat([seg_input, depth_default], dim=1)

        # Get features for TMI enhancements (moved before segmenter for DSP-only mode)
        features = self.feature_proj(denoised_base)

        # =====================================================================
        # DSP-ONLY MODE: Skip segmenter, derive everything from DSP boundaries
        # =====================================================================
        boundary_heatmaps = None
        boundary_positions_subpixel = None
        boundary_deep_heatmaps = None

        if self.dsp_only:
            # DSP-only: run DSP first, then derive seg_logits from boundaries
            # DSP boundaries computed below, seg_logits/probs set after DSP block
            seg_logits = None  # Will be set after DSP
            seg_probs = None   # Will be set after DSP
            seg_out = {}       # Empty - no segmenter output
        else:
            # Standard mode: run segmenter
            if self.use_gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                seg_out = checkpoint(self._forward_segmenter, seg_input, use_reentrant=False)
            else:
                seg_out = self.segmenter(seg_input)
            seg_logits = seg_out['seg_logits']  # [B, 4, H, W]
            seg_probs = seg_out['seg_probs']    # [B, 4, H, W]

            # Extract Gaussian heatmap boundary outputs if using hybrid segmenter
            boundary_heatmaps = seg_out.get('boundary_heatmaps', None)  # [B, 4, H, W]
            boundary_positions_subpixel = seg_out.get('boundary_positions', None)  # [B, 3, W]
            boundary_deep_heatmaps = seg_out.get('boundary_deep_heatmaps', None)  # [B, 4, H, W]

        # =====================================================================
        # NEW TMI v2: Columnar Attention (run before DSP to get better features)
        # =====================================================================
        col_features = None
        if self.use_columnar_attention:
            features, col_features = self.columnar_encoder(features)

        # =====================================================================
        # NEW TMI v3: Differentiable Shortest Path (DSP) for boundary detection
        # Run DSP EARLY so we can use DSP-derived layer probs for ALL downstream
        # components: Seg-Guided Attention, Layer-Specific Heads, etc.
        # Memory efficient, CPU-friendly, guaranteed layer ordering
        # =====================================================================
        dsp_boundaries = None
        dsp_boundaries_pixels = None
        dsp_costs = None
        dsp_seg_mask = None
        dsp_layer_probs = None
        dsp_physics_info = None  # v3.3: Fresnel physics diagnostics

        if self.use_dsp_boundaries:
            # Get DSP boundaries from features
            # Only pass return_physics_info if using PhysicsAwareDSPBoundaryDetector
            if self.use_fresnel_physics:
                dsp_out = self.dsp_detector(
                    features,
                    image=denoised_base,  # Use denoised image for gradient hints
                    return_costs=True,
                    return_physics_info=True,  # v3.3: Physics diagnostics
                )
            else:
                dsp_out = self.dsp_detector(
                    features,
                    image=denoised_base,  # Use denoised image for gradient hints
                    return_costs=True,
                )
            dsp_boundaries = dsp_out['boundaries']  # [B, num_boundaries, W]
            dsp_boundaries_pixels = dsp_out['boundaries_pixels']  # [B, num_boundaries, W]
            dsp_costs = dsp_out.get('costs', None)  # [B, num_boundaries, H, W]
            dsp_physics_info = dsp_out.get('physics_info', None)  # v3.3: Fresnel physics

            # Convert DSP boundaries to segmentation mask
            B_dsp, C_dsp, H_dsp, W_dsp = noisy.shape
            dsp_seg_mask = dsp_boundaries_to_segmentation(
                dsp_boundaries, H_dsp, num_classes=self.num_classes
            )  # [B, H, W]

            # Convert DSP boundaries to soft layer probabilities
            # This is KEY for ALL downstream components!
            dsp_layer_probs = self._boundaries_to_layer_probs(
                dsp_boundaries, H_dsp, sigma=5.0
            )  # [B, num_classes, H, W]

            # =====================================================================
            # DSP-ONLY: Derive seg_logits and seg_probs from DSP layer probs
            # =====================================================================
            if self.dsp_only:
                # Convert layer probs to logits (log space) for compatibility
                seg_probs = dsp_layer_probs  # [B, num_classes, H, W]
                seg_logits = torch.log(dsp_layer_probs + 1e-8)  # [B, num_classes, H, W]

        # BUG FIX #12: Ensure seg_logits is never None (critical for downstream)
        if seg_logits is None:
            raise RuntimeError(
                "seg_logits is None after forward pass. This indicates a bug: "
                "dsp_only=True but DSP boundaries were not computed. "
                "Check that use_dsp_boundaries=True when dsp_only=True."
            )

        # Determine which layer probabilities to use for ALL downstream components
        # Priority: DSP (if enabled) > Segmenter
        if self.use_dsp_boundaries and dsp_layer_probs is not None:
            # Use DSP-derived layer probabilities for everything
            layer_probs_for_components = dsp_layer_probs
        else:
            # Use segmenter's probabilities
            layer_probs_for_components = seg_probs

        # TMI Enhancement: Segmentation-guided attention
        # NOW uses DSP-derived probs if DSP is enabled!
        if self.use_seg_guided_attention:
            attn_out = self.seg_attention(features, seg_probs=layer_probs_for_components)
            features = attn_out['features']
            boundary_attn = attn_out['boundary_attn']
        else:
            boundary_attn = None

        # TMI Enhancement: Layer-specific heads (4 heads for 4 clinical regions)
        # NEW v4: Uses adaptive strength map for per-pixel denoising strength
        strength_out = None
        if self.use_layer_specific_heads:
            # Get layer-specific denoising using layer probabilities
            # In DSP mode, this uses DSP-derived probs; otherwise segmenter probs
            head_out = self.layer_heads(
                features,
                seg_probs=layer_probs_for_components,  # DSP or segmenter probs
                boundary_mask=boundary_attn,
            )

            # NEW v4: Combine base denoising with layer-specific refinement
            # Using per-pixel adaptive strength map instead of scalar
            if self.use_adaptive_strength_map:
                # Predict per-pixel strength based on features, layer probs, and boundaries
                strength_out = self.strength_predictor(
                    features=features,
                    layer_probs=layer_probs_for_components,
                    image=denoised_base,  # For noise estimation
                    boundaries=dsp_boundaries,  # For boundary proximity (may be None)
                )
                strength_map = strength_out['strength_map']  # [B, 1, H, W]

                # Adaptive mixing: blend base and head outputs per-pixel
                # denoised = (1 - strength) * base + strength * head
                # Equivalent to: denoised = base + strength * (head - base)
                head_output = head_out['output']  # [B, 1, H, W]
                denoised = denoised_base + strength_map * (head_output - denoised_base)
            else:
                # Legacy: scalar adaptive_scale
                denoised = denoised_base + self.adaptive_scale * head_out['output']
        else:
            denoised = denoised_base
            head_out = None

        # =====================================================================
        # Legacy TMI v2: Direct Boundary Regression (kept for compatibility)
        # =====================================================================
        boundary_positions = None
        boundary_uncertainty = None
        if self.use_boundary_regression:
            if col_features is not None:
                # Use columnar features from columnar encoder
                boundary_input = col_features  # [B, W, H, dim]
            else:
                # Project spatial features to columnar format
                proj_features = self.spatial_to_columnar(features)  # [B, C, H, W]
                B, C, H, W = proj_features.shape
                boundary_input = proj_features.permute(0, 3, 2, 1)  # [B, W, H, C]

            # Predict boundary positions
            boundary_positions, boundary_uncertainty = self.boundary_head(
                boundary_input, return_uncertainty=True
            )  # [B, W, num_boundaries]

            # Refine boundaries
            boundary_positions = self.boundary_refiner(boundary_positions)

        outputs = {
            'denoised': denoised,
            'denoised_base': denoised_base,
            'seg_logits': seg_logits,
            'seg_probs': seg_probs,
            'boundary_positions': boundary_positions,  # Legacy: [B, W, num_boundaries]
            'boundary_uncertainty': boundary_uncertainty,  # Legacy: [B, W, num_boundaries]
            # Gaussian heatmap boundary outputs
            'boundary_heatmaps': boundary_heatmaps,  # [B, 4, H, W] - Gaussian heatmaps
            'boundary_positions_subpixel': boundary_positions_subpixel,  # [B, 3, W] - sub-pixel positions
            'boundary_deep_heatmaps': boundary_deep_heatmaps,  # [B, 4, H, W] - deep supervision
            # NEW v3: DSP boundary outputs
            'dsp_boundaries': dsp_boundaries,  # [B, num_boundaries, W] normalized 0-1
            'dsp_boundaries_pixels': dsp_boundaries_pixels,  # [B, num_boundaries, W] in pixels
            'dsp_costs': dsp_costs,  # [B, num_boundaries, H, W] cost volumes
            'dsp_seg_mask': dsp_seg_mask,  # [B, H, W] derived segmentation from DSP boundaries
            'dsp_layer_probs': dsp_layer_probs,  # [B, 4, H, W] DSP-derived layer probs
            'layer_probs_for_components': layer_probs_for_components,  # Actual probs used by all components
            # NEW v3.3: Fresnel physics diagnostics
            'dsp_physics_info': dsp_physics_info if self.use_fresnel_physics else None,
            # NEW v4: Adaptive strength map outputs
            'strength_map': strength_out['strength_map'] if strength_out is not None else None,
            'noise_map': strength_out.get('noise_map') if strength_out is not None else None,
            'boundary_distance': strength_out.get('boundary_distance') if strength_out is not None else None,
        }

        if return_all:
            outputs['boundary_attn'] = boundary_attn
            outputs['col_features'] = col_features
            if head_out is not None:
                outputs['layer_outputs'] = head_out.get('layer_outputs')
                outputs['head_confidences'] = head_out.get('confidences')
            if strength_out is not None:
                outputs['strength_logits'] = strength_out.get('strength_logits')

        return outputs

    def _create_boundary_attention(
        self, boundaries: torch.Tensor, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create attention maps focused on boundary regions.

        Args:
            boundaries: [B, num_boundaries, W] normalized boundary positions (0-1)
            H: Image height
            W: Image width
            device: Device

        Returns:
            attention: [B, 1, H, W] attention map with Gaussian peaks at boundaries
        """
        B, N, W_b = boundaries.shape
        sigma = 3.0  # Gaussian width in pixels

        # Create y-coordinate grid
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

        # Convert boundaries to pixel positions
        boundary_pixels = boundaries * (H - 1)  # [B, N, W]
        boundary_pixels = boundary_pixels.unsqueeze(2)  # [B, N, 1, W]

        # Compute Gaussian attention for each boundary
        # attention[y] = sum over boundaries of exp(-(y - boundary_pos)^2 / (2*sigma^2))
        diff = y_coords - boundary_pixels  # [B, N, H, W]
        gaussian = torch.exp(-diff ** 2 / (2 * sigma ** 2))  # [B, N, H, W]

        # Sum over all boundaries and normalize
        attention = gaussian.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        attention = attention / (attention.max() + 1e-8)

        return attention

    def _boundaries_to_layer_probs(
        self, boundaries: torch.Tensor, H: int, sigma: float = 5.0
    ) -> torch.Tensor:
        """
        Convert DSP boundary positions to soft layer probability maps.

        This is the KEY function that connects DSP boundaries to layer-specific
        denoising heads. Each layer gets a soft probability mask based on its
        position between boundaries.

        CORRECTED MAPPING (4 boundaries for 4 classes):
        - Boundary 0 (ILM): top of retina
        - Boundary 1 (RNFL/INL): between RNFL_GCL and INL_OPL_ONL
        - Boundary 2 (INL/IS_OS): between INL_OPL_ONL and IS_OS
        - Boundary 3 (IS_OS/RPE): between IS_OS and RPE_Choroid

        Layer mapping:
        - Layer 0 (RNFL_GCL): between boundary 0 and boundary 1
        - Layer 1 (INL_OPL_ONL): between boundary 1 and boundary 2
        - Layer 2 (IS_OS): between boundary 2 and boundary 3
        - Layer 3 (RPE_Choroid): below boundary 3

        Args:
            boundaries: [B, num_boundaries, W] normalized positions (0-1)
            H: Image height
            sigma: Softness of layer boundaries (in pixels)

        Returns:
            layer_probs: [B, num_classes, H, W] soft layer probabilities
        """
        B, N, W = boundaries.shape
        device = boundaries.device
        num_classes = N  # 4 boundaries = 4 layers

        # Convert boundaries to pixel positions
        boundaries_px = boundaries * (H - 1)  # [B, N, W]

        # Create y-coordinate grid
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        y_coords = y_coords.view(1, H, 1).expand(B, H, W)  # [B, H, W]

        # Compute soft layer masks using sigmoid transitions at boundaries
        layer_probs = torch.zeros(B, num_classes, H, W, device=device)

        for c in range(num_classes):
            if c == 0:
                # First layer (RNFL_GCL): between boundary 0 (ILM) and boundary 1 (RNFL/INL)
                b_upper = boundaries_px[:, 0:1, :].expand(B, H, W)  # ILM
                b_lower = boundaries_px[:, 1:2, :].expand(B, H, W)  # RNFL/INL

                # Soft mask: 1 between boundaries, 0 outside
                above_upper = torch.sigmoid((y_coords - b_upper) / sigma)
                below_lower = torch.sigmoid((b_lower - y_coords) / sigma)
                layer_probs[:, c] = above_upper * below_lower

            elif c == num_classes - 1:
                # Last layer (RPE_Choroid): below boundary 3 (IS_OS/RPE)
                bn = boundaries_px[:, -1:, :].expand(B, H, W)  # IS_OS/RPE
                layer_probs[:, c] = torch.sigmoid((y_coords - bn) / sigma)

            else:
                # Middle layers: between boundaries c and c+1
                # Layer 1 (INL_OPL_ONL): between b1 (RNFL/INL) and b2 (INL/IS_OS)
                # Layer 2 (IS_OS): between b2 (INL/IS_OS) and b3 (IS_OS/RPE)
                b_upper = boundaries_px[:, c:c+1, :].expand(B, H, W)
                b_lower = boundaries_px[:, c+1:c+2, :].expand(B, H, W)

                # Soft mask: 1 between boundaries, 0 outside
                above_upper = torch.sigmoid((y_coords - b_upper) / sigma)
                below_lower = torch.sigmoid((b_lower - y_coords) / sigma)
                layer_probs[:, c] = above_upper * below_lower

        # Normalize to ensure probabilities sum to 1
        layer_probs = layer_probs / (layer_probs.sum(dim=1, keepdim=True) + 1e-8)

        return layer_probs

    # =========================================================================
    # TMI v3.1: Continual Learning Helper Methods
    # =========================================================================

    def register_domain(self, domain_name: str):
        """
        Register a new domain for continual learning.

        Call this before adapting to a new OCT device/scanner.
        The backbone will be frozen and domain-specific adapters created.

        Args:
            domain_name: Name of the new domain (e.g., 'cirrus', 'topcon')
        """
        if not self.use_continual_learning:
            raise RuntimeError(
                "Continual learning not enabled. Set use_continual_learning=True"
            )
        if not hasattr(self, 'dsp_detector'):
            raise RuntimeError("DSP detector not initialized")

        self.dsp_detector.register_domain(domain_name)
        print(f"Registered new domain: {domain_name}")

    def set_active_domain(self, domain_name: str):
        """
        Set the active domain for inference or training.

        Args:
            domain_name: Name of the domain to activate
        """
        if not self.use_continual_learning:
            raise RuntimeError("Continual learning not enabled")

        self.dsp_detector.set_active_domain(domain_name)

    def freeze_for_adaptation(self):
        """
        Freeze backbone for domain adaptation.

        Only adapter parameters will be trainable after this call.
        """
        if not self.use_continual_learning:
            raise RuntimeError("Continual learning not enabled")

        self.dsp_detector.freeze_backbone()

        # Also freeze the rest of the model except DSP adapters
        for name, param in self.named_parameters():
            if 'dsp_detector' not in name:
                param.requires_grad = False
            elif 'adapter' not in name.lower():
                param.requires_grad = False

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Frozen for adaptation: {trainable:,} trainable / {total:,} total params "
              f"({100*trainable/total:.1f}%)")

    def get_adapter_params(self):
        """
        Get adapter parameters for optimizer.

        Returns only the adapter parameters that should be updated during
        domain adaptation.

        Returns:
            List of adapter parameters
        """
        if not self.use_continual_learning:
            raise RuntimeError("Continual learning not enabled")

        return self.dsp_detector.get_adapter_params()

    def get_domain_names(self) -> list:
        """Get list of registered domain names."""
        if not self.use_continual_learning:
            return []
        return self.dsp_detector.get_domain_names()

    def save_adapters(self, path: str):
        """
        Save adapter weights to file.

        Args:
            path: File path to save adapters
        """
        if not self.use_continual_learning:
            raise RuntimeError("Continual learning not enabled")

        adapter_state = {
            'domains': self.get_domain_names(),
            'active_domain': self.dsp_detector.active_domain,
            'adapter_state_dict': {
                k: v for k, v in self.dsp_detector.state_dict().items()
                if 'adapter' in k.lower()
            }
        }
        torch.save(adapter_state, path)
        print(f"Saved adapters to {path}")

    def load_adapters(self, path: str):
        """
        Load adapter weights from file.

        Args:
            path: File path to load adapters from
        """
        if not self.use_continual_learning:
            raise RuntimeError("Continual learning not enabled")

        adapter_state = torch.load(path, map_location='cpu')

        # Register domains if needed
        for domain in adapter_state['domains']:
            if domain not in self.get_domain_names():
                self.register_domain(domain)

        # Load adapter weights
        current_state = self.dsp_detector.state_dict()
        for k, v in adapter_state['adapter_state_dict'].items():
            if k in current_state:
                current_state[k] = v
        self.dsp_detector.load_state_dict(current_state)

        print(f"Loaded adapters from {path} ({len(adapter_state['domains'])} domains)")


def train_epoch(model, train_loader, optimizer, loss_fn, device, args, ns_loss_fn=None,
                boundary_loss_fn=None, gaussian_boundary_loss_fn=None,
                teacher_segmenter=None, kd_loss_fn=None, dsp_loss_fn=None,
                rayleigh_loss_fn=None, ic_loss_fn=None, epoch=0, lambda_overrides=None):
    """Train for one epoch with full neuro-symbolic constraints and LwF.

    Args:
        model: The TMI enhanced model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Base loss function (TMIJointLoss)
        device: Device to use
        args: Training arguments
        ns_loss_fn: NeuroSymbolicLoss instance (optional, for full neuro-symbolic suite)
        boundary_loss_fn: BoundaryLoss instance (optional, created in main)
        gaussian_boundary_loss_fn: GaussianBoundaryLoss instance (optional, created in main)
        teacher_segmenter: Frozen teacher segmenter for LwF (optional)
        epoch: Current epoch number (for warmup scheduling)
        lambda_overrides: Dict of lambda overrides (e.g., {'lambda_dsp': 5.0, 'lambda_seg': 0.5})
        kd_loss_fn: KnowledgeDistillationLoss instance (optional)
        dsp_loss_fn: DSPBoundaryLoss instance (optional, for DSP boundary detection)
        rayleigh_loss_fn: RayleighLikelihoodLoss instance (optional, for physics-correct OCT denoising)
        ic_loss_fn: InterferometricConsistencyLoss instance (optional, for boundary validation)

    Returns:
        Tuple of (metrics_dict, memory_exceeded_flag)
    """
    model.train()

    # IMPORTANT: If segmenter is frozen, keep it in eval mode to prevent BatchNorm updates
    # BatchNorm running statistics are updated during train() even with requires_grad=False
    if getattr(args, 'freeze_segmenter', False) and model.segmenter is not None:
        model.segmenter.eval()

    total_loss = 0
    total_psnr = 0
    total_ns_loss = 0
    total_diversity_loss = 0
    n_batches = 0

    use_full_ns = ns_loss_fn is not None and getattr(args, 'use_neuro_symbolic', False)
    lambda_ordering = getattr(args, 'lambda_symbolic_ordering', 0.1)

    # Memory-safe options
    grad_accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    memory_limit_mb = getattr(args, 'memory_limit_mb', 3500)
    aggressive_gc_flag = getattr(args, 'aggressive_gc', False)
    memory_exceeded = False

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Memory check every 10 batches
        if batch_idx % 10 == 0:
            is_safe, current_mem = check_memory_safe(memory_limit_mb)
            if not is_safe:
                print(f"\n⚠️ Memory limit exceeded: {current_mem:.0f}MB > {memory_limit_mb}MB. Stopping epoch.")
                memory_exceeded = True
                break

        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)  # 4-class segmentation target
        is_os_boundary = batch['is_os_boundary'].to(device)
        depth = batch.get('depth')  # Depth encoding for spatial awareness
        if depth is not None:
            depth = depth.to(device)

        # Only zero gradients at the start of accumulation
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        # Forward
        # ALWAYS use clean images for segmentation during training to:
        # 1. Maintain segmenter accuracy (trained on clean images)
        # 2. Prevent drift when segmenter is unfrozen (joint training)
        # The segmenter learns from clean images, denoiser learns from noisy->clean
        outputs = model(noisy, clean_for_seg=clean, depth=depth)
        denoised = outputs['denoised']
        seg_logits = outputs['seg_logits']  # [B, 4, H, W]

        # Compute base loss (denoising + 4-class segmentation)
        loss, loss_dict = loss_fn(
            pred_denoised=denoised,
            pred_seg_logits=seg_logits,
            target_clean=clean,
            target_seg=mask,
        )

        # Neuro-Symbolic Loss (KEY TMI CONTRIBUTION)
        # NOTE: When DSP is enabled, ordering is already guaranteed by shortest path algorithm
        # So we can use DSP-derived layer probs OR skip ordering loss entirely
        use_dsp_mode = getattr(args, 'use_dsp_boundaries', False)
        dsp_layer_probs = outputs.get('dsp_layer_probs', None)

        if use_dsp_mode and dsp_layer_probs is not None:
            # DSP mode: ordering is guaranteed, but we can still enforce other constraints
            # Convert DSP layer probs to logits for compatibility with existing loss functions
            dsp_logits = torch.log(dsp_layer_probs + 1e-8)  # Convert probs to log-space

            if use_full_ns:
                # Full neuro-symbolic suite with DSP-derived logits
                # Ordering loss will be ~0 since DSP guarantees ordering
                ns_loss, ns_stats = ns_loss_fn(dsp_logits, denoised=denoised)
                loss = loss + ns_loss
                ns_loss_val = ns_loss.item()
            else:
                # Basic ordering loss - will be ~0 since DSP guarantees ordering
                ordering_loss, _ = compute_symbolic_ordering_loss(dsp_logits, margin=2.0)
                loss = loss + lambda_ordering * ordering_loss
                ns_loss_val = ordering_loss.item()
        else:
            # Standard mode: use segmenter logits
            if use_full_ns:
                # Full neuro-symbolic suite: ordering + thickness + intensity + continuity + anatomy
                ns_loss, ns_stats = ns_loss_fn(seg_logits, denoised=denoised)
                loss = loss + ns_loss
                ns_loss_val = ns_loss.item()
            else:
                # Basic ordering loss only
                ordering_loss, _ = compute_symbolic_ordering_loss(seg_logits, margin=2.0)
                loss = loss + lambda_ordering * ordering_loss
                ns_loss_val = ordering_loss.item()

        # Head Diversity Loss (prevents head collapse)
        diversity_loss_val = 0.0
        lambda_diversity = getattr(args, 'lambda_head_diversity', 0.1)
        if lambda_diversity > 0 and 'layer_outputs' in outputs and outputs['layer_outputs'] is not None:
            layer_outputs = outputs['layer_outputs']
            diversity_loss = compute_head_diversity_loss(layer_outputs, target_correlation=0.5)
            loss = loss + lambda_diversity * diversity_loss
            diversity_loss_val = diversity_loss.item()

        # Per-Layer Supervision Loss (KEY: direct supervision for each head)
        # TMI v4.1: Enhanced with per-head PSNR tracking for debugging
        layer_sup_loss_val = 0.0
        head_psnr_vals = {}  # Track per-head PSNR for debugging
        layer_names = ['RNFL', 'INL', 'ISOS', 'RPE']
        lambda_layer_sup = getattr(args, 'lambda_layer_supervision', 0.5)
        if lambda_layer_sup > 0 and 'layer_outputs' in outputs and outputs['layer_outputs'] is not None:
            layer_outputs = outputs['layer_outputs']  # [B, n_heads, H, W]
            denoised_base = outputs['denoised_base']  # [B, 1, H, W]

            # BUG FIX: Use consistent layer probs (DSP-derived when DSP enabled)
            # This ensures heads are supervised on the same regions they were trained on
            layer_probs_for_sup = outputs.get('layer_probs_for_components', None)
            if layer_probs_for_sup is None:
                # Fallback to segmenter probs
                layer_probs_for_sup = F.softmax(seg_logits, dim=1)  # [B, 4, H, W]

            # Target for each head: clean - base (residual learning target)
            residual_target = clean - denoised_base  # [B, 1, H, W]

            # Per-head supervision: each head should predict its region's residual
            # FIX: Use list to collect losses, then stack to avoid graph accumulation
            head_losses = []
            n_heads = layer_outputs.shape[1]
            for h in range(n_heads):
                head_output = layer_outputs[:, h:h+1, :, :]  # [B, 1, H, W]
                head_mask = layer_probs_for_sup[:, h:h+1, :, :]  # [B, 1, H, W]

                # Weighted L1 loss for this head's region
                if head_mask.sum() > 100:
                    head_loss = (torch.abs(head_output - residual_target) * head_mask).sum()
                    head_loss = head_loss / (head_mask.sum() + 1e-8)
                    head_losses.append(head_loss)

                    # TMI v4.1: Track per-head PSNR for debugging
                    with torch.no_grad():
                        if head_mask.sum() > 1000:
                            # Reconstructed = base + head residual
                            reconstructed = denoised_base + head_output
                            mse = (((reconstructed - clean) ** 2) * head_mask).sum() / head_mask.sum()
                            if mse > 1e-10:
                                head_psnr = 10 * torch.log10(1.0 / mse)
                                head_psnr_vals[layer_names[h]] = head_psnr.item()

            if head_losses:
                layer_sup_loss = torch.stack(head_losses).mean()
                loss = loss + lambda_layer_sup * layer_sup_loss
                layer_sup_loss_val = layer_sup_loss.item()
            del head_losses, layer_outputs, residual_target, layer_probs_for_sup  # Explicit cleanup

        # =====================================================================
        # TMI v4.1: Adaptive Strength Map Loss (CRITICAL FIX)
        # Without this, the strength predictor has NO gradient signal!
        # =====================================================================
        strength_loss_val = 0.0
        strength_reg_val = 0.0
        lambda_strength = lambda_overrides.get('lambda_strength_map', getattr(args, 'lambda_strength_map', 1.0)) if lambda_overrides else getattr(args, 'lambda_strength_map', 1.0)
        lambda_strength_reg = getattr(args, 'lambda_strength_reg', 0.1)

        if (lambda_strength > 0 or lambda_strength_reg > 0) and \
           'strength_map' in outputs and outputs['strength_map'] is not None:
            strength_map = outputs['strength_map']
            denoised_base = outputs.get('denoised_base', None)

            if denoised_base is not None:
                # Get combined head output for oracle computation
                if 'layer_outputs' in outputs and outputs['layer_outputs'] is not None:
                    layer_outputs_for_oracle = outputs['layer_outputs']  # [B, n_heads, H, W]
                    layer_probs = outputs.get('layer_probs_for_components')
                    if layer_probs is not None:
                        # Weighted combination of heads
                        head_output = (layer_outputs_for_oracle * layer_probs).sum(dim=1, keepdim=True)
                    else:
                        head_output = layer_outputs_for_oracle.mean(dim=1, keepdim=True)
                    del layer_outputs_for_oracle
                else:
                    # Fallback: reconstruct from blending formula
                    eps = 1e-6
                    safe_strength = strength_map.clamp(min=eps)
                    head_output = denoised_base + (denoised - denoised_base) / safe_strength

                # Oracle-guided strength loss
                if lambda_strength > 0:
                    oracle_loss, oracle_stats = compute_oracle_strength_loss(
                        strength_map, denoised_base, head_output, clean
                    )
                    loss = loss + lambda_strength * oracle_loss
                    strength_loss_val = oracle_loss.item()

                # Regularization
                if lambda_strength_reg > 0:
                    reg_loss, reg_stats = compute_strength_regularization(
                        strength_map,
                        lambda_smooth=0.1,
                        lambda_entropy=0.05,
                    )
                    loss = loss + lambda_strength_reg * reg_loss
                    strength_reg_val = reg_loss.item()

                del head_output
            del strength_map

        # =====================================================================
        # Learning without Forgetting (LwF) - Knowledge Distillation Loss
        # =====================================================================
        kd_loss_val = 0.0
        lambda_kd = getattr(args, 'lambda_kd', 2.0)
        if teacher_segmenter is not None and kd_loss_fn is not None and lambda_kd > 0:
            # Determine teacher's expected input channels
            teacher_in_channels = getattr(teacher_segmenter, 'expected_in_channels', 1)

            # Create input for teacher based on its expected channels
            if teacher_in_channels == 1:
                # Teacher expects only image (1 channel)
                seg_input_for_teacher = clean
            else:
                # Teacher expects image + depth (2 channels)
                if depth is not None:
                    seg_input_for_teacher = torch.cat([clean, depth], dim=1)
                else:
                    # Create depth encoding if not provided
                    B, C, H, W = clean.shape
                    depth_grid = torch.linspace(0, 1, H, device=clean.device).view(1, 1, H, 1)
                    depth_grid = depth_grid.expand(B, 1, H, W)
                    seg_input_for_teacher = torch.cat([clean, depth_grid], dim=1)

            # Get teacher predictions (frozen, no gradient)
            with torch.no_grad():
                teacher_outputs = teacher_segmenter(seg_input_for_teacher)
                teacher_logits = teacher_outputs['seg_logits']

            # Compute KD loss
            kd_loss, kd_stats = kd_loss_fn(seg_logits, teacher_logits)
            loss = loss + lambda_kd * kd_loss
            kd_loss_val = kd_stats['kd_loss']

            # Cleanup
            del teacher_outputs, teacher_logits, seg_input_for_teacher

        # Boundary Regression Loss (TMI v2: Direct boundary detection) - LEGACY
        boundary_loss_val = 0.0
        lambda_boundary_reg = getattr(args, 'lambda_boundary_regression', 1.0)
        use_boundary_reg = getattr(args, 'use_boundary_regression', False)
        if use_boundary_reg and lambda_boundary_reg > 0 and boundary_loss_fn is not None:
            boundary_positions = outputs.get('boundary_positions', None)
            boundary_uncertainty = outputs.get('boundary_uncertainty', None)
            if boundary_positions is not None:
                # Extract ground truth boundaries from segmentation mask
                # mask is [B, H, W] with 4 classes
                gt_boundaries = extract_boundaries_from_segmentation(
                    mask, num_classes=4
                )  # [B, W, num_boundaries]

                # boundary_uncertainty is not used as mask here (mask is for valid columns)
                boundary_loss, boundary_stats = boundary_loss_fn(
                    boundary_positions, gt_boundaries, mask=None
                )
                loss = loss + lambda_boundary_reg * boundary_loss
                boundary_loss_val = boundary_loss.item()
                del gt_boundaries  # Explicit cleanup

        # =====================================================================
        # NEW: Gaussian Heatmap Boundary Loss (KEY FOR <3px IS/OS MAE)
        # =====================================================================
        gaussian_boundary_loss_val = 0.0
        is_os_mae_val = 0.0
        lambda_gaussian_boundary = getattr(args, 'lambda_gaussian_boundary', 2.0)
        use_hybrid_segmenter = getattr(args, 'use_hybrid_segmenter', True)

        if use_hybrid_segmenter and lambda_gaussian_boundary > 0 and gaussian_boundary_loss_fn is not None:
            boundary_heatmaps = outputs.get('boundary_heatmaps', None)
            boundary_positions_subpixel = outputs.get('boundary_positions_subpixel', None)
            boundary_deep_heatmaps = outputs.get('boundary_deep_heatmaps', None)

            if boundary_heatmaps is not None and boundary_positions_subpixel is not None:
                # Extract ground truth boundary positions
                gt_boundaries, valid_mask = extract_boundary_positions_subpixel(mask, num_classes=4)
                B, num_b, H, W = boundary_heatmaps.shape

                # Main boundary loss
                gauss_loss, gauss_stats = gaussian_boundary_loss_fn(
                    boundary_heatmaps,
                    boundary_positions_subpixel,
                    gt_boundaries,
                    valid_mask,
                    H
                )

                # Deep supervision loss (0.5 weight)
                if boundary_deep_heatmaps is not None:
                    # Extract positions from deep heatmaps - use detach to avoid graph accumulation
                    deep_num_b = boundary_deep_heatmaps.shape[1]
                    with torch.no_grad():
                        deep_positions_list = []
                        for i in range(deep_num_b):
                            heatmap_i = boundary_deep_heatmaps[:, i, :, :]
                            pos_i = soft_argmax_1d(heatmap_i, beta=100.0)
                            deep_positions_list.append(pos_i)
                        deep_positions = torch.stack(deep_positions_list, dim=1)
                        del deep_positions_list

                    # Slice gt_boundaries and valid_mask to match deep heatmap boundaries
                    deep_gt = gt_boundaries[:, :deep_num_b, :]
                    deep_mask = valid_mask[:, :deep_num_b, :]

                    deep_loss, _ = gaussian_boundary_loss_fn(
                        boundary_deep_heatmaps,
                        deep_positions.detach(),  # Detach to prevent graph accumulation
                        deep_gt,
                        deep_mask,
                        H
                    )
                    gauss_loss = gauss_loss + 0.5 * deep_loss
                    del deep_positions, deep_loss

                loss = loss + lambda_gaussian_boundary * gauss_loss
                gaussian_boundary_loss_val = gauss_loss.item() if hasattr(gauss_loss, 'item') else gauss_loss

                # Track IS/OS MAE specifically (boundary index 1)
                if 'INL_ISOS_mae' in gauss_stats:
                    is_os_mae_val = gauss_stats['INL_ISOS_mae']

                del gt_boundaries, valid_mask, gauss_loss  # Explicit cleanup

        # =====================================================================
        # NEW TMI v3: Differentiable Shortest Path (DSP) Loss
        # Memory efficient boundary detection with guaranteed layer ordering
        # =====================================================================
        dsp_loss_val = 0.0
        dsp_is_os_mae_val = 0.0
        # Check for lambda overrides (e.g., during DSP warmup)
        if lambda_overrides and 'lambda_dsp' in lambda_overrides:
            lambda_dsp = lambda_overrides['lambda_dsp']
        else:
            lambda_dsp = getattr(args, 'lambda_dsp_boundary', 2.0)
        use_dsp = getattr(args, 'use_dsp_boundaries', True)

        if use_dsp and lambda_dsp > 0 and dsp_loss_fn is not None:
            dsp_boundaries = outputs.get('dsp_boundaries', None)
            dsp_costs = outputs.get('dsp_costs', None)

            if dsp_boundaries is not None:
                # Extract ground truth boundaries from segmentation mask
                gt_dsp_boundaries, dsp_valid_mask = dsp_extract_boundaries(
                    mask, num_classes=4
                )  # [B, num_boundaries, W]

                B, _, H_img, W_img = noisy.shape

                # Compute DSP loss
                dsp_loss, dsp_stats = dsp_loss_fn(
                    dsp_boundaries,
                    gt_dsp_boundaries,
                    costs=dsp_costs,
                    valid_mask=dsp_valid_mask,
                    H=H_img,
                )

                loss = loss + lambda_dsp * dsp_loss
                dsp_loss_val = dsp_loss.item()

                # Track IS/OS boundary MAE (boundary index 2: INL/IS_OS)
                if 'INL_ISOS_mae_px' in dsp_stats:
                    dsp_is_os_mae_val = dsp_stats['INL_ISOS_mae_px']

                del gt_dsp_boundaries, dsp_valid_mask, dsp_loss  # Explicit cleanup

        # =====================================================================
        # TMI v3.3: Physics-Informed Denoising Losses (Rayleigh + Layer Intensity)
        # Proper OCT speckle model using Rayleigh distribution instead of Gaussian
        # =====================================================================
        rayleigh_loss_val = 0.0
        layer_intensity_loss_val = 0.0
        use_rayleigh = getattr(args, 'use_rayleigh_loss', False) or getattr(args, 'use_physics_denoising_loss', False)
        use_layer_intensity = getattr(args, 'use_layer_intensity_loss', False) or getattr(args, 'use_physics_denoising_loss', False)

        if use_rayleigh and rayleigh_loss_fn is not None:
            lambda_rayleigh = getattr(args, 'lambda_rayleigh', 0.5)
            # Get segmentation probabilities for layer-aware noise estimation
            seg_probs = F.softmax(seg_logits, dim=1)  # [B, 4, H, W]

            # Compute Rayleigh likelihood loss
            rayleigh_loss, rayleigh_stats = rayleigh_loss_fn(denoised, clean, seg_probs)
            loss = loss + lambda_rayleigh * rayleigh_loss
            rayleigh_loss_val = rayleigh_loss.item()

            del rayleigh_loss, seg_probs  # Explicit cleanup

        if use_layer_intensity:
            lambda_layer_int = getattr(args, 'lambda_layer_intensity', 0.2)
            seg_probs = F.softmax(seg_logits, dim=1)  # [B, 4, H, W]

            # Layer intensity consistency: ensure denoised layers have physically plausible intensities
            # Expected relative intensities from OCT physics: RPE > RNFL > IS_OS > INL
            layer_intensity_loss = compute_layer_intensity_consistency(denoised, seg_probs)
            loss = loss + lambda_layer_int * layer_intensity_loss
            layer_intensity_loss_val = layer_intensity_loss.item()

            del layer_intensity_loss, seg_probs  # Explicit cleanup

        # =====================================================================
        # TMI v3.3: Interferometric Consistency Loss (boundary validation)
        # Validates detected boundaries using OCT interferometry physics
        # =====================================================================
        ic_loss_val = 0.0
        use_interferometric = getattr(args, 'use_interferometric_loss', False)

        if use_interferometric and ic_loss_fn is not None:
            lambda_ic = getattr(args, 'lambda_interferometric', 0.3)

            # Get boundary positions from DSP or Gaussian heatmap detector
            boundary_positions = None

            # Prefer DSP boundaries (more reliable)
            dsp_boundaries = outputs.get('dsp_boundaries', None)
            if dsp_boundaries is not None:
                boundary_positions = dsp_boundaries  # [B, num_boundaries, W]

            # Fall back to Gaussian heatmap boundaries
            elif 'boundary_positions_subpixel' in outputs:
                boundary_positions = outputs['boundary_positions_subpixel']  # [B, num_boundaries, W]

            if boundary_positions is not None:
                # Compute IC loss
                ic_loss, ic_stats = ic_loss_fn(denoised, clean, boundary_positions)
                loss = loss + lambda_ic * ic_loss
                ic_loss_val = ic_loss.item()

                del ic_loss  # Explicit cleanup

        # Backward with gradient accumulation
        # Scale loss by accumulation steps for proper averaging
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        # Only step optimizer after accumulating enough gradients
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(denoised, clean)

        total_loss += loss.item()
        total_psnr += psnr
        total_ns_loss += ns_loss_val
        total_diversity_loss += diversity_loss_val
        n_batches += 1

        # Get adaptive scale/strength for monitoring
        # NEW v4: When using adaptive strength map, show mean strength instead of scalar
        strength_map = outputs.get('strength_map', None)
        if strength_map is not None:
            # Per-pixel adaptive strength - show mean
            adaptive_scale_val = strength_map.mean().item()
            strength_min = strength_map.min().item()
            strength_max = strength_map.max().item()
        elif hasattr(model, 'adaptive_scale'):
            # Legacy scalar adaptive_scale
            adaptive_scale_val = model.adaptive_scale.item()
            strength_min = strength_max = adaptive_scale_val
        else:
            adaptive_scale_val = 0.0
            strength_min = strength_max = 0.0

        # Memory info for monitoring
        current_mem_mb = get_memory_usage_mb()

        postfix = {
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr:.2f}',
            'ns': f'{ns_loss_val:.3f}',
            'lsup': f'{layer_sup_loss_val:.3f}',
            'str': f'{adaptive_scale_val:.2f}',  # Renamed from 'scale' to 'str' (strength)
            'mem': f'{current_mem_mb:.0f}MB',
        }
        # Show strength range if using adaptive strength map
        if strength_map is not None:
            postfix['str'] = f'{adaptive_scale_val:.2f}[{strength_min:.2f}-{strength_max:.2f}]'
        if boundary_loss_val > 0:
            postfix['bnd'] = f'{boundary_loss_val:.3f}'
        # LwF: Show knowledge distillation loss
        if kd_loss_val > 0:
            postfix['kd'] = f'{kd_loss_val:.3f}'
        # NEW: Show Gaussian boundary loss and IS/OS MAE
        if gaussian_boundary_loss_val > 0:
            postfix['gbnd'] = f'{gaussian_boundary_loss_val:.3f}'
        if is_os_mae_val > 0:
            mae_status = '!' if is_os_mae_val < 3.0 else ''  # Mark good MAE
            postfix['IS/OS'] = f'{is_os_mae_val:.1f}px{mae_status}'
        # DSP boundary loss and IS/OS MAE
        if dsp_loss_val > 0:
            postfix['dsp'] = f'{dsp_loss_val:.3f}'
        if dsp_is_os_mae_val > 0:
            dsp_mae_status = '!' if dsp_is_os_mae_val < 3.0 else ''  # Mark good MAE
            postfix['dsp_IS/OS'] = f'{dsp_is_os_mae_val:.1f}px{dsp_mae_status}'
        # TMI v3.3: Physics-informed denoising losses
        if rayleigh_loss_val > 0:
            postfix['ray'] = f'{rayleigh_loss_val:.3f}'
        if layer_intensity_loss_val > 0:
            postfix['lint'] = f'{layer_intensity_loss_val:.3f}'
        if ic_loss_val > 0:
            postfix['ic'] = f'{ic_loss_val:.3f}'
        # TMI v4.1: Strength map loss and per-head PSNR
        if strength_loss_val > 0:
            postfix['sLoss'] = f'{strength_loss_val:.3f}'
        if head_psnr_vals:
            # Show abbreviated per-head PSNR (e.g., "R:28 I:27 S:26 P:29")
            head_str = ' '.join([f'{k[0]}:{v:.0f}' for k, v in head_psnr_vals.items()])
            postfix['hPSNR'] = head_str
        pbar.set_postfix(postfix)

        # Explicit cleanup to prevent memory accumulation
        # IMPORTANT: Delete large tensors that may hold computation graphs
        del outputs, denoised, seg_logits, loss, scaled_loss
        del noisy, clean, mask, is_os_boundary
        if depth is not None:  # BUG FIX #15: Clean up depth tensor
            del depth

        # Clean up DSP-specific tensors if they exist (prevent memory leak)
        # BUG FIX: Use locals() instead of dir() - dir() returns module attributes, not local vars
        if 'dsp_boundaries' in locals():
            del dsp_boundaries
        if 'dsp_costs' in locals():
            del dsp_costs
        # Clean up strength map tensor (NEW v4)
        if 'strength_map' in locals() and strength_map is not None:
            del strength_map

        # Aggressive garbage collection if enabled
        if aggressive_gc_flag:
            aggressive_memory_cleanup()

    # Handle remaining gradients if not divisible by accumulation steps
    if n_batches % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()  # BUG FIX #14: Clear gradients to prevent accumulation across epochs

    avg_div = total_diversity_loss / n_batches if n_batches > 0 else 0

    # Cleanup after epoch
    aggressive_memory_cleanup()

    return total_loss / n_batches, total_psnr / n_batches, avg_div, memory_exceeded


def compute_layer_psnr(pred, target, mask, layer_idx):
    """Compute PSNR for a specific layer region."""
    layer_mask = (mask == layer_idx).float().unsqueeze(1)  # [B, 1, H, W]
    if layer_mask.sum() < 100:  # Need enough pixels
        return None

    # Masked MSE
    diff_sq = (pred - target) ** 2
    masked_mse = (diff_sq * layer_mask).sum() / (layer_mask.sum() + 1e-8)

    if masked_mse < 1e-10:
        return 50.0  # Cap at very high PSNR

    psnr = 10 * torch.log10(1.0 / masked_mse)
    return psnr.item()


def compute_layer_ssim(pred, target, mask, layer_idx, window_size=7):
    """Compute SSIM for a specific layer region."""
    layer_mask = (mask == layer_idx).float().unsqueeze(1)  # [B, 1, H, W]
    if layer_mask.sum() < 100:  # Need enough pixels
        return None

    # Compute local means with smaller window for masked regions
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Simple global computation within masked region
    pred_masked = pred * layer_mask
    target_masked = target * layer_mask
    n_pixels = layer_mask.sum()

    mu_x = pred_masked.sum() / n_pixels
    mu_y = target_masked.sum() / n_pixels

    # Variance and covariance
    var_x = ((pred - mu_x) ** 2 * layer_mask).sum() / n_pixels
    var_y = ((target - mu_y) ** 2 * layer_mask).sum() / n_pixels
    cov_xy = ((pred - mu_x) * (target - mu_y) * layer_mask).sum() / n_pixels

    ssim = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))

    return ssim.item()


def compute_dice(pred_logits, target, class_idx):
    """Compute Dice score for a specific class."""
    pred = pred_logits.argmax(dim=1)
    pred_mask = (pred == class_idx).float()
    target_mask = (target == class_idx).float()

    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()

    if union < 1:
        return None

    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()


# =============================================================================
# Clinical Metrics for TMI Publication
# =============================================================================

# Clinical thresholds for good/not good assessment
CLINICAL_THRESHOLDS = {
    'psnr': {'good': 27.0, 'unit': 'dB', 'higher_better': True},
    'ssim': {'good': 0.85, 'unit': '', 'higher_better': True},
    'is_os_mae': {'good': 3.0, 'unit': 'px', 'higher_better': False},  # <3 pixels is good
    'rnfl_thickness_error': {'good': 5.0, 'unit': 'μm', 'higher_better': False},  # <5μm is clinically acceptable
    'anatomical_validity': {'good': 95.0, 'unit': '%', 'higher_better': True},  # >95% is good
    'lpips': {'good': 0.1, 'unit': '', 'higher_better': False},  # <0.1 is good
    'edge_preservation': {'good': 0.85, 'unit': '', 'higher_better': True},  # >0.85 is good
    'dice': {'good': 0.80, 'unit': '', 'higher_better': True},
    'adaptive_gain': {'good': 0.5, 'unit': 'dB', 'higher_better': True},  # >0.5 dB gain is good
}


def get_quality_label(metric_name, value):
    """Return quality label (GOOD/POOR) based on clinical thresholds."""
    if metric_name not in CLINICAL_THRESHOLDS:
        return ""

    thresh = CLINICAL_THRESHOLDS[metric_name]
    if thresh['higher_better']:
        is_good = value >= thresh['good']
    else:
        is_good = value <= thresh['good']

    return "[GOOD]" if is_good else "[POOR]"


def compute_boundary_mae(pred_boundary, target_boundary, boundary_class=2, debug=False):
    """
    Compute Mean Absolute Error for IS/OS boundary detection.

    The IS/OS junction is detected by finding the top edge of the IS/OS region
    in both prediction and target, then computing the MAE between these positions.

    Args:
        pred_boundary: Predicted boundary logits [B, 1, H, W] (from boundary_head)
        target_boundary: Ground truth IS/OS region mask [B, H, W] or [B, 1, H, W]
        boundary_class: Unused (kept for backward compatibility)
        debug: If True, print debug statistics

    Returns:
        MAE in pixels (lower is better)
    """
    # Handle boundary logits (sigmoid for probability)
    if pred_boundary.dim() == 4:
        if pred_boundary.size(1) == 1:
            # This is boundary_logits from boundary_head
            pred_mask = torch.sigmoid(pred_boundary).squeeze(1)  # [B, H, W]
        else:
            # This is seg_logits - find transition between classes
            pred_seg = pred_boundary.argmax(dim=1)  # [B, H, W]
            pred_mask = ((pred_seg == 1) | (pred_seg == 2)).float()
            grad = torch.abs(pred_mask[:, 1:, :] - pred_mask[:, :-1, :])
            pred_mask = F.pad(grad, (0, 0, 0, 1), value=0)
    else:
        pred_mask = pred_boundary.float()

    if target_boundary.dim() == 4:
        target_boundary = target_boundary.squeeze(1)  # [B, H, W]

    B, H, W = pred_mask.shape

    # Debug statistics
    if debug:
        print(f"  DEBUG boundary MAE:")
        print(f"    pred_mask shape: {pred_mask.shape}, range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
        print(f"    target shape: {target_boundary.shape}, sum: {target_boundary.sum():.0f}")
        print(f"    pred > 0.5 count: {(pred_mask > 0.5).sum():.0f}")

    total_mae = 0
    valid_columns = 0
    pred_positions_list = []
    target_positions_list = []

    for b in range(B):
        target_mask = target_boundary[b].float()

        # For each column, find boundary position (top of IS/OS region)
        for col in range(W):
            pred_col = pred_mask[b, :, col]
            target_col = target_mask[:, col]

            # Find positions where value > 0.5
            pred_positions = torch.where(pred_col > 0.5)[0]
            target_positions = torch.where(target_col > 0.5)[0]

            if len(pred_positions) > 0 and len(target_positions) > 0:
                # Use first occurrence (top of region)
                pred_pos = pred_positions[0].float()
                target_pos = target_positions[0].float()
                total_mae += torch.abs(pred_pos - target_pos)
                valid_columns += 1
                pred_positions_list.append(pred_pos.item())
                target_positions_list.append(target_pos.item())

    if valid_columns == 0:
        if debug:
            print(f"    WARNING: No valid columns found!")
        return None

    if debug:
        import numpy as np
        pred_arr = np.array(pred_positions_list)
        target_arr = np.array(target_positions_list)
        print(f"    Valid columns: {valid_columns}/{B * W}")
        print(f"    Pred positions: mean={pred_arr.mean():.1f}, std={pred_arr.std():.1f}")
        print(f"    Target positions: mean={target_arr.mean():.1f}, std={target_arr.std():.1f}")
        print(f"    MAE: {(total_mae / valid_columns).item():.2f} px")

    return (total_mae / valid_columns).item()


def compute_boundary_mae_from_seg(seg_pred, target_boundary, debug=False):
    """
    Compute IS/OS boundary MAE using 4-class segmentation.

    OPTIMIZED: Fully vectorized - no Python loops. ~50x faster than loop version.

    With 4-class segmentation, IS_OS is class 2. We find the top edge of class 2
    and compare it to the ground truth IS/OS region top edge.

    Args:
        seg_pred: Segmentation prediction [B, C, H, W] or [B, H, W] with class labels
        target_boundary: Ground truth IS/OS region mask [B, H, W] or [B, 1, H, W]
        debug: If True, print debug statistics

    Returns:
        MAE in pixels (lower is better)
    """
    if seg_pred.dim() == 4:
        seg_pred = seg_pred.argmax(dim=1)  # [B, H, W]

    if target_boundary.dim() == 4:
        target_boundary = target_boundary.squeeze(1)  # [B, H, W]

    B, H, W = seg_pred.shape
    device = seg_pred.device

    # VECTORIZED: Find first occurrence of class 2 in each column
    row_indices = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

    # Prediction: first row where seg_pred == 2
    is_os_mask = (seg_pred == 2).float()  # [B, H, W]
    pred_masked = torch.where(is_os_mask > 0, row_indices, torch.tensor(float(H), device=device))
    pred_pos = pred_masked.min(dim=1)[0]  # [B, W] - first occurrence
    pred_valid = (pred_pos < H)  # [B, W]

    # Target: first row where target_boundary > 0.5
    target_mask = (target_boundary > 0.5).float()  # [B, H, W]
    target_masked = torch.where(target_mask > 0, row_indices, torch.tensor(float(H), device=device))
    target_pos = target_masked.min(dim=1)[0]  # [B, W]
    target_valid = (target_pos < H)  # [B, W]

    # Valid only where both pred and target have the boundary
    valid_mask = pred_valid & target_valid  # [B, W]
    valid_columns = valid_mask.sum().item()

    # Require at least 50 valid columns for reliable MAE (avoid edge cases)
    if valid_columns < 50:
        if debug:
            print(f"  DEBUG IS/OS MAE (4-class): Too few valid columns ({valid_columns} < 50)!")
        return None

    # Compute MAE only for valid columns
    mae_per_col = torch.abs(pred_pos - target_pos)  # [B, W]
    total_mae = (mae_per_col * valid_mask.float()).sum()

    if debug:
        import numpy as np
        # Extract valid positions for debug stats
        pred_arr = pred_pos[valid_mask].cpu().numpy()
        target_arr = target_pos[valid_mask].cpu().numpy()
        print(f"  DEBUG IS/OS MAE (4-class):")
        print(f"    Valid columns: {valid_columns}/{B * W}")
        print(f"    Pred (class 2 top): mean={pred_arr.mean():.1f}, std={pred_arr.std():.1f}")
        print(f"    Target (IS/OS top): mean={target_arr.mean():.1f}, std={target_arr.std():.1f}")
        print(f"    IS/OS MAE: {(total_mae / valid_columns).item():.2f} px")

    return (total_mae / valid_columns).item()


def compute_rnfl_thickness_error(pred_seg, target_seg, rnfl_class=0, pixel_to_um=3.9):
    """
    Compute RNFL thickness measurement error.

    Args:
        pred_seg: Predicted segmentation [B, C, H, W] or [B, H, W]
        target_seg: Ground truth segmentation [B, H, W]
        rnfl_class: Class index for RNFL layer
        pixel_to_um: Conversion factor (typical OCT: ~3.9 μm/pixel axially)

    Returns:
        Thickness error in micrometers
    """
    if pred_seg.dim() == 4 and pred_seg.size(1) > 1:
        pred_seg = pred_seg.argmax(dim=1)  # [B, H, W]

    B, H, W = pred_seg.shape
    thickness_errors = []

    for b in range(B):
        pred_rnfl = (pred_seg[b] == rnfl_class).float()
        target_rnfl = (target_seg[b] == rnfl_class).float()

        # Compute thickness per column
        for col in range(W):
            pred_thickness = pred_rnfl[:, col].sum()
            target_thickness = target_rnfl[:, col].sum()

            if target_thickness > 0:
                error = torch.abs(pred_thickness - target_thickness) * pixel_to_um
                thickness_errors.append(error.item())

    if len(thickness_errors) == 0:
        return None

    return sum(thickness_errors) / len(thickness_errors)


def compute_anatomical_validity(pred_seg, n_classes=3):
    """
    Check if predicted segmentation respects anatomical layer ordering.

    OPTIMIZED: Fully vectorized - no Python loops. ~100x faster than loop version.

    Layers should appear in order from top to bottom:
    - Class 0 (RNFL_GCL) should be above Class 1 (INL_OPL_ONL)
    - Class 1 (INL_OPL_ONL) should be above Class 2 (RPE_Choroid)

    Returns:
        Percentage of columns with valid anatomical ordering (0-100)
    """
    if pred_seg.dim() == 4 and pred_seg.size(1) > 1:
        pred_seg = pred_seg.argmax(dim=1)  # [B, H, W]

    B, H, W = pred_seg.shape
    device = pred_seg.device

    # Create row indices for centroid calculation [1, H, 1]
    row_indices = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

    # Compute centroid for each class in each column
    # centroids[c] = weighted mean of row indices where class == c
    centroids = torch.zeros(B, n_classes, W, device=device)
    class_present = torch.zeros(B, n_classes, W, dtype=torch.bool, device=device)

    for c in range(n_classes):
        class_mask = (pred_seg == c).float()  # [B, H, W]
        class_count = class_mask.sum(dim=1)  # [B, W]

        # Weighted sum of row indices
        weighted_sum = (class_mask * row_indices).sum(dim=1)  # [B, W]

        # Compute centroid (avoid division by zero)
        has_class = class_count > 0  # [B, W]
        centroids[:, c, :] = torch.where(
            has_class,
            weighted_sum / (class_count + 1e-8),
            torch.tensor(float('nan'), device=device)
        )
        class_present[:, c, :] = has_class

    # Check if at least 2 classes present in each column
    classes_per_column = class_present.sum(dim=1)  # [B, W]
    multi_class_columns = classes_per_column >= 2  # [B, W]
    total_columns = multi_class_columns.sum().item()

    if total_columns == 0:
        return 100.0

    # Check ordering: for each consecutive class pair, centroid[c] < centroid[c+1]
    # (lower row index = higher in image = correct ordering)
    valid_columns = multi_class_columns.clone()  # Start assuming all valid

    for c in range(n_classes - 1):
        both_present = class_present[:, c, :] & class_present[:, c+1, :]
        # Where both classes present, check if c's centroid < c+1's centroid
        ordering_violated = (centroids[:, c, :] > centroids[:, c+1, :]) & both_present
        # Mark violated columns as invalid
        valid_columns = valid_columns & ~ordering_violated

    valid_count = (valid_columns & multi_class_columns).sum().item()

    return (valid_count / total_columns) * 100.0


def compute_edge_preservation(pred, target, mask=None):
    """
    Compute edge preservation ratio between denoised and clean images.

    Uses Sobel edge detection to compare edge structures.

    Returns:
        Edge preservation ratio (0-1, higher is better)
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    # Compute edges
    pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
    pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-8)

    target_edge_x = F.conv2d(target, sobel_x, padding=1)
    target_edge_y = F.conv2d(target, sobel_y, padding=1)
    target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-8)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).float()
        pred_edge = pred_edge * mask
        target_edge = target_edge * mask
        n_pixels = mask.sum() + 1e-8
    else:
        n_pixels = pred_edge.numel()

    # Compute correlation between edge maps
    pred_flat = pred_edge.view(-1)
    target_flat = target_edge.view(-1)

    # Normalize
    pred_norm = pred_flat / (pred_flat.norm() + 1e-8)
    target_norm = target_flat / (target_flat.norm() + 1e-8)

    # Correlation (dot product of normalized vectors)
    correlation = (pred_norm * target_norm).sum()

    return correlation.item()


# Optional: LPIPS metric (requires lpips package)
_lpips_model = None

def compute_lpips(pred, target):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Returns:
        LPIPS score (lower is better, 0 = identical)
    """
    global _lpips_model

    try:
        import lpips

        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(pred.device)
            _lpips_model.eval()

        # LPIPS expects 3-channel images in [-1, 1]
        if pred.size(1) == 1:
            pred_3ch = pred.repeat(1, 3, 1, 1)
            target_3ch = target.repeat(1, 3, 1, 1)
        else:
            pred_3ch = pred
            target_3ch = target

        # Scale to [-1, 1]
        pred_scaled = pred_3ch * 2 - 1
        target_scaled = target_3ch * 2 - 1

        with torch.no_grad():
            lpips_val = _lpips_model(pred_scaled, target_scaled)

        result = lpips_val.mean().item()
        # Explicit cleanup of intermediate tensors
        del pred_3ch, target_3ch, pred_scaled, target_scaled, lpips_val
        return result

    except ImportError:
        return None  # LPIPS not available


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, args, ns_loss_fn=None):
    """Validate the model with per-layer metrics, NAFNet baseline comparison, and neuro-symbolic metrics."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    n_samples = 0

    # Global metrics for NAFNet baseline
    total_nafnet_psnr = 0
    total_nafnet_ssim = 0

    # Neuro-Symbolic metrics (5 components)
    ns_ordering_losses = []
    ns_thickness_losses = []
    ns_intensity_losses = []
    ns_continuity_losses = []
    ns_anatomy_losses = []
    ns_total_losses = []

    # Clinical metrics (TMI publication)
    is_os_mae_values = []  # IS/OS MAE using class 2 (IS_OS) boundary
    rnfl_thickness_errors = []
    anatomical_validity_values = []
    edge_preservation_values = []
    lpips_values = []

    # Per-layer metrics (4 classes: RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid)
    layer_names = CLASS_NAMES  # ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
    layer_psnr_noisy = {name: [] for name in layer_names}
    layer_psnr_nafnet = {name: [] for name in layer_names}  # NAFNet baseline
    layer_psnr_denoised = {name: [] for name in layer_names}  # Adaptive denoising
    layer_ssim_noisy = {name: [] for name in layer_names}
    layer_ssim_nafnet = {name: [] for name in layer_names}
    layer_ssim_denoised = {name: [] for name in layer_names}
    layer_dice = {name: [] for name in layer_names}

    # NEW v4: Adaptive strength map statistics
    strength_values = []  # Mean strength per sample

    for batch in tqdm(val_loader, desc='Validation'):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)  # 4-class segmentation target
        is_os_boundary = batch['is_os_boundary'].to(device)
        depth = batch.get('depth')  # Depth encoding for spatial awareness
        if depth is not None:
            depth = depth.to(device)

        # Full model output (adaptive denoising)
        # ALWAYS use clean images for segmentation to get accurate metrics
        # This ensures consistent evaluation regardless of freeze_segmenter setting
        outputs = model(noisy, clean_for_seg=clean, depth=depth)
        denoised = outputs['denoised']
        seg_logits = outputs['seg_logits']  # [B, 4, H, W]

        # NAFNet baseline (backbone only, no layer-specific heads)
        nafnet_output = outputs.get('denoised_base', denoised)  # Base NAFNet output

        loss, _ = loss_fn(
            pred_denoised=denoised,
            pred_seg_logits=seg_logits,
            target_clean=clean,
            target_seg=mask,
        )

        # Global metrics
        psnr = compute_psnr(denoised, clean)
        ssim = compute_ssim(denoised, clean)
        nafnet_psnr = compute_psnr(nafnet_output, clean)
        nafnet_ssim = compute_ssim(nafnet_output, clean)

        total_loss += loss.item()
        total_psnr += psnr
        total_ssim += ssim
        total_nafnet_psnr += nafnet_psnr
        total_nafnet_ssim += nafnet_ssim
        n_samples += 1

        # NEW v4: Track adaptive strength statistics
        strength_map = outputs.get('strength_map', None)
        if strength_map is not None:
            strength_values.append(strength_map.mean().item())

        # Neuro-Symbolic metrics (compute on each batch)
        if ns_loss_fn is not None:
            ns_loss, ns_stats = ns_loss_fn(seg_logits, denoised=denoised)
            ns_ordering_losses.append(ns_stats.get('ordering_loss', 0))
            ns_thickness_losses.append(ns_stats.get('thickness_loss', 0))
            ns_intensity_losses.append(ns_stats.get('intensity_loss', 0))
            ns_continuity_losses.append(ns_stats.get('continuity_loss', 0))
            ns_anatomy_losses.append(ns_stats.get('anatomy_loss', 0))
            ns_total_losses.append(ns_stats.get('total_symbolic_loss', 0))

        # Clinical metrics (TMI publication)
        # IS/OS Boundary MAE - now using class 2 (IS_OS) directly
        debug_boundary = (n_samples == 1)  # Debug first sample
        seg_pred = seg_logits.argmax(dim=1)  # [B, H, W]

        # Try boundary regression first (TMI v2), fall back to segmentation-based
        use_boundary_reg = getattr(args, 'use_boundary_regression', False)
        boundary_positions = outputs.get('boundary_positions', None)

        # Check for DSP boundaries first (DSP-only mode)
        dsp_boundaries = outputs.get('dsp_boundaries', None)
        use_dsp = getattr(args, 'use_dsp_boundaries', True) or getattr(args, 'dsp_only', False)

        if use_dsp and dsp_boundaries is not None:
            # Compute IS/OS MAE from DSP boundaries
            # dsp_boundaries: [B, num_boundaries, W] where boundary 2 = INL/IS_OS
            from nsnd_oct.nsnd.models.differentiable_shortest_path import extract_boundaries_from_mask

            gt_dsp_boundaries, dsp_valid_mask = extract_boundaries_from_mask(
                mask, num_classes=4
            )  # [B, num_boundaries, W], [B, num_boundaries, W]

            # INL/IS_OS boundary is index 2 (ILM=0, RNFL/INL=1, INL/IS_OS=2, IS_OS/RPE=3)
            is_os_idx = 2
            pred_is_os = dsp_boundaries[:, is_os_idx, :]  # [B, W] in normalized [0,1]
            gt_is_os = gt_dsp_boundaries[:, is_os_idx, :]  # [B, W] in normalized [0,1]

            # Convert to pixel coordinates
            H = noisy.shape[2]
            pred_is_os_px = pred_is_os * H
            gt_is_os_px = gt_is_os * H

            # Valid columns from mask (shape: [B, W])
            col_valid = dsp_valid_mask.bool()  # [B, W] - convert to bool for indexing

            if col_valid.sum() > 50:
                mae = torch.abs(pred_is_os_px - gt_is_os_px)[col_valid].mean().item()
                is_os_mae_values.append(mae)
                if debug_boundary:
                    print(f"[DEBUG] DSP IS/OS MAE: {mae:.2f} px")

            del gt_dsp_boundaries, dsp_valid_mask

        elif use_boundary_reg and boundary_positions is not None:
            # Compute IS/OS MAE from boundary regression output
            # boundary_positions: [B, W, num_boundaries], where boundary 3 = IS_OS_bottom
            gt_boundaries = extract_boundaries_from_segmentation(mask, num_classes=4)  # [B, W, 5]

            # IS/OS bottom boundary is index 3 (after RNFL_GCL_top, RNFL_GCL_bottom, INL_OPL_ONL_bottom)
            is_os_boundary_idx = 3
            pred_is_os = boundary_positions[:, :, is_os_boundary_idx]  # [B, W]
            gt_is_os = gt_boundaries[:, :, is_os_boundary_idx]  # [B, W]

            # Compute MAE in pixels (only where GT has valid values)
            valid_mask = (gt_is_os > 0)  # Valid boundary positions
            if valid_mask.sum() > 100:
                mae = torch.abs(pred_is_os - gt_is_os)[valid_mask].mean().item()
                is_os_mae_values.append(mae)
                if debug_boundary:
                    print(f"[DEBUG] Boundary Regression IS/OS MAE: {mae:.2f} px")
        else:
            # Fall back to segmentation-based boundary detection
            seg_mae = compute_boundary_mae_from_seg(seg_pred, is_os_boundary, debug=debug_boundary)
            if seg_mae is not None:
                is_os_mae_values.append(seg_mae)

        # RNFL Thickness Error
        rnfl_error = compute_rnfl_thickness_error(seg_logits, mask, rnfl_class=0)
        if rnfl_error is not None:
            rnfl_thickness_errors.append(rnfl_error)

        # Anatomical Validity (4-class)
        anat_validity = compute_anatomical_validity(seg_logits, n_classes=NUM_CLASSES)
        anatomical_validity_values.append(anat_validity)

        # Edge Preservation
        edge_pres = compute_edge_preservation(denoised, clean)
        edge_preservation_values.append(edge_pres)

        # LPIPS (optional - may not be available)
        lpips_val = compute_lpips(denoised, clean)
        if lpips_val is not None:
            lpips_values.append(lpips_val)

        # Per-layer metrics (4 classes)
        for idx, name in enumerate(layer_names):
            # PSNR metrics
            noisy_psnr = compute_layer_psnr(noisy, clean, mask, idx)
            nafnet_psnr_layer = compute_layer_psnr(nafnet_output, clean, mask, idx)
            denoised_psnr = compute_layer_psnr(denoised, clean, mask, idx)

            if noisy_psnr is not None:
                layer_psnr_noisy[name].append(noisy_psnr)
            if nafnet_psnr_layer is not None:
                layer_psnr_nafnet[name].append(nafnet_psnr_layer)
            if denoised_psnr is not None:
                layer_psnr_denoised[name].append(denoised_psnr)

            # SSIM metrics
            noisy_ssim = compute_layer_ssim(noisy, clean, mask, idx)
            nafnet_ssim_layer = compute_layer_ssim(nafnet_output, clean, mask, idx)
            denoised_ssim = compute_layer_ssim(denoised, clean, mask, idx)

            if noisy_ssim is not None:
                layer_ssim_noisy[name].append(noisy_ssim)
            if nafnet_ssim_layer is not None:
                layer_ssim_nafnet[name].append(nafnet_ssim_layer)
            if denoised_ssim is not None:
                layer_ssim_denoised[name].append(denoised_ssim)

            # Dice score for segmentation
            dice = compute_dice(seg_logits, mask, idx)
            if dice is not None:
                layer_dice[name].append(dice)

        # Explicit cleanup to prevent memory accumulation during validation
        del outputs, denoised, seg_logits, nafnet_output, noisy, clean, mask, is_os_boundary
        if depth is not None:  # BUG FIX #15b: Clean up depth tensor in validation
            del depth
        # BUG FIX v4: Clean up strength_map tensor in validation
        if 'strength_map' in locals() and strength_map is not None:
            del strength_map

    # Compute averages
    results = {
        'loss': total_loss / n_samples,
        'psnr': total_psnr / n_samples,
        'ssim': total_ssim / n_samples,
        'nafnet_psnr': total_nafnet_psnr / n_samples,
        'nafnet_ssim': total_nafnet_ssim / n_samples,
    }

    # NEW v4: Add average strength if using adaptive strength map
    if strength_values:
        results['avg_strength'] = sum(strength_values) / len(strength_values)
        results['min_strength'] = min(strength_values)
        results['max_strength'] = max(strength_values)

    # Add per-layer metrics
    for name in layer_names:
        # PSNR
        if layer_psnr_noisy[name]:
            results[f'{name}_noisy_psnr'] = sum(layer_psnr_noisy[name]) / len(layer_psnr_noisy[name])
        if layer_psnr_nafnet[name]:
            results[f'{name}_nafnet_psnr'] = sum(layer_psnr_nafnet[name]) / len(layer_psnr_nafnet[name])
        if layer_psnr_denoised[name]:
            results[f'{name}_denoised_psnr'] = sum(layer_psnr_denoised[name]) / len(layer_psnr_denoised[name])

        # PSNR improvements
        if layer_psnr_noisy[name] and layer_psnr_denoised[name]:
            noisy_avg = sum(layer_psnr_noisy[name]) / len(layer_psnr_noisy[name])
            denoised_avg = sum(layer_psnr_denoised[name]) / len(layer_psnr_denoised[name])
            results[f'{name}_psnr_gain_vs_noisy'] = denoised_avg - noisy_avg
        if layer_psnr_nafnet[name] and layer_psnr_denoised[name]:
            nafnet_avg = sum(layer_psnr_nafnet[name]) / len(layer_psnr_nafnet[name])
            denoised_avg = sum(layer_psnr_denoised[name]) / len(layer_psnr_denoised[name])
            results[f'{name}_psnr_gain_vs_nafnet'] = denoised_avg - nafnet_avg

        # SSIM
        if layer_ssim_noisy[name]:
            results[f'{name}_noisy_ssim'] = sum(layer_ssim_noisy[name]) / len(layer_ssim_noisy[name])
        if layer_ssim_nafnet[name]:
            results[f'{name}_nafnet_ssim'] = sum(layer_ssim_nafnet[name]) / len(layer_ssim_nafnet[name])
        if layer_ssim_denoised[name]:
            results[f'{name}_denoised_ssim'] = sum(layer_ssim_denoised[name]) / len(layer_ssim_denoised[name])

        # SSIM improvements
        if layer_ssim_noisy[name] and layer_ssim_denoised[name]:
            noisy_avg = sum(layer_ssim_noisy[name]) / len(layer_ssim_noisy[name])
            denoised_avg = sum(layer_ssim_denoised[name]) / len(layer_ssim_denoised[name])
            results[f'{name}_ssim_gain_vs_noisy'] = denoised_avg - noisy_avg
        if layer_ssim_nafnet[name] and layer_ssim_denoised[name]:
            nafnet_avg = sum(layer_ssim_nafnet[name]) / len(layer_ssim_nafnet[name])
            denoised_avg = sum(layer_ssim_denoised[name]) / len(layer_ssim_denoised[name])
            results[f'{name}_ssim_gain_vs_nafnet'] = denoised_avg - nafnet_avg

        # Dice
        if layer_dice[name]:
            results[f'{name}_dice'] = sum(layer_dice[name]) / len(layer_dice[name])

    # Add Neuro-Symbolic metrics (5 components)
    if ns_ordering_losses:
        results['ns_ordering'] = sum(ns_ordering_losses) / len(ns_ordering_losses)
        results['ns_thickness'] = sum(ns_thickness_losses) / len(ns_thickness_losses)
        results['ns_intensity'] = sum(ns_intensity_losses) / len(ns_intensity_losses)
        results['ns_continuity'] = sum(ns_continuity_losses) / len(ns_continuity_losses)
        results['ns_anatomy'] = sum(ns_anatomy_losses) / len(ns_anatomy_losses)
        results['ns_total'] = sum(ns_total_losses) / len(ns_total_losses)

    # Add Clinical metrics (TMI publication)
    if is_os_mae_values:
        results['is_os_mae'] = sum(is_os_mae_values) / len(is_os_mae_values)
    if rnfl_thickness_errors:
        results['rnfl_thickness_error'] = sum(rnfl_thickness_errors) / len(rnfl_thickness_errors)
    if anatomical_validity_values:
        results['anatomical_validity'] = sum(anatomical_validity_values) / len(anatomical_validity_values)
    if edge_preservation_values:
        results['edge_preservation'] = sum(edge_preservation_values) / len(edge_preservation_values)
    if lpips_values:
        results['lpips'] = sum(lpips_values) / len(lpips_values)

    # Cleanup accumulated lists to free memory
    del ns_ordering_losses, ns_thickness_losses, ns_intensity_losses
    del ns_continuity_losses, ns_anatomy_losses, ns_total_losses
    del is_os_mae_values, rnfl_thickness_errors, anatomical_validity_values
    del edge_preservation_values, lpips_values
    del layer_psnr_noisy, layer_psnr_nafnet, layer_psnr_denoised
    del layer_ssim_noisy, layer_ssim_nafnet, layer_ssim_denoised, layer_dice
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description='TMI Enhanced Training')

    # Data
    parser.add_argument('--train_jsonl', required=True)
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=10)

    # Model
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--nafnet_ckpt', type=str, default=None)
    parser.add_argument('--nafnet_width', type=int, default=40,
                        help='NAFNet width (must match checkpoint: 16, 40, or 41)')
    parser.add_argument('--seg_ckpt', type=str, default=None,
                        help='V4 segmentation checkpoint for initialization')

    # TMI Enhancements
    parser.add_argument('--use_layer_specific_heads', action='store_true', default=True,
                        help='Enable layer-specific denoising heads (enabled by default)')
    parser.add_argument('--no_layer_specific_heads', dest='use_layer_specific_heads', action='store_false',
                        help='Disable layer-specific denoising heads')
    parser.add_argument('--use_seg_guided_attention', action='store_true', default=True,
                        help='Enable segmentation-guided attention (enabled by default)')
    parser.add_argument('--no_seg_guided_attention', dest='use_seg_guided_attention', action='store_false',
                        help='Disable segmentation-guided attention')
    parser.add_argument('--num_head_blocks', type=int, default=2)
    parser.add_argument('--head_hidden_channels', type=int, default=64)
    parser.add_argument('--head_dropout', type=float, default=0.1)
    parser.add_argument('--freeze_backbone_initially', action='store_true')
    parser.add_argument('--freeze_segmenter', action='store_true',
                        help='Freeze segmenter weights to prevent collapse from competing losses')
    parser.add_argument('--use_soft_seg_mixing', action='store_true')

    # TMI v2: Columnar Attention and Boundary Regression (KEY CONTRIBUTION)
    parser.add_argument('--use_columnar_attention', action='store_true', default=True,
                        help='Enable columnar attention for OCT structure (enabled by default)')
    parser.add_argument('--no_columnar_attention', dest='use_columnar_attention', action='store_false',
                        help='Disable columnar attention')
    parser.add_argument('--use_boundary_regression', action='store_true',
                        help='Enable direct boundary regression (legacy, disabled by default)')
    parser.add_argument('--columnar_dim', type=int, default=128,
                        help='Columnar feature dimension')
    parser.add_argument('--num_columnar_blocks', type=int, default=2,
                        help='Number of columnar transformer blocks')
    parser.add_argument('--lambda_boundary_regression', type=float, default=1.0,
                        help='Weight for boundary regression loss (legacy)')

    # TMI v3: Gaussian Heatmap Boundary Detection (KEY FOR <3px IS/OS MAE)
    parser.add_argument('--use_hybrid_segmenter', action='store_true', default=True,
                        help='Use HybridBoundarySegmenterV2 with Gaussian heatmap boundary detection')
    parser.add_argument('--no_hybrid_segmenter', dest='use_hybrid_segmenter', action='store_false',
                        help='Disable hybrid segmenter (use simple segmenter)')
    parser.add_argument('--lambda_gaussian_boundary', type=float, default=4.0,
                        help='Weight for Gaussian heatmap boundary loss (increased 2x to prevent collapse)')

    # TMI v3: Differentiable Shortest Path (DSP) for boundary detection
    # Novel approach: Memory efficient, CPU-friendly, guaranteed layer ordering
    parser.add_argument('--use_dsp_boundaries', action='store_true', default=False,
                        help='Use Differentiable Shortest Path for boundary detection (recommended for CPU)')
    parser.add_argument('--no_dsp_boundaries', dest='use_dsp_boundaries', action='store_false',
                        help='Disable DSP boundary detection')
    parser.add_argument('--lambda_dsp_boundary', type=float, default=2.0,
                        help='Weight for DSP boundary loss')
    parser.add_argument('--dsp_smoothness_weight', type=float, default=1.0,
                        help='DSP smoothness weight (penalizes jumps between columns)')
    parser.add_argument('--dsp_temperature', type=float, default=0.1,
                        help='DSP soft-min temperature (lower = harder selection)')
    parser.add_argument('--dsp_min_gap', type=int, default=5,
                        help='Minimum gap between adjacent boundaries (pixels)')
    parser.add_argument('--dsp_only', action='store_true', default=False,
                        help='DSP-ONLY mode: No segmenter, derive all segmentation from DSP boundaries. '
                             'Guarantees 100%% anatomical validity. Implies --use_dsp_boundaries.')

    # DSP Warmup: Train DSP with higher weight initially for better boundary learning
    parser.add_argument('--dsp_warmup_epochs', type=int, default=0,
                        help='Number of epochs for DSP warmup (higher DSP weight, lower seg weight)')
    parser.add_argument('--dsp_warmup_lambda', type=float, default=5.0,
                        help='DSP loss weight during warmup phase (default: 5.0)')
    parser.add_argument('--dsp_warmup_seg_lambda', type=float, default=0.5,
                        help='Segmentation loss weight during DSP warmup (default: 0.5)')
    parser.add_argument('--learnable_dsp_weights', action='store_true', default=True,
                        help='Enable learnable boundary weights in DSP loss (auto-tuned during training)')
    parser.add_argument('--no_learnable_dsp_weights', dest='learnable_dsp_weights', action='store_false',
                        help='Disable learnable boundary weights (use fixed clinical priors)')

    # TMI v3.1: Continual Learning with Adapters
    parser.add_argument('--continual_learning', action='store_true', default=False,
                        help='Enable adapter-based continual learning for domain adaptation')
    parser.add_argument('--domain_name', type=str, default='spectralis',
                        help='Name of the current domain (e.g., spectralis, cirrus, topcon)')
    parser.add_argument('--adapt_to_domain', type=str, default=None,
                        help='If set, adapt to new domain while preserving previous domain knowledge')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone during domain adaptation (recommended)')
    parser.add_argument('--adapter_bottleneck_ratio', type=int, default=4,
                        help='Bottleneck ratio for adapters (higher = smaller adapters)')
    parser.add_argument('--adapter_dropout', type=float, default=0.1,
                        help='Dropout rate for adapters')
    parser.add_argument('--load_adapters_from', type=str, default=None,
                        help='Load pre-trained adapters from checkpoint')

    # TMI v3.3: Physics-Informed Fresnel Gradient Matching (ENABLED BY DEFAULT)
    parser.add_argument('--use_fresnel_physics', action='store_true', default=True,
                        help='Enable Fresnel physics for boundary detection (auto-enabled). Uses refractive index '
                             'differences at layer interfaces to inform boundary costs. Provides '
                             'physics-based regularization for better generalization.')
    parser.add_argument('--no_fresnel_physics', dest='use_fresnel_physics', action='store_false',
                        help='Disable Fresnel physics (use --no_fresnel_physics to turn off).')
    parser.add_argument('--fresnel_physics_weight', type=float, default=0.3,
                        help='Initial weight for Fresnel physics costs (0-1). Higher = more physics, '
                             'lower = more learned. This weight is learnable during training.')
    parser.add_argument('--fresnel_learnable', action='store_true', default=True,
                        help='Allow learning refractive indices and fusion weights during training. '
                             'If False, use fixed physics parameters from literature.')
    parser.add_argument('--no_fresnel_learnable', action='store_false', dest='fresnel_learnable',
                        help='Disable learning of physics parameters (use fixed values).')

    # TMI v3.3: Physics-Informed Denoising Losses (ENABLED BY DEFAULT)
    parser.add_argument('--use_rayleigh_loss', action='store_true', default=True,
                        help='Enable Rayleigh likelihood loss for proper OCT speckle modeling (auto-enabled). '
                             'This uses the correct statistical model for OCT noise instead of L1/L2.')
    parser.add_argument('--no_rayleigh_loss', dest='use_rayleigh_loss', action='store_false',
                        help='Disable Rayleigh loss (use --no_rayleigh_loss to turn off).')
    parser.add_argument('--lambda_rayleigh', type=float, default=0.5,
                        help='Weight for Rayleigh likelihood loss. Balances physics-correct noise '
                             'model with traditional L1 loss.')
    parser.add_argument('--rayleigh_mode', type=str, default='amplitude', choices=['amplitude', 'intensity'],
                        help='Rayleigh mode: "amplitude" for Rayleigh distribution, '
                             '"intensity" for Exponential distribution on intensity.')
    parser.add_argument('--use_layer_intensity_loss', action='store_true', default=False,
                        help='Enable layer intensity consistency loss. Ensures denoised layers '
                             'maintain physically plausible relative intensities.')
    parser.add_argument('--lambda_layer_intensity', type=float, default=0.2,
                        help='Weight for layer intensity consistency loss.')
    parser.add_argument('--use_physics_denoising_loss', action='store_true', default=False,
                        help='Enable combined physics-informed denoising loss (Rayleigh + intensity). '
                             'This is a convenience flag that enables both losses together.')

    # TMI v3.3: Interferometric Consistency Loss (boundary validation) - ENABLED BY DEFAULT
    parser.add_argument('--use_interferometric_loss', action='store_true', default=True,
                        help='Enable Interferometric Consistency loss for boundary validation (auto-enabled). '
                             'Uses OCT interferometry physics to validate detected boundaries.')
    parser.add_argument('--no_interferometric_loss', dest='use_interferometric_loss', action='store_false',
                        help='Disable interferometric loss (use --no_interferometric_loss to turn off).')
    parser.add_argument('--lambda_interferometric', type=float, default=0.3,
                        help='Weight for interferometric consistency loss.')
    parser.add_argument('--ic_use_fresnel_prior', action='store_true', default=True,
                        help='Use Fresnel-based expected gradient magnitudes in IC loss.')
    parser.add_argument('--no_ic_fresnel_prior', action='store_false', dest='ic_use_fresnel_prior',
                        help='Disable Fresnel prior in IC loss.')

    # TMI v4: Adaptive Strength Map (per-pixel denoising strength)
    parser.add_argument('--use_adaptive_strength_map', action='store_true', default=True,
                        help='Enable per-pixel adaptive denoising strength (default: enabled). '
                             'Instead of a single scalar, predicts spatially-varying strength '
                             'based on layer type, boundary proximity, and local noise level.')
    parser.add_argument('--no_adaptive_strength_map', dest='use_adaptive_strength_map', action='store_false',
                        help='Disable adaptive strength map (use legacy scalar adaptive_scale).')
    parser.add_argument('--strength_hidden_channels', type=int, default=32,
                        help='Hidden channels for strength predictor network.')
    parser.add_argument('--strength_boundary_sigma', type=float, default=10.0,
                        help='Softness of boundary proximity influence (pixels). '
                             'Higher = smoother transition, lower = sharper boundary awareness.')
    parser.add_argument('--strength_init_bias', type=float, default=0.0,
                        help='Initial bias for strength predictor output. TMI v4.1: Changed from -1.0 to 0.0. '
                             'sigmoid(-1.0) ≈ 0.27, so layer heads start with ~27%% contribution. '
                             'Use more negative for safer start (e.g., -2.0 gives ~12%%).')

    # TMI v4.1: Strength Map Learning (CRITICAL for adaptive denoising)
    parser.add_argument('--lambda_strength_map', type=float, default=1.0,
                        help='Weight for oracle-guided strength map loss. '
                             'CRITICAL: Without this, strength predictor never learns!')
    parser.add_argument('--lambda_strength_reg', type=float, default=0.1,
                        help='Weight for strength map regularization (smoothness + entropy).')
    parser.add_argument('--use_strength_curriculum', action='store_true',
                        help='Enable curriculum learning for strength map (warmup then learn).')
    parser.add_argument('--strength_warmup_epochs', type=int, default=2,
                        help='Number of epochs to warmup with fixed strength before learning.')

    # Clinical losses
    parser.add_argument('--use_clinical_loss', action='store_true')
    parser.add_argument('--lambda_clinical_l1', type=float, default=1.0)
    parser.add_argument('--lambda_layer_ssim', type=float, default=0.5)
    parser.add_argument('--lambda_boundary_sharp', type=float, default=0.3)
    parser.add_argument('--lambda_texture', type=float, default=0.1)
    parser.add_argument('--clinical_weight_rnfl_gcl', type=float, default=2.0)
    parser.add_argument('--clinical_weight_inl_opl_onl', type=float, default=2.0,
                        help='Weight multiplier for INL_OPL_ONL (default 2.0, combined with inverse_freq=10 gives 20x)')
    parser.add_argument('--clinical_weight_is_os', type=float, default=2.0)
    parser.add_argument('--clinical_weight_rpe_choroid', type=float, default=1.5)

    # Other losses
    parser.add_argument('--boundary_weight', type=float, default=0.0)  # Disabled by default - boundary_head is unreliable
    parser.add_argument('--pos_weight', type=float, default=10.0)
    parser.add_argument('--boundary_attention_weight', type=float, default=0.5)
    parser.add_argument('--clinical_attention_weight', type=float, default=0.5)

    # Neuro-Symbolic Losses (KEY TMI CONTRIBUTION)
    parser.add_argument('--use_neuro_symbolic', action='store_true',
                        help='Enable full neuro-symbolic loss suite')
    parser.add_argument('--lambda_symbolic_ordering', type=float, default=0.35,
                        help='Weight for layer ordering (increased from 0.1 to prevent thin layer collapse)')
    parser.add_argument('--lambda_symbolic_thickness', type=float, default=0.05,
                        help='Weight for thickness constraint loss')
    parser.add_argument('--lambda_symbolic_intensity', type=float, default=0.05,
                        help='Weight for intensity order constraint')
    parser.add_argument('--lambda_symbolic_continuity', type=float, default=0.05,
                        help='Weight for boundary continuity constraint')
    parser.add_argument('--lambda_symbolic_anatomy', type=float, default=0.02,
                        help='Weight for anatomy template constraint')
    parser.add_argument('--lambda_head_diversity', type=float, default=0.1,
                        help='Weight for head diversity loss (prevents head collapse)')
    parser.add_argument('--lambda_layer_supervision', type=float, default=0.5,
                        help='Weight for per-layer direct supervision loss')

    # Training
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--noise_levels', type=str, default='0.80,0.90,1.00,1.10,1.20')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='outputs/tmi_enhanced')

    # Memory-safe options
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Accumulate gradients over N steps (effective batch = batch_size * N)')
    parser.add_argument('--memory_limit_mb', type=int, default=3500,
                        help='Stop training if memory exceeds this limit (MB)')
    parser.add_argument('--aggressive_gc', action='store_true',
                        help='Run garbage collection after each batch')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers (0 for memory safety)')
    parser.add_argument('--center_focused', action='store_true',
                        help='Use center-focused cropping to ensure IS/OS is in every patch (legacy)')
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified sampling: 50%% IS/OS, 25%% RNFL/GCL, 25%% RPE/Choroid')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to reduce memory (slower but fits larger patches)')

    # Stage-wise and alternating training to prevent segmentation collapse
    parser.add_argument('--seg_only_epochs', type=int, default=0,
                        help='Train segmentation only for first N epochs (freeze denoiser)')
    parser.add_argument('--alternating_epochs', type=int, default=0,
                        help='Alternate training every N epochs (seg then denoising)')

    # Learning without Forgetting (LwF) - Continual Learning (ENABLED BY DEFAULT)
    parser.add_argument('--teacher_ckpt', type=str, default='checkpoints/seg_4class_improved_best.pth',
                        help='Path to teacher segmenter checkpoint for LwF (auto-detected by default)')
    parser.add_argument('--no_lwf', action='store_true',
                        help='Disable Learning without Forgetting (continual learning)')
    parser.add_argument('--lambda_kd', type=float, default=2.0,
                        help='Weight for knowledge distillation loss (default: 2.0)')
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                        help='Temperature for KD soft targets (default: 4.0, higher=softer)')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(',')]

    # Create datasets
    train_dataset = TMITrainDataset(
        args.train_jsonl,
        max_samples=args.max_train,
        noise_levels=noise_levels,
        patch_size=args.patch_size,
        stratified=args.stratified,
        center_focused=args.center_focused,
    )

    if args.stratified:
        print("Using STRATIFIED sampling: 50% IS/OS, 25% RNFL/GCL, 25% RPE/Choroid")
    elif args.center_focused:
        print("Using CENTER-FOCUSED cropping - IS/OS will be in every patch")

    val_dataset = TMIValDataset(
        args.val_jsonl,
        max_samples=args.max_val,
        patch_size=args.patch_size,
        patches_per_image=3,  # IS/OS, top, bottom patches for full layer coverage
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False if args.device == 'cpu' else True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, 2),  # Respect memory settings
    )

    n_val_images = len(val_dataset) // 3  # 3 patches per image
    print(f"Train samples: {len(train_dataset)}, Val: {n_val_images} images x 3 patches = {len(val_dataset)}")

    # Create model
    # When dsp_only is enabled, disable segmentation-based boundary detection
    # DSP will handle all boundary detection (more memory efficient, CPU-friendly)
    use_hybrid_segmenter = args.use_hybrid_segmenter
    use_boundary_regression = args.use_boundary_regression
    if args.dsp_only:
        print("\n" + "="*60)
        print("★ DSP-ONLY MODE ENABLED ★")
        print("="*60)
        print("  • Segmenter: DISABLED (no CNN segmentation head)")
        print("  • Boundaries: DSP shortest path algorithm")
        print("  • Segmentation: Derived from DSP boundaries")
        print("  • Anatomical validity: 100% GUARANTEED")
        print("  • Training signal: DSP boundary loss only (no seg CE)")
        print("="*60 + "\n")
        # Force DSP to be enabled
        args.use_dsp_boundaries = True
        use_hybrid_segmenter = False  # Disable Gaussian heatmap boundaries
        use_boundary_regression = False  # Disable legacy boundary regression

    model = TMIEnhancedModel(
        num_classes=NUM_CLASSES,
        nafnet_width=args.nafnet_width,
        use_layer_specific_heads=args.use_layer_specific_heads,
        use_seg_guided_attention=args.use_seg_guided_attention,
        use_columnar_attention=args.use_columnar_attention,
        use_boundary_regression=use_boundary_regression,
        use_hybrid_segmenter=use_hybrid_segmenter,
        use_dsp_boundaries=args.use_dsp_boundaries,  # NEW v3: Differentiable Shortest Path
        dsp_only=args.dsp_only,  # NEW v3.2: DSP-only mode (no segmenter)
        num_head_blocks=args.num_head_blocks,
        head_hidden_channels=args.head_hidden_channels,
        head_dropout=args.head_dropout,
        columnar_dim=args.columnar_dim,
        num_columnar_blocks=args.num_columnar_blocks,
        dsp_smoothness_weight=args.dsp_smoothness_weight,  # NEW v3
        dsp_temperature=args.dsp_temperature,  # NEW v3
        dsp_min_gap=args.dsp_min_gap,  # NEW v3
        # NEW v3.1: Continual learning with adapters
        use_continual_learning=args.continual_learning,
        adapter_bottleneck_ratio=args.adapter_bottleneck_ratio,
        adapter_dropout=args.adapter_dropout,
        initial_domain=args.domain_name,
        # NEW v3.3: Physics-Informed Fresnel Gradient Matching
        use_fresnel_physics=args.use_fresnel_physics,
        fresnel_physics_weight=args.fresnel_physics_weight,
        fresnel_learnable=args.fresnel_learnable,
        # NEW v4: Adaptive Strength Map
        use_adaptive_strength_map=args.use_adaptive_strength_map,
        strength_hidden_channels=args.strength_hidden_channels,
        strength_boundary_sigma=args.strength_boundary_sigma,
        strength_init_bias=args.strength_init_bias,
    ).to(device)

    # Print configuration summary for enhanced segmenter
    print(f"\n{'='*60}")
    print("ENHANCED SEGMENTER CONFIGURATION (4-Boundary System)")
    print(f"{'='*60}")
    print(f"  Layer-Specific Heads:    {'✓ ENABLED' if args.use_layer_specific_heads else '✗ Disabled'}")
    print(f"  Seg-Guided Attention:    {'✓ ENABLED' if args.use_seg_guided_attention else '✗ Disabled'}")
    print(f"  Columnar Attention:      {'✓ ENABLED' if args.use_columnar_attention else '✗ Disabled'}")
    print(f"  Hybrid Segmenter (4-bd): {'✓ ENABLED' if use_hybrid_segmenter else '✗ Disabled'}")
    print(f"  Legacy Boundary Reg:     {'✓ ENABLED' if use_boundary_regression else '✗ Disabled'}")
    # NEW v3: DSP boundary detection
    dsp_status = '✓ ENABLED' if args.use_dsp_boundaries else '✗ Disabled'
    if args.dsp_only:
        dsp_status += ' (DSP-ONLY MODE)'
    print(f"  DSP Boundaries (v3):     {dsp_status}")

    # NEW v3.3: Physics-Informed Fresnel Gradient Matching
    if args.use_fresnel_physics:
        fresnel_status = f'✓ ENABLED (weight: {args.fresnel_physics_weight:.2f}'
        fresnel_status += ', learnable)' if args.fresnel_learnable else ', fixed)'
        print(f"  Fresnel Physics (v3.3):  {fresnel_status}")
        print(f"    Expected gradient strengths based on refractive indices:")
        print(f"    - ILM (vitreous→RNFL):    Strong (n: 1.336→1.358)")
        print(f"    - RNFL/INL:               Weak   (n: 1.358→1.365)")
        print(f"    - INL/IS-OS:              Medium (n: 1.365→1.392)")
        print(f"    - IS-OS/RPE:              Strong (n: 1.392→1.400)")
    else:
        print(f"  Fresnel Physics (v3.3):  ✗ Disabled")

    # NEW v4: Adaptive Strength Map status
    if args.use_adaptive_strength_map:
        strength_init = 1 / (1 + np.exp(-args.strength_init_bias))  # sigmoid
        print(f"  Adaptive Strength (v4):  ✓ ENABLED (init: {strength_init:.1%})")
        print(f"    Spatially-varying denoising strength per pixel")
        print(f"    - Boundary sigma: {args.strength_boundary_sigma:.1f}px (softer near boundaries)")
        print(f"    - Initial contribution: ~{strength_init:.1%} (learns during training)")
    else:
        print(f"  Adaptive Strength (v4):  ✗ Disabled (using scalar adaptive_scale)")

    # LwF status
    lwf_enabled = not args.no_lwf and not args.dsp_only and args.teacher_ckpt
    lwf_available = lwf_enabled and os.path.exists(args.teacher_ckpt) if args.teacher_ckpt else False
    if args.no_lwf:
        lwf_status = "✗ Disabled (--no_lwf)"
    elif args.dsp_only:
        lwf_status = "- N/A (DSP-only mode, no segmentation)"
    elif lwf_available:
        lwf_status = "✓ ENABLED (auto)"
    elif args.teacher_ckpt:
        lwf_status = f"⚠ Pending (checkpoint not found)"
    else:
        lwf_status = "✗ Disabled (no teacher)"
    print(f"  Continual Learning (LwF): {lwf_status}")

    # NEW v3.1: Adapter-based continual learning status
    if args.continual_learning:
        if args.adapt_to_domain:
            adapter_status = f"✓ ADAPTING ({args.domain_name} → {args.adapt_to_domain})"
        else:
            adapter_status = f"✓ ENABLED (domain: {args.domain_name})"
    else:
        adapter_status = "✗ Disabled"
    print(f"  Adapter Continual Learn: {adapter_status}")

    print(f"\n  Boundaries: ILM, RNFL/INL, INL/IS_OS (critical), IS_OS/RPE (critical)")
    print(f"  Layer Supervision λ:     {getattr(args, 'lambda_layer_supervision', 0.5)}")
    if use_hybrid_segmenter:
        print(f"  Gaussian Boundary λ:     {getattr(args, 'lambda_gaussian_boundary', 4.0)}")
    if args.use_dsp_boundaries:
        if args.dsp_only:
            print(f"  DSP Mode:                ★ DSP-ONLY (no segmenter, 100% anatomical validity)")
        else:
            print(f"  DSP Mode:                Hybrid (DSP + Segmenter)")
        print(f"  DSP Boundary λ:          {args.lambda_dsp_boundary}")
        print(f"  DSP Smoothness:          {args.dsp_smoothness_weight}")
        print(f"  DSP Temperature:         {args.dsp_temperature}")
        print(f"  DSP Min Gap:             {args.dsp_min_gap} px")
        if args.dsp_warmup_epochs > 0:
            print(f"  DSP Warmup Epochs:       {args.dsp_warmup_epochs}")
            print(f"  DSP Warmup λ:            {args.dsp_warmup_lambda} (then → {args.lambda_dsp_boundary})")
    if lwf_available:
        print(f"  KD Weight (λ_kd):        {args.lambda_kd}")
        print(f"  KD Temperature:          {args.kd_temperature}")
    print(f"{'='*60}\n")

    # Enable gradient checkpointing if requested (reduces memory, slower training)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Load checkpoint if provided
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading checkpoint: {args.resume_from}")
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)

        # IMPORTANT: If freeze_segmenter is set, reload segmenter weights from pretrained checkpoint
        # This ensures consistent segmentation even when resuming from a checkpoint that may have
        # modified segmenter weights during training
        if args.freeze_segmenter and args.seg_ckpt and os.path.exists(args.seg_ckpt):
            print(f"Reloading pretrained segmenter (freeze_segmenter=True): {args.seg_ckpt}")
            seg_state = torch.load(args.seg_ckpt, map_location=device, weights_only=False)
            if 'model_state_dict' in seg_state:
                seg_state = seg_state['model_state_dict']
            seg_weights = {}
            for k, v in seg_state.items():
                if 'boundary_head' in k:
                    continue  # Skip boundary_head
                if k.startswith('segmenter.'):
                    seg_weights[k] = v
                elif not k.startswith(('nafnet.', 'layer_heads.', 'refinement.')):
                    seg_weights[f'segmenter.{k}'] = v
            if seg_weights:
                model.load_state_dict(seg_weights, strict=False)
                print(f"  Reloaded {len(seg_weights)} segmenter weights")
    else:
        # Load NAFNet backbone separately if provided
        if args.nafnet_ckpt and os.path.exists(args.nafnet_ckpt):
            print(f"Loading NAFNet backbone: {args.nafnet_ckpt}")
            nafnet_state = torch.load(args.nafnet_ckpt, map_location=device, weights_only=False)
            # Handle different checkpoint formats
            if 'state_dict' in nafnet_state:
                nafnet_state = nafnet_state['state_dict']
            elif 'model_state_dict' in nafnet_state:
                nafnet_state = nafnet_state['model_state_dict']
            # Map checkpoint keys to model keys
            # Checkpoint may have: backbone.* -> model expects: nafnet.*
            nafnet_weights = {}
            other_weights = {}
            for k, v in nafnet_state.items():
                if k.startswith('nafnet.'):
                    nafnet_weights[k] = v
                elif k.startswith('backbone.'):
                    # Map backbone.* to nafnet.*
                    new_key = 'nafnet.' + k[len('backbone.'):]
                    nafnet_weights[new_key] = v
                elif k.startswith('layer_heads.'):
                    # Also load layer_heads weights for denoising
                    other_weights[k] = v
                elif k.startswith('seg_attention.'):
                    # Also load seg_attention weights
                    other_weights[k] = v
                elif k.startswith('strength_predictor.'):
                    # NEW v4: Load adaptive strength predictor weights
                    other_weights[k] = v
                elif k in ['adaptive_scale', 'feature_proj.weight', 'feature_proj.bias']:
                    # Load other model weights
                    other_weights[k] = v
                elif k.startswith('segmenter.') and not args.dsp_only:
                    # Only load segmenter if not in DSP-only mode
                    other_weights[k] = v
                elif not any(k.startswith(p) for p in ['segmenter.', 'refinement.', 'feature_extractor.', 'refinement_scale', 'confidence']):
                    # Add nafnet. prefix for other NAFNet weights
                    nafnet_weights[f'nafnet.{k}'] = v
            model.load_state_dict(nafnet_weights, strict=False)
            print(f"  Loaded {len(nafnet_weights)} NAFNet weights")
            if other_weights:
                model.load_state_dict(other_weights, strict=False)
                print(f"  Loaded {len(other_weights)} additional weights (layer_heads, seg_attention, etc.)")

        # Load V4 segmentation checkpoint separately if provided
        if args.seg_ckpt and os.path.exists(args.seg_ckpt):
            print(f"Loading V4 segmentation: {args.seg_ckpt}")
            seg_state = torch.load(args.seg_ckpt, map_location=device, weights_only=False)
            if 'model_state_dict' in seg_state:
                seg_state = seg_state['model_state_dict']
            # Map V4 weights to current segmenter
            # NOTE: Skip boundary_head weights as they predict wrong positions
            # The boundary_head will be trained from scratch
            seg_weights = {}
            skipped_boundary = 0
            for k, v in seg_state.items():
                # Skip boundary_head weights to reinitialize
                if 'boundary_head' in k:
                    skipped_boundary += 1
                    continue
                if k.startswith('segmenter.'):
                    seg_weights[k] = v
                elif not k.startswith(('nafnet.', 'layer_heads.', 'refinement.')):
                    # Try mapping without prefix
                    seg_weights[f'segmenter.{k}'] = v
            if seg_weights:
                model.load_state_dict(seg_weights, strict=False)
                print(f"  Loaded {len(seg_weights)} segmentation weights (skipped {skipped_boundary} boundary_head weights)")

    # Freeze backbone initially if requested
    if args.freeze_backbone_initially:
        print("Freezing NAFNet backbone initially")
        for param in model.nafnet.parameters():
            param.requires_grad = False

    # Freeze segmenter to prevent collapse from competing losses
    if args.freeze_segmenter and model.segmenter is not None:
        print("Freezing segmenter (using pretrained weights as guidance only)")
        for param in model.segmenter.parameters():
            param.requires_grad = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create loss function with 4-class clinical weights
    # IMPORTANT: Include inverse frequency balancing to handle class imbalance
    # Typical class distribution: RNFL_GCL ~45%, INL_OPL_ONL ~8%, IS_OS ~22%, RPE_Choroid ~25%
    # Inverse frequency weights based on corrected 4-class distribution:
    # RNFL_GCL: 44%, INL_OPL_ONL: 7.5% (~3.35% in patches), IS_OS: 14%, RPE_Choroid: 34%
    # FIX: INL_OPL_ONL needs ~20x weight to prevent layer collapse (was 6.0)
    inverse_freq_weights = {
        0: 1.0,   # RNFL_GCL - most common (44%), baseline
        1: 10.0,  # INL_OPL_ONL - INCREASED (was 6.0) - thin layer needs very high weight
        2: 3.0,   # IS_OS - moderate frequency (14%), clinically critical
        3: 1.3,   # RPE_Choroid - common (34%)
    }
    clinical_weights = {
        0: args.clinical_weight_rnfl_gcl * inverse_freq_weights[0],
        1: args.clinical_weight_inl_opl_onl * inverse_freq_weights[1],  # Boosted!
        2: args.clinical_weight_is_os * inverse_freq_weights[2],
        3: args.clinical_weight_rpe_choroid * inverse_freq_weights[3],
    }
    print(f"\n=== Class Weights (with inverse frequency balancing) ===")
    print(f"  RNFL_GCL:     {clinical_weights[0]:.1f}")
    print(f"  INL_OPL_ONL:  {clinical_weights[1]:.1f} (boosted for rare class)")
    print(f"  IS_OS:        {clinical_weights[2]:.1f}")
    print(f"  RPE_Choroid:  {clinical_weights[3]:.1f}")

    # In DSP-only mode, skip segmentation CE loss (use DSP boundary loss instead)
    dsp_only_seg_lambda = 0.0 if args.dsp_only else 15.0
    if args.dsp_only:
        print("[DSP-ONLY] Segmentation CE loss DISABLED - using DSP boundary loss only")

    loss_fn = TMIJointLoss(
        num_classes=NUM_CLASSES,
        clinical_weights=clinical_weights if args.use_clinical_loss else None,
        lambda_l1=args.lambda_clinical_l1,
        lambda_ssim=args.lambda_layer_ssim,
        lambda_boundary=args.lambda_boundary_sharp,
        lambda_seg=dsp_only_seg_lambda,  # 0 in DSP-only, 15 otherwise
        lambda_texture=args.lambda_texture,
        lambda_boundary_det=0.0,  # No separate boundary_head with 4-class
    ).to(device)

    # Create Neuro-Symbolic Loss (KEY TMI CONTRIBUTION)
    ns_loss_fn = None
    if args.use_neuro_symbolic:
        print("\n=== NEURO-SYMBOLIC LOSS ENABLED ===")
        print(f"  Ordering weight:   {args.lambda_symbolic_ordering}")
        print(f"  Thickness weight:  {args.lambda_symbolic_thickness}")
        print(f"  Intensity weight:  {args.lambda_symbolic_intensity}")
        print(f"  Continuity weight: {args.lambda_symbolic_continuity}")
        print(f"  Anatomy weight:    {args.lambda_symbolic_anatomy}")
        ns_loss_fn = NeuroSymbolicLoss(
            num_classes=NUM_CLASSES,
            lambda_ordering=args.lambda_symbolic_ordering,
            lambda_thickness=args.lambda_symbolic_thickness,
            lambda_intensity=args.lambda_symbolic_intensity,
            lambda_continuity=args.lambda_symbolic_continuity,
            lambda_anatomy=args.lambda_symbolic_anatomy,
        ).to(device)

    # Create boundary loss functions (moved from train_epoch to avoid memory leak)
    # NOTE: Use local variables (not args) to respect dsp_only mode
    boundary_loss_fn = None
    gaussian_boundary_loss_fn = None

    if use_boundary_regression:  # Use local var (disabled in dsp_only mode)
        boundary_loss_fn = BoundaryLoss(
            num_boundaries=5,  # 5 boundaries for 4 classes
            lambda_smooth=0.1,
            lambda_order=0.5,
            lambda_thickness=0.1,
        ).to(device)
        print("Created BoundaryLoss for boundary regression")

    if use_hybrid_segmenter:  # Use local var (disabled in dsp_only mode)
        gaussian_boundary_loss_fn = GaussianBoundaryLoss(
            num_boundaries=4,  # Updated: 4 boundaries (ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE)
            is_os_weight=1.5,  # Reduced from 3.0 to prevent IS/OS domination
            sigma=2.0,
        ).to(device)
        print("Created GaussianBoundaryLoss for hybrid segmenter (4 boundaries, is_os_weight=1.5)")

    # NEW TMI v3: Differentiable Shortest Path (DSP) loss
    dsp_loss_fn = None
    if args.use_dsp_boundaries:
        # Check if learnable weights are enabled
        learnable_dsp_weights = getattr(args, 'learnable_dsp_weights', True)

        dsp_loss_fn = DSPBoundaryLoss(
            num_boundaries=4,  # 4 boundaries for 4-class segmentation
            lambda_position=1.0,
            lambda_cost=0.5,
            lambda_ordering=0.1,
            lambda_smoothness=0.5,
            boundary_weights=[1.0, 1.5, 3.0, 2.5],  # Initial weights (clinical prior)
            learnable_weights=learnable_dsp_weights,  # NEW: Learnable boundary weights
        ).to(device)
        print("Created DSPBoundaryLoss for Differentiable Shortest Path boundary detection")
        print("  - Memory efficient: O(H×W) instead of O(H×W×C²)")
        print("  - CPU-friendly: no attention matrices")
        print("  - Guaranteed layer ordering via dynamic programming")
        if learnable_dsp_weights:
            print("  - LEARNABLE boundary weights (auto-tuned during training)")

    # =========================================================================
    # TMI v3.3: Physics-Informed Denoising Losses (Rayleigh + Layer Intensity)
    # =========================================================================
    rayleigh_loss_fn = None
    use_rayleigh = getattr(args, 'use_rayleigh_loss', False) or getattr(args, 'use_physics_denoising_loss', False)
    use_layer_intensity = getattr(args, 'use_layer_intensity_loss', False) or getattr(args, 'use_physics_denoising_loss', False)

    if use_rayleigh or use_layer_intensity:
        print("\n=== PHYSICS-INFORMED DENOISING LOSSES ENABLED ===")

        if use_rayleigh:
            rayleigh_mode = getattr(args, 'rayleigh_mode', 'amplitude')
            lambda_rayleigh = getattr(args, 'lambda_rayleigh', 0.5)
            rayleigh_loss_fn = RayleighLikelihoodLoss(
                mode=rayleigh_mode,
                noise_estimation='heteroscedastic',
                use_layer_aware=True,
            ).to(device)
            print(f"  Rayleigh Likelihood Loss: mode={rayleigh_mode}, lambda={lambda_rayleigh}")
            print("    - Proper OCT speckle model (not Gaussian)")
            print("    - Heteroscedastic noise estimation (spatially varying)")
            print("    - Layer-aware noise multipliers")

        if use_layer_intensity:
            lambda_layer_int = getattr(args, 'lambda_layer_intensity', 0.2)
            print(f"  Layer Intensity Consistency Loss: lambda={lambda_layer_int}")
            print("    - Enforces physically plausible layer intensities")
            print("    - RPE > RNFL > IS_OS > INL intensity ordering")

    # =========================================================================
    # TMI v3.3: Interferometric Consistency Loss (boundary validation)
    # =========================================================================
    ic_loss_fn = None
    use_interferometric = getattr(args, 'use_interferometric_loss', False)

    if use_interferometric:
        ic_use_fresnel = getattr(args, 'ic_use_fresnel_prior', True)
        lambda_ic = getattr(args, 'lambda_interferometric', 0.3)
        ic_loss_fn = InterferometricConsistencyLoss(
            num_boundaries=4,
            use_multiscale=True,
            gradient_scales=[1, 2, 4],
            use_fresnel_prior=ic_use_fresnel,
            lateral_window=5,
        ).to(device)
        print("\n=== INTERFEROMETRIC CONSISTENCY LOSS ENABLED ===")
        print(f"  Lambda: {lambda_ic}")
        print(f"  Fresnel prior: {ic_use_fresnel}")
        print("  Components:")
        print("    - Gradient magnitude consistency (denoised vs clean)")
        print("    - Lateral boundary smoothness")
        print("    - Gradient direction consistency")
        if ic_use_fresnel:
            print("    - Fresnel-based expected gradient magnitudes")

    # Learning without Forgetting (LwF) - Load teacher model for knowledge distillation
    teacher_segmenter = None
    kd_loss_fn = None

    if args.no_lwf:
        print(f"\n=== LEARNING WITHOUT FORGETTING (LwF) DISABLED ===")
        print("  LwF explicitly disabled via --no_lwf flag")
        print("  Training without knowledge distillation from teacher model")
    elif args.dsp_only:
        # DSP-only mode: LWF is not applicable (no segmentation)
        print(f"\n=== LEARNING WITHOUT FORGETTING (LwF) NOT APPLICABLE ===")
        print("  DSP-only mode: No segmentation network to preserve")
        print("  Boundary detection is entirely via DSP (no knowledge distillation needed)")
    elif args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        print(f"\n=== LEARNING WITHOUT FORGETTING (LwF) ENABLED ===")
        print(f"  Teacher checkpoint: {args.teacher_ckpt}")
        print(f"  KD weight (lambda_kd): {args.lambda_kd}")
        print(f"  KD temperature: {args.kd_temperature}")

        teacher_segmenter = load_teacher_segmenter(
            args.teacher_ckpt, device, num_classes=NUM_CLASSES
        )

        kd_loss_fn = KnowledgeDistillationLoss(
            temperature=args.kd_temperature,
            alpha=0.5,  # Balance between KD and hard labels
        ).to(device)

        print("  LwF will preserve Stage 1 segmentation knowledge during joint training")
    elif args.teacher_ckpt:
        print(f"\n[WARNING] Teacher checkpoint not found: {args.teacher_ckpt}")
        print(f"  Expected: {args.teacher_ckpt}")
        print("  LwF disabled - training without knowledge distillation")
        print("  (Run Stage 1 segmentation training first to create teacher checkpoint)")

    # =========================================================================
    # TMI v3.1: Adapter-based Continual Learning Setup
    # =========================================================================
    is_domain_adaptation = args.continual_learning and args.adapt_to_domain

    if args.continual_learning:
        print(f"\n[Continual Learning Setup]")

        # Load pre-trained adapters if provided
        if args.load_adapters_from and os.path.exists(args.load_adapters_from):
            model.load_adapters(args.load_adapters_from)
            print(f"  Loaded adapters from: {args.load_adapters_from}")

        # Domain adaptation mode: adapting to a new domain
        if args.adapt_to_domain:
            print(f"  Mode: DOMAIN ADAPTATION")
            print(f"  Source domain: {args.domain_name}")
            print(f"  Target domain: {args.adapt_to_domain}")

            # Register the new domain
            model.register_domain(args.adapt_to_domain)
            model.set_active_domain(args.adapt_to_domain)

            # Freeze backbone, only train new domain's adapters
            if args.freeze_backbone:
                model.freeze_for_adaptation()
                print(f"  Backbone: FROZEN (only adapters are trainable)")
        else:
            print(f"  Mode: INITIAL DOMAIN TRAINING")
            print(f"  Domain: {args.domain_name}")
            print(f"  All parameters are trainable")

    # Collect learnable parameters from loss functions
    loss_fn_params = []
    if dsp_loss_fn is not None and getattr(args, 'learnable_dsp_weights', True):
        loss_fn_params.extend(list(dsp_loss_fn.parameters()))
        if loss_fn_params:
            print(f"\n  DSP loss has {len(loss_fn_params)} learnable parameters (boundary weights)")

    # Optimizer - use adapter params only during domain adaptation
    if is_domain_adaptation and args.freeze_backbone:
        # Only optimize adapter parameters
        adapter_params = list(model.get_adapter_params())
        # Also add loss function params if any
        all_params = adapter_params + loss_fn_params
        print(f"\n  Optimizing {len(adapter_params)} adapter parameter groups + {len(loss_fn_params)} loss params")
        optimizer = torch.optim.AdamW(
            all_params,
            lr=args.lr,
            weight_decay=1e-4,
        )
    else:
        # Standard optimizer for all trainable parameters
        # Combine model params with loss function params
        model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        all_params = model_params + loss_fn_params
        optimizer = torch.optim.AdamW(
            all_params,
            lr=args.lr,
            weight_decay=1e-4,
        )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Training loop
    best_psnr = 0
    unfreeze_epoch = args.epochs // 3  # Unfreeze after 1/3 of training

    # TMI v4.1: Initialize curriculum scheduler for strength map learning
    strength_curriculum = None
    if getattr(args, 'use_strength_curriculum', False) and args.use_adaptive_strength_map:
        strength_curriculum = StrengthCurriculumScheduler(
            warmup_epochs=getattr(args, 'strength_warmup_epochs', 2),
            initial_strength=0.5,
        )
        print(f"[TMI v4.1] Strength curriculum enabled: {args.strength_warmup_epochs} warmup epochs")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

        # TMI v4.1: Apply curriculum for strength map learning
        lambda_overrides = {}
        if strength_curriculum is not None:
            curriculum_config = strength_curriculum.apply_to_model(model, epoch)
            lambda_overrides['lambda_strength_map'] = curriculum_config['lambda_strength_map']
            if epoch < strength_curriculum.warmup_epochs:
                print(f"  [Curriculum] Warmup: fixed strength={curriculum_config['fixed_strength']:.2f}, lambda_strength_map=0")
            elif epoch == strength_curriculum.warmup_epochs:
                print(f"  [Curriculum] Learning: strength predictor now trainable, lambda_strength_map={curriculum_config['lambda_strength_map']}")

        # Stage-wise training: Train segmentation only for first N epochs
        if args.seg_only_epochs > 0:
            if epoch < args.seg_only_epochs:
                # Freeze denoiser (NAFNet and layer heads), train only segmenter
                for param in model.nafnet.parameters():
                    param.requires_grad = False
                if hasattr(model, 'layer_heads') and model.layer_heads is not None:
                    for param in model.layer_heads.parameters():
                        param.requires_grad = False
                if model.segmenter is not None:
                    for param in model.segmenter.parameters():
                        param.requires_grad = True
                if epoch == 0:
                    print(f"  Stage-wise: Training SEGMENTATION ONLY for epochs 1-{args.seg_only_epochs}")
            elif epoch == args.seg_only_epochs:
                # Unfreeze denoiser for joint training
                print(f"  Stage-wise: Now starting JOINT TRAINING (epoch {epoch+1})")
                for param in model.parameters():
                    param.requires_grad = True

        # Alternating training: Alternate between seg and denoising every N epochs
        if args.alternating_epochs > 0:
            phase = (epoch // args.alternating_epochs) % 2
            if phase == 0:
                # Segmentation phase: freeze denoiser
                for param in model.nafnet.parameters():
                    param.requires_grad = False
                if hasattr(model, 'layer_heads') and model.layer_heads is not None:
                    for param in model.layer_heads.parameters():
                        param.requires_grad = False
                if model.segmenter is not None:
                    for param in model.segmenter.parameters():
                        param.requires_grad = True
                if epoch % args.alternating_epochs == 0:
                    print(f"  Alternating: SEGMENTATION phase (epochs {epoch+1}-{epoch+args.alternating_epochs})")
            else:
                # Denoising phase: freeze segmenter
                for param in model.nafnet.parameters():
                    param.requires_grad = True
                if hasattr(model, 'layer_heads') and model.layer_heads is not None:
                    for param in model.layer_heads.parameters():
                        param.requires_grad = True
                if model.segmenter is not None:
                    for param in model.segmenter.parameters():
                        param.requires_grad = False
                if epoch % args.alternating_epochs == 0:
                    print(f"  Alternating: DENOISING phase (epochs {epoch+1}-{epoch+args.alternating_epochs})")

        # Unfreeze backbone after initial epochs (only if not using permanent freeze)
        # --freeze_backbone keeps backbone frozen forever
        # --freeze_backbone_initially unfreezes after 1/3 of training
        permanent_freeze = getattr(args, 'freeze_backbone', False)
        if args.freeze_backbone_initially and not permanent_freeze and epoch == unfreeze_epoch:
            print("Unfreezing NAFNet backbone")
            for param in model.nafnet.parameters():
                param.requires_grad = True
            # Update optimizer to include backbone params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,  # Lower LR for fine-tuning
                weight_decay=1e-4,
            )

        # =====================================================================
        # DSP Warmup: Train with higher DSP weight for first N epochs
        # This helps DSP learn good boundary patterns before joint optimization
        # =====================================================================
        lambda_overrides = None
        dsp_warmup_epochs = getattr(args, 'dsp_warmup_epochs', 0)

        if dsp_warmup_epochs > 0 and epoch < dsp_warmup_epochs:
            # DSP warmup phase: higher DSP weight, optionally lower seg weight
            lambda_overrides = {
                'lambda_dsp': getattr(args, 'dsp_warmup_lambda', 5.0),
            }
            if epoch == 0:
                print(f"  DSP Warmup: Training with λ_dsp={lambda_overrides['lambda_dsp']:.1f} for epochs 1-{dsp_warmup_epochs}")
                print(f"              (Normal λ_dsp={args.lambda_dsp_boundary:.1f} will be used after warmup)")
        elif dsp_warmup_epochs > 0 and epoch == dsp_warmup_epochs:
            print(f"  DSP Warmup COMPLETE: Switching to normal λ_dsp={args.lambda_dsp_boundary:.1f}")

        # Train
        train_loss, train_psnr, train_div, memory_exceeded = train_epoch(
            model, train_loader, optimizer, loss_fn, device, args,
            ns_loss_fn=ns_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            gaussian_boundary_loss_fn=gaussian_boundary_loss_fn,
            teacher_segmenter=teacher_segmenter,
            kd_loss_fn=kd_loss_fn,
            dsp_loss_fn=dsp_loss_fn,  # NEW v3: DSP boundary loss
            rayleigh_loss_fn=rayleigh_loss_fn,  # TMI v3.3: Physics-informed denoising
            ic_loss_fn=ic_loss_fn,  # TMI v3.3: Interferometric consistency
            epoch=epoch,
            lambda_overrides=lambda_overrides,
        )
        print(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB, Diversity Loss: {train_div:.4f}")

        # Check if memory limit was exceeded
        if memory_exceeded:
            print("\n" + "="*80)
            print("⚠️ MEMORY LIMIT EXCEEDED - Saving checkpoint and stopping training")
            print("="*80)
            emergency_path = os.path.join(args.output_dir, 'emergency_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, emergency_path)
            print(f"Emergency checkpoint saved to: {emergency_path}")
            print("Reduce nafnet_width or patch_size and resume from this checkpoint.")
            break

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, args, ns_loss_fn=ns_loss_fn)

        # Print global metrics with NAFNet baseline comparison
        nafnet_psnr = val_metrics.get('nafnet_psnr', val_metrics['psnr'])
        nafnet_ssim = val_metrics.get('nafnet_ssim', val_metrics['ssim'])
        adaptive_psnr_gain = val_metrics['psnr'] - nafnet_psnr
        adaptive_ssim_gain = val_metrics['ssim'] - nafnet_ssim

        # Get adaptive scale/strength for logging
        # NEW v4: When using adaptive strength map, report mean strength from validation outputs
        if hasattr(model, 'use_adaptive_strength_map') and model.use_adaptive_strength_map:
            # Strength is computed per-batch, so use validation avg or placeholder
            adaptive_scale_val = val_metrics.get('avg_strength', 0.0)
            strength_info = f"(adaptive map, mean={adaptive_scale_val:.2%})"
        elif hasattr(model, 'adaptive_scale'):
            adaptive_scale_val = model.adaptive_scale.item()
            strength_info = f"(scalar={adaptive_scale_val:.2f})"
        else:
            adaptive_scale_val = 0.0
            strength_info = "(none)"

        print(f"\n{'='*80}")
        print(f"VALIDATION RESULTS - Epoch {epoch}")
        print(f"{'='*80}")
        print(f"  Global Metrics:")
        print(f"    Loss: {val_metrics['loss']:.4f}")
        print(f"    Adaptive Scale: {adaptive_scale_val:.4f}")
        psnr_label = get_quality_label('psnr', val_metrics['psnr'])
        ssim_label = get_quality_label('ssim', val_metrics['ssim'])
        gain_label = get_quality_label('adaptive_gain', adaptive_psnr_gain)
        print(f"    PSNR:  NAFNet={nafnet_psnr:.2f}dB -> Adaptive={val_metrics['psnr']:.2f}dB {psnr_label} (gain: {adaptive_psnr_gain:+.2f}dB {gain_label})")
        print(f"    SSIM:  NAFNet={nafnet_ssim:.4f} -> Adaptive={val_metrics['ssim']:.4f} {ssim_label} (gain: {adaptive_ssim_gain:+.4f})")

        # Print per-layer metrics table
        print(f"\n  Per-Layer PSNR (dB):")
        print(f"    {'Layer':<15s} {'Noisy':>10s} {'NAFNet':>10s} {'Adaptive':>10s} {'vs Noisy':>10s} {'vs NAFNet':>10s}")
        print(f"    {'-'*65}")

        layer_names = CLASS_NAMES  # 4-class: ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
        total_gain_vs_noisy = 0
        total_gain_vs_nafnet = 0
        n_layers = 0

        for name in layer_names:
            noisy_psnr = val_metrics.get(f'{name}_noisy_psnr', 0)
            nafnet_psnr = val_metrics.get(f'{name}_nafnet_psnr', 0)
            denoised_psnr = val_metrics.get(f'{name}_denoised_psnr', 0)
            gain_vs_noisy = val_metrics.get(f'{name}_psnr_gain_vs_noisy', 0)
            gain_vs_nafnet = val_metrics.get(f'{name}_psnr_gain_vs_nafnet', 0)

            if noisy_psnr > 0:
                print(f"    {name:<15s} {noisy_psnr:>10.2f} {nafnet_psnr:>10.2f} {denoised_psnr:>10.2f} {gain_vs_noisy:>+10.2f} {gain_vs_nafnet:>+10.2f}")
                total_gain_vs_noisy += gain_vs_noisy
                total_gain_vs_nafnet += gain_vs_nafnet
                n_layers += 1

        if n_layers > 0:
            print(f"    {'-'*65}")
            print(f"    {'Average':<15s} {'':<10s} {'':<10s} {'':<10s} {total_gain_vs_noisy/n_layers:>+10.2f} {total_gain_vs_nafnet/n_layers:>+10.2f}")

        # Print per-layer SSIM table
        print(f"\n  Per-Layer SSIM:")
        print(f"    {'Layer':<15s} {'Noisy':>10s} {'NAFNet':>10s} {'Adaptive':>10s} {'vs Noisy':>10s} {'vs NAFNet':>10s}")
        print(f"    {'-'*65}")

        for name in layer_names:
            noisy_ssim = val_metrics.get(f'{name}_noisy_ssim', 0)
            nafnet_ssim = val_metrics.get(f'{name}_nafnet_ssim', 0)
            denoised_ssim = val_metrics.get(f'{name}_denoised_ssim', 0)
            gain_vs_noisy = val_metrics.get(f'{name}_ssim_gain_vs_noisy', 0)
            gain_vs_nafnet = val_metrics.get(f'{name}_ssim_gain_vs_nafnet', 0)

            if noisy_ssim > 0:
                print(f"    {name:<15s} {noisy_ssim:>10.4f} {nafnet_ssim:>10.4f} {denoised_ssim:>10.4f} {gain_vs_noisy:>+10.4f} {gain_vs_nafnet:>+10.4f}")

        # Print Dice scores with quality labels
        print(f"\n  Segmentation Dice Scores:")
        for name in layer_names:
            dice = val_metrics.get(f'{name}_dice', 0)
            dice_label = get_quality_label('dice', dice)
            print(f"    {name:<15s}: {dice:.4f} {dice_label}")

        # Print Clinical Metrics (TMI Publication) with quality labels
        print(f"\n  Clinical Metrics (TMI Publication):")
        print(f"    {'Metric':<25s} {'Value':>12s} {'Threshold':>12s} {'Status':>10s}")
        print(f"    {'-'*60}")

        # IS/OS Boundary MAE (using class 2 - IS_OS)
        if 'is_os_mae' in val_metrics:
            is_os_mae = val_metrics['is_os_mae']
            is_os_label = get_quality_label('is_os_mae', is_os_mae)
            print(f"    {'IS/OS Boundary MAE':<25s} {is_os_mae:>10.2f} px {'<3.0 px':>12s} {is_os_label:>10s}")

        # RNFL Thickness Error
        if 'rnfl_thickness_error' in val_metrics:
            rnfl_err = val_metrics['rnfl_thickness_error']
            rnfl_label = get_quality_label('rnfl_thickness_error', rnfl_err)
            print(f"    {'RNFL Thickness Error':<25s} {rnfl_err:>10.2f} μm {'<5.0 μm':>12s} {rnfl_label:>10s}")

        # Anatomical Validity
        if 'anatomical_validity' in val_metrics:
            anat_val = val_metrics['anatomical_validity']
            anat_label = get_quality_label('anatomical_validity', anat_val)
            print(f"    {'Anatomical Validity':<25s} {anat_val:>10.1f} % {'>95.0 %':>12s} {anat_label:>10s}")

        # Edge Preservation
        if 'edge_preservation' in val_metrics:
            edge_pres = val_metrics['edge_preservation']
            edge_label = get_quality_label('edge_preservation', edge_pres)
            print(f"    {'Edge Preservation':<25s} {edge_pres:>10.4f}   {'>0.85':>12s} {edge_label:>10s}")

        # LPIPS
        if 'lpips' in val_metrics:
            lpips_val = val_metrics['lpips']
            lpips_label = get_quality_label('lpips', lpips_val)
            print(f"    {'LPIPS (perceptual)':<25s} {lpips_val:>10.4f}   {'<0.10':>12s} {lpips_label:>10s}")

        print(f"    {'-'*60}")

        # Print Neuro-Symbolic Metrics (5 components)
        if 'ns_total' in val_metrics:
            print(f"\n  Neuro-Symbolic Losses (lower = better):")
            print(f"    {'Component':<15s} {'Loss':>10s}")
            print(f"    {'-'*25}")
            print(f"    {'Ordering':<15s} {val_metrics.get('ns_ordering', 0):>10.4f}")
            print(f"    {'Thickness':<15s} {val_metrics.get('ns_thickness', 0):>10.4f}")
            print(f"    {'Intensity':<15s} {val_metrics.get('ns_intensity', 0):>10.4f}")
            print(f"    {'Continuity':<15s} {val_metrics.get('ns_continuity', 0):>10.4f}")
            print(f"    {'Anatomy':<15s} {val_metrics.get('ns_anatomy', 0):>10.4f}")
            print(f"    {'-'*25}")
            print(f"    {'Total NS':<15s} {val_metrics.get('ns_total', 0):>10.4f}")

        print(f"{'='*80}\n")

        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'best_psnr.pth'))
            print(f"Saved best model (PSNR: {best_psnr:.2f} dB)")

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
        }, os.path.join(args.output_dir, 'latest.pth'))

        # Save adapters separately for continual learning
        if args.continual_learning:
            adapter_path = os.path.join(args.output_dir, 'adapters_latest.pth')
            model.save_adapters(adapter_path)

        scheduler.step()

        # Memory cleanup after each epoch to prevent fragmentation and leaks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Output directory: {args.output_dir}")

    # Final adapter save for continual learning
    if args.continual_learning:
        final_adapter_path = os.path.join(args.output_dir, 'adapters_final.pth')
        model.save_adapters(final_adapter_path)
        print(f"Saved final adapters to: {final_adapter_path}")


if __name__ == '__main__':
    main()
