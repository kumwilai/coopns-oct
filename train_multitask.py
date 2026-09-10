#!/usr/bin/env python3
"""
Multi-Task Training: Joint Denoising and Layer Segmentation

This is the KEY CONTRIBUTION for TMI paper:
- Combines denoising and segmentation in a unified framework
- Shows bidirectional benefit: denoising helps segmentation, segmentation helps denoising
- Anatomically-aware adaptive denoising with real layer boundaries

KEY FIX: Dynamic Loss Scaling to balance multi-task learning
- Denoising loss (~0.0007) is ~1770x smaller than segmentation loss (~1.24)
- Without scaling, segmentation dominates and PSNR degrades over epochs
- Dynamic scaling normalizes both losses to equal magnitude
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class DynamicLossScaler:
    """
    Dynamic loss scaling for multi-task learning.

    Normalizes losses to have equal gradient magnitude, preventing one task
    from dominating the other. Uses exponential moving average for stability.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    """

    def __init__(self, num_tasks=2, ema_decay=0.99, initial_scale=1.0):
        """
        Args:
            num_tasks: Number of tasks to balance
            ema_decay: Exponential moving average decay for loss tracking
            initial_scale: Initial scale factor (losses scaled to this magnitude)
        """
        self.num_tasks = num_tasks
        self.ema_decay = ema_decay
        self.initial_scale = initial_scale

        # Running averages of loss magnitudes
        self.loss_emas = [None] * num_tasks

        # Scale factors for each task (computed from EMAs)
        self.scales = [1.0] * num_tasks

    def update(self, losses):
        """
        Update running averages and compute new scale factors.

        Args:
            losses: List of loss values (as floats, not tensors)

        Returns:
            List of scale factors to apply to each loss
        """
        for i, loss in enumerate(losses):
            if loss <= 0:
                continue

            # Update EMA
            if self.loss_emas[i] is None:
                self.loss_emas[i] = loss
            else:
                self.loss_emas[i] = self.ema_decay * self.loss_emas[i] + (1 - self.ema_decay) * loss

        # Compute scale factors to normalize to initial_scale
        # All non-None EMAs should be scaled to equal magnitude
        valid_emas = [ema for ema in self.loss_emas if ema is not None and ema > 0]

        if len(valid_emas) >= 2:
            # Target magnitude: geometric mean of all EMAs, then scale to initial_scale
            target = self.initial_scale

            for i, ema in enumerate(self.loss_emas):
                if ema is not None and ema > 0:
                    self.scales[i] = target / ema

        return self.scales

    def get_scales(self):
        """Get current scale factors."""
        return self.scales

    def get_info(self):
        """Get diagnostic information."""
        return {
            'loss_emas': self.loss_emas,
            'scales': self.scales,
        }

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.models.noise_features import NoiseFeatureExtractor
from nsnd.models.layer_segmentation import LightweightLayerSegmenter
from nsnd.models.boundary_aware_segmenter import (
    BoundaryAwareSegmenter,
    BoundaryAwareLoss,
    create_boundary_aware_segmenter
)
from nsnd.utils.metrics import compute_psnr, compute_ssim
from nsnd.training.losses import (
    ClinicalWeightedMSELoss,
    ClinicalGateDiversityLoss,
    DEFAULT_CLINICAL_WEIGHTS,
    # CUAP-OCT: Novel unified framework losses
    UncertaintyCalibrationLoss,
    PathologyPreservationLoss,
    BoundarySharpnessLoss,
    CUAPUnifiedLoss,
    # Anatomical Consistency (KEY TMI CONTRIBUTION)
    AnatomicalConsistencyLoss,
    # Confidence-Weighted Denoising (KEY TMI CONTRIBUTION)
    ConfidenceWeightedRefinementLoss,
    ConfidenceConsistencyLoss,
)


# =============================================================================
# 4-LAYER SEGMENTATION SCHEME (Merging ONL with INL_OPL)
# =============================================================================
# Original 5-class scheme had ONL (0.7-1% of pixels) consistently collapsing.
# Solution: Merge ONL into INL_OPL to create a 4-class scheme.
#
# Original:                    New 4-class:
# 0: RNFL_GCL                  0: RNFL_GCL
# 1: INL_OPL                   1: INL_OPL_ONL (merged)
# 2: ONL          ---merged--> 1: INL_OPL_ONL
# 3: IS_OS                     2: IS_OS
# 4: RPE_Choroid               3: RPE_Choroid
# =============================================================================

USE_4_CLASS_SEGMENTATION = True  # Set to False to use original 5-class scheme
NUM_SEG_CLASSES = 4 if USE_4_CLASS_SEGMENTATION else 5

LAYER_NAMES_4CLASS = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
LAYER_NAMES_5CLASS = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']
LAYER_NAMES = LAYER_NAMES_4CLASS if USE_4_CLASS_SEGMENTATION else LAYER_NAMES_5CLASS

# =============================================================================
# BOUNDARY-AWARE SEGMENTATION WITH DEEP SUPERVISION
# =============================================================================
# Enhanced architecture for thin layer segmentation (IS_OS):
# 1. Dual-head: Region segmentation + Boundary detection
# 2. Deep supervision: Auxiliary losses at multiple decoder stages
# 3. Columnar attention: Exploits horizontal layer structure
# 4. ASPP: Multi-scale feature extraction
# =============================================================================
USE_BOUNDARY_AWARE_SEGMENTER = False  # Using simpler segmenter to fix IS_OS collapse
BOUNDARY_LOSS_WEIGHT = 0.5  # Weight for boundary detection loss
DEEP_SUPERVISION_WEIGHTS = [0.4, 0.3, 0.2]  # Weights for aux losses (1/2, 1/4, 1/8 res)

# =============================================================================
# IS_OS BOUNDARY DETECTION MODE (V4 approach - achieved 0.48 Dice)
# =============================================================================
# Instead of treating IS_OS as a region class (which often collapses),
# detect IS_OS as the BOUNDARY between INL_OPL_ONL and RPE_Choroid.
# This approach achieved 0.4787 IS_OS Dice with stable full-image validation.
#
# 3-class segmentation: RNFL_GCL, INL_OPL_ONL, RPE_Choroid
# + IS_OS boundary head (binary classification)
# =============================================================================
USE_BOUNDARY_DETECTION_FOR_ISOS = False  # Set via --use_boundary_detection argument
NUM_BOUNDARY_CLASSES = 3  # RNFL_GCL, INL_OPL_ONL, RPE_Choroid
BOUNDARY_CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'RPE_Choroid']


def remap_mask_5to3_boundary(mask):
    """Remap 5-class mask to 3-class for boundary detection mode.

    IS_OS becomes a boundary (not a class), assigned to INL_OPL_ONL region.

    Input: mask with classes 0,1,2,3,4 (RNFL, INL, ONL, IS_OS, RPE)
    Output: mask with classes 0,1,2 (RNFL_GCL, INL_OPL_ONL, RPE_Choroid)
    """
    new_mask = np.zeros_like(mask)
    new_mask[mask == 0] = 0  # RNFL_GCL
    new_mask[mask == 1] = 1  # INL -> INL_OPL_ONL
    new_mask[mask == 2] = 1  # ONL -> INL_OPL_ONL
    new_mask[mask == 3] = 1  # IS_OS -> assign to INL_OPL_ONL (boundary)
    new_mask[mask == 4] = 2  # RPE_Choroid
    return new_mask


def extract_is_os_boundary_mask(mask_5class):
    """Extract IS_OS boundary from 5-class mask.

    Returns binary mask where IS_OS pixels (class 3) are 1.
    """
    return (mask_5class == 3).astype(np.float32)


def remap_mask_5to4(mask):
    """Remap 5-class mask to 4-class by merging ONL into INL_OPL.

    Input: mask with classes 0,1,2,3,4 (RNFL, INL, ONL, IS_OS, RPE)
    Output: mask with classes 0,1,2,3 (RNFL, INL+ONL, IS_OS, RPE)
    """
    new_mask = mask.copy()
    # ONL (class 2) -> INL_OPL (class 1)
    new_mask[mask == 2] = 1
    # IS_OS (class 3) -> new class 2
    new_mask[mask == 3] = 2
    # RPE_Choroid (class 4) -> new class 3
    new_mask[mask == 4] = 3
    return new_mask


class MultiTaskOCTDataset(Dataset):
    """Dataset for joint denoising and segmentation.

    Supports two JSONL formats:
    1. Original format: noisy_path, clean_path, seg_path (NPY), weights
    2. Duke DME format: image_path (clean), mask_path (PNG)

    For Duke DME format with add_synthetic_noise=False, noisy=clean (segmentation only).
    For Duke DME format with add_synthetic_noise=True, synthetic noise is added.

    Multi-level noise augmentation:
    - When noise_levels is a list, randomly samples a noise level per image
    - Combines Gaussian additive noise + speckle multiplicative noise (realistic OCT)
    - Effectively multiplies training data by len(noise_levels)
    """

    def __init__(self, jsonl_path, patch_size=64, max_samples=None, random_crop=True,
                 ensure_all_layers=True, num_classes=NUM_SEG_CLASSES, add_synthetic_noise=False,
                 noise_level=0.1, noise_levels=None, use_speckle_noise=True,
                 layer_balanced_sampling=False, thin_layer_boost=3):
        """
        Args:
            jsonl_path: Path to JSONL file with sample info
            patch_size: Size of patches to extract
            max_samples: Maximum number of samples to load
            random_crop: If True, use random crops (training). If False, use center crop (validation).
            ensure_all_layers: If True, ensure random crops contain all layer classes.
            num_classes: Number of layer classes (default 5).
            add_synthetic_noise: If True, add synthetic noise to clean images (for Duke DME).
            noise_level: Standard deviation for synthetic Gaussian noise (default 0.1).
                        Ignored if noise_levels is provided.
            noise_levels: List of noise levels to randomly sample from (e.g., [0.05, 0.10, 0.15, 0.20, 0.25]).
                         If provided, overrides noise_level for multi-level augmentation.
            use_speckle_noise: If True, add realistic OCT speckle noise (multiplicative) in addition to Gaussian.
            layer_balanced_sampling: If True, sample patches centered on each layer with equal probability.
                                    This helps with thin layer segmentation by ensuring the model sees
                                    thin layer pixels in a significant portion of the training data.
            thin_layer_boost: Boost factor for sampling thin layers (1-3). Default 3 means thin layers
                             are 3x more likely to be selected as the center.
        """
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.ensure_all_layers = ensure_all_layers
        self.num_classes = num_classes
        self.add_synthetic_noise = add_synthetic_noise
        self.use_speckle_noise = use_speckle_noise
        self.layer_balanced_sampling = layer_balanced_sampling
        self.thin_layer_boost = thin_layer_boost

        # Layer sampling weights: BALANCED boost for all layers
        # Layer order: RNFL_GCL(0), INL_OPL(1), ONL(2), IS_OS(3), RPE_Choroid(4)
        # Goal: Each layer gets reasonable representation without oscillation
        if layer_balanced_sampling:
            # Balanced approach: equal sampling probability for ALL layers
            # This ensures every layer gets seen equally often (20% each)
            # The loss function handles the class weights, sampling handles exposure
            self.layer_sample_weights = np.array([
                1.0,                    # RNFL - equal
                1.0,                    # INL - equal
                1.0,                    # ONL - equal (thinnest, but sampling is about patch center)
                1.0,                    # IS_OS - equal
                1.0                     # RPE - equal
            ])
            self.layer_sample_weights /= self.layer_sample_weights.sum()  # Normalize (20% each)
            print(f"Layer-balanced sampling weights: {self.layer_sample_weights} (EQUAL)")

        # Multi-level noise: if noise_levels provided, use it; else use single noise_level
        if noise_levels is not None:
            self.noise_levels = noise_levels
        else:
            self.noise_levels = [noise_level]

        # Detect format from first sample
        if self.samples:
            first = self.samples[0]
            self.is_duke_dme_format = 'image_path' in first and 'mask_path' in first

    # Dataset-specific noise parameters derived from empirical analysis
    # See: analyze_noise_composition.py, pku37_noise_params.json, duke_noise_params.json
    NOISE_PARAMS = {
        'duke': {
            # Duke analysis (39 images): more balanced noise composition
            'dirichlet_alpha': [1.69, 0.78, 1.27, 1.26],  # [speckle, banding, gaussian, shot]
            'speckle_k': 6.83,           # Higher k = less speckle variance
            'speckle_cv': 0.38,
            'banding_freq_range': (15, 120),
            'banding_amp': 0.21,
            'gaussian_sigma_scale': 0.058,
            'shot_gain': 19.7,
        },
        'pku37': {
            # PKU37 analysis (100 images): speckle-dominant
            'dirichlet_alpha': [2.33, 0.44, 0.62, 1.61],  # [speckle, banding, gaussian, shot]
            'speckle_k': 3.85,           # Lower k = more speckle variance
            'speckle_cv': 0.51,
            'banding_freq_range': (10, 80),
            'banding_amp': 0.15,
            'gaussian_sigma_scale': 0.027,
            'shot_gain': 20.0,
        },
        'oct5k': {
            # OCT5k: similar to Duke (high-quality clinical scans)
            'dirichlet_alpha': [1.5, 0.7, 1.5, 1.3],  # Balanced
            'speckle_k': 5.5,
            'speckle_cv': 0.42,
            'banding_freq_range': (20, 130),
            'banding_amp': 0.18,
            'gaussian_sigma_scale': 0.045,
            'shot_gain': 25.0,
        },
        'generic': {
            # Generic fallback: Gaussian-dominant (conservative)
            'dirichlet_alpha': [1.0, 0.5, 3.0, 1.0],  # Gaussian-dominant (~55%)
            'speckle_k': 5.0,
            'speckle_cv': 0.45,
            'banding_freq_range': (20, 150),
            'banding_amp': 0.18,
            'gaussian_sigma_scale': 0.04,
            'shot_gain': 20.0,
        }
    }

    def _detect_source_dataset(self, image_path: str) -> str:
        """Detect source dataset from image path."""
        path_lower = image_path.lower()
        if 'duke' in path_lower or 'dme' in path_lower:
            return 'duke'
        elif 'pku' in path_lower or 'pku37' in path_lower:
            return 'pku37'
        elif 'oct5k' in path_lower or 'ucl' in path_lower:
            return 'oct5k'
        else:
            return 'generic'

    def _add_synthetic_oct_noise(self, clean, noise_level, source_dataset='generic'):
        """
        Add realistic synthetic OCT noise combining 4 heterogeneous noise types:
        1. Speckle noise (multiplicative, signal-dependent) - dominant in OCT
        2. Banding noise (horizontal stripes) - common OCT artifact
        3. Gaussian noise (additive) - sensor/thermal noise
        4. Shot noise (Poisson, signal-dependent) - photon counting noise

        The mixture weights are sampled from DATASET-SPECIFIC Dirichlet distributions
        derived from empirical analysis of real OCT noise (PKU37, Duke datasets).

        Args:
            clean: Clean image [H, W], range [0, 1]
            noise_level: Controls overall noise intensity
            source_dataset: Source dataset name ('duke', 'pku37', 'oct5k', 'generic')

        Returns:
            noisy: Noisy image [H, W], range [0, 1]
        """
        h, w = clean.shape
        noisy = clean.copy()

        # Get dataset-specific noise parameters
        params = self.NOISE_PARAMS.get(source_dataset, self.NOISE_PARAMS['generic'])

        # Sample noise mixture weights from dataset-specific Dirichlet distribution
        if self.use_speckle_noise:
            alpha = params['dirichlet_alpha']
        else:
            # Reduce speckle contribution if disabled
            alpha = [0.1, params['dirichlet_alpha'][1],
                     params['dirichlet_alpha'][2], params['dirichlet_alpha'][3]]

        weights = np.random.dirichlet(alpha)
        w_speckle, w_banding, w_gaussian, w_shot = weights

        # 1. SPECKLE NOISE (multiplicative, Gamma distribution)
        # Use dataset-specific speckle_k as baseline
        if w_speckle > 0.1:
            base_k = params['speckle_k']
            # Modulate k based on noise_level (higher noise_level = lower k = more variance)
            speckle_k = max(1.0, base_k / (1.0 + noise_level * w_speckle * 5))
            speckle_k = min(speckle_k, 20.0)
            speckle = np.random.gamma(speckle_k, 1.0 / speckle_k, size=(h, w)).astype(np.float32)
            noisy = noisy * speckle

        # 2. BANDING NOISE (horizontal stripes at random frequencies)
        # Use dataset-specific frequency range and amplitude
        if w_banding > 0.05:
            freq_low, freq_high = params['banding_freq_range']
            freq = np.random.randint(freq_low, freq_high)
            base_amp = params['banding_amp']
            amplitude = noise_level * w_banding * base_amp * 2.5
            # Create horizontal banding pattern
            y_coords = np.arange(h).reshape(-1, 1)
            phase = np.random.uniform(0, 2 * np.pi)
            banding = amplitude * np.sin(2 * np.pi * freq * y_coords / h + phase)
            banding = np.tile(banding, (1, w)).astype(np.float32)
            noisy = noisy + banding

        # 3. GAUSSIAN NOISE (additive, signal-independent)
        # Use dataset-specific gaussian sigma scale
        if w_gaussian > 0.05:
            base_sigma = params['gaussian_sigma_scale']
            gaussian_sigma = noise_level * w_gaussian * base_sigma * 10
            gaussian_noise = np.random.randn(h, w).astype(np.float32) * gaussian_sigma
            # Apply vertical smoothing for realistic spatial correlation
            from scipy.ndimage import gaussian_filter1d
            gaussian_noise = gaussian_filter1d(gaussian_noise, sigma=1.5, axis=0)
            # Rescale to maintain target noise level after smoothing
            gaussian_noise = gaussian_noise * (gaussian_sigma / (gaussian_noise.std() + 1e-8))
            noisy = noisy + gaussian_noise

        # 4. SHOT NOISE (Poisson, variance proportional to signal)
        # Use dataset-specific shot gain
        if w_shot > 0.05:
            base_gain = params['shot_gain']
            # Higher gain = less noise, modulate by noise_level
            peak = max(10, base_gain * 2 / (noise_level * w_shot + 0.1))
            # Poisson parameter = clean * peak
            shot_noisy = np.random.poisson(np.maximum(clean * peak, 0.1)) / peak
            # Blend shot noise with current noisy
            noisy = noisy * (1 - w_shot * 0.5) + shot_noisy.astype(np.float32) * (w_shot * 0.5)

        # Clip to valid range
        noisy = np.clip(noisy, 0, 1)
        return noisy

    def __len__(self):
        return len(self.samples)

    def _get_layer_centers(self, seg_mask):
        """
        Get the center y-coordinate of each layer in the segmentation mask.

        Returns:
            dict: {layer_idx: center_y} for each layer present in the mask
        """
        layer_centers = {}
        for c in range(self.num_classes):
            rows_with_layer = np.where(np.any(seg_mask == c, axis=1))[0]
            if len(rows_with_layer) > 0:
                # Use median row as layer center
                layer_centers[c] = int(np.median(rows_with_layer))
        return layer_centers

    def _sample_layer_balanced_crop(self, seg_mask, h, w):
        """
        Sample a crop position centered on a randomly selected layer.
        Thin layers are boosted to be sampled more often.

        Returns:
            (top, left): Crop position
        """
        layer_centers = self._get_layer_centers(seg_mask)

        if not layer_centers:
            # Fallback to center crop
            return max(0, (h - self.patch_size) // 2), max(0, (w - self.patch_size) // 2)

        # Get available layers and their sampling weights
        available_layers = list(layer_centers.keys())
        weights = np.array([self.layer_sample_weights[c] for c in available_layers])
        weights /= weights.sum()  # Normalize

        # Sample a layer
        selected_layer = np.random.choice(available_layers, p=weights)
        center_y = layer_centers[selected_layer]

        # Compute crop position centered on the selected layer
        top = center_y - self.patch_size // 2
        top = max(0, min(top, h - self.patch_size))

        # Horizontal position: random within valid range
        max_left = max(0, w - self.patch_size)
        left = np.random.randint(0, max_left + 1) if max_left > 0 else 0

        return top, left

    def _find_all_layers_region(self, seg_mask):
        """
        Find the vertical region where all layers are present.

        Returns:
            (top_min, top_max): Valid range for crop top position.
            Returns None if no valid region exists.
        """
        h, w = seg_mask.shape

        # Edge case: image smaller than patch
        if h < self.patch_size:
            return None

        # For each row, check which layers are present
        # Find the row range where ALL layers can be captured in a patch_size window

        # Find the extent of each layer
        layer_tops = []
        layer_bottoms = []

        for c in range(self.num_classes):
            rows_with_layer = np.where(np.any(seg_mask == c, axis=1))[0]
            if len(rows_with_layer) == 0:
                # Layer not present in image - can't guarantee all layers
                return None
            layer_tops.append(rows_with_layer.min())
            layer_bottoms.append(rows_with_layer.max())

        # To include all layers, the crop must:
        # - Start at or before the topmost layer's top
        # - End at or after the bottommost layer's bottom
        #
        # For a patch of size P starting at row T:
        # - Top of patch: T
        # - Bottom of patch: T + P - 1
        #
        # Constraints:
        # - T <= min(layer_tops) to include top layer
        # - T + P - 1 >= max(layer_bottoms) to include bottom layer
        # => T >= max(layer_bottoms) - P + 1
        # => T <= min(layer_tops)

        top_layer_start = min(layer_tops)
        bottom_layer_end = max(layer_bottoms)

        # The patch must span from top_layer_start to bottom_layer_end
        required_height = bottom_layer_end - top_layer_start + 1

        # Maximum valid top position (ensure we don't go past image boundary)
        max_valid_top = max(0, h - self.patch_size)

        if required_height > self.patch_size:
            # All layers don't fit in one patch - return region that maximizes coverage
            # Start from where we can get the most layers
            # Use the middle of the layer stack
            middle = (top_layer_start + bottom_layer_end) // 2
            top_min = max(0, middle - self.patch_size // 2)
            top_max = min(max_valid_top, middle - self.patch_size // 2)
            return (top_min, max(top_min, top_max))

        # Valid crop range: top can be anywhere that includes all layers
        top_min = max(0, bottom_layer_end - self.patch_size + 1)
        top_max = min(max_valid_top, top_layer_start)

        if top_min > top_max:
            # No valid region - use best effort (center on layer stack)
            center_top = max(0, min(top_layer_start, max_valid_top))
            return (center_top, center_top)

        return (top_min, top_max)

    def __getitem__(self, idx):
        data = self.samples[idx]

        # Load images based on format
        if self.is_duke_dme_format:
            # Duke DME format: image_path (clean), mask_path (PNG mask)
            clean = np.array(Image.open(data['image_path']).convert('L'), dtype=np.float32) / 255.0
            seg_mask = np.array(Image.open(data['mask_path']), dtype=np.int64)

            # Apply 4-class remapping if enabled (merge ONL into INL_OPL)
            if USE_4_CLASS_SEGMENTATION:
                seg_mask = remap_mask_5to4(seg_mask)

            if self.add_synthetic_noise:
                # Multi-level noise augmentation: randomly sample a noise level
                noise_level = np.random.choice(self.noise_levels)

                # Detect source dataset for dataset-specific noise characteristics
                source_dataset = self._detect_source_dataset(data['image_path'])

                # Add realistic OCT noise with dataset-specific parameters
                noisy = self._add_synthetic_oct_noise(clean, noise_level, source_dataset)
            else:
                # Segmentation-only training: noisy = clean
                noisy = clean.copy()
        else:
            # Original format: noisy_path, clean_path, seg_path (NPY)
            noisy_key = 'noisy_path' if 'noisy_path' in data else 'noisy'
            clean_key = 'clean_path' if 'clean_path' in data else 'clean'
            noisy = np.array(Image.open(data[noisy_key]).convert('L'), dtype=np.float32) / 255.0
            clean = np.array(Image.open(data[clean_key]).convert('L'), dtype=np.float32) / 255.0
            seg_mask = np.load(data['seg_path'])

        h, w = noisy.shape

        # Compute safe crop ranges (handle images smaller than patch_size)
        max_top = max(0, h - self.patch_size)
        max_left = max(0, w - self.patch_size)

        if self.random_crop:
            if self.layer_balanced_sampling:
                # LAYER-BALANCED SAMPLING (KEY FIX for thin layer segmentation)
                # Sample patches centered on each layer with boosted probability for thin layers
                # This ensures the model sees thin layer pixels in a significant portion of training
                top, left = self._sample_layer_balanced_crop(seg_mask, h, w)
            elif self.ensure_all_layers:
                # Find valid vertical region that includes all layers
                valid_region = self._find_all_layers_region(seg_mask)
                if valid_region is not None:
                    top_min, top_max = valid_region
                    if top_min < top_max:
                        top = np.random.randint(top_min, top_max + 1)
                    else:
                        top = top_min
                else:
                    # Fallback to center crop if layers can't all be found
                    top = max_top // 2
                # Horizontal position is always random
                left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
            else:
                # Unconstrained random crop
                top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
                # Horizontal position is always random
                left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
        else:
            # Center crop - for consistent validation
            top = max_top // 2
            left = max_left // 2

        # Ensure indices are valid
        top = max(0, min(top, max_top))
        left = max(0, min(left, max_left))

        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]
        seg_mask = seg_mask[top:top+self.patch_size, left:left+self.patch_size]

        # Weights: use data weights if available, else default uniform
        if 'weights' in data:
            weights = torch.tensor([
                data['weights'].get('speckle', 0.25),
                data['weights'].get('banding', 0.25),
                data['weights'].get('gaussian', 0.25),
                data['weights'].get('shot', 0.25)
            ], dtype=torch.float32)
        else:
            # Duke DME format: no noise weights, use uniform
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)

        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean).unsqueeze(0).float(),
            'seg_mask': torch.from_numpy(seg_mask).long(),
            'weights': weights,
        }


class PerLayerEvalDataset(Dataset):
    """
    Dataset for per-layer evaluation within memory constraints.

    Instead of trying to fit all layers in one 64x64 patch (impossible),
    this dataset returns separate crops per image, each centered on a
    specific layer. This allows evaluation of all layers while staying
    within the 64x64 memory constraint.
    """

    def __init__(self, jsonl_path, patch_size=64, max_samples=None, num_classes=NUM_SEG_CLASSES,
                 precompute_centers=True, add_synthetic_noise=False, noise_level=0.1):
        """
        Args:
            jsonl_path: Path to JSONL file with sample info
            patch_size: Size of patches to extract (default 64)
            max_samples: Maximum number of samples to load
            num_classes: Number of layer classes (default NUM_SEG_CLASSES)
            precompute_centers: If True, precompute all layer centers at init.
                               If False, compute lazily (saves memory but slower).
            add_synthetic_noise: If True, add synthetic noise to clean images (for Duke DME).
            noise_level: Standard deviation for synthetic Gaussian noise (default 0.1).
        """
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.precompute_centers = precompute_centers
        self.add_synthetic_noise = add_synthetic_noise
        self.noise_level = noise_level

        # Detect format from first sample
        if self.samples:
            first = self.samples[0]
            self.is_duke_dme_format = 'image_path' in first and 'mask_path' in first
        else:
            self.is_duke_dme_format = False

        # Layer centers cache (only coordinates, not full masks)
        # Format: {sample_idx: {layer_idx: center_row or None}}
        self.layer_centers_cache = {}

        if precompute_centers:
            # Pre-compute all layer centers (memory efficient: only stores coordinates)
            self._precompute_all_layer_centers()

    def _precompute_all_layer_centers(self):
        """Pre-compute layer centers for all samples (stores only coordinates)."""
        for idx, sample in enumerate(self.samples):
            self.layer_centers_cache[idx] = self._compute_layer_centers_for_sample(sample)
            # Explicit cleanup after each sample to avoid memory accumulation
            gc.collect()

    def _compute_layer_centers_for_sample(self, sample):
        """
        Compute layer centers for a single sample.
        Loads mask, extracts centers, then immediately frees the mask.
        Returns dict of {layer_idx: center_row or None}.
        """
        if self.is_duke_dme_format:
            seg_mask = np.array(Image.open(sample['mask_path']), dtype=np.int64)
        else:
            seg_mask = np.load(sample['seg_path'])

        # Apply 4-class remapping if enabled
        if USE_4_CLASS_SEGMENTATION:
            seg_mask = remap_mask_5to4(seg_mask)

        centers = {}
        for c in range(self.num_classes):
            rows_with_layer = np.where(np.any(seg_mask == c, axis=1))[0]
            if len(rows_with_layer) > 0:
                # Store only the center coordinate (integer), not the mask
                centers[c] = int((rows_with_layer.min() + rows_with_layer.max()) // 2)
            else:
                centers[c] = None

        # Explicitly delete the mask to free memory immediately
        del seg_mask

        return centers

    def _get_layer_centers(self, sample_idx):
        """Get layer centers for a sample, computing lazily if needed."""
        if sample_idx not in self.layer_centers_cache:
            self.layer_centers_cache[sample_idx] = self._compute_layer_centers_for_sample(
                self.samples[sample_idx]
            )
        return self.layer_centers_cache[sample_idx]

    def __len__(self):
        # Each sample produces num_classes crops (one per layer)
        return len(self.samples) * self.num_classes

    def __getitem__(self, idx):
        # Determine which sample and which layer
        sample_idx = idx // self.num_classes
        layer_idx = idx % self.num_classes

        data = self.samples[sample_idx]
        layer_centers = self._get_layer_centers(sample_idx)
        layer_center = layer_centers.get(layer_idx)

        # Load images based on format
        if self.is_duke_dme_format:
            # Duke DME format: image_path (clean), mask_path (PNG mask)
            clean = np.array(Image.open(data['image_path']).convert('L'), dtype=np.float32) / 255.0
            seg_mask = np.array(Image.open(data['mask_path']), dtype=np.int64)

            # Apply 4-class remapping if enabled
            if USE_4_CLASS_SEGMENTATION:
                seg_mask = remap_mask_5to4(seg_mask)

            if self.add_synthetic_noise:
                noise = np.random.randn(*clean.shape).astype(np.float32) * self.noise_level
                noisy = np.clip(clean + noise, 0, 1)
            else:
                noisy = clean.copy()
        else:
            # Original format
            noisy_key = 'noisy_path' if 'noisy_path' in data else 'noisy'
            clean_key = 'clean_path' if 'clean_path' in data else 'clean'
            noisy = np.array(Image.open(data[noisy_key]).convert('L'), dtype=np.float32) / 255.0
            clean = np.array(Image.open(data[clean_key]).convert('L'), dtype=np.float32) / 255.0
            seg_mask = np.load(data['seg_path'])

            # Apply 4-class remapping if enabled
            if USE_4_CLASS_SEGMENTATION:
                seg_mask = remap_mask_5to4(seg_mask)

        h, w = noisy.shape

        # If layer doesn't exist in this sample, or image too small, return invalid flag
        if layer_center is None or h < self.patch_size or w < self.patch_size:
            # Return zeros with a flag indicating invalid
            return {
                'noisy': torch.zeros(1, self.patch_size, self.patch_size),
                'clean': torch.zeros(1, self.patch_size, self.patch_size),
                'seg_mask': torch.zeros(self.patch_size, self.patch_size, dtype=torch.long),
                'layer_idx': layer_idx,
                'valid': False,
                'weights': torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32),
            }

        # Compute crop position centered on this layer
        top = layer_center - self.patch_size // 2
        top = max(0, min(top, h - self.patch_size))

        # Horizontal: use center
        left = (w - self.patch_size) // 2
        left = max(0, left)  # Already guaranteed w >= patch_size

        # Extract crop
        noisy_crop = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean_crop = clean[top:top+self.patch_size, left:left+self.patch_size]
        seg_crop = seg_mask[top:top+self.patch_size, left:left+self.patch_size]

        # Handle weights - Duke DME format doesn't have weights, use defaults
        if 'weights' in data:
            weights = torch.tensor([
                data['weights'].get('speckle', 0.25),
                data['weights'].get('banding', 0.25),
                data['weights'].get('gaussian', 0.25),
                data['weights'].get('shot', 0.25)
            ], dtype=torch.float32)
        else:
            # Default uniform weights for Duke DME format
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)

        return {
            'noisy': torch.from_numpy(noisy_crop).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean_crop).unsqueeze(0).float(),
            'seg_mask': torch.from_numpy(seg_crop).long(),
            'layer_idx': layer_idx,
            'valid': True,
            'weights': weights,
        }


class PerSampleFeatureAligner(nn.Module):
    """
    Domain-Agnostic Feature Alignment for OOD Generalization.

    Normalizes noise features per-sample to have zero mean and unit variance.
    This makes the layer gates invariant to the absolute scale of noise features,
    enabling generalization across different OCT devices and noise levels.

    Key insight: The RELATIVE pattern of noise features matters, not absolute values.
    - High CoefVar region vs low CoefVar region (within same image)
    - This pattern is consistent across domains
    """

    def __init__(self, n_features=7, eps=1e-6):
        super().__init__()
        self.n_features = n_features
        self.eps = eps

        # Optional: learnable affine transform after normalization
        # Allows model to learn optimal scale/shift
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, features):
        """
        Normalize features per-sample for domain-agnostic inference.

        Args:
            features: [B, 7, H, W] - raw noise features

        Returns:
            aligned: [B, 7, H, W] - normalized features (zero mean, unit std per channel)
        """
        # Compute per-sample, per-channel statistics
        mean = features.mean(dim=(2, 3), keepdim=True)  # [B, 7, 1, 1]
        std = features.std(dim=(2, 3), keepdim=True) + self.eps  # [B, 7, 1, 1]

        # Normalize
        normalized = (features - mean) / std

        # Apply learnable affine transform
        gamma = self.gamma.view(1, -1, 1, 1)  # [1, 7, 1, 1]
        beta = self.beta.view(1, -1, 1, 1)    # [1, 7, 1, 1]

        aligned = gamma * normalized + beta

        return aligned


class LayerSpecificNoiseGates(nn.Module):
    """
    Layer-Specific Noise Modeling - Key TMI Contribution

    Different retinal layers have different noise characteristics due to:
    - Tissue-specific light scattering (RPE scatters more than NFL)
    - Layer-specific speckle patterns
    - Varying signal-to-noise ratios across depth

    This module learns SEPARATE noise-adaptive gates for each anatomical layer,
    allowing the denoiser to apply layer-appropriate processing.
    """

    def __init__(self, noise_feat_dim=7, n_layers=None, hidden_dim=16):
        super().__init__()
        # Use global NUM_SEG_CLASSES if not specified
        self.n_layers = n_layers if n_layers is not None else NUM_SEG_CLASSES
        self.layer_names = LAYER_NAMES

        # Separate gate network for each anatomical layer
        # Each learns to respond to noise features differently
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(noise_feat_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, 1),
                nn.Sigmoid()
            ) for _ in range(n_layers)
        ])

        # Learnable layer-specific noise sensitivity
        # Higher value = this layer needs more aggressive denoising
        self.layer_noise_sensitivity = nn.Parameter(torch.ones(n_layers))

        # Learnable layer-specific base refinement
        # Allows different "baseline" refinement per layer even without noise
        self.layer_base_refinement = nn.Parameter(torch.zeros(n_layers))

    def forward(self, noise_features, seg_probs):
        """
        Compute layer-specific noise gates.

        Args:
            noise_features: [B, 7, H, W] - physics-based noise feature maps
            seg_probs: [B, 5, H, W] - softmax segmentation probabilities

        Returns:
            combined_gate: [B, 1, H, W] - final per-pixel gate (weighted by seg)
            layer_gates: [B, 5, H, W] - individual layer gate outputs
            gate_stats: dict with per-layer gate statistics
        """
        B, _, H, W = noise_features.shape
        layer_gate_outputs = []

        for i in range(self.n_layers):
            # Compute gate for this layer's noise response
            gate = self.layer_gates[i](noise_features)  # [B, 1, H, W]

            # Scale by learnable sensitivity
            sensitivity = torch.sigmoid(self.layer_noise_sensitivity[i])
            base = torch.sigmoid(self.layer_base_refinement[i]) * 0.5  # 0-0.5 base

            # Final gate: base + sensitivity * learned_gate
            gate = base + sensitivity * gate
            gate = gate.clamp(0, 1)

            layer_gate_outputs.append(gate)

        # Stack gates: [B, 5, H, W]
        layer_gates = torch.cat(layer_gate_outputs, dim=1)

        # CRITICAL FIX: Clamp gates to prevent extreme values
        # Extreme gates (0.33-0.90) cause over/under-denoising
        # Safe range [0.4, 0.8] prevents this while allowing differentiation
        layer_gates = layer_gates.clamp(min=0.4, max=0.8)

        # Combine gates weighted by segmentation probabilities
        # Each pixel's final gate is weighted avg based on which layer it belongs to
        combined_gate = (layer_gates * seg_probs).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Compute statistics for analysis (use CLAMPED layer_gates for accurate reporting)
        gate_stats = {}
        with torch.no_grad():
            for i, name in enumerate(self.layer_names):
                layer_mask = seg_probs[:, i:i+1, :, :]
                mask_sum = layer_mask.sum()
                if mask_sum > 0:
                    # Use clamped layer_gates[:, i:i+1] instead of raw layer_gate_outputs[i]
                    gate_in_layer = (layer_gates[:, i:i+1, :, :] * layer_mask).sum() / mask_sum
                    gate_stats[f'{name}_gate_mean'] = gate_in_layer.item()
                    gate_stats[f'{name}_sensitivity'] = torch.sigmoid(
                        self.layer_noise_sensitivity[i]).item()

        return combined_gate, layer_gates, gate_stats

    def get_layer_profiles(self):
        """Return learned noise sensitivity profiles for visualization."""
        profiles = {}
        for i, name in enumerate(self.layer_names):
            profiles[name] = {
                'sensitivity': torch.sigmoid(self.layer_noise_sensitivity[i]).item(),
                'base_refinement': torch.sigmoid(self.layer_base_refinement[i]).item() * 0.5,
            }
        return profiles


class MultiTaskDenoiser(nn.Module):
    """
    Multi-Task Model: Joint Denoising + Layer Segmentation

    KEY TMI CONTRIBUTION: Layer-Specific Noise Modeling
    - Different retinal layers get different denoising treatment
    - Segmentation provides anatomical grounding
    - Learns layer-appropriate noise response

    Architecture:
    1. Shared encoder: Extracts features from noisy image
    2. Segmentation branch: Predicts layer masks
    3. Layer-specific noise gates: Per-layer noise response
    4. Denoising branch: Uses layer info to guide denoising
    """

    def __init__(self, backbone_ckpt=None, segmenter_ckpt=None, confidence_threshold=0.01):
        super().__init__()

        # Confidence threshold for segmentation fallback (OOD robustness)
        # If mean segmentation confidence < threshold, use uniform layer weights
        # NOTE: Set to 0.01 (not 0.1) because model confidence is naturally low (~0.06-0.08)
        # due to entropy-based calculation. Only trigger fallback for truly uncertain cases.
        self.confidence_threshold = confidence_threshold

        # Pre-computed constant for entropy normalization (log(num_classes))
        self.max_entropy = np.log(5)  # ~1.609 for 5 classes

        # 1. Physics-based feature extractor
        self.feature_extractor = NoiseFeatureExtractor()

        # 2. Shared backbone (can load pretrained)
        self.backbone = NAFNetFullFiLM(
            img_channel=1, width=64,
            enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
            middle_blk_num=2, cond_dim=32,
        )

        if backbone_ckpt and os.path.exists(backbone_ckpt):
            ckpt = torch.load(backbone_ckpt, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            print(f"[MultiTaskDenoiser] Loaded backbone from {backbone_ckpt}")

        # 3. Layer segmentation module
        # Choose between lightweight and boundary-aware segmenter
        self.use_boundary_aware = USE_BOUNDARY_AWARE_SEGMENTER
        if USE_BOUNDARY_AWARE_SEGMENTER:
            self.segmenter = BoundaryAwareSegmenter(
                in_channels=1, num_classes=NUM_SEG_CLASSES, base_filters=48
            )
            print(f"[MultiTaskDenoiser] Using BoundaryAwareSegmenter with deep supervision")
        else:
            self.segmenter = LightweightLayerSegmenter(num_classes=NUM_SEG_CLASSES)
            print(f"[MultiTaskDenoiser] Using LightweightLayerSegmenter")

        if segmenter_ckpt and os.path.exists(segmenter_ckpt):
            ckpt = torch.load(segmenter_ckpt, map_location='cpu', weights_only=False)
            self.segmenter.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"[MultiTaskDenoiser] Loaded segmenter from {segmenter_ckpt}")

        # 4. Feature normalization (for refinement branch)
        self.feature_norm = nn.Sequential(
            nn.Conv2d(7, 16, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1),
        )

        # 4b. Feature Alignment for Domain Adaptation
        # Normalizes noise features per-sample for OOD generalization
        self.feature_aligner = PerSampleFeatureAligner(n_features=7)

        # 5. Layer-Specific Noise Gates (KEY TMI CONTRIBUTION)
        # Each anatomical layer gets its own noise-adaptive gate
        # This allows the model to learn that e.g., RPE needs different
        # denoising than RNFL due to different noise characteristics
        self.layer_noise_gates = LayerSpecificNoiseGates(
            noise_feat_dim=7, n_layers=NUM_SEG_CLASSES, hidden_dim=16
        )

        # 6. Layer-aware refinement
        # Input: backbone (1) + physics features (8) + layer logits (NUM_SEG_CLASSES)
        self.refinement = nn.Sequential(
            nn.Conv2d(1 + 8 + NUM_SEG_CLASSES, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )

        self.refinement_scale = nn.Parameter(torch.tensor(0.1))  # Increased from 0.05

        # 6b. CUAP-OCT: Confidence-Weighted Denoising (KEY TMI CONTRIBUTION)
        # Modulate refinement based on segmentation confidence:
        # - High confidence → aggressive denoising (certain about layer structure)
        # - Low confidence → conservative denoising (potential pathology/boundary)
        self.confidence_min_gate = nn.Parameter(torch.tensor(0.3))  # Minimum refinement (30%)
        self.use_confidence_weighting = True  # Can be disabled for ablation

        # 7. CUAP-OCT: Uncertainty Head (KEY TMI CONTRIBUTION)
        # Outputs pixel-wise uncertainty map indicating where model is unsure
        # High uncertainty regions: pathology, boundaries, novel noise patterns
        # Clinical use: Flag uncertain regions for clinician review
        # Input: backbone features (1) + noise features (8) + seg logits (NUM_SEG_CLASSES) + gate (1)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(1 + 8 + NUM_SEG_CLASSES + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]: 0=confident, 1=uncertain
        )

        # 8. IS_OS Boundary Detection Head (V4 approach - achieved 0.48 Dice)
        # When USE_BOUNDARY_DETECTION_FOR_ISOS is True, this head detects IS_OS
        # as a boundary between layers rather than a region class.
        # Input: backbone_out (1) + norm_features (8) + seg_logits (NUM_SEG_CLASSES)
        # Output: IS_OS boundary logits [B, 1, H, W]
        boundary_input_ch = 1 + 8 + NUM_SEG_CLASSES  # Same as refinement input
        self.is_os_boundary_head = nn.Sequential(
            nn.Conv2d(boundary_input_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # Output: 1 channel for IS_OS boundary
        )
        self._is_os_boundary_logits = None  # Store for training loop access

    def forward(self, x, return_features=False):
        """
        Forward pass for multi-task model.

        Args:
            x: Noisy input [B, 1, H, W]
            return_features: Whether to return intermediate features

        Returns:
            denoised: Denoised output [B, 1, H, W]
            seg_logits: Segmentation logits [B, 5, H, W]
            (optional) features: Dict of intermediate features
        """
        # 1. Extract physics features
        raw_features = self.feature_extractor(x)
        feature_stack = torch.cat([
            raw_features['coef_variation'],
            raw_features['local_std'],
            raw_features['signal_var_corr'],
            raw_features['horizontal_ratio'],
            raw_features['horizontal_lines'],
            raw_features['high_freq'],
            raw_features['local_range'],
        ], dim=1)
        norm_features = self.feature_norm(feature_stack)

        # 2. Backbone denoising
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # 3. Layer segmentation (on noisy input - testing robustness)
        # Handle different segmenter outputs (BoundaryAwareSegmenter returns dict)
        if self.use_boundary_aware:
            seg_output = self.segmenter(x, return_aux=True, return_boundary=True)
            seg_logits = seg_output['seg_logits']
            seg_probs_raw = seg_output['seg_probs']  # Already softmaxed
            # Store for training loop loss computation
            self._seg_aux_outputs = seg_output.get('aux_outputs', None)
            self._seg_boundary_logits = seg_output.get('boundary_logits', None)
        else:
            seg_logits = self.segmenter(x)
            seg_probs_raw = F.softmax(seg_logits, dim=1)  # [B, NUM_SEG_CLASSES, H, W]
            self._seg_aux_outputs = None
            self._seg_boundary_logits = None

        # 3b. Uncertainty-aware fallback for OOD robustness
        # If segmentation is uncertain, fall back to uniform layer weights
        # This prevents layer-specific gates from being misled by bad segmentation
        # Use clamp for numerical stability instead of adding epsilon before log
        seg_probs_clamped = seg_probs_raw.clamp(min=1e-8)
        seg_entropy = -(seg_probs_raw * seg_probs_clamped.log()).sum(dim=1)  # [B, H, W]
        seg_confidence = 1 - (seg_entropy / self.max_entropy)  # [B, H, W], range [0, 1]
        mean_confidence = seg_confidence.mean().item()  # Convert to scalar for comparison

        # Threshold for fallback
        # During training: use raw probs (training=True, no fallback)
        # During inference: check confidence and fallback if needed

        if self.training:
            # During training, always use predicted probs (for learning)
            seg_probs = seg_probs_raw
            use_fallback = False
        else:
            # During inference, apply fallback for OOD robustness
            use_fallback = mean_confidence < self.confidence_threshold
            if use_fallback:
                # Low confidence → use uniform layer weights (graceful degradation)
                seg_probs = torch.ones_like(seg_probs_raw) / NUM_SEG_CLASSES
            else:
                seg_probs = seg_probs_raw

        # 4. Align features for domain-agnostic inference
        # This normalizes features per-sample, enabling OOD generalization
        aligned_features = self.feature_aligner(feature_stack)

        # 5. Compute LAYER-SPECIFIC noise-adaptive gates (KEY TMI CONTRIBUTION)
        # Each layer has its own gate that responds to noise features differently
        # Combined gate is weighted by segmentation probabilities
        noise_gate, layer_gates, layer_gate_stats = self.layer_noise_gates(
            aligned_features, seg_probs  # Uses fallback probs if uncertain
        )

        # 5. Layer-guided refinement
        refine_input = torch.cat([backbone_out, norm_features, seg_logits], dim=1)
        refinement_raw = self.refinement(refine_input)

        # 6. Apply layer-specific noise gate to refinement
        # This forces layer-appropriate denoising strength
        refinement_noise_gated = refinement_raw * noise_gate

        # 6b. CUAP-OCT: Confidence-Weighted Denoising (KEY TMI CONTRIBUTION)
        # Modulate refinement based on pixel-wise segmentation confidence
        # Low confidence regions (boundaries, pathology) get LESS denoising
        # This preserves diagnostic features that might be smoothed away
        if self.use_confidence_weighting:
            # seg_confidence is [B, H, W], expand to [B, 1, H, W]
            confidence_map = seg_confidence.unsqueeze(1)  # [B, 1, H, W]

            # Confidence gate: min_gate + (1 - min_gate) * confidence
            # This ensures even uncertain regions get some denoising (min_gate)
            # but confident regions get full denoising
            min_gate = torch.sigmoid(self.confidence_min_gate)  # Learnable, 0-1
            confidence_gate = min_gate + (1 - min_gate) * confidence_map

            # Apply confidence gating
            refinement = refinement_noise_gated * confidence_gate
        else:
            refinement = refinement_noise_gated
            confidence_gate = torch.ones_like(refinement_noise_gated)

        scale = torch.sigmoid(self.refinement_scale) * 0.2
        denoised = backbone_out + scale * refinement
        denoised = denoised.clamp(0, 1)

        # 7. CUAP-OCT: Compute pixel-wise uncertainty
        # Uncertainty should be HIGH where:
        # - Segmentation is uncertain (boundaries)
        # - Noise patterns are unusual
        # - Denoising made large changes (potential artifacts)
        uncertainty_input = torch.cat([backbone_out, norm_features, seg_logits, noise_gate], dim=1)
        uncertainty = self.uncertainty_head(uncertainty_input)

        # 8. IS_OS Boundary Detection (V4 approach)
        # Compute boundary logits using same input as refinement
        is_os_boundary_logits = self.is_os_boundary_head(refine_input)
        self._is_os_boundary_logits = is_os_boundary_logits

        if return_features:
            return denoised, seg_logits, {
                'raw_features': raw_features,
                'aligned_features': aligned_features,  # Domain-adapted features
                'norm_features': norm_features,
                'backbone_out': backbone_out,
                'refinement_raw': refinement_raw,
                'refinement_noise_gated': refinement_noise_gated,  # After noise gate, before confidence
                'refinement': refinement,  # Final gated refinement (noise + confidence)
                'noise_gate': noise_gate,  # Combined layer-weighted gate
                'layer_gates': layer_gates,  # [B, NUM_SEG_CLASSES, H, W] per-layer gates
                'layer_gate_stats': layer_gate_stats,
                'seg_probs': seg_probs,
                'seg_probs_raw': seg_probs_raw,  # Original before fallback
                'seg_confidence': mean_confidence,  # Segmentation confidence (scalar mean)
                'seg_confidence_map': seg_confidence,  # CUAP: pixel-wise confidence [B, H, W]
                'confidence_gate': confidence_gate,  # CUAP: confidence modulation [B, 1, H, W]
                'confidence_min_gate': torch.sigmoid(self.confidence_min_gate).item(),  # Learned minimum
                'seg_fallback_used': use_fallback,  # Whether fallback was triggered
                'scale': scale,
                'uncertainty': uncertainty,  # CUAP: pixel-wise uncertainty [B, 1, H, W]
                # BoundaryAwareSegmenter outputs for deep supervision and boundary loss
                'seg_aux_outputs': self._seg_aux_outputs,  # List of aux logits (if using BoundaryAwareSegmenter)
                'seg_boundary_logits': self._seg_boundary_logits,  # Boundary predictions (if using BoundaryAwareSegmenter)
                # IS_OS Boundary Detection (V4 approach)
                'is_os_boundary_logits': is_os_boundary_logits,  # [B, 1, H, W] boundary logits
            }

        return denoised, seg_logits


# Note: LAYER_NAMES is defined at the top of the file based on USE_4_CLASS_SEGMENTATION


def compute_class_weights_from_loader(loader, num_classes=None, device='cpu', max_batches=50):
    """
    Compute class weights from training data for balanced segmentation loss.

    This is CRITICAL for preventing class collapse where the model only predicts
    middle classes (like ONL) and ignores edge layers (RNFL_GCL, RPE_Choroid).

    Uses smoothed inverse frequency: weight = (median_count / count)^0.5
    This gives higher weights to rare classes without making common class weights ~0.

    Args:
        loader: DataLoader with 'seg_mask' in batches
        num_classes: Number of segmentation classes (default: NUM_SEG_CLASSES)
        device: Device for output tensor
        max_batches: Max batches to sample for efficiency

    Returns:
        Tensor of shape [num_classes] with class weights
    """
    if num_classes is None:
        num_classes = NUM_SEG_CLASSES

    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        seg_mask = batch['seg_mask']
        for c in range(num_classes):
            class_counts[c] += (seg_mask == c).sum().item()

    print(f"  Raw class counts: {[f'{c:.0f}' for c in class_counts.tolist()]}")

    # Handle missing classes by using median count as fallback
    median_count = class_counts[class_counts > 0].median().item() if (class_counts > 0).any() else 1.0
    class_counts = torch.where(class_counts > 0, class_counts, torch.tensor(median_count))

    # Smoothed inverse frequency: sqrt gives gentler weighting
    # This ensures rare classes get higher weights but common classes don't get ~0
    weights = (median_count / class_counts).sqrt()

    # Normalize so mean = 1
    weights = weights / weights.mean()

    # Clamp to reasonable range [0.5, 3.0] to prevent extreme weights
    weights = weights.clamp(min=0.5, max=3.0)

    return weights.float().to(device)


def compute_dice_score(pred_logits, target, num_classes=NUM_SEG_CLASSES):
    """Compute Dice score."""
    pred = pred_logits.argmax(dim=1)
    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())

    # Handle edge case where no classes have valid pixels
    return np.mean(dice_scores) if dice_scores else 0.0


def soft_dice_loss(pred_logits, target, num_classes=NUM_SEG_CLASSES, smooth=1.0, class_weights=None):
    """
    Compute soft Tversky loss for segmentation (differentiable).

    Tversky loss generalizes Dice with asymmetric FP/FN penalties.
    Higher beta = more penalty for false negatives (missing thin layers).

    For thin layers: beta=0.7 (penalize FN heavily to prevent collapse)
    For thick layers: beta=0.5 (standard Dice)

    Args:
        pred_logits: Predicted logits [B, C, H, W]
        target: Ground truth [B, H, W] with values 0 to C-1
        num_classes: Number of classes
        smooth: Smoothing factor to prevent division by zero
        class_weights: Optional [C] tensor with per-class weights

    Returns:
        Scalar Tversky loss (1 - Tversky index)
    """
    # Convert logits to probabilities
    pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]

    # One-hot encode target
    target_one_hot = F.one_hot(target, num_classes=num_classes)  # [B, H, W, C]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

    # Per-class Tversky parameters (alpha + beta = 1)
    # Higher beta = more penalty for false negatives (missing the class)
    device = pred_logits.device
    if USE_4_CLASS_SEGMENTATION:
        # 4-class: RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid
        alpha_per_class = torch.tensor([0.5, 0.4, 0.4, 0.5], device=device)  # FP penalty
        beta_per_class = torch.tensor([0.5, 0.6, 0.6, 0.5], device=device)   # FN penalty
    else:
        # 5-class: RNFL_GCL, INL_OPL, ONL, IS_OS, RPE_Choroid
        alpha_per_class = torch.tensor([0.5, 0.3, 0.2, 0.3, 0.5], device=device)  # FP penalty
        beta_per_class = torch.tensor([0.5, 0.7, 0.8, 0.7, 0.5], device=device)   # FN penalty (high for thin)

    # Compute per-class Tversky index
    tversky_per_class = []
    for c in range(num_classes):
        pred_c = pred_probs[:, c]  # [B, H, W]
        target_c = target_one_hot[:, c]  # [B, H, W]

        # True positives, false positives, false negatives
        TP = (pred_c * target_c).sum()
        FP = (pred_c * (1 - target_c)).sum()
        FN = ((1 - pred_c) * target_c).sum()

        alpha = alpha_per_class[c]
        beta = beta_per_class[c]

        # Tversky index: TP / (TP + alpha*FP + beta*FN)
        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        tversky_per_class.append(tversky)

    # Stack and apply class weights if provided
    tversky_tensor = torch.stack(tversky_per_class)  # [C]

    if class_weights is not None:
        # Weighted average
        weighted_tversky = (tversky_tensor * class_weights).sum() / class_weights.sum()
    else:
        weighted_tversky = tversky_tensor.mean()

    # Return 1 - Tversky as loss (so lower is better)
    return 1.0 - weighted_tversky


def segmentation_boundary_loss(pred_logits, target, num_classes=NUM_SEG_CLASSES):
    """
    Boundary-focused loss for thin layer segmentation.

    Instead of trying to segment thin layers (ONL = 4.5px avg), this loss
    focuses on correctly classifying BOUNDARY PIXELS where layer transitions occur.

    Key insight: For very thin layers like ONL, if we correctly detect the
    upper and lower boundaries, we've captured the essential structure.

    Args:
        pred_logits: Predicted logits [B, C, H, W]
        target: Ground truth [B, H, W] with values 0 to C-1

    Returns:
        Scalar boundary loss
    """
    device = pred_logits.device
    B, C, H, W = pred_logits.shape

    # Convert logits to probabilities
    pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]

    # Detect boundary pixels in ground truth (where class changes vertically)
    # Boundaries are critical for thin layers like ONL
    target_shifted = torch.roll(target, shifts=1, dims=1)  # Shift down
    boundary_mask = (target != target_shifted).float()  # [B, H, W]
    boundary_mask[:, 0, :] = 0  # First row has no upper neighbor

    # Also detect horizontal boundaries (less common but helps)
    target_shifted_h = torch.roll(target, shifts=1, dims=2)
    boundary_mask_h = (target != target_shifted_h).float()
    boundary_mask_h[:, :, 0] = 0

    # Combine vertical and horizontal boundaries
    boundary_mask = torch.clamp(boundary_mask + boundary_mask_h, 0, 1)

    # Count boundary pixels
    n_boundary = boundary_mask.sum() + 1e-8

    # Get predicted class at each pixel
    pred_class = pred_logits.argmax(dim=1)  # [B, H, W]

    # Compute accuracy only at boundary pixels
    correct_at_boundary = ((pred_class == target).float() * boundary_mask).sum()
    boundary_accuracy = correct_at_boundary / n_boundary

    # Also compute soft loss at boundaries using cross-entropy
    # Reshape for cross-entropy
    pred_flat = pred_logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    target_flat = target.reshape(-1)  # [B*H*W]
    boundary_flat = boundary_mask.reshape(-1)  # [B*H*W]

    # Per-pixel cross-entropy
    ce_per_pixel = F.cross_entropy(pred_flat, target_flat, reduction='none')  # [B*H*W]

    # Weight by boundary mask (boundary pixels get full weight, others get small weight)
    # This ensures we still learn non-boundary pixels but prioritize boundaries
    weights = boundary_flat * 5.0 + (1 - boundary_flat) * 0.2
    weighted_ce = (ce_per_pixel * weights).sum() / weights.sum()

    # Return weighted CE as loss (lower boundary accuracy = higher loss)
    return weighted_ce


def focal_loss(pred_logits, target, gamma=2.0, alpha=None, num_classes=NUM_SEG_CLASSES, ohem_ratio=0.5):
    """
    Focal Loss with Online Hard Example Mining (OHEM).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples, focuses on hard ones (thin layers).
    OHEM: Only keeps the hardest k% of pixels for loss computation.

    Args:
        pred_logits: Predicted logits [B, C, H, W]
        target: Ground truth [B, H, W] with values 0 to C-1
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Per-class weights [C] (higher for rare classes)
        num_classes: Number of classes
        ohem_ratio: Fraction of hardest pixels to keep (0.5 = top 50%)

    Returns:
        Scalar focal loss
    """
    device = pred_logits.device

    # Default alpha: Balanced weights for all learnable layers
    if alpha is None:
        if USE_4_CLASS_SEGMENTATION:
            # 4-class: RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid
            alpha = torch.tensor([1.0, 2.0, 2.0, 1.0], device=device)
        else:
            # 5-class: RNFL_GCL, INL_OPL, ONL, IS_OS, RPE_Choroid
            alpha = torch.tensor([0.1, 5.0, 10.0, 5.0, 0.1], device=device)

    # Convert logits to probabilities
    pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]

    # Gather probability of correct class for each pixel
    target_flat = target.view(-1)  # [B*H*W]
    pred_flat = pred_probs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)  # [B*H*W, C]

    # Get p_t (probability of true class)
    p_t = pred_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # [B*H*W]

    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1.0 - p_t) ** gamma

    # Compute cross-entropy: -log(p_t)
    ce = -torch.log(p_t + 1e-8)

    # Apply alpha weights per class
    alpha_t = alpha.gather(0, target_flat)  # [B*H*W]

    # Focal loss = alpha_t * (1 - p_t)^gamma * CE
    focal = alpha_t * focal_weight * ce

    # OHEM: Online Hard Example Mining
    # Only keep the hardest k% of pixels (highest loss)
    # This forces the model to focus on thin layer pixels where it's making mistakes
    if ohem_ratio < 1.0:
        n_pixels = focal.numel()
        n_keep = max(1, int(n_pixels * ohem_ratio))

        # Sort by loss (descending) and keep top k
        sorted_loss, _ = torch.sort(focal, descending=True)
        threshold = sorted_loss[n_keep - 1]

        # Mask: only keep pixels with loss >= threshold
        hard_mask = (focal >= threshold).float()
        focal = focal * hard_mask

        # Average over kept pixels only
        return focal.sum() / (hard_mask.sum() + 1e-8)

    return focal.mean()


def combined_seg_loss(pred_logits, target, ce_weight=None, dice_weight=None,
                      lambda_dice=0.5, num_classes=NUM_SEG_CLASSES, use_focal=True):
    """
    Combined Focal/CE + Dice loss for better segmentation.

    Focal loss (default): Down-weights easy examples, focuses on hard thin layers.
    CE loss (fallback): Standard pixel-wise classification.
    Dice loss: Directly optimizes overlap, better for class imbalance.

    Args:
        pred_logits: Predicted logits [B, C, H, W]
        target: Ground truth [B, H, W]
        ce_weight: Class weights for CE loss (ignored if use_focal=True)
        dice_weight: Class weights for Dice loss
        lambda_dice: Weight for Dice loss (0.5 = equal pixel/region loss)
        num_classes: Number of classes
        use_focal: Use Focal Loss instead of CE (better for thin layers)

    Returns:
        Combined loss scalar
    """
    device = pred_logits.device

    # NOTE: We don't clamp logits here - instead we add penalty terms later
    # Clamping caused the model to switch from RNFL to RPE dominance

    # Pixel-wise loss: Focal (default) or CE
    if use_focal:
        if USE_4_CLASS_SEGMENTATION:
            # 4-CLASS SCHEME: RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid
            # With ONL merged into INL_OPL, all layers should be learnable
            # Use balanced weights to ensure all 4 layers learn
            focal_alpha = torch.tensor([1.0, 3.0, 2.0, 1.0], device=device)
        else:
            # Original 5-class scheme (ONL often collapses)
            focal_alpha = torch.tensor([0.01, 30.0, 60.0, 45.0, 0.01], device=device)
        pixel_loss = focal_loss(pred_logits, target, gamma=2.0, alpha=focal_alpha,
                                num_classes=num_classes)
    else:
        pixel_loss = F.cross_entropy(pred_logits, target, weight=ce_weight)

    # Dice weights
    if USE_4_CLASS_SEGMENTATION:
        # 4-CLASS SCHEME: Balanced weights for all layers
        # Layer order: RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid
        # INL_OPL_ONL is now ~3-4% of pixels (more balanced)
        boosted_dice_weight = torch.tensor([1.0, 4.0, 3.0, 1.0], device=device)
    else:
        # Original 5-class scheme
        boosted_dice_weight = torch.tensor([0.01, 40.0, 80.0, 60.0, 0.01], device=device)

    # If dice_weight provided, multiply with boost
    if dice_weight is not None:
        boosted_dice_weight = boosted_dice_weight * dice_weight

    # Dice loss with boosted weights
    dice_loss_val = soft_dice_loss(pred_logits, target, num_classes=num_classes,
                                class_weights=boosted_dice_weight)

    # CLASS DISTRIBUTION REGULARIZATION
    # Penalize the model when predictions are too concentrated on one class
    pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]
    pred_dist = pred_probs.mean(dim=(0, 2, 3))  # [C] - average class probabilities

    # Target distribution based on ground truth analysis
    if USE_4_CLASS_SEGMENTATION:
        # 4-class: RNFL~65%, INL_OPL_ONL~4%, IS_OS~4%, RPE~27%
        target_dist = torch.tensor([0.65, 0.04, 0.04, 0.27], device=device)
    else:
        # 5-class: RNFL~50%, INL~2%, ONL~1%, IS_OS~2%, RPE~45%
        target_dist = torch.tensor([0.50, 0.02, 0.01, 0.02, 0.45], device=device)

    # KL divergence to penalize deviation from expected distribution
    # Add small epsilon to avoid log(0)
    class_dist_reg = F.kl_div(
        (pred_dist + 1e-8).log(),
        target_dist,
        reduction='sum'
    )

    # ANY CLASS DOMINANCE PENALTY (not just RNFL)
    if USE_4_CLASS_SEGMENTATION:
        # 4-class expected max
        expected_max = torch.tensor([0.70, 0.08, 0.08, 0.32], device=device)
    else:
        # 5-class expected max
        expected_max = torch.tensor([0.55, 0.05, 0.03, 0.05, 0.50], device=device)
    class_excess = F.relu(pred_dist - expected_max)  # How much each class exceeds limit
    dominance_penalty = (class_excess ** 2).sum() * 20.0  # Quadratic penalty

    # MAXIMUM ENTROPY REGULARIZATION
    # Penalize low entropy (high confidence) predictions
    # This forces the model to be less certain and consider all classes
    entropy_per_pixel = -(pred_probs * (pred_probs + 1e-8).log()).sum(dim=1)  # [B, H, W]
    max_entropy = np.log(num_classes)  # Maximum entropy for num_classes
    avg_entropy = entropy_per_pixel.mean()
    # Penalize when entropy is too low (model too confident)
    entropy_penalty = F.relu(0.5 * max_entropy - avg_entropy) * 2.0  # Want entropy > 50% of max

    # THIN LAYER EXISTENCE PENALTY (MODERATE)
    # If thin layer predictions are too low, penalize moderately
    if USE_4_CLASS_SEGMENTATION:
        thin_layer_probs = pred_dist[1:3]  # INL_OPL_ONL, IS_OS
    else:
        thin_layer_probs = pred_dist[1:4]  # INL, ONL, IS_OS
    thin_layer_min = 0.005  # At least 0.5% for each thin layer
    thin_layer_penalty = F.relu(thin_layer_min - thin_layer_probs).sum() * 5.0

    # Combined: SIMPLE Focal + Dice only (removed all penalties to stabilize training)
    # Previous issue: Training was oscillating with too many competing loss terms
    # Solution: Just use weighted Focal + Dice, let the class weights do the work
    combined = (1.0 - lambda_dice) * pixel_loss + lambda_dice * dice_loss_val

    return combined, pixel_loss, dice_loss_val


def compute_per_layer_psnr(denoised, clean, seg_mask, num_classes=NUM_SEG_CLASSES):
    """
    Compute PSNR for each anatomical layer.

    Args:
        denoised: Denoised images [B, 1, H, W]
        clean: Clean images [B, 1, H, W]
        seg_mask: Segmentation mask [B, H, W] with values 0-4

    Returns:
        Dict of per-layer PSNR values
    """
    per_layer_psnr = {}

    for c in range(num_classes):
        # Create mask for this layer
        layer_mask = (seg_mask == c).unsqueeze(1).float()  # [B, 1, H, W]

        # Count pixels in this layer
        n_pixels = layer_mask.sum()
        if n_pixels < 100:  # Skip if too few pixels
            per_layer_psnr[LAYER_NAMES[c]] = None
            continue

        # Compute MSE only for this layer
        diff_sq = (denoised - clean) ** 2 * layer_mask
        mse = diff_sq.sum() / n_pixels

        if mse > 0:
            psnr = 10 * torch.log10(1.0 / mse)
            per_layer_psnr[LAYER_NAMES[c]] = psnr.item()
        else:
            per_layer_psnr[LAYER_NAMES[c]] = 50.0  # Perfect reconstruction

    return per_layer_psnr


# Shared cache for sobel kernels (avoid recreating on every call)
# Limited to 2 entries (CPU + 1 GPU device) to prevent memory accumulation
_SOBEL_KERNEL_CACHE = {}
_SOBEL_CACHE_MAX_SIZE = 2


def _get_sobel_kernels(dtype, device):
    """Get cached sobel kernels for given dtype/device."""
    cache_key = (dtype, device)
    if cache_key not in _SOBEL_KERNEL_CACHE:
        # Clear cache if it's getting too large (prevents memory leak when switching devices)
        if len(_SOBEL_KERNEL_CACHE) >= _SOBEL_CACHE_MAX_SIZE:
            _SOBEL_KERNEL_CACHE.clear()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=dtype, device=device).view(1, 1, 3, 3)
        _SOBEL_KERNEL_CACHE[cache_key] = (sobel_x, sobel_y)
    return _SOBEL_KERNEL_CACHE[cache_key]


def clear_sobel_cache():
    """Clear the sobel kernel cache to free memory."""
    _SOBEL_KERNEL_CACHE.clear()


def compute_edge_preservation_index(denoised, clean):
    """
    Compute Edge Preservation Index (EPI).
    EPI = sum(|grad(denoised)|) / sum(|grad(clean)|)

    Values close to 1.0 indicate good edge preservation.
    < 1.0 means edges are blurred, > 1.0 means edges are sharpened/artifacts.
    """
    # Get cached sobel kernels
    sobel_x, sobel_y = _get_sobel_kernels(denoised.dtype, denoised.device)

    # Compute gradients using F.pad (no module creation)
    denoised_padded = F.pad(denoised, (1, 1, 1, 1), mode='reflect')
    clean_padded = F.pad(clean, (1, 1, 1, 1), mode='reflect')

    grad_denoised_x = F.conv2d(denoised_padded, sobel_x)
    grad_denoised_y = F.conv2d(denoised_padded, sobel_y)
    grad_denoised = torch.sqrt(grad_denoised_x**2 + grad_denoised_y**2 + 1e-8)

    grad_clean_x = F.conv2d(clean_padded, sobel_x)
    grad_clean_y = F.conv2d(clean_padded, sobel_y)
    grad_clean = torch.sqrt(grad_clean_x**2 + grad_clean_y**2 + 1e-8)

    epi = grad_denoised.sum() / (grad_clean.sum() + 1e-8)
    return epi.item()


def compute_boundary_metrics(denoised, clean, seg_mask, num_classes=NUM_SEG_CLASSES):
    """
    Compute metrics specifically at layer boundaries.

    Returns:
        boundary_psnr: PSNR at boundary regions
        boundary_ssim: SSIM at boundary regions
        boundary_sharpness: Average gradient magnitude at boundaries
    """
    # Find boundary pixels (where neighboring pixels have different labels)
    # Use morphological gradient: dilate - erode
    seg_float = seg_mask.float().unsqueeze(1)  # [B, 1, H, W]

    # Create boundary mask using max pooling (dilation) and comparison
    kernel_size = 3
    dilated = F.max_pool2d(seg_float, kernel_size, stride=1, padding=1)
    eroded = -F.max_pool2d(-seg_float, kernel_size, stride=1, padding=1)
    boundary_mask = (dilated != eroded).float()  # [B, 1, H, W]

    n_boundary = boundary_mask.sum()
    if n_boundary < 100:
        return None, None, None, None

    # Boundary PSNR
    diff_sq = (denoised - clean) ** 2 * boundary_mask
    mse = diff_sq.sum() / n_boundary
    if mse > 0:
        boundary_psnr = 10 * torch.log10(1.0 / mse).item()
    else:
        boundary_psnr = 50.0

    # Boundary sharpness (gradient magnitude at boundaries)
    # Use cached sobel kernel and F.pad (no module creation)
    _, sobel_y = _get_sobel_kernels(denoised.dtype, denoised.device)

    denoised_padded = F.pad(denoised, (1, 1, 1, 1), mode='reflect')
    clean_padded = F.pad(clean, (1, 1, 1, 1), mode='reflect')

    grad_denoised = torch.abs(F.conv2d(denoised_padded, sobel_y))
    grad_clean = torch.abs(F.conv2d(clean_padded, sobel_y))

    sharpness_denoised = (grad_denoised * boundary_mask).sum() / n_boundary
    sharpness_clean = (grad_clean * boundary_mask).sum() / n_boundary

    # Sharpness ratio (1.0 = same sharpness as clean)
    sharpness_ratio = (sharpness_denoised / (sharpness_clean + 1e-8)).item()

    return boundary_psnr, sharpness_ratio, sharpness_denoised.item(), sharpness_clean.item()


def compute_cnr_per_layer(denoised, seg_mask, num_classes=NUM_SEG_CLASSES):
    """
    Compute Contrast-to-Noise Ratio between adjacent layers.
    CNR = |mean(layer_i) - mean(layer_j)| / sqrt(var(layer_i) + var(layer_j))

    Higher CNR = better layer distinguishability (clinically important).
    """
    cnr_values = {}
    layer_stats = {}

    # Compute mean and std for each layer
    for c in range(num_classes):
        layer_mask = (seg_mask == c).unsqueeze(1).float()
        n_pixels = layer_mask.sum()
        if n_pixels < 100:
            layer_stats[c] = None
            continue

        layer_pixels = denoised * layer_mask
        mean_val = layer_pixels.sum() / n_pixels
        var_val = ((denoised - mean_val) ** 2 * layer_mask).sum() / n_pixels
        layer_stats[c] = {'mean': mean_val.item(), 'std': torch.sqrt(var_val).item()}

    # Compute CNR between adjacent layers
    for c in range(num_classes - 1):
        if layer_stats[c] is None or layer_stats[c+1] is None:
            continue

        mean_diff = abs(layer_stats[c]['mean'] - layer_stats[c+1]['mean'])
        std_pooled = np.sqrt(layer_stats[c]['std']**2 + layer_stats[c+1]['std']**2 + 1e-8)
        cnr = mean_diff / std_pooled

        cnr_values[f"{LAYER_NAMES[c]}_{LAYER_NAMES[c+1]}"] = cnr

    # Average CNR
    if cnr_values:
        avg_cnr = np.mean(list(cnr_values.values()))
    else:
        avg_cnr = None

    return avg_cnr, cnr_values


def compute_per_layer_ssim(denoised, clean, seg_mask, num_classes=NUM_SEG_CLASSES):
    """
    Compute SSIM for each anatomical layer using masked regions.

    Args:
        denoised: Denoised images [B, 1, H, W]
        clean: Clean images [B, 1, H, W]
        seg_mask: Segmentation mask [B, H, W] with values 0-4

    Returns:
        Dict of per-layer SSIM values
    """
    per_layer_ssim = {}

    for c in range(num_classes):
        # Create mask for this layer
        layer_mask = (seg_mask == c).unsqueeze(1).float()  # [B, 1, H, W]

        # Count pixels in this layer
        n_pixels = layer_mask.sum()
        if n_pixels < 100:  # Skip if too few pixels
            per_layer_ssim[LAYER_NAMES[c]] = None
            continue

        # Compute SSIM components for masked region
        # Use simplified SSIM: (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute means within mask
        mu_x = (denoised * layer_mask).sum() / n_pixels
        mu_y = (clean * layer_mask).sum() / n_pixels

        # Compute variances within mask
        sigma_x_sq = ((denoised - mu_x) ** 2 * layer_mask).sum() / n_pixels
        sigma_y_sq = ((clean - mu_y) ** 2 * layer_mask).sum() / n_pixels
        sigma_xy = ((denoised - mu_x) * (clean - mu_y) * layer_mask).sum() / n_pixels

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2))

        per_layer_ssim[LAYER_NAMES[c]] = ssim.item()

    return per_layer_ssim


def compute_per_layer_dice(pred_logits, target, num_classes=NUM_SEG_CLASSES):
    """Compute per-layer Dice scores."""
    pred = pred_logits.argmax(dim=1)
    per_layer_dice = {}

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
            per_layer_dice[LAYER_NAMES[c]] = dice.item()
        else:
            per_layer_dice[LAYER_NAMES[c]] = None

    return per_layer_dice


def _pearson_corr_efficient(x, y):
    """
    Compute Pearson correlation efficiently.
    Uses running stats to avoid creating large intermediate tensors.
    Returns 0.0 if either signal has near-zero variance (undefined correlation).
    """
    # Flatten if needed
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)

    # Compute means
    x_mean = x_flat.mean()
    y_mean = y_flat.mean()

    # Compute correlation using the formula
    x_centered = x_flat - x_mean
    y_centered = y_flat - y_mean

    # Compute sum of squares
    x_ss = (x_centered ** 2).sum()
    y_ss = (y_centered ** 2).sum()

    # Check for near-zero variance (correlation undefined)
    min_variance = 1e-8
    if x_ss < min_variance or y_ss < min_variance:
        return 0.0  # Return 0 for undefined correlation

    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt(x_ss * y_ss)

    return (numerator / denominator).item()


def compute_adaptive_denoising_metrics(noisy, denoised, features):
    """
    Compute metrics to verify adaptive per-pixel denoising is working.

    Key metrics:
    1. Refinement spatial variance: Should be non-zero (not uniform denoising)
    2. Noise-refinement correlation: High noise regions should have stronger refinement
    3. Denoising strength by noise level: Compare high-noise vs low-noise regions

    Args:
        noisy: Noisy input [B, 1, H, W]
        denoised: Denoised output [B, 1, H, W]
        features: Dict from model with return_features=True

    Returns:
        Dict of adaptive denoising metrics
    """
    metrics = {}

    # 1. Refinement spatial variance (proves it's not uniform)
    refinement = features['refinement']  # [B, 1, H, W]
    refinement_abs = refinement.abs()

    # Spatial variance of refinement magnitude
    ref_mean = refinement_abs.mean(dim=[2, 3], keepdim=True)
    ref_var = ((refinement_abs - ref_mean) ** 2).mean(dim=[2, 3])
    metrics['refinement_spatial_var'] = ref_var.mean().item()

    # Coefficient of variation of refinement (normalized measure)
    # Handle dimension carefully: ref_var is [B], ref_mean is [B, 1, 1, 1]
    ref_std = torch.sqrt(ref_var + 1e-8)  # [B]
    ref_mean_squeezed = ref_mean.view(-1)  # [B]
    ref_cv = (ref_std / (ref_mean_squeezed + 1e-8)).mean().item()
    metrics['refinement_cv'] = ref_cv

    # 2. Correlation between noise features and refinement strength
    raw_features = features['raw_features']

    # Use coefficient of variation as noise indicator (high = speckle-like noise)
    coef_var = raw_features['coef_variation']  # [B, 1, H, W]
    local_std = raw_features['local_std']  # [B, 1, H, W]

    # Compute correlation efficiently (views, no copy)
    metrics['corr_coef_var_refinement'] = _pearson_corr_efficient(coef_var, refinement_abs)
    metrics['corr_local_std_refinement'] = _pearson_corr_efficient(local_std, refinement_abs)

    # 3. Denoising strength in high-noise vs low-noise regions
    # Define high/low noise based on local_std
    noise_median = local_std.median()

    high_noise_mask = (local_std > noise_median).float()
    low_noise_mask = (local_std <= noise_median).float()

    # Denoising residual = |noisy - denoised|
    denoise_residual = (noisy - denoised).abs()

    # Average residual in high vs low noise regions
    high_mask_sum = high_noise_mask.sum()
    low_mask_sum = low_noise_mask.sum()

    high_noise_residual = (denoise_residual * high_noise_mask).sum() / (high_mask_sum + 1e-8)
    low_noise_residual = (denoise_residual * low_noise_mask).sum() / (low_mask_sum + 1e-8)

    metrics['residual_high_noise'] = high_noise_residual.item()
    metrics['residual_low_noise'] = low_noise_residual.item()
    metrics['residual_ratio'] = high_noise_residual.item() / (low_noise_residual.item() + 1e-8)

    # 4. Refinement strength in high vs low noise regions
    high_noise_refinement = (refinement_abs * high_noise_mask).sum() / (high_mask_sum + 1e-8)
    low_noise_refinement = (refinement_abs * low_noise_mask).sum() / (low_mask_sum + 1e-8)

    metrics['refinement_high_noise'] = high_noise_refinement.item()
    metrics['refinement_low_noise'] = low_noise_refinement.item()
    metrics['refinement_ratio'] = high_noise_refinement.item() / (low_noise_refinement.item() + 1e-8)

    # 5. Per-noise-type feature activation (shows different noise types detected)
    metrics['avg_coef_variation'] = coef_var.mean().item()
    metrics['avg_local_std'] = local_std.mean().item()
    metrics['avg_horizontal_ratio'] = raw_features['horizontal_ratio'].mean().item()
    metrics['avg_signal_var_corr'] = raw_features['signal_var_corr'].mean().item()

    return metrics


def compute_boundary_loss(denoised, clean, seg_mask):
    """
    Compute boundary preservation loss.

    Encourages the model to preserve edge sharpness at layer boundaries.
    Uses gradient magnitude comparison at boundary regions.

    Args:
        denoised: Denoised output [B, 1, H, W]
        clean: Clean target [B, 1, H, W]
        seg_mask: Segmentation mask [B, H, W]

    Returns:
        Boundary preservation loss (lower = better edge preservation)
    """
    # Find boundary pixels using morphological gradient
    seg_float = seg_mask.float().unsqueeze(1)  # [B, 1, H, W]
    kernel_size = 3
    dilated = F.max_pool2d(seg_float, kernel_size, stride=1, padding=1)
    eroded = -F.max_pool2d(-seg_float, kernel_size, stride=1, padding=1)
    boundary_mask = (dilated != eroded).float()  # [B, 1, H, W]

    n_boundary = boundary_mask.sum()
    if n_boundary < 10:
        # Return zero loss that's on the same device, no grad needed
        return denoised.new_zeros(1).squeeze()

    # Use shared cached sobel kernel
    _, sobel_y = _get_sobel_kernels(denoised.dtype, denoised.device)

    # Compute vertical gradients (most important for OCT layer boundaries)
    # Use F.pad instead of creating nn.ReflectionPad2d each time
    denoised_padded = F.pad(denoised, (1, 1, 1, 1), mode='reflect')
    clean_padded = F.pad(clean, (1, 1, 1, 1), mode='reflect')

    grad_denoised = torch.abs(F.conv2d(denoised_padded, sobel_y))
    grad_clean = torch.abs(F.conv2d(clean_padded, sobel_y))

    # Loss: penalize difference in gradient magnitude at boundaries
    # This encourages preserving edge sharpness
    boundary_grad_diff = torch.abs(grad_denoised - grad_clean) * boundary_mask
    loss = boundary_grad_diff.sum() / (n_boundary + 1e-8)

    return loss


def compute_noise_correlation_loss(noise_gate, noise_features):
    """
    Loss that encourages the noise gate to correlate positively with noise level.

    This explicitly trains the model to apply stronger refinement in noisier regions.

    Args:
        noise_gate: Learned noise gate [B, 1, H, W] (0-1 values)
        noise_features: Dict containing 'coef_variation' and 'local_std'

    Returns:
        Loss value (lower = better correlation with noise)
    """
    # Use coefficient of variation as primary noise indicator
    # High coef_var = speckle-like noise (signal-dependent)
    coef_var = noise_features['coef_variation']  # [B, 1, H, W]

    # Use local_std as secondary noise indicator
    local_std = noise_features['local_std']  # [B, 1, H, W]

    # Combine noise indicators (both normalized)
    coef_var_norm = coef_var / (coef_var.mean() + 1e-8)
    local_std_norm = local_std / (local_std.mean() + 1e-8)
    noise_level = 0.7 * coef_var_norm + 0.3 * local_std_norm  # Weighted combination

    # Normalize noise_gate
    gate_norm = noise_gate / (noise_gate.mean() + 1e-8)

    # Compute correlation loss
    # We want: high noise → high gate, low noise → low gate
    # Loss = -correlation (minimize to maximize correlation)

    # Center both signals
    noise_centered = noise_level - noise_level.mean()
    gate_centered = gate_norm - gate_norm.mean()

    # Pearson correlation
    numerator = (noise_centered * gate_centered).mean()
    denominator = torch.sqrt(
        (noise_centered ** 2).mean() * (gate_centered ** 2).mean() + 1e-8
    )
    correlation = numerator / denominator

    # Loss: negative correlation (we want to maximize correlation)
    # Add a term to encourage gate variance (avoid trivial solution of constant gate)
    gate_variance = noise_gate.var()
    variance_penalty = torch.exp(-10 * gate_variance)  # Penalize low variance

    loss = -correlation + 0.1 * variance_penalty

    return loss, correlation.item()


def compute_layer_gate_diversity_loss(layer_gates, seg_probs):
    """
    Loss that encourages different layers to have different noise gate responses.

    KEY TMI CONTRIBUTION: This forces the model to learn layer-specific noise profiles.

    The loss penalizes similarity between layer gates - we want each anatomical layer
    to develop its own unique response to noise features.

    Args:
        layer_gates: [B, 5, H, W] - gate outputs for each layer
        seg_probs: [B, 5, H, W] - segmentation probabilities

    Returns:
        diversity_loss: Scalar loss (lower = more diverse gates)
        diversity_stats: Dict with per-layer pair similarities
    """
    B, n_layers, H, W = layer_gates.shape

    # Compute mean gate value for each layer (weighted by seg_probs)
    layer_means = []
    for i in range(n_layers):
        mask = seg_probs[:, i:i+1, :, :]
        mask_sum = mask.sum() + 1e-8
        gate_i = layer_gates[:, i:i+1, :, :]
        mean_gate = (gate_i * mask).sum() / mask_sum
        layer_means.append(mean_gate)

    layer_means = torch.stack(layer_means)  # [5]

    # Compute pairwise similarities - we want to minimize these
    similarity_loss = 0
    n_pairs = 0
    diversity_stats = {}

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            # L2 distance between layer means (maximize distance = minimize negative)
            diff = torch.abs(layer_means[i] - layer_means[j])
            # We want to maximize difference, so loss = -diff (or exp(-diff))
            pair_sim = torch.exp(-5 * diff)  # Close to 1 if similar, close to 0 if different
            similarity_loss += pair_sim
            n_pairs += 1

            with torch.no_grad():
                diversity_stats[f'{LAYER_NAMES[i]}_vs_{LAYER_NAMES[j]}'] = diff.item()

    # Average similarity (we want this to be low)
    diversity_loss = similarity_loss / max(n_pairs, 1)

    # Also encourage each layer to have meaningful gate variance (not constant)
    variance_penalty = 0
    for i in range(n_layers):
        gate_i = layer_gates[:, i:i+1, :, :]
        mask = seg_probs[:, i:i+1, :, :]
        mask_sum = mask.sum() + 1e-8

        if mask_sum > 100:  # Only if layer has enough pixels
            gate_values = gate_i * mask
            gate_mean = gate_values.sum() / mask_sum
            gate_var = ((gate_i - gate_mean) ** 2 * mask).sum() / mask_sum
            # Penalize low variance
            variance_penalty += torch.exp(-20 * gate_var)

    variance_penalty = variance_penalty / n_layers

    total_loss = diversity_loss + 0.1 * variance_penalty

    return total_loss, diversity_stats


def compute_symbolic_ordering_loss(seg_logits, margin=2.0):
    """
    Neuro-Symbolic Ordering Loss: Penalizes anatomical layer ordering violations.

    KEY TMI CONTRIBUTION: This loss enforces hard anatomical constraints in a differentiable way.

    In OCT images, retinal layers MUST follow a strict top-to-bottom ordering:
        Layer 0 (RNFL_GCL) - topmost
        Layer 1 (INL_OPL)
        Layer 2 (ONL)
        Layer 3 (IS_OS)
        Layer 4 (RPE_Choroid) - bottommost

    The loss computes the expected y-position (row) for each layer class based on
    segmentation probabilities, then penalizes cases where a lower layer has a
    smaller y-position than an upper layer (ordering violation).

    Args:
        seg_logits: [B, C, H, W] - segmentation logits (5 classes)
        margin: Minimum expected pixel distance between adjacent layers (default 2.0).
                Larger margin = stricter enforcement of layer separation.

    Returns:
        ordering_loss: Scalar loss (0 if all layers correctly ordered, >0 if violations)
        violation_stats: Dict with per-pair violation statistics

    Mathematical formulation:
        For each class c, compute expected y-position:
            E[y|c] = sum_y (y * P(class=c at row y)) / sum_y P(class=c at row y)

        Ordering constraint for adjacent layers:
            E[y|c] + margin < E[y|c+1]  for all c in [0, 1, 2, 3]

        Loss = sum of ReLU violations:
            L = sum_{c=0}^{3} max(0, E[y|c] + margin - E[y|c+1])
    """
    B, C, H, W = seg_logits.shape
    device = seg_logits.device

    # Convert logits to probabilities
    probs = F.softmax(seg_logits, dim=1)  # [B, C, H, W]

    # Create y-coordinate grid (row indices)
    # y = 0 at top, y = H-1 at bottom
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    y_coords = y_coords.expand(B, 1, H, W)  # [B, 1, H, W]

    # Compute expected y-position for each class
    # E[y|c] = sum(y * P(c)) / sum(P(c)) for each column, then average across columns
    expected_y = []
    class_presence = []  # Track if class has significant presence

    for c in range(C):
        prob_c = probs[:, c:c+1, :, :]  # [B, 1, H, W]

        # Weighted y-position: sum over height dimension
        weighted_y = (prob_c * y_coords).sum(dim=2, keepdim=True)  # [B, 1, 1, W]
        prob_sum = prob_c.sum(dim=2, keepdim=True) + 1e-8  # [B, 1, 1, W]

        # Expected y for this class at each column
        exp_y_c = weighted_y / prob_sum  # [B, 1, 1, W]

        # Average across batch and width
        exp_y_c_mean = exp_y_c.mean()  # scalar
        expected_y.append(exp_y_c_mean)

        # Track class presence (for weighted loss)
        class_presence.append(prob_sum.mean())

    expected_y = torch.stack(expected_y)  # [C]
    class_presence = torch.stack(class_presence)  # [C]

    # Compute ordering violations
    # For each adjacent pair (c, c+1), check if E[y|c] + margin < E[y|c+1]
    # Violation occurs when E[y|c] + margin >= E[y|c+1]
    ordering_loss = torch.tensor(0.0, device=device)
    violation_stats = {}

    for c in range(C - 1):
        # Expected gap: E[y|c+1] - E[y|c] should be >= margin
        gap = expected_y[c + 1] - expected_y[c]

        # Violation if gap < margin (upper layer too close to or below lower layer)
        violation = F.relu(margin - gap)

        # Weight by presence of both classes (don't penalize missing classes)
        presence_weight = torch.min(class_presence[c], class_presence[c + 1])
        presence_weight = torch.clamp(presence_weight, 0.1, 1.0)

        weighted_violation = violation * presence_weight
        ordering_loss = ordering_loss + weighted_violation

        # Stats for monitoring
        with torch.no_grad():
            violation_stats[f'{LAYER_NAMES[c]}_to_{LAYER_NAMES[c+1]}'] = {
                'gap': gap.item(),
                'violation': violation.item(),
                'expected_y_upper': expected_y[c].item(),
                'expected_y_lower': expected_y[c + 1].item(),
            }

    # Normalize by number of pairs
    ordering_loss = ordering_loss / (C - 1)

    # Also add a smoothness term: penalize large jumps in expected position
    # This encourages continuous layer boundaries
    smoothness_loss = torch.tensor(0.0, device=device)
    for c in range(C - 1):
        # Penalize if gap is too large (suggests discontinuous segmentation)
        gap = expected_y[c + 1] - expected_y[c]
        max_reasonable_gap = H / C * 2  # Roughly 2x average layer thickness
        excess_gap = F.relu(gap - max_reasonable_gap)
        smoothness_loss = smoothness_loss + excess_gap * 0.01  # Small weight

    total_loss = ordering_loss + smoothness_loss

    return total_loss, violation_stats


# =============================================================================
# SLIDING WINDOW VALIDATION FOR FULL-IMAGE EVALUATION
# =============================================================================
# Provides stable, consistent evaluation by processing full images with
# overlapping patches. This eliminates variance from random patch selection.
# =============================================================================

def sliding_window_inference(model, image, patch_size=128, stride=64, device='cpu'):
    """
    Perform inference on full image using sliding window with overlap.

    Args:
        model: Model with forward(image) -> (denoised, seg_logits) or
               forward(image) -> (denoised, seg_logits, boundary_logits)
        image: Input image tensor [1, C, H, W]
        patch_size: Size of patches (default 128)
        stride: Stride between patches (default 64 for 50% overlap)
        device: Device for computation

    Returns:
        seg_output: Averaged segmentation logits [1, num_classes, H, W]
        boundary_output: Averaged boundary logits [1, 1, H, W] or None
        denoised_output: Averaged denoised image [1, C, H, W]
    """
    model.eval()
    B, C, H, W = image.shape

    # Pad image to be divisible by stride
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')

    _, _, H_pad, W_pad = image.shape

    # Determine number of output classes from model
    # Try a single forward pass to get output shapes
    with torch.no_grad():
        test_patch = image[:, :, :patch_size, :patch_size]
        test_output = model(test_patch)

        # Handle different output formats
        if isinstance(test_output, dict):
            has_boundary = 'boundary_logits' in test_output
            num_seg_classes = test_output['seg_logits'].shape[1]
        elif isinstance(test_output, tuple):
            if len(test_output) >= 3:
                # (denoised, seg_logits, boundary_logits)
                has_boundary = test_output[2] is not None
                num_seg_classes = test_output[1].shape[1]
            else:
                # (denoised, seg_logits) or (seg_logits, boundary_logits)
                has_boundary = False
                num_seg_classes = test_output[1].shape[1] if test_output[1].dim() == 4 else test_output[0].shape[1]
        else:
            has_boundary = False
            num_seg_classes = test_output.shape[1]

    # Initialize output tensors
    seg_output = torch.zeros(B, num_seg_classes, H_pad, W_pad, device=device)
    denoised_output = torch.zeros(B, C, H_pad, W_pad, device=device)
    if has_boundary:
        boundary_output = torch.zeros(B, 1, H_pad, W_pad, device=device)
    else:
        boundary_output = None
    count = torch.zeros(B, 1, H_pad, W_pad, device=device)

    with torch.no_grad():
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = image[:, :, y:y+patch_size, x:x+patch_size]

                output = model(patch)

                # Handle different output formats
                if isinstance(output, dict):
                    denoised = output.get('denoised', patch)
                    seg_logits = output['seg_logits']
                    boundary_logits = output.get('boundary_logits', None)
                elif isinstance(output, tuple):
                    if len(output) >= 3:
                        denoised, seg_logits, boundary_logits = output[0], output[1], output[2]
                    else:
                        denoised, seg_logits = output[0], output[1]
                        boundary_logits = None
                else:
                    seg_logits = output
                    denoised = patch
                    boundary_logits = None

                seg_output[:, :, y:y+patch_size, x:x+patch_size] += seg_logits
                denoised_output[:, :, y:y+patch_size, x:x+patch_size] += denoised
                if has_boundary and boundary_logits is not None:
                    boundary_output[:, :, y:y+patch_size, x:x+patch_size] += boundary_logits
                count[:, :, y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping regions
    seg_output = seg_output / count.clamp(min=1)
    denoised_output = denoised_output / count.clamp(min=1)
    if has_boundary:
        boundary_output = boundary_output / count.clamp(min=1)

    # Remove padding
    seg_output = seg_output[:, :, :H, :W]
    denoised_output = denoised_output[:, :, :H, :W]
    if has_boundary:
        boundary_output = boundary_output[:, :, :H, :W]

    return seg_output, boundary_output, denoised_output


def compute_boundary_dice(pred_boundary, target_boundary, threshold=0.5):
    """Compute Dice score for boundary detection.

    Args:
        pred_boundary: Predicted boundary logits [B, 1, H, W]
        target_boundary: Ground truth boundary mask [B, 1, H, W] or [B, H, W]

    Returns:
        Dice score (float)
    """
    pred_binary = (torch.sigmoid(pred_boundary) > threshold).float()
    if target_boundary.dim() == 3:
        target_boundary = target_boundary.unsqueeze(1)

    intersection = (pred_binary * target_boundary).sum()
    union = pred_binary.sum() + target_boundary.sum()

    if union > 0:
        return (2 * intersection / union).item()
    return 1.0 if intersection == 0 else 0.0


def train_epoch(model, loader, optimizer, device, epoch, lambda_seg=1.0, lambda_boundary=0.1,
                lambda_noise_corr=0.1, lambda_diversity=0.05, lambda_clinical_gate=0.1,
                lambda_uncertainty=0.1, lambda_pathology=0.1, lambda_sharpness=0.1,
                lambda_anatomical=0.1, lambda_confidence=0.1, lambda_dice=0.5,
                lambda_symbolic_ordering=0.1,
                lambda_boundary_detection=2.0, boundary_pos_weight=10.0,
                seg_class_weights=None, clinical_loss_fn=None, clinical_gate_loss_fn=None,
                uncertainty_loss_fn=None, pathology_loss_fn=None, boundary_sharpness_loss_fn=None,
                anatomical_loss_fn=None, confidence_loss_fn=None, use_cuap=False,
                loss_scaler=None, use_boundary_detection=False):
    """Train one epoch with multi-task loss including CUAP-OCT framework.

    KEY FIX: Dynamic Loss Scaling
    - Uses loss_scaler to normalize denoising and segmentation losses to equal magnitude
    - This prevents segmentation from dominating (seg_loss ~1770x larger than denoise_loss)
    - Without this, PSNR degrades over epochs as model over-optimizes for segmentation

    Args:
        seg_class_weights: Optional tensor [5] with class weights for segmentation loss.
                          If None, uses uniform weights. Higher weights for rarer classes
                          help prevent class collapse.
        clinical_loss_fn: Optional ClinicalWeightedMSELoss for clinical importance weighting.
                         If None, uses standard MSE loss.
        clinical_gate_loss_fn: Optional ClinicalGateDiversityLoss to encourage gates to
                              match clinical importance. If None, uses standard diversity loss.
        lambda_clinical_gate: Weight for clinical gate alignment loss (default 0.1).

        CUAP-OCT Framework (KEY TMI CONTRIBUTION):
        uncertainty_loss_fn: UncertaintyCalibrationLoss for clinically-meaningful uncertainty.
        pathology_loss_fn: PathologyPreservationLoss to preserve diagnostic features.
        boundary_sharpness_loss_fn: BoundarySharpnessLoss for accurate thickness measurement.
        anatomical_loss_fn: AnatomicalConsistencyLoss for anatomical plausibility.
        confidence_loss_fn: ConfidenceWeightedRefinementLoss for confidence-aware denoising.
        lambda_uncertainty: Weight for uncertainty calibration loss.
        lambda_pathology: Weight for pathology preservation loss.
        lambda_sharpness: Weight for boundary sharpness loss.
        lambda_anatomical: Weight for anatomical consistency loss.
        lambda_confidence: Weight for confidence-weighted refinement loss.
        lambda_symbolic_ordering: Weight for neuro-symbolic layer ordering loss (default 0.1).
                                  Enforces anatomical constraint: layers must follow top-to-bottom order.
        use_cuap: Whether to use the full CUAP framework.
    """
    model.train()

    total_denoise_loss = 0
    total_seg_loss = 0
    total_dice_loss = 0  # Dice loss component (for thin layer segmentation)
    total_seg_boundary_loss = 0  # Segmentation boundary loss (for thin layers like ONL)
    total_boundary_loss = 0
    total_noise_corr_loss = 0
    total_diversity_loss = 0  # NEW: layer gate diversity
    total_clinical_gate_loss = 0  # NEW: clinical gate alignment
    total_uncertainty_loss = 0  # CUAP: uncertainty calibration
    total_pathology_loss = 0  # CUAP: pathology preservation
    total_sharpness_loss = 0  # CUAP: boundary sharpness
    total_anatomical_loss = 0  # CUAP: anatomical consistency
    total_confidence_loss = 0  # CUAP: confidence-weighted refinement
    total_symbolic_ordering_loss = 0  # Neuro-symbolic: layer ordering constraint
    total_deep_supervision_loss = 0  # Deep supervision: auxiliary decoder losses
    total_boundary_aware_loss = 0  # Boundary-aware: boundary detection for thin layers
    total_is_os_boundary_loss = 0  # IS_OS boundary detection (V4 approach)
    total_is_os_boundary_dice = 0  # IS_OS boundary dice for monitoring
    total_gate_corr = 0
    total_loss = 0
    total_dice = 0
    num_batches = 0

    # Create BoundaryAwareLoss instance if using boundary-aware segmenter
    boundary_aware_loss_fn = None
    if USE_BOUNDARY_AWARE_SEGMENTER:
        # IS_OS is class 2 (thin layer needing special attention)
        boundary_aware_loss_fn = BoundaryAwareLoss(
            num_classes=NUM_SEG_CLASSES,
            thin_layer_ids=[2],  # IS_OS
            aux_weights=DEEP_SUPERVISION_WEIGHTS,
            boundary_weight=BOUNDARY_LOSS_WEIGHT
        ).to(device)

    # Track per-layer gate means for monitoring (with counts for proper averaging)
    layer_gate_means = {name: 0.0 for name in LAYER_NAMES}
    layer_gate_counts = {name: 0 for name in LAYER_NAMES}

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        seg_mask = batch['seg_mask'].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward with features (to get layer-specific gates)
        denoised, seg_logits, features = model(noisy, return_features=True)

        # Multi-task loss with boundary preservation
        # Use clinical weighted loss if provided, otherwise standard MSE
        if clinical_loss_fn is not None:
            denoise_loss = clinical_loss_fn(denoised, clean, seg_mask)
        else:
            denoise_loss = F.mse_loss(denoised, clean)

        # Combined CE + Dice loss for better thin layer segmentation
        if lambda_dice > 0:
            seg_loss, ce_loss_val, dice_loss_val = combined_seg_loss(
                seg_logits, seg_mask,
                ce_weight=seg_class_weights,
                dice_weight=seg_class_weights,
                lambda_dice=lambda_dice,
                num_classes=NUM_SEG_CLASSES
            )
        else:
            seg_loss = F.cross_entropy(seg_logits, seg_mask, weight=seg_class_weights)
            ce_loss_val = seg_loss
            dice_loss_val = torch.tensor(0.0, device=device)

        # SEGMENTATION BOUNDARY LOSS (KEY FOR THIN LAYERS)
        # ONL is only 4.5 pixels tall on average - boundary detection is more effective
        # than trying to fill in all pixels of a 4-pixel-tall region
        seg_boundary_loss = segmentation_boundary_loss(seg_logits, seg_mask, num_classes=NUM_SEG_CLASSES)

        # DEEP SUPERVISION + BOUNDARY-AWARE LOSS (KEY ENHANCEMENT)
        # When using BoundaryAwareSegmenter, compute auxiliary losses for better thin layer segmentation
        deep_supervision_loss = torch.tensor(0.0, device=device)
        boundary_aware_loss = torch.tensor(0.0, device=device)
        if boundary_aware_loss_fn is not None and features.get('seg_aux_outputs') is not None:
            # Get auxiliary outputs and boundary predictions from model
            aux_outputs = features['seg_aux_outputs']
            boundary_logits = features.get('seg_boundary_logits')

            # Build outputs dict as expected by BoundaryAwareLoss
            seg_outputs = {
                'seg_probs': features['seg_probs_raw'],  # Main predictions
                'seg_logits': seg_logits,
                'aux_outputs': aux_outputs,
                'boundary_logits': boundary_logits
            }

            # Compute boundary-aware loss (includes deep supervision + boundary detection)
            # Pass the boundary_head from the segmenter for GT boundary extraction
            boundary_aware_loss, loss_dict = boundary_aware_loss_fn(
                seg_outputs, seg_mask, model.segmenter.boundary_head
            )
            # Extract individual components for logging
            deep_supervision_loss = loss_dict.get('aux_loss', torch.tensor(0.0, device=device))

        boundary_loss = compute_boundary_loss(denoised, clean, seg_mask)

        # Noise correlation loss (encourages gate to correlate with noise)
        noise_corr_loss, gate_corr = compute_noise_correlation_loss(
            features['noise_gate'], features['raw_features']
        )

        # NEW: Layer gate diversity loss (KEY TMI CONTRIBUTION)
        # Encourages different layers to have different noise responses
        diversity_loss, _ = compute_layer_gate_diversity_loss(
            features['layer_gates'], features['seg_probs']
        )

        # NEURO-SYMBOLIC: Layer ordering loss (KEY TMI CONTRIBUTION)
        # Enforces anatomical constraint: layers must follow strict top-to-bottom order
        # Loss = 0 if layers are correctly ordered, > 0 if violations exist
        symbolic_ordering_loss, ordering_stats = compute_symbolic_ordering_loss(seg_logits, margin=2.0)

        # NEW: Gate regularization loss
        # Penalizes gates that deviate too far from center (0.6)
        # This prevents over/under-denoising while still allowing differentiation
        gate_center = 0.6
        gate_reg_loss = ((features['layer_gates'] - gate_center) ** 2).mean()

        # NEW: Clinical gate alignment loss
        # Encourages gates for clinically important layers (RNFL, IS/OS) to be higher
        if clinical_gate_loss_fn is not None:
            clinical_gate_loss = clinical_gate_loss_fn(
                features['layer_gates'], features['seg_probs']
            )
        else:
            clinical_gate_loss = torch.tensor(0.0, device=device)

        # CUAP-OCT Framework Losses (KEY TMI CONTRIBUTION)
        # These losses create clinically-meaningful, uncertainty-aware denoising
        uncertainty_loss = torch.tensor(0.0, device=device)
        pathology_loss = torch.tensor(0.0, device=device)
        sharpness_loss = torch.tensor(0.0, device=device)
        anatomical_loss = torch.tensor(0.0, device=device)
        confidence_loss = torch.tensor(0.0, device=device)

        if use_cuap:
            uncertainty = features['uncertainty']  # [B, 1, H, W]

            # 1. Uncertainty Calibration Loss
            # Ensures uncertainty is HIGH at boundaries/pathology, LOW in homogeneous regions
            if uncertainty_loss_fn is not None:
                uncertainty_loss = uncertainty_loss_fn(
                    uncertainty, denoised, clean, seg_mask, features['seg_probs']
                )

            # 2. Pathology Preservation Loss
            # Protects diagnostic features (drusen, fluid, lesions) from over-smoothing
            if pathology_loss_fn is not None:
                pathology_loss = pathology_loss_fn(denoised, clean, seg_mask)

            # 3. Boundary Sharpness Loss
            # Maintains sharp layer boundaries for accurate thickness measurement
            if boundary_sharpness_loss_fn is not None:
                sharpness_loss = boundary_sharpness_loss_fn(denoised, clean, seg_mask)

            # 4. Anatomical Consistency Loss
            # Ensures denoising maintains anatomical plausibility (layer order, thickness, continuity)
            if anatomical_loss_fn is not None:
                anatomical_loss = anatomical_loss_fn(features['seg_probs'])

            # 5. Confidence-Weighted Refinement Loss
            # Penalizes aggressive denoising in low-confidence regions (potential pathology)
            if confidence_loss_fn is not None:
                confidence_loss = confidence_loss_fn(
                    features['refinement'],
                    features['seg_confidence_map'],
                    denoised,
                    clean
                )

        # IS_OS BOUNDARY DETECTION LOSS (V4 approach - achieved 0.48 Dice)
        # When enabled, detect IS_OS as boundary between layers rather than region class
        is_os_boundary_loss = torch.tensor(0.0, device=device)
        is_os_boundary_dice = 0.0
        if use_boundary_detection and USE_BOUNDARY_DETECTION_FOR_ISOS:
            # Extract IS_OS boundary target from 5-class mask
            # IS_OS is class 3 in the 5-class (or class 2 in 4-class) scheme
            if USE_4_CLASS_SEGMENTATION:
                is_os_class = 2  # IS_OS in 4-class
            else:
                is_os_class = 3  # IS_OS in 5-class
            is_os_boundary_target = (seg_mask == is_os_class).float().unsqueeze(1)  # [B, 1, H, W]

            # Get IS_OS boundary logits from model
            is_os_boundary_logits = features.get('is_os_boundary_logits')
            if is_os_boundary_logits is not None:
                # BCE loss with pos_weight for class imbalance
                pos_weight_tensor = torch.tensor([boundary_pos_weight], device=device)
                bce_loss = F.binary_cross_entropy_with_logits(
                    is_os_boundary_logits, is_os_boundary_target,
                    pos_weight=pos_weight_tensor
                )

                # Dice loss for boundary detection
                boundary_probs = torch.sigmoid(is_os_boundary_logits)
                intersection = (boundary_probs * is_os_boundary_target).sum()
                union = boundary_probs.sum() + is_os_boundary_target.sum() + 1e-6
                dice_loss = 1 - (2 * intersection / union)

                # Combined loss
                is_os_boundary_loss = bce_loss + dice_loss

                # Compute boundary dice for monitoring
                is_os_boundary_dice = compute_boundary_dice(
                    is_os_boundary_logits.detach(), is_os_boundary_target
                )

        # Total loss with all components
        # KEY FIX: Dynamic loss scaling to balance denoising and segmentation
        # Without this, seg_loss (~1.24) dominates denoise_loss (~0.0007) by ~1770x
        lambda_gate_reg = 0.1  # Regularize gates toward center

        # Apply dynamic loss scaling if provided
        if loss_scaler is not None:
            # Update scaler with raw loss values and get scale factors
            denoise_val = denoise_loss.item()
            seg_val = seg_loss.item()
            scales = loss_scaler.update([denoise_val, seg_val])

            # Scale the losses to equal magnitude
            # denoise_scale will be large (~1770) to match seg_loss magnitude
            # seg_scale will be ~1.0 (or slightly adjusted)
            denoise_scale, seg_scale = scales

            # Apply scaling: scaled losses should have similar magnitude
            scaled_denoise_loss = denoise_loss * denoise_scale
            scaled_seg_loss = seg_loss * seg_scale * lambda_seg
        else:
            # Fallback: no scaling (original behavior)
            scaled_denoise_loss = denoise_loss
            scaled_seg_loss = lambda_seg * seg_loss

        # Segmentation boundary loss weight (helps with thin layers like ONL)
        # CRITICAL: 0.3 was too weak, 2.0 still not enough
        # Increasing to 5.0 to force boundary detection
        lambda_seg_boundary = 5.0  # Extreme weight for boundary detection

        # Boundary-aware loss weight (when using BoundaryAwareSegmenter)
        # This combines deep supervision + boundary detection for thin layers
        # Increased to 2.0 for stronger IS_OS boundary detection
        lambda_boundary_aware = 2.0  # Weight for boundary-aware segmentation loss

        loss = scaled_denoise_loss + scaled_seg_loss + lambda_boundary * boundary_loss \
               + lambda_noise_corr * noise_corr_loss + lambda_diversity * diversity_loss \
               + lambda_clinical_gate * clinical_gate_loss \
               + lambda_gate_reg * gate_reg_loss \
               + lambda_uncertainty * uncertainty_loss \
               + lambda_pathology * pathology_loss \
               + lambda_sharpness * sharpness_loss \
               + lambda_anatomical * anatomical_loss \
               + lambda_confidence * confidence_loss \
               + lambda_symbolic_ordering * symbolic_ordering_loss \
               + lambda_seg_boundary * seg_boundary_loss \
               + lambda_boundary_aware * boundary_aware_loss \
               + lambda_boundary_detection * is_os_boundary_loss  # IS_OS boundary detection (V4)

        if torch.isnan(loss):
            continue

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Metrics
        dice = compute_dice_score(seg_logits.detach(), seg_mask)

        total_denoise_loss += denoise_loss.item()
        total_seg_loss += seg_loss.item()
        total_dice_loss += dice_loss_val.item() if isinstance(dice_loss_val, torch.Tensor) else dice_loss_val
        total_seg_boundary_loss += seg_boundary_loss.item()
        total_boundary_loss += boundary_loss.item()
        total_noise_corr_loss += noise_corr_loss.item()
        total_diversity_loss += diversity_loss.item()
        total_clinical_gate_loss += clinical_gate_loss.item() if isinstance(clinical_gate_loss, torch.Tensor) else clinical_gate_loss
        # CUAP losses
        total_uncertainty_loss += uncertainty_loss.item() if isinstance(uncertainty_loss, torch.Tensor) else uncertainty_loss
        total_pathology_loss += pathology_loss.item() if isinstance(pathology_loss, torch.Tensor) else pathology_loss
        total_sharpness_loss += sharpness_loss.item() if isinstance(sharpness_loss, torch.Tensor) else sharpness_loss
        total_anatomical_loss += anatomical_loss.item() if isinstance(anatomical_loss, torch.Tensor) else anatomical_loss
        total_confidence_loss += confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else confidence_loss
        # Neuro-symbolic ordering loss
        total_symbolic_ordering_loss += symbolic_ordering_loss.item() if isinstance(symbolic_ordering_loss, torch.Tensor) else symbolic_ordering_loss
        # Deep supervision and boundary-aware losses
        total_deep_supervision_loss += deep_supervision_loss.item() if isinstance(deep_supervision_loss, torch.Tensor) else deep_supervision_loss
        total_boundary_aware_loss += boundary_aware_loss.item() if isinstance(boundary_aware_loss, torch.Tensor) else boundary_aware_loss
        # IS_OS boundary detection (V4)
        total_is_os_boundary_loss += is_os_boundary_loss.item() if isinstance(is_os_boundary_loss, torch.Tensor) else is_os_boundary_loss
        total_is_os_boundary_dice += is_os_boundary_dice
        total_gate_corr += gate_corr
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1

        # Track per-layer gate statistics (only count when key exists)
        for name in LAYER_NAMES:
            key = f'{name}_gate_mean'
            if key in features['layer_gate_stats']:
                layer_gate_means[name] += features['layer_gate_stats'][key]
                layer_gate_counts[name] += 1

        # Progress bar shows different metrics based on CUAP usage
        postfix = {
            'L_den': f'{total_denoise_loss/num_batches:.4f}',
            'L_seg': f'{total_seg_loss/num_batches:.4f}',
            'Dice': f'{total_dice/num_batches:.4f}'
        }
        if use_cuap:
            postfix['Unc'] = f'{total_uncertainty_loss/num_batches:.3f}'
            postfix['Conf'] = f'{total_confidence_loss/num_batches:.3f}'
            postfix['Anat'] = f'{total_anatomical_loss/num_batches:.3f}'
        else:
            postfix['Div'] = f'{total_diversity_loss/num_batches:.3f}'
            postfix['Clin'] = f'{total_clinical_gate_loss/num_batches:.3f}'
            postfix['GateCorr'] = f'{total_gate_corr/num_batches:+.3f}'
        pbar.set_postfix(postfix)

    # Handle edge case where all batches had NaN loss
    if num_batches == 0:
        return {
            'total_loss': float('nan'),
            'denoise_loss': float('nan'),
            'seg_loss': float('nan'),
            'boundary_loss': float('nan'),
            'noise_corr_loss': float('nan'),
            'diversity_loss': float('nan'),
            'clinical_gate_loss': float('nan'),
            # CUAP losses
            'uncertainty_loss': float('nan'),
            'pathology_loss': float('nan'),
            'sharpness_loss': float('nan'),
            'anatomical_loss': float('nan'),
            'confidence_loss': float('nan'),
            'gate_corr': 0.0,
            'layer_gate_means': {name: 0.0 for name in LAYER_NAMES},
            'dice': 0.0,
        }

    # Compute average layer gate means (use per-layer counts for accuracy)
    avg_layer_gate_means = {
        name: (layer_gate_means[name] / layer_gate_counts[name]
               if layer_gate_counts[name] > 0 else 0.0)
        for name in LAYER_NAMES
    }

    # Get loss scaler info for monitoring
    loss_scaler_info = loss_scaler.get_info() if loss_scaler is not None else None

    return {
        'total_loss': total_loss / num_batches,
        'denoise_loss': total_denoise_loss / num_batches,
        'seg_loss': total_seg_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches,  # Dice loss component
        'seg_boundary_loss': total_seg_boundary_loss / num_batches,  # Boundary loss for thin layers
        'boundary_loss': total_boundary_loss / num_batches,
        'noise_corr_loss': total_noise_corr_loss / num_batches,
        'diversity_loss': total_diversity_loss / num_batches,
        'clinical_gate_loss': total_clinical_gate_loss / num_batches,
        # CUAP losses
        'uncertainty_loss': total_uncertainty_loss / num_batches,
        'pathology_loss': total_pathology_loss / num_batches,
        'sharpness_loss': total_sharpness_loss / num_batches,
        'anatomical_loss': total_anatomical_loss / num_batches,
        'confidence_loss': total_confidence_loss / num_batches,
        # Neuro-symbolic loss
        'symbolic_ordering_loss': total_symbolic_ordering_loss / num_batches,
        # Deep supervision and boundary-aware losses
        'deep_supervision_loss': total_deep_supervision_loss / num_batches,
        'boundary_aware_loss': total_boundary_aware_loss / num_batches,
        # IS_OS boundary detection (V4)
        'is_os_boundary_loss': total_is_os_boundary_loss / num_batches,
        'is_os_boundary_dice': total_is_os_boundary_dice / num_batches,
        'gate_corr': total_gate_corr / num_batches,
        'layer_gate_means': avg_layer_gate_means,
        'dice': total_dice / num_batches,
        # KEY FIX: Loss scaling info
        'loss_scaler_info': loss_scaler_info,
    }


def validate(model, loader, device, base_model, use_boundary_detection=False,
             use_full_image_validation=False, val_jsonl=None, val_patch_size=128, val_stride=64):
    """Validate multi-task model with per-layer metrics and adaptive denoising analysis.

    Args:
        model: Model to validate
        loader: DataLoader for validation patches
        device: Device for computation
        base_model: Baseline model for comparison
        use_boundary_detection: If True, compute IS_OS boundary dice
        use_full_image_validation: If True, use sliding window on full images
                                   (requires val_jsonl to load full images)
        val_jsonl: Path to validation JSONL (needed for full-image validation)
        val_patch_size: Patch size for sliding window (default 128)
        val_stride: Stride for sliding window (default 64)
    """
    model.eval()
    base_model.eval()

    psnr_base_sum = 0
    psnr_ours_sum = 0
    ssim_base_sum = 0
    ssim_ours_sum = 0
    dice_sum = 0
    seg_loss_sum = 0
    is_os_boundary_dice_sum = 0  # IS_OS boundary detection
    is_os_boundary_count = 0
    n_samples = 0

    # Clinical metrics accumulators (using running sums + counts for memory efficiency)
    epi_base_sum, epi_base_count = 0.0, 0
    epi_ours_sum, epi_ours_count = 0.0, 0
    cnr_base_sum, cnr_base_count = 0.0, 0
    cnr_ours_sum, cnr_ours_count = 0.0, 0
    boundary_psnr_base_sum, boundary_psnr_base_count = 0.0, 0
    boundary_psnr_ours_sum, boundary_psnr_ours_count = 0.0, 0
    boundary_sharpness_base_sum, boundary_sharpness_base_count = 0.0, 0
    boundary_sharpness_ours_sum, boundary_sharpness_ours_count = 0.0, 0

    # Per-layer metrics accumulators (using running sums + counts)
    per_layer_psnr_base_sum = {name: 0.0 for name in LAYER_NAMES}
    per_layer_psnr_base_count = {name: 0 for name in LAYER_NAMES}
    per_layer_psnr_ours_sum = {name: 0.0 for name in LAYER_NAMES}
    per_layer_psnr_ours_count = {name: 0 for name in LAYER_NAMES}
    per_layer_ssim_base_sum = {name: 0.0 for name in LAYER_NAMES}
    per_layer_ssim_base_count = {name: 0 for name in LAYER_NAMES}
    per_layer_ssim_ours_sum = {name: 0.0 for name in LAYER_NAMES}
    per_layer_ssim_ours_count = {name: 0 for name in LAYER_NAMES}
    per_layer_dice_sum = {name: 0.0 for name in LAYER_NAMES}
    per_layer_dice_count = {name: 0 for name in LAYER_NAMES}

    # Adaptive denoising metrics accumulators (running sums)
    adaptive_metrics_sum = {}
    adaptive_metrics_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            seg_mask = batch['seg_mask'].to(device)

            # FIX: If noisy ≈ clean (segmentation-only mode), add synthetic noise for fair PSNR eval
            # This ensures we always measure actual denoising performance, not identity mapping
            mse_noisy_clean = ((noisy - clean) ** 2).mean().item()
            if mse_noisy_clean < 1e-6:  # noisy is essentially clean
                # Add realistic OCT noise for evaluation (σ≈0.126 based on PKU37 analysis)
                # Apply vertical correlation to match real OCT noise structure
                noise = torch.randn_like(clean) * 0.126
                # Simple vertical smoothing via average pooling + upsample
                if noise.dim() == 4:  # [B, C, H, W]
                    # Use global F (torch.nn.functional) - already imported at top
                    noise_smooth = F.avg_pool2d(noise, kernel_size=(3, 1), stride=1, padding=(1, 0))
                    noise = noise_smooth * (0.126 / (noise_smooth.std() + 1e-8))
                noisy_for_denoise = torch.clamp(clean + noise, 0, 1)
            else:
                noisy_for_denoise = noisy

            # Baseline - use noisy_for_denoise for fair comparison
            base_out = base_model(noisy_for_denoise, spatial_map=None, basis=None, alpha=0.0, gate=None)

            # Ours (multi-task) - use noisy_for_denoise for denoising, but original for segmentation
            denoised, seg_logits, features = model(noisy_for_denoise, return_features=True)

            # Compute adaptive denoising metrics (use running sums)
            adaptive_metrics = compute_adaptive_denoising_metrics(noisy_for_denoise, denoised, features)
            for key, val in adaptive_metrics.items():
                if key not in adaptive_metrics_sum:
                    adaptive_metrics_sum[key] = 0.0
                adaptive_metrics_sum[key] += val
            adaptive_metrics_count += 1

            # Segmentation metrics
            seg_loss = F.cross_entropy(seg_logits, seg_mask)
            dice = compute_dice_score(seg_logits, seg_mask)

            seg_loss_sum += seg_loss.item()
            dice_sum += dice

            # Per-layer Dice (running sums)
            layer_dice = compute_per_layer_dice(seg_logits, seg_mask)
            for name in LAYER_NAMES:
                if layer_dice[name] is not None:
                    per_layer_dice_sum[name] += layer_dice[name]
                    per_layer_dice_count[name] += 1

            # IS_OS Boundary Dice (V4 approach)
            if use_boundary_detection:
                # Get IS_OS boundary logits from model
                is_os_boundary_logits = model._is_os_boundary_logits
                if is_os_boundary_logits is not None:
                    # Extract IS_OS boundary target from mask
                    if USE_4_CLASS_SEGMENTATION:
                        is_os_class = 2
                    else:
                        is_os_class = 3
                    is_os_target = (seg_mask == is_os_class).float().unsqueeze(1)
                    b_dice = compute_boundary_dice(is_os_boundary_logits, is_os_target)
                    is_os_boundary_dice_sum += b_dice
                    is_os_boundary_count += 1

            # Per-layer PSNR (base and ours) - running sums
            layer_psnr_base = compute_per_layer_psnr(base_out, clean, seg_mask)
            layer_psnr_ours = compute_per_layer_psnr(denoised, clean, seg_mask)

            for name in LAYER_NAMES:
                if layer_psnr_base[name] is not None:
                    per_layer_psnr_base_sum[name] += layer_psnr_base[name]
                    per_layer_psnr_base_count[name] += 1
                if layer_psnr_ours[name] is not None:
                    per_layer_psnr_ours_sum[name] += layer_psnr_ours[name]
                    per_layer_psnr_ours_count[name] += 1

            # Per-layer SSIM (base and ours) - running sums
            layer_ssim_base = compute_per_layer_ssim(base_out, clean, seg_mask)
            layer_ssim_ours = compute_per_layer_ssim(denoised, clean, seg_mask)

            for name in LAYER_NAMES:
                if layer_ssim_base[name] is not None:
                    per_layer_ssim_base_sum[name] += layer_ssim_base[name]
                    per_layer_ssim_base_count[name] += 1
                if layer_ssim_ours[name] is not None:
                    per_layer_ssim_ours_sum[name] += layer_ssim_ours[name]
                    per_layer_ssim_ours_count[name] += 1

            # Clinical metrics: EPI (Edge Preservation Index) - running sums
            epi_base = compute_edge_preservation_index(base_out, clean)
            epi_ours = compute_edge_preservation_index(denoised, clean)
            epi_base_sum += epi_base
            epi_base_count += 1
            epi_ours_sum += epi_ours
            epi_ours_count += 1

            # Clinical metrics: CNR (Contrast-to-Noise Ratio) - running sums
            cnr_base, _ = compute_cnr_per_layer(base_out, seg_mask)
            cnr_ours, _ = compute_cnr_per_layer(denoised, seg_mask)
            if cnr_base is not None:
                cnr_base_sum += cnr_base
                cnr_base_count += 1
            if cnr_ours is not None:
                cnr_ours_sum += cnr_ours
                cnr_ours_count += 1

            # Clinical metrics: Boundary metrics - running sums
            bp_base, bs_base, _, _ = compute_boundary_metrics(base_out, clean, seg_mask)
            bp_ours, bs_ours, _, _ = compute_boundary_metrics(denoised, clean, seg_mask)
            if bp_base is not None:
                boundary_psnr_base_sum += bp_base
                boundary_psnr_base_count += 1
                boundary_sharpness_base_sum += bs_base
                boundary_sharpness_base_count += 1
            if bp_ours is not None:
                boundary_psnr_ours_sum += bp_ours
                boundary_psnr_ours_count += 1
                boundary_sharpness_ours_sum += bs_ours
                boundary_sharpness_ours_count += 1

            # Global denoising metrics
            for i in range(noisy.size(0)):
                psnr_base_sum += compute_psnr(base_out[i:i+1], clean[i:i+1])
                psnr_ours_sum += compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim_base_sum += compute_ssim(base_out[i:i+1], clean[i:i+1])
                ssim_ours_sum += compute_ssim(denoised[i:i+1], clean[i:i+1])
                n_samples += 1

    # Compute per-layer averages from running sums
    per_layer_results = {}
    for name in LAYER_NAMES:
        psnr_base = (per_layer_psnr_base_sum[name] / per_layer_psnr_base_count[name]
                     if per_layer_psnr_base_count[name] > 0 else None)
        psnr_ours = (per_layer_psnr_ours_sum[name] / per_layer_psnr_ours_count[name]
                     if per_layer_psnr_ours_count[name] > 0 else None)
        ssim_base = (per_layer_ssim_base_sum[name] / per_layer_ssim_base_count[name]
                     if per_layer_ssim_base_count[name] > 0 else None)
        ssim_ours = (per_layer_ssim_ours_sum[name] / per_layer_ssim_ours_count[name]
                     if per_layer_ssim_ours_count[name] > 0 else None)
        dice = (per_layer_dice_sum[name] / per_layer_dice_count[name]
                if per_layer_dice_count[name] > 0 else None)

        per_layer_results[name] = {
            'psnr_base': psnr_base,
            'psnr_ours': psnr_ours,
            'psnr_gain': (psnr_ours - psnr_base) if (psnr_base is not None and psnr_ours is not None) else None,
            'ssim_base': ssim_base,
            'ssim_ours': ssim_ours,
            'ssim_gain': (ssim_ours - ssim_base) if (ssim_base is not None and ssim_ours is not None) else None,
            'dice': dice,
        }

    # Compute clinical metrics averages from running sums
    clinical_metrics = {
        'epi_base': epi_base_sum / epi_base_count if epi_base_count > 0 else None,
        'epi_ours': epi_ours_sum / epi_ours_count if epi_ours_count > 0 else None,
        'cnr_base': cnr_base_sum / cnr_base_count if cnr_base_count > 0 else None,
        'cnr_ours': cnr_ours_sum / cnr_ours_count if cnr_ours_count > 0 else None,
        'boundary_psnr_base': boundary_psnr_base_sum / boundary_psnr_base_count if boundary_psnr_base_count > 0 else None,
        'boundary_psnr_ours': boundary_psnr_ours_sum / boundary_psnr_ours_count if boundary_psnr_ours_count > 0 else None,
        'boundary_sharpness_base': boundary_sharpness_base_sum / boundary_sharpness_base_count if boundary_sharpness_base_count > 0 else None,
        'boundary_sharpness_ours': boundary_sharpness_ours_sum / boundary_sharpness_ours_count if boundary_sharpness_ours_count > 0 else None,
    }

    # Compute adaptive denoising metrics averages from running sums
    adaptive_metrics_avg = {}
    if adaptive_metrics_count > 0:
        for key, val in adaptive_metrics_sum.items():
            adaptive_metrics_avg[key] = val / adaptive_metrics_count

    # Handle edge case where validation set is empty
    n_batches = len(loader) if len(loader) > 0 else 1
    n_samples = n_samples if n_samples > 0 else 1

    # IS_OS boundary dice (V4 approach)
    is_os_boundary_dice = (is_os_boundary_dice_sum / is_os_boundary_count
                           if is_os_boundary_count > 0 else None)

    return {
        'psnr_base': psnr_base_sum / n_samples,
        'psnr': psnr_ours_sum / n_samples,
        'ssim_base': ssim_base_sum / n_samples,
        'ssim': ssim_ours_sum / n_samples,
        'seg_loss': seg_loss_sum / n_batches,
        'dice': dice_sum / n_batches,
        'is_os_boundary_dice': is_os_boundary_dice,  # V4 IS_OS boundary detection
        'per_layer': per_layer_results,
        'adaptive': adaptive_metrics_avg,
        'clinical': clinical_metrics,
    }


def validate_per_layer(model, jsonl_path, device, base_model, patch_size=64, max_samples=100):
    """
    Validate model using per-layer targeted evaluation.

    This addresses the RAM constraint problem by evaluating each layer separately
    with 64x64 patches centered on that layer. This guarantees that every layer
    gets evaluated even when the full image height (496px) exceeds patch size.

    Args:
        model: Multi-task model to evaluate
        jsonl_path: Path to validation JSONL file
        device: Device to use
        base_model: Baseline model for comparison
        patch_size: Patch size (default 64)
        max_samples: Maximum number of base samples

    Returns:
        Dict with per-layer metrics for each anatomical layer
    """
    model.eval()
    base_model.eval()

    # Create per-layer evaluation dataset
    eval_ds = PerLayerEvalDataset(jsonl_path, patch_size=patch_size, max_samples=max_samples)
    eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False, num_workers=0)

    # Accumulators for each layer
    per_layer_stats = {name: {
        'psnr_base': [], 'psnr_ours': [],
        'ssim_base': [], 'ssim_ours': [],
        'dice': [], 'count': 0
    } for name in LAYER_NAMES}

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Per-Layer Validation"):
            # Check validity mask (convert to list of bools for safe iteration)
            valid_mask = batch['valid']  # Boolean tensor [B]
            if not valid_mask.any().item():
                continue

            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            seg_mask = batch['seg_mask'].to(device)
            layer_indices = batch['layer_idx']  # [B] tensor of layer indices

            # FIX: If noisy ≈ clean (segmentation-only mode), add synthetic noise for fair PSNR eval
            # Use σ=0.126 based on PKU37/Duke analysis of real OCT noise
            mse_noisy_clean = ((noisy - clean) ** 2).mean().item()
            if mse_noisy_clean < 1e-6:  # noisy is essentially clean
                noise = torch.randn_like(clean) * 0.126
                # Apply vertical smoothing for realistic correlation
                if noise.dim() == 4:
                    import torch.nn.functional as F
                    noise_smooth = F.avg_pool2d(noise, kernel_size=(3, 1), stride=1, padding=(1, 0))
                    noise = noise_smooth * (0.126 / (noise_smooth.std() + 1e-8))
                noisy_for_denoise = torch.clamp(clean + noise, 0, 1)
            else:
                noisy_for_denoise = noisy

            # Run models with noisy_for_denoise for fair PSNR comparison
            base_out = base_model(noisy_for_denoise, spatial_map=None, basis=None, alpha=0.0, gate=None)
            denoised, seg_logits = model(noisy_for_denoise)

            # Process each sample in the batch
            for i in range(noisy_for_denoise.size(0)):
                if not valid_mask[i].item():  # Use .item() for explicit bool conversion
                    continue

                layer_idx = layer_indices[i].item()
                layer_name = LAYER_NAMES[layer_idx]

                # Create mask for the target layer in this crop
                layer_mask = (seg_mask[i:i+1] == layer_idx).unsqueeze(1).float()  # [1, 1, H, W]
                n_pixels = layer_mask.sum()

                if n_pixels < 50:  # Skip if too few pixels of this layer
                    del layer_mask  # Clean up before continuing
                    continue

                # Compute per-layer PSNR
                # Use in-place operations where possible to reduce memory
                diff_base_sq = (base_out[i:i+1] - clean[i:i+1]).pow_(2).mul_(layer_mask)
                diff_ours_sq = (denoised[i:i+1] - clean[i:i+1]).pow_(2).mul_(layer_mask)

                mse_base = diff_base_sq.sum() / n_pixels
                mse_ours = diff_ours_sq.sum() / n_pixels

                # Only add to both lists if both are valid (ensures matched counts for gain calc)
                if mse_base > 0 and mse_ours > 0:
                    psnr_base = 10 * torch.log10(1.0 / mse_base).item()
                    psnr_ours = 10 * torch.log10(1.0 / mse_ours).item()
                    per_layer_stats[layer_name]['psnr_base'].append(psnr_base)
                    per_layer_stats[layer_name]['psnr_ours'].append(psnr_ours)

                # Clean up intermediate tensors
                del diff_base_sq, diff_ours_sq

                # Compute per-layer SSIM (simplified version)
                C1 = 0.01 ** 2
                C2 = 0.03 ** 2

                # Compute means (reuse layer_mask, no new tensors for masked pixels)
                mu_base = (base_out[i:i+1] * layer_mask).sum() / n_pixels
                mu_clean = (clean[i:i+1] * layer_mask).sum() / n_pixels
                mu_ours = (denoised[i:i+1] * layer_mask).sum() / n_pixels

                # Compute variances and covariances
                var_base = ((base_out[i:i+1] - mu_base) ** 2 * layer_mask).sum() / n_pixels
                var_clean = ((clean[i:i+1] - mu_clean) ** 2 * layer_mask).sum() / n_pixels
                var_ours = ((denoised[i:i+1] - mu_ours) ** 2 * layer_mask).sum() / n_pixels
                cov_base = ((base_out[i:i+1] - mu_base) * (clean[i:i+1] - mu_clean) * layer_mask).sum() / n_pixels
                cov_ours = ((denoised[i:i+1] - mu_ours) * (clean[i:i+1] - mu_clean) * layer_mask).sum() / n_pixels

                # Compute SSIM for both
                ssim_base = ((2 * mu_base * mu_clean + C1) * (2 * cov_base + C2)) / \
                           ((mu_base ** 2 + mu_clean ** 2 + C1) * (var_base + var_clean + C2))
                ssim_ours = ((2 * mu_ours * mu_clean + C1) * (2 * cov_ours + C2)) / \
                           ((mu_ours ** 2 + mu_clean ** 2 + C1) * (var_ours + var_clean + C2))

                # Add both together to ensure matched counts
                per_layer_stats[layer_name]['ssim_base'].append(ssim_base.item())
                per_layer_stats[layer_name]['ssim_ours'].append(ssim_ours.item())

                # Compute per-layer Dice
                pred = seg_logits[i].argmax(dim=0)  # [H, W]
                target = seg_mask[i]  # [H, W]

                pred_layer = (pred == layer_idx).float()
                target_layer = (target == layer_idx).float()

                intersection = (pred_layer * target_layer).sum()
                union = pred_layer.sum() + target_layer.sum()

                if union > 0:
                    dice = (2.0 * intersection / (union + 1e-8)).item()
                    per_layer_stats[layer_name]['dice'].append(dice)

                per_layer_stats[layer_name]['count'] += 1

                # Clean up per-sample tensors (target is a view, but delete for clarity)
                del layer_mask, pred, target, pred_layer, target_layer

            # Clean up batch tensors after processing
            del noisy, noisy_for_denoise, clean, seg_mask, base_out, denoised, seg_logits

    # Clean up dataset and loader
    del eval_ds, eval_loader
    gc.collect()

    # Compute averages
    results = {}
    for name in LAYER_NAMES:
        stats = per_layer_stats[name]
        results[name] = {
            'psnr_base': np.mean(stats['psnr_base']) if stats['psnr_base'] else None,
            'psnr_ours': np.mean(stats['psnr_ours']) if stats['psnr_ours'] else None,
            'psnr_gain': (np.mean(stats['psnr_ours']) - np.mean(stats['psnr_base']))
                        if (stats['psnr_base'] and stats['psnr_ours']) else None,
            'ssim_base': np.mean(stats['ssim_base']) if stats['ssim_base'] else None,
            'ssim_ours': np.mean(stats['ssim_ours']) if stats['ssim_ours'] else None,
            'ssim_gain': (np.mean(stats['ssim_ours']) - np.mean(stats['ssim_base']))
                        if (stats['ssim_base'] and stats['ssim_ours']) else None,
            'dice': np.mean(stats['dice']) if stats['dice'] else None,
            'n_samples': stats['count'],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-task denoising + segmentation')
    parser.add_argument('--train_jsonl', default='seg_data/seg_train.jsonl')
    parser.add_argument('--val_jsonl', default='seg_data/seg_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_train', type=int, default=500)
    parser.add_argument('--max_val', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_seg', type=float, default=1.0,
                       help='Weight for segmentation loss (increased to prioritize better Dice)')
    parser.add_argument('--lambda_boundary', type=float, default=0.1,
                       help='Weight for boundary preservation loss')
    parser.add_argument('--lambda_noise_corr', type=float, default=0.1,
                       help='Weight for noise correlation loss (encourages gate-noise correlation)')
    parser.add_argument('--lambda_diversity', type=float, default=0.05,
                       help='Weight for layer gate diversity loss (encourages different layer responses)')
    parser.add_argument('--lambda_clinical_gate', type=float, default=0.1,
                       help='Weight for clinical gate alignment loss (encourages higher gates for RNFL/IS_OS)')
    parser.add_argument('--use_clinical_weighting', action='store_true',
                       help='Enable clinical importance weighting for denoising loss (KEY TMI CONTRIBUTION)')
    parser.add_argument('--clinical_weight_rnfl', type=float, default=2.0,
                       help='Clinical weight for RNFL_GCL layer (glaucoma - highest priority)')
    parser.add_argument('--clinical_weight_is_os', type=float, default=1.5,
                       help='Clinical weight for IS_OS layer (visual acuity correlation)')
    parser.add_argument('--clinical_weight_rpe', type=float, default=1.2,
                       help='Clinical weight for RPE_Choroid layer (AMD)')

    # CUAP-OCT Framework Arguments (KEY TMI CONTRIBUTION)
    parser.add_argument('--use_cuap', action='store_true',
                       help='Enable full CUAP-OCT framework (uncertainty, pathology preservation, boundary sharpness)')
    parser.add_argument('--lambda_uncertainty', type=float, default=0.1,
                       help='Weight for uncertainty calibration loss')
    parser.add_argument('--lambda_pathology', type=float, default=0.1,
                       help='Weight for pathology preservation loss')
    parser.add_argument('--lambda_sharpness', type=float, default=0.1,
                       help='Weight for boundary sharpness loss')
    parser.add_argument('--lambda_anatomical', type=float, default=0.1,
                       help='Weight for anatomical consistency loss')
    parser.add_argument('--lambda_confidence', type=float, default=0.1,
                       help='Weight for confidence-weighted refinement loss')
    parser.add_argument('--lambda_symbolic_ordering', type=float, default=0.1,
                       help='Weight for neuro-symbolic layer ordering loss (enforces anatomical constraints)')
    parser.add_argument('--lambda_dice', type=float, default=0.5,
                       help='Weight for Dice loss in segmentation (0=CE only, 1=Dice only, 0.7=Dice priority)')

    # IS_OS Boundary Detection Arguments (V4 approach - achieved 0.48 Dice)
    parser.add_argument('--use_boundary_detection', action='store_true',
                       help='Use IS_OS boundary detection instead of 4-class segmentation. '
                            '3-class segmentation + IS_OS boundary head. Achieved 0.48 IS_OS Dice.')
    parser.add_argument('--use_full_image_validation', action='store_true',
                       help='Use full-image sliding window validation for stable evaluation. '
                            'Eliminates variance from random patch selection.')
    parser.add_argument('--val_patch_size', type=int, default=128,
                       help='Patch size for sliding window validation (default 128)')
    parser.add_argument('--val_stride', type=int, default=64,
                       help='Stride for sliding window validation (default 64 = 50%% overlap)')
    parser.add_argument('--lambda_boundary_detection', type=float, default=2.0,
                       help='Weight for IS_OS boundary detection loss (default 2.0)')
    parser.add_argument('--boundary_pos_weight', type=float, default=10.0,
                       help='Positive weight for boundary BCE loss (handles class imbalance)')

    # Ablation study flags
    parser.add_argument('--ablation_no_uncertainty', action='store_true',
                       help='Ablation: disable uncertainty calibration')
    parser.add_argument('--ablation_no_pathology', action='store_true',
                       help='Ablation: disable pathology preservation')
    parser.add_argument('--ablation_no_sharpness', action='store_true',
                       help='Ablation: disable boundary sharpness')
    parser.add_argument('--ablation_no_anatomical', action='store_true',
                       help='Ablation: disable anatomical consistency')
    parser.add_argument('--ablation_no_confidence', action='store_true',
                       help='Ablation: disable confidence-weighted refinement')
    parser.add_argument('--ablation_no_clinical_weight', action='store_true',
                       help='Ablation: disable clinical importance weighting')
    parser.add_argument('--ablation_no_gates', action='store_true',
                       help='Ablation: disable layer-specific noise gates (use global gate only)')

    parser.add_argument('--backbone_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--segmenter_ckpt', default=None,
                       help='Pretrained segmenter (None=train from scratch, recommended)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='checkpoints/multitask')

    # Curriculum learning: new arguments for Duke DME support
    parser.add_argument('--segmentation_only', action='store_true',
                       help='Train segmentation only (no denoising loss)')
    parser.add_argument('--denoising_only', action='store_true',
                       help='Train denoising only (no segmentation loss)')
    parser.add_argument('--add_synthetic_noise', action='store_true',
                       help='Add synthetic noise to clean images (Duke DME)')
    parser.add_argument('--noise_level', type=float, default=0.1,
                       help='Std of synthetic noise (default 0.1). Ignored if --noise_levels is set.')
    parser.add_argument('--noise_levels', type=str, default=None,
                       help='Comma-separated list of noise levels for multi-level augmentation '
                            '(e.g., "0.05,0.10,0.15,0.20,0.25"). Multiplies effective training data.')
    parser.add_argument('--use_speckle_noise', action='store_true', default=True,
                       help='Use realistic OCT speckle noise (multiplicative) in addition to Gaussian')
    parser.add_argument('--no_speckle_noise', action='store_true',
                       help='Disable speckle noise, use Gaussian only')
    parser.add_argument('--resume_from', default=None,
                       help='Resume training from checkpoint (loads full model state)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("MULTI-TASK TRAINING: DENOISING + LAYER SEGMENTATION")
    print("="*70)
    print("\nKey for TMI paper:")
    print("  - Joint optimization improves both tasks")
    print("  - Real anatomical grounding (not just soft probs)")
    print("  - Bidirectional benefit demonstrated")
    print("="*70)

    # Parse noise levels for multi-level augmentation
    if args.noise_levels:
        noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]
        print(f"\nMulti-level noise augmentation: {noise_levels}")
        print(f"  Effective training multiplier: {len(noise_levels)}x")
    else:
        noise_levels = None  # Will use single noise_level

    # Determine speckle noise usage
    use_speckle = args.use_speckle_noise and not args.no_speckle_noise

    # IS_OS Boundary Detection Mode (V4 approach - achieved 0.48 Dice)
    # When enabled, uses 3-class segmentation + IS_OS boundary head
    global USE_BOUNDARY_DETECTION_FOR_ISOS
    if args.use_boundary_detection:
        USE_BOUNDARY_DETECTION_FOR_ISOS = True
        print(f"\n{'='*70}")
        print("IS_OS BOUNDARY DETECTION MODE ENABLED (V4 approach)")
        print("  - 3-class segmentation: RNFL_GCL, INL_OPL_ONL, RPE_Choroid")
        print("  - IS_OS detected as boundary, not region class")
        print(f"  - Boundary loss weight: {args.lambda_boundary_detection}")
        print(f"  - Boundary pos_weight: {args.boundary_pos_weight}")
        if args.use_full_image_validation:
            print("  - Full-image sliding window validation ENABLED")
            print(f"    - Patch size: {args.val_patch_size}, Stride: {args.val_stride}")
        print(f"{'='*70}")
    else:
        USE_BOUNDARY_DETECTION_FOR_ISOS = False

    # Data
    # Use ensure_all_layers=False for truly random vertical crops
    # This samples from ALL positions (top/middle/bottom of retina)
    # Class weights handle the resulting class imbalance in the loss
    train_ds = MultiTaskOCTDataset(
        args.train_jsonl, args.patch_size, args.max_train,
        random_crop=True, ensure_all_layers=False,
        add_synthetic_noise=args.add_synthetic_noise,
        noise_level=args.noise_level,
        noise_levels=noise_levels,
        use_speckle_noise=use_speckle,
        layer_balanced_sampling=True,  # CRITICAL: Sample patches centered on thin layers
        thin_layer_boost=3  # 3x probability for thin layers (INL, ONL, IS_OS)
    )
    val_ds = MultiTaskOCTDataset(
        args.val_jsonl, args.patch_size, args.max_val,
        random_crop=True, ensure_all_layers=False,
        add_synthetic_noise=args.add_synthetic_noise,
        noise_level=args.noise_level,
        noise_levels=noise_levels,
        use_speckle_noise=use_speckle
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nData: {len(train_ds)} train, {len(val_ds)} val")

    # Compute class weights for balanced segmentation loss
    # This prevents class collapse (model predicting only middle classes like ONL)
    # Edge layers (RNFL_GCL, RPE_Choroid) are less likely in center crops, need higher weights
    print("Computing segmentation class weights...")
    seg_class_weights = compute_class_weights_from_loader(train_loader, num_classes=NUM_SEG_CLASSES, device=args.device)
    print(f"Class weights (raw): {[f'{w:.3f}' for w in seg_class_weights.tolist()]}")

    # AGGRESSIVE boost weights for thin layers to prevent class collapse
    if USE_4_CLASS_SEGMENTATION:
        # 4-class: RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid
        # IS_OS is only 4% of pixels vs 60% for RNFL - needs 10x boost to prevent collapse
        seg_boost = torch.tensor([1.0, 3.0, 10.0, 1.5], device=args.device)
        seg_class_weights = seg_class_weights * seg_boost
        print(f"Class weights (aggressive): {[f'{w:.3f}' for w in seg_class_weights.tolist()]}")
        print(f"  (Boost: RNFL 1.0x, INL_OPL_ONL 3.0x, IS_OS 10.0x, RPE 1.5x)")
    else:
        # 5-class: RNFL_GCL, INL_OPL, ONL, IS_OS, RPE_Choroid
        seg_boost = torch.tensor([1.5, 3.0, 4.0, 3.0, 1.0], device=args.device)
        seg_class_weights = seg_class_weights * seg_boost
        print(f"Class weights (aggressive): {[f'{w:.3f}' for w in seg_class_weights.tolist()]}")
        print(f"  (Boost: RNFL 1.5x, INL 3.0x, ONL 4.0x, IS_OS 3.0x, RPE 1.0x)")

    # Show segmentation loss configuration
    if args.lambda_dice > 0:
        print(f"\nSegmentation Loss: Combined CE ({1-args.lambda_dice:.0%}) + Dice ({args.lambda_dice:.0%})")
        print(f"  (Dice loss helps with thin layer segmentation)")
    else:
        print(f"\nSegmentation Loss: Cross-Entropy only")

    # Clinical importance weighting (KEY TMI CONTRIBUTION)
    # Weight denoising loss by clinical importance of each retinal layer
    clinical_loss_fn = None
    clinical_gate_loss_fn = None

    if args.use_clinical_weighting:
        print("\n" + "="*70)
        print("CLINICAL IMPORTANCE WEIGHTING ENABLED (KEY TMI CONTRIBUTION)")
        print("="*70)

        # Build clinical weights from command line arguments
        if USE_4_CLASS_SEGMENTATION:
            clinical_weights = {
                'RNFL_GCL': args.clinical_weight_rnfl,
                'INL_OPL_ONL': 1.0,  # Merged layer
                'IS_OS': args.clinical_weight_is_os,
                'RPE_Choroid': args.clinical_weight_rpe,
            }
        else:
            clinical_weights = {
                'RNFL_GCL': args.clinical_weight_rnfl,
                'INL_OPL': 1.0,
                'ONL': 1.0,
                'IS_OS': args.clinical_weight_is_os,
                'RPE_Choroid': args.clinical_weight_rpe,
            }

        print("Clinical weights (higher = more important for diagnosis):")
        for name, weight in clinical_weights.items():
            importance = "HIGH" if weight >= 1.5 else "MODERATE" if weight > 1.0 else "BASELINE"
            print(f"  {name:15s}: {weight:.1f} ({importance})")

        # Create clinical loss functions
        clinical_loss_fn = ClinicalWeightedMSELoss(
            clinical_weights=clinical_weights,
            normalize=True
        ).to(args.device)

        clinical_gate_loss_fn = ClinicalGateDiversityLoss(
            clinical_weights=clinical_weights,
            min_gate_threshold=0.3
        ).to(args.device)

        print("\nExpected behavior:")
        print("  - RNFL_GCL: Higher denoising priority (glaucoma detection)")
        print("  - IS_OS: Higher denoising priority (visual acuity)")
        print("  - Gates will be encouraged to be higher for important layers")
        print("="*70)

    # CUAP-OCT Framework Setup (KEY TMI CONTRIBUTION)
    uncertainty_loss_fn = None
    pathology_loss_fn = None
    boundary_sharpness_loss_fn = None
    anatomical_loss_fn = None
    confidence_loss_fn = None

    if args.use_cuap:
        print("\n" + "="*70)
        print("CUAP-OCT FRAMEWORK: Clinically-guided Uncertainty-Aware Pathology-preserving")
        print("="*70)
        print("\nNovel contributions:")
        print("  1. Uncertainty Calibration: HIGH at boundaries/pathology, LOW in homogeneous")
        print("  2. Pathology Preservation: Protects diagnostic features from over-smoothing")
        print("  3. Boundary Sharpness: Maintains sharp layer boundaries for thickness measurement")
        print("  4. Anatomical Consistency: Enforces layer order, thickness, and continuity")
        print("  5. Confidence-Weighted Denoising: Conservative in uncertain regions")
        print("")

        # Set up CUAP losses (respecting ablation flags)
        if not args.ablation_no_uncertainty:
            uncertainty_loss_fn = UncertaintyCalibrationLoss(
                boundary_weight=2.0,
                pathology_weight=3.0
            ).to(args.device)
            print(f"  Uncertainty Loss:   lambda={args.lambda_uncertainty}")
        else:
            print("  Uncertainty Loss:   DISABLED (ablation)")

        if not args.ablation_no_pathology:
            pathology_loss_fn = PathologyPreservationLoss(
                sensitivity=1.0
            ).to(args.device)
            print(f"  Pathology Loss:     lambda={args.lambda_pathology}")
        else:
            print("  Pathology Loss:     DISABLED (ablation)")

        if not args.ablation_no_sharpness:
            boundary_sharpness_loss_fn = BoundarySharpnessLoss().to(args.device)
            print(f"  Sharpness Loss:     lambda={args.lambda_sharpness}")
        else:
            print("  Sharpness Loss:     DISABLED (ablation)")

        if not args.ablation_no_anatomical:
            anatomical_loss_fn = AnatomicalConsistencyLoss(
                lambda_ordering=1.0,
                lambda_thickness=0.5,
                lambda_continuity=1.0,
            ).to(args.device)
            print(f"  Anatomical Loss:    lambda={args.lambda_anatomical}")
            print("    - Layer ordering:   enforced (RNFL above INL, etc.)")
            print("    - Thickness range:  50-150μm (RNFL), 30-80μm (INL), etc.")
            print("    - Continuity:       smooth boundaries enforced")
        else:
            print("  Anatomical Loss:    DISABLED (ablation)")

        if not args.ablation_no_confidence:
            confidence_loss_fn = ConfidenceWeightedRefinementLoss(
                low_conf_threshold=0.5,
                high_conf_threshold=0.8,
                penalty_weight=1.0,
            ).to(args.device)
            print(f"  Confidence Loss:    lambda={args.lambda_confidence}")
            print("    - Low confidence:   conservative denoising (preserve pathology)")
            print("    - High confidence:  aggressive denoising (safe regions)")
            print("    - Boundaries:       moderate denoising (clinical critical)")
        else:
            print("  Confidence Loss:    DISABLED (ablation)")

        print("="*70)

    # Handle ablation for clinical weighting
    if args.ablation_no_clinical_weight:
        clinical_loss_fn = None
        clinical_gate_loss_fn = None
        print("\n[ABLATION] Clinical importance weighting DISABLED")

    # Model
    if args.resume_from and os.path.exists(args.resume_from):
        # Resume from full model checkpoint (curriculum learning)
        model = MultiTaskDenoiser(
            backbone_ckpt=None,  # Don't load backbone separately
            segmenter_ckpt=None
        ).to(args.device)
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        print(f"[RESUME] Loaded full model from {args.resume_from}")
        del ckpt
    else:
        model = MultiTaskDenoiser(
            backbone_ckpt=args.backbone_ckpt,
            segmenter_ckpt=args.segmenter_ckpt
        ).to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Baseline for comparison
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)
    ckpt = torch.load(args.backbone_ckpt, map_location=args.device, weights_only=False)
    base_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    del ckpt
    base_model.eval()

    # Optimizer with differential learning rates
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    # Training loop
    best_psnr = 0
    best_dice = 0

    # KEY FIX: Dynamic Loss Scaler for multi-task balancing
    # This normalizes denoising loss (~0.0007) and segmentation loss (~1.24) to equal magnitude
    # Without this, segmentation dominates and PSNR degrades over epochs
    loss_scaler = DynamicLossScaler(num_tasks=2, ema_decay=0.99, initial_scale=1.0)
    print(f"\n{'='*70}")
    print("DYNAMIC LOSS SCALING ENABLED (KEY FIX)")
    print("  - Normalizes denoising and segmentation losses to equal magnitude")
    print("  - Prevents segmentation from dominating (was ~1770x larger)")
    print("  - Ensures stable PSNR improvement throughout training")
    print(f"{'='*70}")

    # Curriculum learning mode adjustments
    if args.segmentation_only:
        print(f"\n{'='*70}")
        print("SEGMENTATION ONLY MODE")
        print("  - Denoising loss weight set to 0")
        print("  - Focus on Dice/CE segmentation loss")
        print(f"{'='*70}")
        # Disable denoising-related losses
        args.lambda_noise_corr = 0.0
        args.lambda_diversity = 0.0
        args.lambda_confidence = 0.0
        # Don't use CUAP denoising losses
        args.use_cuap = False
        denoising_weight = 0.0
    elif args.denoising_only:
        print(f"\n{'='*70}")
        print("DENOISING ONLY MODE")
        print("  - Segmentation loss weight set to 0")
        print("  - Focus on denoising metrics (PSNR/SSIM)")
        print(f"{'='*70}")
        args.lambda_seg = 0.0
        args.lambda_dice = 0.0
        args.lambda_boundary = 0.0
        args.lambda_anatomical = 0.0
        denoising_weight = 1.0
    else:
        denoising_weight = 1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, epoch,
            lambda_seg=args.lambda_seg, lambda_boundary=args.lambda_boundary,
            lambda_noise_corr=args.lambda_noise_corr, lambda_diversity=args.lambda_diversity,
            lambda_clinical_gate=args.lambda_clinical_gate,
            lambda_uncertainty=args.lambda_uncertainty,
            lambda_pathology=args.lambda_pathology,
            lambda_sharpness=args.lambda_sharpness,
            lambda_anatomical=args.lambda_anatomical,
            lambda_confidence=args.lambda_confidence,
            lambda_dice=args.lambda_dice,
            lambda_symbolic_ordering=args.lambda_symbolic_ordering,
            lambda_boundary_detection=args.lambda_boundary_detection,
            boundary_pos_weight=args.boundary_pos_weight,
            seg_class_weights=seg_class_weights,
            clinical_loss_fn=clinical_loss_fn,
            clinical_gate_loss_fn=clinical_gate_loss_fn,
            uncertainty_loss_fn=uncertainty_loss_fn,
            pathology_loss_fn=pathology_loss_fn,
            boundary_sharpness_loss_fn=boundary_sharpness_loss_fn,
            anatomical_loss_fn=anatomical_loss_fn,
            confidence_loss_fn=confidence_loss_fn,
            use_cuap=args.use_cuap,
            loss_scaler=loss_scaler,
            use_boundary_detection=args.use_boundary_detection  # V4 IS_OS boundary detection
        )
        val_metrics = validate(
            model, val_loader, args.device, base_model,
            use_boundary_detection=args.use_boundary_detection,
            use_full_image_validation=args.use_full_image_validation,
            val_jsonl=args.val_jsonl,
            val_patch_size=args.val_patch_size,
            val_stride=args.val_stride
        )

        print(f"\nTrain:")
        print(f"  Denoise Loss:     {train_metrics['denoise_loss']:.4f}")
        if args.use_clinical_weighting:
            print(f"    (Clinical weighted - RNFL:{args.clinical_weight_rnfl}x, IS_OS:{args.clinical_weight_is_os}x)")
        print(f"  Seg Loss:         {train_metrics['seg_loss']:.4f}")
        if args.lambda_dice > 0:
            print(f"    (CE+Dice combined, Dice component: {train_metrics['dice_loss']:.4f})")
        print(f"  Seg Boundary:     {train_metrics['seg_boundary_loss']:.4f} [KEY for thin layers like ONL]")
        print(f"  Boundary Loss:    {train_metrics['boundary_loss']:.4f}")
        print(f"  Noise Corr Loss:  {train_metrics['noise_corr_loss']:.4f}")
        print(f"  Diversity Loss:   {train_metrics['diversity_loss']:.4f} [lower = more diverse layer gates]")
        if args.use_clinical_weighting and not args.ablation_no_clinical_weight:
            print(f"  Clinical Gate:    {train_metrics['clinical_gate_loss']:.4f} [encourages high gates for RNFL/IS_OS]")
        print(f"  Gate Correlation: {train_metrics['gate_corr']:+.4f} [target: >+0.3]")
        print(f"  Dice:             {train_metrics['dice']:.4f}")

        # CUAP-OCT Framework losses
        if args.use_cuap:
            print(f"\n  CUAP-OCT Framework:")
            if not args.ablation_no_uncertainty:
                print(f"    Uncertainty:    {train_metrics['uncertainty_loss']:.4f} [calibrates uncertainty maps]")
            if not args.ablation_no_pathology:
                print(f"    Pathology:      {train_metrics['pathology_loss']:.4f} [preserves diagnostic features]")
            if not args.ablation_no_sharpness:
                print(f"    Sharpness:      {train_metrics['sharpness_loss']:.4f} [maintains layer boundaries]")
            if not args.ablation_no_anatomical:
                print(f"    Anatomical:     {train_metrics['anatomical_loss']:.4f} [enforces layer structure]")
            if not args.ablation_no_confidence:
                print(f"    Confidence:     {train_metrics['confidence_loss']:.4f} [confidence-weighted refinement]")

        # Neuro-Symbolic Ordering Loss (KEY TMI CONTRIBUTION)
        if args.lambda_symbolic_ordering > 0:
            print(f"\n  Neuro-Symbolic Constraints:")
            print(f"    Ordering Loss:  {train_metrics['symbolic_ordering_loss']:.4f} [0 = valid layer order, >0 = violations]")

        # IS_OS Boundary Detection (V4 approach)
        if args.use_boundary_detection:
            print(f"\n  IS_OS Boundary Detection (V4):")
            print(f"    Boundary Loss:  {train_metrics['is_os_boundary_loss']:.4f}")
            print(f"    Boundary Dice:  {train_metrics['is_os_boundary_dice']:.4f} [target: >0.45]")

        # Print layer-specific gate means (KEY TMI METRIC)
        print(f"\n  Layer-Specific Gate Means (higher = more refinement for that layer):")
        layer_gates = train_metrics['layer_gate_means']
        for name in LAYER_NAMES:
            print(f"    {name:15s}: {layer_gates.get(name, 0):.4f}")

        # Print loss scaling info (KEY FIX)
        scaler_info = train_metrics.get('loss_scaler_info')
        if scaler_info is not None:
            print(f"\n  Dynamic Loss Scaling (KEY FIX):")
            emas = scaler_info['loss_emas']
            scales = scaler_info['scales']
            if emas[0] is not None and emas[1] is not None:
                print(f"    Denoise EMA:    {emas[0]:.6f} -> Scale: {scales[0]:.1f}x")
                print(f"    Seg EMA:        {emas[1]:.4f} -> Scale: {scales[1]:.2f}x")
                print(f"    Ratio (seg/den):{emas[1]/emas[0]:.0f}x [was dominating before fix]")

        print(f"\nValidation:")
        print(f"  PSNR (base):  {val_metrics['psnr_base']:.2f} dB")
        print(f"  PSNR (ours):  {val_metrics['psnr']:.2f} dB")
        psnr_gain = val_metrics['psnr'] - val_metrics['psnr_base']
        print(f"  PSNR GAIN:    {'+' if psnr_gain >= 0 else ''}{psnr_gain:.2f} dB")
        print(f"  SSIM (base):  {val_metrics['ssim_base']:.4f}")
        print(f"  SSIM (ours):  {val_metrics['ssim']:.4f}")
        ssim_gain = val_metrics['ssim'] - val_metrics['ssim_base']
        print(f"  SSIM GAIN:    {'+' if ssim_gain >= 0 else ''}{ssim_gain:.4f}")
        print(f"  Dice:         {val_metrics['dice']:.4f}")

        # IS_OS Boundary Detection (V4 approach)
        if args.use_boundary_detection and val_metrics.get('is_os_boundary_dice') is not None:
            is_os_dice = val_metrics['is_os_boundary_dice']
            status = "EXCELLENT" if is_os_dice >= 0.45 else "GOOD" if is_os_dice >= 0.3 else "LEARNING"
            print(f"  IS_OS Boundary Dice: {is_os_dice:.4f} [{status}]")

        # =====================================================================
        # PER-LAYER DICE (EVERY EPOCH) - Monitor thin layer learning
        # =====================================================================
        print(f"\n  Per-Layer Segmentation Dice (EVERY EPOCH):")
        print(f"  {'-'*65}")
        print(f"  {'Layer':<15} {'Dice':<10} {'Status':<12} {'Progress':<20}")
        print(f"  {'-'*65}")

        # Define thresholds for status
        GOOD_THRESHOLD = 0.3
        WARN_THRESHOLD = 0.1

        for name in LAYER_NAMES:
            layer_data = val_metrics['per_layer'][name]
            dice_val = layer_data.get('dice')

            if dice_val is not None:
                # Determine status
                if dice_val >= GOOD_THRESHOLD:
                    status = "OK"
                    bar_char = "#"
                elif dice_val >= WARN_THRESHOLD:
                    status = "LEARNING"
                    bar_char = "="
                else:
                    status = "COLLAPSED"
                    bar_char = "."

                # Create progress bar (20 chars max)
                bar_len = int(min(dice_val, 1.0) * 20)
                progress_bar = f"[{bar_char * bar_len}{' ' * (20 - bar_len)}]"

                # Print with formatting
                print(f"  {name:<15} {dice_val:<10.4f} {status:<12} {progress_bar}")
            else:
                print(f"  {name:<15} {'N/A':<10} {'NO DATA':<12} {'[                    ]'}")

        print(f"  {'-'*65}")

        # Summary line for thin layers
        if USE_4_CLASS_SEGMENTATION:
            thin_layers = ['INL_OPL_ONL', 'IS_OS']  # Only 2 thin layers in 4-class scheme
        else:
            thin_layers = ['INL_OPL', 'ONL', 'IS_OS']
        thin_dice = [val_metrics['per_layer'].get(n, {}).get('dice', 0) or 0 for n in thin_layers]
        avg_thin = sum(thin_dice) / len(thin_dice) if thin_dice else 0
        thin_status = "OK" if avg_thin >= WARN_THRESHOLD else "NEEDS ATTENTION"
        print(f"  Thin Layer Avg: {avg_thin:.4f} ({thin_status})")
        print(f"  {'-'*65}")

        # Clinical metrics (always print - important for paper)
        clinical = val_metrics['clinical']
        print(f"\n  Clinical Metrics:")
        if clinical['epi_base'] and clinical['epi_ours']:
            epi_diff = clinical['epi_ours'] - clinical['epi_base']
            # EPI closer to 1.0 is better
            print(f"  EPI (base):           {clinical['epi_base']:.4f}")
            print(f"  EPI (ours):           {clinical['epi_ours']:.4f} ({'+' if epi_diff >= 0 else ''}{epi_diff:.4f}) [1.0 = perfect]")
        if clinical['cnr_base'] and clinical['cnr_ours']:
            cnr_gain = clinical['cnr_ours'] - clinical['cnr_base']
            print(f"  CNR (base):           {clinical['cnr_base']:.4f}")
            print(f"  CNR (ours):           {clinical['cnr_ours']:.4f} ({'+' if cnr_gain >= 0 else ''}{cnr_gain:.4f}) [higher = better]")
        if clinical['boundary_psnr_base'] and clinical['boundary_psnr_ours']:
            bp_gain = clinical['boundary_psnr_ours'] - clinical['boundary_psnr_base']
            print(f"  Boundary PSNR (base): {clinical['boundary_psnr_base']:.2f} dB")
            print(f"  Boundary PSNR (ours): {clinical['boundary_psnr_ours']:.2f} dB ({'+' if bp_gain >= 0 else ''}{bp_gain:.2f} dB)")
        if clinical['boundary_sharpness_base'] and clinical['boundary_sharpness_ours']:
            print(f"  Boundary Sharpness:   {clinical['boundary_sharpness_ours']:.4f} [1.0 = same as clean]")

        # Adaptive denoising metrics (always print - key for TMI contribution)
        adaptive = val_metrics.get('adaptive', {})
        if adaptive:
            print(f"\n  Adaptive Per-Pixel Denoising Metrics:")
            print(f"  {'-'*70}")
            # Spatial variation (proves non-uniform denoising)
            ref_cv = adaptive.get('refinement_cv', 0)
            print(f"  Refinement Spatial CV:     {ref_cv:.4f} [>0 = spatially varying]")

            # Correlation with noise features (proves noise-aware)
            corr_cv = adaptive.get('corr_coef_var_refinement', 0)
            corr_std = adaptive.get('corr_local_std_refinement', 0)
            print(f"  Corr(CoefVar, Refinement): {corr_cv:+.4f} [+ve = more refinement in high-noise]")
            print(f"  Corr(LocalStd, Refinement):{corr_std:+.4f} [+ve = noise-adaptive]")

            # High vs low noise regions (proves adaptive strength)
            ref_ratio = adaptive.get('refinement_ratio', 1)
            res_ratio = adaptive.get('residual_ratio', 1)
            print(f"  Refinement Ratio (H/L):    {ref_ratio:.3f} [>1 = stronger in noisy regions]")
            print(f"  Residual Ratio (H/L):      {res_ratio:.3f} [>1 = more denoising where needed]")

            # Interpretation
            is_adaptive = ref_cv > 0.1 and (corr_cv > 0.1 or corr_std > 0.1) and ref_ratio > 1.0
            if is_adaptive:
                print(f"  Status: ADAPTIVE DENOISING WORKING")
            else:
                print(f"  Status: Check if adaptation is sufficient")
            print(f"  {'-'*70}")

        # Per-layer anatomy metrics - only print every 5 epochs or final epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"\n  Per-Layer Anatomy Metrics (PSNR):")
            print(f"  {'-'*70}")
            print(f"  {'Layer':<15} {'PSNR(base)':<12} {'PSNR(ours)':<12} {'PSNR Gain':<12} {'Dice':<8}")
            print(f"  {'-'*70}")
            for name in LAYER_NAMES:
                layer_data = val_metrics['per_layer'][name]
                base_str = f"{layer_data['psnr_base']:.2f}" if layer_data['psnr_base'] else "N/A"
                ours_str = f"{layer_data['psnr_ours']:.2f}" if layer_data['psnr_ours'] else "N/A"
                if layer_data['psnr_gain'] is not None:
                    gain_str = f"{'+' if layer_data['psnr_gain'] >= 0 else ''}{layer_data['psnr_gain']:.2f}"
                else:
                    gain_str = "N/A"
                dice_str = f"{layer_data['dice']:.3f}" if layer_data['dice'] else "N/A"
                print(f"  {name:<15} {base_str:<12} {ours_str:<12} {gain_str:<12} {dice_str:<8}")
            print(f"  {'-'*70}")

            # Per-layer SSIM metrics
            print(f"\n  Per-Layer Anatomy Metrics (SSIM):")
            print(f"  {'-'*70}")
            print(f"  {'Layer':<15} {'SSIM(base)':<12} {'SSIM(ours)':<12} {'SSIM Gain':<12}")
            print(f"  {'-'*70}")
            for name in LAYER_NAMES:
                layer_data = val_metrics['per_layer'][name]
                base_str = f"{layer_data['ssim_base']:.4f}" if layer_data['ssim_base'] else "N/A"
                ours_str = f"{layer_data['ssim_ours']:.4f}" if layer_data['ssim_ours'] else "N/A"
                if layer_data['ssim_gain'] is not None:
                    gain_str = f"{'+' if layer_data['ssim_gain'] >= 0 else ''}{layer_data['ssim_gain']:.4f}"
                else:
                    gain_str = "N/A"
                print(f"  {name:<15} {base_str:<12} {ours_str:<12} {gain_str:<12}")
            print(f"  {'-'*70}")

        # Save best models
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            print(f"*** NEW BEST PSNR: {best_psnr:.2f} dB ***")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'psnr': best_psnr,
                'dice': val_metrics['dice'],
            }, os.path.join(args.output_dir, 'best_psnr.pth'))

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            print(f"*** NEW BEST DICE: {best_dice:.4f} ***")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'psnr': val_metrics['psnr'],
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best_dice.pth'))

        # Memory cleanup
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # Final validation with best model for complete per-layer report
    print(f"\n{'='*70}")
    print("MULTI-TASK TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best Dice: {best_dice:.4f}")

    # Load best model and get final per-layer metrics
    print(f"\n{'='*70}")
    print("FINAL PER-LAYER ANATOMY ANALYSIS (FOR PAPER)")
    print(f"{'='*70}")
    best_ckpt = torch.load(os.path.join(args.output_dir, 'best_psnr.pth'), map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt['state_dict'])
    final_metrics = validate(model, val_loader, args.device, base_model)

    print(f"\nGlobal Metrics:")
    print(f"  PSNR (base):     {final_metrics['psnr_base']:.2f} dB")
    print(f"  PSNR (ours):     {final_metrics['psnr']:.2f} dB")
    print(f"  PSNR GAIN:       +{final_metrics['psnr'] - final_metrics['psnr_base']:.2f} dB")
    print(f"  SSIM (base):     {final_metrics['ssim_base']:.4f}")
    print(f"  SSIM (ours):     {final_metrics['ssim']:.4f}")
    print(f"  SSIM GAIN:       +{final_metrics['ssim'] - final_metrics['ssim_base']:.4f}")
    print(f"  Dice:            {final_metrics['dice']:.4f}")

    # Run per-layer targeted evaluation (guarantees all 5 layers get evaluated)
    print(f"\n{'='*70}")
    print("PER-LAYER TARGETED EVALUATION (64x64 patches centered on each layer)")
    print(f"{'='*70}")
    print("This evaluation uses separate 64x64 crops centered on each layer,")
    print(f"ensuring all {NUM_SEG_CLASSES} anatomical layers are evaluated even within memory constraints.")
    print("")

    per_layer_metrics = validate_per_layer(
        model, args.val_jsonl, args.device, base_model,
        patch_size=args.patch_size, max_samples=args.max_val
    )

    print(f"\nPer-Layer PSNR Improvement (Table for Paper):")
    print(f"{'='*70}")
    print(f"| {'Layer':<15} | {'Base PSNR':<12} | {'Our PSNR':<12} | {'Improvement':<12} | {'N':<6} |")
    print(f"|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*8}|")
    for name in LAYER_NAMES:
        layer_data = per_layer_metrics[name]
        if layer_data['psnr_base'] and layer_data['psnr_ours']:
            gain = layer_data['psnr_gain']
            gain_str = f"{'+' if gain >= 0 else ''}{gain:.2f}"
            print(f"| {name:<15} | {layer_data['psnr_base']:.2f} dB{' '*4} | {layer_data['psnr_ours']:.2f} dB{' '*4} | {gain_str} dB{' '*4} | {layer_data['n_samples']:<6} |")
        else:
            print(f"| {name:<15} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {layer_data['n_samples']:<6} |")
    print(f"{'='*70}")

    print(f"\nPer-Layer SSIM Improvement (Table for Paper):")
    print(f"{'='*70}")
    print(f"| {'Layer':<15} | {'Base SSIM':<12} | {'Our SSIM':<12} | {'Improvement':<12} |")
    print(f"|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*14}|")
    for name in LAYER_NAMES:
        layer_data = per_layer_metrics[name]
        if layer_data['ssim_base'] and layer_data['ssim_ours']:
            gain = layer_data['ssim_gain']
            gain_str = f"{'+' if gain >= 0 else ''}{gain:.4f}"
            print(f"| {name:<15} | {layer_data['ssim_base']:.4f}{' '*6} | {layer_data['ssim_ours']:.4f}{' '*6} | {gain_str}{' '*5} |")
        else:
            print(f"| {name:<15} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} |")
    print(f"{'='*70}")

    print(f"\nPer-Layer Segmentation (Dice Scores):")
    print(f"{'='*70}")
    for name in LAYER_NAMES:
        layer_data = per_layer_metrics[name]
        if layer_data['dice'] is not None:
            print(f"  {name:<15}: Dice = {layer_data['dice']:.4f} (n={layer_data['n_samples']})")
        else:
            print(f"  {name:<15}: Dice = N/A (n={layer_data['n_samples']})")
    print(f"{'='*70}")

    # Clinical Metrics Table (for paper)
    print(f"\nClinical Metrics (Table for Paper):")
    print(f"{'='*70}")
    print(f"| {'Metric':<25} | {'Baseline':<12} | {'Ours':<12} | {'Improvement':<12} |")
    print(f"|{'-'*27}|{'-'*14}|{'-'*14}|{'-'*14}|")

    clinical = final_metrics['clinical']
    if clinical['epi_base'] and clinical['epi_ours']:
        epi_diff = clinical['epi_ours'] - clinical['epi_base']
        # For EPI, closer to 1.0 is better
        epi_better = "closer to 1.0" if abs(clinical['epi_ours'] - 1.0) < abs(clinical['epi_base'] - 1.0) else ""
        print(f"| {'Edge Preservation (EPI)':<25} | {clinical['epi_base']:.4f}{' '*6} | {clinical['epi_ours']:.4f}{' '*6} | {'+' if epi_diff >= 0 else ''}{epi_diff:.4f}{' '*5} |")

    if clinical['cnr_base'] and clinical['cnr_ours']:
        cnr_gain = clinical['cnr_ours'] - clinical['cnr_base']
        cnr_pct = (cnr_gain / clinical['cnr_base']) * 100 if clinical['cnr_base'] > 0 else 0
        print(f"| {'Contrast-to-Noise (CNR)':<25} | {clinical['cnr_base']:.4f}{' '*6} | {clinical['cnr_ours']:.4f}{' '*6} | +{cnr_pct:.1f}%{' '*6} |")

    if clinical['boundary_psnr_base'] and clinical['boundary_psnr_ours']:
        bp_gain = clinical['boundary_psnr_ours'] - clinical['boundary_psnr_base']
        print(f"| {'Boundary PSNR':<25} | {clinical['boundary_psnr_base']:.2f} dB{' '*4} | {clinical['boundary_psnr_ours']:.2f} dB{' '*4} | +{bp_gain:.2f} dB{' '*4} |")

    if clinical['boundary_sharpness_base'] and clinical['boundary_sharpness_ours']:
        bs_diff = clinical['boundary_sharpness_ours'] - clinical['boundary_sharpness_base']
        print(f"| {'Boundary Sharpness':<25} | {clinical['boundary_sharpness_base']:.4f}{' '*6} | {clinical['boundary_sharpness_ours']:.4f}{' '*6} | {'+' if bs_diff >= 0 else ''}{bs_diff:.4f}{' '*5} |")

    print(f"{'='*70}")

    print(f"\nNote on Clinical Metrics:")
    print(f"  - EPI (Edge Preservation Index): 1.0 = perfect edge preservation")
    print(f"  - CNR (Contrast-to-Noise Ratio): Higher = better layer distinguishability")
    print(f"  - Boundary PSNR: Quality specifically at layer boundaries")
    print(f"  - Boundary Sharpness: 1.0 = same sharpness as clean image")

    # Cleanup
    del best_ckpt
    gc.collect()


if __name__ == '__main__':
    main()
