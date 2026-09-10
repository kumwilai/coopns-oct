#!/usr/bin/env python3
"""
Phase 1A-2: Joint Denoising + Segmentation with IS_OS Boundary Detection

Integrates the proven SimpleBoundarySegmenter (3-class + IS_OS boundary) into
the MultiTaskDenoiser architecture with all denoising innovations:
- NAFNet backbone for high-quality denoising
- Layer-specific noise gates
- Confidence-weighted denoising
- Physics-based noise features
- Uncertainty quantification

Properly loads segmentation weights from Phase 1A-1 checkpoint.
"""

import os
import sys
import argparse
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

from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.models.noise_features import NoiseFeatureExtractor

# ============================================================================
# Configuration - 3-class segmentation with IS_OS as boundary
# ============================================================================
NUM_CLASSES = 3
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'RPE_Choroid']


def remap_mask_to_3class(mask):
    """Remap 5-class mask to 3-class."""
    new_mask = np.zeros_like(mask)
    new_mask[mask == 0] = 0  # RNFL_GCL
    new_mask[mask == 1] = 1  # INL
    new_mask[mask == 2] = 1  # ONL -> INL_OPL_ONL
    new_mask[mask == 3] = 1  # IS_OS -> assign to INL_OPL_ONL
    new_mask[mask == 4] = 2  # RPE_Choroid
    return new_mask


def extract_is_os_boundary(mask_5class):
    """Extract IS_OS boundary (class 3)."""
    return (mask_5class == 3).astype(np.float32)


def add_synthetic_noise(image, noise_level, calibrated=False):
    """Add synthetic OCT noise (speckle + Gaussian).

    Args:
        image: Clean image (0-1 range)
        noise_level: Gaussian noise std (or speckle scale if calibrated=True)
        calibrated: If True, use Duke17-calibrated noise model
                   (speckle_scale=0.4 gives PSNR ~17 dB matching Duke17)

    Note on SOTA comparison:
        Duke17 dataset: Noisy PSNR ~17 dB, SSIM ~0.31 (masked, threshold=100)
        Original model (noise_level=0.15): PSNR ~13 dB (more aggressive)
        Calibrated model (noise_level=0.4): PSNR ~17 dB (matches Duke17)
    """
    if calibrated:
        # Duke17-calibrated noise model
        # speckle_scale of 0.4 gives ~17 dB PSNR matching Duke17
        speckle_scale = noise_level
        speckle = 1.0 + speckle_scale * (np.random.exponential(1.0, image.shape) - 1.0)
        noisy = image * speckle
        gaussian = np.random.normal(0, 0.02, image.shape)
        noisy = noisy + gaussian
    else:
        # Original aggressive noise model (for challenging training)
        # Speckle noise (multiplicative)
        speckle = np.random.exponential(1.0, image.shape)
        noisy = image * speckle
        # Gaussian noise (additive)
        gaussian = np.random.normal(0, noise_level, image.shape)
        noisy = noisy + gaussian
    return np.clip(noisy, 0, 1)


# Import calibrated noise generator
try:
    from calibrated_oct_noise import add_calibrated_oct_noise, NOISE_PARAMS
    HAS_CALIBRATED_NOISE = True
except ImportError:
    HAS_CALIBRATED_NOISE = False
    print("[Warning] calibrated_oct_noise.py not found, using simple noise model")

# Dataset noise profiles for 3-group training
NOISE_PROFILES = ['duke17', 'duke28', 'pku37']


def add_calibrated_noise_by_profile(image, profile='duke17', noise_scale=1.0):
    """Add calibrated OCT noise matching specific dataset profile.

    This function is used when training images are divided into 3 groups,
    each with different noise characteristics matching real datasets.

    Args:
        image: Clean image (0-1 range)
        profile: 'duke17', 'duke28', or 'pku37'
        noise_scale: Scale factor for noise level (default 1.0 matches real data)

    Returns:
        Noisy image with calibrated noise
    """
    if HAS_CALIBRATED_NOISE:
        return add_calibrated_oct_noise(image, dataset=profile, noise_scale=noise_scale, random_mix=True)

    # Fallback to simple calibrated noise if module not available
    # Parameters calibrated to match real dataset PSNR levels:
    # Duke17: PSNR ~17 dB, Duke28: PSNR ~17.5 dB, PKU37: PSNR ~20 dB
    params = {
        'duke17': {'speckle_scale': 0.50, 'gaussian_std': 0.10, 'v_corr': 0.19},  # PSNR ~17 dB
        'duke28': {'speckle_scale': 0.48, 'gaussian_std': 0.10, 'v_corr': 0.20},  # PSNR ~17.5 dB
        'pku37':  {'speckle_scale': 0.35, 'gaussian_std': 0.07, 'v_corr': 0.43},  # PSNR ~20 dB
        'combined': {'speckle_scale': 0.45, 'gaussian_std': 0.09, 'v_corr': 0.28}, # PSNR ~18 dB
    }
    p = params.get(profile, params['combined'])

    H, W = image.shape

    # Speckle noise with spatial correlation
    speckle = 1.0 + p['speckle_scale'] * noise_scale * (np.random.exponential(1.0, (H, W)) - 1.0)

    # Add vertical correlation to speckle
    if p['v_corr'] > 0.1:
        phi = p['v_corr']
        corr_speckle = np.zeros_like(speckle)
        corr_speckle[0, :] = speckle[0, :]
        for i in range(1, H):
            corr_speckle[i, :] = phi * corr_speckle[i-1, :] + np.sqrt(1 - phi**2) * speckle[i, :]
        speckle = 1.0 + (corr_speckle - np.mean(corr_speckle)) * p['speckle_scale'] * noise_scale

    noisy = image * np.maximum(speckle, 0.01)

    # Gaussian noise with vertical correlation
    gaussian = np.random.randn(H, W).astype(np.float32) * p['gaussian_std'] * noise_scale
    if p['v_corr'] > 0.1:
        phi = p['v_corr']
        corr_gaussian = np.zeros_like(gaussian)
        corr_gaussian[0, :] = gaussian[0, :]
        for i in range(1, H):
            corr_gaussian[i, :] = phi * corr_gaussian[i-1, :] + np.sqrt(1 - phi**2) * gaussian[i, :]
        gaussian = corr_gaussian * (p['gaussian_std'] * noise_scale / (np.std(corr_gaussian) + 1e-8))

    noisy = noisy + gaussian

    return np.clip(noisy, 0, 1).astype(np.float32)


# ============================================================================
# Dataset Classes - Random Patch Training (Like Phase 1A-1)
# ============================================================================
class JointTrainDataset(Dataset):
    """Training dataset with random patches and noise augmentation (like Phase 1A-1).

    Key Feature: Divides samples into 3 groups with different noise characteristics:
    - Group 1 (~33%): Duke17 noise profile (PSNR ~17 dB, strong speckle)
    - Group 2 (~33%): Duke28 noise profile (PSNR ~17.5 dB, similar to Duke17)
    - Group 3 (~33%): PKU37 noise profile (PSNR ~20 dB, less noisy, more vertical correlation)

    This helps the model generalize to different noise profiles encountered in real OCT data.
    """

    def __init__(self, jsonl_path, max_samples=None, noise_levels=None, patch_size=64,
                 use_calibrated_noise=True, noise_scale_range=(0.8, 1.2)):
        """
        Args:
            jsonl_path: Path to JSONL file with sample paths
            max_samples: Maximum samples to use (None = all)
            noise_levels: Noise scale factors for augmentation (default: [0.8, 0.9, 1.0, 1.1, 1.2])
            patch_size: Size of random patches
            use_calibrated_noise: If True, use calibrated noise matching real datasets
            noise_scale_range: Range for random noise scaling (min, max)
        """
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.noise_levels = noise_levels or [0.8, 0.9, 1.0, 1.1, 1.2]  # Scale factors
        self.patch_size = patch_size
        self.use_calibrated_noise = use_calibrated_noise
        self.noise_scale_range = noise_scale_range

        # Divide samples into 3 groups for different noise profiles
        np.random.seed(42)  # Reproducible assignment
        n_samples = len(self.samples)
        group_assignments = np.random.choice(3, size=n_samples)
        self.sample_noise_profiles = [NOISE_PROFILES[g] for g in group_assignments]

        # Print group statistics
        group_counts = [np.sum(group_assignments == i) for i in range(3)]
        print(f"[JointTrainDataset] Noise profile groups:")
        for i, (profile, count) in enumerate(zip(NOISE_PROFILES, group_counts)):
            pct = count / n_samples * 100
            print(f"  Group {i+1} ({profile}): {count} samples ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples) * len(self.noise_levels)

    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        noise_idx = idx // len(self.samples)
        noise_scale = self.noise_levels[noise_idx]

        sample = self.samples[sample_idx]
        clean_image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        H, W = clean_image.shape

        # Layer-aware patch extraction: try to get patches with multiple layers
        # This ensures the model sees all layer types during training
        max_attempts = 10
        best_patch = None
        best_n_classes = 0

        for attempt in range(max_attempts):
            top = np.random.randint(0, max(1, H - self.patch_size))
            left = np.random.randint(0, max(1, W - self.patch_size))

            mask_patch = mask_5class[top:top+self.patch_size, left:left+self.patch_size]
            mask_3class_temp = remap_mask_to_3class(mask_patch)
            n_classes = len(np.unique(mask_3class_temp))

            if n_classes >= 2:
                # Good patch with multiple layers
                clean_patch = clean_image[top:top+self.patch_size, left:left+self.patch_size]
                break
            elif n_classes > best_n_classes:
                # Keep track of best patch so far
                best_n_classes = n_classes
                best_patch = (top, left)
        else:
            # Use best patch found if no multi-layer patch found
            if best_patch:
                top, left = best_patch
            clean_patch = clean_image[top:top+self.patch_size, left:left+self.patch_size]
            mask_patch = mask_5class[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Add calibrated noise based on sample's assigned profile
        if self.use_calibrated_noise:
            noise_profile = self.sample_noise_profiles[sample_idx]
            noisy_patch = add_calibrated_noise_by_profile(clean_patch, profile=noise_profile, noise_scale=noise_scale)
        else:
            # Fallback to old simple noise
            noisy_patch = add_synthetic_noise(clean_patch, noise_scale * 0.15)

        mask_3class = remap_mask_to_3class(mask_patch)
        is_os_boundary = extract_is_os_boundary(mask_patch)

        return {
            'noisy': torch.from_numpy(noisy_patch).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_patch).float().unsqueeze(0),
            'mask_3class': torch.from_numpy(mask_3class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
        }


class JointValDataset(Dataset):
    """Validation dataset with full images and calibrated noise.

    Uses 'combined' noise profile for validation to represent average real-world OCT noise.
    """

    def __init__(self, jsonl_path, max_samples=None, noise_level=0.15,
                 use_calibrated_noise=True, noise_profile='combined'):
        """
        Args:
            jsonl_path: Path to JSONL file with sample paths
            max_samples: Maximum samples to use
            noise_level: Noise scale factor (1.0 = match real data)
            use_calibrated_noise: If True, use calibrated noise matching real datasets
            noise_profile: Noise profile to use ('duke17', 'duke28', 'pku37', 'combined')
        """
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.noise_level = noise_level
        self.use_calibrated_noise = use_calibrated_noise
        self.noise_profile = noise_profile

        if use_calibrated_noise:
            print(f"[JointValDataset] Using calibrated noise profile: {noise_profile}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clean_image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        if self.use_calibrated_noise:
            noisy_image = add_calibrated_noise_by_profile(
                clean_image, profile=self.noise_profile, noise_scale=self.noise_level
            )
        else:
            noisy_image = add_synthetic_noise(clean_image, self.noise_level)

        mask_3class = remap_mask_to_3class(mask_5class)
        is_os_boundary = extract_is_os_boundary(mask_5class)

        return {
            'noisy': torch.from_numpy(noisy_image).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_image).float().unsqueeze(0),
            'mask_3class': torch.from_numpy(mask_3class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
        }


# ============================================================================
# SimpleBoundarySegmenter - Proven V4 Architecture
# ============================================================================
class SimpleBoundarySegmenter(nn.Module):
    """Proven V4 architecture that achieved 0.48+ IS_OS Boundary Dice."""

    def __init__(self, num_classes=3):
        super().__init__()

        self.enc1 = self._block(1, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)

        self.seg_head = nn.Conv2d(32, num_classes, 1)

        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seg_logits = self.seg_head(d1)
        boundary_logits = self.boundary_head(d1)

        return seg_logits, boundary_logits


# ============================================================================
# Layer-Specific Noise Gates (Key TMI Contribution)
# ============================================================================
class LayerSpecificNoiseGates(nn.Module):
    """Layer-specific noise-adaptive gates for anatomically-aware denoising."""

    def __init__(self, noise_feat_dim=7, n_layers=3, hidden_dim=16):
        super().__init__()
        self.n_layers = n_layers

        # Per-layer noise response networks
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(noise_feat_dim, hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.Sigmoid(),
            ) for _ in range(n_layers)
        ])

    def forward(self, noise_features, seg_probs):
        """
        Args:
            noise_features: [B, 7, H, W] physics-based noise features
            seg_probs: [B, n_layers, H, W] segmentation probabilities

        Returns:
            combined_gate: [B, 1, H, W] weighted combination of layer gates
            layer_gates: [B, n_layers, H, W] individual layer gates
        """
        B, _, H, W = noise_features.shape

        # Compute per-layer gates
        gates = []
        for i, gate_net in enumerate(self.layer_gates):
            gate = gate_net(noise_features)  # [B, 1, H, W]
            gates.append(gate)

        layer_gates = torch.cat(gates, dim=1)  # [B, n_layers, H, W]

        # Combine gates weighted by segmentation probabilities
        combined_gate = (layer_gates * seg_probs).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Compute stats for logging
        layer_gate_stats = {
            f'gate_{CLASS_NAMES[i]}': layer_gates[:, i].mean().item()
            for i in range(self.n_layers)
        }

        return combined_gate, layer_gates, layer_gate_stats


# ============================================================================
# Joint Denoising + Segmentation Model (Integrated Architecture)
# ============================================================================
class JointDenoiserSegmenter(nn.Module):
    """
    Integrated Joint Denoising + Segmentation with IS_OS Boundary Detection.

    Combines:
    - NAFNet backbone for high-quality denoising
    - SimpleBoundarySegmenter for 3-class segmentation + IS_OS boundary
    - Layer-specific noise gates
    - Confidence-weighted denoising
    - Physics-based noise features
    """

    def __init__(self, nafnet_ckpt=None):
        super().__init__()

        # 1. Physics-based noise feature extractor
        self.feature_extractor = NoiseFeatureExtractor()

        # 2. NAFNet backbone for denoising
        self.backbone = NAFNetFullFiLM(
            img_channel=1, width=64,
            enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
            middle_blk_num=2, cond_dim=32,
        )

        if nafnet_ckpt and os.path.exists(nafnet_ckpt):
            ckpt = torch.load(nafnet_ckpt, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            print(f"[JointDenoiserSegmenter] Loaded NAFNet backbone from {nafnet_ckpt}")

        # 3. SimpleBoundarySegmenter (3-class + IS_OS boundary)
        self.segmenter = SimpleBoundarySegmenter(num_classes=NUM_CLASSES)

        # 4. Feature normalization
        self.feature_norm = nn.Sequential(
            nn.Conv2d(7, 16, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1),
        )

        # 5. Layer-specific noise gates (Key TMI Contribution)
        self.layer_noise_gates = LayerSpecificNoiseGates(
            noise_feat_dim=7, n_layers=NUM_CLASSES, hidden_dim=16
        )

        # 6. Layer-aware refinement
        # Input: backbone (1) + norm_features (8) + seg_logits (3)
        self.refinement = nn.Sequential(
            nn.Conv2d(1 + 8 + NUM_CLASSES, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))

        # 7. Confidence-weighted denoising (Key TMI Contribution)
        self.confidence_min_gate = nn.Parameter(torch.tensor(0.3))
        self.max_entropy = np.log(NUM_CLASSES)

        # 8. Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(1 + 8 + NUM_CLASSES + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def load_segmenter_checkpoint(self, checkpoint_path, device='cpu'):
        """Load checkpoint from SimpleBoundarySegmenter (Phase 1A-1)."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle both raw state_dict and wrapped checkpoint
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Load into segmenter
        self.segmenter.load_state_dict(state_dict, strict=True)
        print(f"[JointDenoiserSegmenter] Loaded segmenter from {checkpoint_path}")
        return True

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: Noisy input [B, 1, H, W]
            return_features: Whether to return intermediate features

        Returns:
            denoised: Denoised output [B, 1, H, W]
            seg_logits: Segmentation logits [B, 3, H, W]
            boundary_logits: IS_OS boundary logits [B, 1, H, W]
        """
        # 1. Extract physics-based noise features
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

        # 2. NAFNet backbone denoising
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # 3. Segmentation + IS_OS boundary detection on DENOISED output
        # (segmenter was trained on clean images, so feed it denoised output)
        seg_logits, boundary_logits = self.segmenter(backbone_out)
        seg_probs = F.softmax(seg_logits, dim=1)

        # 4. Compute segmentation confidence
        seg_probs_clamped = seg_probs.clamp(min=1e-8)
        seg_entropy = -(seg_probs * seg_probs_clamped.log()).sum(dim=1)
        seg_confidence = 1 - (seg_entropy / self.max_entropy)

        # 5. Layer-specific noise gates
        noise_gate, layer_gates, layer_gate_stats = self.layer_noise_gates(
            feature_stack, seg_probs
        )

        # 6. Layer-guided refinement
        refine_input = torch.cat([backbone_out, norm_features, seg_logits], dim=1)
        refinement_raw = self.refinement(refine_input)

        # 7. Apply layer-specific noise gate
        refinement_gated = refinement_raw * noise_gate

        # 8. Confidence-weighted denoising
        confidence_map = seg_confidence.unsqueeze(1)
        min_gate = torch.sigmoid(self.confidence_min_gate)
        confidence_gate = min_gate + (1 - min_gate) * confidence_map
        refinement = refinement_gated * confidence_gate

        # 9. Final denoised output
        scale = torch.sigmoid(self.refinement_scale) * 0.2
        denoised = backbone_out + scale * refinement
        denoised = denoised.clamp(0, 1)

        # 10. Uncertainty estimation
        uncertainty_input = torch.cat([backbone_out, norm_features, seg_logits, noise_gate], dim=1)
        uncertainty = self.uncertainty_head(uncertainty_input)

        if return_features:
            return denoised, seg_logits, boundary_logits, {
                'backbone_out': backbone_out,
                'noise_gate': noise_gate,
                'layer_gates': layer_gates,
                'layer_gate_stats': layer_gate_stats,
                'seg_confidence': seg_confidence.mean().item(),
                'confidence_gate': confidence_gate,
                'uncertainty': uncertainty,
            }

        return denoised, seg_logits, boundary_logits


# ============================================================================
# Inference and Metrics
# ============================================================================
def sliding_window_inference(model, image, patch_size=64, stride=32, device='cpu', return_backbone=False):
    """Perform inference on full image using sliding window with overlap.

    Args:
        model: The joint denoiser-segmenter model
        image: Input image [B, 1, H, W]
        patch_size: Size of patches for inference
        stride: Stride between patches
        device: Device to run on
        return_backbone: If True, also return the backbone (base NAFNet) output

    Returns:
        denoised, seg_logits, boundary_logits, [backbone_out if return_backbone]
    """
    model.eval()
    B, C, H, W = image.shape

    # Ensure image is at least patch_size in each dimension
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)

    # Also pad to align with stride
    if (H + pad_h) % stride != 0:
        pad_h += stride - ((H + pad_h) % stride)
    if (W + pad_w) % stride != 0:
        pad_w += stride - ((W + pad_w) % stride)

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')

    _, _, H_pad, W_pad = image.shape

    denoise_output = torch.zeros(B, 1, H_pad, W_pad, device=device)
    seg_output = torch.zeros(B, NUM_CLASSES, H_pad, W_pad, device=device)
    boundary_output = torch.zeros(B, 1, H_pad, W_pad, device=device)
    backbone_output = torch.zeros(B, 1, H_pad, W_pad, device=device) if return_backbone else None
    count = torch.zeros(B, 1, H_pad, W_pad, device=device)

    with torch.no_grad():
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = image[:, :, y:y+patch_size, x:x+patch_size]

                if return_backbone:
                    denoised, seg_logits, boundary_logits, features = model(patch, return_features=True)
                    backbone_output[:, :, y:y+patch_size, x:x+patch_size] += features['backbone_out']
                else:
                    denoised, seg_logits, boundary_logits = model(patch)

                denoise_output[:, :, y:y+patch_size, x:x+patch_size] += denoised
                seg_output[:, :, y:y+patch_size, x:x+patch_size] += seg_logits
                boundary_output[:, :, y:y+patch_size, x:x+patch_size] += boundary_logits
                count[:, :, y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping regions
    denoise_output = denoise_output / count.clamp(min=1)
    seg_output = seg_output / count.clamp(min=1)
    boundary_output = boundary_output / count.clamp(min=1)
    if return_backbone:
        backbone_output = backbone_output / count.clamp(min=1)

    # Crop back to original size and free memory
    denoised_result = denoise_output[:, :, :H, :W].clone()
    seg_result = seg_output[:, :, :H, :W].clone()
    boundary_result = boundary_output[:, :, :H, :W].clone()

    if return_backbone:
        backbone_result = backbone_output[:, :, :H, :W].clone()
        del denoise_output, seg_output, boundary_output, backbone_output, count
        return denoised_result, seg_result, boundary_result, backbone_result

    del denoise_output, seg_output, boundary_output, count
    return denoised_result, seg_result, boundary_result


def compute_dice(pred, target, num_classes):
    """Compute per-class Dice scores."""
    dice_scores = []
    pred_classes = pred.argmax(dim=1)

    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2 * intersection / union).item()
        else:
            dice = 1.0 if intersection == 0 else 0.0
        dice_scores.append(dice)

    return dice_scores


def compute_boundary_dice(pred_boundary, target_boundary, threshold=0.5):
    """Compute Dice for boundary detection."""
    pred_binary = (torch.sigmoid(pred_boundary) > threshold).float()

    intersection = (pred_binary * target_boundary).sum()
    union = pred_binary.sum() + target_boundary.sum()

    if union > 0:
        return (2 * intersection / union).item()
    return 1.0 if intersection == 0 else 0.0


def compute_psnr(pred, target):
    """Compute PSNR between predicted and target images."""
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 50.0
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM between predicted and target images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)

    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2)
    mu2 = F.conv2d(target, window, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_psnr_masked(pred, target, mask):
    """Compute PSNR for a specific masked region."""
    if mask.sum() < 100:  # Skip if region too small
        return None
    pred_masked = pred[mask]
    target_masked = target[mask]
    mse = F.mse_loss(pred_masked, target_masked)
    if mse < 1e-10:
        return 50.0
    return (10 * torch.log10(1.0 / mse)).item()


def compute_sota_metrics(pred, target, intensity_threshold=100/255):
    """
    Compute PSNR and SSIM matching SOTA paper methodology (SNA-SKAN, etc.).

    Key insight: SOTA papers compute metrics only on "meaningful" pixels
    where the clean image intensity is above a threshold. This focuses
    metrics on actual retinal structure, ignoring dark background.

    Args:
        pred: Predicted/denoised image [B, 1, H, W] (0-1 range)
        target: Clean target image [B, 1, H, W] (0-1 range)
        intensity_threshold: Pixel intensity threshold (default 100/255 for 0-1 range)

    Returns:
        dict with sota_psnr, sota_ssim
    """
    # Create mask based on clean image intensity (retinal tissue region)
    mask = target > intensity_threshold  # [B, 1, H, W]
    n_pixels = mask.sum().item()

    if n_pixels < 1000:
        return {'sota_psnr': None, 'sota_ssim': None, 'mask_ratio': 0}

    # Extract masked pixels
    pred_masked = pred[mask]
    target_masked = target[mask]

    # Compute PSNR on masked region
    mse = F.mse_loss(pred_masked, target_masked)
    if mse < 1e-10:
        psnr_val = 50.0
    else:
        psnr_val = (10 * torch.log10(1.0 / mse)).item()

    # Compute SSIM on masked region
    # Reshape to square for spatial SSIM computation (matching SOTA methodology)
    side = int(np.sqrt(n_pixels))
    if side < 7:  # SSIM needs minimum window size
        ssim_val = None
    else:
        pred_sq = pred_masked[:side*side].reshape(1, 1, side, side)
        target_sq = target_masked[:side*side].reshape(1, 1, side, side)

        # Use our SSIM function on the reshaped square
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        window_size = min(11, side - 1 if side % 2 == 0 else side)
        if window_size < 3:
            window_size = 3

        sigma = 1.5
        device = pred.device
        coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)

        mu1 = F.conv2d(pred_sq, window, padding=window_size // 2)
        mu2 = F.conv2d(target_sq, window, padding=window_size // 2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(pred_sq ** 2, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target_sq ** 2, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred_sq * target_sq, window, padding=window_size // 2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_val = ssim_map.mean().item()

    return {
        'sota_psnr': psnr_val,
        'sota_ssim': ssim_val,
        'mask_ratio': n_pixels / target.numel()
    }


def compute_ssim_masked(pred, target, mask, window_size=11):
    """Compute SSIM for a specific masked region using bounding box."""
    # Find bounding box of mask
    mask_2d = mask.squeeze()
    if mask_2d.sum() < 100:
        return None

    rows = torch.any(mask_2d, dim=1)
    cols = torch.any(mask_2d, dim=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]

    # Add padding for SSIM window
    pad = window_size // 2
    rmin = max(0, rmin - pad)
    rmax = min(mask_2d.shape[0], rmax + pad + 1)
    cmin = max(0, cmin - pad)
    cmax = min(mask_2d.shape[1], cmax + pad + 1)

    # Extract region
    pred_region = pred[:, :, rmin:rmax, cmin:cmax]
    target_region = target[:, :, rmin:rmax, cmin:cmax]
    mask_region = mask[:, :, rmin:rmax, cmin:cmax]

    # Compute SSIM on the region
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(pred_region, window, padding=window_size // 2)
    mu2 = F.conv2d(target_region, window, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_region ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target_region ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred_region * target_region, window, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Apply mask and compute mean
    masked_ssim = ssim_map * mask_region
    return (masked_ssim.sum() / mask_region.sum()).item()


# ============================================================================
# Clinical Metrics
# ============================================================================
def compute_clinical_metrics(seg_logits, gt_mask, boundary_logits, gt_boundary):
    """
    Compute clinically relevant metrics for OCT layer segmentation.

    Returns:
        dict with:
        - layer_thickness: Mean thickness (in pixels) per layer
        - thickness_error: Absolute error vs ground truth
        - boundary_mae: Mean absolute error for IS_OS boundary localization
        - layer_coverage: Percentage of image covered by each layer
    """
    pred_mask = seg_logits.argmax(dim=1)  # [B, H, W]
    B, H, W = pred_mask.shape

    metrics = {
        'layer_thickness_pred': {},
        'layer_thickness_gt': {},
        'thickness_error': {},
        'layer_coverage_pred': {},
        'layer_coverage_gt': {},
    }

    # Compute per-layer thickness (average height span per column)
    for c, name in enumerate(CLASS_NAMES):
        pred_layer = (pred_mask == c).float()  # [B, H, W]
        gt_layer = (gt_mask == c).float()

        # Thickness per column: sum of pixels in that column belonging to layer
        pred_thickness_per_col = pred_layer.sum(dim=1)  # [B, W]
        gt_thickness_per_col = gt_layer.sum(dim=1)

        # Mean thickness across columns (excluding empty columns)
        pred_valid = pred_thickness_per_col > 0
        gt_valid = gt_thickness_per_col > 0

        if pred_valid.sum() > 0:
            pred_mean_thickness = pred_thickness_per_col[pred_valid].mean().item()
        else:
            pred_mean_thickness = 0.0

        if gt_valid.sum() > 0:
            gt_mean_thickness = gt_thickness_per_col[gt_valid].mean().item()
        else:
            gt_mean_thickness = 0.0

        metrics['layer_thickness_pred'][name] = pred_mean_thickness
        metrics['layer_thickness_gt'][name] = gt_mean_thickness
        metrics['thickness_error'][name] = abs(pred_mean_thickness - gt_mean_thickness)

        # Coverage
        metrics['layer_coverage_pred'][name] = (pred_layer.sum() / (B * H * W)).item() * 100
        metrics['layer_coverage_gt'][name] = (gt_layer.sum() / (B * H * W)).item() * 100

    # IS_OS Boundary localization error (Mean Absolute Error in y-position)
    pred_boundary = (torch.sigmoid(boundary_logits) > 0.5).float()

    # Find y-position of boundary per column
    boundary_mae_list = []
    for b in range(B):
        for x in range(W):
            pred_col = pred_boundary[b, 0, :, x]
            gt_col = gt_boundary[b, 0, :, x]

            pred_positions = torch.where(pred_col > 0.5)[0]
            gt_positions = torch.where(gt_col > 0.5)[0]

            if len(pred_positions) > 0 and len(gt_positions) > 0:
                # Use centroid of boundary pixels
                pred_y = pred_positions.float().mean()
                gt_y = gt_positions.float().mean()
                boundary_mae_list.append(abs(pred_y - gt_y).item())

    metrics['boundary_mae'] = np.mean(boundary_mae_list) if boundary_mae_list else float('nan')

    return metrics


def compute_ordering_verification(seg_logits):
    """
    Verify that predicted segmentation respects anatomical layer ordering.

    Returns:
        dict with:
        - is_valid: Boolean, True if ordering is correct
        - expected_y: Expected y-position for each layer
        - ordering_gaps: Gap between adjacent layers (should be positive)
        - violations: List of any ordering violations
    """
    B, C, H, W = seg_logits.shape
    device = seg_logits.device

    probs = F.softmax(seg_logits, dim=1)

    # Create y-coordinate grid
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    y_coords = y_coords.expand(B, 1, H, W)

    # Compute expected y-position for each class
    expected_y = {}
    for c, name in enumerate(CLASS_NAMES):
        prob_c = probs[:, c:c+1, :, :]
        weighted_y = (prob_c * y_coords).sum()
        prob_sum = prob_c.sum() + 1e-8
        expected_y[name] = (weighted_y / prob_sum).item()

    # Check ordering: RNFL_GCL < INL_OPL_ONL < RPE_Choroid
    gaps = {}
    violations = []

    layer_order = ['RNFL_GCL', 'INL_OPL_ONL', 'RPE_Choroid']
    for i in range(len(layer_order) - 1):
        upper = layer_order[i]
        lower = layer_order[i + 1]
        gap = expected_y[lower] - expected_y[upper]
        gaps[f'{upper}->{lower}'] = gap

        if gap < 0:
            violations.append(f'{upper} below {lower} (gap={gap:.1f}px)')

    return {
        'is_valid': len(violations) == 0,
        'expected_y': expected_y,
        'ordering_gaps': gaps,
        'violations': violations,
        'n_violations': len(violations),
    }


def compute_adaptive_denoising_metrics(features):
    """
    Compute metrics showing adaptive behavior of layer-specific denoising.

    Args:
        features: Dict from model forward with return_features=True

    Returns:
        dict with:
        - layer_gate_means: Mean gate activation per layer (shows differentiation)
        - gate_std: Std of gate activations (higher = more adaptive)
        - confidence_mean: Mean segmentation confidence
        - gate_range: Range between min/max layer gates (shows layer-specificity)
    """
    metrics = {}

    # Layer-specific gate statistics
    if 'layer_gate_stats' in features:
        gate_stats = features['layer_gate_stats']
        metrics['layer_gate_means'] = gate_stats

        gate_values = list(gate_stats.values())
        metrics['gate_std'] = np.std(gate_values)
        metrics['gate_range'] = max(gate_values) - min(gate_values)
        metrics['gate_min'] = min(gate_values)
        metrics['gate_max'] = max(gate_values)

    # Segmentation confidence
    if 'seg_confidence' in features:
        metrics['seg_confidence'] = features['seg_confidence']

    # Noise gate spatial statistics
    if 'noise_gate' in features:
        noise_gate = features['noise_gate']
        metrics['noise_gate_mean'] = noise_gate.mean().item()
        metrics['noise_gate_std'] = noise_gate.std().item()

    # Uncertainty statistics
    if 'uncertainty' in features:
        uncertainty = features['uncertainty']
        metrics['uncertainty_mean'] = uncertainty.mean().item()
        metrics['uncertainty_std'] = uncertainty.std().item()

    return metrics


# ============================================================================
# Neuro-Symbolic Ordering Loss (Key TMI Contribution)
# ============================================================================
def compute_symbolic_ordering_loss(seg_logits, margin=2.0):
    """
    Neuro-Symbolic Ordering Loss: Penalizes anatomical layer ordering violations.

    KEY TMI CONTRIBUTION: Enforces hard anatomical constraints in a differentiable way.

    In OCT images, retinal layers MUST follow a strict top-to-bottom ordering:
        Layer 0 (RNFL_GCL) - topmost
        Layer 1 (INL_OPL_ONL) - middle
        Layer 2 (RPE_Choroid) - bottommost

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


# ============================================================================
# Sliding Window Training on Full Images
# ============================================================================
def sliding_window_train_step(model, noisy, clean, mask_3class, is_os_boundary,
                               patch_size, stride, device, class_weights,
                               boundary_weight, pos_weight, denoise_weight,
                               lambda_symbolic_ordering):
    """
    Train on a full image using sliding window with gradient accumulation.

    This allows training on whole images while fitting in memory by:
    1. Processing patches sequentially
    2. Accumulating gradients across all patches
    3. Updating weights once per image
    """
    B, C, H, W = noisy.shape
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    # Pad image if needed
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride

    if pad_h > 0 or pad_w > 0:
        noisy = F.pad(noisy, (0, pad_w, 0, pad_h), mode='reflect')
        clean = F.pad(clean, (0, pad_w, 0, pad_h), mode='reflect')
        mask_3class = F.pad(mask_3class.unsqueeze(1).float(), (0, pad_w, 0, pad_h), mode='reflect').squeeze(1).long()
        is_os_boundary = F.pad(is_os_boundary, (0, pad_w, 0, pad_h), mode='reflect')

    _, _, H_pad, W_pad = noisy.shape

    # Accumulate outputs for full-image metrics
    total_loss = torch.tensor(0.0, device=device)
    total_seg_loss = 0
    total_boundary_loss = 0
    total_denoise_loss = 0
    total_ordering_loss = 0
    n_patches = 0

    # Accumulate predictions for metrics
    seg_output = torch.zeros(B, NUM_CLASSES, H_pad, W_pad, device=device)
    boundary_output = torch.zeros(B, 1, H_pad, W_pad, device=device)
    denoise_output = torch.zeros(B, 1, H_pad, W_pad, device=device)
    count = torch.zeros(B, 1, H_pad, W_pad, device=device)

    # Process patches
    for y in range(0, H_pad - patch_size + 1, stride):
        for x in range(0, W_pad - patch_size + 1, stride):
            noisy_patch = noisy[:, :, y:y+patch_size, x:x+patch_size]
            clean_patch = clean[:, :, y:y+patch_size, x:x+patch_size]
            mask_patch = mask_3class[:, y:y+patch_size, x:x+patch_size]
            boundary_patch = is_os_boundary[:, :, y:y+patch_size, x:x+patch_size]

            # Forward pass
            denoised, seg_logits, boundary_logits = model(noisy_patch)

            # Segmentation loss
            seg_ce = F.cross_entropy(seg_logits, mask_patch, weight=class_weights)
            seg_probs = F.softmax(seg_logits, dim=1)

            dice_loss = 0
            for c in range(NUM_CLASSES):
                pred_c = seg_probs[:, c]
                target_c = (mask_patch == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum() + 1e-6
                dice_loss += 1 - (2 * intersection / union)
            dice_loss /= NUM_CLASSES
            seg_loss = seg_ce + dice_loss

            # Boundary loss
            boundary_bce = F.binary_cross_entropy_with_logits(
                boundary_logits, boundary_patch, pos_weight=pos_weight_tensor
            )
            boundary_probs = torch.sigmoid(boundary_logits)
            b_intersection = (boundary_probs * boundary_patch).sum()
            b_union = boundary_probs.sum() + boundary_patch.sum() + 1e-6
            boundary_dice_loss = 1 - (2 * b_intersection / b_union)
            boundary_loss = boundary_bce + boundary_dice_loss

            # Denoising loss
            denoise_l1 = F.l1_loss(denoised, clean_patch)
            denoise_mse = F.mse_loss(denoised, clean_patch)
            denoise_loss = denoise_l1 + 0.1 * denoise_mse

            # Neuro-symbolic ordering loss
            ordering_loss, _ = compute_symbolic_ordering_loss(seg_logits, margin=2.0)

            # Patch loss
            patch_loss = (seg_loss
                         + boundary_weight * boundary_loss
                         + denoise_weight * denoise_loss
                         + lambda_symbolic_ordering * ordering_loss)

            # Accumulate loss (will backward after all patches)
            total_loss = total_loss + patch_loss

            # Track losses for logging
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_denoise_loss += denoise_loss.item()
            total_ordering_loss += ordering_loss.item()
            n_patches += 1

            # Accumulate predictions (detached for metrics)
            with torch.no_grad():
                seg_output[:, :, y:y+patch_size, x:x+patch_size] += seg_logits
                boundary_output[:, :, y:y+patch_size, x:x+patch_size] += boundary_logits
                denoise_output[:, :, y:y+patch_size, x:x+patch_size] += denoised
                count[:, :, y:y+patch_size, x:x+patch_size] += 1

    # Average loss across patches
    total_loss = total_loss / n_patches

    # Compute full-image predictions for metrics
    with torch.no_grad():
        seg_output = seg_output / count.clamp(min=1)
        boundary_output = boundary_output / count.clamp(min=1)
        denoise_output = denoise_output / count.clamp(min=1)

        # Crop back to original size
        seg_output = seg_output[:, :, :H, :W]
        boundary_output = boundary_output[:, :, :H, :W]
        denoise_output = denoise_output[:, :, :H, :W]
        mask_3class_orig = mask_3class[:, :H, :W]
        is_os_boundary_orig = is_os_boundary[:, :, :H, :W]
        clean_orig = clean[:, :, :H, :W]

        # Compute metrics on full image
        dice_scores = compute_dice(seg_output, mask_3class_orig, NUM_CLASSES)
        b_dice = compute_boundary_dice(boundary_output, is_os_boundary_orig)
        psnr = compute_psnr(denoise_output, clean_orig)

    return {
        'loss': total_loss,
        'seg_loss': total_seg_loss / n_patches,
        'boundary_loss': total_boundary_loss / n_patches,
        'denoise_loss': total_denoise_loss / n_patches,
        'ordering_loss': total_ordering_loss / n_patches,
        'dice': dice_scores,
        'boundary_dice': b_dice,
        'psnr': psnr,
    }


# ============================================================================
# Training and Validation
# ============================================================================
def train_epoch(model, train_loader, optimizer, device, class_weights,
                boundary_weight=2.0, pos_weight=10.0, denoise_weight=1.0,
                lambda_symbolic_ordering=0.1):
    """Train for one epoch using random patches with DataLoader (like Phase 1A-1)."""
    model.train()
    total_seg_loss = 0
    total_boundary_loss = 0
    total_denoise_loss = 0
    total_ordering_loss = 0
    total_dice = [0] * NUM_CLASSES
    total_boundary_dice = 0
    total_psnr = 0
    n_batches = 0

    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    for batch in tqdm(train_loader, desc='Training'):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask_3class = batch['mask_3class'].to(device)
        is_os_boundary = batch['is_os_boundary'].to(device)

        optimizer.zero_grad()

        # Forward pass
        denoised, seg_logits, boundary_logits = model(noisy)

        # Segmentation loss (CE + Dice)
        seg_ce = F.cross_entropy(seg_logits, mask_3class, weight=class_weights)
        seg_probs = F.softmax(seg_logits, dim=1)

        dice_loss = 0
        for c in range(NUM_CLASSES):
            pred_c = seg_probs[:, c]
            target_c = (mask_3class == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() + 1e-6
            dice_loss += 1 - (2 * intersection / union)
        dice_loss /= NUM_CLASSES
        seg_loss = seg_ce + dice_loss

        # Boundary loss (BCE + Dice)
        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_logits, is_os_boundary, pos_weight=pos_weight_tensor
        )
        boundary_probs = torch.sigmoid(boundary_logits)
        b_intersection = (boundary_probs * is_os_boundary).sum()
        b_union = boundary_probs.sum() + is_os_boundary.sum() + 1e-6
        boundary_dice_loss = 1 - (2 * b_intersection / b_union)
        boundary_loss = boundary_bce + boundary_dice_loss

        # Denoising loss (L1 + MSE)
        denoise_l1 = F.l1_loss(denoised, clean)
        denoise_mse = F.mse_loss(denoised, clean)
        denoise_loss = denoise_l1 + 0.1 * denoise_mse

        # Neuro-symbolic ordering loss
        ordering_loss, _ = compute_symbolic_ordering_loss(seg_logits, margin=2.0)

        # Total loss
        loss = (seg_loss
                + boundary_weight * boundary_loss
                + denoise_weight * denoise_loss
                + lambda_symbolic_ordering * ordering_loss)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            dice_scores = compute_dice(seg_logits, mask_3class, NUM_CLASSES)
            b_dice = compute_boundary_dice(boundary_logits, is_os_boundary)
            psnr = compute_psnr(denoised, clean)

        # Accumulate metrics
        total_seg_loss += seg_loss.item()
        total_boundary_loss += boundary_loss.item()
        total_denoise_loss += denoise_loss.item()
        total_ordering_loss += ordering_loss.item()
        for c in range(NUM_CLASSES):
            total_dice[c] += dice_scores[c]
        total_boundary_dice += b_dice
        total_psnr += psnr
        n_batches += 1

    return {
        'seg_loss': total_seg_loss / n_batches,
        'boundary_loss': total_boundary_loss / n_batches,
        'denoise_loss': total_denoise_loss / n_batches,
        'ordering_loss': total_ordering_loss / n_batches,
        'dice': [d / n_batches for d in total_dice],
        'boundary_dice': total_boundary_dice / n_batches,
        'psnr': total_psnr / n_batches,
    }


def validate_full_image(model, val_ds, device, patch_size=64, stride=32):
    """Validate on full images with sliding window inference.

    Returns comprehensive metrics including:
    - Global PSNR/SSIM (noisy, backbone/base, denoised)
    - Per-layer PSNR/SSIM for each retinal layer
    - Improvement of our method over base NAFNet
    - Clinical metrics (layer thickness, boundary MAE)
    - Neuro-symbolic ordering verification
    - Adaptive denoising metrics (gate activations, confidence)
    """
    model.eval()
    total_dice = [0] * NUM_CLASSES
    total_boundary_dice = 0

    # Global metrics (full image)
    total_psnr_noisy = 0
    total_psnr_backbone = 0  # Base NAFNet output
    total_psnr_denoised = 0  # Our full method
    total_ssim_noisy = 0
    total_ssim_backbone = 0
    total_ssim_denoised = 0

    # SOTA-comparable metrics (masked to retinal tissue, threshold=100/255)
    total_sota_psnr_noisy = 0
    total_sota_psnr_backbone = 0
    total_sota_psnr_denoised = 0
    total_sota_ssim_noisy = 0
    total_sota_ssim_backbone = 0
    total_sota_ssim_denoised = 0
    sota_count = 0

    # Per-layer metrics: [noisy, backbone, denoised] for each layer
    layer_psnr = {name: {'noisy': 0, 'backbone': 0, 'denoised': 0, 'count': 0}
                  for name in CLASS_NAMES}
    layer_ssim = {name: {'noisy': 0, 'backbone': 0, 'denoised': 0, 'count': 0}
                  for name in CLASS_NAMES}

    # Clinical metrics accumulators
    clinical_thickness_error = {name: 0 for name in CLASS_NAMES}
    clinical_thickness_pred = {name: 0 for name in CLASS_NAMES}
    clinical_thickness_gt = {name: 0 for name in CLASS_NAMES}
    clinical_boundary_mae = 0
    clinical_boundary_count = 0

    # Neuro-symbolic ordering metrics
    ordering_valid_count = 0
    ordering_total_violations = 0
    ordering_gaps = {f'{CLASS_NAMES[i]}->{CLASS_NAMES[i+1]}': 0 for i in range(len(CLASS_NAMES)-1)}

    # Adaptive denoising metrics accumulators
    adaptive_gate_means = {f'gate_{name}': 0 for name in CLASS_NAMES}
    adaptive_confidence = 0
    adaptive_noise_gate_mean = 0
    adaptive_noise_gate_std = 0
    adaptive_count = 0

    n_samples = 0

    with torch.no_grad():
        for i in tqdm(range(len(val_ds)), desc='Validation'):
            sample = val_ds[i]

            noisy = sample['noisy'].unsqueeze(0).to(device)
            clean = sample['clean'].unsqueeze(0).to(device)
            mask_3class = sample['mask_3class'].unsqueeze(0).to(device)
            is_os_boundary = sample['is_os_boundary'].unsqueeze(0).to(device)

            # Get denoised, segmentation, and backbone output
            denoised, seg_logits, boundary_logits, backbone = sliding_window_inference(
                model, noisy, patch_size, stride, device, return_backbone=True
            )

            # Segmentation metrics
            dice_scores = compute_dice(seg_logits, mask_3class, NUM_CLASSES)
            b_dice = compute_boundary_dice(boundary_logits, is_os_boundary)

            # Global denoising metrics
            psnr_noisy = compute_psnr(noisy, clean)
            psnr_backbone = compute_psnr(backbone, clean)
            psnr_denoised = compute_psnr(denoised, clean)
            ssim_noisy = compute_ssim(noisy, clean)
            ssim_backbone = compute_ssim(backbone, clean)
            ssim_denoised = compute_ssim(denoised, clean)

            # SOTA-comparable metrics (masked to retinal tissue region)
            sota_noisy = compute_sota_metrics(noisy, clean)
            sota_backbone = compute_sota_metrics(backbone, clean)
            sota_denoised = compute_sota_metrics(denoised, clean)

            # Per-layer metrics using ground truth mask
            for c, name in enumerate(CLASS_NAMES):
                layer_mask = (mask_3class == c).float().unsqueeze(1)  # [B, 1, H, W]

                # Compute per-layer PSNR
                psnr_n = compute_psnr_masked(noisy, clean, layer_mask > 0.5)
                psnr_b = compute_psnr_masked(backbone, clean, layer_mask > 0.5)
                psnr_d = compute_psnr_masked(denoised, clean, layer_mask > 0.5)

                if psnr_n is not None and psnr_b is not None and psnr_d is not None:
                    layer_psnr[name]['noisy'] += psnr_n
                    layer_psnr[name]['backbone'] += psnr_b
                    layer_psnr[name]['denoised'] += psnr_d
                    layer_psnr[name]['count'] += 1

                # Compute per-layer SSIM
                ssim_n = compute_ssim_masked(noisy, clean, layer_mask)
                ssim_b = compute_ssim_masked(backbone, clean, layer_mask)
                ssim_d = compute_ssim_masked(denoised, clean, layer_mask)

                if ssim_n is not None and ssim_b is not None and ssim_d is not None:
                    layer_ssim[name]['noisy'] += ssim_n
                    layer_ssim[name]['backbone'] += ssim_b
                    layer_ssim[name]['denoised'] += ssim_d
                    layer_ssim[name]['count'] += 1

            # Clinical metrics
            clinical = compute_clinical_metrics(seg_logits, mask_3class, boundary_logits, is_os_boundary)
            for name in CLASS_NAMES:
                clinical_thickness_error[name] += clinical['thickness_error'][name]
                clinical_thickness_pred[name] += clinical['layer_thickness_pred'][name]
                clinical_thickness_gt[name] += clinical['layer_thickness_gt'][name]
            if not np.isnan(clinical['boundary_mae']):
                clinical_boundary_mae += clinical['boundary_mae']
                clinical_boundary_count += 1

            # Neuro-symbolic ordering verification
            ordering = compute_ordering_verification(seg_logits)
            if ordering['is_valid']:
                ordering_valid_count += 1
            ordering_total_violations += ordering['n_violations']
            for gap_name, gap_val in ordering['ordering_gaps'].items():
                ordering_gaps[gap_name] += gap_val

            # Adaptive denoising metrics (run on a single patch to get features)
            # Use center patch for feature extraction
            H, W = noisy.shape[2], noisy.shape[3]
            cy, cx = H // 2, W // 2
            ps = patch_size // 2
            patch = noisy[:, :, max(0,cy-ps):cy+ps, max(0,cx-ps):cx+ps]
            if patch.shape[2] >= patch_size // 2 and patch.shape[3] >= patch_size // 2:
                _, _, _, features = model(patch, return_features=True)
                adaptive = compute_adaptive_denoising_metrics(features)
                if 'layer_gate_means' in adaptive:
                    for gate_name, gate_val in adaptive['layer_gate_means'].items():
                        adaptive_gate_means[gate_name] += gate_val
                if 'seg_confidence' in adaptive:
                    adaptive_confidence += adaptive['seg_confidence']
                if 'noise_gate_mean' in adaptive:
                    adaptive_noise_gate_mean += adaptive['noise_gate_mean']
                    adaptive_noise_gate_std += adaptive['noise_gate_std']
                adaptive_count += 1

            # Accumulate global metrics
            for c in range(NUM_CLASSES):
                total_dice[c] += dice_scores[c]
            total_boundary_dice += b_dice
            total_psnr_noisy += psnr_noisy
            total_psnr_backbone += psnr_backbone
            total_psnr_denoised += psnr_denoised
            total_ssim_noisy += ssim_noisy
            total_ssim_backbone += ssim_backbone
            total_ssim_denoised += ssim_denoised

            # Accumulate SOTA-comparable metrics
            if sota_noisy['sota_psnr'] is not None and sota_denoised['sota_psnr'] is not None:
                total_sota_psnr_noisy += sota_noisy['sota_psnr']
                total_sota_psnr_backbone += sota_backbone['sota_psnr']
                total_sota_psnr_denoised += sota_denoised['sota_psnr']
                if sota_noisy['sota_ssim'] is not None:
                    total_sota_ssim_noisy += sota_noisy['sota_ssim']
                    total_sota_ssim_backbone += sota_backbone['sota_ssim']
                    total_sota_ssim_denoised += sota_denoised['sota_ssim']
                sota_count += 1

            n_samples += 1

            # Free memory after each sample
            del noisy, clean, mask_3class, is_os_boundary
            del denoised, seg_logits, boundary_logits, backbone

    # Clear GPU cache if using CUDA
    if device == 'cuda' or (isinstance(device, str) and 'cuda' in device):
        torch.cuda.empty_cache()

    # Compute averages for per-layer metrics
    layer_psnr_avg = {}
    layer_ssim_avg = {}
    for name in CLASS_NAMES:
        cnt = layer_psnr[name]['count']
        if cnt > 0:
            layer_psnr_avg[name] = {
                'noisy': layer_psnr[name]['noisy'] / cnt,
                'backbone': layer_psnr[name]['backbone'] / cnt,
                'denoised': layer_psnr[name]['denoised'] / cnt,
            }
        cnt = layer_ssim[name]['count']
        if cnt > 0:
            layer_ssim_avg[name] = {
                'noisy': layer_ssim[name]['noisy'] / cnt,
                'backbone': layer_ssim[name]['backbone'] / cnt,
                'denoised': layer_ssim[name]['denoised'] / cnt,
            }

    # Compute adaptive gate variance (measure of differentiation)
    gate_values = [adaptive_gate_means[f'gate_{name}'] / max(1, adaptive_count) for name in CLASS_NAMES]
    gate_std = np.std(gate_values) if adaptive_count > 0 else 0
    gate_range = max(gate_values) - min(gate_values) if adaptive_count > 0 else 0

    # Compute SOTA metric averages
    sota_metrics = {}
    if sota_count > 0:
        sota_metrics = {
            'sota_psnr_noisy': total_sota_psnr_noisy / sota_count,
            'sota_psnr_backbone': total_sota_psnr_backbone / sota_count,
            'sota_psnr_denoised': total_sota_psnr_denoised / sota_count,
            'sota_ssim_noisy': total_sota_ssim_noisy / sota_count,
            'sota_ssim_backbone': total_sota_ssim_backbone / sota_count,
            'sota_ssim_denoised': total_sota_ssim_denoised / sota_count,
            'sota_count': sota_count,
        }

    return {
        'dice': [d / n_samples for d in total_dice],
        'boundary_dice': total_boundary_dice / n_samples,
        # Global metrics
        'psnr_noisy': total_psnr_noisy / n_samples,
        'psnr_backbone': total_psnr_backbone / n_samples,
        'psnr_denoised': total_psnr_denoised / n_samples,
        'psnr_gain_over_noisy': (total_psnr_denoised - total_psnr_noisy) / n_samples,
        'psnr_gain_over_base': (total_psnr_denoised - total_psnr_backbone) / n_samples,
        'ssim_noisy': total_ssim_noisy / n_samples,
        'ssim_backbone': total_ssim_backbone / n_samples,
        'ssim_denoised': total_ssim_denoised / n_samples,
        'ssim_gain_over_noisy': (total_ssim_denoised - total_ssim_noisy) / n_samples,
        'ssim_gain_over_base': (total_ssim_denoised - total_ssim_backbone) / n_samples,
        # SOTA-comparable metrics (masked to retinal tissue, threshold=100/255)
        'sota': sota_metrics,
        # Per-layer metrics
        'layer_psnr': layer_psnr_avg,
        'layer_ssim': layer_ssim_avg,
        # Clinical metrics
        'clinical': {
            'thickness_error': {name: clinical_thickness_error[name] / n_samples for name in CLASS_NAMES},
            'thickness_pred': {name: clinical_thickness_pred[name] / n_samples for name in CLASS_NAMES},
            'thickness_gt': {name: clinical_thickness_gt[name] / n_samples for name in CLASS_NAMES},
            'boundary_mae': clinical_boundary_mae / max(1, clinical_boundary_count),
        },
        # Neuro-symbolic ordering metrics
        'ordering': {
            'valid_rate': ordering_valid_count / n_samples,
            'avg_violations': ordering_total_violations / n_samples,
            'gaps': {k: v / n_samples for k, v in ordering_gaps.items()},
        },
        # Adaptive denoising metrics
        'adaptive': {
            'gate_means': {k: v / max(1, adaptive_count) for k, v in adaptive_gate_means.items()},
            'gate_std': gate_std,
            'gate_range': gate_range,
            'confidence': adaptive_confidence / max(1, adaptive_count),
            'noise_gate_mean': adaptive_noise_gate_mean / max(1, adaptive_count),
            'noise_gate_std': adaptive_noise_gate_std / max(1, adaptive_count),
            'is_adaptive': gate_range > 0.05,  # True if gates differentiate between layers
        },
    }


def print_validation_metrics(val_metrics, prefix=""):
    """Print comprehensive validation metrics including per-layer details."""
    print(f"{prefix}Segmentation:")
    for i, name in enumerate(CLASS_NAMES):
        dice = val_metrics['dice'][i]
        status = "OK" if dice > 0.5 else "LEARNING" if dice > 0.3 else "WEAK"
        print(f"{prefix}  {name:15s}: {dice:.4f} [{status}]")
    b_dice = val_metrics['boundary_dice']
    b_status = "EXCELLENT" if b_dice > 0.5 else "GOOD" if b_dice > 0.4 else "LEARNING"
    print(f"{prefix}  IS_OS Boundary   : {b_dice:.4f} [{b_status}]")

    print(f"{prefix}Global Denoising:")
    print(f"{prefix}  {'':15s}  {'Noisy':>10s}  {'Base(NAF)':>10s}  {'Ours':>10s}  {'Gain/Noisy':>12s}  {'Gain/Base':>10s}")
    print(f"{prefix}  {'PSNR (dB)':15s}  {val_metrics['psnr_noisy']:10.2f}  {val_metrics['psnr_backbone']:10.2f}  {val_metrics['psnr_denoised']:10.2f}  {val_metrics['psnr_gain_over_noisy']:+12.2f}  {val_metrics['psnr_gain_over_base']:+10.2f}")
    print(f"{prefix}  {'SSIM':15s}  {val_metrics['ssim_noisy']:10.4f}  {val_metrics['ssim_backbone']:10.4f}  {val_metrics['ssim_denoised']:10.4f}  {val_metrics['ssim_gain_over_noisy']:+12.4f}  {val_metrics['ssim_gain_over_base']:+10.4f}")

    # SOTA-comparable metrics (masked to retinal tissue, for fair comparison with papers)
    if val_metrics.get('sota') and val_metrics['sota']:
        sota = val_metrics['sota']
        print(f"{prefix}SOTA-Comparable Metrics (masked SSIM/PSNR on retinal tissue, threshold=100):")
        print(f"{prefix}  {'':15s}  {'Noisy':>10s}  {'Base(NAF)':>10s}  {'Ours':>10s}")
        print(f"{prefix}  {'PSNR (dB)':15s}  {sota['sota_psnr_noisy']:10.2f}  {sota['sota_psnr_backbone']:10.2f}  {sota['sota_psnr_denoised']:10.2f}")
        print(f"{prefix}  {'SSIM':15s}  {sota['sota_ssim_noisy']:10.4f}  {sota['sota_ssim_backbone']:10.4f}  {sota['sota_ssim_denoised']:10.4f}")
        print(f"{prefix}  Reference (SNA-SKAN paper Duke17):")
        print(f"{prefix}    Noisy:    PSNR=16.27, SSIM=0.311")
        print(f"{prefix}    SNA-SKAN: PSNR=26.65, SSIM=0.814")
        # Compare our result to SOTA
        ssim_vs_sota = sota['sota_ssim_denoised'] / 0.814 * 100 if sota['sota_ssim_denoised'] else 0
        print(f"{prefix}  Our SSIM vs SNA-SKAN: {ssim_vs_sota:.1f}% of SOTA")

    # Per-layer metrics
    if val_metrics.get('layer_psnr'):
        print(f"{prefix}Per-Layer PSNR (dB):")
        print(f"{prefix}  {'Layer':15s}  {'Noisy':>10s}  {'Base(NAF)':>10s}  {'Ours':>10s}  {'Gain/Noisy':>12s}  {'Gain/Base':>10s}")
        for name in CLASS_NAMES:
            if name in val_metrics['layer_psnr']:
                lp = val_metrics['layer_psnr'][name]
                gain_noisy = lp['denoised'] - lp['noisy']
                gain_base = lp['denoised'] - lp['backbone']
                print(f"{prefix}  {name:15s}  {lp['noisy']:10.2f}  {lp['backbone']:10.2f}  {lp['denoised']:10.2f}  {gain_noisy:+12.2f}  {gain_base:+10.2f}")

    if val_metrics.get('layer_ssim'):
        print(f"{prefix}Per-Layer SSIM:")
        print(f"{prefix}  {'Layer':15s}  {'Noisy':>10s}  {'Base(NAF)':>10s}  {'Ours':>10s}  {'Gain/Noisy':>12s}  {'Gain/Base':>10s}")
        for name in CLASS_NAMES:
            if name in val_metrics['layer_ssim']:
                ls = val_metrics['layer_ssim'][name]
                gain_noisy = ls['denoised'] - ls['noisy']
                gain_base = ls['denoised'] - ls['backbone']
                print(f"{prefix}  {name:15s}  {ls['noisy']:10.4f}  {ls['backbone']:10.4f}  {ls['denoised']:10.4f}  {gain_noisy:+12.4f}  {gain_base:+10.4f}")

    # Clinical metrics
    if val_metrics.get('clinical'):
        clinical = val_metrics['clinical']
        print(f"{prefix}Clinical Metrics:")
        print(f"{prefix}  Layer Thickness (px):")
        print(f"{prefix}    {'Layer':15s}  {'GT':>8s}  {'Pred':>8s}  {'Error':>8s}")
        for name in CLASS_NAMES:
            gt = clinical['thickness_gt'].get(name, 0)
            pred = clinical['thickness_pred'].get(name, 0)
            err = clinical['thickness_error'].get(name, 0)
            status = "OK" if err < 5 else "WARN" if err < 10 else "HIGH"
            print(f"{prefix}    {name:15s}  {gt:8.1f}  {pred:8.1f}  {err:8.1f} [{status}]")
        mae = clinical.get('boundary_mae', float('nan'))
        mae_status = "EXCELLENT" if mae < 2 else "GOOD" if mae < 5 else "NEEDS_WORK"
        print(f"{prefix}  IS_OS Boundary MAE: {mae:.2f} px [{mae_status}]")

    # Neuro-symbolic ordering verification
    if val_metrics.get('ordering'):
        ordering = val_metrics['ordering']
        valid_rate = ordering['valid_rate'] * 100
        valid_status = "PASS" if valid_rate > 95 else "PARTIAL" if valid_rate > 80 else "FAIL"
        print(f"{prefix}Neuro-Symbolic Ordering:")
        print(f"{prefix}  Valid ordering rate: {valid_rate:.1f}% [{valid_status}]")
        print(f"{prefix}  Avg violations/image: {ordering['avg_violations']:.2f}")
        print(f"{prefix}  Layer gaps (should be positive):")
        for gap_name, gap_val in ordering['gaps'].items():
            gap_status = "OK" if gap_val > 10 else "TIGHT" if gap_val > 0 else "VIOLATION"
            print(f"{prefix}    {gap_name}: {gap_val:.1f} px [{gap_status}]")

    # Adaptive denoising metrics
    if val_metrics.get('adaptive'):
        adaptive = val_metrics['adaptive']
        print(f"{prefix}Adaptive Denoising (Layer-Specific Gates):")
        is_adaptive = adaptive.get('is_adaptive', False)
        adapt_status = "ACTIVE" if is_adaptive else "UNIFORM"
        print(f"{prefix}  Adaptation status: {adapt_status}")
        print(f"{prefix}  Gate activations per layer:")
        for name in CLASS_NAMES:
            gate_key = f'gate_{name}'
            if gate_key in adaptive['gate_means']:
                gate_val = adaptive['gate_means'][gate_key]
                print(f"{prefix}    {name:15s}: {gate_val:.4f}")
        print(f"{prefix}  Gate differentiation:")
        print(f"{prefix}    Std across layers : {adaptive['gate_std']:.4f} (higher = more adaptive)")
        print(f"{prefix}    Range (max-min)   : {adaptive['gate_range']:.4f}")
        print(f"{prefix}  Segmentation confidence: {adaptive['confidence']:.4f}")
        print(f"{prefix}  Noise gate: mean={adaptive['noise_gate_mean']:.4f}, std={adaptive['noise_gate_std']:.4f}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Phase 1A-2: Joint Denoising + Segmentation')
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--resume_from', default='best_boundary_model_v4.pth',
                        help='Resume segmenter from Phase 1A-1 checkpoint (default: V4 checkpoint)')
    parser.add_argument('--nafnet_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth',
                        help='NAFNet backbone checkpoint')
    parser.add_argument('--output_dir', default='outputs/phase1a2_joint')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch size for training and validation (64 for memory efficiency)')
    parser.add_argument('--val_stride', type=int, default=32,
                        help='Stride for validation sliding window (typically patch_size/2)')
    parser.add_argument('--noise_levels', default='0.80,0.90,1.00,1.10,1.20',
                        help='Noise scale factors for calibrated noise (0.8-1.2 = 80-120%% of real noise)')
    parser.add_argument('--boundary_weight', type=float, default=2.0)
    parser.add_argument('--pos_weight', type=float, default=10.0)
    parser.add_argument('--denoise_weight', type=float, default=1.0)
    parser.add_argument('--lambda_symbolic_ordering', type=float, default=0.1,
                        help='Weight for neuro-symbolic layer ordering loss')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip_initial_val', action='store_true',
                        help='Skip initial validation (saves time)')
    parser.add_argument('--val_frequency', type=int, default=1,
                        help='Validate every N epochs (default: 1, use 5 for faster training)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    noise_levels = [float(x) for x in args.noise_levels.split(',')]

    print("=" * 70)
    print("PHASE 1A-2: JOINT DENOISING + SEGMENTATION")
    print("(Integrated NAFNet + SimpleBoundarySegmenter)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Noise levels: {noise_levels}")
    print(f"Segmenter checkpoint: {args.resume_from}")
    print(f"NAFNet checkpoint: {args.nafnet_ckpt}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Neuro-Symbolic Ordering: lambda={args.lambda_symbolic_ordering}")
    print("=" * 70)

    # Data - Random patches for training, full images for validation (like Phase 1A-1)
    train_ds = JointTrainDataset(args.train_jsonl, args.max_train, noise_levels, patch_size=args.patch_size)
    # Use noise_level=1.0 for calibrated noise (1.0 = match real dataset noise levels)
    val_ds = JointValDataset(args.val_jsonl, args.max_val, noise_level=1.0)

    # DataLoader for batch training
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"Train: {len(train_ds)} samples ({len(noise_levels)}x noise augmentation)")
    print(f"  -> {len(train_loader)} batches of {args.batch_size}")
    print(f"  -> Training patch size: {args.patch_size}x{args.patch_size}")
    print(f"Val: {len(val_ds)} full images")
    print(f"  -> Validation uses sliding window: patch_size={args.patch_size}, stride={args.val_stride}")

    # Model
    model = JointDenoiserSegmenter(nafnet_ckpt=args.nafnet_ckpt).to(args.device)

    # Load Phase 1A-1 segmenter checkpoint
    if args.resume_from and os.path.exists(args.resume_from):
        model.load_segmenter_checkpoint(args.resume_from, args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    class_weights = torch.tensor([2.0, 4.0, 1.0], device=args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Initial validation (optional)
    if args.skip_initial_val:
        print("\nSkipping initial validation (--skip_initial_val)")
        best_boundary_dice = 0.0
        best_psnr = 0.0
        val_metrics = None
    else:
        print("\nInitial validation...")
        val_metrics = validate_full_image(model, val_ds, args.device, args.patch_size, args.val_stride)
        print_validation_metrics(val_metrics, prefix="  ")
        best_boundary_dice = val_metrics['boundary_dice']
        best_psnr = val_metrics['psnr_denoised']

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, class_weights,
            args.boundary_weight, args.pos_weight, args.denoise_weight,
            args.lambda_symbolic_ordering
        )
        scheduler.step()

        # Validate every N epochs (saves time on CPU)
        run_validation = (epoch % args.val_frequency == 0) or (epoch == args.epochs)
        if run_validation:
            val_metrics = validate_full_image(model, val_ds, args.device, args.patch_size, args.val_stride)

        print(f"\nTraining:")
        print(f"  Seg Loss: {train_metrics['seg_loss']:.4f}, Boundary Loss: {train_metrics['boundary_loss']:.4f}")
        print(f"  Denoise Loss: {train_metrics['denoise_loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB")
        print(f"  Ordering Loss: {train_metrics['ordering_loss']:.4f} [0=valid order, >0=violations]")
        print(f"  Dice: {[f'{d:.4f}' for d in train_metrics['dice']]}")
        print(f"  IS_OS Boundary Dice: {train_metrics['boundary_dice']:.4f}")

        if run_validation:
            print(f"\nValidation:")
            print_validation_metrics(val_metrics, prefix="  ")
        else:
            print(f"\n  (Validation skipped - next at epoch {((epoch // args.val_frequency) + 1) * args.val_frequency})")

        # Save best models (only when validation was run)
        if run_validation and val_metrics is not None:
            if val_metrics['boundary_dice'] > best_boundary_dice:
                best_boundary_dice = val_metrics['boundary_dice']
                torch.save({
                    'state_dict': model.state_dict(),
                    'segmenter_state_dict': model.segmenter.state_dict(),
                    'epoch': epoch,
                    'boundary_dice': best_boundary_dice,
                }, f"{args.output_dir}/best_boundary.pth")
                print(f"  *** NEW BEST IS_OS Boundary: {best_boundary_dice:.4f} ***")

            if val_metrics['psnr_denoised'] > best_psnr:
                best_psnr = val_metrics['psnr_denoised']
                torch.save({
                    'state_dict': model.state_dict(),
                    'segmenter_state_dict': model.segmenter.state_dict(),
                    'epoch': epoch,
                    'psnr': best_psnr,
                }, f"{args.output_dir}/best_psnr.pth")
                print(f"  *** NEW BEST PSNR: {best_psnr:.2f} dB (Gain over noisy: +{val_metrics['psnr_gain_over_noisy']:.2f} dB, over base: +{val_metrics['psnr_gain_over_base']:.2f} dB) ***")

        torch.save({
            'state_dict': model.state_dict(),
            'segmenter_state_dict': model.segmenter.state_dict(),
            'epoch': epoch,
        }, f"{args.output_dir}/latest.pth")

    print(f"\n{'='*70}")
    print(f"PHASE 1A-2 COMPLETE")
    print(f"{'='*70}")
    print(f"Best Results:")
    print(f"  Best IS_OS Boundary Dice: {best_boundary_dice:.4f}")
    print(f"  Best PSNR: {best_psnr:.2f} dB")
    print(f"\nFinal Validation Metrics:")
    print_validation_metrics(val_metrics, prefix="  ")
    print(f"\nModels saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
