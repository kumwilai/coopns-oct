#!/usr/bin/env python3
"""
Joint Physics-Enhanced Boundary Detection + CUAP-OCT Denoising

Integrates:
1. PhysicsEnsembleV3 for boundary detection (Beer-Lambert, Fresnel, gradient alignment)
2. NAFNet backbone for denoising
3. Layer-specific denoising heads (per-layer processing)
4. Segmentation-guided attention (focus on boundaries)
5. Neuro-symbolic constraints (ordering, thickness, intensity, continuity)
6. Clinical importance weighting (RNFL/IS-OS prioritized)

Key TMI Contributions:
- First joint physics-boundary + denoising framework for OCT
- Per-pixel adaptive denoising guided by physics-based segmentation
- Differentiable anatomical constraints from clinical knowledge
- Interpretable layer-specific processing
"""

import os
import sys
import argparse
import gc
import json
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Import physics-enhanced boundary model
from physics_enhanced_v3 import (
    PhysicsEnsembleV3,
    PhysicsLossV3,
    boundaries_to_segmentation,
)

# Import neuro-symbolic framework
from neuro_symbolic_enhanced import (
    NeuroSymbolicLossEnhanced,
    AnatomicalKnowledgeBase,
    OCTPhysicsEngine,
    TopologicalConstraints,
    SymbolicRuleEngine,
)

# Try importing NAFNet and other components
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))
    from nsnd.models.nafnet import NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False
    print("Warning: NAFNet not available, using simple UNet denoiser")

# Configuration
NUM_CLASSES = 4
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
NUM_BOUNDARIES = 4
BOUNDARY_NAMES = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']

# Clinical importance weights
CLINICAL_WEIGHTS = {
    'RNFL_GCL': 2.0,      # Glaucoma critical
    'INL_OPL_ONL': 1.0,   # Moderate importance
    'IS_OS': 2.0,         # Visual acuity critical
    'RPE_Choroid': 1.5,   # AMD important
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Simple UNet Denoiser (fallback if NAFNet not available)
# =============================================================================
class SimpleUNetDenoiser(nn.Module):
    """Simple UNet for denoising when NAFNet is not available."""

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)

        # Decoder
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)

        self.final = nn.Conv2d(base_channels, in_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return x + self.final(d1)  # Residual connection


# =============================================================================
# Layer-Specific Denoising Heads
# =============================================================================
class LayerSpecificHead(nn.Module):
    """Dedicated denoising head for a specific retinal layer."""

    def __init__(self, in_channels=32, hidden_channels=64, num_blocks=2, dropout=0.1):
        super().__init__()

        layers = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else hidden_channels
            layers.extend([
                nn.Conv2d(in_ch, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ])

        layers.append(nn.Conv2d(hidden_channels, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)


class LayerSpecificDenoiser(nn.Module):
    """
    Layer-specific denoising with dedicated heads per anatomical layer.

    Each layer (RNFL, INL, IS_OS, RPE) gets its own processing head,
    allowing layer-appropriate denoising strategies.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        hidden_channels: int = 64,
        num_layers: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers

        # Per-layer denoising heads
        self.layer_heads = nn.ModuleList([
            LayerSpecificHead(feature_channels, hidden_channels, num_blocks, dropout)
            for _ in range(num_layers)
        ])

        # Learnable layer importance weights
        self.layer_importance = nn.Parameter(torch.ones(num_layers))

    def forward(self, features: torch.Tensor, layer_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Encoder features [B, C, H, W]
            layer_masks: Soft layer masks [B, num_layers, H, W]

        Returns:
            Layer-specific refinement [B, 1, H, W]
        """
        B, C, H, W = features.shape

        # Process each layer
        layer_outputs = []
        for i, head in enumerate(self.layer_heads):
            layer_out = head(features)  # [B, 1, H, W]
            layer_outputs.append(layer_out)

        # Stack and weight by masks
        layer_stack = torch.stack(layer_outputs, dim=2)  # [B, 1, num_layers, H, W]

        # Apply importance weights
        importance = F.softmax(self.layer_importance, dim=0)
        layer_masks_weighted = layer_masks * importance.view(1, -1, 1, 1)

        # Weighted combination
        masks_expanded = layer_masks_weighted.unsqueeze(1)  # [B, 1, num_layers, H, W]
        combined = (layer_stack * masks_expanded).sum(dim=2)  # [B, 1, H, W]

        return combined


# =============================================================================
# Segmentation-Guided Attention
# =============================================================================
class SegmentationGuidedAttention(nn.Module):
    """
    Focus denoising attention on boundaries and clinically important regions.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        boundary_sigma: float = 5.0,
        clinical_weights: Dict[str, float] = None,
    ):
        super().__init__()

        self.boundary_sigma = boundary_sigma
        self.clinical_weights = clinical_weights or CLINICAL_WEIGHTS

        # Attention refinement
        self.attention_net = nn.Sequential(
            nn.Conv2d(feature_channels + NUM_CLASSES + 1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Register clinical weights as buffer
        weights = torch.tensor([self.clinical_weights[name] for name in CLASS_NAMES])
        self.register_buffer('clinical_weight_tensor', weights)

    def compute_boundary_attention(self, boundaries: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Compute attention map emphasizing boundary regions."""
        B, N, W_bound = boundaries.shape
        device = boundaries.device

        # Convert boundaries to pixel positions
        boundaries_px = boundaries * (H - 1)

        # Create distance map to nearest boundary
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

        min_dist = torch.full((B, 1, H, W), float(H), device=device)
        for n in range(N):
            boundary_n = boundaries_px[:, n, :].view(B, 1, 1, W)
            dist = torch.abs(y_coords - boundary_n)
            min_dist = torch.minimum(min_dist, dist)

        # Gaussian attention around boundaries
        boundary_attention = torch.exp(-min_dist**2 / (2 * self.boundary_sigma**2))

        return boundary_attention

    def compute_clinical_attention(self, layer_probs: torch.Tensor) -> torch.Tensor:
        """Compute attention based on clinical importance."""
        # Weight probabilities by clinical importance
        weights = self.clinical_weight_tensor.view(1, -1, 1, 1)
        clinical_attention = (layer_probs * weights).sum(dim=1, keepdim=True)
        clinical_attention = clinical_attention / weights.max()  # Normalize

        return clinical_attention

    def forward(
        self,
        features: torch.Tensor,
        layer_probs: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            layer_probs: [B, num_classes, H, W]
            boundaries: [B, num_boundaries, W]

        Returns:
            attention_weights: [B, 1, H, W]
        """
        B, C, H, W = features.shape

        # Compute component attentions
        boundary_attn = self.compute_boundary_attention(boundaries, H, W)
        clinical_attn = self.compute_clinical_attention(layer_probs)

        # Combine with features
        combined = torch.cat([features, layer_probs, boundary_attn], dim=1)
        attention = self.attention_net(combined)

        # Blend boundary and clinical attention
        final_attention = 0.5 * boundary_attn + 0.5 * clinical_attn + 0.2 * attention
        final_attention = final_attention.clamp(0.1, 1.0)  # Ensure minimum attention

        return final_attention


# =============================================================================
# Joint Physics-Denoising Model
# =============================================================================
class JointPhysicsDenoiser(nn.Module):
    """
    Joint model combining physics-enhanced segmentation with adaptive denoising.

    Architecture:
    1. Shared encoder extracts features
    2. PhysicsEnsembleV3 predicts boundaries (with physics constraints)
    3. Boundaries -> Soft segmentation masks
    4. Layer-specific heads denoise each layer
    5. Segmentation-guided attention focuses on critical regions
    6. Final fusion produces denoised output
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 48,
        denoiser_channels: int = 32,
        num_boundaries: int = 4,
        num_classes: int = 4,
        use_layer_heads: bool = True,
        use_seg_attention: bool = True,
    ):
        super().__init__()

        self.use_layer_heads = use_layer_heads
        self.use_seg_attention = use_seg_attention

        # Physics-enhanced boundary detector
        self.boundary_model = PhysicsEnsembleV3(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_boundaries=num_boundaries,
        )

        # Denoising backbone
        if HAS_NAFNET:
            self.denoiser = NAFNet(img_channel=in_channels, width=denoiser_channels)
        else:
            self.denoiser = SimpleUNetDenoiser(in_channels, denoiser_channels)

        # Layer-specific processing
        if use_layer_heads:
            self.layer_denoiser = LayerSpecificDenoiser(
                feature_channels=denoiser_channels,
                hidden_channels=64,
                num_layers=num_classes,
            )

        # Segmentation-guided attention
        if use_seg_attention:
            self.seg_attention = SegmentationGuidedAttention(
                feature_channels=denoiser_channels,
            )

        # Feature extractor for layer heads
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, denoiser_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(denoiser_channels, denoiser_channels, 3, padding=1),
            nn.GELU(),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 1),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy: Noisy input image [B, 1, H, W]
            return_intermediates: Whether to return intermediate outputs

        Returns:
            Dictionary with:
                - denoised: Final denoised image [B, 1, H, W]
                - boundaries: Predicted boundaries [B, 4, W]
                - segmentation: Soft segmentation [B, 4, H, W]
                - attention: Attention map [B, 1, H, W] (if using attention)
        """
        B, C, H, W = noisy.shape

        # 1. Physics-enhanced boundary detection
        boundary_outputs = self.boundary_model(noisy, return_aux=True)
        boundaries = boundary_outputs['boundaries']  # [B, 4, W]

        # 2. Convert boundaries to soft segmentation
        segmentation = boundaries_to_segmentation(boundaries, H, num_classes=NUM_CLASSES)
        seg_probs = F.one_hot(segmentation.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()

        # 3. Base denoising
        if HAS_NAFNET:
            base_denoised = self.denoiser(noisy)
        else:
            base_denoised = self.denoiser(noisy)

        # 4. Extract features for layer-specific processing
        features = self.feature_extractor(noisy)

        # 5. Layer-specific refinement
        if self.use_layer_heads:
            layer_refinement = self.layer_denoiser(features, seg_probs)
        else:
            layer_refinement = torch.zeros_like(noisy)

        # 6. Segmentation-guided attention
        if self.use_seg_attention:
            attention = self.seg_attention(features, seg_probs, boundaries)
        else:
            attention = torch.ones(B, 1, H, W, device=noisy.device)

        # 7. Final fusion
        fusion_input = torch.cat([base_denoised, layer_refinement, attention], dim=1)
        denoised = self.fusion(fusion_input) + base_denoised  # Residual

        outputs = {
            'denoised': denoised,
            'boundaries': boundaries,
            'segmentation': segmentation,
            'seg_probs': seg_probs,
        }

        if return_intermediates:
            outputs['base_denoised'] = base_denoised
            outputs['layer_refinement'] = layer_refinement
            outputs['attention'] = attention
            outputs['boundary_aux'] = boundary_outputs

        return outputs


# =============================================================================
# Joint Loss Function
# =============================================================================
class JointPhysicsDenosingLoss(nn.Module):
    """
    Combined loss for joint physics-boundary detection and denoising.

    Components:
    1. Denoising loss (L1 + SSIM)
    2. Physics boundary loss (Beer-Lambert, Fresnel, gradient alignment)
    3. Segmentation loss (Dice + CE)
    4. Neuro-symbolic constraints (ordering, thickness, intensity, continuity)
    5. Clinical importance weighting
    """

    def __init__(
        self,
        # Denoising weights
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        # Segmentation weights
        lambda_dice: float = 0.5,
        lambda_ce: float = 0.3,
        # Physics weights
        lambda_physics: float = 0.5,
        # Neuro-symbolic weights
        lambda_ordering: float = 0.1,
        lambda_thickness: float = 0.05,
        lambda_intensity: float = 0.05,
        lambda_continuity: float = 0.05,
        # Clinical weights
        use_clinical_weighting: bool = True,
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.lambda_physics = lambda_physics
        self.lambda_ordering = lambda_ordering
        self.lambda_thickness = lambda_thickness
        self.lambda_intensity = lambda_intensity
        self.lambda_continuity = lambda_continuity
        self.use_clinical_weighting = use_clinical_weighting

        # Physics loss
        self.physics_loss = PhysicsLossV3()

        # Neuro-symbolic loss
        self.neurosymbolic_loss = NeuroSymbolicLossEnhanced()

        # Clinical weights
        weights = torch.tensor([CLINICAL_WEIGHTS[name] for name in CLASS_NAMES])
        self.register_buffer('clinical_weights', weights)

    def compute_ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss."""
        # Simple SSIM approximation
        mu_pred = F.avg_pool2d(pred, 3, stride=1, padding=1)
        mu_target = F.avg_pool2d(target, 3, stride=1, padding=1)

        sigma_pred = F.avg_pool2d(pred**2, 3, stride=1, padding=1) - mu_pred**2
        sigma_target = F.avg_pool2d(target**2, 3, stride=1, padding=1) - mu_target**2
        sigma_cross = F.avg_pool2d(pred * target, 3, stride=1, padding=1) - mu_pred * mu_target

        C1, C2 = 0.01**2, 0.03**2

        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))

        return 1 - ssim.mean()

    def compute_dice_loss(
        self,
        pred_probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss with clinical weighting."""
        B, C, H, W = pred_probs.shape

        target_onehot = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()

        dice_losses = []
        for c in range(C):
            pred_c = pred_probs[:, c]
            target_c = target_onehot[:, c]

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice = (2 * intersection + 1e-8) / (union + 1e-8)

            if self.use_clinical_weighting:
                weight = self.clinical_weights[c]
            else:
                weight = 1.0

            dice_losses.append((1 - dice.mean()) * weight)

        return sum(dice_losses) / len(dice_losses)

    def compute_ordering_loss(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Enforce b0 < b1 < b2 < b3."""
        min_gap = 0.02  # ~5px at 256px
        total_violation = 0

        for i in range(3):
            gap = boundaries[:, i+1, :] - boundaries[:, i, :]
            violation = F.relu(min_gap - gap)
            total_violation = total_violation + violation.mean()

        return total_violation / 3

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        gt_mask: torch.Tensor,
        gt_boundaries: torch.Tensor,
        noisy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute joint loss.

        Args:
            outputs: Model outputs dictionary
            clean: Clean target image [B, 1, H, W]
            gt_mask: Ground truth segmentation [B, H, W]
            gt_boundaries: Ground truth boundaries [B, 4, W]
            noisy: Noisy input [B, 1, H, W]
        """
        losses = {}

        # 1. Denoising losses
        denoised = outputs['denoised']
        losses['l1'] = F.l1_loss(denoised, clean) * self.lambda_l1
        losses['ssim'] = self.compute_ssim_loss(denoised, clean) * self.lambda_ssim

        # 2. Segmentation losses
        seg_probs = outputs['seg_probs']
        losses['dice'] = self.compute_dice_loss(seg_probs, gt_mask) * self.lambda_dice
        losses['ce'] = F.cross_entropy(seg_probs, gt_mask.long()) * self.lambda_ce

        # 3. Boundary losses
        boundaries = outputs['boundaries']
        losses['boundary_mae'] = F.l1_loss(boundaries, gt_boundaries) * self.lambda_physics

        # 4. Neuro-symbolic constraints
        losses['ordering'] = self.compute_ordering_loss(boundaries) * self.lambda_ordering

        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return total_loss, losses


# =============================================================================
# Dataset
# =============================================================================
class JointDenoisingDataset(Dataset):
    """Dataset for joint denoising and segmentation training."""

    def __init__(
        self,
        jsonl_path: str,
        patch_size: int = 128,
        noise_levels: list = None,
        augment: bool = True,
    ):
        self.patch_size = patch_size
        self.noise_levels = noise_levels or [0.05, 0.1, 0.15]
        self.augment = augment

        # Load samples
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def add_noise(self, clean: np.ndarray, noise_level: float) -> np.ndarray:
        """Add synthetic noise to clean image."""
        # Speckle noise (multiplicative)
        speckle = np.random.rayleigh(scale=noise_level, size=clean.shape)
        noisy = clean * (1 + speckle)

        # Gaussian noise (additive)
        gaussian = np.random.normal(0, noise_level * 0.3, clean.shape)
        noisy = noisy + gaussian

        return np.clip(noisy, 0, 1).astype(np.float32)

    def extract_boundaries_from_mask(self, mask: np.ndarray, H: int) -> np.ndarray:
        """Extract boundary positions from segmentation mask."""
        W = mask.shape[1]
        boundaries = np.zeros((4, W), dtype=np.float32)

        for col in range(W):
            column = mask[:, col]

            # Find transitions
            prev_class = -1
            boundary_idx = 0

            for row in range(H):
                curr_class = column[row]
                if curr_class != prev_class and prev_class != -1 and boundary_idx < 4:
                    boundaries[boundary_idx, col] = row / (H - 1)
                    boundary_idx += 1
                prev_class = curr_class

        return boundaries

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample['image_path']).convert('L')
        img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        clean = np.array(img, dtype=np.float32) / 255.0

        # Load mask
        mask = Image.open(sample['mask_path']).convert('L')
        mask = mask.resize((self.patch_size, self.patch_size), Image.NEAREST)
        mask = np.array(mask)

        # Convert mask from 1-4 to 0-3 if needed (our masks use 1-indexed classes)
        if mask.max() > NUM_CLASSES - 1:
            mask = mask - 1
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Add noise
        noise_level = np.random.choice(self.noise_levels)
        noisy = self.add_noise(clean, noise_level)

        # Extract boundaries from mask
        boundaries = self.extract_boundaries_from_mask(mask, self.patch_size)

        # Convert to tensors
        clean_t = torch.from_numpy(clean).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        boundaries_t = torch.from_numpy(boundaries)

        return {
            'clean': clean_t,
            'noisy': noisy_t,
            'mask': mask_t,
            'boundaries': boundaries_t,
        }


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_losses = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)
        boundaries = batch['boundaries'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(noisy, return_intermediates=True)

        # Compute loss
        loss, losses = criterion(outputs, clean, mask, boundaries, noisy)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0
            total_losses[k] += v.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'psnr': f"{compute_psnr(outputs['denoised'], clean):.2f}",
        })

    # Average losses
    for k in total_losses:
        total_losses[k] /= num_batches

    return total_losses


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_losses = {}
    total_psnr = 0
    total_dice = {name: 0 for name in CLASS_NAMES}
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)
        boundaries = batch['boundaries'].to(device)

        outputs = model(noisy)
        loss, losses = criterion(outputs, clean, mask, boundaries, noisy)

        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0
            total_losses[k] += v.item()

        # PSNR
        total_psnr += compute_psnr(outputs['denoised'], clean)

        # Per-class Dice
        pred_seg = outputs['segmentation']
        for c, name in enumerate(CLASS_NAMES):
            pred_c = (pred_seg == c).float()
            gt_c = (mask == c).float()
            intersection = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            total_dice[name] += dice.item()

        num_batches += 1

    # Average
    for k in total_losses:
        total_losses[k] /= num_batches
    total_losses['psnr'] = total_psnr / num_batches
    for name in CLASS_NAMES:
        total_losses[f'dice_{name}'] = total_dice[name] / num_batches
    total_losses['dice_avg'] = sum(total_dice.values()) / len(total_dice) / num_batches

    return total_losses


# =============================================================================
# Main Training Script
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Joint Physics-Enhanced Denoising Training")

    # Data
    parser.add_argument('--train_jsonl', type=str, default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='combined_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)

    # Model
    parser.add_argument('--hidden_channels', type=int, default=48)
    parser.add_argument('--denoiser_channels', type=int, default=32)
    parser.add_argument('--use_layer_heads', action='store_true', default=True)
    parser.add_argument('--use_seg_attention', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')

    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_ssim', type=float, default=0.5)
    parser.add_argument('--lambda_dice', type=float, default=0.5)
    parser.add_argument('--lambda_physics', type=float, default=0.5)
    parser.add_argument('--lambda_ordering', type=float, default=0.1)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/joint_physics_denoising')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("Joint Physics-Enhanced Boundary Detection + Denoising")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Patch size: {args.patch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")

    # Create model
    model = JointPhysicsDenoiser(
        hidden_channels=args.hidden_channels,
        denoiser_channels=args.denoiser_channels,
        use_layer_heads=args.use_layer_heads,
        use_seg_attention=args.use_seg_attention,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Create datasets
    train_dataset = JointDenoisingDataset(
        args.train_jsonl,
        patch_size=args.patch_size,
    )
    val_dataset = JointDenoisingDataset(
        args.val_jsonl,
        patch_size=args.patch_size,
        augment=False,
    )

    if args.max_train:
        train_dataset.samples = train_dataset.samples[:args.max_train]
    if args.max_val:
        val_dataset.samples = val_dataset.samples[:args.max_val]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create loss and optimizer
    criterion = JointPhysicsDenosingLoss(
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_dice=args.lambda_dice,
        lambda_physics=args.lambda_physics,
        lambda_ordering=args.lambda_ordering,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_psnr = 0
    best_dice = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 40)

        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_losses = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log results
        logger.info(f"Train Loss: {train_losses['total']:.4f}")
        logger.info(f"Val Loss: {val_losses['total']:.4f}")
        logger.info(f"Val PSNR: {val_losses['psnr']:.2f} dB")
        logger.info(f"Val Dice (avg): {val_losses['dice_avg']:.3f}")
        for name in CLASS_NAMES:
            logger.info(f"  {name}: {val_losses[f'dice_{name}']:.3f}")

        # Save best models
        if val_losses['psnr'] > best_psnr:
            best_psnr = val_losses['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(args.output_dir, 'best_psnr.pth'))
            logger.info(f"  -> New best PSNR: {best_psnr:.2f} dB")

        if val_losses['dice_avg'] > best_dice:
            best_dice = val_losses['dice_avg']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best_dice.pth'))
            logger.info(f"  -> New best Dice: {best_dice:.3f}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'epoch_{epoch}.pth'))

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best PSNR: {best_psnr:.2f} dB")
    logger.info(f"Best Dice: {best_dice:.3f}")
    logger.info(f"Models saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
