#!/usr/bin/env python3
"""
Joint Physics-Enhanced Boundary Detection + CUAP-OCT Denoising V2

Improvements over V1:
1. Uses pretrained NAFNet checkpoint (outputs/nafnet_calibrated)
2. Soft masks with gradual boundary transitions (~5-10px blending)
3. Multi-frame prediction averaging for stable segmentation
4. Better PKU37 compatibility

Key Components:
- PhysicsEnsembleV3 for boundary detection (Beer-Lambert, Fresnel)
- Pretrained NAFNet backbone for denoising
- Layer-specific denoising with soft blending
- Segmentation-guided attention
- Neuro-symbolic constraints
"""

import os
import sys
import argparse
import gc
import json
import logging
from typing import Dict, Optional, Tuple, List

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

# Try importing NAFNet
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))
    from nsnd.models.nafnet import NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False
    print("Warning: NAFNet not available")

# Configuration
NUM_CLASSES = 4
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
NUM_BOUNDARIES = 4
BOUNDARY_NAMES = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']

# Default NAFNet checkpoint
DEFAULT_NAFNET_CKPT = "outputs/nafnet_calibrated/nafnet_best.pth"

# Clinical importance weights
CLINICAL_WEIGHTS = {
    'RNFL_GCL': 2.0,
    'INL_OPL_ONL': 1.0,
    'IS_OS': 2.0,
    'RPE_Choroid': 1.5,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Soft Mask Generation with Boundary Blending
# =============================================================================
def create_soft_masks_from_boundaries(
    boundaries: torch.Tensor,
    H: int,
    W: int,
    blend_sigma: float = 7.0,
) -> torch.Tensor:
    """
    Create soft segmentation masks with gradual transitions at boundaries.

    This is critical for PKU37 compatibility - hard masks cause artifacts
    at boundaries where segmentation is uncertain.

    Args:
        boundaries: [B, 4, W] boundary positions (normalized 0-1)
        H: Image height
        W: Image width
        blend_sigma: Blending zone width in pixels (~5-10px recommended)

    Returns:
        soft_masks: [B, 4, H, W] soft probability masks
    """
    B, N, W_bound = boundaries.shape
    device = boundaries.device

    # Convert to pixel positions
    boundaries_px = boundaries * (H - 1)  # [B, 4, W]

    # Create y-coordinate grid
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)  # [1, H, 1]

    # Initialize soft masks
    soft_masks = torch.zeros(B, NUM_CLASSES, H, W, device=device)

    for b in range(B):
        for col in range(W):
            # Get boundary positions for this column
            b0 = boundaries_px[b, 0, col]  # ILM
            b1 = boundaries_px[b, 1, col]  # RNFL_INL
            b2 = boundaries_px[b, 2, col]  # INL_ISOS
            b3 = boundaries_px[b, 3, col]  # ISOS_RPE

            y = torch.arange(H, device=device, dtype=torch.float32)

            # Soft transitions using sigmoid with temperature
            temp = blend_sigma

            # Class 0: RNFL_GCL (above b1)
            soft_masks[b, 0, :, col] = torch.sigmoid((b1 - y) / temp)

            # Class 1: INL_OPL_ONL (between b1 and b2)
            soft_masks[b, 1, :, col] = torch.sigmoid((y - b1) / temp) * torch.sigmoid((b2 - y) / temp)

            # Class 2: IS_OS (between b2 and b3)
            soft_masks[b, 2, :, col] = torch.sigmoid((y - b2) / temp) * torch.sigmoid((b3 - y) / temp)

            # Class 3: RPE_Choroid (below b3)
            soft_masks[b, 3, :, col] = torch.sigmoid((y - b3) / temp)

    # Normalize to sum to 1
    soft_masks = soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

    return soft_masks


def create_soft_masks_vectorized(
    boundaries: torch.Tensor,
    H: int,
    blend_sigma: float = 7.0,
) -> torch.Tensor:
    """
    Vectorized version of soft mask creation (faster).

    Args:
        boundaries: [B, 4, W] boundary positions (normalized 0-1)
        H: Image height
        blend_sigma: Blending zone width in pixels

    Returns:
        soft_masks: [B, 4, H, W] soft probability masks
    """
    B, N, W = boundaries.shape
    device = boundaries.device

    # Convert to pixel positions [B, 4, W]
    boundaries_px = boundaries * (H - 1)

    # Create y-coordinate grid [1, 1, H, 1] for proper broadcasting
    y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

    # Expand boundaries for broadcasting [B, 4, 1, W]
    b_exp = boundaries_px.unsqueeze(2)  # [B, 4, 1, W]

    # Temperature for soft transitions
    temp = blend_sigma

    # Extract individual boundaries [B, 1, 1, W]
    b1 = b_exp[:, 1:2, :, :]  # RNFL_INL boundary
    b2 = b_exp[:, 2:3, :, :]  # INL_ISOS boundary
    b3 = b_exp[:, 3:4, :, :]  # ISOS_RPE boundary

    # Create soft masks using sigmoid transitions
    # Broadcasting: y [1,1,H,1] with b [B,1,1,W] -> [B,1,H,W]

    # Class 0: RNFL_GCL (y < b1)
    mask_0 = torch.sigmoid((b1 - y) / temp)  # [B, 1, H, W]

    # Class 1: INL_OPL_ONL (b1 < y < b2)
    above_b1 = torch.sigmoid((y - b1) / temp)
    below_b2 = torch.sigmoid((b2 - y) / temp)
    mask_1 = above_b1 * below_b2

    # Class 2: IS_OS (b2 < y < b3)
    above_b2 = torch.sigmoid((y - b2) / temp)
    below_b3 = torch.sigmoid((b3 - y) / temp)
    mask_2 = above_b2 * below_b3

    # Class 3: RPE_Choroid (y > b3)
    mask_3 = torch.sigmoid((y - b3) / temp)

    # Stack masks [B, 4, H, W]
    soft_masks = torch.cat([mask_0, mask_1, mask_2, mask_3], dim=1)

    # Normalize to sum to 1
    soft_masks = soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

    return soft_masks


# =============================================================================
# Layer-Specific Denoising with Soft Blending
# =============================================================================
class SoftLayerDenoiser(nn.Module):
    """
    Layer-specific denoising with soft mask blending.

    Key for PKU37: Uses broad regions instead of pixel-precise edges.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        num_layers: int = 4,
        blend_sigma: float = 7.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.blend_sigma = blend_sigma

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.GELU(),
        )

        # Per-layer refinement heads
        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(feature_channels, 1, 1),
            )
            for _ in range(num_layers)
        ])

        # Learnable layer-specific denoising strength
        self.layer_strength = nn.Parameter(torch.ones(num_layers) * 0.5)

    def forward(
        self,
        x: torch.Tensor,
        soft_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 1, H, W]
            soft_masks: Soft layer masks [B, num_layers, H, W]

        Returns:
            Layer-specific refinement [B, 1, H, W]
        """
        features = self.feature_net(x)

        # Process each layer
        layer_outputs = []
        for i, head in enumerate(self.layer_heads):
            layer_out = head(features)
            strength = torch.sigmoid(self.layer_strength[i])
            layer_outputs.append(layer_out * strength)

        # Stack outputs [B, num_layers, H, W]
        layer_stack = torch.cat(layer_outputs, dim=1)

        # Soft blend using masks (key for smooth transitions)
        blended = (layer_stack * soft_masks).sum(dim=1, keepdim=True)

        return blended


# =============================================================================
# Multi-Frame Prediction Averaging
# =============================================================================
class MultiFramePredictor(nn.Module):
    """
    Wrapper that averages predictions across multiple noisy frames
    for more stable segmentation on PKU37.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    @torch.no_grad()
    def predict_averaged(
        self,
        frames: List[torch.Tensor],
        return_std: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Average predictions across multiple noisy frames.

        Args:
            frames: List of noisy frames [B, 1, H, W]
            return_std: Whether to return standard deviation (uncertainty)

        Returns:
            Averaged outputs
        """
        all_boundaries = []
        all_denoised = []

        for frame in frames:
            outputs = self.base_model(frame)
            all_boundaries.append(outputs['boundaries'])
            all_denoised.append(outputs['denoised'])

        # Average
        avg_boundaries = torch.stack(all_boundaries).mean(dim=0)
        avg_denoised = torch.stack(all_denoised).mean(dim=0)

        result = {
            'boundaries': avg_boundaries,
            'denoised': avg_denoised,
        }

        if return_std:
            result['boundaries_std'] = torch.stack(all_boundaries).std(dim=0)
            result['denoised_std'] = torch.stack(all_denoised).std(dim=0)

        return result


# =============================================================================
# Joint Model with Pretrained NAFNet
# =============================================================================
class JointPhysicsDenoiserV2(nn.Module):
    """
    Joint model V2 with:
    1. Pretrained NAFNet backbone
    2. Soft mask blending at boundaries
    3. Physics-enhanced boundary detection
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 48,
        nafnet_width: int = 32,
        nafnet_ckpt: str = None,
        physics_ckpt: str = None,
        num_boundaries: int = 4,
        num_classes: int = 4,
        blend_sigma: float = 7.0,
        freeze_nafnet: bool = False,
        freeze_boundary: bool = False,
    ):
        super().__init__()

        self.blend_sigma = blend_sigma
        self.num_classes = num_classes

        # Physics-enhanced boundary detector
        self.boundary_model = PhysicsEnsembleV3(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_boundaries=num_boundaries,
        )

        # Load pretrained physics/segmentation weights (trained at 256x256)
        if physics_ckpt and os.path.exists(physics_ckpt):
            logger.info(f"Loading physics/segmentation checkpoint: {physics_ckpt}")
            ckpt = torch.load(physics_ckpt, map_location='cpu', weights_only=False)
            if 'model_state_dict' in ckpt:
                self.boundary_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            else:
                self.boundary_model.load_state_dict(ckpt, strict=False)
            logger.info(f"Physics model loaded (epoch {ckpt.get('epoch', '?')}, dice={ckpt.get('best_dice', '?'):.3f})")

        # Optionally freeze boundary model
        if freeze_boundary:
            for param in self.boundary_model.parameters():
                param.requires_grad = False
            logger.info("Boundary model frozen")

        # NAFNet denoiser backbone
        # Architecture must match checkpoint: width=64, enc/dec=[2,2,2], middle=2
        if HAS_NAFNET:
            self.denoiser = NAFNet(
                img_channel=in_channels,
                width=nafnet_width,
                middle_blk_num=2,
                enc_blk_nums=[2, 2, 2],
                dec_blk_nums=[2, 2, 2],
            )

            # Load pretrained weights
            if nafnet_ckpt and os.path.exists(nafnet_ckpt):
                logger.info(f"Loading NAFNet checkpoint: {nafnet_ckpt}")
                ckpt = torch.load(nafnet_ckpt, map_location='cpu', weights_only=False)
                if 'model_state_dict' in ckpt:
                    self.denoiser.load_state_dict(ckpt['model_state_dict'], strict=False)
                elif 'state_dict' in ckpt:
                    self.denoiser.load_state_dict(ckpt['state_dict'], strict=False)
                else:
                    self.denoiser.load_state_dict(ckpt, strict=False)
                logger.info("NAFNet weights loaded successfully")

            # Optionally freeze NAFNet
            if freeze_nafnet:
                for param in self.denoiser.parameters():
                    param.requires_grad = False
                logger.info("NAFNet backbone frozen")
        else:
            # Fallback to simple UNet
            self.denoiser = self._create_simple_denoiser(in_channels, nafnet_width)

        # Layer-specific denoising with soft blending
        self.layer_denoiser = SoftLayerDenoiser(
            in_channels=in_channels,
            feature_channels=nafnet_width,
            num_layers=num_classes,
            blend_sigma=blend_sigma,
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 1),
        )

        # Clinical importance for attention
        weights = torch.tensor([CLINICAL_WEIGHTS[name] for name in CLASS_NAMES])
        self.register_buffer('clinical_weights', weights)

    def _create_simple_denoiser(self, in_ch, width):
        """Simple fallback denoiser."""
        return nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, in_ch, 3, padding=1),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with soft mask blending.
        """
        B, C, H, W = noisy.shape

        # 1. Physics-enhanced boundary detection
        boundary_outputs = self.boundary_model(noisy, return_aux=True)
        boundaries = boundary_outputs['boundaries']  # [B, 4, W]

        # 2. Create SOFT masks with gradual transitions
        soft_masks = create_soft_masks_vectorized(
            boundaries, H, blend_sigma=self.blend_sigma
        )

        # 3. Base denoising with NAFNet
        if HAS_NAFNET:
            base_denoised = self.denoiser(noisy)
        else:
            base_denoised = noisy + self.denoiser(noisy)

        # 4. Layer-specific refinement with soft blending
        layer_refinement = self.layer_denoiser(noisy, soft_masks)

        # 5. Final fusion
        fusion_input = torch.cat([base_denoised, layer_refinement], dim=1)
        denoised = base_denoised + self.fusion(fusion_input)

        # Hard segmentation for metrics (argmax of soft masks)
        segmentation = soft_masks.argmax(dim=1)

        outputs = {
            'denoised': denoised,
            'boundaries': boundaries,
            'segmentation': segmentation,
            'soft_masks': soft_masks,
        }

        if return_intermediates:
            outputs['base_denoised'] = base_denoised
            outputs['layer_refinement'] = layer_refinement
            outputs['boundary_aux'] = boundary_outputs

        return outputs


# =============================================================================
# Loss Function
# =============================================================================
class JointLossV2(nn.Module):
    """Loss function with soft mask support."""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_dice: float = 0.5,
        lambda_boundary: float = 0.3,
        lambda_ordering: float = 0.1,
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary
        self.lambda_ordering = lambda_ordering

        weights = torch.tensor([CLINICAL_WEIGHTS[name] for name in CLASS_NAMES])
        self.register_buffer('clinical_weights', weights)

    def soft_dice_loss(self, soft_masks: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss using soft masks."""
        B, C, H, W = soft_masks.shape

        # Convert target to one-hot
        target_onehot = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()

        dice_losses = []
        for c in range(C):
            pred = soft_masks[:, c]
            gt = target_onehot[:, c]

            intersection = (pred * gt).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))

            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            weight = self.clinical_weights[c]
            dice_losses.append((1 - dice.mean()) * weight)

        return sum(dice_losses) / C

    def ordering_loss(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Enforce b0 < b1 < b2 < b3."""
        min_gap = 0.02
        violation = 0
        for i in range(3):
            gap = boundaries[:, i+1] - boundaries[:, i]
            violation = violation + F.relu(min_gap - gap).mean()
        return violation / 3

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        gt_mask: torch.Tensor,
        gt_boundaries: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        losses = {}

        # Denoising losses
        denoised = outputs['denoised']
        losses['l1'] = F.l1_loss(denoised, clean) * self.lambda_l1

        # Soft Dice loss
        soft_masks = outputs['soft_masks']
        losses['dice'] = self.soft_dice_loss(soft_masks, gt_mask) * self.lambda_dice

        # Boundary loss
        boundaries = outputs['boundaries']
        losses['boundary'] = F.l1_loss(boundaries, gt_boundaries) * self.lambda_boundary

        # Ordering constraint
        losses['ordering'] = self.ordering_loss(boundaries) * self.lambda_ordering

        total = sum(losses.values())
        losses['total'] = total

        return total, losses


# =============================================================================
# Dataset
# =============================================================================
class JointDatasetV2(Dataset):
    """Dataset with boundary extraction."""

    def __init__(
        self,
        jsonl_path: str,
        patch_size: int = 128,
        noise_levels: list = None,
    ):
        self.patch_size = patch_size
        self.noise_levels = noise_levels or [0.05, 0.1, 0.15]

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def add_noise(self, clean: np.ndarray, level: float) -> np.ndarray:
        speckle = np.random.rayleigh(scale=level, size=clean.shape)
        noisy = clean * (1 + speckle)
        noisy = noisy + np.random.normal(0, level * 0.3, clean.shape)
        return np.clip(noisy, 0, 1).astype(np.float32)

    def extract_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract boundary positions from segmentation mask.

        Boundaries are at transitions between classes:
        - b0 (ILM): top of retina (first non-background pixel)
        - b1 (RNFL_INL): transition from class 0 to class 1
        - b2 (INL_ISOS): transition from class 1 to class 2
        - b3 (ISOS_RPE): transition from class 2 to class 3
        """
        H, W = mask.shape
        boundaries = np.zeros((4, W), dtype=np.float32)

        for col in range(W):
            column = mask[:, col]

            # Find first occurrence of each class transition
            b_positions = []

            # Find where each class starts (boundary positions)
            for target_class in range(1, 4):  # Classes 1, 2, 3
                # Find first row where class >= target_class
                indices = np.where(column >= target_class)[0]
                if len(indices) > 0:
                    b_positions.append(indices[0])
                else:
                    # If class not found, extrapolate from previous
                    if b_positions:
                        b_positions.append(min(b_positions[-1] + 5, H - 1))
                    else:
                        b_positions.append(H // 2)

            # b0 (ILM) - top of retina (where class 0 starts, typically row 0 or first non-zero)
            nonzero = np.where(column >= 0)[0]  # All rows have some class
            b0 = nonzero[0] if len(nonzero) > 0 else 0

            # Assign boundaries (normalized 0-1)
            boundaries[0, col] = b0 / (H - 1) if H > 1 else 0
            for i, pos in enumerate(b_positions):
                if i + 1 < 4:
                    boundaries[i + 1, col] = pos / (H - 1) if H > 1 else 0

            # Ensure ordering: b0 < b1 < b2 < b3
            for i in range(1, 4):
                if boundaries[i, col] <= boundaries[i-1, col]:
                    boundaries[i, col] = boundaries[i-1, col] + 0.02

            # Clamp to valid range
            boundaries[:, col] = np.clip(boundaries[:, col], 0, 1)

        return boundaries

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and resize image
        img = Image.open(sample['image_path']).convert('L')
        img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        clean = np.array(img, dtype=np.float32) / 255.0

        # Load and resize mask
        mask = Image.open(sample['mask_path']).convert('L')
        mask = mask.resize((self.patch_size, self.patch_size), Image.NEAREST)
        mask = np.array(mask)

        # Convert 1-indexed to 0-indexed
        if mask.max() > NUM_CLASSES - 1:
            mask = mask - 1
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Add noise
        noise_level = np.random.choice(self.noise_levels)
        noisy = self.add_noise(clean, noise_level)

        # Extract boundaries
        boundaries = self.extract_boundaries(mask)

        return {
            'clean': torch.from_numpy(clean).unsqueeze(0),
            'noisy': torch.from_numpy(noisy).unsqueeze(0),
            'mask': torch.from_numpy(mask.astype(np.int64)),
            'boundaries': torch.from_numpy(boundaries),
        }


# =============================================================================
# Training
# =============================================================================
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_psnr = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)
        boundaries = batch['boundaries'].to(device)

        optimizer.zero_grad()
        outputs = model(noisy)
        loss, losses = criterion(outputs, clean, mask, boundaries)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute metrics before clearing
        with torch.no_grad():
            psnr = compute_psnr(outputs['denoised'].detach(), clean)

        total_loss += loss.item()
        total_psnr += psnr
        n += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'psnr': f"{psnr:.2f}"})

        # Memory cleanup every 10 batches
        if n % 10 == 0:
            del outputs, loss, losses
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / n, total_psnr / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    dice_per_class = {name: 0 for name in CLASS_NAMES}
    n = 0

    for batch in tqdm(loader, desc="Validation"):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)
        boundaries = batch['boundaries'].to(device)

        outputs = model(noisy)
        loss, _ = criterion(outputs, clean, mask, boundaries)

        total_loss += loss.item()
        total_psnr += compute_psnr(outputs['denoised'], clean)

        # Per-class Dice
        pred_seg = outputs['segmentation']
        for c, name in enumerate(CLASS_NAMES):
            pred_c = (pred_seg == c).float()
            gt_c = (mask == c).float()
            inter = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice_per_class[name] += (2 * inter / (union + 1e-8)).item()

        n += 1

        # Cleanup
        del outputs, loss

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        'loss': total_loss / n if n > 0 else 0,
        'psnr': total_psnr / n if n > 0 else 0,
    }
    for name in CLASS_NAMES:
        results[f'dice_{name}'] = dice_per_class[name] / n if n > 0 else 0
    results['dice_avg'] = sum(dice_per_class.values()) / len(dice_per_class) / n if n > 0 else 0

    return results


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size (256 recommended to match segmentation checkpoint)')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)

    # Model
    parser.add_argument('--hidden_channels', type=int, default=48)
    parser.add_argument('--nafnet_width', type=int, default=64,
                        help='NAFNet width (must match checkpoint, default=64)')
    parser.add_argument('--nafnet_ckpt', default=DEFAULT_NAFNET_CKPT)
    parser.add_argument('--physics_ckpt', default='outputs/physics_v3_dice_v2/stage3_256/best_model.pt',
                        help='Pretrained physics/segmentation checkpoint (256x256)')
    parser.add_argument('--blend_sigma', type=float, default=7.0)
    parser.add_argument('--freeze_nafnet', action='store_true')
    parser.add_argument('--freeze_boundary', action='store_true',
                        help='Freeze the boundary detection model')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')

    # Loss
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_dice', type=float, default=0.5)
    parser.add_argument('--lambda_boundary', type=float, default=0.3)
    parser.add_argument('--lambda_ordering', type=float, default=0.1)

    # Output
    parser.add_argument('--output_dir', default='outputs/joint_physics_v2')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("Joint Physics Denoising V2")
    logger.info("=" * 60)
    logger.info(f"Physics checkpoint: {args.physics_ckpt}")
    logger.info(f"NAFNet checkpoint: {args.nafnet_ckpt}")
    logger.info(f"Blend sigma: {args.blend_sigma}px (soft boundary transitions)")
    logger.info(f"Patch size: {args.patch_size}x{args.patch_size}")
    logger.info(f"Device: {device}")

    # Create model
    model = JointPhysicsDenoiserV2(
        hidden_channels=args.hidden_channels,
        nafnet_width=args.nafnet_width,
        nafnet_ckpt=args.nafnet_ckpt,
        physics_ckpt=args.physics_ckpt,
        blend_sigma=args.blend_sigma,
        freeze_nafnet=args.freeze_nafnet,
        freeze_boundary=args.freeze_boundary,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {params:,} total, {trainable:,} trainable")

    # Datasets
    train_ds = JointDatasetV2(args.train_jsonl, args.patch_size)
    val_ds = JointDatasetV2(args.val_jsonl, args.patch_size)

    if args.max_train:
        train_ds.samples = train_ds.samples[:args.max_train]
    if args.max_val:
        val_ds.samples = val_ds.samples[:args.max_val]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Loss and optimizer
    criterion = JointLossV2(
        lambda_l1=args.lambda_l1,
        lambda_dice=args.lambda_dice,
        lambda_boundary=args.lambda_boundary,
        lambda_ordering=args.lambda_ordering,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_psnr = 0
    best_dice = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_results = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Train: loss={train_loss:.4f}, psnr={train_psnr:.2f}")
        logger.info(f"Val: loss={val_results['loss']:.4f}, psnr={val_results['psnr']:.2f}, dice={val_results['dice_avg']:.3f}")
        logger.info(f"  RNFL={val_results['dice_RNFL_GCL']:.3f}, INL={val_results['dice_INL_OPL_ONL']:.3f}, "
                   f"IS_OS={val_results['dice_IS_OS']:.3f}, RPE={val_results['dice_RPE_Choroid']:.3f}")

        # Save best
        if val_results['psnr'] > best_psnr:
            best_psnr = val_results['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(args.output_dir, 'best_psnr.pth'))
            logger.info(f"  -> New best PSNR: {best_psnr:.2f}")

        if val_results['dice_avg'] > best_dice:
            best_dice = val_results['dice_avg']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best_dice.pth'))
            logger.info(f"  -> New best Dice: {best_dice:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Training complete! Best PSNR: {best_psnr:.2f}, Best Dice: {best_dice:.3f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
