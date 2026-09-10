#!/usr/bin/env python3
"""
Improved 4-class segmentation model for OCT layers with IS/OS focus.

Key improvements over train_seg_4class.py:
1. Aggressive class weighting (10x for IS_OS) - addresses 62:1 class imbalance
2. Dilated convolutions for larger receptive field (33px vs 15px)
3. Columnar attention for horizontal layer continuity
4. Weighted Dice loss with IS_OS emphasis
5. Surface distance loss for boundary precision
6. Deep supervision at multiple decoder stages
7. 4-boundary regression head (ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE)
8. Post-processing with curve fitting for smooth boundaries
9. Outlier detection and handling for boundary predictions
10. Enhanced data augmentation for robustness

Classes:
  0: RNFL_GCL (Retinal Nerve Fiber Layer + Ganglion Cell Layer)
  1: INL_OPL_ONL (Inner Nuclear + Outer Plexiform + Outer Nuclear Layers)
  2: IS_OS (Inner/Outer Segment junction - photoreceptors) <- CRITICAL
  3: RPE_Choroid (Retinal Pigment Epithelium + Choroid)

Boundaries (4 total):
  - Boundary 0: ILM (top of retina - top of RNFL_GCL)
  - Boundary 1: RNFL_GCL / INL_OPL_ONL interface
  - Boundary 2: INL_OPL_ONL / IS_OS interface (critical)
  - Boundary 3: IS_OS / RPE_Choroid interface (critical)
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# Import base classes from original
from train_seg_4class import (
    compute_dice,
    compute_anatomical_validity,
    compute_is_os_mae,
)


# =============================================================================
# ENHANCED DATASET WITH MORE AUGMENTATIONS
# =============================================================================
class EnhancedOCTSegDataset(Dataset):
    """
    Enhanced dataset with more diverse augmentations for boundary robustness.

    Additional augmentations:
    - Gaussian noise injection (simulates different SNR)
    - Elastic deformation (simulates retinal curvature variation)
    - Local contrast adjustment (simulates local intensity variations)
    - Random crop with boundary-aware sampling
    """

    def __init__(self, jsonl_path, max_samples=None, patch_size=128, augment=True,
                 use_full_image=False, add_pos_encoding=True, enhanced_augment=True):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.patch_size = patch_size
        self.augment = augment
        self.use_full_image = use_full_image
        self.add_pos_encoding = add_pos_encoding
        self.enhanced_augment = enhanced_augment and augment

    def __len__(self):
        return len(self.samples)

    def _apply_enhanced_augmentations(self, image, mask_4class):
        """Apply enhanced augmentations while preserving layer structure."""

        # 1. Gaussian noise injection (30% probability)
        if np.random.random() < 0.3:
            noise_std = np.random.uniform(0.02, 0.08)
            noise = np.random.normal(0, noise_std, image.shape)
            image = np.clip(image + noise, 0, 1)

        # 2. Local contrast adjustment (30% probability)
        if np.random.random() < 0.3:
            # Apply local gamma correction
            gamma = np.random.uniform(0.7, 1.4)
            image = np.power(image + 1e-6, gamma)
            image = np.clip(image, 0, 1)

        # 3. Simulated speckle noise (20% probability) - OCT-specific
        if np.random.random() < 0.2:
            speckle = np.random.gamma(4, 0.25, image.shape)
            image = np.clip(image * speckle, 0, 1)

        # 4. Intensity scaling with random reference point (30% probability)
        if np.random.random() < 0.3:
            # Scale around a random point (not just center)
            ref = np.random.uniform(0.3, 0.7)
            scale = np.random.uniform(0.8, 1.2)
            image = ref + (image - ref) * scale
            image = np.clip(image, 0, 1)

        # 5. Horizontal elastic deformation - DISABLED
        # This causes ambiguous training signals for thin class 1 layer,
        # leading to a local minimum where model outputs low confidence
        if False:  # Disabled - was causing class 1 to get stuck at Dice=0.26
            H, W = image.shape
            # Create smooth horizontal displacement field
            dx = np.zeros((H, W), dtype=np.float32)
            # Only displace horizontally (preserves layer structure)
            num_ctrl = np.random.randint(3, 6)
            ctrl_x = np.linspace(0, W-1, num_ctrl)
            ctrl_y = np.random.uniform(-2, 2, num_ctrl)  # REDUCED from ±5 to ±2 pixels
            for x in range(W):
                # Interpolate displacement
                dx[:, x] = np.interp(x, ctrl_x, ctrl_y)

            # Apply displacement using map_coordinates
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            new_x = np.clip(x_coords + dx, 0, W-1)

            # Bilinear interpolation for image
            image = ndimage.map_coordinates(image, [y_coords.flatten(), new_x.flatten()],
                                           order=1, mode='reflect').reshape(H, W)
            # Nearest for mask - safe with reduced displacement
            mask_4class = ndimage.map_coordinates(mask_4class.astype(np.float32),
                                                  [y_coords.flatten(), new_x.flatten()],
                                                  order=0, mode='reflect').reshape(H, W).astype(np.int64)

        return image, mask_4class

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(sample['mask_path']))

        H, W = image.shape
        orig_H = H

        if self.use_full_image:
            target_h, target_w = 256, 512
            image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize(
                (target_w, target_h), Image.BILINEAR)) / 255.0
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (target_w, target_h), Image.NEAREST))
            patch_y_offset = 0
            orig_H = target_h
        else:
            if H > self.patch_size and W > self.patch_size:
                top = np.random.randint(0, H - self.patch_size)
                left = np.random.randint(0, W - self.patch_size)
                image = image[top:top+self.patch_size, left:left+self.patch_size]
                mask = mask[top:top+self.patch_size, left:left+self.patch_size]
                patch_y_offset = top
            else:
                pad_h = max(0, self.patch_size - H)
                pad_w = max(0, self.patch_size - W)
                if pad_h > 0 or pad_w > 0:
                    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
                    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
                image = image[:self.patch_size, :self.patch_size]
                mask = mask[:self.patch_size, :self.patch_size]
                patch_y_offset = 0

        # Remap mask to 4 classes with IS_OS splitting
        is_os_thickness = 30
        H_mask, W_mask = mask.shape
        mask_4class = np.zeros_like(mask)
        mask_4class[mask == 0] = 0
        mask_4class[mask == 1] = 0
        mask_4class[mask == 2] = 0
        mask_4class[mask == 3] = 1

        for col in range(W_mask):
            col_mask = mask[:, col]
            class4_positions = np.where(col_mask == 4)[0]
            if len(class4_positions) == 0:
                continue
            is_os_start = class4_positions[0]
            is_os_end = min(is_os_start + is_os_thickness, H_mask)
            mask_4class[is_os_start:is_os_end, col] = 2
            if is_os_end < H_mask:
                for row in range(is_os_end, H_mask):
                    if col_mask[row] == 4 or col_mask[row] == 0:
                        mask_4class[row, col] = 3

        mask_4class[mask >= 5] = 3

        # Basic augmentation
        if self.augment:
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask_4class = np.fliplr(mask_4class).copy()

            if np.random.random() > 0.5:
                alpha = 0.8 + np.random.random() * 0.4
                beta = -0.1 + np.random.random() * 0.2
                image = np.clip(alpha * image + beta, 0, 1)

        # Enhanced augmentations
        if self.enhanced_augment:
            image, mask_4class = self._apply_enhanced_augmentations(image, mask_4class)

        # Positional encoding
        if self.add_pos_encoding:
            pos_h, pos_w = image.shape
            y_coords = np.arange(pos_h, dtype=np.float32).reshape(-1, 1)
            y_coords = (y_coords + patch_y_offset) / max(orig_H, 1)
            y_coords = np.tile(y_coords, (1, pos_w))

            image_with_pos = np.stack([image, y_coords], axis=0)
            return {
                'image': torch.from_numpy(image_with_pos.astype(np.float32)),
                'mask': torch.from_numpy(mask_4class.astype(np.int64)),
            }
        else:
            return {
                'image': torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
                'mask': torch.from_numpy(mask_4class.astype(np.int64)),
            }


# =============================================================================
# 4-BOUNDARY EXTRACTION (ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE)
# =============================================================================
def extract_boundary_positions_4(mask, num_classes=4):
    """
    Extract ground truth positions for ALL 4 boundaries from segmentation mask.

    Boundaries:
      - Boundary 0: ILM (top of class 0 - top of retina)
      - Boundary 1: Class 0 / Class 1 interface (RNFL_GCL / INL_OPL_ONL)
      - Boundary 2: Class 1 / Class 2 interface (INL_OPL_ONL / IS_OS) <- critical
      - Boundary 3: Class 2 / Class 3 interface (IS_OS / RPE_Choroid) <- critical

    Args:
        mask: [B, H, W] class labels

    Returns:
        boundaries: [B, 4, W] normalized y-positions (0-1)
        valid_mask: [B, 4, W] bool mask for valid boundaries
    """
    B, H, W = mask.shape
    num_boundaries = 4
    device = mask.device

    boundaries = torch.zeros(B, num_boundaries, W, device=device)
    valid_mask = torch.zeros(B, num_boundaries, W, dtype=torch.bool, device=device)

    for b in range(B):
        for col in range(W):
            column = mask[b, :, col]

            # Boundary 0: ILM (top of class 0)
            positions_0 = (column == 0).nonzero(as_tuple=True)[0]
            if len(positions_0) > 0:
                top_0 = positions_0[0].float()
                boundaries[b, 0, col] = top_0 / (H - 1)
                valid_mask[b, 0, col] = True

            # Boundary 1: Class 0 / Class 1 (bottom of 0, top of 1)
            positions_1 = (column == 1).nonzero(as_tuple=True)[0]
            if len(positions_0) > 0 and len(positions_1) > 0:
                bottom_0 = positions_0[-1].float()
                top_1 = positions_1[0].float()
                boundary_y = (bottom_0 + top_1) / 2.0
                boundaries[b, 1, col] = boundary_y / (H - 1)
                valid_mask[b, 1, col] = True

            # Boundary 2: Class 1 / Class 2 (INL_OPL_ONL / IS_OS)
            positions_2 = (column == 2).nonzero(as_tuple=True)[0]
            if len(positions_1) > 0 and len(positions_2) > 0:
                bottom_1 = positions_1[-1].float()
                top_2 = positions_2[0].float()
                boundary_y = (bottom_1 + top_2) / 2.0
                boundaries[b, 2, col] = boundary_y / (H - 1)
                valid_mask[b, 2, col] = True

            # Boundary 3: Class 2 / Class 3 (IS_OS / RPE_Choroid)
            positions_3 = (column == 3).nonzero(as_tuple=True)[0]
            if len(positions_2) > 0 and len(positions_3) > 0:
                bottom_2 = positions_2[-1].float()
                top_3 = positions_3[0].float()
                boundary_y = (bottom_2 + top_3) / 2.0
                boundaries[b, 3, col] = boundary_y / (H - 1)
                valid_mask[b, 3, col] = True

    return boundaries, valid_mask


# =============================================================================
# POST-PROCESSING: CURVE FITTING & OUTLIER DETECTION
# =============================================================================
def smooth_boundary_curve(boundary, window_length=31, polyorder=3):
    """
    Smooth a boundary curve using Savitzky-Golay filter.

    Args:
        boundary: [W] boundary y-coordinates (numpy array)
        window_length: filter window (must be odd)
        polyorder: polynomial order

    Returns:
        smoothed: [W] smoothed boundary
    """
    W = len(boundary)
    if W < window_length:
        window_length = W if W % 2 == 1 else W - 1
    if window_length < polyorder + 1:
        return boundary  # Can't smooth

    try:
        smoothed = savgol_filter(boundary, window_length, polyorder)
        return smoothed
    except Exception:
        return boundary


def detect_boundary_outliers(boundary, threshold=2.5):
    """
    Detect outliers in boundary predictions using median absolute deviation (MAD).

    Args:
        boundary: [W] boundary y-coordinates
        threshold: MAD multiplier for outlier detection

    Returns:
        outlier_mask: [W] bool mask where True = outlier
        mad: median absolute deviation value
    """
    median = np.median(boundary)
    mad = np.median(np.abs(boundary - median))
    if mad < 1e-6:
        mad = np.std(boundary)  # Fallback to std
    if mad < 1e-6:
        return np.zeros(len(boundary), dtype=bool), 0.0

    deviation = np.abs(boundary - median) / (mad + 1e-6)
    outlier_mask = deviation > threshold
    return outlier_mask, mad


def interpolate_outliers(boundary, outlier_mask):
    """
    Replace outliers with interpolated values from neighboring valid points.

    Args:
        boundary: [W] boundary y-coordinates
        outlier_mask: [W] bool mask where True = outlier

    Returns:
        corrected: [W] boundary with outliers interpolated
    """
    W = len(boundary)
    corrected = boundary.copy()
    valid_indices = np.where(~outlier_mask)[0]
    outlier_indices = np.where(outlier_mask)[0]

    if len(valid_indices) < 2 or len(outlier_indices) == 0:
        return corrected

    # Interpolate outliers from valid neighbors
    valid_values = boundary[valid_indices]
    corrected[outlier_indices] = np.interp(outlier_indices, valid_indices, valid_values)

    return corrected


def postprocess_boundaries(boundary_pred, image_height, smooth=True, fix_outliers=True):
    """
    Post-process boundary predictions with curve fitting and outlier detection.

    Args:
        boundary_pred: [B, 4, W] normalized boundary predictions (torch tensor)
        image_height: height of the image in pixels
        smooth: whether to apply smoothing
        fix_outliers: whether to detect and fix outliers

    Returns:
        processed: [B, 4, W] processed boundary predictions (torch tensor)
        outlier_counts: [B, 4] number of outliers detected per boundary
    """
    B, num_boundaries, W = boundary_pred.shape
    device = boundary_pred.device

    # Convert to numpy for processing
    pred_np = boundary_pred.cpu().numpy() * image_height  # Convert to pixels
    processed_np = pred_np.copy()
    outlier_counts = np.zeros((B, num_boundaries), dtype=np.int32)

    for b in range(B):
        for i in range(num_boundaries):
            boundary = pred_np[b, i, :]

            # Step 1: Detect outliers
            if fix_outliers:
                outlier_mask, _ = detect_boundary_outliers(boundary, threshold=2.5)
                outlier_counts[b, i] = outlier_mask.sum()

                # Interpolate outliers
                if outlier_counts[b, i] > 0:
                    boundary = interpolate_outliers(boundary, outlier_mask)

            # Step 2: Smooth the curve
            if smooth:
                boundary = smooth_boundary_curve(boundary, window_length=31, polyorder=3)

            processed_np[b, i, :] = boundary

        # Step 3: Enforce ordering constraint
        for i in range(num_boundaries - 1):
            # Boundary i should be above (smaller y) than boundary i+1
            violation = processed_np[b, i, :] > processed_np[b, i+1, :] - 2  # 2px margin
            if violation.any():
                # Average the conflicting boundaries
                avg = (processed_np[b, i, violation] + processed_np[b, i+1, violation]) / 2
                processed_np[b, i, violation] = avg - 1
                processed_np[b, i+1, violation] = avg + 1

    # Convert back to normalized coordinates
    processed_np = processed_np / image_height
    processed = torch.from_numpy(processed_np.astype(np.float32)).to(device)

    return processed, outlier_counts


# =============================================================================
# IMPROVEMENT 1: Columnar Attention Module
# =============================================================================
class ColumnarAttention(nn.Module):
    """
    Column-wise attention for OCT layer segmentation.
    OCT layers are roughly horizontal, so we apply attention along vertical axis.
    This helps the model understand that IS/OS should be continuous horizontally.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels

        # Vertical attention (along height) - lightweight version
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # Horizontal smoothing (layers are continuous horizontally)
        self.horizontal_smooth = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3),
                                           groups=channels, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # For memory efficiency, process in chunks if W is large
        if W > 64:
            # Process every other column for attention, interpolate the rest
            x_sampled = x[:, :, :, ::2]  # [B, C, H, W/2]
            out_sampled = self._attention(x_sampled)
            out = F.interpolate(out_sampled, size=(H, W), mode='bilinear', align_corners=False)
        else:
            out = self._attention(x)

        out = self.norm(x + self.gamma * out)
        return out

    def _attention(self, x):
        B, C, H, W = x.shape

        # Compute Q, K, V
        q = self.query(x)  # [B, C/r, H, W]
        k = self.key(x)
        v = self.value(x)

        # Reshape for vertical attention
        q = q.permute(0, 3, 2, 1).reshape(B * W, H, -1)  # [B*W, H, C/r]
        k = k.permute(0, 3, 2, 1).reshape(B * W, H, -1)
        v = v.permute(0, 3, 2, 1).reshape(B * W, H, -1)

        # Self-attention along vertical axis
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v  # [B*W, H, C]

        # Reshape back
        out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)  # [B, C, H, W]

        # Horizontal smoothing for layer continuity
        out = self.horizontal_smooth(out)

        return out


# =============================================================================
# IMPROVEMENT 2: Improved Segmenter with Dilated Convolutions
# =============================================================================
class ImprovedBoundarySegmenter(nn.Module):
    """
    Improved 4-class segmenter with:
    1. Dilated convolutions for larger receptive field (critical for thin IS/OS)
    2. Columnar attention for horizontal continuity
    3. Deep supervision for better gradient flow to early layers
    4. Boundary-aware refinement
    5. 4-boundary regression head (ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE)
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_boundaries = 4  # Changed from num_classes - 1 to 4

        # Encoder with increasing dilation for larger receptive field
        # Stage 1: dilation=1, receptive field ~5px
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2: dilation=1, receptive field ~11px after pooling
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3: DILATED convolutions for larger receptive field
        # dilation=2 gives receptive field ~27px (covers IS/OS layer ~8.5px)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )

        # Stage 4: Even larger dilation for global context
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )

        # Columnar attention at bottleneck
        self.col_attn = ColumnarAttention(base_filters*4, reduction=4)

        # Decoder with skip connections
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

        # Main segmentation head
        self.seg_head = nn.Conv2d(base_filters, num_classes, 1)

        # Deep supervision heads (auxiliary losses)
        self.aux_head2 = nn.Conv2d(base_filters*2, num_classes, 1)  # 1/2 resolution
        self.aux_head3 = nn.Conv2d(base_filters*4, num_classes, 1)  # 1/4 resolution

        # Boundary regression head (for direct IS/OS localization)
        self.boundary_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool height, keep width
        self.boundary_conv1 = nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2)
        self.boundary_bn1 = nn.BatchNorm1d(base_filters*2)
        self.boundary_conv2 = nn.Conv1d(base_filters*2, base_filters*2, kernel_size=5, padding=2)
        self.boundary_bn2 = nn.BatchNorm1d(base_filters*2)
        self.boundary_head = nn.Conv1d(base_filters*2, self.num_boundaries, kernel_size=1)

        # Column-wise refinement (layers are roughly horizontal)
        self.column_refine = nn.Conv2d(num_classes, num_classes, (7, 1), padding=(3, 0))

    def forward(self, x, return_aux=True):
        """
        Forward pass.

        Returns:
            seg_logits: [B, 4, H, W] main segmentation logits
            boundary_pred: [B, 4, W] boundary y-coordinates (normalized 0-1)
                          - 0: ILM (top of retina)
                          - 1: RNFL_GCL / INL_OPL_ONL
                          - 2: INL_OPL_ONL / IS_OS (critical)
                          - 3: IS_OS / RPE_Choroid (critical)
            aux_outputs: list of auxiliary segmentation logits (if return_aux=True)
        """
        input_size = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)  # [B, 32, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # [B, 128, H/4, W/4]
        e4 = self.enc4(e3)  # [B, 128, H/4, W/4] - dilated, same size

        # Apply columnar attention
        e4 = self.col_attn(e4)

        # Decoder with skip connections
        d2 = self.up2(e4)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Main segmentation output with column-wise refinement
        seg_logits = self.seg_head(d1)
        seg_logits = seg_logits + 0.1 * self.column_refine(seg_logits)

        # Boundary regression (4 boundaries: ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE)
        B, C, H, W = d1.shape
        col_features = self.boundary_pool(d1).squeeze(2)  # [B, 32, W]
        b = F.relu(self.boundary_bn1(self.boundary_conv1(col_features)))
        b = F.relu(self.boundary_bn2(self.boundary_conv2(b)))
        boundary_pred = torch.sigmoid(self.boundary_head(b))  # [B, 4, W] in [0, 1]

        if return_aux:
            # Auxiliary outputs for deep supervision
            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=False)

            aux3 = self.aux_head3(e4)
            aux3 = F.interpolate(aux3, size=input_size, mode='bilinear', align_corners=False)

            return seg_logits, boundary_pred, [aux2, aux3]

        return seg_logits, boundary_pred, None


# =============================================================================
# IMPROVEMENT 3: Weighted Dice Loss with IS/OS Emphasis
# =============================================================================
class WeightedDiceLoss(nn.Module):
    """
    Dice loss with per-class weighting.
    Critical for IS/OS which is only 1% of pixels but clinically important.
    """

    def __init__(self, num_classes=4, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

        # Default weights: heavily emphasize IS_OS (class 2)
        if class_weights is None:
            # Based on inverse frequency analysis: IS_OS needs ~10x weight
            class_weights = [1.0, 4.0, 10.0, 4.0]

        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)

        dice_per_class = []
        for c in range(self.num_classes):
            pred_c = pred_soft[:, c]
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(1.0 - dice)

        dice_losses = torch.stack(dice_per_class)

        # Weighted average
        weighted_loss = (dice_losses * self.class_weights).sum() / self.class_weights.sum()

        return weighted_loss


# =============================================================================
# IMPROVEMENT 4: Surface Distance Loss for Boundary Precision
# =============================================================================
class SurfaceDistanceLoss(nn.Module):
    """
    Penalizes boundary distance errors, especially for IS/OS.
    Uses distance transform approximation for efficiency.
    """

    def __init__(self, is_os_class=2, is_os_weight=5.0):
        super().__init__()
        self.is_os_class = is_os_class
        self.is_os_weight = is_os_weight

        # Sobel filter for edge detection
        self.register_buffer('sobel_y', torch.tensor([
            [[-1., -2., -1.],
             [0., 0., 0.],
             [1., 2., 1.]]
        ]).unsqueeze(0) / 4.0)

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        B, C, H, W = pred_soft.shape
        device = pred.device

        total_loss = 0.0

        for c in range(C):
            pred_c = pred_soft[:, c:c+1]  # [B, 1, H, W]
            target_c = (target == c).float().unsqueeze(1)  # [B, 1, H, W]

            # Detect boundaries
            pred_c_pad = F.pad(pred_c, [1, 1, 1, 1], mode='reflect')
            target_c_pad = F.pad(target_c, [1, 1, 1, 1], mode='reflect')

            pred_boundary = torch.abs(F.conv2d(pred_c_pad, self.sobel_y.to(device)))
            target_boundary = torch.abs(F.conv2d(target_c_pad, self.sobel_y.to(device)))

            # Surface distance approximation: L2 between boundary probability maps
            boundary_diff = (pred_boundary - target_boundary) ** 2

            # Weight IS/OS boundaries more heavily
            weight = self.is_os_weight if c == self.is_os_class else 1.0
            total_loss += weight * boundary_diff.mean()

        return total_loss / C


# =============================================================================
# IMPROVEMENT 5: Ordering Loss (from original, but with stronger margins)
# =============================================================================
class ImprovedOrderingLoss(nn.Module):
    """
    Ordering loss with adaptive margin based on expected layer thickness.
    IS/OS is thin (~8.5px), so margin should be smaller for IS/OS boundaries.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        # Margin per boundary: [RNFL-INL, INL-IS/OS, IS/OS-RPE]
        # Smaller margin for IS/OS boundaries due to thin layer
        self.register_buffer('margins', torch.tensor([15.0, 5.0, 5.0]))

    def forward(self, pred, target=None):
        pred_soft = F.softmax(pred, dim=1)
        B, C, H, W = pred_soft.shape
        device = pred.device

        # y-coordinate grid
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        y_coords = y_coords.expand(B, 1, H, W)

        total_loss = 0.0

        for c in range(self.num_classes - 1):
            prob_c = pred_soft[:, c:c+1]
            prob_c_next = pred_soft[:, c+1:c+2]

            sum_c = prob_c.sum(dim=2, keepdim=True) + 1e-6
            sum_c_next = prob_c_next.sum(dim=2, keepdim=True) + 1e-6

            centroid_c = (prob_c * y_coords).sum(dim=2, keepdim=True) / sum_c
            centroid_c_next = (prob_c_next * y_coords).sum(dim=2, keepdim=True) / sum_c_next

            # Get margin for this boundary
            margin = self.margins[c]

            # Violation: class c should be ABOVE class c+1
            violation = F.relu(centroid_c - centroid_c_next + margin)

            # Weight by class presence - NO min floor (causes local minimum trap)
            # When a class is missing, let ordering loss naturally reduce to avoid
            # conflicting gradients that push the class further down
            weight = torch.min(prob_c.sum(dim=2, keepdim=True),
                               prob_c_next.sum(dim=2, keepdim=True)) / H

            total_loss += (violation * weight).sum() / (weight.sum() + 1e-6)

        return total_loss / (self.num_classes - 1) / H


# =============================================================================
# IMPROVEMENT 6: Boundary Regression Loss with IS/OS Focus (4 boundaries)
# =============================================================================
class ImprovedBoundaryRegressionLoss(nn.Module):
    """
    Boundary regression loss with:
    1. Heavy weighting on IS/OS boundaries (indices 2 and 3)
    2. Ordering constraint built-in
    3. Support for 4 boundaries: ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE

    Boundary indices:
      - 0: ILM (top of retina)
      - 1: RNFL_GCL / INL_OPL_ONL
      - 2: INL_OPL_ONL / IS_OS (critical - top of IS_OS)
      - 3: IS_OS / RPE_Choroid (critical - bottom of IS_OS)
    """

    def __init__(self, num_boundaries=4, is_os_weight=5.0):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.is_os_weight = is_os_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_boundaries, gt_boundaries, valid_mask, image_height=None):
        """
        Args:
            pred_boundaries: [B, 4, W] predicted normalized y-positions (0-1)
            gt_boundaries: [B, 4, W] ground truth normalized y-positions
            valid_mask: [B, 4, W] bool mask for valid boundaries
        """
        # Position regression loss
        pos_loss = self.smooth_l1(pred_boundaries, gt_boundaries)
        pos_loss = pos_loss * valid_mask.float()

        # Weight boundaries by importance
        # FIX: Only apply high weights when boundary has sufficient valid samples
        # to avoid conflicting gradients when class 1 is collapsed
        weights = torch.ones_like(pos_loss)
        weights[:, 0, :] = 1.0   # ILM - normal weight (always reliable)

        # Boundaries 1, 2 depend on class 1 - only weight highly if valid
        valid_b1 = valid_mask[:, 1, :].float().sum() > 50  # RNFL/INL boundary
        valid_b2 = valid_mask[:, 2, :].float().sum() > 50  # INL/IS_OS boundary
        weights[:, 1, :] = 3.0 if valid_b1 else 0.5  # Reduce weight if class 1 missing
        weights[:, 2, :] = self.is_os_weight if valid_b2 else 0.5
        weights[:, 3, :] = self.is_os_weight  # IS_OS/RPE - usually reliable
        pos_loss = pos_loss * weights

        num_valid = valid_mask.float().sum()
        if num_valid > 0:
            pos_loss = pos_loss.sum() / num_valid
        else:
            pos_loss = torch.tensor(0.0, device=pred_boundaries.device)

        # Ordering loss: boundary[i] should be < boundary[i+1]
        ordering_loss = 0.0
        for i in range(self.num_boundaries - 1):
            # pred_boundaries[:, i] should be < pred_boundaries[:, i+1]
            violation = F.relu(pred_boundaries[:, i] - pred_boundaries[:, i+1] + 0.01)
            ordering_loss += violation.mean()
        ordering_loss /= max(self.num_boundaries - 1, 1)

        # IS_OS thickness constraint: boundary 3 - boundary 2 should be ~30px / image_height
        if image_height is not None:
            expected_thickness = 30.0 / image_height  # ~30 pixels normalized
            actual_thickness = pred_boundaries[:, 3, :] - pred_boundaries[:, 2, :]
            thickness_loss = F.relu(torch.abs(actual_thickness - expected_thickness) - 0.02).mean()
        else:
            thickness_loss = 0.0

        return pos_loss + 0.5 * ordering_loss + 0.3 * thickness_loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, optimizer, device, class_weights, epoch,
                ce_weight=0.3, dice_weight=0.4, boundary_weight=0.1,
                ordering_weight=0.1, surface_weight=0.05, reg_weight=0.2,
                use_deep_supervision=True):
    """
    Train one epoch with all improvements.

    Loss = ce_weight * WeightedCE
         + dice_weight * WeightedDice
         + boundary_weight * BoundaryLoss
         + ordering_weight * OrderingLoss
         + surface_weight * SurfaceDistanceLoss
         + reg_weight * BoundaryRegressionLoss
    """
    model.train()

    # Loss functions
    dice_loss_fn = WeightedDiceLoss(num_classes=4, class_weights=class_weights.tolist()).to(device)
    ordering_loss_fn = ImprovedOrderingLoss(num_classes=4).to(device)
    surface_loss_fn = SurfaceDistanceLoss(is_os_class=2, is_os_weight=5.0).to(device)
    boundary_reg_fn = ImprovedBoundaryRegressionLoss(num_boundaries=4, is_os_weight=5.0).to(device)

    total_loss = 0
    total_ce = 0
    total_dice = 0
    total_ordering = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        B, _, H, W = image.shape

        optimizer.zero_grad()

        # Forward pass
        seg_logits, boundary_pred, aux_outputs = model(image, return_aux=use_deep_supervision)

        # Main losses
        ce_loss = F.cross_entropy(seg_logits, mask, weight=class_weights.to(device))
        dice_loss = dice_loss_fn(seg_logits, mask)
        ordering_loss = ordering_loss_fn(seg_logits, mask)
        surface_loss = surface_loss_fn(seg_logits, mask)

        # Boundary regression loss (4 boundaries)
        gt_boundaries, valid_mask = extract_boundary_positions_4(mask)
        reg_loss = boundary_reg_fn(boundary_pred, gt_boundaries, valid_mask, image_height=H)

        # Deep supervision (auxiliary losses at 0.4 weight)
        aux_loss = 0.0
        if use_deep_supervision and aux_outputs is not None:
            for aux in aux_outputs:
                aux_ce = F.cross_entropy(aux, mask, weight=class_weights.to(device))
                aux_dice = dice_loss_fn(aux, mask)
                aux_loss += 0.4 * (aux_ce + aux_dice)
            aux_loss /= len(aux_outputs)

        # Combined loss
        loss = (ce_weight * ce_loss +
                dice_weight * dice_loss +
                ordering_weight * ordering_loss +
                surface_weight * surface_loss +
                reg_weight * reg_loss +
                aux_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_dice += dice_loss.item()
        total_ordering += ordering_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'dice': f'{dice_loss.item():.4f}',
        })

    n = len(loader)
    return total_loss / n, total_ce / n, total_dice / n, total_ordering / n


@torch.no_grad()
def evaluate(model, loader, device, use_postprocessing=False):
    """
    Evaluate model with comprehensive metrics.

    Args:
        model: segmentation model
        loader: data loader
        device: torch device
        use_postprocessing: whether to apply curve fitting and outlier detection
    """
    model.eval()

    all_dice = {k: [] for k in ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid', 'mean']}
    all_validity = []
    all_is_os_mae = []

    # 4 boundaries: ILM, RNFL/INL, INL/IS_OS, IS_OS/RPE
    boundary_names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']
    all_boundary_mae = {name: [] for name in boundary_names}
    all_boundary_mae_postproc = {name: [] for name in boundary_names}
    total_outliers = {name: 0 for name in boundary_names}

    for batch in tqdm(loader, desc='Evaluating'):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        B, H, W = mask.shape

        seg_logits, boundary_pred, _ = model(image, return_aux=False)

        # Dice scores
        dice = compute_dice(seg_logits, mask)
        for k, v in dice.items():
            all_dice[k].append(v)

        # Anatomical validity
        validity = compute_anatomical_validity(seg_logits)
        all_validity.append(validity)

        # IS/OS MAE (from segmentation)
        is_os_mae = compute_is_os_mae(seg_logits, mask)
        if is_os_mae is not None:
            all_is_os_mae.append(is_os_mae)

        # Boundary MAE (from 4-boundary regression)
        gt_boundaries, valid_mask = extract_boundary_positions_4(mask)

        # Raw boundary MAE
        for i, name in enumerate(boundary_names):
            if i < boundary_pred.shape[1]:
                mask_i = valid_mask[:, i, :]
                if mask_i.sum() > 0:
                    pred_i = boundary_pred[:, i, :][mask_i]
                    gt_i = gt_boundaries[:, i, :][mask_i]
                    mae_pixels = (torch.abs(pred_i - gt_i) * H).mean().item()
                    all_boundary_mae[name].append(mae_pixels)

        # Post-processed boundary MAE
        if use_postprocessing:
            processed_pred, outlier_counts = postprocess_boundaries(
                boundary_pred, image_height=H, smooth=True, fix_outliers=True)

            for i, name in enumerate(boundary_names):
                if i < processed_pred.shape[1]:
                    mask_i = valid_mask[:, i, :]
                    if mask_i.sum() > 0:
                        pred_i = processed_pred[:, i, :][mask_i]
                        gt_i = gt_boundaries[:, i, :][mask_i]
                        mae_pixels = (torch.abs(pred_i - gt_i) * H).mean().item()
                        all_boundary_mae_postproc[name].append(mae_pixels)
                    total_outliers[name] += outlier_counts[:, i].sum()

    # Compute averages
    results = {}
    for k, v in all_dice.items():
        results[f'dice_{k}'] = np.mean(v) if v else 0.0

    results['validity'] = np.mean(all_validity) if all_validity else 0.0
    results['is_os_mae_seg'] = np.mean(all_is_os_mae) if all_is_os_mae else float('inf')

    # Raw boundary MAE
    for name, maes in all_boundary_mae.items():
        results[f'{name}_mae'] = np.mean(maes) if maes else float('inf')

    # Post-processed boundary MAE
    if use_postprocessing:
        for name, maes in all_boundary_mae_postproc.items():
            results[f'{name}_mae_postproc'] = np.mean(maes) if maes else float('inf')
        for name, count in total_outliers.items():
            results[f'{name}_outliers'] = count

    return results


def main():
    parser = argparse.ArgumentParser(description='Train improved 4-class segmenter')
    parser.add_argument('--train_jsonl', type=str, default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size (256 recommended for IS/OS)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Class weight customization
    parser.add_argument('--is_os_weight', type=float, default=10.0,
                        help='Class weight for IS_OS (default: 10x)')

    # Loss weights (ordering_weight reduced to avoid local minimum trap)
    parser.add_argument('--ce_weight', type=float, default=0.3)
    parser.add_argument('--dice_weight', type=float, default=0.4)
    parser.add_argument('--ordering_weight', type=float, default=0.15,
                        help='Weight for ordering loss (reduced from 0.35 to avoid local minimum)')
    parser.add_argument('--surface_weight', type=float, default=0.05)
    parser.add_argument('--reg_weight', type=float, default=0.2)

    # Enhanced augmentation
    parser.add_argument('--enhanced_augment', action='store_true',
                        help='Use enhanced augmentations (noise, elastic deform, etc.)')
    parser.add_argument('--use_postprocessing', action='store_true',
                        help='Apply post-processing (curve fitting, outlier detection)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Class weights based on inverse frequency + clinical importance
    # INL_OPL_ONL (class 1) is ~3.35% of pixels → needs higher weight but 50x was too aggressive
    # 30x is a balance: higher than 20x (stuck at 0.26) but not 50x (hurt IS_OS)
    class_weights = torch.tensor([1.0, 30.0, args.is_os_weight, 4.0], dtype=torch.float32)
    print(f"\n=== CLASS WEIGHTS (BALANCED) ===")
    print(f"  RNFL_GCL:    {class_weights[0]:.1f}")
    print(f"  INL_OPL_ONL: {class_weights[1]:.1f}  <-- BALANCED: 30x (20x stuck, 50x hurt IS_OS)")
    print(f"  IS_OS:       {class_weights[2]:.1f}  <-- KEY FOR IS/OS DETECTION")
    print(f"  RPE_Choroid: {class_weights[3]:.1f}")

    # Data - use enhanced dataset with more augmentations
    train_dataset = EnhancedOCTSegDataset(
        args.train_jsonl,
        max_samples=args.max_train,
        patch_size=args.patch_size,
        augment=True,
        use_full_image=False,
        add_pos_encoding=True,  # ENABLED: 2-channel input (image + depth encoding)
        enhanced_augment=args.enhanced_augment,  # Enhanced augmentations for robustness
    )
    val_dataset = EnhancedOCTSegDataset(
        args.val_jsonl,
        max_samples=args.max_val,
        patch_size=args.patch_size,
        augment=False,
        use_full_image=False,
        add_pos_encoding=True,  # ENABLED: 2-channel input (image + depth encoding)
        enhanced_augment=False,  # No augmentation for validation
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")

    # Model - 2 channels for image + depth encoding
    model = ImprovedBoundarySegmenter(in_channels=2, num_classes=4, base_filters=32).to(args.device)
    print(f"Model uses 2-channel input (image + depth encoding)")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            # Try to load compatible weights
            state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()

            # Filter out incompatible keys
            compatible = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(compatible)
            model.load_state_dict(model_dict)
            print(f"  Loaded {len(compatible)}/{len(state_dict)} weights")
        else:
            model.load_state_dict(checkpoint, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_is_os_dice = 0.0
    best_epoch = 0

    print(f"\n=== TRAINING ===")
    print(f"Loss weights: CE={args.ce_weight}, Dice={args.dice_weight}, "
          f"Order={args.ordering_weight}, Surface={args.surface_weight}, Reg={args.reg_weight}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} (lr: {scheduler.get_last_lr()[0]:.2e}) ---")

        # Train
        train_loss, train_ce, train_dice, train_ordering = train_epoch(
            model, train_loader, optimizer, args.device, class_weights, epoch,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            ordering_weight=args.ordering_weight,
            surface_weight=args.surface_weight,
            reg_weight=args.reg_weight,
        )

        scheduler.step()

        # Evaluate (with post-processing on final epoch or if requested)
        use_pp = args.use_postprocessing or (epoch == args.epochs)
        val_results = evaluate(model, val_loader, args.device, use_postprocessing=use_pp)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"\nValidation Results:")
        print(f"  Dice Scores:")
        print(f"    RNFL_GCL:    {val_results['dice_RNFL_GCL']:.4f}")
        print(f"    INL_OPL_ONL: {val_results['dice_INL_OPL_ONL']:.4f}")
        print(f"    IS_OS:       {val_results['dice_IS_OS']:.4f}  {'✓' if val_results['dice_IS_OS'] > 0.7 else ''}")
        print(f"    RPE_Choroid: {val_results['dice_RPE_Choroid']:.4f}")
        print(f"    Mean:        {val_results['dice_mean']:.4f}")
        print(f"  Anatomical Validity: {val_results['validity']:.1f}%")
        print(f"  IS/OS MAE (seg):     {val_results['is_os_mae_seg']:.2f} px")
        print(f"  Boundary MAE (4-boundary regression):")
        print(f"    ILM:         {val_results['ILM_mae']:.2f} px")
        print(f"    RNFL/INL:    {val_results['RNFL_INL_mae']:.2f} px")
        print(f"    INL/IS_OS:   {val_results['INL_ISOS_mae']:.2f} px  <-- critical")
        print(f"    IS_OS/RPE:   {val_results['ISOS_RPE_mae']:.2f} px  <-- critical")
        if use_pp:
            print(f"  Post-processed Boundary MAE:")
            print(f"    INL/IS_OS:   {val_results.get('INL_ISOS_mae_postproc', float('inf')):.2f} px")
            print(f"    IS_OS/RPE:   {val_results.get('ISOS_RPE_mae_postproc', float('inf')):.2f} px")
            print(f"  Outliers detected: INL/IS_OS={val_results.get('INL_ISOS_outliers', 0)}, "
                  f"IS_OS/RPE={val_results.get('ISOS_RPE_outliers', 0)}")

        # Save best model based on IS_OS Dice (our primary target)
        is_os_dice = val_results['dice_IS_OS']
        if is_os_dice > best_is_os_dice:
            best_is_os_dice = is_os_dice
            best_epoch = epoch

            save_path = os.path.join(args.output_dir, 'seg_4class_improved_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'is_os_dice': is_os_dice,
                'mean_dice': val_results['dice_mean'],
                'class_weights': class_weights.tolist(),
            }, save_path)
            print(f"  -> Saved best model (IS_OS Dice: {is_os_dice:.4f})")

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Best IS_OS Dice: {best_is_os_dice:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'seg_4class_improved_best.pth')}")


if __name__ == '__main__':
    main()
