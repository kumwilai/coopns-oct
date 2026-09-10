#!/usr/bin/env python3
"""
Train 4-class segmentation model for OCT layers.

Classes:
  0: RNFL_GCL (Retinal Nerve Fiber Layer + Ganglion Cell Layer)
  1: INL_OPL_ONL (Inner Nuclear + Outer Plexiform + Outer Nuclear Layers)
  2: IS_OS (Inner/Outer Segment junction - photoreceptors)
  3: RPE_Choroid (Retinal Pigment Epithelium + Choroid)

This creates pretrained weights for the TMI Enhanced model.
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


class SimpleBoundarySegmenter(nn.Module):
    """Simple 4-class segmenter for clinical OCT regions."""

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

        # Segmentation head
        self.seg_head = nn.Conv2d(base_filters, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Decoder with skip connections
        d2 = self.up2(e3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Segmentation output
        seg_logits = self.seg_head(d1)

        return seg_logits


class HybridBoundarySegmenter(nn.Module):
    """
    Hybrid model combining segmentation + direct boundary regression.

    Key insight: Semantic segmentation struggles with precise boundary localization.
    This model adds a dedicated boundary regression head that directly predicts
    the y-coordinate of each layer boundary per column.

    Boundaries (3 total for 4 classes):
      - Boundary 0: RNFL_GCL / INL_OPL_ONL interface
      - Boundary 1: INL_OPL_ONL / IS_OS interface (critical for IS/OS MAE)
      - Boundary 2: IS_OS / RPE_Choroid interface
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_boundaries = num_classes - 1  # 3 boundaries for 4 classes

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

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )

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

        # Segmentation head
        self.seg_head = nn.Conv2d(base_filters, num_classes, 1)

        # Boundary Regression Head (operates on column-pooled features)
        # Uses 1D convolutions along width dimension after pooling height
        self.boundary_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool height, keep width
        self.boundary_conv1 = nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2)
        self.boundary_bn1 = nn.BatchNorm1d(base_filters*2)
        self.boundary_conv2 = nn.Conv1d(base_filters*2, base_filters*2, kernel_size=5, padding=2)
        self.boundary_bn2 = nn.BatchNorm1d(base_filters*2)
        self.boundary_conv3 = nn.Conv1d(base_filters*2, self.num_boundaries, kernel_size=1)

        # Also use deep supervision: predict boundaries from intermediate features
        self.boundary_deep = nn.Conv1d(base_filters*2, self.num_boundaries, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Shared Encoder
        e1 = self.enc1(x)  # [B, 32, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # [B, 128, H/4, W/4]

        # Segmentation Decoder
        d2 = self.up2(e3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 64, H/2, W/2]

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # [B, 32, H, W]

        # Segmentation output
        seg_logits = self.seg_head(d1)  # [B, 4, H, W]

        # Boundary Regression Head
        # Pool height dimension to get column features
        col_features = self.boundary_pool(d1)  # [B, 32, 1, W]
        col_features = col_features.squeeze(2)  # [B, 32, W]

        # 1D convolutions for boundary prediction
        b = F.relu(self.boundary_bn1(self.boundary_conv1(col_features)))  # [B, 64, W]
        b = F.relu(self.boundary_bn2(self.boundary_conv2(b)))  # [B, 64, W]
        boundary_pred = self.boundary_conv3(b)  # [B, 3, W]

        # Scale to [0, 1] range (relative position in image height)
        # Using sigmoid ensures output is in valid range
        boundary_pred = torch.sigmoid(boundary_pred)  # [B, 3, W] in [0, 1]

        # Deep supervision from intermediate decoder
        d2_up = F.interpolate(d2, size=(1, W), mode='bilinear', align_corners=False)
        d2_col = d2_up.squeeze(2)  # [B, 64, W]
        boundary_deep = torch.sigmoid(self.boundary_deep(d2_col))  # [B, 3, W]

        return seg_logits, boundary_pred, boundary_deep


def extract_boundary_positions(mask, num_classes=4):
    """
    Extract ground truth boundary positions from segmentation mask.

    Returns normalized y-coordinates (0-1) for each boundary per column.
    Boundaries are at the transition between adjacent classes.

    Args:
        mask: [B, H, W] class labels

    Returns:
        boundaries: [B, num_boundaries, W] normalized y-positions
        valid_mask: [B, num_boundaries, W] bool mask for valid boundaries
    """
    B, H, W = mask.shape
    num_boundaries = num_classes - 1
    device = mask.device

    boundaries = torch.zeros(B, num_boundaries, W, device=device)
    valid_mask = torch.zeros(B, num_boundaries, W, dtype=torch.bool, device=device)

    for b in range(B):
        for col in range(W):
            column = mask[b, :, col]

            for boundary_idx in range(num_boundaries):
                # Boundary i is between class i and class i+1
                class_above = boundary_idx
                class_below = boundary_idx + 1

                # Find last occurrence of class_above (bottom of upper region)
                positions_above = (column == class_above).nonzero(as_tuple=True)[0]
                # Find first occurrence of class_below (top of lower region)
                positions_below = (column == class_below).nonzero(as_tuple=True)[0]

                if len(positions_above) > 0 and len(positions_below) > 0:
                    # Boundary is between bottom of class_above and top of class_below
                    bottom_above = positions_above[-1].float()
                    top_below = positions_below[0].float()
                    boundary_y = (bottom_above + top_below) / 2.0

                    # Normalize to [0, 1]
                    boundaries[b, boundary_idx, col] = boundary_y / (H - 1)
                    valid_mask[b, boundary_idx, col] = True

    return boundaries, valid_mask


class BoundaryRegressionLoss(nn.Module):
    """
    Loss for direct boundary y-coordinate regression.

    Uses Smooth L1 loss for robustness to outliers.
    Focuses especially on IS/OS boundary (boundary index 1).
    """

    def __init__(self, num_boundaries=3, is_os_weight=3.0):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.is_os_weight = is_os_weight  # Extra weight for IS/OS boundary
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_boundaries, gt_boundaries, valid_mask):
        """
        Args:
            pred_boundaries: [B, 3, W] predicted normalized y-positions
            gt_boundaries: [B, 3, W] ground truth normalized y-positions
            valid_mask: [B, 3, W] bool mask for valid boundaries
        """
        # Compute Smooth L1 loss
        loss = self.smooth_l1(pred_boundaries, gt_boundaries)  # [B, 3, W]

        # Apply validity mask
        loss = loss * valid_mask.float()

        # Weight IS/OS boundary (index 1) more heavily
        weights = torch.ones_like(loss)
        weights[:, 1, :] = self.is_os_weight
        loss = loss * weights

        # Average over valid positions
        num_valid = valid_mask.float().sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        return torch.tensor(0.0, device=pred_boundaries.device)


class OCTSegDataset(Dataset):
    """Dataset for OCT segmentation training with positional encoding."""

    def __init__(self, jsonl_path, max_samples=None, patch_size=128, augment=True,
                 use_full_image=False, add_pos_encoding=True):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.patch_size = patch_size
        self.augment = augment
        self.use_full_image = use_full_image
        self.add_pos_encoding = add_pos_encoding

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(sample['mask_path']))

        H, W = image.shape
        orig_H = H  # Store original height for positional encoding

        if self.use_full_image:
            # Use full image - resize to consistent size for batching
            target_h, target_w = 256, 512  # Standard OCT aspect ratio
            image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((target_w, target_h), Image.BILINEAR)) / 255.0
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((target_w, target_h), Image.NEAREST))
            # For full image, positional encoding is just normalized y coordinates
            patch_y_offset = 0
            patch_H = target_h
            orig_H = target_h
        else:
            # Random crop - track position for positional encoding
            if H > self.patch_size and W > self.patch_size:
                top = np.random.randint(0, H - self.patch_size)
                left = np.random.randint(0, W - self.patch_size)
                image = image[top:top+self.patch_size, left:left+self.patch_size]
                mask = mask[top:top+self.patch_size, left:left+self.patch_size]
                patch_y_offset = top
                patch_H = self.patch_size
            else:
                # Pad if needed
                pad_h = max(0, self.patch_size - H)
                pad_w = max(0, self.patch_size - W)
                if pad_h > 0 or pad_w > 0:
                    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
                    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
                image = image[:self.patch_size, :self.patch_size]
                mask = mask[:self.patch_size, :self.patch_size]
                patch_y_offset = 0
                patch_H = self.patch_size

        # Remap mask to 4 classes
        # IMPORTANT: Duke dataset labels Class 4 as everything from IS_OS to bottom.
        # We split Class 4 into:
        # - IS_OS (class 2): Top ~30 pixels of Class 4 region (photoreceptor junction)
        # - RPE_Choroid (class 3): Everything below IS_OS
        is_os_thickness = 30
        H_mask, W_mask = mask.shape
        mask_4class = np.zeros_like(mask)
        mask_4class[mask == 0] = 0  # Background -> RNFL_GCL region (top)
        mask_4class[mask == 1] = 0  # RNFL -> class 0
        mask_4class[mask == 2] = 0  # GCL -> class 0
        mask_4class[mask == 3] = 1  # INL/OPL/ONL -> class 1

        # Handle Class 4: Split into IS_OS (top 30px) and RPE_Choroid (rest)
        for col in range(W_mask):
            col_mask = mask[:, col]
            class4_positions = np.where(col_mask == 4)[0]
            if len(class4_positions) == 0:
                continue
            # Top of Class 4 region is the IS_OS junction
            is_os_start = class4_positions[0]
            is_os_end = min(is_os_start + is_os_thickness, H_mask)
            # IS_OS: first 30 pixels of Class 4
            mask_4class[is_os_start:is_os_end, col] = 2  # IS_OS
            # RPE_Choroid: everything below IS_OS
            if is_os_end < H_mask:
                for row in range(is_os_end, H_mask):
                    if col_mask[row] == 4 or col_mask[row] == 0:
                        mask_4class[row, col] = 3  # RPE_Choroid

        mask_4class[mask >= 5] = 3  # Explicit RPE/Choroid labels

        # Data augmentation (only horizontal flip - no vertical flip to preserve layer order)
        if self.augment:
            # Horizontal flip only - preserves anatomical layer ordering
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask_4class = np.fliplr(mask_4class).copy()

            # Brightness/contrast
            if np.random.random() > 0.5:
                alpha = 0.8 + np.random.random() * 0.4  # 0.8-1.2
                beta = -0.1 + np.random.random() * 0.2  # -0.1 to 0.1
                image = np.clip(alpha * image + beta, 0, 1)

        # Create positional encoding channel
        # Normalized y-coordinates (0 at top of full image, 1 at bottom)
        if self.add_pos_encoding:
            pos_h, pos_w = image.shape
            y_coords = np.arange(pos_h, dtype=np.float32).reshape(-1, 1)
            # Map to absolute position in original image
            y_coords = (y_coords + patch_y_offset) / max(orig_H, 1)
            y_coords = np.tile(y_coords, (1, pos_w))  # [H, W]

            # Stack image and position as 2 channels
            image_with_pos = np.stack([image, y_coords], axis=0)  # [2, H, W]
            return {
                'image': torch.from_numpy(image_with_pos.astype(np.float32)),
                'mask': torch.from_numpy(mask_4class.astype(np.int64)),
            }
        else:
            return {
                'image': torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
                'mask': torch.from_numpy(mask_4class.astype(np.int64)),
            }


def compute_dice(pred, target, num_classes=4):
    """Compute per-class Dice scores."""
    dice_scores = {}
    class_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    pred_classes = pred.argmax(dim=1)

    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2.0 * intersection / (union + 1e-8)).item()
        else:
            dice = 1.0  # Both empty

        dice_scores[class_names[c]] = dice

    dice_scores['mean'] = np.mean(list(dice_scores.values())[:4])
    return dice_scores


def compute_anatomical_validity(pred_seg, n_classes=4):
    """Check if layers are in correct anatomical order (top to bottom)."""
    if pred_seg.dim() == 4:
        pred_seg = pred_seg.argmax(dim=1)

    B, H, W = pred_seg.shape
    valid_columns = 0
    total_columns = 0

    for b in range(B):
        for col in range(W):
            column = pred_seg[b, :, col]

            # Find centroid of each class
            centroids = {}
            for c in range(n_classes):
                positions = torch.where(column == c)[0]
                if len(positions) > 0:
                    centroids[c] = positions.float().mean().item()

            # Check ordering: class 0 should be above class 1, etc.
            valid = True
            for i in range(n_classes - 1):
                if i in centroids and (i + 1) in centroids:
                    if centroids[i] > centroids[i + 1]:  # Wrong order
                        valid = False
                        break

            if valid and len(centroids) >= 2:
                valid_columns += 1
            total_columns += 1

    return 100.0 * valid_columns / max(total_columns, 1)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)

        loss = 0.0
        for c in range(self.num_classes):
            pred_c = pred_soft[:, c]
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss += 1.0 - dice

        return loss / self.num_classes


class OrderingLoss(nn.Module):
    """
    Ordering loss to enforce anatomical layer ordering in OCT images.

    In OCT images, retinal layers appear in a consistent order from top to bottom:
    - Class 0 (RNFL_GCL) at top
    - Class 1 (INL_OPL_ONL) below that
    - Class 2 (IS_OS) below that
    - Class 3 (RPE_Choroid) at bottom

    This loss penalizes predictions where class centroids are in the wrong order.
    """

    def __init__(self, num_classes=4, margin=15.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin  # Minimum pixel gap between layer centroids (increased for stronger ordering)

    def forward(self, pred, target=None):
        """
        Compute ordering loss.

        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] class labels (optional, for weighted loss)
        """
        pred_soft = F.softmax(pred, dim=1)  # [B, C, H, W]
        B, C, H, W = pred_soft.shape
        device = pred.device

        # Create y-coordinate grid (0 at top, H-1 at bottom)
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        y_coords = y_coords.expand(B, 1, H, W)  # [B, 1, H, W]

        total_loss = 0.0
        valid_pairs = 0

        # For each adjacent pair of classes, enforce ordering
        for c in range(self.num_classes - 1):
            # Get probability maps for adjacent classes
            prob_c = pred_soft[:, c:c+1, :, :]  # [B, 1, H, W]
            prob_c_next = pred_soft[:, c+1:c+2, :, :]  # [B, 1, H, W]

            # Compute weighted centroids (expected y position)
            # centroid = sum(p * y) / sum(p)
            sum_c = prob_c.sum(dim=2, keepdim=True) + 1e-6  # [B, 1, 1, W]
            sum_c_next = prob_c_next.sum(dim=2, keepdim=True) + 1e-6

            centroid_c = (prob_c * y_coords).sum(dim=2, keepdim=True) / sum_c  # [B, 1, 1, W]
            centroid_c_next = (prob_c_next * y_coords).sum(dim=2, keepdim=True) / sum_c_next

            # Ordering violation: class c should be ABOVE class c+1 (smaller y value)
            # Penalize when centroid_c > centroid_c_next - margin
            violation = F.relu(centroid_c - centroid_c_next + self.margin)  # [B, 1, 1, W]

            # Weight by presence of both classes in column
            weight = torch.min(prob_c.sum(dim=2, keepdim=True),
                             prob_c_next.sum(dim=2, keepdim=True))
            weight = weight / (H + 1e-6)  # Normalize

            # Weighted loss
            pair_loss = (violation * weight).sum() / (weight.sum() + 1e-6)
            total_loss += pair_loss
            valid_pairs += 1

        if valid_pairs > 0:
            total_loss = total_loss / valid_pairs

        # Normalize by image height for scale invariance
        return total_loss / H


class BoundaryLoss(nn.Module):
    """Boundary loss for IS/OS layer - penalizes errors at layer boundaries."""

    def __init__(self, is_os_class=2):
        super().__init__()
        self.is_os_class = is_os_class
        # Sobel filters for edge detection
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred, target):
        """
        Compute boundary loss focusing on IS/OS boundary.

        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] class labels
        """
        pred_soft = F.softmax(pred, dim=1)

        # Get IS/OS probability and ground truth
        pred_is_os = pred_soft[:, self.is_os_class:self.is_os_class+1, :, :]  # [B, 1, H, W]
        target_is_os = (target == self.is_os_class).float().unsqueeze(1)  # [B, 1, H, W]

        # Detect boundaries using vertical gradient (IS/OS boundary is horizontal)
        sobel_y = self.sobel_y.to(pred.device)
        pred_boundary = torch.abs(F.conv2d(pred_is_os, sobel_y, padding=1))
        target_boundary = torch.abs(F.conv2d(target_is_os, sobel_y, padding=1))

        # Focus loss on boundary regions
        boundary_mask = (target_boundary > 0.1).float()

        # L1 loss at boundaries
        boundary_loss = (torch.abs(pred_is_os - target_is_os) * boundary_mask).sum()
        boundary_loss = boundary_loss / (boundary_mask.sum() + 1e-6)

        # Also penalize boundary position error
        # Find top of IS/OS in each column
        B, _, H, W = pred_is_os.shape
        position_loss = 0.0
        valid_cols = 0

        for b in range(B):
            for col in range(0, W, 8):  # Sample every 8th column for efficiency
                pred_col = pred_is_os[b, 0, :, col]
                target_col = target_is_os[b, 0, :, col]

                # Find expected position (weighted by probability)
                y_coords = torch.arange(H, device=pred.device, dtype=torch.float32)

                pred_sum = pred_col.sum() + 1e-6
                target_sum = target_col.sum() + 1e-6

                if target_sum > 1.0:  # Has IS/OS in this column
                    pred_pos = (pred_col * y_coords).sum() / pred_sum
                    target_pos = (target_col * y_coords).sum() / target_sum
                    position_loss += torch.abs(pred_pos - target_pos)
                    valid_cols += 1

        if valid_cols > 0:
            position_loss = position_loss / valid_cols / H  # Normalize by image height

        # Stronger weight on position (0.5 instead of 0.1)
        return boundary_loss + 0.5 * position_loss


def train_epoch(model, loader, optimizer, device, ce_weight=1.0, dice_weight=1.0,
                boundary_weight=1.0, ordering_weight=0.5, use_full_image=False,
                use_hybrid=False, boundary_reg_weight=2.0):
    """Train one epoch with optional boundary and ordering losses.

    Loss weights are adjusted based on training mode:
    - Patch mode (Stage 1): Focus on Dice, minimal ordering
    - Full-image mode (Stage 2): Strong boundary + ordering for IS/OS MAE
    - Hybrid mode: Add boundary regression for precise IS/OS localization
    """
    # Adjust weights for full-image training (Stage 2)
    if use_full_image and not use_hybrid:
        boundary_weight = 1.0
        ordering_weight = 0.2  # Light ordering to prevent collapse

    # For hybrid mode, remove ordering loss entirely (regression handles ordering implicitly)
    if use_hybrid:
        ordering_weight = 0.0
        boundary_weight = 0.5  # Reduce old boundary loss, use regression instead

    model.train()
    total_loss = 0
    dice_loss_fn = DiceLoss(num_classes=4).to(device)
    boundary_loss_fn = BoundaryLoss(is_os_class=2).to(device)
    ordering_loss_fn = OrderingLoss(num_classes=4, margin=10.0).to(device)
    boundary_reg_loss_fn = BoundaryRegressionLoss(num_boundaries=3, is_os_weight=3.0).to(device)

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        if use_hybrid:
            # Hybrid model returns (seg_logits, boundary_pred, boundary_deep)
            seg_logits, boundary_pred, boundary_deep = model(image)

            # Extract ground truth boundaries
            gt_boundaries, valid_mask = extract_boundary_positions(mask)

            # Segmentation losses
            ce_loss = F.cross_entropy(seg_logits, mask)
            dice_loss = dice_loss_fn(seg_logits, mask)
            boundary_loss = boundary_loss_fn(seg_logits, mask)

            # Boundary regression loss (main + deep supervision)
            reg_loss_main = boundary_reg_loss_fn(boundary_pred, gt_boundaries, valid_mask)
            reg_loss_deep = boundary_reg_loss_fn(boundary_deep, gt_boundaries, valid_mask)
            boundary_reg_loss = reg_loss_main + 0.5 * reg_loss_deep

            loss = (ce_weight * ce_loss +
                    dice_weight * dice_loss +
                    boundary_weight * boundary_loss +
                    boundary_reg_weight * boundary_reg_loss)

            logits = seg_logits
        else:
            logits = model(image)

            # Combined loss with boundary and ordering terms
            ce_loss = F.cross_entropy(logits, mask)
            dice_loss = dice_loss_fn(logits, mask)
            boundary_loss = boundary_loss_fn(logits, mask)
            ordering_loss = ordering_loss_fn(logits, mask)

            loss = (ce_weight * ce_loss +
                    dice_weight * dice_loss +
                    boundary_weight * boundary_loss +
                    ordering_weight * ordering_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


def compute_is_os_mae(pred, target, is_os_class=2):
    """Compute IS/OS boundary MAE in pixels."""
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)  # [B, H, W]

    B, H, W = pred.shape
    total_mae = 0
    valid_cols = 0

    for b in range(B):
        for col in range(W):
            pred_col = pred[b, :, col]
            target_col = target[b, :, col]

            # Find top of IS/OS (first occurrence)
            pred_is_os = (pred_col == is_os_class).nonzero(as_tuple=True)[0]
            target_is_os = (target_col == is_os_class).nonzero(as_tuple=True)[0]

            if len(pred_is_os) > 0 and len(target_is_os) > 0:
                pred_top = pred_is_os[0].float()
                target_top = target_is_os[0].float()
                total_mae += torch.abs(pred_top - target_top)
                valid_cols += 1

    if valid_cols > 0:
        return (total_mae / valid_cols).item()
    return None


def compute_boundary_mae(pred_boundaries, gt_boundaries, valid_mask, image_height):
    """
    Compute MAE for each boundary in pixels.

    Args:
        pred_boundaries: [B, 3, W] predicted normalized y-positions (0-1)
        gt_boundaries: [B, 3, W] ground truth normalized y-positions
        valid_mask: [B, 3, W] bool mask for valid boundaries
        image_height: int, height of image for denormalization

    Returns:
        dict with MAE for each boundary in pixels
    """
    boundary_names = ['RNFL_INL', 'INL_ISOS', 'ISOS_RPE']
    results = {}

    for i, name in enumerate(boundary_names):
        mask_i = valid_mask[:, i, :]
        if mask_i.sum() > 0:
            pred_i = pred_boundaries[:, i, :][mask_i]
            gt_i = gt_boundaries[:, i, :][mask_i]
            # Convert from normalized (0-1) to pixels
            mae_pixels = (torch.abs(pred_i - gt_i) * image_height).mean().item()
            results[name] = mae_pixels
        else:
            results[name] = None

    return results


@torch.no_grad()
def evaluate(model, loader, device, use_hybrid=False):
    """Evaluate model with IS/OS MAE."""
    model.eval()

    all_dice = {k: [] for k in ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid', 'mean']}
    all_validity = []
    all_is_os_mae = []
    all_is_os_mae_reg = []  # From boundary regression

    for batch in tqdm(loader, desc='Evaluating'):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        _, H, W = mask.shape

        if use_hybrid:
            seg_logits, boundary_pred, _ = model(image)
            logits = seg_logits

            # Compute boundary MAE from regression head
            gt_boundaries, valid_mask = extract_boundary_positions(mask)
            boundary_mae = compute_boundary_mae(boundary_pred, gt_boundaries, valid_mask, H)
            if boundary_mae['INL_ISOS'] is not None:
                all_is_os_mae_reg.append(boundary_mae['INL_ISOS'])
        else:
            logits = model(image)

        dice = compute_dice(logits, mask)
        for k, v in dice.items():
            all_dice[k].append(v)

        validity = compute_anatomical_validity(logits)
        all_validity.append(validity)

        # IS/OS boundary MAE from segmentation
        is_os_mae = compute_is_os_mae(logits, mask)
        if is_os_mae is not None:
            all_is_os_mae.append(is_os_mae)

    results = {k: np.mean(v) for k, v in all_dice.items()}
    results['anatomical_validity'] = np.mean(all_validity)
    if all_is_os_mae:
        results['is_os_mae'] = np.mean(all_is_os_mae)
    if all_is_os_mae_reg:
        results['is_os_mae_reg'] = np.mean(all_is_os_mae_reg)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--full_image', action='store_true', help='Use full images instead of patches (slower, more memory)')
    parser.add_argument('--pos_encoding', action='store_true', default=True, help='Add positional encoding channel')
    parser.add_argument('--no_pos_encoding', action='store_true', help='Disable positional encoding')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid segmentation + boundary regression model')
    parser.add_argument('--output', default='checkpoints/seg_4class_pretrained.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load data
    print("\nLoading data...")
    use_full = getattr(args, 'full_image', False)
    use_pos_encoding = not getattr(args, 'no_pos_encoding', False)
    use_hybrid = getattr(args, 'hybrid', False)

    print(f"Using positional encoding: {use_pos_encoding}")
    print(f"Using hybrid model: {use_hybrid}")

    train_dataset = OCTSegDataset(args.train_jsonl, max_samples=args.max_train,
                                   patch_size=args.patch_size, augment=True,
                                   use_full_image=use_full, add_pos_encoding=use_pos_encoding)
    val_dataset = OCTSegDataset(args.val_jsonl, max_samples=args.max_val,
                                 patch_size=args.patch_size, augment=False,
                                 use_full_image=use_full, add_pos_encoding=use_pos_encoding)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model - 2 channels if using positional encoding, else 1
    in_channels = 2 if use_pos_encoding else 1
    if use_hybrid:
        model = HybridBoundarySegmenter(in_channels=in_channels, num_classes=4, base_filters=32).to(device)
        print("Model: HybridBoundarySegmenter (Segmentation + Boundary Regression)")
    else:
        model = SimpleBoundarySegmenter(in_channels=in_channels, num_classes=4, base_filters=32).to(device)
        print("Model: SimpleBoundarySegmenter")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 use_full_image=use_full, use_hybrid=use_hybrid)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        metrics = evaluate(model, val_loader, device, use_hybrid=use_hybrid)

        print(f"\nValidation Results:")
        print(f"  RNFL_GCL    : {metrics['RNFL_GCL']:.4f}")
        print(f"  INL_OPL_ONL : {metrics['INL_OPL_ONL']:.4f}")
        print(f"  IS_OS       : {metrics['IS_OS']:.4f}")
        print(f"  RPE_Choroid : {metrics['RPE_Choroid']:.4f}")
        print(f"  Mean Dice   : {metrics['mean']:.4f}")
        print(f"  Anatomical  : {metrics['anatomical_validity']:.1f}%")
        if 'is_os_mae' in metrics:
            is_os_status = "[GOOD]" if metrics['is_os_mae'] < 3.0 else "[POOR]"
            print(f"  IS/OS MAE (seg): {metrics['is_os_mae']:.1f} px {is_os_status}")
        if 'is_os_mae_reg' in metrics:
            is_os_status = "[GOOD]" if metrics['is_os_mae_reg'] < 3.0 else "[POOR]"
            print(f"  IS/OS MAE (reg): {metrics['is_os_mae_reg']:.2f} px {is_os_status}")

        # Save best model - for hybrid, prefer regression MAE; otherwise use Dice
        save_metric = metrics.get('is_os_mae_reg', metrics['mean']) if use_hybrid else metrics['mean']
        is_better = (save_metric < best_dice) if use_hybrid and 'is_os_mae_reg' in metrics else (save_metric > best_dice)

        if is_better or (epoch == 0):
            best_dice = save_metric

            # Save in format compatible with TMI model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_dice': metrics['mean'],
                'best_mae_reg': metrics.get('is_os_mae_reg', None),
                'metrics': metrics,
                'in_channels': in_channels,  # Save model config
                'use_pos_encoding': use_pos_encoding,
                'use_hybrid': use_hybrid,
            }
            torch.save(checkpoint, args.output)
            if use_hybrid and 'is_os_mae_reg' in metrics:
                print(f"  -> Saved best model (Reg MAE: {metrics['is_os_mae_reg']:.2f} px)")
            else:
                print(f"  -> Saved best model (Dice: {metrics['mean']:.4f})")

        scheduler.step()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Mean Dice: {best_dice:.4f}")
    print(f"Checkpoint saved to: {args.output}")

    # Also save just the state dict for easy loading
    state_dict_path = args.output.replace('.pth', '_state_dict.pth')
    best_ckpt = torch.load(args.output, map_location='cpu')
    torch.save(best_ckpt['model_state_dict'], state_dict_path)
    print(f"State dict saved to: {state_dict_path}")


if __name__ == '__main__':
    main()
