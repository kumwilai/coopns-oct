#!/usr/bin/env python3
"""
Quick Boundary Detection Training for IS_OS Layer

Strategy:
- 3-class segmentation: RNFL_GCL, INL_OPL_ONL, RPE_Choroid
- IS_OS detected as boundary between INL_OPL_ONL and RPE_Choroid
- Fast iterations with small data subset

This approach treats IS_OS as a BOUNDARY rather than a region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm

# ============================================================================
# Configuration for FAST iterations
# ============================================================================
NUM_CLASSES = 3  # RNFL_GCL, INL_OPL_ONL, RPE_Choroid (IS_OS as boundary)
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'RPE_Choroid']
MAX_TRAIN_SAMPLES = 400  # Full training data for better results
MAX_VAL_SAMPLES = 100
PATCH_SIZE = 128  # Smaller patches for speed
BATCH_SIZE = 8
NUM_EPOCHS = 15  # More epochs for convergence
LR = 5e-4  # Stable LR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def remap_mask_to_3class(mask):
    """
    Remap 5-class mask to 3-class (IS_OS becomes boundary, not a class).

    Original: 0=RNFL, 1=INL, 2=ONL, 3=IS_OS, 4=RPE
    New: 0=RNFL_GCL, 1=INL_OPL_ONL, 2=RPE_Choroid
    IS_OS pixels → assigned to nearest neighbor (INL or RPE)
    """
    new_mask = np.zeros_like(mask)

    # RNFL_GCL (class 0) stays 0
    new_mask[mask == 0] = 0

    # INL_OPL (class 1) + ONL (class 2) → class 1
    new_mask[mask == 1] = 1
    new_mask[mask == 2] = 1

    # IS_OS (class 3) → will be boundary, assign to class 1 (above) for now
    new_mask[mask == 3] = 1  # Could also be 2, will be refined by boundary

    # RPE_Choroid (class 4) → class 2
    new_mask[mask == 4] = 2

    return new_mask


def extract_is_os_boundary(mask_5class):
    """
    Extract IS_OS boundary from 5-class mask.
    Returns binary mask where IS_OS pixels are 1.
    """
    return (mask_5class == 3).astype(np.float32)


class QuickOCTDataset(Dataset):
    """Fast dataset for quick iterations."""

    def __init__(self, jsonl_path, max_samples=100, patch_size=128):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f][:max_samples]
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        # Random crop
        H, W = image.shape
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        image = image[top:top+self.patch_size, left:left+self.patch_size]
        mask_5class = mask_5class[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if image.shape[0] < self.patch_size or image.shape[1] < self.patch_size:
            pad_h = self.patch_size - image.shape[0]
            pad_w = self.patch_size - image.shape[1]
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_5class = np.pad(mask_5class, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Create 3-class mask and IS_OS boundary
        mask_3class = remap_mask_to_3class(mask_5class)
        is_os_boundary = extract_is_os_boundary(mask_5class)

        return {
            'image': torch.from_numpy(image).float().unsqueeze(0),
            'mask_3class': torch.from_numpy(mask_3class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
        }


class SimpleBoundarySegmenter(nn.Module):
    """
    Simple U-Net style segmenter with dual heads:
    1. 3-class segmentation head
    2. IS_OS boundary detection head
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder
        self.enc1 = self._block(1, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)

        # Segmentation head (3 classes)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

        # Boundary detection head (1 channel for IS_OS boundary)
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
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Heads
        seg_logits = self.seg_head(d1)
        boundary_logits = self.boundary_head(d1)

        return seg_logits, boundary_logits


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


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_seg_loss = 0
    total_boundary_loss = 0
    total_dice = [0, 0, 0]
    total_boundary_dice = 0
    n_batches = 0

    # Class weights for 3-class segmentation
    # RNFL=60%, INL=4%, RPE=31% -> weight inversely
    # But RNFL was collapsing, so boost it
    class_weights = torch.tensor([2.0, 4.0, 1.0], device=device)

    for batch in tqdm(loader, desc='Training'):
        image = batch['image'].to(device)
        mask_3class = batch['mask_3class'].to(device)
        is_os_boundary = batch['is_os_boundary'].to(device)

        optimizer.zero_grad()

        seg_logits, boundary_logits = model(image)

        # Segmentation loss (weighted CE + Dice)
        seg_ce = F.cross_entropy(seg_logits, mask_3class, weight=class_weights)
        seg_probs = F.softmax(seg_logits, dim=1)

        # Dice loss per class
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
        # Use high positive weight since boundaries are rare (~4% of pixels)
        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_logits, is_os_boundary,
            pos_weight=torch.tensor([10.0], device=device)
        )

        # Boundary Dice
        boundary_probs = torch.sigmoid(boundary_logits)
        b_intersection = (boundary_probs * is_os_boundary).sum()
        b_union = boundary_probs.sum() + is_os_boundary.sum() + 1e-6
        boundary_dice_loss = 1 - (2 * b_intersection / b_union)

        boundary_loss = boundary_bce + boundary_dice_loss

        # Total loss - weight boundary (2.0 works better than 4.0)
        loss = seg_loss + 2.0 * boundary_loss

        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            dice_scores = compute_dice(seg_logits, mask_3class, NUM_CLASSES)
            b_dice = compute_boundary_dice(boundary_logits, is_os_boundary)

        total_seg_loss += seg_loss.item()
        total_boundary_loss += boundary_loss.item()
        for i in range(NUM_CLASSES):
            total_dice[i] += dice_scores[i]
        total_boundary_dice += b_dice
        n_batches += 1

    return {
        'seg_loss': total_seg_loss / n_batches,
        'boundary_loss': total_boundary_loss / n_batches,
        'dice': [d / n_batches for d in total_dice],
        'boundary_dice': total_boundary_dice / n_batches,
    }


def validate(model, loader, device):
    model.eval()
    total_dice = [0, 0, 0]
    total_boundary_dice = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            image = batch['image'].to(device)
            mask_3class = batch['mask_3class'].to(device)
            is_os_boundary = batch['is_os_boundary'].to(device)

            seg_logits, boundary_logits = model(image)

            dice_scores = compute_dice(seg_logits, mask_3class, NUM_CLASSES)
            b_dice = compute_boundary_dice(boundary_logits, is_os_boundary)

            for i in range(NUM_CLASSES):
                total_dice[i] += dice_scores[i]
            total_boundary_dice += b_dice
            n_batches += 1

    return {
        'dice': [d / n_batches for d in total_dice],
        'boundary_dice': total_boundary_dice / n_batches,
    }


def main():
    print("=" * 60)
    print("QUICK BOUNDARY DETECTION EXPERIMENT")
    print("=" * 60)
    print(f"Classes: {CLASS_NAMES}")
    print(f"IS_OS: Detected as BOUNDARY (not region)")
    print(f"Train samples: {MAX_TRAIN_SAMPLES}, Val samples: {MAX_VAL_SAMPLES}")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LR}")
    print("=" * 60)

    # Data
    train_ds = QuickOCTDataset('combined_train.jsonl', MAX_TRAIN_SAMPLES, PATCH_SIZE)
    val_ds = QuickOCTDataset('combined_val.jsonl', MAX_VAL_SAMPLES, PATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = SimpleBoundarySegmenter(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_boundary_dice = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = validate(model, val_loader, DEVICE)

        scheduler.step()

        # Print results
        print(f"\nTraining:")
        print(f"  Seg Loss: {train_metrics['seg_loss']:.4f}")
        print(f"  Boundary Loss: {train_metrics['boundary_loss']:.4f}")
        print(f"  Dice: {[f'{d:.4f}' for d in train_metrics['dice']]}")
        print(f"  IS_OS Boundary Dice: {train_metrics['boundary_dice']:.4f}")

        print(f"\nValidation:")
        print(f"  Per-class Dice:")
        for i, name in enumerate(CLASS_NAMES):
            dice = val_metrics['dice'][i]
            status = "OK" if dice > 0.3 else "LEARNING" if dice > 0.1 else "COLLAPSED"
            print(f"    {name:15s}: {dice:.4f} [{status}]")
        print(f"  IS_OS Boundary Dice: {val_metrics['boundary_dice']:.4f}")

        # Track best
        if val_metrics['boundary_dice'] > best_boundary_dice:
            best_boundary_dice = val_metrics['boundary_dice']
            print(f"  *** NEW BEST IS_OS Boundary: {best_boundary_dice:.4f} ***")
            torch.save(model.state_dict(), 'best_boundary_model.pth')

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best IS_OS Boundary Dice: {best_boundary_dice:.4f}")
    print(f"Final 3-class Dice: {[f'{d:.4f}' for d in val_metrics['dice']]}")

    return best_boundary_dice, val_metrics['dice']


if __name__ == '__main__':
    main()
