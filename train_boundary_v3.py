#!/usr/bin/env python3
"""
Boundary Detection V3: Focused Boundary Detection

Key changes:
1. Simpler architecture - just dual head (seg + boundary)
2. Higher boundary dilation (5px) for stronger signal
3. Focus on boundary loss with higher weight
4. Random cropping for validation (like original which got 0.47)
5. Smaller model for faster learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from scipy import ndimage

# ============================================================================
# Configuration
# ============================================================================
NUM_CLASSES = 3
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'RPE_Choroid']
MAX_TRAIN_SAMPLES = 200
MAX_VAL_SAMPLES = 50
PATCH_SIZE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 1e-3  # Higher LR for faster learning
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BOUNDARY_DILATION = 5  # More dilation for stronger signal


def remap_mask_to_3class(mask):
    """Remap 5-class mask to 3-class."""
    new_mask = np.zeros_like(mask)
    new_mask[mask == 0] = 0  # RNFL_GCL
    new_mask[mask == 1] = 1  # INL
    new_mask[mask == 2] = 1  # ONL -> INL_OPL_ONL
    new_mask[mask == 3] = 1  # IS_OS -> assign to INL_OPL_ONL
    new_mask[mask == 4] = 2  # RPE_Choroid
    return new_mask


def extract_is_os_boundary(mask_5class, dilate=0):
    """Extract IS_OS boundary with optional dilation."""
    boundary = (mask_5class == 3).astype(np.float32)
    if dilate > 0:
        boundary = ndimage.binary_dilation(boundary, iterations=dilate).astype(np.float32)
    return boundary


class BoundaryDatasetV3(Dataset):
    """Simple dataset with boundary dilation."""

    def __init__(self, jsonl_path, max_samples=100, patch_size=128, boundary_dilation=0):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f][:max_samples]
        self.patch_size = patch_size
        self.boundary_dilation = boundary_dilation

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

        # Create targets
        mask_3class = remap_mask_to_3class(mask_5class)
        is_os_boundary = extract_is_os_boundary(mask_5class, self.boundary_dilation)

        return {
            'image': torch.from_numpy(image).float().unsqueeze(0),
            'mask_3class': torch.from_numpy(mask_3class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
        }


class SimpleBoundaryNet(nn.Module):
    """Smaller, focused network for boundary detection."""

    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder (smaller)
        self.enc1 = self._block(1, 24)
        self.enc2 = self._block(24, 48)
        self.enc3 = self._block(48, 96)
        self.enc4 = self._block(96, 192)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec3 = self._block(192, 96)

        self.up2 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec2 = self._block(96, 48)

        self.up1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec1 = self._block(48, 24)

        # Segmentation head
        self.seg_head = nn.Conv2d(24, num_classes, 1)

        # Boundary head - deeper for better boundary detection
        self.boundary_head = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 1, 1),
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


def focal_bce_loss(pred, target, gamma=2.0, pos_weight=10.0):
    """Focal BCE loss for imbalanced boundary detection."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)  # Probability of correct class
    focal_weight = (1 - pt) ** gamma

    # Apply pos_weight to positive samples
    weight = torch.where(target > 0.5, pos_weight, 1.0)

    return (focal_weight * weight * bce).mean()


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_seg_loss = 0
    total_boundary_loss = 0
    total_dice = [0, 0, 0]
    total_boundary_dice = 0
    n_batches = 0

    class_weights = torch.tensor([2.0, 4.0, 1.0], device=device)

    # Increase boundary weight over epochs
    boundary_weight = min(3.0 + epoch * 0.2, 5.0)

    for batch in tqdm(loader, desc=f'Training (bw={boundary_weight:.1f})'):
        image = batch['image'].to(device)
        mask_3class = batch['mask_3class'].to(device)
        is_os_boundary = batch['is_os_boundary'].to(device)

        optimizer.zero_grad()

        seg_logits, boundary_logits = model(image)

        # Segmentation loss
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

        # Boundary loss - focal BCE + Dice
        boundary_focal = focal_bce_loss(boundary_logits, is_os_boundary, gamma=2.0, pos_weight=15.0)

        boundary_probs = torch.sigmoid(boundary_logits)
        b_intersection = (boundary_probs * is_os_boundary).sum()
        b_union = boundary_probs.sum() + is_os_boundary.sum() + 1e-6
        boundary_dice_loss = 1 - (2 * b_intersection / b_union)

        boundary_loss = boundary_focal + boundary_dice_loss

        # Total loss
        loss = seg_loss + boundary_weight * boundary_loss

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    print("BOUNDARY DETECTION V3: Focused Boundary")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Train: {MAX_TRAIN_SAMPLES}, Val: {MAX_VAL_SAMPLES}")
    print(f"Boundary dilation: {BOUNDARY_DILATION}px")
    print(f"LR: {LR}, Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    # Data - training with dilation, validation without
    train_ds = BoundaryDatasetV3(
        'combined_train.jsonl', MAX_TRAIN_SAMPLES, PATCH_SIZE,
        boundary_dilation=BOUNDARY_DILATION
    )
    val_ds = BoundaryDatasetV3(
        'combined_val.jsonl', MAX_VAL_SAMPLES, PATCH_SIZE,
        boundary_dilation=0  # No dilation for validation
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = SimpleBoundaryNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_boundary_dice = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
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
            torch.save(model.state_dict(), 'best_boundary_model_v3.pth')

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best IS_OS Boundary Dice: {best_boundary_dice:.4f}")
    print(f"Final 3-class Dice: {[f'{d:.4f}' for d in val_metrics['dice']]}")

    return best_boundary_dice, val_metrics['dice']


if __name__ == '__main__':
    main()
