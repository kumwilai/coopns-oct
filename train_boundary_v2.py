#!/usr/bin/env python3
"""
Boundary Detection V2: Column-wise IS_OS Position Prediction

Key improvements:
1. Full-image validation (no random crops) for consistent evaluation
2. Column-wise regression: predict IS_OS y-position per column
3. Combination: 3-class segmentation + column-wise IS_OS position
4. Dilated boundary for pixel-wise training to give more signal
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
NUM_CLASSES = 3  # RNFL_GCL, INL_OPL_ONL, RPE_Choroid
CLASS_NAMES = ['RNFL_GCL', 'INL_OPL_ONL', 'RPE_Choroid']
MAX_TRAIN_SAMPLES = 200  # Start small for fast iteration
MAX_VAL_SAMPLES = 50
PATCH_SIZE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 12
LR = 5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BOUNDARY_DILATION = 3  # Dilate boundary for training (more signal)


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


def get_is_os_column_positions(mask_5class):
    """
    Get IS_OS y-position for each column.
    Returns array of shape (width,) with y-positions.
    -1 indicates no IS_OS in that column.
    """
    H, W = mask_5class.shape
    positions = np.full(W, -1, dtype=np.float32)

    for x in range(W):
        col = mask_5class[:, x]
        is_os_pixels = np.where(col == 3)[0]
        if len(is_os_pixels) > 0:
            # Use mean position if multiple IS_OS pixels in column
            positions[x] = is_os_pixels.mean()

    return positions


class BoundaryDatasetV2(Dataset):
    """Dataset with column-wise boundary positions."""

    def __init__(self, jsonl_path, max_samples=100, patch_size=128,
                 is_validation=False, boundary_dilation=0):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f][:max_samples]
        self.patch_size = patch_size
        self.is_validation = is_validation
        self.boundary_dilation = boundary_dilation

    def __len__(self):
        return len(self.samples)

    def _center_crop(self, image, mask, size):
        """Center crop for validation (deterministic)."""
        H, W = image.shape
        top = max(0, (H - size) // 2)
        left = max(0, (W - size) // 2)

        image = image[top:top+size, left:left+size]
        mask = mask[top:top+size, left:left+size]

        # Pad if needed
        if image.shape[0] < size or image.shape[1] < size:
            pad_h = size - image.shape[0]
            pad_w = size - image.shape[1]
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')

        return image, mask

    def _random_crop(self, image, mask, size):
        """Random crop for training."""
        H, W = image.shape
        top = np.random.randint(0, max(1, H - size))
        left = np.random.randint(0, max(1, W - size))

        image = image[top:top+size, left:left+size]
        mask = mask[top:top+size, left:left+size]

        # Pad if needed
        if image.shape[0] < size or image.shape[1] < size:
            pad_h = size - image.shape[0]
            pad_w = size - image.shape[1]
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')

        return image, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        # Crop
        if self.is_validation:
            image, mask_5class = self._center_crop(image, mask_5class, self.patch_size)
        else:
            image, mask_5class = self._random_crop(image, mask_5class, self.patch_size)

        # Create targets
        mask_3class = remap_mask_to_3class(mask_5class)
        is_os_boundary = extract_is_os_boundary(mask_5class, self.boundary_dilation if not self.is_validation else 0)
        col_positions = get_is_os_column_positions(mask_5class)

        # Valid mask for column regression (only columns with IS_OS)
        col_valid = (col_positions >= 0).astype(np.float32)
        # Normalize positions to [0, 1] range
        col_positions_norm = np.clip(col_positions / self.patch_size, 0, 1)
        col_positions_norm[col_positions < 0] = 0.5  # Default for invalid columns

        return {
            'image': torch.from_numpy(image).float().unsqueeze(0),
            'mask_3class': torch.from_numpy(mask_3class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
            'col_positions': torch.from_numpy(col_positions_norm).float(),
            'col_valid': torch.from_numpy(col_valid).float(),
        }


class BoundarySegmenterV2(nn.Module):
    """
    U-Net with three heads:
    1. 3-class segmentation
    2. IS_OS boundary (pixel-wise)
    3. IS_OS column position (column-wise regression)
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

        # Head 1: Segmentation (3 classes)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

        # Head 2: Boundary detection (pixel-wise)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

        # Head 3: Column-wise position regression
        # Uses global average pooling along height to get per-column features
        self.col_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # Output: (B, 1, H, W)
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

        # Column position: use weighted average of y-positions
        # Each column outputs a position prediction
        col_weights = torch.sigmoid(self.col_head(d1))  # (B, 1, H, W)

        B, _, H, W = col_weights.shape
        y_coords = torch.arange(H, device=x.device).float() / H  # [0, 1]
        y_coords = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)

        # Weighted average of y-positions per column
        col_positions = (col_weights * y_coords).sum(dim=2) / (col_weights.sum(dim=2) + 1e-6)
        col_positions = col_positions.squeeze(1)  # (B, W)

        return seg_logits, boundary_logits, col_positions


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


def compute_col_position_error(pred_pos, target_pos, valid_mask):
    """Compute mean absolute error for column positions."""
    valid_count = valid_mask.sum()
    if valid_count > 0:
        errors = torch.abs(pred_pos - target_pos) * valid_mask
        mae = errors.sum() / valid_count
        return mae.item()
    return 0.0


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_seg_loss = 0
    total_boundary_loss = 0
    total_col_loss = 0
    total_dice = [0, 0, 0]
    total_boundary_dice = 0
    total_col_error = 0
    n_batches = 0

    class_weights = torch.tensor([2.0, 4.0, 1.0], device=device)

    for batch in tqdm(loader, desc='Training'):
        image = batch['image'].to(device)
        mask_3class = batch['mask_3class'].to(device)
        is_os_boundary = batch['is_os_boundary'].to(device)
        col_positions = batch['col_positions'].to(device)
        col_valid = batch['col_valid'].to(device)

        optimizer.zero_grad()

        seg_logits, boundary_logits, pred_col_pos = model(image)

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

        # Boundary loss with focal-like weighting
        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_logits, is_os_boundary,
            pos_weight=torch.tensor([15.0], device=device)
        )

        # Boundary Dice
        boundary_probs = torch.sigmoid(boundary_logits)
        b_intersection = (boundary_probs * is_os_boundary).sum()
        b_union = boundary_probs.sum() + is_os_boundary.sum() + 1e-6
        boundary_dice_loss = 1 - (2 * b_intersection / b_union)

        boundary_loss = boundary_bce + boundary_dice_loss

        # Column position loss (smooth L1)
        col_diff = F.smooth_l1_loss(pred_col_pos, col_positions, reduction='none')
        col_loss = (col_diff * col_valid).sum() / (col_valid.sum() + 1e-6)

        # Total loss
        loss = seg_loss + 2.0 * boundary_loss + 1.0 * col_loss

        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            dice_scores = compute_dice(seg_logits, mask_3class, NUM_CLASSES)
            b_dice = compute_boundary_dice(boundary_logits, is_os_boundary)
            col_err = compute_col_position_error(pred_col_pos, col_positions, col_valid)

        total_seg_loss += seg_loss.item()
        total_boundary_loss += boundary_loss.item()
        total_col_loss += col_loss.item()
        for i in range(NUM_CLASSES):
            total_dice[i] += dice_scores[i]
        total_boundary_dice += b_dice
        total_col_error += col_err
        n_batches += 1

    return {
        'seg_loss': total_seg_loss / n_batches,
        'boundary_loss': total_boundary_loss / n_batches,
        'col_loss': total_col_loss / n_batches,
        'dice': [d / n_batches for d in total_dice],
        'boundary_dice': total_boundary_dice / n_batches,
        'col_error': total_col_error / n_batches,
    }


def validate(model, loader, device):
    model.eval()
    total_dice = [0, 0, 0]
    total_boundary_dice = 0
    total_col_error = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            image = batch['image'].to(device)
            mask_3class = batch['mask_3class'].to(device)
            is_os_boundary = batch['is_os_boundary'].to(device)
            col_positions = batch['col_positions'].to(device)
            col_valid = batch['col_valid'].to(device)

            seg_logits, boundary_logits, pred_col_pos = model(image)

            dice_scores = compute_dice(seg_logits, mask_3class, NUM_CLASSES)
            b_dice = compute_boundary_dice(boundary_logits, is_os_boundary)
            col_err = compute_col_position_error(pred_col_pos, col_positions, col_valid)

            for i in range(NUM_CLASSES):
                total_dice[i] += dice_scores[i]
            total_boundary_dice += b_dice
            total_col_error += col_err
            n_batches += 1

    return {
        'dice': [d / n_batches for d in total_dice],
        'boundary_dice': total_boundary_dice / n_batches,
        'col_error': total_col_error / n_batches,
    }


def main():
    print("=" * 60)
    print("BOUNDARY DETECTION V2: Column-wise Position")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Train: {MAX_TRAIN_SAMPLES}, Val: {MAX_VAL_SAMPLES}")
    print(f"Boundary dilation: {BOUNDARY_DILATION} (training only)")
    print(f"Validation: Center crop (deterministic)")
    print("=" * 60)

    # Data
    train_ds = BoundaryDatasetV2(
        'combined_train.jsonl', MAX_TRAIN_SAMPLES, PATCH_SIZE,
        is_validation=False, boundary_dilation=BOUNDARY_DILATION
    )
    val_ds = BoundaryDatasetV2(
        'combined_val.jsonl', MAX_VAL_SAMPLES, PATCH_SIZE,
        is_validation=True, boundary_dilation=0
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = BoundarySegmenterV2(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_boundary_dice = 0
    best_col_error = float('inf')

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
        print(f"  Col Loss: {train_metrics['col_loss']:.4f}")
        print(f"  Dice: {[f'{d:.4f}' for d in train_metrics['dice']]}")
        print(f"  IS_OS Boundary Dice: {train_metrics['boundary_dice']:.4f}")
        print(f"  Col Position Error: {train_metrics['col_error']:.4f}")

        print(f"\nValidation:")
        print(f"  Per-class Dice:")
        for i, name in enumerate(CLASS_NAMES):
            dice = val_metrics['dice'][i]
            status = "OK" if dice > 0.3 else "LEARNING" if dice > 0.1 else "COLLAPSED"
            print(f"    {name:15s}: {dice:.4f} [{status}]")
        print(f"  IS_OS Boundary Dice: {val_metrics['boundary_dice']:.4f}")
        print(f"  Col Position Error: {val_metrics['col_error']:.4f} (lower is better)")

        # Track best
        if val_metrics['boundary_dice'] > best_boundary_dice:
            best_boundary_dice = val_metrics['boundary_dice']
            print(f"  *** NEW BEST IS_OS Boundary: {best_boundary_dice:.4f} ***")
            torch.save(model.state_dict(), 'best_boundary_model_v2.pth')

        if val_metrics['col_error'] < best_col_error:
            best_col_error = val_metrics['col_error']
            print(f"  *** NEW BEST Col Error: {best_col_error:.4f} ***")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best IS_OS Boundary Dice: {best_boundary_dice:.4f}")
    print(f"Best Col Position Error: {best_col_error:.4f}")
    print(f"Final 3-class Dice: {[f'{d:.4f}' for d in val_metrics['dice']]}")

    return best_boundary_dice, val_metrics['dice']


if __name__ == '__main__':
    main()
