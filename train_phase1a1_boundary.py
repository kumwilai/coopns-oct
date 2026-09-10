#!/usr/bin/env python3
"""
Phase 1A-1: Segmentation-Only Training with IS_OS Boundary Detection

Uses the proven V4 architecture (SimpleBoundarySegmenter) that achieved 0.48+ IS_OS Dice.
Starts from the pre-trained V4 checkpoint for faster convergence.

Key features:
- 3-class segmentation: RNFL_GCL, INL_OPL_ONL, RPE_Choroid
- IS_OS detected as boundary, not region class
- Full-image validation with sliding window
- Pre-trained initialization from V4 checkpoint
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# ============================================================================
# Configuration
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


class TrainDataset(Dataset):
    """Training dataset with random patches."""

    def __init__(self, jsonl_path, max_samples=None, patch_size=128):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        H, W = image.shape
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        image = image[top:top+self.patch_size, left:left+self.patch_size]
        mask_5class = mask_5class[top:top+self.patch_size, left:left+self.patch_size]

        if image.shape[0] < self.patch_size or image.shape[1] < self.patch_size:
            pad_h = self.patch_size - image.shape[0]
            pad_w = self.patch_size - image.shape[1]
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_5class = np.pad(mask_5class, ((0, pad_h), (0, pad_w)), mode='reflect')

        mask_3class = remap_mask_to_3class(mask_5class)
        is_os_boundary = extract_is_os_boundary(mask_5class)

        return {
            'image': torch.from_numpy(image).float().unsqueeze(0),
            'mask_3class': torch.from_numpy(mask_3class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
        }


class ValDataset(Dataset):
    """Validation dataset with full images."""

    def __init__(self, jsonl_path, max_samples=None):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask_5class = np.array(Image.open(sample['mask_path']))

        mask_3class = remap_mask_to_3class(mask_5class)
        is_os_boundary = extract_is_os_boundary(mask_5class)

        return {
            'image': torch.from_numpy(image).float().unsqueeze(0),
            'mask_3class': torch.from_numpy(mask_3class).long(),
            'is_os_boundary': torch.from_numpy(is_os_boundary).float().unsqueeze(0),
        }


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


def sliding_window_inference(model, image, patch_size=64, stride=32, device='cpu'):
    """Perform inference on full image using sliding window with overlap."""
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

    seg_output = torch.zeros(B, 3, H_pad, W_pad, device=device)
    boundary_output = torch.zeros(B, 1, H_pad, W_pad, device=device)
    count = torch.zeros(B, 1, H_pad, W_pad, device=device)

    with torch.no_grad():
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                seg_logits, boundary_logits = model(patch)

                seg_output[:, :, y:y+patch_size, x:x+patch_size] += seg_logits
                boundary_output[:, :, y:y+patch_size, x:x+patch_size] += boundary_logits
                count[:, :, y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping regions
    seg_output = seg_output / count.clamp(min=1)
    boundary_output = boundary_output / count.clamp(min=1)

    # Crop back to original size and free memory
    result = (seg_output[:, :, :H, :W].clone(),
              boundary_output[:, :, :H, :W].clone())

    del seg_output, boundary_output, count
    return result


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


def train_epoch(model, loader, optimizer, device, class_weights, boundary_weight=2.0, pos_weight=10.0):
    model.train()
    total_seg_loss = 0
    total_boundary_loss = 0
    total_dice = [0, 0, 0]
    total_boundary_dice = 0
    n_batches = 0

    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    for batch in tqdm(loader, desc='Training'):
        image = batch['image'].to(device)
        mask_3class = batch['mask_3class'].to(device)
        is_os_boundary = batch['is_os_boundary'].to(device)

        optimizer.zero_grad()

        seg_logits, boundary_logits = model(image)

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
            boundary_logits, is_os_boundary,
            pos_weight=pos_weight_tensor
        )

        boundary_probs = torch.sigmoid(boundary_logits)
        b_intersection = (boundary_probs * is_os_boundary).sum()
        b_union = boundary_probs.sum() + is_os_boundary.sum() + 1e-6
        boundary_dice_loss = 1 - (2 * b_intersection / b_union)

        boundary_loss = boundary_bce + boundary_dice_loss

        # Total loss
        loss = seg_loss + boundary_weight * boundary_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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


def validate_full_image(model, val_ds, device, patch_size=64, stride=32):
    """Validate on full images with sliding window inference."""
    model.eval()
    total_dice = [0, 0, 0]
    total_boundary_dice = 0
    n_samples = 0

    with torch.no_grad():
        for i in tqdm(range(len(val_ds)), desc='Validation'):
            sample = val_ds[i]

            image = sample['image'].unsqueeze(0).to(device)
            mask_3class = sample['mask_3class'].unsqueeze(0).to(device)
            is_os_boundary = sample['is_os_boundary'].unsqueeze(0).to(device)

            seg_logits, boundary_logits = sliding_window_inference(
                model, image, patch_size, stride, device
            )

            dice_scores = compute_dice(seg_logits, mask_3class, NUM_CLASSES)
            b_dice = compute_boundary_dice(boundary_logits, is_os_boundary)

            for c in range(NUM_CLASSES):
                total_dice[c] += dice_scores[c]
            total_boundary_dice += b_dice
            n_samples += 1

            # Free memory after each sample
            del image, mask_3class, is_os_boundary
            del seg_logits, boundary_logits

    # Clear GPU cache if using CUDA
    if device == 'cuda' or (isinstance(device, str) and 'cuda' in device):
        torch.cuda.empty_cache()

    return {
        'dice': [d / n_samples for d in total_dice],
        'boundary_dice': total_boundary_dice / n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 1A-1: Segmentation with IS_OS Boundary Detection')
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--checkpoint', default='best_boundary_model_v4.pth',
                        help='Initial checkpoint to load')
    parser.add_argument('--output_dir', default='outputs/phase1a1_boundary')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--finetune_lr_factor', type=float, default=0.1,
                        help='When loading checkpoint, multiply LR by this factor (default: 0.1 = 10x lower)')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Warmup epochs when fine-tuning (gradually increase LR)')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch size for training and validation (64 for memory efficiency)')
    parser.add_argument('--val_stride', type=int, default=32,
                        help='Stride for validation sliding window (typically patch_size/2)')
    parser.add_argument('--boundary_weight', type=float, default=2.0)
    parser.add_argument('--pos_weight', type=float, default=10.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 1A-1: SEGMENTATION WITH IS_OS BOUNDARY DETECTION")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Initial checkpoint: {args.checkpoint}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Boundary weight: {args.boundary_weight}, Pos weight: {args.pos_weight}")
    print("=" * 70)

    # Data
    train_ds = TrainDataset(args.train_jsonl, args.max_train, args.patch_size)
    val_ds = ValDataset(args.val_jsonl, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"Val: {len(val_ds)} samples (full images)")

    # Model
    model = SimpleBoundarySegmenter(num_classes=NUM_CLASSES).to(args.device)

    # Load checkpoint
    loaded_from_checkpoint = False
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint)
        loaded_from_checkpoint = True
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded, training from scratch")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for imbalanced classes
    class_weights = torch.tensor([2.0, 4.0, 1.0], device=args.device)

    # Use lower learning rate when fine-tuning from checkpoint
    # This prevents catastrophic forgetting of learned features
    if loaded_from_checkpoint:
        effective_lr = args.lr * args.finetune_lr_factor
        print(f"Fine-tuning mode: LR reduced from {args.lr} to {effective_lr}")
        print(f"  (Using {args.warmup_epochs} warmup epochs to gradually increase LR)")
    else:
        effective_lr = args.lr

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    # Use ReduceLROnPlateau instead of CosineAnnealing for more stable fine-tuning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Initial validation
    print("\nInitial validation (from checkpoint)...")
    val_metrics = validate_full_image(model, val_ds, args.device, args.patch_size, args.val_stride)
    print(f"  3-class Dice: {[f'{d:.4f}' for d in val_metrics['dice']]}")
    print(f"  IS_OS Boundary Dice: {val_metrics['boundary_dice']:.4f}")

    best_boundary_dice = val_metrics['boundary_dice']
    best_mean_dice = np.mean(val_metrics['dice'])
    initial_mean_dice = best_mean_dice  # Track for early stopping
    patience_counter = 0
    max_patience = 5  # Stop if no improvement for 5 epochs

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        # Warmup: gradually increase learning rate during first few epochs
        if loaded_from_checkpoint and epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = effective_lr * warmup_factor
            print(f"  Warmup: LR = {effective_lr * warmup_factor:.2e}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            class_weights, args.boundary_weight, args.pos_weight
        )
        val_metrics = validate_full_image(model, val_ds, args.device, args.patch_size, args.val_stride)

        # Update scheduler with validation metric
        mean_dice = np.mean(val_metrics['dice'])
        scheduler.step(mean_dice)

        # Print results
        print(f"\nTraining:")
        print(f"  Seg Loss: {train_metrics['seg_loss']:.4f}")
        print(f"  Boundary Loss: {train_metrics['boundary_loss']:.4f}")
        print(f"  Dice: {[f'{d:.4f}' for d in train_metrics['dice']]}")
        print(f"  IS_OS Boundary Dice: {train_metrics['boundary_dice']:.4f}")

        print(f"\nValidation (full images):")
        print(f"  Per-class Dice:")
        for i, name in enumerate(CLASS_NAMES):
            dice = val_metrics['dice'][i]
            status = "OK" if dice > 0.5 else "LEARNING" if dice > 0.3 else "WEAK"
            print(f"    {name:15s}: {dice:.4f} [{status}]")

        is_os_dice = val_metrics['boundary_dice']
        status = "EXCELLENT" if is_os_dice > 0.5 else "GOOD" if is_os_dice > 0.4 else "LEARNING"
        print(f"  IS_OS Boundary Dice: {is_os_dice:.4f} [{status}]")
        print(f"  Mean Dice: {mean_dice:.4f}")

        # Early stopping check - detect catastrophic forgetting
        if mean_dice < initial_mean_dice * 0.7:  # More than 30% drop
            print(f"  WARNING: Mean Dice dropped to {mean_dice:.4f} from initial {initial_mean_dice:.4f}")
            print(f"           This indicates catastrophic forgetting!")

        # Save best models
        improved = False
        if val_metrics['boundary_dice'] > best_boundary_dice:
            best_boundary_dice = val_metrics['boundary_dice']
            torch.save(model.state_dict(), f"{args.output_dir}/best_boundary.pth")
            print(f"  *** NEW BEST IS_OS Boundary: {best_boundary_dice:.4f} ***")
            improved = True

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(model.state_dict(), f"{args.output_dir}/best_dice.pth")
            print(f"  *** NEW BEST Mean Dice: {best_mean_dice:.4f} ***")
            improved = True
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n  Early stopping: No improvement for {max_patience} epochs")
                break

        # Save latest
        torch.save(model.state_dict(), f"{args.output_dir}/latest.pth")

    print(f"\n{'='*70}")
    print(f"PHASE 1A-1 COMPLETE")
    print(f"{'='*70}")
    print(f"Best IS_OS Boundary Dice: {best_boundary_dice:.4f}")
    print(f"Best Mean Dice: {best_mean_dice:.4f}")
    print(f"Models saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
