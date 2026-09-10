#!/usr/bin/env python3
"""
Train Layer Segmentation Model for OCT Images

Pre-trains the layer segmentation module that will be used in
multi-task denoising training.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.layer_segmentation import LightweightLayerSegmenter


class OCTSegmentationDataset(Dataset):
    """Dataset for layer segmentation training."""

    def __init__(self, jsonl_path, patch_size=128, max_samples=None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        # Load image (use noisy for robustness)
        image = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0

        # Load segmentation mask
        seg_mask = np.load(data['seg_path'])

        # Random crop
        h, w = image.shape
        if h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            image = image[top:top+self.patch_size, left:left+self.patch_size]
            seg_mask = seg_mask[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Center crop if smaller
            top, left = (h - self.patch_size) // 2, (w - self.patch_size) // 2
            image = image[max(0,top):max(0,top)+self.patch_size, max(0,left):max(0,left)+self.patch_size]
            seg_mask = seg_mask[max(0,top):max(0,top)+self.patch_size, max(0,left):max(0,left)+self.patch_size]

        # Data augmentation
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            seg_mask = np.fliplr(seg_mask).copy()

        return {
            'image': torch.from_numpy(image).unsqueeze(0).float(),
            'seg_mask': torch.from_numpy(seg_mask).long(),
        }


def compute_dice_score(pred, target, num_classes):
    """Compute Dice score for segmentation."""
    dice_scores = []

    pred = pred.argmax(dim=1)  # [B, H, W]

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())

    return np.mean(dice_scores)


def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_dice = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        image = batch['image'].to(device)
        seg_mask = batch['seg_mask'].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        pred_logits = model(image)

        # Loss: Cross-entropy
        loss = F.cross_entropy(pred_logits, seg_mask)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        dice = compute_dice_score(pred_logits.detach(), seg_mask, num_classes=5)

        total_loss += loss.item()
        total_dice += dice
        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'Dice': f'{total_dice/num_batches:.4f}'
        })

    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
    }


def validate(model, loader, device):
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_dice = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            image = batch['image'].to(device)
            seg_mask = batch['seg_mask'].to(device)

            pred_logits = model(image)
            loss = F.cross_entropy(pred_logits, seg_mask)
            dice = compute_dice_score(pred_logits, seg_mask, num_classes=5)

            total_loss += loss.item()
            total_dice += dice
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train layer segmentation model')
    parser.add_argument('--train_jsonl', default='seg_data/seg_train.jsonl')
    parser.add_argument('--val_jsonl', default='seg_data/seg_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_train', type=int, default=1000)
    parser.add_argument('--max_val', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='models/layer_segmenter')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("TRAINING LAYER SEGMENTATION MODEL")
    print("="*70)

    # Data
    train_ds = OCTSegmentationDataset(args.train_jsonl, args.patch_size, args.max_train)
    val_ds = OCTSegmentationDataset(args.val_jsonl, args.patch_size, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Data: {len(train_ds)} train, {len(val_ds)} val")

    # Model
    model = LightweightLayerSegmenter(num_classes=5).to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_dice = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)
        val_metrics = validate(model, val_loader, args.device)
        scheduler.step()

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")

        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            print(f"*** NEW BEST: Dice={best_dice:.4f} ***")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best.pth'))

        # Memory cleanup
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved to: {args.output_dir}/best.pth")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
