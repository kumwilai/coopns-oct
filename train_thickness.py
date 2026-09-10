#!/usr/bin/env python3
"""
Training script for Physics DSP with Coupled Thickness Prediction.

This model predicts ILM + layer thicknesses instead of independent boundaries,
which is better for thin layers like INL (~7.7px).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from physics_dsp_thickness import (
    PhysicsDSPThickness,
    ThicknessLoss,
    boundaries_to_segmentation,
)


# =============================================================================
# Dataset
# =============================================================================
class OCTBoundaryDataset(Dataset):
    """Dataset for OCT boundary detection."""

    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        target_size: tuple = (256, 256),
        augment: bool = False,
    ):
        self.samples = []
        self.target_size = target_size
        self.augment = augment

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(entry['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(entry['mask_path']))

        H_orig, W_orig = image.shape
        H_target, W_target = self.target_size

        # Extract boundaries BEFORE resizing
        boundaries, valid_mask = self._extract_boundaries(mask)

        # Resize
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        image_resized = np.array(image_pil.resize((W_target, H_target), Image.BILINEAR)) / 255.0
        mask_resized = np.array(mask_pil.resize((W_target, H_target), Image.NEAREST))

        # Resize boundaries
        if W_orig != W_target:
            x_orig = np.linspace(0, 1, W_orig)
            x_new = np.linspace(0, 1, W_target)
            boundaries_new = np.zeros((4, W_target))
            valid_mask_new = np.zeros(W_target)

            for b in range(4):
                boundaries_new[b] = np.interp(x_new, x_orig, boundaries[b])

            valid_mask_new = np.interp(x_new, x_orig, valid_mask.astype(float)) > 0.5
            boundaries = boundaries_new
            valid_mask = valid_mask_new

        # Convert to tensors
        image_t = torch.from_numpy(image_resized).float().unsqueeze(0)
        boundaries_t = torch.from_numpy(boundaries).float()
        valid_mask_t = torch.from_numpy(valid_mask.astype(np.float32))
        mask_t = torch.from_numpy(mask_resized).long()

        return {
            'image': image_t,
            'boundaries': boundaries_t,
            'valid_mask': valid_mask_t,
            'mask': mask_t,
        }

    def _extract_boundaries(self, mask):
        """Extract boundary positions from segmentation mask."""
        H, W = mask.shape
        boundaries = np.zeros((4, W))
        valid_mask = np.ones(W, dtype=bool)

        for col in range(W):
            col_data = mask[:, col]

            for orig_class in [1, 2, 3, 4]:
                rows = np.where(col_data == orig_class)[0]
                if len(rows) > 0:
                    boundaries[orig_class - 1, col] = rows[0] / (H - 1)
                else:
                    valid_mask[col] = False
                    boundaries[orig_class - 1, col] = (orig_class * 0.1 + 0.2)

        return boundaries, valid_mask


# =============================================================================
# Dice Score Computation
# =============================================================================
def compute_dice_scores(pred_boundaries, gt_mask, H):
    """Compute Dice scores for each layer."""
    pred_seg = boundaries_to_segmentation(pred_boundaries, H, num_classes=4)

    dice_scores = {}
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    for c, name in enumerate(layer_names):
        pred_c = (pred_seg == c).float()
        gt_c = (gt_mask == (c + 1)).float()  # GT uses 1-indexed classes

        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()

        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        dice_scores[name] = dice.item()

    dice_scores['avg'] = np.mean(list(dice_scores.values()))
    return dice_scores


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, loss_fn, loader, optimizer, device, H):
    model.train()
    total_loss = 0
    total_mae = 0
    total_thick_mae = {'RNFL': 0, 'INL': 0, 'ISOS': 0}

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        gt_bounds = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(images, return_aux=True)
        pred_bounds = outputs['boundaries']

        # Get predicted thicknesses
        pred_thick = {
            'rnfl_thick': outputs['rnfl_thick'],
            'inl_thick': outputs['inl_thick'],
            'isos_thick': outputs['isos_thick'],
        }

        loss, stats = loss_fn(pred_bounds, gt_bounds, pred_thick, valid_mask, H)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mae += stats['avg_mae']
        total_thick_mae['RNFL'] += stats['RNFL_thick_mae']
        total_thick_mae['INL'] += stats['INL_thick_mae']
        total_thick_mae['ISOS'] += stats['ISOS_thick_mae']

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'MAE': f"{stats['avg_mae']:.1f}px",
            'INL_th': f"{stats['INL_thick_mae']:.1f}px",
        })

    n = len(loader)
    return {
        'loss': total_loss / n,
        'mae': total_mae / n,
        'rnfl_thick_mae': total_thick_mae['RNFL'] / n,
        'inl_thick_mae': total_thick_mae['INL'] / n,
        'isos_thick_mae': total_thick_mae['ISOS'] / n,
    }


def validate(model, loss_fn, loader, device, H):
    model.eval()
    total_mae = 0
    total_thick_mae = {'RNFL': 0, 'INL': 0, 'ISOS': 0}
    all_dice = []
    boundary_maes = {f'b{i}': 0 for i in range(4)}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            gt_bounds = batch['boundaries'].to(device)
            gt_mask = batch['mask'].to(device)
            valid_mask = batch['valid_mask'].to(device)

            outputs = model(images, return_aux=True)
            pred_bounds = outputs['boundaries']

            pred_thick = {
                'rnfl_thick': outputs['rnfl_thick'],
                'inl_thick': outputs['inl_thick'],
                'isos_thick': outputs['isos_thick'],
            }

            _, stats = loss_fn(pred_bounds, gt_bounds, pred_thick, valid_mask, H)

            total_mae += stats['avg_mae']
            total_thick_mae['RNFL'] += stats['RNFL_thick_mae']
            total_thick_mae['INL'] += stats['INL_thick_mae']
            total_thick_mae['ISOS'] += stats['ISOS_thick_mae']

            boundary_maes['b0'] += stats['ILM_mae']
            boundary_maes['b1'] += stats['RNFL_INL_mae']
            boundary_maes['b2'] += stats['INL_ISOS_mae']
            boundary_maes['b3'] += stats['ISOS_RPE_mae']

            # Compute Dice
            dice = compute_dice_scores(pred_bounds, gt_mask, H)
            all_dice.append(dice)

    n = len(loader)

    # Average Dice scores
    avg_dice = {}
    for key in all_dice[0].keys():
        avg_dice[key] = np.mean([d[key] for d in all_dice])

    return {
        'mae': total_mae / n,
        'rnfl_thick_mae': total_thick_mae['RNFL'] / n,
        'inl_thick_mae': total_thick_mae['INL'] / n,
        'isos_thick_mae': total_thick_mae['ISOS'] / n,
        'boundary_maes': {k: v / n for k, v in boundary_maes.items()},
        'dice': avg_dice,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)

    # Model
    parser.add_argument('--hidden_channels', type=int, default=48)
    parser.add_argument('--use_fresnel', action='store_true', default=True)
    parser.add_argument('--use_depth_comp', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', default='cpu')

    # Output
    parser.add_argument('--output_dir', default='outputs/thickness_model')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    H = 256

    print("=" * 60)
    print("Physics DSP with Coupled Thickness Prediction")
    print("=" * 60)

    # Data
    train_ds = OCTBoundaryDataset(args.train_jsonl, args.max_train)
    val_ds = OCTBoundaryDataset(args.val_jsonl, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = PhysicsDSPThickness(
        hidden_channels=args.hidden_channels,
        use_fresnel=args.use_fresnel,
        use_depth_comp=args.use_depth_comp,
    ).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss
    loss_fn = ThicknessLoss(learnable_weights=True).to(device)
    print(f"Loss function: {sum(p.numel() for p in loss_fn.parameters())} learnable weights")

    # Optimizer (include loss function parameters)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_mae = float('inf')

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_stats = train_epoch(model, loss_fn, train_loader, optimizer, device, H)

        # Validate
        val_stats = validate(model, loss_fn, val_loader, device, H)

        scheduler.step()

        # Print results
        print(f"\nEpoch {epoch}: Train MAE={train_stats['mae']:.2f}px, Val MAE={val_stats['mae']:.2f}px")

        # Boundary MAEs
        bm = val_stats['boundary_maes']
        print(f"  Boundary MAE: ILM={bm['b0']:.2f}px, RNFL_INL={bm['b1']:.2f}px, "
              f"INL_ISOS={bm['b2']:.2f}px, ISOS_RPE={bm['b3']:.2f}px")

        # Thickness MAEs (key metric for thin layers)
        print(f"  Thickness MAE: RNFL={val_stats['rnfl_thick_mae']:.2f}px, "
              f"INL={val_stats['inl_thick_mae']:.2f}px (thin!), "
              f"ISOS={val_stats['isos_thick_mae']:.2f}px")

        # Dice scores
        d = val_stats['dice']
        print(f"  Val Dice: avg={d['avg']:.4f}")
        print(f"    RNFL_GCL: {d['RNFL_GCL']:.4f}, INL_OPL_ONL: {d['INL_OPL_ONL']:.4f}, "
              f"IS_OS: {d['IS_OS']:.4f}, RPE_Choroid: {d['RPE_Choroid']:.4f}")

        # Learned weights
        weights = loss_fn.get_learned_weights()
        bw = weights['boundary_weights']
        tw = weights['thickness_weights']
        print(f"  Learned boundary weights: [{bw[0]:.2f}, {bw[1]:.2f}, {bw[2]:.2f}, {bw[3]:.2f}]")
        print(f"  Learned thickness weights: [{tw[0]:.2f}, {tw[1]:.2f}(INL!), {tw[2]:.2f}]")

        # Save best
        if val_stats['mae'] < best_mae:
            best_mae = val_stats['mae']
            torch.save({
                'model': model.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'epoch': epoch,
                'mae': best_mae,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> New best! MAE={best_mae:.2f}px")

    # Save final
    torch.save({
        'model': model.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'epoch': args.epochs,
    }, os.path.join(args.output_dir, 'final_model.pth'))

    print("\n" + "=" * 60)
    print(f"Training complete! Best MAE: {best_mae:.2f}px")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
