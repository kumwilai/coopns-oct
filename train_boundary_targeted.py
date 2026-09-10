#!/usr/bin/env python3
"""
Targeted boundary training to achieve specific MAE targets.

This script fine-tunes from a pre-trained model with:
1. Higher loss weights for underperforming boundaries
2. Boundary-specific attention mechanism
3. Adaptive loss weighting based on target gaps

Targets:
- ILM (b0): <5px ✅ (already achieved)
- RNFL_INL (b1): <4px
- INL_ISOS (b2): <3px
- ISOS_RPE (b3): <3px
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from physics_dsp_ensemble_v2 import (
    PhysicsEnsemble,
    PhysicsEnsembleLoss,
    boundaries_to_segmentation,
    BoundaryGradientModule,
)


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'training.log')
    root_logger = logging.getLogger()
    root_logger.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# =============================================================================
# Dataset (same as before)
# =============================================================================
class OCTBoundaryDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None, target_size=(256, 256)):
        self.samples = []
        self.target_size = target_size
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
        image = np.array(Image.open(entry['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(entry['mask_path']))

        H_orig, W_orig = image.shape
        H_target, W_target = self.target_size

        boundaries, valid_mask = self._extract_boundaries(mask)

        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        image_resized = np.array(image_pil.resize((W_target, H_target), Image.BILINEAR)) / 255.0
        mask_resized = np.array(mask_pil.resize((W_target, H_target), Image.NEAREST))

        if W_orig != W_target:
            x_orig = np.linspace(0, 1, W_orig)
            x_new = np.linspace(0, 1, W_target)
            boundaries_new = np.zeros((4, W_target))
            for b in range(4):
                boundaries_new[b] = np.interp(x_new, x_orig, boundaries[b])
            valid_mask_new = np.interp(x_new, x_orig, valid_mask.astype(float)) > 0.5
            boundaries = boundaries_new
            valid_mask = valid_mask_new

        return {
            'image': torch.from_numpy(image_resized).float().unsqueeze(0),
            'boundaries': torch.from_numpy(boundaries).float(),
            'valid_mask': torch.from_numpy(valid_mask.astype(np.float32)),
            'mask': torch.from_numpy(mask_resized).long(),
        }

    def _extract_boundaries(self, mask):
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
# Targeted Loss Function
# =============================================================================
class TargetedBoundaryLoss(nn.Module):
    """
    Loss function with adaptive boundary weighting based on targets.

    Gives MUCH higher weight to boundaries that haven't achieved targets.
    """

    TARGETS = {
        'b0': 5.0,   # ILM
        'b1': 4.0,   # RNFL_INL
        'b2': 3.0,   # INL_ISOS
        'b3': 3.0,   # ISOS_RPE
    }

    def __init__(self, base_loss_fn, target_scale=3.0):
        super().__init__()
        self.base_loss = base_loss_fn
        self.target_scale = target_scale

        # Fixed high weights for underperforming boundaries
        # b0 achieved, so weight=1; b1,b2,b3 need work, so higher weights
        self.register_buffer('boundary_weights', torch.tensor([1.0, 3.0, 4.0, 4.0]))

        self.gradient_module = BoundaryGradientModule()

    def forward(self, outputs, gt_bounds, valid_mask, H, image=None, physics_warmup=1.0):
        # Get base loss and stats
        base_loss, stats = self.base_loss(
            outputs, gt_bounds, valid_mask, H,
            image=image, physics_warmup=physics_warmup
        )

        # Add targeted position loss with high weights for b1, b2, b3
        pred = outputs['boundaries']
        targeted_loss = 0

        for i in range(4):
            diff = (pred[:, i, :] - gt_bounds[:, i, :]).abs()
            weighted_diff = (diff * valid_mask * self.boundary_weights[i]).sum()
            weighted_diff = weighted_diff / (valid_mask.sum() + 1e-8)
            targeted_loss = targeted_loss + weighted_diff

        # Add smoothness loss for b1, b2, b3 (reduce oscillations)
        smoothness_loss = 0
        for i in [1, 2, 3]:  # Only for underperforming boundaries
            # Penalize sharp changes between adjacent columns
            boundary_diff = (pred[:, i, 1:] - pred[:, i, :-1]).abs()
            smoothness_loss = smoothness_loss + boundary_diff.mean()

        # Add boundary-specific gradient alignment for b1, b2, b3
        grad_align_loss = 0
        if image is not None:
            gradients = self.gradient_module(image).abs()
            grad_max = gradients.max() + 1e-8
            grad_norm = gradients / grad_max

            y_coords = torch.linspace(0, 1, H, device=image.device).view(1, 1, H, 1)
            sigma = 3.0 / H  # Slightly wider for stability

            for i in [1, 2, 3]:  # Focus on underperforming boundaries
                bound_pos = pred[:, i, :].unsqueeze(1).unsqueeze(2)
                weights = torch.exp(-((y_coords - bound_pos) ** 2) / (2 * sigma ** 2))
                weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
                sampled_grad = (grad_norm * weights).sum(dim=2)
                grad_align_loss = grad_align_loss + (1.0 - sampled_grad.mean())

        # Total loss
        total_loss = (
            base_loss +
            self.target_scale * targeted_loss +
            0.1 * smoothness_loss +
            0.2 * physics_warmup * grad_align_loss
        )

        return total_loss, stats


# =============================================================================
# Metrics
# =============================================================================
def compute_dice_scores(pred_boundaries, gt_mask, H):
    pred_seg = boundaries_to_segmentation(pred_boundaries, H, num_classes=4)
    dice_scores = {}
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    for c, name in enumerate(layer_names):
        pred_c = (pred_seg == c).float()
        gt_c = (gt_mask == (c + 1)).float()
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        dice_scores[name] = dice.item()

    dice_scores['avg'] = np.mean(list(dice_scores.values()))
    return dice_scores


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, loss_fn, loader, optimizer, device, H, grad_clip=0.3):
    model.train()
    total_loss = 0
    total_mae = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        gt_bounds = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images, return_aux=True)
        loss, stats = loss_fn(outputs, gt_bounds, valid_mask, H, image=images)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_mae += stats['avg_mae']

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'MAE': f"{stats['avg_mae']:.1f}px",
        })

    n = len(loader)
    return {'loss': total_loss / n, 'mae': total_mae / n}


def validate(model, loss_fn, loader, device, H):
    model.eval()
    total_mae = 0
    all_dice = []
    boundary_maes = {f'b{i}': 0 for i in range(4)}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            gt_bounds = batch['boundaries'].to(device)
            gt_mask = batch['mask'].to(device)
            valid_mask = batch['valid_mask'].to(device)

            outputs = model(images, return_aux=True)
            _, stats = loss_fn(outputs, gt_bounds, valid_mask, H, image=images)

            total_mae += stats['avg_mae']
            boundary_maes['b0'] += stats['ILM_mae']
            boundary_maes['b1'] += stats['RNFL_INL_mae']
            boundary_maes['b2'] += stats['INL_ISOS_mae']
            boundary_maes['b3'] += stats['ISOS_RPE_mae']

            dice = compute_dice_scores(outputs['boundaries'], gt_mask, H)
            all_dice.append(dice)

    n = len(loader)
    avg_dice = {key: np.mean([d[key] for d in all_dice]) for key in all_dice[0].keys()}

    return {
        'mae': total_mae / n,
        'boundary_maes': {k: v / n for k, v in boundary_maes.items()},
        'dice': avg_dice,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--max_val', type=int, default=20)
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='outputs/boundary_targeted')
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--grad_clip', type=float, default=0.3)
    parser.add_argument('--target_scale', type=float, default=3.0,
                        help='Scale factor for targeted boundary loss')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    H = 256

    setup_logging(args.output_dir)

    logging.info("=" * 60)
    logging.info("Targeted Boundary Training")
    logging.info("=" * 60)
    logging.info("Targets:")
    logging.info("  ILM (b0): <5px (achieved)")
    logging.info("  RNFL_INL (b1): <4px")
    logging.info("  INL_ISOS (b2): <3px")
    logging.info("  ISOS_RPE (b3): <3px")
    logging.info("")
    logging.info("Strategy:")
    logging.info("  - Load pretrained model")
    logging.info("  - Higher loss weights: b0=1, b1=3, b2=4, b3=4")
    logging.info("  - Smoothness regularization for b1, b2, b3")
    logging.info("  - Boundary-specific gradient alignment")
    logging.info("  - Lower learning rate for fine-tuning")

    # Data
    train_ds = OCTBoundaryDataset(args.train_jsonl, args.max_train)
    val_ds = OCTBoundaryDataset(args.val_jsonl, args.max_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Model
    model = PhysicsEnsemble(hidden_channels=48).to(device)

    # Load pretrained weights
    logging.info(f"Loading pretrained model: {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    pretrained_mae = checkpoint.get('mae', 'unknown')
    pretrained_epoch = checkpoint.get('epoch', 'unknown')
    logging.info(f"  Pretrained from epoch {pretrained_epoch}, MAE={pretrained_mae}")

    # Loss - wrap base loss with targeted weighting
    base_loss = PhysicsEnsembleLoss(
        lambda_position=2.0,
        lambda_thickness=1.5,
        lambda_dice=1.0,
        lambda_gradient=0.3,
        lambda_intensity=0.2,
        lambda_fresnel=0.2,
    ).to(device)

    if 'loss_fn' in checkpoint:
        base_loss.load_state_dict(checkpoint['loss_fn'])

    loss_fn = TargetedBoundaryLoss(base_loss, target_scale=args.target_scale).to(device)

    # Optimizer - only model parameters, lower LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )

    logging.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    logging.info(f"Learning rate: {args.lr}")
    logging.info("=" * 60)
    logging.info("Training...")
    logging.info("=" * 60)

    best_mae = float('inf')
    best_boundary_maes = {'b0': float('inf'), 'b1': float('inf'), 'b2': float('inf'), 'b3': float('inf')}
    patience_counter = 0
    all_targets_achieved = False

    targets = {'b0': 5.0, 'b1': 4.0, 'b2': 3.0, 'b3': 3.0}
    target_names = {'b0': 'ILM', 'b1': 'RNFL_INL', 'b2': 'INL_ISOS', 'b3': 'ISOS_RPE'}

    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}")
        logging.info("-" * 40)

        train_stats = train_epoch(model, loss_fn, train_loader, optimizer, device, H, args.grad_clip)
        val_stats = validate(model, loss_fn, val_loader, device, H)
        scheduler.step(val_stats['mae'])

        logging.info(f"Epoch {epoch}: Train MAE={train_stats['mae']:.2f}px, Val MAE={val_stats['mae']:.2f}px")

        bm = val_stats['boundary_maes']
        logging.info(f"  Boundary MAE: ILM={bm['b0']:.2f}px, RNFL_INL={bm['b1']:.2f}px, "
                     f"INL_ISOS={bm['b2']:.2f}px, ISOS_RPE={bm['b3']:.2f}px")

        # Check targets
        targets_achieved = []
        targets_status = []
        for bkey in ['b0', 'b1', 'b2', 'b3']:
            if bm[bkey] < best_boundary_maes[bkey]:
                best_boundary_maes[bkey] = bm[bkey]

            target = targets[bkey]
            name = target_names[bkey]
            if bm[bkey] < target:
                targets_achieved.append(f"{name}<{target}")
                targets_status.append(f"✅ {name}={bm[bkey]:.2f}px")
            else:
                gap = bm[bkey] - target
                targets_status.append(f"❌ {name}={bm[bkey]:.2f}px (gap={gap:.2f})")

        if targets_achieved:
            logging.info(f"  TARGETS ACHIEVED: {', '.join(targets_achieved)}")

        # Check if ALL targets achieved
        if len(targets_achieved) == 4:
            all_targets_achieved = True
            logging.info("  🎉 ALL TARGETS ACHIEVED! 🎉")

        d = val_stats['dice']
        logging.info(f"  Val Dice: avg={d['avg']:.4f}")

        # Save best model
        if val_stats['mae'] < best_mae:
            best_mae = val_stats['mae']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'loss_fn': base_loss.state_dict(),
                'epoch': epoch,
                'mae': best_mae,
                'boundary_maes': dict(bm),
                'dice': d['avg'],
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logging.info(f"  -> New best MAE! {best_mae:.2f}px")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"  -> Early stopping after {patience_counter} epochs without improvement")
                break

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"  LR: {current_lr:.2e}, Patience: {patience_counter}/{args.patience}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'loss_fn': base_loss.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))

        gc.collect()

        # Exit early if all targets achieved
        if all_targets_achieved:
            break

    # Final summary
    logging.info("=" * 60)
    logging.info("Training Complete!")
    logging.info(f"Best MAE: {best_mae:.2f}px")
    logging.info("Best per-boundary MAE:")
    for bkey in ['b0', 'b1', 'b2', 'b3']:
        target = targets[bkey]
        best = best_boundary_maes[bkey]
        name = target_names[bkey]
        status = "✅" if best < target else "❌"
        logging.info(f"  {status} {name}: {best:.2f}px (target: <{target}px)")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
