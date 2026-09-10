#!/usr/bin/env python3
"""
Training script for Physics-Informed DSP Model.

Trains the PhysicsInformedDSPModel for OCT boundary detection using:
1. Rayleigh-aware noise normalization
2. Beer-Lambert depth compensation
3. Fresnel physics-based cost priors
4. Differentiable shortest path optimization
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from physics_informed_dsp import (
    PhysicsInformedDSPModel,
    PhysicsInformedDSPLoss,
    boundaries_to_segmentation,
)


# =============================================================================
# Dataset
# =============================================================================
class OCTBoundaryDataset(Dataset):
    """Dataset for OCT boundary detection training."""

    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        add_noise: bool = True,
        noise_level: float = 0.1,
    ):
        self.samples = []
        self.add_noise = add_noise
        self.noise_level = noise_level

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                entry = json.loads(line)
                self.samples.append(entry)

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # Load image
        image = np.array(Image.open(entry['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(entry['mask_path']))

        H, W = image.shape

        # Add synthetic noise for training
        if self.add_noise:
            noise = np.random.randn(*image.shape) * self.noise_level
            noisy = np.clip(image + noise, 0, 1)
        else:
            noisy = image

        # Extract boundaries from mask
        boundaries, valid_mask = self.extract_boundaries(mask)

        # Convert to tensors
        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
        clean_t = torch.from_numpy(image).float().unsqueeze(0)
        boundaries_t = torch.from_numpy(boundaries).float()
        valid_mask_t = torch.from_numpy(valid_mask).float()

        return {
            'noisy': noisy_t,
            'clean': clean_t,
            'boundaries': boundaries_t,  # [4, W] normalized
            'valid_mask': valid_mask_t,  # [W]
            'image_path': entry['image_path'],
        }

    def extract_boundaries(self, mask: np.ndarray) -> tuple:
        """
        Extract boundary positions from segmentation mask.

        Returns:
            boundaries: [4, W] normalized boundary positions
            valid_mask: [W] boolean mask for valid columns
        """
        H, W = mask.shape

        # Remap to 4-class if needed
        mask_4class = self.remap_to_4class(mask)

        boundaries = np.zeros((4, W))
        valid_mask = np.ones(W, dtype=bool)

        for col in range(W):
            col_mask = mask_4class[:, col]
            classes_present = np.unique(col_mask)

            # Check if all 4 classes present
            if not all(c in classes_present for c in range(4)):
                valid_mask[col] = False
                # Use default positions
                boundaries[0, col] = 0.15
                boundaries[1, col] = 0.35
                boundaries[2, col] = 0.55
                boundaries[3, col] = 0.75
                continue

            # Extract boundaries (top of each class except class 0)
            for b in range(4):
                if b == 0:
                    # ILM: top of class 0 (RNFL_GCL)
                    rows = np.where(col_mask == 0)[0]
                    boundaries[b, col] = rows[0] / (H - 1) if len(rows) > 0 else 0.1
                else:
                    # Other boundaries: top of corresponding class
                    rows = np.where(col_mask == b)[0]
                    if len(rows) > 0:
                        boundaries[b, col] = rows[0] / (H - 1)
                    else:
                        # Fallback
                        boundaries[b, col] = boundaries[b-1, col] + 0.15

        return boundaries, valid_mask

    def remap_to_4class(self, mask: np.ndarray) -> np.ndarray:
        """Remap mask to 4 classes."""
        mask_out = np.zeros_like(mask)

        # Map class values (adjust based on your dataset)
        # Assuming original classes: 0-8 or similar
        unique_vals = np.unique(mask)

        if len(unique_vals) <= 4:
            return mask

        # Example mapping (adjust as needed):
        # 0,1 -> 0 (RNFL_GCL)
        # 2,3,4 -> 1 (INL_OPL_ONL)
        # 5,6 -> 2 (IS_OS)
        # 7,8 -> 3 (RPE_Choroid)

        mask_out[mask <= 1] = 0
        mask_out[(mask >= 2) & (mask <= 4)] = 1
        mask_out[(mask >= 5) & (mask <= 6)] = 2
        mask_out[mask >= 7] = 3

        return mask_out


# =============================================================================
# Training Functions
# =============================================================================
def compute_dice(pred_seg: torch.Tensor, gt_seg: torch.Tensor, num_classes: int = 4) -> dict:
    """Compute per-class Dice scores."""
    dice_scores = {}
    class_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    for c in range(num_classes):
        pred_c = (pred_seg == c).float()
        gt_c = (gt_seg == c).float()

        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()

        if union > 0:
            dice = (2 * intersection / union).item()
        else:
            dice = 1.0 if intersection == 0 else 0.0

        dice_scores[class_names[c]] = dice

    return dice_scores


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_mae = 0
    all_stats = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        boundaries_gt = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        B, _, H, W = noisy.shape

        # Forward
        outputs = model(noisy, return_aux=True)
        pred_boundaries = outputs['boundaries']

        # Loss
        loss, stats = loss_fn(
            pred_boundaries,
            boundaries_gt,
            costs=outputs['costs'],
            physics_aux=outputs.get('physics_aux'),
            valid_mask=valid_mask,
            H=H,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_mae += stats['avg_mae_px']
        all_stats.append(stats)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'MAE': f"{stats['avg_mae_px']:.2f}px",
        })

    n_batches = len(dataloader)

    # Aggregate stats
    avg_stats = {
        'loss': total_loss / n_batches,
        'avg_mae_px': total_mae / n_batches,
    }

    # Per-boundary MAE
    for key in ['ILM_mae_px', 'RNFL_INL_mae_px', 'INL_ISOS_mae_px', 'ISOS_RPE_mae_px']:
        avg_stats[key] = np.mean([s[key] for s in all_stats])

    return avg_stats


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> dict:
    """Validate model."""
    model.eval()

    total_loss = 0
    total_mae = 0
    all_stats = []
    all_dice = {k: [] for k in ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']}

    for batch in tqdm(dataloader, desc="Validating"):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        boundaries_gt = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        B, _, H, W = noisy.shape

        # Forward
        outputs = model(noisy, return_aux=True)
        pred_boundaries = outputs['boundaries']

        # Loss
        loss, stats = loss_fn(
            pred_boundaries,
            boundaries_gt,
            costs=outputs['costs'],
            physics_aux=outputs.get('physics_aux'),
            valid_mask=valid_mask,
            H=H,
        )

        total_loss += loss.item()
        total_mae += stats['avg_mae_px']
        all_stats.append(stats)

        # Compute Dice from segmentation
        pred_seg = boundaries_to_segmentation(pred_boundaries, H)
        gt_seg = boundaries_to_segmentation(boundaries_gt, H)

        dice = compute_dice(pred_seg[0], gt_seg[0])
        for k, v in dice.items():
            all_dice[k].append(v)

    n_batches = len(dataloader)

    avg_stats = {
        'val_loss': total_loss / n_batches,
        'val_mae_px': total_mae / n_batches,
    }

    # Per-boundary MAE
    for key in ['ILM_mae_px', 'RNFL_INL_mae_px', 'INL_ISOS_mae_px', 'ISOS_RPE_mae_px']:
        avg_stats[f'val_{key}'] = np.mean([s[key] for s in all_stats])

    # Dice scores
    for k, v in all_dice.items():
        avg_stats[f'val_dice_{k}'] = np.mean(v)

    avg_stats['val_dice_avg'] = np.mean([np.mean(v) for v in all_dice.values()])

    return avg_stats


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Physics-Informed DSP Model')

    # Data
    parser.add_argument('--train_jsonl', type=str, default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)

    # Model
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--num_levels', type=int, default=4)
    parser.add_argument('--use_noise_estimation', action='store_true', default=True)
    parser.add_argument('--use_depth_compensation', action='store_true', default=True)
    parser.add_argument('--use_fresnel_physics', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--min_gap', type=int, default=5)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cpu')

    # Loss weights
    parser.add_argument('--lambda_position', type=float, default=1.0)
    parser.add_argument('--lambda_cost', type=float, default=0.5)
    parser.add_argument('--lambda_ordering', type=float, default=0.1)
    parser.add_argument('--lambda_smoothness', type=float, default=0.5)
    parser.add_argument('--lambda_physics', type=float, default=0.2)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/physics_dsp')
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("Physics-Informed DSP Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")

    # Create datasets
    train_dataset = OCTBoundaryDataset(
        args.train_jsonl,
        max_samples=args.max_train,
        add_noise=True,
        noise_level=0.1,
    )

    val_dataset = OCTBoundaryDataset(
        args.val_jsonl,
        max_samples=args.max_val,
        add_noise=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = PhysicsInformedDSPModel(
        in_channels=1,
        base_channels=args.base_channels,
        num_levels=args.num_levels,
        num_boundaries=4,
        use_noise_estimation=args.use_noise_estimation,
        use_depth_compensation=args.use_depth_compensation,
        use_fresnel_physics=args.use_fresnel_physics,
        temperature=args.temperature,
        min_gap=args.min_gap,
    ).to(args.device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print physics info
    physics_summary = model.get_physics_summary()
    print(f"\nPhysics configuration:")
    print(f"  Noise estimation: {args.use_noise_estimation}")
    print(f"  Depth compensation: {args.use_depth_compensation}")
    print(f"  Fresnel physics: {args.use_fresnel_physics}")
    if args.use_fresnel_physics:
        print(f"  Expected gradient strengths: {physics_summary.get('expected_gradient_strength', 'N/A')}")

    # Loss function
    loss_fn = PhysicsInformedDSPLoss(
        num_boundaries=4,
        lambda_position=args.lambda_position,
        lambda_cost=args.lambda_cost,
        lambda_ordering=args.lambda_ordering,
        lambda_smoothness=args.lambda_smoothness,
        lambda_physics=args.lambda_physics,
    ).to(args.device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 100,
    )

    # Training loop
    best_mae = float('inf')
    history = []

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_stats = train_epoch(
            model, train_loader, loss_fn, optimizer, args.device, epoch
        )

        # Validate
        val_stats = validate(model, val_loader, loss_fn, args.device)

        # Update scheduler
        scheduler.step()

        # Combine stats
        epoch_stats = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            **train_stats,
            **val_stats,
        }

        # Add physics parameters
        physics_summary = model.get_physics_summary()
        if 'attenuation_mu' in physics_summary:
            epoch_stats['attenuation_mu'] = physics_summary['attenuation_mu']
        if 'physics_weight' in physics_summary:
            epoch_stats['physics_weight'] = physics_summary['physics_weight']

        history.append(epoch_stats)

        # Print stats
        print(f"\nTrain: Loss={train_stats['loss']:.4f}, MAE={train_stats['avg_mae_px']:.2f}px")
        print(f"Val:   Loss={val_stats['val_loss']:.4f}, MAE={val_stats['val_mae_px']:.2f}px")
        print(f"       Dice_avg={val_stats['val_dice_avg']:.4f}")
        print(f"       IS_OS MAE={val_stats['val_INL_ISOS_mae_px']:.2f}px, "
              f"RPE MAE={val_stats['val_ISOS_RPE_mae_px']:.2f}px")

        if args.use_depth_compensation:
            print(f"       Attenuation μ={physics_summary.get('attenuation_mu', 0):.6f}")

        # Save best model
        if val_stats['val_mae_px'] < best_mae:
            best_mae = val_stats['val_mae_px']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': best_mae,
                'physics_summary': physics_summary,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> New best! MAE={best_mae:.2f}px")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'physics_summary': model.get_physics_summary(),
    }, os.path.join(args.output_dir, 'final_model.pth'))

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation MAE: {best_mae:.2f}px")
    print(f"Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
