#!/usr/bin/env python3
"""
Training script for Physics-Informed DSP Lite Model.

Lightweight, CPU-optimized training for OCT boundary detection.
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

from physics_dsp_lite import (
    PhysicsDSPLite,
    PhysicsDSPLiteLoss,
    boundaries_to_segmentation,
)


# =============================================================================
# Dataset
# =============================================================================
class OCTBoundaryDataset(Dataset):
    """Dataset for OCT boundary detection with augmentation."""

    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        add_noise: bool = True,
        noise_level: float = 0.1,
        target_size: tuple = (256, 256),  # Fixed output size
        augment: bool = False,  # Enable data augmentation
    ):
        self.samples = []
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.target_size = target_size  # (H, W)
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

        # Extract boundaries BEFORE resizing (in normalized coordinates)
        boundaries, valid_mask = self._extract_boundaries(mask)

        # Resize image and mask to target size
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        image_resized = np.array(image_pil.resize((W_target, H_target), Image.BILINEAR)) / 255.0
        mask_resized = np.array(mask_pil.resize((W_target, H_target), Image.NEAREST))

        # Resize boundaries (columns) - use interpolation
        # boundaries shape: [4, W_orig] -> [4, W_target]
        boundaries_tensor = torch.from_numpy(boundaries).float().unsqueeze(0)  # [1, 4, W_orig]
        boundaries_resized = F.interpolate(
            boundaries_tensor, size=W_target, mode='linear', align_corners=True
        )[0]  # [4, W_target]

        # Resize valid_mask similarly
        valid_tensor = torch.from_numpy(valid_mask.astype(float)).float().unsqueeze(0).unsqueeze(0)  # [1, 1, W_orig]
        valid_resized = F.interpolate(
            valid_tensor, size=W_target, mode='nearest'
        )[0, 0]  # [W_target]
        valid_resized = (valid_resized > 0.5).float()

        # Add noise
        if self.add_noise:
            noise = np.random.randn(*image_resized.shape) * self.noise_level
            noisy = np.clip(image_resized + noise, 0, 1)
        else:
            noisy = image_resized

        # Convert to tensors
        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
        clean_t = torch.from_numpy(image_resized).float().unsqueeze(0)
        mask_t = torch.from_numpy(self._remap_mask(mask_resized)).long()

        # Apply augmentation
        if self.augment and np.random.random() > 0.5:
            # Horizontal flip
            noisy_t = torch.flip(noisy_t, dims=[2])  # Flip W dimension
            clean_t = torch.flip(clean_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[1])
            boundaries_resized = torch.flip(boundaries_resized, dims=[1])  # [N, W] -> flip W
            valid_resized = torch.flip(valid_resized.unsqueeze(0), dims=[1]).squeeze(0)

        return {
            'noisy': noisy_t,
            'clean': clean_t,
            'boundaries': boundaries_resized,
            'valid_mask': valid_resized,
            'mask': mask_t,
        }

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Remap mask to 4 retinal layer classes.

        Original Duke DME mask:
        - 0 = Background/vitreous (above retina)
        - 1 = RNFL_GCL
        - 2 = INL_OPL_ONL
        - 3 = IS_OS
        - 4 = RPE_Choroid

        Output (4 classes):
        - 0 = RNFL_GCL (original 1)
        - 1 = INL_OPL_ONL (original 2)
        - 2 = IS_OS (original 3)
        - 3 = RPE_Choroid (original 4)
        """
        out = np.zeros_like(mask)
        out[mask == 1] = 0  # RNFL_GCL
        out[mask == 2] = 1  # INL_OPL_ONL
        out[mask == 3] = 2  # IS_OS
        out[mask == 4] = 3  # RPE_Choroid
        # Background (mask == 0) stays as 0, but will be above ILM boundary
        return out

    def _extract_boundaries(self, mask: np.ndarray):
        """
        Extract boundary positions from mask.

        Boundaries:
        - boundary[0] = ILM = top of original class 1 (RNFL_GCL)
        - boundary[1] = RNFL/INL = top of original class 2 (INL_OPL_ONL)
        - boundary[2] = INL/IS_OS = top of original class 3 (IS_OS)
        - boundary[3] = IS_OS/RPE = top of original class 4 (RPE_Choroid)
        """
        H, W = mask.shape

        boundaries = np.zeros((4, W))
        valid_mask = np.ones(W, dtype=bool)

        # Original classes to find boundaries for
        original_classes = [1, 2, 3, 4]  # RNFL, INL, IS_OS, RPE

        for col in range(W):
            col_data = mask[:, col]

            # Check if all required classes present
            classes_present = np.unique(col_data)
            if not all(c in classes_present for c in original_classes):
                valid_mask[col] = False
                # Use reasonable defaults
                boundaries[:, col] = [0.25, 0.35, 0.40, 0.45]
                continue

            # Extract boundary for each class (top row of each class)
            for b, orig_class in enumerate(original_classes):
                rows = np.where(col_data == orig_class)[0]
                if len(rows) > 0:
                    boundaries[b, col] = rows[0] / (H - 1)
                else:
                    # Fallback (shouldn't happen if validation passed)
                    boundaries[b, col] = boundaries[b-1, col] + 0.05 if b > 0 else 0.2

        return boundaries, valid_mask


# =============================================================================
# Training
# =============================================================================
def compute_dice(pred_seg, gt_seg, num_classes=4):
    """Compute per-class Dice."""
    dice = {}
    names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    for c in range(num_classes):
        pred_c = (pred_seg == c).float()
        gt_c = (gt_seg == c).float()
        inter = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice[names[c]] = (2 * inter / (union + 1e-8)).item()

    return dice


def train_epoch(model, loader, loss_fn, optimizer, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_mae = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        gt_boundaries = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        H = noisy.shape[2]

        # Forward
        outputs = model(noisy)
        pred = outputs['boundaries']

        # Loss
        loss, stats = loss_fn(pred, gt_boundaries, valid_mask, H)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mae += stats['avg_mae']
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'MAE': f"{stats['avg_mae']:.1f}px"})

    return {'loss': total_loss / n_batches, 'mae': total_mae / n_batches}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Validate."""
    model.eval()

    total_loss = 0
    total_mae = 0
    all_mae = {k: [] for k in ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']}
    all_dice = {k: [] for k in ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']}
    n_batches = 0

    for batch in tqdm(loader, desc="Validating"):
        noisy = batch['noisy'].to(device)
        gt_boundaries = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        gt_mask = batch['mask'].to(device)

        H = noisy.shape[2]

        outputs = model(noisy)
        pred = outputs['boundaries']

        loss, stats = loss_fn(pred, gt_boundaries, valid_mask, H)

        total_loss += loss.item()
        total_mae += stats['avg_mae']

        for k in all_mae.keys():
            all_mae[k].append(stats[f'{k}_mae'])

        # Dice
        pred_seg = boundaries_to_segmentation(pred, H)
        dice = compute_dice(pred_seg[0], gt_mask[0])
        for k, v in dice.items():
            all_dice[k].append(v)

        n_batches += 1

    result = {
        'val_loss': total_loss / n_batches,
        'val_mae': total_mae / n_batches,
    }

    for k, v in all_mae.items():
        result[f'val_{k}_mae'] = np.mean(v)

    for k, v in all_dice.items():
        result[f'val_dice_{k}'] = np.mean(v)

    result['val_dice_avg'] = np.mean([np.mean(v) for v in all_dice.values()])

    return result


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
    parser.add_argument('--hidden_channels', type=int, default=48)  # Increased from 32
    parser.add_argument('--use_fresnel', action='store_true', default=True)
    parser.add_argument('--use_depth_comp', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=0.03)  # Lower for sharper boundaries
    parser.add_argument('--physics_weight', type=float, default=0.4)  # Increased physics influence

    # Training
    parser.add_argument('--epochs', type=int, default=50)  # More epochs
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)  # Slightly lower LR
    parser.add_argument('--device', default='cpu')

    # Output
    parser.add_argument('--output_dir', default='outputs/physics_dsp_lite')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Physics-Informed DSP Lite Training")
    print("=" * 60)

    # Data (enable augmentation for training)
    train_data = OCTBoundaryDataset(args.train_jsonl, args.max_train, add_noise=True, augment=True)
    val_data = OCTBoundaryDataset(args.val_jsonl, args.max_val, add_noise=False, augment=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Model
    model = PhysicsDSPLite(
        hidden_channels=args.hidden_channels,
        use_fresnel=args.use_fresnel,
        use_depth_comp=args.use_depth_comp,
        temperature=args.temperature,
        physics_weight=args.physics_weight,
    ).to(args.device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Physics info
    info = model.get_physics_info()
    print(f"\nPhysics configuration:")
    print(f"  Fresnel: {args.use_fresnel}")
    print(f"  Depth compensation: {args.use_depth_comp}")
    if args.use_fresnel:
        print(f"  Expected gradient strengths: {[f'{s:.3f}' for s in info['expected_strength']]}")

    # Loss function with LEARNABLE weights
    loss_fn = PhysicsDSPLiteLoss(
        learnable_weights=True,
        lambda_thickness=2.0,  # High weight for thickness (thin layers!)
        lambda_dice=1.0,  # Soft Dice loss
    ).to(args.device)

    # Optimizer: include BOTH model AND loss function parameters
    # This allows learning optimal boundary/thickness weights
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/100)

    print(f"\nLoss function parameters: {sum(p.numel() for p in loss_fn.parameters())} learnable weights")

    # Training
    best_mae = float('inf')
    history = []

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, loss_fn, optimizer, args.device, epoch)
        val_stats = validate(model, val_loader, loss_fn, args.device)
        scheduler.step()

        # Physics info
        physics = model.get_physics_info()

        epoch_stats = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            **train_stats,
            **val_stats,
            **physics,
        }
        history.append(epoch_stats)

        # Get learned weights from loss function
        learned_weights = loss_fn.get_learned_weights()

        print(f"\nEpoch {epoch}: Train MAE={train_stats['mae']:.2f}px, Val MAE={val_stats['val_mae']:.2f}px")
        print(f"  Boundary MAE: ILM={val_stats['val_ILM_mae']:.2f}px, RNFL_INL={val_stats['val_RNFL_INL_mae']:.2f}px, "
              f"INL_ISOS={val_stats['val_INL_ISOS_mae']:.2f}px, ISOS_RPE={val_stats['val_ISOS_RPE_mae']:.2f}px")
        print(f"  Val Dice: avg={val_stats['val_dice_avg']:.4f}")
        print(f"    RNFL_GCL: {val_stats['val_dice_RNFL_GCL']:.4f}, INL_OPL_ONL: {val_stats['val_dice_INL_OPL_ONL']:.4f}, "
              f"IS_OS: {val_stats['val_dice_IS_OS']:.4f}, RPE_Choroid: {val_stats['val_dice_RPE_Choroid']:.4f}")

        # Show learned weights (should auto-increase for thin layers)
        bw = learned_weights['boundary_weights']
        tw = learned_weights['thickness_weights']
        dw = learned_weights['dice_weights']
        print(f"  Learned boundary weights: [{bw[0]:.2f}, {bw[1]:.2f}, {bw[2]:.2f}, {bw[3]:.2f}]")
        print(f"  Learned thickness weights: [{tw[0]:.2f}, {tw[1]:.2f}(INL!), {tw[2]:.2f}]")

        if args.use_depth_comp:
            print(f"  Depth μ: {physics.get('depth_mu', 0):.5f}")
        if args.use_fresnel:
            print(f"  Physics weight: {physics.get('physics_weight', 0):.3f}")

        # Save best
        if val_stats['val_mae'] < best_mae:
            best_mae = val_stats['val_mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': best_mae,
                'physics': physics,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> New best! MAE={best_mae:.2f}px")

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(args.output_dir, 'final_model.pth'))

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Best MAE: {best_mae:.2f}px")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
