#!/usr/bin/env python3
"""
Training script for Physics-Enhanced Ensemble Model v3.

Features:
- All physics contributions maintained (Beer-Lambert, gradient, intensity, Fresnel)
- Depth-aware boundary weighting
- Multi-scale boundary prediction
- Layer-specific refinement
- Smoothness regularization

Targets:
- ILM (b0): <5px
- RNFL_INL (b1): <4px
- INL_ISOS (b2): <3px
- ISOS_RPE (b3): <3px
"""

import argparse
import gc
import json
import logging
import os
import signal
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from physics_enhanced_v3 import (
    PhysicsEnsembleV3,
    PhysicsLossV3,
    boundaries_to_segmentation,
)

_shutdown_requested = False

def signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logging.warning(f"Received signal {signum}. Will save and exit after current epoch.")


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'training.log')
    root_logger = logging.getLogger()
    # Close existing handlers before removing to prevent file handle leaks
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )


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

        # Load image with proper cleanup
        with Image.open(entry['image_path']) as img_file:
            img_gray = img_file.convert('L')
            image = np.array(img_gray) / 255.0
            img_gray.close()

        # Load mask with proper cleanup
        with Image.open(entry['mask_path']) as mask_file:
            mask = np.array(mask_file)

        H_orig, W_orig = image.shape
        H_target, W_target = self.target_size

        boundaries, valid_mask = self._extract_boundaries(mask)

        # Resize with proper cleanup
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        image_resized = np.array(image_pil.resize((W_target, H_target), Image.BILINEAR)) / 255.0
        mask_resized = np.array(mask_pil.resize((W_target, H_target), Image.NEAREST))

        # Close intermediate PIL images
        image_pil.close()
        mask_pil.close()

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
        # Avoid division by zero when H=1
        H_divisor = max(H - 1, 1)
        for col in range(W):
            col_data = mask[:, col]
            for orig_class in [1, 2, 3, 4]:
                rows = np.where(col_data == orig_class)[0]
                if len(rows) > 0:
                    boundaries[orig_class - 1, col] = rows[0] / H_divisor
                else:
                    valid_mask[col] = False
                    boundaries[orig_class - 1, col] = (orig_class * 0.1 + 0.2)
        return boundaries, valid_mask


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


def train_epoch(model, loss_fn, loader, optimizer, device, H, physics_warmup=1.0, grad_clip=0.5):
    model.train()
    total_loss = 0
    total_mae = 0

    pbar = tqdm(loader, desc=f"Training (physics={physics_warmup:.1f})")
    for batch in pbar:
        images = batch['image'].to(device)
        gt_bounds = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images, return_aux=True)
        loss, stats = loss_fn(outputs, gt_bounds, valid_mask, H, image=images, physics_warmup=physics_warmup)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Extract scalar values before deleting tensors
        loss_val = loss.item()
        mae_val = stats['avg_mae']
        mu_val = stats['depth_mu']

        total_loss += loss_val
        total_mae += mae_val

        pbar.set_postfix({
            'loss': f"{loss_val:.4f}",
            'MAE': f"{mae_val:.1f}px",
            'μ': f"{mu_val:.2f}",
        })

        # Free memory
        del outputs, loss

    n = len(loader)
    if n == 0:
        return {'loss': float('inf'), 'mae': float('inf')}
    return {'loss': total_loss / n, 'mae': total_mae / n}


def validate(model, loss_fn, loader, device, H):
    model.eval()
    total_mae = 0
    all_dice = []
    boundary_maes = {f'b{i}': 0 for i in range(4)}
    total_inl_thick_mae = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            gt_bounds = batch['boundaries'].to(device)
            gt_mask = batch['mask'].to(device)
            valid_mask = batch['valid_mask'].to(device)

            outputs = model(images, return_aux=True)
            _, stats = loss_fn(outputs, gt_bounds, valid_mask, H, image=images)

            total_mae += stats['avg_mae']
            total_inl_thick_mae += stats['INL_thick_mae']

            boundary_maes['b0'] += stats['ILM_mae']
            boundary_maes['b1'] += stats['RNFL_INL_mae']
            boundary_maes['b2'] += stats['INL_ISOS_mae']
            boundary_maes['b3'] += stats['ISOS_RPE_mae']

            dice = compute_dice_scores(outputs['boundaries'], gt_mask, H)
            all_dice.append(dice)

            # Free memory
            del outputs

    n = len(loader)
    # Handle empty loader
    if n == 0 or not all_dice:
        return {
            'mae': float('inf'),
            'inl_thick_mae': float('inf'),
            'boundary_maes': {f'b{i}': float('inf') for i in range(4)},
            'dice': {'avg': 0.0, 'RNFL_GCL': 0.0, 'INL_OPL_ONL': 0.0, 'IS_OS': 0.0, 'RPE_Choroid': 0.0},
        }

    avg_dice = {key: np.mean([d[key] for d in all_dice]) for key in all_dice[0].keys()}

    return {
        'mae': total_mae / n,
        'inl_thick_mae': total_inl_thick_mae / n,
        'boundary_maes': {k: v / n for k, v in boundary_maes.items()},
        'dice': avg_dice,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--max_val', type=int, default=20)
    parser.add_argument('--hidden_channels', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='outputs/physics_v3')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--physics_warmup_epochs', type=int, default=5)
    parser.add_argument('--checkpoint_every', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=256, help='Image size (height and width)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    H = args.image_size

    setup_logging(args.output_dir)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("=" * 60)
    logging.info("Physics-Enhanced Ensemble Training V3")
    logging.info("=" * 60)
    logging.info("Physics components (ALL ACTIVE):")
    logging.info("  1. Depth compensation (Beer-Lambert attenuation)")
    logging.info("  2. Boundary gradient alignment loss")
    logging.info("  3. Layer intensity consistency loss")
    logging.info("  4. Fresnel reflection consistency loss")
    logging.info("  5. Layer-specific physics refinement")
    logging.info("Enhancements in V3:")
    logging.info("  - Depth-aware boundary loss weighting")
    logging.info("  - Multi-scale boundary prediction fusion")
    logging.info("  - Boundary smoothness regularization")
    logging.info("Targets:")
    logging.info("  - ILM (b0): <5px")
    logging.info("  - RNFL_INL (b1): <4px")
    logging.info("  - INL_ISOS (b2): <3px")
    logging.info("  - ISOS_RPE (b3): <3px")
    logging.info(f"Settings: image_size={H}, lr={args.lr}, batch_size={args.batch_size}")
    logging.info(f"Data: max_train={args.max_train}, max_val={args.max_val}")

    # Data
    target_size = (H, H)  # Square images
    train_ds = OCTBoundaryDataset(args.train_jsonl, args.max_train, target_size=target_size)
    val_ds = OCTBoundaryDataset(args.val_jsonl, args.max_val, target_size=target_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = PhysicsEnsembleV3(hidden_channels=args.hidden_channels).to(device)
    logging.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss
    loss_fn = PhysicsLossV3(
        lambda_position=2.0,
        lambda_thickness=1.5,
        lambda_dice=1.0,
        lambda_gradient=0.4,
        lambda_intensity=0.3,
        lambda_fresnel=0.3,
        lambda_smooth=0.2,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training state
    best_dice = 0
    best_mae = float('inf')
    start_epoch = 1
    patience_counter = 0
    best_boundary_maes = {'b0': float('inf'), 'b1': float('inf'), 'b2': float('inf'), 'b3': float('inf')}

    targets = {'b0': 5.0, 'b1': 4.0, 'b2': 3.0, 'b3': 3.0}
    target_names = {'b0': 'ILM', 'b1': 'RNFL_INL', 'b2': 'INL_ISOS', 'b3': 'ISOS_RPE'}

    # Resume
    if args.resume:
        if os.path.exists(args.resume):
            logging.info(f"Resuming from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            if 'loss_fn' in checkpoint:
                loss_fn.load_state_dict(checkpoint['loss_fn'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0)
            best_mae = checkpoint.get('best_mae', float('inf'))
            logging.info(f"Resumed from epoch {checkpoint['epoch']}")

    logging.info("=" * 60)
    logging.info("Training...")
    logging.info("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        if _shutdown_requested:
            logging.warning("Shutdown requested...")
            break

        physics_warmup = min(1.0, (epoch - 1) / args.physics_warmup_epochs) if args.physics_warmup_epochs > 0 else 1.0

        logging.info(f"Epoch {epoch}/{args.epochs} (physics_warmup={physics_warmup:.2f})")
        logging.info("-" * 40)

        train_stats = train_epoch(model, loss_fn, train_loader, optimizer, device, H, physics_warmup, args.grad_clip)
        val_stats = validate(model, loss_fn, val_loader, device, H)
        scheduler.step(val_stats['mae'])

        logging.info(f"Epoch {epoch}: Train MAE={train_stats['mae']:.2f}px, Val MAE={val_stats['mae']:.2f}px")

        bm = val_stats['boundary_maes']
        logging.info(f"  Boundary MAE: ILM={bm['b0']:.2f}px, RNFL_INL={bm['b1']:.2f}px, "
                     f"INL_ISOS={bm['b2']:.2f}px, ISOS_RPE={bm['b3']:.2f}px")

        # Check targets
        targets_achieved = []
        for bkey in ['b0', 'b1', 'b2', 'b3']:
            if bm[bkey] < best_boundary_maes[bkey]:
                best_boundary_maes[bkey] = bm[bkey]
            if bm[bkey] < targets[bkey]:
                targets_achieved.append(f"{target_names[bkey]}<{targets[bkey]}")

        if targets_achieved:
            logging.info(f"  TARGETS ACHIEVED: {', '.join(targets_achieved)}")

        if len(targets_achieved) == 4:
            logging.info("  ALL TARGETS ACHIEVED!")

        logging.info(f"  INL thickness MAE: {val_stats['inl_thick_mae']:.2f}px")

        d = val_stats['dice']
        logging.info(f"  Val Dice: avg={d['avg']:.4f}")
        logging.info(f"    RNFL_GCL: {d['RNFL_GCL']:.4f}, INL_OPL_ONL: {d['INL_OPL_ONL']:.4f}, "
                     f"IS_OS: {d['IS_OS']:.4f}, RPE_Choroid: {d['RPE_Choroid']:.4f}")

        # Physics parameters
        with torch.no_grad():
            mu_val = model.depth_compensation.get_mu().item()
            refine_scales = model.layer_refine.refine_scales.detach().cpu().numpy()
        logging.info(f"  Physics: depth_μ={mu_val:.3f}, refine_scales={refine_scales}")

        alpha = torch.sigmoid(model.blend_logits).detach()
        logging.info(f"  Blend α: b0={alpha[0]:.2f}, b1={alpha[1]:.2f}, b2={alpha[2]:.2f}, b3={alpha[3]:.2f}")

        # Save best
        if d['avg'] > best_dice:
            best_dice = d['avg']
            torch.save({
                'model': model.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'epoch': epoch,
                'dice': best_dice,
                'mae': val_stats['mae'],
            }, os.path.join(args.output_dir, 'best_dice_model.pth'))
            logging.info(f"  -> New best Dice! avg={best_dice:.4f}")

        if val_stats['mae'] < best_mae:
            best_mae = val_stats['mae']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'epoch': epoch,
                'dice': d['avg'],
                'mae': best_mae,
                'boundary_maes': dict(bm),
            }, os.path.join(args.output_dir, 'best_mae_model.pth'))
            logging.info(f"  -> New best MAE! {best_mae:.2f}px")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch > args.physics_warmup_epochs:
                logging.info(f"  -> Early stopping after {patience_counter} epochs")
                break

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"  LR: {current_lr:.2e}, Patience: {patience_counter}/{args.patience}")

        if epoch % args.checkpoint_every == 0 or _shutdown_requested:
            torch.save({
                'model': model.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))
            logging.info(f"  -> Checkpoint saved")

        gc.collect()

    # Final summary
    logging.info("=" * 60)
    logging.info("Training Complete!")
    logging.info(f"Best Dice: {best_dice:.4f}")
    logging.info(f"Best MAE: {best_mae:.2f}px")
    logging.info("Best per-boundary MAE:")
    for bkey in ['b0', 'b1', 'b2', 'b3']:
        target = targets[bkey]
        best = best_boundary_maes[bkey]
        name = target_names[bkey]
        status = "ACHIEVED" if best < target else f"gap={best-target:.2f}px"
        logging.info(f"  {name}: {best:.2f}px (target: <{target}px) - {status}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
