#!/usr/bin/env python3
"""
Staged Training for Physics-Enhanced OCT Layer Segmentation V3

Stage 1: 128×128, 300 images  - Quick iteration, find good features
Stage 2: 192×192, 500 images  - Scale up, refine boundaries
Stage 3: 256×256, 800 images  - Full resolution, final polish

Each stage loads weights from the previous stage for curriculum learning.
"""

import argparse
import gc
import json
import logging
import math
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    name: str
    image_size: int
    max_train: int
    max_val: int
    batch_size: int
    lr: float
    epochs: int
    patience: int
    min_mae_to_advance: float  # Must achieve this MAE to proceed to next stage


class OCTBoundaryDataset(Dataset):
    """Dataset for OCT boundary detection."""

    def __init__(self, jsonl_path, max_samples=None, target_size=(256, 256)):
        self.samples = []
        self.target_size = target_size

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    item = json.loads(line)
                    # Validate required fields - need either boundaries OR mask_path
                    if 'image_path' not in item:
                        continue  # Skip invalid entries
                    if 'boundaries' not in item and 'mask_path' not in item:
                        continue  # Need at least one way to get boundaries
                    self.samples.append(item)
                except json.JSONDecodeError:
                    continue  # Skip malformed JSON lines

    def __len__(self):
        return len(self.samples)

    def _extract_boundaries_from_mask(self, mask):
        """Extract normalized [0,1] boundary positions from segmentation mask."""
        H, W = mask.shape
        boundaries = np.zeros((4, W), dtype=np.float32)
        valid_mask = np.ones(W, dtype=bool)

        # Avoid division by zero when H=1
        H_divisor = max(H - 1, 1)

        for col in range(W):
            col_data = mask[:, col]
            for orig_class in [1, 2, 3, 4]:
                rows = np.where(col_data == orig_class)[0]
                if len(rows) > 0:
                    # Normalize to [0, 1] range
                    boundaries[orig_class - 1, col] = rows[0] / H_divisor
                else:
                    valid_mask[col] = False
                    # Default fallback positions
                    boundaries[orig_class - 1, col] = (orig_class * 0.1 + 0.2)

        return boundaries, valid_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        target_h, target_w = self.target_size

        # Load image
        img_path = item['image_path']
        orig_w, orig_h = target_w, target_h  # Default in case of error
        try:
            with Image.open(img_path) as img_file:
                img = img_file.convert('L')  # Creates a copy, original closed by context manager
                orig_w, orig_h = img.size  # PIL returns (width, height)

                # Resize - PIL.resize takes (width, height)
                img_resized = img.resize((target_w, target_h), Image.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)
                img_resized.close()  # Close resized image
                img.close()  # Close the converted copy
        except (FileNotFoundError, IOError) as e:
            # Return blank image if file not found
            img_tensor = torch.zeros(1, target_h, target_w, dtype=torch.float32)

        # Load mask and extract boundaries
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        boundaries = None
        valid_mask = None

        if 'mask_path' in item and os.path.exists(item['mask_path']):
            try:
                with Image.open(item['mask_path']) as mask_file:
                    mask_img = mask_file.convert('L')  # Creates a copy
                    mask_orig = np.array(mask_img)
                    mask_img.close()

                # Extract boundaries from original-resolution mask (normalized to [0,1])
                boundaries_orig, valid_mask_orig = self._extract_boundaries_from_mask(mask_orig)

                # Interpolate boundaries to target width if needed
                if orig_w != target_w:
                    x_orig = np.linspace(0, 1, orig_w)
                    x_new = np.linspace(0, 1, target_w)
                    boundaries = np.zeros((4, target_w), dtype=np.float32)
                    for b in range(4):
                        boundaries[b] = np.interp(x_new, x_orig, boundaries_orig[b])
                    valid_mask = np.interp(x_new, x_orig, valid_mask_orig.astype(float)) > 0.5
                else:
                    boundaries = boundaries_orig
                    valid_mask = valid_mask_orig

                # Resize mask for Dice computation
                mask_pil = Image.fromarray(mask_orig)
                mask_resized = mask_pil.resize((target_w, target_h), Image.NEAREST)
                mask = np.array(mask_resized)
                mask_resized.close()
                mask_pil.close()

            except (FileNotFoundError, IOError) as e:
                pass  # Keep default zeros

        # Fallback: use pre-computed boundaries from JSONL if available
        if boundaries is None:
            if 'boundaries' in item:
                boundaries = np.array(item['boundaries'], dtype=np.float32)
                # Scale if boundaries are in pixel space (values > 1)
                if boundaries.max() > 1.0:
                    boundaries = boundaries / orig_h
            else:
                # Default fallback - each boundary at a constant normalized position
                boundaries = np.array([
                    [0.2] * target_w,  # b0 (ILM)
                    [0.3] * target_w,  # b1 (RNFL_INL)
                    [0.4] * target_w,  # b2 (INL_ISOS)
                    [0.5] * target_w,  # b3 (ISOS_RPE)
                ], dtype=np.float32)

        if valid_mask is None:
            if 'valid_boundaries' in item:
                valid_mask = np.array(item['valid_boundaries'], dtype=bool)
            else:
                valid_mask = np.ones(target_w, dtype=bool)

        return {
            'image': img_tensor,
            'boundaries': torch.from_numpy(boundaries).float(),
            'valid_mask': torch.from_numpy(valid_mask.astype(np.float32)),
            'mask': torch.from_numpy(mask).long(),
        }


def setup_logging(output_dir: str, stage_name: str = "training") -> logging.Logger:
    """Setup logging for a stage."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{stage_name}.log")

    # Clear and close existing handlers to prevent leaks
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_dice_scores(pred_boundaries, gt_mask, H):
    """Compute Dice scores for each layer."""
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

    # Clean up intermediate tensor
    del pred_seg

    return dice_scores


def train_epoch(model, loss_fn, loader, optimizer, device, H, physics_warmup=1.0, grad_clip=0.5):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mae = 0

    with tqdm(loader, desc=f"Training (physics={physics_warmup:.2f})") as pbar:
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

            total_loss += loss_val
            total_mae += mae_val

            pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'MAE': f"{mae_val:.1f}px",
            })

            # Clear intermediate tensors to free memory
            del outputs, loss

    n = len(loader)
    if n == 0:
        return {'loss': float('inf'), 'mae': float('inf')}
    return {'loss': total_loss / n, 'mae': total_mae / n}


def validate(model, loss_fn, loader, device, H):
    """Validate model."""
    model.eval()
    total_mae = 0
    all_dice = []
    boundary_maes = {f'b{i}': 0 for i in range(4)}
    total_inl_thick_mae = 0

    with torch.no_grad():
        with tqdm(loader, desc="Validating") as pbar:
            for batch in pbar:
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

                # Clear intermediate tensors to free memory
                del outputs

    n = len(loader)
    if n == 0:
        return {
            'mae': float('inf'),
            'inl_thick_mae': float('inf'),
            'boundary_maes': {f'b{i}': float('inf') for i in range(4)},
            'dice': {'avg': 0.0, 'RNFL_GCL': 0.0, 'INL_OPL_ONL': 0.0, 'IS_OS': 0.0, 'RPE_Choroid': 0.0},
        }

    avg_dice = {key: np.mean([d[key] for d in all_dice]) for key in all_dice[0].keys()} if all_dice else {
        'avg': 0.0, 'RNFL_GCL': 0.0, 'INL_OPL_ONL': 0.0, 'IS_OS': 0.0, 'RPE_Choroid': 0.0
    }

    return {
        'mae': total_mae / n,
        'inl_thick_mae': total_inl_thick_mae / n,
        'boundary_maes': {k: v / n for k, v in boundary_maes.items()},
        'dice': avg_dice,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mae, best_dice, stage_name):
    """Save training checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'best_mae': best_mae,
        'best_dice': best_dice,
        'stage_name': stage_name,
    }, path)


def train_stage(
    stage: StageConfig,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    device: torch.device,
    prev_checkpoint: Optional[str] = None,
    hidden_channels: int = 48,
) -> Dict[str, Any]:
    """
    Train a single stage.

    Returns dict with:
        - success: bool
        - best_mae: float
        - best_dice: float
        - checkpoint_path: str
    """
    global _shutdown_requested

    stage_dir = os.path.join(output_dir, stage.name)
    os.makedirs(stage_dir, exist_ok=True)

    logger = setup_logging(stage_dir, "training")

    logger.info("=" * 60)
    logger.info(f"STAGE: {stage.name}")
    logger.info("=" * 60)
    logger.info(f"  Image size: {stage.image_size}x{stage.image_size}")
    logger.info(f"  Training samples: {stage.max_train}")
    logger.info(f"  Validation samples: {stage.max_val}")
    logger.info(f"  Batch size: {stage.batch_size}")
    logger.info(f"  Learning rate: {stage.lr}")
    logger.info(f"  Max epochs: {stage.epochs}")
    logger.info(f"  Patience: {stage.patience}")
    logger.info(f"  Target MAE to advance: <{stage.min_mae_to_advance}px")
    if prev_checkpoint:
        logger.info(f"  Loading weights from: {prev_checkpoint}")
    logger.info("=" * 60)

    H = stage.image_size
    target_size = (H, H)

    # Create datasets
    train_ds = OCTBoundaryDataset(train_jsonl, stage.max_train, target_size=target_size)
    val_ds = OCTBoundaryDataset(val_jsonl, stage.max_val, target_size=target_size)

    logger.info(f"Loaded {len(train_ds)} training, {len(val_ds)} validation samples")

    # Warn if dataset is empty
    if len(train_ds) == 0:
        logger.error("No training samples loaded! Check your JSONL file.")
        return {
            'success': False,
            'best_mae': float('inf'),
            'best_dice': 0.0,
            'checkpoint_path': os.path.join(stage_dir, "best_model.pt"),
            'stage_name': stage.name,
        }

    # Adjust batch size if larger than dataset
    effective_batch_size = min(stage.batch_size, len(train_ds))
    if effective_batch_size != stage.batch_size:
        logger.warning(f"Batch size {stage.batch_size} > dataset size {len(train_ds)}, using {effective_batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(stage.batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = PhysicsEnsembleV3(
        in_channels=1,
        hidden_channels=hidden_channels,
        num_boundaries=4,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {param_count:,} parameters")

    # Load previous checkpoint if available
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        logger.info(f"Loading weights from previous stage: {prev_checkpoint}")
        try:
            checkpoint = torch.load(prev_checkpoint, map_location=device, weights_only=False)
            # Handle both checkpoint key formats:
            # - train_physics_v3.py uses 'model'
            # - train_physics_v3_staged.py uses 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                raise KeyError(f"Checkpoint has no 'model' or 'model_state_dict' key. Keys: {list(checkpoint.keys())}")

            # Use strict=False to allow loading even if some keys don't match
            # (e.g., model architecture slightly changed between stages)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"  Missing keys when loading checkpoint: {missing}")
            if unexpected:
                logger.warning(f"  Unexpected keys when loading checkpoint: {unexpected}")
            logger.info(f"  Previous best MAE: {checkpoint.get('best_mae', checkpoint.get('mae', 'N/A'))}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Starting from scratch instead")

    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=stage.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    loss_fn = PhysicsLossV3().to(device)

    # Training loop
    best_mae = float('inf')
    best_dice = 0.0
    patience_counter = 0
    best_checkpoint_path = os.path.join(stage_dir, "best_model.pt")
    epoch = 0  # Initialize in case epochs <= 0

    for epoch in range(1, stage.epochs + 1):
        if _shutdown_requested:
            logger.info("Shutdown requested, saving checkpoint and exiting...")
            # Save current state before exiting
            shutdown_checkpoint_path = os.path.join(stage_dir, "shutdown_checkpoint.pt")
            save_checkpoint(
                shutdown_checkpoint_path, model, optimizer, scheduler,
                epoch, best_mae, best_dice, stage.name
            )
            logger.info(f"  Saved shutdown checkpoint to {shutdown_checkpoint_path}")
            break

        physics_warmup = min(1.0, epoch / 5)

        logger.info(f"Epoch {epoch}/{stage.epochs} (physics_warmup={physics_warmup:.2f})")
        logger.info("-" * 40)

        # Train
        train_stats = train_epoch(
            model, loss_fn, train_loader, optimizer, device, H,
            physics_warmup=physics_warmup
        )

        # Validate
        val_stats = validate(model, loss_fn, val_loader, device, H)

        val_mae = val_stats['mae']
        val_dice = val_stats['dice']['avg']

        # Log results
        logger.info(f"Epoch {epoch}: Train MAE={train_stats['mae']:.2f}px, Val MAE={val_mae:.2f}px")
        logger.info(f"  Boundary MAE: ILM={val_stats['boundary_maes']['b0']:.2f}px, "
                   f"RNFL_INL={val_stats['boundary_maes']['b1']:.2f}px, "
                   f"INL_ISOS={val_stats['boundary_maes']['b2']:.2f}px, "
                   f"ISOS_RPE={val_stats['boundary_maes']['b3']:.2f}px")
        logger.info(f"  INL thickness MAE: {val_stats['inl_thick_mae']:.2f}px")
        logger.info(f"  Val Dice: avg={val_dice:.4f}")
        logger.info(f"    {', '.join(f'{k}: {v:.4f}' for k, v in val_stats['dice'].items() if k != 'avg')}")

        # Check for improvement
        improved = False
        if val_mae < best_mae:
            best_mae = val_mae
            improved = True
            logger.info(f"  -> New best MAE! {best_mae:.2f}px")

        if val_dice > best_dice:
            best_dice = val_dice
            if not improved:
                logger.info(f"  -> New best Dice! {best_dice:.4f}")
            else:
                logger.info(f"  -> New best Dice! {best_dice:.4f}")
            improved = True

        if improved:
            patience_counter = 0
            save_checkpoint(
                best_checkpoint_path, model, optimizer, scheduler,
                epoch, best_mae, best_dice, stage.name
            )
        else:
            patience_counter += 1

        # Update scheduler (skip if val_mae is inf to avoid issues)
        if not math.isinf(val_mae):
            scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  LR: {current_lr:.2e}, Patience: {patience_counter}/{stage.patience}")

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(stage_dir, f"checkpoint_epoch{epoch}.pt")
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler,
                epoch, best_mae, best_dice, stage.name
            )
            logger.info(f"  -> Checkpoint saved")

        # Early stopping
        if patience_counter >= stage.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Check if we've hit target
        if best_mae < stage.min_mae_to_advance:
            logger.info(f"Reached target MAE ({best_mae:.2f} < {stage.min_mae_to_advance})!")

        # Garbage collection
        gc.collect()

    # Stage summary
    logger.info("=" * 60)
    logger.info(f"STAGE {stage.name} COMPLETE")
    logger.info(f"  Best MAE: {best_mae:.2f}px")
    logger.info(f"  Best Dice: {best_dice:.4f}")
    logger.info(f"  Target: <{stage.min_mae_to_advance}px")

    success = best_mae < stage.min_mae_to_advance
    if success:
        logger.info(f"  Status: PASSED - Ready for next stage")
    else:
        logger.info(f"  Status: DID NOT MEET TARGET - Consider more training")
    logger.info("=" * 60)

    # Ensure best checkpoint exists (save current if not)
    if not os.path.exists(best_checkpoint_path):
        logger.warning("No best checkpoint saved during training, saving final model")
        save_checkpoint(
            best_checkpoint_path, model, optimizer, scheduler,
            epoch, best_mae, best_dice, stage.name
        )

    # Clean up to free memory before next stage
    del model, optimizer, scheduler, loss_fn
    del train_ds, val_ds, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'success': success,
        'best_mae': best_mae,
        'best_dice': best_dice,
        'checkpoint_path': best_checkpoint_path,
        'stage_name': stage.name,
    }


def run_staged_training(
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    device: torch.device,
    stages: List[StageConfig],
    start_stage: int = 1,
    prev_checkpoint: Optional[str] = None,
    hidden_channels: int = 48,
):
    """Run full staged training pipeline."""
    global _shutdown_requested

    # Reset shutdown flag at start of training
    _shutdown_requested = False

    os.makedirs(output_dir, exist_ok=True)

    # Setup main logging
    logger = setup_logging(output_dir, "staged_training")

    logger.info("=" * 60)
    logger.info("STAGED TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Total stages: {len(stages)}")
    logger.info(f"Starting from stage: {start_stage}")
    logger.info(f"Output directory: {output_dir}")
    for i, stage in enumerate(stages, start=1):
        logger.info(f"  Stage {i}: {stage.name} - {stage.image_size}px, {stage.max_train} samples, target <{stage.min_mae_to_advance}px")
    logger.info("=" * 60)

    # Track results
    results = []
    current_checkpoint = prev_checkpoint

    for i, stage in enumerate(stages, start=1):
        if _shutdown_requested:
            logger.info("Shutdown requested, stopping pipeline")
            break

        if i < start_stage:
            # Skip stages before start_stage, but use their checkpoints if available
            checkpoint_path = os.path.join(output_dir, stage.name, "best_model.pt")
            if os.path.exists(checkpoint_path):
                current_checkpoint = checkpoint_path
                logger.info(f"Skipping stage {i} ({stage.name}), using existing checkpoint")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Stage {i}/{len(stages)}: {stage.name}")
        logger.info(f"{'='*60}\n")

        # Train this stage
        stage_result = train_stage(
            stage=stage,
            train_jsonl=train_jsonl,
            val_jsonl=val_jsonl,
            output_dir=output_dir,
            device=device,
            prev_checkpoint=current_checkpoint,
            hidden_channels=hidden_channels,
        )

        results.append(stage_result)
        current_checkpoint = stage_result['checkpoint_path']

        # Save progress (convert inf to string for JSON compatibility)
        progress_file = os.path.join(output_dir, "training_progress.json")
        serializable_results = []
        for r in results:
            sr = r.copy()
            # Handle non-JSON-serializable float values
            if isinstance(sr['best_mae'], float) and math.isinf(sr['best_mae']):
                sr['best_mae'] = "inf"
            if isinstance(sr['best_dice'], float) and math.isinf(sr['best_dice']):
                sr['best_dice'] = "inf"
            serializable_results.append(sr)

        with open(progress_file, 'w') as f:
            json.dump({
                'completed_stages': [r['stage_name'] for r in results],
                'results': serializable_results,
                'last_checkpoint': current_checkpoint,
            }, f, indent=2)

        # Check if we should continue
        if not stage_result['success'] and i < len(stages):
            logger.warning(f"Stage {i} did not meet target MAE.")
            logger.warning(f"  Achieved: {stage_result['best_mae']:.2f}px")
            logger.warning(f"  Target: <{stage.min_mae_to_advance}px")
            logger.warning("Continuing to next stage anyway (results may be suboptimal)")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("STAGED TRAINING COMPLETE")
    logger.info("=" * 60)

    for i, result in enumerate(results, start=start_stage):
        status = "PASSED" if result['success'] else "BELOW TARGET"
        logger.info(f"Stage {i} ({result['stage_name']}): MAE={result['best_mae']:.2f}px, Dice={result['best_dice']:.4f} [{status}]")

    if results:
        final_result = results[-1]
        logger.info(f"\nFinal model checkpoint: {final_result['checkpoint_path']}")
        logger.info(f"Final MAE: {final_result['best_mae']:.2f}px")
        logger.info(f"Final Dice: {final_result['best_dice']:.4f}")

    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Staged Training for OCT Layer Segmentation")

    # Data arguments
    parser.add_argument('--train_jsonl', type=str, default='combined_train.jsonl',
                        help='Training data JSONL')
    parser.add_argument('--val_jsonl', type=str, default='combined_val.jsonl',
                        help='Validation data JSONL')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/staged_training',
                        help='Output directory')

    # Stage control
    parser.add_argument('--start_stage', type=int, default=1,
                        help='Stage to start from (1-indexed)')
    parser.add_argument('--prev_checkpoint', type=str, default=None,
                        help='Checkpoint to load for first stage')

    # Stage overrides (optional)
    parser.add_argument('--stage1_samples', type=int, default=300,
                        help='Training samples for stage 1')
    parser.add_argument('--stage2_samples', type=int, default=500,
                        help='Training samples for stage 2')
    parser.add_argument('--stage3_samples', type=int, default=800,
                        help='Training samples for stage 3')

    # Model
    parser.add_argument('--hidden_channels', type=int, default=48,
                        help='Hidden channels in model')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu, cuda, mps)')

    args = parser.parse_args()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Define stages
    stages = [
        StageConfig(
            name="stage1_128",
            image_size=128,
            max_train=args.stage1_samples,
            max_val=60,
            batch_size=4,
            lr=2e-4,
            epochs=40,
            patience=12,
            min_mae_to_advance=8.0,
        ),
        StageConfig(
            name="stage2_192",
            image_size=192,
            max_train=args.stage2_samples,
            max_val=80,
            batch_size=3,
            lr=1e-4,
            epochs=35,
            patience=12,
            min_mae_to_advance=5.0,
        ),
        StageConfig(
            name="stage3_256",
            image_size=256,
            max_train=args.stage3_samples,
            max_val=100,
            batch_size=2,
            lr=5e-5,
            epochs=30,
            patience=15,
            min_mae_to_advance=4.0,
        ),
    ]

    # Run staged training
    run_staged_training(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        device=device,
        stages=stages,
        start_stage=args.start_stage,
        prev_checkpoint=args.prev_checkpoint,
        hidden_channels=args.hidden_channels,
    )


if __name__ == "__main__":
    main()
