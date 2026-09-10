#!/usr/bin/env python3
"""
Dice Fine-tuning Script for CUAP-OCT

Run this AFTER main training (tmi_retrain_clinical.sh) completes.

Purpose:
- Improve segmentation Dice score while preserving denoising quality
- Freezes backbone (denoising) and trains only segmentation head
- Use for TMI paper if making segmentation claims

Usage:
    python finetune_dice.py --checkpoint path/to/best_psnr.pth --output_dir finetune_output
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main training script
from train_multitask import (
    MultiTaskOCTDataset,
    MultiTaskDenoiser,
    compute_dice_score,
    compute_per_layer_dice,
    combined_seg_loss,
    compute_psnr,
    compute_ssim,
    LAYER_NAMES,
    validate_per_layer,  # For accurate per-layer evaluation
)
from nsnd.models.nafnet import NAFNetFullFiLM


def compute_dice_present_classes(pred_logits, target):
    """
    Compute Dice score only for classes present in ground truth.

    This is important for 64x64 patches which only contain middle layers (ONL, IS_OS).
    Computing Dice for all 5 classes would penalize predictions of absent classes.
    """
    pred = pred_logits.argmax(dim=1)

    # Find classes present in ground truth
    present_classes = torch.unique(target).tolist()

    dice_scores = []
    for c in present_classes:
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())

    return np.mean(dice_scores) if dice_scores else 0.0


def finetune_epoch(model, loader, optimizer, device, epoch, seg_class_weights=None, lambda_dice=0.7):
    """Fine-tune segmentation head only."""
    model.train()

    # Ensure backbone is frozen
    for param in model.backbone.parameters():
        param.requires_grad = False

    total_seg_loss = 0
    total_dice = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Fine-tune Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        seg_mask = batch['seg_mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        denoised, seg_logits = model(noisy)

        # Only segmentation loss (no denoising loss)
        seg_loss, ce_loss, dice_loss = combined_seg_loss(
            seg_logits, seg_mask,
            ce_weight=seg_class_weights,
            dice_weight=seg_class_weights,
            lambda_dice=lambda_dice,
            num_classes=5
        )

        # Add focal-like weighting for hard examples
        # This helps with thin layers (INL_OPL, ONL)
        with torch.no_grad():
            pred_probs = F.softmax(seg_logits, dim=1)
            pt = pred_probs.gather(1, seg_mask.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt).pow(2).mean()

        # Total loss with focal boost
        loss = seg_loss * (1 + 0.5 * focal_weight)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics - use Dice only for classes present in ground truth
        # This gives consistent metrics since patches only have middle layers
        dice = compute_dice_present_classes(seg_logits.detach(), seg_mask)

        total_seg_loss += seg_loss.item()
        total_dice += dice
        num_batches += 1

        pbar.set_postfix({
            'Loss': f'{total_seg_loss/num_batches:.4f}',
            'Dice': f'{total_dice/num_batches:.4f}'
        })

    return {
        'seg_loss': total_seg_loss / num_batches,
        'dice': total_dice / num_batches,
    }


def validate_finetune(model, loader, device, base_model=None):
    """Validate fine-tuned model."""
    model.eval()

    psnr_sum = 0
    ssim_sum = 0
    dice_sum = 0
    n_samples = 0

    # Per-layer dice
    per_layer_dice_sum = {name: 0.0 for name in LAYER_NAMES}
    per_layer_dice_count = {name: 0 for name in LAYER_NAMES}

    # For PSNR comparison with baseline
    psnr_base_sum = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            seg_mask = batch['seg_mask'].to(device)

            # Our model
            denoised, seg_logits = model(noisy)

            # Baseline (if provided)
            if base_model is not None:
                base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
                for i in range(noisy.size(0)):
                    psnr_base_sum += compute_psnr(base_out[i:i+1], clean[i:i+1])

            # Segmentation metrics - use consistent Dice for present classes only
            dice = compute_dice_present_classes(seg_logits, seg_mask)
            dice_sum += dice

            # Per-layer dice (only for classes present in patch)
            layer_dice = compute_per_layer_dice(seg_logits, seg_mask)
            for name in LAYER_NAMES:
                if layer_dice[name] is not None:
                    per_layer_dice_sum[name] += layer_dice[name]
                    per_layer_dice_count[name] += 1

            # Denoising metrics
            for i in range(noisy.size(0)):
                psnr_sum += compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim_sum += compute_ssim(denoised[i:i+1], clean[i:i+1])
                n_samples += 1

    # Compute per-layer averages
    per_layer_results = {}
    for name in LAYER_NAMES:
        if per_layer_dice_count[name] > 0:
            per_layer_results[name] = per_layer_dice_sum[name] / per_layer_dice_count[name]
        else:
            per_layer_results[name] = 0.0

    n_batches = len(loader)

    return {
        'psnr': psnr_sum / n_samples,
        'psnr_base': psnr_base_sum / n_samples if base_model else None,
        'ssim': ssim_sum / n_samples,
        'dice': dice_sum / n_batches,
        'per_layer_dice': per_layer_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune segmentation for better Dice')
    parser.add_argument('--checkpoint', required=True, help='Path to best_psnr.pth from main training')
    parser.add_argument('--backbone_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth',
                        help='Original backbone checkpoint (for baseline comparison)')
    parser.add_argument('--train_jsonl', default='seg_data/seg_train.jsonl')
    parser.add_argument('--val_jsonl', default='seg_data/seg_val.jsonl')
    parser.add_argument('--epochs', type=int, default=15, help='Fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (can be larger since backbone frozen)')
    parser.add_argument('--max_train', type=int, default=2000)
    parser.add_argument('--max_val', type=int, default=400)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--lambda_dice', type=float, default=0.7, help='Dice loss weight (higher = more Dice focus)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='finetune_dice_output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DICE FINE-TUNING FOR CUAP-OCT")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"\nStrategy:")
    print("  - Backbone (denoising): FROZEN")
    print("  - Segmentation head: TRAINABLE")
    print("  - Focus: Improve Dice without hurting PSNR")
    print(f"\nParameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr} (low for fine-tuning)")
    print(f"  Lambda Dice: {args.lambda_dice} (high for Dice focus)")
    print("=" * 70)

    # Load datasets
    # IMPORTANT: Use ensure_all_layers=False to match main training
    # This allows random crops from anywhere in the image (top/middle/bottom)
    # giving patches with different layers, not just ONL/IS_OS from center
    print(f"\nLoading data...")
    train_ds = MultiTaskOCTDataset(
        args.train_jsonl, patch_size=64, max_samples=args.max_train,
        random_crop=True, ensure_all_layers=False
    )
    val_ds = MultiTaskOCTDataset(
        args.val_jsonl, patch_size=64, max_samples=args.max_val,
        random_crop=False, ensure_all_layers=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Use predefined class weights based on full-image distribution
    # Note: 64x64 patches only capture middle layers (ONL, IS_OS) due to 496px image height
    # But the model needs to learn all layers, so we use balanced weights with clinical boosts
    #
    # Full image class distribution (from seg_data):
    #   RNFL_GCL: ~15%, INL_OPL: ~20%, ONL: ~20%, IS_OS: ~20%, RPE_Choroid: ~25%
    #
    # Inverse frequency weights (before boost):
    #   RNFL_GCL: 1.33, INL_OPL: 1.0, ONL: 1.0, IS_OS: 1.0, RPE_Choroid: 0.8
    print("Setting class weights...")

    # LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']
    # Base weights from inverse frequency, with clinical importance boost
    seg_class_weights = torch.tensor([
        1.6,   # RNFL_GCL: 1.33 * 1.2 (glaucoma-critical)
        1.3,   # INL_OPL: 1.0 * 1.3 (thin layer boost)
        1.0,   # ONL: 1.0 * 1.0 (common in patches)
        1.5,   # IS_OS: 1.0 * 1.5 (critical photoreceptor junction)
        0.8,   # RPE_Choroid: 0.8 * 1.0 (most common layer)
    ], dtype=torch.float32).to(args.device)

    print(f"  Class weights: {[f'{w:.3f}' for w in seg_class_weights.tolist()]}")
    print(f"  Layer order: {LAYER_NAMES}")
    print(f"  Note: Using predefined weights (64x64 patches only capture middle layers)")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = MultiTaskDenoiser(backbone_ckpt=args.backbone_ckpt).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    original_psnr = checkpoint.get('psnr', 0)
    original_dice = checkpoint.get('dice', 0)
    print(f"Original checkpoint - PSNR: {original_psnr:.2f} dB, Dice: {original_dice:.4f}")

    # Freeze backbone
    print("\nFreezing backbone (protecting denoising quality)...")
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    print(f"  Frozen params: {frozen_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Load baseline for comparison
    print("\nLoading baseline model for PSNR comparison...")
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)
    base_ckpt = torch.load(args.backbone_ckpt, map_location=args.device, weights_only=False)
    base_model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=False)
    base_model.eval()

    # Optimizer - only for trainable params (segmentation head)
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initial validation using per-layer targeted evaluation
    # Note: Standard patches only capture middle layers; validate_per_layer extracts
    # layer-targeted crops to properly evaluate all 5 layers
    print("\n" + "=" * 70)
    print("INITIAL VALIDATION (before fine-tuning)")
    print("=" * 70)
    val_metrics = validate_finetune(model, val_loader, args.device, base_model)

    # Get accurate per-layer metrics using targeted evaluation
    per_layer_metrics = validate_per_layer(
        model, args.val_jsonl, args.device, base_model,
        patch_size=64, max_samples=min(args.max_val, 100)
    )

    print(f"PSNR: {val_metrics['psnr']:.2f} dB (base: {val_metrics['psnr_base']:.2f} dB)")
    print(f"PSNR Gain: {val_metrics['psnr'] - val_metrics['psnr_base']:+.2f} dB")
    print(f"Dice (patch-based): {val_metrics['dice']:.4f}")
    print("\nPer-layer Dice (targeted evaluation):")
    for name in LAYER_NAMES:
        dice_val = per_layer_metrics.get(name, {}).get('dice', 0)
        print(f"  {name}: {dice_val:.4f}")

    initial_psnr = val_metrics['psnr']
    initial_psnr_gain = val_metrics['psnr'] - val_metrics['psnr_base']

    # Training loop
    # Use mean Dice from per-layer targeted evaluation for proper comparison
    best_dice = np.mean([
        per_layer_metrics.get(name, {}).get('dice', 0)
        for name in LAYER_NAMES
    ])
    best_epoch = 0
    print(f"\nInitial targeted mean Dice: {best_dice:.4f}")

    print("\n" + "=" * 70)
    print("FINE-TUNING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        train_metrics = finetune_epoch(
            model, train_loader, optimizer, args.device, epoch,
            seg_class_weights=seg_class_weights,
            lambda_dice=args.lambda_dice
        )

        # Validate
        val_metrics = validate_finetune(model, val_loader, args.device, base_model)

        # Get accurate per-layer metrics using targeted evaluation
        per_layer_metrics = validate_per_layer(
            model, args.val_jsonl, args.device, base_model,
            patch_size=64, max_samples=min(args.max_val, 100)
        )

        # Update scheduler
        scheduler.step()

        # Print results
        psnr_gain = val_metrics['psnr'] - val_metrics['psnr_base']
        print(f"\nTrain: Seg Loss={train_metrics['seg_loss']:.4f}, Dice={train_metrics['dice']:.4f}")
        print(f"Val:   PSNR={val_metrics['psnr']:.2f} dB (Gain: {psnr_gain:+.2f} dB), Dice={val_metrics['dice']:.4f}")
        print(f"\nPer-layer Dice (targeted):")
        for name in LAYER_NAMES:
            dice_val = per_layer_metrics.get(name, {}).get('dice', 0)
            print(f"  {name}: {dice_val:.4f}")

        # Check PSNR didn't degrade
        psnr_drop = initial_psnr - val_metrics['psnr']
        if psnr_drop > 0.5:
            print(f"\n⚠️ WARNING: PSNR dropped by {psnr_drop:.2f} dB from initial!")

        # Compute mean Dice from per-layer targeted evaluation
        mean_dice_targeted = np.mean([
            per_layer_metrics.get(name, {}).get('dice', 0)
            for name in LAYER_NAMES
        ])

        # Save best dice
        if mean_dice_targeted > best_dice:
            best_dice = mean_dice_targeted
            best_epoch = epoch
            print(f"*** NEW BEST DICE: {best_dice:.4f} ***")

            # Save per-layer dice dict from targeted evaluation
            per_layer_dice_dict = {
                name: per_layer_metrics.get(name, {}).get('dice', 0)
                for name in LAYER_NAMES
            }

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'psnr': val_metrics['psnr'],
                'psnr_gain': psnr_gain,
                'dice': mean_dice_targeted,
                'per_layer_dice': per_layer_dice_dict,
                'original_checkpoint': args.checkpoint,
            }, os.path.join(args.output_dir, 'best_dice_finetuned.pth'))

    # Final summary
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Initial Dice: {original_dice:.4f}")
    print(f"  Final Dice:   {best_dice:.4f} (epoch {best_epoch})")
    print(f"  Improvement:  {best_dice - original_dice:+.4f} ({(best_dice/original_dice - 1)*100:+.1f}%)")
    print(f"\n  Initial PSNR Gain: {initial_psnr_gain:+.2f} dB")
    print(f"  Final PSNR Gain:   {psnr_gain:+.2f} dB")
    print(f"  PSNR Change:       {psnr_gain - initial_psnr_gain:+.2f} dB")
    print(f"\nOutput: {args.output_dir}/best_dice_finetuned.pth")
    print("=" * 70)


if __name__ == '__main__':
    main()
