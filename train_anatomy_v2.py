#!/usr/bin/env python3
"""
Improved Anatomy-Aware NSAD v2 - Addressing Critical Issues

KEY IMPROVEMENTS:
1. Supervised noise classification loss (fix 13% → 60%+ accuracy)
2. Hard example mining (focus on backbone failure cases)
3. Expert specialization loss (encourage different outputs on different noise)
4. Simplified architecture option (ablation-ready)

Usage:
    python train_anatomy_v2.py --epochs 10 --device cpu
"""

import argparse
import json
import os
import sys
import gc
import psutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.sansd import AnatomyAwareSANSD
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.utils.metrics import compute_psnr, compute_ssim
from nsnd.utils.anatomy_metrics import (
    compute_layer_specific_psnr,
    LAYER_ZONES,
)


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


class OCTDenoiseDataset(Dataset):
    def __init__(self, jsonl_path, patch_size=64, max_samples=None):
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
        noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

        h, w = noisy.shape
        top, left = (h - self.patch_size) // 2, (w - self.patch_size) // 2
        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean).unsqueeze(0).float(),
            'weights': torch.tensor([
                data['weights']['speckle'],
                data['weights']['banding'],
                data['weights']['gaussian'],
                data['weights']['shot']
            ], dtype=torch.float32),
        }


# =============================================================================
# KEY FIX 1: Supervised Noise Classification Loss
# =============================================================================

def noise_classification_loss(pred_noise_type, gt_weights, temperature=2.0):
    """
    Supervised loss for noise type prediction.

    Args:
        pred_noise_type: [B, 4, H, W] per-pixel predictions
        gt_weights: [B, 4] ground truth noise mixture weights
        temperature: softmax temperature for soft labels

    Returns:
        Cross-entropy loss between predicted and ground truth
    """
    # Global prediction: average over spatial dimensions
    pred_global = pred_noise_type.mean(dim=[2, 3])  # [B, 4]

    # Convert GT weights to soft labels
    gt_soft = F.softmax(gt_weights / temperature, dim=1)

    # KL divergence (soft cross-entropy)
    pred_log = F.log_softmax(pred_global, dim=1)
    loss = F.kl_div(pred_log, gt_soft, reduction='batchmean')

    return loss


# =============================================================================
# KEY FIX 2: Hard Example Mining
# =============================================================================

def hard_example_weighted_loss(denoised, clean, backbone_out, hard_weight=5.0):
    """
    Focus training on regions where backbone fails.

    Args:
        denoised: our output [B, 1, H, W]
        clean: ground truth [B, 1, H, W]
        backbone_out: frozen backbone output [B, 1, H, W]
        hard_weight: extra weight for hard regions

    Returns:
        Weighted MSE loss focusing on hard cases
    """
    # Backbone error map
    backbone_error = (backbone_out - clean).abs()

    # Identify hard regions (top 30% error)
    B = backbone_error.shape[0]
    threshold = backbone_error.view(B, -1).quantile(0.7, dim=1, keepdim=True)
    threshold = threshold.view(B, 1, 1, 1)

    hard_mask = (backbone_error > threshold).float()

    # Weight: 1.0 for easy, hard_weight for hard
    weight = 1.0 + (hard_weight - 1.0) * hard_mask

    # Weighted MSE
    loss = (weight * (denoised - clean).pow(2)).mean()

    return loss, hard_mask.mean().item()  # Return hard ratio for logging


# =============================================================================
# KEY FIX 3: Expert Specialization Loss
# =============================================================================

def expert_specialization_loss(expert_outputs, noise_type, gt_weights):
    """
    Encourage experts to specialize on their designated noise type.

    The expert for the dominant noise type should contribute most.

    Args:
        expert_outputs: dict of expert outputs {name: [B, 1, H, W]}
        noise_type: predicted routing [B, 4, H, W]
        gt_weights: ground truth [B, 4]

    Returns:
        Loss encouraging correct expert to be weighted highest
    """
    expert_names = ['speckle', 'banding', 'gaussian', 'shot']

    # Get dominant GT noise type per sample
    gt_dominant = gt_weights.argmax(dim=1)  # [B]

    # Get predicted dominant per pixel, then global
    pred_dominant = noise_type.mean(dim=[2, 3]).argmax(dim=1)  # [B]

    # Simple cross-entropy on dominant class
    pred_logits = noise_type.mean(dim=[2, 3])  # [B, 4]
    loss = F.cross_entropy(pred_logits, gt_dominant)

    return loss


# =============================================================================
# KEY FIX 4: Diversity Without Forcing Balance
# =============================================================================

def output_diversity_loss(expert_outputs):
    """
    Encourage experts to produce DIFFERENT outputs.
    Unlike balance loss, this doesn't force equal usage.
    """
    outputs = torch.stack(list(expert_outputs.values()), dim=0)  # [4, B, 1, H, W]

    # Compute variance across experts for each pixel
    variance = outputs.var(dim=0)  # [B, 1, H, W]

    # We want HIGH variance = experts produce different things
    # Minimize negative variance
    loss = -variance.mean()

    del outputs
    return loss


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, args):
    model.train()

    metrics = {k: 0.0 for k in ['loss', 'recon', 'cls', 'hard', 'spec', 'div']}
    hard_ratio_sum = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        gt_weights = batch['weights'].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward
        denoised, interp = model(noisy, return_interpretation=True)
        backbone_out = interp['backbone_out']
        noise_type = interp['noise_type']
        expert_outputs = interp['expert_outputs']

        # =================================================================
        # LOSS COMPUTATION
        # =================================================================

        # 1. Hard example weighted reconstruction (KEY FIX 2)
        recon_loss, hard_ratio = hard_example_weighted_loss(
            denoised, clean, backbone_out, hard_weight=args.hard_weight
        )

        # 2. Supervised noise classification (KEY FIX 1)
        cls_loss = noise_classification_loss(noise_type, gt_weights)

        # 3. Expert specialization (KEY FIX 3)
        spec_loss = expert_specialization_loss(expert_outputs, noise_type, gt_weights)

        # 4. Output diversity (KEY FIX 4)
        div_loss = output_diversity_loss(expert_outputs)

        # Total loss
        total = (
            recon_loss +
            args.lambda_cls * cls_loss +
            args.lambda_spec * spec_loss +
            args.lambda_div * div_loss
        )

        # Track before backward
        metrics['loss'] += total.item()
        metrics['recon'] += recon_loss.item()
        metrics['cls'] += cls_loss.item()
        metrics['spec'] += spec_loss.item()
        metrics['div'] += div_loss.item()
        hard_ratio_sum += hard_ratio
        num_batches += 1

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Cleanup
        del noisy, clean, gt_weights, denoised, interp, backbone_out
        del noise_type, expert_outputs, recon_loss, cls_loss, spec_loss, div_loss, total

        pbar.set_postfix({
            'Loss': f'{metrics["loss"]/num_batches:.4f}',
            'Cls': f'{metrics["cls"]/num_batches:.3f}',
            'Hard': f'{hard_ratio_sum/num_batches:.2f}',
        })

    return {k: v/num_batches for k, v in metrics.items()}


def validate(model, val_loader, device, epoch, base_model):
    model.eval()
    base_model.eval()

    total_psnr_base = 0
    total_psnr_ours = 0
    total_ssim_ours = 0
    correct = 0
    total = 0

    # Per-class tracking
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]

    expert_usage = torch.zeros(4)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            gt_weights = batch['weights'].to(device)

            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            denoised, interp = model(noisy, return_interpretation=True)

            for i in range(noisy.size(0)):
                total_psnr_base += compute_psnr(base_out[i:i+1], clean[i:i+1])
                total_psnr_ours += compute_psnr(denoised[i:i+1], clean[i:i+1])
                total_ssim_ours += compute_ssim(denoised[i:i+1], clean[i:i+1])

                # Classification accuracy
                gt_cls = gt_weights[i].argmax().item()
                pred_cls = interp['global_weights'][i].argmax().item()

                class_total[gt_cls] += 1
                if gt_cls == pred_cls:
                    correct += 1
                    class_correct[gt_cls] += 1

                total += 1

            expert_usage += interp['noise_type'].mean(dim=[0, 2, 3]).cpu()

            del noisy, clean, gt_weights, base_out, denoised, interp

    n_batches = len(val_loader)
    expert_usage = expert_usage / n_batches

    return {
        'psnr_base': total_psnr_base / total,
        'psnr': total_psnr_ours / total,
        'ssim': total_ssim_ours / total,
        'accuracy': 100 * correct / total,
        'class_acc': [100 * class_correct[i] / max(class_total[i], 1) for i in range(4)],
        'class_total': class_total,
        'expert_usage': expert_usage,
    }


def print_report(train_metrics, val_metrics, epoch):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch} SUMMARY")
    print(f"{'='*60}")

    print(f"\nTraining:")
    print(f"  Recon Loss: {train_metrics['recon']:.4f}")
    print(f"  Classification Loss: {train_metrics['cls']:.4f}")
    print(f"  Specialization Loss: {train_metrics['spec']:.4f}")
    print(f"  Diversity Loss: {train_metrics['div']:.4f}")

    print(f"\nValidation:")
    print(f"  PSNR (base):  {val_metrics['psnr_base']:.2f} dB")
    print(f"  PSNR (ours):  {val_metrics['psnr']:.2f} dB")
    gain = val_metrics['psnr'] - val_metrics['psnr_base']
    print(f"  PSNR GAIN:    {'+' if gain >= 0 else ''}{gain:.2f} dB")
    print(f"  SSIM:         {val_metrics['ssim']:.4f}")

    print(f"\n  NOISE CLASSIFICATION: {val_metrics['accuracy']:.1f}%")
    names = ['speckle', 'banding', 'gaussian', 'shot']
    for i, name in enumerate(names):
        acc = val_metrics['class_acc'][i]
        cnt = val_metrics['class_total'][i]
        bar = '#' * int(acc / 5)
        print(f"    {name:10s}: {acc:5.1f}% ({cnt:2d} samples) {bar}")

    print(f"\n  EXPERT USAGE:")
    for i, name in enumerate(names):
        usage = val_metrics['expert_usage'][i].item()
        bar = '#' * int(usage * 40)
        print(f"    {name:10s}: {usage:.3f} {bar}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--train_jsonl', default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--max_val', type=int, default=30)

    # Model
    parser.add_argument('--base_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')

    # Loss weights (KEY HYPERPARAMETERS)
    parser.add_argument('--hard_weight', type=float, default=5.0, help='Weight for hard examples')
    parser.add_argument('--lambda_cls', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--lambda_spec', type=float, default=0.5, help='Specialization loss weight')
    parser.add_argument('--lambda_div', type=float, default=0.1, help='Diversity loss weight')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output_dir', default='checkpoints/anatomy_v2')
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ANATOMY-AWARE NSAD v2 - FIXING CRITICAL ISSUES")
    print("=" * 60)
    print("\nKEY FIXES:")
    print(f"  1. Supervised noise classification (lambda={args.lambda_cls})")
    print(f"  2. Hard example mining (weight={args.hard_weight}x)")
    print(f"  3. Expert specialization loss (lambda={args.lambda_spec})")
    print(f"  4. Output diversity (lambda={args.lambda_div})")
    print("=" * 60)

    # Data
    train_dataset = OCTDenoiseDataset(args.train_jsonl, args.patch_size, args.max_train)
    val_dataset = OCTDenoiseDataset(args.val_jsonl, args.patch_size, args.max_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nData: {len(train_dataset)} train, {len(val_dataset)} val")

    # Model
    model = AnatomyAwareSANSD(
        backbone_width=64,
        backbone_ckpt=args.base_ckpt,
        fusion_mode='anatomy',
    ).to(args.device)

    # Frozen baseline
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)

    if os.path.exists(args.base_ckpt):
        ckpt = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        base_model.load_state_dict(state, strict=False)

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    # Optimizer
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.01},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_psnr = 0
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        gc.collect()

        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch, args)
        val_metrics = validate(model, val_loader, args.device, epoch, base_model)

        print_report(train_metrics, val_metrics, epoch)

        # Save best
        if val_metrics['psnr'] > best_psnr or val_metrics['accuracy'] > best_acc:
            best_psnr = max(best_psnr, val_metrics['psnr'])
            best_acc = max(best_acc, val_metrics['accuracy'])

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': val_metrics['psnr'],
                'accuracy': val_metrics['accuracy'],
            }, os.path.join(args.output_dir, 'best_model.pth'))

            print(f"\n*** BEST MODEL: PSNR={val_metrics['psnr']:.2f} dB, Acc={val_metrics['accuracy']:.1f}% ***")

        scheduler.step()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best Classification Accuracy: {best_acc:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
