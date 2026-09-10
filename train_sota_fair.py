#!/usr/bin/env python3
"""
Train SOTA Baselines with Fair Parameter Counts

For TMI paper comparison - ensures all methods have similar model capacity.

Configurations:
- DnCNN: 31 layers, 192 features = ~9.6M params
- Restormer: dim=48, blocks=[2,2,2,2] = ~10.2M params
- Ours: ~9.5M params
"""

import argparse
import json
import os
import sys
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_multitask import MultiTaskOCTDataset
from nsnd.models.dncnn import DnCNN
from nsnd.models.restormer import Restormer
from nsnd.utils.metrics import compute_psnr, compute_ssim


def train_epoch(model, loader, optimizer, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad(set_to_none=True)

        denoised = model(noisy)
        loss = F.mse_loss(denoised, clean)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})

    return total_loss / max(num_batches, 1)


def validate(model, loader, device):
    """Validate model."""
    model.eval()

    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            denoised = model(noisy)

            for i in range(noisy.size(0)):
                c = clean[i, 0].cpu().numpy()
                d = denoised[i, 0].cpu().numpy()
                psnr_list.append(compute_psnr(d, c))
                ssim_list.append(compute_ssim(d, c))

    return {
        'psnr': np.mean(psnr_list),
        'ssim': np.mean(ssim_list),
    }


def main():
    parser = argparse.ArgumentParser(description='Train SOTA baselines with fair params')
    parser.add_argument('--model', choices=['dncnn', 'restormer'], required=True)
    parser.add_argument('--dncnn_layers', type=int, default=31)
    parser.add_argument('--dncnn_features', type=int, default=192)
    parser.add_argument('--restormer_dim', type=int, default=48)
    parser.add_argument('--train_jsonl', required=True)
    parser.add_argument('--val_jsonl', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_train', type=int, default=2000)
    parser.add_argument('--max_val', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"TRAINING SOTA BASELINE: {args.model.upper()}")
    print("=" * 70)

    # Create model
    if args.model == 'dncnn':
        model = DnCNN(
            in_channels=1,
            out_channels=1,
            num_layers=args.dncnn_layers,
            features=args.dncnn_features
        )
        model_name = f"DnCNN-{args.dncnn_layers}L-{args.dncnn_features}F"
    else:  # restormer
        model = Restormer(
            inp_channels=1,
            out_channels=1,
            dim=args.restormer_dim,
            num_blocks=[2, 2, 2, 2],
            num_refinement_blocks=2,
            heads=[1, 2, 4, 8],
        )
        model_name = f"Restormer-dim{args.restormer_dim}"

    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)

    # Save config
    config = {
        'model': args.model,
        'model_name': model_name,
        'num_params': num_params,
        'dncnn_layers': args.dncnn_layers if args.model == 'dncnn' else None,
        'dncnn_features': args.dncnn_features if args.model == 'dncnn' else None,
        'restormer_dim': args.restormer_dim if args.model == 'restormer' else None,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Data
    train_ds = MultiTaskOCTDataset(
        args.train_jsonl, patch_size=64, max_samples=args.max_train,
        random_crop=True, ensure_all_layers=True
    )
    val_ds = MultiTaskOCTDataset(
        args.val_jsonl, patch_size=64, max_samples=args.max_val,
        random_crop=True, ensure_all_layers=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Data: {len(train_ds)} train, {len(val_ds)} val")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training
    best_psnr = 0
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        train_loss = train_epoch(model, train_loader, optimizer, args.device, epoch)
        val_metrics = validate(model, val_loader, args.device)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val PSNR:   {val_metrics['psnr']:.2f} dB")
        print(f"Val SSIM:   {val_metrics['ssim']:.4f}")

        # Log metrics
        metrics_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_psnr': val_metrics['psnr'],
            'val_ssim': val_metrics['ssim'],
        })

        # Save best
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'psnr': best_psnr,
                'ssim': val_metrics['ssim'],
                'config': config,
            }, os.path.join(args.output_dir, 'best.pth'))
            print(f"*** NEW BEST PSNR: {best_psnr:.2f} dB ***")

        # Memory cleanup
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # Save metrics log
    with open(os.path.join(args.output_dir, 'metrics.jsonl'), 'w') as f:
        for m in metrics_log:
            f.write(json.dumps(m) + '\n')

    print("\n" + "=" * 70)
    print(f"{model_name} TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoint: {args.output_dir}/best.pth")
    print("=" * 70)


if __name__ == '__main__':
    main()
