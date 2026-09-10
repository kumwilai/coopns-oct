#!/usr/bin/env python3
"""
Train SOTA baseline methods for fair comparison.

Trains DnCNN, UNet on the same OCT data for fair comparison in TMI paper.
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

from nsnd.baselines.sota_methods import DnCNN, UNetDenoiser, RestormerLite, SwinIR
from nsnd.utils.metrics import compute_psnr, compute_ssim


class OCTDataset(Dataset):
    """Simple OCT dataset for baseline training."""

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
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)

        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean).unsqueeze(0).float(),
        }


def train_model(model, train_loader, val_loader, device, epochs, lr, save_path, model_name):
    """Train a denoising model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_psnr = 0

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            optimizer.zero_grad(set_to_none=True)
            denoised = model(noisy)
            loss = F.mse_loss(denoised, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_psnr = 0
        val_ssim = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch['noisy'].to(device)
                clean = batch['clean'].to(device)
                denoised = model(noisy).clamp(0, 1)

                for i in range(noisy.size(0)):
                    val_psnr += compute_psnr(denoised[i:i+1], clean[i:i+1])
                    val_ssim += compute_ssim(denoised[i:i+1], clean[i:i+1])
                    n_val += 1

        val_psnr /= n_val
        val_ssim /= n_val

        print(f"  Loss: {train_loss:.6f}, PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'psnr': best_psnr,
            }, save_path)
            print(f"  *** Saved best model: {best_psnr:.2f} dB ***")

        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{model_name} training complete. Best PSNR: {best_psnr:.2f} dB")
    return best_psnr


def main():
    parser = argparse.ArgumentParser(description='Train SOTA baselines')
    parser.add_argument('--train_jsonl', default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_train', type=int, default=1000)
    parser.add_argument('--max_val', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='models/sota_baselines')
    parser.add_argument('--methods', nargs='+', default=['swinir', 'unet'],
                       help='Methods to train: swinir, unet, restormer (dncnn also available but smaller)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("TRAINING SOTA BASELINES FOR TMI COMPARISON")
    print("="*60)

    # Data
    train_ds = OCTDataset(args.train_jsonl, args.patch_size, args.max_train)
    val_ds = OCTDataset(args.val_jsonl, args.patch_size, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nData: {len(train_ds)} train, {len(val_ds)} val")

    results = {}

    # Train SwinIR (Swin Transformer - SOTA for image restoration)
    # Using embed_dim=120, 8 RSTB blocks with 6 Swin blocks each for ~9.6M params (comparable to ours)
    if 'swinir' in args.methods:
        model = SwinIR(in_channels=1, out_channels=1, embed_dim=120, num_heads=8,
                       window_size=8, num_rstb=8, num_blocks=6).to(args.device)
        params = sum(p.numel() for p in model.parameters())
        print(f"\nSwinIR parameters: {params:,}")

        best_psnr = train_model(
            model, train_loader, val_loader, args.device,
            args.epochs, args.lr * 0.5,  # Lower LR for transformer
            os.path.join(args.output_dir, 'swinir_best.pth'),
            'SwinIR'
        )
        results['swinir'] = best_psnr
        del model
        gc.collect()

    # Train DnCNN
    # Note: DnCNN-B with 17 layers is the standard configuration from the original paper
    # Even with feat=192, it's only 5M params. We use feat=128 as a reasonable middle ground.
    if 'dncnn' in args.methods:
        model = DnCNN(in_channels=1, out_channels=1, num_layers=17, num_features=128).to(args.device)
        params = sum(p.numel() for p in model.parameters())
        print(f"\nDnCNN parameters: {params:,} (lightweight baseline)")

        best_psnr = train_model(
            model, train_loader, val_loader, args.device,
            args.epochs, args.lr,
            os.path.join(args.output_dir, 'dncnn_best.pth'),
            'DnCNN'
        )
        results['dncnn'] = best_psnr
        del model
        gc.collect()

    # Train UNet
    if 'unet' in args.methods:
        model = UNetDenoiser(in_channels=1, out_channels=1, base_features=32).to(args.device)
        params = sum(p.numel() for p in model.parameters())
        print(f"\nUNet parameters: {params:,}")

        best_psnr = train_model(
            model, train_loader, val_loader, args.device,
            args.epochs, args.lr,
            os.path.join(args.output_dir, 'unet_best.pth'),
            'UNet'
        )
        results['unet'] = best_psnr
        del model
        gc.collect()

    # Train RestormerLite (optional - slower)
    # Using dim=80 with standard blocks [2,3,3,4] for ~7.3M params (comparable to our model)
    if 'restormer' in args.methods:
        model = RestormerLite(in_channels=1, out_channels=1, dim=80, num_blocks=[2, 3, 3, 4]).to(args.device)
        params = sum(p.numel() for p in model.parameters())
        print(f"\nRestormerLite parameters: {params:,}")

        best_psnr = train_model(
            model, train_loader, val_loader, args.device,
            args.epochs, args.lr * 0.5,  # Lower LR for transformer
            os.path.join(args.output_dir, 'restormer_best.pth'),
            'RestormerLite'
        )
        results['restormer'] = best_psnr
        del model
        gc.collect()

    # Summary
    print("\n" + "="*60)
    print("SOTA BASELINE TRAINING COMPLETE")
    print("="*60)
    print("\nResults Summary:")
    print("-"*40)
    for method, psnr in results.items():
        print(f"  {method.upper():<15}: {psnr:.2f} dB")
    print("-"*40)
    print(f"\nModels saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
