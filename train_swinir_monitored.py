#!/usr/bin/env python3
"""
Train SwinIR with per-epoch PSNR/SSIM monitoring and memory tracking.
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add sota models to path
sys.path.append('sota/models')
from swinir_fair import SwinIR

from adaptive_oct_denoise import (
    PairedOCTDataset,
    resize_to,
    compute_psnr,
    compute_ssim,
)


def get_memory_stats():
    """Get GPU memory statistics in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated,
            'device': 'cuda'
        }
    return {
        'allocated_mb': 0,
        'reserved_mb': 0,
        'max_allocated_mb': 0,
        'device': 'cpu'
    }


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss (smooth L1)."""
    return torch.mean(torch.sqrt((x - y) ** 2 + eps * eps))


def primary_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "l2":
        return F.mse_loss(pred, target)
    if loss_type == "charbonnier":
        return charbonnier(pred, target)
    raise ValueError(f"Unknown loss type: {loss_type}")


_SOBEL_KERNELS = None

def gradient_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gradient loss using Sobel filters."""
    global _SOBEL_KERNELS
    
    if _SOBEL_KERNELS is None:
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        _SOBEL_KERNELS = (sobel_x, sobel_y)
    
    sobel_x, sobel_y = _SOBEL_KERNELS
    
    # Ensure kernels are on the correct device and have correct dtype
    if sobel_x.device != x.device or sobel_x.dtype != x.dtype:
        sobel_x = sobel_x.to(device=x.device, dtype=x.dtype)
        sobel_y = sobel_y.to(device=x.device, dtype=x.dtype)
        _SOBEL_KERNELS = (sobel_x, sobel_y)

    gx_p = F.conv2d(x, sobel_x, padding=1)
    gy_p = F.conv2d(x, sobel_y, padding=1)
    gx_t = F.conv2d(y, sobel_x, padding=1)
    gy_t = F.conv2d(y, sobel_y, padding=1)
    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


def evaluate_model(model, dataloader, device, log_domain=True):
    """Evaluate model on validation set."""
    model.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for x_noisy, y_clean in dataloader:
            x_noisy = x_noisy.to(device)
            pred = model(x_noisy).detach().cpu()

            psnr_list.append(compute_psnr(pred, y_clean))
            ssim_list.append(compute_ssim(pred, y_clean))

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    return avg_psnr, avg_ssim


def train_swinir(
    train_pairs: str,
    val_pairs: str,
    out_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 5e-4,
    size: int = 64,
    grad_w: float = 0.0,
    log_domain: bool = True,
    use_crop: bool = False,
    loss_type: str = "l1",
    embed_dim: int = 48,
    depths: list = None,
    num_heads: list = None,
    window_size: int = 8,
    early_stopping: int = 10,
):
    """Train SwinIR with monitoring."""

    # Default architecture
    if depths is None:
        depths = [2, 2, 2]
    if num_heads is None:
        num_heads = [3, 3, 3]

    # Setup
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("SwinIR Training with Monitoring")
    print("="*80)
    print(f"Device: {device}")
    print(f"Train pairs: {train_pairs}")
    print(f"Val pairs: {val_pairs}")
    print(f"Output dir: {out_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Patch size: {size} | Use crop: {use_crop}")
    print(f"Loss: {loss_type} (grad_w={grad_w}, log_domain={log_domain})")
    print(f"SwinIR config: embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}, window={window_size}")
    print(f"Early stopping patience: {early_stopping}")

    # Model
    model = SwinIR(
        img_size=size,
        patch_size=1,
        in_chans=1,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=2.,
        upscale=1,  # No upscaling for denoising
        img_range=1.,
        upsampler=None,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print("="*80)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    # Datasets
    if use_crop:
        train_ds = PairedOCTDataset(train_pairs, crop_size=size, random_crop=True)
        val_ds = PairedOCTDataset(val_pairs, crop_size=size, random_crop=False)
    else:
        tfm = resize_to((size, size))
        train_ds = PairedOCTDataset(train_pairs, transform=tfm)
        val_ds = PairedOCTDataset(val_pairs, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("="*80)

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Training history
    history = {
        'epochs': [],
        'train_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'memory_mb': [],
        'time_per_epoch': []
    }

    best_psnr = 0.0
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = []
        for batch_idx, (x_noisy, y_clean) in enumerate(train_loader):
            x_noisy = x_noisy.to(device)
            y_clean = y_clean.to(device)

            # Forward
            pred = model(x_noisy)

            # Loss
            if log_domain:
                pred_t = torch.log(pred.clamp(min=0.01, max=1.0))
                tgt_t = torch.log(y_clean.clamp(min=0.01, max=1.0))
            else:
                pred_t = pred
                tgt_t = y_clean

            loss = primary_loss(pred_t, tgt_t, loss_type)
            if grad_w > 0:
                loss = loss + grad_w * gradient_loss(pred_t, tgt_t)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss.append(loss.item())

            # Progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = sum(running_loss[-min(50, len(running_loss)):]) / len(running_loss[-min(50, len(running_loss)):])
                progress = 100.0 * (batch_idx + 1) / len(train_loader)
                mem = get_memory_stats()
                mem_str = f"{mem['allocated_mb']:.1f}MB" if mem['device'] == 'cuda' else 'CPU'
                print(f"  Epoch {epoch:3d}/{epochs} [{batch_idx+1:4d}/{len(train_loader)}] "
                      f"({progress:5.1f}%) loss={avg_loss:.4f} "
                      f"mem={mem_str}", flush=True)

        epoch_loss = sum(running_loss) / len(running_loss)

        # Evaluate
        val_psnr, val_ssim = evaluate_model(model, val_loader, device, log_domain)

        # Step scheduler
        scheduler.step()

        # Memory stats
        mem = get_memory_stats()
        epoch_time = time.time() - epoch_start

        # Save history
        history['epochs'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['memory_mb'].append(mem['max_allocated_mb'])
        history['time_per_epoch'].append(epoch_time)

        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*80}")
        print(f"Epoch {epoch:3d}/{epochs} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val PSNR:   {val_psnr:.2f} dB")
        print(f"  Val SSIM:   {val_ssim:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        if mem['device'] == 'cuda':
            print(f"  Memory:     {mem['allocated_mb']:.1f}MB allocated, "
                  f"{mem['reserved_mb']:.1f}MB reserved, "
                  f"{mem['max_allocated_mb']:.1f}MB peak")
        else:
            print(f"  Memory:     Running on CPU (no GPU memory tracked)")
        print(f"  Time:       {epoch_time:.1f}s")
        print(f"{'='*80}\n")

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'swinir_best.pth'))
            print(f"  ✓ New best model saved! PSNR={best_psnr:.2f} dB\n")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs (best: {best_psnr:.2f} dB)")
            
        if epochs_no_improve >= early_stopping:
            print(f"\nEarly stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'history': history,
            }, os.path.join(out_dir, f'swinir_checkpoint_epoch{epoch}.pth'))

    # Save final model and history
    torch.save(model.state_dict(), os.path.join(out_dir, 'swinir_final.pth'))

    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Models saved to: {out_dir}")
    print("="*80)

    return history


def main():
    def _parse_int_list(value: str | None):
        if not value:
            return None
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return None
        return [int(p) for p in parts]

    parser = argparse.ArgumentParser(description='Train SwinIR with monitoring')
    parser.add_argument('--train_pairs', type=str, required=True,
                       help='Path to training pairs file')
    parser.add_argument('--val_pairs', type=str, required=True,
                       help='Path to validation pairs file')
    parser.add_argument('--out_dir', type=str, default='checkpoints/swinir_fair_universal',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--size', type=int, default=64,
                       help='Image size (patches)')
    parser.add_argument('--use_crop', action='store_true',
                       help='Use random crop for train and center crop for val')
    parser.add_argument('--grad_w', type=float, default=0.0,
                       help='Gradient loss weight')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2', 'charbonnier'],
                       help='Primary loss (paper default: l1)')
    parser.add_argument('--embed_dim', type=int, default=48,
                       help='SwinIR embedding dimension')
    parser.add_argument('--depths', type=str, default="",
                       help='Comma-separated SwinIR block depths (e.g., 8,8,8)')
    parser.add_argument('--num_heads', type=str, default="",
                       help='Comma-separated SwinIR head counts (e.g., 6,6,6)')
    parser.add_argument('--window_size', type=int, default=8,
                       help='SwinIR window size')
    parser.add_argument('--log_domain', action='store_true',
                       help='Use log-domain loss')
    parser.add_argument('--early_stopping', type=int, default=10,
                       help='Early stopping patience (epochs)')

    args = parser.parse_args()

    depths = _parse_int_list(args.depths)
    num_heads = _parse_int_list(args.num_heads)

    train_swinir(
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        size=args.size,
        grad_w=args.grad_w,
        log_domain=args.log_domain,
        use_crop=args.use_crop,
        loss_type=args.loss,
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=args.window_size,
        early_stopping=args.early_stopping,
    )


if __name__ == '__main__':
    main()
