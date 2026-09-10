#!/usr/bin/env python3
"""
Train U-Net with per-epoch PSNR/SSIM monitoring and memory tracking.
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from nsnd_oct.nsnd.models.unet import UNet

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


def primary_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "l2":
        return F.mse_loss(pred, target)
    if loss_type == "charbonnier":
        return charbonnier(pred, target)
    raise ValueError(f"Unknown loss type: {loss_type}")


def evaluate_model(model, dataloader, device):
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


def train_unet(
    train_pairs: str,
    val_pairs: str,
    out_dir: str,
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 1e-3,
    size: int = 64,
    grad_w: float = 0.0,
    log_domain: bool = True,
    use_crop: bool = False,
    features: int = 32,
    loss_type: str = "l2",
    early_stopping: int = 10,
):
    """Train U-Net with monitoring."""

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("U-Net Training with Monitoring")
    print("="*80)
    print(f"Device: {device}")
    print(f"Train pairs: {train_pairs}")
    print(f"Val pairs: {val_pairs}")
    print(f"Output dir: {out_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Patch size: {size} | Use crop: {use_crop}")
    print(f"Loss: {loss_type} (grad_w={grad_w}, log_domain={log_domain})")
    print(f"UNet config: features={features}")
    print(f"Early stopping patience: {early_stopping}")

    model = UNet(in_channels=1, out_channels=1, features=features).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print("="*80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

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

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = []
        for batch_idx, (x_noisy, y_clean) in enumerate(train_loader):
            x_noisy = x_noisy.to(device)
            y_clean = y_clean.to(device)

            pred = model(x_noisy)

            if log_domain:
                pred_t = torch.log(pred.clamp(min=0.01, max=1.0))
                tgt_t = torch.log(y_clean.clamp(min=0.01, max=1.0))
            else:
                pred_t = pred
                tgt_t = y_clean

            loss = primary_loss(pred_t, tgt_t, loss_type)
            if grad_w > 0:
                loss = loss + grad_w * gradient_loss(pred_t, tgt_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss.append(loss.item())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = sum(running_loss[-min(50, len(running_loss)):]) / len(running_loss[-min(50, len(running_loss)):])
                progress = 100.0 * (batch_idx + 1) / len(train_loader)
                mem = get_memory_stats()
                mem_str = f"{mem['allocated_mb']:.1f}MB" if mem['device'] == 'cuda' else 'CPU'
                print(f"  Epoch {epoch:3d}/{epochs} [{batch_idx+1:4d}/{len(train_loader)}] "
                      f"({progress:5.1f}%) loss={avg_loss:.4f} "
                      f"mem={mem_str}", flush=True)

        epoch_loss = sum(running_loss) / len(running_loss)
        val_psnr, val_ssim = evaluate_model(model, val_loader, device)
        scheduler.step()

        history['epochs'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['memory_mb'].append(get_memory_stats())
        history['time_per_epoch'].append(time.time() - epoch_start)

        print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'unet_best.pth'))
            print(f"  ✓ New best model saved! PSNR={best_psnr:.2f} dB")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs (best: {best_psnr:.2f} dB)")
        
        if epochs_no_improve >= early_stopping:
            print(f"\nEarly stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'history': history,
            }, os.path.join(out_dir, f'unet_checkpoint_epoch{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(out_dir, 'unet_final.pth'))
    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Models saved to: {out_dir}")
    print("="*80)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train U-Net with monitoring')
    parser.add_argument('--train_pairs', type=str, required=True,
                       help='Path to training pairs file')
    parser.add_argument('--val_pairs', type=str, required=True,
                       help='Path to validation pairs file')
    parser.add_argument('--out_dir', type=str, default='checkpoints/unet_fair',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--size', type=int, default=64,
                       help='Image size (patches)')
    parser.add_argument('--use_crop', action='store_true',
                       help='Use random crop for train and center crop for val')
    parser.add_argument('--grad_w', type=float, default=0.0,
                       help='Gradient loss weight (paper default: 0)')
    parser.add_argument('--features', type=int, default=32,
                       help='U-Net base features')
    parser.add_argument('--loss', type=str, default='l2', choices=['l1', 'l2', 'charbonnier'],
                       help='Primary loss (paper default: l2)')
    parser.add_argument('--log_domain', action='store_true',
                       help='Use log-domain loss')
    parser.add_argument('--early_stopping', type=int, default=10,
                       help='Early stopping patience (epochs)')

    args = parser.parse_args()

    train_unet(
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
        features=args.features,
        loss_type=args.loss,
        early_stopping=args.early_stopping,
    )


if __name__ == '__main__':
    main()
