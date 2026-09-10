"""
Pretrain NAFNet Backbone (width=40) on PKU37 Real Noise Dataset

This script trains a lightweight NAFNet backbone for use with the V8 Enhanced corrector.
Uses real OCT noise from the PKU37 dataset for realistic denoising performance.

Key Features:
- NAFNet with width=40 (lightweight to leave room for corrector improvements)
- PKU37 real noise dataset (33 training + 4 validation samples)
- PSNR and SSIM metric tracking
- Best model checkpoint saving
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'nsnd_oct'))

import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

from nsnd.models.nafnet import NAFNetSmall, count_parameters


class PKU37Dataset(Dataset):
    """PKU37 Real Noise OCT Dataset from JSONL file"""

    def __init__(self, jsonl_path, patch_size=128, augment=True):
        self.pairs = []
        self.patch_size = patch_size
        self.augment = augment

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.pairs.append((entry['clean_path'], entry['noisy_path']))

        print(f"Loaded {len(self.pairs)} image pairs from {jsonl_path}")

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        """Load image from TIF file"""
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        # Normalize to [0, 1]
        if arr.max() > 1:
            arr = arr / 255.0
        return arr

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]

        clean = self._load_image(clean_path)
        noisy = self._load_image(noisy_path)

        H, W = clean.shape

        # Random crop for training
        if H >= self.patch_size and W >= self.patch_size:
            top = np.random.randint(0, H - self.patch_size + 1)
            left = np.random.randint(0, W - self.patch_size + 1)
            clean = clean[top:top+self.patch_size, left:left+self.patch_size]
            noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]

        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                clean = np.fliplr(clean).copy()
                noisy = np.fliplr(noisy).copy()

            # Vertical flip
            if np.random.rand() > 0.5:
                clean = np.flipud(clean).copy()
                noisy = np.flipud(noisy).copy()

            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            if k > 0:
                clean = np.rot90(clean, k).copy()
                noisy = np.rot90(noisy, k).copy()

        clean = torch.from_numpy(clean.copy()).unsqueeze(0).float()
        noisy = torch.from_numpy(noisy.copy()).unsqueeze(0).float()

        return noisy, clean


class PKU37ValDataset(Dataset):
    """PKU37 Validation Dataset (no augmentation, full images)"""

    def __init__(self, jsonl_path):
        self.pairs = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.pairs.append((entry['clean_path'], entry['noisy_path']))

        print(f"Loaded {len(self.pairs)} validation pairs from {jsonl_path}")

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        """Load image from TIF file"""
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        # Normalize to [0, 1]
        if arr.max() > 1:
            arr = arr / 255.0
        return arr

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]

        clean = self._load_image(clean_path)
        noisy = self._load_image(noisy_path)

        clean = torch.from_numpy(clean.copy()).unsqueeze(0).float()
        noisy = torch.from_numpy(noisy.copy()).unsqueeze(0).float()

        return noisy, clean


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM between two images"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def medical_loss(pred, target, ssim_weight=0.6):
    """Medical imaging loss: SSIM (60%) + MSE (40%)"""
    # MSE loss
    mse_loss = F.mse_loss(pred, target)

    # SSIM loss
    ssim_val = compute_ssim(pred, target)
    ssim_loss = 1 - ssim_val

    return ssim_weight * ssim_loss + (1 - ssim_weight) * mse_loss


def evaluate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Pad to multiple of 16 for NAFNet
            _, _, H, W = noisy.shape
            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16
            if pad_h > 0 or pad_w > 0:
                noisy = F.pad(noisy, (0, pad_w, 0, pad_h), mode='reflect')

            denoised = model(noisy)

            # Unpad
            if pad_h > 0 or pad_w > 0:
                denoised = denoised[:, :, :H, :W]

            # PSNR
            mse = F.mse_loss(denoised, clean)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            psnr_values.append(psnr.item())

            # SSIM
            ssim = compute_ssim(denoised, clean)
            ssim_values.append(ssim.item())

    return np.mean(psnr_values), np.mean(ssim_values)


def main():
    print("=" * 90)
    print("NAFNet BACKBONE PRETRAINING (width=40) on PKU37 Real Noise")
    print("=" * 90)
    print()
    print("Purpose: Train lightweight backbone for V8 Enhanced corrector")
    print("Dataset: PKU37 Real OCT Noise (33 train + 4 val samples)")
    print("=" * 90)
    print()

    # Configuration
    WIDTH = 40
    EPOCHS = 50
    BATCH_SIZE = 4
    PATCH_SIZE = 128
    LR = 1e-3
    PATIENCE = 15
    SEED = 42

    TRAIN_JSONL = "/home/kumwilai/OCT/pku37_train.jsonl"
    VAL_JSONL = "/home/kumwilai/OCT/pku37_val.jsonl"
    OUTPUT_DIR = Path("/home/kumwilai/OCT/outputs/nafnet_pku37_w40")

    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\n" + "=" * 90)
    print("LOADING DATASETS")
    print("=" * 90)

    train_dataset = PKU37Dataset(TRAIN_JSONL, patch_size=PATCH_SIZE, augment=True)
    val_dataset = PKU37ValDataset(VAL_JSONL)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full images for validation
        shuffle=False,
        num_workers=0
    )

    # Create model
    print("\n" + "=" * 90)
    print("CREATING NAFNet MODEL")
    print("=" * 90)

    model = NAFNetSmall(width=WIDTH).to(device)
    num_params = count_parameters(model)
    print(f"NAFNet Small (width={WIDTH})")
    print(f"  - Total parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    # Training loop
    print("\n" + "=" * 90)
    print("TRAINING")
    print("=" * 90)
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Patch size: {PATCH_SIZE}")
    print(f"Optimizer: AdamW (lr={LR}, weight_decay=1e-4)")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=15, T_mult=2)")
    print(f"Loss: Medical loss (60% SSIM + 40% MSE)")
    print(f"Early stopping patience: {PATIENCE}")
    print("=" * 90)

    best_psnr = 0
    best_ssim = 0
    no_improve = 0
    training_log = []

    # Initial evaluation
    val_psnr, val_ssim = evaluate(model, val_loader, device)
    print(f"\nBefore training: PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_psnr = []
        train_ssim = []
        train_losses = []

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            denoised = model(noisy)
            loss = medical_loss(denoised, clean)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())

            # Training metrics
            with torch.no_grad():
                mse = F.mse_loss(denoised, clean)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                train_psnr.append(psnr.item())

                ssim = compute_ssim(denoised, clean)
                train_ssim.append(ssim.item())

        scheduler.step()

        # Validation
        val_psnr, val_ssim = evaluate(model, val_loader, device)

        # Track best
        improved = val_psnr > best_psnr
        if improved:
            best_psnr = val_psnr
            best_ssim = val_ssim
            no_improve = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': best_ssim,
                'width': WIDTH,
                'config': {
                    'width': WIDTH,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'patch_size': PATCH_SIZE,
                    'lr': LR,
                }
            }, OUTPUT_DIR / 'best_model.pth')
            marker = " [BEST]"
        else:
            no_improve += 1
            marker = f" (no improve: {no_improve}/{PATIENCE})"

        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': np.mean(train_losses),
            'train_psnr': np.mean(train_psnr),
            'train_ssim': np.mean(train_ssim),
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log.append(log_entry)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {np.mean(train_psnr):.2f} dB, {np.mean(train_ssim):.4f} | "
              f"Val: {val_psnr:.2f} dB, {val_ssim:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.0f}s{marker}")

        # Early stopping
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save training log
    with open(OUTPUT_DIR / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 90)
    print("TRAINING COMPLETE")
    print("=" * 90)
    print(f"Best Validation PSNR: {best_psnr:.4f} dB")
    print(f"Best Validation SSIM: {best_ssim:.4f}")
    print(f"Total training time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
    print(f"Model saved to: {OUTPUT_DIR / 'best_model.pth'}")
    print("=" * 90)

    # Save final metrics
    final_metrics = {
        'best_psnr': best_psnr,
        'best_ssim': best_ssim,
        'total_epochs': len(training_log),
        'total_time_seconds': total_time,
        'model_params': num_params,
        'width': WIDTH,
    }
    with open(OUTPUT_DIR / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    return best_psnr, best_ssim


if __name__ == '__main__':
    main()
