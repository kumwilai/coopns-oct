#!/usr/bin/env python3
"""
Memory-Efficient Training Script for Limited RAM Systems

This script trains with:
- 64x64 patches only
- Batch size 2
- Reduced columnar attention dimensions
- Aggressive memory cleanup
- Gradient checkpointing
"""

import os
import sys
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.nafnet import NAFNet
from nsnd.models.boundary_regression import (
    BoundaryRegressionHead,
    BoundaryLoss,
    extract_boundaries_from_segmentation,
)

# Configuration for memory-limited systems
PATCH_SIZE = 64
BATCH_SIZE = 2
NUM_CLASSES = 4


def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_mb():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


# =============================================================================
# Lightweight Model Components
# =============================================================================
class LightweightColumnarEncoder(nn.Module):
    """
    Memory-efficient columnar encoder using depthwise separable convolutions
    instead of full attention.
    """

    def __init__(self, in_channels=32, dim=32):
        super().__init__()
        self.dim = dim

        # Use depthwise separable convs instead of attention
        self.input_proj = nn.Conv2d(in_channels, dim, 1)

        # Depth-wise (vertical) processing
        self.depth_conv = nn.Sequential(
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Cross-column (horizontal) processing
        self.cross_conv = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Mix channels
        self.mix = nn.Conv2d(dim, dim, 1)
        self.output_proj = nn.Conv2d(dim, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x_proj = self.input_proj(x)
        x_depth = self.depth_conv(x_proj)
        x_cross = self.cross_conv(x_depth)
        x_mix = self.mix(x_cross)
        out = self.output_proj(x_mix) + x

        # Columnar features for boundary regression [B, W, H, dim]
        col_features = x_proj.permute(0, 3, 2, 1)

        return out, col_features


class LightweightSegmenter(nn.Module):
    """Minimal segmenter for 4-class OCT segmentation."""

    def __init__(self, in_channels=1, num_classes=4, base_filters=16):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(base_filters, num_classes, 1)

    def forward(self, x):
        e = self.enc(x)
        d = self.dec(e)
        if d.shape[2:] != x.shape[2:]:
            d = F.interpolate(d, x.shape[2:], mode='bilinear', align_corners=False)
        return self.seg_head(d)


class MemoryEfficientModel(nn.Module):
    """
    Memory-efficient joint denoising + boundary regression model.

    Uses:
    - Lightweight NAFNet (width=32)
    - Depthwise separable columnar processing
    - Direct boundary regression
    """

    def __init__(
        self,
        num_classes=4,
        nafnet_width=32,
        use_boundary_regression=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_boundary_regression = use_boundary_regression

        # Lightweight NAFNet
        self.nafnet = NAFNet(
            img_channel=1,
            width=nafnet_width,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1],
            dec_blk_nums=[1, 1, 1],
        )

        # Lightweight segmenter
        self.segmenter = LightweightSegmenter(
            in_channels=1,
            num_classes=num_classes,
            base_filters=16,
        )

        # Columnar encoder (depthwise separable)
        self.columnar = LightweightColumnarEncoder(
            in_channels=nafnet_width,
            dim=32,
        )

        # Boundary regression head
        if use_boundary_regression:
            self.boundary_head = BoundaryRegressionHead(
                in_dim=32,
                hidden_dim=64,
                num_boundaries=num_classes + 1,
                dropout=0.1,
            )

        # Feature extraction
        self.feature_proj = nn.Conv2d(1, nafnet_width, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Denoising
        denoised = self.nafnet(x)

        # Segmentation
        seg_logits = self.segmenter(denoised)

        # Extract features for columnar processing
        features = self.feature_proj(denoised)

        # Columnar encoding
        enhanced, col_features = self.columnar(features)

        # Boundary regression
        boundary_positions = None
        if self.use_boundary_regression:
            boundary_positions, _ = self.boundary_head(col_features)

        return {
            'denoised': denoised,
            'seg_logits': seg_logits,
            'boundary_positions': boundary_positions,
        }


# =============================================================================
# Dataset
# =============================================================================
class MemoryEfficientDataset(Dataset):
    """Dataset that loads 64x64 patches efficiently."""

    def __init__(self, jsonl_path, max_samples=None, patch_size=64):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(sample['mask_path']))

        H, W = clean.shape

        # Random 64x64 patch
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        clean_patch = clean[top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Add noise
        noisy_patch = self._add_noise(clean_patch)

        # Remap mask to 4-class
        mask_4class = self._remap_mask(mask_patch)

        return {
            'noisy': torch.from_numpy(noisy_patch).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_patch.astype(np.float32)).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_4class).long(),
        }

    def _add_noise(self, image):
        speckle = 1.0 + 0.4 * (np.random.exponential(1.0, image.shape) - 1.0)
        noisy = image * np.maximum(speckle, 0.01)
        gaussian = np.random.randn(*image.shape).astype(np.float32) * 0.08
        noisy = noisy + gaussian
        return np.clip(noisy, 0, 1).astype(np.float32)

    def _remap_mask(self, mask):
        new_mask = np.zeros_like(mask)
        new_mask[mask == 0] = 0
        new_mask[mask == 1] = 1
        new_mask[mask == 2] = 1
        new_mask[mask == 3] = 2
        new_mask[mask == 4] = 3
        return new_mask


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, loader, optimizer, device, boundary_loss_fn):
    model.train()
    total_loss = 0
    total_psnr = 0
    n_batches = 0

    pbar = tqdm(loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        outputs = model(noisy)
        denoised = outputs['denoised']
        seg_logits = outputs['seg_logits']
        boundary_positions = outputs['boundary_positions']

        # Denoising loss
        denoise_loss = F.l1_loss(denoised, clean)

        # Segmentation loss
        seg_loss = F.cross_entropy(seg_logits, mask)

        # Boundary loss
        boundary_loss = torch.tensor(0.0, device=device)
        if boundary_positions is not None:
            gt_boundaries = extract_boundaries_from_segmentation(mask, num_classes=4)
            boundary_loss, _ = boundary_loss_fn(boundary_positions, gt_boundaries)

        loss = denoise_loss + 0.5 * seg_loss + 0.5 * boundary_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # PSNR
        with torch.no_grad():
            mse = F.mse_loss(denoised, clean)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

        total_loss += loss.item()
        total_psnr += psnr.item()
        n_batches += 1

        # CRITICAL: Memory cleanup
        del noisy, clean, mask, outputs, denoised, seg_logits, boundary_positions
        del denoise_loss, seg_loss, boundary_loss, loss

        # Periodic aggressive cleanup
        if (batch_idx + 1) % 10 == 0:
            clear_memory()

        pbar.set_postfix({
            'loss': f'{total_loss/n_batches:.4f}',
            'psnr': f'{total_psnr/n_batches:.2f}',
            'mem': f'{get_memory_mb():.0f}MB',
        })

    return total_loss / n_batches, total_psnr / n_batches


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = 0
    n_batches = 0

    for batch in tqdm(loader, desc='Validating'):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        outputs = model(noisy)
        denoised = outputs['denoised']

        mse = F.mse_loss(denoised, clean)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

        total_psnr += psnr.item()
        n_batches += 1

        del noisy, clean, outputs, denoised

    clear_memory()
    return total_psnr / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='data/train.jsonl')
    parser.add_argument('--val_jsonl', default='data/val.jsonl')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', default='checkpoints/memory_efficient')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Dataset
    train_dataset = MemoryEfficientDataset(
        args.train_jsonl,
        max_samples=args.max_train,
        patch_size=PATCH_SIZE,
    )
    val_dataset = MemoryEfficientDataset(
        args.val_jsonl,
        max_samples=args.max_val,
        patch_size=PATCH_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # No workers to save memory
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = MemoryEfficientModel(
        num_classes=NUM_CLASSES,
        nafnet_width=32,
        use_boundary_regression=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Loss and optimizer
    boundary_loss_fn = BoundaryLoss(num_boundaries=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_psnr = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}, LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_psnr = train_epoch(
            model, train_loader, optimizer, device, boundary_loss_fn
        )
        print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f} dB")

        val_psnr = validate(model, val_loader, device)
        print(f"Val - PSNR: {val_psnr:.2f} dB")

        scheduler.step()

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_psnr,
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"Saved best model (PSNR: {best_psnr:.2f} dB)")

        # Memory report
        print(f"GPU Memory: {get_memory_mb():.0f} MB")
        clear_memory()

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
