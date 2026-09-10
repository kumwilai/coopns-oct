#!/usr/bin/env python3
"""
Train DSP (Differentiable Shortest Path) boundary detection module only.

This script trains the DSP cost network to accurately detect OCT layer boundaries.
Once trained, the DSP model can be used as a starting point for full adaptive denoising training.

Key components:
1. NAFNet backbone (frozen) - extracts features from images
2. DSP Cost Decoder (trainable) - predicts per-pixel costs for boundary detection
3. DSP Boundary Loss - supervises boundary positions against ground truth
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root
sys.path.insert(0, '.')

from train_tmi_enhanced import remap_mask_to_4class


class OCTBoundaryDataset(Dataset):
    """Dataset for DSP boundary training."""

    def __init__(self, jsonl_path, max_samples=None, patch_size=256):
        self.samples = []
        self.patch_size = patch_size

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                entry = json.loads(line)
                self.samples.append(entry)

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(entry['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(entry['mask_path']))

        H, W = image.shape

        # Remap to 4-class
        mask_4class = remap_mask_to_4class(mask)

        # Random crop if larger than patch_size
        if H > self.patch_size or W > self.patch_size:
            # Try to center on retina (middle rows usually contain retina)
            center_h = H // 2
            start_h = max(0, center_h - self.patch_size // 2)
            start_h = min(start_h, H - self.patch_size)

            start_w = np.random.randint(0, max(1, W - self.patch_size))

            image = image[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
            mask_4class = mask_4class[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]

        # Extract ground truth boundaries from mask
        gt_boundaries, valid_mask = self.extract_boundaries(mask_4class)

        # Convert to tensors
        image_t = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]
        mask_t = torch.from_numpy(mask_4class).long()  # [H, W]
        boundaries_t = torch.from_numpy(gt_boundaries).float()  # [4, W]
        valid_mask_t = torch.from_numpy(valid_mask).float()  # [W]

        return image_t, mask_t, boundaries_t, valid_mask_t

    def extract_boundaries(self, mask):
        """Extract boundary positions from segmentation mask.

        Returns:
            boundaries: [4, W] boundary positions (normalized 0-1)
            valid_mask: [W] boolean mask indicating valid columns (all 4 classes present)
        """
        H, W = mask.shape
        boundaries = np.zeros((4, W), dtype=np.float32)
        valid_mask = np.ones(W, dtype=np.float32)

        for col in range(W):
            col_mask = mask[:, col]

            # Check if all 4 classes are present in this column
            classes_present = [len(np.where(col_mask == c)[0]) > 0 for c in range(4)]

            if not all(classes_present):
                # Mark this column as invalid - don't compute loss here
                valid_mask[col] = 0.0
                # Set default values (won't be used in loss)
                boundaries[0, col] = 0.1
                boundaries[1, col] = 0.3
                boundaries[2, col] = 0.5
                boundaries[3, col] = 0.7
                continue

            # Boundary 0: ILM (top of retina, start of class 0)
            rows_0 = np.where(col_mask == 0)[0]
            boundaries[0, col] = rows_0[0] / H

            # Boundary 1: RNFL/INL (start of class 1)
            rows_1 = np.where(col_mask == 1)[0]
            boundaries[1, col] = rows_1[0] / H

            # Boundary 2: INL/IS_OS (start of class 2)
            rows_2 = np.where(col_mask == 2)[0]
            boundaries[2, col] = rows_2[0] / H

            # Boundary 3: IS_OS/RPE (start of class 3)
            rows_3 = np.where(col_mask == 3)[0]
            boundaries[3, col] = rows_3[0] / H

        return boundaries, valid_mask


class SimpleDSPModel(nn.Module):
    """
    Simplified DSP model for boundary detection training.

    Architecture:
    1. Feature extractor (can be NAFNet encoder or simple CNN)
    2. Cost decoder (predicts per-pixel costs for each boundary)
    3. DSP layer (differentiable shortest path)
    """

    def __init__(self, in_channels=1, feature_channels=64, num_boundaries=4):
        super().__init__()

        self.num_boundaries = num_boundaries

        # Simple feature extractor (encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Cost decoder - predicts cost for each boundary at each pixel
        self.cost_decoder = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, num_boundaries, 1),  # [B, num_boundaries, H, W]
        )

        # DSP parameters
        self.temperature = 0.1
        self.smoothness_weight = 1.0
        self.min_gap = 5

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] input image

        Returns:
            boundaries: [B, num_boundaries, W] boundary positions (normalized 0-1)
            costs: [B, num_boundaries, H, W] cost volumes
        """
        B, C, H, W = x.shape

        # Extract features
        features = self.encoder(x)  # [B, 64, H, W]

        # Predict costs
        costs = self.cost_decoder(features)  # [B, num_boundaries, H, W]
        costs = F.softplus(costs)  # Ensure positive costs

        # Run DSP for each boundary
        boundaries = self.run_dsp(costs, H, W)

        return boundaries, costs

    def run_dsp(self, costs, H, W):
        """
        Run differentiable shortest path for boundary detection.

        Args:
            costs: [B, num_boundaries, H, W] cost volumes
            H, W: image dimensions

        Returns:
            boundaries: [B, num_boundaries, W] normalized boundary positions
        """
        B, N, _, _ = costs.shape
        device = costs.device

        all_boundaries = []
        prev_boundary = None

        for n in range(N):
            cost_n = costs[:, n, :, :]  # [B, H, W]

            # Apply ordering constraint
            if prev_boundary is not None:
                # Boundaries must be below previous boundary
                y_coords = torch.arange(H, device=device).float().view(1, H, 1)
                min_valid = prev_boundary.unsqueeze(1) * H + self.min_gap
                invalid_mask = y_coords < min_valid
                cost_n = cost_n + invalid_mask.float() * 1e6

            # Soft-argmin to get boundary position
            # Use softmax with temperature
            weights = F.softmax(-cost_n / self.temperature, dim=1)  # [B, H, W]

            # Compute expected position
            y_coords = torch.arange(H, device=device).float().view(1, H, 1)
            boundary_n = (weights * y_coords).sum(dim=1) / H  # [B, W] normalized

            all_boundaries.append(boundary_n)
            prev_boundary = boundary_n

        boundaries = torch.stack(all_boundaries, dim=1)  # [B, N, W]
        return boundaries


def compute_boundary_loss(pred_boundaries, gt_boundaries, valid_mask, H):
    """
    Compute boundary detection loss with valid mask.

    Args:
        pred_boundaries: [B, 4, W] predicted boundaries (normalized 0-1)
        gt_boundaries: [B, 4, W] ground truth boundaries (normalized 0-1)
        valid_mask: [B, W] mask indicating valid columns (1=valid, 0=invalid)
        H: image height (for MAE in pixels)

    Returns:
        loss: scalar loss
        stats: dictionary with per-boundary MAE
    """
    B, N, W = pred_boundaries.shape

    # Expand valid_mask to match boundaries shape [B, 4, W]
    valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, N, -1)  # [B, 4, W]

    # Masked L1 loss - only on valid columns
    diff = torch.abs(pred_boundaries - gt_boundaries)
    masked_diff = diff * valid_mask_expanded
    valid_count = valid_mask_expanded.sum() + 1e-8
    l1_loss = masked_diff.sum() / valid_count

    # Smoothness loss (on all columns - helps with invalid regions too)
    dx = pred_boundaries[:, :, 1:] - pred_boundaries[:, :, :-1]
    smoothness_loss = (dx ** 2).mean()

    # Ordering loss (boundaries should be in order: 0 < 1 < 2 < 3)
    deltas = pred_boundaries[:, 1:, :] - pred_boundaries[:, :-1, :]
    ordering_loss = F.relu(-deltas + 0.01).mean()  # 0.01 = minimum gap

    # Total loss
    total_loss = l1_loss + 0.5 * smoothness_loss + 0.1 * ordering_loss

    # Compute per-boundary MAE in pixels (only on valid columns)
    with torch.no_grad():
        boundary_names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']
        stats = {
            'loss': total_loss.item(),
            'l1': l1_loss.item(),
            'smooth': smoothness_loss.item(),
            'order': ordering_loss.item(),
            'valid_ratio': valid_mask.mean().item(),
        }
        for i, name in enumerate(boundary_names):
            boundary_diff = torch.abs(pred_boundaries[:, i, :] - gt_boundaries[:, i, :]) * H
            boundary_diff_masked = boundary_diff * valid_mask
            valid_per_boundary = valid_mask.sum() + 1e-8
            stats[f'{name}_mae'] = (boundary_diff_masked.sum() / valid_per_boundary).item()

    return total_loss, stats


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_stats = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, (images, masks, gt_boundaries, valid_mask) in enumerate(pbar):
        images = images.to(device)
        gt_boundaries = gt_boundaries.to(device)
        valid_mask = valid_mask.to(device)

        B, C, H, W = images.shape

        # Forward pass
        pred_boundaries, costs = model(images)

        # Compute loss
        loss, stats = compute_boundary_loss(pred_boundaries, gt_boundaries, valid_mask, H)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate stats
        total_loss += loss.item()
        for k, v in stats.items():
            total_stats[k] = total_stats.get(k, 0) + v

        # Update progress bar
        avg_mae = np.mean([stats[f'{n}_mae'] for n in ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']])
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'MAE': f'{avg_mae:.1f}px',
            'IS/OS': f'{stats["INL_ISOS_mae"]:.1f}px',
        })

    # Average stats
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_stats = {k: v / n_batches for k, v in total_stats.items()}

    return avg_loss, avg_stats


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    total_stats = {}

    with torch.no_grad():
        for images, masks, gt_boundaries in dataloader:
            images = images.to(device)
            gt_boundaries = gt_boundaries.to(device)

            B, C, H, W = images.shape

            # Forward pass
            pred_boundaries, costs = model(images)

            # Compute loss
            loss, stats = compute_boundary_loss(pred_boundaries, gt_boundaries, H)

            total_loss += loss.item()
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_stats = {k: v / n_batches for k, v in total_stats.items()}

    return avg_loss, avg_stats


def main():
    parser = argparse.ArgumentParser(description='Train DSP boundary detection')
    parser.add_argument('--train_jsonl', type=str, default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=500)
    parser.add_argument('--max_val', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='outputs/dsp_pretrained')
    parser.add_argument('--patch_size', type=int, default=256)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DSP Boundary Detection Training")
    print("=" * 60)
    print(f"Train: {args.train_jsonl} (max {args.max_train})")
    print(f"Val: {args.val_jsonl} (max {args.max_val})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create datasets
    train_dataset = OCTBoundaryDataset(args.train_jsonl, args.max_train, args.patch_size)
    val_dataset = OCTBoundaryDataset(args.val_jsonl, args.max_val, args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    model = SimpleDSPModel(in_channels=1, feature_channels=64, num_boundaries=4)
    model = model.to(args.device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    best_val_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

        # Train
        train_loss, train_stats = train_epoch(model, train_loader, optimizer, args.device, epoch)

        # Validate
        val_loss, val_stats = validate(model, val_loader, args.device)

        # Print stats
        train_mae = np.mean([train_stats[f'{n}_mae'] for n in ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']])
        val_mae = np.mean([val_stats[f'{n}_mae'] for n in ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']])

        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f} px")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f} px")
        print(f"  Per-boundary MAE (val):")
        for name in ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']:
            mae = val_stats[f'{name}_mae']
            status = "OK" if mae < 5 else "POOR" if mae < 10 else "BAD"
            print(f"    {name}: {mae:.2f} px [{status}]")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_path = os.path.join(args.output_dir, 'dsp_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_stats': val_stats,
            }, save_path)
            print(f"  -> Saved best model (MAE: {val_mae:.2f} px)")

        scheduler.step()

    # Save final model
    final_path = os.path.join(args.output_dir, 'dsp_final.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mae': val_mae,
    }, final_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Val MAE: {best_val_mae:.2f} px")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
