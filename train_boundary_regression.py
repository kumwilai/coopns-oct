#!/usr/bin/env python3
"""
Boundary Regression for OCT Layer Segmentation

Instead of pixel-wise classification (which suffers from class imbalance),
this approach directly predicts the 4 boundary curves:
  - Boundary 0: ILM (top of retina)
  - Boundary 1: RNFL/INL junction
  - Boundary 2: INL/IS_OS junction
  - Boundary 3: IS_OS/RPE junction

Layers are derived from boundaries:
  - Layer 0 (RNFL_GCL): ILM to RNFL/INL
  - Layer 1 (INL_OPL_ONL): RNFL/INL to INL/IS_OS
  - Layer 2 (IS_OS): INL/IS_OS to IS_OS/RPE
  - Layer 3 (RPE_Choroid): IS_OS/RPE to bottom

Advantages:
  - No class imbalance (predicting 4 curves, not classifying millions of pixels)
  - Naturally enforces layer ordering via loss function
  - Directly models the anatomical structure
  - Matches how clinical OCT software works
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class BoundaryDataset(Dataset):
    """Dataset that extracts boundary positions from segmentation masks."""

    def __init__(self, jsonl_path, max_samples=None, patch_size=256, augment=False):
        self.samples = []
        self.patch_size = patch_size
        self.augment = augment

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.samples.append(entry)
                if max_samples and len(self.samples) >= max_samples:
                    break

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def extract_boundaries(self, mask):
        """
        Extract 4 boundary positions from segmentation mask.

        For each column x, find the y-positions where layers transition:
        - Boundary 0 (ILM): First non-background pixel (top of layer 0)
        - Boundary 1: Transition from layer 0 to layer 1
        - Boundary 2: Transition from layer 1 to layer 2
        - Boundary 3: Transition from layer 2 to layer 3

        Returns: (4, W) array of boundary y-positions
        """
        H, W = mask.shape
        boundaries = np.zeros((4, W), dtype=np.float32)
        valid_mask = np.ones((4, W), dtype=np.float32)

        for x in range(W):
            col = mask[:, x]

            # Find first non-background (layer 0, 1, 2, or 3)
            retina_mask = col >= 0  # All layers are valid
            if not np.any(retina_mask):
                # No retina in this column - mark as invalid
                valid_mask[:, x] = 0
                boundaries[:, x] = H // 2  # Default to middle
                continue

            # Boundary 0 (ILM): Top of retina (first pixel of layer 0)
            layer0_mask = col == 0
            if np.any(layer0_mask):
                boundaries[0, x] = np.where(layer0_mask)[0][0]
            else:
                # Layer 0 missing - use first non-background
                boundaries[0, x] = np.where(retina_mask)[0][0]
                valid_mask[0, x] = 0.5  # Partial validity

            # Boundary 1: RNFL/INL junction (first pixel of layer 1)
            layer1_mask = col == 1
            if np.any(layer1_mask):
                boundaries[1, x] = np.where(layer1_mask)[0][0]
            else:
                # Layer 1 missing - interpolate or use layer 2 start
                layer2_mask = col == 2
                if np.any(layer2_mask):
                    boundaries[1, x] = np.where(layer2_mask)[0][0]
                else:
                    boundaries[1, x] = boundaries[0, x] + 10  # Estimate
                valid_mask[1, x] = 0.5

            # Boundary 2: INL/IS_OS junction (first pixel of layer 2)
            layer2_mask = col == 2
            if np.any(layer2_mask):
                boundaries[2, x] = np.where(layer2_mask)[0][0]
            else:
                valid_mask[2, x] = 0.5
                # Use midpoint between boundary 1 and 3
                layer3_mask = col == 3
                if np.any(layer3_mask):
                    boundaries[2, x] = np.where(layer3_mask)[0][0] - 5
                else:
                    boundaries[2, x] = boundaries[1, x] + 10

            # Boundary 3: IS_OS/RPE junction (first pixel of layer 3)
            layer3_mask = col == 3
            if np.any(layer3_mask):
                boundaries[3, x] = np.where(layer3_mask)[0][0]
            else:
                valid_mask[3, x] = 0.5
                boundaries[3, x] = boundaries[2, x] + 5

            # Enforce ordering: boundary[i] <= boundary[i+1]
            for i in range(1, 4):
                if boundaries[i, x] < boundaries[i-1, x]:
                    boundaries[i, x] = boundaries[i-1, x] + 1

        return boundaries, valid_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask (support both data formats)
        image_key = 'image_path' if 'image_path' in sample else 'noisy'
        mask_key = 'mask_path' if 'mask_path' in sample else 'seg_mask'

        image = np.array(Image.open(sample[image_key]).convert('L'), dtype=np.float32)
        mask = np.array(Image.open(sample[mask_key]), dtype=np.int64)

        H, W = image.shape

        # Extract patch if image is larger than patch_size
        if H > self.patch_size or W > self.patch_size:
            # Random crop for training, center crop for validation
            if self.augment:
                y_start = np.random.randint(0, max(1, H - self.patch_size))
                x_start = np.random.randint(0, max(1, W - self.patch_size))
            else:
                y_start = (H - self.patch_size) // 2
                x_start = (W - self.patch_size) // 2

            image = image[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
            mask = mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

        # Pad if smaller than patch_size
        H, W = image.shape
        if H < self.patch_size or W < self.patch_size:
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='edge')

        # Augmentation
        if self.augment:
            # Horizontal flip
            if np.random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            # Brightness/contrast
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # Contrast
                beta = np.random.uniform(-20, 20)    # Brightness
                image = np.clip(alpha * image + beta, 0, 255)

        # Extract boundaries from mask
        boundaries, valid_mask = self.extract_boundaries(mask)

        # Normalize image
        image = image / 255.0

        # Normalize boundaries to [0, 1] range
        boundaries = boundaries / self.patch_size

        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        boundaries = torch.from_numpy(boundaries).float()      # (4, W)
        valid_mask = torch.from_numpy(valid_mask).float()      # (4, W)
        mask = torch.from_numpy(mask.copy()).long()            # (H, W)

        return {
            'image': image,
            'boundaries': boundaries,
            'valid_mask': valid_mask,
            'seg_mask': mask,
        }


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class BoundaryRegressionNet(nn.Module):
    """
    U-Net style encoder-decoder for boundary regression.

    Output: (B, 4, W) - 4 boundary y-positions for each x-position
    """

    def __init__(self, in_channels=1, base_features=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_features)
        self.enc2 = ConvBlock(base_features, base_features * 2)
        self.enc3 = ConvBlock(base_features * 2, base_features * 4)
        self.enc4 = ConvBlock(base_features * 4, base_features * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 8, base_features * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_features * 16, base_features * 8)

        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = ConvBlock(base_features * 2, base_features)

        # Boundary regression head
        # Collapse height dimension and predict 4 boundaries per column
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(base_features, base_features, 3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, 32, 1),
            nn.ReLU(inplace=True),
        )

        # Soft-attention over height to get boundary positions
        self.height_attention = nn.Sequential(
            nn.Conv2d(32, 4, 1),  # 4 boundaries
        )

        # Alternative: Direct regression head
        self.direct_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # Pool height, keep width
            nn.Flatten(1, 2),  # (B, 32, W)
            # Will apply 1D conv
        )
        self.boundary_predictor = nn.Conv1d(32, 4, kernel_size=5, padding=2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Boundary regression using soft-argmax
        feat = self.boundary_conv(d1)  # (B, 32, H, W)

        # Method: Soft-argmax for differentiable boundary extraction
        attention_logits = self.height_attention(feat)  # (B, 4, H, W)
        B, C, H, W = attention_logits.shape

        # Softmax over height dimension
        attention_weights = F.softmax(attention_logits, dim=2)  # (B, 4, H, W)

        # Create height coordinates
        height_coords = torch.linspace(0, 1, H, device=x.device)
        height_coords = height_coords.view(1, 1, H, 1).expand(B, 4, H, W)

        # Weighted sum to get boundary positions (soft-argmax)
        boundaries = (attention_weights * height_coords).sum(dim=2)  # (B, 4, W)

        return boundaries, attention_weights


class BoundaryRegressionLoss(nn.Module):
    """
    Combined loss for boundary regression:
    1. Position loss (L1/L2 on boundary positions)
    2. Ordering loss (enforce boundary[i] < boundary[i+1])
    3. Smoothness loss (boundaries should be smooth along x)
    """

    def __init__(self, ordering_weight=1.0, smoothness_weight=0.5):
        super().__init__()
        self.ordering_weight = ordering_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, pred_boundaries, gt_boundaries, valid_mask):
        """
        Args:
            pred_boundaries: (B, 4, W) predicted boundary positions [0, 1]
            gt_boundaries: (B, 4, W) ground truth boundary positions [0, 1]
            valid_mask: (B, 4, W) validity weights for each boundary point
        """
        B, num_boundaries, W = pred_boundaries.shape

        # 1. Position loss (weighted L1)
        position_error = torch.abs(pred_boundaries - gt_boundaries)
        position_loss = (position_error * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        # 2. Ordering loss: boundary[i] should be < boundary[i+1]
        # Penalize violations where boundary[i] >= boundary[i+1]
        ordering_loss = 0.0
        for i in range(num_boundaries - 1):
            # diff should be positive (boundary[i+1] > boundary[i])
            diff = pred_boundaries[:, i+1, :] - pred_boundaries[:, i, :]
            # Penalize when diff <= 0 (violation) with margin
            margin = 0.01  # Minimum separation
            violation = F.relu(margin - diff)  # Positive when too close or inverted
            ordering_loss = ordering_loss + violation.mean()
        ordering_loss = ordering_loss / (num_boundaries - 1)

        # 3. Smoothness loss: penalize large jumps between adjacent x positions
        smoothness_loss = 0.0
        if W > 1:
            dx = pred_boundaries[:, :, 1:] - pred_boundaries[:, :, :-1]
            smoothness_loss = (dx ** 2).mean()

        # Total loss
        total_loss = (
            position_loss +
            self.ordering_weight * ordering_loss +
            self.smoothness_weight * smoothness_loss
        )

        return {
            'total': total_loss,
            'position': position_loss,
            'ordering': ordering_loss,
            'smoothness': smoothness_loss,
        }


def boundaries_to_segmentation(boundaries, H):
    """
    Convert boundary positions to segmentation mask.

    Args:
        boundaries: (B, 4, W) boundary positions in [0, 1]
        H: height of output mask

    Returns:
        seg_mask: (B, H, W) segmentation mask with layer labels 0-3
    """
    B, _, W = boundaries.shape
    device = boundaries.device

    # Scale boundaries to pixel coordinates
    boundaries_px = boundaries * H  # (B, 4, W)

    # Create output mask
    seg_mask = torch.zeros(B, H, W, dtype=torch.long, device=device)

    # Create y coordinates
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

    # Assign layers based on boundary positions
    # Layer 0: y < boundary[1] (above RNFL/INL junction)
    # Layer 1: boundary[1] <= y < boundary[2]
    # Layer 2: boundary[2] <= y < boundary[3]
    # Layer 3: y >= boundary[3]

    b1 = boundaries_px[:, 1:2, :].expand(B, H, W)  # RNFL/INL
    b2 = boundaries_px[:, 2:3, :].expand(B, H, W)  # INL/IS_OS
    b3 = boundaries_px[:, 3:4, :].expand(B, H, W)  # IS_OS/RPE

    seg_mask = torch.where(y_coords >= b3, torch.tensor(3, device=device), seg_mask)
    seg_mask = torch.where((y_coords >= b2) & (y_coords < b3), torch.tensor(2, device=device), seg_mask)
    seg_mask = torch.where((y_coords >= b1) & (y_coords < b2), torch.tensor(1, device=device), seg_mask)
    seg_mask = torch.where(y_coords < b1, torch.tensor(0, device=device), seg_mask)

    return seg_mask


def compute_metrics(pred_boundaries, gt_boundaries, pred_seg, gt_seg, patch_size):
    """Compute boundary MAE and segmentation Dice scores."""

    # Boundary MAE in pixels
    boundary_mae = torch.abs(pred_boundaries - gt_boundaries) * patch_size
    boundary_mae = boundary_mae.mean(dim=(0, 2))  # Mean per boundary

    # Segmentation Dice per class
    dice_scores = []
    for c in range(4):
        pred_c = (pred_seg == c).float()
        gt_c = (gt_seg == c).float()

        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()

        if union > 0:
            dice = (2 * intersection / union).item()
        else:
            dice = 1.0 if pred_c.sum() == 0 else 0.0
        dice_scores.append(dice)

    return boundary_mae, dice_scores


def train_epoch(model, loader, criterion, optimizer, device, patch_size):
    model.train()
    total_loss = 0
    all_boundary_mae = []
    all_dice = []

    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        gt_boundaries = batch['boundaries'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        gt_seg = batch['seg_mask'].to(device)

        optimizer.zero_grad()

        pred_boundaries, attention = model(images)
        losses = criterion(pred_boundaries, gt_boundaries, valid_mask)

        losses['total'].backward()
        optimizer.step()

        total_loss += losses['total'].item()

        # Compute metrics - MUST detach tensors to prevent memory leak!
        with torch.no_grad():
            pred_boundaries_detached = pred_boundaries.detach()
            pred_seg = boundaries_to_segmentation(pred_boundaries_detached, patch_size)
            boundary_mae, dice = compute_metrics(
                pred_boundaries_detached, gt_boundaries, pred_seg, gt_seg, patch_size
            )
            all_boundary_mae.append(boundary_mae.cpu().numpy())
            all_dice.append(dice)

        # Free memory from unused attention tensor
        del attention

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'pos': f"{losses['position'].item():.4f}",
            'ord': f"{losses['ordering'].item():.4f}",
        })

        # Periodic memory cleanup to prevent fragmentation
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    avg_loss = total_loss / len(loader)
    avg_mae = np.mean(all_boundary_mae, axis=0)
    avg_dice = np.mean(all_dice, axis=0)

    return avg_loss, avg_mae, avg_dice


def validate(model, loader, criterion, device, patch_size):
    model.eval()
    total_loss = 0
    all_boundary_mae = []
    all_dice = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch['image'].to(device)
            gt_boundaries = batch['boundaries'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            gt_seg = batch['seg_mask'].to(device)

            pred_boundaries, attention = model(images)
            del attention  # Free unused memory

            losses = criterion(pred_boundaries, gt_boundaries, valid_mask)

            total_loss += losses['total'].item()

            pred_seg = boundaries_to_segmentation(pred_boundaries, patch_size)
            boundary_mae, dice = compute_metrics(
                pred_boundaries, gt_boundaries, pred_seg, gt_seg, patch_size
            )
            all_boundary_mae.append(boundary_mae.cpu().numpy())
            all_dice.append(dice)

            # Clean up batch tensors
            del pred_boundaries, pred_seg

    avg_loss = total_loss / len(loader)
    avg_mae = np.mean(all_boundary_mae, axis=0)
    avg_dice = np.mean(all_dice, axis=0)

    return avg_loss, avg_mae, avg_dice


def visualize_predictions(model, dataset, device, output_dir, num_samples=5):
    """Visualize boundary predictions."""
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_boundaries = sample['boundaries'].numpy()
        gt_seg = sample['seg_mask'].numpy()

        with torch.no_grad():
            pred_boundaries, attention = model(image)
            pred_boundaries = pred_boundaries[0].cpu().numpy()
            pred_seg = boundaries_to_segmentation(
                torch.from_numpy(pred_boundaries).unsqueeze(0),
                dataset.patch_size
            )[0].numpy()

        H, W = gt_seg.shape

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image with boundaries
        axes[0, 0].imshow(sample['image'][0].numpy(), cmap='gray')
        axes[0, 0].set_title('Input Image')
        for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
            axes[0, 0].plot(np.arange(W), gt_boundaries[i] * H, color=color,
                           linewidth=1, label=f'GT B{i}')
        axes[0, 0].legend(fontsize=8)

        # Predicted boundaries
        axes[0, 1].imshow(sample['image'][0].numpy(), cmap='gray')
        axes[0, 1].set_title('Predicted Boundaries')
        for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
            axes[0, 1].plot(np.arange(W), pred_boundaries[i] * H, color=color,
                           linewidth=1, label=f'Pred B{i}')
        axes[0, 1].legend(fontsize=8)

        # Boundary comparison
        axes[0, 2].set_title('Boundary Comparison')
        boundary_names = ['ILM', 'RNFL/INL', 'INL/IS_OS', 'IS_OS/RPE']
        for i in range(4):
            axes[0, 2].plot(gt_boundaries[i] * H, label=f'GT {boundary_names[i]}', linestyle='--')
            axes[0, 2].plot(pred_boundaries[i] * H, label=f'Pred {boundary_names[i]}')
        axes[0, 2].legend(fontsize=6)
        axes[0, 2].set_xlabel('X position')
        axes[0, 2].set_ylabel('Y position (pixels)')

        # GT segmentation
        axes[1, 0].imshow(gt_seg, cmap='tab10', vmin=0, vmax=3)
        axes[1, 0].set_title('GT Segmentation')

        # Predicted segmentation
        axes[1, 1].imshow(pred_seg, cmap='tab10', vmin=0, vmax=3)
        axes[1, 1].set_title('Predicted Segmentation')

        # Difference
        diff = (pred_seg != gt_seg).astype(float)
        axes[1, 2].imshow(diff, cmap='Reds')
        axes[1, 2].set_title(f'Errors (Acc: {100*(1-diff.mean()):.1f}%)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boundary_pred_{idx}.png'), dpi=150)
        plt.close()

    print(f"Saved {len(indices)} visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Boundary Regression for OCT Segmentation')

    parser.add_argument('--train_jsonl', type=str,
                        default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', type=str,
                        default='combined_val.jsonl')
    parser.add_argument('--output_dir', type=str, default='./boundary_regression_output')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=256)

    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)

    parser.add_argument('--ordering_weight', type=float, default=1.0)
    parser.add_argument('--smoothness_weight', type=float, default=0.5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n=== BOUNDARY REGRESSION APPROACH ===")
    print("Instead of pixel-wise classification, we predict 4 boundary curves:")
    print("  Boundary 0: ILM (top of retina)")
    print("  Boundary 1: RNFL/INL junction")
    print("  Boundary 2: INL/IS_OS junction")
    print("  Boundary 3: IS_OS/RPE junction")
    print("\nAdvantages:")
    print("  - No class imbalance (4 curves vs millions of pixels)")
    print("  - Natural layer ordering enforcement")
    print("  - Matches clinical OCT software approach")
    print("=" * 50)

    # Data
    train_dataset = BoundaryDataset(
        args.train_jsonl,
        max_samples=args.max_train,
        patch_size=args.patch_size,
        augment=True,
    )
    val_dataset = BoundaryDataset(
        args.val_jsonl,
        max_samples=args.max_val,
        patch_size=args.patch_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = BoundaryRegressionNet(in_channels=1, base_features=64).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = BoundaryRegressionLoss(
        ordering_weight=args.ordering_weight,
        smoothness_weight=args.smoothness_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_val_loss = float('inf')
    best_is_os_mae = float('inf')

    boundary_names = ['ILM', 'RNFL/INL', 'INL/IS_OS', 'IS_OS/RPE']
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_loss, train_mae, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, args.patch_size
        )

        val_loss, val_mae, val_dice = validate(
            model, val_loader, criterion, device, args.patch_size
        )

        scheduler.step()

        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        print("\nBoundary MAE (pixels):")
        for i, name in enumerate(boundary_names):
            print(f"  {name}: Train={train_mae[i]:.2f} | Val={val_mae[i]:.2f}")

        print("\nDerived Segmentation Dice:")
        for i, name in enumerate(layer_names):
            print(f"  {name}: Train={train_dice[i]:.4f} | Val={val_dice[i]:.4f}")

        # Key metrics
        is_os_mae = val_mae[2]  # INL/IS_OS boundary
        inl_dice = val_dice[1]  # INL_OPL_ONL layer
        is_os_dice = val_dice[2]  # IS_OS layer

        print(f"\n>>> KEY METRICS:")
        print(f"    IS/OS Boundary MAE: {is_os_mae:.2f} px (target: <3 px)")
        print(f"    INL_OPL_ONL Dice: {inl_dice:.4f} (previously stuck at 0.26)")
        print(f"    IS_OS Dice: {is_os_dice:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_dice': val_dice,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"    Saved best model (val_loss: {val_loss:.4f})")

        if is_os_mae < best_is_os_mae:
            best_is_os_mae = is_os_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'is_os_mae': is_os_mae,
            }, os.path.join(args.output_dir, 'best_is_os_model.pth'))
            print(f"    Saved best IS/OS model (MAE: {is_os_mae:.2f} px)")

        # Clear GPU cache between epochs to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final visualization
    print("\nGenerating visualizations...")
    visualize_predictions(model, val_dataset, device,
                         os.path.join(args.output_dir, 'visualizations'))

    print("\n=== TRAINING COMPLETE ===")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best IS/OS MAE: {best_is_os_mae:.2f} px")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
