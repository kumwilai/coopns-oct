#!/usr/bin/env python3
"""
Train segmenter on NAFNet-denoised images to prevent domain shift.

Problem: Segmenter trained on clean images fails on NAFNet output during TMI inference.
Solution: Fine-tune segmenter on NAFNet-denoised images.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class DenoisedDataset(Dataset):
    """Dataset that returns NAFNet-denoised images for segmentation training."""

    def __init__(self, jsonl_path, nafnet_model, device, max_samples=None, patch_size=128):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        if max_samples:
            self.samples = self.samples[:max_samples]

        self.nafnet = nafnet_model
        self.device = device
        self.patch_size = patch_size

        # Pre-compute denoised images
        print(f"Pre-computing denoised images for {len(self.samples)} samples...")
        self.denoised_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        mask = np.array(Image.open(sample['mask_path']))

        H, W = clean.shape
        ps = self.patch_size

        # Random crop
        if H > ps and W > ps:
            top = np.random.randint(0, H - ps)
            left = np.random.randint(0, W - ps)
            clean = clean[top:top+ps, left:left+ps]
            mask = mask[top:top+ps, left:left+ps]
        else:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode='reflect')[:ps, :ps]
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')[:ps, :ps]

        # Add noise
        noise_std = np.random.uniform(0.05, 0.15)
        noise = np.random.randn(*clean.shape) * noise_std
        noisy = np.clip(clean + noise, 0, 1).astype(np.float32)

        # Denoise with NAFNet
        with torch.no_grad():
            noisy_tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(self.device)
            denoised_tensor = self.nafnet(noisy_tensor)
            denoised = denoised_tensor.squeeze().cpu().numpy()

        # Remap mask to 4 classes - MUST MATCH train_seg_4class.py mapping
        # Data: 0=background, 1=RNFL, 2=GCL, 3=INL+OPL+ONL, 4=IS_OS, 5+=RPE_Choroid
        # Target: 0=RNFL_GCL, 1=INL_OPL_ONL, 2=IS_OS, 3=RPE_Choroid
        mask_4class = np.zeros_like(mask)
        mask_4class[mask == 0] = 0  # Background -> RNFL_GCL region (top)
        mask_4class[mask == 1] = 0  # RNFL -> class 0 (RNFL_GCL)
        mask_4class[mask == 2] = 0  # GCL -> class 0 (RNFL_GCL)
        mask_4class[mask == 3] = 1  # INL/OPL/ONL -> class 1 (INL_OPL_ONL)
        mask_4class[mask == 4] = 2  # IS_OS -> class 2 (IS_OS)
        mask_4class[mask >= 5] = 3  # RPE/Choroid -> class 3 (RPE_Choroid)

        return {
            'denoised': torch.from_numpy(denoised.astype(np.float32)).float().unsqueeze(0),
            'clean': torch.from_numpy(clean.astype(np.float32)).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_4class.astype(np.int64)),
        }


class SimpleBoundarySegmenter(nn.Module):
    """Simple 4-class segmenter for clinical OCT regions.

    Same architecture as train_seg_4class.py to match pretrained weights.
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=32):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(base_filters, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Decoder with skip connections
        d2 = self.up2(e3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seg_logits = self.seg_head(d1)
        seg_probs = F.softmax(seg_logits, dim=1)

        return {
            'seg_logits': seg_logits,
            'seg_probs': seg_probs,
        }


def compute_dice_per_class(pred_logits, target, num_classes=4):
    """Compute Dice score per class."""
    pred = pred_logits.argmax(dim=1)
    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2.0 * intersection / (union + 1e-8)).item()
        else:
            dice = 1.0
        dice_scores.append(dice)

    return dice_scores


def dice_loss(pred_logits, target, num_classes=4):
    """Compute Dice loss."""
    pred_probs = F.softmax(pred_logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

    dice_per_class = []
    for c in range(num_classes):
        pred_c = pred_probs[:, c].flatten()
        target_c = target_one_hot[:, c].flatten()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        dice_per_class.append(1 - dice)

    return sum(dice_per_class) / num_classes


def main():
    device = torch.device('cpu')
    print(f"Device: {device}")

    # Load NAFNet for denoising
    print("\nLoading NAFNet...")
    from nsnd_oct.nsnd.models.nafnet import NAFNetFullFiLM

    # Use NAFNetFullFiLM to match checkpoint (from train_tmi_enhanced.py)
    nafnet = NAFNetFullFiLM(
        img_channel=1,
        width=64,  # Match checkpoint: width=64
        middle_blk_num=2,  # Match checkpoint
        enc_blk_nums=[2, 2, 2],  # Match checkpoint: 3 stages with 2 blocks each
        dec_blk_nums=[2, 2, 2],  # Match checkpoint
    ).to(device)

    # Load NAFNet weights
    nafnet_ckpt = torch.load('checkpoints/multitask/best_psnr.pth', map_location=device, weights_only=False)
    if 'model_state_dict' in nafnet_ckpt:
        nafnet_state = nafnet_ckpt['model_state_dict']
    elif 'state_dict' in nafnet_ckpt:
        nafnet_state = nafnet_ckpt['state_dict']
    else:
        nafnet_state = nafnet_ckpt

    # Map backbone.* to direct keys (NAFNet model keys)
    nafnet_weights = {}
    for k, v in nafnet_state.items():
        if k.startswith('backbone.'):
            # Map backbone.intro.weight -> intro.weight
            nafnet_weights[k[len('backbone.'):]] = v
        elif k.startswith('nafnet.'):
            nafnet_weights[k[len('nafnet.'):]] = v
        # Skip non-NAFNet keys
        elif not any(k.startswith(p) for p in ['segmenter.', 'layer_heads.', 'seg_head.',
                                                'feature_extractor.', 'refinement', 'confidence']):
            nafnet_weights[k] = v

    missing, unexpected = nafnet.load_state_dict(nafnet_weights, strict=False)
    print(f"Loaded NAFNet: {len(nafnet_weights)} weights")
    if missing:
        print(f"  Missing: {len(missing)}")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)}")

    nafnet.eval()

    # Create datasets
    print("\nCreating datasets...")

    # Simple dataset for efficiency
    class SimpleDenoisedDataset(Dataset):
        def __init__(self, jsonl_path, nafnet, device, max_samples=None, patch_size=128):
            with open(jsonl_path, 'r') as f:
                self.samples = [json.loads(line) for line in f]
            if max_samples:
                self.samples = self.samples[:max_samples]
            self.nafnet = nafnet
            self.device = device
            self.patch_size = patch_size

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
            mask = np.array(Image.open(sample['mask_path']))

            H, W = clean.shape
            ps = self.patch_size

            # Random crop
            if H > ps and W > ps:
                top = np.random.randint(0, H - ps)
                left = np.random.randint(0, W - ps)
                clean = clean[top:top+ps, left:left+ps]
                mask = mask[top:top+ps, left:left+ps]
            else:
                pad_h = max(0, ps - H)
                pad_w = max(0, ps - W)
                clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode='reflect')[:ps, :ps]
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')[:ps, :ps]

            # Add noise
            noise_std = np.random.uniform(0.05, 0.15)
            noise = np.random.randn(*clean.shape) * noise_std
            noisy = np.clip(clean + noise, 0, 1).astype(np.float32)

            # Denoise with NAFNet
            with torch.no_grad():
                noisy_tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(self.device)
                denoised_tensor = self.nafnet(noisy_tensor)
                denoised = denoised_tensor.squeeze().cpu().numpy()

            # Remap mask to 4 classes - MUST MATCH train_seg_4class.py
            mask_4class = np.zeros_like(mask)
            mask_4class[mask == 0] = 0  # Background -> RNFL_GCL region
            mask_4class[mask == 1] = 0  # RNFL -> RNFL_GCL
            mask_4class[mask == 2] = 0  # GCL -> RNFL_GCL
            mask_4class[mask == 3] = 1  # INL/OPL/ONL -> INL_OPL_ONL
            mask_4class[mask == 4] = 2  # IS_OS -> IS_OS
            mask_4class[mask >= 5] = 3  # RPE_Choroid -> RPE_Choroid

            return {
                'denoised': torch.from_numpy(denoised.astype(np.float32)).float().unsqueeze(0),
                'clean': torch.from_numpy(clean.astype(np.float32)).float().unsqueeze(0),
                'mask': torch.from_numpy(mask_4class.astype(np.int64)),
            }

    train_dataset = SimpleDenoisedDataset('combined_train.jsonl', nafnet, device, max_samples=300, patch_size=128)
    val_dataset = SimpleDenoisedDataset('combined_val.jsonl', nafnet, device, max_samples=50, patch_size=128)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create segmenter and load pretrained weights
    segmenter = SimpleBoundarySegmenter(in_channels=1, num_classes=4, base_filters=32).to(device)

    # Load pretrained weights
    if os.path.exists('checkpoints/seg_4class_pretrained.pth'):
        print("\nLoading pretrained segmenter...")
        ckpt = torch.load('checkpoints/seg_4class_pretrained.pth', map_location=device, weights_only=False)
        segmenter.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded pretrained weights (Best Dice: {ckpt.get('best_dice', 'N/A')})")

    print(f"Segmenter parameters: {sum(p.numel() for p in segmenter.parameters()):,}")

    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(segmenter.parameters(), lr=5e-5, weight_decay=1e-4)

    # Training
    print("\n" + "="*70)
    print("Fine-tuning segmenter on NAFNet-denoised images")
    print("="*70)

    best_dice = 0

    for epoch in range(10):
        # Train
        segmenter.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in pbar:
            denoised = batch['denoised'].to(device)  # Use denoised images!
            mask = batch['mask'].to(device)

            optimizer.zero_grad()

            outputs = segmenter(denoised)
            seg_logits = outputs['seg_logits']

            # Loss
            ce_loss = F.cross_entropy(seg_logits, mask)
            d_loss = dice_loss(seg_logits, mask)
            loss = ce_loss + d_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'dice': f'{d_loss.item():.4f}',
            })

        # Validate
        segmenter.eval()
        all_dice = [[] for _ in range(4)]

        with torch.no_grad():
            for batch in val_loader:
                denoised = batch['denoised'].to(device)
                mask = batch['mask'].to(device)

                outputs = segmenter(denoised)
                dice_scores = compute_dice_per_class(outputs['seg_logits'], mask)

                for c in range(4):
                    all_dice[c].append(dice_scores[c])

        mean_dice = [np.mean(all_dice[c]) for c in range(4)]
        mean_all = np.mean(mean_dice)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {total_loss/n_batches:.4f}")
        print(f"  Dice Scores:")
        print(f"    RNFL_GCL:    {mean_dice[0]:.4f}")
        print(f"    INL_OPL_ONL: {mean_dice[1]:.4f}")
        print(f"    IS_OS:       {mean_dice[2]:.4f}")
        print(f"    RPE_Choroid: {mean_dice[3]:.4f}")
        print(f"    Mean:        {mean_all:.4f}")

        # Save best
        if mean_all > best_dice:
            best_dice = mean_all
            torch.save({
                'model_state_dict': segmenter.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
                'dice_per_class': mean_dice,
            }, 'checkpoints/seg_4class_denoised.pth')
            print(f"  Saved best model (Dice: {best_dice:.4f})")

    print("\n" + "="*70)
    print(f"Training complete! Best Dice: {best_dice:.4f}")
    print("Saved to: checkpoints/seg_4class_denoised.pth")
    print("="*70)


if __name__ == '__main__':
    main()
