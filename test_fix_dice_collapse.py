#!/usr/bin/env python3
"""
Test script to fix Dice score collapse in TMI training.

Problem: Dice scores for classes 1,2,3 collapsed to near-zero during training.
Solution: Higher seg loss weight + monitor seg loss + staged training.
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


class SimpleDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=50, patch_size=128):
        with open(jsonl_path, 'r') as f:
            self.samples = [json.loads(line) for line in f][:max_samples]
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
        noise = np.random.randn(*clean.shape) * 0.1
        noisy = np.clip(clean + noise, 0, 1).astype(np.float32)

        # Remap to 4 classes
        mask_4class = np.zeros_like(mask)
        mask_4class[mask == 0] = 0
        mask_4class[mask == 1] = 0
        mask_4class[mask == 2] = 0
        mask_4class[mask == 3] = 1
        mask_4class[mask == 4] = 2
        mask_4class[mask >= 5] = 3

        return {
            'noisy': torch.from_numpy(noisy).float().unsqueeze(0),
            'clean': torch.from_numpy(clean.astype(np.float32)).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_4class.astype(np.int64)),
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

    # Load data
    train_dataset = SimpleDataset('combined_train.jsonl', max_samples=100, patch_size=128)
    val_dataset = SimpleDataset('combined_val.jsonl', max_samples=20, patch_size=128)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Load TMI model
    from train_tmi_enhanced import TMIEnhancedModel

    model = TMIEnhancedModel(
        num_classes=4,
        nafnet_width=32,
        use_layer_specific_heads=True,
        use_seg_guided_attention=True,
        use_columnar_attention=True,
        use_boundary_regression=True,
        num_head_blocks=2,
        head_hidden_channels=32,
        columnar_dim=64,
        num_columnar_blocks=1,
    ).to(device)

    # Load pretrained weights
    nafnet_ckpt = torch.load('checkpoints/multitask/best_psnr.pth', map_location=device, weights_only=False)
    if 'model_state_dict' in nafnet_ckpt:
        nafnet_state = nafnet_ckpt['model_state_dict']
    else:
        nafnet_state = nafnet_ckpt

    nafnet_weights = {k: v for k, v in nafnet_state.items() if k.startswith('nafnet.')}
    model.load_state_dict(nafnet_weights, strict=False)
    print(f"Loaded {len(nafnet_weights)} NAFNet weights")

    seg_ckpt = torch.load('checkpoints/seg_4class_pretrained.pth', map_location=device, weights_only=False)
    seg_state = seg_ckpt['model_state_dict']
    seg_weights = {f'segmenter.{k}': v for k, v in seg_state.items()}
    model.load_state_dict(seg_weights, strict=False)
    print(f"Loaded {len(seg_weights)} segmentation weights")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =========================================================================
    # TEST 1: High seg loss weight (lambda_seg = 5.0)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: High segmentation loss weight (lambda_seg=5.0)")
    print("="*70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(5):
        model.train()
        total_denoise_loss = 0
        total_seg_loss = 0
        total_dice_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(noisy)

            # Denoising loss
            denoise_loss = F.l1_loss(outputs['denoised'], clean)

            # Segmentation losses - HIGH WEIGHT
            seg_ce = F.cross_entropy(outputs['seg_logits'], mask)
            seg_dice = dice_loss(outputs['seg_logits'], mask)

            # Combined loss with HIGH seg weight
            lambda_seg = 5.0  # Much higher than before
            loss = denoise_loss + lambda_seg * (seg_ce + seg_dice)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_denoise_loss += denoise_loss.item()
            total_seg_loss += seg_ce.item()
            total_dice_loss += seg_dice.item()
            n_batches += 1

            pbar.set_postfix({
                'denoise': f'{denoise_loss.item():.4f}',
                'seg_ce': f'{seg_ce.item():.4f}',
                'dice_loss': f'{seg_dice.item():.4f}',
            })

        # Evaluate
        model.eval()
        all_dice = [[] for _ in range(4)]

        with torch.no_grad():
            for batch in val_loader:
                noisy = batch['noisy'].to(device)
                mask = batch['mask'].to(device)

                outputs = model(noisy)
                dice_scores = compute_dice_per_class(outputs['seg_logits'], mask)

                for c in range(4):
                    all_dice[c].append(dice_scores[c])

        mean_dice = [np.mean(all_dice[c]) for c in range(4)]
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Denoise Loss: {total_denoise_loss/n_batches:.4f}")
        print(f"  Seg CE Loss:  {total_seg_loss/n_batches:.4f}")
        print(f"  Dice Loss:    {total_dice_loss/n_batches:.4f}")
        print(f"  Dice Scores:")
        print(f"    RNFL_GCL:    {mean_dice[0]:.4f}")
        print(f"    INL_OPL_ONL: {mean_dice[1]:.4f}")
        print(f"    IS_OS:       {mean_dice[2]:.4f}")
        print(f"    RPE_Choroid: {mean_dice[3]:.4f}")
        print(f"    Mean:        {np.mean(mean_dice):.4f}")

        # Check if Dice collapsed
        if mean_dice[1] < 0.1 or mean_dice[2] < 0.1:
            print("\n  WARNING: Dice scores collapsing!")
        else:
            print("\n  Dice scores maintained!")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Final Dice Scores:")
    for c, name in enumerate(['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']):
        status = "OK" if mean_dice[c] > 0.3 else "COLLAPSED"
        print(f"  {name}: {mean_dice[c]:.4f} [{status}]")

    if all(d > 0.3 for d in mean_dice):
        print("\nSUCCESS: Dice scores maintained with high seg loss weight!")
        return True
    else:
        print("\nFAILED: Dice still collapsing. Need different approach.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
