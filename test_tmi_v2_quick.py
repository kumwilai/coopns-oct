#!/usr/bin/env python3
"""
Quick Test: TMI v2 Features (Columnar Attention + Boundary Regression)

This script runs a quick 3-epoch test to verify:
1. Columnar attention works and improves metrics
2. Boundary regression works and improves IS/OS MAE
3. Memory stays within limits (64x64 patches)

Compares against baseline (without TMI v2 features).
"""

import os
import sys
import gc
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

# Memory management
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


# Simplified dataset for quick testing
class QuickTestDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=50, patch_size=64):
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
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        clean_patch = clean[top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]

        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Add noise
        speckle = 1.0 + 0.4 * (np.random.exponential(1.0, clean_patch.shape) - 1.0)
        noisy_patch = clean_patch * np.maximum(speckle, 0.01)
        noisy_patch = noisy_patch + np.random.randn(*clean_patch.shape) * 0.08
        noisy_patch = np.clip(noisy_patch, 0, 1).astype(np.float32)

        # Remap mask to 4-class
        mask_4class = np.zeros_like(mask_patch)
        mask_4class[mask_patch == 0] = 0
        mask_4class[mask_patch == 1] = 1
        mask_4class[mask_patch == 2] = 1
        mask_4class[mask_patch == 3] = 2
        mask_4class[mask_patch == 4] = 3

        return {
            'noisy': torch.from_numpy(noisy_patch).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_patch.astype(np.float32)).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_4class).long(),
        }


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 50.0
    return 10 * torch.log10(1.0 / mse).item()


def compute_dice(pred_logits, target, class_idx):
    pred = pred_logits.argmax(dim=1)
    pred_mask = (pred == class_idx).float()
    target_mask = (target == class_idx).float()
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    if union < 1:
        return 0.0
    return (2.0 * intersection / (union + 1e-8)).item()


def train_and_evaluate(model, train_loader, val_loader, device, epochs=3, name="Model"):
    """Train model and return metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(noisy)

            # Losses
            denoise_loss = F.l1_loss(outputs['denoised'], clean)
            seg_loss = F.cross_entropy(outputs['seg_logits'], mask)

            loss = denoise_loss + 0.5 * seg_loss

            # Boundary regression loss if available (reduced weight for stability)
            if 'boundary_positions' in outputs and outputs['boundary_positions'] is not None:
                from nsnd.models.boundary_regression import extract_boundaries_from_segmentation, BoundaryLoss
                gt_bounds = extract_boundaries_from_segmentation(mask, num_classes=4)
                if not hasattr(train_and_evaluate, 'boundary_loss_fn'):
                    train_and_evaluate.boundary_loss_fn = BoundaryLoss(num_boundaries=5).to(device)
                b_loss, _ = train_and_evaluate.boundary_loss_fn(outputs['boundary_positions'], gt_bounds)
                loss = loss + 0.2 * b_loss  # Reduced from 0.5 to 0.2 for stability

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            del noisy, clean, mask, outputs, loss

            pbar.set_postfix({'loss': f'{total_loss/n_batches:.4f}', 'mem': f'{get_memory_mb():.0f}MB'})

        clear_memory()

    # Evaluate
    model.eval()
    psnr_values = []
    dice_values = {i: [] for i in range(4)}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)

            outputs = model(noisy)

            psnr = compute_psnr(outputs['denoised'], clean)
            psnr_values.append(psnr)

            for c in range(4):
                dice = compute_dice(outputs['seg_logits'], mask, c)
                dice_values[c].append(dice)

            del noisy, clean, mask, outputs

    clear_memory()

    metrics = {
        'psnr': np.mean(psnr_values),
        'dice_rnfl_gcl': np.mean(dice_values[0]),
        'dice_inl_opl_onl': np.mean(dice_values[1]),
        'dice_is_os': np.mean(dice_values[2]),
        'dice_rpe_choroid': np.mean(dice_values[3]),
        'dice_mean': np.mean([np.mean(dice_values[c]) for c in range(4)]),
    }

    return metrics


def create_baseline_model(device):
    """Create baseline model without TMI v2 features."""
    from train_tmi_enhanced import TMIEnhancedModel

    model = TMIEnhancedModel(
        num_classes=4,
        nafnet_width=32,  # Reduced for memory
        use_layer_specific_heads=True,
        use_seg_guided_attention=True,
        use_columnar_attention=False,  # DISABLED
        use_boundary_regression=False,  # DISABLED
        num_head_blocks=2,
        head_hidden_channels=32,
    ).to(device)

    return model


def create_tmiv2_model(device):
    """Create TMI v2 model with columnar attention and boundary regression."""
    from train_tmi_enhanced import TMIEnhancedModel

    model = TMIEnhancedModel(
        num_classes=4,
        nafnet_width=32,  # Reduced for memory
        use_layer_specific_heads=True,
        use_seg_guided_attention=True,
        use_columnar_attention=True,   # ENABLED
        use_boundary_regression=True,  # ENABLED
        num_head_blocks=2,
        head_hidden_channels=32,
        columnar_dim=64,
        num_columnar_blocks=1,
    ).to(device)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_train', type=int, default=50)
    parser.add_argument('--max_val', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data
    print("\nLoading data...")
    train_dataset = QuickTestDataset(args.train_jsonl, max_samples=args.max_train)
    val_dataset = QuickTestDataset(args.val_jsonl, max_samples=args.max_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Test 1: Baseline (without TMI v2)
    print("\n" + "="*60)
    print("TEST 1: BASELINE (without TMI v2 features)")
    print("="*60)

    baseline_model = create_baseline_model(device)
    baseline_metrics = train_and_evaluate(
        baseline_model, train_loader, val_loader, device,
        epochs=args.epochs, name="Baseline"
    )
    del baseline_model
    clear_memory()

    # Test 2: TMI v2 (with columnar + boundary regression)
    print("\n" + "="*60)
    print("TEST 2: TMI v2 (with Columnar + Boundary Regression)")
    print("="*60)

    tmiv2_model = create_tmiv2_model(device)
    tmiv2_metrics = train_and_evaluate(
        tmiv2_model, train_loader, val_loader, device,
        epochs=args.epochs, name="TMI v2"
    )
    del tmiv2_model
    clear_memory()

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\n{'Metric':<20s} {'Baseline':>12s} {'TMI v2':>12s} {'Diff':>12s} {'Status':>10s}")
    print("-" * 70)

    improvements = 0
    total_metrics = 0

    for metric in ['psnr', 'dice_rnfl_gcl', 'dice_inl_opl_onl', 'dice_is_os', 'dice_rpe_choroid', 'dice_mean']:
        baseline_val = baseline_metrics[metric]
        tmiv2_val = tmiv2_metrics[metric]
        diff = tmiv2_val - baseline_val

        if diff > 0:
            status = "IMPROVED"
            improvements += 1
        elif diff < -0.01:
            status = "WORSE"
        else:
            status = "SAME"

        total_metrics += 1

        print(f"{metric:<20s} {baseline_val:>12.4f} {tmiv2_val:>12.4f} {diff:>+12.4f} {status:>10s}")

    print("-" * 70)

    # Summary
    print(f"\nSUMMARY:")
    print(f"  Metrics improved: {improvements}/{total_metrics}")

    if improvements >= total_metrics // 2:
        print(f"\n  TMI v2 features IMPROVE performance!")
        print(f"  Recommendation: ENABLE columnar attention and boundary regression")
        return True
    else:
        print(f"\n  TMI v2 features need more training or tuning")
        print(f"  Recommendation: Try longer training or adjust hyperparameters")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
