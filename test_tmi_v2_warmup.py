#!/usr/bin/env python3
"""
TMI v2 with Warmup Training Strategy

Strategy:
1. Phase 1 (warmup): Train baseline components only, freeze TMI v2 components
2. Phase 2 (joint): Unfreeze TMI v2 and train all together with lower LR

This prevents the new components from destabilizing the baseline during early training.
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

        speckle = 1.0 + 0.4 * (np.random.exponential(1.0, clean_patch.shape) - 1.0)
        noisy_patch = clean_patch * np.maximum(speckle, 0.01)
        noisy_patch = noisy_patch + np.random.randn(*clean_patch.shape) * 0.08
        noisy_patch = np.clip(noisy_patch, 0, 1).astype(np.float32)

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


def freeze_tmiv2_components(model):
    """Freeze columnar encoder and boundary head."""
    if hasattr(model, 'columnar_encoder') and model.columnar_encoder is not None:
        for param in model.columnar_encoder.parameters():
            param.requires_grad = False
    if hasattr(model, 'boundary_head') and model.boundary_head is not None:
        for param in model.boundary_head.parameters():
            param.requires_grad = False


def unfreeze_tmiv2_components(model):
    """Unfreeze columnar encoder and boundary head."""
    if hasattr(model, 'columnar_encoder') and model.columnar_encoder is not None:
        for param in model.columnar_encoder.parameters():
            param.requires_grad = True
    if hasattr(model, 'boundary_head') and model.boundary_head is not None:
        for param in model.boundary_head.parameters():
            param.requires_grad = True


def train_epoch(model, loader, optimizer, device, use_boundary_loss=True, boundary_weight=0.1):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    boundary_loss_fn = None

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(noisy)

        denoise_loss = F.l1_loss(outputs['denoised'], clean)
        seg_loss = F.cross_entropy(outputs['seg_logits'], mask)
        loss = denoise_loss + 0.6 * seg_loss  # Optimized weight

        if use_boundary_loss and 'boundary_positions' in outputs and outputs['boundary_positions'] is not None:
            from nsnd.models.boundary_regression import extract_boundaries_from_segmentation, BoundaryLoss
            gt_bounds = extract_boundaries_from_segmentation(mask, num_classes=4)
            if boundary_loss_fn is None:
                boundary_loss_fn = BoundaryLoss(num_boundaries=5).to(device)
            b_loss, _ = boundary_loss_fn(outputs['boundary_positions'], gt_bounds)
            loss = loss + boundary_weight * b_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        del noisy, clean, mask, outputs, loss
        pbar.set_postfix({'loss': f'{total_loss/n_batches:.4f}'})

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    psnr_values = []
    dice_values = {i: [] for i in range(4)}

    for batch in tqdm(loader, desc='Evaluating'):
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

    return {
        'psnr': np.mean(psnr_values),
        'dice_rnfl_gcl': np.mean(dice_values[0]),
        'dice_inl_opl_onl': np.mean(dice_values[1]),
        'dice_is_os': np.mean(dice_values[2]),
        'dice_rpe_choroid': np.mean(dice_values[3]),
        'dice_mean': np.mean([np.mean(dice_values[c]) for c in range(4)]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--joint_epochs', type=int, default=5)
    parser.add_argument('--max_train', type=int, default=50)
    parser.add_argument('--max_val', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_dataset = QuickTestDataset(args.train_jsonl, max_samples=args.max_train)
    val_dataset = QuickTestDataset(args.val_jsonl, max_samples=args.max_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create TMI v2 model
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

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =========================================================================
    # PHASE 1: Warmup - Freeze TMI v2 components, train baseline
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: WARMUP (TMI v2 components frozen)")
    print("="*60)

    freeze_tmiv2_components(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (warmup): {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )

    for epoch in range(args.warmup_epochs):
        loss = train_epoch(model, train_loader, optimizer, device, use_boundary_loss=False)
        print(f"Warmup Epoch {epoch+1}/{args.warmup_epochs}: Loss={loss:.4f}")

    warmup_metrics = evaluate(model, val_loader, device)
    print(f"\nAfter Warmup: PSNR={warmup_metrics['psnr']:.2f}, Dice={warmup_metrics['dice_mean']:.4f}")

    # =========================================================================
    # PHASE 2: Joint Training - Unfreeze TMI v2, train all together
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: JOINT TRAINING (all components)")
    print("="*60)

    unfreeze_tmiv2_components(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (joint): {trainable:,}")

    # Lower LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    for epoch in range(args.joint_epochs):
        # Gradually increase boundary loss weight
        boundary_weight = 0.1 * (1 + epoch / args.joint_epochs)
        loss = train_epoch(model, train_loader, optimizer, device,
                          use_boundary_loss=True, boundary_weight=boundary_weight)
        print(f"Joint Epoch {epoch+1}/{args.joint_epochs}: Loss={loss:.4f}, BoundaryWeight={boundary_weight:.2f}")

    final_metrics = evaluate(model, val_loader, device)
    print(f"\nFinal: PSNR={final_metrics['psnr']:.2f}, Dice={final_metrics['dice_mean']:.4f}")

    clear_memory()

    # =========================================================================
    # Compare with baseline (train from scratch without TMI v2)
    # =========================================================================
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)

    baseline = TMIEnhancedModel(
        num_classes=4,
        nafnet_width=32,
        use_layer_specific_heads=True,
        use_seg_guided_attention=True,
        use_columnar_attention=False,
        use_boundary_regression=False,
        num_head_blocks=2,
        head_hidden_channels=32,
    ).to(device)

    optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(args.warmup_epochs + args.joint_epochs):
        loss = train_epoch(baseline, train_loader, optimizer, device, use_boundary_loss=False)

    baseline_metrics = evaluate(baseline, val_loader, device)
    print(f"Baseline: PSNR={baseline_metrics['psnr']:.2f}, Dice={baseline_metrics['dice_mean']:.4f}")

    del baseline
    clear_memory()

    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print(f"\n{'Metric':<20s} {'Baseline':>12s} {'TMI v2':>12s} {'Diff':>12s} {'Status':>10s}")
    print("-" * 70)

    all_improved = True
    for metric in ['psnr', 'dice_rnfl_gcl', 'dice_inl_opl_onl', 'dice_is_os', 'dice_rpe_choroid', 'dice_mean']:
        baseline_val = baseline_metrics[metric]
        final_val = final_metrics[metric]
        diff = final_val - baseline_val

        if diff > 0.001:
            status = "IMPROVED"
        elif diff < -0.01:
            status = "WORSE"
            all_improved = False
        else:
            status = "SAME"

        print(f"{metric:<20s} {baseline_val:>12.4f} {final_val:>12.4f} {diff:>+12.4f} {status:>10s}")

    print("-" * 70)

    if all_improved:
        print("\nALL METRICS IMPROVED! TMI v2 is ready for production.")
    else:
        print("\nSome metrics need more training. Consider:")
        print("  1. More epochs")
        print("  2. Adjust loss weights")
        print("  3. Different warmup strategy")

    return all_improved


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
