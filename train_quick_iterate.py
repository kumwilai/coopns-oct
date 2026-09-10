#!/usr/bin/env python3
"""
Quick Iteration Training Script for TMI OCT Denoising + Segmentation

Fast training with small dataset to debug and iterate on metrics.
Goal: Fix all issues and achieve SOTA performance.

Target Metrics:
- PSNR: >28 dB
- SSIM: >0.90
- Dice: >0.75 per class
- IS/OS Boundary MAE: <5 px
- RNFL Thickness Error: <5 μm
- Anatomical Validity: >95%
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsnd_oct.nsnd.models.nafnet import NAFNet, NAFNetFullFiLM
from nsnd_oct.nsnd.models.boundary_aware_segmenter import BoundaryAwareSegmenter
from nsnd_oct.nsnd.models.layer_specific_heads import EnhancedLayerSpecificDenoiser
from nsnd_oct.nsnd.losses import compute_psnr

# Use skimage SSIM if available (more robust)
try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def compute_ssim_robust(pred, target, mask=None):
    """
    Compute SSIM robustly with optional mask.

    Args:
        pred: Predicted image [B, 1, H, W] or [H, W]
        target: Target image [B, 1, H, W] or [H, W]
        mask: Optional binary mask [B, 1, H, W] or [H, W] - compute SSIM only where mask > 0

    Returns:
        SSIM value (0-1, higher is better)
    """
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    # Apply mask if provided
    if mask is not None:
        mask_np = mask.squeeze().cpu().numpy()
        # Create masked version - only compute on non-zero mask regions
        if mask_np.sum() < 100:  # Need enough pixels
            return 0.0

        # Get bounding box of mask for more efficient computation
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        pred_crop = pred_np[rmin:rmax+1, cmin:cmax+1]
        target_crop = target_np[rmin:rmax+1, cmin:cmax+1]
    else:
        pred_crop = pred_np
        target_crop = target_np

    # Ensure minimum size for SSIM
    if pred_crop.shape[0] < 7 or pred_crop.shape[1] < 7:
        return 0.0

    if HAS_SKIMAGE:
        try:
            return skimage_ssim(pred_crop, target_crop, data_range=1.0)
        except Exception:
            return 0.0
    else:
        # Simple SSIM fallback
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = pred_crop.mean()
        mu_target = target_crop.mean()
        sigma_pred = pred_crop.std()
        sigma_target = target_crop.std()
        sigma_pred_target = ((pred_crop - mu_pred) * (target_crop - mu_target)).mean()

        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred**2 + sigma_target**2 + C2))

        return max(0.0, float(ssim))

# Try to import calibrated noise
try:
    from calibrated_oct_noise import add_calibrated_oct_noise
    HAS_CALIBRATED_NOISE = True
except ImportError:
    HAS_CALIBRATED_NOISE = False


def add_noise(image, noise_scale=1.0):
    """Add synthetic OCT noise."""
    speckle = 1.0 + 0.4 * noise_scale * (np.random.exponential(1.0, image.shape) - 1.0)
    noisy = image * np.maximum(speckle, 0.01)
    gaussian = np.random.randn(*image.shape).astype(np.float32) * 0.08 * noise_scale
    noisy = noisy + gaussian
    return np.clip(noisy, 0, 1).astype(np.float32)


class QuickDataset(Dataset):
    """Quick dataset for fast iteration."""

    def __init__(self, jsonl_path, max_samples=50, patch_size=64, noise_scale=1.0):
        self.samples = []
        self.patch_size = patch_size
        self.noise_scale = noise_scale

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load clean image
        clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0

        # Load 5-class mask
        mask_5class = np.array(Image.open(sample['mask_path']))

        # Convert 5-class to 3-class:
        # 0: RNFL_GCL (was 0,1)
        # 1: INL_OPL_ONL (was 2)
        # 2: RPE_Choroid (was 3,4 - includes IS_OS and RPE)
        mask_3class = np.zeros_like(mask_5class, dtype=np.int64)
        mask_3class[(mask_5class == 0) | (mask_5class == 1)] = 0  # RNFL + GCL -> class 0
        mask_3class[mask_5class == 2] = 1  # INL_OPL_ONL -> class 1
        mask_3class[(mask_5class == 3) | (mask_5class == 4)] = 2  # IS_OS + RPE -> class 2

        # Extract IS/OS boundary (transition from class 2 in 5-class to class 3)
        # This is where INL/OPL ends and IS_OS begins
        is_os_boundary = (mask_5class == 3).astype(np.float32)

        # Random crop
        H, W = clean.shape
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        clean_patch = clean[top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask_3class[top:top+self.patch_size, left:left+self.patch_size]
        boundary_patch = is_os_boundary[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            boundary_patch = np.pad(boundary_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Add noise
        noisy_patch = add_noise(clean_patch, self.noise_scale)

        return {
            'noisy': torch.from_numpy(noisy_patch).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_patch.astype(np.float32)).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_patch).long(),
            'is_os_boundary': torch.from_numpy(boundary_patch).float().unsqueeze(0),
        }


class JointModel(nn.Module):
    """Joint denoising + segmentation model."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes

        # NAFNet backbone - match calibrated checkpoint architecture
        # Checkpoint: width=64, 3 encoder/decoder stages with 2 blocks each, middle=2
        self.nafnet = NAFNetFullFiLM(
            img_channel=1,
            width=64,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2],  # 3 stages, not 4
            dec_blk_nums=[2, 2, 2],  # 3 stages, not 4
            cond_dim=4,
            condition_middle=False,  # Don't use DBM blocks for checkpoint compatibility
            condition_decoders=False,
            use_spatial_cue=True,
            spatial_cue_channels=4
        )

        # Segmenter
        self.segmenter = BoundaryAwareSegmenter(
            in_channels=1,
            num_classes=n_classes,
            base_filters=32
        )

        # Layer-specific heads
        self.layer_heads = EnhancedLayerSpecificDenoiser(
            encoder_channels=64,  # NAFNet width (64 for calibrated checkpoint)
            n_layer_classes=n_classes,
            hidden_channels=32,
            num_head_blocks=2,
            dropout=0.1,
            use_refinement=True
        )

        # Final combiner
        self.combiner = nn.Conv2d(2, 1, 1)  # Combine NAFNet + layer-specific

    def forward(self, noisy):
        # NAFNet denoising (no spatial cue for now)
        denoised_base = self.nafnet(noisy, spatial_map=None)

        # Segmentation
        seg_out = self.segmenter(denoised_base)
        seg_logits = seg_out['seg_logits']
        seg_probs = F.softmax(seg_logits, dim=1)
        seg_mask = seg_logits.argmax(dim=1)

        # Get encoder features for layer-specific heads
        # Use denoised_base as features (simplified)
        encoder_features = denoised_base.repeat(1, 64, 1, 1)[:, :64, :, :]  # Expand to 64 channels

        # Layer-specific denoising
        layer_out = self.layer_heads(
            encoder_features,
            seg_mask=seg_mask,
            seg_probs=seg_probs,
        )
        layer_denoised = layer_out['output']

        # Combine outputs
        combined = torch.cat([denoised_base, layer_denoised], dim=1)
        final_denoised = self.combiner(combined)

        return {
            'denoised': final_denoised,
            'denoised_base': denoised_base,
            'seg_logits': seg_logits,
            'seg_probs': seg_probs,
            'layer_outputs': layer_out.get('layer_outputs'),
        }


def compute_dice(pred_logits, target, class_idx):
    """Compute Dice score for a specific class."""
    pred = (pred_logits.argmax(dim=1) == class_idx).float()
    target_mask = (target == class_idx).float()

    intersection = (pred * target_mask).sum()
    union = pred.sum() + target_mask.sum()

    if union < 1:
        return None

    return (2.0 * intersection / (union + 1e-8)).item()


def compute_boundary_mae_fixed(pred_seg, target_boundary):
    """
    Compute boundary MAE properly.

    pred_seg: [B, C, H, W] logits or [B, H, W] labels
    target_boundary: [B, 1, H, W] or [B, H, W] binary mask of IS/OS boundary

    We find the boundary in pred as transition from class 1 to class 2.
    """
    if pred_seg.dim() == 4 and pred_seg.size(1) > 1:
        pred_mask = pred_seg.argmax(dim=1)  # [B, H, W]
    else:
        pred_mask = pred_seg

    if target_boundary.dim() == 4:
        target_boundary = target_boundary.squeeze(1)  # [B, H, W]

    B, H, W = pred_mask.shape
    total_mae = 0
    valid_columns = 0

    for b in range(B):
        for col in range(W):
            # Find predicted boundary: first row where class changes from 1 to 2
            pred_col = pred_mask[b, :, col]

            # Find first occurrence of class 2 (RPE/IS_OS)
            pred_class2 = (pred_col == 2).float()
            pred_positions = torch.where(pred_class2 > 0.5)[0]

            # Find target boundary position
            target_col = target_boundary[b, :, col]
            target_positions = torch.where(target_col > 0.5)[0]

            if len(pred_positions) > 0 and len(target_positions) > 0:
                pred_pos = pred_positions[0].float()
                target_pos = target_positions[0].float()
                total_mae += torch.abs(pred_pos - target_pos).item()
                valid_columns += 1

    if valid_columns == 0:
        return None

    return total_mae / valid_columns


def compute_rnfl_thickness_error(pred_seg, target_seg, pixel_to_um=3.9):
    """
    Compute RNFL thickness measurement error.

    RNFL is class 0 in our 3-class scheme.
    """
    if pred_seg.dim() == 4:
        pred_mask = pred_seg.argmax(dim=1)
    else:
        pred_mask = pred_seg

    B, H, W = pred_mask.shape
    total_error = 0
    valid_columns = 0

    for b in range(B):
        for col in range(W):
            # Count RNFL pixels in prediction
            pred_rnfl = (pred_mask[b, :, col] == 0).sum().item()
            target_rnfl = (target_seg[b, :, col] == 0).sum().item()

            if target_rnfl > 0:
                error_px = abs(pred_rnfl - target_rnfl)
                total_error += error_px * pixel_to_um
                valid_columns += 1

    if valid_columns == 0:
        return None

    return total_error / valid_columns


def compute_anatomical_validity(pred_seg, n_classes=3):
    """
    Check if predicted segmentation has valid anatomical ordering.

    Valid ordering (top to bottom): class 0 -> class 1 -> class 2
    """
    if pred_seg.dim() == 4:
        pred_mask = pred_seg.argmax(dim=1)
    else:
        pred_mask = pred_seg

    B, H, W = pred_mask.shape
    valid_count = 0
    total_count = 0

    for b in range(B):
        for col in range(W):
            col_pred = pred_mask[b, :, col]

            # Find first occurrence of each class
            class_positions = {}
            for c in range(n_classes):
                positions = torch.where(col_pred == c)[0]
                if len(positions) > 0:
                    class_positions[c] = positions[0].item()

            # Check ordering: class 0 should be above class 1, class 1 above class 2
            if len(class_positions) >= 2:
                total_count += 1
                valid = True
                for c1 in range(n_classes - 1):
                    for c2 in range(c1 + 1, n_classes):
                        if c1 in class_positions and c2 in class_positions:
                            if class_positions[c1] > class_positions[c2]:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    valid_count += 1

    if total_count == 0:
        return 0.0

    return 100.0 * valid_count / total_count


def dice_loss(pred_logits, target, smooth=1.0):
    """Compute Dice loss."""
    n_classes = pred_logits.size(1)
    pred_probs = F.softmax(pred_logits, dim=1)

    total_dice = 0
    for c in range(n_classes):
        pred_c = pred_probs[:, c]
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        total_dice += dice

    return 1.0 - total_dice / n_classes


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_psnr = 0
    n_batches = 0

    for batch in tqdm(loader, desc='Training'):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        outputs = model(noisy)
        denoised = outputs['denoised']
        seg_logits = outputs['seg_logits']

        # Denoising loss (L1)
        denoise_loss = F.l1_loss(denoised, clean)

        # Segmentation loss (Dice + weighted CE)
        # Class weights to handle imbalance: INL_OPL_ONL (class 1) is often underrepresented
        class_weights = torch.tensor([1.0, 2.0, 1.0], device=device)  # Boost INL_OPL_ONL
        d_loss = dice_loss(seg_logits, mask)
        ce_loss = F.cross_entropy(seg_logits, mask, weight=class_weights)
        seg_loss = 0.5 * d_loss + 0.5 * ce_loss

        # Combined loss
        loss = denoise_loss + 0.5 * seg_loss

        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(denoised, clean)

        total_loss += loss.item()
        total_psnr += psnr
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'psnr': total_psnr / n_batches,
    }


def validate(model, loader, device):
    """Validate and compute all metrics."""
    model.eval()

    total_psnr = 0
    total_ssim = 0
    dice_scores = {0: [], 1: [], 2: []}
    boundary_maes = []
    rnfl_errors = []
    anat_validity = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask = batch['mask'].to(device)
            is_os_boundary = batch['is_os_boundary'].to(device)

            outputs = model(noisy)
            denoised = outputs['denoised']
            seg_logits = outputs['seg_logits']

            # PSNR/SSIM
            psnr = compute_psnr(denoised, clean)

            # Create mask for SSIM - use non-background regions from segmentation
            # Any pixel that's not all zeros in the target mask is valid
            ssim_mask = (mask > 0).float().unsqueeze(1)  # [B, 1, H, W]

            ssim = compute_ssim_robust(denoised, clean, mask=ssim_mask)
            total_psnr += psnr
            total_ssim += ssim

            # Dice per class
            for c in range(3):
                dice = compute_dice(seg_logits, mask, c)
                if dice is not None:
                    dice_scores[c].append(dice)

            # Boundary MAE
            mae = compute_boundary_mae_fixed(seg_logits, is_os_boundary)
            if mae is not None:
                boundary_maes.append(mae)

            # RNFL thickness error
            rnfl_err = compute_rnfl_thickness_error(seg_logits, mask)
            if rnfl_err is not None:
                rnfl_errors.append(rnfl_err)

            # Anatomical validity
            anat = compute_anatomical_validity(seg_logits)
            anat_validity.append(anat)

            n_samples += 1

    # Aggregate metrics
    results = {
        'psnr': total_psnr / n_samples,
        'ssim': total_ssim / n_samples,
        'dice_0': np.mean(dice_scores[0]) if dice_scores[0] else 0,
        'dice_1': np.mean(dice_scores[1]) if dice_scores[1] else 0,
        'dice_2': np.mean(dice_scores[2]) if dice_scores[2] else 0,
        'boundary_mae': np.mean(boundary_maes) if boundary_maes else float('inf'),
        'rnfl_error': np.mean(rnfl_errors) if rnfl_errors else float('inf'),
        'anat_validity': np.mean(anat_validity) if anat_validity else 0,
    }

    return results


def print_results(epoch, train_metrics, val_metrics):
    """Print formatted results with GOOD/POOR labels."""
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} RESULTS")
    print(f"{'='*70}")

    print(f"\nTraining: Loss={train_metrics['loss']:.4f}, PSNR={train_metrics['psnr']:.2f} dB")

    print(f"\nValidation Metrics:")
    print(f"  {'Metric':<25} {'Value':<15} {'Target':<15} {'Status'}")
    print(f"  {'-'*65}")

    # PSNR
    psnr = val_metrics['psnr']
    status = "[GOOD]" if psnr >= 28 else "[POOR]"
    print(f"  {'PSNR':<25} {psnr:.2f} dB{'':<7} {'>28 dB':<15} {status}")

    # SSIM
    ssim = val_metrics['ssim']
    status = "[GOOD]" if ssim >= 0.90 else "[POOR]"
    print(f"  {'SSIM':<25} {ssim:.4f}{'':<9} {'>0.90':<15} {status}")

    # Dice scores
    for c, name in [(0, 'RNFL_GCL'), (1, 'INL_OPL_ONL'), (2, 'RPE_Choroid')]:
        dice = val_metrics[f'dice_{c}']
        status = "[GOOD]" if dice >= 0.75 else "[POOR]"
        print(f"  {f'Dice ({name})':<25} {dice:.4f}{'':<9} {'>0.75':<15} {status}")

    # Boundary MAE
    mae = val_metrics['boundary_mae']
    status = "[GOOD]" if mae <= 5 else "[POOR]"
    print(f"  {'IS/OS Boundary MAE':<25} {mae:.2f} px{'':<7} {'<5 px':<15} {status}")

    # RNFL Error
    rnfl = val_metrics['rnfl_error']
    status = "[GOOD]" if rnfl <= 5 else "[POOR]"
    print(f"  {'RNFL Thickness Error':<25} {rnfl:.2f} μm{'':<6} {'<5 μm':<15} {status}")

    # Anatomical validity
    anat = val_metrics['anat_validity']
    status = "[GOOD]" if anat >= 95 else "[POOR]"
    print(f"  {'Anatomical Validity':<25} {anat:.1f}%{'':<9} {'>95%':<15} {status}")

    print(f"{'='*70}\n")

    # Check if SOTA achieved
    is_sota = (
        psnr >= 28 and
        ssim >= 0.90 and
        val_metrics['dice_0'] >= 0.75 and
        val_metrics['dice_1'] >= 0.75 and
        val_metrics['dice_2'] >= 0.75 and
        mae <= 5 and
        rnfl <= 5 and
        anat >= 95
    )

    if is_sota:
        print("🎉 SOTA ACHIEVED! All metrics meet targets!")
    else:
        poor_metrics = []
        if psnr < 28: poor_metrics.append('PSNR')
        if ssim < 0.90: poor_metrics.append('SSIM')
        if val_metrics['dice_0'] < 0.75: poor_metrics.append('Dice_RNFL')
        if val_metrics['dice_1'] < 0.75: poor_metrics.append('Dice_INL')
        if val_metrics['dice_2'] < 0.75: poor_metrics.append('Dice_RPE')
        if mae > 5: poor_metrics.append('BoundaryMAE')
        if rnfl > 5: poor_metrics.append('RNFLError')
        if anat < 95: poor_metrics.append('AnatValidity')
        print(f"Metrics needing improvement: {', '.join(poor_metrics)}")

    return is_sota


def load_pretrained_weights(model, nafnet_ckpt=None, seg_ckpt=None):
    """Load pretrained weights into model."""
    if nafnet_ckpt and os.path.exists(nafnet_ckpt):
        print(f"Loading NAFNet: {nafnet_ckpt}")
        ckpt = torch.load(nafnet_ckpt, map_location='cpu')

        # Try different key formats
        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

        # If state_dict itself is not a valid state dict, skip
        if not isinstance(state_dict, dict) or 'state_dict' in state_dict:
            state_dict = state_dict.get('state_dict', state_dict)

        # Load NAFNet weights
        nafnet_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):  # Only process actual weights
                if k.startswith('nafnet.'):
                    nafnet_state[k.replace('nafnet.', '')] = v
                elif not any(k.startswith(p) for p in ['seg', 'layer_heads', 'combiner']):
                    nafnet_state[k] = v

        if nafnet_state:
            try:
                missing, unexpected = model.nafnet.load_state_dict(nafnet_state, strict=False)
                loaded = len(nafnet_state) - len(missing)
                print(f"  Loaded {loaded}/{len(nafnet_state)} NAFNet weights (missing: {len(missing)})")
            except Exception as e:
                print(f"  Warning: Could not load NAFNet weights: {e}")

    # Skip V4 checkpoint for now - architecture mismatch
    # Segmentation will train from scratch
    print("  Segmentation training from scratch (V4 architecture mismatch)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=50)
    parser.add_argument('--max_val', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='outputs/quick_iterate')
    parser.add_argument('--nafnet_ckpt', default='outputs/nafnet_calibrated/nafnet_best.pth')
    parser.add_argument('--seg_ckpt', default='best_boundary_model_v4.pth')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Quick Iteration Training")
    print(f"  Train samples: {args.max_train}")
    print(f"  Val samples: {args.max_val}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")

    # Data
    train_dataset = QuickDataset(args.train_jsonl, args.max_train, patch_size=64)
    val_dataset = QuickDataset(args.val_jsonl, args.max_val, patch_size=64)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = JointModel(n_classes=3)

    # Load pretrained weights
    load_pretrained_weights(model, args.nafnet_ckpt, args.seg_ckpt)

    model = model.to(args.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_psnr = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device)

        # Validate
        val_metrics = validate(model, val_loader, args.device)

        # Print results
        is_sota = print_results(epoch, train_metrics, val_metrics)

        # Save best
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model (PSNR: {best_psnr:.2f} dB)")

        scheduler.step()

        if is_sota:
            print("SOTA achieved! Stopping early.")
            break

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
