#!/usr/bin/env python3
"""
NSND-MultiTask Training Script

Train the Neuro-Symbolic Noise Decomposition model with layer-aware denoising.

KEY TMI CONTRIBUTIONS:
1. Neuro-symbolic noise decomposition - interpretable noise classification
2. Layer-aware component fusion - anatomical structure guides denoising
3. Physics-based component denoisers - speckle, banding, Gaussian, shot
4. Joint denoising + segmentation training

Usage:
    python train_nsnd_multitask.py --data_jsonl combined_train.jsonl --val_jsonl combined_val.jsonl

"""

import os
import sys
import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tifffile import imread as tiff_imread

# Import the NSND-MultiTask model
from nsnd_multitask_model import (
    NSNDMultiTaskDenoiser,
    NSNDConsistencyLoss,
    NSNDComponentLoss,
    NSNDLayerAwareLoss,
    count_parameters,
)


# =============================================================================
# Dataset
# =============================================================================

class NSNDMultiTaskDataset(Dataset):
    """
    Dataset for NSND-MultiTask training.

    Supports multiple data formats:
    1. Joint data (combined A+B): noisy_path, clean_path, mask_path, noise_type
    2. Denoising pairs: noisy_path, clean_path
    3. Segmentation-only: image_path, mask_path (will add synthetic noise)

    Clinical priors are applied based on mask_type (real vs pseudo).
    """

    LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']

    def __init__(
        self,
        jsonl_path: str,
        patch_size: int = 256,
        augment: bool = True,
        num_classes: int = 5,
        add_synthetic_noise: bool = False,
        noise_sigma: float = 0.126,
        multi_level_noise: bool = False,
        noise_levels: list = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with sample data
            patch_size: Size of random crops
            augment: Whether to apply augmentation
            num_classes: Number of segmentation classes
            add_synthetic_noise: If True, add noise to clean images (for seg-only data)
            noise_sigma: Base noise standard deviation (default: 0.126 from real OCT analysis)
            multi_level_noise: If True, randomly select noise level per sample
            noise_levels: List of noise levels for multi-level augmentation
        """
        self.patch_size = patch_size
        self.augment = augment
        self.num_classes = num_classes
        self.add_synthetic_noise = add_synthetic_noise
        self.noise_sigma = noise_sigma
        self.multi_level_noise = multi_level_noise
        self.noise_levels = noise_levels or [0.05, 0.10, 0.15, 0.20, 0.25]

        # Load samples
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        # Analyze dataset composition
        n_synthetic = sum(1 for s in self.samples if s.get('noise_type') == 'synthetic')
        n_real = sum(1 for s in self.samples if s.get('noise_type') == 'real')
        n_other = len(self.samples) - n_synthetic - n_real

        print(f"[NSNDDataset] Loaded {len(self.samples)} samples from {jsonl_path}")
        if n_synthetic > 0 or n_real > 0:
            print(f"[NSNDDataset] Composition: synthetic={n_synthetic}, real={n_real}, other={n_other}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image and normalize to [0, 1]."""
        if path.endswith('.tif') or path.endswith('.tiff'):
            img = tiff_imread(path)
        else:
            img = np.array(Image.open(path))

        # Handle multi-page TIFF (take first page)
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = img[0]

        # Normalize
        img = img.astype(np.float32)
        if img.max() > 1.0:
            if img.max() > 255:
                img = img / 65535.0
            else:
                img = img / 255.0

        return img

    def _load_segmentation(self, path: str) -> np.ndarray:
        """Load segmentation mask with layer indices 0-4."""
        seg = np.array(Image.open(path))

        # Convert to layer indices if needed
        if seg.max() > self.num_classes:
            # Assume grayscale values map to layers
            seg = np.digitize(seg, bins=np.linspace(0, 255, self.num_classes + 1)[1:-1])

        seg = seg.astype(np.int64)
        seg = np.clip(seg, 0, self.num_classes - 1)

        return seg

    def _add_realistic_noise(self, clean: np.ndarray, sigma: float = None) -> np.ndarray:
        """Add realistic mixed OCT noise for NSND training.

        Simulates 4 noise types that NSND is designed to handle:
        1. Speckle: Multiplicative noise (coherent imaging artifact)
        2. Banding: Horizontal stripe artifacts (electronic interference)
        3. Gaussian: Additive thermal/readout noise
        4. Shot (Poisson): Signal-dependent photon noise

        Args:
            clean: Clean image normalized to [0, 1]
            sigma: Overall noise level (if None, uses self.noise_sigma or random)
        """
        from scipy.ndimage import convolve
        H, W = clean.shape

        # Determine noise level
        if sigma is None:
            if self.multi_level_noise:
                sigma = random.choice(self.noise_levels)
            else:
                sigma = self.noise_sigma

        # Random mix of noise types (varies per sample for robustness)
        # Typical OCT: 40% speckle, 10% banding, 35% gaussian, 15% shot
        mix = np.random.dirichlet([4.0, 1.0, 3.5, 1.5])  # Random around typical
        speckle_weight, banding_weight, gaussian_weight, shot_weight = mix

        noisy = clean.copy()

        # 1. SPECKLE NOISE (multiplicative)
        # Models coherent interference in OCT
        if speckle_weight > 0.05:
            speckle_sigma = sigma * speckle_weight * 2.0
            speckle = np.random.exponential(1.0, (H, W)).astype(np.float32)
            speckle = (speckle - 1.0) * speckle_sigma + 1.0  # Center around 1
            noisy = noisy * speckle

        # 2. BANDING NOISE (horizontal stripes)
        # Models electronic interference, DC offset variations
        if banding_weight > 0.05:
            banding_sigma = sigma * banding_weight * 1.5
            # Create horizontal bands at random frequencies
            n_bands = random.randint(2, 6)
            banding = np.zeros((H, W), dtype=np.float32)
            for _ in range(n_bands):
                freq = random.uniform(0.01, 0.1)  # Low frequency
                phase = random.uniform(0, 2 * np.pi)
                amplitude = random.uniform(0.5, 1.0) * banding_sigma
                rows = np.arange(H).reshape(-1, 1)
                banding += amplitude * np.sin(2 * np.pi * freq * rows + phase)
            noisy = noisy + banding

        # 3. GAUSSIAN NOISE (additive, with vertical correlation for OCT)
        # Models thermal noise, readout noise
        if gaussian_weight > 0.05:
            gaussian_sigma = sigma * gaussian_weight * 1.5
            gaussian = np.random.randn(H, W).astype(np.float32) * gaussian_sigma
            # Add vertical correlation (typical of A-scan acquisition)
            kernel = np.array([[0.2], [0.6], [0.2]], dtype=np.float32)
            gaussian = convolve(gaussian, kernel, mode='reflect')
            gaussian = gaussian * (gaussian_sigma / (gaussian.std() + 1e-8))
            noisy = noisy + gaussian

        # 4. SHOT NOISE (Poisson, signal-dependent)
        # Models photon counting statistics
        if shot_weight > 0.05:
            shot_scale = sigma * shot_weight * 50  # Scale for Poisson
            # Poisson noise is signal-dependent
            noisy_scaled = np.maximum(noisy * shot_scale, 0.1)
            shot_noisy = np.random.poisson(noisy_scaled).astype(np.float32) / shot_scale
            noisy = shot_noisy

        noisy = np.clip(noisy, 0, 1).astype(np.float32)

        return noisy

    def _random_crop(self, *arrays, size: int):
        """Random crop multiple arrays to the same location."""
        H, W = arrays[0].shape[:2]

        if H < size or W < size:
            # Pad if needed
            pad_h = max(0, size - H)
            pad_w = max(0, size - W)
            arrays = [np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect') for arr in arrays]
            H, W = arrays[0].shape[:2]

        y = random.randint(0, H - size)
        x = random.randint(0, W - size)

        return [arr[y:y+size, x:x+size] for arr in arrays]

    def _augment(self, *arrays):
        """Apply random augmentation to multiple arrays."""
        # Random horizontal flip
        if random.random() > 0.5:
            arrays = [np.flip(arr, axis=1).copy() for arr in arrays]

        # Random vertical flip (less common for OCT but useful)
        if random.random() > 0.8:
            arrays = [np.flip(arr, axis=0).copy() for arr in arrays]

        return arrays

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Handle different JSONL formats:
        # Format 1: noisy_path, clean_path, mask_path (joint data from combined A+B)
        # Format 2: noisy_path, clean_path (paired denoising data)
        # Format 3: image_path, mask_path (segmentation-only data)

        need_synthetic_noise = False

        if 'noisy_path' in sample and os.path.exists(sample['noisy_path']):
            # Has pre-computed noisy image (from joint data or denoising pairs)
            noisy = self._load_image(sample['noisy_path'])
            clean = self._load_image(sample['clean_path'])
        elif 'image_path' in sample:
            # Segmentation-only data (image_path = clean, need to add noise)
            clean = self._load_image(sample['image_path'])
            noisy = clean.copy()
            need_synthetic_noise = True
        elif 'clean_path' in sample:
            # Has clean but no noisy (need to add synthetic noise)
            clean = self._load_image(sample['clean_path'])
            noisy = clean.copy()
            need_synthetic_noise = True
        else:
            raise ValueError(f"Unknown sample format: {sample.keys()}")

        # Load segmentation mask
        seg_path = sample.get('mask_path') or sample.get('seg_path')
        if seg_path and os.path.exists(seg_path):
            seg_mask = self._load_segmentation(seg_path)
        else:
            # Create depth-based proxy segmentation
            H, W = clean.shape
            seg_mask = np.zeros((H, W), dtype=np.int64)
            boundaries = [int(H * 0.15), int(H * 0.35), int(H * 0.55), int(H * 0.75)]
            for i, b in enumerate(boundaries):
                seg_mask[b:] = i + 1

        # Add synthetic noise if needed
        if need_synthetic_noise or self.add_synthetic_noise:
            # Use pre-specified noise level if available
            noise_level = sample.get('noise_level', None)
            noisy = self._add_realistic_noise(clean, sigma=noise_level)

        # Random crop
        noisy, clean, seg_mask = self._random_crop(noisy, clean, seg_mask, size=self.patch_size)

        # Augmentation
        if self.augment:
            noisy, clean, seg_mask = self._augment(noisy, clean, seg_mask)

        # Convert to tensors
        noisy = torch.from_numpy(noisy.copy()).unsqueeze(0).float()
        clean = torch.from_numpy(clean.copy()).unsqueeze(0).float()
        seg_mask = torch.from_numpy(seg_mask.copy()).long()

        return {
            'noisy': noisy,
            'clean': clean,
            'seg_mask': seg_mask,
            'noise_type': sample.get('noise_type', 'unknown'),
            'mask_type': sample.get('mask_type', 'unknown'),
        }


# =============================================================================
# Loss Functions
# =============================================================================

def soft_dice_loss(pred_logits, target, num_classes=5, smooth=1.0, class_weights=None):
    """Soft Dice loss for segmentation."""
    pred_probs = F.softmax(pred_logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dice_per_class = []
    for c in range(num_classes):
        pred_c = pred_probs[:, c]
        target_c = target_one_hot[:, c]

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)

    dice_tensor = torch.stack(dice_per_class)

    if class_weights is not None:
        class_weights = class_weights.to(dice_tensor.device)
        weighted_dice = (dice_tensor * class_weights).sum() / class_weights.sum()
        return 1 - weighted_dice
    else:
        return 1 - dice_tensor.mean()


def combined_seg_loss(pred_logits, target, ce_weight=None, dice_weight=None, num_classes=5):
    """Combined CE + Dice loss for segmentation."""
    ce_loss = F.cross_entropy(pred_logits, target, weight=ce_weight)
    dice_loss = soft_dice_loss(pred_logits, target, num_classes=num_classes, class_weights=dice_weight)
    return 0.5 * ce_loss + 0.5 * dice_loss


def compute_class_weights(loader, num_classes=5, device='cpu', max_batches=50):
    """Compute class weights from training data."""
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        seg_mask = batch['seg_mask']
        for c in range(num_classes):
            class_counts[c] += (seg_mask == c).sum().item()

    median_count = class_counts[class_counts > 0].median().item() if (class_counts > 0).any() else 1.0
    class_counts = torch.where(class_counts > 0, class_counts, torch.tensor(median_count))

    weights = (median_count / class_counts).sqrt()
    weights = weights / weights.sum() * num_classes

    return weights.float().to(device)


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    lambda_seg=1.0,
    lambda_consistency=0.1,
    lambda_component=0.1,
    lambda_layer_aware=0.5,
    seg_class_weights=None,
    use_amp=True,
    verbose=True,
):
    """Train one epoch with comprehensive metrics tracking."""
    model.train()

    # Loss functions
    consistency_loss_fn = NSNDConsistencyLoss()
    component_loss_fn = NSNDComponentLoss()
    layer_aware_loss_fn = NSNDLayerAwareLoss()

    # Metrics accumulators
    total_loss = 0.0
    total_recon_loss = 0.0
    total_seg_loss = 0.0
    total_consistency_loss = 0.0
    total_component_loss = 0.0

    # Noise weight tracking
    noise_weights_sum = {'speckle': 0.0, 'banding': 0.0, 'gaussian': 0.0, 'shot': 0.0}
    noise_weight_counts = 0

    # Component PSNR tracking
    component_psnr_sum = {'speckle': 0.0, 'banding': 0.0, 'gaussian': 0.0, 'shot': 0.0}
    component_counts = 0

    # Noise type distribution (from dataset)
    noise_type_counts = {'synthetic': 0, 'real': 0, 'unknown': 0}

    for batch_idx, batch in enumerate(loader):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        seg_mask = batch['seg_mask'].to(device)

        # Track noise types
        for nt in batch.get('noise_type', ['unknown'] * noisy.shape[0]):
            noise_type_counts[nt] = noise_type_counts.get(nt, 0) + 1

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp and torch.cuda.is_available()):
            # Forward pass
            denoised, seg_logits, intermediates = model(noisy, return_intermediates=True)

            # 1. Reconstruction loss (layer-aware)
            recon_loss = layer_aware_loss_fn(denoised, clean, seg_mask)

            # 2. Segmentation loss
            seg_loss = combined_seg_loss(
                seg_logits, seg_mask,
                ce_weight=seg_class_weights,
                dice_weight=seg_class_weights,
            )

            # 3. NSND consistency loss
            noise_weights = intermediates['noise_weights']
            consistency_loss = consistency_loss_fn(noise_weights, denoised, clean, noisy)

            # 4. Component loss
            component_outputs = intermediates['component_outputs']
            component_loss = component_loss_fn(component_outputs, noise_weights, clean)

            # Total loss
            loss = (
                recon_loss +
                lambda_seg * seg_loss +
                lambda_consistency * consistency_loss +
                lambda_component * component_loss
            )

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_seg_loss += seg_loss.item()
        total_consistency_loss += consistency_loss.item()
        total_component_loss += component_loss.item()

        # Track noise weights (average over batch)
        for comp in ['speckle', 'banding', 'gaussian', 'shot']:
            noise_weights_sum[comp] += noise_weights[comp].mean().item()
        noise_weight_counts += 1

        # Track component denoiser performance
        with torch.no_grad():
            for comp in ['speckle', 'banding', 'gaussian', 'shot']:
                comp_out = component_outputs[comp]
                mse = F.mse_loss(comp_out, clean).item()
                psnr = 10 * math.log10(1.0 / (mse + 1e-10))
                component_psnr_sum[comp] += psnr
            component_counts += 1

        # Print progress
        if verbose and (batch_idx + 1) % 20 == 0:
            # Compute batch PSNR
            with torch.no_grad():
                batch_mse = F.mse_loss(denoised, clean).item()
                batch_psnr = 10 * math.log10(1.0 / (batch_mse + 1e-10))

            print(f"  Batch {batch_idx+1}/{len(loader)}: "
                  f"loss={loss.item():.4f}, PSNR={batch_psnr:.1f}dB, "
                  f"recon={recon_loss.item():.4f}, seg={seg_loss.item():.4f}")

    n_batches = len(loader)

    # Compute averages
    avg_noise_weights = {k: v / noise_weight_counts for k, v in noise_weights_sum.items()}
    avg_component_psnr = {k: v / component_counts for k, v in component_psnr_sum.items()}

    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'seg_loss': total_seg_loss / n_batches,
        'consistency_loss': total_consistency_loss / n_batches,
        'component_loss': total_component_loss / n_batches,
        'noise_weights': avg_noise_weights,
        'component_psnr': avg_component_psnr,
        'noise_type_counts': noise_type_counts,
    }


@torch.no_grad()
def validate(model, loader, device, use_amp=True, baseline_model=None):
    """Validate model with comprehensive per-layer metrics and baseline comparison."""
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()

    LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']
    CLINICAL_WEIGHTS = [1.5, 1.0, 0.8, 1.3, 1.2]  # Clinical importance

    # Metrics accumulators
    total_input_psnr = 0.0
    total_output_psnr = 0.0
    total_baseline_psnr = 0.0
    total_input_ssim = 0.0
    total_output_ssim = 0.0
    total_baseline_ssim = 0.0
    n_samples = 0

    # Per-layer Dice tracking
    layer_dice_sum = {name: 0.0 for name in LAYER_NAMES}
    layer_dice_count = {name: 0 for name in LAYER_NAMES}

    # Noise weight statistics
    noise_weights_sum = {'speckle': 0.0, 'banding': 0.0, 'gaussian': 0.0, 'shot': 0.0}
    noise_weight_batches = 0

    # SSIM computation
    try:
        from skimage.metrics import structural_similarity as ssim
        has_ssim = True
    except ImportError:
        has_ssim = False

    for batch in loader:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        seg_mask = batch['seg_mask'].to(device)

        with autocast('cuda', enabled=use_amp and torch.cuda.is_available()):
            denoised, seg_logits, intermediates = model(noisy, return_intermediates=True)

            # Baseline model inference
            if baseline_model is not None:
                baseline_out = baseline_model(noisy)
                if isinstance(baseline_out, tuple):
                    baseline_denoised = baseline_out[0]
                else:
                    baseline_denoised = baseline_out

        # Input PSNR (noisy vs clean)
        input_mse = F.mse_loss(noisy, clean, reduction='none').mean(dim=(1, 2, 3))
        input_psnr = 10 * torch.log10(1.0 / (input_mse + 1e-10))
        total_input_psnr += input_psnr.sum().item()

        # Output PSNR (denoised vs clean)
        output_mse = F.mse_loss(denoised, clean, reduction='none').mean(dim=(1, 2, 3))
        output_psnr = 10 * torch.log10(1.0 / (output_mse + 1e-10))
        total_output_psnr += output_psnr.sum().item()

        # Baseline PSNR
        if baseline_model is not None:
            baseline_mse = F.mse_loss(baseline_denoised, clean, reduction='none').mean(dim=(1, 2, 3))
            baseline_psnr = 10 * torch.log10(1.0 / (baseline_mse + 1e-10))
            total_baseline_psnr += baseline_psnr.sum().item()

        # SSIM (per sample)
        if has_ssim:
            for i in range(denoised.shape[0]):
                n = noisy[i, 0].cpu().numpy()
                d = denoised[i, 0].cpu().numpy()
                c = clean[i, 0].cpu().numpy()
                total_input_ssim += ssim(n, c, data_range=1.0)
                total_output_ssim += ssim(d, c, data_range=1.0)
                if baseline_model is not None:
                    b = baseline_denoised[i, 0].cpu().numpy()
                    total_baseline_ssim += ssim(b, c, data_range=1.0)

        # Per-layer Dice
        pred_seg = seg_logits.argmax(dim=1)
        for c, name in enumerate(LAYER_NAMES):
            pred_c = (pred_seg == c).float()
            target_c = (seg_mask == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            if union > 0:
                layer_dice_sum[name] += (2.0 * intersection / union).item()
                layer_dice_count[name] += 1

        # Track noise weights
        if intermediates and 'noise_weights' in intermediates:
            for comp in ['speckle', 'banding', 'gaussian', 'shot']:
                noise_weights_sum[comp] += intermediates['noise_weights'][comp].mean().item()
            noise_weight_batches += 1

        n_samples += noisy.shape[0]

    # Compute per-layer Dice
    layer_dice = {}
    for name in LAYER_NAMES:
        if layer_dice_count[name] > 0:
            layer_dice[name] = layer_dice_sum[name] / layer_dice_count[name]
        else:
            layer_dice[name] = 0.0

    # Compute clinical-weighted Dice
    clinical_dice = sum(
        layer_dice[name] * weight
        for name, weight in zip(LAYER_NAMES, CLINICAL_WEIGHTS)
    ) / sum(CLINICAL_WEIGHTS)

    # Average noise weights
    avg_noise_weights = {}
    if noise_weight_batches > 0:
        avg_noise_weights = {k: v / noise_weight_batches for k, v in noise_weights_sum.items()}

    # Compute improvements
    input_psnr_avg = total_input_psnr / n_samples
    output_psnr_avg = total_output_psnr / n_samples
    input_ssim_avg = total_input_ssim / n_samples if has_ssim else 0.0
    output_ssim_avg = total_output_ssim / n_samples if has_ssim else 0.0

    result = {
        'input_psnr': input_psnr_avg,
        'psnr': output_psnr_avg,
        'psnr_improvement': output_psnr_avg - input_psnr_avg,
        'input_ssim': input_ssim_avg,
        'ssim': output_ssim_avg,
        'ssim_improvement': output_ssim_avg - input_ssim_avg,
        'dice': sum(layer_dice.values()) / len(layer_dice),
        'clinical_dice': clinical_dice,
        'layer_dice': layer_dice,
        'noise_weights': avg_noise_weights,
    }

    # Add baseline comparison if available
    if baseline_model is not None:
        baseline_psnr_avg = total_baseline_psnr / n_samples
        baseline_ssim_avg = total_baseline_ssim / n_samples if has_ssim else 0.0
        result['baseline_psnr'] = baseline_psnr_avg
        result['baseline_ssim'] = baseline_ssim_avg
        result['improvement_over_baseline_psnr'] = output_psnr_avg - baseline_psnr_avg
        result['improvement_over_baseline_ssim'] = output_ssim_avg - baseline_ssim_avg

    return result


@torch.no_grad()
def print_noise_analysis(model, loader, device):
    """Print noise analysis for a few samples."""
    model.eval()

    for i, batch in enumerate(loader):
        if i >= 2:
            break

        noisy = batch['noisy'].to(device)
        print(f"\n--- Sample {i+1} ---")
        print(model.get_noise_analysis_report(noisy[:1]))


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train NSND-MultiTask model')

    # Data
    parser.add_argument('--data_jsonl', type=str, required=True, help='Training data JSONL')
    parser.add_argument('--val_jsonl', type=str, default=None, help='Validation data JSONL')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')

    # Model
    parser.add_argument('--use_neuro_symbolic', action='store_true', default=True,
                        help='Use neuro-symbolic analyzer (learnable weights)')
    parser.add_argument('--gaussian_type', type=str, default='dncnn',
                        choices=['dncnn', 'nafnet'], help='Gaussian denoiser type')
    parser.add_argument('--feature_channels', type=int, default=64, help='Fusion feature channels')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    # Loss weights
    parser.add_argument('--lambda_seg', type=float, default=1.0, help='Segmentation loss weight')
    parser.add_argument('--lambda_consistency', type=float, default=0.1, help='NSND consistency weight')
    parser.add_argument('--lambda_component', type=float, default=0.1, help='Component loss weight')

    # Training phases
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Epochs to train only fusion (frozen denoisers)')
    parser.add_argument('--finetune_all', action='store_true',
                        help='Fine-tune all components after warmup')

    # Baseline comparison
    parser.add_argument('--baseline_ckpt', type=str, default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth',
                        help='NAFNet baseline checkpoint for comparison')

    # Misc
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints/nsnd_multitask',
                        help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = NSNDMultiTaskDataset(
        args.data_jsonl,
        patch_size=args.patch_size,
        augment=True,
        add_synthetic_noise=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_dataset = NSNDMultiTaskDataset(
            args.val_jsonl,
            patch_size=args.patch_size,
            augment=False,
            add_synthetic_noise=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Create model
    print("\n=== Creating Model ===")
    model = NSNDMultiTaskDenoiser(
        use_neuro_symbolic=args.use_neuro_symbolic,
        gaussian_type=args.gaussian_type,
        feature_channels=args.feature_channels,
        device=device,
    ).to(device)

    # Print parameter counts
    counts = count_parameters(model)
    print("\nParameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    # Load NAFNet baseline for comparison (optional)
    baseline_model = None
    if args.baseline_ckpt and os.path.exists(args.baseline_ckpt):
        print(f"\n=== Loading NAFNet Baseline ===")
        print(f"Checkpoint: {args.baseline_ckpt}")
        try:
            from models.nafnet import NAFNetDenoiser
            baseline_model = NAFNetDenoiser(width=64).to(device)
            ckpt = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
            baseline_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            baseline_model.eval()
            for p in baseline_model.parameters():
                p.requires_grad = False
            print("NAFNet baseline loaded for comparison")
        except Exception as e:
            print(f"Could not load baseline: {e}")
            baseline_model = None
    else:
        print("\n=== No Baseline Model ===")
        print("(Set --baseline_ckpt to compare against NAFNet)")

    # Compute class weights
    print("\n=== Computing Class Weights ===")
    seg_class_weights = compute_class_weights(train_loader, device=device)
    print(f"Class weights: {[f'{w:.3f}' for w in seg_class_weights.tolist()]}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # Training loop
    print("\n=== Starting Training ===")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Print configuration summary
    print("\n" + "=" * 60)
    print("NSND-MULTITASK TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Data:        {args.data_jsonl} ({len(train_dataset)} samples)")
    print(f"  Validation:  {args.val_jsonl}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Patch size:  {args.patch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Warmup:      {args.warmup_epochs} epochs")
    print("")
    print("Loss weights:")
    print(f"  λ_seg:         {args.lambda_seg}")
    print(f"  λ_consistency: {args.lambda_consistency}")
    print(f"  λ_component:   {args.lambda_component}")
    print("")
    print("Clinical importance weights:")
    print("  RNFL_GCL: 1.5 (glaucoma)")
    print("  INL_OPL:  1.0 (standard)")
    print("  ONL:      0.8 (thicker)")
    print("  IS_OS:    1.3 (visual acuity)")
    print("  RPE:      1.2 (AMD)")
    print("")
    print("Noise model: Mixed (speckle + banding + gaussian + shot)")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        # Phase-based training
        if epoch < args.warmup_epochs:
            print("Phase: Warmup (frozen symbolic analyzer and denoisers)")
            model.freeze_symbolic_and_denoisers()
        elif args.finetune_all:
            print("Phase: Full fine-tuning")
            model.unfreeze_all()

        # Train
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            lambda_seg=args.lambda_seg,
            lambda_consistency=args.lambda_consistency,
            lambda_component=args.lambda_component,
            seg_class_weights=seg_class_weights,
        )
        train_time = time.time() - t0

        # Print comprehensive training metrics
        print(f"\n--- Training Metrics ---")
        print(f"  Total loss:       {train_metrics['loss']:.4f}")
        print(f"  Reconstruction:   {train_metrics['recon_loss']:.4f}")
        print(f"  Segmentation:     {train_metrics['seg_loss']:.4f}")
        print(f"  Consistency:      {train_metrics['consistency_loss']:.4f}")
        print(f"  Component:        {train_metrics['component_loss']:.4f}")
        print(f"  Time:             {train_time:.1f}s")

        # Print noise weight distribution
        nw = train_metrics.get('noise_weights', {})
        if nw:
            print(f"\n--- Noise Weight Distribution ---")
            print(f"  Speckle:  {nw.get('speckle', 0)*100:.1f}%")
            print(f"  Banding:  {nw.get('banding', 0)*100:.1f}%")
            print(f"  Gaussian: {nw.get('gaussian', 0)*100:.1f}%")
            print(f"  Shot:     {nw.get('shot', 0)*100:.1f}%")

        # Print component denoiser performance
        cp = train_metrics.get('component_psnr', {})
        if cp:
            print(f"\n--- Component Denoiser PSNR ---")
            print(f"  Speckle:  {cp.get('speckle', 0):.1f} dB")
            print(f"  Banding:  {cp.get('banding', 0):.1f} dB")
            print(f"  Gaussian: {cp.get('gaussian', 0):.1f} dB")
            print(f"  Shot:     {cp.get('shot', 0):.1f} dB")

        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device, baseline_model=baseline_model)

            print(f"\n--- Denoising Performance ---")
            print(f"  Input PSNR:     {val_metrics['input_psnr']:.2f} dB")
            print(f"  Output PSNR:    {val_metrics['psnr']:.2f} dB")
            print(f"  Improvement:    +{val_metrics['psnr_improvement']:.2f} dB")

            if 'baseline_psnr' in val_metrics:
                print(f"\n--- Comparison with NAFNet Baseline ---")
                print(f"  NAFNet PSNR:    {val_metrics['baseline_psnr']:.2f} dB")
                print(f"  NSND PSNR:      {val_metrics['psnr']:.2f} dB")
                imp = val_metrics['improvement_over_baseline_psnr']
                print(f"  Improvement:    {'+' if imp >= 0 else ''}{imp:.2f} dB {'(better)' if imp > 0 else '(worse)' if imp < 0 else ''}")

            print(f"\n--- SSIM ---")
            print(f"  Input SSIM:     {val_metrics['input_ssim']:.4f}")
            print(f"  Output SSIM:    {val_metrics['ssim']:.4f}")
            print(f"  Improvement:    +{val_metrics['ssim_improvement']:.4f}")

            print(f"\n--- Segmentation ---")
            print(f"  Mean Dice:      {val_metrics['dice']:.4f}")
            print(f"  Clinical Dice:  {val_metrics['clinical_dice']:.4f}")

            # Print per-layer Dice
            ld = val_metrics.get('layer_dice', {})
            if ld:
                print(f"\n--- Per-Layer Dice Scores ---")
                print(f"  RNFL_GCL:   {ld.get('RNFL_GCL', 0):.4f}  (weight: 1.5)")
                print(f"  INL_OPL:    {ld.get('INL_OPL', 0):.4f}  (weight: 1.0)")
                print(f"  ONL:        {ld.get('ONL', 0):.4f}  (weight: 0.8)")
                print(f"  IS_OS:      {ld.get('IS_OS', 0):.4f}  (weight: 1.3)")
                print(f"  RPE_Choroid:{ld.get('RPE_Choroid', 0):.4f}  (weight: 1.2)")

            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"\n  >> Saved best model (PSNR: {best_psnr:.2f} dB)")

        # Update scheduler
        scheduler.step()

        # Print noise analysis periodically
        if (epoch + 1) % 10 == 0:
            print("\n--- Noise Analysis Sample ---")
            print_noise_analysis(model, train_loader, device)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'args': vars(args),
            }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))

    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_psnr': best_psnr,
        'args': vars(args),
    }, os.path.join(args.save_dir, 'final_model.pth'))

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("")
    print("KEY TMI CONTRIBUTIONS:")
    print("  1. Neuro-symbolic noise decomposition (interpretable)")
    print("  2. Physics-based component denoisers")
    print("  3. Layer-aware fusion with clinical weights")
    print("  4. Joint denoising + segmentation training")
    print("=" * 60)


if __name__ == '__main__':
    main()
