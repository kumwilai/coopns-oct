#!/usr/bin/env python3
"""
Train Constrained Neuro-Symbolic OCT Denoiser on PKU37 dataset.

Uses augmented Lagrangian for hard constraint satisfaction.
Supports JSONL data format from existing pipeline.

Key features:
- Sliding window validation on full images
- NAFNet baseline comparison (whole-image)
- Per-layer improvement tracking
"""

import os
import sys
import argparse
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/kumwilai/OCT')

from constrained_neurosymbolic_denoiser import (
    ConstrainedNeuroSymbolicDenoiser,
    DenoiserConfig,
    ClinicalLosses,
)

# Try to import NAFNet for baseline comparison
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Sliding Window Inference
# =============================================================================

def sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: int = 128,
    stride: int = 96,  # Large stride for speed
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    """Fast sliding window inference with minimal overlap."""
    model.eval()
    B, C, H, W = image.shape
    image_dev = image.to(device)

    # Small images - process directly
    if H <= patch_size and W <= patch_size:
        pad_h, pad_w = max(0, patch_size - H), max(0, patch_size - W)
        if pad_h > 0 or pad_w > 0:
            image_dev = F.pad(image_dev, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            outputs = model(image_dev)
        if pad_h > 0 or pad_w > 0:
            outputs['denoised'] = outputs['denoised'][:, :, :H, :W]
            outputs['soft_masks'] = outputs['soft_masks'][:, :, :H, :W]
        return outputs

    # Pre-allocate on device
    denoised_sum = torch.zeros(1, 1, H, W, device=device)
    masks_sum = torch.zeros(1, 4, H, W, device=device)
    weight_sum = torch.zeros(1, 1, H, W, device=device)

    # Calculate positions - ensure coverage
    h_pos = list(range(0, H - patch_size + 1, stride))
    w_pos = list(range(0, W - patch_size + 1, stride))
    if not h_pos or h_pos[-1] + patch_size < H:
        h_pos.append(max(0, H - patch_size))
    if not w_pos or w_pos[-1] + patch_size < W:
        w_pos.append(max(0, W - patch_size))

    # Simple box window (faster than Gaussian, similar quality for large stride)
    window = torch.ones(1, 1, patch_size, patch_size, device=device)

    # Single boundary estimate from center patch
    ch, cw = H // 2 - patch_size // 2, W // 2 - patch_size // 2
    ch, cw = max(0, ch), max(0, cw)

    with torch.no_grad():
        # Get boundaries from center patch
        center_patch = image_dev[:, :, ch:ch+patch_size, cw:cw+patch_size]
        center_out = model(center_patch)
        boundaries = center_out['boundaries'].detach()

        # Process all patches
        for h in h_pos:
            for w in w_pos:
                patch = image_dev[:, :, h:h+patch_size, w:w+patch_size]
                out = model(patch)
                denoised_sum[:, :, h:h+patch_size, w:w+patch_size] += out['denoised']
                masks_sum[:, :, h:h+patch_size, w:w+patch_size] += out['soft_masks']
                weight_sum[:, :, h:h+patch_size, w:w+patch_size] += window

    # Average
    denoised = denoised_sum / weight_sum.clamp(min=1e-8)
    soft_masks = masks_sum / weight_sum.clamp(min=1e-8)
    soft_masks = soft_masks / soft_masks.sum(dim=1, keepdim=True).clamp(min=1e-8)

    return {'denoised': denoised, 'soft_masks': soft_masks, 'boundaries': boundaries}


def nafnet_sliding_window(
    nafnet: nn.Module,
    image: torch.Tensor,
    patch_size: int = 128,
    stride: int = 96,  # Large stride for speed
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Fast sliding window for NAFNet baseline."""
    nafnet.eval()
    B, C, H, W = image.shape
    image_dev = image.to(device)

    if H <= patch_size and W <= patch_size:
        pad_h, pad_w = max(0, patch_size - H), max(0, patch_size - W)
        if pad_h > 0 or pad_w > 0:
            image_dev = F.pad(image_dev, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            output = nafnet(image_dev)
        return output[:, :, :H, :W] if (pad_h > 0 or pad_w > 0) else output

    denoised_sum = torch.zeros(1, 1, H, W, device=device)
    weight_sum = torch.zeros(1, 1, H, W, device=device)

    h_pos = list(range(0, H - patch_size + 1, stride))
    w_pos = list(range(0, W - patch_size + 1, stride))
    if not h_pos or h_pos[-1] + patch_size < H:
        h_pos.append(max(0, H - patch_size))
    if not w_pos or w_pos[-1] + patch_size < W:
        w_pos.append(max(0, W - patch_size))

    with torch.no_grad():
        for h in h_pos:
            for w in w_pos:
                patch = image_dev[:, :, h:h+patch_size, w:w+patch_size]
                out = nafnet(patch)
                denoised_sum[:, :, h:h+patch_size, w:w+patch_size] += out
                weight_sum[:, :, h:h+patch_size, w:w+patch_size] += 1

    return denoised_sum / weight_sum.clamp(min=1)


# =============================================================================
# Dataset
# =============================================================================

class PKU37Dataset(Dataset):
    """PKU37 OCT dataset from JSONL format."""

    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        patch_size: int = 256,
        num_realizations: int = 1,
    ):
        self.patch_size = patch_size
        self.num_realizations = num_realizations
        self.samples = []

        # Load JSONL
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.samples.append(entry)
                if max_samples and len(self.samples) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # Load clean image
        clean_path = entry.get('clean_path') or entry.get('gt_path')
        clean_img = Image.open(clean_path)
        clean = np.array(clean_img, dtype=np.float32)
        if clean.max() > 1:
            clean = clean / 255.0

        # Load noisy image(s)
        noisy_paths = entry.get('noisy_paths', [entry.get('noisy_path')])
        if self.num_realizations > 1 and len(noisy_paths) >= self.num_realizations:
            selected = noisy_paths[:self.num_realizations]
        else:
            selected = [noisy_paths[0]]

        noisy_list = []
        for np_path in selected:
            noisy_img = Image.open(np_path)
            noisy = np.array(noisy_img, dtype=np.float32)
            if noisy.max() > 1:
                noisy = noisy / 255.0
            noisy_list.append(noisy)

        noisy = noisy_list[0]  # Primary noisy image

        # Resize if needed
        H, W = clean.shape
        if H != self.patch_size or W != self.patch_size:
            clean_pil = Image.fromarray((clean * 255).astype(np.uint8))
            noisy_pil = Image.fromarray((noisy * 255).astype(np.uint8))
            clean = np.array(clean_pil.resize((self.patch_size, self.patch_size), Image.BILINEAR), dtype=np.float32) / 255.0
            noisy = np.array(noisy_pil.resize((self.patch_size, self.patch_size), Image.BILINEAR), dtype=np.float32) / 255.0

        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0),  # [1, H, W]
            'clean': torch.from_numpy(clean).unsqueeze(0),
            'name': os.path.basename(clean_path),
        }


class PKU37DirectDataset(Dataset):
    """Load directly from PKU37 directory structure with random patches.

    FAST: Pre-loads all images into memory for speed.
    """

    def __init__(
        self,
        pku37_root: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        patch_size: int = 256,
        patches_per_image: int = 4,
    ):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.images = []  # Pre-load for speed

        clean_dir = os.path.join(pku37_root, "clean")
        noisy_dir = os.path.join(pku37_root, "noisy")

        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])

        n_total = len(clean_files)
        n_train = int(n_total * 0.8)

        if split == 'train':
            clean_files = clean_files[:n_train]
        else:
            clean_files = clean_files[n_train:]

        count = 0
        for fname in clean_files:
            clean_path = os.path.join(clean_dir, fname)
            base_name = os.path.splitext(fname)[0]

            noisy_files = sorted([
                f for f in os.listdir(noisy_dir)
                if f.startswith(base_name) and f.endswith('.tif')
            ])

            for noisy_fname in noisy_files:
                # Pre-load and normalize
                clean = np.array(Image.open(clean_path), dtype=np.float32) / 255.0
                noisy = np.array(Image.open(os.path.join(noisy_dir, noisy_fname)), dtype=np.float32) / 255.0
                self.images.append((clean, noisy))
                count += 1

                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break

        logger.info(f"Pre-loaded {len(self.images)} images for {split} ({patches_per_image} patches each)")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        clean, noisy = self.images[img_idx]
        H, W = clean.shape

        # Random crop
        h = np.random.randint(0, max(1, H - self.patch_size + 1))
        w = np.random.randint(0, max(1, W - self.patch_size + 1))

        clean_patch = clean[h:h+self.patch_size, w:w+self.patch_size]
        noisy_patch = noisy[h:h+self.patch_size, w:w+self.patch_size]

        # Pad if needed (rare)
        ph, pw = clean_patch.shape
        if ph < self.patch_size or pw < self.patch_size:
            clean_patch = np.pad(clean_patch, ((0, self.patch_size-ph), (0, self.patch_size-pw)), mode='reflect')
            noisy_patch = np.pad(noisy_patch, ((0, self.patch_size-ph), (0, self.patch_size-pw)), mode='reflect')

        # Random horizontal flip
        if np.random.random() > 0.5:
            clean_patch = clean_patch[:, ::-1].copy()
            noisy_patch = noisy_patch[:, ::-1].copy()

        return {
            'noisy': torch.from_numpy(noisy_patch).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean_patch).unsqueeze(0).float(),
        }


class PKU37FullImageDataset(Dataset):
    """Full images for validation - pre-loaded for speed."""

    def __init__(self, pku37_root: str, split: str = 'val', max_samples: Optional[int] = None):
        self.images = []  # Pre-load for speed

        clean_dir = os.path.join(pku37_root, "clean")
        noisy_dir = os.path.join(pku37_root, "noisy")

        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])

        n_total = len(clean_files)
        n_train = int(n_total * 0.8)

        if split == 'train':
            clean_files = clean_files[:n_train]
        else:
            clean_files = clean_files[n_train:]

        for fname in clean_files:
            clean_path = os.path.join(clean_dir, fname)
            base_name = os.path.splitext(fname)[0]

            noisy_files = sorted([
                f for f in os.listdir(noisy_dir)
                if f.startswith(base_name) and f.endswith('.tif')
            ])

            for noisy_fname in noisy_files:
                # Pre-load and convert to tensor
                clean = torch.from_numpy(
                    np.array(Image.open(clean_path), dtype=np.float32) / 255.0
                ).unsqueeze(0)
                noisy = torch.from_numpy(
                    np.array(Image.open(os.path.join(noisy_dir, noisy_fname)), dtype=np.float32) / 255.0
                ).unsqueeze(0)
                self.images.append({'clean': clean, 'noisy': noisy, 'name': noisy_fname})

                if max_samples and len(self.images) >= max_samples:
                    break
            if max_samples and len(self.images) >= max_samples:
                break

        logger.info(f"Pre-loaded {len(self.images)} full images for {split} validation")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 50.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Simple global SSIM
    mu_p = pred.mean()
    mu_t = target.mean()
    var_p = pred.var()
    var_t = target.var()
    cov = ((pred - mu_p) * (target - mu_t)).mean()

    ssim = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))

    return ssim.item()


# =============================================================================
# Training
# =============================================================================

class ConstrainedTrainer:
    """Trainer with augmented Lagrangian for constraint satisfaction."""

    CONSTRAINT_NAMES = ['ordering', 'thickness', 'range', 'smoothness', 'intensity']

    def __init__(
        self,
        model: ConstrainedNeuroSymbolicDenoiser,
        config: DenoiserConfig,
        device: torch.device,
        lr: float = 1e-4,
        nafnet_ckpt: Optional[str] = None,
        patch_size: int = 128,
        val_stride: int = 96,  # Large stride for fast validation
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.patch_size = patch_size
        self.val_stride = val_stride

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # Clinical losses
        self.clinical_losses = ClinicalLosses().to(device)

        # Lagrange multipliers
        self.lambdas = {name: torch.tensor(1.0, device=device)
                       for name in self.CONSTRAINT_NAMES}

        # Penalty parameter
        self.rho = config.rho_init

        # Best metrics tracking
        self.best_psnr = 0
        self.best_epoch = 0

        # Load NAFNet baseline for comparison
        self.nafnet = None
        if HAS_NAFNET and nafnet_ckpt and os.path.exists(nafnet_ckpt):
            logger.info(f"Loading NAFNet baseline from {nafnet_ckpt}")
            self.nafnet = NAFNet(img_channel=1, width=64, middle_blk_num=2,
                                enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2])
            ckpt = torch.load(nafnet_ckpt, map_location=device)
            if 'state_dict' in ckpt:
                self.nafnet.load_state_dict(ckpt['state_dict'])
            else:
                self.nafnet.load_state_dict(ckpt)
            self.nafnet.to(device)
            self.nafnet.eval()
            logger.info("NAFNet baseline loaded for comparison")

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = defaultdict(list)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(noisy)

            # Clinical loss
            clinical_loss, clinical_details = self.clinical_losses.compute_all(outputs, clean)

            # Disentanglement loss
            disentangle_loss, _ = self.model.encoder.compute_disentanglement_loss(
                outputs['representations']
            )

            # Constraint violations
            violations = self.model.compute_constraint_violations(outputs)

            # Augmented Lagrangian penalty
            constraint_penalty = torch.tensor(0.0, device=self.device)
            constraint_scale = 0.1  # Balance constraints vs denoising

            for name, viol in violations.items():
                v = viol.mean()
                penalty = self.lambdas[name] * v + (self.rho / 2) * F.relu(v) ** 2
                constraint_penalty = constraint_penalty + constraint_scale * penalty
                epoch_metrics[f'{name}_viol'].append(v.item())

            # Total loss
            total_loss = clinical_loss + \
                        self.config.disentangle_weight * disentangle_loss + \
                        constraint_penalty

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            epoch_metrics['loss'].append(total_loss.item())
            epoch_metrics['clinical_loss'].append(clinical_loss.item())

            # PSNR for progress bar
            with torch.no_grad():
                psnr = compute_psnr(outputs['denoised'], clean)
                epoch_metrics['psnr'].append(psnr)

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'psnr': f"{psnr:.2f}",
            })

            # Free memory after each batch
            del outputs, noisy, clean, clinical_loss, disentangle_loss, total_loss, violations

        # Update Lagrangian parameters at end of epoch
        avg_violations = {name: np.mean(epoch_metrics[f'{name}_viol'])
                         for name in self.CONSTRAINT_NAMES}
        self._update_lagrangian(avg_violations)

        # Step scheduler
        self.scheduler.step()

        # Average metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}

    def _update_lagrangian(self, violations: Dict[str, float]):
        """Update Lagrange multipliers and penalty."""
        max_viol = max(violations.values())

        for name, viol in violations.items():
            # Update in-place to avoid creating new tensors
            self.lambdas[name] = torch.clamp(self.lambdas[name] + self.rho * viol, min=0.0)

        if max_viol > 0.01:
            self.rho = min(self.rho * self.config.rho_mult, self.config.rho_max)

    def _masked_ssim(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
        """Compute SSIM for masked region."""
        ms = mask.sum().clamp(min=1.0)
        mu_p = (pred * mask).sum() / ms
        mu_t = (target * mask).sum() / ms
        var_p = ((pred - mu_p)**2 * mask).sum() / ms
        var_t = ((target - mu_t)**2 * mask).sum() / ms
        cov = ((pred - mu_p) * (target - mu_t) * mask).sum() / ms
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu_p*mu_t + C1) * (2*cov + C2)) / ((mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2))
        return ssim.item()

    @torch.no_grad()
    def validate(self, val_dataset: Dataset) -> Dict[str, float]:
        """Validation with sliding window on full images + NAFNet comparison."""
        self.model.eval()

        metrics = defaultdict(list)

        for i in tqdm(range(len(val_dataset)), desc="Validating", leave=False):
            sample = val_dataset[i]
            noisy = sample['noisy'].unsqueeze(0)  # [1, 1, H, W]
            clean = sample['clean'].unsqueeze(0).to(self.device)

            # Our model with sliding window
            outputs = sliding_window_inference(
                self.model, noisy, self.patch_size, self.val_stride, self.device
            )
            denoised = outputs['denoised']

            # NAFNet baseline with sliding window (whole-image denoising)
            nafnet_out = None
            if self.nafnet is not None:
                nafnet_out = nafnet_sliding_window(
                    self.nafnet, noisy, self.patch_size, self.val_stride, self.device
                )

            # Input metrics
            noisy_dev = noisy.to(self.device)
            metrics['input_psnr'].append(compute_psnr(noisy_dev, clean))
            metrics['input_ssim'].append(compute_ssim(noisy_dev, clean))

            # Our model metrics
            metrics['psnr'].append(compute_psnr(denoised, clean))
            metrics['ssim'].append(compute_ssim(denoised, clean))

            # NAFNet baseline metrics
            if nafnet_out is not None:
                metrics['nafnet_psnr'].append(compute_psnr(nafnet_out, clean))
                metrics['nafnet_ssim'].append(compute_ssim(nafnet_out, clean))

            # Per-layer metrics (using our soft masks)
            soft_masks = outputs['soft_masks']
            for j, name in enumerate(self.model.LAYER_NAMES):
                mask = soft_masks[:, j:j+1]
                mask_sum = mask.sum().clamp(min=1.0)

                # Input per layer - PSNR & SSIM
                inp_mse = ((noisy_dev - clean) ** 2 * mask).sum() / mask_sum
                metrics[f'{name}_input_psnr'].append(10 * np.log10(1.0 / max(inp_mse.item(), 1e-10)))
                metrics[f'{name}_input_ssim'].append(self._masked_ssim(noisy_dev, clean, mask))

                # Ours per layer - PSNR & SSIM
                our_mse = ((denoised - clean) ** 2 * mask).sum() / mask_sum
                metrics[f'{name}_psnr'].append(10 * np.log10(1.0 / max(our_mse.item(), 1e-10)))
                metrics[f'{name}_ssim'].append(self._masked_ssim(denoised, clean, mask))

                # NAFNet per layer - PSNR & SSIM (same masks for fair comparison)
                if nafnet_out is not None:
                    naf_mse = ((nafnet_out - clean) ** 2 * mask).sum() / mask_sum
                    metrics[f'{name}_nafnet_psnr'].append(10 * np.log10(1.0 / max(naf_mse.item(), 1e-10)))
                    metrics[f'{name}_nafnet_ssim'].append(self._masked_ssim(nafnet_out, clean, mask))

            # Constraints
            violations = self.model.compute_constraint_violations(outputs)
            for name, v in violations.items():
                metrics[f'{name}_viol'].append(v.mean().item())

            # Clean up to prevent memory buildup
            del outputs, denoised, clean, noisy, noisy_dev, soft_masks
            if nafnet_out is not None:
                del nafnet_out

        # Average all metrics
        results = {k: np.mean(v) for k, v in metrics.items()}

        # Compute gains
        results['gain_vs_input'] = results['psnr'] - results['input_psnr']
        if 'nafnet_psnr' in results:
            results['gain_vs_nafnet'] = results['psnr'] - results['nafnet_psnr']

        # Per-layer gains
        for name in self.model.LAYER_NAMES:
            results[f'{name}_gain_vs_input'] = results[f'{name}_psnr'] - results[f'{name}_input_psnr']
            if f'{name}_nafnet_psnr' in results:
                results[f'{name}_gain_vs_nafnet'] = results[f'{name}_psnr'] - results[f'{name}_nafnet_psnr']

        results['avg_layer_psnr'] = np.mean([results[f'{name}_psnr'] for name in self.model.LAYER_NAMES])

        results['n_constraints_satisfied'] = sum(
            1 for name in self.CONSTRAINT_NAMES
            if results.get(f'{name}_viol', 1.0) < 0.01
        )

        return results

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'lambdas': {k: v.item() for k, v in self.lambdas.items()},
            'rho': self.rho,
            'metrics': metrics,
            'config': self.config,
        }, path)


# =============================================================================
# Main
# =============================================================================

def print_validation_results(results: Dict[str, float], layer_names: List[str], epoch: int = -1):
    """Print comprehensive validation results for all contributions."""
    header = f"EPOCH {epoch+1}" if epoch >= 0 else "INITIAL"
    print(f"\n{'='*80}")
    print(f"{header} - COMPREHENSIVE METRICS")
    print(f"{'='*80}")

    # ==========================================================================
    # 1. GLOBAL DENOISING QUALITY
    # ==========================================================================
    print(f"\n[1] GLOBAL DENOISING QUALITY")
    print(f"{'Method':<12} {'PSNR (dB)':>10} {'SSIM':>10} {'Gain vs NAF':>12}")
    print(f"{'-'*46}")

    inp_psnr = results.get('input_psnr', 0)
    naf_psnr = results.get('nafnet_psnr', inp_psnr)
    our_psnr = results.get('psnr', 0)
    our_ssim = results.get('ssim', 0)
    gain = results.get('gain_vs_nafnet', our_psnr - naf_psnr)

    print(f"{'Input':<12} {inp_psnr:>10.2f} {results.get('input_ssim', 0):>10.4f}")
    print(f"{'NAFNet':<12} {naf_psnr:>10.2f} {results.get('nafnet_ssim', 0):>10.4f}")
    gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
    print(f"{'Ours':<12} {our_psnr:>10.2f} {our_ssim:>10.4f} {gain_str:>12}")

    # ==========================================================================
    # 2. LAYER-SPECIFIC PSNR (Contribution: Layer-aware processing)
    # ==========================================================================
    print(f"\n[2] LAYER-SPECIFIC PSNR (dB)")
    print(f"{'Layer':<12} {'Input':>8} {'NAFNet':>8} {'Ours':>8} {'Gain':>8}")
    print(f"{'-'*48}")

    for name in layer_names:
        inp = results.get(f'{name}_input_psnr', 0)
        naf = results.get(f'{name}_nafnet_psnr', inp)
        ours = results.get(f'{name}_psnr', 0)
        gain = results.get(f'{name}_gain_vs_nafnet', ours - naf)
        gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
        print(f"{name:<12} {inp:>8.2f} {naf:>8.2f} {ours:>8.2f} {gain_str:>8}")

    # ==========================================================================
    # 3. LAYER-SPECIFIC SSIM (Contribution: Structural preservation per layer)
    # ==========================================================================
    print(f"\n[3] LAYER-SPECIFIC SSIM")
    print(f"{'Layer':<12} {'Input':>8} {'NAFNet':>8} {'Ours':>8} {'Gain':>8}")
    print(f"{'-'*48}")

    for name in layer_names:
        inp = results.get(f'{name}_input_ssim', 0)
        naf = results.get(f'{name}_nafnet_ssim', inp)
        ours = results.get(f'{name}_ssim', 0)
        gain = ours - naf
        gain_str = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"
        print(f"{name:<12} {inp:>8.4f} {naf:>8.4f} {ours:>8.4f} {gain_str:>8}")

    # ==========================================================================
    # 4. ANATOMICAL CONSTRAINTS (Contribution: Neuro-symbolic constraints)
    # ==========================================================================
    print(f"\n[4] ANATOMICAL CONSTRAINTS (Augmented Lagrangian)")
    constraint_names = ['ordering', 'thickness', 'range', 'smoothness', 'intensity']
    n_sat = results.get('n_constraints_satisfied', 0)
    print(f"Satisfied: {n_sat}/5")
    print(f"{'Constraint':<12} {'Violation':>10} {'Status':>10}")
    print(f"{'-'*34}")
    for name in constraint_names:
        viol = results.get(f'{name}_viol', 0)
        status = "OK" if viol < 0.01 else "VIOLATED"
        print(f"{name:<12} {viol:>10.4f} {status:>10}")

    # ==========================================================================
    # 5. BOUNDARY PREDICTION (Contribution: Learned layer segmentation)
    # ==========================================================================
    print(f"\n[5] BOUNDARY PREDICTION")
    for i in range(4):
        b_mean = results.get(f'boundary_{i}_mean', 0)
        b_std = results.get(f'boundary_{i}_std', 0)
        if b_mean > 0:
            print(f"  Boundary {i}: mean={b_mean:.3f}, std={b_std:.4f}")
        else:
            print(f"  Boundary positions: (computed during inference)")
            break

    # ==========================================================================
    # 6. CLINICAL LOSSES (Contribution: Layer-specific loss functions)
    # ==========================================================================
    clinical_names = ['rnfl', 'inl', 'onl', 'rpe']
    clinical_types = ['texture', 'edge', 'edge', 'texture']
    has_clinical = any(f'{n}_clinical' in results for n in clinical_names)

    if has_clinical:
        print(f"\n[6] CLINICAL LOSSES (Layer-specific)")
        print(f"{'Layer':<12} {'Type':>10} {'Loss':>10}")
        print(f"{'-'*34}")
        for name, ctype in zip(clinical_names, clinical_types):
            loss = results.get(f'{name}_clinical', 0)
            print(f"{name.upper():<12} {ctype:>10} {loss:>10.4f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'='*80}")
    status = "BETTER" if gain > 0 else ("COMPETITIVE" if gain > -1 else "TRAINING...")
    print(f"STATUS: {status} | PSNR Gain vs NAFNet: {gain_str} dB | Constraints: {n_sat}/5")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Constrained Neuro-Symbolic OCT Denoiser')

    # Data
    parser.add_argument('--train_jsonl', type=str, default=None,
                       help='Training data JSONL file')
    parser.add_argument('--val_jsonl', type=str, default=None,
                       help='Validation data JSONL file')
    parser.add_argument('--pku37_root', type=str,
                       default='/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising',
                       help='PKU37 dataset root (used if no JSONL provided)')
    parser.add_argument('--max_train', type=int, default=100,
                       help='Max training samples')
    parser.add_argument('--max_val', type=int, default=20,
                       help='Max validation samples')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Image patch size')
    parser.add_argument('--val_stride', type=int, default=96,
                       help='Stride for validation sliding window (larger=faster)')
    parser.add_argument('--patches_per_image', type=int, default=4,
                       help='Number of random patches per image for training')
    parser.add_argument('--num_realizations', type=int, default=1)

    # Model
    parser.add_argument('--encoder_channels', type=str, default='32,64,128',
                       help='Encoder channel sizes')
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--head_width', type=int, default=24)

    # Baseline
    parser.add_argument('--nafnet_ckpt', type=str, default='outputs/nafnet_pku37/nafnet_best.pth',
                       help='NAFNet checkpoint for baseline comparison')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_every', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)

    # Augmented Lagrangian
    parser.add_argument('--rho_init', type=float, default=0.1)
    parser.add_argument('--rho_max', type=float, default=10.0)
    parser.add_argument('--rho_mult', type=float, default=1.2)
    parser.add_argument('--disentangle_weight', type=float, default=0.05)

    # Other
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs/constrained_denoiser')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CONSTRAINED NEURO-SYMBOLIC OCT DENOISER")
    logger.info("=" * 70)

    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("\nLoading data...")
    logger.info(f"Using PKU37 from {args.pku37_root}")

    # Training: random patches
    train_dataset = PKU37DirectDataset(
        args.pku37_root,
        split='train',
        max_samples=args.max_train,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
    )

    # Validation: full images (sliding window)
    val_dataset = PKU37FullImageDataset(
        args.pku37_root,
        split='val',
        max_samples=args.max_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )

    logger.info(f"Train: {len(train_dataset)} patches, Val: {len(val_dataset)} full images")

    # Create model
    logger.info("\nCreating model...")
    encoder_channels = [int(x) for x in args.encoder_channels.split(',')]
    config = DenoiserConfig(
        encoder_channels=encoder_channels,
        latent_dim=args.latent_dim,
        head_width=args.head_width,
        rho_init=args.rho_init,
        rho_max=args.rho_max,
        rho_mult=args.rho_mult,
        disentangle_weight=args.disentangle_weight,
        lr=args.lr,
    )

    model = ConstrainedNeuroSymbolicDenoiser(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Create trainer with NAFNet baseline
    trainer = ConstrainedTrainer(
        model, config, device,
        lr=args.lr,
        nafnet_ckpt=args.nafnet_ckpt,
        patch_size=args.patch_size,
        val_stride=args.val_stride,
    )

    # Initial validation
    logger.info("\nInitial validation (sliding window on full images)...")
    init_metrics = trainer.validate(val_dataset)
    print_validation_results(init_metrics, model.LAYER_NAMES, epoch=-1)

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)

    best_psnr = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = trainer.validate(val_dataset)

            # Print detailed results
            print_validation_results(val_metrics, model.LAYER_NAMES, epoch)

            # Quick summary
            gain = val_metrics.get('gain_vs_nafnet', 0)
            gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
            logger.info(
                f"Epoch {epoch+1:3d} | Loss: {train_metrics['loss']:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f} | Gain vs NAFNet: {gain_str} dB | "
                f"Constraints: {val_metrics['n_constraints_satisfied']}/5"
            )

            # Save best
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                best_epoch = epoch + 1
                trainer.save_checkpoint(
                    os.path.join(args.output_dir, 'best_model.pt'),
                    epoch, val_metrics
                )
                logger.info(f"[NEW BEST] Saved best model (PSNR: {best_psnr:.2f} dB)")

    # Final summary
    print("\n" + "=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)

    print(f"\nBest Model: Epoch {best_epoch} | PSNR: {best_psnr:.2f} dB")

    # Final validation
    final_metrics = trainer.validate(val_dataset)
    print_validation_results(final_metrics, model.LAYER_NAMES, epoch=args.epochs-1)

    # SOTA comparison
    nafnet_psnr = final_metrics.get('nafnet_psnr', 0)
    our_psnr = final_metrics['psnr']
    gain = final_metrics.get('gain_vs_nafnet', 0)

    print("\n" + "=" * 75)
    print("COMPARISON vs NAFNet BASELINE")
    print("=" * 75)
    print(f"\n  NAFNet PSNR: {nafnet_psnr:.2f} dB")
    print(f"  Ours PSNR:   {our_psnr:.2f} dB")
    gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
    print(f"  Gain:        {gain_str} dB")

    if gain > 0:
        print("\n  [SUCCESS] Our method outperforms NAFNet!")
        print("  Layer-specific processing provides improvements.")
    elif gain > -0.5:
        print("\n  [COMPETITIVE] Within 0.5 dB of NAFNet")
        print("  May need more training or architectural improvements.")
    else:
        print("\n  [NEEDS IMPROVEMENT] Below NAFNet baseline")
        print("  Consider: larger model, more training, or smart enhancements.")

    # Save final checkpoint
    trainer.save_checkpoint(
        os.path.join(args.output_dir, 'final_model.pt'),
        args.epochs - 1, final_metrics
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
