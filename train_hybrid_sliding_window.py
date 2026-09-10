#!/usr/bin/env python3
"""
Train Constrained NAFNet Hybrid with Sliding Window Validation.

Key features:
1. NAFNet backbone (frozen) + layer-specific refinement heads
2. Sliding window inference for validation on full images
3. Per-layer PSNR/SSIM comparison vs NAFNet baseline
4. Anatomical constraint satisfaction
"""

import os
import sys
import argparse
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

from constrained_nafnet_hybrid import (
    ConstrainedNAFNetHybrid,
    ClinicalLosses,
)


# =============================================================================
# Sliding Window Inference
# =============================================================================

def sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: int = 128,
    stride: int = 64,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    """
    Perform sliding window inference on full image.

    Args:
        model: The denoising model
        image: Input tensor [1, 1, H, W]
        patch_size: Size of patches
        stride: Stride between patches (overlap = patch_size - stride)
        device: Device to use

    Returns:
        Model outputs averaged over overlapping regions
    """
    model.eval()

    B, C, H, W = image.shape
    assert B == 1, "Batch size must be 1 for sliding window"

    # Handle small images
    if H <= patch_size and W <= patch_size:
        # Pad if needed
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            outputs = model(image.to(device))

        # Unpad
        if pad_h > 0 or pad_w > 0:
            outputs['denoised'] = outputs['denoised'][:, :, :H, :W]
            outputs['soft_masks'] = outputs['soft_masks'][:, :, :H, :W]

        return outputs

    # Create output tensors
    denoised_sum = torch.zeros(1, 1, H, W, device=device)
    weight_sum = torch.zeros(1, 1, H, W, device=device)

    # For soft masks (4 layers)
    masks_sum = torch.zeros(1, 4, H, W, device=device)

    # Boundaries will be averaged
    boundaries_list = []

    # Calculate number of patches
    n_h = max(1, (H - patch_size) // stride + 1)
    n_w = max(1, (W - patch_size) // stride + 1)

    # Ensure we cover the entire image
    h_positions = [i * stride for i in range(n_h)]
    w_positions = [i * stride for i in range(n_w)]

    # Add edge positions
    if h_positions[-1] + patch_size < H:
        h_positions.append(H - patch_size)
    if w_positions[-1] + patch_size < W:
        w_positions.append(W - patch_size)

    # Create Gaussian window for smooth blending
    window = _create_gaussian_window(patch_size, device)

    with torch.no_grad():
        for h_start in h_positions:
            for w_start in w_positions:
                h_end = h_start + patch_size
                w_end = w_start + patch_size

                patch = image[:, :, h_start:h_end, w_start:w_end].to(device)

                outputs = model(patch)

                # Accumulate with Gaussian weighting
                denoised_sum[:, :, h_start:h_end, w_start:w_end] += outputs['denoised'] * window
                masks_sum[:, :, h_start:h_end, w_start:w_end] += outputs['soft_masks'] * window
                weight_sum[:, :, h_start:h_end, w_start:w_end] += window

                boundaries_list.append(outputs['boundaries'])

    # Average
    denoised = denoised_sum / weight_sum.clamp(min=1e-8)
    soft_masks = masks_sum / weight_sum.clamp(min=1e-8)
    soft_masks = soft_masks / soft_masks.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Average boundaries
    boundaries = torch.stack(boundaries_list, dim=0).mean(dim=0)

    return {
        'denoised': denoised,
        'soft_masks': soft_masks,
        'boundaries': boundaries,
        'nafnet_out': denoised,  # For compatibility
        'layer_outputs': denoised.expand(-1, 4, -1, -1),
        'layer_refinements': torch.zeros(1, 4, H, W, device=device),
    }


def _create_gaussian_window(size: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian window for smooth blending."""
    sigma = size / 6.0
    x = torch.arange(size, device=device, dtype=torch.float32)
    gauss_1d = torch.exp(-((x - size/2) ** 2) / (2 * sigma ** 2))
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    return gauss_2d.unsqueeze(0).unsqueeze(0)


def nafnet_sliding_window(
    nafnet: nn.Module,
    image: torch.Tensor,
    patch_size: int = 128,
    stride: int = 64,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Sliding window inference for NAFNet baseline."""
    nafnet.eval()

    B, C, H, W = image.shape

    if H <= patch_size and W <= patch_size:
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            output = nafnet(image.to(device))
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :H, :W]
        return output

    denoised_sum = torch.zeros(1, 1, H, W, device=device)
    weight_sum = torch.zeros(1, 1, H, W, device=device)

    n_h = max(1, (H - patch_size) // stride + 1)
    n_w = max(1, (W - patch_size) // stride + 1)

    h_positions = [i * stride for i in range(n_h)]
    w_positions = [i * stride for i in range(n_w)]

    if h_positions[-1] + patch_size < H:
        h_positions.append(H - patch_size)
    if w_positions[-1] + patch_size < W:
        w_positions.append(W - patch_size)

    window = _create_gaussian_window(patch_size, device)

    with torch.no_grad():
        for h_start in h_positions:
            for w_start in w_positions:
                h_end = h_start + patch_size
                w_end = w_start + patch_size

                patch = image[:, :, h_start:h_end, w_start:w_end].to(device)
                output = nafnet(patch)

                denoised_sum[:, :, h_start:h_end, w_start:w_end] += output * window
                weight_sum[:, :, h_start:h_end, w_start:w_end] += window

    return denoised_sum / weight_sum.clamp(min=1e-8)


# =============================================================================
# Dataset for Full Images (Validation)
# =============================================================================

class PKU37FullImageDataset(Dataset):
    """Load full images without resizing for validation."""

    def __init__(self, pku37_root: str, split: str = 'val', max_samples: Optional[int] = None):
        self.pairs = []

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
                self.pairs.append({
                    'clean_path': clean_path,
                    'noisy_path': os.path.join(noisy_dir, noisy_fname),
                })

                if max_samples and len(self.pairs) >= max_samples:
                    break

            if max_samples and len(self.pairs) >= max_samples:
                break

        print(f"Loaded {len(self.pairs)} full-image pairs for {split}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        clean = np.array(Image.open(pair['clean_path']), dtype=np.float32)
        noisy = np.array(Image.open(pair['noisy_path']), dtype=np.float32)

        if clean.max() > 1:
            clean = clean / 255.0
        if noisy.max() > 1:
            noisy = noisy / 255.0

        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0),
            'clean': torch.from_numpy(clean).unsqueeze(0),
            'name': os.path.basename(pair['noisy_path']),
        }


class PKU37PatchDataset(Dataset):
    """Extract random patches for training."""

    def __init__(
        self,
        pku37_root: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        patch_size: int = 128,
        patches_per_image: int = 4,
    ):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.images = []

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
                # Load images
                clean = np.array(Image.open(clean_path), dtype=np.float32)
                noisy = np.array(Image.open(os.path.join(noisy_dir, noisy_fname)), dtype=np.float32)

                if clean.max() > 1:
                    clean = clean / 255.0
                if noisy.max() > 1:
                    noisy = noisy / 255.0

                self.images.append({
                    'clean': clean,
                    'noisy': noisy,
                })

                count += 1
                if max_samples and count >= max_samples:
                    break

            if max_samples and count >= max_samples:
                break

        print(f"Loaded {len(self.images)} images for {split} (will extract {patches_per_image} patches each)")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img_data = self.images[img_idx]

        clean = img_data['clean']
        noisy = img_data['noisy']

        H, W = clean.shape

        # Random crop
        h_start = np.random.randint(0, max(1, H - self.patch_size))
        w_start = np.random.randint(0, max(1, W - self.patch_size))

        clean_patch = clean[h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
        noisy_patch = noisy[h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            noisy_patch = np.pad(noisy_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Random augmentation
        if np.random.random() > 0.5:
            clean_patch = np.flip(clean_patch, axis=1).copy()
            noisy_patch = np.flip(noisy_patch, axis=1).copy()

        return {
            'noisy': torch.from_numpy(noisy_patch).unsqueeze(0),
            'clean': torch.from_numpy(clean_patch).unsqueeze(0),
        }


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 50.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM using sliding window."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Use 11x11 window
    kernel_size = 11
    sigma = 1.5

    # Create Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=pred.device)
    gauss = torch.exp(-((x - kernel_size//2) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Compute local statistics
    pad = kernel_size // 2
    mu_p = F.conv2d(F.pad(pred, (pad, pad, pad, pad), mode='reflect'), kernel)
    mu_t = F.conv2d(F.pad(target, (pad, pad, pad, pad), mode='reflect'), kernel)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(F.pad(pred ** 2, (pad, pad, pad, pad), mode='reflect'), kernel) - mu_p_sq
    sigma_t_sq = F.conv2d(F.pad(target ** 2, (pad, pad, pad, pad), mode='reflect'), kernel) - mu_t_sq
    sigma_pt = F.conv2d(F.pad(pred * target, (pad, pad, pad, pad), mode='reflect'), kernel) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))

    return ssim_map.mean().item()


def compute_masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    """Compute PSNR and SSIM for masked region."""
    mask_sum = mask.sum().clamp(min=1.0)

    # PSNR
    mse = ((pred - target) ** 2 * mask).sum() / mask_sum
    psnr = 10 * np.log10(1.0 / max(mse.item(), 1e-10))

    # SSIM (simplified for masked region)
    mu_p = (pred * mask).sum() / mask_sum
    mu_t = (target * mask).sum() / mask_sum
    var_p = ((pred - mu_p) ** 2 * mask).sum() / mask_sum
    var_t = ((target - mu_t) ** 2 * mask).sum() / mask_sum
    cov = ((pred - mu_p) * (target - mu_t) * mask).sum() / mask_sum

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))

    return psnr, ssim.item()


# =============================================================================
# Trainer
# =============================================================================

class HybridTrainer:
    """Train hybrid model with sliding window validation."""

    LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL_IS', 'RPE_Choroid']
    CONSTRAINT_NAMES = ['ordering', 'thickness', 'smoothness']

    def __init__(
        self,
        model: ConstrainedNAFNetHybrid,
        nafnet_ckpt: str,
        device: torch.device,
        lr: float = 1e-3,
        patch_size: int = 128,
        stride: int = 64,
    ):
        self.model = model.to(device)
        self.device = device
        self.patch_size = patch_size
        self.stride = stride

        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)

        self.clinical_losses = ClinicalLosses().to(device)

        # Load NAFNet baseline for comparison
        self.nafnet = None
        if os.path.exists(nafnet_ckpt):
            try:
                from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
                self.nafnet = NAFNet(img_channel=1, width=64, middle_blk_num=2,
                                    enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2])
                ckpt = torch.load(nafnet_ckpt, map_location=device)
                if 'state_dict' in ckpt:
                    self.nafnet.load_state_dict(ckpt['state_dict'])
                else:
                    self.nafnet.load_state_dict(ckpt)
                self.nafnet.to(device)
                self.nafnet.eval()
                print(f"Loaded NAFNet baseline from {nafnet_ckpt}")
            except Exception as e:
                print(f"Could not load NAFNet: {e}")

        # Lagrangian parameters
        self.lambdas = {name: 1.0 for name in self.CONSTRAINT_NAMES}
        self.rho = 0.1

        self.best_psnr = 0

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        metrics = defaultdict(list)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(noisy)

            # Clinical loss
            clinical_loss, _ = self.clinical_losses.compute_all(outputs, clean)

            # Constraint penalty
            violations = self.model.compute_constraint_violations(outputs)
            constraint_penalty = 0
            for name, viol in violations.items():
                v = viol.mean()
                constraint_penalty += 0.1 * (self.lambdas[name] * v + (self.rho / 2) * v ** 2)
                metrics[f'{name}_viol'].append(v.item())

            loss = clinical_loss + constraint_penalty
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            metrics['loss'].append(loss.item())

            with torch.no_grad():
                psnr = compute_psnr(outputs['denoised'], clean)
                metrics['train_psnr'].append(psnr)

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'psnr': f"{psnr:.1f}"})

        # Update Lagrangian
        for name in self.CONSTRAINT_NAMES:
            avg_viol = np.mean(metrics[f'{name}_viol'])
            if avg_viol > 0.01:
                self.lambdas[name] = max(0, self.lambdas[name] + self.rho * avg_viol)
                self.rho = min(self.rho * 1.1, 1.0)

        return {k: np.mean(v) for k, v in metrics.items()}

    @torch.no_grad()
    def validate_sliding_window(self, dataset: PKU37FullImageDataset) -> Dict[str, float]:
        """Validate using sliding window on full images."""
        self.model.eval()

        metrics = defaultdict(list)

        for i in tqdm(range(len(dataset)), desc="Validating (sliding window)", leave=False):
            sample = dataset[i]
            noisy = sample['noisy'].unsqueeze(0)  # [1, 1, H, W]
            clean = sample['clean'].unsqueeze(0).to(self.device)

            # Our model with sliding window
            outputs = sliding_window_inference(
                self.model, noisy, self.patch_size, self.stride, self.device
            )
            denoised = outputs['denoised']

            # NAFNet baseline with sliding window
            nafnet_out = None
            if self.nafnet is not None:
                nafnet_out = nafnet_sliding_window(
                    self.nafnet, noisy, self.patch_size, self.stride, self.device
                )

            # Global metrics
            metrics['input_psnr'].append(compute_psnr(noisy.to(self.device), clean))
            metrics['input_ssim'].append(compute_ssim(noisy.to(self.device), clean))

            metrics['ours_psnr'].append(compute_psnr(denoised, clean))
            metrics['ours_ssim'].append(compute_ssim(denoised, clean))

            if nafnet_out is not None:
                metrics['nafnet_psnr'].append(compute_psnr(nafnet_out, clean))
                metrics['nafnet_ssim'].append(compute_ssim(nafnet_out, clean))

            # Per-layer metrics
            soft_masks = outputs['soft_masks']
            for j, name in enumerate(self.LAYER_NAMES):
                mask = soft_masks[:, j:j+1]

                inp_psnr, inp_ssim = compute_masked_metrics(noisy.to(self.device), clean, mask)
                ours_psnr, ours_ssim = compute_masked_metrics(denoised, clean, mask)

                metrics[f'{name}_input_psnr'].append(inp_psnr)
                metrics[f'{name}_input_ssim'].append(inp_ssim)
                metrics[f'{name}_ours_psnr'].append(ours_psnr)
                metrics[f'{name}_ours_ssim'].append(ours_ssim)

                if nafnet_out is not None:
                    naf_psnr, naf_ssim = compute_masked_metrics(nafnet_out, clean, mask)
                    metrics[f'{name}_nafnet_psnr'].append(naf_psnr)
                    metrics[f'{name}_nafnet_ssim'].append(naf_ssim)

            # Constraints
            violations = self.model.compute_constraint_violations(outputs)
            for name, v in violations.items():
                metrics[f'{name}_viol'].append(v.mean().item())

        return {k: np.mean(v) for k, v in metrics.items()}

    def print_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict, time_elapsed: float):
        """Print detailed results."""
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1} | Time: {time_elapsed:.1f}s")
        print(f"{'='*80}")

        # Loss
        print(f"\nLoss: {train_metrics.get('loss', 0):.4f}")

        # Global metrics
        print(f"\n--- GLOBAL METRICS (Full Image, Sliding Window) ---")
        print(f"{'Metric':<12} {'Input':>10} {'NAFNet':>10} {'Ours':>10} {'Gain vs NAF':>12}")
        print(f"{'-'*56}")

        inp_psnr = val_metrics.get('input_psnr', 0)
        naf_psnr = val_metrics.get('nafnet_psnr', inp_psnr)
        ours_psnr = val_metrics.get('ours_psnr', 0)
        gain_psnr = ours_psnr - naf_psnr

        inp_ssim = val_metrics.get('input_ssim', 0)
        naf_ssim = val_metrics.get('nafnet_ssim', inp_ssim)
        ours_ssim = val_metrics.get('ours_ssim', 0)
        gain_ssim = ours_ssim - naf_ssim

        gain_psnr_str = f"+{gain_psnr:.2f}" if gain_psnr >= 0 else f"{gain_psnr:.2f}"
        gain_ssim_str = f"+{gain_ssim:.4f}" if gain_ssim >= 0 else f"{gain_ssim:.4f}"

        print(f"{'PSNR (dB)':<12} {inp_psnr:>10.2f} {naf_psnr:>10.2f} {ours_psnr:>10.2f} {gain_psnr_str:>12}")
        print(f"{'SSIM':<12} {inp_ssim:>10.4f} {naf_ssim:>10.4f} {ours_ssim:>10.4f} {gain_ssim_str:>12}")

        # Per-layer PSNR
        print(f"\n--- PER-LAYER PSNR (dB) ---")
        print(f"{'Layer':<12} {'Input':>8} {'NAFNet':>8} {'Ours':>8} {'Gain':>8}")
        print(f"{'-'*48}")

        for name in self.LAYER_NAMES:
            inp = val_metrics.get(f'{name}_input_psnr', 0)
            naf = val_metrics.get(f'{name}_nafnet_psnr', inp)
            ours = val_metrics.get(f'{name}_ours_psnr', 0)
            gain = ours - naf
            gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
            print(f"{name:<12} {inp:>8.2f} {naf:>8.2f} {ours:>8.2f} {gain_str:>8}")

        # Per-layer SSIM
        print(f"\n--- PER-LAYER SSIM ---")
        print(f"{'Layer':<12} {'Input':>8} {'NAFNet':>8} {'Ours':>8} {'Gain':>8}")
        print(f"{'-'*48}")

        for name in self.LAYER_NAMES:
            inp = val_metrics.get(f'{name}_input_ssim', 0)
            naf = val_metrics.get(f'{name}_nafnet_ssim', inp)
            ours = val_metrics.get(f'{name}_ours_ssim', 0)
            gain = ours - naf
            gain_str = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"
            print(f"{name:<12} {inp:>8.4f} {naf:>8.4f} {ours:>8.4f} {gain_str:>8}")

        # Constraints
        print(f"\n--- CONSTRAINTS ---")
        n_sat = sum(1 for name in self.CONSTRAINT_NAMES if val_metrics.get(f'{name}_viol', 1) < 0.01)
        print(f"Satisfied: {n_sat}/{len(self.CONSTRAINT_NAMES)}")
        for name in self.CONSTRAINT_NAMES:
            viol = val_metrics.get(f'{name}_viol', 0)
            status = "OK" if viol < 0.01 else f"VIOL ({viol:.4f})"
            print(f"  {name}: {status}")

        print(f"{'='*80}\n")

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_psnr': self.best_psnr,
        }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Model with Sliding Window Validation')

    parser.add_argument('--pku37_root', type=str,
                       default='/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising')
    parser.add_argument('--nafnet_ckpt', type=str, default='outputs/nafnet_pku37/nafnet_best.pth')

    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--max_val', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--patches_per_image', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--val_every', type=int, default=2)

    parser.add_argument('--refinement_width', type=int, default=32)
    parser.add_argument('--freeze_nafnet', action='store_true', default=True)
    parser.add_argument('--unfreeze_nafnet', action='store_true')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs/hybrid_sliding')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.unfreeze_nafnet:
        args.freeze_nafnet = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("CONSTRAINED NAFNET HYBRID - Training with Sliding Window Validation")
    print("=" * 80)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    print("\nLoading data...")
    train_dataset = PKU37PatchDataset(
        args.pku37_root, split='train',
        max_samples=args.max_train,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
    )
    val_dataset = PKU37FullImageDataset(
        args.pku37_root, split='val',
        max_samples=args.max_val,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} full images")

    # Model
    print("\nCreating model...")
    model = ConstrainedNAFNetHybrid(
        nafnet_ckpt=args.nafnet_ckpt,
        freeze_nafnet=args.freeze_nafnet,
        refinement_width=args.refinement_width,
    )

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_total:,}")
    print(f"Trainable params: {n_trainable:,}")
    print(f"NAFNet frozen: {args.freeze_nafnet}")

    # Trainer
    trainer = HybridTrainer(
        model, args.nafnet_ckpt, device,
        lr=args.lr,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    # Initial validation
    print("\nInitial validation (sliding window)...")
    init_metrics = trainer.validate_sliding_window(val_dataset)
    trainer.print_results(-1, {}, init_metrics, 0)

    # Training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_psnr = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_metrics = trainer.train_epoch(train_loader, epoch)

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = trainer.validate_sliding_window(val_dataset)
            trainer.print_results(epoch, train_metrics, val_metrics, epoch_time)

            if val_metrics['ours_psnr'] > best_psnr:
                best_psnr = val_metrics['ours_psnr']
                best_epoch = epoch + 1
                trainer.save_checkpoint(os.path.join(args.output_dir, 'best_model.pt'), epoch, val_metrics)
                print(f"Saved best model (PSNR: {best_psnr:.2f} dB)")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest Model: Epoch {best_epoch} | PSNR: {best_psnr:.2f} dB")

    # Final validation
    final_metrics = trainer.validate_sliding_window(val_dataset)
    trainer.print_results(args.epochs - 1, train_metrics, final_metrics, 0)

    trainer.save_checkpoint(os.path.join(args.output_dir, 'final_model.pt'), args.epochs - 1, final_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
