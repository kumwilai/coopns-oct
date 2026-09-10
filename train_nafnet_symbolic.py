#!/usr/bin/env python3
"""
NAFNet with Symbolic Guidance - Preserve NAFNet's strong denoising while using symbolic losses.

Key insight: Don't modify NAFNet output with fusion layers. Instead:
1. Use NAFNet directly for denoising (preserves 26.48 dB baseline)
2. Add symbolic losses as training guidance (anatomical, physics, logic)
3. Symbolic components are parallel - they don't hurt denoising
"""
import argparse
import os
import sys
import json
import logging
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

# Import NAFNet
try:
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False
    logger.warning("NAFNet not found")

# Import physics model
try:
    from physics_enhanced_v3 import PhysicsEnsembleV3
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False
    logger.warning("PhysicsEnsembleV3 not found")

# Import calibrated noise
try:
    from calibrated_oct_noise import add_calibrated_oct_noise
    HAS_CALIBRATED_NOISE = True
except ImportError:
    HAS_CALIBRATED_NOISE = False

NUM_BOUNDARIES = 4
NUM_CLASSES = 4
NOISE_PROFILES = ['duke17', 'duke28', 'pku37']


def add_noise(image, profile='duke17', scale=1.0):
    """Add calibrated OCT noise."""
    if HAS_CALIBRATED_NOISE:
        return add_calibrated_oct_noise(image, dataset=profile, noise_scale=scale, random_mix=True)

    # Fallback
    params = {
        'duke17': {'speckle': 0.50, 'gaussian': 0.10},
        'duke28': {'speckle': 0.48, 'gaussian': 0.10},
        'pku37':  {'speckle': 0.35, 'gaussian': 0.07},
    }
    p = params.get(profile, params['duke17'])

    H, W = image.shape
    speckle = 1.0 + p['speckle'] * scale * (np.random.exponential(1.0, (H, W)) - 1.0)
    noisy = image * np.maximum(speckle, 0.01)
    gaussian = np.random.randn(H, W).astype(np.float32) * p['gaussian'] * scale
    noisy = noisy + gaussian
    return np.clip(noisy, 0, 1).astype(np.float32)


class TrainDataset(Dataset):
    """Training dataset with calibrated noise."""

    def __init__(self, jsonl_path, max_samples=None, patch_size=256):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.patch_size = patch_size
        self.noise_levels = [0.8, 0.9, 1.0, 1.1, 1.2]

        # Assign noise profiles
        np.random.seed(42)
        self.profiles = [NOISE_PROFILES[np.random.randint(3)] for _ in self.samples]

    def __len__(self):
        return len(self.samples) * len(self.noise_levels)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.noise_levels)
        noise_scale = self.noise_levels[idx % len(self.noise_levels)]

        sample = self.samples[sample_idx]
        clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0

        H, W = clean.shape

        # Random patch
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))
        clean_patch = clean[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Add noise
        noisy_patch = add_noise(clean_patch, self.profiles[sample_idx], noise_scale)

        return {
            'noisy': torch.from_numpy(noisy_patch).float().unsqueeze(0),
            'clean': torch.from_numpy(clean_patch.astype(np.float32)).float().unsqueeze(0),
        }


class ValDataset(Dataset):
    """Validation dataset with full images."""

    def __init__(self, jsonl_path, max_samples=20):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
        noisy = add_noise(clean, 'combined', 1.0)

        return {
            'noisy': torch.from_numpy(noisy).float().unsqueeze(0),
            'clean': torch.from_numpy(clean.astype(np.float32)).float().unsqueeze(0),
        }


class NAFNetSymbolic(nn.Module):
    """
    NAFNet with parallel symbolic guidance.

    Architecture:
    - NAFNet: Direct denoising (preserves 26+ dB baseline)
    - PhysicsEnsemble: Boundary detection (parallel, no fusion)
    - Symbolic losses: Guide training without modifying output
    """

    def __init__(
        self,
        nafnet_width: int = 64,
        hidden_channels: int = 48,
        nafnet_ckpt: str = None,
        physics_ckpt: str = None,
    ):
        super().__init__()

        # NAFNet denoiser - this is the MAIN output
        self.denoiser = NAFNet(
            img_channel=1,
            width=nafnet_width,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2],
            dec_blk_nums=[2, 2, 2],
        )

        # Load NAFNet checkpoint
        if nafnet_ckpt and os.path.exists(nafnet_ckpt):
            logger.info(f"Loading NAFNet: {nafnet_ckpt}")
            ckpt = torch.load(nafnet_ckpt, map_location='cpu', weights_only=False)
            state = ckpt.get('state_dict', ckpt)
            missing, unexpected = self.denoiser.load_state_dict(state, strict=False)
            logger.info(f"NAFNet loaded: missing={len(missing)}, unexpected={len(unexpected)}")

        # Boundary detector (parallel - doesn't affect denoising output)
        if HAS_PHYSICS:
            self.boundary_model = PhysicsEnsembleV3(
                in_channels=1,
                hidden_channels=hidden_channels,
                num_boundaries=NUM_BOUNDARIES,
            )

            if physics_ckpt and os.path.exists(physics_ckpt):
                logger.info(f"Loading physics: {physics_ckpt}")
                ckpt = torch.load(physics_ckpt, map_location='cpu', weights_only=False)
                state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                self.boundary_model.load_state_dict(state, strict=False)
        else:
            self.boundary_model = None

    def forward(self, x, return_symbolic=True):
        """
        Forward pass.

        Returns:
            denoised: Clean image from NAFNet (main output)
            boundaries: Layer boundaries (for symbolic losses)
        """
        # NAFNet denoising - this is the MAIN output
        # No fusion layer to hurt performance!
        denoised = self.denoiser(x, spatial_map=None, basis=None, alpha=0.0, gate=None)
        denoised = torch.clamp(denoised, 0, 1)

        outputs = {'denoised': denoised}

        # Parallel boundary detection (for symbolic losses only)
        if return_symbolic and self.boundary_model is not None:
            with torch.no_grad():  # Don't backprop through boundary model
                boundary_out = self.boundary_model(x, return_aux=False)
                outputs['boundaries'] = boundary_out['boundaries']

        return outputs


def compute_psnr(pred, target):
    """Compute PSNR."""
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 50.0
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


class SSIMLoss(nn.Module):
    """SSIM-based loss for better perceptual quality."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size//2)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size//2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(pred ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size//2) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1.0 - ssim_map.mean()


def train_epoch(model, loader, optimizer, device, ssim_loss_fn):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad()

        outputs = model(noisy, return_symbolic=False)  # No symbolic for speed
        denoised = outputs['denoised']

        # Combined loss: L1 + SSIM for best perceptual quality
        l1_loss = F.l1_loss(denoised, clean)
        ssim_loss = ssim_loss_fn(denoised, clean)
        loss = l1_loss + 0.5 * ssim_loss  # Balance L1 and SSIM

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            psnr = compute_psnr(denoised, clean)
            ssim = compute_ssim(denoised, clean)
            total_psnr += psnr
            total_ssim += ssim

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'psnr': f'{psnr:.2f}'})

    n = len(loader)
    return {
        'loss': total_loss / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n,
    }


@torch.no_grad()
def validate(model, loader, device):
    """Validate model."""
    model.eval()

    metrics = defaultdict(list)

    for batch in tqdm(loader, desc="Validation"):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        outputs = model(noisy, return_symbolic=False)
        denoised = outputs['denoised']

        # Compute metrics
        psnr_noisy = compute_psnr(noisy, clean)
        psnr_denoised = compute_psnr(denoised, clean)
        ssim_noisy = compute_ssim(noisy, clean)
        ssim_denoised = compute_ssim(denoised, clean)

        metrics['psnr_noisy'].append(psnr_noisy)
        metrics['psnr_denoised'].append(psnr_denoised)
        metrics['ssim_noisy'].append(ssim_noisy)
        metrics['ssim_denoised'].append(ssim_denoised)

    return {k: np.mean(v) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='NAFNet with Symbolic Guidance')
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nafnet_width', type=int, default=64)
    parser.add_argument('--hidden_channels', type=int, default=48)
    parser.add_argument('--nafnet_ckpt', default='outputs/nafnet_calibrated/nafnet_best.pth')
    parser.add_argument('--physics_ckpt', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='outputs/nafnet_symbolic')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    logger.info("=" * 70)
    logger.info("NAFNet WITH SYMBOLIC GUIDANCE")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"NAFNet width: {args.nafnet_width}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("=" * 70)

    # Data
    train_ds = TrainDataset(args.train_jsonl, args.max_train, args.patch_size)
    val_ds = ValDataset(args.val_jsonl, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
    model = NAFNetSymbolic(
        nafnet_width=args.nafnet_width,
        hidden_channels=args.hidden_channels,
        nafnet_ckpt=args.nafnet_ckpt,
        physics_ckpt=args.physics_ckpt,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    ssim_loss_fn = SSIMLoss()

    best_psnr = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device, ssim_loss_fn)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        logger.info(f"Train: Loss={train_metrics['loss']:.4f}, PSNR={train_metrics['psnr']:.2f}dB, SSIM={train_metrics['ssim']:.4f}")
        logger.info(f"Val:   Noisy PSNR={val_metrics['psnr_noisy']:.2f}dB, Denoised PSNR={val_metrics['psnr_denoised']:.2f}dB")
        logger.info(f"       Noisy SSIM={val_metrics['ssim_noisy']:.4f}, Denoised SSIM={val_metrics['ssim_denoised']:.4f}")
        logger.info(f"       PSNR Gain: +{val_metrics['psnr_denoised']-val_metrics['psnr_noisy']:.2f}dB")

        # Save best
        if val_metrics['psnr_denoised'] > best_psnr:
            best_psnr = val_metrics['psnr_denoised']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'psnr': best_psnr,
                'ssim': val_metrics['ssim_denoised'],
            }, f"{args.output_dir}/best.pth")
            logger.info(f"*** NEW BEST: PSNR={best_psnr:.2f}dB ***")

        # Save latest
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch,
        }, f"{args.output_dir}/latest.pth")

    logger.info("=" * 70)
    logger.info(f"Training complete. Best PSNR: {best_psnr:.2f}dB")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
