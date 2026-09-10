#!/usr/bin/env python3
"""
Train NAFNet with calibrated OCT noise matching real datasets (Duke17, Duke28, PKU37).
"""
import argparse
import os
import sys
import json
import gc
from collections import Counter
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))
from nsnd.models.nafnet import NAFNetFullFiLM

# Import calibrated noise
try:
    from calibrated_oct_noise import add_calibrated_oct_noise, NOISE_PARAMS
    HAS_CALIBRATED_NOISE = True
except ImportError:
    HAS_CALIBRATED_NOISE = False
    print("WARNING: calibrated_oct_noise not found, using fallback")


# Noise profiles
NOISE_PROFILES = ['duke17', 'duke28', 'pku37']


def add_calibrated_noise_by_profile(image, profile='duke17', noise_scale=1.0):
    """Add calibrated OCT noise matching specific dataset profile."""
    if HAS_CALIBRATED_NOISE:
        return add_calibrated_oct_noise(image, dataset=profile, noise_scale=noise_scale, random_mix=True)

    # Fallback
    params = {
        'duke17': {'speckle_scale': 0.50, 'gaussian_std': 0.10, 'v_corr': 0.19},
        'duke28': {'speckle_scale': 0.48, 'gaussian_std': 0.10, 'v_corr': 0.20},
        'pku37':  {'speckle_scale': 0.35, 'gaussian_std': 0.07, 'v_corr': 0.43},
        'combined': {'speckle_scale': 0.44, 'gaussian_std': 0.09, 'v_corr': 0.27},  # Average of all
    }
    p = params.get(profile, params['duke17'])

    H, W = image.shape
    speckle = (1.0 + p['speckle_scale'] * noise_scale * (np.random.exponential(1.0, (H, W)).astype(np.float32) - 1.0))
    noisy = image * np.maximum(speckle, 0.01)
    gaussian = np.random.randn(H, W).astype(np.float32) * p['gaussian_std'] * noise_scale
    noisy = noisy + gaussian

    return np.clip(noisy, 0, 1, out=noisy).astype(np.float32)


class CalibratedNoiseDataset(Dataset):
    """Dataset with calibrated OCT noise for NAFNet training."""

    def __init__(self, jsonl_path, max_samples=None, noise_levels=None, patch_size=64):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.noise_levels = noise_levels or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.patch_size = patch_size

        # Assign noise profiles to samples (use local RNG to avoid affecting global state)
        n_samples = len(self.samples)
        rng = np.random.RandomState(42)
        group_assignments = rng.choice(3, size=n_samples)
        self.sample_noise_profiles = [NOISE_PROFILES[g] for g in group_assignments]

        # Print distribution
        profile_counts = Counter(self.sample_noise_profiles)
        print(f"[CalibratedNoiseDataset] Noise profile groups:")
        for i, profile in enumerate(NOISE_PROFILES):
            count = profile_counts[profile]
            pct = 100 * count / max(1, n_samples)  # Avoid division by zero
            print(f"  Group {i+1} ({profile}): {count} samples ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples) * len(self.noise_levels)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.noise_levels)
        noise_idx = idx % len(self.noise_levels)
        noise_scale = self.noise_levels[noise_idx]

        sample = self.samples[sample_idx]
        with Image.open(sample['image_path']) as img:
            clean_image = np.array(img.convert('L'), dtype=np.float32) / 255.0

        H, W = clean_image.shape

        # Random patch
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))
        clean_patch = clean_image[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Add calibrated noise
        noise_profile = self.sample_noise_profiles[sample_idx]
        noisy_patch = add_calibrated_noise_by_profile(clean_patch, profile=noise_profile, noise_scale=noise_scale)

        return {
            'noisy': torch.from_numpy(noisy_patch).unsqueeze(0),
            'clean': torch.from_numpy(clean_patch).unsqueeze(0),
        }


class RealNoisePairsDataset(Dataset):
    """Dataset with real noisy-clean pairs (e.g., PKU37, Duke17)."""

    def __init__(self, jsonl_path, max_samples=None, patch_size=64, augment=True):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                # Must have both noisy_path and clean_path
                if 'noisy_path' in sample and 'clean_path' in sample:
                    self.samples.append(sample)

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.patch_size = patch_size
        self.augment = augment

        print(f"[RealNoisePairsDataset] Loaded {len(self.samples)} real noise pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load both noisy and clean images
        with Image.open(sample['noisy_path']) as img:
            noisy_image = np.array(img.convert('L'), dtype=np.float32) / 255.0
        with Image.open(sample['clean_path']) as img:
            clean_image = np.array(img.convert('L'), dtype=np.float32) / 255.0

        H, W = clean_image.shape
        Hn, Wn = noisy_image.shape

        # Resize if dimensions don't match (use scipy for float32 precision)
        if (H, W) != (Hn, Wn):
            from scipy.ndimage import zoom
            noisy_image = zoom(noisy_image, (H / Hn, W / Wn), order=1).astype(np.float32)

        # Random patch
        top = np.random.randint(0, max(1, H - self.patch_size))
        left = np.random.randint(0, max(1, W - self.patch_size))

        noisy_patch = noisy_image[top:top+self.patch_size, left:left+self.patch_size]
        clean_patch = clean_image[top:top+self.patch_size, left:left+self.patch_size]

        # Pad if needed
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - clean_patch.shape[0]
            pad_w = self.patch_size - clean_patch.shape[1]
            noisy_patch = np.pad(noisy_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            clean_patch = np.pad(clean_patch, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                noisy_patch = np.ascontiguousarray(np.fliplr(noisy_patch))
                clean_patch = np.ascontiguousarray(np.fliplr(clean_patch))
            # Random vertical flip
            if np.random.random() > 0.5:
                noisy_patch = np.ascontiguousarray(np.flipud(noisy_patch))
                clean_patch = np.ascontiguousarray(np.flipud(clean_patch))

        return {
            'noisy': torch.from_numpy(noisy_patch).unsqueeze(0),
            'clean': torch.from_numpy(clean_patch).unsqueeze(0),
        }


class ValDataset(Dataset):
    """Validation dataset with full images."""

    def __init__(self, jsonl_path, max_samples=10, noise_level=1.0, real_noise=False):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                if real_noise:
                    # For real noise, must have both paths
                    if 'noisy_path' in sample and 'clean_path' in sample:
                        self.samples.append(sample)
                else:
                    self.samples.append(sample)

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.noise_level = noise_level
        self.real_noise = real_noise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.real_noise:
            # Load real noisy-clean pair
            with Image.open(sample['noisy_path']) as img:
                noisy_image = np.array(img.convert('L'), dtype=np.float32) / 255.0
            with Image.open(sample['clean_path']) as img:
                clean_image = np.array(img.convert('L'), dtype=np.float32) / 255.0
            # Resize noisy if needed (use scipy for float32 precision)
            H, W = clean_image.shape
            Hn, Wn = noisy_image.shape
            if (H, W) != (Hn, Wn):
                from scipy.ndimage import zoom
                noisy_image = zoom(noisy_image, (H / Hn, W / Wn), order=1).astype(np.float32)
        else:
            # Add synthetic noise
            with Image.open(sample['image_path']) as img:
                clean_image = np.array(img.convert('L'), dtype=np.float32) / 255.0
            noisy_image = add_calibrated_noise_by_profile(clean_image, profile='combined', noise_scale=self.noise_level)

        return {
            'noisy': torch.from_numpy(noisy_image).unsqueeze(0),
            'clean': torch.from_numpy(clean_image).unsqueeze(0),
        }


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


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_psnr = 0

    use_cuda = 'cuda' in str(device)
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        noisy = batch['noisy'].to(device, non_blocking=use_cuda)
        clean = batch['clean'].to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)

        # Forward (NAFNet with no FiLM conditioning)
        denoised = model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # Loss
        loss = F.l1_loss(denoised, clean) + 0.1 * F.mse_loss(denoised, clean)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val

        with torch.no_grad():
            psnr = compute_psnr(denoised, clean)
            total_psnr += psnr

        pbar.set_postfix({'loss': f'{loss_val:.4f}', 'psnr': f'{psnr:.1f}'})

        # Cleanup to prevent memory buildup
        del noisy, clean, denoised, loss

    n_batches = max(1, len(loader))  # Avoid division by zero
    return {'loss': total_loss / n_batches, 'psnr': total_psnr / n_batches}


def validate(model, loader, device):
    model.eval()
    total_psnr_noisy = 0
    total_psnr_denoised = 0
    total_ssim_noisy = 0
    total_ssim_denoised = 0
    use_cuda = 'cuda' in str(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            noisy = batch['noisy'].to(device, non_blocking=use_cuda)
            clean = batch['clean'].to(device, non_blocking=use_cuda)

            denoised = model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)

            total_psnr_noisy += compute_psnr(noisy, clean)
            total_psnr_denoised += compute_psnr(denoised, clean)
            total_ssim_noisy += compute_ssim(noisy, clean)
            total_ssim_denoised += compute_ssim(denoised, clean)

            # Cleanup to prevent memory buildup
            del noisy, clean, denoised

    n = max(1, len(loader))  # Avoid division by zero
    return {
        'psnr_noisy': total_psnr_noisy / n,
        'psnr_denoised': total_psnr_denoised / n,
        'ssim_noisy': total_ssim_noisy / n,
        'ssim_denoised': total_ssim_denoised / n,
    }


def main():
    parser = argparse.ArgumentParser(description='Train NAFNet with calibrated OCT noise')
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--output_dir', default='outputs/nafnet_calibrated')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--real_noise', action='store_true', help='Use real noise pairs (noisy_path, clean_path) instead of synthetic')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader num_workers (set to 0 for debugging)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (0 to disable)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    if args.real_noise:
        print("TRAIN NAFNET WITH REAL NOISE PAIRS")
    else:
        print("TRAIN NAFNET WITH CALIBRATED OCT NOISE")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Batch size: {args.batch_size}, Patch size: {args.patch_size}")
    if args.pretrained:
        print(f"Pretrained: {args.pretrained}")
    print("=" * 70)

    # Data
    if args.real_noise:
        train_ds = RealNoisePairsDataset(args.train_jsonl, args.max_train, args.patch_size, augment=True)
        val_ds = ValDataset(args.val_jsonl, args.max_val, real_noise=True)
    else:
        noise_levels = [0.8, 0.9, 1.0, 1.1, 1.2]
        train_ds = CalibratedNoiseDataset(args.train_jsonl, args.max_train, noise_levels, args.patch_size)
        val_ds = ValDataset(args.val_jsonl, args.max_val, real_noise=False)

    use_cuda = 'cuda' in args.device
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda)

    print(f"Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"Val: {len(val_ds)} samples")

    # Model - must match architecture used in joint training
    model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32
    ).to(args.device)

    # Load pretrained weights if provided
    if args.pretrained and os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=args.device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
        if missing:
            print(f"  Missing keys: {len(missing)} (e.g., {missing[:3]})")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_psnr = 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        train_metrics = train_epoch(model, train_loader, optimizer, args.device)
        val_metrics = validate(model, val_loader, args.device)
        scheduler.step()

        print(f"\nTraining: Loss={train_metrics['loss']:.4f}, PSNR={train_metrics['psnr']:.2f} dB")
        print(f"Validation:")
        print(f"  Noisy:    PSNR={val_metrics['psnr_noisy']:.2f} dB, SSIM={val_metrics['ssim_noisy']:.4f}")
        print(f"  Denoised: PSNR={val_metrics['psnr_denoised']:.2f} dB, SSIM={val_metrics['ssim_denoised']:.4f}")
        print(f"  Gain:     PSNR=+{val_metrics['psnr_denoised']-val_metrics['psnr_noisy']:.2f} dB, SSIM=+{val_metrics['ssim_denoised']-val_metrics['ssim_noisy']:.4f}")

        # Save best
        if val_metrics['psnr_denoised'] > best_psnr:
            best_psnr = val_metrics['psnr_denoised']
            epochs_no_improve = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'psnr': best_psnr,
            }, f"{args.output_dir}/nafnet_best.pth")
            print(f"  *** NEW BEST PSNR: {best_psnr:.2f} dB ***")
        else:
            epochs_no_improve += 1
            if args.patience > 0:
                print(f"  No improvement for {epochs_no_improve}/{args.patience} epochs")

        # Save latest
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch,
        }, f"{args.output_dir}/nafnet_latest.pth")

        # Memory cleanup
        if 'cuda' in args.device:
            torch.cuda.empty_cache()
        gc.collect()

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to: {args.output_dir}/nafnet_best.pth")
    print("=" * 70)


if __name__ == '__main__':
    main()
