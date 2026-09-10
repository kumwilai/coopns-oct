#!/usr/bin/env python3
"""
Curriculum Learning for Constrained Neuro-Symbolic OCT Denoiser.

Key insight: The model wasn't learning to denoise because constraints
and clinical losses were interfering with the reconstruction objective.

Solution: Train in phases:
  Phase 1: Pure denoising (MSE only)
  Phase 2: Add clinical losses
  Phase 3: Add constraint penalties

This ensures the model first learns to denoise, then refines for clinical
and anatomical requirements.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import logging

sys.path.insert(0, '/home/kumwilai/OCT')

from constrained_neurosymbolic_denoiser import (
    ConstrainedNeuroSymbolicDenoiser,
    DenoiserConfig,
    ClinicalLosses,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 50.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute simplified SSIM."""
    C1, C2 = 0.01**2, 0.03**2

    pred = pred.squeeze()
    target = target.squeeze()

    mu1 = F.avg_pool2d(pred.unsqueeze(0).unsqueeze(0), window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target.unsqueeze(0).unsqueeze(0), window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.avg_pool2d((pred.unsqueeze(0).unsqueeze(0) ** 2), window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d((target.unsqueeze(0).unsqueeze(0) ** 2), window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred.unsqueeze(0).unsqueeze(0) * target.unsqueeze(0).unsqueeze(0), window_size, stride=1, padding=window_size//2) - mu12

    ssim = ((2*mu12 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean().item()


class PKU37Dataset:
    """Efficient PKU37 dataset with pre-loading."""

    def __init__(self, pku37_root: str, split: str = 'train',
                 patch_size: int = 128, max_samples: int = 50,
                 patches_per_image: int = 4):
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
                clean = np.array(Image.open(clean_path), dtype=np.float32) / 255.0
                noisy = np.array(Image.open(os.path.join(noisy_dir, noisy_fname)), dtype=np.float32) / 255.0
                self.images.append((clean, noisy, fname))
                count += 1
                if count >= max_samples:
                    break
            if count >= max_samples:
                break

        logger.info(f"Loaded {len(self.images)} images for {split}")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        clean, noisy, name = self.images[img_idx]
        H, W = clean.shape

        # Random crop
        h = np.random.randint(0, max(1, H - self.patch_size + 1))
        w = np.random.randint(0, max(1, W - self.patch_size + 1))

        clean_patch = clean[h:h+self.patch_size, w:w+self.patch_size]
        noisy_patch = noisy[h:h+self.patch_size, w:w+self.patch_size]

        # Pad if needed
        ph, pw = clean_patch.shape
        if ph < self.patch_size or pw < self.patch_size:
            clean_patch = np.pad(clean_patch, ((0, self.patch_size-ph), (0, self.patch_size-pw)), mode='reflect')
            noisy_patch = np.pad(noisy_patch, ((0, self.patch_size-ph), (0, self.patch_size-pw)), mode='reflect')

        # Random flip
        if np.random.random() > 0.5:
            clean_patch = clean_patch[:, ::-1].copy()
            noisy_patch = noisy_patch[:, ::-1].copy()

        return {
            'noisy': torch.from_numpy(noisy_patch).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean_patch).unsqueeze(0).float(),
        }

    def get_full_image(self, idx):
        """Get full image for validation."""
        clean, noisy, name = self.images[idx % len(self.images)]
        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean).unsqueeze(0).float(),
            'name': name,
        }


def sliding_window_inference(model, image, patch_size=128, stride=96, device='cpu'):
    """Fast sliding window inference."""
    model.eval()

    B, C, H, W = image.shape
    output = torch.zeros_like(image)
    count = torch.zeros_like(image)

    h_indices = list(range(0, max(1, H - patch_size + 1), stride))
    w_indices = list(range(0, max(1, W - patch_size + 1), stride))

    # Add boundary patches if needed
    if h_indices[-1] + patch_size < H:
        h_indices.append(H - patch_size)
    if w_indices[-1] + patch_size < W:
        w_indices.append(W - patch_size)

    with torch.no_grad():
        for h in h_indices:
            for w in w_indices:
                patch = image[:, :, h:h+patch_size, w:w+patch_size].to(device)
                outputs = model(patch)
                denoised_patch = outputs['denoised'].cpu()

                output[:, :, h:h+patch_size, w:w+patch_size] += denoised_patch
                count[:, :, h:h+patch_size, w:w+patch_size] += 1

    output = output / count.clamp(min=1)
    return output


def evaluate(model, dataset, device, max_samples=5):
    """Quick evaluation."""
    model.eval()
    metrics = defaultdict(list)

    for i in range(min(max_samples, len(dataset.images))):
        data = dataset.get_full_image(i)
        noisy = data['noisy'].unsqueeze(0)
        clean = data['clean'].unsqueeze(0)

        # Sliding window inference
        denoised = sliding_window_inference(model, noisy, patch_size=128, stride=96, device=device)

        # Compute metrics
        psnr = compute_psnr(denoised, clean)
        ssim = compute_ssim(denoised.squeeze(), clean.squeeze())
        input_psnr = compute_psnr(noisy, clean)

        metrics['psnr'].append(psnr)
        metrics['ssim'].append(ssim)
        metrics['input_psnr'].append(input_psnr)
        metrics['gain'].append(psnr - input_psnr)

    return {k: np.mean(v) for k, v in metrics.items()}


def main():
    print("=" * 70)
    print("CURRICULUM LEARNING: Constrained Neuro-Symbolic OCT Denoiser")
    print("=" * 70)

    # Data
    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"

    train_dataset = PKU37Dataset(pku37_root, 'train', patch_size=128, max_samples=50, patches_per_image=4)
    val_dataset = PKU37Dataset(pku37_root, 'val', patch_size=128, max_samples=10, patches_per_image=1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Model
    config = DenoiserConfig(
        encoder_channels=[64, 128, 256],
        latent_dim=32,
        head_width=48,
        lr=2e-3,
        rho_init=0.1,
        rho_max=10.0,
        rho_mult=1.2,
        disentangle_weight=0.05,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.manual_seed(42)
    model = ConstrainedNeuroSymbolicDenoiser(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

    clinical_losses = ClinicalLosses().to(device)

    # Lagrangian parameters
    lambdas = {name: 0.1 for name in ['ordering', 'thickness', 'range', 'smoothness', 'intensity']}
    rho = config.rho_init

    # Initial eval
    print("\n[INITIAL EVALUATION]")
    init_metrics = evaluate(model, val_dataset, device, max_samples=3)
    print(f"  Input PSNR: {init_metrics['input_psnr']:.2f} dB")
    print(f"  Model PSNR: {init_metrics['psnr']:.2f} dB")
    print(f"  PSNR Gain:  {init_metrics['gain']:+.2f} dB")
    print(f"  SSIM:       {init_metrics['ssim']:.4f}")

    # Training phases
    n_epochs = 30
    best_psnr = 0

    print("\n" + "=" * 70)
    print("TRAINING PHASES")
    print("=" * 70)
    print("Phase 1 (Epochs 1-10):  MSE loss only (learn to denoise)")
    print("Phase 2 (Epochs 11-20): Add clinical losses (layer-specific)")
    print("Phase 3 (Epochs 21-30): Add constraint penalties")
    print("=" * 70)

    for epoch in range(n_epochs):
        model.train()

        # Determine phase
        if epoch < 10:
            phase = 1
            mse_weight = 1.0
            clinical_weight = 0.0
            constraint_weight = 0.0
        elif epoch < 20:
            phase = 2
            mse_weight = 1.0
            clinical_weight = min(0.5, (epoch - 10) * 0.05)  # Gradually increase
            constraint_weight = 0.0
        else:
            phase = 3
            mse_weight = 1.0
            clinical_weight = 0.5
            constraint_weight = min(0.1, (epoch - 20) * 0.01)  # Gradually increase

        epoch_loss = 0
        epoch_psnr = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}", leave=False)
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            optimizer.zero_grad()

            outputs = model(noisy)
            denoised = outputs['denoised']

            # MSE loss (ALWAYS the primary objective)
            mse_loss = F.mse_loss(denoised, clean)
            total_loss = mse_weight * mse_loss

            # Phase 2+: Add clinical losses
            if clinical_weight > 0:
                clinical_loss, _ = clinical_losses.compute_all(outputs, clean)
                total_loss = total_loss + clinical_weight * clinical_loss

            # Phase 3: Add constraint penalties
            if constraint_weight > 0:
                violations = model.compute_constraint_violations(outputs)
                constraint_penalty = 0
                for name, viol in violations.items():
                    v = viol.mean()
                    constraint_penalty += lambdas[name] * v + (rho / 2) * F.relu(v) ** 2
                total_loss = total_loss + constraint_weight * constraint_penalty

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

            with torch.no_grad():
                psnr = compute_psnr(denoised, clean)
                epoch_psnr += psnr

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'psnr': f"{psnr:.2f}",
                'phase': phase,
            })

        scheduler.step()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_psnr = epoch_psnr / n_batches

        # Update Lagrangian at end of phase 3 epochs
        if phase == 3:
            with torch.no_grad():
                test_batch = next(iter(train_loader))
                test_outputs = model(test_batch['noisy'].to(device))
                violations = model.compute_constraint_violations(test_outputs)

                max_viol = max(v.mean().item() for v in violations.values())
                if max_viol > 0.01:
                    for name, viol in violations.items():
                        lambdas[name] = max(0, lambdas[name] + rho * viol.mean().item())
                    rho = min(rho * config.rho_mult, config.rho_max)

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            val_metrics = evaluate(model, val_dataset, device, max_samples=5)

            print(f"\nEpoch {epoch+1:2d} | Phase {phase} | Train PSNR: {avg_psnr:.2f} | "
                  f"Val PSNR: {val_metrics['psnr']:.2f} | Gain: {val_metrics['gain']:+.2f} dB | "
                  f"SSIM: {val_metrics['ssim']:.4f}")

            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'val_psnr': best_psnr,
                }, '/home/kumwilai/OCT/outputs/curriculum_best.pth')
                print(f"  [NEW BEST] Saved model (PSNR: {best_psnr:.2f} dB)")
        else:
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1:2d} | Phase {phase} | Loss: {avg_loss:.4f} | Train PSNR: {avg_psnr:.2f}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_metrics = evaluate(model, val_dataset, device, max_samples=10)

    print(f"\nInput PSNR:      {final_metrics['input_psnr']:.2f} dB")
    print(f"Initial PSNR:    {init_metrics['psnr']:.2f} dB")
    print(f"Final PSNR:      {final_metrics['psnr']:.2f} dB")
    print(f"Improvement:     {final_metrics['psnr'] - init_metrics['psnr']:+.2f} dB")
    print(f"Gain over input: {final_metrics['gain']:+.2f} dB")
    print(f"Final SSIM:      {final_metrics['ssim']:.4f}")

    # Check constraint satisfaction
    print("\nConstraint Satisfaction:")
    model.eval()
    with torch.no_grad():
        test_data = val_dataset.get_full_image(0)
        test_noisy = test_data['noisy'].unsqueeze(0).to(device)

        # Use a patch for constraint checking
        patch = test_noisy[:, :, :128, :128]
        outputs = model(patch)
        violations = model.compute_constraint_violations(outputs)

        n_satisfied = 0
        for name, viol in violations.items():
            v = viol.mean().item()
            status = "OK" if v < 0.01 else f"VIOL ({v:.4f})"
            if v < 0.01:
                n_satisfied += 1
            print(f"  {name}: {status}")
        print(f"\nConstraints satisfied: {n_satisfied}/5")

    if final_metrics['gain'] > 0:
        print("\n[SUCCESS] Model learned to denoise with positive PSNR gain!")
    else:
        print("\n[WARNING] Model did not achieve positive PSNR gain - needs more tuning")

    return 0


if __name__ == "__main__":
    # Create output directory
    os.makedirs('/home/kumwilai/OCT/outputs', exist_ok=True)
    sys.exit(main())
