#!/usr/bin/env python3
"""
Train CASA+N2V with explicit coherence supervision.

Key innovation: Add loss term that encourages coherent weights to correlate
with local coefficient of variation (CV), which indicates structured speckle.

Loss = Denoising Loss + λ * Coherence Loss

where Coherence Loss = -correlation(coherent_map, local_CV)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from adaptive_oct_denoise import (
    build_model, PairedOCTDataset, device, resize_to, compute_psnr
)

# ============================================================================
# Coherence-Aware Loss Components
# ============================================================================

def compute_local_cv(image, window_size=11):
    """
    Compute local coefficient of variation (CV).

    High CV = structured speckle (multiplicative) = coherent processing needed
    Low CV = additive noise = incoherent processing needed

    Args:
        image: [B, C, H, W] tensor
        window_size: size of local window

    Returns:
        cv_map: [B, C, H, W] local CV map
    """
    # Efficient local statistics using avg pooling
    kernel_size = window_size
    padding = kernel_size // 2

    # Local mean
    mean_local = F.avg_pool2d(image, kernel_size, stride=1, padding=padding)

    # Local variance: E[X^2] - E[X]^2
    mean_sq_local = F.avg_pool2d(image ** 2, kernel_size, stride=1, padding=padding)
    var_local = mean_sq_local - mean_local ** 2
    std_local = torch.sqrt(torch.clamp(var_local, min=1e-8))

    # Coefficient of variation
    cv_local = std_local / (mean_local + 1e-8)

    return cv_local


def pearson_correlation_loss(x, y):
    """
    Compute negative Pearson correlation as a loss (minimize = maximize correlation).

    Args:
        x, y: [B, C, H, W] tensors

    Returns:
        -correlation: scalar loss (negative correlation coefficient)
    """
    # Flatten spatial dimensions
    x_flat = x.view(x.size(0), x.size(1), -1)  # [B, C, N]
    y_flat = y.view(y.size(0), y.size(1), -1)

    # Center
    x_centered = x_flat - x_flat.mean(dim=2, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=2, keepdim=True)

    # Pearson correlation
    numerator = (x_centered * y_centered).sum(dim=2)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=2) * (y_centered ** 2).sum(dim=2) + 1e-8)
    correlation = numerator / denominator

    # Return negative (to minimize = maximize correlation)
    return -correlation.mean()


def coherence_supervision_loss(coherent_map, noisy_image, window_size=11):
    """
    Coherence-aware loss: Encourage coherent weights to correlate with local CV.

    Args:
        coherent_map: [B, 1, H, W] coherent weight from CASA
        noisy_image: [B, 1, H, W] input noisy image

    Returns:
        loss: scalar coherence loss
    """
    # Compute local CV (ground truth coherence indicator)
    cv_map = compute_local_cv(noisy_image, window_size=window_size)

    # Normalize CV to [0, 1] for better training stability
    cv_normalized = (cv_map - cv_map.min()) / (cv_map.max() - cv_map.min() + 1e-8)

    # Encourage positive correlation
    corr_loss = pearson_correlation_loss(coherent_map, cv_normalized)

    return corr_loss


# ============================================================================
# Modified Training Loop
# ============================================================================

class CoherenceAwareTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )

        # Loss weights
        self.lambda_coherence = config.get('lambda_coherence', 0.1)

        # EMA model for stable evaluation
        self.ema_model = self._create_ema_model()
        self.ema_decay = config.get('ema_decay', 0.999)

        # Tracking
        self.best_psnr = 0
        self.best_correlation = -1  # Track coherence learning

    def _create_ema_model(self):
        """Create EMA copy of model."""
        ema_model = build_model(
            base_channels=self.config['base_channels'],
            residual_mode=self.config['residual_mode'],
            adapter_type=self.config['adapter_type'],
            backbone_type=self.config['backbone_type']
        ).to(device)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        return ema_model

    def _update_ema(self):
        """Update EMA model."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _extract_coherent_map(self, noisy):
        """Extract coherent weight map from CASA adapter."""
        coherent_map = None

        def hook_fn(module, input, output):
            nonlocal coherent_map
            if output.shape[1] == 2:  # Decomposition output
                coherent_map = output[:, 0:1]  # Channel 0 = coherent weight

        # Register hook
        hook_handle = None
        for name, module in self.model.named_modules():
            if name == 'adapter.decomposition_net':
                hook_handle = module.register_forward_hook(hook_fn)
                break

        # Forward pass
        _ = self.model(noisy)

        # Remove hook
        if hook_handle:
            hook_handle.remove()

        return coherent_map

    def train_epoch(self, epoch):
        """Train for one epoch with coherence supervision."""
        self.model.train()

        total_loss = 0
        total_denoise_loss = 0
        total_coherence_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')

        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(device)
            clean = clean.to(device)

            self.optimizer.zero_grad()

            # Extract coherent map during forward pass
            coherent_map = self._extract_coherent_map(noisy)

            # Standard denoising loss (N2V uses noisy-to-noisy)
            denoised = self.model(noisy)
            denoise_loss = F.mse_loss(denoised, clean)

            # Coherence supervision loss
            if coherent_map is not None and self.lambda_coherence > 0:
                coherence_loss = coherence_supervision_loss(coherent_map, noisy)
            else:
                coherence_loss = torch.tensor(0.0, device=device)

            # Combined loss
            loss = denoise_loss + self.lambda_coherence * coherence_loss

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self._update_ema()

            # Track
            total_loss += loss.item()
            total_denoise_loss += denoise_loss.item()
            if isinstance(coherence_loss, torch.Tensor):
                total_coherence_loss += coherence_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'denoise': f'{denoise_loss.item():.4f}',
                'coherence': f'{coherence_loss.item() if isinstance(coherence_loss, torch.Tensor) else 0:.4f}'
            })

        n_batches = len(self.train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'denoise_loss': total_denoise_loss / n_batches,
            'coherence_loss': total_coherence_loss / n_batches
        }

    def validate(self, epoch):
        """Validate and check coherence learning."""
        self.ema_model.eval()

        total_psnr = 0
        total_correlation = 0
        n_samples = 0

        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)

                # Denoise
                denoised = self.ema_model(noisy)

                # PSNR
                for i in range(noisy.size(0)):
                    psnr = compute_psnr(denoised[i:i+1], clean[i:i+1])
                    total_psnr += psnr
                    n_samples += 1

                # Check coherence correlation
                # Extract coherent map
                coherent_map = None

                def hook_fn(module, input, output):
                    nonlocal coherent_map
                    if output.shape[1] == 2:
                        coherent_map = output[:, 0:1]

                hook_handle = None
                for name, module in self.ema_model.named_modules():
                    if name == 'adapter.decomposition_net':
                        hook_handle = module.register_forward_hook(hook_fn)
                        break

                _ = self.ema_model(noisy)

                if hook_handle:
                    hook_handle.remove()

                if coherent_map is not None:
                    cv_map = compute_local_cv(noisy)

                    # Compute correlation
                    coherent_flat = coherent_map.view(coherent_map.size(0), -1)
                    cv_flat = cv_map.view(cv_map.size(0), -1)

                    for i in range(coherent_flat.size(0)):
                        c_centered = coherent_flat[i] - coherent_flat[i].mean()
                        cv_centered = cv_flat[i] - cv_flat[i].mean()

                        corr = (c_centered * cv_centered).sum() / \
                               (torch.sqrt((c_centered ** 2).sum() * (cv_centered ** 2).sum()) + 1e-8)

                        total_correlation += corr.item()

        avg_psnr = total_psnr / n_samples
        avg_correlation = total_correlation / n_samples

        print(f"\n[Validation] PSNR: {avg_psnr:.2f} dB, Coherence Correlation: {avg_correlation:.3f}")

        # Save best model
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint(epoch, 'best_psnr')
            print(f"✓ New best PSNR: {avg_psnr:.2f} dB")

        if avg_correlation > self.best_correlation:
            self.best_correlation = avg_correlation
            self.save_checkpoint(epoch, 'best_correlation')
            print(f"✓ New best correlation: {avg_correlation:.3f}")

        return {
            'psnr': avg_psnr,
            'correlation': avg_correlation
        }

    def save_checkpoint(self, epoch, name):
        """Save checkpoint."""
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'best_correlation': self.best_correlation,
            'config': self.config
        }

        torch.save(checkpoint, save_dir / f'{name}.pth')

    def train(self):
        """Full training loop."""
        print("=" * 80)
        print("TRAINING CASA+N2V WITH COHERENCE SUPERVISION")
        print("=" * 80)
        print(f"Lambda (coherence weight): {self.lambda_coherence}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("=" * 80)

        for epoch in range(self.config['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)

            print(f"\n[Epoch {epoch+1}] Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Denoise: {train_metrics['denoise_loss']:.4f}, "
                  f"Coherence: {train_metrics['coherence_loss']:.4f})")

            # Validate
            if (epoch + 1) % self.config.get('val_interval', 1) == 0:
                val_metrics = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()

            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}')

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best Correlation: {self.best_correlation:.3f}")
        print("=" * 80)


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == '__main__':
    # Configuration
    config = {
        # Model
        'base_channels': 48,
        'residual_mode': True,
        'adapter_type': 'casa',
        'backbone_type': 'noise2void',

        # Training
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'min_lr': 1e-6,

        # Coherence supervision
        'lambda_coherence': 0.1,  # Weight for coherence loss (tune this!)

        # EMA
        'ema_decay': 0.999,

        # Data
        'train_pairs': 'train_pairs_universal.txt',
        'val_pairs': 'val_pairs_universal.txt',
        'image_size': 64,

        # Checkpointing
        'save_dir': 'checkpoints/casa_coherence_supervised',
        'val_interval': 1,
        'save_interval': 10
    }

    print("Loading datasets...")
    transform = resize_to((config['image_size'], config['image_size']))

    train_dataset = PairedOCTDataset(config['train_pairs'], transform=transform)
    val_dataset = PairedOCTDataset(config['val_pairs'], transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(
        base_channels=config['base_channels'],
        residual_mode=config['residual_mode'],
        adapter_type=config['adapter_type'],
        backbone_type=config['backbone_type']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Train
    trainer = CoherenceAwareTrainer(model, train_loader, val_loader, config)
    trainer.train()
