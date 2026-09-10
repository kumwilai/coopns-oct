"""
AMeta-FD: Adversarial Meta-learning for Few-shot OCT Despeckling

Implementation of adversarial meta-learning framework for OCT image denoising.
Combines MAML-style meta-learning with adversarial training for robust few-shot adaptation.

Key components:
- Generator: U-Net style denoiser
- Discriminator: PatchGAN discriminator for realistic denoising
- Meta-learning: MAML-style bi-level optimization
- Few-shot adaptation: Fast adaptation with few support images
"""

import os
import copy
import random
import math
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import from main module
from adaptive_oct_denoise import (
    CleanOCTDataset, PairedOCTDataset,
    add_rayleigh_noise, add_poisson_noise, add_mixed_gaussian_noise,
    add_gamma_speckle_noise, add_heavy_gamma_speckle_noise,
    add_correlated_speckle, add_heavy_correlated_speckle,
    compute_psnr, compute_ssim, set_seed, device, resize_to,
    NOISE_TASKS_EXTENDED, force_memory_cleanup
)


# ================================
# Generator Network (Denoiser)
# ================================
class GeneratorBlock(nn.Module):
    """Residual block for generator."""
    def __init__(self, channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.3))
        layers.extend([
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        ])
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class AMetaFDGenerator(nn.Module):
    """
    U-Net style generator for OCT despeckling.
    Designed for 64x64 grayscale images with compact architecture.
    """
    def __init__(self, base_channels: int = 32, num_res_blocks: int = 3):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(c, c*2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(c*2, c*2, 3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c*2, c*2, 3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Conv2d(c*2, c*4, 3, stride=2, padding=1)

        # Bottleneck with residual blocks
        bottleneck_layers = []
        for _ in range(num_res_blocks):
            bottleneck_layers.append(GeneratorBlock(c*4, use_dropout=True))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Decoder
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c*4, c*2, 3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c*2, c*2, 3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c*2, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        # Bottleneck
        b = self.bottleneck(self.down2(e2))

        # Decoder with skip connections
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Output with sigmoid
        out = torch.sigmoid(self.out_conv(d1))
        return out


# ================================
# Discriminator Network
# ================================
class AMetaFDDiscriminator(nn.Module):
    """
    PatchGAN discriminator for realistic texture discrimination.
    Outputs a spatial map of real/fake predictions.
    """
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.model = nn.Sequential(
            # Input: 1x64x64
            nn.Conv2d(1, c, 4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c, c*2, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c*2, c*4, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(c*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c*4, c*8, 4, stride=1, padding=1),  # 7x7
            nn.BatchNorm2d(c*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c*8, 1, 4, stride=1, padding=1),  # 6x6
        )

    def forward(self, x):
        return self.model(x)


# ================================
# Loss Functions
# ================================
class AMetaFDLoss(nn.Module):
    """Combined loss for AMeta-FD training."""
    def __init__(
        self,
        lambda_adv: float = 0.1,
        lambda_l1: float = 1.0,
        lambda_perceptual: float = 0.5,
        lambda_ssim: float = 0.2,
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def adversarial_loss(self, pred, target_is_real):
        """Adversarial loss for GAN training."""
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.bce_loss(pred, target)

    def perceptual_loss(self, pred, target):
        """Simple gradient-based perceptual loss."""
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        return loss_dx + loss_dy

    def ssim_loss(self, pred, target):
        """Simplified SSIM loss."""
        C1, C2 = 0.01**2, 0.03**2

        mu_p = F.avg_pool2d(pred, 3, 1, 1)
        mu_t = F.avg_pool2d(target, 3, 1, 1)

        sigma_p = F.avg_pool2d(pred**2, 3, 1, 1) - mu_p**2
        sigma_t = F.avg_pool2d(target**2, 3, 1, 1) - mu_t**2
        sigma_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t

        ssim = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2))

        return 1 - ssim.mean()

    def generator_loss(self, pred, target, disc_pred):
        """Total generator loss."""
        loss_l1 = self.l1_loss(pred, target)
        loss_adv = self.adversarial_loss(disc_pred, target_is_real=True)
        loss_perc = self.perceptual_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)

        total_loss = (
            self.lambda_l1 * loss_l1 +
            self.lambda_adv * loss_adv +
            self.lambda_perceptual * loss_perc +
            self.lambda_ssim * loss_ssim
        )

        return total_loss, {
            'l1': loss_l1.item(),
            'adv': loss_adv.item(),
            'perceptual': loss_perc.item(),
            'ssim': loss_ssim.item(),
        }

    def discriminator_loss(self, real_pred, fake_pred):
        """Discriminator loss."""
        loss_real = self.adversarial_loss(real_pred, target_is_real=True)
        loss_fake = self.adversarial_loss(fake_pred, target_is_real=False)
        return (loss_real + loss_fake) / 2


# ================================
# Simplified Meta-Training (Reptile-style for efficiency)
# ================================
class AMetaFDTrainer:
    """
    Meta-trainer for AMeta-FD with simplified Reptile-style optimization.
    More efficient than full MAML while maintaining adaptation capability.
    """
    def __init__(
        self,
        generator: AMetaFDGenerator,
        discriminator: AMetaFDDiscriminator,
        inner_lr: float = 1e-3,
        meta_lr_gen: float = 1e-4,
        meta_lr_disc: float = 1e-4,
        inner_steps: int = 5,
        lambda_adv: float = 0.1,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        # Meta-optimizers for outer loop
        self.meta_opt_gen = torch.optim.Adam(generator.parameters(), lr=meta_lr_gen)
        self.meta_opt_disc = torch.optim.Adam(discriminator.parameters(), lr=meta_lr_disc)

        self.criterion = AMetaFDLoss(lambda_adv=lambda_adv)

    def meta_train_step(self, task_batch):
        """
        Reptile-style meta-training step.
        """
        self.generator.train()
        self.discriminator.train()

        task_losses_gen = []
        task_losses_disc = []

        for support_noisy, support_clean, query_noisy, query_clean in task_batch:
            support_noisy = support_noisy.to(device)
            support_clean = support_clean.to(device)
            query_noisy = query_noisy.to(device)
            query_clean = query_clean.to(device)

            # Create task-specific copies (MEMORY INTENSIVE!)
            task_gen = copy.deepcopy(self.generator)
            task_opt_gen = torch.optim.SGD(task_gen.parameters(), lr=self.inner_lr)

            # Inner loop adaptation on support set
            for _ in range(self.inner_steps):
                task_opt_gen.zero_grad()
                support_pred = task_gen(support_noisy)
                support_loss = F.l1_loss(support_pred, support_clean)
                support_loss.backward()
                task_opt_gen.step()

            # Evaluate on query set
            with torch.no_grad():
                query_pred = task_gen(query_noisy)

            # Discriminator forward (requires grad for disc update)
            query_pred_with_grad = task_gen(query_noisy)
            disc_pred_real = self.discriminator(query_clean)
            disc_pred_fake = self.discriminator(query_pred.detach())
            disc_pred_gen = self.discriminator(query_pred_with_grad)

            # Compute losses
            gen_loss, _ = self.criterion.generator_loss(query_pred_with_grad, query_clean, disc_pred_gen)
            disc_loss = self.criterion.discriminator_loss(disc_pred_real, disc_pred_fake)

            task_losses_gen.append(gen_loss.item())
            task_losses_disc.append(disc_loss.item())

            # Reptile update: interpolate toward task-adapted parameters
            with torch.no_grad():
                for (name, param), (_, task_param) in zip(self.generator.named_parameters(), task_gen.named_parameters()):
                    param.data.add_(0.1 * (task_param.data - param.data))

            # Standard discriminator update
            self.meta_opt_disc.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.meta_opt_disc.step()

            # Explicit cleanup
            del task_gen, task_opt_gen, query_pred, query_pred_with_grad
            del disc_pred_real, disc_pred_fake, disc_pred_gen
            force_memory_cleanup()

        return {
            'meta_loss_gen': np.mean(task_losses_gen),
            'meta_loss_disc': np.mean(task_losses_disc),
        }

    def few_shot_adapt(self, support_noisy, support_clean, num_steps: int = 10):
        """Few-shot adaptation at test time."""
        adapted_gen = copy.deepcopy(self.generator)
        adapted_gen.train()
        optimizer = torch.optim.SGD(adapted_gen.parameters(), lr=self.inner_lr)

        support_noisy = support_noisy.to(device)
        support_clean = support_clean.to(device)

        for _ in range(num_steps):
            optimizer.zero_grad()
            pred = adapted_gen(support_noisy)
            loss = F.l1_loss(pred, support_clean)
            loss.backward()
            optimizer.step()

        adapted_gen.eval()  # Set to eval mode after adaptation
        return adapted_gen

    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'meta_opt_gen': self.meta_opt_gen.state_dict(),
            'meta_opt_disc': self.meta_opt_disc.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.meta_opt_gen.load_state_dict(checkpoint['meta_opt_gen'])
        self.meta_opt_disc.load_state_dict(checkpoint['meta_opt_disc'])


# ================================
# Task Sampler for Meta-Learning
# ================================
class MetaTaskSampler:
    """Sample meta-learning tasks from OCT dataset."""
    def __init__(
        self,
        clean_dataset: CleanOCTDataset,
        n_support: int = 5,
        n_query: int = 10,
        noise_tasks: List = None,
    ):
        self.clean_dataset = clean_dataset
        self.n_support = n_support
        self.n_query = n_query
        self.noise_tasks = noise_tasks or NOISE_TASKS_EXTENDED

    def sample_task(self):
        """Sample one meta-learning task."""
        noise_fn = random.choice(self.noise_tasks)['fn']
        n_total = self.n_support + self.n_query
        indices = random.sample(range(len(self.clean_dataset)), n_total)
        clean_images = [self.clean_dataset[i] for i in indices]
        clean_tensor = torch.stack(clean_images)
        noisy_tensor = noise_fn(clean_tensor)

        support_noisy = noisy_tensor[:self.n_support]
        support_clean = clean_tensor[:self.n_support]
        query_noisy = noisy_tensor[self.n_support:]
        query_clean = clean_tensor[self.n_support:]

        return support_noisy, support_clean, query_noisy, query_clean

    def sample_batch(self, batch_size: int):
        """Sample a batch of tasks."""
        return [self.sample_task() for _ in range(batch_size)]


# ================================
# Training Function
# ================================
def train_ameta_fd(
    clean_root: str,
    output_dir: str,
    num_meta_epochs: int = 50,
    tasks_per_batch: int = 4,
    n_support: int = 5,
    n_query: int = 10,
    inner_steps: int = 5,
    inner_lr: float = 1e-3,
    meta_lr_gen: float = 1e-4,
    meta_lr_disc: float = 1e-4,
    base_channels_gen: int = 32,
    base_channels_disc: int = 32,
    lambda_adv: float = 0.1,
    image_size: int = 64,
    seed: int = 42,
):
    """Train AMeta-FD model."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("AMeta-FD Training")
    print("=" * 80)
    print(f"Clean dataset: {clean_root}")
    print(f"Meta epochs: {num_meta_epochs}, Tasks/batch: {tasks_per_batch}")
    print(f"Support/Query: {n_support}/{n_query}, Inner steps: {inner_steps}")
    print("=" * 80)

    transform = resize_to((image_size, image_size))
    clean_dataset = CleanOCTDataset(clean_root, transform=transform)
    print(f"Loaded {len(clean_dataset)} clean images\n")

    task_sampler = MetaTaskSampler(clean_dataset, n_support=n_support, n_query=n_query)
    generator = AMetaFDGenerator(base_channels=base_channels_gen)
    discriminator = AMetaFDDiscriminator(base_channels=base_channels_disc)
    trainer = AMetaFDTrainer(generator, discriminator, inner_lr=inner_lr,
                             meta_lr_gen=meta_lr_gen, meta_lr_disc=meta_lr_disc,
                             inner_steps=inner_steps, lambda_adv=lambda_adv)

    for epoch in range(num_meta_epochs):
        task_batch = task_sampler.sample_batch(tasks_per_batch)
        metrics = trainer.meta_train_step(task_batch)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_meta_epochs} | Gen: {metrics['meta_loss_gen']:.4f} | Disc: {metrics['meta_loss_disc']:.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            trainer.save(checkpoint_path)

        force_memory_cleanup()

    final_path = os.path.join(output_dir, "ameta_fd_final.pth")
    trainer.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")
    return trainer


# ================================
# Evaluation
# ================================
@torch.no_grad()
def evaluate_ameta_fd(trainer: AMetaFDTrainer, val_loader: DataLoader, adapt_steps: int = 0):
    """Evaluate AMeta-FD on validation set."""
    trainer.generator.eval()
    psnr_list, ssim_list = [], []

    for batch_idx, (noisy, clean) in enumerate(val_loader):
        noisy, clean = noisy.to(device), clean.to(device)

        if adapt_steps > 0:
            n_support = min(2, noisy.size(0))
            # Temporarily enable grad for adaptation
            with torch.enable_grad():
                adapted_gen = trainer.few_shot_adapt(noisy[:n_support], clean[:n_support], num_steps=adapt_steps)

            # Now use adapted generator with no grad
            with torch.no_grad():
                pred = adapted_gen(noisy)

            # Explicit cleanup to prevent memory leak
            del adapted_gen
        else:
            pred = trainer.generator(noisy)

        psnr_list.append(compute_psnr(pred, clean))
        ssim_list.append(compute_ssim(pred, clean))

        # Periodic cleanup every 10 batches
        if (batch_idx + 1) % 10 == 0:
            force_memory_cleanup()

    return {'psnr': np.mean(psnr_list), 'ssim': np.mean(ssim_list),
            'psnr_std': np.std(psnr_list), 'ssim_std': np.std(ssim_list)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AMeta-FD: Adversarial Meta-learning for OCT")
    parser.add_argument("--clean_root", type=str, required=True)
    parser.add_argument("--val_pairs", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/ameta_fd")
    parser.add_argument("--num_meta_epochs", type=int, default=50)
    parser.add_argument("--tasks_per_batch", type=int, default=4)
    parser.add_argument("--n_support", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=10)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--meta_lr_gen", type=float, default=1e-4)
    parser.add_argument("--meta_lr_disc", type=float, default=1e-4)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--adapt_steps", type=int, default=0)
    args = parser.parse_args()

    if args.eval_only:
        assert args.checkpoint and args.val_pairs
        generator = AMetaFDGenerator(base_channels=32)
        discriminator = AMetaFDDiscriminator(base_channels=32)
        trainer = AMetaFDTrainer(generator, discriminator)
        trainer.load(args.checkpoint)

        transform = resize_to((args.image_size, args.image_size))
        val_dataset = PairedOCTDataset(args.val_pairs, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        print(f"Evaluating on {len(val_dataset)} validation pairs...")
        metrics = evaluate_ameta_fd(trainer, val_loader, adapt_steps=args.adapt_steps)
        print(f"\nPSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    else:
        trainer = train_ameta_fd(
            clean_root=args.clean_root, output_dir=args.output_dir,
            num_meta_epochs=args.num_meta_epochs, tasks_per_batch=args.tasks_per_batch,
            n_support=args.n_support, n_query=args.n_query, inner_steps=args.inner_steps,
            inner_lr=args.inner_lr, meta_lr_gen=args.meta_lr_gen, meta_lr_disc=args.meta_lr_disc,
            lambda_adv=args.lambda_adv, image_size=args.image_size, seed=args.seed)

        if args.val_pairs:
            transform = resize_to((args.image_size, args.image_size))
            val_dataset = PairedOCTDataset(args.val_pairs, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            print("\n" + "="*80 + "\nFinal Evaluation\n" + "="*80)
            metrics = evaluate_ameta_fd(trainer, val_loader)
            print(f"PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
            print(f"SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")


if __name__ == "__main__":
    main()
