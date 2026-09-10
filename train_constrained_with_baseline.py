#!/usr/bin/env python3
"""
Train Constrained Neuro-Symbolic OCT Denoiser with NAFNet Baseline Comparison.

Features:
- Detailed per-layer monitoring (PSNR, SSIM, clinical losses)
- NAFNet baseline comparison for each layer
- Gain tracking vs baseline
- Comprehensive epoch logging
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

# Try to import NAFNet
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False
    print("Warning: NAFNet not available for baseline comparison")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class PKU37DirectDataset(Dataset):
    """Load directly from PKU37 directory structure."""

    def __init__(
        self,
        pku37_root: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        patch_size: int = 256,
    ):
        self.patch_size = patch_size
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
                noisy_path = os.path.join(noisy_dir, noisy_fname)
                self.pairs.append({
                    'clean_path': clean_path,
                    'noisy_path': noisy_path,
                })

                if max_samples and len(self.pairs) >= max_samples:
                    break

            if max_samples and len(self.pairs) >= max_samples:
                break

        logger.info(f"Loaded {len(self.pairs)} pairs for {split}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        clean_img = Image.open(pair['clean_path'])
        noisy_img = Image.open(pair['noisy_path'])

        clean = np.array(clean_img, dtype=np.float32)
        noisy = np.array(noisy_img, dtype=np.float32)

        if clean.max() > 1:
            clean = clean / 255.0
        if noisy.max() > 1:
            noisy = noisy / 255.0

        H, W = clean.shape
        if H != self.patch_size or W != self.patch_size:
            clean_pil = Image.fromarray((clean * 255).astype(np.uint8))
            noisy_pil = Image.fromarray((noisy * 255).astype(np.uint8))
            clean = np.array(clean_pil.resize((self.patch_size, self.patch_size), Image.BILINEAR), dtype=np.float32) / 255.0
            noisy = np.array(noisy_pil.resize((self.patch_size, self.patch_size), Image.BILINEAR), dtype=np.float32) / 255.0

        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0),
            'clean': torch.from_numpy(clean).unsqueeze(0),
            'name': os.path.basename(pair['noisy_path']),
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
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_p = pred.mean()
    mu_t = target.mean()
    var_p = pred.var()
    var_t = target.var()
    cov = ((pred - mu_p) * (target - mu_t)).mean()
    ssim = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
    return ssim.item()


def compute_masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mask_sum = mask.sum().clamp(min=1.0)
    mse = ((pred - target) ** 2 * mask).sum() / mask_sum
    if mse.item() < 1e-10:
        return 50.0
    return 10 * np.log10(1.0 / mse.item())


def compute_masked_ssim(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    # Masked SSIM approximation
    mask_sum = mask.sum().clamp(min=1.0)

    mu_p = (pred * mask).sum() / mask_sum
    mu_t = (target * mask).sum() / mask_sum

    var_p = ((pred - mu_p) ** 2 * mask).sum() / mask_sum
    var_t = ((target - mu_t) ** 2 * mask).sum() / mask_sum
    cov = ((pred - mu_p) * (target - mu_t) * mask).sum() / mask_sum

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
    return ssim.item()


# =============================================================================
# NAFNet Baseline
# =============================================================================

class NAFNetBaseline:
    """NAFNet baseline for comparison."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.model = None

        if HAS_NAFNET and os.path.exists(checkpoint_path):
            logger.info(f"Loading NAFNet baseline from {checkpoint_path}")
            # Architecture matches the PKU37 checkpoint
            self.model = NAFNet(img_channel=1, width=64, middle_blk_num=2,
                               enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2])

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(device)
            self.model.eval()
            logger.info(f"NAFNet baseline loaded (epoch {checkpoint.get('epoch', '?')}, PSNR {checkpoint.get('psnr', '?')})")
        else:
            logger.warning("NAFNet baseline not available")

    @torch.no_grad()
    def denoise(self, noisy: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return noisy  # Return input if no baseline
        return self.model(noisy)

    def is_available(self) -> bool:
        return self.model is not None


# =============================================================================
# Detailed Metrics Calculator
# =============================================================================

class DetailedMetrics:
    """Calculate detailed per-layer metrics."""

    LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL_IS', 'RPE_Choroid']
    CONSTRAINT_NAMES = ['ordering', 'thickness', 'range', 'smoothness', 'intensity']

    def __init__(self, device: torch.device):
        self.device = device
        self.clinical_losses = ClinicalLosses().to(device)

    def compute_all(
        self,
        model_output: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        noisy: torch.Tensor,
        nafnet_output: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """Compute all metrics including per-layer and baseline comparison."""

        denoised = model_output['denoised']
        soft_masks = model_output['soft_masks']

        metrics = {}

        # === GLOBAL METRICS ===
        metrics['global_psnr'] = compute_psnr(denoised, clean)
        metrics['global_ssim'] = compute_ssim(denoised, clean)
        metrics['input_psnr'] = compute_psnr(noisy, clean)
        metrics['input_ssim'] = compute_ssim(noisy, clean)

        # NAFNet baseline
        if nafnet_output is not None:
            metrics['nafnet_psnr'] = compute_psnr(nafnet_output, clean)
            metrics['nafnet_ssim'] = compute_ssim(nafnet_output, clean)
            metrics['gain_vs_nafnet_psnr'] = metrics['global_psnr'] - metrics['nafnet_psnr']
            metrics['gain_vs_nafnet_ssim'] = metrics['global_ssim'] - metrics['nafnet_ssim']

        # Gain vs input
        metrics['gain_vs_input_psnr'] = metrics['global_psnr'] - metrics['input_psnr']
        metrics['gain_vs_input_ssim'] = metrics['global_ssim'] - metrics['input_ssim']

        # === PER-LAYER METRICS ===
        for i, name in enumerate(self.LAYER_NAMES):
            mask = soft_masks[:, i:i+1]

            # Our model
            metrics[f'{name}_psnr'] = compute_masked_psnr(denoised, clean, mask)
            metrics[f'{name}_ssim'] = compute_masked_ssim(denoised, clean, mask)

            # Input
            metrics[f'{name}_input_psnr'] = compute_masked_psnr(noisy, clean, mask)
            metrics[f'{name}_input_ssim'] = compute_masked_ssim(noisy, clean, mask)

            # Gain vs input
            metrics[f'{name}_gain_psnr'] = metrics[f'{name}_psnr'] - metrics[f'{name}_input_psnr']
            metrics[f'{name}_gain_ssim'] = metrics[f'{name}_ssim'] - metrics[f'{name}_input_ssim']

            # NAFNet baseline per layer
            if nafnet_output is not None:
                metrics[f'{name}_nafnet_psnr'] = compute_masked_psnr(nafnet_output, clean, mask)
                metrics[f'{name}_nafnet_ssim'] = compute_masked_ssim(nafnet_output, clean, mask)
                metrics[f'{name}_gain_vs_nafnet_psnr'] = metrics[f'{name}_psnr'] - metrics[f'{name}_nafnet_psnr']
                metrics[f'{name}_gain_vs_nafnet_ssim'] = metrics[f'{name}_ssim'] - metrics[f'{name}_nafnet_ssim']

        # === CLINICAL LOSSES ===
        _, clinical_details = self.clinical_losses.compute_all(model_output, clean)
        for key in ['rnfl_clinical', 'inl_clinical', 'onl_clinical', 'rpe_clinical']:
            if key in clinical_details:
                metrics[key] = clinical_details[key]

        return metrics


# =============================================================================
# Training with Detailed Monitoring
# =============================================================================

class ConstrainedTrainerWithMonitoring:
    """Trainer with detailed per-layer monitoring and NAFNet baseline comparison."""

    CONSTRAINT_NAMES = ['ordering', 'thickness', 'range', 'smoothness', 'intensity']
    LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL_IS', 'RPE_Choroid']

    def __init__(
        self,
        model: ConstrainedNeuroSymbolicDenoiser,
        config: DenoiserConfig,
        device: torch.device,
        nafnet_baseline: Optional[NAFNetBaseline] = None,
        lr: float = 1e-4,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.nafnet = nafnet_baseline

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

        self.clinical_losses = ClinicalLosses().to(device)
        self.metrics_calculator = DetailedMetrics(device)

        # Lagrange multipliers
        self.lambdas = {name: torch.tensor(1.0, device=device) for name in self.CONSTRAINT_NAMES}
        self.rho = config.rho_init

        self.best_psnr = 0
        self.best_epoch = 0

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch with detailed monitoring."""
        self.model.train()

        epoch_metrics = defaultdict(list)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, ncols=100)
        for batch in pbar:
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(noisy)

            # Clinical loss
            clinical_loss, clinical_details = self.clinical_losses.compute_all(outputs, clean)

            # Disentanglement loss
            disentangle_loss, _ = self.model.encoder.compute_disentanglement_loss(outputs['representations'])

            # Constraint violations
            violations = self.model.compute_constraint_violations(outputs)

            # Augmented Lagrangian penalty
            constraint_penalty = torch.tensor(0.0, device=self.device)
            for name, viol in violations.items():
                v = viol.mean()
                penalty = self.lambdas[name] * v + (self.rho / 2) * F.relu(v) ** 2
                constraint_penalty = constraint_penalty + 0.1 * penalty
                epoch_metrics[f'{name}_viol'].append(v.item())

            # Total loss
            total_loss = clinical_loss + self.config.disentangle_weight * disentangle_loss + constraint_penalty

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track losses
            epoch_metrics['total_loss'].append(total_loss.item())
            epoch_metrics['clinical_loss'].append(clinical_loss.item())
            epoch_metrics['constraint_loss'].append(constraint_penalty.item())

            for key in ['rnfl_clinical', 'inl_clinical', 'onl_clinical', 'rpe_clinical']:
                if key in clinical_details:
                    epoch_metrics[key].append(clinical_details[key])

            # Quick PSNR
            with torch.no_grad():
                psnr = compute_psnr(outputs['denoised'], clean)
                epoch_metrics['train_psnr'].append(psnr)

            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'psnr': f"{psnr:.1f}"})

        # Update Lagrangian
        avg_violations = {name: np.mean(epoch_metrics[f'{name}_viol']) for name in self.CONSTRAINT_NAMES}
        self._update_lagrangian(avg_violations)
        self.scheduler.step()

        return {k: np.mean(v) for k, v in epoch_metrics.items()}

    def _update_lagrangian(self, violations: Dict[str, float]):
        max_viol = max(violations.values())
        for name, viol in violations.items():
            self.lambdas[name] = F.relu(self.lambdas[name] + self.rho * torch.tensor(viol, device=self.device))
        if max_viol > 0.01:
            self.rho = min(self.rho * self.config.rho_mult, self.config.rho_max)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validation with detailed per-layer metrics and NAFNet comparison."""
        self.model.eval()

        all_metrics = defaultdict(list)

        for batch in tqdm(loader, desc="Validating", leave=False, ncols=100):
            noisy = batch['noisy'].to(self.device)
            clean = batch['clean'].to(self.device)

            # Our model
            outputs = self.model(noisy)

            # NAFNet baseline
            nafnet_output = None
            if self.nafnet and self.nafnet.is_available():
                nafnet_output = self.nafnet.denoise(noisy)

            # Compute detailed metrics
            metrics = self.metrics_calculator.compute_all(outputs, clean, noisy, nafnet_output)

            for k, v in metrics.items():
                all_metrics[k].append(v)

            # Constraints
            violations = self.model.compute_constraint_violations(outputs)
            for name, v in violations.items():
                all_metrics[f'{name}_viol'].append(v.mean().item())

        # Average
        results = {k: np.mean(v) for k, v in all_metrics.items()}

        # Count satisfied constraints
        results['n_constraints_satisfied'] = sum(
            1 for name in self.CONSTRAINT_NAMES
            if results.get(f'{name}_viol', 1.0) < 0.01
        )

        return results

    def print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """Print detailed epoch summary."""

        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1} SUMMARY | Time: {epoch_time:.1f}s | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*80}")

        # Losses
        print(f"\n📉 LOSSES:")
        print(f"   Total: {train_metrics.get('total_loss', 0):.4f} | "
              f"Clinical: {train_metrics.get('clinical_loss', 0):.4f} | "
              f"Constraint: {train_metrics.get('constraint_loss', 0):.4f}")

        # Global metrics
        print(f"\n📊 GLOBAL METRICS:")
        print(f"   {'Metric':<12} {'Input':>10} {'NAFNet':>10} {'Ours':>10} {'Gain vs NAF':>12}")
        print(f"   {'-'*56}")

        input_psnr = val_metrics.get('input_psnr', 0)
        nafnet_psnr = val_metrics.get('nafnet_psnr', input_psnr)
        our_psnr = val_metrics.get('global_psnr', 0)
        gain_psnr = val_metrics.get('gain_vs_nafnet_psnr', our_psnr - nafnet_psnr)

        input_ssim = val_metrics.get('input_ssim', 0)
        nafnet_ssim = val_metrics.get('nafnet_ssim', input_ssim)
        our_ssim = val_metrics.get('global_ssim', 0)
        gain_ssim = val_metrics.get('gain_vs_nafnet_ssim', our_ssim - nafnet_ssim)

        gain_psnr_str = f"+{gain_psnr:.2f}" if gain_psnr >= 0 else f"{gain_psnr:.2f}"
        gain_ssim_str = f"+{gain_ssim:.4f}" if gain_ssim >= 0 else f"{gain_ssim:.4f}"

        print(f"   {'PSNR (dB)':<12} {input_psnr:>10.2f} {nafnet_psnr:>10.2f} {our_psnr:>10.2f} {gain_psnr_str:>12}")
        print(f"   {'SSIM':<12} {input_ssim:>10.4f} {nafnet_ssim:>10.4f} {our_ssim:>10.4f} {gain_ssim_str:>12}")

        # Per-layer metrics
        print(f"\n📍 PER-LAYER PSNR (dB):")
        print(f"   {'Layer':<12} {'Input':>8} {'NAFNet':>8} {'Ours':>8} {'Gain':>8} {'Clinical Loss':>14}")
        print(f"   {'-'*62}")

        clinical_loss_names = ['rnfl_clinical', 'inl_clinical', 'onl_clinical', 'rpe_clinical']
        for i, name in enumerate(self.LAYER_NAMES):
            inp = val_metrics.get(f'{name}_input_psnr', 0)
            naf = val_metrics.get(f'{name}_nafnet_psnr', inp)
            ours = val_metrics.get(f'{name}_psnr', 0)
            gain = val_metrics.get(f'{name}_gain_vs_nafnet_psnr', ours - naf)
            clinical = train_metrics.get(clinical_loss_names[i], 0)

            gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
            print(f"   {name:<12} {inp:>8.2f} {naf:>8.2f} {ours:>8.2f} {gain_str:>8} {clinical:>14.4f}")

        # Per-layer SSIM
        print(f"\n📍 PER-LAYER SSIM:")
        print(f"   {'Layer':<12} {'Input':>8} {'NAFNet':>8} {'Ours':>8} {'Gain':>8}")
        print(f"   {'-'*48}")

        for name in self.LAYER_NAMES:
            inp = val_metrics.get(f'{name}_input_ssim', 0)
            naf = val_metrics.get(f'{name}_nafnet_ssim', inp)
            ours = val_metrics.get(f'{name}_ssim', 0)
            gain = val_metrics.get(f'{name}_gain_vs_nafnet_ssim', ours - naf)

            gain_str = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"
            print(f"   {name:<12} {inp:>8.4f} {naf:>8.4f} {ours:>8.4f} {gain_str:>8}")

        # Constraints
        print(f"\n🔒 CONSTRAINTS:")
        n_sat = val_metrics.get('n_constraints_satisfied', 0)
        print(f"   Satisfied: {n_sat}/5 | ρ={self.rho:.2f}")
        print(f"   {'Constraint':<12} {'Violation':>10} {'Status':>10} {'λ':>8}")
        print(f"   {'-'*42}")

        for name in self.CONSTRAINT_NAMES:
            viol = val_metrics.get(f'{name}_viol', 0)
            status = "✓ OK" if viol < 0.01 else f"✗ VIOL"
            lam = self.lambdas[name].item()
            print(f"   {name:<12} {viol:>10.4f} {status:>10} {lam:>8.2f}")

        print(f"{'='*80}\n")

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lambdas': {k: v.item() for k, v in self.lambdas.items()},
            'rho': self.rho,
            'metrics': metrics,
        }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Constrained Neuro-Symbolic OCT Denoiser')

    # Data
    parser.add_argument('--pku37_root', type=str,
                       default='/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising')
    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--max_val', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=128)

    # Model
    parser.add_argument('--encoder_channels', type=str, default='32,64,128')
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--head_width', type=int, default=24)

    # Baseline
    parser.add_argument('--nafnet_ckpt', type=str, default='outputs/nafnet_pku37/nafnet_best.pth')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    # Augmented Lagrangian
    parser.add_argument('--rho_init', type=float, default=0.1)
    parser.add_argument('--rho_max', type=float, default=10.0)
    parser.add_argument('--rho_mult', type=float, default=1.2)
    parser.add_argument('--disentangle_weight', type=float, default=0.05)

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs/constrained_denoiser')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("CONSTRAINED NEURO-SYMBOLIC OCT DENOISER - Training with Baseline Comparison")
    print("=" * 80)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    print("\n📂 Loading data...")
    train_dataset = PKU37DirectDataset(args.pku37_root, split='train', max_samples=args.max_train, patch_size=args.patch_size)
    val_dataset = PKU37DirectDataset(args.pku37_root, split='val', max_samples=args.max_val, patch_size=args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"   Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # NAFNet baseline
    print("\n🔧 Loading NAFNet baseline...")
    nafnet_baseline = NAFNetBaseline(args.nafnet_ckpt, device)

    # Model
    print("\n🏗️ Creating model...")
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
    print(f"   Parameters: {n_params:,}")

    # Trainer
    trainer = ConstrainedTrainerWithMonitoring(model, config, device, nafnet_baseline, lr=args.lr)

    # Initial validation
    print("\n📊 Initial validation...")
    init_metrics = trainer.validate(val_loader)
    trainer.print_epoch_summary(-1, {}, init_metrics, 0)

    # Training
    print("\n" + "=" * 80)
    print("🚀 TRAINING")
    print("=" * 80)

    best_psnr = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        epoch_time = time.time() - epoch_start

        # Validate
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = trainer.validate(val_loader)
            trainer.print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)

            # Save best
            if val_metrics['global_psnr'] > best_psnr:
                best_psnr = val_metrics['global_psnr']
                best_epoch = epoch + 1
                trainer.save_checkpoint(os.path.join(args.output_dir, 'best_model.pt'), epoch, val_metrics)
                print(f"   💾 Saved best model (PSNR: {best_psnr:.2f} dB)")

    # Final summary
    print("\n" + "=" * 80)
    print("🏁 TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest Model: Epoch {best_epoch} | PSNR: {best_psnr:.2f} dB")

    # Final validation
    final_metrics = trainer.validate(val_loader)
    trainer.print_epoch_summary(args.epochs - 1, train_metrics, final_metrics, 0)

    # Save final
    trainer.save_checkpoint(os.path.join(args.output_dir, 'final_model.pt'), args.epochs - 1, final_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
