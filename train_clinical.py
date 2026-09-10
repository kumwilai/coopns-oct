#!/usr/bin/env python3
"""
Train Clinical Neuro-Symbolic OCT Denoising

Focus: CLINICAL UTILITY over PSNR

Key Metrics:
- Contrast Preservation (target: 85%)
- Boundary Sharpness (target: 85%)
- Texture Preservation (target: 75%)
- Edge Strength (target: 85%)

Based on backbone analysis showing:
- Contrast: 47% preserved
- Boundary: 47% preserved
- Texture: 41% preserved
- Edge: 68% preserved
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from clinical_correctors import ClinicalNeuroSymbolicCorrector


# =============================================================================
# CLINICAL LOSS FUNCTION
# =============================================================================

class ClinicalLoss(nn.Module):
    """
    Loss function optimizing for clinical utility.

    Components:
    1. Contrast preservation loss
    2. Boundary sharpness loss
    3. Texture preservation loss
    4. Edge preservation loss
    5. Minimal reconstruction loss (low weight)

    The key insight: We care about RELATIVE improvement on clinical metrics,
    not absolute PSNR values.
    """

    def __init__(self,
                 lambda_contrast: float = 1.0,
                 lambda_boundary: float = 1.0,
                 lambda_texture: float = 0.8,
                 lambda_edge: float = 1.0,
                 lambda_recon: float = 0.3):  # Low weight on PSNR
        super().__init__()

        self.lambda_contrast = lambda_contrast
        self.lambda_boundary = lambda_boundary
        self.lambda_texture = lambda_texture
        self.lambda_edge = lambda_edge
        self.lambda_recon = lambda_recon

        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)

    def contrast_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Loss for contrast preservation.

        Penalizes when predicted image has lower local contrast than target.
        """
        pred_contrast = self._local_std(pred)
        target_contrast = self._local_std(target)

        # We want pred_contrast >= target_contrast
        # Penalize where contrast is reduced
        contrast_diff = (target_contrast - pred_contrast).clamp(min=0)

        return contrast_diff.mean()

    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Loss for boundary sharpness.

        Penalizes blurred layer boundaries (vertical gradients).
        """
        pred_grad = self._vertical_gradient(pred)
        target_grad = self._vertical_gradient(target)

        # Focus on strong boundaries in target (use threshold for stability)
        grad_threshold = target_grad.mean() + target_grad.std()
        boundary_mask = target_grad > grad_threshold

        # Penalize where gradients are weaker in pred
        if boundary_mask.sum() > 10:
            grad_diff = (target_grad - pred_grad).clamp(min=0)
            loss = (grad_diff * boundary_mask.float()).sum() / (boundary_mask.sum() + 1)
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss

    def texture_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Loss for texture preservation.

        Penalizes over-smoothing in textured regions.
        """
        pred_var = self._local_variance(pred, size=5)
        target_var = self._local_variance(target, size=5)

        # Texture regions: moderate variance
        texture_mask = (target_var > 0.001) & (target_var < 0.05)

        # Penalize where variance is reduced
        if texture_mask.sum() > 10:
            var_diff = (target_var - pred_var).clamp(min=0)
            loss = (var_diff * texture_mask.float()).sum() / (texture_mask.sum() + 1)
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss

    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Loss for edge preservation.

        Penalizes weakened edges.
        """
        pred_edge = self._edge_magnitude(pred)
        target_edge = self._edge_magnitude(target)

        # Focus on strong edges (use threshold for stability)
        edge_threshold = target_edge.mean() + 0.5 * target_edge.std()
        edge_mask = target_edge > edge_threshold

        # Penalize where edges are weaker
        if edge_mask.sum() > 10:
            edge_diff = (target_edge - pred_edge).clamp(min=0)
            loss = (edge_diff * edge_mask.float()).sum() / (edge_mask.sum() + 1)
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss

    def forward(self, corrected: torch.Tensor, backbone_out: torch.Tensor,
                target: torch.Tensor, info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute clinical loss.

        Args:
            corrected: Corrector output [B, 1, H, W]
            backbone_out: Backbone output [B, 1, H, W]
            target: Ground truth [B, 1, H, W]
            info: Corrector info dict

        Returns:
            total_loss: Scalar loss
            metrics: Dict with loss components and clinical metrics
        """
        # Clinical losses on corrected output
        l_contrast = self.contrast_loss(corrected, target)
        l_boundary = self.boundary_loss(corrected, target)
        l_texture = self.texture_loss(corrected, target)
        l_edge = self.edge_loss(corrected, target)

        # Minimal reconstruction loss
        l_recon = F.mse_loss(corrected, target)

        # Total loss
        total = (self.lambda_contrast * l_contrast +
                 self.lambda_boundary * l_boundary +
                 self.lambda_texture * l_texture +
                 self.lambda_edge * l_edge +
                 self.lambda_recon * l_recon)

        # Compute clinical metrics (for monitoring)
        with torch.no_grad():
            # PSNR (for reference, not optimization target)
            mse_backbone = F.mse_loss(backbone_out, target)
            mse_corrected = F.mse_loss(corrected, target)
            psnr_backbone = 10 * torch.log10(1.0 / mse_backbone.clamp(min=1e-10))
            psnr_corrected = 10 * torch.log10(1.0 / mse_corrected.clamp(min=1e-10))

            # Clinical metrics
            contrast_backbone = self._compute_contrast_preservation(backbone_out, target)
            contrast_corrected = self._compute_contrast_preservation(corrected, target)

            boundary_backbone = self._compute_boundary_preservation(backbone_out, target)
            boundary_corrected = self._compute_boundary_preservation(corrected, target)

            texture_backbone = self._compute_texture_preservation(backbone_out, target)
            texture_corrected = self._compute_texture_preservation(corrected, target)

            edge_backbone = self._compute_edge_preservation(backbone_out, target)
            edge_corrected = self._compute_edge_preservation(corrected, target)

        metrics = {
            'total': total.item(),
            'contrast_loss': l_contrast.item(),
            'boundary_loss': l_boundary.item(),
            'texture_loss': l_texture.item(),
            'edge_loss': l_edge.item(),
            'recon_loss': l_recon.item(),
            # Reference metrics (not optimization target)
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            # Clinical metrics
            'contrast_backbone': contrast_backbone,
            'contrast_corrected': contrast_corrected,
            'boundary_backbone': boundary_backbone,
            'boundary_corrected': boundary_corrected,
            'texture_backbone': texture_backbone,
            'texture_corrected': texture_corrected,
            'edge_backbone': edge_backbone,
            'edge_corrected': edge_corrected,
        }

        return total, metrics

    def _local_std(self, img: torch.Tensor, size: int = 15) -> torch.Tensor:
        padding = size // 2
        kernel = torch.ones(1, 1, size, size, device=img.device, dtype=img.dtype) / (size**2)
        mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
        mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)
        var = (mean_sq - mean**2).clamp(min=0)
        return torch.sqrt(var + 1e-8)

    def _local_variance(self, img: torch.Tensor, size: int = 7) -> torch.Tensor:
        padding = size // 2
        kernel = torch.ones(1, 1, size, size, device=img.device, dtype=img.dtype) / (size**2)
        mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
        mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)
        return (mean_sq - mean**2).clamp(min=0)

    def _vertical_gradient(self, img: torch.Tensor) -> torch.Tensor:
        pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
        return F.conv2d(pad, self.sobel_y.to(img.device)).abs()

    def _edge_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
        gx = F.conv2d(pad, self.sobel_x.to(img.device))
        gy = F.conv2d(pad, self.sobel_y.to(img.device))
        return torch.sqrt(gx**2 + gy**2)

    def _compute_contrast_preservation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_c = self._local_std(pred)
        target_c = self._local_std(target)
        ratio = pred_c / (target_c + 1e-6)
        return ratio.mean().item()

    def _compute_boundary_preservation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_g = self._vertical_gradient(pred)
        target_g = self._vertical_gradient(target)
        threshold = target_g.mean() + target_g.std()
        mask = target_g > threshold
        if mask.sum() > 10:
            ratio = (pred_g[mask].mean() / (target_g[mask].mean() + 1e-6)).clamp(0, 2)
            return ratio.item()
        return 0.5

    def _compute_texture_preservation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_v = self._local_variance(pred, size=5)
        target_v = self._local_variance(target, size=5)
        mask = (target_v > 0.001) & (target_v < 0.05)
        if mask.sum() > 10:
            ratio = (pred_v[mask].mean() / (target_v[mask].mean() + 1e-6)).clamp(0, 2)
            return ratio.item()
        return 0.5

    def _compute_edge_preservation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_e = self._edge_magnitude(pred)
        target_e = self._edge_magnitude(target)
        threshold = target_e.mean() + 0.5 * target_e.std()
        mask = target_e > threshold
        if mask.sum() > 10:
            ratio = (pred_e[mask].mean() / (target_e[mask].mean() + 1e-6)).clamp(0, 2)
            return ratio.item()
        return 0.5


# =============================================================================
# DATASET
# =============================================================================

class PKU37Dataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int = None,
                 patch_size: int = 128, is_train: bool = True):
        self.samples = []
        self.patch_size = patch_size
        self.is_train = is_train

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'clean_path' in entry and 'noisy_path' in entry:
                    if os.path.exists(entry['clean_path']) and os.path.exists(entry['noisy_path']):
                        self.samples.append({
                            'clean': entry['clean_path'],
                            'noisy': entry['noisy_path'],
                        })

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        clean = np.array(Image.open(sample['clean'])).astype(np.float32)
        noisy = np.array(Image.open(sample['noisy'])).astype(np.float32)

        if clean.max() > 1.0:
            clean = clean / 255.0
        if noisy.max() > 1.0:
            noisy = noisy / 255.0

        H, W = clean.shape

        if self.is_train and self.patch_size < min(H, W):
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            clean = clean[top:top+self.patch_size, left:left+self.patch_size]
            noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]

        clean = torch.from_numpy(clean).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        return {'clean': clean, 'noisy': noisy}


# =============================================================================
# BACKBONE WRAPPER
# =============================================================================

class BackboneWrapper(nn.Module):
    def __init__(self, width: int = 40):
        super().__init__()
        from nsnd.models.nafnet import NAFNetSmall
        self.backbone = NAFNetSmall(img_channel=1, width=width)

    def load_pretrained(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        self.backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded backbone from {path}")
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).clamp(0, 1)


# =============================================================================
# FULL MODEL
# =============================================================================

class ClinicalDenoiser(nn.Module):
    def __init__(self, backbone_width: int = 40, hidden_dim: int = 64):
        super().__init__()
        self.backbone = BackboneWrapper(backbone_width)
        self.corrector = ClinicalNeuroSymbolicCorrector(hidden_dim)

    def forward(self, noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        backbone_out = self.backbone(noisy)
        corrected, info = self.corrector(backbone_out, noisy)
        return corrected, backbone_out, info


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    model.backbone.eval()  # Keep backbone frozen

    metrics_sum = {}
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        corrected, backbone_out, info = model(noisy)
        loss, metrics = criterion(corrected, backbone_out, clean, info)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v
        n += 1

        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'contrast': f"{metrics['contrast_corrected']:.2f}",
            'boundary': f"{metrics['boundary_corrected']:.2f}",
        })

    return {k: v / n for k, v in metrics_sum.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    metrics_sum = {}
    n = 0

    for batch in tqdm(loader, desc="Validation"):
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        corrected, backbone_out, info = model(noisy)
        _, metrics = criterion(corrected, backbone_out, clean, info)

        for k, v in metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v
        n += 1

    return {k: v / n for k, v in metrics_sum.items()}


def print_clinical_metrics(epoch, train_metrics, val_metrics):
    """Print clinical metrics comparison."""
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch} - CLINICAL METRICS")
    print(f"{'='*80}")

    print(f"\n{'METRIC':<20} {'BACKBONE':>12} {'CORRECTED':>12} {'DELTA':>12} {'TARGET':>10}")
    print("-" * 70)

    metrics_info = [
        ('Contrast', 'contrast', 0.85),
        ('Boundary', 'boundary', 0.85),
        ('Texture', 'texture', 0.75),
        ('Edge', 'edge', 0.85),
    ]

    total_improvement = 0
    for name, key, target in metrics_info:
        backbone = val_metrics[f'{key}_backbone']
        corrected = val_metrics[f'{key}_corrected']
        delta = corrected - backbone
        total_improvement += delta

        status = "✓" if corrected >= target else "✗"
        print(f"{name:<20} {backbone:>12.1%} {corrected:>12.1%} {delta:>+12.1%} {target:>9.0%} {status}")

    print("-" * 70)
    print(f"{'AVERAGE IMPROVEMENT':<20} {'':<12} {'':<12} {total_improvement/4:>+12.1%}")

    # PSNR for reference
    print(f"\n{'PSNR (reference)':<20} {val_metrics['psnr_backbone']:>12.2f} {val_metrics['psnr_corrected']:>12.2f} {val_metrics['psnr_corrected']-val_metrics['psnr_backbone']:>+12.3f}")

    print(f"\nLoss: {val_metrics['total']:.4f}")
    print(f"{'='*80}\n")

    return total_improvement / 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='pku37_train.jsonl')
    parser.add_argument('--val_jsonl', default='pku37_val.jsonl')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--backbone_path', default='outputs/nafnet_pku37_w40/best_model.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='outputs/clinical_corrector')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("CLINICAL NEURO-SYMBOLIC OCT DENOISING")
    print("Focus: Clinical Utility (Contrast, Boundary, Texture, Edge)")
    print("=" * 80)

    # Create datasets
    train_dataset = PKU37Dataset(args.train_jsonl, args.max_train, patch_size=128, is_train=True)
    val_dataset = PKU37Dataset(args.val_jsonl, args.max_val, patch_size=256, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create model
    model = ClinicalDenoiser(backbone_width=40, hidden_dim=64)
    model.backbone.load_pretrained(args.backbone_path)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    model = model.to(args.device)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable:,}")

    # Loss and optimizer
    criterion = ClinicalLoss(
        lambda_contrast=1.0,
        lambda_boundary=1.0,
        lambda_texture=0.8,
        lambda_edge=1.0,
        lambda_recon=0.3,  # Low weight on PSNR
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_improvement = -float('inf')

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch)
        val_metrics = validate(model, val_loader, criterion, args.device)

        avg_improvement = print_clinical_metrics(epoch, train_metrics, val_metrics)

        scheduler.step()

        # Save best model (based on clinical improvement, not PSNR)
        if avg_improvement > best_improvement:
            best_improvement = avg_improvement
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'clinical_improvement': avg_improvement,
                'val_metrics': val_metrics,
            }, os.path.join(args.output_dir, 'best_clinical_model.pth'))
            print(f"*** New best clinical model! Improvement: {avg_improvement:+.1%} ***")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best clinical improvement: {best_improvement:+.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
