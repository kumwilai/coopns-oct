#!/usr/bin/env python3
"""
Clinical Layer-Specific Denoiser

Architecture: Pure layer-specific (NO NAFNet backbone)
- Each layer has its own denoiser head
- Each layer has clinically-appropriate loss function
- Soft mask blending at boundaries

Goal: Maximize diagnostic value per layer for physicians
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


# =============================================================================
# Layer-Specific Loss Functions
# =============================================================================

class TexturePreservationLoss(nn.Module):
    """
    For RNFL: Preserve fine nerve fiber texture.
    Uses Laplacian (high-frequency) content preservation.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('laplacian', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_hf = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.laplacian)
        target_hf = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.laplacian)
        diff = torch.abs(pred_hf - target_hf) * mask
        return diff.sum() / (mask.sum() + 1e-8)


class StructuralSimilarityLoss(nn.Module):
    """
    For INL/OPL: Preserve structural patterns.
    """
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = g.outer(g)
        self.register_buffer('window', (window / window.sum()).view(1, 1, window_size, window_size))
        self.pad = window_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mu_p = F.conv2d(F.pad(pred, (self.pad,)*4, mode='replicate'), self.window)
        mu_t = F.conv2d(F.pad(target, (self.pad,)*4, mode='replicate'), self.window)

        sigma_p = F.conv2d(F.pad(pred**2, (self.pad,)*4, mode='replicate'), self.window) - mu_p**2
        sigma_t = F.conv2d(F.pad(target**2, (self.pad,)*4, mode='replicate'), self.window) - mu_t**2
        sigma_pt = F.conv2d(F.pad(pred*target, (self.pad,)*4, mode='replicate'), self.window) - mu_p*mu_t

        ssim = ((2*mu_p*mu_t + self.C1) * (2*sigma_pt + self.C2)) / \
               ((mu_p**2 + mu_t**2 + self.C1) * (sigma_p + sigma_t + self.C2))

        loss = (1 - ssim) * mask
        return loss.sum() / (mask.sum() + 1e-8)


class EdgeSharpnessLoss(nn.Module):
    """
    For ONL/IS: Preserve sharp photoreceptor junction.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_gx = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.sobel_x)
        pred_gy = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.sobel_y)
        target_gx = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.sobel_x)
        target_gy = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.sobel_y)

        pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
        target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8)

        diff = torch.abs(pred_grad - target_grad) * mask
        return diff.sum() / (mask.sum() + 1e-8)


class ContrastPreservationLoss(nn.Module):
    """
    For RPE: Preserve local contrast (critical for drusen detection).
    """
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.register_buffer('avg_kernel', torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2))
        self.pad = kernel_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_mean = F.conv2d(F.pad(pred, (self.pad,)*4, mode='replicate'), self.avg_kernel)
        target_mean = F.conv2d(F.pad(target, (self.pad,)*4, mode='replicate'), self.avg_kernel)

        pred_var = F.conv2d(F.pad((pred - pred_mean)**2, (self.pad,)*4, mode='replicate'), self.avg_kernel)
        target_var = F.conv2d(F.pad((target - target_mean)**2, (self.pad,)*4, mode='replicate'), self.avg_kernel)

        pred_std = torch.sqrt(pred_var + 1e-8)
        target_std = torch.sqrt(target_var + 1e-8)

        diff = torch.abs(pred_std - target_std) * mask
        return diff.sum() / (mask.sum() + 1e-8)


# =============================================================================
# Layer Denoiser Head
# =============================================================================

class LayerDenoiserHead(nn.Module):
    """
    Full denoiser for one layer.
    Capacity adjustable via width parameter.
    """
    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # Residual learning


# =============================================================================
# Boundary Predictor
# =============================================================================

class BoundaryPredictor(nn.Module):
    """
    Predicts 4 layer boundaries from input image.
    """
    def __init__(self, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
        )
        self.boundary_head = nn.Conv2d(width, 4, 1)

        # Initialize to reasonable default positions
        self.register_buffer('default_positions', torch.tensor([0.25, 0.40, 0.55, 0.70]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        features = self.encoder(x)
        offsets = self.boundary_head(features).mean(dim=2) * 0.1  # [B, 4, W]

        boundaries = self.default_positions.view(1, 4, 1) + offsets

        # Enforce ordering
        b0 = boundaries[:, 0:1, :]
        b1 = torch.maximum(boundaries[:, 1:2, :], b0 + 0.05)
        b2 = torch.maximum(boundaries[:, 2:3, :], b1 + 0.05)
        b3 = torch.maximum(boundaries[:, 3:4, :], b2 + 0.05)

        return torch.clamp(torch.cat([b0, b1, b2, b3], dim=1), 0.05, 0.95)


# =============================================================================
# Main Clinical Denoiser
# =============================================================================

class ClinicalLayerDenoiser(nn.Module):
    """
    Pure layer-specific denoiser with clinical losses.

    NO NAFNet backbone - each layer head is a full denoiser.
    """

    LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL_IS', 'RPE_Choroid']

    def __init__(
        self,
        head_width: int = 32,
        boundary_width: int = 32,
        blend_sigma: float = 5.0,
        clinical_loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.blend_sigma = blend_sigma

        # Boundary predictor
        self.boundary_predictor = BoundaryPredictor(width=boundary_width)

        # 4 layer-specific denoiser heads
        self.layer_heads = nn.ModuleList([
            LayerDenoiserHead(width=head_width) for _ in range(4)
        ])

        # Layer-specific clinical losses
        self.clinical_losses = nn.ModuleDict({
            'rnfl': TexturePreservationLoss(),
            'inl': StructuralSimilarityLoss(),
            'onl': EdgeSharpnessLoss(),
            'rpe': ContrastPreservationLoss(),
        })

        # Clinical importance weights
        self.clinical_weights = clinical_loss_weights or {
            'rnfl': 0.3,  # Texture preservation weight
            'inl': 0.3,   # Structure preservation weight
            'onl': 0.3,   # Edge sharpness weight
            'rpe': 0.3,   # Contrast preservation weight
        }

        # Layer importance for diagnosis (can be tuned)
        self.layer_importance = nn.Parameter(
            torch.tensor([1.5, 1.0, 1.2, 1.5]),  # RNFL and RPE more important
            requires_grad=False
        )

    def create_soft_masks(self, boundaries: torch.Tensor, H: int) -> torch.Tensor:
        """Create soft segmentation masks from boundaries."""
        B, N, W = boundaries.shape
        device = boundaries.device

        boundaries_px = boundaries * (H - 1)
        y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        b_exp = boundaries_px.unsqueeze(2)  # [B, 4, 1, W]

        temp = self.blend_sigma

        # Soft masks with sigmoid transitions
        mask_0 = torch.sigmoid((b_exp[:, 1:2] - y) / temp)
        mask_1 = torch.sigmoid((y - b_exp[:, 1:2]) / temp) * torch.sigmoid((b_exp[:, 2:3] - y) / temp)
        mask_2 = torch.sigmoid((y - b_exp[:, 2:3]) / temp) * torch.sigmoid((b_exp[:, 3:4] - y) / temp)
        mask_3 = torch.sigmoid((y - b_exp[:, 3:4]) / temp)

        soft_masks = torch.cat([mask_0, mask_1, mask_2, mask_3], dim=1)
        return soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            noisy: Noisy input [B, 1, H, W]

        Returns:
            Dict with denoised output and intermediate results
        """
        B, C, H, W = noisy.shape

        # Predict boundaries
        boundaries = self.boundary_predictor(noisy)

        # Create soft masks
        soft_masks = self.create_soft_masks(boundaries, H)

        # Process each layer with its specialized head
        layer_outputs = []
        for i, head in enumerate(self.layer_heads):
            layer_out = head(noisy)
            layer_outputs.append(layer_out)

        # Stack and blend
        layer_stack = torch.cat(layer_outputs, dim=1)  # [B, 4, H, W]

        # Weighted blend with soft masks
        denoised = (layer_stack * soft_masks).sum(dim=1, keepdim=True)

        return {
            'denoised': denoised,
            'boundaries': boundaries,
            'soft_masks': soft_masks,
            'layer_outputs': layer_stack,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        noisy: torch.Tensor,
        v4_loss_fn=None,
        v4_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with layer-specific clinical losses.
        """
        losses = {}
        device = clean.device

        denoised = outputs['denoised']
        boundaries = outputs['boundaries']
        soft_masks = outputs['soft_masks']
        layer_outputs = outputs['layer_outputs']

        # === 1. GLOBAL RECONSTRUCTION LOSS ===
        global_l1 = F.l1_loss(denoised, clean)
        losses['global_l1'] = global_l1.item()

        total_loss = global_l1

        # === 2. PER-LAYER L1 LOSS ===
        per_layer_l1 = torch.zeros(1, device=device)
        for i in range(4):
            mask = soft_masks[:, i:i+1, :, :]
            layer_out = layer_outputs[:, i:i+1, :, :]
            layer_l1 = (torch.abs(layer_out - clean) * mask).sum() / (mask.sum() + 1e-8)
            per_layer_l1 = per_layer_l1 + self.layer_importance[i] * layer_l1
            losses[f'layer{i}_l1'] = layer_l1.item()

        per_layer_l1 = per_layer_l1 / 4
        losses['per_layer_l1'] = per_layer_l1.item()
        total_loss = total_loss + per_layer_l1

        # === 3. CLINICAL LAYER-SPECIFIC LOSSES ===
        clinical_loss_names = ['rnfl', 'inl', 'onl', 'rpe']
        clinical_total = torch.zeros(1, device=device)

        for i, name in enumerate(clinical_loss_names):
            mask = soft_masks[:, i:i+1, :, :]
            layer_out = layer_outputs[:, i:i+1, :, :]

            clinical_loss = self.clinical_losses[name](layer_out, clean, mask)
            weighted_loss = self.clinical_weights[name] * clinical_loss

            clinical_total = clinical_total + self.layer_importance[i] * weighted_loss
            losses[f'{name}_clinical'] = clinical_loss.item()

        clinical_total = clinical_total / 4
        losses['clinical_total'] = clinical_total.item()
        total_loss = total_loss + clinical_total

        # === 4. V4 BOUNDARY ANCHORING (if provided) ===
        if v4_loss_fn is not None:
            v4_loss, v4_details = v4_loss_fn(boundaries, noisy)
            total_loss = total_loss + v4_weight * v4_loss
            losses['v4_anchor'] = v4_loss.item()
            losses['detected_ilm'] = v4_details.get('detected_retina_top', 0.0)
            losses['detected_rpe'] = v4_details.get('detected_retina_bottom', 0.0)

        losses['total'] = total_loss.item()

        return total_loss, losses

    def compute_layer_psnr(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute per-layer PSNR for evaluation."""
        soft_masks = outputs['soft_masks']
        denoised = outputs['denoised']

        psnrs = {}

        # Global PSNR
        mse = F.mse_loss(denoised, clean).item()
        psnrs['global'] = 10 * np.log10(1.0 / max(mse, 1e-10))

        # Per-layer PSNR
        for i, name in enumerate(self.LAYER_NAMES):
            mask = soft_masks[:, i:i+1, :, :]
            mask_sum = mask.sum().clamp(min=1.0)
            layer_mse = ((denoised - clean) ** 2 * mask).sum() / mask_sum
            psnrs[name] = 10 * np.log10(1.0 / max(layer_mse.item(), 1e-10))

        psnrs['avg_layer'] = np.mean([psnrs[n] for n in self.LAYER_NAMES])

        return psnrs


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Clinical Layer-Specific Denoiser")
    print("=" * 60)

    B, H, W = 2, 64, 64

    noisy = torch.rand(B, 1, H, W)
    clean = torch.rand(B, 1, H, W)

    model = ClinicalLayerDenoiser(head_width=32)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Boundary predictor: {sum(p.numel() for p in model.boundary_predictor.parameters()):,}")
    print(f"  - Layer heads (x4): {sum(p.numel() for p in model.layer_heads.parameters()):,}")

    # Forward
    outputs = model(noisy)
    print(f"\nOutputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Loss
    total_loss, losses = model.compute_loss(outputs, clean, noisy)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")

    # PSNR
    psnrs = model.compute_layer_psnr(outputs, clean)
    print(f"\nPSNR:")
    for k, v in psnrs.items():
        print(f"  {k}: {v:.2f} dB")

    # Backward
    total_loss.backward()
    print("\nBackward pass successful!")

    print("\n" + "=" * 60)
    print("Architecture Summary:")
    print("  - NO NAFNet backbone")
    print("  - 4 independent layer denoiser heads")
    print("  - Layer-specific clinical losses:")
    print("    * RNFL: Texture preservation (nerve fibers)")
    print("    * INL:  Structural similarity (layer patterns)")
    print("    * ONL:  Edge sharpness (IS/OS junction)")
    print("    * RPE:  Contrast preservation (drusen)")
