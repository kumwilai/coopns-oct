#!/usr/bin/env python3
"""
Layer-Specific Clinical Denoising

Goal: Maximize diagnostic value per layer, not just global PSNR.

Key insight: Different retinal layers need different denoising strategies:
- RNFL: Preserve fine texture (nerve fibers)
- INL/OPL: Preserve structure and boundaries
- ONL/IS: Sharp photoreceptor junction
- RPE: High contrast, edge preservation

Architecture:
- NAFNet provides base denoising
- Layer-specific heads REPLACE (not refine) NAFNet output per region
- Each head optimized with layer-appropriate loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


# =============================================================================
# Layer-Specific Loss Functions
# =============================================================================

class TexturePreservationLoss(nn.Module):
    """
    For RNFL: Preserve fine texture patterns (nerve fiber striations).
    Uses high-frequency content preservation.
    """
    def __init__(self):
        super().__init__()
        # Laplacian kernel for high-frequency detection
        self.register_buffer('laplacian', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Extract high-frequency content
        pred_hf = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.laplacian)
        target_hf = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.laplacian)

        # Compare high-frequency content in masked region
        diff = torch.abs(pred_hf - target_hf) * mask
        return diff.sum() / (mask.sum() + 1e-8)


class EdgeSharpnessLoss(nn.Module):
    """
    For RPE and boundaries: Preserve sharp edges.
    Penalizes blurring of edges.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Compute gradients
        pred_gx = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.sobel_x)
        pred_gy = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.sobel_y)
        target_gx = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.sobel_x)
        target_gy = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.sobel_y)

        pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
        target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8)

        # Penalize gradient magnitude difference (edge sharpness)
        diff = torch.abs(pred_grad - target_grad) * mask
        return diff.sum() / (mask.sum() + 1e-8)


class StructuralSimilarityLoss(nn.Module):
    """
    For INL/OPL: Preserve structural patterns.
    Local SSIM computation.
    """
    def __init__(self, window_size: int = 7):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = g.outer(g)
        window = window / window.sum()
        self.register_buffer('window', window.view(1, 1, window_size, window_size))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2

        mu_pred = F.conv2d(F.pad(pred, (pad,pad,pad,pad), mode='replicate'), self.window)
        mu_target = F.conv2d(F.pad(target, (pad,pad,pad,pad), mode='replicate'), self.window)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(F.pad(pred**2, (pad,pad,pad,pad), mode='replicate'), self.window) - mu_pred_sq
        sigma_target_sq = F.conv2d(F.pad(target**2, (pad,pad,pad,pad), mode='replicate'), self.window) - mu_target_sq
        sigma_pred_target = F.conv2d(F.pad(pred*target, (pad,pad,pad,pad), mode='replicate'), self.window) - mu_pred_target

        ssim = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
               ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        # Return 1 - SSIM as loss, masked
        loss = (1 - ssim) * mask
        return loss.sum() / (mask.sum() + 1e-8)


class ContrastPreservationLoss(nn.Module):
    """
    For RPE: Preserve local contrast (important for drusen detection).
    """
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.register_buffer('avg_kernel', torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2

        # Local mean
        pred_mean = F.conv2d(F.pad(pred, (pad,pad,pad,pad), mode='replicate'), self.avg_kernel)
        target_mean = F.conv2d(F.pad(target, (pad,pad,pad,pad), mode='replicate'), self.avg_kernel)

        # Local contrast (std approximation)
        pred_contrast = torch.sqrt(F.conv2d(F.pad((pred - pred_mean)**2, (pad,pad,pad,pad), mode='replicate'), self.avg_kernel) + 1e-8)
        target_contrast = torch.sqrt(F.conv2d(F.pad((target - target_mean)**2, (pad,pad,pad,pad), mode='replicate'), self.avg_kernel) + 1e-8)

        # Penalize contrast difference
        diff = torch.abs(pred_contrast - target_contrast) * mask
        return diff.sum() / (mask.sum() + 1e-8)


# =============================================================================
# Layer-Specific Denoiser Head
# =============================================================================

class LayerDenoiserHead(nn.Module):
    """
    Full denoiser for a specific layer (not just refinement).
    Can REPLACE NAFNet output in its region.
    """
    def __init__(self, in_channels: int = 1, width: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

        # Confidence predictor: how much to trust this head vs NAFNet
        self.confidence = nn.Sequential(
            nn.Conv2d(width * 2, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            denoised: Layer-specific denoised output
            confidence: How much to trust this output vs NAFNet
        """
        features = self.encoder(x)
        denoised = x + self.decoder(features)  # Residual learning
        confidence = self.confidence(features)
        return denoised, confidence


# =============================================================================
# Clinical Layer-Specific Denoising System
# =============================================================================

class ClinicalLayerDenoiser(nn.Module):
    """
    Layer-specific denoising optimized for clinical diagnosis.

    Key features:
    1. Each layer has its own denoiser head
    2. Each layer uses clinically-appropriate loss
    3. Confidence-weighted blending with NAFNet
    4. V4 ensures correct layer boundaries
    """

    def __init__(self, nafnet_base=None, width: int = 32):
        super().__init__()

        self.nafnet_base = nafnet_base  # Pre-trained NAFNet (frozen)

        # Layer-specific denoiser heads
        self.layer_heads = nn.ModuleList([
            LayerDenoiserHead(in_channels=1, width=width)
            for _ in range(4)
        ])

        # Layer-specific loss functions
        self.layer_losses = nn.ModuleDict({
            'rnfl': TexturePreservationLoss(),      # Layer 0
            'inl_opl': StructuralSimilarityLoss(),  # Layer 1
            'onl_is': EdgeSharpnessLoss(),          # Layer 2
            'rpe': ContrastPreservationLoss(),      # Layer 3
        })

        # Clinical importance weights (can be tuned)
        self.clinical_weights = nn.Parameter(torch.tensor([1.5, 1.0, 1.2, 1.5]))

    def forward(
        self,
        noisy: torch.Tensor,
        soft_masks: torch.Tensor,
        nafnet_output: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy: Noisy input [B, 1, H, W]
            soft_masks: Layer masks [B, 4, H, W] (from V4-anchored boundaries)
            nafnet_output: Optional pre-computed NAFNet output
        """
        B, C, H, W = noisy.shape

        # Get NAFNet base if available
        if nafnet_output is None and self.nafnet_base is not None:
            with torch.no_grad():
                nafnet_output = self.nafnet_base(noisy)

        # Process each layer with its specialized head
        layer_outputs = []
        layer_confidences = []

        for i, head in enumerate(self.layer_heads):
            layer_denoised, confidence = head(noisy)
            layer_outputs.append(layer_denoised)
            layer_confidences.append(confidence)

        layer_stack = torch.stack(layer_outputs, dim=1)  # [B, 4, 1, H, W]
        confidence_stack = torch.stack(layer_confidences, dim=1)  # [B, 4, 1, H, W]

        # Blend layer outputs with soft masks
        layer_stack = layer_stack.squeeze(2)  # [B, 4, H, W]
        confidence_stack = confidence_stack.squeeze(2)  # [B, 4, H, W]

        # If NAFNet available, blend based on confidence
        if nafnet_output is not None:
            # Per-layer blending: confidence determines NAFNet vs layer head
            blended_layers = []
            for i in range(4):
                conf = confidence_stack[:, i:i+1, :, :]  # [B, 1, H, W]
                layer_out = layer_stack[:, i:i+1, :, :]  # [B, 1, H, W]
                # High confidence → use layer head, low confidence → use NAFNet
                blended = conf * layer_out + (1 - conf) * nafnet_output
                blended_layers.append(blended)

            blended_stack = torch.cat(blended_layers, dim=1)  # [B, 4, H, W]

            # Final output: weighted sum by soft masks
            denoised = (blended_stack * soft_masks).sum(dim=1, keepdim=True)
        else:
            # No NAFNet, just use layer heads
            denoised = (layer_stack * soft_masks).sum(dim=1, keepdim=True)

        return {
            'denoised': denoised,
            'layer_outputs': layer_stack,
            'layer_confidences': confidence_stack,
            'nafnet_output': nafnet_output,
        }

    def compute_clinical_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        soft_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute layer-specific clinical losses.
        """
        losses = {}
        layer_names = ['rnfl', 'inl_opl', 'onl_is', 'rpe']
        loss_fns = [
            self.layer_losses['rnfl'],
            self.layer_losses['inl_opl'],
            self.layer_losses['onl_is'],
            self.layer_losses['rpe'],
        ]

        layer_outputs = outputs['layer_outputs']  # [B, 4, H, W]

        total_clinical_loss = 0

        for i, (name, loss_fn) in enumerate(zip(layer_names, loss_fns)):
            mask = soft_masks[:, i:i+1, :, :]
            layer_out = layer_outputs[:, i:i+1, :, :]

            # L1 base loss
            l1_loss = (torch.abs(layer_out - clean) * mask).sum() / (mask.sum() + 1e-8)

            # Clinical-specific loss
            clinical_loss = loss_fn(layer_out, clean, mask)

            # Combined with clinical weight
            weight = self.clinical_weights[i]
            layer_loss = l1_loss + 0.5 * clinical_loss

            losses[f'{name}_l1'] = l1_loss.item()
            losses[f'{name}_clinical'] = clinical_loss.item()
            losses[f'{name}_total'] = layer_loss.item()

            total_clinical_loss = total_clinical_loss + weight * layer_loss

        # Global reconstruction loss
        global_l1 = F.l1_loss(outputs['denoised'], clean)
        losses['global_l1'] = global_l1.item()

        # Total loss
        total = global_l1 + 0.5 * total_clinical_loss
        losses['total'] = total.item()

        return total, losses


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Clinical Layer-Specific Denoising")
    print("=" * 60)

    B, H, W = 2, 128, 128

    # Create test data
    noisy = torch.rand(B, 1, H, W)
    clean = torch.rand(B, 1, H, W)

    # Create soft masks (simulate 4 layers)
    soft_masks = torch.zeros(B, 4, H, W)
    soft_masks[:, 0, :32, :] = 1.0    # RNFL
    soft_masks[:, 1, 32:64, :] = 1.0  # INL/OPL
    soft_masks[:, 2, 64:96, :] = 1.0  # ONL/IS
    soft_masks[:, 3, 96:, :] = 1.0    # RPE

    # Create model
    model = ClinicalLayerDenoiser(nafnet_base=None, width=32)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    outputs = model(noisy, soft_masks)
    print(f"\nOutputs:")
    print(f"  denoised: {outputs['denoised'].shape}")
    print(f"  layer_outputs: {outputs['layer_outputs'].shape}")
    print(f"  layer_confidences: {outputs['layer_confidences'].shape}")

    # Compute loss
    total_loss, losses = model.compute_clinical_loss(outputs, clean, soft_masks)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")

    # Backward pass
    total_loss.backward()
    print("\nBackward pass successful!")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("Each layer now has its own denoiser with clinically-appropriate loss:")
    print("  - RNFL: Texture preservation (nerve fiber visibility)")
    print("  - INL/OPL: Structural similarity (layer structure)")
    print("  - ONL/IS: Edge sharpness (photoreceptor junction)")
    print("  - RPE: Contrast preservation (drusen detection)")
