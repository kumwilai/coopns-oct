#!/usr/bin/env python3
"""
Hybrid Constrained Neuro-Symbolic Denoiser with NAFNet Backbone.

Strategy:
1. Use NAFNet for base denoising (frozen or fine-tuned)
2. Add layer-specific refinement heads
3. Apply anatomical constraints via augmented Lagrangian

This combines NAFNet's denoising power with our novel contributions:
- Disentangled layer representations
- Anatomical constraint satisfaction
- Layer-specific clinical losses
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

sys.path.insert(0, '/home/kumwilai/OCT')

# Import NAFNet
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False
    print("Warning: NAFNet not available")


# =============================================================================
# Layer-Specific Refinement Head
# =============================================================================

class LayerRefinementHead(nn.Module):
    """
    Refines NAFNet output for a specific layer.
    Learns layer-specific details that NAFNet might miss.
    """

    def __init__(self, width: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
            nn.Tanh(),  # Bounded refinement
        )

        # Learnable refinement scale (starts small)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns refinement to be added to NAFNet output."""
        refinement = self.net(x) * self.scale
        return refinement


# =============================================================================
# Boundary Predictor
# =============================================================================

class BoundaryPredictor(nn.Module):
    """Predicts layer boundaries from image."""

    def __init__(self, width: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
        )

        self.boundary_head = nn.Conv2d(width * 2, 4, 1)
        self.register_buffer('default_positions', torch.tensor([0.20, 0.40, 0.60, 0.80]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        features = self.encoder(x)
        offsets = self.boundary_head(features).mean(dim=2) * 0.1

        # Upsample to original width
        offsets = F.interpolate(offsets.unsqueeze(2), size=(1, W), mode='bilinear', align_corners=False).squeeze(2)

        boundaries = self.default_positions.view(1, 4, 1) + offsets

        # Enforce ordering
        b0 = boundaries[:, 0:1, :]
        b1 = torch.maximum(boundaries[:, 1:2, :], b0 + 0.05)
        b2 = torch.maximum(boundaries[:, 2:3, :], b1 + 0.05)
        b3 = torch.maximum(boundaries[:, 3:4, :], b2 + 0.05)

        return torch.clamp(torch.cat([b0, b1, b2, b3], dim=1), 0.05, 0.95)


# =============================================================================
# Symbolic Constraints
# =============================================================================

class SymbolicConstraints(nn.Module):
    """Anatomical constraints for boundaries and layer properties."""

    def __init__(self):
        super().__init__()
        self.min_sep = 0.03

        # Thickness bounds
        self.register_buffer('thickness_min', torch.tensor([0.02, 0.02, 0.02]))
        self.register_buffer('thickness_max', torch.tensor([0.35, 0.30, 0.35]))

    def ordering_constraint(self, boundaries: torch.Tensor) -> torch.Tensor:
        B, N, W = boundaries.shape
        violations = []
        for i in range(N - 1):
            gap_viol = F.relu(boundaries[:, i] + self.min_sep - boundaries[:, i + 1])
            violations.append(gap_viol)
        return torch.stack(violations, dim=1).sum(dim=(1, 2))

    def thickness_constraint(self, boundaries: torch.Tensor) -> torch.Tensor:
        thicknesses = boundaries[:, 1:] - boundaries[:, :-1]
        violations = []
        for i in range(min(3, thicknesses.shape[1])):
            lower_viol = F.relu(self.thickness_min[i] - thicknesses[:, i])
            upper_viol = F.relu(thicknesses[:, i] - self.thickness_max[i])
            violations.append(lower_viol + upper_viol)
        return torch.stack(violations, dim=1).sum(dim=(1, 2))

    def smoothness_constraint(self, boundaries: torch.Tensor, max_var: float = 0.02) -> torch.Tensor:
        diff = torch.abs(boundaries[:, :, 1:] - boundaries[:, :, :-1])
        return F.relu(diff - max_var).sum(dim=(1, 2))

    def all_constraints(self, boundaries: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'ordering': self.ordering_constraint(boundaries),
            'thickness': self.thickness_constraint(boundaries),
            'smoothness': self.smoothness_constraint(boundaries),
        }


# =============================================================================
# Main Hybrid Model
# =============================================================================

class ConstrainedNAFNetHybrid(nn.Module):
    """
    Hybrid model: NAFNet backbone + Layer-specific refinement + Constraints.

    Novel contributions:
    1. Layer-specific refinement heads improve per-layer quality
    2. Anatomical constraints ensure valid boundaries
    3. Clinical losses optimize diagnostic value
    """

    LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL_IS', 'RPE_Choroid']

    def __init__(
        self,
        nafnet_ckpt: Optional[str] = None,
        freeze_nafnet: bool = True,
        refinement_width: int = 32,
        blend_sigma: float = 5.0,
    ):
        super().__init__()

        self.freeze_nafnet = freeze_nafnet
        self.blend_sigma = blend_sigma

        # NAFNet backbone
        if HAS_NAFNET:
            self.nafnet = NAFNet(img_channel=1, width=64, middle_blk_num=2,
                                enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2])
            if nafnet_ckpt and os.path.exists(nafnet_ckpt):
                ckpt = torch.load(nafnet_ckpt, map_location='cpu')
                if 'state_dict' in ckpt:
                    self.nafnet.load_state_dict(ckpt['state_dict'])
                else:
                    self.nafnet.load_state_dict(ckpt)
                print(f"Loaded NAFNet from {nafnet_ckpt}")

            if freeze_nafnet:
                for param in self.nafnet.parameters():
                    param.requires_grad = False
                self.nafnet.eval()
        else:
            self.nafnet = None

        # Boundary predictor
        self.boundary_predictor = BoundaryPredictor(width=32)

        # Layer-specific refinement heads
        self.layer_heads = nn.ModuleList([
            LayerRefinementHead(width=refinement_width) for _ in range(4)
        ])

        # Symbolic constraints
        self.constraints = SymbolicConstraints()

    def create_soft_masks(self, boundaries: torch.Tensor, H: int) -> torch.Tensor:
        B, N, W = boundaries.shape
        device = boundaries.device

        boundaries_px = boundaries * (H - 1)
        y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        b_exp = boundaries_px.unsqueeze(2)

        temp = self.blend_sigma

        mask_0 = torch.sigmoid((b_exp[:, 1:2] - y) / temp)
        mask_1 = torch.sigmoid((y - b_exp[:, 1:2]) / temp) * torch.sigmoid((b_exp[:, 2:3] - y) / temp)
        mask_2 = torch.sigmoid((y - b_exp[:, 2:3]) / temp) * torch.sigmoid((b_exp[:, 3:4] - y) / temp)
        mask_3 = torch.sigmoid((y - b_exp[:, 3:4]) / temp)

        soft_masks = torch.cat([mask_0, mask_1, mask_2, mask_3], dim=1)
        return soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = noisy.shape

        # 1. NAFNet base denoising
        if self.nafnet is not None:
            if self.freeze_nafnet:
                with torch.no_grad():
                    nafnet_out = self.nafnet(noisy)
            else:
                nafnet_out = self.nafnet(noisy)
        else:
            nafnet_out = noisy  # No backbone, just use input

        # 2. Predict boundaries
        boundaries = self.boundary_predictor(nafnet_out)

        # 3. Create soft masks
        soft_masks = self.create_soft_masks(boundaries, H)

        # 4. Layer-specific refinement
        layer_refinements = []
        for i, head in enumerate(self.layer_heads):
            refinement = head(nafnet_out)
            layer_refinements.append(refinement)

        refinement_stack = torch.cat(layer_refinements, dim=1)  # [B, 4, H, W]

        # 5. Apply refinements with soft masks
        weighted_refinement = (refinement_stack * soft_masks).sum(dim=1, keepdim=True)
        denoised = nafnet_out + weighted_refinement

        # 6. Per-layer outputs (for loss computation)
        layer_outputs = nafnet_out.expand(-1, 4, -1, -1) + refinement_stack

        return {
            'denoised': denoised,
            'nafnet_out': nafnet_out,
            'boundaries': boundaries,
            'soft_masks': soft_masks,
            'layer_outputs': layer_outputs,
            'layer_refinements': refinement_stack,
        }

    def compute_constraint_violations(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        return self.constraints.all_constraints(outputs['boundaries'])


# =============================================================================
# Clinical Losses
# =============================================================================

class ClinicalLosses(nn.Module):
    """Layer-specific clinical losses."""

    def __init__(self):
        super().__init__()

        self.register_buffer('laplacian', torch.tensor([
            [0, 1, 0], [1, -4, 1], [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4)

    def texture_loss(self, pred, target, mask):
        pred_hf = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.laplacian)
        target_hf = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.laplacian)
        diff = torch.abs(pred_hf - target_hf) * mask
        return diff.sum() / (mask.sum() + 1e-8)

    def edge_loss(self, pred, target, mask):
        pred_gx = F.conv2d(F.pad(pred, (1,1,1,1), mode='replicate'), self.sobel_x)
        target_gx = F.conv2d(F.pad(target, (1,1,1,1), mode='replicate'), self.sobel_x)
        diff = torch.abs(pred_gx - target_gx) * mask
        return diff.sum() / (mask.sum() + 1e-8)

    def compute_all(self, outputs: Dict, clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        denoised = outputs['denoised']
        soft_masks = outputs['soft_masks']
        layer_outputs = outputs['layer_outputs']

        losses = {}

        # Global L1
        global_l1 = F.l1_loss(denoised, clean)
        losses['global_l1'] = global_l1.item()

        total = global_l1

        # Per-layer losses
        loss_fns = [self.texture_loss, self.edge_loss, self.edge_loss, self.texture_loss]
        weights = [0.3, 0.2, 0.3, 0.3]

        for i in range(4):
            mask = soft_masks[:, i:i+1]
            layer_out = layer_outputs[:, i:i+1]

            layer_l1 = (torch.abs(layer_out - clean) * mask).sum() / (mask.sum() + 1e-8)
            clinical = loss_fns[i](layer_out, clean, mask)

            total = total + weights[i] * (layer_l1 + 0.3 * clinical)
            losses[f'layer{i}_l1'] = layer_l1.item()
            losses[f'layer{i}_clinical'] = clinical.item()

        losses['total'] = total.item()
        return total, losses


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Constrained NAFNet Hybrid")
    print("=" * 60)

    model = ConstrainedNAFNetHybrid(
        nafnet_ckpt='outputs/nafnet_pku37/nafnet_best.pth',
        freeze_nafnet=True,
        refinement_width=32,
    )

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params_total:,}")
    print(f"Trainable params: {n_params_trainable:,}")

    # Test forward
    B, H, W = 2, 128, 128
    noisy = torch.rand(B, 1, H, W)
    clean = torch.rand(B, 1, H, W)

    outputs = model(noisy)
    print(f"\nOutputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Test losses
    clinical = ClinicalLosses()
    loss, details = clinical.compute_all(outputs, clean)
    print(f"\nLosses:")
    for k, v in details.items():
        print(f"  {k}: {v:.4f}")

    # Test constraints
    violations = model.compute_constraint_violations(outputs)
    print(f"\nConstraints:")
    for k, v in violations.items():
        print(f"  {k}: {v.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("Hybrid model ready!")
    print("Key insight: NAFNet provides strong base denoising,")
    print("layer-specific heads add targeted refinements.")
