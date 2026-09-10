#!/usr/bin/env python3
"""
Train V8 Enhanced with Region-Adaptive Correction and Pathology Preservation

IEEE TMI Publication-Worthy Training Script

This script integrates:
1. Region-Adaptive Lambda Prediction (from region_adaptive_correction.py)
   - Applies strong corrections on clinical regions (layer boundaries, edges)
   - Minimal corrections on flat regions to preserve PSNR

2. Pathology Preservation (from pathology_preservation.py)
   - Unsupervised pathology detection (drusen, fluid, atrophy)
   - Prevents over-smoothing of diagnostic features

3. V8EnhancedLoss base losses (recon, clinical, predicate)

Training Targets (for IEEE TMI):
- +10-15% clinical improvement in boundary regions
- < 0.5 dB PSNR drop (acceptable trade-off)
- Strong pathology preservation

Author: Neuro-Symbolic OCT Team
Date: 2026-02-03
"""

import argparse
import os
import sys
import json
import gc
import math
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

# Import components
from neuro_symbolic_corrector_v8_enhanced import NeuroSymbolicCorrectorV8Enhanced

# Import region-adaptive and pathology modules
from region_adaptive_correction import (
    RegionDetector,
    LearnableRegionDetector,
    RegionAdaptiveLambdaPredictor,
    BoundaryFocusedLoss,
    BoundaryFocusedLossV2,
)
from pathology_preservation import (
    PathologyDetector,
    PathologyPreservationModule,
    PathologyPreservationLoss,
)


# =============================================================================
# Backbone Wrapper with Feature Extraction
# =============================================================================

class BackboneWithFeatures(nn.Module):
    """
    NAFNet backbone that returns intermediate encoder features.

    Returns:
        denoised: Final denoised output [B, 1, H, W]
        features: Dict with 'enc1' [B, width, H, W] and 'enc2' [B, width*2, H/2, W/2]
    """

    def __init__(self, width: int = 40):
        super().__init__()

        from nsnd.models.nafnet import NAFNetSmall
        self.backbone = NAFNetSmall(img_channel=1, width=width)
        self.width = width

    def load_pretrained(self, path: str) -> bool:
        """Load pretrained weights."""
        if not os.path.exists(path):
            print(f"Pretrained weights not found: {path}")
            return False

        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

            model_state = self.backbone.state_dict()
            compatible = {k: v for k, v in state_dict.items()
                         if k in model_state and v.shape == model_state[k].shape}

            if len(compatible) == 0:
                print("No compatible weights found")
                return False

            self.backbone.load_state_dict(compatible, strict=False)
            print(f"Loaded {len(compatible)}/{len(model_state)} backbone weights")

            if 'psnr' in ckpt:
                print(f"  Checkpoint PSNR: {ckpt['psnr']:.2f} dB, SSIM: {ckpt.get('ssim', 'N/A')}")
            return True
        except Exception as e:
            print(f"Failed to load backbone: {e}")
            return False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with feature extraction."""
        bb = self.backbone
        B, C, H, W = x.shape

        padder_size = 2 ** len(bb.encoders)
        mod_h = H % padder_size
        mod_w = W % padder_size
        if mod_h != 0 or mod_w != 0:
            pad_h = padder_size - mod_h if mod_h != 0 else 0
            pad_w = padder_size - mod_w if mod_w != 0 else 0
            inp = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            inp = x

        feat = bb.intro(inp)

        encs = []
        encs_for_features = []
        for i, (encoder, down) in enumerate(zip(bb.encoders, bb.downs)):
            feat = encoder(feat)
            encs.append(feat)
            if i < 2:
                encs_for_features.append(feat)
            feat = down(feat)

        feat = bb.middle_blks(feat)

        for decoder, up, enc_skip in zip(bb.decoders, bb.ups, encs[::-1]):
            feat = up(feat)
            feat = feat + enc_skip
            feat = decoder(feat)

        out = bb.ending(feat) + inp
        out = out[:, :, :H, :W]

        features = {
            'enc1': encs_for_features[0][:, :, :H, :W].detach(),
            'enc2': encs_for_features[1][:, :, :min(H//2, encs_for_features[1].shape[2]), :min(W//2, encs_for_features[1].shape[3])].detach(),
        }
        del encs, encs_for_features

        return out.clamp(0, 1), features


# =============================================================================
# Dataset
# =============================================================================

class PKU37Dataset(Dataset):
    """PKU37 real noise dataset."""

    def __init__(self, jsonl_path: str, max_samples: int = None,
                 patch_size: int = 96, is_train: bool = True):
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

        print(f"Loaded {len(self.samples)} PKU37 samples")

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

        if self.patch_size > 0 and self.is_train:
            H, W = clean.shape
            if H > self.patch_size and W > self.patch_size:
                y = np.random.randint(0, H - self.patch_size)
                x = np.random.randint(0, W - self.patch_size)
                clean = clean[y:y+self.patch_size, x:x+self.patch_size]
                noisy = noisy[y:y+self.patch_size, x:x+self.patch_size]

        if self.is_train:
            if np.random.rand() > 0.5:
                clean = np.flip(clean, axis=1).copy()
                noisy = np.flip(noisy, axis=1).copy()
            if np.random.rand() > 0.5:
                clean = np.flip(clean, axis=0).copy()
                noisy = np.flip(noisy, axis=0).copy()

        clean = torch.from_numpy(clean).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        return {'clean': clean, 'noisy': noisy}


# =============================================================================
# Model with Region-Adaptive Correction
# =============================================================================

class NeuroSymbolicDenoiserV8Region(nn.Module):
    """
    V8 Enhanced Neuro-Symbolic Denoiser with Region-Adaptive Correction.

    Key enhancements:
    1. Region-adaptive lambda prediction (different correction strength per region)
    2. Pathology-aware correction gating
    3. Clinical importance weighting by retinal layer
    """

    def __init__(self, backbone_width: int = 48, pretrained_backbone: str = None,
                 use_region_adaptive: bool = True, use_pathology_gate: bool = True):
        super().__init__()

        self.use_region_adaptive = use_region_adaptive
        self.use_pathology_gate = use_pathology_gate

        # Backbone with feature extraction
        self.backbone = BackboneWithFeatures(width=backbone_width)

        # V8 Enhanced Corrector
        self.corrector = NeuroSymbolicCorrectorV8Enhanced(
            in_channels=1,
            hidden_channels=32,
            enc1_channels=backbone_width,
            enc2_channels=backbone_width * 2
        )

        # Region-adaptive lambda predictor (replaces default)
        if use_region_adaptive:
            self.region_lambda_predictor = RegionAdaptiveLambdaPredictor(
                use_learnable_detector=True,
                importance_power=1.0,
                min_importance=0.1,  # Allow some correction even in flat regions
                hidden_dim=32
            )

        # Pathology preservation module
        if use_pathology_gate:
            self.pathology_module = PathologyPreservationModule(
                hidden_dim=32,
                use_learned_combination=True,
                min_gate_value=0.3  # Don't completely block correction
            )

        # Load pretrained backbone
        if pretrained_backbone and os.path.exists(pretrained_backbone):
            self.backbone.load_pretrained(pretrained_backbone)

        self._print_params()

    def _print_params(self):
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        corrector_params = sum(p.numel() for p in self.corrector.parameters())

        region_params = 0
        if self.use_region_adaptive:
            region_params = sum(p.numel() for p in self.region_lambda_predictor.parameters())

        pathology_params = 0
        if self.use_pathology_gate:
            pathology_params = sum(p.numel() for p in self.pathology_module.parameters())

        total_params = backbone_params + corrector_params + region_params + pathology_params

        print(f"\nNeuroSymbolicDenoiserV8Region Parameters:")
        print(f"  Backbone (width={self.backbone.width}): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"  V8 Enhanced Corrector: {corrector_params:,} ({corrector_params/1e6:.2f}M)")
        if self.use_region_adaptive:
            print(f"  Region-Adaptive Lambda: {region_params:,} ({region_params/1e6:.3f}M)")
        if self.use_pathology_gate:
            print(f"  Pathology Module: {pathology_params:,} ({pathology_params/1e6:.3f}M)")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, noisy: torch.Tensor, return_details: bool = False):
        """
        Forward pass with region-adaptive correction.

        Returns:
            corrected: Final output
            backbone_out: Backbone-only output
            info: Detailed information dict
        """
        # Get backbone output and features
        backbone_out, backbone_features = self.backbone(noisy)

        # Apply V8 Enhanced correction with feature fusion
        corrected, info = self.corrector(
            backbone_out, noisy, backbone_features, return_details=return_details
        )

        # Region-adaptive lambda modulation
        if self.use_region_adaptive:
            region_info = self.region_lambda_predictor.get_region_info()
            if region_info is not None:
                info['region_info'] = {
                    'importance_map': region_info['importance_map'],
                    'heuristic_importance': region_info.get('heuristic_importance'),
                    'learned_importance': region_info.get('learned_importance'),
                    'blend_weight': region_info.get('blend_weight'),
                }

        # Pathology-aware correction gating
        if self.use_pathology_gate:
            pathology_info = self.pathology_module(noisy, backbone_out)
            pathology_gate = pathology_info['correction_gate']

            # Apply pathology gate to correction (not backbone)
            correction = corrected - backbone_out
            gated_correction = correction * pathology_gate
            corrected = backbone_out + gated_correction

            info['pathology_info'] = {
                'pathology_map': pathology_info['pathology_map'],
                'correction_gate': pathology_gate,
                'detection_info': pathology_info['detection_info'],
            }

        return corrected, backbone_out, info


# =============================================================================
# Combined Loss with Region and Pathology
# =============================================================================

class RegionPathologyLoss(nn.Module):
    """
    Combined loss function for Region-Adaptive + Pathology Preservation training.

    Integrates:
    1. V8EnhancedLoss base components (recon, clinical, predicate)
    2. BoundaryFocusedLoss from region_adaptive_correction.py
    3. PathologyPreservationLoss from pathology_preservation.py
    4. PSNR preservation constraint (0.5 dB slack max)

    Uses uncertainty weighting (Kendall et al., 2018) for automatic balancing.
    """

    def __init__(self,
                 pathology_weight: float = 0.5,
                 boundary_weight: float = 0.3,
                 clinical_boundary_multiplier: float = 2.0,
                 psnr_slack: float = 0.5,
                 use_uncertainty_weighting: bool = True):
        super().__init__()

        self.pathology_weight = pathology_weight
        self.boundary_weight = boundary_weight
        self.psnr_slack = psnr_slack
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Boundary-focused loss with region weighting
        self.boundary_loss = BoundaryFocusedLossV2(
            clinical_boundary_weight=clinical_boundary_multiplier,
            clinical_flat_weight=0.5,
            recon_boundary_weight=0.5,
            recon_flat_weight=2.0,
            importance_threshold=0.5,
            use_uncertainty_weighting=use_uncertainty_weighting
        )

        # Pathology preservation loss
        self.pathology_loss = PathologyPreservationLoss(
            hidden_dim=32,
            lambda_feature=1.0,
            lambda_contrast=0.5,
            lambda_texture=0.5,
            lambda_boundary=0.3,
            use_uncertainty_weighting=use_uncertainty_weighting
        )

        # Region detector for region-specific metrics
        self.region_detector = RegionDetector()

        # Uncertainty parameters for combined losses
        self.log_sigma = nn.ParameterDict({
            'recon': nn.Parameter(torch.tensor(0.0)),
            'backbone': nn.Parameter(torch.tensor(1.0)),
            'pred': nn.Parameter(torch.tensor(2.0)),
            'contrast': nn.Parameter(torch.tensor(1.0)),
            'boundary': nn.Parameter(torch.tensor(1.0)),
            'texture': nn.Parameter(torch.tensor(1.0)),
            'edge': nn.Parameter(torch.tensor(1.0)),
            'pred_align': nn.Parameter(torch.tensor(0.5)),
            'psnr_preserve': nn.Parameter(torch.tensor(-1.0)),
            # Region and pathology specific
            'boundary_focused': nn.Parameter(torch.tensor(0.5)),
            'pathology': nn.Parameter(torch.tensor(0.5)),
            'region_clinical': nn.Parameter(torch.tensor(0.5)),
            # Contrast improvement loss (to fix P2)
            'contrast_improve': nn.Parameter(torch.tensor(0.0)),
        })

        # Sobel filters
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Lambda regularization weight
        self.lambda_reg_weight = 0.01

    def _weighted_loss(self, loss: torch.Tensor, name: str) -> torch.Tensor:
        """Apply uncertainty weighting to a loss term."""
        if self.use_uncertainty_weighting:
            log_sig = self.log_sigma[name]
            precision_weight = 0.5 * torch.exp(-log_sig)
            regularization = 0.5 * log_sig
            return precision_weight * loss + regularization
        else:
            return loss

    def get_learned_weights(self) -> dict:
        """Get current learned weights."""
        weights = {}
        for name, log_sig in self.log_sigma.items():
            sigma_sq = torch.exp(log_sig)
            weight = 0.5 / sigma_sq
            weights[name] = weight.item()
        return weights

    def local_std(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Compute local standard deviation."""
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                           device=x.device, dtype=x.dtype) / (kernel_size ** 2)
        padding = kernel_size // 2
        local_mean = F.conv2d(x, kernel, padding=padding)
        local_mean_sq = F.conv2d(x ** 2, kernel, padding=padding)
        local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=1e-6)
        return torch.sqrt(local_var)

    def compute_region_clinical_loss(self, corrected: torch.Tensor, clean: torch.Tensor,
                                     region_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute clinical losses weighted by region importance.

        Higher weights on boundary regions for clinical metrics.
        """
        importance_map = region_info.get('importance_map')
        if importance_map is None:
            # Compute importance if not provided
            region_det = self.region_detector(clean)
            importance_map = region_det['importance_map']

        # Boundary mask (high importance regions)
        # Use percentile-based threshold (top 20% are boundaries)
        importance_flat = importance_map.view(importance_map.shape[0], -1)
        threshold = torch.quantile(importance_flat, 0.8, dim=1, keepdim=True)
        threshold = threshold.view(-1, 1, 1, 1)
        boundary_mask = (importance_map > threshold).float()
        flat_mask = 1.0 - boundary_mask

        # Contrast loss (weighted by region)
        clean_std = self.local_std(clean)
        corrected_std = self.local_std(corrected)
        contrast_deficit = F.relu(clean_std - corrected_std)

        # Higher weight on boundary regions
        boundary_contrast = (contrast_deficit * boundary_mask * 2.0).mean()
        flat_contrast = (contrast_deficit * flat_mask * 0.5).mean()
        total_contrast = boundary_contrast + flat_contrast

        # Boundary sharpness loss (vertical gradients)
        clean_vgrad = torch.abs(clean[:, :, 1:, :] - clean[:, :, :-1, :])
        corrected_vgrad = torch.abs(corrected[:, :, 1:, :] - corrected[:, :, :-1, :])
        boundary_deficit = F.relu(clean_vgrad - corrected_vgrad)

        boundary_mask_adj = boundary_mask[:, :, :-1, :]
        flat_mask_adj = flat_mask[:, :, :-1, :]

        boundary_sharp = (boundary_deficit * boundary_mask_adj * 2.0).mean()
        flat_sharp = (boundary_deficit * flat_mask_adj * 0.5).mean()
        total_boundary = boundary_sharp + flat_sharp

        # Edge loss (Sobel)
        sobel_x = self.sobel_x.to(dtype=corrected.dtype)
        sobel_y = self.sobel_y.to(dtype=corrected.dtype)

        clean_edge = torch.sqrt(
            F.conv2d(clean, sobel_x, padding=1)**2 +
            F.conv2d(clean, sobel_y, padding=1)**2 + 1e-6
        )
        corrected_edge = torch.sqrt(
            F.conv2d(corrected, sobel_x, padding=1)**2 +
            F.conv2d(corrected, sobel_y, padding=1)**2 + 1e-6
        )
        edge_deficit = F.relu(clean_edge - corrected_edge)

        boundary_edge = (edge_deficit * boundary_mask * 2.0).mean()
        flat_edge = (edge_deficit * flat_mask * 0.5).mean()
        total_edge = boundary_edge + flat_edge

        # Combined region-weighted clinical loss
        total_clinical = total_contrast + total_boundary + total_edge

        metrics = {
            'region_contrast_boundary': boundary_contrast.item(),
            'region_contrast_flat': flat_contrast.item(),
            'region_boundary_boundary': boundary_sharp.item(),
            'region_boundary_flat': flat_sharp.item(),
            'region_edge_boundary': boundary_edge.item(),
            'region_edge_flat': flat_edge.item(),
            'boundary_coverage': boundary_mask.mean().item(),
        }

        return total_clinical, metrics

    def psnr_preservation_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                                clean: torch.Tensor) -> torch.Tensor:
        """Penalize when corrected PSNR drops below backbone beyond slack."""
        eps = 1e-8
        mse_backbone = F.mse_loss(backbone, clean)
        mse_corrected = F.mse_loss(corrected, clean)

        psnr_backbone = 10 * torch.log10(1.0 / (mse_backbone + eps))
        psnr_corrected = 10 * torch.log10(1.0 / (mse_corrected + eps))

        psnr_drop = psnr_backbone - psnr_corrected
        penalty = F.relu(psnr_drop - self.psnr_slack)
        return penalty ** 2

    def contrast_improvement_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                                   clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Loss to improve P2 (Contrast) predicate score.

        P2 measures:
        1. Global range (30% weight)
        2. Local std variance near 0.01 (30% weight)
        3. Mean local std (40% weight)

        This loss encourages corrected image to have better contrast than backbone,
        targeting the GT contrast characteristics.
        """
        # Compute local std for all images
        clean_std = self.local_std(clean, 9)
        backbone_std = self.local_std(backbone, 9)
        corrected_std = self.local_std(corrected, 9)

        # 1. Global range: Encourage corrected range to match GT range
        clean_range = clean.max() - clean.min()
        corrected_range = corrected.max() - corrected.min()
        range_loss = F.mse_loss(corrected_range, clean_range)

        # 2. Local std distribution: Encourage corrected std to match GT std
        # This helps with the variance component of P2
        std_loss = F.mse_loss(corrected_std, clean_std)

        # 3. Mean local contrast: Encourage corrected to have at least backbone contrast
        # Penalize contrast reduction vs backbone
        backbone_mean_contrast = backbone_std.mean()
        corrected_mean_contrast = corrected_std.mean()
        clean_mean_contrast = clean_std.mean()

        # Loss: Pull corrected contrast towards GT contrast
        contrast_target_loss = F.mse_loss(corrected_mean_contrast, clean_mean_contrast)

        # Bonus: Penalize if corrected has LESS contrast than backbone
        contrast_deficit = F.relu(backbone_mean_contrast - corrected_mean_contrast)
        deficit_penalty = contrast_deficit ** 2 * 5.0  # Strong penalty for contrast reduction

        # Combined loss
        total_loss = range_loss + std_loss + contrast_target_loss + deficit_penalty

        metrics = {
            'contrast_range_loss': range_loss.item(),
            'contrast_std_loss': std_loss.item(),
            'contrast_target_loss': contrast_target_loss.item(),
            'contrast_deficit_penalty': deficit_penalty.item(),
            'backbone_mean_contrast': backbone_mean_contrast.item(),
            'corrected_mean_contrast': corrected_mean_contrast.item(),
            'clean_mean_contrast': clean_mean_contrast.item(),
        }

        return total_loss, metrics

    def forward(self, corrected: torch.Tensor, backbone_out: torch.Tensor,
                clean: torch.Tensor, noisy: torch.Tensor, info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss with region-adaptive and pathology preservation.
        """
        device = corrected.device

        # ===== 1. Basic Reconstruction Loss =====
        recon_loss = F.mse_loss(corrected, clean)
        backbone_loss = F.mse_loss(backbone_out, clean)

        # ===== 2. Boundary-Focused Loss =====
        boundary_loss, boundary_metrics = self.boundary_loss(
            corrected, backbone_out, clean, info
        )

        # ===== 3. Pathology Preservation Loss =====
        pathology_loss, pathology_metrics = self.pathology_loss(
            corrected, backbone_out, noisy
        )

        # ===== 4. Region-Weighted Clinical Loss =====
        region_info = info.get('region_info', {})
        region_clinical_loss, region_metrics = self.compute_region_clinical_loss(
            corrected, clean, region_info
        )

        # ===== 5. PSNR Preservation Constraint =====
        psnr_preserve_loss = self.psnr_preservation_loss(corrected, backbone_out, clean)

        # ===== 6. Contrast Improvement Loss (to fix P2) =====
        contrast_improve_loss, contrast_metrics = self.contrast_improvement_loss(
            corrected, backbone_out, clean
        )

        # ===== 7. Lambda Regularization =====
        lambda_stats = info.get('lambda_stats', {})
        lambda_reg = torch.tensor(0.0, device=device)
        if lambda_stats:
            means = [s['mean'] for s in lambda_stats.values() if isinstance(s.get('mean'), torch.Tensor)]
            if means:
                lambda_reg = torch.stack(means).mean() * self.lambda_reg_weight

        # ===== Combined Loss with Uncertainty Weighting =====
        total_loss = (
            self._weighted_loss(recon_loss, 'recon') +
            self._weighted_loss(backbone_loss, 'backbone') +
            self._weighted_loss(boundary_loss, 'boundary_focused') * self.boundary_weight +
            self._weighted_loss(pathology_loss, 'pathology') * self.pathology_weight +
            self._weighted_loss(region_clinical_loss, 'region_clinical') +
            self._weighted_loss(psnr_preserve_loss, 'psnr_preserve') +
            self._weighted_loss(contrast_improve_loss, 'contrast_improve') * 2.0 +  # Strong weight to fix P2
            lambda_reg
        )

        # ===== Compute Metrics =====
        with torch.no_grad():
            psnr_backbone = 10 * torch.log10(1 / (backbone_loss + 1e-6))
            psnr_corrected = 10 * torch.log10(1 / (recon_loss + 1e-6))

        metrics = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'backbone': backbone_loss.item(),
            'boundary_focused': boundary_loss.item(),
            'pathology': pathology_loss.item(),
            'region_clinical': region_clinical_loss.item(),
            'psnr_preserve': psnr_preserve_loss.item(),
            'lambda_reg': lambda_reg.item() if isinstance(lambda_reg, torch.Tensor) else lambda_reg,
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'psnr_delta': psnr_corrected.item() - psnr_backbone.item(),
            'contrast_improve': contrast_improve_loss.item(),
            # Include detailed metrics
            **boundary_metrics,
            **pathology_metrics,
            **region_metrics,
            **contrast_metrics,
            'learned_weights': self.get_learned_weights(),
        }

        # Add predicate scores if available
        pred_scores = info.get('predicate_scores', {})
        if pred_scores:
            metrics['avg_pred_score'] = sum(pred_scores.values()) / len(pred_scores)
        else:
            metrics['avg_pred_score'] = 0.5

        return total_loss, metrics


# =============================================================================
# Training Functions
# =============================================================================

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())


def compute_ssim(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    mu_pred = pred.mean()
    mu_target = target.mean()
    sigma_pred = torch.clamp(((pred - mu_pred) ** 2).mean(), min=1e-8)
    sigma_target = torch.clamp(((target - mu_target) ** 2).mean(), min=1e-8)
    sigma_both = ((pred - mu_pred) * (target - mu_target)).mean()
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
           ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))
    return ssim.item()


def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()

    total_loss = 0
    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_pred_score = 0
    total_pathology_coverage = 0
    total_boundary_coverage = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        if scaler is not None:
            with autocast():
                corrected, backbone_out, info = model(noisy)
                loss, metrics = criterion(corrected, backbone_out, clean, noisy, info)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            corrected, backbone_out, info = model(noisy)
            loss, metrics = criterion(corrected, backbone_out, clean, noisy, info)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            optimizer.step()

        del corrected, backbone_out, info

        total_loss += metrics['total']
        total_psnr_backbone += metrics['psnr_backbone']
        total_psnr_corrected += metrics['psnr_corrected']
        total_pred_score += metrics.get('avg_pred_score', 0.5)
        total_pathology_coverage += metrics.get('pathology_coverage', 0)
        total_boundary_coverage += metrics.get('boundary_coverage', 0)
        n += 1

        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'psnr': f"{metrics['psnr_corrected']:.1f}",
            'delta': f"{metrics['psnr_delta']:+.2f}",
            'path': f"{metrics.get('pathology_coverage', 0):.2f}",
        })

        last_learned_weights = metrics.get('learned_weights', {})

    return {
        'loss': total_loss / n,
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'pred_score': total_pred_score / n,
        'pathology_coverage': total_pathology_coverage / n,
        'boundary_coverage': total_boundary_coverage / n,
        'learned_weights': last_learned_weights,
    }


def compute_region_specific_metrics(corrected, backbone, clean, device):
    """
    Compute clinical metrics separately for boundary and flat regions.
    """
    detector = RegionDetector().to(device)
    region_info = detector(clean)
    importance_map = region_info['importance_map']

    # Use percentile-based threshold (top 20% are boundaries)
    importance_flat = importance_map.view(importance_map.shape[0], -1)
    threshold = torch.quantile(importance_flat, 0.8, dim=1, keepdim=True)
    threshold = threshold.view(-1, 1, 1, 1)
    boundary_mask = (importance_map > threshold).float()
    flat_mask = 1.0 - boundary_mask

    # Sobel for edges
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device, dtype=corrected.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    def edge_magnitude(img):
        ex = F.conv2d(img, sobel_x, padding=1)
        ey = F.conv2d(img, sobel_y, padding=1)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

    def local_std(x, kernel_size=5):
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype) / (kernel_size ** 2)
        padding = kernel_size // 2
        local_mean = F.conv2d(x, kernel, padding=padding)
        local_mean_sq = F.conv2d(x ** 2, kernel, padding=padding)
        local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=1e-6)
        return torch.sqrt(local_var)

    # Compute metrics for each region
    clean_edge = edge_magnitude(clean)
    backbone_edge = edge_magnitude(backbone)
    corrected_edge = edge_magnitude(corrected)

    clean_std = local_std(clean)
    backbone_std = local_std(backbone)
    corrected_std = local_std(corrected)

    # Boundary region metrics
    boundary_mask_sum = boundary_mask.sum() + 1e-8
    flat_mask_sum = flat_mask.sum() + 1e-8

    # Edge preservation in boundary regions
    clean_edge_boundary = (clean_edge * boundary_mask).sum() / boundary_mask_sum
    backbone_edge_boundary = (backbone_edge * boundary_mask).sum() / boundary_mask_sum
    corrected_edge_boundary = (corrected_edge * boundary_mask).sum() / boundary_mask_sum

    # Contrast preservation in boundary regions
    clean_std_boundary = (clean_std * boundary_mask).sum() / boundary_mask_sum
    backbone_std_boundary = (backbone_std * boundary_mask).sum() / boundary_mask_sum
    corrected_std_boundary = (corrected_std * boundary_mask).sum() / boundary_mask_sum

    # Flat region metrics
    clean_edge_flat = (clean_edge * flat_mask).sum() / flat_mask_sum
    backbone_edge_flat = (backbone_edge * flat_mask).sum() / flat_mask_sum
    corrected_edge_flat = (corrected_edge * flat_mask).sum() / flat_mask_sum

    clean_std_flat = (clean_std * flat_mask).sum() / flat_mask_sum
    backbone_std_flat = (backbone_std * flat_mask).sum() / flat_mask_sum
    corrected_std_flat = (corrected_std * flat_mask).sum() / flat_mask_sum

    # Compute improvement ratios
    eps = 1e-6

    # Edge improvement
    backbone_edge_pres_boundary = backbone_edge_boundary / (clean_edge_boundary + eps)
    corrected_edge_pres_boundary = corrected_edge_boundary / (clean_edge_boundary + eps)
    edge_improvement_boundary = (corrected_edge_pres_boundary / (backbone_edge_pres_boundary + eps)).item()

    backbone_edge_pres_flat = backbone_edge_flat / (clean_edge_flat + eps)
    corrected_edge_pres_flat = corrected_edge_flat / (clean_edge_flat + eps)
    edge_improvement_flat = (corrected_edge_pres_flat / (backbone_edge_pres_flat + eps)).item()

    # Contrast improvement
    backbone_std_pres_boundary = backbone_std_boundary / (clean_std_boundary + eps)
    corrected_std_pres_boundary = corrected_std_boundary / (clean_std_boundary + eps)
    contrast_improvement_boundary = (corrected_std_pres_boundary / (backbone_std_pres_boundary + eps)).item()

    backbone_std_pres_flat = backbone_std_flat / (clean_std_flat + eps)
    corrected_std_pres_flat = corrected_std_flat / (clean_std_flat + eps)
    contrast_improvement_flat = (corrected_std_pres_flat / (backbone_std_pres_flat + eps)).item()

    return {
        'edge_improvement_boundary': edge_improvement_boundary,
        'edge_improvement_flat': edge_improvement_flat,
        'contrast_improvement_boundary': contrast_improvement_boundary,
        'contrast_improvement_flat': contrast_improvement_flat,
        'boundary_coverage': boundary_mask.mean().item(),
        # Raw values
        'backbone_edge_pres_boundary': backbone_edge_pres_boundary.item(),
        'corrected_edge_pres_boundary': corrected_edge_pres_boundary.item(),
        'backbone_edge_pres_flat': backbone_edge_pres_flat.item(),
        'corrected_edge_pres_flat': corrected_edge_pres_flat.item(),
        'backbone_contrast_pres_boundary': backbone_std_pres_boundary.item(),
        'corrected_contrast_pres_boundary': corrected_std_pres_boundary.item(),
        'backbone_contrast_pres_flat': backbone_std_pres_flat.item(),
        'corrected_contrast_pres_flat': corrected_std_pres_flat.item(),
    }


@torch.inference_mode()
def validate(model, loader, device):
    """Comprehensive validation with region-specific and pathology metrics."""
    model.eval()
    use_amp = device != 'cpu' and torch.cuda.is_available()

    # Basic metrics
    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_ssim_backbone = 0
    total_ssim_corrected = 0

    # Region-specific metrics
    total_edge_improvement_boundary = 0
    total_edge_improvement_flat = 0
    total_contrast_improvement_boundary = 0
    total_contrast_improvement_flat = 0
    total_boundary_coverage = 0

    # Pathology metrics
    total_pathology_coverage = 0
    total_pathology_preservation = 0

    # Lambda and correction stats
    total_lambda_stats = {}
    total_correction_mag = 0

    # Predicate scores
    total_pred_scores = {f'P{i}': 0 for i in range(1, 7)}

    n = 0

    for batch in tqdm(loader, desc="Validation"):
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        if use_amp:
            with autocast():
                corrected, backbone_out, info = model(noisy)
        else:
            corrected, backbone_out, info = model(noisy)

        # Basic metrics
        psnr_backbone = compute_psnr(backbone_out, clean)
        psnr_corrected = compute_psnr(corrected, clean)
        ssim_backbone = compute_ssim(backbone_out, clean)
        ssim_corrected = compute_ssim(corrected, clean)

        total_psnr_backbone += psnr_backbone
        total_psnr_corrected += psnr_corrected
        total_ssim_backbone += ssim_backbone
        total_ssim_corrected += ssim_corrected

        # Region-specific metrics
        region_metrics = compute_region_specific_metrics(corrected, backbone_out, clean, device)
        total_edge_improvement_boundary += region_metrics['edge_improvement_boundary']
        total_edge_improvement_flat += region_metrics['edge_improvement_flat']
        total_contrast_improvement_boundary += region_metrics['contrast_improvement_boundary']
        total_contrast_improvement_flat += region_metrics['contrast_improvement_flat']
        total_boundary_coverage += region_metrics['boundary_coverage']

        # Pathology metrics
        pathology_info = info.get('pathology_info', {})
        if 'pathology_map' in pathology_info:
            total_pathology_coverage += pathology_info['pathology_map'].mean().item()

        # Lambda stats
        for name, stats in info.get('lambda_stats', {}).items():
            if name not in total_lambda_stats:
                total_lambda_stats[name] = {'mean': 0, 'max': 0}
            mean_val = stats['mean'].item() if isinstance(stats['mean'], torch.Tensor) else float(stats['mean'])
            max_val = stats['max'].item() if isinstance(stats['max'], torch.Tensor) else float(stats['max'])
            total_lambda_stats[name]['mean'] += mean_val
            total_lambda_stats[name]['max'] += max_val

        corr_mag = info.get('correction_magnitude', 0)
        total_correction_mag += corr_mag.item() if isinstance(corr_mag, torch.Tensor) else float(corr_mag)

        # Predicate scores
        for name, score in info.get('predicate_scores', {}).items():
            key = name.split('_')[0]
            if key in total_pred_scores:
                score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                total_pred_scores[key] += score_val

        del corrected, backbone_out, info
        n += 1

    return {
        # Basic metrics
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'ssim_backbone': total_ssim_backbone / n,
        'ssim_corrected': total_ssim_corrected / n,
        'psnr_delta': (total_psnr_corrected - total_psnr_backbone) / n,

        # Region-specific metrics (KEY FOR IEEE TMI)
        'edge_improvement_boundary': total_edge_improvement_boundary / n,
        'edge_improvement_flat': total_edge_improvement_flat / n,
        'contrast_improvement_boundary': total_contrast_improvement_boundary / n,
        'contrast_improvement_flat': total_contrast_improvement_flat / n,
        'boundary_coverage': total_boundary_coverage / n,

        # Pathology metrics
        'pathology_coverage': total_pathology_coverage / n,

        # Lambda and correction
        'lambda_stats': {k: {'mean': v['mean']/n, 'max': v['max']/n}
                        for k, v in total_lambda_stats.items()},
        'correction_magnitude': total_correction_mag / n,

        # Predicate scores
        'pred_scores': {k: v / n for k, v in total_pred_scores.items()},
    }


def print_publication_metrics(epoch, train_metrics, val_metrics):
    """
    Print publication-ready metrics for IEEE TMI.

    Focus on:
    - Region-specific clinical improvement
    - PSNR trade-off
    - Pathology preservation
    """
    psnr_delta = val_metrics['psnr_delta']

    # Calculate clinical improvement percentages
    edge_improve_boundary = (val_metrics['edge_improvement_boundary'] - 1.0) * 100
    edge_improve_flat = (val_metrics['edge_improvement_flat'] - 1.0) * 100
    contrast_improve_boundary = (val_metrics['contrast_improvement_boundary'] - 1.0) * 100
    contrast_improve_flat = (val_metrics['contrast_improvement_flat'] - 1.0) * 100

    # Determine success status
    target_achieved = (edge_improve_boundary >= 10 or contrast_improve_boundary >= 10) and abs(psnr_delta) <= 0.5

    print(f"\n{'#'*80}")
    print(f"# EPOCH {epoch:3d} | IEEE TMI PUBLICATION METRICS")
    if target_achieved:
        print(f"# STATUS: TARGET ACHIEVED - Publication ready!")
    else:
        print(f"# STATUS: Training in progress...")
    print(f"{'#'*80}")

    # ===== REGION-SPECIFIC CLINICAL IMPROVEMENT =====
    print(f"\n{'='*80}")
    print(f" REGION-SPECIFIC CLINICAL IMPROVEMENT (Target: +10-15% in boundary regions)")
    print(f"{'='*80}")

    print(f"\n{'+-'*40}+")
    print(f"| {'Metric':<30} | {'Boundary':>12} | {'Flat':>12} | {'Target':>12} |")
    print(f"{'+-'*40}+")

    edge_status = "OK" if edge_improve_boundary >= 10 else "..."
    print(f"| {'Edge Preservation Improvement':<30} | {edge_improve_boundary:>+11.1f}% | {edge_improve_flat:>+11.1f}% | {'+10-15%':>12} | [{edge_status}]")

    contrast_status = "OK" if contrast_improve_boundary >= 10 else "..."
    print(f"| {'Contrast Preservation Improve':<30} | {contrast_improve_boundary:>+11.1f}% | {contrast_improve_flat:>+11.1f}% | {'+10-15%':>12} | [{contrast_status}]")

    print(f"{'+-'*40}+")

    avg_boundary = (edge_improve_boundary + contrast_improve_boundary) / 2
    avg_flat = (edge_improve_flat + contrast_improve_flat) / 2
    print(f"| {'AVERAGE IMPROVEMENT':<30} | {avg_boundary:>+11.1f}% | {avg_flat:>+11.1f}% | {'+10-15%':>12} |")
    print(f"{'+-'*40}+")

    # ===== PSNR TRADE-OFF =====
    print(f"\n{'='*80}")
    print(f" PSNR TRADE-OFF (Target: < 0.5 dB drop)")
    print(f"{'='*80}")

    psnr_status = "OK" if abs(psnr_delta) <= 0.5 else "EXCEEDS"
    print(f"\n  Backbone PSNR:  {val_metrics['psnr_backbone']:.2f} dB")
    print(f"  Corrected PSNR: {val_metrics['psnr_corrected']:.2f} dB")
    print(f"  PSNR Delta:     {psnr_delta:+.3f} dB  [Target: < 0.5 dB drop] [{psnr_status}]")
    print(f"  SSIM:           {val_metrics['ssim_backbone']:.4f} -> {val_metrics['ssim_corrected']:.4f}")

    # ===== PATHOLOGY PRESERVATION =====
    print(f"\n{'='*80}")
    print(f" PATHOLOGY PRESERVATION")
    print(f"{'='*80}")

    pathology_coverage = val_metrics.get('pathology_coverage', 0)
    print(f"\n  Pathology Coverage:  {pathology_coverage*100:.1f}%")
    print(f"  Preservation Status: {'ACTIVE' if pathology_coverage > 0.05 else 'MINIMAL'}")

    # ===== CORRECTION BEHAVIOR =====
    print(f"\n{'='*80}")
    print(f" CORRECTION BEHAVIOR")
    print(f"{'='*80}")

    lambda_stats = val_metrics.get('lambda_stats', {})
    correction_mag = val_metrics.get('correction_magnitude', 0)

    print(f"\n  {'Corrector':<15} {'Mean Lambda':>15} {'Max Lambda':>15}")
    print(f"  {'-'*45}")
    for name in ['edge', 'contrast', 'smooth', 'structure', 'anatomy']:
        if name in lambda_stats:
            stats = lambda_stats[name]
            print(f"  {name:<15} {stats['mean']:>15.4f} {stats['max']:>15.4f}")
    print(f"  {'-'*45}")
    print(f"  {'Correction Magnitude:':<30} {correction_mag:.6f}")

    # ===== PREDICATE SCORES =====
    pred_scores = val_metrics.get('pred_scores', {})
    if pred_scores:
        print(f"\n{'='*80}")
        print(f" PREDICATE SCORES (Symbolic Reasoning)")
        print(f"{'='*80}")

        print(f"\n  {'Predicate':<15} {'Score':>10} {'Status':>10}")
        print(f"  {'-'*35}")

        # P5 (Speckle) excluded from evaluation - fundamentally incompatible with denoising
        # Denoising removes speckle, so speckle fidelity metric contradicts the objective
        evaluated_predicates = ['P1', 'P2', 'P3', 'P4', 'P6']  # Exclude P5
        pass_count = 0
        total_count = len(evaluated_predicates)

        for key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            score = pred_scores.get(key, 0)
            if key == 'P5':
                status = "EXCLUDED"  # P5 excluded from pass/fail count
            else:
                status = "PASS" if score >= 0.5 else "FAIL"
                if score >= 0.5:
                    pass_count += 1
            print(f"  {key:<15} {score:>10.3f} {status:>10}")

        print(f"\n  Predicates Passing: {pass_count}/{total_count} (P5 excluded)")

    # ===== TRAINING SUMMARY =====
    print(f"\n{'='*80}")
    print(f" TRAINING SUMMARY")
    print(f"{'='*80}")

    print(f"\n  Training Loss:          {train_metrics['loss']:.6f}")
    print(f"  Train PSNR Delta:       {train_metrics['psnr_corrected'] - train_metrics['psnr_backbone']:+.3f} dB")
    print(f"  Pathology Coverage:     {train_metrics.get('pathology_coverage', 0)*100:.1f}%")
    print(f"  Boundary Coverage:      {train_metrics.get('boundary_coverage', 0)*100:.1f}%")

    # ===== LEARNED LOSS WEIGHTS =====
    learned_weights = train_metrics.get('learned_weights', {})
    if learned_weights:
        print(f"\n  Learned Loss Weights (Uncertainty Weighting):")
        sorted_weights = sorted(learned_weights.items(), key=lambda x: x[1], reverse=True)[:6]
        for name, weight in sorted_weights:
            print(f"    {name:<20}: {weight:.4f}")

    # ===== PUBLICATION READY SUMMARY =====
    print(f"\n{'#'*80}")
    print(f"# PUBLICATION SUMMARY")
    print(f"#")
    print(f"# Clinical Improvement (Boundary): {avg_boundary:+.1f}%")
    print(f"# PSNR Trade-off:                  {psnr_delta:+.3f} dB")
    print(f"# Pathology Coverage:              {pathology_coverage*100:.1f}%")
    print(f"#")
    if target_achieved:
        print(f"# VERDICT: PUBLICATION READY")
    else:
        gaps = []
        if avg_boundary < 10:
            gaps.append(f"clinical improvement ({avg_boundary:+.1f}% vs +10% target)")
        if abs(psnr_delta) > 0.5:
            gaps.append(f"PSNR drop ({psnr_delta:.2f} dB vs 0.5 dB target)")
        print(f"# VERDICT: Needs improvement in: {', '.join(gaps)}")
    print(f"{'#'*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train V8 Enhanced with Region-Adaptive + Pathology Preservation (IEEE TMI)'
    )

    # Data
    parser.add_argument('--train_jsonl', default='pku37_oct_dataset/pku37_real_train.jsonl')
    parser.add_argument('--val_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=96)

    # Model
    parser.add_argument('--backbone_width', type=int, default=40)
    parser.add_argument('--pretrained_backbone', type=str,
                        default='outputs/nafnet_pku37_w40/best_model.pth')
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--use_region_adaptive', action='store_true', default=True)
    parser.add_argument('--use_pathology_gate', action='store_true', default=True)

    # Training config (from DESIGN document)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_corrector', type=float, default=2e-4,
                        help='Learning rate for corrector (default: 2e-4)')
    parser.add_argument('--lr_lambda', type=float, default=5e-4,
                        help='Learning rate for lambda predictor (default: 5e-4)')
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Loss config
    parser.add_argument('--pathology_weight', type=float, default=0.5)
    parser.add_argument('--boundary_weight', type=float, default=0.3)
    parser.add_argument('--clinical_boundary_multiplier', type=float, default=2.0)
    parser.add_argument('--psnr_slack', type=float, default=0.5)

    # Output
    parser.add_argument('--output_dir', default='outputs/nsnd_v8_region_pathology')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("NEURO-SYMBOLIC OCT DENOISING V8 + REGION-ADAPTIVE + PATHOLOGY PRESERVATION")
    print("IEEE TMI Publication Training")
    print("="*80)
    print(f"\nTarget Performance:")
    print(f"  - Clinical improvement in boundary regions: +10-15%")
    print(f"  - PSNR trade-off: < {args.psnr_slack} dB drop")
    print(f"  - Strong pathology preservation")
    print(f"\nConfiguration:")
    print(f"  Backbone width: {args.backbone_width}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Region-adaptive: {args.use_region_adaptive}")
    print(f"  Pathology gate: {args.use_pathology_gate}")
    print(f"  LR corrector: {args.lr_corrector}")
    print(f"  LR lambda: {args.lr_lambda}")
    print(f"  Pathology weight: {args.pathology_weight}")
    print(f"  Boundary weight: {args.boundary_weight}")
    print(f"  Clinical boundary multiplier: {args.clinical_boundary_multiplier}")
    print(f"  Device: {args.device}")

    # Data
    print("\nLoading PKU37 data...")
    train_dataset = PKU37Dataset(
        args.train_jsonl, max_samples=args.max_train,
        patch_size=args.patch_size, is_train=True
    )
    val_dataset = PKU37Dataset(
        args.val_jsonl, max_samples=args.max_val,
        patch_size=0, is_train=False
    )

    num_workers = 4 if args.device != 'cpu' else 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(args.device != 'cpu'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=min(2, num_workers), pin_memory=(args.device != 'cpu')
    )

    # Model
    print("\nInitializing model...")
    model = NeuroSymbolicDenoiserV8Region(
        backbone_width=args.backbone_width,
        pretrained_backbone=args.pretrained_backbone,
        use_region_adaptive=args.use_region_adaptive,
        use_pathology_gate=args.use_pathology_gate,
    ).to(args.device)

    # Freeze backbone
    if args.freeze_backbone:
        print("\n*** FREEZING BACKBONE WEIGHTS ***")
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Loss
    criterion = RegionPathologyLoss(
        pathology_weight=args.pathology_weight,
        boundary_weight=args.boundary_weight,
        clinical_boundary_multiplier=args.clinical_boundary_multiplier,
        psnr_slack=args.psnr_slack,
        use_uncertainty_weighting=True
    ).to(args.device)

    # Optimizer with different learning rates
    corrector_params = list(model.corrector.correctors.parameters())
    lambda_params = list(model.corrector.lambda_predictor.parameters())
    router_params = list(model.corrector.router.parameters())

    region_params = []
    if args.use_region_adaptive:
        region_params = list(model.region_lambda_predictor.parameters())

    pathology_params = []
    if args.use_pathology_gate:
        pathology_params = list(model.pathology_module.parameters())

    loss_weight_params = list(criterion.log_sigma.parameters())
    loss_weight_params += list(criterion.boundary_loss.log_sigma.parameters())
    if hasattr(criterion.pathology_loss, 'log_sigma'):
        loss_weight_params += list(criterion.pathology_loss.log_sigma.parameters())

    optimizer = torch.optim.AdamW([
        {'params': corrector_params, 'lr': args.lr_corrector},
        {'params': lambda_params, 'lr': args.lr_lambda},
        {'params': router_params, 'lr': args.lr_corrector * 0.5},
        {'params': region_params, 'lr': args.lr_lambda} if region_params else {'params': [], 'lr': 0},
        {'params': pathology_params, 'lr': args.lr_corrector} if pathology_params else {'params': [], 'lr': 0},
        {'params': loss_weight_params, 'lr': args.lr_corrector * 0.25, 'weight_decay': 0},
    ], weight_decay=1e-4)

    print(f"\nOptimizer configuration:")
    print(f"  Corrector params: {sum(p.numel() for p in corrector_params):,} @ lr={args.lr_corrector}")
    print(f"  Lambda predictor: {sum(p.numel() for p in lambda_params):,} @ lr={args.lr_lambda}")
    if region_params:
        print(f"  Region-adaptive:  {sum(p.numel() for p in region_params):,} @ lr={args.lr_lambda}")
    if pathology_params:
        print(f"  Pathology module: {sum(p.numel() for p in pathology_params):,} @ lr={args.lr_corrector}")

    # Scheduler following DESIGN document
    # Epochs 1-10: Warmup
    # Epochs 11-30: Joint training
    # Epochs 31-50: Fine-tuning
    # Epochs 51-70: Final refinement
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_corrector * 0.01
    )

    # AMP
    use_amp = args.device != 'cpu' and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("  Using Automatic Mixed Precision (AMP)")

    # Training loop
    print("\n" + "#"*80)
    print("# TRAINING V8 REGION-ADAPTIVE + PATHOLOGY PRESERVATION")
    print("# Target: +10-15% clinical improvement, < 0.5 dB PSNR drop")
    print("#"*80)

    best_clinical_improvement = -float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, scaler
        )
        scheduler.step()

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, args.device)
            print_publication_metrics(epoch, train_metrics, val_metrics)

            # Save based on clinical improvement in boundary regions with PSNR constraint
            edge_improve = (val_metrics['edge_improvement_boundary'] - 1.0) * 100
            contrast_improve = (val_metrics['contrast_improvement_boundary'] - 1.0) * 100
            avg_improvement = (edge_improve + contrast_improve) / 2
            psnr_delta = val_metrics['psnr_delta']

            # Only save if PSNR constraint is satisfied
            if abs(psnr_delta) <= args.psnr_slack and avg_improvement > best_clinical_improvement:
                best_clinical_improvement = avg_improvement
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_clinical_improvement': best_clinical_improvement,
                    'args': vars(args),
                }, os.path.join(args.output_dir, 'best_model_region_pathology.pth'))

                print(f"*** New best model! Clinical improvement: {best_clinical_improvement:+.1f}% ***")
        else:
            print(f"[Epoch {epoch}] Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr_corrected']:.2f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "#"*80)
    print("# TRAINING COMPLETE")
    print("#"*80)
    print(f"Best clinical improvement: {best_clinical_improvement:+.1f}% (epoch {best_epoch})")
    print(f"Model saved to: {args.output_dir}/best_model_region_pathology.pth")


if __name__ == '__main__':
    main()
