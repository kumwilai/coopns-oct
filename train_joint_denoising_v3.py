#!/usr/bin/env python3
"""
Joint Denoising Training with Physics-Enhanced V3 Boundary Detection.

Integrates:
1. PhysicsEnsembleV3 for boundary detection (pretrained or joint training)
2. Adaptive Strength Map for per-pixel denoising strength
3. Enhanced Neuro-Symbolic Framework (5 pillars):
   - Anatomical Knowledge Base (clinical priors)
   - OCT Physics Engine (Beer-Lambert, Fresnel, speckle)
   - Topological Constraints (connectivity, ordering, curvature)
   - Symbolic Rule Engine (explicit reasoning)
   - Interpretable Reports (explainable AI)
4. Layer-specific denoising heads
5. Segmentation-guided attention

Key TMI Contributions:
- Per-pixel noise strength prediction based on layer, boundary proximity, local noise
- First comprehensive neuro-symbolic OCT framework
- First differentiable anatomical knowledge base
- First symbolic rule engine with interpretable confidence scores
- Physics-informed boundary detection (Beer-Lambert, Fresnel, gradient alignment)
"""

import argparse
import gc
import json
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Import V3 boundary model
from physics_enhanced_v3 import (
    PhysicsEnsembleV3,
    PhysicsLossV3,
    boundaries_to_segmentation,
)

# Import Enhanced Neuro-Symbolic Framework
from neuro_symbolic_enhanced import (
    NeuroSymbolicLossEnhanced,
    AnatomicalKnowledgeBase,
    OCTPhysicsEngine,
    TopologicalConstraints,
    SymbolicRuleEngine,
)


# =============================================================================
# Enhanced Neuro-Symbolic Framework (imported from neuro_symbolic_enhanced.py)
# =============================================================================
# The comprehensive 5-pillar neuro-symbolic framework includes:
#
# 1. ANATOMICAL KNOWLEDGE BASE
#    - Clinical thickness priors from literature
#    - Expected reflectivity patterns per layer
#    - Regional anatomy variations (fovea vs peripheral)
#    - Inter-layer thickness correlations
#
# 2. OCT PHYSICS ENGINE
#    - Beer-Lambert attenuation: I(z) = I₀·exp(-μz)
#    - Fresnel reflection: R = ((n₁-n₂)/(n₁+n₂))²
#    - Speckle statistics (Rayleigh distribution)
#    - Learnable tissue optical properties
#
# 3. TOPOLOGICAL CONSTRAINTS
#    - Layer connectivity (no holes/gaps)
#    - Non-crossing boundaries (b₀ < b₁ < b₂ < b₃)
#    - Curvature limits (anatomically plausible)
#    - Boundary smoothness regularization
#
# 4. SYMBOLIC RULE ENGINE
#    - LayerOrderRule: Anatomical layer ordering
#    - ThicknessRule: Clinical thickness norms
#    - IntensityOrderRule: OCT physics reflectivity
#    - Explicit if-then reasoning with confidence scores
#
# 5. INTERPRETABLE REPORTS
#    - Human-readable validation output
#    - Per-rule satisfaction status
#    - Overall confidence scoring
#    - Explainable AI for medical imaging
# =============================================================================


# =============================================================================
# Adaptive Strength Predictor (from TMI v4)
# =============================================================================

class AdaptiveStrengthPredictor(nn.Module):
    """
    Predicts per-pixel denoising strength map.

    Output strength ∈ [0, 1] where:
        - 0 = use only base denoised output
        - 1 = use full layer-specific refinement

    Adapts based on:
        - Layer probabilities
        - Boundary proximity
        - Local noise level
    """

    def __init__(
        self,
        feature_channels: int = 64,
        n_layers: int = 4,
        hidden_channels: int = 32,
        boundary_sigma: float = 10.0,
        min_strength: float = 0.1,
        max_strength: float = 0.9,
        init_bias: float = -1.0,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.boundary_sigma = boundary_sigma
        self.min_strength = min_strength
        self.max_strength = max_strength

        # Input: features + layer_probs + noise_map + boundary_dist
        in_channels = feature_channels + n_layers + 2

        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        # Initialize for low initial strength
        nn.init.normal_(self.predictor[-1].weight, mean=0.0, std=0.1)
        nn.init.constant_(self.predictor[-1].bias, init_bias)

        # Learnable per-layer strength bias
        self.layer_strength_bias = nn.Parameter(torch.zeros(n_layers))

        # Laplacian kernel for noise estimation
        laplacian = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian)

    def estimate_noise_level(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate local noise level using Laplacian."""
        laplacian = F.conv2d(image, self.laplacian_kernel, padding=1)
        laplacian_sq = laplacian ** 2
        local_var = F.avg_pool2d(laplacian_sq, kernel_size=5, stride=1, padding=2)

        B = local_var.shape[0]
        local_var_flat = local_var.view(B, -1)
        max_vals = local_var_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        noise_map = local_var / (max_vals + 1e-8)

        return noise_map

    def compute_boundary_distance(self, boundaries: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Compute distance to nearest boundary for each pixel."""
        B, N, W_bound = boundaries.shape
        device = boundaries.device

        boundaries_px = boundaries * (H - 1)
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)

        # Use large value instead of inf to avoid numerical issues
        min_distance = torch.full((B, 1, H, W), float(H), device=device)

        for n in range(N):
            boundary_n = boundaries_px[:, n, :].view(B, 1, 1, W)
            dist_to_boundary = torch.abs(y_coords.view(1, 1, H, 1) - boundary_n)
            min_distance = torch.minimum(min_distance, dist_to_boundary)

        return min_distance / H

    def forward(
        self,
        features: torch.Tensor,
        layer_probs: torch.Tensor,
        boundaries: torch.Tensor,
        noisy_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict per-pixel denoising strength.

        Args:
            features: NAFNet features [B, C, H, W]
            layer_probs: Segmentation probabilities [B, 4, H, W]
            boundaries: Boundary positions [B, 4, W]
            noisy_image: Input noisy image [B, 1, H, W]

        Returns:
            strength_map: Per-pixel strength [B, 1, H, W]
        """
        B, _, H, W = features.shape

        # Estimate noise level
        noise_map = self.estimate_noise_level(noisy_image)

        # Compute boundary distance
        boundary_dist = self.compute_boundary_distance(boundaries, H, W)

        # Concatenate all inputs
        combined = torch.cat([
            features,
            layer_probs,
            noise_map,
            boundary_dist,
        ], dim=1)

        # Predict base strength
        strength_logits = self.predictor(combined)

        # Apply layer-specific bias
        layer_bias = torch.zeros(B, 1, H, W, device=features.device)
        for c in range(self.n_layers):
            layer_bias += layer_probs[:, c:c+1, :, :] * self.layer_strength_bias[c]

        strength_logits = strength_logits + layer_bias

        # Sigmoid and scale to [min, max]
        strength = torch.sigmoid(strength_logits)
        strength = self.min_strength + (self.max_strength - self.min_strength) * strength

        # Reduce strength near boundaries (preserve edges)
        boundary_weight = torch.exp(-boundary_dist / (self.boundary_sigma / H))
        strength = strength * (1 - 0.3 * boundary_weight)

        return strength


# =============================================================================
# Layer-Specific Denoising Heads
# =============================================================================

class LayerSpecificHead(nn.Module):
    """Single denoising head for one layer type."""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class LayerSpecificDenoiser(nn.Module):
    """Layer-specific denoising with 4 specialized heads."""

    def __init__(self, feature_channels: int = 64, hidden_channels: int = 64):
        super().__init__()

        # One head per layer class
        self.heads = nn.ModuleList([
            LayerSpecificHead(feature_channels + 1, hidden_channels)
            for _ in range(4)
        ])

    def forward(
        self,
        features: torch.Tensor,
        noisy_image: torch.Tensor,
        layer_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply layer-specific denoising.

        Args:
            features: Encoder features [B, C, H, W]
            noisy_image: Input image [B, 1, H, W]
            layer_probs: Segmentation probs [B, 4, H, W]

        Returns:
            refined: Layer-specific refinement [B, 1, H, W]
        """
        B, _, H, W = features.shape

        # Concatenate features with noisy image
        combined = torch.cat([features, noisy_image], dim=1)

        # Apply each head and blend by layer probability
        refined = torch.zeros(B, 1, H, W, device=features.device)
        for c, head in enumerate(self.heads):
            head_output = head(combined)
            refined += layer_probs[:, c:c+1, :, :] * head_output

        return refined


# =============================================================================
# NAFNet-style Denoising Backbone
# =============================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Simplified NAFNet block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = SimpleGate()
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.gate(x)
        x = self.conv2(x)
        return x + residual


class SimpleDenoiser(nn.Module):
    """Simplified denoising backbone (NAFNet-style)."""

    def __init__(self, width: int = 64, num_blocks: int = 4):
        super().__init__()

        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.outro = nn.Conv2d(width, 1, 3, padding=1)

        # Feature output for strength predictor
        self.feature_channels = width

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.intro(x)
        feat = self.blocks(feat)
        out = self.outro(feat)

        if return_features:
            return out, feat
        return out


# =============================================================================
# Joint Denoising Model with V3
# =============================================================================

class JointDenoisingModelV3(nn.Module):
    """
    Joint Denoising + Segmentation Model using PhysicsEnsembleV3.

    Architecture:
    1. PhysicsEnsembleV3: Boundary detection with physics constraints
    2. SimpleDenoiser: Base denoising backbone
    3. LayerSpecificDenoiser: Per-layer refinement heads
    4. AdaptiveStrengthPredictor: Per-pixel denoising strength

    Loss Components:
    - Denoising: L1 + perceptual
    - Boundary: V3 physics loss (Beer-Lambert, Fresnel, gradient)
    - Neuro-symbolic: ordering, thickness, intensity, continuity
    """

    def __init__(
        self,
        boundary_pretrained: Optional[str] = None,
        freeze_boundary: bool = True,
        denoiser_width: int = 64,
        denoiser_blocks: int = 4,
        use_adaptive_strength: bool = True,
        use_layer_heads: bool = True,
    ):
        super().__init__()

        # V3 Boundary model
        self.boundary_model = PhysicsEnsembleV3(hidden_channels=48)

        if boundary_pretrained:
            print(f"Loading pretrained boundary model: {boundary_pretrained}")
            ckpt = torch.load(boundary_pretrained, map_location='cpu', weights_only=False)
            self.boundary_model.load_state_dict(ckpt['model'])
            print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, MAE={ckpt.get('mae', '?')}")

        if freeze_boundary:
            for param in self.boundary_model.parameters():
                param.requires_grad = False
            print("Boundary model frozen")

        # Denoising backbone
        self.denoiser = SimpleDenoiser(width=denoiser_width, num_blocks=denoiser_blocks)

        # Layer-specific heads
        self.use_layer_heads = use_layer_heads
        if use_layer_heads:
            self.layer_heads = LayerSpecificDenoiser(
                feature_channels=denoiser_width,
                hidden_channels=64,
            )

        # Adaptive strength predictor
        self.use_adaptive_strength = use_adaptive_strength
        if use_adaptive_strength:
            self.strength_predictor = AdaptiveStrengthPredictor(
                feature_channels=denoiser_width,
                n_layers=4,
                hidden_channels=32,
            )

    def forward(self, noisy_image: torch.Tensor, return_all: bool = False) -> Dict:
        """
        Forward pass.

        Args:
            noisy_image: [B, 1, H, W] noisy input
            return_all: Return all intermediate outputs

        Returns:
            dict with 'denoised', 'boundaries', 'segmentation', etc.
        """
        B, C, H, W = noisy_image.shape

        # Step 1: Get boundaries and segmentation from V3
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.boundary_model.parameters())):
            boundary_outputs = self.boundary_model(noisy_image, return_aux=True)

        boundaries = boundary_outputs['boundaries']  # [B, 4, W]
        segmentation = boundaries_to_segmentation(boundaries, H, num_classes=4)  # [B, H, W]

        # Convert to probabilities (one-hot)
        seg_probs = F.one_hot(segmentation, num_classes=4).permute(0, 3, 1, 2).float()  # [B, 4, H, W]

        # Step 2: Base denoising
        denoised_base, features = self.denoiser(noisy_image, return_features=True)

        # Step 3: Layer-specific refinement
        if self.use_layer_heads:
            layer_refinement = self.layer_heads(features, noisy_image, seg_probs)

            if self.use_adaptive_strength:
                # Predict per-pixel strength
                strength_map = self.strength_predictor(
                    features, seg_probs, boundaries, noisy_image
                )
                # Blend base and refinement
                denoised = denoised_base + strength_map * layer_refinement
            else:
                denoised = denoised_base + 0.3 * layer_refinement
        else:
            denoised = denoised_base
            strength_map = None

        result = {
            'denoised': denoised,
            'boundaries': boundaries,
            'segmentation': segmentation,
            'seg_probs': seg_probs,
        }

        if return_all:
            result.update({
                'denoised_base': denoised_base,
                'features': features,
                'strength_map': strength_map,
                'boundary_outputs': boundary_outputs,
            })

        return result


# =============================================================================
# Dataset
# =============================================================================

class JointDenoisingDataset(Dataset):
    """Dataset for joint denoising training."""

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None, target_size: Tuple[int, int] = (256, 256)):
        self.samples = []
        self.target_size = target_size

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                # Need both noisy and clean images
                if 'clean_path' in sample or 'image_path' in sample:
                    self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        noisy_path = sample.get('noisy_path', sample.get('image_path'))
        clean_path = sample.get('clean_path', sample.get('image_path'))
        mask_path = sample.get('mask_path')

        noisy = np.array(Image.open(noisy_path).convert('L')) / 255.0
        clean = np.array(Image.open(clean_path).convert('L')) / 255.0

        # Resize
        H, W = self.target_size
        noisy = np.array(Image.fromarray((noisy * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
        clean = np.array(Image.fromarray((clean * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0

        result = {
            'noisy': torch.from_numpy(noisy).float().unsqueeze(0),
            'clean': torch.from_numpy(clean).float().unsqueeze(0),
        }

        if mask_path:
            mask = np.array(Image.open(mask_path))
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((W, H), Image.NEAREST))
            result['mask'] = torch.from_numpy(mask).long()

        return result


# =============================================================================
# Loss Functions
# =============================================================================

class JointLoss(nn.Module):
    """
    Combined loss for joint training with Enhanced Neuro-Symbolic Framework.

    Integrates:
    1. Denoising loss (L1 + perceptual)
    2. Boundary loss (V3 physics)
    3. Enhanced Neuro-Symbolic loss (5 pillars):
       - Anatomical Knowledge Base
       - OCT Physics Engine
       - Topological Constraints
       - Symbolic Rule Engine
       - Interpretable validation
    """

    def __init__(
        self,
        lambda_denoise: float = 1.0,
        lambda_boundary: float = 0.5,
        lambda_neuro_symbolic: float = 0.5,
        # Enhanced NS weights
        lambda_thickness: float = 1.0,
        lambda_reflectivity: float = 0.5,
        lambda_fresnel: float = 0.3,
        lambda_speckle: float = 0.2,
        lambda_topology: float = 0.5,
        lambda_symbolic: float = 0.3,
    ):
        super().__init__()
        self.lambda_denoise = lambda_denoise
        self.lambda_boundary = lambda_boundary
        self.lambda_neuro_symbolic = lambda_neuro_symbolic

        # V3 Physics boundary loss
        self.boundary_loss = PhysicsLossV3()

        # Enhanced Neuro-Symbolic loss (5 pillars)
        self.ns_loss = NeuroSymbolicLossEnhanced(
            lambda_thickness=lambda_thickness,
            lambda_reflectivity=lambda_reflectivity,
            lambda_fresnel=lambda_fresnel,
            lambda_speckle=lambda_speckle,
            lambda_connectivity=lambda_topology,
            lambda_ordering=lambda_topology,
            lambda_curvature=lambda_topology * 0.5,
            lambda_smoothness=lambda_topology,
            lambda_symbolic=lambda_symbolic,
        )

    def forward(
        self,
        outputs: Dict,
        clean: torch.Tensor,
        gt_boundaries: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:

        H = clean.shape[2]
        stats = {}
        noisy = outputs.get('noisy')

        # 1. Denoising loss (L1)
        denoise_loss = F.l1_loss(outputs['denoised'], clean)
        stats['denoise_loss'] = denoise_loss.item()

        # 2. Boundary loss (if GT available)
        boundary_loss = torch.tensor(0.0, device=clean.device)
        if gt_boundaries is not None and valid_mask is not None:
            boundary_loss, boundary_stats = self.boundary_loss(
                outputs['boundary_outputs'], gt_boundaries, valid_mask, H,
                image=noisy, physics_warmup=1.0
            )
            stats.update(boundary_stats)
        stats['boundary_loss'] = boundary_loss.item()

        # 3. Enhanced Neuro-Symbolic loss (5 pillars)
        ns_loss, ns_stats = self.ns_loss(
            image=clean,
            denoised=outputs['denoised'],
            seg_probs=outputs['seg_probs'],
            boundaries=outputs['boundaries'],
            noisy=noisy,
        )
        stats.update(ns_stats)
        stats['ns_loss'] = ns_loss.item()

        # Total
        total_loss = (
            self.lambda_denoise * denoise_loss +
            self.lambda_boundary * boundary_loss +
            self.lambda_neuro_symbolic * ns_loss
        )
        stats['total_loss'] = total_loss.item()

        return total_loss, stats

    def get_interpretable_report(self, outputs: Dict, clean: torch.Tensor) -> Dict:
        """Generate human-interpretable validation report."""
        return self.ns_loss.get_interpretable_report(
            image=clean,
            seg_probs=outputs['seg_probs'],
            boundaries=outputs['boundaries'],
        )


# =============================================================================
# Training Functions
# =============================================================================

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')

    # Clear existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    root_logger.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )


def train_epoch(model, loss_fn, loader, optimizer, device, grad_clip=0.5):
    model.train()
    total_loss = 0
    total_psnr = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad()
        outputs = model(noisy, return_all=True)
        outputs['noisy'] = noisy

        loss, stats = loss_fn(outputs, clean)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        # PSNR (compute without gradients to save memory)
        with torch.no_grad():
            mse = F.mse_loss(outputs['denoised'].detach(), clean)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'PSNR': f"{psnr.item():.1f}dB",
        })

        # Clear outputs to free memory
        del outputs

    n = len(loader)
    return {'loss': total_loss / n, 'psnr': total_psnr / n}


def validate(model, loss_fn, loader, device, generate_report: bool = False):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ns_stats = {}
    report = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Validating")):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            outputs = model(noisy, return_all=True)
            outputs['noisy'] = noisy

            loss, stats = loss_fn(outputs, clean)
            total_loss += loss.item()

            # Accumulate neuro-symbolic stats (skip booleans)
            for k, v in stats.items():
                if isinstance(v, bool):
                    continue  # Skip boolean stats like 'rules_satisfied'
                if k not in total_ns_stats:
                    total_ns_stats[k] = 0
                if isinstance(v, (int, float)):
                    total_ns_stats[k] += v

            mse = F.mse_loss(outputs['denoised'], clean)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.item()

            # Generate interpretable report for first batch
            if generate_report and i == 0:
                report = loss_fn.get_interpretable_report(outputs, clean)

            # Clear to free memory
            del outputs

    n = len(loader)
    result = {
        'loss': total_loss / n,
        'psnr': total_psnr / n,
    }

    # Average NS stats
    for k, v in total_ns_stats.items():
        result[f'avg_{k}'] = v / n

    if report:
        result['interpretable_report'] = report

    return result


def log_interpretable_report(report: Dict):
    """Log the interpretable neuro-symbolic report."""
    logging.info("  Neuro-Symbolic Validation Report:")
    logging.info(f"    Overall Satisfied: {report['overall_satisfied']}")
    logging.info(f"    Overall Confidence: {report['overall_confidence']:.1%}")

    for rule_name, rule_result in report['rules'].items():
        status = "+" if rule_result['satisfied'] else "x"  # Avoid unicode issues
        logging.info(f"    [{status}] {rule_name}: {rule_result['confidence']:.1%}")

    if 'physics' in report:
        logging.info(f"    Physics mu: {report['physics']['attenuation_coefficient']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Joint Denoising Training with V3')
    parser.add_argument('--train_jsonl', default='combined_train.jsonl')
    parser.add_argument('--val_jsonl', default='combined_val.jsonl')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--boundary_pretrained', type=str, default=None,
                        help='Path to pretrained V3 boundary model')
    parser.add_argument('--freeze_boundary', action='store_true',
                        help='Freeze boundary model weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='outputs/joint_denoising_v3')
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    setup_logging(args.output_dir)
    device = torch.device(args.device)

    logging.info("=" * 60)
    logging.info("Joint Denoising Training with Physics-Enhanced V3")
    logging.info("=" * 60)
    logging.info("")
    logging.info("Novel Contributions:")
    logging.info("  1. PhysicsEnsembleV3 for boundary detection")
    logging.info("     - Beer-Lambert depth compensation")
    logging.info("     - Fresnel reflection modeling")
    logging.info("     - Multi-scale boundary fusion")
    logging.info("     - Layer-specific refinement")
    logging.info("")
    logging.info("  2. Enhanced Neuro-Symbolic Framework (5 Pillars):")
    logging.info("     a) Anatomical Knowledge Base")
    logging.info("        - Clinical thickness priors (from literature)")
    logging.info("        - Expected reflectivity patterns")
    logging.info("        - Regional anatomy variations")
    logging.info("     b) OCT Physics Engine")
    logging.info("        - Beer-Lambert attenuation: I(z) = I₀·exp(-μz)")
    logging.info("        - Fresnel reflection: R = ((n₁-n₂)/(n₁+n₂))²")
    logging.info("        - Speckle statistics (Rayleigh distribution)")
    logging.info("     c) Topological Constraints")
    logging.info("        - Layer connectivity (no holes)")
    logging.info("        - Non-crossing boundaries")
    logging.info("        - Curvature limits")
    logging.info("     d) Symbolic Rule Engine")
    logging.info("        - Explicit if-then anatomical rules")
    logging.info("        - Confidence scoring")
    logging.info("     e) Interpretable Reports")
    logging.info("        - Human-readable validation")
    logging.info("        - Explainable AI for medical imaging")
    logging.info("")
    logging.info("  3. Adaptive Strength Map (per-pixel denoising)")
    logging.info("     - Layer-aware strength prediction")
    logging.info("     - Boundary proximity weighting")
    logging.info("     - Local noise estimation")
    logging.info("")
    logging.info("  4. Layer-specific denoising heads (4 specialized heads)")
    logging.info("")

    # Data
    train_ds = JointDenoisingDataset(args.train_jsonl, args.max_train)
    val_ds = JointDenoisingDataset(args.val_jsonl, args.max_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Model
    model = JointDenoisingModelV3(
        boundary_pretrained=args.boundary_pretrained,
        freeze_boundary=args.freeze_boundary,
        denoiser_width=64,
        denoiser_blocks=4,
        use_adaptive_strength=True,
        use_layer_heads=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {total_params:,} total, {trainable_params:,} trainable")

    # Loss and optimizer - move to device for buffer compatibility
    loss_fn = JointLoss().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    logging.info("=" * 60)
    logging.info("Training...")
    logging.info("=" * 60)

    best_psnr = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}")
        logging.info("-" * 40)

        train_stats = train_epoch(model, loss_fn, train_loader, optimizer, device)

        # Generate interpretable report every 5 epochs
        generate_report = (epoch % 5 == 0) or (epoch == 1)
        val_stats = validate(model, loss_fn, val_loader, device, generate_report=generate_report)
        scheduler.step(val_stats['psnr'])

        logging.info(f"Train: loss={train_stats['loss']:.4f}, PSNR={train_stats['psnr']:.2f}dB")
        logging.info(f"Val:   loss={val_stats['loss']:.4f}, PSNR={val_stats['psnr']:.2f}dB")

        # Log neuro-symbolic stats
        if 'avg_thickness_loss' in val_stats:
            logging.info(f"  NS Losses: thickness={val_stats.get('avg_thickness_loss', 0):.4f}, "
                        f"topology={val_stats.get('avg_connectivity_loss', 0):.4f}, "
                        f"fresnel={val_stats.get('avg_fresnel_loss', 0):.4f}")

        # Log interpretable report
        if 'interpretable_report' in val_stats:
            log_interpretable_report(val_stats['interpretable_report'])

        # Save best
        if val_stats['psnr'] > best_psnr:
            best_psnr = val_stats['psnr']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'psnr': best_psnr,
                'ns_confidence': val_stats.get('interpretable_report', {}).get('overall_confidence', 0),
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logging.info(f"  -> New best PSNR! {best_psnr:.2f}dB")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("Early stopping")
                break

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"  LR: {current_lr:.2e}, Patience: {patience_counter}/{args.patience}")

        gc.collect()

    logging.info("=" * 60)
    logging.info("Training Complete!")
    logging.info(f"Best PSNR: {best_psnr:.2f}dB")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
