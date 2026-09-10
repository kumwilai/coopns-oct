#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising Pipeline

A complete, novel pipeline for OCT denoising with:
1. Neural backbone (NAFNet) for initial denoising
2. Symbolic predicates for quality verification
3. Guided refinement when predicates fail
4. Iterative loop until convergence

Key Innovation:
- Denoising as Constraint Satisfaction Problem
- Verifiable quality without ground truth
- Targeted refinement via spatial failure maps

Usage:
    # Training
    python neurosymbolic_oct_pipeline.py --mode train --epochs 10

    # Testing
    python neurosymbolic_oct_pipeline.py --mode test

    # Inference (no GT)
    python neurosymbolic_oct_pipeline.py --mode inference --input image.png

Author: Neuro-Symbolic OCT Framework
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))

# Local imports
from oct_symbolic_knowledge import SymbolicConstraints, LAYER_PROPERTIES
from physics_enhanced_v3 import PhysicsEnsembleV3, boundaries_to_segmentation

# Try NAFNet
try:
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the neuro-symbolic pipeline."""
    # Model architecture
    nafnet_width: int = 64
    refinement_channels: int = 32
    num_boundaries: int = 4

    # Predicate thresholds (calibrated on PKU37)
    speckle_cv: float = 0.40
    speckle_tolerance: float = 0.14
    edge_correlation_threshold: float = 0.5
    min_layer_thickness: float = 0.03

    # Training
    max_refinement_iterations: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 4

    # Loss weights
    lambda_l1: float = 1.0
    lambda_speckle: float = 0.5
    lambda_structure: float = 0.5
    lambda_anatomy: float = 0.1
    lambda_no_degrade: float = 5.0


# =============================================================================
# SYMBOLIC PREDICATES (Lightweight versions for pipeline)
# =============================================================================

class SpecklePredicate(nn.Module):
    """P1: Verify residual follows speckle statistics."""

    def __init__(self, expected_cv: float = 0.40, tolerance: float = 0.14):
        super().__init__()
        self.expected_cv = expected_cv
        self.tolerance = tolerance
        self.window_size = 16

        kernel = torch.ones(1, 1, self.window_size, self.window_size) / (self.window_size ** 2)
        self.register_buffer('kernel', kernel)

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns failure map, loss, and satisfaction status."""
        residual = noisy - denoised

        # Local statistics
        pad = self.window_size // 2
        res_pad = F.pad(residual, (pad, pad, pad, pad), mode='reflect')
        den_pad = F.pad(denoised, (pad, pad, pad, pad), mode='reflect')

        local_var = F.conv2d(res_pad ** 2, self.kernel) - F.conv2d(res_pad, self.kernel) ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-8))
        local_mean = F.conv2d(den_pad, self.kernel).clamp(min=0.05)

        local_cv = F.interpolate(local_std / local_mean, size=noisy.shape[-2:], mode='bilinear', align_corners=False)

        # Failure map
        cv_deviation = (local_cv - self.expected_cv).abs()
        failure_map = (cv_deviation / self.tolerance).clamp(0, 2)

        # Loss
        loss = cv_deviation.mean()

        # Satisfaction
        cv_mean = local_cv.mean()
        satisfied = (cv_mean - self.expected_cv).abs() < self.tolerance

        return {
            'failure_map': failure_map,
            'loss': loss,
            'satisfied': satisfied,
            'cv_mean': cv_mean,
        }


class StructurePredicate(nn.Module):
    """P3: Verify edges are preserved."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_edges(self, img: torch.Tensor) -> torch.Tensor:
        img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(img_pad, self.sobel_x)
        gy = F.conv2d(img_pad, self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns failure map, loss, and satisfaction status."""
        edge_n = self.compute_edges(noisy)
        edge_d = self.compute_edges(denoised)

        # Normalize
        edge_n = edge_n / (edge_n.mean() + 1e-8)
        edge_d = edge_d / (edge_d.mean() + 1e-8)

        # Failure map: where edges lost
        failure_map = F.relu(edge_n - edge_d)
        significant = (edge_n > edge_n.mean()).float()
        failure_map = failure_map * significant
        failure_map = failure_map / (failure_map.max() + 1e-8)

        # Edge correlation
        correlation = (edge_n * edge_d).sum() / (
            torch.sqrt((edge_n ** 2).sum() * (edge_d ** 2).sum()) + 1e-8
        )

        # Loss
        loss = failure_map.mean()

        # Satisfaction
        satisfied = correlation > self.threshold

        return {
            'failure_map': failure_map,
            'loss': loss,
            'satisfied': satisfied,
            'correlation': correlation,
        }


class AnatomyPredicate(nn.Module):
    """P2: Verify boundaries satisfy anatomical constraints."""

    def __init__(self, min_thickness: float = 0.03):
        super().__init__()
        self.min_thickness = min_thickness
        self.ilm_range = (0.05, 0.45)
        self.rpe_range = (0.40, 0.90)
        self.constraint_projector = SymbolicConstraints()

    def forward(self, boundaries: torch.Tensor, height: int) -> Dict[str, torch.Tensor]:
        """Returns failure map, loss, and satisfaction status."""
        B, N, W = boundaries.shape
        device = boundaries.device

        # Check violations
        diffs = boundaries[:, 1:, :] - boundaries[:, :-1, :]
        ordering_viol = F.relu(-diffs + 0.001)
        thickness_viol = F.relu(self.min_thickness - diffs)

        ilm = boundaries[:, 0, :]
        rpe = boundaries[:, -1, :]
        pos_viol = (F.relu(self.ilm_range[0] - ilm) + F.relu(ilm - self.ilm_range[1]) +
                    F.relu(self.rpe_range[0] - rpe) + F.relu(rpe - self.rpe_range[1]))

        total_viol = ordering_viol.sum() + thickness_viol.sum() + pos_viol.sum()

        # Create spatial failure map
        rows = torch.arange(height, device=device, dtype=torch.float32).view(1, 1, -1, 1)
        bounds_px = (boundaries * (height - 1)).unsqueeze(2)
        distances = (rows - bounds_px).abs()
        weights = torch.exp(-distances ** 2 / 200)  # sigma=10
        failure_map = weights.sum(dim=1, keepdim=True) * (total_viol > 0).float().view(B, 1, 1, 1)
        failure_map = failure_map / (failure_map.max() + 1e-8)

        # Loss
        loss = total_viol

        # Satisfaction
        satisfied = total_viol < 0.01

        return {
            'failure_map': failure_map,
            'loss': loss,
            'satisfied': satisfied,
            'total_violation': total_viol,
        }

    def project(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Project boundaries to valid anatomical space."""
        return self.constraint_projector.project_to_valid_space(boundaries)


# =============================================================================
# GUIDED REFINEMENT MODULE
# =============================================================================

class GuidedRefinement(nn.Module):
    """Refinement network guided by predicate failure maps."""

    def __init__(self, channels: int = 32):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Guidance from failure maps (3 maps: speckle, anatomy, structure)
        self.guidance = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid(),
        )

        # Refinement head
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Tanh(),
        )

        # Learnable scale (starts small)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, denoised: torch.Tensor, failure_maps: torch.Tensor) -> torch.Tensor:
        """
        Compute guided residual.

        Args:
            denoised: [B, 1, H, W]
            failure_maps: [B, 3, H, W] - stacked (speckle, anatomy, structure)

        Returns:
            residual: [B, 1, H, W]
        """
        feat = self.features(denoised)
        guide = self.guidance(failure_maps)
        guided_feat = feat * guide
        residual = self.refine(guided_feat)

        # Scale by failure intensity
        total_failure = failure_maps.mean(dim=1, keepdim=True)
        return residual * self.scale * (1 + total_failure)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class NeuroSymbolicOCTPipeline(nn.Module):
    """
    Complete neuro-symbolic OCT denoising pipeline.

    Architecture:
        Noisy → NAFNet → Check Predicates → [Pass?] → Output
                              ↓ No
                         Failure Maps
                              ↓
                      Guided Refinement
                              ↓
                         Re-check → Loop
    """

    def __init__(self, config: PipelineConfig = None):
        super().__init__()

        self.config = config or PipelineConfig()

        # === NEURAL COMPONENTS ===

        # Backbone denoiser
        if HAS_NAFNET:
            self.backbone = NAFNet(
                img_channel=1,
                width=self.config.nafnet_width,
                middle_blk_num=2,
                enc_blk_nums=[2, 2, 2],
                dec_blk_nums=[2, 2, 2],
            )
        else:
            self.backbone = self._simple_backbone()

        # Boundary detector
        self.boundary_model = PhysicsEnsembleV3(
            in_channels=1,
            hidden_channels=48,
            num_boundaries=self.config.num_boundaries,
        )

        # Guided refinement
        self.refinement = GuidedRefinement(channels=self.config.refinement_channels)

        # === SYMBOLIC COMPONENTS ===

        self.P1_speckle = SpecklePredicate(
            expected_cv=self.config.speckle_cv,
            tolerance=self.config.speckle_tolerance,
        )

        self.P2_anatomy = AnatomyPredicate(
            min_thickness=self.config.min_layer_thickness,
        )

        self.P3_structure = StructurePredicate(
            threshold=self.config.edge_correlation_threshold,
        )

        logger.info(f"Pipeline initialized: NAFNet={HAS_NAFNET}, "
                    f"max_iter={self.config.max_refinement_iterations}")

    def _simple_backbone(self):
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def load_pretrained(self, nafnet_path: str = None, boundary_path: str = None):
        """Load pretrained weights."""
        if nafnet_path and os.path.exists(nafnet_path):
            ckpt = torch.load(nafnet_path, map_location='cpu', weights_only=False)
            state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            self.backbone.load_state_dict(state, strict=False)
            logger.info(f"Loaded NAFNet from {nafnet_path}")

        if boundary_path and os.path.exists(boundary_path):
            ckpt = torch.load(boundary_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            self.boundary_model.load_state_dict(state, strict=False)
            logger.info(f"Loaded boundary model from {boundary_path}")

    def get_boundaries(self, img: torch.Tensor) -> torch.Tensor:
        """Get anatomically valid boundaries."""
        with torch.no_grad():
            out = self.boundary_model(img, return_aux=True)
            boundaries = out['boundaries']
        return self.P2_anatomy.project(boundaries)

    def evaluate_predicates(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> Dict[str, any]:
        """Evaluate all predicates."""
        H = noisy.shape[2]

        p1 = self.P1_speckle(noisy, denoised)
        p2 = self.P2_anatomy(boundaries, H)
        p3 = self.P3_structure(noisy, denoised)

        # Stack failure maps
        failure_maps = torch.cat([
            p1['failure_map'],
            p2['failure_map'],
            p3['failure_map'],
        ], dim=1)

        all_satisfied = p1['satisfied'] and p2['satisfied'] and p3['satisfied']

        return {
            'failure_maps': failure_maps,
            'all_satisfied': all_satisfied,
            'p1_speckle': p1,
            'p2_anatomy': p2,
            'p3_structure': p3,
        }

    def forward(
        self,
        noisy: torch.Tensor,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with iterative predicate-guided refinement.

        Args:
            noisy: [B, 1, H, W] noisy input
            return_details: whether to return intermediate results

        Returns:
            dict with 'denoised', 'boundaries', 'iterations', 'satisfied', etc.
        """
        # Step 1: Initial denoising
        denoised = self.backbone(noisy)
        denoised = torch.clamp(denoised, 0, 1)
        denoised_base = denoised.detach().clone()

        # Step 2: Get boundaries
        boundaries = self.get_boundaries(denoised)

        # Step 3: Iterative refinement
        details = []
        for iteration in range(self.config.max_refinement_iterations):
            # Evaluate predicates
            pred_result = self.evaluate_predicates(noisy, denoised, boundaries)

            if return_details:
                details.append({
                    'iteration': iteration,
                    'denoised': denoised.detach().clone(),
                    'satisfied': pred_result['all_satisfied'],
                    'p1_satisfied': pred_result['p1_speckle']['satisfied'],
                    'p2_satisfied': pred_result['p2_anatomy']['satisfied'],
                    'p3_satisfied': pred_result['p3_structure']['satisfied'],
                })

            # Check convergence
            if pred_result['all_satisfied']:
                break

            # Apply guided refinement
            residual = self.refinement(denoised, pred_result['failure_maps'])
            denoised = denoised + residual
            denoised = torch.clamp(denoised, 0, 1)

            # Update boundaries
            boundaries = self.get_boundaries(denoised)

        # Final evaluation
        pred_final = self.evaluate_predicates(noisy, denoised, boundaries)

        # Segmentation from boundaries
        segmentation = boundaries_to_segmentation(
            boundaries, noisy.shape[2], self.config.num_boundaries
        )

        result = {
            'denoised': denoised,
            'denoised_base': denoised_base,
            'boundaries': boundaries,
            'segmentation': segmentation,
            'iterations': iteration + 1,
            'all_satisfied': pred_final['all_satisfied'],
            'p1_speckle': pred_final['p1_speckle'],
            'p2_anatomy': pred_final['p2_anatomy'],
            'p3_structure': pred_final['p3_structure'],
            'failure_maps': pred_final['failure_maps'],
        }

        if return_details:
            result['details'] = details

        return result

    def compute_loss(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        Args:
            noisy: [B, 1, H, W] noisy input
            clean: [B, 1, H, W] clean reference (optional for self-supervised)

        Returns:
            total_loss, loss_dict
        """
        # For training, always apply at least one refinement step
        # Step 1: Initial denoising
        denoised = self.backbone(noisy)
        denoised = torch.clamp(denoised, 0, 1)
        denoised_base = denoised.detach().clone()

        # Step 2: Get boundaries (no grad needed for this)
        boundaries = self.get_boundaries(denoised)

        # Step 3: Always apply refinement during training (for gradient flow)
        pred_result = self.evaluate_predicates(noisy, denoised, boundaries)

        # Apply guided refinement
        residual = self.refinement(denoised, pred_result['failure_maps'])
        denoised = denoised + residual
        denoised = torch.clamp(denoised, 0, 1)

        # Re-evaluate predicates
        boundaries = self.get_boundaries(denoised)
        pred_final = self.evaluate_predicates(noisy, denoised, boundaries)

        outputs = {
            'denoised': denoised,
            'denoised_base': denoised_base,
            'p1_speckle': pred_final['p1_speckle'],
            'p2_anatomy': pred_final['p2_anatomy'],
            'p3_structure': pred_final['p3_structure'],
        }

        losses = {}

        # Supervised L1 loss (if clean available)
        if clean is not None:
            losses['l1'] = F.l1_loss(outputs['denoised'], clean)

            # Don't degrade from backbone
            base_l1 = F.l1_loss(outputs['denoised_base'], clean)
            degradation = F.relu(losses['l1'] - base_l1)
            losses['no_degrade'] = degradation
        else:
            losses['l1'] = torch.tensor(0.0, device=noisy.device)
            losses['no_degrade'] = torch.tensor(0.0, device=noisy.device)

        # Self-supervised predicate losses
        losses['speckle'] = outputs['p1_speckle']['loss']
        losses['structure'] = outputs['p3_structure']['loss']
        losses['anatomy'] = outputs['p2_anatomy']['loss']

        # Total loss
        total = (
            self.config.lambda_l1 * losses['l1'] +
            self.config.lambda_speckle * losses['speckle'] +
            self.config.lambda_structure * losses['structure'] +
            self.config.lambda_anatomy * losses['anatomy'] +
            self.config.lambda_no_degrade * losses['no_degrade']
        )

        losses['total'] = total

        # Convert to float for logging
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

        return total, loss_dict


# =============================================================================
# DATASET
# =============================================================================

class PKU37Dataset(Dataset):
    """PKU37 OCT denoising dataset."""

    def __init__(self, jsonl_path: str, max_samples: int = None):
        self.pairs = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data = json.loads(line)
                self.pairs.append({
                    'clean': data['clean_path'],
                    'noisy': data['noisy_path'],
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        clean = Image.open(pair['clean']).convert('L')
        noisy = Image.open(pair['noisy']).convert('L')

        clean = torch.from_numpy(np.array(clean).astype(np.float32) / 255.0)
        noisy = torch.from_numpy(np.array(noisy).astype(np.float32) / 255.0)

        return {
            'clean': clean.unsqueeze(0),
            'noisy': noisy.unsqueeze(0),
        }


# =============================================================================
# TRAINING AND TESTING
# =============================================================================

def train(args):
    """Train the pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")

    # Config
    config = PipelineConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    # Model
    model = NeuroSymbolicOCTPipeline(config).to(device)
    model.load_pretrained(
        nafnet_path=args.nafnet_ckpt,
        boundary_path=args.boundary_ckpt,
    )

    # Freeze backbone, train only refinement
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen, training refinement only")

    # Dataset
    train_dataset = PKU37Dataset(args.train_jsonl, max_samples=args.max_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    # Training loop
    logger.info(f"Training for {args.epochs} epochs on {len(train_dataset)} samples")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            loss, loss_dict = model.compute_loss(noisy, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss_dict)
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'l1': f"{loss_dict['l1']:.4f}",
            })

        # Epoch summary
        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        logger.info(f"Epoch {epoch+1}: " + ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()]))

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"pipeline_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'config': config,
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")


def test(args):
    """Test the pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing on {device}")

    # Model
    model = NeuroSymbolicOCTPipeline().to(device)
    model.load_pretrained(
        nafnet_path=args.nafnet_ckpt,
        boundary_path=args.boundary_ckpt,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    model.eval()

    # Dataset
    test_dataset = PKU37Dataset(args.test_jsonl, max_samples=args.max_test)

    # Test
    results = []
    for i in tqdm(range(len(test_dataset)), desc="Testing"):
        sample = test_dataset[i]
        noisy = sample['noisy'].unsqueeze(0).to(device)
        clean = sample['clean'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(noisy, return_details=True)

        # PSNR
        psnr_base = 10 * torch.log10(1.0 / F.mse_loss(outputs['denoised_base'], clean)).item()
        psnr_refined = 10 * torch.log10(1.0 / F.mse_loss(outputs['denoised'], clean)).item()

        results.append({
            'psnr_base': psnr_base,
            'psnr_refined': psnr_refined,
            'improvement': psnr_refined - psnr_base,
            'iterations': outputs['iterations'],
            'all_satisfied': outputs['all_satisfied'],
            'p1_ok': outputs['p1_speckle']['satisfied'].item() if torch.is_tensor(outputs['p1_speckle']['satisfied']) else outputs['p1_speckle']['satisfied'],
            'p2_ok': outputs['p2_anatomy']['satisfied'].item() if torch.is_tensor(outputs['p2_anatomy']['satisfied']) else outputs['p2_anatomy']['satisfied'],
            'p3_ok': outputs['p3_structure']['satisfied'].item() if torch.is_tensor(outputs['p3_structure']['satisfied']) else outputs['p3_structure']['satisfied'],
        })

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\nPSNR (Backbone):  {np.mean([r['psnr_base'] for r in results]):.2f} ± {np.std([r['psnr_base'] for r in results]):.2f} dB")
    print(f"PSNR (Refined):   {np.mean([r['psnr_refined'] for r in results]):.2f} ± {np.std([r['psnr_refined'] for r in results]):.2f} dB")
    print(f"Improvement:      {np.mean([r['improvement'] for r in results]):+.3f} dB")
    print(f"\nPredicate Satisfaction:")
    print(f"  P1 (Speckle):   {np.mean([r['p1_ok'] for r in results]):.1%}")
    print(f"  P2 (Anatomy):   {np.mean([r['p2_ok'] for r in results]):.1%}")
    print(f"  P3 (Structure): {np.mean([r['p3_ok'] for r in results]):.1%}")
    print(f"  All satisfied:  {np.mean([r['all_satisfied'] for r in results]):.1%}")
    print(f"\nAverage iterations: {np.mean([r['iterations'] for r in results]):.1f}")


def inference(args):
    """Run inference on a single image (no GT needed)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = NeuroSymbolicOCTPipeline().to(device)
    model.load_pretrained(
        nafnet_path=args.nafnet_ckpt,
        boundary_path=args.boundary_ckpt,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=False)

    model.eval()

    # Load image
    img = Image.open(args.input).convert('L')
    noisy = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
    noisy = noisy.unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(noisy, return_details=True)

    # Results (NO GT NEEDED!)
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS (Verified without Ground Truth)")
    print("=" * 60)
    print(f"\nIterations used: {outputs['iterations']}")
    print(f"\nPredicate Verification:")
    print(f"  P1 (Speckle):   {'✓' if outputs['p1_speckle']['satisfied'] else '✗'} "
          f"(CV={outputs['p1_speckle']['cv_mean'].item():.3f})")
    print(f"  P2 (Anatomy):   {'✓' if outputs['p2_anatomy']['satisfied'] else '✗'}")
    print(f"  P3 (Structure): {'✓' if outputs['p3_structure']['satisfied'] else '✗'} "
          f"(corr={outputs['p3_structure']['correlation'].item():.3f})")
    print(f"\n  ALL SATISFIED: {'✓ VERIFIED' if outputs['all_satisfied'] else '✗ NEEDS REVIEW'}")

    # Save output
    if args.output:
        denoised = outputs['denoised'][0, 0].cpu().numpy()
        denoised = (denoised * 255).astype(np.uint8)
        Image.fromarray(denoised).save(args.output)
        print(f"\nSaved denoised image to {args.output}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Neuro-Symbolic OCT Denoising Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'inference'])

    # Paths
    parser.add_argument('--train_jsonl', type=str,
                        default='/home/kumwilai/OCT/pku37_oct_dataset/weights_pku37_analysis_train.jsonl')
    parser.add_argument('--test_jsonl', type=str,
                        default='/home/kumwilai/OCT/pku37_oct_dataset/weights_pku37_analysis_val.jsonl')
    parser.add_argument('--nafnet_ckpt', type=str,
                        default='/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth')
    parser.add_argument('--boundary_ckpt', type=str,
                        default='/home/kumwilai/OCT/best_boundary_model_v4.pth')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='/home/kumwilai/OCT/outputs/neurosymbolic')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_test', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--freeze_backbone', action='store_true')

    # Inference
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'inference':
        if not args.input:
            parser.error("--input required for inference mode")
        inference(args)


if __name__ == "__main__":
    main()
