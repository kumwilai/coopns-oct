#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising V6 Training Script

V6 integrates all improvements:
1. Fixed quality_aligned_loss.py (with P7 predicate, soft quality constraint)
2. Fixed powerful_correctors.py (with zero initialization)
3. Adaptive lambda capping from train_v5.py
4. Optional QualityGuidedCorrector from quality_guided_corrector.py

Features:
- Command-line flag --corrector-type: "powerful" (default) or "quality_guided"
- Comprehensive metrics including all 7 predicates (P1-P7)
- Detailed tracking: PSNR/SSIM deltas, predicates, lambda stats, correction magnitude
- Best model selection based on: (avg_predicates + quality_bonus)
  where quality_bonus = max(0, psnr_delta) * 0.5

Hyperparameters (same as train_v5.py):
- BATCH_SIZE = 4
- EPOCHS = 25
- TRAIN_SAMPLES = 100
- VAL_SAMPLES = 10
- PATCH_SIZE = 96
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# MEMORY MANAGEMENT UTILITIES
# =============================================================================

def get_gpu_memory_info():
    """Get GPU memory information if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated
        }
    return None

def force_memory_cleanup():
    """Force memory cleanup for both CPU and GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Import from existing modules
from neuro_symbolic_v2 import (
    BackboneWrapper, AdaptiveLambdaPredictor, SymbolicPredicates,
    OCTDataset, _NUM_THREADS, clear_memory
)
from powerful_correctors import PowerfulAdaptiveCorrectorWithLambda
from quality_aligned_loss import QualityAlignedLoss, CleanReferencedPredicates
from quality_guided_corrector import QualityGuidedCorrector, create_quality_guided_corrector
from torch.utils.data import DataLoader


# =============================================================================
# V6 MODEL WITH CORRECTOR SELECTION
# =============================================================================

class NeuroSymbolicDenoiserV6(nn.Module):
    """
    V6: Unified model supporting multiple corrector architectures.

    Architecture:
    1. Backbone: NAFNet (frozen, 7.5M params)
    2. Lambda Predictor: Per-pixel correction strength (35K params)
    3. Corrector: Either PowerfulAdaptiveCorrectorWithLambda or QualityGuidedCorrector

    Key features:
    - Switchable corrector architecture via corrector_type parameter
    - Clean-referenced predicates with P7 (MSE proximity)
    - Adaptive lambda capping to prevent over-correction
    """

    def __init__(self, backbone_type: str = 'nafnet', width: int = 64,
                 corrector_hidden_dim: int = 128,
                 corrector_type: str = 'powerful'):
        """
        Initialize V6 model.

        Args:
            backbone_type: Type of backbone ('nafnet')
            width: Backbone width
            corrector_hidden_dim: Hidden dimension for correctors
            corrector_type: 'powerful' or 'quality_guided'
        """
        super().__init__()

        self.corrector_type = corrector_type

        # Backbone (shared)
        self.backbone = BackboneWrapper(backbone_type=backbone_type, width=width)

        # Standard predicates for failure map computation
        self.predicates = SymbolicPredicates()

        # Lambda predictor (used by both correctors)
        self.lambda_predictor = AdaptiveLambdaPredictor()

        # Initialize corrector based on type
        if corrector_type == 'powerful':
            self.corrector = PowerfulAdaptiveCorrectorWithLambda(
                enc1_channels=width,
                enc2_channels=width * 2,
                hidden_dim=corrector_hidden_dim
            )
            print(f"Using PowerfulAdaptiveCorrectorWithLambda")
        elif corrector_type == 'quality_guided':
            self.corrector = create_quality_guided_corrector({
                'in_channels': 1,  # OCT is grayscale
                'enc1_channels': width,
                'enc2_channels': width * 2,
                'hidden_dim': corrector_hidden_dim,
                'refiner_hidden': corrector_hidden_dim // 2,
                'max_lambda': 1.0,
                'num_predicates': 5
            })
            print(f"Using QualityGuidedCorrector")
        else:
            raise ValueError(f"Unknown corrector_type: {corrector_type}. "
                           f"Choose 'powerful' or 'quality_guided'")

    def load_pretrained_backbone(self, path: str):
        """Load pretrained backbone weights."""
        return self.backbone.load_pretrained(path)

    def cap_lambda_maps(self, lambda_maps: Dict, max_lambda: float = 0.2) -> Dict:
        """Cap lambda values to prevent over-correction.

        Args:
            lambda_maps: Dictionary of lambda maps for each corrector
            max_lambda: Maximum allowed lambda value

        Returns:
            Dictionary of capped lambda maps
        """
        capped = {}
        for name, lmap in lambda_maps.items():
            capped[name] = lmap.clamp(0, max_lambda)
        return capped

    def forward(self, noisy: torch.Tensor, max_lambda: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            noisy: Noisy input tensor [B, 1, H, W]
            max_lambda: Maximum lambda value for capping

        Returns:
            corrected: Final corrected output
            backbone_out: Backbone-only output (for quality comparison)
            info: Dictionary with lambda maps, correction magnitude, etc.
        """
        # 1. Backbone denoising
        backbone_out, backbone_features = self.backbone(noisy, return_features=True)

        # 2. Compute failure maps (using noisy reference for WHERE to correct)
        pred_with_maps = self.predicates(noisy, backbone_out, return_failure_maps=True)
        failure_maps = {
            name: pred_with_maps[name]['failure_map']
            for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        }

        # 3. Predict lambda maps
        lambda_maps = self.lambda_predictor(backbone_out, failure_maps)

        # 3.5. Cap lambda maps to prevent over-correction
        lambda_maps_uncapped = lambda_maps
        lambda_maps = self.cap_lambda_maps(lambda_maps, max_lambda=max_lambda)

        # 4. Apply corrections based on corrector type
        if self.corrector_type == 'powerful':
            corrected, corrector_info = self.corrector(
                backbone_out, noisy, lambda_maps, backbone_features,
                failure_maps=failure_maps,
                return_individual=self.training
            )
        else:  # quality_guided
            # Pass lambda_maps to integrate trained lambda predictor with corrector
            corrected, corrector_info = self.corrector(
                backbone_out, noisy, backbone_features, failure_maps,
                external_lambda_maps=lambda_maps  # NEW: integrate trained lambda maps
            )
            # Adapt info structure for QualityGuidedCorrector
            corrector_info['correction_magnitude'] = corrector_info.get(
                'total_correction', torch.zeros_like(backbone_out)
            ).abs().mean().item() if 'total_correction' in corrector_info else 0.0

        # Compute capping statistics
        total_values = 0
        capped_values = 0
        for name in lambda_maps.keys():
            uncapped = lambda_maps_uncapped[name]
            capped = lambda_maps[name]
            total_values += uncapped.numel()
            capped_values += (uncapped > max_lambda).sum().item()

        capped_pct = (capped_values / total_values * 100) if total_values > 0 else 0.0

        info = {
            'lambda_maps': lambda_maps,
            'lambda_maps_uncapped': lambda_maps_uncapped,
            'correction_magnitude': corrector_info.get('correction_magnitude', 0),
            'raw_corrections': corrector_info.get('raw_corrections', {}),
            'max_lambda': max_lambda,
            'capped_pct': capped_pct,
            'corrector_type': self.corrector_type,
        }

        # Add quality_guided specific info
        if self.corrector_type == 'quality_guided':
            info['clean_estimate'] = corrector_info.get('clean_estimate', None)
            info['confidence'] = corrector_info.get('confidence', None)

        return corrected, backbone_out, info


# =============================================================================
# TRAINER V6
# =============================================================================

class TrainerV6:
    """
    Trainer for V6 with comprehensive metrics tracking.

    Features:
    - Quality-aligned loss with P7 predicate
    - Adaptive lambda capping
    - Comprehensive logging of all 7 predicates
    - PSNR/SSIM delta tracking
    - Best model selection with combined score
    """

    def __init__(self, model: NeuroSymbolicDenoiserV6,
                 train_dataset: OCTDataset,
                 val_dataset: OCTDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4,
                 finetune_backbone: bool = False,
                 num_workers: int = 2,
                 use_amp: bool = True,
                 gradient_accumulation_steps: int = 1):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.finetune_backbone = finetune_backbone
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP) training")

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Freeze backbone
        if not finetune_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
                {'params': model.corrector.parameters(), 'lr': lr * 2},
            ]
            print(f"Frozen backbone: training corrector (lr={lr*2:.0e}) + lambda (lr={lr*5:.0e})")
        else:
            param_groups = [
                {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
                {'params': model.corrector.parameters(), 'lr': lr * 2},
                {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            ]
            print(f"Joint training: backbone (lr={lr*0.1:.0e}) + corrector (lr={lr*2:.0e}) + lambda (lr={lr*5:.0e})")

        # STRONGER regularization to prevent overfitting
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)  # 10x stronger

        # Quality-aligned loss - uses the updated defaults from the class
        self.loss_fn = QualityAlignedLoss()  # Use defaults: lambda_pred=0.1, lambda_recon=100.0

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=25, eta_min=lr * 0.01
        )

        self.best_score = 0
        self.best_psnr_delta = -float('inf')

        # Adaptive lambda capping parameters
        # CONSERVATIVE values to prevent overfitting/over-correction
        self.max_lambda = 0.10  # Start VERY conservative
        self.max_lambda_min = 0.05  # Minimum allowed (very small)
        self.max_lambda_max = 0.20  # Maximum allowed (don't go higher)
        self.val_max_lambda = 0.08  # Even more conservative for validation
        self.psnr_delta_history = []  # Rolling history of PSNR deltas
        self.psnr_delta_window = 10  # Number of batches for rolling average

        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {total:,} total, {trainable:,} trainable")
        print(f"Training: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        print(f"Corrector type: {model.corrector_type}")

    def _safe_item(self, value):
        """Safely extract scalar value from tensor or return as-is."""
        if isinstance(value, torch.Tensor):
            return value.item()
        return value

    def train_epoch(self) -> Dict:
        """Train one epoch with comprehensive metrics and memory optimization."""
        self.model.train()

        metrics = {
            'loss': [], 'pred_loss': [], 'quality_loss': [], 'recon_loss': [],
            'psnr_backbone': [], 'psnr_corrected': [], 'quality_improved': [],
            'lambda_edge': [], 'lambda_contrast': [], 'lambda_sharpness': [],
            'correction_mag': [], 'max_lambda': [], 'capped_pct': [],
            'P1': [], 'P2': [], 'P3': [], 'P4': [], 'P5': [], 'P6': [], 'P7': [],
        }

        for batch_idx, batch in enumerate(self.train_loader):
            noisy = batch['noisy']
            clean = batch['clean']

            # Zero gradients at appropriate steps for gradient accumulation
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)  # More memory efficient

            try:
                # Forward with adaptive max_lambda and optional AMP
                if self.use_amp:
                    with autocast():
                        corrected, backbone_out, info = self.model(noisy, max_lambda=self.max_lambda)

                        # Quality-aligned loss (includes P7)
                        loss_dict = self.loss_fn(
                            corrected, backbone_out, clean, noisy,
                            lambda_maps=info['lambda_maps']
                        )
                        loss = loss_dict['total']

                        # CRITICAL: Oracle-supervised training for QualityGuidedCorrector
                        if self.model.corrector_type == 'quality_guided' and 'clean_estimate' in info:
                            # === KEY FIX 1: Direct Residual Supervision ===
                            # Teach the model EXACTLY which direction to correct
                            ideal_residual = clean - backbone_out  # What we SHOULD add
                            predicted_residual = info['clean_estimate'] - backbone_out  # What model predicts

                            # Strong L1 supervision on residual direction (MOST IMPORTANT)
                            residual_supervision = F.l1_loss(predicted_residual, ideal_residual) * 50.0

                            # === KEY FIX 2: PSNR Improvement Loss ===
                            # Directly penalize PSNR degradation
                            with torch.no_grad():
                                mse_backbone = F.mse_loss(backbone_out, clean)
                            mse_corrected = F.mse_loss(corrected, clean)
                            psnr_degradation_penalty = F.relu(mse_corrected - mse_backbone) * 200.0

                            # === KEY FIX 3: Confidence Calibration ===
                            # High confidence should only be where clean_estimate is better
                            if 'confidence' in info:
                                confidence = info['confidence']
                                # Ideal confidence: high where our estimate is better than backbone
                                est_err = (info['clean_estimate'] - clean).abs()
                                back_err = (backbone_out - clean).abs()
                                # Should be confident (1) where est_err < back_err, else 0
                                ideal_confidence = (back_err > est_err).float()
                                confidence_loss = F.binary_cross_entropy(
                                    confidence.clamp(1e-6, 1-1e-6),
                                    ideal_confidence
                                ) * 5.0
                            else:
                                confidence_loss = torch.tensor(0.0, device=loss.device)

                            # Clean estimate L1 (weaker now, residual_supervision is stronger)
                            clean_est_loss = F.l1_loss(info['clean_estimate'], clean) * 5.0

                            # Total auxiliary losses
                            loss = loss + residual_supervision + psnr_degradation_penalty + confidence_loss + clean_est_loss

                            loss_dict['residual_supervision'] = residual_supervision.detach()
                            loss_dict['psnr_degradation_penalty'] = psnr_degradation_penalty.detach()
                            loss_dict['confidence_loss'] = confidence_loss.detach()
                            loss_dict['clean_est_loss'] = clean_est_loss.detach()

                        loss = loss / self.gradient_accumulation_steps
                else:
                    corrected, backbone_out, info = self.model(noisy, max_lambda=self.max_lambda)

                    # Quality-aligned loss (includes P7)
                    loss_dict = self.loss_fn(
                        corrected, backbone_out, clean, noisy,
                        lambda_maps=info['lambda_maps']
                    )
                    loss = loss_dict['total']

                    # CRITICAL: Oracle-supervised training for QualityGuidedCorrector
                    if self.model.corrector_type == 'quality_guided' and 'clean_estimate' in info:
                        # === KEY FIX 1: Direct Residual Supervision ===
                        ideal_residual = clean - backbone_out
                        predicted_residual = info['clean_estimate'] - backbone_out
                        residual_supervision = F.l1_loss(predicted_residual, ideal_residual) * 50.0

                        # === KEY FIX 2: PSNR Improvement Loss ===
                        with torch.no_grad():
                            mse_backbone = F.mse_loss(backbone_out, clean)
                        mse_corrected = F.mse_loss(corrected, clean)
                        psnr_degradation_penalty = F.relu(mse_corrected - mse_backbone) * 200.0

                        # === KEY FIX 3: Confidence Calibration ===
                        if 'confidence' in info:
                            confidence = info['confidence']
                            est_err = (info['clean_estimate'] - clean).abs()
                            back_err = (backbone_out - clean).abs()
                            ideal_confidence = (back_err > est_err).float()
                            confidence_loss = F.binary_cross_entropy(
                                confidence.clamp(1e-6, 1-1e-6),
                                ideal_confidence
                            ) * 5.0
                        else:
                            confidence_loss = torch.tensor(0.0, device=loss.device)

                        clean_est_loss = F.l1_loss(info['clean_estimate'], clean) * 5.0

                        loss = loss + residual_supervision + psnr_degradation_penalty + confidence_loss + clean_est_loss

                        loss_dict['residual_supervision'] = residual_supervision.detach()
                        loss_dict['psnr_degradation_penalty'] = psnr_degradation_penalty.detach()
                        loss_dict['confidence_loss'] = confidence_loss.detach()
                        loss_dict['clean_est_loss'] = clean_est_loss.detach()

                    loss = loss / self.gradient_accumulation_steps

                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
                    force_memory_cleanup()
                    continue

                # Backward with optional AMP scaling
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step at appropriate intervals
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                # Collect metrics (with safe extraction)
                metrics['loss'].append(self._safe_item(loss))
                metrics['pred_loss'].append(self._safe_item(loss_dict['pred_loss']))
                metrics['quality_loss'].append(self._safe_item(loss_dict['quality_loss']))
                metrics['recon_loss'].append(self._safe_item(loss_dict['recon_loss']))
                metrics['psnr_backbone'].append(self._safe_item(loss_dict['psnr_backbone']))
                metrics['psnr_corrected'].append(self._safe_item(loss_dict['psnr_corrected']))
                metrics['quality_improved'].append(float(self._safe_item(loss_dict['quality_improved'])))
                metrics['lambda_edge'].append(self._safe_item(loss_dict['lambda_edge_mean']))
                metrics['lambda_contrast'].append(self._safe_item(loss_dict['lambda_contrast_mean']))
                metrics['lambda_sharpness'].append(self._safe_item(loss_dict['lambda_sharpness_mean']))
                metrics['correction_mag'].append(info.get('correction_magnitude', 0))
                metrics['max_lambda'].append(self.max_lambda)
                metrics['capped_pct'].append(info.get('capped_pct', 0))

                # All 7 predicates
                for i in range(1, 8):
                    metrics[f'P{i}'].append(self._safe_item(loss_dict[f'P{i}']))

                # Adaptive lambda scaling based on PSNR delta
                psnr_delta = loss_dict['psnr_corrected'] - loss_dict['psnr_backbone']
                self.psnr_delta_history.append(psnr_delta)

                # Keep only last N batches
                if len(self.psnr_delta_history) > self.psnr_delta_window:
                    self.psnr_delta_history.pop(0)

                # Adjust max_lambda based on rolling average
                if len(self.psnr_delta_history) >= self.psnr_delta_window:
                    avg_psnr_delta = np.mean(self.psnr_delta_history)

                    if avg_psnr_delta < -0.5:
                        # Corrections are degrading quality - reduce max_lambda by 10%
                        self.max_lambda = max(self.max_lambda_min, self.max_lambda * 0.9)
                    elif avg_psnr_delta > 0:
                        # Corrections are improving quality - gradually increase max_lambda
                        self.max_lambda = min(self.max_lambda_max, self.max_lambda * 1.02)

                # Clean up intermediate tensors after each batch
                del corrected, backbone_out, info, loss_dict, loss

            except Exception as e:
                print(f"ERROR in batch {batch_idx}: {e}")
                force_memory_cleanup()
                continue

            # Periodic memory cleanup every 10 batches
            if (batch_idx + 1) % 10 == 0:
                gc.collect()

        # End of epoch memory cleanup
        force_memory_cleanup()

        # Safely compute means (handle empty lists)
        result = {}
        for k, v in metrics.items():
            if len(v) > 0:
                result[k] = np.mean(v)
            else:
                result[k] = 0.0

        return result

    def validate(self) -> Dict:
        """Validate the model with comprehensive metrics."""
        self.model.eval()

        metrics = {
            'psnr_backbone': [], 'psnr_corrected': [],
            'ssim_backbone': [], 'ssim_corrected': [],
            'quality_improved': [],
            'lambda_edge': [], 'lambda_contrast': [], 'lambda_sharpness': [],
            'correction_mag': [], 'max_lambda': [], 'capped_pct': [],
            'P1': [], 'P2': [], 'P3': [], 'P4': [], 'P5': [], 'P6': [], 'P7': [],
        }

        with torch.no_grad():
            for batch in self.val_loader:
                noisy = batch['noisy']
                clean = batch['clean']

                try:
                    # Forward with CONSERVATIVE max_lambda for validation
                    val_lambda = getattr(self, 'val_max_lambda', self.max_lambda * 0.5)
                    corrected, backbone_out, info = self.model(noisy, max_lambda=val_lambda)

                    # Compute loss for metrics
                    loss_dict = self.loss_fn(
                        corrected, backbone_out, clean, noisy,
                        lambda_maps=info['lambda_maps']
                    )

                    # Collect metrics (with safe extraction)
                    metrics['psnr_backbone'].append(self._safe_item(loss_dict['psnr_backbone']))
                    metrics['psnr_corrected'].append(self._safe_item(loss_dict['psnr_corrected']))
                    metrics['ssim_backbone'].append(self._safe_item(loss_dict['ssim_backbone']))
                    metrics['ssim_corrected'].append(self._safe_item(loss_dict['ssim_corrected']))
                    metrics['quality_improved'].append(float(self._safe_item(loss_dict['quality_improved'])))
                    metrics['lambda_edge'].append(self._safe_item(loss_dict['lambda_edge_mean']))
                    metrics['lambda_contrast'].append(self._safe_item(loss_dict['lambda_contrast_mean']))
                    metrics['lambda_sharpness'].append(self._safe_item(loss_dict['lambda_sharpness_mean']))
                    metrics['correction_mag'].append(info.get('correction_magnitude', 0))
                    metrics['max_lambda'].append(val_lambda)
                    metrics['capped_pct'].append(info.get('capped_pct', 0))

                    # All 7 predicates
                    for i in range(1, 8):
                        metrics[f'P{i}'].append(self._safe_item(loss_dict[f'P{i}']))

                    # Clean up
                    del corrected, backbone_out, info, loss_dict

                except Exception as e:
                    print(f"ERROR in validation: {e}")
                    force_memory_cleanup()
                    continue

        # End of validation cleanup
        force_memory_cleanup()

        # Safely compute means
        result = {}
        for k, v in metrics.items():
            if len(v) > 0:
                result[k] = np.mean(v)
            else:
                result[k] = 0.0

        return result

    def train(self, epochs: int = 25, val_every: int = 5, output_dir: str = 'outputs/nsnd_v6'):
        """Training loop with comprehensive logging."""
        print('\n' + '#'*70)
        print('# NEURO-SYMBOLIC V6 (UNIFIED TRAINING WITH ALL IMPROVEMENTS)')
        print(f'# Corrector: {self.model.corrector_type}')
        print('#'*70)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train = self.train_epoch()
            train_time = time.time() - t0

            # Print comprehensive training metrics
            psnr_delta = train['psnr_corrected'] - train['psnr_backbone']
            quality_rate = train['quality_improved'] * 100

            print(f'\n[EPOCH {epoch}]')
            print(f'  Loss: {train["loss"]:.4f} (pred={train["pred_loss"]:.3f}, '
                  f'quality={train["quality_loss"]:.4f}, recon={train["recon_loss"]:.4f})')
            print(f'  PSNR: backbone={train["psnr_backbone"]:.2f}, '
                  f'corrected={train["psnr_corrected"]:.2f} (delta={psnr_delta:+.3f})')
            print(f'  Quality improved: {quality_rate:.1f}% of pixels')
            print(f'  Lambda: edge={train["lambda_edge"]:.3f}, '
                  f'contrast={train["lambda_contrast"]:.3f}, '
                  f'sharp={train["lambda_sharpness"]:.3f}')
            print(f'  Lambda capping: max_lambda={self.max_lambda:.3f}, '
                  f'capped={train["capped_pct"]:.1f}%')
            print(f'  Correction magnitude: {train["correction_mag"]:.4f}')

            # Print all 7 predicates
            print(f'  Predicates (P1-P7):')
            print(f'    P1(edge)={train["P1"]:.3f}, P2(contrast)={train["P2"]:.3f}, '
                  f'P3(smooth)={train["P3"]:.3f}, P4(structure)={train["P4"]:.3f}')
            print(f'    P5(sharp)={train["P5"]:.3f}, P6(noise)={train["P6"]:.3f}, '
                  f'P7(mse)={train["P7"]:.3f}')

            avg_pred_train = sum(train[f'P{i}'] for i in range(1, 8)) / 7
            print(f'    Average: {avg_pred_train:.4f}')
            print(f'  Time: {train_time:.1f}s')

            # Validation
            if epoch % val_every == 0 or epoch == 1:
                val = self.validate()
                psnr_delta_val = val['psnr_corrected'] - val['psnr_backbone']
                ssim_delta_val = val['ssim_corrected'] - val['ssim_backbone']
                quality_rate_val = val['quality_improved'] * 100

                print(f'\n  [VALIDATION]')
                print(f'  PSNR: backbone={val["psnr_backbone"]:.2f}, '
                      f'corrected={val["psnr_corrected"]:.2f} (delta={psnr_delta_val:+.3f})')
                print(f'  SSIM: backbone={val["ssim_backbone"]:.4f}, '
                      f'corrected={val["ssim_corrected"]:.4f} (delta={ssim_delta_val:+.4f})')
                print(f'  Quality improved: {quality_rate_val:.1f}%')
                print(f'  Lambda: edge={val["lambda_edge"]:.3f}, '
                      f'contrast={val["lambda_contrast"]:.3f}, '
                      f'sharp={val["lambda_sharpness"]:.3f}')
                print(f'  Lambda capping: max_lambda={self.max_lambda:.3f}, '
                      f'capped={val["capped_pct"]:.1f}%')
                print(f'  Correction magnitude: {val["correction_mag"]:.4f}')

                # Print all 7 predicates
                print(f'  Predicates (P1-P7):')
                print(f'    P1(edge)={val["P1"]:.3f}, P2(contrast)={val["P2"]:.3f}, '
                      f'P3(smooth)={val["P3"]:.3f}, P4(structure)={val["P4"]:.3f}')
                print(f'    P5(sharp)={val["P5"]:.3f}, P6(noise)={val["P6"]:.3f}, '
                      f'P7(mse)={val["P7"]:.3f}')

                # Score: average of all 7 predicates + quality bonus
                # quality_bonus = max(0, psnr_delta) * 0.5
                avg_predicates = sum(val[f'P{i}'] for i in range(1, 8)) / 7
                quality_bonus = max(0, psnr_delta_val) * 0.5
                score = avg_predicates + quality_bonus

                print(f'    Average: {avg_predicates:.4f}')
                print(f'  Combined Score: {score:.4f} (avg_pred={avg_predicates:.4f}, '
                      f'quality_bonus={quality_bonus:.4f})')

                # Save best model (prioritize positive PSNR delta)
                if psnr_delta_val > self.best_psnr_delta or (
                    psnr_delta_val >= 0 and score > self.best_score):
                    self.best_score = score
                    self.best_psnr_delta = psnr_delta_val
                    print(f'  *** New best model! ***')

                    save_path = Path(output_dir) / f'best_model_v6_{self.model.corrector_type}.pth'
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'val_metrics': val,
                        'psnr_delta': psnr_delta_val,
                        'ssim_delta': ssim_delta_val,
                        'score': score,
                        'corrector_type': self.model.corrector_type,
                        'max_lambda': self.max_lambda,
                    }, save_path)
                    print(f'  Saved to: {save_path}')

            self.scheduler.step()

        print('\n' + '#'*70)
        print('# TRAINING COMPLETE')
        print('#'*70)
        print(f'\nBest PSNR delta: {self.best_psnr_delta:+.3f} dB')
        print(f'Best combined score: {self.best_score:.4f}')
        print(f'Corrector type: {self.model.corrector_type}')


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Neuro-Symbolic OCT Denoising V6 Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Corrector selection
    parser.add_argument('--corrector-type', type=str, default='powerful',
                       choices=['powerful', 'quality_guided'],
                       help='Type of corrector architecture to use')

    # Training hyperparameters (same defaults as train_v5.py)
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--train-samples', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=10,
                       help='Number of validation samples')
    parser.add_argument('--patch-size', type=int, default=96,
                       help='Patch size for training')

    # Model parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Base learning rate')
    parser.add_argument('--width', type=int, default=64,
                       help='Backbone width')
    parser.add_argument('--corrector-hidden-dim', type=int, default=128,
                       help='Hidden dimension for corrector')
    parser.add_argument('--finetune-backbone', action='store_true',
                       help='Fine-tune the backbone (default: frozen)')

    # Data paths
    parser.add_argument('--train-jsonl', type=str,
                       default='pku37_oct_dataset/pku37_real_train.jsonl',
                       help='Path to training JSONL file')
    parser.add_argument('--val-jsonl', type=str,
                       default='pku37_oct_dataset/pku37_real_val.jsonl',
                       help='Path to validation JSONL file')
    parser.add_argument('--backbone-path', type=str,
                       default='outputs/nafnet_pku37/nafnet_best.pth',
                       help='Path to pretrained backbone')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/nsnd_v6',
                       help='Output directory for checkpoints')

    # Other
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--val-every', type=int, default=5,
                       help='Validate every N epochs')

    # Memory optimization
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision (AMP) training')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                       help='Disable AMP training')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Number of gradient accumulation steps')

    return parser.parse_args()


def main():
    args = parse_args()

    print('='*70)
    print('NEURO-SYMBOLIC OCT DENOISING V6')
    print('Unified Training with All Improvements')
    print(f'Corrector: {args.corrector_type}')
    print(f'Using {_NUM_THREADS} CPU threads')
    print('='*70)

    # Configuration summary
    print(f'\nConfiguration:')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Epochs: {args.epochs}')
    print(f'  Train samples: {args.train_samples}')
    print(f'  Val samples: {args.val_samples}')
    print(f'  Patch size: {args.patch_size}')
    print(f'  Learning rate: {args.lr}')
    print(f'  Finetune backbone: {args.finetune_backbone}')

    # Create model
    print('\nInitializing V6 model...')
    model = NeuroSymbolicDenoiserV6(
        backbone_type='nafnet',
        width=args.width,
        corrector_hidden_dim=args.corrector_hidden_dim,
        corrector_type=args.corrector_type
    )

    # Load pretrained backbone
    if Path(args.backbone_path).exists():
        print(f'Loading pretrained backbone from {args.backbone_path}...')
        model.load_pretrained_backbone(args.backbone_path)
    else:
        print(f'WARNING: Backbone path not found: {args.backbone_path}')
        print('Training with random backbone initialization.')

    # Load data
    print('\nLoading data...')
    try:
        train_dataset = OCTDataset(
            args.train_jsonl,
            max_samples=args.train_samples,
            patch_size=args.patch_size,
            is_train=True
        )
        val_dataset = OCTDataset(
            args.val_jsonl,
            max_samples=args.val_samples,
            patch_size=0,  # Full image for validation
            is_train=False
        )
        print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    except Exception as e:
        print(f'ERROR loading data: {e}')
        print('Please ensure the data files exist.')
        return

    # Create trainer with memory optimizations
    trainer = TrainerV6(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        finetune_backbone=args.finetune_backbone,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Train
    trainer.train(
        epochs=args.epochs,
        val_every=args.val_every,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
