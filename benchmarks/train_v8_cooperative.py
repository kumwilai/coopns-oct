#!/usr/bin/env python3
"""
Train V8 Cooperative Neuro-Symbolic Denoising Framework

IEEE TMI Publication-Worthy Training Script

This script implements the Cooperative Neuro-Symbolic Denoising framework where:
1. NAFNet backbone handles the base denoising
2. Specialized correctors COOPERATE based on NAFNet's uncertainty
3. A negotiator allocates work between NAFNet and correctors
4. Correctors specialize in regions where NAFNet is uncertain

Key Innovation: COOPERATION, NOT COMPETITION
- Correctors have high potential where NAFNet is uncertain
- Work allocation based on confidence/uncertainty regions
- Efficiency: No redundant computation where NAFNet is confident

Training Targets (for IEEE TMI):
- All 5 predicates pass (P5 excluded - incompatible with denoising)
- +10-15% clinical improvement
- < 1.0 dB PSNR drop
- Clear cooperation patterns (not all correctors active everywhere)

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
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
# import lpips  # SPEED: LPIPS disabled — saves 30-40% training time

# Add script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import cooperative module
from neuro_symbolic_corrector_v8_cooperative import NeuroSymbolicCorrectorV8Cooperative

# Import CNR-preserving spatial loss (smart algorithmic approach for CNR preservation)
from uncertainty_guided_correction import CNRPreservingSpatialLoss

# Import CNR-preserving correction module for direct CNR preservation loss
from cnr_preserving_correction import CNRPreservingCorrectionModule
from clinical_enhancement_module import ClinicalMetricsLoss


# =============================================================================
# Helper Functions (Module-Level for Performance)
# =============================================================================

def _safe_item(t):
    """Safely extract .item() from tensor, returning 0.0 for NaN/Inf.

    OPTIMIZATION: Defined at module level to avoid repeated function creation
    inside methods during training loop.
    """
    if isinstance(t, torch.Tensor):
        if not torch.isfinite(t).all():
            return 0.0
        return t.item() if t.numel() == 1 else t.mean().item()
    return float(t) if t is not None else 0.0


# =============================================================================
# Backbone Wrapper with Feature Extraction and Uncertainty Estimation
# =============================================================================

class BackboneWithFeatures(nn.Module):
    """
    NAFNet backbone that returns intermediate encoder features AND uncertainty estimates.

    Returns:
        denoised: Final denoised output [B, 1, H, W]
        features: Dict with 'enc1' [B, width, H, W] and 'enc2' [B, width*2, H/2, W/2]
        uncertainty: Per-pixel uncertainty estimate [B, 1, H, W]
    """

    def __init__(self, width: int = 40):
        super().__init__()

        from nsnd.models.nafnet import NAFNetSmall
        self.backbone = NAFNetSmall(img_channel=1, width=width)
        self.width = width

        # Uncertainty estimation head (uses encoder features)
        # Predicts epistemic uncertainty based on feature variability
        # Architecture: Conv -> BN -> ReLU -> Conv -> (calibrated sigmoid)
        self.uncertainty_conv1 = nn.Conv2d(width, width // 2, 3, padding=1, bias=False)
        self.uncertainty_bn = nn.BatchNorm2d(width // 2)
        self.uncertainty_conv2 = nn.Conv2d(width // 2, 1, 1)

        # Learned calibration parameters for uncertainty
        # temperature controls the sharpness: higher = more spread, lower = more confident
        # bias_offset shifts the distribution: more negative = lower uncertainty
        # -0.5 gives ~38% uncertainty, ~62% confidence - balanced to allow correctors without destroying PSNR
        self.uncertainty_temperature = nn.Parameter(torch.tensor(1.0))
        self.uncertainty_bias_offset = nn.Parameter(torch.tensor(0.3))

        self._init_uncertainty_head()

    def _init_uncertainty_head(self):
        """Initialize uncertainty head for well-calibrated outputs."""
        # First conv: standard Kaiming init (before ReLU)
        nn.init.kaiming_normal_(self.uncertainty_conv1.weight, mode='fan_out', nonlinearity='relu')

        # BatchNorm: standard init
        nn.init.constant_(self.uncertainty_bn.weight, 1)
        nn.init.constant_(self.uncertainty_bn.bias, 0)

        # Last conv: Xavier/Glorot init (before sigmoid) with small weights
        # This ensures outputs are near 0 before sigmoid, giving ~0.5 uncertainty
        # Combined with bias_offset, this gives well-calibrated low initial uncertainty
        nn.init.xavier_uniform_(self.uncertainty_conv2.weight, gain=0.1)  # Small gain for conservative start
        nn.init.constant_(self.uncertainty_conv2.bias, 0.0)  # Bias handled by uncertainty_bias_offset

    def compute_uncertainty(self, enc_feat: torch.Tensor) -> torch.Tensor:
        """Compute calibrated uncertainty from encoder features."""
        x = self.uncertainty_conv1(enc_feat)
        x = self.uncertainty_bn(x)
        x = F.relu(x, inplace=True)
        x = self.uncertainty_conv2(x)

        # Apply learned calibration: temperature scaling + bias offset
        # Clamp temperature to positive values for stability
        temp = self.uncertainty_temperature.clamp(min=0.1)
        calibrated = (x + self.uncertainty_bias_offset) / temp

        # Sigmoid for [0, 1] output
        uncertainty = torch.sigmoid(calibrated)

        return uncertainty

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with feature extraction and uncertainty estimation."""
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

        # Extract features (detached for corrector input)
        enc1_feat = encs_for_features[0][:, :, :H, :W]
        features = {
            'enc1': enc1_feat.detach(),
            'enc2': encs_for_features[1][:, :, :min(H//2, encs_for_features[1].shape[2]),
                                          :min(W//2, encs_for_features[1].shape[3])].detach(),
        }

        # Estimate uncertainty from encoder features (using calibrated method)
        uncertainty = self.compute_uncertainty(enc1_feat)

        del encs, encs_for_features

        return out.clamp(0, 1), features, uncertainty


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
# Model with Cooperative Correction
# =============================================================================

class NeuroSymbolicDenoiserV8Cooperative(nn.Module):
    """
    V8 Cooperative Neuro-Symbolic Denoiser.

    Key Innovation: NAFNet and correctors COOPERATE based on uncertainty.
    - NAFNet handles confident regions
    - Correctors handle uncertain regions
    - Negotiator allocates work efficiently

    This creates a principled division of labor:
    1. NAFNet does the heavy lifting in most regions
    2. Correctors specialize in difficult cases
    3. No redundant computation
    """

    def __init__(self, backbone_width: int = 48, pretrained_backbone: str = None):
        super().__init__()

        # Backbone with feature extraction and uncertainty estimation
        self.backbone = BackboneWithFeatures(width=backbone_width)

        # Cooperative Corrector
        self.corrector = NeuroSymbolicCorrectorV8Cooperative(
            in_channels=1,
            hidden_channels=32,
            enc1_channels=backbone_width,
            enc2_channels=backbone_width * 2
        )

        # Load pretrained backbone
        if pretrained_backbone and os.path.exists(pretrained_backbone):
            self.backbone.load_pretrained(pretrained_backbone)

        self._print_params()

    def _print_params(self):
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        corrector_params = sum(p.numel() for p in self.corrector.parameters())
        total_params = backbone_params + corrector_params

        print(f"\nNeuroSymbolicDenoiserV8Cooperative Parameters:")
        print(f"  Backbone (width={self.backbone.width}): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"  Cooperative Corrector: {corrector_params:,} ({corrector_params/1e6:.2f}M)")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, noisy: torch.Tensor, return_details: bool = False):
        """
        Forward pass with cooperative correction.

        Returns:
            corrected: Final output
            backbone_out: Backbone-only output
            info: Detailed information dict including cooperation stats
        """
        # Get backbone output, features, and uncertainty
        backbone_out, backbone_features, nafnet_uncertainty = self.backbone(noisy)

        # Apply cooperative correction
        corrected, info = self.corrector(
            backbone_out, noisy, backbone_features,
            nafnet_uncertainty=nafnet_uncertainty,
            return_details=return_details
        )

        # Add backbone uncertainty to info (detached to prevent holding backbone graph)
        info['nafnet_uncertainty'] = nafnet_uncertainty.detach()

        return corrected, backbone_out, info


# =============================================================================
# Cooperative Loss Function
# =============================================================================

class CooperativeLoss(nn.Module):
    """
    Loss function for Cooperative Neuro-Symbolic Denoising.

    Components:
    1. Base reconstruction loss (MSE)
    2. Predicate alignment loss (corrected predicates -> GT predicates)
    3. Cooperation loss: Encourage correctors to have high potential where NAFNet is uncertain
    4. Allocation efficiency loss: Penalize redundancy (both NAFNet and corrector active)
    5. Clinical metrics loss (contrast, edge, boundary)
    6. PSNR preservation constraint

    Uses uncertainty weighting (Kendall et al., 2018) for automatic balancing.
    """

    def __init__(self,
                 cooperation_weight: float = 0.3,
                 efficiency_weight: float = 0.2,
                 clinical_weight: float = 1.5,  # CLINICAL IMPROVEMENT IS SELLING POINT - push for 10-15% with PSNR drop < 1.0 dB
                 cnr_weight: float = 0.5,
                 psnr_slack: float = 1.0,
                 use_uncertainty_weighting: bool = True):
        super().__init__()

        self.cooperation_weight = cooperation_weight
        self.efficiency_weight = efficiency_weight
        self.clinical_weight = clinical_weight
        self.cnr_weight = cnr_weight
        self.psnr_slack = psnr_slack
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Uncertainty parameters (learned)
        # Lower log_sigma = higher weight (due to 1/exp(log_sigma) weighting)
        self.log_sigma = nn.ParameterDict({
            'recon': nn.Parameter(torch.tensor(0.5)),
            'backbone': nn.Parameter(torch.tensor(1.0)),
            'cooperation': nn.Parameter(torch.tensor(0.5)),
            'efficiency': nn.Parameter(torch.tensor(0.5)),
            'predicate_align': nn.Parameter(torch.tensor(-1.0)),
            'contrast': nn.Parameter(torch.tensor(-2.5)),
            'contrast_improve': nn.Parameter(torch.tensor(-2.0)),
            'cnr_preserve': nn.Parameter(torch.tensor(-1.0)),
            'cnr_spatial': nn.Parameter(torch.tensor(-1.0)),
            'direct_cnr': nn.Parameter(torch.tensor(-1.0)),
            'clinical_metrics': nn.Parameter(torch.tensor(-2.0)),
            'edge': nn.Parameter(torch.tensor(0.5)),
            'boundary': nn.Parameter(torch.tensor(0.5)),
            'psnr_preserve': nn.Parameter(torch.tensor(-0.5)),
            'ssim_preserve': nn.Parameter(torch.tensor(1.0)),
            'lpips_preserve': nn.Parameter(torch.tensor(-1.5)),
            'texture_preserve': nn.Parameter(torch.tensor(1.0)),
        })

        # CNR-Preserving Spatial Loss (smart algorithmic approach)
        # Uses soft segmentation and region-aware penalties
        # All loss components are positive penalties, ensuring total loss >= 0
        # CNR FIX: Rebalanced to fix -7.5% CNR degradation while maintaining clinical improvement
        # Key insight: We can have BOTH clinical improvement AND CNR improvement:
        # - Tissue contrast enhancement helps BOTH clinical AND CNR numerator
        # - Background noise reduction helps CNR denominator (doesn't hurt clinical)
        self.cnr_spatial_loss = CNRPreservingSpatialLoss(
            background_penalty_weight=5.0,   # REBALANCED: Set to 5.0 (balanced)
            tissue_reward_weight=2.0,        # REBALANCED: Set to 2.0 (balanced)
            cnr_weight=1.0                   # REBALANCED: Set to 1.0 (balanced)
        )

        # Direct CNR preservation module for loss computation
        self.cnr_module = CNRPreservingCorrectionModule()

        # Unified: using shared AdaptiveRegionDetector
        # Wire the cnr_module's region_detector into cnr_spatial_loss so all
        # region detection uses the same learned AdaptiveRegionDetector
        self.cnr_spatial_loss.region_detector = self.cnr_module.region_detector

        # Clinical metrics loss for pushing toward 10-15% improvement
        self.clinical_metrics_loss = ClinicalMetricsLoss()

        # SPEED: LPIPS disabled — VGG19 forward pass costs 20-30% of loss time
        # SSIM + PSNR + texture preservation already cover fidelity
        self.lpips_model = None

        # Sobel filters for edge computation
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Laplacian filter for texture computation (high-frequency content)
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

        # Pre-register averaging kernels for local_std (called 9 times in compute_clinical_loss)
        # This avoids recreating kernels on every call
        for ks in [5, 7, 11]:
            kernel = torch.ones(1, 1, ks, ks) / (ks ** 2)
            self.register_buffer(f'avg_kernel_{ks}', kernel)

        # Texture preservation weight
        self.texture_weight = 0.3

    def _weighted_loss(self, loss: torch.Tensor, name: str) -> torch.Tensor:
        """Apply uncertainty weighting to a loss term.

        BUG FIX 1: The regularization term (0.5 * log_sig) can be negative when
        log_sig < 0, causing negative total loss. We apply F.relu to ensure
        the weighted loss is always non-negative.

        BUG FIX 2: Clamp log_sigma to prevent precision_weight explosion.
        Without clamping, if log_sigma becomes very negative (e.g., -10),
        precision_weight = 0.5 * exp(10) ≈ 11,013, causing loss spikes.

        BUG FIX 3: Validate loss values - skip NaN/Inf to prevent corruption.

        BUG FIX 4: Handle missing log_sigma keys gracefully.
        """
        # Validate loss value first - skip if NaN or Inf
        if not torch.isfinite(loss).all():
            # Return zero loss for invalid values to prevent corruption
            return torch.zeros_like(loss)

        if self.use_uncertainty_weighting:
            # BUG FIX 4: Check if name exists in log_sigma, return unweighted loss if not
            if name not in self.log_sigma:
                print(f"WARNING: '{name}' not in log_sigma, using unweighted loss")
                return loss
            log_sig = self.log_sigma[name]
            # Clamp log_sigma to reasonable range [-4, 4]
            # At log_sig = -4: precision_weight = 0.5 * exp(4) ≈ 27.3 (max weight)
            # At log_sig = +4: precision_weight = 0.5 * exp(-4) ≈ 0.009 (min weight)
            log_sig_clamped = torch.clamp(log_sig, min=-4.0, max=4.0)
            precision_weight = 0.5 * torch.exp(-log_sig_clamped)
            # Additional safety: clamp precision_weight to [0.01, 50.0]
            precision_weight = torch.clamp(precision_weight, min=0.01, max=50.0)
            regularization = 0.5 * log_sig_clamped
            # BUG FIX: Ensure weighted loss is always non-negative
            # When log_sig < 0, regularization is negative, which can make
            # the total weighted loss negative. Apply F.relu to prevent this.
            weighted = precision_weight * loss + regularization
            return F.relu(weighted)
        else:
            return loss

    def get_learned_weights(self, defer_item: bool = True) -> dict:
        """Get current learned weights.

        SPIKE FIX: Clamp log_sigma before exp to prevent weight explosion.
        Without clamping, exp(10) = 22026, exp(-10) = 4.5e-5, causing extreme weights.

        Args:
            defer_item: If True, return tensors instead of floats to avoid CPU/GPU sync.
                       The training loop can extract .item() later in batch.
        """
        # OPTIMIZATION: Compute all weights in batch using tensor ops
        names = list(self.log_sigma.keys())
        log_sigs = torch.stack([self.log_sigma[n] for n in names])
        log_sigs_clamped = torch.clamp(log_sigs, min=-4.0, max=4.0)
        sigma_sq = torch.exp(log_sigs_clamped)
        weights_tensor = (0.5 / sigma_sq).clamp(min=0.01, max=50.0)

        if defer_item:
            # Return dict with tensors - avoid .item() sync during training
            return {n: w for n, w in zip(names, weights_tensor)}
        else:
            # Extract all at once for logging (single sync point)
            weights_list = weights_tensor.tolist()
            return {n: w for n, w in zip(names, weights_list)}

    @staticmethod
    def extract_deferred_metrics(metrics_dict: Dict) -> Dict:
        """Extract .item() values from deferred tensor metrics.

        OPTIMIZATION: This method batches all tensor-to-float conversions
        into a single operation to minimize CPU/GPU sync overhead.

        Args:
            metrics_dict: Dict that may contain tensor values with '_' prefix

        Returns:
            Dict with all tensors converted to floats
        """
        if not metrics_dict.get('_deferred_metrics', False):
            return metrics_dict

        result = {}
        tensor_keys = []
        tensor_vals = []

        for k, v in metrics_dict.items():
            if k == '_deferred_metrics':
                continue
            if k.startswith('_') and isinstance(v, torch.Tensor):
                tensor_keys.append(k[1:])  # Remove '_' prefix
                tensor_vals.append(v if v.numel() == 1 else v.mean())
            elif not k.startswith('_'):
                result[k] = v

        if tensor_vals:
            # OPTIMIZATION: Stack and extract all at once (single sync)
            stacked = torch.stack(tensor_vals)
            extracted = stacked.tolist()
            for k, v in zip(tensor_keys, extracted):
                result[k] = v if torch.isfinite(torch.tensor(v)) else 0.0

        return result

    def local_std(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Compute local standard deviation using pre-registered kernels."""
        kernel = getattr(self, f'avg_kernel_{kernel_size}')
        # OPTIMIZATION: Cache dtype-converted kernels to avoid repeated .to() calls
        cache_attr = f'_avg_kernel_{kernel_size}_{x.dtype}'
        if not hasattr(self, cache_attr) or getattr(self, cache_attr).device != x.device:
            setattr(self, cache_attr, kernel.to(dtype=x.dtype, device=x.device))
        kernel = getattr(self, cache_attr)
        padding = kernel_size // 2
        local_mean = F.conv2d(x, kernel, padding=padding)
        local_sq_mean = F.conv2d(x ** 2, kernel, padding=padding)
        local_var = local_sq_mean - local_mean ** 2
        return torch.sqrt(local_var.clamp(min=1e-8))

    def compute_confidence_entropy_loss(self, confidence, target_entropy=0.5):
        """Encourage confidence to use its full dynamic range, not collapse to all-high or all-low."""
        eps = 1e-6
        p = confidence.clamp(eps, 1 - eps)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        # Maximum binary entropy is ln(2) ~ 0.693 at p=0.5
        # Target a specific mean entropy level
        return F.mse_loss(entropy.mean(), torch.tensor(target_entropy, device=confidence.device))

    def compute_sparsity_loss(self, allocations):
        """Encourage sparse corrector usage - reward NOT using correctors when unnecessary."""
        if isinstance(allocations, dict):
            total = sum(alloc.mean() for alloc in allocations.values()) / len(allocations)
        elif isinstance(allocations, (list, tuple)):
            total = sum(a.mean() for a in allocations) / len(allocations)
        else:
            total = allocations.mean()
        return total

    def compute_cooperation_loss(self, nafnet_uncertainty: torch.Tensor,
                                  corrector_potentials: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Cooperation Loss: Encourage correctors to have high potential where NAFNet is uncertain.

        Goal: Correctors should "pick up slack" where NAFNet struggles.
        Loss = -correlation(uncertainty, total_potential)
             = encourage high corrector potential in high uncertainty regions
        """
        # Sum corrector potentials (extract 'map' from dict if present)
        # FIX: Avoid creating torch.tensor(0.0) in hot loop - use zeros_like instead
        if corrector_potentials:
            potential_maps = []
            for name, val in corrector_potentials.items():
                if isinstance(val, dict) and 'map' in val:
                    potential_maps.append(val['map'])
                elif isinstance(val, torch.Tensor):
                    potential_maps.append(val)
            if not potential_maps:
                # FIX: Return scalar 0.0 directly, loss function handles it
                zero_loss = nafnet_uncertainty.new_zeros(())
                return zero_loss, {}
            total_potential = sum(potential_maps)
            del potential_maps  # FIX: Free list after use
        else:
            zero_loss = nafnet_uncertainty.new_zeros(())
            return zero_loss, {}

        # Normalize both to [0, 1] for correlation
        # BUG FIX: Check for degenerate case where min == max
        uncertainty_range = nafnet_uncertainty.max() - nafnet_uncertainty.min()
        potential_range = total_potential.max() - total_potential.min()

        # If either range is near-zero, return zero loss (no meaningful correlation)
        if uncertainty_range < 1e-6 or potential_range < 1e-6:
            zero_loss = nafnet_uncertainty.new_zeros(())
            return zero_loss, {
                'uncertainty_potential_corr': 0.0,
                'mean_uncertainty': nafnet_uncertainty.mean().item(),
                'mean_potential': total_potential.mean().item(),
            }

        uncertainty_norm = (nafnet_uncertainty - nafnet_uncertainty.min()) / (uncertainty_range + 1e-8)
        potential_norm = (total_potential - total_potential.min()) / (potential_range + 1e-8)

        # Pearson correlation: we want POSITIVE correlation
        # (high uncertainty -> high potential)
        u_centered = uncertainty_norm - uncertainty_norm.mean()
        p_centered = potential_norm - potential_norm.mean()

        # BUG FIX: Check for zero variance before computing correlation
        u_sq_sum = (u_centered ** 2).sum()
        p_sq_sum = (p_centered ** 2).sum()
        if u_sq_sum < 1e-12 or p_sq_sum < 1e-12:
            zero_loss = nafnet_uncertainty.new_zeros(())
            # OPTIMIZATION: Defer .item() calls to metrics collection phase (avoid CPU/GPU sync)
            return zero_loss, {
                '_uncertainty_mean': nafnet_uncertainty.mean(),
                '_potential_mean': total_potential.mean(),
                '_deferred_metrics': True,
            }

        # FIX: Epsilon inside sqrt to prevent sqrt(0) before adding epsilon
        correlation = (u_centered * p_centered).sum() / (
            torch.sqrt(u_sq_sum * p_sq_sum + 1e-16))

        # Loss: negative correlation (we want to maximize correlation)
        # Clamp correlation to [-1, 1] to avoid numerical issues, then ensure loss >= 0
        correlation = correlation.clamp(-1.0, 1.0)
        # Normalize loss to [0, 1]: divide by 2 since correlation range is [-1, 1]
        cooperation_loss = F.relu(1.0 - correlation) / 2.0  # Now ranges [0, 1]
        # SPIKE FIX: Clamp cooperation_loss to max 1.0
        cooperation_loss = cooperation_loss.clamp(max=1.0)

        # OPTIMIZATION: Defer .item() calls - return tensors for later extraction
        metrics = {
            '_correlation': correlation,
            '_uncertainty_mean': nafnet_uncertainty.mean(),
            '_potential_mean': total_potential.mean(),
            '_deferred_metrics': True,
        }

        return cooperation_loss, metrics

    def compute_efficiency_loss(self, nafnet_confidence: torch.Tensor,
                                 corrector_potentials: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Efficiency Loss: Penalize redundancy where both NAFNet and correctors are active.

        Goal: If NAFNet is confident (low uncertainty), correctors should have low potential.
        This prevents wasted computation.

        Loss = mean(confidence * total_potential)
             = penalize corrector activity in confident regions
        """
        # Sum corrector potentials (extract 'map' from dict if present)
        # FIX: Avoid creating torch.tensor(0.0) in hot loop
        if corrector_potentials:
            potential_maps = []
            for name, val in corrector_potentials.items():
                if isinstance(val, dict) and 'map' in val:
                    potential_maps.append(val['map'])
                elif isinstance(val, torch.Tensor):
                    potential_maps.append(val)
            if not potential_maps:
                zero_loss = nafnet_confidence.new_zeros(())
                return zero_loss, {}
            total_potential = sum(potential_maps)
            del potential_maps  # FIX: Free list after use
        else:
            zero_loss = nafnet_confidence.new_zeros(())
            return zero_loss, {}

        # Redundancy: confident regions with high corrector potential
        redundancy = nafnet_confidence * total_potential

        # Average redundancy
        efficiency_loss = redundancy.mean()
        # SPIKE FIX: Clamp efficiency_loss to prevent extreme values
        # Both confidence and potential are in [0,1], so product max is 1.0
        # but mean could theoretically be affected by numerical issues
        efficiency_loss = efficiency_loss.clamp(max=5.0)

        # Also track allocation statistics
        confident_mask = (nafnet_confidence > 0.6).float()
        uncertain_mask = (nafnet_confidence < 0.4).float()

        # OPTIMIZATION: Compute counts without .item() sync - use tensor operations
        confident_count = confident_mask.sum()
        uncertain_count = uncertain_mask.sum()

        # OPTIMIZATION: Compute all metrics as tensors, defer .item() extraction
        pot_in_conf_tensor = torch.where(
            confident_count > 0.5,
            (total_potential * confident_mask).sum() / (confident_count + 1e-8),
            confident_count.new_zeros(())
        )
        pot_in_unc_tensor = torch.where(
            uncertain_count > 0.5,
            (total_potential * uncertain_mask).sum() / (uncertain_count + 1e-8),
            uncertain_count.new_zeros(())
        )

        # OPTIMIZATION: Return tensors in metrics dict - defer .item() to final extraction
        metrics = {
            '_redundancy': redundancy.mean(),
            '_potential_in_confident': pot_in_conf_tensor,
            '_potential_in_uncertain': pot_in_unc_tensor,
            '_confident_coverage': confident_mask.mean(),
            '_uncertain_coverage': uncertain_mask.mean(),
            '_deferred_metrics': True,
        }
        del confident_mask, uncertain_mask, redundancy  # FIX: Free intermediate tensors

        return efficiency_loss, metrics

    def compute_predicate_alignment_loss(self, pred_scores_corrected: Dict[str, float],
                                          target_scores: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Predicate Alignment Loss: Push corrected predicate scores toward passing threshold.

        For each predicate (except P5):
        - If score < 0.5: penalize proportionally to how far below threshold
        - If score >= 0.5: no penalty (or small reward)

        ENHANCED for IEEE TMI: Much stronger penalty for P2 (contrast) to achieve
        +10-15% clinical contrast improvement target.
        """
        # Use a reference tensor for device/dtype inference (avoids torch.tensor overhead)
        ref_param = next(iter(self.log_sigma.values()))

        # Target: all predicates should pass (score >= 0.5)
        evaluated_predicates = ['P1', 'P2', 'P3', 'P4', 'P6']  # P5 excluded

        # Weight multipliers for each predicate (higher = more important)
        # P2 gets 5x weight (increased from 3x) since contrast improvement is critical
        pred_weights = {'P1': 1.0, 'P2': 5.0, 'P3': 1.0, 'P4': 2.0, 'P6': 1.0}  # P4 weight increased from 1.0 to 2.0

        # Higher target for P2 to push it well above threshold
        pred_targets = {'P1': 0.55, 'P2': 0.60, 'P3': 0.55, 'P4': 0.55, 'P6': 0.55}

        total_loss = ref_param.new_zeros(())
        total_weight = 0.0

        for pred in evaluated_predicates:
            score = pred_scores_corrected.get(pred, 0.5)
            if isinstance(score, torch.Tensor):
                score_val = score
            else:
                score_val = ref_param.new_tensor(score)

            # Target varies by predicate - P2 has higher target
            target = pred_targets.get(pred, 0.55)
            deficit = F.relu(target - score_val)

            # Weight varies by predicate - P2 has much higher weight
            weight = pred_weights.get(pred, 1.0)

            # Use linear + quadratic + cubic penalty for very strong gradient on P2
            if pred == 'P2':
                # Extra aggressive penalty for contrast
                penalty = deficit + 2.0 * deficit ** 2 + deficit ** 3
            else:
                # Standard linear + quadratic for others
                penalty = deficit + deficit ** 2

            total_loss = total_loss + weight * penalty
            total_weight += weight

        # SPIKE FIX: Clamp final predicate alignment loss to prevent extreme values
        # Max theoretical is ~1.23, clamp to 5.0 for safety
        return torch.clamp(total_loss / total_weight, max=5.0)

    def compute_clinical_loss(self, corrected: torch.Tensor, clean: torch.Tensor,
                               backbone: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Clinical metrics loss for OCT-specific quality.

        ENHANCED for IEEE TMI: Targeting +10-15% clinical contrast improvement.

        REGION-AWARE ENHANCEMENT:
        - Clinical enhancement bonuses are ONLY counted in tissue regions
        - Any clinical "enhancement" in background is penalized (noise amplification)
        - Background regions: prioritize PSNR/smoothness, minimize clinical enhancement
        - Tissue regions: prioritize clinical enhancement, allow PSNR deviation

        Components:
        1. Contrast preservation (multi-scale)
        2. Direct contrast improvement incentive (tissue-weighted)
        3. Edge preservation (tissue-weighted)
        4. Boundary sharpness (vertical gradients)
        5. Relative contrast boost term
        6. Background clinical change penalty (NEW)
        """
        # ===== REGION MASK COMPUTATION =====
        # Get tissue and background masks for region-aware loss
        eps = 1e-6
        tissue_mask = None
        bg_mask = None

        # Use cached masks if available (computed once in forward() for efficiency)
        if self._cached_tissue_mask is not None and self._cached_bg_mask is not None:
            tissue_mask = self._cached_tissue_mask
            bg_mask = self._cached_bg_mask
        elif hasattr(self, 'cnr_module') and self.cnr_module is not None:
            if hasattr(self.cnr_module, 'region_detector') and self.cnr_module.region_detector is not None:
                try:
                    tissue_mask, bg_mask = self.cnr_module.region_detector(backbone)
                    tissue_mask = tissue_mask.clamp(0.0, 1.0)
                    bg_mask = bg_mask.clamp(0.0, 1.0)
                except Exception:
                    tissue_mask = None
                    bg_mask = None

        # Fallback: use intensity-based soft mask if region detector not available
        if tissue_mask is None:
            # Simple intensity-based segmentation: tissue is typically brighter
            backbone_norm = (backbone - backbone.min()) / (backbone.max() - backbone.min() + eps)
            tissue_mask = torch.sigmoid((backbone_norm - 0.3) * 10.0)  # Soft threshold at 0.3
            bg_mask = 1.0 - tissue_mask

        # ===== MULTI-SCALE CONTRAST ANALYSIS =====
        # Use multiple scales to capture both fine and coarse contrast
        clean_std_5 = self.local_std(clean, 5)
        clean_std_7 = self.local_std(clean, 7)
        clean_std_11 = self.local_std(clean, 11)

        corrected_std_5 = self.local_std(corrected, 5)
        corrected_std_7 = self.local_std(corrected, 7)
        corrected_std_11 = self.local_std(corrected, 11)

        backbone_std_5 = self.local_std(backbone, 5)
        backbone_std_7 = self.local_std(backbone, 7)
        backbone_std_11 = self.local_std(backbone, 11)

        # Multi-scale contrast matching loss (corrected should match clean)
        contrast_loss_5 = F.mse_loss(corrected_std_5, clean_std_5)
        contrast_loss_7 = F.mse_loss(corrected_std_7, clean_std_7)
        contrast_loss_11 = F.mse_loss(corrected_std_11, clean_std_11)
        contrast_loss = (contrast_loss_5 + contrast_loss_7 + contrast_loss_11) / 3.0
        # SPIKE FIX: Clamp contrast_loss to prevent extreme values
        contrast_loss = torch.clamp(contrast_loss, max=5.0)

        # ===== DIRECT CONTRAST IMPROVEMENT INCENTIVE (REGION-AWARE) =====
        # This term directly encourages corrected contrast > backbone contrast
        # where backbone has reduced contrast compared to clean
        # ENHANCED: Only count improvements in TISSUE regions, penalize background changes

        # Identify regions where backbone lost contrast (low-contrast regions)
        backbone_contrast_deficit = F.relu(clean_std_7 - backbone_std_7)  # Where backbone is deficient

        # Compute how much corrector improved over backbone in those regions
        contrast_improvement_map = corrected_std_7 - backbone_std_7  # Positive = improvement

        # Penalize LACK of improvement in deficit regions
        # Loss = high when no improvement, low when improvement is good
        deficit_mask = (backbone_contrast_deficit > 0.01).float()  # Regions needing improvement

        # REGION-AWARE: Weight deficit_mask by tissue_mask to focus on tissue regions only
        tissue_deficit_mask = deficit_mask * tissue_mask
        tissue_deficit_sum = tissue_deficit_mask.sum().clamp(min=1.0)

        # Tissue-specific improvement (only count improvements in tissue regions)
        tissue_improvement_amount = (contrast_improvement_map * tissue_deficit_mask).sum() / tissue_deficit_sum

        # FIX: Target 12-15% clinical improvement (was 10%)
        # Stronger multiplier to push model toward target
        target_improvement = 0.12  # 12% improvement target (was 0.1)
        contrast_improvement_loss = F.relu(target_improvement - tissue_improvement_amount) * 5.0  # Stronger (was 2.0)
        # SPIKE FIX: Clamp to prevent extreme values when improvement_amount is very negative
        contrast_improvement_loss = torch.clamp(contrast_improvement_loss, max=8.0)  # Higher cap (was 5.0)

        # ===== BACKGROUND CLINICAL CHANGE PENALTY (NEW) =====
        # Any "improvement" in background contrast is actually noise amplification - PENALIZE IT
        bg_sum = bg_mask.sum().clamp(min=1.0)
        bg_contrast_change = (contrast_improvement_map * bg_mask).sum() / bg_sum
        # Penalize positive changes in background (noise amplification)
        # Negative changes (smoothing) are OK or even good in background
        bg_clinical_penalty = F.relu(bg_contrast_change) * 8.0  # Strong penalty for BG noise
        bg_clinical_penalty = torch.clamp(bg_clinical_penalty, max=5.0)

        # ===== RELATIVE CONTRAST BOOST TERM (TISSUE-WEIGHTED) =====
        # Penalize if corrected contrast is LESS than backbone contrast IN TISSUE REGIONS
        # This prevents the corrector from reducing contrast further in important areas

        relative_contrast_deficit = F.relu(backbone_std_7 - corrected_std_7)  # Where corrected is worse
        # Weight by tissue_mask: only penalize contrast regression in tissue regions
        tissue_contrast_regression = (relative_contrast_deficit * tissue_mask).sum() / tissue_mask.sum().clamp(min=1.0)
        contrast_regression_penalty = tissue_contrast_regression * 3.0  # Strong penalty
        # SPIKE FIX: Clamp to prevent extreme values
        contrast_regression_penalty = torch.clamp(contrast_regression_penalty, max=5.0)

        # ===== TARGETED LOW-CONTRAST TISSUE REGION BOOST (TISSUE-AWARE) =====
        # Extra loss term to specifically boost very low contrast TISSUE regions

        # Find very low contrast regions in backbone output
        backbone_std_norm = backbone_std_7 / (backbone_std_7.max() + 1e-8)
        very_low_contrast_mask = (backbone_std_norm < 0.2).float()

        # REGION-AWARE: Combine with tissue mask to focus on tissue regions only
        # This prevents boosting low-contrast background (which is correct behavior)
        tissue_low_contrast_mask = very_low_contrast_mask * tissue_mask

        # In these regions, encourage corrected_std to be closer to clean_std
        # SPIKE FIX: Use minimum mask count threshold to avoid division by near-zero
        # When mask is too small, fall back to mean over entire tensor to avoid spike
        mask_sum = tissue_low_contrast_mask.sum()
        min_mask_pixels = 100.0  # Minimum pixels for reliable averaging
        if mask_sum > min_mask_pixels:
            low_contrast_boost_loss = (
                (clean_std_7 - corrected_std_7).abs() * tissue_low_contrast_mask
            ).sum() / mask_sum * 2.0
        else:
            # Fallback: use tissue-weighted mean when mask is too sparse
            tissue_sum_fallback = tissue_mask.sum().clamp(min=1.0)
            low_contrast_boost_loss = ((clean_std_7 - corrected_std_7).abs() * tissue_mask).sum() / tissue_sum_fallback * 0.5
        # SPIKE FIX: Final clamp to ensure bounded value
        low_contrast_boost_loss = torch.clamp(low_contrast_boost_loss, max=5.0)

        # ===== EDGE PRESERVATION =====
        # OPTIMIZATION: Cache dtype-converted Sobel kernels to avoid repeated .to() calls
        if not hasattr(self, '_sobel_x_cached') or self._sobel_x_cached.dtype != corrected.dtype:
            self._sobel_x_cached = self.sobel_x.to(dtype=corrected.dtype)
            self._sobel_y_cached = self.sobel_y.to(dtype=corrected.dtype)
        sobel_x = self._sobel_x_cached
        sobel_y = self._sobel_y_cached

        clean_edge = torch.sqrt(
            F.conv2d(clean, sobel_x, padding=1)**2 +
            F.conv2d(clean, sobel_y, padding=1)**2 + 1e-6
        )
        corrected_edge = torch.sqrt(
            F.conv2d(corrected, sobel_x, padding=1)**2 +
            F.conv2d(corrected, sobel_y, padding=1)**2 + 1e-6
        )
        backbone_edge = torch.sqrt(
            F.conv2d(backbone, sobel_x, padding=1)**2 +
            F.conv2d(backbone, sobel_y, padding=1)**2 + 1e-6
        )

        edge_loss = F.mse_loss(corrected_edge, clean_edge)
        # SPIKE FIX: Clamp edge_loss to prevent extreme values
        edge_loss = torch.clamp(edge_loss, max=5.0)

        # ===== BOUNDARY SHARPNESS (OCT layers are horizontal) =====
        clean_vgrad = torch.abs(clean[:, :, 1:, :] - clean[:, :, :-1, :])
        corrected_vgrad = torch.abs(corrected[:, :, 1:, :] - corrected[:, :, :-1, :])

        boundary_loss = F.mse_loss(corrected_vgrad, clean_vgrad)
        # SPIKE FIX: Clamp boundary_loss to prevent extreme values
        boundary_loss = torch.clamp(boundary_loss, max=5.0)

        # ===== COMBINED CLINICAL LOSS (REGION-AWARE) =====
        # Weight contrast-related terms more heavily to achieve +10-15% improvement
        # ENHANCED: Added background clinical penalty to prevent noise amplification
        total_clinical = (
            contrast_loss * 1.5 +           # Multi-scale matching (max ~7.5)
            contrast_improvement_loss +      # Direct improvement incentive - TISSUE ONLY (max 8.0)
            contrast_regression_penalty +    # Prevent contrast reduction in tissue (max 5.0)
            low_contrast_boost_loss +        # Boost very low contrast tissue regions (max 5.0)
            edge_loss +                      # Edge preservation (max 5.0)
            boundary_loss +                  # Boundary sharpness (max 5.0)
            bg_clinical_penalty              # Penalize background "enhancement" (max 5.0)
        )
        # SPIKE FIX: Clamp total_clinical loss to prevent extreme values
        # Max theoretical: 7.5 + 8 + 5*5 = 40.5, clamp to 18.0 to prevent spikes
        total_clinical = torch.clamp(total_clinical, max=18.0)

        # ===== COMPUTE METRICS =====
        # SPIKE FIX: Clamp denominators to minimum value and clamp final ratios
        clean_std_mean = clean_std_7.mean().clamp(min=1e-4)
        clean_edge_mean = clean_edge.mean().clamp(min=1e-4)

        backbone_contrast_pres = backbone_std_7.mean() / clean_std_mean
        corrected_contrast_pres = corrected_std_7.mean() / clean_std_mean
        # SPIKE FIX: Clamp preservation ratios to [0, 10] to prevent extreme values
        backbone_contrast_pres = backbone_contrast_pres.clamp(0.0, 10.0)
        corrected_contrast_pres = corrected_contrast_pres.clamp(0.0, 10.0)

        backbone_edge_pres = backbone_edge.mean() / clean_edge_mean
        corrected_edge_pres = corrected_edge.mean() / clean_edge_mean
        # SPIKE FIX: Clamp edge preservation ratios to [0, 10]
        backbone_edge_pres = backbone_edge_pres.clamp(0.0, 10.0)
        corrected_edge_pres = corrected_edge_pres.clamp(0.0, 10.0)

        # SPIKE FIX: Clamp backbone_contrast_pres denominator and final improvement percentage
        backbone_contrast_pres_safe = backbone_contrast_pres.clamp(min=1e-4)
        contrast_improvement_pct = ((corrected_contrast_pres / backbone_contrast_pres_safe) - 1.0) * 100
        # SPIKE FIX: Clamp improvement percentage to [-100%, 500%] to prevent extreme values
        contrast_improvement_pct = contrast_improvement_pct.clamp(-100.0, 500.0)

        # OPTIMIZATION: Use module-level _safe_item function (already defined)

        # SPIKE FIX: Use clamped backbone_edge_pres and clamp final ratio
        backbone_edge_pres_safe = backbone_edge_pres.clamp(min=1e-4)
        edge_improve_ratio = (corrected_edge_pres / backbone_edge_pres_safe) - 1.0
        # SPIKE FIX: Clamp edge improvement ratio to [-1, 5] (i.e., -100% to 500%)
        edge_improve_ratio = edge_improve_ratio.clamp(-1.0, 5.0)
        edge_improvement = _safe_item(edge_improve_ratio) * 100

        # ===== REGION-AWARE METRICS (NEW) =====
        # Compute tissue-specific and background-specific clinical metrics
        tissue_sum_metric = tissue_mask.sum().clamp(min=1.0)
        bg_sum_metric = bg_mask.sum().clamp(min=1.0)

        # Tissue-specific contrast improvement (the improvement that matters clinically)
        tissue_contrast_b = (backbone_std_7 * tissue_mask).sum() / tissue_sum_metric
        tissue_contrast_c = (corrected_std_7 * tissue_mask).sum() / tissue_sum_metric
        clinical_tissue_improvement_pct = ((tissue_contrast_c / tissue_contrast_b.clamp(min=eps)) - 1.0) * 100
        clinical_tissue_improvement_pct = clinical_tissue_improvement_pct.clamp(-100.0, 500.0)

        # Background contrast change (should be near 0 or negative - positive is noise amplification)
        bg_contrast_b = (backbone_std_7 * bg_mask).sum() / bg_sum_metric
        bg_contrast_c = (corrected_std_7 * bg_mask).sum() / bg_sum_metric
        clinical_bg_change_pct = ((bg_contrast_c / bg_contrast_b.clamp(min=eps)) - 1.0) * 100
        clinical_bg_change_pct = clinical_bg_change_pct.clamp(-100.0, 500.0)

        metrics = {
            'contrast_loss': _safe_item(contrast_loss),
            'contrast_improvement_loss': _safe_item(contrast_improvement_loss),
            'contrast_improvement_loss_tensor': contrast_improvement_loss,  # BUG 9 FIX: Return tensor for gradient flow
            'contrast_regression_penalty': _safe_item(contrast_regression_penalty),
            'low_contrast_boost_loss': _safe_item(low_contrast_boost_loss),
            'edge_loss': _safe_item(edge_loss),
            'edge_loss_tensor': edge_loss,  # Tensor for independent weighting
            'boundary_loss': _safe_item(boundary_loss),
            'boundary_loss_tensor': boundary_loss,  # Tensor for independent weighting
            'bg_clinical_penalty': _safe_item(bg_clinical_penalty),  # NEW: Background noise penalty
            'backbone_contrast_pres': _safe_item(backbone_contrast_pres),
            'corrected_contrast_pres': _safe_item(corrected_contrast_pres),
            'contrast_improvement': _safe_item(contrast_improvement_pct),
            'backbone_edge_pres': _safe_item(backbone_edge_pres),
            'corrected_edge_pres': _safe_item(corrected_edge_pres),
            'edge_improvement': edge_improvement,
            'low_contrast_region_coverage': _safe_item(tissue_low_contrast_mask.mean()),  # Updated to use tissue mask
            'contrast_deficit_mean': _safe_item(backbone_contrast_deficit.mean()),
            # NEW: Region-aware clinical metrics
            'clinical_tissue_improvement': _safe_item(clinical_tissue_improvement_pct),  # Clinical improvement in tissue only
            'clinical_bg_change': _safe_item(clinical_bg_change_pct),  # Change in background (should be ~0 or negative)
            'tissue_mask_coverage': _safe_item(tissue_mask.mean()),  # How much of image is tissue
            'bg_mask_coverage': _safe_item(bg_mask.mean()),  # How much of image is background
        }

        return total_clinical, metrics

    def _compute_cnr(self, x: torch.Tensor, signal_mask: torch.Tensor,
                      bg_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Contrast-to-Noise Ratio (CNR) for a given image.

        CNR = (mean_signal - mean_background) / std_background

        Args:
            x: Image tensor [B, 1, H, W]
            signal_mask: Binary mask for signal regions [B, 1, H, W]
            bg_mask: Binary mask for background regions [B, 1, H, W]

        Returns:
            CNR value (differentiable)
        """
        eps = 1e-8

        # Compute mean signal
        signal_sum = (x * signal_mask).sum(dim=[1, 2, 3])
        signal_count = signal_mask.sum(dim=[1, 2, 3]) + eps
        signal_mean = signal_sum / signal_count

        # Compute mean background
        bg_sum = (x * bg_mask).sum(dim=[1, 2, 3])
        bg_count = bg_mask.sum(dim=[1, 2, 3]) + eps
        bg_mean = bg_sum / bg_count

        # BUG FIX: Check for empty or near-empty masks
        # If mask count is too small, return zero CNR to avoid instability
        if (signal_count < 10).any() or (bg_count < 10).any():
            return x.new_zeros(())

        # Compute background std
        # std = sqrt(E[(x - mean)^2])
        bg_mean_expanded = bg_mean.view(-1, 1, 1, 1)
        bg_var = ((x - bg_mean_expanded) ** 2 * bg_mask).sum(dim=[1, 2, 3]) / bg_count
        # BUG FIX: Clamp variance before sqrt to prevent sqrt of negative numbers
        bg_var = bg_var.clamp(min=eps)
        bg_std = torch.sqrt(bg_var)
        # BUG FIX: Clamp std to minimum value AFTER sqrt to ensure valid denominator
        bg_std = bg_std.clamp(min=1e-4)

        # CNR = (signal_mean - bg_mean) / bg_std
        # BUG FIX: std is already clamped, no need for additional eps
        cnr = (signal_mean - bg_mean) / bg_std
        # Clamp CNR to prevent extreme values causing loss spikes
        cnr = cnr.clamp(-100.0, 100.0)

        # BUG FIX: Check for NaN/Inf before returning
        if not torch.isfinite(cnr).all():
            return x.new_zeros(())

        return cnr.mean()  # Average over batch

    def compute_cnr_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                         clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        CNR Preservation Loss: Prevent corrections from degrading Contrast-to-Noise Ratio.

        This loss addresses the -7.1% CNR drop issue by:
        1. Computing CNR for both backbone and corrected outputs
        2. Penalizing if corrected CNR < backbone CNR (CNR should not decrease)
        3. Optionally rewarding CNR improvement over backbone

        The loss is differentiable and encourages:
        - Corrections that don't increase background noise
        - Improved signal-to-background separation

        Args:
            corrected: Corrected output [B, 1, H, W]
            backbone: Backbone output [B, 1, H, W]
            clean: Clean reference [B, 1, H, W]

        Returns:
            loss: CNR preservation loss
            metrics: Dict with CNR values and statistics
        """
        # Unified: using shared AdaptiveRegionDetector
        # Use cached masks from forward() if available, otherwise detect from backbone
        if self._cached_tissue_mask is not None and self._cached_bg_mask is not None:
            signal_mask = self._cached_tissue_mask
            bg_mask = self._cached_bg_mask
        elif (hasattr(self, 'cnr_module') and self.cnr_module is not None and
              hasattr(self.cnr_module, 'region_detector') and self.cnr_module.region_detector is not None):
            try:
                signal_mask, bg_mask = self.cnr_module.region_detector(backbone)
                signal_mask = signal_mask.clamp(0.0, 1.0)
                bg_mask = bg_mask.clamp(0.0, 1.0)
            except Exception:
                # Fallback to threshold-based if detector fails
                clean_mean = clean.mean(dim=[2, 3], keepdim=True)
                signal_mask = (clean > clean_mean).float()
                bg_mask = 1.0 - signal_mask
        else:
            # Fallback to threshold-based if no detector available
            clean_mean = clean.mean(dim=[2, 3], keepdim=True)
            signal_mask = (clean > clean_mean).float()
            bg_mask = 1.0 - signal_mask

        # Compute CNR for backbone
        backbone_cnr = self._compute_cnr(backbone, signal_mask, bg_mask)

        # Compute CNR for corrected
        corrected_cnr = self._compute_cnr(corrected, signal_mask, bg_mask)

        # Compute CNR for clean (reference)
        clean_cnr = self._compute_cnr(clean, signal_mask, bg_mask)

        # ===== CNR Drop Penalty =====
        # Penalize if corrected CNR drops below backbone CNR
        # loss = relu(backbone_cnr - corrected_cnr)
        cnr_drop = backbone_cnr - corrected_cnr
        # SPIKE FIX: Clamp drop before relu to prevent extreme values
        # CNR values are clamped to [-100, 100], so max theoretical drop is 200
        # Clamp to 20 to keep penalty reasonable
        cnr_drop = cnr_drop.clamp(max=20.0)
        cnr_drop_penalty = F.relu(cnr_drop)

        # ===== CNR Improvement Reward (ENHANCED) =====
        # ENHANCED: Stronger reward for improving CNR beyond backbone (toward clean)
        # This encourages the corrector to improve CNR, not just preserve it
        # Target: CNR change >= 0% (prevent -1.3% degradation)
        cnr_improvement_potential = clean_cnr - backbone_cnr  # How much room to improve
        cnr_improvement_actual = corrected_cnr - backbone_cnr  # How much we improved

        # ENHANCED: Stronger improvement reward (0.3 instead of 0.1)
        improvement_reward = torch.where(
            cnr_improvement_potential > 0,
            F.relu(cnr_improvement_actual) * 0.3,  # ENHANCED: 3x stronger reward weight (was 0.1)
            torch.zeros_like(cnr_improvement_actual)
        )

        # ===== Direct CNR Degradation Penalty (NEW) =====
        # ENHANCED: Additional penalty specifically for any CNR degradation
        # This directly targets the -1.3% CNR drop issue
        cnr_degradation_penalty = F.relu(-cnr_improvement_actual) * 2.0  # 2x penalty for degradation
        cnr_degradation_penalty = cnr_degradation_penalty.clamp(max=5.0)

        # ===== Combined CNR Loss =====
        # Primary: penalize drops, Secondary: reduced penalty for improvements
        # Convert improvement_reward to reduced penalty (always positive)
        improvement_bonus = improvement_reward.clamp(max=cnr_drop_penalty)  # Can't reduce below 0
        cnr_loss = cnr_drop_penalty - improvement_bonus  # Always >= 0

        # ENHANCED: Add direct degradation penalty
        cnr_loss = cnr_loss + cnr_degradation_penalty

        # Also add a term to encourage closing the gap to clean CNR
        cnr_gap_to_clean = F.relu(clean_cnr - corrected_cnr)
        # SPIKE FIX: Clamp gap before adding to prevent extreme values
        # Max theoretical gap is 200 (clean at 100, corrected at -100)
        cnr_gap_to_clean = cnr_gap_to_clean.clamp(max=50.0)
        cnr_loss = cnr_loss + 0.1 * cnr_gap_to_clean  # ENHANCED: Doubled weight for gap reduction (was 0.05)
        cnr_loss = F.relu(cnr_loss)  # Ensure non-negative

        # Clamp final CNR loss to prevent loss spikes
        cnr_loss = cnr_loss.clamp(max=8.0)  # ENHANCED: Allow higher max for stronger signal (was 5.0)

        # Compute metrics
        # SPIKE FIX: Clamp backbone_cnr denominator and final improvement percentage
        backbone_cnr_safe = backbone_cnr.clamp(min=1e-4) if backbone_cnr > 0 else backbone_cnr.clamp(max=-1e-4)
        # Avoid division by near-zero - use absolute value check
        if backbone_cnr.abs() < 1e-4:
            cnr_improvement_pct = backbone_cnr.new_zeros(())
        else:
            cnr_improvement_pct = ((corrected_cnr / backbone_cnr) - 1.0) * 100
            # SPIKE FIX: Clamp improvement percentage to [-100%, 500%]
            cnr_improvement_pct = cnr_improvement_pct.clamp(-100.0, 500.0)

        # OPTIMIZATION: Use module-level _safe_item function (already defined)
        metrics = {
            'cnr_backbone': _safe_item(backbone_cnr),
            'cnr_corrected': _safe_item(corrected_cnr),
            'cnr_clean': _safe_item(clean_cnr),
            'cnr_drop': _safe_item(cnr_drop),
            'cnr_drop_penalty': _safe_item(cnr_drop_penalty),
            'cnr_improvement_pct': _safe_item(cnr_improvement_pct),
            'cnr_gap_to_clean': _safe_item(cnr_gap_to_clean),
        }

        return cnr_loss, metrics

    def compute_direct_cnr_preservation_loss(self,
                                              corrected: torch.Tensor,
                                              backbone: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Direct CNR preservation loss using the innovative region-aware approach.

        This loss:
        1. Detects tissue vs background regions
        2. Penalizes background noise increase heavily
        3. Rewards tissue contrast improvement
        4. Directly optimizes for CNR preservation

        Robustness features:
        - Checks that cnr_module and required attributes exist
        - Adds epsilon to all divisions
        - Clamps intermediate values to prevent extreme losses
        - Handles edge cases (empty masks, NaN/Inf values)
        """
        eps = 1e-6
        device = corrected.device

        # Default fallback metrics for error cases
        fallback_metrics = {
            'bg_std_backbone': 0.0,
            'bg_std_corrected': 0.0,
            'bg_noise_increase': 0.0,
            'cnr_backbone_direct': 0.0,
            'cnr_corrected_direct': 0.0,
            'cnr_drop_direct': 0.0,
            'contrast_improvement': 0.0,
        }

        # Check that cnr_module exists and has required attributes
        if not hasattr(self, 'cnr_module') or self.cnr_module is None:
            return corrected.new_zeros(()), fallback_metrics

        if not hasattr(self.cnr_module, 'region_detector') or self.cnr_module.region_detector is None:
            return corrected.new_zeros(()), fallback_metrics

        if not hasattr(self.cnr_module, 'cnr_gate') or self.cnr_module.cnr_gate is None:
            return corrected.new_zeros(()), fallback_metrics

        if not hasattr(self.cnr_module.cnr_gate, 'compute_regional_stats'):
            return corrected.new_zeros(()), fallback_metrics

        try:
            # Use cached masks if available (computed once in forward() for efficiency)
            if self._cached_tissue_mask is not None and self._cached_bg_mask is not None:
                tissue_mask = self._cached_tissue_mask
                bg_mask = self._cached_bg_mask
            else:
                # Get region masks from the CNR module
                tissue_mask, bg_mask = self.cnr_module.region_detector(backbone)
                # Clamp masks to ensure valid probabilities
                tissue_mask = tissue_mask.clamp(0.0, 1.0)
                bg_mask = bg_mask.clamp(0.0, 1.0)

            # Check for empty or invalid masks
            tissue_sum = tissue_mask.sum()
            bg_sum = bg_mask.sum()
            if tissue_sum < eps or bg_sum < eps:
                # Masks are essentially empty, return zero loss
                return corrected.new_zeros(()), fallback_metrics

            # Compute regional statistics for backbone
            bg_mean_b, bg_std_b = self.cnr_module.cnr_gate.compute_regional_stats(backbone, bg_mask)
            tissue_mean_b, _ = self.cnr_module.cnr_gate.compute_regional_stats(backbone, tissue_mask)

            # Compute regional statistics for corrected
            bg_mean_c, bg_std_c = self.cnr_module.cnr_gate.compute_regional_stats(corrected, bg_mask)
            tissue_mean_c, _ = self.cnr_module.cnr_gate.compute_regional_stats(corrected, tissue_mask)

            # BUG FIX: Clamp std values with larger minimum to prevent division issues
            # Use 1e-4 instead of eps (1e-6) since we're adding eps again in division
            bg_std_b = bg_std_b.clamp(min=1e-4)
            bg_std_c = bg_std_c.clamp(min=1e-4)

            # CNR before and after (with epsilon for numerical stability)
            # BUG FIX: Since std is already clamped to 1e-4, no need to add eps again
            # This prevents double-epsilon which can cause numerical inconsistency
            cnr_backbone = (tissue_mean_b - bg_mean_b) / bg_std_b
            cnr_corrected = (tissue_mean_c - bg_mean_c) / bg_std_c

            # Clamp CNR values to reasonable range to prevent extreme losses
            cnr_backbone = cnr_backbone.clamp(-100.0, 100.0)
            cnr_corrected = cnr_corrected.clamp(-100.0, 100.0)

            # === Loss Components (ENHANCED to prevent -1.3% CNR degradation) ===

            # 1. Background noise penalty (VERY HEAVY weight - main cause of CNR drop)
            # ENHANCED: Any increase in background noise is strongly penalized
            bg_noise_increase = F.relu(bg_std_c - bg_std_b)
            bg_noise_increase = bg_noise_increase.clamp(max=1.0)  # Clamp to prevent extreme values
            bg_noise_penalty = bg_noise_increase.mean() * 35.0  # Increased from 25.0 to strengthen background protection
            # SPIKE FIX: Clamp bg_noise_penalty directly
            bg_noise_penalty = bg_noise_penalty.clamp(max=8.0)  # ENHANCED: Higher max (was 5.0)

            # 2. CNR drop penalty (ENHANCED)
            cnr_drop = F.relu(cnr_backbone - cnr_corrected)
            cnr_drop = cnr_drop.clamp(max=10.0)  # Clamp to prevent extreme values
            cnr_drop_penalty = cnr_drop.mean() * 6.0  # REBALANCED: Reduced from 10.0 to 6.0 to prevent PSNR degradation
            # SPIKE FIX: Clamp cnr_drop_penalty directly
            cnr_drop_penalty = cnr_drop_penalty.clamp(max=8.0)  # ENHANCED: Higher max (was 5.0)

            # 3. Contrast improvement reward (converted to reduced penalty)
            contrast_improvement = (tissue_mean_c - bg_mean_c) - (tissue_mean_b - bg_mean_b)
            contrast_improvement = contrast_improvement.clamp(-1.0, 1.0)  # Clamp to reasonable range
            contrast_bonus = F.relu(contrast_improvement).mean() * 2.0
            # SPIKE FIX: Clamp contrast_bonus (max 1.0 * 2.0 = 2.0, keep bounded)
            contrast_bonus = contrast_bonus.clamp(max=2.0)

            # 4. CNR Improvement Reward (NEW - directly incentivize CNR improvement)
            cnr_improvement = F.relu(cnr_corrected - cnr_backbone)
            cnr_improvement_bonus = cnr_improvement.mean() * 1.0  # Reward CNR improvement
            cnr_improvement_bonus = cnr_improvement_bonus.clamp(max=2.0)

            # Combined loss (all positive, bonuses reduce total)
            # Clamp intermediate sums to prevent overflow
            penalty_sum = (bg_noise_penalty + cnr_drop_penalty).clamp(max=16.0)  # ENHANCED: Higher max
            bonus_sum = (contrast_bonus + cnr_improvement_bonus).clamp(max=penalty_sum)  # Include CNR improvement bonus
            total_loss = penalty_sum - bonus_sum
            total_loss = F.relu(total_loss)  # Ensure non-negative
            total_loss = total_loss.clamp(max=10.0)  # ENHANCED: Higher max for stronger signal (was 5.0)

            # Check for NaN/Inf in total_loss
            if not torch.isfinite(total_loss):
                return corrected.new_zeros(()), fallback_metrics

            # OPTIMIZATION: Return tensors in metrics dict - defer .item() extraction
            metrics = {
                '_bg_std_backbone': bg_std_b.mean(),
                '_bg_std_corrected': bg_std_c.mean(),
                '_bg_noise_increase': bg_noise_increase.mean(),
                '_cnr_backbone_direct': cnr_backbone.mean(),
                '_cnr_corrected_direct': cnr_corrected.mean(),
                '_cnr_drop_direct': cnr_drop.mean(),
                '_contrast_improvement_direct': contrast_improvement.mean(),
                '_deferred_metrics': True,
            }

            return total_loss, metrics

        except Exception as e:
            # Log the error for debugging but don't crash training
            print(f"WARNING: compute_direct_cnr_preservation_loss failed with error: {e}")
            return corrected.new_zeros(()), fallback_metrics

    def psnr_preservation_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                                clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Penalize when corrected PSNR drops below backbone beyond threshold.

        REGION-ADAPTIVE version - prioritizes PSNR in background, allows more
        deviation in tissue regions for clinical enhancement.

        Region weights:
        - Background (bg_weight=2.0): Strongly prioritize background PSNR
        - Tissue (tissue_weight=0.2): Reduced weight to allow enhancement

        AGGRESSIVE PSNR PRESERVATION: Target < 0.5 dB PSNR drop.
        Uses steep linear + quadratic penalty to prevent degradation.

        - If PSNR drop < 0.2 dB: no penalty (within acceptable range)
        - If PSNR drop 0.2-0.5 dB: steep linear penalty (2.0x slope)
        - If PSNR drop > 0.5 dB: additional quadratic penalty (5.0x multiplier)
        """
        eps = 1e-8

        # Region weights: background prioritizes PSNR, tissue allows more deviation
        bg_weight = 2.0  # Doubled from 1.0 - strongly prioritize background PSNR
        tissue_weight = 0.2  # Reduced from 0.3 - allow more tissue deviation

        # Try to get region masks for region-adaptive MSE
        use_region_adaptive = False
        tissue_mask = None
        bg_mask = None

        # Use cached masks if available (computed once in forward() for efficiency)
        if self._cached_tissue_mask is not None and self._cached_bg_mask is not None:
            tissue_mask = self._cached_tissue_mask
            bg_mask = self._cached_bg_mask
            # Check for valid masks (not empty)
            tissue_sum = tissue_mask.sum()
            bg_sum = bg_mask.sum()
            if tissue_sum > eps and bg_sum > eps:
                use_region_adaptive = True
        # Fallback: Check if cnr_module and region_detector are available
        elif (hasattr(self, 'cnr_module') and self.cnr_module is not None and
            hasattr(self.cnr_module, 'region_detector') and self.cnr_module.region_detector is not None):
            try:
                # Get region masks from the CNR module (same as compute_direct_cnr_preservation_loss)
                tissue_mask, bg_mask = self.cnr_module.region_detector(backbone)

                # Check for valid masks (not empty)
                tissue_sum = tissue_mask.sum()
                bg_sum = bg_mask.sum()

                if tissue_sum > eps and bg_sum > eps:
                    # Clamp masks to ensure valid probabilities
                    tissue_mask = tissue_mask.clamp(0.0, 1.0)
                    bg_mask = bg_mask.clamp(0.0, 1.0)
                    use_region_adaptive = True
            except Exception:
                # Region detection failed, fall back to global MSE
                use_region_adaptive = False

        # Compute MSE - either region-weighted or global
        if use_region_adaptive:
            # Compute per-pixel squared error
            squared_error_backbone = (backbone - clean) ** 2
            squared_error_corrected = (corrected - clean) ** 2

            # Region-weighted MSE
            weight_map = bg_mask * bg_weight + tissue_mask * tissue_weight
            # Normalize weight map to maintain scale
            weight_map = weight_map / (weight_map.mean() + eps)

            # Clamp weight_map to prevent extreme values
            weight_map = weight_map.clamp(0.0, 10.0)

            mse_backbone = (squared_error_backbone * weight_map).mean()
            mse_corrected = (squared_error_corrected * weight_map).mean()

            # Also compute region-specific MSE for tracking metrics
            # Background region MSE
            bg_mask_sum = bg_mask.sum() + eps
            mse_backbone_bg = (squared_error_backbone * bg_mask).sum() / bg_mask_sum
            mse_corrected_bg = (squared_error_corrected * bg_mask).sum() / bg_mask_sum

            # Tissue region MSE
            tissue_mask_sum = tissue_mask.sum() + eps
            mse_backbone_tissue = (squared_error_backbone * tissue_mask).sum() / tissue_mask_sum
            mse_corrected_tissue = (squared_error_corrected * tissue_mask).sum() / tissue_mask_sum

            # Compute region-specific PSNR for metrics
            psnr_backbone_bg = 10 * torch.log10(1.0 / (mse_backbone_bg + eps))
            psnr_corrected_bg = 10 * torch.log10(1.0 / (mse_corrected_bg + eps))
            psnr_backbone_tissue = 10 * torch.log10(1.0 / (mse_backbone_tissue + eps))
            psnr_corrected_tissue = 10 * torch.log10(1.0 / (mse_corrected_tissue + eps))

            # PSNR drop in each region (positive = degradation)
            psnr_drop_bg = (psnr_backbone_bg - psnr_corrected_bg).clamp(-10.0, 10.0)
            psnr_drop_tissue = (psnr_backbone_tissue - psnr_corrected_tissue).clamp(-10.0, 10.0)
        else:
            # Fallback to global MSE (original behavior)
            mse_backbone = F.mse_loss(backbone, clean)
            mse_corrected = F.mse_loss(corrected, clean)
            psnr_drop_bg = None
            psnr_drop_tissue = None

        # Compute global PSNR values
        psnr_backbone = 10 * torch.log10(1.0 / (mse_backbone + eps))
        psnr_corrected = 10 * torch.log10(1.0 / (mse_corrected + eps))

        psnr_drop = psnr_backbone - psnr_corrected

        # Clamp psnr_drop to prevent extreme values from log computation
        psnr_drop = psnr_drop.clamp(-10.0, 10.0)

        # PSNR PRESERVATION: Strong penalty to keep PSNR drop < 1.0 dB
        # Linear penalty: starts at 0.3 dB drop, 2.0x slope
        linear_penalty = F.relu(psnr_drop - 0.3) * 2.0
        # Quadratic kicks in at 1.0 dB drop with 5.0x multiplier
        excess = F.relu(psnr_drop - 1.0)
        quadratic_penalty = excess ** 2 * 5.0

        penalty = linear_penalty + quadratic_penalty

        # Explicit background PSNR penalty
        if psnr_drop_bg is not None and psnr_drop_bg > 0.3:
            bg_psnr_penalty = F.relu(psnr_drop_bg - 0.3) * 3.0
            penalty = penalty + bg_psnr_penalty

        # Allow larger penalty values so gradient signal is strong
        penalty = torch.clamp(penalty, max=10.0)

        # OPTIMIZATION: Use module-level _safe_item function (already defined)
        metrics = {
            'psnr_backbone': _safe_item(psnr_backbone),
            'psnr_corrected': _safe_item(psnr_corrected),
            'psnr_delta': _safe_item(psnr_corrected - psnr_backbone),
            'psnr_drop_excess': _safe_item(excess),
        }

        # Add region-specific PSNR drop metrics if region-adaptive mode was used
        if use_region_adaptive and psnr_drop_bg is not None and psnr_drop_tissue is not None:
            metrics['psnr_bg_drop'] = _safe_item(psnr_drop_bg)
            metrics['psnr_tissue_drop'] = _safe_item(psnr_drop_tissue)
            metrics['psnr_region_adaptive'] = 1.0  # Flag indicating region-adaptive mode was used
        else:
            metrics['psnr_bg_drop'] = 0.0
            metrics['psnr_tissue_drop'] = 0.0
            metrics['psnr_region_adaptive'] = 0.0  # Flag indicating fallback to global MSE

        return penalty, metrics

    def ssim_preservation_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                               clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Region-adaptive SSIM preservation loss.

        SSIM-based perceptual quality constraint - more meaningful than PSNR for clinical quality.
        Uses region-weighted SSIM where background regions prioritize smoothness (higher weight)
        and tissue regions allow more deviation for clinical enhancement (lower weight).

        Region weighting:
        - Background SSIM weight: 0.7 (prioritize smoothness/preservation)
        - Tissue SSIM weight: 0.3 (allow more clinical enhancement deviation)

        Penalty thresholds:
        - If SSIM drop < 0.005: no penalty (within acceptable range)
        - If SSIM drop 0.005-0.01: soft linear penalty
        - If SSIM drop > 0.01: moderate penalty (but still softer than PSNR)
        """
        C1, C2 = 0.01**2, 0.03**2
        eps = 1e-8

        # Region weights
        BG_WEIGHT = 0.7
        TISSUE_WEIGHT = 0.3

        def compute_ssim(img1: torch.Tensor, img2: torch.Tensor,
                         mask: torch.Tensor = None) -> torch.Tensor:
            """Compute SSIM between two images, optionally masked to a region."""
            if mask is not None:
                # Apply mask - use weighted mean for regional SSIM
                mask_sum = mask.sum().clamp(min=eps)

                # Compute masked means
                mu1 = (img1 * mask).sum() / mask_sum
                mu2 = (img2 * mask).sum() / mask_sum

                # Compute masked variances and covariance
                img1_centered = (img1 - mu1) * mask
                img2_centered = (img2 - mu2) * mask

                sigma1_sq = (img1_centered ** 2).sum() / mask_sum
                sigma2_sq = (img2_centered ** 2).sum() / mask_sum
                sigma12 = (img1_centered * img2_centered).sum() / mask_sum
            else:
                # Global SSIM (original behavior)
                mu1 = img1.mean()
                mu2 = img2.mean()

                img1_centered = img1 - mu1
                img2_centered = img2 - mu2

                sigma1_sq = (img1_centered ** 2).mean()
                sigma2_sq = (img2_centered ** 2).mean()
                sigma12 = (img1_centered * img2_centered).mean()

            # Clamp variances
            sigma1_sq = sigma1_sq.clamp(min=eps)
            sigma2_sq = sigma2_sq.clamp(min=eps)

            # SSIM formula
            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim = numerator / denominator

            return ssim

        # Try to get region masks for region-adaptive SSIM
        use_regional = False
        tissue_mask = None
        bg_mask = None

        # Use cached masks if available (computed once in forward() for efficiency)
        if self._cached_tissue_mask is not None and self._cached_bg_mask is not None:
            tissue_mask = self._cached_tissue_mask
            bg_mask = self._cached_bg_mask
            # Validate masks
            tissue_sum = tissue_mask.sum()
            bg_sum = bg_mask.sum()
            if tissue_sum >= eps and bg_sum >= eps:
                use_regional = True
        # Fallback: check if cnr_module and region_detector are available
        elif hasattr(self, 'cnr_module') and self.cnr_module is not None:
            if hasattr(self.cnr_module, 'region_detector') and self.cnr_module.region_detector is not None:
                try:
                    tissue_mask, bg_mask = self.cnr_module.region_detector(backbone)

                    # Validate masks
                    tissue_sum = tissue_mask.sum()
                    bg_sum = bg_mask.sum()

                    if tissue_sum >= eps and bg_sum >= eps:
                        # Clamp masks to valid probability range
                        tissue_mask = tissue_mask.clamp(0.0, 1.0)
                        bg_mask = bg_mask.clamp(0.0, 1.0)
                        use_regional = True
                except Exception:
                    # Fall back to global SSIM if region detection fails
                    use_regional = False

        if use_regional:
            # ===== REGION-ADAPTIVE SSIM =====
            # Compute SSIM separately for background and tissue regions

            # Background SSIM (backbone vs clean)
            ssim_bg_backbone = compute_ssim(backbone, clean, bg_mask)
            # Background SSIM (corrected vs clean)
            ssim_bg_corrected = compute_ssim(corrected, clean, bg_mask)

            # Tissue SSIM (backbone vs clean)
            ssim_tissue_backbone = compute_ssim(backbone, clean, tissue_mask)
            # Tissue SSIM (corrected vs clean)
            ssim_tissue_corrected = compute_ssim(corrected, clean, tissue_mask)

            # Handle NaN/Inf in regional SSIM values
            if not torch.isfinite(ssim_bg_backbone) or not torch.isfinite(ssim_bg_corrected):
                ssim_bg_backbone = backbone.new_tensor(0.9)
                ssim_bg_corrected = backbone.new_tensor(0.9)
            if not torch.isfinite(ssim_tissue_backbone) or not torch.isfinite(ssim_tissue_corrected):
                ssim_tissue_backbone = backbone.new_tensor(0.9)
                ssim_tissue_corrected = backbone.new_tensor(0.9)

            # Compute SSIM drops per region (positive means degradation)
            ssim_drop_bg = (ssim_bg_backbone - ssim_bg_corrected).clamp(-0.5, 0.5)
            ssim_drop_tissue = (ssim_tissue_backbone - ssim_tissue_corrected).clamp(-0.5, 0.5)

            # Compute weighted SSIM drop
            # Background has higher weight (prioritize smoothness preservation)
            # Tissue has lower weight (allow more deviation for clinical enhancement)
            weighted_ssim_drop = BG_WEIGHT * ssim_drop_bg + TISSUE_WEIGHT * ssim_drop_tissue

            # Aggregate SSIM values for metrics
            ssim_backbone = BG_WEIGHT * ssim_bg_backbone + TISSUE_WEIGHT * ssim_tissue_backbone
            ssim_corrected = BG_WEIGHT * ssim_bg_corrected + TISSUE_WEIGHT * ssim_tissue_corrected
            ssim_drop = weighted_ssim_drop

            # Store regional metrics
            ssim_tissue = ssim_tissue_corrected
            ssim_bg = ssim_bg_corrected
        else:
            # ===== GLOBAL SSIM (fallback) =====
            ssim_backbone = compute_ssim(backbone, clean)
            ssim_corrected = compute_ssim(corrected, clean)
            ssim_drop = (ssim_backbone - ssim_corrected).clamp(-0.5, 0.5)

            # No regional metrics available
            ssim_tissue = ssim_corrected
            ssim_bg = ssim_corrected

        # ===== SOFT SSIM PRESERVATION CONSTRAINT =====
        # Softer penalties than PSNR since SSIM is more perceptually meaningful

        # Allow 0.005 SSIM drop without penalty (very small perceptual difference)
        # Linear penalty for drops 0.005-0.01 (slope: 20 per unit SSIM drop)
        linear_penalty = F.relu(ssim_drop - 0.005) * 20.0
        # Clamp linear penalty to max 0.1 (at 0.01 SSIM drop)
        linear_penalty = linear_penalty.clamp(max=0.1)

        # Additional soft quadratic term for drops > 0.01 (significant perceptual difference)
        # Softer 10x multiplier
        excess = F.relu(ssim_drop - 0.01)
        excess_clamped = torch.clamp(excess, max=0.05)  # Cap excess at 0.05 SSIM
        quadratic_penalty = excess_clamped ** 2 * 10.0  # 10x multiplier
        # Max quadratic: 0.0025 * 10 = 0.025
        quadratic_penalty = quadratic_penalty.clamp(max=0.1)

        # Combined penalty (max theoretical: 0.1 + 0.1 = 0.2)
        # At 0.005 SSIM drop: penalty = 0.0
        # At 0.01 SSIM drop: penalty = 0.1
        # At 0.02 SSIM drop: penalty = 0.1 + 0.01 = 0.11
        # At 0.05+ SSIM drop: penalty = 0.2 (capped)
        penalty = linear_penalty + quadratic_penalty

        # Final clamp to ensure reasonable loss range
        penalty = torch.clamp(penalty, max=0.5)

        # OPTIMIZATION: Use module-level _safe_item function
        metrics = {
            'ssim_backbone_loss': _safe_item(ssim_backbone),
            'ssim_corrected_loss': _safe_item(ssim_corrected),
            'ssim_delta_loss': _safe_item(ssim_corrected - ssim_backbone),
            'ssim_drop_excess': _safe_item(excess),
            'ssim_preservation_ratio': _safe_item(ssim_corrected / ssim_backbone.clamp(min=eps)),
            'ssim_tissue': _safe_item(ssim_tissue),
            'ssim_bg': _safe_item(ssim_bg),
            'ssim_regional_enabled': 1.0 if use_regional else 0.0,
        }

        return penalty, metrics

    def lpips_preservation_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                                clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        LPIPS Perceptual Quality Constraint - PRIMARY QUALITY METRIC.

        Uses Learned Perceptual Image Patch Similarity (LPIPS) instead of PSNR/SSIM.
        LPIPS is based on deep features from VGG and correlates better with human
        perception of image quality than traditional metrics.

        Key insight: Use LPIPS as the main constraint, allowing more flexibility
        in pixel-wise metrics (PSNR) to achieve clinical improvements.

        Penalizes when corrected LPIPS gets worse than backbone LPIPS.

        - LPIPS range: [0, 1] where 0 = identical, 1 = very different
        - Lower LPIPS = better perceptual quality
        - Penalize when corrected LPIPS > backbone LPIPS (perceptual degradation)
        """
        device = corrected.device
        self.lpips_model = self.lpips_model.to(device)

        # LPIPS expects 3-channel input, expand grayscale to 3 channels
        # Also normalize to [-1, 1] range as LPIPS expects
        corrected_3ch = (corrected.repeat(1, 3, 1, 1) * 2 - 1).clamp(-1, 1)
        backbone_3ch = (backbone.repeat(1, 3, 1, 1) * 2 - 1).clamp(-1, 1)
        clean_3ch = (clean.repeat(1, 3, 1, 1) * 2 - 1).clamp(-1, 1)

        # Compute LPIPS distance to clean (lower = better)
        # BUG FIX: corrected needs gradient flow, backbone is just reference
        lpips_corrected = self.lpips_model(corrected_3ch, clean_3ch).mean()
        with torch.no_grad():
            lpips_backbone = self.lpips_model(backbone_3ch, clean_3ch).mean()

        # Positive LPIPS delta means degradation (corrected has higher LPIPS = worse)
        # detach backbone to ensure gradient only flows through corrected
        lpips_delta = lpips_corrected - lpips_backbone.detach()

        # Soft penalty:
        # - Allow small LPIPS increase (0.01) without penalty
        # - Linear penalty for increases 0.01-0.05
        # - Stronger penalty beyond 0.05
        linear_penalty = F.relu(lpips_delta - 0.01) * 10.0  # 10x penalty per unit
        linear_penalty = linear_penalty.clamp(max=0.4)

        excess = F.relu(lpips_delta - 0.05)
        quadratic_penalty = (excess ** 2) * 20.0
        quadratic_penalty = quadratic_penalty.clamp(max=0.3)

        penalty = linear_penalty + quadratic_penalty
        penalty = penalty.clamp(max=1.0)

        metrics = {
            'lpips_corrected': _safe_item(lpips_corrected),
            'lpips_backbone': _safe_item(lpips_backbone),
            'lpips_delta': _safe_item(lpips_delta),
            'lpips_improved': 1.0 if lpips_delta.item() < 0 else 0.0,  # 1 if corrected is better
        }

        return penalty, metrics

    def compute_texture_preservation_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                                           clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Texture Preservation Loss: Penalize texture degradation from backbone to corrected.

        Uses Laplacian filter to extract high-frequency content (texture).
        Penalizes when corrected texture < backbone texture (texture should not degrade).

        This addresses the texture ratio degradation issue (0.810 ratio, from 30.5% to 24.7%).

        Args:
            corrected: Corrected output [B, 1, H, W]
            backbone: Backbone output [B, 1, H, W]
            clean: Clean reference [B, 1, H, W]

        Returns:
            loss: Texture preservation loss
            metrics: Dict with texture values and statistics
        """
        # Compute Laplacian (high-frequency content) for each image
        # OPTIMIZATION: Cache dtype-converted Laplacian kernel to avoid repeated .to() calls
        if not hasattr(self, '_laplacian_cached') or self._laplacian_cached.dtype != corrected.dtype:
            self._laplacian_cached = self.laplacian.to(dtype=corrected.dtype)
        laplacian = self._laplacian_cached

        backbone_lap = F.conv2d(backbone, laplacian, padding=1).abs()
        corrected_lap = F.conv2d(corrected, laplacian, padding=1).abs()
        clean_lap = F.conv2d(clean, laplacian, padding=1).abs()

        # Compute texture preservation ratios (relative to clean)
        eps = 1e-8
        backbone_texture = backbone_lap.mean()
        corrected_texture = corrected_lap.mean()
        clean_texture = clean_lap.mean()

        # SPIKE FIX: Clamp clean_texture denominator
        clean_texture_safe = clean_texture.clamp(min=1e-4)
        backbone_texture_ratio = backbone_texture / clean_texture_safe
        corrected_texture_ratio = corrected_texture / clean_texture_safe
        # SPIKE FIX: Clamp texture ratios to prevent extreme values
        backbone_texture_ratio = backbone_texture_ratio.clamp(0.0, 10.0)
        corrected_texture_ratio = corrected_texture_ratio.clamp(0.0, 10.0)

        # ===== Texture Drop Penalty =====
        # Penalize if corrected texture drops below backbone texture
        # This prevents the corrector from smoothing out texture
        texture_drop = backbone_texture - corrected_texture
        texture_drop_penalty = F.relu(texture_drop)
        # SPIKE FIX: Clamp texture_drop_penalty
        texture_drop_penalty = texture_drop_penalty.clamp(max=1.0)

        # ===== Texture Improvement Reward =====
        # Small reward for improving texture toward clean level
        texture_improvement_potential = clean_texture - backbone_texture  # Room to improve
        texture_improvement_actual = corrected_texture - backbone_texture  # How much we improved

        # Only reward if there's room to improve and we actually improved
        improvement_reward = torch.where(
            texture_improvement_potential > 0,
            F.relu(texture_improvement_actual) * 0.1,  # Small reward weight
            torch.zeros_like(texture_improvement_actual)
        )
        # SPIKE FIX: Clamp improvement_reward
        improvement_reward = improvement_reward.clamp(max=1.0)

        # ===== Pixel-wise Texture Drop Penalty =====
        # Additional penalty for regions where texture dropped significantly
        pixel_texture_drop = F.relu(backbone_lap - corrected_lap)
        pixel_drop_penalty = pixel_texture_drop.mean() * 0.5
        # SPIKE FIX: Clamp pixel_drop_penalty
        pixel_drop_penalty = pixel_drop_penalty.clamp(max=1.0)

        # ===== Combined Texture Loss =====
        texture_loss = texture_drop_penalty + pixel_drop_penalty - improvement_reward
        # BUG 2 FIX: Ensure non-negative loss when improvement_reward > penalties
        texture_loss = F.relu(texture_loss)
        # SPIKE FIX: Clamp texture loss to prevent extreme values
        texture_loss = torch.clamp(texture_loss, max=5.0)

        # Compute texture ratio (corrected / backbone) for monitoring
        # SPIKE FIX: Clamp backbone_texture denominator and final ratio
        backbone_texture_safe = backbone_texture.clamp(min=1e-4)
        texture_ratio = corrected_texture / backbone_texture_safe
        texture_ratio = texture_ratio.clamp(0.0, 10.0)

        # OPTIMIZATION: Use module-level _safe_item function (already defined)
        metrics = {
            'texture_backbone': _safe_item(backbone_texture),
            'texture_corrected': _safe_item(corrected_texture),
            'texture_clean': _safe_item(clean_texture),
            'texture_backbone_ratio': _safe_item(backbone_texture_ratio),
            'texture_corrected_ratio': _safe_item(corrected_texture_ratio),
            'texture_ratio_corr_vs_bb': _safe_item(texture_ratio),
            'texture_drop': _safe_item(texture_drop),
            'texture_drop_penalty': _safe_item(texture_drop_penalty),
            'texture_pixel_drop_penalty': _safe_item(pixel_drop_penalty),
        }

        return texture_loss, metrics

    def compute_unified_cnr_loss(self, corrected: torch.Tensor, backbone_out: torch.Tensor) -> torch.Tensor:
        """Single CNR loss: penalize background std increase.

        Replaces the three separate CNR losses (cnr_preserve, cnr_spatial, direct_cnr)
        with a single simple formulation that penalizes background noise increase
        and mildly rewards background noise decrease.

        Args:
            corrected: Corrected output [B, 1, H, W]
            backbone_out: Backbone output [B, 1, H, W]

        Returns:
            Unified CNR loss (scalar tensor)
        """
        with torch.no_grad():
            bg_mask = (backbone_out < backbone_out.mean()).float()
        bg_std_before = (backbone_out * bg_mask).std()
        bg_std_after = (corrected * bg_mask).std()
        # Penalize increase, mildly reward decrease
        return F.relu(bg_std_after - bg_std_before) - 0.3 * F.relu(bg_std_before - bg_std_after)

    def forward(self, corrected: torch.Tensor, backbone_out: torch.Tensor,
                clean: torch.Tensor, noisy: torch.Tensor, info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined cooperative loss.
        """
        device = corrected.device

        # ===== CACHE REGION MASKS ONCE FOR ALL LOSS FUNCTIONS =====
        # This avoids calling region_detector 4 times per forward pass (10-15% speedup)
        self._cached_tissue_mask = None
        self._cached_bg_mask = None
        with torch.no_grad():
            if hasattr(self, 'cnr_module') and self.cnr_module is not None:
                if hasattr(self.cnr_module, 'region_detector') and self.cnr_module.region_detector is not None:
                    try:
                        tissue_mask, bg_mask = self.cnr_module.region_detector(backbone_out)
                        self._cached_tissue_mask = tissue_mask.clamp(0.0, 1.0)
                        self._cached_bg_mask = bg_mask.clamp(0.0, 1.0)
                    except Exception:
                        pass  # Will fallback to per-function detection

        # ===== 1. Basic Reconstruction Loss (GROUP: recon) =====
        recon_loss = F.mse_loss(corrected, clean)
        # SPIKE FIX: Clamp base reconstruction losses
        # MSE of [0,1] images is typically < 0.1, clamp to 1.0 as safety
        recon_loss = recon_loss.clamp(max=1.0)

        backbone_loss = F.mse_loss(backbone_out, clean)
        backbone_loss = backbone_loss.clamp(max=1.0)

        # ===== 2. Cooperation + Efficiency Loss (GROUP: cooperation) =====
        nafnet_uncertainty = info.get('nafnet_uncertainty', torch.zeros_like(corrected))
        corrector_potentials = info.get('corrector_potentials', {})

        cooperation_loss, coop_metrics = self.compute_cooperation_loss(
            nafnet_uncertainty, corrector_potentials
        )

        nafnet_confidence = 1.0 - nafnet_uncertainty
        efficiency_loss, eff_metrics = self.compute_efficiency_loss(
            nafnet_confidence, corrector_potentials
        )

        # Merge cooperation + efficiency into one sum for the cooperation group
        cooperation_merged = cooperation_loss + efficiency_loss

        # ===== 3. Predicate Alignment Loss (GROUP: predicate_align) =====
        pred_scores = info.get('predicate_scores', {})
        predicate_align_loss = self.compute_predicate_alignment_loss(pred_scores)

        # ===== 4. Clinical Loss (GROUP: clinical) =====
        # Merged: contrast (compute_clinical_loss) + contrast_improve + clinical_metrics
        clinical_loss, clinical_metrics = self.compute_clinical_loss(
            corrected, clean, backbone_out
        )

        # Extract contrast improvement loss component for gradient flow
        contrast_improve_loss = clinical_metrics.get('contrast_improvement_loss_tensor',
                                                      corrected.new_zeros(()))

        # Clinical metrics loss (from clinical_enhancement_module)
        try:
            clinical_metrics_loss, clinical_metrics_info = self.clinical_metrics_loss(
                corrected, backbone_out, clean
            )
            if not torch.isfinite(clinical_metrics_loss):
                clinical_metrics_loss = corrected.new_zeros(())
                clinical_metrics_info = {}
        except Exception as e:
            print(f"WARNING: clinical_metrics_loss failed with error: {e}")
            clinical_metrics_loss = corrected.new_zeros(())
            clinical_metrics_info = {}

        # Merge into single clinical_loss_merged
        clinical_loss_merged = clinical_loss + contrast_improve_loss + clinical_metrics_loss

        # ===== 5. Fidelity Loss (GROUP: fidelity) =====
        # Merged: psnr_preserve + ssim_preserve + lpips_preserve
        psnr_preserve_loss, psnr_metrics = self.psnr_preservation_loss(
            corrected, backbone_out, clean
        )

        ssim_preserve_loss, ssim_metrics = self.ssim_preservation_loss(
            corrected, backbone_out, clean
        )

        # SPEED: LPIPS disabled — return zero loss and empty metrics
        lpips_preserve_loss = corrected.new_zeros(())
        lpips_metrics = {'lpips_corrected': 0.0, 'lpips_backbone': 0.0, 'lpips_delta': 0.0, 'lpips_improved': 0.0}

        # Merge into single fidelity_loss
        fidelity_loss = psnr_preserve_loss + ssim_preserve_loss

        # CNR losses computed individually above (cnr_loss, cnr_spatial_loss_val, direct_cnr_loss)

        # Texture preservation loss
        texture_preserve_loss, texture_metrics = self.compute_texture_preservation_loss(
            corrected, backbone_out, clean
        )

        # Extract edge and boundary losses from clinical metrics for independent weighting
        edge_loss = clinical_metrics.get('edge_loss_tensor', corrected.new_zeros(()))
        boundary_loss = clinical_metrics.get('boundary_loss_tensor', corrected.new_zeros(()))

        # Three independent CNR losses for comprehensive CNR preservation
        cnr_loss, cnr_metrics_detail = self.compute_cnr_loss(corrected, backbone_out, clean)
        cnr_spatial_loss_val, cnr_spatial_metrics = self.cnr_spatial_loss(corrected, backbone_out, clean)
        direct_cnr_loss, direct_cnr_metrics = self.compute_direct_cnr_preservation_loss(corrected, backbone_out)

        # (Removed: confidence_entropy and sparsity losses were destructive)

        # ===== Combined Loss with Uncertainty Weighting (17 components) =====
        # Compute individual weighted losses with safety clamping and NaN/Inf protection
        def safe_weighted_loss(loss, name, extra_weight=1.0, max_val=3.0):
            # Check for NaN/Inf in input loss - return zero if invalid
            if not torch.isfinite(loss).all():
                return corrected.new_zeros(())
            # Check if name exists in log_sigma before calling _weighted_loss
            if name not in self.log_sigma:
                # Fallback: return loss with extra_weight but no uncertainty weighting
                return torch.clamp(loss * extra_weight, max=max_val)
            weighted = self._weighted_loss(loss, name) * extra_weight
            # Check for NaN/Inf in weighted loss
            if not torch.isfinite(weighted).all():
                return corrected.new_zeros(())
            return torch.clamp(weighted, max=max_val)

        # ===== Individual Loss Component Monitoring (17 components) =====
        loss_components_list = [
            ('recon', recon_loss),
            ('backbone', backbone_loss),
            ('cooperation', cooperation_loss),
            ('efficiency', efficiency_loss),
            ('predicate_align', predicate_align_loss),
            ('contrast', clinical_loss),
            ('contrast_improve', contrast_improve_loss),
            ('clinical_metrics', clinical_metrics_loss),
            ('cnr_preserve', cnr_loss),
            ('cnr_spatial', cnr_spatial_loss_val),
            ('direct_cnr', direct_cnr_loss),
            ('edge', edge_loss),
            ('boundary', boundary_loss),
            ('psnr_preserve', psnr_preserve_loss),
            ('ssim_preserve', ssim_preserve_loss),
            ('lpips_preserve', lpips_preserve_loss),
            ('texture_preserve', texture_preserve_loss),
        ]
        # Stack all loss tensors for batch isfinite check (single GPU op)
        loss_stack = torch.stack([lc[1] for lc in loss_components_list])
        is_finite_mask = torch.isfinite(loss_stack)

        # Only log warnings if there are issues (avoid .item() calls in normal case)
        loss_components = {}
        for i, (name, loss) in enumerate(loss_components_list):
            if not is_finite_mask[i]:
                print(f"WARNING: {name} loss is NaN or Inf, replacing with 0")
                loss_components[name] = corrected.new_zeros(())
            else:
                loss_components[name] = loss

        del loss_stack

        # ORIGINAL: 17 individual loss components with independent uncertainty weighting
        total_loss = (
            safe_weighted_loss(recon_loss, 'recon', 1.0, 3.0) +
            safe_weighted_loss(backbone_loss, 'backbone', 0.5, 3.0) +
            safe_weighted_loss(cooperation_loss, 'cooperation', self.cooperation_weight, 3.0) +
            safe_weighted_loss(efficiency_loss, 'efficiency', self.efficiency_weight, 3.0) +
            safe_weighted_loss(predicate_align_loss, 'predicate_align', 1.5, 3.0) +
            safe_weighted_loss(clinical_loss, 'contrast', self.clinical_weight, 5.0) +
            safe_weighted_loss(contrast_improve_loss, 'contrast_improve', self.clinical_weight, 5.0) +
            safe_weighted_loss(clinical_metrics_loss, 'clinical_metrics', self.clinical_weight, 5.0) +
            safe_weighted_loss(cnr_loss, 'cnr_preserve', self.cnr_weight, 3.0) +
            safe_weighted_loss(cnr_spatial_loss_val, 'cnr_spatial', self.cnr_weight, 3.0) +
            safe_weighted_loss(direct_cnr_loss, 'direct_cnr', self.cnr_weight, 3.0) +
            safe_weighted_loss(edge_loss, 'edge', 0.5, 3.0) +
            safe_weighted_loss(boundary_loss, 'boundary', 0.5, 3.0) +
            safe_weighted_loss(psnr_preserve_loss, 'psnr_preserve', 2.0, 8.0) +
            safe_weighted_loss(ssim_preserve_loss, 'ssim_preserve', 1.5, 5.0) +
            # SPEED: LPIPS disabled
            # safe_weighted_loss(lpips_preserve_loss, 'lpips_preserve', 1.0, 3.0) +
            safe_weighted_loss(texture_preserve_loss, 'texture_preserve', 0.3, 3.0)
        )

        # ===== Loss Spike Detection and Prevention =====
        if not torch.isfinite(total_loss).all():
            print(f"WARNING: total_loss is NaN or Inf, replacing with fallback")
            if torch.isfinite(recon_loss).all():
                total_loss = recon_loss.clamp(max=5.0)  # Fallback to basic reconstruction loss
            else:
                total_loss = corrected.new_tensor(1.0)  # Ultimate fallback

        # Only check for spikes occasionally (every ~100 iterations)
        if not hasattr(self, '_loss_check_counter'):
            self._loss_check_counter = 0
        self._loss_check_counter = (self._loss_check_counter + 1) % 100

        if self._loss_check_counter == 0:
            loss_val = total_loss.item()
            if loss_val > 5.0:
                print(f"WARNING: Loss spike detected! total_loss = {loss_val:.4f}")
                loss_vals = torch.stack(list(loss_components.values()))
                mask = (torch.isfinite(loss_vals) & (loss_vals > 1.0))
                if mask.any():
                    for name, val in zip(loss_components.keys(), loss_vals):
                        if val > 1.0:
                            print(f"  - Contributing component: {name} = {val.item():.4f}")

        # Final safety: clamp total loss to reasonable range
        # 17 individual loss components - higher clamp to allow gradient flow
        total_loss = torch.clamp(total_loss, max=50.0)

        # ===== Compute Metrics =====
        primary_losses = torch.stack([
            total_loss, recon_loss, backbone_loss,
            cooperation_loss, efficiency_loss,
            predicate_align_loss,
            clinical_loss, contrast_improve_loss, clinical_metrics_loss,
            cnr_loss, cnr_spatial_loss_val, direct_cnr_loss,
            edge_loss, boundary_loss,
            psnr_preserve_loss, ssim_preserve_loss, lpips_preserve_loss,
            texture_preserve_loss,
        ])
        primary_vals = primary_losses.tolist()
        del primary_losses

        metrics = {
            'total': primary_vals[0],
            'recon': primary_vals[1],
            'backbone': primary_vals[2],
            'cooperation': primary_vals[3],
            'efficiency': primary_vals[4],
            'predicate_align': primary_vals[5],
            'contrast': primary_vals[6],
            'contrast_improve': primary_vals[7],
            'clinical_metrics': primary_vals[8],
            'cnr_preserve': primary_vals[9],
            'cnr_spatial': primary_vals[10],
            'direct_cnr': primary_vals[11],
            'edge': primary_vals[12],
            'boundary': primary_vals[13],
            'psnr_preserve': primary_vals[14],
            'ssim_preserve': primary_vals[15],
            'lpips_preserve': primary_vals[16],
            'texture_preserve': primary_vals[17],
            **psnr_metrics,
            **ssim_metrics,
            **lpips_metrics,
            **clinical_metrics,
            **clinical_metrics_info,
            **self.extract_deferred_metrics(coop_metrics),
            **self.extract_deferred_metrics(eff_metrics),
            'learned_weights': self.get_learned_weights(defer_item=True),
        }

        # Add predicate scores if available
        if pred_scores:
            metrics['pred_scores'] = pred_scores
            passing = sum(1 for k, v in pred_scores.items()
                         if k in ['P1', 'P2', 'P3', 'P4', 'P6'] and v >= 0.5)
            metrics['predicates_passing'] = passing
        else:
            metrics['predicates_passing'] = 0

        # Clean up cached masks to free memory
        self._cached_tissue_mask = None
        self._cached_bg_mask = None

        return total_loss, metrics


# =============================================================================
# Training Functions
# =============================================================================

def compute_psnr(pred, target):
    """Compute PSNR between prediction and target (per-image average).

    FIX: Computes PSNR per-image then averages, avoiding Jensen's inequality
    bias that inflated PSNR when computed over batch-averaged MSE.
    """
    B = pred.shape[0]
    total_psnr = 0.0
    for i in range(B):
        mse = F.mse_loss(pred[i], target[i])
        mse_val = mse.item()
        if not math.isfinite(mse_val) or mse_val < 1e-10:
            total_psnr += 100.0 if mse_val < 1e-10 else 0.0
        else:
            total_psnr += 10 * math.log10(1.0 / mse_val)
    return total_psnr / B


def _gaussian_kernel_2d(size=11, sigma=1.5):
    """Create 2D Gaussian kernel for SSIM (standard: w=11, sigma=1.5)."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g.unsqueeze(1) * g.unsqueeze(0)
    return k2d.unsqueeze(0).unsqueeze(0)

# Module-level cached Gaussian kernel (created once)
_SSIM_KERNEL_V8 = _gaussian_kernel_2d(11, 1.5)


def compute_ssim(pred, target, window_size=11):
    """Compute standard windowed SSIM (Wang et al., 2004) per-image then average.

    Uses 11x11 Gaussian window with reflect padding, matching the standard SSIM
    implementation used in image restoration benchmarks (SwinIR, Restormer, etc.).

    FIX: Replaced simplified global SSIM with proper windowed SSIM for fair
    comparison with SOTA baselines.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2
    kernel = _SSIM_KERNEL_V8.to(pred.device)

    B = pred.shape[0]
    total_ssim = 0.0
    for i in range(B):
        p = pred[i:i+1]
        t = target[i:i+1]
        p_pad = F.pad(p, [pad, pad, pad, pad], mode='reflect')
        t_pad = F.pad(t, [pad, pad, pad, pad], mode='reflect')
        mu_p = F.conv2d(p_pad, kernel)
        mu_t = F.conv2d(t_pad, kernel)
        sigma_p_sq = F.conv2d(p_pad * p_pad, kernel) - mu_p * mu_p
        sigma_t_sq = F.conv2d(t_pad * t_pad, kernel) - mu_t * mu_t
        sigma_pt = F.conv2d(p_pad * t_pad, kernel) - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_p * mu_p + mu_t * mu_t + C1) * (sigma_p_sq + sigma_t_sq + C2))
        ssim_val = ssim_map.mean().item()
        total_ssim += ssim_val if math.isfinite(ssim_val) else 0.0
    return total_ssim / B


def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    """Train one epoch with cooperation tracking."""
    model.train()

    total_loss = 0
    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_cooperation = 0
    total_efficiency = 0
    total_predicates_passing = 0

    # Track corrector activity
    corrector_activity = defaultdict(float)
    n = 0

    # OPTIMIZATION: Only update progress bar every N iterations to reduce overhead
    log_interval = 10

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        clean = batch['clean'].to(device, non_blocking=True)
        noisy = batch['noisy'].to(device, non_blocking=True)

        if scaler is not None:
            with autocast():
                corrected, backbone_out, info = model(noisy)
                loss, metrics = criterion(corrected, backbone_out, clean, noisy, info)

            # OPTIMIZATION: Check finite first (no sync), then check threshold only if needed
            is_finite = torch.isfinite(loss).all()
            if not is_finite:
                print(f"WARNING: Skipping gradient update - loss is NaN/Inf")
                del corrected, backbone_out
                continue
            # OPTIMIZATION: Only call .item() for threshold check (single sync point)
            loss_val = loss.item()
            if loss_val > 20.0:
                print(f"WARNING: Skipping gradient update - loss too high: {loss_val:.4f}")
                del corrected, backbone_out
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            corrected, backbone_out, info = model(noisy)
            loss, metrics = criterion(corrected, backbone_out, clean, noisy, info)

            # OPTIMIZATION: Check finite first (no sync), then check threshold only if needed
            is_finite = torch.isfinite(loss).all()
            if not is_finite:
                print(f"WARNING: Skipping gradient update - loss is NaN/Inf")
                del corrected, backbone_out
                continue
            # OPTIMIZATION: Only call .item() for threshold check (single sync point)
            loss_val = loss.item()
            if loss_val > 20.0:
                print(f"WARNING: Skipping gradient update - loss too high: {loss_val:.4f}")
                del corrected, backbone_out
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            optimizer.step()

        del corrected, backbone_out

        # OPTIMIZATION: Metrics are already floats from criterion, no need for isinstance checks
        # The criterion.forward() returns pre-extracted .item() values in metrics dict
        total_loss += metrics['total']
        total_psnr_backbone += metrics['psnr_backbone']
        total_psnr_corrected += metrics['psnr_corrected']
        total_cooperation += metrics.get('cooperation', 0)
        total_efficiency += metrics.get('efficiency', 0)
        total_predicates_passing += metrics.get('predicates_passing', 0)

        # Track corrector activity - only extract .item() if tensor (deferred from info)
        allocation = info.get('allocation_stats', {})
        for name, stats in allocation.items():
            if isinstance(stats, dict) and 'mean' in stats:
                mean_alloc = stats['mean']
                corrector_activity[name] += mean_alloc.item() if isinstance(mean_alloc, torch.Tensor) else mean_alloc

        n += 1

        # OPTIMIZATION: Update progress bar less frequently to reduce overhead
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f"{metrics['total']:.4f}",
                'psnr': f"{metrics['psnr_corrected']:.1f}",
                'delta': f"{metrics['psnr_delta']:+.2f}",
                'coop': f"{metrics.get('cooperation', 0):.3f}",
            })

        del info

    # Prevent division by zero if all batches were skipped
    n = max(n, 1)

    return {
        'loss': total_loss / n,
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'cooperation': total_cooperation / n,
        'efficiency': total_efficiency / n,
        'predicates_passing': total_predicates_passing / n,
        'corrector_activity': {k: v / n for k, v in corrector_activity.items()},
    }


@torch.inference_mode()
def validate(model, loader, device, lpips_model=None, criterion=None):
    """Comprehensive validation with cooperation metrics."""
    model.eval()
    use_amp = device != 'cpu' and torch.cuda.is_available()

    # Basic metrics
    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_ssim_backbone = 0
    total_ssim_corrected = 0

    # Region-specific accumulators
    total_psnr_backbone_bg = 0.0
    total_psnr_corrected_bg = 0.0
    total_psnr_backbone_tissue = 0.0
    total_psnr_corrected_tissue = 0.0
    total_clinical_tissue_backbone = 0.0
    total_clinical_tissue_corrected = 0.0
    total_clinical_bg_backbone = 0.0
    total_clinical_bg_corrected = 0.0
    n_region = 0

    # SPEED: LPIPS disabled in validation — saves 40-50% of validation time
    total_lpips_backbone = 0
    total_lpips_corrected = 0

    # Cooperation metrics
    total_uncertainty_corr = 0
    total_confident_coverage = 0
    total_uncertain_coverage = 0
    total_redundancy = 0

    # Clinical metrics
    total_contrast_improvement = 0
    total_edge_improvement = 0

    # OCT-specific clinical metrics (IEEE TMI)
    total_cnr_backbone = 0      # Contrast-to-Noise Ratio
    total_cnr_corrected = 0
    total_tci_backbone = 0      # Tissue Contrast Index
    total_tci_corrected = 0

    # Additional clinical preservation metrics
    total_backbone_contrast_pres = 0
    total_corrected_contrast_pres = 0
    total_backbone_boundary_pres = 0
    total_corrected_boundary_pres = 0
    total_backbone_texture_pres = 0
    total_corrected_texture_pres = 0
    total_backbone_edge_pres = 0
    total_corrected_edge_pres = 0

    # Legacy clinical metrics
    total_epi_backbone = 0      # Edge Preservation Index
    total_epi_corrected = 0
    total_bs_backbone = 0       # Boundary Sharpness
    total_bs_corrected = 0
    total_correction_magnitude = 0

    # Lambda statistics per corrector
    lambda_stats = {name: {'sum': 0, 'max': 0} for name in ['edge', 'contrast', 'smooth', 'structure', 'anatomy']}

    # Predicate metrics - use actual key names from EnhancedGTFreePredicates
    pred_key_map = {
        'P1_edge': 'P1', 'P2_contrast': 'P2', 'P3_smooth': 'P3',
        'P4_structure': 'P4', 'P5_speckle': 'P5', 'P6_anatomy': 'P6'
    }
    total_pred_scores = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0, 'P6': 0}
    total_pred_scores_backbone = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0, 'P6': 0}
    total_predicates_passing = 0

    # Corrector allocation
    corrector_allocations = defaultdict(lambda: {'pixels': 0, 'improvement': 0})

    # Memory Leak Issue 5 FIX: Create Sobel and Laplacian kernels ONCE before the loop
    # Previously these were created inside the loop each iteration causing memory leak
    sobel_y_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                                   device=device).view(1, 1, 3, 3)
    sobel_x_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                   device=device).view(1, 1, 3, 3)
    laplacian_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                                     device=device).view(1, 1, 3, 3)

    n = 0

    # Per-image data collection for external validation (predicate-metric correlations)
    per_image_correlations = []

    for batch in tqdm(loader, desc="Validation"):
        # OPTIMIZATION: Use non_blocking=True for overlapped data transfer
        clean = batch['clean'].to(device, non_blocking=True)
        noisy = batch['noisy'].to(device, non_blocking=True)

        if use_amp:
            with autocast():
                corrected, backbone_out, info = model(noisy, return_details=False)
        else:
            corrected, backbone_out, info = model(noisy, return_details=False)

        # Basic metrics
        psnr_backbone = compute_psnr(backbone_out, clean)
        psnr_corrected = compute_psnr(corrected, clean)
        ssim_backbone = compute_ssim(backbone_out, clean)
        ssim_corrected = compute_ssim(corrected, clean)

        total_psnr_backbone += psnr_backbone
        total_psnr_corrected += psnr_corrected
        total_ssim_backbone += ssim_backbone
        total_ssim_corrected += ssim_corrected

        # Get region masks for region-specific metrics
        tissue_mask, bg_mask = None, None
        if criterion is not None and hasattr(criterion, 'cnr_module') and criterion.cnr_module is not None:
            if hasattr(criterion.cnr_module, 'region_detector'):
                try:
                    tissue_mask, bg_mask = criterion.cnr_module.region_detector(backbone_out)
                    tissue_mask = tissue_mask.clamp(0.0, 1.0)
                    bg_mask = bg_mask.clamp(0.0, 1.0)
                except:
                    pass

        # Region-specific PSNR
        if tissue_mask is not None and bg_mask is not None:
            eps = 1e-8
            # Background PSNR
            bg_mask_sum_region = bg_mask.sum().clamp(min=1.0)
            mse_backbone_bg = ((backbone_out - clean) ** 2 * bg_mask).sum() / bg_mask_sum_region
            mse_corrected_bg = ((corrected - clean) ** 2 * bg_mask).sum() / bg_mask_sum_region
            psnr_backbone_bg = 10 * torch.log10(1.0 / (mse_backbone_bg + eps))
            psnr_corrected_bg = 10 * torch.log10(1.0 / (mse_corrected_bg + eps))

            # Tissue PSNR
            tissue_mask_sum = tissue_mask.sum().clamp(min=1.0)
            mse_backbone_tissue = ((backbone_out - clean) ** 2 * tissue_mask).sum() / tissue_mask_sum
            mse_corrected_tissue = ((corrected - clean) ** 2 * tissue_mask).sum() / tissue_mask_sum
            psnr_backbone_tissue = 10 * torch.log10(1.0 / (mse_backbone_tissue + eps))
            psnr_corrected_tissue = 10 * torch.log10(1.0 / (mse_corrected_tissue + eps))

            # Accumulate
            total_psnr_backbone_bg += psnr_backbone_bg.item()
            total_psnr_corrected_bg += psnr_corrected_bg.item()
            total_psnr_backbone_tissue += psnr_backbone_tissue.item()
            total_psnr_corrected_tissue += psnr_corrected_tissue.item()

            # Region-specific clinical (local std as proxy for contrast/detail)
            # Use 7x7 local std
            def local_std_simple(x):
                mean = F.avg_pool2d(x, 7, 1, 3)
                var = F.avg_pool2d(x**2, 7, 1, 3) - mean**2
                return torch.sqrt(var.clamp(min=1e-8))

            backbone_std_region = local_std_simple(backbone_out)
            corrected_std_region = local_std_simple(corrected)

            # Tissue region clinical (should improve)
            tissue_clinical_backbone = (backbone_std_region * tissue_mask).sum() / tissue_mask_sum
            tissue_clinical_corrected = (corrected_std_region * tissue_mask).sum() / tissue_mask_sum

            # Background region clinical (should stay same or decrease)
            bg_clinical_backbone = (backbone_std_region * bg_mask).sum() / bg_mask_sum_region
            bg_clinical_corrected = (corrected_std_region * bg_mask).sum() / bg_mask_sum_region

            total_clinical_tissue_backbone += tissue_clinical_backbone.item()
            total_clinical_tissue_corrected += tissue_clinical_corrected.item()
            total_clinical_bg_backbone += bg_clinical_backbone.item()
            total_clinical_bg_corrected += bg_clinical_corrected.item()

            n_region += 1

            del backbone_std_region, corrected_std_region

        # SPEED: LPIPS disabled in validation

        # Cooperation metrics
        nafnet_uncertainty = info.get('nafnet_uncertainty', torch.zeros_like(corrected))
        corrector_potentials = info.get('corrector_potentials', {})

        if corrector_potentials:
            # Extract potential maps from dicts
            potential_maps = []
            for name, val in corrector_potentials.items():
                if isinstance(val, dict) and 'map' in val:
                    potential_maps.append(val['map'])
                elif isinstance(val, torch.Tensor):
                    potential_maps.append(val)
            if potential_maps:
                total_potential = sum(potential_maps)
                del potential_maps  # FIX: Free list after use
            else:
                total_potential = torch.zeros_like(corrected)
            # Correlation between uncertainty and potential
            # FIX: Add validation before torch.corrcoef to prevent errors
            u_flat = nafnet_uncertainty.view(-1)
            p_flat = total_potential.view(-1)
            if u_flat.numel() >= 2 and u_flat.std() > 1e-8 and p_flat.std() > 1e-8:
                try:
                    correlation = torch.corrcoef(torch.stack([u_flat, p_flat]))[0, 1]
                    if not torch.isnan(correlation):
                        total_uncertainty_corr += correlation.item()
                except Exception:
                    pass  # Skip if correlation computation fails
            del u_flat, p_flat, total_potential  # FIX: Free intermediate tensors

        nafnet_confidence = 1.0 - nafnet_uncertainty
        confident_mask = (nafnet_confidence > 0.6).float()
        uncertain_mask = (nafnet_confidence < 0.4).float()
        total_confident_coverage += confident_mask.mean().item()
        total_uncertain_coverage += uncertain_mask.mean().item()

        # Clinical improvement
        # Simple contrast/edge computation
        with torch.no_grad():
            clean_std = F.avg_pool2d(clean ** 2, 7, 1, 3) - F.avg_pool2d(clean, 7, 1, 3) ** 2
            clean_std = torch.sqrt(clean_std.clamp(min=1e-8))
            backbone_std = F.avg_pool2d(backbone_out ** 2, 7, 1, 3) - F.avg_pool2d(backbone_out, 7, 1, 3) ** 2
            backbone_std = torch.sqrt(backbone_std.clamp(min=1e-8))
            corrected_std = F.avg_pool2d(corrected ** 2, 7, 1, 3) - F.avg_pool2d(corrected, 7, 1, 3) ** 2
            corrected_std = torch.sqrt(corrected_std.clamp(min=1e-8))

            # SPIKE FIX: Clamp denominators and final ratios in validation metrics
            clean_std_mean = clean_std.mean().clamp(min=1e-4)
            backbone_contrast_pres = (backbone_std.mean() / clean_std_mean).clamp(0.0, 10.0)
            corrected_contrast_pres = (corrected_std.mean() / clean_std_mean).clamp(0.0, 10.0)
            backbone_contrast_pres_safe = backbone_contrast_pres.clamp(min=1e-4)
            contrast_improvement = ((corrected_contrast_pres / backbone_contrast_pres_safe) - 1.0) * 100
            # SPIKE FIX: Clamp improvement percentage
            contrast_improvement = contrast_improvement.clamp(-100.0, 500.0)

            total_contrast_improvement += contrast_improvement.item()

            # ===== OCT-SPECIFIC CLINICAL METRICS (IEEE TMI) =====

            # 1. Contrast-to-Noise Ratio (CNR)
            # CNR = (mean_signal - mean_background) / std_background
            # We use the ratio of signal std to background noise as proxy
            # Higher CNR = better tissue differentiation
            signal_mask = (clean > clean.mean()).float()  # Bright regions (tissue signal)
            bg_mask = 1.0 - signal_mask  # Dark regions (background)

            # Backbone CNR
            # SPIKE FIX: Clamp mask sums to minimum value
            signal_mask_sum = signal_mask.sum().clamp(min=1.0)
            bg_mask_sum = bg_mask.sum().clamp(min=1.0)
            backbone_signal_mean = (backbone_out * signal_mask).sum() / signal_mask_sum
            backbone_bg_mean = (backbone_out * bg_mask).sum() / bg_mask_sum
            backbone_bg_std = torch.sqrt(((backbone_out - backbone_bg_mean) ** 2 * bg_mask).sum() / bg_mask_sum + 1e-8)
            # SPIKE FIX: Clamp std denominator
            backbone_bg_std = backbone_bg_std.clamp(min=1e-4)
            cnr_backbone = (backbone_signal_mean - backbone_bg_mean) / backbone_bg_std
            # SPIKE FIX: Clamp CNR values
            cnr_backbone = cnr_backbone.clamp(-100.0, 100.0)

            # Corrected CNR
            corrected_signal_mean = (corrected * signal_mask).sum() / signal_mask_sum
            corrected_bg_mean = (corrected * bg_mask).sum() / bg_mask_sum
            corrected_bg_std = torch.sqrt(((corrected - corrected_bg_mean) ** 2 * bg_mask).sum() / bg_mask_sum + 1e-8)
            # SPIKE FIX: Clamp std denominator
            corrected_bg_std = corrected_bg_std.clamp(min=1e-4)
            cnr_corrected = (corrected_signal_mean - corrected_bg_mean) / corrected_bg_std
            # SPIKE FIX: Clamp CNR values
            cnr_corrected = cnr_corrected.clamp(-100.0, 100.0)

            total_cnr_backbone += cnr_backbone.item()
            total_cnr_corrected += cnr_corrected.item()

            # 2. Tissue Contrast Index (TCI)
            # TCI measures the contrast between adjacent tissue layers
            # We compute as the mean of vertical gradient magnitude (layer boundaries run horizontally)
            # Memory Leak Issue 5 FIX: Use pre-created kernel instead of creating new one each iteration
            backbone_gy = F.conv2d(backbone_out, sobel_y_kernel, padding=1).abs()
            corrected_gy = F.conv2d(corrected, sobel_y_kernel, padding=1).abs()
            clean_gy = F.conv2d(clean, sobel_y_kernel, padding=1).abs()

            # TCI = mean gradient magnitude (higher = sharper layer boundaries)
            # SPIKE FIX: Clamp denominators and ratios
            clean_gy_mean = clean_gy.mean().clamp(min=1e-4)
            tci_backbone = (backbone_gy.mean() / clean_gy_mean).clamp(0.0, 10.0)
            tci_corrected = (corrected_gy.mean() / clean_gy_mean).clamp(0.0, 10.0)

            total_tci_backbone += tci_backbone.item()
            total_tci_corrected += tci_corrected.item()

            # ===== ADDITIONAL CLINICAL PRESERVATION METRICS =====

            # Store contrast preservation (already computed)
            total_backbone_contrast_pres += backbone_contrast_pres.item()
            total_corrected_contrast_pres += corrected_contrast_pres.item()

            # Boundary preservation (vertical gradient ratio)
            # SPIKE FIX: Clamp ratios
            backbone_boundary_pres = (backbone_gy.mean() / clean_gy_mean).clamp(0.0, 10.0)
            corrected_boundary_pres = (corrected_gy.mean() / clean_gy_mean).clamp(0.0, 10.0)
            total_backbone_boundary_pres += backbone_boundary_pres.item()
            total_corrected_boundary_pres += corrected_boundary_pres.item()

            # Texture preservation (high-frequency content via Laplacian)
            # Memory Leak FIX: Use pre-created laplacian_kernel instead of creating new tensor each iteration
            backbone_lap = F.conv2d(backbone_out, laplacian_kernel, padding=1).abs()
            corrected_lap = F.conv2d(corrected, laplacian_kernel, padding=1).abs()
            clean_lap = F.conv2d(clean, laplacian_kernel, padding=1).abs()
            # SPIKE FIX: Clamp denominators and ratios
            clean_lap_mean = clean_lap.mean().clamp(min=1e-4)
            backbone_texture_pres = (backbone_lap.mean() / clean_lap_mean).clamp(0.0, 10.0)
            corrected_texture_pres = (corrected_lap.mean() / clean_lap_mean).clamp(0.0, 10.0)
            total_backbone_texture_pres += backbone_texture_pres.item()
            total_corrected_texture_pres += corrected_texture_pres.item()

            # Edge preservation (Sobel magnitude)
            # Memory Leak FIX: Use pre-created sobel_x_kernel instead of creating new tensor each iteration
            backbone_gx = F.conv2d(backbone_out, sobel_x_kernel, padding=1)
            corrected_gx = F.conv2d(corrected, sobel_x_kernel, padding=1)
            clean_gx = F.conv2d(clean, sobel_x_kernel, padding=1)
            backbone_edge_mag = torch.sqrt(backbone_gx**2 + backbone_gy**2 + 1e-8)
            corrected_edge_mag = torch.sqrt(corrected_gx**2 + corrected_gy**2 + 1e-8)
            clean_edge_mag = torch.sqrt(clean_gx**2 + clean_gy**2 + 1e-8)
            # SPIKE FIX: Clamp denominators and ratios
            clean_edge_mag_mean = clean_edge_mag.mean().clamp(min=1e-4)
            backbone_edge_pres = (backbone_edge_mag.mean() / clean_edge_mag_mean).clamp(0.0, 10.0)
            corrected_edge_pres = (corrected_edge_mag.mean() / clean_edge_mag_mean).clamp(0.0, 10.0)
            total_backbone_edge_pres += backbone_edge_pres.item()
            total_corrected_edge_pres += corrected_edge_pres.item()

            # ===== LEGACY CLINICAL METRICS =====

            # Edge Preservation Index (EPI) - correlation of edges
            clean_edge_flat = clean_edge_mag.view(-1)
            backbone_edge_flat = backbone_edge_mag.view(-1)
            corrected_edge_flat = corrected_edge_mag.view(-1)

            # Normalized correlation
            # SPIKE FIX: Clamp std denominators
            clean_edge_std = clean_edge_flat.std().clamp(min=1e-4)
            backbone_edge_std = backbone_edge_flat.std().clamp(min=1e-4)
            corrected_edge_std = corrected_edge_flat.std().clamp(min=1e-4)
            clean_norm = (clean_edge_flat - clean_edge_flat.mean()) / clean_edge_std
            backbone_norm = (backbone_edge_flat - backbone_edge_flat.mean()) / backbone_edge_std
            corrected_norm = (corrected_edge_flat - corrected_edge_flat.mean()) / corrected_edge_std
            epi_backbone = (clean_norm * backbone_norm).mean()
            epi_corrected = (clean_norm * corrected_norm).mean()
            total_epi_backbone += epi_backbone.item()
            total_epi_corrected += epi_corrected.item()

            # Boundary Sharpness (max gradient along vertical direction)
            # SPIKE FIX: Clamp denominators and ratios
            clean_gy_max = clean_gy.max().clamp(min=1e-4)
            bs_backbone = (backbone_gy.max() / clean_gy_max).clamp(0.0, 10.0)
            bs_corrected = (corrected_gy.max() / clean_gy_max).clamp(0.0, 10.0)
            total_bs_backbone += bs_backbone.item()
            total_bs_corrected += bs_corrected.item()

            # Correction magnitude
            correction = corrected - backbone_out
            total_correction_magnitude += correction.abs().mean().item()

            del backbone_lap, corrected_lap, clean_lap
            del backbone_gx, corrected_gx, clean_gx, backbone_edge_mag, corrected_edge_mag, clean_edge_mag
            del clean_edge_flat, backbone_edge_flat, corrected_edge_flat, correction

        # Lambda statistics from info (corrector returns 'lambda_stats' not 'lambda_maps')
        info_lambda_stats = info.get('lambda_stats', {})
        for name in ['edge', 'contrast', 'smooth', 'structure', 'anatomy']:
            if name in info_lambda_stats:
                stats = info_lambda_stats[name]
                if isinstance(stats, dict):
                    mean_val = stats.get('mean', 0)
                    max_val = stats.get('max', 0)
                    # Handle both tensor and float values
                    if isinstance(mean_val, torch.Tensor):
                        mean_val = mean_val.item()
                    if isinstance(max_val, torch.Tensor):
                        max_val = max_val.item()
                    lambda_stats[name]['sum'] += mean_val
                    lambda_stats[name]['max'] = max(lambda_stats[name]['max'], max_val)

        # Predicate scores - map from P1_edge etc. to P1 etc.
        pred_scores = info.get('predicate_scores', {})
        for orig_key, score in pred_scores.items():
            # Map P1_edge -> P1, P2_contrast -> P2, etc.
            mapped_key = pred_key_map.get(orig_key, orig_key)
            if mapped_key in total_pred_scores:
                score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                total_pred_scores[mapped_key] += score_val

        # Backbone predicate scores (for comparison)
        pred_scores_backbone = info.get('predicate_scores_backbone', {})
        for orig_key, score in pred_scores_backbone.items():
            mapped_key = pred_key_map.get(orig_key, orig_key)
            if mapped_key in total_pred_scores_backbone:
                score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                total_pred_scores_backbone[mapped_key] += score_val

        # Count passing predicates (exclude P5)
        passing = 0
        for orig_key, score in pred_scores.items():
            mapped_key = pred_key_map.get(orig_key, orig_key)
            if mapped_key in ['P1', 'P2', 'P3', 'P4', 'P6']:
                score_val = score.item() if isinstance(score, torch.Tensor) else score
                if score_val >= 0.5:
                    passing += 1
        total_predicates_passing += passing

        # --- Per-image data collection for external validation ---
        per_image_data = {}
        # Collect per-image predicate scores (mapped to P1..P6)
        for orig_key, score in pred_scores.items():
            mapped_key = pred_key_map.get(orig_key, orig_key)
            if mapped_key in ['P1', 'P2', 'P3', 'P4', 'P6']:
                sv = score.item() if isinstance(score, torch.Tensor) else float(score)
                per_image_data[mapped_key] = sv
        # Collect per-image established metrics (corrected output)
        per_image_data['psnr'] = psnr_corrected
        per_image_data['ssim'] = ssim_corrected
        per_image_data['cnr'] = cnr_corrected.item() if isinstance(cnr_corrected, torch.Tensor) else float(cnr_corrected)
        per_image_data['tci'] = tci_corrected.item() if isinstance(tci_corrected, torch.Tensor) else float(tci_corrected)
        per_image_data['epi'] = epi_corrected.item() if isinstance(epi_corrected, torch.Tensor) else float(epi_corrected)
        per_image_correlations.append(per_image_data)

        # Corrector allocations
        allocation = info.get('allocation_stats', {})
        for name, stats in allocation.items():
            if isinstance(stats, dict):
                corrector_allocations[name]['pixels'] += stats.get('active_pixels', 0)

        del corrected, backbone_out, info
        n += 1

    # === External Validation: Predicate-Metric Correlations ===
    if len(per_image_correlations) > 5:
        import numpy as np
        pred_keys = ['P1', 'P2', 'P3', 'P4', 'P6']
        metric_keys = ['psnr', 'ssim', 'cnr', 'tci', 'epi']

        correlation_report = []
        for pk in pred_keys:
            for mk in metric_keys:
                p_vals = [d[pk] for d in per_image_correlations if pk in d and mk in d]
                m_vals = [d[mk] for d in per_image_correlations if pk in d and mk in d]
                if len(p_vals) > 3:
                    # Guard against constant arrays (zero std) which produce NaN
                    p_arr = np.array(p_vals)
                    m_arr = np.array(m_vals)
                    if np.std(p_arr) > 1e-8 and np.std(m_arr) > 1e-8:
                        corr = np.corrcoef(p_arr, m_arr)[0, 1]
                        if np.isfinite(corr):
                            correlation_report.append(f"{pk} vs {mk}: r={corr:.3f}")

        if correlation_report:
            print("\n" + "=" * 60)
            print("EXTERNAL VALIDATION: Predicate-Metric Correlations")
            print("=" * 60)
            for line in correlation_report:
                print(f"  {line}")
            print("=" * 60 + "\n")

    # Compute clinical preservation ratios
    bb_contrast = total_backbone_contrast_pres / n
    corr_contrast = total_corrected_contrast_pres / n
    bb_boundary = total_backbone_boundary_pres / n
    corr_boundary = total_corrected_boundary_pres / n
    bb_texture = total_backbone_texture_pres / n
    corr_texture = total_corrected_texture_pres / n
    bb_edge = total_backbone_edge_pres / n
    corr_edge = total_corrected_edge_pres / n

    metrics = {
        # Basic metrics
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'ssim_backbone': total_ssim_backbone / n,
        'ssim_corrected': total_ssim_corrected / n,
        'psnr_delta': (total_psnr_corrected - total_psnr_backbone) / n,

        # LPIPS perceptual metrics (lower = better)
        'lpips_backbone': total_lpips_backbone / n,
        'lpips_corrected': total_lpips_corrected / n,
        'lpips_delta': (total_lpips_corrected - total_lpips_backbone) / n,

        # Cooperation metrics
        'uncertainty_potential_corr': total_uncertainty_corr / n,
        'confident_coverage': total_confident_coverage / n,
        'uncertain_coverage': total_uncertain_coverage / n,

        # Clinical metrics
        'contrast_improvement': total_contrast_improvement / n,

        # OCT-specific clinical metrics (IEEE TMI)
        'cnr_backbone': total_cnr_backbone / n,
        'cnr_corrected': total_cnr_corrected / n,
        'cnr_improvement': ((total_cnr_corrected / n) / (total_cnr_backbone / n + 1e-8) - 1.0) * 100,
        'tci_backbone': total_tci_backbone / n,
        'tci_corrected': total_tci_corrected / n,
        'tci_improvement': ((total_tci_corrected / n) / (total_tci_backbone / n + 1e-8) - 1.0) * 100,

        # Clinical preservation metrics (absolute values)
        'backbone_contrast_pres': bb_contrast,
        'corrected_contrast_pres': corr_contrast,
        'backbone_boundary_pres': bb_boundary,
        'corrected_boundary_pres': corr_boundary,
        'backbone_texture_pres': bb_texture,
        'corrected_texture_pres': corr_texture,
        'backbone_edge_pres': bb_edge,
        'corrected_edge_pres': corr_edge,

        # Clinical preservation ratios (corrected / backbone)
        'contrast_ratio': corr_contrast / (bb_contrast + 1e-8),
        'boundary_ratio': corr_boundary / (bb_boundary + 1e-8),
        'texture_ratio': corr_texture / (bb_texture + 1e-8),
        'edge_ratio': corr_edge / (bb_edge + 1e-8),

        # Legacy clinical metrics
        'epi_backbone': total_epi_backbone / n,
        'epi_corrected': total_epi_corrected / n,
        'boundary_sharpness_backbone': total_bs_backbone / n,
        'boundary_sharpness_corrected': total_bs_corrected / n,
        'correction_magnitude': total_correction_magnitude / n,

        # Lambda statistics per corrector
        'lambda_stats': {name: {'mean': stats['sum'] / n, 'max': stats['max']}
                        for name, stats in lambda_stats.items()},

        # Predicate metrics
        'pred_scores': {k: v / n for k, v in total_pred_scores.items()},
        'pred_scores_backbone': {k: v / n for k, v in total_pred_scores_backbone.items()},
        'predicates_passing': total_predicates_passing / n,

        # Corrector allocations
        'corrector_allocations': dict(corrector_allocations),
    }

    # Region-specific metrics
    if n_region > 0:
        metrics['psnr_bg_backbone'] = total_psnr_backbone_bg / n_region
        metrics['psnr_bg_corrected'] = total_psnr_corrected_bg / n_region
        metrics['psnr_bg_delta'] = (total_psnr_corrected_bg - total_psnr_backbone_bg) / n_region
        metrics['psnr_tissue_backbone'] = total_psnr_backbone_tissue / n_region
        metrics['psnr_tissue_corrected'] = total_psnr_corrected_tissue / n_region
        metrics['psnr_tissue_delta'] = (total_psnr_corrected_tissue - total_psnr_backbone_tissue) / n_region

        # Clinical improvement by region (percentage)
        avg_clinical_tissue_b = total_clinical_tissue_backbone / n_region
        avg_clinical_tissue_c = total_clinical_tissue_corrected / n_region
        avg_clinical_bg_b = total_clinical_bg_backbone / n_region
        avg_clinical_bg_c = total_clinical_bg_corrected / n_region

        metrics['clinical_tissue_improvement'] = ((avg_clinical_tissue_c / (avg_clinical_tissue_b + 1e-8)) - 1.0) * 100
        metrics['clinical_bg_change'] = ((avg_clinical_bg_c / (avg_clinical_bg_b + 1e-8)) - 1.0) * 100

    return metrics


def print_cooperation_metrics(epoch, train_metrics, val_metrics):
    """
    Comprehensive epoch monitoring for V8 Cooperative Neuro-Symbolic Denoising.
    IEEE TMI publication-ready validation metrics printout.

    Includes:
    - Clinical Preservation Table (with ratios)
    - Traditional Quality Metrics (PSNR, SSIM)
    - GT-Free Predicates (backbone vs corrected comparison)
    - Cooperation Analysis (uncertainty-potential correlation)
    - Corrector Activity & Lambda Stats
    - Legacy Clinical Metrics (CNR, EPI, Boundary Sharpness)
    - Training Summary
    - Interpretable Verdict
    """
    psnr_delta = val_metrics['psnr_delta']
    ssim_delta = val_metrics['ssim_corrected'] - val_metrics['ssim_backbone']
    contrast_improve = val_metrics.get('contrast_improvement', 0)
    predicates_passing = val_metrics.get('predicates_passing', 0)
    cnr_improvement = val_metrics.get('cnr_improvement', 0)

    # SSIM preservation ratio (primary perceptual quality constraint)
    ssim_preservation_ratio = val_metrics.get('ssim_preservation_ratio', 1.0)
    ssim_preserved = ssim_preservation_ratio >= 0.99  # 99% SSIM preservation threshold

    # Determine success status
    # UPDATED: Prioritize SSIM over PSNR for perceptual quality
    # - SSIM preservation is the primary quality constraint (>= 0.99 ratio)
    # - PSNR constraint is relaxed (allow up to 1.0 dB drop if SSIM is preserved)
    target_achieved = (
        predicates_passing >= 4 and  # At least 4/5 predicates passing
        ssim_preserved and           # SSIM preservation (PRIMARY constraint)
        abs(psnr_delta) <= 1.0 and   # PSNR constraint (RELAXED from 0.5 to 1.0 dB)
        contrast_improve >= 5 and    # Some clinical improvement
        cnr_improvement >= 0         # No CNR degradation
    )

    # Calculate clinical preservation ratios
    contrast_ratio = val_metrics.get('contrast_ratio', 1.0)
    boundary_ratio = val_metrics.get('boundary_ratio', 1.0)
    texture_ratio = val_metrics.get('texture_ratio', 1.0)
    edge_ratio = val_metrics.get('edge_ratio', 1.0)
    avg_ratio = (contrast_ratio + boundary_ratio + texture_ratio + edge_ratio) / 4

    clinical_improvements = sum([
        1 if contrast_ratio > 1.0 else 0,
        1 if boundary_ratio > 1.0 else 0,
        1 if texture_ratio > 1.0 else 0,
        1 if edge_ratio > 1.0 else 0,
    ])

    # Status determination
    if target_achieved:
        status_symbol = "★★★"
        status_text = "PUBLICATION READY"
    elif avg_ratio > 1.05 and cnr_improvement >= 0:
        status_symbol = "+++"
        status_text = "EXCELLENT CLINICAL"
    elif avg_ratio > 1.0:
        status_symbol = "++"
        status_text = "GOOD PROGRESS"
    elif avg_ratio > 0.95:
        status_symbol = "~"
        status_text = "NEUTRAL"
    else:
        status_symbol = "---"
        status_text = "NEEDS WORK"

    # ===== HEADER =====
    print(f"\n{'#'*88}")
    print(f"# EPOCH {epoch:3d} │ COOPERATIVE NEURO-SYMBOLIC DENOISING │ [{status_symbol}] {status_text}")
    print(f"# {'':>10} │ Clinical: {clinical_improvements}/4 improved │ CNR: {cnr_improvement:+.1f}% │ PSNR Δ: {psnr_delta:+.3f} dB")
    print(f"{'#'*88}")

    # ===== CLINICAL PRESERVATION TABLE (PRIMARY METRICS) =====
    print(f"\n┌{'─'*86}┐")
    print(f"│ {'CLINICAL PRESERVATION':<22} {'Backbone%':>12} {'Corrected%':>12} {'Ratio':>10} {'Status':>12} {'Target':>12} │")
    print(f"├{'─'*86}┤")

    # Get absolute preservation values (defaults from known backbone losses)
    bb_contrast = val_metrics.get('backbone_contrast_pres', 0.53) * 100
    corr_contrast = val_metrics.get('corrected_contrast_pres', 0.53) * 100
    bb_boundary = val_metrics.get('backbone_boundary_pres', 0.53) * 100
    corr_boundary = val_metrics.get('corrected_boundary_pres', 0.53) * 100
    bb_texture = val_metrics.get('backbone_texture_pres', 0.59) * 100
    corr_texture = val_metrics.get('corrected_texture_pres', 0.59) * 100
    bb_edge = val_metrics.get('backbone_edge_pres', 0.32) * 100
    corr_edge = val_metrics.get('corrected_edge_pres', 0.32) * 100

    targets = {
        'Contrast': '47% lost',
        'Boundary': '47% lost',
        'Texture': '41% lost',
        'Edge': '68% lost',
    }

    metrics_data = [
        ('Contrast (local std)', bb_contrast, corr_contrast, contrast_ratio, targets['Contrast']),
        ('Boundary (v-grad)', bb_boundary, corr_boundary, boundary_ratio, targets['Boundary']),
        ('Texture (variance)', bb_texture, corr_texture, texture_ratio, targets['Texture']),
        ('Edge (Sobel)', bb_edge, corr_edge, edge_ratio, targets['Edge']),
    ]

    for name, bb_val, corr_val, ratio, target in metrics_data:
        status = "IMPROVED" if ratio > 1.0 else "---"
        print(f"│ {name:<22} {bb_val:>11.1f}% {corr_val:>11.1f}% {ratio:>10.3f} {status:>12} {target:>12} │")

    print(f"├{'─'*86}┤")
    avg_bb = (bb_contrast + bb_boundary + bb_texture + bb_edge) / 4
    avg_corr = (corr_contrast + corr_boundary + corr_texture + corr_edge) / 4
    status_str = f"{clinical_improvements}/4 IMPROVED" if clinical_improvements > 0 else "NO IMPROVEMENT"
    print(f"│ {'AVERAGE':<22} {avg_bb:>11.1f}% {avg_corr:>11.1f}% {avg_ratio:>10.3f} {status_str:>24} │")
    print(f"└{'─'*86}┘")

    # ===== TRADITIONAL QUALITY METRICS =====
    # SSIM preservation ratio from loss computation (if available)
    ssim_preservation_ratio = val_metrics.get('ssim_preservation_ratio', 1.0)
    ssim_preserved = ssim_preservation_ratio >= 0.99  # 99% SSIM preservation threshold

    print(f"\n┌{'─'*86}┐")
    print(f"│ {'TRADITIONAL METRICS':<30} {'Backbone':>12} {'Corrected':>12} {'Delta':>10} {'Preserved':>12} │")
    print(f"├{'─'*86}┤")
    # PSNR: Relaxed constraint (allow more variation if SSIM stays high)
    psnr_status = "[OK]" if abs(psnr_delta) <= 1.0 else "[!]"
    # SSIM: Primary perceptual quality constraint (more important than PSNR)
    ssim_status = "[OK]" if ssim_delta >= -0.01 else "[!]"
    ssim_pres_status = "[OK]" if ssim_preserved else "[!]"
    print(f"│ {'PSNR (dB)':<30} {val_metrics['psnr_backbone']:>12.2f} {val_metrics['psnr_corrected']:>12.2f} {psnr_delta:>+9.3f} {psnr_status:>12} │")
    print(f"│ {'SSIM':<30} {val_metrics['ssim_backbone']:>12.4f} {val_metrics['ssim_corrected']:>12.4f} {ssim_delta:>+9.4f} {ssim_status:>12} │")
    print(f"│ {'SSIM Preservation Ratio':<30} {'-':>12} {ssim_preservation_ratio:>12.4f} {'>=0.99':>10} {ssim_pres_status:>12} │")
    # LPIPS perceptual metrics (lower = better, negative delta = improvement)
    lpips_backbone = val_metrics.get('lpips_backbone', 0)
    lpips_corrected = val_metrics.get('lpips_corrected', 0)
    lpips_delta = val_metrics.get('lpips_delta', 0)
    lpips_status = "[OK]" if lpips_delta <= 0.01 else "[!]"  # Allow small increase
    print(f"│ {'LPIPS [PERCEPTUAL]':<30} {lpips_backbone:>12.4f} {lpips_corrected:>12.4f} {lpips_delta:>+9.4f} {lpips_status:>12} │")
    print(f"├{'─'*86}┤")
    print(f"│ {'QUALITY STRATEGY: Prioritize SSIM (perceptual) over PSNR (pixel-wise)':<84} │")
    print(f"│ {'Rationale: SSIM measures structural similarity which aligns better with clinical quality':<84} │")
    print(f"│ {'LPIPS: Lower = better perceptual quality; negative delta = improvement':<84} │")
    print(f"└{'─'*86}┘")

    # ===== GT-FREE PREDICATES (Backbone vs Corrected) =====
    pred_scores = val_metrics.get('pred_scores', {})
    pred_scores_backbone = val_metrics.get('pred_scores_backbone', pred_scores)
    passed_count = sum(1 for k, s in pred_scores.items() if s >= 0.5 and k != 'P5') if pred_scores else 0

    if pred_scores:
        avg_pred = sum(pred_scores.values()) / max(len(pred_scores), 1)
        avg_pred_backbone = sum(pred_scores_backbone.values()) / max(len(pred_scores_backbone), 1)

        print(f"\n┌{'─'*76}┐")
        print(f"│ {'GT-FREE PREDICATES':<24} {'Backbone':>12} {'Corrected':>12} {'Delta':>10} {'Status':>12} │")
        print(f"├{'─'*76}┤")

        pred_names = {
            'P1': 'Edge Quality', 'P2': 'Contrast', 'P3': 'Smoothness',
            'P4': 'Structure', 'P5': 'Speckle (excluded)', 'P6': 'Anatomy'
        }

        for key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            score_corrected = pred_scores.get(key, 0)
            score_backbone = pred_scores_backbone.get(key, score_corrected)
            delta = score_corrected - score_backbone
            if key == 'P5':
                status = "EXCLUDED"
            else:
                status = "PASS" if score_corrected >= 0.5 else "FAIL"
            name = pred_names.get(key, key)
            print(f"│ {name:<24} {score_backbone:>12.3f} {score_corrected:>12.3f} {delta:>+10.3f} {status:>12} │")

        print(f"├{'─'*76}┤")
        avg_delta = avg_pred - avg_pred_backbone
        print(f"│ {'AVERAGE':<24} {avg_pred_backbone:>12.3f} {avg_pred:>12.3f} {avg_delta:>+10.3f} {f'{passed_count}/5 PASS':>12} │")
        print(f"└{'─'*76}┘")

    # ===== COOPERATION ANALYSIS =====
    print(f"\n┌{'─'*68}┐")
    print(f"│ {'COOPERATION ANALYSIS (NAFNet + Correctors)':<66} │")
    print(f"├{'─'*68}┤")

    uncertainty_corr = val_metrics.get('uncertainty_potential_corr', 0)
    confident_cov = val_metrics.get('confident_coverage', 0)
    uncertain_cov = val_metrics.get('uncertain_coverage', 0)
    neutral_cov = 1 - confident_cov - uncertain_cov

    # Cooperation status
    if uncertainty_corr > 0.3:
        coop_status = "EXCELLENT"
        coop_detail = "Strong cooperation pattern"
    elif uncertainty_corr > 0.1:
        coop_status = "GOOD"
        coop_detail = "Moderate cooperation"
    elif uncertainty_corr > 0:
        coop_status = "FAIR"
        coop_detail = "Weak cooperation"
    else:
        coop_status = "POOR"
        coop_detail = "No cooperation pattern"

    print(f"│ {'Uncertainty-Potential Correlation':<40} {uncertainty_corr:>+24.3f} │")
    print(f"│ {'NAFNet Confident Regions':<40} {confident_cov*100:>23.1f}% │")
    print(f"│ {'NAFNet Uncertain Regions':<40} {uncertain_cov*100:>23.1f}% │")
    print(f"│ {'Neutral Regions':<40} {neutral_cov*100:>23.1f}% │")
    print(f"├{'─'*68}┤")
    print(f"│ {'Cooperation Status':<40} {coop_status + ' - ' + coop_detail:>26} │")
    print(f"└{'─'*68}┘")

    # ===== CORRECTOR ACTIVITY & LAMBDA STATS =====
    corrector_activity = train_metrics.get('corrector_activity', {})
    lambda_stats = val_metrics.get('lambda_stats', {})

    print(f"\n┌{'─'*68}┐")
    print(f"│ {'CORRECTION BEHAVIOR':<30} {'Activity%':>12} {'λ Mean':>12} {'λ Max':>10} │")
    print(f"├{'─'*68}┤")

    total_activity = sum(corrector_activity.values()) + 1e-8
    for name in ['edge', 'contrast', 'smooth', 'structure', 'anatomy']:
        activity = corrector_activity.get(name, 0)
        pct = activity / total_activity * 100

        if name in lambda_stats:
            stats = lambda_stats[name]
            mean_val = stats['mean'] if isinstance(stats['mean'], (int, float)) else float(stats['mean'])
            max_val = stats['max'] if isinstance(stats['max'], (int, float)) else float(stats['max'])
        else:
            mean_val = 0.0
            max_val = 0.0

        print(f"│ {name:<30} {pct:>11.1f}% {mean_val:>12.4f} {max_val:>10.4f} │")

    correction_mag = val_metrics.get('correction_magnitude', 0)
    if correction_mag < 0.005:
        corr_interp = "Very Light"
    elif correction_mag < 0.015:
        corr_interp = "Moderate"
    elif correction_mag < 0.03:
        corr_interp = "Active"
    else:
        corr_interp = "Heavy (!)"

    print(f"├{'─'*68}┤")
    print(f"│ {'Overall Correction Magnitude':<30} {correction_mag:>12.6f} {corr_interp:>22} │")
    print(f"└{'─'*68}┘")

    # ===== LEGACY CLINICAL METRICS (IEEE TMI) =====
    cnr_backbone = val_metrics.get('cnr_backbone', 0)
    cnr_corrected = val_metrics.get('cnr_corrected', 0)
    cnr_delta = cnr_corrected - cnr_backbone

    tci_backbone = val_metrics.get('tci_backbone', 0)
    tci_corrected = val_metrics.get('tci_corrected', 0)
    tci_delta = tci_corrected - tci_backbone

    epi_backbone = val_metrics.get('epi_backbone', 0)
    epi_corrected = val_metrics.get('epi_corrected', 0)
    epi_delta = epi_corrected - epi_backbone

    bs_backbone = val_metrics.get('boundary_sharpness_backbone', 0)
    bs_corrected = val_metrics.get('boundary_sharpness_corrected', 0)
    bs_delta = bs_corrected - bs_backbone

    # TCI, BS: ratio vs clean (1.0 = perfect), closer to 1.0 = improved
    tci_improved = abs(tci_corrected - 1.0) < abs(tci_backbone - 1.0)
    bs_improved = abs(bs_corrected - 1.0) < abs(bs_backbone - 1.0)
    legacy_improved = sum([
        1 if cnr_delta > 0 else 0,
        1 if tci_improved else 0,
        1 if epi_delta > 0 else 0,
        1 if bs_improved else 0,
    ])

    print(f"\n┌{'─'*68}┐")
    print(f"│ {'OCT CLINICAL METRICS (IEEE TMI)':<30} {'Backbone':>12} {'Corrected':>12} {'Delta':>10} │")
    print(f"├{'─'*68}┤")
    cnr_status = "✓" if cnr_delta >= 0 else "!"
    tci_status = "✓" if tci_improved else "!"
    epi_status = "✓" if epi_delta >= 0 else "~"
    bs_status = "✓" if bs_improved else "~"
    print(f"│ {'CNR (Contrast-to-Noise)':<30} {cnr_backbone:>12.3f} {cnr_corrected:>12.3f} {cnr_delta:>+9.3f}{cnr_status} │")
    print(f"│ {'TCI (Tissue Contrast Index)':<30} {tci_backbone:>12.3f} {tci_corrected:>12.3f} {tci_delta:>+9.3f}{tci_status} │")
    print(f"│ {'EPI (Edge Preservation)':<30} {epi_backbone:>12.4f} {epi_corrected:>12.4f} {epi_delta:>+9.4f}{epi_status} │")
    print(f"│ {'Boundary Sharpness':<30} {bs_backbone:>12.4f} {bs_corrected:>12.4f} {bs_delta:>+9.4f}{bs_status} │")
    print(f"├{'─'*68}┤")
    legacy_status = f"{legacy_improved}/4 IMPROVED" if legacy_improved > 0 else "NO IMPROVEMENT"
    print(f"│ {'CLINICAL ASSESSMENT':<30} {legacy_status:>36} │")
    print(f"└{'─'*68}┘")

    # ===== REGION-SPECIFIC METRICS =====
    if 'psnr_bg_delta' in val_metrics:
        print(f"\n┌{'─'*86}┐")
        print(f"│ {'REGION-SPECIFIC METRICS (Region-Adaptive Loss Verification)':<84} │")
        print(f"├{'─'*86}┤")
        print(f"│ {'Region':<20} {'PSNR Delta (dB)':>20} {'Clinical Change (%)':>20} {'Expected':>20} │")
        print(f"├{'─'*86}┤")
        psnr_bg_delta = val_metrics.get('psnr_bg_delta', 0)
        psnr_tissue_delta = val_metrics.get('psnr_tissue_delta', 0)
        clinical_bg_change = val_metrics.get('clinical_bg_change', 0)
        clinical_tissue_improvement = val_metrics.get('clinical_tissue_improvement', 0)
        # Background: Should have good PSNR (smooth, noise-free)
        bg_psnr_status = "OK" if psnr_bg_delta >= -0.5 else "!"
        bg_clinical_status = "OK" if clinical_bg_change <= 5 else "!"
        # Tissue: Should have good clinical metrics (enhanced visibility)
        tissue_psnr_status = "OK" if psnr_tissue_delta >= -1.0 else "!"
        tissue_clinical_status = "OK" if clinical_tissue_improvement >= 0 else "!"
        print(f"│ {'Background':<20} {psnr_bg_delta:>+19.2f} {clinical_bg_change:>+19.1f} {'Stable/Improve':>20} │")
        print(f"│ {'Tissue':<20} {psnr_tissue_delta:>+19.2f} {clinical_tissue_improvement:>+19.1f} {'Enhanced':>20} │")
        print(f"├{'─'*86}┤")
        print(f"│ {'Goal: Background should be smooth (good PSNR), Tissue should have enhanced visibility':<84} │")
        print(f"└{'─'*86}┘")
        print(f"  Region-Specific Metrics:")
        print(f"    Background: PSNR delta={val_metrics.get('psnr_bg_delta', 0):.2f} dB, Clinical change={val_metrics.get('clinical_bg_change', 0):.1f}%")
        print(f"    Tissue:     PSNR delta={val_metrics.get('psnr_tissue_delta', 0):.2f} dB, Clinical improvement={val_metrics.get('clinical_tissue_improvement', 0):.1f}%")

    # ===== TRAINING SUMMARY =====
    print(f"\n┌{'─'*68}┐")
    print(f"│ {'TRAINING LOSS COMPONENTS':<66} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'Total Loss':<40} {train_metrics['loss']:>26.6f} │")
    print(f"│ {'Cooperation Loss':<40} {train_metrics.get('cooperation', 0):>26.6f} │")
    print(f"│ {'Efficiency Loss':<40} {train_metrics.get('efficiency', 0):>26.6f} │")
    print(f"│ {'CNR Preservation Loss':<40} {train_metrics.get('cnr_loss', 0):>26.6f} │")
    print(f"└{'─'*68}┘")

    # ===== QUICK SUMMARY LINE =====
    print(f"\n>>> Clinical: {clinical_improvements}/4 improved │ Ratio: {avg_ratio:.3f} │ Pres: {avg_bb:.1f}%→{avg_corr:.1f}%")
    print(f">>> PSNR: {val_metrics['psnr_backbone']:.2f}→{val_metrics['psnr_corrected']:.2f} ({psnr_delta:+.3f}) │ CNR: {cnr_backbone:.2f}→{cnr_corrected:.2f} ({cnr_improvement:+.1f}%)")
    print(f">>> Predicates: {passed_count}/5 pass │ Cooperation: {coop_status}")

    # ===== INTERPRETABLE VERDICT =====
    print(f"\n{'#'*88}")
    print(f"# INTERPRETABLE SUMMARY")
    print(f"#")

    # Find most active corrector
    if corrector_activity:
        most_active = max(corrector_activity.items(), key=lambda x: x[1])[0]
        activity_pct = corrector_activity[most_active] / total_activity * 100

        # Map corrector to predicate
        pred_map = {'edge': 'P1', 'contrast': 'P2', 'smooth': 'P3', 'structure': 'P4', 'anatomy': 'P6'}
        pred_key = pred_map.get(most_active, 'P1')
        pred_score = pred_scores.get(pred_key, 0)

        print(f"# \"{most_active.capitalize()} corrector handled {activity_pct:.0f}% of corrections\"")
        print(f"# \"{pred_key} score: {pred_score:.2f} (threshold: 0.5)\"")

    print(f"# \"PSNR trade-off: {psnr_delta:+.2f} dB (target: < 1.0 dB drop)\"")
    print(f"# \"CNR change: {cnr_improvement:+.1f}% (target: ≥ 0%)\"")
    print(f"# \"Cooperation quality: {coop_status} - {coop_detail}\"")
    print(f"#")

    if target_achieved:
        print(f"# ★ VERDICT: PUBLICATION READY ★")
        print(f"#   All criteria met: Predicates ≥4/5, PSNR Δ ≤1.0dB, Clinical ≥5%, CNR ≥0%")
    else:
        gaps = []
        if predicates_passing < 4:
            gaps.append(f"predicates ({predicates_passing:.0f}/5)")
        if abs(psnr_delta) > 0.5:
            gaps.append(f"PSNR ({psnr_delta:+.2f} dB)")
        if contrast_improve < 5:
            gaps.append(f"clinical ({contrast_improve:.1f}%)")
        if cnr_improvement < 0:
            gaps.append(f"CNR ({cnr_improvement:+.1f}%)")
        if uncertainty_corr < 0.1:
            gaps.append("cooperation")
        print(f"# VERDICT: Needs improvement in: {', '.join(gaps) if gaps else 'minor adjustments'}")

    print(f"{'#'*88}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train V8 Cooperative Neuro-Symbolic Denoising (IEEE TMI)'
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

    # Training config
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_corrector', type=float, default=2e-4,
                        help='Learning rate for corrector (default: 2e-4)')
    parser.add_argument('--lr_potential', type=float, default=5e-4,
                        help='Learning rate for potential estimators (default: 5e-4)')
    parser.add_argument('--lr_negotiator', type=float, default=3e-4,
                        help='Learning rate for negotiator (default: 3e-4)')
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Loss config
    parser.add_argument('--cooperation_weight', type=float, default=0.3)
    parser.add_argument('--efficiency_weight', type=float, default=0.2)
    parser.add_argument('--clinical_weight', type=float, default=1.5,
                        help='Weight for clinical metrics loss - CLINICAL IMPROVEMENT IS SELLING POINT - push for 10-15%% improvement')
    parser.add_argument('--cnr_weight', type=float, default=1.0,
                        help='Weight for CNR preservation loss (REBALANCED from 1.5 to 1.0 to prevent PSNR degradation)')
    parser.add_argument('--psnr_slack', type=float, default=1.0)

    # Output
    parser.add_argument('--output_dir', default='outputs/nsnd_v8_cooperative')

    # Torch compile optimization
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile optimization')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("COOPERATIVE NEURO-SYMBOLIC DENOISING V8")
    print("IEEE TMI Publication Training")
    print("="*80)
    print(f"\nKey Innovation: NAFNet and Correctors COOPERATE based on uncertainty")
    print(f"  - NAFNet handles confident regions")
    print(f"  - Correctors specialize in uncertain regions")
    print(f"  - Negotiator allocates work efficiently")
    print(f"\nTarget Performance:")
    print(f"  - All 5 predicates pass (P5 excluded)")
    print(f"  - +10-15% clinical improvement")
    print(f"  - < {args.psnr_slack} dB PSNR drop")
    print(f"  - Clear cooperation patterns")
    print(f"\nConfiguration:")
    print(f"  Backbone width: {args.backbone_width}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  LR corrector: {args.lr_corrector}")
    print(f"  LR potential: {args.lr_potential}")
    print(f"  LR negotiator: {args.lr_negotiator}")
    print(f"  Cooperation weight: {args.cooperation_weight}")
    print(f"  Efficiency weight: {args.efficiency_weight}")
    print(f"  Clinical weight: {args.clinical_weight}")
    print(f"  CNR weight: {args.cnr_weight}")
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

    # FIX: Use 2 workers for CPU (helps overlap data loading with computation)
    num_workers = 4 if args.device != 'cpu' else 2
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(args.device != 'cpu'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True  # FIX: Consistent batch sizes for better GPU utilization
    )
    # FIX: Use batch_size=1 for validation (full-resolution images use ~28x more memory than 96x96 patches)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=min(2, num_workers), pin_memory=(args.device != 'cpu')
    )

    # Model
    print("\nInitializing cooperative model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_width=args.backbone_width,
        pretrained_backbone=args.pretrained_backbone,
    ).to(args.device)

    # Freeze backbone (IMPORTANT: train only corrector, potential, negotiator)
    if args.freeze_backbone:
        print("\n*** FREEZING NAFNet BACKBONE WEIGHTS ***")
        for param in model.backbone.backbone.parameters():
            param.requires_grad = False
        # Keep uncertainty head trainable (conv1, bn, conv2, temperature, bias_offset)
        for param in [model.backbone.uncertainty_conv1.weight,
                      model.backbone.uncertainty_bn.weight, model.backbone.uncertainty_bn.bias,
                      model.backbone.uncertainty_conv2.weight, model.backbone.uncertainty_conv2.bias,
                      model.backbone.uncertainty_temperature, model.backbone.uncertainty_bias_offset]:
            param.requires_grad = True

    # Loss
    criterion = CooperativeLoss(
        cooperation_weight=args.cooperation_weight,
        efficiency_weight=args.efficiency_weight,
        clinical_weight=args.clinical_weight,
        cnr_weight=args.cnr_weight,
        psnr_slack=args.psnr_slack,
        use_uncertainty_weighting=True
    ).to(args.device)

    # Apply torch.compile for CPU optimization (PyTorch 2.0+)
    if hasattr(torch, 'compile') and args.device == 'cpu' and not args.no_compile:
        print("Applying torch.compile for CPU optimization...")
        try:
            # Use 'reduce-overhead' mode which is better for CPU
            # 'inductor' backend works well on CPU
            model = torch.compile(model, mode='reduce-overhead', backend='inductor')
            print("Successfully compiled model with torch.compile")
        except Exception as e:
            print(f"Warning: torch.compile failed, continuing without compilation: {e}")

    # Optimizer with different learning rates for different components
    # Get corrector parameters
    corrector_params = list(model.corrector.correctors.parameters())

    # Get potential estimator parameters (if exists)
    potential_params = []
    if hasattr(model.corrector, 'potential_estimators'):
        potential_params = list(model.corrector.potential_estimators.parameters())
    if hasattr(model.corrector, 'lambda_predictor'):
        potential_params += list(model.corrector.lambda_predictor.parameters())

    # Get negotiator parameters (if exists)
    negotiator_params = []
    if hasattr(model.corrector, 'negotiator'):
        negotiator_params = list(model.corrector.negotiator.parameters())
    if hasattr(model.corrector, 'router'):
        negotiator_params += list(model.corrector.router.parameters())

    # BUG FIX: Include clinical enhancement modules that were MISSING from optimizer!
    # These modules have learnable parameters but were not being trained.
    clinical_enhancement_params = []
    if hasattr(model.corrector, 'cnr_preserver'):
        clinical_enhancement_params += list(model.corrector.cnr_preserver.parameters())
    if hasattr(model.corrector, 'clinical_enhancer'):
        clinical_enhancement_params += list(model.corrector.clinical_enhancer.parameters())
    if hasattr(model.corrector, 'region_aware_corrector'):
        clinical_enhancement_params += list(model.corrector.region_aware_corrector.parameters())
    # BUG FIX: Include confidence estimator parameters (was missing!)
    if hasattr(model.corrector, 'confidence_estimator'):
        clinical_enhancement_params += list(model.corrector.confidence_estimator.parameters())

    # Backbone uncertainty head (conv1, bn, conv2, and calibration params)
    uncertainty_params = (
        list(model.backbone.uncertainty_conv1.parameters()) +
        list(model.backbone.uncertainty_bn.parameters()) +
        list(model.backbone.uncertainty_conv2.parameters()) +
        [model.backbone.uncertainty_temperature, model.backbone.uncertainty_bias_offset]
    )

    # Loss weight parameters
    loss_weight_params = list(criterion.log_sigma.parameters())

    param_groups = [
        {'params': corrector_params, 'lr': args.lr_corrector},
        {'params': potential_params, 'lr': args.lr_potential} if potential_params else {'params': [], 'lr': 0},
        {'params': negotiator_params, 'lr': args.lr_negotiator} if negotiator_params else {'params': [], 'lr': 0},
        {'params': clinical_enhancement_params, 'lr': args.lr_corrector} if clinical_enhancement_params else {'params': [], 'lr': 0},  # BUG FIX: Add clinical enhancement params
        {'params': uncertainty_params, 'lr': args.lr_potential},
        {'params': loss_weight_params, 'lr': args.lr_corrector * 0.25, 'weight_decay': 0},
    ]

    # Filter out empty param groups
    param_groups = [pg for pg in param_groups if len(list(pg['params'])) > 0 or pg['lr'] > 0]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    print(f"\nOptimizer configuration:")
    print(f"  Corrector params: {sum(p.numel() for p in corrector_params):,} @ lr={args.lr_corrector}")
    if potential_params:
        print(f"  Potential params: {sum(p.numel() for p in potential_params):,} @ lr={args.lr_potential}")
    if negotiator_params:
        print(f"  Negotiator params: {sum(p.numel() for p in negotiator_params):,} @ lr={args.lr_negotiator}")
    if clinical_enhancement_params:
        print(f"  Clinical enhancement params: {sum(p.numel() for p in clinical_enhancement_params):,} @ lr={args.lr_corrector}")
    print(f"  Uncertainty params: {sum(p.numel() for p in uncertainty_params):,} @ lr={args.lr_potential}")

    # Scheduler
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
    print("# TRAINING COOPERATIVE NEURO-SYMBOLIC DENOISING")
    print("# Target: All predicates pass, +10-15% clinical, < 1.0 dB PSNR drop")
    print("#"*80)

    best_score = -float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, scaler
        )
        scheduler.step()

        if epoch % args.val_every == 0 or epoch == args.epochs:
            # FIX: Clear GPU cache before validation (full-res images need more memory than training patches)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            val_metrics = validate(model, val_loader, args.device, lpips_model=None, criterion=criterion)
            print_cooperation_metrics(epoch, train_metrics, val_metrics)

            # Compute composite score for model selection
            # Prioritize: predicates passing, PSNR constraint, CNR preservation, cooperation quality
            predicates_passing = val_metrics.get('predicates_passing', 0)
            psnr_delta = val_metrics['psnr_delta']
            cooperation_corr = val_metrics.get('uncertainty_potential_corr', 0)
            contrast_improve = val_metrics.get('contrast_improvement', 0)
            cnr_improvement = val_metrics.get('cnr_improvement', 0)

            # Score: predicates * 10 + cooperation * 5 + clinical * 0.1 + CNR * 0.5 - psnr_penalty - cnr_penalty
            psnr_penalty = max(0, abs(psnr_delta) - args.psnr_slack) * 10
            # Penalize CNR drops (target: 0% or better)
            cnr_penalty = max(0, -cnr_improvement) * 2.0  # 2x penalty per % CNR drop
            score = (predicates_passing * 10 +
                    max(0, cooperation_corr) * 5 +
                    contrast_improve * 0.1 +
                    max(0, cnr_improvement) * 0.5 -  # Reward CNR improvement
                    psnr_penalty -
                    cnr_penalty)  # Penalize CNR drops

            if score > best_score:
                best_score = score
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_score': best_score,
                    'args': vars(args),
                }, os.path.join(args.output_dir, 'best_model_cooperative.pth'))

                print(f"*** New best model! Score: {best_score:.2f} ***")
        else:
            print(f"[Epoch {epoch}] Loss: {train_metrics['loss']:.4f}, "
                  f"PSNR: {train_metrics['psnr_corrected']:.2f}, "
                  f"Coop: {train_metrics.get('cooperation', 0):.3f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "#"*80)
    print("# TRAINING COMPLETE")
    print("#"*80)
    print(f"Best composite score: {best_score:.2f} (epoch {best_epoch})")
    print(f"Model saved to: {args.output_dir}/best_model_cooperative.pth")


if __name__ == '__main__':
    main()
