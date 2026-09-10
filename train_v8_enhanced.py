#!/usr/bin/env python3
"""
Train V8 Enhanced Neuro-Symbolic OCT Denoising

V8 Enhanced includes:
1. All V8 novel features (fuzzy logic, hierarchical reasoning, physics speckle)
2. AdaptiveLambdaPredictor (per-pixel correction strength)
3. Feature fusion from backbone (enc1, enc2)
4. Attention mechanisms (channel + spatial)
5. Multi-scale dilated convolutions

Training Data: PKU37 real noise pairs
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

        # Use NAFNetSmall which matches the pretrained checkpoint
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

            # Try direct loading first
            compatible = {k: v for k, v in state_dict.items()
                         if k in model_state and v.shape == model_state[k].shape}

            # If no direct match, the checkpoint might not have 'backbone.' prefix
            if len(compatible) < len(model_state) // 2:
                # Try without prefix modification (weights saved directly from NAFNet)
                compatible = {}
                for k, v in state_dict.items():
                    if k in model_state and v.shape == model_state[k].shape:
                        compatible[k] = v

            if len(compatible) == 0:
                print("No compatible weights found")
                return False

            self.backbone.load_state_dict(compatible, strict=False)
            print(f"Loaded {len(compatible)}/{len(model_state)} backbone weights")

            # Print checkpoint metrics if available
            if 'psnr' in ckpt:
                print(f"  Checkpoint PSNR: {ckpt['psnr']:.2f} dB, SSIM: {ckpt.get('ssim', 'N/A')}")
            return True
        except Exception as e:
            print(f"Failed to load backbone: {e}")
            import traceback
            traceback.print_exc()
            return False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with feature extraction.

        Returns:
            denoised: [B, 1, H, W]
            features: {'enc1': [B, width, H, W], 'enc2': [B, width*2, H/2, W/2]}
        """
        bb = self.backbone
        B, C, H, W = x.shape

        # Pad to multiple of 16 for NAFNetSmall (4 encoder stages)
        padder_size = 2 ** len(bb.encoders)
        mod_h = H % padder_size
        mod_w = W % padder_size
        if mod_h != 0 or mod_w != 0:
            pad_h = padder_size - mod_h if mod_h != 0 else 0
            pad_w = padder_size - mod_w if mod_w != 0 else 0
            inp = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            inp = x

        # Feature extraction
        feat = bb.intro(inp)

        # Encoder pass - capture features
        # MEMORY FIX: Only store first 2 encoder outputs (needed for feature fusion)
        encs = []
        encs_for_features = []  # Only keep enc1 and enc2 for feature extraction
        for i, (encoder, down) in enumerate(zip(bb.encoders, bb.downs)):
            feat = encoder(feat)
            encs.append(feat)
            if i < 2:  # Only keep enc1 and enc2 for feature extraction
                encs_for_features.append(feat)
            feat = down(feat)

        # Middle blocks
        feat = bb.middle_blks(feat)

        # Decoder pass
        for decoder, up, enc_skip in zip(bb.decoders, bb.ups, encs[::-1]):
            feat = up(feat)
            feat = feat + enc_skip
            feat = decoder(feat)

        # Output
        out = bb.ending(feat) + inp
        out = out[:, :, :H, :W]

        # Return denoised and intermediate features
        # Note: encs_for_features[0] is at full resolution, encs_for_features[1] is at half resolution
        # MEMORY FIX: Detach features to break gradient graph when backbone is frozen
        features = {
            'enc1': encs_for_features[0][:, :, :H, :W].detach(),
            'enc2': encs_for_features[1][:, :, :min(H//2, encs_for_features[1].shape[2]), :min(W//2, encs_for_features[1].shape[3])].detach(),
        }
        del encs, encs_for_features  # Free encoder lists

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
# Model
# =============================================================================

class NeuroSymbolicDenoiserV8Enhanced(nn.Module):
    """
    V8 Enhanced Neuro-Symbolic Denoiser.

    Components:
    1. NAFNet backbone with feature extraction
    2. NeuroSymbolicCorrectorV8Enhanced with:
       - AdaptiveLambdaPredictor
       - Feature fusion
       - Attention mechanisms
       - All V8 novel features
    """

    def __init__(self, backbone_width: int = 48, pretrained_backbone: str = None):
        super().__init__()

        # Backbone with feature extraction
        self.backbone = BackboneWithFeatures(width=backbone_width)

        # V8 Enhanced Corrector
        self.corrector = NeuroSymbolicCorrectorV8Enhanced(
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

        print(f"\nNeuroSymbolicDenoiserV8Enhanced Parameters:")
        print(f"  Backbone (width={self.backbone.width}): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"  V8 Enhanced Corrector: {corrector_params:,} ({corrector_params/1e6:.2f}M)")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, noisy: torch.Tensor, return_details: bool = False):
        """
        Forward pass with feature fusion.

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

        return corrected, backbone_out, info


# =============================================================================
# Loss
# =============================================================================

class V8EnhancedLoss(nn.Module):
    """
    Loss for V8 Enhanced training with UNCERTAINTY WEIGHTING (Kendall et al., 2018).

    Automatically learns optimal weights for each loss term based on task uncertainty.
    Instead of manual tuning: λ_i * L_i
    We learn: (1/2σ_i²) * L_i + log(σ_i)

    This balances PSNR vs clinical metrics automatically:
    - High uncertainty (hard to optimize) → lower weight
    - Low uncertainty (easy to optimize) → higher weight
    - log(σ) term prevents weights from collapsing to zero

    Reference:
    Kendall, A., Gal, Y., & Cipolla, R. (2018).
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    CVPR 2018. https://arxiv.org/abs/1705.07115

    Clinical Focus (based on backbone analysis):
    - 53% contrast loss -> contrast_loss
    - 53% boundary sharpness loss -> boundary_loss
    - 59% texture loss -> texture_loss
    - 32% edge loss -> edge_loss
    """

    def __init__(self,
                 lambda_reg: float = 0.01,       # Lambda regularization (fixed, not learned)
                 use_uncertainty_weighting: bool = True):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # =====================================================================
        # LEARNABLE UNCERTAINTY PARAMETERS (log σ²)
        # We learn log(σ²) for numerical stability
        # Weight for loss_i = 1/(2*exp(log_sigma_i)) = 0.5 * exp(-log_sigma_i)
        # Regularization = 0.5 * log_sigma_i
        #
        # Initialize based on expected loss magnitudes:
        # - Recon (MSE ~0.01): log_sigma = 0 → weight = 0.5
        # - Clinical losses (~0.1): log_sigma = 1 → weight ≈ 0.18
        # =====================================================================
        self.log_sigma = nn.ParameterDict({
            'recon': nn.Parameter(torch.tensor(0.0)),      # PSNR/MSE loss
            'backbone': nn.Parameter(torch.tensor(1.0)),   # Backbone supervision
            'pred': nn.Parameter(torch.tensor(2.0)),       # Predicate consistency
            'contrast': nn.Parameter(torch.tensor(1.0)),   # Clinical: contrast
            'boundary': nn.Parameter(torch.tensor(1.0)),   # Clinical: boundary
            'texture': nn.Parameter(torch.tensor(1.0)),    # Clinical: texture
            'edge': nn.Parameter(torch.tensor(1.0)),       # Clinical: edge
            # NEW: GT-Aligned Predicate Learning losses
            'pred_align': nn.Parameter(torch.tensor(0.5)), # Predicate alignment to GT
            'psnr_preserve': nn.Parameter(torch.tensor(-1.0)), # PSNR preservation (high weight)
        })

        # Fixed weights as fallback (if uncertainty weighting disabled)
        self.fixed_weights = {
            'recon': 1.0,
            'backbone': 0.1,
            'pred': 0.05,
            'contrast': 0.15,
            'boundary': 0.15,
            'texture': 0.10,
            'edge': 0.15,
            # NEW: GT-Aligned Predicate Learning
            'pred_align': 0.5,      # Predicate alignment to GT
            'psnr_preserve': 2.0,   # PSNR preservation constraint (high weight)
        }

        # PSNR preservation threshold (allow this much dB drop before penalty)
        # 0.5 dB is barely perceptible, allows meaningful clinical corrections
        self.psnr_slack = 0.5  # dB

        # Register Sobel filters as buffers (created once, reused)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_learned_weights(self) -> dict:
        """
        Get current learned weights for monitoring.
        Weight = 1 / (2 * σ²) = 0.5 * exp(-log_sigma)
        """
        weights = {}
        for name, log_sig in self.log_sigma.items():
            sigma_sq = torch.exp(log_sig)
            weight = 0.5 / sigma_sq
            weights[name] = weight.item()
        return weights

    def get_uncertainties(self) -> dict:
        """Get learned uncertainties (σ) for each task."""
        uncertainties = {}
        for name, log_sig in self.log_sigma.items():
            sigma = torch.exp(0.5 * log_sig)  # σ = exp(0.5 * log(σ²))
            uncertainties[name] = sigma.item()
        return uncertainties

    def _weighted_loss(self, loss: torch.Tensor, name: str) -> torch.Tensor:
        """
        Apply uncertainty weighting to a loss term.
        L_weighted = (1/2σ²) * L + (1/2) * log(σ²)
                   = 0.5 * exp(-log_sigma) * L + 0.5 * log_sigma
        """
        if self.use_uncertainty_weighting:
            log_sig = self.log_sigma[name]
            # Precision weighting: 0.5 * exp(-log_sigma) * loss
            precision_weight = 0.5 * torch.exp(-log_sig)
            # Regularization: 0.5 * log_sigma (prevents sigma → ∞)
            regularization = 0.5 * log_sig
            return precision_weight * loss + regularization
        else:
            return self.fixed_weights[name] * loss

    def local_std(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Compute local standard deviation in sliding windows.
        Differentiable implementation using convolutions.
        """
        # Create averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                           device=x.device, dtype=x.dtype) / (kernel_size ** 2)
        padding = kernel_size // 2

        # Local mean
        local_mean = F.conv2d(x, kernel, padding=padding)

        # Local mean of squares
        local_mean_sq = F.conv2d(x ** 2, kernel, padding=padding)

        # Local variance = E[X^2] - E[X]^2
        local_var = local_mean_sq - local_mean ** 2

        # Clamp to avoid negative values due to numerical precision
        local_var = torch.clamp(local_var, min=1e-6)

        # Local std
        return torch.sqrt(local_var)

    def local_variance(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Compute local variance in sliding windows.
        """
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                           device=x.device, dtype=x.dtype) / (kernel_size ** 2)
        padding = kernel_size // 2
        local_mean = F.conv2d(x, kernel, padding=padding)
        local_mean_sq = F.conv2d(x ** 2, kernel, padding=padding)
        local_var = local_mean_sq - local_mean ** 2
        return torch.clamp(local_var, min=1e-6)

    # =========================================================================
    # GT-ALIGNED PREDICATE COMPUTATION
    # Compute predicates on any single image (not comparison-based)
    # These predicates capture absolute quality characteristics
    # =========================================================================

    def compute_image_predicates(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute predicate-like quality scores for a single image.

        These scores represent absolute quality characteristics that can be
        compared between images (backbone, corrected, GT) to guide corrections.

        Returns dict with:
            - edge_quality: Edge coherence/strength [0, 1]
            - contrast: Local contrast measure [0, 1]
            - smoothness: Smoothness in flat regions [0, 1]
            - texture: Texture richness [0, 1]
            - structure: Structural coherence [0, 1]
        """
        eps = 1e-6

        # Edge quality: Mean normalized edge magnitude
        sobel_x = self.sobel_x.to(dtype=img.dtype)
        sobel_y = self.sobel_y.to(dtype=img.dtype)
        edge_x = F.conv2d(img, sobel_x, padding=1)
        edge_y = F.conv2d(img, sobel_y, padding=1)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + eps)
        edge_quality = edge_mag.mean(dim=[2, 3]).clamp(0, 1)  # [B, 1]

        # Contrast: Local standard deviation (normalized)
        local_std = self.local_std(img, kernel_size=7)
        contrast = local_std.mean(dim=[2, 3]).clamp(0, 1)  # [B, 1]

        # Smoothness: Inverse of high-frequency content in flat regions
        # High value = smooth image
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                                 device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        hf = F.conv2d(img, laplacian, padding=1).abs()
        # Smoothness is higher when high-frequency content is lower
        smoothness = 1.0 - hf.mean(dim=[2, 3]).clamp(0, 1)  # [B, 1]

        # Texture: Local variance (indicates texture richness)
        local_var = self.local_variance(img, kernel_size=5)
        texture = local_var.mean(dim=[2, 3]).clamp(0, 1)  # [B, 1]

        # Structure: Gradient coherence (structure tensor eigenvalue ratio)
        # High coherence = clear directional structure
        Ixx = edge_x * edge_x
        Iyy = edge_y * edge_y
        Ixy = edge_x * edge_y
        # Average over local window
        kernel = torch.ones(1, 1, 5, 5, device=img.device, dtype=img.dtype) / 25
        Sxx = F.conv2d(Ixx, kernel, padding=2)
        Syy = F.conv2d(Iyy, kernel, padding=2)
        Sxy = F.conv2d(Ixy, kernel, padding=2)
        # Eigenvalues of structure tensor
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy * Sxy
        # Coherence = (λ1 - λ2)² / (λ1 + λ2)² = 1 - 4*det/trace²
        coherence = 1.0 - 4.0 * det / (trace**2 + eps)
        structure = coherence.mean(dim=[2, 3]).clamp(0, 1)  # [B, 1]

        return {
            'edge_quality': edge_quality.squeeze(1),  # [B]
            'contrast': contrast.squeeze(1),
            'smoothness': smoothness.squeeze(1),
            'texture': texture.squeeze(1),
            'structure': structure.squeeze(1),
        }

    def predicate_alignment_loss(self, corrected: torch.Tensor,
                                  clean: torch.Tensor) -> torch.Tensor:
        """
        GT-Aligned Predicate Loss: Match corrected predicates to GT predicates.

        This aligns predicate optimization with clinical fidelity because
        GT predicates represent the "clinically correct" quality profile.

        If GT has low contrast (pathology), we learn to preserve low contrast.
        If GT has high contrast (healthy), we learn to restore high contrast.
        """
        # Compute predicates on corrected and GT
        pred_corrected = self.compute_image_predicates(corrected)
        pred_gt = self.compute_image_predicates(clean)

        # MSE loss for each predicate
        total_loss = torch.tensor(0.0, device=corrected.device)
        for key in pred_corrected:
            loss = F.mse_loss(pred_corrected[key], pred_gt[key])
            total_loss = total_loss + loss

        return total_loss / len(pred_corrected)

    def deficiency_aware_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                              clean: torch.Tensor, lambda_stats: dict) -> torch.Tensor:
        """
        Deficiency-Aware Loss: Correction magnitude should match predicate deficiency.

        If backbone predicate is close to GT → don't correct (λ should be low)
        If backbone predicate is far from GT → correct more (λ should be higher)

        This prevents over-correction where it's not needed.
        """
        # Compute predicates
        pred_backbone = self.compute_image_predicates(backbone)
        pred_gt = self.compute_image_predicates(clean)

        # Compute deficiency for each predicate (how much backbone deviates from GT)
        deficiencies = {}
        for key in pred_backbone:
            # Positive deficiency = backbone is worse than GT
            deficiency = (pred_gt[key] - pred_backbone[key]).clamp(min=0)
            deficiencies[key] = deficiency

        # Average deficiency across predicates
        avg_deficiency = torch.stack(list(deficiencies.values())).mean()

        # Get average lambda activation
        if lambda_stats:
            lambda_means = []
            for name, stats in lambda_stats.items():
                if isinstance(stats['mean'], torch.Tensor):
                    lambda_means.append(stats['mean'])
                else:
                    lambda_means.append(torch.tensor(stats['mean'], device=corrected.device))
            avg_lambda = torch.stack(lambda_means).mean() if lambda_means else torch.tensor(0.0, device=corrected.device)
        else:
            avg_lambda = torch.tensor(0.0, device=corrected.device)

        # Loss: lambda should be proportional to deficiency
        # If deficiency is low but lambda is high → over-correction penalty
        # If deficiency is high but lambda is low → under-correction penalty
        efficiency_loss = F.mse_loss(avg_lambda, avg_deficiency)

        return efficiency_loss

    def psnr_preservation_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                                clean: torch.Tensor) -> torch.Tensor:
        """
        PSNR Preservation Constraint: Penalize when corrected PSNR drops below backbone.

        Allows a small slack (self.psnr_slack dB) before applying penalty.
        This prevents corrections that hurt overall image quality.
        """
        eps = 1e-8

        # Compute MSE for both
        mse_backbone = F.mse_loss(backbone, clean)
        mse_corrected = F.mse_loss(corrected, clean)

        # PSNR = 10 * log10(1 / MSE)
        psnr_backbone = 10 * torch.log10(1.0 / (mse_backbone + eps))
        psnr_corrected = 10 * torch.log10(1.0 / (mse_corrected + eps))

        # Penalty when PSNR drops more than slack
        # psnr_drop = psnr_backbone - psnr_corrected (positive when corrected is worse)
        psnr_drop = psnr_backbone - psnr_corrected

        # Only penalize drops beyond slack threshold
        # ReLU(drop - slack) = 0 if drop < slack, else (drop - slack)
        penalty = F.relu(psnr_drop - self.psnr_slack)

        # Square the penalty to make it stronger for large drops
        return penalty ** 2

    def contrast_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Contrast Loss: Penalize when local contrast (std) is reduced vs ground truth.

        Analysis showed backbone loses 53% of contrast.
        Uses threshold-based mask (mean + std) to focus on high-contrast regions.
        """
        # Compute local standard deviation (contrast measure)
        pred_std = self.local_std(pred, kernel_size=5)
        target_std = self.local_std(target, kernel_size=5)

        # Create mask for regions with significant contrast in ground truth
        # Using per-image mean + std threshold instead of global for correct batch handling
        target_mean = target_std.mean(dim=[2, 3], keepdim=True)
        target_std_val = target_std.std(dim=[2, 3], keepdim=True)
        contrast_threshold = target_mean + target_std_val
        contrast_mask = (target_std > contrast_threshold).float()

        # Penalize where prediction has LESS contrast than ground truth
        # Only in regions where GT has significant contrast
        contrast_deficit = F.relu(target_std - pred_std)  # Positive when pred < target

        # Masked loss focusing on high-contrast regions
        masked_loss = (contrast_deficit * contrast_mask).sum() / (contrast_mask.sum() + 1e-6)

        return masked_loss

    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Boundary Loss: Penalize when vertical gradients (layer boundaries) are weaker vs ground truth.

        Analysis showed backbone loses 53% of boundary sharpness.
        OCT layer boundaries are horizontal, so we focus on vertical gradients.
        Uses threshold-based mask to focus on actual boundary regions.
        """
        # Vertical gradient (layer boundaries in OCT are horizontal)
        pred_grad = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # Create mask for regions with strong boundaries in ground truth
        # Using per-image mean + std threshold instead of global for correct batch handling
        target_grad_mean = target_grad.mean(dim=[2, 3], keepdim=True)
        target_grad_std = target_grad.std(dim=[2, 3], keepdim=True)
        boundary_threshold = target_grad_mean + target_grad_std
        boundary_mask = (target_grad > boundary_threshold).float()

        # Penalize where prediction has WEAKER boundaries than ground truth
        boundary_deficit = F.relu(target_grad - pred_grad)  # Positive when pred < target

        # Masked loss focusing on actual boundary regions
        masked_loss = (boundary_deficit * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)

        return masked_loss

    def texture_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Texture Loss: Penalize when local variance is reduced vs ground truth (over-smoothing).

        Analysis showed backbone loses 59% of texture.
        Uses threshold-based mask to focus on textured regions.
        """
        # Compute local variance (texture measure)
        pred_var = self.local_variance(pred, kernel_size=5)
        target_var = self.local_variance(target, kernel_size=5)

        # Create mask for regions with texture in ground truth
        # Using per-image mean + std threshold instead of global for correct batch handling
        target_var_mean = target_var.mean(dim=[2, 3], keepdim=True)
        target_var_std = target_var.std(dim=[2, 3], keepdim=True)
        texture_threshold = target_var_mean + 0.5 * target_var_std  # Lower threshold to catch more texture
        texture_mask = (target_var > texture_threshold).float()

        # Penalize where prediction has LESS texture/variance than ground truth
        # This specifically targets over-smoothing
        texture_deficit = F.relu(target_var - pred_var)  # Positive when pred < target

        # Masked loss focusing on textured regions
        masked_loss = (texture_deficit * texture_mask).sum() / (texture_mask.sum() + 1e-6)

        return masked_loss

    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Edge Loss: Penalize when edge magnitude is weaker vs ground truth.

        Analysis showed backbone loses 32% of edges.
        Uses Sobel edge detection with threshold-based mask.
        """
        # Use registered Sobel buffers (created once in __init__, reused)
        sobel_x = self.sobel_x.to(dtype=pred.dtype)
        sobel_y = self.sobel_y.to(dtype=pred.dtype)

        # Compute edge magnitudes for prediction
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

        # Compute edge magnitudes for target
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

        # Create mask for regions with edges in ground truth
        # Using per-image mean + std threshold instead of global for correct batch handling
        target_edge_mean = target_edges.mean(dim=[2, 3], keepdim=True)
        target_edge_std = target_edges.std(dim=[2, 3], keepdim=True)
        edge_threshold = target_edge_mean + target_edge_std
        edge_mask = (target_edges > edge_threshold).float()

        # Penalize where prediction has WEAKER edges than ground truth
        edge_deficit = F.relu(target_edges - pred_edges)  # Positive when pred < target

        # Masked loss focusing on edge regions
        masked_loss = (edge_deficit * edge_mask).sum() / (edge_mask.sum() + 1e-6)

        return masked_loss

    def forward(self, corrected, backbone_out, clean, info):
        """
        Compute loss with UNCERTAINTY WEIGHTING (Kendall et al., 2018).

        Automatically balances PSNR vs clinical metrics by learning task uncertainties.
        Loss_i_weighted = (1/2σ²_i) * Loss_i + (1/2) * log(σ²_i)

        Benefits:
        - No manual tuning of loss weights
        - Hard tasks (high uncertainty) get lower weight automatically
        - Easy tasks (low uncertainty) get higher weight automatically
        - log(σ²) regularization prevents collapse
        """
        # =================================================================
        # Compute individual losses
        # =================================================================

        # Reconstruction loss (PSNR)
        recon_loss = F.mse_loss(corrected, clean)

        # Backbone supervision loss
        backbone_loss = F.mse_loss(backbone_out, clean)

        # Correction quality loss (predicate-based)
        correction_mag = info.get('correction_magnitude', 0.0)
        if isinstance(correction_mag, (int, float)):
            correction_mag = torch.tensor(correction_mag, device=recon_loss.device)
        elif isinstance(correction_mag, torch.Tensor):
            correction_mag = correction_mag.to(recon_loss.device)

        improvement = backbone_loss - recon_loss
        correction_penalty = correction_mag * 0.05
        pred_loss_val = -improvement + correction_penalty

        # Lambda regularization - ensure gradient flow
        lambda_stats = info.get('lambda_stats', {})
        if lambda_stats:
            means = [s['mean'] for s in lambda_stats.values()]
            if means and isinstance(means[0], torch.Tensor):
                avg_lambda = torch.stack(means).mean()
            elif means:
                avg_lambda = torch.tensor(sum(means) / len(means), device=recon_loss.device, requires_grad=False)
            else:
                avg_lambda = torch.tensor(0.0, device=recon_loss.device)
            lambda_reg = avg_lambda * self.lambda_reg
        else:
            lambda_reg = torch.tensor(0.0, device=recon_loss.device)

        # =================================================================
        # Clinical Quality Losses
        # =================================================================
        # Contrast loss: penalize reduced local contrast (53% lost by backbone)
        contrast_loss_val = self.contrast_loss(corrected, clean)

        # Boundary loss: penalize weaker layer boundaries (53% lost by backbone)
        boundary_loss_val = self.boundary_loss(corrected, clean)

        # Texture loss: penalize over-smoothing (59% lost by backbone)
        texture_loss_val = self.texture_loss(corrected, clean)

        # Edge loss: penalize weaker edges (32% lost by backbone)
        edge_loss_val = self.edge_loss(corrected, clean)

        # =================================================================
        # GT-ALIGNED PREDICATE LEARNING (NEW)
        # These losses align predicate optimization with clinical fidelity
        # =================================================================

        # 1. Predicate Alignment Loss: Match corrected predicates to GT predicates
        #    This ensures corrections push toward GT quality profile, not generic "ideal"
        pred_align_loss = self.predicate_alignment_loss(corrected, clean)

        # 2. Deficiency-Aware Loss: Correction magnitude matches predicate deficiency
        #    Prevents over-correction where backbone is already good
        deficiency_loss = self.deficiency_aware_loss(corrected, backbone_out, clean, lambda_stats)

        # 3. PSNR Preservation Loss: Penalize significant PSNR drops
        #    Allows small slack (0.3 dB) before applying penalty
        psnr_preserve_loss = self.psnr_preservation_loss(corrected, backbone_out, clean)

        # =================================================================
        # UNCERTAINTY-WEIGHTED TOTAL LOSS
        # Each loss is weighted by learned precision (1/σ²) with regularization
        # =================================================================
        total_loss = (
            self._weighted_loss(recon_loss, 'recon') +
            self._weighted_loss(backbone_loss, 'backbone') +
            self._weighted_loss(pred_loss_val, 'pred') +
            lambda_reg +  # Fixed regularization
            self._weighted_loss(contrast_loss_val, 'contrast') +
            self._weighted_loss(boundary_loss_val, 'boundary') +
            self._weighted_loss(texture_loss_val, 'texture') +
            self._weighted_loss(edge_loss_val, 'edge') +
            # NEW: GT-Aligned Predicate Learning losses
            self._weighted_loss(pred_align_loss, 'pred_align') +
            self._weighted_loss(psnr_preserve_loss, 'psnr_preserve') +
            deficiency_loss * 0.1  # Fixed weight for deficiency loss
        )

        # Metrics
        with torch.no_grad():
            psnr_backbone = 10 * torch.log10(1 / (backbone_loss + 1e-6))
            psnr_corrected = 10 * torch.log10(1 / (recon_loss + 1e-6))

        # Predicate scores for logging
        pred_scores = info.get('predicate_scores', {})
        avg_score = sum(pred_scores.values()) / len(pred_scores) if pred_scores else 0.5

        # Get learned weights for monitoring
        learned_weights = self.get_learned_weights()
        learned_uncertainties = self.get_uncertainties()

        # Compute predicate scores for logging (GT-aligned)
        with torch.no_grad():
            pred_gt = self.compute_image_predicates(clean)
            pred_corr = self.compute_image_predicates(corrected)
            pred_bb = self.compute_image_predicates(backbone_out)
            pred_alignment_scores = {
                k: (pred_corr[k] - pred_gt[k]).abs().mean().item()
                for k in pred_corr
            }
            pred_improvement = {
                k: ((pred_corr[k] - pred_gt[k]).abs() - (pred_bb[k] - pred_gt[k]).abs()).mean().item()
                for k in pred_corr
            }

        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'backbone': backbone_loss.item(),
            'pred': pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val,
            'lambda_reg': lambda_reg if isinstance(lambda_reg, float) else lambda_reg.item(),
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'psnr_delta': psnr_corrected.item() - psnr_backbone.item(),
            'avg_pred_score': avg_score,
            # Clinical quality loss values
            'contrast_loss': contrast_loss_val.item(),
            'boundary_loss': boundary_loss_val.item(),
            'texture_loss': texture_loss_val.item(),
            'edge_loss': edge_loss_val.item(),
            # NEW: GT-Aligned Predicate Learning losses
            'pred_align_loss': pred_align_loss.item(),
            'deficiency_loss': deficiency_loss.item(),
            'psnr_preserve_loss': psnr_preserve_loss.item(),
            # Predicate alignment scores (lower is better)
            'pred_alignment': pred_alignment_scores,
            'pred_improvement': pred_improvement,  # Negative = better alignment than backbone
            # LEARNED weights (uncertainty weighting)
            'learned_weights': learned_weights,
            'learned_uncertainties': learned_uncertainties,
        }


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
    # Clamp variance to prevent negative values due to numerical precision
    sigma_pred = torch.clamp(((pred - mu_pred) ** 2).mean(), min=1e-8)
    sigma_target = torch.clamp(((target - mu_target) ** 2).mean(), min=1e-8)
    sigma_both = ((pred - mu_pred) * (target - mu_target)).mean()
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
           ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))
    return ssim.item()


# =============================================================================
# GT-Aligned Predicate Computation (standalone for validation)
# =============================================================================

def compute_gt_aligned_predicates(img: torch.Tensor) -> Dict[str, float]:
    """
    Compute GT-aligned predicate-like quality scores for a single image.

    Standalone function for use in validation loop.
    Returns dict with quality scores that can be compared between images.
    """
    eps = 1e-6

    # Sobel filters
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

    # Edge quality: Mean normalized edge magnitude
    edge_x = F.conv2d(img, sobel_x, padding=1)
    edge_y = F.conv2d(img, sobel_y, padding=1)
    edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + eps)
    edge_quality = edge_mag.mean().clamp(0, 1).item()

    # Local std for contrast
    kernel_size = 7
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                       device=img.device, dtype=img.dtype) / (kernel_size ** 2)
    padding = kernel_size // 2
    local_mean = F.conv2d(img, kernel, padding=padding)
    local_mean_sq = F.conv2d(img ** 2, kernel, padding=padding)
    local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=eps)
    local_std = torch.sqrt(local_var)

    contrast = local_std.mean().clamp(0, 1).item()

    # Smoothness: Inverse of high-frequency content
    laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                             device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    hf = F.conv2d(img, laplacian, padding=1).abs()
    smoothness = (1.0 - hf.mean().clamp(0, 1)).item()

    # Texture: Local variance
    texture = local_var.mean().clamp(0, 1).item()

    # Structure: Gradient coherence
    Ixx = edge_x * edge_x
    Iyy = edge_y * edge_y
    Ixy = edge_x * edge_y
    kernel5 = torch.ones(1, 1, 5, 5, device=img.device, dtype=img.dtype) / 25
    Sxx = F.conv2d(Ixx, kernel5, padding=2)
    Syy = F.conv2d(Iyy, kernel5, padding=2)
    Sxy = F.conv2d(Ixy, kernel5, padding=2)
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    coherence = 1.0 - 4.0 * det / (trace**2 + eps)
    structure = coherence.mean().clamp(0, 1).item()

    return {
        'edge_quality': edge_quality,
        'contrast': contrast,
        'smoothness': smoothness,
        'texture': texture,
        'structure': structure,
    }


# =============================================================================
# Clinical Quality Metrics (Simple, CPU-efficient)
# =============================================================================

def compute_cnr_simple(img: torch.Tensor) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR) using intensity quantiles.

    CNR = (high_intensity_mean - low_intensity_mean) / noise_std

    Higher CNR indicates better contrast between bright and dark regions,
    which is important for layer visibility in OCT images.

    Args:
        img: [B, C, H, W] or [C, H, W] image tensor

    Returns:
        CNR value (higher is better)
    """
    img_flat = img.flatten()
    high_thresh = img_flat.quantile(0.9)
    low_thresh = img_flat.quantile(0.1)

    high_region = img_flat[img_flat > high_thresh]
    low_region = img_flat[img_flat < low_thresh]

    if len(high_region) == 0 or len(low_region) == 0:
        return 0.0

    high_mean = high_region.mean()
    low_mean = low_region.mean()
    noise_std = img_flat.std()

    cnr = (high_mean - low_mean) / (noise_std + 1e-6)
    return cnr.item()


def compute_epi_simple(denoised: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute Edge Preservation Index (EPI) as correlation of gradients.

    EPI measures how well edges in the denoised image correlate with
    edges in the clean reference image. Higher EPI = better edge preservation.

    Args:
        denoised: Denoised image [B, C, H, W]
        reference: Clean reference [B, C, H, W]

    Returns:
        EPI value (higher is better, max ~1.0)
    """
    # Create Sobel filters once at top of function
    device = denoised.device
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device, dtype=denoised.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    def gradient_magnitude(img):
        # Ensure 4D
        if img.dim() == 3:
            img = img.unsqueeze(0)
        gx = F.conv2d(F.pad(img, [1, 1, 1, 1], mode='reflect'), sobel_x)
        gy = F.conv2d(F.pad(img, [1, 1, 1, 1], mode='reflect'), sobel_y)
        return torch.sqrt(gx**2 + gy**2)

    grad_denoised = gradient_magnitude(denoised).flatten()
    grad_reference = gradient_magnitude(reference).flatten()

    # Pearson correlation
    d_centered = grad_denoised - grad_denoised.mean()
    r_centered = grad_reference - grad_reference.mean()

    corr = (d_centered * r_centered).sum() / (
        torch.sqrt((d_centered**2).sum() * (r_centered**2).sum()) + 1e-8
    )

    # Explicit cleanup
    del grad_denoised, grad_reference, d_centered, r_centered

    return corr.item()


def compute_boundary_sharpness_simple(img: torch.Tensor) -> float:
    """
    Compute boundary sharpness as average vertical gradient magnitude.

    OCT images have horizontal layer boundaries, so vertical gradient
    strength indicates how sharp/visible these boundaries are.

    Args:
        img: [B, C, H, W] image tensor

    Returns:
        Boundary sharpness value (higher is better)
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)

    # Vertical Sobel for horizontal edges (layer boundaries)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

    grad_y = F.conv2d(F.pad(img, [1, 1, 1, 1], mode='reflect'), sobel_y)

    # Focus on typical layer boundary regions (15%, 40%, 55%, 75% of height)
    H = img.shape[2]
    boundary_rows = [int(0.15 * H), int(0.40 * H), int(0.55 * H), int(0.75 * H)]

    boundary_strength = 0.0
    count = 0
    for row in boundary_rows:
        if 2 < row < H - 2:
            # Average gradient in a small band around the boundary
            boundary_strength += grad_y[:, :, row-2:row+3, :].abs().mean().item()
            count += 1

    return boundary_strength / max(count, 1)


def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()

    total_loss = 0
    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_pred_score = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        # Forward with optional AMP
        if scaler is not None:
            with autocast():
                corrected, backbone_out, info = model(noisy)
                loss, metrics = criterion(corrected, backbone_out, clean, info)

            # Backward with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Clip gradients for both model and criterion (learnable loss weights)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            corrected, backbone_out, info = model(noisy)
            loss, metrics = criterion(corrected, backbone_out, clean, info)

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients for both model and criterion (learnable loss weights)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            optimizer.step()

        # MEMORY FIX: Delete intermediates
        del corrected, backbone_out, info

        total_loss += metrics['total']
        total_psnr_backbone += metrics['psnr_backbone']
        total_psnr_corrected += metrics['psnr_corrected']
        total_pred_score += metrics['avg_pred_score']
        n += 1

        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'psnr': f"{metrics['psnr_corrected']:.1f}",
            'delta': f"{metrics['psnr_corrected'] - metrics['psnr_backbone']:+.2f}",
        })

        # Store last batch's learned weights for monitoring
        last_learned_weights = metrics.get('learned_weights', {})
        last_learned_uncertainties = metrics.get('learned_uncertainties', {})

    return {
        'loss': total_loss / n,
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'pred_score': total_pred_score / n,
        # Learned weights from last batch
        'learned_weights': last_learned_weights,
        'learned_uncertainties': last_learned_uncertainties,
    }


def compute_local_std(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Compute local standard deviation for clinical metrics."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                       device=img.device, dtype=img.dtype) / (kernel_size ** 2)
    padding = kernel_size // 2
    local_mean = F.conv2d(img, kernel, padding=padding)
    local_mean_sq = F.conv2d(img ** 2, kernel, padding=padding)
    local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=1e-6)
    return torch.sqrt(local_var)


def compute_clinical_preservation(corrected: torch.Tensor, backbone: torch.Tensor,
                                   clean: torch.Tensor, device: str) -> dict:
    """
    Compute clinical preservation ratios comparing corrected vs backbone to ground truth.

    Returns preservation ratios (higher = better, >1.0 means improvement over backbone):
    - contrast_ratio: Local std preservation
    - boundary_ratio: Vertical gradient preservation
    - texture_ratio: Local variance preservation
    - edge_ratio: Edge magnitude preservation
    """
    # Ensure 4D tensors
    if corrected.dim() == 3:
        corrected = corrected.unsqueeze(0)
    if backbone.dim() == 3:
        backbone = backbone.unsqueeze(0)
    if clean.dim() == 3:
        clean = clean.unsqueeze(0)

    # Create Sobel filters once at top of function
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device, dtype=corrected.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    # === Contrast Preservation (local std) ===
    clean_std = compute_local_std(clean)
    backbone_std = compute_local_std(backbone)
    corrected_std = compute_local_std(corrected)

    # Preservation ratio: how much of GT contrast is preserved
    clean_std_mean = clean_std.mean().item() + 1e-6
    backbone_contrast_pres = backbone_std.mean().item() / clean_std_mean
    corrected_contrast_pres = corrected_std.mean().item() / clean_std_mean
    # Clamp ratio to prevent extreme values from corrupting metrics
    contrast_ratio = max(0.1, min(10.0, corrected_contrast_pres / (backbone_contrast_pres + 1e-6)))

    # === Boundary Preservation (vertical gradient) ===
    clean_vgrad = torch.abs(clean[:, :, 1:, :] - clean[:, :, :-1, :])
    backbone_vgrad = torch.abs(backbone[:, :, 1:, :] - backbone[:, :, :-1, :])
    corrected_vgrad = torch.abs(corrected[:, :, 1:, :] - corrected[:, :, :-1, :])

    clean_vgrad_mean = clean_vgrad.mean().item() + 1e-6
    backbone_boundary_pres = backbone_vgrad.mean().item() / clean_vgrad_mean
    corrected_boundary_pres = corrected_vgrad.mean().item() / clean_vgrad_mean
    # Clamp ratio to prevent extreme values from corrupting metrics
    boundary_ratio = max(0.1, min(10.0, corrected_boundary_pres / (backbone_boundary_pres + 1e-6)))

    # === Texture Preservation (local variance) ===
    clean_var = compute_local_std(clean) ** 2
    backbone_var = compute_local_std(backbone) ** 2
    corrected_var = compute_local_std(corrected) ** 2

    clean_var_mean = clean_var.mean().item() + 1e-6
    backbone_texture_pres = backbone_var.mean().item() / clean_var_mean
    corrected_texture_pres = corrected_var.mean().item() / clean_var_mean
    # Clamp ratio to prevent extreme values from corrupting metrics
    texture_ratio = max(0.1, min(10.0, corrected_texture_pres / (backbone_texture_pres + 1e-6)))

    # === Edge Preservation (Sobel magnitude) ===
    def edge_magnitude(img):
        ex = F.conv2d(img, sobel_x, padding=1)
        ey = F.conv2d(img, sobel_y, padding=1)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

    clean_edges = edge_magnitude(clean)
    backbone_edges = edge_magnitude(backbone)
    corrected_edges = edge_magnitude(corrected)

    clean_edge_mean = clean_edges.mean().item() + 1e-6
    backbone_edge_pres = backbone_edges.mean().item() / clean_edge_mean
    corrected_edge_pres = corrected_edges.mean().item() / clean_edge_mean
    # Clamp ratio to prevent extreme values from corrupting metrics
    edge_ratio = max(0.1, min(10.0, corrected_edge_pres / (backbone_edge_pres + 1e-6)))

    # Explicit cleanup to prevent memory leaks
    del clean_std, backbone_std, corrected_std
    del clean_vgrad, backbone_vgrad, corrected_vgrad
    del clean_var, backbone_var, corrected_var
    del clean_edges, backbone_edges, corrected_edges

    return {
        'contrast_ratio': contrast_ratio,
        'boundary_ratio': boundary_ratio,
        'texture_ratio': texture_ratio,
        'edge_ratio': edge_ratio,
        # Also return absolute preservation values
        'backbone_contrast_pres': backbone_contrast_pres,
        'corrected_contrast_pres': corrected_contrast_pres,
        'backbone_boundary_pres': backbone_boundary_pres,
        'corrected_boundary_pres': corrected_boundary_pres,
        'backbone_texture_pres': backbone_texture_pres,
        'corrected_texture_pres': corrected_texture_pres,
        'backbone_edge_pres': backbone_edge_pres,
        'corrected_edge_pres': corrected_edge_pres,
    }


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    use_amp = device != 'cpu' and torch.cuda.is_available()

    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_ssim_backbone = 0
    total_ssim_corrected = 0
    total_pred_scores = {f'P{i}': 0 for i in range(1, 7)}  # Corrected output predicates
    total_pred_scores_backbone = {f'P{i}': 0 for i in range(1, 7)}  # Backbone predicates
    total_lambda_stats = {}
    total_correction_mag = 0

    # Clinical quality metrics accumulators
    total_cnr_backbone = 0
    total_cnr_corrected = 0
    total_epi_backbone = 0
    total_epi_corrected = 0
    total_boundary_sharpness_backbone = 0
    total_boundary_sharpness_corrected = 0

    # Clinical preservation ratios accumulators
    total_contrast_ratio = 0
    total_boundary_ratio = 0
    total_texture_ratio = 0
    total_edge_ratio = 0
    # Absolute preservation values
    total_backbone_contrast_pres = 0
    total_corrected_contrast_pres = 0
    total_backbone_boundary_pres = 0
    total_corrected_boundary_pres = 0
    total_backbone_texture_pres = 0
    total_corrected_texture_pres = 0
    total_backbone_edge_pres = 0
    total_corrected_edge_pres = 0

    # GT-Aligned Predicate accumulators (NEW)
    total_pred_alignment = {'edge_quality': 0, 'contrast': 0, 'smoothness': 0, 'texture': 0, 'structure': 0}
    total_pred_improvement = {'edge_quality': 0, 'contrast': 0, 'smoothness': 0, 'texture': 0, 'structure': 0}

    n = 0
    total_batches = len(loader)
    last_detailed_info = None  # Store interpretability info from last batch

    for batch_idx, batch in enumerate(tqdm(loader, desc="Validation")):
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)

        # Get detailed info on last batch for interpretability printout
        is_last_batch = (batch_idx == total_batches - 1)

        # SPEED FIX: Use AMP for validation too
        if use_amp:
            with autocast():
                corrected, backbone_out, info = model(noisy, return_details=is_last_batch)
        else:
            corrected, backbone_out, info = model(noisy, return_details=is_last_batch)

        # Save detailed info from last batch
        if is_last_batch:
            last_detailed_info = {
                'clinical_report': info.get('clinical_report', ''),
                'verification': info.get('verification', {}),
                'explanations': info.get('explanations', {}),
                'layer_analysis': info.get('layer_analysis', {}),
                'counterfactuals': info.get('counterfactuals', {}),
            }

        psnr_backbone = compute_psnr(backbone_out, clean)
        psnr_corrected = compute_psnr(corrected, clean)
        ssim_backbone = compute_ssim(backbone_out, clean)
        ssim_corrected = compute_ssim(corrected, clean)

        total_psnr_backbone += psnr_backbone
        total_psnr_corrected += psnr_corrected
        total_ssim_backbone += ssim_backbone
        total_ssim_corrected += ssim_corrected

        # Compute clinical quality metrics
        # CNR - Contrast-to-Noise Ratio (higher = better layer visibility)
        cnr_backbone = compute_cnr_simple(backbone_out)
        cnr_corrected = compute_cnr_simple(corrected)
        total_cnr_backbone += cnr_backbone
        total_cnr_corrected += cnr_corrected

        # EPI - Edge Preservation Index (correlation with clean edges)
        epi_backbone = compute_epi_simple(backbone_out, clean)
        epi_corrected = compute_epi_simple(corrected, clean)
        total_epi_backbone += epi_backbone
        total_epi_corrected += epi_corrected

        # Boundary Sharpness (gradient magnitude at layer boundaries)
        bs_backbone = compute_boundary_sharpness_simple(backbone_out)
        bs_corrected = compute_boundary_sharpness_simple(corrected)
        total_boundary_sharpness_backbone += bs_backbone
        total_boundary_sharpness_corrected += bs_corrected

        # Compute clinical preservation ratios
        clinical_pres = compute_clinical_preservation(corrected, backbone_out, clean, device)
        total_contrast_ratio += clinical_pres['contrast_ratio']
        total_boundary_ratio += clinical_pres['boundary_ratio']
        total_texture_ratio += clinical_pres['texture_ratio']
        total_edge_ratio += clinical_pres['edge_ratio']
        total_backbone_contrast_pres += clinical_pres['backbone_contrast_pres']
        total_corrected_contrast_pres += clinical_pres['corrected_contrast_pres']
        total_backbone_boundary_pres += clinical_pres['backbone_boundary_pres']
        total_corrected_boundary_pres += clinical_pres['corrected_boundary_pres']
        total_backbone_texture_pres += clinical_pres['backbone_texture_pres']
        total_corrected_texture_pres += clinical_pres['corrected_texture_pres']
        total_backbone_edge_pres += clinical_pres['backbone_edge_pres']
        total_corrected_edge_pres += clinical_pres['corrected_edge_pres']

        # Accumulate CORRECTED output predicate scores
        for name, score in info['predicate_scores'].items():
            key = name.split('_')[0]
            if key in total_pred_scores:
                score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                total_pred_scores[key] += score_val

        # Accumulate BACKBONE predicate scores (before correction)
        for name, score in info.get('predicate_scores_backbone', {}).items():
            key = name.split('_')[0]
            if key in total_pred_scores_backbone:
                score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                total_pred_scores_backbone[key] += score_val

        # Lambda statistics
        for name, stats in info.get('lambda_stats', {}).items():
            if name not in total_lambda_stats:
                total_lambda_stats[name] = {'mean': 0, 'max': 0}
            mean_val = stats['mean'].item() if isinstance(stats['mean'], torch.Tensor) else float(stats['mean'])
            max_val = stats['max'].item() if isinstance(stats['max'], torch.Tensor) else float(stats['max'])
            total_lambda_stats[name]['mean'] += mean_val
            total_lambda_stats[name]['max'] += max_val

        # Correction magnitude type handling
        corr_mag = info['correction_magnitude']
        total_correction_mag += corr_mag.item() if isinstance(corr_mag, torch.Tensor) else float(corr_mag)

        # GT-Aligned Predicate computation (NEW)
        with torch.no_grad():
            pred_gt = compute_gt_aligned_predicates(clean)
            pred_corr = compute_gt_aligned_predicates(corrected)
            pred_bb = compute_gt_aligned_predicates(backbone_out)

            for key in total_pred_alignment:
                # Alignment error: distance from GT (lower is better)
                align_err = abs(pred_corr[key] - pred_gt[key])
                total_pred_alignment[key] += align_err

                # Improvement: negative = corrected is better aligned than backbone
                bb_err = abs(pred_bb[key] - pred_gt[key])
                improvement = align_err - bb_err  # Negative means corrected is better
                total_pred_improvement[key] += improvement

        # MEMORY FIX: Clean up
        del corrected, backbone_out, info
        n += 1

    return {
        'psnr_backbone': total_psnr_backbone / n,
        'psnr_corrected': total_psnr_corrected / n,
        'ssim_backbone': total_ssim_backbone / n,
        'ssim_corrected': total_ssim_corrected / n,
        'pred_scores': {k: v / n for k, v in total_pred_scores.items()},  # On corrected output
        'pred_scores_backbone': {k: v / n for k, v in total_pred_scores_backbone.items()},  # On backbone
        'lambda_stats': {k: {'mean': v['mean']/n, 'max': v['max']/n}
                        for k, v in total_lambda_stats.items()},
        'correction_magnitude': total_correction_mag / n,
        'psnr_delta': (total_psnr_corrected - total_psnr_backbone) / n,
        # Clinical quality metrics
        'cnr_backbone': total_cnr_backbone / n,
        'cnr_corrected': total_cnr_corrected / n,
        'epi_backbone': total_epi_backbone / n,
        'epi_corrected': total_epi_corrected / n,
        'boundary_sharpness_backbone': total_boundary_sharpness_backbone / n,
        'boundary_sharpness_corrected': total_boundary_sharpness_corrected / n,
        # Clinical preservation ratios (corrected vs backbone, >1.0 = improvement)
        'contrast_ratio': total_contrast_ratio / n,
        'boundary_ratio': total_boundary_ratio / n,
        'texture_ratio': total_texture_ratio / n,
        'edge_ratio': total_edge_ratio / n,
        # Absolute preservation values (vs ground truth)
        'backbone_contrast_pres': total_backbone_contrast_pres / n,
        'corrected_contrast_pres': total_corrected_contrast_pres / n,
        'backbone_boundary_pres': total_backbone_boundary_pres / n,
        'corrected_boundary_pres': total_corrected_boundary_pres / n,
        'backbone_texture_pres': total_backbone_texture_pres / n,
        'corrected_texture_pres': total_corrected_texture_pres / n,
        'backbone_edge_pres': total_backbone_edge_pres / n,
        'corrected_edge_pres': total_corrected_edge_pres / n,
        # GT-Aligned Predicate metrics (NEW)
        'pred_alignment': {k: v / n for k, v in total_pred_alignment.items()},
        'pred_improvement': {k: v / n for k, v in total_pred_improvement.items()},
        # Interpretability info from last sample
        'interpretability': last_detailed_info,
    }


def print_epoch_metrics(epoch, train_metrics, val_metrics, prev_val_metrics=None):
    """
    Comprehensive epoch monitoring for V8 Enhanced neuro-symbolic correction.

    Key metrics to watch (clinical focus):
    - Clinical Preservation Ratios: Primary success metrics (>1.0 = improvement)
    - Contrast, Boundary, Texture, Edge preservation
    - PSNR Delta: Secondary metric
    """
    psnr_delta = val_metrics['psnr_delta']
    ssim_delta = val_metrics['ssim_corrected'] - val_metrics['ssim_backbone']

    # Get clinical preservation ratios
    contrast_ratio = val_metrics.get('contrast_ratio', 1.0)
    boundary_ratio = val_metrics.get('boundary_ratio', 1.0)
    texture_ratio = val_metrics.get('texture_ratio', 1.0)
    edge_ratio = val_metrics.get('edge_ratio', 1.0)

    # Count clinical improvements (ratio > 1.0 means corrector improved over backbone)
    clinical_improvements = sum([
        1 if contrast_ratio > 1.0 else 0,
        1 if boundary_ratio > 1.0 else 0,
        1 if texture_ratio > 1.0 else 0,
        1 if edge_ratio > 1.0 else 0,
    ])

    # Determine success indicator based on clinical metrics
    avg_ratio = (contrast_ratio + boundary_ratio + texture_ratio + edge_ratio) / 4
    if avg_ratio > 1.05:
        status_symbol = "+++"
        status_text = "CLINICAL IMPROVEMENT"
    elif avg_ratio > 1.0:
        status_symbol = "+"
        status_text = "SLIGHT IMPROVEMENT"
    elif avg_ratio > 0.95:
        status_symbol = "~"
        status_text = "NEUTRAL"
    else:
        status_symbol = "---"
        status_text = "DEGRADING"

    # Header with primary clinical metric
    print(f"\n{'#'*78}")
    print(f"# EPOCH {epoch:3d} | Clinical Avg Ratio: {avg_ratio:.3f} [{status_symbol}] {status_text}")
    print(f"# {'':>10} | Improvements: {clinical_improvements}/4 metrics | PSNR Delta: {psnr_delta:+.3f} dB")
    print(f"{'#'*78}")

    # ===== CLINICAL PRESERVATION TABLE (PRIMARY METRICS) =====
    print(f"\n┌{'─'*86}┐")
    print(f"│ {'CLINICAL PRESERVATION':<22} {'Backbone%':>12} {'Corrected%':>12} {'Ratio':>10} {'Status':>12} {'Target':>12} │")
    print(f"├{'─'*86}┤")

    # Get absolute preservation values
    bb_contrast = val_metrics.get('backbone_contrast_pres', 0.47) * 100
    corr_contrast = val_metrics.get('corrected_contrast_pres', 0.47) * 100
    bb_boundary = val_metrics.get('backbone_boundary_pres', 0.47) * 100
    corr_boundary = val_metrics.get('corrected_boundary_pres', 0.47) * 100
    bb_texture = val_metrics.get('backbone_texture_pres', 0.41) * 100
    corr_texture = val_metrics.get('corrected_texture_pres', 0.41) * 100
    bb_edge = val_metrics.get('backbone_edge_pres', 0.68) * 100
    corr_edge = val_metrics.get('corrected_edge_pres', 0.68) * 100

    # Known backbone losses from analysis
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
    print(f"\n┌{'─'*68}┐")
    print(f"│ {'TRADITIONAL METRICS':<30} {'Backbone':>12} {'Corrected':>12} {'Delta':>10} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'PSNR (dB)':<30} {val_metrics['psnr_backbone']:>12.2f} {val_metrics['psnr_corrected']:>12.2f} {psnr_delta:>+10.3f} │")
    print(f"│ {'SSIM':<30} {val_metrics['ssim_backbone']:>12.4f} {val_metrics['ssim_corrected']:>12.4f} {ssim_delta:>+10.4f} │")
    print(f"└{'─'*68}┘")

    # ===== PREDICATE SCORES (Backbone vs Corrected) =====
    pred_scores = val_metrics['pred_scores']  # On corrected output
    pred_scores_backbone = val_metrics.get('pred_scores_backbone', pred_scores)  # On backbone
    avg_pred = sum(pred_scores.values()) / len(pred_scores)
    avg_pred_backbone = sum(pred_scores_backbone.values()) / len(pred_scores_backbone)
    passed_count = sum(1 for s in pred_scores.values() if s >= 0.5)

    print(f"\n┌{'─'*76}┐")
    print(f"│ {'GT-FREE PREDICATES':<24} {'Backbone':>12} {'Corrected':>12} {'Delta':>10} {'Status':>12} │")
    print(f"├{'─'*76}┤")

    pred_names = {'P1': 'Edge Quality', 'P2': 'Contrast', 'P3': 'Smoothness',
                  'P4': 'Structure', 'P5': 'Speckle (monitor)', 'P6': 'Anatomy'}

    for key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        score_corrected = pred_scores.get(key, 0)
        score_backbone = pred_scores_backbone.get(key, 0)
        delta = score_corrected - score_backbone
        status = "PASS" if score_corrected >= 0.5 else "FAIL"
        name = pred_names.get(key, key)
        delta_str = f"{delta:+.3f}"
        print(f"│ {name:<24} {score_backbone:>12.3f} {score_corrected:>12.3f} {delta_str:>10} {status:>12} │")

    print(f"├{'─'*76}┤")
    avg_delta = avg_pred - avg_pred_backbone
    avg_status = "PASS" if avg_pred >= 0.5 else "FAIL"
    print(f"│ {'AVERAGE':<24} {avg_pred_backbone:>12.3f} {avg_pred:>12.3f} {avg_delta:>+10.3f} {f'{passed_count}/6 {avg_status}':>12} │")
    print(f"└{'─'*76}┘")

    # ===== GT-ALIGNED PREDICATES (NEW) =====
    pred_alignment = val_metrics.get('pred_alignment', {})
    pred_improvement = val_metrics.get('pred_improvement', {})

    if pred_alignment:
        print(f"\n┌{'─'*76}┐")
        print(f"│ {'GT-ALIGNED PREDICATES':<24} {'Align Error':>12} {'Improvement':>12} {'Status':>10} {'Note':>12} │")
        print(f"├{'─'*76}┤")

        align_names = {
            'edge_quality': 'Edge Quality',
            'contrast': 'Contrast',
            'smoothness': 'Smoothness',
            'texture': 'Texture',
            'structure': 'Structure'
        }

        total_improved = 0
        for key in ['edge_quality', 'contrast', 'smoothness', 'texture', 'structure']:
            align_err = pred_alignment.get(key, 0)
            improvement = pred_improvement.get(key, 0)
            # Negative improvement = corrected is better aligned to GT
            status = "BETTER" if improvement < -0.001 else ("SAME" if abs(improvement) < 0.001 else "WORSE")
            if improvement < -0.001:
                total_improved += 1
            note = "← GT target" if abs(align_err) < 0.01 else ""
            name = align_names.get(key, key)
            print(f"│ {name:<24} {align_err:>12.4f} {improvement:>+12.4f} {status:>10} {note:>12} │")

        print(f"├{'─'*76}┤")
        avg_align_err = sum(pred_alignment.values()) / len(pred_alignment) if pred_alignment else 0
        avg_improvement = sum(pred_improvement.values()) / len(pred_improvement) if pred_improvement else 0
        overall_status = f"{total_improved}/5 BETTER" if total_improved > 0 else "NO IMPROVEMENT"
        print(f"│ {'AVERAGE':<24} {avg_align_err:>12.4f} {avg_improvement:>+12.4f} {overall_status:>22} │")
        print(f"└{'─'*76}┘")

    # ===== CORRECTION BEHAVIOR =====
    lambda_stats = val_metrics.get('lambda_stats', {})
    correction_mag = val_metrics['correction_magnitude']

    if correction_mag < 0.005:
        corr_interp = "Very Light"
    elif correction_mag < 0.015:
        corr_interp = "Moderate"
    elif correction_mag < 0.03:
        corr_interp = "Active"
    else:
        corr_interp = "Heavy (!)"

    print(f"\n┌{'─'*68}┐")
    print(f"│ {'CORRECTION BEHAVIOR':<30} {'Mean':>12} {'Max':>12} {'Status':>10} │")
    print(f"├{'─'*68}┤")

    if lambda_stats:
        for name in ['edge', 'contrast', 'smooth', 'structure', 'anatomy']:
            if name in lambda_stats:
                stats = lambda_stats[name]
                mean_val = stats['mean'] if isinstance(stats['mean'], float) else stats['mean'].item() if hasattr(stats['mean'], 'item') else float(stats['mean'])
                max_val = stats['max'] if isinstance(stats['max'], float) else stats['max'].item() if hasattr(stats['max'], 'item') else float(stats['max'])

                if mean_val < 0.02:
                    level = "Light"
                elif mean_val < 0.05:
                    level = "Moderate"
                elif mean_val < 0.10:
                    level = "Active"
                else:
                    level = "Max (!)"

                print(f"│ {'lambda_' + name:<30} {mean_val:>12.4f} {max_val:>12.4f} {level:>10} │")

    print(f"├{'─'*68}┤")
    print(f"│ {'Correction Magnitude':<30} {correction_mag:>12.6f} {'':>12} {corr_interp:>10} │")
    print(f"└{'─'*68}┘")

    # ===== LEGACY CLINICAL METRICS =====
    cnr_backbone = val_metrics.get('cnr_backbone', 0)
    cnr_corrected = val_metrics.get('cnr_corrected', 0)
    cnr_delta = cnr_corrected - cnr_backbone

    epi_backbone = val_metrics.get('epi_backbone', 0)
    epi_corrected = val_metrics.get('epi_corrected', 0)
    epi_delta = epi_corrected - epi_backbone

    bs_backbone = val_metrics.get('boundary_sharpness_backbone', 0)
    bs_corrected = val_metrics.get('boundary_sharpness_corrected', 0)
    bs_delta = bs_corrected - bs_backbone

    legacy_improved = sum([
        1 if cnr_delta > 0 else 0,
        1 if epi_delta > 0 else 0,
        1 if bs_delta > 0 else 0,
    ])

    print(f"\n┌{'─'*68}┐")
    print(f"│ {'LEGACY CLINICAL METRICS':<30} {'Backbone':>12} {'Corrected':>12} {'Delta':>10} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'CNR (Contrast-to-Noise)':<30} {cnr_backbone:>12.3f} {cnr_corrected:>12.3f} {cnr_delta:>+10.3f} │")
    print(f"│ {'EPI (Edge Preservation)':<30} {epi_backbone:>12.4f} {epi_corrected:>12.4f} {epi_delta:>+10.4f} │")
    print(f"│ {'Boundary Sharpness':<30} {bs_backbone:>12.4f} {bs_corrected:>12.4f} {bs_delta:>+10.4f} │")
    print(f"├{'─'*68}┤")
    legacy_status = f"{legacy_improved}/3 IMPROVED" if legacy_improved > 0 else "NO IMPROVEMENT"
    print(f"│ {'LEGACY ASSESSMENT':<30} {legacy_status:>36} │")
    print(f"└{'─'*68}┘")

    # ===== TRAINING LOSS =====
    print(f"\n┌{'─'*68}┐")
    print(f"│ {'TRAINING LOSS':<66} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'Total Loss':<30} {train_metrics['loss']:>36.6f} │")
    train_delta = train_metrics['psnr_corrected'] - train_metrics['psnr_backbone']
    print(f"│ {'Train PSNR Delta':<30} {train_delta:>+36.3f} │")
    print(f"└{'─'*68}┘")

    # ===== LEARNED LOSS WEIGHTS (Uncertainty Weighting) =====
    learned_weights = train_metrics.get('learned_weights', {})
    learned_uncertainties = train_metrics.get('learned_uncertainties', {})
    if learned_weights:
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'LEARNED LOSS WEIGHTS (Uncertainty Weighting)':<66} │")
        print(f"├{'─'*68}┤")
        print(f"│   {'Loss Term':<20} {'Weight':<15} {'Uncertainty (σ)':<15} {'Priority':<12} │")
        print(f"├{'─'*68}┤")

        # Sort by weight to show priority
        sorted_weights = sorted(learned_weights.items(), key=lambda x: x[1], reverse=True)
        for name, weight in sorted_weights:
            sigma = learned_uncertainties.get(name, 0)
            # Determine priority level
            if weight > 1.0:
                priority = "HIGH"
            elif weight > 0.3:
                priority = "Medium"
            elif weight > 0.1:
                priority = "Low"
            else:
                priority = "Minimal"
            print(f"│   {name:<20} {weight:<15.4f} {sigma:<15.4f} {priority:<12} │")
        print(f"└{'─'*68}┘")

    # ===== QUICK SUMMARY LINE =====
    avg_lambda = sum(s['mean'] if isinstance(s['mean'], float) else float(s['mean']) for s in lambda_stats.values()) / len(lambda_stats) if lambda_stats else 0
    print(f"\n>>> Clinical Summary: {clinical_improvements}/4 improved | Ratio: {avg_ratio:.3f} | Preservation: {avg_bb:.1f}%→{avg_corr:.1f}%")
    print(f">>> PSNR: {val_metrics['psnr_backbone']:.2f}→{val_metrics['psnr_corrected']:.2f} ({psnr_delta:+.3f}) | Corr: {correction_mag:.4f}")
    print(f"{'#'*78}\n")

    # ===== PRINT INTERPRETABILITY INFO =====
    interp = val_metrics.get('interpretability')
    if interp:
        print_interpretability_info(interp)


def print_interpretability_info(interp: dict):
    """
    Print V8 interpretability outputs: symbolic reasoning, verification,
    layer analysis, and counterfactuals.
    """
    print("\n" + "=" * 78)
    print("V8 INTERPRETABILITY & SYMBOLIC REASONING (Last Sample)")
    print("=" * 78)

    # ===== 1. CLINICAL REPORT =====
    clinical_report = interp.get('clinical_report', '')
    if clinical_report:
        print("\n" + clinical_report)

    # ===== 2. FORMAL VERIFICATION GUARANTEES =====
    verification = interp.get('verification', {})
    if verification:
        print(f"\n┌{'─'*76}┐")
        print(f"│ {'FORMAL VERIFICATION GUARANTEES':<74} │")
        print(f"├{'─'*76}┤")

        decision = verification.get('decision', 'N/A')
        passed = verification.get('guarantees_passed', 0)
        print(f"│ {'Decision:':<20} {decision:<54} │")
        print(f"│ {'Guarantees Passed:':<20} {passed}/3{' ':49} │")
        print(f"├{'─'*76}┤")

        guarantees = verification.get('guarantees', {})
        for gname, ginfo in guarantees.items():
            if isinstance(ginfo, dict):
                status = "PASS" if ginfo.get('passed', False) else "FAIL"
                if gname == 'energy':
                    before = ginfo.get('before', 0)
                    after = ginfo.get('after', 0)
                    delta = ginfo.get('delta', 0)
                    detail = f"E_before={before:.4f}, E_after={after:.4f}, delta={delta:+.4f}"
                elif gname == 'pareto':
                    improved = ginfo.get('improved', [])
                    degraded = ginfo.get('degraded', [])
                    detail = f"improved={improved}, degraded={degraded}"
                elif gname == 'lipschitz':
                    ratio = ginfo.get('ratio', 0)
                    bound = ginfo.get('bound', 0)
                    detail = f"ratio={ratio:.4f}, bound={bound:.2f}"
                else:
                    detail = str(ginfo)
                print(f"│   {gname:<12} [{status}]: {detail:<53} │")
        print(f"└{'─'*76}┘")

    # ===== 3. SYMBOLIC REASONING EXPLANATIONS =====
    explanations = interp.get('explanations', {})
    if explanations:
        print(f"\n┌{'─'*76}┐")
        print(f"│ {'SYMBOLIC REASONING TRACE':<74} │")
        print(f"├{'─'*76}┤")
        for exp_name, exp_text in explanations.items():
            # Highlight active rules
            if 'ACTIVE' in str(exp_text):
                print(f"│   [*] {str(exp_text):<69} │")
            elif 'CONFLICT' in str(exp_name):
                print(f"│   [!] {str(exp_text):<69} │")
            elif 'COMPOUND' in str(exp_name):
                print(f"│   [+] {str(exp_text):<69} │")
            else:
                print(f"│   [ ] {str(exp_text):<69} │")
        print(f"└{'─'*76}┘")

    # ===== 4. CLINICAL LAYER ANALYSIS =====
    layer_analysis = interp.get('layer_analysis', {})
    if layer_analysis:
        print(f"\n┌{'─'*76}┐")
        print(f"│ {'CLINICAL LAYER ANALYSIS (OCT Retinal Layers)':<74} │")
        print(f"├{'─'*76}┤")
        print(f"│   {'Layer':<12} {'Status':<10} {'Symbol':<8} {'Severity':<12} {'Assessment':<26} │")
        print(f"├{'─'*76}┤")
        for layer_name, layer_info in layer_analysis.items():
            if isinstance(layer_info, dict):
                status = layer_info.get('status', 'N/A')
                symbol = layer_info.get('symbol', '?')
                severity = layer_info.get('severity', 0)

                # Color-code severity
                if severity < 0.2:
                    assessment = "Excellent"
                elif severity < 0.35:
                    assessment = "Good"
                elif severity < 0.5:
                    assessment = "Fair - monitor"
                else:
                    assessment = "Poor - needs attention"

                print(f"│   {layer_name:<12} {status:<10} [{symbol}]{' ':4} {severity:<12.3f} {assessment:<26} │")
        print(f"└{'─'*76}┘")

    # ===== 5. COUNTERFACTUAL ANALYSIS =====
    counterfactuals = interp.get('counterfactuals', {})
    if counterfactuals:
        print(f"\n┌{'─'*76}┐")
        print(f"│ {'COUNTERFACTUAL ANALYSIS':<74} │")
        print(f"├{'─'*76}┤")
        target = counterfactuals.get('target', 'P1')
        original = counterfactuals.get('original_score', 0)
        print(f"│   Target Predicate: {target}, Original Score: {original:.3f}{' ':32} │")
        print(f"│   'What if {target} score were different?'{' ':37} │")
        print(f"├{'─'*76}┤")

        interventions = counterfactuals.get('interventions', [])
        for item in interventions:
            if isinstance(item, dict):
                score_key = f'{target}_score'
                score = item.get(score_key, 0)
                # Find the activation key (e.g., 'edge_activation')
                for k, v in item.items():
                    if 'activation' in k:
                        print(f"│   If {target}={score:.1f} → {k}={v:.3f}{' ':45} │")
                        break
        print(f"└{'─'*76}┘")

    print("\n" + "=" * 78)


def main():
    parser = argparse.ArgumentParser(description='Train V8 Enhanced Neuro-Symbolic OCT Denoising')

    # Data
    parser.add_argument('--train_jsonl', default='pku37_oct_dataset/pku37_real_train.jsonl',
                        help='Training data JSONL (default: 1388 pairs)')
    parser.add_argument('--val_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl',
                        help='Validation data JSONL (default: 173 pairs)')
    parser.add_argument('--max_train', type=int, default=None,
                        help='Max training samples (default: use all)')
    parser.add_argument('--max_val', type=int, default=None,
                        help='Max validation samples (default: use all)')
    parser.add_argument('--patch_size', type=int, default=96)

    # Model
    parser.add_argument('--backbone_width', type=int, default=40)
    parser.add_argument('--pretrained_backbone', type=str,
                        default='outputs/nafnet_pku37_w40/best_model.pth')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone weights (default: True)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output
    parser.add_argument('--output_dir', default='outputs/nsnd_v8_enhanced')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("NEURO-SYMBOLIC OCT DENOISING V8 ENHANCED")
    print("V8 Features + AdaptiveLambda + Feature Fusion + Attention")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backbone width: {args.backbone_width}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Pretrained: {args.pretrained_backbone}")
    print(f"  Train data: {args.train_jsonl}")
    print(f"  Val data: {args.val_jsonl}")
    print(f"  Max train: {args.max_train if args.max_train else 'all'}")
    print(f"  Max val: {args.max_val if args.max_val else 'all'}")
    print(f"  Device: {args.device}")

    # Data
    print("\nLoading PKU37 data...")
    train_dataset = PKU37Dataset(
        args.train_jsonl,
        max_samples=args.max_train,
        patch_size=args.patch_size,
        is_train=True
    )
    val_dataset = PKU37Dataset(
        args.val_jsonl,
        max_samples=args.max_val,
        patch_size=0,
        is_train=False
    )

    # SPEED FIX: Optimize DataLoader settings
    num_workers = 4 if args.device != 'cpu' else 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(args.device != 'cpu'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None  # SPEED FIX: Prefetch more batches for better throughput
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=min(2, num_workers), pin_memory=(args.device != 'cpu')
    )

    # Model
    print("\nInitializing model...")
    model = NeuroSymbolicDenoiserV8Enhanced(
        backbone_width=args.backbone_width,
        pretrained_backbone=args.pretrained_backbone
    ).to(args.device)

    # Freeze backbone if requested
    if args.freeze_backbone:
        print("\n*** FREEZING BACKBONE WEIGHTS ***")
        for param in model.backbone.parameters():
            param.requires_grad = False
        backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        corrector_trainable = sum(p.numel() for p in model.corrector.parameters() if p.requires_grad)
        print(f"  Backbone trainable params: {backbone_trainable:,} (frozen)")
        print(f"  Corrector trainable params: {corrector_trainable:,}")

    # Loss and optimizer
    # Use uncertainty weighting to automatically learn loss weights
    criterion = V8EnhancedLoss(use_uncertainty_weighting=True)
    criterion = criterion.to(args.device)  # Move to device for learnable params

    # Only train corrector components (backbone is frozen)
    corrector_params = list(model.corrector.correctors.parameters())
    lambda_params = list(model.corrector.lambda_predictor.parameters())
    router_params = list(model.corrector.router.parameters())

    # Learnable loss weights (uncertainty parameters)
    loss_weight_params = list(criterion.log_sigma.parameters())

    if args.freeze_backbone:
        # Corrector + loss weight parameters
        optimizer = torch.optim.AdamW([
            {'params': corrector_params, 'lr': args.lr * 2},   # Correctors
            {'params': lambda_params, 'lr': args.lr * 5},      # Lambda predictor (fastest)
            {'params': router_params, 'lr': args.lr},          # Router
            {'params': loss_weight_params, 'lr': args.lr * 0.5, 'weight_decay': 0},  # Loss weights (no decay)
        ], weight_decay=1e-4)
        print(f"  Optimizer: AdamW (corrector lr={args.lr*2:.0e}, lambda lr={args.lr*5:.0e})")
        print(f"  Uncertainty Weighting: ENABLED (learning loss weights automatically)")
    else:
        backbone_params = list(model.backbone.parameters())
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.5},
            {'params': corrector_params, 'lr': args.lr * 2},
            {'params': lambda_params, 'lr': args.lr * 5},
            {'params': router_params, 'lr': args.lr},
            {'params': loss_weight_params, 'lr': args.lr * 0.5, 'weight_decay': 0},
        ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # SPEED FIX: Mixed precision training
    use_amp = args.device != 'cpu' and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("  Using Automatic Mixed Precision (AMP)")

    # Training loop
    print("\n" + "#"*70)
    print("# TRAINING V8 ENHANCED NEURO-SYMBOLIC DENOISER")
    print("# Backbone: FROZEN (pretrained NAFNet w40, 30.91 dB)")
    print("# Focus: Corrector improvement via symbolic reasoning")
    print("#"*70)

    best_psnr_delta = -float('inf')
    prev_val_metrics = None  # Track previous validation metrics for trend display

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch, scaler)
        scheduler.step()

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, args.device)
            print_epoch_metrics(epoch, train_metrics, val_metrics, prev_val_metrics)
            prev_val_metrics = val_metrics  # Store for next epoch's trend comparison

            if val_metrics['psnr_delta'] > best_psnr_delta:
                best_psnr_delta = val_metrics['psnr_delta']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_psnr_delta': best_psnr_delta,
                }, os.path.join(args.output_dir, 'best_model_v8_enhanced.pth'))
                print(f"*** New best model! PSNR delta: {best_psnr_delta:+.3f} dB ***")
        else:
            print(f"[Epoch {epoch}] Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr_corrected']:.2f}")

        # MEMORY FIX: Clean up between epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "#"*70)
    print("# TRAINING COMPLETE")
    print("#"*70)
    print(f"Best PSNR delta: {best_psnr_delta:+.3f} dB")
    print(f"Model saved to: {args.output_dir}/best_model_v8_enhanced.pth")


if __name__ == '__main__':
    main()
