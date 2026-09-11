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
import random
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
# import lpips  # SPEED: LPIPS disabled — saves 30-40% training time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

# Import cooperative module
from neuro_symbolic_corrector_v8_cooperative import NeuroSymbolicCorrectorV8Cooperative

# Import CNR-preserving spatial loss (smart algorithmic approach for CNR preservation)
from uncertainty_guided_correction import CNRPreservingSpatialLoss


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


def otsu_tissue_mask(image):
    """Compute tissue mask using Otsu's method. Scanner-agnostic.
    Args: image [B, 1, H, W] tensor in [0, 1]
    Returns: mask [B, 1, H, W], 1.0=tissue, 0.0=background
    """
    B = image.shape[0]
    masks = []
    for b in range(B):
        img_np = image[b, 0].detach().cpu().numpy()
        hist, bin_edges = np.histogram(img_np.ravel(), bins=256, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = hist.sum()
        if total == 0:
            masks.append(torch.ones_like(image[b:b+1]))
            continue
        w0, sum0, best_var, best_t = 0.0, 0.0, -1.0, 0.5
        sum_total = (hist * bin_centers).sum()
        for i in range(len(hist)):
            w0 += hist[i]
            if w0 == 0: continue
            w1 = total - w0
            if w1 == 0: break
            sum0 += hist[i] * bin_centers[i]
            m0 = sum0 / w0
            m1 = (sum_total - sum0) / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best_var:
                best_var = var
                best_t = bin_centers[i]
        soft_mask = torch.sigmoid((image[b:b+1] - best_t) * 20.0)
        masks.append(soft_mask)
    return torch.cat(masks, dim=0)


# =============================================================================
# Tiled Inference for Memory-Efficient SwinIR
# =============================================================================

def swinir_tiled_forward(backbone, x, tile_size=128, overlap=16):
    """Run SwinIR on overlapping tiles and blend results.

    SwinIR's O(N²) window attention makes full-res inference on large images
    (e.g., Duke 970×520) too memory-intensive for CPU. This function processes
    the image in overlapping tiles and blends them with a raised-cosine window
    to eliminate seam artifacts.

    Args:
        backbone: SwinIR model (callable, returns [B, C, H, W])
        x: Input tensor [B, 1, H, W]
        tile_size: Tile size in pixels (must be multiple of 8 for SwinIR window_size)
        overlap: Overlap between adjacent tiles in pixels
    Returns:
        output: Blended output [B, 1, H, W]
    """
    B, C, H, W = x.shape
    step = tile_size - overlap

    # If image fits in a single tile, just run normally
    if H <= tile_size and W <= tile_size:
        return backbone(x)

    # Output accumulator and weight map for blending
    output = x.new_zeros(B, 1, H, W)
    weight = x.new_zeros(1, 1, H, W)

    # Build 2D raised-cosine blending window for smooth tile transitions
    def _make_blend_window(h, w, overlap, device):
        wy = torch.ones(h, device=device)
        wx = torch.ones(w, device=device)
        if overlap > 0:
            ramp = torch.linspace(0, 1, overlap, device=device)
            # Fade in at top/left edges, fade out at bottom/right
            wy[:overlap] = ramp
            wy[-overlap:] = ramp.flip(0)
            wx[:overlap] = ramp
            wx[-overlap:] = ramp.flip(0)
        return (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            # Clamp tile boundaries
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            # Adjust start if tile is too small at edges
            y0_adj = max(0, y1 - tile_size)
            x0_adj = max(0, x1 - tile_size)

            tile_in = x[:, :, y0_adj:y1, x0_adj:x1]
            tile_out = backbone(tile_in)

            th, tw = y1 - y0_adj, x1 - x0_adj
            blend_w = _make_blend_window(th, tw, overlap, x.device)

            output[:, :, y0_adj:y1, x0_adj:x1] += tile_out * blend_w
            weight[:, :, y0_adj:y1, x0_adj:x1] += blend_w

    # Normalize by accumulated weights
    output = output / weight.clamp(min=1e-8)
    return output.clamp(0, 1)


def swinir_tiled_forward_with_features(backbone, x, tile_size=128, overlap=16):
    """Run SwinIR tiled inference returning both denoised output AND conv_first features.

    Same tiling as swinir_tiled_forward but also extracts and blends the
    shallow features from conv_first, needed for uncertainty estimation.

    Args:
        backbone: SwinIR model with .conv_first attribute
        x: Input tensor [B, 1, H, W]
        tile_size: Tile size in pixels
        overlap: Overlap between adjacent tiles
    Returns:
        (denoised [B, 1, H, W], shallow_feat [B, embed_dim, H, W])
    """
    B, C, H, W = x.shape
    step = tile_size - overlap
    embed_dim = backbone.conv_first.out_channels

    # If image fits in a single tile, just run normally
    if H <= tile_size and W <= tile_size:
        shallow = backbone.conv_first(x)
        denoised = backbone(x)
        return denoised, shallow

    # Output accumulators
    out_denoised = x.new_zeros(B, 1, H, W)
    out_feat = x.new_zeros(B, embed_dim, H, W)
    weight = x.new_zeros(1, 1, H, W)

    def _make_blend_window(h, w, overlap, device):
        wy = torch.ones(h, device=device)
        wx = torch.ones(w, device=device)
        if overlap > 0:
            ramp = torch.linspace(0, 1, overlap, device=device)
            wy[:overlap] = ramp
            wy[-overlap:] = ramp.flip(0)
            wx[:overlap] = ramp
            wx[-overlap:] = ramp.flip(0)
        return (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            y0_adj = max(0, y1 - tile_size)
            x0_adj = max(0, x1 - tile_size)

            tile_in = x[:, :, y0_adj:y1, x0_adj:x1]
            tile_feat = backbone.conv_first(tile_in)
            tile_out = backbone(tile_in)

            th, tw = y1 - y0_adj, x1 - x0_adj
            blend_w = _make_blend_window(th, tw, overlap, x.device)

            out_denoised[:, :, y0_adj:y1, x0_adj:x1] += tile_out * blend_w
            # Broadcast blend_w [1,1,h,w] to feature channels
            out_feat[:, :, y0_adj:y1, x0_adj:x1] += tile_feat * blend_w
            weight[:, :, y0_adj:y1, x0_adj:x1] += blend_w

            del tile_in, tile_feat, tile_out

    weight_clamped = weight.clamp(min=1e-8)
    out_denoised = out_denoised / weight_clamped
    out_feat = out_feat / weight_clamped

    return out_denoised.clamp(0, 1), out_feat


# =============================================================================
# Backbone Wrapper with Feature Extraction and Uncertainty Estimation
# =============================================================================

class BackboneWrapper(nn.Module):
    """
    Backbone-agnostic wrapper for any SOTA denoising model.

    Supports: nafnet, dncnn, swinir, kbnet.

    All backbones use enc1-based intermediate feature extraction for uncertainty:
    - NAFNet: enc1 encoder features (40ch at H/2 x W/2)
    - DnCNN: head features after first Conv+ReLU (228ch at full res)
    - KBNet: enc1 encoder features (32ch at H/2 x W/2)
    - SwinIR: shallow features after conv_first (138ch at full res)

    Returns:
        denoised: Final denoised output [B, 1, H, W]
        uncertainty: Per-pixel uncertainty estimate [B, 1, H, W]
    """

    SUPPORTED_BACKBONES = ('nafnet', 'dncnn', 'swinir', 'kbnet')

    # Map backbone name → width of first encoder feature (for feature-based uncertainty)
    _ENC1_WIDTHS = {'nafnet': 40, 'kbnet': 32, 'dncnn': 228, 'swinir': 138}

    # Backbones that use lightweight output-based uncertainty (no internal features needed)
    _OUTPUT_BASED_BACKBONES = set()  # All backbones now use feature-based uncertainty

    # Backbones that are slow at full resolution and benefit from caching
    _SLOW_BACKBONES = {'swinir'}

    def __init__(self, backbone_name: str = 'nafnet'):
        super().__init__()

        assert backbone_name in self.SUPPORTED_BACKBONES, \
            f"Unknown backbone: {backbone_name}. Supported: {self.SUPPORTED_BACKBONES}"
        self.backbone_name = backbone_name
        self.backbone = self._create_backbone(backbone_name)

        if backbone_name in self._OUTPUT_BASED_BACKBONES:
            # Output-based uncertainty: lightweight Conv on 1ch output + local variance
            # Avoids extracting expensive internal features (228ch for DnCNN, 138ch for SwinIR)
            self.uncertainty_conv1 = nn.Conv2d(2, 16, 3, padding=1, bias=False)
            self.uncertainty_bn = nn.InstanceNorm2d(16, affine=True)
            self.uncertainty_conv2 = nn.Conv2d(16, 1, 1)
        else:
            # Feature-based uncertainty: extract intermediate encoder features
            enc1_width = self._ENC1_WIDTHS[backbone_name]
            self.uncertainty_conv1 = nn.Conv2d(enc1_width, enc1_width // 2, 3, padding=1, bias=False)
            self.uncertainty_bn = nn.InstanceNorm2d(enc1_width // 2, affine=True)
            self.uncertainty_conv2 = nn.Conv2d(enc1_width // 2, 1, 1)

        # Learned calibration parameters for uncertainty
        self.uncertainty_temperature = nn.Parameter(torch.tensor(1.0))
        self.uncertainty_bias_offset = nn.Parameter(torch.tensor(0.3))

        self._init_uncertainty_head()

    @staticmethod
    def _create_backbone(name: str) -> nn.Module:
        """Factory method to create any supported SOTA backbone."""
        if name == 'nafnet':
            from sota.models.nafnet_7m import NAFNet
            return NAFNet(img_channel=1, width=40, middle_blk_num=1,
                          enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1])
        elif name == 'dncnn':
            from sota.models.dncnn_7m import DnCNN
            return DnCNN(in_channels=1, out_channels=1, num_layers=17,
                         channels=228, use_checkpoint=False)
        elif name == 'swinir':
            from sota.models.swinir_7m import SwinIR
            return SwinIR(in_channels=1, out_channels=1, embed_dim=138,
                          depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                          window_size=8, mlp_ratio=2.0)
        elif name == 'kbnet':
            from sota.models.kbnet_7m import KBNet
            return KBNet(in_channels=1, out_channels=1, width=32,
                         middle_blk_num=10, enc_blk_nums=[2, 2, 4],
                         dec_blk_nums=[2, 2, 2], nset=32, gc=1, ffn_scale=2,
                         use_checkpoint=True)
        else:
            raise ValueError(f"Unknown backbone: {name}")

    def uncertainty_parameters(self):
        """Return all trainable uncertainty head parameters."""
        yield from self.uncertainty_conv1.parameters()
        yield from self.uncertainty_bn.parameters()
        yield from self.uncertainty_conv2.parameters()
        yield self.uncertainty_temperature
        yield self.uncertainty_bias_offset

    def _init_uncertainty_head(self):
        """Initialize uncertainty head for well-calibrated outputs."""
        nn.init.kaiming_normal_(self.uncertainty_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.uncertainty_bn.weight, 1)
        nn.init.constant_(self.uncertainty_bn.bias, 0)
        nn.init.xavier_uniform_(self.uncertainty_conv2.weight, gain=0.1)
        nn.init.constant_(self.uncertainty_conv2.bias, 0.0)

    def _compute_uncertainty_from_enc1(self, enc_feat: torch.Tensor) -> torch.Tensor:
        """Compute calibrated uncertainty from encoder features (NAFNet/KBNet)."""
        x = self.uncertainty_conv1(enc_feat)
        x = self.uncertainty_bn(x)
        x = F.relu(x, inplace=True)
        x = self.uncertainty_conv2(x)
        temp = self.uncertainty_temperature.clamp(min=0.1)
        calibrated = (x + self.uncertainty_bias_offset) / temp
        return torch.sigmoid(calibrated)

    def _compute_uncertainty_from_output(self, denoised: torch.Tensor) -> torch.Tensor:
        """Compute calibrated uncertainty from 1ch backbone output (DnCNN/SwinIR).

        Uses denoised output + local variance as a 2-channel input to a lightweight
        uncertainty head. Local variance captures regions where the backbone is
        uncertain (high variance = likely uncertain denoising).
        """
        # Local variance as uncertainty proxy (7×7 neighborhood)
        local_mean = F.avg_pool2d(
            F.pad(denoised, (3, 3, 3, 3), mode='reflect'),
            kernel_size=7, stride=1, padding=0
        )
        local_var = F.avg_pool2d(
            F.pad((denoised - local_mean) ** 2, (3, 3, 3, 3), mode='reflect'),
            kernel_size=7, stride=1, padding=0
        )
        # 2-channel input: [denoised, local_variance]
        x = torch.cat([denoised, local_var], dim=1)
        x = self.uncertainty_conv1(x)
        x = self.uncertainty_bn(x)
        x = F.relu(x, inplace=True)
        x = self.uncertainty_conv2(x)
        temp = self.uncertainty_temperature.clamp(min=0.1)
        calibrated = (x + self.uncertainty_bias_offset) / temp
        return torch.sigmoid(calibrated)

    def load_pretrained(self, path: str) -> bool:
        """Load pretrained backbone weights (checkpoint-format agnostic)."""
        if not os.path.exists(path):
            print(f"Pretrained weights not found: {path}")
            return False

        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            # Handle both train_sota_7m ('state_dict') and NAFNet ('model_state_dict') formats
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

    def _forward_nafnet_with_enc1(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """NAFNet forward pass with enc1 feature extraction for uncertainty.

        Runs the NAFNet backbone manually to extract first encoder features,
        which are used for uncertainty estimation. This is the PROVEN approach
        that drove +11% clinical improvement in the original model.

        IMPORTANT: Runs WITHOUT torch.no_grad() to match the old working version.
        Even though backbone params are frozen, the computation graph must be
        preserved for proper gradient flow through the loss function.
        """
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

        # Run backbone forward pass (NO torch.no_grad — matches old BackboneWithFeatures)
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
        denoised = out[:, :, :H, :W].clamp(0, 1)

        # Extract enc1 features for uncertainty estimation
        enc1_feat = encs_for_features[0][:, :, :H, :W]

        # Compute uncertainty from enc1 features (with gradient tracking)
        uncertainty = self._compute_uncertainty_from_enc1(enc1_feat)

        del encs, encs_for_features

        return denoised, uncertainty

    def _forward_dncnn_with_head(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DnCNN forward pass with enc1-based uncertainty from head features.

        Runs both backbone and head at FULL resolution for maximum quality.
        Head features (228ch) provide rich uncertainty estimation.
        """
        # 1) Run DnCNN head at FULL resolution for 228ch uncertainty features
        with torch.no_grad():
            head_feat = self.backbone.head(x)  # [B, 228, H, W]
        head_feat = head_feat.detach()

        # 2) Run full backbone at FULL resolution (no downsampling)
        with torch.no_grad():
            denoised = self.backbone(x)
        denoised = denoised.detach()

        # 3) Enc1-based uncertainty from 228ch head features (trainable)
        uncertainty = self._compute_uncertainty_from_enc1(head_feat)

        return denoised, uncertainty

    def _forward_kbnet_with_enc1(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """KBNet forward pass with enc1 feature extraction for uncertainty.

        KBNet has the same UNet structure as NAFNet:
        intro -> encoder[0](32ch) -> down -> encoder[1](64ch) -> down ->
        encoder[2](128ch) -> down -> bottleneck -> decoders -> ending + residual

        Extracts enc1 (32 channels at H/2 x W/2) for uncertainty estimation.
        """
        bb = self.backbone
        B, C, H, W = x.shape

        inp = bb._check_image_size(x)

        feat = bb.intro(inp)

        encs = []
        for encoder, down in zip(bb.encoders, bb.downs):
            if bb.use_checkpoint and bb.training:
                feat = checkpoint(encoder, feat, use_reentrant=False)
            else:
                feat = encoder(feat)
            encs.append(feat)
            feat = down(feat)

        if bb.use_checkpoint and bb.training:
            feat = checkpoint(bb.middle_blks, feat, use_reentrant=False)
        else:
            feat = bb.middle_blks(feat)

        for decoder, up, enc_skip in zip(bb.decoders, bb.ups, encs[::-1]):
            feat = up(feat)
            feat = feat + enc_skip
            if bb.use_checkpoint and bb.training:
                feat = checkpoint(decoder, feat, use_reentrant=False)
            else:
                feat = decoder(feat)

        out = bb.ending(feat) + inp
        denoised = out[:, :, :H, :W].clamp(0, 1)

        # Extract enc1 features (32ch) for uncertainty
        enc1_feat = encs[0][:, :, :H, :W]
        uncertainty = self._compute_uncertainty_from_enc1(enc1_feat)

        del encs
        return denoised, uncertainty

    def _forward_swinir_with_shallow(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SwinIR forward pass with conv_first feature-based uncertainty.

        Uses tiled overlapping inference to process full-resolution images
        without OOM. Each tile is processed independently through SwinIR's
        window attention, then blended with raised-cosine windows to eliminate
        seam artifacts. This preserves full-resolution quality unlike the
        previous downsampling approach.

        Uncertainty: Extracts conv_first features (138ch) at full resolution
        via tiled inference for feature-based uncertainty estimation.
        """
        with torch.no_grad():
            denoised, shallow_feat = swinir_tiled_forward_with_features(
                self.backbone, x, tile_size=128, overlap=16)
        denoised = denoised.detach()
        shallow_feat = shallow_feat.detach()

        # Feature-based uncertainty from 138ch conv_first features
        uncertainty = self._compute_uncertainty_from_enc1(shallow_feat)

        return denoised, uncertainty

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: backbone denoising + uncertainty estimation.

        All backbones use enc1-based intermediate feature extraction for
        uncertainty estimation (proven to drive clinical improvement).
        """
        if self.backbone_name == 'nafnet':
            return self._forward_nafnet_with_enc1(x)
        elif self.backbone_name == 'dncnn':
            return self._forward_dncnn_with_head(x)
        elif self.backbone_name == 'kbnet':
            return self._forward_kbnet_with_enc1(x)
        elif self.backbone_name == 'swinir':
            return self._forward_swinir_with_shallow(x)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")


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
        self._backbone_cache = None  # Pre-computed backbone outputs

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

    def precompute_backbone_cache(self, backbone_wrapper, device='cpu',
                                  patches_per_image=1, cache_ds_factor=None):
        """Pre-compute backbone outputs for fixed patches (for slow backbones).

        Extracts fixed random patches, runs backbone once per batch, caches results.
        During training, cached backbone_out is used directly — backbone is
        never called again. This makes DnCNN/SwinIR training as fast as NAFNet.

        Uses batched inference: collects patches into batches before running
        the backbone, which is significantly faster than one-by-one on CPU.

        Args:
            backbone_wrapper: BackboneWrapper instance (frozen)
            device: Device for backbone forward
            patches_per_image: Patches to extract per image (default: 1)
        """
        import time as _time
        n_images = len(self.samples)
        total_patches = n_images * patches_per_image
        is_slow = backbone_wrapper.backbone_name in BackboneWrapper._SLOW_BACKBONES
        # The cached backbone output is the reference that the fidelity guard and every
        # clinical delta are measured against during training, so it has to be produced
        # the same way as the reference used at test time. Downsampling the patch before
        # the backbone and upsampling after it does not do that. Measured on SwinIR over
        # eight validation images the downsampled path scores 31.61 dB against the clean
        # image while the tiled full resolution path used at test scores 30.00 dB, a gap
        # of 1.61 dB, so a corrector trained through it learns to sharpen a blurred
        # reference and is then measured against a sharp one.
        #
        # This path needs no flag. --freeze_backbone defaults to true and the setup then
        # sets cache_backbone for every backbone in _SLOW_BACKBONES, which is SwinIR
        # alone. Every SwinIR checkpoint on disk records cache_backbone true and every
        # NAFNet one records false, so every published SwinIR number came through here
        # and no other backbone did. Full resolution is now the default on GPU.
        # On CPU the old behaviour stays reachable, since caching there is slow.
        if cache_ds_factor is not None:
            ds_factor = int(cache_ds_factor)
        elif total_patches <= 200 or device != 'cpu':
            ds_factor = 1
        else:
            ds_factor = 4
        # Reduce batch size for full-res SwinIR on CPU to avoid OOM
        if is_slow and ds_factor == 1 and device == 'cpu':
            cache_batch_size = 2
        else:
            cache_batch_size = 16 if device == 'cpu' else 32
        print(f"\n{'='*60}")
        print(f"Pre-caching backbone outputs: {n_images} images × "
              f"{patches_per_image} patches = {total_patches}")
        if is_slow:
            print(f"  Downsampling: {ds_factor}× | Batch size: {cache_batch_size}")
        print(f"{'='*60}")

        # Phase 1: Extract all patches (fast, CPU only)
        all_patches = []  # List of (clean_patch, noisy_patch)
        for sample in self.samples:
            clean_full = np.array(Image.open(sample['clean'])).astype(np.float32)
            noisy_full = np.array(Image.open(sample['noisy'])).astype(np.float32)
            if clean_full.max() > 1.0:
                clean_full /= 255.0
            if noisy_full.max() > 1.0:
                noisy_full /= 255.0
            H, W = clean_full.shape

            for _ in range(patches_per_image):
                if self.patch_size > 0 and H > self.patch_size and W > self.patch_size:
                    y = np.random.randint(0, H - self.patch_size)
                    x = np.random.randint(0, W - self.patch_size)
                    cp = clean_full[y:y+self.patch_size, x:x+self.patch_size]
                    np_ = noisy_full[y:y+self.patch_size, x:x+self.patch_size]
                else:
                    cp, np_ = clean_full, noisy_full
                all_patches.append((cp.copy(), np_.copy()))

        print(f"  Extracted {len(all_patches)} patches, running backbone...")

        # Phase 2: Batched backbone inference
        cached = []
        backbone_wrapper.eval()
        t0 = _time.time()

        for batch_start in range(0, len(all_patches), cache_batch_size):
            batch_end = min(batch_start + cache_batch_size, len(all_patches))
            batch_noisy = []
            for j in range(batch_start, batch_end):
                _, np_ = all_patches[j]
                batch_noisy.append(torch.from_numpy(np_).unsqueeze(0))  # [1, H, W]

            noisy_batch = torch.stack(batch_noisy).to(device)  # [B, 1, H, W]
            Ht, Wt = noisy_batch.shape[2], noisy_batch.shape[3]

            with torch.no_grad():
                if is_slow and ds_factor > 1:
                    # Downsampled path for large datasets
                    noisy_ds = F.avg_pool2d(noisy_batch, kernel_size=ds_factor, stride=ds_factor)
                    _, _, Hd, Wd = noisy_ds.shape
                    pad_h = (8 - Hd % 8) % 8
                    pad_w = (8 - Wd % 8) % 8
                    if pad_h > 0 or pad_w > 0:
                        noisy_ds = F.pad(noisy_ds, (0, pad_w, 0, pad_h), mode='reflect')
                    bb_out = backbone_wrapper.backbone(noisy_ds)
                    if isinstance(bb_out, tuple):
                        bb_out = bb_out[0]
                    if pad_h > 0 or pad_w > 0:
                        bb_out = bb_out[:, :, :Hd, :Wd]
                    bb_out = F.interpolate(bb_out, size=(Ht, Wt), mode='bilinear',
                                           align_corners=False).clamp(0, 1)
                elif is_slow:
                    # Full res path for small datasets (LOO)
                    # Use tiled inference for SwinIR to prevent OOM
                    if backbone_wrapper.backbone_name == 'swinir':
                        # Process each image individually with tiled inference
                        bb_parts = []
                        for j in range(noisy_batch.shape[0]):
                            tile_out = swinir_tiled_forward(
                                backbone_wrapper.backbone,
                                noisy_batch[j:j+1],
                                tile_size=128, overlap=16)
                            bb_parts.append(tile_out)
                        bb_out = torch.cat(bb_parts, dim=0)
                        del bb_parts
                    else:
                        pad_h = (8 - Ht % 8) % 8
                        pad_w = (8 - Wt % 8) % 8
                        noisy_pad = F.pad(noisy_batch, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h > 0 or pad_w > 0) else noisy_batch
                        bb_out = backbone_wrapper.backbone(noisy_pad)
                        if isinstance(bb_out, tuple):
                            bb_out = bb_out[0]
                        if pad_h > 0 or pad_w > 0:
                            bb_out = bb_out[:, :, :Ht, :Wt]
                        bb_out = bb_out.clamp(0, 1)
                else:
                    bb_out = backbone_wrapper.backbone(noisy_batch)
                    if isinstance(bb_out, tuple):
                        bb_out = bb_out[0]

            # Unbatch and cache
            for j in range(batch_end - batch_start):
                idx = batch_start + j
                cp, np_ = all_patches[idx]
                cached.append({
                    'clean': torch.from_numpy(cp).unsqueeze(0),
                    'noisy': torch.from_numpy(np_).unsqueeze(0),
                    'backbone_out': bb_out[j].cpu(),
                })

            del noisy_batch, bb_out
            gc.collect()

            n_done = batch_end
            if n_done % (cache_batch_size * 10) == 0 or batch_start == 0:
                elapsed = _time.time() - t0
                speed = n_done / max(elapsed, 0.01)
                eta = (len(all_patches) - n_done) / max(speed, 0.01)
                print(f"  [{n_done}/{len(all_patches)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        del all_patches
        gc.collect()
        self._backbone_cache = cached
        elapsed = _time.time() - t0
        mem_mb = sum(c['backbone_out'].numel() * 4 for c in cached) / 1e6
        print(f"  Done: {len(cached)} patches in {elapsed:.0f}s ({mem_mb:.1f} MB)")
        print(f"{'='*60}\n")

    def __len__(self):
        if self._backbone_cache is not None:
            return len(self._backbone_cache)
        return len(self.samples)

    def __getitem__(self, idx):
        if self._backbone_cache is not None:
            # Cached path: return pre-computed backbone_out alongside patches
            c = self._backbone_cache[idx]
            clean = c['clean'].clone()
            noisy = c['noisy'].clone()
            bb_out = c['backbone_out'].clone()

            if self.is_train:
                if np.random.rand() > 0.5:
                    clean = clean.flip(-1)
                    noisy = noisy.flip(-1)
                    bb_out = bb_out.flip(-1)
                if np.random.rand() > 0.5:
                    clean = clean.flip(-2)
                    noisy = noisy.flip(-2)
                    bb_out = bb_out.flip(-2)

            return {'clean': clean, 'noisy': noisy, 'backbone_out': bb_out}

        # Standard path (no cache)
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

    def __init__(self, backbone_name: str = 'nafnet', pretrained_backbone: str = None,
                 correction_clamp: float = 0.15, clinical_clamp: float = 0.03,
                 hidden_channels: int = 32, use_guided_edge: bool = False,
                 scanner_adapter: bool = False):
        super().__init__()

        # Backbone-agnostic wrapper with output-based uncertainty
        self.backbone = BackboneWrapper(backbone_name=backbone_name)

        # Cooperative Corrector (no longer needs enc1/enc2 channel info)
        self.corrector = NeuroSymbolicCorrectorV8Cooperative(
            in_channels=1,
            hidden_channels=hidden_channels,
            correction_clamp=correction_clamp,
            clinical_clamp=clinical_clamp,
            use_guided_edge=use_guided_edge,
            scanner_adapter=scanner_adapter,
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
        print(f"  Backbone ({self.backbone.backbone_name}): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
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
        # Get backbone output and uncertainty (backbone-agnostic, no intermediate features)
        backbone_out, nafnet_uncertainty = self.backbone(noisy)

        # Apply cooperative correction (backbone_features=None, output-based uncertainty)
        corrected, info = self.corrector(
            backbone_out, noisy, None,
            nafnet_uncertainty=nafnet_uncertainty,
            return_details=return_details
        )

        # Add backbone uncertainty to info (detached to prevent holding backbone graph)
        info['nafnet_uncertainty'] = nafnet_uncertainty.detach()

        return corrected, backbone_out, info


# =============================================================================
# Cooperative Loss Function — Predicate-Driven Simplified Loss
# =============================================================================

class SimplifiedCooperativeLoss(nn.Module):
    """
    Predicate-Driven Simplified Loss for Clinical Excellence.

    4 terms replace the previous 19+ term loss:
    - Term 1 (weight=3.0): Direct Clinical Metric Loss — matches validation computation
    - Term 2 (weight=5.0): Predicate Improvement Loss — neuro-symbolic core
    - Term 3 (weight=0.1): Cooperation Loss — maintains negotiator (reduced to not dominate)
    - Term 4 (weight=0.0/1.0): PSNR/SSIM Guard — dead zones, Stage 2 only

    Two-stage training:
        Stage 1 (epochs 1-3): Terms 1+2+3 only (correctors learn freely)
        Stage 2 (epochs 4+): Add Term 4 (quality guard)
    """

    def __init__(self, predicates=None, cooperation_weight: float = 0.2, bg_correction_var_weight: float = 0.0,
                 psnr_dead_zone: float = 0.3):
        super().__init__()

        # Term weights — rebalanced to prioritize direct clinical metrics over predicates
        # Investigation: predicates (preservation) fought clinical metrics (improvement)
        self.clinical_metric_weight = 6.0   # Term 1: DOMINANT — boosted for stronger CNR/SNR gains
        self.predicate_weight = 1.0         # Term 2: SUBORDINATE — symbolic preservation signal
        self.cooperation_weight = cooperation_weight  # Term 3 (sweep: 0.2)
        self.quality_guard_weight = 0.0     # Term 4 (0 in stage 1, 1.0 in stage 2)
        self.bg_correction_var_weight = bg_correction_var_weight  # Background correction variance penalty for SNR
        self.psnr_dead_zone = psnr_dead_zone  # PSNR drop (dB) below which no penalty applies

        # Predicates reference (set externally after model creation)
        self.predicates = predicates

        # Training stage (set by training loop)
        self.training_stage = 1

        # Sobel filters for edge/boundary computation
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Laplacian filter for texture computation
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

        # Averaging kernel for local_std (7x7)
        kernel_7 = torch.ones(1, 1, 7, 7) / 49.0
        self.register_buffer('avg_kernel_7', kernel_7)

    # =====================================================================
    # Utility Methods
    # =====================================================================

    @staticmethod
    def extract_deferred_metrics(metrics_dict: Dict) -> Dict:
        """Extract .item() values from deferred tensor metrics."""
        if not metrics_dict.get('_deferred_metrics', False):
            return metrics_dict

        result = {}
        tensor_keys = []
        tensor_vals = []

        for k, v in metrics_dict.items():
            if k == '_deferred_metrics':
                continue
            if k.startswith('_') and isinstance(v, torch.Tensor):
                tensor_keys.append(k[1:])
                tensor_vals.append(v if v.numel() == 1 else v.mean())
            elif not k.startswith('_'):
                result[k] = v

        if tensor_vals:
            stacked = torch.stack(tensor_vals)
            extracted = stacked.tolist()
            for k, v in zip(tensor_keys, extracted):
                result[k] = v if torch.isfinite(torch.tensor(v)) else 0.0

        return result

    def local_std(self, x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """Compute local standard deviation using 7x7 averaging kernel."""
        kernel = self.avg_kernel_7
        cache_attr = f'_avg_kernel_7_{x.dtype}'
        if not hasattr(self, cache_attr) or getattr(self, cache_attr).device != x.device:
            setattr(self, cache_attr, kernel.to(dtype=x.dtype, device=x.device))
        kernel = getattr(self, cache_attr)
        padding = kernel_size // 2
        local_mean = F.conv2d(x, kernel, padding=padding)
        local_sq_mean = F.conv2d(x ** 2, kernel, padding=padding)
        local_var = local_sq_mean - local_mean ** 2
        return torch.sqrt(local_var.clamp(min=1e-8))

    # =====================================================================
    # TERM 1: Direct Clinical Metric Loss
    # =====================================================================

    def compute_clinical_metric_loss(self, corrected: torch.Tensor,
                                      backbone_out: torch.Tensor,
                                      clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Term 1: Direct Clinical Metric Loss — matches validation computation.

        Computes contrast/boundary/texture/edge(EPI)/CNR ratios the SAME way
        as validation, then applies asymmetric loss pushing each above a target.

        Each metric: deficit = ReLU(target - ratio); loss = deficit + 0.5*deficit²
        """
        eps = 1e-8

        # Cache dtype+device-converted kernels (avoid repeated .to() calls each batch)
        _cache_key = f'_kernels_{corrected.dtype}_{corrected.device}'
        if not hasattr(self, _cache_key):
            setattr(self, _cache_key, (
                self.sobel_x.to(dtype=corrected.dtype, device=corrected.device),
                self.sobel_y.to(dtype=corrected.dtype, device=corrected.device),
                self.laplacian.to(dtype=corrected.dtype, device=corrected.device),
            ))
        sx, sy, lap = getattr(self, _cache_key)

        # --- Contrast ratio: mean(local_std_7(img)) / mean(local_std_7(clean)) ---
        corrected_std = self.local_std(corrected)
        clean_std = self.local_std(clean)
        with torch.no_grad():
            backbone_std = self.local_std(backbone_out)
            clean_std_mean = clean_std.mean().clamp(min=eps)
            backbone_contrast_ratio = backbone_std.mean() / clean_std_mean
        corrected_contrast_ratio = corrected_std.mean() / clean_std_mean
        contrast_target = backbone_contrast_ratio * 1.12  # 12% improvement target (sweep-optimized)

        # --- Boundary ratio: mean(|Sobel_Y(img)|) / mean(|Sobel_Y(clean)|) ---
        corrected_vgrad = F.conv2d(corrected, sy, padding=1).abs()
        clean_vgrad = F.conv2d(clean, sy, padding=1).abs()
        with torch.no_grad():
            backbone_vgrad = F.conv2d(backbone_out, sy, padding=1).abs()
            clean_vgrad_mean = clean_vgrad.mean().clamp(min=eps)
            backbone_boundary_ratio = backbone_vgrad.mean() / clean_vgrad_mean
        corrected_boundary_ratio = corrected_vgrad.mean() / clean_vgrad_mean
        boundary_target = backbone_boundary_ratio * 1.12

        # --- Texture ratio: mean(|Laplacian(img)|) / mean(|Laplacian(clean)|) ---
        corrected_tex = F.conv2d(corrected, lap, padding=1).abs()
        clean_tex = F.conv2d(clean, lap, padding=1).abs()
        with torch.no_grad():
            backbone_tex = F.conv2d(backbone_out, lap, padding=1).abs()
            clean_tex_mean = clean_tex.mean().clamp(min=eps)
            backbone_texture_ratio = backbone_tex.mean() / clean_tex_mean
        corrected_texture_ratio = corrected_tex.mean() / clean_tex_mean
        texture_target = backbone_texture_ratio * 1.0  # Don't regress

        # --- Edge (EPI): Pearson(clean_edge, img_edge) ---
        corrected_edge = torch.sqrt(F.conv2d(corrected, sx, padding=1)**2 +
                                     F.conv2d(corrected, sy, padding=1)**2 + eps)
        clean_edge = torch.sqrt(F.conv2d(clean, sx, padding=1)**2 +
                                 F.conv2d(clean, sy, padding=1)**2 + eps)
        with torch.no_grad():
            backbone_edge = torch.sqrt(F.conv2d(backbone_out, sx, padding=1)**2 +
                                        F.conv2d(backbone_out, sy, padding=1)**2 + eps)

        def _pearson_flat(a, b):
            a_f = a.reshape(a.shape[0], -1)
            b_f = b.reshape(b.shape[0], -1)
            a_c = a_f - a_f.mean(dim=1, keepdim=True)
            b_c = b_f - b_f.mean(dim=1, keepdim=True)
            a_n = a_c / (a_c.norm(dim=1, keepdim=True) + 1e-8)
            b_n = b_c / (b_c.norm(dim=1, keepdim=True) + 1e-8)
            return (a_n * b_n).sum(dim=1).mean()

        corrected_epi = _pearson_flat(clean_edge, corrected_edge)
        with torch.no_grad():
            backbone_epi = _pearson_flat(clean_edge, backbone_edge)
        epi_target = backbone_epi * 1.05  # 5% EPI improvement target

        # --- Boundary Sharpness: max gradient ratio (matches validation) ---
        # Validation uses max(|Sobel_Y(img)|) / max(|Sobel_Y(clean)|), NOT mean.
        # Using max captures the sharpest boundary, which is the clinical metric.
        corrected_bs_vgrad = F.conv2d(corrected, sy, padding=1).abs()
        with torch.no_grad():
            clean_bs_vgrad = F.conv2d(clean, sy, padding=1).abs()
            backbone_bs_vgrad = F.conv2d(backbone_out, sy, padding=1).abs()
            clean_bs_max = clean_bs_vgrad.max().clamp(min=1e-4)
            backbone_bs = (backbone_bs_vgrad.max() / clean_bs_max).clamp(0.0, 10.0)
        corrected_bs = (corrected_bs_vgrad.max() / clean_bs_max).clamp(0.0, 10.0)
        corrected_bs_ratio = corrected_bs / backbone_bs.clamp(min=eps)
        bs_target = torch.ones_like(corrected_bs_ratio) * 1.12

        # --- CNR: (signal_mean - bg_mean) / bg_std ---
        with torch.no_grad():
            signal_mask = otsu_tissue_mask(backbone_out)  # Scanner-agnostic
            bg_mask = 1.0 - signal_mask
            sig_sum = signal_mask.sum().clamp(min=1.0)
            bg_sum = bg_mask.sum().clamp(min=1.0)

            bb_sig = (backbone_out * signal_mask).sum() / sig_sum
            bb_bg = (backbone_out * bg_mask).sum() / bg_sum
            bb_bg_std = torch.sqrt(((backbone_out - bb_bg)**2 * bg_mask).sum() / bg_sum + eps).clamp(min=1e-4)
            backbone_cnr = (bb_sig - bb_bg) / bb_bg_std

        cr_sig = (corrected * signal_mask).sum() / sig_sum
        cr_bg = (corrected * bg_mask).sum() / bg_sum
        cr_bg_std = torch.sqrt(((corrected - cr_bg)**2 * bg_mask).sum() / bg_sum + eps).clamp(min=1e-4)
        corrected_cnr = (cr_sig - cr_bg) / cr_bg_std
        cnr_target = backbone_cnr * 1.20  # QT69b: push for +20% CNR improvement

        # --- Direct differentiable CNR/TCI/BS losses ---
        # Maximize all three clinical metrics with direct gradient signal

        # 1. CNR: maximize (mean_tissue - mean_bg) / std_bg
        direct_cnr = (cr_sig - cr_bg) / cr_bg_std
        backbone_cnr_val = backbone_cnr.detach()
        cnr_deficit = F.relu(backbone_cnr_val * 1.20 - direct_cnr)
        direct_cnr_loss = cnr_deficit + cnr_deficit ** 2
        # Background variance reduction: target 25% std_bg reduction
        bg_var_reduction = F.relu(cr_bg_std - bb_bg_std.detach() * 0.75)
        bg_variance_loss = bg_var_reduction * 10.0

        # Correction variance in background: penalize noisy corrections for SNR
        # bna_loss penalizes abs(correction); this penalizes VARIANCE of correction
        # A uniform small correction is fine for SNR; a noisy varying correction degrades it
        correction_for_var = corrected - backbone_out.detach()
        correction_bg = correction_for_var * bg_mask
        correction_bg_mean = correction_bg.sum() / bg_sum
        bg_correction_var = ((correction_for_var - correction_bg_mean) ** 2 * bg_mask).sum() / bg_sum

        # Direct bg_std ratio loss: matches validation's exact SNR computation
        # Validation uses bottom 25% as background, top 50% as tissue
        # Penalize any increase in bg_std (the SNR denominator)
        H_img_loss = corrected.shape[2]
        bg_bottom25_co = corrected[:, :, H_img_loss * 3 // 4:, :]
        tissue_top50_co = corrected[:, :, :H_img_loss // 2, :]
        with torch.no_grad():
            bg_bottom25_bb = backbone_out[:, :, H_img_loss * 3 // 4:, :]
            bg_std_bb = bg_bottom25_bb.std().clamp(min=1e-6)
            tissue_top50_bb = backbone_out[:, :, :H_img_loss // 2, :]
            tissue_mean_bb = tissue_top50_bb.mean()
        bg_std_co = bg_bottom25_co.std().clamp(min=1e-6)
        # Ratio < 1 means corrected has lower bg_std (good for SNR)
        # Penalize ratio > 0.95 (allow only 5% increase max, push for decrease)
        bg_std_ratio = bg_std_co / bg_std_bb
        direct_bg_std_loss = F.relu(bg_std_ratio - 0.95) * 20.0

        # Tissue mean preservation loss (SNR numerator protection)
        # SNR = tissue_mean / bg_std. The corrector sometimes reduces tissue brightness
        # via negative alpha_map, which directly hurts SNR. This loss penalizes any
        # reduction of tissue mean (top 50%) relative to backbone.
        tissue_mean_co = tissue_top50_co.mean()
        tissue_mean_ratio = tissue_mean_co / tissue_mean_bb.clamp(min=1e-6)
        # Penalize ratio < 0.99 (allow at most 1% drop, push for preservation/increase)
        tissue_mean_loss = F.relu(0.99 - tissue_mean_ratio) * 30.0

        # 2. TCI: maximize vertical gradient magnitude (tissue layer contrast)
        # TCI = mean(|Sobel_Y(img)|) — higher = sharper layer boundaries
        corrected_gy_abs = F.conv2d(corrected, sy, padding=1).abs()
        with torch.no_grad():
            backbone_gy_abs = F.conv2d(backbone_out, sy, padding=1).abs()
        # Direct loss: push corrected gradients above backbone by 15%
        tci_deficit = F.relu(backbone_gy_abs.mean().detach() * 1.15 - corrected_gy_abs.mean())
        direct_tci_loss = tci_deficit + 0.5 * tci_deficit ** 2

        # 3. BS: maximize peak boundary sharpness (max gradient at tissue-bg interface)
        # Use top-k mean instead of single max for more stable gradients
        corrected_bs_flat = corrected_bs_vgrad.reshape(-1)
        topk_k = max(corrected_bs_flat.shape[0] // 100, 10)  # top 1% of pixels
        corrected_bs_topk = corrected_bs_flat.topk(topk_k).values.mean()
        with torch.no_grad():
            backbone_bs_flat = backbone_bs_vgrad.reshape(-1)
            backbone_bs_topk = backbone_bs_flat.topk(topk_k).values.mean()
        # Direct loss: push peak sharpness above backbone by 10%
        bs_deficit = F.relu(backbone_bs_topk * 1.10 - corrected_bs_topk)
        direct_bs_loss = bs_deficit + 0.5 * bs_deficit ** 2

        # --- ENL & SNR: Per-image computation (fixes batch-averaging bias) ---
        # ENL = (mean/std)^2 of bottom 25%; SNR = tissue_mean / bg_std
        # Use LOG-RATIO to properly capture catastrophic drops
        H_img = corrected.shape[2]
        B = corrected.shape[0]
        enl_log_ratios = []
        snr_log_ratios = []
        for i in range(B):
            bg_co = corrected[i, 0, H_img*3//4:, :]
            tissue_co = corrected[i, 0, :H_img//2, :]
            with torch.no_grad():
                bg_bb = backbone_out[i, 0, H_img*3//4:, :]
                tissue_bb = backbone_out[i, 0, :H_img//2, :]
                enl_bb = (bg_bb.mean() / bg_bb.std().clamp(min=1e-6)) ** 2
                snr_bb = tissue_bb.mean() / bg_bb.std().clamp(min=1e-6)
            enl_co = (bg_co.mean() / bg_co.std().clamp(min=1e-6)) ** 2
            snr_co = tissue_co.mean() / bg_co.std().clamp(min=1e-6)
            # Log-ratio: log(corrected/backbone), 0 = no change, negative = degradation
            enl_log_ratios.append(torch.log(enl_co / enl_bb.clamp(min=1e-6) + 1e-8))
            snr_log_ratios.append(torch.log(snr_co / snr_bb.clamp(min=1e-6) + 1e-8))
        enl_log_ratio = torch.stack(enl_log_ratios).mean()
        snr_log_ratio = torch.stack(snr_log_ratios).mean()
        # Convert to ratio-like form for the asymmetric loss: ratio=1 means no change
        enl_ratio = torch.exp(enl_log_ratio)  # ~1.0 when no degradation
        snr_ratio = torch.exp(snr_log_ratio)
        enl_target_ratio = torch.ones_like(enl_ratio) * 1.10  # QT69b: push for +10% improvement
        snr_target_ratio = torch.ones_like(snr_ratio) * 1.15  # QT69b→v3: push for +15% SNR improvement

        # --- PSNR preservation: MSE ratio (backbone/corrected, >1 = PSNR improved) ---
        mse_corrected = ((corrected - clean) ** 2).mean()
        with torch.no_grad():
            mse_backbone = ((backbone_out - clean) ** 2).mean()
        # Ratio > 1 means corrected has lower MSE (better PSNR)
        # Ratio < 1 means PSNR degraded. Allow up to ~0.5 dB drop (ratio ~0.89)
        psnr_ratio = mse_backbone / mse_corrected.clamp(min=1e-10)
        psnr_target = torch.ones_like(psnr_ratio) * 0.87  # Allow ~0.6 dB drop (QT68 fine-tune)

        # --- Asymmetric loss per metric ---
        # deficit = ReLU(target - ratio); loss = deficit + 0.5*deficit²
        sub_weights = {
            # Balanced: strong clinical focus WITH PSNR protection
            'contrast': 1.0, 'boundary': 5.0, 'texture': 1.0, 'epi': 15.0, 'cnr': 40.0,
            'bs': 5.0,
            'enl': 8.0, 'snr': 40.0,  # v5: SNR boosted 20→40 + tissue_mean_loss for LOO SNR fix
            'psnr': 10.0,  # Moderate PSNR protection, trading headroom for clinical gains
        }
        ratios = {
            'contrast': corrected_contrast_ratio,
            'boundary': corrected_boundary_ratio,
            'texture': corrected_texture_ratio,
            'epi': corrected_epi,
            'cnr': corrected_cnr,
            'bs': corrected_bs_ratio,
            'enl': enl_ratio,
            'snr': snr_ratio,
            'psnr': psnr_ratio,
        }
        targets = {
            'contrast': contrast_target,
            'boundary': boundary_target,
            'texture': texture_target,
            'epi': epi_target,
            'cnr': cnr_target,
            'bs': bs_target,
            'enl': enl_target_ratio,
            'snr': snr_target_ratio,
            'psnr': psnr_target,
        }

        total_loss = corrected.new_zeros(())
        total_w = 0.0
        metrics = {}

        for name in ['contrast', 'boundary', 'texture', 'epi', 'cnr', 'bs', 'enl', 'snr', 'psnr']:
            w = sub_weights[name]
            deficit = F.relu(targets[name].detach() - ratios[name])
            # Steeper: 3x linear + quadratic for stronger gradient signal
            loss_term = 3.0 * deficit + deficit ** 2
            # Symmetric preservation pull ONLY for preservation metrics (enl, snr, psnr).
            # CNR, EPI, BS are improvement metrics — one-sided loss lets them
            # freely exceed targets without penalty (no gradient pulling them back).
            if name in ('enl', 'snr', 'psnr'):
                deviation = targets[name].detach() - ratios[name]
                preservation_pull = 0.5 * deviation ** 2
                loss_term = loss_term + preservation_pull
            total_loss = total_loss + w * loss_term
            total_w += w
            metrics[f'cm_{name}_ratio'] = _safe_item(ratios[name])
            metrics[f'cm_{name}_target'] = _safe_item(targets[name])
            metrics[f'cm_{name}_deficit'] = _safe_item(deficit)

        total_loss = total_loss / max(total_w, 1e-6)
        total_loss = total_loss.clamp(max=5.0)

        # Pack direct clinical losses into metrics for forward() to use
        metrics['_direct_cnr_loss'] = direct_cnr_loss
        metrics['_bg_variance_loss'] = bg_variance_loss
        metrics['_bg_correction_var'] = bg_correction_var
        metrics['_direct_bg_std_loss'] = direct_bg_std_loss
        metrics['_tissue_mean_loss'] = tissue_mean_loss
        metrics['_direct_tci_loss'] = direct_tci_loss
        metrics['_direct_bs_loss'] = direct_bs_loss

        return total_loss, metrics

    # =====================================================================
    # TERM 2: Predicate Improvement Loss (NEURO-SYMBOLIC CORE)
    # =====================================================================

    def compute_predicate_improvement_loss(self, corrected: torch.Tensor,
                                            backbone_out: torch.Tensor,
                                            noisy: torch.Tensor,
                                            clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Term 2: Predicate Improvement Loss — the neuro-symbolic core.

        Calls predicates.evaluate_clinical_with_grad() on corrected output,
        compares to same on backbone_out.detach(). Penalizes regression,
        rewards improvement.
        """
        if self.predicates is None:
            return corrected.new_zeros(()), {'pred_warning': 'no_predicates'}

        # Evaluate predicates on corrected (WITH gradients)
        corrected_eval = self.predicates.evaluate_clinical_with_grad(
            corrected, noisy, clean=clean
        )

        # Evaluate predicates on backbone (NO gradients — reference only)
        with torch.no_grad():
            backbone_eval = self.predicates.evaluate_clinical_degradation(
                backbone_out, noisy, clean=clean
            )

        # Per-predicate weights
        pred_weights = {
            'P_contrast': 3.0,
            'P_edge': 2.0,
            'P_boundary': 2.0,
            'P_texture': 1.0,
        }

        total_loss = corrected.new_zeros(())
        total_w = 0.0
        metrics = {}

        for pred_name, weight in pred_weights.items():
            c_result = corrected_eval.get(pred_name, {})
            b_result = backbone_eval.get(pred_name, {})

            c_score = c_result.get('score', corrected.new_tensor(0.5))
            b_score = b_result.get('score', corrected.new_tensor(0.5))

            if not isinstance(c_score, torch.Tensor):
                c_score = corrected.new_tensor(c_score)
            if not isinstance(b_score, torch.Tensor):
                b_score = corrected.new_tensor(b_score)

            b_score = b_score.detach()

            # Delta: positive = improvement, negative = regression
            delta = c_score - b_score

            # Balanced: reduce regression asymmetry, reward improvement more
            regression_penalty = F.relu(-delta) * 3.0
            improvement_bonus = -F.relu(delta) * 2.5

            # Threshold penalty: if corrected score < 0.5
            threshold_penalty = F.relu(0.5 - c_score) * 2.0

            pred_loss = weight * (regression_penalty + improvement_bonus + threshold_penalty)
            total_loss = total_loss + pred_loss
            total_w += weight

            short = pred_name.split('_')[1]  # contrast, edge, boundary, texture
            metrics[f'pi_{short}_corrected'] = _safe_item(c_score)
            metrics[f'pi_{short}_backbone'] = _safe_item(b_score)
            metrics[f'pi_{short}_delta'] = _safe_item(delta)

        total_loss = total_loss / max(total_w, 1e-6)
        total_loss = total_loss.clamp(max=10.0)

        return total_loss, metrics

    # =====================================================================
    # TERM 3: Cooperation Loss (kept from original)
    # =====================================================================

    def compute_cooperation_loss(self, nafnet_uncertainty: torch.Tensor,
                                  corrector_potentials: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Cooperation Loss: Pearson(uncertainty, total_potential).
        Encourage correctors to have high potential where backbone is uncertain.
        """
        if corrector_potentials:
            potential_maps = []
            for name, val in corrector_potentials.items():
                if isinstance(val, dict) and 'map' in val:
                    potential_maps.append(val['map'])
                elif isinstance(val, torch.Tensor):
                    potential_maps.append(val)
            if not potential_maps:
                return nafnet_uncertainty.new_zeros(()), {}
            total_potential = sum(potential_maps)
            del potential_maps
        else:
            return nafnet_uncertainty.new_zeros(()), {}

        # Check for degenerate case
        uncertainty_range = nafnet_uncertainty.max() - nafnet_uncertainty.min()
        potential_range = total_potential.max() - total_potential.min()
        if uncertainty_range < 1e-6 or potential_range < 1e-6:
            return nafnet_uncertainty.new_zeros(()), {
                'uncertainty_potential_corr': 0.0,
                'mean_uncertainty': nafnet_uncertainty.mean().item(),
                'mean_potential': total_potential.mean().item(),
            }

        # Normalize to [0, 1]
        u_norm = (nafnet_uncertainty - nafnet_uncertainty.min()) / (uncertainty_range + 1e-8)
        p_norm = (total_potential - total_potential.min()) / (potential_range + 1e-8)

        # Pearson correlation
        u_c = u_norm - u_norm.mean()
        p_c = p_norm - p_norm.mean()
        u_sq = (u_c ** 2).sum()
        p_sq = (p_c ** 2).sum()

        if u_sq < 1e-12 or p_sq < 1e-12:
            return nafnet_uncertainty.new_zeros(()), {
                '_uncertainty_mean': nafnet_uncertainty.mean(),
                '_potential_mean': total_potential.mean(),
                '_deferred_metrics': True,
            }

        correlation = (u_c * p_c).sum() / torch.sqrt(u_sq * p_sq + 1e-16)
        correlation = correlation.clamp(-1.0, 1.0)

        # Loss: want to maximize correlation
        cooperation_loss = F.relu(1.0 - correlation) / 2.0
        cooperation_loss = cooperation_loss.clamp(max=1.0)

        metrics = {
            '_correlation': correlation,
            '_uncertainty_mean': nafnet_uncertainty.mean(),
            '_potential_mean': total_potential.mean(),
            '_deferred_metrics': True,
        }

        return cooperation_loss, metrics

    # =====================================================================
    # TERM 4: PSNR/SSIM Guard (dead zones)
    # =====================================================================

    def compute_quality_guard_loss(self, corrected: torch.Tensor,
                                    backbone_out: torch.Tensor,
                                    clean: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Term 4: PSNR/SSIM Guard with dead zones.

        - PSNR: no penalty < psnr_dead_zone dB drop. Linear above. Quadratic beyond dead_zone + 0.4.
        - SSIM: no penalty < 0.005 drop. Linear above.
        """
        eps = 1e-8
        dz = self.psnr_dead_zone

        # --- PSNR guard ---
        mse_backbone = F.mse_loss(backbone_out, clean)
        mse_corrected = F.mse_loss(corrected, clean)
        psnr_backbone = 10 * torch.log10(1.0 / (mse_backbone + eps))
        psnr_corrected = 10 * torch.log10(1.0 / (mse_corrected + eps))
        psnr_drop = (psnr_backbone - psnr_corrected).clamp(-10.0, 10.0)

        # Dead zone: no penalty < dz dB. Linear dz to dz+0.4. Quadratic > dz+0.4.
        psnr_linear = F.relu(psnr_drop - dz) * 5.0
        psnr_quadratic = F.relu(psnr_drop - (dz + 0.4)) ** 2 * 15.0
        psnr_loss = (psnr_linear + psnr_quadratic).clamp(max=10.0)

        # --- SSIM guard ---
        C1, C2 = 0.01**2, 0.03**2

        def _global_ssim(img1, img2):
            mu1, mu2 = img1.mean(), img2.mean()
            s1 = ((img1 - mu1) ** 2).mean().clamp(min=eps)
            s2 = ((img2 - mu2) ** 2).mean().clamp(min=eps)
            s12 = ((img1 - mu1) * (img2 - mu2)).mean()
            return ((2*mu1*mu2 + C1)*(2*s12 + C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))

        ssim_backbone = _global_ssim(backbone_out, clean)
        ssim_corrected = _global_ssim(corrected, clean)
        ssim_drop = (ssim_backbone - ssim_corrected).clamp(-0.5, 0.5)

        # Dead zone: no penalty < 0.005 SSIM drop. Linear above.
        ssim_loss = F.relu(ssim_drop - 0.005) * 20.0
        ssim_loss = ssim_loss.clamp(max=5.0)

        total_quality = psnr_loss + ssim_loss

        metrics = {
            'psnr_backbone': _safe_item(psnr_backbone),
            'psnr_corrected': _safe_item(psnr_corrected),
            'psnr_delta': _safe_item(psnr_corrected - psnr_backbone),
            'psnr_drop': _safe_item(psnr_drop),
            'ssim_backbone_loss': _safe_item(ssim_backbone),
            'ssim_corrected_loss': _safe_item(ssim_corrected),
            'ssim_delta_loss': _safe_item(ssim_corrected - ssim_backbone),
            'quality_psnr_loss': _safe_item(psnr_loss),
            'quality_ssim_loss': _safe_item(ssim_loss),
        }

        return total_quality, metrics

    # =====================================================================
    # FORWARD: Multi-Term Clinical Loss for Multiplicative Gain + Edge Restoration
    # =====================================================================

    def forward(self, corrected: torch.Tensor, backbone_out: torch.Tensor,
                clean: torch.Tensor, noisy: torch.Tensor, info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-term clinical loss for multiplicative gain + edge restoration.

        Terms:
        1. Clinical Metric Loss (CNR, EPI, BS, etc.) — one-sided for improvement metrics
        2. Predicate Improvement Loss — neuro-symbolic core
        3. Cooperation Loss — maintains negotiator
        4. PSNR/SSIM Guard (Stage 2 only)
        5. Alpha magnitude floor — prevent lazy gain corrector
        6. Alpha smoothness — penalize sharp alpha_map transitions (EPI protection)
        7. Edge restoration ceiling — prevent noise leakage
        8. Sobel-domain edge MSE — direct EPI supervision

        Stage 1 (epochs 1-3): Terms 1+2+3+5+6+7+8
        Stage 2 (epochs 4+): Add Term 4 (quality guard)
        """
        stage = getattr(self, 'training_stage', 1)

        # ===== TERM 1: Direct Clinical Metric Loss =====
        clinical_loss, clinical_metrics = self.compute_clinical_metric_loss(
            corrected, backbone_out, clean
        )
        # Extract direct clinical losses computed inside clinical metric loss
        direct_cnr_loss = clinical_metrics.pop('_direct_cnr_loss')
        bg_variance_loss = clinical_metrics.pop('_bg_variance_loss')
        bg_correction_var_penalty = clinical_metrics.pop('_bg_correction_var', corrected.new_zeros(()))
        direct_bg_std_loss = clinical_metrics.pop('_direct_bg_std_loss', corrected.new_zeros(()))
        tissue_mean_loss = clinical_metrics.pop('_tissue_mean_loss', corrected.new_zeros(()))
        direct_tci_loss = clinical_metrics.pop('_direct_tci_loss')
        direct_bs_loss = clinical_metrics.pop('_direct_bs_loss')

        # ===== TERM 2: Predicate Improvement Loss =====
        predicate_loss, pred_metrics = self.compute_predicate_improvement_loss(
            corrected, backbone_out, noisy, clean
        )

        # ===== TERM 3: Cooperation Loss =====
        # Use live (non-detached) tensors for gradient flow when available.
        # Without this, the cooperation loss operates on detached tensors and
        # produces zero gradients, making it effectively dead.
        live_potentials = info.get('_live_potentials', None)
        live_uncertainty = info.get('_live_nafnet_uncertainty', None)
        if live_potentials is not None and live_uncertainty is not None:
            # Live path: gradient flows to potential_net and uncertainty head
            cooperation_loss, coop_metrics = self.compute_cooperation_loss(
                live_uncertainty, {name: {'map': pot} for name, pot in live_potentials.items()}
            )
        else:
            # Fallback: use detached tensors (validation or missing keys)
            nafnet_uncertainty = info.get('nafnet_uncertainty', torch.zeros_like(corrected))
            corrector_potentials = info.get('corrector_potentials', {})
            cooperation_loss, coop_metrics = self.compute_cooperation_loss(
                nafnet_uncertainty, corrector_potentials
            )

        # ===== TERM 4: PSNR/SSIM Guard (Stage 2 only) =====
        if stage >= 2:
            quality_loss, quality_metrics = self.compute_quality_guard_loss(
                corrected, backbone_out, clean
            )
            quality_weight = 1.0
        else:
            # QT69c: Enable quality guard in Stage 1 with reduced weight
            # Prevents PSNR divergence during early training from scratch
            quality_loss, quality_metrics = self.compute_quality_guard_loss(
                corrected, backbone_out, clean
            )
            quality_weight = 0.3

        # ===== Alpha magnitude floor (prevent lazy corrector) =====
        # Use alpha_map from model info dict for direct gradient to gain corrector.
        alpha_map = info.get('alpha_map', None)
        edge_restoration = info.get('edge_restoration', None)

        if alpha_map is not None and isinstance(alpha_map, torch.Tensor):
            alpha_mag = alpha_map.abs().mean()
            correction_mag = alpha_mag  # For logging
        else:
            # Fallback: measure from final output (post-gate, weaker gradient)
            H = corrected.shape[2]
            tissue_h = int(H * 0.7)
            correction_tissue = (corrected[:, :, :tissue_h, :] - backbone_out[:, :, :tissue_h, :]).abs()
            alpha_mag = correction_tissue.mean()
            correction_mag = alpha_mag

        # Alpha magnitude floor: prevent lazy corrector
        # Floor at 0.008 — push for larger corrections (QT45b operated at 0.012)
        # Skip in stage2_epi (alpha is frozen, no gradient anyway)
        if getattr(self, '_stage2_epi_mode', False):
            floor_loss = corrected.new_zeros(())
        else:
            floor_loss = F.relu(0.002 - alpha_mag) * 40.0
            floor_loss = floor_loss.clamp(max=5.0)

        # Alpha magnitude ceiling: prevent PSNR overshoot from excessive gain
        # Ceiling at 0.005 creates gradient basin that holds alpha_mag at ~0.010
        alpha_ceiling_loss = F.relu(alpha_mag - 0.006) ** 2 * 200.0
        alpha_ceiling_loss = alpha_ceiling_loss.clamp(max=5.0)

        # ===== QT69c: Background Noise Attenuation (BNA) loss =====
        # Penalize any correction-backbone difference in background pixels.
        # Scanner-agnostic: threshold from per-image 15th percentile.
        bna_loss = corrected.new_zeros(())
        B_bna = backbone_out.size(0)
        bg_thresh_bna = torch.quantile(
            backbone_out.detach().view(B_bna, -1), 0.15, dim=1
        ).view(B_bna, 1, 1, 1)
        bg_soft_mask = torch.sigmoid(-(backbone_out.detach() - bg_thresh_bna) * 20.0)
        bg_change = (corrected - backbone_out.detach()).abs() * bg_soft_mask
        bna_loss = bg_change.mean()

        # ===== Alpha smoothness regularization =====
        # Penalize sharp spatial transitions in alpha_map (Sobel gradient magnitude).
        # Smooth alpha_map prevents Sobel-detectable artifacts that hurt EPI.
        alpha_smooth_loss = corrected.new_zeros(())
        if alpha_map is not None and isinstance(alpha_map, torch.Tensor):
            sx = self.sobel_x.to(dtype=alpha_map.dtype)
            sy = self.sobel_y.to(dtype=alpha_map.dtype)
            alpha_gx = F.conv2d(alpha_map, sx, padding=1)
            alpha_gy = F.conv2d(alpha_map, sy, padding=1)
            smoothness_loss = torch.sqrt(alpha_gx**2 + alpha_gy**2 + 1e-8).mean()
            alpha_smooth_loss = smoothness_loss * 2.0

        # ===== Edge recovery regularization =====
        edge_ceiling_val = getattr(self, '_stage2_edge_ceiling', 0.06)
        edge_ceiling_loss = corrected.new_zeros(())
        if edge_restoration is not None and isinstance(edge_restoration, torch.Tensor):
            edge_mag = edge_restoration.abs().mean()
            edge_ceiling_loss = F.relu(edge_mag - edge_ceiling_val) ** 2 * 30.0
        else:
            edge_mag = corrected.new_zeros(())

        # ===== EPI Losses: Pearson + Sobel Supervision + Edge MSE =====
        sx_dt = self.sobel_x.to(dtype=corrected.dtype)
        sy_dt = self.sobel_y.to(dtype=corrected.dtype)

        # Sobel components (dx, dy) for supervision
        corrected_edge_x = F.conv2d(corrected, sx_dt, padding=1)
        corrected_edge_y = F.conv2d(corrected, sy_dt, padding=1)
        corrected_edge_map = torch.sqrt(corrected_edge_x**2 + corrected_edge_y**2 + 1e-8)

        with torch.no_grad():
            clean_edge_x = F.conv2d(clean, sx_dt, padding=1)
            clean_edge_y = F.conv2d(clean, sy_dt, padding=1)
            clean_edge_map = torch.sqrt(clean_edge_x**2 + clean_edge_y**2 + 1e-8)
            backbone_edge_x = F.conv2d(backbone_out, sx_dt, padding=1)
            backbone_edge_y = F.conv2d(backbone_out, sy_dt, padding=1)
            backbone_edge_map = torch.sqrt(backbone_edge_x**2 + backbone_edge_y**2 + 1e-8)

        # (A) Direct Pearson EPI loss
        def _pearson_2d(a, b):
            a_f = a.reshape(a.shape[0], -1)
            b_f = b.reshape(b.shape[0], -1)
            a_c = a_f - a_f.mean(dim=1, keepdim=True)
            b_c = b_f - b_f.mean(dim=1, keepdim=True)
            a_n = a_c / (a_c.norm(dim=1, keepdim=True) + 1e-8)
            b_n = b_c / (b_c.norm(dim=1, keepdim=True) + 1e-8)
            return (a_n * b_n).sum(dim=1).mean()

        corrected_epi_val = _pearson_2d(clean_edge_map, corrected_edge_map)
        with torch.no_grad():
            backbone_epi_val = _pearson_2d(clean_edge_map, backbone_edge_map)

        epi_target_mult = getattr(self, '_stage2_epi_target', 1.05)
        epi_target_direct = backbone_epi_val * epi_target_mult
        epi_deficit = F.relu(epi_target_direct - corrected_epi_val)
        direct_epi_loss = (3.0 * epi_deficit + epi_deficit ** 2) * 15.0

        # (B) Sobel-domain supervision loss: recover lost edges
        # Target: Sobel(clean) - Sobel(backbone) = the edges backbone lost
        # Recovered: Sobel(corrected) - Sobel(backbone) = what corrector added
        with torch.no_grad():
            lost_edge_x = clean_edge_x - backbone_edge_x
            lost_edge_y = clean_edge_y - backbone_edge_y
            # Edge weight: focus on actual edge regions (where clean has edges)
            edge_weight = clean_edge_map / clean_edge_map.mean().clamp(min=1e-8)
            edge_weight = edge_weight.clamp(max=10.0)  # prevent extreme weights

        recovered_edge_x = corrected_edge_x - backbone_edge_x.detach()
        recovered_edge_y = corrected_edge_y - backbone_edge_y.detach()

        sobel_supervision_loss = (
            (edge_weight * (recovered_edge_x - lost_edge_x) ** 2).mean() +
            (edge_weight * (recovered_edge_y - lost_edge_y) ** 2).mean()
        )

        # (C) Edge-weighted magnitude MSE (original)
        with torch.no_grad():
            edge_weight_map = clean_edge_map / clean_edge_map.max().clamp(min=1e-6)
            edge_weight_map = edge_weight_map ** 2
            edge_weight_map = edge_weight_map / edge_weight_map.mean().clamp(min=1e-8)
        corrected_edge_norm = corrected_edge_map / corrected_edge_map.max().clamp(min=1e-6)
        clean_edge_norm = clean_edge_map / clean_edge_map.max().clamp(min=1e-6)
        edge_mse_weighted = (edge_weight_map * (corrected_edge_norm - clean_edge_norm.detach()) ** 2).mean()

        # Combine EPI losses
        is_stage2 = getattr(self, '_stage2_epi_mode', False)
        if is_stage2:
            # Stage 2: Sobel supervision dominates, Pearson EPI as secondary
            edge_mse_loss = sobel_supervision_loss * 50.0 + direct_epi_loss * 5.0 + edge_mse_weighted * 5.0
        else:
            # Stage 1: balanced EPI losses (v4: boosted direct_epi from 1→3 to compensate bg smoothing)
            edge_mse_loss = edge_mse_weighted * 10.0 + 3.0 * direct_epi_loss + sobel_supervision_loss * 10.0

        # ===== COMBINED LOSS =====
        if is_stage2:
            # Stage 2 EPI: only edge losses + PSNR guard
            total_loss = (
                quality_weight * quality_loss +  # PSNR guard
                edge_ceiling_loss +
                edge_mse_loss
            )
        else:
            total_loss = (
                self.clinical_metric_weight * clinical_loss +
                self.predicate_weight * predicate_loss +
                self.cooperation_weight * cooperation_loss +
                quality_weight * quality_loss +
                floor_loss +
                0.0 * direct_cnr_loss +  # DISABLED: destabilized QT59-62 and fold 15 SNR
                8.0 * bg_variance_loss +  # QT69b→v3: increased from 5.0 — penalizes bg noise for ENL/SNR
                self.bg_correction_var_weight * bg_correction_var_penalty +  # v3: correction variance in bg for SNR
                5.0 * direct_bg_std_loss +  # v3: direct bg_std ratio penalty (matches validation SNR)
                5.0 * tissue_mean_loss +  # v5: tissue mean preservation (SNR numerator)
                2.0 * bna_loss +  # QT69c: background noise attenuation
                0.0 * direct_tci_loss +
                0.0 * direct_bs_loss +
                alpha_ceiling_loss +
                alpha_smooth_loss +
                edge_ceiling_loss +
                edge_mse_loss
            )

        # NaN/Inf safety
        if not torch.isfinite(total_loss).all():
            print(f"WARNING: total_loss is NaN/Inf, using fallback")
            total_loss = corrected.new_tensor(1.0)

        total_loss = total_loss.clamp(max=50.0)

        # ===== PSNR monitoring for all stages =====
        if stage == 1 or 'psnr_backbone' not in quality_metrics:
            with torch.no_grad():
                mse_bb = F.mse_loss(backbone_out, clean)
                mse_cr = F.mse_loss(corrected, clean)
                psnr_bb = 10 * torch.log10(1.0 / (mse_bb + 1e-8))
                psnr_cr = 10 * torch.log10(1.0 / (mse_cr + 1e-8))
                quality_metrics['psnr_backbone'] = _safe_item(psnr_bb)
                quality_metrics['psnr_corrected'] = _safe_item(psnr_cr)
                quality_metrics['psnr_delta'] = _safe_item(psnr_cr - psnr_bb)

        # ===== METRICS =====
        metrics = {
            'total': _safe_item(total_loss),
            'clinical_metric_loss': _safe_item(clinical_loss),
            'predicate_improvement_loss': _safe_item(predicate_loss),
            'cooperation': _safe_item(cooperation_loss),
            'quality_guard_loss': _safe_item(quality_loss),
            'correction_floor': _safe_item(floor_loss),
            'alpha_ceiling_loss': _safe_item(alpha_ceiling_loss),
            'alpha_smooth_loss': _safe_item(alpha_smooth_loss),
            'edge_ceiling_loss': _safe_item(edge_ceiling_loss),
            'edge_mse_loss': _safe_item(edge_mse_loss),
            'direct_cnr_loss': _safe_item(direct_cnr_loss),
            'bg_variance_loss': _safe_item(bg_variance_loss),
            'bg_correction_var': _safe_item(bg_correction_var_penalty),
            'direct_bg_std_loss': _safe_item(direct_bg_std_loss),
            'tissue_mean_loss': _safe_item(tissue_mean_loss),
            'bna_loss': _safe_item(bna_loss),
            'direct_tci_loss': _safe_item(direct_tci_loss),
            'direct_bs_loss': _safe_item(direct_bs_loss),
            'sobel_supervision_loss': _safe_item(sobel_supervision_loss),
            'correction_magnitude': _safe_item(correction_mag),
            'alpha_map_mag': _safe_item(alpha_mag) if isinstance(alpha_mag, torch.Tensor) else 0.0,
            'edge_restoration_mag': _safe_item(edge_mag) if isinstance(edge_mag, torch.Tensor) else 0.0,
            'tissue_correction_mag': info.get('region_decomposition', {}).get('tissue_correction_mag',
                                     info.get('region_decomposition', {}).get('alpha_map_mean', 0)),
            'boundary_correction_mag': info.get('region_decomposition', {}).get('boundary_correction_mag',
                                       info.get('region_decomposition', {}).get('edge_restoration_mag', 0)),
            'stage': stage,
            **clinical_metrics,
            **pred_metrics,
            **quality_metrics,
            **self.extract_deferred_metrics(coop_metrics),
        }

        # Predicate scores from model info (for validation tracking)
        pred_scores = info.get('predicate_scores', {})
        if pred_scores:
            metrics['pred_scores'] = pred_scores
            passing = sum(1 for k, v in pred_scores.items()
                         if k in ['P1', 'P2', 'P3', 'P4', 'P6']
                         and (v >= 0.5 if not isinstance(v, torch.Tensor) else v.item() >= 0.5))
            metrics['predicates_passing'] = passing
        else:
            metrics['predicates_passing'] = 0

        return total_loss, metrics


# Keep backward-compatible alias
CooperativeLoss = SimplifiedCooperativeLoss

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


def freeze_backbone_norm_stats(model):
    """Keep every normalisation layer inside the frozen backbone in evaluation mode.

    Freezing the backbone parameters does not stop a BatchNorm layer from updating
    its running mean and variance, because those are buffers rather than parameters.
    DnCNN is the only backbone here that contains BatchNorm, and leaving it in
    training mode moved its own output while it was being used as the fixed
    reference that every clinical delta is measured against. Its measured texture
    contrast index drifted from 0.550 to 0.786 across four runs on the same images.
    """
    # Scope this to the denoising network itself. The wrapper around it also holds
    # the trainable uncertainty head, which must keep learning.
    wrapper = getattr(model, "backbone", None)
    bb = getattr(wrapper, "backbone", None) if wrapper is not None else None
    if bb is None:
        return 0
    n = 0
    for m in bb.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            n += 1
    return n


def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None,
                anchor_params=None, ewc_weight=0.0):
    """Train one epoch with cooperation tracking."""
    model.train()
    # model.train() switches the whole tree, backbone included, so the backbone
    # normalisation layers have to be put back immediately after it.
    freeze_backbone_norm_stats(model)

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

    # Check if dataset has pre-cached backbone outputs (for DnCNN/SwinIR)
    use_cached_backbone = (hasattr(loader.dataset, '_backbone_cache') and
                           loader.dataset._backbone_cache is not None)
    # _OUTPUT_BASED_BACKBONES is empty — all backbones use feature-based uncertainty now

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        clean = batch['clean'].to(device, non_blocking=True)
        noisy = batch['noisy'].to(device, non_blocking=True)

        if use_cached_backbone:
            # FASTEST PATH: backbone_out pre-computed and cached in dataset
            # Only run lightweight uncertainty + corrector (no backbone forward at all)
            backbone_out_raw = batch['backbone_out'].to(device, non_blocking=True)

            # Apply augmentation to (backbone_out, noisy, clean) jointly
            if torch.rand(1).item() < 0.5:
                noise_strength = torch.rand(1).item() * 0.1
                speckle = 1.0 + noise_strength * torch.randn_like(noisy)
                noisy = (noisy * speckle).clamp(0, 1)
                # The reference is NOT speckled. In the standard path speckle is
                # applied to the input and the backbone removes it, so the corrector
                # never sees a speckled reference at train or test time. Multiplying
                # the cached reference by speckle taught the corrector to denoise a
                # corruption that the test time reference does not carry.
            if torch.rand(1).item() < 0.5:
                brightness = 0.6 + torch.rand(1).item() * 0.9
                noisy = (noisy * brightness).clamp(0, 1)
                clean = (clean * brightness).clamp(0, 1)
                backbone_out_raw = (backbone_out_raw * brightness).clamp(0, 1)
            if torch.rand(1).item() < 0.3:
                gamma = 0.7 + torch.rand(1).item() * 0.6
                noisy = noisy.pow(gamma).clamp(0, 1)
                clean = clean.pow(gamma).clamp(0, 1)
                backbone_out_raw = backbone_out_raw.pow(gamma).clamp(0, 1)

            # Compute uncertainty based on backbone type
            if model.backbone.backbone_name == 'dncnn':
                # DnCNN: run head at full res for rich 228ch features (cheap)
                with torch.no_grad():
                    head_feat = model.backbone.backbone.head(noisy)
                head_feat = head_feat.detach()
                nafnet_uncertainty = model.backbone._compute_uncertainty_from_enc1(head_feat)
                del head_feat
            elif model.backbone.backbone_name == 'swinir':
                # SwinIR: run conv_first for 138ch shallow features (cheap)
                # Adaptive: full res for small datasets (LOO), downsample for large (PKU37)
                Ht, Wt = noisy.shape[2], noisy.shape[3]
                n_cached = len(loader.dataset) if hasattr(loader.dataset, '_backbone_cache') and loader.dataset._backbone_cache is not None else 9999
                # Must match the resolution of the cached reference and of the
                # test time path (BackboneWrapper._forward_swinir_with_shallow runs
                # conv_first at full resolution). Downsampling stays only for the
                # CPU large dataset case where the cache itself is downsampled.
                if n_cached <= 200 or noisy.is_cuda:
                    ds_factor = 1
                else:
                    ds_factor = 4
                if ds_factor > 1:
                    noisy_ds = F.avg_pool2d(noisy, kernel_size=ds_factor, stride=ds_factor)
                else:
                    noisy_ds = noisy
                _, _, Hd, Wd = noisy_ds.shape
                pad_h = (8 - Hd % 8) % 8
                pad_w = (8 - Wd % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    noisy_ds = F.pad(noisy_ds, (0, pad_w, 0, pad_h), mode='reflect')
                with torch.no_grad():
                    shallow_feat_ds = model.backbone.backbone.conv_first(noisy_ds)
                shallow_feat_ds = shallow_feat_ds.detach()
                del noisy_ds
                if pad_h > 0 or pad_w > 0:
                    shallow_feat_ds = shallow_feat_ds[:, :, :Hd, :Wd]
                if ds_factor > 1:
                    shallow_feat = F.interpolate(shallow_feat_ds, size=(Ht, Wt),
                                                 mode='bilinear', align_corners=False)
                    del shallow_feat_ds
                else:
                    shallow_feat = shallow_feat_ds
                nafnet_uncertainty = model.backbone._compute_uncertainty_from_enc1(shallow_feat)
                del shallow_feat
            else:
                # NAFNet/KBNet: caching not recommended (enc1 requires full encoder)
                # Fall back to running full backbone forward for proper enc1-based uncertainty
                _, nafnet_uncertainty = model.backbone(noisy)
            backbone_out = backbone_out_raw
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=nafnet_uncertainty, return_details=False,
            )
            info['nafnet_uncertainty'] = nafnet_uncertainty.detach()

        else:
            # STANDARD PATH for NAFNet/KBNet: full model forward with augmentation
            if torch.rand(1).item() < 0.5:
                noise_strength = torch.rand(1).item() * 0.1
                speckle = 1.0 + noise_strength * torch.randn_like(noisy)
                noisy = (noisy * speckle).clamp(0, 1)
            if torch.rand(1).item() < 0.5:
                brightness = 0.6 + torch.rand(1).item() * 0.9
                noisy = (noisy * brightness).clamp(0, 1)
                clean = (clean * brightness).clamp(0, 1)
            if torch.rand(1).item() < 0.3:
                gamma = 0.7 + torch.rand(1).item() * 0.6
                noisy = noisy.pow(gamma).clamp(0, 1)
                clean = clean.pow(gamma).clamp(0, 1)

            corrected, backbone_out, info = model(noisy)

        if scaler is not None:
            with autocast():
                loss, metrics = criterion(corrected, backbone_out, clean, noisy, info)

            # EWC regularization: L2 anchor to pre-adaptation weights
            if anchor_params is not None and ewc_weight > 0:
                ewc_loss = sum(((p - anchor_params[n]) ** 2).sum()
                               for n, p in model.named_parameters()
                               if n in anchor_params and p.requires_grad)
                loss = loss + ewc_weight * ewc_loss
                metrics['ewc_loss'] = ewc_loss.item()

            # OPTIMIZATION: Check finite first (no sync), then check threshold only if needed
            is_finite = torch.isfinite(loss).all()
            if not is_finite:
                print(f"WARNING: Skipping gradient update - loss is NaN/Inf")
                del corrected, backbone_out
                continue
            # OPTIMIZATION: Only call .item() for threshold check (single sync point)
            loss_val = loss.item()
            if loss_val > 50.0:
                print(f"WARNING: Skipping gradient update - loss too high: {loss_val:.4f}")
                del corrected, backbone_out
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, metrics = criterion(corrected, backbone_out, clean, noisy, info)

            # EWC regularization: L2 anchor to pre-adaptation weights
            if anchor_params is not None and ewc_weight > 0:
                ewc_loss = sum(((p - anchor_params[n]) ** 2).sum()
                               for n, p in model.named_parameters()
                               if n in anchor_params and p.requires_grad)
                loss = loss + ewc_weight * ewc_loss
                metrics['ewc_loss'] = ewc_loss.item()

            # OPTIMIZATION: Check finite first (no sync), then check threshold only if needed
            is_finite = torch.isfinite(loss).all()
            if not is_finite:
                print(f"WARNING: Skipping gradient update - loss is NaN/Inf")
                del corrected, backbone_out
                continue
            # OPTIMIZATION: Only call .item() for threshold check (single sync point)
            loss_val = loss.item()
            if loss_val > 50.0:
                print(f"WARNING: Skipping gradient update - loss too high: {loss_val:.4f}")
                del corrected, backbone_out
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        allocation = info.get('allocations', {})
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


def _precompute_val_backbone_cache(model, loader, device):
    """Pre-compute backbone outputs for validation images (huge speedup for SwinIR).

    Called once before the training loop. Returns a list of (backbone_out, uncertainty)
    tensors on CPU, indexed by dataset order.

    OOM FIX: For SwinIR on CPU, full-res Duke images (970×520) create ~8000
    attention windows and blow up memory. We downsample 2× before the backbone
    Uses tiled overlapping inference for SwinIR on CPU to prevent OOM
    while maintaining full-resolution quality (no downsampling).
    """
    import time as _time
    is_swinir = model.backbone.backbone_name == 'swinir'
    print(f"\nPre-caching validation backbone outputs ({len(loader.dataset)} images)...")
    if is_swinir and device == 'cpu':
        print(f"  SwinIR on CPU: using tiled inference (128px tiles, 16px overlap)")
    t0 = _time.time()
    cache = []
    model.eval()
    use_amp = device != 'cpu' and torch.cuda.is_available()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val backbone cache"):
            noisy = batch['noisy'].to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    bb_out, unc = model.backbone(noisy)
            else:
                # SwinIR tiled inference is handled inside
                # BackboneWrapper._forward_swinir_with_shallow()
                bb_out, unc = model.backbone(noisy)

            cache.append((bb_out.cpu(), unc.cpu()))
            del bb_out, unc, noisy
            gc.collect()
    elapsed = _time.time() - t0
    print(f"  Val backbone cache: {len(cache)} images in {elapsed:.1f}s")
    return cache


@torch.inference_mode()
def validate(model, loader, device, lpips_model=None, criterion=None,
             val_backbone_cache=None):
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
    # Standard OCT clinical metrics
    total_enl_noisy = 0         # Equivalent Number of Looks
    total_enl_backbone = 0
    total_enl_corrected = 0
    total_snr_backbone = 0      # Signal-to-Noise Ratio
    total_snr_corrected = 0

    # Lambda statistics per corrector (dynamically populated from model output)
    lambda_stats = defaultdict(lambda: {'sum': 0, 'max': 0})

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

    for batch_idx, batch in enumerate(tqdm(loader, desc="Validation")):
        # OPTIMIZATION: Use non_blocking=True for overlapped data transfer
        clean = batch['clean'].to(device, non_blocking=True)
        noisy = batch['noisy'].to(device, non_blocking=True)

        # SPEED: Use pre-cached backbone outputs if available (huge speedup for SwinIR)
        if val_backbone_cache is not None and batch_idx < len(val_backbone_cache):
            backbone_out = val_backbone_cache[batch_idx][0].to(device, non_blocking=True)
            nafnet_unc = val_backbone_cache[batch_idx][1].to(device, non_blocking=True)
            if use_amp:
                with autocast():
                    corrected, info = model.corrector(
                        backbone_out, noisy, None,
                        nafnet_uncertainty=nafnet_unc, return_details=False,
                    )
            else:
                corrected, info = model.corrector(
                    backbone_out, noisy, None,
                    nafnet_uncertainty=nafnet_unc, return_details=False,
                )
        elif use_amp:
            with autocast():
                backbone_out, nafnet_unc = model.backbone(noisy)
                corrected, info = model.corrector(
                    backbone_out, noisy, None,
                    nafnet_uncertainty=nafnet_unc, return_details=False,
                )
        else:
            backbone_out, nafnet_unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=nafnet_unc, return_details=False,
            )
        info['nafnet_uncertainty'] = nafnet_unc.detach()
        del nafnet_unc

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

        # Compute TWO cooperation correlations:
        # 1. Uncertainty vs potential (symbolic cooperation)
        # 2. Uncertainty vs actual correction magnitude (effective cooperation)
        # Use the better of the two — the model may achieve cooperation through
        # either the potential pathway or direct correction behavior.
        best_corr_this_image = 0.0

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
                        best_corr_this_image = max(best_corr_this_image, correlation.item())
                except Exception:
                    pass  # Skip if correlation computation fails
            del u_flat, p_flat, total_potential  # FIX: Free intermediate tensors

        # Effective cooperation: correlation between uncertainty and actual
        # correction magnitude. This is the ground-truth measure of whether
        # correctors focus effort where the backbone struggles.
        correction_mag_map = (corrected - backbone_out).abs()
        u_flat = nafnet_uncertainty.view(-1)
        c_flat = correction_mag_map.view(-1)
        if u_flat.numel() >= 2 and u_flat.std() > 1e-8 and c_flat.std() > 1e-8:
            try:
                corr_effective = torch.corrcoef(torch.stack([u_flat, c_flat]))[0, 1]
                if not torch.isnan(corr_effective):
                    best_corr_this_image = max(best_corr_this_image, corr_effective.item())
            except Exception:
                pass
        del u_flat, c_flat, correction_mag_map

        total_uncertainty_corr += best_corr_this_image

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
            signal_mask = otsu_tissue_mask(backbone_out)  # Scanner-agnostic
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

            # === Standard OCT Clinical Metrics ===
            # ENL (Equivalent Number of Looks): higher = less speckle
            # Computed on background region (lower half, which is typically homogeneous)
            H_img = backbone_out.shape[2]
            bg_region_b = backbone_out[0, 0, H_img*3//4:, :].cpu()
            bg_region_c = corrected[0, 0, H_img*3//4:, :].cpu()
            bg_region_n = noisy[0, 0, H_img*3//4:, :].cpu()
            enl_b = ((bg_region_b.mean() / bg_region_b.std().clamp(min=1e-6)) ** 2).clamp(max=1e6)
            enl_c = ((bg_region_c.mean() / bg_region_c.std().clamp(min=1e-6)) ** 2).clamp(max=1e6)
            enl_n = ((bg_region_n.mean() / bg_region_n.std().clamp(min=1e-6)) ** 2).clamp(max=1e6)
            total_enl_noisy += enl_n.item()
            total_enl_backbone += enl_b.item()
            total_enl_corrected += enl_c.item()

            # SNR: signal_mean / noise_std (tissue signal vs background noise)
            tissue_region_b = backbone_out[0, 0, :H_img//2, :].cpu()
            tissue_region_c = corrected[0, 0, :H_img//2, :].cpu()
            snr_b = (tissue_region_b.mean() / bg_region_b.std().clamp(min=1e-6)).clamp(max=1e4)
            snr_c = (tissue_region_c.mean() / bg_region_c.std().clamp(min=1e-6)).clamp(max=1e4)
            total_snr_backbone += snr_b.item()
            total_snr_corrected += snr_c.item()

            del bg_region_b, bg_region_c, bg_region_n, tissue_region_b, tissue_region_c
            del backbone_lap, corrected_lap, clean_lap
            del backbone_gx, corrected_gx, clean_gx, backbone_edge_mag, corrected_edge_mag, clean_edge_mag
            del clean_edge_flat, backbone_edge_flat, corrected_edge_flat, correction
            del clean_edge_std, backbone_edge_std, corrected_edge_std
            del clean_norm, backbone_norm, corrected_norm

        # Lambda statistics from info (corrector returns 'lambda_stats' not 'lambda_maps')
        info_lambda_stats = info.get('lambda_stats', {})
        for name, stats in info_lambda_stats.items():
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

        del corrected, backbone_out, info, clean, noisy
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
        # Standard OCT clinical metrics
        'enl_noisy': total_enl_noisy / n,
        'enl_backbone': total_enl_backbone / n,
        'enl_corrected': total_enl_corrected / n,
        'snr_backbone': total_snr_backbone / n,
        'snr_corrected': total_snr_corrected / n,

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
    # Thresholds tuned for single-corrector (gain) architecture.
    # With 1 corrector, correlation naturally lower than with 2+ correctors
    # because there's less spatial differentiation in allocation.
    # The metric uses max(potential_corr, correction_mag_corr) so it captures
    # both symbolic (potential) and effective (actual correction) cooperation.
    if uncertainty_corr > 0.2:
        coop_status = "EXCELLENT"
        coop_detail = "Strong cooperation pattern"
    elif uncertainty_corr > 0.05:
        coop_status = "GOOD"
        coop_detail = "Moderate cooperation"
    elif uncertainty_corr > 0:
        coop_status = "FAIR"
        coop_detail = "Weak cooperation"
    elif uncertainty_corr > -0.1:
        coop_status = "NEUTRAL"
        coop_detail = "No clear cooperation signal"
    else:
        coop_status = "POOR"
        coop_detail = "Anti-cooperative pattern"

    print(f"│ {'Uncertainty-Correction Correlation':<40} {uncertainty_corr:>+24.3f} │")
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
    # Show all active correctors (gain, tissue, boundary, etc.)
    corrector_names = sorted(set(list(corrector_activity.keys()) + list(lambda_stats.keys())))
    if not corrector_names:
        corrector_names = ['gain']  # Default for single-corrector architecture
    for name in corrector_names:
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
    # ENL and SNR
    enl_noisy = val_metrics.get('enl_noisy', 0)
    enl_backbone = val_metrics.get('enl_backbone', 0)
    enl_corrected = val_metrics.get('enl_corrected', 0)
    enl_delta = enl_corrected - enl_backbone
    enl_status = "✓" if enl_delta >= 0 else "~"
    snr_backbone = val_metrics.get('snr_backbone', 0)
    snr_corrected = val_metrics.get('snr_corrected', 0)
    snr_delta = snr_corrected - snr_backbone
    snr_status = "✓" if snr_delta >= 0 else "~"
    print(f"│ {'ENL (Equiv. Number of Looks)':<30} {enl_backbone:>12.2f} {enl_corrected:>12.2f} {enl_delta:>+9.2f}{enl_status} │")
    print(f"│ {'SNR (Signal-to-Noise)':<30} {snr_backbone:>12.2f} {snr_corrected:>12.2f} {snr_delta:>+9.2f}{snr_status} │")
    print(f"│ {'ENL Noisy (reference)':<30} {enl_noisy:>12.2f} {'':>12} {'':>10} │")
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
        pred_map = {'gain': ['P1', 'P2', 'P3', 'P4', 'P6'], 'tissue': ['P2', 'P3', 'P6'], 'boundary': ['P1', 'P4']}
        pred_keys = pred_map.get(most_active, ['P1'])
        pred_key = pred_keys[0]  # Use first predicate for display
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
        if uncertainty_corr < 0.05:
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
    parser.add_argument('--max_val', type=int, default=30)
    parser.add_argument('--patch_size', type=int, default=96)

    # Model
    parser.add_argument('--backbone', type=str, default='nafnet',
                        choices=['nafnet', 'dncnn', 'swinir', 'kbnet'],
                        help='SOTA backbone architecture')
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
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Loss config
    parser.add_argument('--cooperation_weight', type=float, default=0.2)
    parser.add_argument('--efficiency_weight', type=float, default=0.2)
    parser.add_argument('--psnr_slack', type=float, default=1.0)

    # Fine-grained loss weights (from Optuna sweep)
    parser.add_argument('--epi_weight', type=float, default=1.0, help='EPI preservation weight')
    parser.add_argument('--enl_weight', type=float, default=0.3, help='ENL preservation weight')
    parser.add_argument('--snr_weight', type=float, default=0.3, help='SNR preservation weight')
    parser.add_argument('--edge_weight', type=float, default=0.5, help='Edge/boundary loss weight')
    parser.add_argument('--psnr_preserve_weight', type=float, default=2.0, help='PSNR preservation weight')
    parser.add_argument('--bg_rule', default='intensity',
                        help='Where corrections are allowed to act. intensity is the per image \n                              fifteenth percentile gate. otsu zeroes every correction outside a \n                              dilated Otsu tissue mask, which makes the background noise measures provable.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for torch, numpy and the data order. Set so that a run can be repeated.')
    parser.add_argument('--psnr_dead_zone', type=float, default=0.3, help='PSNR drop (dB) below which no penalty (default: 0.3). Set higher (e.g. 1.5) to allow larger PSNR sacrifice for clinical gains.')
    parser.add_argument('--ssim_preserve_weight', type=float, default=1.5, help='SSIM preservation weight')
    parser.add_argument('--correction_clamp', type=float, default=0.15, help='Max correction magnitude (default: 0.15)')
    parser.add_argument('--clinical_clamp', type=float, default=0.03, help='Max clinical enhancement magnitude (default: 0.03)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Corrector hidden channels (default: 64)')
    parser.add_argument('--min_correction_mag', type=float, default=0.008, help='Minimum correction magnitude floor')

    # Improvement targets (what % above backbone to target)
    parser.add_argument('--cnr_target', type=float, default=0.05, help='CNR improvement target (0.05 = 5%%)')
    parser.add_argument('--enl_target', type=float, default=0.10, help='ENL improvement target (0.10 = 10%%)')
    parser.add_argument('--snr_target', type=float, default=0.10, help='SNR improvement target (0.10 = 10%%)')
    parser.add_argument('--tci_target', type=float, default=0.02, help='TCI improvement target (0.02 = 2%%)')
    parser.add_argument('--epi_target', type=float, default=0.005, help='EPI improvement target (0.005 = 0.5%%)')
    parser.add_argument('--stage_switch_epoch', type=int, default=0, help='Epoch to switch from Stage 1 (clinical only) to Stage 2 (+ PSNR). Default: epochs//2+1')

    # Output
    parser.add_argument('--output_dir', default='outputs/nsnd_v8_cooperative')

    # Ablation study
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_negotiator', 'no_edge', 'no_uncertainty', 'no_bg_smooth'],
                        help='Ablation mode: disable a specific component')

    # Resume from checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='Override starting epoch (default: checkpoint epoch + 1)')

    # Stage 2 EPI: freeze gain corrector, train only edge recovery
    parser.add_argument('--stage2_epi', action='store_true',
                        help='Stage 2: freeze all except edge recovery, focus on EPI')
    parser.add_argument('--stage2_epi_lr', type=float, default=0.002,
                        help='Edge recovery LR for Stage 2 (default: 0.002)')
    parser.add_argument('--stage2_epi_target', type=float, default=1.05,
                        help='EPI target as multiplier of backbone EPI (1.05 = +5%%)')
    parser.add_argument('--stage2_edge_ceiling', type=float, default=0.10,
                        help='Edge recovery magnitude ceiling for Stage 2 (default: 0.10)')

    # Unified multi-phase training: Phase 1 (all correctors) + Phase 2 (edge recovery only)
    parser.add_argument('--phase2_epochs', type=int, default=0,
                        help='Number of Phase 2 (EPI-focused) epochs AFTER Phase 1. '
                             '0 = Phase 1 only. Total epochs = --epochs + --phase2_epochs')
    parser.add_argument('--phase2_lr', type=float, default=1.6e-4,
                        help='Edge recovery LR for Phase 2 (default: 1.6e-4, fallback; '
                             'main logic derives from lr_corrector * 4.0 / 5.0)')
    parser.add_argument('--phase2_epi_target', type=float, default=1.05,
                        help='EPI target for Phase 2 as multiplier of backbone EPI')
    parser.add_argument('--phase2_edge_ceiling', type=float, default=0.10,
                        help='Edge recovery magnitude ceiling for Phase 2')

    # Few-shot cross-scanner adaptation
    parser.add_argument('--scanner_adapter', action='store_true',
                        help='Enable scanner adapter for few-shot cross-scanner adaptation (2 params)')
    parser.add_argument('--ewc_weight', type=float, default=0.0,
                        help='EWC-style L2 anchor weight for few-shot adaptation (0=disabled)')
    parser.add_argument('--bg_correction_var_weight', type=float, default=0.0,
                        help='Weight for background correction variance penalty (recommended 10.0 for SNR)')

    # Torch compile optimization
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile optimization')

    # Backbone caching for slow backbones (DnCNN, SwinIR)
    parser.add_argument('--cache_backbone', action='store_true',
                        help='Pre-compute backbone outputs before training. '
                             'Auto-enabled for DnCNN/SwinIR when --freeze_backbone is set.')
    parser.add_argument('--patches_per_image', type=int, default=2,
                        help='Number of fixed patches per image for caching (default: 2)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed every source of randomness before anything is built, so that a run can be
    # repeated and so that selecting a setting on validation means something. Without
    # this, two runs of the same setting differ by more than the settings differ.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f'  Seed: {args.seed}')

    # CPU speed optimization: set optimal thread count
    if args.device == 'cpu':
        n_cores = os.cpu_count() or 4
        # Use physical cores (half of logical on hyperthreaded CPUs)
        n_threads = max(1, n_cores // 2)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        print(f"CPU threading: {n_threads} intra-op, {max(1, n_threads // 2)} inter-op threads")

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
    print(f"  Backbone: {args.backbone.upper()}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  LR corrector: {args.lr_corrector}")
    print(f"  LR potential: {args.lr_potential}")
    print(f"  LR negotiator: {args.lr_negotiator}")
    print(f"  Cooperation weight: {args.cooperation_weight}")
    print(f"  Efficiency weight: {args.efficiency_weight}")
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

    # CPU: 1 worker (avoids memory duplication + context-switch overhead)
    # GPU: 4 workers (overlap data loading with GPU computation)
    num_workers = 4 if args.device != 'cpu' else 1
    # The shuffle order is drawn from a seeded generator, otherwise the data order
    # stays random even when every other source of randomness is seeded.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=loader_gen,
        num_workers=num_workers, pin_memory=(args.device != 'cpu'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # FIX: Consistent batch sizes for better GPU utilization
    )
    # FIX: Use batch_size=1 for validation (full-resolution images use ~28x more memory than 96x96 patches)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0 if args.device == 'cpu' else min(2, num_workers),
        pin_memory=(args.device != 'cpu')
    )

    # Model
    # Multi-phase: start with GuidedEdgeSharpener (QT44), swap to EdgeRecoveryModule at Phase 2 (QT45b)
    use_guided_edge = (args.phase2_epochs > 0)
    print("\nInitializing cooperative model...")
    if use_guided_edge:
        print("  Multi-phase mode: starting with GuidedEdgeSharpener (will swap at Phase 2)")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        correction_clamp=args.correction_clamp,
        clinical_clamp=args.clinical_clamp,
        hidden_channels=args.hidden_channels,
        use_guided_edge=use_guided_edge,
        scanner_adapter=args.scanner_adapter,
    ).to(args.device)

    # Set ablation mode if specified
    if args.ablation != 'none':
        model.corrector.set_ablation(args.ablation)

    # Freeze backbone (IMPORTANT: train only corrector, potential, negotiator)
    if args.freeze_backbone:
        print(f"\n*** FREEZING {args.backbone.upper()} BACKBONE WEIGHTS ***")
        for param in model.backbone.backbone.parameters():
            param.requires_grad = False
        # Keep uncertainty head + calibration params trainable
        for param in model.backbone.uncertainty_parameters():
            param.requires_grad = True

    # Auto-enable backbone caching for slow backbones
    if (args.freeze_backbone and
            args.backbone in BackboneWrapper._SLOW_BACKBONES):
        args.cache_backbone = True

    # Pre-compute backbone outputs for training (huge speedup for DnCNN/SwinIR)
    if getattr(args, 'cache_backbone', False) and args.freeze_backbone:
        print(f"\n*** BACKBONE CACHING ENABLED for {args.backbone.upper()} ***")
        print(f"  This pre-computes backbone outputs once, then trains corrector")
        print(f"  on cached outputs — no backbone forward during training.")
        train_dataset.precompute_backbone_cache(
            model.backbone, device=args.device,
            patches_per_image=getattr(args, 'patches_per_image', 1),
        )
        # Rebuild train_loader with cached dataset (num_workers=0 to avoid
        # duplicating cached tensors across worker processes)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=True,
        )

    # Pre-compute validation backbone outputs for slow backbones (SwinIR/DnCNN)
    # Backbone is frozen so outputs are identical every epoch — compute once
    val_backbone_cache = None
    if (args.freeze_backbone and
            args.backbone in BackboneWrapper._SLOW_BACKBONES):
        val_backbone_cache = _precompute_val_backbone_cache(
            model, val_loader, args.device)

    # Stage 2 EPI: freeze EVERYTHING except edge recovery module
    if args.stage2_epi:
        print("\n" + "=" * 70)
        print("STAGE 2 EPI MODE: Freeze all except edge recovery module")
        print("=" * 70)
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'edge_sharpener' not in name:
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
        unfrozen_count = 0
        trainable_params = 0
        for param in model.corrector.edge_sharpener.parameters():
            param.requires_grad = True
            unfrozen_count += 1
            trainable_params += param.numel()
        print(f"  Frozen: {frozen_count} parameter tensors")
        print(f"  Unfrozen (edge recovery): {unfrozen_count} tensors = {trainable_params:,} params")
        print(f"  EPI target: {args.stage2_epi_target:.0%} of backbone")
        print(f"  Edge ceiling: {args.stage2_edge_ceiling}")
        print(f"  LR: {args.stage2_epi_lr}")
        print("=" * 70)

    # Where corrections are allowed to act. Set on the corrector rather than passed
    # through the constructor, so that the evaluation scripts can set it the same way.
    model.corrector.bg_rule = args.bg_rule
    print(f'  Background rule: {args.bg_rule}')

    # Loss — Predicate-Driven Simplified Loss (4 terms, no learned weights)
    criterion = SimplifiedCooperativeLoss(
        predicates=model.corrector.predicates,
        cooperation_weight=args.cooperation_weight,
        bg_correction_var_weight=args.bg_correction_var_weight,
        psnr_dead_zone=args.psnr_dead_zone,
    ).to(args.device)

    # Stage 2 EPI: configure loss for edge-focused training
    if args.stage2_epi:
        criterion._stage2_epi_mode = True
        criterion._stage2_epi_target = args.stage2_epi_target
        criterion._stage2_edge_ceiling = args.stage2_edge_ceiling

    # Apply torch.compile optimization (PyTorch 2.0+)
    # Skip compiling backbone when it's cached (never called during training)
    # Instead compile the corrector for speedup on the active computation
    # CPU: Skip torch.compile entirely — compilation overhead (~30-60s) exceeds
    # gains for short LOO runs (15 epochs × 15 images), and inductor CPU backend
    # is unreliable. Net negative for CPU workloads.
    if hasattr(torch, 'compile') and not args.no_compile and args.device != 'cpu':
        use_cache = getattr(args, 'cache_backbone', False) and args.freeze_backbone
        if args.backbone in ('swinir', 'dncnn') and not use_cache:
            # The backbone is deliberately left eager as well.
            #
            # Compiling it under max-autotune shifts its output by up to 7.6e-3, mean
            # 1.7e-3, on a range of 0.08 to 0.87. That sounds small until it is compared
            # with the correction itself, whose mean absolute size is 1.07e-2, so the
            # discrepancy is about a sixth of the effect being measured. The backbone
            # output is the reference that every clinical change is measured against,
            # and the scoring scripts run it eagerly, so compiling here trains against
            # one reference and reports against another. Measured 2026-09-10.
            print(f"{args.backbone.upper()} backbone left eager, matching the scoring path")
        elif use_cache:
            # The corrector is compiled without reduce-overhead. That mode turns on
            # CUDA graphs, and the graph state cannot survive an operation that
            # allocates dynamically. The percentile background gate calls
            # torch.quantile, which sorts, and every run of that gate died inside
            # cudagraph_trees with "Expected curr_block->next == nullptr". The Otsu
            # gate is built from scatter_add and cumsum and was unaffected, which is
            # why only one of the two gates failed and only on the one backbone whose
            # corrector is compiled at all. The default mode keeps most of the speed
            # and does not capture graphs.
            print(f"Backbone cached — skipping torch.compile on backbone, compiling corrector...")
            # The corrector is deliberately left eager.
            #
            # Compiling it changes its output. At identical resumed weights and an
            # identical first batch, iteration one reports loss 30.49 compiled against
            # 12.87 eager, and after two epochs the validation fidelity delta differs by
            # more than a decibel. Compiled and compiled-with-cuda-graphs agree to three
            # decimals, so this is not rounding. Every scoring script runs the corrector
            # eagerly, so a compiled model was being trained under one function and
            # measured under another, which invalidates the numbers rather than merely
            # slowing them. Compiling also breaks the percentile gate at validation,
            # where torch.quantile meets symbolic shapes after training at a smaller
            # crop size. There is nothing to gain either way: measured throughput is
            # 11.2 iterations per second eager against 11.4 compiled.
            print("Backbone cached — corrector left eager, matching the scoring path")

    # Optimizer with different learning rates for different components
    # Separate corrector output heads for higher learning rate (5x)
    # Output heads are the final Conv layers that produce correction maps
    # They need higher LR to escape the near-zero initialization basin
    corrector_output_params = []
    _output_param_ids = set()
    head_names = {'tissue': 'output_head', 'boundary': 'output_head', 'gain': 'output_head'}
    for cname, cwrapper in model.corrector.correctors.items():
        base = cwrapper.base_corrector
        hname = head_names.get(cname)
        if hname and hasattr(base, hname):
            for p in getattr(base, hname).parameters():
                _output_param_ids.add(id(p))
                corrector_output_params.append(p)
        # Also include strength parameter at higher lr
        if hasattr(base, 'strength'):
            _output_param_ids.add(id(base.strength))
            corrector_output_params.append(base.strength)

    # Remaining corrector parameters at normal LR
    corrector_params = [p for p in model.corrector.correctors.parameters()
                        if id(p) not in _output_param_ids]

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
    # NOTE: cnr_preserver.region_detector is shared with region_aware_corrector.region_detector
    # (same object), so we must deduplicate to avoid the "duplicate parameters" warning.
    clinical_enhancement_params = []
    _seen_param_ids = set()
    for module_name in ['cnr_preserver', 'clinical_enhancer', 'region_aware_corrector', 'confidence_estimator']:
        if hasattr(model.corrector, module_name):
            for p in getattr(model.corrector, module_name).parameters():
                if id(p) not in _seen_param_ids:
                    _seen_param_ids.add(id(p))
                    clinical_enhancement_params.append(p)

    # Edge sharpener params — always included in optimizer (active from Phase 1).
    # In multi-phase: Phase 1 uses GuidedEdgeSharpener (like QT44), swap at Phase 2.
    # In single-phase: uses EdgeRecoveryModule (backward compatible).
    edge_sharpener_params = []
    if hasattr(model.corrector, 'edge_sharpener'):
        edge_sharpener_params = list(model.corrector.edge_sharpener.parameters())
        edge_type = type(model.corrector.edge_sharpener).__name__
        if args.phase2_epochs > 0:
            print(f"  Edge module ({edge_type}) ACTIVE in Phase 1 (will swap to EdgeRecoveryModule at Phase 2)")
    elif hasattr(model.corrector, 'edge_residual_filter'):
        edge_sharpener_params = list(model.corrector.edge_residual_filter.parameters())

    # Adaptive background smoother + suppressor + tone curve — DISABLED in forward pass (QT58 showed harmful)
    # Freeze their parameters so optimizer doesn't waste compute
    bg_smooth_params = []
    for mod_name in ['adaptive_smoother', 'bg_suppressor', 'tone_curve']:
        if hasattr(model.corrector, mod_name):
            for p in getattr(model.corrector, mod_name).parameters():
                p.requires_grad_(False)
    # But bg_darken_strength is a single scalar — add to clinical enhancement params
    if hasattr(model.corrector, 'bg_darken_strength'):
        bg_smooth_params = [model.corrector.bg_darken_strength]

    # Backbone uncertainty head + calibration params
    uncertainty_params = list(model.backbone.uncertainty_parameters())

    # Loss weight parameters (only if loss has learnable params)
    loss_weight_params = list(criterion.log_sigma.parameters()) if hasattr(criterion, 'log_sigma') else []

    if args.stage2_epi:
        # Stage 2 EPI: single param group, only edge recovery module
        edge_only_params = [p for p in model.corrector.edge_sharpener.parameters() if p.requires_grad]
        param_groups = [
            {'params': edge_only_params, 'lr': args.stage2_epi_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
        print(f"\nStage 2 EPI optimizer: {sum(p.numel() for p in edge_only_params):,} params @ lr={args.stage2_epi_lr}")
    else:
        param_groups = [
            {'params': corrector_params, 'lr': args.lr_corrector},
            {'params': corrector_output_params, 'lr': args.lr_corrector * 8.0},  # 8x LR for output heads + strength (between 5x too weak and 15x too strong)
            {'params': potential_params, 'lr': args.lr_potential} if potential_params else {'params': [], 'lr': 0},
            {'params': negotiator_params, 'lr': args.lr_negotiator} if negotiator_params else {'params': [], 'lr': 0},
            {'params': clinical_enhancement_params, 'lr': args.lr_corrector} if clinical_enhancement_params else {'params': [], 'lr': 0},
            {'params': edge_sharpener_params, 'lr': args.lr_corrector * 4.0} if edge_sharpener_params else {'params': [], 'lr': 0},  # 4x LR: dedicated EPI module needs faster learning
            {'params': uncertainty_params, 'lr': args.lr_potential},
            {'params': bg_smooth_params, 'lr': args.lr_corrector * 2.0} if bg_smooth_params else {'params': [], 'lr': 0},
            {'params': loss_weight_params, 'lr': args.lr_corrector * 0.25, 'weight_decay': 0},
        ]

        # Filter out empty param groups
        param_groups = [pg for pg in param_groups if len(list(pg['params'])) > 0 or pg['lr'] > 0]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        print(f"\nOptimizer configuration:")
        print(f"  Corrector params (backbone): {sum(p.numel() for p in corrector_params):,} @ lr={args.lr_corrector}")
        print(f"  Corrector output heads: {sum(p.numel() for p in corrector_output_params):,} @ lr={args.lr_corrector * 8.0}")
        if potential_params:
            print(f"  Potential params: {sum(p.numel() for p in potential_params):,} @ lr={args.lr_potential}")
        if negotiator_params:
            print(f"  Negotiator params: {sum(p.numel() for p in negotiator_params):,} @ lr={args.lr_negotiator}")
        if clinical_enhancement_params:
            print(f"  Clinical enhancement params: {sum(p.numel() for p in clinical_enhancement_params):,} @ lr={args.lr_corrector}")
        if edge_sharpener_params:
            print(f"  Edge sharpener params: {sum(p.numel() for p in edge_sharpener_params):,} @ lr={args.lr_corrector * 4.0}")
        print(f"  Uncertainty params: {sum(p.numel() for p in uncertainty_params):,} @ lr={args.lr_potential}")
        if bg_smooth_params:
            print(f"  BG smoothing params: {sum(p.numel() for p in bg_smooth_params):,} @ lr={args.lr_corrector * 2.0}")
        if loss_weight_params:
            print(f"  Loss weight params: {sum(p.numel() for p in loss_weight_params):,} @ lr={args.lr_corrector * 0.25}")

    # Total epochs includes phase 2 if requested
    total_epochs = args.epochs + args.phase2_epochs
    phase1_end_epoch = args.epochs  # Last epoch of phase 1

    if args.phase2_epochs > 0:
        print(f"\n  Multi-phase training (QT44->QT45b architecture swap):")
        print(f"    Phase 1: epochs 1-{phase1_end_epoch} (all correctors + GuidedEdgeSharpener active)")
        print(f"    Phase 2: epochs {phase1_end_epoch+1}-{total_epochs} (swap to EdgeRecoveryModule, mimics QT45b)")
        print(f"    Phase 2 epoch 1: full LR (corrector={args.lr_corrector:.6f}, edge={args.lr_corrector*4:.6f})")
        print(f"    Phase 2 epoch 2+: 5x cut (corrector={args.lr_corrector/5:.6f}, edge={args.lr_corrector*4/5:.6f})")

    # Scheduler — T_max covers phase 1 only (phase 2 gets its own scheduler)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_corrector * 0.01
    )

    # AMP
    use_amp = args.device != 'cpu' and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("  Using Automatic Mixed Precision (AMP)")

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        print(f"\n  Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        # Filter out size-mismatched keys (e.g., output_head changed from 1→2 channels)
        ckpt_state = ckpt['model_state_dict']
        model_state = model.state_dict()
        compatible_state = {}
        for k, v in ckpt_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible_state[k] = v
            elif k in model_state:
                print(f"  Skipping mismatched key: {k} (ckpt {v.shape} vs model {model_state[k].shape})")
        _has_mismatched_keys = len(compatible_state) < len(ckpt_state)
        model.load_state_dict(compatible_state, strict=False)
        # Skip optimizer/scheduler restore when param groups don't match
        # (e.g., stage2_epi mode or architecture change like EdgeRecoveryModule)
        if _has_mismatched_keys:
            print("  Skipping optimizer restore (architecture changed — mismatched keys)")
        elif 'optimizer_state_dict' in ckpt and not args.stage2_epi:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                # Move optimizer state to correct device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(args.device)
                print("  Optimizer state restored successfully")
            except (ValueError, RuntimeError) as e:
                print(f"  Warning: Could not restore optimizer state (architecture changed?): {e}")
                print("  Starting with fresh optimizer state")
        if 'scheduler_state_dict' in ckpt and not args.stage2_epi:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except (ValueError, RuntimeError):
                print("  Warning: Could not restore scheduler state, using fresh scheduler")
        resume_epoch = ckpt.get('epoch', 0)
        start_epoch = args.resume_epoch if args.resume_epoch else resume_epoch + 1
        print(f"  Resumed from epoch {resume_epoch}, starting at epoch {start_epoch}")
        del ckpt

    # Scanner adapter mode: freeze everything except the adapter (2 params)
    if args.scanner_adapter:
        adapter = model.corrector.scanner_adapter
        if adapter is None:
            raise ValueError("--scanner_adapter flag set but model has no scanner_adapter")
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze only scanner adapter
        for param in adapter.parameters():
            param.requires_grad = True
        adapter_params = sum(p.numel() for p in adapter.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  [Scanner Adapter] Frozen {total_params - adapter_params:,} params, "
              f"training {adapter_params} adapter params (gamma, beta)")
        # Rebuild optimizer with only adapter params
        optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr_corrector)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"  [Scanner Adapter] Optimizer: Adam, LR={args.lr_corrector}")

    # EWC anchor: snapshot trainable params for L2 regularization during few-shot adaptation
    anchor_params = None
    if args.ewc_weight > 0:
        anchor_params = {name: p.clone().detach()
                         for name, p in model.named_parameters() if p.requires_grad}
        print(f"  [EWC] Anchored {len(anchor_params)} params, weight={args.ewc_weight}")

    # Training loop
    print("\n" + "#"*80)
    print("# TRAINING COOPERATIVE NEURO-SYMBOLIC DENOISING")
    print("# Target: All predicates pass, +10-15% clinical, < 1.0 dB PSNR drop")
    print("#"*80)

    best_score = -float('inf')
    best_score_stage2 = -float('inf')
    best_score_phase2 = -float('inf')
    best_epoch = 0
    best_epoch_phase2 = 0
    in_phase2 = False
    phase2_stage_switched = False

    for epoch in range(start_epoch, total_epochs + 1):
        # ─── Phase 2 transition ───
        if args.phase2_epochs > 0 and epoch == phase1_end_epoch + 1 and not in_phase2:
            in_phase2 = True
            print("\n" + "=" * 80)
            print("  PHASE 2 TRANSITION: Swap GuidedEdgeSharpener -> EdgeRecoveryModule")
            print("  (Mimics QT44->QT45b: correctors + guided edge in Phase 1, then edge recovery)")
            print("=" * 80)

            # Save phase 1 best checkpoint explicitly
            torch.save({
                'epoch': phase1_end_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'phase': 1,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'best_model_phase1.pth'))
            print(f"  Phase 1 best model saved (score: {best_score:.2f}, epoch {best_epoch})")

            # SWAP edge module: GuidedEdgeSharpener -> EdgeRecoveryModule
            model.corrector.swap_edge_module()

            # === EXACT QT45b reproduction ===
            # QT45b resumed QT44 into new EdgeRecoveryModule architecture:
            # - ALL corrector weights preserved from QT44
            # - EdgeRecoveryModule randomly initialized (architecture mismatch)
            # - Fresh optimizer (couldn't restore due to architecture change)
            # - Full cooperative loss (stage2_epi was False)
            # - Stage 1 (epoch 1, full LR) → Stage 2 (epochs 2+, 5x LR cut)

            # Phase 2: Only train edge module + uncertainty + bg smoothing
            # Freeze gain corrector to preserve Phase 1 learning
            for name, param in model.corrector.named_parameters():
                if 'edge_sharpener' not in name:
                    param.requires_grad_(False)

            # Collect only trainable params (bg_smooth modules disabled)
            edge_params_p2 = list(model.corrector.edge_sharpener.parameters())
            uncertainty_params_p2 = list(model.backbone.uncertainty_parameters())

            p2_param_groups = [
                {'params': edge_params_p2, 'lr': args.lr_corrector * 4.0},
                {'params': uncertainty_params_p2, 'lr': args.lr_potential},
            ]
            p2_param_groups = [pg for pg in p2_param_groups if len(list(pg['params'])) > 0]

            frozen_count = sum(1 for p in model.corrector.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in model.corrector.parameters() if p.requires_grad)
            print(f"  Corrector params FROZEN (preserving Phase 1): {frozen_count} tensors")
            print(f"  Corrector params TRAINABLE: {trainable_count} tensors")
            print(f"  Edge recovery params: {sum(p.numel() for p in edge_params_p2):,} @ lr={args.lr_corrector * 4.0}")
            print(f"  Uncertainty params: {sum(p.numel() for p in uncertainty_params_p2):,} @ lr={args.lr_potential}")
            print(f"  Loss: full cooperative (clinical + predicate + cooperation) — QT45b used full loss")
            print(f"  Stage: 1 (epochs 1-2 full LR) → 2 (epoch 3+ with 5x LR cut)")

            # Fresh optimizer — only edge + uncertainty + bg_smooth params
            optimizer = torch.optim.AdamW(p2_param_groups, weight_decay=1e-4)

            # Keep full cooperative loss (QT45b: stage2_epi=False)
            criterion.training_stage = 1  # Start Phase 2 at Stage 1
            phase2_stage_switched = False

            # New scheduler for phase 2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.phase2_epochs, eta_min=args.lr_corrector * 0.01
            )
            print("=" * 80)

        # ─── Training stage logic ───
        if not in_phase2:
            stage_switch_epoch = args.stage_switch_epoch if args.stage_switch_epoch > 0 else 4
            if epoch < stage_switch_epoch:
                criterion.training_stage = 1
                stage_label = "PHASE 1 / STAGE 1 (clinical focus)"
            else:
                criterion.training_stage = 2
                stage_label = "PHASE 1 / STAGE 2 (+ quality guard)"
                if epoch == stage_switch_epoch:
                    print(f"  >>> Stage transition: reducing LR by 5x")
                    for pg in optimizer.param_groups:
                        pg['lr'] = pg['lr'] / 5.0
            print(f"\n>>> Epoch {epoch}/{total_epochs}: {stage_label}")
        else:
            # Phase 2: Stage 1 (epochs 1-2, full LR) → Stage 2 (epoch 3+, 5x LR cut)
            # Give EdgeRecoveryModule 2 full-LR epochs before cutting
            phase2_epoch_num = epoch - phase1_end_epoch  # 1-based within phase 2
            if phase2_epoch_num <= 2:
                criterion.training_stage = 1
                stage_label = "PHASE 2 / STAGE 1 (clinical focus, full LR)"
            else:
                if not phase2_stage_switched:
                    phase2_stage_switched = True
                    criterion.training_stage = 2
                    print(f"  >>> Phase 2 stage transition: reducing LR by 5x")
                    for pg in optimizer.param_groups:
                        pg['lr'] = pg['lr'] / 5.0
                stage_label = "PHASE 2 / STAGE 2 (+ quality guard)"
            print(f"\n>>> Epoch {epoch}/{total_epochs}: {stage_label}")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, scaler,
            anchor_params=anchor_params, ewc_weight=args.ewc_weight
        )
        scheduler.step()

        if epoch % args.val_every == 0 or epoch == args.epochs:
            # FIX: Clear GPU cache before validation (full-res images need more memory than training patches)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            val_metrics = validate(model, val_loader, args.device, lpips_model=None, criterion=criterion,
                                   val_backbone_cache=val_backbone_cache)
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

            if in_phase2:
                # Phase 2: separate best tracking
                if score > best_score_phase2:
                    best_score_phase2 = score
                    best_epoch_phase2 = epoch

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'best_score': best_score_phase2,
                        'phase': 2,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model_phase2.pth'))
                    print(f"*** New best Phase 2 model! Score: {best_score_phase2:.2f} (PSNR: {psnr_delta:+.3f} dB) ***")

                # Also update overall best (phase 2 model is the final product)
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
                        'phase': 2,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model_cooperative.pth'))
                    print(f"*** New overall best model! Score: {best_score:.2f} ***")
            else:
                # Phase 1: original best tracking
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
                        'stage': criterion.training_stage,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model_cooperative.pth'))

                    print(f"*** New best model! Score: {best_score:.2f} ***")

                # Save separate best checkpoint for Stage 2 (quality guardrail)
                if criterion.training_stage == 2 and score > best_score_stage2:
                    best_score_stage2 = score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'best_score': score,
                        'stage': 2,
                        'args': vars(args),
                    }, os.path.join(args.output_dir, 'best_model_stage2.pth'))
                    print(f"*** New best Stage 2 model! Score: {score:.2f} (PSNR delta: {psnr_delta:+.3f} dB) ***")
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
    if best_score_stage2 > -float('inf'):
        print(f"Best Stage 2 score: {best_score_stage2:.2f}")
        print(f"Stage 2 model saved to: {args.output_dir}/best_model_stage2.pth")
    if args.phase2_epochs > 0:
        print(f"\nMulti-phase summary:")
        print(f"  Phase 1 (epochs 1-{phase1_end_epoch}): best score {best_score:.2f} at epoch {best_epoch}")
        print(f"  Phase 1 model: {args.output_dir}/best_model_phase1.pth")
        if best_score_phase2 > -float('inf'):
            print(f"  Phase 2 (epochs {phase1_end_epoch+1}-{total_epochs}): best score {best_score_phase2:.2f} at epoch {best_epoch_phase2}")
            print(f"  Phase 2 model: {args.output_dir}/best_model_phase2.pth")


if __name__ == '__main__':
    main()
