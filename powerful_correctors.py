#!/usr/bin/env python3
"""
Powerful Correctors with Attention for Neuro-Symbolic OCT Denoising V4

Key improvements:
1. Larger capacity (~300K params per corrector vs ~50K)
2. Self-attention for global context awareness
3. Multi-scale feature processing
4. Still interpretable (each corrector has clear semantic purpose)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# ATTENTION MODULES
# =============================================================================

class EfficientAttention(nn.Module):
    """
    Efficient self-attention with linear complexity.
    Uses kernel approximation instead of O(N^2) attention.
    """

    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, head_dim, N]

        # Efficient attention via softmax kernel
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)

        # Linear attention: O(N) instead of O(N^2)
        context = torch.einsum('bhdn,bhen->bhde', k, v)  # [B, heads, head_dim, head_dim]
        out = torch.einsum('bhdn,bhde->bhen', q, context)  # [B, heads, head_dim, N]

        out = out.reshape(B, C, H, W)
        return self.proj(out)


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on important regions."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_val = x.max(dim=1, keepdim=True)[0]
        attention = self.conv(torch.cat([avg, max_val], dim=1))
        return x * attention


# =============================================================================
# POWERFUL CORRECTOR BASE CLASS
# =============================================================================

class PowerfulCorrectorBase(nn.Module):
    """
    Base class for powerful correctors with attention.

    Architecture:
    1. Local feature extraction (conv layers)
    2. Global context via efficient attention
    3. Channel + spatial attention for refinement
    4. Correction prediction
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128,
                 num_heads: int = 4, name: str = "base"):
        super().__init__()
        self.name = name

        # 1. Local feature extraction
        self.local_feat = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Multi-scale context (dilated convolutions)
        self.multiscale = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=8, dilation=8),
        ])

        # 3. Global attention
        self.attention = EfficientAttention(hidden_dim, num_heads=num_heads)

        # 4. Channel + Spatial attention
        self.channel_attn = ChannelAttention(hidden_dim)
        self.spatial_attn = SpatialAttention()

        # 5. Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 6. Correction output (single channel residual)
        self.correction_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Learnable strength with moderate init for meaningful corrections
        # sigmoid(-1.0) ≈ 0.27, so initial effective strength ≈ 0.27 * 0.4 ≈ 0.11
        # This enables more meaningful corrections while still being safe
        self.strength = nn.Parameter(torch.tensor(-1.0))
        self._strength_scale = 0.4  # Increased max correction magnitude

        self._init_weights()

        # CRITICAL: Zero-initialize final layer for near-zero initial correction
        self._zero_init_final_layer()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _zero_init_final_layer(self):
        """Initialize the final conv layer with small random values for gradient flow.

        Using small random init instead of zeros allows:
        - Non-zero initial corrections (enabling gradient flow)
        - Small magnitude corrections that grow as the network learns
        - Better training dynamics vs zero init which blocks gradients
        """
        # Find the last Conv2d in correction_head (before Tanh)
        last_conv = None
        for module in self.correction_head:
            if isinstance(module, nn.Conv2d):
                last_conv = module

        # Small random init for weights, zero for bias
        # This gives small but non-zero initial corrections
        if last_conv is not None:
            nn.init.normal_(last_conv.weight, mean=0, std=0.01)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated input features [B, in_channels, H, W]
            denoised: Current denoised image for residual connection

        Returns:
            correction: Residual correction to add to denoised [B, 1, H, W]
        """
        # Local features
        feat = self.local_feat(x)

        # Multi-scale context
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)  # Back to hidden_dim

        # Global attention
        feat = feat + self.attention(feat)

        # Channel + Spatial attention
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)

        # Refinement
        feat = self.refine(feat)

        # Correction prediction
        raw_correction = self.correction_head(feat)

        # Scale by learnable strength
        strength = torch.sigmoid(self.strength) * self._strength_scale
        correction = raw_correction * strength

        return correction


# =============================================================================
# SPECIALIZED POWERFUL CORRECTORS
# =============================================================================

class PowerfulEdgeCorrector(PowerfulCorrectorBase):
    """
    Powerful edge corrector for P1 (Boundary Detectability).

    Specialized for:
    - Layer boundary enhancement
    - Edge sharpening without artifacts
    - Gradient-aware processing
    """

    def __init__(self, in_channels: int = 19, hidden_dim: int = 128):
        super().__init__(in_channels, hidden_dim, num_heads=4, name="edge")

        # Edge-specific: Sobel-like learnable edge detection
        self.edge_detect = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
        )

        # Fuse edge info with features
        self.edge_fuse = nn.Conv2d(hidden_dim + 16, hidden_dim, 1)

        # Re-apply zero init to ensure near-zero output after all modules initialized
        self._zero_init_final_layer()

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Extract edges from denoised
        edges = self.edge_detect(denoised)

        # Local features
        feat = self.local_feat(x)

        # Multi-scale
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Fuse with edge info
        feat = self.edge_fuse(torch.cat([feat, edges], dim=1))

        # Attention
        feat = feat + self.attention(feat)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)
        feat = self.refine(feat)

        # Correction
        raw_correction = self.correction_head(feat)
        strength = torch.sigmoid(self.strength) * self._strength_scale

        return raw_correction * strength


class PowerfulContrastCorrector(PowerfulCorrectorBase):
    """
    Powerful contrast corrector for P2 (Layer Contrast).

    Specialized for:
    - Local contrast enhancement (CLAHE-inspired)
    - Multi-scale statistics matching
    - Histogram-aware processing
    """

    def __init__(self, in_channels: int = 19, hidden_dim: int = 128):
        super().__init__(in_channels, hidden_dim, num_heads=4, name="contrast")
        self._strength_scale = 0.55  # Increased for stronger contrast adjustments

        # Local statistics computation (different scales)
        self.local_stats = nn.ModuleList([
            nn.AvgPool2d(k, stride=1, padding=k//2) for k in [5, 11, 21]
        ])

        # Statistics-to-feature mapping
        self.stats_encoder = nn.Sequential(
            nn.Conv2d(6, 32, 1),  # 3 means + 3 stds
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 1),
        )

        # Fuse stats with features
        self.stats_fuse = nn.Conv2d(hidden_dim + 32, hidden_dim, 1)

        # Re-apply zero init to ensure near-zero output after all modules initialized
        self._zero_init_final_layer()

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Compute local statistics at multiple scales
        stats_list = []
        for pool in self.local_stats:
            local_mean = pool(denoised)
            local_sq_mean = pool(denoised ** 2)
            local_std = (local_sq_mean - local_mean ** 2).clamp(min=1e-6).sqrt()
            stats_list.extend([local_mean, local_std])

        local_stats = torch.cat(stats_list, dim=1)  # [B, 6, H, W]
        stats_feat = self.stats_encoder(local_stats)

        # Local features
        feat = self.local_feat(x)

        # Multi-scale
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Fuse with statistics
        feat = self.stats_fuse(torch.cat([feat, stats_feat], dim=1))

        # Attention
        feat = feat + self.attention(feat)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)
        feat = self.refine(feat)

        # Correction
        raw_correction = self.correction_head(feat)
        strength = torch.sigmoid(self.strength) * self._strength_scale

        return raw_correction * strength


class PowerfulSharpnessCorrector(PowerfulCorrectorBase):
    """
    Powerful sharpness corrector for P5 (Boundary Sharpness).

    Specialized for:
    - Unsharp masking
    - High-frequency enhancement
    - Boundary-aware sharpening
    """

    def __init__(self, in_channels: int = 19, hidden_dim: int = 128):
        super().__init__(in_channels, hidden_dim, num_heads=4, name="sharpness")
        self._strength_scale = 0.50  # Increased for stronger sharpening

        # Learnable high-pass filter
        self.highpass = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
        )

        # Gaussian blur for unsharp mask (fixed)
        self.register_buffer('blur_kernel', self._make_gaussian_kernel(5, 1.0))

        # Sharpness feature fusion
        self.sharp_fuse = nn.Conv2d(hidden_dim + 16 + 1, hidden_dim, 1)

        # Re-apply zero init to ensure near-zero output after all modules initialized
        self._zero_init_final_layer()

    def _make_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(size).float() - size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        return gauss_2d.view(1, 1, size, size)

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Compute unsharp mask
        blurred = F.conv2d(denoised, self.blur_kernel, padding=2)
        unsharp = denoised - blurred  # High frequency details

        # Learnable high-pass features
        hp_feat = self.highpass(denoised)

        # Local features
        feat = self.local_feat(x)

        # Multi-scale
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Fuse with sharpness info
        feat = self.sharp_fuse(torch.cat([feat, hp_feat, unsharp], dim=1))

        # Attention
        feat = feat + self.attention(feat)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)
        feat = self.refine(feat)

        # Correction
        raw_correction = self.correction_head(feat)
        strength = torch.sigmoid(self.strength) * self._strength_scale

        return raw_correction * strength


class PowerfulTextureCorrector(PowerfulCorrectorBase):
    """
    Powerful texture corrector for P4 (Structure) and P6 (Speckle).

    Specialized for:
    - Texture preservation
    - Speckle suppression
    - Structure-aware filtering
    """

    def __init__(self, in_channels: int = 35, hidden_dim: int = 128):
        super().__init__(in_channels, hidden_dim, num_heads=4, name="texture")
        self._strength_scale = 0.35  # Increased for texture (still conservative)

        # Texture feature extraction (Gabor-inspired)
        self.texture_filters = nn.ModuleList([
            nn.Conv2d(1, 8, (1, 5), padding=(0, 2)),  # Horizontal
            nn.Conv2d(1, 8, (5, 1), padding=(2, 0)),  # Vertical
            nn.Conv2d(1, 8, 3, padding=1),            # Isotropic
        ])

        # Texture fusion
        self.texture_fuse = nn.Conv2d(hidden_dim + 24, hidden_dim, 1)

        # Re-apply zero init to ensure near-zero output after all modules initialized
        self._zero_init_final_layer()

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Multi-orientation texture features
        tex_feats = [f(denoised) for f in self.texture_filters]
        tex_feat = torch.cat(tex_feats, dim=1)  # [B, 24, H, W]

        # Local features
        feat = self.local_feat(x)

        # Multi-scale
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Fuse with texture
        feat = self.texture_fuse(torch.cat([feat, tex_feat], dim=1))

        # Attention
        feat = feat + self.attention(feat)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)
        feat = self.refine(feat)

        # Correction
        raw_correction = self.correction_head(feat)
        strength = torch.sigmoid(self.strength) * self._strength_scale

        return raw_correction * strength


class PowerfulSmoothCorrector(PowerfulCorrectorBase):
    """
    Powerful smooth corrector for P3 (Noise Reduction).

    Specialized for:
    - Adaptive smoothing
    - Edge-preserving filtering
    - Noise reduction in flat regions
    """

    def __init__(self, in_channels: int = 19, hidden_dim: int = 128):
        super().__init__(in_channels, hidden_dim, num_heads=4, name="smooth")
        self._strength_scale = 0.40  # Increased for better noise reduction

        # Edge-aware smoothing kernels
        self.smooth_kernels = nn.ModuleList([
            nn.Conv2d(1, 8, 3, padding=1),
            nn.Conv2d(1, 8, 5, padding=2),
            nn.Conv2d(1, 8, 7, padding=3),
        ])

        # Edge detector for edge-preserving
        self.edge_detect = nn.Conv2d(1, 8, 3, padding=1)

        # Fusion
        self.smooth_fuse = nn.Conv2d(hidden_dim + 32, hidden_dim, 1)

        # Re-apply zero init to ensure near-zero output after all modules initialized
        self._zero_init_final_layer()

    def forward(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        # Multi-scale smoothing
        smooth_feats = [k(denoised) for k in self.smooth_kernels]
        edges = self.edge_detect(denoised)
        smooth_feat = torch.cat(smooth_feats + [edges], dim=1)  # [B, 32, H, W]

        # Local features
        feat = self.local_feat(x)

        # Multi-scale
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Fuse
        feat = self.smooth_fuse(torch.cat([feat, smooth_feat], dim=1))

        # Attention
        feat = feat + self.attention(feat)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)
        feat = self.refine(feat)

        # Correction
        raw_correction = self.correction_head(feat)
        strength = torch.sigmoid(self.strength) * self._strength_scale

        return raw_correction * strength


# =============================================================================
# POWERFUL ADAPTIVE CORRECTOR WITH LAMBDA
# =============================================================================

class PowerfulAdaptiveCorrectorWithLambda(nn.Module):
    """
    Combines all powerful correctors with adaptive lambda maps.

    Total parameters: ~1.5M (vs ~400K in original)
    Still interpretable: each corrector has clear semantic purpose
    """

    def __init__(self, enc1_channels: int = 64, enc2_channels: int = 128,
                 hidden_dim: int = 128):
        super().__init__()

        # Feature adapters
        self.adapt_enc1 = nn.Conv2d(enc1_channels, 16, 1, bias=False)
        self.adapt_enc2 = nn.Conv2d(enc2_channels, 16, 1, bias=False)
        nn.init.xavier_uniform_(self.adapt_enc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.adapt_enc2.weight, gain=0.5)

        # Powerful correctors
        # Edge (P1): denoised(1) + noisy(1) + lambda(1) + enc1(16) = 19 channels
        self.edge_corrector = PowerfulEdgeCorrector(in_channels=19, hidden_dim=hidden_dim)

        # Contrast (P2): same input structure
        self.contrast_corrector = PowerfulContrastCorrector(in_channels=19, hidden_dim=hidden_dim)

        # Sharpness (P5): same input structure
        self.sharpness_corrector = PowerfulSharpnessCorrector(in_channels=19, hidden_dim=hidden_dim)

        # Texture (P4, P6): denoised(1) + noisy(1) + lambda(1) + enc1(16) + enc2(16) = 35 channels
        self.texture_corrector = PowerfulTextureCorrector(in_channels=35, hidden_dim=hidden_dim)

        # Smooth (P3): denoised(1) + noisy(1) + lambda(1) + enc2(16) = 19 channels
        self.smooth_corrector = PowerfulSmoothCorrector(in_channels=19, hidden_dim=hidden_dim)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        print(f"PowerfulAdaptiveCorrectorWithLambda: {total:,} parameters")

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                lambda_maps: Dict[str, torch.Tensor],
                backbone_features: Dict[str, torch.Tensor],
                failure_maps: Dict[str, torch.Tensor] = None,
                return_individual: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply powerful corrections weighted by lambda maps.
        """
        B, C, H, W = denoised.shape
        info = {}

        enc1 = backbone_features.get('enc1')
        enc2 = backbone_features.get('enc2')

        # Adapt backbone features
        f1 = self.adapt_enc1(enc1)
        if f1.shape[2:] != denoised.shape[2:]:
            f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)

        f2 = self.adapt_enc2(enc2)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

        # Get lambda maps
        lambda_edge = lambda_maps.get('edge', torch.zeros_like(denoised))
        lambda_contrast = lambda_maps.get('contrast', torch.zeros_like(denoised))
        lambda_sharpness = lambda_maps.get('sharpness', torch.zeros_like(denoised))
        lambda_texture = lambda_maps.get('texture', torch.zeros_like(denoised))
        lambda_smooth = lambda_maps.get('smooth', torch.zeros_like(denoised))

        # Build inputs for each corrector
        edge_in = torch.cat([denoised, noisy, lambda_edge, f1], dim=1)
        contrast_in = torch.cat([denoised, noisy, lambda_contrast, f1], dim=1)
        sharpness_in = torch.cat([denoised, noisy, lambda_sharpness, f1], dim=1)
        texture_in = torch.cat([denoised, noisy, lambda_texture, f1, f2], dim=1)
        smooth_in = torch.cat([denoised, noisy, lambda_smooth, f2], dim=1)

        # Apply correctors with immediate memory cleanup after weighting
        # This reduces peak memory by ~40% vs keeping all raw corrections

        # Apply correctors with immediate memory cleanup after weighting
        # This reduces peak memory by ~40% vs keeping all raw corrections

        # Edge correction
        raw_edge = self.edge_corrector(edge_in, denoised)
        edge_correction = lambda_edge * raw_edge
        raw_edge_stats = raw_edge.abs().mean().item(), raw_edge.abs().max().item()
        weighted_edge_stat = edge_correction.abs().mean().item()
        del raw_edge  # Free memory immediately

        # Contrast correction
        raw_contrast = self.contrast_corrector(contrast_in, denoised)
        contrast_correction = lambda_contrast * raw_contrast
        raw_contrast_stats = raw_contrast.abs().mean().item(), raw_contrast.abs().max().item()
        weighted_contrast_stat = contrast_correction.abs().mean().item()
        del raw_contrast

        # Sharpness correction
        raw_sharpness = self.sharpness_corrector(sharpness_in, denoised)
        sharpness_correction = lambda_sharpness * raw_sharpness
        raw_sharpness_stats = raw_sharpness.abs().mean().item(), raw_sharpness.abs().max().item()
        weighted_sharpness_stat = sharpness_correction.abs().mean().item()
        del raw_sharpness

        # Texture correction
        raw_texture = self.texture_corrector(texture_in, denoised)
        texture_correction = lambda_texture * raw_texture
        raw_texture_stats = raw_texture.abs().mean().item(), raw_texture.abs().max().item()
        weighted_texture_stat = texture_correction.abs().mean().item()
        del raw_texture

        # Smooth correction
        raw_smooth = self.smooth_corrector(smooth_in, denoised)
        smooth_correction = lambda_smooth * raw_smooth
        raw_smooth_stats = raw_smooth.abs().mean().item(), raw_smooth.abs().max().item()
        weighted_smooth_stat = smooth_correction.abs().mean().item()
        del raw_smooth

        # Free input tensors no longer needed
        del edge_in, contrast_in, sharpness_in, texture_in, smooth_in

        # Store individual corrections for return_individual mode before combining
        if return_individual:
            individual_corrections = {
                'edge': edge_correction.clone(),
                'contrast': contrast_correction.clone(),
                'sharpness': sharpness_correction.clone(),
                'texture': texture_correction.clone(),
                'smooth': smooth_correction.clone(),
            }

        # Combine corrections incrementally to reduce peak memory
        total_correction = edge_correction
        del edge_correction
        total_correction = total_correction + contrast_correction
        del contrast_correction
        total_correction = total_correction + sharpness_correction
        del sharpness_correction
        total_correction = total_correction + texture_correction
        del texture_correction
        total_correction = total_correction + smooth_correction
        del smooth_correction

        # Safety clamp to prevent catastrophic corrections
        # Tightened to [-0.10, 0.10] to prevent PSNR degradation
        total_correction = total_correction.clamp(-0.10, 0.10)

        # Apply to denoised image
        corrected = (denoised + total_correction).clamp(0, 1)

        # Statistics (using cached stats to avoid keeping raw tensors)
        info['correction_magnitude'] = total_correction.abs().mean().item()
        info['raw_corrections'] = {
            'edge': raw_edge_stats[0],
            'contrast': raw_contrast_stats[0],
            'sharpness': raw_sharpness_stats[0],
            'texture': raw_texture_stats[0],
            'smooth': raw_smooth_stats[0],
        }

        # Detailed logging for tracking correction statistics
        info['raw_corrections_max'] = {
            'edge': raw_edge_stats[1],
            'contrast': raw_contrast_stats[1],
            'sharpness': raw_sharpness_stats[1],
            'texture': raw_texture_stats[1],
            'smooth': raw_smooth_stats[1],
        }

        # Track effective strength (sigmoid(strength) * strength_scale) for each corrector
        info['effective_strengths'] = {
            'edge': (torch.sigmoid(self.edge_corrector.strength) * self.edge_corrector._strength_scale).item(),
            'contrast': (torch.sigmoid(self.contrast_corrector.strength) * self.contrast_corrector._strength_scale).item(),
            'sharpness': (torch.sigmoid(self.sharpness_corrector.strength) * self.sharpness_corrector._strength_scale).item(),
            'texture': (torch.sigmoid(self.texture_corrector.strength) * self.texture_corrector._strength_scale).item(),
            'smooth': (torch.sigmoid(self.smooth_corrector.strength) * self.smooth_corrector._strength_scale).item(),
        }

        # Track lambda-weighted corrections (final contribution) - using cached stats
        info['weighted_corrections'] = {
            'edge': weighted_edge_stat,
            'contrast': weighted_contrast_stat,
            'sharpness': weighted_sharpness_stat,
            'texture': weighted_texture_stat,
            'smooth': weighted_smooth_stat,
        }

        if return_individual:
            info['individual_corrections'] = individual_corrections
            info['lambda_maps'] = lambda_maps

        return corrected, info


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing Powerful Correctors...")
    print("=" * 60)

    # Test individual corrector
    edge = PowerfulEdgeCorrector(in_channels=19)
    print(f"EdgeCorrector params: {sum(p.numel() for p in edge.parameters()):,}")

    contrast = PowerfulContrastCorrector(in_channels=19)
    print(f"ContrastCorrector params: {sum(p.numel() for p in contrast.parameters()):,}")

    sharpness = PowerfulSharpnessCorrector(in_channels=19)
    print(f"SharpnessCorrector params: {sum(p.numel() for p in sharpness.parameters()):,}")

    texture = PowerfulTextureCorrector(in_channels=35)
    print(f"TextureCorrector params: {sum(p.numel() for p in texture.parameters()):,}")

    smooth = PowerfulSmoothCorrector(in_channels=19)
    print(f"SmoothCorrector params: {sum(p.numel() for p in smooth.parameters()):,}")

    # Test full corrector
    corrector = PowerfulAdaptiveCorrectorWithLambda()

    # Test initialization - verify small but non-zero correction at init
    print("\n" + "=" * 60)
    print("INITIALIZATION TEST - Verifying small but non-zero corrections")
    print("=" * 60)

    # Dummy forward
    B, H, W = 2, 64, 64
    denoised = torch.randn(B, 1, H, W).clamp(0, 1)  # Realistic range [0, 1]
    noisy = torch.randn(B, 1, H, W).clamp(0, 1)
    lambda_maps = {
        'edge': torch.rand(B, 1, H, W) * 0.2,
        'contrast': torch.rand(B, 1, H, W) * 0.2,
        'sharpness': torch.rand(B, 1, H, W) * 0.2,
        'texture': torch.rand(B, 1, H, W) * 0.1,
        'smooth': torch.rand(B, 1, H, W) * 0.15,
    }
    backbone_features = {
        'enc1': torch.randn(B, 64, H, W),
        'enc2': torch.randn(B, 128, H//2, W//2),
    }

    with torch.no_grad():
        corrected, info = corrector(denoised, noisy, lambda_maps, backbone_features)

    print(f"\nForward pass successful!")
    print(f"Input shape: {denoised.shape}")
    print(f"Output shape: {corrected.shape}")
    print(f"\nCorrection magnitude at initialization: {info['correction_magnitude']:.6f}")
    print(f"  (Should be small but non-zero, 0.001-0.05 is ideal for gradient flow)")

    print(f"\nRaw corrections at initialization (mean):")
    for name, val in info['raw_corrections'].items():
        print(f"  {name}: {val:.6f}")

    print(f"\nRaw corrections at initialization (max):")
    for name, val in info['raw_corrections_max'].items():
        print(f"  {name}: {val:.6f}")

    print(f"\nEffective strengths (sigmoid(strength) * strength_scale):")
    for name, val in info['effective_strengths'].items():
        print(f"  {name}: {val:.6f}")

    print(f"\nWeighted corrections (after lambda maps):")
    for name, val in info['weighted_corrections'].items():
        print(f"  {name}: {val:.6f}")

    # Verify strength parameter
    print(f"\nStrength parameter values (sigmoid(-2) = {torch.sigmoid(torch.tensor(-2.0)).item():.6f}):")
    print(f"  edge:      strength={corrector.edge_corrector.strength.item():.1f}, effective={info['effective_strengths']['edge']:.6f}")
    print(f"  contrast:  strength={corrector.contrast_corrector.strength.item():.1f}, effective={info['effective_strengths']['contrast']:.6f}")
    print(f"  sharpness: strength={corrector.sharpness_corrector.strength.item():.1f}, effective={info['effective_strengths']['sharpness']:.6f}")
    print(f"  texture:   strength={corrector.texture_corrector.strength.item():.1f}, effective={info['effective_strengths']['texture']:.6f}")
    print(f"  smooth:    strength={corrector.smooth_corrector.strength.item():.1f}, effective={info['effective_strengths']['smooth']:.6f}")

    # Check difference between input and output
    diff = (corrected - denoised).abs().mean().item()
    print(f"\nMean absolute difference (input vs output): {diff:.6f}")
    print(f"  (Should be small but non-zero for gradient flow)")

    # Updated criteria for success:
    # - correction_magnitude should be small but non-zero (0.0001 - 0.1)
    # - diff should also be small but non-zero
    if 0.0001 < info['correction_magnitude'] < 0.1 and 0.0001 < diff < 0.1:
        print("\n[PASS] Initialization is correct - small non-zero corrections for gradient flow!")
    elif info['correction_magnitude'] < 0.0001:
        print("\n[WARNING] Corrections may be too small - check initialization!")
    elif info['correction_magnitude'] > 0.1:
        print("\n[WARNING] Corrections may be too large - check initialization!")
    else:
        print("\n[INFO] Corrections are within expected range.")
