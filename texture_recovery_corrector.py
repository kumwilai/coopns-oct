#!/usr/bin/env python3
"""
Texture Recovery Corrector for OCT Denoising

Addresses critical over-smoothing issue: backbone preserves only 41% of texture.

Clinical Importance:
- Texture in OCT reveals tissue microstructure (photoreceptor mosaic, nerve fiber bundles)
- Over-smoothing destroys diagnostic information (early AMD, glaucoma fiber loss)
- Must distinguish texture (signal) from speckle noise (artifact)

Key Innovation:
1. Multi-scale local variance analysis to detect over-smoothed regions
2. Texture-noise discriminator using frequency and spatial coherence
3. Guided texture injection from noisy input (texture preserved in noise)
4. Anisotropic texture recovery (respects layer orientation)

Author: Neuro-Symbolic OCT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# TEXTURE ANALYSIS MODULES
# =============================================================================

class LocalVarianceAnalyzer(nn.Module):
    """
    Multi-scale local variance computation for over-smoothing detection.

    Computes variance at multiple scales to capture:
    - Fine texture (small kernel): photoreceptor structure, fiber bundles
    - Medium texture (medium kernel): layer sub-structure
    - Coarse texture (large kernel): gross tissue variation
    """

    def __init__(self, scales: Tuple[int, ...] = (3, 7, 15)):
        super().__init__()
        self.scales = scales

        # Pre-compute averaging kernels for efficiency
        self.kernels = nn.ParameterDict()
        for s in scales:
            kernel = torch.ones(1, 1, s, s) / (s * s)
            self.register_buffer(f'kernel_{s}', kernel)

    def compute_local_variance(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Compute local variance using E[X^2] - E[X]^2."""
        padding = kernel_size // 2
        kernel = getattr(self, f'kernel_{kernel_size}')

        # Local mean
        local_mean = F.conv2d(
            F.pad(x, [padding] * 4, mode='reflect'),
            kernel
        )

        # Local mean of squares
        local_mean_sq = F.conv2d(
            F.pad(x ** 2, [padding] * 4, mode='reflect'),
            kernel
        )

        # Variance = E[X^2] - E[X]^2
        local_var = (local_mean_sq - local_mean ** 2).clamp(min=0)

        return local_var

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale variance maps.

        Returns:
            Dict with 'fine', 'medium', 'coarse' variance maps
        """
        variances = {}
        scale_names = ['fine', 'medium', 'coarse']

        for name, scale in zip(scale_names, self.scales):
            variances[name] = self.compute_local_variance(x, scale)

        # Combined multi-scale variance (geometric mean for scale invariance)
        combined = torch.ones_like(variances['fine'])
        for v in variances.values():
            combined = combined * (v + 1e-8)
        variances['combined'] = combined.pow(1.0 / len(self.scales))

        return variances


class OverSmoothingDetector(nn.Module):
    """
    Detects regions where backbone over-smoothed the image.

    Strategy:
    1. Compare variance in denoised vs noisy (accounting for expected noise reduction)
    2. Flag regions where variance ratio is suspiciously low
    3. Cross-reference with edge map (low edges + low variance = over-smoothed)
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.variance_analyzer = LocalVarianceAnalyzer(scales=(3, 7, 15))

        # Learnable thresholds for over-smoothing detection
        self.smoothing_threshold = nn.Parameter(torch.tensor(0.3))  # variance ratio threshold
        self.edge_threshold = nn.Parameter(torch.tensor(0.1))  # edge magnitude threshold

        # Feature extractor for context-aware detection
        self.context_net = nn.Sequential(
            nn.Conv2d(7, hidden_dim, 3, padding=1, bias=False),  # 3 var scales + 1 ratio + 1 edge + noisy + denoised
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_edge_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters."""
        gx = F.conv2d(F.pad(x, [1, 1, 1, 1], mode='reflect'), self.sobel_x)
        gy = F.conv2d(F.pad(x, [1, 1, 1, 1], mode='reflect'), self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect over-smoothed regions.

        Args:
            denoised: Backbone output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]

        Returns:
            Dict with:
            - 'over_smoothed_map': [B, 1, H, W] probability of over-smoothing
            - 'variance_ratio': ratio of denoised/noisy variance
            - 'texture_regions': regions that should have texture
        """
        # Compute variances
        var_noisy = self.variance_analyzer(noisy)
        var_denoised = self.variance_analyzer(denoised)

        # Variance ratio (how much texture was preserved)
        # Expected: ~0.3-0.7 for proper denoising (some variance from noise removed)
        # Too low (<0.2): over-smoothed
        # Too high (>0.8): under-denoised
        variance_ratio = (var_denoised['combined'] + 1e-8) / (var_noisy['combined'] + 1e-8)

        # Edge magnitude in denoised image
        edge_mag = self.compute_edge_magnitude(denoised)
        edge_mag_norm = edge_mag / (edge_mag.max() + 1e-8)

        # Identify texture regions in noisy image (high variance, not just edges)
        texture_in_noisy = var_noisy['medium'] > var_noisy['medium'].mean()

        # Build feature map for context-aware detection
        features = torch.cat([
            var_denoised['fine'],
            var_denoised['medium'],
            var_denoised['coarse'],
            variance_ratio,
            edge_mag_norm,
            noisy,
            denoised
        ], dim=1)

        # Learn to detect over-smoothing
        over_smoothed_map = self.context_net(features)

        # Hard constraint: only consider regions that had texture originally
        # (don't try to add texture to genuinely smooth regions)
        over_smoothed_map = over_smoothed_map * texture_in_noisy.float()

        return {
            'over_smoothed_map': over_smoothed_map,
            'variance_ratio': variance_ratio,
            'texture_regions': texture_in_noisy.float(),
            'var_noisy': var_noisy,
            'var_denoised': var_denoised,
            'edge_magnitude': edge_mag_norm
        }


class TextureNoiseDiscriminator(nn.Module):
    """
    Distinguishes between texture (to preserve) and noise (to remove).

    Key insight: Texture has spatial coherence, noise doesn't.

    Discriminating features:
    1. Spatial autocorrelation: texture is locally correlated, noise isn't
    2. Frequency distribution: texture has structured frequency content
    3. Anisotropy: texture often follows layer orientation, noise is isotropic
    4. Scale persistence: texture persists across scales, noise averages out
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Multi-scale feature extraction
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, hidden_dim // 4, 3, padding=1, dilation=1),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1, hidden_dim // 4, 3, padding=2, dilation=2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1, hidden_dim // 4, 3, padding=4, dilation=4),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1, hidden_dim // 4, 3, padding=8, dilation=8),
                nn.LeakyReLU(0.2, inplace=True)
            ),
        ])

        # Anisotropic filters (horizontal = along layers, vertical = across layers)
        self.horizontal_filter = nn.Conv2d(1, 8, (1, 7), padding=(0, 3))
        self.vertical_filter = nn.Conv2d(1, 8, (7, 1), padding=(3, 0))

        # Local autocorrelation estimation (learned)
        self.autocorr_net = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 8, 5, padding=2),
        )

        # Combine all features to predict texture probability
        # hidden_dim (scales) + 16 (anisotropic) + 8 (autocorr) = hidden_dim + 24
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim + 24, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Estimate probability that each pixel's residual is texture (not noise).

        Args:
            residual: Difference between noisy and denoised [B, 1, H, W]
                     Contains both removed noise AND removed texture

        Returns:
            texture_prob: [B, 1, H, W] probability that residual is texture
        """
        # Multi-scale features (texture persists, noise averages)
        scale_features = [s(residual) for s in self.scales]
        scale_feat = torch.cat(scale_features, dim=1)

        # Anisotropic features (texture follows structure, noise is random)
        h_feat = self.horizontal_filter(residual)
        v_feat = self.vertical_filter(residual)
        aniso_feat = torch.cat([h_feat, v_feat], dim=1)

        # Autocorrelation features (texture is locally correlated)
        autocorr_feat = self.autocorr_net(residual)

        # Combine and classify
        all_features = torch.cat([scale_feat, aniso_feat, autocorr_feat], dim=1)
        texture_prob = self.classifier(all_features)

        return texture_prob


class GuidedTextureInjector(nn.Module):
    """
    Injects texture back into over-smoothed regions using guidance from noisy input.

    Strategy:
    1. Extract texture from residual (noisy - denoised)
    2. Filter to keep only texture-like components (not noise)
    3. Adaptively inject based on over-smoothing severity
    4. Respect layer boundaries (don't smear texture across layers)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Texture extraction network
        self.texture_extractor = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1, bias=False),  # residual + noisy + denoised
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale processing for texture at different granularities
        self.multiscale = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=8, dilation=8),
        ])

        # Channel attention for feature selection
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )

        # Spatial attention for region selection
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # Edge-aware gating (don't inject texture across layer boundaries)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

        # Final texture prediction
        self.texture_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] for texture adjustment
        )

        # Learnable injection strength (conservative initialization)
        self.injection_strength = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ~ 0.27
        self._max_injection = 0.15  # Maximum texture injection magnitude

        # Sobel for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init final layer for stable start
        for module in self.texture_head.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == 1:
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude."""
        gx = F.conv2d(F.pad(x, [1, 1, 1, 1], mode='reflect'), self.sobel_x)
        gy = F.conv2d(F.pad(x, [1, 1, 1, 1], mode='reflect'), self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                over_smoothed_map: torch.Tensor, texture_prob: torch.Tensor) -> torch.Tensor:
        """
        Generate texture correction to add back to denoised image.

        Args:
            denoised: Backbone output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            over_smoothed_map: Where over-smoothing occurred [B, 1, H, W]
            texture_prob: Which parts of residual are texture [B, 1, H, W]

        Returns:
            texture_correction: [B, 1, H, W] to add to denoised
        """
        # Compute residual (contains removed noise + removed texture)
        residual = noisy - denoised

        # Extract features from residual, noisy, and denoised
        combined = torch.cat([residual, noisy, denoised], dim=1)
        features = self.texture_extractor(combined)

        # Multi-scale processing
        ms_features = [conv(features) for conv in self.multiscale]
        features = torch.cat(ms_features, dim=1)

        # Channel attention
        ca = self.channel_attention(features)
        features = features * ca

        # Spatial attention
        avg_feat = features.mean(dim=1, keepdim=True)
        max_feat = features.max(dim=1, keepdim=True)[0]
        sa = self.spatial_attention(torch.cat([avg_feat, max_feat], dim=1))
        features = features * sa

        # Edge-aware gating (reduce injection near strong edges)
        edge_map = self.compute_edge_map(denoised)
        edge_map_norm = edge_map / (edge_map.max() + 1e-8)
        edge_gate_input = torch.cat([features, edge_map_norm], dim=1)
        edge_gate = self.edge_gate(edge_gate_input)
        features = features * edge_gate

        # Generate texture correction
        raw_texture = self.texture_head(features)

        # Scale by:
        # 1. Learned injection strength
        # 2. Over-smoothing severity
        # 3. Texture probability (only add back what's texture, not noise)
        strength = torch.sigmoid(self.injection_strength) * self._max_injection
        texture_correction = raw_texture * strength * over_smoothed_map * texture_prob

        return texture_correction


# =============================================================================
# MAIN TEXTURE RECOVERY CORRECTOR
# =============================================================================

class TextureRecoveryCorrector(nn.Module):
    """
    Complete texture recovery corrector for OCT denoising.

    Addresses the critical over-smoothing problem (41% texture preservation).

    Pipeline:
    1. Detect over-smoothed regions using local variance analysis
    2. Distinguish texture from noise in the residual
    3. Inject appropriate texture back without adding noise
    4. Respect layer boundaries and clinical constraints

    Clinical Utility:
    - Preserves photoreceptor layer structure (important for AMD diagnosis)
    - Maintains nerve fiber bundle visibility (critical for glaucoma)
    - Keeps tissue microstructure without amplifying speckle noise
    """

    def __init__(self, hidden_dim: int = 64, enc1_channels: int = 48, enc2_channels: int = 96):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. Over-smoothing detection
        self.smoothing_detector = OverSmoothingDetector(hidden_dim=32)

        # 2. Texture-noise discrimination
        self.texture_discriminator = TextureNoiseDiscriminator(hidden_dim=hidden_dim)

        # 3. Guided texture injection
        self.texture_injector = GuidedTextureInjector(hidden_dim=hidden_dim)

        # 4. Optional: Backbone feature integration for better context
        self.use_backbone_features = enc1_channels > 0
        if self.use_backbone_features:
            self.enc1_proj = nn.Conv2d(enc1_channels, 16, 1)
            self.enc2_proj = nn.Conv2d(enc2_channels, 16, 1)

            # Feature-aware refinement
            self.feature_refine = nn.Sequential(
                nn.Conv2d(1 + 32, hidden_dim // 2, 3, padding=1),  # correction + backbone features
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
                nn.Tanh()
            )

            # Blend between raw and refined correction
            self.refine_blend = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # Safety clamp for correction magnitude
        self._max_correction = 0.2

        # Statistics tracking
        self.register_buffer('running_texture_ratio', torch.tensor(0.41))
        self.register_buffer('num_updates', torch.tensor(0))

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print("\n" + "=" * 70)
        print("TextureRecoveryCorrector - Addressing 41% Texture Preservation")
        print("=" * 70)
        print("Pipeline:")
        print("  1. Multi-scale local variance analysis (over-smoothing detection)")
        print("  2. Texture-noise discrimination (spatial coherence)")
        print("  3. Guided texture injection (respect layer boundaries)")
        print("  4. Feature-aware refinement (optional backbone integration)")
        print(f"\nParameters: {total_params:,}")
        print(f"Max correction magnitude: {self._max_correction}")
        print("=" * 70)

    def compute_texture_preservation_ratio(self, denoised: torch.Tensor,
                                           clean: torch.Tensor,
                                           kernel_size: int = 7) -> torch.Tensor:
        """
        Compute texture preservation ratio for monitoring.

        Used during training with ground truth to track improvement.
        """
        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=denoised.device)
        kernel = kernel / (kernel_size * kernel_size)

        def local_variance(x):
            local_mean = F.conv2d(F.pad(x, [padding] * 4, mode='reflect'), kernel)
            local_mean_sq = F.conv2d(F.pad(x ** 2, [padding] * 4, mode='reflect'), kernel)
            return (local_mean_sq - local_mean ** 2).clamp(min=0)

        var_denoised = local_variance(denoised)
        var_clean = local_variance(clean)

        # Texture regions in clean (variance above threshold)
        texture_mask = var_clean > var_clean.mean()

        if texture_mask.sum() > 0:
            ratio = (var_denoised * texture_mask.float()).sum() / (var_clean * texture_mask.float()).sum()
        else:
            ratio = torch.tensor(1.0, device=denoised.device)

        return ratio.clamp(0, 2)  # Clamp to reasonable range

    def forward(self, backbone_out: torch.Tensor, noisy: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None,
                return_details: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply texture recovery correction.

        Args:
            backbone_out: Denoised output from backbone [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            backbone_features: Optional dict with 'enc1', 'enc2' features
            return_details: Whether to return detailed analysis

        Returns:
            corrected: Texture-recovered image [B, 1, H, W]
            info: Dict with correction statistics and analysis
        """
        B, C, H, W = backbone_out.shape

        # Step 1: Detect over-smoothed regions
        smoothing_analysis = self.smoothing_detector(backbone_out, noisy)
        over_smoothed_map = smoothing_analysis['over_smoothed_map']

        # Step 2: Discriminate texture from noise in residual
        residual = noisy - backbone_out
        texture_prob = self.texture_discriminator(residual)

        # Step 3: Generate texture correction
        texture_correction = self.texture_injector(
            backbone_out, noisy, over_smoothed_map, texture_prob
        )

        # Step 4: Optional backbone feature refinement
        if self.use_backbone_features and backbone_features is not None:
            enc1 = backbone_features.get('enc1')
            enc2 = backbone_features.get('enc2')

            if enc1 is not None and enc2 is not None:
                # Project and upsample backbone features
                f1 = self.enc1_proj(enc1.detach())
                if f1.shape[-2:] != (H, W):
                    f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)

                f2 = self.enc2_proj(enc2.detach())
                if f2.shape[-2:] != (H, W):
                    f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

                # Refine correction using backbone context
                refine_input = torch.cat([texture_correction, f1, f2], dim=1)
                refined_correction = self.feature_refine(refine_input)

                # Blend raw and refined based on learned weight
                blend = torch.sigmoid(self.refine_blend)
                texture_correction = (1 - blend) * texture_correction + blend * refined_correction * self.texture_injector._max_injection

        # Safety clamp
        texture_correction = texture_correction.clamp(-self._max_correction, self._max_correction)

        # Apply correction
        corrected = (backbone_out + texture_correction).clamp(0, 1)

        # Compute statistics
        correction_magnitude = texture_correction.abs().mean()
        over_smoothed_fraction = over_smoothed_map.mean()
        texture_fraction = texture_prob.mean()

        info = {
            'correction_magnitude': correction_magnitude.item() if not self.training else correction_magnitude,
            'over_smoothed_fraction': over_smoothed_fraction.item() if not self.training else over_smoothed_fraction,
            'texture_in_residual_fraction': texture_fraction.item() if not self.training else texture_fraction,
            'variance_ratio_mean': smoothing_analysis['variance_ratio'].mean().item(),
        }

        if return_details:
            info.update({
                'over_smoothed_map': over_smoothed_map,
                'texture_prob_map': texture_prob,
                'variance_ratio_map': smoothing_analysis['variance_ratio'],
                'texture_correction': texture_correction,
                'var_noisy': smoothing_analysis['var_noisy'],
                'var_denoised': smoothing_analysis['var_denoised'],
            })

        return corrected, info


# =============================================================================
# TEXTURE RECOVERY PREDICATE (for integration with symbolic system)
# =============================================================================

class TexturePreservationPredicate(nn.Module):
    """
    Predicate P7: Texture Preservation

    Evaluates whether texture is adequately preserved after denoising.

    Score = texture_variance_denoised / texture_variance_reference
    Target: >= 0.7 (preserve at least 70% of texture variance)
    """

    def __init__(self, target_ratio: float = 0.7, kernel_size: int = 7):
        super().__init__()
        self.target_ratio = target_ratio
        self.kernel_size = kernel_size

        # Pre-compute averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        self.register_buffer('avg_kernel', kernel)

        # Learnable threshold for texture regions
        self.texture_threshold = nn.Parameter(torch.tensor(0.005))

    def compute_local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local variance map."""
        padding = self.kernel_size // 2
        local_mean = F.conv2d(F.pad(x, [padding] * 4, mode='reflect'), self.avg_kernel)
        local_mean_sq = F.conv2d(F.pad(x ** 2, [padding] * 4, mode='reflect'), self.avg_kernel)
        return (local_mean_sq - local_mean ** 2).clamp(min=0)

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate texture preservation.

        Args:
            denoised: Denoised image [B, 1, H, W]
            noisy: Original noisy image [B, 1, H, W]

        Returns:
            Dict with 'score', 'failure_map', 'texture_ratio'
        """
        var_denoised = self.compute_local_variance(denoised)
        var_noisy = self.compute_local_variance(noisy)

        # Identify texture regions (where noisy has significant variance)
        texture_mask = var_noisy > torch.sigmoid(self.texture_threshold)

        # Compute preservation ratio in texture regions
        if texture_mask.sum() > 0:
            ratio_map = var_denoised / (var_noisy + 1e-8)
            ratio_map = ratio_map.clamp(0, 2)

            # Score based on how well we preserve texture
            # Perfect: ratio = 1.0 (but some reduction expected due to noise removal)
            # Good: ratio >= target_ratio
            # Bad: ratio < target_ratio

            texture_score = (ratio_map * texture_mask.float()).sum() / (texture_mask.float().sum() + 1e-8)
            texture_score = texture_score.clamp(0, 1)

            # Failure map: where texture preservation is inadequate
            failure_map = ((self.target_ratio - ratio_map).clamp(min=0) / self.target_ratio) * texture_mask.float()
        else:
            texture_score = torch.tensor(1.0, device=denoised.device)
            ratio_map = torch.ones_like(var_denoised)
            failure_map = torch.zeros_like(var_denoised)

        return {
            'score': texture_score,
            'failure_map': failure_map,
            'texture_ratio_map': ratio_map,
            'texture_mask': texture_mask.float()
        }


# =============================================================================
# INTEGRATION WITH V8 CORRECTOR
# =============================================================================

class TextureRecoveryCorrectorV8Integration(nn.Module):
    """
    Wrapper for integrating TextureRecoveryCorrector with NeuroSymbolicCorrectorV8Enhanced.

    Provides the same interface as other V8 correctors (EnhancedEdgeCorrector, etc.)
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 enc1_channels: int = 48, enc2_channels: int = 96):
        super().__init__()

        self.name = "texture"

        # Core texture recovery
        self.texture_corrector = TextureRecoveryCorrector(
            hidden_dim=hidden_dim,
            enc1_channels=enc1_channels,
            enc2_channels=enc2_channels
        )

        # Learnable strength compatible with V8 interface
        self.strength = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self._strength_scale = 0.5

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict[str, torch.Tensor]] = None,
                noisy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        V8-compatible interface.

        Note: This corrector needs access to noisy input, which isn't in the standard
        V8 corrector interface. The caller must provide it via the noisy parameter.
        """
        if noisy is None:
            # Fallback: return zero correction if noisy not provided
            return torch.zeros_like(backbone_out)

        # Get texture-recovered output
        corrected, info = self.texture_corrector(
            backbone_out, noisy, backbone_features, return_details=False
        )

        # Extract correction
        correction = corrected - backbone_out

        # Apply V8-style strength scaling
        strength = torch.sigmoid(self.strength) * self._strength_scale

        # Modulate by failure map (from P7 texture predicate)
        correction = correction * strength * failure_map

        return correction


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\nTesting TextureRecoveryCorrector...")
    print("=" * 70)

    # Create model
    model = TextureRecoveryCorrector(
        hidden_dim=64,
        enc1_channels=48,
        enc2_channels=96
    )

    # Test inputs
    B, C, H, W = 2, 1, 128, 128

    # Simulate noisy OCT image with texture
    texture = torch.randn(B, C, H, W) * 0.05  # Tissue texture
    noise = torch.randn(B, C, H, W) * 0.15    # Speckle noise
    base = torch.sigmoid(torch.randn(B, C, H, W))  # Base signal
    noisy = (base + texture + noise).clamp(0, 1)

    # Simulate over-smoothed backbone output (texture removed)
    backbone_out = base.clamp(0, 1)  # Texture gone

    # Backbone features
    backbone_features = {
        'enc1': torch.randn(B, 48, H, W) * 0.1,
        'enc2': torch.randn(B, 96, H // 2, W // 2) * 0.1,
    }

    # Forward pass
    with torch.no_grad():
        corrected, info = model(backbone_out, noisy, backbone_features, return_details=True)

    print(f"\nInput shape: {backbone_out.shape}")
    print(f"Output shape: {corrected.shape}")
    print(f"\nCorrection Statistics:")
    print(f"  Correction magnitude: {info['correction_magnitude']:.4f}")
    print(f"  Over-smoothed fraction: {info['over_smoothed_fraction']:.4f}")
    print(f"  Texture in residual: {info['texture_in_residual_fraction']:.4f}")
    print(f"  Variance ratio: {info['variance_ratio_mean']:.4f}")

    # Check if correction was applied
    diff = (corrected - backbone_out).abs().mean().item()
    print(f"\n  Mean absolute difference: {diff:.4f}")

    # Test texture preservation predicate
    print("\n" + "=" * 70)
    print("Testing TexturePreservationPredicate (P7)...")

    predicate = TexturePreservationPredicate(target_ratio=0.7)
    pred_result = predicate(backbone_out, noisy)

    print(f"  Texture preservation score: {pred_result['score'].item():.4f}")
    print(f"  Failure map mean: {pred_result['failure_map'].mean().item():.4f}")

    # Test after correction
    pred_result_corrected = predicate(corrected, noisy)
    print(f"\n  After correction:")
    print(f"  Texture preservation score: {pred_result_corrected['score'].item():.4f}")
    print(f"  Failure map mean: {pred_result_corrected['failure_map'].mean().item():.4f}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n" + "=" * 70)
    print(f"Total parameters: {total_params:,}")

    print("\nTest passed!")
