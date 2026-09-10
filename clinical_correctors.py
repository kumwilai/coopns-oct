#!/usr/bin/env python3
"""
Clinical Correctors for OCT Denoising

Based on backbone weakness analysis:
- Contrast: Only 47% preserved (53% lost!)
- Boundary Sharpness: Only 47% preserved
- Edge Strength: 32% of edges weakened
- Texture: Only 41% preserved (severe over-smoothing)

These correctors target CLINICAL UTILITY, not PSNR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# =============================================================================
# CONTRAST RESTORATION CORRECTOR
# =============================================================================

class ContrastRestorationCorrector(nn.Module):
    """
    Restore lost local contrast in OCT images.

    Problem: Backbone preserves only 47% of local contrast.
    Solution: Detect low-contrast regions and boost local standard deviation.

    Clinical Importance: Contrast between layers enables diagnosis.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.name = "contrast"

        # Local contrast analyzer
        self.window_size = 15

        # Contrast enhancement network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),  # denoised + noisy + contrast_ratio
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale contrast enhancement
        self.scales = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=8, dilation=8),
        ])

        # Contrast boost predictor
        self.boost_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

        # Learnable max boost
        self.max_boost = nn.Parameter(torch.tensor(0.3))

        self._init_weights()

    def _init_weights(self):
        # Zero-init output for stable start
        nn.init.zeros_(self.boost_head[-1].weight)
        nn.init.zeros_(self.boost_head[-1].bias)

    def compute_local_contrast(self, img: torch.Tensor) -> torch.Tensor:
        """Compute local standard deviation as contrast measure."""
        B, C, H, W = img.shape
        padding = self.window_size // 2

        # Local mean
        kernel = torch.ones(1, 1, self.window_size, self.window_size,
                           device=img.device, dtype=img.dtype)
        kernel = kernel / (self.window_size ** 2)

        local_mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
        local_mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)

        local_var = (local_mean_sq - local_mean**2).clamp(min=0)
        local_std = torch.sqrt(local_var + 1e-8)

        return local_std

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict] = None) -> torch.Tensor:
        """
        Generate contrast restoration correction.

        Args:
            backbone_out: Denoised output [B, 1, H, W]
            failure_map: Contrast failure map [B, 1, H, W]
            backbone_features: Optional encoder features

        Returns:
            correction: Additive contrast correction [B, 1, H, W]
        """
        B, C, H, W = backbone_out.shape

        # Compute contrast ratio (how much was lost)
        # We estimate "expected" contrast from analyzing the backbone output
        local_contrast = self.compute_local_contrast(backbone_out)

        # Use failure map as guide for where contrast is needed
        x = torch.cat([backbone_out, local_contrast, failure_map], dim=1)

        # Encode
        feat = self.encoder(x)

        # Multi-scale processing
        ms_feats = [scale(feat) for scale in self.scales]
        feat = torch.cat(ms_feats, dim=1)

        # Predict contrast boost
        boost = self.boost_head(feat)

        # Apply boost relative to local mean (stretches contrast)
        local_mean = F.avg_pool2d(F.pad(backbone_out, [7]*4, mode='reflect'),
                                   15, stride=1)

        # Correction pushes pixels away from local mean
        deviation = backbone_out - local_mean
        max_b = torch.sigmoid(self.max_boost)
        correction = boost * deviation * failure_map * max_b

        return correction.clamp(-0.3, 0.3)


# =============================================================================
# BOUNDARY SHARPNESS CORRECTOR
# =============================================================================

class BoundarySharpnessCorrector(nn.Module):
    """
    Restore sharpness at layer boundaries.

    Problem: Backbone preserves only 47% of boundary sharpness.
    Solution: Detect blurred boundaries and apply targeted sharpening.

    Clinical Importance: Sharp boundaries enable layer thickness measurement.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.name = "boundary"

        # Vertical gradient kernels (detect horizontal boundaries)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)

        # Boundary enhancement network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),  # denoised + gradient + failure
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Vertical-focused processing (for horizontal layers)
        self.vert_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Sharpening head
        self.sharpen_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

        # Laplacian for unsharp masking
        self.register_buffer('laplacian', torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)

        self.max_sharpen = nn.Parameter(torch.tensor(0.25))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.sharpen_head[-1].weight)
        nn.init.zeros_(self.sharpen_head[-1].bias)

    def compute_vertical_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Compute vertical gradient magnitude."""
        pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
        grad = F.conv2d(pad, self.sobel_y.to(img.device))
        return grad.abs()

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict] = None) -> torch.Tensor:
        """Generate boundary sharpening correction."""
        B, C, H, W = backbone_out.shape

        # Compute gradient
        gradient = self.compute_vertical_gradient(backbone_out)

        # Combine inputs
        x = torch.cat([backbone_out, gradient, failure_map], dim=1)

        # Encode
        feat = self.encoder(x)

        # Vertical processing
        feat = self.vert_conv(feat)

        # Predict sharpening
        sharpen = self.sharpen_head(feat)

        # Add Laplacian-based sharpening
        laplacian = F.conv2d(F.pad(backbone_out, [1]*4, mode='reflect'),
                             self.laplacian.to(backbone_out.device))

        # Combine learned + classical sharpening
        max_s = torch.sigmoid(self.max_sharpen)
        correction = (sharpen + 0.3 * laplacian) * failure_map * max_s

        return correction.clamp(-0.3, 0.3)


# =============================================================================
# TEXTURE RECOVERY CORRECTOR
# =============================================================================

class TextureRecoveryCorrector(nn.Module):
    """
    Recover texture lost to over-smoothing.

    Problem: Backbone preserves only 41% of texture.
    Solution: Detect over-smoothed regions and restore appropriate texture.

    Clinical Importance: Texture reveals tissue microstructure.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.name = "texture"

        # Texture analysis network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),  # denoised + noisy_texture + failure
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # High-frequency extraction
        self.hf_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Texture synthesis head
        self.texture_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, 1),
        )

        # Texture amount predictor
        self.amount_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        self.max_texture = nn.Parameter(torch.tensor(0.15))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.texture_head[-1].weight)
        nn.init.zeros_(self.texture_head[-1].bias)

    def compute_local_variance(self, img: torch.Tensor, size: int = 7) -> torch.Tensor:
        """Compute local variance as texture measure."""
        padding = size // 2
        kernel = torch.ones(1, 1, size, size, device=img.device, dtype=img.dtype)
        kernel = kernel / (size ** 2)

        mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
        mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)

        var = (mean_sq - mean**2).clamp(min=0)
        return var

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict] = None) -> torch.Tensor:
        """Generate texture recovery correction."""
        B, C, H, W = backbone_out.shape

        # Compute texture measure
        texture_var = self.compute_local_variance(backbone_out)

        # Combine inputs
        x = torch.cat([backbone_out, texture_var, failure_map], dim=1)

        # Encode
        feat = self.encoder(x)

        # Predict texture amount needed
        amount = self.amount_head(feat)

        # Extract high-frequency features
        hf_feat = self.hf_branch(feat)

        # Generate texture
        texture = self.texture_head(hf_feat)

        # Apply with learned amount
        max_t = torch.sigmoid(self.max_texture)
        correction = texture * amount * failure_map * max_t

        return correction.clamp(-0.2, 0.2)


# =============================================================================
# EDGE ENHANCEMENT CORRECTOR
# =============================================================================

class EdgeEnhancementCorrector(nn.Module):
    """
    Enhance weakened edges.

    Problem: 32% of edges are weakened by backbone.
    Solution: Detect weak edges and strengthen them.

    Clinical Importance: Edges define anatomical structures.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.name = "edge"

        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)

        # Edge enhancement network
        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1),  # denoised + grad_x + grad_y + failure
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Direction-aware processing
        self.dir_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Edge enhancement head
        self.edge_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

        self.max_enhance = nn.Parameter(torch.tensor(0.2))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)

    def compute_gradients(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute x and y gradients."""
        pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
        gx = F.conv2d(pad, self.sobel_x.to(img.device))
        gy = F.conv2d(pad, self.sobel_y.to(img.device))
        return gx, gy

    def forward(self, backbone_out: torch.Tensor, failure_map: torch.Tensor,
                backbone_features: Optional[Dict] = None) -> torch.Tensor:
        """Generate edge enhancement correction."""
        B, C, H, W = backbone_out.shape

        # Compute gradients
        gx, gy = self.compute_gradients(backbone_out)

        # Combine inputs
        x = torch.cat([backbone_out, gx, gy, failure_map], dim=1)

        # Encode
        feat = self.encoder(x)

        # Direction-aware processing
        feat = self.dir_branch(feat)

        # Predict enhancement
        enhance = self.edge_head(feat)

        # Apply at failure locations
        max_e = torch.sigmoid(self.max_enhance)
        correction = enhance * failure_map * max_e

        return correction.clamp(-0.25, 0.25)


# =============================================================================
# CLINICAL PREDICATES (GT-Free)
# =============================================================================

class ClinicalPredicates(nn.Module):
    """
    GT-free predicates focused on clinical quality.

    These predicates measure clinically-relevant properties without
    requiring ground truth, enabling real-world deployment.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0)

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor) -> Dict:
        """
        Evaluate clinical predicates.

        Returns dict with scores and failure maps for:
        - P_contrast: Local contrast preservation
        - P_boundary: Boundary sharpness
        - P_texture: Texture preservation
        - P_edge: Edge strength
        """
        B, C, H, W = denoised.shape
        device = denoised.device
        results = {'scores': {}}

        # === P_contrast: Local contrast ===
        contrast_denoised = self._local_std(denoised)
        contrast_noisy = self._local_std(noisy)

        # Ratio of preserved contrast
        contrast_ratio = contrast_denoised / (contrast_noisy + 1e-6)
        contrast_score = contrast_ratio.mean().clamp(0, 1)

        # Failure where contrast reduced below 70%
        contrast_failure = (1 - contrast_ratio).clamp(0, 1)
        contrast_failure = contrast_failure * (contrast_ratio < 0.7).float()

        results['P_contrast'] = {
            'score': contrast_score,
            'failure_map': contrast_failure,
        }
        results['scores']['P_contrast'] = contrast_score.item()

        # === P_boundary: Boundary sharpness ===
        grad_denoised = self._vertical_gradient(denoised)
        grad_noisy = self._vertical_gradient(noisy)

        # Focus on boundary regions (use threshold instead of quantile for stability)
        grad_threshold = grad_noisy.mean() + grad_noisy.std()
        boundary_mask = grad_noisy > grad_threshold

        if boundary_mask.sum() > 10:  # Need minimum pixels
            boundary_ratio = grad_denoised[boundary_mask].mean() / (grad_noisy[boundary_mask].mean() + 1e-6)
            boundary_ratio = boundary_ratio.clamp(0, 2)  # Clamp for stability
        else:
            boundary_ratio = torch.tensor(0.5, device=device)

        boundary_score = boundary_ratio.clamp(0, 1)

        # Failure where gradients significantly reduced
        boundary_failure = ((grad_noisy - grad_denoised) / (grad_noisy + 1e-6)).clamp(0, 1)
        boundary_failure = boundary_failure * boundary_mask.float()

        results['P_boundary'] = {
            'score': boundary_score,
            'failure_map': boundary_failure,
        }
        results['scores']['P_boundary'] = boundary_score.item() if isinstance(boundary_score, torch.Tensor) else boundary_score

        # === P_texture: Texture preservation ===
        var_denoised = self._local_variance(denoised, size=5)
        var_noisy = self._local_variance(noisy, size=5)

        # Texture regions (moderate variance in noisy)
        texture_mask = (var_noisy > 0.001) & (var_noisy < 0.05)

        if texture_mask.sum() > 10:  # Need minimum pixels
            texture_ratio = var_denoised[texture_mask].mean() / (var_noisy[texture_mask].mean() + 1e-6)
            texture_ratio = texture_ratio.clamp(0, 2)  # Clamp for stability
        else:
            texture_ratio = torch.tensor(0.5, device=device)

        texture_score = texture_ratio.clamp(0, 1)

        # Failure where texture lost
        texture_failure = ((var_noisy - var_denoised) / (var_noisy + 1e-6)).clamp(0, 1)
        texture_failure = texture_failure * texture_mask.float()

        results['P_texture'] = {
            'score': texture_score,
            'failure_map': texture_failure,
        }
        results['scores']['P_texture'] = texture_score.item() if isinstance(texture_score, torch.Tensor) else texture_score

        # === P_edge: Edge strength ===
        edge_denoised = self._edge_magnitude(denoised)
        edge_noisy = self._edge_magnitude(noisy)

        # Strong edge regions (use threshold instead of quantile)
        edge_threshold = edge_noisy.mean() + 0.5 * edge_noisy.std()
        edge_mask = edge_noisy > edge_threshold

        if edge_mask.sum() > 10:  # Need minimum pixels
            edge_ratio = edge_denoised[edge_mask].mean() / (edge_noisy[edge_mask].mean() + 1e-6)
            edge_ratio = edge_ratio.clamp(0, 2)  # Clamp for stability
        else:
            edge_ratio = torch.tensor(0.5, device=device)

        edge_score = edge_ratio.clamp(0, 1)

        # Failure where edges weakened
        edge_failure = ((edge_noisy - edge_denoised) / (edge_noisy + 1e-6)).clamp(0, 1)
        edge_failure = edge_failure * edge_mask.float()

        results['P_edge'] = {
            'score': edge_score,
            'failure_map': edge_failure,
        }
        results['scores']['P_edge'] = edge_score.item() if isinstance(edge_score, torch.Tensor) else edge_score

        return results

    def _local_std(self, img: torch.Tensor, size: int = 15) -> torch.Tensor:
        """Compute local standard deviation."""
        padding = size // 2
        kernel = torch.ones(1, 1, size, size, device=img.device, dtype=img.dtype) / (size**2)

        mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
        mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)

        var = (mean_sq - mean**2).clamp(min=0)
        return torch.sqrt(var + 1e-8)

    def _local_variance(self, img: torch.Tensor, size: int = 7) -> torch.Tensor:
        """Compute local variance."""
        padding = size // 2
        kernel = torch.ones(1, 1, size, size, device=img.device, dtype=img.dtype) / (size**2)

        mean = F.conv2d(F.pad(img, [padding]*4, mode='reflect'), kernel)
        mean_sq = F.conv2d(F.pad(img**2, [padding]*4, mode='reflect'), kernel)

        return (mean_sq - mean**2).clamp(min=0)

    def _vertical_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Compute vertical gradient magnitude."""
        pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
        return F.conv2d(pad, self.sobel_y.to(img.device)).abs()

    def _edge_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude."""
        pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
        gx = F.conv2d(pad, self.sobel_x.to(img.device))
        gy = F.conv2d(pad, self.sobel_y.to(img.device))
        return torch.sqrt(gx**2 + gy**2)


# =============================================================================
# CLINICAL NEURO-SYMBOLIC CORRECTOR
# =============================================================================

class ClinicalNeuroSymbolicCorrector(nn.Module):
    """
    Neuro-Symbolic Corrector focused on Clinical Utility.

    Key differences from V8:
    1. Predicates measure clinical features (contrast, boundary, texture, edge)
    2. Correctors target specific clinical weaknesses
    3. Success measured by clinical metrics, not PSNR

    Based on backbone analysis:
    - Contrast: 47% → Target 85%
    - Boundary: 47% → Target 85%
    - Texture: 41% → Target 75%
    - Edge: 68% → Target 85%
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Clinical predicates (GT-free)
        self.predicates = ClinicalPredicates()

        # Clinical correctors
        self.correctors = nn.ModuleDict({
            'contrast': ContrastRestorationCorrector(hidden_dim),
            'boundary': BoundarySharpnessCorrector(hidden_dim),
            'texture': TextureRecoveryCorrector(hidden_dim),
            'edge': EdgeEnhancementCorrector(hidden_dim),
        })

        # Predicate to corrector mapping
        self.pred_to_corrector = {
            'P_contrast': 'contrast',
            'P_boundary': 'boundary',
            'P_texture': 'texture',
            'P_edge': 'edge',
        }

        # Learnable activation thresholds (when to apply correction)
        self.activation_thresholds = nn.ParameterDict({
            'contrast': nn.Parameter(torch.tensor(0.7)),  # Activate if score < 0.7
            'boundary': nn.Parameter(torch.tensor(0.7)),
            'texture': nn.Parameter(torch.tensor(0.6)),
            'edge': nn.Parameter(torch.tensor(0.7)),
        })

        # Learnable correction strengths
        self.correction_strengths = nn.ParameterDict({
            'contrast': nn.Parameter(torch.tensor(1.0)),
            'boundary': nn.Parameter(torch.tensor(1.0)),
            'texture': nn.Parameter(torch.tensor(0.8)),
            'edge': nn.Parameter(torch.tensor(1.0)),
        })

        self._print_info()

    def _print_info(self):
        print("\n" + "=" * 70)
        print("CLINICAL NEURO-SYMBOLIC CORRECTOR")
        print("=" * 70)
        print("Focus: Clinical Utility (NOT PSNR)")
        print("")
        print("Clinical Predicates (GT-Free):")
        print("  - P_contrast: Local contrast preservation")
        print("  - P_boundary: Layer boundary sharpness")
        print("  - P_texture: Tissue texture preservation")
        print("  - P_edge: Edge/structure strength")
        print("")
        print("Targeted Corrections:")
        print("  - Contrast: 47% → 85% (restore layer visibility)")
        print("  - Boundary: 47% → 85% (sharpen layer transitions)")
        print("  - Texture: 41% → 75% (recover microstructure)")
        print("  - Edge: 68% → 85% (strengthen anatomical edges)")
        print("=" * 70)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, backbone_out: torch.Tensor, noisy: torch.Tensor,
                backbone_features: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Apply clinical corrections.

        Args:
            backbone_out: Backbone denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            backbone_features: Optional encoder features

        Returns:
            corrected: Clinically-enhanced output [B, 1, H, W]
            info: Dict with predicate scores and correction details
        """
        # Step 1: Evaluate clinical predicates
        with torch.no_grad():
            pred_results = self.predicates(backbone_out, noisy)

        # Step 2: Apply corrections based on predicate failures
        corrections = {}
        activations = {}

        for pred_name, corrector_name in self.pred_to_corrector.items():
            pred_info = pred_results[pred_name]
            score = pred_info['score']
            failure_map = pred_info['failure_map']

            # Get threshold and strength
            threshold = torch.sigmoid(self.activation_thresholds[corrector_name])
            strength = torch.sigmoid(self.correction_strengths[corrector_name])

            # Compute activation (how much to correct)
            # Higher activation when score is lower than threshold
            if isinstance(score, torch.Tensor):
                activation = ((threshold - score) / threshold).clamp(0, 1)
            else:
                activation = max(0, min(1, (threshold.item() - score) / threshold.item()))
                activation = torch.tensor(activation, device=backbone_out.device)

            activations[corrector_name] = activation.item() if isinstance(activation, torch.Tensor) else activation

            # Apply corrector
            corrector = self.correctors[corrector_name]
            correction = corrector(backbone_out, failure_map, backbone_features)

            # Scale by activation and strength
            corrections[corrector_name] = correction * activation * strength

        # Step 3: Combine corrections
        total_correction = sum(corrections.values())
        total_correction = total_correction.clamp(-0.5, 0.5)

        # Step 4: Apply correction
        corrected = (backbone_out + total_correction).clamp(0, 1)

        # Step 5: Re-evaluate predicates on corrected output
        with torch.no_grad():
            pred_after = self.predicates(corrected, noisy)

        # Build info dict
        info = {
            'predicate_scores_before': pred_results['scores'],
            'predicate_scores_after': pred_after['scores'],
            'activations': activations,
            'correction_magnitude': total_correction.abs().mean().item(),
            'corrections': {k: v.abs().mean().item() for k, v in corrections.items()},
        }

        # Compute improvements
        info['improvements'] = {}
        for pred_name in pred_results['scores']:
            before = pred_results['scores'][pred_name]
            after = pred_after['scores'][pred_name]
            info['improvements'][pred_name] = after - before

        return corrected, info


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\nTesting Clinical Neuro-Symbolic Corrector...")

    # Create model
    model = ClinicalNeuroSymbolicCorrector(hidden_dim=64)

    # Test inputs
    B, C, H, W = 2, 1, 128, 128
    noisy = torch.rand(B, C, H, W) * 0.5 + 0.25
    backbone_out = F.avg_pool2d(F.pad(noisy, [2]*4, mode='reflect'), 5, stride=1)  # Simulated smoothing

    # Forward pass
    corrected, info = model(backbone_out, noisy)

    print(f"\nInput shape: {backbone_out.shape}")
    print(f"Output shape: {corrected.shape}")
    print(f"\nPredicate Scores (Before → After):")
    for pred_name in info['predicate_scores_before']:
        before = info['predicate_scores_before'][pred_name]
        after = info['predicate_scores_after'][pred_name]
        improvement = info['improvements'][pred_name]
        print(f"  {pred_name}: {before:.3f} → {after:.3f} ({improvement:+.3f})")

    print(f"\nActivations: {info['activations']}")
    print(f"Correction magnitude: {info['correction_magnitude']:.4f}")
    print(f"Per-corrector: {info['corrections']}")

    print("\nTest passed!")
