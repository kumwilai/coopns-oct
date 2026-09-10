#!/usr/bin/env python3
"""
Region-Adaptive Correction Module for OCT Denoising

This module implements region-adaptive lambda prediction that applies:
- STRONG corrections on clinical regions (layer boundaries, edges)
- MINIMAL corrections on flat/homogeneous regions to preserve PSNR

Key insight: Not all regions need the same correction strength.
- Layer boundaries in OCT (horizontal lines) need strong edge correction
- Flat/homogeneous regions should be left alone to maintain PSNR
- Over-correcting flat regions hurts PSNR without clinical benefit

Author: Neuro-Symbolic OCT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# REGION DETECTION MODULES
# =============================================================================

class RegionDetector(nn.Module):
    """
    Detects clinical regions in OCT images using multiple cues:
    1. Vertical gradients (layer boundaries in OCT)
    2. Edge magnitude (Sobel)
    3. Local variance (texture regions)

    Output: Region importance map [0, 1] where:
    - 1.0 = clinical region (needs strong correction)
    - 0.0 = flat region (preserve PSNR, minimal correction)
    """

    def __init__(self,
                 gradient_weight: float = 0.4,
                 edge_weight: float = 0.3,
                 variance_weight: float = 0.3,
                 smoothing_kernel: int = 5):
        super().__init__()

        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight
        self.variance_weight = variance_weight
        self.smoothing_kernel = smoothing_kernel

        # Sobel filters for gradient computation
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_x', sobel_x)

        # Local variance computation kernel
        self.local_pool_size = 7
        self.local_pool = nn.AvgPool2d(
            kernel_size=self.local_pool_size,
            stride=1,
            padding=self.local_pool_size // 2
        )

        # Gaussian smoothing for final importance map
        if smoothing_kernel > 0:
            self.smoothing = self._create_gaussian_kernel(smoothing_kernel)
        else:
            self.smoothing = None

    def _create_gaussian_kernel(self, kernel_size: int) -> nn.Conv2d:
        """Create a Gaussian smoothing kernel."""
        sigma = kernel_size / 6.0
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gaussian_2d = gaussian_1d.outer(gaussian_1d)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()

        conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        conv.weight.data = gaussian_2d.view(1, 1, kernel_size, kernel_size)
        conv.weight.requires_grad = False
        return conv

    def compute_vertical_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute vertical gradient magnitude (layer boundaries in OCT).

        OCT layer boundaries run horizontally, so vertical gradients (d/dy)
        indicate where layer transitions occur.
        """
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_y.abs()

    def compute_edge_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge magnitude using Sobel operators.
        """
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return edge_mag

    def compute_local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute local variance in sliding windows.

        High variance = textured/feature-rich region
        Low variance = flat/homogeneous region
        """
        local_mean = self.local_pool(x)
        local_sq_mean = self.local_pool(x ** 2)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=1e-8)
        return local_var

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute region importance map.

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            Dict with:
                - importance_map: Combined region importance [B, 1, H, W]
                - vertical_gradient: Vertical gradient map [B, 1, H, W]
                - edge_magnitude: Edge magnitude map [B, 1, H, W]
                - local_variance: Local variance map [B, 1, H, W]
        """
        # Compute individual region indicators
        vert_grad = self.compute_vertical_gradient(x)
        edge_mag = self.compute_edge_magnitude(x)
        local_var = self.compute_local_variance(x)

        # Normalize each indicator to [0, 1] range using per-batch statistics
        # This ensures adaptive thresholding based on image content
        def normalize(t):
            B = t.shape[0]
            t_flat = t.view(B, -1)
            t_min = t_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            t_max = t_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            return (t - t_min) / (t_max - t_min + 1e-8)

        vert_grad_norm = normalize(vert_grad)
        edge_mag_norm = normalize(edge_mag)
        local_var_norm = normalize(local_var)

        # Combine into importance map
        importance = (
            self.gradient_weight * vert_grad_norm +
            self.edge_weight * edge_mag_norm +
            self.variance_weight * local_var_norm
        )

        # Apply Gaussian smoothing for spatial coherence
        if self.smoothing is not None:
            # Move smoothing kernel to same device
            if self.smoothing.weight.device != x.device:
                self.smoothing = self.smoothing.to(x.device)
            importance = self.smoothing(importance)

        # Clamp to [0, 1]
        importance = importance.clamp(0, 1)

        return {
            'importance_map': importance,
            'vertical_gradient': vert_grad_norm,
            'edge_magnitude': edge_mag_norm,
            'local_variance': local_var_norm,
        }


class LearnableRegionDetector(nn.Module):
    """
    Learnable region detector that combines hand-crafted features
    with learned feature extraction.

    This allows the network to learn which regions are clinically important
    beyond hand-crafted heuristics.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        # Hand-crafted region detector
        self.heuristic_detector = RegionDetector()

        # Learnable feature extraction
        # Input: image (1) + 3 heuristic features = 4 channels
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale processing for capturing different boundary types
        self.multiscale = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=3, dilation=3),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4),
        ])

        # Output head for importance map
        self.importance_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Learnable blending weight between heuristic and learned
        self.blend_weight = nn.Parameter(torch.tensor(0.5))

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute region importance using both heuristic and learned features.
        """
        # Get heuristic features
        heuristic = self.heuristic_detector(x)

        # Combine input with heuristic features
        combined = torch.cat([
            x,
            heuristic['vertical_gradient'],
            heuristic['edge_magnitude'],
            heuristic['local_variance'],
        ], dim=1)

        # Extract learned features
        feat = self.feature_net(combined)

        # Multi-scale processing
        ms_feats = [conv(feat) for conv in self.multiscale]
        feat = torch.cat(ms_feats, dim=1)

        # Predict learned importance
        learned_importance = self.importance_head(feat)

        # Blend heuristic and learned importance
        blend = torch.sigmoid(self.blend_weight)
        final_importance = blend * heuristic['importance_map'] + (1 - blend) * learned_importance

        return {
            'importance_map': final_importance,
            'heuristic_importance': heuristic['importance_map'],
            'learned_importance': learned_importance,
            'vertical_gradient': heuristic['vertical_gradient'],
            'edge_magnitude': heuristic['edge_magnitude'],
            'local_variance': heuristic['local_variance'],
            'blend_weight': blend,
        }


# =============================================================================
# REGION-ADAPTIVE LAMBDA PREDICTOR
# =============================================================================

class RegionAdaptiveLambdaPredictor(nn.Module):
    """
    Region-Adaptive Lambda Predictor for V8 Enhanced Corrector.

    This module predicts per-pixel correction strength (lambda) that is
    modulated by region importance:

    lambda_final = lambda_base * region_importance

    Key benefits:
    - Clinical regions (boundaries, edges) get strong correction (importance ~ 1.0)
    - Flat regions get minimal correction (importance ~ 0.0), preserving PSNR
    - Automatically adapts to image content

    Drop-in replacement for AdaptiveLambdaPredictorV8.
    """

    def __init__(self,
                 use_learnable_detector: bool = True,
                 importance_power: float = 1.0,
                 min_importance: float = 0.0,
                 hidden_dim: int = 32):
        super().__init__()

        self.importance_power = importance_power  # Power scaling for importance
        self.min_importance = min_importance  # Minimum importance floor

        # Region detector
        if use_learnable_detector:
            self.region_detector = LearnableRegionDetector(hidden_dim=hidden_dim)
        else:
            self.region_detector = RegionDetector()

        # Corrector to predicate mapping (same as V8)
        self.corrector_pred_map = {
            'edge': 'P1', 'contrast': 'P2', 'smooth': 'P3',
            'structure': 'P4', 'anatomy': 'P6'
        }

        # Input: 6 failure maps (P1-P6) + denoised (1) + importance (1) = 8 channels
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Per-corrector lambda heads (P5/speckle excluded)
        self.heads = nn.ModuleDict({
            'edge': self._make_head(32 + 1),      # +1 for P1
            'contrast': self._make_head(32 + 1),  # +1 for P2
            'smooth': self._make_head(32 + 1),    # +1 for P3
            'structure': self._make_head(32 + 1), # +1 for P4
            'anatomy': self._make_head(32 + 1),   # +1 for P6
        })

        # Learnable scaling factors
        self.scales = nn.ParameterDict({
            'edge': nn.Parameter(torch.tensor(0.0)),       # sigmoid(0)=0.5
            'contrast': nn.Parameter(torch.tensor(0.0)),   # sigmoid(0)=0.5
            'smooth': nn.Parameter(torch.tensor(-0.5)),    # sigmoid(-0.5)=0.38
            'structure': nn.Parameter(torch.tensor(-0.5)), # sigmoid(-0.5)=0.38
            'anatomy': nn.Parameter(torch.tensor(-0.5)),   # sigmoid(-0.5)=0.38
        })

        # Maximum lambda caps
        self.lambda_caps = {
            'edge': 0.80,
            'contrast': 0.80,
            'smooth': 0.50,
            'structure': 0.50,
            'anatomy': 0.50,
        }

        # Region-specific importance weights
        # Some correctors (edge, structure) should weight boundary regions more
        self.importance_weights = nn.ParameterDict({
            'edge': nn.Parameter(torch.tensor(1.5)),       # Higher weight for edges
            'contrast': nn.Parameter(torch.tensor(1.0)),   # Normal weight
            'smooth': nn.Parameter(torch.tensor(0.8)),     # Lower weight (smooth everywhere)
            'structure': nn.Parameter(torch.tensor(1.5)),  # Higher weight for boundaries
            'anatomy': nn.Parameter(torch.tensor(1.0)),    # Normal weight
        })

        self._init_weights()

    def _make_head(self, in_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),  # Always positive
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize lambda head biases
        for head in self.heads.values():
            for module in head:
                if isinstance(module, nn.Conv2d) and module.out_channels == 1:
                    if module.bias is not None:
                        nn.init.constant_(module.bias, -2.0)

    def forward(self, denoised: torch.Tensor,
                failure_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict per-pixel lambda maps modulated by region importance.

        Args:
            denoised: Backbone output [B, 1, H, W]
            failure_maps: Dict of failure maps {'P1': [B,1,H,W], ...}

        Returns:
            Dict of lambda maps {'edge': [B,1,H,W], ...}
        """
        B, C, H, W = denoised.shape
        device = denoised.device

        # Compute region importance
        region_info = self.region_detector(denoised)
        importance_map = region_info['importance_map']

        # Apply power scaling and minimum floor
        importance_map = importance_map ** self.importance_power
        importance_map = importance_map.clamp(min=self.min_importance)

        # Get failure maps with default
        default_map = torch.zeros(B, 1, H, W, device=device, dtype=denoised.dtype)
        p1 = failure_maps.get('P1', default_map)
        p2 = failure_maps.get('P2', default_map)
        p3 = failure_maps.get('P3', default_map)
        p4 = failure_maps.get('P4', default_map)
        p5 = failure_maps.get('P5', default_map)
        p6 = failure_maps.get('P6', default_map)

        # Concat all inputs including importance map
        x = torch.cat([p1, p2, p3, p4, p5, p6, denoised, importance_map], dim=1)

        # Shared features
        shared_feat = self.shared(x)

        # Per-corrector lambda maps
        pred_map = {'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4, 'P6': p6}

        lambda_maps = {}
        for name, head in self.heads.items():
            pred_key = self.corrector_pred_map[name]
            head_input = torch.cat([shared_feat, pred_map[pred_key]], dim=1)
            raw_lambda = head(head_input)

            # Apply learnable scaling
            scaled = raw_lambda * torch.sigmoid(self.scales[name])

            # Cap the lambda
            capped = scaled.clamp(max=self.lambda_caps[name])

            # REGION-ADAPTIVE MODULATION
            # lambda_final = lambda_base * (importance ^ weight)
            # This makes flat regions (low importance) have near-zero correction
            importance_weight = torch.sigmoid(self.importance_weights[name])
            modulated_importance = importance_map ** importance_weight

            # Final lambda is modulated by region importance
            lambda_maps[name] = capped * modulated_importance

        # Store region info for debugging/visualization
        self._last_region_info = region_info

        return lambda_maps

    def get_region_info(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the last computed region information for visualization."""
        return getattr(self, '_last_region_info', None)


# =============================================================================
# BOUNDARY-FOCUSED LOSS
# =============================================================================

class BoundaryFocusedLoss(nn.Module):
    """
    Boundary-Focused Loss that applies different weights to different regions:

    - Clinical regions (boundaries, edges): Higher weight on clinical quality losses
    - Flat regions: Higher weight on reconstruction/PSNR loss

    This allows the model to focus corrections where they matter clinically
    while preserving PSNR in flat regions.
    """

    def __init__(self,
                 clinical_boundary_weight: float = 2.0,
                 clinical_flat_weight: float = 0.5,
                 recon_boundary_weight: float = 0.5,
                 recon_flat_weight: float = 2.0,
                 importance_threshold: float = 0.5):
        super().__init__()

        self.clinical_boundary_weight = clinical_boundary_weight
        self.clinical_flat_weight = clinical_flat_weight
        self.recon_boundary_weight = recon_boundary_weight
        self.recon_flat_weight = recon_flat_weight
        self.importance_threshold = importance_threshold

        # Region detector for computing importance
        self.region_detector = RegionDetector()

        # Sobel filters
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_region_weights(self, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute region-adaptive weights for clinical and reconstruction losses.

        Args:
            reference: Reference image (clean GT or backbone output) [B, 1, H, W]

        Returns:
            clinical_weights: Weight map for clinical losses [B, 1, H, W]
            recon_weights: Weight map for reconstruction loss [B, 1, H, W]
        """
        # Get region importance
        region_info = self.region_detector(reference)
        importance = region_info['importance_map']

        # Create boundary mask (importance > threshold)
        boundary_mask = (importance > self.importance_threshold).float()
        flat_mask = 1.0 - boundary_mask

        # Compute weights
        clinical_weights = (
            boundary_mask * self.clinical_boundary_weight +
            flat_mask * self.clinical_flat_weight
        )

        recon_weights = (
            boundary_mask * self.recon_boundary_weight +
            flat_mask * self.recon_flat_weight
        )

        # Normalize weights to have mean 1.0 (preserves loss scale)
        clinical_weights = clinical_weights / clinical_weights.mean().clamp(min=1e-6)
        recon_weights = recon_weights / recon_weights.mean().clamp(min=1e-6)

        return clinical_weights, recon_weights

    def local_std(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Compute local standard deviation."""
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                           device=x.device, dtype=x.dtype) / (kernel_size ** 2)
        padding = kernel_size // 2
        local_mean = F.conv2d(x, kernel, padding=padding)
        local_mean_sq = F.conv2d(x ** 2, kernel, padding=padding)
        local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=1e-6)
        return torch.sqrt(local_var)

    def weighted_mse_loss(self, pred: torch.Tensor, target: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss."""
        mse = (pred - target) ** 2
        weighted_mse = (mse * weights).mean()
        return weighted_mse

    def weighted_contrast_loss(self, pred: torch.Tensor, target: torch.Tensor,
                               weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted contrast preservation loss."""
        pred_std = self.local_std(pred, kernel_size=5)
        target_std = self.local_std(target, kernel_size=5)
        contrast_loss = F.relu(target_std - pred_std)  # Penalize reduced contrast
        weighted_loss = (contrast_loss * weights).mean()
        return weighted_loss

    def weighted_boundary_loss(self, pred: torch.Tensor, target: torch.Tensor,
                               weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted boundary sharpness loss."""
        pred_grad = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        boundary_loss = F.relu(target_grad - pred_grad)
        # Adjust weights to match gradient map size
        weights_adj = weights[:, :, :-1, :]
        weighted_loss = (boundary_loss * weights_adj).mean()
        return weighted_loss

    def weighted_edge_loss(self, pred: torch.Tensor, target: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted edge preservation loss."""
        sobel_x = self.sobel_x.to(dtype=pred.dtype)
        sobel_y = self.sobel_y.to(dtype=pred.dtype)

        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

        edge_loss = F.relu(target_edges - pred_edges)
        weighted_loss = (edge_loss * weights).mean()
        return weighted_loss

    def forward(self, corrected: torch.Tensor, backbone: torch.Tensor,
                clean: torch.Tensor, info: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute boundary-focused loss.

        Args:
            corrected: Corrected output [B, 1, H, W]
            backbone: Backbone output [B, 1, H, W]
            clean: Clean ground truth [B, 1, H, W]
            info: Optional info dict from corrector

        Returns:
            total_loss: Combined loss scalar
            metrics: Dict with individual loss values
        """
        # Compute region-adaptive weights based on clean image
        clinical_weights, recon_weights = self.compute_region_weights(clean)

        # Reconstruction loss (weighted by flat regions)
        recon_loss = self.weighted_mse_loss(corrected, clean, recon_weights)

        # Clinical losses (weighted by boundary regions)
        contrast_loss = self.weighted_contrast_loss(corrected, clean, clinical_weights)
        boundary_loss = self.weighted_boundary_loss(corrected, clean, clinical_weights)
        edge_loss = self.weighted_edge_loss(corrected, clean, clinical_weights)

        # Backbone supervision (unweighted)
        backbone_loss = F.mse_loss(backbone, clean)

        # Lambda regularization
        lambda_reg = torch.tensor(0.0, device=corrected.device)
        if info is not None and 'lambda_stats' in info:
            lambda_stats = info['lambda_stats']
            means = [s['mean'] for s in lambda_stats.values() if isinstance(s.get('mean'), torch.Tensor)]
            if means:
                lambda_reg = torch.stack(means).mean() * 0.01

        # Total loss with fixed weights
        total_loss = (
            recon_loss * 1.0 +
            backbone_loss * 0.1 +
            contrast_loss * 0.2 +
            boundary_loss * 0.2 +
            edge_loss * 0.2 +
            lambda_reg
        )

        # Compute metrics
        with torch.no_grad():
            psnr_backbone = 10 * torch.log10(1 / (F.mse_loss(backbone, clean) + 1e-6))
            psnr_corrected = 10 * torch.log10(1 / (F.mse_loss(corrected, clean) + 1e-6))

        metrics = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'backbone': backbone_loss.item(),
            'contrast': contrast_loss.item(),
            'boundary': boundary_loss.item(),
            'edge': edge_loss.item(),
            'lambda_reg': lambda_reg.item() if isinstance(lambda_reg, torch.Tensor) else lambda_reg,
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'psnr_delta': psnr_corrected.item() - psnr_backbone.item(),
        }

        return total_loss, metrics


class BoundaryFocusedLossV2(BoundaryFocusedLoss):
    """
    Enhanced boundary-focused loss with uncertainty weighting and
    GT-aligned predicate learning (compatible with V8EnhancedLoss).
    """

    def __init__(self,
                 clinical_boundary_weight: float = 2.0,
                 clinical_flat_weight: float = 0.5,
                 recon_boundary_weight: float = 0.5,
                 recon_flat_weight: float = 2.0,
                 importance_threshold: float = 0.5,
                 use_uncertainty_weighting: bool = True):
        super().__init__(
            clinical_boundary_weight=clinical_boundary_weight,
            clinical_flat_weight=clinical_flat_weight,
            recon_boundary_weight=recon_boundary_weight,
            recon_flat_weight=recon_flat_weight,
            importance_threshold=importance_threshold,
        )

        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Learnable uncertainty parameters (log sigma^2)
        self.log_sigma = nn.ParameterDict({
            'recon': nn.Parameter(torch.tensor(0.0)),
            'backbone': nn.Parameter(torch.tensor(1.0)),
            'contrast': nn.Parameter(torch.tensor(1.0)),
            'boundary': nn.Parameter(torch.tensor(1.0)),
            'edge': nn.Parameter(torch.tensor(1.0)),
            'psnr_preserve': nn.Parameter(torch.tensor(-1.0)),
        })

        # PSNR preservation slack
        self.psnr_slack = 0.5  # dB

    def _weighted_loss(self, loss: torch.Tensor, name: str) -> torch.Tensor:
        """Apply uncertainty weighting to a loss term."""
        if self.use_uncertainty_weighting:
            log_sig = self.log_sigma[name]
            precision_weight = 0.5 * torch.exp(-log_sig)
            regularization = 0.5 * log_sig
            return precision_weight * loss + regularization
        else:
            return loss

    def psnr_preservation_loss(self, corrected: torch.Tensor, backbone: torch.Tensor,
                                clean: torch.Tensor) -> torch.Tensor:
        """Penalize when corrected PSNR drops below backbone."""
        mse_backbone = F.mse_loss(backbone, clean)
        mse_corrected = F.mse_loss(corrected, clean)

        psnr_backbone = 10 * torch.log10(1.0 / (mse_backbone + 1e-8))
        psnr_corrected = 10 * torch.log10(1.0 / (mse_corrected + 1e-8))

        psnr_drop = psnr_backbone - psnr_corrected
        penalty = F.relu(psnr_drop - self.psnr_slack)
        return penalty ** 2

    def forward(self, corrected: torch.Tensor, backbone: torch.Tensor,
                clean: torch.Tensor, info: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """Compute boundary-focused loss with uncertainty weighting."""
        # Compute region-adaptive weights
        clinical_weights, recon_weights = self.compute_region_weights(clean)

        # Individual losses
        recon_loss = self.weighted_mse_loss(corrected, clean, recon_weights)
        backbone_loss = F.mse_loss(backbone, clean)
        contrast_loss = self.weighted_contrast_loss(corrected, clean, clinical_weights)
        boundary_loss = self.weighted_boundary_loss(corrected, clean, clinical_weights)
        edge_loss = self.weighted_edge_loss(corrected, clean, clinical_weights)
        psnr_preserve_loss = self.psnr_preservation_loss(corrected, backbone, clean)

        # Lambda regularization
        lambda_reg = torch.tensor(0.0, device=corrected.device)
        if info is not None and 'lambda_stats' in info:
            lambda_stats = info['lambda_stats']
            means = [s['mean'] for s in lambda_stats.values() if isinstance(s.get('mean'), torch.Tensor)]
            if means:
                lambda_reg = torch.stack(means).mean() * 0.01

        # Total loss with uncertainty weighting
        total_loss = (
            self._weighted_loss(recon_loss, 'recon') +
            self._weighted_loss(backbone_loss, 'backbone') +
            self._weighted_loss(contrast_loss, 'contrast') +
            self._weighted_loss(boundary_loss, 'boundary') +
            self._weighted_loss(edge_loss, 'edge') +
            self._weighted_loss(psnr_preserve_loss, 'psnr_preserve') +
            lambda_reg
        )

        # Compute metrics
        with torch.no_grad():
            psnr_backbone = 10 * torch.log10(1 / (F.mse_loss(backbone, clean) + 1e-6))
            psnr_corrected = 10 * torch.log10(1 / (F.mse_loss(corrected, clean) + 1e-6))

            # Get learned weights
            learned_weights = {}
            for name, log_sig in self.log_sigma.items():
                sigma_sq = torch.exp(log_sig)
                learned_weights[name] = (0.5 / sigma_sq).item()

        metrics = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'backbone': backbone_loss.item(),
            'contrast': contrast_loss.item(),
            'boundary': boundary_loss.item(),
            'edge': edge_loss.item(),
            'psnr_preserve': psnr_preserve_loss.item(),
            'lambda_reg': lambda_reg.item() if isinstance(lambda_reg, torch.Tensor) else lambda_reg,
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'psnr_delta': psnr_corrected.item() - psnr_backbone.item(),
            'learned_weights': learned_weights,
        }

        return total_loss, metrics


# =============================================================================
# INTEGRATION WITH V8 ENHANCED
# =============================================================================

def replace_lambda_predictor(corrector_module: nn.Module,
                              use_learnable_detector: bool = True) -> nn.Module:
    """
    Replace the AdaptiveLambdaPredictorV8 in a V8Enhanced corrector
    with RegionAdaptiveLambdaPredictor.

    Args:
        corrector_module: NeuroSymbolicCorrectorV8Enhanced instance
        use_learnable_detector: Whether to use learnable region detector

    Returns:
        Modified corrector module with region-adaptive lambda predictor
    """
    # Create new region-adaptive predictor
    region_adaptive_predictor = RegionAdaptiveLambdaPredictor(
        use_learnable_detector=use_learnable_detector
    )

    # Replace the lambda predictor
    corrector_module.lambda_predictor = region_adaptive_predictor

    print("\n" + "=" * 60)
    print("Replaced AdaptiveLambdaPredictorV8 with RegionAdaptiveLambdaPredictor")
    print(f"  Use learnable detector: {use_learnable_detector}")
    print("  Region-based modulation: lambda_final = lambda_base * importance")
    print("  - Flat regions: minimal correction (preserves PSNR)")
    print("  - Clinical regions: strong correction (layer boundaries, edges)")
    print("=" * 60)

    return corrector_module


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\nTesting Region-Adaptive Correction Module...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test RegionDetector
    print("\n1. Testing RegionDetector...")
    detector = RegionDetector().to(device)
    test_input = torch.randn(2, 1, 128, 128, device=device)
    region_info = detector(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Importance map shape: {region_info['importance_map'].shape}")
    print(f"   Importance range: [{region_info['importance_map'].min():.3f}, {region_info['importance_map'].max():.3f}]")

    # Test LearnableRegionDetector
    print("\n2. Testing LearnableRegionDetector...")
    learnable_detector = LearnableRegionDetector(hidden_dim=32).to(device)
    region_info = learnable_detector(test_input)
    print(f"   Blend weight: {region_info['blend_weight'].item():.3f}")
    print(f"   Heuristic importance range: [{region_info['heuristic_importance'].min():.3f}, {region_info['heuristic_importance'].max():.3f}]")
    print(f"   Learned importance range: [{region_info['learned_importance'].min():.3f}, {region_info['learned_importance'].max():.3f}]")

    # Test RegionAdaptiveLambdaPredictor
    print("\n3. Testing RegionAdaptiveLambdaPredictor...")
    predictor = RegionAdaptiveLambdaPredictor(use_learnable_detector=True).to(device)

    # Create dummy failure maps
    failure_maps = {
        'P1': torch.rand(2, 1, 128, 128, device=device),
        'P2': torch.rand(2, 1, 128, 128, device=device),
        'P3': torch.rand(2, 1, 128, 128, device=device),
        'P4': torch.rand(2, 1, 128, 128, device=device),
        'P5': torch.rand(2, 1, 128, 128, device=device),
        'P6': torch.rand(2, 1, 128, 128, device=device),
    }

    lambda_maps = predictor(test_input, failure_maps)
    print(f"   Lambda maps:")
    for name, lam in lambda_maps.items():
        print(f"     {name}: mean={lam.mean().item():.4f}, max={lam.max().item():.4f}")

    # Test BoundaryFocusedLoss
    print("\n4. Testing BoundaryFocusedLoss...")
    loss_fn = BoundaryFocusedLoss().to(device)

    corrected = torch.rand(2, 1, 128, 128, device=device)
    backbone = torch.rand(2, 1, 128, 128, device=device)
    clean = torch.rand(2, 1, 128, 128, device=device)

    info = {'lambda_stats': {name: {'mean': lam.mean()} for name, lam in lambda_maps.items()}}
    loss, metrics = loss_fn(corrected, backbone, clean, info)

    print(f"   Total loss: {metrics['total']:.4f}")
    print(f"   Recon loss: {metrics['recon']:.4f}")
    print(f"   PSNR delta: {metrics['psnr_delta']:+.2f} dB")

    # Test BoundaryFocusedLossV2
    print("\n5. Testing BoundaryFocusedLossV2 (with uncertainty weighting)...")
    loss_fn_v2 = BoundaryFocusedLossV2(use_uncertainty_weighting=True).to(device)

    loss_v2, metrics_v2 = loss_fn_v2(corrected, backbone, clean, info)
    print(f"   Total loss: {metrics_v2['total']:.4f}")
    print(f"   Learned weights: {metrics_v2['learned_weights']}")

    # Parameter count
    print("\n6. Parameter counts:")
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"   RegionAdaptiveLambdaPredictor: {total_params:,} parameters")

    loss_params = sum(p.numel() for p in loss_fn_v2.parameters())
    print(f"   BoundaryFocusedLossV2: {loss_params:,} parameters")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
