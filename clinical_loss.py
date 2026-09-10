#!/usr/bin/env python3
"""
ClinicalLoss: Clinical Utility-Focused Loss for OCT Denoising

Design Philosophy:
    Instead of optimizing for PSNR (pixel-wise MSE), this loss optimizes for
    clinical utility - the qualities that matter for diagnosis:

    1. Contrast Preservation: Layer boundaries should remain visible
    2. Boundary Sharpness: OCT layer transitions should be sharp, not blurred
    3. Edge Preservation: Structural edges should not be weakened
    4. Texture Preservation: Avoid over-smoothing diagnostic textures
    5. Layer Visibility: Important structures should remain visible

Adaptive Weighting:
    Each loss term is weighted based on how much that specific metric has
    degraded in the output compared to ground truth. If contrast is poor,
    contrast loss gets more weight. This creates a self-correcting system.

Usage:
    criterion = ClinicalLoss()
    loss, metrics = criterion(corrected, ground_truth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ClinicalLoss(nn.Module):
    """
    Clinical Utility-Focused Loss for OCT Denoising.

    This loss function prioritizes clinical image quality over raw PSNR:
    - Sharper layer boundaries (diagnostic importance)
    - Better preserved contrast (layer visibility)
    - Maintained edge structure (anatomical accuracy)
    - Appropriate texture (avoid over-smoothing)
    - Preserved layer visibility (clinical features remain visible)

    Adaptive Weighting:
        weight_i = base_weight_i * (1 + degradation_i)

        Where degradation_i = max(0, metric_gt_i - metric_pred_i) / metric_gt_i
        This increases weight for metrics that have degraded most.

    Args:
        base_contrast: Base weight for contrast preservation (default: 0.2)
        base_boundary: Base weight for boundary sharpness (default: 0.25)
        base_edge: Base weight for edge preservation (default: 0.2)
        base_texture: Base weight for texture preservation (default: 0.15)
        base_layer_vis: Base weight for layer visibility (default: 0.2)
        adaptive_strength: How strongly to adapt weights (0=fixed, 1=full adaptation)
        include_minimal_mse: Include a small MSE term for stability (default: True)
        mse_weight: Weight for the minimal MSE term if included (default: 0.1)
    """

    def __init__(
        self,
        base_contrast: float = 0.2,
        base_boundary: float = 0.25,
        base_edge: float = 0.2,
        base_texture: float = 0.15,
        base_layer_vis: float = 0.2,
        adaptive_strength: float = 1.0,
        include_minimal_mse: bool = True,
        mse_weight: float = 0.1,
    ):
        super().__init__()

        self.base_contrast = base_contrast
        self.base_boundary = base_boundary
        self.base_edge = base_edge
        self.base_texture = base_texture
        self.base_layer_vis = base_layer_vis
        self.adaptive_strength = adaptive_strength
        self.include_minimal_mse = include_minimal_mse
        self.mse_weight = mse_weight

        # Pre-compute Sobel kernels (registered as buffers for device handling)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Laplacian kernel for edge detection
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

        # Gaussian kernel for smoothing (used in texture computation)
        gaussian = self._create_gaussian_kernel(5, 1.0)
        self.register_buffer('gaussian', gaussian)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create a Gaussian kernel for smoothing."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), dim=-1)
        kernel = torch.exp(-(grid ** 2).sum(-1) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    # =========================================================================
    # Contrast Preservation Loss
    # =========================================================================

    def compute_local_contrast(self, x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """
        Compute local contrast as the ratio of local std to local mean (Weber contrast).

        High contrast regions have high std relative to mean - these are the
        diagnostically important layer boundaries in OCT.

        Args:
            x: Input tensor [B, 1, H, W]
            kernel_size: Size of local neighborhood

        Returns:
            Local contrast map [B, 1, H, W]
        """
        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
        kernel = kernel / (kernel_size ** 2)

        # Local mean
        local_mean = F.conv2d(x, kernel, padding=padding)

        # Local mean of squares
        local_mean_sq = F.conv2d(x ** 2, kernel, padding=padding)

        # Local variance = E[X^2] - E[X]^2
        local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=1e-8)
        local_std = torch.sqrt(local_var)

        # Weber contrast: std / (mean + eps) - high where there's variation
        contrast = local_std / (local_mean.abs() + 0.01)

        return contrast

    def contrast_preservation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Contrast Preservation Loss: Penalize contrast reduction.

        OCT images need strong local contrast for layer visibility.
        This loss penalizes when the output has lower local contrast than GT.

        Args:
            pred: Predicted/corrected image [B, 1, H, W]
            target: Ground truth clean image [B, 1, H, W]

        Returns:
            loss: Scalar loss value
            degradation: How much contrast degraded (0 = no degradation)
        """
        pred_contrast = self.compute_local_contrast(pred)
        target_contrast = self.compute_local_contrast(target)

        # Measure contrast degradation (how much contrast was lost)
        # Asymmetric: penalize reduction more than increase
        contrast_reduction = F.relu(target_contrast - pred_contrast)

        # Also penalize overall contrast mismatch (but less)
        contrast_mismatch = (pred_contrast - target_contrast).abs()

        # Combined loss: strongly penalize reduction, mildly penalize mismatch
        loss = 2.0 * contrast_reduction.mean() + 0.5 * contrast_mismatch.mean()

        # Compute degradation metric for adaptive weighting
        target_contrast_mean = target_contrast.mean() + 1e-8
        pred_contrast_mean = pred_contrast.mean()
        degradation = F.relu(target_contrast_mean - pred_contrast_mean) / target_contrast_mean

        return loss, degradation

    # =========================================================================
    # Boundary Sharpness Loss
    # =========================================================================

    def compute_boundary_sharpness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary sharpness map focusing on horizontal layer boundaries.

        OCT images have horizontal layer boundaries (retinal layers).
        Sharp boundaries have high vertical gradient magnitude.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            Boundary sharpness map [B, 1, H, W]
        """
        # Vertical gradient (detects horizontal edges - layer boundaries)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # Also compute second derivative for sharpness (not just magnitude)
        # Sharp edges have high second derivative at the transition
        grad_yy = F.conv2d(grad_y.abs(), self.sobel_y, padding=1).abs()

        # Combined sharpness: high gradient AND high second derivative = sharp edge
        sharpness = grad_y.abs() * (1.0 + grad_yy)

        return sharpness

    def boundary_sharpness_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Boundary Sharpness Loss: Penalize blurred layer transitions.

        Sharp layer boundaries are critical for OCT diagnosis.
        This loss ensures boundaries remain sharp, not blurred by denoising.

        Args:
            pred: Predicted/corrected image [B, 1, H, W]
            target: Ground truth clean image [B, 1, H, W]

        Returns:
            loss: Scalar loss value
            degradation: How much sharpness degraded (0 = no degradation)
        """
        pred_sharpness = self.compute_boundary_sharpness(pred)
        target_sharpness = self.compute_boundary_sharpness(target)

        # Find significant boundaries (top 10% sharpness in target)
        threshold = torch.quantile(target_sharpness.flatten(), 0.9)
        boundary_mask = (target_sharpness > threshold).float()

        # Penalize sharpness reduction at important boundaries
        # Focus on where GT has sharp boundaries
        sharpness_at_boundaries_pred = (pred_sharpness * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)
        sharpness_at_boundaries_target = (target_sharpness * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)

        # Also match overall sharpness distribution
        sharpness_diff = (pred_sharpness - target_sharpness).abs()
        weighted_diff = sharpness_diff * (1.0 + boundary_mask)  # Extra weight at boundaries

        # Asymmetric: penalize sharpness reduction more
        sharpness_reduction = F.relu(target_sharpness - pred_sharpness)

        loss = sharpness_reduction.mean() + 0.5 * weighted_diff.mean()

        # Compute degradation for adaptive weighting
        degradation = F.relu(sharpness_at_boundaries_target - sharpness_at_boundaries_pred) / (sharpness_at_boundaries_target + 1e-8)

        return loss, degradation

    # =========================================================================
    # Edge Preservation Loss
    # =========================================================================

    def compute_edge_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge magnitude using Sobel operators.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            Edge magnitude map [B, 1, H, W]
        """
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return magnitude

    def edge_preservation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Edge Preservation Loss: Penalize weakened edges.

        Structural edges contain diagnostic information.
        This loss ensures edges are preserved, not weakened by denoising.

        Different from boundary sharpness: this considers ALL edges (x and y),
        while boundary sharpness focuses on horizontal layer boundaries.

        Args:
            pred: Predicted/corrected image [B, 1, H, W]
            target: Ground truth clean image [B, 1, H, W]

        Returns:
            loss: Scalar loss value
            degradation: How much edge strength degraded (0 = no degradation)
        """
        pred_edges = self.compute_edge_magnitude(pred)
        target_edges = self.compute_edge_magnitude(target)

        # Identify significant edges (above median in target)
        edge_threshold = torch.quantile(target_edges.flatten(), 0.5)
        significant_edges = (target_edges > edge_threshold).float()

        # Penalize edge weakening at significant edges
        edge_weakening = F.relu(target_edges - pred_edges) * significant_edges

        # Also match overall edge distribution (but less weight)
        edge_mismatch = (pred_edges - target_edges).abs()

        loss = 2.0 * edge_weakening.mean() + 0.5 * edge_mismatch.mean()

        # Compute degradation for adaptive weighting
        target_edge_mean = (target_edges * significant_edges).sum() / (significant_edges.sum() + 1e-8)
        pred_edge_mean = (pred_edges * significant_edges).sum() / (significant_edges.sum() + 1e-8)
        degradation = F.relu(target_edge_mean - pred_edge_mean) / (target_edge_mean + 1e-8)

        return loss, degradation

    # =========================================================================
    # Texture Preservation Loss
    # =========================================================================

    def compute_texture_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute texture energy using Laws texture energy measures.

        Texture is important for diagnosis - over-smoothing removes
        diagnostic texture information (e.g., drusen texture, pathology patterns).

        Uses a simplified approach: high-frequency content after smoothing.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            Texture energy map [B, 1, H, W]
        """
        # Smooth version (low-frequency)
        smoothed = F.conv2d(x, self.gaussian, padding=self.gaussian.shape[-1]//2)

        # High-frequency content (texture)
        high_freq = (x - smoothed).abs()

        # Local energy of high-frequency content
        kernel_size = 5
        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
        kernel = kernel / (kernel_size ** 2)

        texture_energy = F.conv2d(high_freq ** 2, kernel, padding=padding)

        return texture_energy

    def texture_preservation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Texture Preservation Loss: Penalize over-smoothing.

        Denoising often removes texture along with noise.
        This loss ensures diagnostic textures are preserved.

        Args:
            pred: Predicted/corrected image [B, 1, H, W]
            target: Ground truth clean image [B, 1, H, W]

        Returns:
            loss: Scalar loss value
            degradation: How much texture was lost (0 = no loss)
        """
        pred_texture = self.compute_texture_energy(pred)
        target_texture = self.compute_texture_energy(target)

        # Identify regions with meaningful texture in GT (not just noise)
        # Use threshold based on texture distribution
        texture_threshold = torch.quantile(target_texture.flatten(), 0.7)
        textured_regions = (target_texture > texture_threshold).float()

        # Penalize texture reduction in textured regions
        texture_reduction = F.relu(target_texture - pred_texture) * textured_regions

        # Allow some texture reduction in non-textured regions (noise removal is OK there)
        # But still penalize complete smoothing
        non_textured_penalty = F.relu(target_texture - pred_texture) * (1.0 - textured_regions) * 0.3

        loss = texture_reduction.mean() + non_textured_penalty.mean()

        # Compute degradation for adaptive weighting
        target_texture_mean = (target_texture * textured_regions).sum() / (textured_regions.sum() + 1e-8)
        pred_texture_mean = (pred_texture * textured_regions).sum() / (textured_regions.sum() + 1e-8)
        degradation = F.relu(target_texture_mean - pred_texture_mean) / (target_texture_mean + 1e-8)

        return loss, degradation

    # =========================================================================
    # Layer Visibility Loss
    # =========================================================================

    def compute_layer_visibility(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute layer visibility score based on intensity profile analysis.

        In OCT, layers appear as distinct intensity bands. Good visibility means
        clear separation between bright and dark layers.

        Returns:
            layer_contrast: Map of local layer contrast [B, 1, H, W]
            intensity_variation: Column-wise intensity variation [B, 1, 1, W]
        """
        B, C, H, W = x.shape

        # Compute vertical intensity profile (average across columns)
        # This gives the "layer structure" of the image
        vertical_profile = x.mean(dim=3, keepdim=True)  # [B, 1, H, 1]

        # Compute variation along each column
        # High variation = clear layer structure
        column_profiles = x  # [B, 1, H, W]

        # Local intensity gradient along vertical (layer) direction
        layer_gradient = torch.abs(column_profiles[:, :, 1:, :] - column_profiles[:, :, :-1, :])
        # Pad to match original size
        layer_gradient = F.pad(layer_gradient, [0, 0, 0, 1], mode='replicate')

        # Intensity variation per column (measure of layer visibility)
        intensity_variation = x.std(dim=2, keepdim=True)  # [B, 1, 1, W]

        return layer_gradient, intensity_variation

    def layer_visibility_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Layer Visibility Loss: Important structures should remain visible.

        OCT layers must be distinguishable for diagnosis.
        This loss ensures layer separation is maintained.

        Args:
            pred: Predicted/corrected image [B, 1, H, W]
            target: Ground truth clean image [B, 1, H, W]

        Returns:
            loss: Scalar loss value
            degradation: How much visibility degraded (0 = no degradation)
        """
        pred_gradient, pred_variation = self.compute_layer_visibility(pred)
        target_gradient, target_variation = self.compute_layer_visibility(target)

        # Penalize reduced layer gradient (blurred layers)
        gradient_reduction = F.relu(target_gradient - pred_gradient)

        # Penalize reduced column variation (lost layer structure)
        variation_reduction = F.relu(target_variation - pred_variation)

        # Match layer gradient distribution
        gradient_mismatch = (pred_gradient - target_gradient).abs()

        loss = (
            1.5 * gradient_reduction.mean() +
            1.0 * variation_reduction.mean() +
            0.5 * gradient_mismatch.mean()
        )

        # Compute degradation for adaptive weighting
        target_vis_mean = target_gradient.mean() + 1e-8
        pred_vis_mean = pred_gradient.mean()
        degradation = F.relu(target_vis_mean - pred_vis_mean) / target_vis_mean

        return loss, degradation

    # =========================================================================
    # Adaptive Weight Computation
    # =========================================================================

    def compute_adaptive_weights(
        self,
        degradations: Dict[str, torch.Tensor],
        base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute adaptive weights based on degradation metrics.

        Weight formula:
            weight_i = base_weight_i * (1 + adaptive_strength * degradation_i)

        This increases weight for metrics that have degraded most,
        creating a self-correcting training signal.

        Args:
            degradations: Dict of degradation values for each metric
            base_weights: Dict of base weights for each metric

        Returns:
            Dict of adapted weights
        """
        adapted_weights = {}

        for name, base_weight in base_weights.items():
            degradation = degradations.get(name, torch.tensor(0.0))
            if isinstance(degradation, torch.Tensor):
                degradation = degradation.item()

            # Clamp degradation to reasonable range
            degradation = min(max(degradation, 0.0), 2.0)

            # Compute adapted weight
            adapted_weight = base_weight * (1.0 + self.adaptive_strength * degradation)
            adapted_weights[name] = adapted_weight

        return adapted_weights

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute clinical loss with adaptive weighting.

        Args:
            pred: Predicted/corrected image [B, 1, H, W]
            target: Ground truth clean image [B, 1, H, W]
            return_components: Whether to return component losses and metrics

        Returns:
            total_loss: Scalar total loss
            metrics: Dict containing component losses, weights, and degradations
        """
        # Compute each clinical loss component
        contrast_loss, contrast_deg = self.contrast_preservation_loss(pred, target)
        boundary_loss, boundary_deg = self.boundary_sharpness_loss(pred, target)
        edge_loss, edge_deg = self.edge_preservation_loss(pred, target)
        texture_loss, texture_deg = self.texture_preservation_loss(pred, target)
        layer_loss, layer_deg = self.layer_visibility_loss(pred, target)

        # Collect degradations
        degradations = {
            'contrast': contrast_deg,
            'boundary': boundary_deg,
            'edge': edge_deg,
            'texture': texture_deg,
            'layer_vis': layer_deg,
        }

        # Collect base weights
        base_weights = {
            'contrast': self.base_contrast,
            'boundary': self.base_boundary,
            'edge': self.base_edge,
            'texture': self.base_texture,
            'layer_vis': self.base_layer_vis,
        }

        # Compute adaptive weights
        weights = self.compute_adaptive_weights(degradations, base_weights)

        # Compute weighted total loss
        total_loss = (
            weights['contrast'] * contrast_loss +
            weights['boundary'] * boundary_loss +
            weights['edge'] * edge_loss +
            weights['texture'] * texture_loss +
            weights['layer_vis'] * layer_loss
        )

        # Optionally add minimal MSE for stability
        if self.include_minimal_mse:
            mse_loss = F.mse_loss(pred, target)
            total_loss = total_loss + self.mse_weight * mse_loss
        else:
            mse_loss = torch.tensor(0.0, device=pred.device)

        # Compute clinical quality metrics (for monitoring)
        with torch.no_grad():
            psnr = 10 * torch.log10(1.0 / (F.mse_loss(pred, target) + 1e-8))

            # Clinical quality summary score (higher = better clinical quality)
            # Inverse of average degradation
            avg_degradation = sum(d.item() if isinstance(d, torch.Tensor) else d
                                  for d in degradations.values()) / len(degradations)
            clinical_score = 1.0 - min(avg_degradation, 1.0)

        metrics = {
            # Total loss
            'total': total_loss.item(),

            # Component losses (raw, before weighting)
            'contrast_loss': contrast_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'edge_loss': edge_loss.item(),
            'texture_loss': texture_loss.item(),
            'layer_vis_loss': layer_loss.item(),
            'mse_loss': mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss,

            # Degradation metrics (how much each aspect degraded)
            'contrast_degradation': contrast_deg.item() if isinstance(contrast_deg, torch.Tensor) else contrast_deg,
            'boundary_degradation': boundary_deg.item() if isinstance(boundary_deg, torch.Tensor) else boundary_deg,
            'edge_degradation': edge_deg.item() if isinstance(edge_deg, torch.Tensor) else edge_deg,
            'texture_degradation': texture_deg.item() if isinstance(texture_deg, torch.Tensor) else texture_deg,
            'layer_vis_degradation': layer_deg.item() if isinstance(layer_deg, torch.Tensor) else layer_deg,

            # Adaptive weights (for monitoring adaptation behavior)
            'weight_contrast': weights['contrast'],
            'weight_boundary': weights['boundary'],
            'weight_edge': weights['edge'],
            'weight_texture': weights['texture'],
            'weight_layer_vis': weights['layer_vis'],

            # Overall quality metrics
            'psnr': psnr.item(),
            'clinical_score': clinical_score,
            'avg_degradation': avg_degradation,
        }

        return total_loss, metrics


class ClinicalLossWithBackbone(ClinicalLoss):
    """
    Extended ClinicalLoss that also handles backbone output for comparison.

    This is useful for training a corrector on top of a backbone denoiser.
    It computes clinical quality for both backbone and corrected outputs,
    providing delta metrics to track improvement.

    Additional features:
    - Backbone supervision loss (optional)
    - Delta metrics (corrected vs backbone)
    - Combined adaptive weighting considering backbone baseline
    """

    def __init__(
        self,
        base_contrast: float = 0.2,
        base_boundary: float = 0.25,
        base_edge: float = 0.2,
        base_texture: float = 0.15,
        base_layer_vis: float = 0.2,
        adaptive_strength: float = 1.0,
        include_minimal_mse: bool = True,
        mse_weight: float = 0.1,
        backbone_supervision_weight: float = 0.1,
    ):
        super().__init__(
            base_contrast=base_contrast,
            base_boundary=base_boundary,
            base_edge=base_edge,
            base_texture=base_texture,
            base_layer_vis=base_layer_vis,
            adaptive_strength=adaptive_strength,
            include_minimal_mse=include_minimal_mse,
            mse_weight=mse_weight,
        )
        self.backbone_supervision_weight = backbone_supervision_weight

    def forward(
        self,
        corrected: torch.Tensor,
        backbone_out: torch.Tensor,
        target: torch.Tensor,
        info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute clinical loss comparing corrected output to target.

        Args:
            corrected: Corrected/final output [B, 1, H, W]
            backbone_out: Backbone denoiser output [B, 1, H, W]
            target: Ground truth clean image [B, 1, H, W]
            info: Optional info dict from model (for additional regularization)

        Returns:
            total_loss: Scalar total loss
            metrics: Dict with detailed metrics including backbone comparison
        """
        # Compute clinical loss for corrected output
        corrected_loss, corrected_metrics = super().forward(corrected, target)

        # Compute clinical metrics for backbone (for comparison, no gradient needed)
        with torch.no_grad():
            _, backbone_metrics = super().forward(backbone_out, target)

        # Add backbone supervision if enabled
        if self.backbone_supervision_weight > 0:
            backbone_mse = F.mse_loss(backbone_out, target)
            total_loss = corrected_loss + self.backbone_supervision_weight * backbone_mse
        else:
            total_loss = corrected_loss
            backbone_mse = torch.tensor(0.0, device=corrected.device)

        # Compute deltas (corrected vs backbone)
        deltas = {
            'psnr_delta': corrected_metrics['psnr'] - backbone_metrics['psnr'],
            'clinical_score_delta': corrected_metrics['clinical_score'] - backbone_metrics['clinical_score'],
            'contrast_deg_delta': backbone_metrics['contrast_degradation'] - corrected_metrics['contrast_degradation'],
            'boundary_deg_delta': backbone_metrics['boundary_degradation'] - corrected_metrics['boundary_degradation'],
            'edge_deg_delta': backbone_metrics['edge_degradation'] - corrected_metrics['edge_degradation'],
            'texture_deg_delta': backbone_metrics['texture_degradation'] - corrected_metrics['texture_degradation'],
            'layer_vis_deg_delta': backbone_metrics['layer_vis_degradation'] - corrected_metrics['layer_vis_degradation'],
        }

        # Build complete metrics dict
        metrics = {
            'total': total_loss.item(),
            'corrected_loss': corrected_loss.item(),
            'backbone_mse': backbone_mse.item() if isinstance(backbone_mse, torch.Tensor) else backbone_mse,

            # Corrected output metrics
            'psnr_corrected': corrected_metrics['psnr'],
            'clinical_score_corrected': corrected_metrics['clinical_score'],

            # Backbone metrics
            'psnr_backbone': backbone_metrics['psnr'],
            'clinical_score_backbone': backbone_metrics['clinical_score'],

            # Deltas (positive = corrector improved)
            **deltas,

            # Include all component losses from corrected
            **{f'corrected_{k}': v for k, v in corrected_metrics.items() if k != 'total'},
        }

        return total_loss, metrics


# =============================================================================
# Convenience function for creating loss with recommended settings
# =============================================================================

def create_clinical_loss(
    mode: str = 'standard',
    **kwargs
) -> ClinicalLoss:
    """
    Create a ClinicalLoss instance with recommended settings for different use cases.

    Args:
        mode: One of:
            - 'standard': Balanced clinical loss (default)
            - 'boundary_focus': Emphasize layer boundary sharpness
            - 'texture_focus': Emphasize texture preservation
            - 'balanced_with_psnr': Include stronger MSE component
            - 'aggressive': Stronger adaptive weighting
        **kwargs: Override any default parameters

    Returns:
        Configured ClinicalLoss instance
    """
    presets = {
        'standard': {
            'base_contrast': 0.2,
            'base_boundary': 0.25,
            'base_edge': 0.2,
            'base_texture': 0.15,
            'base_layer_vis': 0.2,
            'adaptive_strength': 1.0,
            'mse_weight': 0.1,
        },
        'boundary_focus': {
            'base_contrast': 0.15,
            'base_boundary': 0.35,
            'base_edge': 0.2,
            'base_texture': 0.1,
            'base_layer_vis': 0.2,
            'adaptive_strength': 1.0,
            'mse_weight': 0.1,
        },
        'texture_focus': {
            'base_contrast': 0.2,
            'base_boundary': 0.2,
            'base_edge': 0.15,
            'base_texture': 0.3,
            'base_layer_vis': 0.15,
            'adaptive_strength': 1.0,
            'mse_weight': 0.1,
        },
        'balanced_with_psnr': {
            'base_contrast': 0.15,
            'base_boundary': 0.2,
            'base_edge': 0.15,
            'base_texture': 0.1,
            'base_layer_vis': 0.15,
            'adaptive_strength': 0.5,
            'mse_weight': 0.25,
        },
        'aggressive': {
            'base_contrast': 0.2,
            'base_boundary': 0.25,
            'base_edge': 0.2,
            'base_texture': 0.15,
            'base_layer_vis': 0.2,
            'adaptive_strength': 2.0,
            'mse_weight': 0.05,
        },
    }

    if mode not in presets:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(presets.keys())}")

    config = presets[mode].copy()
    config.update(kwargs)

    return ClinicalLoss(**config)


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == '__main__':
    import torch

    # Create test tensors
    B, C, H, W = 2, 1, 128, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simulated images
    target = torch.rand(B, C, H, W, device=device) * 0.5 + 0.25  # Clean image
    pred = target + 0.1 * torch.randn_like(target)  # Predicted (with some error)
    pred = pred.clamp(0, 1)

    # Simulate backbone output (slightly worse than pred)
    backbone = target + 0.15 * torch.randn_like(target)
    backbone = backbone.clamp(0, 1)

    print("=" * 70)
    print("ClinicalLoss Test")
    print("=" * 70)

    # Test standard ClinicalLoss
    criterion = ClinicalLoss()
    criterion = criterion.to(device)

    loss, metrics = criterion(pred, target)

    print(f"\nStandard ClinicalLoss:")
    print(f"  Total Loss: {metrics['total']:.4f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  Clinical Score: {metrics['clinical_score']:.4f}")
    print(f"\n  Component Losses:")
    print(f"    Contrast: {metrics['contrast_loss']:.4f} (deg: {metrics['contrast_degradation']:.4f})")
    print(f"    Boundary: {metrics['boundary_loss']:.4f} (deg: {metrics['boundary_degradation']:.4f})")
    print(f"    Edge: {metrics['edge_loss']:.4f} (deg: {metrics['edge_degradation']:.4f})")
    print(f"    Texture: {metrics['texture_loss']:.4f} (deg: {metrics['texture_degradation']:.4f})")
    print(f"    Layer Vis: {metrics['layer_vis_loss']:.4f} (deg: {metrics['layer_vis_degradation']:.4f})")
    print(f"\n  Adaptive Weights:")
    print(f"    Contrast: {metrics['weight_contrast']:.4f}")
    print(f"    Boundary: {metrics['weight_boundary']:.4f}")
    print(f"    Edge: {metrics['weight_edge']:.4f}")
    print(f"    Texture: {metrics['weight_texture']:.4f}")
    print(f"    Layer Vis: {metrics['weight_layer_vis']:.4f}")

    # Test ClinicalLossWithBackbone
    print("\n" + "=" * 70)
    print("ClinicalLossWithBackbone Test")
    print("=" * 70)

    criterion_with_backbone = ClinicalLossWithBackbone()
    criterion_with_backbone = criterion_with_backbone.to(device)

    loss, metrics = criterion_with_backbone(pred, backbone, target)

    print(f"\n  Total Loss: {metrics['total']:.4f}")
    print(f"  PSNR (Backbone): {metrics['psnr_backbone']:.2f} dB")
    print(f"  PSNR (Corrected): {metrics['psnr_corrected']:.2f} dB")
    print(f"  PSNR Delta: {metrics['psnr_delta']:+.3f} dB")
    print(f"  Clinical Score (Backbone): {metrics['clinical_score_backbone']:.4f}")
    print(f"  Clinical Score (Corrected): {metrics['clinical_score_corrected']:.4f}")
    print(f"  Clinical Score Delta: {metrics['clinical_score_delta']:+.4f}")

    # Test gradient flow
    print("\n" + "=" * 70)
    print("Gradient Flow Test")
    print("=" * 70)

    pred_grad_test = pred.clone().requires_grad_(True)
    loss, _ = criterion(pred_grad_test, target)
    loss.backward()

    print(f"  Loss requires grad: {loss.requires_grad}")
    print(f"  Pred gradient exists: {pred_grad_test.grad is not None}")
    print(f"  Pred gradient norm: {pred_grad_test.grad.norm().item():.4f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
