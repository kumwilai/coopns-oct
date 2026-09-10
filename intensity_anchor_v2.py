#!/usr/bin/env python3
"""
IntensityAnchoredBoundaryLoss V2 - Improved Self-Supervised Boundary Learning

Improvements over V1:
1. Multi-scale edge detection (better boundary localization)
2. Confidence-weighted losses (trust high-confidence detections more)
3. Gradient magnitude anchoring (boundaries should have strong edges)
4. Layer proportion constraints (expected thickness ratios)
5. Column-wise smoothness (boundaries should be smooth)
6. Middle boundary interpolation (not just ILM/RPE anchoring)
7. Robust retina detection (ensemble of methods)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class IntensityAnchoredBoundaryLossV2(nn.Module):
    """
    Improved self-supervised boundary loss with:
    - Multi-scale edge detection
    - Confidence weighting
    - Layer proportion constraints
    - Boundary smoothness
    """

    def __init__(
        self,
        # Anchor weights
        lambda_ilm_anchor: float = 1.0,
        lambda_rpe_anchor: float = 1.0,
        lambda_middle_anchor: float = 0.5,  # NEW: middle boundary anchoring
        # Edge/gradient weights
        lambda_edge_align: float = 0.5,
        lambda_gradient_magnitude: float = 0.3,  # NEW: encourage strong gradients at boundaries
        # Constraint weights
        lambda_intensity_order: float = 0.3,
        lambda_layer_proportion: float = 0.3,  # NEW: expected layer thickness ratios
        lambda_smoothness: float = 0.2,  # NEW: boundary smoothness
        # Detection settings
        use_multiscale_edges: bool = True,  # NEW: multi-scale edge detection
        use_confidence_weighting: bool = True,  # NEW: weight by detection confidence
    ):
        super().__init__()

        # Weights
        self.lambda_ilm_anchor = lambda_ilm_anchor
        self.lambda_rpe_anchor = lambda_rpe_anchor
        self.lambda_middle_anchor = lambda_middle_anchor
        self.lambda_edge_align = lambda_edge_align
        self.lambda_gradient_magnitude = lambda_gradient_magnitude
        self.lambda_intensity_order = lambda_intensity_order
        self.lambda_layer_proportion = lambda_layer_proportion
        self.lambda_smoothness = lambda_smoothness

        # Settings
        self.use_multiscale_edges = use_multiscale_edges
        self.use_confidence_weighting = use_confidence_weighting

        # Multi-scale Sobel filters for edge detection
        # Scale 1: 3x3 (fine details)
        self.register_buffer('sobel_y_3', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

        # Scale 2: 5x5 (medium features)
        self.register_buffer('sobel_y_5', torch.tensor([
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32).view(1, 1, 5, 5) / 48.0)

        # Laplacian of Gaussian approximation (for blob/edge detection)
        self.register_buffer('log_kernel', torch.tensor([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=torch.float32).view(1, 1, 5, 5) / 16.0)

        # Expected layer thickness proportions (based on OCT anatomy)
        # RNFL:INL:IS_OS:RPE_Choroid ≈ 0.15:0.35:0.25:0.25
        self.expected_proportions = torch.tensor([0.15, 0.35, 0.25, 0.25])

    def detect_retina_band_robust(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Robust retina detection using ensemble of methods.

        Returns:
            retina_top: [B, W] detected ILM position
            retina_bottom: [B, W] detected RPE position
            confidence: [B, W] detection confidence (0-1)
        """
        B, C, H, W = image.shape
        device = image.device

        # Method 1: Intensity-based (adaptive threshold)
        intensity = image[:, 0, :, :]  # [B, H, W]

        # Smooth along height
        kernel_size = 5
        if H > kernel_size:
            intensity_reshaped = intensity.permute(0, 2, 1).reshape(B * W, 1, H)
            smoothed = F.avg_pool1d(
                intensity_reshaped,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            )
            intensity_smooth = smoothed.reshape(B, W, H).permute(0, 2, 1)  # [B, H, W]
        else:
            intensity_smooth = intensity

        # Adaptive threshold per image
        img_mean = intensity_smooth.mean(dim=(1, 2), keepdim=True)
        img_std = intensity_smooth.std(dim=(1, 2), keepdim=True)
        threshold = img_mean + 0.5 * img_std
        bright_mask = intensity_smooth > threshold

        # Method 2: Gradient-based (find strong vertical transitions)
        padded = F.pad(image, (1, 1, 1, 1), mode='replicate')
        grad_y = F.conv2d(padded, self.sobel_y_3.to(device), padding=0)[:, 0, :, :]  # [B, H, W]

        # Top of retina: strong negative-to-positive gradient (dark to bright)
        # Bottom of retina: strong positive-to-negative gradient (bright to dark)
        grad_positive = F.relu(grad_y)  # Dark-to-bright transitions
        grad_negative = F.relu(-grad_y)  # Bright-to-dark transitions

        # Combine intensity and gradient methods
        # Weight gradient by intensity (only count gradients in bright regions)
        top_score = grad_positive * intensity_smooth  # [B, H, W]
        bottom_score = grad_negative * intensity_smooth

        # Find peaks along height dimension
        row_indices = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

        # Soft argmax for top boundary (weighted by top_score)
        top_weights = F.softmax(top_score * 10, dim=1)  # Temperature=10 for sharper peaks
        retina_top_grad = (top_weights * row_indices).sum(dim=1) / (H - 1)  # [B, W]

        # Soft argmax for bottom boundary
        bottom_weights = F.softmax(bottom_score * 10, dim=1)
        retina_bottom_grad = (bottom_weights * row_indices).sum(dim=1) / (H - 1)  # [B, W]

        # Method 1 result: percentile-based from bright mask
        cumsum = bright_mask.float().cumsum(dim=1)
        total_bright = cumsum[:, -1, :].clamp(min=1)

        target_5pct = total_bright * 0.05
        target_95pct = total_bright * 0.95

        above_5pct = cumsum >= target_5pct.unsqueeze(1)
        above_95pct = cumsum >= target_95pct.unsqueeze(1)

        retina_top_int = above_5pct.float().argmax(dim=1).float() / (H - 1)
        retina_bottom_int = above_95pct.float().argmax(dim=1).float() / (H - 1)

        # Ensemble: weighted average of both methods
        # Trust gradient method more when gradients are strong
        grad_strength = (top_score.max(dim=1).values + bottom_score.max(dim=1).values) / 2
        grad_weight = torch.sigmoid(grad_strength * 5 - 1)  # 0-1 based on gradient strength

        retina_top = grad_weight * retina_top_grad + (1 - grad_weight) * retina_top_int
        retina_bottom = grad_weight * retina_bottom_grad + (1 - grad_weight) * retina_bottom_int

        # Ensure top < bottom
        retina_bottom = torch.maximum(retina_bottom, retina_top + 0.1)

        # Compute confidence based on:
        # 1. Band width (should be reasonable: 20-60% of image)
        # 2. Gradient strength at boundaries
        # 3. Intensity contrast
        band_width = retina_bottom - retina_top
        width_confidence = 1 - torch.abs(band_width - 0.4) * 2  # Peak at 40% width
        width_confidence = width_confidence.clamp(0, 1)

        contrast = intensity_smooth.max(dim=1).values - intensity_smooth.min(dim=1).values
        contrast_confidence = (contrast / contrast.max()).clamp(0, 1)

        confidence = (width_confidence + contrast_confidence + grad_weight) / 3

        return retina_top, retina_bottom, confidence

    def compute_multiscale_edges(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute edges at multiple scales and combine.

        Returns:
            edges: [B, 1, H, W] combined edge map
        """
        device = image.device

        # Scale 1: 3x3 Sobel
        padded_3 = F.pad(image, (1, 1, 1, 1), mode='replicate')
        edges_3 = torch.abs(F.conv2d(padded_3, self.sobel_y_3.to(device), padding=0))

        if self.use_multiscale_edges:
            # Scale 2: 5x5 Sobel
            padded_5 = F.pad(image, (2, 2, 2, 2), mode='replicate')
            edges_5 = torch.abs(F.conv2d(padded_5, self.sobel_y_5.to(device), padding=0))

            # Scale 3: Laplacian of Gaussian
            edges_log = torch.abs(F.conv2d(padded_5, self.log_kernel.to(device), padding=0))

            # Combine: weighted sum favoring fine details
            edges = 0.5 * edges_3 + 0.3 * edges_5 + 0.2 * edges_log
        else:
            edges = edges_3

        return edges

    def ilm_anchor_loss(
        self,
        boundaries: torch.Tensor,
        retina_top: torch.Tensor,
        confidence: torch.Tensor,
        tolerance: float = 0.02,
    ) -> torch.Tensor:
        """ILM should align with detected retina top."""
        ilm = boundaries[:, 0, :]  # [B, W]

        deviation = torch.abs(ilm - retina_top)
        loss = F.relu(deviation - tolerance)

        if self.use_confidence_weighting:
            loss = loss * confidence

        return loss.mean()

    def rpe_anchor_loss(
        self,
        boundaries: torch.Tensor,
        retina_bottom: torch.Tensor,
        confidence: torch.Tensor,
        tolerance: float = 0.02,
    ) -> torch.Tensor:
        """RPE should align with detected retina bottom."""
        rpe = boundaries[:, -1, :]  # [B, W]

        # RPE at ~95% of detected bottom (leave room for choroid)
        rpe_target = retina_bottom * 0.95

        deviation = torch.abs(rpe - rpe_target)
        loss = F.relu(deviation - tolerance)

        if self.use_confidence_weighting:
            loss = loss * confidence

        return loss.mean()

    def middle_boundary_anchor_loss(
        self,
        boundaries: torch.Tensor,
        retina_top: torch.Tensor,
        retina_bottom: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        NEW: Anchor middle boundaries based on expected layer proportions.

        Given ILM and RPE positions, interpolate expected positions for
        RNFL_INL and INL_ISOS boundaries based on anatomical proportions.
        """
        B, num_boundaries, W = boundaries.shape
        device = boundaries.device

        # Get current ILM and RPE
        ilm = boundaries[:, 0, :]  # [B, W]
        rpe = boundaries[:, -1, :]  # [B, W]

        # Compute expected positions based on proportions
        retina_height = rpe - ilm  # [B, W]

        # Expected boundary positions (cumulative proportions)
        # b0 = ILM (top)
        # b1 = ILM + 15% (after RNFL)
        # b2 = ILM + 15% + 35% = ILM + 50% (after INL)
        # b3 = RPE (bottom)
        expected_b1 = ilm + retina_height * 0.15  # RNFL_INL
        expected_b2 = ilm + retina_height * 0.50  # INL_ISOS

        # Loss for middle boundaries
        b1_deviation = torch.abs(boundaries[:, 1, :] - expected_b1)
        b2_deviation = torch.abs(boundaries[:, 2, :] - expected_b2)

        loss = (b1_deviation + b2_deviation) / 2

        if self.use_confidence_weighting:
            loss = loss * confidence

        return loss.mean()

    def edge_alignment_loss(
        self,
        boundaries: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        """Boundaries should align with intensity edges."""
        B, C, H, W = edges.shape
        device = edges.device
        num_boundaries = boundaries.shape[1]

        # Convert to pixel positions
        boundaries_px = (boundaries * (H - 1)).long()
        boundaries_px = torch.clamp(boundaries_px, 0, H - 1)

        # Gather edge values at boundary positions
        col_indices = torch.arange(W, device=device).view(1, 1, W).expand(B, num_boundaries, W)
        edges_flat = edges[:, 0, :, :].reshape(B, H * W)
        flat_indices = boundaries_px * W + col_indices

        edge_at_boundaries = torch.gather(
            edges_flat.unsqueeze(1).expand(-1, num_boundaries, -1),
            dim=2,
            index=flat_indices
        )

        # Normalize and compute loss (higher edge = better)
        max_edge = edges.max() + 1e-8
        normalized_edges = edge_at_boundaries / max_edge

        return 1.0 - normalized_edges.mean()

    def gradient_magnitude_loss(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        NEW: Encourage strong gradient magnitude at boundary positions.

        Boundaries should occur where there are strong intensity transitions.
        """
        B, C, H, W = image.shape
        device = image.device
        num_boundaries = boundaries.shape[1]

        # Compute gradient magnitude
        padded = F.pad(image, (1, 1, 1, 1), mode='replicate')
        grad_y = F.conv2d(padded, self.sobel_y_3.to(device), padding=0)
        grad_mag = torch.abs(grad_y)  # [B, 1, H, W]

        # Sample gradient at boundary positions
        boundaries_px = (boundaries * (H - 1)).long().clamp(0, H - 1)
        col_indices = torch.arange(W, device=device).view(1, 1, W).expand(B, num_boundaries, W)

        grad_flat = grad_mag[:, 0, :, :].reshape(B, H * W)
        flat_indices = boundaries_px * W + col_indices

        grad_at_boundaries = torch.gather(
            grad_flat.unsqueeze(1).expand(-1, num_boundaries, -1),
            dim=2,
            index=flat_indices
        )

        # Normalize and return loss (want high gradient)
        max_grad = grad_mag.max() + 1e-8
        normalized_grad = grad_at_boundaries / max_grad

        return 1.0 - normalized_grad.mean()

    def layer_proportion_loss(
        self,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        NEW: Encourage layer thicknesses to match expected anatomical proportions.
        """
        B, num_boundaries, W = boundaries.shape
        device = boundaries.device

        # Compute layer thicknesses
        thicknesses = []
        for i in range(num_boundaries):
            if i == 0:
                # First layer: from boundary 0 to boundary 1
                thickness = boundaries[:, 1, :] - boundaries[:, 0, :]
            elif i < num_boundaries - 1:
                thickness = boundaries[:, i + 1, :] - boundaries[:, i, :]
            else:
                # Last layer: from boundary 3 to end (estimate as same as boundary 3 to 1.0)
                thickness = 1.0 - boundaries[:, -1, :]
            thicknesses.append(thickness)

        # Stack: [B, 4, W]
        thicknesses = torch.stack(thicknesses, dim=1)

        # Normalize to proportions
        total_thickness = thicknesses.sum(dim=1, keepdim=True).clamp(min=1e-8)
        proportions = thicknesses / total_thickness  # [B, 4, W]

        # Compare to expected proportions
        expected = self.expected_proportions.to(device).view(1, 4, 1)

        # L1 loss on proportions
        loss = torch.abs(proportions - expected).mean()

        return loss

    def boundary_smoothness_loss(
        self,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        NEW: Encourage smooth boundaries across columns (total variation).
        """
        # Compute horizontal differences
        diff = boundaries[:, :, 1:] - boundaries[:, :, :-1]  # [B, 4, W-1]

        # L1 total variation
        tv_loss = torch.abs(diff).mean()

        return tv_loss

    def intensity_ordering_loss(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce expected layer brightness ordering."""
        B, C, H, W = image.shape
        device = image.device

        boundaries_px = (boundaries * (H - 1)).long().clamp(0, H - 1)
        image_2d = image[:, 0, :, :]

        layer_intensities = []

        for i in range(4):
            if i == 0:
                top = boundaries_px[:, 0, :]
                bottom = boundaries_px[:, 1, :]
            elif i < 3:
                top = boundaries_px[:, i, :]
                bottom = boundaries_px[:, i + 1, :]
            else:
                top = boundaries_px[:, 3, :]
                bottom = torch.full((B, W), H - 1, device=device, dtype=torch.long)

            # Create mask
            top_exp = top.unsqueeze(1)
            bottom_exp = bottom.unsqueeze(1)
            row_idx = torch.arange(H, device=device).view(1, H, 1)

            layer_mask = (row_idx >= top_exp) & (row_idx < bottom_exp)
            masked_intensity = image_2d * layer_mask.float()
            total_intensity = masked_intensity.sum(dim=(1, 2))
            total_pixels = layer_mask.sum(dim=(1, 2)).float().clamp(min=1)
            layer_mean = total_intensity / total_pixels

            layer_intensities.append(layer_mean)

        layer_stack = torch.stack(layer_intensities, dim=0)

        # Ordering constraints: RNFL > INL, IS_OS > INL
        loss = torch.zeros(1, device=device)
        loss = loss + F.relu(0.3 * (layer_stack[1] - layer_stack[0])).mean()  # RNFL > INL
        loss = loss + F.relu(0.3 * (layer_stack[1] - layer_stack[2])).mean()  # IS_OS > INL

        return loss / 2

    def forward(
        self,
        boundaries: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all losses.

        Args:
            boundaries: [B, 4, W] boundary positions (normalized 0-1)
            image: [B, 1, H, W] OCT image

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
        """
        device = boundaries.device

        # Detect retina with confidence
        retina_top, retina_bottom, confidence = self.detect_retina_band_robust(image)

        # Compute multi-scale edges
        edges = self.compute_multiscale_edges(image)

        # Compute all losses
        losses = {}

        # Anchor losses
        losses['ilm_anchor'] = self.ilm_anchor_loss(boundaries, retina_top, confidence)
        losses['rpe_anchor'] = self.rpe_anchor_loss(boundaries, retina_bottom, confidence)
        losses['middle_anchor'] = self.middle_boundary_anchor_loss(
            boundaries, retina_top, retina_bottom, confidence
        )

        # Edge/gradient losses
        losses['edge_align'] = self.edge_alignment_loss(boundaries, edges)
        losses['gradient_mag'] = self.gradient_magnitude_loss(boundaries, image)

        # Constraint losses
        losses['intensity_order'] = self.intensity_ordering_loss(boundaries, image)
        losses['layer_proportion'] = self.layer_proportion_loss(boundaries)
        losses['smoothness'] = self.boundary_smoothness_loss(boundaries)

        # Total weighted loss
        total = (
            self.lambda_ilm_anchor * losses['ilm_anchor'] +
            self.lambda_rpe_anchor * losses['rpe_anchor'] +
            self.lambda_middle_anchor * losses['middle_anchor'] +
            self.lambda_edge_align * losses['edge_align'] +
            self.lambda_gradient_magnitude * losses['gradient_mag'] +
            self.lambda_intensity_order * losses['intensity_order'] +
            self.lambda_layer_proportion * losses['layer_proportion'] +
            self.lambda_smoothness * losses['smoothness']
        )

        losses['total'] = total

        # Add detection info
        losses['detected_retina_top'] = retina_top.mean().item()
        losses['detected_retina_bottom'] = retina_bottom.mean().item()
        losses['detection_confidence'] = confidence.mean().item()

        return total, {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing IntensityAnchoredBoundaryLossV2")
    print("=" * 60)

    # Create test data
    B, C, H, W = 2, 1, 128, 128
    device = torch.device('cpu')

    # Synthetic OCT image
    image = torch.zeros(B, C, H, W)
    for b in range(B):
        image[b, 0, :, :] = torch.rand(H, W) * 0.1
        image[b, 0, 40:90, :] = 0.5 + torch.rand(50, W) * 0.3

    # Boundaries
    boundaries = torch.zeros(B, 4, W)
    boundaries[:, 0, :] = 0.33
    boundaries[:, 1, :] = 0.40
    boundaries[:, 2, :] = 0.50
    boundaries[:, 3, :] = 0.60

    # Create loss
    loss_fn = IntensityAnchoredBoundaryLossV2()

    # Forward
    import time
    start = time.time()
    total_loss, loss_dict = loss_fn(boundaries, image)
    elapsed = time.time() - start

    print(f"\nForward time: {elapsed:.3f}s")
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print("\nV2 improvements:")
    print("  - Multi-scale edge detection")
    print("  - Confidence-weighted losses")
    print("  - Middle boundary anchoring")
    print("  - Gradient magnitude loss")
    print("  - Layer proportion constraints")
    print("  - Boundary smoothness")
