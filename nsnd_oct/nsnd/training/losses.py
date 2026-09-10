"""
Loss functions for NSND training

Includes:
- Blind2Unblind (B2U) self-supervised loss
- Symbolic consistency loss
- Combined NSND loss
- Clinical importance weighted loss (for TMI)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


# =============================================================================
# Clinical Importance Weights for Retinal Layers
# =============================================================================
# These weights reflect the clinical significance of each retinal layer.
# Higher weights = more important for clinical diagnosis = more denoising effort.
#
# Rationale:
# - RNFL_GCL (2.0): Critical for glaucoma diagnosis. Thinning of even a few
#   microns is clinically significant. Most important layer.
# - IS_OS (1.5): Ellipsoid zone integrity directly correlates with visual acuity.
#   Disruption is key biomarker in AMD, macular holes, etc.
# - RPE_Choroid (1.2): Important for AMD (drusen, geographic atrophy, CNV).
# - INL_OPL (1.0): Moderate importance for diabetic macular edema.
# - ONL (1.0): Moderate importance for inherited retinal diseases.
# =============================================================================

DEFAULT_CLINICAL_WEIGHTS = {
    'RNFL_GCL': 2.0,      # Glaucoma - highest clinical priority
    'INL_OPL': 1.0,       # Diabetic macular edema
    'ONL': 1.0,           # Photoreceptor diseases
    'IS_OS': 1.5,         # Visual acuity correlation
    'RPE_Choroid': 1.2,   # AMD, drusen, CNV
}

LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']


class ClinicalWeightedMSELoss(nn.Module):
    """
    Clinical Importance Weighted MSE Loss for OCT Denoising.

    KEY TMI CONTRIBUTION: Weight denoising loss by clinical importance of
    each retinal layer. This ensures the model prioritizes accurate denoising
    in clinically critical regions (e.g., RNFL for glaucoma, IS/OS for visual acuity).

    The loss is computed as:
        L = sum(weight_map * (pred - target)^2) / sum(weight_map)

    where weight_map[i,j] = clinical_weight[layer[i,j]]

    Args:
        clinical_weights: Dict mapping layer names to importance weights.
                         Default uses evidence-based clinical priorities.
        num_classes: Number of segmentation classes (default 5 for retinal layers).
        normalize: If True, normalize weights so mean=1 (preserves loss scale).
    """

    def __init__(
        self,
        clinical_weights: Optional[Dict[str, float]] = None,
        num_classes: int = 5,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

        # Use default clinical weights if not provided
        weights_dict = clinical_weights or DEFAULT_CLINICAL_WEIGHTS

        # Convert to tensor in layer order
        weights_list = [weights_dict.get(name, 1.0) for name in LAYER_NAMES]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

        # Normalize so mean = 1 (preserves overall loss scale)
        if normalize:
            weights_tensor = weights_tensor / weights_tensor.mean()

        # Register as buffer (moves with model to device, but not a parameter)
        self.register_buffer('layer_weights', weights_tensor)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute clinical importance weighted MSE loss.

        Args:
            pred: Predicted (denoised) image [B, 1, H, W]
            target: Target (clean) image [B, 1, H, W]
            seg_mask: Segmentation mask [B, H, W] with values 0 to num_classes-1

        Returns:
            loss: Scalar weighted MSE loss
        """
        # Create per-pixel weight map from segmentation
        # weight_map[b, h, w] = layer_weights[seg_mask[b, h, w]]
        weight_map = self.layer_weights[seg_mask]  # [B, H, W]
        weight_map = weight_map.unsqueeze(1)  # [B, 1, H, W]

        # Compute weighted MSE
        squared_error = (pred - target) ** 2  # [B, 1, H, W]
        weighted_error = weight_map * squared_error

        # Normalize by sum of weights (not number of pixels)
        # This ensures loss scale is independent of layer distribution
        loss = weighted_error.sum() / (weight_map.sum() + 1e-8)

        return loss

    def get_per_layer_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute loss breakdown by layer (for monitoring/debugging).

        Returns:
            Dict mapping layer names to their contribution to total loss.
        """
        per_layer_loss = {}
        squared_error = (pred - target) ** 2

        for c, name in enumerate(LAYER_NAMES):
            layer_mask = (seg_mask == c).unsqueeze(1).float()
            n_pixels = layer_mask.sum()

            if n_pixels > 0:
                layer_mse = (squared_error * layer_mask).sum() / n_pixels
                weighted_mse = layer_mse * self.layer_weights[c]
                per_layer_loss[name] = {
                    'mse': layer_mse.item(),
                    'weight': self.layer_weights[c].item(),
                    'weighted_mse': weighted_mse.item(),
                    'n_pixels': n_pixels.item(),
                }
            else:
                per_layer_loss[name] = None

        return per_layer_loss


class ClinicalWeightedL1Loss(nn.Module):
    """
    Clinical Importance Weighted L1 Loss (alternative to MSE).

    Same concept as ClinicalWeightedMSELoss but uses L1 (absolute error)
    which is more robust to outliers.
    """

    def __init__(
        self,
        clinical_weights: Optional[Dict[str, float]] = None,
        num_classes: int = 5,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights_dict = clinical_weights or DEFAULT_CLINICAL_WEIGHTS
        weights_list = [weights_dict.get(name, 1.0) for name in LAYER_NAMES]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

        if normalize:
            weights_tensor = weights_tensor / weights_tensor.mean()

        self.register_buffer('layer_weights', weights_tensor)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clinical importance weighted L1 loss."""
        weight_map = self.layer_weights[seg_mask].unsqueeze(1)
        abs_error = torch.abs(pred - target)
        weighted_error = weight_map * abs_error
        loss = weighted_error.sum() / (weight_map.sum() + 1e-8)
        return loss


class ClinicalGateDiversityLoss(nn.Module):
    """
    Encourage layer-specific gates to match clinical importance.

    This loss penalizes when clinically important layers (RNFL, IS/OS)
    have low gate values (insufficient refinement).

    The intuition: If a layer is clinically important, its gate should
    be high enough to allow meaningful refinement, regardless of noise level.
    """

    def __init__(
        self,
        clinical_weights: Optional[Dict[str, float]] = None,
        min_gate_threshold: float = 0.3,
    ):
        super().__init__()
        self.min_gate_threshold = min_gate_threshold

        weights_dict = clinical_weights or DEFAULT_CLINICAL_WEIGHTS
        weights_list = [weights_dict.get(name, 1.0) for name in LAYER_NAMES]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

        # Normalize to [0, 1] range for gate targets
        weights_tensor = weights_tensor / weights_tensor.max()

        self.register_buffer('target_gates', weights_tensor)

    def forward(
        self,
        layer_gates: torch.Tensor,
        seg_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute clinical gate alignment loss.

        Args:
            layer_gates: Per-layer gate outputs [B, 5, H, W]
            seg_probs: Segmentation probabilities [B, 5, H, W]

        Returns:
            loss: Scalar loss encouraging gates to match clinical importance
        """
        loss = 0.0

        for c in range(len(LAYER_NAMES)):
            # Get average gate value for this layer
            layer_mask = seg_probs[:, c:c+1, :, :]
            mask_sum = layer_mask.sum()

            if mask_sum > 100:  # Only compute if enough pixels
                avg_gate = (layer_gates[:, c:c+1, :, :] * layer_mask).sum() / mask_sum
                target = self.target_gates[c] * self.min_gate_threshold + self.min_gate_threshold

                # Penalize if gate is below clinical target
                # One-sided loss: only penalize low gates, not high gates
                shortfall = F.relu(target - avg_gate)
                loss = loss + shortfall * self.target_gates[c]  # Weight by importance

        return loss / len(LAYER_NAMES)


class Blind2UnblindLoss(nn.Module):
    """
    Blind2Unblind loss for self-supervised training

    The model predicts clean image from masked noisy input,
    and is supervised on the unmasked (visible) regions
    """

    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type: 'l1', 'l2', or 'charbonnier'
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute B2U loss

        Args:
            pred: Predicted clean image [B, 1, H, W]
            target: Noisy target (for unmasked regions) [B, 1, H, W]
            mask: Binary mask [B, 1, H, W] (1 = masked, 0 = visible)

        Returns:
            loss: Scalar loss
        """
        # Only supervise on visible (unmasked) regions
        visible_mask = 1.0 - mask

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred * visible_mask, target * visible_mask, reduction='sum')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred * visible_mask, target * visible_mask, reduction='sum')
        elif self.loss_type == 'charbonnier':
            eps = 1e-3
            diff = pred * visible_mask - target * visible_mask
            loss = torch.sqrt(diff ** 2 + eps ** 2).sum()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Normalize by number of visible pixels
        num_visible = visible_mask.sum() + 1e-8
        loss = loss / num_visible

        return loss


class SymbolicConsistencyLoss(nn.Module):
    """
    Symbolic consistency loss: predicted noise composition
    should match actual residual statistics

    This enforces that the symbolic analyzer produces meaningful decompositions
    """

    def __init__(self, consistency_type: str = 'mse'):
        super().__init__()
        self.consistency_type = consistency_type

    def forward(
        self,
        predicted_weights: Dict[str, torch.Tensor],
        noisy_input: torch.Tensor,
        denoised_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symbolic consistency loss

        Args:
            predicted_weights: Predicted noise composition {'speckle': [B], ...}
            noisy_input: Original noisy image [B, 1, H, W]
            denoised_output: Denoised output [B, 1, H, W]

        Returns:
            loss: Consistency loss
        """
        # Compute residual
        residual = noisy_input - denoised_output

        # Estimate actual noise composition from residual
        actual_weights = self._estimate_noise_composition(residual)

        # Consistency loss: predicted weights should match actual
        loss = 0.0
        for component in ['speckle', 'banding', 'gaussian', 'shot']:
            pred_w = predicted_weights[component]
            actual_w = actual_weights[component]

            if self.consistency_type == 'mse':
                loss = loss + F.mse_loss(pred_w, actual_w)
            elif self.consistency_type == 'kl':
                # KL divergence (treat as distributions)
                loss = loss + F.kl_div(
                    torch.log(pred_w + 1e-8),
                    actual_w,
                    reduction='batchmean'
                )

        return loss / 4.0  # Average over components

    def _estimate_noise_composition(
        self,
        residual: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Heuristically estimate noise composition from residual

        Uses simple statistical features:
        - High CV → speckle
        - Vertical power → banding
        - Low kurtosis → Gaussian
        - Depth correlation → shot
        """
        B = residual.shape[0]
        device = residual.device

        # Compute statistics
        cv = residual.std() / (residual.mean().abs() + 1e-6)
        kurtosis = self._compute_kurtosis(residual)

        # Simple heuristic weights (could be more sophisticated)
        weights = {}

        # Speckle: high CV
        weights['speckle'] = torch.sigmoid(5.0 * (cv - 0.5))

        # Gaussian: low kurtosis
        weights['gaussian'] = torch.sigmoid(-3.0 * (kurtosis - 3.0))

        # Banding: FFT vertical power
        vertical_power = self._compute_vertical_power(residual)
        weights['banding'] = torch.sigmoid(10.0 * (vertical_power - 0.1))

        # Shot: depth correlation (simplified)
        depth_corr = self._compute_depth_correlation(residual)
        weights['shot'] = torch.sigmoid(5.0 * (depth_corr - 0.3))

        # Normalize
        total = sum(weights.values()) + 1e-8
        weights = {k: v / total for k, v in weights.items()}

        # Broadcast to batch dimension
        weights = {k: v.expand(B) for k, v in weights.items()}

        return weights

    def _compute_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis"""
        mean = x.mean()
        std = x.std()
        centered = x - mean
        m4 = (centered ** 4).mean()
        return m4 / (std ** 4 + 1e-8)

    def _compute_vertical_power(self, x: torch.Tensor) -> torch.Tensor:
        """Compute vertical frequency power"""
        fft = torch.fft.fft2(x.squeeze(1))
        magnitude = torch.abs(torch.fft.fftshift(fft, dim=(-2, -1)))

        H = magnitude.shape[-2]
        center_h = H // 2
        vertical_band = magnitude[:, center_h-2:center_h+3, :]

        total_power = magnitude.sum(dim=(-2, -1), keepdim=True)
        return vertical_band.sum(dim=(-2, -1)).mean() / (total_power.mean() + 1e-8)

    def _compute_depth_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute correlation with depth"""
        B, C, H, W = x.shape

        # Create depth coordinate
        depth = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)

        # Flatten
        x_flat = x.reshape(B, -1)
        depth_flat = depth.reshape(B, -1)

        # Pearson correlation
        x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
        d_centered = depth_flat - depth_flat.mean(dim=-1, keepdim=True)

        numerator = (x_centered * d_centered).sum(dim=-1)
        denominator = torch.sqrt(
            (x_centered ** 2).sum(dim=-1) * (d_centered ** 2).sum(dim=-1)
        )

        return (numerator / (denominator + 1e-8)).abs().mean()


class NSND_B2U_Loss(nn.Module):
    """
    Combined NSND loss for self-supervised training

    Combines:
    1. Blind2Unblind reconstruction loss
    2. Symbolic consistency loss
    3. Optional regularization (TV, uncertainty)
    """

    def __init__(
        self,
        lambda_consistency: float = 0.1,
        lambda_tv: float = 1e-5,
        lambda_uncertainty: float = 0.01,
        use_uncertainty: bool = True,
    ):
        """
        Args:
            lambda_consistency: Weight for symbolic consistency
            lambda_tv: Weight for total variation regularization
            lambda_uncertainty: Weight for uncertainty regularization
            use_uncertainty: Whether to use uncertainty loss
        """
        super().__init__()

        self.lambda_consistency = lambda_consistency
        self.lambda_tv = lambda_tv
        self.lambda_uncertainty = lambda_uncertainty
        self.use_uncertainty = use_uncertainty

        self.b2u_loss = Blind2UnblindLoss(loss_type='charbonnier')
        self.consistency_loss = SymbolicConsistencyLoss(consistency_type='mse')

    def forward(
        self,
        pred_clean: torch.Tensor,
        target_noisy: torch.Tensor,
        mask: torch.Tensor,
        symbolic_weights: Dict[str, torch.Tensor],
        noisy_input: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss

        Args:
            pred_clean: Predicted clean image [B, 1, H, W]
            target_noisy: Target noisy image [B, 1, H, W]
            mask: B2U mask [B, 1, H, W]
            symbolic_weights: Predicted noise weights
            noisy_input: Original noisy input [B, 1, H, W]
            uncertainty: Predicted uncertainty map [B, 1, H, W]

        Returns:
            losses: Dict of individual losses and total
        """
        losses = {}

        # 1. Blind2Unblind reconstruction loss
        losses['b2u'] = self.b2u_loss(pred_clean, target_noisy, mask)

        # 2. Symbolic consistency loss
        losses['consistency'] = self.consistency_loss(
            symbolic_weights,
            noisy_input,
            pred_clean
        )

        # 3. Total variation regularization
        losses['tv'] = self._total_variation(pred_clean)

        # 4. Uncertainty regularization (if available)
        if self.use_uncertainty and uncertainty is not None:
            # Encourage low uncertainty in homogeneous regions
            # High uncertainty near edges (detected via gradient)
            grad_magnitude = self._gradient_magnitude(pred_clean)
            losses['uncertainty'] = F.mse_loss(uncertainty, grad_magnitude.detach())
        else:
            losses['uncertainty'] = torch.tensor(0.0, device=pred_clean.device)

        # Total loss
        losses['total'] = (
            losses['b2u']
            + self.lambda_consistency * losses['consistency']
            + self.lambda_tv * losses['tv']
            + self.lambda_uncertainty * losses['uncertainty']
        )

        return losses

    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation regularization"""
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return (diff_h.abs().mean() + diff_w.abs().mean()) / 2.0

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude for uncertainty supervision"""
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), sobel_x)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), sobel_y)

        magnitude = torch.sqrt(gx ** 2 + gy ** 2)

        # Normalize to [0, 1]
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        return magnitude


# =============================================================================
# CUAP-OCT: Clinically-guided Uncertainty-Aware Pathology-preserving Losses
# =============================================================================
# KEY TMI CONTRIBUTION: Unified framework where clinical importance, uncertainty,
# and pathology preservation are jointly optimized.
# =============================================================================

class UncertaintyCalibrationLoss(nn.Module):
    """
    CUAP-OCT: Uncertainty Calibration Loss

    KEY NOVEL CONTRIBUTION: Uncertainty should be calibrated to clinical needs:
    - HIGH uncertainty at layer boundaries (segmentation ambiguity)
    - HIGH uncertainty at pathology (lesions, drusen - flag for review)
    - LOW uncertainty in clinically critical regions where we MUST be confident

    This creates a clinically-meaningful uncertainty that helps clinicians
    know where to focus their attention.
    """

    def __init__(self, boundary_weight: float = 1.0, pathology_weight: float = 1.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.pathology_weight = pathology_weight

    def forward(
        self,
        uncertainty: torch.Tensor,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        seg_mask: torch.Tensor,
        seg_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute uncertainty calibration loss.

        Args:
            uncertainty: Predicted uncertainty [B, 1, H, W]
            denoised: Denoised output [B, 1, H, W]
            clean: Clean target [B, 1, H, W]
            seg_mask: Segmentation mask [B, H, W]
            seg_probs: Segmentation probabilities [B, 5, H, W]

        Returns:
            loss: Calibration loss encouraging meaningful uncertainty
        """
        B, _, H, W = uncertainty.shape
        device = uncertainty.device

        # 1. Boundary uncertainty: should be HIGH at layer boundaries
        # Detect boundaries using segmentation entropy
        seg_entropy = -(seg_probs * (seg_probs + 1e-8).log()).sum(dim=1, keepdim=True)
        seg_entropy = seg_entropy / (math.log(5) + 1e-8)  # Normalize to [0, 1]

        # Also detect boundaries from seg_mask using gradient
        seg_mask_float = seg_mask.unsqueeze(1).float()
        seg_grad_y = torch.abs(seg_mask_float[:, :, 1:, :] - seg_mask_float[:, :, :-1, :])
        seg_grad_x = torch.abs(seg_mask_float[:, :, :, 1:] - seg_mask_float[:, :, :, :-1])
        # Pad to match size
        seg_grad_y = F.pad(seg_grad_y, (0, 0, 0, 1), mode='constant', value=0)
        seg_grad_x = F.pad(seg_grad_x, (0, 1, 0, 0), mode='constant', value=0)
        boundary_mask = (seg_grad_y + seg_grad_x > 0).float()

        # Uncertainty should be high at boundaries
        boundary_target = torch.max(seg_entropy, boundary_mask)
        boundary_loss = F.mse_loss(uncertainty * boundary_mask, boundary_target * boundary_mask)

        # 2. Pathology-aware uncertainty: HIGH where reconstruction error is high
        # (potential pathology that was incorrectly denoised)
        recon_error = torch.abs(denoised - clean)
        # Normalize reconstruction error
        recon_error_norm = recon_error / (recon_error.max() + 1e-8)

        # High reconstruction error regions should have high uncertainty
        # This flags potential pathology or denoising artifacts
        error_threshold = recon_error_norm.mean() + recon_error_norm.std()
        high_error_mask = (recon_error_norm > error_threshold).float()
        pathology_loss = F.mse_loss(
            uncertainty * high_error_mask,
            high_error_mask  # Target: uncertainty=1 where error is high
        )

        # 3. Confidence in homogeneous regions: LOW uncertainty where texture is uniform
        # Compute local variance
        local_mean = F.avg_pool2d(denoised, kernel_size=5, stride=1, padding=2)
        local_var = F.avg_pool2d((denoised - local_mean) ** 2, kernel_size=5, stride=1, padding=2)
        homogeneous_mask = (local_var < local_var.mean() * 0.5).float()

        # Uncertainty should be low in homogeneous regions
        homogeneous_loss = (uncertainty * homogeneous_mask).mean()

        total_loss = (
            self.boundary_weight * boundary_loss +
            self.pathology_weight * pathology_loss +
            0.5 * homogeneous_loss
        )

        return total_loss


class PathologyPreservationLoss(nn.Module):
    """
    CUAP-OCT: Pathology Preservation Loss

    KEY NOVEL CONTRIBUTION: Ensure denoising doesn't destroy diagnostic features.

    Pathological features (drusen, fluid, lesions) often appear as:
    - Local intensity anomalies
    - High local contrast regions
    - Irregular textures

    This loss penalizes over-smoothing in regions with high local variance,
    preserving diagnostic biomarkers that clinicians need to see.
    """

    def __init__(self, sensitivity: float = 1.0):
        super().__init__()
        self.sensitivity = sensitivity

    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pathology preservation loss.

        Args:
            denoised: Denoised output [B, 1, H, W]
            clean: Clean target [B, 1, H, W]
            noisy: Noisy input [B, 1, H, W]

        Returns:
            loss: Preservation loss for pathological features
        """
        # Detect potential pathology regions using local variance in clean image
        local_mean = F.avg_pool2d(clean, kernel_size=7, stride=1, padding=3)
        local_var = F.avg_pool2d((clean - local_mean) ** 2, kernel_size=7, stride=1, padding=3)

        # High variance regions likely contain pathology or important features
        var_threshold = local_var.mean() + self.sensitivity * local_var.std()
        pathology_mask = (local_var > var_threshold).float()

        # In pathology regions, denoised should closely match clean
        # (don't over-smooth important features)
        preservation_loss = F.l1_loss(
            denoised * pathology_mask,
            clean * pathology_mask,
            reduction='sum'
        ) / (pathology_mask.sum() + 1e-8)

        # Also preserve local contrast in pathology regions
        denoised_local_var = F.avg_pool2d(
            (denoised - F.avg_pool2d(denoised, 7, 1, 3)) ** 2, 7, 1, 3
        )
        contrast_loss = F.l1_loss(
            denoised_local_var * pathology_mask,
            local_var * pathology_mask,
            reduction='sum'
        ) / (pathology_mask.sum() + 1e-8)

        return preservation_loss + 0.5 * contrast_loss


class BoundarySharpnessLoss(nn.Module):
    """
    CUAP-OCT: Boundary Sharpness Loss

    KEY NOVEL CONTRIBUTION: Maintain sharp layer boundaries for accurate
    thickness measurement.

    Clinical importance: Layer thickness (especially RNFL) is measured in microns.
    Blurred boundaries lead to measurement error. This loss explicitly preserves
    the sharpness of layer transitions.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels for gradient computation
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary sharpness loss.

        Args:
            denoised: Denoised output [B, 1, H, W]
            clean: Clean target [B, 1, H, W]
            seg_mask: Segmentation mask [B, H, W]

        Returns:
            loss: Sharpness loss at layer boundaries
        """
        # Find layer boundaries from segmentation
        seg_mask_float = seg_mask.unsqueeze(1).float()
        boundary_grad = torch.abs(
            seg_mask_float[:, :, 1:, :] - seg_mask_float[:, :, :-1, :]
        )
        boundary_grad = F.pad(boundary_grad, (0, 0, 0, 1), mode='constant', value=0)

        # Dilate boundary mask slightly to capture transition region
        boundary_mask = F.max_pool2d(boundary_grad, kernel_size=3, stride=1, padding=1)
        boundary_mask = (boundary_mask > 0).float()

        # Compute vertical gradients (layer boundaries are mostly horizontal)
        sobel_y = self.sobel_y.to(denoised.device)

        denoised_grad = F.conv2d(F.pad(denoised, (1, 1, 1, 1), mode='reflect'), sobel_y)
        clean_grad = F.conv2d(F.pad(clean, (1, 1, 1, 1), mode='reflect'), sobel_y)

        # At boundaries, gradient magnitude should be preserved
        # This ensures sharp transitions are maintained
        grad_diff = torch.abs(torch.abs(denoised_grad) - torch.abs(clean_grad))
        sharpness_loss = (grad_diff * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)

        # Also encourage gradient direction preservation at boundaries
        direction_loss = F.l1_loss(
            denoised_grad * boundary_mask,
            clean_grad * boundary_mask,
            reduction='sum'
        ) / (boundary_mask.sum() + 1e-8)

        return sharpness_loss + 0.5 * direction_loss


class CUAPUnifiedLoss(nn.Module):
    """
    CUAP-OCT: Unified Loss Function

    Combines all CUAP components into a single coherent objective:
    - Clinical importance weighted reconstruction
    - Uncertainty calibration
    - Pathology preservation
    - Boundary sharpness

    KEY INSIGHT: These losses interact - uncertainty informs reconstruction,
    pathology detection triggers uncertainty, boundaries need special treatment.
    """

    def __init__(
        self,
        clinical_weights: Optional[Dict[str, float]] = None,
        lambda_uncertainty: float = 0.1,
        lambda_pathology: float = 0.1,
        lambda_boundary: float = 0.1,
    ):
        super().__init__()

        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_pathology = lambda_pathology
        self.lambda_boundary = lambda_boundary

        # Component losses
        self.clinical_mse = ClinicalWeightedMSELoss(clinical_weights=clinical_weights)
        self.uncertainty_loss = UncertaintyCalibrationLoss()
        self.pathology_loss = PathologyPreservationLoss()
        self.boundary_loss = BoundarySharpnessLoss()

    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        seg_mask: torch.Tensor,
        seg_probs: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified CUAP loss.

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # 1. Clinical weighted reconstruction
        losses['clinical_recon'] = self.clinical_mse(denoised, clean, seg_mask)

        # 2. Uncertainty calibration
        losses['uncertainty'] = self.uncertainty_loss(
            uncertainty, denoised, clean, seg_mask, seg_probs
        )

        # 3. Pathology preservation
        losses['pathology'] = self.pathology_loss(denoised, clean, noisy)

        # 4. Boundary sharpness
        losses['boundary'] = self.boundary_loss(denoised, clean, seg_mask)

        # Total loss
        losses['total'] = (
            losses['clinical_recon'] +
            self.lambda_uncertainty * losses['uncertainty'] +
            self.lambda_pathology * losses['pathology'] +
            self.lambda_boundary * losses['boundary']
        )

        return losses


# =============================================================================
# Anatomical Consistency Loss (KEY TMI CONTRIBUTION)
# =============================================================================

# Typical RNFL thickness ranges in microns (from clinical literature)
# These are approximate and vary by region (temporal, nasal, superior, inferior)
LAYER_THICKNESS_RANGES = {
    'RNFL_GCL': (50, 150),      # RNFL + GCL combined: 50-150 μm
    'INL_OPL': (30, 80),        # INL + OPL: 30-80 μm
    'ONL': (80, 150),           # ONL (Henle fiber layer): 80-150 μm
    'IS_OS': (40, 80),          # Photoreceptor IS/OS: 40-80 μm
    'RPE_Choroid': (100, 400),  # RPE + Choroid: highly variable
}

# Expected layer order from top (vitreous) to bottom (sclera)
LAYER_ORDER = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']


class AnatomicalConsistencyLoss(nn.Module):
    """
    CUAP-OCT: Anatomical Consistency Loss

    KEY NOVEL CONTRIBUTION: Ensures denoising maintains anatomical plausibility
    of retinal structure. This is critical because:

    1. Layer Ordering: Retinal layers have fixed anatomical order (RNFL always
       above INL, etc.). Denoising should not create artifacts that violate this.

    2. Thickness Plausibility: Each layer has clinically expected thickness ranges.
       Abnormal thickness suggests pathology OR denoising artifacts.

    3. Spatial Continuity: Layer boundaries should be smooth and continuous.
       Abrupt discontinuities indicate artifacts, not anatomy.

    Clinical importance: RNFL thickness is measured in microns for glaucoma.
    A 10μm change can indicate disease progression. Artifacts that affect
    apparent layer boundaries directly impact clinical measurements.
    """

    def __init__(
        self,
        lambda_ordering: float = 1.0,
        lambda_thickness: float = 0.5,
        lambda_continuity: float = 1.0,
        pixels_per_micron: float = 3.87,  # Typical for Spectralis OCT
    ):
        """
        Args:
            lambda_ordering: Weight for layer ordering constraint
            lambda_thickness: Weight for thickness plausibility constraint
            lambda_continuity: Weight for spatial continuity constraint
            pixels_per_micron: OCT axial resolution (device-dependent)
        """
        super().__init__()
        self.lambda_ordering = lambda_ordering
        self.lambda_thickness = lambda_thickness
        self.lambda_continuity = lambda_continuity
        self.pixels_per_micron = pixels_per_micron

        # Convert thickness ranges to pixels
        self.thickness_ranges_px = {
            layer: (min_um * pixels_per_micron, max_um * pixels_per_micron)
            for layer, (min_um, max_um) in LAYER_THICKNESS_RANGES.items()
        }

    def compute_layer_boundaries(
        self,
        seg_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute layer boundaries from segmentation probabilities.

        For each layer, find the top and bottom boundary (expected y-coordinates).
        Uses soft boundaries based on probability mass.

        Args:
            seg_probs: [B, 5, H, W] segmentation probabilities

        Returns:
            Dict with 'top' and 'bottom' boundaries for each layer [B, W]
        """
        B, C, H, W = seg_probs.shape
        device = seg_probs.device

        # Create y-coordinate grid [H]
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        y_coords = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)  # [B, 1, H, W]

        boundaries = {}

        for i, layer_name in enumerate(LAYER_ORDER):
            layer_prob = seg_probs[:, i:i+1, :, :]  # [B, 1, H, W]

            # Soft boundary: weighted average of y-coordinates by probability
            # Top boundary: minimum y where layer exists
            # Bottom boundary: maximum y where layer exists

            # Normalize probabilities per column
            col_sum = layer_prob.sum(dim=2, keepdim=True) + 1e-8  # [B, 1, 1, W]
            normalized_prob = layer_prob / col_sum  # [B, 1, H, W]

            # Center of mass (mean y-coordinate)
            center = (normalized_prob * y_coords).sum(dim=2)  # [B, 1, W]

            # Variance for boundary estimation
            variance = (normalized_prob * (y_coords - center.unsqueeze(2)) ** 2).sum(dim=2)
            std = torch.sqrt(variance + 1e-8)  # [B, 1, W]

            # Top and bottom boundaries (mean ± 2*std covers ~95% of mass)
            boundaries[f'{layer_name}_top'] = (center - 2 * std).squeeze(1)  # [B, W]
            boundaries[f'{layer_name}_bottom'] = (center + 2 * std).squeeze(1)  # [B, W]
            boundaries[f'{layer_name}_center'] = center.squeeze(1)  # [B, W]
            boundaries[f'{layer_name}_thickness'] = 4 * std.squeeze(1)  # [B, W]

        return boundaries

    def ordering_loss(
        self,
        boundaries: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Enforce that layers appear in correct anatomical order.

        Penalizes when layer N's bottom boundary is below layer N+1's top boundary.
        In a well-segmented image, RNFL_GCL should be entirely above INL_OPL, etc.
        """
        loss = torch.tensor(0.0, device=next(iter(boundaries.values())).device)

        for i in range(len(LAYER_ORDER) - 1):
            current_layer = LAYER_ORDER[i]
            next_layer = LAYER_ORDER[i + 1]

            current_bottom = boundaries[f'{current_layer}_bottom']  # [B, W]
            next_top = boundaries[f'{next_layer}_top']  # [B, W]

            # Penalize if current layer extends below next layer's top
            # (they should not overlap significantly)
            overlap = F.relu(current_bottom - next_top)  # Positive when overlapping
            loss = loss + overlap.mean()

        return loss / (len(LAYER_ORDER) - 1)

    def thickness_loss(
        self,
        boundaries: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Enforce plausible layer thicknesses.

        Penalizes thicknesses outside expected clinical ranges.
        This catches both over-smoothing (artificially thin layers) and
        artifacts (unrealistically thick layers).
        """
        loss = torch.tensor(0.0, device=next(iter(boundaries.values())).device)

        for layer_name in LAYER_ORDER:
            thickness = boundaries[f'{layer_name}_thickness']  # [B, W]
            min_px, max_px = self.thickness_ranges_px[layer_name]

            # Penalize thicknesses below minimum
            too_thin = F.relu(min_px - thickness)

            # Penalize thicknesses above maximum
            too_thick = F.relu(thickness - max_px)

            # Normalize by expected range
            range_px = max_px - min_px
            layer_loss = (too_thin + too_thick) / (range_px + 1e-8)

            loss = loss + layer_loss.mean()

        return loss / len(LAYER_ORDER)

    def continuity_loss(
        self,
        boundaries: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Enforce spatial continuity of layer boundaries.

        Layer boundaries should be smooth - abrupt jumps indicate artifacts.
        Uses total variation (gradient magnitude) as smoothness measure.
        """
        loss = torch.tensor(0.0, device=next(iter(boundaries.values())).device)

        for layer_name in LAYER_ORDER:
            # Use center boundary for continuity (more stable than top/bottom)
            center = boundaries[f'{layer_name}_center']  # [B, W]

            # Horizontal gradient (difference between adjacent columns)
            if center.shape[-1] > 1:
                gradient = torch.abs(center[:, 1:] - center[:, :-1])

                # Penalize large gradients (discontinuities)
                # Allow small gradients for natural curvature
                threshold = 2.0  # Allow up to 2 pixel difference per column
                discontinuity = F.relu(gradient - threshold)

                loss = loss + discontinuity.mean()

        return loss / len(LAYER_ORDER)

    def forward(
        self,
        seg_probs: torch.Tensor,
        seg_probs_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anatomical consistency loss.

        Args:
            seg_probs: Segmentation probabilities from denoised image [B, 5, H, W]
            seg_probs_clean: Optional segmentation probs from clean image [B, 5, H, W]
                            If provided, also penalizes deviation from clean anatomy.

        Returns:
            Anatomical consistency loss
        """
        # Compute boundaries from denoised segmentation
        boundaries = self.compute_layer_boundaries(seg_probs)

        # Individual loss components
        ordering = self.ordering_loss(boundaries)
        thickness = self.thickness_loss(boundaries)
        continuity = self.continuity_loss(boundaries)

        total_loss = (
            self.lambda_ordering * ordering +
            self.lambda_thickness * thickness +
            self.lambda_continuity * continuity
        )

        # If clean segmentation provided, add consistency term
        if seg_probs_clean is not None:
            clean_boundaries = self.compute_layer_boundaries(seg_probs_clean)

            # Penalize deviation of denoised boundaries from clean boundaries
            boundary_deviation = torch.tensor(0.0, device=seg_probs.device)
            for layer_name in LAYER_ORDER:
                center_diff = torch.abs(
                    boundaries[f'{layer_name}_center'] -
                    clean_boundaries[f'{layer_name}_center']
                )
                boundary_deviation = boundary_deviation + center_diff.mean()

            boundary_deviation = boundary_deviation / len(LAYER_ORDER)
            total_loss = total_loss + boundary_deviation

        return total_loss


# =============================================================================
# Confidence-Weighted Refinement Loss (KEY TMI CONTRIBUTION)
# =============================================================================

class ConfidenceWeightedRefinementLoss(nn.Module):
    """
    CUAP-OCT: Confidence-Weighted Refinement Loss

    KEY NOVEL CONTRIBUTION: Explicitly train the model to denoise conservatively
    in regions where segmentation is uncertain.

    Clinical rationale:
    - Uncertain segmentation often indicates: layer boundaries, pathology, or artifacts
    - These regions contain diagnostic information that should NOT be smoothed away
    - Over-denoising pathology (drusen, fluid, lesions) can mask disease

    This loss penalizes:
    1. Large refinement in low-confidence regions (preserve pathology)
    2. Small refinement in high-confidence regions (ensure effective denoising)

    The combination ensures:
    - Aggressive denoising where we're certain about anatomy
    - Conservative denoising where diagnostic features might exist
    """

    def __init__(
        self,
        low_conf_threshold: float = 0.5,
        high_conf_threshold: float = 0.8,
        penalty_weight: float = 1.0,
    ):
        """
        Args:
            low_conf_threshold: Below this confidence, penalize large refinement
            high_conf_threshold: Above this confidence, expect stronger refinement
            penalty_weight: Overall weight for the loss
        """
        super().__init__()
        self.low_conf_threshold = low_conf_threshold
        self.high_conf_threshold = high_conf_threshold
        self.penalty_weight = penalty_weight

    def forward(
        self,
        refinement: torch.Tensor,
        confidence_map: torch.Tensor,
        denoised: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence-weighted refinement loss.

        Args:
            refinement: Refinement applied [B, 1, H, W]
            confidence_map: Pixel-wise segmentation confidence [B, H, W]
            denoised: Denoised output [B, 1, H, W]
            clean: Clean target [B, 1, H, W]

        Returns:
            Loss encouraging confidence-appropriate refinement strength
        """
        B, _, H, W = refinement.shape
        device = refinement.device

        # Expand confidence_map to match refinement shape
        if confidence_map.dim() == 3:
            confidence_map = confidence_map.unsqueeze(1)  # [B, 1, H, W]

        # Absolute refinement magnitude
        refinement_mag = torch.abs(refinement)

        # 1. LOW CONFIDENCE PENALTY: Penalize large refinement in uncertain regions
        # These regions may contain pathology - don't smooth them away
        low_conf_mask = (confidence_map < self.low_conf_threshold).float()

        # Reconstruction error in low-confidence regions
        # If error is high AND refinement was large, model may have damaged pathology
        recon_error = torch.abs(denoised - clean)

        # Penalize refinement proportional to (1 - confidence) and error
        low_conf_penalty = (refinement_mag * (1 - confidence_map) * low_conf_mask).mean()

        # 2. HIGH CONFIDENCE ENCOURAGEMENT: Ensure we denoise effectively where certain
        # High confidence = certain about layer structure = safe to denoise
        high_conf_mask = (confidence_map > self.high_conf_threshold).float()

        # In high-confidence regions with noise, we SHOULD see refinement
        # Penalize if high-confidence noisy regions have too little refinement
        # Use reconstruction error as proxy for "how much denoising was needed"
        expected_refinement = recon_error * high_conf_mask
        actual_refinement = refinement_mag * high_conf_mask

        # Penalize if actual refinement is much less than expected (under-denoising)
        under_denoising_penalty = F.relu(expected_refinement - actual_refinement).mean()

        # 3. BOUNDARY AWARENESS: Extra penalty for boundaries (transition regions)
        # Boundaries have intermediate confidence and are clinically critical
        boundary_mask = ((confidence_map >= self.low_conf_threshold) &
                        (confidence_map <= self.high_conf_threshold)).float()

        # At boundaries, refinement should be moderate (not too aggressive)
        boundary_refinement = refinement_mag * boundary_mask
        # Penalize if boundary refinement is too high (use mean as soft target)
        mean_refinement = refinement_mag.mean()
        boundary_penalty = F.relu(boundary_refinement - mean_refinement * 1.5).mean()

        total_loss = self.penalty_weight * (
            low_conf_penalty +
            0.5 * under_denoising_penalty +
            0.5 * boundary_penalty
        )

        return total_loss


class ConfidenceConsistencyLoss(nn.Module):
    """
    CUAP-OCT: Confidence Consistency Loss

    Ensures that the confidence gate behaves consistently:
    1. High uncertainty (from uncertainty head) should correlate with low confidence
    2. Boundary regions (from segmentation) should have intermediate confidence
    3. Homogeneous regions should have high confidence

    This creates a coherent uncertainty-aware system where all components agree.
    """

    def __init__(self, correlation_weight: float = 1.0):
        super().__init__()
        self.correlation_weight = correlation_weight

    def forward(
        self,
        confidence_map: torch.Tensor,
        uncertainty: torch.Tensor,
        seg_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            confidence_map: Segmentation confidence [B, H, W] or [B, 1, H, W]
            uncertainty: Uncertainty map from uncertainty head [B, 1, H, W]
            seg_probs: Segmentation probabilities [B, 5, H, W]

        Returns:
            Consistency loss
        """
        # Ensure shapes match
        if confidence_map.dim() == 3:
            confidence_map = confidence_map.unsqueeze(1)

        # 1. Confidence-Uncertainty anti-correlation
        # High confidence should mean low uncertainty and vice versa
        # Target: confidence ≈ 1 - uncertainty
        target_confidence = 1 - uncertainty
        conf_unc_loss = F.mse_loss(confidence_map, target_confidence)

        # 2. Boundary detection from segmentation entropy
        seg_entropy = -(seg_probs * (seg_probs + 1e-8).log()).sum(dim=1, keepdim=True)
        max_entropy = math.log(5)
        seg_entropy_norm = seg_entropy / (max_entropy + 1e-8)

        # High entropy (boundary) should have lower confidence
        # This is already captured by our confidence computation, but we reinforce it
        boundary_conf_loss = (confidence_map * seg_entropy_norm).mean()

        total_loss = self.correlation_weight * (
            conf_unc_loss + 0.5 * boundary_conf_loss
        )

        return total_loss
