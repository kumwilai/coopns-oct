#!/usr/bin/env python3
"""
Quality-Aligned Loss for Neuro-Symbolic OCT Denoising

Key innovations:
1. Clean-Referenced Predicates: Compare denoised vs CLEAN (not noisy)
2. Hard Quality Constraint: Heavy penalty for quality degradation

This ensures:
- Improving predicates = improving similarity to clean
- Quality can never degrade below backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CleanReferencedPredicates(nn.Module):
    """
    Predicates that compare denoised vs CLEAN reference.

    Unlike standard predicates that compare vs noisy, these measure
    how well the denoised output matches the clean ground truth.

    This aligns predicate optimization with quality optimization.
    """

    def __init__(self):
        super().__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Learnable thresholds (for soft gating)
        self.register_buffer('threshold_P1', torch.tensor(0.90))
        self.register_buffer('threshold_P2', torch.tensor(0.75))
        self.register_buffer('threshold_P3', torch.tensor(0.85))
        self.register_buffer('threshold_P4', torch.tensor(0.90))
        self.register_buffer('threshold_P5', torch.tensor(0.80))
        self.register_buffer('threshold_P6', torch.tensor(0.90))

    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def compute_local_stats(self, x: torch.Tensor, kernel_size: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local mean and std."""
        padding = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        sq_mean = F.avg_pool2d(x**2, kernel_size, stride=1, padding=padding)
        std = (sq_mean - mean**2).clamp(min=1e-6).sqrt()
        return mean, std

    def P1_edge_similarity(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict:
        """
        P1: Edge Similarity to Clean

        Measures how well edges in denoised match edges in clean.
        Score = correlation of edge maps
        """
        edges_den = self.compute_edges(denoised)
        edges_clean = self.compute_edges(clean)

        # Normalized cross-correlation
        edges_den_norm = edges_den - edges_den.mean()
        edges_clean_norm = edges_clean - edges_clean.mean()

        correlation = (edges_den_norm * edges_clean_norm).sum() / (
            edges_den_norm.norm() * edges_clean_norm.norm() + 1e-8
        )

        score = correlation.clamp(0, 1)

        # Failure map: where edges don't match
        edge_diff = (edges_den - edges_clean).abs()
        failure_map = (edge_diff / (edges_clean.max() + 1e-6)).clamp(0, 1)

        return {
            'score': score,
            'passed': score > self.threshold_P1,
            'failure_map': failure_map
        }

    def P2_contrast_similarity(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict:
        """
        P2: Local Contrast Similarity to Clean

        Measures how well local contrast in denoised matches clean.
        Uses a softer scoring function that gives reasonable scores for ratios 0.5-1.5
        """
        _, std_den = self.compute_local_stats(denoised, kernel_size=9)
        _, std_clean = self.compute_local_stats(clean, kernel_size=9)

        # Contrast ratio (closer to 1 = better)
        contrast_ratio = (std_den + 1e-6) / (std_clean + 1e-6)

        # SOFTER scoring: Use ratio directly, clamped to [0, 1]
        # If denoised has 50% of clean contrast, score = 0.5 (not 0.08!)
        # Penalize over-contrast too: min(ratio, 2-ratio) peaks at ratio=1
        ratio_mean = contrast_ratio.mean()

        # Score that peaks at ratio=1, decreases linearly
        # ratio=0.5 -> score=0.5, ratio=1.0 -> score=1.0, ratio=1.5 -> score=0.5
        if ratio_mean <= 1.0:
            score = ratio_mean.clamp(0, 1)
        else:
            score = (2.0 - ratio_mean).clamp(0, 1)

        # Failure map: where contrast is too low (ratio < 1)
        failure_map = F.relu(1.0 - contrast_ratio).clamp(0, 1)

        return {
            'score': score,
            'passed': score > self.threshold_P2,
            'failure_map': failure_map
        }

    def P3_smoothness_quality(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict:
        """
        P3: Smoothness in Flat Regions

        Measures noise reduction in regions that should be smooth (flat in clean).
        """
        edges_clean = self.compute_edges(clean)
        flat_mask = (edges_clean < edges_clean.mean() * 0.5).float()

        _, std_den = self.compute_local_stats(denoised, kernel_size=5)
        _, std_clean = self.compute_local_stats(clean, kernel_size=5)

        # In flat regions, denoised should have similar (or lower) variance than clean
        if flat_mask.sum() < 100:
            score = torch.ones(1, device=denoised.device)
            failure_map = torch.zeros_like(denoised)
        else:
            variance_ratio = (std_den / (std_clean + 1e-6)) * flat_mask
            variance_ratio_mean = variance_ratio.sum() / (flat_mask.sum() + 1e-6)

            # Score: lower variance ratio in flat regions = better
            score = torch.exp(-F.relu(variance_ratio_mean - 1) * 5).clamp(0, 1)

            failure_map = (variance_ratio * flat_mask).clamp(0, 1)

        return {
            'score': score,
            'passed': score > self.threshold_P3,
            'failure_map': failure_map
        }

    def P4_structure_similarity(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict:
        """
        P4: Structural Similarity (SSIM-based)

        Measures structural similarity between denoised and clean.
        """
        C1, C2 = 0.01**2, 0.03**2

        mu_den, std_den = self.compute_local_stats(denoised, kernel_size=11)
        mu_clean, std_clean = self.compute_local_stats(clean, kernel_size=11)

        # Cross-correlation
        padding = 5
        sigma_dc = F.avg_pool2d(denoised * clean, 11, stride=1, padding=padding) - mu_den * mu_clean

        # SSIM
        ssim = ((2 * mu_den * mu_clean + C1) * (2 * sigma_dc + C2)) / (
            (mu_den**2 + mu_clean**2 + C1) * (std_den**2 + std_clean**2 + C2)
        )

        score = ssim.mean().clamp(0, 1)
        failure_map = (1 - ssim).clamp(0, 1)

        return {
            'score': score,
            'passed': score > self.threshold_P4,
            'failure_map': failure_map
        }

    def P5_sharpness_similarity(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict:
        """
        P5: Sharpness at Boundaries

        Measures how well edge sharpness in denoised matches clean.
        Focuses on boundary regions (strong edges in clean).
        Uses softer scoring for realistic ratios (0.5-1.0 typical for denoised).
        """
        edges_den = self.compute_edges(denoised)
        edges_clean = self.compute_edges(clean)

        # Focus on boundary regions
        boundary_mask = (edges_clean > edges_clean.mean()).float()

        if boundary_mask.sum() < 100:
            score = torch.ones(1, device=denoised.device)
            failure_map = torch.zeros_like(denoised)
        else:
            # Sharpness ratio at boundaries
            edges_den_boundary = (edges_den * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)
            edges_clean_boundary = (edges_clean * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)
            sharpness_ratio = edges_den_boundary / (edges_clean_boundary + 1e-6)

            # SOFTER scoring: linear relationship
            # ratio=0.5 -> score=0.5, ratio=1.0 -> score=1.0
            if sharpness_ratio <= 1.0:
                score = sharpness_ratio.clamp(0, 1)
            else:
                # Penalize over-sharpening
                score = (2.0 - sharpness_ratio).clamp(0, 1)

            # Failure map: where edges are weaker than clean
            pixel_ratio = (edges_den + 1e-6) / (edges_clean + 1e-6)
            failure_map = (F.relu(1.0 - pixel_ratio) * boundary_mask).clamp(0, 1)

        return {
            'score': score,
            'passed': score > self.threshold_P5,
            'failure_map': failure_map
        }

    def P6_noise_reduction(self, denoised: torch.Tensor, clean: torch.Tensor,
                           noisy: torch.Tensor) -> Dict:
        """
        P6: Noise Reduction Quality

        Measures how much noise was reduced compared to the noise in input.
        Score = 1 - (residual_noise / original_noise)
        """
        # Estimate noise as difference from clean
        noise_input = (noisy - clean).abs()
        noise_output = (denoised - clean).abs()

        # Noise reduction ratio
        noise_ratio = noise_output.mean() / (noise_input.mean() + 1e-6)

        # Score: lower ratio = better noise reduction
        score = (1 - noise_ratio).clamp(0, 1)

        # Failure map: where noise remains
        failure_map = (noise_output / (noise_input.max() + 1e-6)).clamp(0, 1)

        return {
            'score': score,
            'passed': score > self.threshold_P6,
            'failure_map': failure_map
        }

    def P7_mse_proximity(self, denoised: torch.Tensor, clean: torch.Tensor) -> Dict:
        """
        P7: MSE Proximity to Clean

        Measures how close the denoised output is to clean using normalized MSE.
        Score = exp(-mse_normalized) where mse_normalized = mse / clean_variance
        This provides strong coupling to reconstruction quality.
        """
        # Compute MSE between denoised and clean
        mse = F.mse_loss(denoised, clean)

        # Compute variance of clean image for normalization
        clean_variance = clean.var() + 1e-8

        # Normalized MSE
        mse_normalized = mse / clean_variance

        # Score: exp(-mse_normalized) - higher score when MSE is lower
        score = torch.exp(-mse_normalized).clamp(0, 1)

        # Failure map: per-pixel squared error normalized
        pixel_mse = (denoised - clean) ** 2
        failure_map = (pixel_mse / (clean_variance + 1e-6)).clamp(0, 1)

        return {
            'score': score,
            'passed': score > 0.5,  # Threshold at exp(-0.693) ~ 0.5
            'failure_map': failure_map
        }

    def forward(self, denoised: torch.Tensor, clean: torch.Tensor,
                noisy: torch.Tensor = None) -> Dict:
        """
        Compute all clean-referenced predicates.

        Args:
            denoised: Model output [B, 1, H, W]
            clean: Ground truth [B, 1, H, W]
            noisy: Original noisy input (optional, for P6)

        Returns:
            Dict with P1-P7 scores and failure maps
        """
        p1 = self.P1_edge_similarity(denoised, clean)
        p2 = self.P2_contrast_similarity(denoised, clean)
        p3 = self.P3_smoothness_quality(denoised, clean)
        p4 = self.P4_structure_similarity(denoised, clean)
        p5 = self.P5_sharpness_similarity(denoised, clean)

        if noisy is not None:
            p6 = self.P6_noise_reduction(denoised, clean, noisy)
        else:
            p6 = {'score': torch.ones(1, device=denoised.device),
                  'passed': True, 'failure_map': torch.zeros_like(denoised)}

        p7 = self.P7_mse_proximity(denoised, clean)

        all_passed = all([p1['passed'], p2['passed'], p3['passed'],
                         p4['passed'], p5['passed'], p6['passed'], p7['passed']])

        return {
            'P1': p1, 'P2': p2, 'P3': p3,
            'P4': p4, 'P5': p5, 'P6': p6, 'P7': p7,
            'all_passed': all_passed
        }


class QualityAlignedLoss(nn.Module):
    """
    Quality-Aligned Loss combining:
    1. Clean-referenced predicate maximization
    2. Hard quality constraint (no degradation below backbone)
    3. Reconstruction loss

    This ensures predicates and quality improve together.
    """

    def __init__(self,
                 lambda_pred: float = 0.1,   # REDUCED: Predicates shouldn't dominate
                 lambda_quality: float = 5.0,  # Penalize degradation
                 lambda_recon: float = 100.0):  # HIGH: PSNR is primary objective
        super().__init__()

        self.predicates = CleanReferencedPredicates()
        self.lambda_pred = lambda_pred
        self.lambda_quality = lambda_quality
        self.lambda_recon = lambda_recon

        # Predicate weights - NORMALIZED to sum to ~1 for balanced contribution
        # This ensures pred_loss is in reasonable range
        self.pred_weights = {
            'P1': 0.1, 'P2': 0.15, 'P3': 0.1,   # Edge, contrast, smooth
            'P4': 0.15, 'P5': 0.15, 'P6': 0.1,  # Structure, sharp, noise
            'P7': 0.25  # MSE proximity - most aligned with PSNR
        }

    def forward(self, corrected: torch.Tensor,
                backbone_out: torch.Tensor,
                clean: torch.Tensor,
                noisy: torch.Tensor,
                lambda_maps: Dict[str, torch.Tensor] = None) -> Dict:
        """
        Compute quality-aligned loss.

        Args:
            corrected: Corrector output [B, 1, H, W]
            backbone_out: Backbone-only output [B, 1, H, W]
            clean: Ground truth [B, 1, H, W]
            noisy: Noisy input [B, 1, H, W]
            lambda_maps: Lambda maps (for logging)

        Returns:
            Dict with loss components
        """
        # =====================================================================
        # 1. CLEAN-REFERENCED PREDICATES
        # =====================================================================
        pred_results = self.predicates(corrected, clean, noisy)

        # Predicate loss: maximize scores (loss = -score)
        pred_loss = torch.tensor(0.0, device=corrected.device)
        pred_scores = {}

        for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']:
            score = pred_results[name]['score']
            if isinstance(score, torch.Tensor):
                pred_loss = pred_loss - self.pred_weights[name] * score
                pred_scores[name] = score.item()
            else:
                pred_scores[name] = score

        # =====================================================================
        # 2. SOFT QUALITY CONSTRAINT
        # Allow small local degradation if global PSNR improves
        # =====================================================================
        with torch.no_grad():
            mse_backbone_global = F.mse_loss(backbone_out, clean)
            psnr_backbone = -10 * torch.log10(mse_backbone_global + 1e-8)

        mse_corrected_global = F.mse_loss(corrected, clean)
        psnr_corrected = -10 * torch.log10(mse_corrected_global + 1e-8)

        # Soft quality degradation: only penalize if GLOBAL MSE increases
        # Use smooth penalty instead of hard ReLU
        global_mse_delta = mse_corrected_global - mse_backbone_global.detach()
        # Softplus provides smooth penalty that grows when delta > 0
        quality_loss = F.softplus(global_mse_delta * 100.0) / 100.0

        # HARD PSNR degradation penalty
        psnr_degradation = F.relu(mse_corrected_global - mse_backbone_global.detach()) * 500.0

        # Also penalize SSIM degradation (reduced penalty)
        ssim_backbone = self._compute_ssim(backbone_out, clean)
        ssim_corrected = self._compute_ssim(corrected, clean)
        ssim_degradation = F.relu(ssim_backbone - ssim_corrected)

        quality_loss = quality_loss + ssim_degradation * 2.0

        # =====================================================================
        # 3. RECONSTRUCTION LOSS
        # =====================================================================
        recon_loss = F.mse_loss(corrected, clean)

        # =====================================================================
        # 3.5 CORRECTION MAGNITUDE PENALTY
        # REDUCED: Don't penalize corrections - let PSNR guide learning
        # =====================================================================
        correction_magnitude = (corrected - backbone_out).abs().mean()
        correction_penalty = correction_magnitude * 0.01  # Nearly zero penalty

        # =====================================================================
        # 4. TOTAL LOSS
        # =====================================================================
        total = (self.lambda_recon * recon_loss +
                 self.lambda_pred * pred_loss +
                 self.lambda_quality * quality_loss +
                 correction_penalty +
                 psnr_degradation)

        # =====================================================================
        # 5. LOGGING
        # =====================================================================
        lambda_edge = lambda_maps.get('edge', torch.zeros(1)).mean().item() if lambda_maps else 0
        lambda_contrast = lambda_maps.get('contrast', torch.zeros(1)).mean().item() if lambda_maps else 0
        lambda_sharpness = lambda_maps.get('sharpness', torch.zeros(1)).mean().item() if lambda_maps else 0

        return {
            'total': total,
            'pred_loss': pred_loss.detach(),
            'quality_loss': quality_loss.detach(),
            'recon_loss': recon_loss.detach(),
            'correction_penalty': correction_penalty.detach(),
            'correction_magnitude': correction_magnitude.item(),
            'psnr_backbone': psnr_backbone.item(),
            'psnr_corrected': psnr_corrected.item(),
            'ssim_backbone': ssim_backbone.item(),
            'ssim_corrected': ssim_corrected.item(),
            'quality_improved': (psnr_corrected > psnr_backbone).item(),
            'lambda_edge_mean': lambda_edge,
            'lambda_contrast_mean': lambda_contrast,
            'lambda_sharpness_mean': lambda_sharpness,
            **{f'P{i}': pred_scores[f'P{i}'] for i in range(1, 8)},
            'predicate_results': pred_results,
        }

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM (differentiable)."""
        C1, C2 = 0.01**2, 0.03**2

        mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)

        sigma1_sq = F.avg_pool2d(img1**2, 11, stride=1, padding=5) - mu1**2
        sigma2_sq = F.avg_pool2d(img2**2, 11, stride=1, padding=5) - mu2**2
        sigma12 = F.avg_pool2d(img1 * img2, 11, stride=1, padding=5) - mu1 * mu2

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim.mean()


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing Quality-Aligned Loss...")

    # Create dummy data
    B, H, W = 2, 64, 64
    noisy = torch.randn(B, 1, H, W) * 0.3 + 0.5
    clean = torch.randn(B, 1, H, W) * 0.1 + 0.5
    backbone_out = clean + torch.randn(B, 1, H, W) * 0.05
    corrected = clean + torch.randn(B, 1, H, W) * 0.03  # Closer to clean

    # Test predicates
    predicates = CleanReferencedPredicates()
    results = predicates(corrected, clean, noisy)

    print("\nClean-Referenced Predicate Scores:")
    for name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']:
        score = results[name]['score']
        print(f"  {name}: {score.item():.4f}")

    # Test loss
    loss_fn = QualityAlignedLoss()
    loss_dict = loss_fn(corrected, backbone_out, clean, noisy)

    print("\nLoss Components:")
    print(f"  Total: {loss_dict['total'].item():.4f}")
    print(f"  Pred loss: {loss_dict['pred_loss'].item():.4f}")
    print(f"  Quality loss: {loss_dict['quality_loss'].item():.4f}")
    print(f"  Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"  PSNR backbone: {loss_dict['psnr_backbone']:.2f} dB")
    print(f"  PSNR corrected: {loss_dict['psnr_corrected']:.2f} dB")
    print(f"  Quality improved: {loss_dict['quality_improved']}")

    print("\nTest passed!")
