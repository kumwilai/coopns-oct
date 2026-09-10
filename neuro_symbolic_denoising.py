#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising with Clinical Predicate Verification

True neuro-symbolic integration:
- Neural backbone (NAFNet) trained with symbolic predicate constraints
- Symbolic predicates based on clinical quality requirements
- Separate neural correctors guided by symbolic failure maps
- Conflict-free correction with priority-based resolution

Optimized for CPU training with memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import time
import sys
import gc
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, 'nsnd_oct')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_memory(force: bool = False):
    """Clear GPU/CPU memory. Only runs gc.collect() occasionally for performance."""
    # FIX: gc.collect() is slow (~77ms), only call when forced or periodically
    if force:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Counter for periodic memory cleanup
_memory_cleanup_counter = 0


def maybe_clear_memory():
    """Periodically clear memory (every 10 calls) for performance."""
    global _memory_cleanup_counter
    _memory_cleanup_counter += 1
    if _memory_cleanup_counter >= 10:
        _memory_cleanup_counter = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def to_float(x) -> float:
    """Convert tensor or number to float. Used for metrics logging."""
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR in dB."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM (simplified for speed)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu1 = F.avg_pool2d(pred, 3, stride=1, padding=1)
    mu2 = F.avg_pool2d(target, 3, stride=1, padding=1)

    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred ** 2, 3, stride=1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(target ** 2, 3, stride=1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, 3, stride=1, padding=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# =============================================================================
# DIFFERENTIABLE SYMBOLIC PREDICATES
# =============================================================================

class SobelFilter(nn.Module):
    """Fixed Sobel filter for edge detection."""

    def __init__(self):
        super().__init__()
        # Sobel Y (vertical edges - detects horizontal layer boundaries)
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
        self.register_buffer('sobel_y', sobel_y)

        # Sobel X
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns gradient magnitude."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def vertical(self, x: torch.Tensor) -> torch.Tensor:
        """Returns vertical gradient (for horizontal boundaries)."""
        return torch.abs(F.conv2d(x, self.sobel_y, padding=1))


class GaussianSmooth(nn.Module):
    """Fixed Gaussian smoothing."""

    def __init__(self, sigma: float = 1.0, kernel_size: int = 5):
        super().__init__()

        # Create Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        self.register_buffer('kernel', kernel_2d)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=self.padding)


class BoundaryPredicate(nn.Module):
    """
    P1: Boundary Detectability

    Clinical basis: Layer boundaries must be detectable for segmentation.
    19-46% of OCT scans have segmentation errors due to weak boundaries.
    """

    def __init__(self):
        super().__init__()
        self.sobel = SobelFilter()
        self.smooth = GaussianSmooth(sigma=1.0)
        self.threshold = 0.85  # High bar for boundary quality

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict:
        # Reference edges from smoothed noisy
        noisy_smooth = self.smooth(noisy)
        edges_ref = self.sobel.vertical(noisy_smooth)

        # Edges in denoised
        edges_den = self.sobel.vertical(denoised)

        # Normalize both
        edges_ref_norm = edges_ref / (edges_ref.max() + 1e-6)
        edges_den_norm = edges_den / (edges_den.max() + 1e-6)

        # Find boundary locations (top 20% edges)
        boundary_mask = (edges_ref_norm > 0.3).float()

        # Preservation ratio at boundaries (clamped)
        preservation = (edges_den_norm / (edges_ref_norm + 0.1)).clamp(0, 2)

        # Score: mean preservation at boundaries (want close to 1)
        mask_sum = boundary_mask.sum() + 1e-6
        mean_preservation = (preservation * boundary_mask).sum() / mask_sum

        # Score: 1 when preservation is ~1, lower otherwise
        score = (1 - torch.abs(mean_preservation - 1) * 0.5).clamp(0, 1)

        # Failure map: where edges are weaker than reference
        weakness = (1 - edges_den_norm / (edges_ref_norm + 0.1)).clamp(0, 1)
        failure_map = boundary_mask * weakness

        return {
            'score': score,
            'passed': (score > self.threshold).item(),  # FIX: Convert to bool
            'failure_map': failure_map,
            'name': 'P1_Boundary'
        }


class ContrastPredicate(nn.Module):
    """
    P2: Layer Contrast

    Clinical basis: Adjacent layers must have distinguishable intensities.
    Low contrast = layers indistinguishable = unusable for diagnosis.
    """

    def __init__(self):
        super().__init__()
        self.sobel = SobelFilter()
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=3)
        self.threshold = 0.45  # Moderate bar for layer contrast

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict:
        # Find boundary locations
        edges = self.sobel.vertical(denoised)
        edges_norm = edges / (edges.max() + 1e-6)
        boundary_mask = (edges_norm > 0.2).float()

        # Local contrast at boundaries
        local_mean = self.avg_pool(denoised)
        local_contrast = torch.abs(denoised - local_mean)
        local_contrast_norm = local_contrast / (local_contrast.max() + 1e-6)

        # Contrast at boundaries
        mask_sum = boundary_mask.sum() + 1e-6
        boundary_contrast = (local_contrast_norm * boundary_mask).sum() / mask_sum

        # Vertical variation (layers are horizontal)
        vertical_profile = denoised.mean(dim=-1)  # [B, 1, H]
        vertical_std = vertical_profile.std(dim=-1).mean()

        # Combined score (normalized)
        contrast_score = boundary_contrast.clamp(0, 1)
        vertical_score = (vertical_std / 0.1).clamp(0, 1)
        score = 0.5 * contrast_score + 0.5 * vertical_score

        # Failure map: low contrast at boundaries
        failure_map = boundary_mask * (1 - local_contrast_norm)

        return {
            'score': score,
            'passed': (score > self.threshold).item(),  # FIX: Convert to bool
            'failure_map': failure_map.clamp(0, 1),
            'name': 'P2_Contrast'
        }


class NoisePredicate(nn.Module):
    """
    P3: Noise Reduction

    Clinical basis: Speckle noise complicates boundary identification.
    Flat regions should be smooth after denoising.
    """

    def __init__(self):
        super().__init__()
        self.sobel = SobelFilter()
        self.pool_3 = nn.AvgPool2d(3, stride=1, padding=1)
        # FIX: Removed unused pool_7 to save memory
        self.threshold = 0.6  # Adjusted threshold

    def local_variance(self, x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
        mean = pool(x)
        mean_sq = pool(x ** 2)
        var = (mean_sq - mean ** 2).clamp(min=0)
        return var

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict:
        # Identify flat regions (low edge content)
        edges = self.sobel(denoised)
        edges_norm = edges / (edges.max() + 1e-6)
        flat_mask = (edges_norm < 0.3).float()

        # Variance in denoised (should be low in flat regions)
        var_denoised = self.local_variance(denoised, self.pool_3)
        var_denoised_norm = var_denoised / (var_denoised.max() + 1e-6)

        # Compare with noisy variance (should be reduced)
        var_noisy = self.local_variance(noisy, self.pool_3)
        var_noisy_norm = var_noisy / (var_noisy.max() + 1e-6)

        # Noise reduction ratio in flat regions
        mask_sum = flat_mask.sum() + 1e-6
        denoised_var_flat = (var_denoised_norm * flat_mask).sum() / mask_sum
        noisy_var_flat = (var_noisy_norm * flat_mask).sum() / mask_sum

        # Score: how much variance was reduced (higher = better)
        reduction_ratio = 1 - (denoised_var_flat / (noisy_var_flat + 1e-6))
        score = reduction_ratio.clamp(0, 1)

        # Failure map: where variance is still high in flat regions
        failure_map = flat_mask * var_denoised_norm

        return {
            'score': score,
            'passed': (score > self.threshold).item(),  # FIX: Convert to bool
            'failure_map': failure_map.clamp(0, 1),
            'name': 'P3_Noise'
        }


class StructurePredicate(nn.Module):
    """
    P4: Structure Preservation

    Clinical basis: Denoising should not over-smooth and blur boundaries.
    Residual should contain noise, not structure.
    """

    def __init__(self):
        super().__init__()
        self.sobel = SobelFilter()
        self.smooth = GaussianSmooth(sigma=1.0)
        self.threshold = 0.6  # High bar for structure preservation

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict:
        residual = noisy - denoised

        # Edge content in residual (normalized)
        residual_edges = self.sobel(residual)
        residual_edge_norm = residual_edges / (residual_edges.max() + 1e-6)

        # Reference edges (normalized)
        noisy_smooth = self.smooth(noisy)
        ref_edges = self.sobel(noisy_smooth)
        ref_edge_norm = ref_edges / (ref_edges.max() + 1e-6)

        # True boundary locations
        boundary_mask = (ref_edge_norm > 0.3).float()

        # Structure in residual at boundaries (should be low)
        # High value = we removed structure = BAD
        mask_sum = boundary_mask.sum() + 1e-6
        structure_in_residual = (residual_edge_norm * boundary_mask).sum() / mask_sum

        # Score: want low structure in residual
        # Normalize to 0-1 range (structure_in_residual typically 0-0.5)
        score = (1 - 2 * structure_in_residual).clamp(0, 1)

        # Failure map: where residual has edge content at boundaries
        failure_map = boundary_mask * residual_edge_norm

        return {
            'score': score,
            'passed': (score > self.threshold).item(),  # FIX: Convert to bool
            'failure_map': failure_map.clamp(0, 1),
            'name': 'P4_Structure'
        }


class SymbolicPredicates(nn.Module):
    """
    Combined symbolic predicates for OCT quality verification.
    All predicates are differentiable for gradient-based training.
    Memory-optimized: only keep failure maps when needed.
    """

    def __init__(self):
        super().__init__()
        self.P1 = BoundaryPredicate()
        self.P2 = ContrastPredicate()
        self.P3 = NoisePredicate()
        self.P4 = StructurePredicate()

        # Clinical importance weights
        self.register_buffer('weights', torch.tensor([1.0, 0.8, 0.6, 0.9]))

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor,
                return_failure_maps: bool = True) -> Dict:
        """
        Args:
            noisy: Input noisy image
            denoised: Denoised output
            return_failure_maps: If False, don't keep failure maps (saves memory)
        """
        # Compute predicates one by one to reduce peak memory
        p1_result = self.P1(noisy, denoised)
        p2_result = self.P2(noisy, denoised)
        p3_result = self.P3(noisy, denoised)
        p4_result = self.P4(noisy, denoised)

        # Combined score (keep gradients for training)
        scores = torch.stack([
            p1_result['score'],
            p2_result['score'],
            p3_result['score'],
            p4_result['score'],
        ])
        combined_score = (scores * self.weights).sum() / self.weights.sum()

        # All pass check
        all_passed = (p1_result['passed'] and p2_result['passed'] and
                      p3_result['passed'] and p4_result['passed'])

        if return_failure_maps:
            # Detach failure maps to free gradient graph memory
            results = {
                'P1': {'score': p1_result['score'], 'passed': p1_result['passed'],
                       'failure_map': p1_result['failure_map'].detach(), 'name': 'P1_Boundary'},
                'P2': {'score': p2_result['score'], 'passed': p2_result['passed'],
                       'failure_map': p2_result['failure_map'].detach(), 'name': 'P2_Contrast'},
                'P3': {'score': p3_result['score'], 'passed': p3_result['passed'],
                       'failure_map': p3_result['failure_map'].detach(), 'name': 'P3_Noise'},
                'P4': {'score': p4_result['score'], 'passed': p4_result['passed'],
                       'failure_map': p4_result['failure_map'].detach(), 'name': 'P4_Structure'},
            }
        else:
            # Scores only mode - minimal memory
            results = {
                'P1': {'score': p1_result['score'], 'passed': p1_result['passed'], 'name': 'P1_Boundary'},
                'P2': {'score': p2_result['score'], 'passed': p2_result['passed'], 'name': 'P2_Contrast'},
                'P3': {'score': p3_result['score'], 'passed': p3_result['passed'], 'name': 'P3_Noise'},
                'P4': {'score': p4_result['score'], 'passed': p4_result['passed'], 'name': 'P4_Structure'},
            }
            # Explicitly delete failure maps
            del p1_result, p2_result, p3_result, p4_result

        return {
            'individual': results,
            'combined_score': combined_score,
            'all_passed': all_passed,
        }


# =============================================================================
# CONFLICT-FREE NEURAL CORRECTORS
# =============================================================================

class ConflictResolver(nn.Module):
    """
    Resolves conflicts when multiple predicates fail at same location.
    Uses clinical priority for winner-take-all at each pixel.
    Memory-optimized: processes without large stacked tensors.
    """

    def __init__(self):
        super().__init__()
        # Priority: P1 (boundary) > P4 (structure) > P2 (contrast) > P3 (noise)
        self.priorities = {'P1': 1.0, 'P2': 0.7, 'P3': 0.5, 'P4': 0.9}

    def forward(self, failure_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Memory-efficient conflict resolution without stacking."""
        # Get reference shape from any failure map
        ref_map = failure_maps['P1']

        # Initialize max_priority by computing on-the-fly (no intermediate storage)
        max_priority = torch.zeros_like(ref_map)
        for name in ['P1', 'P2', 'P3', 'P4']:
            weighted = failure_maps[name] * self.priorities[name]
            max_priority = torch.maximum(max_priority, weighted)
            # Don't store weighted - recompute when needed

        # Only activate where there's actual failure
        has_failure = (max_priority > 0.01).float()

        # Create exclusive masks (memory efficient - one at a time)
        exclusive = {}
        remaining_mask = has_failure  # Don't clone, we'll update in place

        # Process in priority order: P1 > P4 > P2 > P3
        for name in ['P1', 'P4', 'P2', 'P3']:
            # Recompute weighted for this predicate
            weighted = failure_maps[name] * self.priorities[name]
            # This predicate wins where it equals max and there's remaining area
            is_winner = (weighted >= max_priority - 1e-6) & (remaining_mask > 0.5)
            mask = is_winner.float() * failure_maps[name]
            exclusive[name] = mask
            # Remove from remaining
            remaining_mask = remaining_mask * (1 - is_winner.float())
            del weighted, is_winner  # Free immediately

        # Clean up
        del max_priority, remaining_mask

        return exclusive


class BoundaryCorrector(nn.Module):
    """P1 Corrector: Edge Enhancement - sharpens boundaries"""

    def __init__(self):
        super().__init__()
        # Larger network for more capacity
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        # Initialize for non-trivial output
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
        self.strength = nn.Parameter(torch.tensor(0.15))

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([denoised, noisy], dim=1)
        return torch.tanh(self.net(x)) * self.strength


class ContrastCorrector(nn.Module):
    """P2 Corrector: Contrast Enhancement - improves layer separation"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
        self.strength = nn.Parameter(torch.tensor(0.15))

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([denoised, noisy], dim=1)
        return torch.tanh(self.net(x)) * self.strength


class NoiseCorrector(nn.Module):
    """P3 Corrector: Additional Smoothing in flat regions"""

    def __init__(self):
        super().__init__()
        # Larger kernel for smoothing effect
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 5, padding=2),
        )
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
        self.strength = nn.Parameter(torch.tensor(0.15))

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([denoised, noisy], dim=1)
        return torch.tanh(self.net(x)) * self.strength


class StructureCorrector(nn.Module):
    """P4 Corrector: Detail Restoration - recovers lost structure"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
        self.strength = nn.Parameter(torch.tensor(0.15))

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy, denoised], dim=1)
        return torch.tanh(self.net(x)) * self.strength


class ConflictFreeCorrector(nn.Module):
    """
    Conflict-free correction with separate networks per predicate.
    Memory-optimized: processes one corrector at a time and frees masks.
    """

    def __init__(self):
        super().__init__()
        self.correctors = nn.ModuleDict({
            'P1': BoundaryCorrector(),
            'P2': ContrastCorrector(),
            'P3': NoiseCorrector(),
            'P4': StructureCorrector(),
        })
        self.resolver = ConflictResolver()

        # Application order: P4 -> P1 -> P2 -> P3
        self.order = ['P4', 'P1', 'P2', 'P3']

    def forward(self, denoised: torch.Tensor, noisy: torch.Tensor,
                predicate_results: Dict) -> Tuple[torch.Tensor, Dict]:

        # Collect failure maps (already detached in SymbolicPredicates)
        failure_maps = {
            name: predicate_results['individual'][name]['failure_map']
            for name in ['P1', 'P2', 'P3', 'P4']
        }

        # Resolve conflicts
        exclusive_masks = self.resolver(failure_maps)

        # Free original failure maps - we have exclusive masks now
        del failure_maps
        maybe_clear_memory()

        # Apply corrections one at a time
        current = denoised
        corrections_info = {}

        # FIX: Don't delete from dict during iteration - collect masks first
        masks_to_apply = [(name, exclusive_masks[name]) for name in self.order]

        # Now safe to clear the original dict
        exclusive_masks.clear()

        for name, mask in masks_to_apply:
            mask_sum = mask.sum().item()
            mask_max = mask.max().item()

            if mask_sum > 1:
                correction = self.correctors[name](current, noisy)
                # Apply correction with continuous mask - preserves PSNR better
                # Scale mask to [0, 1] based on its max value for consistent correction strength
                scaled_mask = mask / (mask_max + 1e-6)
                # Cap correction magnitude to prevent extreme changes
                correction_capped = correction.clamp(-0.15, 0.15)
                current = current + scaled_mask * correction_capped
                corrections_info[name] = {'sum': mask_sum, 'max': mask_max}
                del correction, correction_capped, scaled_mask

            del mask

        del masks_to_apply  # Clean up list
        current = current.clamp(0, 1)

        return current, corrections_info


# =============================================================================
# COMPLETE NEURO-SYMBOLIC DENOISER
# =============================================================================

class NeuroSymbolicDenoiser(nn.Module):
    """
    Complete Neuro-Symbolic OCT Denoiser.

    Integration:
    - Neural backbone (NAFNet) for initial denoising
    - Symbolic predicates for quality verification
    - Conflict-free neural correctors guided by predicates
    - Iterative refinement until predicates pass
    """

    def __init__(self, max_iterations: int = 2, width: int = 64):
        super().__init__()

        # Neural backbone - use original architecture to match checkpoints
        from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
        if width == 64:
            # Original architecture (matches pretrained checkpoint)
            self.backbone = NAFNet(
                img_channel=1, width=64,
                middle_blk_num=2,
                enc_blk_nums=[2, 2, 2],
                dec_blk_nums=[2, 2, 2]
            )
        else:
            # Light architecture for low-memory systems
            self.backbone = NAFNet(
                img_channel=1, width=width,
                middle_blk_num=1,
                enc_blk_nums=[1, 2, 2],
                dec_blk_nums=[2, 2, 1]
            )

        # Symbolic predicates
        self.predicates = SymbolicPredicates()

        # Conflict-free corrector
        self.corrector = ConflictFreeCorrector()

        self.max_iterations = max_iterations

    def load_pretrained_backbone(self, path: str) -> bool:
        """Load pretrained NAFNet weights. Returns True if successful."""
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            # Check if state dict is compatible
            model_state = self.backbone.state_dict()
            ckpt_state = ckpt.get('state_dict', ckpt)

            # Only load matching keys
            compatible_keys = []
            for k in ckpt_state.keys():
                if k in model_state and ckpt_state[k].shape == model_state[k].shape:
                    compatible_keys.append(k)

            if len(compatible_keys) == 0:
                print(f"No compatible weights found (architecture mismatch)")
                return False

            # Load compatible weights
            filtered_state = {k: ckpt_state[k] for k in compatible_keys}
            self.backbone.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(compatible_keys)}/{len(model_state)} compatible weights")
            print(f"Checkpoint PSNR: {ckpt.get('psnr', 'N/A')}")
            return True
        except Exception as e:
            print(f"Could not load backbone: {e}")
            return False

    def forward(self, noisy: torch.Tensor,
                return_details: bool = False) -> Dict:

        # Step 1: Neural denoising (may be frozen, so detach to save memory)
        with torch.set_grad_enabled(self.backbone.training or any(p.requires_grad for p in self.backbone.parameters())):
            initial = self.backbone(noisy).clamp(0, 1)

        # Step 2: Iterative correction
        # Track if any correction was made (for gradient flow)
        made_correction = False
        correction_sum = None
        iteration_info = [] if return_details else None
        iterations_done = 0

        for i in range(self.max_iterations):
            iterations_done = i + 1

            # Current state
            if correction_sum is None:
                current = initial.detach().clamp(0, 1)
            else:
                current = (initial.detach() + correction_sum).clamp(0, 1)

            # Symbolic: Evaluate predicates (need failure maps for correction)
            pred_results = self.predicates(noisy, current, return_failure_maps=True)

            if return_details:
                # Store only scalar scores, not tensors
                iteration_info.append({
                    'iteration': i,
                    'scores': {
                        name: pred_results['individual'][name]['score'].item()
                        for name in ['P1', 'P2', 'P3', 'P4']
                    },
                    'all_passed': pred_results['all_passed'],
                })

            # Check if all pass - early exit
            if pred_results['all_passed']:
                del pred_results
                maybe_clear_memory()
                break

            # Neural: Correct based on symbolic guidance
            corrected, corr_info = self.corrector(current, noisy, pred_results)
            delta = corrected - current.detach()

            if correction_sum is None:
                correction_sum = delta
            else:
                correction_sum = correction_sum + delta

            made_correction = True

            # Free predicate results after correction
            del pred_results, corrected, delta
            maybe_clear_memory()

        # Final output
        if correction_sum is None or not made_correction:
            # No correction was made (all predicates passed)
            final_denoised = initial.detach()
            has_gradient = False
            correction_coverage = 0.0
        else:
            final_denoised = (initial.detach() + correction_sum).clamp(0, 1)
            has_gradient = True
            # Compute correction coverage (% pixels that changed significantly)
            diff = torch.abs(final_denoised - initial.detach())
            correction_coverage = (diff > 0.01).float().mean().item()

        # Evaluate predicates on INITIAL (backbone only) for comparison
        with torch.no_grad():
            initial_results = self.predicates(noisy, initial.detach(), return_failure_maps=False)

        # Final evaluation on corrected output
        final_results = self.predicates(noisy, final_denoised, return_failure_maps=False)

        output = {
            'denoised': final_denoised,
            'initial': initial.detach(),
            'predicate_results': final_results,
            'initial_predicate_results': initial_results,  # For delta computation
            'iterations': iterations_done,
            'has_gradient': has_gradient,
            'correction_coverage': correction_coverage,
        }

        if return_details:
            output['iteration_info'] = iteration_info

        return output


# =============================================================================
# TRAINING LOSS
# =============================================================================

class NeuroSymbolicLoss(nn.Module):
    """
    Combined loss for neuro-symbolic training.
    Separate losses per predicate to avoid gradient conflicts.
    Memory-optimized: doesn't require failure maps.

    PSNR-aware: penalizes corrections that degrade image quality.
    """

    def __init__(self, lambda_pred: float = 0.05, lambda_smooth: float = 0.05,
                 lambda_psnr: float = 0.5):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_smooth = lambda_smooth
        self.lambda_psnr = lambda_psnr  # Penalize PSNR degradation

    def forward(self, output: Dict, clean: torch.Tensor,
                noisy: torch.Tensor) -> Dict:

        denoised = output['denoised']
        initial = output['initial']
        pred_results = output['predicate_results']

        # Reconstruction loss (main training signal)
        loss_recon = F.mse_loss(denoised, clean)
        loss_initial = F.mse_loss(initial, clean)

        # Predicate losses (use scores directly - no failure maps needed)
        score_P1 = pred_results['individual']['P1']['score']
        score_P2 = pred_results['individual']['P2']['score']
        score_P3 = pred_results['individual']['P3']['score']
        score_P4 = pred_results['individual']['P4']['score']

        loss_P1 = 1 - score_P1
        loss_P2 = 1 - score_P2
        loss_P3 = 1 - score_P3
        loss_P4 = 1 - score_P4

        # Weighted predicate loss
        loss_pred = 0.35 * loss_P1 + 0.2 * loss_P2 + 0.15 * loss_P3 + 0.3 * loss_P4

        # Smoothness: don't change too much from initial
        loss_smooth = F.mse_loss(denoised, initial.detach())

        # PSNR preservation: penalize if correction makes MSE worse than backbone
        # This ensures corrections improve predicates without degrading image quality
        loss_psnr_degrade = F.relu(loss_recon - loss_initial.detach())  # Only penalize if worse

        # Total loss: reconstruction + predicate scores + smoothness + PSNR preservation
        # All terms have gradients that flow through the corrector networks
        total = (loss_recon +
                 self.lambda_pred * loss_pred +
                 self.lambda_smooth * loss_smooth +
                 self.lambda_psnr * loss_psnr_degrade)

        # Return dict - total keeps gradient, others are detached for logging
        return {
            'total': total,  # Keep gradient for backward
            'reconstruction': loss_recon.detach(),
            'initial_recon': loss_initial.detach(),
            'predicate': loss_pred.detach() if isinstance(loss_pred, torch.Tensor) else loss_pred,
            'P1': loss_P1.detach() if isinstance(loss_P1, torch.Tensor) else loss_P1,
            'P2': loss_P2.detach() if isinstance(loss_P2, torch.Tensor) else loss_P2,
            'P3': loss_P3.detach() if isinstance(loss_P3, torch.Tensor) else loss_P3,
            'P4': loss_P4.detach() if isinstance(loss_P4, torch.Tensor) else loss_P4,
            'smooth': loss_smooth.detach(),
        }


# =============================================================================
# DATA LOADING (Memory Efficient)
# =============================================================================

class OCTDataset:
    """OCT dataset with patch-based training support."""

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None,
                 patch_size: int = 128, is_train: bool = True):
        self.samples = []
        self.patch_size = patch_size
        self.is_train = is_train

        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        # Subsample if needed
        if max_samples and len(lines) > max_samples:
            step = len(lines) // max_samples
            lines = lines[::step][:max_samples]

        for line in lines:
            data = json.loads(line)
            self.samples.append({
                'clean_path': data['clean_path'],
                'noisy_path': data['noisy_path'],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load images
        clean = Image.open(sample['clean_path']).convert('L')
        noisy = Image.open(sample['noisy_path']).convert('L')

        # Convert to tensor [H, W]
        clean = torch.from_numpy(np.array(clean)).float() / 255.0
        noisy = torch.from_numpy(np.array(noisy)).float() / 255.0

        if self.is_train and self.patch_size > 0:
            # Random crop for training
            H, W = clean.shape
            ps = self.patch_size

            # FIX: Use >= to handle exact size case
            if H >= ps and W >= ps:
                top = np.random.randint(0, H - ps + 1)  # +1 to include 0 when H==ps
                left = np.random.randint(0, W - ps + 1)
                clean = clean[top:top+ps, left:left+ps]
                noisy = noisy[top:top+ps, left:left+ps]

            # Random augmentations
            if np.random.rand() > 0.5:
                clean = torch.flip(clean, [1])  # Horizontal flip
                noisy = torch.flip(noisy, [1])
            if np.random.rand() > 0.5:
                clean = torch.flip(clean, [0])  # Vertical flip
                noisy = torch.flip(noisy, [0])

        return {
            'clean': clean.unsqueeze(0),
            'noisy': noisy.unsqueeze(0),
        }


class SlidingWindowInference:
    """
    Fast sliding window inference for validation.

    Strategy:
    1. Batch backbone processing on patches for speed
    2. Apply corrector on full aggregated image (single pass)
    3. Evaluate predicates on sparse samples
    """

    def __init__(self, model: nn.Module, patch_size: int = 128, stride: int = 64,
                 batch_size: int = 8):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, noisy: torch.Tensor) -> Dict:
        """
        Apply model with sliding window and neuro-symbolic correction.

        Args:
            noisy: [1, 1, H, W] input image

        Returns:
            Dict with 'denoised', 'initial', 'predicate_results'
        """
        B, C, H, W = noisy.shape
        ps = self.patch_size
        stride = self.stride

        # Handle images smaller than patch size
        if H < ps or W < ps:
            output = self.model(noisy, return_details=False)
            return output

        # ===== STEP 1: Batch backbone processing =====
        initial_sum = torch.zeros_like(noisy)
        weight_sum = torch.zeros_like(noisy)

        # Collect patch positions
        positions = []
        for top in range(0, H - ps + 1, stride):
            for left in range(0, W - ps + 1, stride):
                positions.append((top, left))

        # Edge patches
        last_top = ((H - ps) // stride) * stride
        last_left = ((W - ps) // stride) * stride

        if last_left + ps < W:
            for top in range(0, H - ps + 1, stride):
                if (top, W - ps) not in positions:
                    positions.append((top, W - ps))

        if last_top + ps < H:
            for left in range(0, W - ps + 1, stride):
                if (H - ps, left) not in positions:
                    positions.append((H - ps, left))

        if last_left + ps < W and last_top + ps < H:
            if (H - ps, W - ps) not in positions:
                positions.append((H - ps, W - ps))

        # Process patches in batches
        for batch_start in range(0, len(positions), self.batch_size):
            batch_positions = positions[batch_start:batch_start + self.batch_size]
            patches = [noisy[:, :, top:top+ps, left:left+ps] for top, left in batch_positions]
            batch = torch.cat(patches, dim=0)

            output_batch = self.model.backbone(batch).clamp(0, 1)

            for i, (top, left) in enumerate(batch_positions):
                initial_sum[:, :, top:top+ps, left:left+ps] += output_batch[i:i+1]
                weight_sum[:, :, top:top+ps, left:left+ps] += 1.0

            del batch, output_batch, patches

        # Average overlapping regions
        weight_sum = weight_sum.clamp(min=1.0)
        initial = initial_sum / weight_sum
        del initial_sum, weight_sum

        # ===== STEP 2: Apply neuro-symbolic correction on full image =====
        # Evaluate predicates on initial
        initial_pred = self.model.predicates(noisy, initial, return_failure_maps=True)

        # Apply corrector if any predicates fail
        if not initial_pred['all_passed']:
            corrected, corr_info = self.model.corrector(initial, noisy, initial_pred)
            denoised = corrected
            # Lower threshold for detecting changes (was 0.01)
            diff = torch.abs(denoised - initial)
            correction_coverage = (diff > 0.001).float().mean().item()
            max_correction = diff.max().item()
        else:
            denoised = initial
            correction_coverage = 0.0
            max_correction = 0.0
            corr_info = {}

        # ===== STEP 3: Evaluate final predicates =====
        final_pred = self.model.predicates(noisy, denoised, return_failure_maps=False)

        # Build initial predicate results (backbone only)
        initial_pred_results = {
            'individual': {
                name: {
                    'score': initial_pred['individual'][name]['score'],
                    'passed': initial_pred['individual'][name]['passed'],
                    'name': name
                }
                for name in ['P1', 'P2', 'P3', 'P4']
            },
            'all_passed': initial_pred['all_passed'],
            'combined_score': initial_pred['combined_score'],
        }

        # Build final predicate results (after correction)
        final_pred_results = {
            'individual': {
                name: {
                    'score': final_pred['individual'][name]['score'],
                    'passed': final_pred['individual'][name]['passed'],
                    'name': name
                }
                for name in ['P1', 'P2', 'P3', 'P4']
            },
            'all_passed': final_pred['all_passed'],
            'combined_score': final_pred['combined_score'],
        }

        return {
            'denoised': denoised,
            'initial': initial,
            'predicate_results': final_pred_results,
            'initial_predicate_results': initial_pred_results,
            'iterations': 1 if not initial_pred['all_passed'] else 0,
            'has_gradient': False,
            'correction_coverage': correction_coverage,
            'max_correction': max_correction,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        'clean': torch.stack([b['clean'] for b in batch]),
        'noisy': torch.stack([b['noisy'] for b in batch]),
    }


# =============================================================================
# TRAINING AND VALIDATION
# =============================================================================

class Trainer:
    """Training manager with patch-based training and sliding window validation."""

    def __init__(self, model: NeuroSymbolicDenoiser,
                 train_dataset: OCTDataset,
                 val_dataset: OCTDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4,
                 freeze_backbone: bool = True,
                 patch_size: int = 128,
                 val_stride: int = 64,
                 val_batch_size: int = 8):

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.patch_size = patch_size

        # Sliding window inference for validation (batched for speed)
        self.sliding_window = SlidingWindowInference(
            model, patch_size=patch_size, stride=val_stride, batch_size=val_batch_size
        )

        if freeze_backbone:
            # Freeze backbone to save gradient memory
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen - only training correctors")

            # Only train corrector with higher LR
            self.optimizer = torch.optim.Adam([
                {'params': model.corrector.parameters(), 'lr': lr * 3},  # Higher LR for correctors
            ])
        else:
            # Train everything
            self.optimizer = torch.optim.Adam([
                {'params': model.corrector.parameters(), 'lr': lr * 3},  # Higher LR for correctors
                {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            ])

        self.loss_fn = NeuroSymbolicLoss()
        self.best_val_score = 0

        print(f"Training with {patch_size}x{patch_size} patches, batch_size={batch_size}")
        print(f"Validation: sliding window stride={val_stride}, batch_size={val_batch_size}")

    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch with aggressive memory management."""
        self.model.train()

        metrics = {
            'loss': [], 'loss_recon': [], 'loss_pred': [],
            'P1': [], 'P2': [], 'P3': [], 'P4': [],
            'psnr': [], 'ssim': [],
        }

        indices = np.random.permutation(len(self.train_dataset))
        n_batches = len(indices) // self.batch_size

        # FIX: Handle case when dataset < batch_size
        if n_batches == 0 and len(indices) > 0:
            n_batches = 1  # Process at least one batch with available samples

        start_time = time.time()

        for batch_idx in range(n_batches):
            # Get batch
            batch_indices = indices[batch_idx * self.batch_size:
                                   (batch_idx + 1) * self.batch_size]

            samples = [self.train_dataset[i] for i in batch_indices]
            batch = collate_fn(samples)
            del samples  # Free sample list

            clean = batch['clean']
            noisy = batch['noisy']
            del batch  # Free batch dict

            # Forward
            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
            output = self.model(noisy)

            # Loss
            losses = self.loss_fn(output, clean, noisy)

            # Get metrics before backward (denoised is still valid)
            with torch.no_grad():
                psnr = compute_psnr(output['denoised'], clean)
                ssim = compute_ssim(output['denoised'], clean)

            # Store metrics (use module-level to_float)
            metrics['loss'].append(to_float(losses['total']))
            metrics['loss_recon'].append(to_float(losses['reconstruction']))
            metrics['loss_pred'].append(to_float(losses['predicate']))
            metrics['P1'].append(1 - to_float(losses['P1']))
            metrics['P2'].append(1 - to_float(losses['P2']))
            metrics['P3'].append(1 - to_float(losses['P3']))
            metrics['P4'].append(1 - to_float(losses['P4']))
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)

            # Backward (only if corrections were made, otherwise no gradient)
            has_grad = output.get('has_gradient', True)
            loss_total = losses['total']
            del output, losses  # Free before backward

            if has_grad and loss_total.requires_grad:
                loss_total.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            del loss_total

            # Clear memory periodically (not every batch - too slow)
            del clean, noisy
            maybe_clear_memory()

        elapsed = time.time() - start_time

        # Aggregate metrics (handle empty lists)
        result = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}
        result['time'] = elapsed
        result['samples_per_sec'] = len(indices) / elapsed if elapsed > 0 else 0.0

        return result

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate model using fast sliding window inference."""
        self.model.eval()

        metrics = {
            'loss': [], 'loss_recon': [], 'loss_pred': [],
            # Final predicate scores (after correction)
            'P1_score': [], 'P2_score': [], 'P3_score': [], 'P4_score': [],
            'P1_pass': [], 'P2_pass': [], 'P3_pass': [], 'P4_pass': [],
            'all_pass': [],
            # Initial predicate scores (backbone only, before correction)
            'P1_init': [], 'P2_init': [], 'P3_init': [], 'P4_init': [],
            'all_pass_init': [],
            # Image quality metrics
            'psnr_initial': [], 'psnr_final': [],
            'ssim_initial': [], 'ssim_final': [],
            # Correction stats
            'iterations': [],
            'correction_coverage': [],
            'max_correction': [],
        }

        start_time = time.time()

        for i in range(len(self.val_dataset)):
            sample = self.val_dataset[i]
            clean = sample['clean'].unsqueeze(0)
            noisy = sample['noisy'].unsqueeze(0)
            del sample

            # Use sliding window for full image inference
            output = self.sliding_window(noisy)

            # Loss
            losses = self.loss_fn(output, clean, noisy)

            # Predicate results
            pred = output['predicate_results']

            # Compute metrics
            psnr_initial = compute_psnr(output['initial'], clean)
            psnr_final = compute_psnr(output['denoised'], clean)
            ssim_initial = compute_ssim(output['initial'], clean)
            ssim_final = compute_ssim(output['denoised'], clean)

            # Store metrics (use module-level to_float)
            metrics['loss'].append(to_float(losses['total']))
            metrics['loss_recon'].append(to_float(losses['reconstruction']))
            metrics['loss_pred'].append(to_float(losses['predicate']))

            # Final predicate scores (after correction)
            for name in ['P1', 'P2', 'P3', 'P4']:
                score = pred['individual'][name]['score']
                metrics[f'{name}_score'].append(to_float(score))
                metrics[f'{name}_pass'].append(float(pred['individual'][name]['passed']))

            metrics['all_pass'].append(float(pred['all_passed']))

            # Initial predicate scores (backbone only, before correction)
            init_pred = output.get('initial_predicate_results', {})
            if init_pred:
                for name in ['P1', 'P2', 'P3', 'P4']:
                    init_score = init_pred['individual'][name]['score']
                    metrics[f'{name}_init'].append(to_float(init_score))
                metrics['all_pass_init'].append(float(init_pred.get('all_passed', False)))

            # Image quality metrics
            metrics['psnr_initial'].append(psnr_initial)
            metrics['psnr_final'].append(psnr_final)
            metrics['ssim_initial'].append(ssim_initial)
            metrics['ssim_final'].append(ssim_final)

            # Correction stats
            metrics['iterations'].append(output['iterations'])
            metrics['correction_coverage'].append(output.get('correction_coverage', 0.0))
            metrics['max_correction'].append(output.get('max_correction', 0.0))

            # Memory cleanup
            del output, losses, pred, clean, noisy
            maybe_clear_memory()

        elapsed = time.time() - start_time

        # Aggregate (handle empty lists)
        result = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}
        result['time'] = elapsed

        return result

    def print_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Print comprehensive metrics with delta comparisons."""
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch}")
        print("=" * 80)

        print("\n[TRAINING]")
        print(f"  Loss: {train_metrics['loss']:.4f} "
              f"(Recon: {train_metrics['loss_recon']:.4f}, "
              f"Pred: {train_metrics['loss_pred']:.4f})")
        print(f"  PSNR: {train_metrics['psnr']:.2f} dB, SSIM: {train_metrics['ssim']:.4f}")
        print(f"  Predicate Scores: P1={train_metrics['P1']:.3f}, "
              f"P2={train_metrics['P2']:.3f}, P3={train_metrics['P3']:.3f}, "
              f"P4={train_metrics['P4']:.3f}")
        print(f"  Time: {train_metrics['time']:.1f}s "
              f"({train_metrics['samples_per_sec']:.2f} samples/sec)")

        print("\n[VALIDATION]")

        # Image quality with deltas
        delta_psnr = val_metrics['psnr_final'] - val_metrics['psnr_initial']
        delta_ssim = val_metrics['ssim_final'] - val_metrics['ssim_initial']
        print(f"  Image Quality:")
        print(f"    PSNR: {val_metrics['psnr_initial']:.2f} → {val_metrics['psnr_final']:.2f} dB (Δ={delta_psnr:+.3f})")
        print(f"    SSIM: {val_metrics['ssim_initial']:.4f} → {val_metrics['ssim_final']:.4f} (Δ={delta_ssim:+.4f})")

        # Predicate performance with before/after comparison
        print(f"\n  Predicate Scores (Backbone → +Correction):")
        for name, label in [('P1', 'Boundary'), ('P2', 'Contrast'),
                            ('P3', 'Noise'), ('P4', 'Structure')]:
            init_score = val_metrics.get(f'{name}_init', 0)
            final_score = val_metrics[f'{name}_score']
            delta = final_score - init_score
            pass_rate = val_metrics[f'{name}_pass'] * 100
            print(f"    {name} ({label:9s}): {init_score:.3f} → {final_score:.3f} "
                  f"(Δ={delta:+.3f}) | Pass: {pass_rate:.1f}%")

        # Overall pass rates
        init_all_pass = val_metrics.get('all_pass_init', 0) * 100
        final_all_pass = val_metrics['all_pass'] * 100
        print(f"\n  ALL PASS Rate: {init_all_pass:.1f}% → {final_all_pass:.1f}% "
              f"(Δ={final_all_pass - init_all_pass:+.1f}%)")

        # Correction statistics
        coverage = val_metrics.get('correction_coverage', 0) * 100
        max_corr = val_metrics.get('max_correction', 0)
        print(f"\n  Correction Stats:")
        print(f"    Coverage: {coverage:.1f}% of pixels modified (>0.001 change)")
        print(f"    Max correction: {max_corr:.4f}")
        print(f"    Iterations: {val_metrics['iterations']:.2f}")
        print(f"    Time: {val_metrics['time']:.1f}s")

        print("=" * 80)
        sys.stdout.flush()  # Ensure output is written immediately

    def train(self, epochs: int = 10, save_path: str = 'outputs/neuro_symbolic'):
        """Full training loop."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 80)
        print("# NEURO-SYMBOLIC OCT DENOISING TRAINING")
        print("#" * 80)
        print(f"\nTrain samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Print
            self.print_metrics(epoch, train_metrics, val_metrics)

            # Save best - balance predicate improvement with PSNR preservation
            # Score = avg predicate score + bonus for minimal PSNR degradation
            avg_pred_score = (
                val_metrics['P1_score'] + val_metrics['P2_score'] +
                val_metrics['P3_score'] + val_metrics['P4_score']
            ) / 4

            # PSNR penalty: penalize if final PSNR is much worse than initial
            psnr_delta = val_metrics['psnr_final'] - val_metrics['psnr_initial']
            # Convert to [0, 1] range: 0 if psnr_delta <= -5, 1 if psnr_delta >= 0
            psnr_bonus = max(0, min(1, (psnr_delta + 5) / 5)) * 0.1

            # Combined score: predicates + PSNR preservation bonus
            combined_score = avg_pred_score + psnr_bonus

            if combined_score > self.best_val_score:
                self.best_val_score = combined_score
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                }, f"{save_path}/best_model.pth")
                print(f"\n  *** Saved best model (pred: {avg_pred_score:.4f}, psnr_delta: {psnr_delta:+.2f} dB, combined: {combined_score:.4f}) ***")
                sys.stdout.flush()

            # Save latest
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, f"{save_path}/latest_model.pth")

            maybe_clear_memory()

        print("\n" + "#" * 80)
        print("# TRAINING COMPLETE")
        print("#" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("NEURO-SYMBOLIC OCT DENOISING")
    print("=" * 80)

    # Memory-optimized settings for CPU
    # Use width=64 to match pretrained checkpoint, but with memory optimizations
    USE_PRETRAINED = True
    BACKBONE_WIDTH = 64 if USE_PRETRAINED else 32

    # Create model
    print("\nInitializing model...")
    print(f"Using width={BACKBONE_WIDTH} (pretrained={USE_PRETRAINED})")
    model = NeuroSymbolicDenoiser(max_iterations=1, width=BACKBONE_WIDTH)  # Reduced iterations

    # Load pretrained backbone if available and compatible
    if USE_PRETRAINED:
        backbone_path = 'outputs/nafnet_pku37/nafnet_best.pth'
        if Path(backbone_path).exists():
            print("Loading pretrained NAFNet backbone...")
            if model.load_pretrained_backbone(backbone_path):
                print("Successfully loaded pretrained weights")
            else:
                print("Training from scratch...")
        else:
            print("Checkpoint not found, training from scratch...")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Estimate memory usage
    param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Estimated parameter memory: {param_memory_mb:.1f} MB")

    # Configuration - optimized for speed
    PATCH_SIZE = 128  # Training patch size
    VAL_STRIDE = 112  # Validation stride (larger = faster, 112 gives good coverage)
    BATCH_SIZE = 4    # Training batch size
    VAL_BATCH_SIZE = 8  # Validation batch size (batched backbone inference)

    # Load REAL noisy data from PKU37 (not synthetic)
    print("\nLoading REAL noisy data from PKU37...")
    train_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_train.jsonl',  # Real noisy images
        max_samples=300,  # More training data
        patch_size=PATCH_SIZE,
        is_train=True
    )
    val_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_val.jsonl',  # Real noisy images
        max_samples=20,  # More validation
        patch_size=0,  # Full images for validation
        is_train=False
    )
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    print(f"Training patches: {PATCH_SIZE}x{PATCH_SIZE}, Batch size: {BATCH_SIZE}")

    # Force garbage collection before training
    maybe_clear_memory()

    # Create trainer with patch-based training and fast validation
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        freeze_backbone=True,
        patch_size=PATCH_SIZE,
        val_stride=VAL_STRIDE,
        val_batch_size=VAL_BATCH_SIZE
    )

    # Train longer for better convergence
    trainer.train(epochs=20, save_path='outputs/neuro_symbolic')


def benchmark_sliding_window():
    """Benchmark sliding window inference speed."""
    print("=" * 60)
    print("SLIDING WINDOW BENCHMARK")
    print("=" * 60)

    # Create model
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    model = NeuroSymbolicDenoiser(max_iterations=1, width=64)

    # Create dummy input (640x640 typical OCT size)
    H, W = 640, 640
    noisy = torch.rand(1, 1, H, W)

    # Test with different settings
    configs = [
        {'stride': 64, 'batch_size': 1, 'name': 'Old (stride=64, batch=1)'},
        {'stride': 96, 'batch_size': 1, 'name': 'Larger stride (stride=96, batch=1)'},
        {'stride': 112, 'batch_size': 4, 'name': 'Batched (stride=112, batch=4)'},
        {'stride': 112, 'batch_size': 8, 'name': 'Batched (stride=112, batch=8)'},
    ]

    model.eval()
    for cfg in configs:
        slider = SlidingWindowInference(
            model, patch_size=128, stride=cfg['stride'], batch_size=cfg['batch_size']
        )

        # Warmup
        with torch.inference_mode():
            _ = slider(noisy)

        # Benchmark
        start = time.time()
        n_runs = 2
        for _ in range(n_runs):
            with torch.inference_mode():
                _ = slider(noisy)
        elapsed = (time.time() - start) / n_runs

        print(f"  {cfg['name']}: {elapsed:.2f}s per {H}x{W} image")

    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_sliding_window()
    else:
        main()
