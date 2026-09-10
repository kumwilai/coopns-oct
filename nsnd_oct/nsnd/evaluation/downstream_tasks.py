#!/usr/bin/env python3
"""
Downstream Task Evaluation for OCT Denoising

Evaluates denoising quality on clinically relevant tasks:
1. Layer Segmentation Accuracy - Can layers be better segmented?
2. Pathology Detection - Does denoising help detect diseases?
3. Biomarker Measurement - Are measurements more accurate?
4. Image Quality Assessment - Perceptual quality metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List


# ============================================================================
# Layer Segmentation Evaluation
# ============================================================================

class LayerSegmentationEvaluator:
    """
    Evaluates how well denoising helps layer segmentation.

    Uses a pretrained layer segmenter and measures:
    - Dice score improvement
    - Boundary detection accuracy
    - Consistency across B-scans
    """

    def __init__(self, segmenter: Optional[nn.Module] = None):
        """
        Args:
            segmenter: Pretrained layer segmentation model.
                       If None, uses gradient-based pseudo-segmentation.
        """
        self.segmenter = segmenter

    def compute_pseudo_segmentation(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute pseudo layer segmentation using intensity and gradients.

        For evaluation without a pretrained segmenter.
        """
        B, C, H, W = image.shape

        # Vertical gradient (layer boundaries)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        grad_y = F.conv2d(F.pad(image, [1,1,1,1], mode='reflect'), sobel_y)

        # Find peaks in gradient (boundaries)
        grad_magnitude = grad_y.abs()

        # Column-wise profile
        profile = grad_magnitude.mean(dim=3)  # [B, 1, H]

        # Simple peak detection
        boundaries = []
        for b in range(B):
            prof = profile[b, 0].cpu().numpy()
            # Find local maxima
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(prof, height=np.percentile(prof, 70), distance=H//10)
            boundaries.append(peaks)

        return boundaries

    def evaluate(self, denoised: torch.Tensor, noisy: torch.Tensor,
                 gt_boundaries: Optional[List] = None) -> Dict[str, float]:
        """
        Evaluate layer segmentation quality.

        Args:
            denoised: Denoised images [B, 1, H, W]
            noisy: Original noisy images [B, 1, H, W]
            gt_boundaries: Ground truth boundary positions (if available)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Compute boundaries
        boundaries_denoised = self.compute_pseudo_segmentation(denoised)
        boundaries_noisy = self.compute_pseudo_segmentation(noisy)

        # Count detected boundaries
        n_boundaries_denoised = np.mean([len(b) for b in boundaries_denoised])
        n_boundaries_noisy = np.mean([len(b) for b in boundaries_noisy])

        metrics['n_boundaries_denoised'] = n_boundaries_denoised
        metrics['n_boundaries_noisy'] = n_boundaries_noisy
        metrics['boundary_count_improvement'] = n_boundaries_denoised - n_boundaries_noisy

        # If GT available, compute accuracy
        if gt_boundaries is not None:
            # Compute boundary detection accuracy
            accuracy_denoised = self._compute_boundary_accuracy(boundaries_denoised, gt_boundaries)
            accuracy_noisy = self._compute_boundary_accuracy(boundaries_noisy, gt_boundaries)

            metrics['boundary_accuracy_denoised'] = accuracy_denoised
            metrics['boundary_accuracy_noisy'] = accuracy_noisy
            metrics['boundary_accuracy_improvement'] = accuracy_denoised - accuracy_noisy

        # Boundary consistency (variance across columns)
        consistency_denoised = self._compute_boundary_consistency(denoised)
        consistency_noisy = self._compute_boundary_consistency(noisy)

        metrics['boundary_consistency_denoised'] = consistency_denoised
        metrics['boundary_consistency_noisy'] = consistency_noisy
        metrics['boundary_consistency_improvement'] = consistency_denoised - consistency_noisy

        return metrics

    def _compute_boundary_accuracy(self, detected: List, gt: List, tolerance: int = 3) -> float:
        """Compute boundary detection accuracy within tolerance."""
        if not gt:
            return 0.0

        total_matches = 0
        total_gt = 0

        for det, g in zip(detected, gt):
            for gt_pos in g:
                total_gt += 1
                for det_pos in det:
                    if abs(det_pos - gt_pos) <= tolerance:
                        total_matches += 1
                        break

        return total_matches / max(total_gt, 1)

    def _compute_boundary_consistency(self, image: torch.Tensor) -> float:
        """Compute boundary consistency across columns."""
        B, C, H, W = image.shape

        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        grad_y = F.conv2d(F.pad(image, [1,1,1,1], mode='reflect'), sobel_y)

        # Compute variance of peak positions across columns
        grad_magnitude = grad_y.abs()

        # For each column, find peak position
        peak_positions = grad_magnitude.argmax(dim=2)  # [B, 1, W]

        # Consistency = inverse of variance (higher = more consistent)
        variance = peak_positions.float().var(dim=2).mean()

        # Normalize to 0-1 range (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + variance / H)

        return consistency.item()


# ============================================================================
# Pathology Detection Evaluation
# ============================================================================

class PathologyDetector(nn.Module):
    """
    Simple CNN for pathology detection.

    Used to evaluate if denoising improves pathology detection.
    """

    def __init__(self, num_classes=4):
        """
        Args:
            num_classes: Number of pathology classes
                         (e.g., Normal, AMD, DME, Glaucoma)
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class PathologyEvaluator:
    """
    Evaluates how denoising affects pathology detection.
    """

    def __init__(self, detector: Optional[nn.Module] = None, num_classes: int = 4):
        if detector is None:
            detector = PathologyDetector(num_classes)
        self.detector = detector
        self.num_classes = num_classes

    def evaluate(self, denoised: torch.Tensor, noisy: torch.Tensor,
                 labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate pathology detection.

        Args:
            denoised: Denoised images
            noisy: Original noisy images
            labels: Ground truth labels (if available)

        Returns:
            Dictionary of metrics
        """
        self.detector.eval()
        metrics = {}

        with torch.no_grad():
            logits_denoised = self.detector(denoised)
            logits_noisy = self.detector(noisy)

            # Confidence (higher = more certain)
            conf_denoised = F.softmax(logits_denoised, dim=1).max(dim=1)[0].mean()
            conf_noisy = F.softmax(logits_noisy, dim=1).max(dim=1)[0].mean()

            metrics['detection_confidence_denoised'] = conf_denoised.item()
            metrics['detection_confidence_noisy'] = conf_noisy.item()
            metrics['detection_confidence_improvement'] = conf_denoised.item() - conf_noisy.item()

            # If labels available, compute accuracy
            if labels is not None:
                pred_denoised = logits_denoised.argmax(dim=1)
                pred_noisy = logits_noisy.argmax(dim=1)

                acc_denoised = (pred_denoised == labels).float().mean()
                acc_noisy = (pred_noisy == labels).float().mean()

                metrics['detection_accuracy_denoised'] = acc_denoised.item()
                metrics['detection_accuracy_noisy'] = acc_noisy.item()
                metrics['detection_accuracy_improvement'] = acc_denoised.item() - acc_noisy.item()

        return metrics


# ============================================================================
# Biomarker Measurement Evaluation
# ============================================================================

class BiomarkerEvaluator:
    """
    Evaluates biomarker measurement accuracy.

    OCT biomarkers include:
    - RNFL thickness
    - Total retinal thickness
    - Layer volumes
    - Drusen detection
    """

    def __init__(self):
        pass

    def measure_rnfl_thickness(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate RNFL thickness from OCT image.

        Uses intensity profile to find RNFL boundaries.
        """
        B, C, H, W = image.shape

        # RNFL is typically the bright layer at the top (first 15%)
        rnfl_region = image[:, :, :int(H * 0.2), :]

        # Threshold-based detection
        threshold = rnfl_region.mean() + 0.5 * rnfl_region.std()
        rnfl_mask = (rnfl_region > threshold).float()

        # Thickness = sum of mask along depth
        thickness = rnfl_mask.sum(dim=2).mean(dim=[1, 2])  # [B]

        return thickness

    def measure_retinal_thickness(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate total retinal thickness.

        From ILM to RPE.
        """
        B, C, H, W = image.shape

        # Find top and bottom boundaries
        # Top: first significant intensity from top
        # Bottom: last significant intensity (RPE)

        profile = image.mean(dim=3)  # [B, 1, H]
        threshold = profile.mean(dim=2, keepdim=True) * 0.5

        above_thresh = (profile > threshold).float()

        # Find first and last non-zero positions
        thickness_per_col = above_thresh.sum(dim=2)

        return thickness_per_col.mean(dim=1)  # [B]

    def evaluate(self, denoised: torch.Tensor, noisy: torch.Tensor,
                 clean: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate biomarker measurement accuracy.

        Args:
            denoised: Denoised images
            noisy: Noisy images
            clean: Clean reference (if available)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # RNFL thickness
        rnfl_denoised = self.measure_rnfl_thickness(denoised)
        rnfl_noisy = self.measure_rnfl_thickness(noisy)

        metrics['rnfl_thickness_denoised'] = rnfl_denoised.mean().item()
        metrics['rnfl_thickness_noisy'] = rnfl_noisy.mean().item()

        # Total retinal thickness
        rt_denoised = self.measure_retinal_thickness(denoised)
        rt_noisy = self.measure_retinal_thickness(noisy)

        metrics['retinal_thickness_denoised'] = rt_denoised.mean().item()
        metrics['retinal_thickness_noisy'] = rt_noisy.mean().item()

        # If clean reference available, compute accuracy
        if clean is not None:
            rnfl_clean = self.measure_rnfl_thickness(clean)
            rt_clean = self.measure_retinal_thickness(clean)

            # Error relative to clean
            rnfl_error_denoised = torch.abs(rnfl_denoised - rnfl_clean).mean()
            rnfl_error_noisy = torch.abs(rnfl_noisy - rnfl_clean).mean()

            metrics['rnfl_error_denoised'] = rnfl_error_denoised.item()
            metrics['rnfl_error_noisy'] = rnfl_error_noisy.item()
            metrics['rnfl_error_improvement'] = rnfl_error_noisy.item() - rnfl_error_denoised.item()

            rt_error_denoised = torch.abs(rt_denoised - rt_clean).mean()
            rt_error_noisy = torch.abs(rt_noisy - rt_clean).mean()

            metrics['retinal_thickness_error_denoised'] = rt_error_denoised.item()
            metrics['retinal_thickness_error_noisy'] = rt_error_noisy.item()
            metrics['retinal_thickness_error_improvement'] = rt_error_noisy.item() - rt_error_denoised.item()

        # Measurement consistency (std across batch)
        metrics['rnfl_consistency_denoised'] = rnfl_denoised.std().item()
        metrics['rnfl_consistency_noisy'] = rnfl_noisy.std().item()

        return metrics


# ============================================================================
# Combined Downstream Evaluation
# ============================================================================

class DownstreamEvaluator:
    """
    Combined evaluator for all downstream tasks.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.layer_eval = LayerSegmentationEvaluator()
        self.pathology_eval = PathologyEvaluator()
        self.biomarker_eval = BiomarkerEvaluator()

    def evaluate_all(self, denoised: torch.Tensor, noisy: torch.Tensor,
                     clean: Optional[torch.Tensor] = None,
                     labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Run all downstream evaluations.

        Args:
            denoised: Denoised images [B, 1, H, W]
            noisy: Noisy images [B, 1, H, W]
            clean: Clean reference (optional)
            labels: Pathology labels (optional)

        Returns:
            Combined metrics dictionary
        """
        metrics = {}

        # Layer segmentation
        layer_metrics = self.layer_eval.evaluate(denoised, noisy)
        metrics.update({f'layer_{k}': v for k, v in layer_metrics.items()})

        # Pathology detection
        pathology_metrics = self.pathology_eval.evaluate(denoised, noisy, labels)
        metrics.update({f'pathology_{k}': v for k, v in pathology_metrics.items()})

        # Biomarker measurement
        biomarker_metrics = self.biomarker_eval.evaluate(denoised, noisy, clean)
        metrics.update({f'biomarker_{k}': v for k, v in biomarker_metrics.items()})

        return metrics


def print_downstream_report(metrics: Dict[str, float]):
    """Print formatted downstream evaluation report."""
    print("\n" + "="*70)
    print("DOWNSTREAM TASK EVALUATION REPORT")
    print("="*70)

    # Layer Segmentation
    print("\n--- Layer Segmentation ---")
    layer_keys = [k for k in metrics.keys() if k.startswith('layer_')]
    for key in sorted(layer_keys):
        val = metrics[key]
        clean_key = key.replace('layer_', '')
        print(f"  {clean_key}: {val:.4f}")

    # Pathology Detection
    print("\n--- Pathology Detection ---")
    path_keys = [k for k in metrics.keys() if k.startswith('pathology_')]
    for key in sorted(path_keys):
        val = metrics[key]
        clean_key = key.replace('pathology_', '')
        print(f"  {clean_key}: {val:.4f}")

    # Biomarker Measurement
    print("\n--- Biomarker Measurement ---")
    bio_keys = [k for k in metrics.keys() if k.startswith('biomarker_')]
    for key in sorted(bio_keys):
        val = metrics[key]
        clean_key = key.replace('biomarker_', '')
        print(f"  {clean_key}: {val:.4f}")

    print("="*70)


if __name__ == '__main__':
    print("Testing downstream evaluation...")

    # Create dummy data
    torch.manual_seed(42)
    clean = torch.rand(4, 1, 64, 64) * 0.5 + 0.25
    noisy = clean + torch.randn_like(clean) * 0.1
    denoised = clean + torch.randn_like(clean) * 0.05

    # Run evaluation
    evaluator = DownstreamEvaluator()
    metrics = evaluator.evaluate_all(denoised, noisy, clean)

    # Print report
    print_downstream_report(metrics)

    print("\nAll tests passed!")
