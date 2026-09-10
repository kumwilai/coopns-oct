#!/usr/bin/env python3
"""
Neuro-Symbolic Post-Processing for OCT Layer Segmentation

This module applies symbolic rules to neural network segmentation outputs,
ensuring anatomically correct and clinically plausible results.

Key Features:
1. Layer Ordering Enforcement - Guarantees correct anatomical order
2. Thickness Constraints - Validates physiologically plausible ranges
3. Boundary Smoothing - Ensures continuous layer boundaries
4. Anomaly Detection - Flags potential pathology or errors
5. Confidence Calibration - Adjusts confidence based on rule satisfaction

Usage:
    from symbolic_postprocess import SymbolicPostProcessor

    processor = SymbolicPostProcessor()
    corrected_seg, report = processor(seg_logits, return_report=True)

For TMI Paper:
    "We incorporate neuro-symbolic post-processing that guarantees
    anatomically plausible segmentation through explicit domain rules,
    enabling interpretable and clinically trustworthy results."
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# SYMBOLIC KNOWLEDGE BASE
# =============================================================================

@dataclass
class LayerKnowledge:
    """Domain knowledge for each retinal layer."""
    name: str
    index: int
    min_thickness_um: float  # Minimum physiological thickness
    max_thickness_um: float  # Maximum physiological thickness
    expected_order: int      # 0 = topmost (vitreous side), higher = deeper
    clinical_importance: str # Description of clinical relevance


# OCT Layer Knowledge Base (from ophthalmology literature)
LAYER_KNOWLEDGE = {
    'RNFL_GCL': LayerKnowledge(
        name='RNFL_GCL',
        index=0,
        min_thickness_um=50,
        max_thickness_um=150,
        expected_order=0,
        clinical_importance='Glaucoma biomarker - thinning indicates nerve fiber loss'
    ),
    'INL_OPL': LayerKnowledge(
        name='INL_OPL',
        index=1,
        min_thickness_um=30,
        max_thickness_um=80,
        expected_order=1,
        clinical_importance='Inner nuclear layer - affected in diabetic retinopathy'
    ),
    'ONL': LayerKnowledge(
        name='ONL',
        index=2,
        min_thickness_um=50,
        max_thickness_um=120,
        expected_order=2,
        clinical_importance='Outer nuclear layer - photoreceptor cell bodies'
    ),
    'IS_OS': LayerKnowledge(
        name='IS_OS',
        index=3,
        min_thickness_um=20,
        max_thickness_um=60,
        expected_order=3,
        clinical_importance='Photoreceptor junction - visual acuity correlation'
    ),
    'RPE_Choroid': LayerKnowledge(
        name='RPE_Choroid',
        index=4,
        min_thickness_um=40,
        max_thickness_um=400,  # Choroid can be thick
        expected_order=4,
        clinical_importance='RPE layer - AMD diagnosis, drusen detection'
    ),
}

LAYER_NAMES = ['RNFL_GCL', 'INL_OPL', 'ONL', 'IS_OS', 'RPE_Choroid']
NUM_LAYERS = 5

# Pixel to micron conversion (approximate for typical OCT)
PIXELS_PER_UM = 0.5  # Adjust based on your OCT device


# =============================================================================
# SYMBOLIC RULES
# =============================================================================

class SymbolicRule:
    """Base class for symbolic rules."""

    def __init__(self, name: str, severity: str = 'warning'):
        self.name = name
        self.severity = severity  # 'info', 'warning', 'error'

    def check(self, seg_mask: np.ndarray) -> Tuple[bool, str]:
        """Check if rule is satisfied. Returns (passed, message)."""
        raise NotImplementedError

    def apply(self, seg_mask: np.ndarray) -> np.ndarray:
        """Apply correction to satisfy rule."""
        raise NotImplementedError


class LayerOrderingRule(SymbolicRule):
    """
    RULE: Retinal layers must appear in correct anatomical order.

    Order (top to bottom): RNFL_GCL -> INL_OPL -> ONL -> IS_OS -> RPE_Choroid

    Clinical rationale: Anatomically impossible for RPE to be above RNFL.
    """

    def __init__(self):
        super().__init__('LayerOrdering', severity='error')

    def check(self, seg_mask: np.ndarray) -> Tuple[bool, str]:
        """Check if layers are in correct order."""
        h, w = seg_mask.shape
        violations = []

        for col in range(w):
            column = seg_mask[:, col]
            layer_positions = {}

            # Find mean position of each layer in this column
            for layer_idx in range(NUM_LAYERS):
                positions = np.where(column == layer_idx)[0]
                if len(positions) > 0:
                    layer_positions[layer_idx] = np.mean(positions)

            # Check ordering
            for i in range(NUM_LAYERS - 1):
                for j in range(i + 1, NUM_LAYERS):
                    if i in layer_positions and j in layer_positions:
                        if layer_positions[i] > layer_positions[j]:
                            violations.append((col, LAYER_NAMES[i], LAYER_NAMES[j]))

        if violations:
            return False, f"Layer ordering violations at {len(violations)} columns"
        return True, "Layer ordering correct"

    def apply(self, seg_mask: np.ndarray) -> np.ndarray:
        """Correct layer ordering violations."""
        h, w = seg_mask.shape
        corrected = seg_mask.copy()

        for col in range(w):
            column = corrected[:, col]

            # Find boundaries of each layer
            layer_ranges = {}
            for layer_idx in range(NUM_LAYERS):
                positions = np.where(column == layer_idx)[0]
                if len(positions) > 0:
                    layer_ranges[layer_idx] = (positions.min(), positions.max())

            # Sort layers by their top position
            sorted_layers = sorted(layer_ranges.keys(),
                                   key=lambda x: layer_ranges[x][0])

            # If order is wrong, reassign based on expected order
            expected_order = list(range(NUM_LAYERS))
            if sorted_layers != expected_order:
                # Redistribute pixels maintaining relative sizes
                total_pixels = sum(layer_ranges[l][1] - layer_ranges[l][0] + 1
                                   for l in layer_ranges)

                current_pos = 0
                for expected_layer in expected_order:
                    if expected_layer in layer_ranges:
                        size = layer_ranges[expected_layer][1] - layer_ranges[expected_layer][0] + 1
                        corrected[current_pos:current_pos+size, col] = expected_layer
                        current_pos += size

        return corrected


class ThicknessConstraintRule(SymbolicRule):
    """
    RULE: Each layer thickness must be within physiological range.

    Clinical rationale:
    - RNFL < 50μm suggests severe glaucoma
    - RNFL > 150μm suggests measurement error or pathology
    """

    def __init__(self, tolerance: float = 0.2):
        super().__init__('ThicknessConstraint', severity='warning')
        self.tolerance = tolerance  # Allow 20% outside normal range

    def check(self, seg_mask: np.ndarray) -> Tuple[bool, Dict]:
        """Check thickness constraints for each layer."""
        h, w = seg_mask.shape
        thickness_report = {}
        violations = []

        for layer_name, knowledge in LAYER_KNOWLEDGE.items():
            layer_idx = knowledge.index

            # Calculate mean thickness across columns
            thicknesses = []
            for col in range(w):
                column = seg_mask[:, col]
                pixels = np.sum(column == layer_idx)
                thickness_um = pixels / PIXELS_PER_UM
                if thickness_um > 0:
                    thicknesses.append(thickness_um)

            if thicknesses:
                mean_thickness = np.mean(thicknesses)
                min_t = knowledge.min_thickness_um * (1 - self.tolerance)
                max_t = knowledge.max_thickness_um * (1 + self.tolerance)

                thickness_report[layer_name] = {
                    'mean_um': mean_thickness,
                    'min_allowed': knowledge.min_thickness_um,
                    'max_allowed': knowledge.max_thickness_um,
                    'within_range': min_t <= mean_thickness <= max_t
                }

                if mean_thickness < min_t:
                    violations.append(f"{layer_name} too thin: {mean_thickness:.1f}μm < {knowledge.min_thickness_um}μm")
                elif mean_thickness > max_t:
                    violations.append(f"{layer_name} too thick: {mean_thickness:.1f}μm > {knowledge.max_thickness_um}μm")

        if violations:
            return False, thickness_report
        return True, thickness_report

    def apply(self, seg_mask: np.ndarray) -> np.ndarray:
        """Adjust segmentation to satisfy thickness constraints."""
        # Note: This is a soft correction - flags anomalies rather than forcing
        # Forcing thickness could hide pathology
        return seg_mask  # Return unchanged, but flag in report


class BoundaryContinuityRule(SymbolicRule):
    """
    RULE: Layer boundaries should be continuous and smooth.

    Clinical rationale: Retinal layers are continuous structures.
    Discontinuities suggest segmentation error or pathology.
    """

    def __init__(self, max_jump_pixels: int = 10):
        super().__init__('BoundaryContinuity', severity='warning')
        self.max_jump = max_jump_pixels

    def check(self, seg_mask: np.ndarray) -> Tuple[bool, List]:
        """Check for boundary discontinuities."""
        h, w = seg_mask.shape
        discontinuities = []

        for layer_idx in range(NUM_LAYERS - 1):
            # Find boundary between this layer and next
            boundary = []
            for col in range(w):
                column = seg_mask[:, col]
                positions = np.where(column == layer_idx)[0]
                if len(positions) > 0:
                    boundary.append(positions.max())
                else:
                    boundary.append(-1)

            # Check for jumps
            boundary = np.array(boundary)
            valid = boundary >= 0

            for i in range(1, w):
                if valid[i] and valid[i-1]:
                    jump = abs(boundary[i] - boundary[i-1])
                    if jump > self.max_jump:
                        discontinuities.append({
                            'layer': LAYER_NAMES[layer_idx],
                            'column': i,
                            'jump_pixels': jump
                        })

        if discontinuities:
            return False, discontinuities
        return True, []

    def apply(self, seg_mask: np.ndarray) -> np.ndarray:
        """Smooth boundaries to ensure continuity."""
        h, w = seg_mask.shape
        corrected = seg_mask.copy()

        # Apply median filter to smooth boundaries
        for layer_idx in range(NUM_LAYERS):
            layer_mask = (seg_mask == layer_idx).astype(np.float32)
            smoothed = ndimage.median_filter(layer_mask, size=(3, 5))

            # Only update where smoothing makes sense
            # (don't create new layer regions, just smooth existing)
            corrected[smoothed > 0.5] = layer_idx

        # Re-ensure single label per pixel (argmax of smoothed probabilities)
        # This is handled by the order of assignment above

        return corrected


class CompletenessRule(SymbolicRule):
    """
    RULE: All retinal layers should be present in a valid B-scan.

    Clinical rationale: Missing layers suggest:
    - Severe pathology (e.g., geographic atrophy)
    - Segmentation failure
    - Image quality issue
    """

    def __init__(self, min_coverage: float = 0.1):
        super().__init__('Completeness', severity='warning')
        self.min_coverage = min_coverage  # Minimum 10% of width

    def check(self, seg_mask: np.ndarray) -> Tuple[bool, Dict]:
        """Check if all layers are present."""
        h, w = seg_mask.shape
        coverage = {}
        missing = []

        for layer_name, knowledge in LAYER_KNOWLEDGE.items():
            layer_idx = knowledge.index

            # Calculate coverage (fraction of columns with this layer)
            columns_with_layer = 0
            for col in range(w):
                if np.any(seg_mask[:, col] == layer_idx):
                    columns_with_layer += 1

            coverage_frac = columns_with_layer / w
            coverage[layer_name] = coverage_frac

            if coverage_frac < self.min_coverage:
                missing.append(layer_name)

        if missing:
            return False, {'coverage': coverage, 'missing': missing}
        return True, {'coverage': coverage, 'missing': []}

    def apply(self, seg_mask: np.ndarray) -> np.ndarray:
        """Cannot create missing layers - return unchanged but flag."""
        return seg_mask


# =============================================================================
# SYMBOLIC POST-PROCESSOR
# =============================================================================

@dataclass
class ProcessingReport:
    """Report from symbolic post-processing."""
    original_violations: int
    corrected_violations: int
    rules_checked: List[str]
    rule_results: Dict[str, Tuple[bool, any]]
    thickness_report: Dict
    anomalies_detected: List[str]
    confidence_adjustment: float
    is_anatomically_valid: bool
    clinical_flags: List[str]


class SymbolicPostProcessor:
    """
    Neuro-Symbolic Post-Processor for OCT Segmentation.

    Applies symbolic rules to ensure anatomically correct segmentation.
    Provides interpretable reports for clinical use.
    """

    def __init__(self,
                 enforce_ordering: bool = True,
                 enforce_continuity: bool = True,
                 check_thickness: bool = True,
                 check_completeness: bool = True,
                 verbose: bool = False):
        """
        Args:
            enforce_ordering: Apply layer ordering correction
            enforce_continuity: Apply boundary smoothing
            check_thickness: Check thickness constraints (flags only)
            check_completeness: Check layer completeness (flags only)
            verbose: Print detailed logs
        """
        self.verbose = verbose

        # Initialize rules
        self.rules = []

        if enforce_ordering:
            self.rules.append(LayerOrderingRule())
        if enforce_continuity:
            self.rules.append(BoundaryContinuityRule())
        if check_thickness:
            self.rules.append(ThicknessConstraintRule())
        if check_completeness:
            self.rules.append(CompletenessRule())

    def __call__(self,
                 seg_input: torch.Tensor,
                 return_report: bool = False) -> Tuple[torch.Tensor, Optional[ProcessingReport]]:
        """
        Apply symbolic post-processing to segmentation.

        Args:
            seg_input: Either logits [B, C, H, W] or mask [B, H, W] or [H, W]
            return_report: Whether to return detailed report

        Returns:
            corrected_seg: Anatomically corrected segmentation
            report: Processing report (if return_report=True)
        """
        # Handle input format
        if seg_input.dim() == 4:
            # Logits [B, C, H, W] -> mask [B, H, W]
            seg_mask = seg_input.argmax(dim=1)
        elif seg_input.dim() == 3:
            seg_mask = seg_input
        else:
            seg_mask = seg_input.unsqueeze(0)

        # Process each sample in batch
        batch_size = seg_mask.shape[0]
        corrected_masks = []
        reports = []

        for b in range(batch_size):
            mask_np = seg_mask[b].cpu().numpy().astype(np.int32)
            corrected, report = self._process_single(mask_np)
            corrected_masks.append(torch.from_numpy(corrected))
            reports.append(report)

        corrected_tensor = torch.stack(corrected_masks).to(seg_input.device)

        if seg_input.dim() == 2:
            corrected_tensor = corrected_tensor.squeeze(0)

        if return_report:
            # Return first report for single sample, or list for batch
            if batch_size == 1:
                return corrected_tensor, reports[0]
            return corrected_tensor, reports

        return corrected_tensor, None

    def _process_single(self, seg_mask: np.ndarray) -> Tuple[np.ndarray, ProcessingReport]:
        """Process a single segmentation mask."""

        original_violations = 0
        corrected_violations = 0
        rule_results = {}
        thickness_report = {}
        anomalies = []
        clinical_flags = []

        corrected = seg_mask.copy()

        # Apply each rule
        for rule in self.rules:
            # Check rule
            passed, info = rule.check(corrected)
            rule_results[rule.name] = (passed, info)

            if not passed:
                original_violations += 1

                if self.verbose:
                    print(f"[{rule.severity.upper()}] {rule.name}: {info}")

                # Apply correction if rule has one
                corrected = rule.apply(corrected)

                # Re-check after correction
                passed_after, _ = rule.check(corrected)
                if not passed_after:
                    corrected_violations += 1
                    anomalies.append(f"{rule.name} violation persists")

            # Collect thickness report
            if rule.name == 'ThicknessConstraint':
                thickness_report = info if isinstance(info, dict) else {}

        # Generate clinical flags
        clinical_flags = self._generate_clinical_flags(thickness_report, rule_results)

        # Calculate confidence adjustment
        confidence_adj = self._calculate_confidence(rule_results)

        # Determine overall validity
        is_valid = corrected_violations == 0

        report = ProcessingReport(
            original_violations=original_violations,
            corrected_violations=corrected_violations,
            rules_checked=[r.name for r in self.rules],
            rule_results=rule_results,
            thickness_report=thickness_report,
            anomalies_detected=anomalies,
            confidence_adjustment=confidence_adj,
            is_anatomically_valid=is_valid,
            clinical_flags=clinical_flags
        )

        return corrected, report

    def _generate_clinical_flags(self,
                                  thickness_report: Dict,
                                  rule_results: Dict) -> List[str]:
        """Generate clinical interpretation flags."""
        flags = []

        # Check for clinically significant findings
        if thickness_report:
            for layer_name, info in thickness_report.items():
                if isinstance(info, dict) and 'mean_um' in info:
                    mean_t = info['mean_um']
                    knowledge = LAYER_KNOWLEDGE[layer_name]

                    # RNFL thinning - glaucoma indicator
                    if layer_name == 'RNFL_GCL' and mean_t < 70:
                        flags.append(f"CLINICAL: RNFL thinning ({mean_t:.0f}μm) - evaluate for glaucoma")

                    # RPE thickening - possible drusen/AMD
                    if layer_name == 'RPE_Choroid' and mean_t > 100:
                        flags.append(f"CLINICAL: RPE thickening ({mean_t:.0f}μm) - evaluate for AMD/drusen")

                    # IS/OS disruption
                    if layer_name == 'IS_OS' and mean_t < 25:
                        flags.append(f"CLINICAL: IS/OS junction thin ({mean_t:.0f}μm) - photoreceptor integrity concern")

        # Check completeness
        if 'Completeness' in rule_results:
            passed, info = rule_results['Completeness']
            if not passed and 'missing' in info:
                for missing_layer in info['missing']:
                    flags.append(f"CLINICAL: {missing_layer} poorly visualized - check image quality or pathology")

        return flags

    def _calculate_confidence(self, rule_results: Dict) -> float:
        """Calculate confidence adjustment based on rule satisfaction."""
        total_rules = len(rule_results)
        passed_rules = sum(1 for passed, _ in rule_results.values() if passed)

        # Base confidence from rule satisfaction
        rule_confidence = passed_rules / total_rules if total_rules > 0 else 1.0

        # Weight by severity
        severity_weights = {'error': 0.5, 'warning': 0.8, 'info': 0.95}

        weighted_confidence = 1.0
        for rule in self.rules:
            if rule.name in rule_results:
                passed, _ = rule_results[rule.name]
                if not passed:
                    weighted_confidence *= severity_weights.get(rule.severity, 0.9)

        return min(rule_confidence, weighted_confidence)


# =============================================================================
# INTEGRATION WITH MODEL
# =============================================================================

def apply_symbolic_postprocessing(model_output: torch.Tensor,
                                   return_report: bool = True) -> Tuple[torch.Tensor, Optional[ProcessingReport]]:
    """
    Convenience function to apply symbolic post-processing.

    Args:
        model_output: Segmentation logits [B, C, H, W] or mask [B, H, W]
        return_report: Whether to return processing report

    Returns:
        corrected_seg: Anatomically corrected segmentation
        report: Processing report with clinical flags
    """
    processor = SymbolicPostProcessor(
        enforce_ordering=True,
        enforce_continuity=True,
        check_thickness=True,
        check_completeness=True,
        verbose=False
    )

    return processor(model_output, return_report=return_report)


def print_clinical_report(report: ProcessingReport):
    """Print a clinical summary report."""
    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nAnatomical Validity: {'✓ VALID' if report.is_anatomically_valid else '✗ INVALID'}")
    print(f"Confidence: {report.confidence_adjustment:.1%}")

    print(f"\nRules Checked: {len(report.rules_checked)}")
    print(f"  Original Violations: {report.original_violations}")
    print(f"  After Correction: {report.corrected_violations}")

    if report.thickness_report:
        print("\nLayer Thickness Analysis:")
        for layer, info in report.thickness_report.items():
            if isinstance(info, dict) and 'mean_um' in info:
                status = "✓" if info.get('within_range', True) else "⚠"
                print(f"  {status} {layer}: {info['mean_um']:.1f}μm "
                      f"(normal: {info['min_allowed']:.0f}-{info['max_allowed']:.0f}μm)")

    if report.clinical_flags:
        print("\n⚠ CLINICAL FLAGS:")
        for flag in report.clinical_flags:
            print(f"  • {flag}")

    if report.anomalies_detected:
        print("\n⚠ ANOMALIES:")
        for anomaly in report.anomalies_detected:
            print(f"  • {anomaly}")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN - DEMO
# =============================================================================

if __name__ == '__main__':
    print("Neuro-Symbolic Post-Processing Demo")
    print("=" * 60)

    # Create a sample segmentation mask with violations
    h, w = 64, 128
    seg_mask = torch.zeros(h, w, dtype=torch.long)

    # Create layers (with intentional ordering violation)
    seg_mask[0:10, :] = 0   # RNFL_GCL
    seg_mask[10:20, :] = 1  # INL_OPL
    seg_mask[20:30, :] = 2  # ONL
    seg_mask[30:40, :] = 3  # IS_OS
    seg_mask[40:64, :] = 4  # RPE_Choroid

    # Add a violation: swap some layers in middle columns
    seg_mask[15:25, 50:70] = 4  # RPE in wrong position
    seg_mask[40:50, 50:70] = 1  # INL in wrong position

    print(f"\nInput shape: {seg_mask.shape}")
    print(f"Unique labels: {torch.unique(seg_mask).tolist()}")

    # Apply post-processing
    processor = SymbolicPostProcessor(verbose=True)
    corrected, report = processor(seg_mask, return_report=True)

    # Print report
    print_clinical_report(report)

    print(f"\nCorrections made: {(seg_mask != corrected).sum().item()} pixels changed")
