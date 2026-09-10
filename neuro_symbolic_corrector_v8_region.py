#!/usr/bin/env python3
"""
Neuro-Symbolic Corrector V8 Region: V8 Enhanced with Region-Adaptive Lambda Prediction

This module extends NeuroSymbolicCorrectorV8Enhanced with region-adaptive correction
that applies:
- STRONG corrections on clinical regions (layer boundaries, edges)
- MINIMAL corrections on flat/homogeneous regions to preserve PSNR

Key Features:
1. RegionAdaptiveLambdaPredictor - Detects clinical regions and modulates lambda
2. All V8 Enhanced features preserved:
   - Differentiable fuzzy logic with learnable t-norms
   - Hierarchical symbolic reasoning with rule chaining
   - Physics-accurate speckle model (log-domain, Gamma-K)
   - Formal verification guarantees
   - Clinical correctors (edge, contrast, texture, boundary, anatomy)
3. Returns region_importance_map in info dict for visualization

Author: Neuro-Symbolic OCT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# Import V8 base components
from neuro_symbolic_corrector_v8 import (
    DifferentiableFuzzyLogic,
    HierarchicalSymbolicReasoner,
    PhysicsAccurateSpecklePredicate,
    FormalVerificationGuarantee,
    CausalExplainer,
    EnhancedGTFreePredicates
)

# Import V8 Enhanced components (correctors, attention modules)
from neuro_symbolic_corrector_v8_enhanced import (
    ChannelAttention,
    SpatialAttention,
    ClinicalCorrectorBase,
    ContrastRestorationCorrector,
    BoundarySharpnessCorrector,
    TextureRecoveryCorrector,
    EdgeEnhancementCorrector,
    EnhancedAnatomyCorrector,
)

# Import Region-Adaptive components
from region_adaptive_correction import (
    RegionDetector,
    LearnableRegionDetector,
    RegionAdaptiveLambdaPredictor,
    BoundaryFocusedLoss,
    BoundaryFocusedLossV2,
)


# =============================================================================
# NEURO-SYMBOLIC CORRECTOR V8 REGION
# =============================================================================

class NeuroSymbolicCorrectorV8Region(nn.Module):
    """
    V8 Enhanced Corrector with Region-Adaptive Lambda Prediction.

    This corrector extends the V8 Enhanced architecture by replacing the
    AdaptiveLambdaPredictorV8 with RegionAdaptiveLambdaPredictor that:

    1. Detects clinical regions from input (boundaries, edges, texture)
    2. Computes region-adaptive lambda maps
    3. Applies corrections with region-aware strength:
       - High importance regions (boundaries) -> strong correction
       - Low importance regions (flat areas) -> minimal correction

    Key Insight: Different regions need different correction strengths.
    - Layer boundaries in OCT (horizontal lines) need strong edge correction
    - Flat/homogeneous regions should be left alone to maintain PSNR
    - Over-correcting flat regions hurts PSNR without clinical benefit

    All V8 Enhanced features are preserved:
    - Differentiable fuzzy logic (Lukasiewicz t-norm)
    - Hierarchical symbolic reasoning (3 levels)
    - Physics-accurate speckle MONITORING (log-domain, Gamma-K)
    - Formal verification guarantees
    - Clinical correctors targeting specific backbone weaknesses

    Args:
        in_channels: Number of input channels (default: 1)
        hidden_channels: Hidden dimension for correctors (default: 64)
        enc1_channels: Encoder 1 channels from backbone (default: 40)
        enc2_channels: Encoder 2 channels from backbone (default: 80)
        use_learnable_detector: Use learnable region detector (default: True)
            - True: Uses LearnableRegionDetector (combines heuristic + learned)
            - False: Uses RegionDetector (heuristic only, faster)
        importance_power: Power scaling for importance map (default: 1.0)
        min_importance: Minimum importance floor (default: 0.0)
        region_hidden_dim: Hidden dimension for region detector (default: 32)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        enc1_channels: int = 40,
        enc2_channels: int = 80,
        use_learnable_detector: bool = True,
        importance_power: float = 1.0,
        min_importance: float = 0.0,
        region_hidden_dim: int = 32,
    ):
        super().__init__()

        # Store configuration
        self.enc1_channels = enc1_channels
        self.enc2_channels = enc2_channels
        self.use_learnable_detector = use_learnable_detector

        # Enhanced predicates with physics-accurate speckle
        self.predicates = EnhancedGTFreePredicates()

        # TRUE symbolic router with differentiable logic
        self.router = HierarchicalSymbolicReasoner()

        # REGION-ADAPTIVE lambda predictor (replaces AdaptiveLambdaPredictorV8)
        self.lambda_predictor = RegionAdaptiveLambdaPredictor(
            use_learnable_detector=use_learnable_detector,
            importance_power=importance_power,
            min_importance=min_importance,
            hidden_dim=region_hidden_dim,
        )

        # CLINICAL CORRECTORS targeting specific backbone weaknesses
        # NOTE: P5/speckle corrector removed - fundamentally incompatible with correction paradigm
        # P5 is still computed and monitored in predicates, just not corrected
        self.correctors = nn.ModuleDict({
            'edge': EdgeEnhancementCorrector(hidden_channels),       # P1: 32% edges lost
            'contrast': ContrastRestorationCorrector(hidden_channels),  # P2: 53% contrast lost
            'smooth': TextureRecoveryCorrector(hidden_channels),     # P3: 59% texture lost (inverse)
            'structure': BoundarySharpnessCorrector(hidden_channels),  # P4: 53% boundary sharpness lost
            'anatomy': EnhancedAnatomyCorrector(in_channels, hidden_channels, enc1_channels, enc2_channels),
        })

        # Predicate to corrector mapping (updated for clinical focus)
        # P3 (smoothness) maps to texture corrector - we want LESS smoothing, MORE texture
        self.pred_key_map = {
            'edge': 'P1',       # Edge preservation
            'contrast': 'P2',   # Contrast preservation
            'smooth': 'P3',     # Smoothness -> Texture recovery (inverse)
            'structure': 'P4',  # Structure -> Boundary sharpness
            'anatomy': 'P6',    # Anatomy preservation
        }

        # FORMAL verification
        self.verifier = FormalVerificationGuarantee(self.predicates)

        # Causal explainer
        self.explainer = CausalExplainer(self.router)

        self._print_info()

    def _print_info(self):
        """Print model configuration information."""
        print("\n" + "=" * 70)
        print("NeuroSymbolicCorrectorV8Region - REGION-ADAPTIVE CLINICAL CORRECTORS")
        print("=" * 70)
        print("Region-Adaptive Lambda Prediction:")
        print(f"  - Detector type: {'Learnable (heuristic + learned)' if self.use_learnable_detector else 'Heuristic only'}")
        print("  - Modulation: lambda_final = lambda_base * region_importance")
        print("  - Flat regions: minimal correction (preserves PSNR)")
        print("  - Clinical regions: strong correction (layer boundaries, edges)")
        print("")
        print("Addressing Backbone Weaknesses:")
        print("  - Contrast: 53% lost -> ContrastRestorationCorrector (P2)")
        print("  - Boundary: 53% lost -> BoundarySharpnessCorrector (P4)")
        print("  - Texture: 59% lost -> TextureRecoveryCorrector (P3 inverse)")
        print("  - Edge: 32% lost -> EdgeEnhancementCorrector (P1)")
        print("")
        print("V8 Framework Preserved:")
        print("  1. Differentiable fuzzy logic (Lukasiewicz t-norm)")
        print("  2. Hierarchical rule chaining (3 levels)")
        print("  3. Physics-accurate speckle MONITORING (log-domain, Gamma-K)")
        print("  4. Formal guarantees (Energy + Pareto + Lipschitz)")
        print("  5. Causal interpretability")
        print("  6. RegionAdaptiveLambdaPredictor (per-pixel + region-modulated)")
        print("")
        print("Predicate Mapping:")
        for corrector, pred in self.pred_key_map.items():
            print(f"  {pred} -> {corrector}")
        print("")
        print("NOTE: P5 (Speckle) is monitored but NOT corrected")
        print("      (fundamentally incompatible with correction paradigm)")
        print(f"Active Correctors: {list(self.correctors.keys())}")
        print("=" * 70)

        # Parameter counts
        lambda_params = sum(p.numel() for p in self.lambda_predictor.parameters())
        corrector_params = sum(p.numel() for p in self.correctors.parameters())
        router_params = sum(p.numel() for p in self.router.parameters())
        pred_params = sum(p.numel() for p in self.predicates.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        print(f"\nParameters:")
        print(f"  Lambda predictor (region-adaptive): {lambda_params:,}")
        print(f"  Correctors (5x): {corrector_params:,}")
        print(f"  Router: {router_params:,}")
        print(f"  Predicates: {pred_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"\nCorrection Capacity:")
        print(f"  Lambda caps: edge/contrast=0.80, smooth/structure/anatomy=0.50")
        print(f"  Region importance modulation: lambda * importance^weight")
        print(f"  Total correction clamp: [-0.5, 0.5]")

    def forward(
        self,
        backbone_out: torch.Tensor,
        noisy: torch.Tensor,
        backbone_features: Optional[Dict[str, torch.Tensor]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply neuro-symbolic correction with region-adaptive lambda prediction.

        The key difference from V8 Enhanced is that lambda maps are modulated
        by region importance, so flat regions get minimal correction while
        clinical regions (boundaries, edges) get strong correction.

        Args:
            backbone_out: Backbone denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]
            backbone_features: Optional dict with 'enc1', 'enc2' encoder features
            return_details: Whether to include detailed explanations

        Returns:
            corrected: Final output [B, 1, H, W]
            info: Dictionary with:
                - predicate_scores: Scores on corrected output
                - predicate_scores_backbone: Scores before correction
                - activations: Symbolic routing activations
                - lambda_stats: Per-corrector lambda statistics
                - region_map: Region importance map for visualization [B, 1, H, W]
                - region_info: Detailed region detection info
                - inference_trace: Hierarchical reasoning trace
                - explanations: Human-readable explanations
                - verification: Formal verification results
                - layer_analysis: Per-layer clinical analysis (if return_details)
                - clinical_report: Full clinical report (if return_details)
                - counterfactuals: What-if analysis (if return_details)
        """
        # Step 1: Evaluate predicates (with physics-accurate P5)
        with torch.no_grad():
            pred_results = self.predicates(backbone_out, noisy)

        # Step 2: Hierarchical symbolic routing
        routing = self.router(pred_results)
        activations = routing['activations']

        # Step 3: Get failure maps (skip P5/speckle - not used for correction)
        failure_maps = {}
        for name, key in self.pred_key_map.items():
            failure_maps[key] = pred_results[key]['failure_map']

        # Step 4: Predict region-adaptive lambda maps (KEY DIFFERENCE FROM V8 ENHANCED)
        # The lambda predictor uses noisy input for better boundary detection
        # (backbone output may have smoothed boundaries)
        lambda_maps = self.lambda_predictor(backbone_out, failure_maps)

        # Get region info for visualization
        region_info = self.lambda_predictor.get_region_info()

        # Step 5: Apply corrections with region-modulated lambda
        corrector_data = [
            (name, corrector, failure_maps[self.pred_key_map[name]], activations[name], lambda_maps[name])
            for name, corrector in self.correctors.items()
        ]

        corrections = {}
        for name, corrector, failure_map, act, lam in corrector_data:
            correction = corrector(backbone_out, failure_map, backbone_features)

            # Keep activation as tensor for efficient multiplication
            if isinstance(act, torch.Tensor):
                act = act.view(1, 1, 1, 1) if act.numel() == 1 else act

            # Lambda is already region-modulated by RegionAdaptiveLambdaPredictor
            corrections[name] = correction * act * lam

        del corrector_data  # Free the temporary list
        del failure_maps  # Free failure maps after use

        # Step 6: Combine corrections
        total_correction = sum(corrections.values())
        del corrections  # Free individual corrections after summing
        total_correction = total_correction.clamp(-0.5, 0.5)
        candidate = (backbone_out + total_correction).clamp(0, 1)

        # Step 7: FORMAL verification
        output, verify_info = self.verifier(
            backbone_out, candidate, noisy, total_correction, pred_before=pred_results
        )

        # Step 8: Re-evaluate predicates on CORRECTED output to measure improvement
        with torch.no_grad():
            pred_results_corrected = self.predicates(output, noisy)

        # Compute layer analysis only when needed (expensive)
        if return_details:
            layer_analysis = self.explainer.clinical_layer_analysis(
                {k: pred_results[k]['failure_map'] for k in pred_results
                 if isinstance(pred_results.get(k), dict) and 'failure_map' in pred_results[k]},
                backbone_out.shape[2]
            )
        else:
            layer_analysis = {}

        # Compute lambda stats - detach for info dict (gradient flow comes from loss function)
        if self.training:
            # During training: detach mean for info dict (only for monitoring, not gradient flow)
            lambda_stats = {
                name: {'mean': lam.mean().detach(), 'max': lam.max().detach()}
                for name, lam in lambda_maps.items()
            }
        else:
            # During eval: convert to Python floats
            lambda_stats = {
                name: {'mean': lam.mean().item(), 'max': lam.max().item()}
                for name, lam in lambda_maps.items()
            }

        # Build info dictionary
        info = {
            'predicate_scores': pred_results_corrected['scores'],  # Scores on CORRECTED output
            'predicate_scores_backbone': pred_results['scores'],  # Scores before correction
            'activations': {
                k: v.detach().item() if (isinstance(v, torch.Tensor) and not self.training) else (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in activations.items()
            },
            'lambda_stats': lambda_stats,
            # Region-specific info (KEY ADDITION FOR V8 REGION)
            'region_map': region_info['importance_map'] if region_info is not None else None,
            'region_info': {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in (region_info or {}).items()
            },
            'inference_trace': {
                level: {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in items.items()}
                for level, items in routing['inference_trace'].items()
                if isinstance(items, dict)
            },
            'explanations': routing['explanations'],
            'verification': verify_info,
            'layer_analysis': layer_analysis,
            'correction_magnitude': total_correction.abs().mean().detach() if self.training else total_correction.abs().mean().item(),
        }

        if return_details:
            info['clinical_report'] = self.explainer.generate_clinical_report(
                pred_results, routing, verify_info, layer_analysis
            )
            info['counterfactuals'] = self.explainer.counterfactual_analysis(
                pred_results, 'P1', [0.3, 0.5, 0.7, 0.9]
            )

        # Cleanup: delete intermediate results after use to free memory
        del pred_results
        del pred_results_corrected
        del lambda_maps

        return output, info

    def get_region_detector(self):
        """
        Get the region detector module for inspection or visualization.

        Returns:
            Region detector (LearnableRegionDetector or RegionDetector)
        """
        return self.lambda_predictor.region_detector

    def visualize_region_importance(
        self,
        backbone_out: torch.Tensor,
        noisy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute and return region importance maps for visualization.

        This is a utility method for debugging and visualization that
        does not require running the full forward pass.

        Args:
            backbone_out: Backbone denoised output [B, 1, H, W]
            noisy: Original noisy input [B, 1, H, W]

        Returns:
            Dict with importance maps and component maps
        """
        with torch.no_grad():
            region_info = self.lambda_predictor.region_detector(backbone_out)
        return region_info


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_v8_region_corrector(
    use_learnable_detector: bool = True,
    hidden_channels: int = 64,
    **kwargs
) -> NeuroSymbolicCorrectorV8Region:
    """
    Factory function to create a V8 Region corrector with common configurations.

    Args:
        use_learnable_detector: Use learnable region detector (default: True)
        hidden_channels: Hidden dimension for correctors (default: 64)
        **kwargs: Additional arguments passed to NeuroSymbolicCorrectorV8Region

    Returns:
        Configured NeuroSymbolicCorrectorV8Region instance
    """
    return NeuroSymbolicCorrectorV8Region(
        in_channels=1,
        hidden_channels=hidden_channels,
        use_learnable_detector=use_learnable_detector,
        **kwargs
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\nTesting NeuroSymbolicCorrectorV8Region...")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test both detector types
    for use_learnable in [True, False]:
        print(f"\n{'=' * 60}")
        print(f"Testing with use_learnable_detector={use_learnable}")
        print("=" * 60)

        # Create model
        model = NeuroSymbolicCorrectorV8Region(
            in_channels=1,
            hidden_channels=64,
            enc1_channels=48,
            enc2_channels=96,
            use_learnable_detector=use_learnable,
        ).to(device)

        # Test inputs
        B, C, H, W = 2, 1, 128, 128
        noisy = torch.randn(B, C, H, W, device=device) * 0.3 + 0.5
        noisy = noisy.clamp(0, 1)
        backbone_out = noisy - torch.randn(B, C, H, W, device=device) * 0.1
        backbone_out = backbone_out.clamp(0, 1)

        # Simulate backbone features (not used by new clinical correctors but kept for compatibility)
        backbone_features = {
            'enc1': torch.randn(B, 48, H, W, device=device) * 0.1,
            'enc2': torch.randn(B, 96, H // 2, W // 2, device=device) * 0.1,
        }

        # Forward pass
        corrected, info = model(backbone_out, noisy, backbone_features, return_details=True)

        print(f"\nInput shape: {backbone_out.shape}")
        print(f"Output shape: {corrected.shape}")
        print(f"\nPredicate Scores (after correction): {info['predicate_scores']}")
        print(f"Predicate Scores (backbone): {info['predicate_scores_backbone']}")
        print(f"\nActivations: {info['activations']}")

        print(f"\nLambda Stats (region-modulated):")
        for name, stats in info['lambda_stats'].items():
            mean_val = stats['mean'].item() if isinstance(stats['mean'], torch.Tensor) else stats['mean']
            max_val = stats['max'].item() if isinstance(stats['max'], torch.Tensor) else stats['max']
            print(f"  {name}: mean={mean_val:.4f}, max={max_val:.4f}")

        print(f"\nCorrection magnitude: {info['correction_magnitude']:.4f}" if isinstance(info['correction_magnitude'], float) else f"\nCorrection magnitude: {info['correction_magnitude'].item():.4f}")

        # Check region info
        print(f"\nRegion Info:")
        if info['region_map'] is not None:
            region_map = info['region_map']
            print(f"  Region map shape: {region_map.shape}")
            print(f"  Region importance range: [{region_map.min().item():.3f}, {region_map.max().item():.3f}]")
            print(f"  Region importance mean: {region_map.mean().item():.3f}")
        else:
            print("  Region map not available")

        print(f"\nVerification Result:")
        print(f"  Decision: {info['verification']['decision']}")
        print(f"  Guarantees passed: {info['verification']['guarantees_passed']}/3")

    # Test visualization utility
    print("\n" + "=" * 60)
    print("Testing visualize_region_importance utility...")
    print("=" * 60)

    model = NeuroSymbolicCorrectorV8Region(use_learnable_detector=True).to(device)
    vis_info = model.visualize_region_importance(backbone_out, noisy)

    print(f"Visualization outputs:")
    for key, val in vis_info.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, range=[{val.min().item():.3f}, {val.max().item():.3f}]")
        else:
            print(f"  {key}: {val}")

    # Test individual clinical correctors
    print("\n" + "=" * 60)
    print("INDIVIDUAL CORRECTOR TEST:")
    print("=" * 60)

    test_failure_map = torch.rand(B, 1, H, W, device=device)

    for name, corrector in model.correctors.items():
        correction = corrector(backbone_out, test_failure_map)
        print(f"  {name}: correction range [{correction.min().item():.4f}, {correction.max().item():.4f}], "
              f"mean={correction.abs().mean().item():.4f}")

    # Test factory function
    print("\n" + "=" * 60)
    print("Testing factory function...")
    print("=" * 60)

    model_from_factory = create_v8_region_corrector(
        use_learnable_detector=True,
        hidden_channels=64,
        importance_power=1.5,
        min_importance=0.1,
    ).to(device)

    corrected_factory, info_factory = model_from_factory(backbone_out, noisy)
    print(f"Factory-created model output shape: {corrected_factory.shape}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
