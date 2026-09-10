"""
NSND-MultiTask Model: Neuro-Symbolic Noise Decomposition with Layer-Aware Denoising

KEY TMI CONTRIBUTIONS:
1. Neuro-symbolic noise decomposition - interpretable noise type classification
2. Layer-aware component fusion - different retinal layers get different noise treatment
3. Joint denoising + segmentation - anatomical structure guides noise removal

Architecture:
                              Input Image
                                   |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
SymbolicNoiseAnalyzer    ComponentDenoiserBank      LightweightLayerSegmenter
        |                           |                           |
        v                           v                           v
  noise_weights            component_outputs              seg_logits
  {speckle, banding,       {4 denoised imgs}            {5-layer probs}
   gaussian, shot}
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                                    v
                     LayerAwareNSNDFusion
                     - Cross-attention on component outputs
                     - Layer-specific noise weight modulation
                     - FiLM conditioning on noise + layer info
                                    |
                                    v
                             Denoised Output
                             Uncertainty Map
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List

# Add NSND to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))

from nsnd.models.symbolic_analyzer import SymbolicNoiseAnalyzer, NeuroSymbolicNoiseAnalyzer
from nsnd.models.component_denoisers import ComponentDenoiserBank
from nsnd.symbolic.rules import SymbolicReasoningEngine


# =============================================================================
# Lightweight Layer Segmenter (from train_multitask.py)
# =============================================================================

class LightweightLayerSegmenter(nn.Module):
    """
    Lightweight 5-layer retinal segmenter.

    Layers: RNFL_GCL, INL_OPL, ONL, IS_OS, RPE_Choroid
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        # Simple encoder-decoder with skip connections
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # [B, 32, H, W]
        e2 = self.enc2(e1)     # [B, 64, H/2, W/2]
        e3 = self.enc3(e2)     # [B, 128, H/4, W/4]

        # Decoder with skip connections
        d2 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 64, H/2, W/2]

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # [B, 32, H, W]

        logits = self.final(d1)  # [B, num_classes, H, W]
        return logits


# =============================================================================
# Layer-Aware NSND Fusion Network (KEY NOVEL CONTRIBUTION)
# =============================================================================

class LayerAwareNSNDFusion(nn.Module):
    """
    Layer-Aware NSND Fusion Network (Simplified & Robust)

    KEY TMI CONTRIBUTION: Different retinal layers have different noise characteristics:
    - RNFL: Thin, needs edge preservation, sensitive to speckle
    - ONL: Thicker, more tolerant of smoothing
    - RPE: High contrast boundary, needs careful banding removal

    This module learns layer-specific component weights, combining:
    1. Global noise type (from symbolic analyzer)
    2. Local layer identity (from segmentation)

    DESIGN: Uses weighted sum of component images with learned refinement.
    This is simpler and more robust than encoding-decoding features.

    CLINICAL PRIORS (from ophthalmology domain knowledge):
    - Components: [speckle, banding, gaussian, shot]
    - RNFL_GCL: Thin layer, edges critical → prefer speckle (edge-preserving)
    - INL_OPL: Medium layer → balanced
    - ONL: Thick layer → can tolerate gaussian smoothing
    - IS_OS: Thin junction, high contrast → prefer speckle
    - RPE_Choroid: Boundary layer, banding artifacts common → prefer banding removal
    """

    # Clinical importance weights (from ophthalmology literature)
    CLINICAL_WEIGHTS = {
        0: 1.5,  # RNFL_GCL - critical for glaucoma
        1: 1.0,  # INL_OPL - standard
        2: 0.8,  # ONL - thicker, more tolerant
        3: 1.3,  # IS_OS - visual acuity marker
        4: 1.2,  # RPE_Choroid - AMD detection
    }

    # Layer-specific denoiser preferences (domain knowledge)
    # Rows: layers (RNFL, INL, ONL, IS_OS, RPE)
    # Cols: components (speckle, banding, gaussian, shot)
    LAYER_DENOISER_PRIORS = torch.tensor([
        [0.35, 0.15, 0.25, 0.25],  # RNFL: prefer speckle (edge-preserving for thin layer)
        [0.25, 0.25, 0.25, 0.25],  # INL: balanced
        [0.20, 0.20, 0.35, 0.25],  # ONL: can use more gaussian (thick layer)
        [0.35, 0.15, 0.25, 0.25],  # IS_OS: prefer speckle (thin junction)
        [0.20, 0.35, 0.25, 0.20],  # RPE: prefer banding removal (common artifact)
    ])

    def __init__(
        self,
        n_components: int = 4,
        n_layers: int = 5,
        feature_channels: int = 64,
        use_uncertainty: bool = True,
        use_clinical_priors: bool = True,
    ):
        super().__init__()

        self.n_components = n_components
        self.n_layers = n_layers
        self.feature_channels = feature_channels
        self.use_uncertainty = use_uncertainty

        # Register clinical weights as buffer
        clinical_weights = torch.tensor([self.CLINICAL_WEIGHTS[i] for i in range(n_layers)])
        self.register_buffer('clinical_weights', clinical_weights)

        # 1. Layer-specific noise weight modulation
        # Initialize with clinical priors if enabled, else uniform
        if use_clinical_priors:
            init_modulation = self.LAYER_DENOISER_PRIORS.clone()
        else:
            init_modulation = torch.ones(n_layers, n_components) / n_components

        self.layer_noise_modulation = nn.Parameter(init_modulation)

        # 2. Small refinement network (residual)
        # Takes weighted sum + original components to learn small corrections
        self.refinement = nn.Sequential(
            nn.Conv2d(1 + n_components, 32, 3, padding=1),  # weighted_sum + 4 components
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh(),  # Output in [-1, 1], will be scaled
        )
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))

        # 3. Uncertainty head
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(1 + n_components + n_layers, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        component_outputs: Dict[str, torch.Tensor],
        noise_weights: Dict[str, torch.Tensor],
        seg_probs: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Layer-aware fusion of component-denoised images.

        Args:
            component_outputs: Dict {component_name: denoised_img [B,1,H,W]}
            noise_weights: Dict {component_name: weight [B]}
            seg_probs: Layer probabilities [B, n_layers, H, W]
            confidence: Noise analysis confidence [B]

        Returns:
            output: Fused denoised image [B, 1, H, W]
            uncertainty: Uncertainty map [B, 1, H, W] (if enabled)
        """
        component_names = ['speckle', 'banding', 'gaussian', 'shot']
        B = seg_probs.shape[0]
        H, W = seg_probs.shape[2], seg_probs.shape[3]

        # Stack component outputs (images, not features)
        component_stack = torch.stack([
            component_outputs[name] for name in component_names
        ], dim=1)  # [B, 4, 1, H, W]
        component_stack = component_stack.squeeze(2)  # [B, 4, H, W]

        weight_stack = torch.stack([
            noise_weights[name] for name in component_names
        ], dim=1)  # [B, 4]

        # 1. Compute layer-modulated noise weights (KEY CONTRIBUTION)
        # layer_noise_modulation: [5, 4] - how each layer prefers each noise component
        # seg_probs: [B, 5, H, W] - layer probability at each pixel

        # Normalize layer_noise_modulation per layer (softmax)
        layer_mod_normalized = F.softmax(self.layer_noise_modulation, dim=1)  # [5, 4]

        # Apply clinical importance weights to layer modulation
        # This makes critical layers (RNFL, IS_OS) have stronger influence
        clinical_scaled_mod = layer_mod_normalized * self.clinical_weights.view(-1, 1)  # [5, 4]

        # Compute per-pixel weights: weighted sum of layer modulations by layer probability
        # seg_probs: [B, 5, H, W] -> [B, H, W, 5]
        seg_probs_hwl = seg_probs.permute(0, 2, 3, 1)  # [B, H, W, 5]

        # Per-pixel component weights = seg_probs @ clinical_scaled_mod
        # [B, H, W, 5] @ [5, 4] -> [B, H, W, 4]
        local_component_weights = torch.matmul(seg_probs_hwl, clinical_scaled_mod)
        local_component_weights = local_component_weights.permute(0, 3, 1, 2)  # [B, 4, H, W]

        # Combine global noise weights with local layer-aware weights
        global_weights = weight_stack.view(B, 4, 1, 1).expand(-1, -1, H, W)
        combined_weights = global_weights * local_component_weights
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 2. Weighted sum of component images (SIMPLE & ROBUST)
        # component_stack: [B, 4, H, W], combined_weights: [B, 4, H, W]
        weighted_sum = (component_stack * combined_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # 3. Small learned refinement (residual)
        # Input: weighted_sum + all component images
        refine_input = torch.cat([weighted_sum, component_stack], dim=1)  # [B, 5, H, W]
        refinement = self.refinement(refine_input) * self.refinement_scale

        # 4. Final output = weighted_sum + small refinement
        output = weighted_sum + refinement
        output = torch.clamp(output, 0, 1)

        # 5. Uncertainty estimation
        uncertainty = None
        if self.use_uncertainty:
            unc_input = torch.cat([output, component_stack, seg_probs], dim=1)  # [B, 1+4+5, H, W]
            uncertainty = self.uncertainty_head(unc_input)

        return output, uncertainty


# =============================================================================
# NSND MultiTask Denoiser (Main Model)
# =============================================================================

class NSNDMultiTaskDenoiser(nn.Module):
    """
    NSND-MultiTask: Neuro-Symbolic Noise Decomposition with Layer-Aware Denoising

    KEY TMI CONTRIBUTIONS:
    1. Interpretable noise classification via soft-logic rules
    2. Physics-based component denoisers (speckle, banding, Gaussian, shot)
    3. Layer-aware fusion that adapts to retinal anatomy
    4. Joint training of segmentation + noise-aware denoising

    Pipeline:
    1. SymbolicNoiseAnalyzer: Classify noise type with interpretable rules
    2. ComponentDenoiserBank: Apply physics-based denoisers
    3. LightweightLayerSegmenter: Predict retinal layer structure
    4. LayerAwareNSNDFusion: Fuse components with layer guidance
    """

    def __init__(
        self,
        # Symbolic analyzer params
        window_size: int = 7,
        kurtosis_window: int = 15,
        use_neuro_symbolic: bool = True,

        # Component denoiser params
        speckle_iterations: int = 10,
        banding_adaptive: bool = True,
        gaussian_depth: int = 5,
        gaussian_type: str = "dncnn",  # "dncnn" or "nafnet"

        # Fusion params
        feature_channels: int = 64,
        use_uncertainty: bool = True,

        # Segmenter params
        num_classes: int = 5,
        segmenter_ckpt: str = None,

        # Device
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()

        self.device = device
        self.use_uncertainty = use_uncertainty
        self.num_classes = num_classes

        # 1. Symbolic Noise Analyzer
        if use_neuro_symbolic:
            self.symbolic_analyzer = NeuroSymbolicNoiseAnalyzer(
                window_size=window_size,
                kurtosis_window=kurtosis_window,
                learnable_features=True,
                use_neural_predicates=False,
                use_neural_weights=True,
            )
        else:
            self.symbolic_analyzer = SymbolicNoiseAnalyzer(
                window_size=window_size,
                kurtosis_window=kurtosis_window,
                learnable_features=False,
            )

        # 2. Component Denoiser Bank
        self.denoiser_bank = ComponentDenoiserBank(
            speckle_iterations=speckle_iterations,
            banding_adaptive=banding_adaptive,
            gaussian_depth=gaussian_depth,
            gaussian_type=gaussian_type,
            device=device,
        )

        # 3. Layer Segmenter
        self.segmenter = LightweightLayerSegmenter(num_classes=num_classes)

        if segmenter_ckpt and os.path.exists(segmenter_ckpt):
            ckpt = torch.load(segmenter_ckpt, map_location='cpu', weights_only=False)
            self.segmenter.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            print(f"[NSNDMultiTask] Loaded segmenter from {segmenter_ckpt}")

        # 4. Layer-Aware NSND Fusion
        self.fusion = LayerAwareNSNDFusion(
            n_components=4,
            n_layers=num_classes,
            feature_channels=feature_channels,
            use_uncertainty=use_uncertainty,
        )

        # 5. Residual refinement (optional, for fine details)
        self.refinement = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # input + initial_output
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Full NSND-MultiTask pipeline.

        Args:
            x: Noisy input [B, 1, H, W]
            return_intermediates: If True, return all intermediate outputs

        Returns:
            denoised: Denoised output [B, 1, H, W]
            seg_logits: Segmentation logits [B, 5, H, W]
            intermediates: Dict of intermediate outputs (if requested)
        """
        B, C, H, W = x.shape

        # 1. Symbolic noise analysis
        noise_weights, noise_features = self.symbolic_analyzer(x)
        confidence = noise_weights.pop('_confidence', torch.ones(B, device=x.device))

        # 2. Apply component denoisers
        component_outputs = self.denoiser_bank.denoise_all(x)

        # 3. Layer segmentation
        seg_logits = self.segmenter(x)
        seg_probs = F.softmax(seg_logits, dim=1)

        # 4. Layer-aware fusion
        fused_output, uncertainty = self.fusion(
            component_outputs,
            noise_weights,
            seg_probs,
            confidence,
        )

        # 5. Residual refinement
        refine_input = torch.cat([x, fused_output], dim=1)
        refinement = self.refinement(refine_input) * self.refinement_scale
        denoised = fused_output + refinement
        denoised = torch.clamp(denoised, 0, 1)

        # Collect intermediates
        intermediates = None
        if return_intermediates:
            intermediates = {
                'noise_weights': noise_weights,
                'noise_features': noise_features,
                'confidence': confidence,
                'component_outputs': component_outputs,
                'seg_probs': seg_probs,
                'fused_output': fused_output,
                'uncertainty': uncertainty,
                'refinement': refinement,
            }

        return denoised, seg_logits, intermediates

    def get_noise_analysis_report(self, x: torch.Tensor) -> str:
        """
        Generate human-readable noise analysis report.

        Args:
            x: Input noisy image [B, 1, H, W]

        Returns:
            report: Text report explaining detected noise
        """
        noise_weights, _ = self.symbolic_analyzer(x)
        confidence = noise_weights.pop('_confidence', torch.tensor(0.5))

        report = "=" * 50 + "\n"
        report += "NSND-OCT Noise Analysis Report\n"
        report += "=" * 50 + "\n\n"
        report += "Detected Noise Composition:\n"

        for component in ['speckle', 'banding', 'gaussian', 'shot']:
            weight = noise_weights[component].mean().item() * 100
            report += f"  - {component.capitalize()}: {weight:.1f}%\n"

        report += f"\nAnalysis Confidence: {confidence.mean().item() * 100:.1f}%\n"
        report += "=" * 50

        return report

    def freeze_symbolic_and_denoisers(self):
        """Freeze symbolic analyzer and component denoisers, train only fusion."""
        for param in self.symbolic_analyzer.parameters():
            param.requires_grad = False
        for param in self.denoiser_bank.parameters():
            param.requires_grad = False
        print("[NSNDMultiTask] Frozen symbolic analyzer and component denoisers")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("[NSNDMultiTask] Unfrozen all parameters")


# =============================================================================
# NSND-Specific Losses
# =============================================================================

class NSNDConsistencyLoss(nn.Module):
    """
    Consistency loss for NSND: ensure noise weights are consistent with residual.

    If the model predicts high speckle weight, the denoised-clean difference
    should have speckle-like characteristics.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        noise_weights: Dict[str, torch.Tensor],
        denoised: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency between predicted noise weights and actual residual.

        Args:
            noise_weights: Dict {component: weight [B]}
            denoised: Denoised output [B, 1, H, W]
            clean: Clean target [B, 1, H, W]
            noisy: Noisy input [B, 1, H, W]

        Returns:
            loss: Consistency loss
        """
        # Residual = noisy - clean (actual noise)
        actual_noise = noisy - clean

        # Predicted noise = noisy - denoised
        predicted_noise = noisy - denoised

        # Basic consistency: predicted noise should approximate actual noise
        recon_loss = F.l1_loss(predicted_noise, actual_noise)

        # Weight regularization: encourage peaky distributions (clear noise type)
        weight_stack = torch.stack([
            noise_weights['speckle'],
            noise_weights['banding'],
            noise_weights['gaussian'],
            noise_weights['shot'],
        ], dim=1)  # [B, 4]

        # Entropy regularization (lower entropy = more confident classification)
        entropy = -(weight_stack * (weight_stack + 1e-8).log()).sum(dim=1).mean()

        return recon_loss + 0.1 * entropy


class NSNDComponentLoss(nn.Module):
    """
    Component-wise loss: ensure each component denoiser is doing its job.

    The Gaussian denoiser should handle Gaussian noise better than speckle denoiser.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        component_outputs: Dict[str, torch.Tensor],
        noise_weights: Dict[str, torch.Tensor],
        clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted component loss based on noise weights.

        Args:
            component_outputs: Dict {component: denoised [B,1,H,W]}
            noise_weights: Dict {component: weight [B]}
            clean: Clean target [B, 1, H, W]

        Returns:
            loss: Weighted component loss
        """
        total_loss = 0.0

        for component in ['speckle', 'banding', 'gaussian', 'shot']:
            output = component_outputs[component]
            weight = noise_weights[component]

            # Per-component reconstruction loss, weighted by noise weight
            # Components with higher weight should have lower loss
            component_loss = F.l1_loss(output, clean, reduction='none')
            component_loss = component_loss.mean(dim=(1, 2, 3))  # [B]

            # Weight the loss: high-weight components contribute more
            weighted_loss = (weight * component_loss).mean()
            total_loss = total_loss + weighted_loss

        return total_loss


class NSNDLayerAwareLoss(nn.Module):
    """
    Layer-aware loss: different layers should have different denoising quality.

    Clinical importance weights ensure critical layers (RNFL, IS/OS) are prioritized.
    """

    CLINICAL_WEIGHTS = {
        0: 1.5,  # RNFL_GCL - critical for glaucoma
        1: 1.0,  # INL_OPL
        2: 0.8,  # ONL - thicker, more tolerant
        3: 1.3,  # IS_OS - visual acuity
        4: 1.2,  # RPE_Choroid - AMD
    }

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        weights = torch.tensor([self.CLINICAL_WEIGHTS[i] for i in range(num_classes)])
        weights = weights / weights.mean()  # Normalize to mean=1
        self.register_buffer('layer_weights', weights)

    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute layer-weighted reconstruction loss.

        Args:
            denoised: Denoised output [B, 1, H, W]
            clean: Clean target [B, 1, H, W]
            seg_mask: Layer segmentation [B, H, W] with values 0-4

        Returns:
            loss: Layer-weighted MSE loss
        """
        # Per-pixel weight from layer
        weight_map = self.layer_weights[seg_mask]  # [B, H, W]
        weight_map = weight_map.unsqueeze(1)  # [B, 1, H, W]

        # Weighted MSE
        squared_error = (denoised - clean) ** 2
        weighted_error = weight_map * squared_error

        loss = weighted_error.sum() / (weight_map.sum() + 1e-8)

        return loss


# =============================================================================
# Helper: Model Summary
# =============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable parameters by module."""
    counts = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts[name] = params
    counts['total'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def test_nsnd_multitask():
    """Quick test of the model."""
    print("Testing NSNDMultiTaskDenoiser...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NSNDMultiTaskDenoiser(device=device).to(device)

    # Print parameter counts
    counts = count_parameters(model)
    print("\nParameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    # Test forward pass
    x = torch.randn(2, 1, 256, 256).to(device)

    with torch.no_grad():
        denoised, seg_logits, intermediates = model(x, return_intermediates=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Denoised shape: {denoised.shape}")
    print(f"Seg logits shape: {seg_logits.shape}")

    if intermediates:
        print("\nIntermediate outputs:")
        for key, value in intermediates.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")

    # Test noise analysis report
    print("\n" + model.get_noise_analysis_report(x))

    print("\nTest passed!")


if __name__ == "__main__":
    test_nsnd_multitask()
