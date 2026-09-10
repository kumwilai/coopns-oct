"""
NSND Model: Complete neuro-symbolic noise decomposition pipeline

Integrates:
1. Symbolic Noise Analyzer
2. Component Denoisers
3. Neural Fusion Network
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .symbolic_analyzer import SymbolicNoiseAnalyzer, NeuroSymbolicNoiseAnalyzer
from .component_denoisers import ComponentDenoiserBank
from .fusion_network import NeuralFusionNetwork, SimpleFusionNetwork


class NSNDModel(nn.Module):
    """
    Complete NSND-OCT denoising pipeline

    Pipeline:
    1. Analyze noise composition (symbolic analyzer)
    2. Denoise each component (physics-based denoisers)
    3. Fuse results (neural network)
    """

    def __init__(
        self,
        # Symbolic analyzer params
        window_size: int = 7,
        kurtosis_window: int = 15,
        learnable_features: bool = False,
        use_neuro_symbolic: bool = False,
        neuro_symbolic_analyzer: Optional[nn.Module] = None,

        # Component denoiser params
        speckle_iterations: int = 10,
        banding_adaptive: bool = True,
        gaussian_depth: int = 5,  # Changed from 7 to 5 to match checkpoint

        # Fusion network params
        fusion_type: str = 'neural',  # 'neural' or 'simple'
        feature_channels: int = 64,
        attention_heads: int = 4,
        use_uncertainty: bool = True,

        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            window_size: Window size for local statistics
            kurtosis_window: Window size for kurtosis computation
            learnable_features: Whether to use learnable feature refinement
            use_neuro_symbolic: If True, use neuro-symbolic analyzer with learnable predicates
            neuro_symbolic_analyzer: Optional external analyzer override
            speckle_iterations: Number of anisotropic diffusion iterations
            banding_adaptive: Whether to adaptively detect banding frequencies
            gaussian_depth: Depth of Gaussian denoiser network
            fusion_type: 'neural' (learned attention) or 'simple' (weighted sum)
            feature_channels: Feature dimension for fusion network
            attention_heads: Number of attention heads
            use_uncertainty: Whether to predict uncertainty maps
            device: Device to run on
        """
        super().__init__()

        self.device = device
        self.fusion_type = fusion_type
        self.use_uncertainty = use_uncertainty
        self.use_neuro_symbolic = bool(use_neuro_symbolic)

        # Module 1: Symbolic Noise Analyzer
        if neuro_symbolic_analyzer is not None:
            self.symbolic_analyzer = neuro_symbolic_analyzer.to(device)
        elif self.use_neuro_symbolic:
            self.symbolic_analyzer = NeuroSymbolicNoiseAnalyzer(
                window_size=window_size,
                kurtosis_window=kurtosis_window,
                learnable_features=learnable_features,
            ).to(device)
        else:
            self.symbolic_analyzer = SymbolicNoiseAnalyzer(
                window_size=window_size,
                kurtosis_window=kurtosis_window,
                learnable_features=learnable_features,
            ).to(device)

        # Module 2: Component Denoisers
        self.denoiser_bank = ComponentDenoiserBank(
            speckle_iterations=speckle_iterations,
            banding_adaptive=banding_adaptive,
            gaussian_depth=gaussian_depth,
            device=device,
        )

        # Module 3: Fusion Network
        if fusion_type == 'neural':
            self.fusion_network = NeuralFusionNetwork(
                n_components=4,
                feature_channels=feature_channels,
                attention_heads=attention_heads,
                use_uncertainty=use_uncertainty,
            ).to(device)
        elif fusion_type == 'simple':
            self.fusion_network = SimpleFusionNetwork().to(device)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Complete NSND pipeline

        Args:
            x: Input noisy image [B, 1, H, W]
            return_intermediates: If True, return all intermediate outputs

        Returns:
            output: Denoised image [B, 1, H, W]
            uncertainty: Uncertainty map [B, 1, H, W] (if enabled)
            intermediates: Dict of intermediate outputs (if requested)
        """
        B, C, H, W = x.shape
        assert C == 1, "Expecting single-channel images"

        # Step 1: Analyze noise composition
        symbolic_weights, features = self.symbolic_analyzer(x)

        # Extract confidence
        confidence = symbolic_weights.pop('_confidence')

        # Step 2: Apply component denoisers
        denoised_components = self.denoiser_bank.denoise_all(x)

        # Step 3: Fuse results
        if self.fusion_type == 'neural':
            output, uncertainty = self.fusion_network(
                denoised_components,
                symbolic_weights,
                confidence,
            )
        else:  # simple
            output = self.fusion_network(
                denoised_components,
                symbolic_weights,
            )
            uncertainty = None

        # Collect intermediates
        intermediates = None
        if return_intermediates:
            intermediates = {
                'symbolic_weights': symbolic_weights,
                'confidence': confidence,
                'features': features,
                'denoised_components': denoised_components,
            }

        return output, uncertainty, intermediates

    def analyze_noise(self, x: torch.Tensor) -> str:
        """
        Generate human-readable noise analysis report

        Args:
            x: Input noisy image [B, 1, H, W]

        Returns:
            report: Text report
        """
        return self.symbolic_analyzer.analyze_and_explain(x)

    def get_component_denoisers(self) -> ComponentDenoiserBank:
        """Access component denoisers for fine-tuning"""
        return self.denoiser_bank

    def freeze_components_unfreeze_fusion(self):
        """
        Freeze symbolic analyzer and component denoisers,
        only train fusion network
        """
        # Freeze analyzer
        for param in self.symbolic_analyzer.parameters():
            param.requires_grad = False

        # Freeze denoisers (only Gaussian denoiser has parameters)
        for denoiser in self.denoiser_bank.denoisers.values():
            for param in denoiser.parameters():
                param.requires_grad = False

        # Unfreeze fusion
        for param in self.fusion_network.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all trainable components"""
        for param in self.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NSNDModelWithTTA(NSNDModel):
    """
    NSND Model with Test-Time Adaptation

    Adds self-supervised refinement at inference time
    """

    def __init__(self, *args, tta_iterations: int = 5, tta_lr: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.tta_iterations = tta_iterations
        self.tta_lr = tta_lr

    def forward_with_tta(
        self,
        x: torch.Tensor,
        consistency_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with test-time adaptation

        Uses augmentation consistency:
        - Flip/rotate input
        - Denoise
        - Inverse transform
        - Enforce consistency across augmentations

        Args:
            x: Input noisy image [B, 1, H, W]
            consistency_weight: Weight for consistency loss

        Returns:
            output: Denoised image with TTA
            uncertainty: Uncertainty map
        """
        # Save original parameters
        original_state = {
            name: param.clone()
            for name, param in self.fusion_network.named_parameters()
            if param.requires_grad
        }

        # Optimizer for TTA
        optimizer = torch.optim.Adam(
            self.fusion_network.parameters(),
            lr=self.tta_lr
        )

        # TTA iterations
        for _ in range(self.tta_iterations):
            # Generate augmentations
            x_flip_h = torch.flip(x, dims=[-2])
            x_flip_v = torch.flip(x, dims=[-1])
            x_rot90 = torch.rot90(x, k=1, dims=[-2, -1])

            # Denoise each augmentation
            out_orig, _, _ = self.forward(x)
            out_flip_h, _, _ = self.forward(x_flip_h)
            out_flip_v, _, _ = self.forward(x_flip_v)
            out_rot90, _, _ = self.forward(x_rot90)

            # Inverse transforms
            out_flip_h_inv = torch.flip(out_flip_h, dims=[-2])
            out_flip_v_inv = torch.flip(out_flip_v, dims=[-1])
            out_rot90_inv = torch.rot90(out_rot90, k=-1, dims=[-2, -1])

            # Consistency loss
            consistency_loss = (
                torch.mean((out_orig - out_flip_h_inv) ** 2) +
                torch.mean((out_orig - out_flip_v_inv) ** 2) +
                torch.mean((out_orig - out_rot90_inv) ** 2)
            ) / 3.0

            # Total variation regularization
            tv_loss = self._total_variation_loss(out_orig)

            # Total loss
            loss = consistency_weight * consistency_loss + 0.01 * tv_loss

            # Update fusion network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final forward pass
        output, uncertainty, _ = self.forward(x)

        # Restore original parameters (optional: keep adapted parameters)
        # for name, param in self.fusion_network.named_parameters():
        #     if name in original_state:
        #         param.data = original_state[name]

        return output, uncertainty

    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation regularization"""
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))
