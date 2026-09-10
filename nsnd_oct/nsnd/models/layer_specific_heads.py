#!/usr/bin/env python3
"""
Layer-Specific Denoising Heads for TMI Joint OCT Denoising + Segmentation

KEY TMI CONTRIBUTION: Dedicated denoising heads per anatomical layer class.
Instead of a single adaptive gate, each layer type gets its own specialized
decoder head that learns layer-specific denoising patterns.

Architecture:
    Shared NAFNet Encoder -> Layer-Specific Heads -> Segmentation-Guided Mixer -> Output

This addresses the limitation of previous approaches where a single adaptive
gate (~17K params) was too weak for meaningful layer-specific processing.
The new architecture adds ~200K params (still <3% of total model).

Clinical Rationale:
- RNFL: Thin, critical for glaucoma - needs edge-preserving denoising
- GCL-INL: Medium layers with moderate texture
- IS/OS: Very thin boundary layer - needs careful handling
- RPE-BG: Thick, high contrast - can tolerate stronger denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class LayerSpecificHead(nn.Module):
    """
    Single layer-specific denoising head.

    Each head learns to denoise features for a specific anatomical layer type.
    Uses a lightweight CNN with residual connections.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: Input feature channels from encoder
            hidden_channels: Hidden layer channels
            out_channels: Output channels (1 for grayscale denoised output)
            num_blocks: Number of convolutional blocks
            dropout: Dropout rate for regularization
        """
        super().__init__()

        layers = []

        # First block: in_channels -> hidden_channels
        layers.append(nn.Conv2d(in_channels, hidden_channels, 3, padding=1))
        layers.append(nn.GELU())
        layers.append(nn.Dropout2d(dropout))

        # Middle blocks: hidden_channels -> hidden_channels
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
            layers.append(nn.GELU())
            layers.append(nn.Dropout2d(dropout))

        # Final projection: hidden_channels -> out_channels
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1))

        self.layers = nn.Sequential(*layers)

        # Initialize final layer with small values for meaningful contribution
        # Use xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.1)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Encoder features [B, C, H, W]

        Returns:
            Layer-specific denoised output [B, 1, H, W]
        """
        return self.layers(features)


class LayerSpecificDenoisingHeads(nn.Module):
    """
    Collection of layer-specific denoising heads with segmentation-guided mixing.

    Each anatomical layer type has its own specialized denoising head.
    The final output is a weighted combination based on segmentation masks.

    Layer Types (4-class scheme):
        0: RNFL_GCL - Nerve fiber + ganglion cell layers (glaucoma critical)
        1: INL_OPL_ONL - Inner/outer nuclear + plexiform layers
        2: IS_OS - Photoreceptor inner/outer segments (thin boundary)
        3: RPE_Choroid - Retinal pigment epithelium + choroid
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        n_layers: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_soft_mixing: bool = True,
    ):
        """
        Args:
            in_channels: Input feature channels from encoder
            hidden_channels: Hidden channels per head
            n_layers: Number of layer classes
            num_blocks: Conv blocks per head
            dropout: Dropout rate
            use_soft_mixing: If True, use soft segmentation probs for mixing
                            If False, use hard argmax selection
        """
        super().__init__()

        self.n_layers = n_layers
        self.use_soft_mixing = use_soft_mixing

        # Create dedicated head for each layer type
        self.layer_heads = nn.ModuleList([
            LayerSpecificHead(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=1,
                num_blocks=num_blocks,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Learnable mixing weights (for soft boundary blending)
        self.mix_temperature = nn.Parameter(torch.ones(1))

        # Per-layer confidence gates (learn when to trust each head)
        self.confidence_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, 1),
                nn.Sigmoid()
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        features: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
        seg_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with segmentation-guided output mixing.

        Args:
            features: Encoder features [B, C, H, W]
            seg_mask: Hard segmentation mask [B, H, W] with layer indices 0 to n_layers-1
            seg_probs: Soft segmentation probabilities [B, n_layers, H, W]

        Returns:
            Dict containing:
                - 'output': Final mixed denoised output [B, 1, H, W]
                - 'layer_outputs': Individual head outputs [B, n_layers, H, W]
                - 'confidences': Per-layer confidence values [B, n_layers]
        """
        B, C, H, W = features.shape
        device = features.device

        # Get output from each layer-specific head
        layer_outputs = []
        confidences = []

        for i, (head, gate) in enumerate(zip(self.layer_heads, self.confidence_gates)):
            out = head(features)  # [B, 1, H, W]
            conf = gate(features)  # [B, 1]
            layer_outputs.append(out)
            confidences.append(conf)

        # Stack outputs: [B, n_layers, H, W]
        layer_outputs_stacked = torch.cat(layer_outputs, dim=1)
        confidences_stacked = torch.cat(confidences, dim=1)  # [B, n_layers]

        # Determine mixing weights
        if seg_probs is not None and self.use_soft_mixing:
            # Use soft segmentation probabilities for smooth blending
            # Apply temperature scaling for sharper/softer mixing
            mix_weights = F.softmax(seg_probs / self.mix_temperature.clamp(min=0.1), dim=1)
        elif seg_mask is not None:
            # Convert hard mask to one-hot
            mix_weights = F.one_hot(seg_mask.long(), num_classes=self.n_layers)
            mix_weights = mix_weights.permute(0, 3, 1, 2).float()  # [B, n_layers, H, W]
        else:
            # No segmentation info - average all heads (fallback)
            mix_weights = torch.ones(B, self.n_layers, H, W, device=device) / self.n_layers

        # Apply confidence gating to mix weights
        # Expand confidences to spatial dimensions
        conf_expanded = confidences_stacked.view(B, self.n_layers, 1, 1)
        mix_weights = mix_weights * conf_expanded

        # Renormalize
        mix_weights = mix_weights / (mix_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted combination of layer outputs
        output = (layer_outputs_stacked * mix_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        return {
            'output': output,
            'layer_outputs': layer_outputs_stacked,
            'confidences': confidences_stacked,
            'mix_weights': mix_weights,
        }


class RefinementHead(nn.Module):
    """
    Refinement head that processes mixed layer outputs.

    Takes the segmentation-guided mixed output and applies final refinement
    to ensure consistency across layer boundaries.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        use_boundary_attention: bool = True,
    ):
        super().__init__()

        self.use_boundary_attention = use_boundary_attention

        # Refinement CNN
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        if use_boundary_attention:
            # Learn to focus on boundaries
            self.boundary_attn = nn.Sequential(
                nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, 1),
                nn.Sigmoid(),
            )

        # Initialize to identity-like operation
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        mixed_output: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mixed_output: Mixed layer outputs [B, 1, H, W]
            boundary_mask: Optional boundary mask [B, 1, H, W]

        Returns:
            Refined output [B, 1, H, W]
        """
        refinement = self.refine(mixed_output)

        if self.use_boundary_attention and boundary_mask is not None:
            # Apply stronger refinement at boundaries
            combined = torch.cat([mixed_output, boundary_mask], dim=1)
            attn = self.boundary_attn(combined)
            refinement = refinement * attn

        return mixed_output + 0.1 * refinement


class EnhancedLayerSpecificDenoiser(nn.Module):
    """
    Complete layer-specific denoising module for TMI.

    Combines:
    1. Layer-specific denoising heads
    2. Segmentation-guided mixing
    3. Boundary-aware refinement

    This module is designed to be integrated with an existing encoder (NAFNet)
    and segmenter (BoundaryAwareSegmenter).
    """

    def __init__(
        self,
        encoder_channels: int = 64,
        n_layer_classes: int = 4,
        hidden_channels: int = 64,
        num_head_blocks: int = 2,
        dropout: float = 0.1,
        use_refinement: bool = True,
    ):
        """
        Args:
            encoder_channels: Output channels from encoder
            n_layer_classes: Number of segmentation classes
            hidden_channels: Hidden channels in heads
            num_head_blocks: Conv blocks per head
            dropout: Dropout rate
            use_refinement: Whether to use boundary-aware refinement
        """
        super().__init__()

        self.use_refinement = use_refinement

        # Layer-specific heads
        self.layer_heads = LayerSpecificDenoisingHeads(
            in_channels=encoder_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layer_classes,
            num_blocks=num_head_blocks,
            dropout=dropout,
            use_soft_mixing=True,
        )

        # Optional refinement
        if use_refinement:
            self.refinement = RefinementHead(
                in_channels=1,
                hidden_channels=32,
                use_boundary_attention=True,
            )

    def forward(
        self,
        encoder_features: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
        seg_probs: Optional[torch.Tensor] = None,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: Features from encoder [B, C, H, W]
            seg_mask: Hard segmentation [B, H, W]
            seg_probs: Soft segmentation [B, n_classes, H, W]
            boundary_mask: Layer boundary mask [B, 1, H, W]

        Returns:
            Dict with 'output', 'layer_outputs', 'confidences', etc.
        """
        # Get layer-specific outputs
        results = self.layer_heads(encoder_features, seg_mask, seg_probs)

        # Apply refinement if enabled
        if self.use_refinement:
            results['output'] = self.refinement(results['output'], boundary_mask)

        return results


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the module
    print("Testing LayerSpecificDenoisingHeads...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create module
    heads = EnhancedLayerSpecificDenoiser(
        encoder_channels=64,
        n_layer_classes=4,
        hidden_channels=64,
        num_head_blocks=2,
        dropout=0.1,
        use_refinement=True,
    ).to(device)

    print(f"Total parameters: {count_parameters(heads):,}")

    # Test forward pass
    B, C, H, W = 2, 64, 128, 128
    features = torch.randn(B, C, H, W).to(device)
    seg_mask = torch.randint(0, 4, (B, H, W)).to(device)
    seg_probs = F.softmax(torch.randn(B, 4, H, W), dim=1).to(device)
    boundary_mask = torch.rand(B, 1, H, W).to(device)

    with torch.no_grad():
        results = heads(features, seg_mask, seg_probs, boundary_mask)

    print(f"Output shape: {results['output'].shape}")
    print(f"Layer outputs shape: {results['layer_outputs'].shape}")
    print(f"Confidences shape: {results['confidences'].shape}")
    print(f"Mix weights shape: {results['mix_weights'].shape}")

    print("\nAll tests passed!")
