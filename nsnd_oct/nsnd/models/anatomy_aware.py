"""
Anatomy-Aware OCT Denoising Modules

This module provides anatomy-aware components for OCT image denoising,
leveraging the known layered structure of retinal OCT images.

Key Contributions:
1. Layer-Aware Noise Decomposition: Different retinal layers have different noise characteristics
2. Depth-Adaptive Processing: Top/middle/bottom regions processed differently
3. Structure-Preserving Constraints: Preserve layer boundaries during denoising
4. Anatomical Priors: Incorporate known OCT anatomy into noise modeling

OCT Retinal Layers (from vitreous to choroid):
- ILM (Internal Limiting Membrane)
- NFL (Nerve Fiber Layer) - High reflectivity, strong speckle
- GCL (Ganglion Cell Layer)
- IPL (Inner Plexiform Layer)
- INL (Inner Nuclear Layer) - Lower reflectivity
- OPL (Outer Plexiform Layer)
- ONL (Outer Nuclear Layer) - Dark region, high relative noise
- ELM (External Limiting Membrane)
- IS/OS (Inner/Outer Segment junction) - Bright band
- RPE (Retinal Pigment Epithelium) - Very bright, strong speckle
- Choroid - Complex structure, shadow artifacts

Reference: Huang et al., "Optical Coherence Tomography", Science 1991
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class RetinalLayerDetector(nn.Module):
    """
    Lightweight CNN to detect approximate retinal layer regions.

    Instead of precise segmentation, we detect broad anatomical zones
    that inform noise characteristics:
    - Zone 0: Vitreous/NFL (top) - bright, high speckle
    - Zone 1: Inner retina (GCL-INL) - moderate reflectivity
    - Zone 2: Outer nuclear (OPL-ONL) - darker, more gaussian noise
    - Zone 3: Photoreceptors (ELM-IS/OS) - bright bands
    - Zone 4: RPE/Choroid (bottom) - very bright, complex noise

    This is trained end-to-end with the denoiser, learning layer-like
    features that help noise decomposition.
    """

    def __init__(self, num_zones: int = 5, base_channels: int = 32):
        super().__init__()
        self.num_zones = num_zones

        # Encoder: extract features at multiple scales
        self.encoder = nn.Sequential(
            # Initial conv
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            # Deeper features
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            # Context aggregation
            nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
        )

        # Vertical context: OCT layers are horizontal, so we need vertical context
        self.vertical_context = nn.Sequential(
            # Tall kernel to capture vertical layer structure
            nn.Conv2d(base_channels, base_channels, (7, 3), padding=(3, 1)),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
        )

        # Zone classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_zones, 1),
        )

        # Depth prior: encourage zone ordering (top to bottom)
        self.register_buffer('depth_prior', self._create_depth_prior())

    def _create_depth_prior(self) -> torch.Tensor:
        """Create a soft prior that zones follow depth ordering."""
        # This will be applied during forward to bias predictions
        # Shape: [num_zones] - relative vertical position preference
        positions = torch.linspace(0, 1, self.num_zones)
        return positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect retinal layer zones.

        Args:
            x: Input OCT image [B, 1, H, W]

        Returns:
            zone_prob: Per-pixel zone probabilities [B, num_zones, H, W]
        """
        B, C, H, W = x.shape

        # Extract features
        features = self.encoder(x)
        vertical_features = self.vertical_context(features)

        # Combine
        combined = torch.cat([features, vertical_features], dim=1)

        # Classify zones
        logits = self.classifier(combined)  # [B, num_zones, H, W]

        # Add depth prior: bias predictions based on vertical position
        depth_positions = torch.linspace(0, 1, H, device=x.device)
        depth_positions = depth_positions.view(1, 1, H, 1).expand(B, self.num_zones, H, W)

        # Each zone prefers a certain depth range
        zone_centers = self.depth_prior.view(1, self.num_zones, 1, 1)
        depth_bias = -5.0 * (depth_positions - zone_centers).pow(2)  # Gaussian bias

        # Combine learned logits with depth prior
        logits = logits + depth_bias

        # Softmax to get probabilities
        zone_prob = F.softmax(logits, dim=1)

        return zone_prob


class LayerAwareNoiseDecomposer(nn.Module):
    """
    Decompose noise based on retinal layer context.

    Different retinal layers have characteristic noise properties:
    - NFL/RPE (bright layers): High speckle due to strong backscatter
    - ONL (dark layer): Higher relative noise, more Gaussian-like
    - IS/OS junction: Strong speckle from organized photoreceptors
    - Choroid: Complex noise from blood vessels and shadows

    This module learns a mapping from detected layer zones to
    noise type distributions, providing anatomically-informed priors
    for the noise decomposition.
    """

    def __init__(
        self,
        num_zones: int = 5,
        num_noise_types: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.num_zones = num_zones
        self.num_noise_types = num_noise_types

        # Layer detector
        self.layer_detector = RetinalLayerDetector(num_zones, base_channels)

        # Learnable layer-to-noise-type mapping
        # Initial values based on OCT physics:
        # Columns: [speckle, banding, gaussian, shot]
        # Rows: [vitreous/NFL, inner_retina, outer_nuclear, photoreceptors, RPE/choroid]
        initial_mapping = torch.tensor([
            [0.55, 0.10, 0.15, 0.20],  # Vitreous/NFL - high speckle (bright)
            [0.40, 0.10, 0.25, 0.25],  # Inner retina - moderate
            [0.25, 0.15, 0.35, 0.25],  # Outer nuclear - darker, more gaussian
            [0.50, 0.10, 0.15, 0.25],  # Photoreceptors - bright, speckle
            [0.55, 0.10, 0.10, 0.25],  # RPE/Choroid - very bright, speckle
        ])
        self.layer_noise_mapping = nn.Parameter(initial_mapping)

        # Refinement network: adjust prior based on local image content
        self.refine = nn.Sequential(
            nn.Conv2d(1 + num_zones, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_noise_types, 1),
        )

        # Mixing weight between prior and refined
        self.mix_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        return_layers: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute layer-aware noise type prior.

        Args:
            x: Input OCT image [B, 1, H, W]
            return_layers: If True, also return layer probabilities

        Returns:
            noise_prior: Layer-informed noise type prior [B, num_noise_types, H, W]
            layer_prob: (optional) Layer zone probabilities [B, num_zones, H, W]
        """
        # Detect layer zones
        layer_prob = self.layer_detector(x)  # [B, num_zones, H, W]

        # Compute prior from layer-noise mapping
        # layer_prob: [B, num_zones, H, W]
        # layer_noise_mapping: [num_zones, num_noise_types]
        mapping = F.softmax(self.layer_noise_mapping, dim=1)  # Normalize to valid distribution

        # Einstein summation: for each pixel, weighted sum of noise distributions
        prior = torch.einsum('bnhw,nt->bthw', layer_prob, mapping)  # [B, num_noise_types, H, W]

        # Refine based on local content
        refine_input = torch.cat([x, layer_prob], dim=1)
        refined = self.refine(refine_input)
        refined = F.softmax(refined, dim=1)

        # Mix prior and refined
        mix = torch.sigmoid(self.mix_weight)
        noise_prior = mix * prior + (1 - mix) * refined

        # Normalize
        noise_prior = noise_prior / (noise_prior.sum(dim=1, keepdim=True) + 1e-8)

        if return_layers:
            return noise_prior, layer_prob
        return noise_prior, None


class DepthAdaptiveProcessor(nn.Module):
    """
    Process different depth regions with different strategies.

    OCT images have a natural depth structure:
    - Top: Vitreous (dark) and inner retinal layers
    - Middle: Nuclear layers and plexiform layers
    - Bottom: Photoreceptors, RPE, and choroid

    Each region benefits from different processing weights
    for the symbolic experts.
    """

    def __init__(
        self,
        num_depth_zones: int = 5,
        num_noise_types: int = 4,
    ):
        super().__init__()
        self.num_zones = num_depth_zones
        self.num_noise_types = num_noise_types

        # Learnable depth-specific weights for each noise type
        # These modulate how strongly each expert is used at different depths
        self.depth_weights = nn.Parameter(torch.ones(num_depth_zones, num_noise_types))

        # Learnable zone boundaries (soft)
        # Instead of fixed zones, learn optimal boundaries
        zone_positions = torch.linspace(0, 1, num_depth_zones + 1)[1:-1]
        self.zone_boundaries = nn.Parameter(zone_positions)

    def forward(
        self,
        noise_type: torch.Tensor,
        height: int
    ) -> torch.Tensor:
        """
        Apply depth-adaptive weighting to noise type predictions.

        Args:
            noise_type: Noise type predictions [B, num_noise_types, H, W]
            height: Image height (for computing depth positions)

        Returns:
            weighted_noise_type: Depth-weighted noise types [B, num_noise_types, H, W]
        """
        B, N, H, W = noise_type.shape
        device = noise_type.device

        # Compute soft zone membership for each row
        depths = torch.linspace(0, 1, H, device=device)  # [H]

        # Compute distance to each zone boundary
        boundaries = torch.sigmoid(self.zone_boundaries)  # [num_zones-1], in (0, 1)
        boundaries = torch.cat([
            torch.zeros(1, device=device),
            boundaries,
            torch.ones(1, device=device)
        ])  # [num_zones+1]

        # Compute zone centers
        zone_centers = (boundaries[:-1] + boundaries[1:]) / 2  # [num_zones]
        zone_widths = boundaries[1:] - boundaries[:-1]  # [num_zones]

        # Soft zone membership using Gaussian
        depths_expanded = depths.view(1, H)  # [1, H]
        centers_expanded = zone_centers.view(self.num_zones, 1)  # [num_zones, H]
        widths_expanded = zone_widths.view(self.num_zones, 1).clamp(min=0.1)

        # Gaussian membership
        zone_membership = torch.exp(
            -0.5 * ((depths_expanded - centers_expanded) / widths_expanded) ** 2
        )  # [num_zones, H]
        zone_membership = zone_membership / (zone_membership.sum(dim=0, keepdim=True) + 1e-8)

        # Compute depth-adaptive weights
        # zone_membership: [num_zones, H]
        # depth_weights: [num_zones, num_noise_types]
        weights = F.softplus(self.depth_weights)  # Ensure positive

        # Weighted combination for each row
        depth_modulation = torch.einsum('zh,zn->nh', zone_membership, weights)  # [num_noise_types, H]
        depth_modulation = depth_modulation.view(1, N, H, 1)  # [1, N, H, 1]

        # Apply modulation
        weighted_noise_type = noise_type * depth_modulation

        # Re-normalize
        weighted_noise_type = weighted_noise_type / (
            weighted_noise_type.sum(dim=1, keepdim=True) + 1e-8
        )

        return weighted_noise_type


class AnatomyPreservingLoss(nn.Module):
    """
    Loss function that preserves OCT anatomical structures.

    Key principles:
    1. Layer boundaries should be preserved (horizontal edges in B-scans)
    2. Vertical continuity within layers should be maintained
    3. Bright layer structures (NFL, IS/OS, RPE) should remain bright
    4. Dark regions (ONL, vitreous) should remain dark

    This loss encourages the denoiser to respect anatomical structure
    while removing noise.
    """

    def __init__(
        self,
        lambda_edge: float = 0.1,
        lambda_structure: float = 0.05,
        lambda_intensity: float = 0.02,
    ):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_structure = lambda_structure
        self.lambda_intensity = lambda_intensity

        # Sobel filters for edge detection
        # Vertical edges (layer boundaries are horizontal, so we detect vertical gradients)
        sobel_y = torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0)

        # Horizontal edges (for within-layer continuity)
        sobel_x = torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0)

        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_x', sobel_x)

        # Laplacian for structure detection
        laplacian = torch.tensor([
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]]
        ], dtype=torch.float32).unsqueeze(0)
        self.register_buffer('laplacian', laplacian)

    def compute_edges(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute horizontal and vertical edges."""
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        return edge_y, edge_x

    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute anatomy-preserving loss.

        Args:
            denoised: Denoised output [B, 1, H, W]
            clean: Clean target [B, 1, H, W]
            noisy: Noisy input [B, 1, H, W]

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        losses = {}

        # 1. Primary reconstruction loss
        losses['recon'] = F.l1_loss(denoised, clean)

        # 2. Layer boundary preservation (vertical edges)
        edge_y_clean, edge_x_clean = self.compute_edges(clean)
        edge_y_denoised, edge_x_denoised = self.compute_edges(denoised)

        # Preserve strong vertical edges (layer boundaries)
        # Weight by edge strength in clean image
        edge_weight = torch.abs(edge_y_clean).detach()
        edge_weight = edge_weight / (edge_weight.max() + 1e-8)

        losses['edge_y'] = (edge_weight * (edge_y_denoised - edge_y_clean).abs()).mean()

        # 3. Horizontal continuity (within-layer smoothness)
        # In clean OCT, horizontal variations within layers are gradual
        losses['edge_x'] = F.l1_loss(edge_x_denoised, edge_x_clean)

        # 4. Structure preservation (Laplacian)
        structure_clean = F.conv2d(clean, self.laplacian, padding=1)
        structure_denoised = F.conv2d(denoised, self.laplacian, padding=1)
        losses['structure'] = F.l1_loss(structure_denoised, structure_clean)

        # 5. Intensity distribution preservation
        # Preserve local mean (avoid over-smoothing or artifacts)
        local_mean_clean = F.avg_pool2d(clean, 7, stride=1, padding=3)
        local_mean_denoised = F.avg_pool2d(denoised, 7, stride=1, padding=3)
        losses['intensity'] = F.l1_loss(local_mean_denoised, local_mean_clean)

        # Combine losses
        total = (
            losses['recon'] +
            self.lambda_edge * (losses['edge_y'] + 0.5 * losses['edge_x']) +
            self.lambda_structure * losses['structure'] +
            self.lambda_intensity * losses['intensity']
        )

        return total, losses


class AnatomyAwareFusion(nn.Module):
    """
    Fuse neural backbone output with symbolic expert outputs
    using anatomy-aware gating.

    The fusion weights depend on:
    1. Detected layer zones (different layers need different experts)
    2. Local noise characteristics
    3. Confidence of layer detection
    """

    def __init__(
        self,
        num_noise_types: int = 4,
        num_zones: int = 5,
    ):
        super().__init__()
        self.num_noise_types = num_noise_types
        self.num_zones = num_zones

        # Fusion gate: combines layer info, noise type, and uncertainty
        # Input: noise_type (4) + layer_prob (5) + uncertainty (1) = 10
        input_dim = num_noise_types + num_zones + 1

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # CRITICAL FIX: Initialize last conv to output low gate values
        # This ensures backbone output is used initially (gate~0 means more backbone)
        # As training progresses, the network learns when to trust symbolic experts
        nn.init.zeros_(self.fusion_gate[-2].weight)
        nn.init.constant_(self.fusion_gate[-2].bias, -4.0)  # sigmoid(-4) ≈ 0.018

        # Per-zone fusion bias (some layers benefit more from symbolic)
        self.zone_bias = nn.Parameter(torch.zeros(num_zones))

    def forward(
        self,
        backbone_out: torch.Tensor,
        symbolic_out: torch.Tensor,
        noise_type: torch.Tensor,
        layer_prob: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse backbone and symbolic outputs with anatomy awareness.

        Args:
            backbone_out: Neural backbone output [B, 1, H, W]
            symbolic_out: Symbolic expert output [B, 1, H, W]
            noise_type: Noise type predictions [B, 4, H, W]
            layer_prob: Layer zone probabilities [B, 5, H, W]
            uncertainty: Uncertainty estimates [B, 1, H, W]

        Returns:
            fused: Fused output [B, 1, H, W]
            gate: Fusion gate values [B, 1, H, W]
        """
        # Compute zone-specific bias
        zone_bias = torch.einsum('bzhw,z->bhw', layer_prob, torch.sigmoid(self.zone_bias))
        zone_bias = zone_bias.unsqueeze(1)  # [B, 1, H, W]

        # Concatenate inputs for gate
        gate_input = torch.cat([noise_type, layer_prob, uncertainty], dim=1)

        # Compute fusion gate
        gate = self.fusion_gate(gate_input)

        # Add zone bias
        gate = gate + 0.1 * zone_bias
        gate = torch.clamp(gate, 0, 1)

        # Fuse outputs
        # Higher gate = more symbolic, lower gate = more backbone
        fused = (1 - gate) * backbone_out + gate * symbolic_out

        return fused, gate


# =============================================================================
# Utility functions
# =============================================================================

def visualize_layer_detection(
    image: torch.Tensor,
    layer_prob: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize detected layer zones overlaid on OCT image.

    Args:
        image: OCT image [1, 1, H, W] or [H, W]
        layer_prob: Layer probabilities [1, num_zones, H, W]
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare image
    if image.dim() == 4:
        image = image.squeeze()
    image = image.cpu().numpy()

    # Get dominant zone per pixel
    if layer_prob.dim() == 4:
        layer_prob = layer_prob.squeeze(0)
    zones = layer_prob.argmax(dim=0).cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('OCT Image')
    axes[0].axis('off')

    # Layer zones
    axes[1].imshow(zones, cmap='viridis')
    axes[1].set_title('Detected Layer Zones')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(zones, cmap='viridis', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
