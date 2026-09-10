#!/usr/bin/env python3
"""
Segmentation-Guided Spatial Attention for TMI Joint OCT Denoising

KEY TMI CONTRIBUTION: Use segmentation information to guide denoising attention.
The model learns to focus denoising effort on:
1. Layer boundaries (clinically critical for thickness measurement)
2. Clinically important regions (RNFL, IS/OS)
3. Areas with high segmentation uncertainty

Architecture:
    Seg Logits -> Boundary Detector -> Attention Map
                                          |
    Denoising Features -----------------> Spatial Modulation -> Enhanced Features

Clinical Rationale:
- Layer boundaries are where thickness is measured (critical for glaucoma, AMD)
- Blurred boundaries lead to measurement errors
- Different layers need different denoising strengths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class BoundaryDetector(nn.Module):
    """
    Detect layer boundaries from segmentation logits/probabilities.

    Boundaries are detected as regions of high segmentation entropy
    (uncertainty about which class a pixel belongs to).
    """

    def __init__(
        self,
        n_classes: int = 4,
        hidden_dim: int = 32,
        use_gradient: bool = True,
    ):
        """
        Args:
            n_classes: Number of segmentation classes
            hidden_dim: Hidden dimension for boundary conv
            use_gradient: Also use gradient-based boundary detection
        """
        super().__init__()

        self.n_classes = n_classes
        self.use_gradient = use_gradient

        # Learned boundary detector from segmentation probs
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(n_classes, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        if use_gradient:
            # Sobel filters for gradient-based boundary detection
            sobel_y = torch.tensor([
                [-1., -2., -1.],
                [0., 0., 0.],
                [1., 2., 1.]
            ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0

            sobel_x = torch.tensor([
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]
            ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0

            self.register_buffer('sobel_y', sobel_y)
            self.register_buffer('sobel_x', sobel_x)

    def forward(
        self,
        seg_logits: Optional[torch.Tensor] = None,
        seg_probs: Optional[torch.Tensor] = None,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Detect boundaries from segmentation.

        Args:
            seg_logits: Raw segmentation logits [B, C, H, W]
            seg_probs: Segmentation probabilities [B, C, H, W]
            seg_mask: Hard segmentation mask [B, H, W]

        Returns:
            boundary_map: Boundary attention map [B, 1, H, W] in range [0, 1]
        """
        device = seg_logits.device if seg_logits is not None else \
                 seg_probs.device if seg_probs is not None else seg_mask.device

        # Get probabilities
        if seg_probs is None and seg_logits is not None:
            seg_probs = F.softmax(seg_logits, dim=1)
        elif seg_probs is None and seg_mask is not None:
            seg_probs = F.one_hot(seg_mask.long(), num_classes=self.n_classes)
            seg_probs = seg_probs.permute(0, 3, 1, 2).float()

        # Learned boundary detection
        boundary_learned = self.boundary_conv(seg_probs)

        # Entropy-based boundary (high entropy = boundary region)
        entropy = -(seg_probs * (seg_probs + 1e-8).log()).sum(dim=1, keepdim=True)
        max_entropy = np.log(self.n_classes)
        entropy_normalized = entropy / (max_entropy + 1e-8)

        # Gradient-based boundary from hard mask
        if self.use_gradient and seg_mask is not None:
            mask_float = seg_mask.unsqueeze(1).float()
            padded = F.pad(mask_float, (1, 1, 1, 1), mode='reflect')

            grad_y = torch.abs(F.conv2d(padded, self.sobel_y.to(device)))
            grad_x = torch.abs(F.conv2d(padded, self.sobel_x.to(device)))
            grad_boundary = (grad_y + grad_x).clamp(0, 1)

            # Combine all boundary signals
            boundary_map = 0.4 * boundary_learned + 0.3 * entropy_normalized + 0.3 * grad_boundary
        else:
            boundary_map = 0.6 * boundary_learned + 0.4 * entropy_normalized

        return boundary_map.clamp(0, 1)


class ClinicalImportanceMap(nn.Module):
    """
    Generate spatial attention based on clinical importance of regions.

    Some layers are more clinically important than others:
    - RNFL: Critical for glaucoma diagnosis
    - IS/OS: Critical for visual acuity assessment
    - RPE: Important for AMD

    This module creates a spatial map highlighting clinically important regions.
    """

    def __init__(
        self,
        n_classes: int = 4,
        clinical_weights: Optional[Dict[int, float]] = None,
    ):
        """
        Args:
            n_classes: Number of segmentation classes
            clinical_weights: Per-class clinical importance weights
                             Default: {0: 2.0, 1: 1.0, 2: 2.0, 3: 1.5}
                             (RNFL_GCL=2.0, INL=1.0, IS_OS=2.0, RPE=1.5)
        """
        super().__init__()

        self.n_classes = n_classes

        # Default clinical importance weights
        if clinical_weights is None:
            clinical_weights = {
                0: 2.0,   # RNFL_GCL - glaucoma critical
                1: 1.0,   # INL_OPL_ONL - moderate importance
                2: 2.0,   # IS_OS - visual acuity critical (thin layer needs extra care)
                3: 1.5,   # RPE_Choroid - AMD important
            }

        # Register as buffer
        weights = torch.tensor(
            [clinical_weights.get(i, 1.0) for i in range(n_classes)],
            dtype=torch.float32
        )
        # Normalize to mean=1
        weights = weights / weights.mean()
        self.register_buffer('clinical_weights', weights)

    def forward(
        self,
        seg_probs: Optional[torch.Tensor] = None,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate clinical importance map.

        Args:
            seg_probs: Segmentation probabilities [B, C, H, W]
            seg_mask: Hard segmentation mask [B, H, W]

        Returns:
            importance_map: [B, 1, H, W] with higher values for important regions
        """
        if seg_probs is not None:
            # Weighted sum using soft probabilities
            weights = self.clinical_weights.view(1, -1, 1, 1)  # [1, C, 1, 1]
            importance_map = (seg_probs * weights).sum(dim=1, keepdim=True)
        elif seg_mask is not None:
            # Direct lookup using hard mask
            importance_map = self.clinical_weights[seg_mask].unsqueeze(1)
        else:
            raise ValueError("Either seg_probs or seg_mask must be provided")

        return importance_map


class SegmentationGuidedAttention(nn.Module):
    """
    Segmentation-Guided Spatial Attention Module.

    Uses segmentation information to create spatial attention that:
    1. Focuses on layer boundaries
    2. Prioritizes clinically important regions
    3. Modulates denoising features accordingly

    This is the main module to be integrated into the denoising pipeline.
    """

    def __init__(
        self,
        n_classes: int = 4,
        feature_dim: int = 64,
        boundary_weight: float = 0.5,
        clinical_weight: float = 0.5,
    ):
        """
        Args:
            n_classes: Number of segmentation classes
            feature_dim: Feature dimension from denoising encoder
            boundary_weight: Weight for boundary attention component
            clinical_weight: Weight for clinical importance component
        """
        super().__init__()

        self.boundary_weight = boundary_weight
        self.clinical_weight = clinical_weight

        # Boundary detection
        self.boundary_detector = BoundaryDetector(n_classes=n_classes)

        # Clinical importance mapping
        self.clinical_mapper = ClinicalImportanceMap(n_classes=n_classes)

        # Feature modulation network
        # Takes features + attention map and produces modulated features
        self.modulation = nn.Sequential(
            nn.Conv2d(feature_dim + 2, feature_dim, 3, padding=1),  # +2 for boundary + clinical
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )

        # Attention scaling (learnable)
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        features: torch.Tensor,
        seg_logits: Optional[torch.Tensor] = None,
        seg_probs: Optional[torch.Tensor] = None,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply segmentation-guided attention to features.

        Args:
            features: Denoising encoder features [B, C, H, W]
            seg_logits: Raw segmentation logits [B, n_classes, H, W]
            seg_probs: Segmentation probabilities [B, n_classes, H, W]
            seg_mask: Hard segmentation mask [B, H, W]

        Returns:
            Dict containing:
                - 'features': Modulated features [B, C, H, W]
                - 'boundary_attn': Boundary attention map [B, 1, H, W]
                - 'clinical_attn': Clinical importance map [B, 1, H, W]
                - 'combined_attn': Combined attention [B, 1, H, W]
        """
        B, C, H, W = features.shape

        # Get probabilities if not provided
        if seg_probs is None and seg_logits is not None:
            seg_probs = F.softmax(seg_logits, dim=1)

        # Resize segmentation to match features if needed
        if seg_probs is not None and seg_probs.shape[-2:] != features.shape[-2:]:
            seg_probs = F.interpolate(seg_probs, size=(H, W), mode='bilinear', align_corners=False)
        if seg_mask is not None and seg_mask.shape[-2:] != features.shape[-2:]:
            seg_mask = F.interpolate(
                seg_mask.unsqueeze(1).float(), size=(H, W), mode='nearest'
            ).squeeze(1).long()

        # Compute boundary attention
        boundary_attn = self.boundary_detector(seg_logits, seg_probs, seg_mask)

        # Compute clinical importance
        clinical_attn = self.clinical_mapper(seg_probs, seg_mask)

        # Combine attention maps
        combined_attn = (
            self.boundary_weight * boundary_attn +
            self.clinical_weight * clinical_attn
        )
        # Normalize to reasonable range
        combined_attn = combined_attn / (self.boundary_weight + self.clinical_weight)

        # Modulate features with attention
        # Concatenate features with attention maps
        combined_input = torch.cat([features, boundary_attn, clinical_attn], dim=1)
        modulation = self.modulation(combined_input)

        # Residual connection with attention scaling
        # Higher attention = stronger modulation
        modulated_features = features + self.attn_scale * combined_attn * modulation

        return {
            'features': modulated_features,
            'boundary_attn': boundary_attn,
            'clinical_attn': clinical_attn,
            'combined_attn': combined_attn,
        }


class MultiScaleSegGuidedAttention(nn.Module):
    """
    Multi-scale segmentation-guided attention.

    Applies attention at multiple feature scales to capture both
    fine boundary details and broader layer context.
    """

    def __init__(
        self,
        n_classes: int = 4,
        feature_dims: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        """
        Args:
            n_classes: Number of segmentation classes
            feature_dims: Feature dimensions at each scale
        """
        super().__init__()

        self.attention_modules = nn.ModuleList([
            SegmentationGuidedAttention(
                n_classes=n_classes,
                feature_dim=dim,
            )
            for dim in feature_dims
        ])

    def forward(
        self,
        multi_scale_features: list,
        seg_probs: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Apply attention at multiple scales.

        Args:
            multi_scale_features: List of features at different scales
            seg_probs: Segmentation probabilities [B, C, H, W]
            seg_mask: Optional hard segmentation mask [B, H, W]

        Returns:
            List of modulated features at each scale
        """
        modulated = []
        for features, attn_module in zip(multi_scale_features, self.attention_modules):
            result = attn_module(features, seg_probs=seg_probs, seg_mask=seg_mask)
            modulated.append(result['features'])
        return modulated


class CrossLayerAttention(nn.Module):
    """
    Cross-layer attention mechanism for OCT.

    Captures vertical dependencies between layers (RNFL→GCL→INL→...→RPE).
    OCT layers have strong vertical relationships - the appearance of one
    layer informs processing of adjacent layers.
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        window_size: int = 32,
    ):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            window_size: Vertical window size for columnar attention
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Horizontal smoothing (layers are continuous horizontally)
        self.h_smooth = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply columnar attention.

        Args:
            x: Features [B, C, H, W]

        Returns:
            Attention-enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Reshape for vertical attention: treat each column as a sequence
        # [B, C, H, W] -> [B*W, H, C]
        x_col = x.permute(0, 3, 2, 1).reshape(B * W, H, C)

        # Compute Q, K, V
        qkv = self.qkv(x_col).reshape(B * W, H, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*W, heads, H, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention along vertical axis
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B * W, H, C)
        out = self.proj(out)

        # Reshape back: [B*W, H, C] -> [B, C, H, W]
        out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)

        # Horizontal smoothing for layer continuity
        out = self.h_smooth(out)

        return x + 0.1 * out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the module
    print("Testing SegmentationGuidedAttention...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create module
    attention = SegmentationGuidedAttention(
        n_classes=4,
        feature_dim=64,
    ).to(device)

    print(f"Total parameters: {count_parameters(attention):,}")

    # Test forward pass
    B, C, H, W = 2, 64, 128, 128
    features = torch.randn(B, C, H, W).to(device)
    seg_probs = F.softmax(torch.randn(B, 4, H, W), dim=1).to(device)
    seg_mask = torch.randint(0, 4, (B, H, W)).to(device)

    with torch.no_grad():
        results = attention(features, seg_probs=seg_probs, seg_mask=seg_mask)

    print(f"Modulated features shape: {results['features'].shape}")
    print(f"Boundary attention shape: {results['boundary_attn'].shape}")
    print(f"Clinical attention shape: {results['clinical_attn'].shape}")
    print(f"Combined attention shape: {results['combined_attn'].shape}")

    # Test cross-layer attention
    print("\nTesting CrossLayerAttention...")
    cross_attn = CrossLayerAttention(dim=64, num_heads=4).to(device)
    print(f"CrossLayerAttention parameters: {count_parameters(cross_attn):,}")

    with torch.no_grad():
        out = cross_attn(features)
    print(f"Cross-layer attention output shape: {out.shape}")

    print("\nAll tests passed!")
