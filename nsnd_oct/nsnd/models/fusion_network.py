"""
Neural Fusion Network

Attention-based fusion of component-denoised images,
conditioned on symbolic weights and confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class LightweightEncoder(nn.Module):
    """Lightweight feature encoder for each denoised component"""

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for component interaction

    Allows each component to attend to features from other components
    """

    def __init__(self, dim: int = 64, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Stacked features [B, N, C, H, W] where N = num_components
            weights: Component weights [B, N] for weighted attention

        Returns:
            attended: Fused features [B, C, H, W]
        """
        B, N, C, H, W = x.shape

        # Reshape for attention: [B*H*W, N, C]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, N, C)

        # Compute Q, K, V
        qkv = self.qkv(x_flat).reshape(B*H*W, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*H*W, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*H*W, heads, N, N]

        # Optional: modulate attention by symbolic weights
        if weights is not None:
            # weights: [B, N] -> [B*H*W, 1, N, 1]
            weight_mask = weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            weight_mask = weight_mask.repeat(H*W, self.heads, 1, N)  # [B*H*W, heads, N, N]
            attn = attn + torch.log(weight_mask + 1e-8)

        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B*H*W, N, C)  # [B*H*W, N, C]
        out = self.proj(out)

        # Aggregate across components (weighted sum)
        if weights is not None:
            weight_expand = weights.view(B, 1, N, 1).repeat(H*W, 1, 1, C)  # [B*H*W, 1, N, C]
            out = (out.unsqueeze(1) * weight_expand).sum(dim=2)  # [B*H*W, 1, C]
        else:
            out = out.mean(dim=1, keepdim=True)  # [B*H*W, 1, C]

        # Reshape back to spatial
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        return out


class UNetDecoder(nn.Module):
    """Simple U-Net style decoder"""

    def __init__(self, in_channels: int = 64, out_channels: int = 1):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class NeuralFusionNetwork(nn.Module):
    """
    Complete fusion network that combines component-denoised images

    Key features:
    - Per-component feature encoding
    - Cross-attention for component interaction
    - FiLM conditioning on symbolic weights
    - Uncertainty estimation
    """

    def __init__(
        self,
        n_components: int = 4,
        feature_channels: int = 64,
        attention_heads: int = 4,
        use_uncertainty: bool = True,
    ):
        """
        Args:
            n_components: Number of noise components (default: 4)
            feature_channels: Feature dimension for encoders
            attention_heads: Number of attention heads
            use_uncertainty: Whether to predict uncertainty map
        """
        super().__init__()

        self.n_components = n_components
        self.use_uncertainty = use_uncertainty

        # Feature encoder for each component
        self.encoders = nn.ModuleList([
            LightweightEncoder(in_channels=1, out_channels=feature_channels)
            for _ in range(n_components)
        ])

        # Cross-attention for component interaction
        self.cross_attention = CrossAttention(
            dim=feature_channels,
            heads=attention_heads
        )

        # FiLM generator: condition on symbolic weights + confidence
        self.film_generator = nn.Sequential(
            nn.Linear(n_components + 1, 128),  # weights + confidence
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_channels * 2)  # gamma and beta
        )

        # Decoder
        self.decoder = UNetDecoder(
            in_channels=feature_channels,
            out_channels=1
        )

        # Uncertainty head (optional)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(feature_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()  # Uncertainty in [0, 1]
            )

    def forward(
        self,
        denoised_components: Dict[str, torch.Tensor],
        symbolic_weights: Dict[str, torch.Tensor],
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse component-denoised images

        Args:
            denoised_components: Dict of {component_name: denoised_image [B,1,H,W]}
            symbolic_weights: Dict of {component_name: weight [B]}
            confidence: Overall confidence score [B]

        Returns:
            output: Fused clean image [B, 1, H, W]
            uncertainty: Uncertainty map [B, 1, H, W] (if enabled)
        """
        # Stack denoised components in consistent order
        component_names = ['speckle', 'banding', 'gaussian', 'shot']
        denoised_stack = torch.stack([
            denoised_components[name] for name in component_names
        ], dim=1)  # [B, N, 1, H, W]

        weight_stack = torch.stack([
            symbolic_weights[name] for name in component_names
        ], dim=1)  # [B, N]

        B, N, _, H, W = denoised_stack.shape

        # Encode each component
        features = []
        for i, encoder in enumerate(self.encoders):
            feat = encoder(denoised_stack[:, i])  # [B, C, H, W]
            features.append(feat)

        # Stack features
        stacked_features = torch.stack(features, dim=1)  # [B, N, C, H, W]

        # Apply cross-attention with weight guidance
        attended = self.cross_attention(stacked_features, weight_stack)  # [B, C, H, W]

        # FiLM conditioning on weights + confidence
        weight_vec = torch.cat([weight_stack, confidence.unsqueeze(-1)], dim=-1)  # [B, N+1]
        film_params = self.film_generator(weight_vec)  # [B, 2*C]
        gamma, beta = film_params.chunk(2, dim=-1)  # Each [B, C]

        # Apply FiLM
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        conditioned = gamma * attended + beta

        # Decode to final output
        output = self.decoder(conditioned)

        # Uncertainty estimation
        uncertainty = None
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(conditioned)

        return output, uncertainty


class SimpleFusionNetwork(nn.Module):
    """
    Simplified fusion via weighted sum (baseline)

    Uses symbolic weights directly without learned attention
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        denoised_components: Dict[str, torch.Tensor],
        symbolic_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Simple weighted fusion

        Args:
            denoised_components: Dict of {component_name: denoised_image [B,1,H,W]}
            symbolic_weights: Dict of {component_name: weight [B]}

        Returns:
            output: Fused clean image [B, 1, H, W]
        """
        component_names = ['speckle', 'banding', 'gaussian', 'shot']

        # Weighted sum
        output = torch.zeros_like(denoised_components['speckle'])

        for name in component_names:
            weight = symbolic_weights[name].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            output = output + weight * denoised_components[name]

        return output
