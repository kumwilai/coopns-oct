#!/usr/bin/env python3
"""
Columnar Attention for OCT Analysis

KEY TMI CONTRIBUTION: Novel attention mechanism exploiting OCT's columnar structure.

OCT images have a unique property: each column (A-scan) is an independent depth scan.
This module processes columns with:
1. Intra-column attention: Model layer relationships within each column
2. Inter-column attention: Model spatial continuity across columns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ColumnPositionalEncoding(nn.Module):
    """Positional encoding for column (depth) positions."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, W, H, C] or [B*W, H, C]
        """
        seq_len = x.size(-2)
        return x + self.pe[:seq_len].unsqueeze(0)


class IntraColumnAttention(nn.Module):
    """
    Self-attention within each column (A-scan).

    Models relationships between different depths in the same column.
    This captures: layer ordering, relative positions, intensity patterns.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, max_seq_len: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len  # Memory-efficient: only 64x64 supported

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # MEMORY FIX: Use 64x64 max to fit in limited RAM
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, max_seq_len, max_seq_len))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, W, H, C] - batch, width (columns), height (depth), channels
        Returns:
            out: [B, W, H, C]
        """
        B, W, H, C = x.shape

        # Reshape for per-column processing: [B*W, H, C]
        x_flat = x.view(B * W, H, C)

        # QKV projection
        qkv = self.qkv(x_flat).reshape(B * W, H, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*W, heads, H, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*W, heads, H, H]

        # Add relative position bias
        attn = attn + self.relative_bias[:, :H, :H].unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B * W, H, C)
        out = self.proj(out)

        return out.view(B, W, H, C)


class InterColumnAttention(nn.Module):
    """
    Attention across columns at the same depth.

    Models spatial continuity of layer boundaries across the image width.
    This ensures smooth, continuous boundaries.
    """

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, W, H, C]
        Returns:
            out: [B, W, H, C]
        """
        B, W, H, C = x.shape

        # Reshape for per-row (same depth) processing: [B*H, W, C]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * H, W, C)

        # Use windowed attention for efficiency
        if W > self.window_size:
            # Pad to multiple of window size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_w > 0:
                x_flat = F.pad(x_flat, (0, 0, 0, pad_w))

            W_padded = x_flat.size(1)
            num_windows = W_padded // self.window_size

            # Reshape into windows: [B*H*num_windows, window_size, C]
            x_windows = x_flat.reshape(B * H, num_windows, self.window_size, C)
            x_windows = x_windows.reshape(-1, self.window_size, C)

            # QKV
            qkv = self.qkv(x_windows).reshape(-1, self.window_size, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, C)

            # Reshape back
            out = out.reshape(B * H, num_windows, self.window_size, C)
            out = out.reshape(B * H, W_padded, C)
            out = out[:, :W, :].contiguous()  # Remove padding
        else:
            # Full attention for small widths
            qkv = self.qkv(x_flat).reshape(B * H, W, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = (attn @ v).transpose(1, 2).reshape(B * H, W, C)

        out = self.proj(out)

        return out.reshape(B, H, W, C).permute(0, 2, 1, 3).contiguous()


class ColumnarTransformerBlock(nn.Module):
    """
    Full columnar transformer block combining intra and inter-column attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        window_size: int = 32,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.intra_attn = IntraColumnAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.inter_attn = InterColumnAttention(dim, num_heads, window_size, dropout)

        self.norm3 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, W, H, C]
        Returns:
            out: [B, W, H, C]
        """
        # Intra-column attention (depth relationships)
        x = x + self.intra_attn(self.norm1(x))

        # Inter-column attention (spatial continuity)
        x = x + self.inter_attn(self.norm2(x))

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x


class ColumnarEncoder(nn.Module):
    """
    Columnar encoder that processes OCT image column-wise.

    Converts spatial features [B, C, H, W] to columnar representation
    and applies columnar transformer blocks.
    """

    def __init__(
        self,
        in_channels: int = 64,
        dim: int = 128,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project features to columnar space
        self.input_proj = nn.Conv2d(in_channels, dim, 1)

        # Positional encoding for depth
        self.pos_enc = ColumnPositionalEncoding(dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ColumnarTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Conv2d(dim, in_channels, 1)

        self.dim = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] spatial features from backbone
        Returns:
            out: [B, C, H, W] enhanced features
            col_features: [B, W, H, dim] columnar features for boundary regression
        """
        B, C, H, W = x.shape

        # Project to columnar dimension
        x_proj = self.input_proj(x)  # [B, dim, H, W]

        # Reshape to columnar format: [B, W, H, dim]
        x_col = x_proj.permute(0, 3, 2, 1)  # [B, W, H, dim]

        # Add positional encoding
        x_col = self.pos_enc(x_col)

        # Apply transformer blocks
        for block in self.blocks:
            x_col = block(x_col)

        # Store columnar features for boundary regression
        col_features = x_col

        # Reshape back to spatial format
        x_spatial = x_col.permute(0, 3, 2, 1)  # [B, dim, H, W]

        # Project back to original channels
        out = self.output_proj(x_spatial)

        # Residual connection
        out = out + x

        return out, col_features


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    print("Testing Columnar Attention...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test input: [B, C, H, W]
    B, C, H, W = 2, 64, 256, 512
    x = torch.randn(B, C, H, W).to(device)

    # Test columnar encoder
    encoder = ColumnarEncoder(in_channels=C, dim=128, num_blocks=2).to(device)
    out, col_features = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Columnar features shape: {col_features.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Check shapes
    assert out.shape == x.shape
    assert col_features.shape == (B, W, H, 128)

    print("\nAll tests passed!")
