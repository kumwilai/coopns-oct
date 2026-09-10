"""SwinIR: Image Restoration Using Swin Transformer (Liang et al., CVPRW 2021)
Simplified small version for OCT denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA) module"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""

    def __init__(self, dim, num_heads, window_size=4, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        B, C, H, W = x.shape
        shortcut = x
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C

        # Window partition
        x = self.norm1(x)
        x_windows = self.window_partition(x, H, W, self.window_size)

        # W-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        x = self.window_reverse(attn_windows, H, W, self.window_size)

        # FFN
        x = x + self.mlp(self.norm2(x))

        # Reshape back
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x + shortcut

    def window_partition(self, x, H, W, window_size):
        """
        Args:
            x: (B, H*W, C)
            H, W: height and width
            window_size: window size
        Returns:
            windows: (num_windows*B, window_size*window_size, C)
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Pad if needed
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w

        # Partition
        x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows

    def window_reverse(self, windows, H, W, window_size):
        """
        Args:
            windows: (num_windows*B, window_size*window_size, C)
            H, W: height and width (before padding)
            window_size: window size
        Returns:
            x: (B, H*W, C)
        """
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        H_pad, W_pad = H + pad_h, W + pad_w

        B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
        x = windows.view(B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        return x.view(B, H * W, -1)


class SwinIRSmall(nn.Module):
    """Small SwinIR for OCT denoising"""

    def __init__(self, in_channels=1, out_channels=1, embed_dim=32, num_blocks=4,
                 num_heads=4, window_size=4):
        super().__init__()

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, num_heads, window_size)
            for _ in range(num_blocks)
        ])

        # Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

    def forward(self, x):
        # Shallow feature extraction
        x_first = self.conv_first(x)

        # Deep feature extraction (Swin Transformer blocks)
        x_deep = x_first
        for block in self.blocks:
            x_deep = block(x_deep)

        # Reconstruction
        x_out = self.conv_last(x_deep)

        # Residual connection
        return x_out + x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
