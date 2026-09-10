"""
SwinIR: Image Restoration Using Swin Transformer (7M parameter version)
Based on: "SwinIR: Image Restoration Using Swin Transformer" (Liang et al., ICCVW 2021)
https://github.com/JingyunLiang/SwinIR

Configuration targeting ~7M parameters for fair comparison with NAFNet backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (no timm dependency)."""
    with torch.no_grad():
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def window_partition(x, window_size):
    """Partition image into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    nH, nW = H // window_size, W // window_size
    # reshape avoids the contiguous() memory copy that permute+view requires
    x = x.view(B, nH, window_size, nW, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to image.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H, W: original spatial dimensions (must be multiples of window_size)
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    nH, nW = H // window_size, W // window_size
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention (W-MSA) with relative position bias.

    Args:
        dim: input feature dimension
        window_size: (wh, ww) window size tuple
        num_heads: number of attention heads
    """

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*wh-1) * (2*ww-1) entries, one per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute pairwise relative position index for each token in the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, wh, ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, wh*ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size*window_size
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B_, num_heads, N, head_dim)

        # Relative position bias (computed once per forward)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        ).permute(2, 0, 1).contiguous()  # (nH, N, N)

        if mask is not None:
            nW = mask.shape[0]
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0) + relative_position_bias.unsqueeze(0).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        else:
            # Fused SDPA with position bias (PyTorch 2.0+, faster on CPU)
            pos_bias = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=pos_bias)
            x = x.transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        return x


class SwinTransformerLayer(nn.Module):
    """Single Swin Transformer layer with optional shifted window attention.

    Args:
        dim: feature dimension
        num_heads: number of attention heads
        window_size: window size for local attention
        shift_size: shift size for SW-MSA (0 = no shift, window_size//2 = shifted)
        mlp_ratio: MLP hidden dim expansion ratio
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self._attn_mask_cache = {}

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def _get_attn_mask(self, H, W, device):
        """Compute or retrieve cached attention mask for shifted window attention."""
        cache_key = (H, W, str(device))
        if cache_key in self._attn_mask_cache:
            return self._attn_mask_cache[cache_key]

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

        self._attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def forward(self, x, x_size):
        """
        Args:
            x: (B, H*W, C)
            x_size: (H, W)
        """
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad to multiples of window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            # Pad directly in (B, H, W, C) format: (left_C, right_C, left_W, right_W, left_H, right_H)
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        if self.shift_size > 0:
            attn_mask = self._get_attn_mask(Hp, Wp, x.device)
        else:
            attn_mask = None
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Residual connections
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block.

    Contains multiple SwinTransformerLayers with alternating regular/shifted windows,
    followed by a convolution and a residual connection.

    Args:
        dim: feature dimension
        depth: number of Swin Transformer layers in this block
        num_heads: number of attention heads
        window_size: window size
        mlp_ratio: MLP expansion ratio
    """

    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        """
        Args:
            x: (B, H*W, C)
            x_size: (H, W)
        """
        identity = x
        for layer in self.layers:
            x = layer(x, x_size)
        # Reshape to spatial, apply conv, reshape back
        H, W = x_size
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + identity


class SwinIR(nn.Module):
    """SwinIR: Image Restoration Using Swin Transformer.

    Architecture: shallow feature extraction -> deep feature extraction (RSTB blocks)
    -> reconstruction, with a global residual connection from input.

    Default config targets ~7M parameters:
        embed_dim=138, depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6],
        window_size=8, mlp_ratio=2.0

    Args:
        in_channels: input channels (1 for grayscale)
        out_channels: output channels
        embed_dim: embedding dimension
        depths: number of STL layers in each RSTB
        num_heads: number of attention heads in each RSTB
        window_size: window size for local attention
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        embed_dim=138,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.use_checkpoint = True

        # --- Shallow feature extraction ---
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # --- Deep feature extraction (RSTB blocks) ---
        self.rstb_blocks = nn.ModuleList([
            RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- Reconstruction ---
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        # Initialize weights
        self.apply(self._init_weights)

        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SwinIR-7M | Total params: {n_params:,} ({n_params/1e6:.2f}M) | "
              f"Trainable: {n_trainable:,} ({n_trainable/1e6:.2f}M)")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) input grayscale image
        Returns:
            output: (B, 1, H, W) denoised image
        """
        B, C, orig_H, orig_W = x.shape

        # Pad input to multiple of window_size
        pad_h = (self.window_size - orig_H % self.window_size) % self.window_size
        pad_w = (self.window_size - orig_W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x

        # Shallow feature extraction
        shallow_feat = self.conv_first(x_padded)  # (B, embed_dim, Hp, Wp)
        _, _, Hp, Wp = shallow_feat.shape
        x_size = (Hp, Wp)

        # Reshape to token sequence: (B, embed_dim, H, W) -> (B, H*W, embed_dim)
        tokens = shallow_feat.flatten(2).transpose(1, 2)

        # Deep feature extraction through RSTB blocks (with gradient checkpointing)
        for rstb in self.rstb_blocks:
            if self.use_checkpoint and self.training:
                tokens = checkpoint(rstb, tokens, x_size, use_reentrant=False)
            else:
                tokens = rstb(tokens, x_size)

        tokens = self.norm(tokens)

        # Reshape back to spatial
        deep_feat = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)

        # Reconstruction with global residual
        deep_feat = self.conv_after_body(deep_feat) + shallow_feat
        output = self.conv_last(deep_feat) + x_padded

        # Remove padding
        output = output[:, :, :orig_H, :orig_W]

        return output


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("=" * 80)
    print("SwinIR-7M: Testing model creation and forward pass")
    print("=" * 80)

    model = SwinIR(
        in_channels=1,
        out_channels=1,
        embed_dim=138,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.0,
    )

    total, trainable = count_parameters(model)
    print(f"\nParameter count: {total:,} ({total/1e6:.2f}M)")

    # Test with different spatial sizes
    test_sizes = [(1, 1, 64, 64), (1, 1, 128, 128), (1, 1, 100, 73)]
    for size in test_sizes:
        x = torch.randn(*size)
        with torch.no_grad():
            y = model(x)
        assert y.shape == size, f"Shape mismatch: input {size} -> output {y.shape}"
        print(f"Input {size} -> Output {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")

    print("\nAll tests passed.")
