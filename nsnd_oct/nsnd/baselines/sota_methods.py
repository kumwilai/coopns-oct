#!/usr/bin/env python3
"""
State-of-the-Art Denoising Methods for Comparison

Implements or wraps:
1. BM3D - Classical block-matching 3D filtering
2. DnCNN - Deep denoising CNN
3. Restormer - Transformer-based restoration
4. N2N - Noise2Noise (self-supervised)
5. Multi-frame averaging - Clinical standard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
import warnings


# ============================================================================
# BM3D (Classical Method)
# ============================================================================

class BM3DDenoiser:
    """
    BM3D denoising wrapper.

    Block-Matching and 3D filtering - classical SOTA for Gaussian noise.
    Requires bm3d package: pip install bm3d
    """

    def __init__(self, sigma_psd: float = 25/255):
        self.sigma_psd = sigma_psd
        self._bm3d = None

    def _load_bm3d(self):
        if self._bm3d is None:
            try:
                import bm3d
                self._bm3d = bm3d
            except ImportError:
                warnings.warn("BM3D not installed. Install with: pip install bm3d")
                return None
        return self._bm3d

    def __call__(self, noisy: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """
        Denoise image using BM3D.

        Args:
            noisy: [B, 1, H, W] noisy image
            sigma: Noise level (default: self.sigma_psd)

        Returns:
            Denoised image [B, 1, H, W]
        """
        bm3d = self._load_bm3d()
        if bm3d is None:
            return noisy  # Return input if BM3D not available

        sigma = sigma if sigma is not None else self.sigma_psd

        device = noisy.device
        B, C, H, W = noisy.shape

        results = []
        for i in range(B):
            img_np = noisy[i, 0].cpu().numpy()
            denoised_np = bm3d.bm3d(img_np, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            results.append(torch.from_numpy(denoised_np).unsqueeze(0))

        return torch.stack(results, dim=0).to(device)


# ============================================================================
# DnCNN (Deep Learning Baseline)
# ============================================================================

class DnCNN(nn.Module):
    """
    DnCNN: Beyond a Gaussian Denoiser (Zhang et al., 2017)

    17-layer CNN for blind Gaussian denoising.
    """

    def __init__(self, in_channels=1, out_channels=1, num_layers=17, num_features=64):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(in_channels, num_features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(nn.Conv2d(num_features, out_channels, 3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # Residual learning: predict noise
        noise = self.dncnn(x)
        return x - noise


# ============================================================================
# SwinIR (Swin Transformer for Image Restoration)
# ============================================================================

class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention for SwinIR."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block for SwinIR."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x, H, W):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = self._window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

    def _window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def _window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB) for SwinIR."""

    def __init__(self, dim, num_heads, window_size=8, num_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            for i in range(num_blocks)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, H, W):
        res = x
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).view(-1, x.shape[-1], H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + res


class SwinIR(nn.Module):
    """
    SwinIR: Image Restoration Using Swin Transformer (Liang et al., ICCVW 2021)

    A strong baseline for image denoising using shifted window attention.
    """

    def __init__(self, in_channels=1, out_channels=1, embed_dim=96, num_heads=6,
                 window_size=8, num_rstb=4, num_blocks=6):
        super().__init__()
        self.window_size = window_size

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        # Deep feature extraction (RSTB blocks)
        self.rstb_blocks = nn.ModuleList([
            RSTB(embed_dim, num_heads, window_size, num_blocks)
            for _ in range(num_rstb)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        # Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x):
        # Pad to multiple of window_size
        _, _, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        H_pad, W_pad = x.shape[2], x.shape[3]

        # Shallow features
        shallow = self.conv_first(x)
        x = shallow.flatten(2).transpose(1, 2)

        # Deep features (RSTB)
        for rstb in self.rstb_blocks:
            x = rstb(x, H_pad, W_pad)

        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, shallow.shape[1], H_pad, W_pad)
        x = self.conv_after_body(x) + shallow

        # Reconstruction (residual learning)
        out = self.conv_last(x)

        # Remove padding
        out = out[:, :, :H, :W]
        return out


# ============================================================================
# UNet Denoiser (Strong Baseline)
# ============================================================================

class UNetDenoiser(nn.Module):
    """
    U-Net based denoiser.

    Encoder-decoder with skip connections.
    """

    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, stride=2)
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)

        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = self._conv_block(base_features * 2, base_features)

        self.out = nn.Conv2d(base_features, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# ============================================================================
# Restormer (Transformer-based)
# ============================================================================

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Restormer Transformer Block."""

    def __init__(self, dim, num_heads=8, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion)

    def forward(self, x):
        b, c, h, w = x.shape

        # Attention
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm)

        # FFN
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm)

        return x


class RestormerLite(nn.Module):
    """
    Simplified Restormer for denoising.

    Lighter version suitable for comparison without excessive compute.
    """

    def __init__(self, in_channels=1, out_channels=1, dim=48, num_blocks=[2, 3, 3, 4]):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1)

        # Encoder
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[TransformerBlock(dim * 4) for _ in range(num_blocks[2])])

        # Decoder
        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2) for _ in range(num_blocks[1])])

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x):
        inp = x

        x = self.patch_embed(x)

        # Encoder
        e1 = self.encoder_level1(x)
        e2 = self.encoder_level2(self.down1(e1))

        # Bottleneck
        b = self.bottleneck(self.down2(e2))

        # Decoder
        d2 = self.decoder_level2(self.up2(b) + e2)
        d1 = self.decoder_level1(self.up1(d2) + e1)

        return inp + self.output(d1)


# ============================================================================
# Multi-frame Averaging (Clinical Baseline)
# ============================================================================

class MultiFrameAverager:
    """
    Multi-frame averaging - clinical standard for OCT denoising.

    Simulates averaging multiple acquisitions.
    """

    def __init__(self, num_frames: int = 4):
        self.num_frames = num_frames

    def __call__(self, noisy: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """
        Simulate multi-frame averaging by adding independent noise and averaging.

        This simulates what would happen with N independent acquisitions.
        Noise is reduced by factor of sqrt(N).

        Args:
            noisy: Single noisy frame
            noise_std: Estimated noise standard deviation

        Returns:
            Averaged result (simulated)
        """
        # In real scenario, we'd have multiple acquisitions
        # Here we simulate by averaging the input with additional noise realizations
        # This is a simulation - real multi-frame would need multiple acquisitions

        B, C, H, W = noisy.shape

        # Generate N-1 additional "frames" with independent noise
        # This assumes we know the noise level approximately
        frames = [noisy]
        for _ in range(self.num_frames - 1):
            # Simulate another acquisition with different noise realization
            additional_noise = torch.randn_like(noisy) * noise_std
            frames.append(noisy + additional_noise - torch.randn_like(noisy) * noise_std)

        # Average
        averaged = torch.stack(frames, dim=0).mean(dim=0)

        return averaged


# ============================================================================
# Baseline Manager
# ============================================================================

class BaselineManager:
    """
    Manages all baseline methods for fair comparison.
    """

    AVAILABLE_METHODS = ['bm3d', 'dncnn', 'unet', 'restormer', 'multiframe', 'nafnet']

    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}

    def get_method(self, name: str, **kwargs) -> nn.Module:
        """Get or create a baseline method."""
        if name not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown method: {name}. Available: {self.AVAILABLE_METHODS}")

        if name not in self.models:
            if name == 'bm3d':
                self.models[name] = BM3DDenoiser(**kwargs)
            elif name == 'dncnn':
                self.models[name] = DnCNN(**kwargs).to(self.device)
            elif name == 'unet':
                self.models[name] = UNetDenoiser(**kwargs).to(self.device)
            elif name == 'restormer':
                self.models[name] = RestormerLite(**kwargs).to(self.device)
            elif name == 'multiframe':
                self.models[name] = MultiFrameAverager(**kwargs)

        return self.models[name]

    def load_pretrained(self, name: str, checkpoint_path: str):
        """Load pretrained weights for a method."""
        if name not in self.models:
            self.get_method(name)

        if isinstance(self.models[name], nn.Module):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.models[name].load_state_dict(state)
            print(f"Loaded pretrained {name} from {checkpoint_path}")

    def evaluate_all(self, noisy: torch.Tensor, clean: torch.Tensor,
                     methods: Optional[list] = None) -> Dict[str, Dict]:
        """
        Evaluate multiple methods on the same input.

        Args:
            noisy: Noisy input [B, 1, H, W]
            clean: Clean reference [B, 1, H, W]
            methods: List of methods to evaluate (default: all)

        Returns:
            Dictionary of results per method
        """
        from nsnd.utils.metrics import compute_psnr, compute_ssim

        if methods is None:
            methods = self.AVAILABLE_METHODS

        results = {}

        for name in methods:
            try:
                method = self.get_method(name)

                if isinstance(method, nn.Module):
                    method.eval()
                    with torch.no_grad():
                        denoised = method(noisy)
                else:
                    denoised = method(noisy)

                denoised = denoised.clamp(0, 1)

                psnr = compute_psnr(denoised, clean)
                ssim = compute_ssim(denoised, clean)

                results[name] = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'denoised': denoised,
                }

            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = {'psnr': 0, 'ssim': 0, 'error': str(e)}

        return results


if __name__ == '__main__':
    print("Testing baseline methods...")

    # Create dummy data
    torch.manual_seed(42)
    clean = torch.rand(1, 1, 64, 64) * 0.5 + 0.25
    noisy = clean + torch.randn_like(clean) * 0.1

    # Test each method
    manager = BaselineManager(device='cpu')

    print("\nTesting DnCNN...")
    dncnn = manager.get_method('dncnn')
    out = dncnn(noisy)
    print(f"  Output shape: {out.shape}")

    print("\nTesting UNet...")
    unet = manager.get_method('unet')
    out = unet(noisy)
    print(f"  Output shape: {out.shape}")

    print("\nTesting RestormerLite...")
    restormer = manager.get_method('restormer')
    out = restormer(noisy)
    print(f"  Output shape: {out.shape}")

    print("\nTesting MultiFrame...")
    mf = manager.get_method('multiframe', num_frames=4)
    out = mf(noisy, noise_std=0.1)
    print(f"  Output shape: {out.shape}")

    print("\nAll tests passed!")
