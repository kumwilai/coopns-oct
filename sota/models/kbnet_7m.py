"""
KBNet-7M: Kernel Basis Network for Image Restoration (~7M parameters)

Based on:
    "KBNet: Kernel Basis Network for Image Restoration"
    Y. Zhang et al., 2023
    https://github.com/zhangyi-3/KBNet

Architecture (KBNet-s variant, NAFNet-style UNet):
    - 3-level UNet encoder-decoder with element-wise skip connections
    - KBA (Kernel Basis Attention): content-adaptive convolution using learned
      basis kernel decomposition (nset=32 bases, depthwise gc=1 for CPU efficiency)
    - Two sub-layers per block: KBA attention + FFN (with SimpleGate)
    - Learnable residual scaling: beta, gamma, ga1, attgamma
    - PixelShuffle upsampling, strided conv downsampling
    - Global residual learning: output = input + model(input)

    7M configuration: width=32, enc=[2,2,4], mid=10, dec=[2,2,2], nset=32, gc=1
    Channels: [32, 64, 128], bottleneck=256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W).
    Uses fused F.layer_norm via NHWC permute for 2-5x speedup over manual mean/var."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)  # NHWC -> NCHW


class SimpleGate(nn.Module):
    """Split channels in half and multiply: 2C -> C."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class KBBlock_s(nn.Module):
    """KBNet-s block with two sub-layers: KBA attention + FFN.

    Sub-layer 1 (KBA attention):
        LayerNorm -> channel attention (sca) + gated path (conv11) +
        KBA (kernel basis attention with depthwise bases) -> residual with beta

    Sub-layer 2 (FFN):
        LayerNorm -> expand (1x1) -> SimpleGate -> project (1x1) ->
        residual with gamma

    Args:
        c: number of channels
        FFN_Expand: FFN expansion factor (default: 2)
        nset: number of kernel bases (default: 32)
        gc: channels per group in KBA (1=depthwise, 4=original paper)
        lightweight: if True, use 3x3 DW instead of 5x5 in conv11
    """

    def __init__(self, c, FFN_Expand=2, nset=32, gc=1, lightweight=False):
        super().__init__()
        self.c = c
        self.nset = nset
        self.k = 3  # KBA kernel size
        self.g = c // gc  # number of groups

        ffn_ch = int(FFN_Expand * c)

        # KBA basis kernels and biases
        # w: (1, nset, c * gc * k^2) — nset basis kernels (depthwise when gc=1)
        # b: (1, nset, c) — nset basis biases
        self.w = nn.Parameter(torch.zeros(1, nset, c * gc * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self._init_kba(self.w, self.b)

        # ===== Sub-layer 1: KBA attention =====
        self.norm1 = LayerNorm2d(c)

        # Channel attention: GAP + 1x1 conv (no bottleneck, no sigmoid)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
        )

        # Gating path: 1x1 conv + depthwise conv
        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, bias=True),
                nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c // 4, bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, bias=True),
                nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True),
            )

        # KBA input path: 1x1 conv + 3x3 depthwise conv
        self.conv1 = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.conv21 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)

        # Attention coefficient predictor for KBA
        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, interc, kernel_size=3, padding=1, groups=interc, bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, nset, kernel_size=1, bias=True),
        )
        self.conv211 = nn.Conv2d(c, nset, kernel_size=1, bias=True)

        # Output projection: c -> c (dw_ch//2 = c when DW_Expand=2)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, bias=True)

        # Learnable scaling (init 1e-2)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1) + 1e-2)
        self.ga1 = nn.Parameter(torch.zeros(1, c, 1, 1) + 1e-2)
        self.attgamma = nn.Parameter(torch.zeros(1, nset, 1, 1) + 1e-2)

        # ===== Sub-layer 2: FFN =====
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
        self.sg = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1) + 1e-2)

    def _init_kba(self, weight, bias):
        """Init KBA basis kernels with correct fan_in for depthwise k×k kernels.

        PyTorch's kaiming_uniform_ computes fan_in from the full tensor shape
        (1, nset, C*gc*k²), giving fan_in = nset*C*gc*k² ≈ 9216. But each basis
        is an independent depthwise convolution kernel where the correct
        fan_in = k² = 9. The original init makes basis weights ~32x too small.
        """
        fan_in = self.k ** 2
        std = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(weight, -std, std)
        if bias is not None:
            nn.init.uniform_(bias, -std, std)

    def _kba_forward(self, x, att):
        """Kernel Basis Attention: content-adaptive convolution from learned bases.

        For gc=1 (depthwise), uses a single batched grouped convolution instead of
        looping over nset bases. Input is repeated nset times along the channel dim,
        all basis kernels are applied at once, then weighted by attention coefficients.

        For gc>1, uses the full unfold-based grouped matrix multiply.

        Args:
            x: (B, C, H, W) pre-convolved features
            att: (B, nset, H, W) per-pixel basis coefficients
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        gc = C // self.g  # channels per group
        k = self.k

        if gc == 1:
            # Vectorized depthwise KBA via unfold: 2 ops instead of nset-iteration loop
            # att: (B, nset, H, W) -> (B, nset, H*W) — no transpose needed
            att_flat = att.reshape(B, self.nset, H * W)

            # Step 1: Synthesize per-pixel kernels via w^T @ att (avoids 2 transposes)
            # w: (1, nset, C*k²) -> w_t: (C*k², nset); result: (B, C*k², H*W)
            weighted_w = self.w.squeeze(0).t() @ att_flat  # (C*k², nset) @ (B, nset, H*W)

            # Step 2: Unfold input into patches and apply synthesized kernels
            uf = F.unfold(x, kernel_size=k, padding=k // 2)  # (B, C*k², H*W)
            # Element-wise multiply + sum over k² dim
            out = (weighted_w * uf).reshape(B, C, k * k, H * W).sum(dim=2)
            out = out.reshape(B, C, H, W)

            # Bias: b^T @ att_flat (avoids einsum with redundant dim)
            bias = (self.b.squeeze(0).t() @ att_flat).reshape(B, C, H, W)
            return out + bias
        else:
            # Full unfold-based grouped KBA (original approach, memory-intensive)
            KK = k ** 2
            att_flat = att.reshape(B, self.nset, H * W).transpose(-2, -1)

            bias = att_flat @ self.b
            attk = att_flat @ self.w

            uf = F.unfold(x, kernel_size=k, padding=k // 2)
            uf = uf.reshape(B, self.g, gc * KK, H * W).permute(0, 3, 1, 2)
            attk = attk.reshape(B, H * W, self.g, gc, gc * KK)

            out = (attk @ uf.unsqueeze(-1)).squeeze(-1)
            out = out.reshape(B, H * W, C) + bias
            return out.transpose(-1, -2).reshape(B, C, H, W)

    def forward(self, inp):
        x = self.norm1(inp)

        # Channel attention (no sigmoid, multiplicative)
        sca = self.sca(x)

        # Gating path: 1x1 + depthwise conv
        x1 = self.conv11(x)

        # KBA attention coefficients
        att = self.conv2(x) * self.attgamma + self.conv211(x)

        # KBA input
        uf = self.conv21(self.conv1(x))

        # KBA: adaptive convolution + residual
        x = self._kba_forward(uf, att) * self.ga1 + uf

        # Gated merge
        x = x * x1 * sca

        # Output projection
        x = self.conv3(x)
        y = inp + x * self.beta

        # FFN sub-layer
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


# ---------------------------------------------------------------------------
# Main KBNet model
# ---------------------------------------------------------------------------


class KBNet(nn.Module):
    """KBNet: Kernel Basis Network for grayscale image denoising.

    3-level UNet with KBA blocks, SimpleGate activation, and learnable scaling.
    Uses element-wise skip connections and global residual learning.
    PixelShuffle for upsampling, strided conv for downsampling.

    Automatically pads input to a multiple of 2^num_levels (=8) for alignment.

    Args:
        in_channels: Input channels (default: 1).
        out_channels: Output channels (default: 1).
        width: Base channel width (default: 32).
        middle_blk_num: Number of blocks in bottleneck (default: 10).
        enc_blk_nums: Blocks per encoder level (default: [2, 2, 4]).
        dec_blk_nums: Blocks per decoder level (default: [2, 2, 2]).
        nset: Number of kernel bases in KBA (default: 32).
        gc: Channels per group in KBA (default: 1 for CPU efficiency).
        ffn_scale: FFN expansion factor (default: 2).
        lightweight: Use 3x3 DW instead of 5x5 in conv11 (default: False).
        use_checkpoint: Gradient checkpointing (default: True).
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=32,
        middle_blk_num=10,
        enc_blk_nums=(2, 2, 4),
        dec_blk_nums=(2, 2, 2),
        nset=32,
        gc=1,
        ffn_scale=2,
        lightweight=False,
        use_checkpoint=True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        num_levels = len(enc_blk_nums)
        self.padder_size = 2 ** num_levels

        # Input projection
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[
                    KBBlock_s(chan, FFN_Expand=ffn_scale, nset=nset, gc=gc,
                              lightweight=lightweight)
                    for _ in range(num)
                ])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        # Bottleneck
        self.middle_blks = nn.Sequential(*[
            KBBlock_s(chan, FFN_Expand=ffn_scale, nset=nset, gc=gc,
                      lightweight=lightweight)
            for _ in range(middle_blk_num)
        ])

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[
                    KBBlock_s(chan, FFN_Expand=ffn_scale, nset=nset, gc=gc,
                              lightweight=lightweight)
                    for _ in range(num)
                ])
            )

        # Output projection
        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)

        # Initialize weights
        self._init_weights()

        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"KBNet-7M | Total params: {total_params:,} ({total_params/1e6:.2f}M) | "
              f"Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    def _init_weights(self):
        """Initialize weights (KBA bases already initialized in KBBlock_s).

        Conv2d uses PyTorch default init (kaiming_uniform, fan_in, a=sqrt(5))
        per the original KBNet. Only LayerNorm needs explicit init.
        """
        for m in self.modules():
            if isinstance(m, LayerNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _check_image_size(self, x):
        """Pad input so H and W are multiples of padder_size."""
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x

    def forward(self, x):
        """Forward pass with global residual learning.

        Args:
            x: (B, 1, H, W) noisy input image
        Returns:
            (B, 1, H, W) denoised output
        """
        B, C, orig_H, orig_W = x.shape
        inp = self._check_image_size(x)

        feat = self.intro(inp)

        # Encoder: collect skip connections
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            if self.use_checkpoint and self.training:
                feat = checkpoint(encoder, feat, use_reentrant=False)
            else:
                feat = encoder(feat)
            encs.append(feat)
            feat = down(feat)

        # Bottleneck
        if self.use_checkpoint and self.training:
            feat = checkpoint(self.middle_blks, feat, use_reentrant=False)
        else:
            feat = self.middle_blks(feat)

        # Decoder: element-wise skip connections (reversed)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            feat = up(feat)
            feat = feat + enc_skip  # Element-wise addition (NOT concat+conv)
            if self.use_checkpoint and self.training:
                feat = checkpoint(decoder, feat, use_reentrant=False)
            else:
                feat = decoder(feat)

        # Output projection + global residual
        output = self.ending(feat) + inp

        # Crop to original size and clamp to valid pixel range
        return output[:, :, :orig_H, :orig_W].clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("KBNet-7M: Testing model creation and forward pass")
    print("=" * 80)

    model = KBNet()

    total = sum(p.numel() for p in model.parameters())
    print(f"\nVerified total params: {total:,} ({total/1e6:.2f}M)")

    # Test with various input sizes
    for H, W in [(96, 96), (64, 64), (128, 128), (131, 97)]:
        x = torch.randn(1, 1, H, W)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape, f"Shape mismatch: input {x.shape} vs output {y.shape}"
        print(f"  Input {x.shape} -> Output {y.shape}  [OK]")

    print("\nAll tests passed.")
