"""
NAFNet-7M: Nonlinear Activation Free Network for Image Restoration (~7M parameters)

Based on: "Simple Baselines for Image Restoration" (Chen et al. 2022)
https://github.com/megvii-research/NAFNet

7M configuration: width=40, enc=[1,1,1,1], dec=[1,1,1,1], mid=1
Channels: [40, 80, 160, 320], bottleneck=640
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm with manual backward for better performance."""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (published NAFNet uses this)."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """SimpleGate: Split channels and multiply (replaces nonlinear activations)."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet Block with SimpleGate and Simplified Channel Attention (SCA).

    Two sub-layers with learnable residual scaling (beta, gamma):
    1. Depthwise conv + SimpleGate + SCA + projection
    2. FFN + SimpleGate + projection
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand

        # Depthwise conv branch
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )

        # SimpleGate
        self.sg = SimpleGate()

        # LayerNorm
        self.norm1 = LayerNorm2d(c)

        # Feedforward Network
        self.conv4 = nn.Conv2d(c, ffn_expand * c, 1)
        self.conv5 = nn.Conv2d(ffn_expand * c // 2, c, 1)

        # LayerNorm for FFN
        self.norm2 = LayerNorm2d(c)

        # Beta and Gamma for learnable residual scaling (init to zeros per official impl)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # Depthwise conv + SimpleGate + Channel Attention
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        # First residual
        y = inp + x * self.beta

        # Feedforward + SimpleGate
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        # Second residual
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet-7M: Nonlinear Activation Free Network for grayscale image denoising.

    4-level UNet with NAFBlocks, SimpleGate activation, and learnable residual scaling.
    Uses element-wise skip connections and global residual learning.
    PixelShuffle for upsampling, strided conv for downsampling.

    Automatically pads input to a multiple of 2^num_levels (=16) for alignment.

    Args:
        img_channel: Input/output channels (default: 1).
        width: Base channel width (default: 40).
        middle_blk_num: Number of blocks in bottleneck (default: 1).
        enc_blk_nums: Blocks per encoder level (default: [1, 1, 1, 1]).
        dec_blk_nums: Blocks per decoder level (default: [1, 1, 1, 1]).
    """
    def __init__(
        self,
        img_channel=1,
        width=40,
        middle_blk_num=1,
        enc_blk_nums=(1, 1, 1, 1),
        dec_blk_nums=(1, 1, 1, 1),
    ):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NAFNet-7M | Total params: {total_params:,} ({total_params/1e6:.2f}M) | "
              f"Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self._check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W].clamp(0.0, 1.0)

    def _check_image_size(self, x):
        """Pad image to be divisible by padder_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("NAFNet-7M: Testing model creation and forward pass")
    print("=" * 80)

    model = NAFNet()

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
