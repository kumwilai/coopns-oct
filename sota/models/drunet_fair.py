"""
DRUNet: Deep Residual U-Net for Image Denoising
Based on: "Plug-and-Play Image Restoration with Deep Denoiser Prior" (Zhang et al. 2021)

Fair configuration for comparison with N2V+CASA (3.46M params).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and ReLU (as in published DRUNet)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class DRUNet(nn.Module):
    """
    DRUNet with noise level conditioning (published architecture).

    Args:
        in_channels: Input channels (1 for grayscale)
        out_channels: Output channels (1 for grayscale)
        nc: Channel counts at each level [64, 128, 256, 512]
        nb: Number of residual blocks at each level [2, 2, 2, 2]
        act_mode: Activation ('R' for ReLU, 'L' for LeakyReLU)

    Fair config (3.5M params, matches N2V+CASA):
        nc=[64, 128, 256, 512], nb=[2, 2, 2, 2]
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        nc=[64, 128, 256, 512],  # Channels at each U-Net level
        nb=[2, 2, 2, 2],          # Blocks at each level
        act_mode='R',
        use_noise_level=True,     # Noise level conditioning (DRUNet feature)
        use_tanh=False            # If True, squash residual (non-original)
    ):
        super().__init__()
        self.use_noise_level = use_noise_level
        self.use_tanh = use_tanh
        self.depth = len(nc)

        # If noise level conditioning, concatenate noise map with input
        input_ch = in_channels + 1 if use_noise_level else in_channels

        # Encoder
        self.m_head = nn.Conv2d(input_ch, nc[0], 3, padding=1)

        self.m_down = nn.ModuleList()
        for i in range(self.depth):
            blocks = nn.Sequential(*[ResidualBlock(nc[i]) for _ in range(nb[i])])
            self.m_down.append(blocks)

        # Downsampling layers
        self.m_downsample = nn.ModuleList()
        for i in range(self.depth - 1):
            self.m_downsample.append(
                nn.Conv2d(nc[i], nc[i+1], 2, stride=2)
            )

        # Note: No separate bottleneck - last encoder level (m_down[-1]) IS the bottleneck
        # This matches standard U-Net architecture

        # Decoder
        self.m_up = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            # After skip connection, we have nc[i-1] channels
            blocks = nn.Sequential(*[ResidualBlock(nc[i-1]) for _ in range(nb[i])])
            self.m_up.append(blocks)

        # Upsampling layers
        self.m_upsample = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            self.m_upsample.append(
                nn.ConvTranspose2d(nc[i], nc[i-1], 2, stride=2)
            )

        # Output
        self.m_tail = nn.Conv2d(nc[0], out_channels, 3, padding=1)

    def forward(self, x, noise_level=None):
        """
        Args:
            x: Input image [B, 1, H, W]
            noise_level: Noise level sigma [B, 1] or scalar (optional)

        Returns:
            Denoised image [B, 1, H, W]
        """
        B, C, H, W = x.shape

        # Add noise level map if using noise conditioning
        if self.use_noise_level:
            if noise_level is None:
                # Auto-estimate from input variance (simple heuristic)
                noise_level = torch.std(x, dim=(2, 3), keepdim=True).mean()

            # Create noise level map (constant across spatial dimensions)
            if isinstance(noise_level, (int, float)):
                noise_level = torch.full((B, 1, 1, 1), noise_level, device=x.device)
            elif noise_level.dim() == 2:  # [B, 1]
                noise_level = noise_level.view(B, 1, 1, 1)

            noise_map = noise_level.expand(B, 1, H, W)
            x_in = torch.cat([x, noise_map], dim=1)  # [B, 2, H, W]
        else:
            x_in = x

        # Encoder
        h = self.m_head(x_in)

        skip_connections = []
        for i in range(self.depth - 1):
            h = self.m_down[i](h)
            skip_connections.append(h)
            h = self.m_downsample[i](h)

        # Bottleneck (last encoder level)
        h = self.m_down[-1](h)

        # Decoder with skip connections
        for i in range(self.depth - 1):
            h = self.m_upsample[i](h)
            # Crop/pad to match skip connection size
            if h.shape[-2:] != skip_connections[-(i+1)].shape[-2:]:
                h = F.interpolate(h, size=skip_connections[-(i+1)].shape[-2:],
                                 mode='bilinear', align_corners=False)
            h = h + skip_connections[-(i+1)]  # Skip connection
            h = self.m_up[i](h)

        # Output: predict noise residual
        noise = self.m_tail(h)
        if self.use_tanh:
            noise = torch.tanh(noise)

        # Residual learning: clean = noisy - noise
        out = x - noise
        return out.clamp(0.0, 1.0)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test different configs to match N2V+CASA (3.46M params)
    configs = [
        ("tiny", [32, 64, 128], [2, 2, 2]),
        ("small", [48, 96, 192], [2, 2, 2]),
        ("fair", [64, 128, 256], [2, 2, 2]),  # 3.5M params - fair comparison
        ("medium", [48, 96, 192, 384], [2, 2, 2, 2]),
    ]

    print("DRUNet Configurations (targeting 3.5M params):")
    print("-" * 60)
    for name, nc, nb in configs:
        model = DRUNet(nc=nc, nb=nb)
        params = count_parameters(model)
        print(f"{name:10s} | nc={nc} | {params:,} params ({params/1e6:.2f}M)")

    # Test forward pass with fair config
    print("\n" + "="*60)
    print("Testing Forward Pass:")
    print("="*60)
    model_fair = DRUNet(nc=[64, 128, 256], nb=[2, 2, 2])
    x = torch.randn(1, 1, 64, 64)
    y = model_fair(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Parameters:   {count_parameters(model_fair):,} ({count_parameters(model_fair)/1e6:.2f}M)")
