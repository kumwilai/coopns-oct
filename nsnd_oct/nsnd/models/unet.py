"""
U-Net for Image Denoising (Ronneberger et al., 2015)

Classic architecture adapted for OCT denoising.
This serves as a strong supervised baseline to compare against NSND.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for OCT Denoising

    Args:
        in_channels: Number of input channels (1 for grayscale OCT)
        out_channels: Number of output channels (1 for grayscale OCT)
        features: Base number of features (default: 64)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """

    def __init__(self, in_channels=1, out_channels=1, features=64, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # Decoder
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16]

        # Decoder with skip connections
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]

        # Output
        output = self.outc(x)  # [B, 1, H, W]
        return output


class UNetSmall(nn.Module):
    """
    Smaller U-Net for faster training and fair comparison with NSND

    Features: 32 instead of 64 (closer to our NSND's capacity)
    """

    def __init__(self, in_channels=1, out_channels=1, features=32, bilinear=False):
        super(UNetSmall, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        factor = 2 if bilinear else 1
        self.down3 = Down(features * 4, features * 8 // factor)

        # Decoder
        self.up1 = Up(features * 8, features * 4 // factor, bilinear)
        self.up2 = Up(features * 4, features * 2 // factor, bilinear)
        self.up3 = Up(features * 2, features, bilinear)
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # [B, 32, H, W]
        x2 = self.down1(x1)   # [B, 64, H/2, W/2]
        x3 = self.down2(x2)   # [B, 128, H/4, W/4]
        x4 = self.down3(x3)   # [B, 256, H/8, W/8]

        # Decoder with skip connections
        x = self.up1(x4, x3)  # [B, 128, H/4, W/4]
        x = self.up2(x, x2)   # [B, 64, H/2, W/2]
        x = self.up3(x, x1)   # [B, 32, H, W]

        # Output
        output = self.outc(x)  # [B, 1, H, W]
        return output


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test U-Net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Standard U-Net
    unet = UNet(features=64).to(device)
    print(f"U-Net (features=64): {count_parameters(unet):,} parameters")

    # Small U-Net (fair comparison with NSND)
    unet_small = UNetSmall(features=32).to(device)
    print(f"U-Net Small (features=32): {count_parameters(unet_small):,} parameters")

    # Test forward pass
    x = torch.randn(1, 1, 64, 64).to(device)
    with torch.no_grad():
        y = unet(x)
        y_small = unet_small(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape (U-Net): {y.shape}")
    print(f"Output shape (U-Net Small): {y_small.shape}")
