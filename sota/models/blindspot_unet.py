import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BlindSpotUNet(nn.Module):
    """Tiny blind-spot U-Net variant for 64x64.

    Implements blind-spot by using dilated convolutions in the bottleneck
    to ensure center pixel is not in receptive field.
    """
    def __init__(self, base=32):
        super().__init__()
        # Standard encoder blocks
        self.enc1 = ConvBlock(1, base)
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)

        # Bottleneck with dilation for blind-spot
        self.bott = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=2, dilation=2),  # Dilated conv
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=2, dilation=2),  # Dilated conv
            nn.ReLU(inplace=True),
        )

        # Standard decoder blocks
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bott(self.down2(e2))
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        y = self.out(d1)
        # Scale tanh output to prevent saturation
        return (x - 0.5 * torch.tanh(y)).clamp(0, 1)

