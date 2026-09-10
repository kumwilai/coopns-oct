import torch
import torch.nn as nn
from adaptive_oct_denoise import NAFBackbone


class NAFNetGray(nn.Module):
    """Wrap the NAFBackbone from adaptive_oct_denoise for grayscale 64x64 denoising."""
    def __init__(self, base_channels=32):
        super().__init__()
        self.backbone = NAFBackbone(in_channels=1, out_channels=1, base_channels=base_channels)

    def forward(self, x):
        y = self.backbone(x)
        return (x - torch.tanh(y)).clamp(0, 1)

