import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        y = self.conv2(self.act(self.conv1(x)))
        return x + y


class DRUNet(nn.Module):
    """Minimal DRUNet-like network for grayscale denoising (64x64-ready)."""
    def __init__(self, in_ch=1, out_ch=1, base=32, num_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x):
        h = self.head(x)
        b = self.body(h)
        y = self.tail(b)
        # Predict noise residual with small scaling to prevent saturation
        # Scale tanh output to [-0.5, 0.5] for stable residual learning
        return (x - 0.5 * torch.tanh(y)).clamp(0, 1)

