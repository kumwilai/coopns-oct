"""DnCNN: Deep CNN Denoiser (Zhang et al., 2017)"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN architecture with configurable depth"""

    def __init__(self, in_channels=1, out_channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()

        layers = []

        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv (no activation)
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # Residual learning: output noise, subtract from input
        noise = self.dncnn(x)
        return x - noise


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
