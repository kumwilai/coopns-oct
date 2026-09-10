"""
DnCNN-7M: Deep CNN for Image Denoising (~7M parameters)

Based on:
    "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
    K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang
    IEEE Transactions on Image Processing (TIP), 2017

Architecture:
    - Residual learning: model predicts noise residual, output = input - residual
    - 17 layers with 228 channels to reach ~7M parameters
    - Layer 1:     Conv(in_ch, 228, 3x3) + ReLU
    - Layers 2-16: Conv(228, 228, 3x3, bias=False) + BN + ReLU
    - Layer 17:    Conv(228, out_ch, 3x3)

Parameter count derivation (bias=False on middle conv, BN has weight+bias):
    First layer:   1 * 228 * 9 + 228                       =       2,280
    Middle (x15):  15 * (228 * 228 * 9 + 2 * 228)          = 7,024,680
    Last layer:    228 * 1 * 9 + 1                          =       2,053
    Total:                                                  ~ 7,029,013

Usage:
    model = DnCNN()
    output = model(x)  # x: [B, 1, H, W] -> output: [B, 1, H, W]
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DnCNN(nn.Module):
    """
    DnCNN with residual learning for grayscale image denoising.

    The network predicts the noise residual from the input, and the clean
    image is obtained by subtracting the predicted residual from the input:
        output = input - predicted_noise

    Args:
        in_channels:  Number of input channels (default: 1 for grayscale).
        out_channels: Number of output channels (default: 1 for grayscale).
        num_layers:   Total number of convolutional layers (default: 17).
        channels:     Number of intermediate feature channels (default: 228).
        use_checkpoint: Enable gradient checkpointing to reduce memory (default: True).
    """

    def __init__(self, in_channels=1, out_channels=1, num_layers=17, channels=228,
                 use_checkpoint=True):
        super().__init__()

        assert num_layers >= 3, "DnCNN requires at least 3 layers"
        self.use_checkpoint = use_checkpoint

        # ---- Layer 1: Conv + ReLU (no BN) ----
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # ---- Layers 2 to (num_layers - 1): Conv + BN + ReLU ----
        # Group middle layers into chunks of 5 blocks for gradient checkpointing.
        # With 15 middle blocks (num_layers=17), this gives 3 chunks of 5.
        num_middle = num_layers - 2
        chunk_size = 5
        middle_blocks = []
        for _ in range(num_middle):
            middle_blocks.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])

        # Split middle blocks into chunks, each wrapped in nn.Sequential
        self.middle_chunks = nn.ModuleList()
        layers_per_chunk = chunk_size * 3  # 3 nn.Modules per block (Conv+BN+ReLU)
        for i in range(0, len(middle_blocks), layers_per_chunk):
            self.middle_chunks.append(nn.Sequential(*middle_blocks[i:i + layers_per_chunk]))

        # ---- Last layer: Conv (no BN, no ReLU) ----
        self.tail = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=True)

        # Initialize weights using Kaiming normal (standard for ReLU networks)
        self._init_weights()

        # Gradient checkpointing re-runs forward during backward, causing BN
        # running stats to be updated twice per step (PyTorch #96136).
        # Adjust momentum so two updates equal one update at the nominal 0.1:
        #   1 - (1 - m_adj)^2 = 0.1  =>  m_adj = 1 - sqrt(0.9) ≈ 0.0513
        if self.use_checkpoint:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = 1.0 - math.sqrt(1.0 - m.momentum)

        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DnCNN -- Total params: {total_params:,} | Trainable: {trainable_params:,}")

    def _init_weights(self):
        """Initialize conv weights with Kaiming normal, BN with weight=1, bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Re-init tail conv: no ReLU follows, so use fan_in/linear (std ≈ 0.02)
        nn.init.kaiming_normal_(self.tail.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        """
        Forward pass with residual learning.

        Args:
            x: Noisy input image, shape [B, 1, H, W], values in [0, 1].

        Returns:
            Denoised image, shape [B, 1, H, W], clamped to [0, 1].
        """
        # First layer (no checkpointing — cheap)
        out = self.head(x)

        # Middle layers with optional gradient checkpointing
        for chunk in self.middle_chunks:
            if self.use_checkpoint and self.training:
                out = checkpoint(chunk, out, use_reentrant=False)
            else:
                out = chunk(out)

        # Last layer (no checkpointing — cheap)
        noise_residual = self.tail(out)
        return (x - noise_residual).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = DnCNN()
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # Verify parameter count breakdown
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
    print(f"Verified total params: {total:,}")
