import torch
import torch.nn as nn
from .drunet import DRUNet


class Speckle2Speckle(nn.Module):
    """Speckle2Speckle-style model: uses a DRUNet backbone; training should use two independent noisy targets.
    Inference is a single forward pass.
    """
    def __init__(self, base=32):
        super().__init__()
        self.net = DRUNet(in_ch=1, out_ch=1, base=base, num_blocks=4)

    def forward(self, x):
        return self.net(x)

