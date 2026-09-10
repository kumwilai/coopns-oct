import torch
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.pw(self.act(self.dw(x)))


class SwinIRLite(nn.Module):
    """Lightweight SwinIR-style conv substitute for 64x64.

    Not a full transformer; aims to provide a competitive CNN baseline in the SwinIR spirit.
    """
    def __init__(self, in_ch=1, out_ch=1, base=48, depth=6):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base, 3, padding=1)
        blocks = []
        for _ in range(depth):
            blocks += [DWConv(base)]
        self.body = nn.Sequential(*blocks)
        self.tail = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x):
        h = self.head(x)
        b = self.body(h)
        y = self.tail(b)
        # Scale tanh output to prevent saturation
        return (x - 0.5 * torch.tanh(y)).clamp(0, 1)

