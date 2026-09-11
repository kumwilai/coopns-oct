#!/usr/bin/env python3
"""Show that the frozen backbone used to drift, and that it no longer does.

Freezing the parameters of a backbone does not stop a BatchNorm layer from
updating its running mean and variance, because those are buffers. DnCNN is the
only backbone here with BatchNorm. Before the revision the wrapper was trained
with model.train() and nothing else, so the reference that every clinical delta
is measured against moved while the wrapper was being fitted to it.

This script runs the same batch through the backbone twice in training mode,
once without and once with freeze_backbone_norm_stats, and prints how far the
running statistics and the output moved between the two passes. CPU is enough
and it takes a few seconds.

usage
  python revision/verify_norm_freeze.py [--backbone dncnn]
"""
import argparse, os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative, freeze_backbone_norm_stats


def bn_state(bb):
    parts = []
    for m in bb.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            parts += [m.running_mean.detach().clone().flatten(), m.running_var.detach().clone().flatten()]
    return torch.cat(parts) if parts else torch.zeros(0)


def two_passes(model, x, freeze):
    model.train()
    if freeze:
        n = freeze_backbone_norm_stats(model)
    bb = model.backbone.backbone
    with torch.no_grad():
        s0 = bn_state(bb); y0 = bb(x)
        s1 = bn_state(bb); y1 = bb(x)
    return (s1 - s0).abs().max().item(), (y1 - y0).abs().max().item(), s0.numel() // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dncnn")
    args = ap.parse_args()
    weights = f"checkpointpaper/{args.backbone}_backbone.pth"
    torch.manual_seed(0)
    x = torch.rand(2, 1, 96, 96)
    for freeze in (False, True):
        model = NeuroSymbolicDenoiserV8Cooperative(backbone_name=args.backbone,
                                                   pretrained_backbone=weights, hidden_channels=64)
        d_stats, d_out, n_bn = two_passes(model, x, freeze)
        label = "with    freeze_backbone_norm_stats" if freeze else "without freeze_backbone_norm_stats"
        print(f"{label}: {n_bn} BatchNorm layers, running stats moved by {d_stats:.3e}, "
              f"backbone output moved by {d_out:.3e} between two identical passes")
    print("The fix is working if the second line shows 0.000e+00 for both quantities.")


if __name__ == "__main__":
    main()
