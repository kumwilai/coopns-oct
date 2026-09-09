#!/usr/bin/env python3
"""Figure. The six clinical properties and where each one reports a failure.

One backbone output, then the six spatial failure maps that the rule layer reads.
Bright means the property is locally unsatisfied.
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "./revision")
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from figstyle import setup, save, INK, INK2, SEQ, COL2
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset

NAMES = {
    "P1": ("$P_1$", "edge continuity"),
    "P2": ("$P_2$", "local contrast"),
    "P3": ("$P_3$", "flat region smoothness"),
    "P4": ("$P_4$", "structural coherence"),
    "P5": ("$P_5$", "speckle conformity"),
    "P6": ("$P_6$", "layer visibility"),
}

def main():
    torch.set_num_threads(2)
    setup()
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)

    m = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name="nafnet",
        pretrained_backbone="checkpointpaper/nafnet_backbone.pth",
        hidden_channels=64)
    ck = torch.load("checkpointpaper/nafnet_pku37_cooperative.pth",
                    map_location="cpu", weights_only=False)
    sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
          for k, v in ck["model_state_dict"].items()}
    ms = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in ms and v.shape == ms[k].shape}, strict=False)
    m.eval()

    ds = PKU37Dataset("revision/pku37_subset40.jsonl", patch_size=0, is_train=False)
    idx = int(os.environ.get("IMG_INDEX", "3"))
    noisy = ds[idx]["noisy"].unsqueeze(0)
    with torch.no_grad():
        b, unc = m.backbone(noisy)
        pred = m.corrector.predicates(b, noisy)

    base = b[0, 0].numpy()

    fig = plt.figure(figsize=(7.16, 2.95))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.42, 1, 1, 1],
                  wspace=0.05, hspace=0.05,
                  left=0.005, right=0.995, top=0.995, bottom=0.135)

    axb = fig.add_subplot(gs[:, 0])
    axb.imshow(base, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axb.set_xticks([]); axb.set_yticks([])
    for sp in axb.spines.values():
        sp.set_edgecolor(INK2); sp.set_linewidth(0.6)
    axb.text(0.028, 0.965, "backbone output", transform=axb.transAxes,
             fontsize=7.6, color=INK, va="top", ha="left",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.6))

    im = None
    for k, key in enumerate(["P1", "P2", "P3", "P4", "P5", "P6"]):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(gs[r, c + 1])
        fm = pred[key]["failure_map"]
        fm = fm[0, 0].numpy() if fm.dim() == 4 else fm.squeeze().numpy()
        im = ax.imshow(fm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        sym, desc = NAMES[key]
        score = pred[key]["score"]
        score = float(score.mean()) if torch.is_tensor(score) else float(score)
        ax.text(0.035, 0.955, f"{sym} {desc}", transform=ax.transAxes,
                fontsize=7.0, color=INK, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.4))
        ax.text(0.035, 0.045, f"score {score:.2f}", transform=ax.transAxes,
                fontsize=6.6, color=INK, va="bottom", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2))

    cax = fig.add_axes([0.435, 0.055, 0.30, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["property satisfied", "property unsatisfied"])
    cb.ax.tick_params(labelsize=6.6, length=0, colors=INK2, pad=1.5)

    save(fig, "fig_predicate_maps")

if __name__ == "__main__":
    main()
