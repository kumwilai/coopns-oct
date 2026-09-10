#!/usr/bin/env python3
"""Figure 1. The six stages of the method, drawn to match the code."""
import sys
# Resolve the project root from this file rather than from a fixed path, so the
# scripts run unchanged on any machine and from any working directory.
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "revision"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import setup, save, INK, INK2, GREY

S = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#4a3aa7", 5: "#eb6834", 6: "#0f7a53"}

ROW_T, ROW_M, ROW_B = 0.700, 0.420, 0.140
H = 0.170
COL = {"a": 0.062, "b": 0.240, "c": 0.418, "d": 0.596, "e": 0.774}
W = 0.148


def box(ax, x, y, title, sub, stage, w=W, h=H, tag=None):
    for lw, fc, al, z in ((0.9, S[stage], 0.10, 2), (0.9, "none", 1.0, 3)):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     linewidth=lw, edgecolor=S[stage], facecolor=fc, alpha=al, zorder=z))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=7.3,
            color=INK, zorder=4)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=6.6,
            color=INK2, zorder=4)
    if tag is not None:
        ax.text(x + 0.004, y + h + 0.005, str(tag), ha="center", va="center", fontsize=5.9,
                color="white", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.17", facecolor=S[stage], edgecolor="none"))


def arrow(ax, pts, color=GREY, ls="-", lw=0.9):
    """Orthogonal polyline with a head on the last segment."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, linewidth=lw, linestyle=ls, zorder=1,
                solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7,
                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                 shrinkA=0, shrinkB=1.5))


def main():
    setup()
    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    boxes = [
        ("a", ROW_M, "frozen backbone", r"$\mathbf{b}=f_\theta(\mathbf{y})$", 1),
        ("a", ROW_T, "confidence head", r"$\mathbf{c}$", 1),
        ("b", ROW_T, "gain and edge", r"$\Delta,\ \delta_{\mathrm{e}},\ \lambda$", 2),
        ("b", ROW_B, "six properties", r"$s_i,\ \mathbf{m}_i$", 3),
        ("c", ROW_M, "five fuzzy rules", r"allocation $\mathbf{a}$", 4),
        ("d", ROW_M, "bound and gate", r"candidate $\mathbf{q}$", 5),
        ("e", ROW_M, "three constraints", r"blend weight $w$", 6),
    ]
    tags = {("a", ROW_M): 1, ("a", ROW_T): 1, ("b", ROW_T): 2, ("b", ROW_B): 3,
            ("c", ROW_M): 4, ("d", ROW_M): 5, ("e", ROW_M): 6}
    for col, row, t, sub, st in boxes:
        box(ax, COL[col], row, t, sub, st, tag=tags.get((col, row)))

    m = lambda r: r + H / 2
    x0 = lambda c: COL[c]
    x1 = lambda c: COL[c] + W

    ax.text(0.022, m(ROW_M), r"$\mathbf{y}$", fontsize=10, color=INK, ha="center", va="center")
    ax.text(0.022, m(ROW_M) - 0.085, "noisy", fontsize=6.3, color=INK2, ha="center", va="center")
    ax.text(0.965, m(ROW_M), r"$\hat{\mathbf{x}}$", fontsize=10, color=INK, ha="center", va="center")
    ax.text(0.965, m(ROW_M) - 0.085, "corrected", fontsize=6.3, color=INK2, ha="center", va="center")

    arrow(ax, [(0.040, m(ROW_M)), (x0("a"), m(ROW_M))])
    arrow(ax, [(x0("a") + W / 2, ROW_M + H), (x0("a") + W / 2, ROW_T)])
    # backbone feeds the two middle column blocks
    arrow(ax, [(x1("a"), m(ROW_M)), (0.212, m(ROW_M)), (0.212, m(ROW_T)), (x0("b"), m(ROW_T))])
    arrow(ax, [(x1("a"), m(ROW_M)), (0.212, m(ROW_M)), (0.212, m(ROW_B)), (x0("b"), m(ROW_B))])
    # into the rule layer
    arrow(ax, [(x1("b"), m(ROW_T)), (0.412, m(ROW_T)), (0.412, m(ROW_M) + 0.030),
               (x0("c"), m(ROW_M) + 0.030)], color=S[2])
    arrow(ax, [(x1("b"), m(ROW_B)), (0.412, m(ROW_B)), (0.412, m(ROW_M) - 0.030),
               (x0("c"), m(ROW_M) - 0.030)], color=S[3])
    # confidence into the rule layer, routed above everything
    arrow(ax, [(x1("a"), m(ROW_T)), (0.232, m(ROW_T)), (0.222, 0.940), (0.452, 0.940),
               (0.470, ROW_M + H)], color=S[1])
    arrow(ax, [(x1("c"), m(ROW_M)), (x0("d"), m(ROW_M))])
    arrow(ax, [(x1("d"), m(ROW_M)), (x0("e"), m(ROW_M))])
    arrow(ax, [(x1("e"), m(ROW_M)), (0.948, m(ROW_M))])
    # properties are re measured on the candidate for the constraint check
    arrow(ax, [(x0("b") + W / 2, ROW_B), (x0("b") + W / 2, 0.052), (x0("e") + W * 0.32, 0.052),
               (x0("e") + W * 0.32, ROW_M)], color=S[3], ls=(0, (2.4, 1.8)))
    # backbone anchor for the blend
    arrow(ax, [(x0("a") + W / 2, ROW_M), (x0("a") + W / 2, 0.052)], color=GREY, ls=(0, (2.4, 1.8)))
    ax.plot([x0("a") + W / 2, x0("e") + W * 0.68], [0.052, 0.052], color=GREY,
            linewidth=0.9, linestyle=(0, (2.4, 1.8)), zorder=0)
    arrow(ax, [(x0("e") + W * 0.68, 0.052), (x0("e") + W * 0.68, ROW_M)], color=GREY,
          ls=(0, (2.4, 1.8)))
    ax.text(0.505, 0.075, "backbone anchor and re measured properties", fontsize=6.2,
            color=INK2, ha="center", va="bottom")

    save(fig, "fig_architecture")


if __name__ == "__main__":
    main()
