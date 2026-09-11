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

# Geometry. One spine through the middle row, the head above it, the properties below it.
ROW_M = 0.420
PITCH = 0.265
ROW_T, ROW_B = ROW_M + PITCH, ROW_M - PITCH
H = 0.170
W = 0.162
GAP = 0.0215
COL = {k: 0.050 + i * (W + GAP) for i, k in enumerate("abcde")}
DASH = (0, (2.4, 1.8))

# Axes limits and size. The axes is the drawn area, so the limits are set to the
# content and the figure is sized to keep the old scale of 400 pt per x unit and
# 119 pt per y unit. Width stays at 403.8 pt, height falls from 122 pt to 102 pt.
XLIM = (-0.010, 1.015)
YLIM = (0.070, 0.905)
FIGSIZE = (5.568, 0.835 * 119.2 / 72)


def box(ax, x, y, title, sub, stage, w=W, h=H, tag=None):
    for lw, fc, al, z in ((0.9, S[stage], 0.10, 2), (0.9, "none", 1.0, 3)):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     linewidth=lw, edgecolor=S[stage], facecolor=fc, alpha=al, zorder=z))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=6.8,
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
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.axis("off")

    boxes = [
        ("a", ROW_M, "frozen backbone", r"$\mathbf{b}=f_\theta(\mathbf{y})$", 1),
        ("a", ROW_T, "cooperation head", r"$\mathbf{c},\ \ \mathbf{u}=\mathbf{1}-\mathbf{c}$", 1),
        ("b", ROW_M, "two correctors", r"$\Delta,\ \delta_{\mathrm{e}},\ \lambda$", 2),
        ("b", ROW_B, "six properties", r"$s_i,\ \mathbf{m}_i$", 3),
        ("c", ROW_M, "five rules", r"allocation $\mathbf{a}$", 4),
        ("d", ROW_M, "bound and gate", r"candidate $\mathbf{q}$", 5),
        ("e", ROW_M, "safety decision", r"blend weight $w$", 6),
    ]
    tags = {("a", ROW_M): 1, ("a", ROW_T): 1, ("b", ROW_M): 2, ("b", ROW_B): 3,
            ("c", ROW_M): 4, ("d", ROW_M): 5, ("e", ROW_M): 6}
    for col, row, t, sub, st in boxes:
        box(ax, COL[col], row, t, sub, st, tag=tags.get((col, row)))

    m = lambda r: r + H / 2
    x0 = lambda c: COL[c]
    x1 = lambda c: COL[c] + W
    xc = lambda c: COL[c] + W / 2

    ax.text(0.018, m(ROW_M), r"$\mathbf{y}$", fontsize=10, color=INK, ha="center", va="center")
    ax.text(0.018, m(ROW_M) - 0.085, "noisy", fontsize=6.3, color=INK2, ha="center", va="center")
    ax.text(0.972, m(ROW_M), r"$\hat{\mathbf{x}}$", fontsize=10, color=INK, ha="center", va="center")
    ax.text(0.972, m(ROW_M) - 0.085, "corrected", fontsize=6.3, color=INK2, ha="center", va="center")

    # the spine, stages one, two, four, five, six, each arrow in the colour of its source
    arrow(ax, [(0.032, m(ROW_M)), (x0("a"), m(ROW_M))])
    arrow(ax, [(x1("a"), m(ROW_M)), (x0("b"), m(ROW_M))])
    arrow(ax, [(x1("b"), m(ROW_M)), (x0("c"), m(ROW_M))], color=S[2])
    arrow(ax, [(x1("c"), m(ROW_M)), (x0("d"), m(ROW_M))], color=S[4])
    arrow(ax, [(x1("d"), m(ROW_M)), (x0("e"), m(ROW_M))], color=S[5])
    arrow(ax, [(x1("e"), m(ROW_M)), (0.958, m(ROW_M))])
    # backbone features up into the head
    arrow(ax, [(xc("a"), ROW_M + H), (xc("a"), ROW_T)])
    # the cooperation map into the rule layer, one bend, entering from above
    arrow(ax, [(x1("a"), m(ROW_T)), (xc("c"), m(ROW_T)), (xc("c"), ROW_M + H)], color=S[1])
    # backbone output down one trunk. Solid branch into the properties, dashed
    # continuation is the anchor the safety decision blends back to.
    Y_IN = m(ROW_B) + 0.035
    Y_ANCHOR = ROW_B - 0.065
    arrow(ax, [(xc("a"), ROW_M), (xc("a"), Y_IN), (x0("b"), Y_IN)])
    ax.plot([xc("a")], [Y_IN], marker="o", markersize=2.2, color=GREY, zorder=2)
    arrow(ax, [(xc("a"), Y_IN), (xc("a"), Y_ANCHOR), (x0("e") + W * 0.68, Y_ANCHOR),
               (x0("e") + W * 0.68, ROW_M)], color=GREY, ls=DASH)
    # failure maps up into the correctors, for the strength map
    arrow(ax, [(xc("b"), ROW_B + H), (xc("b"), ROW_M)], color=S[3])
    # scores into the rule layer, one bend, entering from below
    arrow(ax, [(x1("b"), Y_IN), (xc("c"), Y_IN), (xc("c"), ROW_M)], color=S[3])
    # properties re measured on the candidate for the constraint check, own channel
    Y_RE = m(ROW_B) - 0.045
    arrow(ax, [(x1("b"), Y_RE), (x0("e") + W * 0.32, Y_RE), (x0("e") + W * 0.32, ROW_M)],
          color=S[3], ls=DASH)
    ax.text(0.665, Y_RE + 0.012, r"$s_i$ re measured on $\mathbf{q}$", fontsize=6.2,
            color=INK2, ha="center", va="bottom")
    ax.text(0.600, Y_ANCHOR + 0.012, r"$\mathbf{b}$ kept as the anchor", fontsize=6.2,
            color=INK2, ha="center", va="bottom")

    save(fig, "fig_architecture")


if __name__ == "__main__":
    main()
