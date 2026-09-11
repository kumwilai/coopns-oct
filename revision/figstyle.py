"""Shared figure style for the manuscript.

One look for every figure. Colours come from a palette that was checked with the
colour vision deficiency validator, and every series also carries a second cue,
either a marker shape, a line style or a hatch, so the figures survive being
printed in grey.

Usage
    from figstyle import setup, C, save, GREY
    setup()
    fig, ax = plt.subplots(figsize=COL1)
    ...
    save(fig, "fig_name")
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# IEEE two column geometry, in inches
COL1 = (3.45, 2.35)      # single column
COL1_TALL = (3.45, 3.10)
COL2 = (7.16, 2.60)      # full width
COL2_TALL = (7.16, 4.20)

# Categorical slots. Checked with the validator, light surface, all pairs.
C = {
    "nafnet": "#2a78d6",   # blue
    "dncnn":  "#eb6834",   # orange
    "swinir": "#1baf7a",   # aqua
    "kbnet":  "#4a3aa7",   # violet
}
ORDER = ["nafnet", "dncnn", "swinir", "kbnet"]
LABEL = {"nafnet": "NAFNet", "dncnn": "DnCNN", "swinir": "SwinIR", "kbnet": "KBNet"}

# Second cue for grey printing
MARKER = {"nafnet": "o", "dncnn": "s", "swinir": "^", "kbnet": "D"}
DASH = {"nafnet": (None, None), "dncnn": (4, 1.6), "swinir": (1.4, 1.4), "kbnet": (5.5, 1.6, 1.2, 1.6)}
HATCH = {"nafnet": "", "dncnn": "///", "swinir": "...", "kbnet": "xxx"}

# Ink. Text never wears a series colour.
INK = "#0b0b0b"
INK2 = "#52514e"
GREY = "#8d8c88"
GRID = "#dcdbd6"
SURFACE = "#ffffff"

# Sequential ramp for magnitude, one hue light to dark
SEQ = ["#dbe8f8", "#a9c8ee", "#6fa3e0", "#2a78d6", "#1b559b", "#123a6b"]
# Diverging pair for signed change, warm and cool with a neutral middle
DIV_NEG = "#eb6834"
DIV_MID = "#e8e7e2"
DIV_POS = "#2a78d6"

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        "legend.handletextpad": 0.5,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tidy(ax, grid_axis="y", minor=False):
    """Recessive axes. Only the spines that carry meaning stay."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    if minor:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    return ax


def zero_line(ax, y=0.0):
    """A reference line at zero, recessive, behind the data."""
    ax.axhline(y, color=INK2, linewidth=0.7, zorder=1)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote {os.path.join(FIGDIR, name)}.pdf")
