#!/usr/bin/env python3
"""Figure. Ablations, rule-constant sensitivity and safety, in one three-panel figure.

Left, the leave-one-out ablation of each clinical property, on the full PKU37 test
set. Middle, how far the result moves when the constants that seed the fuzzy rule
layer are perturbed. Right, four before/after safety diagnostics for the corrected
NAFNet backbone. Every panel reads pre-computed summary statistics from JSON files
under outputs/revision/ -- there is no model loading and nothing runs on a GPU.
"""
import argparse, json, os, sys

# Resolve the project root from this file rather than from a fixed path, so the
# scripts run unchanged on any machine and from any working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "revision"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from figstyle import setup, save, tidy, INK, INK2, GREY, DIV_NEG, DIV_MID, DIV_POS

PROPS = ["P1", "P2", "P3", "P4", "P5", "P6"]
# Canonical measure order; a file may carry a subset of these, never more.
MEASURE_ORDER = ["PSNR (dB)", "CNR", "TCI", "EPI", "BS", "ENL", "SNR"]


def _load(path):
    """Read one JSON result file, or return None and warn if it is absent."""
    if not os.path.exists(path):
        print(f"warning: missing {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def _summary(d):
    return d["summary"] if d else None


def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=7.3, color=INK2)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def panel_a(ax, results_dir):
    """Leave-one-out: change in each measure when one clinical property is dropped."""
    ref = _summary(_load(os.path.join(results_dir, "lopo_none.json")))
    drops = {}
    for p in PROPS:
        s = _summary(_load(os.path.join(results_dir, f"lopo_drop_{p}.json")))
        if s is not None:
            drops[p] = s

    if ref is None or not drops:
        _empty(ax, "leave-one-out\nnot available")
        return None

    measures = [m for m in MEASURE_ORDER if m in ref]
    rows = list(drops.keys())
    mat = np.full((len(rows), len(measures)), np.nan)
    for i, p in enumerate(rows):
        for j, m in enumerate(measures):
            if m in drops[p]:
                mat[i, j] = drops[p][m]["delta_mean"] - ref[m]["delta_mean"]
            else:
                print(f"warning: {p} missing measure '{m}'", file=sys.stderr)

    # A heatmap, not a grouped bar chart: up to 6 properties x 7 measures is 42 bars,
    # unreadable at IEEE single-column width; a colour grid with printed values scales.
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    cmap.set_bad(color="#f2f1ee")
    scale = np.nanmax(np.abs(mat), axis=0)
    scale[~(scale > 0)] = 1.0
    norm = mat / scale

    im = ax.imshow(norm, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels([m.replace(" (dB)", "") for m in measures],
                        rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"${p[0]}_{p[1]}$ dropped" for p in rows])
    for i in range(len(rows)):
        for j in range(len(measures)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            fmt = f"{v:+.2f}" if abs(v) < 10 else f"{v:+.1f}"
            colour = "white" if abs(norm[i, j]) > 0.55 else INK
            ax.text(j, i, fmt, ha="center", va="center", fontsize=5.0, color=colour)
    ax.tick_params(length=0)
    ax.set_title("property removed", fontsize=7.3, color=INK, pad=3)
    return im


# The six clinical measures. Fidelity is deliberately not one of them: it is the
# cost being traded, not a measure of clinical quality, and including it here would
# make the worst case line track the fidelity loss on every point and say nothing
# about the rule constants, which is what this panel exists to show.
CLINICAL = ["CNR", "TCI", "EPI", "BS", "ENL", "SNR"]


def _worst(summary):
    """The minimum delta_mean over the six clinical measures."""
    if not summary:
        return None
    vals = [summary[k]["delta_mean"] for k in CLINICAL if k in summary]
    return min(vals) if vals else None


def panel_b(fig, cell, results_dir):
    """Sensitivity of the worst clinical delta to the rule-layer constants."""
    ref = _summary(_load(os.path.join(results_dir, "fuzz_ref.json")))
    ref_worst = _worst(ref)

    sweeps = [
        ("base", [-1.0, -0.5, 0.5, 1.0], "base alloc."),
        ("usecorr", [2.0, 3.0, 5.0, 6.0], "use_corrector"),
        ("boost", [1.0, 2.0, 4.0, 5.0], "boost_failing"),
    ]
    points = {}
    all_vals = [] if ref_worst is None else [ref_worst]
    for name, xs, _ in sweeps:
        pts = []
        for x in xs:
            s = _summary(_load(os.path.join(results_dir, f"fuzz_{name}_{x:.1f}.json")))
            if s is not None:
                w = _worst(s)
                pts.append((x, w))
                all_vals.append(w)
        points[name] = pts

    tnorm = {}
    for key, fname in [("product", "fuzz_tnorm_product.json"),
                        ("minimum", "fuzz_tnorm_minimum.json")]:
        s = _summary(_load(os.path.join(results_dir, fname)))
        if s is not None:
            tnorm[key] = _worst(s)
            all_vals.append(tnorm[key])

    if not all_vals:
        inner = cell.subgridspec(1, 1)
        ax = fig.add_subplot(inner[0, 0])
        _empty(ax, "sensitivity sweep\nnot available")
        return

    # A fixed, generously padded y range across all four rows -- the point of this
    # panel is that the worst measure barely moves, so we must not auto-zoom onto it.
    span = max(all_vals) - min(all_vals)
    pad = max(span * 0.7, 0.5)
    ylim = (min(all_vals) - pad, max(all_vals) + pad)

    inner = cell.subgridspec(4, 1, hspace=1.05, height_ratios=[1, 1, 1, 0.9])
    axes = [fig.add_subplot(inner[i, 0]) for i in range(4)]
    yrange = ylim[1] - ylim[0]
    yticks = [round(ylim[0] + yrange * p, 2) for p in (0.2, 0.8)]
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)

    for ax, (name, xs, label) in zip(axes[:3], sweeps):
        pts = points[name]
        if pts:
            px, py = zip(*pts)
            ax.plot(px, py, marker="o", color=INK, linewidth=1.1, markersize=3.2, zorder=3)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6.3, color=INK2)
        if ref_worst is not None:
            ax.axhline(ref_worst, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        # Label sits inside the plot area rather than as an axes title, so it never
        # competes for vertical room with the panel's own title above the stack.
        ax.text(0.03, 0.93, label, transform=ax.transAxes, va="top", ha="left",
                fontsize=6.1, color=INK2, bbox=label_box)
        tidy(ax)
        ax.tick_params(labelsize=5.6)

    axes[0].set_title("worst clinical $\\Delta$ vs. constant", fontsize=7.3, color=INK, pad=4)
    if ref_worst is not None:
        from matplotlib.lines import Line2D
        proxy = Line2D([0], [0], color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
        axes[0].legend([proxy], ["reference"], loc="lower right", fontsize=5.3,
                       frameon=False, handlelength=1.4, borderaxespad=0.15)

    axt = axes[3]
    keys = list(tnorm.keys())
    for i, k in enumerate(keys):
        axt.scatter([i], [tnorm[k]], color=INK, marker="D", s=13, zorder=3)
    axt.set_xticks(range(len(keys)))
    axt.set_xticklabels(keys, fontsize=5.6)
    if not keys:
        axt.text(0.5, 0.5, "missing", ha="center", va="center",
                 transform=axt.transAxes, fontsize=6.3, color=INK2)
    if ref_worst is not None:
        axt.axhline(ref_worst, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)
    axt.set_ylim(*ylim)
    axt.set_xlim(-0.6, max(1.6, len(keys) - 0.4))
    axt.set_yticks(yticks)
    axt.text(0.03, 0.93, "t-norm (alt.)", transform=axt.transAxes, va="top", ha="left",
             fontsize=6.1, color=INK2, bbox=label_box)
    tidy(axt)
    axt.tick_params(labelsize=5.6)


def panel_c(ax, results_dir):
    """Paired before/after safety diagnostics, normalised to the backbone value."""
    d = _load(os.path.join(results_dir, "diagnostics_nafnet.json"))
    if d is None or "summary" not in d:
        _empty(ax, "diagnostics not available")
        return
    s = d["summary"]

    # (backbone key, corrected key, axis label, which direction is an improvement)
    pairs = [
        ("invented_edge_rate_backbone", "invented_edge_rate_corrected", "invented\nedges", "lower"),
        ("weak_retention_backbone", "weak_retention_corrected", "weak\nretention", "higher"),
        ("ilm_shift_backbone", "ilm_shift_corrected", "ILM\nshift", "lower"),
        ("rpe_shift_backbone", "rpe_shift_corrected", "RPE\nshift", "lower"),
    ]

    labels, ratios, colours = [], [], []
    for bk, ck, label, better in pairs:
        if bk not in s or ck not in s:
            print(f"warning: diagnostics missing '{bk}' or '{ck}'", file=sys.stderr)
            continue
        b, c = s[bk]["mean"], s[ck]["mean"]
        if not b:
            continue
        ratio = c / b
        improved = (ratio < 1.0) if better == "lower" else (ratio > 1.0)
        labels.append(label)
        ratios.append(ratio)
        colours.append(DIV_POS if improved else DIV_NEG)

    if not labels:
        _empty(ax, "diagnostics not available")
        return

    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, [1.0] * len(labels), width=w, color=GREY, zorder=3, label="backbone")
    ax.bar(x + w / 2, ratios, width=w, color=colours, zorder=3, label="corrected")
    ax.axhline(1.0, color=INK2, linewidth=0.7, zorder=2)
    top = max([1.0] + ratios)
    ax.set_ylim(0, top * 1.30)
    for xi, r, c in zip(x, ratios, colours):
        word = "better" if c == DIV_POS else "worse"
        arrow = "↓" if r < 1.0 else "↑"
        ax.text(xi + w / 2, r + top * 0.03, f"{arrow} {word}", ha="center", va="bottom",
                fontsize=5.6, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel("corrected / backbone", fontsize=6.4, color=INK2)
    tidy(ax)
    ax.tick_params(labelsize=6.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREY),
               plt.Rectangle((0, 0), 1, 1, color=DIV_POS),
               plt.Rectangle((0, 0), 1, 1, color=DIV_NEG)]
    ax.legend(handles, ["backbone (=1)", "improved", "worse"], loc="lower right",
              fontsize=5.4, frameon=False, handlelength=1.1, handletextpad=0.4,
              borderaxespad=0.1)
    ax.set_title("safety: before / after", fontsize=7.3, color=INK, pad=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="revision/figures/fig_studies")
    ap.add_argument("--results_dir", default="outputs/revision")
    args = ap.parse_args()
    setup()

    fig = plt.figure(figsize=(7.16, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 0.95], wspace=0.62,
                          left=0.075, right=0.99, top=0.87, bottom=0.19)

    ax_a = fig.add_subplot(gs[0, 0])
    im = panel_a(ax_a, args.results_dir)
    if im is not None:
        pos = ax_a.get_position()
        cax = fig.add_axes([pos.x1 + 0.006, pos.y0, 0.010, pos.height])
        cb = fig.colorbar(im, cax=cax, orientation="vertical")
        cb.outline.set_visible(False)
        cb.set_ticks([-1, 0, 1])
        cb.set_ticklabels(["worse", "0", "better"])
        cb.ax.tick_params(labelsize=5.8, length=0, colors=INK2, pad=1.5)
        cb.set_label("column-normalised $\\Delta$", fontsize=6.0, color=INK2, labelpad=2)

    panel_b(fig, gs[0, 1], args.results_dir)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_c(ax_c, args.results_dir)

    fig.text(0.008, 0.965, "(a)", fontsize=8, color=INK, weight="bold")
    fig.text(0.375, 0.965, "(b)", fontsize=8, color=INK, weight="bold")
    fig.text(0.715, 0.965, "(c)", fontsize=8, color=INK, weight="bold")

    # figstyle.save() always writes under revision/figures/<name>.{pdf,png}; passing
    # an absolute path as name overrides that (os.path.join drops the leading part
    # for an absolute second argument), so --out is honoured whether it names the
    # default in-tree location or an out-of-tree path used for a scratch test run.
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save(fig, out_path)


if __name__ == "__main__":
    main()
