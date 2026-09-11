#!/usr/bin/env python3
"""Refuse a figure whose type collides with a block, with other type, or with the
edge of the canvas.

A label that crosses the border of a block it does not belong to is a defect a
reader sees before anything else, and it is easy to reintroduce while moving a
box by a few thousandths. This renders each figure that is drawn rather than
photographed and measures every text against every filled patch and against
every other text, in points, on the real renderer.
"""
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# points of clear air demanded between a piece of type and anything it is not part of
MIN_GAP = 0.6

MODULES = ["fig_architecture", "fig_studies"]


def _gap(a, b):
    return max(max(a.x0 - b.x1, b.x0 - a.x1), max(a.y0 - b.y1, b.y0 - a.y1))


def _inside(t, b):
    return t.x0 >= b.x0 - 1 and t.x1 <= b.x1 + 1 and t.y0 >= b.y0 - 1 and t.y1 <= b.y1 + 1


def check(modname):
    import figstyle
    saved = figstyle.save
    figstyle.save = lambda fig, name: None
    mod = importlib.import_module(modname)
    importlib.reload(mod)
    mod.save = lambda fig, name: None
    try:
        mod.main()
    finally:
        figstyle.save = saved

    bad = []
    fig = plt.gcf()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        texts = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
        blocks = [p.get_window_extent(r) for p in ax.patches
                  if isinstance(p, (FancyBboxPatch, Rectangle)) and p.get_facecolor()[3] > 0.01]
        for s, tb in texts:
            for bb in blocks:
                if _inside(tb, bb):
                    continue
                if _gap(tb, bb) < MIN_GAP:
                    bad.append(f"{modname}: {s[:30]!r} touches a block")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if _gap(texts[i][1], texts[j][1]) < MIN_GAP:
                    bad.append(f"{modname}: {texts[i][0][:22]!r} touches {texts[j][0][:22]!r}")
        ab = ax.get_window_extent(r)
        if ax.get_position().width > 0.9:          # only for a single-axes canvas
            for s, tb in texts:
                if tb.x0 < ab.x0 or tb.x1 > ab.x1 or tb.y0 < ab.y0 or tb.y1 > ab.y1:
                    bad.append(f"{modname}: {s[:30]!r} runs off the canvas")
    plt.close("all")
    return bad


def main():
    names = sys.argv[1:] or MODULES
    bad = []
    for n in names:
        bad += check(n)
    for line in dict.fromkeys(bad):
        print(line)
    if bad:
        sys.exit(1)
    for n in names:
        print(f"clean  {n}")


if __name__ == "__main__":
    main()
