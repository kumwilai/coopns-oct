#!/usr/bin/env python3
"""Visual comparison, built so that a small correction is still visible and still honest.

The correction this method applies is small in absolute terms, about one percent of
the dynamic range. Two images placed side by side therefore look identical, which
makes a naive figure useless as evidence. Three choices fix that without overstating
anything.

The images are chosen by a stated rule rather than by eye. The per image results of
the scoring run are ranked by contrast gain and the best, the median and the worst
case are all shown. A reader can object to a hand picked example, but not to a figure
that includes its own worst case.

The crop is placed where the method actually acted, at the peak of a smoothed map of
the absolute change restricted to tissue, rather than where the reference image
happens to have structure. A crop chosen without reference to the correction can
easily show a region where nothing happened.

The difference is drawn on a scale that is fixed across every row and printed in the
axis label, so the panels can be compared with each other and the size of the change
cannot be inflated by per panel autoscaling. Beside it a one dimensional profile
across a layer boundary shows the backbone, the corrected output and the reference
together, which is what makes the sharpening legible rather than merely asserted.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "revision"))
from figstyle import setup, save, INK, INK2, DIV_NEG, DIV_MID, DIV_POS
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset
from eval_pku37_test import compute_psnr
from validate_crossdataset import otsu_tissue_mask

LABEL = {"nafnet": "NAFNet", "dncnn": "DnCNN", "swinir": "SwinIR", "kbnet": "KBNet"}
ACCENT = "#eb6834"


def load_model(bb, ckpt, bg_rule="intensity"):
    m = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=bb, pretrained_backbone=f"checkpointpaper/{bb}_backbone.pth",
        hidden_channels=64)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
          for k, v in sd.get("model_state_dict", sd).items()}
    ms = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in ms and v.shape == ms[k].shape}, strict=False)
    m.corrector.bg_rule = bg_rule
    return m.eval()


def pick_cases(results_json, n_show=3, select_by="gain"):
    """Choose which images to show, by a stated rule rather than by eye.

    Two rules are offered.

    gain      rank by contrast gain and show the best, the median and the worst.
              A reader can object to a hand picked example but not to a figure
              that includes its own worst case.

    headroom  rank by how much the backbone left on the table, measured by its own
              fidelity against the clean reference, and show the image where it left
              most, a middling one, and the one where it left least. This is the rule
              that makes the argument visible. The correction should be large where
              the backbone fell short and small where the backbone was already close,
              because a wrapper that changes an already good image by as much as a
              poor one is not responding to the data, it is applying a fixed effect.

    Returns a list of (image index, label, contrast gain).
    """
    per = json.load(open(results_json))["per_image"]
    if not per:
        return []
    if select_by == "headroom":
        rows = sorted(per, key=lambda r: r.get("psnr_backbone", 0.0))
        picks = [(rows[0], "backbone weakest here"),
                 (rows[len(rows) // 2], "typical"),
                 (rows[-1], "backbone already strong")]
    else:
        rows = sorted(per, key=lambda r: r.get("cnr_change_pct", 0.0))
        picks = [(rows[-1], "best case"), (rows[len(rows) // 2], "median case"),
                 (rows[0], "worst case")]
    return [(int(r.get("image_idx", 0)), lab, float(r.get("cnr_change_pct", 0.0)))
            for r, lab in picks][:n_show]


def pick_roi(delta, tissue, size):
    """Place the crop where the correction acted hardest inside tissue.

    The absolute change is smoothed first, so that a single bright pixel of noise
    cannot decide where the reader is asked to look.
    """
    d = (delta.abs() * tissue)
    k = 9
    d = F.avg_pool2d(F.pad(d, (k // 2,) * 4, mode="reflect"), k, stride=1)
    H, W = d.shape[2], d.shape[3]
    half = size // 2
    inner = d[:, :, half:H - half, half:W - half]
    if inner.numel() == 0:
        return max(0, H // 2 - half), max(0, W // 2 - half)
    idx = int(torch.argmax(inner.reshape(-1)))
    r = idx // inner.shape[3] + half
    c = idx % inner.shape[3] + half
    return int(max(0, min(H - size, r - half))), int(max(0, min(W - size, c - half)))


def local_cnr(t, mask):
    bg = 1.0 - mask
    if float(mask.sum()) < 1 or float(bg.sum()) < 1:
        return float("nan")
    ms = float((t * mask).sum() / mask.sum().clamp(min=1))
    mb = float((t * bg).sum() / bg.sum().clamp(min=1))
    sb = float(torch.sqrt((((t - mb) ** 2) * bg).sum() / bg.sum().clamp(min=1)) + 1e-8)
    return (ms - mb) / sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="nafnet")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--bg_rule", default="")
    ap.add_argument("--results_json", default="outputs/revision/eval_nafnet.json")
    ap.add_argument("--jsonl", default="pku37_oct_dataset/pku37_real_test.jsonl")
    ap.add_argument("--roi", type=int, default=160)
    ap.add_argument("--amplify", type=float, default=5.0,
                    help="Gain on the change in the amplified panel. Stated in the panel title.")
    ap.add_argument("--name", default="fig_subjective_pku37")
    ap.add_argument("--select_by", default="gain", choices=["gain", "headroom"],
                    help="Which rule chooses the three images shown.")
    args = ap.parse_args()
    torch.set_num_threads(4)
    setup()

    ckpt, bg_rule = args.checkpoint, args.bg_rule
    if not ckpt:
        # Fall back to whatever the selector chose, so the figure always shows the
        # model the tables report rather than an older one left on disk.
        w = "outputs/revision/winners.json"
        if os.path.exists(w):
            entry = json.load(open(w)).get(args.backbone, {})
            cfg = entry.get("config", "")
            if cfg:
                ckpt = f"outputs/revision/sw_{args.backbone}_{cfg}_s0/best_model_cooperative.pth"
                bg_rule = bg_rule or cfg.split("_")[0]
    if not ckpt or not os.path.exists(ckpt):
        ckpt = f"outputs/revision/final_{args.backbone}/best_model_cooperative.pth"
    if not os.path.exists(ckpt):
        ckpt = f"checkpointpaper/{args.backbone}_pku37_cooperative.pth"
    bg_rule = bg_rule or "intensity"
    print(f"model      {ckpt}")
    print(f"gate       {bg_rule}")

    cases = (pick_cases(args.results_json, select_by=args.select_by)
             if os.path.exists(args.results_json) else [])
    if not cases:
        print(f"warning: {args.results_json} unavailable, falling back to three fixed images")
        cases = [(7, "example", float("nan")), (23, "example", float("nan")),
                 (61, "example", float("nan"))]

    model = load_model(args.backbone, ckpt, bg_rule)
    ds = PKU37Dataset(args.jsonl, patch_size=0, is_train=False)

    rows = []
    for idx, label, gain in cases:
        if idx >= len(ds):
            print(f"warning: image {idx} beyond the dataset, skipped")
            continue
        s = ds[idx]
        noisy, clean = s["noisy"].unsqueeze(0), s["clean"].unsqueeze(0)
        with torch.no_grad():
            b, u = model.backbone(noisy)
            q, _ = model.corrector(b, noisy, None, nafnet_uncertainty=u, return_details=False)
        tissue = otsu_tissue_mask(b)
        r0, c0 = pick_roi(q - b, tissue, args.roi)
        sl = (slice(r0, r0 + args.roi), slice(c0, c0 + args.roi))
        # Contrast is reported over the whole B scan, not over the crop. The crop is
        # placed where the correction acted hardest, which is not where contrast is
        # measured, so a crop statistic would not correspond to any number in the
        # tables and could carry the opposite sign to the image it is drawn from.
        cnr_b = local_cnr(b[0, 0], tissue[0, 0])
        cnr_q = local_cnr(q[0, 0], tissue[0, 0])
        rows.append({
            "label": label, "idx": idx, "r0": r0, "c0": c0, "sl": sl,
            "noisy": noisy, "clean": clean, "b": b, "q": q,
            "psnr_b": compute_psnr(b, clean), "psnr_q": compute_psnr(q, clean),
            "cnr_b": cnr_b, "cnr_q": cnr_q,
            # measured on the model being drawn, so the caption cannot disagree with it
            "gain": 100.0 * (cnr_q - cnr_b) / (abs(cnr_b) + 1e-12),
        })
    if not rows:
        print("nothing to draw")
        return

    # One scale for every difference panel, so the rows can be compared. The scale
    # is a high percentile rather than the maximum, because a single outlier pixel
    # would otherwise set the range and flatten every panel to one colour.
    alldiff = np.concatenate([(r["q"] - r["b"])[0, 0][r["sl"]].abs().numpy().ravel()
                              for r in rows])
    vmax = float(np.percentile(alldiff, 99.0))
    vmax = max(vmax, 1e-6)
    dcmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])

    # The amplified panel is the backbone with the correction multiplied up. The
    # change is about one percent of the dynamic range, so backbone and corrected
    # are indistinguishable side by side however they are drawn. Showing the change
    # at a stated gain makes its structure legible while the honest pair stays in
    # the two columns to its left for the reader to check.
    amp = args.amplify
    titles = ["noisy input", "backbone", "corrected", f"change $\\times${amp:g}",
              "reference", f"signed change, $\\pm${vmax:.3f}", "profile"]
    ncol, nrow = 7, len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.16, 1.30 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    for i, r in enumerate(rows):
        sl = r["sl"]
        amp_img = (r["b"] + amp * (r["q"] - r["b"])).clamp(0, 1)
        panels = [r["noisy"], r["b"], r["q"], amp_img, r["clean"]]
        for j, t in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(t[0, 0][sl].numpy(), cmap="gray", vmin=0.05, vmax=0.85,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#c9c8c3"); sp.set_linewidth(0.5)

        ax = axes[i][5]
        d = (r["q"] - r["b"])[0, 0][sl].numpy()
        ax.imshow(d, cmap=dcmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#c9c8c3"); sp.set_linewidth(0.5)

        # The profile is averaged over the middle third of the crop's columns, which
        # suppresses speckle without smoothing away the boundary itself, and is then
        # zoomed onto the steepest boundary in the crop. Drawn over the whole crop and
        # the full intensity range the three curves sit on top of one another and the
        # panel says nothing, which is the thing this figure exists to avoid.
        ax = axes[i][6]
        n = args.roi
        band = slice(n // 3, 2 * n // 3)
        profs = {}
        for key, t in (("b", r["b"]), ("q", r["q"]), ("c", r["clean"])):
            profs[key] = t[0, 0][sl][:, band].mean(dim=1).numpy()
        grad = np.abs(np.gradient(profs["b"]))
        centre = int(np.argmax(grad))
        half = max(12, n // 8)
        lo, hi = max(0, centre - half), min(n, centre + half)
        yy = np.arange(lo, hi)
        for key, col, lab, lw in (("b", INK2, "backbone", 1.0),
                                  ("q", ACCENT, "corrected", 1.3),
                                  ("c", "#9a9a94", "reference", 0.8)):
            ax.plot(profs[key][lo:hi], yy, color=col, linewidth=lw, label=lab)
        seg = np.concatenate([profs[k][lo:hi] for k in ("b", "q", "c")])
        pad = 0.06 * (seg.max() - seg.min() + 1e-6)
        ax.invert_yaxis()
        ax.set_xlim(seg.min() - pad, seg.max() + pad)
        ax.set_ylim(hi, lo)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=5.5, colors=INK2, length=2, pad=1)
        for sp in ax.spines.values():
            sp.set_edgecolor("#c9c8c3"); sp.set_linewidth(0.5)
        if i == 0:
            ax.legend(fontsize=4.8, frameon=False, loc="lower right",
                      handlelength=1.1, borderpad=0.2, labelspacing=0.2)

        if i == 0:
            for j in range(ncol):
                axes[i][j].set_title(titles[j], fontsize=7.0, color=INK, pad=3)
        axes[i][0].set_ylabel(r["label"], fontsize=6.4, color=INK, labelpad=2)
        if not np.isnan(r["gain"]):
            axes[i][0].text(0.5, -0.06, f"CNR {r['gain']:+.1f}%",
                            transform=axes[i][0].transAxes, ha="center", va="top",
                            fontsize=6.0, color=INK2)

        for j, txt in ((1, f"{r['psnr_b']:.2f} dB  CNR {r['cnr_b']:.2f}"),
                       (2, f"{r['psnr_q']:.2f} dB  CNR {r['cnr_q']:.2f}")):
            axes[i][j].text(0.5, -0.06, txt, transform=axes[i][j].transAxes,
                            ha="center", va="top", fontsize=6.0, color=INK2)

        # Where the crop sits in the whole B scan.
        ins = axes[i][0].inset_axes([0.60, 0.60, 0.385, 0.385])
        ins.imshow(r["clean"][0, 0].numpy(), cmap="gray", vmin=0, vmax=1, aspect="auto")
        ins.add_patch(Rectangle((r["c0"], r["r0"]), args.roi, args.roi, fill=False,
                                edgecolor=ACCENT, linewidth=0.8))
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_edgecolor(ACCENT); sp.set_linewidth(0.6)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.075,
                        wspace=0.06, hspace=0.26)
    save(fig, args.name)
    print(f"drew {nrow} rows, difference scale +-{vmax:.4f}")


if __name__ == "__main__":
    main()
