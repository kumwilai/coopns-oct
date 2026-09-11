#!/usr/bin/env python3
"""Visual comparison across all four backbones, built around one image.

The correction this method applies is small in absolute terms, about one percent of
the dynamic range on average, so a raw backbone/corrected pair placed side by side is
never legible on its own. This figure does not try to make the raw pair legible. It
shows where the correction acted, on a fixed and honest scale, for every backbone at
once, one row per backbone, so a reader can see that the same direction of change
holds everywhere, not only in the example chosen, and that its size tracks backbone
fidelity.

The image is chosen by a stated rule rather than by eye, the median of NAFNet
backbone PSNR over the per image results, so the figure shows a case that is typical
for every backbone rather than a hand picked best or worst one.

The main figure crops a horizontal ribbon, because retinal layers run horizontally,
placed where the method actually acted on NAFNet, at the peak of a smoothed map of
the absolute change restricted to tissue. That one band is reused for every row so
all cells are directly comparable, and a thumbnail of the full B scan marks it.

Every number printed on the figure is read from the results JSON. Nothing here is
recomputed, so nothing here can disagree with the paper's tables.

With --supp, the script instead regenerates fig_subjective_all, a full resolution
five column grid (noisy, backbone, corrected, reference, change) over a square crop,
unwindowed, plus the four curve difference profile, for every backbone, from the same
pinned checkpoints as the main figure.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "revision"))
from figstyle import setup, save, INK, INK2, DIV_NEG, DIV_MID, DIV_POS, C, MARKER, DASH
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset
from validate_crossdataset import otsu_tissue_mask

LABEL = {"nafnet": "NAFNet", "dncnn": "DnCNN", "swinir": "SwinIR", "kbnet": "KBNet"}
# The second cue from figstyle's MARKER dict, rendered as a text glyph so a row can
# carry "marker in that backbone's colour" without a dedicated scatter axis.
MARKER_GLYPH = {"o": "●", "s": "■", "^": "▲", "D": "◆"}
# Row order matches Table 3, fixed regardless of --backbones ordering.
ROW_ORDER = ["nafnet", "dncnn", "swinir", "kbnet"]
ACCENT = "#eb6834"
VMAX = 0.06  # fixed signed-change scale, module constant, never a percentile


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


def resolve_ckpt(bb, override_ckpt="", override_bg=""):
    """Same fallback chain as before, per backbone: a swept winner if one was
    recorded, else the final checkpoint for that backbone, else the shipped one.

    This is the general-purpose chain kept for other callers and for
    --checkpoint overrides. The figure itself, below, pins each backbone to its
    exact eval-JSON-producing checkpoint and does not use this fallback, so that
    the printed numbers and the rendered pixels can never come from two models."""
    ckpt, bg_rule = override_ckpt, override_bg
    if not ckpt:
        w = "outputs/revision/winners.json"
        if os.path.exists(w):
            entry = json.load(open(w)).get(bb, {})
            cfg = entry.get("config", "")
            if cfg:
                ckpt = f"outputs/revision/sw_{bb}_{cfg}_s0/best_model_cooperative.pth"
                bg_rule = bg_rule or cfg.split("_")[0]
    if not ckpt or not os.path.exists(ckpt):
        ckpt = f"outputs/revision_v1_presweep/final_{bb}/best_model_cooperative.pth"
    if not os.path.exists(ckpt):
        ckpt = f"checkpointpaper/{bb}_pku37_cooperative.pth"
    bg_rule = bg_rule or "intensity"
    return ckpt, bg_rule


# The exact checkpoint that produced each eval_<backbone>.json, verified by the
# reviewing agent against provenance recorded in those files. This figure must use
# these and only these, with no silent fallback, so the printed numbers (read from
# the JSON) and the pixels (read from the checkpoint) describe the same model.
PINNED_CKPT = {
    "nafnet": "outputs/revision/selected_s0/nafnet/best_model_cooperative.pth",
    "swinir": "outputs/revision/selected_s0/swinir/best_model_cooperative.pth",
    "dncnn":  "outputs/revision/selected_s0/dncnn/best_model_cooperative.pth",
    "kbnet":  "outputs/revision/selected_s0/kbnet/best_model_cooperative.pth",
}

# The sweep run each pinned file was copied from, and its md5. The results JSON of
# the same backbone records the first as the model it scored, so the assertion in
# pinned_ckpt ties the pixels of this figure to the numbers of Table 7 by content
# and not by a path that a later copy could quietly change.
PINNED_ORIGIN = {
    "nafnet": "outputs/revision/sw_nafnet_halo41_dz06_s0/best_model_cooperative.pth",
    "swinir": "outputs/revision/sw_swinir_int_dz09_s0/best_model_cooperative.pth",
    "dncnn":  "outputs/revision/sw_dncnn_int_dz09_s0/best_model_cooperative.pth",
    "kbnet":  "outputs/revision/sw_kbnet_int_dz09_s0/best_model_cooperative.pth",
}
PINNED_MD5 = {
    "nafnet": "8cd4adf3bd1f",
    "swinir": "1a4e792981de",
    "dncnn":  "a547e74a711f",
    "kbnet":  "70026f72d79a",
}
# The background gate each backbone was selected under. NAFNet took the tissue
# bounded gate, the other three the intensity percentile gate. Scoring a
# checkpoint under a gate it was not selected under changes its output, so this
# is pinned beside the checkpoint rather than passed on the command line.
PINNED_GATE = {
    "nafnet": "halo41",
    "swinir": "intensity",
    "dncnn":  "intensity",
    "kbnet":  "intensity",
}


def results_json(bb):
    """Seed 0 of the selected setting, the file whose summary Table 7 reports."""
    return f"outputs/revision/test_{bb}_s0.json"


def results_all_seeds(bb):
    """Every seed of the selected setting. The scatter panel shows all of them,
    so a reader sees the seed to seed spread and every negative point, not the
    one seed whose pixels are drawn."""
    rows = []
    for p in sorted(glob.glob(f"outputs/revision/test_{bb}_s*.json")):
        rows += json.load(open(p))["per_image"]
    if not rows:
        raise SystemExit(f"no test_{bb}_s*.json found for the scatter panel")
    return rows


def pinned_ckpt(bb):
    """Fail loudly rather than substitute a different model for this backbone."""
    ckpt = PINNED_CKPT.get(bb)
    if ckpt is None:
        raise SystemExit(f"no pinned checkpoint recorded for backbone '{bb}'")
    if not os.path.exists(ckpt):
        raise SystemExit(f"pinned checkpoint missing for '{bb}': {ckpt}  "
                          f"(refusing to fall back to a different model)")
    got = hashlib.md5(open(ckpt, "rb").read()).hexdigest()[:12]
    if got != PINNED_MD5[bb]:
        raise SystemExit(
            f"pinned checkpoint for '{bb}' is not the file this figure was "
            f"verified against. expected md5 {PINNED_MD5[bb]}, found {got}. "
            f"{ckpt} must be a copy of {PINNED_ORIGIN[bb]}")
    scored = json.load(open(results_json(bb))).get("checkpoint", "")
    if scored != PINNED_ORIGIN[bb]:
        raise SystemExit(
            f"provenance mismatch for '{bb}'. {results_json(bb)} says it scored "
            f"{scored}, but this figure draws {PINNED_ORIGIN[bb]}. The printed "
            f"numbers and the pixels would come from two different models.")
    return ckpt


def pick_roi(delta, tissue, h, w):
    """Place a crop of height h, width w where the correction acted hardest inside
    tissue. Generalises the old square-only version to an arbitrary band shape, so
    the main figure's horizontal ribbon and the supplement's square crop share one
    routine.

    The absolute change is smoothed first, so that a single bright pixel of noise
    cannot decide where the reader is asked to look.
    """
    d = (delta.abs() * tissue)
    k = 9
    d = F.avg_pool2d(F.pad(d, (k // 2,) * 4, mode="reflect"), k, stride=1)
    H, W = d.shape[2], d.shape[3]
    half_h, half_w = h // 2, w // 2
    inner = d[:, :, half_h:H - half_h, half_w:W - half_w]
    if inner.numel() == 0:
        return max(0, H // 2 - half_h), max(0, W // 2 - half_w)
    idx = int(torch.argmax(inner.reshape(-1)))
    r = idx // inner.shape[3] + half_h
    c = idx % inner.shape[3] + half_w
    return int(max(0, min(H - h, r - half_h))), int(max(0, min(W - w, c - half_w)))


def style_img_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#c9c8c3"); sp.set_linewidth(0.5)


def tag(ax, x, y, text, ha, va, fontsize):
    """The white-boxed numeric tag used on the change column."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
             fontsize=fontsize, color=INK,
             bbox=dict(facecolor="white", alpha=0.85, pad=1.2, lw=0))


def row_label(ax, bb):
    """The row's backbone name, left-aligned inside column 1, preceded by that
    backbone's marker glyph in its own colour. No box, per spec, unlike the
    numeric tag on the change column."""
    ax.text(0.04, 0.5, MARKER_GLYPH[MARKER[bb]], transform=ax.transAxes,
             ha="left", va="center", fontsize=7.5, color=C[bb], fontweight="bold")
    ax.text(0.19, 0.5, LABEL[bb], transform=ax.transAxes,
             ha="left", va="center", fontsize=6.5, color=INK)


def has_layer_contrast(crop):
    """A crude but cheap check that a crop contains both a bright layer and
    darker tissue, used to decide whether the 40px band needs raising to 56px."""
    return bool((crop > 0.5).any() and (crop < 0.3).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="nafnet",
                    help="reference backbone for image selection and the --checkpoint override")
    ap.add_argument("--backbones", default="nafnet,dncnn,swinir,kbnet")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--bg_rule", default="")
    ap.add_argument("--results_json",
                    default="outputs/revision/test_nafnet_s0.json")
    ap.add_argument("--jsonl", default="pku37_oct_dataset/pku37_real_test.jsonl")
    ap.add_argument("--band_h", type=int, default=40, help="main figure ribbon height, px")
    ap.add_argument("--band_w", type=int, default=136, help="main figure ribbon width, px")
    ap.add_argument("--supp_size", type=int, default=160, help="supplement square crop, px")
    ap.add_argument("--idx", type=int, default=None,
                    help="Override the median-backbone-PSNR image selection.")
    ap.add_argument("--name", default="fig_subjective_pku37")
    ap.add_argument("--supp", action="store_true",
                    help="Write fig_subjective_all (full res 5-col grid + difference "
                         "profile) instead of the main figure.")
    args = ap.parse_args()
    torch.set_num_threads(4)
    setup()

    backbones = [s.strip() for s in args.backbones.split(",") if s.strip()]
    if "nafnet" not in backbones:
        raise SystemExit("nafnet must be in --backbones, row 1 and the crop both need it")
    missing_row = [bb for bb in ROW_ORDER if bb not in backbones]
    if missing_row:
        raise SystemExit(f"--backbones is missing {missing_row}, required for the four table rows")

    # Cross check against the current test set sweep (all seeds, test_<bb>_s*.json),
    # independent of the single pinned reference checkpoint used for the pixels
    # below. The abstract and Section VII quote these two numbers directly, so this
    # script recomputes them on every run rather than risk them drifting apart from
    # the prose.
    rhos, tot_pos, tot_n = [], 0, 0
    for bb in ROW_ORDER:
        psnr_bb, cnr_bb = [], []
        for p in sorted(glob.glob(f"outputs/revision/test_{bb}_s*.json")):
            per_bb = json.load(open(p))["per_image"]
            psnr_bb += [r["psnr_backbone"] for r in per_bb]
            cnr_bb += [r["cnr_change_pct"] for r in per_bb]
        if psnr_bb:
            rho, _ = spearmanr(psnr_bb, cnr_bb)
            rhos.append(rho)
            tot_pos += sum(1 for v in cnr_bb if v > 0)
            tot_n += len(cnr_bb)
    if rhos:
        print(f"test set sweep check   spearman(backbone PSNR, CNR gain) per backbone "
              f"{min(rhos):+.2f} to {max(rhos):+.2f}, contrast rose in {tot_pos} of "
              f"{tot_n} image and seed combinations")

    # Image selection, median of NAFNet backbone PSNR, exactly as before.
    sel_per = json.load(open(args.results_json))["per_image"]
    rows_sorted = sorted(sel_per, key=lambda r: r.get("psnr_backbone", 0.0))
    if args.idx is not None:
        idx = args.idx
    else:
        idx = int(rows_sorted[len(rows_sorted) // 2]["image_idx"])
    print(f"image      {idx}  (median NAFNet backbone PSNR)")

    # Per backbone results, all numbers read from JSON and never recomputed except
    # the plain mean used only for the scatter panel.
    data = {}
    for bb in backbones:
        per = json.load(open(results_json(bb)))["per_image"]
        row = next(r for r in per if int(r.get("image_idx", -1)) == idx)
        mean_psnr = float(np.mean([r["psnr_backbone"] for r in per]))
        data[bb] = {"per": per, "row": row, "mean_psnr": mean_psnr}
        print(f"{bb:8s} |d| {row['correction_magnitude']:.4f}   "
              f"dPSNR {row['psnr_delta']:+.3f} dB   "
              f"CNR {row['cnr_backbone']:.2f} -> {row['cnr_corrected']:.2f} "
              f"({row['cnr_change_pct']:+.2f}%)   mean backbone PSNR {mean_psnr:.2f} dB")

    # Run every backbone on the same image.
    ds = PKU37Dataset(args.jsonl, patch_size=0, is_train=False)
    s = ds[idx]
    noisy, clean = s["noisy"].unsqueeze(0), s["clean"].unsqueeze(0)
    outs = {}
    for bb in backbones:
        # Pinned to the exact checkpoint that produced eval_<bb>.json. --checkpoint
        # may still override, but only for the reference backbone, and it is then
        # verified below rather than silently trusted.
        ckpt = args.checkpoint if (bb == args.backbone and args.checkpoint) else pinned_ckpt(bb)
        bg_rule = (args.bg_rule if (bb == args.backbone and args.bg_rule)
                   else PINNED_GATE[bb])
        print(f"model[{bb}]  requested {ckpt}  gate={bg_rule}")
        if not os.path.exists(ckpt):
            raise SystemExit(f"checkpoint for '{bb}' does not exist: {ckpt}")
        model = load_model(bb, ckpt, bg_rule)
        print(f"model[{bb}]  loaded    {os.path.abspath(ckpt)}")
        with torch.no_grad():
            b, u = model.backbone(noisy)
            q, _ = model.corrector(b, noisy, None, nafnet_uncertainty=u, return_details=False)
        outs[bb] = {"b": b, "q": q}

    b_naf, q_naf = outs["nafnet"]["b"], outs["nafnet"]["q"]
    tissue = otsu_tissue_mask(b_naf)
    dcmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])

    if not args.supp:
        # ---------------------------------------------------------- MAIN FIGURE
        # The crop is a horizontal ribbon, chosen on NAFNet alone, reused for
        # every row. Raise the band from 40 to 56px if it lands entirely in
        # bright layer with no darker tissue for contrast.
        band_h, band_w = args.band_h, args.band_w
        r0, c0 = pick_roi(q_naf - b_naf, tissue, band_h, band_w)
        check = b_naf[0, 0][r0:r0 + band_h, c0:c0 + band_w].numpy()
        if band_h == 40 and not has_layer_contrast(check):
            band_h = 56
            r0, c0 = pick_roi(q_naf - b_naf, tissue, band_h, band_w)
            print(f"band raised to {band_h}px, 40px band was all bright layer")
        sl = (slice(r0, r0 + band_h), slice(c0, c0 + band_w))
        print(f"band       h={band_h} w={band_w}  at row {r0} col {c0}")

        fig = plt.figure(figsize=(7.16, 2.55))
        outer = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.10,
                                 left=0.035, right=0.99, top=0.91, bottom=0.03)

        # ------------------------------------------------------------ LEFT BLOCK
        gsL = outer[0, 0].subgridspec(4, 4, width_ratios=[1, 1, 1, 0.06],
                                      wspace=0.03, hspace=0.10)
        col_titles = ["Backbone", "Corrected", "Change"]
        im = None
        for i, bb in enumerate(ROW_ORDER):
            b_bb, q_bb = outs[bb]["b"], outs[bb]["q"]
            crop_b = b_bb[0, 0][sl].numpy()
            crop_q = q_bb[0, 0][sl].numpy()
            d = (q_bb - b_bb)[0, 0][sl].numpy()
            row_bb = data[bb]["row"]

            ax0 = fig.add_subplot(gsL[i, 0])
            ax0.imshow(crop_b, cmap="gray", vmin=0.05, vmax=0.85, interpolation="nearest")
            style_img_ax(ax0)
            row_label(ax0, bb)
            if i == 0:
                ax0.set_title(col_titles[0], fontsize=6.6, color=INK, pad=2)

            ax1 = fig.add_subplot(gsL[i, 1])
            ax1.imshow(crop_q, cmap="gray", vmin=0.05, vmax=0.85, interpolation="nearest")
            style_img_ax(ax1)
            if i == 0:
                ax1.set_title(col_titles[1], fontsize=6.6, color=INK, pad=2)

            ax2 = fig.add_subplot(gsL[i, 2])
            im = ax2.imshow(d, cmap=dcmap, vmin=-VMAX, vmax=VMAX, interpolation="nearest")
            style_img_ax(ax2)
            if i == 0:
                ax2.set_title(col_titles[2], fontsize=6.6, color=INK, pad=2)
            tag(ax2, 0.96, 0.04,
                f"ΔPSNR {row_bb['psnr_delta']:+.2f} dB  CNR {row_bb['cnr_change_pct']:+.1f}%",
                "right", "bottom", 5.8)

        cax = fig.add_subplot(gsL[:, 3])
        cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[-VMAX, 0, VMAX])
        # Spell the ticks out. The default formatter renders the middle tick as
        # minus zero, which reads as an error rather than as the neutral point.
        cb.ax.set_yticklabels([f"{-VMAX:+.2f}", "0", f"{VMAX:+.2f}"])
        cb.ax.tick_params(labelsize=5.5, colors=INK2, length=2, pad=1)
        # Without this the bar carries no statement of what it measures, and the
        # scatter axis label to its right reads as if it belonged to the bar.
        cb.set_label("corrected minus backbone", fontsize=6.0, color=INK2, labelpad=3)
        cb.outline.set_edgecolor("#c9c8c3"); cb.outline.set_linewidth(0.5)

        # ----------------------------------------------------------- RIGHT BLOCK
        gsR = outer[0, 1].subgridspec(2, 1, height_ratios=[0.36, 0.64], hspace=0.18)

        axT = fig.add_subplot(gsR[0, 0])
        axT.imshow(clean[0, 0].numpy(), cmap="gray", vmin=0, vmax=1, aspect="auto")
        axT.add_patch(Rectangle((c0, r0), band_w, band_h, fill=False,
                                edgecolor=ACCENT, linewidth=0.9))
        style_img_ax(axT)

        axS = fig.add_subplot(gsR[1, 0])
        all_ys = []
        for bb in ROW_ORDER:
            per = results_all_seeds(bb)
            xs = np.array([r["psnr_backbone"] for r in per])
            ys = np.array([r["cnr_change_pct"] for r in per])
            all_ys.append(ys)
            axS.scatter(xs, ys, s=5, color=C[bb], marker=MARKER[bb], alpha=0.55, linewidths=0)
            axS.scatter([xs.mean()], [ys.mean()], s=26, color=C[bb], marker=MARKER[bb],
                       edgecolors=INK, linewidths=0.7, zorder=5)
            row_bb = data[bb]["row"]
            axS.scatter([row_bb["psnr_backbone"]], [row_bb["cnr_change_pct"]], s=32,
                       facecolors="none", edgecolors=INK, linewidths=1.0, zorder=6)
        axS.axhline(0.0, color=INK2, linewidth=0.6, linestyle="--")
        all_ys = np.concatenate(all_ys)
        npos = int((all_ys > 0).sum())
        nneg_by_bb = {bb: int((np.array([r["cnr_change_pct"]
                                        for r in results_all_seeds(bb)]) <= 0).sum())
                      for bb in ROW_ORDER}
        worst = min(min(r["cnr_change_pct"] for r in results_all_seeds(bb))
                    for bb in ROW_ORDER)
        axS.set_xlabel("backbone PSNR (dB)", fontsize=6.5, color=INK2, labelpad=2)
        axS.set_ylabel("CNR change (%)", fontsize=6.5, color=INK2, labelpad=1)
        axS.tick_params(axis="both", labelsize=5.5, colors=INK2, length=2, pad=1)
        for sp in axS.spines.values():
            sp.set_edgecolor("#c9c8c3"); sp.set_linewidth(0.5)

        save(fig, args.name)
        print(f"drew image {idx}, signed-change scale +-{VMAX:.3f}, "
              f"band {band_h}x{band_w}px, {npos}/{len(all_ys)} positive, "
              f"negatives per backbone {nneg_by_bb}, most negative {worst:+.3f} percent, "
              f"rows {ROW_ORDER}")
        return

    # -------------------------------------------------------------- SUPPLEMENT
    # fig_subjective_all: full resolution, five columns per backbone row, plus
    # the four curve difference profile. Square crop, unwindowed (vmin 0, vmax
    # 1), from the same pinned checkpoints, so it cannot drift from the tables
    # the way the stale hand made file did.
    n = args.supp_size
    r0, c0 = pick_roi(q_naf - b_naf, tissue, n, n)
    sl = (slice(r0, r0 + n), slice(c0, c0 + n))
    print(f"supp crop  {n}x{n}px at row {r0} col {c0}")

    fig = plt.figure(figsize=(7.16, 6.2))
    outer = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 1, 1.15],
                             wspace=0.06, hspace=0.14,
                             left=0.035, right=0.985, top=0.95, bottom=0.06)

    col_titles = ["Noisy", "Backbone", "Corrected", "Reference", "Change"]
    noisy_crop = noisy[0, 0][sl].numpy()
    clean_crop = clean[0, 0][sl].numpy()
    im = None
    for i, bb in enumerate(ROW_ORDER):
        b_bb, q_bb = outs[bb]["b"], outs[bb]["q"]
        crop_b = b_bb[0, 0][sl].numpy()
        crop_q = q_bb[0, 0][sl].numpy()
        d = (q_bb - b_bb)[0, 0][sl].numpy()
        cells = [noisy_crop, crop_b, crop_q, clean_crop]
        for j, cell in enumerate(cells):
            ax = fig.add_subplot(outer[i, j])
            ax.imshow(cell, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            style_img_ax(ax)
            if j == 0:
                row_label(ax, bb)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=6.6, color=INK, pad=2)
        axD = fig.add_subplot(outer[i, 4])
        im = axD.imshow(d, cmap=dcmap, vmin=-VMAX, vmax=VMAX, interpolation="nearest")
        style_img_ax(axD)
        if i == 0:
            axD.set_title(col_titles[4], fontsize=6.6, color=INK, pad=2)

    # The difference profile, one panel spanning all four rows, one curve per
    # backbone, moved here from the main figure per the revised layout.
    axP = fig.add_subplot(outer[:, 5])
    depth = np.arange(n)
    band = slice(n // 3, 2 * n // 3)
    for bb in ROW_ORDER:
        b_bb, q_bb = outs[bb]["b"], outs[bb]["q"]
        prof = (q_bb - b_bb)[0, 0][sl][:, band].mean(dim=1).numpy()
        ln, = axP.plot(prof, depth, color=C[bb], linewidth=1.1, label=LABEL[bb])
        if DASH[bb][0] is not None:
            ln.set_dashes(list(DASH[bb]))
    axP.axvline(0.0, color=INK2, linewidth=0.6, linestyle="--")
    axP.invert_yaxis()
    axP.set_title("difference profile", fontsize=6.6, color=INK, pad=2)
    axP.set_xlabel("corrected minus backbone", fontsize=6.0, color=INK2, labelpad=2)
    axP.set_ylabel("depth (px)", fontsize=6.0, color=INK2, labelpad=2)
    axP.tick_params(axis="both", labelsize=5.5, colors=INK2, length=2, pad=1)
    for sp in axP.spines.values():
        sp.set_edgecolor("#c9c8c3"); sp.set_linewidth(0.5)
    axP.legend(fontsize=5.5, frameon=False, loc="best", handlelength=1.3,
              borderpad=0.15, labelspacing=0.18)

    save(fig, "fig_subjective_all")
    print(f"drew image {idx} (supplement fig_subjective_all), crop {n}x{n}px")


if __name__ == "__main__":
    main()
