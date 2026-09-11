#!/usr/bin/env python3
"""Turn the result files into the LaTeX table bodies used by the manuscript.

Nothing is typed by hand, so a number in the paper cannot disagree with the file
that produced it. Missing results are printed as a dash rather than guessed.
"""
import glob
import json
import os
import re
import statistics

# Resolve both paths from this file rather than from the working directory.
# Running the script from inside revision/ used to create revision/revision/ and
# silently write the tables where nothing reads them, while the manuscript kept
# the previous run's numbers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "outputs", "revision")
DEST = os.path.join(_HERE, "sections", "generated")
BB = ["nafnet", "dncnn", "swinir", "kbnet"]
LABEL = {"nafnet": "NAFNet", "dncnn": "DnCNN", "kbnet": "KBNet", "swinir": "SwinIR"}
CLIN = ["CNR", "TCI", "EPI", "BS", "ENL", "SNR"]

# Shared between table_in_distribution and table_transfer, both of which read
# validate_crossdataset's zeroshot files rather than the per image summaries
# the PKU37 runs produce, so the percentage change is formed the same way in
# both places from this one pair of helpers.
PAIR = {"CNR": "cnr", "TCI": "tci", "EPI": "epi", "BS": "boundary_sharpness",
        "ENL": "enl", "SNR": "snr"}


def transfer_raw_values(rec):
    """psnr_delta, psnr_corrected, plus each clinical measure's percentage change.

    Returns None if rec itself is missing.
    """
    if rec is None:
        return None
    # The first numeric column of the table is the BACKBONE value in every block,
    # so the transfer rows must read psnr_backbone. They previously read
    # psnr_corrected, which made one column carry two different quantities and
    # gave the frozen backbone a spurious spread across seeds.
    out = {"PSNR": rec["psnr_delta"], "PSNR_abs": rec.get("psnr_backbone")}
    for k in CLIN:
        b, c = rec.get(PAIR[k] + "_backbone"), rec.get(PAIR[k] + "_corrected")
        out[k] = None if b is None or c is None else 100.0 * (c - b) / (abs(b) + 1e-12)
    return out


def find_transfer_rec(recs, tag):
    return next((r for r in recs
                 if tag in str(r.get("dataset", "")).lower().replace("-", "")), None)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def d(summary, key, pct=True, signed=True):
    if not summary or key not in summary:
        return "--"
    v = summary[key]["delta_mean"]
    s = f"{v:+.2f}" if signed else f"{v:.2f}"
    return s


def bold_if_pos(txt):
    if txt == "--":
        return txt
    if not txt.startswith("+"):
        return txt
    # fmt_mean_sd may already hand back a self-contained "$\pm$" pair, which
    # would nest math mode inside this wrapper's own $...$. Strip those inner
    # dollars since this wrapper supplies the enclosing math delimiters.
    return f"$\\mathbf{{{txt.replace('$', '')}}}$"


def clinical_cell(values):
    """One clinical cell for the merged results table, mean only, two decimals,
    signed, the spread dropped rather than printed. Bold marks a cell whose mean
    minus one seed standard deviation clears zero, so every seed agrees on the
    sign within one spread. A cell left plain and positive is within one spread
    of zero, and a cell left plain and negative is simply negative. A single
    value, with no seed spread to subtract, reduces to bolding on sign alone.
    The full per seed spread is not lost, it is emitted separately by
    table_full_spreads for the supplementary material.
    """
    if not values:
        return "--"
    mean = statistics.mean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    txt = f"{mean:+.2f}"
    return f"$\\mathbf{{{txt}}}$" if (mean - spread) > 0 else txt


def collect_seeds(prefix):
    """Load every outputs/revision/<prefix>_s*.json, sorted by seed number.

    Returns a list of (seed, loaded_json) or [] if no seeded files exist, so
    callers can fall back to the old single-run path.
    """
    out = []
    for path in glob.glob(f"{OUT}/{prefix}_s*.json"):
        m = re.search(r"_s(\d+)\.json$", path)
        if not m:
            continue
        j = load(path)
        if j is not None:
            out.append((int(m.group(1)), j))
    out.sort(key=lambda t: t[0])
    return out


def fmt_mean_sd(values, fmt="%+.2f"):
    if not values:
        return "--"
    if len(values) == 1:
        return fmt % values[0]
    mean = fmt % statistics.mean(values)
    decimals = int(fmt.split(".")[1][:-1]) if "." in fmt else 0
    sd = f"%.{decimals}f" % statistics.stdev(values)
    return f"{mean} $\\pm$ {sd}"


def transfer_block(tag):
    """Rows for one no adaptation dataset block, backbone name plus absolute PSNR.

    Reuses transfer_raw_values and find_transfer_rec, the same code table_transfer
    uses, so the two tables cannot disagree on how a percentage change is formed.
    """
    lines = []
    seed_counts = []
    for b in BB:
        seeds = collect_seeds(f"zeroshot_{b}")
        if seeds:
            seed_counts.append(len(seeds))
            per_key = {"PSNR": [], "PSNR_abs": []}
            per_key.update({k: [] for k in CLIN})
            for _, recs in seeds:
                rv = transfer_raw_values(find_transfer_rec(recs, tag))
                if rv is None:
                    continue
                per_key["PSNR"].append(rv["PSNR"])
                if rv["PSNR_abs"] is not None:
                    per_key["PSNR_abs"].append(rv["PSNR_abs"])
                for k in CLIN:
                    if rv[k] is not None:
                        per_key[k].append(rv[k])
            # Same guard as the in distribution block. The frozen backbone must read
            # identically at every seed, so print one value and fail loudly rather than
            # average if it does not.
            if per_key["PSNR_abs"]:
                _sp = max(per_key["PSNR_abs"]) - min(per_key["PSNR_abs"])
                if _sp > 0.005:
                    raise SystemExit(
                        "backbone PSNR differs across seeds in a transfer block, spread %.4f dB. "
                        "The frozen reference is not frozen." % _sp)
                abs_psnr = "%.2f" % per_key["PSNR_abs"][0]
            else:
                abs_psnr = "--"
            row = [LABEL[b], abs_psnr, fmt_mean_sd(per_key["PSNR"])]
            row += [clinical_cell(per_key[k]) for k in CLIN]
        else:
            recs = load(f"{OUT}/zeroshot_{b}.json") or []
            rv = transfer_raw_values(find_transfer_rec(recs, tag))
            if rv is None:
                row = [LABEL[b]] + ["--"] * 8
            else:
                abs_psnr = "--" if rv["PSNR_abs"] is None else f"{rv['PSNR_abs']:.2f}"
                clin_cells = [clinical_cell([] if rv[k] is None else [rv[k]]) for k in CLIN]
                row = [LABEL[b], abs_psnr, "%+.2f" % rv["PSNR"]] + clin_cells
        lines.append(" & ".join(row) + r" \\")
    return lines, (max(seed_counts) if seed_counts else 0)


def _backbone_rows(prefix):
    """One row per backbone, the mean over seeds where a seeded sweep exists."""
    lines, seed_counts = [], []
    for b in BB:
        seeds = collect_seeds(f"{prefix}_{b}")
        if seeds:
            seed_counts.append(len(seeds))
            psnr_deltas, psnr_backbones = [], []
            clin_deltas = {k: [] for k in CLIN}
            for _, r in seeds:
                s = r["summary"]
                psnr_deltas.append(s["PSNR (dB)"]["delta_mean"])
                psnr_backbones.append(s["PSNR (dB)"]["backbone_mean"])
                for k in CLIN:
                    clin_deltas[k].append(s[k]["delta_mean"])
            # The backbone is frozen, so every seed must measure it identically. A
            # silent average here would hide exactly the kind of drift that was
            # found in the DnCNN batch normalisation layers, so this fails loudly.
            if max(psnr_backbones) - min(psnr_backbones) > 0.005:
                raise SystemExit(
                    "backbone PSNR differs across seeds for %s, spread %.4f dB. "
                    "The frozen reference is not frozen." % (b, max(psnr_backbones) - min(psnr_backbones)))
            row = [LABEL[b], f"{psnr_backbones[0]:.2f}", fmt_mean_sd(psnr_deltas)]
            row += [clinical_cell(clin_deltas[k]) for k in CLIN]
            lines.append(" & ".join(row) + r" \\")
        else:
            r = load(f"{OUT}/eval_{b}.json")
            s = r["summary"] if r else None
            abs_psnr = f"{s['PSNR (dB)']['backbone_mean']:.2f}" if s else "--"
            clin_cells = [clinical_cell([] if not s else [s[k]["delta_mean"]]) for k in CLIN]
            lines.append(" & ".join([LABEL[b], abs_psnr, d(s, "PSNR (dB)")] + clin_cells) + r" \\")
    # No mean row. Averaging a percentage change across four backbones with
    # different starting values is not a meaningful number, so it is not printed.
    return lines, seed_counts


# The in distribution table is set in one column, so its header has to be short.
# The transfer table keeps the wide header, since it spans both columns.
HEAD9 = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"Backbone & PSNR & $\Delta$PSNR & $\Delta$CNR & $\Delta$TCI & "
         r"$\Delta$EPI & $\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
HEAD9W = [r"\begin{tabular}{lcccccccc}", r"\toprule",
          r"Backbone & Backbone PSNR (dB) & $\Delta$PSNR (dB) & $\Delta$CNR & $\Delta$TCI & "
          r"$\Delta$EPI & $\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
FOOT = [r"\bottomrule", r"\end{tabular}"]


def table_in_distribution():
    """The four backbones on the PKU37 test set, and nothing else.

    Set in one column, so the absolute backbone PSNR moves to the caption. The
    seed spread of the fidelity column stays, since it appears nowhere else and
    dropping it would remove the only uncertainty reported for fidelity."""
    lines, seeds = _backbone_rows("test")
    narrow = []
    for row in lines:
        cells = row.rstrip(" \\\\").split(" & ")
        # drop the absolute backbone PSNR, and the plus or minus from the delta
        cells = [cells[0]] + [cells[2]] + cells[3:]
        narrow.append(" & ".join(cells) + r" \\")
    header = [r"\begin{tabular}{lccccccc}", r"\toprule",
              r"Backbone & $\Delta$PSNR & $\Delta$CNR & $\Delta$TCI & $\Delta$EPI & "
              r"$\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
    return "\n".join(header + narrow + FOOT), (max(seeds) if seeds else 0)


def table_zeroshot():
    """The same four backbones on Duke17 and Duke2013 with nothing refitted.

    Same shape as the in distribution table, so a reader compares a row here
    against a row there without re reading a header. One column, so the absolute
    backbone PSNR moves to the caption."""
    lines, seed_counts = [], []
    for i, (ds, tag) in enumerate((("Duke17", "duke17"), ("Duke2013", "duke2013"))):
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{8}{l}{\textit{%s}} \\" % ds)
        block_lines, block_seeds = transfer_block(tag)
        for row in block_lines:
            cells = row.rstrip(" \\\\").split(" & ")
            lines.append(" & ".join([cells[0]] + [cells[2]] + cells[3:]) + r" \\")
        if block_seeds:
            seed_counts.append(block_seeds)
    header = [r"\begin{tabular}{lccccccc}", r"\toprule",
              r"Backbone & $\Delta$PSNR & $\Delta$CNR & $\Delta$TCI & $\Delta$EPI & "
              r"$\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
    return "\n".join(header + lines + FOOT), (max(seed_counts) if seed_counts else 0)


def table_components():
    """NAFNet on PKU37. Above, one component removed at a time. Below, three other
    ways of spending the same budget on the same backbone output."""
    groups = (
        ("One component removed",
         [("Full method", "lopo_none"), ("No rule layer", "comp_no_negotiator"),
          ("No edge branch", "comp_no_edge"), ("No cooperation map", "comp_no_uncertainty"),
          ("No background smoothing", "comp_no_bg_smooth")]),
        ("Alternatives of similar complexity on the same backbone output",
         [("Uniform allocation, trained", "matched_plain_eval"),
          ("Unsharp masking", "classical_unsharp"),
          ("Adaptive equalization", "classical_clahe")]))
    lines = []
    for title, rows in groups:
        block = []
        for name, stem in rows:
            r = load(f"{OUT}/{stem}.json")
            if not r:
                continue
            sm = r["summary"]
            block.append(" & ".join([name, d(sm, "PSNR (dB)")] + [d(sm, k) for k in CLIN]) + r" \\")
        if block:
            if lines:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{8}{l}{\textit{%s}} \\" % title)
            lines.extend(block)
    header = [r"\begin{tabular}{lccccccc}", r"\toprule",
              r"Variant & $\Delta$PSNR (dB) & $\Delta$CNR & $\Delta$TCI & $\Delta$EPI & "
              r"$\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
    return "\n".join(header + lines + FOOT)


def adapted_block():
    """The adapted protocol of the original submission, NAFNet only, read from the
    leave one subject out result files. One seed and one fold per subject, so no
    spread and no bold. Emitted here rather than typed into the tex file, so the
    numbers cannot drift from the run that produced them."""
    key = {"CNR": "CNR", "TCI": "TCI", "EPI": "EPI", "BS": "Boundary Sharpness",
           "ENL": "ENL", "SNR": "SNR"}
    lines = []
    for ds, tag in (("Duke17", "duke17"), ("Duke2013", "duke2013")):
        f = f"{OUT}/loo_{tag}_nafnet/loo_results.json"
        r = load(f)
        if not r:
            continue
        # Compare by the tail of the path, since the result files record the
        # path as it was on the machine that produced them while OUT is resolved
        # from this file.
        want = "sw_nafnet_halo41_dz06_s0/best_model_cooperative.pth"
        got = (r.get("config", {}) or {}).get("resume_checkpoint") or ""
        if not got.endswith(want):
            raise SystemExit(
                "%s resumed from %s but the reported NAFNet model is %s. The "
                "adapted block would describe a different model from the rest of "
                "the table." % (f, got, want))
        sm = r["summary"]
        row = ["NAFNet, %s" % ds,
               "%.2f" % sm["PSNR (dB)"]["backbone_mean"],
               "%+.2f" % sm["PSNR (dB)"]["delta_mean"]]
        row += ["%+.2f" % sm[key[k]]["delta_mean"] for k in CLIN]
        lines.append(" & ".join(row) + r" \\")
    return lines


def table_full_spreads():
    """Every clinical cell's full mean plus or minus one seed standard deviation,
    four backbones by three blocks by six measures, for the supplementary
    material. tab_in_distribution.tex prints the mean alone and encodes the
    robustness test in the bold, this table gives the spread the bold was
    computed from. Generated from the same twelve test files and twelve
    zeroshot files as the merged results table, never typed by hand.
    """
    def block_rows(prefix, tag=None):
        rows = []
        for b in BB:
            seeds = collect_seeds(prefix % b)
            clin = {k: [] for k in CLIN}
            for _, payload in seeds:
                if tag is None:
                    s = payload["summary"]
                    for k in CLIN:
                        clin[k].append(s[k]["delta_mean"])
                else:
                    rv = transfer_raw_values(find_transfer_rec(payload, tag))
                    if rv is None:
                        continue
                    for k in CLIN:
                        if rv[k] is not None:
                            clin[k].append(rv[k])
            row = [LABEL[b]] + [fmt_mean_sd(clin[k]) for k in CLIN]
            rows.append(" & ".join(row) + r" \\")
        return rows

    blocks = [("PKU37 test set, 173 images", block_rows("test_%s", tag=None)),
              ("Duke17, no adaptation", block_rows("zeroshot_%s", tag="duke17")),
              ("Duke2013, no adaptation", block_rows("zeroshot_%s", tag="duke2013"))]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Backbone & $\Delta$CNR & $\Delta$TCI & $\Delta$EPI & $\Delta$BS & "
             r"$\Delta$ENL & $\Delta$SNR \\"]
    for i, (title, rows) in enumerate(blocks):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % title)
        lines.append(r"\midrule")
        lines.extend(rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_transfer():
    """Rows for the two no adaptation blocks.

    validate_crossdataset writes a list of per dataset records holding the mean of
    each measure before and after correction, not the per image summary that the
    PKU37 runs produce. The percentage change is therefore formed here, by
    transfer_raw_values, from those two means. Peak signal to noise ratio stays an
    absolute difference in decibels.

    No longer input by the main paper, since table_in_distribution now folds
    these same two blocks into the merged results table with transfer_block. Kept
    so tab_transfer.tex still exists for anything else that reads it.
    """
    def cells(rec):
        rv = transfer_raw_values(rec)
        if rv is None:
            return ["--"] * 7
        out = ["%+.2f" % rv["PSNR"]]
        for k in CLIN:
            out.append("--" if rv[k] is None else "%+.2f" % rv[k])
        return out

    lines = []
    seed_counts = []
    for ds, tag in (("Duke17", "duke17"), ("Duke2013", "duke2013")):
        lines.append(r"\multicolumn{9}{l}{\textit{%s, no adaptation}} \\" % ds)
        for b in BB:
            seeds = collect_seeds(f"zeroshot_{b}")
            if seeds:
                seed_counts.append(len(seeds))
                per_key = {"PSNR": []}
                per_key.update({k: [] for k in CLIN})
                for _, recs in seeds:
                    rv = transfer_raw_values(find_transfer_rec(recs, tag))
                    if rv is None:
                        continue
                    per_key["PSNR"].append(rv["PSNR"])
                    for k in CLIN:
                        if rv[k] is not None:
                            per_key[k].append(rv[k])
                row = ["", LABEL[b], fmt_mean_sd(per_key["PSNR"])]
                row += [bold_if_pos(fmt_mean_sd(per_key[k])) for k in CLIN]
            else:
                recs = load(f"{OUT}/zeroshot_{b}.json") or []
                rec = find_transfer_rec(recs, tag)
                vals = cells(rec)
                row = ["", LABEL[b], vals[0]] + [bold_if_pos(v) for v in vals[1:]]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    header = [r"\begin{tabular}{llccccccc}", r"\toprule",
              r"Protocol & Backbone & $\Delta$PSNR (dB) & $\Delta$CNR & $\Delta$TCI & "
              r"$\Delta$EPI & $\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
    footer = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(header + lines[:-1] + footer), (max(seed_counts) if seed_counts else 0)


def table_lopo():
    ref = load(f"{OUT}/lopo_none.json")
    rs = ref["summary"] if ref else None
    lines = [" & ".join(["all six properties", d(rs, "PSNR (dB)")] +
                        [d(rs, k) for k in CLIN]) + r" \\", r"\midrule"]
    for i in range(1, 7):
        r = load(f"{OUT}/lopo_drop_P{i}.json")
        s = r["summary"] if r else None
        lines.append(" & ".join([f"without $P_{i}$", d(s, "PSNR (dB)")] +
                                [d(s, k) for k in CLIN]) + r" \\")
    header = [r"\begin{tabular}{lccccccc}", r"\toprule",
              r"Properties used & $\Delta$PSNR (dB) & $\Delta$CNR & $\Delta$TCI & "
              r"$\Delta$EPI & $\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
    footer = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(header + lines + footer)


def table_ablation():
    rows = [# The unmodified run of the same checkpoint the component rows ablate,
            # so every row of this table describes one model. eval_nafnet.json is
            # an earlier checkpoint and must not be used here.
            ("Full method", f"{OUT}/lopo_none.json"),
            ("No rule layer", f"{OUT}/comp_no_negotiator.json"),
            ("No edge branch", f"{OUT}/comp_no_edge.json"),
            ("No cooperation map", f"{OUT}/comp_no_uncertainty.json"),
            ("No background smoothing", f"{OUT}/comp_no_bg_smooth.json"),
            (None, None),
            ("Uniform allocation, trained", f"{OUT}/matched_plain_eval.json"),
            ("Unsharp masking", f"{OUT}/classical_unsharp.json"),
            ("Adaptive equalization", f"{OUT}/classical_clahe.json")]
    lines = []
    for name, path in rows:
        if name is None:
            lines.append(r"\midrule")
            continue
        r = load(path)
        s = r["summary"] if r else None
        lines.append(" & ".join([name, d(s, "PSNR (dB)")] +
                                [d(s, k) for k in CLIN]) + r" \\")
    header = [r"\begin{tabular}{lccccccc}", r"\toprule",
              r"Configuration & $\Delta$PSNR & $\Delta$CNR & $\Delta$TCI & "
              r"$\Delta$EPI & $\Delta$BS & $\Delta$ENL & $\Delta$SNR \\", r"\midrule"]
    footer = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(header + lines + footer)


def theory_numbers():
    r = load(f"{OUT}/diagnostics_nafnet.json")
    if not r:
        return "% diagnostics not available yet\n"
    g, n = r["summary"], r["n_images"]
    out = [f"% generated from diagnostics_nafnet.json over {n} images",
           r"\newcommand{\LipTheory}{%.4f}" % r["lipschitz_constant_theory"],
           r"\newcommand{\LipMeasured}{%.4f}" % g.get("lipschitz_ratio_max", {}).get("max", 0),
           r"\newcommand{\AllocMean}{%.3f}" % g.get("alloc_mean", {}).get("mean", 0),
           r"\newcommand{\AllocStd}{%.3f}" % g.get("alloc_std", {}).get("mean", 0),
           r"\newcommand{\NImages}{%d}" % n]
    _per = r.get("per_image", [])
    if _per and "cond_margin" in _per[0]:
        out.append(r"\newcommand{\CondBoth}{%d}"
                   % sum(1 for x in _per if x.get("cond_margin") and x.get("cond_sigma")))
    for key, cmd in (("cond_margin", "CondMargin"), ("cond_sigma", "CondSigma"),
                     ("boundary_not_worsened", "BoundOK")):
        if key in g and "true" in g[key]:
            out.append(r"\newcommand{\%s}{%d}" % (cmd, g[key]["true"]))
    # The two leakage terms have means near zero with a long negative tail, so the
    # mean alone misreads them. Emit the per image sign counts as well, which is
    # what the margin condition of Theorem 2 actually depends on.
    per = r.get("per_image", [])
    if per and "invented_edge_rate_backbone" in per[0]:
        out.append(r"\newcommand{\EdgeInvUp}{%d}"
                   % sum(1 for x in per
                         if x["invented_edge_rate_corrected"] > x["invented_edge_rate_backbone"]))
        out.append(r"\newcommand{\WeakRetDown}{%d}"
                   % sum(1 for x in per
                         if x["weak_retention_corrected"] < x["weak_retention_backbone"]))
    if per and "pass_energy_descent" in per[0]:
        for key, cmd in (("pass_energy_descent", "ConEnergy"),
                         ("pass_pareto_efficient", "ConPareto"),
                         ("pass_lipschitz_bounded", "ConLip")):
            out.append(r"\newcommand{\%s}{%d}" % (cmd, sum(1 for x in per if x.get(key))))
        npass = [int(x.get("n_pass", -1)) for x in per]
        out.append(r"\newcommand{\ConAllThree}{%d}" % sum(1 for v in npass if v == 3))
        out.append(r"\newcommand{\ConMin}{%d}" % min(npass))
        out.append(r"\newcommand{\BlendOne}{%d}"
                   % sum(1 for x in per if abs(float(x.get("blend_weight", 0)) - 1.0) < 1e-9))
    if per and "eta_T" in per[0]:
        out.append(r"\newcommand{\EtaTPos}{%d}"
                   % sum(1 for x in per if x["eta_T"] > 0))
        out.append(r"\newcommand{\EtaBNeg}{%d}"
                   % sum(1 for x in per if x["eta_B"] < 0))
    if "calibration" in r:
        out.append(r"\newcommand{\CalibRho}{%.3f}" % r["calibration"]["spearman_pooled"])
        out.append(r"\newcommand{\CalibAuse}{%.3f}" % r["calibration"]["sparsification_error"])

    # The safety study answers the hallucinated boundary concern, so every measure
    # it produces is emitted as a macro rather than copied into the prose by hand.
    # Each pair is reported before and after correction so the reader sees the change.
    def mac(name, key, field="mean", scale=1.0, fmt="%.4f"):
        if key in g and field in g[key]:
            out.append(r"\newcommand{\%s}{%s}" % (name, fmt % (g[key][field] * scale)))

    mac("EdgeInvBack", "invented_edge_rate_backbone", scale=100, fmt="%.2f")
    mac("EdgeInvCorr", "invented_edge_rate_corrected", scale=100, fmt="%.2f")
    mac("WeakRetBack", "weak_retention_backbone", scale=100, fmt="%.1f")
    mac("WeakRetCorr", "weak_retention_corrected", scale=100, fmt="%.1f")
    mac("DarkFrac", "dark_region_fraction", scale=100, fmt="%.1f")
    mac("DarkChange", "dark_region_abs_change", fmt="%.4f")
    mac("GlobalMax", "global_abs_change", field="max", fmt="%.4f")
    mac("IlmShiftBack", "ilm_shift_backbone", fmt="%.2f")
    mac("IlmShiftCorr", "ilm_shift_corrected", fmt="%.2f")
    mac("RpeShiftBack", "rpe_shift_backbone", fmt="%.2f")
    mac("RpeShiftCorr", "rpe_shift_corrected", fmt="%.2f")
    mac("GlobalChange", "global_abs_change", fmt="%.4f")
    mac("AllocMin", "alloc_min", fmt="%.3f")
    mac("AllocMax", "alloc_max", fmt="%.3f")
    mac("AllocClamp", "alloc_at_clamp", scale=100, fmt="%.2f")
    mac("SigmaRatio", "sigma_ratio", fmt="%.3f")
    mac("EtaT", "eta_T", fmt="%.4f")
    mac("EtaB", "eta_B", fmt="%.4f")
    return "\n".join(out) + "\n"


def main():
    os.makedirs(DEST, exist_ok=True)
    in_dist_body, in_dist_seeds = table_in_distribution()
    zs_body, zs_seeds = table_zeroshot()
    transfer_body, transfer_seeds = table_transfer()
    parts = {"tab_in_distribution.tex": in_dist_body,
             "tab_zeroshot.tex": zs_body,
             "tab_components.tex": table_components(),
             "tab_transfer.tex": transfer_body,
             "tab_full_spreads.tex": table_full_spreads(),
             "tab_lopo.tex": table_lopo(),
             "tab_ablation.tex": table_ablation(),
             "theory_numbers.tex": theory_numbers()}
    # Number of seeds used per table, so main() can report whether a table came
    # from the new seeded sweep or fell back to the old single-run files.
    seed_info = {"tab_in_distribution.tex": in_dist_seeds, "tab_zeroshot.tex": zs_seeds,
                 "tab_transfer.tex": transfer_seeds}
    for name, body in parts.items():
        with open(os.path.join(DEST, name), "w") as f:
            # Each table file is a complete tabular (or, for theory_numbers.tex, a
            # macro file) and is included with \input from outside any alignment, so
            # a plain trailing newline is always safe.
            f.write(body + "\n")
        # A body that only says the diagnostics are absent has no dashes in it, so
        # counting dashes alone would report it as complete. Check for that first.
        if body.lstrip().startswith("%") and "not available" in body:
            status = "WAITING on diagnostics"
        else:
            have = body.count("--")
            status = f"missing {have} cells" if have else "complete"
        if name in seed_info:
            n = seed_info[name]
            status += f" ({n} seeds)" if n else " (single run)"
        print(f"  {name:28s} {status}")
    print(f"\nresult files present: {len(glob.glob(OUT + '/*.json'))}")


if __name__ == "__main__":
    main()
