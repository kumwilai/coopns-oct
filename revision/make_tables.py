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

OUT = "outputs/revision"
DEST = "revision/sections/generated"
BB = ["nafnet", "dncnn", "swinir", "kbnet"]
LABEL = {"nafnet": "NAFNet", "dncnn": "DnCNN", "kbnet": "KBNet", "swinir": "SwinIR"}
CLIN = ["CNR", "TCI", "EPI", "BS", "ENL", "SNR"]


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
    return f"$\\mathbf{{{txt}}}$" if txt.startswith("+") else txt


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


def table_in_distribution():
    lines, means = [], {k: [] for k in ["PSNR (dB)"] + CLIN}
    seed_counts = []
    for b in BB:
        seeds = collect_seeds(f"test_{b}")
        if seeds:
            seed_counts.append(len(seeds))
            psnr_deltas, psnr_correcteds = [], []
            clin_deltas = {k: [] for k in CLIN}
            for _, r in seeds:
                s = r["summary"]
                psnr_deltas.append(s["PSNR (dB)"]["delta_mean"])
                psnr_correcteds.append(s["PSNR (dB)"]["corrected_mean"])
                for k in CLIN:
                    clin_deltas[k].append(s[k]["delta_mean"])
            abs_psnr = f"{statistics.mean(psnr_correcteds):.2f}"
            row = [LABEL[b], abs_psnr, fmt_mean_sd(psnr_deltas)]
            row += [bold_if_pos(fmt_mean_sd(clin_deltas[k])) for k in CLIN]
            lines.append(" & ".join(row) + r" \\")
            means["PSNR (dB)"].append(statistics.mean(psnr_deltas))
            for k in CLIN:
                means[k].append(statistics.mean(clin_deltas[k]))
        else:
            r = load(f"{OUT}/eval_{b}.json")
            s = r["summary"] if r else None
            abs_psnr = f"{s['PSNR (dB)']['corrected_mean']:.2f}" if s else "--"
            row = [LABEL[b], abs_psnr, d(s, "PSNR (dB)")] + [bold_if_pos(d(s, k)) for k in CLIN]
            lines.append(" & ".join(row) + r" \\")
            if s:
                means["PSNR (dB)"].append(s["PSNR (dB)"]["delta_mean"])
                for k in CLIN:
                    means[k].append(s[k]["delta_mean"])
    if means["PSNR (dB)"]:
        mrow = ["\\textit{mean}", "", f"{sum(means['PSNR (dB)'])/len(means['PSNR (dB)']):+.2f}"]
        mrow += [f"{sum(means[k])/len(means[k]):+.2f}" for k in CLIN]
        lines.append(r"\midrule")
        lines.append(" & ".join(mrow) + r" \\")
    return "\n".join(lines), (max(seed_counts) if seed_counts else 0)


def table_transfer():
    """Rows for the two no adaptation blocks.

    validate_crossdataset writes a list of per dataset records holding the mean of
    each measure before and after correction, not the per image summary that the
    PKU37 runs produce. The percentage change is therefore formed here from those
    two means. Peak signal to noise ratio stays an absolute difference in decibels.
    """
    PAIR = {"CNR": "cnr", "TCI": "tci", "EPI": "epi", "BS": "boundary_sharpness",
            "ENL": "enl", "SNR": "snr"}

    def raw_values(rec):
        """psnr_delta plus each clinical measure's percentage change, or None if missing."""
        if rec is None:
            return None
        out = {"PSNR": rec["psnr_delta"]}
        for k in CLIN:
            b, c = rec.get(PAIR[k] + "_backbone"), rec.get(PAIR[k] + "_corrected")
            out[k] = None if b is None or c is None else 100.0 * (c - b) / (abs(b) + 1e-12)
        return out

    def cells(rec):
        rv = raw_values(rec)
        if rv is None:
            return ["--"] * 7
        out = ["%+.2f" % rv["PSNR"]]
        for k in CLIN:
            out.append("--" if rv[k] is None else "%+.2f" % rv[k])
        return out

    def find_rec(recs, tag):
        return next((r for r in recs
                     if tag in str(r.get("dataset", "")).lower().replace("-", "")), None)

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
                    rv = raw_values(find_rec(recs, tag))
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
                rec = find_rec(recs, tag)
                vals = cells(rec)
                row = ["", LABEL[b], vals[0]] + [bold_if_pos(v) for v in vals[1:]]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1]), (max(seed_counts) if seed_counts else 0)


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
    return "\n".join(lines)


def table_ablation():
    rows = [("Full method", f"{OUT}/eval_nafnet.json"),
            ("No rule layer", f"{OUT}/comp_no_negotiator.json"),
            ("No edge branch", f"{OUT}/comp_no_edge.json"),
            ("No confidence", f"{OUT}/comp_no_uncertainty.json"),
            ("No background smoothing", f"{OUT}/comp_no_bg_smooth.json"),
            (None, None),
            ("Plain corrector, same size", f"{OUT}/matched_plain_eval.json"),
            ("Unsharp masking", f"{OUT}/classical_unsharp.json"),
            ("Adaptive equalisation", f"{OUT}/classical_clahe.json")]
    lines = []
    for name, path in rows:
        if name is None:
            lines.append(r"\midrule")
            continue
        r = load(path)
        s = r["summary"] if r else None
        lines.append(" & ".join([name, d(s, "PSNR (dB)")] +
                                [d(s, k) for k in CLIN[:5]]) + r" \\")
    return "\n".join(lines)


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
    for key, cmd in (("cond_margin", "CondMargin"), ("cond_sigma", "CondSigma"),
                     ("boundary_not_worsened", "BoundOK")):
        if key in g and "true" in g[key]:
            out.append(r"\newcommand{\%s}{%d}" % (cmd, g[key]["true"]))
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
    transfer_body, transfer_seeds = table_transfer()
    parts = {"tab_in_distribution.tex": in_dist_body,
             "tab_transfer.tex": transfer_body,
             "tab_lopo.tex": table_lopo(),
             "tab_ablation.tex": table_ablation(),
             "theory_numbers.tex": theory_numbers()}
    # Number of seeds used per table, so main() can report whether a table came
    # from the new seeded sweep or fell back to the old single-run files.
    seed_info = {"tab_in_distribution.tex": in_dist_seeds, "tab_transfer.tex": transfer_seeds}
    for name, body in parts.items():
        with open(os.path.join(DEST, name), "w") as f:
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
