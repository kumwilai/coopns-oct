#!/usr/bin/env python3
"""
Aggregate per-seed JSON results into mean/std tables.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load_paths(inputs: str) -> list[Path]:
    paths: list[Path] = []
    for token in inputs.split(","):
        token = token.strip()
        if not token:
            continue
        expanded = glob.glob(token)
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(token))
    return paths


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1, "ci_low": values[0], "ci_high": values[0]}
    mu = mean(values)
    sd = stdev(values)
    ci = 1.96 * sd / (len(values) ** 0.5)
    return {"mean": mu, "std": sd, "n": len(values), "ci_low": mu - ci, "ci_high": mu + ci}


def fmt_summary(summary: dict, digits: int, include_ci: bool) -> str:
    if summary["n"] == 0:
        return "n/a"
    mean_fmt = f"{summary['mean']:.{digits}f}"
    std_fmt = f"{summary['std']:.{digits}f}"
    if include_ci and summary["n"] > 1:
        ci_low = f"{summary['ci_low']:.{digits}f}"
        ci_high = f"{summary['ci_high']:.{digits}f}"
        return f"{mean_fmt} ± {std_fmt} [{ci_low}, {ci_high}]"
    return f"{mean_fmt} ± {std_fmt}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True, help="Comma-separated paths or globs.")
    parser.add_argument("--group_key", type=str, default="model")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_tex", type=str, default="")
    parser.add_argument("--no_ci", action="store_true")
    args = parser.parse_args()

    paths = load_paths(args.inputs)
    if not paths:
        raise SystemExit("No input files found.")

    groups = defaultdict(list)
    for path in paths:
        if not path.is_file():
            print(f"⚠ Skipping missing file: {path}")
            continue
        data = json.loads(path.read_text())
        key = data.get(args.group_key, "unknown")
        groups[key].append(data)

    print("=" * 80)
    print("Aggregated Results")
    print("=" * 80)

    include_ci = not args.no_ci
    csv_rows = []
    tex_rows = []

    for key in sorted(groups.keys()):
        entries = groups[key]
        psnr_vals = [e.get("psnr_mean") for e in entries if e.get("psnr_mean") is not None]
        ssim_vals = [e.get("ssim_mean") for e in entries if e.get("ssim_mean") is not None]
        top1_vals = [e.get("top1") for e in entries if e.get("top1") is not None]

        psnr = summarize(psnr_vals)
        ssim = summarize(ssim_vals)
        top1 = summarize(top1_vals)

        print(f"\n{key}")
        print(f"  PSNR: {fmt_summary(psnr, 2, include_ci)} (n={psnr['n']})")
        print(f"  SSIM: {fmt_summary(ssim, 4, include_ci)} (n={ssim['n']})")
        if top1["n"] > 0:
            print(f"  Top-1: {fmt_summary(top1, 1, include_ci)} (n={top1['n']})")

        csv_rows.append(
            {
                "model": key,
                "psnr_mean": psnr["mean"],
                "psnr_std": psnr["std"],
                "psnr_ci_low": psnr.get("ci_low"),
                "psnr_ci_high": psnr.get("ci_high"),
                "psnr_n": psnr["n"],
                "ssim_mean": ssim["mean"],
                "ssim_std": ssim["std"],
                "ssim_ci_low": ssim.get("ci_low"),
                "ssim_ci_high": ssim.get("ci_high"),
                "ssim_n": ssim["n"],
                "top1_mean": top1["mean"],
                "top1_std": top1["std"],
                "top1_ci_low": top1.get("ci_low"),
                "top1_ci_high": top1.get("ci_high"),
                "top1_n": top1["n"],
            }
        )
        tex_rows.append(
            {
                "model": key,
                "psnr": fmt_summary(psnr, 2, include_ci),
                "ssim": fmt_summary(ssim, 4, include_ci),
                "top1": fmt_summary(top1, 1, include_ci) if top1["n"] > 0 else "n/a",
            }
        )

    print("=" * 80)

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "model",
            "psnr_mean",
            "psnr_std",
            "psnr_ci_low",
            "psnr_ci_high",
            "psnr_n",
            "ssim_mean",
            "ssim_std",
            "ssim_ci_low",
            "ssim_ci_high",
            "ssim_n",
            "top1_mean",
            "top1_std",
            "top1_ci_low",
            "top1_ci_high",
            "top1_n",
        ]
        lines = [",".join(header)]
        for row in csv_rows:
            lines.append(
                ",".join(
                    "" if row[k] is None else f"{row[k]}"
                    for k in header
                )
            )
        out_path.write_text("\n".join(lines) + "\n")
        print(f"✓ Wrote {out_path}")

    if args.out_tex:
        out_path = Path(args.out_tex)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Model & PSNR (mean$\\pm$std) & SSIM (mean$\\pm$std) & Top-1 (mean$\\pm$std) \\\\",
            "\\midrule",
        ]
        for row in tex_rows:
            lines.append(f"{row['model']} & {row['psnr']} & {row['ssim']} & {row['top1']} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        out_path.write_text("\n".join(lines) + "\n")
        print(f"✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
