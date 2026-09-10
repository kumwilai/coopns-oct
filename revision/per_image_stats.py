#!/usr/bin/env python3
"""Per image evidence that a clinical measure improved, rather than a mean.

A mean across images can be carried by a handful of large gains while most images
get nothing or get worse. A reviewer who has already objected that the constants
look heuristic will not be moved by a mean. What is much harder to argue with is
the fraction of individual images on which a measure improved, together with an
exact test that this fraction is not chance.

For each measure this reports the number of images that improved, the two sided
exact binomial sign test against a fair coin, and the median change. The sign test
is used rather than a t test because the per image changes are ratios, are not
symmetric, and have heavy tails, none of which the t test tolerates well. Ties are
dropped, which is the standard conservative treatment.

Results are pooled across seeds when several are given, and the per seed agreement
is reported too, so a measure that only improves under one seed is visible.

usage
  python revision/per_image_stats.py --pattern 'outputs/revision/test_nafnet_s*.json'
  python revision/per_image_stats.py --pattern 'outputs/revision/eval_*.json'
"""
import argparse
import glob
import json
import os
from math import comb

# The per image records name the measures like this.
MEASURES = [("cnr_change_pct", "CNR"), ("tci_change_pct", "TCI"), ("epi_change_pct", "EPI"),
            ("bs_change_pct", "BS"), ("enl_change_pct", "ENL"), ("snr_change_pct", "SNR")]


def sign_test(n_pos, n_neg):
    """Two sided exact binomial test that improvement is no better than a coin flip."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def collect(paths):
    """Return {measure: [per image change, ...]} pooled, and the per file breakdown."""
    pooled = {name: [] for _, name in MEASURES}
    per_file = []
    for p in sorted(paths):
        try:
            per = json.load(open(p))["per_image"]
        except Exception:
            continue
        one = {}
        for key, name in MEASURES:
            vals = [r[key] for r in per if key in r and r[key] is not None]
            pooled[name].extend(vals)
            one[name] = vals
        per_file.append((os.path.basename(p), len(per), one))
    return pooled, per_file


def report(title, pooled, per_file):
    print("\n===== %s =====" % title)
    if per_file:
        print("  from %d file(s): %s" % (len(per_file), ", ".join(f for f, _, _ in per_file)))
    print("  %-5s %10s %9s %11s %12s" % ("", "improved", "of", "median", "sign test p"))
    for _, name in MEASURES:
        v = pooled[name]
        pos = sum(1 for x in v if x > 0)
        neg = sum(1 for x in v if x < 0)
        p = sign_test(pos, neg)
        frac = 100.0 * pos / len(v) if v else float("nan")
        star = "" if p >= 0.05 else ("  ***" if p < 1e-4 else "  *")
        print("  %-5s %6d %4.0f%% %9d %+11.2f %12.2e%s"
              % (name, pos, frac, len(v), median(v), p, star))
    if len(per_file) > 1:
        print("  per file fraction improved, to show seed agreement")
        for f, n, one in per_file:
            cells = []
            for _, name in MEASURES:
                v = one[name]
                cells.append("%s %3.0f%%" % (name, 100.0 * sum(1 for x in v if x > 0) / max(1, len(v))))
            print("    %-34s %s" % (f, "  ".join(cells)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", action="append", required=True,
                    help="Glob of result files. Repeat to compare groups.")
    args = ap.parse_args()
    for pat in args.pattern:
        paths = glob.glob(pat)
        if not paths:
            print("\nno files match %s" % pat)
            continue
        pooled, per_file = collect(paths)
        report(pat, pooled, per_file)


if __name__ == "__main__":
    main()
