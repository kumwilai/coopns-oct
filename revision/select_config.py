#!/usr/bin/env python3
"""Choose one training setting per backbone from validation results across seeds.

Why this replaces pick_winners.py.

The old rule scored one unseeded run per setting, took the setting with the best
worst clinical measure, and then trained a fresh unseeded model with that setting
and reported it. Three things went wrong with that. Nothing was seeded, so the run
that was scored and the run that was reported were different models. The final run
also used a different batch size, so it was not even the same recipe. And the
margins that decided the choice were smaller than the spread between two runs of
the same setting, so the choice was close to a coin toss. KBNet is the visible
casualty. Its selected setting held every clinical measure positive on validation
and lost two of them on test.

This version fixes all three. Every setting is trained at several seeds. A setting
is admissible only when it clears the bounds at its WORST seed, not on average, so
a setting cannot qualify by being lucky once. Among admissible settings the winner
is the one with the largest lower confidence bound on its worst clinical measure,
which prefers a setting that is reliably good over one that is occasionally
excellent. The chosen checkpoints are then scored on test unchanged. Nothing is
retrained after selection, so the reported model is the selected model.

usage
  python revision/select_config.py --sweep_dir outputs/revision --out outputs/revision/winners.json
"""
import argparse
import glob
import json
import os
import re
import statistics

CLINICAL = ["CNR", "TCI", "EPI", "BS", "ENL", "SNR"]

# Applied in order. The first level that admits any setting decides the choice, and
# the level used is recorded so the paper can say which backbones needed a looser
# condition rather than hiding it.
LEVELS = [
    ("strict",   0.8, 0.5),
    ("standard", 1.0, 0.2),
    ("positive", 1.0, 0.0),
    ("unbounded", None, 0.0),
]


def load_runs(sweep_dir, backbone):
    """Group validation results by setting name, collecting every seed of each."""
    runs = {}
    pat = re.compile(r"sw_%s_(?P<cfg>.+?)_s(?P<seed>\d+)_val\.json$" % re.escape(backbone))
    for path in sorted(glob.glob(os.path.join(sweep_dir, "sw_%s_*_val.json" % backbone))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        try:
            summary = json.load(open(path))["summary"]
        except Exception:
            continue
        if not all(k in summary for k in CLINICAL + ["PSNR (dB)"]):
            continue
        runs.setdefault(m.group("cfg"), []).append({
            "seed": int(m.group("seed")),
            "path": path,
            "d_psnr": summary["PSNR (dB)"]["delta_mean"],
            "clinical": {k: summary[k]["delta_mean"] for k in CLINICAL},
        })
    return runs


def summarise(name, seeds):
    """Reduce the seeds of one setting to the statistics the choice is made on."""
    n = len(seeds)
    per_measure = {k: [s["clinical"][k] for s in seeds] for k in CLINICAL}
    worst_per_seed = [min(s["clinical"].values()) for s in seeds]
    mean_worst = statistics.mean(worst_per_seed)
    # One sided lower confidence bound on the worst measure. With a single seed the
    # spread is unknown, so no credit is given for it.
    sd_worst = statistics.stdev(worst_per_seed) if n > 1 else 0.0
    lcb = mean_worst - (sd_worst / (n ** 0.5) if n > 1 else 0.0)
    d_psnrs = [s["d_psnr"] for s in seeds]
    return {
        "config": name,
        "n_seeds": n,
        "seeds": sorted(s["seed"] for s in seeds),
        "d_psnr_mean": statistics.mean(d_psnrs),
        "d_psnr_worst": min(d_psnrs),
        "d_psnr_sd": statistics.stdev(d_psnrs) if n > 1 else 0.0,
        "clinical_mean": {k: statistics.mean(v) for k, v in per_measure.items()},
        "clinical_min": {k: min(v) for k, v in per_measure.items()},
        "worst_measure_min": min(worst_per_seed),
        "worst_measure_mean": mean_worst,
        "worst_measure_lcb": lcb,
        "n_negative_worst_seed": sum(1 for k in CLINICAL if min(per_measure[k]) <= 0),
        "checkpoints": [s["path"].replace("_val.json", "") for s in sorted(seeds, key=lambda s: s["seed"])],
    }


def choose(rows):
    """Return the winning setting and the name of the level that admitted it."""
    for level_name, psnr_cap, floor in LEVELS:
        ok = [r for r in rows
              if (psnr_cap is None or r["d_psnr_worst"] >= -psnr_cap)
              and r["worst_measure_min"] > floor]
        if ok:
            return max(ok, key=lambda r: (r["worst_measure_lcb"],
                                          r["d_psnr_mean"],
                                          statistics.mean(r["clinical_mean"].values()))), level_name
    # Nothing cleared any level. Take the setting that fails on the fewest measures,
    # and say so plainly rather than presenting it as a selection.
    return min(rows, key=lambda r: (r["n_negative_worst_seed"],
                                    -r["worst_measure_lcb"],
                                    -r["d_psnr_mean"])), "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="outputs/revision")
    ap.add_argument("--out", default="outputs/revision/winners.json")
    ap.add_argument("--backbones", default="nafnet,dncnn,swinir,kbnet")
    ap.add_argument("--include", default="",
                    help="Only consider settings whose name contains this string. "
                         "Used to select within one background gate at a time.")
    ap.add_argument("--min_seeds", type=int, default=2,
                    help="Settings with fewer seeds than this are reported but never chosen.")
    args = ap.parse_args()

    winners = {}
    for bb in args.backbones.split(","):
        runs = load_runs(args.sweep_dir, bb)
        if not runs:
            print("\n%s: no seeded validation results found" % bb.upper())
            continue
        if args.include:
            runs = {k: v for k, v in runs.items() if args.include in k}
            if not runs:
                print("\n%s: no settings match %r" % (bb.upper(), args.include))
                continue
        rows = [summarise(name, seeds) for name, seeds in sorted(runs.items())]
        print("\n%s" % bb.upper())
        for r in rows:
            print("  %-10s seeds %-9s dPSNR mean %+6.3f worst %+6.3f sd %5.3f | "
                  "worst clinical min %+6.2f mean %+6.2f lcb %+6.2f"
                  % (r["config"], ",".join(map(str, r["seeds"])), r["d_psnr_mean"],
                     r["d_psnr_worst"], r["d_psnr_sd"], r["worst_measure_min"],
                     r["worst_measure_mean"], r["worst_measure_lcb"]))

        eligible = [r for r in rows if r["n_seeds"] >= args.min_seeds]
        if not eligible:
            print("  no setting has %d seeds yet, nothing chosen" % args.min_seeds)
            continue
        pick, level = choose(eligible)
        pick = dict(pick, level=level)
        winners[bb] = pick
        note = "" if level != "none" else "   NOTHING QUALIFIED, reporting the least bad setting"
        print("  chose %s at level %s, worst clinical lower bound %+.2f, fidelity %+.3f dB%s"
              % (pick["config"], level, pick["worst_measure_lcb"], pick["d_psnr_mean"], note))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(winners, f, indent=2)
    print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()
