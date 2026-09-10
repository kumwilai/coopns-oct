"""Choose the loss weights for each backbone from the validation results only.

Why this rule rather than the obvious one. Selecting the setting that maximises a
single measure subject to a hard constraint tends to land exactly on the
constraint boundary, and a point on the boundary of the selection set is the
first thing to fall outside it on new data. Two changes make the choice robust.
A margin is kept away from every boundary, and the selection criterion is the
worst of the six clinical measures rather than the best of one, so a setting
cannot win by sacrificing one measure to inflate another.

The ladder below is applied in order and the first level that admits any setting
decides the choice. The level that was used is recorded, so the paper can say
which backbones needed a relaxed condition.

  level 0   fidelity loss at most 0.8 dB, every clinical measure at least +0.5 percent
  level 1   fidelity loss at most 1.0 dB, every clinical measure at least +0.2 percent
  level 2   fidelity loss at most 1.0 dB, every clinical measure positive
  level 3   no fidelity bound, every clinical measure positive
  level 4   nothing qualifies, take the fewest negative measures

Within a level, the setting with the largest worst case clinical gain wins. Ties
are broken by the smaller fidelity loss, then by the mean clinical gain.
"""
import json
import glob
import os

K = ["CNR", "TCI", "EPI", "BS", "ENL", "SNR"]
CFG = {"soft":   "--clinical_weight 1.5 --tci_weight 5.0  --psnr_dead_zone 0.3 --cnr_weight 1.0",
       "mid":    "--clinical_weight 3.0 --tci_weight 6.0  --psnr_dead_zone 0.6 --cnr_weight 1.5",
       "strong": "--clinical_weight 5.0 --tci_weight 9.0  --psnr_dead_zone 0.9 --cnr_weight 2.0",
       "max":    "--clinical_weight 8.0 --tci_weight 12.0 --psnr_dead_zone 1.2 --cnr_weight 2.5"}

LEVELS = [(0.8, 0.5), (1.0, 0.2), (1.0, 0.0), (None, 0.0)]


def load(bb):
    rows = []
    for name in CFG:
        f = "outputs/revision/sw_%s_%s_val.json" % (bb, name)
        if not os.path.exists(f):
            continue
        s = json.load(open(f))["summary"]
        clin = {k: s[k]["delta_mean"] for k in K}
        rows.append({"name": name,
                     "dp": s["PSNR (dB)"]["delta_mean"],
                     "clin": clin,
                     "worst": min(clin.values()),
                     "mean": sum(clin.values()) / len(clin),
                     "neg": sum(1 for v in clin.values() if v <= 0)})
    return rows


def choose(rows):
    for lvl, (psnr_cap, floor) in enumerate(LEVELS):
        ok = [r for r in rows
              if (psnr_cap is None or r["dp"] >= -psnr_cap) and r["worst"] > floor]
        if ok:
            return max(ok, key=lambda r: (r["worst"], r["dp"], r["mean"])), lvl
    return min(rows, key=lambda r: (r["neg"], -r["worst"], -r["dp"])), 4


winners = {}
for bb in ["nafnet", "dncnn", "kbnet", "swinir"]:
    rows = load(bb)
    if not rows:
        continue
    print("\n%s" % bb.upper())
    for r in rows:
        print("  %-7s dPSNR %+6.3f  worst %+6.2f  mean %+6.2f   " % (r["name"], r["dp"], r["worst"], r["mean"])
              + " ".join("%s %+6.2f" % (k, r["clin"][k]) for k in K))
    pick, lvl = choose(rows)
    winners[bb] = {"config": pick["name"], "level": lvl,
                   "d_psnr": round(pick["dp"], 3),
                   "worst_clinical": round(pick["worst"], 2),
                   "args": CFG[pick["name"]]}
    print("  chose %s at level %d, worst clinical measure %+.2f, fidelity loss %+.3f dB"
          % (pick["name"], lvl, pick["worst"], pick["dp"]))

with open("outputs/revision/winners_per_backbone.json", "w") as f:
    json.dump(winners, f, indent=2)
print("\nsaved outputs/revision/winners_per_backbone.json")
