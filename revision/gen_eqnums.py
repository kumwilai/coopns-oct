#!/usr/bin/env python3
"""Write the main paper's equation numbers as macros for the supplement.

The supplement is a separate document, so it cannot \ref a label in the main
paper and has to print the number. Hand written numbers went stale the moment
two sections merged and the equations renumbered. These are read from the
compiled aux instead, so the supplement can never disagree with the paper.

Run after the main paper compiles and before the supplement does.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
AUX = os.path.join(_HERE, "paper", "main.aux")
DEST = os.path.join(_HERE, "sections", "generated", "eqnums.tex")

# label to macro name
WANT = {
    "eq:objective": "EqObjective", "eq:confidence": "EqConfidence",
    "eq:soft_failure": "EqFailure", "eq:allocation": "EqAllocation",
    "eq:stability_def": "EqStability", "eq:lipschitz": "EqLipschitz",
    "eq:alpha": "EqAlpha", "eq:gate": "EqGate", "eq:apply": "EqApply",
    "eq:candidate": "EqCandidate", "eq:energy": "EqEnergy", "eq:blend": "EqBlend",
    "eq:cnr_bound": "EqCnrBound", "eq:loss_fid": "EqLossFid",
    "eq:loss_clin": "EqLossClin", "eq:loss_pred": "EqLossPred",
    "eq:loss_coop": "EqLossCoop", "eq:loss_edge": "EqLossEdge",
    "eq:loss_reg": "EqLossReg", "eq:gate_halo": "EqGateHalo", "eq:rules": "EqRules",
}


def main():
    if not os.path.exists(AUX):
        sys.exit("no %s, compile the main paper first" % AUX)
    src = open(AUX, encoding="utf-8").read()
    found = {}
    for m in re.finditer(r"\\newlabel\{(eq:[a-z_]+)\}\{\{(\d+)\}", src):
        found[m.group(1)] = m.group(2)
    missing = [k for k in WANT if k not in found]
    if missing:
        sys.exit("labels absent from the compiled paper: %s" % ", ".join(sorted(missing)))
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, "w", encoding="utf-8") as f:
        f.write("% generated from paper/main.aux, do not edit\n")
        for label, macro in sorted(WANT.items(), key=lambda kv: int(found[kv[0]])):
            f.write("\\newcommand{\\%s}{%s}\n" % (macro, found[label]))
    print("  eqnums.tex                   %d equations" % len(WANT))


if __name__ == "__main__":
    main()
