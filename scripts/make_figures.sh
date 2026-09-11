#!/usr/bin/env bash
# Figures 1 to 4. Writes revision/figures/. Figure 3 and Figure 4 need the result
# files and winners.json, Figures 1 and 2 need only the weights.
set -u
source "$(dirname "$0")/common.sh"
$PY revision/fig_architecture.py
$PY revision/fig_predicates.py
$PY revision/fig_studies.py --results_dir $OUT --out revision/figures/fig_studies
$PY revision/fig_subjective.py --backbone nafnet
