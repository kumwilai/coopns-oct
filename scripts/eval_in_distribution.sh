#!/usr/bin/env bash
# Table 6. Score the SELECTED checkpoints on the PKU37 test split, one file per
# seed, without retraining anything. revision/make_tables.py averages the seeds.
set -u
source "$(dirname "$0")/common.sh"
for BB in $BACKBONES; do
  CFG=$(winner $BB)
  [ -z "$CFG" ] && { echo "no selected setting for $BB, run scripts/select_config.sh first"; continue; }
  BGR="${CFG%%_*}"; [ "$BGR" = int ] && BGR=intensity
  for SEED in $SEEDS; do
    CK=$OUT/sw_${BB}_${CFG}_s${SEED}/best_model_cooperative.pth
    [ -f "$CK" ] || { echo "missing $CK"; continue; }
    score "test_${BB}_s${SEED}" $BB "$CK" "$TEST" "$BGR"
  done
done
