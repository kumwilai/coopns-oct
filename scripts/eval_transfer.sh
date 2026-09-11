#!/usr/bin/env bash
# Table 7, upper block. Apply the selected PKU37 checkpoints to Duke17 and
# Duke2013 with no adaptation of any kind, one file per seed.
set -u
source "$(dirname "$0")/common.sh"
for BB in $BACKBONES; do
  CFG=$(winner $BB)
  [ -z "$CFG" ] && { echo "no selected setting for $BB"; continue; }
  for SEED in $SEEDS; do
    CK=$OUT/sw_${BB}_${CFG}_s${SEED}/best_model_cooperative.pth
    [ -f "$CK" ] || { echo "missing $CK"; continue; }
    [ -f "$OUT/zeroshot_${BB}_s${SEED}.json" ] && { echo "skip zeroshot $BB seed $SEED"; continue; }
    say "zeroshot $BB seed $SEED"
    $PY validate_crossdataset.py --checkpoint "$CK" \
      --backbone checkpointpaper/${BB}_backbone.pth --backbone_name $BB --device $DEVICE \
      --datasets duke17,duke2013 --output_json "$OUT/zeroshot_${BB}_s${SEED}.json" 2>&1 | tail -6
  done
done
