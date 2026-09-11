#!/usr/bin/env bash
# Sections VII H, VII I and VII J. One pass over the test split with the selected
# NAFNet checkpoint. Produces the calibration test of the confidence map, the
# numerical check of both theorems, and the safety study. The macros quoted in the
# text (Lipschitz constant, margins, invented edge rate and so on) are generated
# from this file by revision/make_tables.py.
set -u
source "$(dirname "$0")/common.sh"
CFG=$(winner nafnet)
[ -z "$CFG" ] && { echo "no selected setting for nafnet"; exit 1; }
CK=$OUT/sw_nafnet_${CFG}_s${STUDY_SEED}/best_model_cooperative.pth
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
say "diagnostics on $CK"
$PY revision/diagnostics.py --backbone_name nafnet --checkpoint "$CK" \
  --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
  --test_jsonl "$TEST" --output_json $OUT/diagnostics_nafnet.json 2>&1 | tail -6
