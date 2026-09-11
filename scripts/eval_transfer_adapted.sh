#!/usr/bin/env bash
# Table 7, lower block. The adapted protocol, leave one subject out on each Duke
# set, starting every fold from the selected PKU37 checkpoint at STUDY_SEED and
# fine tuning on the other subjects of that set. This is the protocol the
# submitted paper actually used, described exactly in Section VII A 2.
# It trains one model per fold, so it is the slowest script here.
set -u
source "$(dirname "$0")/common.sh"
for BB in $BACKBONES; do
  CFG=$(winner $BB)
  [ -z "$CFG" ] && { echo "no selected setting for $BB"; continue; }
  CK=$OUT/sw_${BB}_${CFG}_s${STUDY_SEED}/best_model_cooperative.pth
  [ -f "$CK" ] || { echo "missing $CK"; continue; }
  for DS in duke17 duke2013; do
    say "adapted $DS $BB"
    $PY run_duke17_loo.py --dataset $DS --backbone $BB --resume "$CK" \
      --pretrained_backbone checkpointpaper/${BB}_backbone.pth \
      --device $DEVICE --output_dir $OUT/adapted_${DS}_${BB} 2>&1 | tail -8
  done
done
