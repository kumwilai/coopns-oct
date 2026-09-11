#!/usr/bin/env bash
# The recipe that produced every corrector in the revised paper.
#
# Every setting in CONFIGS is trained at every seed in SEEDS for every backbone,
# and each run is scored on the validation split. Nothing here is chosen by hand.
# The choice is made afterwards by scripts/select_config.sh, on validation only,
# and the chosen checkpoints are then scored on test unchanged.
#
# Every run starts from the corrector of the submitted version of the paper
# (checkpointpaper/BACKBONE_pku37_cooperative.pth) and trains for EPOCHS epochs
# with the repaired allocation rule. That warm start is part of the recipe and is
# stated in the paper. The backbone is frozen throughout, including its
# normalisation statistics.
#
# Only two things vary between settings, the fidelity dead zone and the
# background rule. The flags --clinical_weight, --tci_weight and --cnr_weight
# still exist in the parser of train_v8_cooperative.py but have no effect, see
# README.md, section "Defects found during the revision".
set -u
source "$(dirname "$0")/common.sh"
for BB in $BACKBONES; do
  BS=$(bs_for $BB)
  IFS='|' read -ra CFGLIST <<< "$CONFIGS"
  for ENTRY in "${CFGLIST[@]}"; do
    NAME="${ENTRY%%:*}"; REST="${ENTRY#*:}"; BGR="${REST%%:*}"; FLAGS="${REST#*:}"
    for SEED in $SEEDS; do
      TAG=sw_${BB}_${NAME}_s${SEED}
      D=$OUT/$TAG
      if [ ! -f "$D/best_model_cooperative.pth" ]; then
        say "train $BB $NAME seed $SEED  (dead zone and gate: $FLAGS --bg_rule $BGR)"
        $PY -u train_v8_cooperative.py --backbone $BB \
          --pretrained_backbone checkpointpaper/${BB}_backbone.pth \
          --resume checkpointpaper/${BB}_pku37_cooperative.pth --resume_epoch 1 \
          --train_jsonl $TRAIN --val_jsonl $VAL \
          --epochs $EPOCHS --batch_size $BS --patch_size 96 \
          --hidden_channels 64 --device $DEVICE --val_every $EPOCHS --max_val 8 \
          --seed $SEED --bg_rule $BGR \
          --output_dir "$D" $FLAGS 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tee logs/$TAG.log | tail -4
      fi
      score "${TAG}_val" $BB "$D/best_model_cooperative.pth" "$VAL" "$BGR"
    done
  done
done
echo "TRAIN_ALL_DONE $(date -Iseconds)"
