#!/usr/bin/env bash
# Train every setting at every seed, score each on validation, then score the
# SELECTED checkpoints on test without retraining them.
#
# The difference from the previous driver is the last part. Previously a setting
# was chosen on validation and then a fresh unseeded model was trained with that
# setting and reported, so the model in the tables was never the model that was
# selected. Here the checkpoints that won are the checkpoints that get scored.
set -u
cd ~/research/oct
LOCK=/tmp/coopns_sweep_v2.lock; exec 9>"$LOCK"
flock -n 9 || { echo "another copy is running, exiting"; exit 0; }
echo "lock held by $$ at $(date -Iseconds)"

PY=~/research/coopns-slr/.venv/bin/python
export LEGACY_SATURATING_ALLOCATION=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

OUT=outputs/revision
TRAIN=pku37_oct_dataset/pku37_real_train.jsonl
VAL=pku37_oct_dataset/pku37_real_val.jsonl
TEST=pku37_oct_dataset/pku37_real_test.jsonl
mkdir -p $OUT logs

BACKBONES="${BACKBONES:-nafnet dncnn kbnet swinir}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-10}"

# The search space. One entry is "name:bg_rule:flags". The background rule is part
# of the setting rather than a global, so that the two gates can be compared on
# validation like any other choice instead of being decided in advance.
CONFIGS="${CONFIGS:-otsu_dz015:otsu:--psnr_dead_zone 0.15|otsu_dz03:otsu:--psnr_dead_zone 0.3|otsu_dz06:otsu:--psnr_dead_zone 0.6|otsu_dz09:otsu:--psnr_dead_zone 0.9|otsu_dz15:otsu:--psnr_dead_zone 1.5}"

say () { echo "### $* $(date +%H:%M:%S)"; }

# Batch size is a property of the backbone, not of the setting, so it is fixed
# here and used identically for every run including the ones that get reported.
bs_for () { case $1 in swinir|dncnn) echo 8 ;; *) echo 16 ;; esac; }

score () {  # score <tag> <backbone> <ckpt> <jsonl> <bg_rule>
  # The gate must match the one the checkpoint was trained with, otherwise the
  # model is scored under a rule it never saw.
  if [ -f "$OUT/$1.json" ] && [ "$OUT/$1.json" -nt "$3" ]; then echo "skip $1"; return; fi
  say "score $1"
  $PY revision/ablation_runner.py --backbone_name $2 --checkpoint "$3" \
     --pretrained_backbone checkpointpaper/${2}_backbone.pth --device cuda \
     --test_jsonl "$4" --intervention none --bg_rule "$5" \
     --output_json "$OUT/$1.json" 2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -6
}

# ---- train and score every setting at every seed on validation ----
for BB in $BACKBONES; do
  BS=$(bs_for $BB)
  IFS='|' read -ra CFGLIST <<< "$CONFIGS"
  for ENTRY in "${CFGLIST[@]}"; do
    NAME="${ENTRY%%:*}"; REST="${ENTRY#*:}"; BGR="${REST%%:*}"; FLAGS="${REST#*:}"
    for SEED in $SEEDS; do
      TAG=sw_${BB}_${NAME}_s${SEED}
      D=$OUT/$TAG
      if [ ! -f "$D/best_model_cooperative.pth" ]; then
        say "train $BB $NAME seed $SEED"
        $PY -u train_v8_cooperative.py --backbone $BB \
          --pretrained_backbone checkpointpaper/${BB}_backbone.pth \
          --resume checkpointpaper/${BB}_pku37_cooperative.pth --resume_epoch 1 \
          --train_jsonl $TRAIN --val_jsonl $VAL \
          --epochs $EPOCHS --batch_size $BS --patch_size 96 \
          --hidden_channels 64 --device cuda --val_every $EPOCHS --max_val 8 \
          --seed $SEED --bg_rule $BGR \
          --output_dir "$D" $FLAGS 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -4
      fi
      score "${TAG}_val" $BB "$D/best_model_cooperative.pth" "$VAL" "$BGR"
    done
  done
done

# ---- choose, on validation only ----
say "select"
$PY revision/select_config.py --sweep_dir $OUT --out $OUT/winners.json \
    --backbones "$(echo $BACKBONES | tr ' ' ',')" | tail -40

# ---- score the chosen checkpoints on test, unchanged ----
for BB in $BACKBONES; do
  CFG=$($PY -c "
import json,sys
w=json.load(open('$OUT/winners.json'))
print(w['$BB']['config'] if '$BB' in w else '')" )
  [ -z "$CFG" ] && { echo "no winner for $BB, skipping"; continue; }
  CFG_BGR="${CFG%%_*}"   # the setting name begins with the gate it was trained with
  for SEED in $SEEDS; do
    CK=$OUT/sw_${BB}_${CFG}_s${SEED}/best_model_cooperative.pth
    [ -f "$CK" ] || continue
    score "test_${BB}_s${SEED}" $BB "$CK" "$TEST" "$CFG_BGR"
    if [ ! -f "$OUT/zeroshot_${BB}_s${SEED}.json" ]; then
      say "zeroshot $BB seed $SEED"
      $PY validate_crossdataset.py --checkpoint "$CK" \
        --backbone checkpointpaper/${BB}_backbone.pth --backbone_name $BB --device cuda \
        --datasets duke17,duke2013 --output_json "$OUT/zeroshot_${BB}_s${SEED}.json" 2>&1 | tail -6
    fi
  done
done

echo "SWEEP_V2_DONE $(date -Iseconds)"
