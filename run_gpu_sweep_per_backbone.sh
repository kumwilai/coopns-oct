#!/usr/bin/env bash
# Choose the loss weights separately for every backbone, on the validation split
# only. The test split is never used to select anything.
set -u
cd ~/research/oct
LOCK=/tmp/coopns_sweep.lock; exec 9>"$LOCK"
flock -n 9 || { echo "already running"; exit 0; }
PY=~/research/coopns-slr/.venv/bin/python
export LEGACY_SATURATING_ALLOCATION=0 PYTHONUNBUFFERED=1
OUT=outputs/revision; mkdir -p $OUT logs
VAL=pku37_oct_dataset/pku37_real_val.jsonl
TRAIN=pku37_oct_dataset/pku37_real_train.jsonl
EP=10; BS=16
say () { echo "### $* $(date +%H:%M:%S)"; }

# name : clinical : tci : dead zone : cnr
CFGS="soft:1.5:5.0:0.3:1.0 mid:3.0:6.0:0.6:1.5 strong:5.0:9.0:0.9:2.0 max:8.0:12.0:1.2:2.5"

for BB in nafnet dncnn kbnet swinir; do
  for CFG in $CFGS; do
    N=${CFG%%:*}; R=${CFG#*:}; CW=${R%%:*}; R=${R#*:}; TW=${R%%:*}; R=${R#*:}; DZ=${R%%:*}; NW=${R##*:}
    D=$OUT/sw_${BB}_${N}
    if [ ! -f "$D/best_model_cooperative.pth" ]; then
      say "train $BB $N"
      $PY -u train_v8_cooperative.py --backbone $BB \
        --pretrained_backbone checkpointpaper/${BB}_backbone.pth \
        --resume checkpointpaper/${BB}_pku37_cooperative.pth --resume_epoch 1 \
        --train_jsonl $TRAIN --val_jsonl $VAL --epochs $EP --batch_size $BS \
        --patch_size 96 --hidden_channels 64 --device cuda --val_every $EP --max_val 8 \
        --clinical_weight $CW --tci_weight $TW --psnr_dead_zone $DZ --cnr_weight $NW \
        --output_dir "$D" 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -3
    fi
    if [ ! -f "$OUT/sw_${BB}_${N}_val.json" ]; then
      say "score $BB $N on validation"
      $PY revision/ablation_runner.py --backbone_name $BB \
        --checkpoint "$D/best_model_cooperative.pth" \
        --pretrained_backbone checkpointpaper/${BB}_backbone.pth --device cuda \
        --test_jsonl "$VAL" --intervention none \
        --output_json "$OUT/sw_${BB}_${N}_val.json" \
        2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -9
    fi
  done
done
say "per backbone sweep finished"
$PY pick_winners.py
echo "SWEEP_PER_BACKBONE_DONE $(date -Iseconds)"
