#!/usr/bin/env bash
# Everything, on the GPU, in one pass. Safe to rerun, it skips finished work.
set -u
cd ~/research/oct
PY=~/research/coopns-slr/.venv/bin/python
export LEGACY_SATURATING_ALLOCATION=0 PYTHONUNBUFFERED=1
OUT=outputs/revision
mkdir -p $OUT logs
FULL=pku37_oct_dataset/pku37_real_test.jsonl
VAL=pku37_oct_dataset/pku37_real_val.jsonl
TRAIN=pku37_oct_dataset/pku37_real_train.jsonl
SUB=revision/pku37_subset40.jsonl
EP=10
BS=16

say () { echo "### $* $(date +%H:%M:%S)"; }

train () {   # train <outdir> <backbone> [extra args...]
  local d=$1 bb=$2; shift 2
  [ -f "$d/best_model_cooperative.pth" ] && { echo "skip train $d"; return; }
  say "train $d"
  $PY -u train_v8_cooperative.py --backbone $bb \
     --pretrained_backbone checkpointpaper/${bb}_backbone.pth \
     --resume checkpointpaper/${bb}_pku37_cooperative.pth --resume_epoch 1 \
     --train_jsonl $TRAIN --val_jsonl $VAL \
     --epochs $EP --batch_size $BS --patch_size 96 --hidden_channels 64 \
     --device cuda --val_every $EP --max_val 8 --output_dir "$d" "$@" \
     2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -6
}

score () {   # score <tag> <backbone> <ckpt> <intervention> <jsonl>
  [ -f "$OUT/$1.json" ] && { echo "skip $1"; return; }
  say "score $1"
  $PY revision/ablation_runner.py --backbone_name $2 --checkpoint "$3" \
     --pretrained_backbone checkpointpaper/${2}_backbone.pth \
     --device cuda --test_jsonl "$5" --intervention "$4" \
     --output_json "$OUT/$1.json" 2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -11
}

# ---- phase A, choose the loss weights on the validation split only ----
for CFG in "base:1.5:5.0:0.3:1.0" "clin3:3.0:6.0:0.6:1.5" "clin5:5.0:9.0:0.9:2.0" "clin8:8.0:12.0:1.2:2.5"; do
  N=${CFG%%:*}; R=${CFG#*:}; CW=${R%%:*}; R=${R#*:}; TW=${R%%:*}; R=${R#*:}; DZ=${R%%:*}; NW=${R##*:}
  train "$OUT/sweep_$N" nafnet --clinical_weight $CW --tci_weight $TW --psnr_dead_zone $DZ --cnr_weight $NW
  score "sweep_${N}_val" nafnet "$OUT/sweep_$N/best_model_cooperative.pth" none "$VAL"
done
say "phase A done"

echo "GPU_PIPELINE_PHASE_A_DONE $(date -Iseconds)"
