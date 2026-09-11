#!/usr/bin/env bash
# Sourced by every driver in this directory. Nothing here is machine specific.
# Override any variable from the environment, for example
#   DEVICE=cuda PY=/path/to/python bash scripts/train_all.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python3}"
DEVICE="${DEVICE:-cpu}"
export LEGACY_SATURATING_ALLOCATION="${LEGACY_SATURATING_ALLOCATION:-0}"
export PYTHONUNBUFFERED=1
OUT=outputs/revision
TRAIN=pku37_oct_dataset/pku37_real_train.jsonl
VAL=pku37_oct_dataset/pku37_real_val.jsonl
TEST=pku37_oct_dataset/pku37_real_test.jsonl
SUB=revision/pku37_subset40.jsonl
BACKBONES="${BACKBONES:-nafnet dncnn kbnet swinir}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-10}"
STUDY_SEED="${STUDY_SEED:-0}"
# The search space of the paper. One entry is name:background_rule:extra flags.
# The name begins with the gate it was trained with, and the drivers rely on that.
CONFIGS="${CONFIGS:-int_dz03:intensity:--psnr_dead_zone 0.3|int_dz06:intensity:--psnr_dead_zone 0.6|int_dz09:intensity:--psnr_dead_zone 0.9|otsu_dz03:otsu:--psnr_dead_zone 0.3|otsu_dz06:otsu:--psnr_dead_zone 0.6|otsu_dz09:otsu:--psnr_dead_zone 0.9}"
mkdir -p "$OUT" logs

say () { echo "### $* $(date +%H:%M:%S)"; }

# Batch size is a property of the backbone, not of the setting. Sixteen crops,
# reduced to eight for the two backbones with the largest memory footprint.
bs_for () { case $1 in swinir|dncnn) echo 8 ;; *) echo 16 ;; esac; }

# score <tag> <backbone> <checkpoint> <jsonl> <background_rule>
# The gate must match the one the checkpoint was trained with, otherwise the
# model is scored under a rule it never saw.
score () {
  if [ -f "$OUT/$1.json" ] && [ "$OUT/$1.json" -nt "$3" ]; then echo "skip $1, already scored"; return; fi
  say "score $1"
  $PY revision/ablation_runner.py --backbone_name "$2" --checkpoint "$3" \
     --pretrained_backbone "checkpointpaper/${2}_backbone.pth" --device "$DEVICE" \
     --test_jsonl "$4" --intervention none --bg_rule "$5" \
     --output_json "$OUT/$1.json" 2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -6
}

# winner <backbone>  prints the selected setting name, or nothing if none chosen
winner () {
  [ -f "$OUT/winners.json" ] || return 0
  $PY -c "import json; w=json.load(open('$OUT/winners.json')); print(w['$1']['config'] if '$1' in w else '')"
}
