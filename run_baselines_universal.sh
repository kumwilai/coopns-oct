#!/usr/bin/env bash
set -euo pipefail
SIZE=64
EPOCHS=50
TRAIN=train_pairs_universal.txt
VAL=val_pairs_universal.txt
OUT_BASE=outputs/sota
MODELS=(drunet nafnet swinir noise2void speckle2speckle)

for m in "${MODELS[@]}"; do
  OUT_DIR="$OUT_BASE/universal_${m}"
  echo "Running $m -> $OUT_DIR"
  python scripts/eval_sota_from_pairs.py \
    --train_pairs "$TRAIN" \
    --val_pairs "$VAL" \
    --size $SIZE \
    --epochs $EPOCHS \
    --models "$m" \
    --out_dir "$OUT_DIR"
done
