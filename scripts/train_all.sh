#!/usr/bin/env bash
# The one recipe that produced every corrector in the paper.
set -eu
export LEGACY_SATURATING_ALLOCATION=0
for BB in nafnet dncnn kbnet swinir; do
  python3 code/train_v8_cooperative.py \
    --backbone $BB \
    --pretrained_backbone weights/${BB}_backbone.pth \
    --train_jsonl data/pku37_real_train.jsonl \
    --val_jsonl data/pku37_real_val.jsonl \
    --epochs 10 --batch_size 4 --patch_size 96 \
    --hidden_channels 64 --device cpu \
    --val_every 5 --max_val 4 \
    --output_dir outputs/retrain_${BB}
done
