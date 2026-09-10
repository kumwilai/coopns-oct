#!/usr/bin/env bash
# Retrain every corrector on PKU37 after the allocation fix.
# This script is the recorded recipe for the revision. Every number in the
# revised paper comes from a model produced by exactly this command.
set -u
PY=/home/kumwilai/osmnx-env/bin/python
cd /home/kumwilai/OCT
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export LEGACY_SATURATING_ALLOCATION=0

EPOCHS=10
OUT=outputs/revision

for BB in nafnet dncnn kbnet swinir; do
  DIR=$OUT/retrain_${BB}
  if [ -f "$DIR/best_model_cooperative.pth" ]; then echo "skip $BB, already trained"; continue; fi
  echo "=== retraining $BB  $(date -Iseconds) ==="
  $PY -u train_v8_cooperative.py \
      --backbone $BB \
      --pretrained_backbone checkpointpaper/${BB}_backbone.pth \
      --resume checkpointpaper/${BB}_pku37_cooperative.pth \
      --resume_epoch 1 \
      --train_jsonl pku37_oct_dataset/pku37_real_train.jsonl \
      --val_jsonl pku37_oct_dataset/pku37_real_val.jsonl \
      --epochs $EPOCHS --batch_size 4 --patch_size 96 \
      --hidden_channels 64 --device cpu \
      --val_every 5 --max_val 4 \
      --output_dir "$DIR" 2>&1 | tail -60
  echo "=== finished $BB  $(date -Iseconds) ==="
done
echo "RETRAIN_ALL_DONE $(date -Iseconds)"
