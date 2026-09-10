#!/usr/bin/env bash
# Tune the loss weights on the validation split only. The test set is never seen.
# Each configuration trains a NAFNet corrector and is scored on the validation
# images. The winner is then used to retrain every backbone.
set -u
PY=/home/kumwilai/osmnx-env/bin/python
cd /home/kumwilai/OCT
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LEGACY_SATURATING_ALLOCATION=0
OUT=outputs/revision/sweep
mkdir -p $OUT

# name : clinical weight : tci weight : psnr dead zone : cnr weight
CONFIGS="
base:1.5:5.0:0.3:1.0
clin3:3.0:5.0:0.6:1.5
clin5:5.0:8.0:0.9:2.0
clin8:8.0:12.0:1.2:2.5
"

for CFG in $CONFIGS; do
  NAME=$(echo $CFG | cut -d: -f1)
  CW=$(echo $CFG | cut -d: -f2)
  TW=$(echo $CFG | cut -d: -f3)
  DZ=$(echo $CFG | cut -d: -f4)
  NW=$(echo $CFG | cut -d: -f5)
  D=$OUT/$NAME
  [ -f "$D/best_model_cooperative.pth" ] && { echo "skip $NAME"; continue; }
  echo "### sweep $NAME  clinical $CW  tci $TW  dead zone $DZ  cnr $NW  $(date -Iseconds)"
  $PY -u train_v8_cooperative.py --backbone nafnet \
      --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
      --resume checkpointpaper/nafnet_pku37_cooperative.pth --resume_epoch 1 \
      --train_jsonl pku37_oct_dataset/pku37_real_train.jsonl \
      --val_jsonl pku37_oct_dataset/pku37_real_val.jsonl \
      --epochs 8 --batch_size 4 --patch_size 96 --hidden_channels 64 --device cpu \
      --clinical_weight $CW --tci_weight $TW --psnr_dead_zone $DZ --cnr_weight $NW \
      --val_every 8 --max_val 4 --output_dir $D 2>&1 | tail -12
  # score on the validation split, never on the test split
  $PY revision/ablation_runner.py --backbone_name nafnet \
      --checkpoint $D/best_model_cooperative.pth \
      --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
      --test_jsonl pku37_oct_dataset/pku37_real_val.jsonl \
      --intervention none --output_json $OUT/${NAME}_val.json 2>&1 | tail -11
done
echo "SWEEP_DONE $(date -Iseconds)"
