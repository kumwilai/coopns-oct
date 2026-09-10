#!/usr/bin/env bash
# Phase two. Runs after the evaluation queue. Closes the remaining reviewer points.
set -u
PY=/home/kumwilai/osmnx-env/bin/python
cd /home/kumwilai/OCT
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LEGACY_SATURATING_ALLOCATION=0
OUT=outputs/revision
CK=$OUT/retrain_nafnet/best_model_cooperative.pth
BB=checkpointpaper/nafnet_backbone.pth
FULL=pku37_oct_dataset/pku37_real_test.jsonl

while pgrep -f run_revision_evals.sh > /dev/null; do sleep 180; done
echo "evaluation queue finished, starting phase two $(date -Iseconds)"

# 1. diagnostics, closes reviewer 1 comment 1, reviewer 3 comments 11, 12 and 19
if [ ! -f $OUT/diagnostics_nafnet.json ]; then
  echo "### diagnostics $(date -Iseconds)"
  $PY revision/diagnostics.py --backbone_name nafnet --checkpoint "$CK" \
      --pretrained_backbone "$BB" --test_jsonl "$FULL" \
      --output_json $OUT/diagnostics_nafnet.json 2>&1 | tail -6
fi

# 2. a plain corrector of the same size, same objective, no rule layer and no properties
if [ ! -f $OUT/matched_plain/best_model_cooperative.pth ]; then
  echo "### training the matched plain corrector $(date -Iseconds)"
  $PY -u train_v8_cooperative.py --backbone nafnet --pretrained_backbone "$BB" \
      --resume checkpointpaper/nafnet_pku37_cooperative.pth --resume_epoch 1 \
      --train_jsonl pku37_oct_dataset/pku37_real_train.jsonl \
      --val_jsonl pku37_oct_dataset/pku37_real_val.jsonl \
      --epochs 10 --batch_size 4 --patch_size 96 --hidden_channels 64 --device cpu \
      --val_every 5 --max_val 4 --ablation no_negotiator \
      --output_dir $OUT/matched_plain 2>&1 | tail -25
fi
if [ ! -f $OUT/matched_plain_eval.json ] && [ -f $OUT/matched_plain/best_model_cooperative.pth ]; then
  echo "### evaluating the matched plain corrector $(date -Iseconds)"
  $PY revision/ablation_runner.py --backbone_name nafnet \
      --checkpoint $OUT/matched_plain/best_model_cooperative.pth \
      --pretrained_backbone "$BB" --test_jsonl "$FULL" --intervention no_negotiator \
      --output_json $OUT/matched_plain_eval.json 2>&1 | tail -12
fi

# 3. classical operators, tuned on the subset then scored on the full test set
if [ ! -f $OUT/classical_tuning.json ]; then
  echo "### tuning the classical operators $(date -Iseconds)"
  $PY revision/classical_baselines.py --mode tune --pretrained_backbone "$BB" 2>&1 | tail -14
fi
$PY - <<'PYX'
import json, os, subprocess
t = json.load(open("outputs/revision/classical_tuning.json"))
for op, best in t.items():
    out = f"outputs/revision/classical_{op}.json"
    if os.path.exists(out):
        print("skip", op); continue
    subprocess.run(["/home/kumwilai/osmnx-env/bin/python", "revision/classical_baselines.py",
                    "--mode", "eval", "--op", op, "--amount", str(best["amount"]),
                    "--pretrained_backbone", "checkpointpaper/nafnet_backbone.pth",
                    "--output_json", out])
PYX

echo "PHASE2_DONE $(date -Iseconds)"
