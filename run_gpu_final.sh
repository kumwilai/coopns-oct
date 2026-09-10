#!/usr/bin/env bash
# Train the final model for every backbone using its own selected weights, then
# run every evaluation and study the paper needs.
set -u
cd ~/research/oct
LOCK=/tmp/coopns_final.lock; exec 9>"$LOCK"
flock -n 9 || { echo "another copy is running, exiting"; exit 0; }
echo "lock held by $$ at $(date -Iseconds)"
PY=~/research/coopns-slr/.venv/bin/python
export LEGACY_SATURATING_ALLOCATION=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# Record memory every twenty seconds so a failure can be diagnosed rather than guessed at.
( while true; do
    printf "%s  RAM %s  GPU %s\n" "$(date +%H:%M:%S)" \
      "$(free -m | awk 'NR==2{print $3"/"$2" MB"}')" \
      "$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
    sleep 20
  done > logs/memwatch.log 2>&1 ) &
WATCHER=$!
trap 'kill $WATCHER 2>/dev/null' EXIT
OUT=outputs/revision
FULL=pku37_oct_dataset/pku37_real_test.jsonl
VAL=pku37_oct_dataset/pku37_real_val.jsonl
TRAIN=pku37_oct_dataset/pku37_real_train.jsonl
SUB=revision/pku37_subset40.jsonl
say () { echo "### $* $(date +%H:%M:%S)"; }

score () {  # score <tag> <backbone> <ckpt> <intervention> <jsonl>
  # Skip only when the result is newer than the checkpoint it came from. A result
  # older than its model is stale and must be recomputed, otherwise the paper
  # would report numbers that the released weights do not reproduce.
  if [ -f "$OUT/$1.json" ]; then
    if [ "$OUT/$1.json" -nt "$3" ]; then echo "skip $1"; return; fi
    echo "  $1 is older than its checkpoint, recomputing"
    mkdir -p "$OUT/superseded" && mv -f "$OUT/$1.json" "$OUT/superseded/"
  fi
  say "score $1"
  $PY revision/ablation_runner.py --backbone_name $2 --checkpoint "$3" \
     --pretrained_backbone checkpointpaper/${2}_backbone.pth --device cuda \
     --test_jsonl "$5" --intervention "$4" --output_json "$OUT/$1.json" \
     2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -10
}

# ---- final training, one recipe per backbone ----
for BB in nafnet dncnn kbnet swinir; do
  D=$OUT/final_$BB
  ARGS=$($PY -c "import json;print(json.load(open('$OUT/winners_per_backbone.json'))['$BB']['args'])")
  case $BB in swinir|dncnn) BS=8 ;; *) BS=16 ;; esac
  if [ ! -f "$D/best_model_cooperative.pth" ]; then
    say "train final $BB with $ARGS"
    $PY -u train_v8_cooperative.py --backbone $BB \
      --pretrained_backbone checkpointpaper/${BB}_backbone.pth \
      --resume checkpointpaper/${BB}_pku37_cooperative.pth --resume_epoch 1 \
      --train_jsonl $TRAIN --val_jsonl $VAL --epochs 10 --batch_size $BS --patch_size 96 \
      --hidden_channels 64 --device cuda --val_every 10 --max_val 2 \
      --patches_per_image 1 \
      --output_dir "$D" $ARGS 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -4
  fi
done

# ---- results on the test set ----
for BB in nafnet dncnn kbnet swinir; do
  score "eval_$BB" $BB "$OUT/final_$BB/best_model_cooperative.pth" none "$FULL"
done

# ---- transfer with no adaptation ----
for BB in nafnet dncnn kbnet swinir; do
  [ -f "$OUT/zeroshot_$BB.json" ] && { echo "skip zeroshot $BB"; continue; }
  say "zeroshot $BB"
  $PY validate_crossdataset.py --checkpoint "$OUT/final_$BB/best_model_cooperative.pth" \
    --backbone checkpointpaper/${BB}_backbone.pth --backbone_name $BB --device cuda \
    --datasets duke17,duke2013 --output_json "$OUT/zeroshot_$BB.json" 2>&1 | tail -18
done

# ---- the studies the reviewers asked for ----
CK=$OUT/final_nafnet/best_model_cooperative.pth
score lopo_none nafnet "$CK" none "$FULL"
for P in P1 P2 P3 P4 P5 P6; do score lopo_drop_$P nafnet "$CK" drop_$P "$FULL"; done
for A in no_negotiator no_edge no_uncertainty no_bg_smooth; do score comp_$A nafnet "$CK" $A "$FULL"; done
score fuzz_ref nafnet "$CK" none "$SUB"
for V in -1.0 -0.5 0.5 1.0; do score fuzz_base_$V nafnet "$CK" base=$V "$SUB"; done
for V in 2.0 3.0 5.0 6.0; do score fuzz_usecorr_$V nafnet "$CK" rule=use_corrector:$V "$SUB"; done
for V in 1.0 2.0 4.0 5.0; do score fuzz_boost_$V nafnet "$CK" rule=boost_failing:$V "$SUB"; done
for T in product minimum; do score fuzz_tnorm_$T nafnet "$CK" tnorm=$T "$SUB"; done

# ---- diagnostics, matched baseline, classical operators ----
if [ ! -f $OUT/diagnostics_nafnet.json ]; then
  say "diagnostics"
  $PY revision/diagnostics.py --backbone_name nafnet --checkpoint "$CK" \
    --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
    --test_jsonl "$FULL" --output_json $OUT/diagnostics_nafnet.json 2>&1 | tail -4
fi
NAF_ARGS=$($PY -c "import json;print(json.load(open('$OUT/winners_per_backbone.json'))['nafnet']['args'])")
if [ ! -f "$OUT/matched_plain/best_model_cooperative.pth" ]; then
  say "matched plain corrector"
  $PY -u train_v8_cooperative.py --backbone nafnet \
    --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
    --resume checkpointpaper/nafnet_pku37_cooperative.pth --resume_epoch 1 \
    --train_jsonl $TRAIN --val_jsonl $VAL --epochs 10 --batch_size 16 --patch_size 96 \
    --hidden_channels 64 --device cuda --val_every 10 --max_val 4 --ablation no_negotiator \
    --output_dir $OUT/matched_plain $NAF_ARGS 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -3
fi
score matched_plain_eval nafnet "$OUT/matched_plain/best_model_cooperative.pth" no_negotiator "$FULL"
[ -f $OUT/classical_tuning.json ] || $PY revision/classical_baselines.py --mode tune \
  --pretrained_backbone checkpointpaper/nafnet_backbone.pth 2>&1 | tail -8

echo "ALL_DONE $(date -Iseconds)"
