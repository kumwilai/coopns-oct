#!/usr/bin/env bash
# Regenerate every reviewer study from the selected NAFNet checkpoint.
#
# These all have to be recomputed after the sweep, because they describe the model
# that appears in the tables and that model changes. The checkpoint and the gate
# are read from winners.json rather than typed here, so a study can never be run
# against a different model than the one the paper reports.
set -u
cd ~/research/oct
LOCK=/tmp/coopns_studies.lock; exec 9>"$LOCK"
flock -n 9 || { echo "another copy is running, exiting"; exit 0; }

PY=~/research/coopns-slr/.venv/bin/python
export LEGACY_SATURATING_ALLOCATION=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/revision
FULL=pku37_oct_dataset/pku37_real_test.jsonl
SUB=revision/pku37_subset40.jsonl
SEED="${STUDY_SEED:-0}"
say () { echo "### $* $(date +%H:%M:%S)"; }

read -r CFG BGR <<< "$($PY -c "
import json
w = json.load(open('$OUT/winners.json'))['nafnet']
print(w['config'], w['config'].split('_')[0])")"
CK=$OUT/sw_nafnet_${CFG}_s${SEED}/best_model_cooperative.pth
echo "studies use $CK with gate $BGR"
[ -f "$CK" ] || { echo "that checkpoint does not exist, stopping"; exit 1; }

score () {  # score <tag> <intervention> <jsonl>
  if [ -f "$OUT/$1.json" ] && [ "$OUT/$1.json" -nt "$CK" ]; then echo "skip $1"; return; fi
  say "score $1"
  $PY revision/ablation_runner.py --backbone_name nafnet --checkpoint "$CK" \
     --pretrained_backbone checkpointpaper/nafnet_backbone.pth --device cuda \
     --test_jsonl "$3" --intervention "$2" --bg_rule "$BGR" \
     --output_json "$OUT/$1.json" 2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -4
}

# property leave one out, for the predicate ablation the reviewers asked for
score lopo_none none "$FULL"
for P in P1 P2 P3 P4 P5 P6; do score lopo_drop_$P drop_$P "$FULL"; done

# component ablation
for A in no_negotiator no_edge no_uncertainty no_bg_smooth; do score comp_$A $A "$FULL"; done

# rule constant sensitivity, on the forty image subset
score fuzz_ref none "$SUB"
for V in -1.0 -0.5 0.5 1.0; do score fuzz_base_$V base=$V "$SUB"; done
for V in 2.0 3.0 5.0 6.0; do score fuzz_usecorr_$V rule=use_corrector:$V "$SUB"; done
for V in 1.0 2.0 4.0 5.0; do score fuzz_boost_$V rule=boost_failing:$V "$SUB"; done
for T in product minimum; do score fuzz_tnorm_$T tnorm=$T "$SUB"; done

# the theory check, the calibration test and the safety study
if [ ! -f $OUT/diagnostics_nafnet.json ] || [ "$CK" -nt $OUT/diagnostics_nafnet.json ]; then
  say "diagnostics"
  $PY revision/diagnostics.py --backbone_name nafnet --checkpoint "$CK" \
    --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
    --test_jsonl "$FULL" --output_json $OUT/diagnostics_nafnet.json 2>&1 | tail -4
fi

# matched complexity comparison, trained with the same recipe as the winner
DZ=$($PY -c "print('$CFG'.split('dz')[1].replace('015','0.15').replace('03','0.3').replace('06','0.6').replace('09','0.9').replace('15','1.5'))")
if [ ! -f "$OUT/matched_plain/best_model_cooperative.pth" ]; then
  say "matched plain corrector, dead zone $DZ, gate $BGR"
  $PY -u train_v8_cooperative.py --backbone nafnet \
    --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
    --resume checkpointpaper/nafnet_pku37_cooperative.pth --resume_epoch 1 \
    --train_jsonl pku37_oct_dataset/pku37_real_train.jsonl \
    --val_jsonl pku37_oct_dataset/pku37_real_val.jsonl \
    --epochs 10 --batch_size 16 --patch_size 96 --hidden_channels 64 \
    --device cuda --val_every 10 --max_val 8 --ablation no_negotiator \
    --seed $SEED --bg_rule $BGR --psnr_dead_zone $DZ \
    --output_dir $OUT/matched_plain 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -3
fi
if [ ! -f $OUT/matched_plain_eval.json ]; then
  say "score matched plain"
  $PY revision/ablation_runner.py --backbone_name nafnet \
     --checkpoint $OUT/matched_plain/best_model_cooperative.pth \
     --pretrained_backbone checkpointpaper/nafnet_backbone.pth --device cuda \
     --test_jsonl "$FULL" --intervention no_negotiator --bg_rule "$BGR" \
     --output_json $OUT/matched_plain_eval.json 2>&1 | tail -4
fi

# classical operators, tuned then scored at the tuned setting
[ -f $OUT/classical_tuning.json ] || $PY revision/classical_baselines.py --mode tune \
  --pretrained_backbone checkpointpaper/nafnet_backbone.pth 2>&1 | tail -6
for OP in unsharp clahe; do
  [ -f $OUT/classical_${OP}.json ] && continue
  A=$($PY -c "import json;print(json.load(open('$OUT/classical_tuning.json'))['$OP']['amount'])")
  say "classical $OP at $A"
  $PY revision/classical_baselines.py --mode eval --op $OP --amount $A \
    --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
    --jsonl "$FULL" --output_json $OUT/classical_${OP}.json 2>&1 | tail -3
done

echo "STUDIES_DONE $(date -Iseconds)"
