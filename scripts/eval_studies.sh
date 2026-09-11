#!/usr/bin/env bash
# Figure 3 and Table 8. Every study runs on the selected NAFNet checkpoint at
# STUDY_SEED, read from winners.json so a study can never be run against a
# different model than the one the tables report.
#   lopo_*        remove one clinical property at a time         (Section VII D)
#   fuzz_*        rule constants and two other conjunctions      (Section VII E)
#   comp_*        remove one component at a time                 (Section VII F)
#   matched_plain corrector trained without the rule layer       (Section VII G)
#   classical_*   unsharp masking and adaptive equalisation      (Section VII G)
set -u
source "$(dirname "$0")/common.sh"
CFG=$(winner nafnet)
[ -z "$CFG" ] && { echo "no selected setting for nafnet, run scripts/select_config.sh first"; exit 1; }
BGR="${CFG%%_*}"; [ "$BGR" = int ] && BGR=intensity
CK=$OUT/sw_nafnet_${CFG}_s${STUDY_SEED}/best_model_cooperative.pth
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
echo "studies use $CK with gate $BGR"

study () {  # study <tag> <intervention> <jsonl>
  if [ -f "$OUT/$1.json" ] && [ "$OUT/$1.json" -nt "$CK" ]; then echo "skip $1"; return; fi
  say "study $1"
  $PY revision/ablation_runner.py --backbone_name nafnet --checkpoint "$CK" \
     --pretrained_backbone checkpointpaper/nafnet_backbone.pth --device $DEVICE \
     --test_jsonl "$3" --intervention "$2" --bg_rule "$BGR" \
     --output_json "$OUT/$1.json" 2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -4
}

study lopo_none none "$TEST"
for P in P1 P2 P3 P4 P5 P6; do study lopo_drop_$P drop_$P "$TEST"; done
for A in no_negotiator no_edge no_uncertainty no_bg_smooth; do study comp_$A $A "$TEST"; done
study fuzz_ref none "$SUB"
for V in -1.0 -0.5 0.5 1.0; do study fuzz_base_$V base=$V "$SUB"; done
for V in 2.0 3.0 5.0 6.0; do study fuzz_usecorr_$V rule=use_corrector:$V "$SUB"; done
for V in 1.0 2.0 4.0 5.0; do study fuzz_boost_$V rule=boost_failing:$V "$SUB"; done
for T in product minimum; do study fuzz_tnorm_$T tnorm=$T "$SUB"; done

# The matched complexity comparison. Same recipe as the winner, same warm start,
# same dead zone, same gate and same seed, with the rule layer replaced by a
# uniform allocation for the whole of training (--ablation no_negotiator).
DZ=$($PY -c "print({'03':'0.3','06':'0.6','09':'0.9','015':'0.15','15':'1.5'}['$CFG'.split('dz')[1]])")
if [ ! -f "$OUT/matched_plain/best_model_cooperative.pth" ]; then
  say "train matched plain corrector, dead zone $DZ, gate $BGR"
  $PY -u train_v8_cooperative.py --backbone nafnet \
    --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
    --resume checkpointpaper/nafnet_pku37_cooperative.pth --resume_epoch 1 \
    --train_jsonl $TRAIN --val_jsonl $VAL \
    --epochs $EPOCHS --batch_size 16 --patch_size 96 --hidden_channels 64 \
    --device $DEVICE --val_every $EPOCHS --max_val 8 --ablation no_negotiator \
    --seed $STUDY_SEED --bg_rule $BGR --psnr_dead_zone $DZ \
    --output_dir $OUT/matched_plain 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -3
fi
if [ ! -f $OUT/matched_plain_eval.json ]; then
  say "score matched plain"
  $PY revision/ablation_runner.py --backbone_name nafnet \
     --checkpoint $OUT/matched_plain/best_model_cooperative.pth \
     --pretrained_backbone checkpointpaper/nafnet_backbone.pth --device $DEVICE \
     --test_jsonl "$TEST" --intervention no_negotiator --bg_rule "$BGR" \
     --output_json $OUT/matched_plain_eval.json 2>&1 | tail -4
fi

# Classical operators, tuned on the forty image subset, then scored on the test split.
[ -f $OUT/classical_tuning.json ] || $PY revision/classical_baselines.py --mode tune \
  --pretrained_backbone checkpointpaper/nafnet_backbone.pth --output_json $OUT/classical_tuning.json 2>&1 | tail -6
for OP in unsharp clahe; do
  [ -f $OUT/classical_${OP}.json ] && continue
  A=$($PY -c "import json;print(json.load(open('$OUT/classical_tuning.json'))['$OP']['amount'])")
  say "classical $OP at $A"
  $PY revision/classical_baselines.py --mode eval --op $OP --amount $A \
    --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
    --jsonl "$TEST" --output_json $OUT/classical_${OP}.json 2>&1 | tail -3
done
echo "STUDIES_DONE $(date -Iseconds)"
