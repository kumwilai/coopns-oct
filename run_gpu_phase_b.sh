#!/usr/bin/env bash
# Phase B onward. Score the sweep on validation, pick the winner, then run
# everything that the paper needs. Safe to rerun.
set -u
cd ~/research/oct

# Refuse to run twice. A second copy would write the same files and corrupt them.
LOCK=/tmp/coopns_gpu_pipeline.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another copy of this pipeline is already running, exiting"
  exit 0
fi
echo "lock acquired by pid $$ at $(date -Iseconds)"
PY=~/research/coopns-slr/.venv/bin/python
export LEGACY_SATURATING_ALLOCATION=0 PYTHONUNBUFFERED=1
OUT=outputs/revision
FULL=pku37_oct_dataset/pku37_real_test.jsonl
VAL=pku37_oct_dataset/pku37_real_val.jsonl
TRAIN=pku37_oct_dataset/pku37_real_train.jsonl
SUB=revision/pku37_subset40.jsonl
EP=10; BS=16
say () { echo "### $* $(date +%H:%M:%S)"; }

score () {   # score <tag> <backbone> <ckpt> <intervention> <jsonl>
  [ -f "$OUT/$1.json" ] && { echo "skip $1"; return; }
  say "score $1"
  $PY revision/ablation_runner.py --backbone_name $2 --checkpoint "$3" \
     --pretrained_backbone checkpointpaper/${2}_backbone.pth --device cuda \
     --test_jsonl "$5" --intervention "$4" --output_json "$OUT/$1.json" \
     2>&1 | grep -vE "Evaluating|it/s\]|s/img\]" | tail -10
}

say "phase A finished, scoring the sweep on the validation split"
for N in base clin3 clin5 clin8; do
  [ -f "$OUT/sweep_$N/best_model_cooperative.pth" ] || continue
  score "sweep_${N}_val" nafnet "$OUT/sweep_$N/best_model_cooperative.pth" none "$VAL"
done

WIN=$($PY - <<'PYX'
import json, glob, os
best, bestkey = None, None
for f in sorted(glob.glob("outputs/revision/sweep_*_val.json")):
    s = json.load(open(f))["summary"]
    dp = s["PSNR (dB)"]["delta_mean"]
    clin = {k: s[k]["delta_mean"] for k in ("CNR","TCI","EPI","BS","ENL","SNR")}
    name = os.path.basename(f).replace("sweep_","").replace("_val.json","")
    ok = dp > -1.0 and clin["CNR"] > 0 and clin["TCI"] > 0
    key = clin["CNR"] if ok else -999 + clin["CNR"]
    print(f"  {name:8s} dPSNR {dp:+.3f}  CNR {clin['CNR']:+.2f}  TCI {clin['TCI']:+.2f}  "
          f"ENL {clin['ENL']:+.2f}  {'ok' if ok else 'rejected'}")
    if best is None or key > best:
        best, bestkey = key, name
print("WINNER=" + (bestkey or "base"))
PYX
)
echo "$WIN"
NAME=$(echo "$WIN" | grep -o "WINNER=.*" | cut -d= -f2)
say "winning configuration is $NAME"

case $NAME in
  base)  ARGS="--clinical_weight 1.5 --tci_weight 5.0  --psnr_dead_zone 0.3 --cnr_weight 1.0" ;;
  clin3) ARGS="--clinical_weight 3.0 --tci_weight 6.0  --psnr_dead_zone 0.6 --cnr_weight 1.5" ;;
  clin5) ARGS="--clinical_weight 5.0 --tci_weight 9.0  --psnr_dead_zone 0.9 --cnr_weight 2.0" ;;
  clin8) ARGS="--clinical_weight 8.0 --tci_weight 12.0 --psnr_dead_zone 1.2 --cnr_weight 2.5" ;;
esac
echo "$NAME $ARGS" > $OUT/winning_recipe.txt

# ---- phase B, train every backbone with the winning recipe ----
for BB in nafnet dncnn kbnet swinir; do
  D=$OUT/final_$BB
  [ -f "$D/best_model_cooperative.pth" ] && { echo "skip train $BB"; continue; }
  say "train final $BB"
  $PY -u train_v8_cooperative.py --backbone $BB \
     --pretrained_backbone checkpointpaper/${BB}_backbone.pth \
     --resume checkpointpaper/${BB}_pku37_cooperative.pth --resume_epoch 1 \
     --train_jsonl $TRAIN --val_jsonl $VAL --epochs $EP --batch_size $BS \
     --patch_size 96 --hidden_channels 64 --device cuda --val_every $EP --max_val 8 \
     --output_dir "$D" $ARGS 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -5
done

# ---- phase C, results on the test set ----
for BB in nafnet dncnn kbnet swinir; do
  score "eval_$BB" $BB "$OUT/final_$BB/best_model_cooperative.pth" none "$FULL"
done

# ---- phase D, transfer with no adaptation ----
for BB in nafnet dncnn kbnet swinir; do
  [ -f "$OUT/zeroshot_$BB.json" ] && { echo "skip zeroshot $BB"; continue; }
  say "zeroshot $BB"
  $PY validate_crossdataset.py --checkpoint "$OUT/final_$BB/best_model_cooperative.pth" \
     --backbone checkpointpaper/${BB}_backbone.pth --backbone_name $BB --device cuda \
     --datasets duke17,duke2013 --output_json "$OUT/zeroshot_$BB.json" 2>&1 | tail -20
done

# ---- phase E, the studies the reviewers asked for ----
CK=$OUT/final_nafnet/best_model_cooperative.pth
score lopo_none nafnet "$CK" none "$FULL"
for P in P1 P2 P3 P4 P5 P6; do score lopo_drop_$P nafnet "$CK" drop_$P "$FULL"; done
for A in no_negotiator no_edge no_uncertainty no_bg_smooth; do score comp_$A nafnet "$CK" $A "$FULL"; done
score fuzz_ref nafnet "$CK" none "$SUB"
for V in -1.0 -0.5 0.5 1.0; do score fuzz_base_$V nafnet "$CK" base=$V "$SUB"; done
for V in 2.0 3.0 5.0 6.0; do score fuzz_usecorr_$V nafnet "$CK" rule=use_corrector:$V "$SUB"; done
for V in 1.0 2.0 4.0 5.0; do score fuzz_boost_$V nafnet "$CK" rule=boost_failing:$V "$SUB"; done
for T in product minimum; do score fuzz_tnorm_$T nafnet "$CK" tnorm=$T "$SUB"; done

# ---- phase F, diagnostics, matched baseline, classical operators ----
if [ ! -f $OUT/diagnostics_nafnet.json ]; then
  say "diagnostics"
  $PY revision/diagnostics.py --backbone_name nafnet --checkpoint "$CK" \
     --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
     --test_jsonl "$FULL" --output_json $OUT/diagnostics_nafnet.json 2>&1 | tail -5
fi
if [ ! -f "$OUT/matched_plain/best_model_cooperative.pth" ]; then
  say "matched plain corrector"
  $PY -u train_v8_cooperative.py --backbone nafnet \
     --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
     --resume checkpointpaper/nafnet_pku37_cooperative.pth --resume_epoch 1 \
     --train_jsonl $TRAIN --val_jsonl $VAL --epochs $EP --batch_size $BS --patch_size 96 \
     --hidden_channels 64 --device cuda --val_every $EP --max_val 8 --ablation no_negotiator \
     --output_dir $OUT/matched_plain $ARGS 2>&1 | tr '\r' '\n' | grep -viE "^\s*$|it/s\]" | tail -4
fi
score matched_plain_eval nafnet "$OUT/matched_plain/best_model_cooperative.pth" no_negotiator "$FULL"
[ -f $OUT/classical_tuning.json ] || $PY revision/classical_baselines.py --mode tune \
   --pretrained_backbone checkpointpaper/nafnet_backbone.pth 2>&1 | tail -10

echo "GPU_ALL_DONE $(date -Iseconds)"
