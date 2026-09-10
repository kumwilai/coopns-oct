#!/usr/bin/env bash
# Everything that has to run after the sweep, in priority order, unattended.
#
# Ordered so that if the night runs short the essential results exist anyway.
# Every stage is idempotent, so this script can be restarted and will resume.
# A stage that fails is recorded and the next one still runs, because one broken
# study should not cost the whole night.
set -u
cd ~/research/oct
PY=~/research/coopns-slr/.venv/bin/python
export LEGACY_SATURATING_ALLOCATION=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/revision
mkdir -p logs $OUT
STATUS=logs/overnight_status.txt
: > $STATUS

note () { echo "[$(date +%H:%M:%S)] $*" | tee -a $STATUS; }
stage () {  # stage <name> <command...>
  local name="$1"; shift
  note "START $name"
  if "$@" >> "logs/stage_${name}.log" 2>&1; then
    note "OK    $name"
  else
    note "FAIL  $name (see logs/stage_${name}.log)"
  fi
}

# ---- wait for the sweep, however long it takes ----
note "waiting for the sweep to finish"
while pgrep -f run_sweep_v2.sh > /dev/null; do sleep 120; done
if grep -q SWEEP_V2_DONE logs/sweep_v2.log 2>/dev/null; then
  note "sweep finished cleanly"
else
  note "WARNING sweep stopped without reaching its end marker, continuing on what exists"
fi

# A second pass over the same sweep. The first pass lost every SwinIR run that used
# the percentile gate to a CUDA graph fault, and the loop had already moved past
# them by the time it was fixed. The driver skips whatever is already on disk, so
# this costs nothing when there is nothing missing and fills the gaps when there is.
MISSING=0
for BB in nafnet dncnn kbnet swinir; do
  for C in otsu_dz03 int_dz03 otsu_dz06 int_dz06 otsu_dz09 int_dz09; do
    for S in 0 1 2; do
      [ -f "$OUT/sw_${BB}_${C}_s${S}_val.json" ] || MISSING=$((MISSING+1))
    done
  done
done
note "second pass, $MISSING runs still missing"
if [ "$MISSING" -gt 0 ]; then
  BACKBONES="nafnet dncnn kbnet swinir" SEEDS="0 1 2" EPOCHS=10 \
  CONFIGS="otsu_dz03:otsu:--psnr_dead_zone 0.3|int_dz03:intensity:--psnr_dead_zone 0.3|otsu_dz06:otsu:--psnr_dead_zone 0.6|int_dz06:intensity:--psnr_dead_zone 0.6|otsu_dz09:otsu:--psnr_dead_zone 0.9|int_dz09:intensity:--psnr_dead_zone 0.9" \
  ./run_sweep_v2.sh >> logs/sweep_v2_pass2.log 2>&1
  note "second pass finished"
fi

# ---- 1b. the halo gate ----
# The dead zone extension that used to sit here has been cancelled. Measurement
# showed the extra budget buys texture contrast out of the background band and
# crosses one decibel to do it, which is the wrong trade.
#
# Instead, vary the radius of the tissue mask. The contrast and texture gains come
# from the edge branch acting just outside tissue, so a tight mask removes them and
# an unbounded gate lets corrections reach the background the noise measures are
# computed on. At inference, with weights trained under the loose gate, a radius of
# 41 pixels already makes every measure on NAFNet positive at only -0.09 dB, which
# is a better worst case than either trained gate. This trains at that radius.
note "halo gate sweep"
BACKBONES="nafnet kbnet dncnn" SEEDS="0 1 2" EPOCHS=10 \
CONFIGS="halo41_dz03:halo41:--psnr_dead_zone 0.3|halo61_dz03:halo61:--psnr_dead_zone 0.3|halo41_dz06:halo41:--psnr_dead_zone 0.6" \
./run_sweep_v2.sh >> logs/sweep_halo.log 2>&1
note "halo sweep finished"

# ---- 1. select within each gate separately, so both options are on the table ----
# The sweep already selected across everything and scored that winner. These two
# extra selections let the morning compare the gates on equal terms rather than
# being handed one of them.
stage select_otsu $PY revision/select_config.py --sweep_dir $OUT \
      --include otsu_ --out $OUT/winners_otsu.json
stage select_intensity $PY revision/select_config.py --sweep_dir $OUT \
      --include int_ --out $OUT/winners_intensity.json

# ---- 2. score the runner up gate on test, so the comparison is like for like ----
TEST=pku37_oct_dataset/pku37_real_test.jsonl
for GATE in otsu intensity; do
  W=$OUT/winners_${GATE}.json
  [ -f "$W" ] || { note "SKIP  test_${GATE}, no winners file"; continue; }
  for BB in nafnet dncnn kbnet swinir; do
    CFG=$($PY -c "
import json
try: print(json.load(open('$W')).get('$BB',{}).get('config',''))
except Exception: print('')")
    [ -z "$CFG" ] && continue
    BGR="${CFG%%_*}"
    for SEED in 0 1 2; do
      CK=$OUT/sw_${BB}_${CFG}_s${SEED}/best_model_cooperative.pth
      TAG=gate_${GATE}_${BB}_s${SEED}
      [ -f "$CK" ] || continue
      [ -f "$OUT/$TAG.json" ] && continue
      note "score $TAG"
      $PY revision/ablation_runner.py --backbone_name $BB --checkpoint "$CK" \
        --pretrained_backbone checkpointpaper/${BB}_backbone.pth --device cuda \
        --test_jsonl "$TEST" --intervention none --bg_rule "$BGR" \
        --output_json "$OUT/$TAG.json" >> logs/stage_gate_scores.log 2>&1 \
        || note "FAIL  $TAG"
    done
  done
done

# ---- 3. every reviewer study, from the selected NAFNet checkpoint ----
stage studies ./run_studies_v2.sh

# ---- 4. leave one out, retrained from the new checkpoint rather than the old one ----
# This is the block the transfer table cannot fill from anything on disk, because
# the only results that exist predate the allocation fix.
NAF_CFG=$($PY -c "
import json
try: print(json.load(open('$OUT/winners.json'))['nafnet']['config'])
except Exception: print('')")
if [ -n "$NAF_CFG" ]; then
  NAF_CK=$OUT/sw_nafnet_${NAF_CFG}_s0/best_model_cooperative.pth
  NAF_BGR="${NAF_CFG%%_*}"
  for DS in duke17 duke2013; do
    [ -f "$OUT/loo_${DS}_nafnet/loo_results.json" ] && { note "SKIP  loo_$DS, already present"; continue; }
    note "START loo_$DS from $NAF_CK"
    $PY run_duke17_loo.py --backbone nafnet --dataset $DS \
      --resume "$NAF_CK" \
      --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
      --output_dir $OUT/loo_${DS}_nafnet \
      >> logs/stage_loo_${DS}.log 2>&1 && note "OK    loo_$DS" || note "FAIL  loo_$DS"
  done
else
  note "SKIP  loo, no nafnet winner"
fi

# ---- 5. rebuild every figure from the new results ----
NAF_CFG2=$($PY -c "
import json
try: print(json.load(open('$OUT/winners.json'))['nafnet']['config'])
except Exception: print('')")
if [ -n "$NAF_CFG2" ]; then
  NAF_CK2=$OUT/sw_nafnet_${NAF_CFG2}_s0/best_model_cooperative.pth
  NAF_BG2="${NAF_CFG2%%_*}"
  stage fig_subjective $PY revision/fig_subjective.py --backbone nafnet \
        --checkpoint "$NAF_CK2" --bg_rule "$NAF_BG2" --roi 120 \
        --results_json $OUT/eval_nafnet.json --name fig_subjective_pku37
  stage fig_predicates $PY revision/fig_predicates.py
fi
stage fig_studies $PY revision/fig_studies.py
stage fig_architecture $PY revision/fig_architecture.py

note "OVERNIGHT_DONE"
