#!/usr/bin/env bash
# Everything that runs after the retraining finishes. Sequential, two cores.
set -u
PY=/home/kumwilai/osmnx-env/bin/python
cd /home/kumwilai/OCT
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LEGACY_SATURATING_ALLOCATION=0
R=revision/ablation_runner.py
FULL=pku37_oct_dataset/pku37_real_test.jsonl
SUB=revision/pku37_subset40.jsonl
OUT=outputs/revision

ckpt () { echo "$OUT/retrain_$1/best_model_cooperative.pth"; }
bb   () { echo "checkpointpaper/$1_backbone.pth"; }

run () {  # run <tag> <backbone> <intervention> <jsonl>
  local out=$OUT/$1.json
  [ -f "$out" ] && { echo "skip $1"; return; }
  echo "### $1  $(date -Iseconds)"
  $PY $R --backbone_name "$2" --checkpoint "$(ckpt $2)" --pretrained_backbone "$(bb $2)" \
        --test_jsonl "$4" --intervention "$3" --output_json "$out" 2>&1 \
        | grep -vE "Evaluating|it/s\]|s/img\]" | tail -13
}

while pgrep -f run_retrain_all.sh > /dev/null; do sleep 120; done
echo "retraining finished, starting evaluations $(date -Iseconds)"

# 1. in distribution, all four backbones, full test set
for B in nafnet dncnn kbnet swinir; do
  [ -f "$(ckpt $B)" ] || { echo "missing checkpoint for $B, skipping"; continue; }
  run eval_$B $B none $FULL
done

# 2. transfer with no adaptation, all four backbones
for B in nafnet dncnn kbnet swinir; do
  [ -f "$(ckpt $B)" ] || continue
  o=$OUT/zeroshot_$B.json
  [ -f "$o" ] && { echo "skip zeroshot $B"; continue; }
  echo "### zeroshot $B  $(date -Iseconds)"
  $PY validate_crossdataset.py --checkpoint "$(ckpt $B)" --backbone "$(bb $B)" \
      --backbone_name $B --device cpu --datasets duke17,duke2013 --output_json "$o" 2>&1 | tail -25
done

# 3. leave one clinical property out, NAFNet, full test set
run lopo_none nafnet none $FULL
for P in P1 P2 P3 P4 P5 P6; do run lopo_drop_$P nafnet drop_$P $FULL; done

# 4. component ablations, NAFNet, full test set
for A in no_negotiator no_edge no_uncertainty no_bg_smooth; do
  run comp_$A nafnet $A $FULL
done

# 5. rule constant sensitivity, NAFNet, forty image subset
run fuzz_ref nafnet none $SUB
for V in -1.0 -0.5 0.5 1.0; do run fuzz_base_$V nafnet base=$V $SUB; done
for V in 2.0 3.0 5.0 6.0; do run fuzz_usecorr_$V nafnet rule=use_corrector:$V $SUB; done
for V in 1.0 2.0 4.0 5.0; do run fuzz_boost_$V nafnet rule=boost_failing:$V $SUB; done
for T in product minimum; do run fuzz_tnorm_$T nafnet tnorm=$T $SUB; done

echo "EVALS_DONE $(date -Iseconds)"
