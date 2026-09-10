#!/usr/bin/env bash
# Sequential queue for the revision runs. One job at a time, two CPU cores.
set -u
PY=/home/kumwilai/osmnx-env/bin/python
cd /home/kumwilai/OCT
export OMP_NUM_THREADS=2
R=revision/ablation_runner.py
CK=checkpointpaper/nafnet_pku37_cooperative.pth
BB=checkpointpaper/nafnet_backbone.pth
FULL=pku37_oct_dataset/pku37_real_test.jsonl
SUB=revision/pku37_subset40.jsonl

run () {  # run <tag> <intervention> <jsonl>
  local out=outputs/revision/$1.json
  if [ -f "$out" ]; then echo "skip $1"; return; fi
  echo "### $1  $(date -Iseconds)"
  $PY $R --backbone_name nafnet --checkpoint $CK --pretrained_backbone $BB \
        --test_jsonl "$3" --intervention "$2" --output_json "$out" \
        2>&1 | grep -vE "^Evaluating|it/s\]|s/img\]" | tail -14
}

# wait for the zero shot job to finish so the cores are not shared
while pgrep -f run_zeroshot_transfer.sh > /dev/null; do sleep 60; done
echo "zero shot finished, starting queue $(date -Iseconds)"

# 1. predicate leave one out on the full test set
run lopo_none      none    $FULL
for P in P1 P2 P3 P4 P5 P6; do run lopo_drop_$P drop_$P $FULL; done

# 2. component ablations on the full test set, recomputed for the revision
for A in no_negotiator no_edge no_uncertainty no_bg_smooth verifier_soft; do
  run comp_$A $A $FULL
done

# 3. fuzzy rule sensitivity on the fixed forty image subset
run fuzz_base_ref  none    $SUB
for V in 0.5 1.0 2.0 3.0; do run fuzz_base_$V base=$V $SUB; done
for V in 2.0 3.0 5.0 6.0; do run fuzz_usecorr_$V rule=use_corrector:$V $SUB; done
for V in 1.0 2.0 4.0 5.0; do run fuzz_boost_$V   rule=boost_failing:$V $SUB; done
for V in 0.5 1.0 2.5; do    run fuzz_balance_$V  rule=balance:$V $SUB; done
for T in product minimum; do run fuzz_tnorm_$T tnorm=$T $SUB; done

echo "QUEUE_DONE $(date -Iseconds)"
