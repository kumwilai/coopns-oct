#!/usr/bin/env bash
# Table 6 and Figure 3. Property leave one out, rule sensitivity, component
# ablation and the matched complexity comparison.
set -eu
export LEGACY_SATURATING_ALLOCATION=0
CK=outputs/retrain_nafnet/best_model_cooperative.pth
BB=weights/nafnet_backbone.pth
FULL=data/pku37_real_test.jsonl
SUB=data/pku37_subset40.jsonl
run () { python3 code/revision/ablation_runner.py --backbone_name nafnet \
         --checkpoint $CK --pretrained_backbone $BB --test_jsonl "$3" \
         --intervention "$2" --output_json outputs/$1.json; }
run lopo_none none $FULL
for P in P1 P2 P3 P4 P5 P6; do run lopo_drop_$P drop_$P $FULL; done
for A in no_negotiator no_edge no_uncertainty no_bg_smooth; do run comp_$A $A $FULL; done
run fuzz_ref none $SUB
for V in -1.0 -0.5 0.5 1.0; do run fuzz_base_$V base=$V $SUB; done
for V in 2.0 3.0 5.0 6.0; do run fuzz_usecorr_$V rule=use_corrector:$V $SUB; done
for V in 1.0 2.0 4.0 5.0; do run fuzz_boost_$V rule=boost_failing:$V $SUB; done
for T in product minimum; do run fuzz_tnorm_$T tnorm=$T $SUB; done
python3 code/revision/classical_baselines.py --mode tune --pretrained_backbone $BB
