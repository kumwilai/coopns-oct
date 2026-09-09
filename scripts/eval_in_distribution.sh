#!/usr/bin/env bash
# Table 3. In distribution results on the PKU37 test set.
set -eu
export LEGACY_SATURATING_ALLOCATION=0
for BB in nafnet dncnn kbnet swinir; do
  python3 code/revision/ablation_runner.py --backbone_name $BB \
    --checkpoint outputs/retrain_${BB}/best_model_cooperative.pth \
    --pretrained_backbone weights/${BB}_backbone.pth \
    --test_jsonl data/pku37_real_test.jsonl \
    --intervention none --output_json outputs/eval_${BB}.json
done
