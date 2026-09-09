#!/usr/bin/env bash
# The numerical check of both theorems, the calibration test and the safety study.
set -eu
export LEGACY_SATURATING_ALLOCATION=0
python3 code/revision/diagnostics.py --backbone_name nafnet \
  --checkpoint outputs/retrain_nafnet/best_model_cooperative.pth \
  --pretrained_backbone weights/nafnet_backbone.pth \
  --test_jsonl data/pku37_real_test.jsonl \
  --output_json outputs/diagnostics_nafnet.json
