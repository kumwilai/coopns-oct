#!/usr/bin/env bash
# Table 4. Transfer to Duke17 and Duke2013 with no adaptation of any kind.
set -eu
export LEGACY_SATURATING_ALLOCATION=0
for BB in nafnet dncnn kbnet swinir; do
  python3 code/validate_crossdataset.py \
    --checkpoint outputs/retrain_${BB}/best_model_cooperative.pth \
    --backbone weights/${BB}_backbone.pth --backbone_name $BB \
    --device cpu --datasets duke17,duke2013 \
    --output_json outputs/zeroshot_${BB}.json
done
