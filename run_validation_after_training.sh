#!/bin/bash
# Wait for training to finish, then validate on PKU37, Duke17, Duke2013

echo "Waiting for training to finish..."
while ps aux | grep "train_v8_cooperative.py" | grep -v grep | grep -v "run_validation" > /dev/null 2>&1; do
    sleep 60
done
echo "Training finished at $(date)"

CHECKPOINT="outputs/v8_predicate_loss/best_model_cooperative.pth"
BACKBONE="outputs/nafnet_pku37_w40/best_model.pth"

echo ""
echo "============================================================"
echo "VALIDATION: PKU37 + Duke17 + Duke2013"
echo "============================================================"

python validate_crossdataset.py \
    --checkpoint "$CHECKPOINT" \
    --backbone "$BACKBONE" \
    --backbone_name nafnet \
    --device cpu \
    --output_json outputs/v8_predicate_loss/validation_results.json

echo ""
echo "Done at $(date)"
echo "Results saved to outputs/v8_predicate_loss/validation_results.json"
