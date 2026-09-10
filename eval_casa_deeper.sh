#!/bin/bash
# ============================================================================
# Evaluate DEEPER CASA and compare with SwinIR
# ============================================================================

CHECKPOINT="checkpoints/casa_deeper/finetuned_ema.pth"
VAL_PAIRS="val_pairs_universal.txt"

echo "========================================================================"
echo "EVALUATING DEEPER CASA (6-block U-Net)"
echo "========================================================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please train the model first by running:"
    echo "  ./train_casa_deeper.sh"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Validation set: $VAL_PAIRS"
echo ""
echo "Running evaluation..."
echo "========================================================================"

python eval_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --val_pairs "$VAL_PAIRS" \
    --adapter casa \
    --image_size 64

echo ""
echo "========================================================================"
echo "COMPARISON WITH ALL METHODS"
echo "========================================================================"
echo ""

python compare_available_methods.py

echo ""
echo "========================================================================"
echo "Performance Target:"
echo "  - SwinIRLite (6-layer CNN): 28.84 dB"
echo "  - Old CASA (4 blocks):      28.11 dB"
echo "  - Goal: New CASA >= 28.8 dB (beat or match SwinIR)"
echo "========================================================================"
