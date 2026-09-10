#!/bin/bash
# =============================================================================
# TMI Step 1: Training
# =============================================================================
set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmi_config.sh"

print_config

echo ""
echo "=============================================="
echo "STEP 1: TRAINING MAIN MODEL"
echo "=============================================="
echo ""
echo "Parameters:"
echo "  Epochs:       $TRAIN_EPOCHS"
echo "  Batch Size:   $TRAIN_BATCH_SIZE"
echo "  Max Train:    $TRAIN_MAX_SAMPLES"
echo "  Max Val:      $TRAIN_VAL_SAMPLES"
echo "  Learning Rate: $TRAIN_LR"
echo ""

mkdir -p "$CHECKPOINT_DIR"

python train_soft_conditioning.py \
    --epochs $TRAIN_EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --max_train $TRAIN_MAX_SAMPLES \
    --max_val $TRAIN_VAL_SAMPLES \
    --lr $TRAIN_LR \
    --device $DEVICE \
    --output_dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training_log.txt"

echo ""
echo "=============================================="
echo "STEP 1 COMPLETE"
echo "=============================================="
echo "Checkpoint saved to: $CHECKPOINT_PATH"
echo "Log saved to: $OUTPUT_DIR/training_log.txt"
echo ""
echo "Next step: bash tmi_step2_evaluate.sh"
echo "=============================================="
