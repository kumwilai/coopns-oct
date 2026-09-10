#!/bin/bash
#
# End-to-End Joint Training: MAXIMUM ACCURACY MODE
#
# AGGRESSIVE IMPROVEMENTS FOR TOP-1 ACCURACY:
# 1. Very high classification loss weight (2.0)
# 2. Pre-trained analyzer initialization
# 3. Full training set (no sample limit)
# 4. Extended training (20 epochs)
# 5. Lower learning rate for fine-tuning
#

echo "========================================"
echo "End-to-End Training: MAX ACCURACY MODE"
echo "Target: Top-1 > 85%"
echo "========================================"

# Configuration - MAXIMUM ACCURACY
BATCH_SIZE=4
EPOCHS=20                # Extended training
LR=5e-5                  # Lower LR for fine-tuning pre-trained analyzer
USAGE_LOSS=0.3           # Reduced to give more weight to classification
CLASSIFY_LOSS=2.0        # 🎯 20x HIGHER than original for strong classification signal
ALPHA=2.0                # Modulation strength
BASE_DELTA_MARGIN=0.01   # Minimum adaptation strength
MAX_TRAIN_SAMPLES=""     # Use ALL training samples (no limit)

# Data
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PATCH_SIZE=64

# Model initialization
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"

# 🎯 PRE-TRAINED ANALYZER (critical for accuracy)
ANALYZER_INIT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"

# Output
OUTPUT_DIR="checkpoints/end_to_end_max_accuracy"
mkdir -p $OUTPUT_DIR

echo "Settings:"
echo "  - Classification Loss: ${CLASSIFY_LOSS} (20x higher)"
echo "  - Usage Loss: ${USAGE_LOSS} (reduced)"
echo "  - Epochs: ${EPOCHS}"
echo "  - Training samples: ALL (no limit)"
echo "  - Using pre-trained analyzer"
echo ""

# Build command
CMD="python train_end_to_end.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --base_ckpt $BASE_CKPT \
    --analyzer_init $ANALYZER_INIT \
    --alpha $ALPHA \
    --epochs $EPOCHS \
    --lr $LR \
    --usage_loss_weight $USAGE_LOSS \
    --classify_loss_weight $CLASSIFY_LOSS \
    --base_delta_margin $BASE_DELTA_MARGIN \
    --output_dir $OUTPUT_DIR \
    --device cpu"

# Add max_train_samples only if set
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    CMD="$CMD --max_train_samples $MAX_TRAIN_SAMPLES"
fi

# Run training
$CMD

echo ""
echo "========================================"
echo "Training complete!"
echo "Best model: $OUTPUT_DIR/best_model.pth"
echo "Check Top-1 accuracy improvement!"
echo "========================================"
