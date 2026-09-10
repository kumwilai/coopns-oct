#!/bin/bash
#
# Full-Scale End-to-End Training
#
# This is the complete training run (not proof-of-concept)
# Uses full dataset and proper epoch count
#

# Configuration
BATCH_SIZE=4
EPOCHS=20                # Full training epochs
LR=1e-4
USAGE_LOSS=0.5           # Force model to use conditioning
CLASSIFY_LOSS=0.2        # Increased from 0.1 for better guidance
ALPHA=2.0                # Modulation strength
BASE_DELTA_MARGIN=0.01   # Minimum adaptation strength
MAX_TRAIN_SAMPLES=""     # Empty = use all samples (2000)

# Data
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PATCH_SIZE=64

# Model initialization
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
ANALYZER_INIT=""  # Train from scratch for clean experiment

# Output
OUTPUT_DIR="checkpoints/end_to_end_full"
mkdir -p $OUTPUT_DIR

# Log file
LOG_FILE="$OUTPUT_DIR/training.log"

echo "========================================"
echo "FULL-SCALE END-TO-END TRAINING"
echo "========================================"
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Usage loss: $USAGE_LOSS"
echo "  Classification loss: $CLASSIFY_LOSS"
echo "  Training samples: ALL (no limit)"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $LOG_FILE"
echo "========================================"
echo ""
echo "Starting training..."
echo "This will take several hours on CPU."
echo ""

# Run training
python train_end_to_end.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --base_ckpt $BASE_CKPT \
    --alpha $ALPHA \
    --epochs $EPOCHS \
    --lr $LR \
    --usage_loss_weight $USAGE_LOSS \
    --classify_loss_weight $CLASSIFY_LOSS \
    --base_delta_margin $BASE_DELTA_MARGIN \
    --output_dir $OUTPUT_DIR \
    --device cpu 2>&1 | tee $LOG_FILE

echo ""
echo "========================================"
echo "Training complete!"
echo "Best model: $OUTPUT_DIR/best_model.pth"
echo "Full log: $LOG_FILE"
echo "========================================"
echo ""
echo "To evaluate results, run:"
echo "python compare_approaches.py \\"
echo "    --end_to_end_ckpt $OUTPUT_DIR/best_model.pth \\"
echo "    --num_samples 50"
