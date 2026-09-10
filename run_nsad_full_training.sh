#!/bin/bash
#
# NSAD Full Training Script for IEEE TMI
# Neuro-Symbolic Adaptive Denoising for OCT Images
#
# Usage: ./run_nsad_full_training.sh [gpu|cpu]
#

set -e  # Exit on error

echo "========================================"
echo "NSAD Full Training (IEEE TMI)"
echo "========================================"
echo ""
echo "VERIFIED NOVEL CONTRIBUTIONS:"
echo "  1. Per-pixel noise TYPE decomposition"
echo "  2. Soft routing to MULTIPLE classical operators"
echo "  3. Differentiable mixture of NAMED operators"
echo "  4. Full per-pixel interpretability"
echo "========================================"

# Device selection
DEVICE=${1:-cpu}
echo "Device: $DEVICE"

# Configuration
BATCH_SIZE=4
EPOCHS=100
LR=5e-4
ALPHA=2.0
FUSION_MODE="gated"

# Data
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PATCH_SIZE=64
MAX_TRAIN_SAMPLES=1000  # Use more samples for full training

# Model
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/nsad_full_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Train samples: $MAX_TRAIN_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Fusion mode: $FUSION_MODE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Save config
cat > $OUTPUT_DIR/config.txt << EOF
NSAD Full Training Configuration
================================
Date: $(date)
Device: $DEVICE
Epochs: $EPOCHS
Batch size: $BATCH_SIZE
Learning rate: $LR
Alpha: $ALPHA
Fusion mode: $FUSION_MODE
Train samples: $MAX_TRAIN_SAMPLES
Patch size: $PATCH_SIZE
Base checkpoint: $BASE_CKPT
EOF

# Run training
python train_sansd.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers 4 \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --base_ckpt $BASE_CKPT \
    --alpha $ALPHA \
    --fusion_mode $FUSION_MODE \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_level 0.0 \
    --lambda_speckle 0.0 \
    --lambda_shot 0.0 \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    2>&1 | tee $OUTPUT_DIR/training.log

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "Training COMPLETE!"
    echo "Best model: $OUTPUT_DIR/best_model.pth"
    echo "Log: $OUTPUT_DIR/training.log"
    echo "========================================"
else
    echo "========================================"
    echo "Training FAILED with exit code $EXIT_CODE"
    echo "Check log: $OUTPUT_DIR/training.log"
    echo "========================================"
fi

exit $EXIT_CODE
