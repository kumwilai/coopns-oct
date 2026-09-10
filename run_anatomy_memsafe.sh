#!/bin/bash
#
# Memory-Safe Anatomy-Aware NSAD Training
#
# This script runs the anatomy-aware per-pixel adaptive denoising algorithm
# with comprehensive memory management and monitoring.
#

echo "========================================"
echo "Memory-Safe Anatomy-Aware NSAD Training"
echo "========================================"
echo ""
echo "MEMORY OPTIMIZATIONS:"
echo "  - Batch size: 2 (reduced for CPU)"
echo "  - Patch size: 64x64"
echo "  - Num workers: 0 (no multiprocessing)"
echo "  - Pin memory: False"
echo "  - GC between epochs: Yes"
echo ""
echo "MONITORING:"
echo "  - Memory usage tracking"
echo "  - Anatomy ROI quality (layer-specific PSNR/SSIM)"
echo "  - Edge preservation metrics"
echo "  - Expert usage (interpretability)"
echo "  - Layer detection (interpretability)"
echo "========================================"

# Configuration - MEMORY SAFE
BATCH_SIZE=2           # Small batch for CPU memory safety
PATCH_SIZE=64          # 64x64 patches as required
NUM_WORKERS=0          # No multiprocessing for CPU
MAX_TRAIN=50           # Small training set for quick test
MAX_VAL=20             # Small validation set
EPOCHS=5               # Quick training
LR=5e-4
DEVICE=cpu

# Data
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"

# Model
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
FUSION_MODE="anatomy"  # Use anatomy-aware fusion

# Output
OUTPUT_DIR="checkpoints/anatomy_memsafe"
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Train samples: $MAX_TRAIN"
echo "  Val samples: $MAX_VAL"
echo "  Batch size: $BATCH_SIZE"
echo "  Patch size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  Device: $DEVICE"
echo "  Workers: $NUM_WORKERS"
echo "  Fusion mode: $FUSION_MODE"
echo ""

# Run training
python train_anatomy_memsafe.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_train_samples $MAX_TRAIN \
    --max_val_samples $MAX_VAL \
    --base_ckpt $BASE_CKPT \
    --alpha 2.0 \
    --fusion_mode $FUSION_MODE \
    --use_anatomy_loss \
    --epochs $EPOCHS \
    --lr $LR \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "Training COMPLETED SUCCESSFULLY!"
    echo "Best model: $OUTPUT_DIR/best_model.pth"
    echo "========================================"
else
    echo "========================================"
    echo "Training FAILED with exit code $EXIT_CODE"
    echo "========================================"
fi

exit $EXIT_CODE
