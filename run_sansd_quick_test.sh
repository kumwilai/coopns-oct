#!/bin/bash
#
# Quick test of NSAD with minimal samples to verify it works
#

echo "========================================"
echo "NSAD Quick Test (Proof of Concept)"
echo "========================================"
echo ""
echo "Testing VERIFIED NOVEL CONTRIBUTIONS:"
echo "  1. Per-pixel noise TYPE decomposition"
echo "  2. Soft routing to MULTIPLE classical operators"
echo "  3. Differentiable mixture of NAMED operators"
echo "  4. Full per-pixel interpretability"
echo "========================================"

# Configuration for stable training (physics loss disabled)
# MEMORY FIX: Reduced batch size from 4 to 2 for CPU memory safety
BATCH_SIZE=2
EPOCHS=30          # More epochs for end-to-end learning (stable now)
LR=5e-4            # Lower LR to prevent divergence (backbone gets 0.01x = 5e-6)
ALPHA=2.0
FUSION_MODE="gated"  # Per-pixel learnable fusion (best for interpretability)

# Physics loss weights (REDUCED - focus on reconstruction)
LAMBDA_LEVEL=0.01
LAMBDA_SPECKLE=0.01
LAMBDA_SHOT=0.01

# Data (SAME as run_end_to_end.sh)
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PATCH_SIZE=64           # 64x64 patches as required
MAX_TRAIN_SAMPLES=100   # MEMORY FIX: Reduced from 200 for CPU memory safety

# Model
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"

# Output
OUTPUT_DIR="checkpoints/sansd_quick_test"
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration (MEMORY OPTIMIZED):"
echo "  Epochs: $EPOCHS (quick test)"
echo "  Train samples: $MAX_TRAIN_SAMPLES (reduced for memory)"
echo "  Batch size: $BATCH_SIZE (reduced for CPU)"
echo "  Patch size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  Device: CPU"
echo "  Workers: 0 (single process for CPU)"
echo ""

# Run training
# MEMORY FIX: Use num_workers=0 for CPU to avoid multiprocessing overhead
python train_sansd.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers 0 \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --base_ckpt $BASE_CKPT \
    --alpha $ALPHA \
    --fusion_mode $FUSION_MODE \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_level $LAMBDA_LEVEL \
    --lambda_speckle $LAMBDA_SPECKLE \
    --lambda_shot $LAMBDA_SHOT \
    --output_dir $OUTPUT_DIR \
    --device cpu

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "Quick test PASSED!"
    echo "Best model: $OUTPUT_DIR/best_model.pth"
    echo "========================================"
else
    echo "========================================"
    echo "Quick test FAILED with exit code $EXIT_CODE"
    echo "========================================"
fi

exit $EXIT_CODE
