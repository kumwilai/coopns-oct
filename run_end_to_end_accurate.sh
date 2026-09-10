#!/bin/bash
#
# End-to-End Joint Training: OPTIMIZED FOR ACCURACY
#
# IMPROVEMENTS FOR TOP-1 ACCURACY:
# 1. Higher classification loss weight (1.0 → 10x stronger signal)
# 2. Pre-trained analyzer initialization (warm start)
# 3. More training samples (500 instead of 200)
# 4. More epochs (15 instead of 5)
# 5. Balanced loss weights for both denoising and classification
#

echo "========================================"
echo "End-to-End Training: Accuracy-Focused"
echo "Target: Top-1 > 80%"
echo "========================================"

# Configuration - OPTIMIZED FOR ACCURACY
BATCH_SIZE=4
EPOCHS=15                # More epochs for better convergence
LR=1e-4
USAGE_LOSS=0.5           # Force model to use conditioning
CLASSIFY_LOSS=1.0        # 🎯 INCREASED 10x for better classification
ALPHA=2.0                # Modulation strength
BASE_DELTA_MARGIN=0.01   # Minimum adaptation strength
MAX_TRAIN_SAMPLES=500    # More training data for better accuracy

# Data
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PATCH_SIZE=64

# Model initialization
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"

# 🎯 USE PRE-TRAINED ANALYZER for warm start (better accuracy)
ANALYZER_INIT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"

# Output
OUTPUT_DIR="checkpoints/end_to_end_accurate"
mkdir -p $OUTPUT_DIR

# Run training
python train_end_to_end.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --base_ckpt $BASE_CKPT \
    --analyzer_init $ANALYZER_INIT \
    --alpha $ALPHA \
    --epochs $EPOCHS \
    --lr $LR \
    --usage_loss_weight $USAGE_LOSS \
    --classify_loss_weight $CLASSIFY_LOSS \
    --base_delta_margin $BASE_DELTA_MARGIN \
    --output_dir $OUTPUT_DIR \
    --device cpu

echo ""
echo "========================================"
echo "Training complete!"
echo "Best model: $OUTPUT_DIR/best_model.pth"
echo "Check Top-1 accuracy improvement!"
echo "========================================"
