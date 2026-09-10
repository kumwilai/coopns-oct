#!/bin/bash
#
# End-to-End Joint Training: Analyzer + Denoiser
#
# KEY INNOVATION:
# - Train analyzer JOINTLY with denoiser (no separate pre-training)
# - Gradients flow through entire pipeline
# - Noise estimation optimized for denoising quality (not just classification)
# - More robust to analyzer failures
#
# CONTRIBUTION OVER BASELINE:
# - Removes dependency on pre-trained analyzer
# - Shows noise estimation can be learned implicitly from denoising task
# - Better adaptation through joint optimization
#

# Configuration
BATCH_SIZE=4
EPOCHS=5  # Reduced for faster CPU training (proof-of-concept)
LR=1e-4
USAGE_LOSS=0.5           # Force model to use conditioning
CLASSIFY_LOSS=0.1        # Auxiliary guidance (optional)
ALPHA=2.0                # Modulation strength
BASE_DELTA_MARGIN=0.01   # Minimum adaptation strength
MAX_TRAIN_SAMPLES=200    # Limit for quick proof-of-concept

# Data
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PATCH_SIZE=64

# Model initialization
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
# ANALYZER_INIT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"  # Optional
ANALYZER_INIT=""  # Train from scratch for clean experiment

# Output
OUTPUT_DIR="checkpoints/end_to_end"
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
echo "========================================"
