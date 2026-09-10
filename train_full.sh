#!/bin/bash
# =============================================================================
# Full Training Script for Interpretable Anatomy-Aware OCT Denoising
# =============================================================================

set -e

echo "=============================================="
echo "INTERPRETABLE ANATOMY-AWARE OCT DENOISER"
echo "Full Training Pipeline"
echo "=============================================="

# Configuration
EPOCHS=30
BATCH_SIZE=4
MAX_TRAIN=500
MAX_VAL=100
LR=1e-4
DEVICE="cpu"  # Change to "cuda" if GPU available
OUTPUT_DIR="checkpoints/soft_conditioned_full"

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "GPU detected, using CUDA"
        DEVICE="cuda"
        BATCH_SIZE=8
    fi
fi

echo ""
echo "Training Configuration:"
echo "  Epochs:      $EPOCHS"
echo "  Batch Size:  $BATCH_SIZE"
echo "  Max Train:   $MAX_TRAIN"
echo "  Max Val:     $MAX_VAL"
echo "  Learning Rate: $LR"
echo "  Device:      $DEVICE"
echo "  Output Dir:  $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Start training
echo "Starting training..."
python train_soft_conditioning.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_train $MAX_TRAIN \
    --max_val $MAX_VAL \
    --lr $LR \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/training_log.txt

echo ""
echo "Training complete!"
echo ""

# Run comprehensive evaluation
echo "Running comprehensive evaluation..."
python evaluate_anatomy.py \
    --checkpoint $OUTPUT_DIR/best.pth \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/evaluation_log.txt

echo ""
echo "=============================================="
echo "TRAINING AND EVALUATION COMPLETE"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
