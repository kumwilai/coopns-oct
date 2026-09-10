#!/bin/bash
# =============================================================================
# Dice Fine-tuning Script
# =============================================================================
# Run this AFTER tmi_retrain_clinical.sh completes
#
# Purpose:
#   - Improve Dice score for segmentation claims in TMI paper
#   - Freezes backbone (denoising) to protect PSNR gains
#   - Trains only segmentation head
#
# Usage:
#   bash finetune_dice.sh [checkpoint_path]
#
# Example:
#   bash finetune_dice.sh tmi_cuap_oct/full_model/20260115_143743/best_psnr.pth
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find checkpoint
if [ -n "$1" ]; then
    CHECKPOINT="$1"
else
    # Auto-find latest best_psnr.pth
    CHECKPOINT=$(find tmi_cuap_oct -name "best_psnr.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found. Please provide path to best_psnr.pth"
        echo "Usage: bash finetune_dice.sh path/to/best_psnr.pth"
        exit 1
    fi
fi

echo "=============================================="
echo "DICE FINE-TUNING FOR CUAP-OCT"
echo "=============================================="
echo ""
echo "Checkpoint: $CHECKPOINT"
echo ""

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="finetune_dice/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Parameters
EPOCHS=15
BATCH_SIZE=8
LR=5e-5
LAMBDA_DICE=0.7
MAX_TRAIN=2000
MAX_VAL=400

echo "Parameters:"
echo "  Epochs:      $EPOCHS"
echo "  Batch size:  $BATCH_SIZE (larger since backbone frozen)"
echo "  LR:          $LR (low for fine-tuning)"
echo "  Lambda Dice: $LAMBDA_DICE (high for Dice focus)"
echo ""
echo "Strategy:"
echo "  - Backbone: FROZEN (protect PSNR)"
echo "  - Segmentation head: TRAINABLE"
echo "  - Thin layer boost: INL 1.5x, ONL 1.5x, IS_OS 1.3x"
echo "=============================================="
echo ""

python finetune_dice.py \
    --checkpoint "$CHECKPOINT" \
    --train_jsonl seg_data/seg_train.jsonl \
    --val_jsonl seg_data/seg_val.jsonl \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lambda_dice $LAMBDA_DICE \
    --max_train $MAX_TRAIN \
    --max_val $MAX_VAL \
    --device $DEVICE \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/finetune_log.txt"

echo ""
echo "=============================================="
echo "FINE-TUNING COMPLETE"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/best_dice_finetuned.pth"
echo "  - $OUTPUT_DIR/finetune_log.txt"
echo ""
echo "For TMI paper, report:"
echo "  1. Denoising results from: $CHECKPOINT"
echo "  2. Segmentation results from: $OUTPUT_DIR/best_dice_finetuned.pth"
echo "=============================================="
