#!/bin/bash
# =============================================================================
# TMI: Retrain Multi-Task Model (with segmentation fix)
# =============================================================================
# This script retrains the multi-task denoising + segmentation model
# with the fixed class weights and random crop sampling.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "RETRAINING MULTI-TASK MODEL"
echo "=============================================="
echo ""
echo "Fixes applied:"
echo "  - ensure_all_layers=False (truly random crops)"
echo "  - Weighted cross-entropy (balanced class learning)"
echo "  - All 5 retinal layers will be learned"
echo "=============================================="

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"

# Training parameters
EPOCHS=30
BATCH_SIZE=4
MAX_TRAIN=2000
MAX_VAL=400
LR=1e-4

echo ""
echo "Parameters:"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Max train:  $MAX_TRAIN"
echo "  Max val:    $MAX_VAL"
echo "  LR:         $LR"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="tmi_multitask_fixed/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output: $OUTPUT_DIR"
echo ""

# Run training
python train_multitask.py \
    --train_jsonl seg_data/seg_train.jsonl \
    --val_jsonl seg_data/seg_val.jsonl \
    --backbone_ckpt outputs/nafnet_analysis_maps_w64/nafnet_best.pth \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_train $MAX_TRAIN \
    --max_val $MAX_VAL \
    --lr $LR \
    --device $DEVICE \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training_log.txt"

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Checkpoint: $OUTPUT_DIR/checkpoints/best_psnr.pth"
echo "Log: $OUTPUT_DIR/training_log.txt"
echo ""
echo "Next: Run per-layer evaluation to verify fix:"
echo "  bash tmi_step3_4_per_layer_analysis.sh"
echo "=============================================="
