#!/bin/bash
# =============================================================================
# TMI Segmentation Step 2: Train Layer Segmentation Model
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "TMI SEGMENTATION: STEP 2"
echo "Train Layer Segmentation Model"
echo "=============================================="

# Configuration
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi

SEG_DATA_DIR="seg_data"
OUTPUT_DIR="models/layer_segmenter"

# Check if pseudo-labels exist
if [ ! -f "$SEG_DATA_DIR/seg_train.jsonl" ]; then
    echo "ERROR: Pseudo-labels not found!"
    echo "Please run: bash tmi_seg_step1_pseudolabels.sh"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Device:       $DEVICE"
echo "  Train data:   $SEG_DATA_DIR/seg_train.jsonl"
echo "  Val data:     $SEG_DATA_DIR/seg_val.jsonl"
echo "  Output:       $OUTPUT_DIR/"
echo ""
echo "Training parameters:"
echo "  Epochs:       50"
echo "  Batch size:   8"
echo "  Max train:    1000"
echo "  Max val:      200"
echo ""

mkdir -p "$OUTPUT_DIR"

python train_layer_segmentation.py \
    --train_jsonl "$SEG_DATA_DIR/seg_train.jsonl" \
    --val_jsonl "$SEG_DATA_DIR/seg_val.jsonl" \
    --patch_size 128 \
    --batch_size 8 \
    --max_train 1000 \
    --max_val 200 \
    --epochs 50 \
    --lr 1e-3 \
    --device $DEVICE \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training_log.txt"

echo ""
echo "=============================================="
echo "STEP 2 COMPLETE"
echo "=============================================="
echo "Model saved to: $OUTPUT_DIR/best.pth"
echo "Training log:   $OUTPUT_DIR/training_log.txt"
echo ""
echo "Next step: bash tmi_seg_step3_multitask.sh"
echo "=============================================="
