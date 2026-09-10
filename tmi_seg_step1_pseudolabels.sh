#!/bin/bash
# =============================================================================
# TMI Segmentation Step 1: Generate Pseudo-Segmentation Labels
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "TMI SEGMENTATION: STEP 1"
echo "Generate Pseudo-Segmentation Labels"
echo "=============================================="
echo ""
echo "NOTE: Using gradient-based pseudo-labels."
echo "For best results, replace with Duke DME real labels."
echo "=============================================="
echo ""

# Configuration
SEG_DATA_DIR="seg_data"
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"

echo "Input datasets:"
echo "  Train: $TRAIN_JSONL"
echo "  Val:   $VAL_JSONL"
echo ""
echo "Output: $SEG_DATA_DIR/"
echo ""

python generate_pseudo_segmentation.py \
    --train_jsonl "$TRAIN_JSONL" \
    --val_jsonl "$VAL_JSONL" \
    --output_dir "$SEG_DATA_DIR"

echo ""
echo "=============================================="
echo "STEP 1 COMPLETE"
echo "=============================================="
echo "Generated pseudo-labels in: $SEG_DATA_DIR/"
echo ""
echo "Output files:"
echo "  - $SEG_DATA_DIR/seg_train.jsonl"
echo "  - $SEG_DATA_DIR/seg_val.jsonl"
echo "  - $SEG_DATA_DIR/segmentation/train/*.npy"
echo "  - $SEG_DATA_DIR/segmentation/val/*.npy"
echo ""
echo "Next step: bash tmi_seg_step2_segmenter.sh"
echo "=============================================="
