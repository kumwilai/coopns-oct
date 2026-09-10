#!/bin/bash
# =============================================================================
# Test Segmentation Pipeline - Quick Verification
# =============================================================================
set -e

echo "=============================================="
echo "TESTING SEGMENTATION PIPELINE"
echo "=============================================="

# Use minimal samples for quick test
export TMI_OUTPUT_DIR="seg_test_$(date +%Y%m%d_%H%M%S)"
DEVICE="cpu"

echo "Test output: $TMI_OUTPUT_DIR"
echo ""

# Step 1: Generate pseudo-labels
echo ">>> Step 1: Generating pseudo-segmentation labels (20 samples)"
python generate_pseudo_segmentation.py \
    --train_jsonl weights_duke_analysis_maps_train.jsonl \
    --val_jsonl weights_duke_analysis_maps_val.jsonl \
    --output_dir seg_data_test

echo ""
echo ">>> Step 2: Training layer segmenter (3 epochs, 50 samples)"
python train_layer_segmentation.py \
    --train_jsonl seg_data_test/seg_train.jsonl \
    --val_jsonl seg_data_test/seg_val.jsonl \
    --epochs 3 \
    --max_train 50 \
    --max_val 20 \
    --batch_size 4 \
    --device $DEVICE \
    --output_dir models/layer_segmenter_test

echo ""
echo ">>> Step 3: Multi-task training (2 epochs, 30 samples)"
python train_multitask.py \
    --train_jsonl seg_data_test/seg_train.jsonl \
    --val_jsonl seg_data_test/seg_val.jsonl \
    --epochs 2 \
    --max_train 30 \
    --max_val 15 \
    --batch_size 2 \
    --device $DEVICE \
    --backbone_ckpt outputs/nafnet_analysis_maps_w64/nafnet_best.pth \
    --segmenter_ckpt models/layer_segmenter_test/best.pth \
    --output_dir $TMI_OUTPUT_DIR/checkpoints

echo ""
echo "=============================================="
echo "SEGMENTATION PIPELINE TEST COMPLETE!"
echo "=============================================="
echo ""
echo "Generated:"
echo "  1. Pseudo-labels: seg_data_test/"
echo "  2. Segmenter:     models/layer_segmenter_test/best.pth"
echo "  3. Multi-task:    $TMI_OUTPUT_DIR/checkpoints/best_psnr.pth"
echo ""
echo "To clean up test files:"
echo "  rm -rf seg_data_test models/layer_segmenter_test $TMI_OUTPUT_DIR"
echo ""
echo "For FULL pipeline, run:"
echo "  bash tmi_seg_step1_pseudolabels.sh"
echo "  bash tmi_seg_step2_segmenter.sh"
echo "  bash tmi_seg_step3_multitask.sh"
echo "=============================================="
