#!/bin/bash
# =============================================================================
# TMI Ablation Study: Multi-Task WITHOUT Layer-Specific Gates
# =============================================================================
#
# Purpose: Prove the contribution of layer-specific noise modeling
#
# Comparison:
#   Full method:  5 layer-specific gates (RNFL, INL, ONL, IS_OS, RPE)
#   Ablation:     1 global gate (same for all layers)
#
# Expected result: Full method should outperform ablation, proving
#                  that layer-specific adaptation is beneficial.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "TMI ABLATION: WITHOUT Layer-Specific Gates"
echo "=============================================="
echo ""
echo "This trains a multi-task model with a GLOBAL noise gate"
echo "instead of 5 layer-specific gates."
echo ""
echo "Purpose: Prove layer-specific modeling improves results."
echo "=============================================="

# Configuration
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi

# Check for required models
BACKBONE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
SEGMENTER_CKPT="models/layer_segmenter/best.pth"
SEG_DATA_DIR="seg_data"

if [ ! -f "$BACKBONE_CKPT" ]; then
    echo "ERROR: Backbone checkpoint not found: $BACKBONE_CKPT"
    exit 1
fi

if [ ! -f "$SEGMENTER_CKPT" ]; then
    echo "ERROR: Segmenter checkpoint not found: $SEGMENTER_CKPT"
    exit 1
fi

# Output directory
OUTPUT_BASE="tmi_ablation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/no_layer_gates_${TIMESTAMP}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo ""
echo "Configuration:"
echo "  Device:     $DEVICE"
echo "  Backbone:   $BACKBONE_CKPT"
echo "  Segmenter:  $SEGMENTER_CKPT"
echo "  Output:     $CHECKPOINT_DIR/"
echo ""
echo "Training parameters (same as full method):"
echo "  Epochs:         30"
echo "  Batch size:     4"
echo "  Max train:      2000"
echo "  Max val:        400"
echo "  Lambda_seg:     1.0"
echo "  Lambda_boundary: 0.1"
echo ""
echo "KEY DIFFERENCE: Using GLOBAL noise gate (not layer-specific)"
echo ""

python train_multitask_ablation.py \
    --train_jsonl "$SEG_DATA_DIR/seg_train.jsonl" \
    --val_jsonl "$SEG_DATA_DIR/seg_val.jsonl" \
    --patch_size 64 \
    --batch_size 4 \
    --max_train 2000 \
    --max_val 400 \
    --epochs 30 \
    --lr 1e-4 \
    --lambda_seg 1.0 \
    --lambda_boundary 0.1 \
    --backbone_ckpt "$BACKBONE_CKPT" \
    --segmenter_ckpt "$SEGMENTER_CKPT" \
    --device $DEVICE \
    --output_dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/ablation_training_log.txt"

echo ""
echo "=============================================="
echo "ABLATION TRAINING COMPLETE"
echo "=============================================="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Training log: $OUTPUT_DIR/ablation_training_log.txt"
echo ""
echo "Next: Compare results with full method"
echo ""
echo "Expected comparison table:"
echo "┌─────────────────────────────────┬────────┬────────┐"
echo "│ Method                          │ PSNR   │ Dice   │"
echo "├─────────────────────────────────┼────────┼────────┤"
echo "│ Multi-task + Global Gate (this) │ ??     │ ??     │"
echo "│ Multi-task + Layer Gates (full) │ ~31.9  │ ~0.93  │"
echo "└─────────────────────────────────┴────────┴────────┘"
echo "=============================================="
