#!/bin/bash
# =============================================================================
# TMI Segmentation Step 3: Multi-Task Denoising + Segmentation
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "TMI SEGMENTATION: STEP 3"
echo "Multi-Task: Denoising + Layer Segmentation"
echo "=============================================="
echo ""
echo "KEY CONTRIBUTION FOR TMI:"
echo "  - Joint optimization of both tasks"
echo "  - Real anatomical grounding"
echo "  - Bidirectional improvement"
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
    echo "Please run: bash tmi_seg_step2_segmenter.sh"
    exit 1
fi

if [ ! -f "$SEG_DATA_DIR/seg_train.jsonl" ]; then
    echo "ERROR: Segmentation data not found!"
    echo "Please run: bash tmi_seg_step1_pseudolabels.sh"
    exit 1
fi

# Output directory
if [ -z "$TMI_OUTPUT_DIR" ]; then
    OUTPUT_BASE="tmi_multitask"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
else
    OUTPUT_DIR="$TMI_OUTPUT_DIR"
fi

CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo ""
echo "Configuration:"
echo "  Device:         $DEVICE"
echo "  Backbone:       $BACKBONE_CKPT"
echo "  Segmenter:      $SEGMENTER_CKPT"
echo "  Output:         $CHECKPOINT_DIR/"
echo ""
echo "Training parameters:"
echo "  Epochs:         30"
echo "  Batch size:     4"
echo "  Max train:      2000"
echo "  Max val:        400"
echo "  Lambda_seg:       1.0   (segmentation loss weight)"
echo "  Lambda_boundary:  0.1  (boundary preservation loss)"
echo "  Lambda_noise_corr: 0.1 (gate-noise correlation loss)"
echo "  Lambda_diversity: 0.05 (NEW: layer gate diversity - KEY TMI)"
echo "  Ensure all layers: True (crops guaranteed to include all 5 layers)"
echo ""
echo "KEY TMI CONTRIBUTION: Layer-Specific Noise Modeling"
echo "  - Each retinal layer gets its own noise-adaptive gate"
echo "  - Different layers have different noise characteristics"
echo "  - Model learns layer-appropriate denoising strategies"
echo ""

python train_multitask.py \
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
    --lambda_noise_corr 0.1 \
    --lambda_diversity 0.05 \
    --backbone_ckpt "$BACKBONE_CKPT" \
    --segmenter_ckpt "$SEGMENTER_CKPT" \
    --device $DEVICE \
    --output_dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/multitask_training_log.txt"

echo ""
echo "=============================================="
echo "STEP 3 COMPLETE"
echo "=============================================="
echo "Models saved to:"
echo "  Best PSNR: $CHECKPOINT_DIR/best_psnr.pth"
echo "  Best Dice: $CHECKPOINT_DIR/best_dice.pth"
echo ""
echo "Training log: $OUTPUT_DIR/multitask_training_log.txt"
echo ""
echo "Next: Evaluate the multi-task model"
echo "=============================================="
