#!/bin/bash
# =============================================================================
# TMI Step 4.1: Train SOTA Baselines with Fair Parameter Count
# =============================================================================
# Purpose: Train DnCNN and Restormer with ~9.5M params (same as ours)
# Time: ~6 hours total (3 hours each)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI STEP 4.1: TRAIN SOTA BASELINES"
echo "=============================================="
echo ""
echo "Training SOTA methods with FAIR parameter counts:"
echo "  - DnCNN:     31 layers, 192 features = ~9.6M params"
echo "  - Restormer: dim=48, blocks=[2,2,2,2] = ~10.2M params"
echo "  - Ours:      NAFNet + Multi-task      = ~9.5M params"
echo "=============================================="

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"

# Output directory
SOTA_OUTPUT="sota_baselines_fair"
mkdir -p "$SOTA_OUTPUT"

# Training parameters (same as ours for fair comparison)
EPOCHS=30
BATCH_SIZE=4
MAX_TRAIN=2000
MAX_VAL=400
LR=1e-4

echo ""
echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Max train: $MAX_TRAIN"
echo "  Max val: $MAX_VAL"
echo ""

# Train DnCNN
echo "=============================================="
echo "Training DnCNN (31 layers, 192 features)"
echo "=============================================="

python train_sota_fair.py \
    --model dncnn \
    --dncnn_layers 31 \
    --dncnn_features 192 \
    --train_jsonl seg_data/seg_train.jsonl \
    --val_jsonl seg_data/seg_val.jsonl \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_train $MAX_TRAIN \
    --max_val $MAX_VAL \
    --lr $LR \
    --device $DEVICE \
    --output_dir "$SOTA_OUTPUT/dncnn_fair" \
    2>&1 | tee "$SOTA_OUTPUT/dncnn_training_log.txt"

echo ""
echo "=============================================="
echo "Training Restormer (dim=48)"
echo "=============================================="

python train_sota_fair.py \
    --model restormer \
    --restormer_dim 48 \
    --train_jsonl seg_data/seg_train.jsonl \
    --val_jsonl seg_data/seg_val.jsonl \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_train $MAX_TRAIN \
    --max_val $MAX_VAL \
    --lr $LR \
    --device $DEVICE \
    --output_dir "$SOTA_OUTPUT/restormer_fair" \
    2>&1 | tee "$SOTA_OUTPUT/restormer_training_log.txt"

echo ""
echo "=============================================="
echo "STEP 4.1 COMPLETE"
echo "=============================================="
echo "SOTA models saved to: $SOTA_OUTPUT/"
echo "Next: bash tmi_step4_2_eval_sota.sh"
echo "=============================================="
