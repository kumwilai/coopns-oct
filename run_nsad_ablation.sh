#!/bin/bash
#
# NSAD Ablation Study Script
# Tests different configurations to validate contributions
#
# Usage: ./run_nsad_ablation.sh [gpu|cpu]
#

set -e

DEVICE=${1:-cpu}
EPOCHS=20
BATCH_SIZE=4
LR=5e-4
MAX_SAMPLES=200

echo "========================================"
echo "NSAD Ablation Study"
echo "========================================"
echo "Device: $DEVICE"
echo "Epochs per config: $EPOCHS"
echo "========================================"

# Common args
COMMON_ARGS="--train_jsonl weights_duke_analysis_maps_train.jsonl \
    --val_jsonl weights_duke_analysis_maps_val.jsonl \
    --patch_size 64 \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --max_train_samples $MAX_SAMPLES \
    --base_ckpt outputs/nafnet_analysis_maps_w64/nafnet_best.pth \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_level 0.0 \
    --lambda_speckle 0.0 \
    --lambda_shot 0.0 \
    --device $DEVICE"

# Create results directory
RESULTS_DIR="ablation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

run_ablation() {
    local name=$1
    local extra_args=$2

    echo ""
    echo "========================================"
    echo "Running: $name"
    echo "========================================"

    OUTPUT_DIR="$RESULTS_DIR/$name"
    mkdir -p $OUTPUT_DIR

    python train_sansd.py \
        $COMMON_ARGS \
        $extra_args \
        --output_dir $OUTPUT_DIR \
        2>&1 | tee $OUTPUT_DIR/training.log

    # Extract best PSNR from log
    BEST_PSNR=$(grep "Best model saved" $OUTPUT_DIR/training.log | tail -1 | grep -oP '\d+\.\d+ dB' | head -1)
    GAIN=$(grep "Best model saved" $OUTPUT_DIR/training.log | tail -1 | grep -oP '\+\d+\.\d+ dB' | head -1)

    echo "$name: PSNR=$BEST_PSNR, Gain=$GAIN" >> $RESULTS_DIR/summary.txt
}

# ============================================================
# ABLATION 1: Fusion Mode Comparison
# ============================================================
echo ""
echo "ABLATION 1: Fusion Mode Comparison"
echo "========================================"

run_ablation "fusion_gated" "--fusion_mode gated --alpha 2.0"
run_ablation "fusion_residual" "--fusion_mode residual --alpha 2.0"
run_ablation "fusion_weighted" "--fusion_mode weighted --alpha 2.0"

# ============================================================
# ABLATION 2: Alpha (Symbolic Weight) Comparison
# ============================================================
echo ""
echo "ABLATION 2: Alpha Comparison"
echo "========================================"

run_ablation "alpha_1.0" "--fusion_mode gated --alpha 1.0"
run_ablation "alpha_2.0" "--fusion_mode gated --alpha 2.0"
run_ablation "alpha_4.0" "--fusion_mode gated --alpha 4.0"

# ============================================================
# ABLATION 3: Learning Rate Comparison
# ============================================================
echo ""
echo "ABLATION 3: Learning Rate Comparison"
echo "========================================"

run_ablation "lr_1e-4" "--fusion_mode gated --alpha 2.0 --lr 1e-4"
run_ablation "lr_5e-4" "--fusion_mode gated --alpha 2.0 --lr 5e-4"
run_ablation "lr_1e-3" "--fusion_mode gated --alpha 2.0 --lr 1e-3"

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "ABLATION STUDY COMPLETE"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary:"
cat $RESULTS_DIR/summary.txt
echo "========================================"
