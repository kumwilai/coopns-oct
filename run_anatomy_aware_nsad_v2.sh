#!/bin/bash
#
# Anatomy-Aware NSAD Training v2
# Improvements: SSIM loss, layer-specific metrics, stronger diversity
#
# Usage: ./run_anatomy_aware_nsad_v2.sh [quick|full|ssim_focus] [gpu|cpu]
#

MODE=${1:-quick}
DEVICE=${2:-cpu}

echo "========================================"
echo "Anatomy-Aware NSAD Training v2"
echo "========================================"
echo "Mode: $MODE"
echo "Device: $DEVICE"
echo "========================================"
echo ""
echo "v2 Improvements:"
echo "  - SSIM loss (global + layer-specific)"
echo "  - Stronger expert diversity"
echo "  - Layer-specific metrics reporting"
echo "  - Tunable loss weights"
echo "========================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case $MODE in
    quick)
        echo "Running quick test..."
        OUTPUT_DIR="checkpoints/anatomy_v2_quick_${TIMESTAMP}"

        python train_anatomy_aware_nsad_v2.py \
            --output_dir "$OUTPUT_DIR" \
            --epochs 15 \
            --max_train_samples 200 \
            --batch_size 4 \
            --device $DEVICE \
            --use_depth_adaptive \
            --use_anatomy_fusion \
            --fusion_mode anatomy \
            --w_l1 1.0 \
            --w_ssim 0.5 \
            --w_layer_ssim 0.3 \
            --w_anatomy 0.1 \
            --w_diversity 0.2 \
            --w_specialization 0.1
        ;;

    full)
        echo "Running full training..."
        OUTPUT_DIR="checkpoints/anatomy_v2_full_${TIMESTAMP}"

        python train_anatomy_aware_nsad_v2.py \
            --output_dir "$OUTPUT_DIR" \
            --epochs 100 \
            --max_train_samples 1000 \
            --batch_size 8 \
            --device $DEVICE \
            --use_depth_adaptive \
            --use_anatomy_fusion \
            --fusion_mode anatomy \
            --lr 5e-5 \
            --w_l1 1.0 \
            --w_ssim 0.5 \
            --w_layer_ssim 0.3 \
            --w_anatomy 0.1 \
            --w_diversity 0.2 \
            --w_specialization 0.1
        ;;

    ssim_focus)
        echo "Running with SSIM-focused training..."
        OUTPUT_DIR="checkpoints/anatomy_v2_ssim_${TIMESTAMP}"

        # Higher SSIM weights for better structural preservation
        python train_anatomy_aware_nsad_v2.py \
            --output_dir "$OUTPUT_DIR" \
            --epochs 50 \
            --max_train_samples 500 \
            --batch_size 4 \
            --device $DEVICE \
            --use_depth_adaptive \
            --use_anatomy_fusion \
            --fusion_mode anatomy \
            --w_l1 0.5 \
            --w_ssim 1.0 \
            --w_layer_ssim 0.5 \
            --w_anatomy 0.1 \
            --w_diversity 0.3 \
            --w_specialization 0.15
        ;;

    layer_ablation)
        echo "Running layer-specific ablation..."
        BASE_DIR="checkpoints/anatomy_v2_layer_ablation_${TIMESTAMP}"
        mkdir -p "$BASE_DIR"

        EPOCHS=30
        SAMPLES=300

        echo ""
        echo "1/4: Uniform weights (no layer emphasis)"
        python train_anatomy_aware_nsad_v2.py \
            --output_dir "${BASE_DIR}/uniform" \
            --epochs $EPOCHS --max_train_samples $SAMPLES \
            --device $DEVICE --use_depth_adaptive --use_anatomy_fusion \
            --w_layer_ssim 0.0

        echo ""
        echo "2/4: NFL emphasis (top layers)"
        python train_anatomy_aware_nsad_v2.py \
            --output_dir "${BASE_DIR}/nfl_emphasis" \
            --epochs $EPOCHS --max_train_samples $SAMPLES \
            --device $DEVICE --use_depth_adaptive --use_anatomy_fusion \
            --w_layer_ssim 0.5

        echo ""
        echo "3/4: High SSIM focus"
        python train_anatomy_aware_nsad_v2.py \
            --output_dir "${BASE_DIR}/high_ssim" \
            --epochs $EPOCHS --max_train_samples $SAMPLES \
            --device $DEVICE --use_depth_adaptive --use_anatomy_fusion \
            --w_ssim 1.5 --w_layer_ssim 0.5

        echo ""
        echo "4/4: High diversity (force specialization)"
        python train_anatomy_aware_nsad_v2.py \
            --output_dir "${BASE_DIR}/high_diversity" \
            --epochs $EPOCHS --max_train_samples $SAMPLES \
            --device $DEVICE --use_depth_adaptive --use_anatomy_fusion \
            --w_diversity 0.5 --w_specialization 0.3

        echo ""
        echo "Ablation complete! Results in: $BASE_DIR"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Usage: ./run_anatomy_aware_nsad_v2.sh [mode] [device]"
        echo ""
        echo "Modes:"
        echo "  quick         - Quick test (15 epochs, 200 samples)"
        echo "  full          - Full training (100 epochs, 1000 samples)"
        echo "  ssim_focus    - SSIM-focused (higher SSIM weights)"
        echo "  layer_ablation - Test different layer emphasis strategies"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
