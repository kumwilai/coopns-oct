#!/bin/bash
#
# Anatomy-Aware NSAD Training Script
# Novel contribution for IEEE TMI: Layer-aware noise decomposition
#
# Usage: ./run_anatomy_aware_nsad.sh [quick|full|ablation] [gpu|cpu]
#

MODE=${1:-quick}
DEVICE=${2:-cpu}

echo "========================================"
echo "Anatomy-Aware NSAD Training"
echo "========================================"
echo "Mode: $MODE"
echo "Device: $DEVICE"
echo "========================================"
echo ""
echo "Novel Features:"
echo "  - Retinal layer detection (5 zones)"
echo "  - Layer-aware noise decomposition"
echo "  - Depth-adaptive processing"
echo "  - Anatomy-preserving loss"
echo "========================================"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case $MODE in
    quick)
        echo "Running quick proof of concept..."
        EPOCHS=15
        SAMPLES=200
        OUTPUT_DIR="checkpoints/anatomy_aware_quick_${TIMESTAMP}"

        python train_anatomy_aware_nsad.py \
            --output_dir "$OUTPUT_DIR" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --batch_size 4 \
            --device $DEVICE \
            --use_anatomy_loss \
            --use_depth_adaptive \
            --fusion_mode anatomy
        ;;

    full)
        echo "Running full training..."
        EPOCHS=100
        SAMPLES=1000
        OUTPUT_DIR="checkpoints/anatomy_aware_full_${TIMESTAMP}"

        python train_anatomy_aware_nsad.py \
            --output_dir "$OUTPUT_DIR" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --batch_size 8 \
            --device $DEVICE \
            --use_anatomy_loss \
            --use_depth_adaptive \
            --fusion_mode anatomy \
            --lr 5e-5
        ;;

    ablation)
        echo "Running ablation study..."
        BASE_DIR="checkpoints/anatomy_ablation_${TIMESTAMP}"
        mkdir -p "$BASE_DIR"

        EPOCHS=30
        SAMPLES=300

        echo ""
        echo "Ablation 1/5: No anatomy features (baseline)"
        python train_anatomy_aware_nsad.py \
            --output_dir "${BASE_DIR}/no_anatomy" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --device $DEVICE \
            --fusion_mode weighted_sum

        echo ""
        echo "Ablation 2/5: Layer detection only"
        python train_anatomy_aware_nsad.py \
            --output_dir "${BASE_DIR}/layer_only" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --device $DEVICE \
            --fusion_mode anatomy

        echo ""
        echo "Ablation 3/5: Depth adaptive only"
        python train_anatomy_aware_nsad.py \
            --output_dir "${BASE_DIR}/depth_only" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --device $DEVICE \
            --use_depth_adaptive \
            --fusion_mode weighted_sum

        echo ""
        echo "Ablation 4/5: Anatomy loss only"
        python train_anatomy_aware_nsad.py \
            --output_dir "${BASE_DIR}/loss_only" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --device $DEVICE \
            --use_anatomy_loss \
            --fusion_mode weighted_sum

        echo ""
        echo "Ablation 5/5: Full anatomy-aware (all features)"
        python train_anatomy_aware_nsad.py \
            --output_dir "${BASE_DIR}/full_anatomy" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --device $DEVICE \
            --use_anatomy_loss \
            --use_depth_adaptive \
            --fusion_mode anatomy

        echo ""
        echo "========================================"
        echo "Ablation Study Complete"
        echo "Results saved to: $BASE_DIR"
        echo "========================================"

        # Create comparison summary
        echo ""
        echo "=== ABLATION RESULTS SUMMARY ===" | tee "${BASE_DIR}/summary.txt"
        for exp in no_anatomy layer_only depth_only loss_only full_anatomy; do
            if [ -f "${BASE_DIR}/${exp}/training.log" ]; then
                echo "" | tee -a "${BASE_DIR}/summary.txt"
                echo "--- $exp ---" | tee -a "${BASE_DIR}/summary.txt"
                grep "Best model saved" "${BASE_DIR}/${exp}/training.log" | tail -1 | tee -a "${BASE_DIR}/summary.txt"
            fi
        done
        ;;

    compare)
        echo "Comparing anatomy-aware vs standard NSAD..."
        BASE_DIR="checkpoints/anatomy_comparison_${TIMESTAMP}"
        mkdir -p "$BASE_DIR"

        EPOCHS=50
        SAMPLES=500

        echo ""
        echo "Training 1/2: Standard SANSD"
        python train_anatomy_aware_nsad.py \
            --output_dir "${BASE_DIR}/standard_sansd" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --device $DEVICE \
            --fusion_mode weighted_sum

        echo ""
        echo "Training 2/2: Anatomy-Aware SANSD"
        python train_anatomy_aware_nsad.py \
            --output_dir "${BASE_DIR}/anatomy_aware" \
            --epochs $EPOCHS \
            --max_train_samples $SAMPLES \
            --device $DEVICE \
            --use_anatomy_loss \
            --use_depth_adaptive \
            --fusion_mode anatomy

        echo ""
        echo "========================================"
        echo "Comparison Complete"
        echo "========================================"
        echo ""
        echo "Standard SANSD best:"
        grep "Best model saved" "${BASE_DIR}/standard_sansd/training.log" | tail -1
        echo ""
        echo "Anatomy-Aware SANSD best:"
        grep "Best model saved" "${BASE_DIR}/anatomy_aware/training.log" | tail -1
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Usage: ./run_anatomy_aware_nsad.sh [mode] [device]"
        echo ""
        echo "Modes:"
        echo "  quick    - Quick proof of concept (15 epochs, 200 samples)"
        echo "  full     - Full training (100 epochs, 1000 samples)"
        echo "  ablation - Ablation study (5 configurations)"
        echo "  compare  - Compare standard vs anatomy-aware SANSD"
        echo ""
        echo "Device:"
        echo "  cpu      - Use CPU (default)"
        echo "  gpu      - Use CUDA GPU"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
