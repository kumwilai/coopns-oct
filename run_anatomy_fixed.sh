#!/bin/bash
# Run anatomy-aware training with memory and specialization fixes

set -e

MODE=${1:-quick}
DEVICE=${2:-cpu}

echo "========================================"
echo "Anatomy-Aware NSAD Training - FIXED"
echo "========================================"
echo "Mode: $MODE"
echo "Device: $DEVICE"
echo "========================================"

OUTPUT_DIR="checkpoints/anatomy_fixed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

case $MODE in
    quick)
        echo "Running quick test (15 epochs, 200 samples)..."
        python train_anatomy_aware_fixed.py \
            --epochs 15 \
            --max_train_samples 200 \
            --batch_size 4 \
            --device "$DEVICE" \
            --output_dir "$OUTPUT_DIR" \
            --memory_monitor \
            --use_depth_adaptive \
            --use_anatomy_fusion \
            --w_diversity 0.3 \
            --w_entropy 0.2 \
            --w_noise_sup 0.2 \
            2>&1 | tee "${OUTPUT_DIR}/training.log"
        ;;

    full)
        echo "Running full training (100 epochs, 1000 samples)..."
        python train_anatomy_aware_fixed.py \
            --epochs 100 \
            --max_train_samples 1000 \
            --batch_size 4 \
            --device "$DEVICE" \
            --output_dir "$OUTPUT_DIR" \
            --memory_monitor \
            --use_depth_adaptive \
            --use_anatomy_fusion \
            --w_diversity 0.3 \
            --w_entropy 0.2 \
            --w_noise_sup 0.2 \
            2>&1 | tee "${OUTPUT_DIR}/training.log"
        ;;

    ablation)
        echo "Running ablation study..."

        # 1. No fixes (original losses)
        echo "=== Ablation 1: Original losses (no fixes) ==="
        python train_anatomy_aware_fixed.py \
            --epochs 15 \
            --max_train_samples 200 \
            --device "$DEVICE" \
            --output_dir "${OUTPUT_DIR}/ablation_original" \
            --w_diversity 0.1 \
            --w_entropy 0.0 \
            --w_noise_sup 0.0 \
            2>&1 | tee "${OUTPUT_DIR}/ablation_original.log"

        # 2. With diversity fix only
        echo "=== Ablation 2: Strong diversity only ==="
        python train_anatomy_aware_fixed.py \
            --epochs 15 \
            --max_train_samples 200 \
            --device "$DEVICE" \
            --output_dir "${OUTPUT_DIR}/ablation_diversity" \
            --w_diversity 0.3 \
            --w_entropy 0.0 \
            --w_noise_sup 0.0 \
            2>&1 | tee "${OUTPUT_DIR}/ablation_diversity.log"

        # 3. With entropy minimization
        echo "=== Ablation 3: Diversity + Entropy ==="
        python train_anatomy_aware_fixed.py \
            --epochs 15 \
            --max_train_samples 200 \
            --device "$DEVICE" \
            --output_dir "${OUTPUT_DIR}/ablation_entropy" \
            --w_diversity 0.3 \
            --w_entropy 0.2 \
            --w_noise_sup 0.0 \
            2>&1 | tee "${OUTPUT_DIR}/ablation_entropy.log"

        # 4. Full (all fixes)
        echo "=== Ablation 4: All fixes ==="
        python train_anatomy_aware_fixed.py \
            --epochs 15 \
            --max_train_samples 200 \
            --device "$DEVICE" \
            --output_dir "${OUTPUT_DIR}/ablation_full" \
            --use_depth_adaptive \
            --use_anatomy_fusion \
            --w_diversity 0.3 \
            --w_entropy 0.2 \
            --w_noise_sup 0.2 \
            2>&1 | tee "${OUTPUT_DIR}/ablation_full.log"

        echo "Ablation study complete! Check ${OUTPUT_DIR}/"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [quick|full|ablation] [cpu|gpu]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Training Complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
