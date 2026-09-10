#!/bin/bash
#
# NSAD Complete Experiment Suite
# Run all experiments for IEEE TMI submission
#
# Usage: ./run_nsad_experiments.sh [quick|full|ablation|eval|all] [gpu|cpu]
#

MODE=${1:-quick}
DEVICE=${2:-cpu}

echo "========================================"
echo "NSAD Experiment Suite"
echo "========================================"
echo "Mode: $MODE"
echo "Device: $DEVICE"
echo "========================================"
echo ""
echo "Available modes:"
echo "  quick    - Quick proof of concept (15 epochs, 200 samples)"
echo "  full     - Full training (100 epochs, 1000 samples)"
echo "  ablation - Ablation study (multiple configs)"
echo "  eval     - Evaluate existing checkpoint"
echo "  all      - Run everything"
echo ""

case $MODE in
    quick)
        echo "Running quick proof of concept..."
        echo ""
        bash run_sansd_quick_test.sh
        ;;

    full)
        echo "Running full training..."
        echo ""
        bash run_nsad_full_training.sh $DEVICE
        ;;

    ablation)
        echo "Running ablation study..."
        echo ""
        bash run_nsad_ablation.sh $DEVICE
        ;;

    eval)
        CHECKPOINT=${3:-checkpoints/sansd_quick_test/best_model.pth}
        echo "Evaluating checkpoint: $CHECKPOINT"
        echo ""
        CHECKPOINT=$CHECKPOINT DEVICE=$DEVICE bash run_nsad_evaluation.sh $CHECKPOINT $DEVICE
        ;;

    all)
        echo "Running complete experiment suite..."
        echo ""

        # Step 1: Quick test
        echo "Step 1/4: Quick proof of concept"
        bash run_sansd_quick_test.sh

        # Step 2: Full training
        echo ""
        echo "Step 2/4: Full training"
        bash run_nsad_full_training.sh $DEVICE

        # Step 3: Ablation study
        echo ""
        echo "Step 3/4: Ablation study"
        bash run_nsad_ablation.sh $DEVICE

        # Step 4: Final evaluation
        echo ""
        echo "Step 4/4: Final evaluation"
        LATEST_CKPT=$(ls -t checkpoints/nsad_full_*/best_model.pth 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            bash run_nsad_evaluation.sh $LATEST_CKPT $DEVICE
        fi

        echo ""
        echo "========================================"
        echo "ALL EXPERIMENTS COMPLETE"
        echo "========================================"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Use: quick, full, ablation, eval, or all"
        exit 1
        ;;
esac

echo ""
echo "Done!"
