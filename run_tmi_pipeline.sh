#!/bin/bash
# =============================================================================
# TMI Submission Pipeline
# Interpretable Anatomy-Aware OCT Denoising with Physics-Based Conditioning
# =============================================================================

set -e

echo "=============================================="
echo "TMI SUBMISSION PIPELINE"
echo "Interpretable Anatomy-Aware OCT Denoising"
echo "=============================================="

# Configuration
DEVICE="cpu"
OUTPUT_BASE="tmi_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "GPU detected, using CUDA"
        DEVICE="cuda"
    fi
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# =============================================================================
# PHASE 1: Training
# =============================================================================
echo "=============================================="
echo "PHASE 1: TRAINING MAIN MODEL"
echo "=============================================="

python train_soft_conditioning.py \
    --epochs 30 \
    --batch_size 4 \
    --max_train 500 \
    --max_val 100 \
    --lr 1e-4 \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR/checkpoints \
    2>&1 | tee $OUTPUT_DIR/training_log.txt

echo ""
echo "Training complete!"
echo ""

# =============================================================================
# PHASE 2: Comprehensive Evaluation
# =============================================================================
echo "=============================================="
echo "PHASE 2: COMPREHENSIVE EVALUATION"
echo "=============================================="

python run_comprehensive_evaluation.py \
    --checkpoint $OUTPUT_DIR/checkpoints/best.pth \
    --max_val 100 \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR/evaluation \
    2>&1 | tee $OUTPUT_DIR/evaluation_log.txt

echo ""
echo "Comprehensive evaluation complete!"
echo ""

# =============================================================================
# PHASE 3: Ablation Study
# =============================================================================
echo "=============================================="
echo "PHASE 3: ABLATION STUDY"
echo "=============================================="

python run_ablation_study.py \
    --epochs 10 \
    --max_train 200 \
    --max_val 50 \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR/ablation \
    2>&1 | tee $OUTPUT_DIR/ablation_log.txt

echo ""
echo "Ablation study complete!"
echo ""

# =============================================================================
# PHASE 4: Cross-Dataset Evaluation (Duke + PKU37)
# =============================================================================
echo "=============================================="
echo "PHASE 4: CROSS-DATASET EVALUATION"
echo "=============================================="

# Evaluate on Duke dataset
echo "Evaluating on Duke dataset..."
python run_comprehensive_evaluation.py \
    --checkpoint $OUTPUT_DIR/checkpoints/best.pth \
    --val_jsonl weights_duke_analysis_maps_val.jsonl \
    --max_val 100 \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR/duke_eval \
    2>&1 | tee $OUTPUT_DIR/duke_eval_log.txt

# Evaluate on PKU37 dataset (if available)
if [ -f "pku37_oct_dataset/weights_pku37_analysis_val.jsonl" ]; then
    echo "Evaluating on PKU37 dataset..."
    python run_comprehensive_evaluation.py \
        --checkpoint $OUTPUT_DIR/checkpoints/best.pth \
        --val_jsonl pku37_oct_dataset/weights_pku37_analysis_val.jsonl \
        --max_val 100 \
        --device $DEVICE \
        --output_dir $OUTPUT_DIR/pku37_eval \
        2>&1 | tee $OUTPUT_DIR/pku37_eval_log.txt
else
    echo "PKU37 validation file not found, skipping..."
fi

echo ""
echo "Cross-dataset evaluation complete!"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - $OUTPUT_DIR/training_log.txt"
echo "  - $OUTPUT_DIR/checkpoints/best.pth"
echo "  - $OUTPUT_DIR/evaluation/comprehensive_report.txt"
echo "  - $OUTPUT_DIR/evaluation/all_results.json"
echo "  - $OUTPUT_DIR/ablation/ablation_results.json"
echo "  - $OUTPUT_DIR/duke_eval/comprehensive_report.txt"
if [ -f "pku37_oct_dataset/weights_pku37_analysis_val.jsonl" ]; then
    echo "  - $OUTPUT_DIR/pku37_eval/comprehensive_report.txt"
fi
echo ""
echo "Visualizations:"
echo "  - $OUTPUT_DIR/evaluation/sample_comparison.png"
echo "  - $OUTPUT_DIR/evaluation/interpretability.png"
echo ""
echo "=============================================="
