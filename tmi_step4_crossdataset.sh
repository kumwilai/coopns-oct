#!/bin/bash
# =============================================================================
# TMI Step 4: Cross-Dataset Evaluation (Duke + PKU37)
# =============================================================================
set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmi_config.sh"

print_config

echo ""
echo "=============================================="
echo "STEP 4: CROSS-DATASET EVALUATION"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please run tmi_step1_train.sh first"
    exit 1
fi

echo ""
echo "Using checkpoint: $CHECKPOINT_PATH"
echo ""

# Duke Dataset Evaluation
echo "----------------------------------------------"
echo "Evaluating on Duke Dataset"
echo "----------------------------------------------"

DUKE_EVAL_DIR="$OUTPUT_DIR/duke_eval"
mkdir -p "$DUKE_EVAL_DIR"

python run_comprehensive_evaluation.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --backbone "$BACKBONE_PATH" \
    --val_jsonl "$DUKE_VAL_JSONL" \
    --max_val $EVAL_MAX_VAL \
    --device $DEVICE \
    --output_dir "$DUKE_EVAL_DIR" \
    2>&1 | tee "$OUTPUT_DIR/duke_eval_log.txt"

echo ""
echo "Duke evaluation complete!"
echo ""

# PKU37 Dataset Evaluation
echo "----------------------------------------------"
echo "Evaluating on PKU37 Dataset"
echo "----------------------------------------------"

if [ -f "$PKU37_VAL_JSONL" ]; then
    PKU37_EVAL_DIR="$OUTPUT_DIR/pku37_eval"
    mkdir -p "$PKU37_EVAL_DIR"

    python run_comprehensive_evaluation.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --backbone "$BACKBONE_PATH" \
        --val_jsonl "$PKU37_VAL_JSONL" \
        --max_val $EVAL_MAX_VAL \
        --device $DEVICE \
        --output_dir "$PKU37_EVAL_DIR" \
        2>&1 | tee "$OUTPUT_DIR/pku37_eval_log.txt"

    echo ""
    echo "PKU37 evaluation complete!"
else
    echo "PKU37 validation file not found at: $PKU37_VAL_JSONL"
    echo "Skipping PKU37 evaluation..."
fi

echo ""
echo "=============================================="
echo "STEP 4 COMPLETE"
echo "=============================================="
echo "Results saved to:"
echo "  - $DUKE_EVAL_DIR/comprehensive_report.txt"
if [ -f "$PKU37_VAL_JSONL" ]; then
    echo "  - $PKU37_EVAL_DIR/comprehensive_report.txt"
fi
echo ""
echo "=============================================="
echo "ALL PIPELINE STEPS COMPLETE!"
echo "=============================================="
