#!/bin/bash
# =============================================================================
# TMI Step 2: Comprehensive Evaluation
# =============================================================================
set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmi_config.sh"

print_config

echo ""
echo "=============================================="
echo "STEP 2: COMPREHENSIVE EVALUATION"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please run tmi_step1_train.sh first"
    exit 1
fi

echo ""
echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Max validation samples: $EVAL_MAX_VAL"
echo ""

EVAL_DIR="$OUTPUT_DIR/evaluation"
mkdir -p "$EVAL_DIR"

python run_comprehensive_evaluation.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --backbone "$BACKBONE_PATH" \
    --val_jsonl "$DUKE_VAL_JSONL" \
    --max_val $EVAL_MAX_VAL \
    --device $DEVICE \
    --output_dir "$EVAL_DIR" \
    2>&1 | tee "$OUTPUT_DIR/evaluation_log.txt"

echo ""
echo "=============================================="
echo "STEP 2 COMPLETE"
echo "=============================================="
echo "Results saved to: $EVAL_DIR"
echo "  - comprehensive_report.txt"
echo "  - all_results.json"
echo "  - sample_comparison.png"
echo "  - interpretability.png"
echo ""
echo "Next step: bash tmi_step3_ablation.sh"
echo "=============================================="
