#!/bin/bash
# =============================================================================
# TMI Quick Test - Run all steps with minimal samples
# Use this to verify the pipeline works before running the full version
# =============================================================================
set -e

echo "=============================================="
echo "TMI QUICK TEST"
echo "Running all steps with minimal samples"
echo "=============================================="

# Create test output directory
export TMI_OUTPUT_DIR="tmi_test_$(date +%Y%m%d_%H%M%S)"

# Override parameters for quick testing
export TRAIN_EPOCHS=2
export TRAIN_MAX_SAMPLES=20
export TRAIN_VAL_SAMPLES=10
export ABLATION_EPOCHS=1
export ABLATION_MAX_TRAIN=10
export ABLATION_MAX_VAL=5
export EVAL_MAX_VAL=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "Test output directory: $TMI_OUTPUT_DIR"
echo ""

# Step 1: Quick Training
echo ">>> Running Step 1: Training (2 epochs, 20 samples)"
source "$SCRIPT_DIR/tmi_config.sh"
bash "$SCRIPT_DIR/tmi_step1_train.sh"

# Step 2: Quick Evaluation
echo ""
echo ">>> Running Step 2: Evaluation (10 samples)"
bash "$SCRIPT_DIR/tmi_step2_evaluate.sh"

# Step 3: Quick Ablation
echo ""
echo ">>> Running Step 3: Ablation (1 epoch, 10 samples)"
bash "$SCRIPT_DIR/tmi_step3_ablation.sh"

# Step 4: Quick Cross-Dataset
echo ""
echo ">>> Running Step 4: Cross-Dataset (10 samples)"
bash "$SCRIPT_DIR/tmi_step4_crossdataset.sh"

echo ""
echo "=============================================="
echo "QUICK TEST COMPLETE!"
echo "=============================================="
echo ""
echo "All steps executed successfully."
echo "Test results saved to: $TMI_OUTPUT_DIR"
echo ""
echo "To run the full pipeline:"
echo "  1. bash tmi_step1_train.sh"
echo "  2. bash tmi_step2_evaluate.sh"
echo "  3. bash tmi_step3_ablation.sh"
echo "  4. bash tmi_step4_crossdataset.sh"
echo ""
echo "Or set custom output directory:"
echo "  export TMI_OUTPUT_DIR=my_experiment"
echo "  bash tmi_step1_train.sh"
echo "=============================================="
