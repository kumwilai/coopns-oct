#!/bin/bash
# =============================================================================
# TMI Step 3: Ablation Study
# =============================================================================
set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmi_config.sh"

print_config

echo ""
echo "=============================================="
echo "STEP 3: ABLATION STUDY"
echo "=============================================="
echo ""
echo "Parameters:"
echo "  Epochs:     $ABLATION_EPOCHS"
echo "  Max Train:  $ABLATION_MAX_TRAIN"
echo "  Max Val:    $ABLATION_MAX_VAL"
echo ""
echo "Ablation variants:"
echo "  1. full_model       - Full model with all components"
echo "  2. backbone_only    - NAFNet backbone only"
echo "  3. no_physics       - Without physics-based features"
echo "  4. no_layer         - Without anatomical layer guidance"
echo "  5. hard_classification - Hard noise classification"
echo "  6. frozen_backbone  - Frozen backbone (refinement only)"
echo ""

ABLATION_DIR="$OUTPUT_DIR/ablation"
mkdir -p "$ABLATION_DIR"

python run_ablation_study.py \
    --epochs $ABLATION_EPOCHS \
    --max_train $ABLATION_MAX_TRAIN \
    --max_val $ABLATION_MAX_VAL \
    --base_ckpt "$BACKBONE_PATH" \
    --device $DEVICE \
    --output_dir "$ABLATION_DIR" \
    2>&1 | tee "$OUTPUT_DIR/ablation_log.txt"

echo ""
echo "=============================================="
echo "STEP 3 COMPLETE"
echo "=============================================="
echo "Results saved to: $ABLATION_DIR"
echo "  - ablation_results.json"
echo ""
echo "Next step: bash tmi_step4_crossdataset.sh"
echo "=============================================="
