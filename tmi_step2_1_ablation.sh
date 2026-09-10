#!/bin/bash
# =============================================================================
# TMI Step 2.1: Ablation Study - WITHOUT Layer-Specific Gates
# =============================================================================
# Purpose: Prove the contribution of layer-specific noise modeling
# Time: ~3 hours
# =============================================================================
set -e

echo "=============================================="
echo "TMI STEP 2.1: ABLATION STUDY"
echo "=============================================="
echo ""
echo "This trains multi-task model with GLOBAL gate"
echo "instead of 5 layer-specific gates."
echo ""
echo "Expected: PSNR should be ~0.3-0.5 dB LOWER than full method"
echo "=============================================="

# Run the ablation training
bash tmi_ablation_no_layer_gates.sh

echo ""
echo "=============================================="
echo "STEP 2.1 COMPLETE"
echo "=============================================="
echo "Next: bash tmi_step3_1_eval_full.sh"
echo "=============================================="
