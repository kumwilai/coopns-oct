#!/bin/bash
# Quick patch script to apply high-priority fixes for adaptive denoising
# Run this before retraining

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "Applying Quick Fixes for Adaptive Denoising"
echo "════════════════════════════════════════════════════════════════"

cd "$(dirname "$0")"

# Backup original file
if [ ! -f "run_duke_region_focused.sh.backup" ]; then
    echo "✓ Creating backup: run_duke_region_focused.sh.backup"
    cp run_duke_region_focused.sh run_duke_region_focused.sh.backup
fi

# Fix 1: Increase residual blend init from 0.1 to 0.3
echo ""
echo "[Fix 1] Increasing residual_blend_init: 0.1 → 0.3"
echo "  Rationale: Allow heads to contribute 30% (was 10%)"

if grep -q "residual_blend_init" run_duke_region_focused.sh; then
    sed -i 's/--residual_blend_init [0-9.]\+/--residual_blend_init 0.3/g' run_duke_region_focused.sh
    echo "  ✓ Applied"
else
    echo "  ⚠ Parameter not found - may need manual addition"
fi

# Fix 2: Increase noise map loss weight in Phase 2B
echo ""
echo "[Fix 2] Increasing noise_map_loss_weight in Phase 2B: 0.05 → 0.1"
echo "  Rationale: Maintain spatial refiner supervision"

# Target only Phase 2B (after line containing "PHASE 2B")
awk '
/PHASE 2B/ { phase2b=1 }
phase2b && /--noise_map_loss_weight 0\.05/ {
    gsub(/--noise_map_loss_weight 0\.05/, "--noise_map_loss_weight 0.1")
    phase2b=0  # Only replace first occurrence in Phase 2B
}
{ print }
' run_duke_region_focused.sh > run_duke_region_focused.sh.tmp && \
mv run_duke_region_focused.sh.tmp run_duke_region_focused.sh

echo "  ✓ Applied"

# Fix 3: Add enhanced loss parameters (if not present)
echo ""
echo "[Fix 3] Checking for enhanced loss parameters"

if ! grep -q "head_quality_weight" run_duke_region_focused.sh; then
    echo "  ⚠ Enhanced loss parameters not found"
    echo "  → These must be manually integrated into train_hybrid_nsnd_multitask.py"
    echo "  → See FIXES_SUMMARY.md for integration instructions"
else
    echo "  ✓ Already present"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Quick Fixes Applied Successfully"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Run Phase 3 evaluation to test fix:"
echo "     bash run_duke_region_focused.sh"
echo ""
echo "  2. Run diagnostic on current model:"
echo "     python nsnd_oct/scripts/diagnose_head_specialization.py \\"
echo "       --checkpoint checkpoints/multitask_hybrid_nsnd_lambda0p006to0p002_cosine_best.pth \\"
echo "       --test_pairs test_pairs_duke_analysis_maps.txt \\"
echo "       --output_dir diagnostics/head_analysis \\"
echo "       --base_nafnet_width 64"
echo ""
echo "  3. For full fix, integrate enhanced losses from:"
echo "     nsnd_oct/scripts/fix_adaptive_denoising.py"
echo "     (See FIXES_SUMMARY.md for details)"
echo ""
echo "Original file backed up to: run_duke_region_focused.sh.backup"
echo ""
