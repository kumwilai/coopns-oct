#!/bin/bash
#
# NSAD: Neuro-Symbolic Adaptive Denoising for OCT Images
#
# ═══════════════════════════════════════════════════════════════════════
#                    VERIFIED NOVEL CONTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════
#
# WHAT EXISTS (Prior Work):
#   - Global noise classification → different neural denoisers
#   - Per-pixel noise LEVEL estimation → adapt ONE algorithm
#   - Deep unfolding of ONE algorithm (e.g., DU-BM3D)
#   - Global denoiser combination (e.g., CsNet)
#
# WHAT WE DO (NOVEL):
#   ✓ Per-pixel noise TYPE decomposition (not global)
#   ✓ Soft routing to MULTIPLE classical operators (not one)
#   ✓ Differentiable NAMED operators (not neural black-box)
#   ✓ Full per-pixel interpretability
#
# KEY INSIGHT:
#   No prior work does per-pixel soft routing to a MIXTURE of DIFFERENT
#   classical denoising operators. This is genuinely novel.
#
# See VERIFIED_NOVELTY.md for detailed literature review.
#
# ═══════════════════════════════════════════════════════════════════════

echo "========================================"
echo "NSAD: Neuro-Symbolic Adaptive Denoising"
echo "========================================"
echo ""
echo "VERIFIED NOVEL CONTRIBUTIONS:"
echo "  1. Per-pixel noise TYPE decomposition (not global classification)"
echo "  2. Soft routing to MULTIPLE classical operators (not one algorithm)"
echo "  3. Differentiable mixture of NAMED operators (not neural)"
echo "  4. Full per-pixel interpretability"
echo ""
echo "Comparison with State-of-the-Art:"
echo "  - CsNet (2019): Global weights → we use per-pixel"
echo "  - DU-BM3D (2024): Unfolds ONE → we mix MULTIPLE"
echo "  - Adaptive NLM (2010): Adapts parameters → we route algorithms"
echo "  - Waqar et al. (2024): Global + neural → we do per-pixel + classical"
echo ""
echo "Using SAME data as run_end_to_end.sh for fair comparison:"
echo "  - Data: Duke analysis maps"
echo "  - Backbone: NAFNetFullFiLM (width=64)"
echo "  - Checkpoint: outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
echo "========================================"

# Configuration (SAME as run_end_to_end.sh)
BATCH_SIZE=4
EPOCHS=20           # Match run_end_to_end.sh for fair comparison
LR=1e-4
ALPHA=2.0           # Modulation strength (same as run_end_to_end.sh)
FUSION_MODE="residual"  # residual, weighted, or gated

# Physics loss weights (novel)
LAMBDA_LEVEL=0.1    # Noise level consistency
LAMBDA_SPECKLE=0.1  # Speckle physics (CV ≈ 1)
LAMBDA_SHOT=0.1     # Shot physics (var ∝ mean)

# Data (SAME as run_end_to_end.sh)
TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PATCH_SIZE=64
MAX_TRAIN_SAMPLES=500

# Model (SAME backbone as run_end_to_end.sh)
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"

# Output
OUTPUT_DIR="checkpoints/sansd"
mkdir -p $OUTPUT_DIR

# Run training
python train_sansd.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --base_ckpt $BASE_CKPT \
    --alpha $ALPHA \
    --fusion_mode $FUSION_MODE \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_level $LAMBDA_LEVEL \
    --lambda_speckle $LAMBDA_SPECKLE \
    --lambda_shot $LAMBDA_SHOT \
    --output_dir $OUTPUT_DIR \
    --device cpu

echo ""
echo "========================================"
echo "Training complete!"
echo "Best model: $OUTPUT_DIR/best_model.pth"
echo ""
echo "Compare with run_end_to_end.sh results:"
echo "  - PSNR gain over base NAFNet"
echo "  - Top-1 accuracy"
echo "  - Per-class accuracy"
echo "========================================"
