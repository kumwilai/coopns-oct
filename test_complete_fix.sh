#!/usr/bin/env bash
# Complete fix verification with map pre-training

BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
ANALYZER_CKPT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"
TRAIN_PAIRS="train_pairs_duke_analysis_maps.txt"
VAL_PAIRS="val_pairs_duke_analysis_maps.txt"
TRAIN_WEIGHTS="weights_duke_analysis_maps_train.jsonl"
VAL_WEIGHTS="weights_duke_analysis_maps_val.jsonl"

echo "================================================================================"
echo "Complete Fix Verification: Adaptive Conditioning"
echo "--------------------------------------------------------------------------------"
echo "Root Cause Identified:"
echo "  1. Low confidence (~0.04) heavily gates modulation via gate = confidence"
echo "  2. Small basis_init_std (1e-2) produces tiny modulation vectors"
echo "  3. Small alpha (0.1) further reduces modulation magnitude"
echo "  4. Uniform spatial map (untrained) averages zero-mean basis to ~0"
echo ""
echo "Applied Fixes:"
echo "  1. alpha: 0.1 → 1.0 (10x increase for visible modulation)"
echo "  2. gate_floor: 0.0 → 0.3 (prevent confidence from killing modulation)"
echo "  3. basis_init_std: 1e-2 → 0.1 (10x increase in basis magnitude)"
echo "  4. map_only_epochs: 0 → 3 (pre-train map to be non-uniform first)"
echo ""
echo "Expected after fix:"
echo "  - Epoch 1-3: Map sharpens (entropy drops, mapMax increases)"
echo "  - Epoch 4+: Δbase > 0.001, GainPSNR becomes positive"
echo "================================================================================"

python -u nsnd_oct/scripts/train_adaptive_nafnet_full.py \
    --pairs_train "$TRAIN_PAIRS" \
    --pairs_val "$VAL_PAIRS" \
    --weights_jsonl_train "$TRAIN_WEIGHTS" \
    --weights_jsonl_val "$VAL_WEIGHTS" \
    --base_ckpt "$BASE_CKPT" \
    --analyzer_ckpt "$ANALYZER_CKPT" \
    --epochs 5 \
    --max_samples 32 \
    --batch_size 4 \
    --lr 1e-4 \
    --cond_dropout 0.2 \
    --usage_loss_weight 0.1 \
    --noise_map_loss_weight 0.2 \
    --ortho_weight 0.05 \
    --sparsity_weight 0.01 \
    --alpha 1.0 \
    --gate_floor 0.3 \
    --basis_init_std 0.1 \
    --symbolic_strength 0.2 \
    --grad_loss_weight 0.0 \
    --base_delta_weight 0.2 \
    --base_delta_margin 0.001 \
    --map_only_epochs 3 \
    --log_interval 2 \
    --output_dir "checkpoints/adaptive_nafnet_full_smoke" \
    --seed 42
