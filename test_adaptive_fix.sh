#!/usr/bin/env bash
# Quick smoke test to verify adaptive conditioning fix

BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
ANALYZER_CKPT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"
TRAIN_PAIRS="train_pairs_duke_analysis_maps.txt"
VAL_PAIRS="val_pairs_duke_analysis_maps.txt"
TRAIN_WEIGHTS="weights_duke_analysis_maps_train.jsonl"
VAL_WEIGHTS="weights_duke_analysis_maps_val.jsonl"

echo "================================================================================"
echo "Smoke Test: Verifying Adaptive Conditioning Fix"
echo "--------------------------------------------------------------------------------"
echo "Changes:"
echo "  - alpha: 0.1 → 0.5 (5x increase)"
echo "  - gate_floor: 0.0 → 0.3 (prevent confidence gating)"
echo "  - basis_init_std: 1e-2 → 0.1 (10x increase)"
echo "================================================================================"

python -u nsnd_oct/scripts/train_adaptive_nafnet_full.py \
    --pairs_train "$TRAIN_PAIRS" \
    --pairs_val "$VAL_PAIRS" \
    --weights_jsonl_train "$TRAIN_WEIGHTS" \
    --weights_jsonl_val "$VAL_WEIGHTS" \
    --base_ckpt "$BASE_CKPT" \
    --analyzer_ckpt "$ANALYZER_CKPT" \
    --epochs 1 \
    --max_samples 16 \
    --batch_size 4 \
    --lr 1e-4 \
    --cond_dropout 0.2 \
    --usage_loss_weight 0.1 \
    --noise_map_loss_weight 0.2 \
    --ortho_weight 0.05 \
    --sparsity_weight 0.01 \
    --alpha 0.5 \
    --gate_floor 0.3 \
    --basis_init_std 0.1 \
    --symbolic_strength 0.2 \
    --grad_loss_weight 0.0 \
    --base_delta_weight 0.2 \
    --base_delta_margin 0.001 \
    --output_dir "checkpoints/adaptive_nafnet_full_smoke" \
    --seed 42

echo ""
echo "================================================================================"
echo "Expected Results:"
echo "  - Δbase (delta_base) should be > 0.001 (not ~0)"
echo "  - GainPSNR should be measurable (not 0 or negative)"
echo "  - Map entropy should be reasonable (~1.0-1.3)"
echo "================================================================================"
