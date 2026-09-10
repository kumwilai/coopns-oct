#!/usr/bin/env bash
# Robust Adaptive NAFNet Training
# Strategy B: Conditioned Base Model with Full Capacity

# Paths
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
ANALYZER_CKPT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"
TRAIN_PAIRS="train_pairs_duke_analysis_maps.txt"
VAL_PAIRS="val_pairs_duke_analysis_maps.txt"
TRAIN_WEIGHTS="weights_duke_analysis_maps_train.jsonl"
VAL_WEIGHTS="weights_duke_analysis_maps_val.jsonl"

# Configuration - FIXED FOR BETTER ADAPTATION
BATCH_SIZE=4
EPOCHS=10  # Reduced for quick verification
LR=1e-4
DROPOUT=0.0  # Disable dropout to force adaptation
USAGE_LOSS=0.5  # INCREASED: Force model to use conditioning
MAP_LOSS=0.0  # DISABLED: We have ground truth maps, no need to train predictor
ORTHO_LOSS=0.05
SPARSE_LOSS=0.01
ALPHA=2.0  # INCREASED: Stronger modulation
GATE_FLOOR=0.0  # REMOVED: Allow full adaptation even with low confidence
BASIS_INIT_STD=0.1
SYMBOLIC_STRENGTH=0.0  # DISABLED: Don't modify ground truth
GRAD_LOSS=0.1
LOG_REGION="--log_region_psnr"
LOG_ROI="--log_roi_psnr"
BASE_DELTA_WEIGHT=0.5  # INCREASED: Enforce difference from base
BASE_DELTA_MARGIN=0.01  # INCREASED: Stronger enforcement
MAP_ONLY_EPOCHS=0  # DISABLED: Skip map training, use ground truth directly
MAP_ONLY_LOSS=0.0

echo "================================================================================"
echo "Starting Training: Robust Adaptive NAFNet-64"
echo "--------------------------------------------------------------------------------"
echo "Base Model: $BASE_CKPT"
echo "Analyzer:   $ANALYZER_CKPT"
echo "Strategy:   FiLM Modulation + Confidence Gating + Usage Loss"
echo "================================================================================"

python -u nsnd_oct/scripts/train_adaptive_nafnet_full.py \
    --pairs_train "$TRAIN_PAIRS" \
    --pairs_val "$VAL_PAIRS" \
    --weights_jsonl_train "$TRAIN_WEIGHTS" \
    --weights_jsonl_val "$VAL_WEIGHTS" \
    --base_ckpt "$BASE_CKPT" \
    --analyzer_ckpt "$ANALYZER_CKPT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --cond_dropout $DROPOUT \
    --usage_loss_weight $USAGE_LOSS \
    --noise_map_loss_weight $MAP_LOSS \
    --ortho_weight $ORTHO_LOSS \
    --sparsity_weight $SPARSE_LOSS \
    --alpha $ALPHA \
    --gate_floor $GATE_FLOOR \
    --basis_init_std $BASIS_INIT_STD \
    --symbolic_strength $SYMBOLIC_STRENGTH \
    --grad_loss_weight $GRAD_LOSS \
    --base_delta_weight $BASE_DELTA_WEIGHT \
    --base_delta_margin $BASE_DELTA_MARGIN \
    --map_only_epochs $MAP_ONLY_EPOCHS \
    --map_only_loss_weight $MAP_ONLY_LOSS \
    $LOG_REGION \
    $LOG_ROI \
    --output_dir "checkpoints/adaptive_nafnet_full"
