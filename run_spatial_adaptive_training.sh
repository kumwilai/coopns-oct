#!/bin/bash
set -euo pipefail

# ============================================================================
# Spatially-Adaptive NSND Training (TMI NOVELTY)
# ============================================================================
# Novel Contribution: Per-pixel noise type estimation and spatially-adaptive
# denoising, rather than global uniform blending.
#
# Expected Performance:
#   - PSNR: 29.5-30.5 dB (matches NAFNet baseline)
#   - Top-1: 74-76% (strong interpretability)
#   - Spatial maps: Per-pixel noise attribution
# ============================================================================

MAX_SAMPLES=2000
VAL_SAMPLES=400
SEED=0

echo "========================================================================"
echo "SPATIALLY-ADAPTIVE NSND TRAINING"
echo "Novel: Per-pixel noise type estimation + adaptive denoising"
echo "========================================================================"
echo ""
echo "Training Configuration:"
echo "  - Samples: $MAX_SAMPLES train, $VAL_SAMPLES val"
echo "  - Epochs: 40 (with early stopping)"
echo "  - Spatial weights: ENABLED ✓"
echo "  - Base NAFNet: Pre-trained (width=64)"
echo "  - Adapter blend: 0.35 (higher for spatial refinement)"
echo ""
echo "Expected Results:"
echo "  - PSNR: ~29.5-30.5 dB"
echo "  - SSIM: ~0.78-0.80"
echo "  - Top-1: ~74-76%"
echo "  - Spatial maps: Per-pixel noise attribution"
echo "========================================================================"
echo ""

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples "$MAX_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 40 \
  --early_stopping_patience 10 \
  --lr 1e-3 \
  --analyzer_lr 5e-4 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_trunk_width 32 \
  --shared_adapter_channels 96 \
  --shared_adapter_hidden 64 \
  --joint_expert_channels 96 \
  --residual_blend_init 0.35 \
  --use_joint_signal_expert \
  --joint_mix_init 0.05 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --shared_residual \
  --use_spatial_weights \
  --spatial_feature_channels 64 \
  --spatial_hidden_channels 32 \
  --lambda_warmup_epochs 5 \
  --lambda_interp_start 0.08 \
  --lambda_interp_end 0.05 \
  --lambda_interp_schedule cosine \
  --speckle_cycle_weight 0.0 \
  --noise_cycle_weight 0.0 \
  --param_reg_weight 0.0 \
  --composition_loss_weight 0.0 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --seed "$SEED"

echo ""
echo "========================================================================"
echo "Training Complete!"
echo "Model saved to: checkpoints/multitask_hybrid_nsnd_lambda0p08to0p05_cosine_best.pth"
echo ""
echo "Next Steps:"
echo "1. Visualize spatial weight maps:"
echo "   python run_interpretability_analysis.py --checkpoint checkpoints/multitask_hybrid_nsnd_lambda0p08to0p05_cosine_best.pth"
echo ""
echo "2. Compare against baseline (global weights):"
echo "   - This model: Per-pixel adaptive denoising"
echo "   - Baseline: Global uniform blending"
echo ""
echo "3. Generate TMI figures showing:"
echo "   - Spatial noise attribution maps"
echo "   - Per-pixel uncertainty estimation"
echo "   - Heterogeneous noise handling"
echo "========================================================================"
