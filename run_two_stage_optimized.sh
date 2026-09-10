#!/bin/bash
set -euo pipefail

# Optimized for SOTA performance (30 dB) + TMI novelty
MAX_SAMPLES=2000
VAL_SAMPLES=400
SEED=0

echo "=========================================="
echo "STAGE 1: PSNR Maximization (30 epochs)"
echo "Goal: Match NAFNet baseline ~30 dB"
echo "=========================================="

# Stage 1: Pure denoising with frozen base NAFNet
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples "$MAX_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 30 \
  --early_stopping_patience 8 \
  --lr 2e-3 \
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
  --residual_blend_init 0.3 \
  --use_joint_signal_expert \
  --joint_mix_init 0.05 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --shared_residual \
  --lambda_interp 0.0 \
  --freeze_analyzer_epochs 15 \
  --speckle_cycle_weight 0.0 \
  --noise_cycle_weight 0.0 \
  --mix_gate_reg_weight 0 \
  --consistency_weight 0 \
  --logic_reg_weight 0 \
  --composition_loss_weight 0.0 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --seed "$SEED"

echo ""
echo "=========================================="
echo "STAGE 2: Add Interpretability (20 epochs)"
echo "Goal: 73-75% Top-1 while maintaining PSNR"
echo "=========================================="

# Stage 2: Add interpretability with careful tuning
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples "$MAX_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 20 \
  --early_stopping_patience 8 \
  --lr 5e-4 \
  --analyzer_lr 1e-3 \
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
  --residual_blend_init 0.3 \
  --use_joint_signal_expert \
  --joint_mix_init 0.05 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --shared_residual \
  --lambda_interp_start 0.08 \
  --lambda_interp_end 0.05 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 3 \
  --freeze_analyzer_epochs 0 \
  --speckle_cycle_weight 0.0 \
  --noise_cycle_weight 0.0 \
  --mix_gate_reg_weight 0 \
  --consistency_weight 0 \
  --logic_reg_weight 0 \
  --composition_loss_weight 0.0 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --seed "$SEED"

echo ""
echo "=========================================="
echo "Two-Stage Training Complete!"
echo "Stage 1: checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth"
echo "Stage 2: checkpoints/multitask_hybrid_nsnd_lambda0p08to0p05_cosine_best.pth"
echo ""
echo "Expected Performance:"
echo "  PSNR: 29.5-30.5 dB (matches NAFNet)"
echo "  SSIM: 0.78-0.80"
echo "  Top-1: 73-76% (strong interpretability)"
echo "=========================================="
