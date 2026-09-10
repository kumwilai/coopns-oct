#!/bin/bash
set -euo pipefail

MAX_SAMPLES=2000
VAL_SAMPLES=400
SEED=0

echo "=========================================="
echo "STAGE 1: PSNR-Focused Adaptation (20 epochs)"
echo "=========================================="

# Stage 1: PSNR-focused adaptation (same architecture as Stage 2)
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples "$MAX_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 20 \
  --early_stopping_patience 0 \
  --lr 3e-3 \
  --analyzer_lr 1e-3 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_trunk_width 24 \
  --shared_adapter_channels 64 \
  --shared_adapter_hidden 48 \
  --joint_expert_channels 64 \
  --residual_blend_init 0.1 \
  --use_joint_signal_expert \
  --joint_mix_init 0.1 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --shared_residual \
  --lambda_interp 0.0 \
  --freeze_analyzer_epochs 999 \
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
echo "STAGE 2: Add Interpretability (15 epochs)"
echo "Loading from Stage 1 checkpoint..."
echo "=========================================="

# Stage 2: add interpretability
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples "$MAX_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 15 \
  --early_stopping_patience 0 \
  --lr 1e-3 \
  --analyzer_lr 1e-3 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_trunk_width 24 \
  --shared_adapter_channels 64 \
  --shared_adapter_hidden 48 \
  --joint_expert_channels 64 \
  --residual_blend_init 0.15 \
  --use_joint_signal_expert \
  --joint_mix_init 0.1 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --shared_residual \
  --lambda_interp_start 0.04 \
  --lambda_interp_end 0.02 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 0 \
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
echo "Stage 1 model: checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth"
echo "Stage 2 model: checkpoints/multitask_hybrid_nsnd_lambda0p04to0p02_cosine_best.pth"
echo "=========================================="
