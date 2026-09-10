#!/bin/bash
# Strategy 2: Two-Stage Training (Recommended for Best Results)
# Stage 1: Pure denoising (epochs 1-30)
# Stage 2: Add interpretability (epochs 31-50)
# Target: PSNR ~29-30 + Interpretability ~75-80%

echo "=========================================="
echo "STAGE 1: Pure Denoising (18 epochs)"
echo "=========================================="

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
    --pairs_train train_pairs_duke_analysis.txt \
    --pairs_val val_pairs_duke_analysis.txt \
    --weights_jsonl_train weights_duke_analysis_train.jsonl \
    --weights_jsonl_val weights_duke_analysis_val.jsonl \
    --max_samples 500 \
    --val_samples 100 \
    --batch_size 4 \
    --epochs 18 \
    --early_stopping_patience 10 \
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
    --residual_blend_init 0.05 \
    --use_joint_signal_expert \
    --joint_mix_init 0.1 \
    --ns_use_neural_predicates \
    --ns_use_neural_weights \
    --shared_residual \
    --speckle_cycle_weight 0.0 \
    --noise_cycle_weight 0.0 \
    --param_reg_weight 0.0 \
    --composition_loss_weight 0.0 \
    --lambda_interp 0.0 \
    --use_log_domain_analyzer \
    --log_head_usage \
    --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
    --seed 0

echo ""
echo "=========================================="
echo "STAGE 2: Add Interpretability (12 epochs)"
echo "Loading from Stage 1 checkpoint..."
echo "=========================================="

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
    --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0.00_best.pth \
    --pairs_train train_pairs_duke_analysis.txt \
    --pairs_val val_pairs_duke_analysis.txt \
    --weights_jsonl_train weights_duke_analysis_train.jsonl \
    --weights_jsonl_val weights_duke_analysis_val.jsonl \
    --max_samples 500 \
    --val_samples 100 \
    --batch_size 4 \
    --epochs 12 \
    --early_stopping_patience 10 \
    --lr 1e-4 \
    --analyzer_lr 1e-4 \
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
    --residual_blend_init 0.05 \
    --use_joint_signal_expert \
    --joint_mix_init 0.1 \
    --ns_use_neural_predicates \
    --ns_use_neural_weights \
    --shared_residual \
    --speckle_cycle_weight 0.0 \
    --noise_cycle_weight 0.0 \
    --param_reg_weight 0.0 \
    --composition_loss_weight 0.0 \
    --lambda_interp 0.05 \
    --use_log_domain_analyzer \
    --log_head_usage \
    --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
    --seed 0

echo ""
echo "=========================================="
echo "Two-Stage Training Complete!"
echo "Stage 1 model: checkpoints/multitask_hybrid_nsnd_lambda0.00_best.pth"
echo "Stage 2 model: checkpoints/multitask_hybrid_nsnd_lambda0.05_best.pth"
echo "=========================================="
