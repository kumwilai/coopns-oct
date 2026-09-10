#!/bin/bash
# Balanced Hybrid NSND - Best of Both Worlds
# Target: PSNR ~28-29 + Interpretability ~75%

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
    --pairs_train train_pairs_duke_analysis.txt \
    --pairs_val val_pairs_duke_analysis.txt \
    --weights_jsonl_train weights_duke_analysis_train.jsonl \
    --weights_jsonl_val weights_duke_analysis_val.jsonl \
    --max_samples 500 \
    --val_samples 100 \
    --batch_size 4 \
    --epochs 30 \
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
    --lambda_warmup_epochs 6 \
    --lambda_interp_start 0.08 \
    --lambda_interp_end 0.05 \
    --lambda_interp_schedule cosine \
    --use_log_domain_analyzer \
    --log_head_usage \
    --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
    --seed 0
