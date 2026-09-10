#!/usr/bin/env bash
# Stage 2: Fine-tuning with unfrozen base NAFNet
# Run this AFTER train_frozen_base_full.sh completes

cd /home/kumwilai/OCT

echo "================================================================================"
echo "STAGE 2: Fine-Tuning with Unfrozen Base NAFNet"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Base NAFNet: UNFROZEN (lr=1e-6, very conservative)"
echo "  - Head LR: 1e-5 (10x lower than Stage 1)"
echo "  - Base orthogonality: 0.1 (reduced - allow collaboration)"
echo "  - Quality weight: 1.5 (maintained)"
echo "  - Diversity weight: 0.3 (reduced - specialization already learned)"
echo "  - Epochs: 10 (short fine-tuning)"
echo ""
echo "Expected results:"
echo "  - Additional gain: +0.2-0.5 dB"
echo "  - Final PSNR: 32.5-33.8 dB"
echo "  - Preserved head specialization"
echo "================================================================================"
echo ""

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --weights_jsonl_train weights_duke_analysis_maps_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_maps_val.jsonl \
  --max_samples 1000 \
  --val_samples 100 \
  --batch_size 4 \
  --epochs 10 \
  --early_stopping_patience 5 \
  --lr 1e-5 \
  --analyzer_lr 5e-6 \
  --base_nafnet_lr 1e-6 \
  --freeze_analyzer_epochs 0 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_residual \
  --shared_trunk_width 32 \
  --shared_adapter_channels 96 \
  --shared_adapter_hidden 64 \
  --residual_blend_init 0.60 \
  --use_joint_signal_expert \
  --joint_expert_channels 96 \
  --joint_mix_init 0.15 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --use_spatial_weights \
  --spatial_feature_channels 64 \
  --spatial_hidden_channels 32 \
  --noise_map_loss_weight 0.15 \
  --noise_map_loss_weights 3,1,3,1 \
  --noise_map_stage_epochs 0 \
  --use_region_weights \
  --region_min_band_frac 0.15 \
  --region_smooth_ksize 9 \
  --region_strength_mode residual \
  --log_region_psnr \
  --log_roi_psnr \
  --roi_center_frac 0.4 \
  --metrics_json outputs/duke_metrics_stage2_finetune.jsonl \
  --lambda_interp_start 0.003 \
  --lambda_interp_end 0.001 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 0 \
  --routing_loss_weight 0.03 \
  --param_reg_weight 0.02 \
  --param_reg_warmup_epochs 0 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --mix_gate_reg_weight 0 \
  --consistency_weight 0 \
  --logic_reg_weight 0 \
  --composition_loss_weight 0 \
  --composition_consistency_weight 0 \
  --residual_consistency_weight 0 \
  --noise_cycle_weight 0 \
  --speckle_cycle_weight 0 \
  --head_quality_weight 1.5 \
  --head_diversity_weight 0.3 \
  --head_consistency_weight 0.05 \
  --base_orthogonality_weight 0.1 \
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth \
  --seed 42

echo ""
echo "================================================================================"
echo "Fine-tuning complete!"
echo "================================================================================"
echo ""
echo "Compare results:"
echo "  Stage 1 (Frozen):    checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth"
echo "  Stage 2 (Finetuned): checkpoints/multitask_hybrid_nsnd_lambda0p003to0p001_cosine_best.pth"
echo ""
