#!/usr/bin/env bash
# Test Stage 2 fine-tuning with unfrozen base

cd /home/kumwilai/OCT

echo "Testing Stage 2: Fine-Tuning with Unfrozen Base"
echo "================================================"

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --weights_jsonl_train weights_duke_analysis_maps_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_maps_val.jsonl \
  --max_samples 100 \
  --val_samples 20 \
  --batch_size 2 \
  --epochs 2 \
  --early_stopping_patience 10 \
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
  --metrics_json outputs/test_finetune_unfrozen.jsonl \
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
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth \
  --seed 42

echo ""
echo "Test complete! Check for:"
echo "1. Base NAFNet learning (lr=1e-6)"
echo "2. Maintained head diversity"
echo "3. Small PSNR improvement over frozen base"
