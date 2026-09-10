#!/usr/bin/env bash
# FIXED: Reduced base orthogonality weight to prevent heads from making outputs worse

cd /home/kumwilai/OCT

echo "================================================================================"
echo "FIXED CONFIGURATION: Balanced Loss Weights"
echo "================================================================================"
echo ""
echo "Changes from previous run:"
echo "  - Base orthogonality: 0.3 → 0.05 (was too aggressive!)"
echo "  - Head quality: 2.0 → 5.0 (stronger supervision)"
echo "  - Head diversity: 0.5 → 0.3 (reduced)"
echo ""
echo "Why these changes:"
echo "  - High orthogonality (0.3) forced heads to diverge by making outputs WORSE"
echo "  - Shot/Banding heads were producing 18-21 dB (worse than 22.38 dB noisy!)"
echo "  - Now quality loss (5.0) dominates over orthogonality (0.05)"
echo "  - Heads will learn to be BETTER first, DIFFERENT second"
echo ""
echo "Expected results:"
echo "  - All heads > base PSNR (27 dB)"
echo "  - Shot/Banding heads: 28-29 dB (not 18-21 dB!)"
echo "  - Adaptive gain: 1.5-2.5 dB"
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
  --epochs 50 \
  --early_stopping_patience 10 \
  --lr 5e-4 \
  --analyzer_lr 2e-4 \
  --base_nafnet_lr 0.0 \
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
  --metrics_json outputs/duke_metrics_frozen_base_fixed.jsonl \
  --lambda_interp_start 0.008 \
  --lambda_interp_end 0.003 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 0 \
  --routing_loss_weight 0.05 \
  --param_reg_weight 0.03 \
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
  --head_quality_weight 5.0 \
  --head_diversity_weight 0.3 \
  --head_consistency_weight 0.01 \
  --base_orthogonality_weight 0.05 \
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth \
  --seed 42

echo ""
echo "================================================================================"
echo "Training complete! Check that:"
echo "  1. All heads > 27 dB (not 18-21 dB)"
echo "  2. Shot and Banding heads are GOOD"
echo "  3. Quality loss dominates orthogonality loss"
echo "================================================================================"
