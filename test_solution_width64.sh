#!/usr/bin/env bash
# REAL SOLUTION: Equal Capacity Heads (width=64, same as base)
#
# Problem: width=32 heads (26-28 dB) can't beat strong base (30 dB)
# Solution: width=64 heads (same capacity as base) can match and exceed base

cd /home/kumwilai/OCT

echo "================================================================================"
echo "REAL SOLUTION: Equal Capacity Heads (width=64)"
echo "================================================================================"
echo ""
echo "Problem diagnosed:"
echo "  - Strong base: 30.08 dB (very high)"
echo "  - Width=32 heads: 26-28 dB (too low)"
echo "  - Result: -1.38 dB adaptive gain"
echo ""
echo "Solution:"
echo "  - Head width: 64 (SAME as base, was 32)"
echo "  - Full training: 50 epochs (not 3)"
echo "  - Correct base checkpoint"
echo ""
echo "Expected: Heads with equal capacity CAN match base and specialize beyond it"
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
  --base_ckpt outputs/nafnet_analysis_maps_w64/nafnet_best.pth \
  --shared_residual \
  --shared_trunk_width 64 \
  --shared_adapter_channels 96 \
  --shared_adapter_hidden 64 \
  --residual_blend_init 0.60 \
  --residual_head_width 64 \
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
  --metrics_json outputs/duke_metrics_width64_solution.jsonl \
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
  --base_orthogonality_weight 0.0 \
  --seed 42

echo ""
echo "================================================================================"
echo "SOLUTION TEST COMPLETE!"
echo "================================================================================"
echo ""
echo "Key changes from failed attempt:"
echo "  - Head width: 32 → 64 (equal capacity to base)"
echo "  - Training: 3 → 50 epochs (full training)"
echo "  - Base checkpoint: CORRECT (30 dB base)"
echo ""
echo "Check if Overall PSNR > 30 dB (positive adaptive gain)"
echo "================================================================================"
