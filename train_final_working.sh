#!/usr/bin/env bash
# FINAL WORKING CONFIGURATION
# Problem: Heads too small to beat strong base (30 dB)
# Solution: Equal capacity heads (width=64) + removed orthogonality constraint

cd /home/kumwilai/OCT

echo "================================================================================"
echo "FINAL TRAINING: Equal Capacity Heads (width=64)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Residual head width: 64 (SAME as base, equal capacity)"
echo "  - Base orthogonality: 0.0 (no divergence constraint)"
echo "  - Head quality weight: 5.0 (strong supervision)"
echo "  - Head diversity weight: 0.3 (moderate)"
echo "  - Base NAFNet: FROZEN at 30 dB (lr=0.0)"
echo "  - Correct base checkpoint: nafnet_analysis_maps_w64"
echo ""
echo "Strategy:"
echo "  - Heads with equal capacity CAN match base (30 dB)"
echo "  - Specialization allows improvement beyond base"
echo "  - Positive adaptive gain achievable"
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
  --metrics_json outputs/duke_metrics_final_width32.jsonl \
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
echo "Training complete!"
echo "================================================================================"
echo ""
echo "Key changes from original problem:"
echo "  ✓ Head capacity: width 16 → 64 (EQUAL to base)"
echo "  ✓ Base checkpoint: CORRECT (30 dB strong base)"
echo "  ✓ Base orthogonality: 0.0 (allows matching when beneficial)"
echo "  ✓ Quality loss: 5.0 (prevents degradation)"
echo "  ✓ Full 50 epoch training"
echo ""
echo "Expected: Overall PSNR > 30 dB (positive adaptive gain)"
echo "Checkpoint: checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth"
echo "================================================================================"
