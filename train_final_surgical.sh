#!/usr/bin/env bash
# FINAL SURGICAL REFINEMENT CONFIGURATION
# Strategy: Context-Aware Heterogeneous Ensemble with Supervised Residuals
# Goal: SURPASS 30 dB Base Model

cd /home/kumwilai/OCT

echo "================================================================================"
echo "FINAL TRAINING: Context-Aware Surgical Refinement"
echo "================================================================================"
echo ""
echo "Key Capabilities enabled:"
echo "  1. Context Injection: Heads see [Amplified Residual, Base Anatomy]"
echo "  2. Swin Transformer: Global Attention Specialist for Speckle"
echo "  3. Surgical Supervision: Per-head loss when noise type is dominant"
echo "  4. Precision Scaling: 20x Residual Amplification"
echo "  5. Perceptual Focus: Gradient Loss (0.1) for texture preservation"
echo "================================================================================"
echo ""

# Configuration:
# - head_quality_weight 20.0: Strong force on the new residual supervision
# - residual_scale 20.0: High magnification for subtle signals
# - use_swin_speckle: Activate the Swin Transformer Head
# - epochs 100: Sufficient time for attention maps to converge

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --weights_jsonl_train weights_duke_analysis_maps_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_maps_val.jsonl \
  --max_samples 1000 \
  --val_samples 100 \
  --batch_size 4 \
  --epochs 100 \
  --early_stopping_patience 15 \
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
  --residual_blend_init 0.60 \
  --residual_head_width 64 \
  --use_swin_speckle \
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
  --use_region_weights \
  --region_min_band_frac 0.15 \
  --region_smooth_ksize 9 \
  --region_strength_mode residual \
  --log_region_psnr \
  --log_roi_psnr \
  --roi_center_frac 0.4 \
  --metrics_json outputs/duke_metrics_final_surgical.jsonl \
  --lambda_interp_start 0.001 \
  --lambda_interp_end 0.0005 \
  --lambda_interp_schedule cosine \
  --routing_loss_weight 0.05 \
  --param_reg_weight 0.03 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --head_quality_weight 20.0 \
  --head_diversity_weight 0.3 \
  --head_consistency_weight 0.01 \
  --grad_loss_weight 0.1 \
  --use_head_conditioning \
  --conditioner_hidden 16 \
  --residual_scale 20.0 \
  --seed 42

echo ""
echo "================================================================================"
echo "Training complete!"
echo "Expected: Overall PSNR > 30.2 dB"
echo "================================================================================"
