#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "================================================================================"
echo "OPTIMIZED TRAINING FOR TMI PUBLICATION"
echo "================================================================================"
echo ""
echo "Key improvements:"
echo "  1. DISABLED noise map loss (was consuming 72% of optimization)"
echo "  2. REDUCED interpretability loss (0.02-0.05 → 0.005)"
echo "  3. INCREASED model capacity (width 64 → 80)"
echo "  4. LONGER training (80 epochs vs 40)"
echo "  5. BETTER learning rates (more stable)"
echo ""
echo "Expected results:"
echo "  - PSNR: 33.25 → 34.0+ dB (+0.75 dB improvement)"
echo "  - SSIM: 0.907 → 0.920+"
echo "  - Spatial weights will learn better (more epochs)"
echo ""
echo "================================================================================"
echo ""

# Single-phase training: Focus on denoising quality
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples 2000 \
  --val_samples 400 \
  --batch_size 4 \
  --epochs 80 \
  --early_stopping_patience 15 \
  --lr 3e-4 \
  --analyzer_lr 1e-4 \
  --freeze_analyzer_epochs 0 \
  --base_nafnet_type full \
  --base_nafnet_width 80 \
  --base_enc_blk_nums 2 2 2 2 \
  --base_dec_blk_nums 2 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_residual \
  --shared_trunk_width 48 \
  --shared_adapter_channels 128 \
  --shared_adapter_hidden 64 \
  --residual_blend_init 0.40 \
  --use_joint_signal_expert \
  --joint_expert_channels 128 \
  --joint_mix_init 0.08 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --use_spatial_weights \
  --spatial_feature_channels 96 \
  --spatial_hidden_channels 48 \
  --noise_map_loss_weight 0.0 \
  --use_region_weights \
  --region_min_band_frac 0.15 \
  --region_smooth_ksize 9 \
  --region_strength_mode residual \
  --log_region_psnr \
  --log_roi_psnr \
  --roi_center_frac 0.4 \
  --metrics_json outputs/duke_metrics_optimized.jsonl \
  --lambda_interp_start 0.005 \
  --lambda_interp_end 0.005 \
  --lambda_interp_schedule constant \
  --lambda_warmup_epochs 0 \
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
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p02_best.pth \
  --seed 42

echo ""
echo "================================================================================"
echo "Training complete! Check metrics in: outputs/duke_metrics_optimized.jsonl"
echo "================================================================================"
