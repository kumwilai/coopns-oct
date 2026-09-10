#!/usr/bin/env bash
# Train adaptive heads with STRONG frozen base (30-31 dB)
# Use this AFTER training base NAFNet on analysis maps

cd /home/kumwilai/OCT

echo "================================================================================"
echo "Training Adaptive Heads with STRONG Frozen Base"
echo "================================================================================"
echo ""
echo "Prerequisites:"
echo "  - Base NAFNet trained on analysis maps (30-31 dB)"
echo "  - Checkpoint: outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
echo ""
echo "Configuration:"
echo "  - Base NAFNet: FROZEN (lr=0.0) at 30-31 dB"
echo "  - Head quality weight: 5.0 (strong supervision)"
echo "  - Head diversity weight: 0.3"
echo "  - Base orthogonality weight: 0.1 (balanced)"
echo ""
echo "Expected results:"
echo "  - Base: 30-31 dB (frozen)"
echo "  - Adaptive gain: 2-3 dB"
echo "  - Overall PSNR: 32-34 dB ⭐"
echo "  - All 4 heads functional (>30 dB each)"
echo "================================================================================"
echo ""

# Check if base checkpoint exists
BASE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
if [ ! -f "$BASE_CKPT" ]; then
    echo "❌ ERROR: Base NAFNet checkpoint not found!"
    echo "   Expected: $BASE_CKPT"
    echo ""
    echo "Please train base NAFNet first:"
    echo "   bash train_nafnet_base_analysis_maps.sh"
    echo ""
    exit 1
fi

echo "✓ Found base checkpoint: $BASE_CKPT"
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
  --metrics_json outputs/duke_metrics_strong_base.jsonl \
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
  --base_orthogonality_weight 0.1 \
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth \
  --seed 42

echo ""
echo "================================================================================"
echo "Training complete!"
echo "================================================================================"
echo ""
echo "Performance summary:"
echo "  - Check overall PSNR (should be 32-34 dB)"
echo "  - Adaptive gain should be 2-3 dB"
echo "  - All 4 heads should be GOOD"
echo ""
echo "Checkpoint: checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth"
echo "================================================================================"
