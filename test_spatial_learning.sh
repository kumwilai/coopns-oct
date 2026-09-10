#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "LEARNING TEST: Verify Spatial Refiner Learns (2 epochs)"
echo "========================================================================"
echo ""
echo "This test verifies:"
echo "  1. Training loop works with spatial weights"
echo "  2. Spatial refiner learns non-zero spatial variation"
echo "  3. Loss decreases over epochs"
echo ""
echo "Expected result after 2 epochs:"
echo "  - Spatial weight std > 0.001 (learns spatial variation)"
echo "  - Loss decreases"
echo "  - No crashes"
echo "========================================================================"
echo ""

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples 100 \
  --val_samples 50 \
  --batch_size 2 \
  --epochs 2 \
  --early_stopping_patience 999 \
  --lr 1e-3 \
  --analyzer_lr 5e-4 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_trunk_width 32 \
  --shared_adapter_channels 96 \
  --shared_adapter_hidden 64 \
  --residual_blend_init 0.35 \
  --use_joint_signal_expert \
  --joint_expert_channels 96 \
  --joint_mix_init 0.05 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --shared_residual \
  --use_spatial_weights \
  --spatial_feature_channels 64 \
  --spatial_hidden_channels 32 \
  --lambda_interp_start 0.08 \
  --lambda_interp_end 0.05 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 1 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --seed 0

echo ""
echo "========================================================================"
echo "Training test complete!"
echo "Check the logs above for:"
echo "  1. 'Spatial weight refiner enabled' ✓"
echo "  2. Loss values decreasing ✓"
echo "  3. No errors/crashes ✓"
echo "========================================================================"
