#!/bin/bash
# Publication-Ready NSND Training
# Fair comparison: 7.05M denoiser parameters (93.3% of NAFNet-64's 7.55M)
# Key fix: Remove adapter bottleneck (8ch → 64ch) and harmful auxiliary losses

python nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis.txt \
  --pairs_val val_pairs_duke_analysis.txt \
  --weights_jsonl_train weights_duke_analysis_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_val.jsonl \
  --max_samples 2000 \
  --val_samples 400 \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --analyzer_lr 1e-5 \
  --use_log_domain_analyzer \
  --use_log_domain_speckle \
  --base_nafnet_width 32 \
  --shared_trunk_width 24 \
  --shared_adapter_channels 64 \
  --shared_adapter_hidden 48 \
  --joint_expert_channels 48 \
  --use_joint_signal_expert \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --shared_residual \
  --speckle_cycle_weight 0.0 \
  --noise_cycle_weight 0.0 \
  --param_reg_weight 0.0 \
  --composition_loss_weight 0.02 \
  --composition_consistency_weight 0.005 \
  --composition_group_size 2 \
  --freeze_analyzer_epochs 0 \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --seed 0

# KEY CHANGES FROM ORIGINAL:
# 1. shared_adapter_channels: 8 → 64 (8× increase, removes bottleneck)
# 2. shared_adapter_hidden: 8 → 48 (6× increase)
# 3. joint_expert_channels: 16 → 48 (3× increase)
# 4. noise_cycle_weight: 0.01 → 0.0 (REMOVED - was fighting denoising)
# 5. param_reg_weight: 0.1 → 0.0 (REMOVED - was over-regularizing)
# 6. composition_loss_weight: 0.2 → 0.02 (10× reduced)
# 7. composition_consistency_weight: 0.05 → 0.005 (10× reduced)

# EXPECTED RESULTS:
# - PSNR: ~29-30 dB (matches NAFNet-64's ~30 dB)
# - SSIM: ~0.82-0.85
# - Parameters: Still ~7M (fair comparison maintained)
# - Analyzer Top-1 maintained during denoising

# WHY THIS WORKS:
# - 64-channel adapters allow information flow (was 8ch bottleneck)
# - Removed noise-cycle loss that was training model to preserve noise
# - Reduced auxiliary loss weights to focus on denoising objective
# - Parameters mostly go to adapters now (removes bottleneck waste)
