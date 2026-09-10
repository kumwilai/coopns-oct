#!/bin/bash
# Quick Supervised Baseline Test
# This tests if the issue is N2V-specific or general

echo "========================================================================"
echo "QUICK SUPERVISED BASELINE TEST"
echo "Tests basic denoising without Noise2Void complexity"
echo "========================================================================"
echo ""
echo "Expected results:"
echo "  - If PSNR reaches 30+ dB: N2V implementation has issues"
echo "  - If PSNR stuck at 26 dB: Data or model architecture issue"
echo ""

python -u adaptive_oct_denoise.py \
  --paired_list train_pairs_universal.txt \
  --val_paired_list val_pairs_universal.txt \
  --output_dir checkpoints/supervised_baseline_test \
  --backbone unet \
  --adapter casa \
  --base_channels 64 \
  --finetune_epochs 30 \
  --batch_size 4 \
  --finetune_lr_adapter 5e-4 \
  --finetune_lr_backbone 1e-4 \
  --scheduler_type cosine \
  --warmup_epochs 5 \
  --early_stopping_patience 10 \
  --ema \
  --amp \
  --grad_clip_norm 1.0 \
  --seed 42

echo ""
echo "========================================================================"
echo "TEST COMPLETE"
echo "Check final PSNR above to diagnose the issue"
echo "========================================================================"
