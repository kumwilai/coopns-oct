#!/bin/bash
# Quick test to verify residual mode fixes the training issue
# Should reach 29-30 dB in just 20 epochs (vs 26 dB without)

echo "========================================================================"
echo "TESTING BUG #14 FIX: RESIDUAL MODE ACTIVATION"
echo "Expected: PSNR reaches 29-30 dB after 20 epochs"
echo "Previous: PSNR stuck at 26 dB with sigmoid activation"
echo "========================================================================"
echo ""

python -u adaptive_oct_denoise.py \
  --clean_root meta_clean/ \
  --paired_list train_pairs_universal.txt \
  --val_paired_list val_pairs_universal.txt \
  --output_dir checkpoints/casa_n2v_residual_test \
  --backbone noise2void \
  --adapter casa \
  --base_channels 48 \
  --n2v_mask_ratio 0.12 \
  --n2v_box_size 5 \
  --residual_mode \
  --num_meta_epochs 5 \
  --finetune_epochs 20 \
  --batch_size 4 \
  --finetune_lr_adapter 5e-4 \
  --finetune_lr_backbone 1e-4 \
  --ema \
  --amp \
  --seed 42

echo ""
echo "========================================================================"
echo "TEST COMPLETE"
echo ""
echo "If PSNR reaches 29-30 dB → BUG FIXED!"
echo "If still stuck at 26 dB → Additional issues remain"
echo "========================================================================"
