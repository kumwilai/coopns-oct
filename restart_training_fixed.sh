#!/bin/bash
# Restart CASA+N2V Training with Fixed Mask Ratio
# Bug Fix: N2V mask ratio now correctly 12% (was 4.1%)

echo "========================================================================"
echo "RESTARTING CASA+N2V WITH FIXED MASKING"
echo "Bug Fixed: Mask ratio 4.1% → 12.0%"
echo "Expected: PSNR should reach 30-32 dB (was stuck at 26 dB)"
echo "========================================================================"
echo ""

# Backup old checkpoint
if [ -d "checkpoints/casa_noise2void" ]; then
    echo "Backing up old checkpoint to checkpoints/casa_noise2void_old_buggy..."
    mv checkpoints/casa_noise2void checkpoints/casa_noise2void_old_buggy
    echo "✓ Backup complete"
    echo ""
fi

# Run training with fixed code
echo "Starting training with FIXED N2V masking..."
echo ""

python -u adaptive_oct_denoise.py \
  --clean_root meta_clean/ \
  --paired_list train_pairs_universal.txt \
  --val_paired_list val_pairs_universal.txt \
  --output_dir checkpoints/casa_noise2void \
  --backbone noise2void \
  --adapter casa \
  --base_channels 48 \
  --n2v_mask_ratio 0.12 \
  --n2v_box_size 5 \
  --num_meta_epochs 15 \
  --num_tasks_per_meta_batch 4 \
  --inner_steps 5 \
  --inner_lr 1e-4 \
  --meta_step_size 0.1 \
  --finetune_epochs 100 \
  --batch_size 4 \
  --finetune_lr_adapter 5e-4 \
  --finetune_lr_backbone 1e-4 \
  --weight_decay 1e-4 \
  --scheduler_type cosine \
  --warmup_epochs 8 \
  --early_stopping_patience 15 \
  --validation_frequency 1 \
  --gradient_accumulation_steps 2 \
  --ema \
  --ema_decay 0.999 \
  --amp \
  --grad_clip_norm 1.0 \
  --loss_type charbonnier \
  --lambda_grad 0.05 \
  --lambda_tv 1e-5 \
  --log_domain \
  --seed 42

echo ""
echo "========================================================================"
echo "Training complete! Check final PSNR above."
echo "Expected: 31-32 dB (improvement of +5-6 dB over buggy version)"
echo "========================================================================"
