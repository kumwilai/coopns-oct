#!/bin/bash
# Quick Test - Verify N2V Fix Works (5 epochs)
# Use this to confirm the fix works before full training

echo "========================================================================"
echo "QUICK TEST - FIXED N2V MASKING"
echo "Running 5 meta epochs + 20 finetune epochs to verify fix"
echo "========================================================================"
echo ""

python -u adaptive_oct_denoise.py \
  --clean_root meta_clean/ \
  --paired_list train_pairs_universal.txt \
  --val_paired_list val_pairs_universal.txt \
  --output_dir checkpoints/casa_n2v_quicktest \
  --backbone noise2void \
  --adapter casa \
  --base_channels 48 \
  --n2v_mask_ratio 0.12 \
  --n2v_box_size 5 \
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
echo "QUICK TEST COMPLETE"
echo ""
echo "Expected results after 20 epochs:"
echo "  - PSNR: 28-30 dB (not fully trained, but much better than 26 dB)"
echo "  - Should show steady improvement, not plateauing"
echo ""
echo "If PSNR reaches 28+ dB → Fix works! Run full training."
echo "If still stuck at 26 dB → Report back for further diagnosis."
echo "========================================================================"
