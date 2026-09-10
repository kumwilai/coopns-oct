#!/bin/bash
# Test 1: Lower Learning Rate with Constant Schedule
# Goal: Stabilize training at 26.4 dB without degradation, beat BM3D (~27 dB)
# Hypothesis: Current LR too high causing overshoot after warmup

echo "========================================================================"
echo "TEST 1: LOWER LEARNING RATE + CONSTANT SCHEDULE"
echo "========================================================================"
echo ""
echo "Problem: Training peaks at 26.40 dB (epoch 2), then degrades to 25.76 dB"
echo "Cause: Learning rate too aggressive after warmup ends"
echo ""
echo "Changes from previous run:"
echo "  - Adapter LR: 5e-4 → 1e-4 (5× lower)"
echo "  - Backbone LR: 1e-4 → 5e-5 (2× lower)"
echo "  - Scheduler: cosine → plateau (keeps LR constant unless plateau)"
echo "  - Warmup: 8 epochs → 0 epochs (no warmup overshoot)"
echo "  - Patience: 10 → 20 epochs (more patient)"
echo ""
echo "Expected: Stable 26.4-27 dB without degradation"
echo "Target: Beat BM3D baseline (~27 dB)"
echo "========================================================================"
echo ""

python -u adaptive_oct_denoise.py \
  --clean_root meta_clean/ \
  --paired_list train_pairs_universal.txt \
  --val_paired_list val_pairs_universal.txt \
  --output_dir checkpoints/casa_n2v_test1_lower_lr \
  --backbone noise2void \
  --adapter casa \
  --base_channels 48 \
  --n2v_mask_ratio 0.12 \
  --n2v_box_size 5 \
  --residual_mode \
  --num_meta_epochs 5 \
  --finetune_epochs 50 \
  --batch_size 4 \
  --finetune_lr_adapter 1e-4 \
  --finetune_lr_backbone 5e-5 \
  --scheduler_type plateau \
  --warmup_epochs 0 \
  --early_stopping_patience 20 \
  --ema \
  --amp \
  --seed 42

echo ""
echo "========================================================================"
echo "TEST 1 COMPLETE"
echo "========================================================================"
echo ""
echo "Success criteria:"
echo "  ✅ PSNR maintains 26.4+ dB without degradation"
echo "  ✅ PSNR reaches 27+ dB (beats BM3D)"
echo "  ✅ Training curve stable (no overshoot)"
echo ""
echo "If successful → Run full training with these settings for 100 epochs"
echo "If still degrading → Try Test 2 (skip meta-learning)"
echo "========================================================================"
