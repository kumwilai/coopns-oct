#!/usr/bin/env bash
# Quick test: Train base NAFNet for 5 epochs to validate setup

cd /home/kumwilai/OCT

echo "Testing Base NAFNet Training (5 epochs)"
echo "========================================"
echo ""
echo "This is a quick test to validate:"
echo "  1. Data loading works correctly"
echo "  2. Model trains without errors"
echo "  3. PSNR improves during training"
echo ""
echo "Expected: PSNR should reach ~28-29 dB after 5 epochs"
echo ""

python -u nsnd_oct/scripts/train_nafnet_on_analysis_maps.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --max_samples 200 \
  --val_samples 50 \
  --crop_size 64 \
  --width 64 \
  --enc_blk_nums 2 2 2 \
  --dec_blk_nums 2 2 2 \
  --middle_blk_num 2 \
  --epochs 5 \
  --batch_size 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --early_stopping_patience 10 \
  --output_dir outputs/test_nafnet_base \
  --save_every 5

echo ""
echo "Test complete! Check:"
echo "  1. Final validation PSNR (should be ~28-29 dB)"
echo "  2. No errors during training"
echo "  3. If successful, run full training: bash train_nafnet_base_analysis_maps.sh"
