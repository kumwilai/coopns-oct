#!/usr/bin/env bash
# Train Base NAFNet on Analysis Maps Dataset
# This creates a strong base (30-31 dB) for subsequent adaptive head training

cd /home/kumwilai/OCT

echo "================================================================================"
echo "Training Base NAFNet on Analysis Maps Dataset"
echo "================================================================================"
echo ""
echo "Goal: Train strong base NAFNet specifically for analysis maps noise"
echo ""
echo "Configuration:"
echo "  - Dataset: Duke analysis maps (synthesized Dirichlet noise)"
echo "  - Width: 64"
echo "  - Architecture: [2,2,2] encoder/decoder blocks"
echo "  - Training samples: 1000"
echo "  - Validation samples: 100"
echo "  - Batch size: 8"
echo "  - Epochs: 100 (with early stopping)"
echo ""
echo "Expected results:"
echo "  - Validation PSNR: 30-31 dB (vs 27.09 dB with old base)"
echo "  - Training time: 2-3 hours"
echo "  - Subsequent adaptive gain: 2-3 dB → Overall 32-34 dB"
echo "================================================================================"
echo ""

python -u nsnd_oct/scripts/train_nafnet_on_analysis_maps.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --max_samples 1000 \
  --val_samples 100 \
  --crop_size 64 \
  --width 64 \
  --enc_blk_nums 2 2 2 \
  --dec_blk_nums 2 2 2 \
  --middle_blk_num 2 \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --early_stopping_patience 15 \
  --output_dir outputs/nafnet_analysis_maps_w64 \
  --save_every 10

echo ""
echo "================================================================================"
echo "Base NAFNet training complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check validation PSNR (should be 30-31 dB)"
echo "  2. Run frozen head training: bash train_frozen_with_strong_base.sh"
echo "  3. Expected final result: 32-34 dB overall PSNR"
echo ""
echo "Checkpoint saved to: outputs/nafnet_analysis_maps_w64/nafnet_best.pth"
echo "================================================================================"
