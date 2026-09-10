#!/bin/bash
# Train all SOTA models on universal noise with stability improvements

echo "Training DRUNet, NAFNet, and SwinIR on Universal Noise"
echo "======================================================="

# DRUNet
echo ""
echo "1/3 Training DRUNet..."
python -c "
import sys
sys.path.append('sota')
from train_eval import train_and_eval

train_and_eval(
    model_name='drunet',
    pairs_file='train_pairs_universal.txt',
    size=64,
    out_dir='outputs/sota_universal/drunet',
    epochs=40
)
" 2>&1 | tee outputs/drunet_universal_train.log

# NAFNet (with our fixed version)
echo ""
echo "2/3 Training NAFNet..."
python train_nafnet_monitored.py \
  --train_pairs train_pairs_universal.txt \
  --val_pairs val_pairs_universal.txt \
  --out_dir outputs/sota_universal/nafnet \
  --epochs 40 \
  --batch_size 4 \
  --lr 5e-4 \
  --size 64 \
  --grad_w 0.05 \
  --width 48 \
  --middle_blk_num 2 \
  2>&1 | tee outputs/nafnet_universal_train.log

# SwinIR
echo ""
echo "3/3 Training SwinIR..."
python -c "
import sys
sys.path.append('sota')
from train_eval import train_and_eval

train_and_eval(
    model_name='swinir',
    pairs_file='train_pairs_universal.txt',
    size=64,
    out_dir='outputs/sota_universal/swinir',
    epochs=40
)
" 2>&1 | tee outputs/swinir_universal_train.log

echo ""
echo "======================================================="
echo "All training complete!"
echo "Check results in outputs/sota_universal/"
