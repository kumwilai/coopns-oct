#!/bin/bash

# =============================================================================
# AMeta-FD Fair Comparison Script
# =============================================================================
# This script ensures fair comparison with other baseline methods by using:
# - Same training data (clean images from OCT2017)
# - Same validation set (universal balanced validation pairs)
# - Same image size (64x64)
# - Same total training budget (~50 epochs equivalent)
# - Same random seed (42)
# =============================================================================

# Check if data exists
if [ ! -d "data/OCT2017/train" ]; then
    echo "Error: Training data not found at data/OCT2017/train"
    echo "Please ensure OCT2017 dataset is properly set up"
    exit 1
fi

if [ ! -f "val_pairs_universal.txt" ]; then
    echo "Error: Validation pairs not found: val_pairs_universal.txt"
    echo "Please run the data preparation script first"
    exit 1
fi

echo "================================================================================"
echo "AMeta-FD: Adversarial Meta-Learning for Few-shot OCT Despeckling"
echo "Fair Comparison with Baseline Methods"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Clean training images: data/OCT2017/train"
echo "  - Validation pairs: val_pairs_universal.txt"
echo "  - Image size: 64x64"
echo "  - Meta-epochs: 50 (equivalent to ~50 standard epochs)"
echo "  - Tasks per batch: 4"
echo "  - Support/Query: 5/10 images per task"
echo "  - Random seed: 42"
echo ""
echo "================================================================================"
echo ""

# Training
python ameta_fd.py \
    --clean_root data/OCT2017/train \
    --val_pairs val_pairs_universal.txt \
    --output_dir outputs/ameta_fd \
    --num_meta_epochs 50 \
    --tasks_per_batch 4 \
    --n_support 5 \
    --n_query 10 \
    --inner_steps 5 \
    --inner_lr 1e-3 \
    --meta_lr_gen 1e-4 \
    --meta_lr_disc 1e-4 \
    --lambda_adv 0.1 \
    --image_size 64 \
    --seed 42

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"
echo ""
echo "Model saved to: outputs/ameta_fd/ameta_fd_final.pth"
echo ""
echo "To evaluate without test-time adaptation:"
echo "  python ameta_fd.py --eval_only --checkpoint outputs/ameta_fd/ameta_fd_final.pth --val_pairs val_pairs_universal.txt"
echo ""
echo "To evaluate WITH few-shot test-time adaptation (10 steps):"
echo "  python ameta_fd.py --eval_only --checkpoint outputs/ameta_fd/ameta_fd_final.pth --val_pairs val_pairs_universal.txt --adapt_steps 10"
echo ""
echo "================================================================================"
