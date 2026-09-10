#!/bin/bash

# AMeta-FD Training Script
# Train adversarial meta-learning model for OCT despeckling

CLEAN_ROOT="data/OCT2017/train"
VAL_PAIRS="val_pairs_universal.txt"
OUTPUT_DIR="outputs/ameta_fd"

echo "================================================================================"
echo "Training AMeta-FD: Adversarial Meta-Learning for OCT Despeckling"
echo "================================================================================"

python ameta_fd.py \
    --clean_root "$CLEAN_ROOT" \
    --val_pairs "$VAL_PAIRS" \
    --output_dir "$OUTPUT_DIR" \
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
echo "Training complete! Model saved in $OUTPUT_DIR"
