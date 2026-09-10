#!/bin/bash

# SOTA Configuration: NAFNet (architecture) + Neighbor2Neighbor (strategy) + CASA + TTA
# Aiming to beat BM3D (> 27.5 dB)

python -u adaptive_oct_denoise.py \
    --clean_root meta_clean/ \
    --paired_list train_pairs_universal.txt \
    --val_paired_list val_pairs_universal.txt \
    --output_dir checkpoints/sota_n2n_nafnet_final \
    --backbone nafnet \
    --strategy neighbor2neighbor \
    --adapter casa \
    --base_channels 48 \
    --residual_mode \
    --resize_h 64 \
    --resize_w 64 \
    --meta_strategy noise2void \
    --num_meta_epochs 10 \
    --finetune_epochs 100 \
    --batch_size 8 \
    --finetune_lr_adapter 3e-4 \
    --finetune_lr_backbone 1e-4 \
    --weight_decay 1e-4 \
    --scheduler_type cosine \
    --warmup_epochs 5 \
    --early_stopping_patience 10 \
    --validation_frequency 1 \
    --ema \
    --ema_decay 0.999 \
    --amp \
    --lambda_coherence 0.1 \
    --lambda_perceptual 0.1 \
    --use_tta \
    --seed 42
