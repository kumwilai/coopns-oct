#!/bin/bash

# Fine-tune from existing meta checkpoint (skip meta-training)
python -u adaptive_oct_denoise.py \
    --resume_from_checkpoint checkpoints/casa_n2v_dilated3_coherence_nafnet/meta_adapter.pth \
    --paired_list train_pairs_universal.txt \
    --val_paired_list val_pairs_universal.txt \
    --output_dir checkpoints/casa_n2v_dilated3_coherence_nafnet_finetune \
    --strategy noise2void \
    --backbone nafnet \
    --adapter casa \
    --base_channels 48 \
    --resize_h 64 --resize_w 64 \
    --n2v_mask_ratio 0.20 \
    --n2v_blindspot_dilation 3 \
    --meta_strategy noise2void \
    --num_meta_epochs 0 \
    --finetune_epochs 100 \
    --batch_size 8 \
    --finetune_lr_adapter 3e-4 \
    --finetune_lr_backbone 5e-5 \
    --weight_decay 1e-4 \
    --scheduler_type cosine \
    --warmup_epochs 2 \
    --early_stopping_patience 5 \
    --validation_frequency 1 \
    --ema --ema_decay 0.999 \
    --amp \
    --lambda_grad 0.0 \
    --lambda_tv 0.0 \
    --lambda_multiscale 0.0 \
    --lambda_perceptual 0.0 \
    --lambda_coherence 0.01 \
    --seed 42
