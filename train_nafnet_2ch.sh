#!/bin/bash
# Retrain NAFNet correctors with current 2-channel output head architecture
# Backbone: frozen NAFNet 7M (outputs/nafnet_pku37_w40/best_model.pth)
# Training: PKU37 train (1388 samples), val (173 samples)
# Architecture: 2-channel gain output (brightness + contrast decomposition)
#
# Matches KBNet qt69d training args but adapted for NAFNet:
#   - Same lr_corrector, lr_potential, lr_negotiator as KBNet
#   - Same hidden_channels=64, epochs=8, batch_size=4
#   - stage_switch_epoch=4 (same as KBNet)
#   - No --resume (train correctors from scratch with current architecture)

python train_v8_cooperative.py \
    --backbone nafnet \
    --pretrained_backbone outputs/nafnet_pku37_w40/best_model.pth \
    --train_jsonl pku37_oct_dataset/pku37_real_train.jsonl \
    --val_jsonl pku37_oct_dataset/pku37_real_val.jsonl \
    --epochs 8 \
    --batch_size 4 \
    --lr_corrector 1e-4 \
    --lr_potential 1e-4 \
    --lr_negotiator 6e-5 \
    --stage_switch_epoch 4 \
    --hidden_channels 64 \
    --ewc_weight 0.1 \
    --bg_correction_var_weight 10.0 \
    --output_dir outputs/nafnet_qt69d_2ch \
    --val_every 1 \
    --max_val 30
