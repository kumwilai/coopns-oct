#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kumwilai/OCT"
TRAIN_PAIRS="${ROOT}/train_pairs_duke_analysis.txt"
VAL_PAIRS="${ROOT}/val_pairs_duke_analysis.txt"
OUT_DIR="${ROOT}/outputs/baselines_duke_analysis"

mkdir -p "${OUT_DIR}"

# Fair-size baselines (~7.5-8.1M params) with same paired data + 64x64 crops

# NAFNet (width=64 -> ~7.55M params)
python "${ROOT}/train_nafnet_monitored.py" \
  --train_pairs "${TRAIN_PAIRS}" \
  --val_pairs "${VAL_PAIRS}" \
  --out_dir "${OUT_DIR}/nafnet_w64" \
  --epochs 50 \
  --batch_size 4 \
  --size 64 \
  --use_crop \
  --grad_w 0.0 \
  --width 64 \
  --middle_blk_num 2

# SwinIR (embed=184, depths=8,8,8, heads=4,4,4 -> ~7.93M params with RSTB)
python "${ROOT}/train_swinir_monitored.py" \
  --train_pairs "${TRAIN_PAIRS}" \
  --val_pairs "${VAL_PAIRS}" \
  --out_dir "${OUT_DIR}/swinir_e184_d888_h444" \
  --epochs 50 \
  --batch_size 4 \
  --size 64 \
  --use_crop \
  --loss l1 \
  --embed_dim 184 \
  --depths 8,8,8 \
  --num_heads 4,4,4 \
  --window_size 8

# DRUNet (nc=44,88,176,352, nb=2,2,2,2 -> ~8.05M params)
python "${ROOT}/train_drunet_monitored.py" \
  --train_pairs "${TRAIN_PAIRS}" \
  --val_pairs "${VAL_PAIRS}" \
  --out_dir "${OUT_DIR}/drunet_nc44_nb2222" \
  --epochs 50 \
  --batch_size 4 \
  --size 64 \
  --use_crop \
  --loss l2 \
  --nc 44,88,176,352 \
  --nb 2,2,2,2 \
  --noise_level_sigma 25.0

# U-Net (features=32 -> ~7.76M params, full U-Net)
python "${ROOT}/train_unet_monitored.py" \
  --train_pairs "${TRAIN_PAIRS}" \
  --val_pairs "${VAL_PAIRS}" \
  --out_dir "${OUT_DIR}/unet_f32" \
  --epochs 50 \
  --batch_size 4 \
  --size 64 \
  --use_crop \
  --loss l2 \
  --features 32
