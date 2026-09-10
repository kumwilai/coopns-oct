#!/bin/bash
#
# NSND-MultiTask Training Script for TMI Paper
#
# Neuro-Symbolic Noise Decomposition with Layer-Aware OCT Denoising
#
# KEY TMI CONTRIBUTIONS:
# 1. Neuro-symbolic noise decomposition - interpretable noise type classification
# 2. Physics-based component denoisers - speckle, banding, Gaussian, shot
# 3. Layer-aware fusion - anatomical structure guides denoising
# 4. Joint denoising + segmentation training
#
# Usage: bash train_nsnd_tmi.sh
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Data paths
TRAIN_JSONL="combined_train.jsonl"
VAL_JSONL="combined_val.jsonl"
EVAL_JSONL="duke_sota_datasets/Duke17_Eval/combined_eval.jsonl"

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=2
LR=1e-4
PATCH_SIZE=64  # Small patch to avoid OOM on limited memory systems

# NSND-specific
WARMUP_EPOCHS=10
GAUSSIAN_TYPE="dncnn"  # "dncnn" or "nafnet"
FEATURE_CHANNELS=64

# Loss weights
LAMBDA_SEG=1.0
LAMBDA_CONSISTENCY=0.1
LAMBDA_COMPONENT=0.1

# Output
SAVE_DIR="checkpoints/nsnd_multitask_tmi"
LOG_FILE="${SAVE_DIR}/training.log"

# GPU
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# Pre-checks
# =============================================================================

echo "=============================================="
echo "NSND-MultiTask Training for TMI Paper"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Train data: ${TRAIN_JSONL}"
echo "  Val data:   ${VAL_JSONL}"
echo "  Eval data:  ${EVAL_JSONL}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  LR:         ${LR}"
echo "  Patch size: ${PATCH_SIZE}"
echo "  Warmup:     ${WARMUP_EPOCHS} epochs"
echo "  Save dir:   ${SAVE_DIR}"
echo ""

# Check data files exist
if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_JSONL}"
    exit 1
fi

# Create output directory
mkdir -p "${SAVE_DIR}"

# =============================================================================
# Phase 1: Warmup (Frozen Symbolic Analyzer & Denoisers)
# =============================================================================

echo "=============================================="
echo "Phase 1: Warmup Training"
echo "=============================================="
echo "Training fusion network only (frozen symbolic analyzer and component denoisers)"
echo "This allows the fusion to learn good initial weights before fine-tuning"
echo ""

python train_nsnd_multitask.py \
    --data_jsonl "${TRAIN_JSONL}" \
    --val_jsonl "${VAL_JSONL}" \
    --epochs ${WARMUP_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --patch_size ${PATCH_SIZE} \
    --use_neuro_symbolic \
    --gaussian_type ${GAUSSIAN_TYPE} \
    --feature_channels ${FEATURE_CHANNELS} \
    --lambda_seg ${LAMBDA_SEG} \
    --lambda_consistency ${LAMBDA_CONSISTENCY} \
    --lambda_component ${LAMBDA_COMPONENT} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --save_dir "${SAVE_DIR}" \
    2>&1 | tee "${LOG_FILE}"

# =============================================================================
# Phase 2: Full Fine-tuning
# =============================================================================

echo ""
echo "=============================================="
echo "Phase 2: Full Fine-tuning"
echo "=============================================="
echo "Fine-tuning all components (symbolic analyzer, denoisers, fusion, segmenter)"
echo ""

# Resume from warmup checkpoint
WARMUP_CKPT="${SAVE_DIR}/epoch_${WARMUP_EPOCHS}.pth"
if [ ! -f "${WARMUP_CKPT}" ]; then
    WARMUP_CKPT="${SAVE_DIR}/best_model.pth"
fi

python train_nsnd_multitask.py \
    --data_jsonl "${TRAIN_JSONL}" \
    --val_jsonl "${VAL_JSONL}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr $(echo "${LR} * 0.1" | bc -l) \
    --patch_size ${PATCH_SIZE} \
    --use_neuro_symbolic \
    --gaussian_type ${GAUSSIAN_TYPE} \
    --feature_channels ${FEATURE_CHANNELS} \
    --lambda_seg ${LAMBDA_SEG} \
    --lambda_consistency ${LAMBDA_CONSISTENCY} \
    --lambda_component ${LAMBDA_COMPONENT} \
    --warmup_epochs 0 \
    --finetune_all \
    --resume "${WARMUP_CKPT}" \
    --save_dir "${SAVE_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

# =============================================================================
# Evaluation on Duke17 Benchmark
# =============================================================================

echo ""
echo "=============================================="
echo "Evaluation on Duke17 Benchmark"
echo "=============================================="

# Create evaluation script inline
python -c "
import torch
import json
import numpy as np
from PIL import Image
from tifffile import imread as tiff_imread
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from nsnd_multitask_model import NSNDMultiTaskDenoiser

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = NSNDMultiTaskDenoiser(device=device).to(device)
ckpt = torch.load('${SAVE_DIR}/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f'Loaded model from epoch {ckpt[\"epoch\"]}')

# Load evaluation data
eval_data = []
with open('${EVAL_JSONL}', 'r') as f:
    for line in f:
        if line.strip():
            eval_data.append(json.loads(line))

print(f'Evaluating on {len(eval_data)} samples...')

results = {'psnr': [], 'ssim': [], 'subjects': []}

for sample in eval_data:
    # Load images
    noisy = tiff_imread(sample['noisy_path']).astype(np.float32)
    clean = tiff_imread(sample['clean_path']).astype(np.float32)

    # Handle multi-page TIFF
    if noisy.ndim == 3 and noisy.shape[0] < noisy.shape[1]:
        noisy = noisy[0]
        clean = clean[0]

    # Normalize
    noisy = noisy / noisy.max() if noisy.max() > 1 else noisy
    clean = clean / clean.max() if clean.max() > 1 else clean

    # Process
    noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        denoised, _, _ = model(noisy_t, return_intermediates=False)

    denoised = denoised[0, 0].cpu().numpy()

    # Metrics
    mse = np.mean((denoised - clean) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    ssim_val = ssim(denoised, clean, data_range=1.0)

    results['psnr'].append(psnr)
    results['ssim'].append(ssim_val)
    results['subjects'].append(sample['subject'])

print()
print('='*50)
print('NSND-MultiTask Evaluation Results')
print('='*50)
print(f'PSNR: {np.mean(results[\"psnr\"]):.2f} +/- {np.std(results[\"psnr\"]):.2f} dB')
print(f'SSIM: {np.mean(results[\"ssim\"]):.4f} +/- {np.std(results[\"ssim\"]):.4f}')
print('='*50)

# Save results
with open('${SAVE_DIR}/eval_results.json', 'w') as f:
    json.dump({
        'mean_psnr': float(np.mean(results['psnr'])),
        'std_psnr': float(np.std(results['psnr'])),
        'mean_ssim': float(np.mean(results['ssim'])),
        'std_ssim': float(np.std(results['ssim'])),
        'per_subject': list(zip(results['subjects'], results['psnr'], results['ssim']))
    }, f, indent=2)
print(f'Results saved to ${SAVE_DIR}/eval_results.json')
"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Best model: ${SAVE_DIR}/best_model.pth"
echo "  Final model: ${SAVE_DIR}/final_model.pth"
echo "  Training log: ${LOG_FILE}"
echo "  Eval results: ${SAVE_DIR}/eval_results.json"
echo ""
echo "To use the model:"
echo "  from nsnd_multitask_model import NSNDMultiTaskDenoiser"
echo "  model = NSNDMultiTaskDenoiser()"
echo "  model.load_state_dict(torch.load('${SAVE_DIR}/best_model.pth')['state_dict'])"
echo ""
