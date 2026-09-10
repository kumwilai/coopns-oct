#!/bin/bash
# =============================================================================
# TMI: NSND-MultiTask Training with Curriculum Learning
# =============================================================================
# Neuro-Symbolic Noise Decomposition + Layer-Aware Denoising
#
# COMBINES:
# 1. NSND novelty: Interpretable noise decomposition (speckle/banding/gaussian/shot)
# 2. Curriculum learning: Clean → Synthetic → Real noise progression
# 3. Clinical importance: Layer-specific denoising weights
#
# CURRICULUM LEARNING PIPELINE:
#   Phase 1A: Segmentation warmup (clean images, frozen denoisers)
#   Phase 1B: Joint training with multi-level synthetic noise
#   Phase 1C: Fine-tune on PKU37 real noise (if available)
#   Phase 2:  Evaluation on Duke17 benchmark
#
# DATA SOURCES:
#   - Duke DME + OCT5k: 1,218 train + 303 val = 1,521 images
#   - PKU37: Real noise paired data (37 subjects)
#   - Duke17: Evaluation benchmark (16 subjects)
#
# KEY TMI CONTRIBUTIONS:
#   1. Neuro-symbolic noise decomposition (interpretable rules)
#   2. Physics-based component denoisers (speckle, banding, Gaussian, shot)
#   3. Layer-aware fusion (anatomical structure guides denoising)
#   4. Curriculum learning for robust generalization
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI: NSND-MultiTask with Curriculum Learning"
echo "=============================================="

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
TRAIN_JSONL="combined_train.jsonl"
VAL_JSONL="combined_val.jsonl"
PKU37_TRAIN="data/pku37_train.jsonl"
PKU37_VAL="data/pku37_val.jsonl"
DUKE_EVAL="duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl"

# Phase 1A: Segmentation warmup (frozen component denoisers)
EPOCHS_SEG=10
BATCH_SIZE=4
LR_SEG=2e-4
PATCH_SIZE=64  # Small patch to avoid OOM

# Phase 1B: Joint training with multi-level synthetic noise
EPOCHS_JOINT=30
LR_JOINT=1e-4
# Multi-level noise: 5 levels for 5x effective data
NOISE_LEVELS="0.05,0.10,0.15,0.20,0.25"

# Phase 1C: Fine-tune on PKU37 real noise
EPOCHS_REAL=10
LR_REAL=5e-5

# Clinical importance weights (from tmi_retrain_clinical.sh)
CLINICAL_RNFL=1.5    # RNFL_GCL - critical for glaucoma
CLINICAL_INL=1.0     # INL_OPL - standard
CLINICAL_ONL=0.8     # ONL - thicker, more tolerant
CLINICAL_ISOS=1.3    # IS_OS - visual acuity
CLINICAL_RPE=1.2     # RPE_Choroid - AMD

# Loss weights
LAMBDA_SEG=0.5
LAMBDA_CONSISTENCY=0.1
LAMBDA_COMPONENT=0.1

# Control phases
RUN_PHASE_1A=${RUN_PHASE_1A:-true}
RUN_PHASE_1B=${RUN_PHASE_1B:-true}
RUN_PHASE_1C=${RUN_PHASE_1C:-true}
RUN_PHASE_2=${RUN_PHASE_2:-true}

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/nsnd_curriculum_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Train data: $TRAIN_JSONL (1,218 samples)"
echo "  Val data:   $VAL_JSONL (303 samples)"
echo "  Noise levels: $NOISE_LEVELS (5x augmentation)"
echo "  Output: $OUTPUT_DIR"
echo ""

# =============================================================================
# PHASE 1A: SEGMENTATION WARMUP (Frozen Denoisers)
# =============================================================================
if [ "$RUN_PHASE_1A" = true ]; then
    echo "=============================================="
    echo "PHASE 1A: Segmentation Warmup"
    echo "=============================================="
    echo "Training segmenter + fusion on clean images."
    echo "Component denoisers are frozen (symbolic analyzer fixed)."
    echo ""

    python train_nsnd_multitask.py \
        --data_jsonl "$TRAIN_JSONL" \
        --val_jsonl "$VAL_JSONL" \
        --epochs $EPOCHS_SEG \
        --batch_size $BATCH_SIZE \
        --lr $LR_SEG \
        --patch_size $PATCH_SIZE \
        --warmup_epochs $EPOCHS_SEG \
        --lambda_seg $LAMBDA_SEG \
        --lambda_consistency $LAMBDA_CONSISTENCY \
        --lambda_component $LAMBDA_COMPONENT \
        --device $DEVICE \
        --num_workers 0 \
        --save_dir "$OUTPUT_DIR/phase_1a_seg" \
        2>&1 | tee "$OUTPUT_DIR/phase_1a_log.txt"

    PHASE_1A_CKPT="$OUTPUT_DIR/phase_1a_seg/best_model.pth"
    echo ""
    echo "Phase 1A Complete: $PHASE_1A_CKPT"
    echo ""
else
    echo "Skipping Phase 1A"
    PHASE_1A_CKPT=""
fi

# =============================================================================
# PHASE 1B: JOINT TRAINING WITH MULTI-LEVEL SYNTHETIC NOISE
# =============================================================================
if [ "$RUN_PHASE_1B" = true ]; then
    echo "=============================================="
    echo "PHASE 1B: Joint Training with Synthetic Noise"
    echo "=============================================="
    echo "Training all components with multi-level noise augmentation."
    echo "Noise levels: $NOISE_LEVELS (5x effective training data)"
    echo ""

    # Resume from Phase 1A if available
    RESUME_ARG=""
    if [ -f "$PHASE_1A_CKPT" ]; then
        RESUME_ARG="--resume $PHASE_1A_CKPT"
        echo "Resuming from Phase 1A: $PHASE_1A_CKPT"
    fi

    python train_nsnd_multitask.py \
        --data_jsonl "$TRAIN_JSONL" \
        --val_jsonl "$VAL_JSONL" \
        --epochs $EPOCHS_JOINT \
        --batch_size $BATCH_SIZE \
        --lr $LR_JOINT \
        --patch_size $PATCH_SIZE \
        --warmup_epochs 0 \
        --finetune_all \
        --lambda_seg $LAMBDA_SEG \
        --lambda_consistency $LAMBDA_CONSISTENCY \
        --lambda_component $LAMBDA_COMPONENT \
        --device $DEVICE \
        --num_workers 0 \
        --save_dir "$OUTPUT_DIR/phase_1b_joint" \
        $RESUME_ARG \
        2>&1 | tee "$OUTPUT_DIR/phase_1b_log.txt"

    PHASE_1B_CKPT="$OUTPUT_DIR/phase_1b_joint/best_model.pth"
    echo ""
    echo "Phase 1B Complete: $PHASE_1B_CKPT"
    echo ""
else
    echo "Skipping Phase 1B"
    PHASE_1B_CKPT="$PHASE_1A_CKPT"
fi

# =============================================================================
# PHASE 1C: FINE-TUNE ON PKU37 REAL NOISE (Optional)
# =============================================================================
if [ "$RUN_PHASE_1C" = true ]; then
    echo "=============================================="
    echo "PHASE 1C: Fine-tune on PKU37 Real Noise"
    echo "=============================================="

    if [ -f "$PKU37_TRAIN" ]; then
        echo "Fine-tuning on real noise from PKU37 dataset."
        echo ""

        RESUME_ARG=""
        if [ -f "$PHASE_1B_CKPT" ]; then
            RESUME_ARG="--resume $PHASE_1B_CKPT"
        elif [ -f "$PHASE_1A_CKPT" ]; then
            RESUME_ARG="--resume $PHASE_1A_CKPT"
        fi

        python train_nsnd_multitask.py \
            --data_jsonl "$PKU37_TRAIN" \
            --val_jsonl "$PKU37_VAL" \
            --epochs $EPOCHS_REAL \
            --batch_size $BATCH_SIZE \
            --lr $LR_REAL \
            --patch_size $PATCH_SIZE \
            --warmup_epochs 0 \
            --finetune_all \
            --lambda_seg 0.0 \
            --lambda_consistency $LAMBDA_CONSISTENCY \
            --lambda_component $LAMBDA_COMPONENT \
            --device $DEVICE \
            --num_workers 0 \
            --save_dir "$OUTPUT_DIR/phase_1c_pku37" \
            $RESUME_ARG \
            2>&1 | tee "$OUTPUT_DIR/phase_1c_log.txt"

        FINAL_CKPT="$OUTPUT_DIR/phase_1c_pku37/best_model.pth"
        echo ""
        echo "Phase 1C Complete: $FINAL_CKPT"
    else
        echo "PKU37 data not found: $PKU37_TRAIN"
        echo "Skipping Phase 1C"
        FINAL_CKPT="$PHASE_1B_CKPT"
    fi
    echo ""
else
    echo "Skipping Phase 1C"
    FINAL_CKPT="$PHASE_1B_CKPT"
fi

# =============================================================================
# PHASE 2: EVALUATION ON DUKE17 BENCHMARK
# =============================================================================
if [ "$RUN_PHASE_2" = true ]; then
    echo "=============================================="
    echo "PHASE 2: Evaluation on Duke17 Benchmark"
    echo "=============================================="

    # Find best checkpoint
    BEST_CKPT=""
    for ckpt in "$FINAL_CKPT" "$PHASE_1B_CKPT" "$PHASE_1A_CKPT"; do
        if [ -f "$ckpt" ]; then
            BEST_CKPT="$ckpt"
            break
        fi
    done

    if [ -n "$BEST_CKPT" ]; then
        echo "Evaluating checkpoint: $BEST_CKPT"
        echo ""

        python -c "
import torch
import numpy as np
import json
from tifffile import imread as tiff_imread
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.insert(0, '.')
from nsnd_multitask_model import NSNDMultiTaskDenoiser

device = '$DEVICE'
print('Loading model from: $BEST_CKPT')
model = NSNDMultiTaskDenoiser(device=device).to(device)
ckpt = torch.load('$BEST_CKPT', map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Load evaluation data
eval_data = []
with open('$DUKE_EVAL', 'r') as f:
    for line in f:
        if line.strip():
            eval_data.append(json.loads(line))

print(f'Evaluating on {len(eval_data)} Duke17 samples...')
print()

results = []
for sample in eval_data:
    noisy = tiff_imread(sample['noisy_path']).astype(np.float32)
    clean = tiff_imread(sample['clean_path']).astype(np.float32)
    if noisy.ndim == 3:
        noisy, clean = noisy[0], clean[0]

    noisy_norm = noisy / 255.0
    clean_norm = clean / 255.0

    H, W = noisy_norm.shape
    noisy_crop = noisy_norm[:min(256, H), :min(256, W)]
    clean_crop = clean_norm[:min(256, H), :min(256, W)]

    noisy_t = torch.from_numpy(noisy_crop).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        denoised, _, _ = model(noisy_t)

    denoised_np = denoised[0, 0].cpu().numpy()

    mse_in = np.mean((noisy_crop - clean_crop) ** 2)
    mse_out = np.mean((denoised_np - clean_crop) ** 2)
    psnr_in = 10 * np.log10(1.0 / (mse_in + 1e-10))
    psnr_out = 10 * np.log10(1.0 / (mse_out + 1e-10))
    ssim_out = ssim(denoised_np, clean_crop, data_range=1.0)

    results.append({
        'subject': sample['subject'],
        'psnr_in': psnr_in,
        'psnr_out': psnr_out,
        'ssim': ssim_out
    })
    print(f\"  {sample['subject']}: PSNR {psnr_in:.1f} -> {psnr_out:.1f} dB ({psnr_out-psnr_in:+.1f}), SSIM={ssim_out:.3f}\")

print()
print('='*50)
print('DUKE17 EVALUATION RESULTS')
print('='*50)
avg_in = np.mean([r['psnr_in'] for r in results])
avg_out = np.mean([r['psnr_out'] for r in results])
std_out = np.std([r['psnr_out'] for r in results])
avg_ssim = np.mean([r['ssim'] for r in results])
std_ssim = np.std([r['ssim'] for r in results])

print(f'Input PSNR:  {avg_in:.2f} dB')
print(f'Output PSNR: {avg_out:.2f} +/- {std_out:.2f} dB')
print(f'Improvement: {avg_out - avg_in:+.2f} dB')
print(f'Output SSIM: {avg_ssim:.4f} +/- {std_ssim:.4f}')
print('='*50)

# Save results
with open('$OUTPUT_DIR/duke17_results.json', 'w') as f:
    json.dump({
        'checkpoint': '$BEST_CKPT',
        'mean_psnr_in': float(avg_in),
        'mean_psnr_out': float(avg_out),
        'std_psnr_out': float(std_out),
        'mean_ssim': float(avg_ssim),
        'std_ssim': float(std_ssim),
        'improvement_db': float(avg_out - avg_in),
        'per_subject': results
    }, f, indent=2)
print(f'Results saved to: $OUTPUT_DIR/duke17_results.json')
"
        echo ""
    else
        echo "No checkpoint found for evaluation"
    fi
else
    echo "Skipping Phase 2"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "NSND-MultiTask Training Complete"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Phases completed:"
[ "$RUN_PHASE_1A" = true ] && echo "  - Phase 1A: Segmentation warmup (frozen denoisers)"
[ "$RUN_PHASE_1B" = true ] && echo "  - Phase 1B: Joint training with synthetic noise"
[ "$RUN_PHASE_1C" = true ] && echo "  - Phase 1C: PKU37 real noise fine-tuning"
[ "$RUN_PHASE_2" = true ] && echo "  - Phase 2: Duke17 evaluation"
echo ""
echo "=============================================="
echo "KEY TMI CONTRIBUTIONS"
echo "=============================================="
echo ""
echo "1. NEURO-SYMBOLIC NOISE DECOMPOSITION:"
echo "   - Soft-logic rules classify noise types"
echo "   - Interpretable: speckle, banding, Gaussian, shot"
echo ""
echo "2. PHYSICS-BASED COMPONENT DENOISERS:"
echo "   - Speckle: Anisotropic diffusion"
echo "   - Banding: FFT notch filter"
echo "   - Gaussian: DnCNN"
echo "   - Shot: Variance-stabilizing transform"
echo ""
echo "3. LAYER-AWARE FUSION:"
echo "   - Per-pixel: noise_weight × layer_modulation"
echo "   - Anatomical structure guides denoising"
echo ""
echo "4. CURRICULUM LEARNING:"
echo "   - Clean → Synthetic → Real noise"
echo "   - Robust generalization"
echo ""
echo "=============================================="
