#!/bin/bash
# =============================================================================
# NSND-MultiTask Training with Combined Data Strategy (A+B)
# =============================================================================
#
# DATA STRATEGY:
#   Option A: Synthetic noise + Real masks (Duke DME + OCT5k)
#   Option B: Real noise + Pseudo masks (Duke17 + PKU37)
#
# TRAINING PIPELINE:
#   Phase 1: Prepare combined data (synthetic noise + pseudo masks)
#   Phase 2: Train segmenter on clean data (warmup)
#   Phase 3: Joint training on combined data
#   Phase 4: Fine-tune on real noise data
#   Phase 5: Evaluation
#
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "NSND-MultiTask: Combined A+B Training"
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

# Input data
SEG_DATA="combined_train.jsonl"           # Duke DME + OCT5k (1,521 images with masks)
SEG_VAL="combined_val.jsonl"              # Validation set
DUKE17_DATA="duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl"  # Real noise pairs

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/nsnd_combined_${TIMESTAMP}"
DATA_DIR="data/joint_training_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR" "$DATA_DIR"

# Training hyperparameters
BATCH_SIZE=2
PATCH_SIZE=64
LR_WARMUP=2e-4
LR_JOINT=1e-4
LR_FINETUNE=5e-5

# Epochs for each phase
EPOCHS_WARMUP=5        # Segmenter warmup
EPOCHS_JOINT=20        # Joint training
EPOCHS_FINETUNE=10     # Fine-tune on real noise

# Noise levels for synthetic data
NOISE_LEVELS="0.05,0.10,0.15,0.20,0.25"

# Control which phases to run
RUN_DATA_PREP=${RUN_DATA_PREP:-true}
RUN_WARMUP=${RUN_WARMUP:-true}
RUN_JOINT=${RUN_JOINT:-true}
RUN_FINETUNE=${RUN_FINETUNE:-true}
RUN_EVAL=${RUN_EVAL:-true}

echo ""
echo "Configuration:"
echo "  Segmentation data: $SEG_DATA"
echo "  Real noise data:   $DUKE17_DATA"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Noise levels:      $NOISE_LEVELS"
echo ""

# =============================================================================
# PHASE 1: DATA PREPARATION
# =============================================================================
if [ "$RUN_DATA_PREP" = true ]; then
    echo "=============================================="
    echo "PHASE 1: Data Preparation"
    echo "=============================================="
    echo ""
    echo "Option A: Adding synthetic noise to segmentation data"
    echo "Option B: Generating pseudo-masks for denoising data"
    echo ""

    # Check for existing segmenter checkpoint for pseudo-mask generation
    SEGMENTER_CKPT=""
    for ckpt in "checkpoints/nsnd_test/best_model.pth" \
                "checkpoints/nsnd_multitask_tmi/best_model.pth" \
                "checkpoints/multitask_clinical/best_model.pth"; do
        if [ -f "$ckpt" ]; then
            SEGMENTER_CKPT="$ckpt"
            echo "Found segmenter checkpoint: $SEGMENTER_CKPT"
            break
        fi
    done

    if [ -z "$SEGMENTER_CKPT" ]; then
        echo "No existing segmenter checkpoint found."
        echo "Will use randomly initialized segmenter for pseudo-masks."
        echo "(The segmenter will be properly trained in Phase 2)"
    fi

    # Prepare combined data
    python prepare_joint_training_data.py \
        --seg_data "$SEG_DATA" \
        --denoise_data "$DUKE17_DATA" \
        --segmenter_ckpt "$SEGMENTER_CKPT" \
        --output_dir "$DATA_DIR" \
        --output_name "joint_nsnd" \
        --noise_levels "$NOISE_LEVELS" \
        --device "$DEVICE" \
        2>&1 | tee "$OUTPUT_DIR/phase1_data_prep.log"

    JOINT_TRAIN="$DATA_DIR/joint_nsnd_train.jsonl"
    JOINT_VAL="$DATA_DIR/joint_nsnd_val.jsonl"

    echo ""
    echo "Phase 1 Complete: Data prepared"
    echo "  Train: $JOINT_TRAIN"
    echo "  Val:   $JOINT_VAL"
else
    echo "Skipping Phase 1: Data Preparation"
    JOINT_TRAIN="$DATA_DIR/joint_nsnd_train.jsonl"
    JOINT_VAL="$DATA_DIR/joint_nsnd_val.jsonl"
fi

# =============================================================================
# PHASE 2: SEGMENTER WARMUP
# =============================================================================
if [ "$RUN_WARMUP" = true ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 2: Segmenter Warmup"
    echo "=============================================="
    echo ""
    echo "Training segmenter on clean images with real masks."
    echo "Symbolic analyzer and denoisers are frozen."
    echo ""

    python train_nsnd_multitask.py \
        --data_jsonl "$SEG_DATA" \
        --val_jsonl "$SEG_VAL" \
        --epochs $EPOCHS_WARMUP \
        --batch_size $BATCH_SIZE \
        --lr $LR_WARMUP \
        --patch_size $PATCH_SIZE \
        --warmup_epochs $EPOCHS_WARMUP \
        --lambda_seg 1.0 \
        --lambda_consistency 0.0 \
        --lambda_component 0.0 \
        --device $DEVICE \
        --num_workers 0 \
        --save_dir "$OUTPUT_DIR/phase2_warmup" \
        2>&1 | tee "$OUTPUT_DIR/phase2_warmup.log"

    WARMUP_CKPT="$OUTPUT_DIR/phase2_warmup/best_model.pth"
    echo ""
    echo "Phase 2 Complete: $WARMUP_CKPT"
else
    echo "Skipping Phase 2: Segmenter Warmup"
    WARMUP_CKPT=""
fi

# =============================================================================
# PHASE 3: JOINT TRAINING ON COMBINED DATA
# =============================================================================
if [ "$RUN_JOINT" = true ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 3: Joint Training (Combined A+B)"
    echo "=============================================="
    echo ""
    echo "Training on combined data:"
    echo "  - Synthetic noise + Real masks (Option A)"
    echo "  - Real noise + Pseudo masks (Option B)"
    echo ""

    # Resume from warmup if available
    RESUME_ARG=""
    if [ -f "$WARMUP_CKPT" ]; then
        RESUME_ARG="--resume $WARMUP_CKPT"
        echo "Resuming from warmup: $WARMUP_CKPT"
    fi

    # Use joint data if available, otherwise fall back to seg data
    TRAIN_DATA="$JOINT_TRAIN"
    VAL_DATA="$JOINT_VAL"
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "Joint data not found, using segmentation data with synthetic noise"
        TRAIN_DATA="$SEG_DATA"
        VAL_DATA="$SEG_VAL"
    fi

    python train_nsnd_multitask.py \
        --data_jsonl "$TRAIN_DATA" \
        --val_jsonl "$VAL_DATA" \
        --epochs $EPOCHS_JOINT \
        --batch_size $BATCH_SIZE \
        --lr $LR_JOINT \
        --patch_size $PATCH_SIZE \
        --warmup_epochs 0 \
        --finetune_all \
        --lambda_seg 0.5 \
        --lambda_consistency 0.1 \
        --lambda_component 0.1 \
        --device $DEVICE \
        --num_workers 0 \
        --save_dir "$OUTPUT_DIR/phase3_joint" \
        $RESUME_ARG \
        2>&1 | tee "$OUTPUT_DIR/phase3_joint.log"

    JOINT_CKPT="$OUTPUT_DIR/phase3_joint/best_model.pth"
    echo ""
    echo "Phase 3 Complete: $JOINT_CKPT"
else
    echo "Skipping Phase 3: Joint Training"
    JOINT_CKPT="$WARMUP_CKPT"
fi

# =============================================================================
# PHASE 4: FINE-TUNE ON REAL NOISE
# =============================================================================
if [ "$RUN_FINETUNE" = true ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 4: Fine-tune on Real Noise (Duke17)"
    echo "=============================================="

    # Check if Duke17 data exists
    if [ -f "$DUKE17_DATA" ]; then
        echo ""
        echo "Fine-tuning on real noise data from Duke17."
        echo ""

        # Resume from joint training
        RESUME_ARG=""
        for ckpt in "$JOINT_CKPT" "$WARMUP_CKPT"; do
            if [ -f "$ckpt" ]; then
                RESUME_ARG="--resume $ckpt"
                echo "Resuming from: $ckpt"
                break
            fi
        done

        # Create JSONL for Duke17 with pseudo masks (if we have them)
        DUKE17_WITH_MASKS="$DATA_DIR/pseudo_masks/duke17_with_masks.jsonl"
        if [ -f "$DUKE17_WITH_MASKS" ]; then
            FINETUNE_DATA="$DUKE17_WITH_MASKS"
        else
            FINETUNE_DATA="$DUKE17_DATA"
        fi

        python train_nsnd_multitask.py \
            --data_jsonl "$FINETUNE_DATA" \
            --epochs $EPOCHS_FINETUNE \
            --batch_size $BATCH_SIZE \
            --lr $LR_FINETUNE \
            --patch_size $PATCH_SIZE \
            --warmup_epochs 0 \
            --finetune_all \
            --lambda_seg 0.0 \
            --lambda_consistency 0.1 \
            --lambda_component 0.1 \
            --device $DEVICE \
            --num_workers 0 \
            --save_dir "$OUTPUT_DIR/phase4_finetune" \
            $RESUME_ARG \
            2>&1 | tee "$OUTPUT_DIR/phase4_finetune.log"

        FINAL_CKPT="$OUTPUT_DIR/phase4_finetune/best_model.pth"
    else
        echo "Duke17 data not found: $DUKE17_DATA"
        echo "Skipping fine-tuning phase"
        FINAL_CKPT="$JOINT_CKPT"
    fi
    echo ""
    echo "Phase 4 Complete: $FINAL_CKPT"
else
    echo "Skipping Phase 4: Fine-tune"
    FINAL_CKPT="$JOINT_CKPT"
fi

# =============================================================================
# PHASE 5: EVALUATION
# =============================================================================
if [ "$RUN_EVAL" = true ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 5: Evaluation on Duke17"
    echo "=============================================="

    # Find best checkpoint
    BEST_CKPT=""
    for ckpt in "$FINAL_CKPT" "$JOINT_CKPT" "$WARMUP_CKPT"; do
        if [ -f "$ckpt" ]; then
            BEST_CKPT="$ckpt"
            break
        fi
    done

    if [ -n "$BEST_CKPT" ] && [ -f "$DUKE17_DATA" ]; then
        echo "Evaluating: $BEST_CKPT"
        echo ""

        python -c "
import torch
import json
import numpy as np
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
with open('$DUKE17_DATA', 'r') as f:
    for line in f:
        if line.strip():
            eval_data.append(json.loads(line))

print(f'Evaluating on {len(eval_data)} samples...')
print()

results = []
for sample in eval_data:
    noisy = tiff_imread(sample['noisy_path']).astype(np.float32)
    clean = tiff_imread(sample['clean_path']).astype(np.float32)

    if noisy.ndim == 3:
        noisy, clean = noisy[0], clean[0]

    noisy_norm = noisy / 255.0
    clean_norm = clean / 255.0

    # Crop to manageable size
    H, W = noisy_norm.shape
    noisy_crop = noisy_norm[:min(256, H), :min(256, W)]
    clean_crop = clean_norm[:min(256, H), :min(256, W)]

    noisy_t = torch.from_numpy(noisy_crop).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        denoised, seg_logits, _ = model(noisy_t)

    denoised_np = denoised[0, 0].cpu().numpy()

    # Metrics
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
    print(f\"  {sample['subject']}: PSNR {psnr_in:.1f} -> {psnr_out:.1f} dB (+{psnr_out-psnr_in:.1f}), SSIM={ssim_out:.3f}\")

print()
print('=' * 50)
print('DUKE17 EVALUATION RESULTS')
print('=' * 50)
avg_in = np.mean([r['psnr_in'] for r in results])
avg_out = np.mean([r['psnr_out'] for r in results])
std_out = np.std([r['psnr_out'] for r in results])
avg_ssim = np.mean([r['ssim'] for r in results])
std_ssim = np.std([r['ssim'] for r in results])

print(f'Input PSNR:  {avg_in:.2f} dB')
print(f'Output PSNR: {avg_out:.2f} +/- {std_out:.2f} dB')
print(f'Improvement: +{avg_out - avg_in:.2f} dB')
print(f'Output SSIM: {avg_ssim:.4f} +/- {std_ssim:.4f}')
print('=' * 50)

# Save results
with open('$OUTPUT_DIR/eval_results.json', 'w') as f:
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
print(f'Results saved to: $OUTPUT_DIR/eval_results.json')
"
    else
        echo "No checkpoint or evaluation data found"
    fi
else
    echo "Skipping Phase 5: Evaluation"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "NSND Combined Training Complete"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Data Strategy:"
echo "  Option A: Synthetic noise + Real masks (segmentation data)"
echo "  Option B: Real noise + Pseudo masks (denoising data)"
echo ""
echo "Training Phases:"
[ "$RUN_DATA_PREP" = true ] && echo "  1. Data preparation: Combined A+B"
[ "$RUN_WARMUP" = true ] && echo "  2. Segmenter warmup: Clean images + real masks"
[ "$RUN_JOINT" = true ] && echo "  3. Joint training: Combined synthetic + real noise"
[ "$RUN_FINETUNE" = true ] && echo "  4. Fine-tune: Real noise (Duke17)"
[ "$RUN_EVAL" = true ] && echo "  5. Evaluation: Duke17 benchmark"
echo ""
echo "=============================================="
