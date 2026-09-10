#!/bin/bash
# =============================================================================
# TMI COMPLETE PIPELINE: Fine-tuning + Real-World Evaluation + Neuro-Symbolic
# =============================================================================
#
# This script runs the complete evaluation pipeline for the TMI paper:
#
# PHASE 1: Dice Fine-tuning
#   - Freezes backbone (preserves PSNR)
#   - Trains segmentation head for better Dice
#
# PHASE 2: Real-World Evaluation (Duke + PKU)
#   - Tests on Duke Human OCT (real noise)
#   - Tests on PKU37 OCT Benchmark (real noise)
#
# PHASE 3: Neuro-Symbolic Post-Processing
#   - Applies symbolic rules for anatomical correctness
#   - Generates clinical reports
#
# Usage:
#   bash tmi_complete_pipeline.sh [checkpoint_path]
#
# Example:
#   bash tmi_complete_pipeline.sh tmi_cuap_oct/full_model/20260115_143743/best_psnr.pth
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "TMI COMPLETE PIPELINE"
echo "=============================================================================="
echo ""
echo "Clinically-guided Uncertainty-Aware Pathology-preserving OCT Denoising"
echo "with Neuro-Symbolic Post-Processing"
echo ""
echo "Pipeline:"
echo "  Phase 1: Dice Fine-tuning"
echo "  Phase 2: Real-World Evaluation (Duke + PKU37)"
echo "  Phase 3: Neuro-Symbolic Analysis"
echo "=============================================================================="

# =============================================================================
# CONFIGURATION
# =============================================================================

# Find checkpoint
if [ -n "$1" ]; then
    CHECKPOINT="$1"
else
    # Auto-find latest best_psnr.pth
    CHECKPOINT=$(find tmi_cuap_oct -name "best_psnr.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found. Please provide path to best_psnr.pth"
        echo "Usage: bash tmi_complete_pipeline.sh path/to/best_psnr.pth"
        exit 1
    fi
fi

echo ""
echo "Input Checkpoint: $CHECKPOINT"

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="tmi_complete_results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Dataset paths
DUKE_PATH="duke_datasets/organized_test_pairs/human"
PKU_PATH="pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"
BACKBONE_CKPT="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"

echo ""

# =============================================================================
# PHASE 1: DICE FINE-TUNING
# =============================================================================
echo ""
echo "=============================================================================="
echo "PHASE 1: DICE FINE-TUNING"
echo "=============================================================================="
echo ""
echo "Strategy:"
echo "  - Backbone (denoising): FROZEN"
echo "  - Segmentation head: TRAINABLE"
echo "  - Goal: Improve Dice while preserving PSNR"
echo ""

FINETUNE_EPOCHS=15
FINETUNE_LR=5e-5
FINETUNE_LAMBDA_DICE=0.7
FINETUNE_OUTPUT="$OUTPUT_DIR/finetune"
mkdir -p "$FINETUNE_OUTPUT"

python finetune_dice.py \
    --checkpoint "$CHECKPOINT" \
    --backbone_ckpt "$BACKBONE_CKPT" \
    --train_jsonl seg_data/seg_train.jsonl \
    --val_jsonl seg_data/seg_val.jsonl \
    --epochs $FINETUNE_EPOCHS \
    --batch_size 8 \
    --lr $FINETUNE_LR \
    --lambda_dice $FINETUNE_LAMBDA_DICE \
    --max_train 2000 \
    --max_val 400 \
    --device $DEVICE \
    --output_dir "$FINETUNE_OUTPUT" \
    2>&1 | tee "$FINETUNE_OUTPUT/finetune_log.txt"

# Get fine-tuned checkpoint
FINETUNED_CKPT="$FINETUNE_OUTPUT/best_dice_finetuned.pth"
if [ ! -f "$FINETUNED_CKPT" ]; then
    echo "WARNING: Fine-tuned checkpoint not found, using original"
    FINETUNED_CKPT="$CHECKPOINT"
fi

echo ""
echo "Phase 1 Complete: $FINETUNED_CKPT"

# =============================================================================
# PHASE 2: REAL-WORLD EVALUATION (DUKE + PKU37)
# =============================================================================
echo ""
echo "=============================================================================="
echo "PHASE 2: REAL-WORLD EVALUATION"
echo "=============================================================================="
echo ""
echo "Testing generalization to REAL noise (not seen during training)"
echo ""
echo "Datasets:"
echo "  - Duke Human OCT: Real frame-averaged pairs"
echo "  - PKU37 OCT Benchmark: 37 subjects, real noise"
echo ""

EVAL_OUTPUT="$OUTPUT_DIR/real_world_eval"
mkdir -p "$EVAL_OUTPUT"

# Check if evaluation script exists, if not create it
if [ ! -f "evaluate_real_world.py" ]; then
    echo "Creating real-world evaluation script..."
    cat > evaluate_real_world.py << 'EVALSCRIPT'
#!/usr/bin/env python3
"""Real-world evaluation on Duke and PKU37 datasets."""

import argparse
import json
import os
import sys
from glob import glob

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_multitask import MultiTaskDenoiser, LAYER_NAMES
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.utils.metrics import compute_psnr, compute_ssim


def load_image(path):
    """Load image as tensor [1, 1, H, W]."""
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def evaluate_duke(model, base_model, duke_path, device):
    """Evaluate on Duke Human OCT dataset."""
    results = {'psnr_base': [], 'psnr_ours': [], 'ssim_base': [], 'ssim_ours': []}

    noisy_dir = os.path.join(duke_path, 'noisy')
    clean_dir = os.path.join(duke_path, 'clean')

    if not os.path.exists(noisy_dir):
        print(f"Duke dataset not found at {duke_path}")
        return None

    noisy_files = sorted(glob(os.path.join(noisy_dir, '*.png')))

    for noisy_path in tqdm(noisy_files, desc="Duke Evaluation"):
        fname = os.path.basename(noisy_path)
        clean_path = os.path.join(clean_dir, fname)

        if not os.path.exists(clean_path):
            continue

        noisy = load_image(noisy_path).to(device)
        clean = load_image(clean_path).to(device)

        with torch.no_grad():
            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            ours_out, _ = model(noisy)

        results['psnr_base'].append(compute_psnr(base_out, clean))
        results['psnr_ours'].append(compute_psnr(ours_out, clean))
        results['ssim_base'].append(compute_ssim(base_out, clean))
        results['ssim_ours'].append(compute_ssim(ours_out, clean))

    if not results['psnr_base']:
        return None

    return {
        'psnr_base': np.mean(results['psnr_base']),
        'psnr_ours': np.mean(results['psnr_ours']),
        'psnr_gain': np.mean(results['psnr_ours']) - np.mean(results['psnr_base']),
        'ssim_base': np.mean(results['ssim_base']),
        'ssim_ours': np.mean(results['ssim_ours']),
        'ssim_gain': np.mean(results['ssim_ours']) - np.mean(results['ssim_base']),
        'n_samples': len(results['psnr_base']),
    }


def evaluate_pku(model, base_model, pku_path, device):
    """Evaluate on PKU37 OCT Benchmark."""
    results = {'psnr_base': [], 'psnr_ours': [], 'ssim_base': [], 'ssim_ours': []}

    if not os.path.exists(pku_path):
        print(f"PKU37 dataset not found at {pku_path}")
        return None

    # PKU37 structure: subject folders with noisy/clean pairs
    subject_dirs = sorted(glob(os.path.join(pku_path, '*')))

    for subject_dir in tqdm(subject_dirs, desc="PKU37 Evaluation"):
        if not os.path.isdir(subject_dir):
            continue

        noisy_files = sorted(glob(os.path.join(subject_dir, 'noisy', '*.png')))

        for noisy_path in noisy_files:
            fname = os.path.basename(noisy_path)
            clean_path = os.path.join(subject_dir, 'clean', fname)

            if not os.path.exists(clean_path):
                # Try alternative naming
                clean_path = noisy_path.replace('noisy', 'clean')

            if not os.path.exists(clean_path):
                continue

            noisy = load_image(noisy_path).to(device)
            clean = load_image(clean_path).to(device)

            with torch.no_grad():
                base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
                ours_out, _ = model(noisy)

            results['psnr_base'].append(compute_psnr(base_out, clean))
            results['psnr_ours'].append(compute_psnr(ours_out, clean))
            results['ssim_base'].append(compute_ssim(base_out, clean))
            results['ssim_ours'].append(compute_ssim(ours_out, clean))

    if not results['psnr_base']:
        return None

    return {
        'psnr_base': np.mean(results['psnr_base']),
        'psnr_ours': np.mean(results['psnr_ours']),
        'psnr_gain': np.mean(results['psnr_ours']) - np.mean(results['psnr_base']),
        'ssim_base': np.mean(results['ssim_base']),
        'ssim_ours': np.mean(results['ssim_ours']),
        'ssim_gain': np.mean(results['ssim_ours']) - np.mean(results['ssim_base']),
        'n_samples': len(results['psnr_base']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backbone_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--duke_path', default='duke_datasets/organized_test_pairs/human')
    parser.add_argument('--pku_path', default='pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_file', default='real_world_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("REAL-WORLD EVALUATION")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    model = MultiTaskDenoiser(backbone_ckpt=args.backbone_ckpt).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)
    base_ckpt = torch.load(args.backbone_ckpt, map_location=args.device, weights_only=False)
    base_model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=False)
    base_model.eval()

    results = {}

    # Duke evaluation
    print("\n" + "-" * 50)
    print("Duke Human OCT")
    print("-" * 50)
    duke_results = evaluate_duke(model, base_model, args.duke_path, args.device)
    if duke_results:
        results['duke'] = duke_results
        print(f"Samples: {duke_results['n_samples']}")
        print(f"PSNR: {duke_results['psnr_base']:.2f} → {duke_results['psnr_ours']:.2f} ({duke_results['psnr_gain']:+.2f} dB)")
        print(f"SSIM: {duke_results['ssim_base']:.4f} → {duke_results['ssim_ours']:.4f} ({duke_results['ssim_gain']:+.4f})")
    else:
        print("Duke dataset not available")

    # PKU37 evaluation
    print("\n" + "-" * 50)
    print("PKU37 OCT Benchmark")
    print("-" * 50)
    pku_results = evaluate_pku(model, base_model, args.pku_path, args.device)
    if pku_results:
        results['pku37'] = pku_results
        print(f"Samples: {pku_results['n_samples']}")
        print(f"PSNR: {pku_results['psnr_base']:.2f} → {pku_results['psnr_ours']:.2f} ({pku_results['psnr_gain']:+.2f} dB)")
        print(f"SSIM: {pku_results['ssim_base']:.4f} → {pku_results['ssim_ours']:.4f} ({pku_results['ssim_gain']:+.4f})")
    else:
        print("PKU37 dataset not available")

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
EVALSCRIPT
fi

# Evaluate original checkpoint
echo "Evaluating original checkpoint on real-world data..."
python evaluate_real_world.py \
    --checkpoint "$CHECKPOINT" \
    --backbone_ckpt "$BACKBONE_CKPT" \
    --duke_path "$DUKE_PATH" \
    --pku_path "$PKU_PATH" \
    --device $DEVICE \
    --output_file "$EVAL_OUTPUT/original_results.json" \
    2>&1 | tee "$EVAL_OUTPUT/original_eval_log.txt"

# Evaluate fine-tuned checkpoint
echo ""
echo "Evaluating fine-tuned checkpoint on real-world data..."
python evaluate_real_world.py \
    --checkpoint "$FINETUNED_CKPT" \
    --backbone_ckpt "$BACKBONE_CKPT" \
    --duke_path "$DUKE_PATH" \
    --pku_path "$PKU_PATH" \
    --device $DEVICE \
    --output_file "$EVAL_OUTPUT/finetuned_results.json" \
    2>&1 | tee "$EVAL_OUTPUT/finetuned_eval_log.txt"

echo ""
echo "Phase 2 Complete: Results in $EVAL_OUTPUT"

# =============================================================================
# PHASE 3: NEURO-SYMBOLIC POST-PROCESSING
# =============================================================================
echo ""
echo "=============================================================================="
echo "PHASE 3: NEURO-SYMBOLIC POST-PROCESSING"
echo "=============================================================================="
echo ""
echo "Applying symbolic rules for:"
echo "  - Anatomical correctness (layer ordering)"
echo "  - Boundary smoothing"
echo "  - Thickness validation"
echo "  - Clinical flag generation"
echo ""

SYMBOLIC_OUTPUT="$OUTPUT_DIR/neuro_symbolic"
mkdir -p "$SYMBOLIC_OUTPUT"

# Evaluate with symbolic post-processing
python evaluate_with_symbolic.py \
    --checkpoint "$FINETUNED_CKPT" \
    --backbone_ckpt "$BACKBONE_CKPT" \
    --val_jsonl seg_data/seg_val.jsonl \
    --max_val 400 \
    --batch_size 4 \
    --device $DEVICE \
    2>&1 | tee "$SYMBOLIC_OUTPUT/symbolic_eval_log.txt"

echo ""
echo "Phase 3 Complete: Results in $SYMBOLIC_OUTPUT"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "=============================================================================="
echo "TMI COMPLETE PIPELINE - FINAL SUMMARY"
echo "=============================================================================="
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Generated Files:"
echo "  Phase 1 - Dice Fine-tuning:"
echo "    - $FINETUNE_OUTPUT/best_dice_finetuned.pth"
echo "    - $FINETUNE_OUTPUT/finetune_log.txt"
echo ""
echo "  Phase 2 - Real-World Evaluation:"
echo "    - $EVAL_OUTPUT/original_results.json"
echo "    - $EVAL_OUTPUT/finetuned_results.json"
echo ""
echo "  Phase 3 - Neuro-Symbolic:"
echo "    - $SYMBOLIC_OUTPUT/symbolic_eval_log.txt"
echo ""
echo "=============================================================================="
echo "FOR TMI PAPER - KEY RESULTS"
echo "=============================================================================="
echo ""
echo "1. DENOISING PERFORMANCE:"
echo "   - Synthetic data: Check training logs for PSNR gain"
echo "   - Duke Human: Check $EVAL_OUTPUT/finetuned_results.json"
echo "   - PKU37: Check $EVAL_OUTPUT/finetuned_results.json"
echo ""
echo "2. SEGMENTATION PERFORMANCE:"
echo "   - Original Dice: ~0.17-0.20 (from main training)"
echo "   - Fine-tuned Dice: Check $FINETUNE_OUTPUT/finetune_log.txt"
echo ""
echo "3. NEURO-SYMBOLIC CONTRIBUTION:"
echo "   - Anatomical validity improvement"
echo "   - Clinical flags generated"
echo "   - Check $SYMBOLIC_OUTPUT/symbolic_eval_log.txt"
echo ""
echo "=============================================================================="
echo "PIPELINE COMPLETE"
echo "=============================================================================="
