#!/bin/bash
# Fair comparison between CASA adaptive denoiser and baseline filters
# Uses same test data (24 validation images) at 64x64 resolution

set -e

echo "======================================================================"
echo "Fair Comparison: CASA Adaptive Denoiser vs Baseline Filters"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - Test data: oct/normal/val (24 images)"
echo "  - Image size: 64x64"
echo "  - Metrics: PSNR, SSIM"
echo ""

# Create output directories
mkdir -p ./outputs/comparison
mkdir -p ./checkpoints/casa_comparison

echo "======================================================================"
echo "Step 1: Training and evaluating CASA model"
echo "======================================================================"
echo ""

# Run CASA training with evaluation
python -u adaptive_oct_denoise.py \
    --clean_root oct/normal/train/clean \
    --num_meta_epochs 2 \
    --num_tasks_per_meta_batch 3 \
    --inner_steps 5 \
    --inner_lr 3e-4 \
    --meta_step_size 0.2 \
    --batch_size 4 \
    --resize_h 64 \
    --resize_w 64 \
    --base_channels 32 \
    --adapter casa \
    --residual_mode \
    --use_extended_noise \
    --physics_lambda_depth 0.3 \
    --physics_lambda_ascan 0.2 \
    --physics_lambda_speckle 0.1 \
    --physics_psf_sigma 2.0 \
    --physics_patch_size 16 \
    --memory_safe \
    --meta_eval_pairs oct_val_pairs_24.txt \
    --meta_eval_limit 24 \
    --fast_eval \
    --fast_ssim_down 2 \
    --fast_cache_pairs \
    --meta_progress \
    --meta_step_progress \
    --output_dir ./checkpoints/casa_comparison \
    2>&1 | tee ./outputs/comparison/casa_training.log

echo ""
echo "======================================================================"
echo "Step 2: Evaluating baseline filters"
echo "======================================================================"
echo ""

# Run baseline filters on validation set
# Note: We need to construct the domain path correctly for the baseline script
python scripts/run_baselines_filters.py \
    --domains oct/normal \
    --image_size 64 \
    --limit 24 \
    --filters lee kuan frost bilateral guided \
    --win 5 \
    --use_enl_map \
    --output_dir ./outputs/comparison/baselines \
    2>&1 | tee ./outputs/comparison/baselines_eval.log

echo ""
echo "======================================================================"
echo "Step 3: Generating comparison report"
echo "======================================================================"
echo ""

# Create a simple comparison report
python -c "
import pandas as pd
import os

print('\\n' + '='*80)
print('COMPARISON RESULTS: CASA vs Baseline Filters')
print('='*80)
print()

# Check if baseline results exist
baseline_csv = './outputs/comparison/baselines/results.csv'
if os.path.exists(baseline_csv):
    df = pd.read_csv(baseline_csv)

    # Group by method and compute average metrics
    summary = df.groupby('method').agg({
        'psnr': 'mean',
        'ssim': 'mean',
        'psnr_gain': 'mean',
        'ssim_gain': 'mean',
        'time_ms': 'mean'
    }).round(4)

    print('Baseline Filters Performance:')
    print('-'*80)
    print(f\"{'Method':<15} {'PSNR (dB)':<12} {'SSIM':<12} {'Time (ms)':<12}\")
    print('-'*80)
    for method, row in summary.iterrows():
        print(f\"{method.upper():<15} {row['psnr']:>8.2f}    {row['ssim']:>8.4f}    {row['time_ms']:>8.1f}\")
    print()

    # Save summary
    summary.to_csv('./outputs/comparison/baseline_summary.csv')
    print('Baseline summary saved to: ./outputs/comparison/baseline_summary.csv')
else:
    print('Baseline results not found!')

print()
print('CASA model results are in: ./checkpoints/casa_comparison/')
print('Check the training log for CASA performance metrics.')
print()
print('='*80)
"

echo ""
echo "======================================================================"
echo "Comparison complete!"
echo "======================================================================"
echo ""
echo "Results location:"
echo "  - CASA training log: ./outputs/comparison/casa_training.log"
echo "  - Baseline results:  ./outputs/comparison/baselines/results.csv"
echo "  - Baseline summary:  ./outputs/comparison/baseline_summary.csv"
echo ""
