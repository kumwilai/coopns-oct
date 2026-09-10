#!/bin/bash
#
# Optimized NSND Retraining to Surpass All Baselines
#
# This script leverages all NSND contributions:
# 1. Duke-learned realistic noise composition
# 2. Increased denoiser capacity (width 16→24)
# 3. Improved analyzer training
# 4. Optimized multi-task learning
# 5. Longer training with more data
#
# Expected improvements:
#   - Duke synthetic: 25.46 → 27-28 dB (+1.5-2.5 dB)
#   - Duke human OCT: 22.90 → 24-25 dB (+1-2 dB)
#   - Internal test: TBD → 34-35 dB (beat U-Net's 33.60 dB)
#

set -e

echo "================================================================================"
echo "NSND OPTIMIZED RETRAINING - LEVERAGE ALL CONTRIBUTIONS"
echo "================================================================================"
echo ""
echo "Novel contributions being leveraged:"
echo "  1. Hybrid CNN-Symbolic Noise Analyzer"
echo "  2. Noise-Specific Denoisers (increased capacity)"
echo "  3. Multi-Task Learning (optimized)"
echo "  4. Duke-Learned Realistic Noise (Speckle 83.8%)"
echo "  5. Neuro-Symbolic Reasoning"
echo ""
echo "================================================================================"
echo ""

DATA_ROOT="/home/kumwilai/OCT/oct_tmi"
NSND_ROOT="/home/kumwilai/OCT/nsnd_oct"
SEED=123

# Step 1: Generate Duke-tuned training data (if not exists)
echo "================================================================================"
echo "STEP 1: Generate Duke-Tuned Training Data"
echo "================================================================================"
echo ""

cd /home/kumwilai/OCT

if [ ! -f "train_pairs_duke_tuned.txt" ]; then
    python scripts/generate_duke_tuned_pairs.py \
        --splits_json oct_splits_tmi.json \
        --split train \
        --noisy_name noisy_duke_tuned \
        --seed ${SEED} \
        --pairs_out train_pairs_duke_tuned.txt \
        --weights_out weights_duke_tuned_train.jsonl \
        --overwrite
    echo "✓ Training data generated"
else
    echo "✓ Training data already exists"
fi

if [ ! -f "val_pairs_duke_tuned.txt" ]; then
    python scripts/generate_duke_tuned_pairs.py \
        --splits_json oct_splits_tmi.json \
        --split val \
        --noisy_name noisy_duke_tuned \
        --seed ${SEED} \
        --pairs_out val_pairs_duke_tuned.txt \
        --weights_out weights_duke_tuned_val.jsonl \
        --overwrite
    echo "✓ Validation data generated"
else
    echo "✓ Validation data already exists"
fi

echo ""

# Step 2: Retrain Analyzer with Duke-learned noise (OPTIONAL - improves accuracy)
echo "================================================================================"
echo "STEP 2: (Optional) Retrain Analyzer with Duke-Learned Noise"
echo "================================================================================"
echo ""
echo "Using existing analyzer: checkpoints/hybrid_cnn_symbolic_duke_joint_seed0.pth"
echo "To retrain analyzer with duke-tuned data, uncomment the section below"
echo ""

# Uncomment to retrain analyzer:
# cd ${NSND_ROOT}
# python scripts/train_hybrid_analyzer.py \
#     --data_root ${DATA_ROOT} \
#     --noisy_folder noisy_duke_tuned \
#     --weights_jsonl /home/kumwilai/OCT/weights_duke_tuned_train.jsonl \
#     --max_samples 4000 --val_samples 800 \
#     --crop_size 64 --batch_size 32 \
#     --epochs 50 --lr 1e-4 \
#     --seed ${SEED} \
#     --out_path checkpoints/hybrid_analyzer_duke_tuned_optimized.pth

# Step 3: Retrain NSND with Optimized Configuration
echo "================================================================================"
echo "STEP 3: Retrain NSND with Optimized Configuration"
echo "================================================================================"
echo ""

cd ${NSND_ROOT}

echo "Training configuration:"
echo "  - Data: Duke-tuned realistic noise (Speckle 83.8%)"
echo "  - Samples: 4000 train, 800 val (increased from 2000/400)"
echo "  - NAFNet width: 24 (increased from 16)"
echo "  - Base NAFNet width: 32"
echo "  - Shared trunk width: 24"
echo "  - Epochs: 100 (increased from 50)"
echo "  - Noise cycle weight: 0.05 (increased from 0.01)"
echo "  - Pure noise augmentation: 10%"
echo ""

python scripts/train_hybrid_nsnd_multitask.py \
    --data_root ${DATA_ROOT} \
    --noisy_folder noisy_duke_tuned \
    --max_samples 4000 --val_samples 800 \
    --crop_size 64 --batch_size 16 \
    --noise_mode realistic --alpha 0.2 --param_scale 1.0 \
    --shared_residual --shared_trunk_width 24 \
    --shared_adapter_channels 8 --shared_adapter_hidden 8 \
    --base_nafnet_width 32 --nafnet_width 24 \
    --stage1_l1_only --freeze_analyzer_epochs 5 \
    --noise_cycle_weight 0.05 --noise_cycle_use_true --noise_cycle_banding_freq 20 \
    --ns_use_neural_predicates --ns_use_neural_weights \
    --pure_noise_prob 0.1 --pure_noise_epsilon 0.02 \
    --hybrid_analyzer_ckpt /home/kumwilai/OCT/checkpoints/hybrid_cnn_symbolic_duke_joint_seed0.pth \
    --epochs 100 --lr 1e-4 \
    --seed ${SEED} --log_every 20 \
    --out_path checkpoints/nsnd_duke_tuned_optimized_best.pth

echo ""
echo "✓ NSND retraining complete"
echo ""

# Step 4: Evaluate on Duke Dataset
echo "================================================================================"
echo "STEP 4: Evaluate on Duke OCT Dataset"
echo "================================================================================"
echo ""

# Evaluate on Duke Synthetic
echo "Evaluating on Duke Synthetic (18 pairs)..."
python scripts/evaluate_nsnd_fixed_pairs.py \
    --checkpoint checkpoints/nsnd_duke_tuned_optimized_best.pth \
    --pairs /home/kumwilai/OCT/duke_datasets/organized_test_pairs/test_pairs_synthetic.txt \
    --hybrid_analyzer_ckpt /home/kumwilai/OCT/checkpoints/hybrid_cnn_symbolic_duke_joint_seed0.pth \
    --out_json results/nsnd_duke_tuned_optimized_synthetic.json

echo ""
echo "✓ Duke Synthetic evaluation complete"
echo ""

# Evaluate on Duke Human
echo "Evaluating on Duke Human OCT (39 pairs)..."
python scripts/evaluate_nsnd_fixed_pairs.py \
    --checkpoint checkpoints/nsnd_duke_tuned_optimized_best.pth \
    --pairs /home/kumwilai/OCT/duke_datasets/organized_test_pairs/test_pairs_human.txt \
    --hybrid_analyzer_ckpt /home/kumwilai/OCT/checkpoints/hybrid_cnn_symbolic_duke_joint_seed0.pth \
    --out_json results/nsnd_duke_tuned_optimized_human.json

echo ""
echo "✓ Duke Human OCT evaluation complete"
echo ""

# Step 5: Compare Results
echo "================================================================================"
echo "STEP 5: Compare Results with Baselines"
echo "================================================================================"
echo ""

python << 'EOF'
import json
from pathlib import Path

print("=" * 80)
print("RESULTS COMPARISON: Original vs Optimized NSND")
print("=" * 80)
print()

# Load results
results_dir = Path("results")

# Original NSND
orig_syn = json.load(open(results_dir / "nsnd_duke_synthetic_test.json"))
orig_hum = json.load(open(results_dir / "nsnd_duke_human.json"))

# Optimized NSND
try:
    opt_syn = json.load(open(results_dir / "nsnd_duke_tuned_optimized_synthetic.json"))
    opt_hum = json.load(open(results_dir / "nsnd_duke_tuned_optimized_human.json"))

    print("DUKE SYNTHETIC (18 pairs)")
    print("-" * 80)
    print(f"{'Model':<30} {'PSNR':<15} {'SSIM':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'NAFNet-w32 (baseline)':<30} {'25.74 dB':<15} {'':<15} {'':<15}")
    print(f"{'U-Net-f32 (baseline)':<30} {'25.12 dB':<15} {'':<15} {'':<15}")
    print(f"{'NSND (original)':<30} {f'{orig_syn['psnr_mean']:.2f} dB':<15} {f'{orig_syn['ssim_mean']:.4f}':<15} {'':<15}")
    print(f"{'NSND (optimized)':<30} {f'{opt_syn['psnr_mean']:.2f} dB':<15} {f'{opt_syn['ssim_mean']:.4f}':<15} {f'+{opt_syn['psnr_mean']-orig_syn['psnr_mean']:.2f} dB':<15}")
    print()

    print("DUKE HUMAN OCT (39 pairs)")
    print("-" * 80)
    print(f"{'Model':<30} {'PSNR':<15} {'SSIM':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'NAFNet-w32 (baseline)':<30} {'23.03 dB':<15} {'':<15} {'':<15}")
    print(f"{'U-Net-f32 (baseline)':<30} {'22.84 dB':<15} {'':<15} {'':<15}")
    print(f"{'NSND (original)':<30} {f'{orig_hum['psnr_mean']:.2f} dB':<15} {f'{orig_hum['ssim_mean']:.4f}':<15} {'':<15}")
    print(f"{'NSND (optimized)':<30} {f'{opt_hum['psnr_mean']:.2f} dB':<15} {f'{opt_hum['ssim_mean']:.4f}':<15} {f'+{opt_hum['psnr_mean']-orig_hum['psnr_mean']:.2f} dB':<15}")
    print()

    print("=" * 80)
    if opt_syn['psnr_mean'] > 25.74 and opt_hum['psnr_mean'] > 23.03:
        print("SUCCESS! NSND now surpasses all baselines! 🎉")
    elif opt_syn['psnr_mean'] > orig_syn['psnr_mean']:
        print(f"IMPROVEMENT! NSND gained +{opt_syn['psnr_mean']-orig_syn['psnr_mean']:.2f} dB")
    else:
        print("Results ready - check if improvements meet expectations")
    print("=" * 80)

except FileNotFoundError as e:
    print(f"Optimized results not found yet: {e}")
    print("Training may still be in progress...")

EOF

echo ""
echo "================================================================================"
echo "RETRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "Key improvements applied:"
echo "  ✓ Duke-learned realistic noise composition"
echo "  ✓ Increased denoiser capacity (width 16→24)"
echo "  ✓ More training data (2000→4000 samples)"
echo "  ✓ Longer training (50→100 epochs)"
echo "  ✓ Stronger noise cycle consistency (0.01→0.05)"
echo "  ✓ Pure noise augmentation (10%)"
echo ""
echo "Checkpoints:"
echo "  - checkpoints/nsnd_duke_tuned_optimized_best.pth"
echo ""
echo "Results:"
echo "  - results/nsnd_duke_tuned_optimized_synthetic.json"
echo "  - results/nsnd_duke_tuned_optimized_human.json"
echo ""
echo "Expected improvements: +1.5-2.5 dB on Duke dataset"
echo "================================================================================"
