#!/bin/bash
#
# Master script to retrain all models with Duke-learned realistic noise composition
#
# This script:
# 1. Generates Duke-tuned training/val/test data
# 2. Retrains NAFNet with realistic noise
# 3. Retrains U-Net with realistic noise
# 4. (Optional) Retrains NSND with realistic noise
# 5. Evaluates all models on Duke dataset
#
# Expected improvements:
#   - Duke synthetic: 25.74 → 29 dB (+3-4 dB)
#   - Duke human OCT: 23.03 → 26 dB (+2-4 dB)
#

set -e  # Exit on error

echo "================================================================================"
echo "RETRAINING PIPELINE: DUKE-TUNED REALISTIC NOISE"
echo "================================================================================"
echo ""
echo "Duke-learned composition: Speckle 83.8%, Banding 4.3%, Gaussian 4.5%, Shot 7.4%"
echo ""
echo "================================================================================"
echo ""

# Configuration
SPLITS_JSON="/home/kumwilai/OCT/oct_splits_tmi.json"
DATA_ROOT="/home/kumwilai/OCT/oct_tmi"
NSND_ROOT="/home/kumwilai/OCT/nsnd_oct"
SEED=123

# Step 1: Generate Duke-tuned training data
echo "================================================================================"
echo "STEP 1: Generate Duke-Tuned Training Data"
echo "================================================================================"
echo ""

cd /home/kumwilai/OCT

# Generate training pairs
python scripts/generate_duke_tuned_pairs.py \
    --splits_json ${SPLITS_JSON} \
    --split train \
    --noisy_name noisy_duke_tuned \
    --param_scale 1.0 \
    --seed ${SEED} \
    --pairs_out /home/kumwilai/OCT/train_pairs_duke_tuned.txt \
    --weights_out /home/kumwilai/OCT/weights_duke_tuned_train.jsonl \
    --overwrite

echo ""
echo "✓ Training data generated"
echo ""

# Generate validation pairs
python scripts/generate_duke_tuned_pairs.py \
    --splits_json ${SPLITS_JSON} \
    --split val \
    --noisy_name noisy_duke_tuned \
    --param_scale 1.0 \
    --seed ${SEED} \
    --pairs_out /home/kumwilai/OCT/val_pairs_duke_tuned.txt \
    --weights_out /home/kumwilai/OCT/weights_duke_tuned_val.jsonl \
    --overwrite

echo ""
echo "✓ Validation data generated"
echo ""

# Generate test pairs
python scripts/generate_duke_tuned_pairs.py \
    --splits_json ${SPLITS_JSON} \
    --split test \
    --noisy_name noisy_duke_tuned \
    --param_scale 1.0 \
    --seed ${SEED} \
    --pairs_out /home/kumwilai/OCT/test_pairs_duke_tuned.txt \
    --weights_out /home/kumwilai/OCT/weights_duke_tuned_test.jsonl \
    --overwrite

echo ""
echo "✓ Test data generated"
echo ""

# Step 2: Retrain NAFNet
echo "================================================================================"
echo "STEP 2: Retrain NAFNet (width=32) with Duke-Tuned Noise"
echo "================================================================================"
echo ""

cd ${NSND_ROOT}

python scripts/train_nafnet_on_synthetic.py \
    --data_root ${DATA_ROOT} \
    --max_samples 1000 \
    --val_samples 200 \
    --crop_size 64 \
    --width 32 \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --seed ${SEED} \
    --out_path checkpoints/nafnet_w32_duke_tuned_best.pth

echo ""
echo "✓ NAFNet retraining complete"
echo ""

# Step 3: Retrain U-Net
echo "================================================================================"
echo "STEP 3: Retrain U-Net (features=32) with Duke-Tuned Noise"
echo "================================================================================"
echo ""

python scripts/train_unet_on_synthetic.py \
    --data_root ${DATA_ROOT} \
    --max_samples 1000 \
    --val_samples 200 \
    --crop_size 64 \
    --features 32 \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --seed ${SEED} \
    --out_path checkpoints/unet_f32_duke_tuned_best.pth

echo ""
echo "✓ U-Net retraining complete"
echo ""

# Step 4: (Optional) Retrain NSND
echo "================================================================================"
echo "STEP 4: (Optional) Retrain NSND with Duke-Tuned Noise"
echo "================================================================================"
echo ""
echo "NSND retraining requires hybrid analyzer checkpoint."
echo "If you have trained the analyzer, uncomment the section below."
echo ""

# Uncomment to retrain NSND:
# python scripts/train_hybrid_nsnd_multitask.py \
#     --data_root ${DATA_ROOT} \
#     --max_samples 1000 --val_samples 200 \
#     --noise_mode realistic --alpha 0.2 --param_scale 1.0 \
#     --shared_residual --shared_trunk_width 24 \
#     --shared_adapter_channels 8 --shared_adapter_hidden 8 \
#     --base_nafnet_width 32 --stage1_l1_only --freeze_analyzer_epochs 5 \
#     --noise_cycle_weight 0.01 --noise_cycle_use_true --noise_cycle_banding_freq 20 \
#     --ns_use_neural_predicates --ns_use_neural_weights \
#     --hybrid_analyzer_ckpt checkpoints/hybrid_cnn_symbolic.pth \
#     --seed ${SEED} \
#     --out_path checkpoints/nsnd_duke_tuned_best.pth

echo "Skipping NSND (uncomment in script to enable)"
echo ""

# Step 5: Evaluate on Duke dataset
echo "================================================================================"
echo "STEP 5: Evaluate Models on Duke OCT Dataset"
echo "================================================================================"
echo ""

# Evaluate NAFNet
echo "Evaluating NAFNet..."
python scripts/evaluate_duke_baselines.py \
    --model_type nafnet \
    --checkpoint checkpoints/nafnet_w32_duke_tuned_best.pth \
    --width 32 \
    --patch_size 64 \
    --stride 48 \
    --test_both \
    --results_json results/duke_nafnet_duke_tuned_results.json

echo ""
echo "✓ NAFNet evaluation complete"
echo ""

# Evaluate U-Net
echo "Evaluating U-Net..."
python scripts/evaluate_duke_baselines.py \
    --model_type unet \
    --checkpoint checkpoints/unet_f32_duke_tuned_best.pth \
    --features 32 \
    --patch_size 64 \
    --stride 48 \
    --test_both \
    --results_json results/duke_unet_duke_tuned_results.json

echo ""
echo "✓ U-Net evaluation complete"
echo ""

# Step 6: Compare results
echo "================================================================================"
echo "STEP 6: Compare Original vs Duke-Tuned Results"
echo "================================================================================"
echo ""

python << 'EOF'
import json
from pathlib import Path

print("="*80)
print("RESULTS COMPARISON: Original (α=0.2) vs Duke-Tuned")
print("="*80)
print()

# Load results
results_dir = Path("results")

try:
    with open(results_dir / "duke_nafnet_results.json", 'r') as f:
        nafnet_orig = json.load(f)
    with open(results_dir / "duke_nafnet_duke_tuned_results.json", 'r') as f:
        nafnet_tuned = json.load(f)

    with open(results_dir / "duke_unet_results.json", 'r') as f:
        unet_orig = json.load(f)
    with open(results_dir / "duke_unet_duke_tuned_results.json", 'r') as f:
        unet_tuned = json.load(f)

    print("SYNTHETIC DATA (18 pairs)")
    print("-"*80)
    print(f"{'Model':<20} {'Original (α=0.2)':<20} {'Duke-Tuned':<20} {'Improvement':<15}")
    print("-"*80)

    nafnet_orig_syn = nafnet_orig['results']['synthetic']['psnr']['mean']
    nafnet_tuned_syn = nafnet_tuned['results']['synthetic']['psnr']['mean']
    nafnet_gain = nafnet_tuned_syn - nafnet_orig_syn
    print(f"{'NAFNet-w32':<20} {nafnet_orig_syn:<20.2f} {nafnet_tuned_syn:<20.2f} {f'+{nafnet_gain:.2f} dB':<15}")

    unet_orig_syn = unet_orig['results']['synthetic']['psnr']['mean']
    unet_tuned_syn = unet_tuned['results']['synthetic']['psnr']['mean']
    unet_gain = unet_tuned_syn - unet_orig_syn
    print(f"{'U-Net-f32':<20} {unet_orig_syn:<20.2f} {unet_tuned_syn:<20.2f} {f'+{unet_gain:.2f} dB':<15}")

    print()
    print("HUMAN OCT DATA (39 pairs)")
    print("-"*80)
    print(f"{'Model':<20} {'Original (α=0.2)':<20} {'Duke-Tuned':<20} {'Improvement':<15}")
    print("-"*80)

    nafnet_orig_hum = nafnet_orig['results']['human']['psnr']['mean']
    nafnet_tuned_hum = nafnet_tuned['results']['human']['psnr']['mean']
    nafnet_gain_hum = nafnet_tuned_hum - nafnet_orig_hum
    print(f"{'NAFNet-w32':<20} {nafnet_orig_hum:<20.2f} {nafnet_tuned_hum:<20.2f} {f'+{nafnet_gain_hum:.2f} dB':<15}")

    unet_orig_hum = unet_orig['results']['human']['psnr']['mean']
    unet_tuned_hum = unet_tuned['results']['human']['psnr']['mean']
    unet_gain_hum = unet_tuned_hum - unet_orig_hum
    print(f"{'U-Net-f32':<20} {unet_orig_hum:<20.2f} {unet_tuned_hum:<20.2f} {f'+{unet_gain_hum:.2f} dB':<15}")

    print()
    print("="*80)
    print(f"Average improvement: +{(nafnet_gain + unet_gain + nafnet_gain_hum + unet_gain_hum)/4:.2f} dB")
    print("="*80)

except FileNotFoundError as e:
    print(f"Could not load results: {e}")
    print("Make sure both original and Duke-tuned evaluations have completed.")

EOF

echo ""
echo "================================================================================"
echo "RETRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "Models trained with Duke-learned noise composition:"
echo "  - checkpoints/nafnet_w32_duke_tuned_best.pth"
echo "  - checkpoints/unet_f32_duke_tuned_best.pth"
echo ""
echo "Evaluation results:"
echo "  - results/duke_nafnet_duke_tuned_results.json"
echo "  - results/duke_unet_duke_tuned_results.json"
echo ""
echo "Expected improvements: +3-4 dB on Duke dataset"
echo "================================================================================"
