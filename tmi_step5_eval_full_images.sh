#!/bin/bash
# =============================================================================
# TMI Step 5: Full Image Evaluation (Sliding Window)
# =============================================================================
# Purpose: Evaluate all methods on FULL images using 64x64 patches with overlap
# Time: ~1 hour total
#
# This is the proper evaluation for TMI paper:
# - Processes full-resolution images (e.g., 496x512)
# - Uses 64x64 patches with 50% overlap (stride=32)
# - Weighted averaging for smooth patch blending
# - Computes metrics on full reconstructed images
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI STEP 5: FULL IMAGE EVALUATION"
echo "=============================================="
echo ""
echo "Evaluating on full-resolution images using:"
echo "  - 64x64 patch extraction"
echo "  - Stride: 32 (50% overlap)"
echo "  - Weighted averaging for smooth blending"
echo "=============================================="

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"

# Parameters
PATCH_SIZE=64
STRIDE=32  # 50% overlap
MAX_SAMPLES=100  # Adjust based on time constraints
VAL_JSONL="seg_data/seg_val.jsonl"

OUTPUT_DIR="tmi_fullimg_results"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Parameters:"
echo "  Patch size: $PATCH_SIZE"
echo "  Stride: $STRIDE ($(( 100 - 100 * STRIDE / PATCH_SIZE ))% overlap)"
echo "  Max samples: $MAX_SAMPLES"
echo "  Output: $OUTPUT_DIR/"
echo ""

# 1. Evaluate NAFNet Backbone
echo "=============================================="
echo "[1/5] Evaluating NAFNet (Backbone)"
echo "=============================================="
python eval_full_image.py \
    --model nafnet \
    --val_jsonl "$VAL_JSONL" \
    --patch_size $PATCH_SIZE \
    --stride $STRIDE \
    --max_samples $MAX_SAMPLES \
    --device $DEVICE \
    --output "$OUTPUT_DIR/nafnet.json"

# 2. Evaluate DnCNN
echo ""
echo "=============================================="
echo "[2/5] Evaluating DnCNN (Fair)"
echo "=============================================="
if [ -f "sota_baselines_fair/dncnn_fair/best.pth" ]; then
    python eval_full_image.py \
        --model dncnn \
        --val_jsonl "$VAL_JSONL" \
        --patch_size $PATCH_SIZE \
        --stride $STRIDE \
        --max_samples $MAX_SAMPLES \
        --device $DEVICE \
        --output "$OUTPUT_DIR/dncnn.json"
else
    echo "SKIPPED: DnCNN not trained. Run Step 4.1 first."
fi

# 3. Evaluate Restormer
echo ""
echo "=============================================="
echo "[3/5] Evaluating Restormer (Fair)"
echo "=============================================="
if [ -f "sota_baselines_fair/restormer_fair/best.pth" ]; then
    python eval_full_image.py \
        --model restormer \
        --val_jsonl "$VAL_JSONL" \
        --patch_size $PATCH_SIZE \
        --stride $STRIDE \
        --max_samples $MAX_SAMPLES \
        --device $DEVICE \
        --output "$OUTPUT_DIR/restormer.json"
else
    echo "SKIPPED: Restormer not trained. Run Step 4.1 first."
fi

# 4. Evaluate Ours (Ablation - Global Gate)
echo ""
echo "=============================================="
echo "[4/5] Evaluating Ours (Ablation - Global Gate)"
echo "=============================================="
ABLATION_CKPT=$(ls -t tmi_ablation/no_layer_gates_*/checkpoints/best_psnr.pth 2>/dev/null | head -1)
if [ -n "$ABLATION_CKPT" ]; then
    python eval_full_image.py \
        --model ablation \
        --val_jsonl "$VAL_JSONL" \
        --patch_size $PATCH_SIZE \
        --stride $STRIDE \
        --max_samples $MAX_SAMPLES \
        --device $DEVICE \
        --output "$OUTPUT_DIR/ablation.json"
else
    echo "SKIPPED: Ablation not trained. Run Step 2.1 first."
fi

# 5. Evaluate Ours (Full - Layer-Specific Gates)
echo ""
echo "=============================================="
echo "[5/5] Evaluating Ours (Full - Layer-Specific Gates)"
echo "=============================================="
FULL_CKPT=$(ls -t tmi_multitask/*/checkpoints/best_psnr.pth 2>/dev/null | head -1)
if [ -n "$FULL_CKPT" ]; then
    python eval_full_image.py \
        --model ours \
        --val_jsonl "$VAL_JSONL" \
        --patch_size $PATCH_SIZE \
        --stride $STRIDE \
        --max_samples $MAX_SAMPLES \
        --device $DEVICE \
        --output "$OUTPUT_DIR/ours.json"
else
    echo "SKIPPED: Full method not trained. Run Step 1.3 first."
fi

# Generate combined results table
echo ""
echo "=============================================="
echo "GENERATING FINAL COMPARISON TABLE"
echo "=============================================="

python << EOF
import json
import os
import glob

output_dir = "$OUTPUT_DIR"
results = {}

# Load all results
for f in glob.glob(os.path.join(output_dir, '*.json')):
    with open(f, 'r') as fp:
        r = json.load(fp)
        results[r['model_name']] = r

if not results:
    print("No results found!")
    exit(1)

# Sort by PSNR
sorted_results = sorted(results.items(), key=lambda x: x[1]['psnr'], reverse=True)

print()
print('='*100)
print('TABLE FOR TMI PAPER: Full Image Evaluation (Sliding Window, 50% Overlap)')
print('='*100)
print()
print(f'| {"Method":<28} | {"Params":^10} | {"PSNR (dB)":^16} | {"SSIM":^18} |')
print(f'|{"-"*30}|{"-"*12}|{"-"*18}|{"-"*20}|')

for method, r in sorted_results:
    params_str = f'{r["params"]/1e6:.1f}M'
    psnr_str = f'{r["psnr"]:.2f} ± {r["psnr_std"]:.2f}'
    ssim_str = f'{r["ssim"]:.4f} ± {r["ssim_std"]:.4f}'
    print(f'| {method:<28} | {params_str:^10} | {psnr_str:^16} | {ssim_str:^18} |')

print('='*100)

# Key findings
print()
print('KEY FINDINGS (Full Image Evaluation):')
print('-'*60)

if 'Ours (Layer-Specific)' in results:
    ours = results['Ours (Layer-Specific)']

    for baseline in ['NAFNet (Backbone)', 'DnCNN (Fair)', 'Restormer (Fair)', 'Ours (Global Gate)']:
        if baseline in results:
            gain = ours['psnr'] - results[baseline]['psnr']
            print(f'  vs {baseline}: +{gain:.2f} dB')

print()
print('Note: Evaluated on full-resolution images with 64x64 patches, 50% overlap.')
print('='*100)

# Save combined results
combined = {'results': results, 'settings': {'patch_size': $PATCH_SIZE, 'stride': $STRIDE}}
with open(os.path.join(output_dir, 'combined_results.json'), 'w') as f:
    json.dump(combined, f, indent=2)
print(f'Combined results saved to: {output_dir}/combined_results.json')
EOF

echo ""
echo "=============================================="
echo "STEP 5 COMPLETE"
echo "=============================================="
echo "Full image evaluation results saved to: $OUTPUT_DIR/"
echo "=============================================="
