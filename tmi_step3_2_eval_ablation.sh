#!/bin/bash
# =============================================================================
# TMI Step 3.2: Evaluate Ablation Model (Global Gate)
# =============================================================================
# Purpose: Get ablation results to prove layer-specific gates help
# Time: ~10 minutes
# Prerequisite: Run Step 2.1 first
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI STEP 3.2: EVALUATE ABLATION MODEL"
echo "=============================================="

# Check if ablation has been run
ABLATION_DIR=$(ls -td tmi_ablation/no_layer_gates_*/checkpoints 2>/dev/null | head -1)
if [ -z "$ABLATION_DIR" ]; then
    echo "ERROR: No ablation checkpoint found!"
    echo "Please run Step 2.1 first: bash tmi_step2_1_ablation.sh"
    exit 1
fi

CHECKPOINT="$ABLATION_DIR/best_psnr.pth"
echo "Using checkpoint: $CHECKPOINT"
echo ""

python << 'EOF'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'nsnd_oct')

import os
import glob
import torch
from train_multitask_ablation import MultiTaskDenoiserAblation, validate
from train_multitask import MultiTaskOCTDataset
from torch.utils.data import DataLoader
from nsnd.models.nafnet import NAFNetFullFiLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Find latest ablation checkpoint
ckpt_dirs = sorted(glob.glob('tmi_ablation/no_layer_gates_*/checkpoints/best_psnr.pth'))
if not ckpt_dirs:
    print("ERROR: No ablation checkpoint found")
    print("Please run Step 2.1 first: bash tmi_step2_1_ablation.sh")
    exit(1)
checkpoint_path = ckpt_dirs[-1]
print(f'Checkpoint: {checkpoint_path}')

# Load ablation model
model = MultiTaskDenoiserAblation().to(device)
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
print(f'Loaded ablation model from epoch {ckpt["epoch"]}')

# Load baseline
base_model = NAFNetFullFiLM(img_channel=1, width=64, enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2], middle_blk_num=2, cond_dim=32).to(device)
base_ckpt = torch.load('outputs/nafnet_analysis_maps_w64/nafnet_best.pth', map_location=device, weights_only=False)
base_model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=False)
base_model.eval()

# Evaluate
val_ds = MultiTaskOCTDataset('seg_data/seg_val.jsonl', patch_size=64, max_samples=400, random_crop=True, ensure_all_layers=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

print("\nRunning validation...")
results = validate(model, val_loader, device, base_model)

print()
print('='*70)
print('ABLATION RESULTS (Global Gate - NO Layer-Specific)')
print('='*70)
print(f'| {"Metric":<25} | {"Baseline":<12} | {"Ablation":<12} | {"Gain":<12} |')
print(f'|{"-"*27}|{"-"*14}|{"-"*14}|{"-"*14}|')
psnr_gain = results['psnr'] - results['psnr_base']
ssim_gain = results['ssim'] - results['ssim_base']
print(f'| {"PSNR (dB)":<25} | {results["psnr_base"]:<12.2f} | {results["psnr"]:<12.2f} | +{psnr_gain:<11.2f} |')
print(f'| {"SSIM":<25} | {results["ssim_base"]:<12.4f} | {results["ssim"]:<12.4f} | +{ssim_gain:<11.4f} |')
print(f'| {"Dice":<25} | {"-":<12} | {results["dice"]:<12.4f} | {"-":<12} |')
print('='*70)

# Save results
import json
output_dir = os.path.dirname(checkpoint_path).replace('checkpoints', 'evaluation')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
    json.dump({
        'psnr_base': results['psnr_base'],
        'psnr_ablation': results['psnr'],
        'ssim_base': results['ssim_base'],
        'ssim_ablation': results['ssim'],
        'dice': results['dice'],
    }, f, indent=2)
print(f'\nResults saved to: {output_dir}/ablation_results.json')
EOF

echo ""
echo "=============================================="
echo "STEP 3.2 COMPLETE"
echo "=============================================="
echo "Next: bash tmi_step3_3_comparison_table.sh"
echo "=============================================="
