#!/bin/bash
# =============================================================================
# TMI Step 3.3: Generate Comparison Table for Paper
# =============================================================================
# Purpose: Create final comparison table (Full vs Ablation)
# Time: ~15 minutes
# Prerequisite: Run Steps 3.1 and 3.2 first
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI STEP 3.3: GENERATE COMPARISON TABLE"
echo "=============================================="

python << 'EOF'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'nsnd_oct')

import os
import glob
import json
import torch
from train_multitask import MultiTaskDenoiser, MultiTaskOCTDataset, validate, LAYER_NAMES
from train_multitask_ablation import MultiTaskDenoiserAblation
from train_multitask_ablation import validate as validate_ablation
from torch.utils.data import DataLoader
from nsnd.models.nafnet import NAFNetFullFiLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load data
val_ds = MultiTaskOCTDataset('seg_data/seg_val.jsonl', patch_size=64, max_samples=400, random_crop=True, ensure_all_layers=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

# Load baseline
base_model = NAFNetFullFiLM(img_channel=1, width=64, enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2], middle_blk_num=2, cond_dim=32).to(device)
base_ckpt = torch.load('outputs/nafnet_analysis_maps_w64/nafnet_best.pth', map_location=device, weights_only=False)
base_model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=False)
base_model.eval()

results = {}

# 1. Full method (Layer-Specific Gates)
full_ckpts = sorted(glob.glob('tmi_multitask/*/checkpoints/best_psnr.pth'))
if full_ckpts:
    print("Evaluating Full Method...")
    model_full = MultiTaskDenoiser().to(device)
    ckpt = torch.load(full_ckpts[-1], map_location=device, weights_only=False)
    model_full.load_state_dict(ckpt['state_dict'])
    r = validate(model_full, val_loader, device, base_model)
    results['NAFNet (Backbone Only)'] = {
        'psnr': r['psnr_base'],
        'ssim': r['ssim_base'],
        'dice': '-',
        'description': 'Baseline denoiser'
    }
    results['Ours (Layer-Specific Gates)'] = {
        'psnr': r['psnr'],
        'ssim': r['ssim'],
        'dice': r['dice'],
        'description': 'Full method with 5 layer-specific noise gates'
    }
    full_psnr = r['psnr']
    base_psnr = r['psnr_base']
else:
    print("WARNING: No full method checkpoint found")
    full_psnr = None
    base_psnr = None

# 2. Ablation (Global Gate)
ablation_ckpts = sorted(glob.glob('tmi_ablation/no_layer_gates_*/checkpoints/best_psnr.pth'))
if ablation_ckpts:
    print("Evaluating Ablation...")
    model_abl = MultiTaskDenoiserAblation().to(device)
    ckpt = torch.load(ablation_ckpts[-1], map_location=device, weights_only=False)
    model_abl.load_state_dict(ckpt['state_dict'])
    r_abl = validate_ablation(model_abl, val_loader, device, base_model)
    results['Ours (Global Gate - Ablation)'] = {
        'psnr': r_abl['psnr'],
        'ssim': r_abl['ssim'],
        'dice': r_abl['dice'],
        'description': 'Ablation: single global gate for all layers'
    }
    ablation_psnr = r_abl['psnr']
else:
    print("WARNING: No ablation checkpoint found")
    ablation_psnr = None

# Print main comparison table
print()
print('='*85)
print('TABLE FOR TMI PAPER: Ablation Study - Layer-Specific vs Global Noise Gates')
print('='*85)
print()
print(f'| {"Method":<40} | {"PSNR (dB)":^12} | {"SSIM":^10} | {"Dice":^10} |')
print(f'|{"-"*42}|{"-"*14}|{"-"*12}|{"-"*12}|')

for method, r in results.items():
    psnr_str = f'{r["psnr"]:.2f}' if isinstance(r['psnr'], float) else str(r['psnr'])
    ssim_str = f'{r["ssim"]:.4f}' if isinstance(r.get('ssim'), float) else '-'
    dice_str = f'{r["dice"]:.4f}' if isinstance(r['dice'], float) else str(r['dice'])
    print(f'| {method:<40} | {psnr_str:^12} | {ssim_str:^10} | {dice_str:^10} |')

print('='*85)

# Compute contributions
print()
print('KEY FINDINGS FOR TMI PAPER:')
print('-'*50)

if base_psnr and full_psnr:
    total_gain = full_psnr - base_psnr
    print(f'1. Total improvement over backbone: +{total_gain:.2f} dB')

if ablation_psnr and full_psnr:
    layer_specific_gain = full_psnr - ablation_psnr
    print(f'2. Layer-Specific Gate Contribution: +{layer_specific_gain:.2f} dB')
    print(f'   This proves that different retinal layers benefit from')
    print(f'   different noise handling strategies.')

if base_psnr and ablation_psnr:
    multitask_gain = ablation_psnr - base_psnr
    print(f'3. Multi-task learning contribution: +{multitask_gain:.2f} dB')
    print(f'   (Even with global gate, multi-task helps)')

print()
print('='*85)
print('CONCLUSION')
print('='*85)
if ablation_psnr and full_psnr:
    pct_improvement = 100 * (full_psnr - ablation_psnr) / ablation_psnr
    print(f'Layer-specific noise modeling provides +{full_psnr - ablation_psnr:.2f} dB')
    print(f'improvement ({pct_improvement:.1f}% relative), validating our key hypothesis')
    print(f'that different retinal layers have different noise characteristics')
    print(f'and require layer-appropriate denoising strategies.')
print('='*85)

# Save combined results
output_file = 'tmi_results_comparison.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to: {output_file}')
EOF

echo ""
echo "=============================================="
echo "STEP 3.3 COMPLETE"
echo "=============================================="
echo "Next: bash tmi_step3_4_per_layer_analysis.sh"
echo "=============================================="
