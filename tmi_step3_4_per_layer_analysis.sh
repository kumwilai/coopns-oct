#!/bin/bash
# =============================================================================
# TMI Step 3.4: Per-Layer Analysis (Detailed Results)
# =============================================================================
# Purpose: Generate per-layer performance table for TMI paper
# Time: ~15 minutes
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI STEP 3.4: PER-LAYER ANALYSIS"
echo "=============================================="

python << 'EOF'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'nsnd_oct')

import os
import glob
import json
import torch
from train_multitask import MultiTaskDenoiser, validate_per_layer, LAYER_NAMES
from nsnd.models.nafnet import NAFNetFullFiLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Find latest checkpoint
ckpt_dirs = sorted(glob.glob('tmi_multitask/*/checkpoints/best_psnr.pth'))
if not ckpt_dirs:
    print("ERROR: No checkpoint found")
    exit(1)
checkpoint_path = ckpt_dirs[-1]
print(f'Checkpoint: {checkpoint_path}')

# Load models
model = MultiTaskDenoiser().to(device)
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)  # strict=False for older checkpoints

base_model = NAFNetFullFiLM(img_channel=1, width=64, enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2], middle_blk_num=2, cond_dim=32).to(device)
base_ckpt = torch.load('outputs/nafnet_analysis_maps_w64/nafnet_best.pth', map_location=device, weights_only=False)
base_model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=False)
base_model.eval()

# Per-layer evaluation (uses targeted 64x64 crops centered on each layer)
print("\nRunning per-layer targeted evaluation...")
print("(Each layer evaluated with 64x64 patches centered on that layer)")
results = validate_per_layer(model, 'seg_data/seg_val.jsonl', device, base_model, patch_size=64, max_samples=100)

# Table 1: Per-Layer PSNR
print()
print('='*80)
print('TABLE: Per-Layer Denoising Performance (PSNR)')
print('='*80)
print()
print(f'| {"Retinal Layer":<15} | {"Base PSNR":^12} | {"Our PSNR":^12} | {"Improvement":^12} | {"N":^6} |')
print(f'|{"-"*17}|{"-"*14}|{"-"*14}|{"-"*14}|{"-"*8}|')

total_gain = 0
count = 0
for name in LAYER_NAMES:
    r = results[name]
    if r['psnr_base'] and r['psnr_ours']:
        base = f'{r["psnr_base"]:.2f}'
        ours = f'{r["psnr_ours"]:.2f}'
        gain = f'+{r["psnr_gain"]:.2f}'
        total_gain += r["psnr_gain"]
        count += 1
    else:
        base = 'N/A'
        ours = 'N/A'
        gain = 'N/A'
    n = r['n_samples']
    print(f'| {name:<15} | {base:^12} | {ours:^12} | {gain:^12} | {n:^6} |')

print(f'|{"-"*17}|{"-"*14}|{"-"*14}|{"-"*14}|{"-"*8}|')
if count > 0:
    avg_gain = total_gain / count
    print(f'| {"Average":<15} | {"-":^12} | {"-":^12} | {f"+{avg_gain:.2f}":^12} | {"-":^6} |')
print('='*80)

# Table 2: Per-Layer SSIM
print()
print('='*70)
print('TABLE: Per-Layer Denoising Performance (SSIM)')
print('='*70)
print()
print(f'| {"Retinal Layer":<15} | {"Base SSIM":^12} | {"Our SSIM":^12} | {"Improvement":^12} |')
print(f'|{"-"*17}|{"-"*14}|{"-"*14}|{"-"*14}|')

for name in LAYER_NAMES:
    r = results[name]
    if r['ssim_base'] and r['ssim_ours']:
        base = f'{r["ssim_base"]:.4f}'
        ours = f'{r["ssim_ours"]:.4f}'
        gain = f'+{r["ssim_gain"]:.4f}'
    else:
        base = 'N/A'
        ours = 'N/A'
        gain = 'N/A'
    print(f'| {name:<15} | {base:^12} | {ours:^12} | {gain:^12} |')
print('='*70)

# Table 3: Per-Layer Segmentation Dice
print()
print('='*50)
print('TABLE: Per-Layer Segmentation Accuracy')
print('='*50)
print()
print(f'| {"Retinal Layer":<15} | {"Dice Score":^12} | {"N":^8} |')
print(f'|{"-"*17}|{"-"*14}|{"-"*10}|')

for name in LAYER_NAMES:
    r = results[name]
    dice = f'{r["dice"]:.4f}' if r['dice'] else 'N/A'
    n = r['n_samples']
    print(f'| {name:<15} | {dice:^12} | {n:^8} |')
print('='*50)

# Clinical interpretation
print()
print('='*70)
print('CLINICAL INTERPRETATION')
print('='*70)
print()
print('Layer-specific noise characteristics:')
print('  - RNFL_GCL: Nerve fiber layer - high reflectivity, needs aggressive denoising')
print('  - INL_OPL: Inner nuclear layers - moderate signal')
print('  - ONL: Outer nuclear layer - lower signal, photoreceptor nuclei')
print('  - IS_OS: Inner/outer segment junction - critical for disease detection')
print('  - RPE_Choroid: Retinal pigment epithelium - high scatter, texture preservation needed')
print()
print('Our method learns layer-appropriate denoising:')
print('  - Higher gate values for RNFL (more denoising)')
print('  - Lower gate values for RPE (preserve texture)')
print('='*70)

# Save results
output_file = 'tmi_per_layer_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to: {output_file}')
EOF

echo ""
echo "=============================================="
echo "STEP 3.4 COMPLETE"
echo "=============================================="
echo "All evaluation steps complete!"
echo ""
echo "Generated files:"
echo "  - tmi_results_comparison.json"
echo "  - tmi_per_layer_results.json"
echo "=============================================="
