#!/bin/bash
# =============================================================================
# TMI Step 3.1: Evaluate Full Method (Primary Results)
# =============================================================================
# Purpose: Get main results for TMI paper Table 1
# Time: ~10 minutes
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI STEP 3.1: EVALUATE FULL METHOD"
echo "=============================================="

# Find the latest multi-task checkpoint
MULTITASK_DIR=$(ls -td tmi_multitask/*/checkpoints 2>/dev/null | head -1)
if [ -z "$MULTITASK_DIR" ]; then
    echo "ERROR: No multi-task checkpoint found!"
    echo "Please run Step 1.3 first: bash tmi_seg_step3_multitask.sh"
    exit 1
fi

CHECKPOINT="$MULTITASK_DIR/best_psnr.pth"
echo "Using checkpoint: $CHECKPOINT"
echo ""

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"
echo ""

python << 'EOF'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'nsnd_oct')

import os
import glob
import torch
from train_multitask import MultiTaskDenoiser, MultiTaskOCTDataset, validate, LAYER_NAMES
from torch.utils.data import DataLoader
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

# Load best model
model = MultiTaskDenoiser().to(device)
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
print(f'Loaded model from epoch {ckpt["epoch"]}')

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
print('TABLE 1: FULL METHOD RESULTS (Layer-Specific Gates)')
print('='*70)
print(f'| {"Metric":<25} | {"Baseline":<12} | {"Ours":<12} | {"Gain":<12} |')
print(f'|{"-"*27}|{"-"*14}|{"-"*14}|{"-"*14}|')
psnr_gain = results['psnr'] - results['psnr_base']
ssim_gain = results['ssim'] - results['ssim_base']
print(f'| {"PSNR (dB)":<25} | {results["psnr_base"]:<12.2f} | {results["psnr"]:<12.2f} | +{psnr_gain:<11.2f} |')
print(f'| {"SSIM":<25} | {results["ssim_base"]:<12.4f} | {results["ssim"]:<12.4f} | +{ssim_gain:<11.4f} |')
print(f'| {"Dice":<25} | {"-":<12} | {results["dice"]:<12.4f} | {"-":<12} |')
print('='*70)

# Clinical metrics
c = results['clinical']
print()
print('CLINICAL METRICS:')
if c['cnr_base'] and c['cnr_ours']:
    print(f'  CNR:  {c["cnr_base"]:.4f} -> {c["cnr_ours"]:.4f} ({"+" if c["cnr_ours"] > c["cnr_base"] else ""}{c["cnr_ours"]-c["cnr_base"]:.4f})')
if c['epi_base'] and c['epi_ours']:
    print(f'  EPI:  {c["epi_base"]:.4f} -> {c["epi_ours"]:.4f} (target: 1.0)')
if c['boundary_psnr_base'] and c['boundary_psnr_ours']:
    print(f'  Boundary PSNR: {c["boundary_psnr_base"]:.2f} -> {c["boundary_psnr_ours"]:.2f} dB')

# Per-layer results
print()
print('PER-LAYER PSNR IMPROVEMENT:')
print(f'  {"Layer":<15} {"Base":>8} {"Ours":>8} {"Gain":>8}')
print(f'  {"-"*40}')
for name in LAYER_NAMES:
    r = results['per_layer'][name]
    if r['psnr_gain']:
        print(f'  {name:<15} {r["psnr_base"]:>8.2f} {r["psnr_ours"]:>8.2f} {"+"+str(round(r["psnr_gain"],2)):>8}')

# Save results
import json
output_dir = os.path.dirname(checkpoint_path).replace('checkpoints', 'evaluation')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'full_method_results.json'), 'w') as f:
    json.dump({
        'psnr_base': results['psnr_base'],
        'psnr_ours': results['psnr'],
        'ssim_base': results['ssim_base'],
        'ssim_ours': results['ssim'],
        'dice': results['dice'],
        'clinical': c,
        'per_layer': {k: {kk: vv for kk, vv in v.items()} for k, v in results['per_layer'].items()}
    }, f, indent=2)
print(f'\nResults saved to: {output_dir}/full_method_results.json')
EOF

echo ""
echo "=============================================="
echo "STEP 3.1 COMPLETE"
echo "=============================================="
echo "Next: bash tmi_step3_2_eval_ablation.sh"
echo "=============================================="
