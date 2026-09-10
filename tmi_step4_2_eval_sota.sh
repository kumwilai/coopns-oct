#!/bin/bash
# =============================================================================
# TMI Step 4.2: Evaluate All Methods (SOTA Comparison)
# =============================================================================
# Purpose: Generate final comparison table with all SOTA methods
# Time: ~20 minutes
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TMI STEP 4.2: EVALUATE SOTA COMPARISON"
echo "=============================================="

python << 'EOF'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'nsnd_oct')

import os
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from train_multitask import MultiTaskDenoiser, MultiTaskOCTDataset, LAYER_NAMES
from train_multitask_ablation import MultiTaskDenoiserAblation
from nsnd.models.dncnn import DnCNN
from nsnd.models.restormer import Restormer
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.utils.metrics import compute_psnr, compute_ssim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load validation data
val_ds = MultiTaskOCTDataset('seg_data/seg_val.jsonl', patch_size=64, max_samples=400,
                              random_crop=True, ensure_all_layers=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
print(f'Validation samples: {len(val_ds)}')


def evaluate_model(model, loader, device, model_name):
    """Evaluate a denoising model."""
    model.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            # Handle different model interfaces
            if hasattr(model, 'backbone'):
                # Our multi-task model
                denoised, _ = model(noisy, return_features=False)
            elif isinstance(model, NAFNetFullFiLM):
                denoised = model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            else:
                denoised = model(noisy)

            for i in range(noisy.size(0)):
                c = clean[i, 0].cpu().numpy()
                d = denoised[i, 0].cpu().numpy()
                psnr_list.append(compute_psnr(d, c))
                ssim_list.append(compute_ssim(d, c))

    return {
        'psnr': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
    }


results = {}

# 1. NAFNet Backbone (baseline)
print("\n[1/5] Evaluating NAFNet Backbone...")
backbone = NAFNetFullFiLM(img_channel=1, width=64, enc_blk_nums=[2,2,2],
                          dec_blk_nums=[2,2,2], middle_blk_num=2, cond_dim=32).to(device)
ckpt = torch.load('outputs/nafnet_analysis_maps_w64/nafnet_best.pth', map_location=device, weights_only=False)
backbone.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
backbone_params = sum(p.numel() for p in backbone.parameters())
r = evaluate_model(backbone, val_loader, device, "NAFNet")
results['NAFNet (Backbone)'] = {**r, 'params': backbone_params}
del backbone, ckpt

# 2. DnCNN (if trained)
dncnn_ckpt = glob.glob('sota_baselines_fair/dncnn_fair/best.pth')
if dncnn_ckpt:
    print("\n[2/5] Evaluating DnCNN...")
    ckpt = torch.load(dncnn_ckpt[0], map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = DnCNN(in_channels=1, out_channels=1,
                  num_layers=config.get('dncnn_layers', 31),
                  features=config.get('dncnn_features', 192)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    params = sum(p.numel() for p in model.parameters())
    r = evaluate_model(model, val_loader, device, "DnCNN")
    results['DnCNN (Fair)'] = {**r, 'params': params}
    del model, ckpt
else:
    print("\n[2/5] DnCNN not found - run Step 4.1 first")

# 3. Restormer (if trained)
restormer_ckpt = glob.glob('sota_baselines_fair/restormer_fair/best.pth')
if restormer_ckpt:
    print("\n[3/5] Evaluating Restormer...")
    ckpt = torch.load(restormer_ckpt[0], map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = Restormer(inp_channels=1, out_channels=1,
                      dim=config.get('restormer_dim', 48),
                      num_blocks=[2,2,2,2], num_refinement_blocks=2,
                      heads=[1,2,4,8]).to(device)
    model.load_state_dict(ckpt['state_dict'])
    params = sum(p.numel() for p in model.parameters())
    r = evaluate_model(model, val_loader, device, "Restormer")
    results['Restormer (Fair)'] = {**r, 'params': params}
    del model, ckpt
else:
    print("\n[3/5] Restormer not found - run Step 4.1 first")

# 4. Ours - Ablation (Global Gate)
ablation_ckpt = sorted(glob.glob('tmi_ablation/no_layer_gates_*/checkpoints/best_psnr.pth'))
if ablation_ckpt:
    print("\n[4/5] Evaluating Ours (Ablation - Global Gate)...")
    model = MultiTaskDenoiserAblation().to(device)
    ckpt = torch.load(ablation_ckpt[-1], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    params = sum(p.numel() for p in model.parameters())
    r = evaluate_model(model, val_loader, device, "Ours-Ablation")
    results['Ours (Global Gate)'] = {**r, 'params': params}
    del model, ckpt
else:
    print("\n[4/5] Ablation not found - run Step 2.1 first")

# 5. Ours - Full Method (Layer-Specific Gates)
full_ckpt = sorted(glob.glob('tmi_multitask/*/checkpoints/best_psnr.pth'))
if full_ckpt:
    print("\n[5/5] Evaluating Ours (Full - Layer-Specific Gates)...")
    model = MultiTaskDenoiser().to(device)
    ckpt = torch.load(full_ckpt[-1], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    params = sum(p.numel() for p in model.parameters())
    r = evaluate_model(model, val_loader, device, "Ours-Full")
    results['Ours (Layer-Specific)'] = {**r, 'params': params}
    del model, ckpt
else:
    print("\n[5/5] Full method not found - run Step 1.3 first")


# Print comparison table
print()
print('='*100)
print('TABLE FOR TMI PAPER: SOTA Comparison with Fair Parameter Count')
print('='*100)
print()
print(f'| {"Method":<28} | {"Params":^12} | {"PSNR (dB)":^14} | {"SSIM":^14} |')
print(f'|{"-"*30}|{"-"*14}|{"-"*16}|{"-"*16}|')

for method, r in results.items():
    params_str = f'{r["params"]/1e6:.1f}M'
    psnr_str = f'{r["psnr"]:.2f} ± {r["psnr_std"]:.2f}'
    ssim_str = f'{r["ssim"]:.4f} ± {r["ssim_std"]:.4f}'
    print(f'| {method:<28} | {params_str:^12} | {psnr_str:^14} | {ssim_str:^14} |')

print('='*100)

# Highlight key findings
print()
print('KEY FINDINGS:')
print('-'*60)

if 'Ours (Layer-Specific)' in results:
    ours_psnr = results['Ours (Layer-Specific)']['psnr']

    if 'NAFNet (Backbone)' in results:
        gain_vs_backbone = ours_psnr - results['NAFNet (Backbone)']['psnr']
        print(f'1. Improvement over backbone: +{gain_vs_backbone:.2f} dB')

    if 'DnCNN (Fair)' in results:
        gain_vs_dncnn = ours_psnr - results['DnCNN (Fair)']['psnr']
        print(f'2. Improvement over DnCNN:    +{gain_vs_dncnn:.2f} dB')

    if 'Restormer (Fair)' in results:
        gain_vs_restormer = ours_psnr - results['Restormer (Fair)']['psnr']
        print(f'3. Improvement over Restormer: +{gain_vs_restormer:.2f} dB')

    if 'Ours (Global Gate)' in results:
        gain_layer_specific = ours_psnr - results['Ours (Global Gate)']['psnr']
        print(f'4. Layer-specific gate contribution: +{gain_layer_specific:.2f} dB')

print()
print('Note: All models have ~9-10M parameters for fair comparison.')
print('='*100)

# Save results
with open('tmi_sota_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to: tmi_sota_comparison.json')
EOF

echo ""
echo "=============================================="
echo "STEP 4.2 COMPLETE"
echo "=============================================="
echo "Final comparison table generated!"
echo "=============================================="
