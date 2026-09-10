#!/usr/bin/env python3
"""Fast scan: rank PKU37 test images by correction magnitude per backbone."""

import numpy as np
import torch
import gc

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
)

CONFIGS = {
    'nafnet': ('outputs/nafnet_qt69d_2ch/best_model_cooperative.pth',
               'outputs/nafnet_pku37_w40/best_model.pth'),
    'dncnn': ('outputs/dncnn_fullres_qt69d/best_model_cooperative.pth',
              'NukeModel/dncnn_7m/best.pth'),
    'kbnet': ('outputs/kbnet_qt69d/best_model_cooperative.pth',
              'NukeModel/kbnet_7m/best.pth'),
}

dataset = PKU37Dataset('pku37_oct_dataset/pku37_real_test.jsonl', patch_size=0, is_train=False)
print(f"Dataset: {len(dataset)} images\n", flush=True)

all_scores = {}

for bname, (ckpt, bb_path) in CONFIGS.items():
    print(f"=== {bname.upper()} ===", flush=True)
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=bname, pretrained_backbone=bb_path, hidden_channels=64)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('model_state_dict', sd)
    sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    scores = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        noisy = sample['noisy'].unsqueeze(0)
        clean = sample['clean'].unsqueeze(0)

        with torch.no_grad():
            bb_out, unc = model.backbone(noisy)
            co, _ = model.corrector(bb_out, noisy, None, nafnet_uncertainty=unc, return_details=False)

        # Tissue region only (top 70%)
        H = bb_out.shape[-2]
        th = int(H * 0.7)
        corr = (co[:,:,:th,:] - bb_out[:,:,:th,:]).abs().mean().item()
        psnr_d = compute_psnr(co, clean) - compute_psnr(bb_out, clean)
        scores.append((idx, corr, psnr_d))

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/173]", flush=True)

    scores.sort(key=lambda x: x[1], reverse=True)
    all_scores[bname] = scores

    print(f"  Top 10: {[s[0] for s in scores[:10]]}", flush=True)
    print(f"  Mag range: {scores[0][1]:.5f} — {scores[-1][1]:.5f}\n", flush=True)

    del model; gc.collect()

# Find shared good samples
top30 = {b: set(s[0] for s in sc[:30]) for b, sc in all_scores.items()}
shared = top30['nafnet'] & top30['dncnn'] & top30['kbnet']

print(f"\n=== SHARED TOP-30 (good for all 3 backbones): {sorted(shared)} ===")

# Rank shared by avg correction
ranked = []
for idx in shared:
    avg = np.mean([next(s[1] for s in all_scores[b] if s[0] == idx) for b in all_scores])
    ranked.append((idx, avg))
ranked.sort(key=lambda x: x[1], reverse=True)

print("\nBest samples for paper figures:")
for idx, mag in ranked[:10]:
    per_bb = {b: next(s[1] for s in all_scores[b] if s[0] == idx) for b in all_scores}
    print(f"  idx={idx:>3d}  avg_mag={mag:.5f}  "
          f"nafnet={per_bb['nafnet']:.5f}  dncnn={per_bb['dncnn']:.5f}  kbnet={per_bb['kbnet']:.5f}")
