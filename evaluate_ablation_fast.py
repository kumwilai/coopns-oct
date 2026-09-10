#!/usr/bin/env python3
"""Fast ablation evaluation: cache backbone outputs, only re-run corrector."""

import gc
import json
import os
import sys
import torch
import numpy as np
from scipy import ndimage

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
)


def compute_clinical_metrics_np(d, c):
    """Compute clinical metrics from numpy arrays."""
    # PSNR
    mse = np.mean((d - c) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    h, w = d.shape
    tissue_mask = np.zeros_like(d, dtype=bool)
    tissue_mask[:int(0.7 * h), :] = True
    bg_mask = np.zeros_like(d, dtype=bool)
    bg_mask[int(0.75 * h):, :] = True

    tissue_vals = d[tissue_mask]
    bg_vals = d[bg_mask]

    # CNR
    cnr = (np.mean(tissue_vals) - np.mean(bg_vals)) / max(np.std(bg_vals), 1e-8)

    # TCI
    tci = float(np.std(tissue_vals))

    # EPI
    d_gx = ndimage.sobel(d, axis=1)
    d_gy = ndimage.sobel(d, axis=0)
    c_gx = ndimage.sobel(c, axis=1)
    c_gy = ndimage.sobel(c, axis=0)
    d_grad = np.sqrt(d_gx**2 + d_gy**2)
    c_grad = np.sqrt(c_gx**2 + c_gy**2)
    if np.std(d_grad) > 1e-8 and np.std(c_grad) > 1e-8:
        epi = float(np.corrcoef(d_grad.ravel(), c_grad.ravel())[0, 1])
    else:
        epi = 0.0

    # BS
    edge_threshold = np.percentile(c_grad, 90)
    edge_mask = c_grad > edge_threshold
    bs = float(np.mean(d_grad[edge_mask])) if np.any(edge_mask) else 0.0

    # ENL
    enl = (np.mean(tissue_vals) / max(np.std(tissue_vals), 1e-8)) ** 2

    # SNR
    snr = np.mean(tissue_vals) / max(np.std(bg_vals), 1e-8)

    return {'psnr': float(psnr), 'cnr': float(cnr), 'tci': float(tci),
            'epi': float(epi), 'bs': float(bs), 'enl': float(enl), 'snr': float(snr)}


def main():
    device = 'cpu'
    backbone_path = 'outputs/nafnet_pku37_w40/best_model.pth'
    test_jsonl = 'pku37_oct_dataset/pku37_real_test.jsonl'

    ablation_configs = {
        'full': ('outputs/nafnet_relaxed_psnr/best_model_cooperative.pth', 'none'),
        'no_negotiator': ('outputs/ablation_no_negotiator/best_model_cooperative.pth', 'no_negotiator'),
        'no_edge': ('outputs/ablation_no_edge/best_model_cooperative.pth', 'no_edge'),
        'no_uncertainty': ('outputs/ablation_no_uncertainty/best_model_cooperative.pth', 'no_uncertainty'),
        'no_bg_smooth': ('outputs/ablation_no_bg_smooth/best_model_cooperative.pth', 'no_bg_smooth'),
    }

    # Step 1: Load backbone once and cache all backbone outputs
    print("Step 1: Loading backbone and caching outputs for all 173 test images...")
    ref_model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone=backbone_path,
        hidden_channels=64,
    )
    # Load full model weights (any checkpoint - backbone is the same)
    ckpt = torch.load(ablation_configs['full'][0], map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
               for k, v in state_dict.items()}
    model_state = ref_model.state_dict()
    compatible = {k: v for k, v in cleaned.items()
                  if k in model_state and v.shape == model_state[k].shape}
    ref_model.load_state_dict(compatible, strict=False)
    ref_model.eval()

    dataset = PKU37Dataset(test_jsonl, patch_size=0, is_train=False)
    n_images = len(dataset)

    cached_data = []  # list of (clean_np, backbone_np, backbone_tensor, unc_tensor, noisy_tensor)

    for i in range(n_images):
        sample = dataset[i]
        clean = sample['clean'].unsqueeze(0)
        noisy = sample['noisy'].unsqueeze(0)

        with torch.no_grad():
            backbone_out, unc = ref_model.backbone(noisy)

        cached_data.append({
            'clean_np': clean.squeeze().numpy(),
            'backbone_np': backbone_out.squeeze().numpy(),
            'backbone_tensor': backbone_out,
            'unc_tensor': unc,
            'noisy_tensor': noisy,
        })

        if (i + 1) % 20 == 0:
            print(f"  Cached {i+1}/{n_images}")

    # Compute backbone metrics once
    backbone_metrics = []
    for item in cached_data:
        bb_m = compute_clinical_metrics_np(item['backbone_np'], item['clean_np'])
        backbone_metrics.append(bb_m)

    avg_bb = {}
    for key in backbone_metrics[0]:
        avg_bb[key] = np.mean([m[key] for m in backbone_metrics])

    print(f"  Backbone avg PSNR: {avg_bb['psnr']:.2f} dB")
    print(f"  Cached {n_images} images. Backbone computation done.\n")

    del ref_model
    gc.collect()

    # Step 2: For each ablation, load corrector weights and evaluate
    all_results = {}

    for name, (ckpt_path, ablation_mode) in ablation_configs.items():
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: {ckpt_path} not found")
            continue

        print(f"=== Evaluating: {name} ===")

        # Load model with corrector weights
        model = NeuroSymbolicDenoiserV8Cooperative(
            backbone_name='nafnet',
            pretrained_backbone=backbone_path,
            hidden_channels=64,
        )
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
                   for k, v in state_dict.items()}
        model_state = model.state_dict()
        compatible = {k: v for k, v in cleaned.items()
                      if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(compatible, strict=False)

        if ablation_mode != 'none':
            model.corrector.set_ablation(ablation_mode)

        model.eval()

        # Run corrector only (backbone output already cached)
        corrected_metrics = []
        for i, item in enumerate(cached_data):
            with torch.no_grad():
                corrected, _ = model.corrector(
                    item['backbone_tensor'], item['noisy_tensor'], None,
                    nafnet_uncertainty=item['unc_tensor'], return_details=False,
                )
            co_np = corrected.squeeze().numpy()
            co_m = compute_clinical_metrics_np(co_np, item['clean_np'])
            corrected_metrics.append(co_m)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_images}")

        # Average
        avg_co = {}
        for key in corrected_metrics[0]:
            avg_co[key] = np.mean([m[key] for m in corrected_metrics])

        # Deltas
        deltas = {}
        deltas['psnr'] = float(avg_co['psnr'] - avg_bb['psnr'])
        for key in ['cnr', 'tci', 'epi', 'bs', 'enl', 'snr']:
            if abs(avg_bb[key]) > 1e-8:
                deltas[key] = float((avg_co[key] - avg_bb[key]) / abs(avg_bb[key]) * 100)
            else:
                deltas[key] = 0.0

        results = {
            'backbone_avg': {k: float(v) for k, v in avg_bb.items()},
            'corrected_avg': {k: float(v) for k, v in avg_co.items()},
            'deltas': deltas,
            'n_images': n_images,
        }
        all_results[name] = results

        print(f"  dPSNR: {deltas['psnr']:+.2f} dB  dCNR: {deltas['cnr']:+.1f}%  "
              f"dTCI: {deltas['tci']:+.1f}%  dEPI: {deltas['epi']:+.1f}%  "
              f"dBS: {deltas['bs']:+.1f}%  dENL: {deltas['enl']:+.1f}%")

        # Save incrementally
        with open(f'outputs/ablation_results_{name}.json', 'w') as f:
            json.dump(results, f, indent=2)

        del model
        gc.collect()

    # Save combined
    with open('outputs/ablation_results_all.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<25} {'dPSNR':>8} {'dCNR%':>8} {'dTCI%':>8} {'dEPI%':>8} {'dBS%':>8} {'dENL%':>8} {'dSNR%':>8}")
    print("-" * 90)
    for name, res in all_results.items():
        d = res['deltas']
        print(f"{name:<25} {d['psnr']:>+8.2f} {d['cnr']:>+8.1f} {d['tci']:>+8.1f} "
              f"{d['epi']:>+8.1f} {d['bs']:>+8.1f} {d['enl']:>+8.1f} {d['snr']:>+8.1f}")

    print(f"\nResults saved to outputs/ablation_results_all.json")


if __name__ == '__main__':
    main()
