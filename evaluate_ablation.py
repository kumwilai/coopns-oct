#!/usr/bin/env python3
"""Evaluate ablation study models on PKU37 test set and save results."""

import json
import os
import sys
import torch
import numpy as np

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
)


def compute_clinical_metrics(denoised, clean, noisy):
    """Compute all 7 clinical metrics for a single image pair."""
    d = denoised.squeeze().cpu().numpy()
    c = clean.squeeze().cpu().numpy()
    n = noisy.squeeze().cpu().numpy()

    # PSNR
    mse = np.mean((d - c) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    # For backbone PSNR (noisy treated as backbone output when not available)
    # We compute delta metrics: corrected vs backbone
    # But here we compute absolute metrics for corrected

    h, w = d.shape

    # Define tissue region (top 70%) and background (bottom 25%)
    tissue_mask = np.zeros_like(d, dtype=bool)
    tissue_mask[:int(0.7 * h), :] = True
    bg_mask = np.zeros_like(d, dtype=bool)
    bg_mask[int(0.75 * h):, :] = True

    tissue_vals = d[tissue_mask]
    bg_vals = d[bg_mask]

    # CNR
    if len(tissue_vals) > 0 and len(bg_vals) > 0 and np.std(bg_vals) > 1e-8:
        cnr = (np.mean(tissue_vals) - np.mean(bg_vals)) / np.std(bg_vals)
    else:
        cnr = 0.0

    # TCI (Tissue Contrast Index) - std of tissue region
    tci = float(np.std(tissue_vals)) if len(tissue_vals) > 0 else 0.0

    # EPI (Edge Preservation Index) - correlation of gradients
    from scipy import ndimage
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

    # BS (Boundary Sharpness) - mean gradient at edges
    edge_threshold = np.percentile(c_grad, 90)
    edge_mask = c_grad > edge_threshold
    bs = float(np.mean(d_grad[edge_mask])) if np.any(edge_mask) else 0.0

    # ENL (Equivalent Number of Looks)
    if len(tissue_vals) > 0 and np.std(tissue_vals) > 1e-8:
        enl = (np.mean(tissue_vals) / np.std(tissue_vals)) ** 2
    else:
        enl = 0.0

    # SNR
    if len(bg_vals) > 0 and np.std(bg_vals) > 1e-8:
        snr = np.mean(tissue_vals) / np.std(bg_vals)
    else:
        snr = 0.0

    return {
        'psnr': float(psnr),
        'cnr': float(cnr),
        'tci': float(tci),
        'epi': float(epi),
        'bs': float(bs),
        'enl': float(enl),
        'snr': float(snr),
    }


def load_model(ckpt_path, backbone_path, device='cpu', ablation='none'):
    """Load a cooperative model from checkpoint."""
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

    # Set ablation mode
    if ablation != 'none':
        model.corrector.set_ablation(ablation)

    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_jsonl, device='cpu'):
    """Evaluate model on test set, returning per-image metrics."""
    dataset = PKU37Dataset(test_jsonl, patch_size=0, is_train=False)

    all_backbone_metrics = []
    all_corrected_metrics = []

    for i in range(len(dataset)):
        sample = dataset[i]
        clean = sample['clean'].unsqueeze(0).to(device)
        noisy = sample['noisy'].unsqueeze(0).to(device)

        with torch.no_grad():
            backbone_out, unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=unc, return_details=False,
            )

        bb_metrics = compute_clinical_metrics(backbone_out, clean, noisy)
        co_metrics = compute_clinical_metrics(corrected, clean, noisy)

        all_backbone_metrics.append(bb_metrics)
        all_corrected_metrics.append(co_metrics)

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{len(dataset)}")

    # Average metrics
    avg_bb = {}
    avg_co = {}
    for key in all_backbone_metrics[0]:
        avg_bb[key] = np.mean([m[key] for m in all_backbone_metrics])
        avg_co[key] = np.mean([m[key] for m in all_corrected_metrics])

    # Compute deltas (percentage change for clinical, absolute for PSNR)
    deltas = {}
    deltas['psnr'] = float(avg_co['psnr'] - avg_bb['psnr'])
    for key in ['cnr', 'tci', 'epi', 'bs', 'enl', 'snr']:
        if abs(avg_bb[key]) > 1e-8:
            deltas[key] = float((avg_co[key] - avg_bb[key]) / abs(avg_bb[key]) * 100)
        else:
            deltas[key] = 0.0

    return {
        'backbone_avg': {k: float(v) for k, v in avg_bb.items()},
        'corrected_avg': {k: float(v) for k, v in avg_co.items()},
        'deltas': deltas,
        'n_images': len(dataset),
    }


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

    # Allow running a specific ablation
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = list(ablation_configs.keys())

    all_results = {}

    for name in targets:
        if name not in ablation_configs:
            print(f"Unknown ablation: {name}")
            continue

        ckpt_path, ablation_mode = ablation_configs[name]

        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: {ckpt_path} not found")
            continue

        print(f"\n=== Evaluating: {name} ===")
        model = load_model(ckpt_path, backbone_path, device, ablation_mode)
        results = evaluate_model(model, test_jsonl, device)
        all_results[name] = results

        print(f"  PSNR delta: {results['deltas']['psnr']:+.2f} dB")
        print(f"  CNR delta:  {results['deltas']['cnr']:+.1f}%")
        print(f"  TCI delta:  {results['deltas']['tci']:+.1f}%")
        print(f"  EPI delta:  {results['deltas']['epi']:+.1f}%")
        print(f"  BS delta:   {results['deltas']['bs']:+.1f}%")
        print(f"  ENL delta:  {results['deltas']['enl']:+.1f}%")
        print(f"  SNR delta:  {results['deltas']['snr']:+.1f}%")

        del model

        # Save incrementally
        out_path = f'outputs/ablation_results_{name}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {out_path}")

    # Save combined results
    if all_results:
        with open('outputs/ablation_results_all.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to outputs/ablation_results_all.json")

        # Print summary table
        print("\n" + "="*80)
        print(f"{'Config':<25} {'dPSNR':>8} {'dCNR%':>8} {'dTCI%':>8} {'dEPI%':>8} {'dBS%':>8} {'dENL%':>8} {'dSNR%':>8}")
        print("-"*80)
        for name, res in all_results.items():
            d = res['deltas']
            print(f"{name:<25} {d['psnr']:>+8.2f} {d['cnr']:>+8.1f} {d['tci']:>+8.1f} {d['epi']:>+8.1f} {d['bs']:>+8.1f} {d['enl']:>+8.1f} {d['snr']:>+8.1f}")


if __name__ == '__main__':
    main()
