#!/usr/bin/env python3
"""Evaluate ablation study using the exact same validate() function from training."""

import gc
import json
import os
import torch
from torch.utils.data import DataLoader

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    validate,
)


def load_model(ckpt_path, backbone_path, ablation='none'):
    """Load model from checkpoint."""
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

    if ablation != 'none':
        model.corrector.set_ablation(ablation)

    model.eval()
    return model


def main():
    backbone_path = 'outputs/nafnet_pku37_w40/best_model.pth'
    test_jsonl = 'pku37_oct_dataset/pku37_real_test.jsonl'

    ablation_configs = {
        'full': ('outputs/nafnet_relaxed_psnr/best_model_cooperative.pth', 'none'),
        'no_negotiator': ('outputs/ablation_no_negotiator/best_model_cooperative.pth', 'no_negotiator'),
        'no_edge': ('outputs/ablation_no_edge/best_model_cooperative.pth', 'no_edge'),
        'no_uncertainty': ('outputs/ablation_no_uncertainty/best_model_cooperative.pth', 'no_uncertainty'),
        'no_bg_smooth': ('outputs/ablation_no_bg_smooth/best_model_cooperative.pth', 'no_bg_smooth'),
    }

    # Create test dataloader (batch_size=1 for full-res images)
    dataset = PKU37Dataset(test_jsonl, patch_size=0, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test set: {len(dataset)} images")

    all_results = {}

    for name, (ckpt_path, ablation_mode) in ablation_configs.items():
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        model = load_model(ckpt_path, backbone_path, ablation_mode)

        # Use the exact same validate() function from training
        metrics = validate(model, loader, 'cpu', lpips_model=None, criterion=None)

        # Extract the key metrics
        result = {
            'psnr_backbone': metrics['psnr_backbone'],
            'psnr_corrected': metrics['psnr_corrected'],
            'psnr_delta': metrics['psnr_delta'],
            'cnr_backbone': metrics['cnr_backbone'],
            'cnr_corrected': metrics['cnr_corrected'],
            'cnr_improvement': metrics['cnr_improvement'],
            'tci_backbone': metrics['tci_backbone'],
            'tci_corrected': metrics['tci_corrected'],
            'tci_improvement': metrics['tci_improvement'],
            'epi_backbone': metrics['epi_backbone'],
            'epi_corrected': metrics['epi_corrected'],
            'epi_improvement': (metrics['epi_corrected'] / max(metrics['epi_backbone'], 1e-8) - 1.0) * 100,
            'bs_backbone': metrics['boundary_sharpness_backbone'],
            'bs_corrected': metrics['boundary_sharpness_corrected'],
            'bs_improvement': (metrics['boundary_sharpness_corrected'] / max(metrics['boundary_sharpness_backbone'], 1e-8) - 1.0) * 100,
            'enl_backbone': metrics['enl_backbone'],
            'enl_corrected': metrics['enl_corrected'],
            'enl_improvement': (metrics['enl_corrected'] / max(metrics['enl_backbone'], 1e-8) - 1.0) * 100,
            'snr_backbone': metrics['snr_backbone'],
            'snr_corrected': metrics['snr_corrected'],
            'snr_improvement': (metrics['snr_corrected'] / max(metrics['snr_backbone'], 1e-8) - 1.0) * 100,
            'correction_magnitude': metrics.get('correction_magnitude', 0),
        }
        all_results[name] = result

        print(f"  PSNR: {result['psnr_backbone']:.2f} -> {result['psnr_corrected']:.2f} (delta: {result['psnr_delta']:+.2f} dB)")
        print(f"  CNR:  {result['cnr_backbone']:.3f} -> {result['cnr_corrected']:.3f} ({result['cnr_improvement']:+.1f}%)")
        print(f"  TCI:  {result['tci_backbone']:.4f} -> {result['tci_corrected']:.4f} ({result['tci_improvement']:+.1f}%)")
        print(f"  EPI:  {result['epi_backbone']:.4f} -> {result['epi_corrected']:.4f} ({result['epi_improvement']:+.1f}%)")
        print(f"  BS:   {result['bs_backbone']:.4f} -> {result['bs_corrected']:.4f} ({result['bs_improvement']:+.1f}%)")
        print(f"  ENL:  {result['enl_backbone']:.1f} -> {result['enl_corrected']:.1f} ({result['enl_improvement']:+.1f}%)")
        print(f"  SNR:  {result['snr_backbone']:.2f} -> {result['snr_corrected']:.2f} ({result['snr_improvement']:+.1f}%)")

        # Save incrementally
        with open(f'outputs/ablation_v2_{name}.json', 'w') as f:
            json.dump(result, f, indent=2)

        del model
        gc.collect()

    # Save combined
    with open('outputs/ablation_v2_all.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<20} {'dPSNR(dB)':>10} {'dCNR%':>8} {'dTCI%':>8} {'dEPI%':>8} {'dBS%':>8} {'dENL%':>8} {'dSNR%':>8}")
    print("-" * 100)
    for name, res in all_results.items():
        print(f"{name:<20} {res['psnr_delta']:>+10.2f} {res['cnr_improvement']:>+8.1f} "
              f"{res['tci_improvement']:>+8.1f} {res['epi_improvement']:>+8.1f} "
              f"{res['bs_improvement']:>+8.1f} {res['enl_improvement']:>+8.1f} "
              f"{res['snr_improvement']:>+8.1f}")

    print(f"\nResults saved to outputs/ablation_v2_all.json")


if __name__ == '__main__':
    main()
