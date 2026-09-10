#!/usr/bin/env python3
"""
Evaluate CoopNS-OCT with TTA on Duke datasets, computing clinical feature
metrics using the SAME compute_all_metrics function as BM3D/NLM eval.

This ensures apples-to-apples comparison.

Usage:
    python eval_model_with_tta_clinical.py \
        --checkpoint benchmarks/pretrained/best_model_cooperative.pth \
        --backbone benchmarks/pretrained/nafnet_backbone.pth \
        --dataset duke17
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    compute_psnr,
    compute_ssim,
)
from validate_crossdataset import TestTimeAdaptation

# Use the SAME metrics as BM3D/NLM eval
from eval_bm3d_nlm_clinical import compute_all_metrics


class SimpleDataset(Dataset):
    """Simple dataset from JSONL file."""
    def __init__(self, jsonl_path):
        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                self.pairs.append((entry['noisy_path'], entry['clean_path']))
        print(f"  Loaded {len(self.pairs)} pairs from {jsonl_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0
        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0),
            'clean': torch.from_numpy(clean).unsqueeze(0),
        }


def evaluate_clinical(model, loader, device, label=""):
    """Evaluate model with compute_all_metrics (same as BM3D/NLM)."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            corrected, backbone_out, info = model(noisy)

            # Compute metrics using the SAME function as BM3D/NLM eval
            # For backbone: treat backbone_out as "denoised"
            bb_m = compute_all_metrics(backbone_out, noisy, clean)
            # For CoopNS: treat corrected as "denoised"
            co_m = compute_all_metrics(corrected, noisy, clean)

            all_metrics.append({'backbone': bb_m, 'corrected': co_m})

            if (i + 1) % max(1, len(loader) // 3) == 0 or i == 0:
                print(f"  [{i+1}/{len(loader)}] BB PSNR={bb_m['psnr_denoised']:.2f} "
                      f"Corr PSNR={co_m['psnr_denoised']:.2f} "
                      f"BB contrast={bb_m['contrast_denoised']:.4f} "
                      f"Corr contrast={co_m['contrast_denoised']:.4f}", flush=True)

    # Averages
    bb_keys = ['psnr_denoised', 'ssim_denoised', 'cnr_denoised',
               'contrast_denoised', 'boundary_denoised', 'texture_denoised', 'edge_denoised']
    bb_avg = {k: np.mean([m['backbone'][k] for m in all_metrics]) for k in bb_keys}
    co_avg = {k: np.mean([m['corrected'][k] for m in all_metrics]) for k in bb_keys}

    print(f"\n  {label} NAFNet Backbone:")
    print(f"    PSNR={bb_avg['psnr_denoised']:.2f} SSIM={bb_avg['ssim_denoised']:.4f} "
          f"CNR={bb_avg['cnr_denoised']:.2f}")
    print(f"    contrast={bb_avg['contrast_denoised']:.4f} boundary={bb_avg['boundary_denoised']:.4f} "
          f"texture={bb_avg['texture_denoised']:.4f} edge={bb_avg['edge_denoised']:.4f}")

    print(f"  {label} CoopNS-OCT:")
    print(f"    PSNR={co_avg['psnr_denoised']:.2f} SSIM={co_avg['ssim_denoised']:.4f} "
          f"CNR={co_avg['cnr_denoised']:.2f}")
    print(f"    contrast={co_avg['contrast_denoised']:.4f} boundary={co_avg['boundary_denoised']:.4f} "
          f"texture={co_avg['texture_denoised']:.4f} edge={co_avg['edge_denoised']:.4f}")

    return all_metrics, bb_avg, co_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='benchmarks/pretrained/best_model_cooperative.pth')
    parser.add_argument('--backbone', default='benchmarks/pretrained/nafnet_backbone.pth')
    parser.add_argument('--backbone_name', type=str, default='nafnet',
                        choices=['nafnet', 'dncnn', 'swinir', 'kbnet', 'mambair'])
    parser.add_argument('--dataset', required=True, choices=['duke17', 'duke2013', 'pku37', 'pku37_val'])
    parser.add_argument('--force_tta', action='store_true', help='Force TTA even on in-distribution data')
    parser.add_argument('--output_dir', default='outputs/baselines')
    parser.add_argument('--device', default='cpu')
    # TTA params (best config from sweep)
    parser.add_argument('--tta_steps', type=int, default=10)
    parser.add_argument('--tta_lr', type=float, default=5e-4)
    parser.add_argument('--tta_adapt_samples', type=int, default=5)
    parser.add_argument('--tta_w_magnitude', type=float, default=0.5)
    parser.add_argument('--tta_w_cnr', type=float, default=1.5)
    parser.add_argument('--tta_w_consistency', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Dataset paths
    dataset_map = {
        'duke17': 'duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl',
        'duke2013': 'duke_sota_datasets/Duke17_Eval/duke2013_synth_eval.jsonl',
        'pku37': 'pku37_oct_dataset/pku37_real_test.jsonl',
        'pku37_val': 'pku37_oct_dataset/pku37_real_val.jsonl',
    }
    test_jsonl = dataset_map[args.dataset]
    dataset_name = args.dataset.upper()

    # Load model
    print(f"Loading model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.backbone
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    del ckpt
    model = model.to(device)
    model.eval()

    # Create dataset and loader
    dataset = SimpleDataset(test_jsonl)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    t0 = time.time()

    use_tta = args.dataset in ('duke17', 'duke2013') or args.force_tta
    if not use_tta:
        # No TTA for in-distribution
        print(f"\n=== Evaluating on {dataset_name} (no TTA) ===")
        all_metrics, bb_avg, co_avg = evaluate_clinical(model, loader, device, label="[No TTA]")
        tta_label = "no_tta"
    else:
        # --- Pre-TTA evaluation ---
        print(f"\n=== Pre-TTA evaluation on {dataset_name} ===")
        _, bb_avg_pre, co_avg_pre = evaluate_clinical(model, loader, device, label="[Pre-TTA]")

        # --- Apply TTA ---
        print(f"\n=== Applying TTA on {dataset_name} ===")
        tta = TestTimeAdaptation(
            model, device=device,
            tta_steps=args.tta_steps,
            tta_lr=args.tta_lr,
            n_adapt_samples=args.tta_adapt_samples,
            w_magnitude=args.tta_w_magnitude,
            w_cnr=args.tta_w_cnr,
            w_consistency=args.tta_w_consistency,
        )
        tta.adapt(loader, dataset_name=dataset_name)

        # --- Post-TTA evaluation ---
        print(f"\n=== Post-TTA evaluation on {dataset_name} ===")
        all_metrics, bb_avg, co_avg = evaluate_clinical(model, loader, device, label="[Post-TTA]")
        tta_label = "with_tta"

        # Print improvement
        print(f"\n  TTA effect on CoopNS-OCT:")
        for k in ['psnr_denoised', 'ssim_denoised', 'cnr_denoised',
                   'contrast_denoised', 'boundary_denoised', 'texture_denoised', 'edge_denoised']:
            pre = co_avg_pre[k]
            post = co_avg[k]
            print(f"    {k}: {pre:.4f} -> {post:.4f} (delta={post-pre:+.4f})")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'dataset': dataset_name,
        'tta': tta_label,
        'n_images': len(dataset),
        'backbone_averages': bb_avg,
        'corrected_averages': co_avg,
    }
    out_path = os.path.join(args.output_dir, f'model_clinical_{args.dataset}_{tta_label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
