#!/usr/bin/env python3
"""
Full Image Evaluation with Sliding Window Inference

For TMI paper - evaluates on full-resolution images using:
1. 64x64 patch extraction with stride (overlap)
2. Weighted averaging for overlapping regions (reduces boundary artifacts)
3. Full image metric computation

Memory efficient: processes one patch at a time.
"""

import argparse
import json
import os
import sys
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.utils.metrics import compute_psnr, compute_ssim


def create_weight_map(patch_size, device='cpu'):
    """
    Create a 2D weight map for smooth blending of overlapping patches.
    Higher weights in center, lower at edges (cosine window).
    """
    # Create 1D cosine window
    x = torch.linspace(0, np.pi, patch_size, device=device)
    w1d = (1 - torch.cos(x)) / 2  # 0 at edges, 1 at center

    # Create 2D weight map
    weight_map = w1d.unsqueeze(1) * w1d.unsqueeze(0)

    return weight_map


def sliding_window_inference(model, noisy_full, patch_size=64, stride=32, device='cpu', model_type='ours'):
    """
    Perform sliding window inference on a full image.

    Args:
        model: Denoising model
        noisy_full: Full noisy image [1, 1, H, W]
        patch_size: Size of patches (default 64)
        stride: Stride between patches (default 32 for 50% overlap)
        device: Device for computation
        model_type: 'ours', 'nafnet', or 'baseline'

    Returns:
        denoised_full: Full denoised image [1, 1, H, W]
    """
    _, _, H, W = noisy_full.shape

    # Create output tensors
    output_sum = torch.zeros(1, 1, H, W, device='cpu')
    weight_sum = torch.zeros(1, 1, H, W, device='cpu')

    # Create weight map for blending
    weight_map = create_weight_map(patch_size, device='cpu').unsqueeze(0).unsqueeze(0)

    # Calculate number of patches
    n_h = (H - patch_size) // stride + 1
    n_w = (W - patch_size) // stride + 1

    # Handle edge cases where image doesn't divide evenly
    if (H - patch_size) % stride != 0:
        n_h += 1
    if (W - patch_size) % stride != 0:
        n_w += 1

    model.eval()
    with torch.no_grad():
        for i in range(n_h):
            for j in range(n_w):
                # Calculate patch position
                top = min(i * stride, H - patch_size)
                left = min(j * stride, W - patch_size)

                # Extract patch
                patch = noisy_full[:, :, top:top+patch_size, left:left+patch_size].to(device)

                # Denoise patch
                if model_type == 'ours':
                    denoised_patch, _ = model(patch, return_features=False)
                elif model_type == 'nafnet':
                    denoised_patch = model(patch, spatial_map=None, basis=None, alpha=0.0, gate=None)
                else:
                    denoised_patch = model(patch)

                # Move to CPU and accumulate
                denoised_patch = denoised_patch.cpu()
                output_sum[:, :, top:top+patch_size, left:left+patch_size] += denoised_patch * weight_map
                weight_sum[:, :, top:top+patch_size, left:left+patch_size] += weight_map

                # Clear GPU memory
                del patch, denoised_patch

    # Normalize by weights
    denoised_full = output_sum / (weight_sum + 1e-8)

    return denoised_full


def evaluate_full_images(model, data_jsonl, device, patch_size=64, stride=32,
                         model_type='ours', max_samples=None, model_name="Model"):
    """
    Evaluate model on full images using sliding window inference.
    """
    # Load data list
    samples = []
    with open(data_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line.strip()))

    print(f"Evaluating {len(samples)} full images...")
    print(f"  Patch size: {patch_size}, Stride: {stride}")
    print(f"  Overlap: {100 * (1 - stride/patch_size):.0f}%")

    psnr_list = []
    ssim_list = []

    for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
        # Load full images
        noisy = np.array(Image.open(sample['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(sample['clean_path']).convert('L'), dtype=np.float32) / 255.0

        # Convert to tensor
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float()

        # Sliding window inference
        denoised_tensor = sliding_window_inference(
            model, noisy_tensor, patch_size, stride, device, model_type
        )

        # Convert to numpy
        denoised = denoised_tensor.squeeze().numpy()
        denoised = np.clip(denoised, 0, 1)

        # Compute metrics on full image
        psnr = compute_psnr(denoised, clean)
        ssim = compute_ssim(denoised, clean)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        # Memory cleanup
        del noisy_tensor, denoised_tensor
        gc.collect()

    return {
        'psnr': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'n_images': len(psnr_list),
    }


def main():
    parser = argparse.ArgumentParser(description='Full image evaluation with sliding window')
    parser.add_argument('--model', choices=['ours', 'ablation', 'dncnn', 'restormer', 'nafnet'],
                        required=True, help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint (auto-detected if not specified)')
    parser.add_argument('--val_jsonl', default='seg_data/seg_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32, help='Stride (32=50% overlap, 48=25% overlap)')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    print("=" * 70)
    print("FULL IMAGE EVALUATION WITH SLIDING WINDOW")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride} ({100*(1-args.stride/args.patch_size):.0f}% overlap)")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model
    if args.model == 'ours':
        from train_multitask import MultiTaskDenoiser
        import glob
        model = MultiTaskDenoiser().to(args.device)
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpts = sorted(glob.glob('tmi_multitask/*/checkpoints/best_psnr.pth'))
            ckpt_path = ckpts[-1] if ckpts else None
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            print(f"Loaded: {ckpt_path}")
        model_type = 'ours'
        model_name = "Ours (Layer-Specific)"

    elif args.model == 'ablation':
        from train_multitask_ablation import MultiTaskDenoiserAblation
        import glob
        model = MultiTaskDenoiserAblation().to(args.device)
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpts = sorted(glob.glob('tmi_ablation/no_layer_gates_*/checkpoints/best_psnr.pth'))
            ckpt_path = ckpts[-1] if ckpts else None
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            print(f"Loaded: {ckpt_path}")
        model_type = 'ours'
        model_name = "Ours (Global Gate)"

    elif args.model == 'nafnet':
        from nsnd.models.nafnet import NAFNetFullFiLM
        model = NAFNetFullFiLM(img_channel=1, width=64, enc_blk_nums=[2,2,2],
                               dec_blk_nums=[2,2,2], middle_blk_num=2, cond_dim=32).to(args.device)
        ckpt = torch.load('outputs/nafnet_analysis_maps_w64/nafnet_best.pth',
                          map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        model_type = 'nafnet'
        model_name = "NAFNet (Backbone)"

    elif args.model == 'dncnn':
        from nsnd.models.dncnn import DnCNN
        import glob
        ckpts = glob.glob('sota_baselines_fair/dncnn_fair/best.pth')
        if not ckpts:
            print("ERROR: DnCNN checkpoint not found. Run Step 4.1 first.")
            return
        ckpt = torch.load(ckpts[0], map_location=args.device, weights_only=False)
        config = ckpt.get('config', {})
        model = DnCNN(in_channels=1, out_channels=1,
                      num_layers=config.get('dncnn_layers', 31),
                      features=config.get('dncnn_features', 192)).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
        model_type = 'baseline'
        model_name = "DnCNN (Fair)"

    elif args.model == 'restormer':
        from nsnd.models.restormer import Restormer
        import glob
        ckpts = glob.glob('sota_baselines_fair/restormer_fair/best.pth')
        if not ckpts:
            print("ERROR: Restormer checkpoint not found. Run Step 4.1 first.")
            return
        ckpt = torch.load(ckpts[0], map_location=args.device, weights_only=False)
        config = ckpt.get('config', {})
        model = Restormer(inp_channels=1, out_channels=1,
                          dim=config.get('restormer_dim', 48),
                          num_blocks=[2,2,2,2], num_refinement_blocks=2,
                          heads=[1,2,4,8]).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
        model_type = 'baseline'
        model_name = "Restormer (Fair)"

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Evaluate
    results = evaluate_full_images(
        model, args.val_jsonl, args.device,
        args.patch_size, args.stride, model_type,
        args.max_samples, model_name
    )
    results['model'] = args.model
    results['model_name'] = model_name
    results['params'] = num_params
    results['patch_size'] = args.patch_size
    results['stride'] = args.stride

    # Print results
    print()
    print("=" * 70)
    print(f"RESULTS: {model_name}")
    print("=" * 70)
    print(f"PSNR: {results['psnr']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
    print(f"Images evaluated: {results['n_images']}")
    print("=" * 70)

    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = f"tmi_fullimg_{args.model}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
