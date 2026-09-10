#!/usr/bin/env python3
"""
Quick utility to compare different training checkpoints.
Shows which configuration achieved best accuracy and denoising.
"""

import torch
from pathlib import Path

def load_checkpoint_info(ckpt_path):
    """Load and display checkpoint information."""
    if not Path(ckpt_path).exists():
        return None

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    info = {
        'epoch': ckpt.get('epoch', 'N/A'),
        'psnr': ckpt.get('psnr', 'N/A'),
        'psnr_base': ckpt.get('psnr_base', 'N/A'),
        'top1': ckpt.get('top1', 'N/A'),
        'gain_over_base': ckpt.get('gain_over_base', 'N/A'),
    }

    return info

def main():
    checkpoints = [
        ('Original (64% Top-1)', 'checkpoints/end_to_end/best_model.pth'),
        ('Accurate Mode', 'checkpoints/end_to_end_accurate/best_model.pth'),
        ('Max Accuracy Mode', 'checkpoints/end_to_end_max_accuracy/best_model.pth'),
    ]

    print("="*80)
    print("CHECKPOINT COMPARISON")
    print("="*80)
    print()

    results = []
    for name, path in checkpoints:
        info = load_checkpoint_info(path)
        if info:
            results.append((name, info))

    if not results:
        print("No checkpoints found. Train models first!")
        return

    # Display table
    print(f"{'Configuration':<25} {'Epoch':<8} {'Top-1 Acc':<12} {'PSNR':<10} {'Gain/Base':<10}")
    print("-"*80)

    for name, info in results:
        epoch = info['epoch']
        top1 = f"{info['top1']:.1f}%" if isinstance(info['top1'], (int, float)) else 'N/A'
        psnr = f"{info['psnr']:.2f} dB" if isinstance(info['psnr'], (int, float)) else 'N/A'
        gain = f"+{info['gain_over_base']:.2f} dB" if isinstance(info['gain_over_base'], (int, float)) else 'N/A'

        print(f"{name:<25} {epoch:<8} {top1:<12} {psnr:<10} {gain:<10}")

    print()
    print("="*80)

    # Find best
    best_acc = max(results, key=lambda x: x[1]['top1'] if isinstance(x[1]['top1'], (int, float)) else 0)
    best_psnr = max(results, key=lambda x: x[1]['psnr'] if isinstance(x[1]['psnr'], (int, float)) else 0)

    print("BEST RESULTS:")
    print(f"  🎯 Best Accuracy:  {best_acc[0]} ({best_acc[1]['top1']:.1f}%)")
    print(f"  📊 Best PSNR:      {best_psnr[0]} ({best_psnr[1]['psnr']:.2f} dB)")
    print("="*80)

if __name__ == "__main__":
    main()
