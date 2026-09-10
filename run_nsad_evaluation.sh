#!/bin/bash
#
# NSAD Evaluation Script
# Evaluate trained NSAD model against baseline
#
# Usage: ./run_nsad_evaluation.sh <checkpoint_path> [gpu|cpu]
#

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./run_nsad_evaluation.sh <checkpoint_path> [device]"
    echo ""
    echo "Available checkpoints:"
    ls -la checkpoints/*/best_model.pth 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

CHECKPOINT=$1
DEVICE=${2:-cpu}

echo "========================================"
echo "NSAD Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Device: $DEVICE"
echo "========================================"

# Create evaluation script
python << 'EOF'
import sys
import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add paths
sys.path.insert(0, '.')
from nsnd.models.sansd import SANSDWithBackbone
from nsnd.utils.metrics import compute_psnr, compute_ssim

def load_model(checkpoint_path, device):
    """Load trained NSAD model."""
    model = SANSDWithBackbone(
        backbone_width=64,
        fusion_mode='gated'
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, val_jsonl, device, patch_size=64, max_samples=100):
    """Evaluate model on validation set."""
    # Load validation samples
    samples = []
    with open(val_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line.strip()))

    results = {
        'psnr_noisy': [],
        'psnr_denoised': [],
        'ssim_noisy': [],
        'ssim_denoised': [],
        'noise_type_predictions': [],
    }

    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating"):
            # Load images
            noisy = np.array(Image.open(sample['noisy_path']).convert('L'), dtype=np.float32) / 255.0
            clean = np.array(Image.open(sample['clean_path']).convert('L'), dtype=np.float32) / 255.0

            # Center crop
            h, w = noisy.shape
            top = (h - patch_size) // 2
            left = (w - patch_size) // 2
            noisy = noisy[top:top+patch_size, left:left+patch_size]
            clean = clean[top:top+patch_size, left:left+patch_size]

            # To tensor
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

            # Forward pass
            denoised, interpretation = model(noisy_t, return_interpretation=True)
            denoised = denoised.squeeze().cpu().numpy()

            # Compute metrics
            results['psnr_noisy'].append(compute_psnr(noisy, clean))
            results['psnr_denoised'].append(compute_psnr(denoised, clean))
            results['ssim_noisy'].append(compute_ssim(noisy, clean))
            results['ssim_denoised'].append(compute_ssim(denoised, clean))

            # Store noise type prediction
            noise_type = interpretation['noise_type'].squeeze().cpu().numpy()
            results['noise_type_predictions'].append(noise_type.mean(axis=(1, 2)))  # Average per-channel

    return results

if __name__ == '__main__':
    import os

    checkpoint_path = os.environ.get('CHECKPOINT', 'checkpoints/sansd_quick_test/best_model.pth')
    device = os.environ.get('DEVICE', 'cpu')

    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)

    print("\nEvaluating on validation set...")
    results = evaluate_model(model, 'weights_duke_analysis_maps_val.jsonl', device)

    # Compute statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    avg_psnr_noisy = np.mean(results['psnr_noisy'])
    avg_psnr_denoised = np.mean(results['psnr_denoised'])
    avg_ssim_noisy = np.mean(results['ssim_noisy'])
    avg_ssim_denoised = np.mean(results['ssim_denoised'])

    print(f"\nPSNR (noisy):    {avg_psnr_noisy:.2f} dB")
    print(f"PSNR (denoised): {avg_psnr_denoised:.2f} dB")
    print(f"PSNR Gain:       +{avg_psnr_denoised - avg_psnr_noisy:.2f} dB")

    print(f"\nSSIM (noisy):    {avg_ssim_noisy:.4f}")
    print(f"SSIM (denoised): {avg_ssim_denoised:.4f}")
    print(f"SSIM Gain:       +{avg_ssim_denoised - avg_ssim_noisy:.4f}")

    # Noise type usage
    noise_types = np.array(results['noise_type_predictions'])
    avg_usage = noise_types.mean(axis=0)
    print(f"\nExpert Usage (average):")
    print(f"  Speckle:  {avg_usage[0]:.2%}")
    print(f"  Banding:  {avg_usage[1]:.2%}")
    print(f"  Gaussian: {avg_usage[2]:.2%}")
    print(f"  Shot:     {avg_usage[3]:.2%}")

    print("\n" + "="*60)
EOF

echo ""
echo "Evaluation complete!"
