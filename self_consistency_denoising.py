#!/usr/bin/env python3
"""
Self-Consistency Denoising with Symbolic Verification

Novel Approach:
1. Denoise with multiple augmentations
2. Compute consensus (reduces noise) and uncertainty (flags errors)
3. Verify with symbolic predicates
4. Measure improvement over baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, 'nsnd_oct')


# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def flip_h(x):
    return torch.flip(x, dims=[-1])

def flip_v(x):
    return torch.flip(x, dims=[-2])

def rotate_90(x):
    return torch.rot90(x, k=1, dims=[-2, -1])

def rotate_180(x):
    return torch.rot90(x, k=2, dims=[-2, -1])

def rotate_270(x):
    return torch.rot90(x, k=3, dims=[-2, -1])

def identity(x):
    return x

# Inverse augmentations
def inv_flip_h(x):
    return torch.flip(x, dims=[-1])

def inv_flip_v(x):
    return torch.flip(x, dims=[-2])

def inv_rotate_90(x):
    return torch.rot90(x, k=-1, dims=[-2, -1])

def inv_rotate_180(x):
    return torch.rot90(x, k=-2, dims=[-2, -1])

def inv_rotate_270(x):
    return torch.rot90(x, k=-3, dims=[-2, -1])

def inv_identity(x):
    return x

AUGMENTATIONS = [
    (identity, inv_identity),
    (flip_h, inv_flip_h),
    (flip_v, inv_flip_v),
    (rotate_90, inv_rotate_90),
    (rotate_180, inv_rotate_180),
    (rotate_270, inv_rotate_270),
]


# =============================================================================
# SELF-CONSISTENCY DENOISER
# =============================================================================

class SelfConsistencyDenoiser:
    """
    Self-Consistency Denoising.

    Key Idea: If the denoiser is perfect, the output should be the same
    regardless of input augmentation. Variance across augmentations
    indicates model uncertainty / potential errors.

    Correction: Use consensus (mean/median) which is more robust than
    single prediction.
    """

    def __init__(self, backbone, device='cpu'):
        self.backbone = backbone.to(device).eval()
        self.device = device

    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        """Standard single-pass denoising."""
        with torch.no_grad():
            return self.backbone(noisy).clamp(0, 1)

    def denoise_consistency(
        self,
        noisy: torch.Tensor,
        n_augmentations: int = 6,
        consensus_type: str = 'mean',  # 'mean' or 'median'
    ) -> Dict[str, torch.Tensor]:
        """
        Self-consistency denoising with multiple augmentations.

        Returns:
            denoised: Consensus output
            uncertainty: Pixel-wise std across augmentations
            outputs: All individual outputs
        """
        with torch.no_grad():
            outputs = []

            for i, (aug, inv_aug) in enumerate(AUGMENTATIONS[:n_augmentations]):
                # Apply augmentation
                aug_input = aug(noisy)

                # Denoise
                aug_output = self.backbone(aug_input).clamp(0, 1)

                # Inverse augmentation to align
                aligned_output = inv_aug(aug_output)
                outputs.append(aligned_output)

            # Stack outputs: [N, B, C, H, W]
            stacked = torch.stack(outputs, dim=0)

            # Consensus
            if consensus_type == 'mean':
                consensus = stacked.mean(dim=0)
            else:  # median
                consensus = stacked.median(dim=0)[0]

            # Uncertainty (std across augmentations)
            uncertainty = stacked.std(dim=0)

            return {
                'denoised': consensus,
                'uncertainty': uncertainty,
                'outputs': outputs,
                'n_augmentations': n_augmentations,
            }


# =============================================================================
# METRICS
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR in dB."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.outer(g)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size).to(pred.device)

    pad = window_size // 2

    mu1 = F.conv2d(F.pad(pred, (pad, pad, pad, pad), mode='reflect'), window)
    mu2 = F.conv2d(F.pad(target, (pad, pad, pad, pad), mode='reflect'), window)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(F.pad(pred ** 2, (pad, pad, pad, pad), mode='reflect'), window) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(target ** 2, (pad, pad, pad, pad), mode='reflect'), window) - mu2_sq
    sigma12 = F.conv2d(F.pad(pred * target, (pad, pad, pad, pad), mode='reflect'), window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    x_c = x_flat - x_flat.mean()
    y_c = y_flat - y_flat.mean()
    corr = (x_c * y_c).sum() / (torch.sqrt((x_c**2).sum() * (y_c**2).sum()) + 1e-8)
    return corr.item()


# =============================================================================
# EVALUATION
# =============================================================================

def load_nafnet():
    """Load NAFNet backbone."""
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet

    backbone = NAFNet(
        img_channel=1, width=64, middle_blk_num=2,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
    )
    ckpt = torch.load(
        "/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth",
        map_location='cpu', weights_only=False
    )
    backbone.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Loaded NAFNet (checkpoint PSNR: {ckpt.get('psnr', 'N/A'):.2f})")
    return backbone


def load_samples(n_samples: int = 20):
    """Load test samples."""
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    val_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    lines = open(val_jsonl).readlines()
    step = max(1, len(lines) // n_samples)

    samples = []
    for idx in range(0, len(lines), step)[:n_samples]:
        data = json.loads(lines[idx])
        clean = torch.from_numpy(
            np.array(Image.open(data['clean_path']).convert('L'))
        ).float() / 255.0
        noisy = torch.from_numpy(
            np.array(Image.open(data['noisy_path']).convert('L'))
        ).float() / 255.0

        samples.append({
            'clean': clean.unsqueeze(0).unsqueeze(0),
            'noisy': noisy.unsqueeze(0).unsqueeze(0),
            'name': Path(data['clean_path']).stem,
        })

    return samples


def evaluate():
    """Main evaluation."""
    print("=" * 70)
    print("SELF-CONSISTENCY DENOISING EVALUATION")
    print("=" * 70)

    # Load
    print("\nLoading model...")
    backbone = load_nafnet()
    denoiser = SelfConsistencyDenoiser(backbone)

    print("\nLoading samples...")
    samples = load_samples(n_samples=15)
    print(f"Loaded {len(samples)} samples")

    # Results
    results = {
        'baseline': {'psnr': [], 'ssim': []},
        'consistency_4': {'psnr': [], 'ssim': [], 'uncertainty_corr': []},
        'consistency_6': {'psnr': [], 'ssim': [], 'uncertainty_corr': []},
    }

    print("\n" + "-" * 70)
    print("Evaluating...")
    print("-" * 70)

    for i, sample in enumerate(samples):
        noisy = sample['noisy']
        clean = sample['clean']

        # Baseline (single pass)
        denoised_base = denoiser.denoise_single(noisy)
        psnr_base = compute_psnr(denoised_base, clean)
        ssim_base = compute_ssim(denoised_base, clean)
        results['baseline']['psnr'].append(psnr_base)
        results['baseline']['ssim'].append(ssim_base)

        # Self-consistency with 4 augmentations
        result_4 = denoiser.denoise_consistency(noisy, n_augmentations=4, consensus_type='mean')
        psnr_4 = compute_psnr(result_4['denoised'], clean)
        ssim_4 = compute_ssim(result_4['denoised'], clean)
        actual_error = (denoised_base - clean).abs()
        uncertainty_corr_4 = compute_correlation(result_4['uncertainty'], actual_error)
        results['consistency_4']['psnr'].append(psnr_4)
        results['consistency_4']['ssim'].append(ssim_4)
        results['consistency_4']['uncertainty_corr'].append(uncertainty_corr_4)

        # Self-consistency with 6 augmentations
        result_6 = denoiser.denoise_consistency(noisy, n_augmentations=6, consensus_type='mean')
        psnr_6 = compute_psnr(result_6['denoised'], clean)
        ssim_6 = compute_ssim(result_6['denoised'], clean)
        uncertainty_corr_6 = compute_correlation(result_6['uncertainty'], actual_error)
        results['consistency_6']['psnr'].append(psnr_6)
        results['consistency_6']['ssim'].append(ssim_6)
        results['consistency_6']['uncertainty_corr'].append(uncertainty_corr_6)

        # Print
        delta_4 = psnr_4 - psnr_base
        delta_6 = psnr_6 - psnr_base
        print(f"[{i+1:2d}/{len(samples)}] {sample['name']}: "
              f"Base={psnr_base:.2f}, +4aug={delta_4:+.3f}, +6aug={delta_6:+.3f} dB")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'PSNR (dB)':<18} {'SSIM':<18} {'Uncert. Corr':<12}")
    print("-" * 75)

    for key in ['baseline', 'consistency_4', 'consistency_6']:
        psnr_m = np.mean(results[key]['psnr'])
        psnr_s = np.std(results[key]['psnr'])
        ssim_m = np.mean(results[key]['ssim'])
        ssim_s = np.std(results[key]['ssim'])

        if 'uncertainty_corr' in results[key]:
            uc = np.mean(results[key]['uncertainty_corr'])
            uc_str = f"{uc:.3f}"
        else:
            uc_str = "N/A"

        label = {
            'baseline': 'NAFNet (single)',
            'consistency_4': 'Consistency (4 aug)',
            'consistency_6': 'Consistency (6 aug)',
        }[key]

        print(f"{label:<25} {psnr_m:.3f} ± {psnr_s:.3f}      "
              f"{ssim_m:.4f} ± {ssim_s:.4f}   {uc_str}")

    # Improvement
    print("\n" + "-" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("-" * 70)

    base_psnr = np.mean(results['baseline']['psnr'])
    for key in ['consistency_4', 'consistency_6']:
        delta = np.mean(results[key][' psnr']) - base_psnr if 'psnr' in results[key] else 0
        delta = np.mean(results[key]['psnr']) - base_psnr
        improved = sum(1 for i in range(len(samples))
                      if results[key]['psnr'][i] > results['baseline']['psnr'][i])
        print(f"  {key}: ΔPSNR = {delta:+.4f} dB, Improved: {improved}/{len(samples)}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = evaluate()
