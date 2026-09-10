#!/usr/bin/env python3
"""
Evaluate Iterative Neuro-Symbolic Denoising on PKU37 Dataset

Compares:
1. NAFNet baseline (no corrections)
2. NAFNet + 1 iteration of symbolic corrections
3. NAFNet + 3 iterations of symbolic corrections
4. NAFNet + converge (until predicates pass or max_iter)
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
from dataclasses import dataclass
import time

sys.path.insert(0, 'nsnd_oct')


# =============================================================================
# OPTIMIZED STRUCTURE PREDICATE (v13)
# =============================================================================

class OptimizedStructurePredicate(nn.Module):
    """
    Optimized Structure Predicate based on empirical analysis.

    Formula: 0.7 × Int_s3 + 0.2 × (Int_s3 × ResVar_s7) + 0.1 × ResVar_s3

    Achieves ~0.32 correlation with actual reconstruction error.
    """

    def __init__(self):
        super().__init__()
        # Averaging kernels
        for k in [3, 7]:
            kernel = torch.ones(1, 1, k, k) / (k * k)
            self.register_buffer(f'avg_kernel_{k}', kernel)

    def local_stats(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local mean and std."""
        pad = k // 2
        kernel = getattr(self, f'avg_kernel_{k}')
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x_sq_pad = F.pad(x ** 2, (pad, pad, pad, pad), mode='reflect')
        mean = F.conv2d(x_pad, kernel)
        var = F.conv2d(x_sq_pad, kernel) - mean ** 2
        std = torch.sqrt(var.clamp(min=0) + 1e-8)
        return mean, std

    def normalize(self, feat: torch.Tensor, pct: float = 0.95) -> torch.Tensor:
        """Normalize to [0, 1] using percentile."""
        B = feat.shape[0]
        p = torch.quantile(feat.view(B, -1), pct, dim=1, keepdim=True).view(B, 1, 1, 1)
        return (feat / (p + 1e-8)).clamp(0, 1)

    def forward(self, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute structure failure map."""
        residual = noisy - denoised

        # Feature 1: Small-scale intensity (most predictive)
        int_s3, _ = self.local_stats(denoised, 3)
        int_s3_norm = self.normalize(int_s3)

        # Feature 2: Residual variance at different scales
        _, res_std_s3 = self.local_stats(residual, 3)
        _, res_std_s7 = self.local_stats(residual, 7)
        res_var_s3 = self.normalize(res_std_s3 ** 2)
        res_var_s7 = self.normalize(res_std_s7 ** 2)

        # Combined failure map (optimized weights)
        structure_failure = (
            0.70 * int_s3_norm +
            0.20 * (int_s3_norm * res_var_s7) +
            0.10 * res_var_s3
        ).clamp(0, 1)

        # Score (inverse of failure)
        score = 1 - structure_failure.mean(dim=(1, 2, 3))
        satisfied = score > 0.7

        return {
            'satisfied': satisfied,
            'score': score,
            'structure_failure': structure_failure,
            'loss': structure_failure.mean(),
        }


# =============================================================================
# SIMPLE CORRECTOR
# =============================================================================

class SimpleCorrector(nn.Module):
    """
    Simple correction module that refines high-error regions.

    Uses failure map to blend between original and smoothed versions.
    """

    def __init__(self):
        super().__init__()
        # Smoothing kernel
        k = 5
        kernel = torch.ones(1, 1, k, k) / (k * k)
        self.register_buffer('smooth_kernel', kernel)

    def forward(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        failure_map: torch.Tensor,
        alpha: float = 0.3,
    ) -> torch.Tensor:
        """
        Apply correction to high-failure regions.

        Strategy: Blend toward local average in high-error regions
        """
        # Compute local average (smoothed version)
        pad = 2
        denoised_pad = F.pad(denoised, (pad, pad, pad, pad), mode='reflect')
        smoothed = F.conv2d(denoised_pad, self.smooth_kernel)

        # Correction: move toward smoothed in high-failure regions
        # failure_map is [0, 1] where 1 = high error
        correction = smoothed - denoised

        # Apply correction weighted by failure map and alpha
        corrected = denoised + alpha * failure_map * correction

        return corrected.clamp(0, 1)


# =============================================================================
# ITERATIVE DENOISING PIPELINE
# =============================================================================

class IterativeDenoisingPipeline:
    """
    Full iterative neuro-symbolic denoising pipeline.
    """

    def __init__(self, backbone, device='cpu'):
        self.backbone = backbone.to(device).eval()
        self.structure_pred = OptimizedStructurePredicate().to(device)
        self.corrector = SimpleCorrector().to(device)
        self.device = device

    def denoise_baseline(self, noisy: torch.Tensor) -> torch.Tensor:
        """NAFNet baseline only."""
        with torch.no_grad():
            return self.backbone(noisy).clamp(0, 1)

    def denoise_iterative(
        self,
        noisy: torch.Tensor,
        max_iter: int = 5,
        alpha_init: float = 0.3,
        alpha_decay: float = 0.8,
        min_improvement: float = 0.005,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Iterative denoising with symbolic corrections.

        Returns:
            denoised: Final denoised image
            info: Dict with iteration info
        """
        with torch.no_grad():
            # Initial denoising
            denoised = self.backbone(noisy).clamp(0, 1)

            history = []
            prev_score = 0

            for i in range(max_iter):
                # Evaluate structure predicate
                result = self.structure_pred(noisy, denoised)
                score = result['score'].mean().item()
                satisfied = result['satisfied'].all().item()

                history.append({
                    'iteration': i,
                    'score': score,
                    'satisfied': satisfied,
                })

                # Check if satisfied
                if satisfied:
                    break

                # Check improvement
                if i > 0 and (score - prev_score) < min_improvement:
                    break

                # Apply correction with decaying strength
                alpha = alpha_init * (alpha_decay ** i)
                denoised = self.corrector(
                    noisy, denoised,
                    result['structure_failure'],
                    alpha=alpha
                )

                prev_score = score

            # Final evaluation
            final_result = self.structure_pred(noisy, denoised)

            return denoised, {
                'iterations': len(history),
                'final_score': final_result['score'].mean().item(),
                'final_satisfied': final_result['satisfied'].all().item(),
                'history': history,
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


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.outer(g)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size).to(pred.device)

    pad = window_size // 2

    mu1 = F.conv2d(F.pad(pred, (pad, pad, pad, pad), mode='reflect'), window)
    mu2 = F.conv2d(F.pad(target, (pad, pad, pad, pad), mode='reflect'), window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(F.pad(pred ** 2, (pad, pad, pad, pad), mode='reflect'), window) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(target ** 2, (pad, pad, pad, pad), mode='reflect'), window) - mu2_sq
    sigma12 = F.conv2d(F.pad(pred * target, (pad, pad, pad, pad), mode='reflect'), window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# =============================================================================
# MAIN EVALUATION
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
    print(f"Loaded NAFNet checkpoint (PSNR: {ckpt.get('psnr', 'unknown')})")
    return backbone


def load_test_samples(n_samples: int = 20):
    """Load test samples from PKU37."""
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


def main():
    print("=" * 70)
    print("ITERATIVE NEURO-SYMBOLIC DENOISING EVALUATION")
    print("=" * 70)

    # Load model and data
    print("\nLoading NAFNet backbone...")
    backbone = load_nafnet()

    print("\nLoading test samples...")
    samples = load_test_samples(n_samples=15)
    print(f"Loaded {len(samples)} samples")

    # Create pipeline
    pipeline = IterativeDenoisingPipeline(backbone)

    # Results storage
    results = {
        'baseline': {'psnr': [], 'ssim': []},
        'iter_1': {'psnr': [], 'ssim': [], 'score': []},
        'iter_3': {'psnr': [], 'ssim': [], 'score': []},
        'iter_5': {'psnr': [], 'ssim': [], 'score': []},
    }

    print("\n" + "-" * 70)
    print("Evaluating samples...")
    print("-" * 70)

    for i, sample in enumerate(samples):
        noisy = sample['noisy']
        clean = sample['clean']
        name = sample['name']

        # Baseline
        denoised_base = pipeline.denoise_baseline(noisy)
        psnr_base = compute_psnr(denoised_base, clean)
        ssim_base = compute_ssim(denoised_base, clean)
        results['baseline']['psnr'].append(psnr_base)
        results['baseline']['ssim'].append(ssim_base)

        # Iterative corrections
        for max_iter, key in [(1, 'iter_1'), (3, 'iter_3'), (5, 'iter_5')]:
            denoised, info = pipeline.denoise_iterative(noisy, max_iter=max_iter)
            psnr = compute_psnr(denoised, clean)
            ssim = compute_ssim(denoised, clean)
            results[key]['psnr'].append(psnr)
            results[key]['ssim'].append(ssim)
            results[key]['score'].append(info['final_score'])

        # Print per-sample results
        print(f"\n[{i+1}/{len(samples)}] {name}")
        print(f"  Baseline:  PSNR={psnr_base:.2f} dB, SSIM={ssim_base:.4f}")
        print(f"  + 1 iter:  PSNR={results['iter_1']['psnr'][-1]:.2f} dB, "
              f"SSIM={results['iter_1']['ssim'][-1]:.4f}, "
              f"Score={results['iter_1']['score'][-1]:.3f}")
        print(f"  + 3 iter:  PSNR={results['iter_3']['psnr'][-1]:.2f} dB, "
              f"SSIM={results['iter_3']['ssim'][-1]:.4f}, "
              f"Score={results['iter_3']['score'][-1]:.3f}")
        print(f"  + 5 iter:  PSNR={results['iter_5']['psnr'][-1]:.2f} dB, "
              f"SSIM={results['iter_5']['ssim'][-1]:.4f}, "
              f"Score={results['iter_5']['score'][-1]:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<15} {'Pred Score':<15}")
    print("-" * 65)

    for key in ['baseline', 'iter_1', 'iter_3', 'iter_5']:
        psnr_mean = np.mean(results[key]['psnr'])
        psnr_std = np.std(results[key]['psnr'])
        ssim_mean = np.mean(results[key]['ssim'])
        ssim_std = np.std(results[key]['ssim'])

        if 'score' in results[key]:
            score_mean = np.mean(results[key]['score'])
            score_str = f"{score_mean:.3f}"
        else:
            score_str = "N/A"

        label = {
            'baseline': 'NAFNet (baseline)',
            'iter_1': 'NAFNet + 1 iter',
            'iter_3': 'NAFNet + 3 iter',
            'iter_5': 'NAFNet + 5 iter',
        }[key]

        print(f"{label:<20} {psnr_mean:.2f} ± {psnr_std:.2f}    "
              f"{ssim_mean:.4f} ± {ssim_std:.4f}  {score_str}")

    # Improvement analysis
    print("\n" + "-" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("-" * 70)

    baseline_psnr = np.mean(results['baseline']['psnr'])
    for key in ['iter_1', 'iter_3', 'iter_5']:
        iter_psnr = np.mean(results[key]['psnr'])
        delta = iter_psnr - baseline_psnr
        print(f"  {key}: ΔPSNR = {delta:+.3f} dB")

    # Per-sample improvement count
    improved_count = {
        'iter_1': sum(1 for i in range(len(samples))
                     if results['iter_1']['psnr'][i] > results['baseline']['psnr'][i]),
        'iter_3': sum(1 for i in range(len(samples))
                     if results['iter_3']['psnr'][i] > results['baseline']['psnr'][i]),
        'iter_5': sum(1 for i in range(len(samples))
                     if results['iter_5']['psnr'][i] > results['baseline']['psnr'][i]),
    }

    print(f"\n  Samples improved:")
    for key, count in improved_count.items():
        print(f"    {key}: {count}/{len(samples)} ({100*count/len(samples):.0f}%)")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
