#!/usr/bin/env python3
"""
Test V4 IntensityAnchoredBoundaryLoss effectiveness in joint denoising + boundary learning.

Compares:
1. Supervised denoising ONLY (no boundary loss)
2. Supervised denoising + V4 self-supervised boundary learning

Metrics:
- Denoising: PSNR, SSIM
- Boundary: Error vs V4-detected positions (proxy for anatomical correctness)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import time

sys.path.insert(0, '/home/kumwilai/OCT')


def load_pku37_pairs(pku37_root: str, max_images: int = 20, target_size: int = 128):
    """Load clean/noisy pairs from PKU37."""
    clean_dir = os.path.join(pku37_root, "clean")
    noisy_dir = os.path.join(pku37_root, "noisy")

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])[:max_images]

    pairs = []
    for fname in clean_files:
        # Load clean
        clean_path = os.path.join(clean_dir, fname)
        clean_img = Image.open(clean_path)
        clean_np = np.array(clean_img, dtype=np.float32)
        if clean_np.max() > 1:
            clean_np = clean_np / 255.0

        # Load noisy
        noisy_path = os.path.join(noisy_dir, fname)
        if os.path.exists(noisy_path):
            noisy_img = Image.open(noisy_path)
            noisy_np = np.array(noisy_img, dtype=np.float32)
            if noisy_np.max() > 1:
                noisy_np = noisy_np / 255.0
        else:
            # Add synthetic noise if no noisy version
            noisy_np = clean_np + np.random.randn(*clean_np.shape).astype(np.float32) * 0.1
            noisy_np = np.clip(noisy_np, 0, 1)

        # Resize if needed
        H, W = clean_np.shape
        if H > target_size or W > target_size:
            scale = min(target_size / H, target_size / W)
            new_H, new_W = int(H * scale), int(W * scale)
            clean_pil = Image.fromarray((clean_np * 255).astype(np.uint8))
            noisy_pil = Image.fromarray((noisy_np * 255).astype(np.uint8))
            clean_np = np.array(clean_pil.resize((new_W, new_H), Image.BILINEAR), dtype=np.float32) / 255.0
            noisy_np = np.array(noisy_pil.resize((new_W, new_H), Image.BILINEAR), dtype=np.float32) / 255.0

        pairs.append({
            'clean': torch.from_numpy(clean_np).unsqueeze(0),  # [1, H, W]
            'noisy': torch.from_numpy(noisy_np).unsqueeze(0),
            'name': fname,
        })

    return pairs


class SimpleJointModel(nn.Module):
    """
    Simple model for joint denoising + boundary prediction.
    Mimics the structure of NeuroSymbolicDenoiser but simplified.
    """

    def __init__(self, num_boundaries: int = 4):
        super().__init__()
        self.num_boundaries = num_boundaries

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # Denoising decoder
        self.denoiser = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

        # Boundary predictor (column-wise)
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_boundaries, 1),
        )

        # Initial boundary positions (learnable bias)
        self.boundary_bias = nn.Parameter(torch.tensor([0.25, 0.40, 0.55, 0.70]))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape

        # Encode
        features = self.encoder(x)

        # Denoise
        denoised = self.denoiser(features)
        denoised = x + denoised  # Residual learning

        # Predict boundaries
        boundary_logits = self.boundary_conv(features)  # [B, 4, H, W]
        # Pool vertically to get column-wise predictions
        boundary_offsets = boundary_logits.mean(dim=2) * 0.1  # [B, 4, W]

        # Add bias and enforce ordering
        boundaries = self.boundary_bias.view(1, -1, 1) + boundary_offsets

        # Enforce ordering constraints
        b0 = boundaries[:, 0:1, :]
        b1 = torch.maximum(boundaries[:, 1:2, :], b0 + 0.05)
        b2 = torch.maximum(boundaries[:, 2:3, :], b1 + 0.05)
        b3 = torch.maximum(boundaries[:, 3:4, :], b2 + 0.05)
        boundaries = torch.cat([b0, b1, b2, b3], dim=1)
        boundaries = torch.clamp(boundaries, 0.05, 0.95)

        return {
            'denoised': denoised,
            'boundaries': boundaries,
        }


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute simple SSIM approximation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = pred.mean()
    mu_target = target.mean()
    sigma_pred = pred.var()
    sigma_target = target.var()
    sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean()

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

    return ssim.item()


def train_and_evaluate(
    model: nn.Module,
    pairs: list,
    use_v4: bool,
    epochs: int = 30,
    lr: float = 0.001,
) -> Dict[str, float]:
    """Train model and evaluate performance."""

    from intensity_anchor_v4 import IntensityAnchoredBoundaryLossV4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # V4 loss for boundary anchoring
    v4_loss_fn = IntensityAnchoredBoundaryLossV4() if use_v4 else None

    # Split into train/test
    n_train = int(len(pairs) * 0.7)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    print(f"\n{'='*60}")
    print(f"Training: {'WITH V4' if use_v4 else 'WITHOUT V4 (denoising only)'}")
    print(f"{'='*60}")
    print(f"Train: {len(train_pairs)} images, Test: {len(test_pairs)} images")

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_denoise_loss = 0
        epoch_v4_loss = 0

        for pair in train_pairs:
            noisy = pair['noisy'].unsqueeze(0)  # [1, 1, H, W]
            clean = pair['clean'].unsqueeze(0)

            optimizer.zero_grad()

            outputs = model(noisy)

            # Denoising loss (supervised)
            denoise_loss = F.l1_loss(outputs['denoised'], clean)

            # V4 boundary loss (self-supervised)
            if use_v4 and v4_loss_fn is not None:
                v4_loss, v4_details = v4_loss_fn(outputs['boundaries'], noisy)
                total_loss = denoise_loss + 0.3 * v4_loss
                epoch_v4_loss += v4_loss.item()
            else:
                total_loss = denoise_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_denoise_loss += denoise_loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / len(train_pairs)
            avg_denoise = epoch_denoise_loss / len(train_pairs)
            if use_v4:
                avg_v4 = epoch_v4_loss / len(train_pairs)
                print(f"Epoch {epoch+1:3d}: total={avg_loss:.4f}, denoise={avg_denoise:.4f}, v4={avg_v4:.4f}")
            else:
                print(f"Epoch {epoch+1:3d}: total={avg_loss:.4f}, denoise={avg_denoise:.4f}")

    # Evaluation
    model.eval()
    results = {
        'psnr': [],
        'ssim': [],
        'ilm_error': [],
        'rpe_error': [],
        'boundary_smoothness': [],
    }

    # For boundary evaluation, use V4 detection as reference
    eval_v4 = IntensityAnchoredBoundaryLossV4()

    with torch.no_grad():
        for pair in test_pairs:
            noisy = pair['noisy'].unsqueeze(0)
            clean = pair['clean'].unsqueeze(0)

            outputs = model(noisy)

            # Denoising metrics
            results['psnr'].append(compute_psnr(outputs['denoised'], clean))
            results['ssim'].append(compute_ssim(outputs['denoised'], clean))

            # Boundary metrics (use V4 detection as pseudo-GT)
            detected_top, detected_bottom, confidence = eval_v4.detect_retina_band(noisy)

            pred_ilm = outputs['boundaries'][:, 0, :].mean().item()
            pred_rpe = outputs['boundaries'][:, -1, :].mean().item()
            det_ilm = detected_top.mean().item()
            det_rpe = detected_bottom.mean().item() * 0.95  # Same scaling as in loss

            results['ilm_error'].append(abs(pred_ilm - det_ilm))
            results['rpe_error'].append(abs(pred_rpe - det_rpe))

            # Boundary smoothness
            boundaries = outputs['boundaries']
            diff = torch.abs(boundaries[:, :, 1:] - boundaries[:, :, :-1])
            results['boundary_smoothness'].append(diff.mean().item())

    # Aggregate results
    final_results = {
        'psnr': np.mean(results['psnr']),
        'ssim': np.mean(results['ssim']),
        'ilm_error': np.mean(results['ilm_error']),
        'rpe_error': np.mean(results['rpe_error']),
        'boundary_error': (np.mean(results['ilm_error']) + np.mean(results['rpe_error'])) / 2,
        'smoothness': np.mean(results['boundary_smoothness']),
    }

    print(f"\nTest Results:")
    print(f"  PSNR: {final_results['psnr']:.2f} dB")
    print(f"  SSIM: {final_results['ssim']:.4f}")
    print(f"  ILM Error: {final_results['ilm_error']:.2%}")
    print(f"  RPE Error: {final_results['rpe_error']:.2%}")
    print(f"  Avg Boundary Error: {final_results['boundary_error']:.2%}")
    print(f"  Boundary Smoothness: {final_results['smoothness']:.4f}")

    return final_results


def main():
    print("=" * 70)
    print("V4 EFFECTIVENESS TEST: Joint Denoising + Boundary Learning")
    print("=" * 70)

    # Load data
    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"

    if not os.path.exists(pku37_root):
        print(f"ERROR: PKU37 dataset not found at {pku37_root}")
        return 1

    print("\nLoading PKU37 data...")
    pairs = load_pku37_pairs(pku37_root, max_images=20, target_size=128)
    print(f"Loaded {len(pairs)} image pairs")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test 1: Without V4 (denoising only)
    print("\n" + "=" * 70)
    print("TEST 1: Supervised Denoising ONLY (no boundary loss)")
    print("=" * 70)
    model_no_v4 = SimpleJointModel()
    results_no_v4 = train_and_evaluate(model_no_v4, pairs, use_v4=False, epochs=30)

    # Reset seed for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)

    # Test 2: With V4 (denoising + boundary)
    print("\n" + "=" * 70)
    print("TEST 2: Supervised Denoising + V4 Self-Supervised Boundaries")
    print("=" * 70)
    model_with_v4 = SimpleJointModel()
    results_with_v4 = train_and_evaluate(model_with_v4, pairs, use_v4=True, epochs=30)

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Without V4':>15} {'With V4':>15} {'Improvement':>15}")
    print("-" * 70)

    metrics = [
        ('PSNR (dB)', 'psnr', True),  # Higher is better
        ('SSIM', 'ssim', True),  # Higher is better
        ('ILM Error', 'ilm_error', False),  # Lower is better
        ('RPE Error', 'rpe_error', False),  # Lower is better
        ('Avg Boundary Error', 'boundary_error', False),  # Lower is better
        ('Smoothness', 'smoothness', False),  # Lower is better
    ]

    for name, key, higher_better in metrics:
        val_no_v4 = results_no_v4[key]
        val_with_v4 = results_with_v4[key]

        if higher_better:
            improvement = val_with_v4 - val_no_v4
            imp_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
        else:
            improvement = val_no_v4 - val_with_v4
            imp_str = f"-{improvement:.4f}" if improvement > 0 else f"+{abs(improvement):.4f}"

        if key in ['ilm_error', 'rpe_error', 'boundary_error']:
            print(f"{name:<25} {val_no_v4:>14.2%} {val_with_v4:>14.2%} {imp_str:>15}")
        elif key == 'psnr':
            print(f"{name:<25} {val_no_v4:>14.2f} {val_with_v4:>14.2f} {imp_str:>15}")
        else:
            print(f"{name:<25} {val_no_v4:>14.4f} {val_with_v4:>14.4f} {imp_str:>15}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    psnr_maintained = abs(results_with_v4['psnr'] - results_no_v4['psnr']) < 0.5
    boundary_improved = results_with_v4['boundary_error'] < results_no_v4['boundary_error']

    if psnr_maintained and boundary_improved:
        boundary_gain = (results_no_v4['boundary_error'] - results_with_v4['boundary_error']) / results_no_v4['boundary_error'] * 100
        print(f"\n✓ V4 is EFFECTIVE!")
        print(f"  - Denoising quality maintained (PSNR within 0.5 dB)")
        print(f"  - Boundary accuracy improved by {boundary_gain:.1f}%")
        print(f"\n  V4 successfully enables self-supervised boundary learning")
        print(f"  alongside supervised denoising without degrading denoising quality.")
    elif boundary_improved:
        psnr_drop = results_no_v4['psnr'] - results_with_v4['psnr']
        print(f"\n△ V4 improves boundaries but affects denoising")
        print(f"  - PSNR dropped by {psnr_drop:.2f} dB")
        print(f"  - Consider reducing lambda_intensity_anchor weight")
    else:
        print(f"\n✗ V4 did not improve boundary accuracy in this test")
        print(f"  - May need more training epochs or hyperparameter tuning")

    return 0


if __name__ == "__main__":
    sys.exit(main())
