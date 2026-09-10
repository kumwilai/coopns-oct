#!/usr/bin/env python3
"""
Lightweight boundary fine-tuning on PKU37 using IntensityAnchoredBoundaryLoss.
CPU-compatible version for demonstration and validation.

This script:
1. Loads a pre-trained boundary predictor
2. Fine-tunes boundaries on PKU37 using self-supervised intensity anchoring
3. Evaluates boundary quality before/after fine-tuning
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/kumwilai/OCT')


def load_pku37_images(pku37_root, max_images=5, target_size=256):
    """Load PKU37 images, resized for CPU processing."""
    clean_dir = os.path.join(pku37_root, "clean")

    if not os.path.exists(clean_dir):
        raise FileNotFoundError(f"PKU37 clean dir not found: {clean_dir}")

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])[:max_images]
    images = []

    for fname in clean_files:
        img = Image.open(os.path.join(clean_dir, fname))
        img_np = np.array(img, dtype=np.float32)

        # Normalize
        if img_np.max() > 1:
            img_np = img_np / 255.0

        # Resize for CPU
        H, W = img_np.shape
        if H > target_size or W > target_size:
            scale = min(target_size / H, target_size / W)
            new_H, new_W = int(H * scale), int(W * scale)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((new_W, new_H), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0

        images.append(torch.from_numpy(img_np).unsqueeze(0))  # [1, H, W]

    return images, clean_files


class SimpleBoundaryPredictor(nn.Module):
    """
    Simple boundary predictor for testing.
    In practice, this would be the PhysicsEnsembleV3 or similar.
    """

    def __init__(self, num_boundaries=4, initial_positions=None):
        super().__init__()
        self.num_boundaries = num_boundaries

        # Learnable boundary offsets (per-boundary bias)
        if initial_positions is None:
            # Default: boundaries at 33%, 38%, 43%, 48% (OCT5k-like, too narrow)
            initial_positions = torch.tensor([0.33, 0.38, 0.43, 0.48])

        self.boundary_bias = nn.Parameter(initial_positions.clone())

        # Simple convolution to predict per-column offsets
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_boundaries, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] input image

        Returns:
            boundaries: [B, 4, W] normalized boundary positions
        """
        B, C, H, W = x.shape

        # Predict per-column offsets
        offsets = self.conv(x)  # [B, 4, H, W]
        offsets = offsets.mean(dim=2)  # [B, 4, W] - average over height
        offsets = offsets * 0.1  # Scale offsets

        # Add learnable bias
        boundaries = self.boundary_bias.view(1, -1, 1).expand(B, -1, W) + offsets

        # Ensure ordering: each boundary must be below the previous
        boundaries_ordered = torch.zeros_like(boundaries)
        boundaries_ordered[:, 0, :] = boundaries[:, 0, :]

        for i in range(1, self.num_boundaries):
            boundaries_ordered[:, i, :] = torch.maximum(
                boundaries[:, i, :],
                boundaries_ordered[:, i-1, :] + 0.03  # Minimum 3% gap
            )

        # Clamp to valid range
        boundaries_ordered = torch.clamp(boundaries_ordered, 0.05, 0.95)

        return boundaries_ordered


def compute_boundary_metrics(boundaries, detected_top, detected_bottom):
    """Compute metrics for boundary quality."""
    ilm = boundaries[:, 0, :].mean().item()
    rpe = boundaries[:, -1, :].mean().item()

    # Distance to detected positions
    ilm_error = abs(ilm - detected_top)
    rpe_error = abs(rpe - detected_bottom * 0.95)  # RPE at 95% of detected bottom

    # Boundary spread (should cover retina)
    spread = rpe - ilm

    return {
        'ilm_position': ilm,
        'rpe_position': rpe,
        'ilm_error': ilm_error,
        'rpe_error': rpe_error,
        'spread': spread,
    }


def main():
    print("=" * 70)
    print("Boundary Fine-Tuning with IntensityAnchoredBoundaryLoss (CPU)")
    print("=" * 70)

    # Configuration
    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"
    max_images = 5
    target_size = 128  # Small for CPU
    num_epochs = 10
    lr = 0.01

    # Load images
    print(f"\nLoading PKU37 images (max {max_images}, size {target_size}x{target_size})...")
    images, filenames = load_pku37_images(pku37_root, max_images, target_size)
    print(f"  Loaded {len(images)} images")

    # Import loss
    from train_neurosymbolic_denoising import IntensityAnchoredBoundaryLoss

    # Create boundary predictor with OCT5k-like initial positions (misaligned)
    initial_positions = torch.tensor([0.33, 0.38, 0.43, 0.48])  # Too narrow, wrong position
    model = SimpleBoundaryPredictor(initial_positions=initial_positions)

    # Create loss
    loss_fn = IntensityAnchoredBoundaryLoss(
        lambda_ilm_anchor=1.0,
        lambda_rpe_anchor=1.0,
        lambda_edge_align=0.3,
        lambda_intensity_order=0.2,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Evaluate BEFORE fine-tuning
    print("\n--- Before Fine-Tuning ---")
    model.eval()
    before_metrics = []

    for i, img in enumerate(images):
        img_batch = img.unsqueeze(0)  # [1, 1, H, W]
        boundaries = model(img_batch)

        # Get detected positions
        retina_top, retina_bottom = loss_fn.detect_retina_band(img_batch)
        det_top = retina_top.mean().item()
        det_bottom = retina_bottom.mean().item()

        metrics = compute_boundary_metrics(boundaries, det_top, det_bottom)
        before_metrics.append(metrics)

        print(f"  Image {i+1}: ILM={metrics['ilm_position']:.2%}, RPE={metrics['rpe_position']:.2%}, "
              f"ILM_err={metrics['ilm_error']:.2%}, RPE_err={metrics['rpe_error']:.2%}")

    avg_ilm_err_before = np.mean([m['ilm_error'] for m in before_metrics])
    avg_rpe_err_before = np.mean([m['rpe_error'] for m in before_metrics])
    print(f"\n  Average ILM error: {avg_ilm_err_before:.2%}")
    print(f"  Average RPE error: {avg_rpe_err_before:.2%}")

    # Fine-tune
    print(f"\n--- Fine-Tuning ({num_epochs} epochs) ---")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_losses = {'ilm_anchor': 0, 'rpe_anchor': 0, 'edge_align': 0, 'intensity_order': 0}

        for img in images:
            img_batch = img.unsqueeze(0)  # [1, 1, H, W]

            optimizer.zero_grad()
            boundaries = model(img_batch)
            loss, loss_dict = loss_fn(boundaries, img_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0)

        epoch_loss /= len(images)
        for key in epoch_losses:
            epoch_losses[key] /= len(images)

        if epoch % 2 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss:.4f}, "
                  f"ilm={epoch_losses['ilm_anchor']:.4f}, rpe={epoch_losses['rpe_anchor']:.4f}")

    # Evaluate AFTER fine-tuning
    print("\n--- After Fine-Tuning ---")
    model.eval()
    after_metrics = []

    for i, img in enumerate(images):
        img_batch = img.unsqueeze(0)
        boundaries = model(img_batch)

        retina_top, retina_bottom = loss_fn.detect_retina_band(img_batch)
        det_top = retina_top.mean().item()
        det_bottom = retina_bottom.mean().item()

        metrics = compute_boundary_metrics(boundaries, det_top, det_bottom)
        after_metrics.append(metrics)

        print(f"  Image {i+1}: ILM={metrics['ilm_position']:.2%}, RPE={metrics['rpe_position']:.2%}, "
              f"ILM_err={metrics['ilm_error']:.2%}, RPE_err={metrics['rpe_error']:.2%}")

    avg_ilm_err_after = np.mean([m['ilm_error'] for m in after_metrics])
    avg_rpe_err_after = np.mean([m['rpe_error'] for m in after_metrics])
    print(f"\n  Average ILM error: {avg_ilm_err_after:.2%}")
    print(f"  Average RPE error: {avg_rpe_err_after:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nILM Error: {avg_ilm_err_before:.2%} -> {avg_ilm_err_after:.2%} "
          f"({'improved' if avg_ilm_err_after < avg_ilm_err_before else 'worse'})")
    print(f"RPE Error: {avg_rpe_err_before:.2%} -> {avg_rpe_err_after:.2%} "
          f"({'improved' if avg_rpe_err_after < avg_rpe_err_before else 'worse'})")

    ilm_improvement = (avg_ilm_err_before - avg_ilm_err_after) / avg_ilm_err_before * 100
    rpe_improvement = (avg_rpe_err_before - avg_rpe_err_after) / avg_rpe_err_before * 100

    print(f"\nILM improvement: {ilm_improvement:+.1f}%")
    print(f"RPE improvement: {rpe_improvement:+.1f}%")

    print(f"\nLearned boundary positions:")
    print(f"  ILM: {model.boundary_bias[0].item():.2%}")
    print(f"  RNFL_INL: {model.boundary_bias[1].item():.2%}")
    print(f"  INL_ISOS: {model.boundary_bias[2].item():.2%}")
    print(f"  ISOS_RPE: {model.boundary_bias[3].item():.2%}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if ilm_improvement > 10 and rpe_improvement > 10:
        print("IntensityAnchoredBoundaryLoss successfully adapts boundaries to PKU37!")
        print("The self-supervised domain adaptation approach is working.")
    elif ilm_improvement > 0 or rpe_improvement > 0:
        print("Partial improvement observed. May need more epochs or tuning.")
    else:
        print("No improvement. May need to adjust loss weights or detection method.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
