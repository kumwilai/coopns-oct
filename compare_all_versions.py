#!/usr/bin/env python3
"""
Final comparison of all IntensityAnchoredBoundaryLoss versions on PKU37.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/kumwilai/OCT')


def load_pku37_images(pku37_root, max_images=10, target_size=128):
    clean_dir = os.path.join(pku37_root, "clean")
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])[:max_images]
    images = []
    for fname in clean_files:
        img = Image.open(os.path.join(clean_dir, fname))
        img_np = np.array(img, dtype=np.float32)
        if img_np.max() > 1:
            img_np = img_np / 255.0
        H, W = img_np.shape
        if H > target_size or W > target_size:
            scale = min(target_size / H, target_size / W)
            new_H, new_W = int(H * scale), int(W * scale)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((new_W, new_H), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(img_np).unsqueeze(0))
    return images


class BoundaryModel(nn.Module):
    def __init__(self, initial_positions=None):
        super().__init__()
        if initial_positions is None:
            initial_positions = torch.tensor([0.33, 0.38, 0.43, 0.48])
        self.bias = nn.Parameter(initial_positions.clone())
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 4, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        offsets = self.conv(x).mean(dim=2) * 0.1
        boundaries = self.bias.view(1, -1, 1).expand(B, -1, W) + offsets

        # Enforce ordering without in-place operations
        b0 = boundaries[:, 0:1, :]
        b1 = torch.maximum(boundaries[:, 1:2, :], b0 + 0.03)
        b2 = torch.maximum(boundaries[:, 2:3, :], b1 + 0.03)
        b3 = torch.maximum(boundaries[:, 3:4, :], b2 + 0.03)
        boundaries = torch.cat([b0, b1, b2, b3], dim=1)

        return torch.clamp(boundaries, 0.05, 0.95)


def evaluate(loss_fn, images, name, epochs=15, lr=0.01):
    """Train and evaluate a loss function."""
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")

    model = BoundaryModel()

    # Check if loss_fn has learnable parameters
    loss_params = list(loss_fn.parameters()) if hasattr(loss_fn, 'parameters') else []
    optimizer = torch.optim.Adam(list(model.parameters()) + loss_params, lr=lr)

    # Get detection function
    if hasattr(loss_fn, 'detect_retina_band_robust'):
        detect_fn = loss_fn.detect_retina_band_robust
    else:
        detect_fn = loss_fn.detect_retina_band

    # Initial errors
    model.eval()
    init_errors = []
    for img in images:
        img_batch = img.unsqueeze(0)
        boundaries = model(img_batch)
        result = detect_fn(img_batch)
        retina_top = result[0] if isinstance(result, tuple) else result
        retina_bottom = result[1] if isinstance(result, tuple) else result

        ilm_err = torch.abs(boundaries[:, 0, :].mean() - retina_top.mean()).item()
        rpe_err = torch.abs(boundaries[:, -1, :].mean() - retina_bottom.mean() * 0.95).item()
        init_errors.append((ilm_err, rpe_err))

    init_ilm = np.mean([e[0] for e in init_errors])
    init_rpe = np.mean([e[1] for e in init_errors])
    print(f"Initial: ILM={init_ilm:.2%}, RPE={init_rpe:.2%}")

    # Train
    model.train()
    for epoch in range(epochs):
        if hasattr(loss_fn, 'set_epoch'):
            loss_fn.set_epoch(epoch)

        epoch_loss = 0
        for img in images:
            optimizer.zero_grad()
            boundaries = model(img.unsqueeze(0))
            loss, _ = loss_fn(boundaries, img.unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/len(images):.4f}")

    # Final errors
    model.eval()
    final_errors = []
    for img in images:
        img_batch = img.unsqueeze(0)
        boundaries = model(img_batch)
        result = detect_fn(img_batch)
        retina_top = result[0]
        retina_bottom = result[1]

        ilm_err = torch.abs(boundaries[:, 0, :].mean() - retina_top.mean()).item()
        rpe_err = torch.abs(boundaries[:, -1, :].mean() - retina_bottom.mean() * 0.95).item()
        final_errors.append((ilm_err, rpe_err))

    final_ilm = np.mean([e[0] for e in final_errors])
    final_rpe = np.mean([e[1] for e in final_errors])

    ilm_improve = (init_ilm - final_ilm) / init_ilm * 100
    rpe_improve = (init_rpe - final_rpe) / init_rpe * 100

    print(f"Final: ILM={final_ilm:.2%}, RPE={final_rpe:.2%}")
    print(f"Improvement: ILM={ilm_improve:+.1f}%, RPE={rpe_improve:+.1f}%")

    return {
        'name': name,
        'init_ilm': init_ilm, 'init_rpe': init_rpe,
        'final_ilm': final_ilm, 'final_rpe': final_rpe,
        'ilm_improve': ilm_improve, 'rpe_improve': rpe_improve,
    }


def main():
    print("=" * 70)
    print("FINAL COMPARISON: IntensityAnchoredBoundaryLoss V1-V4")
    print("=" * 70)

    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"
    images = load_pku37_images(pku37_root, max_images=10)
    print(f"\nLoaded {len(images)} PKU37 images")

    # Import all versions
    from train_neurosymbolic_denoising import IntensityAnchoredBoundaryLoss as V1
    from intensity_anchor_v2 import IntensityAnchoredBoundaryLossV2 as V2
    from intensity_anchor_v3 import IntensityAnchoredBoundaryLossV3 as V3
    from intensity_anchor_v4 import IntensityAnchoredBoundaryLossV4 as V4

    # Create loss functions
    losses = {
        'V1 (Simple)': V1(),
        'V2 (Multi-constraint)': V2(),
        'V3 (Curriculum)': V3(use_adaptive_weights=False),  # Disable unstable adaptive weights
        'V4 (Best-of)': V4(),
    }

    # Evaluate all
    results = {}
    for name, loss_fn in losses.items():
        results[name] = evaluate(loss_fn, images, name)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'Version':<22} {'Init ILM':>10} {'Init RPE':>10} {'Final ILM':>10} {'Final RPE':>10} {'Improve':>10}")
    print("-" * 72)

    for name, r in results.items():
        avg_improve = (r['ilm_improve'] + r['rpe_improve']) / 2
        print(f"{name:<22} {r['init_ilm']:>9.2%} {r['init_rpe']:>9.2%} "
              f"{r['final_ilm']:>9.2%} {r['final_rpe']:>9.2%} {avg_improve:>+9.1f}%")

    # Find best
    print("\n" + "=" * 70)
    print("RANKINGS")
    print("=" * 70)

    # Best initial detection
    best_init = min(results.values(), key=lambda x: (x['init_ilm'] + x['init_rpe']) / 2)
    print(f"\nBest Initial Detection: {best_init['name']}")
    print(f"  ILM: {best_init['init_ilm']:.2%}, RPE: {best_init['init_rpe']:.2%}")

    # Best final error
    best_final = min(results.values(), key=lambda x: (x['final_ilm'] + x['final_rpe']) / 2)
    print(f"\nBest Final Accuracy: {best_final['name']}")
    print(f"  ILM: {best_final['final_ilm']:.2%}, RPE: {best_final['final_rpe']:.2%}")

    # Best improvement
    best_improve = max(results.values(), key=lambda x: (x['ilm_improve'] + x['rpe_improve']) / 2)
    print(f"\nBest Learning/Improvement: {best_improve['name']}")
    print(f"  ILM: {best_improve['ilm_improve']:+.1f}%, RPE: {best_improve['rpe_improve']:+.1f}%")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Score each version
    scores = {}
    for name, r in results.items():
        # Lower error = better, higher improvement = better
        init_score = 1.0 / (1.0 + (r['init_ilm'] + r['init_rpe']))
        final_score = 1.0 / (1.0 + (r['final_ilm'] + r['final_rpe']))
        improve_score = max(0, (r['ilm_improve'] + r['rpe_improve']) / 100)
        scores[name] = init_score * 0.3 + final_score * 0.5 + improve_score * 0.2

    best_overall = max(scores, key=scores.get)
    print(f"\nBest Overall: {best_overall}")
    print(f"\nScores (higher = better):")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
