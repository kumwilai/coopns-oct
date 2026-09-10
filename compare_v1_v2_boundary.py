#!/usr/bin/env python3
"""
Compare IntensityAnchoredBoundaryLoss V1 vs V2 on PKU37 images.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/kumwilai/OCT')


def load_pku37_images(pku37_root, max_images=10, target_size=128):
    """Load PKU37 images."""
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

    return images, clean_files


class SimpleBoundaryModel(nn.Module):
    """Simple boundary predictor for testing."""

    def __init__(self, num_boundaries=4, initial_positions=None):
        super().__init__()
        if initial_positions is None:
            initial_positions = torch.tensor([0.33, 0.38, 0.43, 0.48])
        self.boundary_bias = nn.Parameter(initial_positions.clone())
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_boundaries, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        offsets = self.conv(x).mean(dim=2) * 0.1
        boundaries = self.boundary_bias.view(1, -1, 1).expand(B, -1, W) + offsets

        # Enforce ordering
        boundaries_ordered = torch.zeros_like(boundaries)
        boundaries_ordered[:, 0, :] = boundaries[:, 0, :]
        for i in range(1, 4):
            boundaries_ordered[:, i, :] = torch.maximum(
                boundaries[:, i, :],
                boundaries_ordered[:, i-1, :] + 0.03
            )
        return torch.clamp(boundaries_ordered, 0.05, 0.95)


def train_and_evaluate(loss_fn, images, name, num_epochs=15, lr=0.01):
    """Train boundary model with given loss function."""
    print(f"\n{'='*60}")
    print(f"Training with {name}")
    print(f"{'='*60}")

    # Create model
    initial_positions = torch.tensor([0.33, 0.38, 0.43, 0.48])
    model = SimpleBoundaryModel(initial_positions=initial_positions)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get initial metrics
    model.eval()
    initial_errors = []
    for img in images:
        img_batch = img.unsqueeze(0)
        boundaries = model(img_batch)

        if hasattr(loss_fn, 'detect_retina_band_robust'):
            retina_top, retina_bottom, _ = loss_fn.detect_retina_band_robust(img_batch)
        else:
            retina_top, retina_bottom = loss_fn.detect_retina_band(img_batch)

        ilm_err = torch.abs(boundaries[:, 0, :].mean() - retina_top.mean()).item()
        rpe_err = torch.abs(boundaries[:, -1, :].mean() - retina_bottom.mean() * 0.95).item()
        initial_errors.append({'ilm': ilm_err, 'rpe': rpe_err})

    avg_ilm_init = np.mean([e['ilm'] for e in initial_errors])
    avg_rpe_init = np.mean([e['rpe'] for e in initial_errors])
    print(f"Initial: ILM_err={avg_ilm_init:.2%}, RPE_err={avg_rpe_init:.2%}")

    # Train
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for img in images:
            img_batch = img.unsqueeze(0)
            optimizer.zero_grad()
            boundaries = model(img_batch)
            loss, loss_dict = loss_fn(boundaries, img_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(images)
        loss_history.append(epoch_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss:.4f}")

    # Final metrics
    model.eval()
    final_errors = []
    for img in images:
        img_batch = img.unsqueeze(0)
        boundaries = model(img_batch)

        if hasattr(loss_fn, 'detect_retina_band_robust'):
            retina_top, retina_bottom, _ = loss_fn.detect_retina_band_robust(img_batch)
        else:
            retina_top, retina_bottom = loss_fn.detect_retina_band(img_batch)

        ilm_err = torch.abs(boundaries[:, 0, :].mean() - retina_top.mean()).item()
        rpe_err = torch.abs(boundaries[:, -1, :].mean() - retina_bottom.mean() * 0.95).item()
        final_errors.append({'ilm': ilm_err, 'rpe': rpe_err})

    avg_ilm_final = np.mean([e['ilm'] for e in final_errors])
    avg_rpe_final = np.mean([e['rpe'] for e in final_errors])

    ilm_improve = (avg_ilm_init - avg_ilm_final) / avg_ilm_init * 100
    rpe_improve = (avg_rpe_init - avg_rpe_final) / avg_rpe_init * 100

    print(f"\nFinal: ILM_err={avg_ilm_final:.2%}, RPE_err={avg_rpe_final:.2%}")
    print(f"Improvement: ILM={ilm_improve:+.1f}%, RPE={rpe_improve:+.1f}%")

    # Get final boundary positions
    final_positions = model.boundary_bias.detach().numpy()

    return {
        'name': name,
        'initial_ilm_err': avg_ilm_init,
        'initial_rpe_err': avg_rpe_init,
        'final_ilm_err': avg_ilm_final,
        'final_rpe_err': avg_rpe_final,
        'ilm_improvement': ilm_improve,
        'rpe_improvement': rpe_improve,
        'final_positions': final_positions,
        'loss_history': loss_history,
    }


def train_and_evaluate_v3(loss_fn, images, name, num_epochs=15, lr=0.01):
    """Train with V3 (curriculum learning support)."""
    print(f"\n{'='*60}")
    print(f"Training with {name}")
    print(f"{'='*60}")

    initial_positions = torch.tensor([0.33, 0.38, 0.43, 0.48])
    model = SimpleBoundaryModel(initial_positions=initial_positions)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)

    # Get initial metrics
    model.eval()
    initial_errors = []
    for img in images:
        img_batch = img.unsqueeze(0)
        boundaries = model(img_batch)
        retina_top, retina_bottom, _ = loss_fn.detect_retina_band(img_batch)
        ilm_err = torch.abs(boundaries[:, 0, :].mean() - retina_top.mean()).item()
        rpe_err = torch.abs(boundaries[:, -1, :].mean() - retina_bottom.mean() * 0.95).item()
        initial_errors.append({'ilm': ilm_err, 'rpe': rpe_err})

    avg_ilm_init = np.mean([e['ilm'] for e in initial_errors])
    avg_rpe_init = np.mean([e['rpe'] for e in initial_errors])
    print(f"Initial: ILM_err={avg_ilm_init:.2%}, RPE_err={avg_rpe_init:.2%}")

    # Train with curriculum
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        loss_fn.set_epoch(epoch)  # Update curriculum
        epoch_loss = 0
        for img in images:
            img_batch = img.unsqueeze(0)
            optimizer.zero_grad()
            boundaries = model(img_batch)
            loss, loss_dict = loss_fn(boundaries, img_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(images)
        loss_history.append(epoch_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            progress = loss_dict.get('curriculum_progress', 1.0)
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss:.4f} (curriculum={progress:.0%})")

    # Final metrics
    model.eval()
    final_errors = []
    for img in images:
        img_batch = img.unsqueeze(0)
        boundaries = model(img_batch)
        retina_top, retina_bottom, _ = loss_fn.detect_retina_band(img_batch)
        ilm_err = torch.abs(boundaries[:, 0, :].mean() - retina_top.mean()).item()
        rpe_err = torch.abs(boundaries[:, -1, :].mean() - retina_bottom.mean() * 0.95).item()
        final_errors.append({'ilm': ilm_err, 'rpe': rpe_err})

    avg_ilm_final = np.mean([e['ilm'] for e in final_errors])
    avg_rpe_final = np.mean([e['rpe'] for e in final_errors])

    ilm_improve = (avg_ilm_init - avg_ilm_final) / avg_ilm_init * 100
    rpe_improve = (avg_rpe_init - avg_rpe_final) / avg_rpe_init * 100

    print(f"\nFinal: ILM_err={avg_ilm_final:.2%}, RPE_err={avg_rpe_final:.2%}")
    print(f"Improvement: ILM={ilm_improve:+.1f}%, RPE={rpe_improve:+.1f}%")

    final_positions = model.boundary_bias.detach().numpy()

    return {
        'name': name,
        'initial_ilm_err': avg_ilm_init,
        'initial_rpe_err': avg_rpe_init,
        'final_ilm_err': avg_ilm_final,
        'final_rpe_err': avg_rpe_final,
        'ilm_improvement': ilm_improve,
        'rpe_improvement': rpe_improve,
        'final_positions': final_positions,
        'loss_history': loss_history,
    }


def main():
    print("=" * 70)
    print("Comparing IntensityAnchoredBoundaryLoss V1 vs V2 vs V3")
    print("=" * 70)

    # Load images
    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"
    images, _ = load_pku37_images(pku37_root, max_images=10, target_size=128)
    print(f"\nLoaded {len(images)} PKU37 images")

    # Import all versions
    from train_neurosymbolic_denoising import IntensityAnchoredBoundaryLoss as V1
    from intensity_anchor_v2 import IntensityAnchoredBoundaryLossV2 as V2
    from intensity_anchor_v3 import IntensityAnchoredBoundaryLossV3 as V3

    # Create loss functions
    loss_v1 = V1(
        lambda_ilm_anchor=1.0,
        lambda_rpe_anchor=1.0,
        lambda_edge_align=0.5,
        lambda_intensity_order=0.3,
    )

    loss_v2 = V2(
        lambda_ilm_anchor=1.0,
        lambda_rpe_anchor=1.0,
        lambda_middle_anchor=0.5,
        lambda_edge_align=0.5,
        lambda_gradient_magnitude=0.3,
        lambda_intensity_order=0.3,
        lambda_layer_proportion=0.3,
        lambda_smoothness=0.2,
    )

    loss_v3 = V3(
        lambda_ilm_anchor=1.0,
        lambda_rpe_anchor=1.0,
        lambda_edge_align=0.5,
        lambda_smoothness=0.2,
        use_curriculum=True,
        curriculum_epochs=10,
        use_adaptive_weights=True,
    )

    # Train and evaluate
    results_v1 = train_and_evaluate(loss_v1, images, "V1 (Original)")
    results_v2 = train_and_evaluate(loss_v2, images, "V2 (Multi-constraint)")
    results_v3 = train_and_evaluate_v3(loss_v3, images, "V3 (Curriculum+Adaptive)")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'V1':>10} {'V2':>10} {'V3':>10} {'Winner':>10}")
    print("-" * 65)

    # Extract metrics
    v1_ilm = results_v1['ilm_improvement']
    v2_ilm = results_v2['ilm_improvement']
    v3_ilm = results_v3['ilm_improvement']

    v1_rpe = results_v1['rpe_improvement']
    v2_rpe = results_v2['rpe_improvement']
    v3_rpe = results_v3['rpe_improvement']

    v1_ilm_err = results_v1['final_ilm_err']
    v2_ilm_err = results_v2['final_ilm_err']
    v3_ilm_err = results_v3['final_ilm_err']

    v1_rpe_err = results_v1['final_rpe_err']
    v2_rpe_err = results_v2['final_rpe_err']
    v3_rpe_err = results_v3['final_rpe_err']

    # ILM improvement
    best = max(v1_ilm, v2_ilm, v3_ilm)
    winner = "V1" if v1_ilm == best else ("V2" if v2_ilm == best else "V3")
    print(f"{'ILM Improvement':<25} {v1_ilm:>+9.1f}% {v2_ilm:>+9.1f}% {v3_ilm:>+9.1f}% {winner:>10}")

    # RPE improvement
    best = max(v1_rpe, v2_rpe, v3_rpe)
    winner = "V1" if v1_rpe == best else ("V2" if v2_rpe == best else "V3")
    print(f"{'RPE Improvement':<25} {v1_rpe:>+9.1f}% {v2_rpe:>+9.1f}% {v3_rpe:>+9.1f}% {winner:>10}")

    # Final ILM error (lower is better)
    best = min(v1_ilm_err, v2_ilm_err, v3_ilm_err)
    winner = "V1" if v1_ilm_err == best else ("V2" if v2_ilm_err == best else "V3")
    print(f"{'Final ILM Error':<25} {v1_ilm_err:>9.2%} {v2_ilm_err:>9.2%} {v3_ilm_err:>9.2%} {winner:>10}")

    # Final RPE error (lower is better)
    best = min(v1_rpe_err, v2_rpe_err, v3_rpe_err)
    winner = "V1" if v1_rpe_err == best else ("V2" if v2_rpe_err == best else "V3")
    print(f"{'Final RPE Error':<25} {v1_rpe_err:>9.2%} {v2_rpe_err:>9.2%} {v3_rpe_err:>9.2%} {winner:>10}")

    # Average improvement
    v1_total = (v1_ilm + v1_rpe) / 2
    v2_total = (v2_ilm + v2_rpe) / 2
    v3_total = (v3_ilm + v3_rpe) / 2
    best = max(v1_total, v2_total, v3_total)
    winner = "V1" if v1_total == best else ("V2" if v2_total == best else "V3")
    print(f"{'Avg Improvement':<25} {v1_total:>+9.1f}% {v2_total:>+9.1f}% {v3_total:>+9.1f}% {winner:>10}")

    # Average final error
    v1_err_avg = (v1_ilm_err + v1_rpe_err) / 2
    v2_err_avg = (v2_ilm_err + v2_rpe_err) / 2
    v3_err_avg = (v3_ilm_err + v3_rpe_err) / 2
    best = min(v1_err_avg, v2_err_avg, v3_err_avg)
    winner = "V1" if v1_err_avg == best else ("V2" if v2_err_avg == best else "V3")
    print(f"{'Avg Final Error':<25} {v1_err_avg:>9.2%} {v2_err_avg:>9.2%} {v3_err_avg:>9.2%} {winner:>10}")

    print("\n" + "-" * 65)

    # Final positions comparison
    print("\nLearned Boundary Positions:")
    print(f"{'Boundary':<12} {'V1':>10} {'V2':>10} {'V3':>10}")
    print("-" * 42)
    names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']
    for i, name in enumerate(names):
        v1_pos = results_v1['final_positions'][i]
        v2_pos = results_v2['final_positions'][i]
        v3_pos = results_v3['final_positions'][i]
        print(f"{name:<12} {v1_pos:>9.2%} {v2_pos:>9.2%} {v3_pos:>9.2%}")

    # Count wins
    scores = {'V1': 0, 'V2': 0, 'V3': 0}
    # Improvement wins (higher is better)
    scores['V1' if v1_ilm == max(v1_ilm, v2_ilm, v3_ilm) else ('V2' if v2_ilm == max(v1_ilm, v2_ilm, v3_ilm) else 'V3')] += 1
    scores['V1' if v1_rpe == max(v1_rpe, v2_rpe, v3_rpe) else ('V2' if v2_rpe == max(v1_rpe, v2_rpe, v3_rpe) else 'V3')] += 1
    # Error wins (lower is better)
    scores['V1' if v1_ilm_err == min(v1_ilm_err, v2_ilm_err, v3_ilm_err) else ('V2' if v2_ilm_err == min(v1_ilm_err, v2_ilm_err, v3_ilm_err) else 'V3')] += 1
    scores['V1' if v1_rpe_err == min(v1_rpe_err, v2_rpe_err, v3_rpe_err) else ('V2' if v2_rpe_err == min(v1_rpe_err, v2_rpe_err, v3_rpe_err) else 'V3')] += 1

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print(f"\nWins: V1={scores['V1']}, V2={scores['V2']}, V3={scores['V3']}")

    best_version = max(scores, key=scores.get)
    if scores[best_version] >= 3:
        print(f"\n{best_version} is the best performer!")
    else:
        print("\nResults are mixed - consider combining approaches.")

    print("\nVersion characteristics:")
    print("  V1: Simple, fast, good improvement rate")
    print("  V2: Rich constraints, better detection, may over-constrain")
    print("  V3: Curriculum learning, adaptive weights, balanced approach")

    return 0


if __name__ == "__main__":
    sys.exit(main())
