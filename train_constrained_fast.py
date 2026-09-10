#!/usr/bin/env python3
"""
Fast training of Constrained Neuro-Symbolic OCT Denoiser.
Uses small images and few epochs for quick iteration.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import defaultdict
import time

sys.path.insert(0, '/home/kumwilai/OCT')

from constrained_neurosymbolic_denoiser import (
    ConstrainedNeuroSymbolicDenoiser,
    DenoiserConfig,
    AugmentedLagrangianTrainer,
    ClinicalLosses,
)


def load_pku37_pairs(pku37_root: str, max_images: int = 10, target_size: int = 64):
    """Load clean/noisy pairs from PKU37 dataset."""
    clean_dir = os.path.join(pku37_root, "clean")
    noisy_dir = os.path.join(pku37_root, "noisy")

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])[:max_images]

    pairs = []
    for fname in clean_files:
        clean_path = os.path.join(clean_dir, fname)
        clean_img = Image.open(clean_path)
        clean_np = np.array(clean_img, dtype=np.float32)
        if clean_np.max() > 1:
            clean_np = clean_np / 255.0

        base_name = os.path.splitext(fname)[0]
        noisy_files = [f for f in os.listdir(noisy_dir) if f.startswith(base_name) and f.endswith('.tif')]

        if noisy_files:
            noisy_img = Image.open(os.path.join(noisy_dir, noisy_files[0]))
            noisy_np = np.array(noisy_img, dtype=np.float32)
            if noisy_np.max() > 1:
                noisy_np = noisy_np / 255.0
        else:
            noisy_np = clean_np + np.random.randn(*clean_np.shape).astype(np.float32) * 0.1
            noisy_np = np.clip(noisy_np, 0, 1)

        H, W = clean_np.shape
        if H > target_size or W > target_size:
            scale = min(target_size / H, target_size / W)
            new_H, new_W = int(H * scale), int(W * scale)
            clean_pil = Image.fromarray((clean_np * 255).astype(np.uint8))
            noisy_pil = Image.fromarray((noisy_np * 255).astype(np.uint8))
            clean_np = np.array(clean_pil.resize((new_W, new_H), Image.BILINEAR), dtype=np.float32) / 255.0
            noisy_np = np.array(noisy_pil.resize((new_W, new_H), Image.BILINEAR), dtype=np.float32) / 255.0

        pairs.append({
            'clean': torch.from_numpy(clean_np).unsqueeze(0),
            'noisy': torch.from_numpy(noisy_np).unsqueeze(0),
            'name': fname,
        })

    return pairs


def evaluate(model, pairs, device):
    """Quick evaluation."""
    model.eval()
    psnrs = []
    violations = defaultdict(list)

    with torch.no_grad():
        for pair in pairs:
            noisy = pair['noisy'].unsqueeze(0).to(device)
            clean = pair['clean'].unsqueeze(0).to(device)

            outputs = model(noisy)

            mse = F.mse_loss(outputs['denoised'], clean).item()
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-10)))

            viols = model.compute_constraint_violations(outputs)
            for name, v in viols.items():
                violations[name].append(v.mean().item())

    return {
        'psnr': np.mean(psnrs),
        'violations': {k: np.mean(v) for k, v in violations.items()}
    }


def main():
    print("=" * 70)
    print("FAST TRAINING: Constrained Neuro-Symbolic OCT Denoiser")
    print("=" * 70)

    # Load data
    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"

    print("\n1. Loading data...")
    pairs = load_pku37_pairs(pku37_root, max_images=10, target_size=64)
    print(f"   Loaded {len(pairs)} image pairs")

    # Split
    n_train = 7
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]
    print(f"   Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Config
    config = DenoiserConfig(
        encoder_channels=[32, 64, 128],
        latent_dim=16,
        head_width=24,
        lr=2e-3,
        rho_init=0.1,
        rho_max=10.0,
        rho_mult=1.2,
        disentangle_weight=0.05,
    )

    # Model
    print("\n2. Creating model...")
    torch.manual_seed(42)
    model = ConstrainedNeuroSymbolicDenoiser(config)
    device = torch.device('cpu')
    model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initial evaluation
    print("\n3. Initial evaluation (before training)...")
    init_results = evaluate(model, test_pairs, device)
    print(f"   PSNR: {init_results['psnr']:.2f} dB")
    print(f"   Constraints:")
    for name, viol in init_results['violations'].items():
        status = "OK" if viol < 0.01 else f"VIOL ({viol:.4f})"
        print(f"     {name}: {status}")

    # Training
    print("\n4. Training (15 epochs)...")
    print("-" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    clinical_losses = ClinicalLosses()

    # Lagrange multipliers
    lambdas = {name: 1.0 for name in ['ordering', 'thickness', 'range', 'smoothness', 'intensity']}
    rho = config.rho_init

    for epoch in range(15):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for pair in train_pairs:
            noisy = pair['noisy'].unsqueeze(0).to(device)
            clean = pair['clean'].unsqueeze(0).to(device)

            optimizer.zero_grad()

            outputs = model(noisy)

            # Clinical loss
            clinical_loss, _ = clinical_losses.compute_all(outputs, clean)

            # Constraint penalties (simplified augmented Lagrangian)
            violations = model.compute_constraint_violations(outputs)
            constraint_penalty = 0
            for name, viol in violations.items():
                v = viol.mean()
                constraint_penalty += 0.1 * (lambdas[name] * v + (rho / 2) * F.relu(v) ** 2)

            loss = clinical_loss + constraint_penalty
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Update Lagrangian parameters
        with torch.no_grad():
            test_viols = evaluate(model, test_pairs[:2], device)['violations']
            max_viol = max(test_viols.values())
            if max_viol > 0.01:
                for name, viol in test_viols.items():
                    lambdas[name] = max(0, lambdas[name] + rho * viol)
                rho = min(rho * config.rho_mult, config.rho_max)

        epoch_time = time.time() - epoch_start

        if epoch % 3 == 0 or epoch == 14:
            results = evaluate(model, test_pairs, device)
            n_satisfied = sum(1 for v in results['violations'].values() if v < 0.01)
            print(f"Epoch {epoch+1:2d} | Loss: {epoch_loss/len(train_pairs):.4f} | "
                  f"PSNR: {results['psnr']:.2f} dB | "
                  f"Constraints: {n_satisfied}/5 | "
                  f"ρ: {rho:.2f} | Time: {epoch_time:.1f}s")

    # Final evaluation
    print("-" * 70)
    print("\n5. Final evaluation...")
    final_results = evaluate(model, test_pairs, device)

    print(f"\n   PSNR: {init_results['psnr']:.2f} dB -> {final_results['psnr']:.2f} dB "
          f"(+{final_results['psnr'] - init_results['psnr']:.2f} dB)")

    print(f"\n   Constraint satisfaction:")
    for name in final_results['violations']:
        init_v = init_results['violations'][name]
        final_v = final_results['violations'][name]
        init_ok = init_v < 0.01
        final_ok = final_v < 0.01
        init_status = "OK" if init_ok else f"{init_v:.4f}"
        final_status = "OK" if final_ok else f"{final_v:.4f}"

        if final_ok and init_ok:
            arrow = "maintained"
        elif final_ok and not init_ok:
            arrow = "FIXED"
        elif final_v < init_v:
            arrow = "improved"
        else:
            arrow = "degraded"
        print(f"     {name}: {init_status} -> {final_status} ({arrow})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    psnr_improved = final_results['psnr'] > init_results['psnr'] + 1.0
    n_satisfied = sum(1 for v in final_results['violations'].values() if v < 0.01)

    if psnr_improved and n_satisfied >= 4:
        print(f"\n[SUCCESS] Training effective:")
        print(f"  - PSNR improved by {final_results['psnr'] - init_results['psnr']:.2f} dB")
        print(f"  - {n_satisfied}/5 constraints satisfied")
    elif psnr_improved:
        print(f"\n[PARTIAL] PSNR improved but constraints need tuning")
    else:
        print(f"\n[CHECK] Model may need more epochs or hyperparameter tuning")

    return 0


if __name__ == "__main__":
    sys.exit(main())
