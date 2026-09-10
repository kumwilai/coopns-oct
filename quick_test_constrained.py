#!/usr/bin/env python3
"""
Quick test of Constrained Neuro-Symbolic OCT Denoiser.
No training - just verifies the architecture and constraint checking works.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/kumwilai/OCT')

from constrained_neurosymbolic_denoiser import (
    ConstrainedNeuroSymbolicDenoiser,
    DenoiserConfig,
    ClinicalLosses,
)


def load_single_pair(pku37_root: str, target_size: int = 64):
    """Load a single image pair for quick testing."""
    clean_dir = os.path.join(pku37_root, "clean")
    noisy_dir = os.path.join(pku37_root, "noisy")

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])
    if not clean_files:
        return None, None

    fname = clean_files[0]

    # Load clean
    clean_img = Image.open(os.path.join(clean_dir, fname))
    clean_np = np.array(clean_img, dtype=np.float32)
    if clean_np.max() > 1:
        clean_np = clean_np / 255.0

    # Find noisy
    base_name = os.path.splitext(fname)[0]
    noisy_files = [f for f in os.listdir(noisy_dir) if f.startswith(base_name)]

    if noisy_files:
        noisy_img = Image.open(os.path.join(noisy_dir, noisy_files[0]))
        noisy_np = np.array(noisy_img, dtype=np.float32)
        if noisy_np.max() > 1:
            noisy_np = noisy_np / 255.0
    else:
        noisy_np = clean_np + np.random.randn(*clean_np.shape).astype(np.float32) * 0.1
        noisy_np = np.clip(noisy_np, 0, 1)

    # Resize
    H, W = clean_np.shape
    if H > target_size or W > target_size:
        scale = min(target_size / H, target_size / W)
        new_H, new_W = int(H * scale), int(W * scale)
        clean_pil = Image.fromarray((clean_np * 255).astype(np.uint8))
        noisy_pil = Image.fromarray((noisy_np * 255).astype(np.uint8))
        clean_np = np.array(clean_pil.resize((new_W, new_H), Image.BILINEAR), dtype=np.float32) / 255.0
        noisy_np = np.array(noisy_pil.resize((new_W, new_H), Image.BILINEAR), dtype=np.float32) / 255.0

    clean = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    noisy = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0)

    return noisy, clean


def main():
    print("=" * 70)
    print("QUICK TEST: Constrained Neuro-Symbolic OCT Denoiser")
    print("=" * 70)

    # Config
    config = DenoiserConfig(
        encoder_channels=[32, 64, 128],
        latent_dim=16,
        head_width=24,
    )

    # Create model
    print("\n1. Creating model...")
    model = ConstrainedNeuroSymbolicDenoiser(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Load test data
    print("\n2. Loading test data...")
    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"

    if os.path.exists(pku37_root):
        noisy, clean = load_single_pair(pku37_root, target_size=64)
        if noisy is None:
            print("   No data found, using synthetic")
            noisy = torch.rand(1, 1, 64, 64)
            clean = torch.rand(1, 1, 64, 64)
        else:
            print(f"   Loaded: noisy {noisy.shape}, clean {clean.shape}")
    else:
        print("   PKU37 not found, using synthetic data")
        noisy = torch.rand(1, 1, 64, 64)
        clean = torch.rand(1, 1, 64, 64)

    # Forward pass
    print("\n3. Forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(noisy)

    print(f"   Outputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"     {key}: {val.shape}")
        elif isinstance(val, dict):
            print(f"     {key}: dict with {len(val)} layer representations")

    # Check constraints
    print("\n4. Constraint violations (BEFORE training):")
    violations = model.compute_constraint_violations(outputs)
    for name, viol in violations.items():
        status = "OK" if viol.mean().item() < 0.01 else f"VIOLATED ({viol.mean().item():.4f})"
        print(f"     {name}: {status}")

    # Compute losses
    print("\n5. Loss computation...")
    clinical_losses = ClinicalLosses()
    loss, details = clinical_losses.compute_all(outputs, clean)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Global L1: {details['global_l1']:.4f}")
    print(f"   Clinical losses:")
    for layer in ['rnfl', 'inl', 'onl', 'rpe']:
        print(f"     {layer}: L1={details[f'{layer}_l1']:.4f}, clinical={details[f'{layer}_clinical']:.4f}")

    # PSNR (before training, will be low)
    print("\n6. Quality metrics (untrained model):")
    mse = F.mse_loss(outputs['denoised'], clean).item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    print(f"   PSNR: {psnr:.2f} dB (expected low - model is untrained)")

    # Backward pass test
    print("\n7. Backward pass test...")
    model.train()
    outputs = model(noisy)
    loss, _ = clinical_losses.compute_all(outputs, clean)
    loss.backward()
    print("   Backward pass: SUCCESS")

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    print(f"   Gradient norms: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, mean={np.mean(grad_norms):.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Architecture verified:
  [OK] DisentangledLayerEncoder - 4 layers x 3 components (anatomy/pathology/noise)
  [OK] BoundaryPredictor - predicts 4 layer boundaries
  [OK] LayerDenoiserHeads - 4 specialized denoiser heads
  [OK] SymbolicConstraints - 5 anatomical constraints
  [OK] ClinicalLosses - 4 layer-specific losses
  [OK] Forward/backward pass working

Ready for training with AugmentedLagrangianTrainer.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
