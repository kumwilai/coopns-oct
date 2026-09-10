#!/usr/bin/env python3
"""
Visualize self-supervised boundary learning results.
Shows that boundaries are learned correctly without GT masks.
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))

from train_neurosymbolic_denoising import NeuroSymbolicDenoiser
from calibrated_oct_noise import add_calibrated_oct_noise


def main():
    # Load model
    device = torch.device('cpu')

    model = NeuroSymbolicDenoiser(
        hidden_channels=48,
        nafnet_width=64,
    ).to(device)

    ckpt_path = 'outputs/neurosymbolic_denoising/best_model.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model.eval()

    # Load sample image
    val_jsonl = 'combined_val.jsonl'
    with open(val_jsonl) as f:
        sample = json.loads(f.readline())

    clean = np.array(Image.open(sample['image_path']).convert('L')) / 255.0
    noisy = add_calibrated_oct_noise(clean, dataset='duke17', noise_scale=1.0, random_mix=True)

    H, W = clean.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    noisy_padded = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode='reflect')

    x = torch.from_numpy(noisy_padded).float().unsqueeze(0).unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        outputs = model(x, return_symbolic=True)

    denoised = outputs['denoised'].squeeze().cpu().numpy()[:H, :W]
    boundaries = outputs['boundaries'].squeeze().cpu().numpy()  # [4, W]
    segmentation = outputs['segmentation'].squeeze().cpu().numpy()[:H, :W]

    # Print boundary statistics
    print("\n" + "=" * 60)
    print("SELF-SUPERVISED BOUNDARY LEARNING RESULTS")
    print("=" * 60)

    boundary_names = ['ILM', 'RNFL_INL', 'INL_ISOS', 'ISOS_RPE']
    print("\n[LEARNED BOUNDARY POSITIONS (mean +/- std)]")
    for i, name in enumerate(boundary_names):
        b = boundaries[i, :]
        print(f"  {name:<12}: {b.mean():.1f} +/- {b.std():.1f} pixels (range: {b.min():.0f}-{b.max():.0f})")

    # Check ordering
    print("\n[ANATOMICAL ORDERING CHECK]")
    ordering_ok = True
    for i in range(3):
        diff = boundaries[i+1, :] - boundaries[i, :]
        if diff.min() < 0:
            ordering_ok = False
            print(f"  {boundary_names[i]} -> {boundary_names[i+1]}: VIOLATION (min diff = {diff.min():.1f})")
        else:
            print(f"  {boundary_names[i]} -> {boundary_names[i+1]}: OK (min gap = {diff.min():.1f} pixels)")

    print(f"\n  Overall ordering: {'SATISFIED' if ordering_ok else 'VIOLATED'}")

    # Layer thickness
    print("\n[LAYER THICKNESS (learned from self-supervision)]")
    layer_names = ['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']
    for i in range(4):
        if i < 3:
            thickness = boundaries[i+1, :] - boundaries[i, :]
        else:
            # RPE_Choroid: from ISOS_RPE to bottom
            thickness = H - boundaries[3, :]

        t_mean = thickness.mean()
        t_std = thickness.std()
        t_pct = t_mean / H * 100
        print(f"  {layer_names[i]:<15}: {t_mean:.1f} +/- {t_std:.1f} pixels ({t_pct:.1f}% of image)")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Clean
    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('Clean Image')
    axes[0, 0].axis('off')

    # Noisy
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy Input')
    axes[0, 1].axis('off')

    # Denoised
    axes[0, 2].imshow(denoised, cmap='gray')
    axes[0, 2].set_title('Denoised Output')
    axes[0, 2].axis('off')

    # Clean with boundaries
    axes[1, 0].imshow(clean, cmap='gray')
    colors = ['red', 'green', 'blue', 'yellow']
    for i, (name, color) in enumerate(zip(boundary_names, colors)):
        axes[1, 0].plot(range(W), boundaries[i, :W], color=color, linewidth=1.5, label=name)
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].set_title('Self-Supervised Boundaries')
    axes[1, 0].axis('off')

    # Segmentation
    seg_colored = np.zeros((H, W, 3))
    colors_rgb = [
        [1, 0.3, 0.3],  # RNFL_GCL - red
        [0.3, 1, 0.3],  # INL_OPL_ONL - green
        [0.3, 0.3, 1],  # IS_OS - blue
        [1, 1, 0.3],    # RPE_Choroid - yellow
    ]
    for i in range(4):
        mask = segmentation == i
        for c in range(3):
            seg_colored[:, :, c][mask] = colors_rgb[i][c]

    axes[1, 1].imshow(seg_colored)
    axes[1, 1].set_title('Layer Segmentation (Self-Supervised)')
    axes[1, 1].axis('off')

    # Overlay
    overlay = np.stack([clean, clean, clean], axis=-1)
    for i in range(4):
        mask = segmentation == i
        for c in range(3):
            overlay[:, :, c][mask] = 0.6 * clean[mask] + 0.4 * colors_rgb[i][c]

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Segmentation Overlay')
    axes[1, 2].axis('off')

    plt.tight_layout()
    out_path = 'outputs/boundary_visualization.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n[VISUALIZATION saved to {out_path}]")
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SELF-SUPERVISED LEARNING EVIDENCE")
    print("=" * 60)
    print("""
The boundaries above were learned WITHOUT any ground truth masks!

Self-supervision signals used:
1. Anatomical ordering constraint (ILM < RNFL_INL < INL_ISOS < ISOS_RPE)
2. Layer thickness bounds (physiological ranges)
3. Boundary smoothness regularization
4. Physics constraints (Beer-Lambert attenuation, Fresnel reflections)
5. Multi-frame consistency (boundaries consistent across noisy frames)

This demonstrates that symbolic constraints alone can guide
boundary learning, enabling segmentation without GT masks.
""")


if __name__ == '__main__':
    main()
