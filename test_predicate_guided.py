#!/usr/bin/env python3
"""
Test Predicate-Guided Denoising on PKU37 with NAFNet backbone.

This demonstrates the full closed-loop system:
1. NAFNet backbone for initial denoising
2. Trained boundary model for anatomy
3. Iterative refinement guided by predicate failure maps
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys
import os

sys.path.insert(0, 'nsnd_oct')

from predicate_guided_denoising import PredicateGuidedDenoiser
from physics_enhanced_v3 import PhysicsEnsembleV3

# Try to import NAFNet
try:
    from nsnd.models.nafnet import NAFNetFullFiLM as NAFNet
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False
    print("Warning: NAFNet not available")


def load_image(path):
    """Load image and normalize to [0, 1]."""
    img = Image.open(path).convert('L')
    img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def compute_psnr(pred, target):
    """Compute PSNR in dB."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def main():
    device = torch.device('cpu')
    print("=" * 70)
    print("PREDICATE-GUIDED DENOISING ON PKU37")
    print("=" * 70)

    # Load NAFNet backbone
    print("\nLoading models...")

    if HAS_NAFNET:
        backbone = NAFNet(
            img_channel=1,
            width=64,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2],
            dec_blk_nums=[2, 2, 2],
        )
        ckpt_path = "/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            backbone.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"Loaded NAFNet: PSNR={ckpt.get('psnr', 'unknown')}")
    else:
        backbone = None
        print("Using simple backbone")

    # Load boundary model
    boundary_model = PhysicsEnsembleV3(
        in_channels=1,
        hidden_channels=48,
        num_boundaries=4,
    )
    boundary_ckpt = "/home/kumwilai/OCT/best_boundary_model_v4.pth"
    if os.path.exists(boundary_ckpt):
        ckpt = torch.load(boundary_ckpt, map_location=device, weights_only=False)
        boundary_model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        print(f"Loaded boundary model")

    # Create predicate-guided denoiser
    model = PredicateGuidedDenoiser(
        backbone=backbone,
        boundary_model=boundary_model,
        max_iterations=3,
        speckle_cv=0.40,
        speckle_tolerance=0.14,
    ).to(device)
    model.eval()

    # Load test samples
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    val_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    pairs = []
    with open(val_jsonl) as f:
        for line in f:
            data = json.loads(line)
            pairs.append({
                'clean': data['clean_path'],
                'noisy': data['noisy_path'],
            })

    print(f"\nTesting on {min(5, len(pairs))} samples...")
    print("-" * 70)

    results = []
    for i in range(min(5, len(pairs))):
        pair = pairs[i]

        clean = load_image(pair['clean']).to(device)
        noisy = load_image(pair['noisy']).to(device)

        with torch.no_grad():
            # Standard NAFNet (no refinement)
            if backbone is not None:
                nafnet_out = backbone(noisy)
                nafnet_out = torch.clamp(nafnet_out, 0, 1)
                psnr_nafnet = compute_psnr(nafnet_out, clean)
            else:
                psnr_nafnet = 0

            # Predicate-guided denoising
            outputs = model(noisy, return_intermediates=True)
            psnr_guided = compute_psnr(outputs['denoised'], clean)

        print(f"\nSample {i+1}:")
        print(f"  NAFNet PSNR:        {psnr_nafnet:.2f} dB")
        print(f"  Guided PSNR:        {psnr_guided:.2f} dB")
        print(f"  Improvement:        {psnr_guided - psnr_nafnet:+.2f} dB")
        print(f"  Iterations used:    {outputs['iterations']}")
        print(f"  Predicates satisfied: {outputs['all_satisfied']}")

        # Show predicate status
        for name, sat in outputs['satisfied'].items():
            status = '✓' if sat else '✗'
            loss = outputs['losses'].get(name, 0)
            print(f"    {name}: {status} (loss={loss:.4f})")

        results.append({
            'psnr_nafnet': psnr_nafnet,
            'psnr_guided': psnr_guided,
            'improvement': psnr_guided - psnr_nafnet,
            'iterations': outputs['iterations'],
            'all_satisfied': outputs['all_satisfied'],
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    psnr_nafnet_avg = np.mean([r['psnr_nafnet'] for r in results])
    psnr_guided_avg = np.mean([r['psnr_guided'] for r in results])
    improvement_avg = np.mean([r['improvement'] for r in results])
    satisfaction_rate = np.mean([r['all_satisfied'] for r in results])

    print(f"\nAverage NAFNet PSNR:     {psnr_nafnet_avg:.2f} dB")
    print(f"Average Guided PSNR:     {psnr_guided_avg:.2f} dB")
    print(f"Average Improvement:     {improvement_avg:+.2f} dB")
    print(f"Predicate satisfaction:  {satisfaction_rate:.1%}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The predicate-guided system provides:

1. QUALITY VERIFICATION: Know if denoising succeeded without GT
2. FAILURE LOCALIZATION: See WHERE predicates fail (spatial maps)
3. TARGETED REFINEMENT: Apply corrections only where needed
4. CONVERGENCE GUARANTEE: Stop when predicates are satisfied

Current status:
- If improvement > 0: refinement is helping
- If improvement < 0: refinement network needs more training
- Either way: we KNOW the quality via predicates (no GT needed!)
""")


if __name__ == "__main__":
    main()
