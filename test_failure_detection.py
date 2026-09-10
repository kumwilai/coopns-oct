#!/usr/bin/env python3
"""
Test Failure Detection: Can predicates detect BAD denoising?

The key value of verifiable denoising is detecting failures when:
1. Denoiser is applied to out-of-distribution data
2. Denoiser over-smooths (removes structure)
3. Denoiser under-denoises (leaves noise)

This test simulates failure modes and checks if predicates catch them.
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

from symbolic_predicates import VerifiableDenoisingPredicate
from physics_enhanced_v3 import PhysicsEnsembleV3
from oct_symbolic_knowledge import SymbolicConstraints


def load_image(path):
    """Load image and normalize to [0, 1]."""
    img = Image.open(path).convert('L')
    img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def main():
    device = torch.device('cpu')
    print("Testing Failure Detection with Symbolic Predicates")
    print("=" * 70)

    # Load sample image
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    train_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"

    with open(train_jsonl) as f:
        data = json.loads(f.readline())

    clean = load_image(data['clean_path'])
    noisy = load_image(data['noisy_path'])

    print(f"Image shape: {noisy.shape}")

    # Initialize models
    boundary_model = PhysicsEnsembleV3(in_channels=1, hidden_channels=48, num_boundaries=4)
    boundary_ckpt = "/home/kumwilai/OCT/best_boundary_model_v4.pth"
    if os.path.exists(boundary_ckpt):
        ckpt = torch.load(boundary_ckpt, map_location=device, weights_only=False)
        boundary_model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    boundary_model.eval()

    constraint_projector = SymbolicConstraints()
    verifier = VerifiableDenoisingPredicate()

    def get_boundaries(img):
        with torch.no_grad():
            out = boundary_model(img, return_aux=True)
            return constraint_projector.project_to_valid_space(out['boundaries'])

    def verify_and_report(name, noisy_img, denoised_img):
        boundaries = get_boundaries(denoised_img)
        result = verifier.verify(noisy_img, denoised_img, boundaries)
        psnr = 10 * torch.log10(1.0 / F.mse_loss(denoised_img, clean)).item()

        print(f"\n{name}:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Satisfied: {result.satisfied}")
        print(f"  Score: {result.score:.3f}")
        print(f"  P1 (Speckle): {result.speckle_satisfied} (CV={result.details['speckle_cv']:.3f})")
        print(f"  P2 (Anatomy): {result.anatomy_satisfied}")
        print(f"  P3 (Structure): {result.structure_satisfied} (corr={result.details['edge_correlation']:.3f})")
        return result

    # Test 1: Good denoising (simple averaging)
    print("\n" + "-" * 70)
    print("TEST 1: GOOD DENOISING (mild smoothing)")
    print("-" * 70)
    good_denoised = F.avg_pool2d(noisy, 3, stride=1, padding=1)
    verify_and_report("Mild smoothing (3x3)", noisy, good_denoised)

    # Test 2: Over-smoothing (aggressive blur)
    print("\n" + "-" * 70)
    print("TEST 2: OVER-SMOOTHING (should fail P1 or P3)")
    print("-" * 70)
    over_smoothed = F.avg_pool2d(noisy, 15, stride=1, padding=7)
    verify_and_report("Aggressive smoothing (15x15)", noisy, over_smoothed)

    over_smoothed2 = F.avg_pool2d(noisy, 31, stride=1, padding=15)
    verify_and_report("Extreme smoothing (31x31)", noisy, over_smoothed2)

    # Test 3: Under-denoising (too little processing)
    print("\n" + "-" * 70)
    print("TEST 3: UNDER-DENOISING (should fail P1)")
    print("-" * 70)
    under_denoised = 0.9 * noisy + 0.1 * F.avg_pool2d(noisy, 3, stride=1, padding=1)
    verify_and_report("Minimal denoising (90% noisy)", noisy, under_denoised)

    # Test 4: No denoising at all
    print("\n" + "-" * 70)
    print("TEST 4: NO DENOISING (identity - should fail)")
    print("-" * 70)
    verify_and_report("Identity (no denoising)", noisy, noisy)

    # Test 5: Wrong output (structure destroyed)
    print("\n" + "-" * 70)
    print("TEST 5: STRUCTURE DESTROYED (should fail P3)")
    print("-" * 70)
    # Add random noise to clean
    destroyed = clean + torch.randn_like(clean) * 0.1
    destroyed = destroyed.clamp(0, 1)
    verify_and_report("Clean + random noise", noisy, destroyed)

    # Shuffle pixels (extreme structure destruction)
    B, C, H, W = clean.shape
    shuffled = clean.view(B, C, -1)
    shuffled = shuffled[:, :, torch.randperm(H * W)]
    shuffled = shuffled.view(B, C, H, W)
    verify_and_report("Pixel shuffle (total destruction)", noisy, shuffled)

    # Test 6: Out-of-distribution (inverted image)
    print("\n" + "-" * 70)
    print("TEST 6: OUT-OF-DISTRIBUTION (inverted)")
    print("-" * 70)
    inverted = 1.0 - good_denoised
    verify_and_report("Inverted image", noisy, inverted)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Expected behavior:
- Good denoising: All predicates PASS
- Over-smoothing: P1 (speckle) or P3 (structure) FAIL
- Under-denoising: P1 (speckle) FAIL (CV too high)
- No denoising: P1 FAIL
- Structure destroyed: P3 FAIL
- Inverted: P2 (anatomy) or P3 FAIL

The predicates should detect failure modes without needing ground truth!
""")


if __name__ == "__main__":
    main()
