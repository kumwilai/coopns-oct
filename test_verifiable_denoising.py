#!/usr/bin/env python3
"""
Test Verifiable Denoising on PKU37 with NAFNet.

This script validates our key hypothesis:
    "Symbolic predicate satisfaction correlates with actual denoising quality"

If this holds, we can verify denoising quality WITHOUT ground truth.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import sys
import os

# Add path for NAFNet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))

from symbolic_predicates import VerifiableDenoisingPredicate
from physics_enhanced_v3 import PhysicsEnsembleV3
from oct_symbolic_knowledge import SymbolicConstraints

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    pku37_root = Path("/home/kumwilai/OCT/pku37_oct_dataset")
    train_jsonl = pku37_root / "weights_pku37_analysis_val.jsonl"  # Use val for testing

    pairs = []
    with open(train_jsonl) as f:
        for line in f:
            data = json.loads(line)
            pairs.append({
                'clean': data['clean_path'],
                'noisy': data['noisy_path'],
            })

    print(f"Found {len(pairs)} validation pairs")

    # Load models
    print("\nLoading models...")

    # NAFNet for denoising
    if HAS_NAFNET:
        nafnet = NAFNet(
            img_channel=1,
            width=64,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2],
            dec_blk_nums=[2, 2, 2],
        ).to(device)

        # Load checkpoint
        ckpt_path = "/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            nafnet.load_state_dict(state, strict=False)
            psnr_ckpt = ckpt.get('psnr', 'unknown')
            print(f"Loaded NAFNet checkpoint: {ckpt_path} (PSNR: {psnr_ckpt})")
        else:
            print(f"Warning: NAFNet checkpoint not found at {ckpt_path}")

        nafnet.eval()
    else:
        print("NAFNet not available, using simple averaging")
        nafnet = None

    # Boundary model for anatomy predicate
    boundary_model = PhysicsEnsembleV3(
        in_channels=1,
        hidden_channels=48,
        num_boundaries=4,
    ).to(device)

    # Load trained boundary model
    boundary_ckpt = "/home/kumwilai/OCT/best_boundary_model_v4.pth"
    if os.path.exists(boundary_ckpt):
        ckpt = torch.load(boundary_ckpt, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        boundary_model.load_state_dict(state, strict=False)
        print(f"Loaded boundary model: {boundary_ckpt}")
    else:
        print("Warning: No trained boundary model, using random init")

    boundary_model.eval()

    # Hard constraint projection (guarantees valid anatomy)
    constraint_projector = SymbolicConstraints()

    # Verifiable denoising predicate
    verifier = VerifiableDenoisingPredicate().to(device)

    # Test on samples
    print("\nTesting verifiable denoising...")
    sample_size = min(20, len(pairs))  # Reduced for faster CPU testing
    sample_indices = np.random.choice(len(pairs), sample_size, replace=False)

    results = []

    for idx in tqdm(sample_indices, desc="Verifying"):
        pair = pairs[idx]

        clean_path = Path(pair['clean'])
        noisy_path = Path(pair['noisy'])

        if not clean_path.exists() or not noisy_path.exists():
            continue

        # Load images
        clean = load_image(clean_path).to(device)
        noisy = load_image(noisy_path).to(device)

        with torch.no_grad():
            # Denoise
            if nafnet is not None:
                denoised = nafnet(noisy)
                denoised = torch.clamp(denoised, 0, 1)
            else:
                denoised = F.avg_pool2d(noisy, 3, stride=1, padding=1)

            # Get boundaries
            boundary_out = boundary_model(denoised, return_aux=True)
            boundaries_raw = boundary_out['boundaries']

            # Apply hard constraint projection (guarantees valid anatomy)
            boundaries = constraint_projector.project_to_valid_space(boundaries_raw)

            # Compute PSNR (actual quality - requires GT)
            psnr = compute_psnr(denoised, clean)

            # Verify (without GT!)
            verification = verifier.verify(noisy, denoised, boundaries)

        results.append({
            'psnr': psnr,
            'satisfied': verification.satisfied,
            'score': verification.score,
            'speckle_ok': verification.speckle_satisfied,
            'anatomy_ok': verification.anatomy_satisfied,
            'structure_ok': verification.structure_satisfied,
            **verification.details,
        })

    # Analyze results
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    psnrs = [r['psnr'] for r in results]
    scores = [r['score'] for r in results]
    satisfied = [r['satisfied'] for r in results]

    print(f"\nPSNR Statistics (with GT):")
    print(f"  Mean: {np.mean(psnrs):.2f} dB")
    print(f"  Std:  {np.std(psnrs):.2f} dB")
    print(f"  Min:  {np.min(psnrs):.2f} dB")
    print(f"  Max:  {np.max(psnrs):.2f} dB")

    print(f"\nVerification Statistics (without GT):")
    print(f"  Satisfaction rate: {np.mean(satisfied):.1%}")
    print(f"  Mean score: {np.mean(scores):.3f}")

    # Per-predicate statistics
    speckle_ok = [r['speckle_ok'] for r in results]
    anatomy_ok = [r['anatomy_ok'] for r in results]
    structure_ok = [r['structure_ok'] for r in results]

    print(f"\nPer-Predicate Satisfaction:")
    print(f"  P1 (Speckle):   {np.mean(speckle_ok):.1%}")
    print(f"  P2 (Anatomy):   {np.mean(anatomy_ok):.1%}")
    print(f"  P3 (Structure): {np.mean(structure_ok):.1%}")

    # KEY: Correlation between score and PSNR
    correlation = np.corrcoef(scores, psnrs)[0, 1]
    print(f"\n*** CORRELATION (score vs PSNR): {correlation:.3f} ***")

    if correlation > 0.3:
        print("    -> Good correlation! Verification is meaningful.")
    elif correlation > 0.1:
        print("    -> Weak correlation. May need calibration.")
    else:
        print("    -> Poor correlation. Predicates need revision.")

    # Compare satisfied vs unsatisfied
    psnr_satisfied = [r['psnr'] for r in results if r['satisfied']]
    psnr_unsatisfied = [r['psnr'] for r in results if not r['satisfied']]

    if psnr_satisfied and psnr_unsatisfied:
        print(f"\nPSNR by Verification Status:")
        print(f"  Satisfied:   {np.mean(psnr_satisfied):.2f} dB (n={len(psnr_satisfied)})")
        print(f"  Unsatisfied: {np.mean(psnr_unsatisfied):.2f} dB (n={len(psnr_unsatisfied)})")
        diff = np.mean(psnr_satisfied) - np.mean(psnr_unsatisfied)
        print(f"  Difference:  {diff:+.2f} dB")

    # Detailed breakdown
    print("\n" + "-" * 70)
    print("DETAILED METRICS")
    print("-" * 70)

    cv_means = [r['speckle_cv'] for r in results]
    edge_corrs = [r['edge_correlation'] for r in results]
    edge_reds = [r['edge_reduction'] for r in results]

    print(f"  Speckle CV:      {np.mean(cv_means):.3f} ± {np.std(cv_means):.3f} (expected: 0.40)")
    print(f"  Edge correlation:{np.mean(edge_corrs):.3f} ± {np.std(edge_corrs):.3f} (threshold: 0.50)")
    print(f"  Edge reduction:  {np.mean(edge_reds):.3f} ± {np.std(edge_reds):.3f} (threshold: 0.30)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if correlation > 0.3:
        print("Symbolic predicates can meaningfully assess denoising quality")
        print("without ground truth. This validates our neuro-symbolic approach.")
    else:
        print("Predicates need further calibration or revision.")
        print("Consider adjusting thresholds based on the metrics above.")


if __name__ == "__main__":
    main()
