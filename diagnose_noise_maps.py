#!/usr/bin/env python3
"""
Diagnostic script to analyze noise map quality and interpretability.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "nsnd_oct"))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer
from nsnd.models.noise_conditioner import SpatialBasisModulator
from nsnd.models.nafnet import NAFNetFullFiLM

def load_analyzer(ckpt_path, device='cuda'):
    analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    analyzer.load_state_dict(ckpt["state_dict"], strict=False)
    analyzer.eval()
    for p in analyzer.parameters():
        p.requires_grad = False
    return analyzer

def analyze_sample(noisy_path, clean_path, analyzer_ckpt, device='cuda'):
    """Analyze a single sample to see what the analyzer predicts."""

    # Load images
    noisy = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
    clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

    # Center crop to 64x64
    h, w = noisy.shape
    top = (h - 64) // 2
    left = (w - 64) // 2
    noisy = noisy[top:top+64, left:left+64]
    clean = clean[top:top+64, left:left+64]

    noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

    # Load analyzer
    analyzer = load_analyzer(analyzer_ckpt, device)

    # Get predictions
    with torch.no_grad():
        weights_dict, features = analyzer(noisy_t, return_feature_map=True, return_predicates=True)

    # Extract results
    speckle = weights_dict["speckle"].item()
    banding = weights_dict["banding"].item()
    gaussian = weights_dict["gaussian"].item()
    shot = weights_dict["shot"].item()
    confidence = weights_dict.get("_confidence", torch.tensor([1.0])).item()

    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(noisy_path).name}")
    print(f"{'='*60}")
    print(f"\nNoise Type Predictions:")
    print(f"  Speckle:  {speckle:.4f}")
    print(f"  Banding:  {banding:.4f}")
    print(f"  Gaussian: {gaussian:.4f}")
    print(f"  Shot:     {shot:.4f}")
    print(f"  Confidence: {confidence:.4f}")

    # Compute entropy
    weights = np.array([speckle, banding, gaussian, shot])
    entropy = -(weights * np.log(weights + 1e-8)).sum()
    max_prob = weights.max()

    print(f"\nStatistics:")
    print(f"  Entropy: {entropy:.4f} (max={np.log(4):.4f}, lower is better)")
    print(f"  Max Prob: {max_prob:.4f} (higher is better, >0.5 is confident)")
    print(f"  Predicted type: {['Speckle', 'Banding', 'Gaussian', 'Shot'][weights.argmax()]}")

    # Check if prediction makes sense
    if max_prob < 0.35:
        print(f"\n⚠️  WARNING: Low confidence prediction (max={max_prob:.3f})")
        print(f"    Model is uncertain about noise type")

    if entropy > 1.2:
        print(f"\n⚠️  WARNING: High entropy (H={entropy:.3f})")
        print(f"    Model is almost randomly guessing")

    return weights_dict, features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy", type=str, required=True)
    parser.add_argument("--clean", type=str, required=True)
    parser.add_argument("--analyzer_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    analyze_sample(args.noisy, args.clean, args.analyzer_ckpt, args.device)
