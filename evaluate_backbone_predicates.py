#!/usr/bin/env python3
"""
Evaluate Pretrained NAFNet Backbone Against GT-Free Predicates (P1-P6)

This script evaluates the pretrained NAFNet backbone output WITHOUT any correction
to establish the baseline quality and identify which predicates fail.

Usage:
    python evaluate_backbone_predicates.py

Output:
    - Per-sample predicate scores (P1-P6)
    - PASS/FAIL status for each predicate (threshold: 0.5)
    - PSNR/SSIM vs ground truth
    - Summary statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'nsnd_oct'))

import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Import NAFNetSmall from training code
from nsnd.models.nafnet import NAFNetSmall

# Import predicates from V8 corrector
from neuro_symbolic_corrector_v8 import EnhancedGTFreePredicates


def load_image(path):
    """Load image from TIF file and normalize to [0, 1]"""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1:
        arr = arr / 255.0
    return arr


def compute_psnr(pred, target):
    """Compute PSNR between two tensors"""
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.item()


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM between two images"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def main():
    print("=" * 90)
    print("EVALUATE PRETRAINED NAFNet BACKBONE AGAINST GT-FREE PREDICATES (P1-P6)")
    print("=" * 90)
    print()
    print("Purpose: Establish baseline quality WITHOUT neuro-symbolic correction")
    print("         Identify which predicates PASS (>=0.5) and which FAIL (<0.5)")
    print("=" * 90)
    print()

    # Configuration
    MODEL_PATH = "/home/kumwilai/OCT/outputs/nafnet_pku37_w40/best_model.pth"
    VAL_JSONL = "/home/kumwilai/OCT/pku37_oct_dataset/pku37_real_val.jsonl"
    NUM_SAMPLES = 10  # Use first 10 samples
    WIDTH = 40
    PASS_THRESHOLD = 0.5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Validation data: {VAL_JSONL}")
    print(f"Samples: {NUM_SAMPLES}")
    print()

    # Load model
    print("Loading pretrained NAFNet backbone...")
    model = NAFNetSmall(width=WIDTH).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # Handle both checkpoint formats (full checkpoint dict or just state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  - NAFNetSmall (width={WIDTH})")
    print(f"  - Parameters: {num_params:,}")
    print()

    # Initialize predicates
    print("Initializing GT-Free Predicates (EnhancedGTFreePredicates from V8)...")
    predicates = EnhancedGTFreePredicates().to(device)
    print("  - P1: Edge Quality (threshold: 0.7)")
    print("  - P2: Contrast Quality (threshold: 0.6)")
    print("  - P3: Smoothness Quality (threshold: 0.8)")
    print("  - P4: Structure Quality (threshold: 0.7)")
    print("  - P5: Speckle Fidelity (Physics-accurate, threshold: 0.7)")
    print("  - P6: Anatomy Validity (threshold: 0.75)")
    print()
    print("Note: For this evaluation, PASS/FAIL uses threshold 0.5")
    print()

    # Load validation data
    print("Loading validation data...")
    val_pairs = []
    with open(VAL_JSONL, 'r') as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES:
                break
            entry = json.loads(line.strip())
            val_pairs.append((entry['clean_path'], entry['noisy_path']))
    print(f"  - Loaded {len(val_pairs)} samples")
    print()

    # Store results
    results = []
    all_scores = {f'P{i}': [] for i in range(1, 7)}
    psnr_values = []
    ssim_values = []

    # Evaluate each sample
    print("=" * 90)
    print("EVALUATING SAMPLES")
    print("=" * 90)
    print()

    for idx, (clean_path, noisy_path) in enumerate(val_pairs):
        # Load images
        clean = load_image(clean_path)
        noisy = load_image(noisy_path)

        # Convert to tensors
        clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float().to(device)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

        # Pad to multiple of 16 for NAFNet
        _, _, H, W = noisy_t.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            noisy_padded = F.pad(noisy_t, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            noisy_padded = noisy_t

        # Run backbone inference
        with torch.no_grad():
            denoised = model(noisy_padded)

            # Unpad
            if pad_h > 0 or pad_w > 0:
                denoised = denoised[:, :, :H, :W]

            # Clamp to valid range
            denoised = denoised.clamp(0, 1)

            # Compute PSNR/SSIM vs ground truth
            psnr = compute_psnr(denoised, clean_t)
            ssim = compute_ssim(denoised, clean_t)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            # Evaluate predicates (denoised vs noisy)
            pred_results = predicates(denoised, noisy_t)

        # Extract scores
        scores = pred_results['scores']

        # Store for averages
        for key in all_scores:
            all_scores[key].append(scores[f'{key}_edge' if key == 'P1' else
                                         f'{key}_contrast' if key == 'P2' else
                                         f'{key}_smooth' if key == 'P3' else
                                         f'{key}_structure' if key == 'P4' else
                                         f'{key}_speckle' if key == 'P5' else
                                         f'{key}_anatomy'])

        # Determine PASS/FAIL
        sample_result = {
            'sample_idx': idx,
            'noisy_path': Path(noisy_path).name,
            'psnr': psnr,
            'ssim': ssim,
            'scores': {},
            'pass_fail': {}
        }

        for key, score_key in [('P1', 'P1_edge'), ('P2', 'P2_contrast'), ('P3', 'P3_smooth'),
                                ('P4', 'P4_structure'), ('P5', 'P5_speckle'), ('P6', 'P6_anatomy')]:
            score = scores[score_key]
            passed = score >= PASS_THRESHOLD
            sample_result['scores'][key] = score
            sample_result['pass_fail'][key] = 'PASS' if passed else 'FAIL'

        results.append(sample_result)

        # Print sample result
        print(f"Sample {idx+1}/{len(val_pairs)}: {Path(noisy_path).name}")
        print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
        print("  Predicates:")
        for key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            score = sample_result['scores'][key]
            status = sample_result['pass_fail'][key]
            status_str = f"[{status}]" if status == 'PASS' else f"[{status}]"
            marker = "+" if status == 'PASS' else "X"
            print(f"    {key}: {score:.4f} {marker} {status_str}")
        print()

    # Summary statistics
    print("=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    print()

    # PSNR/SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Ground Truth Metrics (avg over {len(val_pairs)} samples):")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print()

    # Predicate summary
    print("Predicate Scores (avg over all samples):")
    print("-" * 50)

    predicate_names = {
        'P1': 'Edge Quality',
        'P2': 'Contrast Quality',
        'P3': 'Smoothness Quality',
        'P4': 'Structure Quality',
        'P5': 'Speckle Fidelity',
        'P6': 'Anatomy Validity'
    }

    score_keys = {
        'P1': 'P1_edge',
        'P2': 'P2_contrast',
        'P3': 'P3_smooth',
        'P4': 'P4_structure',
        'P5': 'P5_speckle',
        'P6': 'P6_anatomy'
    }

    total_pass = 0
    total_fail = 0

    for key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        # Collect scores across all samples
        scores_list = [r['scores'][key] for r in results]
        avg_score = np.mean(scores_list)

        # Count passes
        pass_count = sum(1 for r in results if r['pass_fail'][key] == 'PASS')
        fail_count = len(results) - pass_count

        total_pass += pass_count
        total_fail += fail_count

        avg_status = 'PASS' if avg_score >= PASS_THRESHOLD else 'FAIL'
        marker = "+" if avg_status == 'PASS' else "X"

        print(f"  {key} ({predicate_names[key]:20s}): {avg_score:.4f} {marker} [{avg_status}]")
        print(f"       Pass rate: {pass_count}/{len(results)} ({100*pass_count/len(results):.1f}%)")

    print("-" * 50)
    overall_avg = np.mean([np.mean([r['scores'][k] for r in results]) for k in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']])
    overall_pass_rate = total_pass / (total_pass + total_fail) * 100
    print(f"  Overall Average Score: {overall_avg:.4f}")
    print(f"  Overall Pass Rate: {total_pass}/{total_pass+total_fail} ({overall_pass_rate:.1f}%)")
    print()

    # Identify weak predicates
    print("=" * 90)
    print("ANALYSIS: WHERE CORRECTION IS NEEDED")
    print("=" * 90)
    print()

    weak_predicates = []
    strong_predicates = []

    for key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        scores_list = [r['scores'][key] for r in results]
        avg_score = np.mean(scores_list)
        pass_rate = sum(1 for r in results if r['pass_fail'][key] == 'PASS') / len(results)

        if avg_score < PASS_THRESHOLD or pass_rate < 0.7:
            weak_predicates.append((key, predicate_names[key], avg_score, pass_rate))
        else:
            strong_predicates.append((key, predicate_names[key], avg_score, pass_rate))

    if weak_predicates:
        print("WEAK PREDICATES (Correction NEEDED):")
        for key, name, score, rate in weak_predicates:
            print(f"  X {key} ({name}): avg={score:.4f}, pass_rate={rate*100:.1f}%")
    else:
        print("No weak predicates found (all >=0.5 with >70% pass rate)")

    print()

    if strong_predicates:
        print("STRONG PREDICATES (Correction optional):")
        for key, name, score, rate in strong_predicates:
            print(f"  + {key} ({name}): avg={score:.4f}, pass_rate={rate*100:.1f}%")

    print()
    print("=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    print()
    print(f"The pretrained NAFNet backbone (width={WIDTH}) achieves:")
    print(f"  - PSNR: {avg_psnr:.2f} dB")
    print(f"  - SSIM: {avg_ssim:.4f}")
    print(f"  - Overall predicate score: {overall_avg:.4f}")
    print(f"  - Overall pass rate: {overall_pass_rate:.1f}%")
    print()

    if weak_predicates:
        print("The neuro-symbolic corrector should focus on improving:")
        for key, name, score, rate in weak_predicates:
            print(f"  - {key} ({name})")
    else:
        print("All predicates pass - corrector may provide marginal improvements.")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
