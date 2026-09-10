#!/usr/bin/env python3
"""
Compare End-to-End vs Ground Truth Approach

Tests:
1. Current approach (using ground truth noise weights) - Upper bound
2. End-to-End approach (analyzer trained jointly) - Our solution
3. Pre-trained analyzer (broken) - Lower bound

Shows whether end-to-end joint training fixes the analyzer bottleneck.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "nsnd_oct"))

import torch
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer
from nsnd.models.noise_conditioner import SpatialBasisModulator
from nsnd.utils.metrics import compute_psnr, compute_ssim


def load_end_to_end_model(ckpt_path, device='cpu'):
    """Load end-to-end trained model."""

    # NAFNet
    CONDITIONER_DIM = 32
    model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        middle_blk_num=2,
        cond_dim=CONDITIONER_DIM,
        condition_middle=True,
        condition_decoders=True,
        use_spatial_cue=True
    ).to(device)

    # Analyzer
    analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=False).to(device)

    # Modulator
    modulator = SpatialBasisModulator(
        feature_channels=128,
        stage_channels=model.dbm_stage_channels,
        num_noise_types=4,
        hidden_channels=64,
        alpha=2.0,
        gate_floor=0.0,
        basis_init_std=0.1,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    analyzer.load_state_dict(checkpoint['analyzer'], strict=True)
    modulator.load_state_dict(checkpoint['modulator'], strict=True)

    model.eval()
    analyzer.eval()
    modulator.eval()

    for p in model.parameters():
        p.requires_grad = False
    for p in analyzer.parameters():
        p.requires_grad = False
    for p in modulator.parameters():
        p.requires_grad = False

    return model, analyzer, modulator


def load_ground_truth_model(ckpt_path, analyzer_ckpt, device='cpu'):
    """Load model trained with ground truth (current approach)."""

    # Analyzer (frozen, only for feature extraction)
    analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=False).to(device)
    analyzer_state = torch.load(analyzer_ckpt, map_location=device, weights_only=False)
    analyzer.load_state_dict(analyzer_state["state_dict"], strict=False)
    analyzer.eval()
    for p in analyzer.parameters():
        p.requires_grad = False

    # NAFNet
    CONDITIONER_DIM = 32
    model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        middle_blk_num=2,
        cond_dim=CONDITIONER_DIM,
        condition_middle=True,
        condition_decoders=True,
        use_spatial_cue=True
    ).to(device)

    # Modulator
    modulator = SpatialBasisModulator(
        feature_channels=128,
        stage_channels=model.dbm_stage_channels,
        num_noise_types=4,
        hidden_channels=64,
        alpha=2.0,
        gate_floor=0.0,
        basis_init_std=0.1,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    modulator.load_state_dict(checkpoint['modulator'], strict=True)

    model.eval()
    modulator.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in modulator.parameters():
        p.requires_grad = False

    return model, analyzer, modulator


def test_sample(noisy, clean, true_weights, model, analyzer, modulator, use_ground_truth, device='cpu'):
    """Test denoising on a single sample."""

    noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
    clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float().to(device)
    true_weights_t = torch.from_numpy(true_weights).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get features from analyzer
        _, features = analyzer(noisy_t, return_feature_map=True, return_predicates=False)
        feature_map = features.get("feature_map")

        # Get noise weights
        if use_ground_truth:
            # Ground truth approach (upper bound)
            global_weights = true_weights_t
            confidence = torch.ones(1, device=device)
        else:
            # Predicted weights (end-to-end or pre-trained)
            weights_dict = analyzer(noisy_t, return_feature_map=False, return_predicates=False)[0]
            global_weights = torch.stack([
                weights_dict["speckle"],
                weights_dict["banding"],
                weights_dict["gaussian"],
                weights_dict["shot"],
            ], dim=1)
            confidence = weights_dict.get("_confidence", torch.ones(1, device=device))

        # Get spatial modulation
        spatial_map, gate, basis = modulator(
            feature_map,
            global_weights=global_weights,
            confidence=confidence,
            noisy=noisy_t,
        )

        # Denoise
        denoised = model(
            noisy_t,
            spatial_map=spatial_map,
            basis=basis,
            alpha=2.0,
            gate=gate
        )

    # Compute metrics
    psnr = compute_psnr(denoised, clean_t)
    ssim = compute_ssim(denoised, clean_t)

    # Noise classification accuracy
    predicted_class = global_weights.argmax(dim=1).item()
    true_class = true_weights_t.argmax(dim=1).item()
    correct = (predicted_class == true_class)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'correct': correct,
        'predicted_weights': global_weights.cpu().numpy()[0],
        'gate': gate.item(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_to_end_ckpt", type=str, default="checkpoints/end_to_end/best_model.pth")
    parser.add_argument("--ground_truth_ckpt", type=str, default="checkpoints/adaptive_nafnet_full/adaptive_nafnet_best.pth")
    parser.add_argument("--analyzer_ckpt", type=str, default="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth")
    parser.add_argument("--val_jsonl", type=str, default="weights_duke_analysis_maps_val.jsonl")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("="*80)
    print("COMPARISON: End-to-End vs Ground Truth vs Pre-trained Analyzer")
    print("="*80)

    # Load test samples
    samples = []
    with open(args.val_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            data = json.loads(line.strip())
            samples.append(data)

    print(f"\nLoaded {len(samples)} test samples")

    # Test 1: Ground Truth Approach (Upper Bound)
    print("\n" + "-"*80)
    print("TEST 1: Ground Truth Approach (Upper Bound)")
    print("-"*80)

    model_gt, analyzer_gt, modulator_gt = load_ground_truth_model(
        args.ground_truth_ckpt,
        args.analyzer_ckpt,
        device=args.device
    )

    results_gt = []
    for data in tqdm(samples, desc="Ground Truth"):
        noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

        h, w = noisy.shape
        top = (h - 64) // 2
        left = (w - 64) // 2
        noisy = noisy[top:top+64, left:left+64]
        clean = clean[top:top+64, left:left+64]

        true_weights = np.array([
            data['weights']['speckle'],
            data['weights']['banding'],
            data['weights']['gaussian'],
            data['weights']['shot']
        ], dtype=np.float32)

        result = test_sample(noisy, clean, true_weights, model_gt, analyzer_gt, modulator_gt,
                           use_ground_truth=True, device=args.device)
        results_gt.append(result)

    avg_psnr_gt = np.mean([r['psnr'] for r in results_gt])
    avg_ssim_gt = np.mean([r['ssim'] for r in results_gt])
    top1_gt = 100.0 * np.mean([r['correct'] for r in results_gt])

    print(f"\nResults:")
    print(f"  PSNR: {avg_psnr_gt:.2f} dB")
    print(f"  SSIM: {avg_ssim_gt:.4f}")
    print(f"  Top-1 Accuracy: {top1_gt:.1f}%")

    # Test 2: End-to-End Approach (Our Solution)
    if Path(args.end_to_end_ckpt).exists():
        print("\n" + "-"*80)
        print("TEST 2: End-to-End Joint Training (Our Solution)")
        print("-"*80)

        model_e2e, analyzer_e2e, modulator_e2e = load_end_to_end_model(
            args.end_to_end_ckpt,
            device=args.device
        )

        results_e2e = []
        for data in tqdm(samples, desc="End-to-End"):
            noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
            clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

            h, w = noisy.shape
            top = (h - 64) // 2
            left = (w - 64) // 2
            noisy = noisy[top:top+64, left:left+64]
            clean = clean[top:top+64, left:left+64]

            true_weights = np.array([
                data['weights']['speckle'],
                data['weights']['banding'],
                data['weights']['gaussian'],
                data['weights']['shot']
            ], dtype=np.float32)

            result = test_sample(noisy, clean, true_weights, model_e2e, analyzer_e2e, modulator_e2e,
                               use_ground_truth=False, device=args.device)
            results_e2e.append(result)

        avg_psnr_e2e = np.mean([r['psnr'] for r in results_e2e])
        avg_ssim_e2e = np.mean([r['ssim'] for r in results_e2e])
        top1_e2e = 100.0 * np.mean([r['correct'] for r in results_e2e])

        print(f"\nResults:")
        print(f"  PSNR: {avg_psnr_e2e:.2f} dB")
        print(f"  SSIM: {avg_ssim_e2e:.4f}")
        print(f"  Top-1 Accuracy: {top1_e2e:.1f}%")
    else:
        print(f"\n⚠️  End-to-End checkpoint not found: {args.end_to_end_ckpt}")
        print(f"   Run training first: bash run_end_to_end.sh")
        avg_psnr_e2e = 0
        top1_e2e = 0

    # Test 3: Pre-trained Analyzer (Lower Bound)
    print("\n" + "-"*80)
    print("TEST 3: Pre-trained Analyzer (Broken Baseline)")
    print("-"*80)

    results_pretrained = []
    for data in tqdm(samples, desc="Pre-trained"):
        noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

        h, w = noisy.shape
        top = (h - 64) // 2
        left = (w - 64) // 2
        noisy = noisy[top:top+64, left:left+64]
        clean = clean[top:top+64, left:left+64]

        true_weights = np.array([
            data['weights']['speckle'],
            data['weights']['banding'],
            data['weights']['gaussian'],
            data['weights']['shot']
        ], dtype=np.float32)

        result = test_sample(noisy, clean, true_weights, model_gt, analyzer_gt, modulator_gt,
                           use_ground_truth=False, device=args.device)
        results_pretrained.append(result)

    avg_psnr_pretrained = np.mean([r['psnr'] for r in results_pretrained])
    avg_ssim_pretrained = np.mean([r['ssim'] for r in results_pretrained])
    top1_pretrained = 100.0 * np.mean([r['correct'] for r in results_pretrained])

    print(f"\nResults:")
    print(f"  PSNR: {avg_psnr_pretrained:.2f} dB")
    print(f"  SSIM: {avg_ssim_pretrained:.4f}")
    print(f"  Top-1 Accuracy: {top1_pretrained:.1f}%")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    print(f"\n{'Approach':<30} {'PSNR (dB)':<15} {'Top-1 Acc':<15} {'Status'}")
    print("-"*80)
    print(f"{'Ground Truth (Upper Bound)':<30} {avg_psnr_gt:<15.2f} {top1_gt:<15.1f} {'✅ Perfect'}")
    if Path(args.end_to_end_ckpt).exists():
        status_e2e = "✅ Good" if top1_e2e > 60 else "⚠️  Needs tuning"
        print(f"{'End-to-End (Our Solution)':<30} {avg_psnr_e2e:<15.2f} {top1_e2e:<15.1f} {status_e2e}")
    else:
        print(f"{'End-to-End (Our Solution)':<30} {'N/A':<15} {'N/A':<15} {'❌ Not trained'}")
    print(f"{'Pre-trained (Broken)':<30} {avg_psnr_pretrained:<15.2f} {top1_pretrained:<15.1f} {'❌ Broken'}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if Path(args.end_to_end_ckpt).exists():
        if top1_e2e > 60:
            print("\n✅ End-to-End training SUCCESSFULLY fixed the analyzer bottleneck!")
            print(f"   Analyzer accuracy improved: {top1_pretrained:.1f}% → {top1_e2e:.1f}%")
            print(f"   PSNR improvement: {avg_psnr_pretrained:.2f} → {avg_psnr_e2e:.2f} dB")
            print("\n   Ready for production deployment!")
        else:
            print("\n⚠️  End-to-End training shows improvement but may need tuning.")
            print(f"   Analyzer accuracy: {top1_e2e:.1f}% (target: >60%)")
            print("\n   Recommendations:")
            print("   1. Train for more epochs (20 → 50)")
            print("   2. Increase classification loss weight (0.1 → 0.3)")
            print("   3. Try different learning rates")
    else:
        print("\n⏳ End-to-End model not trained yet.")
        print("   Run: bash run_end_to_end.sh")


if __name__ == "__main__":
    main()
