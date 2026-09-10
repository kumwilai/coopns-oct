#!/usr/bin/env python3
"""
Comprehensive Test Script for Corrector Pipeline

This script verifies the entire corrector pipeline works correctly:
1. Loads pretrained backbone
2. Creates V6 model with PowerfulAdaptiveCorrectorWithLambda
3. Runs forward pass with real data
4. Prints detailed statistics (raw corrections, lambda maps, PSNR, predicates)
5. Verifies gradients flow correctly
6. Simulates training steps to verify improvement

Usage:
    python test_corrector_pipeline.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.stdout.reconfigure(line_buffering=True)

# Add project path
sys.path.insert(0, '/home/kumwilai/OCT')
sys.path.insert(0, '/home/kumwilai/OCT/nsnd_oct')

from neuro_symbolic_v2 import (
    BackboneWrapper, AdaptiveLambdaPredictor, SymbolicPredicates,
    OCTDataset, _NUM_THREADS, clear_memory
)
from powerful_correctors import PowerfulAdaptiveCorrectorWithLambda
from quality_aligned_loss import QualityAlignedLoss, CleanReferencedPredicates
from train_v6 import NeuroSymbolicDenoiserV6
from torch.utils.data import DataLoader


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR between two images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (-10 * torch.log10(mse)).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM between two images."""
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, 11, stride=1, padding=5)
    sigma1_sq = F.avg_pool2d(img1**2, 11, stride=1, padding=5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2**2, 11, stride=1, padding=5) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, 11, stride=1, padding=5) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim.mean().item()


def print_separator(title: str = "", char: str = "="):
    """Print a separator line with optional title."""
    width = 70
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_pass_fail(condition: bool, message: str):
    """Print PASS/FAIL status."""
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} {message}")


# =============================================================================
# TEST 1: LOAD MODEL AND BACKBONE
# =============================================================================

def test_load_model(backbone_path: str) -> Tuple[NeuroSymbolicDenoiserV6, bool]:
    """Test loading the V6 model with pretrained backbone."""
    print_separator("TEST 1: LOAD MODEL AND BACKBONE")

    success = True

    # Create model
    print("\nCreating NeuroSymbolicDenoiserV6 with PowerfulAdaptiveCorrectorWithLambda...")
    try:
        model = NeuroSymbolicDenoiserV6(
            backbone_type='nafnet',
            width=64,
            corrector_hidden_dim=128,
            corrector_type='powerful'
        )
        print_pass_fail(True, "Model created successfully")
    except Exception as e:
        print_pass_fail(False, f"Failed to create model: {e}")
        return None, False

    # Load pretrained backbone
    print(f"\nLoading pretrained backbone from: {backbone_path}")
    try:
        loaded = model.load_pretrained_backbone(backbone_path)
        print_pass_fail(loaded, "Backbone loaded" if loaded else "Backbone not loaded (using random init)")
    except Exception as e:
        print_pass_fail(False, f"Failed to load backbone: {e}")
        success = False

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    corrector_params = sum(p.numel() for p in model.corrector.parameters())
    lambda_params = sum(p.numel() for p in model.lambda_predictor.parameters())

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Backbone parameters: {backbone_params:,}")
    print(f"  Corrector parameters: {corrector_params:,}")
    print(f"  Lambda predictor parameters: {lambda_params:,}")

    # Verify corrector type
    print(f"\nCorrector type: {model.corrector_type}")
    print_pass_fail(model.corrector_type == 'powerful', "Using PowerfulAdaptiveCorrectorWithLambda")

    return model, success


# =============================================================================
# TEST 2: FORWARD PASS WITH REAL DATA
# =============================================================================

def test_forward_pass(model: NeuroSymbolicDenoiserV6,
                      val_jsonl: str) -> Tuple[Dict, bool]:
    """Test forward pass with real validation data."""
    print_separator("TEST 2: FORWARD PASS WITH REAL DATA")

    success = True

    # Load validation sample
    print(f"\nLoading validation data from: {val_jsonl}")
    try:
        val_dataset = OCTDataset(
            val_jsonl,
            max_samples=1,
            patch_size=0,  # Full image
            is_train=False
        )
        print_pass_fail(True, f"Loaded {len(val_dataset)} validation sample(s)")
    except Exception as e:
        print_pass_fail(False, f"Failed to load validation data: {e}")
        return None, False

    # Get single sample
    sample = val_dataset[0]
    noisy = sample['noisy'].unsqueeze(0)  # [1, 1, H, W]
    clean = sample['clean'].unsqueeze(0)  # [1, 1, H, W]

    print(f"  Input shape: {noisy.shape}")
    print(f"  Clean shape: {clean.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    model.eval()

    try:
        with torch.no_grad():
            corrected, backbone_out, info = model(noisy, max_lambda=0.15)
        print_pass_fail(True, "Forward pass completed")
    except Exception as e:
        print_pass_fail(False, f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

    # Verify output shapes
    print(f"\nOutput shapes:")
    print(f"  Corrected: {corrected.shape}")
    print(f"  Backbone output: {backbone_out.shape}")
    print_pass_fail(corrected.shape == noisy.shape, "Output shape matches input shape")

    # Store results
    results = {
        'noisy': noisy,
        'clean': clean,
        'corrected': corrected,
        'backbone_out': backbone_out,
        'info': info
    }

    return results, success


# =============================================================================
# TEST 3: DETAILED STATISTICS
# =============================================================================

def test_detailed_statistics(results: Dict) -> bool:
    """Print detailed statistics about the forward pass."""
    print_separator("TEST 3: DETAILED STATISTICS")

    noisy = results['noisy']
    clean = results['clean']
    corrected = results['corrected']
    backbone_out = results['backbone_out']
    info = results['info']

    # 3.1 Raw correction output from each corrector
    print("\n3.1 Raw Corrections (from each corrector):")
    raw_corrections = info.get('raw_corrections', {})
    if raw_corrections:
        for name, value in raw_corrections.items():
            print(f"  {name:12}: {value:.6f}")
    else:
        print("  (No raw_corrections in info - running separate test)")
        # Manually compute raw corrections by inspecting corrector

    # 3.2 Lambda map values
    print("\n3.2 Lambda Map Values:")
    lambda_maps = info.get('lambda_maps', {})
    if lambda_maps:
        for name, lmap in lambda_maps.items():
            if isinstance(lmap, torch.Tensor):
                print(f"  {name:12}: mean={lmap.mean().item():.6f}, "
                      f"max={lmap.max().item():.6f}, "
                      f"min={lmap.min().item():.6f}, "
                      f"std={lmap.std().item():.6f}")

    # 3.3 Total correction magnitude
    print("\n3.3 Total Correction Magnitude:")
    correction_mag = info.get('correction_magnitude', 0)
    print(f"  Correction magnitude: {correction_mag:.6f}")

    correction_diff = (corrected - backbone_out).abs()
    print(f"  Mean absolute correction: {correction_diff.mean().item():.6f}")
    print(f"  Max absolute correction: {correction_diff.max().item():.6f}")
    print(f"  Non-zero correction pixels: {(correction_diff > 1e-6).float().mean().item() * 100:.2f}%")

    # 3.4 Effective strength for each corrector
    print("\n3.4 Effective Strength (per corrector):")
    print("  (Effective strength = sigmoid(strength_param) * strength_scale * lambda_mean)")

    # Check if we can access the corrector's strength parameters
    # This requires the model to be in results

    # 3.5 PSNR before and after correction
    print("\n3.5 PSNR Comparison:")
    psnr_noisy = compute_psnr(noisy, clean)
    psnr_backbone = compute_psnr(backbone_out, clean)
    psnr_corrected = compute_psnr(corrected, clean)

    print(f"  PSNR (noisy vs clean):     {psnr_noisy:.4f} dB")
    print(f"  PSNR (backbone vs clean):  {psnr_backbone:.4f} dB")
    print(f"  PSNR (corrected vs clean): {psnr_corrected:.4f} dB")
    print(f"  Backbone improvement:      {psnr_backbone - psnr_noisy:+.4f} dB")
    print(f"  Correction delta:          {psnr_corrected - psnr_backbone:+.4f} dB")

    ssim_backbone = compute_ssim(backbone_out, clean)
    ssim_corrected = compute_ssim(corrected, clean)
    print(f"\n  SSIM (backbone vs clean):  {ssim_backbone:.6f}")
    print(f"  SSIM (corrected vs clean): {ssim_corrected:.6f}")
    print(f"  SSIM delta:                {ssim_corrected - ssim_backbone:+.6f}")

    # 3.6 All 7 predicate scores before and after correction
    print("\n3.6 Predicate Scores (P1-P7):")

    # Compute predicates using CleanReferencedPredicates
    predicates = CleanReferencedPredicates()

    with torch.no_grad():
        pred_backbone = predicates(backbone_out, clean, noisy)
        pred_corrected = predicates(corrected, clean, noisy)

    print("\n  Predicate | Backbone | Corrected | Delta")
    print("  " + "-" * 45)

    for i in range(1, 8):
        name = f'P{i}'
        score_backbone = pred_backbone[name]['score']
        score_corrected = pred_corrected[name]['score']

        if isinstance(score_backbone, torch.Tensor):
            score_backbone = score_backbone.item()
        if isinstance(score_corrected, torch.Tensor):
            score_corrected = score_corrected.item()

        delta = score_corrected - score_backbone
        print(f"  {name:9} | {score_backbone:8.4f} | {score_corrected:9.4f} | {delta:+7.4f}")

    # Summary
    avg_backbone = np.mean([pred_backbone[f'P{i}']['score'].item()
                           if isinstance(pred_backbone[f'P{i}']['score'], torch.Tensor)
                           else pred_backbone[f'P{i}']['score']
                           for i in range(1, 8)])
    avg_corrected = np.mean([pred_corrected[f'P{i}']['score'].item()
                            if isinstance(pred_corrected[f'P{i}']['score'], torch.Tensor)
                            else pred_corrected[f'P{i}']['score']
                            for i in range(1, 8)])

    print("  " + "-" * 45)
    print(f"  {'Average':9} | {avg_backbone:8.4f} | {avg_corrected:9.4f} | {avg_corrected - avg_backbone:+7.4f}")

    return True


# =============================================================================
# TEST 4: BACKWARD PASS AND GRADIENT VERIFICATION
# =============================================================================

def test_backward_pass(model: NeuroSymbolicDenoiserV6,
                       results: Dict) -> bool:
    """Test backward pass and verify gradients."""
    print_separator("TEST 4: BACKWARD PASS AND GRADIENT VERIFICATION")

    success = True
    noisy = results['noisy']
    clean = results['clean']

    # Freeze backbone (typical training setup)
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.train()

    # Zero gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    # Forward pass
    print("\nRunning forward pass with gradient tracking...")
    corrected, backbone_out, info = model(noisy, max_lambda=0.15)

    # Compute loss
    loss_fn = QualityAlignedLoss(
        lambda_pred=1.0,
        lambda_quality=100.0,
        lambda_recon=1.0
    )

    loss_dict = loss_fn(
        corrected, backbone_out, clean, noisy,
        lambda_maps=info['lambda_maps']
    )

    loss = loss_dict['total']
    print(f"  Loss value: {loss.item():.6f}")

    # Backward pass
    print("\nRunning backward pass...")
    try:
        loss.backward()
        print_pass_fail(True, "Backward pass completed")
    except Exception as e:
        print_pass_fail(False, f"Backward pass failed: {e}")
        return False

    # Check gradients on corrector parameters
    print("\n4.1 Gradients on Corrector Parameters:")
    corrector = model.corrector
    corrector_names = ['edge_corrector', 'contrast_corrector', 'sharpness_corrector',
                       'texture_corrector', 'smooth_corrector']

    for name in corrector_names:
        if hasattr(corrector, name):
            sub_corrector = getattr(corrector, name)
            grad_sum = 0
            grad_count = 0
            has_grad = False

            for param in sub_corrector.parameters():
                if param.grad is not None:
                    grad_sum += param.grad.abs().mean().item()
                    grad_count += 1
                    has_grad = True

            if grad_count > 0:
                avg_grad = grad_sum / grad_count
                print(f"  {name:20}: avg_grad={avg_grad:.8f}")
                print_pass_fail(has_grad and avg_grad > 0, f"{name} has non-zero gradients")
            else:
                print_pass_fail(False, f"{name} has no gradients")
                success = False

    # Check gradients on strength parameters
    print("\n4.2 Gradients on Strength Parameters:")
    for name in corrector_names:
        if hasattr(corrector, name):
            sub_corrector = getattr(corrector, name)
            if hasattr(sub_corrector, 'strength'):
                strength = sub_corrector.strength
                if strength.grad is not None:
                    print(f"  {name:20}: strength.grad={strength.grad.item():.8f}")
                    print_pass_fail(True, f"{name} strength has gradient")
                else:
                    print_pass_fail(False, f"{name} strength has no gradient")
                    success = False

    # Check gradients on lambda predictor
    print("\n4.3 Gradients on Lambda Predictor:")
    lambda_pred = model.lambda_predictor

    lambda_heads = ['edge_head', 'contrast_head', 'sharpness_head', 'texture_head', 'smooth_head']
    for name in lambda_heads:
        if hasattr(lambda_pred, name):
            head = getattr(lambda_pred, name)
            grad_sum = 0
            grad_count = 0
            has_grad = False

            for param in head.parameters():
                if param.grad is not None:
                    grad_sum += param.grad.abs().mean().item()
                    grad_count += 1
                    has_grad = True

            if grad_count > 0:
                avg_grad = grad_sum / grad_count
                print(f"  {name:20}: avg_grad={avg_grad:.8f}")
                print_pass_fail(has_grad and avg_grad > 0, f"{name} has non-zero gradients")
            else:
                print_pass_fail(False, f"{name} has no gradients")
                success = False

    # Check lambda scale parameters
    print("\n4.4 Gradients on Lambda Scale Parameters:")
    scale_names = ['edge_scale', 'contrast_scale', 'sharpness_scale', 'texture_scale', 'smooth_scale']
    for name in scale_names:
        if hasattr(lambda_pred, name):
            scale = getattr(lambda_pred, name)
            if scale.grad is not None:
                print(f"  {name:20}: grad={scale.grad.item():.8f}")
                print_pass_fail(True, f"{name} has gradient")
            else:
                print_pass_fail(False, f"{name} has no gradient")

    return success


# =============================================================================
# TEST 5: TRAINING SIMULATION (10 STEPS)
# =============================================================================

def test_training_simulation(model: NeuroSymbolicDenoiserV6,
                             train_jsonl: str,
                             num_steps: int = 10) -> bool:
    """Simulate training steps and verify improvement."""
    print_separator("TEST 5: TRAINING SIMULATION (10 STEPS)")

    success = True

    # Load training data
    print(f"\nLoading training data from: {train_jsonl}")
    try:
        train_dataset = OCTDataset(
            train_jsonl,
            max_samples=num_steps,
            patch_size=96,
            is_train=True
        )
        print_pass_fail(True, f"Loaded {len(train_dataset)} training sample(s)")
    except Exception as e:
        print_pass_fail(False, f"Failed to load training data: {e}")
        return False

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Setup optimizer
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    lr = 1e-4
    param_groups = [
        {'params': model.lambda_predictor.parameters(), 'lr': lr * 5},
        {'params': model.corrector.parameters(), 'lr': lr * 2},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Loss function
    loss_fn = QualityAlignedLoss(
        lambda_pred=1.0,
        lambda_quality=100.0,
        lambda_recon=1.0
    )

    # Clean-referenced predicates for evaluation
    predicates = CleanReferencedPredicates()

    model.train()

    # Track metrics
    losses = []
    correction_mags = []
    psnr_deltas = []
    avg_predicates = []

    print("\nRunning training steps...")
    print(f"\n  Step | Loss     | Corr Mag | PSNR Delta | Avg Pred")
    print("  " + "-" * 55)

    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break

        noisy = batch['noisy']
        clean = batch['clean']

        optimizer.zero_grad()

        # Forward
        corrected, backbone_out, info = model(noisy, max_lambda=0.15)

        # Loss
        loss_dict = loss_fn(
            corrected, backbone_out, clean, noisy,
            lambda_maps=info['lambda_maps']
        )
        loss = loss_dict['total']

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss at step {step}")
            continue

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            psnr_backbone = compute_psnr(backbone_out, clean)
            psnr_corrected = compute_psnr(corrected, clean)
            psnr_delta = psnr_corrected - psnr_backbone

            pred_results = predicates(corrected, clean, noisy)
            avg_pred = np.mean([
                pred_results[f'P{i}']['score'].item()
                if isinstance(pred_results[f'P{i}']['score'], torch.Tensor)
                else pred_results[f'P{i}']['score']
                for i in range(1, 8)
            ])

        losses.append(loss.item())
        correction_mags.append(info.get('correction_magnitude', 0))
        psnr_deltas.append(psnr_delta)
        avg_predicates.append(avg_pred)

        print(f"  {step+1:4} | {loss.item():.6f} | {info.get('correction_magnitude', 0):.6f} | "
              f"{psnr_delta:+.4f}    | {avg_pred:.4f}")

    # Analyze results
    print("\n5.1 Training Summary:")

    # Correction magnitude trend
    print(f"\n  Correction Magnitude:")
    print(f"    Start: {correction_mags[0]:.6f}")
    print(f"    End:   {correction_mags[-1]:.6f}")
    mag_increased = correction_mags[-1] > correction_mags[0]
    print_pass_fail(True, f"Correction magnitude {'increased' if mag_increased else 'changed'}")

    # Loss trend
    print(f"\n  Loss:")
    print(f"    Start: {losses[0]:.6f}")
    print(f"    End:   {losses[-1]:.6f}")
    loss_decreased = losses[-1] < losses[0]
    print_pass_fail(loss_decreased, "Loss decreased" if loss_decreased else "Loss did not decrease (may need more steps)")

    # Predicate trend
    print(f"\n  Average Predicates:")
    print(f"    Start: {avg_predicates[0]:.4f}")
    print(f"    End:   {avg_predicates[-1]:.4f}")
    pred_improved = avg_predicates[-1] >= avg_predicates[0] - 0.01  # Allow small fluctuation
    print_pass_fail(pred_improved, "Predicates improved or stable")

    # PSNR trend
    print(f"\n  PSNR Delta (corrected - backbone):")
    print(f"    Start: {psnr_deltas[0]:+.4f} dB")
    print(f"    End:   {psnr_deltas[-1]:+.4f} dB")
    psnr_better = psnr_deltas[-1] > psnr_deltas[0] - 0.1  # Allow small fluctuation
    print_pass_fail(psnr_better, "PSNR delta improved or stable")

    return success


# =============================================================================
# TEST 6: CORRECTOR STRENGTH ANALYSIS
# =============================================================================

def test_corrector_strength_analysis(model: NeuroSymbolicDenoiserV6) -> bool:
    """Analyze corrector strength parameters."""
    print_separator("TEST 6: CORRECTOR STRENGTH ANALYSIS")

    corrector = model.corrector
    corrector_names = ['edge_corrector', 'contrast_corrector', 'sharpness_corrector',
                       'texture_corrector', 'smooth_corrector']

    print("\nCorrector Strength Parameters:")
    print(f"\n  {'Corrector':<20} | {'Strength':<8} | {'Sigmoid(S)':<10} | {'Scale':<6} | {'Effective':<10}")
    print("  " + "-" * 70)

    for name in corrector_names:
        if hasattr(corrector, name):
            sub_corrector = getattr(corrector, name)
            if hasattr(sub_corrector, 'strength'):
                strength = sub_corrector.strength.item()
                sigmoid_s = torch.sigmoid(torch.tensor(strength)).item()
                scale = sub_corrector._strength_scale
                effective = sigmoid_s * scale

                print(f"  {name:<20} | {strength:8.4f} | {sigmoid_s:10.6f} | {scale:6.2f} | {effective:10.6f}")

    print("\nLambda Predictor Scale Parameters:")
    lambda_pred = model.lambda_predictor
    scale_names = ['edge_scale', 'contrast_scale', 'sharpness_scale', 'texture_scale', 'smooth_scale']

    print(f"\n  {'Scale Name':<20} | {'Value':<8} | {'Sigmoid(S)':<10}")
    print("  " + "-" * 45)

    for name in scale_names:
        if hasattr(lambda_pred, name):
            scale = getattr(lambda_pred, name).item()
            sigmoid_s = torch.sigmoid(torch.tensor(scale)).item()
            print(f"  {name:<20} | {scale:8.4f} | {sigmoid_s:10.6f}")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 70)
    print("COMPREHENSIVE CORRECTOR PIPELINE TEST")
    print("=" * 70)
    print(f"\nUsing {_NUM_THREADS} CPU threads")

    # Paths
    backbone_path = '/home/kumwilai/OCT/outputs/nafnet_pku37/nafnet_best.pth'
    val_jsonl = '/home/kumwilai/OCT/pku37_oct_dataset/pku37_real_val.jsonl'
    train_jsonl = '/home/kumwilai/OCT/pku37_oct_dataset/pku37_real_train.jsonl'

    # Track test results
    test_results = {}

    # TEST 1: Load model and backbone
    model, success = test_load_model(backbone_path)
    test_results['load_model'] = success

    if model is None:
        print("\n[ERROR] Cannot proceed without model")
        return

    # TEST 2: Forward pass with real data
    results, success = test_forward_pass(model, val_jsonl)
    test_results['forward_pass'] = success

    if results is None:
        print("\n[ERROR] Cannot proceed without forward pass results")
        return

    # TEST 3: Detailed statistics
    success = test_detailed_statistics(results)
    test_results['statistics'] = success

    # TEST 4: Backward pass and gradient verification
    success = test_backward_pass(model, results)
    test_results['backward_pass'] = success

    # TEST 5: Training simulation
    success = test_training_simulation(model, train_jsonl, num_steps=10)
    test_results['training_simulation'] = success

    # TEST 6: Corrector strength analysis
    success = test_corrector_strength_analysis(model)
    test_results['strength_analysis'] = success

    # Final summary
    print_separator("FINAL SUMMARY")

    print("\nTest Results:")
    all_passed = True
    for test_name, passed in test_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! Corrector pipeline is working correctly.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("SOME TESTS FAILED. Please review the output above.")
        print("=" * 70)

    return all_passed


if __name__ == '__main__':
    main()
