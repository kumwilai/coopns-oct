#!/usr/bin/env python3
"""
Gradient Flow Checker for V8 Cooperative Training

This script verifies that gradients flow correctly to all trainable parameters.
Run this to debug gradient issues.

Usage:
    python check_gradient_flow_v8.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# Import the model
from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    CooperativeLoss
)


def check_gradient_flow():
    """Check if gradients flow to all trainable parameters."""
    print("=" * 80)
    print("GRADIENT FLOW CHECKER FOR V8 COOPERATIVE TRAINING")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create model
    print("\nInitializing model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone=None,
    ).to(device)

    # Freeze backbone (like in training)
    print("Freezing backbone...")
    for param in model.backbone.backbone.parameters():
        param.requires_grad = False

    # Keep uncertainty head + calibration params trainable
    for param in model.backbone.uncertainty_head.parameters():
        param.requires_grad = True
    model.backbone.uncertainty_temperature.requires_grad = True
    model.backbone.uncertainty_bias_offset.requires_grad = True

    # Create loss
    criterion = CooperativeLoss(
        cooperation_weight=0.1,
        efficiency_weight=0.05,
        clinical_weight=0.5,
        cnr_weight=0.3,
        psnr_slack=1.0,
        use_uncertainty_weighting=True
    ).to(device)

    # Create dummy batch
    B, C, H, W = 2, 1, 128, 128
    noisy = torch.rand(B, C, H, W, device=device, requires_grad=False)
    clean = torch.rand(B, C, H, W, device=device, requires_grad=False)

    print(f"\nInput shape: {noisy.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    model.train()
    corrected, backbone_out, info = model(noisy)

    print(f"Corrected shape: {corrected.shape}")
    print(f"Backbone output shape: {backbone_out.shape}")

    # Compute loss
    print("\nComputing loss...")
    loss, metrics = criterion(corrected, backbone_out, clean, noisy, info)
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()

    # Check gradients
    print("\n" + "=" * 80)
    print("GRADIENT CHECK RESULTS")
    print("=" * 80)

    # Categorize parameters by module
    module_categories = {
        'backbone.backbone': [],
        'backbone.uncertainty': [],
        'corrector.correctors': [],
        'corrector.potential_estimators': [],
        'corrector.negotiator': [],
        'corrector.router': [],
        'corrector.lambda_predictor': [],
        'corrector.cnr_preserver': [],
        'corrector.clinical_enhancer': [],
        'corrector.region_aware_corrector': [],
        'corrector.confidence_estimator': [],
        'corrector.verifier': [],
        'corrector.predicates': [],
        'corrector.other': [],
        'criterion': [],
    }

    def categorize_param(name):
        for cat in module_categories.keys():
            if name.startswith(cat):
                return cat
        if name.startswith('corrector.'):
            return 'corrector.other'
        return 'other'

    # Track gradients by category
    grad_stats = {cat: {'total': 0, 'with_grad': 0, 'no_grad': 0, 'frozen': 0} for cat in module_categories}

    no_grad_params = []
    frozen_params = []
    with_grad_params = []

    for name, param in model.named_parameters():
        cat = categorize_param(name)
        grad_stats[cat]['total'] += 1

        if not param.requires_grad:
            grad_stats[cat]['frozen'] += 1
            frozen_params.append(name)
        elif param.grad is not None and param.grad.abs().sum() > 0:
            grad_stats[cat]['with_grad'] += 1
            with_grad_params.append(name)
        else:
            grad_stats[cat]['no_grad'] += 1
            no_grad_params.append(name)

    # Print summary by category
    print("\n{:<40} {:>8} {:>10} {:>10} {:>10}".format(
        "Module", "Total", "With Grad", "No Grad", "Frozen"))
    print("-" * 80)

    for cat, stats in grad_stats.items():
        if stats['total'] > 0:
            status = "OK" if stats['no_grad'] == 0 else "ISSUE!"
            print("{:<40} {:>8} {:>10} {:>10} {:>10}  {}".format(
                cat, stats['total'], stats['with_grad'], stats['no_grad'], stats['frozen'],
                status if stats['no_grad'] > 0 else ""))

    # Print parameters with NO gradients (but requires_grad=True)
    if no_grad_params:
        print("\n" + "=" * 80)
        print("WARNING: Parameters with requires_grad=True but NO gradients:")
        print("=" * 80)
        for name in sorted(no_grad_params):
            print(f"  {name}")

    # Check loss function parameters
    print("\n" + "=" * 80)
    print("LOSS FUNCTION PARAMETERS")
    print("=" * 80)
    for name, param in criterion.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            print(f"  {name}: grad OK (mean={param.grad.mean().item():.6f})")
        elif param.requires_grad:
            print(f"  {name}: NO GRAD!")
        else:
            print(f"  {name}: frozen")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    total_no_grad = total_trainable - total_with_grad

    print(f"Total trainable parameters: {total_trainable}")
    print(f"Parameters receiving gradients: {total_with_grad}")
    print(f"Parameters NOT receiving gradients: {total_no_grad}")

    if total_no_grad > 0:
        print("\n*** GRADIENT FLOW ISSUES DETECTED ***")
        print("Some trainable parameters are not receiving gradients.")
        print("Check the forward pass for torch.no_grad() blocks or .detach() calls.")
        return False
    else:
        print("\n*** ALL GRADIENTS FLOWING CORRECTLY ***")
        return True


def check_optimizer_params():
    """Check which parameters are included in the optimizer."""
    print("\n" + "=" * 80)
    print("OPTIMIZER PARAMETER CHECK")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone=None,
    ).to(device)

    # Freeze backbone
    for param in model.backbone.backbone.parameters():
        param.requires_grad = False

    # Collect parameters as done in training
    corrector_params = list(model.corrector.correctors.parameters())

    potential_params = []
    if hasattr(model.corrector, 'potential_estimators'):
        potential_params = list(model.corrector.potential_estimators.parameters())
    if hasattr(model.corrector, 'lambda_predictor'):
        potential_params += list(model.corrector.lambda_predictor.parameters())

    negotiator_params = []
    if hasattr(model.corrector, 'negotiator'):
        negotiator_params = list(model.corrector.negotiator.parameters())
    if hasattr(model.corrector, 'router'):
        negotiator_params += list(model.corrector.router.parameters())

    # BUG FIX: Include clinical enhancement modules
    clinical_enhancement_params = []
    if hasattr(model.corrector, 'cnr_preserver'):
        clinical_enhancement_params += list(model.corrector.cnr_preserver.parameters())
    if hasattr(model.corrector, 'clinical_enhancer'):
        clinical_enhancement_params += list(model.corrector.clinical_enhancer.parameters())
    if hasattr(model.corrector, 'region_aware_corrector'):
        clinical_enhancement_params += list(model.corrector.region_aware_corrector.parameters())
    if hasattr(model.corrector, 'confidence_estimator'):
        clinical_enhancement_params += list(model.corrector.confidence_estimator.parameters())

    uncertainty_params = (
        list(model.backbone.uncertainty_head.parameters()) +
        [model.backbone.uncertainty_temperature, model.backbone.uncertainty_bias_offset]
    )

    # Collect all optimizer params
    optimizer_params = set()
    for p in corrector_params:
        optimizer_params.add(id(p))
    for p in potential_params:
        optimizer_params.add(id(p))
    for p in negotiator_params:
        optimizer_params.add(id(p))
    for p in clinical_enhancement_params:
        optimizer_params.add(id(p))
    for p in uncertainty_params:
        optimizer_params.add(id(p))

    # Check for trainable params not in optimizer
    missing_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in optimizer_params:
            missing_params.append(name)

    print(f"\nParameters in optimizer groups:")
    print(f"  Corrector params: {len(corrector_params)}")
    print(f"  Potential params: {len(potential_params)}")
    print(f"  Negotiator params: {len(negotiator_params)}")
    print(f"  Clinical enhancement params: {len(clinical_enhancement_params)}")
    print(f"  Uncertainty params: {len(uncertainty_params)}")
    print(f"  Total in optimizer: {len(optimizer_params)}")

    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable params in model: {total_trainable}")

    if missing_params:
        print("\n*** WARNING: Trainable parameters NOT in optimizer: ***")
        for name in sorted(missing_params):
            print(f"  {name}")
        return False
    else:
        print("\n*** All trainable parameters are in optimizer ***")
        return True


if __name__ == '__main__':
    grad_ok = check_gradient_flow()
    print("\n")
    opt_ok = check_optimizer_params()

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    if grad_ok and opt_ok:
        print("ALL CHECKS PASSED!")
    else:
        print("ISSUES FOUND - Please review the output above.")
        sys.exit(1)
