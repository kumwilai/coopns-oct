#!/usr/bin/env python3
"""
Evaluate Noise Mixture Estimation (Not Classification!)

The correct evaluation for heterogeneous noise is:
- How well do predicted mixture weights match ground truth weights?
- NOT "which class is dominant?" (classification is wrong target)

Metrics:
1. Weight MSE: Mean squared error between predicted and GT weights
2. KL Divergence: Distribution similarity
3. Cosine Similarity: Direction similarity of weight vectors
4. Per-component correlation: Does predicted speckle correlate with GT speckle?
"""

import torch
import torch.nn.functional as F
import numpy as np


def evaluate_mixture_estimation(pred_weights, gt_weights):
    """
    Evaluate mixture estimation quality.

    Args:
        pred_weights: [B, 4] or [B, 4, H, W] predicted mixture weights
        gt_weights: [B, 4] ground truth mixture weights

    Returns:
        Dictionary of metrics
    """
    # If spatial, take global average
    if pred_weights.dim() == 4:
        pred_global = pred_weights.mean(dim=[2, 3])  # [B, 4]
    else:
        pred_global = pred_weights

    # Normalize to valid distributions
    pred_norm = F.softmax(pred_global, dim=1)
    gt_norm = F.softmax(gt_weights, dim=1)

    B = pred_norm.shape[0]
    metrics = {}

    # 1. Weight MSE (lower is better)
    weight_mse = F.mse_loss(pred_norm, gt_norm).item()
    metrics['weight_mse'] = weight_mse

    # 2. KL Divergence (lower is better)
    # KL(GT || Pred) - how much info lost using pred instead of gt
    kl_div = F.kl_div(
        pred_norm.log().clamp(min=-10),
        gt_norm,
        reduction='batchmean'
    ).item()
    metrics['kl_divergence'] = kl_div

    # 3. Cosine Similarity (higher is better)
    cos_sim = F.cosine_similarity(pred_norm, gt_norm, dim=1).mean().item()
    metrics['cosine_similarity'] = cos_sim

    # 4. Per-component correlation
    pred_np = pred_norm.detach().cpu().numpy()
    gt_np = gt_norm.detach().cpu().numpy()

    component_names = ['speckle', 'banding', 'gaussian', 'shot']
    for i, name in enumerate(component_names):
        if B > 1:
            corr = np.corrcoef(pred_np[:, i], gt_np[:, i])[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        metrics[f'{name}_correlation'] = corr

    # 5. Dominant class accuracy (for comparison, but not main metric)
    pred_dominant = pred_norm.argmax(dim=1)
    gt_dominant = gt_norm.argmax(dim=1)
    accuracy = (pred_dominant == gt_dominant).float().mean().item()
    metrics['dominant_accuracy'] = accuracy

    # 6. Weight ranking accuracy (are the orderings similar?)
    pred_ranks = pred_norm.argsort(dim=1, descending=True)
    gt_ranks = gt_norm.argsort(dim=1, descending=True)

    # Top-1 and Top-2 match
    top1_match = (pred_ranks[:, 0] == gt_ranks[:, 0]).float().mean().item()
    top2_match = (
        (pred_ranks[:, :2].unsqueeze(2) == gt_ranks[:, :2].unsqueeze(1)).any(dim=2).all(dim=1)
    ).float().mean().item()

    metrics['top1_match'] = top1_match
    metrics['top2_match'] = top2_match

    return metrics


def print_mixture_report(metrics):
    """Print mixture estimation report."""
    print("\n" + "=" * 60)
    print("NOISE MIXTURE ESTIMATION EVALUATION")
    print("=" * 60)

    print("\n--- Primary Metrics (Mixture Quality) ---")
    print(f"  Weight MSE:        {metrics['weight_mse']:.4f}  (lower is better)")
    print(f"  KL Divergence:     {metrics['kl_divergence']:.4f}  (lower is better)")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}  (higher is better)")

    print("\n--- Per-Component Correlation ---")
    print(f"  (Does predicted weight correlate with GT weight?)")
    for name in ['speckle', 'banding', 'gaussian', 'shot']:
        corr = metrics[f'{name}_correlation']
        bar = '#' * int(abs(corr) * 20)
        sign = '+' if corr >= 0 else '-'
        print(f"  {name:10s}: {sign}{abs(corr):.3f} {bar}")

    print("\n--- Ranking Metrics ---")
    print(f"  Top-1 Match:       {metrics['top1_match']*100:.1f}%  (dominant type correct)")
    print(f"  Top-2 Match:       {metrics['top2_match']*100:.1f}%  (top 2 types correct)")

    print("\n--- For Comparison (Not Primary Metric) ---")
    print(f"  Classification Acc: {metrics['dominant_accuracy']*100:.1f}%")

    print("=" * 60)


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing mixture evaluation metrics...")

    # Case 1: Perfect prediction
    gt = torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.3, 0.4, 0.2, 0.1]])
    pred = gt.clone()
    metrics = evaluate_mixture_estimation(pred, gt)
    print("\nCase 1: Perfect prediction")
    print_mixture_report(metrics)

    # Case 2: Slight error
    pred = gt + torch.randn_like(gt) * 0.1
    metrics = evaluate_mixture_estimation(pred, gt)
    print("\nCase 2: Slight error")
    print_mixture_report(metrics)

    # Case 3: Always predicts uniform
    pred = torch.ones_like(gt) * 0.25
    metrics = evaluate_mixture_estimation(pred, gt)
    print("\nCase 3: Always predicts uniform")
    print_mixture_report(metrics)

    # Case 4: Always predicts speckle
    pred = torch.zeros_like(gt)
    pred[:, 0] = 1.0
    metrics = evaluate_mixture_estimation(pred, gt)
    print("\nCase 4: Always predicts speckle")
    print_mixture_report(metrics)
