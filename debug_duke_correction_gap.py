#!/usr/bin/env python3
"""
Diagnostic script: Why does the cooperative corrector produce 5x smaller
corrections on Duke17 (0.003) vs Duke2013 val (0.014) vs PKU37 (~0.015)?

Loads the adapted model and runs inference on 2 Duke17 + 2 Duke2013 images,
capturing all intermediate values in the correction pipeline:
  - Backbone output stats
  - Uncertainty map stats
  - Per-corrector allocation (from SymbolicNegotiator)
  - Per-corrector lambda (from AdaptiveLambdaPredictorV8)
  - Raw correction magnitude BEFORE gates
  - Final correction magnitude AFTER all gates
  - Tissue mask coverage
  - Edge restoration magnitude
  - Intensity gate coverage
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    otsu_tissue_mask,
)


def load_image(path):
    """Load a single image as [1, 1, H, W] tensor in [0, 1]."""
    img = np.array(Image.open(path)).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def load_jsonl_samples(jsonl_path, n=2):
    """Load first n samples from a JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if 'noisy_path' in entry and 'clean_path' in entry:
                if os.path.exists(entry['noisy_path']) and os.path.exists(entry['clean_path']):
                    samples.append(entry)
            if len(samples) >= n:
                break
    return samples


def fmt(val, width=10):
    """Format a float value for nice table output."""
    if isinstance(val, str):
        return f"{val:>{width}}"
    return f"{val:>{width}.6f}"


def analyze_image(model, noisy_tensor, label, device='cpu'):
    """
    Run inference with full intermediate capture.
    Returns a dict of all intermediate stats.
    """
    noisy_tensor = noisy_tensor.to(device)

    with torch.no_grad():
        # ---- Step 1: Backbone forward ----
        backbone_out, nafnet_uncertainty = model.backbone(noisy_tensor)

        stats = {}
        stats['label'] = label
        stats['input_shape'] = list(noisy_tensor.shape)

        # Backbone output stats
        stats['backbone_mean'] = backbone_out.mean().item()
        stats['backbone_std'] = backbone_out.std().item()
        stats['backbone_min'] = backbone_out.min().item()
        stats['backbone_max'] = backbone_out.max().item()

        # Noisy input stats
        stats['noisy_mean'] = noisy_tensor.mean().item()
        stats['noisy_std'] = noisy_tensor.std().item()
        stats['noisy_min'] = noisy_tensor.min().item()
        stats['noisy_max'] = noisy_tensor.max().item()

        # Uncertainty stats
        stats['uncertainty_mean'] = nafnet_uncertainty.mean().item()
        stats['uncertainty_std'] = nafnet_uncertainty.std().item()
        stats['uncertainty_min'] = nafnet_uncertainty.min().item()
        stats['uncertainty_max'] = nafnet_uncertainty.max().item()

        # Confidence = 1 - uncertainty
        nafnet_confidence = 1.0 - nafnet_uncertainty.clamp(0, 1)
        stats['confidence_mean'] = nafnet_confidence.mean().item()

        # ---- Step 2: Predicates ----
        corrector = model.corrector
        pred_results = corrector.predicates(backbone_out, noisy_tensor)

        # Extract failure maps and scores
        failure_maps = {}
        pred_score_dict = {}
        for key in ['P1', 'P2', 'P3', 'P4', 'P6']:
            pred_data = pred_results[key]
            failure_maps[key] = pred_data['failure_map'].detach()
            score = pred_data['score']
            pred_score_dict[key] = score if isinstance(score, torch.Tensor) else torch.tensor(score, device=device)
            stats[f'pred_{key}_score'] = pred_score_dict[key].item() if isinstance(pred_score_dict[key], torch.Tensor) else float(pred_score_dict[key])
            stats[f'failure_{key}_mean'] = failure_maps[key].mean().item()
            stats[f'failure_{key}_max'] = failure_maps[key].max().item()

        # Combined failure maps per corrector
        combined_failure = torch.stack([failure_maps[k] for k in ['P1', 'P2', 'P3', 'P4', 'P6']]).max(dim=0)[0]
        combined_pred_score = pred_score_dict['P1']
        for k in ['P2', 'P3', 'P4', 'P6']:
            combined_pred_score = torch.min(combined_pred_score, pred_score_dict[k])
        stats['combined_failure_mean'] = combined_failure.mean().item()
        stats['combined_failure_max'] = combined_failure.max().item()
        stats['combined_pred_score'] = combined_pred_score.item()

        # ---- Step 3: Corrector raw output (gain corrector) ----
        gain_corrector = corrector.correctors['gain']
        correction_raw, potential = gain_corrector(
            backbone_out, noisy_tensor, combined_failure, combined_pred_score, None
        )
        stats['raw_correction_mean'] = correction_raw.abs().mean().item()
        stats['raw_correction_max'] = correction_raw.abs().max().item()
        stats['raw_correction_std'] = correction_raw.std().item()
        stats['potential_mean'] = potential.mean().item()
        stats['potential_max'] = potential.max().item()
        stats['potential_std'] = potential.std().item()

        # ---- Step 3.5: Potential modulated by uncertainty ----
        nafnet_uncertainty_for_mod = 1.0 - nafnet_confidence
        modulated_potential = potential * (0.5 + nafnet_uncertainty_for_mod)
        stats['modulated_potential_mean'] = modulated_potential.mean().item()
        stats['modulated_potential_max'] = modulated_potential.max().item()

        # ---- Step 4: Negotiator allocation ----
        potentials_dict = {'gain': modulated_potential}
        allocations, negotiation_info = corrector.negotiator(
            nafnet_confidence, potentials_dict, pred_score_dict
        )
        alloc = allocations['gain']
        stats['allocation_mean'] = alloc.mean().item()
        stats['allocation_max'] = alloc.max().item()
        stats['allocation_min'] = alloc.min().item()
        stats['allocation_std'] = alloc.std().item()

        # ---- Step 5: Lambda maps ----
        lambda_maps_raw = corrector.lambda_predictor(backbone_out, failure_maps)
        tissue_lam = lambda_maps_raw.get('tissue', torch.ones_like(backbone_out))
        boundary_lam = lambda_maps_raw.get('boundary', torch.ones_like(backbone_out))
        lambda_val = torch.max(tissue_lam, boundary_lam)
        stats['lambda_tissue_mean'] = tissue_lam.mean().item()
        stats['lambda_tissue_max'] = tissue_lam.max().item()
        stats['lambda_boundary_mean'] = boundary_lam.mean().item()
        stats['lambda_boundary_max'] = boundary_lam.max().item()
        stats['lambda_combined_mean'] = lambda_val.mean().item()
        stats['lambda_combined_max'] = lambda_val.max().item()

        # ---- Step 6: Weighted alpha = correction * allocation * lambda ----
        target_shape = backbone_out.shape[2:]
        if correction_raw.shape[2:] != target_shape:
            correction_raw_rs = F.interpolate(correction_raw, size=target_shape, mode='bilinear', align_corners=False)
        else:
            correction_raw_rs = correction_raw
        if alloc.shape[2:] != target_shape:
            alloc_rs = F.interpolate(alloc, size=target_shape, mode='bilinear', align_corners=False)
        else:
            alloc_rs = alloc
        if lambda_val.shape[2:] != target_shape:
            lambda_rs = F.interpolate(lambda_val, size=target_shape, mode='bilinear', align_corners=False)
        else:
            lambda_rs = lambda_val

        weighted_alpha = correction_raw_rs * alloc_rs * lambda_rs
        stats['weighted_alpha_mean'] = weighted_alpha.abs().mean().item()
        stats['weighted_alpha_max'] = weighted_alpha.abs().max().item()

        # Clamp
        alpha_map = weighted_alpha.clamp(-0.20, 0.20)
        stats['clamped_alpha_mean'] = alpha_map.abs().mean().item()

        # ---- Step 7: Intensity gate (content-adaptive) ----
        B = backbone_out.size(0)
        bg_thresh = torch.quantile(
            backbone_out.detach().view(B, -1), 0.15, dim=1
        ).view(B, 1, 1, 1)
        intensity_gate = torch.sigmoid((backbone_out.detach() - bg_thresh) * 20.0)
        stats['intensity_gate_coverage'] = (intensity_gate > 0.5).float().mean().item()
        stats['intensity_gate_mean'] = intensity_gate.mean().item()
        stats['bg_thresh'] = bg_thresh.item()

        alpha_after_gate = alpha_map * intensity_gate
        stats['alpha_after_intgate_mean'] = alpha_after_gate.abs().mean().item()
        stats['alpha_after_intgate_max'] = alpha_after_gate.abs().max().item()

        # ---- Step 8: Multiplicative result ----
        multiplicative_result = backbone_out * (1.0 + alpha_after_gate)
        stats['mult_correction_mean'] = (multiplicative_result - backbone_out).abs().mean().item()

        # ---- Step 9: Edge restoration ----
        from neuro_symbolic_corrector_v8_enhanced import EdgeRecoveryModule, GuidedEdgeSharpener
        if isinstance(corrector.edge_sharpener, GuidedEdgeSharpener):
            edge_restoration = corrector.edge_sharpener(backbone_out)
        else:
            edge_restoration = corrector.edge_sharpener(backbone_out, noisy_tensor)

        # Apply intensity gate to edges too
        if intensity_gate.shape[2:] != edge_restoration.shape[2:]:
            edge_gate = F.interpolate(intensity_gate, size=edge_restoration.shape[2:],
                                      mode='bilinear', align_corners=False)
        else:
            edge_gate = intensity_gate
        edge_restoration_gated = edge_restoration * edge_gate

        stats['edge_raw_mean'] = edge_restoration.abs().mean().item()
        stats['edge_raw_max'] = edge_restoration.abs().max().item()
        stats['edge_gated_mean'] = edge_restoration_gated.abs().mean().item()

        # ---- Step 10: Final output ----
        candidate = (multiplicative_result + edge_restoration_gated).clamp(0, 1)
        total_correction = candidate - backbone_out
        stats['final_correction_mean'] = total_correction.abs().mean().item()
        stats['final_correction_max'] = total_correction.abs().max().item()

        # ---- Tissue mask stats ----
        tissue_mask = otsu_tissue_mask(backbone_out)
        stats['tissue_mask_coverage'] = (tissue_mask > 0.5).float().mean().item()
        stats['tissue_mask_mean'] = tissue_mask.mean().item()

        # ---- Full model forward for comparison ----
        corrected_full, backbone_out_full, info_full = model(noisy_tensor, return_details=False)
        stats['full_model_correction_mean'] = (corrected_full - backbone_out_full).abs().mean().item()

    return stats


def main():
    device = 'cpu'

    # Load model
    print("=" * 80)
    print("DIAGNOSTIC: Why Duke17 corrections are 5x smaller than Duke2013")
    print("=" * 80)
    print("\nLoading model...")

    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone='outputs/nafnet_pku37_w40/best_model.pth',
        hidden_channels=64,
    )

    # Load adapted checkpoint
    ckpt_path = 'outputs/nafnet_duke_adapt/best_model_cooperative.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded model_state_dict from {ckpt_path}")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded state_dict from {ckpt_path}")

    model = model.to(device)
    model.eval()

    # Load samples
    duke17_samples = load_jsonl_samples('duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl', n=2)
    duke2013_samples = load_jsonl_samples('duke2013_adapt_val.jsonl', n=2)

    print(f"\nDuke17 samples: {len(duke17_samples)}")
    print(f"Duke2013 val samples: {len(duke2013_samples)}")

    all_stats = []

    # Process Duke17
    for i, sample in enumerate(duke17_samples):
        print(f"\n{'=' * 70}")
        print(f"Processing Duke17 #{i+1}: {os.path.basename(sample['noisy_path'])}")
        noisy = load_image(sample['noisy_path'])
        label = f"Duke17_{sample.get('subject', i)}"
        stats = analyze_image(model, noisy, label, device)
        all_stats.append(stats)
        del noisy

    # Process Duke2013
    for i, sample in enumerate(duke2013_samples):
        print(f"\n{'=' * 70}")
        print(f"Processing Duke2013 #{i+1}: {os.path.basename(sample['noisy_path'])}")
        noisy = load_image(sample['noisy_path'])
        label = f"Duke2013_{sample.get('subject', i)}"
        stats = analyze_image(model, noisy, label, device)
        all_stats.append(stats)
        del noisy

    # ========================================================================
    # PRINT COMPARATIVE TABLE
    # ========================================================================
    print("\n\n" + "=" * 120)
    print("COMPARATIVE ANALYSIS: Duke17 vs Duke2013")
    print("=" * 120)

    # Group 1: Input and backbone stats
    print("\n--- INPUT & BACKBONE ---")
    headers = ['Metric'] + [s['label'] for s in all_stats]
    print(f"{'Metric':<35} " + "  ".join(f"{s['label']:>16}" for s in all_stats))
    print("-" * (35 + 18 * len(all_stats)))

    input_keys = [
        ('Input shape', 'input_shape'),
        ('Noisy mean', 'noisy_mean'),
        ('Noisy std', 'noisy_std'),
        ('Noisy range', None),
        ('Backbone mean', 'backbone_mean'),
        ('Backbone std', 'backbone_std'),
        ('Backbone range', None),
    ]
    for name, key in input_keys:
        if key is not None:
            vals = [s[key] for s in all_stats]
            if isinstance(vals[0], list):
                print(f"{name:<35} " + "  ".join(f"{str(v):>16}" for v in vals))
            else:
                print(f"{name:<35} " + "  ".join(f"{v:>16.6f}" for v in vals))
        elif name == 'Noisy range':
            print(f"{name:<35} " + "  ".join(f"[{s['noisy_min']:.3f},{s['noisy_max']:.3f}]".rjust(16) for s in all_stats))
        elif name == 'Backbone range':
            print(f"{name:<35} " + "  ".join(f"[{s['backbone_min']:.3f},{s['backbone_max']:.3f}]".rjust(16) for s in all_stats))

    # Group 2: Uncertainty
    print("\n--- UNCERTAINTY ---")
    unc_keys = [
        ('Uncertainty mean', 'uncertainty_mean'),
        ('Uncertainty std', 'uncertainty_std'),
        ('Uncertainty min', 'uncertainty_min'),
        ('Uncertainty max', 'uncertainty_max'),
        ('Confidence mean', 'confidence_mean'),
    ]
    for name, key in unc_keys:
        print(f"{name:<35} " + "  ".join(f"{s[key]:>16.6f}" for s in all_stats))

    # Group 3: Predicate scores
    print("\n--- PREDICATE SCORES ---")
    for pk in ['P1', 'P2', 'P3', 'P4', 'P6']:
        name = f"Pred {pk} score"
        print(f"{name:<35} " + "  ".join(f"{s[f'pred_{pk}_score']:>16.6f}" for s in all_stats))
    for pk in ['P1', 'P2', 'P3', 'P4', 'P6']:
        name = f"Failure {pk} mean"
        print(f"{name:<35} " + "  ".join(f"{s[f'failure_{pk}_mean']:>16.6f}" for s in all_stats))
    print(f"{'Combined failure mean':<35} " + "  ".join(f"{s['combined_failure_mean']:>16.6f}" for s in all_stats))
    print(f"{'Combined pred score (min)':<35} " + "  ".join(f"{s['combined_pred_score']:>16.6f}" for s in all_stats))

    # Group 4: Correction pipeline stages
    print("\n--- CORRECTION PIPELINE ---")
    pipeline_keys = [
        ('1. Raw correction |mean|', 'raw_correction_mean'),
        ('1. Raw correction |max|', 'raw_correction_max'),
        ('1. Raw correction std', 'raw_correction_std'),
        ('2. Potential mean', 'potential_mean'),
        ('2. Potential max', 'potential_max'),
        ('2. Modulated potential mean', 'modulated_potential_mean'),
        ('3. Allocation mean', 'allocation_mean'),
        ('3. Allocation max', 'allocation_max'),
        ('3. Allocation min', 'allocation_min'),
        ('3. Allocation std', 'allocation_std'),
        ('4. Lambda tissue mean', 'lambda_tissue_mean'),
        ('4. Lambda tissue max', 'lambda_tissue_max'),
        ('4. Lambda boundary mean', 'lambda_boundary_mean'),
        ('4. Lambda boundary max', 'lambda_boundary_max'),
        ('4. Lambda combined mean', 'lambda_combined_mean'),
        ('4. Lambda combined max', 'lambda_combined_max'),
        ('5. Weighted alpha |mean|', 'weighted_alpha_mean'),
        ('5. Weighted alpha |max|', 'weighted_alpha_max'),
        ('6. After clamp |mean|', 'clamped_alpha_mean'),
        ('7. BG threshold (15th pctile)', 'bg_thresh'),
        ('7. Intensity gate coverage', 'intensity_gate_coverage'),
        ('7. Intensity gate mean', 'intensity_gate_mean'),
        ('8. Alpha after int.gate |mean|', 'alpha_after_intgate_mean'),
        ('8. Alpha after int.gate |max|', 'alpha_after_intgate_max'),
        ('9. Mult correction |mean|', 'mult_correction_mean'),
        ('10. Edge raw |mean|', 'edge_raw_mean'),
        ('10. Edge raw |max|', 'edge_raw_max'),
        ('10. Edge gated |mean|', 'edge_gated_mean'),
        ('11. FINAL correction |mean|', 'final_correction_mean'),
        ('11. FINAL correction |max|', 'final_correction_max'),
        ('11. Full model corr |mean|', 'full_model_correction_mean'),
    ]
    for name, key in pipeline_keys:
        print(f"{name:<35} " + "  ".join(f"{s[key]:>16.6f}" for s in all_stats))

    # Group 5: Tissue mask
    print("\n--- TISSUE MASK ---")
    mask_keys = [
        ('Tissue mask coverage (>0.5)', 'tissue_mask_coverage'),
        ('Tissue mask mean', 'tissue_mask_mean'),
    ]
    for name, key in mask_keys:
        print(f"{name:<35} " + "  ".join(f"{s[key]:>16.6f}" for s in all_stats))

    # ========================================================================
    # RATIO ANALYSIS
    # ========================================================================
    print("\n\n" + "=" * 120)
    print("RATIO ANALYSIS: Duke17 avg / Duke2013 avg")
    print("=" * 120)

    # Average Duke17 and Duke2013 stats
    duke17_stats = [s for s in all_stats if 'Duke17' in s['label']]
    duke2013_stats = [s for s in all_stats if 'Duke2013' in s['label']]

    ratio_keys = [
        ('Raw correction |mean|', 'raw_correction_mean'),
        ('Potential mean', 'potential_mean'),
        ('Allocation mean', 'allocation_mean'),
        ('Lambda combined mean', 'lambda_combined_mean'),
        ('Weighted alpha |mean|', 'weighted_alpha_mean'),
        ('Intensity gate coverage', 'intensity_gate_coverage'),
        ('Alpha after int.gate |mean|', 'alpha_after_intgate_mean'),
        ('Edge gated |mean|', 'edge_gated_mean'),
        ('FINAL correction |mean|', 'final_correction_mean'),
        ('Uncertainty mean', 'uncertainty_mean'),
        ('Combined failure mean', 'combined_failure_mean'),
    ]

    print(f"\n{'Component':<35} {'Duke17 avg':>14} {'Duke2013 avg':>14} {'Ratio D17/D13':>14} {'Gap factor':>14}")
    print("-" * 95)

    for name, key in ratio_keys:
        d17_avg = np.mean([s[key] for s in duke17_stats])
        d13_avg = np.mean([s[key] for s in duke2013_stats])
        ratio = d17_avg / d13_avg if d13_avg > 1e-10 else float('inf')
        # Gap factor: how much this component contributes to the overall 5x gap
        print(f"{name:<35} {d17_avg:>14.6f} {d13_avg:>14.6f} {ratio:>14.4f}x {'<-- SUSPECT' if ratio < 0.5 or ratio > 2.0 else ''}")

    # ========================================================================
    # MULTIPLICATIVE DECOMPOSITION
    # ========================================================================
    print("\n\n" + "=" * 120)
    print("MULTIPLICATIVE DECOMPOSITION: correction = raw * allocation * lambda * intensity_gate")
    print("=" * 120)

    print(f"\n{'Stage product':<35} {'Duke17 avg':>14} {'Duke2013 avg':>14} {'Ratio':>14}")
    print("-" * 80)

    for ds_name, ds_stats in [('Duke17', duke17_stats), ('Duke2013', duke2013_stats)]:
        for s in ds_stats:
            raw = s['raw_correction_mean']
            alloc = s['allocation_mean']
            lam = s['lambda_combined_mean']
            intgate = s['intensity_gate_mean']
            product = raw * alloc * lam * intgate
            actual = s['alpha_after_intgate_mean']
            print(f"  {s['label']}: raw={raw:.6f} x alloc={alloc:.4f} x lambda={lam:.4f} x intgate={intgate:.4f} = {product:.6f}  (actual: {actual:.6f})")

    # ========================================================================
    # ROOT CAUSE IDENTIFICATION
    # ========================================================================
    print("\n\n" + "=" * 120)
    print("ROOT CAUSE IDENTIFICATION")
    print("=" * 120)

    # For each pipeline component, compute the ratio Duke17/Duke2013
    # The component with the largest gap explains the 5x difference
    components = [
        ('raw_correction_mean', 'Raw correction (gain corrector output)'),
        ('allocation_mean', 'Allocation (negotiator)'),
        ('lambda_combined_mean', 'Lambda (adaptive lambda predictor)'),
        ('intensity_gate_mean', 'Intensity gate (background suppression)'),
        ('edge_gated_mean', 'Edge restoration (after gating)'),
        ('uncertainty_mean', 'Uncertainty (backbone)'),
        ('combined_failure_mean', 'Predicate failure maps'),
    ]

    print(f"\n{'Component':<45} {'D17/D13 ratio':>14} {'Explanation'}")
    print("-" * 100)

    for key, name in components:
        d17_avg = np.mean([s[key] for s in duke17_stats])
        d13_avg = np.mean([s[key] for s in duke2013_stats])
        ratio = d17_avg / d13_avg if d13_avg > 1e-10 else float('inf')

        if ratio < 0.3:
            explanation = "*** MAJOR CAUSE - Duke17 has much less ***"
        elif ratio < 0.6:
            explanation = "** MODERATE CAUSE **"
        elif ratio > 3.0:
            explanation = "*** MAJOR CAUSE - Duke17 has much more ***"
        elif ratio > 1.5:
            explanation = "** MODERATE CAUSE (higher in Duke17) **"
        else:
            explanation = "(similar - not a cause)"

        print(f"  {name:<43} {ratio:>14.4f}x  {explanation}")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)


if __name__ == '__main__':
    main()
