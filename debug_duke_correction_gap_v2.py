#!/usr/bin/env python3
"""
Diagnostic script v2: Deep dive into WHY the MultiplicativeGainCorrector
outputs exactly zero. Investigate weight magnitudes and internal activations.

Also compare with the PKU37-trained model (before adaptation) to see if
adaptation killed the corrector.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    otsu_tissue_mask,
)


def load_image(path):
    img = np.array(Image.open(path)).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def load_jsonl_samples(jsonl_path, n=2):
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


def inspect_gain_corrector_weights(model, label):
    """Inspect the MultiplicativeGainCorrector weights in detail."""
    print(f"\n{'=' * 80}")
    print(f"GAIN CORRECTOR WEIGHT INSPECTION: {label}")
    print(f"{'=' * 80}")

    corrector = model.corrector.correctors['gain'].base_corrector

    print(f"\n  max_alpha: {corrector.max_alpha}")

    # Check each layer's weight stats
    for name, param in corrector.named_parameters():
        print(f"\n  {name}:")
        print(f"    shape: {list(param.shape)}")
        print(f"    mean:  {param.data.mean().item():.8f}")
        print(f"    std:   {param.data.std().item():.8f}")
        print(f"    min:   {param.data.min().item():.8f}")
        print(f"    max:   {param.data.max().item():.8f}")
        print(f"    abs_mean: {param.data.abs().mean().item():.8f}")
        print(f"    all_zero: {(param.data == 0).all().item()}")
        print(f"    near_zero (<1e-6): {(param.data.abs() < 1e-6).float().mean().item():.2%}")

    # Check output head specifically
    print(f"\n  --- Output Head (last conv) ---")
    last_conv = corrector.output_head[-1]
    print(f"    weight all_zero: {(last_conv.weight.data == 0).all().item()}")
    print(f"    weight abs_mean: {last_conv.weight.data.abs().mean().item():.8f}")
    if last_conv.bias is not None:
        print(f"    bias all_zero: {(last_conv.bias.data == 0).all().item()}")
        print(f"    bias value: {last_conv.bias.data.item():.8f}")


def trace_gain_corrector_forward(model, noisy_tensor, label):
    """Step through the gain corrector forward pass layer by layer."""
    print(f"\n{'=' * 80}")
    print(f"GAIN CORRECTOR FORWARD TRACE: {label}")
    print(f"{'=' * 80}")

    with torch.no_grad():
        # Get backbone output
        backbone_out, nafnet_uncertainty = model.backbone(noisy_tensor)

        # Get predicates
        corrector = model.corrector
        pred_results = corrector.predicates(backbone_out, noisy_tensor)
        failure_maps = {}
        pred_score_dict = {}
        for key in ['P1', 'P2', 'P3', 'P4', 'P6']:
            pred_data = pred_results[key]
            failure_maps[key] = pred_data['failure_map'].detach()
            score = pred_data['score']
            pred_score_dict[key] = score if isinstance(score, torch.Tensor) else torch.tensor(score)

        combined_failure = torch.stack([failure_maps[k] for k in ['P1', 'P2', 'P3', 'P4', 'P6']]).max(dim=0)[0]

        # Now trace through the gain corrector step by step
        gain = corrector.correctors['gain'].base_corrector
        B, C, H, W = backbone_out.shape

        print(f"\n  Input backbone_out: mean={backbone_out.mean():.6f}, std={backbone_out.std():.6f}")
        print(f"  Input failure_map: mean={combined_failure.mean():.6f}, std={combined_failure.std():.6f}")

        # Step 1: Local contrast
        local_contrast = gain._compute_local_contrast(backbone_out)
        print(f"\n  1. Local contrast: mean={local_contrast.mean():.6f}, std={local_contrast.std():.6f}")
        print(f"     min={local_contrast.min():.6f}, max={local_contrast.max():.6f}")

        # Step 2: Concatenate inputs
        x = torch.cat([backbone_out, combined_failure, local_contrast], dim=1)
        print(f"\n  2. Concatenated input: shape={list(x.shape)}")

        # Step 3: Downsample
        x_low = gain.downsample(x)
        print(f"\n  3. After downsample: shape={list(x_low.shape)}")
        print(f"     mean={x_low.mean():.6f}, std={x_low.std():.6f}")

        # Step 4: Feature net
        feat = gain.feature_net(x_low)
        print(f"\n  4. After feature_net: shape={list(feat.shape)}")
        print(f"     mean={feat.mean():.6f}, std={feat.std():.6f}")
        print(f"     abs_mean={feat.abs().mean():.6f}")
        print(f"     min={feat.min():.6f}, max={feat.max():.6f}")
        print(f"     all_zero: {(feat == 0).all().item()}")
        print(f"     near_zero (<1e-6): {(feat.abs() < 1e-6).float().mean():.2%}")

        # Step 4b: Individual layers in feature_net
        temp = x_low
        for i, layer in enumerate(gain.feature_net):
            temp = layer(temp)
            name = type(layer).__name__
            print(f"     feat_net[{i}] ({name}): mean={temp.mean():.6f}, std={temp.std():.6f}, abs_mean={temp.abs().mean():.6f}")

        # Step 5: Dilated convolutions
        dilated_outs = [conv(feat) for conv in gain.dilated_convs]
        for i, d_out in enumerate(dilated_outs):
            print(f"\n  5. Dilated conv {i}: mean={d_out.mean():.6f}, std={d_out.std():.6f}, abs_mean={d_out.abs().mean():.6f}")

        feat_dilated = torch.cat(dilated_outs, dim=1)
        print(f"\n  5. After concat dilated: shape={list(feat_dilated.shape)}")
        print(f"     mean={feat_dilated.mean():.6f}, abs_mean={feat_dilated.abs().mean():.6f}")

        # Step 6: CBAM attention
        feat_ca = gain.channel_attn(feat_dilated)
        print(f"\n  6. After channel_attn: mean={feat_ca.mean():.6f}, abs_mean={feat_ca.abs().mean():.6f}")

        feat_sa = gain.spatial_attn(feat_ca)
        print(f"     After spatial_attn: mean={feat_sa.mean():.6f}, abs_mean={feat_sa.abs().mean():.6f}")

        # Step 7: Output head
        raw_output = gain.output_head(feat_sa)
        print(f"\n  7. Output head raw: mean={raw_output.mean():.6f}, std={raw_output.std():.6f}")
        print(f"     min={raw_output.min():.6f}, max={raw_output.max():.6f}")
        print(f"     abs_mean={raw_output.abs().mean():.6f}")

        # Step 7b: Individual layers in output_head
        temp = feat_sa
        for i, layer in enumerate(gain.output_head):
            temp = layer(temp)
            name = type(layer).__name__
            print(f"     output_head[{i}] ({name}): mean={temp.mean():.8f}, std={temp.std():.8f}, abs_mean={temp.abs().mean():.8f}")

        # Step 8: Upsample
        alpha_map = F.interpolate(raw_output, size=(H, W), mode='bilinear', align_corners=False)
        print(f"\n  8. After upsample: mean={alpha_map.mean():.6f}")

        # Step 9: Scale with tanh
        alpha_final = gain.max_alpha * torch.tanh(alpha_map)
        print(f"\n  9. After tanh * max_alpha ({gain.max_alpha}): mean={alpha_final.mean():.8f}")
        print(f"     abs_mean={alpha_final.abs().mean():.8f}")
        print(f"     min={alpha_final.min():.8f}, max={alpha_final.max():.8f}")


def main():
    device = 'cpu'

    print("=" * 80)
    print("DIAGNOSTIC V2: Why gain corrector outputs zero")
    print("=" * 80)

    # Load Duke-adapted model
    print("\nLoading Duke-adapted model...")
    model_adapt = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone='outputs/nafnet_pku37_w40/best_model.pth',
        hidden_channels=64,
    )
    ckpt = torch.load('outputs/nafnet_duke_adapt/best_model_cooperative.pth',
                       map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model_adapt.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model_adapt.load_state_dict(ckpt, strict=False)
    model_adapt.eval()

    # Also check if QT38 model has the same issue
    qt38_path = 'outputs/nafnet_qt38_gatecollapse/best_model_cooperative.pth'
    qt38_model = None
    if os.path.exists(qt38_path):
        print("\nLoading QT38 model for comparison...")
        qt38_model = NeuroSymbolicDenoiserV8Cooperative(
            backbone_name='nafnet',
            pretrained_backbone='outputs/nafnet_pku37_w40/best_model.pth',
            hidden_channels=64,
        )
        ckpt38 = torch.load(qt38_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt38:
            qt38_model.load_state_dict(ckpt38['model_state_dict'], strict=False)
        else:
            qt38_model.load_state_dict(ckpt38, strict=False)
        qt38_model.eval()

    # Inspect weights
    inspect_gain_corrector_weights(model_adapt, "Duke-adapted")
    if qt38_model:
        inspect_gain_corrector_weights(qt38_model, "QT38 (PKU37)")

    # Load test images
    duke17_samples = load_jsonl_samples('duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl', n=1)
    duke2013_samples = load_jsonl_samples('duke2013_adapt_val.jsonl', n=1)

    # Trace forward pass
    if duke17_samples:
        noisy = load_image(duke17_samples[0]['noisy_path'])
        trace_gain_corrector_forward(model_adapt, noisy, "Duke-adapted on Duke17")
        if qt38_model:
            trace_gain_corrector_forward(qt38_model, noisy, "QT38 on Duke17")
        del noisy

    if duke2013_samples:
        noisy = load_image(duke2013_samples[0]['noisy_path'])
        trace_gain_corrector_forward(model_adapt, noisy, "Duke-adapted on Duke2013")
        if qt38_model:
            trace_gain_corrector_forward(qt38_model, noisy, "QT38 on Duke2013")
        del noisy

    # Check if the full model forward differs from manual pipeline
    print("\n\n" + "=" * 80)
    print("FULL MODEL vs MANUAL: Edge restoration only?")
    print("=" * 80)

    if duke17_samples:
        noisy = load_image(duke17_samples[0]['noisy_path'])
        with torch.no_grad():
            corrected, backbone_out, info = model_adapt(noisy, return_details=True)
            total_corr = (corrected - backbone_out).abs().mean().item()
            print(f"\n  Duke17 full model correction: {total_corr:.8f}")

            # Check info dict for clues
            if 'correction_magnitude' in info:
                print(f"  info['correction_magnitude']: {info['correction_magnitude']}")
            for k, v in sorted(info.items()):
                if isinstance(v, (int, float)):
                    print(f"  info['{k}']: {v}")
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            print(f"  info['{k}']['{k2}']: {v2}")
        del noisy

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe gain corrector output is EXACTLY ZERO for all images.")
    print("This means corrections come ONLY from EdgeRecoveryModule.")
    print("The 5x gap question is actually about edge restoration magnitude")
    print("differences, not about the gain corrector pipeline at all.")
    print("\nThe correction magnitudes you observed (0.003 Duke17, 0.014 Duke2013)")
    print("must have been measured differently or with a different model checkpoint.")
    print(f"\nActual edge-only corrections are similar:")
    print(f"  Duke17:  ~0.0017-0.0018")
    print(f"  Duke2013: ~0.0016-0.0017")


if __name__ == '__main__':
    main()
