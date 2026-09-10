#!/usr/bin/env python3
"""
Comprehensive Ablation Study for TMI Submission

Tests the contribution of each component:
1. Physics-based features (coefficient of variation, signal-variance, etc.)
2. Anatomical layer guidance
3. Soft conditioning vs hard classification
4. Refinement network architecture
5. End-to-end vs frozen backbone training
"""

import argparse
import gc
import json
import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_soft_conditioning import SoftConditionedDenoiser, OCTDenoiseDataset
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.utils.metrics import compute_psnr, compute_ssim


# ============================================================================
# Ablation Model Variants
# ============================================================================

class BackboneOnly(nn.Module):
    """Ablation: Just the backbone, no refinement."""

    def __init__(self, backbone_ckpt=None):
        super().__init__()
        self.backbone = NAFNetFullFiLM(
            img_channel=1, width=64,
            enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
            middle_blk_num=2, cond_dim=32,
        )
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            ckpt = torch.load(backbone_ckpt, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)

    def forward(self, x, return_features=False):
        out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)
        if return_features:
            return out, {}
        return out


class NoPhysicsFeatures(SoftConditionedDenoiser):
    """Ablation: Remove physics-based feature conditioning."""

    def forward(self, x, return_features=False):
        B, C, H, W = x.shape

        # Skip physics features, use random features
        norm_features = torch.randn(B, 8, H, W, device=x.device) * 0.1

        # Get backbone output
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # Layer detection
        layer_logits = self.layer_detector(x)
        layer_prob = F.softmax(layer_logits, dim=1)

        # Refinement with random features
        refine_input = torch.cat([backbone_out, norm_features, layer_prob], dim=1)
        refinement = self.refinement(refine_input)

        scale = torch.sigmoid(self.refinement_scale) * 0.2
        denoised = backbone_out + scale * refinement
        denoised = denoised.clamp(0, 1)

        if return_features:
            return denoised, {'layer_prob': layer_prob}
        return denoised


class NoLayerGuidance(SoftConditionedDenoiser):
    """Ablation: Remove anatomical layer guidance."""

    def forward(self, x, return_features=False):
        B, C, H, W = x.shape

        # Physics features
        raw_features = self.feature_extractor(x)
        feature_stack = torch.cat([
            raw_features['coef_variation'],
            raw_features['local_std'],
            raw_features['signal_var_corr'],
            raw_features['horizontal_ratio'],
            raw_features['horizontal_lines'],
            raw_features['high_freq'],
            raw_features['local_range'],
        ], dim=1)
        norm_features = self.feature_norm(feature_stack)

        # Backbone
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # Replace layer prob with uniform distribution
        layer_prob = torch.ones(B, 5, H, W, device=x.device) / 5

        # Refinement
        refine_input = torch.cat([backbone_out, norm_features, layer_prob], dim=1)
        refinement = self.refinement(refine_input)

        scale = torch.sigmoid(self.refinement_scale) * 0.2
        denoised = backbone_out + scale * refinement
        denoised = denoised.clamp(0, 1)

        if return_features:
            return denoised, {'raw_features': raw_features, 'layer_prob': layer_prob}
        return denoised


class HardClassification(SoftConditionedDenoiser):
    """Ablation: Hard noise classification instead of soft conditioning."""

    def forward(self, x, return_features=False):
        B, C, H, W = x.shape

        # Physics features
        raw_features = self.feature_extractor(x)
        feature_stack = torch.cat([
            raw_features['coef_variation'],
            raw_features['local_std'],
            raw_features['signal_var_corr'],
            raw_features['horizontal_ratio'],
            raw_features['horizontal_lines'],
            raw_features['high_freq'],
            raw_features['local_range'],
        ], dim=1)
        norm_features = self.feature_norm(feature_stack)

        # HARD classification: one-hot instead of soft
        # Use argmax to get dominant noise type
        noise_type = norm_features[:, :4, :, :].mean(dim=[2, 3])  # Global
        hard_class = F.one_hot(noise_type.argmax(dim=1), num_classes=4).float()
        hard_class = hard_class.view(B, 4, 1, 1).expand(B, 4, H, W)

        # Replace soft features with hard one-hot
        hard_features = torch.cat([hard_class, norm_features[:, 4:, :, :]], dim=1)

        # Backbone
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # Layer detection
        layer_logits = self.layer_detector(x)
        layer_prob = F.softmax(layer_logits, dim=1)

        # Refinement
        refine_input = torch.cat([backbone_out, hard_features, layer_prob], dim=1)
        refinement = self.refinement(refine_input)

        scale = torch.sigmoid(self.refinement_scale) * 0.2
        denoised = backbone_out + scale * refinement
        denoised = denoised.clamp(0, 1)

        if return_features:
            return denoised, {'raw_features': raw_features, 'layer_prob': layer_prob, 'hard_class': hard_class}
        return denoised


class FrozenBackbone(SoftConditionedDenoiser):
    """Ablation: Frozen backbone (no end-to-end training)."""

    def __init__(self, backbone_ckpt=None):
        super().__init__(backbone_ckpt)
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False


# ============================================================================
# Ablation Study Runner
# ============================================================================

ABLATION_CONFIGS = {
    'full_model': {
        'description': 'Full model with all components',
        'model_class': SoftConditionedDenoiser,
    },
    'backbone_only': {
        'description': 'Just NAFNet backbone, no refinement',
        'model_class': BackboneOnly,
    },
    'no_physics': {
        'description': 'Remove physics-based features (random features)',
        'model_class': NoPhysicsFeatures,
    },
    'no_layer': {
        'description': 'Remove anatomical layer guidance',
        'model_class': NoLayerGuidance,
    },
    'hard_classification': {
        'description': 'Hard noise classification instead of soft',
        'model_class': HardClassification,
    },
    'frozen_backbone': {
        'description': 'Frozen backbone (refinement only)',
        'model_class': FrozenBackbone,
    },
}


def train_ablation_model(model, train_loader, device, epochs=10, lr=1e-4):
    """Train an ablation model."""
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            denoised = model(noisy)
            loss = F.mse_loss(denoised, clean)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        # Clear intermediate tensors
        del noisy, clean, denoised, loss

    return model


def evaluate_model(model, val_loader, device):
    """Evaluate a model."""
    model.eval()

    psnr_sum, ssim_sum = 0, 0
    n_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            denoised = model(noisy)
            denoised = denoised.clamp(0, 1)

            for i in range(noisy.size(0)):
                psnr_sum += compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim_sum += compute_ssim(denoised[i:i+1], clean[i:i+1])
                n_samples += 1

    return {
        'psnr': psnr_sum / n_samples,
        'ssim': ssim_sum / n_samples,
    }


def run_ablation_study(args):
    """Run the full ablation study."""
    print("="*70)
    print("ABLATION STUDY")
    print("="*70)

    # Load data
    train_ds = OCTDenoiseDataset(args.train_jsonl, args.patch_size, args.max_train)
    val_ds = OCTDenoiseDataset(args.val_jsonl, args.patch_size, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nData: {len(train_ds)} train, {len(val_ds)} val")

    results = {}

    for name, config in ABLATION_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Running ablation: {name}")
        print(f"Description: {config['description']}")
        print(f"{'='*70}")

        # Create model
        model = config['model_class'](backbone_ckpt=args.base_ckpt).to(args.device)

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

        # Train
        if trainable > 0:
            model = train_ablation_model(model, train_loader, args.device,
                                         epochs=args.epochs, lr=args.lr)

        # Evaluate
        metrics = evaluate_model(model, val_loader, args.device)

        results[name] = {
            'description': config['description'],
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'trainable_params': trainable,
        }

        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")

        # CRITICAL: Free model memory before next ablation
        del model
        gc.collect()
        if 'cuda' in args.device:
            torch.cuda.empty_cache()

    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"\n{'Ablation':<25} {'PSNR':>10} {'SSIM':>10} {'Params':>12}")
    print("-"*60)

    # Sort by PSNR
    sorted_results = sorted(results.items(), key=lambda x: x[1]['psnr'], reverse=True)
    best_psnr = sorted_results[0][1]['psnr']

    for name, r in sorted_results:
        diff = r['psnr'] - best_psnr
        diff_str = f"({diff:+.2f})" if diff != 0 else "(best)"
        print(f"{name:<25} {r['psnr']:>8.2f} dB {r['ssim']:>10.4f} {r['trainable_params']:>10,}")

    # Compute contribution of each component
    print("\n" + "-"*70)
    print("COMPONENT CONTRIBUTIONS")
    print("-"*70)

    full_psnr = results['full_model']['psnr']
    backbone_psnr = results['backbone_only']['psnr']

    print(f"  Total improvement over backbone: {full_psnr - backbone_psnr:+.2f} dB")

    if 'no_physics' in results:
        print(f"  Physics features contribute:     {full_psnr - results['no_physics']['psnr']:+.2f} dB")

    if 'no_layer' in results:
        print(f"  Layer guidance contributes:      {full_psnr - results['no_layer']['psnr']:+.2f} dB")

    if 'hard_classification' in results:
        print(f"  Soft vs hard routing:            {full_psnr - results['hard_classification']['psnr']:+.2f} dB")

    if 'frozen_backbone' in results:
        print(f"  End-to-end training contributes: {full_psnr - results['frozen_backbone']['psnr']:+.2f} dB")

    print("="*70)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/ablation_results.json")

    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--train_jsonl', default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_train', type=int, default=200)
    parser.add_argument('--max_val', type=int, default=50)
    parser.add_argument('--base_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='ablation_results')

    args = parser.parse_args()
    run_ablation_study(args)


if __name__ == '__main__':
    main()
