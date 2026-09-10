#!/usr/bin/env python3
"""
Ablation Study: Multi-Task WITHOUT Layer-Specific Gates

This script trains the multi-task model with a GLOBAL noise gate instead of
layer-specific gates. Used to prove the contribution of layer-specific modeling.

Comparison:
- Full method: 5 separate gates, one per layer (layer-specific)
- Ablation: 1 global gate for all layers (this script)
"""

import argparse
import json
import os
import sys
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.models.noise_features import NoiseFeatureExtractor
from nsnd.utils.metrics import compute_psnr, compute_ssim

# Import shared components from main training script
from train_multitask import (
    MultiTaskOCTDataset,
    LightweightLayerSegmenter,
    PerSampleFeatureAligner,
    compute_dice_score,
    compute_per_layer_dice,
    compute_boundary_loss,
    compute_class_weights_from_loader,
    LAYER_NAMES,
)


class GlobalNoiseGate(nn.Module):
    """
    ABLATION: Global noise gate (NOT layer-specific).

    This is the control condition - a single gate for all layers,
    instead of 5 separate layer-specific gates.
    """

    def __init__(self, noise_feat_dim=7, hidden_dim=16):
        super().__init__()

        # Single global gate (same architecture as one layer gate)
        self.gate = nn.Sequential(
            nn.Conv2d(noise_feat_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # Learnable sensitivity (single value, not per-layer)
        self.noise_sensitivity = nn.Parameter(torch.tensor(1.0))
        self.base_refinement = nn.Parameter(torch.tensor(0.0))

    def forward(self, noise_features, seg_probs=None):
        """
        Compute global noise gate (ignores segmentation).

        Args:
            noise_features: [B, 7, H, W]
            seg_probs: [B, 5, H, W] - IGNORED in ablation

        Returns:
            gate: [B, 1, H, W] - same gate for all pixels regardless of layer
        """
        gate = self.gate(noise_features)

        sensitivity = torch.sigmoid(self.noise_sensitivity)
        base = torch.sigmoid(self.base_refinement) * 0.5

        gate = base + sensitivity * gate
        gate = gate.clamp(0, 1)

        # Return in same format as LayerSpecificNoiseGates for compatibility
        # But all layers have the SAME gate value
        B, _, H, W = noise_features.shape
        layer_gates = gate.expand(B, 5, H, W)  # Same gate repeated 5 times

        gate_stats = {'global_gate_mean': gate.mean().item()}

        return gate, layer_gates, gate_stats


class MultiTaskDenoiserAblation(nn.Module):
    """
    ABLATION: Multi-Task Model WITHOUT Layer-Specific Gates.

    Uses a single global noise gate instead of 5 layer-specific gates.
    This proves the contribution of layer-specific modeling.
    """

    def __init__(self, backbone_ckpt=None, segmenter_ckpt=None, confidence_threshold=0.01):
        super().__init__()

        # NOTE: Set to 0.01 (not 0.1) because model confidence is naturally low
        self.confidence_threshold = confidence_threshold

        # 1. Physics-based feature extractor
        self.feature_extractor = NoiseFeatureExtractor()

        # 2. Shared backbone
        self.backbone = NAFNetFullFiLM(
            img_channel=1, width=64,
            enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
            middle_blk_num=2, cond_dim=32,
        )

        if backbone_ckpt and os.path.exists(backbone_ckpt):
            ckpt = torch.load(backbone_ckpt, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            print(f"[Ablation] Loaded backbone from {backbone_ckpt}")

        # 3. Layer segmentation (still used for multi-task, but NOT for gating)
        self.segmenter = LightweightLayerSegmenter(num_classes=5)

        if segmenter_ckpt and os.path.exists(segmenter_ckpt):
            ckpt = torch.load(segmenter_ckpt, map_location='cpu', weights_only=False)
            self.segmenter.load_state_dict(ckpt['state_dict'], strict=True)
            print(f"[Ablation] Loaded segmenter from {segmenter_ckpt}")

        # 4. Feature normalization
        self.feature_norm = nn.Sequential(
            nn.Conv2d(7, 16, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1),
        )

        # 4b. Feature alignment (same as full method)
        self.feature_aligner = PerSampleFeatureAligner(n_features=7)

        # 5. ABLATION: Global noise gate (NOT layer-specific)
        self.global_noise_gate = GlobalNoiseGate(noise_feat_dim=7, hidden_dim=16)

        # 6. Layer-aware refinement (same architecture)
        self.refinement = nn.Sequential(
            nn.Conv2d(1 + 8 + 5, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )

        self.refinement_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, return_features=False):
        # 1. Extract physics features
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

        # 2. Backbone denoising
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # 3. Layer segmentation (for multi-task loss, NOT for gating)
        seg_logits = self.segmenter(x)
        seg_probs = F.softmax(seg_logits, dim=1)

        # 4. Align features
        aligned_features = self.feature_aligner(feature_stack)

        # 5. ABLATION: Global noise gate (ignores layer information)
        noise_gate, layer_gates, gate_stats = self.global_noise_gate(
            aligned_features, seg_probs  # seg_probs is IGNORED
        )

        # 6. Refinement
        refine_input = torch.cat([backbone_out, norm_features, seg_logits], dim=1)
        refinement_raw = self.refinement(refine_input)

        # 7. Apply global gate
        refinement = refinement_raw * noise_gate

        scale = torch.sigmoid(self.refinement_scale) * 0.2
        denoised = backbone_out + scale * refinement
        denoised = denoised.clamp(0, 1)

        if return_features:
            return denoised, seg_logits, {
                'raw_features': raw_features,
                'aligned_features': aligned_features,
                'norm_features': norm_features,
                'backbone_out': backbone_out,
                'refinement_raw': refinement_raw,
                'refinement': refinement,
                'noise_gate': noise_gate,
                'layer_gates': layer_gates,  # All same for ablation
                'layer_gate_stats': gate_stats,
                'seg_probs': seg_probs,
                'scale': scale,
            }

        return denoised, seg_logits


def train_epoch(model, loader, optimizer, device, epoch, lambda_seg=1.0, lambda_boundary=0.1, seg_class_weights=None):
    """Train one epoch (simplified - no diversity loss since no layer gates).

    Args:
        seg_class_weights: Optional tensor [5] with class weights for segmentation loss.
    """
    model.train()

    total_denoise_loss = 0
    total_seg_loss = 0
    total_boundary_loss = 0
    total_loss = 0
    total_dice = 0
    total_gate_mean = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        seg_mask = batch['seg_mask'].to(device)

        optimizer.zero_grad(set_to_none=True)

        denoised, seg_logits, features = model(noisy, return_features=True)

        # Losses (no diversity loss for ablation)
        denoise_loss = F.mse_loss(denoised, clean)
        seg_loss = F.cross_entropy(seg_logits, seg_mask, weight=seg_class_weights)
        boundary_loss = compute_boundary_loss(denoised, clean, seg_mask)

        loss = denoise_loss + lambda_seg * seg_loss + lambda_boundary * boundary_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        dice = compute_dice_score(seg_logits.detach(), seg_mask)
        gate_mean = features['noise_gate'].mean().item()

        total_denoise_loss += denoise_loss.item()
        total_seg_loss += seg_loss.item()
        total_boundary_loss += boundary_loss.item()
        total_loss += loss.item()
        total_dice += dice
        total_gate_mean += gate_mean
        num_batches += 1

        pbar.set_postfix({
            'L_den': f'{total_denoise_loss/num_batches:.4f}',
            'L_seg': f'{total_seg_loss/num_batches:.4f}',
            'Gate': f'{total_gate_mean/num_batches:.3f}',
            'Dice': f'{total_dice/num_batches:.4f}'
        })

    if num_batches == 0:
        return {'total_loss': float('nan'), 'dice': 0.0}

    return {
        'total_loss': total_loss / num_batches,
        'denoise_loss': total_denoise_loss / num_batches,
        'seg_loss': total_seg_loss / num_batches,
        'boundary_loss': total_boundary_loss / num_batches,
        'gate_mean': total_gate_mean / num_batches,
        'dice': total_dice / num_batches,
    }


def validate(model, loader, device, base_model):
    """Validate ablation model."""
    model.eval()
    base_model.eval()

    psnr_base_list = []
    psnr_ours_list = []
    ssim_base_list = []
    ssim_ours_list = []
    dice_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            seg_mask = batch['seg_mask'].to(device)

            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            denoised, seg_logits = model(noisy, return_features=False)

            for i in range(noisy.size(0)):
                c = clean[i, 0].cpu().numpy()
                b = base_out[i, 0].cpu().numpy()
                d = denoised[i, 0].cpu().numpy()

                psnr_base_list.append(compute_psnr(b, c))
                psnr_ours_list.append(compute_psnr(d, c))
                ssim_base_list.append(compute_ssim(b, c))
                ssim_ours_list.append(compute_ssim(d, c))

            dice_list.append(compute_dice_score(seg_logits, seg_mask))

    return {
        'psnr_base': np.mean(psnr_base_list),
        'psnr': np.mean(psnr_ours_list),
        'ssim_base': np.mean(ssim_base_list),
        'ssim': np.mean(ssim_ours_list),
        'dice': np.mean(dice_list),
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation: Multi-task WITHOUT layer-specific gates')
    parser.add_argument('--train_jsonl', default='seg_data/seg_train.jsonl')
    parser.add_argument('--val_jsonl', default='seg_data/seg_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_train', type=int, default=2000)
    parser.add_argument('--max_val', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_seg', type=float, default=1.0)
    parser.add_argument('--lambda_boundary', type=float, default=0.1)
    parser.add_argument('--backbone_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--segmenter_ckpt', default=None,
                       help='Pretrained segmenter (None=train from scratch)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='checkpoints/ablation_no_layer_gates')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("ABLATION STUDY: Multi-Task WITHOUT Layer-Specific Gates")
    print("="*70)
    print("\nThis ablation uses a GLOBAL noise gate instead of 5 layer-specific gates.")
    print("Purpose: Prove the contribution of layer-specific noise modeling.")
    print("="*70)

    # Data - use truly random crops, class weights handle imbalance
    train_ds = MultiTaskOCTDataset(
        args.train_jsonl, args.patch_size, args.max_train,
        random_crop=True, ensure_all_layers=False
    )
    val_ds = MultiTaskOCTDataset(
        args.val_jsonl, args.patch_size, args.max_val,
        random_crop=True, ensure_all_layers=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nData: {len(train_ds)} train, {len(val_ds)} val")

    # Compute class weights for balanced segmentation loss
    print("Computing segmentation class weights...")
    seg_class_weights = compute_class_weights_from_loader(train_loader, num_classes=5, device=args.device)
    print(f"Class weights: {[f'{w:.3f}' for w in seg_class_weights.tolist()]}")

    # Model (ablation version)
    model = MultiTaskDenoiserAblation(
        backbone_ckpt=args.backbone_ckpt,
        segmenter_ckpt=args.segmenter_ckpt
    ).to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Baseline for comparison
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)
    ckpt = torch.load(args.backbone_ckpt, map_location=args.device, weights_only=False)
    base_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    del ckpt
    base_model.eval()

    # Optimizer
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    # Training loop
    best_psnr = 0
    best_dice = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"ABLATION EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, epoch,
            lambda_seg=args.lambda_seg, lambda_boundary=args.lambda_boundary,
            seg_class_weights=seg_class_weights
        )
        val_metrics = validate(model, val_loader, args.device, base_model)

        print(f"\nTrain:")
        print(f"  Denoise Loss:  {train_metrics['denoise_loss']:.4f}")
        print(f"  Seg Loss:      {train_metrics['seg_loss']:.4f}")
        print(f"  Boundary Loss: {train_metrics['boundary_loss']:.4f}")
        print(f"  Global Gate:   {train_metrics['gate_mean']:.4f} [same for all layers]")
        print(f"  Dice:          {train_metrics['dice']:.4f}")

        print(f"\nValidation:")
        print(f"  PSNR (base):  {val_metrics['psnr_base']:.2f} dB")
        print(f"  PSNR (ours):  {val_metrics['psnr']:.2f} dB")
        psnr_gain = val_metrics['psnr'] - val_metrics['psnr_base']
        print(f"  PSNR GAIN:    {'+' if psnr_gain >= 0 else ''}{psnr_gain:.2f} dB")
        print(f"  SSIM (base):  {val_metrics['ssim_base']:.4f}")
        print(f"  SSIM (ours):  {val_metrics['ssim']:.4f}")
        print(f"  Dice:         {val_metrics['dice']:.4f}")

        # Save best
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(args.output_dir, 'best_psnr.pth'))
            print(f"*** NEW BEST PSNR: {best_psnr:.2f} dB ***")

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best_dice.pth'))
            print(f"*** NEW BEST DICE: {best_dice:.4f} ***")

        # Memory cleanup
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("ABLATION TRAINING COMPLETE")
    print("="*70)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print("\nCompare with full method to measure layer-specific gate contribution.")
    print("="*70)


if __name__ == '__main__':
    main()
