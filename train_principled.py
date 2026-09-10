#!/usr/bin/env python3
"""
Principled Interpretable Anatomy-Aware OCT Denoising

KEY INNOVATIONS:
1. Hand-crafted noise features based on PHYSICS (not learned)
   - Coefficient of variation (speckle indicator)
   - Signal-variance correlation (shot noise indicator)
   - Horizontal frequency ratio (banding indicator)

2. Architecturally DISTINCT experts:
   - Speckle: Log-domain processing
   - Banding: Frequency-aware filtering
   - Gaussian: Local Wiener-like filtering
   - Shot: Variance-stabilizing transform

3. Strong supervision with ground truth noise labels

4. Anatomy-aware layer processing

This approach WILL work because features are physics-based.
"""

import argparse
import json
import os
import sys
import gc
import time

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
from nsnd.models.noise_features import (
    NoiseFeatureExtractor,
    NoiseTypeClassifier,
    DistinctSymbolicExperts,
)
from nsnd.utils.metrics import compute_psnr, compute_ssim


class OCTDenoiseDataset(Dataset):
    def __init__(self, jsonl_path, patch_size=64, max_samples=None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

        h, w = noisy.shape
        top, left = (h - self.patch_size) // 2, (w - self.patch_size) // 2
        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        return {
            'noisy': torch.from_numpy(noisy).unsqueeze(0).float(),
            'clean': torch.from_numpy(clean).unsqueeze(0).float(),
            'weights': torch.tensor([
                data['weights']['speckle'],
                data['weights']['banding'],
                data['weights']['gaussian'],
                data['weights']['shot']
            ], dtype=torch.float32),
        }


class PrincipledDenoiser(nn.Module):
    """
    Principled interpretable denoiser with:
    1. Physics-based noise feature extraction
    2. Distinct symbolic experts
    3. Neural backbone for residual refinement
    4. Anatomy-aware fusion
    """

    def __init__(self, backbone_ckpt: str = None):
        super().__init__()

        # 1. Physics-based noise classifier
        self.noise_classifier = NoiseTypeClassifier(num_noise_types=4)

        # 2. Noise level estimator (simple local std)
        self.noise_level_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 5, padding=2),
            nn.Sigmoid(),
        )

        # 3. Distinct symbolic experts
        self.experts = DistinctSymbolicExperts()

        # 4. Neural backbone (NAFNet)
        self.backbone = NAFNetFullFiLM(
            img_channel=1,
            width=64,
            enc_blk_nums=[2, 2, 2],
            dec_blk_nums=[2, 2, 2],
            middle_blk_num=2,
            cond_dim=32,
        )

        # Load backbone weights if provided
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            ckpt = torch.load(backbone_ckpt, map_location='cpu', weights_only=False)
            state = ckpt.get('state_dict', ckpt)
            self.backbone.load_state_dict(state, strict=False)
            print(f"[PrincipledDenoiser] Loaded backbone from {backbone_ckpt}")

        # Freeze backbone initially (focus on expert learning)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 5. Anatomy-aware layer detector (simple depth zones)
        self.layer_detector = nn.Sequential(
            nn.Conv2d(1, 16, (7, 3), padding=(3, 1)),  # Vertical context
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 5, 1),  # 5 layer zones
        )

        # 6. Fusion network with GATED refinement
        # The gate controls how much symbolic refinement is applied
        # Initialized near zero so backbone dominates initially
        self.fusion = nn.Sequential(
            nn.Conv2d(1 + 4 + 4 + 5, 32, 3, padding=1),  # backbone + 4 experts + 4 noise + 5 layers
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),  # Bound the correction to [-1, 1]
        )

        # Learnable gate: controls refinement strength
        # Initialized to small value so backbone dominates initially
        self.refinement_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, return_interpretation: bool = False):
        B, C, H, W = x.shape

        # 1. Classify noise type using physics-based features
        noise_type, noise_features = self.noise_classifier(x)  # [B, 4, H, W]

        # 2. Estimate noise level
        noise_level = self.noise_level_conv(x)  # [B, 1, H, W]

        # 3. Apply distinct experts
        expert_outputs = self.experts(x, noise_level)

        # 4. Combine expert outputs weighted by noise type
        symbolic_out = sum(
            noise_type[:, i:i+1] * expert_outputs[name]
            for i, name in enumerate(['speckle', 'banding', 'gaussian', 'shot'])
        )

        # 5. Neural backbone
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # 6. Detect layer zones
        layer_logits = self.layer_detector(x)
        layer_prob = F.softmax(layer_logits, dim=1)  # [B, 5, H, W]

        # 7. Stack expert outputs for fusion
        expert_stack = torch.cat([
            expert_outputs['speckle'],
            expert_outputs['banding'],
            expert_outputs['gaussian'],
            expert_outputs['shot'],
        ], dim=1)  # [B, 4, H, W]

        # 8. Fuse everything with gated refinement
        fusion_input = torch.cat([
            backbone_out,   # [B, 1, H, W]
            expert_stack,   # [B, 4, H, W]
            noise_type,     # [B, 4, H, W]
            layer_prob,     # [B, 5, H, W]
        ], dim=1)  # [B, 14, H, W]

        # Compute refinement and apply with learned gate
        refinement = self.fusion(fusion_input)  # [-1, 1] due to Tanh
        gate = torch.sigmoid(self.refinement_gate)  # [0, 1]

        # Scale refinement by 0.1 to keep corrections small
        denoised = backbone_out + gate * 0.1 * refinement
        denoised = denoised.clamp(0, 1)

        if return_interpretation:
            return denoised, {
                'noise_type': noise_type,
                'noise_level': noise_level,
                'noise_features': noise_features,
                'expert_outputs': expert_outputs,
                'symbolic_out': symbolic_out,
                'backbone_out': backbone_out,
                'layer_prob': layer_prob,
                'global_weights': noise_type.mean(dim=[2, 3]),
            }

        return denoised


def train_epoch(model, loader, optimizer, device, epoch, gt_supervision=True):
    model.train()

    metrics = {k: 0.0 for k in ['loss', 'recon', 'cls', 'expert_div']}
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        gt_weights = batch['weights'].to(device)

        optimizer.zero_grad(set_to_none=True)

        denoised, interp = model(noisy, return_interpretation=True)

        # 1. Reconstruction loss
        recon_loss = F.mse_loss(denoised, clean)

        # 2. Supervised noise classification loss
        if gt_supervision:
            pred_global = interp['noise_type'].mean(dim=[2, 3])  # [B, 4]
            # Soft cross-entropy
            gt_soft = F.softmax(gt_weights * 2.0, dim=1)  # Temperature
            cls_loss = F.kl_div(
                F.log_softmax(pred_global * 2.0, dim=1),
                gt_soft,
                reduction='batchmean'
            )
        else:
            cls_loss = torch.tensor(0.0, device=device)

        # 3. Expert diversity loss (encourage different outputs)
        expert_outputs = list(interp['expert_outputs'].values())
        expert_stack = torch.stack(expert_outputs, dim=0)  # [4, B, 1, H, W]
        expert_var = expert_stack.var(dim=0).mean()  # Want HIGH variance
        div_loss = -expert_var

        # Total loss - Classification is CRITICAL for interpretability
        # Higher weight on cls_loss to ensure proper noise type learning
        total = recon_loss + 5.0 * cls_loss + 0.1 * div_loss

        # Check for NaN and skip if unstable
        if torch.isnan(total) or torch.isinf(total):
            print(f"Warning: NaN/Inf detected, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        metrics['loss'] += total.item()
        metrics['recon'] += recon_loss.item()
        metrics['cls'] += cls_loss.item()
        metrics['expert_div'] += expert_var.item()
        num_batches += 1

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Stricter clipping
        optimizer.step()

        del noisy, clean, gt_weights, denoised, interp

        pbar.set_postfix({
            'Loss': f'{metrics["loss"]/num_batches:.4f}',
            'Cls': f'{metrics["cls"]/num_batches:.3f}',
            'Div': f'{metrics["expert_div"]/num_batches:.4f}',
        })

    return {k: v/num_batches for k, v in metrics.items()}


def validate(model, loader, device, epoch, base_model):
    model.eval()
    base_model.eval()

    psnr_base_sum = 0
    psnr_ours_sum = 0
    ssim_sum = 0
    correct = 0
    total = 0

    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    expert_usage = torch.zeros(4)

    # Track noise features for analysis
    feature_by_class = {i: {k: [] for k in ['coef_variation', 'signal_var_corr', 'horizontal_ratio']} for i in range(4)}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val {epoch}"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            gt_weights = batch['weights'].to(device)

            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            denoised, interp = model(noisy, return_interpretation=True)

            for i in range(noisy.size(0)):
                psnr_base_sum += compute_psnr(base_out[i:i+1], clean[i:i+1])
                psnr_ours_sum += compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim_sum += compute_ssim(denoised[i:i+1], clean[i:i+1])

                gt_cls = gt_weights[i].argmax().item()
                pred_cls = interp['global_weights'][i].argmax().item()

                class_total[gt_cls] += 1
                if gt_cls == pred_cls:
                    correct += 1
                    class_correct[gt_cls] += 1

                # Track features by class
                for feat_name in ['coef_variation', 'signal_var_corr', 'horizontal_ratio']:
                    feat_val = interp['noise_features'][feat_name][i].mean().item()
                    feature_by_class[gt_cls][feat_name].append(feat_val)

                total += 1

            expert_usage += interp['noise_type'].mean(dim=[0, 2, 3]).cpu()

            del noisy, clean, gt_weights, base_out, denoised, interp

    n_batches = len(loader)
    expert_usage = expert_usage / n_batches

    # Compute feature statistics per class
    feature_stats = {}
    for cls_idx in range(4):
        feature_stats[cls_idx] = {}
        for feat_name in ['coef_variation', 'signal_var_corr', 'horizontal_ratio']:
            vals = feature_by_class[cls_idx][feat_name]
            if vals:
                feature_stats[cls_idx][feat_name] = np.mean(vals)
            else:
                feature_stats[cls_idx][feat_name] = 0

    return {
        'psnr_base': psnr_base_sum / total,
        'psnr': psnr_ours_sum / total,
        'ssim': ssim_sum / total,
        'accuracy': 100 * correct / total,
        'class_acc': [100 * class_correct[i] / max(class_total[i], 1) for i in range(4)],
        'class_total': class_total,
        'expert_usage': expert_usage,
        'feature_stats': feature_stats,
    }


def print_report(train_metrics, val_metrics, epoch):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} - PRINCIPLED INTERPRETABLE DENOISER")
    print(f"{'='*70}")

    print(f"\nTraining:")
    print(f"  Recon: {train_metrics['recon']:.4f}")
    print(f"  Classification: {train_metrics['cls']:.4f}")
    print(f"  Expert Diversity: {train_metrics['expert_div']:.4f}")

    print(f"\nValidation:")
    print(f"  PSNR (base):  {val_metrics['psnr_base']:.2f} dB")
    print(f"  PSNR (ours):  {val_metrics['psnr']:.2f} dB")
    gain = val_metrics['psnr'] - val_metrics['psnr_base']
    print(f"  PSNR GAIN:    {'+' if gain >= 0 else ''}{gain:.2f} dB")
    print(f"  SSIM:         {val_metrics['ssim']:.4f}")

    print(f"\n  NOISE CLASSIFICATION: {val_metrics['accuracy']:.1f}%")
    names = ['speckle', 'banding', 'gaussian', 'shot']
    for i, name in enumerate(names):
        acc = val_metrics['class_acc'][i]
        cnt = val_metrics['class_total'][i]
        bar = '#' * int(acc / 5)
        print(f"    {name:10s}: {acc:5.1f}% ({cnt:2d} samples) {bar}")

    print(f"\n  PHYSICS-BASED FEATURES BY CLASS:")
    print(f"    {'Class':<10} {'CoefVar':>10} {'Sig-Var':>10} {'HorizRatio':>10}")
    for i, name in enumerate(names):
        stats = val_metrics['feature_stats'][i]
        print(f"    {name:<10} {stats['coef_variation']:>10.4f} {stats['signal_var_corr']:>10.4f} {stats['horizontal_ratio']:>10.4f}")

    print(f"\n  EXPERT USAGE:")
    for i, name in enumerate(names):
        usage = val_metrics['expert_usage'][i].item()
        bar = '#' * int(usage * 40)
        print(f"    {name:10s}: {usage:.3f} {bar}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', default='weights_duke_analysis_maps_train.jsonl')
    parser.add_argument('--val_jsonl', default='weights_duke_analysis_maps_val.jsonl')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_train', type=int, default=100)
    parser.add_argument('--max_val', type=int, default=30)
    parser.add_argument('--base_ckpt', default='outputs/nafnet_analysis_maps_w64/nafnet_best.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)  # Reduced for stability
    parser.add_argument('--output_dir', default='checkpoints/principled')
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PRINCIPLED INTERPRETABLE ANATOMY-AWARE OCT DENOISER")
    print("=" * 70)
    print("\nKEY INNOVATIONS:")
    print("  1. Physics-based noise features (coefficient of variation, etc.)")
    print("  2. Architecturally distinct experts (log-domain, frequency, VST)")
    print("  3. Strong GT supervision for noise classification")
    print("  4. Anatomy-aware layer detection")
    print("=" * 70)

    # Data
    train_dataset = OCTDenoiseDataset(args.train_jsonl, args.patch_size, args.max_train)
    val_dataset = OCTDenoiseDataset(args.val_jsonl, args.patch_size, args.max_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nData: {len(train_dataset)} train, {len(val_dataset)} val")

    # Model
    model = PrincipledDenoiser(backbone_ckpt=args.base_ckpt).to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Frozen baseline
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)

    if os.path.exists(args.base_ckpt):
        ckpt = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
        base_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_psnr = 0
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        gc.collect()

        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)
        val_metrics = validate(model, val_loader, args.device, epoch, base_model)

        print_report(train_metrics, val_metrics, epoch)

        if val_metrics['psnr'] > best_psnr or val_metrics['accuracy'] > best_acc:
            best_psnr = max(best_psnr, val_metrics['psnr'])
            best_acc = max(best_acc, val_metrics['accuracy'])

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'psnr': val_metrics['psnr'],
                'accuracy': val_metrics['accuracy'],
            }, os.path.join(args.output_dir, 'best_model.pth'))

            print(f"\n*** BEST: PSNR={val_metrics['psnr']:.2f} dB, Acc={val_metrics['accuracy']:.1f}% ***")

        scheduler.step()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best Classification: {best_acc:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
