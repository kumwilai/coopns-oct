#!/usr/bin/env python3
"""
Soft Conditioning Approach for Interpretable OCT Denoising

Key insight: Physics features can't reliably CLASSIFY noise types (54% accuracy),
but they can provide useful CONDITIONING information for adaptive denoising.

Instead of:
  Image → Classify → Route to Expert → Output

We do:
  Image → Extract Physics Features → Condition Refinement → Output

This is more robust and still interpretable:
- We can visualize which features are high/low in different regions
- We can see how the model responds to different feature values
- No hard routing that fails when classification fails
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


class SoftConditionedDenoiser(nn.Module):
    """
    Denoiser that uses physics features as soft conditioning.

    Architecture:
    1. Extract physics-based noise features (interpretable)
    2. Use features to CONDITION the refinement (not hard routing)
    3. Backbone provides strong baseline
    4. Feature-conditioned refinement corrects remaining errors

    This is interpretable because:
    - Physics features are hand-crafted and meaningful
    - We can visualize feature maps
    - We can see how output changes with features
    """

    def __init__(self, backbone_ckpt: str = None):
        super().__init__()

        # 1. Physics-based feature extractor (NOT learned)
        self.feature_extractor = NoiseFeatureExtractor()

        # 2. Neural backbone (NAFNet)
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
            print(f"[SoftConditionedDenoiser] Loaded backbone from {backbone_ckpt}")

        # Train backbone end-to-end for better integration
        # Use smaller LR for backbone via parameter groups
        for param in self.backbone.parameters():
            param.requires_grad = True

        # 3. Feature normalization (learned, to handle varying scales)
        # This learns to normalize the physics features to useful ranges
        self.feature_norm = nn.Sequential(
            nn.Conv2d(7, 16, 1),  # 7 physics features
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1),  # Compress to 8 channels
        )

        # 4. Anatomy-aware layer detector
        self.layer_detector = nn.Sequential(
            nn.Conv2d(1, 16, (7, 3), padding=(3, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 5, 1),
        )

        # 5. Feature-conditioned refinement network
        # Takes: backbone output (1) + normalized features (8) + layer info (5)
        self.refinement = nn.Sequential(
            nn.Conv2d(1 + 8 + 5, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),  # Bound refinement to [-1, 1]
        )

        # Learnable refinement strength (starts small)
        self.refinement_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B, C, H, W = x.shape

        # 1. Extract physics features
        raw_features = self.feature_extractor(x)

        # Stack features
        feature_stack = torch.cat([
            raw_features['coef_variation'],
            raw_features['local_std'],
            raw_features['signal_var_corr'],
            raw_features['horizontal_ratio'],
            raw_features['horizontal_lines'],
            raw_features['high_freq'],
            raw_features['local_range'],
        ], dim=1)  # [B, 7, H, W]

        # 2. Normalize features (learned normalization)
        norm_features = self.feature_norm(feature_stack)  # [B, 8, H, W]

        # 3. Get backbone output
        backbone_out = self.backbone(x, spatial_map=None, basis=None, alpha=0.0, gate=None)

        # 4. Detect layer zones
        layer_logits = self.layer_detector(x)
        layer_prob = F.softmax(layer_logits, dim=1)  # [B, 5, H, W]

        # 5. Compute feature-conditioned refinement
        refine_input = torch.cat([
            backbone_out,
            norm_features,
            layer_prob,
        ], dim=1)  # [B, 14, H, W]

        refinement = self.refinement(refine_input)  # [B, 1, H, W], in [-1, 1]

        # Apply scaled refinement
        scale = torch.sigmoid(self.refinement_scale) * 0.2  # Max 0.2
        denoised = backbone_out + scale * refinement
        denoised = denoised.clamp(0, 1)

        if return_features:
            return denoised, {
                'raw_features': raw_features,
                'norm_features': norm_features,
                'backbone_out': backbone_out,
                'layer_prob': layer_prob,
                'refinement': refinement,
                'scale': scale,
            }

        return denoised


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad(set_to_none=True)

        denoised = model(noisy)

        # Simple reconstruction loss
        loss = F.mse_loss(denoised, clean)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        total_loss += loss.item()
        num_batches += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        pbar.set_postfix({'Loss': f'{total_loss/num_batches:.4f}'})

    return total_loss / max(num_batches, 1)


def validate(model, loader, device, epoch, base_model):
    model.eval()
    base_model.eval()

    psnr_base_sum = 0
    psnr_ours_sum = 0
    ssim_sum = 0
    n_samples = 0

    # Track feature statistics for interpretability
    feature_stats = {
        'coef_var_mean': [],
        'coef_var_std': [],
        'sig_var_mean': [],
        'layer_diversity': [],
        'refinement_magnitude': [],
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val {epoch}"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            base_out = base_model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)
            denoised, features = model(noisy, return_features=True)

            for i in range(noisy.size(0)):
                psnr_base_sum += compute_psnr(base_out[i:i+1], clean[i:i+1])
                psnr_ours_sum += compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim_sum += compute_ssim(denoised[i:i+1], clean[i:i+1])
                n_samples += 1

            # Track interpretability metrics
            raw = features['raw_features']
            feature_stats['coef_var_mean'].append(raw['coef_variation'].mean().item())
            feature_stats['coef_var_std'].append(raw['coef_variation'].std().item())
            feature_stats['sig_var_mean'].append(raw['signal_var_corr'].mean().item())
            feature_stats['layer_diversity'].append(features['layer_prob'].std(dim=1).mean().item())
            feature_stats['refinement_magnitude'].append(features['refinement'].abs().mean().item())

    return {
        'psnr_base': psnr_base_sum / n_samples,
        'psnr': psnr_ours_sum / n_samples,
        'ssim': ssim_sum / n_samples,
        'feature_stats': {k: np.mean(v) for k, v in feature_stats.items()},
    }


def print_results(epoch, train_loss, val_metrics):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} - SOFT CONDITIONED DENOISER")
    print(f"{'='*70}")

    print(f"\nTraining Loss: {train_loss:.4f}")

    print(f"\nValidation:")
    print(f"  PSNR (base):  {val_metrics['psnr_base']:.2f} dB")
    print(f"  PSNR (ours):  {val_metrics['psnr']:.2f} dB")
    gain = val_metrics['psnr'] - val_metrics['psnr_base']
    print(f"  PSNR GAIN:    {'+' if gain >= 0 else ''}{gain:.2f} dB")
    print(f"  SSIM:         {val_metrics['ssim']:.4f}")

    print(f"\n  INTERPRETABILITY METRICS:")
    stats = val_metrics['feature_stats']
    print(f"    CoefVar (mean):        {stats['coef_var_mean']:.3f}")
    print(f"    CoefVar (spatial std): {stats['coef_var_std']:.3f} (higher = more per-pixel variation)")
    print(f"    Sig-Var correlation:   {stats['sig_var_mean']:.3f}")
    print(f"    Layer diversity:       {stats['layer_diversity']:.3f} (higher = more anatomy-aware)")
    print(f"    Refinement magnitude:  {stats['refinement_magnitude']:.4f}")

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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', default='checkpoints/soft_conditioned')
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SOFT CONDITIONED INTERPRETABLE OCT DENOISER")
    print("=" * 70)
    print("\nKEY APPROACH:")
    print("  - Physics features provide CONDITIONING (not classification)")
    print("  - More robust when features don't perfectly separate classes")
    print("  - Still interpretable: can visualize features and responses")
    print("=" * 70)

    # Data
    train_ds = OCTDenoiseDataset(args.train_jsonl, args.patch_size, args.max_train)
    val_ds = OCTDenoiseDataset(args.val_jsonl, args.patch_size, args.max_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nData: {len(train_ds)} train, {len(val_ds)} val")

    # Model
    model = SoftConditionedDenoiser(backbone_ckpt=args.base_ckpt).to(args.device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Base model for comparison
    base_model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
        middle_blk_num=2, cond_dim=32,
    ).to(args.device)
    ckpt = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
    base_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    del ckpt  # Free checkpoint memory
    base_model.eval()

    # Optimizer with different LR for backbone vs refinement
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    best_psnr = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, epoch)
        val_metrics = validate(model, val_loader, args.device, epoch, base_model)

        print_results(epoch, train_loss, val_metrics)

        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            print(f"*** NEW BEST: PSNR={best_psnr:.2f} dB ***")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(args.output_dir, 'best.pth'))

        # Memory cleanup
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
