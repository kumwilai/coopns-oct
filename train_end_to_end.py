#!/usr/bin/env python3
"""
End-to-End Joint Training: Analyzer + Denoiser

KEY INNOVATION:
- Train noise analyzer jointly with denoiser (no separate pre-training)
- Gradients flow through entire pipeline
- Noise estimation optimized for denoising quality (not classification)
- More robust, fewer failure modes

CONTRIBUTION OVER BASELINE:
- Removes dependency on pre-trained analyzer
- Shows noise estimation can be learned implicitly
- Better adaptation through joint optimization
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "nsnd_oct"))

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer
from nsnd.models.noise_conditioner import SpatialBasisModulator
from nsnd.utils.metrics import compute_psnr, compute_ssim


class WeightedNoiseDataset(Dataset):
    """Dataset with ground truth noise weights for validation."""

    def __init__(self, jsonl_path, patch_size=64, max_samples=None):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        # Load images
        noisy = np.array(Image.open(data['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(data['clean_path']).convert('L'), dtype=np.float32) / 255.0

        # Center crop to patch size
        h, w = noisy.shape
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
        clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        # Convert to tensors
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()

        # Ground truth noise weights
        weights = torch.tensor([
            data['weights']['speckle'],
            data['weights']['banding'],
            data['weights']['gaussian'],
            data['weights']['shot']
        ], dtype=torch.float32)

        return {
            'noisy': noisy_t,
            'clean': clean_t,
            'weights': weights,
            'path': data['noisy_path']
        }


def usage_loss(model, input_img, spatial_map, basis, alpha, gate, margin=0.01):
    """
    Force model to produce different outputs for different noise conditioning.

    CRITICAL for end-to-end training:
    - Without this, analyzer can output random weights (model ignores them)
    - Forces gradients to flow back to analyzer
    """
    # 1. Forward with correct conditioning
    with torch.no_grad():
        out_correct = model(input_img, spatial_map=spatial_map, basis=basis, alpha=alpha, gate=gate)
        out_correct = out_correct.detach().clone()

    # 2. Forward with random conditioning (wrong noise estimate)
    perm_idx = torch.randperm(input_img.size(0), device=input_img.device)
    wrong_map = spatial_map[perm_idx] if spatial_map is not None else None
    wrong_gate = gate[perm_idx] if gate is not None else None

    out_wrong = model(input_img, spatial_map=wrong_map, basis=basis, alpha=alpha, gate=wrong_gate)

    # 3. Measure difference
    diff = (out_correct - out_wrong).abs().mean()
    loss = torch.relu(margin - diff)

    return loss, diff.detach()


def noise_classification_loss(predicted_weights, true_weights):
    """
    Auxiliary loss: Guide analyzer toward correct noise types.

    NOTE: This is OPTIONAL in end-to-end training.
    Main gradient signal comes from denoising quality.
    """
    return F.kl_div(
        predicted_weights.log(),
        true_weights,
        reduction='batchmean'
    )


def train_epoch(model, analyzer, modulator, train_loader, optimizer, args, epoch):
    """Single training epoch with end-to-end gradient flow."""

    model.train()
    analyzer.train()
    modulator.train()

    total_recon = 0
    total_usage = 0
    total_classify = 0
    total_delta = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy'].to(args.device)
        clean = batch['clean'].to(args.device)
        true_weights = batch['weights'].to(args.device)

        optimizer.zero_grad()

        # END-TO-END FORWARD PASS
        # 1. Analyzer predicts noise weights (TRAINABLE)
        weights_dict, features = analyzer(noisy, return_feature_map=True, return_predicates=False)
        feature_map = features.get("feature_map")

        predicted_weights = torch.stack([
            weights_dict["speckle"],
            weights_dict["banding"],
            weights_dict["gaussian"],
            weights_dict["shot"],
        ], dim=1)
        confidence = weights_dict.get("_confidence", torch.ones(noisy.size(0), device=args.device))

        # 2. Modulator creates spatial maps (TRAINABLE)
        spatial_map, gate, basis = modulator(
            feature_map,
            global_weights=predicted_weights,  # Use PREDICTED weights (not ground truth!)
            confidence=confidence,
            noisy=noisy,
        )

        # 3. Denoiser applies adaptive denoising (TRAINABLE)
        denoised = model(
            noisy,
            spatial_map=spatial_map,
            basis=basis,
            alpha=args.alpha,
            gate=gate
        )

        # LOSS COMPUTATION
        # 1. Reconstruction loss (primary)
        recon_loss = F.l1_loss(denoised, clean)

        # 2. Usage loss (forces adaptation)
        u_loss, delta = usage_loss(
            model, noisy, spatial_map, basis, args.alpha, gate, margin=args.base_delta_margin
        )

        # 3. Classification loss (auxiliary guidance)
        classify_loss = noise_classification_loss(predicted_weights, true_weights)

        # Total loss
        total_loss = (
            recon_loss +
            args.usage_loss_weight * u_loss +
            args.classify_loss_weight * classify_loss
        )

        # BACKPROP THROUGH ENTIRE PIPELINE
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(analyzer.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(modulator.parameters(), 1.0)

        optimizer.step()

        # Metrics
        total_recon += recon_loss.item()
        total_usage += u_loss.item()
        total_classify += classify_loss.item()
        total_delta += delta.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'Recon': f'{recon_loss.item():.4f}',
            'Usage': f'{u_loss.item():.4f}',
            'Class': f'{classify_loss.item():.4f}',
            'Δ': f'{delta.item():.4f}'
        })

        # Memory cleanup
        if batch_idx % 10 == 0:
            del total_loss, recon_loss, u_loss, classify_loss
            del spatial_map, basis, feature_map, denoised, predicted_weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        'recon': total_recon / num_batches,
        'usage': total_usage / num_batches,
        'classify': total_classify / num_batches,
        'delta': total_delta / num_batches,
    }


def validate(model, analyzer, modulator, val_loader, args, epoch):
    """Validation with ground truth comparison."""

    model.eval()
    analyzer.eval()
    modulator.eval()

    total_psnr_noisy = 0
    total_psnr_base = 0  # NEW: Base NAFNet without conditioning
    total_psnr_denoised = 0  # With end-to-end method
    total_ssim_base = 0
    total_ssim = 0
    correct_top1 = 0
    total_samples = 0

    # Per-class accuracy tracking
    noise_types = ['speckle', 'banding', 'gaussian', 'shot']
    per_class_correct = {nt: 0 for nt in noise_types}
    per_class_total = {nt: 0 for nt in noise_types}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            noisy = batch['noisy'].to(args.device)
            clean = batch['clean'].to(args.device)
            true_weights = batch['weights'].to(args.device)

            # ============================================================
            # 1. BASE NAFNET (no conditioning - baseline comparison)
            # ============================================================
            base_denoised = model(noisy, spatial_map=None, basis=None, alpha=0.0, gate=None)

            # Predict noise weights
            weights_dict, features = analyzer(noisy, return_feature_map=True, return_predicates=False)
            feature_map = features.get("feature_map")

            predicted_weights = torch.stack([
                weights_dict["speckle"],
                weights_dict["banding"],
                weights_dict["gaussian"],
                weights_dict["shot"],
            ], dim=1)
            confidence = weights_dict.get("_confidence", torch.ones(noisy.size(0), device=args.device))

            # Create spatial maps
            spatial_map, gate, basis = modulator(
                feature_map,
                global_weights=predicted_weights,
                confidence=confidence,
                noisy=noisy,
            )

            # ============================================================
            # 2. END-TO-END METHOD (with analyzer conditioning)
            # ============================================================
            denoised = model(
                noisy,
                spatial_map=spatial_map,
                basis=basis,
                alpha=args.alpha,
                gate=gate
            )

            # Metrics
            for i in range(noisy.size(0)):
                psnr_noisy = compute_psnr(noisy[i:i+1], clean[i:i+1])
                psnr_base = compute_psnr(base_denoised[i:i+1], clean[i:i+1])
                psnr_denoised = compute_psnr(denoised[i:i+1], clean[i:i+1])
                ssim_base = compute_ssim(base_denoised[i:i+1], clean[i:i+1])
                ssim = compute_ssim(denoised[i:i+1], clean[i:i+1])

                total_psnr_noisy += psnr_noisy
                total_psnr_base += psnr_base
                total_psnr_denoised += psnr_denoised
                total_ssim_base += ssim_base
                total_ssim += ssim

                # Top-1 accuracy (overall and per-class)
                pred_class = predicted_weights[i].argmax().item()
                true_class = true_weights[i].argmax().item()

                true_noise_type = noise_types[true_class]
                per_class_total[true_noise_type] += 1

                if pred_class == true_class:
                    correct_top1 += 1
                    per_class_correct[true_noise_type] += 1

                total_samples += 1

            # Memory cleanup
            del noisy, clean, base_denoised, denoised, spatial_map, basis, feature_map, predicted_weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_psnr_base = total_psnr_base / total_samples
    avg_psnr_denoised = total_psnr_denoised / total_samples
    avg_ssim_base = total_ssim_base / total_samples
    avg_ssim = total_ssim / total_samples

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for nt in noise_types:
        if per_class_total[nt] > 0:
            per_class_accuracy[nt] = 100.0 * per_class_correct[nt] / per_class_total[nt]
        else:
            per_class_accuracy[nt] = 0.0

    return {
        'psnr_noisy': total_psnr_noisy / total_samples,
        'psnr_base': avg_psnr_base,  # NEW
        'psnr_denoised': avg_psnr_denoised,
        'ssim_base': avg_ssim_base,  # NEW
        'ssim': avg_ssim,
        'top1': 100.0 * correct_top1 / total_samples,
        'gain_over_noisy': avg_psnr_denoised - (total_psnr_noisy / total_samples),
        'gain_over_base': avg_psnr_denoised - avg_psnr_base,  # NEW: Key metric!
        'per_class_accuracy': per_class_accuracy,  # NEW: Per-class breakdown
        'per_class_total': per_class_total,  # NEW: Sample counts per class
    }


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_jsonl", type=str, default="weights_duke_analysis_maps_train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="weights_duke_analysis_maps_val.jsonl")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples for quick testing")

    # Model
    parser.add_argument("--base_ckpt", type=str, default="outputs/nafnet_analysis_maps_w64/nafnet_best.pth")
    parser.add_argument("--analyzer_init", type=str, default=None, help="Optional analyzer initialization")
    parser.add_argument("--alpha", type=float, default=2.0)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--usage_loss_weight", type=float, default=0.5)
    parser.add_argument("--classify_loss_weight", type=float, default=0.1)
    parser.add_argument("--base_delta_margin", type=float, default=0.01)

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="checkpoints/end_to_end")

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("END-TO-END JOINT TRAINING: Analyzer + Denoiser")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Usage loss weight: {args.usage_loss_weight}")
    print(f"  Classification loss weight: {args.classify_loss_weight}")
    print(f"  Alpha (modulation): {args.alpha}")
    print(f"\nKey Innovation:")
    print(f"  → Train analyzer JOINTLY with denoiser (no pre-training)")
    print(f"  → Gradients flow through entire pipeline")
    print(f"  → Noise estimation optimized for denoising quality")
    print("="*70)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = WeightedNoiseDataset(args.train_jsonl, args.patch_size, max_samples=args.max_train_samples)
    val_dataset = WeightedNoiseDataset(args.val_jsonl, args.patch_size, max_samples=50)  # Limit val for speed

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Initialize models
    print("\nInitializing models...")

    # 1. NAFNet (initialize from pre-trained)
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
    ).to(args.device)

    # Load base weights
    base_state = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(base_state, strict=False)
    print(f"  ✓ Loaded base NAFNet from {args.base_ckpt}")

    # 2. Analyzer (train from scratch or optional initialization)
    analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=False).to(args.device)
    if args.analyzer_init:
        analyzer_state = torch.load(args.analyzer_init, map_location=args.device, weights_only=False)
        analyzer.load_state_dict(analyzer_state["state_dict"], strict=False)
        print(f"  ✓ Initialized analyzer from {args.analyzer_init}")
    else:
        print(f"  ✓ Analyzer training from scratch (end-to-end)")

    # 3. Modulator (train from scratch)
    modulator = SpatialBasisModulator(
        feature_channels=128,
        stage_channels=model.dbm_stage_channels,
        num_noise_types=4,
        hidden_channels=64,
        alpha=args.alpha,
        gate_floor=0.0,
        basis_init_std=0.1,
    ).to(args.device)
    print(f"  ✓ Modulator initialized")

    # Optimizer (all parameters jointly)
    all_params = list(model.parameters()) + list(analyzer.parameters()) + list(modulator.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)

    print(f"\nTotal parameters: {sum(p.numel() for p in all_params):,}")

    # Training loop
    best_psnr = 0
    best_top1 = 0
    best_gain_over_base = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        train_metrics = train_epoch(model, analyzer, modulator, train_loader, optimizer, args, epoch)
        print(f"\nTraining Metrics:")
        print(f"  Recon Loss: {train_metrics['recon']:.4f}")
        print(f"  Usage Loss: {train_metrics['usage']:.4f}")
        print(f"  Classify Loss: {train_metrics['classify']:.4f}")
        print(f"  Delta: {train_metrics['delta']:.4f}")

        # Validate
        val_metrics = validate(model, analyzer, modulator, val_loader, args, epoch)
        print(f"\nValidation Metrics:")
        print(f"  PSNR (noisy):      {val_metrics['psnr_noisy']:.2f} dB")
        print(f"  PSNR (base):       {val_metrics['psnr_base']:.2f} dB  ← baseline NAFNet")
        print(f"  PSNR (end-to-end): {val_metrics['psnr_denoised']:.2f} dB  ← your method")
        print(f"")
        print(f"  📊 Gain over noisy: +{val_metrics['gain_over_noisy']:.2f} dB")
        print(f"  🎯 Gain over base:  +{val_metrics['gain_over_base']:.2f} dB  ← KEY METRIC")
        print(f"")
        print(f"  SSIM (base):       {val_metrics['ssim_base']:.4f}")
        print(f"  SSIM (end-to-end): {val_metrics['ssim']:.4f}  (Δ: +{val_metrics['ssim'] - val_metrics['ssim_base']:.4f})")
        print(f"")
        print(f"  Top-1 Accuracy:    {val_metrics['top1']:.1f}%")
        print(f"  Per-class Accuracy:")
        for noise_type in ['speckle', 'banding', 'gaussian', 'shot']:
            acc = val_metrics['per_class_accuracy'][noise_type]
            count = val_metrics['per_class_total'][noise_type]
            print(f"    {noise_type:8s}: {acc:5.1f}%  (n={count})")

        # Save best models
        if val_metrics['psnr_denoised'] > best_psnr:
            best_psnr = val_metrics['psnr_denoised']
            best_top1 = val_metrics['top1']
            best_gain_over_base = val_metrics['gain_over_base']

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'analyzer': analyzer.state_dict(),
                'modulator': modulator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': best_psnr,
                'psnr_base': val_metrics['psnr_base'],
                'top1': best_top1,
                'gain_over_noisy': val_metrics['gain_over_noisy'],
                'gain_over_base': val_metrics['gain_over_base'],
            }

            save_path = Path(args.output_dir) / "best_model.pth"
            torch.save(checkpoint, save_path)
            print(f"\n  ✅ Best model saved! PSNR: {best_psnr:.2f} dB (+{val_metrics['gain_over_base']:.2f} dB over base)")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best PSNR (end-to-end): {best_psnr:.2f} dB")
    print(f"Best Gain over Base:    +{best_gain_over_base:.2f} dB  🎯")
    print(f"Best Top-1 Accuracy:    {best_top1:.1f}%")
    print(f"Checkpoint:             {args.output_dir}/best_model.pth")
    print("="*70)


if __name__ == "__main__":
    main()
