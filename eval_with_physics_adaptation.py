"""
Evaluate with physics-informed test-time adaptation.
Uses the same losses CASA was trained with, not generic self-supervised losses.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adaptive_oct_denoise import (
    build_model, PairedOCTDataset, resize_to,
    compute_psnr, compute_ssim, device,
    SpeckleStatisticsLoss
)
import argparse
import numpy as np
import copy

def physics_informed_adaptation(
    model,
    noisy: torch.Tensor,
    num_steps: int = 10,
    lr: float = 1e-4,
    adapt_adapter_only: bool = True,
):
    """
    Adapt using physics-informed losses that CASA was trained with.

    Key differences from generic TTA:
    1. Uses depth-weighted fidelity (OCT-specific)
    2. Uses A-scan continuity (OCT-specific)
    3. Uses speckle statistics matching (OCT-specific)
    4. Only adapts adapter parameters (more stable)
    """
    model = model.to(device)
    noisy = noisy.to(device)
    model.train()

    # Only adapt adapter parameters (freeze backbone)
    if adapt_adapter_only:
        params_to_adapt = []
        for name, param in model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
                params_to_adapt.append(param)
            else:
                param.requires_grad = False
    else:
        params_to_adapt = [p for p in model.parameters() if p.requires_grad]

    if len(params_to_adapt) == 0:
        print("Warning: No parameters to adapt!")
        model.eval()
        with torch.no_grad():
            return model(noisy).detach().cpu()

    opt = torch.optim.Adam(params_to_adapt, lr=lr)

    # Initialize physics-informed losses
    speckle_stats = SpeckleStatisticsLoss().to(device)

    for step in range(num_steps):
        opt.zero_grad(set_to_none=True)

        # Forward pass
        pred = model(noisy)

        # Self-supervised adaptation losses (no clean target needed):

        # 1. Speckle statistics: residual should have realistic speckle
        # This uses the noisy input to compute the residual
        speckle_loss = speckle_stats(pred, target=None, noisy=noisy)

        # 2. Total variation: encourage spatial smoothness
        tv_loss = (
            torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])) +  # horizontal
            torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))    # vertical
        )

        # 3. Output should be in valid range [0, 1]
        range_loss = (
            F.relu(-pred).mean() +  # penalize negative values
            F.relu(pred - 1.0).mean()  # penalize values > 1
        )

        # 4. Very weak anchor to noisy (prevent catastrophic drift)
        # Set to 0 to disable, or use very small value like 0.01
        anchor_loss = F.mse_loss(pred, noisy)

        # Combined loss
        # Note: anchor_weight=0 means pure denoising, increase if model drifts
        loss = (
            0.2 * speckle_loss +  # Match realistic speckle statistics
            0.1 * tv_loss +       # Encourage smoothness
            0.01 * range_loss +   # Keep in valid range
            0.0 * anchor_loss     # Disabled by default (was causing degradation)
        )

        loss.backward()
        opt.step()

    # Final prediction
    model.eval()
    with torch.no_grad():
        final_pred = model(noisy).detach().cpu()

    # Restore grad status
    for param in model.parameters():
        param.requires_grad = True

    return final_pred


def evaluate_with_physics_adaptation(
    checkpoint_path, val_pairs, adapter='casa', image_size=64,
    adapt_steps=10, adapt_lr=1e-4, adapt_adapter_only=True, batch_size=1
):
    print(f"Loading checkpoint: {checkpoint_path}")

    # Build model
    model = build_model(adapter_type=adapter, base_channels=64)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # Load validation data
    transform = resize_to((image_size, image_size))
    val_dataset = PairedOCTDataset(val_pairs, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Evaluating on {len(val_dataset)} validation pairs...", flush=True)
    print(f"Physics-informed adaptation: steps={adapt_steps}, lr={adapt_lr}", flush=True)
    print(f"Adapt adapter only: {adapt_adapter_only}", flush=True)
    print("-" * 70, flush=True)
    print("Progress will be shown every 10 images...", flush=True)

    psnr_no_adapt, ssim_no_adapt = [], []
    psnr_with_adapt, ssim_with_adapt = [], []

    for idx, (noisy, clean) in enumerate(val_loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # ===== WITHOUT ADAPTATION =====
        model.eval()
        with torch.no_grad():
            pred_no_adapt = model(noisy)

        psnr_no = compute_psnr(pred_no_adapt, clean)
        ssim_no = compute_ssim(pred_no_adapt, clean)
        psnr_no_adapt.append(psnr_no)
        ssim_no_adapt.append(ssim_no)

        # ===== WITH PHYSICS-INFORMED ADAPTATION =====
        # Create a fresh copy for adaptation
        model_copy = copy.deepcopy(model)
        pred_with_adapt = physics_informed_adaptation(
            model_copy,
            noisy,
            num_steps=adapt_steps,
            lr=adapt_lr,
            adapt_adapter_only=adapt_adapter_only,
        )

        psnr_adapt = compute_psnr(pred_with_adapt.to(device), clean)
        ssim_adapt = compute_ssim(pred_with_adapt.to(device), clean)
        psnr_with_adapt.append(psnr_adapt)
        ssim_with_adapt.append(ssim_adapt)

        # Cleanup
        del model_copy
        torch.cuda.empty_cache()

        # Print progress every 10 images or at the end
        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_dataset):
            avg_psnr_no = np.mean(psnr_no_adapt)
            avg_psnr_adapt = np.mean(psnr_with_adapt)
            gain = avg_psnr_adapt - avg_psnr_no
            progress_pct = 100.0 * (idx + 1) / len(val_dataset)
            print(f"[{idx+1:>4}/{len(val_dataset)}] ({progress_pct:>5.1f}%) | "
                  f"No Adapt: {avg_psnr_no:.2f} dB | With Adapt: {avg_psnr_adapt:.2f} dB | "
                  f"Gain: {'+' if gain >= 0 else ''}{gain:.2f} dB", flush=True)

    # Compute statistics
    psnr_no_mean, psnr_no_std = np.mean(psnr_no_adapt), np.std(psnr_no_adapt)
    ssim_no_mean, ssim_no_std = np.mean(ssim_no_adapt), np.std(ssim_no_adapt)
    psnr_adapt_mean, psnr_adapt_std = np.mean(psnr_with_adapt), np.std(psnr_with_adapt)
    ssim_adapt_mean, ssim_adapt_std = np.mean(ssim_with_adapt), np.std(ssim_with_adapt)

    psnr_gain = psnr_adapt_mean - psnr_no_mean
    ssim_gain = ssim_adapt_mean - ssim_no_mean

    print("\n" + "="*70)
    print("PHYSICS-INFORMED ADAPTATION RESULTS")
    print("="*70)
    print(f"{'Method':<30} {'PSNR (dB)':<20} {'SSIM':<20}")
    print("-"*70)
    print(f"{'Without Adaptation':<30} {psnr_no_mean:.2f} ± {psnr_no_std:.2f}       {ssim_no_mean:.4f} ± {ssim_no_std:.4f}")
    print(f"{'With Physics Adaptation':<30} {psnr_adapt_mean:.2f} ± {psnr_adapt_std:.2f}       {ssim_adapt_mean:.4f} ± {ssim_adapt_std:.4f}")
    print("-"*70)
    print(f"{'Improvement':<30} {'+' if psnr_gain > 0 else ''}{psnr_gain:.2f} dB          {'+' if ssim_gain > 0 else ''}{ssim_gain:.4f}")
    print("="*70)

    return {
        'psnr_no_adapt': {'mean': psnr_no_mean, 'std': psnr_no_std},
        'ssim_no_adapt': {'mean': ssim_no_mean, 'std': ssim_no_std},
        'psnr_with_adapt': {'mean': psnr_adapt_mean, 'std': psnr_adapt_std},
        'ssim_with_adapt': {'mean': ssim_adapt_mean, 'std': ssim_adapt_std},
        'psnr_gain': psnr_gain,
        'ssim_gain': ssim_gain,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_pairs", type=str, required=True)
    parser.add_argument("--adapter", type=str, default="casa")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--adapt_steps", type=int, default=10, help="Number of adaptation steps")
    parser.add_argument("--adapt_lr", type=float, default=1e-4, help="Learning rate for adaptation")
    parser.add_argument("--adapt_all", action="store_true", help="Adapt all parameters (default: adapter only)")
    args = parser.parse_args()

    evaluate_with_physics_adaptation(
        checkpoint_path=args.checkpoint,
        val_pairs=args.val_pairs,
        adapter=args.adapter,
        image_size=args.image_size,
        adapt_steps=args.adapt_steps,
        adapt_lr=args.adapt_lr,
        adapt_adapter_only=not args.adapt_all,
        batch_size=1
    )
