"""
Evaluate ensemble of multiple CASA checkpoints.
Often beats single models by 0.3-0.5 dB.
"""
import torch
from torch.utils.data import DataLoader
from adaptive_oct_denoise import (
    build_model, PairedOCTDataset, resize_to,
    compute_psnr, compute_ssim, device
)
import argparse
import numpy as np

ENSEMBLE_CHECKPOINTS = [
    'checkpoints/universal_casa/finetuned.pth',
    'checkpoints/casa_finetune64_phys/finetuned_ema.pth',
    'checkpoints/casa_finetune64/finetuned_ema.pth',
]

def evaluate_ensemble(checkpoints, val_pairs, adapter='casa', image_size=64):
    print(f"Loading {len(checkpoints)} checkpoints for ensemble...")

    # Load all models
    models = []
    for i, ckpt_path in enumerate(checkpoints):
        print(f"  [{i+1}/{len(checkpoints)}] Loading {ckpt_path}")
        model = build_model(adapter_type=adapter, base_channels=64)

        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device).eval()
        models.append(model)

    # Load validation data
    transform = resize_to((image_size, image_size))
    val_dataset = PairedOCTDataset(val_pairs, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"\nEvaluating ensemble on {len(val_dataset)} validation pairs...")
    print("-" * 60)

    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(val_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Average predictions from all models
            preds = []
            for model in models:
                pred = model(noisy)
                preds.append(pred)

            # Ensemble: average all predictions
            ensemble_pred = torch.mean(torch.stack(preds, dim=0), dim=0)

            psnr = compute_psnr(ensemble_pred, clean)
            ssim = compute_ssim(ensemble_pred, clean)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                samples = min((batch_idx + 1) * 16, len(val_dataset))
                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)
                print(f"[{samples}/{len(val_dataset)}] PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}", flush=True)

    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print("="*60)

    print(f"\n📊 vs SwinIR (28.84 dB, 0.8577 SSIM):")
    psnr_diff = np.mean(psnr_list) - 28.84
    ssim_diff = np.mean(ssim_list) - 0.8577
    print(f"   PSNR: {'+' if psnr_diff >= 0 else ''}{psnr_diff:.2f} dB")
    print(f"   SSIM: {'+' if ssim_diff >= 0 else ''}{ssim_diff:.4f}")

    if psnr_diff > 0:
        print("   ✅ ENSEMBLE BEATS SwinIR ON PSNR!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_pairs", type=str, default="val_pairs_universal.txt")
    parser.add_argument("--adapter", type=str, default="casa")
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    evaluate_ensemble(ENSEMBLE_CHECKPOINTS, args.val_pairs, args.adapter, args.image_size)
