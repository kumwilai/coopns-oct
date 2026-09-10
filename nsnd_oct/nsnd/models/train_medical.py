"""
Training functions optimized for medical imaging quality

Uses MedicalImageLoss to optimize BOTH PSNR and SSIM
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from nsnd.utils.medical_losses import MedicalImageLoss, MedicalMultiMetricLoss


def train_with_medical_loss(
    model,
    train_loader,
    device='cpu',
    epochs=50,
    lr=1e-3,
    ssim_weight=0.6,  # Higher for medical imaging!
    mse_weight=0.4
):
    """
    Train model with medical imaging loss (PSNR + SSIM)

    Args:
        model: Denoising model
        train_loader: Training data
        device: Device
        epochs: Number of epochs
        lr: Learning rate
        ssim_weight: Weight for SSIM (recommend 0.5-0.7 for medical)
        mse_weight: Weight for MSE/PSNR

    Returns:
        trained_model: Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Medical imaging loss (PSNR + SSIM)
    criterion = MedicalImageLoss(ssim_weight=ssim_weight, mse_weight=mse_weight)

    print("\n" + "="*70)
    print("TRAINING WITH MEDICAL IMAGE LOSS")
    print("="*70)
    print(f"Optimizing for BOTH PSNR and SSIM")
    print(f"  SSIM weight: {ssim_weight} (structure preservation)")
    print(f"  MSE weight:  {mse_weight} (pixel accuracy)")
    print("="*70)

    best_ssim = 0
    best_psnr = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        num_batches = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Forward
            result = model(noisy)
            output = result[0] if isinstance(result, tuple) else result

            # Medical loss (PSNR + SSIM)
            loss, metrics = criterion(output, clean)

            loss.backward()
            optimizer.step()

            epoch_loss += metrics['total_loss']
            epoch_psnr += metrics['psnr']
            epoch_ssim += metrics['ssim']
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        avg_ssim = epoch_ssim / num_batches

        # Track best
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  PSNR: {avg_psnr:.2f} dB (best: {best_psnr:.2f})")
            print(f"  SSIM: {avg_ssim:.4f}  (best: {best_ssim:.4f}) ← Medical Quality!")

    print("="*70)
    print("Training complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {best_ssim:.4f}")
    print("="*70)

    return model


def train_iterative_with_medical_loss(
    model,
    train_loader,
    device='cpu',
    epochs=50,
    lr=5e-4,
    ssim_weight=0.6,
    mse_weight=0.4
):
    """
    Train iterative refinement with medical loss

    Special handling for multi-stage models
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MedicalImageLoss(ssim_weight=ssim_weight, mse_weight=mse_weight)

    print("\n" + "="*70)
    print("TRAINING ITERATIVE MODEL WITH MEDICAL LOSS")
    print("="*70)
    print(f"SSIM weight: {ssim_weight} (medical quality)")
    print(f"MSE weight:  {mse_weight}")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        num_batches = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Forward (iterative model returns (output, intermediates))
            output, intermediates = model(noisy, clean=clean, return_intermediates=True)

            # Loss on final output
            loss_final, metrics_final = criterion(output, clean)

            # Intermediate supervision (if available)
            loss_intermediate = 0.0
            if 'stage_outputs' in intermediates:
                stage_outputs = intermediates['stage_outputs']
                for i, stage_out in enumerate(stage_outputs[:-1]):
                    weight = 0.2 * (0.5 ** i)
                    stage_loss, _ = criterion(stage_out, clean)
                    loss_intermediate += weight * stage_loss

            # Total loss
            loss = loss_final + loss_intermediate

            loss.backward()
            optimizer.step()

            epoch_loss += metrics_final['total_loss']
            epoch_psnr += metrics_final['psnr']
            epoch_ssim += metrics_final['ssim']
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        avg_ssim = epoch_ssim / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                  f"PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

    print("="*70)
    print("Iterative training complete!")
    print("="*70)

    return model
