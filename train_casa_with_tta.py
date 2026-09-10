"""
Train CASA with TTA-aware training.
Key: Simulate TTA during training so model learns to benefit from it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from adaptive_oct_denoise import (
    build_model, CleanOCTDataset, PairedOCTDataset, resize_to,
    SpeckleStatisticsLoss, device, set_seed, force_memory_cleanup
)

def tta_aware_loss(model, noisy, clean, speckle_loss_fn, tta_prob=0.3):
    """
    During training, randomly apply TTA-style adaptation to teach model.

    Args:
        tta_prob: Probability of applying TTA during training (0.3 = 30% of batches)
    """
    # Regular forward pass
    pred = model(noisy)
    base_loss = F.l1_loss(pred, clean)

    # With probability tta_prob, simulate TTA
    if torch.rand(1).item() < tta_prob and model.training:
        # TTA simulation: adapt on noisy, then denoise
        model.eval()  # Switch to eval for TTA
        with torch.no_grad():
            # Get base prediction
            base_pred = model(noisy)

        # Small adaptation step on self-supervised loss
        model.train()
        adapted_pred = model(noisy)

        # Self-supervised TTA losses
        tv_loss = (
            torch.mean(torch.abs(adapted_pred[:, :, :, 1:] - adapted_pred[:, :, :, :-1])) +
            torch.mean(torch.abs(adapted_pred[:, :, 1:, :] - adapted_pred[:, :, :-1, :]))
        )

        speckle_loss = speckle_loss_fn(adapted_pred, target=None, noisy=noisy)

        # Consistency with base prediction (not with noisy!)
        consistency_loss = F.l1_loss(adapted_pred, base_pred.detach())

        # Combined TTA loss (no anchor to noisy!)
        tta_loss = 0.1 * tv_loss + 0.1 * speckle_loss + 0.5 * consistency_loss

        # Total loss combines supervised + TTA
        total_loss = base_loss + 0.2 * tta_loss

        model.train()  # Back to train mode
        return total_loss, pred

    return base_loss, pred


def train_with_tta_awareness(
    output_dir='checkpoints/casa_tta_aware',
    num_meta_epochs=15,
    finetune_epochs=80,
    batch_size=2,
    base_channels=64,
    seed=42
):
    """Train CASA with TTA awareness."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("TRAINING CASA WITH TTA AWARENESS")
    print("="*80)
    print("This trains the model to benefit from TTA at test time")
    print("="*80)

    # Build model
    model = build_model(adapter_type='casa', base_channels=base_channels)
    model.to(device)

    tfm = resize_to((64, 64))
    speckle_loss_fn = SpeckleStatisticsLoss().to(device)

    # Meta-training phase (standard)
    if num_meta_epochs > 0:
        print("\n[Phase 1] Meta-learning...")
        from adaptive_oct_denoise import reptile_meta_train
        clean_ds = CleanOCTDataset('meta_clean/', transform=tfm)
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)

        reptile_meta_train(
            model,
            clean_loader=clean_loader,
            num_meta_epochs=num_meta_epochs,
            num_tasks_per_meta_batch=4,
            inner_steps=5,
            inner_lr=1e-4,
            meta_step_size=0.1,
            amp=True,
            meta_log_interval=10,
        )

        torch.save(model.state_dict(), os.path.join(output_dir, "meta_trained.pth"))
        print("[Phase 1] Meta-training complete!")

    # Fine-tuning with TTA awareness
    print("\n[Phase 2] Fine-tuning with TTA awareness...")
    paired_ds = PairedOCTDataset('train_pairs_universal.txt', transform=tfm)
    val_ds = PairedOCTDataset('val_pairs_universal.txt', transform=tfm)

    train_loader = DataLoader(paired_ds, batch_size=batch_size, shuffle=True,
                             num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size//2, shuffle=False,
                           num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": 5e-5},
        {"params": model.backbone.parameters(), "lr": 1e-5}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs*len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 12

    # EMA
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema_decay = 0.999

    for epoch in range(finetune_epochs):
        model.train()
        train_losses = []

        for step, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True):
                # Use TTA-aware loss
                loss, pred = tta_aware_loss(model, noisy, clean, speckle_loss_fn, tta_prob=0.3)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())

            # Update EMA
            with torch.no_grad():
                cur = model.state_dict()
                for k in ema_state:
                    ema_state[k].mul_(ema_decay).add_((1-ema_decay)*cur[k])

            if (step + 1) % 50 == 0:
                print(f"[Epoch {epoch+1}/{finetune_epochs}] Step {step+1}/{len(train_loader)} Loss={loss.item():.4f}", flush=True)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(noisy)
                    val_loss = F.l1_loss(pred, clean)
                val_losses.append(val_loss.item())

        val_loss = np.mean(val_losses)
        train_loss = np.mean(train_losses)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            torch.save(ema_state, os.path.join(output_dir, "best_model_ema.pth"))
            print(f"[Epoch {epoch+1}/{finetune_epochs}] Train={train_loss:.4f} Val={val_loss:.4f} ⭐ NEW BEST", flush=True)
        else:
            epochs_no_improve += 1
            print(f"[Epoch {epoch+1}/{finetune_epochs}] Train={train_loss:.4f} Val={val_loss:.4f} (No improve: {epochs_no_improve}/{patience})", flush=True)

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        force_memory_cleanup()

    # Save final model
    torch.save(ema_state, os.path.join(output_dir, "finetuned_ema.pth"))
    print(f"\n✅ Training complete! Models saved to {output_dir}/")
    print(f"   - best_model_ema.pth (best validation)")
    print(f"   - finetuned_ema.pth (final)")


if __name__ == "__main__":
    train_with_tta_awareness(
        output_dir='checkpoints/casa_tta_aware',
        num_meta_epochs=15,
        finetune_epochs=80,
        batch_size=2,
        base_channels=64,
        seed=42
    )
