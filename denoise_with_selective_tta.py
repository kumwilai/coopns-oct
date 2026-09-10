"""
Selective TTA for CASA+N2V: Only uses flips (no rotations)
to avoid CASA orientation issues and residual mode artifacts.
"""
import torch


@torch.no_grad()
def denoise_with_flip_tta(model, noisy: torch.Tensor) -> torch.Tensor:
    """
    Selective TTA using only horizontal/vertical flips (4x instead of 8x).
    Works better with orientation-specific features (CASA) and residual mode.

    Args:
        model: Denoising model
        noisy: Input noisy image [B, C, H, W]

    Returns:
        Denoised image [B, C, H, W]
    """
    model.eval()

    predictions = []

    # 1. Original
    predictions.append(model(noisy))

    # 2. Flip horizontal
    flip_h = torch.flip(noisy, dims=[3])
    pred_flip_h = model(flip_h)
    predictions.append(torch.flip(pred_flip_h, dims=[3]))

    # 3. Flip vertical
    flip_v = torch.flip(noisy, dims=[2])
    pred_flip_v = model(flip_v)
    predictions.append(torch.flip(pred_flip_v, dims=[2]))

    # 4. Flip both
    flip_hv = torch.flip(noisy, dims=[2, 3])
    pred_flip_hv = model(flip_hv)
    predictions.append(torch.flip(pred_flip_hv, dims=[2, 3]))

    # Average all 4 predictions
    return torch.stack(predictions).mean(dim=0)


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, '.')
    from adaptive_oct_denoise import build_model, PairedOCTDataset, resize_to, device
    from torch.utils.data import DataLoader
    import argparse
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_pairs", required=True)
    args = parser.parse_args()

    # Load model
    model = build_model(base_channels=48, residual_mode=True,
                        adapter_type="casa", backbone_type="noise2void")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    # Load data
    val_dataset = PairedOCTDataset(args.val_pairs, transform=resize_to((64, 64)))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("Evaluating with 4x Flip-Only TTA...")
    print(f"Total samples: {len(val_dataset)}, Batches: {len(val_loader)}")
    print("-" * 60)

    psnr_list, ssim_list = [], []

    for batch_idx, (noisy, clean) in enumerate(val_loader):
        noisy, clean = noisy.to(device), clean.to(device)
        pred = denoise_with_flip_tta(model, noisy)

        # Compute metrics per image
        batch_psnrs, batch_ssims = [], []
        for i in range(pred.shape[0]):
            p_np = pred[i, 0].cpu().numpy()
            c_np = clean[i, 0].cpu().numpy()
            psnr = peak_signal_noise_ratio(c_np, p_np, data_range=1.0)
            ssim = structural_similarity(c_np, p_np, data_range=1.0)
            batch_psnrs.append(psnr)
            batch_ssims.append(ssim)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        # Print progress
        samples_processed = min((batch_idx + 1) * 16, len(val_dataset))
        print(
            f"[Batch {batch_idx+1:>4}/{len(val_loader)} | {samples_processed}/{len(val_dataset)}] "
            f"Batch PSNR {np.mean(batch_psnrs):.2f} dB, SSIM {np.mean(batch_ssims):.4f} | "
            f"Running PSNR {np.mean(psnr_list):.2f} dB, SSIM {np.mean(ssim_list):.4f}",
            flush=True
        )

    print("\n" + "="*60)
    print("FLIP-ONLY TTA RESULTS")
    print("="*60)
    print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print("="*60)
