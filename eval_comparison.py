#!/usr/bin/env python3
"""
Comparison experiments: NAFNet backbone + various post-processing methods.

Methods compared:
  1. NAFNet backbone only (baseline)
  2. NAFNet + CoopNS-OCT (ours)
  3. NAFNet + Histogram Equalization (HE)
  4. NAFNet + CLAHE
  5. NAFNet + CNR-aware fine-tuning
  6. NAFNet + Unsharp Masking (USM)
  7. NAFNet + Bilateral Filter + CLAHE (BF+CLAHE)

All methods use the same NAFNet backbone and PKU37 test set (173 images).
Metrics match train_v8_cooperative.py validate() exactly:
  - CNR: Otsu mask computed on BACKBONE output, reused for all methods
  - Sobel: zero-padding (mode='constant') to match F.conv2d(padding=1)
  - ENL: bottom 25% region, (mean/std)^2
  - EPI: normalized edge correlation
"""

import gc
import json
import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    BackboneWrapper,
    PKU37Dataset,
)


# ============================================================
# Metrics — matches train_v8_cooperative.py validate() exactly
# ============================================================

def otsu_threshold_np(img):
    """Compute Otsu threshold for a [0,1] image. Returns soft sigmoid mask.
    Matches otsu_tissue_mask() in train_v8_cooperative.py lines 75-106."""
    hist, bin_edges = np.histogram(img.ravel(), bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return np.ones_like(img)
    w0, sum0, best_var, best_t = 0.0, 0.0, -1.0, 0.5
    sum_total = (hist * bin_centers).sum()
    for i in range(len(hist)):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var = var
            best_t = bin_centers[i]
    # Soft mask via sigmoid with steepness 20 (same as train_v8_cooperative.py)
    return 1.0 / (1.0 + np.exp(-(img - best_t) * 20.0))


def sobel_y_np(img):
    """Sobel vertical gradient with ZERO padding (matches F.conv2d padding=1)."""
    from scipy import ndimage
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return ndimage.convolve(img, kernel, mode='constant', cval=0.0)


def sobel_x_np(img):
    """Sobel horizontal gradient with ZERO padding (matches F.conv2d padding=1)."""
    from scipy import ndimage
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    return ndimage.convolve(img, kernel, mode='constant', cval=0.0)


def compute_clinical_metrics_np(d, c, backbone_otsu_mask=None):
    """Compute clinical metrics matching train_v8_cooperative.py validate().

    CRITICAL: Uses backbone's Otsu mask (not the enhanced image's mask) for CNR,
    matching the paper's validate() which calls otsu_tissue_mask(backbone_out)
    and uses the same mask for both backbone and corrected CNR.

    Args:
        d: denoised/enhanced image [H, W] in [0, 1]
        c: clean reference [H, W] in [0, 1]
        backbone_otsu_mask: pre-computed Otsu soft mask from backbone output.
            If None, computes from d (WRONG for fair comparison — only for backbone-only).
    Returns:
        dict with psnr, cnr, tci, epi, bs, enl, snr
    """
    # PSNR
    mse = np.mean((d - c) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    h, w = d.shape

    # CNR: Use backbone's Otsu mask for ALL methods (matches paper's validate())
    if backbone_otsu_mask is not None:
        signal_mask = backbone_otsu_mask
    else:
        signal_mask = otsu_threshold_np(d)
    bg_mask = 1.0 - signal_mask
    signal_sum = max(signal_mask.sum(), 1.0)
    bg_sum = max(bg_mask.sum(), 1.0)
    signal_mean = (d * signal_mask).sum() / signal_sum
    bg_mean = (d * bg_mask).sum() / bg_sum
    bg_std = np.sqrt(((d - bg_mean) ** 2 * bg_mask).sum() / bg_sum + 1e-8)
    bg_std = max(bg_std, 1e-4)
    cnr = np.clip((signal_mean - bg_mean) / bg_std, -100.0, 100.0)

    # TCI: Sobel-Y gradient ratio (layer boundary sharpness)
    d_gy = np.abs(sobel_y_np(d))
    c_gy = np.abs(sobel_y_np(c))
    c_gy_mean = max(c_gy.mean(), 1e-4)
    tci = np.clip(d_gy.mean() / c_gy_mean, 0.0, 10.0)

    # EPI: normalized edge correlation
    d_gx = sobel_x_np(d)
    c_gx = sobel_x_np(c)
    d_edge_mag = np.sqrt(d_gx**2 + d_gy**2 + 1e-8)
    c_edge_mag = np.sqrt(c_gx**2 + c_gy**2 + 1e-8)
    c_flat = c_edge_mag.ravel()
    d_flat = d_edge_mag.ravel()
    c_std = max(c_flat.std(), 1e-4)
    d_std = max(d_flat.std(), 1e-4)
    c_norm = (c_flat - c_flat.mean()) / c_std
    d_norm = (d_flat - d_flat.mean()) / d_std
    epi = float((c_norm * d_norm).mean())

    # BS: peak Sobel-Y ratio
    c_gy_max = max(c_gy.max(), 1e-4)
    bs = np.clip(d_gy.max() / c_gy_max, 0.0, 10.0)

    # ENL: (mean/std)^2 in background region (bottom 25%)
    bg_region = d[h * 3 // 4:, :]
    bg_std_r = max(bg_region.std(), 1e-6)
    enl = min((bg_region.mean() / bg_std_r) ** 2, 1e6)

    # SNR: tissue mean / background noise std
    tissue_region = d[:h // 2, :]
    snr = min(tissue_region.mean() / bg_std_r, 1e4)

    return {'psnr': float(psnr), 'cnr': float(cnr), 'tci': float(tci),
            'epi': float(epi), 'bs': float(bs), 'enl': float(enl), 'snr': float(snr)}


# ============================================================
# Post-processing methods
# ============================================================

def apply_histogram_equalization(image_np):
    """Apply global histogram equalization to a [0,1] grayscale image."""
    img_uint8 = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    hist, bins = np.histogram(img_uint8.ravel(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype(np.uint8)
    result = cdf_final[img_uint8]
    return result.astype(np.float32) / 255.0


def apply_clahe(image_np, clip_limit=2.0, tile_size=8):
    """Apply CLAHE to a [0,1] grayscale image."""
    try:
        import cv2
        img_uint8 = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_size, tile_size))
        result = clahe.apply(img_uint8)
        return result.astype(np.float32) / 255.0
    except ImportError:
        return _clahe_numpy(image_np, clip_limit, tile_size)


def _clahe_numpy(image_np, clip_limit=2.0, n_tiles=8):
    """Pure numpy CLAHE implementation (no OpenCV dependency)."""
    img_uint8 = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    h, w = img_uint8.shape
    tile_h = h // n_tiles
    tile_w = w // n_tiles

    pad_h = tile_h * n_tiles - h
    pad_w = tile_w * n_tiles - w
    if pad_h > 0 or pad_w > 0:
        img_uint8 = np.pad(img_uint8, ((0, pad_h), (0, pad_w)), mode='reflect')

    h_pad, w_pad = img_uint8.shape
    result = np.zeros_like(img_uint8, dtype=np.float32)

    cdfs = np.zeros((n_tiles, n_tiles, 256), dtype=np.float32)
    for i in range(n_tiles):
        for j in range(n_tiles):
            tile = img_uint8[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            hist, _ = np.histogram(tile, 256, [0, 256])

            n_pixels = tile_h * tile_w
            clip_val = int(clip_limit * n_pixels / 256)
            excess = np.sum(np.maximum(hist - clip_val, 0))
            hist = np.minimum(hist, clip_val)
            hist += excess // 256

            cdf = hist.cumsum().astype(np.float32)
            cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
            denom = max(cdf[-1] - cdf_min, 1)
            cdfs[i, j] = (cdf - cdf_min) / denom * 255

    for i in range(n_tiles):
        for j in range(n_tiles):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            tile = img_uint8[y0:y1, x0:x1]
            ti = min(max(i, 0), n_tiles - 1)
            tj = min(max(j, 0), n_tiles - 1)
            mapped = cdfs[ti, tj][tile]
            result[y0:y1, x0:x1] = mapped

    result = result[:h, :w]
    return result / 255.0


def apply_unsharp_mask(image_np, sigma=2.0, alpha=0.5):
    """Apply unsharp masking: enhanced = image + alpha * (image - blur(image)).

    Standard clinical OCT enhancement used in Heidelberg Spectralis, Topcon, Zeiss.
    Directly targets EPI/BS by amplifying high-frequency edge structure.

    Args:
        image_np: [H, W] in [0, 1]
        sigma: Gaussian blur sigma (controls sharpening scale)
        alpha: sharpening strength (0.5 = moderate)
    Returns:
        enhanced image [H, W] in [0, 1]
    """
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(image_np, sigma=sigma)
    detail = image_np - blurred
    enhanced = image_np + alpha * detail
    return np.clip(enhanced, 0.0, 1.0)


def apply_bilateral_clahe(image_np, d=9, sigma_color=75, sigma_space=75,
                           clip_limit=2.0, tile_size=8):
    """Apply bilateral filter followed by CLAHE.

    Edge-preserving smoothing before contrast enhancement reduces speckle
    amplification in CLAHE's local histograms. Common OCT pipeline in
    clinical segmentation literature (Farsiu et al., Chiu et al.).

    Args:
        image_np: [H, W] in [0, 1]
        d: bilateral filter diameter
        sigma_color: color space sigma
        sigma_space: coordinate space sigma
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile grid size
    """
    try:
        import cv2
        img_uint8 = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        # Step 1: bilateral filter (edge-preserving smoothing)
        filtered = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
        # Step 2: CLAHE on filtered output
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_size, tile_size))
        result = clahe.apply(filtered)
        return result.astype(np.float32) / 255.0
    except ImportError:
        # Fallback: Gaussian approximation of bilateral + numpy CLAHE
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(image_np, sigma=1.5)
        # Blend: keep edges from original, smoothness from filtered
        edge_weight = np.abs(image_np - smoothed)
        edge_weight = edge_weight / (edge_weight.max() + 1e-8)
        blended = smoothed * (1 - edge_weight) + image_np * edge_weight
        blended = np.clip(blended, 0, 1)
        return _clahe_numpy(blended, clip_limit, tile_size)


# ============================================================
# CNR-aware fine-tuning
# ============================================================

def compute_ssim_loss(pred, target, window_size=11):
    """Differentiable SSIM loss (1 - SSIM)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(pred, window, padding=window_size // 2)
    mu2 = F.conv2d(target, window, padding=window_size // 2)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


def train_cnr_aware_finetune(backbone_path, train_jsonl, val_jsonl, output_dir,
                             epochs=15, lr=5e-5, device='cpu'):
    """Fine-tune NAFNet backbone with CNR-aware loss."""
    os.makedirs(output_dir, exist_ok=True)

    wrapper = BackboneWrapper(backbone_name='nafnet')
    ckpt = torch.load(backbone_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
               for k, v in state_dict.items()}

    backbone_state = wrapper.backbone.state_dict()
    compatible = {}
    for k, v in cleaned.items():
        bk = k.replace('backbone.', '') if k.startswith('backbone.') else k
        if bk in backbone_state and v.shape == backbone_state[bk].shape:
            compatible[bk] = v
    if compatible:
        wrapper.backbone.load_state_dict(compatible, strict=False)
        print(f"Loaded {len(compatible)}/{len(backbone_state)} backbone params")
    else:
        try:
            wrapper.backbone.load_state_dict(cleaned, strict=False)
            print("Loaded backbone weights directly")
        except Exception as e:
            print(f"Warning: Could not load backbone weights: {e}")

    wrapper = wrapper.to(device)
    wrapper.train()

    for p in wrapper.backbone.parameters():
        p.requires_grad = True

    train_dataset = PKU37Dataset(train_jsonl, patch_size=96, is_train=True)
    val_dataset = PKU37Dataset(val_jsonl, patch_size=0, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0,
        pin_memory=False, drop_last=True
    )

    optimizer = torch.optim.AdamW(wrapper.backbone.parameters(),
                                  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    w_mse = 1.0
    w_ssim = 0.5
    w_cnr = 0.05
    w_tci = 0.02

    for epoch in range(epochs):
        wrapper.train()
        epoch_losses = []
        epoch_mse = []
        epoch_cnr = []

        for batch_idx, batch in enumerate(train_loader):
            clean = batch['clean'].to(device)
            noisy = batch['noisy'].to(device)

            output = wrapper.backbone(noisy)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = output.clamp(0, 1)

            mse_loss = F.mse_loss(output, clean)
            ssim_loss = compute_ssim_loss(output, clean)

            B, C, H, W = output.shape
            cnr_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                tissue_vals_out = output[b, 0, :int(0.7 * H), :]
                bg_vals_out = output[b, 0, int(0.75 * H):, :]
                mu_t_out = tissue_vals_out.mean()
                mu_b_out = bg_vals_out.mean()
                sigma_b_out = bg_vals_out.std() + 1e-6
                cnr_out = (mu_t_out - mu_b_out) / sigma_b_out

                tissue_vals_c = clean[b, 0, :int(0.7 * H), :]
                bg_vals_c = clean[b, 0, int(0.75 * H):, :]
                mu_t_c = tissue_vals_c.mean()
                mu_b_c = bg_vals_c.mean()
                sigma_b_c = bg_vals_c.std() + 1e-6
                cnr_clean = (mu_t_c - mu_b_c) / sigma_b_c

                ratio = (cnr_out + 1e-6) / (cnr_clean + 1e-6)
                ratio = ratio.clamp(0.1, 10.0)
                cnr_loss = cnr_loss - torch.log(ratio)
            cnr_loss = cnr_loss / B

            tci_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                tci_out = output[b, 0, :int(0.7 * H), :].std() + 1e-6
                tci_clean = clean[b, 0, :int(0.7 * H), :].std() + 1e-6
                ratio = tci_out / tci_clean
                ratio = ratio.clamp(0.1, 10.0)
                tci_loss = tci_loss - torch.log(ratio)
            tci_loss = tci_loss / B

            total_loss = (w_mse * mse_loss + w_ssim * ssim_loss
                          + w_cnr * cnr_loss + w_tci * tci_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.backbone.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_mse.append(mse_loss.item())
            epoch_cnr.append(-cnr_loss.item())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)}: "
                      f"loss={np.mean(epoch_losses[-50:]):.4f} "
                      f"mse={np.mean(epoch_mse[-50:]):.6f} "
                      f"cnr_ratio={np.mean(epoch_cnr[-50:]):.3f}")

        scheduler.step()

        wrapper.eval()
        val_losses = []
        n_val = min(20, len(val_dataset))
        for i in range(n_val):
            sample = val_dataset[i]
            clean_v = sample['clean'].unsqueeze(0).to(device)
            noisy_v = sample['noisy'].unsqueeze(0).to(device)

            with torch.no_grad():
                output_v = wrapper.backbone(noisy_v)
                if isinstance(output_v, (tuple, list)):
                    output_v = output_v[0]
                val_mse = F.mse_loss(output_v.clamp(0, 1), clean_v)
                val_losses.append(val_mse.item())

        avg_val_loss = np.mean(val_losses)
        avg_train_loss = np.mean(epoch_losses)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f} "
              f"val_mse={avg_val_loss:.6f} avg_cnr_ratio={np.mean(epoch_cnr):.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': wrapper.backbone.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"  Saved best model (val_mse={avg_val_loss:.6f})")

    save_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': wrapper.backbone.state_dict(),
        'epoch': epochs,
    }, save_path)
    print(f"Fine-tuning complete. Best val_mse={best_val_loss:.6f}")
    return os.path.join(output_dir, 'best_model.pth')


# ============================================================
# Evaluation
# ============================================================

def evaluate_all_methods(backbone_path, coopns_ckpt, cnr_ft_path,
                         test_jsonl, output_path, device='cpu',
                         skip_coopns=False):
    """Evaluate all comparison methods on PKU37 test set.

    Key fix: Computes Otsu mask from backbone output ONCE, then reuses
    the same mask for all methods' CNR computation (matching paper's validate()).
    """

    print("=" * 70)
    print("COMPARISON EXPERIMENT: NAFNet + Post-Processing Methods")
    print("=" * 70)

    # ---- Step 1: Load dataset and cache backbone outputs ----
    print("\n[1/8] Loading dataset and caching NAFNet backbone outputs...")
    dataset = PKU37Dataset(test_jsonl, patch_size=0, is_train=False)
    n_images = len(dataset)

    # Load full model (backbone + corrector)
    ref_model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone=backbone_path,
        hidden_channels=64,
    )
    if not skip_coopns:
        ckpt = torch.load(coopns_ckpt, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
                   for k, v in state_dict.items()}
        model_state = ref_model.state_dict()
        compatible = {k: v for k, v in cleaned.items()
                      if k in model_state and v.shape == model_state[k].shape}
        ref_model.load_state_dict(compatible, strict=False)
    ref_model = ref_model.to(device)
    ref_model.eval()

    # Cache backbone outputs AND Otsu masks
    cached_data = []
    for i in range(n_images):
        sample = dataset[i]
        clean = sample['clean'].unsqueeze(0).to(device)
        noisy = sample['noisy'].unsqueeze(0).to(device)

        with torch.no_grad():
            backbone_out, unc = ref_model.backbone(noisy)

        backbone_np = backbone_out.squeeze().cpu().numpy()
        # Compute Otsu mask from backbone output ONCE — used for ALL methods
        backbone_otsu = otsu_threshold_np(backbone_np)

        cached_data.append({
            'clean_np': clean.squeeze().cpu().numpy(),
            'noisy_np': noisy.squeeze().cpu().numpy(),
            'backbone_np': backbone_np,
            'backbone_otsu': backbone_otsu,
            'backbone_tensor': backbone_out.cpu(),
            'unc_tensor': unc.cpu(),
            'noisy_tensor': noisy.cpu(),
        })

        if (i + 1) % 50 == 0:
            print(f"  Cached {i+1}/{n_images}")

    print(f"  Cached {n_images} images with Otsu masks.")

    # ---- Step 2: Backbone-only metrics ----
    print("\n[2/8] Computing NAFNet backbone-only metrics...")
    backbone_metrics = []
    for item in cached_data:
        # For backbone: otsu mask is from its own output (consistent)
        m = compute_clinical_metrics_np(item['backbone_np'], item['clean_np'],
                                         backbone_otsu_mask=item['backbone_otsu'])
        backbone_metrics.append(m)

    avg_bb = {}
    for key in backbone_metrics[0]:
        avg_bb[key] = np.mean([m[key] for m in backbone_metrics])
    print(f"  Backbone: PSNR={avg_bb['psnr']:.2f} CNR={avg_bb['cnr']:.3f} "
          f"TCI={avg_bb['tci']:.4f} EPI={avg_bb['epi']:.4f} BS={avg_bb['bs']:.4f} "
          f"ENL={avg_bb['enl']:.1f} SNR={avg_bb['snr']:.2f}")

    all_results = {}
    all_results['backbone_only'] = {
        'method': 'NAFNet (backbone only)',
        'avg_metrics': {k: float(v) for k, v in avg_bb.items()},
    }

    # ---- Step 3: CoopNS-OCT ----
    if skip_coopns:
        print("\n[3/8] SKIPPING CoopNS-OCT (--skip_coopns). Use paper values.")
    else:
        print("\n[3/8] Computing NAFNet + CoopNS-OCT metrics...")
        coopns_metrics = []
        for i, item in enumerate(cached_data):
            with torch.no_grad():
                corrected, _ = ref_model.corrector(
                    item['backbone_tensor'].to(device),
                    item['noisy_tensor'].to(device),
                    None,
                    nafnet_uncertainty=item['unc_tensor'].to(device),
                    return_details=False,
                )
            co_np = corrected.squeeze().cpu().numpy()
            m = compute_clinical_metrics_np(co_np, item['clean_np'],
                                             backbone_otsu_mask=item['backbone_otsu'])
            coopns_metrics.append(m)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_images}")

        avg_coopns = {}
        for key in coopns_metrics[0]:
            avg_coopns[key] = np.mean([m[key] for m in coopns_metrics])

        all_results['coopns_oct'] = {
            'method': 'NAFNet + CoopNS-OCT (ours)',
            'avg_metrics': {k: float(v) for k, v in avg_coopns.items()},
        }
        print(f"  CoopNS-OCT: PSNR={avg_coopns['psnr']:.2f} CNR={avg_coopns['cnr']:.3f} "
              f"EPI={avg_coopns['epi']:.4f} ENL={avg_coopns['enl']:.1f}")

    del ref_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Step 4: Histogram Equalization ----
    print("\n[4/8] Computing NAFNet + Histogram Equalization metrics...")
    he_metrics = []
    for i, item in enumerate(cached_data):
        he_out = apply_histogram_equalization(item['backbone_np'])
        m = compute_clinical_metrics_np(he_out, item['clean_np'],
                                         backbone_otsu_mask=item['backbone_otsu'])
        he_metrics.append(m)

    avg_he = {}
    for key in he_metrics[0]:
        avg_he[key] = np.mean([m[key] for m in he_metrics])

    all_results['hist_eq'] = {
        'method': 'NAFNet + HE',
        'avg_metrics': {k: float(v) for k, v in avg_he.items()},
    }
    print(f"  HE: PSNR={avg_he['psnr']:.2f} CNR={avg_he['cnr']:.3f}")

    # ---- Step 5: CLAHE ----
    print("\n[5/8] Computing NAFNet + CLAHE metrics...")
    clahe_metrics = []
    for i, item in enumerate(cached_data):
        clahe_out = apply_clahe(item['backbone_np'], clip_limit=2.0, tile_size=8)
        m = compute_clinical_metrics_np(clahe_out, item['clean_np'],
                                         backbone_otsu_mask=item['backbone_otsu'])
        clahe_metrics.append(m)

    avg_clahe = {}
    for key in clahe_metrics[0]:
        avg_clahe[key] = np.mean([m[key] for m in clahe_metrics])

    all_results['clahe'] = {
        'method': 'NAFNet + CLAHE',
        'avg_metrics': {k: float(v) for k, v in avg_clahe.items()},
    }
    print(f"  CLAHE: PSNR={avg_clahe['psnr']:.2f} CNR={avg_clahe['cnr']:.3f}")

    # ---- Step 6: Unsharp Masking ----
    print("\n[6/8] Computing NAFNet + Unsharp Masking metrics...")
    # Try multiple alpha values, report best balanced result
    best_usm_alpha = 0.5
    best_usm_metrics = None
    for alpha in [0.3, 0.5, 0.8]:
        usm_metrics_trial = []
        for item in cached_data:
            usm_out = apply_unsharp_mask(item['backbone_np'], sigma=2.0, alpha=alpha)
            m = compute_clinical_metrics_np(usm_out, item['clean_np'],
                                             backbone_otsu_mask=item['backbone_otsu'])
            usm_metrics_trial.append(m)
        avg_trial = {k: np.mean([m[k] for m in usm_metrics_trial]) for k in usm_metrics_trial[0]}
        print(f"  USM alpha={alpha}: PSNR={avg_trial['psnr']:.2f} CNR={avg_trial['cnr']:.3f} "
              f"EPI={avg_trial['epi']:.4f} BS={avg_trial['bs']:.4f}")
        if best_usm_metrics is None:
            best_usm_metrics = usm_metrics_trial
            best_usm_alpha = alpha
        else:
            # Pick alpha with best CNR (the metric we're trying to improve)
            avg_prev = np.mean([m['cnr'] for m in best_usm_metrics])
            if avg_trial['cnr'] > avg_prev:
                best_usm_metrics = usm_metrics_trial
                best_usm_alpha = alpha

    avg_usm = {k: np.mean([m[k] for m in best_usm_metrics]) for k in best_usm_metrics[0]}
    all_results['usm'] = {
        'method': f'NAFNet + USM (alpha={best_usm_alpha})',
        'avg_metrics': {k: float(v) for k, v in avg_usm.items()},
    }
    print(f"  Best USM (alpha={best_usm_alpha}): PSNR={avg_usm['psnr']:.2f} CNR={avg_usm['cnr']:.3f}")

    # ---- Step 7: Bilateral Filter + CLAHE ----
    print("\n[7/8] Computing NAFNet + BF+CLAHE metrics...")
    bf_clahe_metrics = []
    for i, item in enumerate(cached_data):
        bf_clahe_out = apply_bilateral_clahe(item['backbone_np'])
        m = compute_clinical_metrics_np(bf_clahe_out, item['clean_np'],
                                         backbone_otsu_mask=item['backbone_otsu'])
        bf_clahe_metrics.append(m)

    avg_bf_clahe = {}
    for key in bf_clahe_metrics[0]:
        avg_bf_clahe[key] = np.mean([m[key] for m in bf_clahe_metrics])

    all_results['bf_clahe'] = {
        'method': 'NAFNet + BF+CLAHE',
        'avg_metrics': {k: float(v) for k, v in avg_bf_clahe.items()},
    }
    print(f"  BF+CLAHE: PSNR={avg_bf_clahe['psnr']:.2f} CNR={avg_bf_clahe['cnr']:.3f}")

    # ---- Step 8: CNR-aware fine-tuning ----
    print("\n[8/8] Computing NAFNet + CNR-aware fine-tuning metrics...")
    if cnr_ft_path and os.path.exists(cnr_ft_path):
        ft_wrapper = BackboneWrapper(backbone_name='nafnet')
        ft_ckpt = torch.load(cnr_ft_path, map_location='cpu', weights_only=False)
        ft_state = ft_ckpt.get('model_state_dict', ft_ckpt)
        ft_cleaned = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
                      for k, v in ft_state.items()}
        try:
            ft_wrapper.backbone.load_state_dict(ft_cleaned, strict=False)
        except Exception:
            stripped = {k.replace('backbone.', ''): v for k, v in ft_cleaned.items()}
            ft_wrapper.backbone.load_state_dict(stripped, strict=False)
        ft_wrapper = ft_wrapper.to(device)
        ft_wrapper.eval()

        cnr_ft_metrics = []
        for i in range(n_images):
            sample = dataset[i]
            noisy = sample['noisy'].unsqueeze(0).to(device)
            clean_np = cached_data[i]['clean_np']

            with torch.no_grad():
                ft_out = ft_wrapper.backbone(noisy)
                if isinstance(ft_out, (tuple, list)):
                    ft_out = ft_out[0]
            ft_np = ft_out.squeeze().cpu().numpy()
            # Use backbone's Otsu mask for fair comparison
            m = compute_clinical_metrics_np(ft_np, clean_np,
                                             backbone_otsu_mask=cached_data[i]['backbone_otsu'])
            cnr_ft_metrics.append(m)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_images}")

        avg_ft = {}
        for key in cnr_ft_metrics[0]:
            avg_ft[key] = np.mean([m[key] for m in cnr_ft_metrics])

        all_results['cnr_finetune'] = {
            'method': 'NAFNet + CNR-FT',
            'avg_metrics': {k: float(v) for k, v in avg_ft.items()},
        }
        print(f"  CNR-FT: PSNR={avg_ft['psnr']:.2f} CNR={avg_ft['cnr']:.3f}")

        del ft_wrapper
        gc.collect()
    else:
        print(f"  SKIP: Fine-tuned model not found at {cnr_ft_path}")
        print(f"  Run with --train_cnr_ft to train it first.")

    # ---- Compute deltas and print summary ----
    print("\n" + "=" * 120)
    print(f"{'Method':<35} {'PSNR':>8} {'dPSNR':>8} {'CNR':>8} {'dCNR%':>8} "
          f"{'dTCI%':>8} {'dEPI%':>8} {'dBS%':>8} {'dENL%':>8} {'dSNR%':>8}")
    print("-" * 120)

    summary_results = {}
    for name, res in all_results.items():
        avg = res['avg_metrics']
        d_psnr = avg['psnr'] - avg_bb['psnr']
        deltas = {'psnr': d_psnr}
        for key in ['cnr', 'tci', 'epi', 'bs', 'enl', 'snr']:
            if abs(avg_bb[key]) > 1e-8:
                deltas[key] = (avg[key] - avg_bb[key]) / abs(avg_bb[key]) * 100
            else:
                deltas[key] = 0.0

        summary_results[name] = {
            'method': res['method'],
            'absolute': {k: float(v) for k, v in avg.items()},
            'deltas': {k: float(v) for k, v in deltas.items()},
        }

        print(f"{res['method']:<35} {avg['psnr']:>8.2f} {d_psnr:>+8.2f} "
              f"{avg['cnr']:>8.3f} {deltas['cnr']:>+8.1f} "
              f"{deltas['tci']:>+8.1f} {deltas['epi']:>+8.1f} "
              f"{deltas['bs']:>+8.1f} {deltas['enl']:>+8.1f} {deltas['snr']:>+8.1f}")

    print("=" * 120)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return summary_results


def main():
    parser = argparse.ArgumentParser(description='Comparison experiments')
    parser.add_argument('--backbone_path', type=str,
                        default='outputs/nafnet_pku37_w40/best_model.pth')
    parser.add_argument('--coopns_ckpt', type=str,
                        default='outputs/nafnet_relaxed_psnr/best_model_cooperative.pth')
    parser.add_argument('--test_jsonl', type=str,
                        default='pku37_oct_dataset/pku37_real_test.jsonl')
    parser.add_argument('--train_jsonl', type=str,
                        default='pku37_oct_dataset/pku37_real_train.jsonl')
    parser.add_argument('--val_jsonl', type=str,
                        default='pku37_oct_dataset/pku37_real_val.jsonl')
    parser.add_argument('--output', type=str,
                        default='outputs/comparison_results_v2.json')
    parser.add_argument('--cnr_ft_dir', type=str,
                        default='outputs/nafnet_cnr_finetune')
    parser.add_argument('--train_cnr_ft', action='store_true',
                        help='Train CNR-aware fine-tuned model')
    parser.add_argument('--skip_coopns', action='store_true',
                        help='Skip CoopNS-OCT evaluation (use paper values)')
    parser.add_argument('--ft_epochs', type=int, default=15)
    parser.add_argument('--ft_lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    # Step 1: Train CNR-aware fine-tuned model if requested
    cnr_ft_path = os.path.join(args.cnr_ft_dir, 'best_model.pth')
    if args.train_cnr_ft:
        print("\n" + "=" * 70)
        print("PHASE 1: Training CNR-aware fine-tuned NAFNet")
        print("=" * 70)
        cnr_ft_path = train_cnr_aware_finetune(
            backbone_path=args.backbone_path,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            output_dir=args.cnr_ft_dir,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            device=args.device,
        )

    # Step 2: Evaluate all methods
    print("\n" + "=" * 70)
    print("PHASE 2: Evaluating all comparison methods")
    print("=" * 70)
    evaluate_all_methods(
        backbone_path=args.backbone_path,
        coopns_ckpt=args.coopns_ckpt,
        cnr_ft_path=cnr_ft_path,
        test_jsonl=args.test_jsonl,
        output_path=args.output,
        device=args.device,
        skip_coopns=args.skip_coopns,
    )


if __name__ == '__main__':
    main()
