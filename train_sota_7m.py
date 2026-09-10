#!/usr/bin/env python3
"""
Train SOTA 7M-parameter baselines for fair comparison with NAFNet backbone.

Models: DnCNN-7M, SwinIR-7M, KBNet-7M, MambaIR-7M
All ~7M params to match our NAFNet backbone (7.02M).

Memory-safe design for 7.8GB RAM / CPU-only:
  - batch_size=4 (1 for SwinIR/MambaIR), patch_size=96
  - Sequential model training with cleanup between models
  - Memory monitoring every epoch
  - Loss tensor cleanup after backward
  - PKU37Dataset matching train_v8_cooperative.py exactly

Metrics displayed for SOTA baselines (absolute values, not deltas from noisy):
  - Traditional: PSNR, SSIM (absolute)
  - Clinical Preservation: Contrast, Boundary, Texture, Edge as ratio to clean GT (1.0 = perfect)
  - OCT Clinical: CNR, TCI, EPI, Boundary Sharpness (absolute + delta)
  - Correction magnitude
"""
import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time
import random
import resource

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# --- CPU performance tuning (only when run directly, not on import) ---
def _setup_cpu_tuning():
    torch.set_num_threads(2)            # Match CPU count (avoid oversubscription)
    torch.set_num_interop_threads(2)
    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True

from neuro_symbolic_corrector_v8 import EnhancedGTFreePredicates


_LIBC = None

def _release_memory(full=False):
    """Release memory. full=True does gc.collect + malloc_trim (expensive ~200ms)."""
    if full:
        gc.collect()
        global _LIBC
        try:
            if _LIBC is None:
                _LIBC = ctypes.CDLL('libc.so.6')
            _LIBC.malloc_trim(0)
        except Exception:
            pass
    else:
        # Lightweight: only collect generation 0 (young objects, ~5ms)
        gc.collect(0)


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------
def get_current_rss_mb():
    """Get current RSS in MB from /proc."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return 0


def memory_guard(label="", limit_mb=5000):
    """Print memory usage and raise if exceeding limit."""
    rss = get_current_rss_mb()
    if rss > limit_mb:
        raise MemoryError(f"[{label}] RSS={rss:.0f}MB exceeds {limit_mb}MB limit!")
    return rss


# ---------------------------------------------------------------------------
# Dataset (patch-based, jsonl format)
# ---------------------------------------------------------------------------
class PKU37Dataset(Dataset):
    """PKU37 real noise dataset with in-memory caching for fast access."""
    def __init__(self, jsonl_path, max_samples=None, patch_size=96, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train

        # Load all image paths
        paths = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'clean_path' in entry and 'noisy_path' in entry:
                    if os.path.exists(entry['clean_path']) and os.path.exists(entry['noisy_path']):
                        paths.append((entry['clean_path'], entry['noisy_path']))
        if max_samples:
            paths = paths[:max_samples]

        # Cache all images in memory as uint8 numpy arrays (~1.1GB vs 4.5GB for float32)
        # Convert to float32 tensor in __getitem__ (only for the small patch)
        self.clean_imgs = []
        self.noisy_imgs = []
        for clean_path, noisy_path in paths:
            c = np.array(Image.open(clean_path))
            n = np.array(Image.open(noisy_path))
            self.clean_imgs.append(c)
            self.noisy_imgs.append(n)
        est_mb = sum(c.nbytes + n.nbytes for c, n in zip(self.clean_imgs, self.noisy_imgs)) / 1e6
        print(f"  Cached {len(self.clean_imgs)} PKU37 images in memory ({est_mb:.0f} MB)"
              f"{' (train)' if is_train else ' (val, full-res)'}")

    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        c = self.clean_imgs[idx]  # uint8 numpy (H, W)
        n = self.noisy_imgs[idx]
        # Crop patch from uint8 (cheap), then convert to float32
        # Train: random crop, Val: center crop (fast validation on patches)
        if self.patch_size > 0:
            H, W = c.shape
            if H > self.patch_size and W > self.patch_size:
                if self.is_train:
                    y = random.randint(0, H - self.patch_size)
                    x = random.randint(0, W - self.patch_size)
                else:
                    y = (H - self.patch_size) // 2
                    x = (W - self.patch_size) // 2
                c = c[y:y+self.patch_size, x:x+self.patch_size]
                n = n[y:y+self.patch_size, x:x+self.patch_size]
        # Augmentation on numpy (avoids torch.flip + .contiguous() overhead)
        if self.is_train:
            if random.random() > 0.5:
                c = c[:, ::-1]
                n = n[:, ::-1]
            if random.random() > 0.5:
                c = c[::-1]
                n = n[::-1]
        # Convert uint8 -> float32 [0,1] (astype handles non-contiguous strides from flips)
        c_f = c.astype(np.float32)
        c_f *= (1.0 / 255.0)
        n_f = n.astype(np.float32)
        n_f *= (1.0 / 255.0)
        clean = torch.from_numpy(c_f).unsqueeze(0)
        noisy = torch.from_numpy(n_f).unsqueeze(0)
        return {'clean': clean, 'noisy': noisy}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_model(name):
    name = name.lower()
    if name == 'dncnn_7m':
        from sota.models.dncnn_7m import DnCNN
        # DnCNN: 17 sequential Conv+BN+ReLU layers — lightweight, checkpointing unnecessary
        model = DnCNN(in_channels=1, out_channels=1, num_layers=17, channels=228, use_checkpoint=False)
    elif name == 'swinir_7m':
        from sota.models.swinir_7m import SwinIR
        # SwinIR: 36 transformer layers store ~4GB activations without checkpointing
        model = SwinIR(in_channels=1, out_channels=1, embed_dim=138,
                      depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6],
                      window_size=8, mlp_ratio=2.0)
        # use_checkpoint defaults to True in SwinIR.__init__ — keep it enabled
    elif name == 'kbnet_7m':
        from sota.models.kbnet_7m import KBNet
        # KBNet: KBA unfold + 24 blocks store ~2.5GB activations without checkpointing
        model = KBNet(in_channels=1, out_channels=1, width=32,
                     middle_blk_num=10, enc_blk_nums=[2,2,4], dec_blk_nums=[2,2,2],
                     nset=32, gc=1, ffn_scale=2, use_checkpoint=True)
    elif name == 'mambair_7m':
        from sota.models.mambair_7m import MambaIR
        model = MambaIR(in_channels=1, out_channels=1, embed_dim=101,
                       depths=[6,6,6,6,6,6], d_state=16, d_conv=3, expand=2.,
                       drop_path_rate=0.1, use_checkpoint=True)
    elif name == 'nafnet_7m':
        from sota.models.nafnet_7m import NAFNet
        model = NAFNet(img_channel=1, width=40, middle_blk_num=1,
                      enc_blk_nums=[1,1,1,1], dec_blk_nums=[1,1,1,1])
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


# ---------------------------------------------------------------------------
# Metrics (matching train_v8_cooperative.py)
# ---------------------------------------------------------------------------
def compute_psnr_tensor(pred, target):
    """Per-image PSNR then average (avoids Jensen's inequality bias from batch-averaging MSE)."""
    B = pred.shape[0]
    total = 0.0
    for i in range(B):
        mse_val = F.mse_loss(pred[i], target[i], reduction='mean').item()
        if not math.isfinite(mse_val) or mse_val < 1e-10:
            total += 100.0 if mse_val < 1e-10 else 0.0
        else:
            total += 10.0 * math.log10(1.0 / mse_val)
    return total / B


def _gaussian_kernel_1d(size=11, sigma=1.5):
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size=11, sigma=1.5):
    """Create 2D Gaussian kernel for SSIM (standard: w=11, sigma=1.5)."""
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)  # outer product
    return k2d.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


# Module-level cached Gaussian kernel (created once)
_SSIM_KERNEL = _gaussian_kernel_2d(11, 1.5)


def compute_ssim_tensor(pred, target):
    """Structural similarity with 11x11 Gaussian window, sigma=1.5 (Wang et al., 2004).

    Uses reflect padding to avoid zero-padding bias at borders (significant on 64x64 patches).
    Optimized: pad only pred/target, compute products from padded tensors to avoid 3 extra pad ops.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = 5  # 11 // 2
    kernel = _SSIM_KERNEL.to(pred.device)
    # Pad only the two inputs (saves 3 pad operations)
    pred_pad = F.pad(pred, [pad, pad, pad, pad], mode='reflect')
    target_pad = F.pad(target, [pad, pad, pad, pad], mode='reflect')
    mu_p = F.conv2d(pred_pad, kernel)
    mu_t = F.conv2d(target_pad, kernel)
    sigma_p_sq = F.conv2d(pred_pad * pred_pad, kernel) - mu_p * mu_p
    sigma_t_sq = F.conv2d(target_pad * target_pad, kernel) - mu_t * mu_t
    sigma_pt = F.conv2d(pred_pad * target_pad, kernel) - mu_p * mu_t
    del pred_pad, target_pad
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p * mu_p + mu_t * mu_t + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()


# Module-level cached clinical metric kernels (created once, avoids per-batch allocation)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_LAPLACIAN = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
# Combined kernel: 3 filters in single conv call (replaces 3 separate conv2d calls)
_COMBINED_EDGE_KERNEL = torch.cat([_SOBEL_Y, _SOBEL_X, _LAPLACIAN], dim=0)  # [3, 1, 3, 3]


@torch.inference_mode()
def compute_all_clinical_metrics(denoised, noisy, clean):
    """
    Compute all clinical metrics matching train_v8_cooperative.py.

    For SOTA baselines:
      - 'noisy' plays the role of 'backbone' (baseline reference)
      - 'denoised' plays the role of 'corrected' (model output)

    Returns dict with all metrics.
    """
    assert denoised.shape[0] == 1, f"Clinical metrics require batch_size=1, got {denoised.shape[0]}"
    m = {}

    # --- Traditional Quality Metrics ---
    m['psnr_noisy'] = compute_psnr_tensor(noisy, clean)
    m['psnr_denoised'] = compute_psnr_tensor(denoised, clean)
    m['psnr_delta'] = m['psnr_denoised'] - m['psnr_noisy']

    m['ssim_noisy'] = compute_ssim_tensor(noisy, clean)
    m['ssim_denoised'] = compute_ssim_tensor(denoised, clean)
    m['ssim_delta'] = m['ssim_denoised'] - m['ssim_noisy']

    # SSIM preservation ratio (denoised should preserve structure)
    m['ssim_preservation_ratio'] = m['ssim_denoised'] / max(m['ssim_noisy'], 1e-8)

    # --- Use cached combined kernel ---
    device = denoised.device
    combined_kernel = _COMBINED_EDGE_KERNEL.to(device)

    # --- CNR (Contrast-to-Noise Ratio) ---
    signal_mask = (clean > clean.mean()).float()
    bg_mask = 1.0 - signal_mask
    signal_mask_sum = signal_mask.sum().clamp(min=1.0)
    bg_mask_sum = bg_mask.sum().clamp(min=1.0)

    # Noisy CNR
    noisy_signal_mean = (noisy * signal_mask).sum() / signal_mask_sum
    noisy_bg_mean = (noisy * bg_mask).sum() / bg_mask_sum
    noisy_bg_std = torch.sqrt(((noisy - noisy_bg_mean) ** 2 * bg_mask).sum() / bg_mask_sum + 1e-8).clamp(min=1e-4)
    m['cnr_noisy'] = ((noisy_signal_mean - noisy_bg_mean) / noisy_bg_std).clamp(-100, 100).item()

    # Denoised CNR
    den_signal_mean = (denoised * signal_mask).sum() / signal_mask_sum
    den_bg_mean = (denoised * bg_mask).sum() / bg_mask_sum
    den_bg_std = torch.sqrt(((denoised - den_bg_mean) ** 2 * bg_mask).sum() / bg_mask_sum + 1e-8).clamp(min=1e-4)
    m['cnr_denoised'] = ((den_signal_mean - den_bg_mean) / den_bg_std).clamp(-100, 100).item()

    m['cnr_delta'] = m['cnr_denoised'] - m['cnr_noisy']
    # For SOTA baselines, store absolute denoised CNR (not % change from noisy which is misleading)
    m['cnr_improvement'] = m['cnr_denoised']
    del signal_mask, bg_mask

    # --- Batched gradient computation (single conv with 3-output-channel combined kernel) ---
    stacked = torch.cat([noisy, denoised, clean], dim=0)  # [3, 1, H, W]
    all_results = F.conv2d(stacked, combined_kernel, padding=1)  # [3, 3, H, W]
    # all_results[:, 0] = sobel_y, [:, 1] = sobel_x, [:, 2] = laplacian
    noisy_gy   = all_results[0:1, 0:1].abs()
    denoised_gy = all_results[1:2, 0:1].abs()
    clean_gy   = all_results[2:3, 0:1].abs()
    noisy_gx   = all_results[0:1, 1:2]
    denoised_gx = all_results[1:2, 1:2]
    clean_gx   = all_results[2:3, 1:2]
    noisy_lap  = all_results[0:1, 2:3].abs()
    denoised_lap = all_results[1:2, 2:3].abs()
    clean_lap  = all_results[2:3, 2:3].abs()
    del all_results  # keep stacked for local_std reuse below

    # --- TCI (Tissue Contrast Index) ---
    clean_gy_mean = clean_gy.mean().clamp(min=1e-4)

    m['tci_noisy'] = (noisy_gy.mean() / clean_gy_mean).clamp(0, 10).item()
    m['tci_denoised'] = (denoised_gy.mean() / clean_gy_mean).clamp(0, 10).item()
    m['tci_delta'] = m['tci_denoised'] - m['tci_noisy']
    m['tci_improvement'] = ((m['tci_denoised'] / max(m['tci_noisy'], 1e-4)) - 1.0) * 100

    # --- Clinical Preservation: Contrast (local std ratio) ---
    # Reuse stacked tensor from gradient computation above (avoids second torch.cat)
    # Use reflect padding (like SSIM) to avoid zero-padding bias at borders
    stacked_pad = F.pad(stacked, [3, 3, 3, 3], mode='reflect')
    mu_all = F.avg_pool2d(stacked_pad, 7, 1, 0)
    std_all = torch.sqrt((F.avg_pool2d(stacked_pad ** 2, 7, 1, 0) - mu_all ** 2).clamp(min=1e-8))
    noisy_std, denoised_std, clean_std = std_all.chunk(3, dim=0)
    del stacked, stacked_pad, mu_all, std_all
    clean_mean_std = clean_std.mean().clamp(min=1e-4)

    m['noisy_contrast_pres'] = (noisy_std.mean() / clean_mean_std).clamp(0, 10).item()
    m['denoised_contrast_pres'] = (denoised_std.mean() / clean_mean_std).clamp(0, 10).item()
    m['contrast_ratio'] = m['denoised_contrast_pres']  # ratio to clean GT (1.0 = perfect)
    del noisy_std, denoised_std, clean_std

    # --- Clinical Preservation: Boundary (vertical gradient) ---
    m['noisy_boundary_pres'] = (noisy_gy.mean() / clean_gy_mean).clamp(0, 10).item()
    m['denoised_boundary_pres'] = (denoised_gy.mean() / clean_gy_mean).clamp(0, 10).item()
    m['boundary_ratio'] = m['denoised_boundary_pres']  # ratio to clean GT (1.0 = perfect)

    # --- Clinical Preservation: Texture (Laplacian) ---
    clean_lap_mean = clean_lap.mean().clamp(min=1e-4)

    m['noisy_texture_pres'] = (noisy_lap.mean() / clean_lap_mean).clamp(0, 10).item()
    m['denoised_texture_pres'] = (denoised_lap.mean() / clean_lap_mean).clamp(0, 10).item()
    m['texture_ratio'] = m['denoised_texture_pres']  # ratio to clean GT (1.0 = perfect)
    del noisy_lap, denoised_lap, clean_lap

    # --- Clinical Preservation: Edge (Sobel magnitude) ---
    noisy_edge_mag = torch.sqrt(noisy_gx**2 + noisy_gy**2 + 1e-8)
    denoised_edge_mag = torch.sqrt(denoised_gx**2 + denoised_gy**2 + 1e-8)
    clean_edge_mag = torch.sqrt(clean_gx**2 + clean_gy**2 + 1e-8)
    clean_edge_mean = clean_edge_mag.mean().clamp(min=1e-4)

    m['noisy_edge_pres'] = (noisy_edge_mag.mean() / clean_edge_mean).clamp(0, 10).item()
    m['denoised_edge_pres'] = (denoised_edge_mag.mean() / clean_edge_mean).clamp(0, 10).item()
    m['edge_ratio'] = m['denoised_edge_pres']  # ratio to clean GT (1.0 = perfect)

    # --- EPI (Edge Preservation Index) ---
    clean_ef = clean_edge_mag.view(-1)
    noisy_ef = noisy_edge_mag.view(-1)
    denoised_ef = denoised_edge_mag.view(-1)

    def norm_corr(a, b):
        # Scalar-reduction approach: avoids creating full-size intermediate tensors
        n = a.numel()
        sum_a = a.sum()
        sum_b = b.sum()
        mean_a = sum_a / n
        mean_b = sum_b / n
        cov = (a * b).sum() / n - mean_a * mean_b
        var_a = ((a * a).sum() / n - mean_a * mean_a).clamp(min=1e-16)
        var_b = ((b * b).sum() / n - mean_b * mean_b).clamp(min=1e-16)
        denom = (var_a * var_b).sqrt()
        return (cov / denom).item()

    m['epi_noisy'] = norm_corr(clean_ef, noisy_ef)
    m['epi_denoised'] = norm_corr(clean_ef, denoised_ef)
    m['epi_delta'] = m['epi_denoised'] - m['epi_noisy']
    del clean_ef, noisy_ef, denoised_ef

    # --- Boundary Sharpness ---
    clean_gy_max = clean_gy.max().clamp(min=1e-4)
    m['bs_noisy'] = (noisy_gy.max() / clean_gy_max).clamp(0, 10).item()
    m['bs_denoised'] = (denoised_gy.max() / clean_gy_max).clamp(0, 10).item()
    m['bs_delta'] = m['bs_denoised'] - m['bs_noisy']

    # --- Correction Magnitude ---
    m['correction_magnitude'] = (denoised - noisy).abs().mean().item()

    # Cleanup remaining intermediates
    del noisy_gy, denoised_gy, clean_gy, noisy_gx, denoised_gx, clean_gx
    del noisy_edge_mag, denoised_edge_mag, clean_edge_mag

    return m


# ---------------------------------------------------------------------------
# Print metrics in IEEE TMI format (matching train_v8_cooperative.py)
# ---------------------------------------------------------------------------
def print_sota_metrics(epoch, model_name, n_params, train_loss, val_metrics, rss_mb, epoch_time):
    """Print validation metrics for SOTA baselines.

    Differs from train_v8_cooperative.py in key ways:
    - Clinical preservation ratios are vs clean GT (1.0 = perfect), not denoised/noisy
    - CNR shown as absolute value, not % change from noisy
    - PSNR shown as absolute, not delta
    - Status checks use >0.5 threshold (retains >50% of clean features)
    """
    vm = val_metrics
    is_fast = 'cnr_improvement' not in vm  # fast_only mode has only PSNR+SSIM

    psnr_delta = vm['psnr_denoised'] - vm['psnr_noisy']
    ssim_delta = vm['ssim_denoised'] - vm['ssim_noisy']
    cnr_improvement = vm.get('cnr_improvement', 0)

    if is_fast:
        # Compact output for fast-only validation epochs
        print(f"\n{'#'*68}")
        print(f"# EPOCH {epoch:3d} | {model_name.upper()} | RSS={rss_mb:.0f}MB | {epoch_time:.1f}s")
        print(f"# PSNR: {vm['psnr_denoised']:.3f} dB ({psnr_delta:+.3f}) | SSIM: {vm['ssim_denoised']:.4f} ({ssim_delta:+.4f})")
        print(f"{'#'*68}")
        return

    # Clinical preservation ratios
    contrast_ratio = vm.get('contrast_ratio', 1.0)
    boundary_ratio = vm.get('boundary_ratio', 1.0)
    texture_ratio = vm.get('texture_ratio', 1.0)
    edge_ratio = vm.get('edge_ratio', 1.0)
    avg_ratio = (contrast_ratio + boundary_ratio + texture_ratio + edge_ratio) / 4

    # Count how many features retain >50% of clean GT (ratio is now denoised_pres, 1.0 = perfect)
    clinical_improvements = sum([
        1 if contrast_ratio > 0.5 else 0,
        1 if boundary_ratio > 0.5 else 0,
        1 if texture_ratio > 0.5 else 0,
        1 if edge_ratio > 0.5 else 0,
    ])

    # SSIM preservation ratio
    ssim_pres_ratio = vm.get('ssim_preservation_ratio', 1.0)
    ssim_preserved = ssim_pres_ratio >= 0.99

    # ===== HEADER (matches v8 format) =====
    print(f"\n{'#'*88}")
    print(f"# EPOCH {epoch:3d} \u2502 {model_name.upper()} ({n_params/1e6:.2f}M params) \u2502 RSS={rss_mb:.0f}MB \u2502 {epoch_time:.1f}s")
    print(f"# {'':>10} \u2502 Clinical: {clinical_improvements}/4 preserved (>50%) \u2502 CNR: {cnr_improvement:.2f} \u2502 PSNR: {vm['psnr_denoised']:.2f} dB")
    print(f"{'#'*88}")

    # ===== CLINICAL PRESERVATION TABLE (ratio to clean GT, 1.0 = perfect) =====
    print(f"\n\u250c{'\u2500'*67}\u2510")
    print(f"\u2502 {'CLINICAL PRESERVATION':<22} {'Denoised':>12} {'vs Clean':>12} {'Status':>16} \u2502")
    print(f"\u2502 {'(ratio to clean GT)':<22} {'(ratio)':>12} {'(1.0=perf)':>12} {'(>0.5)':>16} \u2502")
    print(f"\u251c{'\u2500'*67}\u2524")

    metrics_data = [
        ('Contrast (local std)', contrast_ratio),
        ('Boundary (v-grad)',    boundary_ratio),
        ('Texture (variance)',   texture_ratio),
        ('Edge (Sobel)',         edge_ratio),
    ]

    for name, ratio in metrics_data:
        status = "PRESERVED" if ratio > 0.5 else "LOW"
        print(f"\u2502 {name:<22} {ratio:>12.3f} {'1.0':>12} {status:>16} \u2502")

    print(f"\u251c{'\u2500'*67}\u2524")
    status_str = f"{clinical_improvements}/4 PRESERVED" if clinical_improvements > 0 else "NONE PRESERVED"
    print(f"\u2502 {'AVERAGE':<22} {avg_ratio:>12.3f} {'1.0':>12} {status_str:>16} \u2502")
    print(f"\u2514{'\u2500'*67}\u2518")

    # ===== TRADITIONAL QUALITY METRICS (matches v8 format) =====
    psnr_status = "[OK]" if abs(psnr_delta) <= 1.0 else "[!]"
    ssim_status = "[OK]" if ssim_delta >= -0.01 else "[!]"
    ssim_pres_status = "[OK]" if ssim_preserved else "[!]"

    print(f"\n\u250c{'\u2500'*86}\u2510")
    print(f"\u2502 {'TRADITIONAL METRICS':<30} {'Noisy':>12} {'Denoised':>12} {'Delta':>10} {'Preserved':>12} \u2502")
    print(f"\u251c{'\u2500'*86}\u2524")
    print(f"\u2502 {'PSNR (dB)':<30} {vm['psnr_noisy']:>12.2f} {vm['psnr_denoised']:>12.2f} {psnr_delta:>+9.3f} {psnr_status:>12} \u2502")
    print(f"\u2502 {'SSIM':<30} {vm['ssim_noisy']:>12.4f} {vm['ssim_denoised']:>12.4f} {ssim_delta:>+9.4f} {ssim_status:>12} \u2502")
    print(f"\u2502 {'SSIM Preservation Ratio':<30} {'-':>12} {ssim_pres_ratio:>12.4f} {'>=0.99':>10} {ssim_pres_status:>12} \u2502")
    print(f"\u251c{'\u2500'*86}\u2524")
    print(f"\u2502 {'Training Loss':<30} {train_loss:>56.6f} \u2502")
    print(f"\u2514{'\u2500'*86}\u2518")

    # ===== GT-FREE PREDICATES (matches v8 format — before OCT clinical) =====
    pred_scores = vm.get('pred_scores', {})
    pred_scores_noisy = vm.get('pred_scores_noisy', pred_scores)
    passed_count = sum(1 for k, s in pred_scores.items() if s >= 0.5 and k != 'P5') if pred_scores else 0

    if pred_scores:
        # Exclude P5 (Speckle) from averages to match pass count denominator
        scores_excl = {k: v for k, v in pred_scores.items() if k != 'P5'}
        avg_pred = sum(scores_excl.values()) / max(len(scores_excl), 1)
        scores_noisy_excl = {k: v for k, v in pred_scores_noisy.items() if k != 'P5'}
        avg_pred_noisy = sum(scores_noisy_excl.values()) / max(len(scores_noisy_excl), 1)

        print(f"\n\u250c{'\u2500'*76}\u2510")
        print(f"\u2502 {'GT-FREE PREDICATES':<24} {'Noisy':>12} {'Denoised':>12} {'Delta':>10} {'Status':>12} \u2502")
        print(f"\u251c{'\u2500'*76}\u2524")

        pred_names = {
            'P1': 'Edge Quality', 'P2': 'Contrast', 'P3': 'Smoothness',
            'P4': 'Structure', 'P5': 'Speckle (excluded)', 'P6': 'Anatomy'
        }

        for key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            score_denoised = pred_scores.get(key, 0)
            score_noisy = pred_scores_noisy.get(key, score_denoised)
            delta = score_denoised - score_noisy
            if key == 'P5':
                status = "EXCLUDED"
            else:
                status = "PASS" if score_denoised >= 0.5 else "FAIL"
            name = pred_names.get(key, key)
            print(f"\u2502 {name:<24} {score_noisy:>12.3f} {score_denoised:>12.3f} {delta:>+10.3f} {status:>12} \u2502")

        print(f"\u251c{'\u2500'*76}\u2524")
        avg_delta = avg_pred - avg_pred_noisy
        print(f"\u2502 {'AVERAGE':<24} {avg_pred_noisy:>12.3f} {avg_pred:>12.3f} {avg_delta:>+10.3f} {f'{passed_count}/5 PASS':>12} \u2502")
        print(f"\u2514{'\u2500'*76}\u2518")

    # ===== CORRECTION BEHAVIOR (matches v8 CORRECTOR BEHAVIOR section) =====
    correction_mag = vm.get('correction_magnitude', 0)
    if correction_mag < 0.005:
        corr_interp = "Very Light"
    elif correction_mag < 0.015:
        corr_interp = "Moderate"
    elif correction_mag < 0.03:
        corr_interp = "Active"
    else:
        corr_interp = "Heavy (!)"

    print(f"\n\u250c{'\u2500'*68}\u2510")
    print(f"\u2502 {'CORRECTION BEHAVIOR':<66} \u2502")
    print(f"\u251c{'\u2500'*68}\u2524")
    print(f"\u2502 {'Overall Correction Magnitude':<30} {correction_mag:>12.6f} {corr_interp:>22} \u2502")
    print(f"\u2514{'\u2500'*68}\u2518")

    # ===== OCT CLINICAL METRICS (matches v8 LEGACY CLINICAL METRICS) =====
    cnr_noisy = vm.get('cnr_noisy', 0)
    cnr_denoised = vm.get('cnr_denoised', 0)
    cnr_delta = cnr_denoised - cnr_noisy

    tci_noisy = vm.get('tci_noisy', 0)
    tci_denoised = vm.get('tci_denoised', 0)
    tci_delta = tci_denoised - tci_noisy

    epi_noisy = vm.get('epi_noisy', 0)
    epi_denoised = vm.get('epi_denoised', 0)
    epi_delta = epi_denoised - epi_noisy

    bs_noisy = vm.get('bs_noisy', 0)
    bs_denoised = vm.get('bs_denoised', 0)
    bs_delta = bs_denoised - bs_noisy

    # TCI, BS: ratio vs clean (1.0 = perfect), closer to 1.0 = improved
    tci_improved = abs(tci_denoised - 1.0) < abs(tci_noisy - 1.0)
    bs_improved = abs(bs_denoised - 1.0) < abs(bs_noisy - 1.0)
    legacy_improved = sum([
        1 if cnr_delta > 0 else 0,
        1 if tci_improved else 0,
        1 if epi_delta > 0 else 0,
        1 if bs_improved else 0,
    ])

    cnr_status = "\u2713" if cnr_delta >= 0 else "!"
    tci_status = "\u2713" if tci_improved else "!"
    epi_status = "\u2713" if epi_delta >= 0 else "~"
    bs_status = "\u2713" if bs_improved else "~"

    print(f"\n\u250c{'\u2500'*68}\u2510")
    print(f"\u2502 {'OCT CLINICAL METRICS (IEEE TMI)':<30} {'Noisy':>12} {'Denoised':>12} {'Delta':>10} \u2502")
    print(f"\u251c{'\u2500'*68}\u2524")
    print(f"\u2502 {'CNR (Contrast-to-Noise)':<30} {cnr_noisy:>12.3f} {cnr_denoised:>12.3f} {cnr_delta:>+9.3f}{cnr_status} \u2502")
    print(f"\u2502 {'TCI (Tissue Contrast Index)':<30} {tci_noisy:>12.3f} {tci_denoised:>12.3f} {tci_delta:>+9.3f}{tci_status} \u2502")
    print(f"\u2502 {'EPI (Edge Preservation)':<30} {epi_noisy:>12.4f} {epi_denoised:>12.4f} {epi_delta:>+9.4f}{epi_status} \u2502")
    print(f"\u2502 {'Boundary Sharpness':<30} {bs_noisy:>12.4f} {bs_denoised:>12.4f} {bs_delta:>+9.4f}{bs_status} \u2502")
    print(f"\u251c{'\u2500'*68}\u2524")
    legacy_status = f"{legacy_improved}/4 IMPROVED" if legacy_improved > 0 else "NO IMPROVEMENT"
    print(f"\u2502 {'CLINICAL ASSESSMENT':<30} {legacy_status:>36} \u2502")
    print(f"\u2514{'\u2500'*68}\u2518")

    # ===== QUICK SUMMARY LINE =====
    print(f"\n>>> Clinical: {clinical_improvements}/4 preserved (>50%) \u2502 Avg preservation: {avg_ratio:.3f}")
    print(f">>> PSNR: {vm['psnr_denoised']:.2f} dB \u2502 SSIM: {vm['ssim_denoised']:.4f} \u2502 CNR: {cnr_denoised:.2f}")
    print(f">>> Predicates: {passed_count}/5 pass \u2502 Correction: {correction_mag:.6f} ({corr_interp})")
    print(f"{'#'*88}\n")


# ---------------------------------------------------------------------------
# Sliding Window Inference (for full-resolution validation)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def sliding_window_inference(model, image, patch_size=96, overlap=16):
    """Apply model on full-resolution image using sliding window with overlap averaging.

    This matches how SwinIR, MambaIR, and other patch-based models are evaluated
    in their original papers. Overlap regions are averaged to avoid boundary artifacts.

    Args:
        model: Denoising model
        image: Input tensor [B, C, H, W]
        patch_size: Size of each patch (should match training patch_size)
        overlap: Overlap between adjacent patches in pixels

    Returns:
        Denoised full-resolution image [B, C, H, W]
    """
    B, C, H, W = image.shape
    assert B == 1, f"sliding_window_inference requires batch_size=1, got B={B}"
    stride = patch_size - overlap

    # If image is smaller than patch_size in either dimension, run directly
    # (avoids IndexError in _positions when range produces empty list)
    if H <= patch_size or W <= patch_size:
        return model(image).clamp(0, 1)

    # Generate patch start positions, ensuring full coverage
    def _positions(length, patch_sz, stride_sz):
        starts = list(range(0, length - patch_sz + 1, stride_sz))
        # Ensure the last patch covers the edge
        if starts[-1] + patch_sz < length:
            starts.append(length - patch_sz)
        return starts

    y_starts = _positions(H, patch_size, stride)
    x_starts = _positions(W, patch_size, stride)

    output = torch.zeros_like(image)
    count = torch.zeros_like(image)

    # Collect all patch positions and batch them for efficiency
    positions = [(y, x) for y in y_starts for x in x_starts]
    batch_sz = 4  # 4 patches of 96x96 is small enough for CPU
    for i in range(0, len(positions), batch_sz):
        batch_pos = positions[i:i+batch_sz]
        patches = torch.cat([
            image[:, :, y:y+patch_size, x:x+patch_size] for y, x in batch_pos
        ], dim=0)  # (N, C, ps, ps)
        preds = model(patches).clamp(0, 1)
        for j, (y, x) in enumerate(batch_pos):
            output[:, :, y:y+patch_size, x:x+patch_size] += preds[j:j+1]
            count[:, :, y:y+patch_size, x:x+patch_size] += 1.0
        del patches, preds

    return output / count


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def validate(model, loader, predicates_evaluator=None, cached_noisy_preds=None,
             patch_size=96, overlap=16, fast_only=False, use_sliding_window=True,
             cached_noisy_metrics=None, max_samples=0):
    """Comprehensive validation with all IEEE TMI metrics + GT-free predicates.

    Args:
        cached_noisy_preds: If provided, skip recomputing noisy baseline predicates
            (they never change since noisy input is fixed with deterministic patches).
            Dict of {P1: score, P2: score, ...}.
        patch_size: Patch size for sliding window inference.
        overlap: Overlap between adjacent patches.
        fast_only: If True, compute only PSNR + SSIM (skip heavy clinical metrics).
        use_sliding_window: If False, run model directly on full-res images
            (safe for fully-convolutional models like DnCNN).
        cached_noisy_metrics: If provided, skip recomputing noisy baseline PSNR/SSIM
            (constant across epochs). Dict of {psnr_noisy: val, ssim_noisy: val}.
        max_samples: If > 0, limit validation to first N samples (faster for
            intermediate evals where precise metrics aren't needed).
    """
    model.eval()
    accum = {}
    n = 0

    # Predicate score accumulators
    pred_key_map = {
        'P1_edge': 'P1', 'P2_contrast': 'P2', 'P3_smooth': 'P3',
        'P4_structure': 'P4', 'P5_speckle': 'P5', 'P6_anatomy': 'P6'
    }
    total_pred_scores = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0, 'P6': 0}
    need_noisy_preds = predicates_evaluator is not None and cached_noisy_preds is None
    total_pred_scores_noisy = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0, 'P6': 0}
    # Skip predicates in fast mode
    if fast_only:
        predicates_evaluator = None

    total_batches = len(loader) if max_samples <= 0 else min(max_samples, len(loader))
    for batch in tqdm(loader, desc="Validation", leave=False, total=total_batches):
        noisy = batch['noisy'].to(memory_format=torch.channels_last)
        clean = batch['clean'].to(memory_format=torch.channels_last)
        if use_sliding_window:
            denoised = sliding_window_inference(model, noisy, patch_size=patch_size, overlap=overlap)
        else:
            denoised = model(noisy).clamp(0, 1)

        if fast_only:
            # Fast path: only PSNR + SSIM (skip all clinical metrics)
            metrics = {}
            metrics['psnr_denoised'] = compute_psnr_tensor(denoised, clean)
            metrics['ssim_denoised'] = compute_ssim_tensor(denoised, clean)
            # Noisy baseline metrics are constant — use cache if available
            if cached_noisy_metrics is not None:
                metrics['psnr_noisy'] = cached_noisy_metrics['psnr_noisy']
                metrics['ssim_noisy'] = cached_noisy_metrics['ssim_noisy']
            else:
                metrics['psnr_noisy'] = compute_psnr_tensor(noisy, clean)
                metrics['ssim_noisy'] = compute_ssim_tensor(noisy, clean)
        else:
            metrics = compute_all_clinical_metrics(denoised, noisy, clean)

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0) + v
        n += 1

        # GT-free predicates on denoised output
        if predicates_evaluator is not None:
            pred_denoised = predicates_evaluator(denoised, noisy, clean=clean)
            for orig_key, score in pred_denoised['scores'].items():
                mapped_key = pred_key_map.get(orig_key, orig_key)
                if mapped_key in total_pred_scores:
                    score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                    total_pred_scores[mapped_key] += score_val
            del pred_denoised

            # Only compute noisy baseline on first epoch (cache for subsequent epochs)
            if need_noisy_preds:
                pred_noisy = predicates_evaluator(noisy, noisy)
                for orig_key, score in pred_noisy['scores'].items():
                    mapped_key = pred_key_map.get(orig_key, orig_key)
                    if mapped_key in total_pred_scores_noisy:
                        score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                        total_pred_scores_noisy[mapped_key] += score_val
                del pred_noisy

        del batch, noisy, clean, denoised, metrics
        if n % 10 == 0:
            _release_memory()
        if max_samples > 0 and n >= max_samples:
            break

    # Average
    result = {k: v / n for k, v in accum.items()}

    # Average predicate scores
    if predicates_evaluator is not None and n > 0:
        result['pred_scores'] = {k: v / n for k, v in total_pred_scores.items()}
        if cached_noisy_preds is not None:
            result['pred_scores_noisy'] = cached_noisy_preds
        else:
            result['pred_scores_noisy'] = {k: v / n for k, v in total_pred_scores_noisy.items()}

    return result


# ---------------------------------------------------------------------------
# Training one model
# ---------------------------------------------------------------------------
def train_single_model(model_name, train_jsonl, val_jsonl, output_dir,
                       epochs=50, batch_size=8, patch_size=128, lr=5e-5,
                       max_train=None, max_val=None, mem_limit_mb=5000,
                       patience=10, eval_every=1):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    rss = memory_guard(f"before {model_name}", mem_limit_mb)
    print(f"  RSS before model creation: {rss:.0f} MB")

    # CPU-safe batch sizes: heavy models get bs=1, others capped at 4
    # KBNet's KBA unfold operations create large dense matrices per block
    if model_name in ('mambair_7m', 'swinir_7m', 'kbnet_7m'):
        effective_bs = 1
    else:
        effective_bs = min(batch_size, 4)
    if effective_bs != batch_size:
        print(f"  NOTE: Using batch_size={effective_bs} for {model_name} (CPU-safe)")

    model = build_model(model_name)
    # Channels-last (NHWC) format: ~1.3-2x faster Conv2d on CPU via MKL-DNN
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Gradient checkpointing: {getattr(model, 'use_checkpoint', False)}")
    print(f"  Memory format: channels_last (NHWC)")

    rss = memory_guard(f"after model creation", mem_limit_mb)
    print(f"  RSS after model creation: {rss:.0f} MB")

    # GT-free predicates evaluator (P1-P6)
    predicates_evaluator = EnhancedGTFreePredicates()
    predicates_evaluator.eval()

    train_ds = PKU37Dataset(train_jsonl, max_samples=max_train, patch_size=patch_size, is_train=True)
    # For transformer models (SwinIR, MambaIR), validate on center-cropped patches
    # instead of full-res to avoid >5min/image inference on CPU.
    # Full-res evaluation is done separately after training.
    val_patch = patch_size if model_name in ('swinir_7m', 'mambair_7m') else 0
    val_ds = PKU37Dataset(val_jsonl, max_samples=max_val, patch_size=val_patch, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=effective_bs, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Exclude normalization (BN/LN) and bias params from weight decay
    # Match by module type (not name) since BN layers may not have 'bn' in name
    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) or \
           type(module).__name__ == 'LayerNorm2d':
            for param in module.parameters():
                no_decay_ids.add(id(param))
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_ids or name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 1e-4},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(output_dir, exist_ok=True)
    best_psnr = 0.0
    epochs_no_improve = 0
    results_log = []
    cached_noisy_preds = None  # Cache noisy baseline predicates after first epoch
    cached_noisy_metrics = None  # Cache noisy baseline PSNR/SSIM (constant across epochs)
    peak_rss = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_losses = []
        _last_rss = get_current_rss_mb()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [{model_name}]")
        for batch_idx, batch in enumerate(pbar):
            noisy = batch['noisy'].to(memory_format=torch.channels_last)
            clean = batch['clean'].to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            pred = model(noisy)
            loss = F.l1_loss(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()
            loss_val = loss.item()
            train_losses.append(loss_val)
            del batch, noisy, clean, pred, loss

            # Memory check + RSS update every 20 batches (avoids /proc read every batch)
            if (batch_idx + 1) % 20 == 0:
                _last_rss = get_current_rss_mb()
                peak_rss = max(peak_rss, _last_rss)
                if _last_rss > mem_limit_mb:
                    tqdm.write(f"  WARNING: RSS={_last_rss:.0f}MB > limit={mem_limit_mb}MB at batch {batch_idx+1}")
                    _release_memory(full=True)

            pbar.set_postfix({'loss': f"{loss_val:.4f}", 'RSS': f"{_last_rss:.0f}MB"})

        scheduler.step()
        avg_loss = np.mean(train_losses)

        # Stagnation check: warn if within-epoch loss variance is near zero
        if len(train_losses) > 10:
            loss_std = np.std(train_losses)
            if loss_std < 1e-6:
                tqdm.write(f"  WARNING: Loss stagnant (std={loss_std:.2e}). "
                           f"Check data diversity / model gradients.")
            # Also check first vs last 10% of batches
            n10 = max(1, len(train_losses) // 10)
            first_avg = np.mean(train_losses[:n10])
            last_avg = np.mean(train_losses[-n10:])
            if epoch == 1:
                tqdm.write(f"  Loss trajectory: first={first_avg:.4f} -> last={last_avg:.4f} "
                           f"(delta={last_avg-first_avg:+.4f}, std={loss_std:.4f})")

        # Release memory before validation
        _release_memory()

        # Skip validation on non-eval epochs (always eval first, last, and every eval_every)
        if epoch != 1 and epoch != epochs and epoch % eval_every != 0:
            dt = time.time() - t0
            rss = memory_guard(f"epoch {epoch}", mem_limit_mb)
            peak_rss = max(peak_rss, rss)
            print(f"  Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | {dt:.1f}s | {rss:.0f}MB (skip val)")
            continue

        # --- Validate ---
        # Full clinical metrics + predicates on final epoch OR when early stopping
        # will trigger (so comprehensive eval always runs before training ends).
        # Fast mode (PSNR+SSIM only) on intermediate evals for much faster validation.
        is_final = (epoch == epochs) or (epochs_no_improve + 1 >= patience)
        eval_preds = is_final
        fast_only = not is_final
        # When using patch-based validation (SwinIR, MambaIR), skip sliding window
        # since patches are already the right size. For full-res validation (DnCNN),
        # also skip sliding window since DnCNN is fully convolutional.
        # Only use sliding window for KBNet on full-res images.
        needs_sliding_window = model_name in ('kbnet_7m',) and val_patch == 0
        # Use 20-image subset for fast intermediate validation,
        # full validation set for final/early-stopping evaluation
        fast_val_samples = 20 if not is_final else 0
        val_metrics = validate(model, val_loader,
                               predicates_evaluator=predicates_evaluator if eval_preds else None,
                               cached_noisy_preds=cached_noisy_preds,
                               patch_size=patch_size, overlap=16,
                               fast_only=fast_only,
                               use_sliding_window=needs_sliding_window,
                               cached_noisy_metrics=cached_noisy_metrics,
                               max_samples=fast_val_samples)
        # Cache noisy baseline predicates after first epoch (never changes with deterministic patches)
        if cached_noisy_preds is None and 'pred_scores_noisy' in val_metrics:
            cached_noisy_preds = val_metrics['pred_scores_noisy']
        # Cache noisy baseline PSNR/SSIM after first validation
        if cached_noisy_metrics is None and 'psnr_noisy' in val_metrics:
            cached_noisy_metrics = {
                'psnr_noisy': val_metrics['psnr_noisy'],
                'ssim_noisy': val_metrics['ssim_noisy'],
            }

        dt = time.time() - t0
        rss = memory_guard(f"epoch {epoch}", mem_limit_mb)
        peak_rss = max(peak_rss, rss)

        # Print full metrics table
        print_sota_metrics(epoch, model_name, n_params, avg_loss, val_metrics, rss, dt)

        results_log.append({
            'epoch': epoch, 'loss': avg_loss,
            'psnr_denoised': val_metrics['psnr_denoised'],
            'ssim_denoised': val_metrics['ssim_denoised'],
            'cnr_denoised': val_metrics.get('cnr_denoised', 0),
            'tci_denoised': val_metrics.get('tci_denoised', 0),
            'correction_magnitude': val_metrics.get('correction_magnitude', 0),
            'pred_scores': val_metrics.get('pred_scores', {}),
            'rss_mb': rss, 'time_s': dt,
        })

        if val_metrics['psnr_denoised'] > best_psnr:
            best_psnr = val_metrics['psnr_denoised']
            epochs_no_improve = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'psnr': best_psnr,
                'ssim': val_metrics['ssim_denoised'],
                'model_name': model_name,
                'n_params': n_params,
                'val_metrics': val_metrics,
            }, os.path.join(output_dir, 'best.pth'))
            print(f"    *** Saved best: {best_psnr:.2f} dB ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping: no improvement for {patience} evaluations"
                      f" ({patience * eval_every} training epochs)")
                break

        _release_memory()

    # Save training log
    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(results_log, f, indent=2)

    print(f"\n  {model_name} complete. Best PSNR: {best_psnr:.2f} dB | Peak RSS: {peak_rss:.0f} MB")

    # Cleanup: clear optimizer state first to break reference cycles
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    # Clear SwinIR attention mask cache if present
    for module in model.modules():
        if hasattr(module, '_attn_mask_cache'):
            module._attn_mask_cache.clear()
    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds, predicates_evaluator
    _release_memory(full=True)

    rss = memory_guard(f"after cleanup", mem_limit_mb)
    print(f"  RSS after cleanup: {rss:.0f} MB")

    return {'model': model_name, 'best_psnr': best_psnr, 'n_params': n_params}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _setup_cpu_tuning()
    parser = argparse.ArgumentParser(description='Train 7M SOTA baselines (IEEE TMI)')
    parser.add_argument('--train_jsonl', default='pku37_oct_dataset/pku37_real_train.jsonl')
    parser.add_argument('--val_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl')
    parser.add_argument('--methods', nargs='+',
                        default=['dncnn_7m', 'swinir_7m', 'kbnet_7m', 'mambair_7m'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--eval_every', type=int, default=1, help='Validate every N epochs (default: every epoch)')
    parser.add_argument('--max_train', type=int, default=None, help='Max training samples (None=all)')
    parser.add_argument('--max_val', type=int, default=None, help='Max validation samples (None=all)')
    parser.add_argument('--output_dir', default='outputs/sota_7m')
    parser.add_argument('--mem_limit_mb', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("TRAINING 7M SOTA BASELINES (IEEE TMI METRICS)")
    print("=" * 60)
    print(f"  Data: {args.train_jsonl} / {args.val_jsonl}")
    print(f"  Methods: {args.methods}")
    print(f"  Epochs: {args.epochs}, BS: {args.batch_size}, Patch: {args.patch_size}, LR: {args.lr}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Memory limit: {args.mem_limit_mb} MB")
    rss = get_current_rss_mb()
    print(f"  Initial RSS: {rss:.0f} MB")

    all_results = []
    for method in args.methods:
        method_dir = os.path.join(args.output_dir, method)
        try:
            result = train_single_model(
                model_name=method,
                train_jsonl=args.train_jsonl,
                val_jsonl=args.val_jsonl,
                output_dir=method_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                lr=args.lr,
                max_train=args.max_train,
                max_val=args.max_val,
                mem_limit_mb=args.mem_limit_mb,
                patience=args.patience,
                eval_every=args.eval_every,
            )
            all_results.append(result)
        except MemoryError as e:
            print(f"\n  !!! OOM for {method}: {e}")
            all_results.append({'model': method, 'best_psnr': 0.0, 'error': str(e)})
            _release_memory(full=True)
        except Exception as e:
            print(f"\n  !!! Error for {method}: {e}")
            import traceback; traceback.print_exc()
            all_results.append({'model': method, 'best_psnr': 0.0, 'error': str(e)})
            _release_memory(full=True)
        # Aggressive memory cleanup between models
        _release_memory(full=True)
        rss = get_current_rss_mb()
        print(f"  RSS between models: {rss:.0f} MB")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        err = r.get('error', '')
        if err:
            print(f"  {r['model']:<15}: FAILED ({err})")
        else:
            print(f"  {r['model']:<15}: {r['best_psnr']:.2f} dB  ({r['n_params']/1e6:.2f}M params)")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
