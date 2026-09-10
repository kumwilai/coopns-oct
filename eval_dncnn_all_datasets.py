#!/usr/bin/env python3
"""
Evaluate DnCNN-7M checkpoint on PKU37-Test, Duke17, and Duke2013.
Reports all metrics needed for the TMI paper tables.

Memory-optimized for 7.8GB RAM CPU-only system:
  - channels_last for ~1.5x faster Conv2d on CPU via MKL-DNN
  - Aggressive GC + malloc_trim every image
  - Explicit del of all intermediates
"""
import ctypes
import gc
import json
import os
import resource
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

_LIBC = None
def _release_memory():
    gc.collect()
    global _LIBC
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL('libc.so.6')
        _LIBC.malloc_trim(0)
    except Exception:
        pass

def get_rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return 0


# ---------------------------------------------------------------------------
# Dataset - loads from disk per image (no caching to save memory)
# ---------------------------------------------------------------------------
class PairedDataset(Dataset):
    def __init__(self, jsonl_path):
        self.pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                cp = entry.get('clean_path', '')
                np_ = entry.get('noisy_path', '')
                if cp and np_ and os.path.exists(cp) and os.path.exists(np_):
                    self.pairs.append((cp, np_))
        print(f"  Loaded {len(self.pairs)} pairs from {os.path.basename(jsonl_path)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cp, np_ = self.pairs[idx]
        clean = np.array(Image.open(cp)).astype(np.float32) / 255.0
        noisy = np.array(Image.open(np_)).astype(np.float32) / 255.0
        clean = torch.from_numpy(clean).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        return {'clean': clean, 'noisy': noisy}


# ---------------------------------------------------------------------------
# Metrics (matching eval_bm3d_nlm_clinical.py and train_sota_7m.py)
# ---------------------------------------------------------------------------
def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse < 1e-10:
        return 50.0
    return (10.0 * torch.log10(1.0 / mse)).item()


def _gaussian_kernel_2d(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g.unsqueeze(1) * g.unsqueeze(0)
    return k2d.unsqueeze(0).unsqueeze(0)

_SSIM_KERNEL = _gaussian_kernel_2d(11, 1.5)

def compute_ssim(pred, target, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2
    kernel = _SSIM_KERNEL.to(pred.device)
    pred_pad = F.pad(pred, [pad]*4, mode='reflect')
    target_pad = F.pad(target, [pad]*4, mode='reflect')
    mu_p = F.conv2d(pred_pad, kernel)
    mu_t = F.conv2d(target_pad, kernel)
    sigma_p_sq = F.conv2d(pred_pad * pred_pad, kernel) - mu_p * mu_p
    sigma_t_sq = F.conv2d(target_pad * target_pad, kernel) - mu_t * mu_t
    sigma_pt = F.conv2d(pred_pad * target_pad, kernel) - mu_p * mu_t
    del pred_pad, target_pad
    ssim_map = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p_sq + sigma_t_sq + C2))
    val = ssim_map.mean().item()
    del mu_p, mu_t, sigma_p_sq, sigma_t_sq, sigma_pt, ssim_map
    return val


_SOBEL_Y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3)
_SOBEL_X = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
_LAPLACIAN = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3)
_COMBINED_EDGE_KERNEL = torch.cat([_SOBEL_Y, _SOBEL_X, _LAPLACIAN], dim=0)


@torch.inference_mode()
def compute_all_metrics(denoised, noisy, clean):
    """Compute all metrics needed for TMI paper."""
    m = {}

    # --- PSNR / SSIM ---
    m['psnr_noisy'] = compute_psnr(noisy, clean)
    m['psnr_denoised'] = compute_psnr(denoised, clean)
    m['ssim_noisy'] = compute_ssim(noisy, clean)
    m['ssim_denoised'] = compute_ssim(denoised, clean)

    # --- CNR ---
    signal_mask = (clean > clean.mean()).float()
    bg_mask = 1.0 - signal_mask
    sm_sum = signal_mask.sum().clamp(min=1.0)
    bg_sum = bg_mask.sum().clamp(min=1.0)

    noisy_sig = (noisy * signal_mask).sum() / sm_sum
    noisy_bg = (noisy * bg_mask).sum() / bg_sum
    noisy_bg_std = torch.sqrt(((noisy - noisy_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
    m['cnr_noisy'] = ((noisy_sig - noisy_bg) / noisy_bg_std).clamp(-100, 100).item()

    den_sig = (denoised * signal_mask).sum() / sm_sum
    den_bg = (denoised * bg_mask).sum() / bg_sum
    den_bg_std = torch.sqrt(((denoised - den_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
    m['cnr_denoised'] = ((den_sig - den_bg) / den_bg_std).clamp(-100, 100).item()

    m['cnr_delta'] = m['cnr_denoised'] - m['cnr_noisy']
    if abs(m['cnr_noisy']) > 1e-6:
        m['cnr_pct'] = (m['cnr_delta'] / abs(m['cnr_noisy'])) * 100.0
    else:
        m['cnr_pct'] = 0.0

    del signal_mask, bg_mask

    # --- Clinical preservation (ratio to clean GT, 1.0 = perfect) ---
    combined_kernel = _COMBINED_EDGE_KERNEL.to(denoised.device)
    stacked = torch.cat([noisy, denoised, clean], dim=0)  # [3, 1, H, W]
    all_results = F.conv2d(stacked, combined_kernel, padding=1)  # [3, 3, H, W]

    noisy_gy = all_results[0:1, 0:1].abs()
    denoised_gy = all_results[1:2, 0:1].abs()
    clean_gy = all_results[2:3, 0:1].abs()
    noisy_gx = all_results[0:1, 1:2]
    denoised_gx = all_results[1:2, 1:2]
    clean_gx = all_results[2:3, 1:2]
    noisy_lap = all_results[0:1, 2:3].abs()
    denoised_lap = all_results[1:2, 2:3].abs()
    clean_lap = all_results[2:3, 2:3].abs()
    del all_results

    # Contrast (local std ratio)
    mu_all = F.avg_pool2d(stacked, 7, 1, 3)
    std_all = torch.sqrt((F.avg_pool2d(stacked**2, 7, 1, 3) - mu_all**2).clamp(min=1e-8))
    noisy_std, denoised_std, clean_std = std_all.chunk(3, dim=0)
    del stacked, mu_all, std_all
    clean_mean_std = clean_std.mean().clamp(min=1e-4)
    m['contrast_ratio'] = (denoised_std.mean() / clean_mean_std).clamp(0, 10).item()
    m['contrast_ratio_noisy'] = (noisy_std.mean() / clean_mean_std).clamp(0, 10).item()
    del noisy_std, denoised_std, clean_std

    # Boundary (vertical gradient ratio)
    clean_gy_mean = clean_gy.mean().clamp(min=1e-4)
    m['boundary_ratio'] = (denoised_gy.mean() / clean_gy_mean).clamp(0, 10).item()
    m['boundary_ratio_noisy'] = (noisy_gy.mean() / clean_gy_mean).clamp(0, 10).item()

    # Texture (Laplacian ratio)
    clean_lap_mean = clean_lap.mean().clamp(min=1e-4)
    m['texture_ratio'] = (denoised_lap.mean() / clean_lap_mean).clamp(0, 10).item()
    m['texture_ratio_noisy'] = (noisy_lap.mean() / clean_lap_mean).clamp(0, 10).item()
    del noisy_lap, denoised_lap, clean_lap

    # Edge (Sobel magnitude ratio)
    noisy_edge = torch.sqrt(noisy_gx**2 + noisy_gy**2 + 1e-8)
    denoised_edge = torch.sqrt(denoised_gx**2 + denoised_gy**2 + 1e-8)
    clean_edge = torch.sqrt(clean_gx**2 + clean_gy**2 + 1e-8)
    clean_edge_mean = clean_edge.mean().clamp(min=1e-4)
    m['edge_ratio'] = (denoised_edge.mean() / clean_edge_mean).clamp(0, 10).item()
    m['edge_ratio_noisy'] = (noisy_edge.mean() / clean_edge_mean).clamp(0, 10).item()

    # EPI
    clean_ef = clean_edge.view(-1)
    denoised_ef = denoised_edge.view(-1)
    mean_a = clean_ef.mean()
    mean_b = denoised_ef.mean()
    cov = (clean_ef * denoised_ef).mean() - mean_a * mean_b
    var_a = (clean_ef * clean_ef).mean() - mean_a**2
    var_b = (denoised_ef * denoised_ef).mean() - mean_b**2
    m['epi'] = (cov / (var_a * var_b).clamp(min=1e-16).sqrt()).item()

    del noisy_gy, denoised_gy, clean_gy, noisy_gx, denoised_gx, clean_gx
    del noisy_edge, denoised_edge, clean_edge, clean_ef, denoised_ef

    # Correction magnitude
    m['correction_magnitude'] = (denoised - noisy).abs().mean().item()

    # --- ENL (Equivalent Number of Looks) = mu^2 / var in background ---
    eps = 1e-8
    bg_mask_enl = (clean < clean.mean()).float()  # background = below mean
    tissue_mask_enl = 1.0 - bg_mask_enl
    bg_sum_enl = bg_mask_enl.sum().clamp(min=1.0)
    tissue_sum_enl = tissue_mask_enl.sum().clamp(min=1.0)

    # Noisy ENL
    noisy_bg_mean = (noisy * bg_mask_enl).sum() / bg_sum_enl
    noisy_bg_var = ((noisy - noisy_bg_mean)**2 * bg_mask_enl).sum() / bg_sum_enl
    m['enl_noisy'] = (noisy_bg_mean**2 / noisy_bg_var.clamp(min=eps)).item()

    # Denoised ENL
    den_bg_mean = (denoised * bg_mask_enl).sum() / bg_sum_enl
    den_bg_var_enl = ((denoised - den_bg_mean)**2 * bg_mask_enl).sum() / bg_sum_enl
    m['enl_denoised'] = (den_bg_mean**2 / den_bg_var_enl.clamp(min=eps)).item()

    # --- SNR = tissue_mean / background_std ---
    noisy_tissue_mean = (noisy * tissue_mask_enl).sum() / tissue_sum_enl
    m['snr_noisy'] = (noisy_tissue_mean / torch.sqrt(noisy_bg_var.clamp(min=eps))).item()

    den_tissue_mean = (denoised * tissue_mask_enl).sum() / tissue_sum_enl
    m['snr_denoised'] = (den_tissue_mean / torch.sqrt(den_bg_var_enl.clamp(min=eps))).item()

    # --- TCI (Tissue Contrast Index) = tissue_std / tissue_mean ---
    noisy_tissue_var = ((noisy - noisy_tissue_mean)**2 * tissue_mask_enl).sum() / tissue_sum_enl
    m['tci_noisy'] = (torch.sqrt(noisy_tissue_var.clamp(min=eps)) / noisy_tissue_mean.clamp(min=eps)).item()

    den_tissue_var = ((denoised - den_tissue_mean)**2 * tissue_mask_enl).sum() / tissue_sum_enl
    m['tci_denoised'] = (torch.sqrt(den_tissue_var.clamp(min=eps)) / den_tissue_mean.clamp(min=eps)).item()

    # --- Boundary Sharpness (max vertical gradient in boundary regions) ---
    # Use clean vertical gradient to find boundary regions
    clean_gy_recomp = F.conv2d(clean, _SOBEL_Y.to(clean.device), padding=1).abs()
    den_gy_recomp = F.conv2d(denoised, _SOBEL_Y.to(denoised.device), padding=1).abs()
    # Top 10% gradient pixels = boundary region
    threshold = torch.quantile(clean_gy_recomp.view(-1), 0.9)
    boundary_mask = (clean_gy_recomp > threshold).float()
    bm_sum = boundary_mask.sum().clamp(min=1.0)
    m['boundary_sharpness_denoised'] = (den_gy_recomp * boundary_mask).sum().item() / bm_sum.item()
    m['boundary_sharpness_clean'] = (clean_gy_recomp * boundary_mask).sum().item() / bm_sum.item()
    del clean_gy_recomp, den_gy_recomp, boundary_mask, bg_mask_enl, tissue_mask_enl

    # Clinical improvement count (>0.5 of clean GT preserved)
    m['clinical_improved'] = sum([
        1 if m['contrast_ratio'] > 0.5 else 0,
        1 if m['boundary_ratio'] > 0.5 else 0,
        1 if m['texture_ratio'] > 0.5 else 0,
        1 if m['edge_ratio'] > 0.5 else 0,
    ])

    return m


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_dataset(model, jsonl_path, dataset_name):
    ds = PairedDataset(jsonl_path)

    accum = {}
    n = 0
    for idx in tqdm(range(len(ds)), desc=f"  Evaluating {dataset_name}", leave=False):
        batch = ds[idx]
        noisy = batch['noisy'].unsqueeze(0).to(memory_format=torch.channels_last)
        clean = batch['clean'].unsqueeze(0).to(memory_format=torch.channels_last)
        denoised = model(noisy).clamp(0, 1)
        metrics = compute_all_metrics(denoised, noisy, clean)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0) + v
        n += 1
        del batch, noisy, clean, denoised, metrics
        # Aggressive cleanup every image to prevent memory buildup
        if n % 5 == 0:
            _release_memory()

    return {k: v / n for k, v in accum.items()}, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checkpoint_path = 'outputs/sota_7m/dncnn_7m/best.pth'
    datasets = {
        'PKU37-Test': 'pku37_oct_dataset/pku37_real_test.jsonl',
        'Duke17': 'duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl',
        'Duke2013': 'duke_sota_datasets/Duke17_Eval/duke2013_synth_eval.jsonl',
    }

    # Load model
    print(f"Loading DnCNN-7M from {checkpoint_path}")
    from sota.models.dncnn_7m import DnCNN
    model = DnCNN(in_channels=1, out_channels=1, num_layers=17, channels=228, use_checkpoint=False)
    model = model.to(memory_format=torch.channels_last)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}, val PSNR: {ckpt.get('psnr', 0):.2f}")
    del ckpt
    _release_memory()
    print(f"  RSS after model load: {get_rss_mb():.0f} MB")

    all_results = {}
    for name, jsonl in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        metrics, n_samples = evaluate_dataset(model, jsonl, name)
        metrics['n_samples'] = n_samples
        all_results[name] = metrics
        _release_memory()

        # Print results
        print(f"\n  --- {name} ({n_samples} images) ---")
        print(f"  PSNR (noisy):     {metrics['psnr_noisy']:.2f} dB")
        print(f"  PSNR (denoised):  {metrics['psnr_denoised']:.2f} dB  (delta: {metrics['psnr_denoised'] - metrics['psnr_noisy']:+.3f})")
        print(f"  SSIM (noisy):     {metrics['ssim_noisy']:.4f}")
        print(f"  SSIM (denoised):  {metrics['ssim_denoised']:.4f}  (delta: {metrics['ssim_denoised'] - metrics['ssim_noisy']:+.4f})")
        print(f"  CNR  (noisy):     {metrics['cnr_noisy']:.3f}")
        print(f"  CNR  (denoised):  {metrics['cnr_denoised']:.3f}  (delta: {metrics['cnr_delta']:+.3f}, {metrics['cnr_pct']:+.1f}%)")
        print(f"  Clinical:         {int(metrics['clinical_improved'])}/4")
        print(f"    Contrast ratio:  {metrics['contrast_ratio']:.3f}  (noisy: {metrics['contrast_ratio_noisy']:.3f})")
        print(f"    Boundary ratio:  {metrics['boundary_ratio']:.3f}  (noisy: {metrics['boundary_ratio_noisy']:.3f})")
        print(f"    Texture ratio:   {metrics['texture_ratio']:.3f}  (noisy: {metrics['texture_ratio_noisy']:.3f})")
        print(f"    Edge ratio:      {metrics['edge_ratio']:.3f}  (noisy: {metrics['edge_ratio_noisy']:.3f})")
        print(f"  EPI:              {metrics['epi']:.4f}")
        print(f"  ENL (noisy):      {metrics['enl_noisy']:.2f}")
        print(f"  ENL (denoised):   {metrics['enl_denoised']:.2f}  (delta: {metrics['enl_denoised'] - metrics['enl_noisy']:+.2f})")
        print(f"  SNR (noisy):      {metrics['snr_noisy']:.2f}")
        print(f"  SNR (denoised):   {metrics['snr_denoised']:.2f}  (delta: {metrics['snr_denoised'] - metrics['snr_noisy']:+.2f})")
        print(f"  TCI (denoised):   {metrics['tci_denoised']:.4f}  (noisy: {metrics['tci_noisy']:.4f})")
        print(f"  Boundary Sharp:   {metrics['boundary_sharpness_denoised']:.4f}  (clean: {metrics['boundary_sharpness_clean']:.4f})")
        print(f"  Correction mag:   {metrics['correction_magnitude']:.6f}")
        print(f"  RSS: {get_rss_mb():.0f} MB")

        # Save intermediate results
        out_path = 'outputs/sota_7m/dncnn_7m/eval_all_datasets.json'
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'SUMMARY TABLE (DnCNN-7M)':^80}")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'PKU37-Test':>15} {'Duke17':>15} {'Duke2013':>15}")
    print(f"{'-'*80}")
    for key, label in [
        ('psnr_denoised', 'PSNR (dB)'),
        ('ssim_denoised', 'SSIM'),
        ('cnr_denoised', 'CNR'),
        ('cnr_pct', 'dCNR (%)'),
        ('clinical_improved', 'Clinical (N/4)'),
        ('contrast_ratio', 'Contrast ratio'),
        ('boundary_ratio', 'Boundary ratio'),
        ('texture_ratio', 'Texture ratio'),
        ('edge_ratio', 'Edge ratio'),
        ('epi', 'EPI'),
        ('enl_denoised', 'ENL'),
        ('snr_denoised', 'SNR'),
        ('tci_denoised', 'TCI'),
        ('boundary_sharpness_denoised', 'Boundary Sharp.'),
        ('correction_magnitude', 'Corr. Magnitude'),
    ]:
        vals = []
        for ds in ['PKU37-Test', 'Duke17', 'Duke2013']:
            v = all_results[ds][key]
            if key == 'clinical_improved':
                vals.append(f"{int(v)}/4")
            elif key in ('psnr_denoised',):
                vals.append(f"{v:.2f}")
            elif key in ('ssim_denoised', 'epi', 'tci_denoised', 'boundary_sharpness_denoised'):
                vals.append(f"{v:.4f}")
            elif key in ('cnr_pct',):
                vals.append(f"{v:+.1f}%")
            elif key in ('enl_denoised', 'snr_denoised'):
                vals.append(f"{v:.2f}")
            elif key in ('correction_magnitude',):
                vals.append(f"{v:.6f}")
            else:
                vals.append(f"{v:.3f}")
        print(f"{label:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print(f"{'='*80}")
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
