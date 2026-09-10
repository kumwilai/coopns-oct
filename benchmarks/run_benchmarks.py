#!/usr/bin/env python3
"""
SOTA 7M-Parameter Baseline Benchmarks for OCT Denoising (IEEE TMI)

Trains and evaluates DnCNN-7M, SwinIR-7M, KBNet-7M, MambaIR-7M on PKU37.
All models ~7M params to match the NAFNet backbone for fair comparison.

GPU-optimized: auto-detects CUDA, uses mixed precision (AMP), batch_size=16.
Produces results.json and LaTeX table for direct paper inclusion.

Usage:
    python run_benchmarks.py --data_dir pku37_oct_dataset --epochs 50
    python run_benchmarks.py --methods dncnn_7m swinir_7m --epochs 30
    python run_benchmarks.py --eval_only  # evaluate existing checkpoints
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  Device: {name} ({mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        n_threads = os.cpu_count() or 2
        torch.set_num_threads(n_threads)
        print(f"  Device: CPU ({n_threads} threads)")
        print(f"  WARNING: CPU training is very slow for 7M-param models.")
    return device


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PKU37Dataset(Dataset):
    """PKU37 real noise dataset with in-memory caching."""

    def __init__(self, jsonl_path, data_dir, patch_size=96, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train

        self.clean_imgs = []
        self.noisy_imgs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'clean_path' not in entry or 'noisy_path' not in entry:
                    continue
                # Remap absolute paths to relative (portable across machines)
                clean_rel = entry['clean_path'].split('pku37_oct_dataset/')[-1]
                noisy_rel = entry['noisy_path'].split('pku37_oct_dataset/')[-1]
                clean_path = os.path.join(data_dir, clean_rel)
                noisy_path = os.path.join(data_dir, noisy_rel)
                if os.path.exists(clean_path) and os.path.exists(noisy_path):
                    self.clean_imgs.append(np.array(Image.open(clean_path)))
                    self.noisy_imgs.append(np.array(Image.open(noisy_path)))

        est_mb = sum(c.nbytes + n.nbytes for c, n in
                     zip(self.clean_imgs, self.noisy_imgs)) / 1e6
        split = "train" if is_train else "eval"
        print(f"    {len(self.clean_imgs)} images ({est_mb:.0f} MB) [{split}]")

    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        c = self.clean_imgs[idx]
        n = self.noisy_imgs[idx]
        if self.patch_size > 0 and self.is_train:
            H, W = c.shape
            if H > self.patch_size and W > self.patch_size:
                y = torch.randint(0, H - self.patch_size, (1,)).item()
                x = torch.randint(0, W - self.patch_size, (1,)).item()
                c = c[y:y+self.patch_size, x:x+self.patch_size]
                n = n[y:y+self.patch_size, x:x+self.patch_size]
        clean = torch.from_numpy(c.astype(np.float32)).unsqueeze(0)
        noisy = torch.from_numpy(n.astype(np.float32)).unsqueeze(0)
        if clean.max() > 1.0:
            clean = clean / 255.0
        if noisy.max() > 1.0:
            noisy = noisy / 255.0
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                clean = torch.flip(clean, dims=[2])
                noisy = torch.flip(noisy, dims=[2])
            if torch.rand(1).item() > 0.5:
                clean = torch.flip(clean, dims=[1])
                noisy = torch.flip(noisy, dims=[1])
            clean = clean.contiguous()
            noisy = noisy.contiguous()
        return {'clean': clean, 'noisy': noisy}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_model(name):
    name = name.lower()
    if name == 'dncnn_7m':
        from models.dncnn_7m import DnCNN
        return DnCNN(in_channels=1, out_channels=1, num_layers=17, channels=228,
                     use_checkpoint=False)
    elif name == 'swinir_7m':
        from models.swinir_7m import SwinIR
        return SwinIR(in_channels=1, out_channels=1, embed_dim=138,
                      depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6],
                      window_size=8, mlp_ratio=2.0)
    elif name == 'kbnet_7m':
        from models.kbnet_7m import KBNet
        return KBNet(in_channels=1, out_channels=1, width=32,
                     middle_blk_num=10, enc_blk_nums=[2,2,4], dec_blk_nums=[2,2,2],
                     nset=32, gc=1, ffn_scale=2, use_checkpoint=False)
    elif name == 'mambair_7m':
        from models.mambair_7m import MambaIR
        return MambaIR(in_channels=1, out_channels=1, embed_dim=101,
                       depths=[6,6,6,6,6,6], d_state=16, d_conv=3, expand=2.,
                       drop_path_rate=0.1, use_checkpoint=False)
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Metrics (IEEE TMI, matches cooperative training exactly)
# ---------------------------------------------------------------------------
def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse < 1e-10:
        return 50.0
    return (10.0 * torch.log10(1.0 / mse)).item()


_SSIM_KERNEL = None
def _get_ssim_kernel(device):
    global _SSIM_KERNEL
    if _SSIM_KERNEL is None or _SSIM_KERNEL.device != device:
        coords = torch.arange(11, dtype=torch.float32) - 5
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        k2d = g.unsqueeze(1) * g.unsqueeze(0)
        _SSIM_KERNEL = k2d.unsqueeze(0).unsqueeze(0).to(device)
    return _SSIM_KERNEL


def compute_ssim(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    pad = 5
    kernel = _get_ssim_kernel(pred.device)
    pp = F.pad(pred, [pad]*4, mode='reflect')
    tp = F.pad(target, [pad]*4, mode='reflect')
    mu_p = F.conv2d(pp, kernel)
    mu_t = F.conv2d(tp, kernel)
    sigma_p_sq = F.conv2d(pp * pp, kernel) - mu_p ** 2
    sigma_t_sq = F.conv2d(tp * tp, kernel) - mu_t ** 2
    sigma_pt = F.conv2d(pp * tp, kernel) - mu_p * mu_t
    ssim_map = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()


_SOBEL_Y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3)
_SOBEL_X = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
_LAPLACIAN = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3)


@torch.no_grad()
def compute_all_metrics(denoised, noisy, clean):
    """All IEEE TMI metrics matching cooperative training."""
    device = denoised.device
    m = {}

    # PSNR / SSIM
    m['psnr_noisy'] = compute_psnr(noisy, clean)
    m['psnr_denoised'] = compute_psnr(denoised, clean)
    m['psnr_delta'] = m['psnr_denoised'] - m['psnr_noisy']
    m['ssim_noisy'] = compute_ssim(noisy, clean)
    m['ssim_denoised'] = compute_ssim(denoised, clean)
    m['ssim_delta'] = m['ssim_denoised'] - m['ssim_noisy']

    # CNR
    signal_mask = (clean > clean.mean()).float()
    bg_mask = 1.0 - signal_mask
    s_sum = signal_mask.sum().clamp(min=1)
    b_sum = bg_mask.sum().clamp(min=1)
    n_sig = (noisy * signal_mask).sum() / s_sum
    n_bg = (noisy * bg_mask).sum() / b_sum
    n_std = torch.sqrt(((noisy - n_bg)**2 * bg_mask).sum() / b_sum + 1e-8).clamp(min=1e-4)
    m['cnr_noisy'] = ((n_sig - n_bg) / n_std).clamp(-100, 100).item()
    d_sig = (denoised * signal_mask).sum() / s_sum
    d_bg = (denoised * bg_mask).sum() / b_sum
    d_std = torch.sqrt(((denoised - d_bg)**2 * bg_mask).sum() / b_sum + 1e-8).clamp(min=1e-4)
    m['cnr_denoised'] = ((d_sig - d_bg) / d_std).clamp(-100, 100).item()
    m['cnr_delta'] = m['cnr_denoised'] - m['cnr_noisy']
    abs_n, abs_d = abs(m['cnr_noisy']), abs(m['cnr_denoised'])
    m['cnr_improvement'] = ((abs_d / max(abs_n, 1e-4)) - 1.0) * 100 if abs_n > 1e-4 else 0
    del signal_mask, bg_mask

    # Gradient-based metrics (batched)
    sobel_y = _SOBEL_Y.to(device)
    sobel_x = _SOBEL_X.to(device)
    laplacian = _LAPLACIAN.to(device)
    stacked = torch.cat([noisy, denoised, clean], dim=0)
    all_gy = F.conv2d(stacked, sobel_y, padding=1).abs()
    all_gx = F.conv2d(stacked, sobel_x, padding=1)
    all_lap = F.conv2d(stacked, laplacian, padding=1).abs()
    n_gy, d_gy, c_gy = all_gy.chunk(3, dim=0)
    n_gx, d_gx, c_gx = all_gx.chunk(3, dim=0)
    n_lap, d_lap, c_lap = all_lap.chunk(3, dim=0)
    del stacked, all_gy, all_gx, all_lap

    # TCI
    c_gy_mean = c_gy.mean().clamp(min=1e-4)
    m['tci_noisy'] = (n_gy.mean() / c_gy_mean).clamp(0, 10).item()
    m['tci_denoised'] = (d_gy.mean() / c_gy_mean).clamp(0, 10).item()

    # Clinical preservation: Contrast
    stk = torch.cat([noisy, denoised, clean], dim=0)
    mu = F.avg_pool2d(stk, 7, 1, 3)
    std_map = torch.sqrt((F.avg_pool2d(stk**2, 7, 1, 3) - mu**2).clamp(min=1e-8))
    n_s, d_s, c_s = std_map.chunk(3, dim=0)
    del stk, mu, std_map
    c_s_mean = c_s.mean().clamp(min=1e-4)
    m['contrast_noisy'] = (n_s.mean() / c_s_mean).clamp(0, 10).item()
    m['contrast_denoised'] = (d_s.mean() / c_s_mean).clamp(0, 10).item()
    m['contrast_ratio'] = m['contrast_denoised'] / max(m['contrast_noisy'], 1e-4)
    del n_s, d_s, c_s

    # Boundary
    m['boundary_noisy'] = (n_gy.mean() / c_gy_mean).clamp(0, 10).item()
    m['boundary_denoised'] = (d_gy.mean() / c_gy_mean).clamp(0, 10).item()
    m['boundary_ratio'] = m['boundary_denoised'] / max(m['boundary_noisy'], 1e-4)

    # Texture
    c_lap_mean = c_lap.mean().clamp(min=1e-4)
    m['texture_noisy'] = (n_lap.mean() / c_lap_mean).clamp(0, 10).item()
    m['texture_denoised'] = (d_lap.mean() / c_lap_mean).clamp(0, 10).item()
    m['texture_ratio'] = m['texture_denoised'] / max(m['texture_noisy'], 1e-4)

    # Edge
    n_edge = torch.sqrt(n_gx**2 + n_gy**2 + 1e-8)
    d_edge = torch.sqrt(d_gx**2 + d_gy**2 + 1e-8)
    c_edge = torch.sqrt(c_gx**2 + c_gy**2 + 1e-8)
    c_edge_mean = c_edge.mean().clamp(min=1e-4)
    m['edge_noisy'] = (n_edge.mean() / c_edge_mean).clamp(0, 10).item()
    m['edge_denoised'] = (d_edge.mean() / c_edge_mean).clamp(0, 10).item()
    m['edge_ratio'] = m['edge_denoised'] / max(m['edge_noisy'], 1e-4)

    # EPI
    def _corr(a, b):
        return ((a - a.mean()) / a.std().clamp(min=1e-4) *
                (b - b.mean()) / b.std().clamp(min=1e-4)).mean().item()
    m['epi_noisy'] = _corr(c_edge.view(-1), n_edge.view(-1))
    m['epi_denoised'] = _corr(c_edge.view(-1), d_edge.view(-1))

    # Boundary sharpness
    c_gy_max = c_gy.max().clamp(min=1e-4)
    m['bs_noisy'] = (n_gy.max() / c_gy_max).clamp(0, 10).item()
    m['bs_denoised'] = (d_gy.max() / c_gy_max).clamp(0, 10).item()

    # Correction magnitude
    m['correction_magnitude'] = (denoised - noisy).abs().mean().item()

    # Clinical improvement count
    m['clinical_improved'] = sum([
        m['contrast_ratio'] > 1.0,
        m['boundary_ratio'] > 1.0,
        m['texture_ratio'] > 1.0,
        m['edge_ratio'] > 1.0,
    ])

    return m


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def sliding_window_inference(model, image, patch_size=96, overlap=16, device=None):
    B, C, H, W = image.shape
    stride = patch_size - overlap
    if H <= patch_size and W <= patch_size:
        return model(image).clamp(0, 1)

    def _pos(length, ps, st):
        starts = list(range(0, length - ps + 1, st))
        if starts[-1] + ps < length:
            starts.append(length - ps)
        return starts

    output = torch.zeros_like(image)
    count = torch.zeros_like(image)
    positions = [(y, x) for y in _pos(H, patch_size, stride)
                        for x in _pos(W, patch_size, stride)]
    batch_sz = 16 if (device and device.type == 'cuda') else 1
    for i in range(0, len(positions), batch_sz):
        bp = positions[i:i+batch_sz]
        patches = torch.cat([image[:, :, y:y+patch_size, x:x+patch_size]
                             for y, x in bp], dim=0)
        preds = model(patches).clamp(0, 1)
        for j, (y, x) in enumerate(bp):
            output[:, :, y:y+patch_size, x:x+patch_size] += preds[j:j+1]
            count[:, :, y:y+patch_size, x:x+patch_size] += 1.0
        del patches, preds
    return output / count


@torch.no_grad()
def infer(model, noisy, model_name, patch_size, device):
    """Run inference: direct for FCN models, sliding window for others."""
    if model_name in ('dncnn_7m',):
        return model(noisy).clamp(0, 1)
    return sliding_window_inference(model, noisy, patch_size=patch_size,
                                    overlap=16, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, model_name, patch_size=96):
    model.eval()
    accum = {}
    n = 0
    for batch in tqdm(loader, desc=f"  Eval {model_name}", leave=False):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        denoised = infer(model, noisy, model_name, patch_size, device)
        metrics = compute_all_metrics(denoised, noisy, clean)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0) + v
        n += 1
        del noisy, clean, denoised, metrics
    return {k: v / n for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------
def train_model(model_name, train_loader, val_loader, device, output_dir,
                epochs=50, lr=5e-5, patience=10, patch_size=96, use_amp=True):
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")

    model = build_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = use_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled)

    os.makedirs(output_dir, exist_ok=True)
    best_psnr = 0.0
    no_improve = 0
    log = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                pred = model(noisy)
                loss = F.l1_loss(pred, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            del noisy, clean, pred, loss
        scheduler.step()
        avg_loss = np.mean(losses)

        # --- Quick validation (PSNR + SSIM) ---
        model.eval()
        psnrs, ssims = [], []
        for batch in val_loader:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            with torch.no_grad():
                denoised = infer(model, noisy, model_name, patch_size, device)
            psnrs.append(compute_psnr(denoised, clean))
            ssims.append(compute_ssim(denoised, clean))
            del noisy, clean, denoised
        val_psnr = np.mean(psnrs)
        val_ssim = np.mean(ssims)
        dt = time.time() - t0

        print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"PSNR={val_psnr:.2f} dB | SSIM={val_ssim:.4f} | {dt:.1f}s")

        log.append({'epoch': epoch, 'loss': float(avg_loss),
                    'psnr': float(val_psnr), 'ssim': float(val_ssim),
                    'time': float(dt)})

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            no_improve = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch, 'psnr': best_psnr, 'ssim': val_ssim,
                'model_name': model_name, 'n_params': n_params,
            }, os.path.join(output_dir, 'best.pth'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Best PSNR: {best_psnr:.2f} dB")

    del optimizer, scheduler, scaler
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'model': model_name, 'n_params': n_params, 'best_psnr': best_psnr}


# ---------------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------------
def final_evaluate(model_name, test_loader, device, output_dir, patch_size=96):
    ckpt_path = os.path.join(output_dir, 'best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint for {model_name}, skipping")
        return None

    model = build_model(model_name)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)

    metrics = evaluate(model, test_loader, device, model_name, patch_size)
    metrics['n_params'] = ckpt['n_params']
    metrics['best_epoch'] = ckpt['epoch']
    metrics['model'] = model_name

    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_results_table(all_results):
    print(f"\n{'='*100}")
    print(f"  BENCHMARK RESULTS — PKU37 Test Set (173 images)")
    print(f"{'='*100}")
    print(f"  {'Method':<12} {'Params':>8} {'PSNR':>9} {'dPSNR':>8} "
          f"{'SSIM':>8} {'CNR':>8} {'dCNR':>8} {'TCI':>7} {'EPI':>8} "
          f"{'Clinical':>9}")
    print(f"  {'-'*94}")
    for r in all_results:
        if r is None:
            continue
        print(f"  {r['model']:<12} {r['n_params']/1e6:>7.2f}M "
              f"{r['psnr_denoised']:>9.2f} {r['psnr_delta']:>+8.2f} "
              f"{r['ssim_denoised']:>8.4f} {r['cnr_denoised']:>8.2f} "
              f"{r['cnr_delta']:>+8.2f} {r['tci_denoised']:>7.3f} "
              f"{r['epi_denoised']:>8.4f} "
              f"{r.get('clinical_improved', 0):>5.0f}/4")
    print(f"{'='*100}\n")


def print_latex_table(all_results):
    print("% Copy into your LaTeX paper:")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\toprule")
    print(r"Method & Params & PSNR & $\Delta$PSNR & SSIM & CNR & "
          r"$\Delta$CNR & TCI & Clinical \\")
    print(r"\midrule")
    for r in all_results:
        if r is None:
            continue
        name = r['model'].replace('_7m', '').replace('_', r'\_')
        clinical = f"{r.get('clinical_improved', 0):.0f}/4"
        print(f"{name} & {r['n_params']/1e6:.2f}M & "
              f"{r['psnr_denoised']:.2f} & {r['psnr_delta']:+.2f} & "
              f"{r['ssim_denoised']:.4f} & {r['cnr_denoised']:.2f} & "
              f"{r['cnr_delta']:+.2f} & {r['tci_denoised']:.3f} & "
              f"{clinical} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='SOTA 7M Baseline Benchmarks for OCT Denoising (IEEE TMI)')
    parser.add_argument('--data_dir', default='pku37_oct_dataset',
                        help='Path to pku37_oct_dataset/ folder')
    parser.add_argument('--methods', nargs='+',
                        default=['dncnn_7m', 'swinir_7m', 'kbnet_7m', 'mambair_7m'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, evaluate existing checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    print("=" * 70)
    print("  SOTA 7M BASELINE BENCHMARKS — OCT Denoising (IEEE TMI)")
    print("=" * 70)
    device = setup_device()
    use_amp = not args.no_amp

    # Check data
    train_jsonl = os.path.join(args.data_dir, 'pku37_real_train.jsonl')
    val_jsonl = os.path.join(args.data_dir, 'pku37_real_val.jsonl')
    test_jsonl = os.path.join(args.data_dir, 'pku37_real_test.jsonl')
    for p in [train_jsonl, val_jsonl, test_jsonl]:
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found.")
            print(f"  Copy pku37_oct_dataset/ into this directory.")
            sys.exit(1)

    # --- Training ---
    if not args.eval_only:
        print(f"\n  Loading data...")
        train_ds = PKU37Dataset(train_jsonl, args.data_dir,
                                patch_size=args.patch_size, is_train=True)
        val_ds = PKU37Dataset(val_jsonl, args.data_dir,
                              patch_size=0, is_train=False)
        nw = args.num_workers if device.type == 'cuda' else 0
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=nw, drop_last=True,
                                  pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        print(f"  epochs={args.epochs}, bs={args.batch_size}, "
              f"patch={args.patch_size}, lr={args.lr}")
        print(f"  AMP: {'on' if (use_amp and device.type == 'cuda') else 'off'}")

        for method in args.methods:
            method_dir = os.path.join(args.output_dir, method)
            try:
                train_model(method, train_loader, val_loader, device,
                           method_dir, epochs=args.epochs, lr=args.lr,
                           patience=args.patience, patch_size=args.patch_size,
                           use_amp=use_amp)
            except Exception as e:
                print(f"  ERROR training {method}: {e}")
                import traceback; traceback.print_exc()

        del train_ds, val_ds, train_loader, val_loader
        gc.collect()

    # --- Test evaluation ---
    print(f"\n{'='*70}")
    print(f"  FINAL EVALUATION ON TEST SET (173 images)")
    print(f"{'='*70}")
    test_ds = PKU37Dataset(test_jsonl, args.data_dir, patch_size=0, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    all_results = []
    for method in args.methods:
        method_dir = os.path.join(args.output_dir, method)
        result = final_evaluate(method, test_loader, device, method_dir,
                               patch_size=args.patch_size)
        if result:
            all_results.append(result)

    # Save + print
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print_results_table(all_results)
    print_latex_table(all_results)
    print(f"  Saved: {results_path}")


if __name__ == '__main__':
    main()
