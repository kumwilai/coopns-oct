#!/usr/bin/env python3
"""
Test: Does random noise of the same magnitude as trained correctors
produce similar clinical improvement?

This script compares three outputs:
  1. backbone_out (NAFNet baseline)
  2. corrected (backbone + trained cooperative correctors)
  3. backbone_out + random_noise (same std as actual corrections)

If random noise gives similar clinical metrics improvement, it would
mean the correctors are not learning meaningful structure -- they are
just adding noise. If the trained correctors do significantly better,
the corrections are structurally meaningful.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---- Setup ----
os.chdir('/home/kumwilai/OCT')
sys.path.insert(0, '/home/kumwilai/OCT')

DEVICE = 'cpu'
CHECKPOINT = 'outputs/v8_overcorrect_fix/best_model_cooperative.pth'
VAL_JSONL = 'pku37_oct_dataset/pku37_real_val.jsonl'
NUM_IMAGES = 10
NUM_RANDOM_TRIALS = 5  # Average over multiple random noise samples

print("=" * 80)
print("EXPERIMENT: Random Noise vs Trained Correctors")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Validation data: {VAL_JSONL}")
print(f"Number of images: {NUM_IMAGES}")
print(f"Random trials per image: {NUM_RANDOM_TRIALS}")
print()

# ---- Load Model ----
print("Loading model...")
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative

model = NeuroSymbolicDenoiserV8Cooperative(backbone_name='nafnet')

ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)
model.eval()
print(f"  Loaded from epoch {ckpt['epoch']}, best_score={ckpt['best_score']:.2f}")
print()

# ---- Load Validation Images ----
print("Loading validation images...")
samples = []
with open(VAL_JSONL, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        if 'clean_path' in entry and 'noisy_path' in entry:
            if os.path.exists(entry['clean_path']) and os.path.exists(entry['noisy_path']):
                samples.append(entry)
        if len(samples) >= NUM_IMAGES:
            break

print(f"  Loaded {len(samples)} image pairs")
print()

# ---- Helper: Clinical Metrics ----
def local_std(x, kernel_size=7):
    """Compute local standard deviation."""
    pad = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    var = F.avg_pool2d(x ** 2, kernel_size, stride=1, padding=pad) - mean ** 2
    return torch.sqrt(var.clamp(min=1e-8))

def compute_clinical_metrics(output, clean, label=""):
    """
    Compute 4 clinical preservation metrics (matching validation in train_v8_cooperative.py).
    Each metric is the ratio of (output metric / clean metric).
    
    Returns dict with:
      - contrast_pres: local std preservation (7x7)
      - boundary_pres: vertical gradient preservation (Sobel-Y)
      - texture_pres: high-frequency preservation (Laplacian)
      - edge_pres: edge magnitude preservation (Sobel)
    """
    eps = 1e-4
    
    # 1. Contrast preservation (local std)
    output_std = local_std(output, 7)
    clean_std = local_std(clean, 7)
    clean_std_mean = clean_std.mean().clamp(min=eps)
    contrast_pres = (output_std.mean() / clean_std_mean).clamp(0.0, 10.0).item()
    
    # 2. Boundary preservation (vertical gradient via Sobel-Y)
    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    output_gy = F.conv2d(output, sobel_y, padding=1).abs()
    clean_gy = F.conv2d(clean, sobel_y, padding=1).abs()
    clean_gy_mean = clean_gy.mean().clamp(min=eps)
    boundary_pres = (output_gy.mean() / clean_gy_mean).clamp(0.0, 10.0).item()
    
    # 3. Texture preservation (Laplacian)
    laplacian = torch.tensor([[ 0, -1,  0],
                               [-1,  4, -1],
                               [ 0, -1,  0]], dtype=torch.float32).reshape(1, 1, 3, 3)
    output_lap = F.conv2d(output, laplacian, padding=1).abs()
    clean_lap = F.conv2d(clean, laplacian, padding=1).abs()
    clean_lap_mean = clean_lap.mean().clamp(min=eps)
    texture_pres = (output_lap.mean() / clean_lap_mean).clamp(0.0, 10.0).item()
    
    # 4. Edge preservation (Sobel magnitude)
    sobel_x = torch.tensor([[-1,  0,  1],
                             [-2,  0,  2],
                             [-1,  0,  1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    output_gx = F.conv2d(output, sobel_x, padding=1)
    clean_gx = F.conv2d(clean, sobel_x, padding=1)
    output_edge = torch.sqrt(output_gx**2 + output_gy**2 + 1e-8)
    clean_edge = torch.sqrt(clean_gx**2 + clean_gy**2 + 1e-8)
    clean_edge_mean = clean_edge.mean().clamp(min=eps)
    edge_pres = (output_edge.mean() / clean_edge_mean).clamp(0.0, 10.0).item()
    
    # PSNR
    mse = F.mse_loss(output, clean)
    psnr = (10 * torch.log10(1.0 / (mse + 1e-10))).item()
    
    # SSIM (simplified)
    ssim_val = compute_ssim(output, clean)
    
    return {
        'contrast_pres': contrast_pres,
        'boundary_pres': boundary_pres,
        'texture_pres': texture_pres,
        'edge_pres': edge_pres,
        'psnr': psnr,
        'ssim': ssim_val,
    }

def compute_ssim(x, y, window_size=11):
    """Simple SSIM computation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x2 = F.avg_pool2d(x ** 2, window_size, stride=1, padding=pad) - mu_x ** 2
    sigma_y2 = F.avg_pool2d(y ** 2, window_size, stride=1, padding=pad) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()


# ---- Run Experiment ----
print("Running experiment...")
print("-" * 80)

# Accumulators
metrics_backbone = {k: 0.0 for k in ['contrast_pres', 'boundary_pres', 'texture_pres', 'edge_pres', 'psnr', 'ssim']}
metrics_corrected = {k: 0.0 for k in metrics_backbone}
metrics_random = {k: 0.0 for k in metrics_backbone}

all_correction_magnitudes = []
all_correction_stds = []

n_images = 0

for i, sample in enumerate(samples):
    # Load images
    clean_img = np.array(Image.open(sample['clean_path'])).astype(np.float32)
    noisy_img = np.array(Image.open(sample['noisy_path'])).astype(np.float32)
    if clean_img.max() > 1.0:
        clean_img = clean_img / 255.0
    if noisy_img.max() > 1.0:
        noisy_img = noisy_img / 255.0
    
    clean_t = torch.from_numpy(clean_img).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]
    noisy_t = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        corrected, backbone_out, info = model(noisy_t, return_details=True)
    
    # Measure the actual correction
    correction = corrected - backbone_out
    corr_magnitude = correction.abs().mean().item()
    corr_std = correction.std().item()
    all_correction_magnitudes.append(corr_magnitude)
    all_correction_stds.append(corr_std)
    
    # Compute metrics for backbone
    m_bb = compute_clinical_metrics(backbone_out, clean_t, "backbone")
    for k in metrics_backbone:
        metrics_backbone[k] += m_bb[k]
    
    # Compute metrics for corrected
    m_corr = compute_clinical_metrics(corrected, clean_t, "corrected")
    for k in metrics_corrected:
        metrics_corrected[k] += m_corr[k]
    
    # Compute metrics for random noise (averaged over multiple trials)
    m_rand_accum = {k: 0.0 for k in metrics_random}
    for trial in range(NUM_RANDOM_TRIALS):
        # Generate random noise with SAME std as actual correction
        random_noise = torch.randn_like(backbone_out) * corr_std
        random_corrected = (backbone_out + random_noise).clamp(0, 1)
        m_rand = compute_clinical_metrics(random_corrected, clean_t, "random")
        for k in m_rand_accum:
            m_rand_accum[k] += m_rand[k]
    for k in metrics_random:
        metrics_random[k] += m_rand_accum[k] / NUM_RANDOM_TRIALS
    
    n_images += 1
    
    # Per-image summary
    print(f"  Image {i+1}/{len(samples)}: correction_magnitude={corr_magnitude:.6f}, correction_std={corr_std:.6f}")
    print(f"    Contrast pres:  backbone={m_bb['contrast_pres']:.4f}  corrected={m_corr['contrast_pres']:.4f}  random={m_rand_accum['contrast_pres']/NUM_RANDOM_TRIALS:.4f}")
    print(f"    Edge pres:      backbone={m_bb['edge_pres']:.4f}  corrected={m_corr['edge_pres']:.4f}  random={m_rand_accum['edge_pres']/NUM_RANDOM_TRIALS:.4f}")
    print(f"    PSNR:           backbone={m_bb['psnr']:.2f}  corrected={m_corr['psnr']:.2f}  random={m_rand_accum['psnr']/NUM_RANDOM_TRIALS:.2f}")

# ---- Compute Averages ----
for k in metrics_backbone:
    metrics_backbone[k] /= n_images
    metrics_corrected[k] /= n_images
    metrics_random[k] /= n_images

avg_corr_mag = np.mean(all_correction_magnitudes)
avg_corr_std = np.mean(all_correction_stds)

# ---- Compute Ratios (relative to backbone) ----
def ratio_pct(val, base):
    if abs(base) < 1e-8:
        return 0.0
    return ((val / base) - 1.0) * 100

# ---- Print Results ----
print()
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(f"Average correction magnitude (abs mean): {avg_corr_mag:.6f}")
print(f"Average correction std:                   {avg_corr_std:.6f}")
print(f"Number of images:                         {n_images}")
print(f"Random noise trials per image:            {NUM_RANDOM_TRIALS}")
print()

# Table header
header = f"{'Metric':<25} {'Backbone':>10} {'Corrected':>10} {'Random':>10} {'Corr vs BB':>12} {'Rand vs BB':>12} {'Corr wins?':>12}"
print(header)
print("-" * len(header))

metric_names = {
    'contrast_pres': 'Contrast (local std)',
    'boundary_pres': 'Boundary (v-grad)',
    'texture_pres': 'Texture (Laplacian)',
    'edge_pres': 'Edge (Sobel)',
    'psnr': 'PSNR (dB)',
    'ssim': 'SSIM',
}

for key in ['contrast_pres', 'boundary_pres', 'texture_pres', 'edge_pres', 'psnr', 'ssim']:
    bb = metrics_backbone[key]
    co = metrics_corrected[key]
    ra = metrics_random[key]
    
    if key == 'psnr':
        # For PSNR, show absolute delta
        corr_delta = co - bb
        rand_delta = ra - bb
        corr_str = f"{corr_delta:+.3f} dB"
        rand_str = f"{rand_delta:+.3f} dB"
        # For PSNR, less negative is "better" for correction but we expect trained to be less damaging
        wins = "YES" if corr_delta > rand_delta else "NO"
    elif key == 'ssim':
        corr_delta = (co - bb) * 1000  # in milli-SSIM
        rand_delta = (ra - bb) * 1000
        corr_str = f"{corr_delta:+.3f} mSSIM"
        rand_str = f"{rand_delta:+.3f} mSSIM"
        wins = "YES" if corr_delta > rand_delta else "NO"
    else:
        # For preservation metrics, higher is better (closer to 1.0 = perfect)
        corr_change = ratio_pct(co, bb)
        rand_change = ratio_pct(ra, bb)
        corr_str = f"{corr_change:+.2f}%"
        rand_str = f"{rand_change:+.2f}%"
        wins = "YES" if corr_change > rand_change else "NO"
    
    name = metric_names[key]
    print(f"{name:<25} {bb:>10.4f} {co:>10.4f} {ra:>10.4f} {corr_str:>12} {rand_str:>12} {wins:>12}")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

# Compute summary
clinical_metrics = ['contrast_pres', 'boundary_pres', 'texture_pres', 'edge_pres']
corr_avg_improvement = np.mean([ratio_pct(metrics_corrected[k], metrics_backbone[k]) for k in clinical_metrics])
rand_avg_improvement = np.mean([ratio_pct(metrics_random[k], metrics_backbone[k]) for k in clinical_metrics])

print(f"Average clinical improvement (4 metrics):")
print(f"  Trained correctors:  {corr_avg_improvement:+.2f}%")
print(f"  Random noise:        {rand_avg_improvement:+.2f}%")
print()

if corr_avg_improvement > rand_avg_improvement + 1.0:
    print("CONCLUSION: Trained correctors provide SIGNIFICANTLY BETTER clinical")
    print(f"  improvement than random noise ({corr_avg_improvement:+.2f}% vs {rand_avg_improvement:+.2f}%).")
    print("  The corrections are learning meaningful, structured enhancements.")
elif corr_avg_improvement > rand_avg_improvement:
    print("CONCLUSION: Trained correctors provide SLIGHTLY BETTER clinical")
    print(f"  improvement than random noise ({corr_avg_improvement:+.2f}% vs {rand_avg_improvement:+.2f}%).")
    print("  The margin is small -- corrections may be partially structured.")
else:
    print("CONCLUSION: Random noise provides SIMILAR OR BETTER clinical metrics")
    print(f"  improvement ({rand_avg_improvement:+.2f}%) compared to trained correctors ({corr_avg_improvement:+.2f}%).")
    print("  This suggests the clinical metrics improvement may be largely a noise artifact,")
    print("  not meaningful structural enhancement.")

print()

# PSNR comparison is key
psnr_corr_delta = metrics_corrected['psnr'] - metrics_backbone['psnr']
psnr_rand_delta = metrics_random['psnr'] - metrics_backbone['psnr']
print(f"PSNR impact:")
print(f"  Trained correctors:  {psnr_corr_delta:+.3f} dB")
print(f"  Random noise:        {psnr_rand_delta:+.3f} dB")
if psnr_corr_delta > psnr_rand_delta:
    print(f"  Trained correctors preserve PSNR better ({psnr_corr_delta:+.3f} vs {psnr_rand_delta:+.3f} dB)")
else:
    print(f"  Random noise preserves PSNR better ({psnr_rand_delta:+.3f} vs {psnr_corr_delta:+.3f} dB)")

print()
print("=" * 80)
print("DETAILED: Per-metric breakdown of where trained corrections differ from random")
print("=" * 80)
print()

for key in clinical_metrics:
    name = metric_names[key]
    corr_change = ratio_pct(metrics_corrected[key], metrics_backbone[key])
    rand_change = ratio_pct(metrics_random[key], metrics_backbone[key])
    advantage = corr_change - rand_change
    print(f"{name}:")
    print(f"  Corrected improvement over backbone: {corr_change:+.2f}%")
    print(f"  Random improvement over backbone:    {rand_change:+.2f}%")
    print(f"  Corrector advantage over random:     {advantage:+.2f} percentage points")
    print()

