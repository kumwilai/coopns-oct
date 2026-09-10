#!/usr/bin/env python3
"""
Part 2: Deeper analysis -- are the trained corrections at least STRUCTURED
even if the aggregate metrics are similar to random noise?

Additional tests:
  A. Spatial correlation: Do corrections concentrate in specific regions?
  B. Correction vs tissue mask: Are corrections tissue-aware?
  C. Frequency analysis: Is the correction spectrum different from white noise?
  D. PSNR with matched magnitude: If we scale random noise to EXACTLY match
     correction magnitude per-pixel, does the corrector still win on PSNR?
"""

import os, sys, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image

os.chdir('/home/kumwilai/OCT')
sys.path.insert(0, '/home/kumwilai/OCT')

DEVICE = 'cpu'
CHECKPOINT = 'outputs/v8_overcorrect_fix/best_model_cooperative.pth'
VAL_JSONL = 'pku37_oct_dataset/pku37_real_val.jsonl'
NUM_IMAGES = 10

# Load model
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative
model = NeuroSymbolicDenoiserV8Cooperative(backbone_name='nafnet')
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load images
samples = []
with open(VAL_JSONL, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        if 'clean_path' in entry and 'noisy_path' in entry:
            if os.path.exists(entry['clean_path']) and os.path.exists(entry['noisy_path']):
                samples.append(entry)
        if len(samples) >= NUM_IMAGES:
            break

print("=" * 80)
print("PART 2: STRUCTURAL ANALYSIS OF CORRECTIONS vs RANDOM NOISE")
print("=" * 80)
print()

# Accumulators
tissue_corr_mags = []   # Mean |correction| in tissue regions
bg_corr_mags = []       # Mean |correction| in background regions
tissue_ratios = []       # Ratio: tissue correction / background correction

correction_spatial_stds = []  # Spatial std of |correction| (structured = high)
random_spatial_stds = []

# Frequency analysis
correction_high_freq_ratios = []
random_high_freq_ratios = []

# Directional analysis
correction_v_h_ratios = []  # Vertical vs horizontal gradient ratio of corrections

# Per-pixel PSNR: correction vs backbone error
corr_helps_psnr_count = []  # Fraction of pixels where correction reduces error

for i, sample in enumerate(samples):
    clean_img = np.array(Image.open(sample['clean_path'])).astype(np.float32)
    noisy_img = np.array(Image.open(sample['noisy_path'])).astype(np.float32)
    if clean_img.max() > 1.0: clean_img /= 255.0
    if noisy_img.max() > 1.0: noisy_img /= 255.0
    
    clean_t = torch.from_numpy(clean_img).unsqueeze(0).unsqueeze(0)
    noisy_t = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        corrected, backbone_out, info = model(noisy_t, return_details=True)
    
    correction = corrected - backbone_out  # [1,1,H,W]
    abs_correction = correction.abs()
    
    # --- A. Tissue-awareness ---
    # Create tissue mask from clean image (bright = tissue, dark = background)
    tissue_mask = (clean_t > clean_t.mean()).float()
    bg_mask = 1.0 - tissue_mask
    
    tissue_sum = tissue_mask.sum().clamp(min=1.0)
    bg_sum = bg_mask.sum().clamp(min=1.0)
    
    tissue_mag = (abs_correction * tissue_mask).sum() / tissue_sum
    bg_mag = (abs_correction * bg_mask).sum() / bg_sum
    tissue_corr_mags.append(tissue_mag.item())
    bg_corr_mags.append(bg_mag.item())
    tissue_ratios.append((tissue_mag / bg_mag.clamp(min=1e-8)).item())
    
    # --- B. Spatial structure ---
    # Spatially structured corrections should have higher spatial variance
    # (concentrated in certain areas) vs random noise (uniform everywhere)
    corr_spatial_std = abs_correction.std().item()
    rand_noise = torch.randn_like(backbone_out) * correction.std()
    rand_spatial_std = rand_noise.abs().std().item()
    correction_spatial_stds.append(corr_spatial_std)
    random_spatial_stds.append(rand_spatial_std)
    
    # --- C. Frequency analysis ---
    # Compare high-frequency content of corrections vs random noise
    # Use Laplacian as high-freq filter
    laplacian = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32).reshape(1,1,3,3)
    
    corr_hf = F.conv2d(correction, laplacian, padding=1).abs().mean().item()
    corr_lf = correction.abs().mean().item()
    rand_hf = F.conv2d(rand_noise, laplacian, padding=1).abs().mean().item()
    rand_lf = rand_noise.abs().mean().item()
    
    corr_hf_ratio = corr_hf / (corr_lf + 1e-8)
    rand_hf_ratio = rand_hf / (rand_lf + 1e-8)
    correction_high_freq_ratios.append(corr_hf_ratio)
    random_high_freq_ratios.append(rand_hf_ratio)
    
    # --- D. Directional analysis ---
    # OCT images have horizontal layer structure, so meaningful corrections
    # should have more vertical gradient (enhancing layer boundaries)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).reshape(1,1,3,3)
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).reshape(1,1,3,3)
    
    corr_gy = F.conv2d(correction, sobel_y, padding=1).abs().mean().item()
    corr_gx = F.conv2d(correction, sobel_x, padding=1).abs().mean().item()
    vh_ratio = corr_gy / (corr_gx + 1e-8)
    correction_v_h_ratios.append(vh_ratio)
    
    # --- E. Per-pixel error improvement ---
    backbone_error = (backbone_out - clean_t).abs()
    corrected_error = (corrected - clean_t).abs()
    pixels_helped = (corrected_error < backbone_error).float().mean().item()
    corr_helps_psnr_count.append(pixels_helped)
    
    print(f"  Image {i+1}: tissue_corr/bg_corr={tissue_ratios[-1]:.3f}, "
          f"spatial_std(corr/rand)={corr_spatial_std/rand_spatial_std:.3f}, "
          f"HF_ratio(corr/rand)={corr_hf_ratio/rand_hf_ratio:.3f}, "
          f"V/H_ratio={vh_ratio:.3f}, "
          f"pixels_helped={pixels_helped:.1%}")

print()
print("=" * 80)
print("STRUCTURAL ANALYSIS RESULTS")
print("=" * 80)
print()

# A. Tissue awareness
print("A. TISSUE AWARENESS (do corrections focus on tissue regions?)")
print(f"   Average correction magnitude in tissue:     {np.mean(tissue_corr_mags):.6f}")
print(f"   Average correction magnitude in background: {np.mean(bg_corr_mags):.6f}")
print(f"   Tissue/Background ratio:                    {np.mean(tissue_ratios):.3f}")
if np.mean(tissue_ratios) > 1.2:
    print(f"   -> Corrections are TISSUE-FOCUSED (ratio > 1.2)")
elif np.mean(tissue_ratios) > 1.05:
    print(f"   -> Corrections are SLIGHTLY tissue-focused")
else:
    print(f"   -> Corrections are NOT tissue-focused (similar to random)")
print()

# B. Spatial structure
print("B. SPATIAL STRUCTURE (are corrections concentrated or uniform?)")
avg_corr_spatial = np.mean(correction_spatial_stds)
avg_rand_spatial = np.mean(random_spatial_stds)
print(f"   Correction spatial std:     {avg_corr_spatial:.6f}")
print(f"   Random noise spatial std:   {avg_rand_spatial:.6f}")
print(f"   Ratio (>1 = more structured): {avg_corr_spatial/avg_rand_spatial:.3f}")
if avg_corr_spatial / avg_rand_spatial > 1.1:
    print(f"   -> Corrections are MORE SPATIALLY STRUCTURED than random noise")
else:
    print(f"   -> Corrections have SIMILAR spatial structure to random noise")
print()

# C. Frequency
print("C. FREQUENCY CONTENT (high-freq ratio: Laplacian/mean)")
avg_corr_hf = np.mean(correction_high_freq_ratios)
avg_rand_hf = np.mean(random_high_freq_ratios)
print(f"   Correction HF ratio:   {avg_corr_hf:.4f}")
print(f"   Random noise HF ratio: {avg_rand_hf:.4f}")
print(f"   Ratio (lower = more low-freq/smooth): {avg_corr_hf/avg_rand_hf:.3f}")
if avg_corr_hf < avg_rand_hf * 0.9:
    print(f"   -> Corrections are SMOOTHER than random (more structured)")
elif avg_corr_hf > avg_rand_hf * 1.1:
    print(f"   -> Corrections are NOISIER than random (more high-freq)")
else:
    print(f"   -> Corrections have SIMILAR frequency content to random noise")
print()

# D. Directional
print("D. DIRECTIONAL STRUCTURE (V/H gradient ratio of corrections)")
avg_vh = np.mean(correction_v_h_ratios)
print(f"   Average V/H ratio: {avg_vh:.4f}")
print(f"   (Random noise expected: ~1.0, OCT-aware expected: >1.0)")
if avg_vh > 1.1:
    print(f"   -> Corrections have VERTICAL BIAS (layer-boundary aware)")
elif avg_vh < 0.9:
    print(f"   -> Corrections have HORIZONTAL BIAS")
else:
    print(f"   -> Corrections are ISOTROPIC (no directional preference, like random)")
print()

# E. Per-pixel improvement
print("E. PER-PIXEL ERROR IMPROVEMENT")
avg_helped = np.mean(corr_helps_psnr_count)
print(f"   Fraction of pixels where correction reduces error: {avg_helped:.1%}")
print(f"   (Random would give ~50%, meaningful corrections > 50%)")
if avg_helped > 0.55:
    print(f"   -> Corrections HELP more pixels than they hurt")
elif avg_helped > 0.50:
    print(f"   -> Corrections help SLIGHTLY more pixels than random")
else:
    print(f"   -> Corrections do NOT help more than random (<=50%)")
print()

print("=" * 80)
print("OVERALL VERDICT")
print("=" * 80)
print()

structured_signals = 0
total_tests = 5

if np.mean(tissue_ratios) > 1.2: structured_signals += 1
if avg_corr_spatial / avg_rand_spatial > 1.1: structured_signals += 1
if avg_corr_hf < avg_rand_hf * 0.9: structured_signals += 1
if avg_vh > 1.1: structured_signals += 1
if avg_helped > 0.55: structured_signals += 1

print(f"Structural signals found: {structured_signals}/{total_tests}")
if structured_signals >= 4:
    print("The corrections are HIGHLY STRUCTURED and meaningfully different from random noise.")
elif structured_signals >= 2:
    print("The corrections show SOME STRUCTURE beyond random noise,")
    print("but the clinical metric improvements are partially explained by noise artifacts.")
else:
    print("The corrections show MINIMAL STRUCTURE beyond random noise.")
    print("The clinical metric improvements (~10%) appear to be largely a mathematical")
    print("artifact: adding ANY small perturbation to a smooth backbone output increases")
    print("local variance, gradient magnitudes, and edge responses -- regardless of whether")
    print("the perturbation is structured or random.")

