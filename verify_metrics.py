#!/usr/bin/env python3
"""
Quick verification: run all methods on the SAME image, compute metrics
with the SAME function, print side-by-side.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative

# Use the SAME metrics function as BM3D/NLM eval
from eval_bm3d_nlm_clinical import compute_all_metrics, compute_psnr, compute_ssim
from eval_bm3d_nlm_clinical import denoise_bm3d_img, denoise_nlm_img


def main():
    device = torch.device('cpu')

    # Load one test image
    test_jsonl = 'pku37_oct_dataset/pku37_real_test.jsonl'
    with open(test_jsonl) as f:
        entry = json.loads(f.readline().strip())
    noisy_path, clean_path = entry['noisy_path'], entry['clean_path']

    noisy_np = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
    clean_np = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0
    print(f"Image: {noisy_path}")
    print(f"Shape: {noisy_np.shape}")

    # --- BM3D ---
    bm3d_np = denoise_bm3d_img(noisy_np)
    bm3d_t = torch.from_numpy(bm3d_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    noisy_t = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    clean_t = torch.from_numpy(clean_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    bm3d_m = compute_all_metrics(bm3d_t, noisy_t, clean_t)

    # --- NLM ---
    nlm_np = denoise_nlm_img(noisy_np)
    nlm_t = torch.from_numpy(nlm_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    nlm_m = compute_all_metrics(nlm_t, noisy_t, clean_t)

    # --- NAFNet + CoopNS ---
    print("Loading model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone='benchmarks/pretrained/nafnet_backbone.pth'
    )
    ckpt = torch.load('benchmarks/pretrained/best_model_cooperative.pth',
                       map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    with torch.no_grad():
        corrected, backbone_out, info = model(noisy_t)

    # Use backbone_out as "denoised" for NAFNet metrics
    nafnet_m = compute_all_metrics(backbone_out, noisy_t, clean_t)
    # Use corrected as "denoised" for CoopNS metrics
    coopns_m = compute_all_metrics(corrected, noisy_t, clean_t)

    # Also compute skimage SSIM for cross-check
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        sk_bm3d = sk_ssim(bm3d_np, clean_np, data_range=1.0)
        sk_nlm = sk_ssim(nlm_np, clean_np, data_range=1.0)
        sk_naf = sk_ssim(backbone_out.squeeze().numpy(), clean_np, data_range=1.0)
        sk_coop = sk_ssim(corrected.squeeze().numpy(), clean_np, data_range=1.0)
        has_skimage = True
    except ImportError:
        has_skimage = False

    # Print comparison
    print(f"\n{'='*80}")
    print(f"{'Metric':<25} {'BM3D':>10} {'NLM':>10} {'NAFNet':>10} {'CoopNS':>10}")
    print(f"{'='*80}")

    for key in ['psnr_denoised', 'ssim_denoised', 'cnr_denoised',
                'contrast_denoised', 'boundary_denoised',
                'texture_denoised', 'edge_denoised',
                'contrast_noisy', 'boundary_noisy', 'texture_noisy', 'edge_noisy']:
        print(f"{key:<25} {bm3d_m[key]:>10.4f} {nlm_m[key]:>10.4f} "
              f"{nafnet_m[key]:>10.4f} {coopns_m[key]:>10.4f}")

    if has_skimage:
        print(f"\n{'skimage SSIM':<25} {sk_bm3d:>10.4f} {sk_nlm:>10.4f} "
              f"{sk_naf:>10.4f} {sk_coop:>10.4f}")

    # Also print the ratio values
    print(f"\n{'contrast_ratio':<25} {bm3d_m['contrast_ratio']:>10.4f} {nlm_m['contrast_ratio']:>10.4f} "
          f"{nafnet_m['contrast_ratio']:>10.4f} {coopns_m['contrast_ratio']:>10.4f}")
    print(f"{'boundary_ratio':<25} {bm3d_m['boundary_ratio']:>10.4f} {nlm_m['boundary_ratio']:>10.4f} "
          f"{nafnet_m['boundary_ratio']:>10.4f} {coopns_m['boundary_ratio']:>10.4f}")
    print(f"{'texture_ratio':<25} {bm3d_m['texture_ratio']:>10.4f} {nlm_m['texture_ratio']:>10.4f} "
          f"{nafnet_m['texture_ratio']:>10.4f} {coopns_m['texture_ratio']:>10.4f}")
    print(f"{'edge_ratio':<25} {bm3d_m['edge_ratio']:>10.4f} {nlm_m['edge_ratio']:>10.4f} "
          f"{nafnet_m['edge_ratio']:>10.4f} {coopns_m['edge_ratio']:>10.4f}")

    # Also check: what does the correction magnitude look like
    corr_mag = (corrected - backbone_out).abs().mean().item()
    print(f"\n  Correction magnitude: {corr_mag:.6f}")
    print(f"  Backbone output range: [{backbone_out.min():.4f}, {backbone_out.max():.4f}]")
    print(f"  Corrected output range: [{corrected.min():.4f}, {corrected.max():.4f}]")

    # Visual check: compare local std maps
    def local_std(img_t):
        mu = F.avg_pool2d(img_t, 7, 1, 3)
        return torch.sqrt((F.avg_pool2d(img_t**2, 7, 1, 3) - mu**2).clamp(min=1e-8))

    print(f"\n  Local std (mean):")
    print(f"    Clean:    {local_std(clean_t).mean().item():.6f}")
    print(f"    Noisy:    {local_std(noisy_t).mean().item():.6f}")
    print(f"    BM3D:     {local_std(bm3d_t).mean().item():.6f}")
    print(f"    NLM:      {local_std(nlm_t).mean().item():.6f}")
    print(f"    NAFNet:   {local_std(backbone_out).mean().item():.6f}")
    print(f"    CoopNS:   {local_std(corrected).mean().item():.6f}")


if __name__ == '__main__':
    main()
