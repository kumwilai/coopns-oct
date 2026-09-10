#!/usr/bin/env python3
"""
Diagnose PSNR degradation during training.
Checks if NAFNet baseline is being preserved.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsnd_oct'))

from train_neurosymbolic_denoising import NeuroSymbolicDenoiser, compute_psnr


def main():
    device = torch.device('cpu')

    # Load PKU37 sample
    train_jsonl = 'pku37_oct_dataset/weights_pku37_analysis_train.jsonl'
    with open(train_jsonl) as f:
        sample = json.loads(f.readline())

    clean_path = sample.get('clean_path')
    noisy_path = sample.get('noisy_path')

    # Load images
    clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0
    noisy = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0

    print(f"Clean: {clean_path}")
    print(f"Noisy: {noisy_path}")
    print(f"Shape: {clean.shape}")

    # Resize to 256x256 for model
    from PIL import Image as PILImage
    clean_img = PILImage.fromarray((clean * 255).astype(np.uint8))
    noisy_img = PILImage.fromarray((noisy * 255).astype(np.uint8))
    clean_256 = np.array(clean_img.resize((256, 256), PILImage.BILINEAR), dtype=np.float32) / 255.0
    noisy_256 = np.array(noisy_img.resize((256, 256), PILImage.BILINEAR), dtype=np.float32) / 255.0

    # Compute input PSNR
    noisy_psnr = compute_psnr(
        torch.from_numpy(noisy_256).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(clean_256).unsqueeze(0).unsqueeze(0)
    )
    print(f"\nNoisy input PSNR: {noisy_psnr:.2f} dB")

    # Test 1: Fresh model (pretrained NAFNet, no training)
    print("\n" + "="*60)
    print("TEST 1: Fresh model (pretrained weights)")
    print("="*60)

    model_fresh = NeuroSymbolicDenoiser(
        hidden_channels=48,
        nafnet_width=64,
        nafnet_ckpt='outputs/nafnet_calibrated/nafnet_best.pth',
        physics_ckpt='outputs/physics_v3_dice_v2/stage3_256/best_model.pt',
        freeze_nafnet=True,
    ).to(device)
    model_fresh.eval()

    with torch.no_grad():
        x = torch.from_numpy(noisy_256).float().unsqueeze(0).unsqueeze(0).to(device)
        out_fresh = model_fresh(x, return_symbolic=True)
        denoised_fresh = out_fresh['denoised']
        psnr_fresh = compute_psnr(denoised_fresh, torch.from_numpy(clean_256).unsqueeze(0).unsqueeze(0))

    print(f"Fresh model PSNR: {psnr_fresh:.2f} dB")
    print(f"Improvement over noisy: +{psnr_fresh - noisy_psnr:.2f} dB")

    # Test 2: Check if best_model.pth exists and test it
    best_ckpt = 'outputs/neurosymbolic_denoising/best_model.pth'
    if os.path.exists(best_ckpt):
        print("\n" + "="*60)
        print("TEST 2: Trained model (best_model.pth)")
        print("="*60)

        model_trained = NeuroSymbolicDenoiser(
            hidden_channels=48,
            nafnet_width=64,
            nafnet_ckpt='outputs/nafnet_calibrated/nafnet_best.pth',
            physics_ckpt='outputs/physics_v3_dice_v2/stage3_256/best_model.pt',
            freeze_nafnet=True,
        ).to(device)

        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model_trained.load_state_dict(state, strict=False)
        model_trained.eval()

        print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
        print(f"Checkpoint PSNR: {ckpt.get('psnr', '?')}")

        with torch.no_grad():
            out_trained = model_trained(x, return_symbolic=True)
            denoised_trained = out_trained['denoised']
            psnr_trained = compute_psnr(denoised_trained, torch.from_numpy(clean_256).unsqueeze(0).unsqueeze(0))

        print(f"Trained model PSNR: {psnr_trained:.2f} dB")
        print(f"Improvement over noisy: +{psnr_trained - noisy_psnr:.2f} dB")
        print(f"Difference from fresh: {psnr_trained - psnr_fresh:+.2f} dB")

        if psnr_trained < psnr_fresh:
            print("\n*** WARNING: Trained model has LOWER PSNR than fresh model! ***")
            print("This indicates training is degrading performance.")
            print("\nPossible causes:")
            print("1. Symbolic loss weights too high (pushing away from optimal PSNR)")
            print("2. Overfitting to boundary positions that don't help denoising")
            print("3. NAFNet weights being modified despite freeze_nafnet")
    else:
        print(f"\nNo trained checkpoint found at {best_ckpt}")

    # Test 3: Check NAFNet weights consistency
    print("\n" + "="*60)
    print("TEST 3: NAFNet weight consistency check")
    print("="*60)

    nafnet_ckpt = torch.load('outputs/nafnet_calibrated/nafnet_best.pth', map_location='cpu', weights_only=False)
    nafnet_state = nafnet_ckpt.get('model_state_dict', nafnet_ckpt.get('state_dict', nafnet_ckpt))

    # Compare first layer weights
    fresh_w = model_fresh.state_dict()
    key = 'denoiser.intro.weight'
    if key in fresh_w and key.replace('denoiser.', '') in nafnet_state:
        orig = nafnet_state[key.replace('denoiser.', '')]
        loaded = fresh_w[key]
        if loaded.shape == orig.shape:
            diff = (orig - loaded).abs().mean().item()
            print(f"NAFNet intro weight diff: {diff:.6f}")
            if diff > 1e-5:
                print("*** WARNING: NAFNet weights don't match original! ***")
            else:
                print("NAFNet weights match original checkpoint.")
        else:
            print(f"Shape mismatch: orig {orig.shape}, loaded {loaded.shape}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
1. Use reduced symbolic loss weights (already updated):
   --lambda_anatomical 0.1 --lambda_physics 0.05 --lambda_logic 0.05

2. Use single-frame mode for faster training:
   --single_frame

3. Freeze both NAFNet and boundary model to only train layer heads:
   --freeze_nafnet --freeze_boundary

4. Example optimized command:
   python train_neurosymbolic_denoising.py \\
       --train_jsonl pku37_oct_dataset/weights_pku37_analysis_train.jsonl \\
       --val_jsonl pku37_oct_dataset/weights_pku37_analysis_val.jsonl \\
       --max_train 100 --max_val 20 \\
       --epochs 5 --batch_size 1 \\
       --freeze_nafnet --single_frame \\
       --lambda_anatomical 0.1 --lambda_physics 0.05 --lambda_logic 0.05 \\
       --device cpu
""")


if __name__ == '__main__':
    main()
