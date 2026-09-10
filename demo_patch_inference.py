#!/usr/bin/env python3
"""
Demonstration of patch-based inference for large images.

This script shows how to use the forward_patch_based method to denoise
large images without running out of memory.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add project to path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "nsnd_oct"))

from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer
from nsnd.models.noise_conditioner import SpatialBasisModulator


def load_models(base_ckpt, analyzer_ckpt, modulator_ckpt, device='cuda'):
    """Load the trained models."""
    # Load Analyzer
    print("Loading Analyzer...")
    analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=False).to(device)
    analyzer_state = torch.load(analyzer_ckpt, map_location=device, weights_only=False)
    analyzer.load_state_dict(analyzer_state["state_dict"], strict=False)
    analyzer.eval()
    for p in analyzer.parameters():
        p.requires_grad = False

    # Load NAFNet
    print("Loading NAFNet...")
    CONDITIONER_DIM = 32
    model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        middle_blk_num=2,
        cond_dim=CONDITIONER_DIM,
        condition_middle=True,
        condition_decoders=True,
        use_spatial_cue=True
    ).to(device)

    # Load Modulator
    print("Loading Modulator...")
    modulator = SpatialBasisModulator(
        feature_channels=128,
        stage_channels=model.dbm_stage_channels,
        num_noise_types=4,
        hidden_channels=64,
        alpha=1.0,
        gate_floor=0.3,
        basis_init_std=0.1,
    ).to(device)

    # Load trained weights
    checkpoint = torch.load(modulator_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    modulator.load_state_dict(checkpoint["modulator"], strict=True)

    model.eval()
    modulator.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in modulator.parameters():
        p.requires_grad = False

    return model, analyzer, modulator


def denoise_large_image(image_path, model, analyzer, modulator, device='cuda',
                       patch_size=64, stride=32, alpha=1.0):
    """
    Denoise a large image using patch-based processing.

    Args:
        image_path: Path to the noisy image
        model: NAFNetFullFiLM model
        analyzer: HybridCNNSymbolicAnalyzer model
        modulator: SpatialBasisModulator model
        device: Device to use
        patch_size: Size of patches (default: 64)
        stride: Stride between patches (default: 32 for 50% overlap)
        alpha: Modulation strength (default: 1.0)

    Returns:
        Denoised image as numpy array
    """
    # Load image
    print(f"Loading image: {image_path}")
    noisy = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    H, W = noisy.shape
    print(f"Image size: {H}x{W}")

    # Convert to tensor
    noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Get analysis from analyzer
        print("Analyzing noise characteristics...")
        weights_dict, features = analyzer(noisy_t, return_feature_map=True, return_predicates=True)
        feature_map = features.get("feature_map")

        # Prepare global weights
        global_weights = torch.stack([
            weights_dict["speckle"],
            weights_dict["banding"],
            weights_dict["gaussian"],
            weights_dict["shot"],
        ], dim=1)

        confidence = weights_dict.get("_confidence", torch.ones(1, device=device))

        print(f"Detected noise composition: "
              f"Speckle={weights_dict['speckle'].item():.3f}, "
              f"Banding={weights_dict['banding'].item():.3f}, "
              f"Gaussian={weights_dict['gaussian'].item():.3f}, "
              f"Shot={weights_dict['shot'].item():.3f}")

        # Get spatial modulation
        print("Computing spatial modulation...")
        spatial_map, gate, basis = modulator(
            feature_map,
            global_weights=global_weights,
            confidence=confidence,
            noisy=noisy_t,
        )

        # Denoise using patch-based inference
        print(f"Denoising with patch-based inference (patch_size={patch_size}, stride={stride})...")
        if H <= patch_size and W <= patch_size:
            # Image is small enough, process directly
            print("Image is small, processing directly...")
            denoised = model(noisy_t, spatial_map=spatial_map, basis=basis, alpha=alpha, gate=gate)
        else:
            # Use patch-based processing for large images
            print(f"Image is large, using patch-based processing...")
            denoised = model.forward_patch_based(
                noisy_t,
                patch_size=patch_size,
                stride=stride,
                spatial_map=spatial_map,
                basis=basis,
                alpha=alpha,
                gate=gate
            )

    # Convert back to numpy
    denoised_np = denoised.squeeze().cpu().numpy()
    denoised_np = np.clip(denoised_np, 0, 1)

    return denoised_np


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Denoise large OCT images using patch-based inference")
    parser.add_argument("--input", type=str, required=True, help="Path to noisy image")
    parser.add_argument("--output", type=str, required=True, help="Path to save denoised image")
    parser.add_argument("--base_ckpt", type=str, required=True, help="Path to base NAFNet checkpoint")
    parser.add_argument("--analyzer_ckpt", type=str, required=True, help="Path to analyzer checkpoint")
    parser.add_argument("--modulator_ckpt", type=str, required=True, help="Path to trained modulator checkpoint")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size (default: 64)")
    parser.add_argument("--stride", type=int, default=32, help="Stride between patches (default: 32)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Modulation strength (default: 1.0)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load models
    model, analyzer, modulator = load_models(
        args.base_ckpt,
        args.analyzer_ckpt,
        args.modulator_ckpt,
        device=args.device
    )

    # Denoise image
    denoised = denoise_large_image(
        args.input,
        model,
        analyzer,
        modulator,
        device=args.device,
        patch_size=args.patch_size,
        stride=args.stride,
        alpha=args.alpha
    )

    # Save result
    print(f"Saving denoised image to: {args.output}")
    output_img = (denoised * 255).astype(np.uint8)
    Image.fromarray(output_img).save(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
