#!/usr/bin/env python3
"""
Quick verification that the fix works by measuring modulation with updated parameters.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "nsnd_oct"))

import torch
import torch.nn.functional as F
from nsnd.models.nafnet import NAFNetFullFiLM
from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer
from nsnd.models.noise_conditioner import SpatialBasisModulator
from nsnd.datasets.synthetic_oct import PairedOCTCropDataset
from torch.utils.data import DataLoader
from nsnd.utils.metrics import compute_psnr, compute_ssim

def verify_fix():
    """Measure modulation with FIXED parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load models
    analyzer = HybridCNNSymbolicAnalyzer(use_log_domain=False).to(device)
    analyzer_ckpt = torch.load(
        "checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth",
        map_location=device,
        weights_only=False
    )
    analyzer.load_state_dict(analyzer_ckpt["state_dict"], strict=False)
    analyzer.eval()
    for p in analyzer.parameters():
        p.requires_grad = False

    model = NAFNetFullFiLM(
        img_channel=1, width=64,
        enc_blk_nums=[2, 2, 2],
        dec_blk_nums=[2, 2, 2],
        middle_blk_num=2,
        cond_dim=32,
        condition_middle=True,
        condition_decoders=True,
        use_spatial_cue=True
    ).to(device)

    base_ckpt = torch.load(
        "outputs/nafnet_analysis_maps_w64/nafnet_best.pth",
        map_location=device,
        weights_only=False
    )
    state = base_ckpt.get("state_dict", base_ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Initialize modulator with FIXED settings
    modulator_fixed = SpatialBasisModulator(
        feature_channels=128,
        stage_channels=model.dbm_stage_channels,
        num_noise_types=4,
        hidden_channels=64,
        alpha=0.5,  # FIXED: 0.1 → 0.5
        gate_floor=0.3,  # FIXED: 0.0 → 0.3
        basis_init_std=0.1,  # FIXED: 1e-2 → 0.1
    ).to(device)

    # Load dataset
    dataset = PairedOCTCropDataset(
        "train_pairs_duke_analysis_maps.txt",
        crop_size=64,
        random_crop=False,
        max_samples=8,
        return_noise_maps=False,
        return_weights=False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    print("="*80)
    print("Verification: Adaptive Conditioning with FIXED Parameters")
    print("="*80)
    print("BEFORE (Broken):")
    print("  - alpha: 0.1")
    print("  - gate_floor: 0.0")
    print("  - basis_init_std: 1e-2")
    print("  Result: Δbase ≈ 0, modulation ≈ 0")
    print("")
    print("AFTER (Fixed):")
    print("  - alpha: 0.5 (5x increase)")
    print("  - gate_floor: 0.3 (prevent total gating)")
    print("  - basis_init_std: 0.1 (10x increase)")
    print("="*80)

    alpha = 0.5
    base_psnr_list = []
    mod_psnr_list = []
    base_ssim_list = []
    mod_ssim_list = []
    delta_base_list = []
    map_entropy_list = []
    map_max_list = []
    gate_list = []
    basis_norm_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 2:
                break

            noisy, clean = batch
            noisy, clean = noisy.to(device), clean.to(device)

            # Get analyzer outputs
            weights_dict, features = analyzer(noisy, return_feature_map=True)
            feature_map = features["feature_map"]
            global_weights = torch.stack([
                weights_dict["speckle"],
                weights_dict["banding"],
                weights_dict["gaussian"],
                weights_dict["shot"],
            ], dim=1)
            confidence = weights_dict.get("_confidence", torch.ones(noisy.size(0), device=device))

            # Get modulator outputs
            spatial_map, gate, basis = modulator_fixed(
                feature_map,
                global_weights=global_weights,
                confidence=confidence,
                noisy=noisy,
            )

            # Forward with conditioning
            denoised = model(
                noisy,
                spatial_map=spatial_map,
                basis=basis,
                alpha=alpha,
                gate=gate,
            )

            # Forward without conditioning (base)
            base_out = model(
                noisy,
                spatial_map=None,
                basis=basis,
                alpha=0.0,
                gate=None,
            )

            # Compute metrics
            entropy = -(spatial_map * torch.log(spatial_map.clamp_min(1e-8))).sum(dim=1).mean()
            map_max = spatial_map.max(dim=1).values.mean()
            delta_base = (denoised - base_out).abs().mean()

            # Basis norms
            basis_norms = []
            for name in modulator_fixed.basis_gamma.keys():
                gamma = modulator_fixed.basis_gamma[name]
                basis_norms.append(torch.norm(gamma, dim=1).mean())
            basis_norm_mean = torch.stack(basis_norms).mean()

            # Per-image metrics
            for i in range(len(denoised)):
                base_psnr = compute_psnr(base_out[i], clean[i])
                mod_psnr = compute_psnr(denoised[i], clean[i])
                base_ssim = compute_ssim(base_out[i], clean[i])
                mod_ssim = compute_ssim(denoised[i], clean[i])

                base_psnr_list.append(base_psnr)
                mod_psnr_list.append(mod_psnr)
                base_ssim_list.append(base_ssim)
                mod_ssim_list.append(mod_ssim)
                delta_base_list.append((denoised[i] - base_out[i]).abs().mean().item())

            map_entropy_list.append(float(entropy))
            map_max_list.append(float(map_max))
            gate_list.append(float(gate.mean()))
            basis_norm_list.append(float(basis_norm_mean))

            print(f"\nBatch {batch_idx}:")
            print(f"  Spatial Map: entropy={entropy:.4f}, max={map_max:.4f}")
            print(f"  Gate: {gate.mean():.4f} (min={gate_floor}, confidence={confidence.mean():.4f})")
            print(f"  Basis norm: {basis_norm_mean:.4f}")
            print(f"  Delta_base: {delta_base:.6f}")
            print(f"  Base PSNR: {sum(base_psnr_list[-len(denoised):]) / len(denoised):.2f}")
            print(f"  Mod PSNR: {sum(mod_psnr_list[-len(denoised):]) / len(denoised):.2f}")
            print(f"  Gain PSNR: {sum(mod_psnr_list[-len(denoised):]) / len(denoised) - sum(base_psnr_list[-len(denoised):]) / len(denoised):.3f}")

    # Summary
    avg_base_psnr = sum(base_psnr_list) / len(base_psnr_list)
    avg_mod_psnr = sum(mod_psnr_list) / len(mod_psnr_list)
    avg_base_ssim = sum(base_ssim_list) / len(base_ssim_list)
    avg_mod_ssim = sum(mod_ssim_list) / len(mod_ssim_list)
    avg_delta_base = sum(delta_base_list) / len(delta_base_list)
    avg_map_entropy = sum(map_entropy_list) / len(map_entropy_list)
    avg_map_max = sum(map_max_list) / len(map_max_list)
    avg_gate = sum(gate_list) / len(gate_list)
    avg_basis_norm = sum(basis_norm_list) / len(basis_norm_list)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Map entropy: {avg_map_entropy:.4f}")
    print(f"Map max: {avg_map_max:.4f}")
    print(f"Gate (floor={gate_floor}): {avg_gate:.4f}")
    print(f"Basis norm: {avg_basis_norm:.4f}")
    print(f"Δbase: {avg_delta_base:.6f}")
    print(f"Base PSNR: {avg_base_psnr:.2f} dB")
    print(f"Modulated PSNR: {avg_mod_psnr:.2f} dB")
    print(f"Gain PSNR: {avg_mod_psnr - avg_base_psnr:+.3f} dB")
    print(f"Base SSIM: {avg_base_ssim:.4f}")
    print(f"Modulated SSIM: {avg_mod_ssim:.4f}")
    print(f"Gain SSIM: {avg_mod_ssim - avg_base_ssim:+.5f}")
    print("="*80)

    success = avg_delta_base > 0.001
    print(f"\nVERIFICATION: {'✓ PASS' if success else '✗ FAIL'}")
    print(f"Expected: Δbase > 0.001")
    print(f"Actual: Δbase = {avg_delta_base:.6f}")

    if success:
        print("\n✓ Adaptive conditioning is now WORKING!")
        print("  The modulation is clearly non-zero and should be visible in training.")
    else:
        print("\n✗ Adaptive conditioning is still not working as expected.")
        print("  Further investigation or parameter tuning may be needed.")

if __name__ == "__main__":
    gate_floor = 0.3  # Define for print statement
    verify_fix()
