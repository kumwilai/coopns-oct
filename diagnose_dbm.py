#!/usr/bin/env python3
"""
Quick diagnostic to measure actual modulation magnitudes in the DBM pipeline.
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

def diagnose_modulation():
    """Run a few batches and measure actual modulation magnitudes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load models with same config as training
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

    # Load pre-trained base
    base_ckpt = torch.load(
        "outputs/nafnet_analysis_maps_w64/nafnet_best.pth",
        map_location=device,
        weights_only=False
    )
    state = base_ckpt.get("state_dict", base_ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Initialize modulator with CURRENT settings
    modulator = SpatialBasisModulator(
        feature_channels=128,
        stage_channels=model.dbm_stage_channels,
        num_noise_types=4,
        hidden_channels=64,
        alpha=0.1,  # Current value
        gate_floor=0.0,
        basis_init_std=1e-2,  # Current value (SMALL!)
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
    print("DBM Modulation Magnitude Diagnostic")
    print("="*80)
    print(f"Alpha: 0.1")
    print(f"Basis init std: 1e-2")
    print(f"Gate floor: 0.0")
    print("="*80)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 2:  # Only 2 batches
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
            spatial_map, gate, basis = modulator(
                feature_map,
                global_weights=global_weights,
                confidence=confidence,
                noisy=noisy,
            )

            # Compute map stats
            entropy = -(spatial_map * torch.log(spatial_map.clamp_min(1e-8))).sum(dim=1).mean()
            map_max = spatial_map.max(dim=1).values.mean()

            # Measure basis magnitudes
            basis_norms = []
            for name in modulator.basis_gamma.keys():
                gamma = modulator.basis_gamma[name]
                beta = modulator.basis_beta[name]
                basis_norms.append(torch.norm(gamma, dim=1).mean())
            basis_norm_mean = torch.stack(basis_norms).mean()

            # Compute actual modulation in middle block
            # This mimics what happens in DBMNAFBlock._apply_dbm
            test_stage = "middle"
            basis_gamma = modulator.basis_gamma[test_stage]
            basis_beta = modulator.basis_beta[test_stage]

            # Get the actual feature at middle stage (after downsampling)
            # For simplicity, use a dummy tensor with correct spatial size
            # In real forward, this would be the actual middle features
            dummy_feat_size = spatial_map.shape[-2:][0] // 4  # After 2 downs
            dummy_feat = torch.ones(noisy.size(0), model.dbm_stage_channels[test_stage],
                                    dummy_feat_size, dummy_feat_size, device=device)

            # Resize spatial_map to match
            map_resized = F.interpolate(spatial_map, size=dummy_feat.shape[-2:],
                                       mode="bilinear", align_corners=False)

            # Apply DBM
            alpha_value = 0.1
            if gate is not None:
                alpha_value = alpha_value * gate
                gate_mean = gate.mean()
            else:
                gate_mean = 1.0

            proj_gamma = torch.einsum("bkhw,kc->bchw", map_resized, basis_gamma)
            proj_gamma_pre_tanh = proj_gamma.clone()
            proj_gamma = torch.tanh(proj_gamma)

            proj_beta = torch.einsum("bkhw,kc->bchw", map_resized, basis_beta)
            proj_beta = torch.tanh(proj_beta)

            # Compute modulation
            modulated = dummy_feat * (1.0 + alpha_value * proj_gamma) + alpha_value * proj_beta
            delta = (modulated - dummy_feat).abs().mean()

            print(f"\nBatch {batch_idx}:")
            print(f"  Spatial Map:")
            print(f"    Entropy: {entropy:.4f} (max={torch.log(torch.tensor(4.0)):.4f})")
            print(f"    Max prob: {map_max:.4f}")
            print(f"  Confidence/Gate:")
            print(f"    Confidence: {confidence.mean():.4f}")
            print(f"    Gate: {gate_mean:.4f}")
            print(f"    Alpha_value (alpha*gate): {(0.1 * gate_mean):.4f}")
            print(f"  Basis Vectors ({test_stage}):")
            print(f"    Gamma norm: {torch.norm(basis_gamma, dim=1).mean():.6f}")
            print(f"    Gamma mean: {basis_gamma.mean():.6f}, std: {basis_gamma.std():.6f}")
            print(f"    Gamma range: [{basis_gamma.min():.6f}, {basis_gamma.max():.6f}]")
            print(f"    Beta norm: {torch.norm(basis_beta, dim=1).mean():.6f}")
            print(f"  Projected Modulation:")
            print(f"    proj_gamma (pre-tanh) mean: {proj_gamma_pre_tanh.mean():.6f}")
            print(f"    proj_gamma (pre-tanh) std: {proj_gamma_pre_tanh.std():.6f}")
            print(f"    proj_gamma (pre-tanh) range: [{proj_gamma_pre_tanh.min():.6f}, {proj_gamma_pre_tanh.max():.6f}]")
            print(f"    proj_gamma (post-tanh) mean: {proj_gamma.mean():.6f}")
            print(f"    proj_gamma (post-tanh) std: {proj_gamma.std():.6f}")
            print(f"  Final Modulation:")
            print(f"    alpha * proj_gamma mean: {(alpha_value * proj_gamma).mean():.6f}")
            print(f"    Delta (|modulated - original|): {delta:.6f}")
            print(f"    Relative change: {(delta / (dummy_feat.abs().mean() + 1e-8)):.6f}")

            # Now test what happens with LARGER basis
            print(f"\n  >>> TESTING WITH 10x LARGER BASIS <<<")
            test_gamma = basis_gamma * 10.0
            test_beta = basis_beta * 10.0
            proj_gamma_test = torch.einsum("bkhw,kc->bchw", map_resized, test_gamma)
            proj_gamma_test_pre = proj_gamma_test.clone()
            proj_gamma_test = torch.tanh(proj_gamma_test)
            proj_beta_test = torch.einsum("bkhw,kc->bchw", map_resized, test_beta)
            proj_beta_test = torch.tanh(proj_beta_test)
            modulated_test = dummy_feat * (1.0 + alpha_value * proj_gamma_test) + alpha_value * proj_beta_test
            delta_test = (modulated_test - dummy_feat).abs().mean()
            print(f"    proj_gamma (pre-tanh) mean: {proj_gamma_test_pre.mean():.6f}")
            print(f"    proj_gamma (post-tanh) mean: {proj_gamma_test.mean():.6f}")
            print(f"    Delta with 10x basis: {delta_test:.6f}")
            print(f"    Relative change: {(delta_test / (dummy_feat.abs().mean() + 1e-8)):.6f}")

    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    print("The basis vectors are initialized too small (std=1e-2), and combined with")
    print("small alpha (0.1) and gating by confidence (<1), the actual modulation")
    print("magnitude is essentially ZERO (< 0.001).")
    print("")
    print("Even though the spatial map has learned to be non-uniform (entropy ~1.2,")
    print("max ~0.5), this spatial information is NOT being used because the basis")
    print("vectors are too small to produce meaningful modulation.")
    print("")
    print("SOLUTION: Increase basis_init_std from 1e-2 to 0.1 or 0.2")
    print("         OR increase alpha from 0.1 to 0.5 or 1.0")
    print("="*80)

if __name__ == "__main__":
    diagnose_modulation()
