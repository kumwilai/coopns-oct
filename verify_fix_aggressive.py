#!/usr/bin/env python3
"""
Test with more aggressive parameters to ensure visible modulation.
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

def test_alpha_sweep():
    """Test different alpha values to find the threshold for visible modulation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Load dataset
    dataset = PairedOCTCropDataset(
        "train_pairs_duke_analysis_maps.txt",
        crop_size=64,
        random_crop=False,
        max_samples=4,
        return_noise_maps=False,
        return_weights=False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Get one batch
    noisy, clean = next(iter(loader))
    noisy, clean = noisy.to(device), clean.to(device)

    weights_dict, features = analyzer(noisy, return_feature_map=True)
    feature_map = features["feature_map"]
    global_weights = torch.stack([
        weights_dict["speckle"],
        weights_dict["banding"],
        weights_dict["gaussian"],
        weights_dict["shot"],
    ], dim=1)
    confidence = weights_dict.get("_confidence", torch.ones(noisy.size(0), device=device))

    print("="*80)
    print("Alpha Sweep Test: Finding Threshold for Visible Modulation")
    print("="*80)
    print(f"Gate floor: 0.3")
    print(f"Basis init std: 0.1")
    print(f"Confidence: {confidence.mean():.4f}")
    print("="*80)

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for alpha in alphas:
        modulator = SpatialBasisModulator(
            feature_channels=128,
            stage_channels=model.dbm_stage_channels,
            num_noise_types=4,
            hidden_channels=64,
            alpha=alpha,
            gate_floor=0.3,
            basis_init_std=0.1,
        ).to(device)

        with torch.no_grad():
            spatial_map, gate, basis = modulator(
                feature_map,
                global_weights=global_weights,
                confidence=confidence,
                noisy=noisy,
            )

            denoised = model(noisy, spatial_map=spatial_map, basis=basis, alpha=alpha, gate=gate)
            base_out = model(noisy, spatial_map=None, basis=basis, alpha=0.0, gate=None)

            delta_base = (denoised - base_out).abs().mean()
            mod_psnr = compute_psnr(denoised[0], clean[0])
            base_psnr = compute_psnr(base_out[0], clean[0])
            gain_psnr = mod_psnr - base_psnr

            status = "✓" if delta_base > 0.001 else "✗"
            print(f"Alpha={alpha:5.1f}: Δbase={delta_base:.6f} | GainPSNR={gain_psnr:+.3f} dB | {status}")

    print("="*80)
    print("Recommendation: Use alpha >= 1.0 for visible modulation with uniform map")
    print("                Or pre-train the spatial map first (map_only_epochs > 0)")
    print("="*80)

if __name__ == "__main__":
    test_alpha_sweep()
