#!/usr/bin/env python3
"""Evaluate cooperative models on PKU37 test set (173 samples).
Reports all 7 clinical metrics: PSNR, CNR, TCI, EPI, BS, ENL, SNR.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
    compute_ssim,
)
from validate_crossdataset import otsu_tissue_mask


def evaluate_pku37(model, test_jsonl, device="cpu"):
    """Evaluate model on PKU37 test set, return per-image and aggregate metrics."""
    dataset = PKU37Dataset(test_jsonl, patch_size=0, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sobel_y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device
    ).view(1, 1, 3, 3)
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device
    ).view(1, 1, 3, 3)

    all_results = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", unit="img")
        for i, batch in enumerate(pbar):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)

            backbone_out, nafnet_unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=nafnet_unc, return_details=False,
            )

            # PSNR / SSIM
            psnr_bb = compute_psnr(backbone_out, clean)
            psnr_co = compute_psnr(corrected, clean)
            ssim_bb = compute_ssim(backbone_out, clean)
            ssim_co = compute_ssim(corrected, clean)

            # Correction magnitude
            corr_mag = (corrected - backbone_out).abs().mean().item()

            # CNR (Otsu)
            signal_mask = otsu_tissue_mask(backbone_out)
            bg_mask = 1.0 - signal_mask
            sig_sum = signal_mask.sum().clamp(min=1.0)
            bg_sum = bg_mask.sum().clamp(min=1.0)

            bb_sig = (backbone_out * signal_mask).sum() / sig_sum
            bb_bg = (backbone_out * bg_mask).sum() / bg_sum
            bb_bg_std = torch.sqrt(((backbone_out - bb_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
            cnr_bb = ((bb_sig - bb_bg) / bb_bg_std).clamp(-100, 100).item()

            co_sig = (corrected * signal_mask).sum() / sig_sum
            co_bg = (corrected * bg_mask).sum() / bg_sum
            co_bg_std = torch.sqrt(((corrected - co_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
            cnr_co = ((co_sig - co_bg) / co_bg_std).clamp(-100, 100).item()

            # TCI
            bb_gy = F.conv2d(backbone_out, sobel_y, padding=1).abs()
            co_gy = F.conv2d(corrected, sobel_y, padding=1).abs()
            cl_gy = F.conv2d(clean, sobel_y, padding=1).abs()
            cl_gy_mean = cl_gy.mean().clamp(min=1e-4)
            tci_bb = (bb_gy.mean() / cl_gy_mean).clamp(0, 10).item()
            tci_co = (co_gy.mean() / cl_gy_mean).clamp(0, 10).item()

            # EPI
            bb_gx = F.conv2d(backbone_out, sobel_x, padding=1)
            co_gx = F.conv2d(corrected, sobel_x, padding=1)
            cl_gx = F.conv2d(clean, sobel_x, padding=1)
            bb_edge = torch.sqrt(bb_gx**2 + bb_gy**2 + 1e-8)
            co_edge = torch.sqrt(co_gx**2 + co_gy**2 + 1e-8)
            cl_edge = torch.sqrt(cl_gx**2 + cl_gy**2 + 1e-8)

            cl_flat = cl_edge.view(-1)
            bb_flat = bb_edge.view(-1)
            co_flat = co_edge.view(-1)
            cl_std = cl_flat.std().clamp(min=1e-4)
            bb_std = bb_flat.std().clamp(min=1e-4)
            co_std = co_flat.std().clamp(min=1e-4)
            cl_norm = (cl_flat - cl_flat.mean()) / cl_std
            bb_norm = (bb_flat - bb_flat.mean()) / bb_std
            co_norm = (co_flat - co_flat.mean()) / co_std
            epi_bb = (cl_norm * bb_norm).mean().item()
            epi_co = (cl_norm * co_norm).mean().item()

            # BS
            cl_gy_max = cl_gy.max().clamp(min=1e-4)
            bs_bb = (bb_gy.max() / cl_gy_max).clamp(0, 10).item()
            bs_co = (co_gy.max() / cl_gy_max).clamp(0, 10).item()

            # ENL
            H = backbone_out.shape[2]
            bg_b = backbone_out[0, 0, H*3//4:, :]
            bg_c = corrected[0, 0, H*3//4:, :]
            enl_bb = (bg_b.mean() / bg_b.std().clamp(min=1e-6)).item() ** 2
            enl_co = (bg_c.mean() / bg_c.std().clamp(min=1e-6)).item() ** 2

            # SNR
            tis_b = backbone_out[0, 0, :H//2, :]
            tis_c = corrected[0, 0, :H//2, :]
            snr_bb = (tis_b.mean() / bg_b.std().clamp(min=1e-6)).item()
            snr_co = (tis_c.mean() / bg_c.std().clamp(min=1e-6)).item()

            all_results.append({
                "image_idx": i,
                "psnr_backbone": psnr_bb, "psnr_corrected": psnr_co,
                "psnr_delta": psnr_co - psnr_bb,
                "ssim_backbone": ssim_bb, "ssim_corrected": ssim_co,
                "cnr_backbone": cnr_bb, "cnr_corrected": cnr_co,
                "cnr_change_pct": (cnr_co - cnr_bb) / max(abs(cnr_bb), 1e-8) * 100,
                "tci_backbone": tci_bb, "tci_corrected": tci_co,
                "tci_change_pct": (tci_co - tci_bb) / max(abs(tci_bb), 1e-8) * 100,
                "epi_backbone": epi_bb, "epi_corrected": epi_co,
                "epi_change_pct": (epi_co - epi_bb) / max(abs(epi_bb), 1e-8) * 100,
                "bs_backbone": bs_bb, "bs_corrected": bs_co,
                "bs_change_pct": (bs_co - bs_bb) / max(abs(bs_bb), 1e-8) * 100,
                "enl_backbone": enl_bb, "enl_corrected": enl_co,
                "enl_change_pct": (enl_co - enl_bb) / max(abs(enl_bb), 1e-8) * 100,
                "snr_backbone": snr_bb, "snr_corrected": snr_co,
                "snr_change_pct": (snr_co - snr_bb) / max(abs(snr_bb), 1e-8) * 100,
                "correction_magnitude": corr_mag,
            })

            # Update progress bar with running averages
            avg_psnr_delta = np.mean([r["psnr_delta"] for r in all_results])
            avg_cnr = np.mean([r["cnr_change_pct"] for r in all_results])
            pbar.set_postfix(psnr_delta=f"{avg_psnr_delta:+.3f}", cnr=f"{avg_cnr:+.1f}%", mag=f"{corr_mag:.4f}")

    return all_results


def print_summary(all_results, backbone_name):
    """Print summary table."""
    from scipy.stats import wilcoxon

    n = len(all_results)
    print(f"\n{'='*70}")
    print(f"  {backbone_name.upper()} + CoopNS on PKU37 Test ({n} images)")
    print(f"{'='*70}")

    metrics = [
        ("PSNR (dB)", "psnr_backbone", "psnr_corrected", "psnr_delta", False),
        ("CNR", "cnr_backbone", "cnr_corrected", "cnr_change_pct", True),
        ("TCI", "tci_backbone", "tci_corrected", "tci_change_pct", True),
        ("EPI", "epi_backbone", "epi_corrected", "epi_change_pct", True),
        ("BS", "bs_backbone", "bs_corrected", "bs_change_pct", True),
        ("ENL", "enl_backbone", "enl_corrected", "enl_change_pct", True),
        ("SNR", "snr_backbone", "snr_corrected", "snr_change_pct", True),
    ]

    summary = {}
    for name, bb_key, co_key, delta_key, use_pct in metrics:
        bb_vals = np.array([r[bb_key] for r in all_results])
        co_vals = np.array([r[co_key] for r in all_results])

        if use_pct:
            delta_vals = np.array([r[delta_key] for r in all_results])
            delta_mean, delta_std = delta_vals.mean(), delta_vals.std()
            change_str = f"{delta_mean:+.2f}±{delta_std:.2f}%"
        else:
            delta_vals = co_vals - bb_vals
            delta_mean, delta_std = delta_vals.mean(), delta_vals.std()
            change_str = f"{delta_mean:+.3f}±{delta_std:.3f}"

        try:
            _, p_val = wilcoxon(co_vals, bb_vals, alternative="two-sided")
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
        except Exception:
            p_val, p_str = 1.0, "N/A"

        print(f"  {name:<12} {change_str:>18}  p={p_str}")
        summary[name] = {
            "delta_mean": float(delta_mean), "delta_std": float(delta_std),
            "p_value": float(p_val),
            "backbone_mean": float(bb_vals.mean()), "corrected_mean": float(co_vals.mean()),
        }

    mag_vals = np.array([r["correction_magnitude"] for r in all_results])
    print(f"  {'Corr. Mag':<12} {mag_vals.mean():.6f}±{mag_vals.std():.6f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate on PKU37 test set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained_backbone", required=True)
    parser.add_argument("--backbone_name", default="nafnet", choices=["nafnet", "kbnet", "dncnn", "swinir"])
    parser.add_argument("--test_jsonl", default="pku37_oct_dataset/pku37_real_test.jsonl")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    print(f"Loading {args.backbone_name} model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        hidden_channels=args.hidden_channels,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Clean keys and filter shape mismatches
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state_dict.items():
        key = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        if key in model_state and v.shape == model_state[key].shape:
            compatible[key] = v
        elif key in model_state:
            skipped.append(f"{key}: ckpt {v.shape} vs model {model_state[key].shape}")

    model.load_state_dict(compatible, strict=False)
    print(f"  Loaded {len(compatible)}/{len(model_state)} keys")
    if skipped:
        print(f"  Skipped {len(skipped)} mismatched keys:")
        for s in skipped:
            print(f"    {s}")

    model = model.to(args.device)
    model.eval()

    print(f"\nEvaluating on PKU37 test ({args.test_jsonl})...")
    results = evaluate_pku37(model, args.test_jsonl, args.device)
    summary = print_summary(results, args.backbone_name)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"backbone": args.backbone_name, "n_images": len(results),
                        "per_image": results, "summary": summary}, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
