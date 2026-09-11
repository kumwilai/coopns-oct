#!/usr/bin/env python3
"""
Leave-One-Out Cross-Validation for Duke17/Duke2013 Few-Shot Adaptation.

Duke17 has 16 subjects (1, 3-17). Duke2013 has 18 subjects (1-18).
LOO cross-validation:
  - N folds, each holding out 1 subject for evaluation
  - N-1 training subjects per fold
  - Reports mean ± std across all N subjects
  - Paired Wilcoxon signed-rank test for statistical significance

Usage:
  # Run all folds for Duke17 (default)
  python run_duke17_loo.py --dataset duke17

  # Run all folds for Duke2013
  python run_duke17_loo.py --dataset duke2013

  # Run a single fold (e.g., subject 5)
  python run_duke17_loo.py --fold 5

  # Skip training, only aggregate existing results
  python run_duke17_loo.py --skip_train

  # Run evaluation only (no training, compute metrics from checkpoints)
  python run_duke17_loo.py --skip_train --eval_only
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Constants ───────────────────────────────────────────────────────────────

DUKE17_SUBJECTS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
DUKE17_BASE = os.environ.get("DUKE17_BASE", "duke_sota_datasets/Sparsity_SDOCT_DATASET_2012")

DUKE2013_SUBJECTS = list(range(1, 19))  # 1-18
DUKE2013_BASE = os.environ.get("DUKE2013_BASE", "duke_sota_datasets/Duke2013_SBSDI/Final_Publication_2013_SBSDI_sourcecode_withpcode/For synthetic experiments")

DATASET_CONFIG = {
    "duke17": {
        "subjects": DUKE17_SUBJECTS,
        "base": DUKE17_BASE,
        "fold_dir": "duke17_loo_folds",
        "output_base": "outputs/duke17_loo",
        "noisy_fmt": "{base}/{subj}/{subj}_Raw Image.tif",
        "clean_fmt": "{base}/{subj}/{subj}_Averaged Image.tif",
        "subject_prefix": "",
    },
    "duke2013": {
        "subjects": DUKE2013_SUBJECTS,
        "base": DUKE2013_BASE,
        "fold_dir": "duke2013_loo_folds",
        "output_base": "outputs/duke2013_loo",
        "noisy_fmt": "{base}/{subj}/test.tif",
        "clean_fmt": "{base}/{subj}/average.tif",
        "subject_prefix": "duke2013_",
    },
}

FOLD_DIR = "duke17_loo_folds"
OUTPUT_BASE = "outputs/duke17_loo"
PRETRAINED_BACKBONES = {
    "nafnet": "outputs/nafnet_pku37_w40/best_model.pth",
    "kbnet": "NukeModel/kbnet_7m/best.pth",
    "dncnn": "NukeModel/dncnn_7m/best.pth",
    "swinir": "NukeModel/swinir_7m/best.pth",
}
RESUME_CHECKPOINTS = {
    "nafnet": "outputs/nafnet_qt69d/best_model_cooperative.pth",
    "kbnet": "outputs/kbnet_qt69d/best_model_cooperative.pth",
    "dncnn": "outputs/dncnn_enc1_qt69d/best_model_cooperative.pth",
    "swinir": "outputs/swinir_qt69d/best_model_cooperative.pth",
}
DEFAULT_BACKBONE = "nafnet"


def parse_args():
    parser = argparse.ArgumentParser(
        description="LOO Cross-Validation for Duke17/Duke2013 Few-Shot Adaptation"
    )
    parser.add_argument("--dataset", type=str, default="duke17",
                        choices=["duke17", "duke2013"],
                        help="Dataset to use (default: duke17)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (subject ID). Default: all.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, only run evaluation and aggregation.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation from existing checkpoints (implies --skip_train).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for evaluation (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs per fold (default: 15)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size (default: 2)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Pretrained checkpoint to resume from (auto-detected from backbone)")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Hidden channels for cooperative model (default: 64)")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        choices=["nafnet", "kbnet", "dncnn", "swinir"],
                        help=f"Backbone architecture (default: {DEFAULT_BACKBONE})")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where the fold checkpoints and loo_results.json go. Point this "
                             "somewhere new when re-running, so an old run is never overwritten "
                             "and never silently reused.")
    parser.add_argument("--reuse_folds", action="store_true",
                        help="Keep fold checkpoints that already exist instead of retraining them.")
    parser.add_argument("--allow_legacy_resume", action="store_true",
                        help="Permit the built in pre revision resume checkpoints when --resume is absent.")
    parser.add_argument("--pretrained_backbone", type=str, default=None,
                        help="Path to pretrained backbone (auto-detected from --backbone)")
    args = parser.parse_args()

    # Auto-detect resume and backbone paths.
    #
    # The built in defaults point at the checkpoints of the original submission,
    # which predate the allocation fix. Falling back to them silently would produce
    # a table that mixes two different models, so the fallback now has to be asked
    # for explicitly.
    if args.resume is None:
        if not args.allow_legacy_resume:
            print("ERROR: --resume is required. The built in defaults are the pre revision\n"
                  "       checkpoints and mixing them with current results would compare two\n"
                  "       different models. Pass --resume explicitly, or --allow_legacy_resume\n"
                  "       if you deliberately want to reproduce the original submission.")
            sys.exit(1)
        args.resume = RESUME_CHECKPOINTS.get(args.backbone)
        print(f"WARNING: using the pre revision checkpoint {args.resume}")
        if args.resume is None:
            print(f"ERROR: No default resume checkpoint for backbone '{args.backbone}'. Use --resume.")
            sys.exit(1)
    if args.pretrained_backbone is None:
        args.pretrained_backbone = PRETRAINED_BACKBONES.get(args.backbone)
        if args.pretrained_backbone is None:
            print(f"ERROR: No default pretrained backbone for '{args.backbone}'. Use --pretrained_backbone.")
            sys.exit(1)
    return args


# ── Step 1: Generate fold JSONLs ────────────────────────────────────────────

def generate_fold_jsonls(ds_cfg):
    """Generate N train/val JSONL pairs for LOO cross-validation."""
    fold_dir = ds_cfg["fold_dir"]
    subjects = ds_cfg["subjects"]
    base = ds_cfg["base"]
    noisy_fmt = ds_cfg["noisy_fmt"]
    clean_fmt = ds_cfg["clean_fmt"]
    prefix = ds_cfg["subject_prefix"]

    os.makedirs(fold_dir, exist_ok=True)

    for held_out in subjects:
        train_path = os.path.join(fold_dir, f"fold_{held_out}_train.jsonl")
        val_path = os.path.join(fold_dir, f"fold_{held_out}_val.jsonl")

        # Train: all subjects except held-out
        with open(train_path, "w") as f:
            for subj in subjects:
                if subj == held_out:
                    continue
                entry = {
                    "noisy_path": noisy_fmt.format(base=base, subj=subj),
                    "clean_path": clean_fmt.format(base=base, subj=subj),
                    "subject": f"{prefix}{subj}",
                }
                f.write(json.dumps(entry) + "\n")

        # Val: held-out subject only
        with open(val_path, "w") as f:
            entry = {
                "noisy_path": noisy_fmt.format(base=base, subj=held_out),
                "clean_path": clean_fmt.format(base=base, subj=held_out),
                "subject": f"{prefix}{held_out}",
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(subjects)} fold JSONL pairs in {fold_dir}/")
    return True


# ── Step 2: Train each fold ─────────────────────────────────────────────────

def train_fold(subject_id, args, ds_cfg):
    """Train one LOO fold by shelling out to train_v8_cooperative.py."""
    fold_output = os.path.join(OUTPUT_BASE, f"fold_{subject_id}")
    fold_dir = ds_cfg["fold_dir"]
    train_jsonl = os.path.join(fold_dir, f"fold_{subject_id}_train.jsonl")
    val_jsonl = os.path.join(fold_dir, f"fold_{subject_id}_val.jsonl")

    # Check if already trained
    best_ckpt = os.path.join(fold_output, "best_model_cooperative.pth")
    if os.path.exists(best_ckpt):
        # Reusing a fold checkpoint from an earlier code version is how a re-run
        # ends up reporting the old model, so this now has to be asked for.
        if args.reuse_folds:
            print(f"  Fold {subject_id}: reusing the existing checkpoint as asked.")
            return True
        print(f"  Fold {subject_id}: a checkpoint already exists at {best_ckpt}.\n"
              f"    Retraining it. Pass --reuse_folds to keep it instead, or point\n"
              f"    --output_dir somewhere new.")

    cmd = [
        sys.executable, "train_v8_cooperative.py",
        "--backbone", args.backbone,
        "--pretrained_backbone", args.pretrained_backbone,
        "--resume", args.resume,
        "--resume_epoch", "1",
        "--train_jsonl", train_jsonl,
        "--val_jsonl", val_jsonl,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr_corrector", "2e-5",
        "--lr_potential", "1e-4",
        "--lr_negotiator", "6e-5",
        "--stage_switch_epoch", "5",
        "--hidden_channels", str(args.hidden_channels),
        "--ewc_weight", "0.1",
        "--bg_correction_var_weight", "10.0",
        "--output_dir", fold_output,
        "--val_every", "1",
        "--max_val", "1",
        "--device", args.device,
    ]

    print(f"\n{'='*70}")
    print(f"  TRAINING FOLD: held-out subject {subject_id}")
    print(f"  Output: {fold_output}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  WARNING: Fold {subject_id} training failed (exit code {result.returncode})")
        return False

    print(f"  Fold {subject_id} training completed in {elapsed/60:.1f} min")
    return True


# ── Step 3: Evaluate each fold ──────────────────────────────────────────────

def evaluate_fold(subject_id, args, ds_cfg):
    """Evaluate one fold's held-out subject using the fold's best checkpoint.

    Returns a dict of per-subject metrics, or None on failure.
    """
    fold_output = os.path.join(OUTPUT_BASE, f"fold_{subject_id}")
    best_ckpt = os.path.join(fold_output, "best_model_cooperative.pth")

    if not os.path.exists(best_ckpt):
        print(f"  Fold {subject_id}: no checkpoint found at {best_ckpt}")
        return None

    # Import model and utilities from training script
    from train_v8_cooperative import (
        NeuroSymbolicDenoiserV8Cooperative,
        PKU37Dataset,
        compute_psnr,
        compute_ssim,
    )

    device = args.device
    fold_dir = ds_cfg["fold_dir"]
    val_jsonl = os.path.join(fold_dir, f"fold_{subject_id}_val.jsonl")

    # Load model
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        hidden_channels=args.hidden_channels,
    )
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Strip _orig_mod. from torch.compile() saved checkpoints
    # Handles both full-model compile (leading _orig_mod.) and
    # sub-module compile (e.g. corrector._orig_mod.xxx)
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        cleaned[key] = v
    # Filter out shape mismatches (uncertainty head may differ)
    model_state = model.state_dict()
    compatible = {}
    for k, v in cleaned.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible[k] = v
    model.load_state_dict(compatible, strict=False)
    print(f"    Loaded {len(compatible)}/{len(model_state)} keys")
    model = model.to(device)
    model.eval()

    # Load held-out subject
    dataset = PKU37Dataset(val_jsonl, patch_size=0, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Sobel / Laplacian kernels
    sobel_y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device
    ).view(1, 1, 3, 3)
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device
    ).view(1, 1, 3, 3)

    # SwinIR on CPU: tiled inference is handled inside BackboneWrapper
    is_swinir_cpu = (args.backbone == 'swinir' and device == 'cpu')
    if is_swinir_cpu:
        print(f"    SwinIR on CPU: using tiled inference (128px tiles, 16px overlap)")

    results = {}
    with torch.no_grad():
        for batch in loader:
            clean = batch["clean"].to(device, non_blocking=True)
            noisy = batch["noisy"].to(device, non_blocking=True)

            # SwinIR tiled inference handled inside BackboneWrapper._forward_swinir_with_shallow()
            backbone_out, nafnet_unc = model.backbone(noisy)

            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=nafnet_unc, return_details=False,
            )

            # PSNR / SSIM
            psnr_backbone = compute_psnr(backbone_out, clean)
            psnr_corrected = compute_psnr(corrected, clean)
            ssim_backbone = compute_ssim(backbone_out, clean)
            ssim_corrected = compute_ssim(corrected, clean)

            # Correction magnitude
            correction_mag = (corrected - backbone_out).abs().mean().item()

            # CNR (Otsu-based)
            from validate_crossdataset import otsu_tissue_mask
            signal_mask = otsu_tissue_mask(backbone_out)
            bg_mask = 1.0 - signal_mask
            signal_sum = signal_mask.sum().clamp(min=1.0)
            bg_sum = bg_mask.sum().clamp(min=1.0)

            bb_sig = (backbone_out * signal_mask).sum() / signal_sum
            bb_bg = (backbone_out * bg_mask).sum() / bg_sum
            bb_bg_std = torch.sqrt(
                ((backbone_out - bb_bg) ** 2 * bg_mask).sum() / bg_sum + 1e-8
            ).clamp(min=1e-4)
            cnr_bb = ((bb_sig - bb_bg) / bb_bg_std).clamp(-100, 100).item()

            co_sig = (corrected * signal_mask).sum() / signal_sum
            co_bg = (corrected * bg_mask).sum() / bg_sum
            co_bg_std = torch.sqrt(
                ((corrected - co_bg) ** 2 * bg_mask).sum() / bg_sum + 1e-8
            ).clamp(min=1e-4)
            cnr_co = ((co_sig - co_bg) / co_bg_std).clamp(-100, 100).item()

            # TCI (vertical gradient ratio)
            backbone_gy = F.conv2d(backbone_out, sobel_y, padding=1).abs()
            corrected_gy = F.conv2d(corrected, sobel_y, padding=1).abs()
            clean_gy = F.conv2d(clean, sobel_y, padding=1).abs()
            clean_gy_mean = clean_gy.mean().clamp(min=1e-4)
            tci_bb = (backbone_gy.mean() / clean_gy_mean).clamp(0, 10).item()
            tci_co = (corrected_gy.mean() / clean_gy_mean).clamp(0, 10).item()

            # EPI (edge preservation index — normalized correlation)
            backbone_gx = F.conv2d(backbone_out, sobel_x, padding=1)
            corrected_gx = F.conv2d(corrected, sobel_x, padding=1)
            clean_gx = F.conv2d(clean, sobel_x, padding=1)
            backbone_edge = torch.sqrt(backbone_gx ** 2 + backbone_gy ** 2 + 1e-8)
            corrected_edge = torch.sqrt(corrected_gx ** 2 + corrected_gy ** 2 + 1e-8)
            clean_edge = torch.sqrt(clean_gx ** 2 + clean_gy ** 2 + 1e-8)

            clean_edge_flat = clean_edge.view(-1)
            backbone_edge_flat = backbone_edge.view(-1)
            corrected_edge_flat = corrected_edge.view(-1)
            clean_edge_std = clean_edge_flat.std().clamp(min=1e-4)
            backbone_edge_std = backbone_edge_flat.std().clamp(min=1e-4)
            corrected_edge_std = corrected_edge_flat.std().clamp(min=1e-4)
            clean_norm = (clean_edge_flat - clean_edge_flat.mean()) / clean_edge_std
            backbone_norm = (backbone_edge_flat - backbone_edge_flat.mean()) / backbone_edge_std
            corrected_norm = (corrected_edge_flat - corrected_edge_flat.mean()) / corrected_edge_std
            epi_bb = (clean_norm * backbone_norm).mean().item()
            epi_co = (clean_norm * corrected_norm).mean().item()

            # Boundary Sharpness (max vertical gradient ratio)
            clean_gy_max = clean_gy.max().clamp(min=1e-4)
            bs_bb = (backbone_gy.max() / clean_gy_max).clamp(0, 10).item()
            bs_co = (corrected_gy.max() / clean_gy_max).clamp(0, 10).item()

            # ENL (bottom 25% background region)
            H_img = backbone_out.shape[2]
            bg_b = backbone_out[0, 0, H_img * 3 // 4:, :]
            bg_c = corrected[0, 0, H_img * 3 // 4:, :]
            bg_n = noisy[0, 0, H_img * 3 // 4:, :]
            enl_bb = (bg_b.mean() / bg_b.std().clamp(min=1e-6)).item() ** 2
            enl_co = (bg_c.mean() / bg_c.std().clamp(min=1e-6)).item() ** 2
            enl_noisy = (bg_n.mean() / bg_n.std().clamp(min=1e-6)).item() ** 2

            # SNR (tissue top 50% signal / background bottom 25% noise)
            tissue_b = backbone_out[0, 0, :H_img // 2, :]
            tissue_c = corrected[0, 0, :H_img // 2, :]
            snr_bb = (tissue_b.mean() / bg_b.std().clamp(min=1e-6)).item()
            snr_co = (tissue_c.mean() / bg_c.std().clamp(min=1e-6)).item()

            results = {
                "subject": subject_id,
                "psnr_backbone": psnr_backbone,
                "psnr_corrected": psnr_corrected,
                "psnr_delta": psnr_corrected - psnr_backbone,
                "ssim_backbone": ssim_backbone,
                "ssim_corrected": ssim_corrected,
                "cnr_backbone": cnr_bb,
                "cnr_corrected": cnr_co,
                "cnr_change_pct": ((cnr_co - cnr_bb) / max(abs(cnr_bb), 1e-8)) * 100,
                "tci_backbone": tci_bb,
                "tci_corrected": tci_co,
                "tci_change_pct": ((tci_co - tci_bb) / max(abs(tci_bb), 1e-8)) * 100,
                "epi_backbone": epi_bb,
                "epi_corrected": epi_co,
                "epi_change_pct": ((epi_co - epi_bb) / max(abs(epi_bb), 1e-8)) * 100,
                "bs_backbone": bs_bb,
                "bs_corrected": bs_co,
                "bs_change_pct": ((bs_co - bs_bb) / max(abs(bs_bb), 1e-8)) * 100,
                "enl_backbone": enl_bb,
                "enl_corrected": enl_co,
                "enl_noisy": enl_noisy,
                "enl_change_pct": ((enl_co - enl_bb) / max(abs(enl_bb), 1e-8)) * 100,
                "snr_backbone": snr_bb,
                "snr_corrected": snr_co,
                "snr_change_pct": ((snr_co - snr_bb) / max(abs(snr_bb), 1e-8)) * 100,
                "correction_magnitude": correction_mag,
            }

    # Free model and all intermediate tensors between folds
    del model, dataset, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ── Step 4: Aggregate results ───────────────────────────────────────────────

def aggregate_results(all_results):
    """Compute mean ± std and Wilcoxon signed-rank tests across all folds."""
    from scipy.stats import wilcoxon

    n = len(all_results)
    print(f"\n{'='*84}")
    print(f"  LOO CROSS-VALIDATION RESULTS ({n} subjects)")
    print(f"{'='*84}")

    # Collect paired arrays for each metric
    metrics = [
        ("PSNR (dB)", "psnr_backbone", "psnr_corrected", "psnr_delta", False),
        ("CNR", "cnr_backbone", "cnr_corrected", "cnr_change_pct", True),
        ("TCI", "tci_backbone", "tci_corrected", "tci_change_pct", True),
        ("EPI", "epi_backbone", "epi_corrected", "epi_change_pct", True),
        ("Boundary Sharpness", "bs_backbone", "bs_corrected", "bs_change_pct", True),
        ("ENL", "enl_backbone", "enl_corrected", "enl_change_pct", True),
        ("SNR", "snr_backbone", "snr_corrected", "snr_change_pct", True),
    ]

    # Per-subject table
    print(f"\n┌{'─'*90}┐")
    print(f"│ {'Subject':>8} │ {'PSNR Δ':>8} │ {'CNR%':>8} │ {'TCI%':>8} │ "
          f"{'EPI%':>8} │ {'BS%':>8} │ {'ENL%':>8} │ {'SNR%':>8} │")
    print(f"├{'─'*90}┤")
    for r in sorted(all_results, key=lambda x: x["subject"]):
        print(f"│ {r['subject']:>8} │ {r['psnr_delta']:>+8.3f} │ "
              f"{r['cnr_change_pct']:>+8.1f} │ {r['tci_change_pct']:>+8.1f} │ "
              f"{r['epi_change_pct']:>+8.1f} │ {r['bs_change_pct']:>+8.1f} │ "
              f"{r['enl_change_pct']:>+8.1f} │ {r['snr_change_pct']:>+8.1f} │")
    print(f"└{'─'*90}┘")

    # Summary table with mean ± std and Wilcoxon
    print(f"\n┌{'─'*82}┐")
    print(f"│ {'Metric':<22} │ {'Backbone':>10} │ {'Corrected':>10} │ "
          f"{'Change':>12} │ {'p-value':>10} │")
    print(f"├{'─'*82}┤")

    summary = {}
    for name, bb_key, co_key, delta_key, use_pct in metrics:
        bb_vals = np.array([r[bb_key] for r in all_results])
        co_vals = np.array([r[co_key] for r in all_results])

        bb_mean, bb_std = bb_vals.mean(), bb_vals.std()
        co_mean, co_std = co_vals.mean(), co_vals.std()

        if use_pct:
            delta_vals = np.array([r[delta_key] for r in all_results])
            delta_mean, delta_std = delta_vals.mean(), delta_vals.std()
            change_str = f"{delta_mean:+.1f}±{delta_std:.1f}%"
        else:
            delta_vals = co_vals - bb_vals
            delta_mean, delta_std = delta_vals.mean(), delta_vals.std()
            change_str = f"{delta_mean:+.3f}±{delta_std:.3f}"

        # Wilcoxon signed-rank test (paired, two-sided)
        try:
            stat, p_val = wilcoxon(co_vals, bb_vals, alternative="two-sided")
            p_str = f"{p_val:.4f}"
            if p_val < 0.001:
                p_str = f"{p_val:.1e}"
        except Exception:
            p_str = "N/A"
            p_val = 1.0

        sig = "*" if p_val < 0.05 else ""
        if p_val < 0.01:
            sig = "**"
        if p_val < 0.001:
            sig = "***"

        print(f"│ {name:<22} │ {bb_mean:>7.3f}±{bb_std:<3.2f} │ "
              f"{co_mean:>7.3f}±{co_std:<3.2f} │ {change_str:>12} │ "
              f"{p_str:>7}{sig:>3} │")

        summary[name] = {
            "backbone_mean": float(bb_mean),
            "backbone_std": float(bb_std),
            "corrected_mean": float(co_mean),
            "corrected_std": float(co_std),
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
            "p_value": float(p_val) if isinstance(p_val, float) else None,
        }

    print(f"├{'─'*82}┤")

    # Correction magnitude
    mag_vals = np.array([r["correction_magnitude"] for r in all_results])
    print(f"│ {'Correction Magnitude':<22} │ {'':>10} │ "
          f"{mag_vals.mean():>7.6f}±{mag_vals.std():<3.5f} │ {'':>12} │ {'':>10} │")

    # SSIM
    ssim_bb = np.array([r["ssim_backbone"] for r in all_results])
    ssim_co = np.array([r["ssim_corrected"] for r in all_results])
    print(f"│ {'SSIM':<22} │ {ssim_bb.mean():>7.4f}±{ssim_bb.std():<3.3f} │ "
          f"{ssim_co.mean():>7.4f}±{ssim_co.std():<3.3f} │ {'':>12} │ {'':>10} │")

    print(f"└{'─'*82}┘")

    # Publication readiness check
    psnr_deltas = np.array([r["psnr_delta"] for r in all_results])
    cnr_deltas = np.array([r["cnr_change_pct"] for r in all_results])
    print(f"\n  Publication Readiness:")
    print(f"    Mean PSNR drop: {psnr_deltas.mean():+.3f} dB (target: < 1.0 dB)")
    print(f"    Mean CNR change: {cnr_deltas.mean():+.1f}% (target: >= 0%)")
    print(f"    Subjects with PSNR drop > 1.0 dB: "
          f"{sum(1 for d in psnr_deltas if d < -1.0)}/{n}")

    return summary


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.eval_only:
        args.skip_train = True

    # Load dataset config
    ds_cfg = DATASET_CONFIG[args.dataset]
    subjects = ds_cfg["subjects"]

    # Validate paths
    if not os.path.exists(ds_cfg["base"]):
        print(f"ERROR: {args.dataset} dataset not found at {ds_cfg['base']}")
        sys.exit(1)
    if not args.skip_train and not os.path.exists(args.resume):
        print(f"ERROR: Resume checkpoint not found at {args.resume}")
        sys.exit(1)
    if not os.path.exists(args.pretrained_backbone):
        print(f"ERROR: Pretrained backbone not found at {args.pretrained_backbone}")
        sys.exit(1)

    # Set output directory based on dataset + backbone
    global OUTPUT_BASE, FOLD_DIR
    OUTPUT_BASE = ds_cfg["output_base"]
    FOLD_DIR = ds_cfg["fold_dir"]
    if args.backbone != DEFAULT_BACKBONE:
        OUTPUT_BASE = f"outputs/{args.dataset}_loo_{args.backbone}"
    # An explicit path always wins, so a re-run can be kept apart from the results
    # of the original submission instead of overwriting them.
    if args.output_dir:
        OUTPUT_BASE = args.output_dir

    # Determine which folds to run
    if args.fold is not None:
        if args.fold not in subjects:
            print(f"ERROR: Subject {args.fold} not in {args.dataset}. "
                  f"Valid: {subjects}")
            sys.exit(1)
        folds = [args.fold]
    else:
        folds = subjects

    print(f"LOO Cross-Validation for {args.dataset}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Folds: {len(folds)} ({'all' if len(folds) == len(subjects) else folds})")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"  Resume: {args.resume}")
    print(f"  Pretrained backbone: {args.pretrained_backbone}")
    print(f"  Output: {OUTPUT_BASE}")

    # Step 1: Generate JSONLs
    print(f"\n{'─'*70}")
    print("Step 1: Generating fold JSONLs...")
    generate_fold_jsonls(ds_cfg)

    # Step 2: Train each fold
    if not args.skip_train:
        print(f"\n{'─'*70}")
        print("Step 2: Training folds...")
        for i, subj in enumerate(folds):
            print(f"\n  [{i+1}/{len(folds)}] Training fold for subject {subj}...")
            success = train_fold(subj, args, ds_cfg)
            if not success:
                print(f"  WARNING: Fold {subj} failed, continuing...")
    else:
        print(f"\n{'─'*70}")
        print("Step 2: Skipping training (--skip_train)")

    # Step 3: Evaluate each fold
    print(f"\n{'─'*70}")
    print("Step 3: Evaluating held-out subjects...")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    all_results = []
    for i, subj in enumerate(folds):
        print(f"\n  [{i+1}/{len(folds)}] Evaluating subject {subj}...")
        result = evaluate_fold(subj, args, ds_cfg)
        if result is not None:
            all_results.append(result)
            print(f"    PSNR: {result['psnr_backbone']:.2f} → {result['psnr_corrected']:.2f} "
                  f"({result['psnr_delta']:+.3f} dB)")
            print(f"    CNR: {result['cnr_backbone']:.2f} → {result['cnr_corrected']:.2f} "
                  f"({result['cnr_change_pct']:+.1f}%)")
            print(f"    Correction mag: {result['correction_magnitude']:.6f}")
        else:
            print(f"    SKIPPED (no checkpoint)")

    if not all_results:
        print("\nERROR: No folds evaluated successfully!")
        sys.exit(1)

    # Step 4: Aggregate and report
    print(f"\n{'─'*70}")
    print("Step 4: Aggregating results...")
    summary = aggregate_results(all_results)

    # Save results
    results_path = os.path.join(OUTPUT_BASE, "loo_results.json")
    output = {
        "dataset": args.dataset,
        "n_folds": len(all_results),
        "subjects": subjects,
        "per_subject": all_results,
        "summary": summary,
        "config": {
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "resume_checkpoint": args.resume,
            "pretrained_backbone": args.pretrained_backbone,
            "hidden_channels": args.hidden_channels,
        },
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"  LOO Cross-Validation COMPLETE")
    print(f"  {len(all_results)}/{len(folds)} folds evaluated successfully")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
