#!/usr/bin/env python3
"""Generate unified grid figure for Duke17 LOO subject across all 4 backbones."""

import gc
import os
import torch
from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
)
from generate_paper_figures import load_model, find_boundary_rois
from generate_unified_grid import (
    BACKBONE_ORDER, DISPLAY_NAMES,
    compute_cnr, generate_compact_roi_grid,
)

# Duke LOO fold configs: backbone_name -> (fold_dir_pattern, backbone_path)
DUKE_CONFIGS = {
    'duke17': {
        'nafnet': ('outputs/duke17_loo', 'outputs/nafnet_pku37_w40/best_model.pth'),
        'dncnn': ('outputs/duke17_loo_dncnn', 'NukeModel/dncnn_7m/best.pth'),
        'swinir': ('outputs/duke17_loo_swinir', 'NukeModel/swinir_7m/best.pth'),
        'kbnet': ('outputs/duke17_loo_kbnet', 'NukeModel/kbnet_7m/best.pth'),
    },
    'duke2013': {
        'nafnet': ('outputs/duke2013_loo', 'outputs/nafnet_pku37_w40/best_model.pth'),
        'dncnn': ('outputs/duke2013_loo_dncnn', 'NukeModel/dncnn_7m/best.pth'),
        'swinir': ('outputs/duke2013_loo_swinir', 'NukeModel/swinir_7m/best.pth'),
        'kbnet': ('outputs/duke2013_loo_kbnet', 'NukeModel/kbnet_7m/best.pth'),
    },
}


def run_duke_unified(subject, dataset_name='duke17', device='cpu'):
    """Run all 4 backbones on a Duke LOO subject."""
    fold_dir_base = DUKE_CONFIGS[dataset_name]

    val_jsonl = f"{dataset_name}_loo_folds/fold_{subject}_val.jsonl"
    dataset = PKU37Dataset(val_jsonl, patch_size=0, is_train=False)
    sample = dataset[0]
    clean = sample["clean"].unsqueeze(0).to(device)
    noisy = sample["noisy"].unsqueeze(0).to(device)

    results = {}
    for bname in BACKBONE_ORDER:
        fold_base, backbone_path = fold_dir_base[bname]
        ckpt = os.path.join(fold_base, f"fold_{subject}", "best_model_cooperative.pth")
        if not os.path.exists(ckpt):
            print(f"  SKIP {bname}: {ckpt} not found")
            continue

        print(f"  Loading {DISPLAY_NAMES[bname]} (fold {subject})...")
        model = load_model(ckpt, backbone_path, bname, 64, device)

        with torch.no_grad():
            backbone_out, unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=unc, return_details=False,
            )

        psnr_bb = compute_psnr(backbone_out, clean)
        psnr_co = compute_psnr(corrected, clean)
        cnr_bb = compute_cnr(backbone_out)
        cnr_co = compute_cnr(corrected)

        results[bname] = {
            'backbone_np': backbone_out.squeeze().cpu().numpy(),
            'corrected_np': corrected.squeeze().cpu().numpy(),
            'psnr_bb': psnr_bb, 'psnr_co': psnr_co,
            'cnr_bb': cnr_bb, 'cnr_co': cnr_co,
            'backbone_tensor': backbone_out,
            'corrected_tensor': corrected,
        }
        print(f"    PSNR: {psnr_bb:.2f} -> {psnr_co:.2f} ({psnr_co - psnr_bb:+.3f})")
        print(f"    CNR:  {cnr_bb:.2f} -> {cnr_co:.2f} ({(cnr_co - cnr_bb)/max(abs(cnr_bb),1e-8)*100:+.1f}%)")

        del model
        gc.collect()

    clean_np = clean.squeeze().cpu().numpy()
    noisy_np = noisy.squeeze().cpu().numpy()
    return results, clean_np, noisy_np, clean, noisy


def generate_duke_grid(dataset_name, subject, out_dir):
    """Generate compact unified grid for a Duke LOO subject."""
    print(f"\n=== {dataset_name} Subject {subject} - Unified Grid ===")
    results, clean_np, noisy_np, clean_t, noisy_t = run_duke_unified(
        subject, dataset_name=dataset_name)

    rois = find_boundary_rois(
        results['nafnet']['backbone_tensor'],
        results['nafnet']['corrected_tensor'],
        clean_t, n_rois=2,
    )
    print(f"  ROIs: {rois}")

    outfile = os.path.join(out_dir, f"{dataset_name}_subj{subject:02d}_compact.png")
    generate_compact_roi_grid(results, clean_np, noisy_np, rois, outfile)

    for bname in BACKBONE_ORDER:
        if bname in results:
            del results[bname]['backbone_tensor']
            del results[bname]['corrected_tensor']
    gc.collect()
    return outfile


if __name__ == "__main__":
    out_dir = "subjective_quality/unified_final"
    os.makedirs(out_dir, exist_ok=True)

    generate_duke_grid("duke17", 9, out_dir)
    generate_duke_grid("duke2013", 11, out_dir)

    print("\nDone!")
