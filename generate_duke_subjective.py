#!/usr/bin/env python3
"""Generate subjective quality figures for best Duke LOO subjects."""

import os
import torch
import numpy as np
from PIL import Image

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
    compute_ssim,
)
from generate_paper_figures import (
    load_model,
    find_boundary_rois,
    generate_comparison_figure,
    generate_correction_map,
)

# Best subjects from LOO results:
# Duke17:  subj 4 (+0.201), 5 (+0.155), 13 (+0.092), 9 (-0.054), 12 (-0.121)
# Duke2013: subj 11 (+0.121), 3 (+0.049), 9 (-0.046), 1 (-0.317), 15 (-0.277)

DUKE17_BEST = [4, 5, 13, 9, 12]
DUKE2013_BEST = [11, 3, 9, 1, 15]

BACKBONE_PATH = "outputs/nafnet_pku37_w40/best_model.pth"
OUTPUT_DIR = "duke_subjective"


def run_duke_subjects(dataset_name, subjects, fold_base, jsonl_base):
    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for subj in subjects:
        ckpt = os.path.join(fold_base, f"fold_{subj}", "best_model_cooperative.pth")
        val_jsonl = os.path.join(jsonl_base, f"fold_{subj}_val.jsonl")

        if not os.path.exists(ckpt):
            print(f"  Skipping {dataset_name} subj {subj}: no checkpoint")
            continue

        print(f"\n{'='*60}")
        print(f"  {dataset_name} Subject {subj}")
        print(f"{'='*60}")

        model = load_model(ckpt, BACKBONE_PATH, "nafnet", 64, "cpu")

        dataset = PKU37Dataset(val_jsonl, patch_size=0, is_train=False)
        sample = dataset[0]
        clean = sample["clean"].unsqueeze(0)
        noisy = sample["noisy"].unsqueeze(0)

        with torch.no_grad():
            backbone_out, nafnet_unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=nafnet_unc, return_details=False,
            )

        psnr_bb = compute_psnr(backbone_out, clean)
        psnr_co = compute_psnr(corrected, clean)
        print(f"  PSNR: {psnr_bb:.2f} -> {psnr_co:.2f} ({psnr_co - psnr_bb:+.3f})")

        rois = find_boundary_rois(backbone_out, corrected, clean, n_rois=2)

        clean_np = clean.squeeze().cpu().numpy()
        noisy_np = noisy.squeeze().cpu().numpy()
        backbone_np = backbone_out.squeeze().cpu().numpy()
        corrected_np = corrected.squeeze().cpu().numpy()

        generate_comparison_figure(
            noisy_np, backbone_np, corrected_np, clean_np,
            rois, None,
            os.path.join(out_dir, f"subj_{subj:02d}.png"),
            sample_name=f"{dataset_name} Subject {subj}",
            backbone_name="NAFNet",
        )

        generate_correction_map(
            backbone_np, corrected_np, clean_np,
            os.path.join(out_dir, f"corrmap_{subj:02d}.png"),
            backbone_name="NAFNet",
        )

        del model, dataset
        import gc; gc.collect()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Duke17 Best Subjects ===")
    run_duke_subjects(
        "duke17", DUKE17_BEST,
        "outputs/duke17_loo", "duke17_loo_folds",
    )

    print("\n=== Duke2013 Best Subjects ===")
    run_duke_subjects(
        "duke2013", DUKE2013_BEST,
        "outputs/duke2013_loo", "duke2013_loo_folds",
    )

    print(f"\nDone! Figures saved to {OUTPUT_DIR}/")
