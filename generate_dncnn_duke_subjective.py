#!/usr/bin/env python3
"""Generate subjective quality figures for best DnCNN Duke LOO subjects."""

import os
import gc
import torch
from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
)
from generate_paper_figures import (
    load_model,
    find_boundary_rois,
    generate_comparison_figure,
    generate_correction_map,
)

BACKBONE_PATH = "NukeModel/dncnn_7m/best.pth"
OUTPUT_DIR = "subjective_quality"

# Best subjects from DnCNN LOO
DUKE17_BEST = [9, 8, 13, 1, 6]
DUKE2013_BEST = [11, 8, 14, 12, 15]


def run_duke_subjects(dataset_name, subjects, fold_base, jsonl_base):
    out_dir = os.path.join(OUTPUT_DIR, f"dncnn_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    for subj in subjects:
        ckpt = os.path.join(fold_base, f"fold_{subj}", "best_model_cooperative.pth")
        val_jsonl = os.path.join(jsonl_base, f"fold_{subj}_val.jsonl")

        if not os.path.exists(ckpt):
            print(f"  Skipping {dataset_name} subj {subj}: no checkpoint")
            continue

        print(f"\n  {dataset_name} Subject {subj}")

        model = load_model(ckpt, BACKBONE_PATH, "dncnn", 64, "cpu")
        dataset = PKU37Dataset(val_jsonl, patch_size=0, is_train=False)
        sample = dataset[0]
        clean = sample["clean"].unsqueeze(0)
        noisy = sample["noisy"].unsqueeze(0)

        with torch.no_grad():
            backbone_out, unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=unc, return_details=False,
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
            backbone_name="DnCNN",
        )
        generate_correction_map(
            backbone_np, corrected_np, clean_np,
            os.path.join(out_dir, f"corrmap_{subj:02d}.png"),
            backbone_name="DnCNN",
        )

        del model, dataset
        gc.collect()


if __name__ == "__main__":
    print("=== DnCNN Duke17 Best Subjects ===")
    run_duke_subjects("duke17", DUKE17_BEST,
                      "outputs/duke17_loo_dncnn", "duke17_loo_folds")

    print("\n=== DnCNN Duke2013 Best Subjects ===")
    run_duke_subjects("duke2013", DUKE2013_BEST,
                      "outputs/duke2013_loo_dncnn", "duke2013_loo_folds")

    print(f"\nDone!")
