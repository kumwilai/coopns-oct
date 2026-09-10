#!/usr/bin/env python3
"""
Generate unified grid figures for IEEE TMI paper (Option 1).

Figure layout:
  - Same sample image across all 4 backbones
  - Row per backbone, columns: Backbone | CoopNS (Ours) | Clean
  - ROI crops only (compact) with PSNR/CNR annotated
  - Separate correction map grid: 1 row × 4 backbones

Usage:
  python generate_unified_grid.py [--sample_idx 34] [--device cpu]
"""

import argparse
import gc
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
)
from generate_paper_figures import (
    load_model,
    find_boundary_rois,
    BACKBONE_CONFIGS,
)
from validate_crossdataset import otsu_tissue_mask


BACKBONE_ORDER = ['nafnet', 'dncnn', 'swinir', 'kbnet']
DISPLAY_NAMES = {'nafnet': 'NAFNet', 'dncnn': 'DnCNN', 'swinir': 'SwinIR', 'kbnet': 'KBNet'}


def compute_cnr(img_tensor):
    """Compute CNR using Otsu mask."""
    signal_mask = otsu_tissue_mask(img_tensor)
    bg_mask = 1.0 - signal_mask
    sig_sum = signal_mask.sum().clamp(min=1.0)
    bg_sum = bg_mask.sum().clamp(min=1.0)
    sig_mean = (img_tensor * signal_mask).sum() / sig_sum
    bg_mean = (img_tensor * bg_mask).sum() / bg_sum
    bg_std = torch.sqrt(((img_tensor - bg_mean)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
    return ((sig_mean - bg_mean) / bg_std).clamp(-100, 100).item()


def run_all_backbones(dataset, sample_idx, device='cpu'):
    """Run all 4 backbones on the same sample, return results dict."""
    sample = dataset[sample_idx]
    clean = sample["clean"].unsqueeze(0).to(device)
    noisy = sample["noisy"].unsqueeze(0).to(device)

    results = {}
    for bname in BACKBONE_ORDER:
        config = BACKBONE_CONFIGS[bname]
        print(f"  Loading {DISPLAY_NAMES[bname]}...")
        model = load_model(config['checkpoint'], config['pretrained_backbone'],
                          bname, 64, device)

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
            'psnr_bb': psnr_bb,
            'psnr_co': psnr_co,
            'cnr_bb': cnr_bb,
            'cnr_co': cnr_co,
            'backbone_tensor': backbone_out,
            'corrected_tensor': corrected,
        }
        print(f"    PSNR: {psnr_bb:.2f} -> {psnr_co:.2f} ({psnr_co - psnr_bb:+.3f})")
        print(f"    CNR:  {cnr_bb:.2f} -> {cnr_co:.2f} ({(cnr_co - cnr_bb) / max(abs(cnr_bb), 1e-8) * 100:+.1f}%)")

        del model
        gc.collect()

    clean_np = clean.squeeze().cpu().numpy()
    noisy_np = noisy.squeeze().cpu().numpy()

    return results, clean_np, noisy_np, clean, noisy


def generate_roi_grid(results, clean_np, noisy_np, clean_tensor, rois, output_path):
    """
    Unified ROI grid figure.
    For each ROI: rows = backbones, cols = Backbone | CoopNS | Clean
    """
    n_rois = len(rois)
    n_backbones = len(BACKBONE_ORDER)
    roi_colors = ['#FF4444', '#44AAFF']

    # Layout: n_rois side by side, each with 3 columns
    # Total columns = n_rois * 3 with small gaps
    # Total rows = n_backbones + 1 (for full image row)

    fig_width = 3.2 * 3 * n_rois + 1.0
    fig_height = 3.0 * (n_backbones + 1) + 0.5

    fig = plt.figure(figsize=(min(fig_width, 20), min(fig_height, 16)))

    # Create gridspec: (n_backbones + 1) rows × (3 * n_rois) cols
    total_cols = 3 * n_rois
    gs = GridSpec(n_backbones + 1, total_cols, figure=fig,
                  hspace=0.20, wspace=0.08)

    # Row 0: Show full image with ROI boxes (span all columns for the first ROI set)
    # Show: Noisy | first backbone | first corrected | Clean across full width
    full_imgs = [noisy_np,
                 results[BACKBONE_ORDER[0]]['backbone_np'],
                 results[BACKBONE_ORDER[0]]['corrected_np'],
                 clean_np]
    full_titles = ['Noisy Input', f'{DISPLAY_NAMES[BACKBONE_ORDER[0]]}',
                   f'{DISPLAY_NAMES[BACKBONE_ORDER[0]]} + CoopNS', 'Clean']

    # Merge ROI columns for full image display
    cols_per_img = total_cols // 4
    remainder = total_cols % 4
    col_spans = []
    start = 0
    for i in range(4):
        span = cols_per_img + (1 if i < remainder else 0)
        col_spans.append((start, start + span))
        start += span

    for i, (img, title) in enumerate(zip(full_imgs, full_titles)):
        c0, c1 = col_spans[i]
        ax = fig.add_subplot(gs[0, c0:c1])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, fontweight='bold' if i == 2 else 'normal')
        ax.axis('off')
        for ri, (ry, rx, rh, rw) in enumerate(rois):
            color = roi_colors[ri % len(roi_colors)]
            rect = Rectangle((rx, ry), rw, rh, linewidth=2,
                            edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(rx + 3, ry + 14, f'{ri+1}', color=color,
                   fontsize=10, fontweight='bold',
                   path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # ROI crop rows: for each backbone, show crops for all ROIs
    for bb_idx, bname in enumerate(BACKBONE_ORDER):
        row = 1 + bb_idx
        r = results[bname]
        display = DISPLAY_NAMES[bname]

        for roi_idx, (ry, rx, rh, rw) in enumerate(rois):
            bb_crop = r['backbone_np'][ry:ry+rh, rx:rx+rw]
            co_crop = r['corrected_np'][ry:ry+rh, rx:rx+rw]
            cl_crop = clean_np[ry:ry+rh, rx:rx+rw]

            vmin = max(0, cl_crop.min() - 0.03)
            vmax = min(1, cl_crop.max() + 0.03)

            color = roi_colors[roi_idx % len(roi_colors)]
            base_col = roi_idx * 3

            crops = [bb_crop, co_crop, cl_crop]
            col_labels = [f'{display}', f'{display}+CoopNS', 'Clean']

            for ci, (crop, label) in enumerate(zip(crops, col_labels)):
                ax = fig.add_subplot(gs[row, base_col + ci])
                ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
                ax.axis('off')

                # Column titles on first backbone row only
                if bb_idx == 0:
                    roi_label = f'ROI {roi_idx+1}: ' if ci == 0 else ''
                    header = ['Backbone', 'CoopNS (Ours)', 'Clean'][ci]
                    ax.set_title(f'{roi_label}{header}', fontsize=8,
                                fontweight='bold' if ci == 1 else 'normal')

                # Border color matching ROI
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)

            # Backbone name label on leftmost column
            ax_left = fig.add_subplot(gs[row, base_col])
            # Add PSNR annotation on the backbone crop
            psnr_delta = r['psnr_co'] - r['psnr_bb']
            cnr_delta_pct = (r['cnr_co'] - r['cnr_bb']) / max(abs(r['cnr_bb']), 1e-8) * 100

            if roi_idx == 0:
                # Add backbone name + metrics as ylabel
                ax_first = fig.add_subplot(gs[row, 0])
                ax_first.set_ylabel(
                    f"{display}\n"
                    f"PSNR:{psnr_delta:+.2f}dB\n"
                    f"CNR:{cnr_delta_pct:+.1f}%",
                    fontsize=7, fontweight='bold',
                    rotation=0, labelpad=55, va='center',
                )

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  Saved ROI grid: {output_path}")


def generate_compact_roi_grid(results, clean_np, noisy_np, rois, output_path):
    """
    Compact grid: rows = backbones, cols = Noisy | Backbone | CoopNS | Clean
    Uses only the first (best) ROI. No text overlays — backbone names as ylabel.
    """
    roi = rois[0]  # Best ROI
    ry, rx, rh, rw = roi
    n_backbones = len(BACKBONE_ORDER)

    fig, axes = plt.subplots(n_backbones, 4, figsize=(12, 3.0 * n_backbones))

    noisy_crop = noisy_np[ry:ry+rh, rx:rx+rw]
    cl_crop = clean_np[ry:ry+rh, rx:rx+rw]
    vmin = max(0, cl_crop.min() - 0.03)
    vmax = min(1, cl_crop.max() + 0.03)

    for bb_idx, bname in enumerate(BACKBONE_ORDER):
        r = results[bname]
        display = DISPLAY_NAMES[bname]

        bb_crop = r['backbone_np'][ry:ry+rh, rx:rx+rw]
        co_crop = r['corrected_np'][ry:ry+rh, rx:rx+rw]

        crops = [noisy_crop, bb_crop, co_crop, cl_crop]

        for ci, crop in enumerate(crops):
            ax = axes[bb_idx, ci]
            ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column headers (top row only)
            if bb_idx == 0:
                headers = ['Noisy', 'Backbone', 'CoopNS (Ours)', 'Clean']
                ax.set_title(headers[ci], fontsize=11,
                            fontweight='bold' if ci == 2 else 'normal')

        # Row label — backbone name only, no metrics
        axes[bb_idx, 0].set_ylabel(display, fontsize=12, fontweight='bold',
                                    rotation=0, labelpad=45, va='center')

    plt.subplots_adjust(wspace=0.02, hspace=0.08)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved compact grid: {output_path}")


def generate_correction_map_grid(results, clean_np, output_path):
    """
    1 row × 4 backbones: correction magnitude maps.
    """
    fig, axes = plt.subplots(1, len(BACKBONE_ORDER), figsize=(4.5 * len(BACKBONE_ORDER), 4.5))

    for i, bname in enumerate(BACKBONE_ORDER):
        r = results[bname]
        display = DISPLAY_NAMES[bname]

        correction = r['corrected_np'] - r['backbone_np']
        amp = correction * 10  # Amplify 10x

        vabs = max(np.percentile(np.abs(amp), 99), 1e-6)
        im = axes[i].imshow(amp, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        axes[i].set_title(f'{display}\nCorrection ×10', fontsize=11, fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved correction map grid: {output_path}")


def generate_full_comparison_grid(results, clean_np, noisy_np, rois, output_path):
    """
    Publication figure: Full image row + ROI crops per backbone.
    Row 0: Noisy | [best backbone] | [best CoopNS] | Clean (full images with ROI boxes)
    Rows 1-4: Per backbone, 2 ROI crops each showing Backbone vs CoopNS vs Clean
    """
    n_backbones = len(BACKBONE_ORDER)
    n_rois = len(rois)
    roi_colors = ['#FF4444', '#44AAFF']

    # Layout: (1 + n_backbones) rows × (3 * n_rois) cols
    n_crop_cols = 3 * n_rois
    fig_width = 3.5 * n_crop_cols
    fig_height = 3.5 + 3.0 * n_backbones

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Top section: full images (use separate axes)
    gs_top = GridSpec(1, 4, figure=fig,
                      left=0.06, right=0.98, top=0.98,
                      bottom=0.98 - 3.2/fig_height,
                      wspace=0.05)

    full_imgs = [noisy_np, results['nafnet']['backbone_np'],
                 results['nafnet']['corrected_np'], clean_np]
    full_titles = ['Noisy Input', 'NAFNet (Backbone)',
                   'NAFNet + CoopNS (Ours)', 'Clean Reference']

    for i, (img, title) in enumerate(zip(full_imgs, full_titles)):
        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, fontweight='bold' if i == 2 else 'normal')
        ax.axis('off')
        for ri, (ry, rx, rh, rw) in enumerate(rois):
            color = roi_colors[ri % len(roi_colors)]
            rect = Rectangle((rx, ry), rw, rh, linewidth=2,
                            edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(rx + 3, ry + 14, f'{ri+1}', color=color, fontsize=10,
                   fontweight='bold',
                   path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Bottom section: ROI crops grid
    bottom_top = 0.98 - 3.5/fig_height
    gs_bot = GridSpec(n_backbones, n_crop_cols, figure=fig,
                      left=0.06, right=0.98,
                      top=bottom_top, bottom=0.02,
                      hspace=0.25, wspace=0.08)

    for bb_idx, bname in enumerate(BACKBONE_ORDER):
        r = results[bname]
        display = DISPLAY_NAMES[bname]
        psnr_delta = r['psnr_co'] - r['psnr_bb']
        cnr_delta_pct = (r['cnr_co'] - r['cnr_bb']) / max(abs(r['cnr_bb']), 1e-8) * 100

        for roi_idx, (ry, rx, rh, rw) in enumerate(rois):
            bb_crop = r['backbone_np'][ry:ry+rh, rx:rx+rw]
            co_crop = r['corrected_np'][ry:ry+rh, rx:rx+rw]
            cl_crop = clean_np[ry:ry+rh, rx:rx+rw]
            vmin = max(0, cl_crop.min() - 0.03)
            vmax = min(1, cl_crop.max() + 0.03)

            color = roi_colors[roi_idx % len(roi_colors)]
            base_col = roi_idx * 3

            for ci, crop in enumerate([bb_crop, co_crop, cl_crop]):
                ax = fig.add_subplot(gs_bot[bb_idx, base_col + ci])
                ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
                ax.axis('off')
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)

                # Column headers
                if bb_idx == 0:
                    roi_prefix = f'ROI {roi_idx+1}: ' if ci == 0 else ''
                    header = ['Backbone', 'CoopNS (Ours)', 'Clean'][ci]
                    ax.set_title(f'{roi_prefix}{header}', fontsize=8,
                                fontweight='bold' if ci == 1 else 'normal')

        # Row label with metrics
        ax0 = fig.add_subplot(gs_bot[bb_idx, 0])
        ax0.set_ylabel(
            f"{display}\n$\\Delta$PSNR:{psnr_delta:+.2f}\n$\\Delta$CNR:{cnr_delta_pct:+.1f}%",
            fontsize=7, fontweight='bold', rotation=0, labelpad=60, va='center',
        )

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  Saved full comparison grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate unified grid figures")
    parser.add_argument('--test_jsonl', default='pku37_oct_dataset/pku37_real_test.jsonl')
    parser.add_argument('--sample_idx', type=int, default=34,
                       help='Sample index to use (default: 34)')
    parser.add_argument('--sample_indices', type=str, default=None,
                       help='Comma-separated sample indices for multiple figures')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='subjective_quality/unified')
    parser.add_argument('--n_rois', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PKU37Dataset(args.test_jsonl, patch_size=0, is_train=False)
    print(f"Dataset: {len(dataset)} images")

    if args.sample_indices:
        indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    else:
        indices = [args.sample_idx]

    for sample_idx in indices:
        print(f"\n{'='*60}")
        print(f"  Sample {sample_idx}")
        print(f"{'='*60}")

        results, clean_np, noisy_np, clean_t, noisy_t = run_all_backbones(
            dataset, sample_idx, args.device)

        # Find ROIs using first backbone (NAFNet)
        rois = find_boundary_rois(
            results['nafnet']['backbone_tensor'],
            results['nafnet']['corrected_tensor'],
            clean_t, n_rois=args.n_rois,
        )
        print(f"  ROIs: {rois}")

        # Generate all figure variants
        prefix = f"sample_{sample_idx:03d}"

        # Compact: single ROI, 4 backbones × 3 cols
        generate_compact_roi_grid(
            results, clean_np, noisy_np, rois,
            os.path.join(args.output_dir, f'{prefix}_compact.png'),
        )

        # Full: full image + ROI crops per backbone
        generate_full_comparison_grid(
            results, clean_np, noisy_np, rois,
            os.path.join(args.output_dir, f'{prefix}_full.png'),
        )

        # Correction map grid: 1 row × 4 backbones
        generate_correction_map_grid(
            results, clean_np,
            os.path.join(args.output_dir, f'{prefix}_corrmap.png'),
        )

        # Cleanup tensors
        for bname in BACKBONE_ORDER:
            del results[bname]['backbone_tensor']
            del results[bname]['corrected_tensor']
        gc.collect()

    print(f"\nDone! Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
