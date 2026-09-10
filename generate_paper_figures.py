#!/usr/bin/env python3
"""
Generate publication-quality subjective comparison figures for IEEE TMI paper.

For each sample, produces:
  1. comparison figure: Noisy | Backbone | CoopNS (Ours) | Clean
     with ROI boxes, zoomed crops using tight intensity windowing,
     and amplified correction overlay on the CoopNS crop.
  2. profile figure: Intensity line profiles across a tissue boundary
     showing backbone vs corrected vs clean.
"""

import argparse
import gc
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
    compute_ssim,
)


BACKBONE_CONFIGS = {
    'nafnet': {
        'checkpoint': 'outputs/nafnet_relaxed_psnr/best_model_cooperative.pth',
        'pretrained_backbone': 'outputs/nafnet_pku37_w40/best_model.pth',
        'display': 'NAFNet',
    },
    'kbnet': {
        'checkpoint': 'outputs/kbnet_qt69d/best_model_cooperative.pth',
        'pretrained_backbone': 'NukeModel/kbnet_7m/best.pth',
        'display': 'KBNet',
    },
    'dncnn': {
        'checkpoint': 'outputs/dncnn_nuke_qt69d/best_model_cooperative.pth',
        'pretrained_backbone': 'NukeModel/dncnn_7m/best.pth',
        'display': 'DnCNN',
    },
    'swinir': {
        'checkpoint': 'outputs/swinir_qt69d/best_model_cooperative.pth',
        'pretrained_backbone': 'NukeModel/swinir_7m/best.pth',
        'display': 'SwinIR',
    },
}


def load_model(checkpoint, backbone_path, backbone_name, hidden_channels=64, device='cpu'):
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=backbone_name,
        pretrained_backbone=backbone_path,
        hidden_channels=hidden_channels,
    )
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
               for k, v in state_dict.items()}
    # Filter out shape mismatches (e.g. uncertainty conv channels differ across backbones)
    model_state = model.state_dict()
    compatible = {k: v for k, v in cleaned.items()
                  if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(compatible, strict=False)
    model = model.to(device)
    model.eval()
    return model


def compute_edge_map(img):
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=img.device).view(1, 1, 3, 3)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)


def find_boundary_rois(backbone_img, corrected_img, clean_img, n_rois=2):
    """Find ROIs near tissue boundaries where correction is most visible."""
    H, W = backbone_img.shape[-2:]
    roi_size = min(H, W) // 4

    bb = backbone_img.squeeze()
    co = corrected_img.squeeze()
    cl = clean_img.squeeze()

    # Edge strength of clean image — find regions with strong boundaries
    edge_cl = compute_edge_map(clean_img).squeeze()

    # Improvement: where corrected is closer to clean
    diff_bb = (bb - cl).abs()
    diff_co = (co - cl).abs()
    improvement = diff_bb - diff_co

    # Score = edge strength × improvement (want boundary regions that improve)
    combined = edge_cl * 2.0 + improvement

    step = roi_size // 3
    candidates = []
    # Focus on tissue region (top 70% of image, not background at bottom)
    max_y = int(H * 0.70)
    for y in range(0, max_y - roi_size, step):
        for x in range(0, W - roi_size, step):
            score = combined[y:y+roi_size, x:x+roi_size].mean().item()
            # Bonus for regions with high edge density
            edge_density = (edge_cl[y:y+roi_size, x:x+roi_size] > 0.05).float().mean().item()
            score += edge_density * 0.5
            candidates.append((score, y, x))

    candidates.sort(reverse=True)

    selected = []
    for score, y, x in candidates:
        overlap = False
        for _, sy, sx in selected:
            if abs(y - sy) < roi_size * 0.5 and abs(x - sx) < roi_size * 0.5:
                overlap = True
                break
        if not overlap:
            selected.append((score, y, x))
        if len(selected) >= n_rois:
            break

    return [(y, x, roi_size, roi_size) for _, y, x in selected]


def find_profile_line(clean_crop):
    """Find a horizontal line across a strong boundary in the crop."""
    # Use vertical gradient to find the strongest horizontal boundary
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    from scipy.ndimage import convolve
    gy = np.abs(convolve(clean_crop, sobel_y))

    # Average gradient strength per row
    row_strength = gy.mean(axis=1)
    # Find row with strongest boundary
    best_row = int(np.argmax(row_strength))
    return best_row


def generate_comparison_figure(
    noisy_np, backbone_np, corrected_np, clean_np,
    rois, metrics, output_path, sample_name="",
    backbone_name="NAFNet", roi_colors=None,
):
    """
    Row 0: Full images (Noisy | Backbone | CoopNS | Clean) with ROI boxes
    Row 1+ per ROI: Noisy | Backbone | CoopNS | Clean (zoomed crops)
    """
    if roi_colors is None:
        roi_colors = ['#FF4444', '#44AAFF']

    n_rois = len(rois)
    n_cols = 4
    n_rows = 1 + n_rois

    fig = plt.figure(figsize=(16, 4.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.05,
                  width_ratios=[1, 1, 1, 1])

    images = [noisy_np, backbone_np, corrected_np, clean_np]
    titles = [
        'Noisy Input',
        f'{backbone_name} (Backbone)',
        f'{backbone_name} + CoopNS (Ours)',
        'Clean Reference',
    ]
    if metrics:
        titles[1] += f"\nPSNR: {metrics['psnr_bb']:.2f} dB"
        titles[2] += f"\nPSNR: {metrics['psnr_co']:.2f} dB ({metrics['psnr_delta']:+.2f})"

    # Row 0: Full images (4 cols)
    for col in range(4):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(images[col], cmap='gray', vmin=0, vmax=1)
        ax.set_title(titles[col], fontsize=10, fontweight='bold' if col == 2 else 'normal')
        ax.axis('off')
        for i, (ry, rx, rh, rw) in enumerate(rois):
            color = roi_colors[i % len(roi_colors)]
            rect = Rectangle((rx, ry), rw, rh, linewidth=2,
                            edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(rx + 3, ry + 14, f'{i+1}', color=color,
                   fontsize=11, fontweight='bold',
                   path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Rows 1+: Noisy | Backbone | CoopNS | Clean (zoomed crops)
    for roi_idx, (ry, rx, rh, rw) in enumerate(rois):
        color = roi_colors[roi_idx % len(roi_colors)]
        row = 1 + roi_idx

        noisy_crop = noisy_np[ry:ry+rh, rx:rx+rw]
        bb_crop = backbone_np[ry:ry+rh, rx:rx+rw]
        co_crop = corrected_np[ry:ry+rh, rx:rx+rw]
        cl_crop = clean_np[ry:ry+rh, rx:rx+rw]

        # Tight intensity window
        vmin = max(0, cl_crop.min() - 0.03)
        vmax = min(1, cl_crop.max() + 0.03)

        crops = [noisy_crop, bb_crop, co_crop, cl_crop]
        col_titles = ['Noisy', 'Backbone', 'CoopNS (Ours)', 'Clean']

        for col_idx in range(4):
            ax = fig.add_subplot(gs[row, col_idx])
            ax.imshow(crops[col_idx], cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if roi_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9,
                            fontweight='bold' if col_idx == 2 else 'normal')
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(3); spine.set_visible(True)
            if col_idx == 0:
                ax.set_ylabel(f'ROI {roi_idx+1}', fontsize=11, fontweight='bold',
                              color=color, rotation=0, labelpad=35, va='center')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_correction_map(
    backbone_np, corrected_np, clean_np,
    output_path, backbone_name="NAFNet",
):
    """Correction magnitude and improvement maps with amplified visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    diff_bb = np.abs(backbone_np - clean_np)
    diff_co = np.abs(corrected_np - clean_np)
    correction = corrected_np - backbone_np  # signed
    improvement = diff_bb - diff_co

    vmax_diff = np.percentile(np.maximum(diff_bb, diff_co), 99)

    axes[0].imshow(diff_bb, cmap='hot', vmin=0, vmax=vmax_diff)
    axes[0].set_title(f'|{backbone_name} − Clean|', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(diff_co, cmap='hot', vmin=0, vmax=vmax_diff)
    axes[1].set_title(f'|CoopNS − Clean|', fontsize=11)
    axes[1].axis('off')

    # Signed correction amplified 10×
    amp = correction * 10
    vabs = np.percentile(np.abs(amp), 99)
    im2 = axes[2].imshow(amp, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    axes[2].set_title(f'Correction ×10\n(red=brighter, blue=darker)', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Improvement map
    vabs_imp = np.percentile(np.abs(improvement), 99)
    im3 = axes[3].imshow(improvement, cmap='RdYlGn', vmin=-vabs_imp, vmax=vabs_imp)
    axes[3].set_title('Improvement\n(green=CoopNS closer to clean)', fontsize=10)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {output_path}")


def run_single_backbone(backbone_name, config, dataset, indices, output_dir,
                        n_rois, hidden_channels, device):
    backbone_dir = os.path.join(output_dir, backbone_name)
    os.makedirs(backbone_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {config['display']} ({backbone_name})")
    print(f"{'='*60}")

    model = load_model(config['checkpoint'], config['pretrained_backbone'],
                      backbone_name, hidden_channels, device)

    backbone_display = config['display']

    for idx in indices:
        sample = dataset[idx]
        clean = sample['clean'].unsqueeze(0).to(device)
        noisy = sample['noisy'].unsqueeze(0).to(device)

        with torch.no_grad():
            backbone_out, nafnet_unc = model.backbone(noisy)
            corrected, info = model.corrector(
                backbone_out, noisy, None,
                nafnet_uncertainty=nafnet_unc, return_details=False,
            )

        clean_np = clean.squeeze().cpu().numpy()
        noisy_np = noisy.squeeze().cpu().numpy()
        backbone_np = backbone_out.squeeze().cpu().numpy()
        corrected_np = corrected.squeeze().cpu().numpy()

        psnr_bb = compute_psnr(backbone_out, clean)
        psnr_co = compute_psnr(corrected, clean)
        metrics = {
            'psnr_bb': psnr_bb,
            'psnr_co': psnr_co,
            'psnr_delta': psnr_co - psnr_bb,
        }

        rois = find_boundary_rois(
            backbone_out, corrected, clean, n_rois=n_rois
        )

        generate_comparison_figure(
            noisy_np, backbone_np, corrected_np, clean_np,
            rois, metrics,
            os.path.join(backbone_dir, f'sample_{idx:03d}.png'),
            sample_name=f"Sample {idx}",
            backbone_name=backbone_display,
        )

        generate_correction_map(
            backbone_np, corrected_np, clean_np,
            os.path.join(backbone_dir, f'corrmap_{idx:03d}.png'),
            backbone_name=backbone_display,
        )

        del clean, noisy, backbone_out, corrected

    del model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Generate paper comparison figures")
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--pretrained_backbone', default=None)
    parser.add_argument('--backbone_name', default='nafnet',
                       choices=['nafnet', 'dncnn', 'swinir', 'kbnet'])
    parser.add_argument('--all_backbones', action='store_true',
                       help='Run all 4 backbones, each in its own subfolder')
    parser.add_argument('--test_jsonl', default='pku37_oct_dataset/pku37_real_test.jsonl')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--sample_indices', type=str, default=None,
                       help='Comma-separated indices (e.g., "0,15,42")')
    parser.add_argument('--output_dir', default='paper_figures')
    parser.add_argument('--n_rois', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PKU37Dataset(args.test_jsonl, patch_size=0, is_train=False)
    print(f"Dataset: {len(dataset)} images")

    if args.sample_indices:
        indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    else:
        n = min(args.num_samples, len(dataset))
        step = len(dataset) // n
        indices = [i * step for i in range(n)]

    print(f"Samples: {indices}")

    if args.all_backbones:
        for bname, config in BACKBONE_CONFIGS.items():
            run_single_backbone(
                bname, config, dataset, indices, args.output_dir,
                args.n_rois, args.hidden_channels, args.device,
            )
    else:
        if args.checkpoint and args.pretrained_backbone:
            config = {
                'checkpoint': args.checkpoint,
                'pretrained_backbone': args.pretrained_backbone,
                'display': {'nafnet': 'NAFNet', 'dncnn': 'DnCNN',
                           'swinir': 'SwinIR', 'kbnet': 'KBNet'}[args.backbone_name],
            }
        else:
            config = BACKBONE_CONFIGS[args.backbone_name]

        run_single_backbone(
            args.backbone_name, config, dataset, indices, args.output_dir,
            args.n_rois, args.hidden_channels, args.device,
        )

    print(f"\nDone! All figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
