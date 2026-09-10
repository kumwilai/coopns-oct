#!/usr/bin/env python3
"""
Visualize V8 Neuro-Symbolic Corrections

Shows before/after comparisons with clinical metrics overlay.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

from neuro_symbolic_denoiser_v8_enhanced import NeuroSymbolicDenoiserV8Enhanced
from train_v8_enhanced import compute_psnr, compute_ssim


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained V8 model."""
    model = NeuroSymbolicDenoiserV8Enhanced(
        backbone_width=40,
        freeze_backbone=True
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def compute_edge_map(img: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude."""
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(img.device)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(img.device)

    if img.dim() == 3:
        img = img.unsqueeze(0)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)


def compute_local_std(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Compute local standard deviation (contrast map)."""
    if img.dim() == 3:
        img = img.unsqueeze(0)

    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size ** 2)
    padding = kernel_size // 2

    local_mean = F.conv2d(img, kernel, padding=padding)
    local_mean_sq = F.conv2d(img ** 2, kernel, padding=padding)
    local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=1e-6)

    return torch.sqrt(local_var)


def visualize_single_sample(model, noisy: torch.Tensor, clean: torch.Tensor,
                           output_path: str, sample_idx: int = 0):
    """
    Visualize a single sample with all corrections and metrics.
    """
    device = next(model.parameters()).device
    noisy = noisy.to(device)
    clean = clean.to(device)

    with torch.no_grad():
        corrected, backbone_out, info = model(noisy, return_details=True)

    # Compute metrics
    psnr_noisy = compute_psnr(noisy, clean)
    psnr_backbone = compute_psnr(backbone_out, clean)
    psnr_corrected = compute_psnr(corrected, clean)

    ssim_backbone = compute_ssim(backbone_out, clean)
    ssim_corrected = compute_ssim(corrected, clean)

    # Compute difference maps
    correction = (corrected - backbone_out).abs()
    improvement = (backbone_out - clean).abs() - (corrected - clean).abs()

    # Compute clinical feature maps
    edge_clean = compute_edge_map(clean)
    edge_backbone = compute_edge_map(backbone_out)
    edge_corrected = compute_edge_map(corrected)

    contrast_clean = compute_local_std(clean)
    contrast_backbone = compute_local_std(backbone_out)
    contrast_corrected = compute_local_std(corrected)

    # Convert to numpy for plotting
    def to_np(x):
        return x.squeeze().cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    # Row 1: Main images
    ax1 = fig.add_subplot(4, 5, 1)
    ax1.imshow(to_np(noisy), cmap='gray')
    ax1.set_title(f'Noisy\nPSNR: {psnr_noisy:.2f} dB')
    ax1.axis('off')

    ax2 = fig.add_subplot(4, 5, 2)
    ax2.imshow(to_np(backbone_out), cmap='gray')
    ax2.set_title(f'Backbone (NAFNet)\nPSNR: {psnr_backbone:.2f} dB')
    ax2.axis('off')

    ax3 = fig.add_subplot(4, 5, 3)
    ax3.imshow(to_np(corrected), cmap='gray')
    ax3.set_title(f'V8 Corrected\nPSNR: {psnr_corrected:.2f} dB')
    ax3.axis('off')

    ax4 = fig.add_subplot(4, 5, 4)
    ax4.imshow(to_np(clean), cmap='gray')
    ax4.set_title('Ground Truth')
    ax4.axis('off')

    ax5 = fig.add_subplot(4, 5, 5)
    corr_map = to_np(correction)
    im = ax5.imshow(corr_map, cmap='hot', vmin=0, vmax=corr_map.max())
    ax5.set_title(f'Correction Map\nMean: {corr_map.mean():.4f}')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5, fraction=0.046)

    # Row 2: Error maps
    ax6 = fig.add_subplot(4, 5, 6)
    err_backbone = to_np((backbone_out - clean).abs())
    ax6.imshow(err_backbone, cmap='hot', vmin=0, vmax=0.2)
    ax6.set_title('Backbone Error')
    ax6.axis('off')

    ax7 = fig.add_subplot(4, 5, 7)
    err_corrected = to_np((corrected - clean).abs())
    ax7.imshow(err_corrected, cmap='hot', vmin=0, vmax=0.2)
    ax7.set_title('Corrected Error')
    ax7.axis('off')

    ax8 = fig.add_subplot(4, 5, 8)
    imp_map = to_np(improvement)
    ax8.imshow(imp_map, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
    ax8.set_title('Improvement\n(green=better)')
    ax8.axis('off')

    # Row 2 continued: Edge maps
    ax9 = fig.add_subplot(4, 5, 9)
    ax9.imshow(to_np(edge_backbone), cmap='gray')
    ax9.set_title('Backbone Edges')
    ax9.axis('off')

    ax10 = fig.add_subplot(4, 5, 10)
    ax10.imshow(to_np(edge_corrected), cmap='gray')
    ax10.set_title('Corrected Edges')
    ax10.axis('off')

    # Row 3: Contrast maps
    ax11 = fig.add_subplot(4, 5, 11)
    ax11.imshow(to_np(contrast_clean), cmap='viridis')
    ax11.set_title('GT Local Contrast')
    ax11.axis('off')

    ax12 = fig.add_subplot(4, 5, 12)
    ax12.imshow(to_np(contrast_backbone), cmap='viridis')
    ax12.set_title('Backbone Contrast')
    ax12.axis('off')

    ax13 = fig.add_subplot(4, 5, 13)
    ax13.imshow(to_np(contrast_corrected), cmap='viridis')
    ax13.set_title('Corrected Contrast')
    ax13.axis('off')

    # Row 3 continued: Predicate scores
    ax14 = fig.add_subplot(4, 5, 14)
    pred_scores_bb = info.get('predicate_scores_backbone', {})
    pred_scores_corr = info.get('predicate_scores', {})

    names = list(pred_scores_bb.keys())[:6]
    scores_bb = [pred_scores_bb.get(n, 0) for n in names]
    scores_corr = [pred_scores_corr.get(n, 0) for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax14.bar(x - width/2, scores_bb, width, label='Backbone', color='orange')
    ax14.bar(x + width/2, scores_corr, width, label='Corrected', color='green')
    ax14.set_ylabel('Score')
    ax14.set_title('Predicate Scores')
    ax14.set_xticks(x)
    ax14.set_xticklabels([n.split('_')[0] for n in names], rotation=45, ha='right')
    ax14.legend(fontsize=8)
    ax14.set_ylim(0, 1)

    # Row 3 continued: Lambda activations
    ax15 = fig.add_subplot(4, 5, 15)
    lambda_stats = info.get('lambda_stats', {})
    if lambda_stats:
        names = list(lambda_stats.keys())
        means = [lambda_stats[n]['mean'] for n in names]
        ax15.bar(names, means, color='blue')
        ax15.set_ylabel('Mean λ')
        ax15.set_title('Corrector Activations')
        ax15.set_xticklabels(names, rotation=45, ha='right')

    # Row 4: Zoomed regions (center crop)
    h, w = clean.shape[-2:]
    ch, cw = h // 2, w // 2
    crop_size = min(h, w) // 4

    def crop_center(x):
        return x[..., ch-crop_size:ch+crop_size, cw-crop_size:cw+crop_size]

    ax16 = fig.add_subplot(4, 5, 16)
    ax16.imshow(to_np(crop_center(backbone_out)), cmap='gray')
    ax16.set_title('Backbone (zoomed)')
    ax16.axis('off')

    ax17 = fig.add_subplot(4, 5, 17)
    ax17.imshow(to_np(crop_center(corrected)), cmap='gray')
    ax17.set_title('Corrected (zoomed)')
    ax17.axis('off')

    ax18 = fig.add_subplot(4, 5, 18)
    ax18.imshow(to_np(crop_center(clean)), cmap='gray')
    ax18.set_title('GT (zoomed)')
    ax18.axis('off')

    # Summary text
    ax19 = fig.add_subplot(4, 5, 19)
    ax19.axis('off')
    summary = f"""
    PSNR Improvement: {psnr_corrected - psnr_backbone:+.3f} dB
    SSIM Improvement: {ssim_corrected - ssim_backbone:+.4f}

    Correction Magnitude: {info.get('correction_magnitude', 0):.4f}

    Predicate Avg (BB):   {np.mean(scores_bb):.3f}
    Predicate Avg (Corr): {np.mean(scores_corr):.3f}
    """
    ax19.text(0.1, 0.5, summary, fontsize=10, family='monospace',
              verticalalignment='center', transform=ax19.transAxes)
    ax19.set_title('Summary')

    # Verification status
    ax20 = fig.add_subplot(4, 5, 20)
    ax20.axis('off')
    verification = info.get('verification', {})
    verify_text = f"""
    Decision: {verification.get('decision', 'N/A')}
    Guarantees: {verification.get('guarantees_passed', 0)}/3

    Energy:    {'PASS' if verification.get('guarantees', {}).get('energy_descent', {}).get('passed', False) else 'FAIL'}
    Pareto:    {'PASS' if verification.get('guarantees', {}).get('pareto_efficient', {}).get('passed', False) else 'FAIL'}
    Lipschitz: {'PASS' if verification.get('guarantees', {}).get('lipschitz_bounded', {}).get('passed', False) else 'FAIL'}
    """
    ax20.text(0.1, 0.5, verify_text, fontsize=10, family='monospace',
              verticalalignment='center', transform=ax20.transAxes)
    ax20.set_title('Formal Verification')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_path}")
    return {
        'psnr_delta': psnr_corrected - psnr_backbone,
        'ssim_delta': ssim_corrected - ssim_backbone,
    }


def visualize_batch(model, dataloader, output_dir: str, num_samples: int = 5):
    """Visualize multiple samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        noisy = batch['noisy']
        clean = batch['clean']

        output_path = output_dir / f'sample_{i:03d}.png'
        metrics = visualize_single_sample(model, noisy, clean, str(output_path), i)
        results.append(metrics)

    # Summary
    avg_psnr_delta = np.mean([r['psnr_delta'] for r in results])
    avg_ssim_delta = np.mean([r['ssim_delta'] for r in results])

    print(f"\n=== Visualization Summary ===")
    print(f"Samples visualized: {len(results)}")
    print(f"Average PSNR delta: {avg_psnr_delta:+.3f} dB")
    print(f"Average SSIM delta: {avg_ssim_delta:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Visualize V8 Corrections')
    parser.add_argument('--checkpoint', default='outputs/nsnd_v8_enhanced/best_model_v8_enhanced.pth',
                       help='Model checkpoint path')
    parser.add_argument('--data_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl',
                       help='Data JSONL file')
    parser.add_argument('--output_dir', default='outputs/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--device', default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    # Load data
    from train_v8_enhanced import PKU37Dataset
    from torch.utils.data import DataLoader

    dataset = PKU37Dataset(args.data_jsonl, patch_size=None, max_samples=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Visualize
    visualize_batch(model, dataloader, args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()
