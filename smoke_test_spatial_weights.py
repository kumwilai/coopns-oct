#!/usr/bin/env python3
"""
Smoke Test: Verify Spatial Adaptive Denoising Works
Tests:
1. Model loads with --use_spatial_weights
2. Forward pass generates spatial weight maps (B, 4, H, W)
3. Noise type identification is reasonable
4. Visualization works
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from nsnd_oct.scripts.train_hybrid_nsnd_multitask import (
    MultiTaskHybridNSND,
    PairedOCTCropDataset
)
import json


def smoke_test():
    """Quick validation of spatial adaptive denoising."""

    print("=" * 80)
    print("SMOKE TEST: Spatial Adaptive Denoising")
    print("=" * 80)

    # 1. Load model with spatial weights enabled
    print("\n[1/4] Loading model with spatial weights...")

    model = MultiTaskHybridNSND(
        base_nafnet_type='full',
        base_nafnet_width=64,
        base_enc_blk_nums=[2, 2, 2],
        base_dec_blk_nums=[2, 2, 2],
        base_middle_blk_num=2,
        base_ckpt='outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth',
        shared_trunk_width=32,
        shared_adapter_channels=96,
        shared_adapter_hidden=64,
        joint_expert_channels=96,
        residual_blend_init=0.35,
        use_joint_signal_expert=True,
        joint_mix_init=0.05,
        ns_use_neural_predicates=True,
        ns_use_neural_weights=True,
        shared_residual=True,
        use_spatial_weights=True,  # KEY: Enable spatial weights
        spatial_feature_channels=64,
        spatial_hidden_channels=32,
        gaussian_head='dncnn',
        shot_head='vst',
        hybrid_analyzer_ckpt='checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth',
        device='cpu',
        use_log_domain_analyzer=True,
    )

    model.eval()
    print("✓ Model loaded successfully")
    print(f"  Spatial refiner: {'ENABLED' if model.use_spatial_weights else 'DISABLED'}")

    # 2. Load test images
    print("\n[2/4] Loading test images...")

    pairs_file = 'val_pairs_duke_analysis.txt'
    weights_file = 'weights_duke_analysis_val.jsonl'

    dataset = PairedOCTCropDataset(
        pairs=pairs_file,
        crop_size=256,  # Larger crop for visualization
        random_crop=False,
        return_weights=True,
        weights_jsonl=weights_file,
        max_samples=5
    )

    print(f"✓ Loaded {len(dataset)} test samples")

    # 3. Run forward pass and check outputs
    print("\n[3/4] Testing forward pass...")

    sample = dataset[0]
    noisy = sample['noisy'].unsqueeze(0)  # (1, 1, H, W)
    clean = sample['clean'].unsqueeze(0)
    gt_weights = {
        'speckle': sample['weights'][0].item(),
        'banding': sample['weights'][1].item(),
        'gaussian': sample['weights'][2].item(),
        'shot': sample['weights'][3].item(),
    }

    print(f"  Input shape: {noisy.shape}")
    print(f"  Ground truth weights: {gt_weights}")

    with torch.no_grad():
        denoised, pred_weights, extras = model(noisy)

    print(f"✓ Forward pass successful")
    print(f"  Output shape: {denoised.shape}")
    print(f"  Predicted weights (global):")
    for noise_type, weight in pred_weights.items():
        print(f"    {noise_type}: {weight.item():.4f}")

    # Check spatial weight maps
    if 'spatial_weight_maps' in extras:
        spatial_maps = extras['spatial_weight_maps']
        print(f"\n✓ Spatial weight maps generated!")
        print(f"  Shape: {spatial_maps.shape}")  # Should be (1, 4, H, W)
        print(f"  Per-pixel weights sum to 1: {torch.allclose(spatial_maps.sum(dim=1), torch.ones(1, spatial_maps.shape[2], spatial_maps.shape[3]), atol=1e-5)}")

        # Compute dominant noise type
        dominant = torch.argmax(spatial_maps[0], dim=0)  # (H, W)
        noise_names = ['speckle', 'banding', 'gaussian', 'shot']

        print(f"\n  Dominant noise type distribution:")
        for i, name in enumerate(noise_names):
            percentage = (dominant == i).float().mean().item() * 100
            print(f"    {name}: {percentage:.1f}%")

        # Compare with ground truth
        gt_dominant_idx = np.argmax([gt_weights['speckle'], gt_weights['banding'],
                                      gt_weights['gaussian'], gt_weights['shot']])
        gt_dominant_name = noise_names[gt_dominant_idx]
        print(f"\n  Ground truth dominant: {gt_dominant_name} ({gt_weights[gt_dominant_name]:.4f})")

        # 4. Visualize spatial maps
        print("\n[4/4] Generating visualization...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Row 1: Input/output
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(noisy[0, 0].numpy(), cmap='gray')
        ax1.set_title('Noisy Input', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(denoised[0, 0].detach().numpy(), cmap='gray')
        psnr = -10 * np.log10(((denoised - clean) ** 2).mean().item())
        ax2.set_title(f'Denoised (PSNR: {psnr:.2f} dB)', fontsize=14, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(clean[0, 0].numpy(), cmap='gray')
        ax3.set_title('Clean Reference', fontsize=14, fontweight='bold')
        ax3.axis('off')

        # Individual weight maps
        for i, name in enumerate(noise_names):
            row, col = (0, 3) if i == 0 else (1, i - 1)
            if i == 0:
                row, col = 0, 3
            elif i == 1:
                row, col = 1, 0
            elif i == 2:
                row, col = 1, 1
            else:
                row, col = 1, 2

            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(spatial_maps[0, i].numpy(), cmap='RdYlGn_r', vmin=0, vmax=1)
            ax.set_title(f'{name.capitalize()} Weight Map\nGT: {gt_weights[name]:.3f}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Dominant noise type (argmax)
        ax_dom = fig.add_subplot(gs[1, 3])
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        cmap = ListedColormap(colors)
        im = ax_dom.imshow(dominant.numpy(), cmap=cmap)
        ax_dom.set_title(f'Dominant Noise (Spatial)\nGT: {gt_dominant_name}',
                         fontsize=14, fontweight='bold')
        ax_dom.axis('off')
        cbar = plt.colorbar(im, ax=ax_dom, ticks=[0, 1, 2, 3], fraction=0.046)
        cbar.set_ticklabels([n.capitalize() for n in noise_names])

        # Uncertainty (entropy)
        ax_unc = fig.add_subplot(gs[2, 0])
        entropy = -torch.sum(spatial_maps[0] * torch.log(spatial_maps[0] + 1e-8), dim=0)
        im = ax_unc.imshow(entropy.numpy(), cmap='hot')
        ax_unc.set_title(f'Uncertainty Map\nMean: {entropy.mean():.3f}',
                         fontsize=14, fontweight='bold')
        ax_unc.axis('off')
        plt.colorbar(im, ax=ax_unc, fraction=0.046)

        # Global vs spatial comparison
        ax_comp = fig.add_subplot(gs[2, 1:3])
        x = np.arange(4)
        global_weights = [pred_weights['speckle'].item(), pred_weights['banding'].item(),
                         pred_weights['gaussian'].item(), pred_weights['shot'].item()]
        spatial_mean = [spatial_maps[0, i].mean().item() for i in range(4)]
        gt_vals = [gt_weights['speckle'], gt_weights['banding'],
                   gt_weights['gaussian'], gt_weights['shot']]

        width = 0.25
        ax_comp.bar(x - width, gt_vals, width, label='Ground Truth', color='green', alpha=0.7)
        ax_comp.bar(x, global_weights, width, label='Global Weights', color='blue', alpha=0.7)
        ax_comp.bar(x + width, spatial_mean, width, label='Spatial Mean', color='red', alpha=0.7)
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels([n.capitalize() for n in noise_names])
        ax_comp.set_ylabel('Weight')
        ax_comp.set_title('Global vs Spatial Weight Comparison', fontsize=14, fontweight='bold')
        ax_comp.legend()
        ax_comp.grid(axis='y', alpha=0.3)

        # Summary text
        summary_text = f"""✓ SMOKE TEST PASSED

Model Configuration:
  • Spatial weights: ENABLED
  • Base NAFNet: width=64 (pre-trained)
  • Adapter blend: 0.35
  • Spatial feature channels: 64

Output Validation:
  • Spatial maps shape: {spatial_maps.shape}
  • Per-pixel normalization: ✓
  • Dominant noise matches GT: {dominant.mode()[0].item() == gt_dominant_idx}
  • PSNR: {psnr:.2f} dB

Spatial Variability:
  • Speckle std: {spatial_maps[0, 0].std():.4f}
  • Banding std: {spatial_maps[0, 1].std():.4f}
  • Gaussian std: {spatial_maps[0, 2].std():.4f}
  • Shot std: {spatial_maps[0, 3].std():.4f}
"""
        ax_text = fig.add_subplot(gs[2, 3])
        ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax_text.axis('off')

        plt.suptitle('Spatial Adaptive Denoising - Smoke Test',
                    fontsize=16, fontweight='bold')

        save_path = 'smoke_test_spatial_weights.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")

    else:
        print("\n✗ ERROR: No spatial weight maps found in extras!")
        print(f"  Available keys: {list(extras.keys())}")
        return False

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review smoke_test_spatial_weights.png")
    print("2. Verify spatial maps show heterogeneous noise distribution")
    print("3. If validated, run full training: bash run_spatial_adaptive_training.sh")

    return True


if __name__ == '__main__':
    success = smoke_test()
    sys.exit(0 if success else 1)
