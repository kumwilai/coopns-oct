#!/usr/bin/env python3
"""Scan all PKU37 test images and rank by correction visibility per backbone."""

import json
import numpy as np
import torch
import torch.nn.functional as F
import gc

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
)

BACKBONE_CONFIGS = {
    'nafnet': {
        'checkpoint': 'outputs/nafnet_qt69d_2ch/best_model_cooperative.pth',
        'pretrained_backbone': 'outputs/nafnet_pku37_w40/best_model.pth',
    },
    'dncnn': {
        'checkpoint': 'outputs/dncnn_fullres_qt69d/best_model_cooperative.pth',
        'pretrained_backbone': 'NukeModel/dncnn_7m/best.pth',
    },
    'kbnet': {
        'checkpoint': 'outputs/kbnet_qt69d/best_model_cooperative.pth',
        'pretrained_backbone': 'NukeModel/kbnet_7m/best.pth',
    },
    'swinir': {
        'checkpoint': 'outputs/swinir_qt69d/best_model_cooperative.pth',
        'pretrained_backbone': 'NukeModel/swinir_7m/best.pth',
    },
}


def load_model(checkpoint, backbone_path, backbone_name, device='cpu'):
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=backbone_name,
        pretrained_backbone=backbone_path,
        hidden_channels=64,
    )
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()
    return model


def compute_edge_improvement(bb, co, cl):
    """Compute edge preservation improvement of corrected over backbone."""
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    edge_bb = F.conv2d(bb, sobel_y, padding=1).abs()
    edge_co = F.conv2d(co, sobel_y, padding=1).abs()
    edge_cl = F.conv2d(cl, sobel_y, padding=1).abs()
    # How much closer is corrected to clean edges vs backbone
    edge_err_bb = (edge_bb - edge_cl).abs().mean().item()
    edge_err_co = (edge_co - edge_cl).abs().mean().item()
    return edge_err_bb - edge_err_co  # positive = corrected is better


def scan_backbone(backbone_name, config, dataset, device='cpu'):
    """Scan all images, return ranked list by correction visibility."""
    print(f"\nScanning {backbone_name}...")
    model = load_model(config['checkpoint'], config['pretrained_backbone'],
                      backbone_name, device)

    results = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        clean = sample['clean'].unsqueeze(0).to(device)
        noisy = sample['noisy'].unsqueeze(0).to(device)

        with torch.no_grad():
            bb_out, unc = model.backbone(noisy)
            corrected, _ = model.corrector(
                bb_out, noisy, None,
                nafnet_uncertainty=unc, return_details=False,
            )

        corr_mag = (corrected - bb_out).abs().mean().item()
        # Focus on tissue region (top 70%)
        H = clean.shape[-2]
        tissue_h = int(H * 0.7)
        corr_tissue = (corrected[:,:,:tissue_h,:] - bb_out[:,:,:tissue_h,:]).abs().mean().item()

        psnr_bb = compute_psnr(bb_out, clean)
        psnr_co = compute_psnr(corrected, clean)

        edge_imp = compute_edge_improvement(bb_out, corrected, clean)

        # Combined visibility score: correction magnitude in tissue + edge improvement
        visibility = corr_tissue * 100 + edge_imp * 50 + max(0, psnr_co - psnr_bb) * 0.1

        results.append({
            'idx': idx,
            'corr_mag': corr_mag,
            'corr_tissue': corr_tissue,
            'psnr_delta': psnr_co - psnr_bb,
            'edge_imp': edge_imp,
            'visibility': visibility,
        })

        del clean, noisy, bb_out, corrected, unc

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(dataset)}]")

    del model
    gc.collect()

    # Sort by visibility score
    results.sort(key=lambda x: x['visibility'], reverse=True)
    return results


if __name__ == '__main__':
    dataset = PKU37Dataset('pku37_oct_dataset/pku37_real_test.jsonl',
                          patch_size=0, is_train=False)
    print(f"Dataset: {len(dataset)} images")

    # Only scan nafnet and dncnn (the problematic ones), skip swinir (too slow)
    all_rankings = {}
    for bname in ['nafnet', 'dncnn', 'kbnet']:
        rankings = scan_backbone(bname, BACKBONE_CONFIGS[bname], dataset)
        all_rankings[bname] = rankings

        print(f"\n=== {bname.upper()} Top 10 most visible corrections ===")
        print(f"{'Idx':>5} {'CorrMag':>10} {'CorrTissue':>12} {'PSNR_delta':>11} {'EdgeImp':>10} {'Score':>8}")
        for r in rankings[:10]:
            print(f"{r['idx']:>5} {r['corr_mag']:>10.5f} {r['corr_tissue']:>12.5f} "
                  f"{r['psnr_delta']:>+11.3f} {r['edge_imp']:>10.5f} {r['visibility']:>8.3f}")

    # Find samples that are in top-20 for ALL backbones (good for cross-backbone comparison)
    print("\n\n=== BEST SHARED SAMPLES (top for all backbones) ===")
    top_n = 30
    sets = {}
    for bname in all_rankings:
        sets[bname] = set(r['idx'] for r in all_rankings[bname][:top_n])

    shared = sets['nafnet'] & sets['dncnn'] & sets['kbnet']
    print(f"Indices in top-{top_n} for all 3 backbones: {sorted(shared)}")

    # Rank shared by average visibility
    shared_ranked = []
    for idx in shared:
        avg_vis = np.mean([
            next(r['visibility'] for r in all_rankings[bname] if r['idx'] == idx)
            for bname in all_rankings
        ])
        avg_corr = np.mean([
            next(r['corr_tissue'] for r in all_rankings[bname] if r['idx'] == idx)
            for bname in all_rankings
        ])
        shared_ranked.append((idx, avg_vis, avg_corr))

    shared_ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"\nBest shared samples (sorted by avg visibility):")
    for idx, vis, corr in shared_ranked[:10]:
        print(f"  Index {idx}: avg_visibility={vis:.3f}, avg_tissue_corr={corr:.5f}")
