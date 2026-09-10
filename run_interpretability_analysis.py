#!/usr/bin/env python3
"""
Quick script to generate interpretability visualizations
Usage: python run_interpretability_analysis.py --checkpoint path/to/model.pth
"""

import torch
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '/home/kumwilai/OCT')

from interpret_hybrid_nsnd import SpatialNoiseAnalyzer, visualize_interpretability
from nsnd_oct.scripts.train_hybrid_nsnd_multitask import (
    MultiTaskHybridNSND, PairedOCTCropDataset
)
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--val_pairs', default='val_pairs_duke_analysis.txt')
    parser.add_argument('--weights_jsonl', default='weights_duke_analysis_val.jsonl')
    parser.add_argument('--output_dir', default='interpretability_results')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Load model from checkpoint metadata
    model_config = ckpt.get('model_config', {})

    print("Initializing model...")
    model = MultiTaskHybridNSND(
        hybrid_analyzer_ckpt=model_config.get('hybrid_analyzer_ckpt',
                                             'checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth'),
        device=args.device,
        use_symbolic_branch=True,
        symbolic_type='neuro',
        ns_use_neural_predicates=True,
        ns_use_neural_weights=True,
        use_log_domain_analyzer=True,
        use_base_nafnet=True,
        base_nafnet_width=model_config.get('base_nafnet_width', 64),
        base_nafnet_type=model_config.get('base_nafnet_type', 'full'),
        base_enc_blk_nums=model_config.get('base_enc_blk_nums', [2, 2, 2]),
        base_dec_blk_nums=model_config.get('base_dec_blk_nums', [2, 2, 2]),
        base_middle_blk_num=model_config.get('base_middle_blk_num', 2),
        shared_residual=True,
        shared_trunk_width=24,
        shared_adapter_channels=64,
        shared_adapter_hidden=48,
        use_joint_signal_expert=True,
        joint_expert_channels=64,
        residual_blend_init=0.05,
    )

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    model.to(args.device)

    print("Loading validation data...")
    val_dataset = PairedOCTCropDataset(
        args.val_pairs,
        crop_size=64,
        random_crop=False,
        return_weights=True,
        weights_jsonl=args.weights_jsonl,
        max_samples=args.num_samples,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"\nGenerating interpretability visualizations for {args.num_samples} samples...")
    analyzer = SpatialNoiseAnalyzer(model, device=args.device)

    all_stats = []

    for idx, batch in enumerate(val_loader):
        noisy, clean = batch[0], batch[1]

        print(f"[{idx+1}/{args.num_samples}] Processing sample {idx}...")

        # Generate spatial maps
        spatial_maps = analyzer.generate_patch_based_maps(noisy, patch_size=32, stride=16)

        # Detect anomalies
        anomalies = analyzer.detect_anomalies(spatial_maps, uncertainty_threshold=0.8)

        # Forward pass for denoised output
        with torch.no_grad():
            denoised, _, _ = model(noisy.to(args.device))

        # Visualize
        fig = visualize_interpretability(
            noisy,
            spatial_maps,
            anomalies,
            denoised=denoised.cpu(),
            save_path=output_dir / f'interpret_sample_{idx:03d}.png'
        )

        # Print summary
        print(f"  Quality Score: {anomalies['quality_score']:.2%}")
        print(f"  Banding: {anomalies['banding_percentage']:.1f}%")
        if anomalies['warnings']:
            for w in anomalies['warnings']:
                print(f"  {w}")
        print()

        all_stats.append({
            'sample_id': idx,
            'quality_score': anomalies['quality_score'],
            'banding_pct': anomalies['banding_percentage'],
            'uncertainty_mean': spatial_maps['uncertainty'].mean().item(),
            'warnings': anomalies['warnings']
        })

    # Save stats
    import json
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n✓ Complete! Results saved to {output_dir}/")
    print(f"  - {args.num_samples} visualization PNGs")
    print(f"  - statistics.json with all metrics")

if __name__ == '__main__':
    main()
