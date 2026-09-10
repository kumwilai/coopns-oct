#!/usr/bin/env python3
"""
Train and evaluate SOTA deep learning baselines using pairs files.
Modified version of run_sota_64.py to support train/val split.
"""
import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sota.train_eval import train_and_eval, evaluate_pairs, build_model
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate SOTA models using pairs files"
    )
    parser.add_argument('--train_pairs', type=str, required=True,
                       help='Path to training pairs file (e.g., train_pairs_gaussian.txt)')
    parser.add_argument('--val_pairs', type=str, default=None,
                       help='Path to validation pairs file (e.g., val_pairs_gaussian.txt)')
    parser.add_argument('--size', type=int, default=64,
                       help='Image size (must be <= 64)')
    parser.add_argument('--out_dir', type=str, default='outputs/sota_64',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--models', nargs='+',
                       default=['drunet', 'swinir', 'noise2void', 'speckle2speckle'],
                       choices=['drunet', 'nafnet', 'swinir', 'noise2void', 'speckle2speckle'],
                       help='Models to train and evaluate')

    args = parser.parse_args()

    assert args.size <= 64, 'Only <= 64 supported per instruction.'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("SOTA Deep Learning Baselines Evaluation")
    print(f"{'='*80}")
    print(f"Training pairs: {args.train_pairs}")
    print(f"Validation pairs: {args.val_pairs or 'Same as training'}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"Epochs: {args.epochs}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {args.out_dir}")
    print(f"{'='*80}\n")

    # If no validation pairs specified, use training pairs for eval
    eval_pairs = args.val_pairs if args.val_pairs else args.train_pairs

    all_results = []
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Training/Evaluating: {model_name.upper()}")
        print(f"{'='*80}\n")

        try:
            # Train on training pairs (this also saves the model)
            train_result = train_and_eval(
                model_name=model_name,
                pairs_file=args.train_pairs,
                size=args.size,
                out_dir=args.out_dir,
                epochs=args.epochs
            )

            # Evaluate on validation pairs (or training if val not specified)
            eval_pairs_file = args.val_pairs if args.val_pairs else args.train_pairs
            print(f"\nEvaluating {model_name} on {'validation' if args.val_pairs else 'training'} set...")

            model = build_model(model_name)
            model.load_state_dict(torch.load(Path(args.out_dir) / f'{model_name}.pth'))
            eval_psnr, eval_ssim, eval_time = evaluate_pairs(model, eval_pairs_file, size=args.size)

            # Build result dict
            result = {
                'model': model_name,
                'train_psnr': train_result['psnr'],
                'train_ssim': train_result['ssim'],
                'psnr': eval_psnr,  # Main metric
                'ssim': eval_ssim,  # Main metric
                'time_ms': eval_time
            }

            if args.val_pairs:
                result['val_psnr'] = eval_psnr
                result['val_ssim'] = eval_ssim
                result['val_time_ms'] = eval_time

            print(f"Evaluation Results - PSNR: {eval_psnr:.2f} dB, SSIM: {eval_ssim:.4f}, Time: {eval_time:.1f} ms")

            # Save evaluation results
            with open(Path(args.out_dir) / f'{model_name}_eval_metrics.json', 'w') as f:
                json.dump(result, f, indent=2)

            all_results.append(result)

        except Exception as e:
            print(f"Error training/evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary_path = Path(args.out_dir) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in all_results:
        print(f"{r['model'].upper():<20} - Train PSNR: {r['psnr']:.2f} dB, SSIM: {r['ssim']:.4f}", end='')
        if 'val_psnr' in r:
            print(f" | Val PSNR: {r['val_psnr']:.2f} dB, SSIM: {r['val_ssim']:.4f}")
        else:
            print()

    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
