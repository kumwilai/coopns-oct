#!/usr/bin/env python3
"""
Comprehensive comparison script for CASA vs Baseline Filters
Extracts metrics from CASA checkpoint and compares with baseline filter results
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_casa_metrics(checkpoint_dir):
    """Load CASA model metrics from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for final_metrics.json or similar
    metrics_file = checkpoint_dir / 'final_metrics.json'
    if not metrics_file.exists():
        # Try to find any metrics file
        metrics_files = list(checkpoint_dir.glob('*metrics*.json'))
        if metrics_files:
            metrics_file = metrics_files[0]
        else:
            print(f"Warning: No metrics file found in {checkpoint_dir}")
            return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return metrics


def load_baseline_results(results_csv):
    """Load baseline filter results from CSV."""
    if not os.path.exists(results_csv):
        print(f"Warning: Baseline results not found at {results_csv}")
        return None

    df = pd.read_csv(results_csv)

    # Group by method and compute statistics
    summary = df.groupby('method').agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'ssim': ['mean', 'std', 'min', 'max'],
        'psnr_gain': ['mean', 'std'],
        'ssim_gain': ['mean', 'std'],
        'time_ms': ['mean', 'std']
    })

    return summary


def print_comparison_table(casa_metrics, baseline_summary):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: CASA Adaptive Denoiser vs Classical Baseline Filters")
    print("="*100)
    print()

    # Print baseline results
    if baseline_summary is not None:
        print("Classical Baseline Filters (Mean ± Std):")
        print("-"*100)
        print(f"{'Method':<15} {'PSNR (dB)':<20} {'SSIM':<20} {'Time (ms)':<20}")
        print("-"*100)

        for method in baseline_summary.index:
            psnr_mean = baseline_summary.loc[method, ('psnr', 'mean')]
            psnr_std = baseline_summary.loc[method, ('psnr', 'std')]
            ssim_mean = baseline_summary.loc[method, ('ssim', 'mean')]
            ssim_std = baseline_summary.loc[method, ('ssim', 'std')]
            time_mean = baseline_summary.loc[method, ('time_ms', 'mean')]
            time_std = baseline_summary.loc[method, ('time_ms', 'std')]

            print(f"{method.upper():<15} {psnr_mean:>7.2f} ± {psnr_std:<5.2f}    "
                  f"{ssim_mean:>6.4f} ± {ssim_std:<6.4f}    "
                  f"{time_mean:>7.1f} ± {time_std:<6.1f}")

        print()

    # Print CASA results
    if casa_metrics is not None:
        print("CASA Adaptive Denoiser:")
        print("-"*100)

        # Extract relevant metrics
        if 'test_psnr' in casa_metrics:
            print(f"  PSNR: {casa_metrics['test_psnr']:.2f} dB")
        if 'test_ssim' in casa_metrics:
            print(f"  SSIM: {casa_metrics['test_ssim']:.4f}")

        # Print any available eval metrics
        for key in casa_metrics:
            if 'eval' in key.lower():
                print(f"  {key}: {casa_metrics[key]}")

        print()

    # Comparison summary
    if baseline_summary is not None and casa_metrics is not None:
        print("="*100)
        print("SUMMARY:")
        print("="*100)

        # Find best baseline
        best_baseline_psnr = baseline_summary[('psnr', 'mean')].max()
        best_baseline_method = baseline_summary[('psnr', 'mean')].idxmax()
        best_baseline_ssim = baseline_summary.loc[best_baseline_method, ('ssim', 'mean')]

        print(f"\nBest Classical Filter: {best_baseline_method.upper()}")
        print(f"  PSNR: {best_baseline_psnr:.2f} dB")
        print(f"  SSIM: {best_baseline_ssim:.4f}")

        if 'test_psnr' in casa_metrics and 'test_ssim' in casa_metrics:
            casa_psnr = casa_metrics['test_psnr']
            casa_ssim = casa_metrics['test_ssim']

            print(f"\nCASA Adaptive Denoiser:")
            print(f"  PSNR: {casa_psnr:.2f} dB ({casa_psnr - best_baseline_psnr:+.2f} dB vs best baseline)")
            print(f"  SSIM: {casa_ssim:.4f} ({casa_ssim - best_baseline_ssim:+.4f} vs best baseline)")

            if casa_psnr > best_baseline_psnr:
                print(f"\n✓ CASA outperforms best baseline by {casa_psnr - best_baseline_psnr:.2f} dB in PSNR")
            else:
                print(f"\n✗ CASA underperforms best baseline by {best_baseline_psnr - casa_psnr:.2f} dB in PSNR")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Compare CASA vs baseline filter results")
    parser.add_argument('--casa_dir', type=str, default='./checkpoints/casa_comparison',
                       help='Path to CASA checkpoint directory')
    parser.add_argument('--baseline_csv', type=str, default='./outputs/comparison/baselines/results.csv',
                       help='Path to baseline results CSV')
    parser.add_argument('--output', type=str, default='./outputs/comparison/final_comparison.txt',
                       help='Output file for comparison report')

    args = parser.parse_args()

    # Load results
    print("Loading results...")
    casa_metrics = load_casa_metrics(args.casa_dir)
    baseline_summary = load_baseline_results(args.baseline_csv)

    # Print comparison
    print_comparison_table(casa_metrics, baseline_summary)

    # Save to file
    if args.output:
        import sys
        from io import StringIO

        # Redirect stdout to capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        print_comparison_table(casa_metrics, baseline_summary)

        # Get the captured content
        output_content = captured_output.getvalue()
        sys.stdout = old_stdout

        # Write to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(output_content)

        print(f"\nComparison report saved to: {output_path}")


if __name__ == '__main__':
    main()
