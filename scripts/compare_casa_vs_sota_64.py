#!/usr/bin/env python3
import json
from pathlib import Path
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--casa_metrics', type=str, default='checkpoints/casa_finetune64_phys/final_metrics.json')
    ap.add_argument('--sota_summary', type=str, default='outputs/unified_64/summary.json')
    ap.add_argument('--out', type=str, default='outputs/unified_64/comparison.txt')
    args = ap.parse_args()

    casa = {}
    if Path(args.casa_metrics).is_file():
        casa = json.load(open(args.casa_metrics, 'r'))
    else:
        print(f"Warning: CASA metrics not found: {args.casa_metrics}")

    sota = []
    if Path(args.sota_summary).is_file():
        sota = json.load(open(args.sota_summary, 'r'))
    else:
        print(f"Warning: SOTA summary not found: {args.sota_summary}")

    lines = []
    lines.append("=== 64x64 Comparison: CASA vs SOTAs ===\n")
    if casa:
        lines.append(f"CASA  PSNR: {casa.get('test_psnr', 'NA')}  SSIM: {casa.get('test_ssim','NA')}\n")
    lines.append("\nSOTA baselines:\n")
    for r in sota:
        lines.append(f"- {r['model'].upper():12s}  PSNR: {r['psnr']:.3f}  SSIM: {r['ssim']:.4f}\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.writelines(lines)
    print(''.join(lines))
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()

