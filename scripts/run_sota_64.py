#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from sota import train_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=str, default='oct_val_pairs_24.txt')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--out_dir', type=str, default='outputs/sota_64')
    ap.add_argument('--epochs', type=int, default=8)
    args = ap.parse_args()

    assert args.size <= 64, 'Only <= 64 supported per instruction.'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # NAFNet wrapper currently unavailable (NAFBackbone missing in adaptive_oct_denoise), skip for now
    models = ['drunet', 'swinir', 'noise2void', 'speckle2speckle']
    all_res = []
    for m in models:
        print(f"\n=== Training/Evaluating {m} at {args.size}x{args.size} ===")
        r = train_and_eval(m, args.pairs, size=args.size, out_dir=args.out_dir, epochs=args.epochs)
        print(r)
        all_res.append(r)

    with open(Path(args.out_dir) / 'summary.json', 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"Saved summary to {Path(args.out_dir) / 'summary.json'}")


if __name__ == '__main__':
    main()
