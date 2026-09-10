#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from sota.unified_trainer import run_unified_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oct_root', type=str, default='oct')
    ap.add_argument('--classes', type=str, default='normal', help="Comma-separated or 'all'")
    ap.add_argument('--noisy_folder', type=str, default='noisy', help='noisy|noisy_gaussian|noisy_moderate_gamma|noisy_heavy_gamma')
    ap.add_argument('--methods', type=str, default='drunet,nafnet,swinir,noise2void,speckle2speckle')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--pretrain_epochs', type=int, default=40)
    ap.add_argument('--finetune_epochs', type=int, default=20)
    ap.add_argument('--out_dir', type=str, default='outputs/unified_64')
    ap.add_argument('--log_domain', action='store_true', help='Train losses in log-domain for speckle')
    args = ap.parse_args()

    assert args.size <= 64, 'Only <=64 supported in this environment.'
    classes = [c.strip() for c in (['cnv','dme','drusen','normal'] if args.classes=='all' else args.classes.split(',')) if c.strip()]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    results = run_unified_training(methods, args.oct_root, classes, args.noisy_folder,
                                   size=args.size, out_dir=args.out_dir,
                                   pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
                                   log_domain=args.log_domain)
    print('\nSUMMARY:')
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
