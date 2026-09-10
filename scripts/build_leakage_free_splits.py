#!/usr/bin/env python3
"""
Build leakage-free train/val/test splits by hashing clean images.

Default behavior:
  - Collect clean images from train+val folders under data_root.
  - De-duplicate per-class by file hash.
  - Stratified sampling with per-class counts.
  - Optional symlink dataset roots for training/evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def hash_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_candidates(
    data_root: Path,
    classes: List[str],
    source_splits: List[str],
) -> Dict[str, List[Path]]:
    per_class: Dict[str, List[Path]] = {c: [] for c in classes}
    for cls in classes:
        for split in source_splits:
            clean_dir = data_root / cls / split / "clean"
            if not clean_dir.exists():
                continue
            per_class[cls].extend(sorted(clean_dir.glob("*.png")))
    return per_class


def dedup_by_hash(paths: List[Path]) -> Tuple[List[Path], int]:
    seen = set()
    uniq: List[Path] = []
    dupes = 0
    for p in paths:
        h = hash_file(p)
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        uniq.append(p)
    return uniq, dupes


def write_symlinks(
    split_paths: Dict[str, Dict[str, List[str]]],
    link_root: Path,
    test_as_val: bool = False,
) -> None:
    for split, per_class in split_paths.items():
        split_dir = "val" if (test_as_val and split == "test") else split
        for cls, items in per_class.items():
            out_dir = link_root / cls / split_dir / "clean"
            out_dir.mkdir(parents=True, exist_ok=True)
            for p_str in items:
                src = Path(p_str)
                dst = out_dir / src.name
                if dst.exists():
                    continue
                os.symlink(src, dst)


def overlap_stats(split_paths: Dict[str, Dict[str, List[str]]]) -> Dict[str, int]:
    def hashes_for(split: str) -> set[str]:
        hs = set()
        for items in split_paths[split].values():
            for p in items:
                hs.add(hash_file(Path(p)))
        return hs

    stats = {}
    splits = list(split_paths.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            stats[f"{a}-{b}"] = len(hashes_for(a) & hashes_for(b))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/kumwilai/OCT/oct")
    parser.add_argument("--output_splits", type=str, default="/home/kumwilai/OCT/oct_splits_tmi.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=str, default="cnv,dme,drusen,normal")
    parser.add_argument("--source_splits", type=str, default="train,val")
    parser.add_argument("--train_per_class", type=int, default=250)
    parser.add_argument("--val_per_class", type=int, default=50)
    parser.add_argument("--test_per_class", type=int, default=100)
    parser.add_argument("--link_root", type=str, default="")
    parser.add_argument("--link_test_root", type=str, default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    source_splits = [s.strip() for s in args.source_splits.split(",") if s.strip()]

    rng = random.Random(args.seed)

    per_class = collect_candidates(data_root, classes, source_splits)
    split_paths: Dict[str, Dict[str, List[str]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    print("Deduplicating and sampling per class...")
    for cls in classes:
        uniq, dupes = dedup_by_hash(per_class[cls])
        rng.shuffle(uniq)
        need = args.train_per_class + args.val_per_class + args.test_per_class
        if len(uniq) < need:
            raise SystemExit(
                f"{cls}: only {len(uniq)} unique clean images, need {need}."
            )
        split_paths["train"][cls] = [str(p) for p in uniq[: args.train_per_class]]
        split_paths["val"][cls] = [
            str(p) for p in uniq[args.train_per_class : args.train_per_class + args.val_per_class]
        ]
        split_paths["test"][cls] = [
            str(p)
            for p in uniq[
                args.train_per_class + args.val_per_class : args.train_per_class + args.val_per_class + args.test_per_class
            ]
        ]
        print(f"{cls}: uniq={len(uniq)} dupes={dupes} | train={args.train_per_class} val={args.val_per_class} test={args.test_per_class}")

    out_path = Path(args.output_splits)
    out_path.write_text(json.dumps(split_paths, indent=2))
    print(f"✓ Wrote {out_path}")

    stats = overlap_stats(split_paths)
    for key, val in stats.items():
        print(f"hash overlap {key}: {val}")

    if args.link_root:
        link_root = Path(args.link_root)
        write_symlinks(split_paths, link_root)
        print(f"✓ Symlink dataset root: {link_root}")

    if args.link_test_root:
        link_test_root = Path(args.link_test_root)
        write_symlinks(split_paths, link_test_root, test_as_val=True)
        print(f"✓ Symlink test root: {link_test_root} (test mapped to val)")


if __name__ == "__main__":
    main()
