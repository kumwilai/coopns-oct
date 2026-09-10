#!/usr/bin/env python3
"""
Verify split sizes and check for leakage by path/hash overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def hash_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten(split_dict: Dict[str, List[str]]) -> List[Path]:
    paths: List[Path] = []
    for items in split_dict.values():
        paths.extend(Path(p) for p in items)
    return paths


def overlap_counts(paths_a: List[Path], paths_b: List[Path]) -> int:
    return len({p.as_posix() for p in paths_a} & {p.as_posix() for p in paths_b})


def hash_overlap(paths_a: List[Path], paths_b: List[Path]) -> Tuple[int, List[Tuple[Path, Path]]]:
    map_a: Dict[str, Path] = {}
    for p in paths_a:
        map_a[hash_file(p)] = p
    overlap = []
    for p in paths_b:
        h = hash_file(p)
        if h in map_a:
            overlap.append((map_a[h], p))
    return len(overlap), overlap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_json", type=str, default="/home/kumwilai/OCT/oct_splits_tmi.json")
    parser.add_argument("--skip_hash", action="store_true")
    parser.add_argument("--show_examples", type=int, default=5)
    args = parser.parse_args()

    splits_path = Path(args.splits_json)
    splits = json.loads(splits_path.read_text())

    print(f"Splits: {splits_path}")
    totals = {}
    missing = 0
    for split, per_cls in splits.items():
        total = 0
        print(f"\n[{split}]")
        for cls, items in per_cls.items():
            total += len(items)
            print(f"  {cls}: {len(items)}")
            for p in items:
                if not Path(p).exists():
                    missing += 1
        totals[split] = total
        print(f"  total: {total}")

    if missing:
        print(f"\n⚠ Missing files: {missing}")
    else:
        print("\n✓ All split paths exist on disk")

    # Path overlap
    splits_flat = {k: flatten(v) for k, v in splits.items()}
    keys = list(splits_flat.keys())
    print("\nPath overlap:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            print(f"  {a}-{b}: {overlap_counts(splits_flat[a], splits_flat[b])}")

    if args.skip_hash:
        print("\n(skip hash overlap)")
        return

    print("\nHash overlap:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            count, examples = hash_overlap(splits_flat[a], splits_flat[b])
            print(f"  {a}-{b}: {count}")
            if examples and args.show_examples > 0:
                for left, right in examples[: args.show_examples]:
                    print(f"    - {left} == {right}")


if __name__ == "__main__":
    main()
