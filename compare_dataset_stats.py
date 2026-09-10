#!/usr/bin/env python3
"""Compare image statistics between PKU37 (test) and Duke17 datasets.

Analyzes domain gap by computing intensity distributions, regional stats,
and the 15th percentile threshold used by intensity_gate.
"""

import json
import numpy as np
from PIL import Image
import os

def load_jsonl(path, max_samples=None):
    """Load image pairs from JSONL manifest."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if 'clean_path' in entry and 'noisy_path' in entry:
                if os.path.exists(entry['clean_path']) and os.path.exists(entry['noisy_path']):
                    samples.append(entry)
    if max_samples:
        # Sample evenly from the dataset
        indices = np.linspace(0, len(samples)-1, max_samples, dtype=int)
        samples = [samples[i] for i in indices]
    return samples


def load_image(path):
    """Load image and normalize to [0, 1] float32 (same as PKU37Dataset)."""
    img = np.array(Image.open(path)).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def compute_stats(img, label=""):
    """Compute comprehensive statistics for a single image."""
    H, W = img.shape
    stats = {}

    # Global stats
    stats['shape'] = f"{H}x{W}"
    stats['mean'] = float(np.mean(img))
    stats['std'] = float(np.std(img))
    stats['min'] = float(np.min(img))
    stats['max'] = float(np.max(img))
    stats['median'] = float(np.median(img))

    # Bottom 25% region (background in OCT)
    bottom_25 = img[int(0.75 * H):, :]
    stats['bot25_mean'] = float(np.mean(bottom_25))
    stats['bot25_std'] = float(np.std(bottom_25))

    # Top 50% region (tissue in OCT)
    top_50 = img[:int(0.50 * H), :]
    stats['top50_mean'] = float(np.mean(top_50))
    stats['top50_std'] = float(np.std(top_50))

    # 15th percentile (intensity_gate threshold)
    stats['p15'] = float(np.percentile(img, 15))
    stats['p25'] = float(np.percentile(img, 25))
    stats['p50'] = float(np.percentile(img, 50))
    stats['p75'] = float(np.percentile(img, 75))
    stats['p85'] = float(np.percentile(img, 85))

    # Histogram (10 bins, 0-1 range)
    hist, bin_edges = np.histogram(img.flatten(), bins=10, range=(0.0, 1.0))
    stats['hist_counts'] = hist.tolist()
    stats['hist_pcts'] = (hist / hist.sum() * 100).tolist()

    # Fraction of pixels below p15 threshold (background fraction)
    p15_val = stats['p15']
    stats['frac_below_p15'] = float(np.mean(img < p15_val))

    # Fraction of pixels that are "dark" (< 0.1)
    stats['frac_below_01'] = float(np.mean(img < 0.1))
    stats['frac_below_02'] = float(np.mean(img < 0.2))

    return stats


def print_separator(char='-', width=100):
    print(char * width)


def main():
    pku37_jsonl = "/home/kumwilai/OCT/pku37_oct_dataset/pku37_real_test.jsonl"
    duke17_jsonl = "/home/kumwilai/OCT/duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl"

    n_samples = 5

    print("=" * 100)
    print("DATASET COMPARISON: PKU37 (test) vs Duke17")
    print("=" * 100)

    # Load samples
    pku37_samples = load_jsonl(pku37_jsonl, max_samples=n_samples)
    duke17_samples = load_jsonl(duke17_jsonl, max_samples=n_samples)

    print(f"\nPKU37 samples: {len(pku37_samples)}")
    print(f"Duke17 samples: {len(duke17_samples)}")

    # =========================================================================
    # Analyze both noisy and clean images
    # =========================================================================
    for img_type in ['noisy', 'clean']:
        key = 'noisy_path' if img_type == 'noisy' else 'clean_path'

        print(f"\n{'=' * 100}")
        print(f"  IMAGE TYPE: {img_type.upper()}")
        print(f"{'=' * 100}")

        for dataset_name, samples in [("PKU37", pku37_samples), ("Duke17", duke17_samples)]:
            print(f"\n--- {dataset_name} ({img_type}) ---")
            all_stats = []
            all_pixels = []

            for i, sample in enumerate(samples):
                img = load_image(sample[key])
                stats = compute_stats(img)
                all_stats.append(stats)
                all_pixels.append(img.flatten())

                print(f"\n  Sample {i+1}: {os.path.basename(sample[key])}")
                print(f"    Shape: {stats['shape']}, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Median: {stats['median']:.4f}")
                print(f"    Bottom 25% (BG):  mean={stats['bot25_mean']:.4f}, std={stats['bot25_std']:.4f}")
                print(f"    Top 50% (tissue): mean={stats['top50_mean']:.4f}, std={stats['top50_std']:.4f}")
                print(f"    Percentiles: p15={stats['p15']:.4f}, p25={stats['p25']:.4f}, "
                      f"p50={stats['p50']:.4f}, p75={stats['p75']:.4f}, p85={stats['p85']:.4f}")
                print(f"    Frac < 0.1: {stats['frac_below_01']:.3f}, "
                      f"Frac < 0.2: {stats['frac_below_02']:.3f}")

                # Print histogram
                bins = [f"{b:.1f}-{b+0.1:.1f}" for b in np.arange(0, 1.0, 0.1)]
                print(f"    Histogram (10 bins, 0-1):")
                for j, (b, c, p) in enumerate(zip(bins, stats['hist_counts'], stats['hist_pcts'])):
                    bar = '#' * int(p / 2)
                    print(f"      [{b}]: {c:8d} ({p:5.1f}%) {bar}")

            # Aggregate stats
            print(f"\n  === {dataset_name} AGGREGATE ({img_type}) ===")
            for field in ['mean', 'std', 'min', 'max', 'median',
                          'bot25_mean', 'bot25_std', 'top50_mean', 'top50_std',
                          'p15', 'p25', 'p50', 'p75', 'p85',
                          'frac_below_01', 'frac_below_02']:
                vals = [s[field] for s in all_stats]
                print(f"    {field:15s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

            # Aggregate histogram
            all_pix = np.concatenate(all_pixels)
            hist, _ = np.histogram(all_pix, bins=10, range=(0.0, 1.0))
            pcts = hist / hist.sum() * 100
            print(f"\n    Aggregate histogram (all {len(all_pixels)} images pooled):")
            bins = [f"{b:.1f}-{b+0.1:.1f}" for b in np.arange(0, 1.0, 0.1)]
            for b, c, p in zip(bins, hist, pcts):
                bar = '#' * int(p / 2)
                print(f"      [{b}]: {c:10d} ({p:5.1f}%) {bar}")

    # =========================================================================
    # Direct comparison table
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("  DIRECT COMPARISON TABLE (NOISY IMAGES)")
    print(f"{'=' * 100}")

    # Re-compute for all noisy images
    pku37_all_stats = []
    duke17_all_stats = []

    for sample in pku37_samples:
        img = load_image(sample['noisy_path'])
        pku37_all_stats.append(compute_stats(img))

    for sample in duke17_samples:
        img = load_image(sample['noisy_path'])
        duke17_all_stats.append(compute_stats(img))

    print(f"\n{'Metric':20s} | {'PKU37 (mean +/- std)':25s} | {'Duke17 (mean +/- std)':25s} | {'Diff':10s}")
    print("-" * 90)
    for field in ['mean', 'std', 'min', 'max', 'median',
                  'bot25_mean', 'bot25_std', 'top50_mean', 'top50_std',
                  'p15', 'p25', 'p50', 'p75', 'p85',
                  'frac_below_01', 'frac_below_02']:
        pku_vals = [s[field] for s in pku37_all_stats]
        duke_vals = [s[field] for s in duke17_all_stats]
        pku_m, pku_s = np.mean(pku_vals), np.std(pku_vals)
        duke_m, duke_s = np.mean(duke_vals), np.std(duke_vals)
        diff = duke_m - pku_m
        print(f"{field:20s} | {pku_m:8.4f} +/- {pku_s:.4f}   | {duke_m:8.4f} +/- {duke_s:.4f}   | {diff:+.4f}")

    # =========================================================================
    # Intensity gate analysis
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("  INTENSITY GATE ANALYSIS")
    print("  (intensity_gate = sigmoid((pixel - p15_threshold) * 20))")
    print(f"{'=' * 100}")

    for dataset_name, samples in [("PKU37", pku37_samples), ("Duke17", duke17_samples)]:
        print(f"\n--- {dataset_name} ---")
        for i, sample in enumerate(samples):
            img = load_image(sample['noisy_path'])
            H, W = img.shape
            p15 = np.percentile(img, 15)

            # Simulate intensity_gate
            gate = 1.0 / (1.0 + np.exp(-(img - p15) * 20.0))

            # Stats about the gate
            print(f"  Sample {i+1}: p15={p15:.4f}")
            print(f"    Gate mean: {np.mean(gate):.4f}, Gate > 0.5: {np.mean(gate > 0.5):.3f}")

            # Gate in different regions
            top50_gate = gate[:int(0.50 * H), :]
            bot25_gate = gate[int(0.75 * H):, :]
            mid_gate = gate[int(0.25 * H):int(0.75 * H), :]

            print(f"    Top 50% (tissue): gate mean={np.mean(top50_gate):.4f}, "
                  f"gate > 0.5: {np.mean(top50_gate > 0.5):.3f}")
            print(f"    Mid 25-75%:       gate mean={np.mean(mid_gate):.4f}, "
                  f"gate > 0.5: {np.mean(mid_gate > 0.5):.3f}")
            print(f"    Bottom 25% (BG):  gate mean={np.mean(bot25_gate):.4f}, "
                  f"gate > 0.5: {np.mean(bot25_gate > 0.5):.3f}")

    # =========================================================================
    # Background smoothing analysis
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("  BACKGROUND SMOOTHING IMPACT ANALYSIS")
    print("  (bg_mask = 1 - intensity_gate; smoothing applied where bg_mask ~ 1)")
    print(f"{'=' * 100}")

    for dataset_name, samples in [("PKU37", pku37_samples), ("Duke17", duke17_samples)]:
        print(f"\n--- {dataset_name} ---")
        bg_fracs = []
        for i, sample in enumerate(samples):
            img = load_image(sample['noisy_path'])
            H, W = img.shape
            p15 = np.percentile(img, 15)
            gate = 1.0 / (1.0 + np.exp(-(img - p15) * 20.0))
            bg_mask = 1.0 - gate  # 1 in background, 0 in tissue

            # Where will smoothing be applied?
            strong_bg = np.mean(bg_mask > 0.5)  # strong background pixels
            tissue_exposed = np.mean(bg_mask > 0.5)  # same thing
            bg_fracs.append(strong_bg)

            # Check if smoothing bleeds into tissue region
            top50_bg = bg_mask[:int(0.50 * H), :]
            bot25_bg = bg_mask[int(0.75 * H):, :]

            print(f"  Sample {i+1}: bg_mask > 0.5 fraction: {strong_bg:.3f}")
            print(f"    Top 50%: bg_mask > 0.5 = {np.mean(top50_bg > 0.5):.3f} "
                  f"(tissue region, should be LOW)")
            print(f"    Bot 25%: bg_mask > 0.5 = {np.mean(bot25_bg > 0.5):.3f} "
                  f"(background, should be HIGH)")

        print(f"  Average bg fraction (bg_mask > 0.5): {np.mean(bg_fracs):.3f}")

    # =========================================================================
    # Key differences summary
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("  SUMMARY: KEY DOMAIN GAP FACTORS")
    print(f"{'=' * 100}")

    pku_means = [s['mean'] for s in pku37_all_stats]
    duke_means = [s['mean'] for s in duke17_all_stats]
    pku_stds = [s['std'] for s in pku37_all_stats]
    duke_stds = [s['std'] for s in duke17_all_stats]
    pku_p15s = [s['p15'] for s in pku37_all_stats]
    duke_p15s = [s['p15'] for s in duke17_all_stats]
    pku_bot = [s['bot25_mean'] for s in pku37_all_stats]
    duke_bot = [s['bot25_mean'] for s in duke17_all_stats]
    pku_top = [s['top50_mean'] for s in pku37_all_stats]
    duke_top = [s['top50_mean'] for s in duke17_all_stats]

    print(f"\n  1. Overall intensity: PKU37 mean={np.mean(pku_means):.4f}, "
          f"Duke17 mean={np.mean(duke_means):.4f} "
          f"({'Duke brighter' if np.mean(duke_means) > np.mean(pku_means) else 'PKU brighter'})")

    print(f"  2. Noise level (std): PKU37={np.mean(pku_stds):.4f}, "
          f"Duke17={np.mean(duke_stds):.4f} "
          f"({'Duke noisier' if np.mean(duke_stds) > np.mean(pku_stds) else 'PKU noisier'})")

    print(f"  3. P15 threshold:     PKU37={np.mean(pku_p15s):.4f}, "
          f"Duke17={np.mean(duke_p15s):.4f}")

    contrast_pku = np.mean(pku_top) - np.mean(pku_bot)
    contrast_duke = np.mean(duke_top) - np.mean(duke_bot)
    print(f"  4. Tissue-BG contrast: PKU37={contrast_pku:.4f}, "
          f"Duke17={contrast_duke:.4f}")

    print(f"  5. Background (bot25): PKU37={np.mean(pku_bot):.4f}, "
          f"Duke17={np.mean(duke_bot):.4f}")

    print(f"  6. Tissue (top50):     PKU37={np.mean(pku_top):.4f}, "
          f"Duke17={np.mean(duke_top):.4f}")

    # Dynamic range
    pku_dynrange = [s['max'] - s['min'] for s in pku37_all_stats]
    duke_dynrange = [s['max'] - s['min'] for s in duke17_all_stats]
    print(f"  7. Dynamic range:      PKU37={np.mean(pku_dynrange):.4f}, "
          f"Duke17={np.mean(duke_dynrange):.4f}")

    print(f"\n  IMPLICATIONS FOR INTENSITY GATE:")
    print(f"    - If Duke images have different intensity distribution shape,")
    print(f"      the per-image p15 threshold adapts automatically.")
    print(f"    - BUT: if tissue/background overlap more in Duke,")
    print(f"      the gate may incorrectly suppress tissue corrections")
    print(f"      or fail to suppress background noise.")
    print(f"    - Check: Does bottom 25% in Duke have higher intensity?")
    print(f"      If so, the gate may not suppress it properly.")
    print(f"    - Check: Does top 50% in Duke have lower intensity?")
    print(f"      If so, the gate may suppress valid tissue corrections.")


if __name__ == '__main__':
    main()
