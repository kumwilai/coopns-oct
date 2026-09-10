#!/usr/bin/env python3
"""
Sequential TTA hyperparameter sweep for cross-dataset validation.

Loads model ONCE, then sweeps TTA configs one at a time to avoid OOM.
Each config: adapt corrector -> evaluate -> restore weights -> next config.
"""
import argparse
import ctypes
import gc
import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader


def _release_memory():
    """Force Python GC and return freed pages to OS via glibc malloc_trim.

    On Linux, glibc's allocator doesn't always return freed memory to the OS,
    causing RSS to grow over many alloc/free cycles (memory fragmentation).
    malloc_trim(0) forces the allocator to release free pages back to the OS.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
    compute_ssim,
)
from validate_crossdataset import (
    TestTimeAdaptation,
    validate_dataset,
    validate_dataset_fast,
)


def _make_loader(dataset, device):
    """Create a DataLoader with memory-safe settings.

    Uses num_workers=0 (main process only) to avoid worker process memory
    overhead on memory-constrained systems (7.8GB RAM + 4GB swap, no GPU).
    """
    use_cuda = device != 'cpu' and torch.cuda.is_available()
    return DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=use_cuda,
    )


def run_single_tta_config(model, cross_datasets, device, config, config_name,
                          backbone_caches=None, fast=False):
    """Run TTA adaptation + validation for CROSS-DATASET datasets only.

    Same-distribution datasets (PKU37) are skipped because TTA only adapts
    for cross-dataset data — their results are identical to baseline.

    Args:
        backbone_caches: Optional dict mapping ds_name -> pre-computed backbone cache.
                        When provided, skips redundant backbone forward passes.
        fast: Use fast PSNR/SSIM-only validation (~5x faster). Use during
              sweep screening; run full validation only on best config(s).
    """
    print(f"\n{'#'*84}")
    print(f"  CONFIG: {config_name}{'  [FAST]' if fast else ''}")
    print(f"  tta_steps={config['tta_steps']}, tta_lr={config['tta_lr']}, "
          f"w_mag={config['w_magnitude']}, w_cnr={config['w_cnr']}, "
          f"w_cons={config['w_consistency']}")
    print(f"{'#'*84}")

    results = []
    validate_fn = validate_dataset_fast if fast else validate_dataset

    for ds_name, dataset in cross_datasets:
        if len(dataset) == 0:
            print(f"  WARNING: {ds_name} has 0 samples, skipping")
            continue

        loader = _make_loader(dataset, device)
        tta_adapter = None
        cache = backbone_caches.get(ds_name) if backbone_caches else None

        try:
            tta_adapter = TestTimeAdaptation(
                model=model, device=device,
                tta_steps=config['tta_steps'],
                tta_lr=config['tta_lr'],
                n_adapt_samples=config.get('n_adapt_samples', 5),
                w_magnitude=config['w_magnitude'],
                w_cnr=config['w_cnr'],
                w_consistency=config['w_consistency'],
            )
            tta_adapter.adapt(loader, dataset_name=ds_name, backbone_cache=cache)
            del loader  # Free loader before creating new one
            _release_memory()  # Reclaim TTA computation graph memory before validation
            # Re-create loader (iterator consumed by adapt)
            loader = _make_loader(dataset, device)

            # Skip potential map storage during fast validation (saves ~5MB/sample)
            if fast:
                model.corrector._skip_maps = True
            result = validate_fn(model, loader, device, dataset_name=ds_name)
            model.corrector._skip_maps = False
            del loader
            if result:
                result['config_name'] = config_name
                result['config'] = config
                result['tta_adapted'] = True
                results.append(result)
        finally:
            if tta_adapter is not None:
                tta_adapter.restore_state()
                tta_adapter = None
            _release_memory()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sequential TTA hyperparameter sweep (OOM-safe, optimized)")
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model_cooperative.pth')
    parser.add_argument('--backbone', required=True,
                        help='Path to NAFNet backbone checkpoint')
    parser.add_argument('--backbone_name', type=str, default='nafnet',
                        choices=['nafnet', 'dncnn', 'swinir', 'kbnet', 'mambair'],
                        help='SOTA backbone architecture')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset filter (e.g. "duke17,duke2013")')
    parser.add_argument('--output_json', default='outputs/tta_sweep_results.json',
                        help='Save sweep results to JSON')
    args = parser.parse_args()

    device = args.device

    # ── Load model ONCE ──────────────────────────────────────────────────
    print("Loading model (one-time)...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.backbone,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded.\n")

    # ── Discover and pre-load datasets ONCE ──────────────────────────────
    dataset_candidates = [
        ('PKU37-Test (same dist)', 'pku37_oct_dataset/pku37_real_test.jsonl'),
        ('PKU37-Val (training val)', 'pku37_oct_dataset/pku37_real_val.jsonl'),
        ('Duke17-Sparsity (cross-dataset)', 'duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl'),
        ('Duke2013-SBSDI (cross-dataset)', 'duke_sota_datasets/Duke17_Eval/duke2013_synth_eval.jsonl'),
        ('Duke-Combined (cross-dataset)', 'duke_sota_datasets/Duke17_Eval/combined_eval.jsonl'),
    ]
    all_datasets = [(n, p) for n, p in dataset_candidates if os.path.exists(p)]

    if args.datasets:
        filters = [f.strip().lower() for f in args.datasets.split(',')]
        all_datasets = [(n, p) for n, p in all_datasets
                        if any(f in n.lower() for f in filters)]

    if not all_datasets:
        print("ERROR: No datasets found!")
        sys.exit(1)

    # Pre-load dataset objects once (avoids re-parsing JSONL for every config)
    loaded_datasets = []
    for name, path in all_datasets:
        ds = PKU37Dataset(path, patch_size=0, is_train=False)
        loaded_datasets.append((name, ds))
        print(f"  Loaded {name}: {len(ds)} samples")

    # Split into same-distribution and cross-dataset
    same_dist_datasets = [(n, ds) for n, ds in loaded_datasets if 'cross-dataset' not in n.lower()]
    cross_datasets = [(n, ds) for n, ds in loaded_datasets if 'cross-dataset' in n.lower()]

    print(f"\nSame-distribution: {len(same_dist_datasets)} datasets")
    print(f"Cross-dataset (TTA targets): {len(cross_datasets)} datasets")

    # ── Define hyperparameter grid ───────────────────────────────────────
    defaults = dict(tta_steps=30, tta_lr=5e-4, w_magnitude=0.5,
                    w_cnr=1.5, w_consistency=1.0, n_adapt_samples=5)

    configs = {}

    for steps in [10, 20, 30, 50]:
        configs[f"steps={steps}"] = {**defaults, 'tta_steps': steps}
    for lr in [1e-4, 3e-4, 5e-4, 1e-3]:
        configs[f"lr={lr}"] = {**defaults, 'tta_lr': lr}
    for w_mag in [0.1, 0.3, 0.5, 1.0]:
        configs[f"w_mag={w_mag}"] = {**defaults, 'w_magnitude': w_mag}
    for w_cnr in [0.5, 1.0, 1.5, 2.5]:
        configs[f"w_cnr={w_cnr}"] = {**defaults, 'w_cnr': w_cnr}
    for w_cons in [0.0, 0.5, 1.0, 2.0]:
        configs[f"w_cons={w_cons}"] = {**defaults, 'w_consistency': w_cons}

    # De-duplicate
    unique_configs = {}
    for name, cfg in configs.items():
        key = tuple(sorted(cfg.items()))
        if key not in unique_configs:
            unique_configs[key] = (name, cfg)
    configs = {name: cfg for _, (name, cfg) in unique_configs.items()}

    print(f"\nTotal unique configs to sweep: {len(configs)}")
    for name in sorted(configs.keys()):
        print(f"  - {name}")

    # ── Run baseline (no TTA) on ALL datasets ────────────────────────────
    print("\n" + "="*84)
    print("  BASELINE (no TTA)")
    print("="*84)
    baseline_results = []
    for ds_name, dataset in loaded_datasets:
        if len(dataset) == 0:
            continue
        loader = _make_loader(dataset, device)
        result = validate_dataset(model, loader, device, dataset_name=ds_name)
        del loader
        if result:
            result['config_name'] = 'baseline_no_tta'
            result['config'] = None
            baseline_results.append(result)

    _release_memory()  # Free baseline validation intermediates before caching

    # ── Pre-compute backbone cache for cross-datasets (one-time cost) ────
    # Backbone is frozen (7.02M params), so its outputs are identical across
    # all TTA configs. Computing once saves ~80% of adaptation time.
    print("\n" + "="*84)
    print("  PRE-COMPUTING BACKBONE CACHE (one-time)")
    print("="*84)
    backbone_caches = {}
    n_adapt = defaults.get('n_adapt_samples', 5)
    # Any config with w_consistency > 0 needs flipped outputs
    any_consistency = any(cfg.get('w_consistency', 0) > 0 for cfg in configs.values())
    for ds_name, dataset in cross_datasets:
        if len(dataset) == 0:
            continue
        loader = _make_loader(dataset, device)
        cache = TestTimeAdaptation.precompute_backbone_cache(
            model, loader, device,
            n_samples=n_adapt,
            include_flips=any_consistency,
        )
        backbone_caches[ds_name] = cache
        del loader
        print(f"  Cached {ds_name}: {len(cache)} samples"
              f"{' (+ flips)' if any_consistency else ''}")

    # ── Sequential sweep with FAST screening ────────────────────────────
    # Phase 1: Fast PSNR/SSIM-only screening (~5x faster per config)
    # Phase 2: Full clinical validation on top-3 configs
    fast_sweep_results = baseline_results.copy()
    total_configs = len(configs)

    print(f"\n{'='*84}")
    print(f"  PHASE 1: FAST SCREENING ({total_configs} configs)")
    print(f"{'='*84}")

    for idx, (config_name, config) in enumerate(sorted(configs.items()), 1):
        print(f"\n>>> Sweep progress: {idx}/{total_configs}")
        t0 = time.time()

        results = run_single_tta_config(
            model, cross_datasets, device, config, config_name,
            backbone_caches=backbone_caches,
            fast=True,
        )
        fast_sweep_results.extend(results)

        elapsed = time.time() - t0
        print(f">>> Config '{config_name}' done in {elapsed:.1f}s")

        _release_memory()

    # ── Fast screening summary ────────────────────────────────────────
    print(f"\n{'='*120}")
    print("  FAST SCREENING RESULTS")
    print(f"{'='*120}")

    from collections import defaultdict
    by_config_fast = defaultdict(list)
    for r in fast_sweep_results:
        by_config_fast[r.get('config_name', 'unknown')].append(r)

    print(f"\n{'Config':<25} {'Dataset':<35} {'PSNR-delta':>10} {'SSIM-delta':>10}")
    print("-"*85)

    for config_name in sorted(by_config_fast.keys()):
        for r in by_config_fast[config_name]:
            ssim_delta = r.get('ssim_corrected', 0) - r.get('ssim_backbone', 0)
            print(f"{config_name:<25} {r['dataset']:<35} {r['psnr_delta']:>+10.3f} "
                  f"{ssim_delta:>+10.4f}")

    # ── Select top-3 configs for full validation ──────────────────────
    cross_fast = [r for r in fast_sweep_results
                  if 'cross-dataset' in r.get('dataset', '').lower()
                  and r.get('config_name') != 'baseline_no_tta']

    # Rank by average PSNR delta across cross-datasets per config
    config_avg_psnr = defaultdict(list)
    for r in cross_fast:
        config_avg_psnr[r['config_name']].append(r['psnr_delta'])
    config_ranking = {name: sum(vals)/len(vals)
                      for name, vals in config_avg_psnr.items() if vals}
    top_n = 3
    top_configs = sorted(config_ranking, key=config_ranking.get, reverse=True)[:top_n]

    print(f"\n{'='*84}")
    print(f"  PHASE 2: FULL VALIDATION (top-{top_n} configs)")
    print(f"{'='*84}")
    for i, name in enumerate(top_configs, 1):
        print(f"  #{i}: {name} (avg PSNR delta: {config_ranking[name]:+.3f} dB)")

    # ── Run full validation on top configs ─────────────────────────────
    all_sweep_results = baseline_results.copy()

    for idx, config_name in enumerate(top_configs, 1):
        config = configs[config_name]
        print(f"\n>>> Full validation {idx}/{len(top_configs)}: {config_name}")
        t0 = time.time()

        results = run_single_tta_config(
            model, cross_datasets, device, config, config_name,
            backbone_caches=backbone_caches,
            fast=False,
        )
        all_sweep_results.extend(results)

        elapsed = time.time() - t0
        print(f">>> Full validation '{config_name}' done in {elapsed:.1f}s")

        _release_memory()

    # ── Full validation summary table ─────────────────────────────────
    print("\n" + "="*120)
    print("  TTA HYPERPARAMETER SWEEP SUMMARY (FULL VALIDATION)")
    print("="*120)

    by_config = defaultdict(list)
    for r in all_sweep_results:
        by_config[r.get('config_name', 'unknown')].append(r)

    print(f"\n{'Config':<25} {'Dataset':<35} {'PSNR-delta':>10} {'CNR%':>8} "
          f"{'Clinical':>10} {'Preds':>6} {'Verdict':<25}")
    print("-"*120)

    for config_name in sorted(by_config.keys()):
        for r in by_config[config_name]:
            print(f"{config_name:<25} {r['dataset']:<35} {r['psnr_delta']:>+10.3f} "
                  f"{r['cnr_change_pct']:>+7.1f}% "
                  f"{r['clinical_improved']}/4 ({r['clinical_ratio']:.3f}) "
                  f"{r['predicates_passing']:>3}/5  {r['verdict']}")
    print("="*120)

    # Find best cross-dataset config
    cross_results = [r for r in all_sweep_results
                     if 'cross-dataset' in r.get('dataset', '').lower()
                     and r.get('config_name') != 'baseline_no_tta']
    if cross_results:
        viable = [r for r in cross_results if r['clinical_improved'] >= 3]
        if not viable:
            viable = cross_results
        best = max(viable, key=lambda r: r['psnr_delta'])
        print(f"\nBest cross-dataset config: {best['config_name']}")
        print(f"  Dataset: {best['dataset']}")
        print(f"  PSNR delta: {best['psnr_delta']:+.3f} dB")
        print(f"  CNR change: {best['cnr_change_pct']:+.1f}%")
        print(f"  Clinical: {best['clinical_improved']}/4")
        print(f"  Predicates: {best['predicates_passing']}/5")
        if best.get('config'):
            print(f"  Config: {json.dumps(best['config'], indent=4)}")

    # ── Save results ─────────────────────────────────────────────────────
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        save_data = {
            'fast_screening': fast_sweep_results,
            'full_validation': all_sweep_results,
            'top_configs': top_configs,
        }
        with open(args.output_json, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
