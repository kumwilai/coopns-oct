#!/usr/bin/env python3
"""Quick probe: test different aggressiveness levels for clinical improvement.

Runs 3-epoch probes with 100 train / 30 val samples to find the best config.
"""

import subprocess
import sys
import json
import re
import os

CONFIGS = {
    "A_moderate": {
        "correction_clamp": 0.25,
        "clinical_weight": 3.5,
        "psnr_preserve_weight": 1.0,
        "ssim_preserve_weight": 1.0,
        "enl_weight": 0.3,
        "snr_weight": 0.3,
        "edge_weight": 0.6,
    },
    "B_aggressive": {
        "correction_clamp": 0.3,
        "clinical_weight": 5.0,
        "psnr_preserve_weight": 0.5,
        "ssim_preserve_weight": 0.5,
        "enl_weight": 0.5,
        "snr_weight": 0.5,
        "edge_weight": 0.8,
    },
    "C_very_aggressive": {
        "correction_clamp": 0.4,
        "clinical_weight": 8.0,
        "psnr_preserve_weight": 0.25,
        "ssim_preserve_weight": 0.25,
        "enl_weight": 0.8,
        "snr_weight": 0.8,
        "edge_weight": 1.2,
    },
    "D_clinical_focused": {
        "correction_clamp": 0.3,
        "clinical_weight": 5.0,
        "psnr_preserve_weight": 1.0,
        "ssim_preserve_weight": 0.5,
        "enl_weight": 1.0,
        "snr_weight": 1.0,
        "edge_weight": 0.8,
    },
}

# Shared params
SHARED = {
    "train_jsonl": "pku37_oct_dataset/pku37_real_train.jsonl",
    "val_jsonl": "pku37_oct_dataset/pku37_real_val.jsonl",
    "pretrained_backbone": "outputs/nafnet_pku37_w40/best_model.pth",
    "backbone": "nafnet",
    "freeze_backbone": True,
    "epochs": 3,
    "batch_size": 4,
    "val_every": 1,
    "max_train": 100,
    "max_val": 10,
    "cnr_weight": 0.3,
    "cooperation_weight": 0.5,
    "epi_weight": 2.5,
    "no_compile": True,
}


def build_cmd(config_name, config):
    """Build command line for a probe run."""
    params = {**SHARED, **config}
    out_dir = f"outputs/probe_{config_name}"
    cmd = ["python", "train_v8_cooperative.py"]
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    cmd.extend(["--output_dir", out_dir])
    return cmd, out_dir


def parse_metrics_from_output(output_text):
    """Parse clinical metrics from training output."""
    metrics = {}

    # Find last epoch's metrics
    lines = output_text.split('\n')

    # Parse standard clinical metrics
    for line in lines:
        # CNR
        m = re.search(r'CNR \(Contrast-to-Noise\)\s+[\d.]+\s+([\d.]+)\s+([+-]?[\d.]+)', line)
        if m:
            metrics['cnr_corrected'] = float(m.group(1))
            metrics['cnr_delta'] = float(m.group(2))

        # TCI
        m = re.search(r'TCI \(Tissue Contrast Index\)\s+[\d.]+\s+([\d.]+)\s+([+-]?[\d.]+)', line)
        if m:
            metrics['tci_corrected'] = float(m.group(1))
            metrics['tci_delta'] = float(m.group(2))

        # EPI
        m = re.search(r'EPI \(Edge Preservation\)\s+[\d.]+\s+([\d.]+)\s+([+-]?[\d.]+)', line)
        if m:
            metrics['epi_corrected'] = float(m.group(1))
            metrics['epi_delta'] = float(m.group(2))

        # Boundary Sharpness
        m = re.search(r'Boundary Sharpness\s+[\d.]+\s+([\d.]+)\s+([+-]?[\d.]+)', line)
        if m:
            metrics['bs_corrected'] = float(m.group(1))
            metrics['bs_delta'] = float(m.group(2))

        # ENL
        m = re.search(r'ENL \(Equiv\. Number of Looks\)\s+[\d.]+\s+([\d.]+)\s+([+-]?[\d.]+)', line)
        if m:
            metrics['enl_corrected'] = float(m.group(1))
            metrics['enl_delta'] = float(m.group(2))

        # SNR
        m = re.search(r'SNR \(Signal-to-Noise\)\s+[\d.]+\s+([\d.]+)\s+([+-]?[\d.]+)', line)
        if m:
            metrics['snr_corrected'] = float(m.group(1))
            metrics['snr_delta'] = float(m.group(2))

        # PSNR
        m = re.search(r'PSNR \(dB\)\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)', line)
        if m:
            metrics['psnr_backbone'] = float(m.group(1))
            metrics['psnr_corrected'] = float(m.group(2))
            metrics['psnr_delta'] = float(m.group(3))

        # Correction magnitude
        m = re.search(r'Overall Correction Magnitude\s+([\d.]+)', line)
        if m:
            metrics['correction_mag'] = float(m.group(1))

        # Cooperation
        m = re.search(r'Uncertainty-Potential Correlation\s+([+-]?[\d.]+)', line)
        if m:
            metrics['cooperation'] = float(m.group(1))

    return metrics


def compute_score(metrics):
    """Composite score: clinical improvement with PSNR constraint."""
    if not metrics:
        return -999

    score = 0

    # Clinical metrics (higher = better)
    cnr_pct = metrics.get('cnr_delta', 0) / max(abs(metrics.get('cnr_corrected', 10) - metrics.get('cnr_delta', 0)), 0.01) * 100
    score += cnr_pct * 2  # CNR weight

    # ENL improvement (percentage)
    enl_base = max(metrics.get('enl_corrected', 100) - metrics.get('enl_delta', 0), 1)
    enl_pct = metrics.get('enl_delta', 0) / enl_base * 100
    score += min(enl_pct, 30) * 0.5  # Cap at 30%

    # SNR improvement (percentage)
    snr_base = max(metrics.get('snr_corrected', 10) - metrics.get('snr_delta', 0), 0.1)
    snr_pct = metrics.get('snr_delta', 0) / snr_base * 100
    score += min(snr_pct, 30) * 0.5

    # EPI (want positive)
    score += metrics.get('epi_delta', 0) * 500

    # TCI (want positive)
    score += metrics.get('tci_delta', 0) * 100

    # Boundary Sharpness (want positive)
    score += metrics.get('bs_delta', 0) * 50

    # PSNR penalty (penalize drops > 0.5 dB)
    psnr_d = metrics.get('psnr_delta', 0)
    if psnr_d < -1.0:
        score -= abs(psnr_d) * 20  # Heavy penalty
    elif psnr_d < -0.5:
        score -= abs(psnr_d) * 5   # Moderate penalty

    return score


def main():
    results = {}

    for name, config in CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"  PROBE: {name}")
        print(f"  Config: {json.dumps(config, indent=2)}")
        print(f"{'='*80}\n")

        cmd, out_dir = build_cmd(name, config)
        os.makedirs(out_dir, exist_ok=True)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=3600,  # 60 min max per probe
                cwd="/home/kumwilai/OCT"
            )
            output = result.stdout + result.stderr

            # Save full output
            with open(f"{out_dir}/probe_output.txt", 'w') as f:
                f.write(output)

            metrics = parse_metrics_from_output(output)
            score = compute_score(metrics)

            results[name] = {
                'config': config,
                'metrics': metrics,
                'score': score,
            }

            print(f"\n  Results for {name}:")
            print(f"    PSNR delta: {metrics.get('psnr_delta', 'N/A')}")
            print(f"    CNR delta:  {metrics.get('cnr_delta', 'N/A')}")
            print(f"    TCI delta:  {metrics.get('tci_delta', 'N/A')}")
            print(f"    EPI delta:  {metrics.get('epi_delta', 'N/A')}")
            print(f"    BS delta:   {metrics.get('bs_delta', 'N/A')}")
            print(f"    ENL delta:  {metrics.get('enl_delta', 'N/A')}")
            print(f"    SNR delta:  {metrics.get('snr_delta', 'N/A')}")
            print(f"    Corr. mag:  {metrics.get('correction_mag', 'N/A')}")
            print(f"    Score:      {score:.2f}")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT for {name}")
            results[name] = {'config': config, 'metrics': {}, 'score': -999}
        except Exception as e:
            print(f"  ERROR for {name}: {e}")
            results[name] = {'config': config, 'metrics': {}, 'score': -999}

    # Final comparison
    print(f"\n\n{'='*100}")
    print(f"  PROBE COMPARISON (3 epochs, 100 train / 30 val)")
    print(f"{'='*100}")
    print(f"{'Config':<25} {'PSNR Δ':>8} {'CNR Δ':>8} {'TCI Δ':>8} {'EPI Δ':>10} {'BS Δ':>8} {'ENL Δ':>8} {'SNR Δ':>8} {'Corr.Mag':>10} {'Score':>8}")
    print("-" * 110)

    best_name = None
    best_score = -999

    for name, r in results.items():
        m = r['metrics']
        s = r['score']
        print(f"{name:<25} {m.get('psnr_delta', 0):>+8.3f} {m.get('cnr_delta', 0):>+8.3f} "
              f"{m.get('tci_delta', 0):>+8.3f} {m.get('epi_delta', 0):>+10.4f} "
              f"{m.get('bs_delta', 0):>+8.4f} {m.get('enl_delta', 0):>+8.2f} "
              f"{m.get('snr_delta', 0):>+8.2f} {m.get('correction_mag', 0):>10.6f} {s:>8.2f}")
        if s > best_score:
            best_score = s
            best_name = name

    print(f"\n  BEST CONFIG: {best_name} (score: {best_score:.2f})")
    print(f"  Config: {json.dumps(results[best_name]['config'], indent=2)}")

    # Save results
    with open("probe_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to probe_results.json")


if __name__ == '__main__':
    main()
