"""
Compare all methods across different pathologies to show clinical generalization.
All models trained on universal (all pathologies), tested on specific pathologies.
"""
import os
import subprocess
import json
import numpy as np

METHODS = {
    'CASA (Ours)': {
        'checkpoint': 'checkpoints/universal_casa/finetuned.pth',
        'adapter': 'casa',
        'type': 'casa',
    },
    'DRUNet': {
        'checkpoint': 'outputs/sota/universal_drunet/drunet.pth',
        'model_name': 'drunet',
        'type': 'baseline',
    },
    'SwinIR': {
        'checkpoint': 'outputs/sota/universal_swinir/swinir.pth',
        'model_name': 'swinir',
        'type': 'baseline',
    },
    'Noise2Void': {
        'checkpoint': 'outputs/sota/universal_noise2void/noise2void.pth',
        'model_name': 'noise2void',
        'type': 'baseline',
    },
    'Speckle2Speckle': {
        'checkpoint': 'outputs/sota/universal_speckle2speckle/speckle2speckle.pth',
        'model_name': 'speckle2speckle',
        'type': 'baseline',
    },
}

PATHOLOGIES = {
    'Normal': 'experiments/data_splits/val_normal_only.txt',
    'DME': 'experiments/data_splits/val_dme_only.txt',
    'CNV': 'experiments/data_splits/val_cnv_only.txt',
    'DRUSEN': 'experiments/data_splits/val_drusen_only.txt',
}

IMAGE_SIZE = 64

def evaluate_on_pathology(method_name, config, pathology_name, val_file):
    """Evaluate single method on single pathology."""
    checkpoint = config['checkpoint']

    if not os.path.exists(checkpoint):
        return None

    if not os.path.exists(val_file):
        return None

    print(f"Evaluating {method_name} on {pathology_name}...", flush=True)

    # Choose evaluation script
    if config.get('type') == 'casa':
        cmd = ['python', 'eval_checkpoint.py',
               '--checkpoint', checkpoint,
               '--val_pairs', val_file,
               '--adapter', config['adapter'],
               '--image_size', str(IMAGE_SIZE)]
    else:
        cmd = ['python', 'eval_baseline.py',
               '--checkpoint', checkpoint,
               '--val_pairs', val_file,
               '--model_name', config['model_name'],
               '--image_size', str(IMAGE_SIZE)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout

        # Parse PSNR and SSIM
        psnr, ssim = None, None
        for line in output.split('\n'):
            if 'PSNR:' in line:
                try:
                    if '±' in line:
                        psnr_str = line.split('±')[0].split(':')[1].strip().split()[0]
                    else:
                        psnr_str = line.split(':')[1].strip().split()[0]
                    psnr = float(psnr_str)
                except (ValueError, IndexError):
                    pass
            if 'SSIM:' in line:
                try:
                    if '±' in line:
                        ssim_str = line.split('±')[0].split(':')[1].strip().split()[0]
                    else:
                        ssim_str = line.split(':')[1].strip().split()[0]
                    ssim = float(ssim_str)
                except (ValueError, IndexError):
                    pass

        if psnr is not None:
            print(f"  ✓ {method_name} on {pathology_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

        return {'psnr': psnr, 'ssim': ssim}

    except Exception as e:
        return None

def main():
    print("\n" + "="*80)
    print("CROSS-PATHOLOGY GENERALIZATION COMPARISON")
    print("="*80)
    print("All models trained on UNIVERSAL (all pathologies)")
    print("Testing on specific pathologies to measure clinical generalization")
    print("="*80 + "\n")

    results = {method: {} for method in METHODS.keys()}

    for pathology_name, val_file in PATHOLOGIES.items():
        print(f"\n{'='*80}")
        print(f"Pathology: {pathology_name}")
        print(f"{'='*80}")

        for method_name, config in METHODS.items():
            result = evaluate_on_pathology(method_name, config, pathology_name, val_file)
            if result:
                results[method_name][pathology_name] = result
        print()

    # Print comparison table
    print("\n" + "="*80)
    print("CROSS-PATHOLOGY PERFORMANCE TABLE (PSNR in dB)")
    print("="*80)

    # Header
    header = f"{'Method':<20}"
    for pathology_name in PATHOLOGIES.keys():
        header += f" | {pathology_name:<8}"
    header += " | Average"
    print(header)
    print("-"*80)

    # Rows
    method_averages = {}
    for method_name in METHODS.keys():
        row = f"{method_name:<20}"
        psnr_values = []
        for pathology_name in PATHOLOGIES.keys():
            if pathology_name in results[method_name] and results[method_name][pathology_name]['psnr']:
                psnr = results[method_name][pathology_name]['psnr']
                row += f" | {psnr:>8.2f}"
                psnr_values.append(psnr)
            else:
                row += f" | {'N/A':>8}"

        avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
        method_averages[method_name] = avg_psnr
        row += f" | {avg_psnr:>7.2f}"
        print(row)

    print("="*80)

    # Find best method
    best_method = max(method_averages.items(), key=lambda x: x[1])
    print(f"\n🏆 Best Overall: {best_method[0]} (Avg PSNR: {best_method[1]:.2f} dB)")

    # Save results
    with open('cross_pathology_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: cross_pathology_results.json")

if __name__ == "__main__":
    main()
