"""
Compare all methods across different noise types to show generalization.
All models trained on universal (mixed) noise, tested on specific noise types.
"""
import os
import subprocess
import json
import numpy as np

METHODS = {
    'CASA+U-Net (Ours)': {
        'checkpoint': 'checkpoints/universal_casa/finetuned.pth',
        'adapter': 'casa',
        'backbone': 'unet',
        'type': 'casa',
    },
    'CASA+NAFNet (Ours)': {
        'checkpoint': 'checkpoints/casa_nafnet/finetuned.pth',
        'adapter': 'casa',
        'backbone': 'nafnet',
        'type': 'casa',
    },
    'NAFNet': {
        'checkpoint': 'outputs/sota/universal_nafnet/nafnet.pth',
        'model_name': 'nafnet',
        'type': 'baseline',
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

NOISE_TYPES = {
    'Gaussian': 'experiments/data_splits/val_gaussian_only.txt',
    'Rayleigh': 'experiments/data_splits/val_rayleigh_only.txt',
    'Poisson': 'experiments/data_splits/val_poisson_only.txt',
    'Moderate Gamma': 'experiments/data_splits/val_moderate_gamma_only.txt',
    'Heavy Gamma': 'experiments/data_splits/val_heavy_gamma_only.txt',
}

IMAGE_SIZE = 64

def evaluate_on_noise_type(method_name, config, noise_name, val_file):
    """Evaluate single method on single noise type."""
    checkpoint = config['checkpoint']

    if not os.path.exists(checkpoint):
        print(f"⚠️  {method_name}: Checkpoint not found - {checkpoint}")
        return None

    if not os.path.exists(val_file):
        print(f"⚠️  {noise_name}: Validation file not found - {val_file}")
        return None

    print(f"Evaluating {method_name} on {noise_name}...", flush=True)

    # Choose evaluation script
    if config.get('type') == 'casa':
        cmd = ['python', 'eval_checkpoint.py',
               '--checkpoint', checkpoint,
               '--val_pairs', val_file,
               '--adapter', config['adapter'],
               '--backbone', config.get('backbone', 'unet'),
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
            print(f"  ✓ {method_name} on {noise_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

        return {'psnr': psnr, 'ssim': ssim}

    except subprocess.TimeoutExpired:
        print(f"❌ {method_name} on {noise_name}: Timeout")
        return None
    except Exception as e:
        print(f"❌ {method_name} on {noise_name}: Error - {e}")
        return None

def main():
    print("\n" + "="*80)
    print("CROSS-NOISE GENERALIZATION COMPARISON")
    print("="*80)
    print("All models trained on UNIVERSAL (mixed) noise")
    print("Testing on specific noise types to measure generalization")
    print("="*80 + "\n")

    results = {method: {} for method in METHODS.keys()}

    for noise_name, val_file in NOISE_TYPES.items():
        print(f"\n{'='*80}")
        print(f"Noise Type: {noise_name}")
        print(f"{'='*80}")

        for method_name, config in METHODS.items():
            result = evaluate_on_noise_type(method_name, config, noise_name, val_file)
            if result:
                results[method_name][noise_name] = result
        print()

    # Print comparison table
    print("\n" + "="*80)
    print("CROSS-NOISE GENERALIZATION TABLE (PSNR in dB)")
    print("="*80)

    # Header
    header = f"{'Method':<20}"
    for noise_name in NOISE_TYPES.keys():
        header += f" | {noise_name:<12}"
    header += " | Average"
    print(header)
    print("-"*80)

    # Rows
    method_averages = {}
    for method_name in METHODS.keys():
        row = f"{method_name:<20}"
        psnr_values = []
        for noise_name in NOISE_TYPES.keys():
            if noise_name in results[method_name] and results[method_name][noise_name]['psnr']:
                psnr = results[method_name][noise_name]['psnr']
                row += f" | {psnr:>12.2f}"
                psnr_values.append(psnr)
            else:
                row += f" | {'N/A':>12}"

        avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
        method_averages[method_name] = avg_psnr
        row += f" | {avg_psnr:>7.2f}"
        print(row)

    print("="*80)

    # Find best method
    best_method = max(method_averages.items(), key=lambda x: x[1])
    print(f"\n🏆 Best Overall: {best_method[0]} (Avg PSNR: {best_method[1]:.2f} dB)")

    # Calculate generalization gap (std dev across noise types)
    print("\nGeneralization Robustness (lower std = better):")
    for method_name in METHODS.keys():
        psnr_values = [results[method_name][noise]['psnr']
                      for noise in NOISE_TYPES.keys()
                      if noise in results[method_name] and results[method_name][noise]['psnr']]
        if psnr_values:
            std = np.std(psnr_values)
            print(f"  {method_name:<20}: σ = {std:.2f} dB")

    # Save results
    with open('cross_noise_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: cross_noise_results.json")

if __name__ == "__main__":
    main()
