"""
Compare available methods on the same validation set.
Only includes checkpoints that exist.
"""
import os
import subprocess
import json
from pathlib import Path

METHODS = {
    'CASA (Ours)': {
        'checkpoint': 'checkpoints/universal_casa/finetuned.pth',
        'adapter': 'casa',
        'type': 'casa',  # Uses eval_checkpoint.py
    },
    'NAFNet': {
        'checkpoint': 'outputs/sota/universal_nafnet/nafnet.pth',
        'model_name': 'nafnet',
        'type': 'baseline',  # Uses eval_baseline.py
    },
    'DRUNet': {
        'checkpoint': 'outputs/sota/universal_drunet/drunet.pth',
        'model_name': 'drunet',
        'type': 'baseline',  # Uses eval_baseline.py
    },
    'SwinIR': {
        'checkpoint': 'outputs/sota/universal_swinir/swinir.pth',
        'model_name': 'swinir',
        'type': 'baseline',  # Uses eval_baseline.py
    },
    'Noise2Void': {
        'checkpoint': 'outputs/sota/universal_noise2void/noise2void.pth',
        'model_name': 'noise2void',
        'type': 'baseline',  # Uses eval_baseline.py
    },
    'Speckle2Speckle': {
        'checkpoint': 'outputs/sota/universal_speckle2speckle/speckle2speckle.pth',
        'model_name': 'speckle2speckle',
        'type': 'baseline',  # Uses eval_baseline.py
    },
}

VAL_PAIRS = 'val_pairs_universal.txt'
IMAGE_SIZE = 64

def evaluate_method(name, config):
    """Evaluate a single method."""
    checkpoint = config['checkpoint']

    # Check if checkpoint exists
    if not os.path.exists(checkpoint):
        print(f"⚠️  {name}: Checkpoint not found - {checkpoint}")
        return None

    print(f"\n{'='*70}")
    print(f"Evaluating: {name}")
    print(f"{'='*70}")

    # Choose evaluation script based on model type
    if config.get('type') == 'casa':
        cmd = [
            'python', 'eval_checkpoint.py',
            '--checkpoint', checkpoint,
            '--val_pairs', VAL_PAIRS,
            '--adapter', config['adapter'],
            '--backbone', config.get('backbone', 'unet'),
            '--image_size', str(IMAGE_SIZE),
        ]
    else:  # baseline models
        cmd = [
            'python', 'eval_baseline.py',
            '--checkpoint', checkpoint,
            '--val_pairs', VAL_PAIRS,
            '--model_name', config['model_name'],
            '--image_size', str(IMAGE_SIZE),
        ]

    try:
        # Stream output in real-time
        import sys
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, universal_newlines=True)

        output_lines = []
        psnr, ssim = None, None

        # Read and print output line by line
        for line in process.stdout:
            print(line, end='', flush=True)  # Print immediately
            output_lines.append(line)

            # Parse PSNR and SSIM as we go
            if 'PSNR:' in line:
                try:
                    # Handle both formats: "PSNR: 28.45 ± 2.3 dB" and "PSNR: 28.45 dB"
                    if '±' in line:
                        psnr_str = line.split('±')[0].split(':')[1].strip().split()[0]
                    else:
                        psnr_str = line.split(':')[1].strip().split()[0]
                    psnr = float(psnr_str)
                except (ValueError, IndexError):
                    pass
            if 'SSIM:' in line:
                try:
                    # Handle both formats: "SSIM: 0.8234 ± 0.05" and "SSIM: 0.8234"
                    if '±' in line:
                        ssim_str = line.split('±')[0].split(':')[1].strip().split()[0]
                    else:
                        ssim_str = line.split(':')[1].strip().split()[0]
                    ssim = float(ssim_str)
                except (ValueError, IndexError):
                    pass

        # Wait for process to complete
        return_code = process.wait(timeout=600)

        if return_code != 0:
            print(f"❌ {name}: Evaluation failed with return code {return_code}")
            return None

        return {'psnr': psnr, 'ssim': ssim}

    except subprocess.TimeoutExpired:
        print(f"❌ {name}: Evaluation timed out")
        return None
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return None

def main():
    print("\n" + "="*70)
    print("BASELINE COMPARISON - Available Methods Only")
    print("="*70)
    print(f"Validation set: {VAL_PAIRS}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("="*70)

    results = {}

    for name, config in METHODS.items():
        result = evaluate_method(name, config)
        if result:
            results[name] = result

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Method':<25} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*70)

    # Sort by PSNR (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['psnr'] or 0, reverse=True)

    for name, metrics in sorted_results:
        psnr_str = f"{metrics['psnr']:.2f}" if metrics['psnr'] else "N/A"
        ssim_str = f"{metrics['ssim']:.4f}" if metrics['ssim'] else "N/A"

        # Highlight best results
        marker = "🏆 " if metrics == sorted_results[0][1] else "   "
        print(f"{marker}{name:<22} {psnr_str:<15} {ssim_str:<15}")

    print("="*70)

    # Save results to JSON
    output_file = 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Create markdown table
    md_file = 'comparison_results.md'
    with open(md_file, 'w') as f:
        f.write("# OCT Denoising Methods Comparison\n\n")
        f.write(f"**Validation Set:** {VAL_PAIRS}\n\n")
        f.write(f"**Image Size:** {IMAGE_SIZE}x{IMAGE_SIZE}\n\n")
        f.write("| Rank | Method | PSNR (dB) | SSIM |\n")
        f.write("|------|--------|-----------|------|\n")

        for rank, (name, metrics) in enumerate(sorted_results, 1):
            psnr = f"{metrics['psnr']:.2f}" if metrics['psnr'] else "N/A"
            ssim = f"{metrics['ssim']:.4f}" if metrics['ssim'] else "N/A"
            marker = "🏆" if rank == 1 else ""
            f.write(f"| {rank} {marker} | {name} | {psnr} | {ssim} |\n")

    print(f"Markdown table saved to: {md_file}")

if __name__ == "__main__":
    main()
