"""
Compare all methods on the same validation set.
"""
import os
import subprocess
import json
from pathlib import Path

METHODS = {
    'CASA (Ours)': {
        'checkpoint': 'checkpoints/universal_casa/finetuned.pth',
        'adapter': 'casa',
    },
    'CASA + EMA (Ours)': {
        'checkpoint': 'checkpoints/universal_casa/finetuned_ema.pth',
        'adapter': 'casa',
    },
    'DRUNet': {
        'checkpoint': 'outputs/sota/universal_drunet/best_model.pth',
        'adapter': 'drunet',
    },
    'NAFNet': {
        'checkpoint': 'outputs/sota/universal_nafnet/best_model.pth',
        'adapter': 'nafnet',
    },
    'SwinIR': {
        'checkpoint': 'outputs/sota/universal_swinir/best_model.pth',
        'adapter': 'swinir',
    },
    'Noise2Void': {
        'checkpoint': 'outputs/sota/universal_noise2void/best_model.pth',
        'adapter': 'noise2void',
    },
    'AMeta-FD': {
        'checkpoint': 'outputs/ameta_fd/ameta_fd_final.pth',
        'type': 'ameta',
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
    
    # Different evaluation for AMeta-FD
    if config.get('type') == 'ameta':
        cmd = [
            'python', 'ameta_fd.py',
            '--eval_only',
            '--checkpoint', checkpoint,
            '--val_pairs', VAL_PAIRS,
            '--image_size', str(IMAGE_SIZE),
        ]
    else:
        cmd = [
            'python', 'eval_checkpoint.py',
            '--checkpoint', checkpoint,
            '--val_pairs', VAL_PAIRS,
            '--adapter', config['adapter'],
            '--image_size', str(IMAGE_SIZE),
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
        
        # Parse PSNR and SSIM from output
        psnr, ssim = None, None
        for line in output.split('\n'):
            if 'PSNR:' in line:
                parts = line.split('±')
                if len(parts) >= 1:
                    psnr_str = parts[0].split(':')[1].strip().split()[0]
                    psnr = float(psnr_str)
            if 'SSIM:' in line:
                parts = line.split('±')
                if len(parts) >= 1:
                    ssim_str = parts[0].split(':')[1].strip().split()[0]
                    ssim = float(ssim_str)
        
        print(output)
        
        return {'psnr': psnr, 'ssim': ssim}
        
    except subprocess.TimeoutExpired:
        print(f"❌ {name}: Evaluation timed out")
        return None
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return None

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE BASELINE COMPARISON")
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
