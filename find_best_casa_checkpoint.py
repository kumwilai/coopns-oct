"""
Test all CASA checkpoints to find the best one.
"""
import subprocess
import json

CASA_CHECKPOINTS = {
    'universal_casa': 'checkpoints/universal_casa/finetuned.pth',
    'casa_finetune64': 'checkpoints/casa_finetune64/finetuned.pth',
    'casa_finetune64_ema': 'checkpoints/casa_finetune64/finetuned_ema.pth',
    'casa_finetune64_phys': 'checkpoints/casa_finetune64_phys/finetuned.pth',
    'casa_finetune64_phys_ema': 'checkpoints/casa_finetune64_phys/finetuned_ema.pth',
    'casa_finetune64_unfrozen': 'checkpoints/casa_finetune64_unfrozen/finetuned.pth',
    'casa_finetune64_unfrozen_ema': 'checkpoints/casa_finetune64_unfrozen/finetuned_ema.pth',
}

VAL_PAIRS = 'val_pairs_universal.txt'

def evaluate_checkpoint(name, checkpoint_path):
    """Evaluate a single checkpoint."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    cmd = [
        'python', 'eval_checkpoint.py',
        '--checkpoint', checkpoint_path,
        '--val_pairs', VAL_PAIRS,
        '--adapter', 'casa',
        '--image_size', '64',
    ]

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

        if psnr:
            print(f"✓ {name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

        return {'psnr': psnr, 'ssim': ssim}

    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return None

def main():
    print("\n" + "="*70)
    print("TESTING ALL CASA CHECKPOINTS")
    print("="*70)

    results = {}

    for name, checkpoint_path in CASA_CHECKPOINTS.items():
        result = evaluate_checkpoint(name, checkpoint_path)
        if result:
            results[name] = result

    # Print comparison
    print("\n" + "="*70)
    print("CASA CHECKPOINT COMPARISON")
    print("="*70)
    print(f"{'Checkpoint':<35} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-"*70)

    # Sort by PSNR
    sorted_results = sorted(results.items(), key=lambda x: x[1]['psnr'] or 0, reverse=True)

    for rank, (name, metrics) in enumerate(sorted_results, 1):
        psnr = f"{metrics['psnr']:.2f}" if metrics['psnr'] else "N/A"
        ssim = f"{metrics['ssim']:.4f}" if metrics['ssim'] else "N/A"
        marker = "🏆" if rank == 1 else f"{rank}."
        print(f"{marker} {name:<32} {psnr:<15} {ssim:<15}")

    print("="*70)

    if sorted_results:
        best_name, best_metrics = sorted_results[0]
        print(f"\n🏆 BEST CHECKPOINT: {best_name}")
        print(f"   PSNR: {best_metrics['psnr']:.2f} dB")
        print(f"   SSIM: {best_metrics['ssim']:.4f}")
        print(f"   Path: {CASA_CHECKPOINTS[best_name]}")

    # Compare with SwinIR
    print(f"\n📊 COMPARISON WITH SwinIR (28.84 dB, 0.8577 SSIM):")
    if sorted_results:
        best_psnr = sorted_results[0][1]['psnr']
        best_ssim = sorted_results[0][1]['ssim']
        psnr_diff = best_psnr - 28.84
        ssim_diff = best_ssim - 0.8577

        print(f"   PSNR: {'+' if psnr_diff >= 0 else ''}{psnr_diff:.2f} dB")
        print(f"   SSIM: {'+' if ssim_diff >= 0 else ''}{ssim_diff:.4f}")

        if psnr_diff > 0:
            print("   ✅ CASA BEATS SwinIR ON PSNR!")
        if ssim_diff > 0:
            print("   ✅ CASA BEATS SwinIR ON SSIM!")

    # Save results
    with open('casa_checkpoint_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: casa_checkpoint_comparison.json")

if __name__ == "__main__":
    main()
