#!/usr/bin/env python3
"""Profile corrector speed after optimizations."""

import torch
import time
import sys
sys.path.insert(0, '/home/kumwilai/OCT')

from neuro_symbolic_v2 import EdgeCorrector, TextureCorrector, SmoothCorrector

def profile_corrector(name, corrector, in_channels, needs_denoised=True, batch_size=4, img_size=96, n_iter=10):
    """Profile corrector forward pass."""
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    denoised = torch.randn(batch_size, 1, img_size, img_size)

    # Warmup
    for _ in range(3):
        if needs_denoised:
            _ = corrector(x, denoised)
        else:
            _ = corrector(x)

    # Time
    times = []
    for _ in range(n_iter):
        start = time.time()
        if needs_denoised:
            _ = corrector(x, denoised)
        else:
            _ = corrector(x)
        times.append((time.time() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"  {name}: {avg:.1f} ms/batch (best: {min(times):.1f} ms)")
    return avg

def main():
    print("=" * 60)
    print("PROFILING OPTIMIZED CORRECTORS")
    print("Batch=4, Image=96x96, 10 iterations")
    print("=" * 60)

    # EdgeCorrector: in_channels=19 (denoised + noisy + lambda + enc1[16])
    edge = EdgeCorrector(in_channels=19)
    edge_time = profile_corrector("EdgeCorrector", edge, 19, needs_denoised=True)

    # TextureCorrector: in_channels=35 (denoised + noisy + lambda + enc1[16] + enc2[16])
    texture = TextureCorrector(in_channels=35)
    texture_time = profile_corrector("TextureCorrector", texture, 35, needs_denoised=False)

    # SmoothCorrector: in_channels=19
    smooth = SmoothCorrector(in_channels=19)
    smooth_time = profile_corrector("SmoothCorrector", smooth, 19, needs_denoised=True)

    total = edge_time + texture_time + smooth_time
    print("-" * 60)
    print(f"  TOTAL: {total:.1f} ms/batch")
    print()
    print("Previous (before optimization):")
    print("  EdgeCorrector: 555.0 ms")
    print("  TextureCorrector: 708.5 ms")
    print("  SmoothCorrector: 138.7 ms")
    print("  TOTAL: 1402.2 ms/batch")
    print()
    print(f"Speedup: {1402.2 / total:.1f}x")

if __name__ == "__main__":
    main()
