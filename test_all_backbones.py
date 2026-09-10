"""
Test all backbones to verify fixes don't break existing functionality.
"""
import torch
import gc
import sys
sys.path.insert(0, '/home/kumwilai/OCT')


def get_rss_mb():
    """Get current process RSS (Resident Set Size) in MB from /proc/self/status.

    Works on CPU-only Linux systems where torch.cuda.memory_allocated() returns 0.
    """
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # Line format: "VmRSS:    123456 kB"
                    return int(line.split()[1]) / 1024.0
    except (IOError, OSError, ValueError):
        pass
    return 0.0

from adaptive_oct_denoise import (
    build_model,
    NAFBackbone,
    Noise2VoidBackbone,
    Neighbor2NeighborBackbone,
    DenoisingBackbone
)

def test_backbone(backbone_type, adapter_type, residual_mode, base_channels=48):
    """Test a specific backbone configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: backbone={backbone_type}, adapter={adapter_type}, residual_mode={residual_mode}")
    print(f"{'='*80}")

    model = None
    x = None
    output = None

    try:
        # Build model
        model = build_model(
            base_channels=base_channels,
            residual_mode=residual_mode,
            adapter_type=adapter_type,
            backbone_type=backbone_type
        )

        # Create dummy input
        x = torch.randn(2, 1, 64, 64)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check output
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        assert output.min() >= 0.0, f"Output has negative values: {output.min()}"
        assert output.max() <= 1.0, f"Output exceeds 1.0: {output.max()}"

        # Check backbone type
        if backbone_type == "nafnet":
            assert isinstance(model.backbone, NAFBackbone), "Wrong backbone type"
        elif backbone_type == "noise2void":
            assert isinstance(model.backbone, Noise2VoidBackbone), "Wrong backbone type"
        elif backbone_type == "neighbor2neighbor":
            assert isinstance(model.backbone, Neighbor2NeighborBackbone), "Wrong backbone type"
        elif backbone_type == "unet":
            assert isinstance(model.backbone, DenoisingBackbone), "Wrong backbone type"

        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Backbone type: {type(model.backbone).__name__}")
        print(f"  Test PASSED")

        return True

    except Exception as e:
        print(f"  Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Explicitly release model and tensors to prevent RSS growth across
        # 24 iterations on a memory-constrained system (7.8GB RAM + 4GB swap).
        del model, x, output
        gc.collect()

def main():
    """Run all tests."""
    print("="*80)
    print("COMPREHENSIVE BACKBONE TEST SUITE")
    print("="*80)

    backbones = ["unet", "nafnet", "noise2void", "neighbor2neighbor"]
    adapters = ["global", "spatial", "casa"]
    residual_modes = [True, False]

    results = []
    total = 0
    passed = 0

    rss_start = get_rss_mb()
    print(f"Initial RSS: {rss_start:.1f} MB")

    for backbone in backbones:
        for adapter in adapters:
            for residual_mode in residual_modes:
                total += 1
                success = test_backbone(backbone, adapter, residual_mode)
                results.append((backbone, adapter, residual_mode, success))
                if success:
                    passed += 1

                # Monitor RSS to detect memory leaks across iterations
                rss_now = get_rss_mb()
                print(f"  [RSS: {rss_now:.1f} MB | delta from start: {rss_now - rss_start:+.1f} MB]")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {100*passed/total:.1f}%")

    # Show failures
    failures = [r for r in results if not r[3]]
    if failures:
        print("\n" + "="*80)
        print("FAILED TESTS:")
        print("="*80)
        for backbone, adapter, residual_mode, _ in failures:
            print(f"  - backbone={backbone}, adapter={adapter}, residual_mode={residual_mode}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
