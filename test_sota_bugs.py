"""
Comprehensive bug and memory leak testing for SOTA models.
Tests for:
1. Memory leaks (tensor accumulation)
2. Shape mismatches
3. Gradient flow issues
4. Numerical stability
5. Architectural bugs
"""
import torch
import torch.nn as nn
import gc
import sys
sys.path.append('sota/models')

from drunet_fair import DRUNet
from nafnet_fair import NAFNet
from swinir_fair import SwinIR


def get_memory_allocated():
    """Get current memory in MB (GPU if available, else RSS)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return 0


def test_memory_leak(model, model_name, num_iterations=10):
    """Test for memory leaks during repeated forward passes."""
    print(f"\n{'='*80}")
    print(f"Memory Leak Test: {model_name}")
    print(f"{'='*80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Warmup
    x = torch.randn(4, 1, 64, 64, device=device)
    with torch.no_grad():
        _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    memory_before = get_memory_allocated()

    # Run multiple iterations
    memory_per_iter = []
    for i in range(num_iterations):
        x = torch.randn(4, 1, 64, 64, device=device)
        with torch.no_grad():
            y = model(x)

        # Force cleanup
        del x, y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mem = get_memory_allocated()
        memory_per_iter.append(mem)
        print(f"  Iter {i+1:2d}: {mem:.2f} MB")

    memory_after = get_memory_allocated()
    memory_growth = memory_after - memory_before

    print(f"\n  Memory before: {memory_before:.2f} MB")
    print(f"  Memory after:  {memory_after:.2f} MB")
    print(f"  Growth:        {memory_growth:.2f} MB")

    if memory_growth > 10:  # More than 10MB growth is suspicious
        print(f"  ⚠️  WARNING: Potential memory leak detected!")
    else:
        print(f"  ✓ No significant memory leak")

    return memory_growth


def test_gradient_flow(model, model_name):
    """Test gradient flow through the model."""
    print(f"\n{'='*80}")
    print(f"Gradient Flow Test: {model_name}")
    print(f"{'='*80}")

    model.train()
    x = torch.randn(2, 1, 64, 64, requires_grad=True)

    # Forward pass
    y = model(x)
    loss = y.mean()

    # Backward pass
    loss.backward()

    # Check for None gradients
    none_grads = []
    zero_grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            none_grads.append(name)
        elif param.grad.abs().sum() == 0:
            zero_grads.append(name)

    if none_grads:
        print(f"  ⚠️  Parameters with None gradients: {len(none_grads)}")
        for name in none_grads[:5]:  # Show first 5
            print(f"    - {name}")
    else:
        print(f"  ✓ All parameters have gradients")

    if zero_grads:
        print(f"  ⚠️  Parameters with zero gradients: {len(zero_grads)}")
        for name in zero_grads[:5]:
            print(f"    - {name}")
    else:
        print(f"  ✓ No zero gradients detected")

    return len(none_grads), len(zero_grads)


def test_nan_inf(model, model_name):
    """Test for NaN/Inf in outputs."""
    print(f"\n{'='*80}")
    print(f"NaN/Inf Test: {model_name}")
    print(f"{'='*80}")

    model.eval()

    test_cases = [
        ("Normal input", torch.randn(2, 1, 64, 64)),
        ("Zero input", torch.zeros(2, 1, 64, 64)),
        ("Ones input", torch.ones(2, 1, 64, 64)),
        ("Large values", torch.randn(2, 1, 64, 64) * 100),
        ("Small values", torch.randn(2, 1, 64, 64) * 0.001),
    ]

    issues = []
    for name, x in test_cases:
        with torch.no_grad():
            try:
                y = model(x)
                has_nan = torch.isnan(y).any().item()
                has_inf = torch.isinf(y).any().item()

                if has_nan or has_inf:
                    issues.append(f"{name}: NaN={has_nan}, Inf={has_inf}")
                    print(f"  ⚠️  {name}: NaN={has_nan}, Inf={has_inf}")
                else:
                    print(f"  ✓ {name}: OK")
            except Exception as e:
                issues.append(f"{name}: {str(e)}")
                print(f"  ❌ {name}: {str(e)}")

    if not issues:
        print(f"  ✓ All tests passed")

    return len(issues)


def test_shape_consistency(model, model_name):
    """Test output shape consistency with different input sizes."""
    print(f"\n{'='*80}")
    print(f"Shape Consistency Test: {model_name}")
    print(f"{'='*80}")

    model.eval()

    test_sizes = [
        (32, 32),
        (64, 64),
        (128, 128),
        (96, 96),  # Non-power-of-2
        (80, 96),  # Non-square
    ]

    issues = []
    for h, w in test_sizes:
        x = torch.randn(1, 1, h, w)
        try:
            with torch.no_grad():
                y = model(x)

            if y.shape[2:] != (h, w):
                issues.append(f"Size ({h}x{w}): Expected {(h,w)}, got {y.shape[2:]}")
                print(f"  ⚠️  Input ({h}x{w}) -> Output {y.shape[2:]} (mismatch!)")
            else:
                print(f"  ✓ Input ({h}x{w}) -> Output {y.shape[2:]}")
        except Exception as e:
            issues.append(f"Size ({h}x{w}): {str(e)}")
            print(f"  ❌ Input ({h}x{w}): {str(e)}")

    if not issues:
        print(f"  ✓ All sizes handled correctly")

    return len(issues)


def test_drunet_specific():
    """Test DRUNet-specific potential bugs."""
    print(f"\n{'='*80}")
    print(f"DRUNet-Specific Bug Tests")
    print(f"{'='*80}")

    model = DRUNet(nc=[48, 96, 192], nb=[2, 2, 2])

    # Bug 1: Check if noise_level creates computation graph
    print("\n1. Noise level computation graph test:")
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        y1 = model(x, noise_level=None)  # Auto-estimate
        y2 = model(x, noise_level=0.1)   # Provided
    print(f"  ✓ Auto noise level: output shape {y1.shape}")
    print(f"  ✓ Manual noise level: output shape {y2.shape}")

    # Bug 2: Check bottleneck architecture
    print("\n2. Bottleneck architecture check:")
    print(f"  Number of encoder levels: {len(model.m_down)}")
    print(f"  Last encoder level (bottleneck) blocks: {len(model.m_down[-1])}")

    # Check if m_body exists (old buggy version)
    if hasattr(model, 'm_body'):
        print(f"  ⚠️  WARNING: model.m_body exists (old buggy version)")
        print(f"      Bottleneck has double blocks!")
        total_bottleneck_blocks = len(model.m_down[-1]) + len(model.m_body)
        print(f"      Total: {total_bottleneck_blocks} blocks")
    else:
        print(f"  ✓ No separate m_body (bug fixed!)")
        print(f"  ✓ Bottleneck correctly uses last encoder level only")


def test_nafnet_specific():
    """Test NAFNet-specific potential bugs."""
    print(f"\n{'='*80}")
    print(f"NAFNet-Specific Bug Tests")
    print(f"{'='*80}")

    model = NAFNet(width=48, middle_blk_num=2, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2])

    # Bug 1: Check orphaned ModuleList
    print("\n1. Module initialization check:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
        if name == "middle_blks":
            if isinstance(module, nn.Sequential):
                print(f"    ✓ middle_blks is Sequential (correct)")
            elif isinstance(module, nn.ModuleList):
                print(f"    ⚠️  middle_blks is ModuleList (potential bug)")

    # Bug 2: Test padding behavior
    print("\n2. Padding behavior test:")
    test_sizes = [(64, 64), (65, 65), (63, 63)]
    for h, w in test_sizes:
        x = torch.randn(1, 1, h, w)
        with torch.no_grad():
            y = model(x)
        print(f"  Input ({h:2d}x{w:2d}) -> Output {y.shape[2:]}, correct={y.shape[2:] == (h, w)}")


def test_swinir_specific():
    """Test SwinIR-specific potential bugs."""
    print(f"\n{'='*80}")
    print(f"SwinIR-Specific Bug Tests")
    print(f"{'='*80}")

    model = SwinIR(img_size=64, embed_dim=120, depths=[8, 8, 8], num_heads=[8, 8, 8], window_size=8)

    # Bug 1: Window size divisibility
    print("\n1. Window size divisibility test:")
    test_sizes = [(64, 64), (72, 72), (80, 80)]
    for h, w in test_sizes:
        divisible_h = (h % model.window_size) == 0
        divisible_w = (w % model.window_size) == 0
        print(f"  Size ({h}x{w}): H divisible={divisible_h}, W divisible={divisible_w}")

        if not (divisible_h and divisible_w):
            print(f"    Testing forward pass with non-divisible size...")
            x = torch.randn(1, 1, h, w)
            try:
                with torch.no_grad():
                    y = model(x)
                print(f"    ✓ Handled successfully, output: {y.shape}")
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")

    # Bug 2: Relative position bias
    print("\n2. Relative position bias check:")
    first_block = model.layers[0][0]
    print(f"  Window size: {first_block.attn.window_size}")
    print(f"  Num heads: {first_block.attn.num_heads}")
    print(f"  Relative position bias table shape: {first_block.attn.relative_position_bias_table.shape}")
    print(f"  Relative position index shape: {first_block.attn.relative_position_index.shape}")
    expected_table_size = (2 * model.window_size - 1) ** 2
    actual_table_size = first_block.attn.relative_position_bias_table.shape[0]
    if actual_table_size == expected_table_size:
        print(f"  ✓ Relative position table size correct: {actual_table_size}")
    else:
        print(f"  ⚠️  Table size mismatch: expected {expected_table_size}, got {actual_table_size}")


if __name__ == "__main__":
    print("="*80)
    print("SOTA Models Bug & Memory Leak Testing")
    print(f"Initial RSS: {get_memory_allocated():.0f} MB")
    print("="*80)

    # Test one model at a time to avoid OOM on 7.8GB RAM system
    model_factories = [
        (lambda: DRUNet(nc=[48, 96, 192], nb=[2, 2, 2]), "DRUNet"),
        (lambda: NAFNet(width=48, middle_blk_num=2, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2]), "NAFNet"),
        (lambda: SwinIR(img_size=64, embed_dim=120, depths=[8, 8, 8], num_heads=[8, 8, 8], window_size=8), "SwinIR"),
    ]

    for factory, name in model_factories:
        print(f"\n--- Testing {name} (RSS={get_memory_allocated():.0f} MB) ---")
        model = factory()
        test_memory_leak(model, name, num_iterations=5)
        test_gradient_flow(model, name)
        model.zero_grad(set_to_none=True)
        test_nan_inf(model, name)
        test_shape_consistency(model, name)
        del model
        gc.collect()
        print(f"--- {name} done, RSS={get_memory_allocated():.0f} MB ---")

    # Run model-specific tests (each creates/destroys its own model)
    test_drunet_specific()
    gc.collect()
    test_nafnet_specific()
    gc.collect()
    test_swinir_specific()
    gc.collect()

    print(f"\n{'='*80}")
    print(f"Testing Complete! Final RSS: {get_memory_allocated():.0f} MB")
    print(f"{'='*80}")
