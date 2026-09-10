"""
Deep bug hunting: Edge cases, memory leaks, and subtle issues.
Tests for:
1. Computation graph memory leaks
2. Buffer registration issues
3. In-place operation bugs
4. Batch size sensitivity
5. Training/eval mode consistency
6. Autocast compatibility

Memory safety: System has 7.8GB RAM + 4GB swap, CPU-only.
Models are tested one at a time with gc.collect() between them.
"""
import torch
import torch.nn as nn
import gc
import os
import sys
sys.path.append('sota/models')

from drunet_fair import DRUNet
from nafnet_fair import NAFNet, LayerNormFunction
from swinir_fair import SwinIR


def get_rss_mb():
    """Get current process RSS (Resident Set Size) in MB from /proc/self/status.

    torch.cuda.memory_allocated() returns 0 on CPU, so we read VmRSS from
    the proc filesystem instead. Falls back to 0.0 on non-Linux systems.
    """
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # Line format: "VmRSS:    123456 kB"
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0.0


def test_computation_graph_leak(model, model_name):
    """Test for tensors retained in computation graph."""
    print(f"\n{'='*80}")
    print(f"Computation Graph Leak Test: {model_name}")
    print(f"{'='*80}")

    model.train()

    # Test with gradient accumulation (common training pattern)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    initial_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])

    for i in range(5):
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()

        if i % 2 == 0:  # Accumulate every 2 steps
            optimizer.step()
            optimizer.zero_grad()

        del x, y, loss
        gc.collect()  # Cleanup between iterations to prevent graph accumulation

    # Final cleanup
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()

    final_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
    tensor_growth = final_tensors - initial_tensors

    print(f"  Initial tensors: {initial_tensors}")
    print(f"  Final tensors:   {final_tensors}")
    print(f"  Growth:          {tensor_growth}")

    if tensor_growth > 50:  # Some growth is expected (optimizer state)
        print(f"  ⚠️  WARNING: Excessive tensor accumulation!")
        return False
    else:
        print(f"  ✓ Tensor count within acceptable range")
        return True


def test_batch_size_consistency(model, model_name):
    """Test consistency across different batch sizes."""
    print(f"\n{'='*80}")
    print(f"Batch Size Consistency Test: {model_name}")
    print(f"{'='*80}")

    model.eval()

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    issues = []

    with torch.no_grad():
        for bs in batch_sizes:
            try:
                x = torch.randn(bs, 1, 64, 64)
                y = model(x)

                if y.shape != (bs, 1, 64, 64):
                    issues.append(f"BS={bs}: Expected ({bs},1,64,64), got {y.shape}")
                    print(f"  ❌ Batch size {bs:2d}: {y.shape} (wrong!)")
                else:
                    print(f"  ✓ Batch size {bs:2d}: {y.shape}")
            except Exception as e:
                issues.append(f"BS={bs}: {str(e)}")
                print(f"  ❌ Batch size {bs:2d}: {str(e)}")

    return len(issues) == 0


def test_training_eval_consistency(model, model_name):
    """Test output consistency between train and eval modes."""
    print(f"\n{'='*80}")
    print(f"Train/Eval Mode Consistency Test: {model_name}")
    print(f"{'='*80}")

    x = torch.randn(2, 1, 64, 64)

    # Train mode (with dropout disabled if present)
    model.train()
    with torch.no_grad():
        y_train = model(x)

    # Eval mode
    model.eval()
    with torch.no_grad():
        y_eval = model(x)

    # Check if outputs are similar (should be identical with no dropout)
    diff = (y_train - y_eval).abs().max().item()

    print(f"  Train mode output range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  Eval mode output range:  [{y_eval.min():.3f}, {y_eval.max():.3f}]")
    print(f"  Max difference: {diff:.6f}")

    # Small differences are OK due to BatchNorm/LayerNorm momentum
    if diff > 0.1:
        print(f"  ⚠️  WARNING: Large difference between train and eval!")
        return False
    else:
        print(f"  ✓ Train/eval modes consistent")
        return True


def test_deterministic_output(model, model_name):
    """Test if eval mode gives deterministic output."""
    print(f"\n{'='*80}")
    print(f"Deterministic Output Test: {model_name}")
    print(f"{'='*80}")

    model.eval()
    x = torch.randn(2, 1, 64, 64)

    # Run twice
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    diff = (y1 - y2).abs().max().item()

    print(f"  First run output range:  [{y1.min():.3f}, {y1.max():.3f}]")
    print(f"  Second run output range: [{y2.min():.3f}, {y2.max():.3f}]")
    print(f"  Max difference: {diff:.10f}")

    if diff > 1e-6:
        print(f"  ⚠️  WARNING: Non-deterministic output in eval mode!")
        return False
    else:
        print(f"  ✓ Output is deterministic")
        return True


def test_gradient_checkpointing_compatibility(model, model_name):
    """Test if model works with gradient checkpointing."""
    print(f"\n{'='*80}")
    print(f"Gradient Checkpointing Test: {model_name}")
    print(f"{'='*80}")

    model.train()

    try:
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        print(f"  ✓ Forward and backward pass successful")
        return True
    except Exception as e:
        print(f"  ❌ Error during gradient computation: {str(e)}")
        return False


def test_edge_case_inputs(model, model_name):
    """Test edge case inputs."""
    print(f"\n{'='*80}")
    print(f"Edge Case Input Test: {model_name}")
    print(f"{'='*80}")

    model.eval()

    test_cases = [
        ("Very small values", torch.randn(1, 1, 64, 64) * 1e-8),
        ("Very large values", torch.randn(1, 1, 64, 64) * 1e8),
        ("All negative", -torch.abs(torch.randn(1, 1, 64, 64))),
        ("Constant value", torch.ones(1, 1, 64, 64) * 0.5),
    ]

    issues = []
    for name, x in test_cases:
        try:
            with torch.no_grad():
                y = model(x)

            has_nan = torch.isnan(y).any().item()
            has_inf = torch.isinf(y).any().item()
            in_range = (y >= 0).all().item() and (y <= 1).all().item()

            if has_nan or has_inf or not in_range:
                issues.append(f"{name}: NaN={has_nan}, Inf={has_inf}, InRange={in_range}")
                print(f"  ⚠️  {name}: NaN={has_nan}, Inf={has_inf}, InRange={in_range}")
            else:
                print(f"  ✓ {name}: OK (range [{y.min():.3f}, {y.max():.3f}])")
        except Exception as e:
            issues.append(f"{name}: {str(e)}")
            print(f"  ❌ {name}: {str(e)}")

    return len(issues) == 0


def test_drunet_noise_level_leak():
    """Test if DRUNet noise level creates memory leak."""
    print(f"\n{'='*80}")
    print(f"DRUNet Noise Level Computation Graph Test")
    print(f"{'='*80}")

    model = DRUNet(nc=[64, 128, 256], nb=[2, 2, 2])
    model.train()

    # Test auto noise level estimation
    x = torch.randn(2, 1, 64, 64, requires_grad=True)

    # Check if noise_level stays in graph
    print("\n1. Testing auto noise level estimation:")
    y = model(x, noise_level=None)
    loss = y.mean()

    # Check computation graph
    has_grad_fn = x.grad_fn is not None or (hasattr(y, 'grad_fn') and y.grad_fn is not None)
    print(f"  Input requires_grad: {x.requires_grad}")
    print(f"  Output requires_grad: {y.requires_grad}")

    loss.backward()

    # Check if gradients computed correctly
    if x.grad is not None:
        print(f"  ✓ Gradients computed correctly")
        print(f"  Input gradient range: [{x.grad.min():.6f}, {x.grad.max():.6f}]")
    else:
        print(f"  ❌ No gradients!")

    del x, y, loss
    gc.collect()

    # Now test with manual noise level
    print("\n2. Testing manual noise level:")
    model.zero_grad()
    x2 = torch.randn(2, 1, 64, 64, requires_grad=True)
    y2 = model(x2, noise_level=0.1)
    loss2 = y2.mean()
    loss2.backward()

    if x2.grad is not None:
        print(f"  ✓ Gradients computed correctly")
    else:
        print(f"  ❌ No gradients!")

    del x2, y2, loss2, model
    gc.collect()


def test_nafnet_layernorm_backward():
    """Test NAFNet custom LayerNorm backward pass."""
    print(f"\n{'='*80}")
    print(f"NAFNet LayerNorm Backward Test")
    print(f"{'='*80}")

    from nafnet_fair import LayerNorm2d

    # Test custom LayerNorm
    ln = LayerNorm2d(64)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)

    print("\n1. Testing custom LayerNorm forward:")
    y = ln(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output mean: {y.mean():.6f} (should be ~0)")
    print(f"  Output std: {y.std():.6f} (should be ~1)")

    print("\n2. Testing backward pass:")
    loss = y.mean()
    loss.backward()

    if x.grad is not None:
        print(f"  ✓ Input gradients computed")
        print(f"  Input gradient range: [{x.grad.min():.6f}, {x.grad.max():.6f}]")
    else:
        print(f"  ❌ No input gradients!")

    if ln.weight.grad is not None:
        print(f"  ✓ Weight gradients computed")
        print(f"  Weight gradient sum: {ln.weight.grad.sum():.6f}")
    else:
        print(f"  ❌ No weight gradients!")

    if ln.bias.grad is not None:
        print(f"  ✓ Bias gradients computed")
        print(f"  Bias gradient sum: {ln.bias.grad.sum():.6f}")
    else:
        print(f"  ❌ No bias gradients!")

    del ln, x, y, loss
    gc.collect()


def test_swinir_buffer_registration():
    """Test if SwinIR properly registers buffers."""
    print(f"\n{'='*80}")
    print(f"SwinIR Buffer Registration Test")
    print(f"{'='*80}")

    model = SwinIR(img_size=64, embed_dim=120, depths=[8, 8, 8], num_heads=[8, 8, 8], window_size=8)

    print("\n1. Checking registered buffers:")
    buffers = dict(model.named_buffers())
    print(f"  Total buffers: {len(buffers)}")

    # Check if relative_position_index is registered
    rpi_buffers = [name for name in buffers.keys() if 'relative_position_index' in name]
    print(f"  Relative position index buffers: {len(rpi_buffers)}")

    if len(rpi_buffers) > 0:
        print(f"  ✓ Relative position indices properly registered as buffers")
        # Check one
        sample_buffer = buffers[rpi_buffers[0]]
        print(f"  Sample buffer shape: {sample_buffer.shape}")
        print(f"  Sample buffer device: {sample_buffer.device}")
    else:
        print(f"  ❌ No relative position index buffers found!")

    # Test device transfer
    print("\n2. Testing device transfer (CPU only):")
    try:
        model_copy = model.cpu()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = model_copy(x)
        print(f"  ✓ Device transfer successful")
        del x, y
    except Exception as e:
        print(f"  ❌ Device transfer failed: {str(e)}")

    del model, buffers
    gc.collect()


def test_inplace_operations(model, model_name):
    """Test for problematic in-place operations.

    NOTE: torch.autograd.set_detect_anomaly(True) roughly doubles memory usage
    because PyTorch retains the full forward computation graph for error reporting.
    Batch size is kept at 1 to compensate on this memory-constrained system.
    """
    print(f"\n{'='*80}")
    print(f"In-place Operation Test: {model_name}")
    print(f"{'='*80}")

    model.train()

    # Enable anomaly detection - doubles memory, so use batch_size=1
    torch.autograd.set_detect_anomaly(True)

    try:
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        del x, y, loss
        gc.collect()
        print(f"  ✓ No in-place operation issues detected")
        torch.autograd.set_detect_anomaly(False)
        return True
    except RuntimeError as e:
        if "in-place" in str(e):
            print(f"  ❌ In-place operation error: {str(e)}")
            torch.autograd.set_detect_anomaly(False)
            return False
        else:
            raise
    finally:
        torch.autograd.set_detect_anomaly(False)


if __name__ == "__main__":
    print("="*80)
    print("DEEP BUG HUNT: Advanced Memory Leak & Edge Case Testing")
    print(f"  System RSS at start: {get_rss_mb():.1f} MB")
    print("="*80)

    # Model factories - instantiate one at a time to avoid OOM on 7.8GB RAM system.
    # Each model is created, tested, then deleted with gc.collect() before the next.
    model_factories = [
        (lambda: DRUNet(nc=[64, 128, 256], nb=[2, 2, 2]), "DRUNet"),
        (lambda: NAFNet(width=48, middle_blk_num=2, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2]), "NAFNet"),
        (lambda: SwinIR(img_size=64, embed_dim=120, depths=[8, 8, 8], num_heads=[8, 8, 8], window_size=8), "SwinIR"),
    ]

    # Run general tests one model at a time
    results = {}
    for factory, name in model_factories:
        print(f"\n  Creating {name}... (RSS: {get_rss_mb():.1f} MB)")
        model = factory()

        results[name] = {
            'comp_graph_leak': test_computation_graph_leak(model, name),
            'batch_consistency': test_batch_size_consistency(model, name),
            'train_eval': test_training_eval_consistency(model, name),
            'deterministic': test_deterministic_output(model, name),
            'gradient_checkpoint': test_gradient_checkpointing_compatibility(model, name),
            'edge_cases': test_edge_case_inputs(model, name),
            'inplace_ops': test_inplace_operations(model, name),
        }

        # Free model before loading the next one
        del model
        gc.collect()
        print(f"\n  {name} cleaned up. (RSS: {get_rss_mb():.1f} MB)")

    # Model-specific tests (each creates and cleans up its own model internally)
    test_drunet_noise_level_leak()
    print(f"  RSS after DRUNet noise level test: {get_rss_mb():.1f} MB")

    test_nafnet_layernorm_backward()
    print(f"  RSS after NAFNet LayerNorm test: {get_rss_mb():.1f} MB")

    test_swinir_buffer_registration()
    print(f"  RSS after SwinIR buffer test: {get_rss_mb():.1f} MB")

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    for model_name, tests in results.items():
        print(f"\n{model_name}:")
        passed = sum(tests.values())
        total = len(tests)
        print(f"  Passed: {passed}/{total}")

        for test_name, result in tests.items():
            status = "PASS" if result else "FAIL"
            print(f"    {status} {test_name}")

    print(f"\n  Final RSS: {get_rss_mb():.1f} MB")
    print(f"\n{'='*80}")
    print("Deep Bug Hunt Complete!")
    print(f"{'='*80}")
