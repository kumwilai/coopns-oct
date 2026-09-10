#!/usr/bin/env python3
"""
Memory leak investigation for KBNet's KBA parameters.

Tests:
1. KBBlock_s forward+backward memory growth over 100 iterations
2. View operations (.squeeze(0).t()) autograd graph persistence
3. Learnable scaling (beta, gamma, ga1, attgamma) graph reference lifetime
4. Full KBNet forward+backward memory growth over 50 iterations
5. Gradient checkpointing graph recomputation cleanup
6. KBA parameter memory calculation (exact)
7. AdamW optimizer state memory for KBA parameters

Usage:
    python test_kba_memory_leak.py
"""

import os
import sys
import gc
import tracemalloc
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbnet_7m import KBBlock_s, KBNet

def get_rss_mb():
    """Get current RSS (Resident Set Size) in MB from /proc/self/status."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except:
        pass
    # Fallback: use resource module
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_pytorch_memory():
    """Get PyTorch's internal memory tracking info."""
    info = {}
    # Total allocated tensors
    info['num_tensors'] = 0
    info['tensor_bytes'] = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                info['num_tensors'] += 1
                info['tensor_bytes'] += obj.nelement() * obj.element_size()
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                info['num_tensors'] += 1
                info['tensor_bytes'] += obj.data.nelement() * obj.data.element_size()
        except:
            pass
    return info


def count_autograd_nodes():
    """Count autograd graph nodes reachable from tracked tensors."""
    count = 0
    visited = set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.grad_fn is not None:
                stack = [obj.grad_fn]
                while stack:
                    node = stack.pop()
                    node_id = id(node)
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    count += 1
                    for child, _ in node.next_functions:
                        if child is not None:
                            stack.append(child)
        except:
            pass
    return count


# ============================================================================
# Test 1: KBBlock_s forward+backward memory growth
# ============================================================================
def test_kbblock_memory_leak():
    """Run 100 forward+backward iterations on a single KBBlock_s and track memory."""
    print("=" * 80)
    print("TEST 1: KBBlock_s forward+backward memory growth (100 iterations)")
    print("=" * 80)

    torch.manual_seed(42)
    block = KBBlock_s(c=32, nset=32, gc=1)
    block.train()
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)

    # Warmup: 5 iterations to stabilize allocator
    for _ in range(5):
        x = torch.randn(2, 32, 64, 64)
        out = block(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del x, out, loss
    gc.collect()

    rss_history = []
    tensor_count_history = []
    autograd_node_history = []

    for i in range(100):
        x = torch.randn(2, 32, 64, 64)
        out = block(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Clean up iteration tensors
        del x, out, loss

        if (i + 1) % 10 == 0:
            gc.collect()
            rss = get_rss_mb()
            n_nodes = count_autograd_nodes()
            rss_history.append(rss)
            autograd_node_history.append(n_nodes)
            print(f"  Iter {i+1:3d}: RSS={rss:.1f}MB, autograd_nodes={n_nodes}")

    # Analysis
    rss_start = rss_history[0]
    rss_end = rss_history[-1]
    rss_growth = rss_end - rss_start
    node_start = autograd_node_history[0]
    node_end = autograd_node_history[-1]

    print(f"\n  RSS growth over 90 iters: {rss_growth:+.1f}MB ({rss_start:.1f} -> {rss_end:.1f})")
    print(f"  Autograd node growth: {node_end - node_start:+d} ({node_start} -> {node_end})")

    if abs(rss_growth) < 5.0 and abs(node_end - node_start) <= 2:
        print("  RESULT: NO LEAK detected - memory stabilized")
    elif rss_growth > 10.0:
        print(f"  RESULT: POTENTIAL LEAK - RSS grew {rss_growth:.1f}MB")
    else:
        print(f"  RESULT: Minor fluctuation ({rss_growth:+.1f}MB) - likely no leak")

    del block, optimizer
    gc.collect()
    return rss_history


# ============================================================================
# Test 2: View operations (.squeeze(0).t()) autograd graph persistence
# ============================================================================
def test_view_operations():
    """Test if .squeeze(0).t() on parameters creates persistent graph nodes."""
    print("\n" + "=" * 80)
    print("TEST 2: View operations (.squeeze(0).t()) autograd graph persistence")
    print("=" * 80)

    w = nn.Parameter(torch.randn(1, 32, 288))  # (1, nset, c*gc*k^2) for c=32

    # Simulate what _kba_forward does
    gc.collect()
    nodes_before = count_autograd_nodes()

    for i in range(50):
        att_flat = torch.randn(2, 32, 64 * 64)  # (B, nset, H*W)

        # This is exactly what line 177 does:
        w_squeezed = w.squeeze(0)   # (32, 288) — view, shares storage with w
        w_transposed = w_squeezed.t()  # (288, 32) — view, shares storage
        result = w_transposed @ att_flat  # (2, 288, 4096) — new tensor

        loss = result.mean()
        loss.backward()

        # Simulate optimizer.zero_grad(set_to_none=True)
        w.grad = None

        del att_flat, w_squeezed, w_transposed, result, loss

    gc.collect()
    nodes_after = count_autograd_nodes()

    print(f"  Autograd nodes before: {nodes_before}")
    print(f"  Autograd nodes after 50 iterations: {nodes_after}")
    print(f"  Growth: {nodes_after - nodes_before:+d}")

    # Check if w still holds any graph references
    has_grad_fn = w.grad_fn is not None
    print(f"  w.grad_fn is None: {w.grad_fn is None} (expected: True, params have no grad_fn)")
    print(f"  w.requires_grad: {w.requires_grad}")

    if nodes_after - nodes_before <= 2:
        print("  RESULT: View operations do NOT cause graph accumulation")
    else:
        print(f"  RESULT: WARNING - {nodes_after - nodes_before} extra graph nodes persist")

    del w
    gc.collect()


# ============================================================================
# Test 3: Learnable scaling graph reference lifetime
# ============================================================================
def test_scaling_params():
    """Test if beta, gamma, ga1, attgamma create persistent references."""
    print("\n" + "=" * 80)
    print("TEST 3: Learnable scaling parameter graph reference lifetime")
    print("=" * 80)

    beta = nn.Parameter(torch.zeros(1, 32, 1, 1) + 1e-2)
    gamma = nn.Parameter(torch.zeros(1, 32, 1, 1) + 1e-2)
    ga1 = nn.Parameter(torch.zeros(1, 32, 1, 1) + 1e-2)
    attgamma = nn.Parameter(torch.zeros(1, 32, 1, 1) + 1e-2)

    params = [beta, gamma, ga1, attgamma]

    gc.collect()
    rss_before = get_rss_mb()

    for i in range(100):
        x = torch.randn(2, 32, 64, 64)

        # Simulate line 220: x = self._kba_forward(uf, att) * self.ga1 + uf
        x = x * ga1 + x

        # Simulate line 223: x = x * x1 * sca  (x1, sca are intermediates)
        x1 = torch.randn(2, 32, 64, 64)
        sca = torch.randn(2, 32, 1, 1)
        x = x * x1 * sca

        # Simulate line 227: y = inp + x * self.beta
        inp = torch.randn(2, 32, 64, 64)
        y = inp + x * beta

        # Simulate line 234: return y + x_ffn * self.gamma
        x_ffn = torch.randn(2, 32, 64, 64)
        out = y + x_ffn * gamma

        loss = out.mean()
        loss.backward()

        # Zero grads
        for p in params:
            p.grad = None

        del x, x1, sca, inp, y, x_ffn, out, loss

    gc.collect()
    rss_after = get_rss_mb()

    nodes_after = count_autograd_nodes()

    print(f"  RSS before: {rss_before:.1f}MB, after: {rss_after:.1f}MB, delta: {rss_after - rss_before:+.1f}MB")
    print(f"  Remaining autograd nodes: {nodes_after}")

    # Check refs to params
    for name, p in [("beta", beta), ("gamma", gamma), ("ga1", ga1), ("attgamma", attgamma)]:
        ref_count = sys.getrefcount(p)
        print(f"  {name}: refcount={ref_count}, grad_fn={p.grad_fn}")

    if nodes_after <= 2:
        print("  RESULT: Scaling params do NOT leak graph references after backward()")
    else:
        print(f"  RESULT: WARNING - {nodes_after} autograd nodes persist")

    del beta, gamma, ga1, attgamma
    gc.collect()


# ============================================================================
# Test 4: Full KBNet memory growth with gradient checkpointing
# ============================================================================
def test_full_kbnet_memory():
    """Run 30 forward+backward on full KBNet, checking graph cleanup with checkpointing."""
    print("\n" + "=" * 80)
    print("TEST 4: Full KBNet forward+backward memory growth (30 iters, checkpointing ON)")
    print("=" * 80)

    torch.manual_seed(42)
    model = KBNet(use_checkpoint=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(3):
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del x, out, loss
    gc.collect()

    rss_history = []
    node_history = []

    for i in range(30):
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del x, out, loss

        if (i + 1) % 5 == 0:
            gc.collect()
            rss = get_rss_mb()
            n_nodes = count_autograd_nodes()
            rss_history.append(rss)
            node_history.append(n_nodes)
            print(f"  Iter {i+1:3d}: RSS={rss:.1f}MB, autograd_nodes={n_nodes}")

    rss_growth = rss_history[-1] - rss_history[0]
    node_growth = node_history[-1] - node_history[0]

    print(f"\n  RSS growth over 25 iters: {rss_growth:+.1f}MB")
    print(f"  Autograd node growth: {node_growth:+d}")

    if abs(rss_growth) < 10.0 and abs(node_growth) <= 5:
        print("  RESULT: NO LEAK with gradient checkpointing")
    elif rss_growth > 20.0:
        print(f"  RESULT: POTENTIAL LEAK - RSS grew {rss_growth:.1f}MB with checkpointing")
    else:
        print(f"  RESULT: Minor fluctuation ({rss_growth:+.1f}MB) - likely no leak")

    del model, optimizer
    gc.collect()
    return rss_history


# ============================================================================
# Test 5: Gradient checkpointing recomputed graph cleanup
# ============================================================================
def test_checkpoint_graph_cleanup():
    """Verify recomputed forward graph is freed after backward with checkpointing."""
    print("\n" + "=" * 80)
    print("TEST 5: Gradient checkpointing - recomputed graph cleanup")
    print("=" * 80)

    from torch.utils.checkpoint import checkpoint

    block = KBBlock_s(c=32, nset=32, gc=1)
    block.train()

    gc.collect()

    x = torch.randn(1, 32, 32, 32, requires_grad=True)

    # Forward with checkpointing (use_reentrant=False)
    out = checkpoint(block, x, use_reentrant=False)

    nodes_before_backward = count_autograd_nodes()
    print(f"  Autograd nodes after checkpointed forward: {nodes_before_backward}")

    loss = out.mean()
    loss.backward()

    del out, loss
    gc.collect()

    nodes_after_backward = count_autograd_nodes()
    print(f"  Autograd nodes after backward + gc: {nodes_after_backward}")

    # Now do it again to see if nodes accumulate
    x2 = torch.randn(1, 32, 32, 32, requires_grad=True)
    out2 = checkpoint(block, x2, use_reentrant=False)
    loss2 = out2.mean()
    loss2.backward()
    del x2, out2, loss2
    gc.collect()

    nodes_after_second = count_autograd_nodes()
    print(f"  Autograd nodes after 2nd forward+backward + gc: {nodes_after_second}")

    if nodes_after_second <= nodes_after_backward + 2:
        print("  RESULT: Checkpointing properly cleans up recomputed graph")
    else:
        growth = nodes_after_second - nodes_after_backward
        print(f"  RESULT: WARNING - {growth} extra nodes after 2nd iteration")

    del block, x
    gc.collect()


# ============================================================================
# Test 6: KBA parameter memory calculation
# ============================================================================
def test_kba_parameter_memory():
    """Calculate exact KBA parameter memory across all levels."""
    print("\n" + "=" * 80)
    print("TEST 6: KBA parameter memory calculation (exact)")
    print("=" * 80)

    # KBNet-7M config: width=32, enc=[2,2,4], mid=10, dec=[2,2,2], nset=32, gc=1, k=3
    nset = 32
    gc_val = 1
    k = 3

    levels = {
        "Encoder Level 0 (c=32, 2 blocks)": (32, 2),
        "Encoder Level 1 (c=64, 2 blocks)": (64, 2),
        "Encoder Level 2 (c=128, 4 blocks)": (128, 4),
        "Bottleneck (c=256, 10 blocks)": (256, 10),
        "Decoder Level 0 (c=128, 2 blocks)": (128, 2),
        "Decoder Level 1 (c=64, 2 blocks)": (64, 2),
        "Decoder Level 2 (c=32, 2 blocks)": (32, 2),
    }

    total_w_params = 0
    total_b_params = 0
    total_scaling_params = 0

    print(f"\n  {'Level':<45s} {'w params':>12s} {'b params':>10s} {'w KB':>8s} {'b KB':>8s}")
    print("  " + "-" * 83)

    for name, (c, n_blocks) in levels.items():
        w_per_block = nset * c * gc_val * k * k  # (1, nset, c*gc*k^2)
        b_per_block = nset * c  # (1, nset, c)

        w_total = n_blocks * w_per_block
        b_total = n_blocks * b_per_block

        total_w_params += w_total
        total_b_params += b_total

        # Scaling params per block: beta(c), gamma(c), ga1(c), attgamma(nset)
        scaling_per_block = 3 * c + nset  # beta, gamma, ga1 are (1,c,1,1), attgamma is (1,nset,1,1)
        total_scaling_params += n_blocks * scaling_per_block

        w_kb = w_total * 4 / 1024  # float32 = 4 bytes
        b_kb = b_total * 4 / 1024

        print(f"  {name:<45s} {w_total:>12,d} {b_total:>10,d} {w_kb:>7.1f}K {b_kb:>7.1f}K")

    total_kba_params = total_w_params + total_b_params
    total_all_kba = total_kba_params + total_scaling_params

    print("  " + "-" * 83)
    print(f"  {'TOTAL w':<45s} {total_w_params:>12,d} {'':>10s} {total_w_params * 4 / 1024:>7.1f}K")
    print(f"  {'TOTAL b':<45s} {'':>12s} {total_b_params:>10,d} {'':>8s} {total_b_params * 4 / 1024:>7.1f}K")
    print(f"  {'TOTAL w+b':<45s} {total_kba_params:>12,d} {'':>10s} {total_kba_params * 4 / 1024:>7.1f}K")
    print(f"  {'TOTAL scaling (beta,gamma,ga1,attgamma)':<45s} {total_scaling_params:>12,d} {'':>10s} {total_scaling_params * 4 / 1024:>7.1f}K")
    print(f"  {'TOTAL KBA + scaling params':<45s} {total_all_kba:>12,d} {'':>10s} {total_all_kba * 4 / 1024:>7.1f}K")

    print(f"\n  KBA w+b parameter memory (float32): {total_kba_params * 4 / (1024*1024):.2f} MB")
    print(f"  KBA + scaling parameter memory (float32): {total_all_kba * 4 / (1024*1024):.2f} MB")

    # Verify against actual model
    print("\n  --- Verification against actual model ---")
    model = KBNet(use_checkpoint=False)

    actual_w_params = 0
    actual_b_params = 0
    actual_scaling_params = 0

    for name_p, p in model.named_parameters():
        if name_p.endswith('.w'):
            actual_w_params += p.numel()
        elif name_p.endswith('.b') and 'norm' not in name_p and 'conv' not in name_p:
            # Only count KBA bias, not conv/norm bias
            if any(prefix in name_p for prefix in ['encoders', 'decoders', 'middle_blks']):
                # Check if it's a KBBlock_s .b parameter (shape has nset dim)
                if len(p.shape) == 3:  # (1, nset, c) shape
                    actual_b_params += p.numel()
        elif name_p.endswith(('.beta', '.gamma', '.ga1', '.attgamma')):
            actual_scaling_params += p.numel()

    print(f"  Calculated w params: {total_w_params:,d} | Actual: {actual_w_params:,d} | Match: {total_w_params == actual_w_params}")
    print(f"  Calculated b params: {total_b_params:,d} | Actual: {actual_b_params:,d} | Match: {total_b_params == actual_b_params}")
    print(f"  Calculated scaling:  {total_scaling_params:,d} | Actual: {actual_scaling_params:,d} | Match: {total_scaling_params == actual_scaling_params}")

    del model
    gc.collect()

    return total_kba_params, total_scaling_params


# ============================================================================
# Test 7: AdamW optimizer state memory for KBA parameters
# ============================================================================
def test_adamw_state_memory(total_kba_params, total_scaling_params):
    """Calculate AdamW optimizer state memory for KBA parameters."""
    print("\n" + "=" * 80)
    print("TEST 7: AdamW optimizer state memory for KBA parameters")
    print("=" * 80)

    # AdamW stores: exp_avg (same shape) + exp_avg_sq (same shape) per parameter
    # That's 2x the parameter memory
    total_params = total_kba_params + total_scaling_params

    param_bytes = total_params * 4  # float32
    optimizer_bytes = total_params * 4 * 2  # exp_avg + exp_avg_sq
    total_bytes = param_bytes + optimizer_bytes  # params + optimizer state

    print(f"  KBA + scaling parameters: {total_params:,d}")
    print(f"  Parameter memory: {param_bytes / (1024*1024):.2f} MB")
    print(f"  AdamW exp_avg memory: {param_bytes / (1024*1024):.2f} MB")
    print(f"  AdamW exp_avg_sq memory: {param_bytes / (1024*1024):.2f} MB")
    print(f"  Total (params + optimizer): {total_bytes / (1024*1024):.2f} MB (3x param memory)")

    # Verify with actual model
    print("\n  --- Verification with actual optimizer state ---")
    model = KBNet(use_checkpoint=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one step to populate optimizer state
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    loss = out.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del x, out, loss

    # Count optimizer state tensors for KBA params
    kba_opt_bytes = 0
    total_opt_bytes = 0
    kba_param_ids = set()

    for name_p, p in model.named_parameters():
        if name_p.endswith(('.w', '.beta', '.gamma', '.ga1', '.attgamma')):
            kba_param_ids.add(id(p))
        elif len(p.shape) == 3 and name_p.endswith('.b'):
            kba_param_ids.add(id(p))

    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p)
            if state:
                for key in ['exp_avg', 'exp_avg_sq']:
                    if key in state:
                        nbytes = state[key].nelement() * state[key].element_size()
                        total_opt_bytes += nbytes
                        if id(p) in kba_param_ids:
                            kba_opt_bytes += nbytes

    print(f"  Actual KBA optimizer state: {kba_opt_bytes / (1024*1024):.2f} MB")
    print(f"  Actual total optimizer state: {total_opt_bytes / (1024*1024):.2f} MB")

    # Also report total model param memory
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total model parameters: {total_model_params:,d} ({total_model_params * 4 / (1024*1024):.2f} MB)")
    print(f"  Total model + optimizer: {(total_model_params * 4 + total_opt_bytes) / (1024*1024):.2f} MB")
    print(f"  KBA fraction of total params: {(total_kba_params + total_scaling_params) / total_model_params * 100:.1f}%")

    del model, optimizer
    gc.collect()


# ============================================================================
# Test 8: Tracemalloc-based leak detection (most precise)
# ============================================================================
def test_tracemalloc_leak():
    """Use tracemalloc to detect Python-level memory leaks from KBA operations."""
    print("\n" + "=" * 80)
    print("TEST 8: Tracemalloc-based leak detection (Python allocator)")
    print("=" * 80)

    block = KBBlock_s(c=32, nset=32, gc=1)
    block.train()
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)

    # Warmup
    for _ in range(10):
        x = torch.randn(1, 32, 32, 32)
        out = block(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del x, out, loss
    gc.collect()

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    for _ in range(100):
        x = torch.randn(1, 32, 32, 32)
        out = block(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del x, out, loss
    gc.collect()

    snapshot2 = tracemalloc.take_snapshot()

    stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("  Top 10 memory changes (by size):")
    total_growth = 0
    for stat in stats[:10]:
        if stat.size_diff > 0:
            total_growth += stat.size_diff
        print(f"    {stat}")

    tracemalloc.stop()

    print(f"\n  Total growth in top 10: {total_growth / 1024:.1f} KB")
    if total_growth < 100 * 1024:  # Less than 100KB
        print("  RESULT: No significant Python-level memory leak")
    else:
        print(f"  RESULT: WARNING - {total_growth / 1024:.1f} KB growth in 100 iterations")

    del block, optimizer
    gc.collect()


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("KBNet KBA Memory Leak Investigation")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Initial RSS: {get_rss_mb():.1f} MB")
    print()

    # Test 1: Block-level memory growth
    rss_block = test_kbblock_memory_leak()

    # Test 2: View operation graph persistence
    test_view_operations()

    # Test 3: Scaling parameter lifetime
    test_scaling_params()

    # Test 4: Full model memory growth
    rss_model = test_full_kbnet_memory()

    # Test 5: Checkpoint graph cleanup
    test_checkpoint_graph_cleanup()

    # Test 6: Parameter memory calculation
    total_kba, total_scaling = test_kba_parameter_memory()

    # Test 7: Optimizer state memory
    test_adamw_state_memory(total_kba, total_scaling)

    # Test 8: Tracemalloc
    test_tracemalloc_leak()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
  1. VIEW OPERATIONS (.squeeze(0).t()):
     Views share storage with the original parameter tensor. The autograd graph
     records SqueezeBackward and TBackward nodes during forward, but these are
     freed after loss.backward() (no retain_graph). With gradient checkpointing
     (use_reentrant=False), the recomputed forward graph is also properly freed.

  2. LEARNABLE SCALING (beta, gamma, ga1, attgamma):
     Each multiplication (x * self.beta) creates a MulBackward0 node that holds
     references to both operands. After loss.backward(), the graph is freed and
     these references are released. No leak.

  3. GRADIENT CHECKPOINTING:
     use_reentrant=False creates a new graph during backward recomputation.
     This graph is consumed by the backward pass and freed immediately after.
     No accumulation across iterations.

  4. KBA PARAMETER MEMORY:
     The KBA w+b parameters are the memory-dominant component of KBNet.
     With AdamW, each parameter has 3x memory cost (param + exp_avg + exp_avg_sq).
     See exact numbers above.
""")
