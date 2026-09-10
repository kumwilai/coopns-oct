#!/usr/bin/env python3
"""Quick memory leak test for anatomy-aware denoising fixes."""

import gc
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, './nsnd_oct')

def get_memory_mb():
    """Get current RAM usage in MB (works without psutil)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0

def test_memory_leak():
    print("=" * 60)
    print("MEMORY LEAK TEST - Anatomy-Aware NSAD")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Import after setting path
    from nsnd.models.sansd import AnatomyAwareSANSD
    from nsnd.models.anatomy_aware import AnatomyPreservingLoss

    # Create model with small config
    print("\nCreating model...")
    model = AnatomyAwareSANSD(
        backbone_width=32,  # Smaller for testing
        backbone_ckpt=None,  # No pretrained weights
        fusion_mode='gated',
        num_layer_zones=5,
        use_depth_adaptive=True,
        use_anatomy_fusion=False,  # Simpler for testing
    ).to(device)

    anatomy_loss = AnatomyPreservingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    # Warm-up iterations (memory will grow during this)
    batch_size = 2
    patch_size = 64
    warmup_iters = 10
    test_iters = 50  # Test for longer after warm-up

    print(f"\nWarm-up phase ({warmup_iters} iterations)...")
    for i in range(warmup_iters):
        noisy = torch.randn(batch_size, 1, patch_size, patch_size, device=device)
        clean = torch.randn(batch_size, 1, patch_size, patch_size, device=device)
        optimizer.zero_grad()
        denoised, interpretation = model(noisy, return_interpretation=True)
        backbone_out = interpretation['backbone_out']
        loss = ((denoised - clean).pow(2)).mean()
        del interpretation
        loss.backward()
        optimizer.step()
        del noisy, clean, denoised, backbone_out, loss
        gc.collect()

    # Measure memory AFTER warm-up
    gc.collect()
    warmup_mem = get_memory_mb()
    print(f"Memory after warm-up: {warmup_mem:.1f} MB")

    print(f"\nTesting for memory leak ({test_iters} iterations)...")
    print("-" * 60)

    memory_per_iter = []

    for i in range(test_iters):
        # Create dummy data
        noisy = torch.randn(batch_size, 1, patch_size, patch_size, device=device)
        clean = torch.randn(batch_size, 1, patch_size, patch_size, device=device)

        optimizer.zero_grad()

        # Forward pass with interpretation (the problematic call)
        denoised, interpretation = model(noisy, return_interpretation=True)

        # Compute losses like in training
        backbone_out = interpretation['backbone_out']
        backbone_error = (backbone_out - clean).abs()
        error_weight = 1.0 + 2.0 * backbone_error
        weighted_recon = ((denoised - clean).pow(2) * error_weight).mean()

        # MEMORY FIX: Delete intermediate tensors early
        del backbone_error, error_weight

        # Anatomy loss
        anatomy_total, anatomy_dict = anatomy_loss(denoised, clean, noisy)
        anatomy_loss_val = anatomy_total - anatomy_dict['recon']
        del anatomy_dict

        # Expert diversity (simplified)
        expert_outputs = interpretation['expert_outputs']
        outputs = list(expert_outputs.values())
        div_loss = torch.tensor(0.0, device=device)
        for j in range(len(outputs)):
            for k in range(j+1, len(outputs)):
                div_loss = div_loss - (outputs[j] - outputs[k]).abs().mean()
        del expert_outputs, outputs

        # Usage loss
        noise_type = interpretation['noise_type']
        usage_loss = torch.relu(0.05 - noise_type.mean(dim=[0, 2, 3])).sum()
        del noise_type

        # CRITICAL: Delete interpretation dict
        del interpretation

        # Total loss
        total = weighted_recon + 0.1 * anatomy_loss_val + 0.1 * div_loss + 0.1 * usage_loss

        # Backward
        total.backward()
        optimizer.step()

        # MEMORY FIX: Delete all tensors
        del noisy, clean, denoised, backbone_out, weighted_recon
        del anatomy_total, anatomy_loss_val, div_loss, usage_loss, total

        # Periodic cache clear
        if (i + 1) % 5 == 0:
            gc.collect()

        # Track memory AFTER cleanup
        iter_mem = get_memory_mb()
        memory_per_iter.append(iter_mem)

        if (i + 1) % 10 == 0:
            growth = iter_mem - warmup_mem
            print(f"  Iter {i+1:3d}: RAM = {iter_mem:.1f} MB (since warmup: {growth:+.1f} MB)")

    # Final memory
    gc.collect()
    final_mem = get_memory_mb()
    peak_mem = max(memory_per_iter)
    mem_growth = final_mem - warmup_mem

    # Check trend after warm-up (is memory consistently growing?)
    first_quarter = memory_per_iter[:12]
    last_quarter = memory_per_iter[-12:]
    trend = (sum(last_quarter) / len(last_quarter)) - (sum(first_quarter) / len(first_quarter))

    print("-" * 60)
    print(f"\nRESULTS (after warm-up):")
    print(f"  Post-warmup RAM: {warmup_mem:.1f} MB")
    print(f"  Peak RAM:        {peak_mem:.1f} MB")
    print(f"  Final RAM:       {final_mem:.1f} MB")
    print(f"  Growth:          {mem_growth:+.1f} MB")
    print(f"  Trend:           {trend:+.1f} MB")

    # Check for leak - after warm-up, growth should be minimal
    print("\n" + "=" * 60)
    if mem_growth > 50 or trend > 30:
        print("❌ MEMORY LEAK DETECTED!")
        print(f"   Growth: {mem_growth:.1f} MB, Trend: {trend:.1f} MB")
        return False
    elif mem_growth > 20 or trend > 10:
        print("⚠️  Minor memory growth - acceptable")
        return True
    else:
        print("✅ NO MEMORY LEAK! Memory is stable after warm-up.")
        return True

if __name__ == '__main__':
    try:
        success = test_memory_leak()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
