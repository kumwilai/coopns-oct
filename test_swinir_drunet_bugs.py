#!/usr/bin/env python3
"""
Hunt for bugs and memory leaks in SwinIR and DRUNet training scripts.
"""
import torch
import gc
import sys
sys.path.insert(0, '.')
sys.path.append('sota/models')

from swinir_fair import SwinIR
from drunet_fair import DRUNet
from adaptive_oct_denoise import (
    PairedOCTDataset,
    resize_to,
    compute_psnr,
    compute_ssim,
)
from torch.utils.data import DataLoader

print("=" * 80)
print("SWINIR & DRUNET BUG & MEMORY LEAK HUNTING")
print("=" * 80)

def get_memory():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return 0

issues = []

# Test 1: Model initialization
print("\n1. Testing model initialization...")

print("  Testing SwinIR...")
try:
    swinir = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=1,
        embed_dim=48,
        depths=[2, 2, 2],
        num_heads=[3, 3, 3],
        window_size=8,
        mlp_ratio=2.,
        upscale=1,
        img_range=1.,
        upsampler=None,
    )
    swinir_params = sum(p.numel() for p in swinir.parameters())
    print(f"    ✓ SwinIR initialized: {swinir_params:,} params ({swinir_params/1e6:.2f}M)")
except Exception as e:
    print(f"    ❌ SwinIR initialization failed: {e}")
    issues.append(f"SwinIR init failed: {e}")

print("  Testing DRUNet...")
try:
    drunet = DRUNet(
        in_channels=1,
        out_channels=1,
        nc=[64, 128, 256, 512],
        nb=[2, 2, 2, 2],
        act_mode='R',
        use_noise_level=True
    )
    drunet_params = sum(p.numel() for p in drunet.parameters())
    print(f"    ✓ DRUNet initialized: {drunet_params:,} params ({drunet_params/1e6:.2f}M)")
except Exception as e:
    print(f"    ❌ DRUNet initialization failed: {e}")
    issues.append(f"DRUNet init failed: {e}")

# Test 2: Forward pass
print("\n2. Testing forward pass...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_input = torch.randn(2, 1, 64, 64).to(device)

print("  Testing SwinIR forward...")
try:
    swinir = swinir.to(device)
    swinir.eval()
    with torch.no_grad():
        output = swinir(test_input)

    if output.shape == test_input.shape:
        print(f"    ✓ SwinIR forward pass: {output.shape}")
    else:
        print(f"    ❌ Shape mismatch: {output.shape} vs {test_input.shape}")
        issues.append(f"SwinIR shape mismatch")
except Exception as e:
    print(f"    ❌ SwinIR forward failed: {e}")
    issues.append(f"SwinIR forward failed: {e}")

print("  Testing DRUNet forward...")
try:
    drunet = drunet.to(device)
    drunet.eval()

    # Test without noise level
    with torch.no_grad():
        output_no_noise = drunet(test_input, noise_level=None)

    if output_no_noise.shape == test_input.shape:
        print(f"    ✓ DRUNet forward (no noise level): {output_no_noise.shape}")
    else:
        print(f"    ❌ Shape mismatch: {output_no_noise.shape} vs {test_input.shape}")
        issues.append(f"DRUNet shape mismatch (no noise)")

    # Test with noise level
    noise_level = torch.FloatTensor([25.0 / 255.0]).to(device)
    with torch.no_grad():
        output_with_noise = drunet(test_input, noise_level)

    if output_with_noise.shape == test_input.shape:
        print(f"    ✓ DRUNet forward (with noise level): {output_with_noise.shape}")
    else:
        print(f"    ❌ Shape mismatch: {output_with_noise.shape} vs {test_input.shape}")
        issues.append(f"DRUNet shape mismatch (with noise)")

except Exception as e:
    print(f"    ❌ DRUNet forward failed: {e}")
    issues.append(f"DRUNet forward failed: {e}")

# Test 3: Gradient flow
print("\n3. Testing gradient flow...")

print("  Testing SwinIR gradients...")
try:
    swinir.train()
    test_input_grad = torch.randn(2, 1, 64, 64, requires_grad=False).to(device)
    test_target = torch.randn(2, 1, 64, 64).to(device)

    optimizer = torch.optim.Adam(swinir.parameters(), lr=1e-4)
    optimizer.zero_grad()

    output = swinir(test_input_grad)
    loss = torch.nn.functional.mse_loss(output, test_target)
    loss.backward()

    has_grad = False
    for param in swinir.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    if has_grad:
        print(f"    ✓ SwinIR gradients flow correctly")
    else:
        print(f"    ❌ No gradients in SwinIR!")
        issues.append("SwinIR no gradients")

except Exception as e:
    print(f"    ❌ SwinIR gradient test failed: {e}")
    issues.append(f"SwinIR gradient failed: {e}")

print("  Testing DRUNet gradients...")
try:
    drunet.train()
    test_input_grad = torch.randn(2, 1, 64, 64, requires_grad=False).to(device)
    test_target = torch.randn(2, 1, 64, 64).to(device)

    optimizer = torch.optim.Adam(drunet.parameters(), lr=1e-4)
    optimizer.zero_grad()

    noise_level = torch.FloatTensor([25.0 / 255.0]).to(device)
    output = drunet(test_input_grad, noise_level)
    loss = torch.nn.functional.mse_loss(output, test_target)
    loss.backward()

    has_grad = False
    for param in drunet.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    if has_grad:
        print(f"    ✓ DRUNet gradients flow correctly")
    else:
        print(f"    ❌ No gradients in DRUNet!")
        issues.append("DRUNet no gradients")

except Exception as e:
    print(f"    ❌ DRUNet gradient test failed: {e}")
    issues.append(f"DRUNet gradient failed: {e}")

# Test 4: Memory leak test
print("\n4. Testing for memory leaks...")

initial_mem = get_memory()
print(f"  Initial memory: {initial_mem:.1f} MB")

print("  Testing SwinIR memory...")
swinir_optimizer = torch.optim.Adam(swinir.parameters(), lr=1e-4)
mem_per_iter_swinir = []
for i in range(10):
    test_in = torch.randn(4, 1, 64, 64).to(device)
    test_tgt = torch.randn(4, 1, 64, 64).to(device)

    swinir_optimizer.zero_grad()
    output = swinir(test_in)
    loss = torch.nn.functional.mse_loss(output, test_tgt)
    loss.backward()
    swinir_optimizer.step()

    del test_in, test_tgt, output, loss

    if i % 3 == 0:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mem_per_iter_swinir.append(get_memory())

swinir_mem_growth = mem_per_iter_swinir[-1] - initial_mem if mem_per_iter_swinir else 0
print(f"  SwinIR memory growth: {swinir_mem_growth:.1f} MB")

if swinir_mem_growth > 50:
    print(f"    ❌ SwinIR memory leak detected!")
    issues.append(f"SwinIR memory leak: {swinir_mem_growth:.1f} MB")
else:
    print(f"    ✓ No significant SwinIR memory leak")

del swinir_optimizer

gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("  Testing DRUNet memory...")
drunet_optimizer = torch.optim.Adam(drunet.parameters(), lr=1e-4)
mem_per_iter_drunet = []
for i in range(10):
    test_in = torch.randn(4, 1, 64, 64).to(device)
    test_tgt = torch.randn(4, 1, 64, 64).to(device)

    drunet_optimizer.zero_grad()
    noise_level = torch.FloatTensor([25.0 / 255.0]).to(device)
    output = drunet(test_in, noise_level)
    loss = torch.nn.functional.mse_loss(output, test_tgt)
    loss.backward()
    drunet_optimizer.step()

    del test_in, test_tgt, output, loss

    if i % 3 == 0:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mem_per_iter_drunet.append(get_memory())

drunet_mem_growth = mem_per_iter_drunet[-1] - initial_mem if mem_per_iter_drunet else 0
print(f"  DRUNet memory growth: {drunet_mem_growth:.1f} MB")

if drunet_mem_growth > 50:
    print(f"    ❌ DRUNet memory leak detected!")
    issues.append(f"DRUNet memory leak: {drunet_mem_growth:.1f} MB")
else:
    print(f"    ✓ No significant DRUNet memory leak")

del drunet_optimizer

# Test 5: Integration with data loader
print("\n5. Testing integration with data loader...")

transform = resize_to((64, 64))
try:
    dataset = PairedOCTDataset('train_pairs_universal.txt', transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
except:
    print("  ⚠️ Could not load dataset, using dummy data")
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx):
            return torch.randn(1, 64, 64), torch.randn(1, 64, 64)
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2)

print("  Testing SwinIR with data loader...")
try:
    swinir.train()
    for i, (noisy, clean) in enumerate(loader):
        if i >= 3:
            break

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        output = swinir(noisy)
        loss = torch.nn.functional.mse_loss(output, clean)
        loss.backward()
        optimizer.step()

    print(f"    ✓ SwinIR training loop works")
except Exception as e:
    print(f"    ❌ SwinIR training loop failed: {e}")
    issues.append(f"SwinIR training loop: {e}")

print("  Testing DRUNet with data loader...")
try:
    drunet.train()
    for i, (noisy, clean) in enumerate(loader):
        if i >= 3:
            break

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        noise_level = torch.FloatTensor([25.0 / 255.0]).to(device)
        output = drunet(noisy, noise_level)
        loss = torch.nn.functional.mse_loss(output, clean)
        loss.backward()
        optimizer.step()

    print(f"    ✓ DRUNet training loop works")
except Exception as e:
    print(f"    ❌ DRUNet training loop failed: {e}")
    issues.append(f"DRUNet training loop: {e}")

# Test 6: PSNR/SSIM computation
print("\n6. Testing PSNR/SSIM computation...")

try:
    pred = torch.rand(1, 1, 64, 64)
    target = torch.rand(1, 1, 64, 64)

    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)

    if torch.isfinite(torch.tensor(psnr)) and torch.isfinite(torch.tensor(ssim)):
        print(f"  ✓ PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    else:
        print(f"  ❌ Invalid metrics: PSNR={psnr}, SSIM={ssim}")
        issues.append("Invalid PSNR/SSIM")
except Exception as e:
    print(f"  ❌ Metric computation failed: {e}")
    issues.append(f"Metrics failed: {e}")

# Test 7: Edge cases
print("\n7. Testing edge cases...")

# Small batch size
print("  Testing batch_size=1...")
try:
    single_input = torch.randn(1, 1, 64, 64).to(device)
    swinir_out = swinir(single_input)
    drunet_out = drunet(single_input, torch.FloatTensor([25.0/255.0]).to(device))
    print(f"    ✓ Batch size 1 works")
except Exception as e:
    print(f"    ❌ Batch size 1 failed: {e}")
    issues.append(f"Batch size 1: {e}")

# Different image sizes
print("  Testing different image sizes...")
for size in [32, 128]:
    try:
        test_size_input = torch.randn(2, 1, size, size).to(device)

        # SwinIR may require specific sizes (window_size divisible)
        if size % 8 == 0:  # window_size=8
            swinir_test = SwinIR(img_size=size, patch_size=1, in_chans=1, embed_dim=48,
                                depths=[2,2,2], num_heads=[3,3,3], window_size=8).to(device)
            swinir_test.eval()
            with torch.no_grad():
                out = swinir_test(test_size_input)
            print(f"    ✓ SwinIR size {size}x{size} works")

        # DRUNet should work with any size
        drunet_test = DRUNet(nc=[64,128,256,512], nb=[2,2,2,2]).to(device)
        drunet_test.eval()
        with torch.no_grad():
            out = drunet_test(test_size_input, torch.FloatTensor([25.0/255.0]).to(device))
        print(f"    ✓ DRUNet size {size}x{size} works")

    except Exception as e:
        print(f"    ⚠️ Size {size}x{size}: {e}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)

if len(issues) == 0:
    print("✓✓ NO BUGS OR MEMORY LEAKS DETECTED!")
    print("\n✓ Safe to proceed with training")

    print("\nRecommended training commands:")
    print("\n1. SwinIR:")
    print("  python train_swinir_monitored.py \\")
    print("    --train_pairs train_pairs_universal.txt \\")
    print("    --val_pairs val_pairs_universal.txt \\")
    print("    --out_dir checkpoints/swinir_fair_universal \\")
    print("    --epochs 100 \\")
    print("    --batch_size 8 \\")
    print("    --lr 5e-4 \\")
    print("    --size 64 \\")
    print("    --grad_w 0.0")

    print("\n2. DRUNet:")
    print("  python train_drunet_monitored.py \\")
    print("    --train_pairs train_pairs_universal.txt \\")
    print("    --val_pairs val_pairs_universal.txt \\")
    print("    --out_dir checkpoints/drunet_fair_universal \\")
    print("    --epochs 100 \\")
    print("    --batch_size 8 \\")
    print("    --lr 5e-4 \\")
    print("    --size 64 \\")
    print("    --grad_w 0.0")
else:
    print(f"❌ FOUND {len(issues)} ISSUE(S):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n⚠️ FIX THESE ISSUES BEFORE TRAINING!")

print("=" * 80)
