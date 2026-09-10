#!/usr/bin/env python3
"""
Verify memory leaks for Baseline Models (SwinIR, DRUNet, U-Net).
"""
import sys
import os
import torch
import torch.nn as nn
import gc
import time
from pathlib import Path

# Add sota models to path
sys.path.append('sota/models')
sys.path.append(str(Path(__file__).resolve().parent)) # For nsnd_oct if needed

# Import models
try:
    from swinir_fair import SwinIR
    from drunet_fair import DRUNet
    # Adjust path for UNet if necessary based on previous reads
    from nsnd_oct.nsnd.models.unet import UNet
except ImportError as e:
    print(f"Error importing models: {e}")
    # Try alternate path for UNet just in case
    try:
        sys.path.append('nsnd_oct/nsnd/models')
        from unet import UNet
    except ImportError:
        pass

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0

def test_model_memory(model_name, model_fn, input_shape=(1, 1, 64, 64), steps=20):
    print(f"\nTesting {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache()
    base_mem = get_memory_mb()
    print(f"  Baseline Memory: {base_mem:.2f} MB")

    try:
        model = model_fn().to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Initial post-load memory
        model_mem = get_memory_mb()
        print(f"  Model Loaded Memory: {model_mem:.2f} MB (Delta: {model_mem - base_mem:.2f} MB)")
        
        # Dummy data
        x = torch.randn(*input_shape).to(device)
        y = torch.randn(*input_shape).to(device)
        
        # Noise level for DRUNet
        sigma = torch.FloatTensor([25.0/255.0]).to(device) if model_name == "DRUNet" else None

        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            if model_name == "DRUNet":
                pred = model(x, sigma)
            else:
                pred = model(x)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()
        start_loop_mem = get_memory_mb()
        print(f"  Start Loop Memory: {start_loop_mem:.2f} MB")

        mem_history = []
        for i in range(steps):
            optimizer.zero_grad()
            
            # Create new tensors every iteration to simulate data loader
            x_step = torch.randn(*input_shape).to(device)
            y_step = torch.randn(*input_shape).to(device)
            
            if model_name == "DRUNet":
                pred = model(x_step, sigma)
            else:
                pred = model(x_step)
                
            loss = nn.MSELoss()(pred, y_step)
            loss.backward()
            optimizer.step()
            
            current_mem = get_memory_mb()
            mem_history.append(current_mem)
            
            # Check for immediate explosion
            if i > 0 and current_mem - mem_history[0] > 100: # 100MB growth
                print(f"  ❌ Rapid memory growth detected at step {i}!")
                return False

        end_loop_mem = get_memory_mb()
        print(f"  End Loop Memory: {end_loop_mem:.2f} MB")
        
        growth = end_loop_mem - start_loop_mem
        print(f"  Loop Growth: {growth:.2f} MB")
        
        if growth > 5.0: # Tolerance threshold
            print(f"  ❌ Memory leak detected in {model_name}!")
            return False
        else:
            print(f"  ✓ {model_name} is memory stable.")
            return True

    except Exception as e:
        print(f"  ❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        del model, optimizer
        if 'x' in locals(): del x
        if 'y' in locals(): del y
        if 'pred' in locals(): del pred
        if 'loss' in locals(): del loss
        gc.collect()
        torch.cuda.empty_cache()

def main():
    print("="*60)
    print("Memory Leak Verification")
    print("="*60)
    
    # 1. SwinIR
    # Config matching command: embed_dim=184, depths=[8,8,8], num_heads=[4,4,4]
    def create_swinir():
        return SwinIR(
            img_size=64, patch_size=1, in_chans=1,
            embed_dim=184, depths=[8, 8, 8], num_heads=[4, 4, 4],
            window_size=8, mlp_ratio=2.
        )
    test_model_memory("SwinIR", create_swinir, input_shape=(4, 1, 64, 64))

    # 2. DRUNet
    # Config matching command: nc=[44,88,176,352], nb=[2,2,2,2]
    def create_drunet():
        return DRUNet(
            in_channels=1, out_channels=1,
            nc=[44, 88, 176, 352], nb=[2, 2, 2, 2],
            act_mode='R', use_noise_level=True, use_tanh=False
        )
    test_model_memory("DRUNet", create_drunet, input_shape=(4, 1, 64, 64))

    # 3. U-Net
    # Config matching command: features=32
    def create_unet():
        return UNet(in_channels=1, out_channels=1, features=32)
    test_model_memory("U-Net", create_unet, input_shape=(4, 1, 64, 64))

    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
