#!/usr/bin/env python3
"""
Compare parameter counts for baseline models to ensure fair comparison.
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add sota models to path
sys.path.append('sota/models')

# Import models
try:
    from nafnet_fair import NAFNet
    from swinir_fair import SwinIR
    from drunet_fair import DRUNet
    # U-Net might be in a different location, checking relative path
    ROOT = Path(__file__).resolve().parent
    sys.path.append(str(ROOT / 'nsnd_oct' / 'nsnd' / 'models'))
    from nsnd_oct.nsnd.models.unet import UNet
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Trying alternative import for U-Net...")
    try:
        from nsnd_oct.nsnd.models.unet import UNet
    except ImportError:
        pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("=" * 80)
    print("Model Parameter Comparison")
    print("=" * 80)
    
    models = []
    
    # 1. NAFNet
    # Config: --width 64 --middle_blk_num 2 --enc_blk_nums [2, 2, 2] --dec_blk_nums [2, 2, 2]
    try:
        nafnet = NAFNet(
            img_channel=1,
            width=64,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2],
            dec_blk_nums=[2, 2, 2]
        )
        params = count_parameters(nafnet)
        models.append(("NAFNet", params, "width=64, mid=2, enc/dec=[2,2,2]"))
    except Exception as e:
        print(f"Failed to init NAFNet: {e}")

    # 2. SwinIR
    # Config: --embed_dim 184 --depths 8,8,8 --num_heads 4,4,4 --window_size 8
    try:
        swinir = SwinIR(
            img_size=64,
            patch_size=1,
            in_chans=1,
            embed_dim=184,
            depths=[8, 8, 8],
            num_heads=[4, 4, 4],
            window_size=8,
            mlp_ratio=2.,
            upscale=1,
            img_range=1.,
            upsampler=None
        )
        params = count_parameters(swinir)
        models.append(("SwinIR", params, "dim=184, depth=[8,8,8], heads=[4,4,4]"))
    except Exception as e:
        print(f"Failed to init SwinIR: {e}")

    # 3. DRUNet
    # Config: --nc 44,88,176,352 --nb 2,2,2,2
    try:
        drunet = DRUNet(
            in_channels=1,
            out_channels=1,
            nc=[44, 88, 176, 352],
            nb=[2, 2, 2, 2],
            act_mode='R',
            use_noise_level=True,
            use_tanh=False # Checked in train_drunet command: default is False
        )
        params = count_parameters(drunet)
        models.append(("DRUNet", params, "nc=[44,88,176,352], nb=[2,2,2,2]"))
    except Exception as e:
        print(f"Failed to init DRUNet: {e}")

    # 4. U-Net
    # Config: --features 32
    try:
        unet = UNet(
            in_channels=1,
            out_channels=1,
            features=32
        )
        params = count_parameters(unet)
        models.append(("U-Net", params, "features=32"))
    except Exception as e:
        print(f"Failed to init U-Net: {e}")

    # Sort by params
    models.sort(key=lambda x: x[1])

    print(f"{'Model':<15} | {'Params':<12} | {'Config':<40}")
    print("-" * 80)
    for name, params, config in models:
        print(f"{name:<15} | {params/1e6:6.2f}M      | {config}")
    print("=" * 80)

if __name__ == "__main__":
    main()
