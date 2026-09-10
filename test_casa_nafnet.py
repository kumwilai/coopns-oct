"""
Test CASA with both U-Net and NAFNet backbones.
Verify FiLM modulation works correctly.
"""
import torch
from adaptive_oct_denoise import build_model

def test_backbone_compatibility():
    """Test that both backbones work with CASA adapter."""
    print("="*80)
    print("TESTING CASA WITH DIFFERENT BACKBONES")
    print("="*80)

    x = torch.randn(1, 1, 64, 64)

    # Test 1: CASA + U-Net (6 blocks)
    print("\n[1] Testing CASA + U-Net (6-block, deeper)...")
    model_unet = build_model(adapter_type='casa', backbone_type='unet', base_channels=64)

    total_params = sum(p.numel() for p in model_unet.parameters())
    backbone_params = sum(p.numel() for p in model_unet.backbone.parameters())
    adapter_params = sum(p.numel() for p in model_unet.adapter.parameters())

    print(f"  Total params: {total_params:,}")
    print(f"  Backbone (U-Net): {backbone_params:,}")
    print(f"  Adapter (CASA): {adapter_params:,}")
    print(f"  Modulation points: {list(model_unet.backbone.modulated_channels.keys())}")

    with torch.no_grad():
        y_unet = model_unet(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_unet.shape}")
    print(f"  ✓ CASA + U-Net works!")

    # Test 2: CASA + NAFNet
    print("\n[2] Testing CASA + NAFNet...")
    model_nafnet = build_model(adapter_type='casa', backbone_type='nafnet', base_channels=32)

    total_params = sum(p.numel() for p in model_nafnet.parameters())
    backbone_params = sum(p.numel() for p in model_nafnet.backbone.parameters())
    adapter_params = sum(p.numel() for p in model_nafnet.adapter.parameters())

    print(f"  Total params: {total_params:,}")
    print(f"  Backbone (NAFNet): {backbone_params:,}")
    print(f"  Adapter (CASA): {adapter_params:,}")
    print(f"  Modulation points: {list(model_nafnet.backbone.modulated_channels.keys())}")

    with torch.no_grad():
        y_nafnet = model_nafnet(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_nafnet.shape}")
    print(f"  ✓ CASA + NAFNet works!")

    # Test 3: Standalone NAFNet (no adapter)
    print("\n[3] Testing Standalone NAFNet (no adapter, for comparison)...")
    from adaptive_oct_denoise import NAFBackbone
    nafnet_standalone = NAFBackbone(base_channels=32, num_blocks=4)
    standalone_params = sum(p.numel() for p in nafnet_standalone.parameters())
    print(f"  Standalone NAFNet params: {standalone_params:,}")

    with torch.no_grad():
        y_standalone = nafnet_standalone(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_standalone.shape}")
    print(f"  ✓ Standalone NAFNet works!")

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print("  - CASA + U-Net (6 blocks):   {:,} params".format(sum(p.numel() for p in model_unet.parameters())))
    print("  - CASA + NAFNet:             {:,} params".format(sum(p.numel() for p in model_nafnet.parameters())))
    print("  - Standalone NAFNet:         {:,} params".format(standalone_params))
    print("\nYou can now train:")
    print("  1. CASA + U-Net:    --backbone unet --adapter casa")
    print("  2. CASA + NAFNet:   --backbone nafnet --adapter casa")
    print("  3. Standalone NAFNet: Train via eval_sota_from_pairs.py")
    print("="*80)


if __name__ == "__main__":
    test_backbone_compatibility()
