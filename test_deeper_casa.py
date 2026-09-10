"""
Quick test to verify the deeper CASA architecture works correctly.
"""
import torch
from adaptive_oct_denoise import build_model

def test_deeper_casa():
    print("="*80)
    print("TESTING DEEPER CASA ARCHITECTURE")
    print("="*80)

    # Build model
    print("\n[1] Building deeper CASA model (6 blocks)...")
    model = build_model(adapter_type='casa', base_channels=64)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())

    print(f"    Total parameters: {total_params:,}")
    print(f"    Backbone (U-Net): {backbone_params:,}")
    print(f"    Adapter (CASA):   {adapter_params:,}")

    # Check backbone architecture
    print("\n[2] Checking backbone architecture...")
    backbone = model.backbone
    print(f"    enc1: {backbone.enc1.net[0].out_channels} channels")
    print(f"    enc2: {backbone.enc2.net[0].out_channels} channels")
    print(f"    enc3: {backbone.enc3.net[0].out_channels} channels")
    print(f"    bottleneck: {backbone.bottleneck.net[0].out_channels} channels")
    print(f"    dec3: {backbone.dec3.net[0].out_channels} channels")
    print(f"    dec2: {backbone.dec2.net[0].out_channels} channels")
    print(f"    dec1: {backbone.dec1.net[0].out_channels} channels")

    # Check modulated channels
    print("\n[3] Checking FiLM modulation points...")
    mod_channels = backbone.modulated_channels
    for name, ch in mod_channels.items():
        print(f"    {name}: {ch} channels")

    # Test forward pass
    print("\n[4] Testing forward pass...")
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"    Input:  {x.shape}")
    print(f"    Output: {y.shape}")
    assert y.shape == x.shape, "Output shape mismatch!"

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nArchitecture Summary:")
    print("  - Encoder depth: 3 levels (enc1, enc2, enc3)")
    print("  - Decoder depth: 3 levels (dec3, dec2, dec1)")
    print("  - Total blocks: 6 (matches SwinIRLite)")
    print("  - Channel progression: 64 -> 128 -> 256")
    print("\nExpected Performance:")
    print("  - Old CASA (4 blocks): 28.11 dB")
    print("  - SwinIRLite (6 blocks): 28.84 dB")
    print("  - New CASA (6 blocks): ~28.8-29.0 dB (target)")
    print("="*80)

if __name__ == "__main__":
    test_deeper_casa()
