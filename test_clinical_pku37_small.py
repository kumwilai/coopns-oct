#!/usr/bin/env python3
"""
Quick test of Clinical Layer-Specific Denoising on PKU37.
Uses small images (64x64) to prevent OOM and get fast results.
"""

import sys
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/kumwilai/OCT')

# =============================================================================
# Memory-Efficient Layer-Specific Denoiser (Lightweight Version)
# =============================================================================

class LightLayerHead(nn.Module):
    """Lightweight layer denoiser head."""
    def __init__(self, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        # Initialize to near-zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)  # Residual


class LightClinicalDenoiser(nn.Module):
    """Memory-efficient clinical layer denoiser."""

    def __init__(self, width: int = 16):
        super().__init__()
        # 4 lightweight heads
        self.heads = nn.ModuleList([LightLayerHead(width) for _ in range(4)])

        # Sobel for edge loss
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

    def create_soft_masks(self, boundaries, H):
        """Create soft masks from boundaries."""
        B, N, W = boundaries.shape
        device = boundaries.device

        boundaries_px = boundaries * (H - 1)
        y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        b_exp = boundaries_px.unsqueeze(2)

        temp = 5.0
        b1, b2, b3 = b_exp[:, 1:2], b_exp[:, 2:3], b_exp[:, 3:4]

        mask_0 = torch.sigmoid((b1 - y) / temp)
        mask_1 = torch.sigmoid((y - b1) / temp) * torch.sigmoid((b2 - y) / temp)
        mask_2 = torch.sigmoid((y - b2) / temp) * torch.sigmoid((b3 - y) / temp)
        mask_3 = torch.sigmoid((y - b3) / temp)

        soft_masks = torch.cat([mask_0, mask_1, mask_2, mask_3], dim=1)
        return soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, noisy, boundaries):
        B, C, H, W = noisy.shape
        soft_masks = self.create_soft_masks(boundaries, H)

        # Process layers ONE AT A TIME to save memory
        denoised = torch.zeros_like(noisy)
        for i, head in enumerate(self.heads):
            layer_out = head(noisy)
            mask = soft_masks[:, i:i+1, :, :]
            denoised = denoised + layer_out * mask
            del layer_out  # Free immediately

        return denoised, soft_masks

    def compute_layer_losses(self, denoised, clean, soft_masks):
        """Compute per-layer losses."""
        losses = {}
        layer_psnrs = []

        for i in range(4):
            mask = soft_masks[:, i:i+1, :, :]
            mask_sum = mask.sum().clamp(min=1.0)

            # Per-layer L1
            layer_l1 = (torch.abs(denoised - clean) * mask).sum() / mask_sum
            losses[f'layer{i}_l1'] = layer_l1.item()

            # Per-layer PSNR
            layer_mse = ((denoised - clean) ** 2 * mask).sum() / mask_sum
            layer_psnr = 10 * np.log10(1.0 / max(layer_mse.item(), 1e-10))
            layer_psnrs.append(layer_psnr)
            losses[f'layer{i}_psnr'] = layer_psnr

        losses['avg_layer_psnr'] = np.mean(layer_psnrs)
        return losses


class SimpleDenoiser(nn.Module):
    """Simple global denoiser (baseline)."""
    def __init__(self, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SimpleBoundaryModel(nn.Module):
    """Simple boundary predictor."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 1),
        )
        self.bias = nn.Parameter(torch.tensor([0.25, 0.40, 0.55, 0.70]))

    def forward(self, x):
        B, C, H, W = x.shape
        offsets = self.conv(x).mean(dim=2) * 0.1
        boundaries = self.bias.view(1, -1, 1) + offsets
        # Enforce ordering
        b0 = boundaries[:, 0:1, :]
        b1 = torch.maximum(boundaries[:, 1:2, :], b0 + 0.05)
        b2 = torch.maximum(boundaries[:, 2:3, :], b1 + 0.05)
        b3 = torch.maximum(boundaries[:, 3:4, :], b2 + 0.05)
        return torch.clamp(torch.cat([b0, b1, b2, b3], dim=1), 0.05, 0.95)


# =============================================================================
# Data Loading
# =============================================================================

def load_pku37_small(pku37_root, max_images=10, target_size=64):
    """Load PKU37 with small size."""
    clean_dir = os.path.join(pku37_root, 'clean')
    noisy_dir = os.path.join(pku37_root, 'noisy')

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.tif')])[:max_images]
    pairs = []

    for fname in clean_files:
        clean_img = Image.open(os.path.join(clean_dir, fname))
        clean_np = np.array(clean_img, dtype=np.float32)
        if clean_np.max() > 1:
            clean_np /= 255.0

        noisy_path = os.path.join(noisy_dir, fname)
        if os.path.exists(noisy_path):
            noisy_img = Image.open(noisy_path)
            noisy_np = np.array(noisy_img, dtype=np.float32)
            if noisy_np.max() > 1:
                noisy_np /= 255.0
        else:
            noisy_np = np.clip(clean_np + np.random.randn(*clean_np.shape).astype(np.float32) * 0.1, 0, 1)

        # Resize to small
        clean_pil = Image.fromarray((clean_np * 255).astype(np.uint8))
        noisy_pil = Image.fromarray((noisy_np * 255).astype(np.uint8))
        clean_np = np.array(clean_pil.resize((target_size, target_size), Image.BILINEAR), dtype=np.float32) / 255.0
        noisy_np = np.array(noisy_pil.resize((target_size, target_size), Image.BILINEAR), dtype=np.float32) / 255.0

        pairs.append({
            'clean': torch.from_numpy(clean_np).unsqueeze(0),
            'noisy': torch.from_numpy(noisy_np).unsqueeze(0),
        })

    return pairs


# =============================================================================
# Training and Evaluation
# =============================================================================

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def train_global_denoiser(pairs, epochs=30, lr=0.001):
    """Train simple global denoiser (baseline)."""
    torch.manual_seed(42)
    model = SimpleDenoiser(width=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = int(len(pairs) * 0.7)
    train_pairs = pairs[:n_train]

    model.train()
    for epoch in range(epochs):
        for pair in train_pairs:
            noisy = pair['noisy'].unsqueeze(0)
            clean = pair['clean'].unsqueeze(0)
            optimizer.zero_grad()
            denoised = model(noisy)
            loss = F.l1_loss(denoised, clean)
            loss.backward()
            optimizer.step()
        gc.collect()

    return model


def train_clinical_denoiser(pairs, use_v4=True, epochs=30, lr=0.001):
    """Train clinical layer-specific denoiser."""
    torch.manual_seed(42)

    model = LightClinicalDenoiser(width=16)
    boundary_model = SimpleBoundaryModel()

    # V4 for boundary anchoring
    if use_v4:
        from intensity_anchor_v4 import IntensityAnchoredBoundaryLossV4
        v4_loss_fn = IntensityAnchoredBoundaryLossV4()

    all_params = list(model.parameters()) + list(boundary_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    n_train = int(len(pairs) * 0.7)
    train_pairs = pairs[:n_train]

    model.train()
    boundary_model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for pair in train_pairs:
            noisy = pair['noisy'].unsqueeze(0)
            clean = pair['clean'].unsqueeze(0)

            optimizer.zero_grad()

            # Predict boundaries
            boundaries = boundary_model(noisy)

            # Denoise with layer-specific heads
            denoised, soft_masks = model(noisy, boundaries)

            # Global loss
            loss = F.l1_loss(denoised, clean)

            # Per-layer loss (weighted by clinical importance)
            clinical_weights = [1.5, 1.0, 1.2, 1.5]
            for i in range(4):
                mask = soft_masks[:, i:i+1, :, :]
                layer_l1 = (torch.abs(denoised - clean) * mask).sum() / (mask.sum() + 1e-8)
                loss = loss + 0.3 * clinical_weights[i] * layer_l1

            # V4 boundary anchoring
            if use_v4:
                v4_loss, _ = v4_loss_fn(boundaries, noisy)
                loss = loss + 0.1 * v4_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        gc.collect()

    return model, boundary_model


def evaluate(model, pairs, boundary_model=None, is_clinical=False):
    """Evaluate model."""
    n_train = int(len(pairs) * 0.7)
    test_pairs = pairs[n_train:]

    model.eval()
    if boundary_model:
        boundary_model.eval()

    results = {'global_psnr': [], 'layer_psnrs': [[] for _ in range(4)]}

    # V4 for reference boundaries
    from intensity_anchor_v4 import IntensityAnchoredBoundaryLossV4
    v4 = IntensityAnchoredBoundaryLossV4()

    with torch.no_grad():
        for pair in test_pairs:
            noisy = pair['noisy'].unsqueeze(0)
            clean = pair['clean'].unsqueeze(0)
            H = noisy.shape[2]

            if is_clinical and boundary_model:
                boundaries = boundary_model(noisy)
                denoised, soft_masks = model(noisy, boundaries)
            else:
                denoised = model(noisy)
                # Use V4-detected boundaries for evaluation
                ref_top, ref_bottom, _ = v4.detect_retina_band(clean)
                boundaries = torch.zeros(1, 4, noisy.shape[3])
                for w in range(noisy.shape[3]):
                    t, b = ref_top[0, w].item(), ref_bottom[0, w].item()
                    boundaries[0, 0, w] = t
                    boundaries[0, 1, w] = t + (b - t) * 0.33
                    boundaries[0, 2, w] = t + (b - t) * 0.66
                    boundaries[0, 3, w] = b
                soft_masks = model.create_soft_masks(boundaries, H) if hasattr(model, 'create_soft_masks') else None

            # Global PSNR
            results['global_psnr'].append(compute_psnr(denoised, clean))

            # Per-layer PSNR (using V4-detected reference)
            if soft_masks is None:
                ref_top, ref_bottom, _ = v4.detect_retina_band(clean)
                boundaries = torch.zeros(1, 4, noisy.shape[3])
                for w in range(noisy.shape[3]):
                    t, b = ref_top[0, w].item(), ref_bottom[0, w].item()
                    boundaries[0, 0, w] = t
                    boundaries[0, 1, w] = t + (b - t) * 0.33
                    boundaries[0, 2, w] = t + (b - t) * 0.66
                    boundaries[0, 3, w] = b
                # Create masks manually
                y = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1) / (H - 1)
                b_exp = boundaries.unsqueeze(2)
                temp = 5.0
                mask_0 = torch.sigmoid((b_exp[:, 1:2] * (H-1) - y * (H-1)) / temp)
                mask_1 = torch.sigmoid((y * (H-1) - b_exp[:, 1:2] * (H-1)) / temp) * torch.sigmoid((b_exp[:, 2:3] * (H-1) - y * (H-1)) / temp)
                mask_2 = torch.sigmoid((y * (H-1) - b_exp[:, 2:3] * (H-1)) / temp) * torch.sigmoid((b_exp[:, 3:4] * (H-1) - y * (H-1)) / temp)
                mask_3 = torch.sigmoid((y * (H-1) - b_exp[:, 3:4] * (H-1)) / temp)
                soft_masks = torch.cat([mask_0, mask_1, mask_2, mask_3], dim=1)
                soft_masks = soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

            for i in range(4):
                mask = soft_masks[:, i:i+1, :, :]
                mask_sum = mask.sum().clamp(min=1.0)
                layer_mse = ((denoised - clean) ** 2 * mask).sum() / mask_sum
                layer_psnr = 10 * np.log10(1.0 / max(layer_mse.item(), 1e-10))
                results['layer_psnrs'][i].append(layer_psnr)

    return {
        'global_psnr': np.mean(results['global_psnr']),
        'layer0_psnr': np.mean(results['layer_psnrs'][0]),
        'layer1_psnr': np.mean(results['layer_psnrs'][1]),
        'layer2_psnr': np.mean(results['layer_psnrs'][2]),
        'layer3_psnr': np.mean(results['layer_psnrs'][3]),
        'avg_layer_psnr': np.mean([np.mean(l) for l in results['layer_psnrs']]),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CLINICAL LAYER-SPECIFIC DENOISING TEST (Small Images)")
    print("=" * 70)

    pku37_root = "/home/kumwilai/OCT/pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"

    if not os.path.exists(pku37_root):
        print(f"ERROR: PKU37 not found at {pku37_root}")
        return 1

    # Load small images
    print("\nLoading PKU37 (64x64 images)...")
    pairs = load_pku37_small(pku37_root, max_images=10, target_size=64)
    print(f"Loaded {len(pairs)} pairs")

    # Test 1: Global denoiser (baseline)
    print("\n" + "=" * 70)
    print("TEST 1: Global Denoiser (Baseline)")
    print("=" * 70)
    print("Training...")
    global_model = train_global_denoiser(pairs, epochs=30)

    # Need to add create_soft_masks to global model for evaluation
    class GlobalWithMasks(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
        def forward(self, x):
            return self.base(x)
        def create_soft_masks(self, boundaries, H):
            B, N, W = boundaries.shape
            device = boundaries.device
            boundaries_px = boundaries * (H - 1)
            y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
            b_exp = boundaries_px.unsqueeze(2)
            temp = 5.0
            mask_0 = torch.sigmoid((b_exp[:, 1:2] - y) / temp)
            mask_1 = torch.sigmoid((y - b_exp[:, 1:2]) / temp) * torch.sigmoid((b_exp[:, 2:3] - y) / temp)
            mask_2 = torch.sigmoid((y - b_exp[:, 2:3]) / temp) * torch.sigmoid((b_exp[:, 3:4] - y) / temp)
            mask_3 = torch.sigmoid((y - b_exp[:, 3:4]) / temp)
            soft_masks = torch.cat([mask_0, mask_1, mask_2, mask_3], dim=1)
            return soft_masks / (soft_masks.sum(dim=1, keepdim=True) + 1e-8)

    global_model_wrapped = GlobalWithMasks(global_model)
    results_global = evaluate(global_model_wrapped, pairs, is_clinical=False)
    print(f"Results: Global PSNR={results_global['global_psnr']:.2f} dB")

    # Test 2: Clinical layer-specific WITHOUT V4
    print("\n" + "=" * 70)
    print("TEST 2: Clinical Layer-Specific (WITHOUT V4)")
    print("=" * 70)
    print("Training...")
    clinical_model_no_v4, boundary_model_no_v4 = train_clinical_denoiser(pairs, use_v4=False, epochs=30)
    results_clinical_no_v4 = evaluate(clinical_model_no_v4, pairs, boundary_model_no_v4, is_clinical=True)
    print(f"Results: Global PSNR={results_clinical_no_v4['global_psnr']:.2f} dB")

    # Test 3: Clinical layer-specific WITH V4
    print("\n" + "=" * 70)
    print("TEST 3: Clinical Layer-Specific (WITH V4)")
    print("=" * 70)
    print("Training...")
    clinical_model_v4, boundary_model_v4 = train_clinical_denoiser(pairs, use_v4=True, epochs=30)
    results_clinical_v4 = evaluate(clinical_model_v4, pairs, boundary_model_v4, is_clinical=True)
    print(f"Results: Global PSNR={results_clinical_v4['global_psnr']:.2f} dB")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<30} {'Global':>10} {'Layer0':>10} {'Layer1':>10} {'Layer2':>10} {'Layer3':>10} {'Avg Layer':>10}")
    print("-" * 90)

    for name, r in [
        ("Global Denoiser", results_global),
        ("Clinical (no V4)", results_clinical_no_v4),
        ("Clinical + V4", results_clinical_v4),
    ]:
        print(f"{name:<30} {r['global_psnr']:>10.2f} {r['layer0_psnr']:>10.2f} {r['layer1_psnr']:>10.2f} {r['layer2_psnr']:>10.2f} {r['layer3_psnr']:>10.2f} {r['avg_layer_psnr']:>10.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    global_best = results_global['global_psnr']
    clinical_v4_global = results_clinical_v4['global_psnr']
    clinical_v4_avg_layer = results_clinical_v4['avg_layer_psnr']
    global_avg_layer = results_global['avg_layer_psnr']

    print(f"\nGlobal PSNR: Clinical+V4 vs Global = {clinical_v4_global - global_best:+.2f} dB")
    print(f"Avg Layer PSNR: Clinical+V4 vs Global = {clinical_v4_avg_layer - global_avg_layer:+.2f} dB")

    if clinical_v4_avg_layer > global_avg_layer:
        print("\n✓ Clinical + V4 achieves BETTER per-layer quality!")
    else:
        print("\n△ Clinical + V4 needs more tuning for per-layer gains")

    return 0


if __name__ == "__main__":
    sys.exit(main())
