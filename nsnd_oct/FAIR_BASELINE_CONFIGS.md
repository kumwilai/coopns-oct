# Fair Baseline Configurations
## Parameter-Matched to NSND (~7.42M params)

**Target:** 7,421,322 params
**Fair Range (±5%):** 7,050,255 - 7,792,388 params

---

## Summary of Fair Configurations

| Model | Configuration | Parameters | % of Target | Status |
|-------|--------------|------------|-------------|--------|
| **NSND (reference)** | base_width=16, 4 residual heads | **6,021,058** | — | ✓ Trained |
| **Restormer** | dim=320, num_blocks=9 | **7,410,561** | 99.9% | ⚠ Needs config update |
| **SwinIR** | embed_dim=276, blocks=12, heads=4 | **7,354,573** | 99.1% | ⚠ Needs config update |
| **DnCNN** | num_layers=19, features=220 | **7,416,640** | 99.9% | ⚠ Needs config update |
| **NAFNet** | width=21 | **7,431,754** | 100.1% | ⚠ Needs config update |
| **U-Net** | features=64 | **7,699,009** | 103.7% | ⚠ Needs config update |

✓ **All 5 baselines are now parameter-matched within ±5%**

---

## Detailed Configurations

### 1. Restormer
```python
RestormerSmall(in_channels=1, out_channels=1, dim=320, num_blocks=9)
# Parameters: 7,410,561 (7.41M)
# Diff from target: -10,761 (-0.1%)
```

**Training script:** `scripts/train_restormer_realistic.py`
**Model file:** `nsnd/models/restormer.py`
**Changes needed:** Update default `dim=32` → `dim=320`, `num_blocks=4` → `num_blocks=9`

---

### 2. SwinIR
```python
SwinIRSmall(in_channels=1, out_channels=1,
            embed_dim=276, num_blocks=12, num_heads=4, window_size=4)
# Parameters: 7,354,573 (7.35M)
# Diff from target: -66,749 (-0.9%)
```

**Training script:** `scripts/train_swinir_realistic.py`
**Model file:** `nsnd/models/swinir.py`
**Changes needed:** Update default `embed_dim=32` → `embed_dim=276`, `num_blocks=4` → `num_blocks=12`

---

### 3. DnCNN
```python
DnCNN(in_channels=1, out_channels=1, num_layers=19, features=220)
# Parameters: 7,416,640 (7.42M)
# Diff from target: -4,682 (-0.06%)
```

**Training script:** `scripts/train_dncnn_realistic.py`
**Model file:** `nsnd/models/dncnn.py`
**Changes needed:** Update default `num_layers=17` → `num_layers=19`, `features=64` → `features=220`

---

### 4. NAFNet
```python
NAFNet(img_channel=1, width=21)
# Parameters: 7,431,754 (7.43M)
# Diff from target: +10,432 (+0.1%)
```

**Training script:** `scripts/train_nafnet_on_synthetic.py`
**Model file:** `nsnd/models/nafnet.py`
**Changes needed:** Update default `width=16` → `width=21`

---

### 5. U-Net
```python
UNetSmall(in_channels=1, out_channels=1, features=64, bilinear=False)
# Parameters: 7,699,009 (7.70M)
# Diff from target: +277,687 (+3.7%)
```

**Training script:** `scripts/train_unet_on_synthetic.py`
**Model file:** `nsnd/models/unet.py`
**Changes needed:** Update default `features=32` → `features=64`

---

## Training Fairness Requirements

All baselines MUST use identical training configuration:

### Dataset
- **Root:** `/home/kumwilai/OCT/oct_tmi`
- **Train:** 1000 images (250 per pathology: CNV, DME, Drusen, Normal)
- **Val:** 200 images (50 per pathology)
- **Test:** Fixed pairs from `/home/kumwilai/OCT/test_pairs_realistic_fixed.txt`

### Noise Model
- **Type:** Realistic synthetic (Dirichlet composition)
- **Alpha:** 0.2 (moderate diversity)
- **Param scale:** 1.0 (normal intensity)
- **Components:** Speckle, Banding, Gaussian, Shot
- **Depth profile:** Enabled (same as NSND)

### Training Protocol
- **Crop size:** 64×64
- **Batch size:** 4
- **Epochs:** 30
- **Learning rate:** 1e-4
- **Optimizer:** Adam with weight_decay=1e-5
- **Loss:** L1Loss
- **Seed:** 123 (for reproducibility)

### Evaluation
- **Metrics:** PSNR, SSIM
- **Test set:** Fixed 100-200 pairs (ensure no train/val overlap)
- **Protocol:** Center crop 64×64 (same as validation)

---

## Implementation Checklist

### Phase 1: Update Training Scripts
- [ ] `train_restormer_realistic.py` - Update dim=320, blocks=9, add --max_samples, --val_samples, --seed
- [ ] `train_swinir_realistic.py` - Update embed_dim=276, blocks=12, add --max_samples, --val_samples, --seed
- [ ] `train_dncnn_realistic.py` - Update layers=19, features=220, add --max_samples, --val_samples, --seed
- [ ] `train_nafnet_on_synthetic.py` - Update width=21, ensure 1000/200 split, add --seed
- [ ] `train_unet_on_synthetic.py` - Update features=64, ensure 1000/200 split, add --seed

### Phase 2: Verify Fairness
- [ ] All scripts use same data_root
- [ ] All scripts use same max_samples=1000, val_samples=200
- [ ] All scripts use same crop_size=64
- [ ] All scripts use same epochs=30
- [ ] All scripts use same lr=1e-4
- [ ] All scripts use same seed=123

### Phase 3: Create Evaluation Script
- [ ] `evaluate_fixed_pairs.py` - Load test_pairs_realistic_fixed.txt, compute PSNR/SSIM

### Phase 4: Training Commands
```bash
# Restormer
python scripts/train_restormer_realistic.py --data_root /home/kumwilai/OCT/oct_tmi --max_samples 1000 --val_samples 200 --seed 123

# SwinIR
python scripts/train_swinir_realistic.py --data_root /home/kumwilai/OCT/oct_tmi --max_samples 1000 --val_samples 200 --seed 123

# DnCNN
python scripts/train_dncnn_realistic.py --data_root /home/kumwilai/OCT/oct_tmi --max_samples 1000 --val_samples 200 --seed 123

# NAFNet
python scripts/train_nafnet_on_synthetic.py --data_root /home/kumwilai/OCT/oct_tmi --max_samples 1000 --val_samples 200 --seed 123 --width 21

# U-Net
python scripts/train_unet_on_synthetic.py --data_root /home/kumwilai/OCT/oct_tmi --max_samples 1000 --val_samples 200 --seed 123 --features 64
```

---

## Verification Script

```python
# Run this to verify all configs are fair
python scripts/count_baseline_params.py
python scripts/find_fair_configs.py
```

---

## Status: Ready for Implementation

All parameter-matched configurations have been identified and verified.
Next step: Update training scripts with these configurations.

**Generated:** 2025-12-31
**Verified:** Parameter counts computed programmatically
