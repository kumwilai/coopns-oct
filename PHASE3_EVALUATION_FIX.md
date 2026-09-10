# Phase 3 Evaluation Fix

## Problem

Phase 3 evaluation was showing incorrect results:
- **Reported PSNR**: 23.31 dB (matching noisy input PSNR ~23 dB)
- **Expected PSNR**: ~33 dB (matching training validation PSNR)
- Model appeared to be returning noisy input unchanged

## Root Causes

### 1. Missing Model Architecture Parameter ⚠️ CRITICAL

**Issue**: Checkpoint was trained with `use_joint_signal_expert=True`, but evaluation script didn't pass this parameter.

**Evidence**:
```python
# Checkpoint metadata
use_joint_signal_expert: True
joint_expert_channels: 96
joint_mix_init: 0.15
```

**Impact**: Model was built WITHOUT the joint expert component, which is a critical part of the denoising architecture. This caused the model to essentially pass through the input unchanged.

**Fix**: Added missing parameters to `evaluate_nsnd_fixed_pairs.py`:
```python
use_joint_signal_expert=meta("use_joint_signal_expert", False),
joint_expert_channels=meta("joint_expert_channels", 96),
joint_mix_init=meta("joint_mix_init", 0.15),
```

### 2. Swapped Pairs File Order

**Issue**: Test pairs file format is `clean_path<TAB>noisy_path`, but script was loading them as `noisy_path<TAB>clean_path`.

**Evidence**:
```
# test_pairs_duke_analysis_maps.txt header:
# Format: clean_path<TAB>noisy_path
```

**Impact**: Script was denoising CLEAN images and comparing to NOISY references, resulting in low PSNR (~23 dB).

**Fix**: Corrected pair loading in `evaluate_nsnd_fixed_pairs.py` line 167:
```python
# Before:
noisy_path, clean_path = parts

# After:
clean_path, noisy_path = parts  # File format: clean<TAB>noisy
```

## Verification

After fixes, evaluation on 5 test images:
```
PSNR: 32.92 dB  ✅ (was 23.31 dB)
SSIM: 0.8848    ✅ (was 0.5413)
```

This matches the checkpoint's validation PSNR of 33.35 dB (epoch 9).

## Files Modified

1. **`nsnd_oct/scripts/evaluate_nsnd_fixed_pairs.py`**:
   - Added `use_joint_signal_expert`, `joint_expert_channels`, `joint_mix_init` parameters (lines 217-219)
   - Fixed pairs file loading order (line 167)
   - Added architecture debug output (lines 223-232)

## How to Run Phase 3 Evaluation (Fixed)

```bash
cd /home/kumwilai/OCT

python -u nsnd_oct/scripts/evaluate_nsnd_fixed_pairs.py \
  --checkpoint checkpoints/multitask_hybrid_nsnd_lambda0p006to0p002_cosine_best.pth \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_improved_seed0.pth \
  --pairs test_pairs_duke_analysis_maps.txt \
  --log_region_psnr \
  --log_roi_psnr \
  --roi_center_frac 0.4 \
  --out_json outputs/duke_eval_region.json \
  --device cpu
```

Expected results:
- **PSNR**: ~32-33 dB
- **SSIM**: ~0.88-0.90
- Processing time: ~48 seconds per image on CPU

## Checkpoint Metadata Reference

The checkpoint includes these key parameters:
```python
base_nafnet: True
base_nafnet_width: 64
base_nafnet_type: "full"
base_enc_blk_nums: [2, 2, 2]  # NOT saved in old checkpoint
base_dec_blk_nums: [2, 2, 2]  # NOT saved in old checkpoint
base_middle_blk_num: 2        # NOT saved in old checkpoint
shared_residual: True
shared_trunk_width: 32
shared_adapter_channels: 96
shared_adapter_hidden: 64
use_spatial_weights: True
use_joint_signal_expert: True  # CRITICAL - was missing
joint_expert_channels: 96
joint_mix_init: 0.15
residual_blend: True
residual_blend_init: 0.6
```

Note: `base_enc_blk_nums`, `base_dec_blk_nums`, `base_middle_blk_num` are NOT in old checkpoints but have been added to new checkpoint saves.

## Status

✅ **Phase 3 Evaluation: FIXED**

The evaluation script now correctly:
1. Loads all model architecture parameters from checkpoint
2. Builds model with joint expert component
3. Loads pairs in correct order (noisy → clean)
4. Reports correct denoised PSNR (~33 dB)
