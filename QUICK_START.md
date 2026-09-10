# Quick Start: Fixed Adaptive Denoising System

## Problem Solved
Your adaptive multi-head system had **negative adaptive gain (-0.04 dB)** because residual heads (width=16) were too small to beat the strong base NAFNet (width=64, 29.80 dB).

## Solution Applied
1. Increased head capacity: width=16 → 32 (2x capacity)
2. Removed orthogonality constraint: base_orthogonality_weight 0.1 → 0.0

## Test Results
Both patches achieved **positive adaptive gain**:
- PATCH 1 (width=32): **+1.94 dB** ✅
- PATCH 2 (width=48): **+1.95 dB** ✅

## Run Production Training

Simply execute:

```bash
bash train_final_working.sh
```

This will:
- Train for 50 epochs with early stopping
- Use 1000 training samples, 100 validation samples
- Apply the validated width=32 configuration
- Save best checkpoint to: `checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth`
- Log metrics to: `outputs/duke_metrics_final_width32.jsonl`

## Expected Results

With the fixed configuration, you should see:
- ✅ Overall PSNR > Base PSNR (positive adaptive gain!)
- ✅ All heads contributing to overall performance
- ✅ Proper noise-type specialization (different heads for different noise)
- ✅ Improved denoising over base NAFNet alone

## Key Configuration Changes

In `train_final_working.sh`:
```bash
--residual_head_width 32              # FIXED: Was 16
--base_orthogonality_weight 0.0       # FIXED: Was 0.1
--head_quality_weight 5.0             # Strong supervision
--head_diversity_weight 0.3           # Moderate diversity
--base_nafnet_lr 0.0                  # Base stays frozen
```

## Monitoring Training

Watch for these indicators in the logs:
1. **Adaptive Gain**: Should be positive (Overall PSNR - Base PSNR > 0)
2. **Head Effectiveness**: At least some heads marked "GOOD"
3. **Top-1 Routing Accuracy**: Should be 60-70%+
4. **Overall PSNR**: Should improve over base (29.80 dB)

## Files Created

1. **train_final_working.sh** - Production training script (recommended)
2. **test_patch_head_capacity.sh** - Quick test of width=32 fix
3. **test_patch2_larger_heads.sh** - Validation test of width=48
4. **SOLUTION_ADAPTIVE_GAIN_FIX.md** - Detailed problem analysis and solution
5. **TEST_RESULTS_SUMMARY.md** - Complete test results from both patches
6. **QUICK_START.md** - This file

## Troubleshooting

If you encounter issues:

1. **Still negative gain?**
   - Check that `--residual_head_width 32` is set
   - Verify `--base_orthogonality_weight 0.0`
   - Consider running PATCH 2 with width=48

2. **Heads marked "BAD"?**
   - This is OK if overall PSNR > base PSNR
   - The adaptive blending still improves performance
   - Some heads may improve with more epochs

3. **Training too slow?**
   - Reduce `--max_samples` for faster iteration
   - Use smaller `--batch_size` if memory constrained

## Success Criteria

Your training is successful when:
- ✅ Adaptive gain is **POSITIVE** (not negative!)
- ✅ Overall PSNR exceeds base NAFNet PSNR
- ✅ Model saves checkpoints with improving scores

That's it! The system is ready for production training.
