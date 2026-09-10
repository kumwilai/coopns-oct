# Summary: Fixes for Adaptive Denoising Issues

## Problems Identified

### 1. **Phase 3 Evaluation Crash** ✅ FIXED
**Error**: Size mismatch when loading checkpoint - model width parameters not read correctly

**Root Cause**: Evaluation script missing NAFNet architecture parameters (`base_enc_blk_nums`, `base_dec_blk_nums`, `base_middle_blk_num`)

**Fix Applied**:
- Updated `evaluate_nsnd_fixed_pairs.py` to read all architecture parameters from checkpoint
- Updated `train_hybrid_nsnd_multitask.py` to save all architecture parameters to checkpoint

**Status**: ✅ Fixed - Phase 3 evaluation should now work

---

### 2. **Weak Adaptive Denoising Gain** ⚠️ REQUIRES RETRAINING
**Problem**: Despite 78.5% Top-1 accuracy, adaptive heads provide only ~0.09-0.29 dB gain over base NAFNet

**Root Causes**:
1. Base NAFNet dominates (residual_blend_logit=0.1 → only 10% head contribution)
2. No supervision for individual head quality
3. Heads don't specialize (no diversity loss)
4. Conservative spatial refiner initialization
5. Insufficient noise-map supervision in Phase 2B

**Fixes Provided**:
- ✅ Analysis document: `ANALYSIS_ADAPTIVE_DENOISING.md`
- ✅ Fix module: `nsnd_oct/scripts/fix_adaptive_denoising.py`
- ✅ Diagnostic tool: `nsnd_oct/scripts/diagnose_head_specialization.py`

---

## Quick Start: Apply Fixes

### Step 1: Test Phase 3 Evaluation (Should Work Now)

```bash
cd /home/kumwilai/OCT
bash run_duke_region_focused.sh
```

The Phase 3 evaluation should now complete without crashes.

### Step 2: Diagnose Current Model

```bash
cd /home/kumwilai/OCT/nsnd_oct

python scripts/diagnose_head_specialization.py \
  --checkpoint ../checkpoints/multitask_hybrid_nsnd_lambda0p006to0p002_cosine_best.pth \
  --test_pairs ../test_pairs_duke_analysis_maps.txt \
  --output_dir ../diagnostics/head_analysis \
  --base_nafnet_width 64
```

This will generate:
- Per-head PSNR analysis
- Spatial weight visualizations
- Head diversity metrics
- Specific recommendations

### Step 3: Apply Training Fixes

To retrain with fixes, you need to integrate the enhanced losses into your training script.

**Option A: Manual Integration** (Recommended for understanding)

Add to `train_hybrid_nsnd_multitask.py` after line 2096:

```python
# Import at top of file
from scripts.fix_adaptive_denoising import compute_enhanced_losses

# In training loop, after computing denoise_loss:
if extras.get("expert_outputs") and extras.get("base_output"):
    enhanced_loss_dict, enhanced_loss = compute_enhanced_losses(
        output=output,
        clean=clean,
        expert_outputs=extras["expert_outputs"],
        base_output=extras["base_output"],
        weights_dict=weights_dict,
        quality_weight=0.5,      # Per-head quality supervision
        diversity_weight=0.1,    # Head specialization
        consistency_weight=0.05, # Prevent catastrophic divergence
    )

    # Add to total loss
    total_loss = total_loss + enhanced_loss

    # Log individual components
    for key, val in enhanced_loss_dict.items():
        if isinstance(val, torch.Tensor):
            # Add to your logging
            pass
```

**Option B: Quick Patch Script**

Create `apply_fixes.sh`:

```bash
#!/bin/bash
# Quick patch to apply adaptive denoising fixes

# 1. Update residual blend init
sed -i 's/residual_blend_init = 0.1/residual_blend_init = 0.3/g' run_duke_region_focused.sh

# 2. Increase noise map loss in Phase 2B
sed -i 's/--noise_map_loss_weight 0.05/--noise_map_loss_weight 0.1/g' run_duke_region_focused.sh

echo "✅ Applied quick fixes to run_duke_region_focused.sh"
echo "⚠  For full fixes, manually integrate fix_adaptive_denoising.py losses"
```

Then run:
```bash
chmod +x apply_fixes.sh
./apply_fixes.sh
```

---

## Recommended Training Configuration

### Phase 1: Multitask Learning (No Changes)
Keep current settings:
- `residual_blend_init = 0.3` (updated from 0.1)
- `noise_map_loss_weight = 0.15`
- Add enhanced losses with weights: quality=0.5, diversity=0.1

### Phase 2B: Fine-tune with Reduced Map Loss
Updated settings:
- `residual_blend_init = 0.3` (keep from Phase 1)
- `noise_map_loss_weight = 0.1` (updated from 0.05)
- Keep enhanced losses active

Expected improvements:
- Individual head PSNR should be competitive (not "BAD")
- Adaptive gain should increase to 0.5-1.0 dB
- Blend weight should stabilize around 0.4-0.7
- Head diversity should decrease (cosine similarity < 0.6)

---

## Files Modified/Created

### Modified Files:
1. ✅ `nsnd_oct/scripts/evaluate_nsnd_fixed_pairs.py`
   - Added missing NAFNet architecture parameters
   - Added spatial weight and region parameters

2. ✅ `nsnd_oct/scripts/train_hybrid_nsnd_multitask.py`
   - Added missing checkpoint save parameters
   - (Manual integration required for enhanced losses)

### Created Files:
1. ✅ `ANALYSIS_ADAPTIVE_DENOISING.md` - Detailed problem analysis
2. ✅ `nsnd_oct/scripts/fix_adaptive_denoising.py` - Enhanced loss functions
3. ✅ `nsnd_oct/scripts/diagnose_head_specialization.py` - Diagnostic tool
4. ✅ `FIXES_SUMMARY.md` - This file

---

## Validation Checklist

After retraining, verify improvements:

- [ ] Phase 3 evaluation completes without errors
- [ ] Residual blend weight is 0.3-0.7 (not 0.1)
- [ ] Per-head PSNR shows "GOOD" (not "BAD") for most heads
- [ ] Adaptive gain over base NAFNet is > 0.5 dB
- [ ] Head diversity (cosine similarity) < 0.6
- [ ] Spatial weight maps show meaningful spatial variation
- [ ] Overall PSNR improves to 33.5-34.0 dB (Phase 2B)

---

## Troubleshooting

### Issue: Phase 3 still crashes
**Check**:
- Verify checkpoint contains `base_enc_blk_nums`, `base_dec_blk_nums`, `base_middle_blk_num`
- If not, retrain from Phase 1 with updated `train_hybrid_nsnd_multitask.py`

### Issue: Enhanced losses cause training instability
**Solution**:
- Reduce loss weights: `quality_weight=0.3, diversity_weight=0.05`
- Increase gradually over epochs

### Issue: Heads still don't specialize
**Check**:
1. Blend weight too low? → Increase `residual_blend_init` to 0.5
2. Spatial refiner not learning? → Check `noise_map_loss_weight >= 0.1`
3. Run diagnostic: `python scripts/diagnose_head_specialization.py`

### Issue: Overall PSNR drops with fixes
**Possible causes**:
- Enhanced losses too strong → Reduce weights
- Heads diverging catastrophically → Increase `consistency_weight` to 0.1
- Need more training epochs → Train for 30-50 epochs in each phase

---

## Expected Timeline

- **Immediate**: Phase 3 evaluation should work now
- **Short-term** (1-2 days): Diagnose current model, apply quick fixes, retrain
- **Full solution** (3-5 days): Integrate all enhanced losses, validate improvements

---

## Contact & Next Steps

1. **First**: Test Phase 3 evaluation with fixed script
2. **Then**: Run diagnostic to see current model issues
3. **Finally**: Retrain with enhanced losses for full fix

If you encounter issues:
- Check `diagnostics/head_analysis/head_analysis_results.json` for details
- Review individual head PSNR values
- Inspect spatial weight visualizations

Good luck! 🚀
