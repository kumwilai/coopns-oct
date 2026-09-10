# Training Analysis: Epoch 6 Results

## Problem: Loss Weights STILL Imbalanced

### Current Loss Composition (Epoch 6):
```
Weighted contributions:
  Denoising:        15.5%  ← STILL TOO LOW!
  Interpretability:  22.2%
  Noise map:        62.2%  ← STILL DOMINATING!
```

### Performance Gap:
```
Expected improvement: +0.7-0.9 dB
Actual improvement:   +0.10-0.15 dB
Missing:              0.55-0.75 dB ← MAJOR GAP!
```

## Root Cause

### Issue #1: Interpretability Loss is HUGE (0.6739)
**Why so large?**
- Interpretability loss = NLL loss on noise type classification
- When model is uncertain or wrong: loss can be 0.5-1.0
- Denoising loss = L1 pixel loss: typically 0.01-0.02
- **Ratio**: Interp is 40x larger than denoise!

**Current weighting:**
```
lambda_interp = 0.031
Contribution = 0.6739 * 0.031 = 0.0209 (22% of total loss)
```

**Problem with your config:**
```bash
--lambda_interp_start 0.03   # Starting high
--lambda_interp_end 0.08     # INCREASING to 0.08! ← BACKWARDS!
```

Lambda is INCREASING during training → interpretability gets MORE emphasis → denoising gets LESS!

### Issue #2: Noise Map Weight Still Too High
```
noise_map_loss_weight = 0.4
Noise map loss = 0.1463 (raw)
Contribution = 0.1463 * 0.4 = 0.0585 (62% of total!)
```

## Current Results

### Performance Metrics (Best Epoch 3):
```
Overall PSNR:      33.18 dB
Base NAFNet PSNR:  33.03 dB
Improvement:       +0.15 dB ← INSUFFICIENT!

Region-specific:
  Inner retina:  32.89 dB
  Outer retina:  33.73 dB
  Difference:    0.84 dB (showing noise variation exists)
```

### Why Only +0.15 dB?
1. **Only 15.5% of loss optimizes denoising** → minimal PSNR improvement
2. **62% optimizes noise map matching** → learns accurate maps, but no denoising gain
3. **22% optimizes classification** → learns noise types, but no denoising gain

**The model is learning what you're asking it to learn** (noise maps + classification),
**but you're NOT asking it to denoise better!**

## Solution: Rebalance Loss Weights

### Target Loss Composition:
```
Denoising:        ~60%  (main objective)
Noise map:        ~20%  (still learned, but not dominant)
Interpretability: ~15%  (keep for novelty)
Param reg:        ~5%
```

### Required Changes:

#### Change #1: Fix Lambda Schedule (CRITICAL)
```bash
# WRONG (current):
--lambda_interp_start 0.03
--lambda_interp_end 0.08    # Increasing ← BAD!

# CORRECT:
--lambda_interp_start 0.008  # Start lower
--lambda_interp_end 0.003    # Decrease over time ← Focus on denoising!
```

**Rationale:**
- Early: Higher lambda (0.008) helps learn noise types
- Late: Lower lambda (0.003) focuses on denoising quality
- Total contribution: ~15% (vs 22% current)

#### Change #2: Reduce Noise Map Weight
```bash
# Current:
--noise_map_loss_weight 0.4   # 62% of loss

# Recommended:
--noise_map_loss_weight 0.15  # ~20% of loss
```

**Rationale:**
- Phase 1 already pre-trained noise maps (with weight=3.0)
- Phase 2 should maintain them, not re-optimize them
- 0.15 is enough to keep maps accurate without dominating

### Expected Outcome After Fix:

**New loss composition:**
```
Denoising:   0.015 * 1.0  = 0.015 (60%)
Interp:      0.67  * 0.005 = 0.0034 (14%)
Noise map:   0.146 * 0.15  = 0.022 (22%)
Param:       0.001 * 0.03  = 0.00003 (0.1%)
Total:       ~0.025

Denoise contribution: 0.015 / 0.025 = 60% ✓
```

**Expected PSNR:**
```
With 60% loss on denoising (vs 15.5% current):
  - 3.9x more optimization effort on PSNR
  - Expected gain: 0.15 dB * 3.9 ≈ +0.6 dB
  - Predicted final: 33.18 + 0.6 = 33.8 dB ✓ TARGET!
```

## Recommendation

### Option 1: Stop & Restart with Fixed Weights (RECOMMENDED)
```bash
# Kill current training (Ctrl+C)
# Edit run_duke_region_focused.sh Phase 2:
--lambda_interp_start 0.008
--lambda_interp_end 0.003
--noise_map_loss_weight 0.15

# Resume from Phase 1 checkpoint
--resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth
```

**Expected:** 33.7-34.0 dB in 40-50 epochs

### Option 2: Continue Current Training (NOT RECOMMENDED)
- Current trajectory: Will plateau at ~33.2-33.3 dB
- Will NOT reach 33.8+ dB target
- Wastes compute time on suboptimal training

## Why This Matters for TMI

### Current State:
```
+0.15 dB improvement → Reviewers will say: "Marginal improvement, not significant"
```

### After Fix:
```
+0.6-0.8 dB improvement → Reviewers will say: "Clear improvement, publishable"
```

### The 0.5 dB Difference = Acceptance vs Rejection

**TMI reviewers expect:**
- Clear improvement over strong baseline (≥0.5 dB)
- Novel contribution (you have: per-pixel noise maps)
- Clinical relevance (you have: region-adaptive)

**With +0.15 dB:** Missing the performance criterion
**With +0.7 dB:** Meeting all criteria ✓

## Action Items

1. **STOP current training** (it won't reach target)

2. **Edit `run_duke_region_focused.sh` Phase 2:**
   ```bash
   --lambda_interp_start 0.008    # Was 0.03
   --lambda_interp_end 0.003      # Was 0.08
   --noise_map_loss_weight 0.15   # Was 0.4
   ```

3. **Restart Phase 2:**
   ```bash
   # Comment out Phase 1 (already done)
   # Run only Phase 2 with fixed weights
   bash run_duke_region_focused.sh
   ```

4. **Monitor:**
   - Loss composition should show ~60% denoising
   - PSNR should improve to 33.7-34.0 dB
   - Region-specific gains should emerge

## Bottom Line

**Problem:** You're optimizing for noise map accuracy (62%), not denoising quality (15.5%)

**Fix:** Reduce noise map weight (0.4 → 0.15) and lambda (0.03-0.08 → 0.008-0.003)

**Expected Gain:** +0.5-0.6 dB improvement → Reach 33.7-34.0 dB target ✓

**Do This NOW** - Every epoch with wrong weights is wasted compute!
