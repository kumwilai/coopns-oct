# Testing Status: Learning Rate Hypothesis

## Current Test: Test 1 - Lower Learning Rate + Plateau Scheduler

**Status:** 🔄 **RUNNING**

**Started:** 2025-12-03 10:25 UTC

**Progress:** Fine-tuning Epoch 1/50 (Step 100/3150)

---

## What We're Testing

### Problem Being Solved:
CASA+N2V peaks at 26.40 dB (epoch 2), then degrades to 25.76 dB, performing WORSE than classical BM3D (~27 dB).

### Hypothesis:
Learning rate too aggressive causing:
1. Quick peak during warmup (epoch 2: 26.40 dB)
2. Overshoot when warmup ends
3. Consistent degradation afterward

### Test 1 Changes:

| Parameter | Previous | Test 1 | Reason |
|-----------|----------|---------|---------|
| Adapter LR | 5e-4 | **1e-4** | 5× lower to prevent overshoot |
| Backbone LR | 1e-4 | **5e-5** | 2× lower to prevent overshoot |
| Scheduler | cosine+warmup(8) | **plateau+no warmup** | Constant LR, no overshoot |
| Early Stop Patience | 10 | **20** | More patient |

### Success Criteria:
1. ✅ PSNR maintains 26.4+ dB without degradation
2. ✅ PSNR reaches 27+ dB (beats BM3D)
3. ✅ Training curve stable (no overshoot pattern)

---

## Key Milestones to Watch

### Epoch 1-2: Initial Learning
- **Previous behavior:** Quick improvement to ~25-26 dB
- **Expected with fix:** Slower but steady improvement
- **Watch for:** No sudden spikes (indicates stable LR)

### Epoch 3-5: Stability Test
- **Previous behavior:** Started degrading (26.40 → 26.11 → 25.94 dB)
- **Expected with fix:** Continue improving or plateau
- **Critical:** Should NOT degrade below epoch 2 performance

### Epoch 10-20: Convergence
- **Expected:** Steady improvement toward 27-28 dB
- **Goal:** Beat BM3D baseline (~27 dB)
- **Stretch:** Approach SwinIR level (~28.8 dB)

---

## Validation Results (Updated as training progresses)

### Meta-Learning Phase:
- Epoch 1: Loss=0.0463
- Epoch 2: Loss=0.0241
- Epoch 3: Loss=0.0337
- Epoch 4: Loss=0.0170
- Epoch 5: Loss=0.0257
- Status: ✅ Complete

### Fine-Tuning Phase:
- Epoch 1: In progress (Step 100/3150, Loss=0.0040)
- First validation: Pending...

**Waiting for first validation PSNR...**

---

## Monitoring Commands

**Real-time log:**
```bash
tail -f /home/kumwilai/OCT/test1_lower_lr.log
```

**Training progress:**
```bash
bash /home/kumwilai/OCT/monitor_test1.sh
```

**Extract validation PSNRs:**
```bash
grep "PSNR:" /home/kumwilai/OCT/test1_lower_lr.log | grep "Val "
```

---

## Next Steps Based on Results

### If Test 1 Succeeds (PSNR ≥ 27 dB, stable):
1. ✅ **Hypothesis confirmed:** Learning rate was the issue
2. Document the fix in `FINAL_SOLUTION.md`
3. Run full training (100 epochs) with these settings
4. Expected final: 27-28 dB (beating BM3D)
5. Update baseline comparison table

### If Test 1 Partially Succeeds (PSNR stable at 26.4 dB):
1. ⚠️ **Progress:** No degradation (stable training)
2. ❌ **Issue:** Still not beating BM3D
3. Next action: Test 2 (skip meta-learning)
4. Hypothesis: Meta-learning on clean images biases model

### If Test 1 Fails (PSNR still degrades):
1. ❌ **Learning rate not the issue**
2. Next action: Test 2 (skip meta-learning) immediately
3. Alternative hypothesis: Meta-learning interference
4. Or Test 3: Even lower LR (5e-5, 2e-5) for 200 epochs

---

## Baseline Comparison Table

Current standings to beat:

| Method | PSNR (dB) | SSIM | Gain | Status |
|--------|-----------|------|------|--------|
| Noisy Input | 20.55 | 0.398 | --- | Baseline |
| Vanilla N2V | 24.73 | 0.724 | +4.2 dB | Self-supervised baseline |
| DRUNet | 26.19 | 0.803 | +5.6 dB | Supervised |
| **CASA+N2V (prev)** | **26.40** | --- | **+5.9 dB** | **Degraded to 25.76** |
| **Test 1 Target** | **≥27.0** | **≥0.82** | **≥+6.5 dB** | **Beat BM3D** |
| BM3D | ~27.0 | ~0.82 | +6.5 dB | Classical benchmark |
| SwinIR | 28.84 | 0.858 | +8.3 dB | Strong supervised |
| NAFNet | 29.53 | 0.880 | +9.0 dB | Best supervised |

**Critical:** We need >27 dB to prove self-supervised N2V can beat classical methods

---

## Dataset Context

**Why this is challenging:**
- OCT speckle noise (multiplicative, not additive Gaussian)
- High variance: 20.55 ± 4.43 dB (some images only 7.91 dB!)
- Even supervised NAFNet only reaches 29.53 dB
- Typical denoising benchmarks see +15-20 dB gains, we're seeing +7-9 dB

**Realistic expectations:**
- 27-28 dB: Good (beats classical methods)
- 28-29 dB: Excellent (matches strong supervised methods)
- 30+ dB: Outstanding (would exceed current supervised baselines)

---

## Technical Notes

### Why Lower LR Might Help:
1. **N2V is harder than supervised:** Model must learn from noise statistics alone
2. **Blind-spot constraint:** Network can't see center pixel, must infer from neighbors
3. **Overfitting risk:** High LR might memorize noise patterns instead of learning denoising
4. **Fine-tuning sensitivity:** Starting from meta-learned weights needs gentle updates

### Why No Warmup:
1. Previous runs peaked during warmup (epoch 2)
2. Degradation started when warmup ended
3. N2V might not need warmup like supervised methods do
4. Constant LR avoids schedule-induced instabilities

### Why Plateau Scheduler:
1. Only options: 'cosine' (decays continuously) or 'plateau' (constant until plateau)
2. Plateau keeps LR constant unless validation stops improving
3. Closest to "constant LR" behavior we want
4. Only reduces LR when truly stuck (conservative)

---

**Last Updated:** 2025-12-03 10:28 UTC

**Estimated Completion:** 2025-12-03 11:30 UTC (~60-90 minutes for 50 epochs)

**Monitor progress:** The first few validation results (epochs 1-5) will tell us if the fix is working.
