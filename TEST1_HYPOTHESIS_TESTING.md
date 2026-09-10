# Test 1: Lower Learning Rate Hypothesis

## Date: 2025-12-03

## Problem Being Investigated

**Symptom:** CASA+N2V training peaks at 26.40 dB (epoch 2), then degrades to 25.76 dB by epoch 6

**Critical Issue:** Model performance (26.4 dB) is WORSE than classical BM3D (~27 dB)

**Previous attempts:**
- Bug #13 Fixed: N2V mask ratio (4.1% → 12%) ✅
- Bug #14 Tested: Residual mode vs sigmoid activation ❌ (no improvement)

---

## Hypothesis: Learning Rate Too Aggressive

**Evidence:**
1. Training peaks at epoch 2 (26.40 dB) - during warmup phase
2. Consistent degradation after epoch 2 → 25.76 dB by epoch 6
3. Degradation starts when warmup ends and cosine decay begins
4. Pattern suggests overshoot: learns too fast, then overshoots optimal weights

**Root Cause Theory:**
- Warmup reaches peak at epoch 2
- After warmup, cosine scheduler increases LR too much
- Model overshoots and degrades
- Learning rate schedule not suited for N2V self-supervised learning

---

## Test 1 Configuration

### Changes from Previous Run:

| Parameter | Previous (Buggy) | Test 1 (Fixed) | Reason |
|-----------|------------------|----------------|---------|
| Adapter LR | 5e-4 | **1e-4** | 5× lower to prevent overshoot |
| Backbone LR | 1e-4 | **5e-5** | 2× lower to prevent overshoot |
| Scheduler | cosine + warmup | **plateau** | Keeps LR constant unless plateau |
| Warmup Epochs | 8 | **0** | Eliminate warmup overshoot |
| Early Stop Patience | 10 | **20** | More patient to avoid premature stopping |
| Finetune Epochs | 100 | **50** | Quick test, will extend if successful |

### Expected Outcomes:

**Success Criteria:**
- ✅ PSNR maintains 26.4+ dB without degradation
- ✅ PSNR reaches 27+ dB (beats BM3D)
- ✅ Training curve stable (no overshoot pattern)

**If Successful:**
- Confirms learning rate was the issue
- Run full training (100 epochs) with these settings
- Expected final: 27-28 dB (realistic target for this dataset)

**If Still Fails:**
- Move to Test 2: Skip meta-learning (test for meta-learning bias)
- Test 3: Even lower LR (5e-5, 2e-5) for 200 epochs

---

## Current Status

**Status:** 🔄 RUNNING

**Progress:**
- Meta-learning: Epoch 3/5 (in progress)
- Fine-tuning: Not started yet

**Output Log:** `/home/kumwilai/OCT/test1_lower_lr.log`

**Checkpoint:** `checkpoints/casa_n2v_test1_lower_lr/`

---

## Baseline Comparison

| Method | PSNR (dB) | SSIM | Status |
|--------|-----------|------|---------|
| **Noisy Input** | 20.55 ± 4.43 | 0.398 | Baseline |
| **BM3D (classical)** | ~27.0 | ~0.82 | **Target to beat** |
| NLM (classical) | ~27.5 | ~0.81 | Classical benchmark |
| DRUNet (supervised) | 26.19 | 0.803 | Supervised baseline |
| **CASA+N2V (previous)** | **26.40** | ? | **Current (worse than BM3D)** |
| Vanilla N2V | 24.73 | 0.724 | Self-supervised baseline |
| SwinIR (supervised) | 28.84 | 0.858 | Strong supervised |
| NAFNet (supervised) | 29.53 | 0.880 | Best supervised |

**Goal:** Beat BM3D (>27 dB) to prove N2V self-supervised approach works

**Realistic Target:** 27-28 dB (approaching SwinIR)

**Stretch Goal:** 28-29 dB (matching SwinIR/NAFNet range)

---

## Dataset Difficulty

**Challenge:** OCT speckle noise is extremely difficult

**Evidence:**
- Noisy input: 20.55 ± 4.43 dB (very noisy)
- High variance: ±4.43 dB (some images 7.91 dB, others 27.62 dB)
- Even supervised NAFNet only gets 29.53 dB
- Gain from denoising: +7-9 dB (vs +15-20 dB on easier datasets)

**Implication:** Expecting 31-32 dB was unrealistic for this dataset

---

## Next Actions

1. **Wait for Test 1 to complete** (~30-60 minutes)
   - Monitor training curve for stability
   - Check if degradation still occurs
   - Verify final PSNR > 27 dB

2. **If Test 1 succeeds:**
   - Document the fix
   - Run full training (100 epochs)
   - Update final_diagnosis.md with solution

3. **If Test 1 fails (still degrading):**
   - Move to Test 2: Skip meta-learning
   - Hypothesis: Meta-learning on clean images biases model incorrectly
   - Test with direct N2V training (no meta phase)

4. **If Test 2 fails:**
   - Move to Test 3: Very low LR (5e-5, 2e-5) for 200 epochs
   - Or investigate other potential issues (dataset quality, model architecture)

---

## Technical Notes

### Why Plateau Scheduler?

- Only available options: 'cosine' or 'plateau'
- Cosine: LR decays over time (may cause instability)
- Plateau: LR stays constant until validation plateaus, then reduces
- Plateau is closest to "constant LR" behavior we want

### Why No Warmup?

- Previous runs peaked during warmup (epoch 2)
- Then degraded after warmup ended
- Suggests warmup may be causing instability
- N2V might not benefit from warmup like supervised methods do

### Why Lower LR?

- N2V is self-supervised: learns from noise statistics
- May need gentler updates than supervised learning
- Classical algorithms (BM3D) don't "learn" - they denoise optimally
- To beat BM3D, N2V must learn subtle noise patterns without overfitting

---

**Test Started:** 2025-12-03 10:25 UTC

**Expected Completion:** 2025-12-03 11:00 UTC (approximately)

**Monitoring Command:** `tail -f /home/kumwilai/OCT/test1_lower_lr.log`
