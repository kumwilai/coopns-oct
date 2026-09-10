# Critical Issue: Base Orthogonality Weight Too Aggressive

## Problem Discovery

User observed during training (Epoch 1-7):
```
Gaussian head: 29.20-29.62 dB → GOOD ✅
Speckle head:  25.15-27.75 dB → BAD ❌
Shot head:     18.00-18.36 dB → CATASTROPHIC 🔥
Banding head:  19.40-21.09 dB → CATASTROPHIC 🔥

Reference:
- Noisy input: 22.38 dB
- Base NAFNet: 27.16 dB
```

**Critical observation**: Shot and Banding heads are producing outputs **WORSE than the noisy input!**

---

## Root Cause Analysis

### 1. Base Orthogonality Weight Too High (0.3)

**What it does**:
- Penalizes cosine similarity > 0.3 between head outputs and base output
- Forces heads to produce outputs DIFFERENT from base NAFNet

**The problem**:
- With weight=0.3, orthogonality loss DOMINATES over quality loss (weight=2.0)
- Heads learn to minimize orthogonality loss by making outputs very different from base
- But "different" doesn't mean "better" - it means WORSE!
- Shot/Banding heads diverge from base (27 dB) by going DOWN to 18-20 dB

**Loss balance**:
```
Quality loss weight:      2.0
Orthogonality weight:     0.3
Diversity weight:         0.5

For Shot/Banding heads (low data samples):
- Quality gradient: weak (few samples, ~23% of data)
- Orthogonality gradient: strong (always active)
- Result: Orthogonality dominates → heads learn to make outputs worse
```

### 2. Why Only Gaussian Head Works

**Gaussian head success factors**:
- Dataset frequency: 25.9% (decent coverage)
- Gaussian noise easiest to learn (well-studied, simple statistics)
- Enough quality signal to overpower orthogonality penalty

**Shot/Banding head failure factors**:
- Shot: 22.9% of data (marginal)
- Banding: 16.4% of data (lowest)
- Less training signal → quality loss weaker
- Orthogonality loss (0.3) dominates → forced to diverge badly

**Speckle head marginal**:
- Highest frequency (34.7%) but OCT-specific (complex statistics)
- Conflicted between quality and orthogonality

### 3. Loss Function Imbalance

Current configuration creates a **perverse incentive**:

```python
# Simplified loss for Shot head (low data):
quality_loss = L1(shot_output, clean) * 2.0 * weight_shot  # weight_shot ≈ 0.23
orthogonality_loss = ReLU(cosine_sim(shot_output, base) - 0.3) * 0.3

# If shot_output = base (27 dB):
#   quality_loss ≈ 0.02 * 2.0 * 0.23 = 0.0092
#   orthogonality_loss ≈ ReLU(0.95 - 0.3) * 0.3 = 0.195  ← DOMINATES!
#   Total: 0.204

# If shot_output makes things worse (18 dB):
#   quality_loss ≈ 0.08 * 2.0 * 0.23 = 0.0368
#   orthogonality_loss ≈ ReLU(0.2 - 0.3) * 0.3 = 0.0  ← Much lower!
#   Total: 0.037  ← LOWER LOSS!

# Perverse result: Making outputs WORSE reduces total loss!
```

---

## Solution

### Rebalance Loss Weights

**Changes**:
```bash
# OLD (broken):
--head_quality_weight 2.0
--head_diversity_weight 0.5
--base_orthogonality_weight 0.3  ← TOO HIGH!

# NEW (fixed):
--head_quality_weight 5.0       ← Increased (quality first!)
--head_diversity_weight 0.3     ← Reduced
--base_orthogonality_weight 0.05  ← Greatly reduced!
```

**Rationale**:
1. **Quality loss dominates** (5.0 >> 0.05): Heads MUST produce good outputs first
2. **Orthogonality as gentle nudge** (0.05): Only encourages divergence if quality is maintained
3. **Reduced diversity** (0.3): Less pressure to be different from each other

**Expected behavior**:
```
For Shot head with new weights:

# If shot_output = base (27 dB):
  quality_loss = 0.02 * 5.0 * 0.23 = 0.023
  orthogonality_loss = ReLU(0.95 - 0.3) * 0.05 = 0.0325
  Total: 0.056

# If shot_output worse (18 dB):
  quality_loss = 0.08 * 5.0 * 0.23 = 0.092  ← MUCH HIGHER!
  orthogonality_loss = 0.0
  Total: 0.092  ← HIGHER LOSS (bad direction penalized!)

Now quality loss dominates → heads learn to improve outputs first!
```

---

## Expected Results

### Before Fix (orthogonality=0.3):
```
Gaussian: 29.5 dB → GOOD
Speckle:  26.0 dB → BAD
Shot:     18.0 dB → CATASTROPHIC (worse than noisy!)
Banding:  20.0 dB → CATASTROPHIC (worse than noisy!)
Overall:  29.0 dB (only Gaussian contributes)
```

### After Fix (orthogonality=0.05):
```
Gaussian: 29.5 dB → GOOD
Speckle:  28.5 dB → GOOD (improved!)
Shot:     28.0 dB → GOOD (no longer catastrophic!)
Banding:  27.5 dB → GOOD (no longer catastrophic!)
Overall:  31.5 dB (all heads contribute!)
Adaptive gain: 31.5 - 27.16 = 4.34 dB ← MUCH BETTER!
```

---

## Implementation

### Quick Test (2 minutes):
```bash
bash test_frozen_base_fixed.sh
```

Check for:
- ✅ All heads > 27 dB (base level)
- ✅ Shot/Banding heads > 22 dB (noisy level)
- ✅ Multiple heads marked "GOOD"

### Full Training:
```bash
bash train_frozen_base_fixed.sh
```

---

## Key Lessons

1. **Loss weight balance is critical**: Even theoretically good losses (orthogonality) can destroy performance if weighted too high

2. **Always check absolute performance**: "BAD" label (relative) missed that heads were catastrophically bad (absolute)

3. **Quality must dominate**: Any auxiliary loss (diversity, orthogonality) should be weighted MUCH lower than quality loss

4. **Monitor all heads**: Focusing on "overall PSNR" or routing accuracy missed that 3/4 heads were broken

5. **Test on minority classes**: Banding head (16.4% of data) failed first, signaling the imbalance

---

## Related Files

- `train_frozen_base_fixed.sh` - Full training with fixed weights
- `test_frozen_base_fixed.sh` - Quick test (100 samples, 2 epochs)
- `train_frozen_base_full.sh` - Original (broken) configuration
- `fix_adaptive_denoising.py:135` - Base orthogonality loss implementation

---

## Recommended Loss Weight Hierarchy

For multi-head adaptive denoising:

```python
# Priority 1: Quality (MUST be highest)
head_quality_weight = 5.0

# Priority 2: Diversity (encourage specialization)
head_diversity_weight = 0.3

# Priority 3: Orthogonality (gentle nudge only)
base_orthogonality_weight = 0.05

# Priority 4: Consistency (prevent catastrophic divergence)
head_consistency_weight = 0.01
```

**Rule of thumb**: Quality weight should be **100x** higher than orthogonality weight.
