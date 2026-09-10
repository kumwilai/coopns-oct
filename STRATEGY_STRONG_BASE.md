# Strong Base NAFNet Strategy: Maximum Performance Path

## Overview

This document outlines the optimal training strategy to achieve **maximum denoising performance** by first training a strong base NAFNet specifically on the analysis maps dataset, then using it for adaptive head training.

---

## The Problem We're Solving

### Current Situation (Weak Base):
```
Base NAFNet: 27.09 dB (trained on different dataset)
Overall PSNR: 28-29 dB
Adaptive gain: 1-2 dB
❌ Heads limited by weak base
```

### Target (Strong Base):
```
Base NAFNet: 30-31 dB (trained on analysis maps)
Overall PSNR: 32-34 dB
Adaptive gain: 2-3 dB
✅ Heads can specialize on top of strong base
```

**Improvement**: **+4-5 dB overall** (28 → 33 dB)

---

## Three-Stage Training Strategy

### **Stage 1: Train Strong Base NAFNet** 🎯

**Goal**: Create a base NAFNet specifically optimized for analysis maps dataset

**Script**: `train_nafnet_base_analysis_maps.sh`

**Configuration**:
- Dataset: Duke analysis maps (Dirichlet mixture noise)
- Training samples: 1000
- Validation samples: 100
- Width: 64
- Architecture: [2,2,2] encoder/decoder, middle=2
- Epochs: 100 (with early stopping)
- Batch size: 8
- Learning rate: 1e-3

**Expected Results**:
- Validation PSNR: **30-31 dB** (vs 27.09 dB before)
- Training time: **2-3 hours**

**Checkpoint**: `outputs/nafnet_analysis_maps_w64/nafnet_best.pth`

---

### **Stage 2: Train Adaptive Heads (Frozen Strong Base)** 🔥

**Goal**: Train specialized heads on top of frozen strong base

**Script**: `train_frozen_with_strong_base.sh`

**Configuration**:
- Base NAFNet: **FROZEN** at 30-31 dB (lr=0.0)
- Head quality weight: 5.0 (strong supervision)
- Head diversity weight: 0.3
- Base orthogonality weight: 0.1 (balanced)
- Head consistency weight: 0.01
- Epochs: 50 (with early stopping)

**Expected Results**:
- Base: 30-31 dB (frozen)
- Individual heads: 30-32 dB (all functional)
- Overall PSNR: **32-34 dB**
- Adaptive gain: **2-3 dB**
- Training time: **2-3 hours**

**Checkpoint**: `checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth`

---

### **Stage 3: Fine-Tuning (Optional)** 💎

**Goal**: Final polish by unfreezing base with ultra-low LR

**Script**: `train_finetune_unfrozen.sh` (update base_ckpt path)

**Configuration**:
- Base NAFNet: UNFROZEN (lr=1e-6)
- Resume from Stage 2 checkpoint
- Epochs: 10

**Expected Results**:
- Additional gain: **+0.2-0.5 dB**
- Final PSNR: **33-35 dB**
- Training time: **30 minutes**

---

## Complete Workflow

### Step-by-Step Instructions

#### **Option A: Quick Test (30 minutes)**
```bash
# 1. Test base NAFNet training (5 epochs, ~10 min)
bash test_nafnet_base.sh
# Expected: Val PSNR ~28-29 dB

# 2. Test frozen training with test base (2 epochs, ~5 min)
bash test_frozen_with_strong_base.sh
# Expected: All heads functional, PSNR improvement

# 3. If tests pass, proceed to full training
```

#### **Option B: Full Training (4-6 hours)**
```bash
# 1. Train strong base NAFNet (100 epochs, ~2-3 hours)
bash train_nafnet_base_analysis_maps.sh
# Expected: Val PSNR 30-31 dB
# Output: outputs/nafnet_analysis_maps_w64/nafnet_best.pth

# 2. Train adaptive heads with frozen strong base (~2-3 hours)
bash train_frozen_with_strong_base.sh
# Expected: Overall PSNR 32-34 dB
# Output: checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth

# 3. (Optional) Fine-tune with unfrozen base (~30 min)
bash train_finetune_unfrozen.sh  # (update base_ckpt path first)
# Expected: Final PSNR 33-35 dB
```

---

## Performance Projections

| Stage | Base PSNR | Adaptive Gain | Overall PSNR | Time | Cumulative Time |
|-------|-----------|---------------|--------------|------|-----------------|
| **Baseline (old base)** | 27.09 dB | 1-2 dB | **28-29 dB** | - | - |
| **Stage 1: Strong Base** | 30-31 dB | - | **30-31 dB** | 2-3h | 2-3h |
| **Stage 2: Frozen Heads** | 30-31 dB | 2-3 dB | **32-34 dB** | 2-3h | 4-6h |
| **Stage 3: Fine-tuning** | 31-32 dB | 2-3 dB | **33-35 dB** | 30min | 4.5-6.5h |

**Total improvement**: **+5-6 dB** over baseline (28 → 34 dB)

---

## Why This Works

### 1. **Domain-Matched Base**
- Old base trained on general Duke OCT
- New base trained on analysis maps with Dirichlet noise
- **Result**: +3-4 dB base improvement (27 → 30-31 dB)

### 2. **Frozen Base Forces Specialization**
- With weak base (27 dB), heads had to do heavy lifting
- With strong base (30-31 dB), heads can specialize on residual patterns
- **Result**: Better adaptive gain (1-2 → 2-3 dB)

### 3. **Balanced Loss Weights**
- Quality loss (5.0) >> Orthogonality loss (0.1)
- Heads learn to improve first, specialize second
- **Result**: All 4 heads functional (no catastrophic failures)

---

## Key Insights

### Why Retrain Base (vs Fine-tune)?

| Approach | Base PSNR | Reason |
|----------|-----------|---------|
| **Use existing** | 27.09 dB | Different dataset, 3 dB gap |
| **Fine-tune existing** | 29-30 dB | Partial adaptation, 1-2 dB gap |
| **Retrain from scratch** ⭐ | 30-31 dB | Full adaptation, optimal |

**Verdict**: Retraining is worth the extra 2 hours for +3-4 dB improvement

### Loss Weight Hierarchy (Lessons Learned)

```python
# CORRECT (prevents catastrophic failures):
head_quality_weight = 5.0      # Priority 1: Quality
head_diversity_weight = 0.3     # Priority 2: Specialization
base_orthogonality_weight = 0.1 # Priority 3: Divergence (gentle nudge)
head_consistency_weight = 0.01  # Priority 4: Stability

# WRONG (causes heads to make outputs worse):
head_quality_weight = 2.0       # Too weak!
base_orthogonality_weight = 0.3 # Too strong! Dominates quality
# Result: Shot/Banding heads produced 18-20 dB (worse than 22 dB noisy!)
```

**Rule of thumb**: Quality weight should be **50x** higher than orthogonality

---

## Monitoring Training

### Stage 1 (Base NAFNet):
**Good indicators**:
- Epoch 1: ~26 dB
- Epoch 20: ~29 dB
- Epoch 50+: 30-31 dB (converged)
- Training loss decreasing smoothly

**Bad indicators**:
- PSNR not improving after 20 epochs
- Validation PSNR < 29 dB at convergence
- Training loss oscillating

### Stage 2 (Frozen Heads):
**Good indicators**:
- Base PSNR stays constant (frozen correctly)
- All 4 heads: 30-32 dB (all functional)
- Head similarity: 0.2-0.4 (good diversity)
- Overall PSNR: 32-34 dB

**Bad indicators**:
- Any head < 27 dB (catastrophic failure)
- Head similarity > 0.7 (collapse)
- Overall PSNR < 31 dB (weak specialization)

---

## Troubleshooting

### Stage 1: Base training stuck at 28 dB
**Causes**:
- Learning rate too high/low
- Insufficient data augmentation
- Model capacity mismatch

**Solutions**:
- Adjust LR: Try 5e-4 or 2e-3
- Check data loading (print first batch)
- Try width=32 or width=96

### Stage 2: Heads not specializing
**Symptoms**: All heads produce similar outputs (similarity >0.7)

**Solutions**:
- Increase diversity weight (try 0.5)
- Increase orthogonality weight (try 0.15)
- Check that base is actually frozen (look for "🔒" message)

### Stage 2: Heads making outputs worse
**Symptoms**: Individual heads < 27 dB (worse than base)

**Solutions**:
- REDUCE orthogonality weight (try 0.05)
- INCREASE quality weight (try 7.0)
- Check loss balance in training logs

---

## Files Overview

| File | Purpose |
|------|---------|
| **Training Scripts** | |
| `train_nafnet_base_analysis_maps.sh` | Stage 1: Train strong base (100 epochs) |
| `train_frozen_with_strong_base.sh` | Stage 2: Train heads with frozen strong base |
| `train_finetune_unfrozen.sh` | Stage 3: Fine-tune (update base_ckpt path) |
| **Test Scripts** | |
| `test_nafnet_base.sh` | Quick test for Stage 1 (5 epochs) |
| `test_frozen_with_strong_base.sh` | Quick test for Stage 2 (2 epochs) |
| **Python Scripts** | |
| `nsnd_oct/scripts/train_nafnet_on_analysis_maps.py` | NAFNet training implementation |
| **Checkpoints** | |
| `outputs/nafnet_analysis_maps_w64/nafnet_best.pth` | Stage 1 output (strong base) |
| `checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth` | Stage 2 output (final model) |
| **Documentation** | |
| `STRATEGY_STRONG_BASE.md` | This document |
| `ANALYSIS_ORTHOGONALITY_ISSUE.md` | Why we fixed loss weights |
| `TWO_STAGE_TRAINING_STRATEGY.md` | Original two-stage strategy |

---

## Quick Decision Guide

**Q: Should I retrain base or use existing?**
- Want best performance (33-35 dB)? → **Retrain** ⭐
- Need quick results (28-29 dB)? → Use existing

**Q: Should I test first or run full training?**
- First time? → **Test first** (30 min)
- Confident in setup? → Full training (4-6 hours)

**Q: Is Stage 3 (fine-tuning) necessary?**
- Stage 2 achieves 32-34 dB → Usually sufficient
- Want to squeeze out last 0.5 dB? → Run Stage 3

---

## Expected Timeline

### Conservative (with testing):
```
Day 1:
- 10:00: Test base training (10 min)
- 10:15: Test frozen training (5 min)
- 10:30: Start full base training (3 hours)
- 13:30: Lunch break
- 14:00: Start frozen head training (3 hours)
- 17:00: Results analysis
Total: ~7 hours (with breaks)

Day 2 (optional):
- 09:00: Fine-tuning (30 min)
- 09:30: Final evaluation
Total: ~1 hour
```

### Aggressive (direct to full training):
```
Single session:
- Start base training (3 hours)
- Start frozen training (3 hours)
- (Optional) Fine-tuning (30 min)
Total: 6.5 hours (can run overnight)
```

---

## Recommended Workflow

1. ✅ **Test base training** (10 min)
   ```bash
   bash test_nafnet_base.sh
   ```

2. ✅ **Run full base training** (3 hours)
   ```bash
   bash train_nafnet_base_analysis_maps.sh
   ```

3. ✅ **Test frozen training** (5 min)
   ```bash
   bash test_frozen_with_strong_base.sh
   ```

4. ✅ **Run full frozen training** (3 hours)
   ```bash
   bash train_frozen_with_strong_base.sh
   ```

5. ✅ **Evaluate results**
   - Check overall PSNR (should be 32-34 dB)
   - Check individual heads (all should be GOOD)
   - Check adaptive gain (should be 2-3 dB)

6. ⭐ **(Optional) Fine-tuning** (30 min)
   ```bash
   # Update base_ckpt path in train_finetune_unfrozen.sh first
   bash train_finetune_unfrozen.sh
   ```

**Total time**: 6-7 hours for complete pipeline

**Total improvement**: +5-6 dB (28 → 33-34 dB)
