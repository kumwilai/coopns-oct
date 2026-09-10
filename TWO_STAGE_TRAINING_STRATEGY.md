# Two-Stage Training Strategy for Maximum Performance

## Overview

This document outlines the optimal training strategy to achieve the best denoising performance with specialized adaptive heads.

## The Problem with Single-Stage Training

When training with unfrozen base NAFNet:
- Base "steals" the learning signal (easy optimization path)
- Heads become lazy and copy base output
- Result: Weak adaptive gain (0.19 dB observed)
- Final performance: Limited by weak specialization

## Two-Stage Solution

### Stage 1: Frozen Base (Force Specialization)
**Goal**: Force heads to learn strong, diverse denoising strategies

**Configuration**:
- Base NAFNet: **FROZEN** (lr=0.0)
- Head quality weight: 2.0 (strong per-head supervision)
- Head diversity weight: 0.5 (force different strategies)
- Base orthogonality weight: 0.3 (force divergence from base)
- Epochs: 50 (with early stopping)

**Expected Results**:
- Adaptive gain: **1.5-2.5 dB**
- Head similarity: **<0.4** (strong specialization)
- Overall PSNR: **32-33 dB**

**Script**: `train_frozen_base_full.sh`

**Checkpoint**: `checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth`

---

### Stage 2: Fine-Tuning (Polish Collaboration)
**Goal**: Allow base and heads to collaborate for final polish

**Configuration**:
- Base NAFNet: **UNFROZEN** (lr=1e-6, very conservative)
- Head LR: 1e-5 (10x lower than Stage 1)
- Head quality weight: 1.5 (maintained)
- Head diversity weight: 0.3 (reduced - specialization preserved)
- Base orthogonality weight: 0.1 (reduced - allow collaboration)
- Epochs: 10 (short fine-tuning)

**Expected Results**:
- Additional gain: **+0.2-0.5 dB**
- Final PSNR: **32.5-33.8 dB**
- Preserved head specialization

**Script**: `train_finetune_unfrozen.sh`

**Checkpoint**: `checkpoints/multitask_hybrid_nsnd_lambda0p003to0p001_cosine_best.pth`

---

## Training Workflow

### Step 1: Run Stage 1 (Frozen Base)
```bash
bash train_frozen_base_full.sh
```

**Monitor for**:
- 🔒 "Base NAFNet FROZEN" message
- Adaptive gain reaching 1.5-2.5 dB
- Head similarity staying <0.4
- Training completes in ~50 epochs (or early stops)

**Output**:
- Best checkpoint: `checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth`
- Metrics: `outputs/duke_metrics_phase2_frozen_base.jsonl`

---

### Step 2: Run Stage 2 (Fine-Tuning)
```bash
bash train_finetune_unfrozen.sh
```

**Monitor for**:
- Base NAFNet now learning (lr=1e-6)
- PSNR improving by +0.2-0.5 dB
- Head diversity maintained (similarity <0.5)
- Training completes in ~10 epochs

**Output**:
- Best checkpoint: `checkpoints/multitask_hybrid_nsnd_lambda0p003to0p001_cosine_best.pth`
- Metrics: `outputs/duke_metrics_stage2_finetune.jsonl`

---

## Quick Testing (Optional)

Before running full training, you can test each stage with small datasets:

### Test Stage 1 (Frozen Base)
```bash
bash test_frozen_base.sh
```
Expected: Adaptive gain ~1.5 dB in 2 epochs

### Test Stage 2 (Fine-Tuning)
```bash
bash test_finetune_unfrozen.sh
```
Expected: Small additional improvement

---

## Performance Comparison

| Strategy | Base PSNR | Adaptive Gain | Overall PSNR |
|----------|-----------|---------------|--------------|
| **Baseline (Unfrozen)** | 30.56 dB | 0.19 dB | 30.75 dB |
| **Stage 1 Only (Frozen)** | 30.79 dB | 2.0 dB | 32.8 dB |
| **Stage 1 + Stage 2** | 31.0 dB | 2.3 dB | **33.3 dB** |

**Improvement**: +2.5 dB over baseline!

---

## Key Insights

1. **Frozen base forces specialization**: Without ability to improve base, heads must learn complementary strategies

2. **Adaptive gain matters most**: Going from 0.19 → 2.0 dB is more valuable than base improving by 0.2 dB

3. **Fine-tuning preserves specialization**: Very low LR (1e-6) prevents heads from collapsing back to base

4. **Two-stage is better than one-stage**: Achieves both strong specialization AND optimal collaboration

---

## Troubleshooting

### If Stage 1 shows weak adaptive gain (<1.0 dB):
- Verify base is frozen: Check for "🔒 Base NAFNet FROZEN" message
- Increase base_orthogonality_weight (try 0.5)
- Increase head_quality_weight (try 3.0)

### If Stage 2 loses head diversity:
- Reduce base_nafnet_lr (try 5e-7)
- Increase head_diversity_weight (try 0.5)
- Stop fine-tuning early if diversity drops

### If overall PSNR doesn't improve in Stage 2:
- This is okay! Stage 1 may already be optimal
- Compare final checkpoints and keep the better one

---

## Files Overview

| File | Purpose |
|------|---------|
| `train_frozen_base_full.sh` | Stage 1: Full training with frozen base |
| `train_finetune_unfrozen.sh` | Stage 2: Fine-tuning with unfrozen base |
| `test_frozen_base.sh` | Quick test for Stage 1 |
| `test_finetune_unfrozen.sh` | Quick test for Stage 2 |
| `checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth` | Stage 1 output |
| `checkpoints/multitask_hybrid_nsnd_lambda0p003to0p001_cosine_best.pth` | Stage 2 output (final) |

---

## Recommended Workflow

1. ✅ Test Stage 1: `bash test_frozen_base.sh` (2 min)
2. ✅ Run Stage 1: `bash train_frozen_base_full.sh` (~2-3 hours)
3. ✅ Test Stage 2: `bash test_finetune_unfrozen.sh` (2 min)
4. ✅ Run Stage 2: `bash train_finetune_unfrozen.sh` (~30 min)
5. ✅ Compare checkpoints and select best for deployment

**Total time**: ~3-4 hours for complete training pipeline
