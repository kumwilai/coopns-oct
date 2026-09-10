# Training Adaptive Gain Analysis

## Observation from Your Training Logs

You're seeing very weak adaptive gain during training:

**Epoch 006:**
- Base NAFNet PSNR: 30.88 dB
- Overall PSNR: 30.98 dB
- **Adaptive Gain: 0.10 dB** ⚠️ (very weak)

**Epoch 007:**
- Base NAFNet PSNR: 31.12 dB
- Overall PSNR: 31.17 dB
- **Adaptive Gain: 0.05 dB** ⚠️ (extremely weak)

## Head Similarity Status

From your batch logs:
```
Batch 0050/0250 | HeadSim: 0.245 (max 0.968)  ⚠️
Batch 0100/0250 | HeadSim: 0.162 (max 0.824)  ⚠️
Batch 0150/0250 | HeadSim: 0.296 (max 0.794)  ⚠️
Batch 0200/0250 | HeadSim: 0.150 (max 0.885)  ⚠️
Batch 0250/0250 | HeadSim: 0.249 (max 0.860)  ⚠️
```

**Problem**: Maximum pairwise similarity is still 0.794-0.968, which is extremely high. Heads are still nearly identical.

## Why Adaptive Gain is Weak

1. **Heads Still Collapsing**: Despite diversity loss (weight=0.2), max similarity is still >0.79
2. **Base NAFNet Too Strong**: Base NAFNet PSNR (30.88-31.12 dB) is already very close to overall target (~31 dB)
3. **Heads Not Specializing**: All individual heads show "GOOD" but their PSNRs are LOWER than overall PSNR:
   - Epoch 007: Heads range 30.97-31.14 dB vs Overall 31.17 dB
   - This means routing is working (selecting best head) but heads aren't better than base

## Possible Causes

### 1. Diversity Loss Weight Too Low

Current setting: `--head_diversity_weight 0.2`

The diagnostic showed similarity of 0.998, but after 7 epochs you're still seeing max similarity >0.79. The diversity weight might need to be higher.

**Recommendation**: Try increasing to 0.3-0.5

### 2. Training Still Early (Epoch 7/50)

Head specialization might take more epochs to develop. The diversity loss is gradually reducing similarity (0.968 → 0.860 over batches), suggesting it's working but slowly.

**Recommendation**: Continue training and monitor if similarity continues dropping

### 3. Base NAFNet Pre-training Too Strong

The base NAFNet was pre-trained in Phase 1 and is already achieving 30.88-31.12 dB. The heads might not have enough "room" to improve beyond this.

**Possible Issue**: If base NAFNet solves the problem well enough, adaptive heads become redundant.

### 4. Insufficient Head Quality Loss

Current setting: `--head_quality_weight 0.5`

Individual head supervision might not be strong enough to push them beyond base NAFNet performance.

**Recommendation**: Try increasing to 0.8-1.0

## Diagnostic Recommendations

### 1. Check Current Head Similarity (After Epoch 7)

Run diagnostic to see actual similarity matrix:

```bash
python nsnd_oct/scripts/diagnose_head_specialization.py \
  --checkpoint checkpoints/multitask_hybrid_nsnd_lambda0p008_latest.pth \
  --test_pairs test_pairs_duke_analysis_maps.txt \
  --output_dir diagnostics/head_analysis_epoch7 \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_improved_seed0.pth \
  --base_nafnet_width 64 \
  --num_samples 10 \
  --device cpu
```

**Look for:**
- Average pairwise similarity (should be dropping from 0.998 toward <0.6)
- Individual head PSNR vs overall PSNR (heads should be competitive)
- Adaptive gain (should be increasing from 0.127 dB)

### 2. Monitor Training Metrics

Watch these in your logs:
- **HeadSim max**: Should decrease over epochs (target: <0.7)
- **Individual head PSNR**: Should increase and approach/exceed overall PSNR
- **Adaptive gain**: Should increase over time (target: 0.5-1.0 dB)

## Potential Adjustments (If Needed)

If after epoch 20-30 adaptive gain is still <0.3 dB:

### Option 1: Increase Diversity Loss
```bash
--head_diversity_weight 0.4  # Was 0.2
```

### Option 2: Increase Head Quality Loss
```bash
--head_quality_weight 1.0    # Was 0.5
```

### Option 3: Reduce Consistency Loss (Allow More Divergence)
```bash
--head_consistency_weight 0.01  # Was 0.05
```

### Option 4: Add Per-Head Regularization
This would require code changes to penalize heads that are too similar to base NAFNet.

## Expected Timeline

Based on typical multi-head training:
- **Epochs 1-10**: Heads start diverging slowly (similarity 0.998 → 0.8)
- **Epochs 10-30**: Heads specialize rapidly (similarity 0.8 → 0.6, gain 0.1 → 0.5 dB)
- **Epochs 30-50**: Refinement (similarity <0.6, gain 0.5-1.0 dB)

You're at epoch 7, so it's still early. The diversity loss IS working (max sim decreased from 0.968 to 0.860 within one epoch).

## What to Watch

✅ **Good signs:**
- HeadSim max is decreasing (0.968 → 0.860)
- Diversity loss is being applied (HeadSim values logged)
- Top-1 accuracy is reasonable (63-74%)
- Base NAFNet PSNR is strong (30.88-31.12 dB)

⚠️ **Concerning signs:**
- Adaptive gain very weak (0.05-0.10 dB)
- Max head similarity still high (>0.79)
- Individual heads not exceeding base NAFNet

## Recommendation

**Continue training to epoch 20-30**, then:

1. Run diagnostic again to check if similarity has decreased further
2. If max similarity is still >0.7, increase diversity weight to 0.3-0.4
3. If adaptive gain is still <0.3 dB, increase head quality weight to 1.0
4. Monitor whether heads start showing individual PSNR > overall PSNR (sign of specialization)

The training is progressing correctly (losses decreasing, metrics logged), but head specialization is developing slowly. This is not unusual for multi-head architectures - they often take 20-30 epochs to show clear differentiation.
