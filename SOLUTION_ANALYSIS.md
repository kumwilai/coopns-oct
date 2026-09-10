# Real Problem Analysis with Correct Base

## Test Results with Correct Strong Base (30.08 dB)

### Configuration Tested
- Residual head width: 32 (50% of base width=64)
- Base orthogonality: 0.0
- Head quality weight: 5.0
- Base NAFNet: outputs/nafnet_analysis_maps_w64/nafnet_best.pth

### Results (Epoch 1)
```
Base NAFNet PSNR: 30.08 dB
Overall PSNR: 28.70 dB
Adaptive Gain: -1.38 dB ❌ NEGATIVE!

Individual Heads:
- speckle:  27.40 dB → BAD
- gaussian: 26.87 dB → BAD
- shot:     27.95 dB → BAD
- banding:  26.50 dB → BAD
```

## Root Cause

**Capacity AND Initialization Problem:**

1. **Randomly initialized heads start at ~0 dB (noise)**
2. **After 1 epoch**: Heads reach 26-28 dB (learning, but still below base)
3. **Base is 30 dB**: Heads are 2-3 dB below base
4. **Blending formula**: `overall = 0.6 × heads + 0.4 × base`
   - When heads (27 dB) < base (30 dB), overall drops to 28.70 dB
   - Adaptive gain = 28.70 - 30.08 = **-1.38 dB**

## Why Width=32 Failed

Even with doubled capacity (16→32), heads can't reach 30 dB in just 3 epochs when:
- Starting from random initialization (~0 dB)
- Base is very strong (30 dB)
- Training time is limited

## Solution Strategy

To achieve positive adaptive gain with 30 dB base, we need:

### Option 1: EQUAL CAPACITY HEADS (Recommended)
**Make heads same width as base (64)**
- Heads have equal capacity to base
- Can match base performance (30 dB)
- Then specialize to improve beyond base
- **Most straightforward solution**

### Option 2: INITIALIZE FROM BASE
**Copy base weights to initialize each head**
- Heads start at ~30 dB (same as base)
- Then fine-tune to specialize
- Requires modifying initialization code
- **Most efficient, but requires code changes**

### Option 3: MUCH LONGER TRAINING
**Train width=32 heads for 50+ epochs**
- Eventually heads might reach 30 dB
- Very slow and uncertain
- **Least reliable**

## Recommended Solution

**Multi-pronged approach:**
1. ✅ **Increase head width to 64** (match base capacity)
2. ✅ **Remove orthogonality constraint** (already done: 0.0)
3. ✅ **Strong quality supervision** (already done: 5.0)
4. ✅ **Train for 50 epochs** (not just 3)
5. ✅ **Use correct base checkpoint**

With width=64 heads:
- Heads can match 30 dB base performance
- Specialization allows improvement beyond base
- Positive adaptive gain achievable

## Implementation

Create new test with:
- `--residual_head_width 64` (was 32)
- `--base_ckpt outputs/nafnet_analysis_maps_w64/nafnet_best.pth` (correct base)
- `--epochs 50` (full training)
- Keep other fixes (orthogonality=0.0, quality=5.0)
