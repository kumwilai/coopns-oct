# Adaptive NAFNet Training Analysis

## Current Training Status

### Memory Leaks: ✅ FIXED
- Original training crashed at epoch 2 with exit code 137 (OOM kill)
- Fixed training successfully completed epoch 1 and is progressing through epoch 2
- Memory is stable - no accumulation or crashes

### Denoising Performance: ⚠️ UNDER INVESTIGATION

## Epoch 1 Results Analysis

```
Epoch 1 Complete |
  Train: loss=0.1844 map=0.1844 usage=0.0000
  Val: PSNR=30.61 SSIM=0.8687
       NoisyPSNR=22.96 NoisySSIM=0.4984
       BasePSNR=30.60 BaseSSIM=0.8687
       GainPSNR=+0.010 GainSSIM=-0.0001
       Δbase=0.0006
```

### Key Observations:

1. **Recon Loss = 0.0** - Model denoising network is frozen (expected in map-only phase)
2. **GainPSNR = +0.010 dB** - Almost no improvement over base model
3. **Δbase = 0.0006** - Adaptive output nearly identical to base model
4. **Usage Loss = 0.0** - Not yet active (model frozen)

### Why This Is Expected:

The training uses a **two-phase approach**:

#### Phase 1: MAP_ONLY_EPOCHS (Epochs 1-3)
**Status:** Currently in this phase
**What's happening:**
- Main NAFNet model is **FROZEN** (`model.eval()`, gradients disabled)
- Only the **spatial noise map predictor** is being trained
- Goal: Learn to predict accurate spatial noise distributions
- Performance: Should match base model exactly (no adaptation yet)

**Configuration:**
```bash
MAP_ONLY_EPOCHS=3
MAP_ONLY_LOSS=1.0
```

#### Phase 2: FULL ADAPTIVE TRAINING (Epochs 4-50)
**Status:** Starts at epoch 4
**What should happen:**
- Main NAFNet model **UNFROZEN** and starts training
- Spatial modulation becomes active
- Usage loss enforces the model to react to conditioning
- Model should start showing improvement over base model

**Expected metrics after epoch 4:**
- `Recon > 0.0` (reconstruction loss active)
- `usage > 0.0` (usage loss active)
- `GainPSNR > 0` (improvement over base)
- `Δbase > 0.01` (adaptive output differs from base)

## Training Configuration Analysis

### Potential Issues:

1. **Base Model Already Strong (30.60 dB)**
   - Less room for improvement
   - Adaptive gains may be modest (1-2 dB typical)

2. **Alpha = 1.0 (Modulation Strength)**
   - Reasonable value
   - Should allow meaningful adaptation

3. **Gate Floor = 0.3**
   - Minimum confidence threshold
   - Prevents over-modulation on uncertain cases

4. **Usage Loss Weight = 0.1**
   - Should be sufficient to enforce adaptation
   - May need tuning if model doesn't adapt

## What to Monitor Starting Epoch 4

### Critical Metrics:

1. **Reconstruction Loss** (`Recon > 0`)
   - Should appear and decrease over epochs
   - Indicates model is learning to denoise

2. **Usage Loss** (`usage > 0`)
   - Should be non-zero
   - Indicates model reacts to different conditioning

3. **Adaptation Delta** (`Δ > 0`)
   - Difference when given correct vs. wrong conditioning
   - Should be > 0.01 for meaningful adaptation

4. **Performance Gain** (`GainPSNR`)
   - Should increase over base model
   - Target: +0.5 to +2.0 dB depending on noise complexity

5. **Base Delta** (`Δbase > 0`)
   - Difference between adaptive and base outputs
   - Should increase, showing model is using conditioning

## Decision Points

### After Epoch 4-5:

**If adaptation is working:**
- `Recon > 0` ✓
- `usage > 0` ✓
- `Δ > 0.01` ✓
- `GainPSNR > +0.3` ✓

→ **Continue training to convergence**

**If adaptation is NOT working:**
- `Recon > 0` ✓ but `Δ ≈ 0` ✗
- `usage ≈ 0` ✗
- `GainPSNR ≈ 0` ✗
- `Δbase ≈ 0` ✗

→ **Potential fixes:**

1. **Increase usage_loss_weight** (0.1 → 0.5)
   ```bash
   USAGE_LOSS=0.5  # Force stronger adaptation
   ```

2. **Increase alpha** (1.0 → 2.0)
   ```bash
   ALPHA=2.0  # Stronger modulation
   ```

3. **Reduce gate_floor** (0.3 → 0.1)
   ```bash
   GATE_FLOOR=0.1  # Allow more aggressive adaptation
   ```

4. **Add noise map supervision weight**
   ```bash
   MAP_LOSS=0.5  # Stronger noise map guidance
   ```

## Expected Timeline

- **Epochs 1-3:** Map-only training (current phase)
  - Performance = Base model
  - No adaptation expected

- **Epochs 4-10:** Adaptation warm-up
  - Performance starts improving
  - Usage loss becomes active
  - Model learns to use conditioning

- **Epochs 10-30:** Main training
  - Performance steadily improves
  - Adaptation becomes stable
  - Best model likely around epoch 20-25

- **Epochs 30-50:** Fine-tuning
  - Marginal improvements
  - Convergence

## Next Steps

1. ✅ **Monitor epoch 3 completion** - Should still show no gains
2. ⏳ **Watch epoch 4 carefully** - Adaptation should activate
3. ⏳ **Analyze epoch 5-6** - Confirm adaptation is working
4. ⏳ **Decision point at epoch 10** - Adjust if needed
5. ⏳ **Let training complete** - If adaptation is working

## Success Criteria

**Training is successful if by epoch 10:**
- GainPSNR > +0.5 dB (showing meaningful improvement)
- Δbase > 0.05 (model outputs differ from base)
- usage > 0.05 (model reacts to conditioning)
- Stable loss curves (no divergence)

**Training needs adjustment if by epoch 10:**
- GainPSNR ≈ 0 (no improvement)
- Δbase ≈ 0 (identical to base)
- usage ≈ 0 (no reaction to conditioning)
- Loss plateaued at map-only level
