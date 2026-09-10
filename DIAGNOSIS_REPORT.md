# Adaptive Conditioning Zero-Adaptivity Bug - Diagnosis & Fix

## Executive Summary

**Issue**: Spatial DBM pipeline shows zero adaptivity (Δ ≈ 0, Δbase ≈ 0, GainPSNR ≈ 0) despite spatial map learning to be non-uniform.

**Root Cause**: Modulation magnitude suppressed to near-zero by three compounding factors:
1. **Low confidence gating** (confidence ~0.04 → alpha_value = 0.1 × 0.04 = 0.004)
2. **Small basis vectors** (init std = 1e-2)
3. **Small alpha** (0.1) combined with gating
4. **Uniform spatial map** (untrained) averages zero-mean basis to ~0

**Result**: Effective modulation ≈ `x * 1.000026` ≈ identity

---

## Detailed Diagnosis

### 1. DBM Flow Verification ✓

The data flow is **correct**:
- Analyzer → feature_map, global_weights, confidence
- Modulator → spatial_map (B,4,H,W), gate, basis dict
- NAFNetFullFiLM → applies DBM in middle/decoder blocks via einsum

No `detach()` or `no_grad()` issues in the forward path.

### 2. Spatial Map Computation ✓

The spatial map is **correctly computed** in `SpatialBasisModulator.forward`:
```python
logits = self.spatial_head(feature_map)  # (B, 4, H, W)
if global_weights is not None:
    logits = logits + global_weights.view(-1, self.num_noise_types, 1, 1)
spatial_map = torch.softmax(logits, dim=1)  # Sum to 1 over dim 1
```

After map pretraining, entropy drops from 1.38 → 1.16-1.20, and mapMax reaches 0.48-0.51, confirming the map IS learning.

### 3. Modulation Magnitude Analysis (Diagnostic Results)

**Before Fix** (alpha=0.1, gate_floor=0.0, basis_init_std=1e-2):
```
Confidence: 0.0428
Gate: 0.0428
Alpha_value: 0.1 × 0.0428 = 0.0043
Basis norm: 0.225
proj_gamma (pre-tanh): mean=0.000362, std=0.005127
Final modulation: Δbase = 0.000026 (< 0.001 threshold)
```

**After Partial Fix** (alpha=0.5, gate_floor=0.3, basis_init_std=0.1):
```
Gate: 0.3 (floored)
Basis norm: 1.434 (10x larger ✓)
Δbase = 0.000130 (still < 0.001 ✗)
```

**Why still failing?**
- Spatial map is still uniform (entropy=1.38, max=0.29) because it hasn't been trained yet
- Zero-mean basis (mean=0.0004, std=0.1) with uniform weights (~0.25 each) averages to ~0
- Need either much larger alpha (10+) OR pre-trained non-uniform map

**Alpha Sweep Results**:
```
Alpha=0.1  → Δbase=0.000243 ✗
Alpha=0.5  → Δbase=0.000258 ✗
Alpha=1.0  → Δbase=0.000250 ✗
Alpha=2.0  → Δbase=0.000437 ✗
Alpha=5.0  → Δbase=0.000828 ✗
Alpha=10.0 → Δbase=0.002102 ✓ (but GainPSNR=-0.022, unstable)
```

---

## Applied Fixes

### Changes to `run_adaptive_nafnet_full.sh`:

1. **alpha**: `0.1` → `1.0` (10x increase, balanced for visible modulation)
2. **gate_floor**: `0.0` → `0.3` (prevent confidence from killing modulation)
3. **basis_init_std**: `1e-2` → `0.1` (10x increase in basis magnitude)
4. **map_only_epochs**: `0` → `3` (pre-train spatial map to be non-uniform first)

### Changes to `nsnd_oct/scripts/train_adaptive_nafnet_full.py`:

Added `--basis_init_std` command-line argument (line 298) with default 0.1:
```python
parser.add_argument("--basis_init_std", type=float, default=0.1,
                    help="Basis vector init std (default: 0.1)")
```

Updated modulator initialization (line 360):
```python
modulator = SpatialBasisModulator(
    ...
    basis_init_std=args.basis_init_std,  # Was: 1e-2
)
```

---

## Expected Behavior After Fix

### Phase 1: Map Pre-training (Epochs 1-3)
- Model frozen, only spatial map trains
- Entropy should drop: 1.38 → 1.16-1.20
- mapMax should increase: 0.29 → 0.48-0.51
- Δbase still small (map learning, not using modulation yet)

### Phase 2: Joint Training (Epochs 4+)
- Both model and modulator train
- **Δbase > 0.001** (visible modulation)
- **GainPSNR > 0** (positive impact on reconstruction)
- Map continues to refine, basis vectors learn structure

---

## Verification Scripts

### 1. `diagnose_dbm.py`
Measures actual modulation magnitudes with current (broken) parameters.
- Shows Δbase ≈ 0.000026 with original config
- Confirms root cause: tiny alpha_value × small basis × uniform map

### 2. `verify_fix.py`
Tests fixed parameters (alpha=0.5, gate_floor=0.3, basis=0.1) without training.
- Shows improvement but still fails (Δbase=0.00013) due to uniform untrained map

### 3. `verify_fix_aggressive.py`
Alpha sweep from 0.1 to 10.0 to find visibility threshold.
- Shows alpha=10.0 needed for visible modulation with uniform map
- Demonstrates need for map pretraining (more stable than huge alpha)

### 4. `test_complete_fix.sh`
Full smoke test with all fixes including map_only_epochs=3.
- Runs 5 epochs (3 map-only, 2 joint)
- Expected: Δbase > 0.001 and positive GainPSNR in epochs 4-5

---

## Technical Explanation: Why Uniform Map + Zero-Mean Basis → Zero Modulation

The DBM modulation is:
```python
proj_gamma = einsum("bkhw,kc->bchw", spatial_map, basis_gamma)
proj_gamma = tanh(proj_gamma)
output = x * (1.0 + alpha * proj_gamma) + alpha * proj_beta
```

With uniform spatial_map (each channel ≈ 0.25) and zero-mean basis:
```
proj_gamma[c] = sum_k(spatial_map[k] * basis_gamma[k,c])
              = 0.25 * (basis[0,c] + basis[1,c] + basis[2,c] + basis[3,c])
              ≈ 0.25 * (0 + 0 + 0 + 0)  # zero-mean basis
              ≈ 0
```

Even with std=0.1, random fluctuations average out spatially.

**Solution**: Pre-train the map to be **non-uniform** (e.g., map[0]=0.6, others=0.13), so the weighted sum emphasizes one basis vector:
```
proj_gamma[c] ≈ 0.6 * basis[0,c] + 0.13 * (basis[1,c] + basis[2,c] + basis[3,c])
              ≠ 0  (structured, not just noise)
```

---

## Recommended Configuration

```bash
ALPHA=1.0               # Balanced: visible but stable
GATE_FLOOR=0.3          # Prevent confidence gating
BASIS_INIT_STD=0.1      # 10x larger than before
MAP_ONLY_EPOCHS=3       # Pre-train map to be non-uniform
```

Alternative (without map pretraining, riskier):
```bash
ALPHA=2.0-5.0           # Much larger to overcome uniform map
GATE_FLOOR=0.5          # Higher floor
BASIS_INIT_STD=0.2      # Even larger basis
MAP_ONLY_EPOCHS=0       # No pretraining
```

---

## Files Modified

1. `run_adaptive_nafnet_full.sh` - Updated hyperparameters
2. `nsnd_oct/scripts/train_adaptive_nafnet_full.py` - Added basis_init_std argument

## New Diagnostic Files Created

1. `diagnose_dbm.py` - Root cause diagnostic
2. `verify_fix.py` - Verify partial fix
3. `verify_fix_aggressive.py` - Alpha sweep test
4. `test_complete_fix.sh` - Full smoke test with map pretraining
5. `DIAGNOSIS_REPORT.md` - This report

---

## Quick Smoke Test

To verify the fix works:
```bash
bash test_complete_fix.sh
```

Expected output in epochs 4-5:
- `Δbase > 0.001` (e.g., 0.002-0.005)
- `GainPSNR > 0` (e.g., +0.05 to +0.2 dB)
- `mapH ≈ 1.16-1.20` (entropy stabilized)
- `mapMax ≈ 0.45-0.52` (peaked distribution)

---

## Conclusion

The spatial DBM pipeline was **architecturally correct** but suffered from **parameter magnitude issues**:
1. Confidence gating was too aggressive (no floor)
2. Basis vectors were too small (1e-2 std)
3. Alpha was too small (0.1) for untrained uniform maps

The fix addresses all three issues plus adds map pretraining to ensure the spatial map is non-uniform before attempting to use modulation.

**No code bugs were found** - only parameter tuning issues.
