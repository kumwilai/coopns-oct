# Architecture Fix for Head Specialization

## Problem Diagnosis

The shared residual architecture prevents head specialization:

### Evidence:
1. **Adapter outputs nearly identical**:
   - Speckle vs Shot similarity: **0.9999** (99.99% identical!)
   - Shot adapter produces **constant output** (std=0.000)

2. **Architectural constraint**:
   - All 4 adapters receive IDENTICAL shared trunk features (96 channels)
   - Adapters have only 2 conv layers with 64 hidden channels
   - Not enough capacity to meaningfully diverge

3. **Expert heads produce outputs nearly identical to base NAFNet**:
   - Speckle differs by 0.0089 from base
   - Banding differs by 0.0038
   - Gaussian differs by 0.0036
   - Shot differs by 0.0078

### Why Diversity Loss Doesn't Work:
- Adapters are architecturally constrained to stay similar
- Operating on shared features limits divergence
- Loss can't overcome architectural limitations

## Solution: Three Options

### Option 1: Independent Heads (Strongest Specialization)

**Remove shared trunk**, give each head its own NAFNet:

```python
# In MultiTaskHybridNSND.__init__()

# Replace SharedResidualAdapterBank with independent heads:
self.residual_heads = nn.ModuleDict({
    "speckle": NAFNetSmall(img_channel=1, width=16),   # Independent
    "banding": NAFNetSmall(img_channel=1, width=16),   # Independent
    "gaussian": NAFNetSmall(img_channel=1, width=16),  # Independent
    "shot": NAFNetSmall(img_channel=1, width=16),      # Independent
})

# In forward():
residual = noisy - base_output
expert_outputs = {
    name: head(residual) for name, head in self.residual_heads.items()
}
```

**Pros:**
- Maximum capacity for specialization
- Each head can learn completely different functions
- Diversity loss will work effectively

**Cons:**
- 4x more parameters (4 separate NAFNets instead of 1 shared trunk + adapters)
- Current: 4.8M params for heads
- New: ~2M params × 4 = 8M params for heads

### Option 2: Larger Adapters (Moderate Specialization)

Keep shared trunk but make adapters **much deeper**:

```python
class SharedResidualAdapterBank(nn.Module):
    def __init__(self, trunk_width=32, adapter_channels=96, adapter_hidden=64):
        super().__init__()
        self.in_proj = nn.Conv2d(1, adapter_channels, 3, padding=1)
        self.trunk = NAFNetSmall(img_channel=adapter_channels, width=trunk_width)

        # MUCH DEEPER adapters (6 layers instead of 2)
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(adapter_channels, adapter_hidden, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adapter_hidden, adapter_hidden, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adapter_hidden, adapter_hidden, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adapter_hidden, adapter_hidden, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adapter_hidden, adapter_hidden//2, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(adapter_hidden//2, 1, 3, padding=1),
            )
            for name in ["speckle", "banding", "gaussian", "shot"]
        })
```

**Pros:**
- More capacity for specialization while sharing trunk
- Moderate parameter increase
- Shared trunk provides common features, adapters specialize

**Cons:**
- Still constrained by shared trunk features
- May not fully solve the collapse problem

### Option 3: Different Architectures Per Noise Type (Best Specialization)

Use specialized architectures for each noise type:

```python
self.residual_heads = nn.ModuleDict({
    "speckle": LogDomainSpeckleHead(width=16),    # Log-domain for multiplicative noise
    "banding": NAFNetSmall(width=16),             # Standard for additive
    "gaussian": NAFNetSmall(width=16),            # Standard for additive
    "shot": SignalDependentHead(width=16),        # VST for Poisson noise
})
```

Where LogDomainSpeckleHead already exists in your code (line 108-119).

**Pros:**
- Architectures matched to noise characteristics
- Natural differentiation (log-domain vs linear-domain)
- Each head specialized by design

**Cons:**
- More complex to maintain
- Different architectures may have different training dynamics

## Recommended Approach

**I recommend Option 1: Independent Heads**

Reasons:
1. Your diagnostic showed 0.998 similarity - extreme collapse
2. Shared trunk is the bottleneck
3. Parameter increase (4.8M → 8M) is acceptable for better performance
4. Diversity loss will work effectively with independent heads

## Implementation Steps

### 1. Backup Current Code
```bash
cp nsnd_oct/scripts/train_hybrid_nsnd_multitask.py nsnd_oct/scripts/train_hybrid_nsnd_multitask.py.backup
```

### 2. Modify Architecture

Replace SharedResidualAdapterBank usage with independent heads:

```python
# Around line 420-450 in MultiTaskHybridNSND.__init__()

if shared_residual:
    # OLD (causes collapse):
    # self.residual_shared = SharedResidualAdapterBank(...)

    # NEW (allows specialization):
    self.residual_heads = nn.ModuleDict({
        "speckle": NAFNetSmall(img_channel=1, width=residual_head_width),
        "banding": NAFNetSmall(img_channel=1, width=residual_head_width),
        "gaussian": NAFNetSmall(img_channel=1, width=residual_head_width),
        "shot": NAFNetSmall(img_channel=1, width=residual_head_width),
    })
else:
    # Existing code for non-shared residual
    pass
```

### 3. Update Forward Pass

Modify the forward method to use independent heads:

```python
# Around line 600-650 in MultiTaskHybridNSND.forward()

if self.use_base_nafnet and self.base_denoiser is not None:
    base_output = self.base_denoiser(noisy)
    residual = noisy - base_output

    # OLD:
    # expert_residuals = self.residual_shared(residual)

    # NEW:
    expert_residuals = {
        name: head(residual) for name, head in self.residual_heads.items()
    }

    expert_outputs = {
        name: base_output + res for name, res in expert_residuals.items()
    }
```

### 4. Retrain from Scratch

Since architecture changed, you MUST retrain from Phase 1:

```bash
cd /home/kumwilai/OCT

# Phase 1: Noise map pre-training
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  [... Phase 1 args ...]

# Phase 2: Multi-task with diversity loss (NOW WILL WORK)
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --head_quality_weight 0.5 \
  --head_diversity_weight 0.3 \
  --head_consistency_weight 0.05 \
  [... other args ...]
```

## Expected Results After Fix

With independent heads:

### Training Metrics:
- **Head similarity**: Should drop from 0.998 to <0.5 by epoch 20
- **Adaptive gain**: Should increase from 0.05 dB to 0.8-1.5 dB
- **Individual head PSNR**: Competitive with or exceeding overall PSNR

### Diagnostic Output:
```
Head Diversity (Cosine Similarity Matrix):
          speckle   banding   gaussian  shot
speckle      1.000    0.450    0.520    0.380  ✅ (was 0.999)
banding      0.450    1.000    0.490    0.410  ✅ (was 0.999)
gaussian     0.520    0.490    1.000    0.550  ✅ (was 0.997)
shot         0.380    0.410    0.550    1.000  ✅ (was 1.000)

Average pairwise similarity: 0.472 ✅ (was 0.998)
Adaptive Gain: 1.2 dB ✅ (was 0.127 dB)
```

## Why This Will Work

1. **Independent parameters**: Each head has its own NAFNet - no shared bottleneck
2. **Diversity loss effective**: Can actually create divergence since heads are independent
3. **Routing meaningful**: Different heads will produce genuinely different outputs
4. **Architectural capacity**: 16-width NAFNet has enough capacity to specialize

## Alternative: If You Want to Keep Shared Trunk

If parameter count is critical, try **Option 2** (deeper adapters):
- Increase adapter depth from 2 layers to 6+ layers
- Increase adapter hidden channels from 64 to 128+
- This gives adapters more capacity to diverge from shared features

But Option 1 (independent heads) is strongly recommended for best results.
