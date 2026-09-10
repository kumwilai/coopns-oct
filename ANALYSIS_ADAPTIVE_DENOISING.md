# Analysis: Weak Adaptive Denoising Performance

## Problem Statement

Despite achieving 78.5% Top-1 noise type classification accuracy (Phase 2B), the adaptive denoising heads provide minimal gain over the base NAFNet:

- **Base NAFNet PSNR**: 33.08-33.27 dB
- **Overall PSNR**: 33.17-33.36 dB
- **Adaptive Gain**: Only ~0.09-0.29 dB improvement

Many individual heads show "BAD" performance (performing worse than overall PSNR), indicating poor specialization.

## Root Causes Identified

### 1. **Dominant Base NAFNet**
```
Issue: Base NAFNet is too strong and dominates the output
Evidence: Base PSNR ≈ Overall PSNR
Root Cause: residual_blend_logit starts at 0.1 → heads contribute only ~10%
```

**Impact**: Heads learn weak corrections that get suppressed by the blend weight

### 2. **Weak Residual Learning**
```python
# Current blending strategy (train_hybrid_nsnd_multitask.py:602-615)
blend = torch.sigmoid(self.residual_blend_logit)  # starts at ~0.1
expert_outputs = {
    name: base + blend * refined[name]  # heads contribute only 10%
    for name in ["speckle", "banding", "gaussian", "shot"]
}
```

**Issue**: The heads predict residuals that are heavily damped by the small blend weight

### 3. **Conservative Spatial Refiner Initialization**
```python
# Current initialization (train_hybrid_nsnd_multitask.py:190-205)
nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Too small!
nn.init.zeros_(self.spatial_predictor[-1].weight)  # Starts from zero
```

**Issue**: The spatial refiner is initialized to output near-zero corrections, preventing effective learning

### 4. **Insufficient Head Specialization Incentive**
```
Current losses:
- Denoise loss: Optimizes overall output
- Interp loss: Encourages prediction accuracy
- NoiseMap loss: Supervises spatial weight maps (Phase 2B only)

Missing:
- No individual head quality loss
- No specialization regularization
- No diversity enforcement
```

**Impact**: Heads don't learn to specialize because they're only supervised through the blended output

### 5. **Spatial Refiner Training Issues**
```
Phase 1: noise_map_loss_weight = 0.15 (reasonable)
Phase 2B: noise_map_loss_weight = 0.05 (too small!)

Effect: Spatial refiner doesn't get enough supervision to refine weights effectively
```

## Proposed Fixes

### Fix 1: **Increase Residual Blend Init Weight**
```python
# Change in train_hybrid_nsnd_multitask.py
# Old: residual_blend_init = 0.1
# New: residual_blend_init = 0.3  # Start with 30% head contribution

parser.add_argument("--residual_blend_init", type=float, default=0.3,
                    help="Initial residual blend weight (0=base only, 1=heads only)")
```

**Rationale**: Allow heads to contribute more significantly from the start

### Fix 2: **Add Per-Head Quality Loss**
```python
def compute_head_quality_loss(expert_outputs, clean, weights_dict):
    """
    Encourage individual heads to produce high-quality outputs for their noise type.
    """
    head_losses = {}
    for name in ["speckle", "banding", "gaussian", "shot"]:
        head_output = expert_outputs[name]
        head_loss = F.l1_loss(head_output, clean)
        # Weight by confidence in this noise type
        weight = weights_dict[name].mean()
        head_losses[name] = weight * head_loss

    # Total: average of weighted head losses
    return sum(head_losses.values()) / len(head_losses), head_losses
```

**Rationale**: Directly supervise each head's output quality

### Fix 3: **Enhance Spatial Refiner Initialization**
```python
# Change in train_hybrid_nsnd_multitask.py SpatialWeightRefiner
# Old: gain=0.1 (too conservative)
# New: gain=0.5 (balanced)

for layer in self.feature_encoder:
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Increased
```

**Rationale**: Allow spatial refiner to learn faster

### Fix 4: **Add Head Diversity Loss**
```python
def compute_head_diversity_loss(expert_outputs):
    """
    Encourage heads to produce diverse outputs (specialize differently).
    """
    outputs = [expert_outputs[k] for k in ["speckle", "banding", "gaussian", "shot"]]

    diversity_loss = 0.0
    count = 0
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            # Penalize similar outputs
            similarity = F.cosine_similarity(
                outputs[i].flatten(1),
                outputs[j].flatten(1),
                dim=1
            ).mean()
            diversity_loss += torch.relu(similarity - 0.5)  # Encourage < 0.5 similarity
            count += 1

    return diversity_loss / count if count > 0 else 0.0
```

**Rationale**: Prevent heads from collapsing to similar solutions

### Fix 5: **Increase NoiseMap Loss Weight in Phase 2B**
```bash
# In run_duke_region_focused.sh Phase 2B
# Old: --noise_map_loss_weight 0.05
# New: --noise_map_loss_weight 0.1  # Keep supervision strong
```

**Rationale**: Maintain spatial refiner supervision during fine-tuning

### Fix 6: **Add Adaptive Blend Weight Warmup**
```python
def apply_blend_warmup(epoch, max_epochs, init_weight=0.3, target_weight=0.7):
    """
    Gradually increase blend weight to allow heads to specialize.
    """
    progress = min(epoch / (max_epochs * 0.5), 1.0)  # First half of training
    blend_target = init_weight + (target_weight - init_weight) * progress
    return blend_target
```

**Rationale**: Start conservative, gradually trust heads more as they learn

## Implementation Priority

**High Priority (Do First):**
1. Fix 1: Increase residual_blend_init to 0.3
2. Fix 2: Add per-head quality loss (weight=0.5)
3. Fix 5: Increase noise_map_loss_weight to 0.1 in Phase 2B

**Medium Priority:**
4. Fix 3: Enhance spatial refiner initialization
5. Fix 4: Add head diversity loss (weight=0.1)

**Low Priority:**
6. Fix 6: Add blend weight warmup (optional, for advanced control)

## Expected Results

After fixes:
- **Individual heads** should show significant quality (not "BAD")
- **Adaptive gain** should increase to 0.5-1.0 dB over base NAFNet
- **Spatial weights** should show more meaningful spatial variation
- **Overall PSNR** should reach 33.5-34.0 dB (Phase 2B)

## Diagnostic Commands

To verify improvements, check:
1. Head effectiveness in logs (should be "GOOD" not "BAD")
2. Residual blend weight: `sigmoid(blend_logit)` should be 0.3-0.7
3. Spatial weight maps: Should show clear spatial patterns (visualize)
4. Individual head PSNR: Each head should be competitive with overall PSNR
