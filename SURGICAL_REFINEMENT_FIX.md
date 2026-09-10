# Surgical Refinement Training Stability Fix

## Problem Diagnosed

**Root Cause:** 20x residual amplification causes activation explosion in Swin Transformer before LayerNorm can stabilize.

**Evidence:**
- Loss > 100 or NaN during early training
- Batches being skipped due to abnormal loss
- Gradient clipping (0.1) insufficient to prevent internal explosion

## Solution: 4-Layer Stability Protocol

### 1. Input Pre-Normalization (Immediate Stability)
Add a learnable normalization layer BEFORE the Swin head to tame the 20x input:

```python
class SwinResidualHead(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # NEW: Pre-normalize the amplified input to prevent explosion
        self.input_norm = nn.LayerNorm([in_chans, img_size, img_size])

        # Or use GroupNorm for better stability with small batches:
        # self.input_norm = nn.GroupNorm(num_groups=1, num_channels=in_chans)

        # Rest of architecture unchanged...
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        ...

    def forward(self, x):
        # Normalize BEFORE processing
        x = self.input_norm(x)
        x_first = self.conv_first(x)
        ...
```

**Why this works:**
- Normalizes the 20x amplified input to reasonable range
- LayerNorm preserves relative magnitudes (doesn't destroy the signal)
- Learnable affine parameters allow the head to adapt the scale

### 2. Conservative Initialization for First Conv
Scale down the first convolutional layer to handle large inputs:

```python
def __init__(self, ...):
    ...
    self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

    # NEW: Scale down initialization to prevent explosion
    with torch.no_grad():
        self.conv_first.weight.mul_(0.1)  # 10x smaller than default Kaiming
        if self.conv_first.bias is not None:
            self.conv_first.bias.zero_()
```

### 3. Warmup Schedule for Amplification (Progressive Training)
Instead of 20x from epoch 1, ramp up gradually:

```python
# In MultiTaskHybridNSND.__init__
self.residual_scale_target = torch.tensor(float(residual_scale))
self.register_buffer("residual_scale", torch.tensor(1.0))  # Start at 1.0
self.warmup_epochs = 10  # Warmup over 10 epochs

# In training loop (after each epoch)
def update_residual_scale(model, epoch, warmup_epochs=10):
    """Gradually increase amplification from 1x to target over warmup_epochs"""
    if epoch < warmup_epochs:
        alpha = epoch / warmup_epochs
        # Cosine warmup schedule (smoother than linear)
        alpha = 0.5 * (1 - math.cos(math.pi * alpha))
        current_scale = 1.0 + alpha * (model.residual_scale_target - 1.0)
        model.residual_scale.fill_(current_scale)
        print(f"Epoch {epoch}: Residual scale = {current_scale:.2f}")
```

### 4. Gradient Scaling (Balanced Clipping)
Replace aggressive 0.1 clipping with layer-wise scaling:

```python
# In training loop (replace lines 2334-2338)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Global clipping

# Layer-wise gradient scaling for Swin head
if model.residual_heads is not None:
    for name, head in model.residual_heads.items():
        if isinstance(head, SwinResidualHead):
            # Gentler clipping for Swin attention layers
            for layer in head.layers:
                torch.nn.utils.clip_grad_norm_(layer.parameters(), 0.5)
            # Tighter clipping only for final layer
            torch.nn.utils.clip_grad_norm_(head.conv_last.parameters(), 0.1)
        else:
            # NAFNet heads can handle normal clipping
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.3)
```

## Implementation Priority

**Immediate (Critical):**
1. Add `input_norm` to SwinResidualHead ← **Do this first!**
2. Scale down `conv_first` initialization

**Next (Recommended):**
3. Implement warmup schedule for residual_scale
4. Improve gradient clipping strategy

**Optional (Fine-tuning):**
- Reduce initial `residual_scale` from 20.0 to 10.0
- Increase gradient clipping threshold from 0.1 to 0.3 for Swin head

## Expected Results

**Before Fix:**
- 20-30% batches skipped
- Loss spikes to 100+ or NaN
- Training stalls in first few epochs

**After Fix:**
- 0-2% batches skipped (only genuine outliers)
- Loss stays < 10 throughout training
- Stable convergence from epoch 1
- **Preserves amplification power** (heads still see 20x signal after warmup)

## Technical Explanation

**Why not just remove amplification?**
- Residuals from a 30 dB base are ~0.01-0.05 magnitude
- Without amplification, heads see vanishing gradients
- Amplification is essential for surgical refinement

**Why LayerNorm specifically?**
- Normalizes per-channel statistics (handles 2-channel input correctly)
- Learnable affine params preserve representational power
- Cheaper than BatchNorm (no running stats, works with batch_size=1)

**Why warmup helps?**
- Transformer attention weights start random → large amplification = chaos
- By epoch 5-10, attention has learned basic patterns → can handle full amplification
- Progressive scaling = training stability + final performance

## Testing Checklist

After applying the fix:
- [ ] No "Skipped batch" warnings in first 5 epochs
- [ ] Loss < 5.0 by epoch 3
- [ ] PSNR improving each epoch (no divergence)
- [ ] Head outputs in reasonable range (check `refined[key].abs().max()` < 1.0)
- [ ] Residual scale reaches target value by epoch 10

## Rollback Plan

If the fix causes issues:
1. Remove `input_norm` layer
2. Set `residual_scale = 5.0` (conservative value)
3. Revert to aggressive gradient clipping (0.1)

This gives you a stable baseline to iterate from.
