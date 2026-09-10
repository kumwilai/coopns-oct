# Memory Leak Fixes and Patch-Based Inference Implementation

## Summary

This document describes the memory leak issues identified in the training code and the fixes implemented to prevent system crashes. Additionally, patch-based inference has been implemented for processing large images during evaluation.

## Memory Leak Root Causes Identified

### 1. **usage_loss Function - Computation Graph Leak**
**Location**: `nsnd_oct/scripts/train_adaptive_nafnet_full.py:217-246`

**Problem**: The `usage_loss` function computed `out_correct` with `torch.no_grad()`, but didn't properly detach it before using it in the loss computation with `out_wrong`. This kept the computation graph in memory.

**Fix**:
```python
# Before:
with torch.no_grad():
    out_correct = model(...)

out_wrong = model(...)
diff = (out_correct - out_wrong).abs().mean()
```

```python
# After:
with torch.no_grad():
    out_correct = model(...)
    # CRITICAL: Clone and detach to avoid memory leak
    out_correct = out_correct.detach().clone()

out_wrong = model(...)
# CRITICAL: Detach out_correct in the computation to prevent gradients flowing back
diff = (out_correct.detach() - out_wrong).abs().mean()
loss = torch.relu(margin - diff)
return loss, diff.detach()
```

### 2. **Validation Loop - Expensive usage_loss Calls**
**Location**: `nsnd_oct/scripts/train_adaptive_nafnet_full.py:712-722`

**Problem**: The validation loop called `usage_loss` again for the first 10 batches, creating additional forward passes and doubling memory usage. This was unnecessary since usage loss is only for training.

**Fix**: Removed the validation `usage_loss` calls entirely:
```python
# REMOVED: usage_loss call during validation - too expensive and causes memory issues
# The usage loss is only for training anyway
```

### 3. **DataLoader Workers Exceeding System Capacity**
**Location**: `nsnd_oct/scripts/train_adaptive_nafnet_full.py:277`

**Problem**: The script used 4 workers, but the system could only handle 2, causing potential memory issues and slowdowns.

**Fix**:
```python
# Before:
parser.add_argument("--num_workers", type=int, default=4)

# After:
parser.add_argument("--num_workers", type=int, default=2)  # Reduced to avoid memory issues
```

### 4. **Tensor Accumulation in Training Loop**
**Location**: `nsnd_oct/scripts/train_adaptive_nafnet_full.py:611-616`

**Problem**: Tensors from each batch were not being explicitly deleted, causing gradual memory accumulation across batches.

**Fix**: Added explicit cleanup after each batch:
```python
# Clear unused tensors to prevent memory accumulation
# Do this after logging to avoid accessing deleted variables
del total_loss, u_loss, map_loss, recon_loss, grad_loss
del spatial_map, basis, feature_map, global_weights
if batch_idx % 10 == 0 and torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 5. **Tensor Accumulation in Validation Loop**
**Location**: `nsnd_oct/scripts/train_adaptive_nafnet_full.py:762-765`

**Problem**: Similar to training, validation tensors were not being cleaned up, causing memory accumulation.

**Fix**: Added cleanup after processing each validation batch:
```python
# Clear tensors to prevent memory accumulation
del noisy, clean, denoised, base_out, spatial_map, basis, feature_map, global_weights
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Patch-Based Inference for Large Images

### Problem
Training uses 64×64 patches, but during evaluation on larger images, the model tried to process the entire image at once, causing memory overflow.

### Solution
Implemented `forward_patch_based()` method in the `NAFNetFullFiLM` class to process large images using overlapping patches with Gaussian blending.

**Location**: `nsnd_oct/nsnd/models/nafnet.py:738-841`

**Key Features**:
- Processes images in patches (default: 64×64)
- Uses stride for overlap (default: 32 for 50% overlap)
- Gaussian-weighted blending for smooth transitions
- Automatic memory cleanup after each patch
- Handles edge cases at image boundaries

**Usage Example**:
```python
# For small images (≤ patch_size), use regular forward:
denoised = model(noisy_t, spatial_map=spatial_map, basis=basis, alpha=alpha, gate=gate)

# For large images, use patch-based processing:
denoised = model.forward_patch_based(
    noisy_t,
    patch_size=64,
    stride=32,
    spatial_map=spatial_map,
    basis=basis,
    alpha=alpha,
    gate=gate
)
```

### Demo Script
A complete demonstration script has been created: `demo_patch_inference.py`

**Usage**:
```bash
python demo_patch_inference.py \
    --input path/to/noisy_image.png \
    --output path/to/denoised_image.png \
    --base_ckpt outputs/nafnet_analysis_maps_w64/nafnet_best.pth \
    --analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
    --modulator_ckpt checkpoints/adaptive_nafnet_full/adaptive_nafnet_best.pth \
    --patch_size 64 \
    --stride 32
```

## Impact

### Before Fixes:
- System crashes during training due to memory accumulation
- Cannot evaluate on images larger than training patch size (64×64)
- DataLoader workers causing system slowdowns
- Gradual memory growth leading to OOM errors

### After Fixes:
- Stable training with controlled memory usage
- Explicit memory cleanup prevents accumulation
- Can process arbitrarily large images using patch-based inference
- Optimal DataLoader worker count for the system
- Removed expensive validation operations

## Testing

The training is now running successfully with the fixes applied:
```bash
bash run_adaptive_nafnet_full.sh
```

Key indicators of success:
1. No DataLoader worker warning
2. Training progressing without crashes
3. Memory usage remains stable across batches
4. Validation completes without OOM errors

## Additional Recommendations

### For Future Development:
1. **Monitor GPU memory**: Use `nvidia-smi` to monitor memory usage during training
2. **Gradient accumulation**: If memory is still tight, consider gradient accumulation over multiple batches
3. **Mixed precision**: Consider using `torch.cuda.amp` for automatic mixed precision to reduce memory usage
4. **Batch size**: If still experiencing issues, reduce batch size (currently 4)
5. **Checkpoint frequency**: Reduce checkpoint saving frequency if disk I/O is causing issues

### For Inference on Very Large Images:
1. Use the `forward_patch_based()` method with appropriate patch_size and stride
2. Smaller stride = more overlap = smoother results but slower processing
3. Larger patch_size = fewer patches = faster but requires more memory
4. Recommended: patch_size=64, stride=32 for good balance

## Files Modified

1. `nsnd_oct/scripts/train_adaptive_nafnet_full.py`
   - Fixed usage_loss function (line 217-246)
   - Reduced DataLoader workers (line 277)
   - Added training cleanup (line 611-616)
   - Removed validation usage_loss (line 721-722)
   - Added validation cleanup (line 762-765)

2. `nsnd_oct/nsnd/models/nafnet.py`
   - Added forward_patch_based method (line 738-814)
   - Added _create_blend_weights static method (line 816-841)

3. `demo_patch_inference.py` (NEW)
   - Complete demo script for patch-based inference on large images
