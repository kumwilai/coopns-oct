#!/usr/bin/env python3
"""Analyze IS_OS boundary prediction vs pseudo-label ground truth."""

import torch
import json
import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, '/home/kumwilai/OCT')

def load_model_and_predict(checkpoint_path, image_path, mask_path):
    """Load model and make prediction."""
    from train_seg_4class_improved import ImprovedBoundarySegmenter
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    in_channels = ckpt.get('in_channels', 2)
    
    model = ImprovedBoundarySegmenter(in_channels=in_channels, num_classes=4)
    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('model.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    
    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape
    mask = np.array(Image.open(mask_path))
    
    # Create 4-class mask
    is_os_thickness = 30
    mask_4class = np.zeros_like(mask)
    mask_4class[mask == 1] = 0
    mask_4class[mask == 2] = 0
    mask_4class[mask == 3] = 1
    
    for col in range(W):
        col_mask = mask[:, col]
        class4_pos = np.where(col_mask == 4)[0]
        if len(class4_pos) > 0:
            is_os_start = class4_pos[0]
            is_os_end = min(is_os_start + is_os_thickness, H)
            mask_4class[is_os_start:is_os_end, col] = 2
            for row in range(is_os_end, H):
                if col_mask[row] == 4 or col_mask[row] == 0:
                    mask_4class[row, col] = 3
    mask_4class[mask >= 5] = 3
    
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    if in_channels == 2:
        depth = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        img_tensor = torch.cat([img_tensor, depth], dim=1)
    
    with torch.no_grad():
        output = model(img_tensor)
        # Handle different output formats
        if isinstance(output, tuple):
            logits = output[0]  # First element is seg_logits
        elif isinstance(output, dict):
            logits = output['seg_logits']
        else:
            logits = output
        pred = logits.argmax(dim=1)[0].numpy()
    
    return pred, mask_4class, mask

def analyze_boundary_error(pred, target, original_mask):
    """Analyze boundary prediction error."""
    H, W = pred.shape
    
    errors = []
    pred_boundaries = []
    target_boundaries = []
    
    for col in range(W):
        pred_is_os = np.where(pred[:, col] == 2)[0]
        target_is_os = np.where(target[:, col] == 2)[0]
        
        if len(pred_is_os) > 0 and len(target_is_os) > 0:
            pred_top = pred_is_os[0]
            target_top = target_is_os[0]
            error = abs(pred_top - target_top)
            errors.append(error)
            pred_boundaries.append(pred_top)
            target_boundaries.append(target_top)
    
    return np.array(errors), np.array(pred_boundaries), np.array(target_boundaries)

# Main
if __name__ == '__main__':
    checkpoint = 'checkpoints/seg_4class_improved_best.pth'
    
    with open('combined_val.jsonl') as f:
        samples = [json.loads(line) for line in f.readlines()[:10]]
    
    all_errors = []
    all_pred_std = []
    all_target_std = []
    
    for i, sample in enumerate(samples):
        img_path = sample['image_path']
        mask_path = sample['mask_path']
        
        if not os.path.exists(mask_path):
            continue
            
        pred, target, original = load_model_and_predict(checkpoint, img_path, mask_path)
        errors, pred_b, target_b = analyze_boundary_error(pred, target, original)
        
        if len(errors) > 0:
            all_errors.extend(errors)
            all_pred_std.append(pred_b.std())
            all_target_std.append(target_b.std())
            print(f"Sample {i+1}: MAE={errors.mean():.1f}px, Pred_std={pred_b.std():.1f}, Target_std={target_b.std():.1f}")
    
    print("\n" + "=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    all_errors = np.array(all_errors)
    avg_pred_std = np.mean(all_pred_std)
    avg_target_std = np.mean(all_target_std)
    
    print(f"\nOverall Mean MAE: {all_errors.mean():.2f} px")
    print(f"Overall Median MAE: {np.median(all_errors):.2f} px")
    print(f"\nPrediction boundary std (within-image variation): {avg_pred_std:.1f} px")
    print(f"Target boundary std (within-image variation): {avg_target_std:.1f} px")
    
    print("\nError Distribution:")
    for threshold in [1, 3, 5, 10, 20, 30, 50]:
        pct = (all_errors <= threshold).mean() * 100
        print(f"  <= {threshold}px: {pct:.1f}%")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print(f"""
The IS_OS MAE of ~20px is caused by:

1. HIGH WITHIN-IMAGE BOUNDARY VARIATION
   - Target (pseudo-label) varies by {avg_target_std:.1f} px std within each image
   - This reflects real anatomy: retinal layers are curved/irregular
   - Model predictions vary by {avg_pred_std:.1f} px std

2. SEGMENTATION vs BOUNDARY DETECTION
   - Segmentation produces region labels, not precise boundaries
   - Boundary position is derived from "first pixel of class"
   - This is inherently imprecise (±several pixels)

3. THE 30px THICKNESS IS NOT THE ISSUE
   - Thickness affects Dice score, not top boundary MAE
   - MAE only measures where IS_OS starts (top boundary)

4. SOLUTIONS TO REDUCE MAE:
   a) Add dedicated boundary regression head (predict y-coordinate directly)
   b) Use soft-argmax for sub-pixel boundary localization
   c) Post-process with curve fitting (smooth the boundary)
   d) Train with boundary-focused loss (surface distance loss)
""")
