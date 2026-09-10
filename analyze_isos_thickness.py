#!/usr/bin/env python3
"""Analyze actual IS_OS (Class 4) thickness in Duke dataset."""

import json
import numpy as np
from PIL import Image
import os

def analyze_class4_thickness(jsonl_path, max_samples=50):
    """Analyze the thickness of Class 4 region in the dataset."""
    
    thicknesses = []
    class4_heights = []
    total_class4_pixels = []
    
    with open(jsonl_path) as f:
        lines = f.readlines()[:max_samples]
    
    for line in lines:
        data = json.loads(line)
        mask_path = data.get('mask_path') or data.get('label_path')
        
        if not mask_path or not os.path.exists(mask_path):
            continue
            
        # Load mask
        mask = np.array(Image.open(mask_path))
        H, W = mask.shape
        
        # Analyze Class 4 per column
        col_thicknesses = []
        for col in range(W):
            col_mask = mask[:, col]
            class4_pos = np.where(col_mask == 4)[0]
            
            if len(class4_pos) > 0:
                thickness = class4_pos[-1] - class4_pos[0] + 1
                col_thicknesses.append(thickness)
                class4_heights.append(class4_pos[0])  # Top of Class 4
        
        if col_thicknesses:
            thicknesses.extend(col_thicknesses)
            total_class4_pixels.append(np.mean(col_thicknesses))
    
    thicknesses = np.array(thicknesses)
    class4_heights = np.array(class4_heights)
    
    print("=" * 60)
    print("CLASS 4 (IS_OS + RPE_Choroid) THICKNESS ANALYSIS")
    print("=" * 60)
    print(f"Samples analyzed: {len(lines)}")
    print(f"Total columns analyzed: {len(thicknesses)}")
    print()
    print("Class 4 Total Thickness (pixels):")
    print(f"  Min:    {thicknesses.min():.1f}")
    print(f"  Max:    {thicknesses.max():.1f}")
    print(f"  Mean:   {thicknesses.mean():.1f}")
    print(f"  Median: {np.median(thicknesses):.1f}")
    print(f"  Std:    {thicknesses.std():.1f}")
    print()
    print("Class 4 Start Position (from top):")
    print(f"  Min:    {class4_heights.min():.1f}")
    print(f"  Max:    {class4_heights.max():.1f}")
    print(f"  Mean:   {class4_heights.mean():.1f}")
    print()
    
    # Thickness distribution
    print("Thickness Distribution:")
    for threshold in [20, 30, 40, 50, 60, 80, 100]:
        pct = (thicknesses <= threshold).mean() * 100
        print(f"  <= {threshold}px: {pct:.1f}%")
    
    print()
    print("=" * 60)
    print("ISSUE ANALYSIS")
    print("=" * 60)
    print(f"Current assumption: IS_OS = top 30px of Class 4")
    print(f"But Class 4 mean thickness = {thicknesses.mean():.1f}px")
    print()
    
    # Estimate what IS_OS should be
    # IS_OS junction is typically ~20-40 microns thick
    # If 1px ≈ 4 microns, IS_OS should be ~5-10 pixels, not 30!
    print("Anatomical Reality:")
    print("  - IS/OS junction is ~20-40 microns thick")
    print("  - If 1px ≈ 4 microns, IS_OS should be ~5-10 pixels")
    print("  - Current 30px assumption is TOO THICK")
    print()
    print("This causes:")
    print("  1. IS_OS label includes part of RPE layer")
    print("  2. Boundary localization becomes imprecise")
    print("  3. High MAE because true boundary is blurred")

if __name__ == '__main__':
    analyze_class4_thickness('combined_train.jsonl', max_samples=50)
