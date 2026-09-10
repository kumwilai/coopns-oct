#!/usr/bin/env python3
"""
Diagnose why physics-based noise classification isn't working.
Check if features actually separate the noise types.
"""

import torch
import json
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsnd_oct'))

from nsnd.models.noise_features import NoiseFeatureExtractor

def main():
    # Load samples
    samples = []
    with open('weights_duke_analysis_maps_val.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            samples.append(json.loads(line.strip()))

    extractor = NoiseFeatureExtractor()

    # Collect features by dominant noise type
    features_by_type = {
        'speckle': {'coef_var': [], 'sig_var': [], 'horiz': []},
        'banding': {'coef_var': [], 'sig_var': [], 'horiz': []},
        'gaussian': {'coef_var': [], 'sig_var': [], 'horiz': []},
        'shot': {'coef_var': [], 'sig_var': [], 'horiz': []},
    }

    type_names = ['speckle', 'banding', 'gaussian', 'shot']

    for sample in samples:
        # Load image
        noisy = np.array(Image.open(sample['noisy_path']).convert('L'), dtype=np.float32) / 255.0
        noisy = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)

        # Get weights
        weights = [
            sample['weights']['speckle'],
            sample['weights']['banding'],
            sample['weights']['gaussian'],
            sample['weights']['shot']
        ]
        dominant_idx = np.argmax(weights)
        dominant_type = type_names[dominant_idx]

        # Extract features
        with torch.no_grad():
            features = extractor(noisy)

        # Store mean features
        features_by_type[dominant_type]['coef_var'].append(features['coef_variation'].mean().item())
        features_by_type[dominant_type]['sig_var'].append(features['signal_var_corr'].mean().item())
        features_by_type[dominant_type]['horiz'].append(features['horizontal_ratio'].mean().item())

    # Print statistics
    print("\n" + "="*70)
    print("PHYSICS FEATURE ANALYSIS BY DOMINANT NOISE TYPE")
    print("="*70)
    print("\nExpected patterns based on noise physics:")
    print("  - Speckle: HIGH coef_var (multiplicative noise)")
    print("  - Banding: HIGH horiz_ratio (horizontal patterns)")
    print("  - Gaussian: LOW coef_var, LOW sig_var_corr (additive, uniform)")
    print("  - Shot: HIGH sig_var_corr (variance proportional to signal)")

    print("\n" + "-"*70)
    print(f"{'Type':<10} {'Count':>6} {'CoefVar':>12} {'Sig-Var':>12} {'Horiz':>12}")
    print("-"*70)

    for noise_type in type_names:
        data = features_by_type[noise_type]
        n = len(data['coef_var'])
        if n > 0:
            cv_mean = np.mean(data['coef_var'])
            cv_std = np.std(data['coef_var'])
            sv_mean = np.mean(data['sig_var'])
            sv_std = np.std(data['sig_var'])
            hz_mean = np.mean(data['horiz'])
            hz_std = np.std(data['horiz'])
            print(f"{noise_type:<10} {n:>6} {cv_mean:>6.3f}±{cv_std:.3f} {sv_mean:>6.3f}±{sv_std:.3f} {hz_mean:>6.3f}±{hz_std:.3f}")
        else:
            print(f"{noise_type:<10} {0:>6} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    print("-"*70)

    # Check separability
    print("\nFEATURE SEPARABILITY ANALYSIS:")

    # Check if any feature clearly separates types
    all_coef_vars = []
    all_sig_vars = []
    all_labels = []

    for i, noise_type in enumerate(type_names):
        for cv, sv in zip(features_by_type[noise_type]['coef_var'],
                          features_by_type[noise_type]['sig_var']):
            all_coef_vars.append(cv)
            all_sig_vars.append(sv)
            all_labels.append(i)

    if len(all_coef_vars) > 10:
        # Simple linear separability check
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = np.column_stack([all_coef_vars, all_sig_vars])
        y = np.array(all_labels)

        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        scores = cross_val_score(clf, X, y, cv=min(5, len(y)//4))
        print(f"  Linear separability (2 features): {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

        # Also try with all features
        all_features = []
        for i, noise_type in enumerate(type_names):
            for cv, sv, hz in zip(features_by_type[noise_type]['coef_var'],
                                   features_by_type[noise_type]['sig_var'],
                                   features_by_type[noise_type]['horiz']):
                all_features.append([cv, sv, hz])

        X_all = np.array(all_features)
        scores_all = cross_val_score(clf, X_all, y, cv=min(5, len(y)//4))
        print(f"  Linear separability (3 features): {scores_all.mean()*100:.1f}% ± {scores_all.std()*100:.1f}%")

        print(f"\n  Random baseline: {100/4:.1f}%")

        # Check which features matter most
        clf.fit(X_all, y)
        print(f"\n  Feature importance (logistic regression coefficients):")
        for feat_name, coef in zip(['CoefVar', 'Sig-Var', 'Horiz'], clf.coef_.mean(axis=0)):
            print(f"    {feat_name}: {coef:.3f}")

    print("="*70)

if __name__ == '__main__':
    main()
