# Layer-Adaptive OCT Denoising (LAOD)
## A Focused, Defensible Contribution for TMI

---

## 1. Core Contribution (One Sentence)

**We propose layer-adaptive denoising for retinal OCT, where segmentation-guided
gates learn anatomically-varying denoising strengths, with clinical importance
weighting for diagnostically critical layers.**

---

## 2. Key Innovation

### What's Novel
1. **Layer-Specific Noise Gates**: Each retinal layer gets its own learned
   denoising strength (not a single global denoiser)

2. **Segmentation-Guided Adaptation**: The gates are conditioned on segmentation
   features, so denoising adapts to local anatomy

3. **Clinical Importance Weighting**: Diagnostically critical layers (RNFL for
   glaucoma, RPE for AMD) receive higher optimization priority

### What's NOT Claimed
- ~~Uncertainty quantification~~ (we don't have proper uncertainty)
- ~~Pathology preservation~~ (we don't have pathology labels)
- ~~Novel loss functions~~ (we use standard losses with anatomical weighting)

---

## 3. Method Overview

```
Input (Noisy OCT) ──────────────────────────────────────┐
                                                         │
                    ┌────────────────────────────────────▼───────────┐
                    │           NAFNet Backbone                      │
                    │         (Feature Extraction)                   │
                    └────────────────────┬───────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────────────┐
                    │        Segmentation Head                       │
                    │   (5 Retinal Layers + Background)              │
                    └────────────────────┬───────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────────────┐
                    │     Layer-Specific Gate Network                │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │ RNFL_GCL gate:    g₁ ∈ [0,1]            │   │
                    │  │ INL_OPL gate:     g₂ ∈ [0,1]            │   │
                    │  │ ONL gate:         g₃ ∈ [0,1]            │   │
                    │  │ IS_OS gate:       g₄ ∈ [0,1]            │   │
                    │  │ RPE_Choroid gate: g₅ ∈ [0,1]            │   │
                    │  └─────────────────────────────────────────┘   │
                    └────────────────────┬───────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────────────┐
                    │        Adaptive Refinement                     │
                    │                                                │
                    │   R(x,y) = Σᵢ gᵢ · Mᵢ(x,y) · refinement       │
                    │                                                │
                    │   where Mᵢ = soft segmentation mask for layer i│
                    └────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                              Denoised Output
```

---

## 4. Loss Function (Simplified)

### Total Loss
```
L_total = L_denoise + λ_seg · L_seg + λ_boundary · L_boundary + λ_gate · L_gate
```

### Components

| Loss | Purpose | Formula |
|------|---------|---------|
| L_denoise | Reconstruction | Clinically-weighted L1 per layer |
| L_seg | Segmentation | Weighted Cross-Entropy |
| L_boundary | Edge preservation | Sobel gradient matching |
| L_gate | Clinical priority | Encourage high gates for RNFL/RPE |

### Clinical Weights (Fixed, Based on Literature)
```python
CLINICAL_WEIGHTS = {
    'RNFL_GCL': 2.0,      # Glaucoma diagnosis (PMID: 28384450)
    'INL_OPL': 1.0,       # Baseline
    'ONL': 1.0,           # Baseline
    'IS_OS': 1.5,         # Visual acuity correlation (PMID: 23287585)
    'RPE_Choroid': 1.2,   # AMD diagnosis (PMID: 26066786)
}
```

---

## 5. What We Remove (Overclaimed)

| Component | Why Remove |
|-----------|------------|
| UncertaintyCalibrationLoss | Not real uncertainty - just boundary detection |
| PathologyPreservationLoss | No pathology labels to validate |
| AnatomicalConsistencyLoss | Adds complexity, marginal benefit |
| ConfidenceWeightedRefinement | Softmax confidence is poorly calibrated |

---

## 6. Experiments to Run

### 6.1 Main Results (Table 1)
Compare against baselines on synthetic + real data:

| Method | Duke PSNR | Duke SSIM | PKU37 PSNR | PKU37 SSIM |
|--------|-----------|-----------|------------|------------|
| Noisy input | - | - | - | - |
| BM3D | - | - | - | - |
| DnCNN | - | - | - | - |
| NAFNet (global) | - | - | - | - |
| **LAOD (Ours)** | - | - | - | - |

### 6.2 Layer-Specific Analysis (Table 2)
Show per-layer improvement (KEY CONTRIBUTION):

| Layer | Clinical Use | Gate Value | PSNR Gain |
|-------|--------------|------------|-----------|
| RNFL_GCL | Glaucoma | 0.XX | +X.XX dB |
| INL_OPL | Diabetic retinopathy | 0.XX | +X.XX dB |
| ONL | Photoreceptor health | 0.XX | +X.XX dB |
| IS_OS | Visual acuity | 0.XX | +X.XX dB |
| RPE_Choroid | AMD | 0.XX | +X.XX dB |

### 6.3 Ablation Study (Table 3)
Show each component matters:

| Configuration | PSNR | Dice |
|---------------|------|------|
| Global denoising (no gates) | - | - |
| Layer gates (uniform weights) | - | - |
| Layer gates + clinical weights | - | - |
| **Full LAOD** | - | - |

### 6.4 Clinical Validation (Table 4)
**This is what reviewers will care about:**

| Metric | Before Denoising | After LAOD | p-value |
|--------|------------------|------------|---------|
| RNFL thickness measurement error (μm) | - | - | - |
| Layer boundary detection accuracy | - | - | - |
| Inter-observer agreement (Dice) | - | - | - |

---

## 7. Paper Framing

### Title Options
1. "Layer-Adaptive Denoising for Retinal OCT with Clinical Importance Weighting"
2. "Anatomically-Guided Adaptive Denoising for Retinal OCT"
3. "Learning Layer-Specific Denoising Strengths for Clinical OCT Analysis"

### Abstract Structure
1. **Problem**: OCT denoising typically uses global parameters, ignoring that
   different retinal layers have different noise characteristics and clinical importance
2. **Method**: We propose layer-adaptive denoising with segmentation-guided gates
   and clinical importance weighting
3. **Results**: Our method achieves X dB improvement on RNFL (glaucoma-critical)
   and Y dB on RPE (AMD-critical) while maintaining layer boundaries
4. **Conclusion**: Layer-aware denoising better serves clinical diagnosis

---

## 8. Honest Limitations (Include in Paper)

1. Trained on synthetic noise - may not capture all real noise patterns
2. Clinical weights are fixed based on literature, not learned
3. Evaluated on PSNR/SSIM - clinical utility requires prospective study
4. Layer segmentation accuracy affects denoising quality

---

## 9. Implementation Checklist

- [ ] Simplify losses to: L_denoise, L_seg, L_boundary, L_gate
- [ ] Remove: uncertainty, pathology, anatomical, confidence losses
- [ ] Run baselines: BM3D, DnCNN, NAFNet-global
- [ ] Run ablation: gates vs no-gates, weights vs no-weights
- [ ] Generate per-layer analysis tables
- [ ] Test on Duke + PKU37 real data
