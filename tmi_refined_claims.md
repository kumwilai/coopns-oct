# Refined Claims for TMI Paper
## Honest Reframing of CUAP-OCT Components

---

## Framing Strategy

**Don't claim:** "We invented X"
**Do claim:** "We integrate X into a unified framework for OCT, demonstrating its effectiveness for clinical layer analysis"

**The novelty is the SYSTEM, not individual components.**

---

## Component-by-Component Reframing

### 1. Multi-task Denoising + Segmentation

| Overclaimed | Refined Claim |
|-------------|---------------|
| "Novel joint optimization" | "We leverage multi-task learning to enable anatomically-informed denoising, where segmentation provides spatial priors for layer-specific processing" |

**What to write:**
> "While multi-task learning is established, its application to OCT denoising with
> layer-specific adaptation has not been explored. The segmentation branch provides
> anatomical context that guides spatially-varying denoising."

**Evidence needed:** Show that joint training outperforms sequential (denoise→segment or segment→denoise)

---

### 2. Uncertainty Quantification → "Segmentation Entropy Prior"

| Overclaimed | Refined Claim |
|-------------|---------------|
| "Calibrated uncertainty quantification" | "Segmentation entropy as a spatial prior indicating regions requiring careful processing (boundaries, ambiguous anatomy)" |

**What to write:**
> "We use segmentation entropy (not Bayesian uncertainty) as a proxy for spatial
> ambiguity. High-entropy regions (layer boundaries, pathological areas) receive
> more conservative denoising to preserve diagnostically relevant detail."

**Key change:** Call it "entropy-guided" not "uncertainty-aware"

**Evidence needed:** Show entropy is high at boundaries, and that this correlation helps preserve edges

---

### 3. Pathology Preservation → "Structure-Preserving Regularization"

| Overclaimed | Refined Claim |
|-------------|---------------|
| "Protects diagnostic pathology features" | "Structure-preserving regularization that penalizes over-smoothing in high-detail regions" |

**What to write:**
> "We apply structure-preserving constraints that discourage aggressive denoising
> in regions with high local variance. While we cannot guarantee pathology
> preservation without explicit pathology labels, this encourages retention of
> fine structural details that may include pathological features."

**Key change:** Acknowledge the limitation (no pathology labels)

**Evidence needed:** Visual examples showing texture preservation

---

### 4. Boundary Sharpness → "Layer Boundary Fidelity"

| Overclaimed | Refined Claim |
|-------------|---------------|
| "Novel boundary sharpness loss" | "We enforce layer boundary fidelity, critical for accurate thickness measurements used in glaucoma/AMD diagnosis" |

**What to write:**
> "Layer boundary sharpness directly impacts clinical measurements. Retinal nerve
> fiber layer (RNFL) thickness, measured from layer boundaries, is a primary
> biomarker for glaucoma progression. We incorporate gradient-based boundary
> matching to maintain measurement accuracy post-denoising."

**Key change:** Emphasize CLINICAL IMPORTANCE, not technical novelty

**Evidence needed:** Show boundary gradient preservation; ideally show thickness measurement accuracy

---

### 5. Anatomical Consistency → "Domain Knowledge Regularization"

| Overclaimed | Refined Claim |
|-------------|---------------|
| "Novel anatomical consistency enforcement" | "We incorporate ophthalmological domain knowledge as soft constraints: expected layer ordering and physiologically plausible thickness ranges" |

**What to write:**
> "We regularize segmentation using established anatomical priors: retinal layers
> appear in consistent order (RNFL→GCL→IPL→...), and layer thicknesses fall within
> physiologically normal ranges (e.g., RNFL: 50-150μm). These constraints improve
> segmentation plausibility, especially in noisy regions."

**Key change:** Frame as "incorporating domain knowledge" not "novel method"

**Evidence needed:** Show anatomically implausible predictions are reduced

---

### 6. Confidence-Weighted Denoising → "Adaptive Denoising Strength"

| Overclaimed | Refined Claim |
|-------------|---------------|
| "Novel confidence-weighted refinement" | "Spatially-adaptive denoising where refinement strength is modulated by segmentation confidence" |

**What to write:**
> "Denoising strength adapts spatially based on segmentation confidence. In regions
> where layer identity is certain, aggressive denoising is safe. In ambiguous
> regions (boundaries, potential pathology), conservative denoising preserves
> detail for clinical interpretation."

**Key change:** Describe the mechanism, don't claim novelty

**Evidence needed:** Show gate values vary spatially in interpretable ways

---

### 7. Clinical Importance Weighting → "Clinically-Motivated Optimization"

| Overclaimed | Refined Claim |
|-------------|---------------|
| "Novel clinical weighting scheme" | "Optimization priorities based on established clinical significance of retinal layers for specific diseases" |

**What to write:**
> "We weight the optimization objective according to clinical importance:
> - RNFL (2.0×): Primary biomarker for glaucoma [cite: PMID 28384450]
> - IS/OS junction (1.5×): Correlates with visual acuity [cite: PMID 23287585]
> - RPE (1.2×): Critical for AMD diagnosis [cite: PMID 26066786]
>
> This ensures the model prioritizes quality in diagnostically critical layers."

**Key change:** CITE clinical literature to justify weights

**Evidence needed:** Show RNFL metrics improve more than baseline layers

---

### 8. Layer-Specific Noise Gates → "Learned Layer-Adaptive Processing" ✅

| Current | Refined Claim |
|---------|---------------|
| "Potentially novel" | "We propose learned layer-specific denoising strengths, where the network discovers that different retinal layers benefit from different processing intensities" |

**What to write:**
> "Our key technical contribution is layer-specific noise gates: learned parameters
> that control denoising intensity per anatomical layer. Unlike global denoising,
> this allows the network to discover optimal layer-specific processing—applying
> stronger denoising to homogeneous layers (choroid) while preserving detail in
> fine-structured layers (photoreceptors)."

**This IS your main contribution - emphasize it**

**Evidence needed:**
- Show gates converge to different values per layer
- Show correlation between gate values and layer characteristics
- Ablation: gates ON vs OFF

---

## Paper Structure Recommendation

### Title
"Layer-Adaptive Denoising for Retinal OCT: A Clinically-Informed Multi-Task Framework"

### Abstract (Refined)
> Optical coherence tomography (OCT) image quality is limited by speckle noise,
> impacting clinical measurements and diagnosis. We present a layer-adaptive
> denoising framework that leverages anatomical segmentation to guide spatially-
> varying denoising. Our key contributions are: (1) **layer-specific noise gates**
> that learn optimal denoising strength per retinal layer, (2) **clinically-
> weighted optimization** prioritizing diagnostically critical layers (RNFL for
> glaucoma, RPE for AMD), and (3) a **unified multi-task architecture** where
> segmentation and denoising mutually reinforce each other. We incorporate
> anatomical priors (layer ordering, thickness constraints) and entropy-guided
> processing for boundary preservation. Experiments on synthetic and real-world
> OCT datasets (Duke, PKU37) demonstrate improved layer-specific quality metrics
> while maintaining segmentation accuracy.

### Contributions Section
> Our contributions are:
> 1. A **layer-adaptive denoising architecture** with learned per-layer processing
>    strengths, enabling anatomically-informed noise removal
> 2. **Clinically-weighted optimization** based on established diagnostic
>    importance of retinal layers, prioritizing quality in disease-relevant regions
> 3. **Integration of anatomical priors** (layer ordering, thickness constraints,
>    boundary preservation) as regularization for improved clinical plausibility
> 4. **Comprehensive evaluation** on synthetic and real-world OCT data, with
>    per-layer analysis demonstrating targeted improvements in critical layers

---

## What NOT to Claim

❌ "First to apply multi-task learning to OCT" (probably false)
❌ "Uncertainty-aware" (we don't have proper uncertainty)
❌ "Pathology-preserving" (we don't have pathology labels)
❌ "Novel loss functions" (they're standard losses with domain-specific application)
❌ "State-of-the-art" (unless you beat published baselines)

---

## What TO Claim

✅ "Layer-adaptive denoising with learned per-layer strengths" (novel for OCT)
✅ "Clinically-motivated optimization based on diagnostic importance" (well-justified)
✅ "Unified framework integrating segmentation-guided denoising" (system contribution)
✅ "Demonstrated on real-world data (Duke, PKU37)" (practical validation)
✅ "Per-layer analysis showing targeted improvement" (rigorous evaluation)

---

## Ablation Study Design (Critical for Reviewers)

| Experiment | What it proves |
|------------|----------------|
| Full model vs. no layer gates | Gates are essential |
| Full model vs. uniform weights | Clinical weighting helps |
| Full model vs. no segmentation | Joint training helps |
| Full model vs. no boundary loss | Boundary loss preserves edges |
| Full model vs. no anatomical prior | Priors improve plausibility |

Run these with `ABLATION_MODE` in `tmi_retrain_clinical.sh`
