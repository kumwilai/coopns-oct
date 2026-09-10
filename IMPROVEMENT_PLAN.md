# Improvement Plan for Anatomy-Aware OCT Denoising

## Problem Diagnosis

| Issue | Root Cause | Impact |
|-------|-----------|--------|
| Marginal gains (+0.07 dB) | Strong baseline already solves most cases | Weak contribution |
| Poor noise classification (13%) | No supervision, weak architecture | Undermines core claim |
| Forced expert balance | Experts not architecturally different | Fake interpretability |
| Crude layer detection | Fixed depth bands, no ground truth | Overstated "anatomy-aware" |

---

## OPTION A: Fix Current Approach (Medium Effort)

### A1. Supervised Noise Type Pre-training

**Problem**: Noise classifier is random (13-20% accuracy)

**Solution**: Pre-train noise decomposer with supervision

```python
# Add noise classification loss during training
def noise_classification_loss(predicted_weights, gt_weights):
    """
    predicted_weights: [B, 4, H, W] per-pixel noise type
    gt_weights: [B, 4] global ground truth from dataset
    """
    # Global prediction
    pred_global = predicted_weights.mean(dim=[2, 3])  # [B, 4]

    # Cross-entropy loss
    return F.cross_entropy(pred_global, gt_weights.argmax(dim=1))
```

**Expected improvement**: 60-80% classification accuracy

---

### A2. Expert Architectural Differentiation

**Problem**: All experts are similar CNN-based, naturally converge

**Solution**: Make experts architecturally distinct

```python
class DifferentiatedExperts(nn.Module):
    def __init__(self):
        self.experts = nn.ModuleDict({
            # Speckle: Multiplicative noise → Log-domain processing
            'speckle': LogDomainDenoiser(),

            # Banding: Periodic noise → Frequency-domain processing
            'banding': FrequencyDomainDenoiser(),

            # Gaussian: Additive noise → Local filtering
            'gaussian': LocalAdaptiveFilter(),

            # Shot: Signal-dependent → Variance-stabilizing
            'shot': VSTDenoiser(),
        })
```

**Key**: Each expert should be fundamentally different in HOW it processes, not just learned parameters.

---

### A3. Hard Example Mining

**Problem**: Baseline handles most cases well, gains are marginal

**Solution**: Focus training on cases where baseline fails

```python
def hard_example_weighted_loss(denoised, clean, backbone_out):
    """Weight loss by backbone error - focus on hard cases."""
    backbone_error = (backbone_out - clean).abs()

    # Identify hard regions (top 20% error)
    threshold = backbone_error.quantile(0.8)
    hard_mask = (backbone_error > threshold).float()

    # 5x weight on hard regions
    weight = 1.0 + 4.0 * hard_mask

    loss = (weight * (denoised - clean).pow(2)).mean()
    return loss
```

**Expected improvement**: +0.2-0.3 dB on hard cases

---

## OPTION B: Reframe the Contribution (Recommended)

### B1. New Framing: "Adaptive Residual Refinement"

**Drop the "neuro-symbolic" claim** (experts aren't truly symbolic)

**New narrative**:
- Neural backbone provides strong baseline
- Lightweight refinement network corrects remaining errors
- Spatially-adaptive: learns WHERE refinement is needed
- Interpretable: can visualize refinement regions

This is **honest** and still publishable.

---

### B2. Focus on Clinical Metrics

**Problem**: PSNR/SSIM gains are marginal and not clinically meaningful

**Solution**: Evaluate on clinically relevant metrics

```python
def clinical_metrics(denoised, clean, noisy):
    """Metrics that matter for OCT diagnosis."""

    metrics = {}

    # 1. Layer Boundary Sharpness (can you see layer edges?)
    metrics['boundary_sharpness'] = compute_boundary_sharpness(denoised)

    # 2. Contrast-to-Noise Ratio between adjacent layers
    metrics['cnr_nfl_gcl'] = compute_cnr(denoised, zone1='nfl', zone2='gcl')
    metrics['cnr_isos_rpe'] = compute_cnr(denoised, zone1='isos', zone2='rpe')

    # 3. Feature Visibility Score (clinical expert rating proxy)
    metrics['feature_visibility'] = compute_feature_visibility(denoised)

    # 4. Artifact Reduction (banding, shadows)
    metrics['banding_reduction'] = measure_banding(noisy) - measure_banding(denoised)

    return metrics
```

**Why this matters**: A method with +0.05 dB PSNR but +20% better layer visibility is more publishable than +0.5 dB PSNR with same visibility.

---

### B3. True Layer Segmentation Integration

**Problem**: Current "layer zones" are fixed depth percentages

**Solution**: Use actual layer segmentation

```python
# Option 1: Pre-trained layer segmentation
from retina_layer_seg import RetinaLayerSegmenter

class AnatomyAwareDenoiser(nn.Module):
    def __init__(self):
        # Pre-trained, frozen layer segmenter
        self.layer_seg = RetinaLayerSegmenter.load_pretrained()
        self.layer_seg.eval()
        for p in self.layer_seg.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Get actual layer boundaries
        with torch.no_grad():
            layer_masks = self.layer_seg(x)  # [B, 9, H, W] for 9 layers

        # Use real anatomy for routing
        ...

# Option 2: Joint training with layer segmentation auxiliary loss
# Requires layer boundary ground truth
```

**Datasets with layer annotations**:
- Duke DME dataset (has layer boundaries)
- RETOUCH challenge dataset
- AROI dataset

---

## OPTION C: Simplify and Strengthen (Low Effort, High Impact)

### C1. Ablation-Driven Simplification

Remove components that don't help:

```python
# Test each component's contribution
ablations = {
    'full_model': use_all_components(),
    'no_symbolic': disable_symbolic_experts(),      # Just backbone
    'no_layer_aware': disable_layer_detection(),   # No anatomy
    'no_diversity': disable_diversity_loss(),      # Let experts converge
    'single_expert': use_only_best_expert(),       # Which expert matters?
}

# Keep only components that provide >0.1 dB improvement
```

**Likely outcome**: You may find a simpler model that works just as well.

---

### C2. Honest Minimal Contribution

If after ablation, the contribution is small, be honest:

**Weak claim (avoid)**:
> "We propose a novel neuro-symbolic framework with per-pixel noise decomposition achieving state-of-the-art results"

**Strong claim (honest)**:
> "We investigate whether classical denoising operators can complement neural denoisers for OCT. Our analysis shows that while hybrid approaches offer interpretability, the performance gains are marginal (+0.07 dB) over strong neural baselines. We identify conditions under which symbolic components help."

This is a **valid negative result** paper.

---

## Recommended Action Plan

### Phase 1: Diagnosis (1-2 days)
1. Run ablation study - which components actually help?
2. Analyze failure cases - where does baseline fail?
3. Check noise classification on held-out set

### Phase 2: Focus (1 week)
Based on ablation:
- If symbolic experts help → Strengthen them (Option A)
- If they don't help → Reframe contribution (Option B)
- If nothing helps much → Write honest analysis (Option C)

### Phase 3: Strengthen (2-3 weeks)
Pick ONE strong contribution:
1. **Best case**: Supervised noise routing that actually works (+0.3 dB)
2. **Good case**: Clinical metrics showing better layer visibility
3. **Acceptable**: Thorough analysis of when hybrid approaches help/fail

---

## Quick Wins (Do Today)

1. **Add noise classification supervision** - Should boost accuracy to 60%+
2. **Hard example mining** - Focus on backbone failure cases
3. **Report clinical metrics** - CNR, edge sharpness, not just PSNR

```bash
# Run ablation study
python run_ablation.py --components symbolic,layer_aware,diversity,anatomy_fusion
```
