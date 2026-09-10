# Proposal: Spatially-Adaptive Neuro-Symbolic Denoising (SANS-D)

## The Problem with Current Approaches

### Existing Methods:
1. **Pure Neural** (NAFNet, SwinIR): No interpretability, treats all noise the same
2. **Pure Classical** (BM3D, Anisotropic Diffusion): No learning, fixed parameters
3. **Your Current End-to-End**: Neural denoising with conditioning (NOT neuro-symbolic)
4. **Your NSNDModel**: Symbolic analysis → classical denoisers → neural fusion
   - But: Global weights only, no per-pixel adaptation
   - But: Classical operators have fixed parameters

### The Gap:
No method provides **per-pixel learnable symbolic denoising** with **spatially-varying operator parameters**.

---

## Proposed Innovation: SANS-D

### Core Idea: Mixture of Differentiable Symbolic Experts (MoDSE)

```
                    ┌─────────────────────────────────────────────┐
                    │                 Noisy Image                 │
                    └─────────────────────┬───────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────┐
                    │     Spatial Noise Analyzer (per-pixel)      │
                    │  Outputs: noise_type[H,W,4], noise_level[H,W] │
                    └─────────────────────┬───────────────────────┘
                                          │
              ┌───────────────┬───────────┼───────────┬───────────────┐
              ▼               ▼           ▼           ▼               ▼
    ┌─────────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │ Speckle Expert  │ │ Banding  │ │ Gaussian │ │   Shot   │ │   Neural     │
    │ (Learnable      │ │  Expert  │ │  Expert  │ │  Expert  │ │   Residual   │
    │  Anisotropic    │ │(Learnable│ │(Learnable│ │(Learnable│ │   Expert     │
    │  Diffusion)     │ │ Notch)   │ │ NLM/BM3D)│ │   VST)   │ │  (NAFNet)    │
    └────────┬────────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
             │               │            │            │              │
             └───────────────┴────────────┼────────────┴──────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────┐
                    │    Spatially-Weighted Fusion (per-pixel)    │
                    │       out[i,j] = Σ w[i,j,k] * expert_k[i,j] │
                    └─────────────────────┬───────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────┐
                    │              Denoised Output                │
                    └─────────────────────────────────────────────┘
```

---

## Novel Contributions (5 Major)

### Contribution 1: Differentiable Symbolic Operators with Learnable Parameters

**Current Problem**: Classical denoisers have fixed hyperparameters.

**Our Innovation**: Make each classical operator fully differentiable with **spatially-varying learnable parameters**.

```python
class LearnableAnisotropicDiffusion(nn.Module):
    """
    Perona-Malik diffusion with LEARNABLE parameters that vary spatially.

    Novel: Instead of fixed K and iterations, predict per-pixel:
    - K(x,y): edge sensitivity threshold
    - τ(x,y): diffusion time step
    - n(x,y): number of iterations (soft, differentiable)
    """
    def __init__(self):
        # Parameter predictor: predicts K, τ, n from local features
        self.param_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),  # K, τ, n
        )

    def forward(self, x, noise_level):
        # Predict spatially-varying parameters
        params = self.param_net(x)
        K = torch.sigmoid(params[:, 0:1]) * noise_level  # Scale by noise level
        tau = torch.sigmoid(params[:, 1:2]) * 0.25
        n_soft = torch.sigmoid(params[:, 2:3]) * 20  # Soft iteration count

        # Differentiable anisotropic diffusion
        out = self.differentiable_diffusion(x, K, tau, n_soft)
        return out
```

**Why Novel**:
- Previous work: Fixed parameters for entire image
- Our work: Per-pixel learnable parameters adapted to local noise

---

### Contribution 2: Per-Pixel Noise Decomposition Maps

**Current Problem**: Global noise type weights (e.g., [0.7, 0.1, 0.1, 0.1] for whole image).

**Our Innovation**: Predict **per-pixel noise composition**.

```python
class SpatialNoiseDecomposer(nn.Module):
    """
    Predicts per-pixel:
    1. Noise type distribution [B, 4, H, W] - which noise type dominates at each pixel
    2. Noise level map [B, 1, H, W] - local noise intensity
    3. Noise uncertainty [B, 1, H, W] - confidence in prediction
    """
    def __init__(self):
        self.encoder = UNetEncoder(in_channels=1, features=[32, 64, 128])
        self.type_head = nn.Conv2d(32, 4, 1)  # 4 noise types
        self.level_head = nn.Conv2d(32, 1, 1)  # Noise level
        self.uncertainty_head = nn.Conv2d(32, 1, 1)  # Confidence

    def forward(self, noisy):
        features = self.encoder(noisy)

        noise_type = F.softmax(self.type_head(features), dim=1)  # [B,4,H,W]
        noise_level = F.softplus(self.level_head(features))  # [B,1,H,W]
        uncertainty = torch.sigmoid(self.uncertainty_head(features))  # [B,1,H,W]

        return noise_type, noise_level, uncertainty
```

**Why Novel**:
- Captures spatially-varying noise (e.g., more speckle in dark regions, more shot noise in bright regions)
- Enables per-pixel operator selection
- Provides interpretable noise maps for clinical use

---

### Contribution 3: Mixture of Differentiable Symbolic Experts (MoDSE)

**Current Problem**: Either use one denoiser for all, or separate denoisers without proper fusion.

**Our Innovation**: **Soft mixture of symbolic experts with per-pixel gating**.

```python
class MoDSE(nn.Module):
    """
    Mixture of Differentiable Symbolic Experts

    Each expert is a differentiable symbolic operator.
    Per-pixel soft gating determines which expert(s) to use.
    """
    def __init__(self):
        self.experts = nn.ModuleList([
            LearnableAnisotropicDiffusion(),   # Speckle
            LearnableFourierNotch(),            # Banding
            LearnableNonLocalMeans(),           # Gaussian
            LearnableVarianceStabilizing(),     # Shot
            LightweightNeuralResidual(),        # Neural fallback
        ])

    def forward(self, noisy, noise_type_map, noise_level_map):
        """
        Args:
            noisy: [B, 1, H, W]
            noise_type_map: [B, 4, H, W] per-pixel noise type weights
            noise_level_map: [B, 1, H, W] per-pixel noise level
        """
        expert_outputs = []
        for i, expert in enumerate(self.experts[:-1]):
            # Each symbolic expert gets noise level for parameter adaptation
            out = expert(noisy, noise_level_map)
            expert_outputs.append(out)

        # Neural expert for residual/fallback
        neural_out = self.experts[-1](noisy)
        expert_outputs.append(neural_out)

        # Stack: [B, 5, H, W]
        expert_stack = torch.stack(expert_outputs, dim=1)

        # Add neural weight to noise_type_map
        neural_weight = 1.0 - noise_type_map.sum(dim=1, keepdim=True).clamp(0, 1)
        weights = torch.cat([noise_type_map, neural_weight], dim=1)  # [B, 5, H, W]

        # Per-pixel weighted combination
        output = (expert_stack * weights.unsqueeze(2)).sum(dim=1)  # [B, 1, H, W]

        return output
```

**Why Novel**:
- First work to combine **Mixture of Experts** with **Symbolic Operators**
- Per-pixel routing enables spatially-adaptive denoising
- Neural fallback handles cases where symbolic operators fail

---

### Contribution 4: Physics-Constrained Loss Functions

**Current Problem**: Only L1/L2 reconstruction loss, no physics constraints.

**Our Innovation**: Loss functions that enforce **physical consistency** of noise decomposition.

```python
class PhysicsConstrainedLoss(nn.Module):
    """
    Loss functions that enforce physical properties of noise.
    """
    def __init__(self):
        pass

    def forward(self, noisy, clean, pred_noise_type, pred_noise_level, denoised):
        # 1. Reconstruction loss
        L_recon = F.l1_loss(denoised, clean)

        # 2. Noise level consistency: predicted level should match actual noise
        actual_noise = noisy - clean
        actual_level = actual_noise.abs().mean(dim=[2,3], keepdim=True)
        L_level = F.mse_loss(pred_noise_level.mean(dim=[2,3], keepdim=True), actual_level)

        # 3. Speckle physics: multiplicative noise → CV ≈ 1 in speckle regions
        # Where speckle weight is high, coefficient of variation should be high
        local_cv = compute_local_cv(noisy)
        speckle_weight = pred_noise_type[:, 0:1]  # Speckle channel
        L_speckle_physics = F.mse_loss(
            speckle_weight * local_cv,
            speckle_weight * torch.ones_like(local_cv)
        )

        # 4. Banding physics: periodic artifacts in Fourier domain
        # Where banding weight is high, should have strong vertical frequencies
        vertical_power = compute_vertical_fft_power(noisy)
        banding_weight = pred_noise_type[:, 1:2]  # Banding channel
        L_banding_physics = -torch.mean(banding_weight * vertical_power)  # Maximize correlation

        # 5. Shot physics: variance proportional to signal (Poisson)
        local_mean = F.avg_pool2d(noisy, 5, stride=1, padding=2)
        local_var = F.avg_pool2d((noisy - local_mean)**2, 5, stride=1, padding=2)
        shot_weight = pred_noise_type[:, 3:4]  # Shot channel
        L_shot_physics = F.mse_loss(
            shot_weight * local_var,
            shot_weight * local_mean  # Poisson: var = mean
        )

        # 6. Decomposition consistency: noise types should sum to 1
        L_sum = F.mse_loss(pred_noise_type.sum(dim=1), torch.ones_like(pred_noise_type[:, 0]))

        # Total loss
        L_total = (
            L_recon +
            0.1 * L_level +
            0.1 * L_speckle_physics +
            0.1 * L_banding_physics +
            0.1 * L_shot_physics +
            0.01 * L_sum
        )

        return L_total, {
            'recon': L_recon,
            'level': L_level,
            'speckle_physics': L_speckle_physics,
            'banding_physics': L_banding_physics,
            'shot_physics': L_shot_physics,
        }
```

**Why Novel**:
- First work to use **physics of each noise type** as training constraints
- Encourages physically meaningful noise decomposition
- Provides self-supervision signal beyond reconstruction

---

### Contribution 5: Interpretable Denoising with Clinical Transparency

**Current Problem**: Neural networks are black boxes for clinicians.

**Our Innovation**: Every step is **interpretable and visualizable**.

```python
class InterpretableSANSD(nn.Module):
    """
    Returns not just denoised image, but full interpretation.
    """
    def forward(self, noisy, return_interpretation=True):
        # 1. Noise decomposition (interpretable)
        noise_type, noise_level, uncertainty = self.decomposer(noisy)

        # 2. Expert outputs (each is a classical algorithm output)
        expert_outputs = self.modse.get_expert_outputs(noisy, noise_level)

        # 3. Final fusion
        denoised = self.modse.fuse(expert_outputs, noise_type)

        if return_interpretation:
            interpretation = {
                # Per-pixel noise type (clinician can see: "speckle here, banding there")
                'noise_type_map': noise_type,  # [B, 4, H, W]
                'dominant_noise': noise_type.argmax(dim=1),  # [B, H, W] - which type dominates

                # Per-pixel noise level (clinician can see: "high noise here")
                'noise_level_map': noise_level,  # [B, 1, H, W]

                # Uncertainty (clinician can see: "model is uncertain here")
                'uncertainty_map': uncertainty,  # [B, 1, H, W]

                # What each expert would do (clinician can compare)
                'speckle_denoised': expert_outputs[0],
                'banding_denoised': expert_outputs[1],
                'gaussian_denoised': expert_outputs[2],
                'shot_denoised': expert_outputs[3],

                # How experts were combined (clinician can see weights)
                'expert_weights': noise_type,  # Same as noise_type_map
            }
            return denoised, interpretation

        return denoised
```

**Why Novel**:
- Every pixel has an explanation (which noise type, what level, which expert)
- Clinicians can override (e.g., "apply more speckle denoising here")
- Builds trust in AI-assisted diagnosis

---

## Complete Architecture: SANS-D

```python
class SANSD(nn.Module):
    """
    Spatially-Adaptive Neuro-Symbolic Denoiser

    Novel contributions:
    1. Per-pixel noise decomposition (type + level)
    2. Differentiable symbolic experts with learnable parameters
    3. Mixture of experts with per-pixel gating
    4. Physics-constrained training
    5. Full interpretability
    """

    def __init__(self, num_noise_types=4):
        super().__init__()

        # Module 1: Spatial Noise Decomposer
        self.decomposer = SpatialNoiseDecomposer(
            out_channels=num_noise_types + 2  # types + level + uncertainty
        )

        # Module 2: Differentiable Symbolic Experts
        self.experts = nn.ModuleDict({
            'speckle': LearnableAnisotropicDiffusion(
                learnable_K=True, learnable_tau=True, learnable_iterations=True
            ),
            'banding': LearnableFourierNotch(
                learnable_frequencies=True, learnable_bandwidth=True
            ),
            'gaussian': LearnableNonLocalMeans(
                learnable_h=True, learnable_patch_size=True
            ),
            'shot': LearnableVarianceStabilizing(
                learnable_transform=True
            ),
            'neural': LightweightResidualNet(width=32, depth=4),  # Fallback
        })

        # Module 3: Uncertainty-Weighted Fusion
        self.fusion = UncertaintyAwareFusion()

    def forward(self, noisy, return_all=False):
        B, C, H, W = noisy.shape

        # Step 1: Per-pixel noise analysis
        noise_type, noise_level, uncertainty = self.decomposer(noisy)
        # noise_type: [B, 4, H, W] - per-pixel noise type distribution
        # noise_level: [B, 1, H, W] - per-pixel noise intensity
        # uncertainty: [B, 1, H, W] - prediction confidence

        # Step 2: Apply each differentiable symbolic expert
        expert_outputs = {}
        for name, expert in self.experts.items():
            if name == 'neural':
                expert_outputs[name] = expert(noisy)
            else:
                # Symbolic experts receive noise level for parameter adaptation
                expert_outputs[name] = expert(noisy, noise_level)

        # Step 3: Per-pixel weighted fusion
        # Stack expert outputs: [B, 5, H, W]
        stacked = torch.stack([
            expert_outputs['speckle'],
            expert_outputs['banding'],
            expert_outputs['gaussian'],
            expert_outputs['shot'],
            expert_outputs['neural'],
        ], dim=1).squeeze(2)  # [B, 5, H, W]

        # Create weight map including neural fallback
        neural_weight = uncertainty  # Use uncertainty as neural fallback weight
        weights = torch.cat([noise_type, neural_weight], dim=1)  # [B, 5, H, W]
        weights = F.softmax(weights, dim=1)  # Normalize

        # Per-pixel weighted combination
        denoised = (stacked * weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        if return_all:
            return denoised, {
                'noise_type': noise_type,
                'noise_level': noise_level,
                'uncertainty': uncertainty,
                'expert_outputs': expert_outputs,
                'fusion_weights': weights,
            }

        return denoised
```

---

## Training Strategy

### Phase 1: Expert Pre-training (Optional)
```python
# Pre-train each symbolic expert on synthetic noise of that type
for noise_type in ['speckle', 'banding', 'gaussian', 'shot']:
    train_expert_on_synthetic(experts[noise_type], noise_type)
```

### Phase 2: End-to-End Joint Training
```python
optimizer = AdamW([
    {'params': decomposer.parameters(), 'lr': 1e-4},
    {'params': experts.parameters(), 'lr': 1e-5},  # Lower LR for experts
    {'params': fusion.parameters(), 'lr': 1e-4},
])

for batch in dataloader:
    noisy, clean = batch

    # Forward
    denoised, interpretation = model(noisy, return_all=True)

    # Physics-constrained loss
    loss, loss_dict = physics_loss(
        noisy, clean,
        interpretation['noise_type'],
        interpretation['noise_level'],
        denoised
    )

    # Usage loss (ensure experts are actually used)
    usage_loss = compute_expert_usage_diversity(interpretation['fusion_weights'])

    # Total
    total_loss = loss + 0.1 * usage_loss

    total_loss.backward()
    optimizer.step()
```

---

## Expected Results

### Quantitative Improvements:
| Method | PSNR | SSIM | Interpretability |
|--------|------|------|------------------|
| NAFNet (baseline) | 30.5 dB | 0.89 | None |
| NAFNet + FiLM (your current) | 31.0 dB | 0.90 | Global weights only |
| NSNDModel (classical fusion) | 31.5 dB | 0.91 | Component outputs |
| **SANS-D (proposed)** | **33.0+ dB** | **0.93** | **Full per-pixel** |

### Qualitative Improvements:
1. **Spatially-varying denoising**: Different treatment for different regions
2. **Noise-level adaptation**: Stronger denoising where noise is higher
3. **Interpretable outputs**: Clinicians can see and understand decisions
4. **Robust fusion**: Neural fallback handles edge cases

---

## Implementation Roadmap

### Week 1: Core Components
- [ ] Implement SpatialNoiseDecomposer
- [ ] Implement LearnableAnisotropicDiffusion
- [ ] Implement LearnableFourierNotch
- [ ] Basic forward pass working

### Week 2: Training Infrastructure
- [ ] Implement PhysicsConstrainedLoss
- [ ] Implement expert usage loss
- [ ] Set up training pipeline
- [ ] Test on small subset

### Week 3: Full Training
- [ ] Train on full dataset
- [ ] Ablation studies
- [ ] Compare with baselines
- [ ] Tune hyperparameters

### Week 4: Evaluation & Visualization
- [ ] Comprehensive benchmarking
- [ ] Visualization tools for interpretation
- [ ] Qualitative analysis
- [ ] Write results

---

## Why This is Novel (Reviewer Response)

### vs. Prior Neural Methods:
"Unlike pure neural approaches (NAFNet, SwinIR), SANS-D maintains interpretability through explicit noise decomposition and symbolic operators while achieving superior performance through learned parameter adaptation."

### vs. Prior Classical Methods:
"Unlike classical methods with fixed parameters, SANS-D learns spatially-varying operator parameters end-to-end, enabling adaptation to local noise characteristics."

### vs. Prior Hybrid Methods:
"Unlike existing hybrid methods that use global noise weights, SANS-D performs per-pixel noise decomposition and per-pixel expert routing, enabling truly spatially-adaptive denoising."

### vs. Mixture of Experts:
"Unlike standard MoE with neural experts, SANS-D uses differentiable symbolic experts grounded in noise physics, providing both interpretability and strong inductive bias."

---

## Paper Positioning

**Title**: "SANS-D: Spatially-Adaptive Neuro-Symbolic Denoising via Mixture of Differentiable Symbolic Experts"

**Venue**: CVPR, ICCV, MICCAI, or TMI

**Key Claims**:
1. First per-pixel noise type and level estimation for adaptive denoising
2. First differentiable symbolic operators with learnable spatially-varying parameters
3. First mixture-of-experts framework with symbolic experts
4. Physics-constrained training for meaningful noise decomposition
5. State-of-the-art OCT denoising with full interpretability

This would be a **strong, novel contribution** with clear differentiation from prior work.
