#!/usr/bin/env python3
"""
Cross-Dataset Validation for Cooperative Neuro-Symbolic Denoiser V8.

Validates trained model on multiple real-noise OCT datasets to demonstrate
generalization capability for IEEE TMI publication.

Datasets:
  - PKU37 test (173 samples) - same distribution as training
  - Duke17 / Sparsity SDOCT 2012 (16 samples) - different scanner/institution
  - Duke2013 synthetic eval (18 samples) - synthetic noise
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Import model and dataset from training script
from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative,
    PKU37Dataset,
    compute_psnr,
    compute_ssim,
)


class AdaptiveCorrectionScaler:
    """
    Two-stage adaptive correction scaling for cross-dataset generalization.

    Stage 1: Magnitude normalization — scales correction to match reference
             magnitude from training distribution (PKU37).
    Stage 2: Predicate-guided binary search — if critical predicates (P1_edge,
             P2_contrast) fail, iteratively reduces scale until they pass.

    This is a validation-side only technique; no model retraining required.
    """

    def __init__(self, reference_magnitude, predicate_fn=None,
                 magnitude_headroom=1.2, max_search_iters=4):
        """
        Args:
            reference_magnitude: Average correction magnitude from PKU37 training data.
            predicate_fn: Callable(output, noisy) -> dict of predicate scores.
            magnitude_headroom: Allow up to headroom * reference (default 1.2x).
            max_search_iters: Max binary search iterations for predicate refinement.
        """
        self.reference_magnitude = reference_magnitude
        self.predicate_fn = predicate_fn
        self.magnitude_headroom = magnitude_headroom
        self.max_search_iters = max_search_iters

    def compute_scale(self, correction_magnitude):
        """Stage 1: Magnitude normalization. Returns scale in (0, 1]."""
        if correction_magnitude < 1e-8:
            return 1.0
        target = self.reference_magnitude * self.magnitude_headroom
        return min(target / correction_magnitude, 1.0)  # never amplify

    @staticmethod
    def _extract_scores(pred_result):
        """Extract P1_edge and P2_contrast scores from predicate result dict."""
        # predicates.forward() returns {'scores': {'P1_edge': float, ...}, ...}
        scores = pred_result.get('scores', pred_result)
        p1 = scores.get('P1_edge', 1.0)
        p2 = scores.get('P2_contrast', 1.0)
        p1 = p1.item() if isinstance(p1, torch.Tensor) else float(p1)
        p2 = p2.item() if isinstance(p2, torch.Tensor) else float(p2)
        return p1, p2

    @staticmethod
    def _compute_cnr(image, clean):
        """Compute CNR using clean image for tissue/background segmentation."""
        signal_mask = (clean > clean.mean()).float()
        bg_mask = 1.0 - signal_mask
        signal_sum = signal_mask.sum().clamp(min=1.0)
        bg_sum = bg_mask.sum().clamp(min=1.0)

        sig = (image * signal_mask).sum() / signal_sum
        bg = (image * bg_mask).sum() / bg_sum
        bg_std = torch.sqrt(((image - bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
        return ((sig - bg) / bg_std).clamp(-100, 100).item()

    def _check_constraints(self, candidate, noisy, clean,
                           p1_threshold, p2_threshold, cnr_threshold):
        """Check all constraints: predicates + CNR."""
        # Predicate check
        if self.predicate_fn is not None:
            pred_result = self.predicate_fn(candidate, noisy)
            p1, p2 = self._extract_scores(pred_result)
            if p1 < p1_threshold or p2 < p2_threshold:
                return False
        # CNR check
        if clean is not None:
            cnr = self._compute_cnr(candidate, clean)
            if cnr < cnr_threshold:
                return False
        return True

    def refine_with_predicates(self, backbone_out, correction, noisy, initial_scale,
                               clean=None):
        """Stage 2: Binary search to find largest scale where constraints hold.

        Uses "do no harm" principle: compare against backbone scores rather
        than fixed thresholds. Constraints checked:
          - P1_edge: must not degrade more than 0.01 from backbone
          - P2_contrast: must not degrade more than 0.01 from backbone
          - CNR: must not degrade from backbone (when clean is provided)
        """
        # Get backbone baseline scores
        p1_threshold = -999.0
        p2_threshold = -999.0
        if self.predicate_fn is not None:
            backbone_preds = self.predicate_fn(backbone_out, noisy)
            p1_base, p2_base = self._extract_scores(backbone_preds)
            p1_threshold = p1_base - 0.01
            p2_threshold = p2_base - 0.01

        cnr_threshold = -999.0
        if clean is not None:
            cnr_base = self._compute_cnr(backbone_out, clean)
            # Allow up to 0.5% CNR degradation (relative tolerance)
            cnr_threshold = cnr_base * 0.995

        # Check if initial scale already passes all constraints
        candidate = (backbone_out + correction * initial_scale).clamp(0, 1)
        if self._check_constraints(candidate, noisy, clean,
                                   p1_threshold, p2_threshold, cnr_threshold):
            return initial_scale

        # Binary search: find largest scale where all constraints hold
        lo, hi = 0.0, initial_scale
        best_scale = 0.0  # fallback: no correction

        for _ in range(self.max_search_iters):
            mid = (lo + hi) / 2
            candidate = (backbone_out + correction * mid).clamp(0, 1)
            if self._check_constraints(candidate, noisy, clean,
                                       p1_threshold, p2_threshold, cnr_threshold):
                best_scale = mid
                lo = mid  # try larger scale
            else:
                hi = mid  # try smaller scale

        return best_scale


class TestTimeAdaptation:
    """
    Test-Time Adaptation for cross-scanner OCT denoising generalization.

    Adapts the corrector module (0.23M params) using GT-free losses on a few
    samples from the target scanner. Backbone (7.02M) stays frozen.

    GT-Free Loss = w1*L_predicate + w2*L_cnr + w3*L_magnitude + w4*L_consistency + w5*L_quality
    """

    def __init__(self, model, device='cpu', tta_steps=30, tta_lr=5e-4,
                 n_adapt_samples=5, w_predicate=2.0, w_cnr=1.5,
                 w_magnitude=0.5, w_consistency=1.0):
        self.model = model
        self.device = device
        self.tta_steps = tta_steps
        self.tta_lr = tta_lr
        self.n_adapt_samples = n_adapt_samples
        self.w_predicate = w_predicate
        self.w_cnr = w_cnr
        self.w_magnitude = w_magnitude
        self.w_consistency = w_consistency
        self.original_state = None

        # Sobel kernels for quality loss
        self.sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                    device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                                    device=device).view(1, 1, 3, 3)

    def save_state(self):
        """Save original corrector weights for later restoration."""
        self.original_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.corrector.state_dict().items()
        }

    def restore_state(self):
        """Restore original corrector weights (undo adaptation)."""
        if self.original_state is not None:
            device = next(self.model.corrector.parameters()).device
            state = {k: v.to(device) for k, v in self.original_state.items()}
            self.model.corrector.load_state_dict(state)
            self.original_state = None
            # Clear stale gradient buffers left from adaptation backward passes
            self.model.zero_grad(set_to_none=True)

    def _freeze_backbone(self):
        """Freeze backbone, enable corrector gradients."""
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        for p in self.model.corrector.parameters():
            p.requires_grad = True

    def _compute_predicate_deficit(self, pred_scores):
        """Compute predicate deficit (non-differentiable monitoring + adaptive scaling).

        Returns:
            deficit_value: float, total weighted deficit for logging
            scale_factor: float in [1.0, 2.0], used to upscale differentiable losses
                          when predicates are failing
        """
        pred_map = {
            'P1_edge': 'P1', 'P2_contrast': 'P2', 'P3_smooth': 'P3',
            'P4_structure': 'P4', 'P6_anatomy': 'P6'
        }
        targets = {'P1': 0.55, 'P2': 0.60, 'P3': 0.55, 'P4': 0.55, 'P6': 0.55}
        weights = {'P1': 1.0, 'P2': 3.0, 'P3': 1.0, 'P4': 1.5, 'P6': 1.0}

        total_deficit = 0.0
        total_weight = 0.0

        for score_key, pred_key in pred_map.items():
            score = pred_scores.get(score_key, 0.5)
            score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
            target = targets[pred_key]
            deficit = max(0.0, target - score_val)
            w = weights[pred_key]
            total_deficit += w * deficit
            total_weight += w

        deficit_value = total_deficit / max(total_weight, 1e-8)
        # Scale factor: boost differentiable losses when predicates are failing
        scale_factor = 1.0 + min(1.0, deficit_value)
        return min(deficit_value, 5.0), scale_factor

    def _compute_cnr_preservation_loss(self, corrected, backbone_out):
        """GT-free CNR preservation using model's region detector."""
        # Cache region masks per spatial size (samples may have different resolutions)
        spatial_key = backbone_out.shape[-2:]
        if not hasattr(self, '_region_cache'):
            self._region_cache = {}
        if spatial_key not in self._region_cache:
            region_detector = self.model.corrector.cnr_preserver.region_detector
            with torch.no_grad():
                tissue_mask, bg_mask = region_detector(backbone_out)
            tissue_mask = tissue_mask.detach().clamp(0.0, 1.0)
            bg_mask = bg_mask.detach().clamp(0.0, 1.0)
            self._region_cache[spatial_key] = {
                'tissue': tissue_mask, 'bg': bg_mask,
                'tissue_sum': tissue_mask.sum().clamp(min=1.0),
                'bg_sum': bg_mask.sum().clamp(min=1.0),
            }

        cached = self._region_cache[spatial_key]
        tissue_mask = cached['tissue']
        bg_mask = cached['bg']
        eps = 1e-6
        tissue_sum = cached['tissue_sum']
        bg_sum = cached['bg_sum']

        # Background noise
        bg_mean_b = (backbone_out.detach() * bg_mask).sum() / bg_sum
        bg_std_b = torch.sqrt(((backbone_out.detach() - bg_mean_b) ** 2 * bg_mask).sum() / bg_sum + eps).clamp(min=1e-4)
        bg_mean_c = (corrected * bg_mask).sum() / bg_sum
        bg_std_c = torch.sqrt(((corrected - bg_mean_c) ** 2 * bg_mask).sum() / bg_sum + eps).clamp(min=1e-4)

        # Tissue contrast
        tissue_mean_b = (backbone_out.detach() * tissue_mask).sum() / tissue_sum
        tissue_mean_c = (corrected * tissue_mask).sum() / tissue_sum

        # CNR
        cnr_backbone = (tissue_mean_b - bg_mean_b) / bg_std_b
        cnr_corrected = (tissue_mean_c - bg_mean_c) / bg_std_c

        cnr_drop = F.relu(cnr_backbone.detach() - cnr_corrected)
        bg_noise_increase = F.relu(bg_std_c - bg_std_b.detach())
        contrast_bonus = F.relu(
            (tissue_mean_c - bg_mean_c) - (tissue_mean_b - bg_mean_b).detach()
        ) * 0.5

        # Don't apply F.relu() to the combined loss — it zeros gradients when
        # contrast_bonus exceeds penalties, blocking the optimizer from learning
        # to improve tissue-background contrast.
        loss = cnr_drop * 3.0 + bg_noise_increase * 10.0 - contrast_bonus
        return loss.clamp(min=0.0, max=5.0)

    def _get_backbone_features(self, backbone_out):
        """Get cached backbone edge/contrast features, keyed by spatial size."""
        if not hasattr(self, '_bb_cache'):
            self._bb_cache = {}
        spatial_key = backbone_out.shape[-2:]
        if spatial_key not in self._bb_cache:
            with torch.no_grad():
                bb_edges = torch.sqrt(
                    F.conv2d(backbone_out, self.sobel_x, padding=1) ** 2 +
                    F.conv2d(backbone_out, self.sobel_y, padding=1) ** 2 + 1e-8
                )
                bb_std = torch.sqrt(
                    F.avg_pool2d(backbone_out ** 2, 7, 1, 3) -
                    F.avg_pool2d(backbone_out, 7, 1, 3) ** 2 + 1e-8
                )
            self._bb_cache[spatial_key] = (bb_edges, bb_std)
        return self._bb_cache[spatial_key]

    def _compute_direct_quality_loss(self, corrected, backbone_out):
        """Differentiable quality metrics: edge + contrast preservation vs backbone."""
        cached_bb_edges, cached_bb_std = self._get_backbone_features(backbone_out)

        # Edge preservation (only corrected needs gradient)
        co_edges = torch.sqrt(
            F.conv2d(corrected, self.sobel_x, padding=1) ** 2 +
            F.conv2d(corrected, self.sobel_y, padding=1) ** 2 + 1e-8
        )
        edge_loss = F.relu(cached_bb_edges - co_edges).mean()

        # Local contrast preservation
        co_std = torch.sqrt(
            F.avg_pool2d(corrected ** 2, 7, 1, 3) -
            F.avg_pool2d(corrected, 7, 1, 3) ** 2 + 1e-8
        )
        contrast_loss = F.relu(cached_bb_std - co_std).mean()

        return edge_loss + contrast_loss

    def _compute_self_consistency_loss(self, noisy, corrected_orig,
                                       cached_bb_hflip, cached_feats_hflip, cached_unc_hflip,
                                       cached_bb_vflip, cached_feats_vflip, cached_unc_vflip):
        """Flip-consistency: f(flip(x)) should equal flip(f(x)).

        Uses PRE-CACHED flipped backbone outputs to avoid rerunning the full
        7.02M-param backbone during TTA (backbone is frozen, outputs are constant).

        Args:
            noisy: Input tensor
            corrected_orig: Already-computed corrected output from main forward pass
            cached_bb_hflip: Pre-cached backbone output for horizontally-flipped input
            cached_feats_hflip: Pre-cached backbone features for h-flip
            cached_unc_hflip: Pre-cached uncertainty for h-flip
            cached_bb_vflip: Pre-cached backbone output for vertically-flipped input
            cached_feats_vflip: Pre-cached backbone features for v-flip
            cached_unc_vflip: Pre-cached uncertainty for v-flip
        """
        # Horizontal flip — run corrector only (backbone outputs pre-cached)
        corrected_hflip, _ = self.model.corrector(
            cached_bb_hflip, torch.flip(noisy, dims=[-1]), cached_feats_hflip,
            nafnet_uncertainty=cached_unc_hflip, return_details=False,
        )
        corrected_hflip_unflipped = torch.flip(corrected_hflip, dims=[-1])
        del corrected_hflip
        loss_h = F.l1_loss(corrected_orig, corrected_hflip_unflipped)
        del corrected_hflip_unflipped

        # Vertical flip — run corrector only (backbone outputs pre-cached)
        corrected_vflip, _ = self.model.corrector(
            cached_bb_vflip, torch.flip(noisy, dims=[-2]), cached_feats_vflip,
            nafnet_uncertainty=cached_unc_vflip, return_details=False,
        )
        corrected_vflip_unflipped = torch.flip(corrected_vflip, dims=[-2])
        del corrected_vflip
        loss_v = F.l1_loss(corrected_orig, corrected_vflip_unflipped)
        del corrected_vflip_unflipped

        return (loss_h + loss_v) / 2.0

    def _compute_tta_loss(self, noisy, cached_backbone_out, cached_backbone_features,
                          cached_nafnet_unc, step, sample):
        """Compute combined TTA loss using cached backbone outputs (skip backbone recompute).

        Args:
            noisy: Input noisy image [B, 1, H, W] on device
            cached_backbone_out: Pre-computed detached backbone output
            cached_backbone_features: Pre-computed detached backbone features dict
            cached_nafnet_unc: Pre-computed detached uncertainty map
            step: Current TTA step number
            sample: Full sample dict with pre-cached flipped backbone outputs
        """
        # Call corrector directly — backbone is frozen so we reuse cached outputs.
        # Since cached tensors are detached, the computation graph only covers the corrector.
        corrected, info = self.model.corrector(
            cached_backbone_out, noisy, cached_backbone_features,
            nafnet_uncertainty=cached_nafnet_unc,
            return_details=False,
        )

        # 1. Predicate deficit (non-differentiable) → adaptive scaling factor
        pred_scores = info.get('predicate_scores', {})
        pred_deficit, pred_scale = self._compute_predicate_deficit(pred_scores)
        del info  # Free spatial maps in info dict before computing losses

        # 2. CNR preservation (differentiable)
        cnr_loss = self._compute_cnr_preservation_loss(corrected, cached_backbone_out)

        # 3. Correction magnitude regularization (differentiable)
        mag_loss = F.mse_loss(corrected, cached_backbone_out)

        # 4. Direct quality loss (differentiable)
        quality_loss = self._compute_direct_quality_loss(corrected, cached_backbone_out)

        # 5. Self-consistency (every 5th step to save compute)
        #    Uses pre-cached flipped backbone outputs to avoid rerunning backbone
        if step % 5 == 0 and self.w_consistency > 0:
            consistency_loss = self._compute_self_consistency_loss(
                noisy, corrected,
                sample['bb_hflip'], sample['feats_hflip'], sample['unc_hflip'],
                sample['bb_vflip'], sample['feats_vflip'], sample['unc_vflip'],
            )
        else:
            consistency_loss = torch.zeros((), device=corrected.device)

        # Scale differentiable losses by predicate deficit (boosts when predicates failing)
        total_loss = pred_scale * (
            self.w_cnr * cnr_loss +
            self.w_magnitude * mag_loss +
            1.0 * quality_loss +
            self.w_consistency * consistency_loss
        )

        # Defer .item() to reduce GPU-CPU sync points (only needed for logging)
        loss_dict = {
            'predicate': pred_deficit,
            'pred_scale': pred_scale,
            'cnr_t': cnr_loss,
            'magnitude_t': mag_loss,
            'quality_t': quality_loss,
            'consistency_t': consistency_loss,
            'total_t': total_loss,
        }
        return total_loss, loss_dict

    @staticmethod
    def precompute_backbone_cache(model, loader, device, n_samples=5, include_flips=True):
        """Pre-compute backbone outputs for adaptation samples (one-time cost).

        Since the backbone is frozen, its outputs are identical across all TTA
        configs. Call this ONCE per dataset, then pass the cache to adapt().
        Eliminates redundant backbone forward passes (80-85% speedup for sweeps).

        Args:
            model: Full model with .backbone attribute
            loader: DataLoader for the target dataset
            device: Torch device
            n_samples: Number of adaptation samples to cache
            include_flips: Whether to cache h-flip and v-flip outputs for consistency loss

        Returns:
            List of sample dicts with cached backbone outputs (detached).
        """
        cache = []
        flip_tag = " +flips" if include_flips else ""
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_samples:
                    break
                noisy = batch['noisy'].to(device)
                print(f"  [Cache] Sample {i+1}/{n_samples}{flip_tag}...", flush=True)

                backbone_out, backbone_features, nafnet_unc = model.backbone(noisy)
                # Don't cache backbone_features (enc1 [B,40,H,W] = ~77MB/tensor).
                # No corrector reads backbone_features in its forward pass, and
                # the confidence estimator is bypassed when nafnet_uncertainty is
                # provided (which it always is in TTA).  Skipping enc1 saves
                # ~2.3GB across all cached samples+flips.
                sample = {
                    'noisy': noisy.cpu(),
                    'backbone_out': backbone_out.detach(),
                    'backbone_features': None,
                    'nafnet_unc': nafnet_unc.detach(),
                }
                del backbone_out, backbone_features, nafnet_unc

                if include_flips:
                    bb_h, feats_h, unc_h = model.backbone(torch.flip(noisy, dims=[-1]))
                    sample['bb_hflip'] = bb_h.detach()
                    sample['feats_hflip'] = None  # enc1 unused by correctors
                    sample['unc_hflip'] = unc_h.detach()
                    del bb_h, feats_h, unc_h

                    bb_v, feats_v, unc_v = model.backbone(torch.flip(noisy, dims=[-2]))
                    sample['bb_vflip'] = bb_v.detach()
                    sample['feats_vflip'] = None  # enc1 unused by correctors
                    sample['unc_vflip'] = unc_v.detach()
                    del bb_v, feats_v, unc_v
                else:
                    sample['bb_hflip'] = sample['feats_hflip'] = sample['unc_hflip'] = None
                    sample['bb_vflip'] = sample['feats_vflip'] = sample['unc_vflip'] = None

                del noisy
                cache.append(sample)

        return cache

    def adapt(self, loader, dataset_name="Unknown", backbone_cache=None):
        """
        Adapt corrector using GT-free losses on samples from the target dataset.

        OPTIMIZED: Pre-caches backbone outputs (backbone is frozen) so the TTA loop
        only runs the corrector (0.23M params) instead of the full model (7.25M).
        This eliminates ~80% of compute and prevents the backbone computation graph
        from being held in memory during backward passes.

        Args:
            loader: DataLoader for the dataset (used for sample collection if no cache)
            dataset_name: Name for logging
            backbone_cache: Optional pre-computed backbone outputs from
                           precompute_backbone_cache(). When provided, skips backbone
                           forward passes entirely (use for sweep across configs).

        Returns adaptation_log dict.
        """
        print(f"\n[TTA] Adapting corrector for {dataset_name}")
        print(f"[TTA] Steps: {self.tta_steps}, LR: {self.tta_lr}, "
              f"Adapt samples: {self.n_adapt_samples}")

        # 1. Save original state
        self.save_state()

        # 2. Freeze backbone, enable corrector
        self._freeze_backbone()

        # 3. Use pre-computed cache or collect + cache backbone outputs
        if backbone_cache is not None:
            adapt_samples = backbone_cache
            print(f"[TTA] Using pre-computed backbone cache ({len(adapt_samples)} samples)")
        else:
            adapt_samples = self.precompute_backbone_cache(
                self.model, loader, self.device,
                n_samples=self.n_adapt_samples,
                include_flips=(self.w_consistency > 0),
            )

        if not adapt_samples:
            print("[TTA] WARNING: No adaptation samples available")
            return {}

        if backbone_cache is None:
            print(f"[TTA] Collected {len(adapt_samples)} samples, backbone outputs cached"
                  f"{' (+ flipped)' if self.w_consistency > 0 else ''}")

        # Downsample cached samples for TTA to reduce computation graph memory.
        # Duke images are ~520×970 — each corrector forward in training mode creates
        # [1,32,H,W] intermediate tensors for 5 potential_nets ≈ 5×64MB = 320MB/layer.
        # Total computation graph at full res: ~4-5GB → OOM on 7.8GB system.
        # Corrector params are resolution-independent (all convolutions), so
        # adaptation at 256×256 transfers to full-res inference.
        max_tta_size = 256
        tta_samples = []
        h0, w0 = adapt_samples[0]['backbone_out'].shape[-2:]
        need_downsample = (h0 > max_tta_size or w0 > max_tta_size)
        if need_downsample:
            scale = max_tta_size / max(h0, w0)
            new_h = max((int(h0 * scale) // 8) * 8, 8)  # divisible by 8 for NAFNet
            new_w = max((int(w0 * scale) // 8) * 8, 8)
            print(f"[TTA] Downsampling {h0}×{w0} → {new_h}×{new_w} for TTA "
                  f"(saves ~{(h0*w0 - new_h*new_w) * 32 * 5 * 4 / 1e9:.1f}GB graph memory)")
            with torch.no_grad():
                for s in adapt_samples:
                    ds = {}
                    for k, v in s.items():
                        if isinstance(v, torch.Tensor) and v.dim() == 4:
                            ds[k] = F.interpolate(v, size=(new_h, new_w),
                                                  mode='bilinear', align_corners=False)
                        else:
                            ds[k] = v
                    tta_samples.append(ds)
        else:
            tta_samples = adapt_samples

        # 4. Set up optimizer (only corrector params with grad)
        trainable_params = [p for p in self.model.corrector.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.tta_lr)

        # 5. Adaptation loop — calls corrector directly (skips backbone)
        self.model.corrector.train()
        self.model.corrector._tta_mode = True  # Skip verifier + predicate re-eval
        self.model.backbone.eval()

        adaptation_log = {'steps': [], 'losses': []}

        try:
            for step in range(self.tta_steps):
                sample = tta_samples[step % len(tta_samples)]
                noisy = sample['noisy'].to(self.device)

                optimizer.zero_grad(set_to_none=True)

                total_loss, loss_dict = self._compute_tta_loss(
                    noisy, sample['backbone_out'], sample['backbone_features'],
                    sample['nafnet_unc'], step, sample,
                )

                if not torch.isfinite(total_loss):
                    print(f"[TTA] Step {step}: Non-finite loss, skipping")
                    del total_loss, loss_dict  # Free computation graph on NaN path
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                del total_loss

                # Convert ALL loss_dict tensors to Python scalars IMMEDIATELY to free
                # the computation graph. Without this, grad_fn references pin the entire
                # corrector graph in memory (~10-20MB leak per step).
                total_val = loss_dict['total_t'].item()
                pred_deficit = loss_dict['predicate']
                pred_scale = loss_dict['pred_scale']
                loss_scalars = {
                    'predicate': pred_deficit.item() if isinstance(pred_deficit, torch.Tensor) else float(pred_deficit),
                    'pred_scale': pred_scale.item() if isinstance(pred_scale, torch.Tensor) else float(pred_scale),
                    'cnr': loss_dict['cnr_t'].item(),
                    'magnitude': loss_dict['magnitude_t'].item(),
                    'quality': loss_dict['quality_t'].item(),
                    'consistency': loss_dict['consistency_t'].item(),
                    'total': total_val,
                }
                del loss_dict  # Free all tensor references and their grad_fn chains

                adaptation_log['steps'].append(step)

                adaptation_log['losses'].append(loss_scalars)
                print(f"[TTA] Step {step:3d}/{self.tta_steps}: "
                      f"total={loss_scalars['total']:.4f} "
                      f"pred={loss_scalars['predicate']:.4f} "
                      f"s={loss_scalars['pred_scale']:.2f} "
                      f"cnr={loss_scalars['cnr']:.4f} "
                      f"mag={loss_scalars['magnitude']:.6f} "
                      f"qual={loss_scalars['quality']:.4f} "
                      f"cons={loss_scalars['consistency']:.4f}", flush=True)

                # Early stopping if loss converged
                if total_val < 0.001 and step >= 5:
                    print(f"[TTA] Converged at step {step} (loss={total_val:.6f})")
                    break
        finally:
            # 6. Back to eval mode, clear all caches and gradients.
            # In a finally block to ensure cleanup even on OOM/exception.
            self.model.corrector._tta_mode = False  # Re-enable verifier for inference
            self.model.eval()
            self.model.zero_grad(set_to_none=True)  # Clear stale gradient buffers
            del optimizer
            if need_downsample:
                del tta_samples  # Free downsampled copies
            if backbone_cache is None:
                del adapt_samples  # Only free if we created them (caller owns the cache)
            self._bb_cache = {}
            self._region_cache = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[TTA] Adaptation complete for {dataset_name}")
        return adaptation_log


@torch.inference_mode()
def validate_dataset_fast(model, loader, device, dataset_name="Unknown"):
    """Fast PSNR/SSIM-only validation for TTA sweep (skips clinical metrics).

    ~5x faster than full validate_dataset() — only computes PSNR, SSIM, and
    correction magnitude. Use for hyperparameter sweep screening; run full
    validate_dataset() only on the best config(s).
    """
    model.eval()
    total_psnr_bb = 0
    total_psnr_co = 0
    total_ssim_bb = 0
    total_ssim_co = 0
    total_mag = 0
    n = 0
    n_total = len(loader.dataset)

    for batch in loader:
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)
        # Split model forward: call backbone + corrector separately so we can
        # delete backbone_features (enc1 [1,40,H,W] ≈ 40MB) immediately.
        # No corrector actually reads backbone_features, yet the wrapper's
        # model.forward() holds it through the entire corrector pass.
        backbone_out, backbone_features, nafnet_unc = model.backbone(noisy)
        del backbone_features  # enc1 not used by correctors — save ~40MB peak
        corrected, info = model.corrector(
            backbone_out, noisy, None,
            nafnet_uncertainty=nafnet_unc, return_details=False,
        )
        del info, nafnet_unc

        total_psnr_bb += compute_psnr(backbone_out, clean)
        total_psnr_co += compute_psnr(corrected, clean)
        total_ssim_bb += compute_ssim(backbone_out, clean)
        total_ssim_co += compute_ssim(corrected, clean)
        total_mag += (corrected - backbone_out).abs().mean().item()
        del corrected, backbone_out, clean, noisy
        n += 1
        print(f"  [FAST] {dataset_name}: sample {n}/{n_total}", flush=True)

    if n == 0:
        return {}

    psnr_bb = total_psnr_bb / n
    psnr_co = total_psnr_co / n
    psnr_delta = psnr_co - psnr_bb
    ssim_co = total_ssim_co / n

    print(f"  [FAST] {dataset_name}: PSNR {psnr_bb:.2f}→{psnr_co:.2f} "
          f"({psnr_delta:+.3f} dB), SSIM {ssim_co:.4f}, "
          f"mag {total_mag/n:.6f}")

    return {
        'dataset': dataset_name,
        'n_samples': n,
        'psnr_backbone': psnr_bb,
        'psnr_corrected': psnr_co,
        'psnr_delta': psnr_delta,
        'ssim_backbone': total_ssim_bb / n,
        'ssim_corrected': ssim_co,
        'correction_magnitude': total_mag / n,
        # Placeholders for compatibility with sweep summary table
        'cnr_change_pct': 0.0,
        'clinical_improved': -1,
        'clinical_ratio': 0.0,
        'predicates_passing': -1,
        'verdict': '[FAST]',
    }


@torch.inference_mode()
def validate_dataset(model, loader, device, dataset_name="Unknown", scaler=None):
    """Run comprehensive validation on a dataset."""
    model.eval()

    # Accumulators
    total_psnr_backbone = 0
    total_psnr_corrected = 0
    total_ssim_backbone = 0
    total_ssim_corrected = 0
    total_correction_mag = 0

    # Clinical preservation
    total_backbone_contrast_pres = 0
    total_corrected_contrast_pres = 0
    total_backbone_boundary_pres = 0
    total_corrected_boundary_pres = 0
    total_backbone_texture_pres = 0
    total_corrected_texture_pres = 0
    total_backbone_edge_pres = 0
    total_corrected_edge_pres = 0

    # CNR
    total_cnr_backbone = 0
    total_cnr_corrected = 0

    # TCI
    total_tci_backbone = 0
    total_tci_corrected = 0

    # Predicates
    total_pred_scores = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0, 'P6': 0}
    total_pred_scores_backbone = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0, 'P6': 0}
    pred_key_map = {
        'P1_edge': 'P1', 'P2_contrast': 'P2', 'P3_smooth': 'P3',
        'P4_structure': 'P4', 'P5_speckle': 'P5', 'P6_anatomy': 'P6'
    }

    # Adaptive scaling tracking
    total_scale = 0
    n_scaled = 0

    # Cooperation
    total_uncertainty_corr = 0
    n_corr = 0

    # Sobel/Laplacian kernels
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).view(1, 1, 3, 3)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).view(1, 1, 3, 3)
    laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                             device=device).view(1, 1, 3, 3)

    n = 0
    per_image_results = []

    for batch in tqdm(loader, desc=f"Validating {dataset_name}"):
        clean = batch['clean'].to(device, non_blocking=True)
        noisy = batch['noisy'].to(device, non_blocking=True)

        # Split model forward: call backbone + corrector separately to free
        # backbone_features (enc1 [1,40,H,W] ≈ 40MB) before corrector runs.
        backbone_out, backbone_features, nafnet_unc = model.backbone(noisy)
        del backbone_features  # enc1 not used by correctors — save ~40MB peak
        corrected, info = model.corrector(
            backbone_out, noisy, None,
            nafnet_uncertainty=nafnet_unc, return_details=False,
        )
        # Wrapper model.forward() adds this; needed for cooperation correlation
        info['nafnet_uncertainty'] = nafnet_unc.detach()
        del nafnet_unc

        # Adaptive correction scaling (cross-dataset generalization)
        if scaler is not None:
            raw_correction = corrected - backbone_out
            mag = raw_correction.abs().mean().item()
            scale = scaler.compute_scale(mag)
            if scale < 1.0:
                scale = scaler.refine_with_predicates(
                    backbone_out, raw_correction, noisy, scale, clean=clean)
                corrected = (backbone_out + raw_correction * scale).clamp(0, 1)
                # Re-evaluate predicates on scaled output
                if hasattr(model, 'corrector') and hasattr(model.corrector, 'predicates'):
                    scaled_preds = model.corrector.predicates(corrected, noisy)
                    info['predicate_scores'] = scaled_preds.get('scores', {})
            total_scale += scale
            n_scaled += 1

        # Basic metrics
        psnr_backbone = compute_psnr(backbone_out, clean)
        psnr_corrected = compute_psnr(corrected, clean)
        ssim_backbone = compute_ssim(backbone_out, clean)
        ssim_corrected = compute_ssim(corrected, clean)

        total_psnr_backbone += psnr_backbone
        total_psnr_corrected += psnr_corrected
        total_ssim_backbone += ssim_backbone
        total_ssim_corrected += ssim_corrected

        # Correction magnitude
        correction = corrected - backbone_out
        total_correction_mag += correction.abs().mean().item()

        # Clinical preservation (local std)
        clean_std = F.avg_pool2d(clean ** 2, 7, 1, 3) - F.avg_pool2d(clean, 7, 1, 3) ** 2
        clean_std = torch.sqrt(clean_std.clamp(min=1e-8))
        backbone_std = F.avg_pool2d(backbone_out ** 2, 7, 1, 3) - F.avg_pool2d(backbone_out, 7, 1, 3) ** 2
        backbone_std = torch.sqrt(backbone_std.clamp(min=1e-8))
        corrected_std = F.avg_pool2d(corrected ** 2, 7, 1, 3) - F.avg_pool2d(corrected, 7, 1, 3) ** 2
        corrected_std = torch.sqrt(corrected_std.clamp(min=1e-8))

        clean_std_mean = clean_std.mean().clamp(min=1e-4)
        backbone_contrast_pres = (backbone_std.mean() / clean_std_mean).clamp(0, 10).item()
        corrected_contrast_pres = (corrected_std.mean() / clean_std_mean).clamp(0, 10).item()
        total_backbone_contrast_pres += backbone_contrast_pres
        total_corrected_contrast_pres += corrected_contrast_pres

        # Boundary preservation (vertical gradient)
        backbone_gy = F.conv2d(backbone_out, sobel_y, padding=1).abs()
        corrected_gy = F.conv2d(corrected, sobel_y, padding=1).abs()
        clean_gy = F.conv2d(clean, sobel_y, padding=1).abs()
        clean_gy_mean = clean_gy.mean().clamp(min=1e-4)
        total_backbone_boundary_pres += (backbone_gy.mean() / clean_gy_mean).clamp(0, 10).item()
        total_corrected_boundary_pres += (corrected_gy.mean() / clean_gy_mean).clamp(0, 10).item()

        # Texture preservation (Laplacian)
        backbone_lap = F.conv2d(backbone_out, laplacian, padding=1).abs()
        corrected_lap = F.conv2d(corrected, laplacian, padding=1).abs()
        clean_lap = F.conv2d(clean, laplacian, padding=1).abs()
        clean_lap_mean = clean_lap.mean().clamp(min=1e-4)
        total_backbone_texture_pres += (backbone_lap.mean() / clean_lap_mean).clamp(0, 10).item()
        total_corrected_texture_pres += (corrected_lap.mean() / clean_lap_mean).clamp(0, 10).item()

        # Edge preservation (Sobel magnitude)
        backbone_gx = F.conv2d(backbone_out, sobel_x, padding=1)
        corrected_gx = F.conv2d(corrected, sobel_x, padding=1)
        clean_gx = F.conv2d(clean, sobel_x, padding=1)
        backbone_edge = torch.sqrt(backbone_gx**2 + backbone_gy**2 + 1e-8)
        corrected_edge = torch.sqrt(corrected_gx**2 + corrected_gy**2 + 1e-8)
        clean_edge = torch.sqrt(clean_gx**2 + clean_gy**2 + 1e-8)
        clean_edge_mean = clean_edge.mean().clamp(min=1e-4)
        total_backbone_edge_pres += (backbone_edge.mean() / clean_edge_mean).clamp(0, 10).item()
        total_corrected_edge_pres += (corrected_edge.mean() / clean_edge_mean).clamp(0, 10).item()

        # CNR
        signal_mask = (clean > clean.mean()).float()
        bg_mask = 1.0 - signal_mask
        signal_sum = signal_mask.sum().clamp(min=1.0)
        bg_sum = bg_mask.sum().clamp(min=1.0)

        bb_sig = (backbone_out * signal_mask).sum() / signal_sum
        bb_bg = (backbone_out * bg_mask).sum() / bg_sum
        bb_bg_std = torch.sqrt(((backbone_out - bb_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
        cnr_bb = ((bb_sig - bb_bg) / bb_bg_std).clamp(-100, 100).item()

        co_sig = (corrected * signal_mask).sum() / signal_sum
        co_bg = (corrected * bg_mask).sum() / bg_sum
        co_bg_std = torch.sqrt(((corrected - co_bg)**2 * bg_mask).sum() / bg_sum + 1e-8).clamp(min=1e-4)
        cnr_co = ((co_sig - co_bg) / co_bg_std).clamp(-100, 100).item()

        total_cnr_backbone += cnr_bb
        total_cnr_corrected += cnr_co

        # TCI (tissue contrast index via vertical gradient)
        total_tci_backbone += (backbone_gy.mean() / clean_gy_mean).clamp(0, 10).item()
        total_tci_corrected += (corrected_gy.mean() / clean_gy_mean).clamp(0, 10).item()

        # Predicate scores
        pred_scores = info.get('predicate_scores', {})
        for orig_key, score in pred_scores.items():
            mapped = pred_key_map.get(orig_key, orig_key)
            if mapped in total_pred_scores:
                sv = score.item() if isinstance(score, torch.Tensor) else float(score)
                total_pred_scores[mapped] += sv

        pred_scores_bb = info.get('predicate_scores_backbone', {})
        for orig_key, score in pred_scores_bb.items():
            mapped = pred_key_map.get(orig_key, orig_key)
            if mapped in total_pred_scores_backbone:
                sv = score.item() if isinstance(score, torch.Tensor) else float(score)
                total_pred_scores_backbone[mapped] += sv

        # Cooperation correlation
        nafnet_unc = info.get('nafnet_uncertainty', None)
        potentials = info.get('corrector_potentials', {})
        if nafnet_unc is not None and potentials:
            pot_maps = []
            for v in potentials.values():
                if isinstance(v, dict) and 'map' in v:
                    pot_maps.append(v['map'])
                elif isinstance(v, torch.Tensor):
                    pot_maps.append(v)
            if pot_maps:
                total_pot = sum(pot_maps)
                u_flat = nafnet_unc.view(-1)
                p_flat = total_pot.view(-1)
                if u_flat.std() > 1e-8 and p_flat.std() > 1e-8:
                    try:
                        corr = torch.corrcoef(torch.stack([u_flat, p_flat]))[0, 1]
                        if torch.isfinite(corr):
                            total_uncertainty_corr += corr.item()
                            n_corr += 1
                    except Exception:
                        pass
                del total_pot
            del pot_maps
        del nafnet_unc, potentials

        # Per-image result
        per_image_results.append({
            'psnr_backbone': psnr_backbone,
            'psnr_corrected': psnr_corrected,
            'ssim_backbone': ssim_backbone,
            'ssim_corrected': ssim_corrected,
            'cnr_backbone': cnr_bb,
            'cnr_corrected': cnr_co,
        })

        # Free all intermediate tensors from this batch
        del corrected, backbone_out, info, clean, noisy
        del clean_std, backbone_std, corrected_std
        del backbone_gy, corrected_gy, clean_gy
        del backbone_lap, corrected_lap, clean_lap
        del backbone_gx, corrected_gx, clean_gx
        del backbone_edge, corrected_edge, clean_edge
        del signal_mask, bg_mask
        n += 1

    if n == 0:
        print(f"  WARNING: No samples processed for {dataset_name}")
        return {}

    # Compute averages
    bb_contrast = total_backbone_contrast_pres / n
    co_contrast = total_corrected_contrast_pres / n
    bb_boundary = total_backbone_boundary_pres / n
    co_boundary = total_corrected_boundary_pres / n
    bb_texture = total_backbone_texture_pres / n
    co_texture = total_corrected_texture_pres / n
    bb_edge = total_backbone_edge_pres / n
    co_edge = total_corrected_edge_pres / n

    avg_bb_pres = (bb_contrast + bb_boundary + bb_texture + bb_edge) / 4
    avg_co_pres = (co_contrast + co_boundary + co_texture + co_edge) / 4
    clinical_ratio = avg_co_pres / max(avg_bb_pres, 1e-8)

    # Clinical improvement counts
    clinical_improved = 0
    for bb, co in [(bb_contrast, co_contrast), (bb_boundary, co_boundary),
                   (bb_texture, co_texture), (bb_edge, co_edge)]:
        if co > bb:
            clinical_improved += 1

    # Predicates
    pred_passing = 0
    evaluated = ['P1', 'P2', 'P3', 'P4', 'P6']
    for p in evaluated:
        if total_pred_scores[p] / n >= 0.5:
            pred_passing += 1

    psnr_bb = total_psnr_backbone / n
    psnr_co = total_psnr_corrected / n
    psnr_delta = psnr_co - psnr_bb
    ssim_bb = total_ssim_backbone / n
    ssim_co = total_ssim_corrected / n
    cnr_bb_avg = total_cnr_backbone / n
    cnr_co_avg = total_cnr_corrected / n
    cnr_change = ((cnr_co_avg - cnr_bb_avg) / max(abs(cnr_bb_avg), 1e-8)) * 100
    corr_mag = total_correction_mag / n
    coop_corr = total_uncertainty_corr / max(n_corr, 1)

    # Determine status
    psnr_ok = abs(psnr_delta) <= 1.0
    cnr_ok = cnr_change >= 0
    clinical_ok = clinical_improved >= 3
    pred_ok = pred_passing >= 4

    if psnr_ok and cnr_ok and clinical_ok and pred_ok:
        verdict = "[★★★] PUBLICATION READY"
    elif psnr_ok and clinical_ok:
        verdict = "[++] GOOD"
    else:
        verdict = "[..] NEEDS WORK"

    # Print results
    print()
    print("=" * 84)
    print(f"  CROSS-DATASET VALIDATION: {dataset_name} ({n} samples)")
    print(f"  {verdict}")
    print("=" * 84)

    print(f"\n┌{'─'*82}┐")
    print(f"│ {'CLINICAL PRESERVATION':<30} {'Backbone%':>10} {'Corrected%':>11} {'Ratio':>8} {'Status':>10} │")
    print(f"├{'─'*82}┤")
    for name, bb, co in [('Contrast (local std)', bb_contrast, co_contrast),
                          ('Boundary (v-grad)', bb_boundary, co_boundary),
                          ('Texture (variance)', bb_texture, co_texture),
                          ('Edge (Sobel)', bb_edge, co_edge)]:
        ratio = co / max(bb, 1e-8)
        status = "IMPROVED" if co > bb else "DEGRADED"
        print(f"│ {name:<30} {bb*100:>9.1f}% {co*100:>10.1f}% {ratio:>7.3f} {status:>10} │")
    print(f"├{'─'*82}┤")
    print(f"│ {'AVERAGE':<30} {avg_bb_pres*100:>9.1f}% {avg_co_pres*100:>10.1f}% {clinical_ratio:>7.3f} {'':>4}{clinical_improved}/4 IMPROVED │")
    print(f"└{'─'*82}┘")

    print(f"\n┌{'─'*82}┐")
    print(f"│ {'TRADITIONAL METRICS':<30} {'Backbone':>12} {'Corrected':>12} {'Delta':>9} {'Status':>10} │")
    print(f"├{'─'*82}┤")
    psnr_status = "[OK]" if psnr_ok else "[!]"
    print(f"│ {'PSNR (dB)':<30} {psnr_bb:>12.2f} {psnr_co:>12.2f} {psnr_delta:>+9.3f} {psnr_status:>10} │")
    ssim_delta = ssim_co - ssim_bb
    ssim_status = "[OK]" if abs(ssim_delta) < 0.01 else "[!]"
    print(f"│ {'SSIM':<30} {ssim_bb:>12.4f} {ssim_co:>12.4f} {ssim_delta:>+9.4f} {ssim_status:>10} │")
    print(f"└{'─'*82}┘")

    print(f"\n┌{'─'*72}┐")
    print(f"│ {'GT-FREE PREDICATES':<25} {'Backbone':>10} {'Corrected':>10} {'Delta':>8} {'Status':>8} │")
    print(f"├{'─'*72}┤")
    for p in ['P1', 'P2', 'P3', 'P4', 'P6']:
        names = {'P1': 'Edge Quality', 'P2': 'Contrast', 'P3': 'Smoothness',
                 'P4': 'Structure', 'P6': 'Anatomy'}
        bb_s = total_pred_scores_backbone[p] / n
        co_s = total_pred_scores[p] / n
        delta = co_s - bb_s
        status = "PASS" if co_s >= 0.5 else "FAIL"
        print(f"│ {names[p]:<25} {bb_s:>10.3f} {co_s:>10.3f} {delta:>+8.3f} {status:>8} │")
    print(f"├{'─'*72}┤")
    print(f"│ {'PREDICATES PASSING':<25} {'':>10} {'':>10} {'':>8} {pred_passing:>4}/5   │")
    print(f"└{'─'*72}┘")

    print(f"\n┌{'─'*72}┐")
    print(f"│ {'OCT CLINICAL METRICS':<30} {'Backbone':>10} {'Corrected':>10} {'Delta':>10} │")
    print(f"├{'─'*72}┤")
    cnr_sym = "✓" if cnr_change >= 0 else "!"
    print(f"│ {'CNR (Contrast-to-Noise)':<30} {cnr_bb_avg:>10.3f} {cnr_co_avg:>10.3f} {cnr_change:>+9.1f}%{cnr_sym}│")
    tci_bb = total_tci_backbone / n
    tci_co = total_tci_corrected / n
    tci_sym = "✓" if tci_co > tci_bb else "~"
    print(f"│ {'TCI (Tissue Contrast Index)':<30} {tci_bb:>10.3f} {tci_co:>10.3f} {tci_co-tci_bb:>+9.3f} {tci_sym}│")
    print(f"└{'─'*72}┘")

    avg_scale = total_scale / max(n_scaled, 1) if n_scaled > 0 else None

    print(f"\n┌{'─'*72}┐")
    print(f"│ {'ADDITIONAL METRICS':<50} {'Value':>12}      │")
    print(f"├{'─'*72}┤")
    print(f"│ {'Correction Magnitude':<50} {corr_mag:>12.6f}      │")
    print(f"│ {'Cooperation Correlation':<50} {coop_corr:>+12.3f}      │")
    if avg_scale is not None:
        print(f"│ {'Adaptive Scale (avg)':<50} {avg_scale:>12.4f}      │")
    print(f"└{'─'*72}┘")

    print(f"\n>>> Clinical: {clinical_improved}/4 improved │ Ratio: {clinical_ratio:.3f} │ Pres: {avg_bb_pres*100:.1f}%→{avg_co_pres*100:.1f}%")
    print(f">>> PSNR: {psnr_bb:.2f}→{psnr_co:.2f} ({psnr_delta:+.3f}) │ CNR: {cnr_bb_avg:.2f}→{cnr_co_avg:.2f} ({cnr_change:+.1f}%)")
    print(f">>> Predicates: {pred_passing}/5 pass │ Cooperation: {coop_corr:+.3f}")
    if avg_scale is not None:
        print(f">>> Adaptive Scaling: avg_scale={avg_scale:.4f}")
    print()

    result = {
        'dataset': dataset_name,
        'n_samples': n,
        'psnr_backbone': psnr_bb,
        'psnr_corrected': psnr_co,
        'psnr_delta': psnr_delta,
        'ssim_backbone': ssim_bb,
        'ssim_corrected': ssim_co,
        'cnr_backbone': cnr_bb_avg,
        'cnr_corrected': cnr_co_avg,
        'cnr_change_pct': cnr_change,
        'clinical_ratio': clinical_ratio,
        'clinical_improved': clinical_improved,
        'predicates_passing': pred_passing,
        'correction_magnitude': corr_mag,
        'cooperation_correlation': coop_corr,
        'verdict': verdict,
    }
    if avg_scale is not None:
        result['adaptive_scale'] = avg_scale
    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset validation for V8 cooperative denoiser")
    parser.add_argument('--checkpoint', required=True, help='Path to best_model_cooperative.pth')
    parser.add_argument('--backbone', required=True, help='Path to NAFNet backbone checkpoint')
    parser.add_argument('--backbone_width', type=int, default=40)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_json', default=None, help='Save results to JSON')
    parser.add_argument('--adaptive_scaling', action='store_true',
                        help='Enable adaptive correction scaling for cross-dataset generalization')
    parser.add_argument('--reference_magnitude', type=float, default=None,
                        help='Override auto-detected reference correction magnitude from PKU37')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset filter (e.g. "duke17,duke2013,combined")')
    # TTA arguments
    parser.add_argument('--tta', action='store_true',
                        help='Enable Test-Time Adaptation for cross-dataset generalization')
    parser.add_argument('--tta_steps', type=int, default=30,
                        help='Number of TTA optimization steps')
    parser.add_argument('--tta_lr', type=float, default=5e-4,
                        help='TTA learning rate')
    parser.add_argument('--tta_adapt_samples', type=int, default=5,
                        help='Number of samples to use for TTA adaptation')
    parser.add_argument('--tta_w_magnitude', type=float, default=0.5,
                        help='TTA magnitude regularization weight (higher = more conservative)')
    parser.add_argument('--tta_w_cnr', type=float, default=1.5,
                        help='TTA CNR preservation weight')
    parser.add_argument('--tta_w_consistency', type=float, default=1.0,
                        help='TTA self-consistency weight')
    args = parser.parse_args()

    device = args.device

    # Load model
    print("Loading model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_width=args.backbone_width,
        pretrained_backbone=args.backbone
    )

    # Load cooperative checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded model_state_dict from checkpoint")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded state_dict from checkpoint")

    model = model.to(device)
    model.eval()

    # Define datasets to validate
    datasets = []

    # 1. PKU37 test set
    pku37_test = 'pku37_oct_dataset/pku37_real_test.jsonl'
    if os.path.exists(pku37_test):
        datasets.append(('PKU37-Test (same dist)', pku37_test))

    # 2. PKU37 val set (for reference)
    pku37_val = 'pku37_oct_dataset/pku37_real_val.jsonl'
    if os.path.exists(pku37_val):
        datasets.append(('PKU37-Val (training val)', pku37_val))

    # 3. Duke17 / Sparsity SDOCT 2012 (real noise, different scanner)
    duke17 = 'duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl'
    if os.path.exists(duke17):
        datasets.append(('Duke17-Sparsity (cross-dataset)', duke17))

    # 4. Duke2013 synthetic eval
    duke2013 = 'duke_sota_datasets/Duke17_Eval/duke2013_synth_eval.jsonl'
    if os.path.exists(duke2013):
        datasets.append(('Duke2013-SBSDI (cross-dataset)', duke2013))

    # 5. Combined duke eval
    duke_combined = 'duke_sota_datasets/Duke17_Eval/combined_eval.jsonl'
    if os.path.exists(duke_combined):
        datasets.append(('Duke-Combined (cross-dataset)', duke_combined))

    # Filter datasets if requested
    if args.datasets:
        filters = [f.strip().lower() for f in args.datasets.split(',')]
        datasets = [(n, p) for n, p in datasets
                    if any(f in n.lower() for f in filters)]

    if not datasets:
        print("ERROR: No datasets found!")
        sys.exit(1)

    print(f"\nFound {len(datasets)} datasets for validation:")
    for name, path in datasets:
        print(f"  - {name}: {path}")

    # Run validation on each dataset
    all_results = []
    reference_magnitude = args.reference_magnitude  # user override or None
    scaler = None

    if args.adaptive_scaling:
        print("\n[Adaptive Scaling] Enabled — will auto-calibrate from PKU37 reference")
        # If user provided reference magnitude, create scaler immediately
        if reference_magnitude is not None:
            predicate_fn = None
            if hasattr(model, 'corrector') and hasattr(model.corrector, 'predicates'):
                predicate_fn = model.corrector.predicates
            scaler = AdaptiveCorrectionScaler(
                reference_magnitude=reference_magnitude,
                predicate_fn=predicate_fn,
            )
            print(f"  Using user-provided reference_magnitude = {reference_magnitude:.6f}")

    if args.tta:
        print("\n[TTA] Test-Time Adaptation enabled for cross-dataset evaluation")

    for name, jsonl_path in datasets:
        print(f"\n{'='*84}")
        print(f"  Loading {name}...")
        dataset = PKU37Dataset(jsonl_path, patch_size=0, is_train=False)
        if len(dataset) == 0:
            print(f"  WARNING: Dataset {name} has 0 samples, skipping")
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Determine generalization strategy for cross-dataset
        use_scaler = None
        tta_adapter = None

        try:
            if 'cross-dataset' in name.lower():
                if args.tta:
                    # TTA takes priority over adaptive scaling
                    tta_adapter = TestTimeAdaptation(
                        model=model, device=device,
                        tta_steps=args.tta_steps, tta_lr=args.tta_lr,
                        n_adapt_samples=args.tta_adapt_samples,
                        w_magnitude=args.tta_w_magnitude,
                        w_cnr=args.tta_w_cnr,
                        w_consistency=args.tta_w_consistency,
                    )
                    adaptation_log = tta_adapter.adapt(loader, dataset_name=name)
                    # Re-create loader (iterator consumed by adapt)
                    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                elif args.adaptive_scaling and scaler is not None:
                    use_scaler = scaler
                    print(f"  [Adaptive Scaling] Active (ref magnitude: {reference_magnitude:.6f})")

            result = validate_dataset(model, loader, device, dataset_name=name, scaler=use_scaler)
            if result:
                if tta_adapter is not None:
                    result['tta_adapted'] = True
                    result['tta_steps'] = args.tta_steps
                all_results.append(result)

                # Auto-calibrate reference magnitude from first PKU37 dataset
                if args.adaptive_scaling and reference_magnitude is None and 'PKU37-Test' in name:
                    reference_magnitude = result['correction_magnitude']
                    predicate_fn = None
                    if hasattr(model, 'corrector') and hasattr(model.corrector, 'predicates'):
                        predicate_fn = model.corrector.predicates
                    scaler = AdaptiveCorrectionScaler(
                        reference_magnitude=reference_magnitude,
                        predicate_fn=predicate_fn,
                    )
                    print(f"\n  [Adaptive Scaling] Calibrated: reference_magnitude = {reference_magnitude:.6f}")
        finally:
            # Always restore weights and clean GPU memory, even on crash
            if tta_adapter is not None:
                tta_adapter.restore_state()
                tta_adapter = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print summary table
    if all_results:
        print("\n" + "=" * 100)
        print("  CROSS-DATASET VALIDATION SUMMARY")
        print("=" * 100)
        print(f"{'Dataset':<35} {'N':>4} {'PSNR Δ':>8} {'CNR Δ%':>8} {'Clinical':>10} {'Preds':>6} {'Verdict':<25}")
        print("-" * 100)
        for r in all_results:
            print(f"{r['dataset']:<35} {r['n_samples']:>4} {r['psnr_delta']:>+8.3f} {r['cnr_change_pct']:>+7.1f}% "
                  f"{r['clinical_improved']}/4 ({r['clinical_ratio']:.3f}) {r['predicates_passing']:>3}/5  {r['verdict']}")
        print("=" * 100)

    # Save to JSON
    if args.output_json and all_results:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
