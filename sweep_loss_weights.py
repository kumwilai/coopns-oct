#!/usr/bin/env python3
"""
Optuna-based hyperparameter sweep for SimplifiedCooperativeLoss weights.

Uses Tree-structured Parzen Estimator (TPE) to find optimal loss weights
that maximize clinical improvement while maintaining PSNR/CNR constraints.

Each trial: 3 epochs on 200 training samples, validated on 50 samples.
Total: 20 trials.

Usage:
    python sweep_loss_weights.py [--n_trials 20] [--epochs 3] [--max_train 200]
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import optuna
from optuna.trial import TrialState
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import everything from the training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v8_cooperative import (
    PKU37Dataset,
    NeuroSymbolicDenoiserV8Cooperative,
    SimplifiedCooperativeLoss,
    train_epoch,
    validate,
    compute_psnr,
    compute_ssim,
)


def create_model_and_criterion(trial, device, pretrained_backbone):
    """Create model and criterion with Optuna-suggested hyperparameters."""

    # ===== Suggest hyperparameters =====
    # Term weights (how much each loss group matters)
    clinical_metric_weight = trial.suggest_float('clinical_metric_weight', 1.0, 8.0, step=0.5)
    predicate_weight = trial.suggest_float('predicate_weight', 2.0, 12.0, step=1.0)
    cooperation_weight = trial.suggest_float('cooperation_weight', 0.05, 0.5, step=0.05)

    # Clinical metric targets (how aggressively to push improvement)
    contrast_target_mult = trial.suggest_float('contrast_target_mult', 1.02, 1.20, step=0.02)
    boundary_target_mult = trial.suggest_float('boundary_target_mult', 1.02, 1.20, step=0.02)

    # Predicate loss dynamics
    regression_penalty = trial.suggest_float('regression_penalty', 2.0, 10.0, step=1.0)
    improvement_bonus = trial.suggest_float('improvement_bonus', 0.5, 4.0, step=0.5)

    # ===== Create model =====
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet',
        pretrained_backbone=pretrained_backbone,
    ).to(device)

    # Freeze backbone
    for param in model.backbone.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.uncertainty_parameters():
        param.requires_grad = True

    # ===== Create criterion with suggested weights =====
    criterion = SimplifiedCooperativeLoss(
        predicates=model.corrector.predicates,
        cooperation_weight=cooperation_weight,
    ).to(device)

    # Override term weights
    criterion.clinical_metric_weight = clinical_metric_weight
    criterion.predicate_weight = predicate_weight
    criterion.cooperation_weight = cooperation_weight

    # Store sweep params for monkey-patching in forward
    criterion._sweep_contrast_target_mult = contrast_target_mult
    criterion._sweep_boundary_target_mult = boundary_target_mult
    criterion._sweep_regression_penalty = regression_penalty
    criterion._sweep_improvement_bonus = improvement_bonus

    # Monkey-patch compute_clinical_metric_loss to use swept targets
    original_clinical = criterion.compute_clinical_metric_loss

    def patched_clinical_metric_loss(corrected, backbone_out, clean):
        loss, metrics = original_clinical(corrected, backbone_out, clean)
        return loss, metrics

    # Monkey-patch compute_predicate_improvement_loss to use swept penalties
    original_predicate = criterion.compute_predicate_improvement_loss

    def patched_predicate_loss(corrected, backbone_out, noisy, clean):
        # Temporarily override the penalty/bonus values
        loss, metrics = original_predicate(corrected, backbone_out, noisy, clean)
        return loss, metrics

    return model, criterion, contrast_target_mult, boundary_target_mult, regression_penalty, improvement_bonus


def apply_sweep_overrides(criterion, contrast_mult, boundary_mult, reg_penalty, imp_bonus):
    """
    Apply sweep parameter overrides to the criterion.
    Must be called before each forward pass since we can't easily monkey-patch
    the inner computation. Instead, we directly modify the target multipliers
    and penalty values in the loss object.
    """
    # Override contrast/boundary target multipliers in compute_clinical_metric_loss
    # We do this by storing them as attributes and patching the method
    criterion._contrast_target_mult = contrast_mult
    criterion._boundary_target_mult = boundary_mult
    criterion._regression_penalty_val = reg_penalty
    criterion._improvement_bonus_val = imp_bonus


def patch_criterion_for_sweep(criterion):
    """
    Patch the criterion methods to use sweep-specific values.
    """
    # Save original methods
    _orig_clinical = criterion.compute_clinical_metric_loss.__func__
    _orig_predicate = criterion.compute_predicate_improvement_loss.__func__

    def compute_clinical_metric_loss_patched(self, corrected, backbone_out, clean):
        """Patched version that uses sweep target multipliers."""
        eps = 1e-8
        sx = self.sobel_x.to(dtype=corrected.dtype)
        sy = self.sobel_y.to(dtype=corrected.dtype)
        lap = self.laplacian.to(dtype=corrected.dtype)

        # Contrast ratio
        corrected_std = self.local_std(corrected)
        clean_std = self.local_std(clean)
        with torch.no_grad():
            backbone_std = self.local_std(backbone_out)
            clean_std_mean = clean_std.mean().clamp(min=eps)
            backbone_contrast_ratio = backbone_std.mean() / clean_std_mean
        corrected_contrast_ratio = corrected_std.mean() / clean_std_mean
        contrast_target = backbone_contrast_ratio * getattr(self, '_contrast_target_mult', 1.10)

        # Boundary ratio
        corrected_vgrad = F.conv2d(corrected, sy, padding=1).abs()
        clean_vgrad = F.conv2d(clean, sy, padding=1).abs()
        with torch.no_grad():
            backbone_vgrad = F.conv2d(backbone_out, sy, padding=1).abs()
            clean_vgrad_mean = clean_vgrad.mean().clamp(min=eps)
            backbone_boundary_ratio = backbone_vgrad.mean() / clean_vgrad_mean
        corrected_boundary_ratio = corrected_vgrad.mean() / clean_vgrad_mean
        boundary_target = backbone_boundary_ratio * getattr(self, '_boundary_target_mult', 1.10)

        # Texture ratio (keep at 1.0 — don't regress)
        corrected_tex = F.conv2d(corrected, lap, padding=1).abs()
        clean_tex = F.conv2d(clean, lap, padding=1).abs()
        with torch.no_grad():
            backbone_tex = F.conv2d(backbone_out, lap, padding=1).abs()
            clean_tex_mean = clean_tex.mean().clamp(min=eps)
            backbone_texture_ratio = backbone_tex.mean() / clean_tex_mean
        corrected_texture_ratio = corrected_tex.mean() / clean_tex_mean
        texture_target = backbone_texture_ratio * 1.0

        # Edge (EPI)
        corrected_edge = torch.sqrt(F.conv2d(corrected, sx, padding=1)**2 +
                                     F.conv2d(corrected, sy, padding=1)**2 + eps)
        clean_edge = torch.sqrt(F.conv2d(clean, sx, padding=1)**2 +
                                 F.conv2d(clean, sy, padding=1)**2 + eps)
        with torch.no_grad():
            backbone_edge = torch.sqrt(F.conv2d(backbone_out, sx, padding=1)**2 +
                                        F.conv2d(backbone_out, sy, padding=1)**2 + eps)

        def _pearson_flat(a, b):
            a_f = a.reshape(a.shape[0], -1)
            b_f = b.reshape(b.shape[0], -1)
            a_c = a_f - a_f.mean(dim=1, keepdim=True)
            b_c = b_f - b_f.mean(dim=1, keepdim=True)
            a_n = a_c / (a_c.norm(dim=1, keepdim=True) + 1e-8)
            b_n = b_c / (b_c.norm(dim=1, keepdim=True) + 1e-8)
            return (a_n * b_n).sum(dim=1).mean()

        corrected_epi = _pearson_flat(clean_edge, corrected_edge)
        with torch.no_grad():
            backbone_epi = _pearson_flat(clean_edge, backbone_edge)
        epi_target = backbone_epi * 1.02

        # CNR
        with torch.no_grad():
            signal_mask = (clean > clean.mean()).float()
            bg_mask = 1.0 - signal_mask
            sig_sum = signal_mask.sum().clamp(min=1.0)
            bg_sum = bg_mask.sum().clamp(min=1.0)
            bb_sig = (backbone_out * signal_mask).sum() / sig_sum
            bb_bg = (backbone_out * bg_mask).sum() / bg_sum
            bb_bg_std = torch.sqrt(((backbone_out - bb_bg)**2 * bg_mask).sum() / bg_sum + eps).clamp(min=1e-4)
            backbone_cnr = (bb_sig - bb_bg) / bb_bg_std

        cr_sig = (corrected * signal_mask).sum() / sig_sum
        cr_bg = (corrected * bg_mask).sum() / bg_sum
        cr_bg_std = torch.sqrt(((corrected - cr_bg)**2 * bg_mask).sum() / bg_sum + eps).clamp(min=1e-4)
        corrected_cnr = (cr_sig - cr_bg) / cr_bg_std
        cnr_target = backbone_cnr * 1.0

        # Asymmetric loss
        sub_weights = {'contrast': 2.0, 'boundary': 1.5, 'texture': 1.0, 'epi': 1.5, 'cnr': 1.5}
        ratios = {
            'contrast': corrected_contrast_ratio, 'boundary': corrected_boundary_ratio,
            'texture': corrected_texture_ratio, 'epi': corrected_epi, 'cnr': corrected_cnr,
        }
        targets = {
            'contrast': contrast_target, 'boundary': boundary_target,
            'texture': texture_target, 'epi': epi_target, 'cnr': cnr_target,
        }

        total_loss = corrected.new_zeros(())
        total_w = 0.0
        metrics = {}
        for name in ['contrast', 'boundary', 'texture', 'epi', 'cnr']:
            w = sub_weights[name]
            deficit = F.relu(targets[name].detach() - ratios[name])
            loss_term = 3.0 * deficit + deficit ** 2
            total_loss = total_loss + w * loss_term
            total_w += w
            with torch.no_grad():
                metrics[f'cm_{name}_ratio'] = ratios[name].item() if isinstance(ratios[name], torch.Tensor) else ratios[name]
                metrics[f'cm_{name}_target'] = targets[name].item() if isinstance(targets[name], torch.Tensor) else targets[name]

        total_loss = total_loss / max(total_w, 1e-6)
        total_loss = total_loss.clamp(max=5.0)
        return total_loss, metrics

    def compute_predicate_improvement_loss_patched(self, corrected, backbone_out, noisy, clean):
        """Patched version that uses sweep regression/improvement values."""
        if self.predicates is None:
            return corrected.new_zeros(()), {'pred_warning': 'no_predicates'}

        corrected_eval = self.predicates.evaluate_clinical_with_grad(corrected, noisy, clean=clean)
        with torch.no_grad():
            backbone_eval = self.predicates.evaluate_clinical_degradation(backbone_out, noisy, clean=clean)

        pred_weights = {'P_contrast': 3.0, 'P_edge': 2.0, 'P_boundary': 2.0, 'P_texture': 1.0}
        reg_pen = getattr(self, '_regression_penalty_val', 5.0)
        imp_bon = getattr(self, '_improvement_bonus_val', 1.5)

        total_loss = corrected.new_zeros(())
        total_w = 0.0
        metrics = {}

        for pred_name, weight in pred_weights.items():
            c_result = corrected_eval.get(pred_name, {})
            b_result = backbone_eval.get(pred_name, {})
            c_score = c_result.get('score', corrected.new_tensor(0.5))
            b_score = b_result.get('score', corrected.new_tensor(0.5))
            if not isinstance(c_score, torch.Tensor):
                c_score = corrected.new_tensor(c_score)
            if not isinstance(b_score, torch.Tensor):
                b_score = corrected.new_tensor(b_score)
            b_score = b_score.detach()

            delta = c_score - b_score
            regression_penalty = F.relu(-delta) * reg_pen
            improvement_bonus = -F.relu(delta) * imp_bon
            threshold_penalty = F.relu(0.5 - c_score) * 2.0

            pred_loss = weight * (regression_penalty + improvement_bonus + threshold_penalty)
            total_loss = total_loss + pred_loss
            total_w += weight

            short = pred_name.split('_')[1]
            with torch.no_grad():
                metrics[f'pi_{short}_delta'] = delta.item()

        total_loss = total_loss / max(total_w, 1e-6)
        total_loss = total_loss.clamp(max=10.0)
        return total_loss, metrics

    # Bind patched methods
    import types
    criterion.compute_clinical_metric_loss = types.MethodType(compute_clinical_metric_loss_patched, criterion)
    criterion.compute_predicate_improvement_loss = types.MethodType(compute_predicate_improvement_loss_patched, criterion)


def compute_objective_score(val_metrics):
    """
    Composite objective score from validation metrics.
    Maximizes clinical improvement, penalizes PSNR drop.
    """
    contrast_ratio = val_metrics.get('contrast_ratio', 1.0)
    boundary_ratio = val_metrics.get('boundary_ratio', 1.0)
    texture_ratio = val_metrics.get('texture_ratio', 1.0)
    edge_ratio = val_metrics.get('edge_ratio', 1.0)
    psnr_delta = val_metrics.get('psnr_delta', 0)
    cnr_improvement = val_metrics.get('cnr_improvement', 0)

    # Clinical improvement score (each % above 1.0 = 1 point)
    contrast_pct = max(0, (contrast_ratio - 1.0) * 100)
    boundary_pct = max(0, (boundary_ratio - 1.0) * 100)
    texture_pct = max(0, (texture_ratio - 1.0) * 100)
    edge_pct = max(0, (edge_ratio - 1.0) * 100)
    clinical_score = contrast_pct + boundary_pct + texture_pct + edge_pct

    # Count how many metrics improved
    n_improved = sum([
        1 if contrast_ratio > 1.0 else 0,
        1 if boundary_ratio > 1.0 else 0,
        1 if texture_ratio > 1.0 else 0,
        1 if edge_ratio > 1.0 else 0,
    ])

    # Bonus for all 4 improving
    clinical_score += n_improved * 5.0

    # CNR bonus/penalty
    if cnr_improvement > 0:
        clinical_score += min(cnr_improvement * 0.5, 5.0)
    else:
        clinical_score += max(cnr_improvement * 1.0, -10.0)

    # PSNR penalty (only beyond 0.5 dB drop)
    if psnr_delta < -1.0:
        clinical_score -= (abs(psnr_delta) - 0.5) * 30.0  # Hard penalty
    elif psnr_delta < -0.5:
        clinical_score -= (abs(psnr_delta) - 0.5) * 10.0  # Mild penalty

    return clinical_score


def objective(trial, args):
    """Optuna objective function: train short probe and return composite score."""
    device = args.device

    # Create model and criterion with suggested hyperparameters
    model, criterion, c_mult, b_mult, reg_pen, imp_bon = create_model_and_criterion(
        trial, device, args.pretrained_backbone
    )

    # Apply sweep overrides and patch methods
    apply_sweep_overrides(criterion, c_mult, b_mult, reg_pen, imp_bon)
    patch_criterion_for_sweep(criterion)

    # Data loaders
    train_dataset = PKU37Dataset(
        args.train_jsonl, max_samples=args.max_train,
        patch_size=96, is_train=True
    )
    val_dataset = PKU37Dataset(
        args.val_jsonl, max_samples=args.max_val,
        patch_size=0, is_train=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Optimizer (same as main training but without log_sigma)
    corrector_params = list(model.corrector.correctors.parameters())
    potential_params = []
    if hasattr(model.corrector, 'potential_estimators'):
        potential_params = list(model.corrector.potential_estimators.parameters())
    if hasattr(model.corrector, 'lambda_predictor'):
        potential_params += list(model.corrector.lambda_predictor.parameters())

    negotiator_params = []
    if hasattr(model.corrector, 'negotiator'):
        negotiator_params += list(model.corrector.negotiator.parameters())
    if hasattr(model.corrector, 'router'):
        negotiator_params += list(model.corrector.router.parameters())

    clinical_enhancement_params = []
    for attr in ['cnr_preserver', 'clinical_enhancer', 'region_aware_corrector', 'confidence_estimator']:
        if hasattr(model.corrector, attr):
            clinical_enhancement_params += list(getattr(model.corrector, attr).parameters())

    uncertainty_params = list(model.backbone.uncertainty_parameters())

    param_groups = [
        {'params': corrector_params, 'lr': 2e-4},
        {'params': potential_params, 'lr': 5e-4} if potential_params else {'params': [], 'lr': 0},
        {'params': negotiator_params, 'lr': 3e-4} if negotiator_params else {'params': [], 'lr': 0},
        {'params': clinical_enhancement_params, 'lr': 2e-4} if clinical_enhancement_params else {'params': [], 'lr': 0},
        {'params': uncertainty_params, 'lr': 5e-4},
    ]
    param_groups = [pg for pg in param_groups if len(list(pg['params'])) > 0]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=2e-6
    )

    # Training loop
    best_score = -float('inf')
    for epoch in range(1, args.epochs + 1):
        # Stage switching (same logic as main training)
        stage_switch = args.epochs // 2 + 1
        if epoch < stage_switch:
            criterion.training_stage = 1
        else:
            criterion.training_stage = 2

        model.train()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler=None
        )
        scheduler.step()

        # Validate
        gc.collect()
        val_metrics = validate(model, val_loader, device, lpips_model=None, criterion=criterion)

        score = compute_objective_score(val_metrics)

        # Report intermediate value for pruning
        trial.report(score, epoch)
        if trial.should_prune():
            del model, criterion, optimizer, scheduler
            del train_loader, val_loader, train_dataset, val_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        if score > best_score:
            best_score = score
            best_metrics = val_metrics.copy()

        # Log
        contrast_r = val_metrics.get('contrast_ratio', 1.0)
        boundary_r = val_metrics.get('boundary_ratio', 1.0)
        texture_r = val_metrics.get('texture_ratio', 1.0)
        edge_r = val_metrics.get('edge_ratio', 1.0)
        psnr_d = val_metrics.get('psnr_delta', 0)
        cnr_i = val_metrics.get('cnr_improvement', 0)

        print(f"  Trial {trial.number} Ep{epoch}: score={score:.1f} | "
              f"C={contrast_r:.3f} B={boundary_r:.3f} T={texture_r:.3f} E={edge_r:.3f} | "
              f"PSNR={psnr_d:+.3f} CNR={cnr_i:+.1f}%")

    # Store best metrics
    trial.set_user_attr('contrast_ratio', best_metrics.get('contrast_ratio', 1.0))
    trial.set_user_attr('boundary_ratio', best_metrics.get('boundary_ratio', 1.0))
    trial.set_user_attr('texture_ratio', best_metrics.get('texture_ratio', 1.0))
    trial.set_user_attr('edge_ratio', best_metrics.get('edge_ratio', 1.0))
    trial.set_user_attr('psnr_delta', best_metrics.get('psnr_delta', 0))
    trial.set_user_attr('cnr_improvement', best_metrics.get('cnr_improvement', 0))

    # Cleanup
    del model, criterion, optimizer, scheduler
    del train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_score


def print_results(study):
    """Print comprehensive results from the Optuna study."""
    print("\n" + "=" * 90)
    print("SWEEP RESULTS")
    print("=" * 90)

    best = study.best_trial
    print(f"\nBest Trial: #{best.number}")
    print(f"Best Score: {best.value:.2f}")
    print(f"\nOptimal Hyperparameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    print(f"\nMetrics at Best Trial:")
    for key in ['contrast_ratio', 'boundary_ratio', 'texture_ratio', 'edge_ratio', 'psnr_delta', 'cnr_improvement']:
        val = best.user_attrs.get(key, 'N/A')
        if isinstance(val, float):
            print(f"  {key}: {val:+.4f}")
        else:
            print(f"  {key}: {val}")

    # Top 5 trials
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    print(f"\nTop 5 Trials (out of {len(completed)} completed):")
    print(f"{'#':>4} {'Score':>8} {'Contr':>7} {'Bound':>7} {'Text':>7} {'Edge':>7} {'PSNR':>7} {'CNR%':>7}")
    print("-" * 60)
    for t in completed[:5]:
        print(f"{t.number:>4} {t.value:>8.1f} "
              f"{t.user_attrs.get('contrast_ratio', 0):>7.3f} "
              f"{t.user_attrs.get('boundary_ratio', 0):>7.3f} "
              f"{t.user_attrs.get('texture_ratio', 0):>7.3f} "
              f"{t.user_attrs.get('edge_ratio', 0):>7.3f} "
              f"{t.user_attrs.get('psnr_delta', 0):>+7.3f} "
              f"{t.user_attrs.get('cnr_improvement', 0):>+7.1f}")

    # Parameter importance
    print("\nParameter Importance:")
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in importances.items():
            bar = "#" * int(importance * 40)
            print(f"  {param:>25}: {importance:.3f} {bar}")
    except Exception as e:
        print(f"  (Could not compute: {e})")

    # Generate recommended config
    print("\n" + "=" * 90)
    print("RECOMMENDED CONFIGURATION:")
    print("=" * 90)
    p = best.params
    print(f"""
# Best sweep config (Trial #{best.number}, Score: {best.value:.1f})
# In SimplifiedCooperativeLoss.__init__():
self.clinical_metric_weight = {p.get('clinical_metric_weight', 3.0)}
self.predicate_weight = {p.get('predicate_weight', 5.0)}
self.cooperation_weight = {p.get('cooperation_weight', 0.1)}

# In compute_clinical_metric_loss():
contrast_target = backbone_contrast_ratio * {p.get('contrast_target_mult', 1.10)}
boundary_target = backbone_boundary_ratio * {p.get('boundary_target_mult', 1.10)}

# In compute_predicate_improvement_loss():
regression_penalty = F.relu(-delta) * {p.get('regression_penalty', 5.0)}
improvement_bonus = -F.relu(delta) * {p.get('improvement_bonus', 1.5)}
""")

    return best.params


def main():
    parser = argparse.ArgumentParser(description='Sweep SimplifiedCooperativeLoss Weights')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per trial')
    parser.add_argument('--max_train', type=int, default=200, help='Training samples per trial')
    parser.add_argument('--max_val', type=int, default=50, help='Validation samples per trial')
    parser.add_argument('--train_jsonl', default='pku37_oct_dataset/pku37_real_train.jsonl')
    parser.add_argument('--val_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl')
    parser.add_argument('--pretrained_backbone', default='outputs/nafnet_pku37_w40/best_model.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='outputs/sweep_simplified_loss')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("SIMPLIFIED LOSS WEIGHT SWEEP")
    print("=" * 80)
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs per trial: {args.epochs}")
    print(f"  Train samples: {args.max_train}")
    print(f"  Val samples: {args.max_val}")
    print(f"  Device: {args.device}")
    print()
    print("Search Space:")
    print("  clinical_metric_weight:  [1.0,  8.0]  (Term 1 — direct clinical metrics)")
    print("  predicate_weight:        [2.0, 12.0]  (Term 2 — neuro-symbolic predicates)")
    print("  cooperation_weight:      [0.05, 0.5]  (Term 3 — backbone-corrector cooperation)")
    print("  contrast_target_mult:    [1.02, 1.20] (Target: X× backbone contrast)")
    print("  boundary_target_mult:    [1.02, 1.20] (Target: X× backbone boundary)")
    print("  regression_penalty:      [2.0, 10.0]  (Don't-regress penalty strength)")
    print("  improvement_bonus:       [0.5,  4.0]  (Improvement reward strength)")
    print()

    # Create study
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=8,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=6,
        n_warmup_steps=1,
    )

    study = optuna.create_study(
        study_name='simplified_loss_sweep',
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=f'sqlite:///{args.output_dir}/optuna_study.db',
        load_if_exists=True,
    )

    # Run optimization
    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )
    elapsed = time.time() - start_time

    # Print results
    best_params = print_results(study)

    # Save results
    results = {
        'best_params': best_params,
        'best_score': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'elapsed_seconds': elapsed,
        'best_metrics': dict(study.best_trial.user_attrs),
    }
    results_path = os.path.join(args.output_dir, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
