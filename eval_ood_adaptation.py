import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import adaptive_oct_denoise as aod


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _write_json(path: str, obj: dict) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _append_jsonl(path: str, obj: dict) -> None:
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _summary_stats(values):
    if not values:
        return {"mean": None, "std": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60.0
    return f"{hours:.1f}h"


def _compute_casa_map_metrics(coh_map: torch.Tensor, incoh_map: torch.Tensor, noisy: torch.Tensor, pred: torch.Tensor):
    # coh_map/incoh_map: [B,1,H,W], noisy/pred: [B,1,H,W]
    with torch.no_grad():
        entropy = float(aod.casa_entropy_regularizer(coh_map, incoh_map).item())
        coh_mean = float(coh_map.mean().item())
        coh_std = float(coh_map.std().item())

        residual = (noisy - pred).abs()
        cv_map = aod.compute_local_cv(residual)
        cv_norm = (cv_map - cv_map.min()) / (cv_map.max() - cv_map.min() + 1e-8)
        corr = float((1.0 - aod.pearson_correlation_loss(coh_map, cv_norm)).item())

    return {
        "coh_mean": coh_mean,
        "coh_std": coh_std,
        "entropy": entropy,
        "corr_residual_cv": corr,
    }


def _compute_moe_metrics(aux: dict):
    weights = aux.get("moe_weights") if isinstance(aux, dict) else None
    entropy_adapter = aux.get("moe_entropy") if isinstance(aux, dict) else None
    if weights is None:
        return None
    # weights: [B,K]
    with torch.no_grad():
        w = weights.detach().float()
        top1 = float(w.max(dim=1).values.mean().item())
        ent = float((-(w * (w.clamp_min(1e-12).log())).sum(dim=1)).mean().item())
        ent2 = float(entropy_adapter.item()) if entropy_adapter is not None and hasattr(entropy_adapter, "item") else None
    return {"gate_top1_mean": top1, "gate_entropy_mean": ent, "gate_entropy_adapter": ent2}


def _accum_siminv_theta_stats(acc: dict, aux: dict):
    theta = aux.get("theta_hat") if isinstance(aux, dict) else None
    if theta is None:
        return
    with torch.no_grad():
        t = theta.detach().float()
        if acc.get("theta_names") is None:
            acc["theta_names"] = aux.get("theta_names")
        if acc.get("_sum") is None:
            acc["_sum"] = t.sum(dim=0)
            acc["_sumsq"] = (t * t).sum(dim=0)
            acc["_n"] = int(t.shape[0])
        else:
            acc["_sum"] = acc["_sum"] + t.sum(dim=0)
            acc["_sumsq"] = acc["_sumsq"] + (t * t).sum(dim=0)
            acc["_n"] = int(acc["_n"] + int(t.shape[0]))


def _finalize_siminv_theta_stats(acc: dict):
    n = int(acc.get("_n") or 0)
    if n <= 0 or acc.get("_sum") is None:
        return {"theta_names": acc.get("theta_names"), "theta_mean": None, "theta_std": None}
    s = acc["_sum"]
    ss = acc["_sumsq"]
    mean = (s / n).cpu().numpy().tolist()
    var = (ss / n - (s / n) ** 2).clamp(min=0.0)
    std = var.sqrt().cpu().numpy().tolist()
    return {"theta_names": acc.get("theta_names"), "theta_mean": mean, "theta_std": std}

def _estimate_sigma_log_domain(noisy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Robust per-image sigma estimate in log domain using a high-pass residual.

    Args:
        noisy: [B,1,H,W] in [0,1]
    Returns:
        sigma: [B,1,1,1] (float)
    """
    z = torch.log(noisy.clamp(0.0, 1.0) + eps)
    z_pad = F.pad(z, (2, 2, 2, 2), mode="replicate")
    z_blur = F.avg_pool2d(z_pad, kernel_size=5, stride=1, padding=0)
    hp = (z - z_blur).abs().flatten(1)
    med = hp.median(dim=1).values.view(-1, 1, 1, 1)
    return (med / 0.6745).clamp(min=1e-6)


def _sure_log_domain_loss(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    sigma: torch.Tensor,
    eps_img: float,
    eps_fd: float,
    hutchinson_samples: int,
    return_aux: bool,
):
    """Monte-Carlo SURE loss in log domain.

    Assumes z = log(y + eps_img) = x_log + n, n ~ N(0, sigma^2).
    Uses Hutchinson estimator for divergence of h(z) = log(f(y) + eps_img) wrt z,
    where y = exp(z) - eps_img.
    """
    z = torch.log(noisy.clamp(0.0, 1.0) + eps_img)
    if return_aux:
        out = model(noisy, return_aux=True)
        pred, _aux = out if isinstance(out, tuple) else (out, {})
    else:
        pred = model(noisy)
    h = torch.log(pred.clamp(0.0, 1.0) + eps_img)

    data_term = torch.mean((h - z) ** 2)

    div_est_total = h.new_zeros(())
    for _ in range(max(1, int(hutchinson_samples))):
        v = torch.randn_like(z)
        z_pert = z + float(eps_fd) * v
        noisy_pert = (torch.exp(z_pert) - eps_img).clamp(0.0, 1.0)
        if return_aux:
            out_p = model(noisy_pert, return_aux=True)
            pred_p, _aux_p = out_p if isinstance(out_p, tuple) else (out_p, {})
        else:
            pred_p = model(noisy_pert)
        h_p = torch.log(pred_p.clamp(0.0, 1.0) + eps_img)
        div_est_total = div_est_total + torch.mean(v * (h_p - h)) / float(eps_fd)

    div_est = div_est_total / float(max(1, int(hutchinson_samples)))
    # Drop constants (-sigma^2) since they do not affect optimization.
    sure = data_term + 2.0 * torch.mean((sigma ** 2)) * div_est
    return sure, pred, {"sure_data": float(data_term.detach().item()), "sure_div": float(div_est.detach().item())}


def _tta_once(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    scope: str,
    objective: str,
    reset_state: bool,
    num_steps: int,
    lr: float,
    tv_weight: float,
    self_consistency: float,
    anchor_weight: float,
    anchor_target: str,
    n2v_mask_ratio: float,
    n2v_box_size: int,
    n2v_blindspot_dilation: int,
    b2u_mask_ratio: float,
    b2u_block_size: int,
    hybrid_corr_threshold: float,
    hybrid_corr_low: float | None,
    hybrid_corr_high: float | None,
    hybrid_corr_source: str,
    speckle_weight: float,
    banding_weight: float,
    coherence_weight: float,
    casa_entropy_weight: float,
    casa_std_weight: float,
    casa_min_std: float,
    spectral_weight: float,
    augmentations,
    noise_characterizer,
    return_aux: bool,
    sure_eps: float = 1e-3,
    sure_sigma: float = 0.0,
    sure_hutchinson_samples: int = 1,
):
    if num_steps <= 0:
        out = model(noisy, return_aux=True) if return_aux else model(noisy)
        return out

    # Configure trainable params.
    if scope == "adapter":
        aod.freeze_backbone_unfreeze_adapter(model)
    elif scope == "full":
        aod.unfreeze_all(model)
    else:
        raise ValueError("--tta_scope must be 'adapter' or 'full'")

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for TTA (check requires_grad flags).")

    # Cache state to restore after per-image adaptation (avoid contamination).
    original_state = None
    if reset_state:
        original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Anchor target: either the noisy input (stability) or the base prediction (less likely to drift toward noise).
    if anchor_weight > 0:
        if anchor_target not in ("noisy", "base"):
            raise ValueError("--anchor_target must be 'noisy' or 'base'")
        if anchor_target == "noisy":
            anchor_ref = noisy.detach()
        else:
            model.eval()
            with torch.no_grad():
                base_out = model(noisy, return_aux=True) if return_aux else model(noisy)
                base_pred = base_out[0] if (return_aux and isinstance(base_out, tuple)) else base_out
                anchor_ref = base_pred.detach()
            model.train()
    else:
        anchor_ref = None

    model.train()
    opt = torch.optim.Adam(params, lr=lr)
    objective = str(objective).lower().strip()
    if objective not in ("consistency", "n2v", "b2u", "hybrid", "sure"):
        raise ValueError("--tta_objective must be one of: consistency, n2v, b2u, hybrid, sure")

    aug_fns = aod._build_tta_augs(augmentations) if objective == "consistency" else [(lambda x: x, lambda x: x)]
    n2v_loss_fn = aod.Noise2VoidLoss() if objective in ("n2v", "hybrid") else None
    b2u_masker = (
        aod.GlobalAwareMaskMapper(mask_ratio=b2u_mask_ratio, block_size=b2u_block_size)
        if objective in ("b2u", "hybrid")
        else None
    )
    b2u_loss_fn = aod.RevisibleLoss(lambda_tv=0.0, lambda_anchor=0.0) if objective in ("b2u", "hybrid") else None
    speckle_loss_fn = aod.SpeckleStatisticsLoss() if speckle_weight > 0 else None

    corr_score = None
    if objective == "hybrid":
        corr_source = str(hybrid_corr_source).lower().strip()
        if corr_source not in ("noisy", "residual_base"):
            raise ValueError("--hybrid_corr_source must be 'noisy' or 'residual_base'")
        if corr_source == "noisy":
            corr_input = noisy
        else:
            if anchor_ref is not None and anchor_target == "base":
                corr_input = noisy - anchor_ref
            else:
                model.eval()
                with torch.no_grad():
                    base_out = model(noisy, return_aux=True) if return_aux else model(noisy)
                    base_pred = base_out[0] if (return_aux and isinstance(base_out, tuple)) else base_out
                model.train()
                corr_input = noisy - base_pred.detach()
        corr_score = float(aod.estimate_noise_correlation_score(corr_input).mean().item())
        low_th = hybrid_corr_low if hybrid_corr_low is not None else 0.9 * hybrid_corr_threshold
        high_th = hybrid_corr_high if hybrid_corr_high is not None else 1.1 * hybrid_corr_threshold
        if low_th > high_th:
            raise ValueError("--hybrid_corr_low must be <= --hybrid_corr_high")

    if spectral_weight > 0 and noise_characterizer is None:
        noise_characterizer = aod.SpectralNoiseCharacterizer().to(noisy.device)
    if noise_characterizer is not None:
        noise_characterizer.eval()
        with torch.no_grad():
            spectral_target = noise_characterizer(noisy.clamp(0.0, 1.0))
    else:
        spectral_target = None

    sure_sigma_t = None
    if objective == "sure":
        if float(sure_sigma) > 0:
            sure_sigma_t = noisy.new_full((noisy.shape[0], 1, 1, 1), float(sure_sigma))
        else:
            sure_sigma_t = _estimate_sigma_log_domain(noisy, eps=1e-6)

    for step_idx in range(num_steps):
        opt.zero_grad(set_to_none=True)
        loss = None
        coh_anchor = incoh_anchor = anchor = None

        if objective == "consistency":
            preds = []
            coh_maps = []
            incoh_maps = []
            for aug, inv in aug_fns:
                aug_noisy = aug(noisy)
                if return_aux:
                    out_aug = model(aug_noisy, return_aux=True)
                    pred_aug, aux_aug = out_aug if isinstance(out_aug, tuple) else (out_aug, {})
                    coh_aug = aux_aug.get("coherent_map")
                    incoh_aug = aux_aug.get("incoherent_map")
                else:
                    pred_aug = model(aug_noisy)
                    coh_aug = incoh_aug = None

                preds.append(inv(pred_aug))
                if return_aux and coh_aug is not None and incoh_aug is not None:
                    coh_maps.append(inv(coh_aug))
                    incoh_maps.append(inv(incoh_aug))
            preds_stack = torch.stack(preds, dim=0)  # [N,B,1,H,W]
            anchor = preds_stack[0]
            consistency = torch.mean(torch.abs(preds_stack - anchor))
            loss = self_consistency * consistency
            if tv_weight > 0:
                loss = loss + tv_weight * aod.total_variation_loss(anchor)
            if anchor_weight > 0:
                loss = loss + anchor_weight * F.l1_loss(anchor, anchor_ref)
            if (
                coherence_weight > 0
                and return_aux
                and len(coh_maps) == len(preds)
                and len(incoh_maps) == len(preds)
            ):
                coh_stack = torch.stack(coh_maps, dim=0)  # [N,B,1,H,W]
                incoh_stack = torch.stack(incoh_maps, dim=0)  # [N,B,1,H,W]
                coh_anchor = coh_stack[0]
                incoh_anchor = incoh_stack[0]

        elif objective == "sure":
            loss, anchor, _sure_info = _sure_log_domain_loss(
                model=model,
                noisy=noisy,
                sigma=sure_sigma_t,
                eps_img=1e-6,
                eps_fd=float(sure_eps),
                hutchinson_samples=int(sure_hutchinson_samples),
                return_aux=return_aux,
            )
            if tv_weight > 0:
                loss = loss + tv_weight * aod.total_variation_loss(anchor)
            if anchor_weight > 0:
                loss = loss + anchor_weight * F.l1_loss(anchor, anchor_ref)

        else:
            # Self-supervised TTA objectives:
            # - N2V: blind-spot masking + masked-pixel reconstruction (unbiased under pixel-wise noise)
            # - B2U: revisible loss using full prediction as pseudo-target on blind spots (more robust for correlated noise)
            # - Hybrid: route between N2V and B2U based on a correlation score computed on the noisy input.
            mode = objective
            if objective == "hybrid":
                if corr_score <= low_th:
                    mode = "n2v"
                elif corr_score >= high_th:
                    mode = "b2u"
                else:
                    # Deadband: alternate to leverage both without committing.
                    mode = "n2v" if (step_idx % 2 == 0) else "b2u"

            if mode == "n2v":
                masked, mask_coords = aod.apply_blind_spot_mask(
                    noisy,
                    mask_ratio=float(n2v_mask_ratio),
                    box_size=int(n2v_box_size),
                    seed=None,
                    blindspot_dilation=int(n2v_blindspot_dilation),
                )
                if return_aux:
                    # IMPORTANT: For backbones with global residual skips (e.g., NAFNet),
                    # use the *original* noisy image as residual base so masked pixels do not leak
                    # into the output via the skip connection.
                    out_masked = model(masked, return_aux=True, residual_base=noisy)
                    anchor, aux_masked = out_masked if isinstance(out_masked, tuple) else (out_masked, {})
                    coh_anchor = aux_masked.get("coherent_map")
                    incoh_anchor = aux_masked.get("incoherent_map")
                else:
                    anchor = model(masked, residual_base=noisy)

                loss = n2v_loss_fn(anchor, noisy, mask_coords)
                if tv_weight > 0:
                    loss = loss + tv_weight * aod.total_variation_loss(anchor)
                if anchor_weight > 0:
                    loss = loss + anchor_weight * F.l1_loss(anchor, anchor_ref)
                del masked, mask_coords

            elif mode == "b2u":
                masked_input, mask, _blind_channel = b2u_masker(noisy)
                if return_aux:
                    out_full = model(noisy, return_aux=True)
                    pred_full, aux_full = out_full if isinstance(out_full, tuple) else (out_full, {})
                    out_masked = model(masked_input, return_aux=True, residual_base=noisy)
                    pred_masked, _aux_masked = out_masked if isinstance(out_masked, tuple) else (out_masked, {})
                    coh_anchor = aux_full.get("coherent_map")
                    incoh_anchor = aux_full.get("incoherent_map")
                else:
                    pred_full = model(noisy)
                    pred_masked = model(masked_input, residual_base=noisy)

                loss = b2u_loss_fn(pred_full, pred_masked, mask, noisy_input=None)
                if tv_weight > 0:
                    loss = loss + tv_weight * aod.total_variation_loss(pred_masked)
                if anchor_weight > 0:
                    loss = loss + anchor_weight * F.l1_loss(pred_full, anchor_ref)
                anchor = pred_full
                del masked_input, mask, _blind_channel, pred_masked

            else:
                raise RuntimeError(f"Unexpected hybrid mode: {mode}")

        if loss is None:
            raise RuntimeError("Internal error: loss is None during TTA.")

        if speckle_loss_fn is not None and speckle_weight > 0 and anchor is not None:
            # OCT-specific unsupervised prior: residual speckle should match Rayleigh-like statistics.
            loss = loss + speckle_weight * speckle_loss_fn(anchor, noisy=noisy)
        if banding_weight > 0 and anchor is not None:
            # Penalize row-wise fixed-pattern noise in the residual (simple banding proxy).
            residual = noisy - anchor
            row_mean = residual.mean(dim=-1, keepdim=True)
            loss = loss + banding_weight * aod.total_variation_loss(row_mean)

        if (
            coherence_weight > 0
            and return_aux
            and coh_anchor is not None
            and incoh_anchor is not None
            and anchor is not None
        ):
            residual = (noisy - anchor).abs()
            cv_map = aod.compute_local_cv(residual)
            cv_norm = (cv_map - cv_map.min()) / (cv_map.max() - cv_map.min() + 1e-8)
            corr_loss = aod.pearson_correlation_loss(coh_anchor, cv_norm)
            loss = loss + coherence_weight * corr_loss

            if casa_entropy_weight > 0:
                loss = loss + casa_entropy_weight * aod.casa_entropy_regularizer(coh_anchor, incoh_anchor)
            if casa_std_weight > 0 and casa_min_std > 0:
                loss = loss + casa_std_weight * aod.casa_std_floor_regularizer(coh_anchor, min_std=casa_min_std)

        if spectral_weight > 0 and spectral_target is not None and anchor is not None:
            residual = (noisy - anchor).clamp(0.0, 1.0)
            spec_pred = noise_characterizer(residual)
            loss = loss + spectral_weight * F.l1_loss(spec_pred["band_power"], spectral_target["band_power"])
            loss = loss + 0.1 * spectral_weight * F.l1_loss(spec_pred["spectral_slope"], spectral_target["spectral_slope"])

        loss.backward()
        opt.step()

    del opt
    if spectral_target is not None:
        del spectral_target
    if anchor_ref is not None:
        del anchor_ref
    if sure_sigma_t is not None:
        del sure_sigma_t

    model.eval()
    with torch.no_grad():
        out = model(noisy, return_aux=True) if return_aux else model(noisy)

    # Restore state (per-image TTA mode).
    if original_state is not None:
        model.load_state_dict({k: v.to(noisy.device) for k, v in original_state.items()})
        del original_state

    return out


def main():
    parser = argparse.ArgumentParser(description="OOD evaluation + optional per-image TTA with CASA map metrics.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True, help="paired_list file: noisy,clean per line")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None, help="Optional per-image JSONL output")

    parser.add_argument("--adapter", type=str, default="casa", choices=["none", "global", "spatial", "casa", "moe", "siminv"])
    parser.add_argument("--moe_experts", type=int, default=4, help="MoE adapter: number of experts (>=2).")
    parser.add_argument("--moe_hidden_channels", type=int, default=32, help="MoE adapter: expert hidden channels.")
    parser.add_argument("--moe_temperature", type=float, default=1.0, help="MoE adapter: softmax temperature (>0).")
    parser.add_argument("--siminv_hidden_channels", type=int, default=32, help="SimInv adapter: parameter estimator hidden channels.")
    parser.add_argument("--siminv_film_scale_gamma", type=float, default=0.1, help="SimInv adapter: FiLM gamma scale (match training).")
    parser.add_argument("--siminv_film_scale_beta", type=float, default=0.1, help="SimInv adapter: FiLM beta scale (match training).")
    parser.add_argument("--backbone", type=str, default="nafnet", choices=["unet", "nafnet", "noise2void", "neighbor2neighbor", "b2unet", "s2s"])
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--residual_mode", action="store_true")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional cap on number of images to evaluate (takes first N in pair list).")

    parser.add_argument("--tta_steps", type=int, default=0)
    parser.add_argument("--tta_scope", type=str, default="adapter", choices=["adapter", "full"])
    parser.add_argument("--tta_mode", type=str, default="per_image", choices=["per_image", "online", "calibrate"],
                        help="TTA application mode: per_image (reset weights each image), online (carry updates), "
                             "calibrate (adapt on first K images, then evaluate with fixed adapted weights).")
    parser.add_argument("--tta_calib_images", type=int, default=50,
                        help="For --tta_mode calibrate: number of first images used for unsupervised calibration.")
    parser.add_argument(
        "--tta_objective",
        type=str,
        default="consistency",
        choices=["consistency", "n2v", "b2u", "hybrid", "sure"],
        help="TTA self-supervision objective: 'consistency' (augmentation consistency), "
             "'n2v' (Noise2Void masked-pixel), 'b2u' (Blind2Unblind revisible), "
             "'hybrid' (route by noise correlation), 'sure' (Monte-Carlo SURE in log domain).",
    )
    parser.add_argument("--sure_eps", type=float, default=1e-3,
                        help="For --tta_objective sure: finite-difference epsilon in log domain (e.g., 1e-3).")
    parser.add_argument("--sure_sigma", type=float, default=0.0,
                        help="For --tta_objective sure: fixed sigma in log domain (0=estimate per image).")
    parser.add_argument("--sure_hutchinson_samples", type=int, default=1,
                        help="For --tta_objective sure: number of Hutchinson samples (>=1).")
    parser.add_argument("--tta_lr", type=float, default=8e-4)
    parser.add_argument("--tv_weight", type=float, default=1e-4)
    parser.add_argument("--self_consistency", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=0.05)
    parser.add_argument("--anchor_target", type=str, default="base", choices=["base", "noisy"],
                        help="Anchor reference for TTA: 'base' anchors to the pre-TTA prediction (recommended), "
                             "'noisy' anchors to the noisy input (more conservative but can hurt PSNR).")
    parser.add_argument("--tta_coherence_weight", type=float, default=0.0,
                        help="ReCASA-only: weight for residual-guided coherence loss during TTA (start with 0.01-0.05).")
    parser.add_argument("--tta_casa_entropy_weight", type=float, default=0.0,
                        help="ReCASA-only: entropy regularizer weight during TTA (start with 1e-3).")
    parser.add_argument("--tta_casa_std_weight", type=float, default=0.0,
                        help="ReCASA-only: std-floor regularizer weight during TTA (start with 1e-2 to 1e-1).")
    parser.add_argument("--tta_casa_min_std", type=float, default=0.0,
                        help="ReCASA-only: minimum coherent-map std target during TTA (e.g., 0.03-0.08).")
    parser.add_argument("--tta_speckle_weight", type=float, default=0.0,
                        help="OCT prior: speckle-statistics regularizer on residual during TTA (try 1e-3 to 1e-1).")
    parser.add_argument("--tta_banding_weight", type=float, default=0.0,
                        help="OCT prior: row-banding TV penalty on residual during TTA (try 1e-4 to 1e-2).")
    parser.add_argument("--n2v_mask_ratio", type=float, default=0.25)
    parser.add_argument("--n2v_box_size", type=int, default=5)
    parser.add_argument("--n2v_blindspot_dilation", type=int, default=3)
    parser.add_argument("--b2u_mask_ratio", type=float, default=0.5)
    parser.add_argument("--b2u_block_size", type=int, default=2)
    parser.add_argument("--hybrid_corr_threshold", type=float, default=0.18)
    parser.add_argument("--hybrid_corr_low", type=float, default=None)
    parser.add_argument("--hybrid_corr_high", type=float, default=None)
    parser.add_argument("--hybrid_corr_source", type=str, default="noisy", choices=["noisy", "residual_base"],
                        help="Correlation score source for hybrid routing: 'noisy' (faster) or "
                             "'residual_base' (uses noisy - base_pred; more noise-specific).")
    parser.add_argument("--spectral_weight", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=50, help="Print progress every N images (0 disables).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aod.device = device

    model = aod.build_model(
        base_channels=args.base_channels,
        residual_mode=args.residual_mode,
        adapter_type=args.adapter,
        backbone_type=args.backbone,
        moe_experts=args.moe_experts,
        moe_hidden_channels=args.moe_hidden_channels,
        moe_temperature=args.moe_temperature,
        siminv_hidden_channels=args.siminv_hidden_channels,
        siminv_film_scale_gamma=args.siminv_film_scale_gamma,
        siminv_film_scale_beta=args.siminv_film_scale_beta,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        msg = str(e)
        if ("Missing key(s) in state_dict" in msg) or ("Unexpected key(s) in state_dict" in msg):
            raise RuntimeError(
                msg
                + "\n\n"
                + "Checkpoint architecture mismatch.\n"
                + "Make sure your CLI matches how the checkpoint was trained.\n"
                + "Examples:\n"
                + "  - For NAFNet baseline checkpoints (adapter=none): add `--backbone nafnet --adapter none`.\n"
                + "  - For ReCASA checkpoints: add `--backbone nafnet --adapter casa`.\n"
                + "  - If base channels differ, pass `--base_channels` to match training.\n"
            ) from e
        raise
    model.eval()

    transform = aod.resize_to((args.image_size, args.image_size))
    ds = aod.PairedOCTDataset(args.pairs, transform=transform)
    if args.max_images is not None:
        max_n = int(args.max_images)
        if max_n <= 0:
            raise ValueError("--max_images must be > 0")
        ds = Subset(ds, list(range(min(max_n, len(ds)))))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl1 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    if args.output_json is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_json = os.path.join("results", f"ood_eval_{stem}_{args.adapter}_{args.backbone}.json")

    if args.output_jsonl is not None:
        # Clear existing file for clean runs.
        _ensure_dir(args.output_jsonl)
        if os.path.exists(args.output_jsonl):
            os.remove(args.output_jsonl)

    base_psnr, base_ssim, noisy_psnr, noisy_ssim = [], [], [], []
    base_casa = {"coh_mean": [], "coh_std": [], "entropy": [], "corr_residual_cv": []}
    base_moe = {"gate_top1_mean": [], "gate_entropy_mean": [], "gate_entropy_adapter": []}
    base_siminv_acc = {"theta_names": None, "_sum": None, "_sumsq": None, "_n": 0}

    print(f"[Eval] Images: {len(ds)} | device={device} | batch_size={args.batch_size}")
    print(f"[Eval] Checkpoint: {args.checkpoint}")
    print(f"[Eval] Model: backbone={args.backbone} adapter={args.adapter} base_channels={args.base_channels} residual={args.residual_mode}")

    t0 = time.time()
    n_total = len(ds)
    progress_every = max(0, int(args.progress_every))
    next_report = progress_every if progress_every > 0 else None
    n_done = 0
    with torch.inference_mode():
        for batch_idx, (x_noisy, x_clean) in enumerate(dl, start=1):
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            if args.adapter in ("casa", "moe", "siminv"):
                out = model(x_noisy, return_aux=True)
                pred, aux = out if isinstance(out, tuple) else (out, {})
                if args.adapter == "casa":
                    coh_map = aux.get("coherent_map")
                    incoh_map = aux.get("incoherent_map")
                else:
                    coh_map = incoh_map = None
            else:
                pred = model(x_noisy)
                coh_map = incoh_map = None

            for i in range(pred.size(0)):
                base_psnr.append(aod.compute_psnr(pred[i : i + 1], x_clean[i : i + 1]))
                base_ssim.append(aod.compute_ssim(pred[i : i + 1], x_clean[i : i + 1]))
                noisy_psnr.append(aod.compute_psnr(x_noisy[i : i + 1], x_clean[i : i + 1]))
                noisy_ssim.append(aod.compute_ssim(x_noisy[i : i + 1], x_clean[i : i + 1]))

            if args.adapter == "casa" and coh_map is not None and incoh_map is not None:
                metrics = _compute_casa_map_metrics(coh_map, incoh_map, x_noisy, pred)
                for k in base_casa.keys():
                    base_casa[k].extend([metrics[k]] * pred.size(0))
            if args.adapter == "moe":
                m = _compute_moe_metrics(aux)
                if m is not None:
                    for k in base_moe.keys():
                        base_moe[k].extend([m[k]] * pred.size(0))
            if args.adapter == "siminv":
                _accum_siminv_theta_stats(base_siminv_acc, aux)

            batch_n = int(pred.size(0))
            del x_noisy, x_clean, pred, coh_map, incoh_map

            n_done += batch_n
            if next_report is not None and n_done >= next_report:
                elapsed = max(1e-9, time.time() - t0)
                rate = n_done / elapsed
                eta = (n_total - n_done) / max(1e-9, rate)
                print(
                    f"[Eval] {n_done}/{n_total} ({100.0*n_done/max(1,n_total):.1f}%) "
                    f"| {rate:.2f} img/s | ETA {_format_eta(eta)} (batch {batch_idx}/{len(dl)})",
                    flush=True,
                )
                while next_report is not None and n_done >= next_report:
                    next_report += progress_every

    base_time = time.time() - t0

    tta_psnr, tta_ssim = [], []
    tta_casa = {"coh_mean": [], "coh_std": [], "entropy": [], "corr_residual_cv": []}
    tta_moe = {"gate_top1_mean": [], "gate_entropy_mean": [], "gate_entropy_adapter": []}
    tta_siminv_acc = {"theta_names": None, "_sum": None, "_sumsq": None, "_n": 0}
    if args.tta_steps > 0:
        noise_characterizer = aod.SpectralNoiseCharacterizer().to(device) if args.spectral_weight > 0 else None
        print(
            f"[TTA] objective={args.tta_objective} scope={args.tta_scope} steps={args.tta_steps} "
            f"lr={args.tta_lr} spectral={args.spectral_weight} speckle={args.tta_speckle_weight} banding={args.tta_banding_weight}",
            flush=True,
        )
        t1 = time.time()
        progress_every_tta = max(0, int(args.progress_every))
        next_report_tta = progress_every_tta if progress_every_tta > 0 else None

        if args.tta_mode == "calibrate":
            calib_n = int(args.tta_calib_images)
            if calib_n <= 0:
                raise ValueError("--tta_calib_images must be > 0 for --tta_mode calibrate")
            calib_n = min(calib_n, len(ds))
            print(f"[TTA-Calib] Adapting on first {calib_n}/{len(ds)} images", flush=True)
            for idx, (x_noisy, _x_clean) in enumerate(dl1, start=1):
                if idx > calib_n:
                    break
                x_noisy = x_noisy.to(device)
                _ = _tta_once(
                    model,
                    x_noisy,
                    scope=args.tta_scope,
                    objective=args.tta_objective,
                    reset_state=False,
                    num_steps=args.tta_steps,
                    lr=args.tta_lr,
                    tv_weight=args.tv_weight,
                    self_consistency=args.self_consistency,
                    anchor_weight=args.anchor_weight,
                    anchor_target=args.anchor_target,
                    n2v_mask_ratio=args.n2v_mask_ratio,
                    n2v_box_size=args.n2v_box_size,
                    n2v_blindspot_dilation=args.n2v_blindspot_dilation,
                    b2u_mask_ratio=args.b2u_mask_ratio,
                    b2u_block_size=args.b2u_block_size,
                    hybrid_corr_threshold=args.hybrid_corr_threshold,
                    hybrid_corr_low=args.hybrid_corr_low,
                    hybrid_corr_high=args.hybrid_corr_high,
                    hybrid_corr_source=args.hybrid_corr_source,
                    speckle_weight=args.tta_speckle_weight,
                    banding_weight=args.tta_banding_weight,
                    coherence_weight=args.tta_coherence_weight,
                    casa_entropy_weight=args.tta_casa_entropy_weight,
                    casa_std_weight=args.tta_casa_std_weight,
                    casa_min_std=args.tta_casa_min_std,
                    spectral_weight=args.spectral_weight,
                    augmentations=None,
                    noise_characterizer=noise_characterizer,
                    return_aux=(args.adapter == "casa"),
                    sure_eps=args.sure_eps,
                    sure_sigma=args.sure_sigma,
                    sure_hutchinson_samples=args.sure_hutchinson_samples,
                )
                del x_noisy, _x_clean, _
                if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == calib_n):
                    elapsed = max(1e-9, time.time() - t1)
                    rate = idx / elapsed
                    eta = (calib_n - idx) / max(1e-9, rate)
                    print(
                        f"[TTA-Calib] {idx}/{calib_n} ({100.0*idx/max(1,calib_n):.1f}%) "
                        f"| {rate:.2f} img/s | ETA {_format_eta(eta)}",
                        flush=True,
                    )

            print("[TTA-Calib] Evaluating adapted model on full set", flush=True)
            # Evaluate adapted model (no further updates) in the original order.
            idx_global = 0
            with torch.inference_mode():
                for batch_idx, (x_noisy, x_clean) in enumerate(dl, start=1):
                    x_noisy = x_noisy.to(device)
                    x_clean = x_clean.to(device)

                    if args.adapter in ("casa", "moe", "siminv"):
                        out = model(x_noisy, return_aux=True)
                        pred, aux = out if isinstance(out, tuple) else (out, {})
                        if args.adapter == "casa":
                            coh_map = aux.get("coherent_map")
                            incoh_map = aux.get("incoherent_map")
                        else:
                            coh_map = incoh_map = None
                    else:
                        pred = model(x_noisy)
                        coh_map = incoh_map = None

                    for i in range(pred.size(0)):
                        idx_global += 1
                        ps = aod.compute_psnr(pred[i : i + 1], x_clean[i : i + 1])
                        ss = aod.compute_ssim(pred[i : i + 1], x_clean[i : i + 1])
                        tta_psnr.append(ps)
                        tta_ssim.append(ss)
                        if args.output_jsonl is not None:
                            _append_jsonl(
                                args.output_jsonl,
                                {
                                    "idx": idx_global,
                                    "base_psnr": float(base_psnr[idx_global - 1]),
                                    "base_ssim": float(base_ssim[idx_global - 1]),
                                    "tta_psnr": float(ps),
                                    "tta_ssim": float(ss),
                                    "delta_psnr": float(ps - base_psnr[idx_global - 1]),
                                    "delta_ssim": float(ss - base_ssim[idx_global - 1]),
                                    "base_casa": {k: float(base_casa[k][idx_global - 1]) for k in base_casa.keys()} if args.adapter == "casa" else None,
                                },
                            )

                    if args.adapter == "casa" and coh_map is not None and incoh_map is not None:
                        m = _compute_casa_map_metrics(coh_map, incoh_map, x_noisy, pred)
                        for k in tta_casa.keys():
                            tta_casa[k].extend([m[k]] * pred.size(0))
                    if args.adapter == "moe":
                        m = _compute_moe_metrics(aux)
                        if m is not None:
                            for k in tta_moe.keys():
                                tta_moe[k].extend([m[k]] * pred.size(0))
                    if args.adapter == "siminv":
                        _accum_siminv_theta_stats(tta_siminv_acc, aux)
                    if next_report_tta is not None and idx_global >= next_report_tta:
                        elapsed = max(1e-9, time.time() - t1)
                        rate = idx_global / elapsed
                        eta = (len(ds) - idx_global) / max(1e-9, rate)
                        mean_delta = float(np.mean(np.asarray(tta_psnr) - np.asarray(base_psnr[:idx_global])))
                        print(
                            f"[TTA] {idx_global}/{len(ds)} ({100.0*idx_global/max(1,len(ds)):.1f}%) "
                            f"| mean ΔPSNR {mean_delta:+.2f} dB | {rate:.2f} img/s | ETA {_format_eta(eta)}",
                            flush=True,
                        )
                        while next_report_tta is not None and idx_global >= next_report_tta:
                            next_report_tta += progress_every_tta
                    del x_noisy, x_clean, pred, coh_map, incoh_map

            tta_time = time.time() - t1

        else:
            reset_each = args.tta_mode == "per_image"
            for idx, (x_noisy, x_clean) in enumerate(dl1, start=1):
                x_noisy = x_noisy.to(device)
                x_clean = x_clean.to(device)

                out = _tta_once(
                    model,
                    x_noisy,
                    scope=args.tta_scope,
                    objective=args.tta_objective,
                    reset_state=reset_each,
                    num_steps=args.tta_steps,
                    lr=args.tta_lr,
                    tv_weight=args.tv_weight,
                    self_consistency=args.self_consistency,
                    anchor_weight=args.anchor_weight,
                    anchor_target=args.anchor_target,
                    n2v_mask_ratio=args.n2v_mask_ratio,
                    n2v_box_size=args.n2v_box_size,
                    n2v_blindspot_dilation=args.n2v_blindspot_dilation,
                    b2u_mask_ratio=args.b2u_mask_ratio,
                    b2u_block_size=args.b2u_block_size,
                    hybrid_corr_threshold=args.hybrid_corr_threshold,
                    hybrid_corr_low=args.hybrid_corr_low,
                    hybrid_corr_high=args.hybrid_corr_high,
                    hybrid_corr_source=args.hybrid_corr_source,
                    speckle_weight=args.tta_speckle_weight,
                    banding_weight=args.tta_banding_weight,
                    coherence_weight=args.tta_coherence_weight,
                    casa_entropy_weight=args.tta_casa_entropy_weight,
                    casa_std_weight=args.tta_casa_std_weight,
                    casa_min_std=args.tta_casa_min_std,
                    spectral_weight=args.spectral_weight,
                    augmentations=None,
                    noise_characterizer=noise_characterizer,
                    return_aux=(args.adapter == "casa"),
                    sure_eps=args.sure_eps,
                    sure_sigma=args.sure_sigma,
                    sure_hutchinson_samples=args.sure_hutchinson_samples,
                )
                if args.adapter == "casa":
                    pred, aux = out if isinstance(out, tuple) else (out, {})
                    coh_map = aux.get("coherent_map")
                    incoh_map = aux.get("incoherent_map")
                else:
                    pred = out
                    coh_map = incoh_map = None

                ps = aod.compute_psnr(pred, x_clean)
                ss = aod.compute_ssim(pred, x_clean)
                tta_psnr.append(ps)
                tta_ssim.append(ss)

                if args.adapter == "casa" and coh_map is not None and incoh_map is not None:
                    m = _compute_casa_map_metrics(coh_map, incoh_map, x_noisy, pred)
                    for k in tta_casa.keys():
                        tta_casa[k].append(m[k])

                if args.output_jsonl is not None:
                    _append_jsonl(
                        args.output_jsonl,
                        {
                            "idx": idx,
                            "base_psnr": float(base_psnr[idx - 1]),
                            "base_ssim": float(base_ssim[idx - 1]),
                            "tta_psnr": float(ps),
                            "tta_ssim": float(ss),
                            "delta_psnr": float(ps - base_psnr[idx - 1]),
                            "delta_ssim": float(ss - base_ssim[idx - 1]),
                            "base_casa": {k: float(base_casa[k][idx - 1]) for k in base_casa.keys()} if args.adapter == "casa" else None,
                            "tta_casa": m if (args.adapter == "casa" and coh_map is not None and incoh_map is not None) else None,
                        },
                    )

                if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(ds)):
                    elapsed = max(1e-9, time.time() - t1)
                    rate = idx / elapsed
                    eta = (len(ds) - idx) / max(1e-9, rate)
                    mean_delta = float(np.mean(np.asarray(tta_psnr) - np.asarray(base_psnr[:idx])))
                    print(
                        f"[TTA] {idx}/{len(ds)} ({100.0*idx/max(1,len(ds)):.1f}%) "
                        f"| mean ΔPSNR {mean_delta:+.2f} dB | {rate:.2f} img/s | ETA {_format_eta(eta)}",
                        flush=True,
                    )

                del x_noisy, x_clean, pred, coh_map, incoh_map

            tta_time = time.time() - t1
    else:
        tta_time = 0.0

    base_siminv = _finalize_siminv_theta_stats(base_siminv_acc) if args.adapter == "siminv" else None
    tta_siminv = _finalize_siminv_theta_stats(tta_siminv_acc) if (args.adapter == "siminv" and args.tta_steps > 0) else None

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pairs": args.pairs,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "n_images": int(len(ds)),
        "timing_sec": {
            "base_total": float(base_time),
            "base_per_image": float(base_time / max(1, len(ds))),
            "tta_total": float(tta_time),
            "tta_per_image": float(tta_time / max(1, len(ds))) if args.tta_steps > 0 else None,
        },
        "noisy": {"psnr": _summary_stats(noisy_psnr), "ssim": _summary_stats(noisy_ssim)},
        "base": {"psnr": _summary_stats(base_psnr), "ssim": _summary_stats(base_ssim)},
        "base_casa": {k: _summary_stats(v) for k, v in base_casa.items()} if args.adapter == "casa" else None,
        "base_moe": {k: _summary_stats(v) for k, v in base_moe.items()} if args.adapter == "moe" else None,
        "base_siminv": base_siminv,
        "tta": {"psnr": _summary_stats(tta_psnr), "ssim": _summary_stats(tta_ssim)} if args.tta_steps > 0 else None,
        "tta_casa": {k: _summary_stats(v) for k, v in tta_casa.items()} if (args.adapter == "casa" and args.tta_steps > 0) else None,
        "tta_moe": {k: _summary_stats(v) for k, v in tta_moe.items()} if (args.adapter == "moe" and args.tta_steps > 0) else None,
        "tta_siminv": tta_siminv,
        "gain": {
            "psnr": float(np.mean(np.asarray(tta_psnr) - np.asarray(base_psnr))) if args.tta_steps > 0 else None,
            "ssim": float(np.mean(np.asarray(tta_ssim) - np.asarray(base_ssim))) if args.tta_steps > 0 else None,
        },
        "args": vars(args),
    }

    _write_json(args.output_json, result)
    print(f"[Saved] {args.output_json}")
    if args.output_jsonl:
        print(f"[Saved] {args.output_jsonl}")


if __name__ == "__main__":
    main()
