#!/usr/bin/env python3
"""One pass over the test set that produces the evidence for four reviewer points.

Calibration   does the confidence map predict the error of the backbone
Theory        does the Lipschitz bound hold, do the contrast conditions hold,
              which safety constraint binds and how often
Safety        does the correction invent edges, does it erase weak structure,
              what happens inside dark enclosed regions, what is the worst case

usage
  python revision/diagnostics.py --backbone_name nafnet \
     --checkpoint outputs/revision/retrain_nafnet/best_model_cooperative.pth \
     --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
     --output_json outputs/revision/diagnostics_nafnet.json
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset
from validate_crossdataset import otsu_tissue_mask

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def edges(t):
    gx = F.conv2d(t, SOBEL_X, padding=1)
    gy = F.conv2d(t, SOBEL_Y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-12
    return float((ra * rb).sum() / d)


def sparsification_error(unc, err, bins=20):
    """Remove the most uncertain fraction and watch the remaining error fall.

    Returns the area between the curve driven by the predicted uncertainty and
    the curve driven by the true error. Zero means a perfect ranking.
    """
    order_pred = np.argsort(-unc)
    order_true = np.argsort(-err)
    fr = np.linspace(0, 0.9, bins)
    cp, ct = [], []
    n = len(err)
    for f in fr:
        k = int(n * f)
        cp.append(err[order_pred[k:]].mean())
        ct.append(err[order_true[k:]].mean())
    cp, ct = np.array(cp), np.array(ct)
    denom = cp[0] + 1e-12
    return float(np.trapz(cp - ct, fr) / denom), [float(x) for x in cp / denom], [float(x) for x in ct / denom]


def dark_enclosed_mask(b, tissue):
    """Dark regions surrounded by tissue, a geometric stand in for fluid."""
    dark = (b < torch.quantile(b, 0.35)).float()
    inside = F.max_pool2d(tissue, 15, stride=1, padding=7)
    inside = -F.max_pool2d(-inside, 15, stride=1, padding=7)
    return dark * inside



def layer_boundaries(t):
    """Locate two retinal boundaries per column without any training.

    The inner limiting membrane is taken as the first row whose intensity rises
    above half of the column maximum. The retinal pigment epithelium is taken as
    the brightest row of the column. Both are smoothed laterally with a median
    filter so that a single noisy column cannot move the estimate.
    """
    x = t[0, 0]
    H, W = x.shape
    col_max = x.max(dim=0).values.clamp(min=1e-6)
    above = (x > 0.5 * col_max.unsqueeze(0)).float()
    idx = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)
    big = torch.full_like(idx, float(H))
    ilm = torch.where(above > 0, idx, big).min(dim=0).values
    rpe = x.argmax(dim=0).float()

    def smooth(v, k=9):
        pad = k // 2
        vv = F.pad(v.view(1, 1, -1), (pad, pad), mode="replicate")
        return vv.unfold(2, k, 1).median(dim=-1).values.view(-1)

    return smooth(ilm), smooth(rpe)


def boundary_shift(pred, ref):
    """Mean absolute displacement in pixels between two boundary traces."""
    return float((pred - ref).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_name", default="nafnet")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pretrained_backbone", required=True)
    ap.add_argument("--test_jsonl", default="pku37_oct_dataset/pku37_real_test.jsonl")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)

    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone, hidden_channels=64)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
          for k, v in ck.get("model_state_dict", ck).items()}
    ms = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items()
                           if k in ms and v.shape == ms[k].shape}, strict=False)
    model.eval()
    neg = model.corrector.negotiator
    L_theory = float(neg.lipschitz_constant(include_rule5=False))

    ds = PKU37Dataset(args.test_jsonl, patch_size=0, is_train=False)
    n_img = len(ds) if not args.limit else min(args.limit, len(ds))
    rows = []
    unc_all, err_all = [], []

    for i in range(n_img):
        s = ds[i]
        noisy, clean = s["noisy"].unsqueeze(0), s["clean"].unsqueeze(0)
        with torch.no_grad():
            b, u_in = model.backbone(noisy)
            out, info = model.corrector(b, noisy, None, nafnet_uncertainty=u_in,
                                        return_details=True)

        tissue = otsu_tissue_mask(b)
        bg = 1.0 - tissue
        r = {}

        # calibration, subsample pixels to keep memory small
        umap = info.get("nafnet_confidence_map")
        if umap is not None:
            unc = (1.0 - umap).flatten().numpy()
            err = (b - clean).abs().flatten().numpy()
            k = np.random.RandomState(i).choice(len(err), size=min(4000, len(err)), replace=False)
            unc_all.append(unc[k]); err_all.append(err[k])
            r["rho_unc_err"] = spearman(unc[k], err[k])

        # allocation and the Lipschitz check on the observed rule vector
        a = info["allocation_maps"]["gain"]
        r["alloc_min"], r["alloc_mean"] = float(a.min()), float(a.mean())
        r["alloc_max"], r["alloc_std"] = float(a.max()), float(a.std())
        r["alloc_at_clamp"] = float(((a >= 0.9999) | (a <= 1e-4)).float().mean())
        act = info.get("negotiation_trace", {}).get("gain", {}) or {}
        rv = torch.tensor([float(act.get(k, 0.0)) for k in
                           ("rule1_trust_nafnet", "rule2_use_corrector",
                            "rule3_boost_failing", "rule4_balance")]).clamp(0, 1)
        rng = np.random.RandomState(1000 + i)
        ratios = []
        for _ in range(24):
            d = torch.tensor(rng.uniform(-0.05, 0.05, size=4).astype(np.float32))
            rv2 = (rv + d).clamp(0, 1)
            a1 = neg.combine_rules_bounded(rv[0], rv[1], rv[2], rv[3])
            a2 = neg.combine_rules_bounded(rv2[0], rv2[1], rv2[2], rv2[3])
            den = float((rv2 - rv).abs().max())
            if den > 1e-6:
                ratios.append(abs(float(a1) - float(a2)) / den)
        r["lipschitz_ratio_max"] = float(max(ratios)) if ratios else 0.0

        # contrast margin conditions
        def mean_on(t, m):
            return float((t * m).sum() / m.sum().clamp(min=1))
        muT_b, muT_q = mean_on(b, tissue), mean_on(out, tissue)
        muB_b, muB_q = mean_on(b, bg), mean_on(out, bg)
        sdB = lambda t: float(torch.sqrt((((t - mean_on(t, bg)) ** 2) * bg).sum() / bg.sum().clamp(min=1)))
        r["eta_T"], r["eta_B"] = muT_q - muT_b, muB_q - muB_b
        r["sigma_ratio"] = sdB(out) / (sdB(b) + 1e-9)
        r["cond_margin"] = bool(r["eta_T"] > r["eta_B"])
        r["cond_sigma"] = bool(r["sigma_ratio"] <= 1.0)

        # safety layer
        v = info.get("verification", {})
        r["n_pass"] = int(v.get("guarantees_passed", -1))
        r["blend_weight"] = float(v.get("blend_weight", float("nan")))
        for g in (v.get("guarantees") or []):
            if isinstance(g, (list, tuple)) and len(g) == 2:
                r["pass_" + g[0]] = bool(g[1].get("passed", False))

        # safety measurements against the clean reference
        e_b, e_q, e_c = edges(b), edges(out), edges(clean)
        thr = float(torch.quantile(e_c, 0.90))
        no_edge_in_ref = (e_c < float(torch.quantile(e_c, 0.50))).float()
        r["invented_edge_rate_backbone"] = float((((e_b > thr).float() * no_edge_in_ref).sum()
                                                  / no_edge_in_ref.sum().clamp(min=1)))
        r["invented_edge_rate_corrected"] = float((((e_q > thr).float() * no_edge_in_ref).sum()
                                                   / no_edge_in_ref.sum().clamp(min=1)))
        weak = ((e_c > float(torch.quantile(e_c, 0.50))) &
                (e_c < float(torch.quantile(e_c, 0.60)))).float()
        r["weak_retention_backbone"] = float((e_b * weak).sum() / (e_c * weak).sum().clamp(min=1e-6))
        r["weak_retention_corrected"] = float((e_q * weak).sum() / (e_c * weak).sum().clamp(min=1e-6))
        dm = dark_enclosed_mask(b, tissue)
        r["dark_region_fraction"] = float(dm.mean())
        r["dark_region_abs_change"] = float(((out - b).abs() * dm).sum() / dm.sum().clamp(min=1))
        r["global_abs_change"] = float((out - b).abs().mean())
        # downstream check, do the layer boundaries move away from the reference
        ilm_c, rpe_c = layer_boundaries(clean)
        ilm_b, rpe_b = layer_boundaries(b)
        ilm_q, rpe_q = layer_boundaries(out)
        r["ilm_shift_backbone"] = boundary_shift(ilm_b, ilm_c)
        r["ilm_shift_corrected"] = boundary_shift(ilm_q, ilm_c)
        r["rpe_shift_backbone"] = boundary_shift(rpe_b, rpe_c)
        r["rpe_shift_corrected"] = boundary_shift(rpe_q, rpe_c)
        r["boundary_not_worsened"] = bool(
            r["ilm_shift_corrected"] <= r["ilm_shift_backbone"] + 0.5 and
            r["rpe_shift_corrected"] <= r["rpe_shift_backbone"] + 0.5)

        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_img}", flush=True)

    agg = {}
    keys = set().union(*[set(r) for r in rows])
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if isinstance(vals[0], bool):
            agg[k] = {"true": int(sum(vals)), "of": len(vals)}
        else:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "min": float(np.min(vals)), "max": float(np.max(vals))}

    out = {"backbone": args.backbone_name, "checkpoint": args.checkpoint,
           "n_images": len(rows), "lipschitz_constant_theory": L_theory,
           "per_image": rows, "summary": agg}
    if unc_all:
        u = np.concatenate(unc_all); e = np.concatenate(err_all)
        ause, cp, ct = sparsification_error(u, e)
        out["calibration"] = {"spearman_pooled": spearman(u, e),
                              "sparsification_error": ause,
                              "curve_predicted": cp, "curve_oracle": ct}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    json.dump(out, open(args.output_json, "w"), indent=2)
    print("saved", args.output_json)
    print(f"  Lipschitz theory {L_theory:.4f}   worst measured "
          f"{agg.get('lipschitz_ratio_max',{}).get('max',0):.4f}")


if __name__ == "__main__":
    main()
