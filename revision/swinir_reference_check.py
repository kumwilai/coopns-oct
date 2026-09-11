"""Was the SwinIR corrector trained against a different reference than it is tested on?

Part A reads what every SwinIR checkpoint recorded about its own run: whether the
backbone cache was on, the patch recipe, and the PSNR delta the training loop saw
(measured against the cached reference) next to the delta the validation loop saw
(measured against the tiled full resolution reference). No GPU needed.

Part B rebuilds both references for N validation images and runs the SAME corrector
on each:
    R_test  = tiled full resolution SwinIR output        (what ablation_runner scores)
    R_train = up2(SwinIR(down2(noisy)))                  (what the cache held, ds_factor 2)
and likewise the uncertainty from full resolution versus 2x downsampled conv_first
features. If the corrector loses far less PSNR on R_train than on R_test, it was
calibrated to the blurred reference. The high frequency share of the correction
says whether the loss on R_test comes from added high frequency energy.

Run from the repo root on the GPU box:
    python revision/swinir_reference_check.py --device cuda --n 8
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from train_v8_cooperative import (NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset,
                                  compute_psnr)


def load_model(ckpt_path, backbone_pth, device):
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='swinir', pretrained_backbone=backbone_pth, hidden_channels=64)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    ms = model.state_dict()
    ok = {}
    for k, v in state.items():
        key = k.replace('._orig_mod.', '.').replace('_orig_mod.', '')
        if key in ms and v.shape == ms[key].shape:
            ok[key] = v
    model.load_state_dict(ok, strict=False)
    print(f"loaded {len(ok)}/{len(ms)} tensors from {ckpt_path}")
    return model.to(device).eval()


def part_a(paths):
    print("\n=== Part A: what the checkpoints recorded about their own training run")
    for p in paths:
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        c = torch.load(p, map_location='cpu', weights_only=False)
        a = c.get('args', {})
        tm, vm = c.get('train_metrics', {}) or {}, c.get('val_metrics', {}) or {}
        tr_d = tm.get('psnr_corrected', float('nan')) - tm.get('psnr_backbone', float('nan'))
        vk = [k for k in vm if 'psnr' in k.lower()]
        print(f"  {p}")
        print(f"    cache_backbone={a.get('cache_backbone')} patches_per_image={a.get('patches_per_image')} "
              f"batch_size={a.get('batch_size')} dz={a.get('psnr_dead_zone')} epoch={c.get('epoch')}")
        print(f"    TRAIN (vs cached reference): psnr_backbone={tm.get('psnr_backbone', float('nan')):.3f} "
              f"psnr_corrected={tm.get('psnr_corrected', float('nan')):.3f} delta={tr_d:+.3f}")
        print(f"    VAL   (vs tiled full res):   " + " ".join(f"{k}={vm[k]:.3f}" for k in vk if isinstance(vm[k], (int, float))))


def hf_share(c):
    lp = F.interpolate(F.avg_pool2d(c, 2, 2), size=c.shape[2:], mode='bilinear', align_corners=False)
    hf = c - lp
    return (hf.pow(2).sum() / c.pow(2).sum().clamp(min=1e-12)).item(), lp, hf


@torch.inference_mode()
def part_b(model, val_jsonl, n, device):
    print(f"\n=== Part B: same corrector, two references, {n} validation images at full resolution")
    ds = PKU37Dataset(val_jsonl, max_samples=n, patch_size=0, is_train=False)
    bw = model.backbone
    acc = {}

    def add(k, v):
        acc[k] = acc.get(k, 0.0) + v

    for i in range(len(ds)):
        b = ds[i]
        clean = b['clean'].unsqueeze(0).to(device)
        noisy = b['noisy'].unsqueeze(0).to(device)
        H, W = noisy.shape[2:]

        # test time path (tiled full res backbone + full res conv_first uncertainty)
        R_test, U_test = bw(noisy)

        # old cached training path, ds_factor 2
        nd = F.avg_pool2d(noisy, 2, 2)
        ph, pw = (8 - nd.shape[2] % 8) % 8, (8 - nd.shape[3] % 8) % 8
        ndp = F.pad(nd, (0, pw, 0, ph), mode='reflect') if (ph or pw) else nd
        bb = bw.backbone(ndp)
        bb = bb[0] if isinstance(bb, tuple) else bb
        bb = bb[:, :, :nd.shape[2], :nd.shape[3]]
        R_train = F.interpolate(bb, size=(H, W), mode='bilinear', align_corners=False).clamp(0, 1)
        feat = bw.backbone.conv_first(ndp)[:, :, :nd.shape[2], :nd.shape[3]]
        U_train = bw._compute_uncertainty_from_enc1(
            F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False))

        def run(R, U):
            out, _ = model.corrector(R, noisy, None, nafnet_uncertainty=U, return_details=False)
            return out

        c_tt = run(R_test, U_test)     # exactly what the evaluator scores
        c_rr = run(R_train, U_train)   # exactly what training optimised
        c_rt = run(R_train, U_test)    # blurred reference, sharp uncertainty
        c_tr = run(R_test, U_train)    # sharp reference, blurred uncertainty

        p = lambda x: compute_psnr(x, clean)
        add('psnr R_test', p(R_test)); add('psnr R_train', p(R_train))
        add('d test ref/test unc', p(c_tt) - p(R_test))
        add('d train ref/train unc', p(c_rr) - p(R_train))
        add('d train ref/test unc', p(c_rt) - p(R_train))
        add('d test ref/train unc', p(c_tr) - p(R_test))
        add('unc mean test', U_test.mean().item()); add('unc mean train', U_train.mean().item())

        # decompose the test time correction into low and high frequency halves
        c = c_tt - R_test
        share, lp, hf = hf_share(c)
        add('hf share of correction (test)', share)
        add('d if only LP part applied', p((R_test + lp).clamp(0, 1)) - p(R_test))
        add('d if only HF part applied', p((R_test + hf).clamp(0, 1)) - p(R_test))
        add('|corr| mean (test)', c.abs().mean().item())
        add('|corr| mean (train)', (c_rr - R_train).abs().mean().item())
        # does the backbone residual itself carry high frequency error?
        e = R_test - clean
        add('hf share of backbone error', hf_share(e)[0])

    n_img = len(ds)
    for k, v in acc.items():
        print(f"  {k:34s} {v / n_img:+.4f}")
    print("\nReading: if 'd train ref/train unc' sits inside the dead zone while 'd test ref/test unc'")
    print("is one to two dB worse, the corrector was calibrated to the blurred cached reference.")
    print("If both are equally bad, the cause is elsewhere (see the no_edge/no_uncertainty runs).")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--val_jsonl', default='pku37_oct_dataset/pku37_real_val.jsonl')
    ap.add_argument('--checkpoint', default='outputs/revision/final_swinir/best_model_cooperative.pth')
    ap.add_argument('--backbone', default='checkpointpaper/swinir_backbone.pth')
    ap.add_argument('--skip_b', action='store_true')
    args = ap.parse_args()

    part_a(sorted(glob.glob('outputs/revision/sw_swinir_*/best_model_cooperative.pth')) +
           [args.checkpoint])
    if not args.skip_b:
        model = load_model(args.checkpoint, args.backbone, args.device)
        part_b(model, args.val_jsonl, args.n, args.device)
