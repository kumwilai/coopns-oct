#!/usr/bin/env python3
"""Classical post correction operators applied to the same frozen backbone output.

These answer the part of the fair comparison request that asks for competitors of
similar complexity. They act on exactly the same input as our wrapper and are
scored with exactly the same metric code, imported from eval_pku37_test.

  unsharp   b + amount * (b - blur(b))
  clahe     contrast limited adaptive histogram equalisation on b

usage
  python revision/classical_baselines.py --mode tune          # pick the setting on the subset
  python revision/classical_baselines.py --mode eval --op unsharp --amount 0.5
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Resolve the project root from this file rather than from a fixed path, so the
# scripts run unchanged on any machine and from any working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "revision"))
from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative
from eval_pku37_test import evaluate_pku37, print_summary


def gaussian(t, sigma):
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma * sigma)); g = (g / g.sum()).view(1, 1, 1, k)
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="reflect"), g)
    return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(2, 3))


class Classical:
    """Exposes the two calls that the shared evaluation loop makes on a model."""

    def __init__(self, backbone, op, amount):
        self._bb, self.op, self.amount = backbone, op, amount

    def backbone_fn(self, noisy):
        return self._bb(noisy)

    def corrector_fn(self, b, noisy, _unused, nafnet_uncertainty=None, return_details=False):
        if self.op == "unsharp":
            out = (b + self.amount * (b - gaussian(b, 2.0))).clamp(0, 1)
        elif self.op == "clahe":
            arr = (b[0, 0].numpy() * 255).astype(np.uint8)
            cl = cv2.createCLAHE(clipLimit=self.amount, tileGridSize=(8, 8)).apply(arr)
            out = torch.from_numpy(cl.astype(np.float32) / 255.0).view_as(b)
        else:
            raise ValueError(self.op)
        return out, {}

    # the evaluation loop uses attribute access
    @property
    def backbone(self):
        return self.backbone_fn

    @property
    def corrector(self):
        return self.corrector_fn

    def to(self, _):
        return self

    def eval(self):
        return self


def load_backbone(name, path, ckpt):
    m = NeuroSymbolicDenoiserV8Cooperative(backbone_name=name, pretrained_backbone=path,
                                           hidden_channels=64)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = {k.replace("._orig_mod.", ".").replace("_orig_mod.", ""): v
              for k, v in sd.get("model_state_dict", sd).items()}
        ms = m.state_dict()
        m.load_state_dict({k: v for k, v in sd.items()
                           if k in ms and v.shape == ms[k].shape}, strict=False)
    m.eval()
    return m.backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tune", choices=["tune", "eval"])
    ap.add_argument("--op", default="unsharp", choices=["unsharp", "clahe"])
    ap.add_argument("--amount", type=float, default=0.5)
    ap.add_argument("--backbone_name", default="nafnet")
    ap.add_argument("--pretrained_backbone", default="checkpointpaper/nafnet_backbone.pth")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--jsonl", default="pku37_oct_dataset/pku37_real_test.jsonl")
    ap.add_argument("--tune_jsonl", default="revision/pku37_subset40.jsonl")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()
    torch.set_num_threads(2)

    bb = load_backbone(args.backbone_name, args.pretrained_backbone, args.checkpoint)

    if args.mode == "tune":
        grid = {"unsharp": [0.25, 0.5, 0.75, 1.0], "clahe": [1.0, 2.0, 3.0, 4.0]}
        best = {}
        for op in ("unsharp", "clahe"):
            rows = []
            for a in grid[op]:
                res = evaluate_pku37(Classical(bb, op, a), args.tune_jsonl, "cpu")
                dp = float(np.mean([r["psnr_corrected"] - r["psnr_backbone"] for r in res]))
                dc = float(np.mean([100 * (r["cnr_corrected"] - r["cnr_backbone"]) /
                                    (abs(r["cnr_backbone"]) + 1e-6) for r in res]))
                rows.append({"amount": a, "d_psnr": dp, "d_cnr": dc})
                print(f"  {op} {a}  dPSNR {dp:+.3f}  dCNR {dc:+.2f}%", flush=True)
            ok = [r for r in rows if r["d_psnr"] > -1.0] or rows
            best[op] = max(ok, key=lambda r: r["d_cnr"])
            print(f"  chosen for {op}: {best[op]}")
        json.dump(best, open("outputs/revision/classical_tuning.json", "w"), indent=2)
        print("saved outputs/revision/classical_tuning.json")
        return

    res = evaluate_pku37(Classical(bb, args.op, args.amount), args.jsonl, "cpu")
    summary = print_summary(res, f"{args.op} amount {args.amount}")
    out = args.output_json or f"outputs/revision/classical_{args.op}.json"
    json.dump({"operator": args.op, "amount": args.amount, "n_images": len(res),
               "per_image": res, "summary": summary}, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
