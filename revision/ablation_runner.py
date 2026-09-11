#!/usr/bin/env python3
"""Run one intervention on the trained corrector and score it.

Interventions
  none                      unmodified model
  drop_P1 .. drop_P6        remove one clinical predicate at inference
  no_negotiator             uniform allocation instead of the fuzzy rules
  no_edge                   disable the edge recovery branch
  no_uncertainty            replace the confidence map by a constant
  no_bg_smooth              disable background smoothing
  rule=NAME:VALUE           set a negotiator rule weight, pre sigmoid
  base=VALUE                set the base allocation, pre sigmoid
  tnorm=NAME                replace the conjunction, one of lukasiewicz product minimum
  verifier_soft             use the training time blend schedule at inference

The scoring code is imported from eval_pku37_test so every number is produced by
exactly the same metric implementation as the published tables.

usage
  python revision/ablation_runner.py --backbone_name nafnet \
      --checkpoint checkpointpaper/nafnet_pku37_cooperative.pth \
      --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
      --intervention drop_P2 --output_json outputs/revision/lopo_P2.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

# Resolve the project root from this file rather than from a fixed path, so the
# scripts run unchanged on any machine and from any working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "revision"))

from train_v8_cooperative import NeuroSymbolicDenoiserV8Cooperative  # noqa: E402
from eval_pku37_test import evaluate_pku37, print_summary  # noqa: E402

PRED_KEYS = ["P1", "P2", "P3", "P4", "P5", "P6"]


class PredicateDropper(nn.Module):
    """Wrap the predicate bank and neutralise one predicate.

    The dropped predicate reports a perfect score and an all zero failure map,
    so it can neither trigger an allocation boost nor steer a corrector.
    """

    def __init__(self, inner, drop_key):
        super().__init__()
        self.inner = inner
        self.drop_key = drop_key

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        k = self.drop_key
        if k in out:
            entry = dict(out[k])
            fm = entry.get("failure_map")
            if isinstance(fm, torch.Tensor):
                entry["failure_map"] = torch.zeros_like(fm)
            sc = entry.get("score")
            if isinstance(sc, torch.Tensor):
                entry["score"] = torch.ones_like(sc)
            else:
                entry["score"] = 1.0
            entry["passed"] = True
            out[k] = entry
        if "scores" in out and isinstance(out["scores"], dict) and k in out["scores"]:
            s = out["scores"][k]
            out["scores"][k] = torch.ones_like(s) if isinstance(s, torch.Tensor) else 1.0
        return out


def make_tnorm(name):
    if name == "product":
        return lambda a, b: a * b
    if name == "minimum":
        return lambda a, b: torch.minimum(a, b) if isinstance(b, torch.Tensor) else torch.clamp(a, max=b)
    raise ValueError(name)


def apply_intervention(model, spec):
    """Mutate the model in place. Returns a short human readable description."""
    corr = model.corrector
    if spec == "none":
        return "unmodified model"

    if spec.startswith("drop_"):
        key = spec.split("_", 1)[1]
        if key not in PRED_KEYS:
            raise ValueError(spec)
        corr.predicates = PredicateDropper(corr.predicates, key)
        return f"predicate {key} removed at inference"

    if spec in ("no_negotiator", "no_edge", "no_uncertainty", "no_bg_smooth"):
        setattr(corr, f"_ablation_{spec}", True)
        return f"flag _ablation_{spec} enabled"

    if spec.startswith("rule="):
        name, value = spec[5:].split(":")
        value = float(value)
        with torch.no_grad():
            corr.negotiator.rule_weights[name].fill_(value)
        return f"rule weight {name} set to {value} before the sigmoid"

    if spec.startswith("base="):
        value = float(spec[5:])
        with torch.no_grad():
            corr.negotiator.base_allocation.fill_(value)
        return f"base allocation set to {value} before the sigmoid"

    if spec.startswith("tnorm="):
        name = spec[6:]
        if name == "lukasiewicz":
            return "conjunction unchanged"
        corr.negotiator.logic.soft_and = make_tnorm(name)
        return f"conjunction replaced by the {name} t norm"

    if spec == "verifier_soft":
        v = corr.verifier
        orig = v.forward

        def soft_forward(*a, **kw):
            was = v.training
            v.train(True)
            try:
                return orig(*a, **kw)
            finally:
                v.train(was)
        v.forward = soft_forward
        return "training time blend schedule used at inference"

    raise ValueError(f"unknown intervention {spec}")


def load_model(args):
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name=args.backbone_name,
        pretrained_backbone=args.pretrained_backbone,
        hidden_channels=args.hidden_channels,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    compatible = {}
    for k, v in state.items():
        key = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        if key in model_state and v.shape == model_state[key].shape:
            compatible[key] = v
    model.load_state_dict(compatible, strict=False)
    return model, len(compatible), len(model_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_name", default="nafnet",
                    choices=["nafnet", "kbnet", "dncnn", "swinir"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pretrained_backbone", required=True)
    ap.add_argument("--test_jsonl", default="pku37_oct_dataset/pku37_real_test.jsonl")
    ap.add_argument("--hidden_channels", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--intervention", default="none")
    ap.add_argument("--bg_rule", default="intensity",
                    help="Must match the rule the checkpoint was trained with.")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N images, 0 means all")
    args = ap.parse_args()

    torch.set_num_threads(2)

    test_jsonl = args.test_jsonl
    if args.limit:
        import tempfile
        lines = open(test_jsonl).read().strip().split("\n")[: args.limit]
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        fh.write("\n".join(lines) + "\n")
        fh.close()
        test_jsonl = fh.name

    model, n_ok, n_tot = load_model(args)
    model.corrector.bg_rule = args.bg_rule
    desc = apply_intervention(model, args.intervention)
    model = model.to(args.device).eval()

    print(f"intervention  {args.intervention}")
    print(f"background    {args.bg_rule}")
    print(f"description   {desc}")
    print(f"weights       {n_ok} of {n_tot} tensors loaded")
    print(f"images        {test_jsonl}")

    t0 = time.time()
    results = evaluate_pku37(model, test_jsonl, args.device)
    summary = print_summary(results, f"{args.backbone_name} [{args.intervention}]")
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    # Write to a temporary file and rename, so a crash or a second writer can
    # never leave a half written result behind.
    tmp = args.output_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "backbone": args.backbone_name,
            "intervention": args.intervention,
            "description": desc,
            "checkpoint": args.checkpoint,
            "test_jsonl": args.test_jsonl,
            "n_images": len(results),
            "seconds": elapsed,
            "per_image": results,
            "summary": summary,
        }, f, indent=2)
    os.replace(tmp, args.output_json)
    print(f"\nsaved {args.output_json}  in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
