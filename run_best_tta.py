#!/usr/bin/env python3
"""Run TTA with best config and report per-dimension clinical ratios."""
import os, sys, json, torch
from torch.utils.data import DataLoader

from train_v8_cooperative import (
    NeuroSymbolicDenoiserV8Cooperative, PKU37Dataset,
)
from validate_crossdataset import TestTimeAdaptation, validate_dataset


def main():
    checkpoint = 'outputs/v8_overcorrect_fix/best_model_cooperative.pth'
    backbone = 'outputs/nafnet_pku37_w40/best_model.pth'
    device = 'cpu'

    # Best TTA config
    config = dict(tta_steps=10, tta_lr=5e-4, n_adapt_samples=5,
                  w_magnitude=0.5, w_cnr=1.5, w_consistency=1.0)

    # Load model
    print("Loading model...")
    model = NeuroSymbolicDenoiserV8Cooperative(
        backbone_name='nafnet', pretrained_backbone=backbone)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    print("Model loaded.\n")

    # Datasets
    datasets = [
        ('Duke17-Sparsity (cross-dataset)',
         'duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl'),
        ('Duke2013-SBSDI (cross-dataset)',
         'duke_sota_datasets/Duke17_Eval/duke2013_synth_eval.jsonl'),
    ]

    results = {}
    for ds_name, ds_path in datasets:
        if not os.path.exists(ds_path):
            print(f"WARNING: {ds_path} not found, skipping")
            continue

        ds = PKU37Dataset(ds_path, patch_size=0, is_train=False)
        print(f"\n{'='*84}")
        print(f"  {ds_name}: {len(ds)} samples")
        print(f"  TTA config: steps={config['tta_steps']}, lr={config['tta_lr']}")
        print(f"{'='*84}")

        # TTA adapt
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        tta = TestTimeAdaptation(
            model=model, device=device,
            tta_steps=config['tta_steps'], tta_lr=config['tta_lr'],
            n_adapt_samples=config['n_adapt_samples'],
            w_magnitude=config['w_magnitude'],
            w_cnr=config['w_cnr'], w_consistency=config['w_consistency'],
        )
        tta.adapt(loader, dataset_name=ds_name)

        # Full validation (prints per-dimension ratios)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        result = validate_dataset(model, loader, device, dataset_name=ds_name)
        if result:
            results[ds_name] = result

        # Restore for next dataset
        tta.restore_state()

    # Save
    out_path = 'outputs/best_tta_clinical_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
