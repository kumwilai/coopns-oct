# OCT Denoising Benchmarks — IEEE TMI

Self-contained package: trains SOTA baselines + our cooperative corrector, evaluates on PKU37.

## Setup

1. Copy this `benchmarks/` folder to a GPU machine
2. Copy `pku37_oct_dataset/` into the folder
3. Install: `pip install torch torchvision numpy pillow tqdm`

```
benchmarks/
├── run_benchmarks.py                        # SOTA baselines (Step 1)
├── train_v8_cooperative.py                  # Our method (Step 2)
├── sweep_tta_hyperparams.py                 # TTA sweep (Step 3)
├── validate_crossdataset.py                 # Cross-dataset eval
├── models/                                  # SOTA model definitions
│   ├── dncnn_7m.py, swinir_7m.py, kbnet_7m.py, mambair_7m.py
├── neuro_symbolic_corrector_v8*.py          # Our cooperative framework
├── pretrained/
│   └── nafnet_backbone.pth                  # Frozen NAFNet backbone (7.01M)
├── pku37_oct_dataset/                       # <-- copy dataset here
└── README.md
```

## Step 1: Train SOTA Baselines

```bash
python run_benchmarks.py --data_dir pku37_oct_dataset --epochs 50
```

Trains DnCNN-7M, SwinIR-7M, KBNet-7M, MambaIR-7M (all ~7M params).
Outputs `results/results.json` with all IEEE TMI metrics + LaTeX table.

Train specific models only:
```bash
python run_benchmarks.py --methods dncnn_7m swinir_7m --epochs 50
```

## Step 2: Train Our Cooperative Corrector

```bash
python train_v8_cooperative.py \
    --train_jsonl pku37_oct_dataset/pku37_real_train.jsonl \
    --val_jsonl pku37_oct_dataset/pku37_real_val.jsonl \
    --pretrained_backbone pretrained/nafnet_backbone.pth \
    --backbone_width 40 --freeze_backbone \
    --epochs 15 --batch_size 4 --val_every 1 \
    --output_dir outputs/v8_overcorrect_fix \
    --no_compile
```

## Step 3: TTA Hyperparameter Sweep (Cross-Dataset)

```bash
python sweep_tta_hyperparams.py \
    --checkpoint outputs/v8_overcorrect_fix/best_model_cooperative.pth \
    --backbone pretrained/nafnet_backbone.pth \
    --backbone_width 40 \
    --device cuda \
    --datasets "duke17,duke2013" \
    --output_json outputs/tta_sweep_results.json
```

## Models

| Model | Params | Type |
|-------|--------|------|
| DnCNN-7M | 7.03M | 17-layer CNN, 228 channels |
| SwinIR-7M | ~7M | 6 RSTB blocks, Swin Transformer |
| KBNet-7M | ~7M | UNet + Kernel Basis Attention |
| MambaIR-7M | ~7M | 6 RSSB blocks, 4-dir SSM scan |
| NAFNet (backbone) | 7.01M | Frozen, width=40 |
| Ours (cooperative) | 7.01M + 0.23M | NAFNet + 5 cooperative correctors |

## Output

- `results/results.json` — SOTA baseline metrics (paper table)
- `outputs/v8_overcorrect_fix/best_model_cooperative.pth` — our best model
- `outputs/tta_sweep_results.json` — TTA sweep results

## Expected GPU Time

- SOTA baselines: ~3-5 hours (T4/V100)
- Our cooperative: ~1-2 hours (15 epochs)
- TTA sweep: ~30 min
