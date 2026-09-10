#!/usr/bin/env python3
"""
Neuro-Symbolic OCT Denoising V3 Training Script
Adaptive Lambda Prediction for TMI Publication
"""

import sys
import torch
import numpy as np
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from neuro_symbolic_v2 import (
    NeuroSymbolicDenoiserV3, NeuroSymbolicLossV3, TrainerV3, OCTDataset,
    _NUM_THREADS, clear_memory
)

def main():
    print('='*70, flush=True)
    print('NEURO-SYMBOLIC OCT DENOISING V3 (ADAPTIVE LAMBDA)', flush=True)
    print(f'Using {_NUM_THREADS} CPU threads', flush=True)
    print('='*70, flush=True)

    # Configuration
    FINETUNE_BACKBONE = True  # Enable backbone fine-tuning for P2/P5 improvement
    BATCH_SIZE = 4
    EPOCHS = 15
    TRAIN_SAMPLES = 100
    VAL_SAMPLES = 10
    NUM_WORKERS = 2
    VAL_EVERY = 3
    PATCH_SIZE = 96

    # Create V3 model
    print('\nInitializing V3 model...', flush=True)
    model = NeuroSymbolicDenoiserV3(backbone_type='nafnet', width=64)

    # Load pretrained backbone
    backbone_path = 'outputs/nafnet_pku37/nafnet_best.pth'
    if Path(backbone_path).exists():
        print('Loading pretrained backbone...', flush=True)
        model.load_pretrained_backbone(backbone_path)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lambda_params = sum(p.numel() for p in model.lambda_predictor.parameters())
    corrector_params = sum(p.numel() for p in model.corrector.parameters())
    print(f'Parameters: {total:,} total, {trainable:,} trainable', flush=True)
    print(f'  Lambda predictor: {lambda_params:,}', flush=True)
    print(f'  Corrector: {corrector_params:,}', flush=True)

    # Load data
    print('\nLoading data...', flush=True)
    train_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_train.jsonl',
        max_samples=TRAIN_SAMPLES,
        patch_size=PATCH_SIZE,
        is_train=True
    )
    val_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_val.jsonl',
        max_samples=VAL_SAMPLES,
        patch_size=0,
        is_train=False
    )
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}', flush=True)

    # Create V3 trainer
    trainer = TrainerV3(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        finetune_backbone=FINETUNE_BACKBONE,
        num_workers=NUM_WORKERS,
    )

    # Train
    trainer.train(epochs=EPOCHS, val_every=VAL_EVERY)


if __name__ == '__main__':
    main()
