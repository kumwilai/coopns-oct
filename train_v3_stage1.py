#!/usr/bin/env python3
"""
Stage 1: Train correctors with FROZEN backbone
Test if correctors can contribute when forced to
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from neuro_symbolic_v2 import (
    NeuroSymbolicDenoiserV3, NeuroSymbolicLossV3, OCTDataset,
    _NUM_THREADS
)
from torch.utils.data import DataLoader

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()

def main():
    print('='*70)
    print('STAGE 1: TRAIN CORRECTORS WITH FROZEN BACKBONE')
    print(f'Using {_NUM_THREADS} CPU threads')
    print('='*70)

    # Configuration
    BATCH_SIZE = 4
    EPOCHS = 20
    TRAIN_SAMPLES = 100
    VAL_SAMPLES = 10
    LR = 1e-4

    # Create model
    print('\nInitializing model...')
    model = NeuroSymbolicDenoiserV3(backbone_type='nafnet', width=64)

    # Load pretrained backbone
    backbone_path = 'outputs/nafnet_pku37/nafnet_best.pth'
    if Path(backbone_path).exists():
        print('Loading pretrained backbone...')
        model.load_pretrained_backbone(backbone_path)

    # FREEZE backbone completely
    print('\n*** FREEZING BACKBONE ***')
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f'Total params: {total:,}')
    print(f'Backbone params: {backbone_params:,} (FROZEN)')
    print(f'Trainable params: {trainable:,} (correctors + lambda)')

    # Load data
    print('\nLoading data...')
    train_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_train.jsonl',
        max_samples=TRAIN_SAMPLES, patch_size=96, is_train=True
    )
    val_dataset = OCTDataset(
        'pku37_oct_dataset/pku37_real_val.jsonl',
        max_samples=VAL_SAMPLES, patch_size=0, is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Optimizer - higher LR for correctors since backbone is frozen
    optimizer = torch.optim.Adam([
        {'params': model.lambda_predictor.parameters(), 'lr': LR * 10},  # Lambda learns fast
        {'params': model.corrector.parameters(), 'lr': LR * 5},          # Correctors
    ])
    
    loss_fn = NeuroSymbolicLossV3()
    
    print(f'\nTraining: {len(train_dataset)} samples')
    print(f'Validation: {len(val_dataset)} samples')
    print(f'LR: lambda={LR*10:.0e}, corrector={LR*5:.0e}')
    
    print('\n' + '#'*70)
    print('# STAGE 1: CORRECTOR-ONLY TRAINING')
    print('#'*70)

    best_score = 0
    
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_lambda_edge = 0
        epoch_lambda_texture = 0
        epoch_lambda_smooth = 0
        epoch_predicates = {f'P{i}': 0 for i in range(1, 7)}
        n_batches = 0
        
        t0 = time.time()
        for noisy, clean in train_loader:
            optimizer.zero_grad()
            
            # Forward
            output, info = model(noisy)
            
            # Loss
            loss, loss_info = loss_fn(output, clean, info, noisy)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Stats
            epoch_loss += loss.item()
            epoch_psnr += compute_psnr(output.detach(), clean)
            epoch_lambda_edge += info['lambda_maps']['edge'].mean().item()
            epoch_lambda_texture += info['lambda_maps']['texture'].mean().item()
            epoch_lambda_smooth += info['lambda_maps']['smooth'].mean().item()
            for k in epoch_predicates:
                epoch_predicates[k] += loss_info['predicates'][k]
            n_batches += 1
        
        train_time = time.time() - t0
        
        # Average
        epoch_loss /= n_batches
        epoch_psnr /= n_batches
        epoch_lambda_edge /= n_batches
        epoch_lambda_texture /= n_batches
        epoch_lambda_smooth /= n_batches
        for k in epoch_predicates:
            epoch_predicates[k] /= n_batches
        
        # Print training stats
        print(f'\n[EPOCH {epoch}]')
        print(f'  Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.2f} dB')
        print(f'  Lambda: edge={epoch_lambda_edge:.3f}, tex={epoch_lambda_texture:.3f}, smooth={epoch_lambda_smooth:.3f}')
        print(f'  Predicates: P1={epoch_predicates["P1"]:.3f}, P2={epoch_predicates["P2"]:.3f}, P3={epoch_predicates["P3"]:.3f}')
        print(f'              P4={epoch_predicates["P4"]:.3f}, P5={epoch_predicates["P5"]:.3f}, P6={epoch_predicates["P6"]:.3f}')
        print(f'  Time: {train_time:.1f}s')
        
        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_psnr_backbone = 0
            val_psnr_corrected = 0
            val_ssim_backbone = 0
            val_ssim_corrected = 0
            val_lambda_edge = 0
            val_predicates = {f'P{i}': 0 for i in range(1, 7)}
            
            with torch.no_grad():
                for noisy, clean in val_loader:
                    # Backbone only
                    backbone_out, _ = model.backbone(noisy)
                    
                    # Full model
                    output, info = model(noisy)
                    _, loss_info = loss_fn(output, clean, info, noisy)
                    
                    # PSNR
                    val_psnr_backbone += compute_psnr(backbone_out, clean)
                    val_psnr_corrected += compute_psnr(output, clean)
                    
                    # Lambda
                    val_lambda_edge += info['lambda_maps']['edge'].mean().item()
                    
                    # Predicates
                    for k in val_predicates:
                        val_predicates[k] += loss_info['predicates'][k]
            
            n = len(val_loader)
            val_psnr_backbone /= n
            val_psnr_corrected /= n
            val_lambda_edge /= n
            for k in val_predicates:
                val_predicates[k] /= n
            
            delta_psnr = val_psnr_corrected - val_psnr_backbone
            
            print(f'\n  [VALIDATION]')
            print(f'  Backbone PSNR: {val_psnr_backbone:.2f} dB')
            print(f'  Corrected PSNR: {val_psnr_corrected:.2f} dB (Δ={delta_psnr:+.3f} dB)')
            print(f'  Lambda edge: {val_lambda_edge:.3f}')
            print(f'  Predicates: P1={val_predicates["P1"]:.3f}, P2={val_predicates["P2"]:.3f}, P3={val_predicates["P3"]:.3f}')
            print(f'              P4={val_predicates["P4"]:.3f}, P5={val_predicates["P5"]:.3f}, P6={val_predicates["P6"]:.3f}')
            
            # Score
            score = sum(val_predicates.values()) / 6
            if score > best_score:
                best_score = score
                print(f'  *** Best model (score={score:.4f}) ***')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'val_metrics': {
                        'psnr_backbone': val_psnr_backbone,
                        'psnr_corrected': val_psnr_corrected,
                        'delta_psnr': delta_psnr,
                        **{f'P{i}': val_predicates[f'P{i}'] for i in range(1, 7)},
                        'lambda_edge': val_lambda_edge,
                    }
                }, 'outputs/nsnd_v3/stage1_best.pth')
    
    print('\n' + '#'*70)
    print('# STAGE 1 COMPLETE')
    print('#'*70)
    
    # Final summary
    print(f'\nBest validation score: {best_score:.4f}')
    print(f'Corrector contribution (PSNR delta): {delta_psnr:+.3f} dB')
    
    if abs(delta_psnr) > 0.05:
        print('\n✓ Correctors ARE contributing when forced!')
    else:
        print('\n⚠️ Correctors still not contributing significantly')


if __name__ == '__main__':
    main()
