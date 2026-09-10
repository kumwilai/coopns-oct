#!/bin/bash
# ============================================================================
# Train CASA with ULTRA LOW MEMORY (2-3 GB GPU)
# ============================================================================
# EXTREME memory optimization for very limited GPU
# ============================================================================

SEED=42

echo "========================================================================"
echo "TRAINING CASA FOR ULTRA LOW MEMORY GPU (2-3 GB)"
echo "========================================================================"
echo "EXTREME Optimizations:"
echo "  - Batch size: 1 (minimum possible)"
echo "  - Base channels: 48 (very small model)"
echo "  - Mixed precision training (AMP)"
echo "  - Small validation batches"
echo "========================================================================"

# ULTRA MEMORY-EFFICIENT TRAINING
python -u adaptive_oct_denoise.py \
    --clean_root meta_clean/ \
    --paired_list train_pairs_universal.txt \
    --val_paired_list val_pairs_universal.txt \
    --num_meta_epochs 15 \
    --finetune_epochs 80 \
    --batch_size 1 \
    --base_channels 48 \
    --finetune_lr_adapter 5e-5 \
    --finetune_lr_backbone 1e-5 \
    --early_stopping_patience 12 \
    --amp \
    --output_dir checkpoints/casa_improved_ultralowmem \
    --adapter casa \
    --ema \
    --seed $SEED

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Model saved to: checkpoints/casa_improved_ultralowmem/"
echo "========================================================================"
