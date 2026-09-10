#!/bin/bash
# ============================================================================
# Train CASA with LOW MEMORY (3-4 GB GPU)
# ============================================================================
# Memory optimization strategies:
# 1. Small batch size (2-4)
# 2. Smaller base_channels (64 instead of 96)
# 3. AMP enabled (mixed precision)
# 4. Smaller validation batches
# 5. Gradient accumulation to simulate larger batches
# ============================================================================

SEED=42

echo "========================================================================"
echo "TRAINING CASA FOR LOW MEMORY GPU (3-4 GB)"
echo "========================================================================"
echo "Optimizations:"
echo "  - Batch size: 2 (very small to fit in memory)"
echo "  - Base channels: 64 (smaller model)"
echo "  - Mixed precision training (AMP)"
echo "  - Parallel data loading (num_workers=4)"
echo "========================================================================"

# MEMORY-EFFICIENT TRAINING
python -u adaptive_oct_denoise.py \
    --clean_root meta_clean/ \
    --paired_list train_pairs_universal.txt \
    --val_paired_list val_pairs_universal.txt \
    --num_meta_epochs 20 \
    --finetune_epochs 100 \
    --batch_size 2 \
    --base_channels 64 \
    --finetune_lr_adapter 5e-5 \
    --finetune_lr_backbone 1e-5 \
    --early_stopping_patience 15 \
    --amp \
    --output_dir checkpoints/casa_improved_lowmem \
    --adapter casa \
    --ema \
    --seed $SEED

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
echo "Model saved to: checkpoints/casa_improved_lowmem/"
echo ""
echo "Evaluate:"
echo "  python eval_checkpoint.py \\"
echo "    --checkpoint checkpoints/casa_improved_lowmem/finetuned_ema.pth \\"
echo "    --val_pairs val_pairs_universal.txt \\"
echo "    --adapter casa \\"
echo "    --image_size 64"
echo ""
echo "========================================================================"
