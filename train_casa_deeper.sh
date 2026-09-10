#!/bin/bash
# ============================================================================
# Train DEEPER CASA (6 blocks instead of 4) to match SwinIRLite depth
# ============================================================================
# Architecture: 3 encoder levels + 3 decoder levels (6 blocks total)
# Memory: Optimized for 3-4 GB GPU
# Goal: Beat SwinIR's 28.84 dB by matching its depth advantage
# ============================================================================

SEED=42

echo "========================================================================"
echo "TRAINING DEEPER CASA (6-block U-Net to beat SwinIR)"
echo "========================================================================"
echo "Architecture Changes:"
echo "  - Old: 2 encoder + 2 decoder = 4 blocks"
echo "  - New: 3 encoder + 3 decoder = 6 blocks (matches SwinIRLite)"
echo "  - Channels: 64 -> 128 -> 256 (deeper feature hierarchy)"
echo ""
echo "Memory Optimizations (3-4 GB GPU):"
echo "  - Batch size: 2"
echo "  - Base channels: 64"
echo "  - Mixed precision (AMP)"
echo "  - Parallel data loading"
echo "========================================================================"

# TRAIN DEEPER CASA
python -u adaptive_oct_denoise.py \
    --clean_root meta_clean/ \
    --paired_list train_pairs_universal.txt \
    --val_paired_list val_pairs_universal.txt \
    --num_meta_epochs 15 \
    --finetune_epochs 100 \
    --batch_size 2 \
    --base_channels 64 \
    --finetune_lr_adapter 5e-5 \
    --finetune_lr_backbone 1e-5 \
    --early_stopping_patience 12 \
    --amp \
    --output_dir checkpoints/casa_deeper \
    --adapter casa \
    --ema \
    --seed $SEED

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
echo "Model saved to: checkpoints/casa_deeper/"
echo ""
echo "Expected improvements:"
echo "  - SwinIRLite (6 blocks): 28.84 dB"
echo "  - Old CASA (4 blocks):   28.11 dB (-0.73 dB)"
echo "  - New CASA (6 blocks):   ~28.8-29.0 dB (target)"
echo ""
echo "========================================================================"
echo "Evaluate with:"
echo "  python eval_checkpoint.py \\"
echo "    --checkpoint checkpoints/casa_deeper/finetuned_ema.pth \\"
echo "    --val_pairs val_pairs_universal.txt \\"
echo "    --adapter casa \\"
echo "    --image_size 64"
echo ""
echo "Compare with SwinIR:"
echo "  python compare_available_methods.py"
echo "========================================================================"
