#!/bin/bash
# Quick test of baseline evaluation on small subset
# Use this to verify everything works before running full evaluation

set -e

echo "=========================================="
echo "Quick Baseline Test (100 pairs, 5 epochs)"
echo "=========================================="

# Test with Gaussian noise, 100 pairs, 5 epochs
NOISE="gaussian"
LIMIT=100
SIZE=64
EPOCHS=5

echo ""
echo ">>> Testing Classical Filters (100 pairs)..."
python scripts/eval_baselines_from_pairs.py \
    --pairs_file val_pairs_${NOISE}.txt \
    --image_size ${SIZE} \
    --limit ${LIMIT} \
    --filters lee kuan bilateral \
    --win 5 \
    --use_enl_map \
    --output_dir outputs/test_baselines

echo ""
echo ">>> Testing Deep Learning Baselines (100 pairs, 5 epochs)..."
python scripts/eval_sota_from_pairs.py \
    --train_pairs train_pairs_${NOISE}.txt \
    --val_pairs val_pairs_${NOISE}.txt \
    --size ${SIZE} \
    --epochs ${EPOCHS} \
    --models drunet \
    --out_dir outputs/test_sota

echo ""
echo "=========================================="
echo "Quick Test Complete!"
echo "=========================================="
echo "Check outputs/test_baselines/ and outputs/test_sota/"
echo ""
echo "If everything looks good, run full evaluation:"
echo "  ./eval_all_baselines.sh"
echo ""
