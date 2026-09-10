#!/bin/bash
# Comprehensive evaluation of all baselines on full dataset
# Uses the new scripts that support pairs files

set -e

NOISE_TYPES=("gaussian" "heavy_gamma" "moderate_gamma")
IMAGE_SIZE=64
EPOCHS=50

echo "=========================================="
echo "Baseline Evaluation on Full Dataset"
echo "=========================================="
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Noise types: ${NOISE_TYPES[@]}"
echo "DL training epochs: ${EPOCHS}"
echo ""

# Create output directories
mkdir -p outputs/baselines
mkdir -p outputs/sota

for noise in "${NOISE_TYPES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing Noise Type: ${noise}"
    echo "=========================================="

    # 1. Classical Filters (2700 validation pairs, no training needed)
    echo ""
    echo ">>> Step 1: Evaluating Classical Filters..."
    echo "    Processing 2700 validation pairs from val_pairs_${noise}.txt"

    python scripts/eval_baselines_from_pairs.py \
        --pairs_file val_pairs_${noise}.txt \
        --image_size ${IMAGE_SIZE} \
        --filters lee kuan frost bilateral guided \
        --win 5 \
        --use_enl_map \
        --output_dir outputs/baselines/${noise} \
        2>&1 | tee outputs/baselines/${noise}_log.txt

    echo "    ✓ Classical filters done!"
    echo "    Results: outputs/baselines/${noise}/val_pairs_${noise}_results.csv"

    # 2. Deep Learning Baselines (train on 12600, eval on 2700)
    echo ""
    echo ">>> Step 2: Training Deep Learning Baselines..."
    echo "    Training on 12600 pairs from train_pairs_${noise}.txt"
    echo "    Validating on 2700 pairs from val_pairs_${noise}.txt"

    python scripts/eval_sota_from_pairs.py \
        --train_pairs train_pairs_${noise}.txt \
        --val_pairs val_pairs_${noise}.txt \
        --size ${IMAGE_SIZE} \
        --epochs ${EPOCHS} \
        --models drunet swinir noise2void speckle2speckle \
        --out_dir outputs/sota/${noise} \
        2>&1 | tee outputs/sota/${noise}_log.txt

    echo "    ✓ Deep learning baselines done!"
    echo "    Results: outputs/sota/${noise}/summary.json"

    echo ""
    echo "✓✓✓ Completed: ${noise}"
done

echo ""
echo "=========================================="
echo "All Baselines Evaluated Successfully!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "----------------"
for noise in "${NOISE_TYPES[@]}"; do
    echo ""
    echo "Noise Type: ${noise}"
    echo "  Classical Filters:"
    echo "    - CSV: outputs/baselines/${noise}/val_pairs_${noise}_results.csv"
    echo "    - Log: outputs/baselines/${noise}_log.txt"
    echo "  Deep Learning:"
    echo "    - Summary: outputs/sota/${noise}/summary.json"
    echo "    - Models: outputs/sota/${noise}/*.pth"
    echo "    - Log: outputs/sota/${noise}_log.txt"
done

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Compare with your CASA model results"
echo "2. Generate comparison tables and figures"
echo "3. Run: python compare_results.py --help"
echo ""
