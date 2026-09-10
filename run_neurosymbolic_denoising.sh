#!/bin/bash
# =============================================================================
# NEURO-SYMBOLIC OCT DENOISING
# =============================================================================
#
# NOVEL CONTRIBUTIONS FOR PAPER:
#
# 1. FIRST NEURO-SYMBOLIC FRAMEWORK FOR OCT DENOISING
#    - Clear separation of neural (learned) and symbolic (rule-based) components
#    - Interpretable architecture with explicit anatomical knowledge
#
# 2. SELF-SUPERVISED SEGMENTATION (NO GT MASKS NEEDED)
#    - Multi-frame consistency: same scene → same segmentation
#    - Augmentation equivariance: seg(aug(x)) = aug(seg(x))
#    - Pseudo-labels from pretrained model refined by symbolic rules
#
# 3. DIFFERENTIABLE PHYSICS CONSTRAINTS
#    - Beer-Lambert law for tissue attenuation
#    - Fresnel reflections at layer boundaries
#    - Rayleigh distribution for speckle noise model
#
# 4. ANATOMICAL LOGIC LAYER
#    - Hard constraints: layer ordering (ILM < RNFL < ... < RPE)
#    - Soft constraints: physiological thickness bounds
#    - Continuity: boundary smoothness
#    - Intensity: layer-specific reflectivity patterns
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "NEURO-SYMBOLIC OCT DENOISING"
echo "=============================================="
echo ""
echo "Novel Contributions:"
echo "  1. First neuro-symbolic framework for OCT denoising"
echo "  2. Self-supervised segmentation via symbolic consistency"
echo "  3. Differentiable physics (Beer-Lambert, Fresnel)"
echo "  4. Anatomical logic layer with hard/soft rules"
echo "=============================================="

# Device detection
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi
echo "Device: $DEVICE"

# =============================================================================
# CONFIGURATION
# =============================================================================

MODE=${1:-quick}
SELF_SUPERVISED=${2:-false}  # Second argument: true for self-supervised

case $MODE in
    quick)
        echo "Mode: QUICK (testing)"
        EPOCHS=5
        BATCH_SIZE=2
        MAX_TRAIN=100
        MAX_VAL=30
        NUM_REALIZATIONS=2
        LR=1e-4
        ;;
    medium)
        echo "Mode: MEDIUM"
        EPOCHS=30
        BATCH_SIZE=2  # Reduced from 4 for memory
        MAX_TRAIN=500
        MAX_VAL=100
        NUM_REALIZATIONS=2  # Reduced from 3 for memory
        LR=5e-5
        ;;
    full)
        echo "Mode: FULL"
        EPOCHS=100
        BATCH_SIZE=2  # Reduced from 4 for memory
        MAX_TRAIN=""
        MAX_VAL=""
        NUM_REALIZATIONS=2  # Reduced from 3 for memory
        LR=1e-4
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [quick|medium|full] [self_supervised:true/false]"
        exit 1
        ;;
esac

# Self-supervised mode settings
if [ "$SELF_SUPERVISED" = "true" ]; then
    echo ""
    echo ">>> SELF-SUPERVISED MODE (No Clean Reference Needed) <<<"
    echo ""
    SELF_SUPERVISED_FLAG="--self_supervised"
    # Increase consistency weight for self-supervised
    LAMBDA_CONSISTENCY=1.0
else
    SELF_SUPERVISED_FLAG=""
fi

# Resolution (must match physics checkpoint)
PATCH_SIZE=256

# Data
TRAIN_JSONL="combined_train.jsonl"
VAL_JSONL="combined_val.jsonl"

# Checkpoints
PHYSICS_CKPT="outputs/physics_v3_dice_v2/stage3_256/best_model.pt"
NAFNET_CKPT="outputs/nafnet_calibrated/nafnet_best.pth"

# Verify checkpoints
if [ ! -f "$PHYSICS_CKPT" ]; then
    echo "WARNING: Physics checkpoint not found: $PHYSICS_CKPT"
    PHYSICS_CKPT=""
fi
if [ ! -f "$NAFNET_CKPT" ]; then
    echo "WARNING: NAFNet checkpoint not found: $NAFNET_CKPT"
    NAFNET_CKPT=""
fi

# Model architecture
HIDDEN_CHANNELS=48
NAFNET_WIDTH=64           # Must match checkpoint (nafnet_best.pth has width=64)
BLEND_SIGMA=7.0

# Loss weights (SYMBOLIC COMPONENTS)
LAMBDA_L1=1.0
LAMBDA_ANATOMICAL=0.5      # Anatomical logic constraints
LAMBDA_PHYSICS=0.3         # Physics (Beer-Lambert, Fresnel)
LAMBDA_CONSISTENCY=0.5     # Multi-frame consistency
LAMBDA_SPECKLE=0.1         # Rayleigh distribution matching
LAMBDA_LOGIC=0.3           # TRUE NEURO-SYMBOLIC: Differentiable logic layer

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/neurosymbolic_${MODE}_${TIMESTAMP}"

echo ""
echo "=============================================="
echo "CONFIGURATION"
echo "=============================================="
echo "  Epochs:          $EPOCHS"
echo "  Batch size:      $BATCH_SIZE"
echo "  Patch size:      ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  Learning rate:   $LR"
echo "  Max train:       ${MAX_TRAIN:-all}"
echo "  Max val:         ${MAX_VAL:-all}"
echo ""
echo "  Noise realizations: $NUM_REALIZATIONS (for consistency)"
echo ""
echo "  Physics checkpoint: $PHYSICS_CKPT"
echo "  NAFNet checkpoint:  $NAFNET_CKPT"
echo ""
echo "  SYMBOLIC LOSS WEIGHTS:"
echo "    Anatomical:  $LAMBDA_ANATOMICAL (ordering, thickness)"
echo "    Physics:     $LAMBDA_PHYSICS (Beer-Lambert, Fresnel)"
echo "    Consistency: $LAMBDA_CONSISTENCY (multi-frame)"
echo "    Speckle:     $LAMBDA_SPECKLE (Rayleigh dist)"
echo "    Logic:       $LAMBDA_LOGIC (TRUE NEURO-SYMBOLIC: FOL predicates)"
echo ""
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# RUN TRAINING
# =============================================================================
echo ""
echo "Starting neuro-symbolic training..."

TRAIN_CMD="python train_neurosymbolic_denoising.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --hidden_channels $HIDDEN_CHANNELS \
    --nafnet_width $NAFNET_WIDTH \
    --blend_sigma $BLEND_SIGMA \
    --num_realizations $NUM_REALIZATIONS \
    --lambda_l1 $LAMBDA_L1 \
    --lambda_anatomical $LAMBDA_ANATOMICAL \
    --lambda_physics $LAMBDA_PHYSICS \
    --lambda_consistency $LAMBDA_CONSISTENCY \
    --lambda_speckle $LAMBDA_SPECKLE \
    --lambda_logic $LAMBDA_LOGIC \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR"

# Add checkpoints
if [ -n "$PHYSICS_CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --physics_ckpt $PHYSICS_CKPT"
fi
if [ -n "$NAFNET_CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --nafnet_ckpt $NAFNET_CKPT"
fi

# Add sample limits
if [ -n "$MAX_TRAIN" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_train $MAX_TRAIN"
fi
if [ -n "$MAX_VAL" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_val $MAX_VAL"
fi

# Add self-supervised flag if enabled
if [ -n "$SELF_SUPERVISED_FLAG" ]; then
    TRAIN_CMD="$TRAIN_CMD $SELF_SUPERVISED_FLAG"
fi

# Run
$TRAIN_CMD 2>&1 | tee "$OUTPUT_DIR/training_log.txt"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""
echo "PAPER CONTRIBUTIONS:"
echo "  1. First neuro-symbolic OCT denoising framework"
echo "  2. Self-supervised segmentation (no GT masks)"
echo "  3. Differentiable physics constraints"
echo "  4. Anatomical logic layer"
echo ""
echo "Key files:"
echo "  - best_model.pth"
echo "  - training_log.txt"
echo "=============================================="
