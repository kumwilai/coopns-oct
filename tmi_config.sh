#!/bin/bash
# =============================================================================
# TMI Pipeline Configuration
# Source this file in each step script
# =============================================================================

# Device configuration
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    fi
fi

# Output directory - use existing or create new
# Set TMI_OUTPUT_DIR environment variable to use a specific directory
if [ -z "$TMI_OUTPUT_DIR" ]; then
    OUTPUT_BASE="tmi_results"
    # Check for most recent run
    LATEST=$(ls -td ${OUTPUT_BASE}/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        OUTPUT_DIR="${LATEST%/}"
        echo "Using existing output directory: $OUTPUT_DIR"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
        echo "Creating new output directory: $OUTPUT_DIR"
    fi
else
    OUTPUT_DIR="$TMI_OUTPUT_DIR"
    echo "Using specified output directory: $OUTPUT_DIR"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Paths
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
CHECKPOINT_PATH="$CHECKPOINT_DIR/best.pth"
BACKBONE_PATH="outputs/nafnet_analysis_maps_w64/nafnet_best.pth"

# Training parameters
TRAIN_EPOCHS=30
TRAIN_BATCH_SIZE=4
TRAIN_MAX_SAMPLES=500
TRAIN_VAL_SAMPLES=100
TRAIN_LR="1e-4"

# Ablation parameters
ABLATION_EPOCHS=10
ABLATION_MAX_TRAIN=200
ABLATION_MAX_VAL=50

# Evaluation parameters
EVAL_MAX_VAL=100

# Dataset paths
DUKE_TRAIN_JSONL="weights_duke_analysis_maps_train.jsonl"
DUKE_VAL_JSONL="weights_duke_analysis_maps_val.jsonl"
PKU37_VAL_JSONL="pku37_oct_dataset/weights_pku37_analysis_val.jsonl"

# Print configuration
print_config() {
    echo "=============================================="
    echo "TMI PIPELINE CONFIGURATION"
    echo "=============================================="
    echo "Device:          $DEVICE"
    echo "Output Dir:      $OUTPUT_DIR"
    echo "Checkpoint:      $CHECKPOINT_PATH"
    echo "Backbone:        $BACKBONE_PATH"
    echo "=============================================="
}

export DEVICE OUTPUT_DIR CHECKPOINT_DIR CHECKPOINT_PATH BACKBONE_PATH
export TRAIN_EPOCHS TRAIN_BATCH_SIZE TRAIN_MAX_SAMPLES TRAIN_VAL_SAMPLES TRAIN_LR
export ABLATION_EPOCHS ABLATION_MAX_TRAIN ABLATION_MAX_VAL
export EVAL_MAX_VAL
export DUKE_TRAIN_JSONL DUKE_VAL_JSONL PKU37_VAL_JSONL
