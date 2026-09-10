#!/bin/bash
# =============================================================================
# Joint Physics-Enhanced Boundary Detection + CUAP-OCT Denoising V2
# =============================================================================
#
# KEY IMPROVEMENTS:
# 1. Uses pretrained NAFNet checkpoint (outputs/nafnet_calibrated)
# 2. Soft masks with gradual boundary transitions (~7px blending)
# 3. Multi-frame prediction averaging for stable PKU37 segmentation
# 4. Layer-specific denoising with clinical importance weighting
#
# PKU37 COMPATIBILITY:
# - Soft masks avoid hard boundary artifacts
# - Broad region processing (not pixel-precise edges)
# - Clinical weights prioritize RNFL/IS-OS
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "JOINT PHYSICS DENOISING V2"
echo "=============================================="
echo ""
echo "Key Features:"
echo "  - Pretrained NAFNet backbone"
echo "  - Soft mask blending (~7px transitions)"
echo "  - Physics-enhanced boundary detection"
echo "  - Clinical importance weighting"
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

case $MODE in
    quick)
        echo "Mode: QUICK"
        EPOCHS=5
        BATCH_SIZE=2
        MAX_TRAIN=100
        MAX_VAL=30
        PATCH_SIZE=256           # Match physics checkpoint resolution
        LR=1e-4
        ;;
    medium)
        echo "Mode: MEDIUM"
        EPOCHS=30
        BATCH_SIZE=4
        MAX_TRAIN=500
        MAX_VAL=100
        PATCH_SIZE=256           # Match physics checkpoint resolution
        LR=5e-5
        ;;
    full)
        echo "Mode: FULL"
        EPOCHS=100
        BATCH_SIZE=4
        MAX_TRAIN=""
        MAX_VAL=""
        PATCH_SIZE=256           # Match physics checkpoint resolution
        LR=1e-4
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [quick|medium|full]"
        exit 1
        ;;
esac

# Data paths
TRAIN_JSONL="combined_train.jsonl"
VAL_JSONL="combined_val.jsonl"

# IMPORTANT: Use pretrained NAFNet checkpoint
NAFNET_CKPT="outputs/nafnet_calibrated/nafnet_best.pth"
if [ ! -f "$NAFNET_CKPT" ]; then
    echo "WARNING: NAFNet checkpoint not found: $NAFNET_CKPT"
    echo "Training will use random initialization"
    NAFNET_CKPT=""
fi

# IMPORTANT: Use pretrained physics/segmentation checkpoint (256x256)
PHYSICS_CKPT="outputs/physics_v3_dice_v2/stage3_256/best_model.pt"
if [ ! -f "$PHYSICS_CKPT" ]; then
    echo "WARNING: Physics checkpoint not found: $PHYSICS_CKPT"
    echo "Segmentation will use random initialization"
    PHYSICS_CKPT=""
fi

# Model architecture
HIDDEN_CHANNELS=48        # Physics boundary model
NAFNET_WIDTH=64           # NAFNet width (must match checkpoint)
BLEND_SIGMA=7.0           # Soft boundary blending (pixels)

# Loss weights
LAMBDA_L1=1.0
LAMBDA_DICE=0.5
LAMBDA_BOUNDARY=0.3
LAMBDA_ORDERING=0.1

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/joint_physics_v2_${MODE}_${TIMESTAMP}"

echo ""
echo "=============================================="
echo "CONFIGURATION"
echo "=============================================="
echo "  Epochs:          $EPOCHS"
echo "  Batch size:      $BATCH_SIZE"
echo "  Patch size:      $PATCH_SIZE"
echo "  Learning rate:   $LR"
echo "  Max train:       ${MAX_TRAIN:-all}"
echo "  Max val:         ${MAX_VAL:-all}"
echo ""
echo "  Physics checkpoint: $PHYSICS_CKPT"
echo "  NAFNet checkpoint:  $NAFNET_CKPT"
echo "  Blend sigma:        ${BLEND_SIGMA}px (soft boundaries)"
echo ""
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# RUN TRAINING
# =============================================================================
echo ""
echo "Starting training..."

TRAIN_CMD="python train_joint_physics_denoising_v2.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --hidden_channels $HIDDEN_CHANNELS \
    --nafnet_width $NAFNET_WIDTH \
    --blend_sigma $BLEND_SIGMA \
    --lambda_l1 $LAMBDA_L1 \
    --lambda_dice $LAMBDA_DICE \
    --lambda_boundary $LAMBDA_BOUNDARY \
    --lambda_ordering $LAMBDA_ORDERING \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR"

# Add checkpoints if available
if [ -n "$NAFNET_CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --nafnet_ckpt $NAFNET_CKPT"
fi
if [ -n "$PHYSICS_CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --physics_ckpt $PHYSICS_CKPT"
fi

# Add sample limits
if [ -n "$MAX_TRAIN" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_train $MAX_TRAIN"
fi
if [ -n "$MAX_VAL" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_val $MAX_VAL"
fi

# Run
$TRAIN_CMD 2>&1 | tee "$OUTPUT_DIR/training_log.txt"

# =============================================================================
# TEST ON PKU37
# =============================================================================
PKU37_DIR="pku37_oct_dataset/PKU37_OCT_Denoising/PKU37_OCT_Denoising"
if [ -d "$PKU37_DIR" ] && [ -f "$OUTPUT_DIR/best_psnr.pth" ]; then
    echo ""
    echo "=============================================="
    echo "EVALUATING ON PKU37 (Real Data)"
    echo "=============================================="

    python3 << EOF
import torch
import numpy as np
from PIL import Image
import glob
import os

# Metrics
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Load model
from train_joint_physics_denoising_v2 import JointPhysicsDenoiserV2, MultiFramePredictor

device = torch.device('$DEVICE')
model = JointPhysicsDenoiserV2(
    hidden_channels=$HIDDEN_CHANNELS,
    nafnet_width=$NAFNET_WIDTH,
    blend_sigma=$BLEND_SIGMA,
).to(device)

ckpt = torch.load('$OUTPUT_DIR/best_psnr.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded model from epoch {ckpt['epoch']}, PSNR={ckpt['psnr']:.2f}")

# Wrap for multi-frame averaging
multi_frame_model = MultiFramePredictor(model)

clean_dir = "$PKU37_DIR/clean"
noisy_dir = "$PKU37_DIR/noisy"
clean_files = sorted(glob.glob(f"{clean_dir}/*.tif"))[:5]

print(f"\\nTesting on {len(clean_files)} PKU37 scenes...")
print("Using multi-frame averaging (3 frames) for stable segmentation")
print("-" * 60)

results = {'psnr_gain': [], 'ssim_gain': []}

for clean_path in clean_files:
    scene_id = os.path.basename(clean_path).replace('.tif', '')

    # Load clean reference
    clean = np.array(Image.open(clean_path).convert('L')).astype(np.float32) / 255.0

    # Load multiple noisy frames for averaging
    noisy_frames = []
    for frame_idx in range(1, 4):  # Frames 01, 02, 03
        noisy_path = f"{noisy_dir}/{scene_id}{frame_idx:02d}.tif"
        if os.path.exists(noisy_path):
            noisy = np.array(Image.open(noisy_path).convert('L')).astype(np.float32) / 255.0
            noisy_frames.append(noisy)

    if len(noisy_frames) < 1:
        continue

    # Resize for model (use same size as training)
    H = $PATCH_SIZE
    clean_r = np.array(Image.fromarray((clean * 255).astype(np.uint8)).resize((H, H))) / 255.0
    noisy_r = [np.array(Image.fromarray((n * 255).astype(np.uint8)).resize((H, H))) / 255.0
               for n in noisy_frames]

    # Convert to tensors
    noisy_tensors = [torch.from_numpy(n).float().unsqueeze(0).unsqueeze(0).to(device)
                     for n in noisy_r]

    # Multi-frame averaged prediction (KEY FOR STABLE SEGMENTATION)
    with torch.no_grad():
        outputs = multi_frame_model.predict_averaged(noisy_tensors, return_std=True)
        denoised = outputs['denoised'][0, 0].cpu().numpy()

    # Compute metrics
    noisy_avg = np.mean(noisy_r, axis=0)
    psnr_before = psnr(clean_r, noisy_avg, data_range=1.0)
    psnr_after = psnr(clean_r, denoised, data_range=1.0)
    ssim_before = ssim(clean_r, noisy_avg, data_range=1.0)
    ssim_after = ssim(clean_r, denoised, data_range=1.0)

    results['psnr_gain'].append(psnr_after - psnr_before)
    results['ssim_gain'].append(ssim_after - ssim_before)

    print(f"Scene {scene_id}: PSNR {psnr_before:.2f} -> {psnr_after:.2f} (+{psnr_after-psnr_before:+.2f}), "
          f"SSIM {ssim_before:.3f} -> {ssim_after:.3f}")

print("-" * 60)
print(f"Average PSNR gain: {np.mean(results['psnr_gain']):+.2f} dB")
print(f"Average SSIM gain: {np.mean(results['ssim_gain']):+.3f}")
print("=" * 60)
EOF
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""
echo "Models:"
echo "  - best_psnr.pth"
echo "  - best_dice.pth"
echo ""
echo "Usage:"
echo "  from train_joint_physics_denoising_v2 import JointPhysicsDenoiserV2"
echo "  model = JointPhysicsDenoiserV2(nafnet_ckpt='$NAFNET_CKPT')"
echo "  model.load_state_dict(torch.load('$OUTPUT_DIR/best_psnr.pth')['model_state_dict'])"
echo "=============================================="
