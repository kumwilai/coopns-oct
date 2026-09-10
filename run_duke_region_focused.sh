#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PHASE1_CKPT="checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth"
PHASE2_CKPT="checkpoints/multitask_hybrid_nsnd_lambda0p008to0p003_cosine_best.pth"
PHASE2B_CKPT="checkpoints/multitask_hybrid_nsnd_lambda0p006to0p002_cosine_best.pth"
FINAL_CKPT="$PHASE2B_CKPT"
PHASE1_RESUME_CKPT="${PHASE1_RESUME_CKPT:-}"
PHASE2_RESUME_CKPT=""
PHASE2B_RESUME_CKPT=""
TRAIN_SAMPLES=1000
VAL_SAMPLES=100
TEST_SAMPLES=400
TEST_PAIRS_FILE="test_pairs_duke_analysis_maps.txt"
TEST_PAIRS_SUBSET="outputs/test_pairs_duke_analysis_maps_${TEST_SAMPLES}.txt"
HEAD_QUALITY_WEIGHT=1.0          # Strong per-head supervision
HEAD_DIVERSITY_WEIGHT=0.5         # Force head specialization (increased from 0.2)
HEAD_DIVERSITY_WEIGHT_FT=0.5      # Keep diversity in fine-tuning
HEAD_CONSISTENCY_WEIGHT=0.05      # Light regularization to prevent divergence
ROUTING_LOSS_WEIGHT=0.05

echo "================================================================================"
echo "REGION-FOCUSED TRAINING FOR OCT DENOISING (TMI)"
echo "================================================================================"
echo ""
echo "Key strategy:"
echo "  1. Per-pixel noise maps → Guide region-adaptive denoising"
echo "  2. Balance noise map accuracy (30%) vs denoising quality (70%)"
echo "  3. Regions are CRITICAL: inner/outer retina have different noise"
echo "  4. Show per-pixel maps improve region-specific PSNR"
echo ""
echo "Novel contributions:"
echo "  - Per-pixel noise type & level estimation (interpretability)"
echo "  - Region-adaptive denoising guided by noise maps"
echo "  - Multi-head architecture specialized per noise type"
echo ""
echo "================================================================================"
echo ""

# Two-phase training optimized for region-adaptive denoising

echo "PHASE 1: Noise Map Pre-training (Teach spatial noise estimation)"
echo "-------------------------------------------------------------------------------"
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --weights_jsonl_train weights_duke_analysis_maps_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_maps_val.jsonl \
  --max_samples "$TRAIN_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 10 \
  --early_stopping_patience 0 \
  --lr 3e-4 \
  --analyzer_lr 2e-4 \
  --freeze_analyzer_epochs 0 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_residual \
  --shared_trunk_width 32 \
  --shared_adapter_channels 96 \
  --shared_adapter_hidden 64 \
  --residual_blend_init 0.60 \
  --use_joint_signal_expert \
  --joint_expert_channels 96 \
  --joint_mix_init 0.15 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --use_spatial_weights \
  --spatial_feature_channels 64 \
  --spatial_hidden_channels 32 \
  --noise_map_loss_weight 3.0 \
  --noise_map_loss_weights 3,1,3,1 \
  --noise_map_stage_epochs 6 \
  --noise_map_stage_only \
  --use_region_weights \
  --region_min_band_frac 0.15 \
  --region_smooth_ksize 9 \
  --region_strength_mode residual \
  --log_region_psnr \
  --log_roi_psnr \
  --roi_center_frac 0.4 \
  --metrics_json outputs/duke_metrics_phase1.jsonl \
  --lambda_interp_start 0.00 \
  --lambda_interp_end 0.00 \
  --lambda_interp_schedule constant \
  --param_reg_weight 0.03 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --mix_gate_reg_weight 0 \
  --consistency_weight 0 \
  --logic_reg_weight 0 \
  --composition_loss_weight 0 \
  --composition_consistency_weight 0 \
  --residual_consistency_weight 0 \
  --noise_cycle_weight 0 \
  --speckle_cycle_weight 0 \
  ${PHASE1_RESUME_CKPT:+--resume_ckpt "$PHASE1_RESUME_CKPT"} \
  --seed 42

echo ""
echo "PHASE 2: Region-Adaptive Denoising (Balance noise maps + denoising)"
echo "-------------------------------------------------------------------------------"
if [[ -f "$PHASE1_CKPT" ]]; then
  PHASE2_RESUME_CKPT="$PHASE1_CKPT"
else
  echo "⚠ Missing Phase 1 checkpoint: $PHASE1_CKPT (Phase 2 will start from scratch)"
fi
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --weights_jsonl_train weights_duke_analysis_maps_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_maps_val.jsonl \
  --max_samples "$TRAIN_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 50 \
  --early_stopping_patience 10 \
  --lr 5e-4 \
  --analyzer_lr 2e-4 \
  --freeze_analyzer_epochs 0 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_residual \
  --shared_trunk_width 32 \
  --shared_adapter_channels 96 \
  --shared_adapter_hidden 64 \
  --residual_blend_init 0.60 \
  --use_joint_signal_expert \
  --joint_expert_channels 96 \
  --joint_mix_init 0.15 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --use_spatial_weights \
  --spatial_feature_channels 64 \
  --spatial_hidden_channels 32 \
  --noise_map_loss_weight 0.15 \
  --noise_map_loss_weights 3,1,3,1 \
  --noise_map_stage_epochs 0 \
  --use_region_weights \
  --region_min_band_frac 0.15 \
  --region_smooth_ksize 9 \
  --region_strength_mode residual \
  --log_region_psnr \
  --log_roi_psnr \
  --roi_center_frac 0.4 \
  --metrics_json outputs/duke_metrics_phase2_region.jsonl \
  --lambda_interp_start 0.008 \
  --lambda_interp_end 0.003 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 0 \
  --routing_loss_weight "$ROUTING_LOSS_WEIGHT" \
  --param_reg_weight 0.03 \
  --param_reg_warmup_epochs 0 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --mix_gate_reg_weight 0 \
  --consistency_weight 0 \
  --logic_reg_weight 0 \
  --composition_loss_weight 0 \
  --composition_consistency_weight 0 \
  --residual_consistency_weight 0 \
  --noise_cycle_weight 0 \
  --speckle_cycle_weight 0 \
  --head_quality_weight "$HEAD_QUALITY_WEIGHT" \
  --head_diversity_weight "$HEAD_DIVERSITY_WEIGHT" \
  --head_consistency_weight "$HEAD_CONSISTENCY_WEIGHT" \
  ${PHASE2_RESUME_CKPT:+--resume_ckpt "$PHASE2_RESUME_CKPT"} \
  --seed 42

echo ""
echo "PHASE 2B: Fine-tune with reduced map loss (push PSNR/SSIM)"
echo "-------------------------------------------------------------------------------"
if [[ -f "$PHASE2_CKPT" ]]; then
  PHASE2B_RESUME_CKPT="$PHASE2_CKPT"
else
  echo "⚠ Missing Phase 2 checkpoint: $PHASE2_CKPT (Phase 2B will start from scratch)"
fi
python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --weights_jsonl_train weights_duke_analysis_maps_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_maps_val.jsonl \
  --max_samples "$TRAIN_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --batch_size 4 \
  --epochs 20 \
  --early_stopping_patience 6 \
  --lr 3e-4 \
  --analyzer_lr 1e-4 \
  --freeze_analyzer_epochs 0 \
  --base_nafnet_type full \
  --base_nafnet_width 64 \
  --base_enc_blk_nums 2 2 2 \
  --base_dec_blk_nums 2 2 2 \
  --base_middle_blk_num 2 \
  --base_ckpt outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth \
  --shared_residual \
  --shared_trunk_width 32 \
  --shared_adapter_channels 96 \
  --shared_adapter_hidden 64 \
  --residual_blend_init 0.60 \
  --use_joint_signal_expert \
  --joint_expert_channels 96 \
  --joint_mix_init 0.15 \
  --ns_use_neural_predicates \
  --ns_use_neural_weights \
  --use_spatial_weights \
  --spatial_feature_channels 64 \
  --spatial_hidden_channels 32 \
  --noise_map_loss_weight 0.1 \
  --noise_map_loss_weights 3,1,3,1 \
  --noise_map_stage_epochs 0 \
  --use_region_weights \
  --region_min_band_frac 0.15 \
  --region_smooth_ksize 9 \
  --region_strength_mode residual \
  --log_region_psnr \
  --log_roi_psnr \
  --roi_center_frac 0.4 \
  --metrics_json outputs/duke_metrics_phase2b_region.jsonl \
  --lambda_interp_start 0.006 \
  --lambda_interp_end 0.002 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 0 \
  --routing_loss_weight "$ROUTING_LOSS_WEIGHT" \
  --param_reg_weight 0.03 \
  --param_reg_warmup_epochs 0 \
  --use_log_domain_analyzer \
  --log_head_usage \
  --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
  --mix_gate_reg_weight 0 \
  --consistency_weight 0 \
  --logic_reg_weight 0 \
  --composition_loss_weight 0 \
  --composition_consistency_weight 0 \
  --residual_consistency_weight 0 \
  --noise_cycle_weight 0 \
  --speckle_cycle_weight 0 \
  --head_quality_weight "$HEAD_QUALITY_WEIGHT" \
  --head_diversity_weight "$HEAD_DIVERSITY_WEIGHT_FT" \
  --head_consistency_weight "$HEAD_CONSISTENCY_WEIGHT" \
  ${PHASE2B_RESUME_CKPT:+--resume_ckpt "$PHASE2B_RESUME_CKPT"} \
  --seed 42

echo ""
echo "PHASE 3: Evaluation (PSNR/SSIM + region/ROI + noise-map accuracy)"
echo "-------------------------------------------------------------------------------"
if [[ -f "$FINAL_CKPT" ]]; then
  mkdir -p outputs
  awk 'NF && $1 !~ /^#/' "$TEST_PAIRS_FILE" | head -n "$TEST_SAMPLES" > "$TEST_PAIRS_SUBSET"

  python -u nsnd_oct/scripts/evaluate_nsnd_fixed_pairs.py \
    --checkpoint "$FINAL_CKPT" \
    --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
    --pairs "$TEST_PAIRS_SUBSET" \
    --log_region_psnr \
    --log_roi_psnr \
    --roi_center_frac 0.4 \
    --out_json outputs/duke_eval_region.json \
    --device cpu

  python -u nsnd_oct/scripts/eval_noise_maps.py \
    --ckpt "$FINAL_CKPT" \
    --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
    --pairs "$TEST_PAIRS_SUBSET" \
    --weights_jsonl weights_duke_analysis_maps_test.jsonl \
    --batch_size 4 \
    --crop_size 64 \
    --max_samples "$TEST_SAMPLES" \
    --device cpu \
    --save_json outputs/duke_noise_map_eval.json

  python -u nsnd_oct/scripts/eval_noise_maps.py \
    --ckpt "$FINAL_CKPT" \
    --hybrid_analyzer_ckpt checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth \
    --data_root /home/kumwilai/OCT/oct_tmi \
    --split val \
    --max_samples "$TEST_SAMPLES" \
    --batch_size 4 \
    --alpha_vec 2.0,0.6,2.0,0.6 \
    --device cpu \
    --save_json outputs/oct_tmi_noise_map_eval_biased.json

  TEST_PAIRS_PATH="$TEST_PAIRS_SUBSET" python -u - <<'PY'
import sys
from pathlib import Path
import json
import numpy as np
import torch
import os
from PIL import Image

root = Path.cwd()
sys.path.insert(0, str(root / "nsnd_oct"))
sys.path.insert(0, str(root))

from nsnd.models.nafnet import NAFNet
from nsnd.utils.metrics import compute_psnr, compute_ssim
from scripts.train_hybrid_nsnd_multitask import (
    _region_masks_from_noisy,
    _roi_mask_from_noisy,
    compute_psnr_masked,
    compute_ssim_masked,
)

def denoise_image_patches(model, noisy_img, patch_size=64, stride=48, device="cpu"):
    h, w = noisy_img.shape
    if h <= patch_size and w <= patch_size:
        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)
        return denoised_tensor.squeeze().cpu().numpy()

    denoised = np.zeros_like(noisy_img)
    weights = np.zeros_like(noisy_img)

    model.eval()
    with torch.no_grad():
        for top in range(0, h - patch_size + 1, stride):
            for left in range(0, w - patch_size + 1, stride):
                patch = noisy_img[top:top + patch_size, left:left + patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = model(patch_tensor).squeeze().cpu().numpy()
                denoised[top:top + patch_size, left:left + patch_size] += denoised_patch
                weights[top:top + patch_size, left:left + patch_size] += 1.0

        if (w - patch_size) % stride != 0:
            left = w - patch_size
            for top in range(0, h - patch_size + 1, stride):
                patch = noisy_img[top:top + patch_size, left:left + patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = model(patch_tensor).squeeze().cpu().numpy()
                denoised[top:top + patch_size, left:left + patch_size] += denoised_patch
                weights[top:top + patch_size, left:left + patch_size] += 1.0

        if (h - patch_size) % stride != 0:
            top = h - patch_size
            for left in range(0, w - patch_size + 1, stride):
                patch = noisy_img[top:top + patch_size, left:left + patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = model(patch_tensor).squeeze().cpu().numpy()
                denoised[top:top + patch_size, left:left + patch_size] += denoised_patch
                weights[top:top + patch_size, left:left + patch_size] += 1.0

        if (h - patch_size) % stride != 0 and (w - patch_size) % stride != 0:
            top = h - patch_size
            left = w - patch_size
            patch = noisy_img[top:top + patch_size, left:left + patch_size]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            denoised_patch = model(patch_tensor).squeeze().cpu().numpy()
            denoised[top:top + patch_size, left:left + patch_size] += denoised_patch
            weights[top:top + patch_size, left:left + patch_size] += 1.0

    return denoised / np.maximum(weights, 1.0)


pairs_file = Path(os.environ.get("TEST_PAIRS_PATH", "test_pairs_duke_analysis_maps.txt"))
ckpt_path = root / "outputs/baselines_duke_analysis/nafnet_w64/nafnet_best.pth"
device = "cpu"

model = NAFNet(
    img_channel=1,
    width=64,
    middle_blk_num=2,
    enc_blk_nums=[2, 2, 2],
    dec_blk_nums=[2, 2, 2],
).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
model.load_state_dict(state, strict=False)
model.eval()

psnrs = []
ssims = []
inner_psnrs = []
outer_psnrs = []
inner_ssims = []
outer_ssims = []
roi_psnrs = []
roi_ssims = []

with open(pairs_file, "r", encoding="utf-8") as f:
    pairs = [line.strip().split("\t") for line in f if line.strip() and not line.startswith("#")]

for noisy_path, clean_path in pairs:
    noisy = np.array(Image.open(noisy_path).convert("L"), dtype=np.float32) / 255.0
    clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
    denoised = denoise_image_patches(model, noisy, patch_size=64, stride=48, device=device)

    noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float()
    clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()
    denoised_t = torch.from_numpy(denoised).unsqueeze(0).unsqueeze(0).float()

    psnrs.append(float(compute_psnr(denoised_t, clean_t)))
    ssims.append(float(compute_ssim(denoised_t, clean_t)))

    inner_mask, outer_mask = _region_masks_from_noisy(noisy_t, min_band_frac=0.15, smooth_ksize=9)
    roi_mask = _roi_mask_from_noisy(noisy_t, roi_center_frac=0.4)

    inner_psnrs.append(float(compute_psnr_masked(denoised_t, clean_t, inner_mask)))
    outer_psnrs.append(float(compute_psnr_masked(denoised_t, clean_t, outer_mask)))
    inner_ssims.append(float(compute_ssim_masked(denoised_t, clean_t, inner_mask)))
    outer_ssims.append(float(compute_ssim_masked(denoised_t, clean_t, outer_mask)))
    roi_psnrs.append(float(compute_psnr_masked(denoised_t, clean_t, roi_mask)))
    roi_ssims.append(float(compute_ssim_masked(denoised_t, clean_t, roi_mask)))

results = {
    "model": "nafnet_w64",
    "checkpoint": str(ckpt_path),
    "pairs": str(pairs_file),
    "num_samples": len(pairs),
    "psnr_mean": float(np.mean(psnrs)),
    "ssim_mean": float(np.mean(ssims)),
    "region_psnr_inner": float(np.mean(inner_psnrs)),
    "region_psnr_outer": float(np.mean(outer_psnrs)),
    "region_ssim_inner": float(np.mean(inner_ssims)),
    "region_ssim_outer": float(np.mean(outer_ssims)),
    "roi_psnr_center": float(np.mean(roi_psnrs)),
    "roi_ssim_center": float(np.mean(roi_ssims)),
}

out_path = root / "outputs/duke_eval_baseline.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))

print("=" * 80)
print("Baseline NAFNet (w64) evaluation complete")
print(f"PSNR: {results['psnr_mean']:.2f} dB | SSIM: {results['ssim_mean']:.4f}")
print(f"Region PSNR: inner={results['region_psnr_inner']:.2f} | outer={results['region_psnr_outer']:.2f}")
print(f"ROI PSNR: center={results['roi_psnr_center']:.2f}")
print(f"✓ Wrote {out_path}")
print("=" * 80)
PY
else
  echo "⚠ Missing checkpoint: $FINAL_CKPT"
fi

echo ""
echo "================================================================================"
echo "Training complete!"
echo ""
echo "Expected results:"
echo "  - Per-pixel noise maps: Accurate spatial noise estimation"
echo "  - Region-specific PSNR improvements:"
echo "    * Inner retina: +0.5-0.8 dB (high speckle/shot noise)"
echo "    * Outer retina: +0.3-0.5 dB (lower noise, preserve detail)"
echo "  - Overall PSNR: 33.3-33.8 dB"
echo "  - SSIM: 0.907-0.915"
echo ""
echo "Novel contributions for TMI:"
echo "  1. Per-pixel noise type estimation with spatial weight maps"
echo "  2. Region-adaptive denoising guided by retinal structure"
echo "  3. Multi-head architecture for noise-specific processing"
echo "================================================================================"
