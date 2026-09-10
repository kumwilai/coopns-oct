# Diversity Loss Fix - Training with Head Specialization

## Problem

Training showed no improvement in adaptive gain after epoch 7:
- **Epoch 6**: Adaptive gain 0.10 dB
- **Epoch 7**: Adaptive gain 0.05 dB ⚠️ (DECREASED!)
- **Head similarity**: max 0.79-0.96 (way too high, target <0.6)

## Root Cause

Training was running with **diversity loss DISABLED**:
```bash
--head_quality_weight 0.0     # ❌ Should be >0
--head_diversity_weight 0.0   # ❌ Should be >0
--head_consistency_weight 0.0 # ❌ Should be >0
```

The `run_duke_region_focused.sh` script had these variables set to 0.0:
```bash
HEAD_QUALITY_WEIGHT=0.0
HEAD_DIVERSITY_WEIGHT=0.0
HEAD_DIVERSITY_WEIGHT_FT=0.0
HEAD_CONSISTENCY_WEIGHT=0.0
```

## Fix Applied

Updated `run_duke_region_focused.sh` (lines 12-15):
```bash
HEAD_QUALITY_WEIGHT=1.0          # Strong per-head supervision
HEAD_DIVERSITY_WEIGHT=0.5         # Force head specialization (aggressive)
HEAD_DIVERSITY_WEIGHT_FT=0.5      # Keep diversity in fine-tuning
HEAD_CONSISTENCY_WEIGHT=0.05      # Light regularization to prevent divergence
```

**Why 0.5 instead of 0.2?**
- Original recommendation was 0.2, but that's for similarity ~0.8
- Your checkpoint showed similarity **0.998** (complete collapse)
- Even after 7 epochs without diversity loss, max similarity was still 0.79-0.96
- Need aggressive diversity weight (0.5) to force specialization

## Training Command (Updated)

Restart Phase 2 with diversity loss enabled:

```bash
cd /home/kumwilai/OCT

python -u nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
  --pairs_train train_pairs_duke_analysis_maps.txt \
  --pairs_val val_pairs_duke_analysis_maps.txt \
  --weights_jsonl_train weights_duke_analysis_maps_train.jsonl \
  --weights_jsonl_val weights_duke_analysis_maps_val.jsonl \
  --max_samples 1000 \
  --val_samples 100 \
  --batch_size 4 \
  --epochs 50 \
  --early_stopping_patience 10 \
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
  --metrics_json outputs/duke_metrics_phase2_diversity.jsonl \
  --lambda_interp_start 0.008 \
  --lambda_interp_end 0.003 \
  --lambda_interp_schedule cosine \
  --lambda_warmup_epochs 0 \
  --routing_loss_weight 0.05 \
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
  --head_quality_weight 1.0 \
  --head_diversity_weight 0.5 \
  --head_consistency_weight 0.05 \
  --resume_ckpt checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth \
  --seed 42
```

## Expected Results

With diversity loss properly enabled:

### Epoch 1-10
- HeadSim max should drop rapidly (0.95 → 0.7)
- Diversity loss will be HIGH initially (>0.3)
- Adaptive gain should start increasing (0.05 → 0.2 dB)

### Epoch 10-30
- HeadSim max should drop below 0.6
- Individual heads start showing PSNR > overall PSNR
- Adaptive gain increases to 0.5-1.0 dB

### Epoch 30-50
- HeadSim stabilizes around 0.4-0.5
- Strong specialization: each head excels on its noise type
- Adaptive gain reaches 1.0+ dB

## Monitoring

Watch these metrics in training logs:

```
Batch 0100/0250 | HeadSim: 0.XXX (max 0.XXX)
                             ^       ^
                             |       |
                          Average  Maximum (should DROP below 0.6)

Head Effectiveness (vs Overall PSNR XX.XX):
  speckle : XX.XX dB → GOOD/BAD
  gaussian: XX.XX dB → GOOD/BAD
  shot    : XX.XX dB → GOOD/BAD
  banding : XX.XX dB → GOOD/BAD
  (Individual heads should show PSNR close to or exceeding overall)
```

## Success Criteria

✅ Training is working if:
- Max head similarity drops below 0.6 by epoch 20
- Adaptive gain increases to >0.5 dB by epoch 30
- Individual head PSNR competitive with overall PSNR

❌ If not working by epoch 15:
- Increase diversity weight to 0.7
- Reduce consistency weight to 0.01

## Why Previous Training Failed

The script had diversity loss weights set to 0.0, which means:
- No penalty for head similarity → heads learned identical functions
- No per-head quality supervision → heads didn't specialize
- Base NAFNet learned everything, adaptive heads added nothing

With diversity loss at 0.5, heads will be FORCED to produce different outputs, which will drive specialization.
