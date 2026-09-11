#!/usr/bin/env bash
# One command per defect described in README.md, so a reader can check each fix
# rather than take our word for it. Everything here runs on CPU. The seed check
# trains two tiny models and is the only step that takes more than a minute.
set -u
source "$(dirname "$0")/common.sh"

echo "== 1. Three loss weight flags are parsed and never used =="
echo "Every line of train_v8_cooperative.py that reads the three flags:"
grep -nE 'args\.(clinical_weight|tci_weight|cnr_weight)\b' train_v8_cooperative.py || echo "  (none)"
echo "The loss is built here, and the only tuned quantity that reaches it is the dead zone:"
grep -nE 'criterion = SimplifiedCooperativeLoss|psnr_dead_zone=args\.psnr_dead_zone' train_v8_cooperative.py
echo

echo "== 2. Seeding. Two runs with the same seed give the same model =="
if [ "${RUN_SEED_CHECK:-0}" = 1 ]; then
  for R in a b; do
    $PY train_v8_cooperative.py --backbone nafnet \
      --pretrained_backbone checkpointpaper/nafnet_backbone.pth \
      --resume checkpointpaper/nafnet_pku37_cooperative.pth --resume_epoch 1 \
      --train_jsonl $TRAIN --val_jsonl $VAL --max_train 8 --max_val 2 \
      --epochs 1 --batch_size 4 --patch_size 96 --hidden_channels 64 --device $DEVICE \
      --seed 0 --bg_rule otsu --psnr_dead_zone 0.3 --output_dir $OUT/seedcheck_$R > logs/seedcheck_$R.log 2>&1
  done
  $PY - <<'PY'
import torch
a = torch.load("outputs/revision/seedcheck_a/best_model_cooperative.pth", map_location="cpu", weights_only=False)
b = torch.load("outputs/revision/seedcheck_b/best_model_cooperative.pth", map_location="cpu", weights_only=False)
sa, sb = a.get("model_state_dict", a), b.get("model_state_dict", b)
worst = max((sa[k].float() - sb[k].float()).abs().max().item() for k in sa if torch.is_tensor(sa[k]) and sa[k].is_floating_point())
print("largest difference between the two runs over every tensor:", worst)
print("identical" if worst == 0.0 else "NOT identical, see README.md on GPU determinism")
PY
else
  echo "skipped. Set RUN_SEED_CHECK=1 to train two one epoch models on eight images and compare them."
fi
echo

echo "== 3. Backbone normalisation statistics no longer drift =="
$PY revision/verify_norm_freeze.py --backbone dncnn
echo

echo "== 4. SwinIR was trained against the reference it is tested on =="
echo "Part A reads what each SwinIR checkpoint recorded about its own cache and reference."
CFG=$(winner swinir)
if [ -n "$CFG" ] && [ -f "$OUT/sw_swinir_${CFG}_s${STUDY_SEED}/best_model_cooperative.pth" ]; then
  $PY revision/swinir_reference_check.py --skip_b --device $DEVICE \
     --checkpoint "$OUT/sw_swinir_${CFG}_s${STUDY_SEED}/best_model_cooperative.pth" \
     --backbone checkpointpaper/swinir_backbone.pth --val_jsonl $VAL
else
  echo "skipped, no selected SwinIR checkpoint present. The cache resolution is set at"
  grep -n 'cache_ds_factor' train_v8_cooperative.py | head -3
fi
echo

echo "== 5. The allocation map is no longer the constant one =="
echo "See ALLOCATION_FIX.md section 9. The legacy path is selected by LEGACY_SATURATING_ALLOCATION=1."
grep -n 'LEGACY_SATURATING_ALLOCATION' neuro_symbolic_corrector_v8_cooperative.py | head -2
echo

echo "== 6. Nothing is compiled, so training and scoring run the same function =="
echo "Remaining calls to torch.compile in the training script:"
grep -nE '^\s*[^#]*torch\.compile\(' train_v8_cooperative.py || echo "  (none)"
echo "The block where compilation used to be applied, with the measurements, begins at:"
grep -n 'Apply torch.compile optimization' train_v8_cooperative.py
