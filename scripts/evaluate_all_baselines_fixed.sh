#!/usr/bin/env bash
set -euo pipefail

PAIRS_FILE="${PAIRS_FILE:-/home/kumwilai/OCT/test_pairs_realistic_fixed.tsv}"
OUT_DIR="${OUT_DIR:-/home/kumwilai/OCT/results}"

mkdir -p "${OUT_DIR}"

SEEDS="${SEEDS:-0 1 2 3 4}"

# Fair configs (match ~7.42M params)
RESTORMER_DIM="${RESTORMER_DIM:-50}"
RESTORMER_BLOCKS="${RESTORMER_BLOCKS:-1}"
SWINIR_EMBED="${SWINIR_EMBED:-276}"
SWINIR_BLOCKS="${SWINIR_BLOCKS:-12}"
SWINIR_HEADS="${SWINIR_HEADS:-4}"
DNCNN_LAYERS="${DNCNN_LAYERS:-19}"
DNCNN_FEATURES="${DNCNN_FEATURES:-220}"
NAFNET_WIDTH="${NAFNET_WIDTH:-41}"
UNET_FEATURES="${UNET_FEATURES:-62}"

echo "Pairs: ${PAIRS_FILE}"
echo "Output: ${OUT_DIR}"
echo "Seeds: ${SEEDS}"

for seed in ${SEEDS}; do
  echo "== Seed ${seed} =="

  python nsnd_oct/scripts/evaluate_fixed_pairs.py \
    --model_type restormer \
    --checkpoint "checkpoints/restormer_fair_seed${seed}.pth" \
    --pairs_file "${PAIRS_FILE}" \
    --dim "${RESTORMER_DIM}" --num_blocks "${RESTORMER_BLOCKS}" \
    --out_json "${OUT_DIR}/restormer_seed${seed}.json"

  python nsnd_oct/scripts/evaluate_fixed_pairs.py \
    --model_type swinir \
    --checkpoint "checkpoints/swinir_fair_seed${seed}.pth" \
    --pairs_file "${PAIRS_FILE}" \
    --embed_dim "${SWINIR_EMBED}" --num_blocks "${SWINIR_BLOCKS}" --num_heads "${SWINIR_HEADS}" \
    --out_json "${OUT_DIR}/swinir_seed${seed}.json"

  python nsnd_oct/scripts/evaluate_fixed_pairs.py \
    --model_type dncnn \
    --checkpoint "checkpoints/dncnn_fair_seed${seed}.pth" \
    --pairs_file "${PAIRS_FILE}" \
    --num_layers "${DNCNN_LAYERS}" --features "${DNCNN_FEATURES}" \
    --out_json "${OUT_DIR}/dncnn_seed${seed}.json"

  python nsnd_oct/scripts/evaluate_fixed_pairs.py \
    --model_type nafnet \
    --checkpoint "checkpoints/nafnet_w41_seed${seed}.pth" \
    --pairs_file "${PAIRS_FILE}" \
    --width "${NAFNET_WIDTH}" \
    --out_json "${OUT_DIR}/nafnet_w41_seed${seed}.json"

  python nsnd_oct/scripts/evaluate_fixed_pairs.py \
    --model_type unet \
    --checkpoint "checkpoints/unet_f62_seed${seed}.pth" \
    --pairs_file "${PAIRS_FILE}" \
    --features "${UNET_FEATURES}" \
    --out_json "${OUT_DIR}/unet_f62_seed${seed}.json"
done

echo "Done. JSON outputs in ${OUT_DIR}."
