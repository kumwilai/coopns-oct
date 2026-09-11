#!/usr/bin/env bash
# Everything, in order. See README.md for how long each step takes, and set
# DEVICE=cuda if a GPU is available.
set -eu
cd "$(dirname "$0")/.."
bash scripts/check_data.sh
bash scripts/train_all.sh
bash scripts/select_config.sh
bash scripts/eval_in_distribution.sh
bash scripts/eval_transfer.sh
bash scripts/eval_studies.sh
bash scripts/run_diagnostics.sh
bash scripts/make_tables.sh
bash scripts/make_figures.sh
# The adapted transfer protocol is separate because it trains one model per fold.
# bash scripts/eval_transfer_adapted.sh
