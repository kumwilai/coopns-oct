#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
bash scripts/train_all.sh
bash scripts/eval_in_distribution.sh
bash scripts/eval_transfer.sh
bash scripts/eval_ablation.sh
bash scripts/run_diagnostics.sh
bash scripts/make_figures.sh
