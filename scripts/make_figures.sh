#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
python3 code/revision/fig_architecture.py
python3 code/revision/fig_predicates.py
