#!/usr/bin/env bash
# Choose one setting per backbone from the validation scores, on validation only.
# Writes outputs/revision/winners.json, which every later script reads. The rule
# is documented at the top of revision/select_config.py and in Section VI D of
# the paper. The level that admitted each backbone is recorded in the file.
set -u
source "$(dirname "$0")/common.sh"
$PY revision/select_config.py --sweep_dir $OUT --out $OUT/winners.json \
    --backbones "$(echo $BACKBONES | tr ' ' ',')"
