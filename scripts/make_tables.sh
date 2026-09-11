#!/usr/bin/env bash
# Turn the result files into the LaTeX table bodies and the macros used by the
# manuscript. Writes revision/sections/generated/*.tex. A missing result prints
# as a dash, never as a guessed number.
set -u
source "$(dirname "$0")/common.sh"
$PY revision/make_tables.py
