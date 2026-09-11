#!/usr/bin/env bash
# Confirms that every image named in the data lists and every weight file is present.
# Run this first. It needs no GPU and finishes in seconds.
set -u
source "$(dirname "$0")/common.sh"
status=0
for L in $TRAIN $VAL $TEST $SUB duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl duke_sota_datasets/Duke17_Eval/duke2013_synth_eval.jsonl; do
  n=$($PY -c "
import json,os,sys
miss=0; n=0
for line in open('$L'):
    if not line.strip(): continue
    e=json.loads(line); n+=1
    for k in ('clean_path','noisy_path'):
        if k in e and not os.path.exists(e[k]): miss+=1
print(n, miss)")
  set -- $n
  if [ "$2" != "0" ]; then echo "MISSING  $L: $2 of $(( $1 * 2 )) image paths not found"; status=1; else echo "ok       $L: $1 pairs"; fi
done
for BB in nafnet dncnn kbnet swinir; do
  for W in checkpointpaper/${BB}_backbone.pth checkpointpaper/${BB}_pku37_cooperative.pth; do
    if [ -f "$W" ]; then echo "ok       $W"; else echo "MISSING  $W"; status=1; fi
  done
done
[ $status = 0 ] && echo "all data and weights present" || echo "see checkpointpaper/README.md and the Data section of README.md"
exit $status
