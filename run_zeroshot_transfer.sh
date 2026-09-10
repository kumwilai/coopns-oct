#!/usr/bin/env bash
# Zero-shot cross-dataset transfer: PKU37-trained corrector applied to Duke
# with no per-fold adaptation. Answers reviewer 2 comment 1.
set -u
PY=/home/kumwilai/osmnx-env/bin/python
cd /home/kumwilai/OCT
export OMP_NUM_THREADS=2
for BB in nafnet dncnn kbnet swinir; do
  OUT=outputs/revision/zeroshot_${BB}.json
  if [ -f "$OUT" ]; then echo "skip $BB (exists)"; continue; fi
  echo "=== zero-shot $BB $(date -Iseconds) ==="
  $PY validate_crossdataset.py \
      --checkpoint checkpointpaper/${BB}_pku37_cooperative.pth \
      --backbone checkpointpaper/${BB}_backbone.pth \
      --backbone_name ${BB} --device cpu \
      --datasets duke17,duke2013 \
      --output_json "$OUT" 2>&1 | tail -40
  echo "=== done $BB $(date -Iseconds) ==="
done
echo ALL_ZEROSHOT_DONE
