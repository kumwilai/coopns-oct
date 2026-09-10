#!/usr/bin/env bash
# Exit as soon as the next milestone lands, so the session is notified.
cd /home/kumwilai/OCT
DONE_BEFORE=$(ls -d outputs/revision/retrain_*/best_model_cooperative.pth 2>/dev/null | wc -l)
EVAL_BEFORE=$(ls outputs/revision/*.json 2>/dev/null | wc -l)
while true; do
  D=$(ls -d outputs/revision/retrain_*/best_model_cooperative.pth 2>/dev/null | wc -l)
  E=$(ls outputs/revision/*.json 2>/dev/null | wc -l)
  if [ "$D" -gt "$DONE_BEFORE" ] || [ "$E" -gt "$EVAL_BEFORE" ]; then
    echo "MILESTONE at $(date +%H:%M)"
    echo "  trained correctors  $DONE_BEFORE -> $D"
    echo "  result files        $EVAL_BEFORE -> $E"
    ls -d outputs/revision/retrain_* 2>/dev/null | sed 's/^/  /'
    ls outputs/revision/*.json 2>/dev/null | sed 's/^/  /'
    exit 0
  fi
  sleep 60
done
