#!/bin/bash
# Run sweep with memory monitoring. Kills sweep if RSS exceeds threshold.
THRESHOLD_MB=6000  # Kill if RSS > 6GB (leaves headroom for 7.8GB + 4GB swap)
LOG=/tmp/sweep_mem.log

echo "=== Memory Monitor: threshold=${THRESHOLD_MB}MB ===" | tee $LOG

python3 sweep_tta_hyperparams.py \
    --checkpoint outputs/v8_overcorrect_fix/best_model_cooperative.pth \
    --backbone outputs/nafnet_pku37_w40/best_model.pth \
    --backbone_width 40 \
    --device cpu \
    --datasets duke17,duke2013 \
    --output_json outputs/tta_sweep_results.json &
SWEEP_PID=$!

echo "Sweep PID: $SWEEP_PID" | tee -a $LOG

while kill -0 $SWEEP_PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $SWEEP_PID 2>/dev/null || echo 0)
    RSS_MB=$((RSS_KB / 1024))
    TIMESTAMP=$(date +%H:%M:%S)
    echo "$TIMESTAMP  RSS=${RSS_MB}MB" >> $LOG

    if [ $RSS_MB -gt $THRESHOLD_MB ]; then
        echo "$TIMESTAMP  *** RSS ${RSS_MB}MB > ${THRESHOLD_MB}MB threshold! KILLING ***" | tee -a $LOG
        kill $SWEEP_PID 2>/dev/null
        wait $SWEEP_PID 2>/dev/null
        echo "Sweep killed to prevent OOM. Last RSS: ${RSS_MB}MB" | tee -a $LOG
        echo "=== Memory trace ==="
        cat $LOG
        exit 1
    fi
    sleep 3
done

wait $SWEEP_PID
EXIT_CODE=$?
echo "Sweep exited with code $EXIT_CODE" | tee -a $LOG
echo "=== Memory trace ==="
cat $LOG
exit $EXIT_CODE
