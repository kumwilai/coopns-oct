#!/bin/bash
# Wait for all ablation training to finish, then evaluate

echo "$(date): Waiting for all 4 ablation training processes to finish..."

# PIDs of training processes
PIDS="53589 53653 53755 53786"

# Wait for all to finish
for pid in $PIDS; do
    if kill -0 $pid 2>/dev/null; then
        echo "$(date): Waiting for PID $pid..."
        while kill -0 $pid 2>/dev/null; do
            sleep 30
        done
        echo "$(date): PID $pid finished!"
    else
        echo "$(date): PID $pid already finished"
    fi
done

echo ""
echo "$(date): All training complete! Starting evaluation..."
echo ""

# Evaluate each ablation variant
for abl in no_negotiator no_edge no_uncertainty no_bg_smooth; do
    echo ""
    echo "$(date): Evaluating $abl..."
    python evaluate_ablation.py "$abl" 2>&1
done

# Also evaluate full model for consistent comparison
echo ""
echo "$(date): Evaluating full model..."
python evaluate_ablation.py full 2>&1

# Generate combined results
echo ""
echo "$(date): Generating combined results..."
python evaluate_ablation.py 2>&1

echo ""
echo "$(date): All evaluations complete!"
