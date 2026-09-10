#!/bin/bash
# Monitor ablation training and run evaluation when checkpoints appear

ABLATIONS="no_negotiator no_edge no_uncertainty no_bg_smooth"
BACKBONE="outputs/nafnet_pku37_w40/best_model.pth"
TEST_JSONL="pku37_splits/test.jsonl"

echo "Monitoring ablation training progress..."
echo "Will evaluate each model as its checkpoint appears."
echo ""

completed=0
total=4

while [ $completed -lt $total ]; do
    for abl in $ABLATIONS; do
        ckpt="outputs/ablation_${abl}/best_model_cooperative.pth"
        result="outputs/ablation_results_${abl}.json"

        # Skip if already evaluated
        if [ -f "$result" ]; then
            continue
        fi

        # Check if checkpoint exists
        if [ -f "$ckpt" ]; then
            echo ""
            echo "$(date): Checkpoint found for ${abl}! Running evaluation..."
            python evaluate_ablation.py "$abl"
            completed=$((completed + 1))
            echo "$(date): ${abl} evaluation complete ($completed/$total done)"
        fi
    done

    # Print status
    echo -n "$(date '+%H:%M:%S') - Waiting... Checkpoints found: "
    for abl in $ABLATIONS; do
        if [ -f "outputs/ablation_${abl}/best_model_cooperative.pth" ]; then
            echo -n "${abl}:YES "
        else
            echo -n "${abl}:no "
        fi
    done
    echo ""

    sleep 60
done

echo ""
echo "All ablation evaluations complete!"
echo "Now evaluating full model..."
python evaluate_ablation.py full

echo ""
echo "Running combined evaluation..."
python evaluate_ablation.py
echo "Done!"
