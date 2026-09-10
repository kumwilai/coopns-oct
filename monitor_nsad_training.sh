#!/bin/bash
#
# Monitor NSAD Training Progress
#
# Usage: ./monitor_nsad_training.sh [log_file]
#

LOG_FILE=${1:-"checkpoints/sansd_quick_test/training.log"}

if [ ! -f "$LOG_FILE" ]; then
    # Try to find the most recent log
    LOG_FILE=$(ls -t checkpoints/*/training.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "No training log found."
        echo "Usage: ./monitor_nsad_training.sh <log_file>"
        exit 1
    fi
fi

echo "========================================"
echo "Monitoring: $LOG_FILE"
echo "========================================"
echo ""

# Function to show summary
show_summary() {
    echo ""
    echo "=== TRAINING SUMMARY ==="

    # Show best results
    echo ""
    echo "Best Results:"
    grep "Best model saved" "$LOG_FILE" | tail -5

    # Show latest validation metrics
    echo ""
    echo "Latest Validation Metrics:"
    grep -A 15 "Validation Metrics:" "$LOG_FILE" | tail -15

    # Show expert usage
    echo ""
    echo "Expert Usage:"
    grep "speckle\|banding\|gaussian\|shot" "$LOG_FILE" | tail -8
}

# Initial summary
show_summary

echo ""
echo "========================================"
echo "Watching for updates... (Ctrl+C to stop)"
echo "========================================"

# Watch for updates
tail -f "$LOG_FILE" 2>/dev/null | while read line; do
    echo "$line"

    # Show summary when epoch completes
    if echo "$line" | grep -q "Best model saved"; then
        show_summary
    fi
done
