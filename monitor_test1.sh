#!/bin/bash
# Monitor Test 1 training progress

echo "========================================================================" echo "MONITORING TEST 1: Lower Learning Rate Hypothesis"
echo "========================================================================"
echo ""

LOG_FILE="/home/kumwilai/OCT/test1_lower_lr.log"
CHECKPOINT_DIR="checkpoints/casa_n2v_test1_lower_lr"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Training may not have started yet."
    exit 1
fi

echo "📊 Current Training Progress:"
echo "------------------------------------------------------------------------"

# Extract meta-learning progress
echo ""
echo "Meta-Learning Phase:"
grep -E "\[Meta\] Epoch" "$LOG_FILE" | tail -5

# Extract fine-tuning progress
echo ""
echo "Fine-Tuning Phase:"
grep -E "Epoch \[" "$LOG_FILE" | tail -10

# Extract best PSNR
echo ""
echo "Best Results So Far:"
grep -E "New best model" "$LOG_FILE" | tail -1

# Check for early stopping
echo ""
if grep -q "Early stopping triggered" "$LOG_FILE"; then
    echo "⚠️  Early stopping was triggered"
    grep "Early stopping triggered" "$LOG_FILE" | tail -1
else
    echo "✓  Training still active or not yet stopped"
fi

# Check checkpoint
echo ""
echo "Checkpoint Status:"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✓  Checkpoint directory exists"
    echo "   Files:"
    ls -lh "$CHECKPOINT_DIR"/*.pth 2>/dev/null | awk '{print "   - " $9 " (" $5 ")"}'
else
    echo "❌ Checkpoint directory not found"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "📈 Training Curve Pattern:"
echo "------------------------------------------------------------------------"

# Extract validation PSNR values
echo ""
python3 << 'PYTHON_SCRIPT'
import re
import sys

log_file = "/home/kumwilai/OCT/test1_lower_lr.log"

try:
    with open(log_file, 'r') as f:
        content = f.read()

    # Find all validation PSNR values
    pattern = r"Epoch \[(\d+)/\d+\].*?Val Loss: ([\d.]+), PSNR: ([\d.]+) dB"
    matches = re.findall(pattern, content)

    if matches:
        print("Epoch | Val Loss | PSNR (dB) | Trend")
        print("------|----------|-----------|-------")

        prev_psnr = None
        for epoch, val_loss, psnr in matches:
            psnr_val = float(psnr)

            if prev_psnr is None:
                trend = "---"
            else:
                diff = psnr_val - prev_psnr
                if diff > 0.1:
                    trend = f"⬆️ +{diff:.2f}"
                elif diff < -0.1:
                    trend = f"⬇️ {diff:.2f}"
                else:
                    trend = f"➡️ {diff:+.2f}"

            print(f"  {epoch:3s} | {val_loss:8s} | {psnr:9s} | {trend}")
            prev_psnr = psnr_val
    else:
        print("No validation results found yet.")
        print("Training may still be in meta-learning phase.")

except FileNotFoundError:
    print(f"Error: Log file not found: {log_file}")
except Exception as e:
    print(f"Error parsing log: {e}")
PYTHON_SCRIPT

echo ""
echo "========================================================================"
echo "To monitor in real-time:"
echo "  tail -f $LOG_FILE"
echo "========================================================================"
