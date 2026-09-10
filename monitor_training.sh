#!/bin/bash
# Monitor training progress in real-time

echo "=== TRAINING MONITOR ==="
echo "Time: $(date)"
echo ""

echo "Running processes:"
ps aux | grep "train_hybrid" | grep -v grep | awk '{print "PID "$2" - Runtime: "$10" - CPU: "$3"%"}'
echo ""

echo "Recent checkpoints:"
ls -lth checkpoints/multitask_hybrid_nsnd_lambda* 2>/dev/null | head -3
echo ""

echo "Checking latest progress..."
python3 << 'PYEOF'
import torch
import os
from datetime import datetime

ckpts = [
    ('Balanced', 'checkpoints/multitask_hybrid_nsnd_lambda0p08to0p05_cosine_best.pth'),
    ('Two-Stage S1', 'checkpoints/multitask_hybrid_nsnd_lambda0p0_best.pth'),
]

for name, path in ckpts:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        mod_time = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"{name}:")
        print(f"  Last saved: {mod_time.strftime('%H:%M:%S')}")
        print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  PSNR: {ckpt.get('psnr', 0):.2f} dB")
        print(f"  Top-1: {ckpt.get('top1_accuracy', 0):.1f}%")
        print()
PYEOF

echo "=== END MONITOR ==="
